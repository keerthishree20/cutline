"""Feature engineering, in two stages that leak differently.

STAGE 1 — history features (`add_history_features`)
    Pure function of the time-sorted frame, no fitted state. Every value depends
    only on *earlier* rows, which is exactly what production sees, so computing
    these across the full frame before splitting is legitimate.

STAGE 2 — the fitted encoder (`FeatureBuilder`)
    Frequency maps and category levels ARE fitted state, and this is where the
    real leak hides. Fit the builder on the full frame and a card that appears
    only in the test period contributes its count to a training feature. Fit on
    the training slice alone; unseen keys at transform time get a sentinel of 0,
    never NaN — "this card has no history" is signal, and it should be one
    consistent value rather than something quietly imputed to the median later.

The category-level pinning in `transform` matters more than it looks. Calling
`.astype("category")` on train and test independently produces *different*
integer codings for the same string, silently, and LightGBM then reads garbage
at inference with no error raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .split import DT_COL, SECONDS_PER_DAY, day_index, hour_of_day

FREQ_COLUMNS = ["card1", "addr1", "P_emaildomain", "card2"]
CATEGORICAL = ["ProductCD", "card4", "card6", "M4", "DeviceType"]
UNSEEN = 0  # frequency sentinel; genuine counts are always >= 1

PASSTHROUGH = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "dist1", "C1", "C13", "C14", "D1", "D15", "has_identity",
]


def _prior_count_within(dt_sorted: np.ndarray, window: int) -> np.ndarray:
    """Number of EARLIER rows within `window` seconds. Excludes the row itself."""
    left = np.searchsorted(dt_sorted, dt_sorted - window, side="left")
    return np.arange(len(dt_sorted)) - left


def add_history_features(df: pd.DataFrame, key: str = "card1") -> pd.DataFrame:
    """Velocity and deviation features, strictly backward-looking.

    Run this ONCE on the whole frame, before splitting. Nothing here peeks
    forward: counts look back over a window, and the deviation baseline is an
    expanding statistic shifted by one row so a transaction never contributes
    to its own baseline.
    """
    out = df.sort_values(DT_COL, kind="mergesort").reset_index(drop=True).copy()

    out["hour"] = hour_of_day(out[DT_COL])
    out["dayofweek"] = (day_index(out[DT_COL]) % 7).astype("int8")

    amt = out["TransactionAmt"].astype("float64")
    out["log_amt"] = np.log1p(amt).astype("float32")
    # Fraud converted from another currency lands on odd fractions far more
    # often than a human typing a round price does.
    out["amt_cents"] = ((amt * 100) % 100).astype("float32")
    out["amt_is_round"] = (amt % 1 == 0).astype("int8")

    counts_1h = np.zeros(len(out), dtype="float32")
    counts_24h = np.zeros(len(out), dtype="float32")
    since_prev = np.full(len(out), np.nan, dtype="float32")
    amt_z = np.full(len(out), np.nan, dtype="float32")

    for _, idx in out.groupby(key, observed=True, sort=False).indices.items():
        idx = np.sort(idx)  # groupby.indices is unordered; the frame is time-sorted
        dt = out[DT_COL].to_numpy()[idx]

        counts_1h[idx] = _prior_count_within(dt, 3_600)
        counts_24h[idx] = _prior_count_within(dt, SECONDS_PER_DAY)

        gap = np.diff(dt, prepend=np.nan)
        since_prev[idx] = gap

        group_amt = pd.Series(amt.to_numpy()[idx])
        prior_mean = group_amt.expanding().mean().shift(1)
        prior_std = group_amt.expanding().std().shift(1)
        amt_z[idx] = ((group_amt - prior_mean) / (prior_std + 1.0)).to_numpy()

    out[f"{key}_count_1h"] = counts_1h
    out[f"{key}_count_24h"] = counts_24h
    out[f"{key}_sec_since_prev"] = since_prev
    out[f"{key}_amt_z"] = amt_z
    # No prior history is a state worth naming, not a missing value.
    out[f"{key}_is_first_seen"] = np.isnan(since_prev).astype("int8")

    if "P_emaildomain" in out.columns and "R_emaildomain" in out.columns:
        p = out["P_emaildomain"].astype("string")
        r = out["R_emaildomain"].astype("string")
        out["email_mismatch"] = ((p != r) & r.notna()).astype("int8")

    return out


HISTORY_COLUMNS = [
    "hour", "dayofweek", "log_amt", "amt_cents", "amt_is_round",
    "card1_count_1h", "card1_count_24h", "card1_sec_since_prev",
    "card1_amt_z", "card1_is_first_seen", "email_mismatch",
]


@dataclass
class FeatureBuilder:
    """Fitted on the training slice ONLY. See the module docstring."""

    freq_maps_: dict[str, dict] = field(default_factory=dict)
    levels_: dict[str, list] = field(default_factory=dict)
    columns_: list[str] = field(default_factory=list)
    categorical_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.freq_maps_ = {}
        for col in FREQ_COLUMNS:
            if col in df.columns:
                self.freq_maps_[col] = df[col].value_counts(dropna=True).to_dict()

        self.levels_ = {}
        for col in CATEGORICAL:
            if col in df.columns:
                self.levels_[col] = sorted(
                    pd.Series(df[col].astype("string")).dropna().unique().tolist()
                )

        self.columns_ = []
        self.categorical_ = []
        _ = self.transform(df.head(min(len(df), 50)), _recording=True)
        return self

    def transform(self, df: pd.DataFrame, _recording: bool = False) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)

        for col in PASSTHROUGH:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col in HISTORY_COLUMNS:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col, mapping in self.freq_maps_.items():
            if col in df.columns:
                # UNSEEN, not NaN: a key absent from training is a real state.
                X[f"{col}_freq"] = (
                    df[col].map(mapping).fillna(UNSEEN).astype("float32")
                )

        for col, levels in self.levels_.items():
            if col in df.columns:
                # Pin the levels from training so the integer coding is stable.
                X[col] = pd.Categorical(df[col].astype("string"), categories=levels)

        if _recording:
            self.columns_ = list(X.columns)
            self.categorical_ = [c for c in X.columns if str(X[c].dtype) == "category"]
            return X

        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"columns absent at transform time: {missing}")
        return X[self.columns_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def unseen_rate(self, df: pd.DataFrame) -> dict[str, float]:
        """Share of rows whose key was never seen in training, per frequency column.

        A rate of exactly 0.0 on a held-out set is the signature of an encoder
        fitted on too much data. Check it; do not assume it.
        """
        X = self.transform(df)
        return {
            col: float((X[f"{col}_freq"] == UNSEEN).mean())
            for col in self.freq_maps_
            if f"{col}_freq" in X.columns
        }
