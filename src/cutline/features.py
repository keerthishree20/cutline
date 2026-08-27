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

# The column the per-card history features group by. Serving filters on this
# same constant — hardcoding "card1" in two places is how the training/serving
# parity guarantee quietly becomes false.
HISTORY_KEY = "card1"

FREQ_COLUMNS = ["card1", "addr1", "P_emaildomain", "card2"]
CATEGORICAL = ["ProductCD", "card4", "card6", "M4", "DeviceType"]
UNSEEN = 0  # frequency sentinel; genuine counts are always >= 1

# A z-score needs a believable baseline. With one or two prior transactions the
# expanding std is noise and the +1.0 denominator floor dominates, so the score
# ends up measuring how much history a card has rather than how unusual this
# amount is — mean |z| falls 1.63 -> 0.56 as prior count rises, with no change
# in the underlying amounts. Below this many priors, emit nothing: the "no
# history" signal is already carried by the count and first-seen features.
MIN_PRIOR_FOR_Z = 3

# --- the V and id_ blocks -------------------------------------------------
# 404 of the dataset's 435 columns went unused for the first four phases. The
# V block is Vesta's own engineered features and it carries most of the signal
# the winning Kaggle solutions found.
#
# The V columns fall into 14 groups that go missing together — Vesta telling us
# they came from 14 upstream sources. Within a group the columns are only
# modestly correlated (median |r| 0.14-0.26), so aggressive reduction would
# discard real signal; only genuine near-duplicates are dropped. The
# group-present flags are kept as features in their own right, for the same
# reason has_identity is: absence is informative here.
V_CORR_THRESHOLD = 0.95
V_CORR_SAMPLE = 40_000

# id_30/31/33 are OS, browser and screen-resolution strings with long tails. A
# browser version seen once in training is noise occupying a category slot, and
# LightGBM will happily split on it.
MAX_CATEGORY_LEVELS = 30

# The full counting and timedelta blocks, not the three of each that Phase 2
# sampled — C1-C14 and D1-D15 are among the strongest features on this dataset.
PASSTHROUGH = (
    ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
     "dist1", "dist2", "has_identity"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"id_{i:02d}" for i in range(1, 12)]      # the numeric identity columns
)


def required_columns() -> list[str]:
    """Raw columns worth loading. Everything else stays off the heap.

    The parquet is 435 columns and the V block alone is 800MB in memory, so
    reading the whole frame and then selecting is the difference between a
    training run that fits and one that swaps.
    """
    base = ["TransactionID", "isFraud", "TransactionDT", "P_emaildomain",
            "R_emaildomain", "DeviceInfo"]
    return sorted(set(
        base + PASSTHROUGH + FREQ_COLUMNS + CATEGORICAL
        + [f"V{i}" for i in range(1, 340)]
        + [f"id_{i:02d}" for i in range(1, 39)]
    ))


def _prior_count_within(dt_sorted: np.ndarray, window: int) -> np.ndarray:
    """Number of EARLIER rows within `window` seconds. Excludes the row itself."""
    left = np.searchsorted(dt_sorted, dt_sorted - window, side="left")
    return np.arange(len(dt_sorted)) - left


def add_history_features(df: pd.DataFrame, key: str = HISTORY_KEY) -> pd.DataFrame:
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
        z = (group_amt - prior_mean) / (prior_std + 1.0)
        z[np.arange(len(idx)) < MIN_PRIOR_FOR_Z] = np.nan
        amt_z[idx] = z.to_numpy()

    out[f"{key}_count_1h"] = counts_1h
    out[f"{key}_count_24h"] = counts_24h
    out[f"{key}_sec_since_prev"] = since_prev
    # Named for what it means, because the SHAP panel will quote it back to a
    # user as "amount is N x this card's usual".
    out[f"{key}_amt_z"] = amt_z
    # No prior history is a state worth naming, not a missing value.
    out[f"{key}_is_first_seen"] = np.isnan(since_prev).astype("int8")

    if "P_emaildomain" in out.columns and "R_emaildomain" in out.columns:
        p = out["P_emaildomain"].astype("string")
        r = out["R_emaildomain"].astype("string")
        # Both sides must be present for "mismatch" to mean anything. Comparing
        # a null yields NA under pandas' nullable string dtype, and NA & True is
        # still NA, so the int cast below fails outright. Absence is recorded
        # separately rather than being folded into the mismatch flag.
        known = p.notna() & r.notna()
        out["email_mismatch"] = ((p != r) & known).fillna(False).astype("int8")
        out["email_known"] = known.fillna(False).astype("int8")

    return out


HISTORY_COLUMNS = [
    "hour", "dayofweek", "log_amt", "amt_cents", "amt_is_round",
    "card1_count_1h", "card1_count_24h", "card1_sec_since_prev",
    "card1_amt_z", "card1_is_first_seen", "email_mismatch", "email_known",
]


@dataclass
class FeatureBuilder:
    """Fitted on the training slice ONLY. See the module docstring."""

    freq_maps_: dict[str, dict] = field(default_factory=dict)
    levels_: dict[str, list] = field(default_factory=dict)
    columns_: list[str] = field(default_factory=list)
    categorical_: list[str] = field(default_factory=list)
    source_columns_: list[str] = field(default_factory=list)
    v_keep_: list[str] = field(default_factory=list)
    v_groups_: dict[str, list[str]] = field(default_factory=dict)
    extra_categorical_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        # The V selection is FITTED, not read from a file: correlations are
        # computed on the training slice alone, so a column pair that happens
        # to be redundant only in the test period cannot influence what the
        # model is given.
        self._fit_v_block(df)
        self._fit_extra_categoricals(df)

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

        # Remember which raw columns were present at fit time. A live request
        # carries fewer fields than a training row, and in this dataset an
        # absent column is missing DATA, not a schema error — has_identity
        # exists precisely because most rows have no device information.
        wanted = (set(PASSTHROUGH) | set(HISTORY_COLUMNS) | set(FREQ_COLUMNS)
                  | set(CATEGORICAL) | set(self.v_keep_) | set(self.extra_categorical_)
                  | {c for g in self.v_groups_.values() for c in g})
        self.source_columns_ = [c for c in df.columns if c in wanted]

        self.columns_ = []
        self.categorical_ = []
        _ = self.transform(df.head(min(len(df), 50)), _recording=True)
        return self

    def transform(self, df: pd.DataFrame, _recording: bool = False) -> pd.DataFrame:
        # Reinstate any source column the caller did not supply, as NaN, so
        # training and serving take the same path through the code below.
        absent = [c for c in self.source_columns_ if c not in df.columns]
        if absent:
            # One concat, not 380 assigns. A live request carries a handful of
            # fields against a 400-column fit-time schema, and adding them one
            # at a time fragments the frame and floods the log with pandas
            # performance warnings on every single call.
            filler = pd.DataFrame(
                np.nan, index=df.index, columns=absent, dtype="float32"
            )
            df = pd.concat([df, filler], axis=1)

        # Accumulate into a dict and build the frame ONCE. Assigning 380
        # columns one at a time into a DataFrame reallocates the block manager
        # on nearly every write; at serving time that alone cost ~200ms per
        # request, several times more than the model itself.
        cols: dict[str, pd.Series] = {}

        def numeric(name: str, source: str) -> None:
            if source in df.columns:
                cols[name] = pd.to_numeric(df[source], errors="coerce").astype("float32")

        for col in PASSTHROUGH:
            numeric(col, col)
        for col in HISTORY_COLUMNS:
            numeric(col, col)
        for col in self.v_keep_:
            numeric(col, col)

        for gid, members in self.v_groups_.items():
            probe = next((c for c in members if c in df.columns), None)
            cols[f"Vgrp{gid}_present"] = (
                df[probe].notna().astype("float32") if probe is not None
                else pd.Series(np.float32(0.0), index=df.index)
            )

        for col, mapping in self.freq_maps_.items():
            if col in df.columns:
                # UNSEEN, not NaN: a key absent from training is a real state.
                cols[f"{col}_freq"] = df[col].map(mapping).fillna(UNSEEN).astype("float32")

        categorical: list[str] = []
        for col, levels in {**self.levels_, **self._extra_levels}.items():
            if col in df.columns:
                # Pin the levels from training so the integer coding is stable.
                cols[col] = pd.Series(
                    pd.Categorical(df[col].astype("string"), categories=levels),
                    index=df.index,
                )
                categorical.append(col)

        X = pd.DataFrame(cols, index=df.index)

        if _recording:
            self.columns_ = list(X.columns)
            self.categorical_ = categorical
            return X

        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            # Reaching here means the feature set itself changed, not that a
            # request was sparse — that case is handled above.
            raise ValueError(
                f"feature columns could not be built: {missing}. The builder and "
                f"the code that produces features are out of sync; retrain."
            )
        return X[self.columns_]

    def missing_source_columns(self, df: pd.DataFrame) -> list[str]:
        """Fit-time columns this frame does not carry at all."""
        return [c for c in self.source_columns_ if c not in df.columns]

    def missing_source_values(self, df: pd.DataFrame) -> list[str]:
        """Fit-time columns that are absent OR null for every row given.

        Column presence is the wrong measure of a sparse request. A serving
        store accumulates columns from earlier transactions, so a three-field
        request arrives carrying every column with almost all of them NaN —
        `missing_source_columns` sees nothing wrong. This looks at the values.
        """
        out = []
        for col in self.source_columns_:
            if col not in df.columns or bool(df[col].isna().all()):
                out.append(col)
        return out

    @property
    def _extra_levels(self) -> dict:
        return getattr(self, "_extra_levels_store", {})

    def _fit_extra_categoricals(self, df: pd.DataFrame) -> None:
        """Identity string columns, capped to their commonest levels."""
        cands = [c for c in df.columns
                 if (c.startswith("id_") or c == "DeviceInfo")
                 and (df[c].dtype == object or str(df[c].dtype) in ("category", "str"))]
        store: dict[str, list] = {}
        for col in cands:
            counts = df[col].astype("string").value_counts(dropna=True)
            store[col] = sorted(counts.head(MAX_CATEGORY_LEVELS).index.tolist())
        self.extra_categorical_ = cands
        self._extra_levels_store = store

    def _fit_v_block(self, df: pd.DataFrame) -> None:
        """Group the V columns by null-pattern, then drop near-duplicates."""
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if not v_cols:
            self.v_keep_, self.v_groups_ = [], {}
            return

        nulls = df[v_cols].isna().mean().round(4)
        buckets: dict[float, list[str]] = {}
        for col, share in nulls.items():
            buckets.setdefault(float(share), []).append(col)

        groups, keep = {}, []
        for gid, (_, members) in enumerate(
            sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        ):
            groups[str(gid)] = sorted(members)
            sub = df[members].dropna()
            if len(sub) > V_CORR_SAMPLE:
                sub = sub.sample(V_CORR_SAMPLE, random_state=0)
            if len(sub) < 50 or len(members) == 1:
                keep.extend(members)
                continue
            corr = sub.corr().abs()
            # Prefer the higher-cardinality member of a near-duplicate pair.
            order = df[members].nunique().sort_values(ascending=False).index.tolist()
            kept: list[str] = []
            for col in order:
                if all(not (corr.loc[col, k] > V_CORR_THRESHOLD) for k in kept):
                    kept.append(col)
            keep.extend(kept)

        self.v_keep_ = sorted(keep)
        self.v_groups_ = groups

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
