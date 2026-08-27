"""Scoring one transaction at a time, without a second copy of the feature logic.

THE HARD PART, stated plainly: velocity features are backward-looking over a
card's history, and a single incoming transaction does not carry its own
history. Production would read a feature store. For the demo, `HistoryStore`
keeps recent transactions in memory.

Why the restriction to one card is safe: `add_history_features` computes
row-local values (hour, amount shape) and per-card values grouped by `card1`.
A row's features therefore depend only on itself and on earlier rows of the
SAME card. Running the function over just that card's rows returns bit-identical
output to running it over the whole frame — which is what lets serving call the
exact training code instead of reimplementing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import bundle as bundle_mod
from . import config, costs, features


class HistoryStore:
    """Recent transactions, kept so velocity features can be computed live."""

    def __init__(self, seed: pd.DataFrame | None = None, max_rows: int = 250_000):
        self.max_rows = max_rows
        self._rows = seed.copy() if seed is not None else None

    def __len__(self) -> int:
        return 0 if self._rows is None else len(self._rows)

    #: Temporary marker column used to find the incoming row after sorting.
    _UID = "_cutline_uid"

    def add(self, txn: dict) -> pd.DataFrame:
        """Append one transaction and return it WITH history features."""
        row = pd.DataFrame([txn])

        if self._rows is None:
            self._rows = row
        else:
            row = row.reindex(columns=self._rows.columns.union(row.columns))
            self._rows = pd.concat([self._rows, row], ignore_index=True)
            if len(self._rows) > self.max_rows:
                self._rows = self._rows.iloc[-self.max_rows:].reset_index(drop=True)

        key = features.HISTORY_KEY
        card = txn.get(key)
        subset = self._rows[self._rows[key] == card] if card is not None else self._rows
        subset = subset.reset_index(drop=True).copy()

        # Find the incoming row by an explicit marker, never by position.
        # `add_history_features` sorts by TransactionDT, so a transaction that
        # is not the latest for its card does NOT come back last — taking the
        # final row would silently return a DIFFERENT transaction's features
        # and score it as this one. Out-of-order arrivals are normal: replayed
        # feeds, retries, and clock skew all produce them.
        subset[self._UID] = 0
        subset.loc[subset.index[-1], self._UID] = 1

        featurised = features.add_history_features(subset)
        hit = featurised[featurised[self._UID] == 1]
        if len(hit) != 1:
            raise RuntimeError(
                f"history reconstruction lost the incoming transaction "
                f"({len(hit)} matches). This must never happen."
            )
        return hit.drop(columns=[self._UID]).reset_index(drop=True)


class Scorer:
    """Bundle + calibrator + policy, loaded once."""

    def __init__(self, bundle_path: Path | None = None,
                 curve_path: Path | None = None):
        self.bundle = bundle_mod.load(bundle_path)
        self.constants = costs.CostConstants()
        self._explainer = None

        curve_path = curve_path or (config.REPORTS / "cost_curve.json")
        if curve_path.exists():
            payload = json.loads(curve_path.read_text())
            self.tau_block = float(payload["tau_star"])
            self.cost_do_nothing = float(payload.get("cost_do_nothing", float("nan")))
            self.cost_at_tau = float(payload.get("cost_at_tau_star", float("nan")))
            self.curve_source = str(curve_path)
        else:
            # No curve means no cost-optimal threshold, and defaulting to 0.5
            # silently would undo the entire argument of the project.
            self.tau_block = 0.5
            self.cost_do_nothing = float("nan")
            self.cost_at_tau = float("nan")
            self.curve_source = "MISSING — falling back to 0.5, run 03_cost_model.py"

        self.tau_review = self.tau_block / 3.0

    @property
    def explainer(self):
        if self._explainer is None:
            from .explain import Explainer
            self._explainer = Explainer(self.bundle)
        return self._explainer

    def features_for(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.bundle["feature_builder"].transform(df)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return bundle_mod.score(self.bundle, df)

    def decide(self, p: float) -> str:
        if p >= self.tau_block:
            return "block"
        if p >= self.tau_review:
            return "review"
        return "allow"

    def policy(self) -> dict:
        return {
            "tau_block": self.tau_block,
            "tau_review": self.tau_review,
            "currency": config.CURRENCY,
            "constants": {
                "dispute_fee": self.constants.dispute_fee,
                "gross_margin": self.constants.gross_margin,
                "support_cost": self.constants.support_cost,
                "analyst_cost": self.constants.analyst_cost,
            },
            "cost_do_nothing": self.cost_do_nothing,
            "cost_at_tau_block": self.cost_at_tau,
            "curve": self.curve_source,
            "trained_on": self.bundle.get("source"),
            "trained_at": self.bundle.get("trained_at"),
        }
