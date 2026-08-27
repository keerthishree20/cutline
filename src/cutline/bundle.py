"""Load a trained bundle and score with it — the ONE scoring path.

Phase 3's sweep, Phase 4's `/score` endpoint and Phase 5's dashboard all call
`score()` here. That is deliberate. The failure this prevents is a service that
loads the model and rebuilds features from a second copy of the logic, then
returns numbers that quietly disagree with the metrics table — no exception, no
warning, just a demo whose figures do not match the deck.

A note on serving a single transaction: `add_history_features` needs the card's
prior transactions to compute velocity. In production that means a history
store; for the hackathon demo the replay script keeps the frame in memory and
appends, which is why `score` takes a frame rather than a row.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config

DEFAULT_PATH = config.MODELS / "cutline.joblib"


def load(path: Path | None = None) -> dict:
    path = path or DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/02_model.py first")
    return joblib.load(path)


def score(bundle: dict, df: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
    """Calibrated fraud probability for every row of `df`.

    `df` must already carry history features (`features.add_history_features`).
    """
    X = bundle["feature_builder"].transform(df)
    raw = bundle["model"].predict_proba(X)[:, 1]
    return bundle["isotonic"].predict(raw) if calibrated else raw


def describe(bundle: dict) -> str:
    return (
        f"bundle v{bundle.get('version')} | source={bundle.get('source')} | "
        f"trained {bundle.get('trained_at')} | "
        f"{len(bundle.get('feature_columns', []))} features | "
        f"{bundle.get('best_iteration')} trees"
    )
