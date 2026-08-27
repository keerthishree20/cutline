#!/usr/bin/env python
"""Phase 1 — the honest floor.

Logistic regression on ten obvious numeric columns. This is deliberately not a
good model; it is the number every later model has to beat, established on a
split that does not lie.

It also runs the *wrong* split once, on purpose. The gap between the two is the
most credible slide in the deck: it shows the leak you avoided rather than
asserting you avoided it.

    python scripts/01_baseline.py            # real data if prepared, else synthetic
    python scripts/01_baseline.py --synthetic  # force the stand-in
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import config, data, metrics, split, synthetic  # noqa: E402

# Ten obvious numeric columns, plus two we derive. Nothing engineered yet —
# feature work is Phase 2, and doing it here would blur what the floor means.
BASE_COLUMNS = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "dist1", "C1", "C13", "D15", "has_identity",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in BASE_COLUMNS if c in df.columns]
    missing = set(BASE_COLUMNS) - set(available)
    if missing:
        print(f"  note: columns absent from this frame, skipped: {sorted(missing)}")

    X = df[available].astype("float32").copy()
    X["log_amt"] = np.log1p(df["TransactionAmt"].astype("float64")).astype("float32")
    X["hour"] = split.hour_of_day(df["TransactionDT"]).astype("float32")
    return X


def fit_and_score(fit_frame: pd.DataFrame, eval_frame: pd.DataFrame,
                  run_name: str, label: str) -> metrics.Result:
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",   # not SMOTE — resampling wrecks calibration
            max_iter=1000,
        )),
    ])

    model.fit(build_features(fit_frame), fit_frame["isFraud"])
    scores = model.predict_proba(build_features(eval_frame))[:, 1]

    result = metrics.evaluate(
        y_true=eval_frame["isFraud"],
        y_score=scores,
        amounts=eval_frame["TransactionAmt"],
        run=run_name,
        split=label,
        notes=f"{len(BASE_COLUMNS)} raw cols + log_amt + hour",
    )
    print("\n" + result.render())
    metrics.append_result(result)
    return result


def main() -> None:
    forced = "--synthetic" in sys.argv
    if forced or not config.TRAIN_PARQUET.exists():
        if not forced:
            print("!! no prepared parquet found — falling back to SYNTHETIC data.")
            print("!! these numbers are meaningless. Run scripts/download_data.py")
            print("!! then scripts/prepare_data.py for the real thing.\n")
        df = synthetic.make_frame(60_000)
        source = "synthetic"
    else:
        df = data.load()
        source = "ieee-cis"

    print(f"source: {source}  rows={len(df):,}  fraud={df['isFraud'].mean():.3%}\n")

    # ---- 1. the floor, on the split we will use for everything else ----
    parts = split.time_ordered_split(df)
    print(parts.describe())
    # The baseline has nothing to calibrate yet, so pooling calib into the fit
    # only avoids wasting rows. From Phase 2 on, calib is reserved and never
    # trained on — that reservation is what keeps the cost model honest.
    floor = fit_and_score(
        pd.concat([parts.train, parts.calib], ignore_index=True),
        parts.test,
        "baseline-logreg",
        f"time ({source})",
    )

    # ---- 2. what a random split would have bought us ----
    demo = split.leakage_demo(df)
    print("\n" + demo.describe())
    honest = fit_and_score(demo.honest_train, demo.evaluate_on,
                           "leak-demo-honest", f"past-only ({source})")
    leaky = fit_and_score(demo.leaky_train, demo.evaluate_on,
                          "leak-demo-leaky", f"saw-the-future ({source})")

    gap = leaky.pr_auc - honest.pr_auc
    print("\n" + "=" * 70)
    print("SAME evaluation rows, same fraud rate. Only the training window differs.")
    print(f"  trained on past only       PR-AUC {honest.pr_auc:.4f}   <- the real number")
    print(f"  trained on past + future   PR-AUC {leaky.pr_auc:.4f}   <- what a random split gives you")
    print(f"  leakage premium            {gap:+.4f}  ({gap / max(honest.pr_auc, 1e-9):+.1%})")
    print("=" * 70)
    print("\nBecause the evaluation set is held fixed, that gap is a difference")
    print("in skill, not in prevalence. Put both numbers on a slide.")
    print(f"\nFloor for Phase 2 to beat: PR-AUC {floor.pr_auc:.4f} "
          f"({floor.pr_auc_lift:.2f}x base rate)")


if __name__ == "__main__":
    main()
