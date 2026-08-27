#!/usr/bin/env python
"""Phase 2 — the real model, and the calibration the cost model depends on.

Four slices, three of them held back for a specific reason:

    [ ---- fit ---- ][ early-stop ][ calib ][ ---- test ---- ]
     <------ train 70% ---------->    10%        20%

  fit         gradient boosting trains here, and the encoder is fitted here
  early-stop  chooses the tree count; never seen during fitting
  calib       isotonic regression only — kept pristine so the probabilities
              the cost model consumes are honest
  test        touched once, at the end

Isotonic is fitted directly rather than through CalibratedClassifierCV so it is
obvious exactly what was fitted on what, and so the reliability curve can show
raw and calibrated scores on the same axes.

    python scripts/02_model.py [--synthetic]
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import config, data, features, metrics, split, synthetic  # noqa: E402

EARLY_STOP_FRAC = 0.15  # of the train slice, taken from its END (time-ordered)
BUNDLE = config.MODELS / "cutline.joblib"


def inner_time_split(train: pd.DataFrame, frac: float = EARLY_STOP_FRAC):
    """Carve the early-stopping set off the END of train, by time.

    Taken from the end rather than at random so that early stopping is judged
    on the most recent data — the same direction the real test set sits in.
    """
    dt_values = train[split.DT_COL].to_numpy()
    cut = np.quantile(dt_values, 1.0 - frac)
    return train[dt_values <= cut], train[dt_values > cut], int(cut)


def main() -> None:
    forced = "--synthetic" in sys.argv
    if forced or not config.TRAIN_PARQUET.exists():
        if not forced:
            print("!! no prepared parquet — using SYNTHETIC data. Numbers are meaningless.\n")
        df = synthetic.make_frame(80_000)
        source = "synthetic"
    else:
        df = data.load()
        source = "ieee-cis"

    print(f"source: {source}  rows={len(df):,}  fraud={df['isFraud'].mean():.3%}")

    print("\nbuilding history features (backward-looking only) ...")
    df = features.add_history_features(df)

    parts = split.time_ordered_split(df)
    fit_df, es_df, es_cut = inner_time_split(parts.train)
    print(parts.describe())
    print(f"  -> fit n={len(fit_df):,}   early-stop n={len(es_df):,}")

    # ---- encoder: fitted on the fit slice ONLY ----
    builder = features.FeatureBuilder().fit(fit_df)
    X_fit, y_fit = builder.transform(fit_df), fit_df["isFraud"]
    X_es, y_es = builder.transform(es_df), es_df["isFraud"]
    X_cal, y_cal = builder.transform(parts.calib), parts.calib["isFraud"]
    X_test, y_test = builder.transform(parts.test), parts.test["isFraud"]

    unseen = builder.unseen_rate(parts.test)
    print(f"\n  {len(builder.columns_)} features, {len(builder.categorical_)} categorical")
    print(f"  unseen-key rate on test: {({k: round(v, 4) for k, v in unseen.items()})}")
    if unseen and max(unseen.values()) == 0.0:
        print("  !! every key was seen — the encoder was fitted on too much data")

    # ---- model ----
    pos = int(y_fit.sum())
    neg = len(y_fit) - pos
    print(f"\ntraining LightGBM  (scale_pos_weight={neg / max(pos, 1):.1f}, not SMOTE)")

    model = lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        scale_pos_weight=neg / max(pos, 1),
        # Replace the default binary_logloss. With scale_pos_weight applied,
        # logloss degrades from the first iteration by construction — leaving it
        # in the metric list makes early stopping fire on it and return a
        # 1-tree model that scores below the Phase 1 floor.
        metric="average_precision",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_fit, y_fit,
        eval_X=X_es, eval_y=y_es,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    best = model.best_iteration_ or model.n_estimators
    print(f"  stopped at {best} trees")
    if best <= 5:
        print("  !! stopped almost immediately — early stopping is watching the")
        print("     wrong metric, or the features carry no signal. Do not trust this.")

    # ---- calibration: isotonic, fitted on calib and nothing else ----
    raw_cal = model.predict_proba(X_cal)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_cal, y_cal)

    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = isotonic.predict(raw_test)

    # ---- evaluate ----
    amounts = parts.test["TransactionAmt"]
    before = metrics.evaluate(y_test, raw_test, amounts, "lgbm-raw", f"time ({source})",
                              notes=f"{len(builder.columns_)} feats, {best} trees")
    after = metrics.evaluate(y_test, cal_test, amounts, "lgbm-calibrated", f"time ({source})",
                             notes="isotonic on held-out calib slice")
    print("\n" + before.render())
    print("\n" + after.render())
    metrics.append_result(before)
    metrics.append_result(after)

    print("\n" + "-" * 70)
    print("Isotonic is monotone, so it barely moves the ranking metrics — that is")
    print("expected, not a failure. What it moves is the calibration error, and")
    print("that is the number cost(tau) is built on.")
    print(f"  ECE   {before.ece:.5f}  ->  {after.ece:.5f}   ({(after.ece - before.ece) / max(before.ece, 1e-9):+.1%})")
    print(f"  Brier {before.brier:.5f}  ->  {after.brier:.5f}")

    _plot_reliability(y_test, raw_test, cal_test, source)

    # ---- one artifact: features, model and calibrator versioned together ----
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": 2,
            "source": source,
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "feature_builder": builder,
            "model": model,
            "isotonic": isotonic,
            "best_iteration": best,
            "feature_columns": builder.columns_,
            "categorical": builder.categorical_,
            "boundaries": {
                "early_stop_dt": es_cut,
                "train_end_dt": parts.train_end_dt,
                "calib_end_dt": parts.calib_end_dt,
            },
        },
        BUNDLE,
    )
    print(f"\nsaved {BUNDLE}")
    print("  encoder + model + calibrator in ONE file, so Phase 4 cannot load a")
    print("  model and rebuild features from a second copy of the logic.")

    _compare_to_floor(after, source)


def _plot_reliability(y_test, raw, cal, source: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#869596", lw=1, label="perfect")
    for scores, label, color in ((raw, "raw LightGBM", "#A8402F"), (cal, "isotonic", "#0E6B70")):
        n_bins = min(10, max(3, len(np.unique(scores)) // 50))
        frac, mean = calibration_curve(y_test, scores, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, "o-", color=color, lw=1.6, ms=4, label=label)

    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraud rate")
    ax.set_title(f"Reliability — {source}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS / "calibration.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


def _compare_to_floor(result: metrics.Result, source: str) -> None:
    if not config.RESULTS_CSV.exists():
        return
    hist = pd.read_csv(config.RESULTS_CSV)
    # Scope to the same data source AND the most recent run. results.csv is an
    # append-only notebook, so it accumulates rows from earlier versions of the
    # feature set and (once the real data lands) from a different dataset
    # entirely. Comparing against max() across all of that compares nothing.
    floor = hist[(hist["run"] == "baseline-logreg")
                 & (hist["split"].str.contains(source, regex=False))]
    if floor.empty:
        print(f"\n(no Phase 1 floor recorded for source={source} — "
              f"run scripts/01_baseline.py)")
        return

    latest = floor.sort_values("stamp").iloc[-1]
    best_floor = float(latest["pr_auc"])
    print(f"\n(floor from {latest['stamp']}, same source)")
    print("\n" + "=" * 70)
    print(f"  Phase 1 floor (logreg)   PR-AUC {best_floor:.4f}")
    print(f"  Phase 2 (calibrated)     PR-AUC {result.pr_auc:.4f}   "
          f"({(result.pr_auc - best_floor) / max(best_floor, 1e-9):+.1%})")
    print("=" * 70)
    if result.pr_auc >= best_floor:
        return
    if source == "synthetic":
        print("  Below the floor — on synthetic data this is weak evidence at best.")
        print("  The stand-in carries a mostly-linear signal across a handful of")
        print("  columns, which is the one regime where boosting has no edge.")
        print("  Judge Phase 2 on IEEE-CIS, not here.")
    else:
        print("  !! BELOW THE FLOOR on real data. That is a bug, not a modelling")
        print("     result. Check, in order: category coding identical across")
        print("     train/test, the frequency sentinel, and whether early stopping")
        print("     fired on the wrong metric.")


if __name__ == "__main__":
    main()
