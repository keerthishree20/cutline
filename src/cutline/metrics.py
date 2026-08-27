"""Metrics chosen so the numbers survive a judge who knows fraud.

A warning that bites: PR-AUC is NOT comparable across test sets with different
positive rates — its no-skill floor *is* the positive rate. A time-ordered test
set and a random one almost never share a fraud rate, so comparing their
PR-AUCs directly compares prevalence, not models. `pr_auc_lift` normalises for
that, and `split.leakage_demo` sidesteps it entirely by holding the evaluation
set fixed.

ROC-AUC is not the headline. At a ~3.5% positive rate it reads 0.95 while
precision sits at 8%, because the huge negative class flatters it. PR-AUC is
the headline; ROC-AUC is reported alongside only so the gap is visible.

And recall is reported twice: by count, and by *value*. Catching half of fraud
transactions is a modelling result; catching half of fraud dollars is the
number a merchant actually cares about, and the two are rarely the same.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config


@dataclass
class Result:
    run: str
    split: str
    n_test: int
    fraud_rate: float
    pr_auc: float
    pr_auc_lift: float
    roc_auc: float
    precision_at_50_recall: float
    recall_at_1pct_fpr: float
    value_recall_at_50_recall: float
    brier: float = 0.0
    ece: float = 0.0
    notes: str = ""
    stamp: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def render(self) -> str:
        return (
            f"{self.run}  [{self.split}]\n"
            f"  n={self.n_test:,}  fraud={self.fraud_rate:.3%}\n"
            f"  PR-AUC                     {self.pr_auc:.4f}   <- headline\n"
            f"  PR-AUC lift over base rate {self.pr_auc_lift:.2f}x  (1.0x = no skill)\n"
            f"  ROC-AUC                    {self.roc_auc:.4f}   (flattered by imbalance)\n"
            f"  precision @ 50% recall     {self.precision_at_50_recall:.4f}\n"
            f"  recall @ 1% FPR            {self.recall_at_1pct_fpr:.4f}\n"
            f"  VALUE recall @ 50% recall  {self.value_recall_at_50_recall:.4f}   <- what the merchant feels\n"
            f"  Brier                      {self.brier:.5f}\n"
            f"  ECE (10 bins)              {self.ece:.5f}   <- the cost model depends on this"
        )


def _threshold_at_recall(y_true, y_score, target_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns len(thresholds) == len(recall) - 1
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok[-1]])


def _precision_at_recall(y_true, y_score, target_recall: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = np.where(recall >= target_recall)[0]
    return float(precision[ok].max()) if len(ok) else 0.0


def _recall_at_fpr(y_true, y_score, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok].max()) if len(ok) else 0.0


def expected_calibration_error(y_true, y_score, bins: int = 10) -> float:
    """Mean gap between predicted probability and observed frequency.

    This is the number the cost model actually rests on. A model can rank
    perfectly and still be useless here: if it says 0.30 for a bucket that
    defaults at 0.05, every rupee in cost(tau) is wrong even though PR-AUC
    looks fine.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1], right=True), 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(y_score[mask].mean() - y_true[mask].mean())
    return float(total)


def value_recall(y_true, y_score, amounts, threshold: float) -> float:
    """Fraction of fraudulent *value* flagged at this threshold."""
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)
    flagged = np.asarray(y_score) >= threshold
    fraud_value = amounts[y_true == 1].sum()
    if fraud_value <= 0:
        return 0.0
    return float(amounts[(y_true == 1) & flagged].sum() / fraud_value)


def evaluate(y_true, y_score, amounts, run: str, split: str, notes: str = "") -> Result:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thr_50 = _threshold_at_recall(y_true, y_score, 0.50)

    return Result(
        run=run,
        split=split,
        n_test=int(len(y_true)),
        fraud_rate=float(y_true.mean()),
        pr_auc=float(average_precision_score(y_true, y_score)),
        pr_auc_lift=float(average_precision_score(y_true, y_score) / max(y_true.mean(), 1e-12)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        precision_at_50_recall=_precision_at_recall(y_true, y_score, 0.50),
        recall_at_1pct_fpr=_recall_at_fpr(y_true, y_score, 0.01),
        value_recall_at_50_recall=value_recall(y_true, y_score, amounts, thr_50),
        brier=float(np.mean((y_score - y_true) ** 2)),
        ece=expected_calibration_error(y_true, y_score),
        notes=notes,
    )


def append_result(result: Result) -> None:
    """Every experiment appends here. The file is the project's lab notebook."""
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(result)])
    header = not config.RESULTS_CSV.exists()
    row.to_csv(config.RESULTS_CSV, mode="a", header=header, index=False)
    print(f"\nappended to {config.RESULTS_CSV}")
