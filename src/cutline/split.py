"""The time-ordered splitter. Every experiment in this project calls this.

WHY THIS FILE EXISTS
--------------------
Fraud tactics drift. A random train/test split trains on next month and tests
on last month, which inflates every score and means nothing. We split by time:
the model only ever sees the past.

    [ ---------- train 70% ---------- ][ calib 10% ][ ---- test 20% ---- ]
                                          ^ isotonic calibration fits here,
                                            never on test (that is leakage)

ABOUT `TransactionDT`
---------------------
It is NOT a timestamp. It is an integer offset in *seconds* from an arbitrary
reference point chosen by the dataset authors. So:

    hour of day  ->  (dt % 86400) // 3600      NOT pd.to_datetime(dt)
    day index    ->  dt // 86400

Sorting by it is still exactly right, which is all the split needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

SECONDS_PER_DAY = 86_400
DT_COL = "TransactionDT"


@dataclass
class Split:
    """Three disjoint, time-ordered frames plus the boundaries between them."""

    train: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    train_end_dt: int
    calib_end_dt: int
    kind: str = "time"

    def describe(self) -> str:
        rows = []
        for name, part in (("train", self.train), ("calib", self.calib), ("test", self.test)):
            rate = part["isFraud"].mean()
            rows.append(
                f"  {name:6} n={len(part):>7,}  fraud={rate:6.3%}  "
                f"days {part[DT_COL].min() // SECONDS_PER_DAY:>3}"
                f"–{part[DT_COL].max() // SECONDS_PER_DAY:>3}"
            )
        return f"{self.kind} split\n" + "\n".join(rows)


def hour_of_day(dt: pd.Series) -> pd.Series:
    """Hour 0–23. See the module docstring: TransactionDT is a seconds offset."""
    return ((dt % SECONDS_PER_DAY) // 3600).astype("int16")


def day_index(dt: pd.Series) -> pd.Series:
    """Whole days since the dataset's reference point."""
    return (dt // SECONDS_PER_DAY).astype("int32")


def time_ordered_split(
    df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    calib_frac: float = config.CALIB_FRAC,
) -> Split:
    """Sort by transaction time, then cut at the given fractions.

    The cut is placed on a *time* boundary, not a row boundary, so no single
    instant is straddled by two splits.
    """
    if DT_COL not in df.columns:
        raise KeyError(f"{DT_COL!r} missing — cannot split by time")

    ordered = df.sort_values(DT_COL, kind="mergesort").reset_index(drop=True)
    dt = ordered[DT_COL].to_numpy()

    train_end = np.quantile(dt, train_frac)
    calib_end = np.quantile(dt, train_frac + calib_frac)

    train = ordered[dt <= train_end]
    calib = ordered[(dt > train_end) & (dt <= calib_end)]
    test = ordered[dt > calib_end]

    return Split(
        train=train.reset_index(drop=True),
        calib=calib.reset_index(drop=True),
        test=test.reset_index(drop=True),
        train_end_dt=int(train_end),
        calib_end_dt=int(calib_end),
        kind="time",
    )


@dataclass
class LeakageDemo:
    """Two training sets scored on ONE evaluation set, so the gap is real.

    The naive demo — time split vs random split, compare PR-AUC — is confounded:
    the two test sets have different fraud rates, and PR-AUC's floor is the
    fraud rate. You end up measuring prevalence, not leakage.

    This construction fixes the evaluation set and varies only what the model
    was allowed to see:

        [ ------- past 80% ------- ][ future 20% ]
                                     F1     F2
        honest : trains on past            -> scored on F2
        leaky  : trains on past + F1       -> scored on F2

    Same rows evaluated, same prevalence, same everything except one model got
    contemporaneous data. Whatever separates them is the value of seeing the
    future — which is exactly what a random split hands you for free.
    """

    honest_train: pd.DataFrame
    leaky_train: pd.DataFrame
    evaluate_on: pd.DataFrame

    def describe(self) -> str:
        return (
            "leakage demo (one fixed evaluation set)\n"
            f"  honest train  n={len(self.honest_train):>7,}  (past only)\n"
            f"  leaky train   n={len(self.leaky_train):>7,}  (past + half the future)\n"
            f"  evaluated on  n={len(self.evaluate_on):>7,}  "
            f"fraud={self.evaluate_on['isFraud'].mean():6.3%}"
        )


def leakage_demo(
    df: pd.DataFrame,
    future_frac: float = config.TEST_FRAC,
    random_state: int = config.RANDOM_STATE,
) -> LeakageDemo:
    """Build the honest/leaky pair described in `LeakageDemo`."""
    ordered = df.sort_values(DT_COL, kind="mergesort").reset_index(drop=True)
    dt = ordered[DT_COL].to_numpy()
    cut = np.quantile(dt, 1.0 - future_frac)

    past = ordered[dt <= cut]
    future = ordered[dt > cut]

    f1 = future.sample(frac=0.5, random_state=random_state)
    f2 = future.drop(f1.index)

    return LeakageDemo(
        honest_train=past.reset_index(drop=True),
        leaky_train=pd.concat([past, f1], ignore_index=True),
        evaluate_on=f2.reset_index(drop=True),
    )


def random_split(
    df: pd.DataFrame,
    train_frac: float = config.TRAIN_FRAC,
    calib_frac: float = config.CALIB_FRAC,
    random_state: int = config.RANDOM_STATE,
) -> Split:
    """The wrong way to split — kept deliberately.

    Run it once, report the gap against `time_ordered_split`, and put both
    numbers on a slide. Demonstrating the leak you avoided is far more
    convincing than asserting you avoided it.
    """
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(shuffled)
    a = int(n * train_frac)
    b = int(n * (train_frac + calib_frac))

    return Split(
        train=shuffled.iloc[:a].reset_index(drop=True),
        calib=shuffled.iloc[a:b].reset_index(drop=True),
        test=shuffled.iloc[b:].reset_index(drop=True),
        train_end_dt=-1,
        calib_end_dt=-1,
        kind="random (leaky — comparison only)",
    )
