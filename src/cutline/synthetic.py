"""A small stand-in frame with IEEE-CIS's shape, so the pipeline is testable
before a gigabyte of real data finishes downloading.

It is NOT a modelling substitute — the signal here is hand-planted and the
numbers it produces mean nothing. Its only job is to prove the splitter, the
metrics and the baseline script run end to end.

Two properties are deliberate:
  * a genuine but noisy signal, so PR-AUC lands somewhere between 0.05 and 0.5
  * concept drift over time, so the random-vs-time split gap is visible
  * heavy, uneven card reuse, so the velocity features actually have history
    to look back at
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .split import SECONDS_PER_DAY

PRODUCTS = ["W", "C", "R", "H", "S"]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"]
DEVICES = ["desktop", "mobile"]


def make_frame(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Transactions arrive over ~180 days, unevenly.
    dt = np.sort(rng.integers(0, 180 * SECONDS_PER_DAY, size=n))
    day = dt / SECONDS_PER_DAY

    amount = np.round(np.exp(rng.normal(4.0, 1.1, size=n)), 2)

    # Cards repeat, heavily and unevenly — a handful of cards carry many
    # transactions while most carry one or two. Drawing card1 uniformly over a
    # wide range instead would mean no card ever recurs, and every velocity
    # feature in features.py would silently compute to zero.
    pool = rng.choice(np.arange(1000, 18000), size=1500, replace=False)
    weights = 1.0 / np.arange(1, len(pool) + 1) ** 0.9
    card1 = rng.choice(pool, size=n, p=weights / weights.sum())
    addr1 = rng.choice([np.nan, *range(100, 550)], size=n)
    dist1 = np.where(rng.random(n) < 0.6, np.nan, rng.exponential(50, size=n))
    hour = (dt % SECONDS_PER_DAY) // 3600
    has_identity = (rng.random(n) < 0.24).astype(np.int8)

    # Planted signal: large amounts, small hours, missing identity, and a
    # drifting card-range effect that only appears in the back half of the year.
    drift = np.clip((day - 90) / 90, 0, 1)
    logit = (
        -5.1
        + 0.55 * (np.log1p(amount) - 4.0)
        + 0.9 * ((hour < 6) | (hour > 22))
        + 0.7 * (1 - has_identity)
        + 2.1 * drift * (card1 > 14000)
        + rng.normal(0, 0.8, size=n)
    )
    p = 1 / (1 + np.exp(-logit))
    is_fraud = (rng.random(n) < p).astype(np.int8)

    df = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n, dtype=np.int32),
            "isFraud": is_fraud,
            "TransactionDT": dt.astype(np.int32),
            "TransactionAmt": amount.astype(np.float32),
            "ProductCD": rng.choice(PRODUCTS, size=n),
            "card1": card1.astype(np.int32),
            "card2": rng.choice([np.nan, *range(100, 600)], size=n),
            "card3": rng.choice([150.0, 185.0, np.nan], size=n, p=[0.85, 0.1, 0.05]),
            "card4": rng.choice(["visa", "mastercard", "amex", "discover"], size=n),
            "card5": rng.choice([np.nan, *range(100, 240)], size=n),
            "card6": rng.choice(["debit", "credit"], size=n),
            "addr1": addr1,
            "dist1": dist1,
            "P_emaildomain": rng.choice(EMAILS, size=n),
            "R_emaildomain": rng.choice([*EMAILS, None], size=n),
            "C1": rng.poisson(2.0, size=n).astype(np.float32),
            "C13": rng.poisson(8.0, size=n).astype(np.float32),
            "C14": rng.poisson(3.0, size=n).astype(np.float32),
            "D1": rng.exponential(60, size=n).astype(np.float32),
            "D15": np.where(rng.random(n) < 0.15, np.nan, rng.exponential(90, size=n)),
            "M4": rng.choice(["M0", "M1", "M2", None], size=n),
            "DeviceType": np.where(has_identity == 1, rng.choice(DEVICES, size=n), None),
            "DeviceInfo": np.where(has_identity == 1, rng.choice(["Windows", "iOS", "MacOS", "Android"], size=n), None),
            "id_01": np.where(has_identity == 1, rng.normal(-5, 10, size=n), np.nan),
            "id_02": np.where(has_identity == 1, rng.normal(200000, 90000, size=n), np.nan),
            "has_identity": has_identity,
        }
    )
    return df
