"""Paths and the cost assumptions the whole project is judged on.

Every constant below is an *assumption*, not a measurement. They belong on
screen during the demo so a judge can argue with them — a hidden number that
happens to be right is worth less than a stated one that can be challenged.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

TRAIN_PARQUET = INTERIM / "train.parquet"
RESULTS_CSV = REPORTS / "results.csv"

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
# IEEE-CIS is Vesta data and `TransactionAmt` is denominated in USD. Everything
# downstream stays in USD so the cost model never mixes units. Convert once, at
# presentation time, and show the rate you used.
CURRENCY = "USD"
USD_TO_INR = 83.0  # display only — never used inside the cost model

# ---------------------------------------------------------------------------
# Cost model constants (Phase 3)
# ---------------------------------------------------------------------------
DISPUTE_FEE = 25.0      # USD, fixed fee the merchant eats per chargeback
GROSS_MARGIN = 0.08     # fraction of a legitimate sale actually earned
SUPPORT_COST = 2.0      # USD, cost of one "why was I declined" contact
ANALYST_COST = 1.5      # USD, cost of one manual review

# Split fractions — time-ordered, never random. See split.time_ordered_split.
TRAIN_FRAC = 0.70
CALIB_FRAC = 0.10
TEST_FRAC = 0.20

RANDOM_STATE = 42
