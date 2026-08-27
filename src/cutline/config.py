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

# A false decline does not end with the support ticket. Some share of wrongly
# declined customers never come back, and that loss dwarfs the ticket. Modelled
# as an expected value — P(customer lost) x their remaining lifetime value —
# and kept as its OWN constant rather than folded into SUPPORT_COST, so the
# assumption stays visible and arguable instead of hiding inside another number.
CHURN_RATE = 0.25             # share of falsely declined customers who leave
CUSTOMER_LIFETIME_VALUE = 120.0   # USD, remaining value of one retained customer
CHURN_COST = CHURN_RATE * CUSTOMER_LIFETIME_VALUE   # 30.00 USD expected

# Minimising total cost alone drives the threshold to decline ~10% of good
# customers, which no real merchant tolerates. The unconstrained optimum is
# still reported — it is the honest answer to "what does pure cost say" — but
# the policy actually recommended respects this ceiling.
MAX_DECLINE_RATE = 0.01

# Split fractions — time-ordered, never random. See split.time_ordered_split.
TRAIN_FRAC = 0.70
CALIB_FRAC = 0.10
TEST_FRAC = 0.20

RANDOM_STATE = 42
