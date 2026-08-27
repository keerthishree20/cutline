# Cutline

A payment risk scorer whose decline threshold comes from **cost**, not accuracy.

Every fraud model outputs a probability. Turning that probability into a decision
is an economics problem, not a modelling one: a missed chargeback costs the
transaction value plus a dispute fee, while a wrongly declined sale costs margin
plus a support ticket. Those are not equal, so `0.5` is almost never the right
cut. This project finds the cut that minimises actual cost, and shows the curve
it came from.


## Status

| Phase | What | State |
|---|---|---|
| 1 | Data, time-ordered split, baseline floor | **done** |
| 2 | Feature engineering, LightGBM, calibration | next |
| 3 | Cost model and threshold sweep | — |
| 4 | FastAPI `/score` + `/explain` | — |
| 5 | Next.js review queue + threshold slider | — |

## Setup

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

Training data is **IEEE-CIS Fraud Detection** — ~590k real payment transactions
at a ~3.5% fraud rate. It needs Kaggle credentials at `~/.kaggle/kaggle.json`
**and** a one-time acceptance of the competition rules in a browser, which the
API cannot do for you: <https://www.kaggle.com/c/ieee-fraud-detection/rules>

```bash
.venv/bin/python scripts/download_data.py   # train files only, ~700 MB
.venv/bin/python scripts/prepare_data.py    # -> data/interim/train.parquet
rm data/raw/*.csv                           # reclaim the space
```

Only the two `train_*` files are fetched. The competition's test labels were
never released, so `test_transaction.csv` is 600 MB you cannot evaluate against;
our test set comes from time-splitting within train instead.

### Two things about this dataset that cost people a day each

- **`TransactionDT` is not a timestamp.** It is a seconds offset from an
  arbitrary reference. Hour of day is `(dt % 86400) // 3600`, never a datetime
  parse. See `src/cutline/split.py`.
- **The identity join is mostly misses.** `train_identity` covers a minority of
  transactions, so `DeviceType`, `DeviceInfo` and every `id_*` column are null
  for most rows. `has_identity` records that absence, which is itself
  predictive. Any explanation shown to a user must fall back to
  transaction-side features.

## Running without the data

Every script falls back to a synthetic stand-in with the same column names,
dtypes and drift, so the pipeline is testable while the download runs:

```bash
.venv/bin/python scripts/01_baseline.py --synthetic
```

Those numbers are meaningless by construction. The stand-in exists to prove the
code runs, not to say anything about fraud.

`scripts/smoke_test.py` goes further and round-trips synthetic data through the
*real* ingestion path — two CSVs on disk, `build_parquet`, read back, split —
in a temp directory. Run it before trusting a change to `data.py`, so that code
is not executing for the first time on a 700 MB file:

```bash
.venv/bin/python scripts/smoke_test.py
```

## Phase 1 results

`scripts/01_baseline.py` does two things.

**The floor.** Logistic regression on ten obvious numeric columns, on a
time-ordered split. Deliberately not a good model — it is the number Phase 2
has to beat.

**The leakage demo.** The naive version of this — compare a time split against a
random split — is confounded, because the two test sets have different fraud
rates and PR-AUC's no-skill floor *is* the fraud rate. You end up measuring
prevalence, not leakage. So the evaluation set is held fixed and only the
training window varies:

```
[ ---------- past 80% ---------- ][ future 20% ]
                                    F1     F2
honest : trains on past                  -> scored on F2
leaky  : trains on past + F1             -> scored on F2
```

Same rows, same prevalence, same everything except one model saw
contemporaneous data. The gap between them is what a random split silently
hands you.

Results append to `reports/results.csv` — that file is the project's lab
notebook, and every experiment from here on adds a row.

## Metrics, and why these ones

- **PR-AUC** is the headline. At a 3.5% positive rate, ROC-AUC reads ~0.95 while
  precision sits near 8%, because the huge negative class flatters it. ROC-AUC
  is reported alongside only so that gap stays visible.
- **PR-AUC lift** over the base rate, because raw PR-AUC is not comparable
  across sets with different prevalence.
- **Recall by value, not just by count.** Catching half of fraudulent
  transactions is a modelling result; catching half of fraudulent dollars is
  what a merchant feels.

## Layout

```
src/cutline/
  config.py      paths + the cost constants, stated as assumptions
  split.py       time-ordered splitter and the leakage demo — read this first
  data.py        CSV -> parquet, identity join, dtype downcasting
  metrics.py     PR-AUC, value-recall, the results-table appender
  synthetic.py   stand-in frame for running without the download
scripts/
  download_data.py  train files only
  prepare_data.py   CSV -> parquet, once
  01_baseline.py    Phase 1: the floor and the leakage demo
  smoke_test.py     ingestion round trip on synthetic CSVs, in a temp dir
reports/
  results.csv       every experiment, appended
```

## Currency

`TransactionAmt` is USD — IEEE-CIS is Vesta data. The cost model stays in USD so
units never mix; convert once at presentation time and show the rate you used
(`config.USD_TO_INR`).
