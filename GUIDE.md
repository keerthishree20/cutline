# Cutline — Complete Project Guide

A complete guide from zero to a deployed-ready fraud risk scorer with a cost-based decision threshold.
Covers every phase, every feature, every design decision and the reason behind it, with the real code.
It is self-contained: you can paste it into any AI chat and ask questions about the project without
sharing the repository.

**Repository:** https://github.com/keerthishree20/cutline
**All projects:** https://github.com/keerthishree20

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack & Why](#2-tech-stack--why)
3. [Project Setup from Scratch](#3-project-setup-from-scratch)
4. [Core Ideas in Plain Words](#4-core-ideas-in-plain-words)
5. [Project Structure](#5-project-structure)
6. [The Dataset](#6-the-dataset)
7. [Phase 1: Time-Ordered Split, Baseline & Leakage Demo](#7-phase-1-time-ordered-split-baseline--leakage-demo)
8. [Phase 2: Features](#8-phase-2-features)
9. [Phase 2: LightGBM & Calibration](#9-phase-2-lightgbm--calibration)
10. [Metrics, and Why These Ones](#10-metrics-and-why-these-ones)
11. [Phase 3: The Cost Model](#11-phase-3-the-cost-model)
12. [Two Thresholds: τ\* and τ_ship](#12-two-thresholds-τ-and-τ_ship)
13. [Three Decision Bands](#13-three-decision-bands)
14. [Sensitivity Analysis](#14-sensitivity-analysis)
15. [One Scoring Path](#15-one-scoring-path)
16. [Phase 4: The Scoring Service](#16-phase-4-the-scoring-service)
17. [Live History for Velocity Features](#17-live-history-for-velocity-features)
18. [Explanations with SHAP](#18-explanations-with-shap)
19. [Phase 5: The Dashboard](#19-phase-5-the-dashboard)
20. [Scoring Your Own Transactions](#20-scoring-your-own-transactions)
21. [Results](#21-results)
22. [Bugs Found Along the Way](#22-bugs-found-along-the-way)
23. [Changing the Cost Assumptions](#23-changing-the-cost-assumptions)
24. [Deployment](#24-deployment)
25. [Environment Variables & Security](#25-environment-variables--security)
26. [Troubleshooting](#26-troubleshooting)
27. [Complete Feature Summary](#27-complete-feature-summary)

---

## 1. Project Overview

Cutline scores payment transactions for fraud risk and decides whether to **allow**, **review** or
**block** each one.

Every fraud model outputs a probability. Turning that probability into a decision is an **economics
problem, not a modelling one**. A missed fraud costs the transaction value plus a dispute fee. A
wrongly declined customer costs lost margin, a support contact, and possibly the customer. Those costs
are not equal, so `0.5` is almost never the right cut. Cutline finds the cut that minimises actual
cost, and shows the curve it came from.

It is built in five phases:

| Phase | What | State |
|---|---|---|
| 1 | Data, time-ordered split, baseline floor, leakage demo | done |
| 2 | Feature engineering, LightGBM, calibration | done |
| 3 | Cost model and threshold sweep | done |
| 4 | FastAPI `/score` and `/explain` | done |
| 5 | Next.js review queue and threshold slider | done |

**Status:** all phases done and pushed. Not deployed yet (needs a Railway or Render and Vercel login).

---

## 2. Tech Stack & Why

| Technology | Role | Why We Chose It |
|---|---|---|
| **Python 3.12** | ML and API | the standard for tabular ML |
| **pandas, numpy** | Data | 590,000 rows fit comfortably in memory |
| **LightGBM** | Model | the strongest simple choice for tabular fraud data; handles missing values natively |
| **scikit-learn** | Baseline, isotonic calibration | logistic regression floor and `IsotonicRegression` |
| **SHAP** | Explanations | per-transaction reasons from the tree model |
| **FastAPI** | Scoring service | typed, fast, automatic docs |
| **Next.js 16, React 19, Tailwind 4** | Dashboard | slider, cost curve and review queue |
| **Docker** | Deployment | one image carries the model |
| **Parquet** | Data storage | fast, compact, keeps dtypes |

---

## 3. Project Setup from Scratch

The system `python3` is 3.6, so build the environment from 3.12.

### Run the demo with the committed model (no dataset needed)
```bash
git clone https://github.com/keerthishree20/cutline.git
cd cutline
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --port 8000        # terminal 1
cd frontend && npm install && npm run dev         # terminal 2 → http://localhost:3000
.venv/bin/python scripts/04_api_check.py          # checks every endpoint in-process
```

The trained model `models/cutline.joblib` and every file the dashboard reads (`reports/`) are
committed.

### Rebuild everything from the raw data
Needs Kaggle credentials in `~/.kaggle/kaggle.json` **and** a one-time acceptance of the competition
rules in a browser at https://www.kaggle.com/c/ieee-fraud-detection/rules.

```bash
.venv/bin/python scripts/download_data.py    # train files only, about 700 MB
.venv/bin/python scripts/prepare_data.py     # → data/interim/train.parquet
rm data/raw/*.csv                            # reclaim the space
.venv/bin/python scripts/01_baseline.py
.venv/bin/python scripts/02_model.py
.venv/bin/python scripts/03_cost_model.py
.venv/bin/python scripts/05_demo_queue.py
```

### Without the data
The phase scripts fall back to a synthetic stand-in with the same columns; `01_baseline.py` takes
`--synthetic` to force it. The numbers are meaningless; it only proves the code runs.
`scripts/smoke_test.py` round-trips synthetic CSVs through the real ingestion path in a temp folder.

---

## 4. Core Ideas in Plain Words

| Idea | Meaning |
|---|---|
| **Fraud rate** | about 3.5% of transactions here are fraud: a heavily imbalanced problem |
| **Threshold τ** | the probability above which a transaction is blocked |
| **Calibration** | a calibrated "0.30" really means about 30% of such transactions are fraud |
| **Cost curve** | total cost at every possible threshold; it is U-shaped |
| **PR-AUC** | area under the precision-recall curve; the honest headline for rare events |
| **Leakage** | the model accidentally learning from information it would not have in real life |
| **SHAP** | a method that says how much each feature pushed one prediction up or down |

---

## 5. Project Structure

```
src/cutline/
  config.py      paths, split fractions, and every cost assumption
  split.py       time-ordered splitter and the leakage demo (read this first)
  data.py        CSV → parquet, identity join, dtype downcasting
  features.py    history features (unfitted) + FeatureBuilder (fitted on training data)
  costs.py       cost(τ), the sweep, sensitivity, three-band policy
  bundle.py      load + score: the ONE scoring path, shared by every phase
  explain.py     SHAP → sentences an analyst would write
  serving.py     HistoryStore + Scorer (threshold read from the cost curve)
  metrics.py     PR-AUC, recall by value, ECE, the results appender
  synthetic.py   stand-in frame for running without the download
scripts/
  download_data.py  prepare_data.py
  01_baseline.py  02_model.py  03_cost_model.py  04_api_check.py  05_demo_queue.py
  smoke_test.py
api/main.py         FastAPI service
frontend/
  app/page.tsx      dashboard
  components/       CostCurve (hand-drawn SVG), QueueTable, ScorePanel, StatTile, DecisionBar
  lib/api.ts        API types and the single definition of the decision bands
reports/
  results.csv       every experiment, appended (the lab notebook)
  cost_curve.json   the thinned curve the slider reads
  demo_queue.json   the review queue
  sensitivity.csv   how far τ* moves when constants change
  v_selection.json  which V columns were kept
  sample_upload.csv sample file for the upload panel
models/cutline.joblib   encoder + model + calibrator, versioned together
Dockerfile  railway.json  render.yaml  DEPLOY.md
```

---

## 6. The Dataset

**IEEE-CIS Fraud Detection** (Kaggle, Vesta): 590,540 real card transactions, 3.50% fraud, 435 columns.

Two things that cost people a day each:
- **`TransactionDT` is not a timestamp.** It is seconds from an arbitrary start. Hour of day is
  `(dt % 86400) // 3600`, never a date parse.
- **The identity join is mostly misses.** `train_identity` covers only about a quarter of
  transactions, so device and `id_*` columns are empty for most rows. `has_identity` records that
  absence, and it is predictive: fraud runs at 7.85% on rows with identity data against 2.09% without.

Only the two `train_*` files are downloaded. The competition's test labels were never released, so the
test set here comes from splitting the training data by time.

Amounts are **USD**. `USD_TO_INR` in config is for display only.

---

## 7. Phase 1: Time-Ordered Split, Baseline & Leakage Demo

### Split by time, never randomly
```python
def time_ordered_split(df, train_frac=0.70, calib_frac=0.10):
    ordered = df.sort_values(DT_COL, kind="mergesort").reset_index(drop=True)
    dt = ordered[DT_COL].to_numpy()
    train_end = np.quantile(dt, train_frac)
    calib_end = np.quantile(dt, train_frac + calib_frac)
    train = ordered[dt <= train_end]
    calib = ordered[(dt > train_end) & (dt <= calib_end)]
    test  = ordered[dt > calib_end]
```
The cut is placed on a time boundary, so no single instant is split between two sets. A random split
would let the model train on the same period it is tested on.

### The baseline floor
`01_baseline.py` trains a logistic regression on ten obvious numeric columns. It is deliberately not
good; it is the number Phase 2 has to beat.

### The leakage demo
Comparing a time split with a random split is confounded, because the two test sets have different
fraud rates, and PR-AUC's floor *is* the fraud rate. So the evaluation set is held fixed and only the
training window changes:

```
[ ---------- past 80% ---------- ][ future 20% ]
                                    F1     F2
honest : trains on past                  → scored on F2
leaky  : trains on past + F1             → scored on F2
```

**Honest result:** on the baseline's features, the leak premium is −0.5%, i.e. noise, because static
card attributes give a leak nothing to exploit. The guard stays in the code, but "look at the gap" is
not a slide on these features.

Every experiment appends a row to `reports/results.csv`.

---

## 8. Phase 2: Features

### History features (no fitting, strictly backward-looking)
Run once on the whole frame before splitting. Every value depends only on the row itself and earlier
rows of the same card:

```python
out["hour"] = hour_of_day(out[DT_COL])
out["log_amt"] = np.log1p(amt)
out["amt_cents"] = (amt * 100) % 100          # converted fraud lands on odd cents
out["amt_is_round"] = (amt % 1 == 0)
for each card1 group (time-sorted):
    card1_count_1h, card1_count_24h           # how busy this card was
    card1_sec_since_prev                      # time since the card's last transaction
    card1_amt_z = (amt - prior mean) / (prior std + 1), only after 3 prior transactions
card1_is_first_seen                           # no history is a state, not a missing value
email_mismatch, email_known                   # purchaser vs recipient email domain
```

### FeatureBuilder (fitted on the training slice only)
- Frequency encodings fitted on the **fit slice alone**. Unseen keys get `0`, never NaN.
- Category levels **pinned from training**, so the same string always gets the same code.
- **V-column selection is fitted, not configured:** the 339 `V` columns fall into 14 groups that go
  missing together. Within-group correlation was measured first (median |r| only 0.14–0.26), so only
  true near-duplicates (r > 0.95) are dropped. The 14 "group present" flags are kept as features.
- High-cardinality identity strings (OS, browser, screen size) are capped at 30 levels.
- Absent source columns in a request become NaN, because here a missing column is missing data.

Result: 34 features in the first model, **380** in the full one.

### Two leaks guarded structurally
1. **The encoder is the leak, not the velocity features.** Fitting frequency maps on the full frame
   would let a card that appears only in the test period feed a training feature.
   `unseen_rate()` returning exactly 0.0 on held-out data is the signature of that; the smoke test
   asserts against it.
2. **`.astype("category")` on train and test separately** gives the same string different integer
   codes. `transform` pins levels with `pd.Categorical(..., categories=...)`.

---

## 9. Phase 2: LightGBM & Calibration

Four slices, each held back for a named reason:

```
[ ---- fit ---- ][ early-stop ][ calib ][ ---- test ---- ]
 <------ train 70% ---------->    10%        20%
```

- **fit** trains the model and the encoder,
- **early-stop** picks the number of trees,
- **calib** is used only by isotonic regression,
- **test** is touched once.

Isotonic regression is fitted directly (not via `CalibratedClassifierCV`) so it is obvious what was
fitted on what, and the reliability plot can show raw and calibrated scores together.

**Isotonic barely moves PR-AUC, and that is correct.** It preserves ranking. What it moves is
calibration error, which is the only number the cost model depends on. It costs a little PR-AUC
(ties) and buys about 11× lower ECE.

The model, encoder and calibrator are saved **together** in `models/cutline.joblib`.

---

## 10. Metrics, and Why These Ones

| Metric | Why |
|---|---|
| **PR-AUC** (headline) | at 3.5% fraud, ROC-AUC reads ~0.95 while precision sits near 8%; PR-AUC is honest |
| **PR-AUC lift** | PR-AUC divided by the fraud rate, comparable across datasets |
| **Recall by value** | catching half the fraudulent *dollars* is what a merchant feels |
| **ECE with quantile bins** | calibration error; equal-width bins put 99% of scores in one bucket |
| **Brier score** | overall probability accuracy |

---

## 11. Phase 3: The Cost Model

`src/cutline/costs.py`. The constants:

```python
dispute_fee   25.00 USD  per missed chargeback
gross_margin   8%        earned on a legitimate sale
support_cost   2.00 USD  per false decline
churn_cost    30.00 USD  per false decline (25% × 120 lifetime value)
analyst_cost   1.50 USD  per manual review
```

### The sweep: every threshold in one pass
```python
order = np.argsort(-p, kind="mergesort")      # highest score first
caught_n     = np.cumsum(y)
caught_value = np.cumsum(y * a)
fp_n         = np.arange(1, n + 1) - caught_n
fp_value     = np.cumsum((1.0 - y) * a)
missed_value = total_fraud_value - caught_value
fraud_cost   = missed_value + c.dispute_fee * missed_n
decline_cost = c.gross_margin * fp_value + c.false_decline_fixed * fp_n
keep = np.r_[p[:-1] != p[1:], True]           # collapse tied scores
```

Sorting by score turns "block everything above τ" into "block the top k", so every quantity is a
running sum and the whole curve costs one sort.

### Why collapse tied scores?
"Block the top k" equals "block everything ≥ τ" only if no row outside the top k shares the k-th
score. **Isotonic calibration collapses 118,108 scores to 156 distinct values**, so ties are
everywhere. Without collapsing, the curve priced policies no threshold could deliver.

Outputs: `cost_curve.png`, `cost_curve.json` (what the slider reads), `sensitivity.csv`.

---

## 12. Two Thresholds: τ\* and τ_ship

```python
def optimal(curve):            # τ*: the bottom of the U
    return curve.loc[curve["cost"].idxmin()]

def constrained_optimal(curve, max_decline_rate=0.01):   # τ_ship
    feasible = curve[curve["good_declined_rate"] <= max_decline_rate]
    if feasible.empty:
        return curve.loc[curve["good_declined_rate"].idxmin()]
    return feasible.loc[feasible["cost"].idxmin()]
```

Pure cost minimisation treats a wrongly declined customer as a small fixed loss, so it can happily
block a lot of good customers. Real risk teams work under a **decline-rate budget** set by the
business. So both are reported, and **the service uses τ_ship** (at most 1% of good customers
declined).

`policy_at(τ)` evaluates an exact threshold. `at_threshold(curve, τ)` only snaps to the nearest curve
row, which once made "τ = 0.50" silently mean 0.5937.

---

## 13. Three Decision Bands

```python
def decide(self, p):
    if p >= self.tau_block:  return "block"
    if p >= self.tau_review: return "review"     # tau_review = tau_block / 3
    return "allow"
```

The review band needed two brakes: an analyst **catch rate of 0.70** and a **queue cap of 2% of
traffic** (overflow is allowed through). Without them, sending most traffic to humans "won" on an
artifact: no analyst team can clear 65% of all transactions.

---

## 14. Sensitivity Analysis

`sensitivity.csv` shows how far τ\* moves when each constant changes. It moves a lot, so **the
constants are load-bearing**. The method is sound; the constants need a real merchant's numbers. Say
that first rather than let a judge find it.

---

## 15. One Scoring Path

```python
def score(bundle, df, calibrated=True):
    X = bundle["feature_builder"].transform(df)
    raw = bundle["model"].predict_proba(X)[:, 1]
    return bundle["isotonic"].predict(raw) if calibrated else raw
```

`bundle.py` holds the **only** `score()` in the project. Phase 3's sweep, Phase 4's endpoint and
Phase 5's queue all call it. A service that rebuilt features from a second copy of the logic would
return numbers that disagree with the metrics table, usually discovered the night before a demo.
`smoke_test.py` asserts the saved bundle round-trips to byte-identical scores.

---

## 16. Phase 4: The Scoring Service

```bash
.venv/bin/uvicorn api.main:app --reload --port 8000
.venv/bin/python scripts/04_api_check.py
```

| Endpoint | Returns |
|---|---|
| `GET /health` | model loaded, what it was trained on, synthetic warning |
| `GET /policy` | thresholds in force and the constants behind them |
| `GET /curve` | `reports/cost_curve.json`, verbatim |
| `GET /queue` | `reports/demo_queue.json`, verbatim |
| `GET /sample` | held-out rows for the upload panel, labels stripped |
| `POST /score` | probability and decision |
| `POST /explain` | the same plus three plain-language reasons |
| `POST /replay` | a batch, `?explain=true` for reasons |

**The threshold is read from `cost_curve.json`, never hardcoded.** If the file failed to load, the
service would silently fall back to 0.5 and undo the whole point of the project. `04_api_check.py`
asserts the threshold is **not** 0.5. `/curve` and `/queue` return files verbatim, and a missing file
is a 503 with instructions, never a made-up default.

### A sparse request scores high, and that is correct
A three-field `curl` gets `block` at p=0.19. Missing columns become NaN, LightGBM routes NaN down its
default branch, and `card1_is_first_seen` fires: no history really is predictive here. So responses
include `sparse`, `fields_absent` and `history_seen`. **Demo through `/replay` with history, never one
cold request.**

---

## 17. Live History for Velocity Features

Velocity features look back over a card's past transactions, but a single request carries no history.
`HistoryStore` keeps recent transactions in memory and, for a new one, runs **the training function**
over just that card's rows. Because `add_history_features` depends only on a row and earlier rows of
the same card, restricting to that card gives bit-identical output. `smoke_test.py` proves batch and
live scoring agree to 1e-9.

In production, a feature store would replace the in-memory store; the feature code would not change.

---

## 18. Explanations with SHAP

`src/cutline/explain.py`:
- `Explainer.top_reasons()` returns the three features that pushed the score most.
- `phrase()` turns each into a sentence an analyst would write, e.g. "placed at 05:00", "no device or
  browser information", "risk pattern learned for this specific card".

```
POST /explain → review  p=0.0338
  - placed at 05:00                              (+0.533)
  - no device or browser information             (+0.219)
  - risk pattern learned for this specific card  (+0.112)
```

Rules that took two attempts:
- **SHAP explains the raw output, not the calibrated probability.** Order and direction carry over;
  magnitudes are log-odds. The payload says so.
- `TransactionAmt` and `log_amt` are the same fact; reasons are deduplicated by their text.
- `card1`, `C13`, `V294` have no public meaning, so phrasing never pretends: anonymised fields are
  labelled as anonymised.
- A separate **context** line gives plain facts (time, prior activity, device, email domain), marked
  as facts, not as reasons. Add information; never reweight attribution to read better.

The explainer is built at startup: lazily it cost about 2.5 s on the first `/explain`.

---

## 19. Phase 5: The Dashboard

Next.js 16, React 19, Tailwind 4. `frontend/app/page.tsx`.

| Panel | Details |
|---|---|
| **Threshold slider** | `min=0 max=155 step=1` over the curve array, so it can never land between two real thresholds |
| **Stat tiles** | cost, fraud caught, decline rate, read from the curve row covering all 118,108 test rows; never recomputed from the visible sample |
| **Cost curve** | hand-drawn SVG; y-axis cut at 1.12× the do-nothing cost, and the chart says so |
| **Review queue** | 300 held-out rows sorted by **expected loss** (probability × amount), each showing whether the decision was right |
| **Upload panel** | score your own CSV or the sample file live |

### Two things the first render got wrong
- The full y-axis squashed the U into the bottom sixth (blocking everything costs about $4.9M against
  $711k for nothing), so it read as a flat line.
- Sorting by raw probability put ten identical `1.0000` rows on top.

### Colours validated, not chosen
The first review and block colours were almost identical to red-green colourblind viewers (ΔE 4.7
under deuteranopia). Re-stepped: light `#C2183C / #D08700 / #127A4A`, dark `#D9466A / #C68420 /
#0E9280`. Every chip also carries a text label.

`frontend/lib/api.ts` mirrors `Scorer.decide` exactly (block at τ, review at τ/3).

### The queue is generated offline, on purpose
`HistoryStore` starts empty, so live-scored rows would lack history and disagree with the curve.
`05_demo_queue.py` builds the queue from the same batch-featurised test slice as the curve, and
asserts agreement by rescoring a probe row.

---

## 20. Scoring Your Own Transactions

Drop a CSV into the upload panel, or press the sample button; rows go through
`POST /replay?explain=true` against the live model.

- The sample file is a **disclosed mix**: half from the high-risk tail (12 of 25 block), and the panel
  says so. At a 3.4% fraud rate, 25 random rows flag nothing.
- The button scores **real held-out rows**, not an invented "suspicious" one; a hand-made row was
  allowed, because the model leans on anonymised counters nobody can fabricate sensibly.

---

## 21. Results

590,540 transactions, 3.50% fraud. Test slice: the last 118,108 by time.

| Run | Features | PR-AUC | Lift | Precision @ 50% recall | ECE |
|---|---|---|---|---|---|
| baseline logreg (floor) | 13 | 0.1325 | 3.85× | 0.115 | 0.3645 |
| LightGBM, first pass | 34 | 0.4263 | 12.39× | 0.333 | 0.0043 |
| **LightGBM, full feature set** | **380** | **0.4718** | **13.71×** | **0.392** | **0.0053** |

**+256% PR-AUC over the floor.**

### The cost table (test set: 4,064 fraudulent, 16.2M USD)
| Policy | τ | Cost (USD) | Fraud caught by value | Good declined |
|---|---|---|---|---|
| approve everything | — | 711,534 | 0.0% | 0.00% |
| τ = 0.50 (default) | 0.5000 | 554,789 | 23.7% | 0.54% |
| τ\* (pure cost minimum) | 0.2423 | 527,893 | 35.4% | 1.57% |
| **τ_ship (≤ 1% declines)** | **0.3559** | **539,991** | **28.4%** | **0.88%** |

**The shipped policy catches 28.4% of fraud by value while declining 0.88% of good customers, saving
171,543 USD on the test set, about 14,524 per 10,000 transactions.**

### Latency
The 380-feature model first took 248 ms to score and 550 ms with SHAP. The cause was `transform`
assigning 380 columns one at a time. Building the frame in one pass brought it to **83 ms scoring,
207 ms with SHAP**.

---

## 22. Bugs Found Along the Way

| Bug | What happened | Fix |
|---|---|---|
| early stopping on the wrong metric | with class weights, log loss worsens from tree 1, so training stopped at **one tree**, silently | `metric="average_precision"`, `first_metric_only=True` |
| equal-width ECE bins | 99% of scores in the first bucket | quantile bins |
| `card1_amt_z` measuring card age | with 1-2 prior transactions the std is noise; mean \|z\| fell from 1.63 to 0.56 as history grew | require 3 prior transactions, else NaN |
| named policies snapping | "τ = 0.50" reported τ = 0.5937 | `policy_at` evaluates the exact threshold |
| sweep pricing impossible policies | tied scores | collapse ties |
| review band "winning" | analysts assumed to clear everything | catch rate 0.70, queue cap 2% |
| unused columns | the first model used 31 of 435 columns | V, C, D and id blocks brought in, fitted selection |
| slow transform | column-by-column assignment | build in one pass |

---

## 23. Changing the Cost Assumptions

Every constant in `src/cutline/config.py` is an **assumption**, not a measurement:

| Constant | Meaning |
|---|---|
| `DISPUTE_FEE` | fee per chargeback |
| `GROSS_MARGIN` | share of a legitimate sale actually earned |
| `SUPPORT_COST` | one "why was I declined" contact |
| `ANALYST_COST` | one manual review |
| `CHURN_RATE`, `CUSTOMER_LIFETIME_VALUE` | expected cost of losing a falsely declined customer |
| `MAX_DECLINE_RATE` | the ceiling that picks τ_ship |

To use a real merchant's numbers: edit them, rerun `03_cost_model.py` and `05_demo_queue.py`, then
restart the API.

---

## 24. Deployment

Full checklist in `DEPLOY.md`.

1. **API** on Railway or Render, built from the `Dockerfile` (health check `/health`). Needs roughly
   512 MB to 1 GB of RAM for LightGBM, scikit-learn and SHAP. Set `ALLOWED_ORIGINS` to the dashboard's
   URL. If a free tier runs out of memory, drop `/explain` before the model.
2. **Dashboard** on Vercel from `frontend/`. Set `NEXT_PUBLIC_API_URL` **before building**; it is baked
   in at build time.
3. **Verify in order:** `/health`, then `/policy` (τ must NOT be 0.5), then `/curve`.

---

## 25. Environment Variables & Security

| Variable | Where | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | API | CORS allowlist; never `*` |
| `NEXT_PUBLIC_API_URL` | frontend | where the dashboard calls; defaults to `http://localhost:8000` |
| Kaggle `~/.kaggle/kaggle.json` | local only | downloading the dataset; never committed |

No personal data is stored; the dataset is anonymised by Vesta. The service holds history in memory
only.

---

## 26. Troubleshooting

| Problem | Fix |
|---|---|
| dashboard shows network errors | API not on port 8000, or `NEXT_PUBLIC_API_URL` wrong; in production, set `ALLOWED_ORIGINS` |
| `/curve` or `/queue` returns 503 | run `03_cost_model.py` or `05_demo_queue.py` |
| `02_model.py` reports no floor | run `01_baseline.py` once first |
| Kaggle download 403 | accept the competition rules in a browser with the same account |
| API out of memory on a free tier | drop `/explain` |
| every uploaded row allowed | random rows rarely include fraud; use the mixed sample file |

---

## 27. Complete Feature Summary

### All Features Built

| # | Feature | Type | Key Files |
|---|---|---|---|
| 1 | CSV → parquet ingestion with identity join | Data | `data.py`, `prepare_data.py` |
| 2 | Time-ordered split and leakage demo | ML | `split.py` |
| 3 | Baseline floor | ML | `01_baseline.py` |
| 4 | Backward-looking history features | ML | `features.py` |
| 5 | Fitted FeatureBuilder with leak guards | ML | `features.py` |
| 6 | Fitted V-column selection | ML | `features.py`, `v_selection.json` |
| 7 | LightGBM with correct early stopping | ML | `02_model.py` |
| 8 | Isotonic calibration | ML | `02_model.py`, `bundle.py` |
| 9 | Cost sweep with tie collapsing | Economics | `costs.py` |
| 10 | τ\* and constrained τ_ship | Economics | `costs.py` |
| 11 | Three-band policy with brakes | Economics | `costs.py`, `serving.py` |
| 12 | Sensitivity analysis | Economics | `costs.py` |
| 13 | Single shared scoring path | Serving | `bundle.py` |
| 14 | FastAPI scoring service | Serving | `api/main.py` |
| 15 | Live history store with parity test | Serving | `serving.py`, `smoke_test.py` |
| 16 | SHAP reasons in plain language | Serving | `explain.py` |
| 17 | Threshold slider bound to curve index | Frontend | `page.tsx` |
| 18 | Hand-drawn cost curve | Frontend | `CostCurve.tsx` |
| 19 | Expected-loss review queue | Frontend | `QueueTable.tsx`, `05_demo_queue.py` |
| 20 | CSV upload scoring | Frontend | `ScorePanel.tsx` |
| 21 | Docker and deploy configs | Ops | `Dockerfile`, `railway.json`, `render.yaml` |

### Data Flow Architecture

```
IEEE-CIS CSVs ──► prepare_data.py ──► train.parquet
  └── add_history_features (whole frame, backward-looking)
        └── time_ordered_split: fit | early-stop | calib | test
              ├── FeatureBuilder.fit(fit slice) ──► LightGBM ──► isotonic(calib)
              │        └── models/cutline.joblib (builder + model + calibrator)
              └── test ──► bundle.score() ──► costs.sweep() ──► cost_curve.json, τ*, τ_ship
                                            └── 05_demo_queue.py ──► demo_queue.json

API (FastAPI :8000)
  ├── Scorer: τ_block from cost_curve.json, τ_review = τ_block / 3
  ├── HistoryStore ──► add_history_features(card's rows) ──► bundle.score() ──► decide()
  └── Explainer (SHAP) ──► top reasons ──► phrase()

Dashboard (Next.js :3000)
  ├── GET /curve ──► slider over curve rows ──► stat tiles
  ├── GET /queue ──► expected-loss queue
  └── upload ──► POST /replay?explain=true
```

### Tech Stack at a Glance

```
ML:        pandas, numpy, LightGBM, scikit-learn (isotonic), SHAP
Backend:   FastAPI + uvicorn, joblib bundle
Frontend:  Next.js 16 + React 19 + Tailwind 4, hand-drawn SVG
Data:      IEEE-CIS Fraud Detection (Kaggle), parquet
Deploy:    Docker (Railway or Render) + Vercel
```
