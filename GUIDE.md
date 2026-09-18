# Cutline — Complete Project Guide

## Table of Contents
1. [What is Cutline?](#what-is-cutline)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Architecture](#architecture)
5. [The Five Phases](#the-five-phases)
6. [Code Walkthrough](#code-walkthrough)
7. [API Reference](#api-reference)
8. [The Dashboard](#the-dashboard)
9. [Changing the Cost Assumptions](#changing-the-cost-assumptions)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)

---

## What is Cutline?

Cutline scores payment transactions for fraud risk and decides whether to **allow**, **review** or
**block** each one.

Most fraud models stop at a probability and use `0.5` as the cut. Cutline treats the cut as an
economics problem. A missed fraud costs the transaction value plus a dispute fee. A wrongly declined
customer costs lost margin, a support contact and possibly the customer. Those costs are not equal,
so Cutline sweeps every possible threshold, prices each one, and ships the cheapest one that keeps
declines of good customers under a ceiling.

It is trained on the IEEE-CIS Fraud Detection dataset: about 590,000 real card transactions with a
fraud rate of about 3.5%. The model is LightGBM with isotonic calibration, served by FastAPI, with a
Next.js dashboard.

---

## Quick Start

The system `python3` is 3.6. Build the venv from Python 3.12.

### Run the service and dashboard with the committed model
The trained model and every file the dashboard reads are committed, so you do not need the dataset
to run the demo.

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api.main:app --port 8000        # terminal 1
cd frontend && npm install && npm run dev         # terminal 2, http://localhost:3000
```

Check the service:

```bash
.venv/bin/python scripts/04_api_check.py          # every endpoint, in-process
```

### Rebuild everything from the raw data
Needs Kaggle credentials in `~/.kaggle/kaggle.json` and a one-time acceptance of the competition
rules in a browser at https://www.kaggle.com/c/ieee-fraud-detection/rules.

```bash
.venv/bin/python scripts/download_data.py     # train files only, about 700 MB
.venv/bin/python scripts/prepare_data.py      # to data/interim/train.parquet
.venv/bin/python scripts/01_baseline.py
.venv/bin/python scripts/02_model.py
.venv/bin/python scripts/03_cost_model.py
.venv/bin/python scripts/05_demo_queue.py
```

### Without the data
The phase scripts fall back to a synthetic stand-in frame with the same columns, and `01_baseline.py`
takes `--synthetic` to force it. The numbers are
meaningless. It only proves the code runs.

```bash
.venv/bin/python scripts/01_baseline.py --synthetic
.venv/bin/python scripts/smoke_test.py        # ingestion round trip and parity checks
```

---

## Core Concepts

### Time-ordered split
Transactions are split by time, never randomly. A random split lets the model train on data from the
same period it is tested on, which flatters it. `src/cutline/split.py` is the place to start reading.

### PR-AUC and lift
At a 3.5% fraud rate, ROC-AUC looks excellent even for weak models. PR-AUC, precision against recall,
is the honest headline. Lift is PR-AUC divided by the fraud rate, so results stay comparable across
datasets.

### Calibration
A calibrated model's "0.30" really means about 30% of such transactions are fraud. The cost model
multiplies probabilities by money, so calibration matters more than ranking. Isotonic regression is
fitted on a separate slice. ECE, expected calibration error, measures it with quantile bins because
almost all scores sit near zero.

### The cost of a threshold, `cost(τ)`
For a threshold τ, block everything scoring at or above it:
- each fraud let through costs its amount plus `DISPUTE_FEE`,
- each good customer blocked costs margin, `SUPPORT_COST` and expected churn.

Sweeping τ gives a U-shaped curve.

### Two thresholds
- **τ\*** is the pure cost minimum.
- **τ_ship** is the cheapest threshold that declines at most `MAX_DECLINE_RATE` of good customers,
  1% by default. The service uses τ_ship.

### Three decision bands
Block at τ and above, review between τ/3 and τ, allow below. The review band is limited by an analyst
catch rate and a queue cap, so it cannot "win" by sending most traffic to humans.

### One scoring path
`src/cutline/bundle.py` holds the only `score()` function. Training evaluation, the API and the
dashboard all use it, so their numbers cannot drift apart.

---

## Architecture

```
  IEEE-CIS CSVs ─► prepare_data.py ─► data/interim/train.parquet
                                              │
                   01_baseline ─ 02_model ─ 03_cost_model ─ 05_demo_queue
                                    │              │               │
                          models/cutline.joblib    │               │
                          (encoder + LightGBM +    │               │
                           isotonic calibrator)    ▼               ▼
                                    │     reports/cost_curve.json  reports/demo_queue.json
                                    ▼              │               │
                         ┌──────────────────────── api/main.py (FastAPI :8000) ─────┐
                         │ Scorer: threshold read from cost_curve.json               │
                         │ HistoryStore: per-card history for velocity features      │
                         │ SHAP explainer built at startup                           │
                         └───────────────────────────┬──────────────────────────────┘
                                                     │ JSON
                                          frontend/ (Next.js 16 :3000)
                                          slider, cost curve, review queue, upload
```

---

## The Five Phases

| phase | script | produces |
|---|---|---|
| 1 | `01_baseline.py` | the logistic-regression floor and the leakage demonstration |
| 2 | `02_model.py` | features, LightGBM, isotonic calibration, `models/cutline.joblib` |
| 3 | `03_cost_model.py` | the threshold sweep, `cost_curve.json`, `cost_curve.png`, `sensitivity.csv` |
| 4 | `api/main.py` | the scoring service, checked by `04_api_check.py` |
| 5 | `frontend/` and `05_demo_queue.py` | the dashboard and its review queue |

Every experiment appends a row to `reports/results.csv`, the project's lab notebook. The README has
the numbers from each phase and the bugs each one uncovered.

---

## Code Walkthrough

### `src/cutline/`
| file | responsibility |
|---|---|
| `config.py` | paths, split fractions, and every cost assumption |
| `split.py` | the time-ordered splitter and the leakage demo |
| `data.py` | CSV to parquet, the identity join, dtype downcasting |
| `features.py` | history features, which need no fitting, and `FeatureBuilder`, fitted on training data only |
| `costs.py` | `cost(τ)`, the sweep with tied scores collapsed, sensitivity, the three-band policy |
| `bundle.py` | load the model bundle and `score()`, the single scoring path |
| `explain.py` | turns SHAP values into sentences an analyst would write |
| `serving.py` | `HistoryStore` and `Scorer`, which reads its threshold from the cost curve |
| `metrics.py` | PR-AUC, recall by value, ECE, and the results appender |
| `synthetic.py` | the stand-in frame for running without data |

### Things that look odd but are deliberate
- **`TransactionDT` is not a timestamp.** It is seconds from an arbitrary start. Hour of day is
  `(dt % 86400) // 3600`.
- **Most identity columns are empty.** Only about a quarter of transactions have identity data.
  `has_identity` records that absence, and it is predictive.
- **Encoders are fitted on the training slice only.** Unseen keys get `0`, never NaN. Category levels
  are pinned from training so the same string always gets the same code.
- **V-column selection is fitted, not configured.** Near-duplicates are dropped within each of the 14
  missing-together groups, using correlations from the training slice.
- **SHAP explains the raw model output**, not the calibrated probability. Order and direction carry
  over. Magnitudes are log-odds, and the API says so.

### `scripts/`
The five phase scripts, the data download and preparation, and `smoke_test.py`, which round-trips
synthetic CSVs through the real ingestion path and checks that batch and live scoring agree to
1e-9.

---

## API Reference

Run with `.venv/bin/uvicorn api.main:app --port 8000`.

| endpoint | returns |
|---|---|
| `GET /health` | whether the model loaded and what it was trained on |
| `GET /policy` | the thresholds in force and the cost constants behind them |
| `GET /curve` | `reports/cost_curve.json`, served verbatim |
| `GET /queue` | `reports/demo_queue.json`, served verbatim |
| `GET /sample` | a sample upload file of held-out rows |
| `POST /score` | probability and decision |
| `POST /explain` | the same plus three plain-language reasons and a context line |
| `POST /replay` | a batch of transactions, with `?explain=true` for reasons |

A request with only a few fields often scores high. Missing fields become NaN, and a card with no
history is genuinely riskier in this data. The response flags this with `sparse`, `fields_absent` and
`history_seen`. Demonstrate through `/replay` with history, not one cold request.

If `/policy` shows `tau_block` of exactly `0.5`, the cost curve failed to load and the service fell
back to the default. `04_api_check.py` asserts against this.

---

## The Dashboard

`frontend/app/page.tsx`, built with Next.js 16, React 19 and Tailwind 4.

- **Threshold slider.** Moves over the stored curve points by index, so it can never land between
  two real thresholds. Moving it changes the threshold, never the model, which is why it is instant.
- **Stat tiles.** Cost, fraud caught and decline rate, read from the curve row covering the whole test
  set. Never recomputed from the visible queue sample.
- **Cost curve.** Hand-written SVG in `components/CostCurve.tsx`. The y-axis is cut at 1.12 times the
  do-nothing cost, and the chart says so.
- **Review queue.** Sorted by expected loss, probability times amount. Each row shows whether the
  decision was right, since this is held-out data with known labels.
- **Upload panel.** Score your own CSV, or the sample file, through the live model.

Status colours were validated for colour-vision deficiency, and every chip also carries a text label.

`frontend/lib/api.ts` holds the API types and the single frontend definition of the decision bands,
mirroring `Scorer.decide`.

---

## Changing the Cost Assumptions

Every constant in `src/cutline/config.py` is an assumption, not a measurement:

| constant | meaning |
|---|---|
| `DISPUTE_FEE` | fee per chargeback |
| `GROSS_MARGIN` | share of a legitimate sale actually earned |
| `SUPPORT_COST` | one "why was I declined" contact |
| `ANALYST_COST` | one manual review |
| `CHURN_RATE`, `CUSTOMER_LIFETIME_VALUE` | expected cost of losing a falsely declined customer |
| `MAX_DECLINE_RATE` | the ceiling that picks τ_ship |

To use a real merchant's numbers:
1. Edit the constants.
2. Rerun `scripts/03_cost_model.py` to regenerate `cost_curve.json`.
3. Rerun `scripts/05_demo_queue.py` so the queue matches.
4. Restart the API. It reads the new threshold at startup.

`reports/sensitivity.csv` shows how far τ\* moves when the constants change. The answer is a lot,
which is why the README calls the constants load-bearing.

Amounts are USD throughout. `USD_TO_INR` is for display only.

---

## Deployment

`DEPLOY.md` has the full checklist. In short:

1. **API** on Railway or Render, built from the `Dockerfile`. It needs roughly 512 MB to 1 GB of RAM
   for LightGBM, scikit-learn and SHAP. Set `ALLOWED_ORIGINS` to the dashboard's URL.
2. **Dashboard** on Vercel from `frontend/`. Set `NEXT_PUBLIC_API_URL` to the API's URL **before**
   building, because it is baked in at build time.
3. Verify `/health`, then `/policy` shows a threshold other than 0.5, then `/curve`.

---

## Troubleshooting

### The dashboard shows network errors
The API is not running on port 8000, or `NEXT_PUBLIC_API_URL` points somewhere else. In production,
CORS is the usual cause: set `ALLOWED_ORIGINS` on the API.

### `GET /curve` or `/queue` returns 503
The generated file is missing. Run `scripts/03_cost_model.py` or `scripts/05_demo_queue.py`.

### `02_model.py` reports no baseline floor
`reports/results.csv` has no baseline row for this data source. Run `01_baseline.py` once first.

### The Kaggle download fails with 403
Accept the competition rules in a browser, logged in as the account whose `kaggle.json` you use.

### The API runs out of memory on a free tier
The SHAP explainer is the largest extra. Dropping `/explain` saves memory while keeping scoring.

### Every uploaded row is allowed
At a 3.5% fraud rate, a small random sample often contains no risky rows. The bundled sample file
deliberately mixes in high-risk held-out rows, and the panel says so.
