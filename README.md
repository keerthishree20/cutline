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
| 2 | Feature engineering, LightGBM, calibration | **done** |
| 3 | Cost model and threshold sweep | **done** |
| 4 | FastAPI `/score` + `/explain` | **done** |
| 5 | Next.js review queue + threshold slider | next |

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
notebook, and every experiment from here on adds a row. It is per-machine until
committed, so a fresh clone will report no floor on the first run of
`02_model.py` until `01_baseline.py` has been run once. The floor comparison is
scoped to the same data source and the most recent baseline row, because an
append-only notebook accumulates rows from older feature sets and, once the real
data lands, from a different dataset entirely.

## Phase 2 results

`scripts/02_model.py`. Four slices, three held back for a named reason:

```
[ ---- fit ---- ][ early-stop ][ calib ][ ---- test ---- ]
 <------ train 70% ---------->    10%        20%
```

`fit` trains the model and fits the encoder. `early-stop` picks the tree count.
`calib` is touched only by isotonic regression, so the probabilities the cost
model consumes are honest. `test` is touched once.

Isotonic is fitted directly rather than through `CalibratedClassifierCV` so it
is obvious exactly what was fitted on what, and so the reliability curve can
show raw and calibrated scores on the same axes
(`reports/calibration.png`).

**Isotonic barely moves PR-AUC, and that is correct.** It is a monotone
transform, so it preserves ranking almost exactly. What it moves is calibration
error — which is the only number `cost(τ)` actually rests on.

### Two bugs this phase produced, both silent

- **Early stopping fired on the wrong metric.** With `scale_pos_weight` applied,
  `binary_logloss` degrades from the first iteration by construction. Left in
  the metric list it stopped training at **one tree**, and the model still
  trained, still scored, still saved. Fixed by setting
  `metric="average_precision"` and `first_metric_only=True`.
- **Equal-width ECE bins say nothing at a 3.5% fraud rate.** Nearly every score
  sits near zero, so ~99% of the mass lands in the first bucket. `metrics.py`
  uses quantile bins.

### One feature that was measuring the wrong thing

`card1_amt_z` is meant to say "this amount is unusual for this card", and the
SHAP panel in Phase 4 will quote it back to a user in those words. Built from an
expanding std with a `+1.0` denominator floor, it wasn't saying that: with one
or two prior transactions the std estimate is noise and the floor dominates, so
mean `|z|` fell from 1.63 to 0.56 purely as card history grew, with no change in
the underlying amounts. It now requires `MIN_PRIOR_FOR_Z = 3` prior
transactions and emits NaN below that — the "no history" state is already
carried by the count and first-seen features.

### Two leaks the code guards against structurally

Both compile, train, and score fine while being wrong, so neither shows up as
an error:

- **The encoder is the leak, not the velocity features.** Backward-looking
  rolling windows computed before splitting are fine — every value depends only
  on earlier rows. But fitting frequency maps on the full frame lets a card that
  appears only in the test period contribute to a training feature.
  `FeatureBuilder` is fitted on the fit slice alone, and unseen keys get a
  sentinel of `0`, never NaN. `unseen_rate()` on held-out data returning exactly
  `0.0` is the signature of an encoder fitted on too much; the smoke test
  asserts against it.
- **`.astype("category")` on train and test independently gives the same string
  different integer codes.** LightGBM then reads garbage at inference, silently.
  `transform` pins levels from training via `pd.Categorical(..., categories=...)`.

### Real numbers, on IEEE-CIS

590,540 transactions, 3.50% fraud. Test slice is the last 118,108 by time.

| run | PR-AUC | lift over base rate | ECE | Brier |
|---|---|---|---|---|
| baseline logreg (floor) | 0.1325 | 3.85x | 0.3645 | 0.1827 |
| LightGBM raw | 0.4391 | 12.76x | 0.1073 | 0.0551 |
| **LightGBM calibrated** | **0.4263** | **12.39x** | **0.0043** | **0.0244** |

**+221.7% over the floor**, and calibration error down 96%. Isotonic costs a
little PR-AUC (ties) and buys a 25x improvement in ECE, which is the trade the
cost model needs.

`has_identity` earned its place: fraud runs at **7.85%** on the 24.4% of
transactions that carry identity data, against **2.09%** on those that do not.

**The leakage demo does not fire on this feature set.** Trained on past only:
0.1289. Trained on past plus half the future: 0.1282. The premium is -0.5% —
noise. The baseline's thirteen features are static card attributes with nothing
time-varying for a leak to exploit, so seeing the future buys nothing. The
time-ordered split is still the right call and the guard still belongs in the
code, but "look at the gap" is not a slide on these features. It would take
re-running the demo with the Phase 2 feature set to make that argument, and the
honest thing is to say so rather than show a gap that is not there.

## Phase 3 results — the cost model

`scripts/03_cost_model.py`. The argument in one table (synthetic figures, shape
not magnitude):

On IEEE-CIS: 118,108 test transactions, 4,064 fraudulent, 16.2M USD of value.

| policy | τ | cost (USD) | fraud caught by value | good declined |
|---|---|---|---|---|
| approve everything | — | 711,534 | 0.0% | 0.00% |
| τ = 0.50 (default) | 0.5000 | 571,963 | 18.6% | 0.51% |
| **τ\* (cost-optimal)** | **0.0561** | **422,401** | **64.6%** | 10.51% |
| τ = 0.05 (paranoid) | 0.0500 | 432,814 | 67.9% | 12.57% |

**At τ\* we catch 64.6% of fraud by value while declining 10.5% of good
customers, saving 289,133 USD on the test set — 24,480 USD per 10,000
transactions.**

The default threshold captures 18.6% of fraud value and costs 571,963 against
711,534 for having no model at all. A classifier at 12.4x base-rate lift
delivers about a fifth of its available value until someone chooses the
threshold on purpose. That is the whole argument.

Outputs: `cost_curve.png` (the U, τ\* marked), `cost_curve.json` (what the
Phase 5 slider reads — snap it to the 156 distinct values, not a continuous
range, or it will show changing numbers for an unchanged decision — dragging changes the threshold, never the model, which
is why it is instant), `sensitivity.csv`.

### Two results here that are uncomfortable, and are reported anyway

- **τ\* declines 30% of good customers.** No merchant would accept that. The
  cost model is not wrong — it is reporting that the *model* is weak. When a
  classifier cannot separate, blocking indiscriminately genuinely is cheaper
  under these constants. The fix is a better classifier, or a decline-rate
  ceiling imposed as a business constraint on top of the cost minimum. The
  script prints this warning whenever the rate exceeds 5%.
- **τ\* moves 5.0x across the sensitivity variants** (0.040 → 0.201). The
  assumed constants are load-bearing. Lead with that rather than letting a judge
  find it: the method is sound, and the constants need a real merchant's
  numbers. `sensitivity.csv` has the full table.

### Two bugs the cost model produced

- **Named policies were snapping to the nearest score present.** `at_threshold`
  finds the closest row on the curve, so "τ = 0.50" reported τ=0.5937 — the cost
  of blocking two transactions, labelled as the default policy. `policy_at`
  evaluates the exact threshold. The "default is barely better than nothing"
  claim survives, at a genuine 0.50.
- **The sweep priced policies no threshold could deliver.** It costed "block the
  top k", which equals "block everything at or above τ" only when no row outside
  the prefix shares the k-th score. **Isotonic calibration collapses 118,108
  scores to 156 distinct values**, so ties were everywhere and the curve's
  minimum disagreed with direct evaluation. The sweep now collapses tied scores;
  `optimal()` and `policy_at()` agree exactly.

That 156 is worth keeping in mind for Phase 5: it is how many positions the
threshold slider actually has. A finer slider would misrepresent the model's
resolution.

### The three-band policy needed two brakes

With reviewers catching *everything* they look at for one analyst fee, routing
most traffic to review buys near-perfect detection for pocket change and "wins"
against the single threshold on an artifact. `review_band` now takes a
`catch_rate` (0.70) and a queue cap (2% of traffic, with overflow falling
through to allow) — because an analyst team that can clear 65% of all
transactions does not exist.

### The scoring path is shared, deliberately

`src/cutline/bundle.py` holds the only `score()` in the project. Phase 3's
sweep, Phase 4's endpoint and Phase 5's dashboard all call it. A service that
loads the model and rebuilds features from a second copy of the logic returns
numbers that disagree with the metrics table — silently, and usually the night
before a demo.

## Phase 4 — the service

```bash
.venv/bin/uvicorn api.main:app --reload --port 8000
.venv/bin/python scripts/04_api_check.py     # every endpoint, in-process
```

| endpoint | what |
|---|---|
| `GET /health` | model loaded, what it was trained on, synthetic warning |
| `GET /policy` | thresholds in force + the constants behind them |
| `POST /score` | probability + decision, ~30 ms |
| `POST /explain` | the same plus three reasons in plain words |
| `POST /replay` | a batch, for the demo feed |

**The decision is not a fixed 0.5.** It is read from `cost_curve.json`, and
`04_api_check.py` asserts the threshold is not exactly 0.5 — if the curve fails
to load, the service silently reverts to the default and undoes the entire
argument of the project. That assert is the guard.

A real response:

```
POST /explain -> review  p=0.0338
  - placed at 05:00                              (+0.533)
  - no device or browser information             (+0.219)
  - risk pattern learned for this specific card   (+0.112)
```

### Serving needed history, and got it without a second code path

Velocity features look backward over a card's transactions; a single incoming
request carries no history. `HistoryStore` keeps recent transactions in memory
and, for a new one, runs **the training function** over just that card's rows.
That is safe rather than approximate: `add_history_features` computes row-local
values plus per-card values grouped by `card1`, so a row's features depend only
on itself and earlier rows of the same card. Restricting to that card returns
bit-identical output. Production would swap the store for a feature store; the
feature code would not change.

### The parity claim is now tested, not argued

Batch scoring runs `add_history_features` over the whole frame; serving rebuilds
one row's history from the store. Different code paths, and the promise is they
agree exactly. `smoke_test.py` now takes a card with real history, seeds a store
with every earlier row, scores the target both ways and asserts agreement to
1e-9. A silent mismatch here means the dashboard disagrees with the metrics
table — the worst thing to find out while someone is watching.

### A sparse request scores high, and that is correct

A three-field `curl` returns `block` at p=0.19 — the highest score in a typical
session, from the *least* information. Absent columns become NaN, LightGBM
routes NaN down its default branch, and `card1_is_first_seen` fires. The model
is not broken: missing history is genuinely predictive in this dataset. But an
unlabelled high score on a hand-written request reads as broken, so the response
carries `sparse`, `fields_absent` and `history_seen`. **Demo through `/replay`
with history warmed, never one cold request.**

### What SHAP is actually explaining

SHAP explains the **raw** model output, not the calibrated probability. Isotonic
sits downstream and is monotone, so the order and direction of reasons carry
over exactly — the magnitudes are log-odds, not probability. Saying "this added
0.4 to the probability" would be false; the endpoint returns that caveat in the
payload rather than leaving it to the reader.

### Phrasing rules that took two attempts

- **`TransactionAmt` and `log_amt` are the same fact told twice.** The first
  version listed "amount is 249.99" as two separate reasons, which reads as a
  bug. Reasons are deduplicated by rendered text, keeping the stronger one.
- **`card1 = 4564.0` is not an explanation.** IEEE-CIS never says what `card1`,
  `C13` or `D15 `mean, so the phrasing does not pretend: "risk pattern learned
  for this specific card", and anonymised counters are labelled as anonymised.
  Inventing a meaning would be worse than admitting there isn't one.
- Requests are sparse. `FeatureBuilder` records its fit-time source columns and
  reinstates absent ones as NaN, because in this dataset a missing column is
  missing **data**, not a schema error — `has_identity` exists for exactly that
  reason. A genuine feature-set mismatch still raises.

## Metrics, and why these ones

- **PR-AUC** is the headline. At a 3.5% positive rate, ROC-AUC reads ~0.95 while
  precision sits near 8%, because the huge negative class flatters it. ROC-AUC
  is reported alongside only so that gap stays visible.
- **PR-AUC lift** over the base rate, because raw PR-AUC is not comparable
  across sets with different prevalence.
- **Recall by value, not just by count.** Catching half of fraudulent
  transactions is a modelling result; catching half of fraudulent dollars is
  what a merchant feels.
- **ECE and Brier**, because a model can rank perfectly and still be useless
  for pricing: if it says 0.30 for a bucket that defaults at 0.05, every figure
  in `cost(τ)` is wrong while PR-AUC looks fine.

## Layout

```
src/cutline/
  config.py      paths + the cost constants, stated as assumptions
  split.py       time-ordered splitter and the leakage demo — read this first
  data.py        CSV -> parquet, identity join, dtype downcasting
  features.py    history features (unfitted) + FeatureBuilder (fitted on train)
  costs.py       cost(tau), the sweep, sensitivity, three-band policy
  bundle.py      load + score — the ONE scoring path, shared by every phase
  explain.py     SHAP -> sentences an analyst would write
  serving.py     HistoryStore + Scorer (threshold read from the cost curve)
  metrics.py     PR-AUC, value-recall, the results-table appender
  synthetic.py   stand-in frame for running without the download
scripts/
  download_data.py  train files only
  prepare_data.py   CSV -> parquet, once
  01_baseline.py    Phase 1: the floor and the leakage demo
  02_model.py       Phase 2: LightGBM + isotonic calibration
  03_cost_model.py  Phase 3: threshold sweep, sensitivity, the money sentence
  04_api_check.py   Phase 4: exercises every endpoint in-process
api/
  main.py           FastAPI service
  smoke_test.py     ingestion round trip on synthetic CSVs, in a temp dir
reports/
  results.csv       every experiment, appended
  calibration.png   reliability, raw vs calibrated
  cost_curve.png    cost vs threshold, with tau* marked
  cost_curve.json   thinned curve for the Phase 5 slider
  sensitivity.csv   how far tau* moves when the constants are wrong
models/
  cutline.joblib    encoder + model + calibrator, versioned together
```

The bundle is deliberately one file holding three objects. If Phase 4 loads a
model and rebuilds features from a second copy of the logic, it ships a service
whose scores disagree with the metrics table. `scripts/smoke_test.py` asserts
the round trip returns byte-identical scores.

## Currency

`TransactionAmt` is USD — IEEE-CIS is Vesta data. The cost model stays in USD so
units never mix; convert once at presentation time and show the rate you used
(`config.USD_TO_INR`).
