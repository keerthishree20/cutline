"""Cutline scoring service.

    .venv/bin/uvicorn api.main:app --reload --port 8000

Endpoints
    GET  /health    is the model loaded, and what was it trained on
    GET  /policy    the thresholds in force and the constants behind them
    POST /score     probability + decision for one transaction
    POST /explain   the same, plus the three strongest reasons in plain words
    POST /replay    a batch, for the demo feed

The decision is NOT a fixed 0.5. It comes from the cost curve written by
scripts/03_cost_model.py, which is the point of the whole project — see
/policy for the threshold in force and the assumptions that produced it.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import serving  # noqa: E402

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["scorer"] = serving.Scorer()
    STATE["history"] = serving.HistoryStore()
    print(f"loaded: {STATE['scorer'].bundle.get('source')} bundle, "
          f"tau_block={STATE['scorer'].tau_block:.4f}")
    yield
    STATE.clear()


app = FastAPI(
    title="Cutline",
    description="Payment risk scoring whose threshold comes from cost, not accuracy.",
    version="0.4.0",
    lifespan=lifespan,
)

# The Next.js dashboard in Phase 5 runs on another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    """Only amount, time and card are required; the rest degrade gracefully.

    Missing values are signal in this dataset, not an error — `has_identity`
    exists precisely because most transactions carry no device information.
    """

    TransactionID: int | None = None
    TransactionDT: int = Field(..., description="seconds offset, NOT a unix timestamp")
    TransactionAmt: float
    card1: int
    ProductCD: str | None = None
    card2: float | None = None
    card3: float | None = None
    card4: str | None = None
    card5: float | None = None
    card6: str | None = None
    addr1: float | None = None
    dist1: float | None = None
    P_emaildomain: str | None = None
    R_emaildomain: str | None = None
    C1: float | None = None
    C13: float | None = None
    C14: float | None = None
    D1: float | None = None
    D15: float | None = None
    M4: str | None = None
    DeviceType: str | None = None
    DeviceInfo: str | None = None
    id_01: float | None = None
    id_02: float | None = None
    has_identity: int | None = None

    def to_row(self) -> dict:
        row = self.model_dump()
        if row.get("has_identity") is None:
            row["has_identity"] = int(row.get("DeviceType") is not None)
        return row


def _scorer() -> serving.Scorer:
    s = STATE.get("scorer")
    if s is None:
        raise HTTPException(503, "model not loaded")
    return s


@app.get("/health")
def health() -> dict:
    s = STATE.get("scorer")
    return {
        "ok": s is not None,
        "trained_on": None if s is None else s.bundle.get("source"),
        "history_rows": len(STATE.get("history", [])) if STATE.get("history") else 0,
        "warning": (
            None if s is None or s.bundle.get("source") != "synthetic"
            else "model trained on synthetic data — scores are shape, not magnitude"
        ),
    }


@app.get("/policy")
def policy() -> dict:
    return _scorer().policy()


def _serve_report(name: str, hint: str) -> dict:
    """Return a generated report file verbatim.

    Verbatim matters. Re-deriving any of this in the API would create a second
    source for numbers the deck already quotes, and the two would drift. A
    missing file is a 503 with instructions, never a synthesised default — a
    dashboard silently showing tau=0.5 is precisely the failure the whole
    project argues against.
    """
    from cutline import config

    path = config.REPORTS / name
    if not path.exists():
        raise HTTPException(503, f"{name} not generated — run {hint}")
    return json.loads(path.read_text())


@app.get("/curve")
def curve() -> dict:
    """The cost curve. The dashboard slider reads this and never recomputes."""
    return _serve_report("cost_curve.json", "scripts/03_cost_model.py")


@app.get("/queue")
def queue() -> dict:
    """Scored transactions for the review queue, built by scripts/05_demo_queue.py."""
    return _serve_report("demo_queue.json", "scripts/05_demo_queue.py")


def _score_one(txn: Transaction, explain: bool) -> dict:
    scorer, history = _scorer(), STATE["history"]
    started = time.perf_counter()

    featurised = history.add(txn.to_row())
    p = float(scorer.score(featurised)[0])
    decision = scorer.decide(p)

    # A sparse request scores high for a real reason — missing history is
    # predictive in this dataset — but an unlabelled high score on a 3-field
    # curl reads as a broken model. Say how much was missing.
    builder = scorer.bundle["feature_builder"]
    absent = builder.missing_source_values(featurised)
    supplied = len(builder.source_columns_) - len(absent)
    result = {
        "transaction_id": txn.TransactionID,
        "probability": p,
        "decision": decision,
        "tau_block": scorer.tau_block,
        "tau_review": scorer.tau_review,
        "amount": txn.TransactionAmt,
        "fields_absent": len(absent),
        "fields_supplied": supplied,
        "sparse": supplied <= len(builder.source_columns_) // 2,
        "history_seen": int(featurised.get("card1_count_24h", pd.Series([0])).iloc[0] or 0),
    }

    if explain:
        X = scorer.features_for(featurised)
        result["reasons"] = scorer.explainer.top_reasons(X, k=3)[0]
        result["explanation_note"] = (
            "SHAP explains the raw model output. Isotonic calibration sits "
            "downstream and is monotone, so the order and direction of these "
            "reasons hold exactly; the magnitudes are log-odds, not probability."
        )

    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


@app.post("/score")
def score(txn: Transaction) -> dict:
    return _score_one(txn, explain=False)


@app.post("/explain")
def explain(txn: Transaction) -> dict:
    return _score_one(txn, explain=True)


@app.post("/replay")
def replay(batch: list[Transaction]) -> dict:
    """Score a batch in order, as the demo feed does."""
    if len(batch) > 2000:
        raise HTTPException(413, "batch too large; send 2000 or fewer")
    scored = [_score_one(t, explain=False) for t in batch]
    counts: dict[str, int] = {}
    for r in scored:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    return {"n": len(scored), "decisions": counts, "results": scored}
