#!/usr/bin/env python
"""Exercise every endpoint in-process, so a broken API fails here not on stage.

Uses FastAPI's TestClient — no server, no port, no sleep-and-hope.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from cutline import config, data, synthetic  # noqa: E402


def main() -> None:
    # Post the same kind of rows the model was trained on. Feeding synthetic
    # transactions to a model fitted on IEEE-CIS exercises the plumbing but
    # tells you nothing about whether the scores are sane.
    if config.TRAIN_PARQUET.exists():
        frame = data.load().tail(400)
        origin = "ieee-cis"
    else:
        frame = synthetic.make_frame(4_000)
        origin = "synthetic"
    sample = frame.drop(columns=["isFraud"]).tail(40)
    print(f"posting {origin} rows\n")

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("GET /health ->", health)
        assert health["ok"]
        assert health["trained_on"] == origin, (
            f"model was trained on {health['trained_on']} but this check is "
            f"posting {origin} rows — the scores below would be meaningless"
        )

        pol = client.get("/policy").json()
        print(f"\nGET /policy -> tau_block={pol['tau_block']:.4f} "
              f"tau_review={pol['tau_review']:.4f} currency={pol['currency']}")
        print(f"  curve: {pol['curve']}")
        assert pol["tau_block"] != 0.5, (
            "threshold is exactly 0.5 — the cost curve did not load, which "
            "silently undoes the entire argument of the project"
        )

        cv = client.get("/curve")
        assert cv.status_code == 200, cv.text
        cvj = cv.json()
        print(f"\nGET /curve -> {len(cvj['points'])} points, "
              f"tau*={cvj['tau_star']:.4f}, ship={cvj['tau_recommended']:.4f}")
        assert abs(cvj["tau_recommended"] - pol["tau_block"]) < 1e-12, (
            "the curve's recommended threshold and the one the service uses "
            "disagree — two sources of truth for the same number"
        )

        q = client.get("/queue")
        assert q.status_code == 200, q.text
        qj = q.json()
        print(f"GET /queue -> {qj['n']} transactions, "
              f"{sum(i['is_fraud'] for i in qj['items'])} fraudulent")
        assert qj["source"] == origin, "queue and model were built from different data"

        payload = _row_to_payload(sample.iloc[0])
        r = client.post("/score", json=payload)
        assert r.status_code == 200, r.text
        print(f"\nPOST /score -> {r.json()['decision']} "
              f"p={r.json()['probability']:.4f} in {r.json()['latency_ms']}ms")

        r = client.post("/explain", json=payload).json()
        print(f"\nPOST /explain -> {r['decision']} p={r['probability']:.4f}")
        for reason in r["reasons"]:
            print(f"   - {reason['reason']}   (contribution {reason['contribution']:+.3f})")
        if not r["reasons"]:
            print("   (nothing pushed toward fraud on this transaction)")

        batch = [_row_to_payload(sample.iloc[i]) for i in range(1, 25)]
        rb = client.post("/replay", json=batch).json()
        print(f"\nPOST /replay -> {rb['n']} scored, decisions: {rb['decisions']}")

        # Degrade gracefully: only the three required fields.
        minimal = {"TransactionDT": int(sample.iloc[0]["TransactionDT"]) + 60,
                   "TransactionAmt": 249.99, "card1": int(sample.iloc[0]["card1"])}
        rm = client.post("/explain", json=minimal)
        assert rm.status_code == 200, rm.text
        j = rm.json()
        print(f"\nminimal payload (3 fields) -> {j['decision']} p={j['probability']:.4f} "
              f"[sparse={j['sparse']}, {j['fields_supplied']} of "
              f"{j['fields_supplied'] + j['fields_absent']} fields supplied, "
              f"history={j['history_seen']}]")
        for reason in j["reasons"]:
            print(f"   - {reason['reason']}")
        assert j["sparse"], "a 3-field request should be flagged sparse"
        print(f"   ^ this is a SPARSE request — {j['fields_supplied']} of "
              f"{j['fields_supplied'] + j['fields_absent']} fields. Whichever way it")
        print("     scores, it is not evidence about the model: demo through /replay")
        print("     with history warmed, never one cold hand-written curl.")

        bad = client.post("/score", json={"TransactionAmt": 10.0})
        assert bad.status_code == 422, f"expected validation error, got {bad.status_code}"
        print(f"\nmissing required fields -> {bad.status_code} (validation), as intended")

    print("\nAPI check PASSED.")


def _row_to_payload(row) -> dict:
    """Drop nulls and coerce numpy scalars to JSON-safe Python types.

    `isinstance(v, float)` is not enough: np.float32 is not a Python float, so
    NaNs in real float32 columns sail straight through into the request body
    and json.dumps rejects them. pd.isna handles every null flavour here —
    NaN, NaT and pd.NA alike.
    """
    import numpy as np
    import pandas as pd

    out = {}
    for k, v in row.items():
        if v is None or (np.ndim(v) == 0 and pd.isna(v)):
            continue
        if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, str):
            out[k] = v
        else:
            out[k] = str(v)

    for k in ("TransactionID", "TransactionDT", "card1", "has_identity"):
        if k in out:
            out[k] = int(out[k])
    return out


if __name__ == "__main__":
    main()
