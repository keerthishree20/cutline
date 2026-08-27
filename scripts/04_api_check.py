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
from cutline import features, split, synthetic  # noqa: E402


def main() -> None:
    frame = synthetic.make_frame(4_000)
    drop = ["isFraud"]
    sample = frame.drop(columns=drop).tail(40)

    with TestClient(app) as client:
        health = client.get("/health").json()
        print("GET /health ->", health)
        assert health["ok"]

        pol = client.get("/policy").json()
        print(f"\nGET /policy -> tau_block={pol['tau_block']:.4f} "
              f"tau_review={pol['tau_review']:.4f} currency={pol['currency']}")
        print(f"  curve: {pol['curve']}")
        assert pol["tau_block"] != 0.5, (
            "threshold is exactly 0.5 — the cost curve did not load, which "
            "silently undoes the entire argument of the project"
        )

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
        print(f"\nminimal payload (3 fields) -> {rm.json()['decision']} "
              f"p={rm.json()['probability']:.4f}")
        for reason in rm.json()["reasons"]:
            print(f"   - {reason['reason']}")

        bad = client.post("/score", json={"TransactionAmt": 10.0})
        assert bad.status_code == 422, f"expected validation error, got {bad.status_code}"
        print(f"\nmissing required fields -> {bad.status_code} (validation), as intended")

    print("\nAPI check PASSED.")


def _row_to_payload(row) -> dict:
    import numpy as np
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and np.isnan(v):
            continue
        if v is None:
            continue
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = str(v) if not isinstance(v, (int, float, str)) else v
    for k in ("TransactionID", "TransactionDT", "card1", "has_identity"):
        if k in out:
            out[k] = int(out[k])
    return out


if __name__ == "__main__":
    main()
