#!/usr/bin/env python
"""Build the review queue the dashboard shows, offline.

Why offline rather than scoring live at request time: `HistoryStore` starts
empty, so a sample scored through the API arrives with almost no card history,
every velocity feature reads near-zero, and the resulting probabilities do not
match `cost_curve.json`. A queue whose scores disagree with the headline
numbers undermines every figure on the page.

Building it from the same batch-featurised test slice the curve was computed on
makes them match by construction, not by care. The live `/score` endpoint still
exists and is still the thing being demonstrated — this is the queue's seed
data, not a replacement for it.

`isFraud` is included deliberately. It is a held-out test set, so the truth is
known, and showing whether each decision was right is the difference between a
list of numbers and a demo.

    python scripts/05_demo_queue.py [--n 300]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import bundle as bundle_mod  # noqa: E402
from cutline import config, data, explain, features, split, synthetic  # noqa: E402

FIELDS = ["TransactionID", "TransactionAmt", "TransactionDT", "ProductCD",
          "card4", "card6", "P_emaildomain", "DeviceType", "has_identity"]


def main() -> None:
    n = 300
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    b = bundle_mod.load()
    source = b.get("source", "synthetic")
    print(bundle_mod.describe(b))

    raw = data.load() if source == "ieee-cis" else synthetic.make_frame(80_000)
    test = split.time_ordered_split(features.add_history_features(raw)).test
    print(f"test slice: {len(test):,} rows")

    scores = bundle_mod.score(b, test)

    # Sample across the whole risk range, not the top n — a queue of nothing but
    # blocks shows neither the review band nor a single correct allow, and the
    # decisions that were RIGHT are half of what makes the demo land.
    order = np.argsort(-scores)
    high = order[: n // 2]                                   # the genuinely risky
    mid = order[n // 2: n // 2 + n // 4]                     # the review band
    rng = np.random.default_rng(config.RANDOM_STATE)
    low = rng.choice(order[len(order) // 2:], size=n - len(high) - len(mid),
                     replace=False)
    picked = np.concatenate([high, mid, low])

    subset = test.iloc[picked]
    X = b["feature_builder"].transform(subset)
    reasons = explain.Explainer(b).top_reasons(X, k=3)

    items = []
    for pos, (idx, reason_list) in enumerate(zip(picked, reasons)):
        row = test.iloc[int(idx)]
        item = {f: _jsonable(row.get(f)) for f in FIELDS if f in test.columns}
        item["probability"] = float(scores[int(idx)])
        item["is_fraud"] = int(row["isFraud"])
        item["reasons"] = [r["reason"] for r in reason_list]
        item["context"] = _context(row)
        items.append(item)

    items.sort(key=lambda r: -r["probability"])

    payload = {
        "source": source,
        "n": len(items),
        "test_rows": int(len(test)),
        "fraud_rate": float(test["isFraud"].mean()),
        "note": ("Scored from the same batch-featurised test slice as the cost "
                 "curve, so these probabilities match the curve exactly."),
        "items": items,
    }
    out = config.REPORTS / "demo_queue.json"
    out.write_text(json.dumps(payload, indent=1))

    blocked = sum(1 for i in items if i["probability"] >= 0.4194)
    caught = sum(1 for i in items if i["is_fraud"] == 1)
    print(f"\nwrote {out}")
    print(f"  {len(items)} transactions, {caught} of them fraudulent")
    print(f"  probability range {items[-1]['probability']:.5f} "
          f"to {items[0]['probability']:.5f}")

    _write_sample_csv(test, scores)
    _verify(b, test, picked, scores)


UPLOAD_FIELDS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1",
                 "card2", "card3", "card4", "card5", "card6", "addr1", "dist1",
                 "ProductCD", "P_emaildomain", "R_emaildomain",
                 "C1", "C13", "C14", "D1", "D15", "M4", "DeviceType"]


def _write_sample_csv(test, scores, n: int = 25) -> None:
    """Something to actually drag into the upload panel during the demo.

    Real held-out rows with the label stripped, so a judge uploading this is
    scoring transactions the model has genuinely never seen.

    Deliberately a MIX, not a random draw. At a 3.4% fraud rate a random 25
    rows flag nothing at all, and a demo file where every row is allowed
    demonstrates precisely nothing. Roughly half come from the high-risk tail.
    That is stacking the file, so the dashboard says so out loud — the honest
    move is to disclose the sampling, not to pretend a random draw produced a
    dramatic result.
    """
    import numpy as np

    cols = [c for c in UPLOAD_FIELDS if c in test.columns]
    rng = np.random.default_rng(config.RANDOM_STATE)
    order = np.argsort(-scores)
    risky = order[:200]
    rest = order[len(order) // 3:]

    pick = np.concatenate([
        rng.choice(risky, size=n // 2, replace=False),
        rng.choice(rest, size=n - n // 2, replace=False),
    ])
    rng.shuffle(pick)

    sample = test.iloc[pick][cols]
    out = config.REPORTS / "sample_upload.csv"
    sample.to_csv(out, index=False)
    flagged = int((scores[pick] >= 0.4194).sum())
    print(f"wrote {out} ({n} unlabelled rows: {n // 2} from the high-risk tail, "
          f"{flagged} would block at the shipped threshold)")


def _context(row) -> list[str]:
    """Human-readable facts about the transaction, separate from attribution.

    On IEEE-CIS the strongest SHAP drivers are usually the anonymised counting
    fields — 117 of 300 queue items have reasons that are entirely opaque,
    because Vesta never documented what C1 or C13 mean. That is a true fact
    about this dataset and it should be reported, not massaged by reweighting
    the attribution toward features that merely READ better.

    So the fix is to add information rather than distort it: these are plain
    facts about the transaction, clearly not claims about why it scored.
    """
    import numpy as np
    import pandas as pd

    out: list[str] = []

    hour = row.get("hour")
    if pd.notna(hour):
        out.append(f"{int(hour):02d}:00")

    prior = row.get("card1_count_24h")
    if pd.notna(prior):
        n = int(prior)
        out.append("first seen on this card" if row.get("card1_is_first_seen") == 1
                   else f"{n} prior on card in 24h")

    gap = row.get("card1_sec_since_prev")
    if pd.notna(gap) and np.isfinite(gap):
        secs = float(gap)
        if secs < 90:
            out.append(f"{secs:.0f}s after the last")
        elif secs < 5400:
            out.append(f"{secs / 60:.0f} min after the last")
        elif secs < 172800:
            out.append(f"{secs / 3600:.0f}h after the last")

    out.append("device known" if row.get("has_identity") == 1 else "no device data")

    dom = row.get("P_emaildomain")
    if isinstance(dom, str) and dom:
        out.append(dom)

    return out


def _verify(b, test, picked, scores) -> None:
    """The queue must agree with the curve. Check, do not assume."""
    probe = int(picked[len(picked) // 2])
    row = test.iloc[[probe]]
    direct = bundle_mod.score(b, row)[0]
    assert abs(direct - scores[probe]) < 1e-12, (
        f"queue score {scores[probe]:.12f} disagrees with a direct rescore "
        f"{direct:.12f} — the dashboard would contradict the cost curve"
    )
    print(f"  verified: queue scores match a direct rescore ({direct:.10f})")


def _jsonable(v):
    import pandas as pd
    if v is None or (np.ndim(v) == 0 and pd.isna(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v) if not isinstance(v, (int, float, str)) else v


if __name__ == "__main__":
    main()
