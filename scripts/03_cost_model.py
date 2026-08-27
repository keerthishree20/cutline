#!/usr/bin/env python
"""Phase 3 — the cost model. This is the part that wins the track.

Every team ships a classifier. Far fewer can answer "why did you decline at
0.14 and not 0.5?" with anything but a shrug. The answer is that a missed
chargeback and a wrongly declined sale carry different costs, so the
cost-minimising threshold is nowhere near the default.

Outputs, all of which belong in the deck:
  reports/cost_curve.png    the U, with tau* marked
  reports/cost_curve.json   the curve the Phase 5 slider reads (it does not
                            recompute — dragging changes the threshold, never
                            the model)
  reports/sensitivity.csv   how far tau* moves when the constants are wrong

    python scripts/03_cost_model.py [--synthetic]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import bundle as bundle_mod  # noqa: E402
from cutline import config, costs, data, features, split, synthetic  # noqa: E402


def rebuild_test_set(source: str):
    """Reproduce exactly the test slice Phase 2 held out."""
    if source == "synthetic":
        df = synthetic.make_frame(80_000)
    else:
        df = data.load()
    return split.time_ordered_split(features.add_history_features(df)).test


def main() -> None:
    b = bundle_mod.load()
    source = b.get("source", "synthetic")
    print(bundle_mod.describe(b))
    if source == "synthetic":
        print("!! bundle was trained on SYNTHETIC data — every figure below is\n"
              "!! shape, not magnitude. Do not put these numbers in a deck.\n")

    test = rebuild_test_set(source)
    p = bundle_mod.score(b, test)
    y = test["isFraud"].to_numpy()
    amounts = test["TransactionAmt"].to_numpy(dtype=float)

    c = costs.CostConstants()
    print(f"cost constants (all {config.CURRENCY}, all assumptions):")
    print(c.render())

    curve = costs.sweep(y, p, amounts, c)
    best = costs.optimal(curve)
    nothing = costs.do_nothing_cost(y, amounts, c)
    default = costs.at_threshold(curve, 0.50)
    paranoid = costs.at_threshold(curve, 0.05)

    print(f"\ntest set: {len(y):,} transactions, {int(y.sum()):,} fraudulent "
          f"({y.mean():.2%}), {amounts.sum():,.0f} {config.CURRENCY} total value")

    rows = [
        ("approve everything", np.nan, nothing, 0.0, 0.0),
        ("tau = 0.50 (default)", float(default["tau"]), float(default["cost"]),
         float(default["fraud_value_caught"]), float(default["good_declined_rate"])),
        ("tau* (cost-optimal)", float(best["tau"]), float(best["cost"]),
         float(best["fraud_value_caught"]), float(best["good_declined_rate"])),
        ("tau = 0.05 (paranoid)", float(paranoid["tau"]), float(paranoid["cost"]),
         float(paranoid["fraud_value_caught"]), float(paranoid["good_declined_rate"])),
    ]
    print(f"\n{'policy':<24}{'tau':>8}{'cost':>14}{'fraud caught':>15}{'good declined':>15}")
    print("-" * 76)
    for label, tau, cost, caught, declined in rows:
        tau_s = "  —" if np.isnan(tau) else f"{tau:.4f}"
        print(f"{label:<24}{tau_s:>8}{cost:>14,.0f}{caught:>14.1%}{declined:>15.2%}")

    saved = nothing - float(best["cost"])
    per_10k = saved / len(y) * 10_000
    print("\n" + "=" * 76)
    print(f"At tau* = {best['tau']:.4f} we catch {best['fraud_value_caught']:.1%} of fraud")
    print(f"BY VALUE while declining {best['good_declined_rate']:.2%} of good customers,")
    print(f"saving {saved:,.0f} {config.CURRENCY} on this test set "
          f"({per_10k:,.0f} {config.CURRENCY} per 10,000 transactions).")
    print(f"In INR at {config.USD_TO_INR:.0f}/USD that is "
          f"Rs {per_10k * config.USD_TO_INR:,.0f} per 10,000 — conversion shown once,")
    print("the model itself never leaves the dataset's own currency.")
    print("=" * 76)
    print(f"\nNote the default: tau=0.50 costs {default['cost']:,.0f}, barely better than")
    print(f"the {nothing:,.0f} of doing nothing at all. That is the argument.")

    if float(best["good_declined_rate"]) > 0.05:
        print(f"\n!! tau* declines {best['good_declined_rate']:.1%} of good customers. No")
        print("!! real merchant would accept that, and the cost model is not wrong —")
        print("!! it is telling you the MODEL is weak. When the classifier cannot")
        print("!! separate, blocking indiscriminately really is the cheaper policy")
        print("!! under these constants. The fix is a better model, or a decline-rate")
        print("!! ceiling as a business constraint on top of the cost minimum.")

    # ---- three bands ----
    band = costs.review_band(y, p, amounts, tau_block=float(best["tau"]),
                             tau_review=float(best["tau"]) / 3.0, c=c)
    print(f"\nthree bands (block >= {band['tau_block']:.4f}, "
          f"review >= {band['tau_review']:.4f}):")
    print(f"  blocked {band['blocked']:,}  reviewed {band['reviewed']:,}  "
          f"allowed {band['allowed']:,}   cost {band['cost']:,.0f}")
    print("  (assumes review catches everything it looks at — optimistic; say so)")

    # ---- sensitivity ----
    sens = costs.sensitivity(y, p, amounts, c)
    print("\nsensitivity — does tau* survive the constants being wrong?")
    print(sens[["variant", "tau_star", "cost", "fraud_value_caught",
                "good_declined_rate"]].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))
    lo, hi = float(sens["tau_star"].min()), float(sens["tau_star"].max())
    print(f"\n  tau* ranges {lo:.4f} - {hi:.4f} ({hi / max(lo, 1e-9):.1f}x) across variants.")
    if hi / max(lo, 1e-9) > 2:
        print("  That is a WIDE range: the constants are load-bearing. Lead with this")
        print("  rather than letting a judge find it — the honest framing is that the")
        print("  method is sound and the constants need a real merchant's numbers.")
    else:
        print("  Narrow: the assumed constants are not load-bearing, which is the")
        print("  strongest possible answer to \'you made those numbers up\'.")

    _write_outputs(curve, sens, best, nothing, source)


def _write_outputs(curve, sens, best, nothing, source: str) -> None:
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    sens.to_csv(config.REPORTS / "sensitivity.csv", index=False)

    # Thin the curve for the frontend: 400 points is plenty for a slider and
    # keeps the payload small. The slider reads this, it never recomputes.
    step = max(1, len(curve) // 400)
    thin = curve.iloc[::step].copy()
    payload = {
        "source": source,
        "tau_star": float(best["tau"]),
        "cost_at_tau_star": float(best["cost"]),
        "cost_do_nothing": float(nothing),
        "currency": config.CURRENCY,
        "points": thin.round(6).to_dict(orient="records"),
    }
    (config.REPORTS / "cost_curve.json").write_text(json.dumps(payload, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(curve["tau"], curve["cost"], color="#0E6B70", lw=1.8)
    ax.axvline(float(best["tau"]), color="#A8402F", lw=1.2, ls="--")
    ax.axhline(nothing, color="#869596", lw=1, ls=":")
    ax.annotate(f"τ* = {best['tau']:.3f}", xy=(float(best["tau"]), float(best["cost"])),
                xytext=(8, 18), textcoords="offset points", fontsize=9, color="#A8402F")
    ax.annotate("approve everything", xy=(0.55, nothing), xytext=(0, 5),
                textcoords="offset points", fontsize=8, color="#869596")
    ax.set_xlabel("threshold τ")
    ax.set_ylabel(f"total cost ({config.CURRENCY})")
    ax.set_title(f"Cost is U-shaped in τ — {source}", fontsize=11)
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.REPORTS / "cost_curve.png", dpi=140)

    print(f"\nwrote {config.REPORTS / 'cost_curve.png'}")
    print(f"wrote {config.REPORTS / 'cost_curve.json'}  (the Phase 5 slider reads this)")
    print(f"wrote {config.REPORTS / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
