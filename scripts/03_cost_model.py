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
    parts = split.time_ordered_split(features.add_history_features(df))
    return parts


def main() -> None:
    b = bundle_mod.load()
    source = b.get("source", "synthetic")
    print(bundle_mod.describe(b))
    if source == "synthetic":
        print("!! bundle was trained on SYNTHETIC data — every figure below is\n"
              "!! shape, not magnitude. Do not put these numbers in a deck.\n")

    parts = rebuild_test_set(source)
    # The regeneration above must reproduce Phase 2's split exactly. Without
    # this assert, changing the row count in 02_model.py silently scores a
    # different test set here and nothing complains.
    recorded = b.get("boundaries", {})
    for name, got in (("train_end_dt", parts.train_end_dt),
                      ("calib_end_dt", parts.calib_end_dt)):
        want = recorded.get(name)
        if want is not None and want != got:
            raise SystemExit(
                f"split mismatch: bundle recorded {name}={want}, regenerated {got}.\n"
                f"The bundle and this script disagree about the test set. "
                f"Re-run scripts/02_model.py."
            )
    test = parts.test
    p = bundle_mod.score(b, test)
    y = test["isFraud"].to_numpy()
    amounts = test["TransactionAmt"].to_numpy(dtype=float)

    c = costs.CostConstants()
    print(f"cost constants (all {config.CURRENCY}, all assumptions):")
    print(c.render())

    curve = costs.sweep(y, p, amounts, c)
    best = costs.optimal(curve)
    shipped = costs.constrained_optimal(curve, config.MAX_DECLINE_RATE)
    nothing = costs.do_nothing_cost(y, amounts, c)
    # Exact thresholds, not the nearest score present. Snapping to a sample
    # would price "block two transactions" and label it the 0.5 default.
    default = costs.policy_at(y, p, amounts, 0.50, c)
    paranoid = costs.policy_at(y, p, amounts, 0.05, c)

    print(f"\nisotonic collapses {len(p):,} scores to {curve['tau'].nunique():,} "
          f"distinct values,")
    print("so that is how many threshold positions the Phase 5 slider actually has.")
    print("A finer slider would be a lie about the model's resolution.")

    print(f"\ntest set: {len(y):,} transactions, {int(y.sum()):,} fraudulent "
          f"({y.mean():.2%}), {amounts.sum():,.0f} {config.CURRENCY} total value")

    rows = [
        ("approve everything", np.nan, nothing, 0.0, 0.0),
        ("tau = 0.50 (default)", default["tau"], default["cost"],
         default["fraud_value_caught"], default["good_declined_rate"]),
        ("tau* (pure cost min)", float(best["tau"]), float(best["cost"]),
         float(best["fraud_value_caught"]), float(best["good_declined_rate"])),
        (f"tau_ship (<= {config.MAX_DECLINE_RATE:.0%} declines)", float(shipped["tau"]),
         float(shipped["cost"]), float(shipped["fraud_value_caught"]),
         float(shipped["good_declined_rate"])),
        ("tau = 0.05 (paranoid)", paranoid["tau"], paranoid["cost"],
         paranoid["fraud_value_caught"], paranoid["good_declined_rate"]),
    ]
    print(f"\n{'policy':<24}{'tau':>8}{'cost':>14}{'fraud caught':>15}{'good declined':>15}")
    print("-" * 76)
    for label, tau, cost, caught, declined in rows:
        tau_s = "  —" if np.isnan(tau) else f"{tau:.4f}"
        print(f"{label:<24}{tau_s:>8}{cost:>14,.0f}{caught:>14.1%}{declined:>15.2%}")

    saved = nothing - float(shipped["cost"])
    per_10k = saved / len(y) * 10_000
    print("\n" + "=" * 76)
    print(f"RECOMMENDED POLICY  tau = {shipped['tau']:.4f}")
    print(f"We catch {shipped['fraud_value_caught']:.1%} of fraud BY VALUE while declining")
    print(f"{shipped['good_declined_rate']:.2%} of good customers — inside the "
          f"{config.MAX_DECLINE_RATE:.0%} ceiling the business sets —")
    print(f"saving {saved:,.0f} {config.CURRENCY} on this test set "
          f"({per_10k:,.0f} {config.CURRENCY} per 10,000 transactions).")
    print(f"In INR at {config.USD_TO_INR:.0f}/USD that is "
          f"Rs {per_10k * config.USD_TO_INR:,.0f} per 10,000 — conversion shown once,")
    print("the model itself never leaves the dataset's own currency.")
    print("=" * 76)
    print(f"\nNote the default: tau=0.50 costs {default['cost']:,.0f}, barely better than")
    print(f"the {nothing:,.0f} of doing nothing at all. That is the argument.")

    gap = float(shipped["cost"]) - float(best["cost"])
    print(f"\nThe ceiling costs {gap:,.0f} {config.CURRENCY} against pure cost "
          f"minimisation ({gap / max(float(best['cost']), 1e-9):+.1%}), and takes")
    print(f"declines from {best['good_declined_rate']:.2%} to "
          f"{shipped['good_declined_rate']:.2%} while giving up "
          f"{best['fraud_value_caught'] - shipped['fraud_value_caught']:.1%} of fraud value.")
    print("That trade is the business's to make, not the model's, which is why both")
    print("optima are reported instead of one being chosen silently.")

    # The ceiling is a dial, and its setting changes the answer a lot. Show the
    # dial rather than making the reader edit config and re-run to find out.
    print(f"\nwhere the ceiling could sit:")
    print(f"  {'ceiling':>9}{'tau':>9}{'cost':>12}{'fraud caught':>15}{'declined':>11}")
    print("  " + "-" * 54)
    for ceiling in (0.005, 0.01, 0.02, 0.05, 1.0):
        row = costs.constrained_optimal(curve, ceiling)
        label = "none" if ceiling >= 1.0 else f"{ceiling:.1%}"
        marker = "  <- current" if abs(ceiling - config.MAX_DECLINE_RATE) < 1e-9 else ""
        print(f"  {label:>9}{row['tau']:>9.4f}{row['cost']:>12,.0f}"
              f"{row['fraud_value_caught']:>14.1%}{row['good_declined_rate']:>11.2%}{marker}")
    print(f"  Set it in config.MAX_DECLINE_RATE. A tighter ceiling buys goodwill")
    print("  with fraud losses; the table is what that exchange rate actually is.")

    decline = float(best["good_declined_rate"])
    if decline > config.MAX_DECLINE_RATE:
        print(f"\nNote: pure cost minimisation would decline {decline:.2%} of good")
        print(f"customers, above the {config.MAX_DECLINE_RATE:.0%} ceiling — which is "
              f"why tau_ship exists and")
        print("is the policy recommended above. Churn is already priced into the")
        print("constants; without it the unconstrained optimum blocks harder still.")
        if decline > 0.20:
            print("At this rate the classifier is also genuinely weak, and a better")
            print("model would move the optimum more than either the ceiling or the")
            print("churn constant does.")

    # ---- three bands ----
    cap = int(0.02 * len(y))  # an analyst queue is ~2% of traffic, not 65%
    band = costs.review_band(y, p, amounts, tau_block=float(shipped["tau"]),
                             tau_review=float(shipped["tau"]) / 3.0, c=c,
                             catch_rate=0.70, max_reviews=cap)
    print(f"\nthree bands (block >= {band['tau_block']:.4f}, "
          f"review >= {band['tau_review']:.4f}, catch rate {band['catch_rate']:.0%}, "
          f"queue cap {cap:,}):")
    print(f"  blocked {band['blocked']:,}  reviewed {band['reviewed']:,}  "
          f"(overflowed to allow: {band['review_overflow']:,})  "
          f"allowed {band['allowed']:,}")
    print(f"  cost {band['cost']:,.0f}  vs {shipped['cost']:,.0f} for the single threshold")
    print("  Reviewers catch 70%, not everything, and the queue is capped at 2% of")
    print("  traffic. Without both, routing most traffic to review buys near-perfect")
    print("  detection for pocket change and the band policy 'wins' on an artifact.")

    # ---- sensitivity ----
    sens = costs.sensitivity(y, p, amounts, c)
    print("\nsensitivity — does tau* survive the constants being wrong?")
    print(sens[["variant", "tau_star", "cost", "fraud_value_caught",
                "good_declined_rate"]].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))
    base_tau = float(sens.loc[sens["variant"] == "baseline", "tau_star"].iloc[0])
    print("\n  Which constant actually governs tau*:")
    for _, r in sens[sens["variant"] != "baseline"].iterrows():
        ratio = r["tau_star"] / max(base_tau, 1e-9)
        arrow = "up" if ratio > 1.05 else ("down" if ratio < 0.95 else "flat")
        print(f"    {r['variant']:<18} tau* {arrow:<5} {ratio:>5.2f}x")
    # Derive the claim from the table rather than asserting it. Hardcoding
    # "the dispute fee barely matters" would be a sentence that silently
    # becomes false on a different dataset or a different model.
    moves = sens[sens["variant"] != "baseline"].assign(
        ratio=lambda d: (d["tau_star"] / max(base_tau, 1e-9)).clip(lower=1e-9)
    )
    moves["shift"] = np.abs(np.log(moves["ratio"]))
    biggest = moves.sort_values("shift", ascending=False).iloc[0]
    smallest = moves.sort_values("shift").iloc[0]
    print(f"  Biggest mover: {biggest['variant']} ({biggest['ratio']:.2f}x). "
          f"Smallest: {smallest['variant']} ({smallest['ratio']:.2f}x).")
    print("  Name whichever constant dominates when you present — 'tau* is governed")
    print("  by X' is a far better line than quoting a range, and the table above")
    print("  tells you which X without you having to guess.")

    lo, hi = float(sens["tau_star"].min()), float(sens["tau_star"].max())
    print(f"\n  tau* ranges {lo:.4f} - {hi:.4f} ({hi / max(lo, 1e-9):.1f}x) across variants.")
    if hi / max(lo, 1e-9) > 2:
        print("  That is a WIDE range: the constants are load-bearing. Lead with this")
        print("  rather than letting a judge find it — the honest framing is that the")
        print("  method is sound and the constants need a real merchant's numbers.")
    else:
        print("  Narrow: the assumed constants are not load-bearing, which is the")
        print("  strongest possible answer to \'you made those numbers up\'.")

    _write_outputs(curve, sens, best, shipped, nothing, source)


def _write_outputs(curve, sens, best, shipped, nothing, source: str) -> None:
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
        # What the service actually uses. Pure cost minimisation is reported
        # for honesty; it is not what anyone would ship.
        "tau_recommended": float(shipped["tau"]),
        "cost_at_tau_recommended": float(shipped["cost"]),
        "max_decline_rate": config.MAX_DECLINE_RATE,
        "decline_rate_at_recommended": float(shipped["good_declined_rate"]),
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
    ax.axvline(float(shipped["tau"]), color="#0E6B70", lw=1.2, ls="-.")
    ax.annotate(f"ship {shipped['tau']:.3f}", xy=(float(shipped["tau"]), float(shipped["cost"])),
                xytext=(8, -22), textcoords="offset points", fontsize=9, color="#0E6B70")
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
