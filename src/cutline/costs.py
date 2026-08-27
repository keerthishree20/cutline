"""The cost model. This is the part of the project that is actually the point.

A classifier outputs a probability. Turning that probability into a decision is
an economics problem, and the constants below are where the economics live:

    missed fraud    the merchant shipped goods and lost the dispute
                    -> transaction amount + a fixed dispute fee
    false decline   a real customer was turned away
                    -> lost margin on the sale + a support contact
    manual review   the queue is not free
                    -> one analyst's time per item

Those are not equal — a missed chargeback here costs roughly an order of
magnitude more than a wrongly declined sale — so the cost-minimising threshold
sits nowhere near 0.5. Sweeping tau and reading off the minimum is the whole
argument.

Every constant is an ASSUMPTION and belongs on screen during the demo. A judge
arguing with a stated number is worth more than a hidden number that happens to
be right.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config


@dataclass(frozen=True)
class CostConstants:
    """All in the dataset's own currency. See config.CURRENCY."""

    dispute_fee: float = config.DISPUTE_FEE
    gross_margin: float = config.GROSS_MARGIN
    support_cost: float = config.SUPPORT_COST
    analyst_cost: float = config.ANALYST_COST

    def render(self) -> str:
        return (
            f"  dispute fee   {self.dispute_fee:>8.2f} {config.CURRENCY}  per missed chargeback\n"
            f"  gross margin  {self.gross_margin:>8.2%}      earned on a legitimate sale\n"
            f"  support cost  {self.support_cost:>8.2f} {config.CURRENCY}  per false decline\n"
            f"  analyst cost  {self.analyst_cost:>8.2f} {config.CURRENCY}  per manual review"
        )


def do_nothing_cost(y_true, amounts, c: CostConstants) -> float:
    """Approve everything — the reference every threshold is measured against.

    Without this number, "we save X" is meaningless. This is the X it saves
    against.
    """
    y = np.asarray(y_true)
    a = np.asarray(amounts, dtype=float)
    fraud = y == 1
    return float(a[fraud].sum() + c.dispute_fee * fraud.sum())


def sweep(y_true, y_score, amounts, c: CostConstants | None = None) -> pd.DataFrame:
    """Total cost at every distinct threshold, in one pass.

    Sort descending by score; blocking the top k transactions is then a prefix,
    so every quantity is a cumulative sum and the whole curve costs one sort
    rather than one pass per candidate threshold.
    """
    c = c or CostConstants()
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    a = np.asarray(amounts, dtype=float)

    order = np.argsort(-p, kind="mergesort")
    y, p, a = y[order], p[order], a[order]

    n = len(y)
    total_fraud_n = float(y.sum())
    total_fraud_value = float((y * a).sum())

    # Prefix sums over "the top k scores are blocked".
    caught_n = np.cumsum(y)
    caught_value = np.cumsum(y * a)
    fp_n = np.arange(1, n + 1) - caught_n
    fp_value = np.cumsum((1.0 - y) * a)

    missed_n = total_fraud_n - caught_n
    missed_value = total_fraud_value - caught_value

    fraud_cost = missed_value + c.dispute_fee * missed_n
    decline_cost = c.gross_margin * fp_value + c.support_cost * fp_n

    # Collapse tied scores. A prefix of length k is only an implementable
    # policy if no row outside it shares the k-th score — otherwise "block the
    # top k" is not the same as "block everything at or above tau", and the
    # curve reports costs no threshold can actually deliver. Isotonic
    # calibration produces heavily tied outputs, so this is the common case
    # here, not an edge case.
    keep = np.r_[p[:-1] != p[1:], True]

    # Include k=0 (block nothing) so the curve starts at the do-nothing cost.
    tau = np.concatenate([[1.0], p[keep]])
    total = np.concatenate([[total_fraud_value + c.dispute_fee * total_fraud_n],
                            (fraud_cost + decline_cost)[keep]])
    caught_value_frac = np.concatenate([[0.0], (caught_value / max(total_fraud_value, 1e-9))[keep]])
    caught_n_frac = np.concatenate([[0.0], (caught_n / max(total_fraud_n, 1e-9))[keep]])
    decline_rate = np.concatenate([[0.0], (fp_n / max(n - total_fraud_n, 1e-9))[keep]])
    block_rate = np.concatenate([[0.0], (np.arange(1, n + 1) / n)[keep]])

    return pd.DataFrame({
        "tau": tau,
        "cost": total,
        "fraud_value_caught": caught_value_frac,
        "fraud_count_caught": caught_n_frac,
        "good_declined_rate": decline_rate,
        "block_rate": block_rate,
    })


def optimal(curve: pd.DataFrame) -> pd.Series:
    """The bottom of the U."""
    return curve.loc[curve["cost"].idxmin()]


def at_threshold(curve: pd.DataFrame, tau: float) -> pd.Series:
    """The curve row closest to a given threshold.

    Use `policy_at` for a named policy such as "the 0.5 default". This snaps to
    the nearest score actually present, which on a set with no scores near 0.5
    silently prices blocking two transactions instead of the policy you meant.
    """
    return curve.iloc[int((curve["tau"] - tau).abs().idxmin())]


def policy_at(y_true, y_score, amounts, tau: float,
              c: CostConstants | None = None) -> dict:
    """Cost of blocking at EXACTLY this threshold, no snapping."""
    c = c or CostConstants()
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    a = np.asarray(amounts, dtype=float)

    blocked = p >= tau
    missed = (y == 1) & ~blocked
    false_decline = (y == 0) & blocked
    fraud_value = a[y == 1].sum()

    cost = (
        a[missed].sum() + c.dispute_fee * missed.sum()
        + c.gross_margin * a[false_decline].sum() + c.support_cost * false_decline.sum()
    )
    return {
        "tau": float(tau),
        "cost": float(cost),
        "fraud_value_caught": float(a[(y == 1) & blocked].sum() / max(fraud_value, 1e-9)),
        "good_declined_rate": float(false_decline.sum() / max((y == 0).sum(), 1e-9)),
        "block_rate": float(blocked.mean()),
    }


def sensitivity(y_true, y_score, amounts, base: CostConstants | None = None) -> pd.DataFrame:
    """Does tau* survive the constants being wrong?

    The honest answer to "you made those numbers up" is not a better guess, it
    is showing how far tau* moves when they change. If doubling the dispute fee
    barely shifts it, the made-up number was not load-bearing.
    """
    base = base or CostConstants()
    rows = []
    variants = {
        "baseline": base,
        "dispute fee x2": CostConstants(base.dispute_fee * 2, base.gross_margin,
                                        base.support_cost, base.analyst_cost),
        "dispute fee /2": CostConstants(base.dispute_fee / 2, base.gross_margin,
                                        base.support_cost, base.analyst_cost),
        "margin x2": CostConstants(base.dispute_fee, base.gross_margin * 2,
                                   base.support_cost, base.analyst_cost),
        "support cost x5": CostConstants(base.dispute_fee, base.gross_margin,
                                         base.support_cost * 5, base.analyst_cost),
    }
    for label, c in variants.items():
        best = optimal(sweep(y_true, y_score, amounts, c))
        rows.append({
            "variant": label,
            "tau_star": float(best["tau"]),
            "cost": float(best["cost"]),
            "fraud_value_caught": float(best["fraud_value_caught"]),
            "good_declined_rate": float(best["good_declined_rate"]),
            **asdict(c),
        })
    return pd.DataFrame(rows)


def review_band(y_true, y_score, amounts, tau_block: float, tau_review: float,
                c: CostConstants | None = None, catch_rate: float = 0.70,
                max_reviews: int | None = None) -> dict:
    """Three bands rather than two: block, review, allow.

    Two parameters exist to stop this looking better than it is. With a review
    that catches EVERYTHING it sees for one analyst fee, routing most of the
    traffic to review buys near-perfect detection for pocket change, and the
    band policy beats the single threshold for free — which is an artifact of
    the assumption, not a finding.

      catch_rate   share of fraud a reviewer actually catches (0.7 is generous)
      max_reviews  queue capacity. Beyond it, items overflow to ALLOW, because
                   an analyst team that can clear 65% of all traffic does not
                   exist. Overflow takes the lowest-scoring reviews first.
    """
    c = c or CostConstants()
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    a = np.asarray(amounts, dtype=float)

    blocked = p >= tau_block
    reviewed = (p >= tau_review) & ~blocked

    overflowed = 0
    if max_reviews is not None and reviewed.sum() > max_reviews:
        # Keep the highest-scoring reviews; the rest overflow to allow.
        candidates = np.flatnonzero(reviewed)
        keep = candidates[np.argsort(-p[candidates], kind="mergesort")][:max_reviews]
        overflowed = int(reviewed.sum() - len(keep))
        reviewed = np.zeros_like(reviewed)
        reviewed[keep] = True

    allowed = ~blocked & ~reviewed

    # Review catches only `catch_rate` of the fraud it sees; the rest slips
    # through and costs the same as fraud that was never looked at.
    reviewed_fraud_value = a[(y == 1) & reviewed].sum()
    reviewed_fraud_n = float(((y == 1) & reviewed).sum())
    missed_in_review_value = (1 - catch_rate) * reviewed_fraud_value
    missed_in_review_n = (1 - catch_rate) * reviewed_fraud_n

    missed = (y == 1) & allowed
    false_decline = (y == 0) & blocked

    cost = (
        a[missed].sum() + c.dispute_fee * missed.sum()
        + missed_in_review_value + c.dispute_fee * missed_in_review_n
        + c.gross_margin * a[false_decline].sum() + c.support_cost * false_decline.sum()
        + c.analyst_cost * reviewed.sum()
    )
    caught_value = a[(y == 1) & blocked].sum() + catch_rate * reviewed_fraud_value
    return {
        "tau_block": float(tau_block),
        "tau_review": float(tau_review),
        "catch_rate": catch_rate,
        "cost": float(cost),
        "blocked": int(blocked.sum()),
        "reviewed": int(reviewed.sum()),
        "review_overflow": overflowed,
        "allowed": int(allowed.sum()),
        "fraud_value_caught": float(caught_value / max(a[y == 1].sum(), 1e-9)),
        "good_declined_rate": float(false_decline.sum() / max((y == 0).sum(), 1e-9)),
    }
