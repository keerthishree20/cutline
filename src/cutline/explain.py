"""Turn SHAP numbers into sentences a fraud analyst would actually write.

An explanation panel showing `card1_amt_z = 3.2` explains nothing. It has to say
"amount is 3.2x this card's usual", and the phrasing has to be true — which is
why `MIN_PRIOR_FOR_Z` exists in features.py: before that fix, that sentence was
a lie for cards with little history.

ONE HONESTY NOTE, and it belongs on the slide too. SHAP explains the *raw* model
output, not the calibrated probability. Isotonic sits downstream and is monotone,
so the direction and the ranking of reasons carry over exactly; the magnitudes
are in the model's log-odds space, not in probability. Saying "this feature added
0.4 to the probability" would be false. Saying "this was the strongest reason,
pushing toward fraud" is true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _money(v: float) -> str:
    return f"{v:,.2f}"


def _duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "never"
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.0f} hours ago"
    return f"{seconds / 86400:.0f} days ago"


# value -> phrase. Written from the analyst's side of the screen: what happened,
# not which column it lives in.
PHRASES = {
    "TransactionAmt": lambda v: f"amount is {_money(v)}",
    "log_amt": lambda v: f"amount is {_money(np.expm1(v))}",
    "amt_cents": lambda v: (
        "amount ends in a round number of cents" if v == 0
        else f"amount ends in an odd {v:.0f} cents"
    ),
    "amt_is_round": lambda v: (
        "amount is a whole number" if v >= 0.5 else "amount has cents"
    ),
    "hour": lambda v: f"placed at {int(v):02d}:00",
    "dayofweek": lambda v: f"placed on day {int(v)} of the week",
    "has_identity": lambda v: (
        "no device or browser information" if v < 0.5 else "device information present"
    ),
    "card1_amt_z": lambda v: (
        f"amount is {abs(v):.1f}x this card's usual spread "
        f"({'above' if v > 0 else 'below'} its recent average)"
    ),
    "card1_count_1h": lambda v: f"{int(v)} prior transactions on this card in the last hour",
    "card1_count_24h": lambda v: f"{int(v)} prior transactions on this card in 24 hours",
    "card1_sec_since_prev": lambda v: f"previous transaction on this card {_duration(v)}",
    "card1_is_first_seen": lambda v: (
        "first transaction ever seen on this card" if v >= 0.5
        else "this card has prior history"
    ),
    "card1_freq": lambda v: (
        "card never seen during training" if v == 0
        else f"card seen {int(v)} times during training"
    ),
    "addr1_freq": lambda v: (
        "billing region never seen during training" if v == 0
        else f"billing region seen {int(v)} times during training"
    ),
    "P_emaildomain_freq": lambda v: (
        "email domain never seen during training" if v == 0
        else f"email domain seen {int(v)} times during training"
    ),
    "email_mismatch": lambda v: (
        "payer and recipient email domains differ" if v >= 0.5
        else "payer and recipient email domains match"
    ),
    # Anonymised columns. IEEE-CIS does not say what card1 or C13 mean, so the
    # phrasing must not pretend otherwise — "card1 = 4564" is not an
    # explanation, and neither is inventing a meaning for it. Name the kind of
    # thing it is and be explicit that the model learned a pattern.
    "card1": lambda v: "risk pattern learned for this specific card",
    "card2": lambda v: "risk pattern learned for this card group",
    "card3": lambda v: "risk pattern learned for this card's issuing country code",
    "card5": lambda v: "risk pattern learned for this card's issuer band",
    "addr1": lambda v: "risk pattern learned for this billing region",
    "dist1": lambda v: f"billing-to-shipping distance field is {v:,.0f}",
    "C1": lambda v: f"counting field C1 is {v:,.0f} (anonymised in this dataset)",
    "C13": lambda v: f"counting field C13 is {v:,.0f} (anonymised in this dataset)",
    "C14": lambda v: f"counting field C14 is {v:,.0f} (anonymised in this dataset)",
    "D1": lambda v: f"days-since field D1 is {v:,.0f} (anonymised in this dataset)",
    "D15": lambda v: f"days-since field D15 is {v:,.0f} (anonymised in this dataset)",
    "ProductCD": lambda v: f"product code {v}",
    "card4": lambda v: f"card network {v}",
    "card6": lambda v: f"{v} card",
    "DeviceType": lambda v: f"{v} device",
}


def phrase(feature: str, value) -> str:
    """Plain-language rendering of one feature's value."""
    if pd.isna(value):
        return f"{feature.replace('_', ' ')} is missing"
    fn = PHRASES.get(feature)
    if fn is None:
        return f"{feature.replace('_', ' ')} = {value}"
    try:
        return fn(value)
    except Exception:
        return f"{feature.replace('_', ' ')} = {value}"


class Explainer:
    """SHAP over the LightGBM booster, phrased.

    Built once and reused — constructing a TreeExplainer per request is the
    easiest way to make a fast endpoint slow.
    """

    def __init__(self, bundle: dict):
        import shap

        import warnings

        self.model = bundle["model"]
        self.columns = list(bundle["feature_columns"])
        with warnings.catch_warnings():
            # shap warns that LightGBM binary output is now a list of ndarray.
            # `contributions` already handles both shapes.
            warnings.simplefilter("ignore", UserWarning)
            self._explainer = shap.TreeExplainer(self.model)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_features) SHAP values toward the positive class."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            values = self._explainer.shap_values(X)
        if isinstance(values, list):          # older shap: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # (n, features, classes)
            values = values[:, :, -1]
        return values

    def top_reasons(self, X: pd.DataFrame, k: int = 3,
                    direction: str = "toward") -> list[list[dict]]:
        """The k strongest reasons per row, already phrased.

        `direction="toward"` returns what pushed the score up, which is what a
        review queue wants. Reasons pushing the other way are still available in
        `contributions` for a fuller panel.
        """
        shap_values = self.contributions(X)
        out: list[list[dict]] = []

        for i in range(len(X)):
            row = shap_values[i]
            order = np.argsort(-row) if direction == "toward" else np.argsort(np.abs(row))[::-1]
            reasons: list[dict] = []
            seen: set[str] = set()

            for j in order:
                if len(reasons) >= k:
                    break
                col = self.columns[j]
                if row[j] <= 0 and direction == "toward":
                    continue  # nothing genuinely pushing toward fraud
                text = phrase(col, X.iloc[i][col])
                # TransactionAmt and log_amt are the same fact told twice, and a
                # panel that lists it twice looks broken. Keep the stronger one.
                if text in seen:
                    continue
                seen.add(text)
                reasons.append({
                    "feature": col,
                    "value": _jsonable(X.iloc[i][col]),
                    "contribution": float(row[j]),
                    "reason": text,
                })
            out.append(reasons)
        return out


def _jsonable(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v) if not isinstance(v, (int, float, str, bool)) else v
