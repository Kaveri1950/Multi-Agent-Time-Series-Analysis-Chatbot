"""Why was this window flagged?

Two questions that must not be confused, so they are two functions:

  explain_window()  WHAT is unusual about it -- purely a statement about the
                    numbers, identical in every domain. "The flux dipped 4.2
                    standard deviations below its usual level" is a fact.

  likely_cause()    WHAT IT MIGHT MEAN -- domain vocabulary, and necessarily
                    a guess. "Possibly a transiting object" is an
                    interpretation, and it is only available because a human
                    told this module that the series came from a telescope.

Keeping them apart matters for the generalisation claim. The detectors and
their reasons are domain-neutral; only the final interpretive sentence is
not. If the reasons themselves needed domain knowledge, the shared pipeline
would be a fiction.

Everything here is rule-based and derived from already-computed features --
no LLM is involved in producing a reason. The same principle the rest of
this project follows: an explanation the model can be held to, rather than
prose generated after the fact that may or may not match what happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from anomaly.core import window_features

#: Plain-language names, with the direction that makes a high value notable.
FEATURE_PHRASES = {
    "dev_below": ("dipped below its local level", "below"),
    "dev_above": ("spiked above its local level", "above"),
    "std": ("was more variable than usual", "spread"),
    "range": ("covered an unusually wide peak-to-trough range", "spread"),
    "max_abs_diff": ("contained an abrupt single-step jump", "jump"),
    "mean_abs_diff": ("moved in larger steps than usual throughout", "jump"),
    "skew": ("was lopsided rather than symmetric", "shape"),
    "kurtosis": ("was sharply peaked, with heavy tails", "shape"),
    "median_crossings": ("oscillated around its own median unusually often", "shape"),
    "mean": ("sat at an unusual level", "level"),
    "median": ("sat at an unusual level", "level"),
    "min": ("reached an unusually low value", "below"),
    "max": ("reached an unusually high value", "above"),
}

#: Below this many robust standard deviations a feature is not worth
#: mentioning -- listing every feature would bury the one that mattered.
NOTABLE_SIGMA = 2.0


def feature_deviations(windows: np.ndarray, index: int | None = None) -> pd.DataFrame:
    """How far each window's features sit from the population, in robust sigmas.

    Uses median and MAD rather than mean and standard deviation for the same
    reason the Hampel detector does: the population being compared against
    CONTAINS the anomalies, so a mean-and-sigma baseline is dragged towards
    them and the anomalies partially hide themselves.
    """
    feats = window_features(windows)
    med = feats.median()
    mad = (feats - med).abs().median() * 1.4826
    mad = mad.replace(0, np.nan).fillna(feats.std().replace(0, 1.0)).replace(0, 1.0)
    z = (feats - med) / mad
    return z if index is None else z.iloc[[index]]


def explain_window(deviation_row: pd.Series, top_k: int = 3) -> list[str]:
    """Domain-neutral reasons, strongest first.

    Returns statements of fact about the numbers. Nothing here knows or
    cares what instrument produced the series.
    """
    scored = sorted(
        ((f, float(v)) for f, v in deviation_row.items() if f in FEATURE_PHRASES),
        key=lambda kv: -abs(kv[1]),
    )
    reasons = []
    for feature, value in scored[:top_k]:
        if abs(value) < NOTABLE_SIGMA:
            continue
        phrase, _ = FEATURE_PHRASES[feature]
        # A large NEGATIVE deviation is also notable, and reads differently.
        if value < 0:
            phrase = {
                "dev_below": "was unusually flat on the downside",
                "dev_above": "was unusually flat on the upside",
                "std": "was unusually quiet",
                "range": "covered an unusually narrow range",
                "max_abs_diff": "moved unusually smoothly",
                "mean_abs_diff": "moved in unusually small steps",
                "median_crossings": "barely crossed its own median -- a one-way excursion",
            }.get(feature, phrase)
        reasons.append(f"{phrase} ({abs(value):.1f}σ, `{feature}`)")
    if not reasons:
        reasons.append("no single feature stands out -- flagged on its overall "
                       "statistical profile rather than one dominant signal")
    return reasons


def likely_cause(deviation_row: pd.Series, domain: str) -> tuple[str, str]:
    """Hedged interpretation. The one place domain vocabulary is allowed.

    Deliberately cautious. A flagged window is a candidate, never a
    classification -- a single dip cannot distinguish a planet from an
    eclipsing binary, and a single volatile day cannot distinguish news from
    noise. Overclaiming here would undo the care taken everywhere else.
    """
    below = float(deviation_row.get("dev_below", 0.0))
    above = float(deviation_row.get("dev_above", 0.0))
    jump = float(deviation_row.get("max_abs_diff", 0.0))
    spread = float(deviation_row.get("std", 0.0))
    shape = abs(float(deviation_row.get("skew", 0.0)))
    crossings = float(deviation_row.get("median_crossings", 0.0))

    if domain == "kepler":
        if below > NOTABLE_SIGMA and below > above:
            # A one-way excursion (few median crossings) is the cleaner
            # transit signature, but requiring it as a precondition made this
            # branch nearly unreachable -- every detection came back
            # "stellar flare". It belongs in the description, not the test.
            shape_note = (" The window crosses its own median unusually rarely, which is the "
                          "clean one-way shape a transit makes."
                          if crossings < -NOTABLE_SIGMA else "")
            return ("Possible transit-like dip",
                    "A localised drop in brightness, the shape a transiting object produces."
                    + shape_note +
                    " It cannot be called a planet on this evidence: an eclipsing binary "
                    "produces the same signature, and confirmation needs repeated dips at a "
                    "consistent period, depth and duration.")
        if above > NOTABLE_SIGMA and above > below:
            return ("Possible stellar flare",
                    "A brief brightening rather than a dimming. Consistent with a stellar "
                    "flare, or with a cosmic-ray hit on the detector -- which is why single "
                    "upward excursions are usually treated as instrumental until repeated.")
        if spread > NOTABLE_SIGMA and max(below, above) < NOTABLE_SIGMA:
            return ("Possible stellar variability or instrumental noise",
                    "Elevated scatter without a clean excursion in either direction. Typical "
                    "of starspot-driven variability or a noisy segment of the light curve, "
                    "rather than a discrete event.")
        return ("Unclear -- statistical outlier only",
                "Flagged on its combined statistical profile without one dominant feature. "
                "Worth looking at the raw window before drawing any conclusion.")

    if domain == "equity":
        if jump > NOTABLE_SIGMA and jump > spread:
            return ("Possible news or event shock",
                    "A single abrupt move rather than a sustained change in conditions -- the "
                    "signature of information arriving at a point in time. Whether it was "
                    "scheduled news, a surprise, or a liquidity event cannot be told from "
                    "price alone.")
        if spread > NOTABLE_SIGMA and jump < NOTABLE_SIGMA:
            return ("Possible volatility regime shift",
                    "Elevated variability across the whole window without one dominant move. "
                    "More consistent with a change in market conditions than with a single "
                    "event, though the two often arrive together.")
        if max(below, above) > NOTABLE_SIGMA:
            direction = "drawdown" if below > above else "rally"
            return (f"Possible one-way {direction}",
                    f"A sustained {direction} rather than a single jump. Trend-like moves of "
                    f"this kind are frequently ordinary; the label calendar is the only thing "
                    f"here that can say whether anything actually happened.")
        return ("Unclear -- statistical outlier only",
                "No single feature dominates. Flagged on overall profile, which on market "
                "data is frequently just an unremarkable stretch that happens to sit in the "
                "tail of the score distribution.")

    return ("Unclear", "No domain interpretation available for this series.")


def suppress_overlapping(scores: np.ndarray, top_n: int, window: int, stride: int) -> list[int]:
    """Highest-scoring windows, keeping only one per distinct event.

    Windows overlap heavily -- at width 24 and stride 6, four windows cover
    any given point -- so a single bright event produces a run of four or
    five consecutive high scores. Taking the raw top-N then lists the same
    event five times and hides the next five real ones.

    This is ordinary non-maximum suppression: walk down the ranking and skip
    anything that overlaps a window already kept. The chart solves the same
    problem by merging contiguous spans; this is the table's version, and
    they should agree about how many distinct things were found.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    span = max(1, int(np.ceil(window / max(stride, 1))))
    kept: list[int] = []
    for idx in np.argsort(-scores, kind="stable"):
        if all(abs(int(idx) - k) >= span for k in kept):
            kept.append(int(idx))
        if len(kept) >= top_n:
            break
    return kept


def annotate_flagged(windows: np.ndarray, scores: np.ndarray, domain: str,
                     y_true: np.ndarray | None = None, top_n: int = 20,
                     window: int = 1, stride: int = 1) -> pd.DataFrame:
    """The top-scoring distinct events, with reasons and whether they were right.

    `y_true` turns this from a list of assertions into something checkable --
    a flagged window that no label supports is a false positive, and showing
    those beside the true ones is the difference between a demonstration and
    a sales pitch.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    order = suppress_overlapping(scores, top_n, window, stride)
    deviations = feature_deviations(windows)

    records = []
    for rank, idx in enumerate(order, start=1):
        dev = deviations.iloc[idx]
        title, detail = likely_cause(dev, domain)
        row = {
            "rank": rank,
            "window": int(idx),
            "start_index": int(idx) * stride,
            "score": round(float(scores[idx]), 4),
            "likely_cause": title,
            "reasons": explain_window(dev),
            "explanation": detail,
        }
        if y_true is not None:
            row["label"] = int(y_true[idx])
            row["verdict"] = "true positive" if y_true[idx] else "false positive"
        records.append(row)
    return pd.DataFrame(records)
