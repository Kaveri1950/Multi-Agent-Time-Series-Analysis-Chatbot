"""Adaptive Conformal Inference for the forecast intervals (Gibbs & Candes 2021).

WHAT THIS CHANGES. The GP and ARIMA intervals in the forecast log are
calibrated -- 82.5% and 83.1% coverage of a nominal 80% -- but they are
calibrated BY LUCK OF A CORRECT MODEL. Both assume a parametric error
distribution; if that assumption drifts, coverage drifts with it and nothing
notices. A conformal interval is calibrated BY CONSTRUCTION, from the
model's own recent errors, whatever distribution they happen to have.

THE EXCHANGEABILITY PROBLEM. Plain split conformal guarantees coverage only
when the data are exchangeable, which a price series is not: volatility
clusters, so the errors of a calm week do not predict the errors of a crash.
Adaptive Conformal Inference handles this by updating the working
miscoverage level online after every observation,

    alpha_{t+1} = alpha_t + gamma * (alpha_target - 1{Y_t outside interval})

so a run of misses widens the interval and a run of hits narrows it. The
guarantee is a long-run coverage FREQUENCY and holds with no assumption on
the data-generating process, even under arbitrary distribution shift.

WIDTH IS REPORTED ALONGSIDE COVERAGE, ALWAYS. A wider interval trivially
covers more; an interval that matches the target coverage at a smaller width
is the only kind that is actually better. Reporting coverage on its own is
how a useless interval passes review.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_ALPHA = 0.20        # 80% interval, matching what the log already stores
GAMMA = 0.02               # ACI step size; larger adapts faster but is noisier
WARMUP = 15                # observations needed before the first interval


def aci_intervals(pred: np.ndarray, actual: np.ndarray,
                  alpha_target: float = TARGET_ALPHA, gamma: float = GAMMA,
                  warmup: int = WARMUP) -> pd.DataFrame:
    """Online conformal intervals for one series, strictly causal.

    At each step the interval is built from the absolute residuals seen so
    far -- never including the point being predicted -- so nothing about the
    outcome enters its own interval.
    """
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    n = len(pred)
    scores: list[float] = []
    alpha_t = alpha_target
    rows = []

    for t in range(n):
        if len(scores) < warmup:
            lo = hi = np.nan
            covered = np.nan
        else:
            # Working level is clipped to (0,1): a long miss streak can push
            # alpha_t negative, which would ask for an infinitely wide band.
            a = float(np.clip(alpha_t, 1e-3, 0.999))
            q = float(np.quantile(scores, 1.0 - a, method="higher"))
            lo, hi = pred[t] - q, pred[t] + q
            covered = float(lo <= actual[t] <= hi)
            alpha_t = alpha_t + gamma * (alpha_target - (1.0 - covered))

        rows.append({"pred": pred[t], "actual": actual[t], "lower": lo, "upper": hi,
                     "covered": covered, "alpha_t": alpha_t,
                     "width": (hi - lo) if np.isfinite(hi) else np.nan})
        scores.append(abs(actual[t] - pred[t]))

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str = "") -> dict:
    d = df.dropna(subset=["covered"])
    if d.empty:
        return {"label": label, "n": 0}
    return {"label": label, "n": int(len(d)),
            "coverage": float(d["covered"].mean()),
            "mean_width": float(d["width"].mean()),
            "median_width": float(d["width"].median()),
            "target": 1.0 - TARGET_ALPHA}


__all__ = ["aci_intervals", "summarise", "TARGET_ALPHA", "GAMMA", "WARMUP"]
