"""Latent-volatility state space model, fitted by Kalman filter.

WHY THIS AND NOT ANOTHER REGRESSION. Every model tried so far treats the
observed realized volatility as the thing to predict. It is not: it is a
NOISY MEASUREMENT of a smoother latent process. The project's own diagnostic
shows how much noise -- the same unchanged forecast scores R-squared 0.322
against the next single day and 0.405 against the next 5-day average, so a
large part of the apparent error is in the target rather than the model. A
practitioner estimate in the state-space literature puts latent log
volatility moving about 0.07 per day while the observation is off by about
0.48 from measurement noise alone.

HAR cannot express that distinction. It regresses one noisy number on
averages of other noisy numbers, and its weekly and monthly terms are doing
noise reduction implicitly, by averaging. A state space model does it
explicitly:

    observation   log RV_t = mu_t + eps_t        eps ~ N(0, R)
    state         mu_t - c = phi (mu_{t-1} - c) + eta_t   eta ~ N(0, Q)

mu_t is the latent log volatility; R is how badly a single day's range
measures it; Q is how fast the truth actually moves. The Kalman filter
estimates mu_t optimally given everything seen so far, and the signal-to-
noise ratio Q/R decides how much to trust today's observation versus the
accumulated history. HAR fixes that trade-off with hard-coded 1/5/22-day
windows; here it is estimated from the data.

This is the daily-data analogue of HARQ, which corrects HAR for measurement
error using realized quarticity. Quarticity needs intraday returns, which
this project deliberately does not use, so the error variance is inferred
from the series instead.

Run:  python -m quant.statespace [--tickers 20]
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.volatility import MIN_TRAIN, REFIT_EVERY, RESULTS, mse, qlike, qlike_series

REFIT_SS = 63          # the filter is the expensive part; the sweep showed
                       # refit frequency barely matters (0.00121 vs 0.02826)


def _kalman_nll(theta: np.ndarray, y: np.ndarray) -> float:
    """Negative log likelihood of the local AR(1)-plus-noise model."""
    c, logit_phi, log_q, log_r = theta
    phi = 1.0 / (1.0 + np.exp(-logit_phi))          # keep |phi| < 1
    q, r = np.exp(log_q), np.exp(log_r)
    if not np.isfinite([c, phi, q, r]).all() or q <= 0 or r <= 0:
        return 1e10

    x = 0.0                                          # state, centred on c
    p = q / max(1.0 - phi ** 2, 1e-6)                # stationary variance
    nll = 0.0
    for t in range(len(y)):
        x_pred = phi * x
        p_pred = phi * phi * p + q
        v = (y[t] - c) - x_pred                      # innovation
        f = p_pred + r
        if f <= 0:
            return 1e10
        nll += 0.5 * (np.log(2 * np.pi * f) + v * v / f)
        k = p_pred / f                               # Kalman gain
        x = x_pred + k * v
        p = (1.0 - k) * p_pred
    return float(nll) if np.isfinite(nll) else 1e10


def fit_kalman(y: np.ndarray) -> dict:
    """ML estimate of (c, phi, Q, R) on log realized volatility."""
    c0 = float(np.mean(y))
    v0 = float(np.var(y)) or 1e-4
    start = np.array([c0, 2.0, np.log(v0 * 0.2), np.log(v0 * 0.8)])
    res = minimize(_kalman_nll, start, args=(y,), method="Nelder-Mead",
                   options={"maxiter": 1200, "xatol": 1e-5, "fatol": 1e-5})
    c, logit_phi, log_q, log_r = res.x
    phi = 1.0 / (1.0 + np.exp(-logit_phi))
    return {"c": float(c), "phi": float(phi),
            "q": float(np.exp(log_q)), "r": float(np.exp(log_r)),
            "converged": bool(res.success), "nll": float(res.fun)}


def filter_state(y: np.ndarray, par: dict) -> tuple:
    """Run the filter forward; return the final state and its variance."""
    phi, q, r, c = par["phi"], par["q"], par["r"], par["c"]
    x, p = 0.0, q / max(1.0 - phi ** 2, 1e-6)
    for t in range(len(y)):
        x_pred, p_pred = phi * x, phi * phi * p + q
        f = p_pred + r
        k = p_pred / f
        x = x_pred + k * ((y[t] - c) - x_pred)
        p = (1.0 - k) * p_pred
    return x, p


def forecast_next(y: np.ndarray, par: dict) -> float:
    """One-step forecast of the OBSERVED series, on the original scale.

    The state forecast is for log latent volatility. Exponentiating needs
    the usual lognormal correction, and the relevant variance is the state
    uncertainty PLUS the measurement noise, because what will be observed
    tomorrow is the latent value plus a fresh measurement error.
    """
    x, p = filter_state(y, par)
    x_next = par["phi"] * x
    var_next = par["phi"] ** 2 * p + par["q"] + par["r"]
    return float(np.exp(par["c"] + x_next + 0.5 * var_next))


def walk_forward_ss(frame: pd.DataFrame, min_train: int = MIN_TRAIN,
                    refit_every: int = REFIT_SS, lookback: int = 750,
                    verbose: bool = True) -> np.ndarray:
    """Per-ticker walk-forward forecasts from the state space model."""
    preds = np.full(len(frame), np.nan)
    for n, (ticker, g) in enumerate(frame.groupby("ticker"), 1):
        g = g.sort_values("date")
        idx = g.index.to_numpy()
        rv = g["naive"].to_numpy(float)
        y = np.log(np.clip(rv, 1e-8, None))
        if len(g) <= min_train:
            continue
        par = None
        for i in range(min_train, len(g)):
            lo = max(0, i - lookback)
            if par is None or (i - min_train) % refit_every == 0:
                par = fit_kalman(y[lo:i])
            preds[idx[i]] = forecast_next(y[lo:i], par)
        if verbose and n % 10 == 0:
            print(f"    {n} tickers done", flush=True)
    return preds


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="garman_klass")
    ap.add_argument("--tickers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from quant.mcs import model_confidence_set
    from quant.vol_models import FEATURE_SETS, build_features, walk_forward

    frame = build_features(args.estimator, verbose=True)
    if args.tickers:
        keep = sorted(frame["ticker"].unique())[: args.tickers]
        frame = frame[frame["ticker"].isin(keep)].reset_index(drop=True)
    print(f"  {len(frame):,} rows, {frame['ticker'].nunique()} tickers")

    cols = {"naive": frame["naive"].to_numpy(float)}
    for m in ("har", "har_ovn"):
        print(f"    base {m} ...", flush=True)
        cols[m] = walk_forward(frame, m)
    print("    state space (Kalman ML per refit) ...", flush=True)
    cols["kalman_ss"] = walk_forward_ss(frame)
    cols["har_plus_ss"] = np.nanmean(
        np.column_stack([cols["har_ovn"], cols["kalman_ss"]]), axis=1)

    sc = frame.copy()
    for k, v in cols.items():
        sc[k] = v
    models = list(cols)
    sc = sc.dropna(subset=models).reset_index(drop=True)
    a = sc["target"].to_numpy(float)

    def r2(y, yh):
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

    summary = {m: {"qlike": qlike(a, sc[m].to_numpy(float)),
                   "mse": mse(a, sc[m].to_numpy(float)),
                   "r2": float(r2(a, sc[m].to_numpy(float))),
                   "corr": float(np.corrcoef(a, sc[m].to_numpy(float))[0, 1])}
               for m in models}
    mcs = model_confidence_set({m: qlike_series(a, sc[m].to_numpy(float))
                                for m in models}, verbose=True)

    print(f"\n{'=' * 62}\nSTATE SPACE vs HAR   {len(sc):,} forecasts, "
          f"{sc['ticker'].nunique()} tickers\n{'=' * 62}")
    print(f"  {'model':<14}{'QLIKE':>10}{'R2':>8}{'corr':>8}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
        tag = "  <- MCS" if m in mcs["mcs"] else ""
        print(f"  {m:<14}{v['qlike']:>10.5f}{v['r2']:>8.3f}{v['corr']:>8.3f}{tag}")
    print(f"\n  MCS (95%): {mcs['mcs']}")

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "n_forecasts": int(len(sc)),
           "n_tickers": int(sc["ticker"].nunique()),
           "summary": summary, "mcs": mcs}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_statespace.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
