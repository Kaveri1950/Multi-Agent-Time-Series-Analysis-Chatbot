"""GARCH and CARR: the two benchmarks a volatility viva will ask about.

WHY THESE TWO SPECIFICALLY.

GARCH(1,1) is the canonical conditional-variance model and its absence from
a volatility comparison is the first thing anyone will notice. GJR-GARCH adds
the asymmetry that HAR-L captures crudely -- a negative shock raises
tomorrow's variance more than a positive one of the same size -- so the pair
tests whether the leverage effect is better handled inside the variance
recursion than as a regression term.

CARR (Chou 2005) is the one built for the data this project actually has. It
models the daily RANGE directly with a GARCH-style recursion, and the
literature reports it beating return-based GARCH precisely because the range
carries intraday information that a close-to-close return discards. Given
that the whole volatility arm here rests on a range estimator, omitting the
model family designed around that choice would be a strange gap.

    GARCH(1,1)   sigma2_t = omega + alpha*eps2_{t-1} + beta*sigma2_{t-1}
    GJR-GARCH    sigma2_t = omega + (alpha + gamma*I[eps_{t-1}<0])*eps2_{t-1}
                            + beta*sigma2_{t-1}
    CARR         lambda_t = omega + alpha*R_{t-1} + beta*lambda_{t-1}

A NOTE ON COMPARING THEM WITH HAR. GARCH forecasts the variance of
close-to-close RETURNS, while the target here is a range-based realized
volatility. Scoring one against the other is standard practice -- both
estimate the same latent daily volatility, the range estimator simply less
noisily -- but it does mean GARCH is being judged on a proxy it was not
fitted to. CARR has no such mismatch, since it is fitted on the range
itself. Worth stating rather than hiding, because it is the kind of
asymmetry that quietly decides a benchmark.

CARR is estimated by exponential quasi-maximum likelihood,

    log L = -sum( log lambda_t + R_t / lambda_t ),

which is the standard QML choice for a positive-valued series and stays
consistent even when the true density is not exponential.

Run:  python -m quant.garch_carr [--tickers 48]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.volatility import MIN_TRAIN, REFIT_EVERY, RESULTS, TRADING_DAYS

#: arch works best on returns scaled to percent; variance is divided back out.
RET_SCALE = 100.0


# ------------------------------------------------------------------ CARR
def _carr_nll(theta: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return 1e10
    lam = np.empty(len(r))
    lam[0] = r.mean()
    for t in range(1, len(r)):
        lam[t] = omega + alpha * r[t - 1] + beta * lam[t - 1]
        if lam[t] <= 1e-12:
            return 1e10
    return float(np.sum(np.log(lam) + r / lam))


def fit_carr(r: np.ndarray) -> tuple:
    """Exponential QML fit. Returns (params, last lambda)."""
    m = float(np.mean(r))
    start = np.array([m * 0.05, 0.2, 0.75])
    res = minimize(_carr_nll, start, args=(r,), method="Nelder-Mead",
                   options={"maxiter": 800, "xatol": 1e-6, "fatol": 1e-6})
    omega, alpha, beta = res.x
    if not np.isfinite(res.fun) or res.fun >= 1e10:
        return None, m
    lam = m
    for t in range(1, len(r)):
        lam = omega + alpha * r[t - 1] + beta * lam
    return res.x, lam


def carr_forecast(r_hist: np.ndarray) -> float:
    """One-step-ahead range forecast from history ending at t-1.

    No rescaling twin is needed: CARR is fitted on the range itself, so it is
    already in the target's units. Measured ratio to the target is 1.03,
    against GARCH's 1.37.
    """
    params, lam = fit_carr(r_hist)
    if params is None:
        return float(np.mean(r_hist))
    omega, alpha, beta = params
    return float(omega + alpha * r_hist[-1] + beta * lam)


# ----------------------------------------------------------------- GARCH
#: Counts fits that fell back to a plain standard deviation. A silent
#: fallback is indistinguishable from a successful fit in the output, which
#: is how an invalid `reinit=` argument once made GARCH and GJR-GARCH return
#: byte-identical forecasts for every ticker without anything failing loudly.
FALLBACKS = {"garch": 0, "gjr_garch": 0}


def garch_forecast(ret_hist: np.ndarray, asymmetric: bool = False,
                   rv_hist: np.ndarray | None = None) -> tuple:
    """One-step-ahead annualised volatility. Returns (raw, rescaled).

    THE RESCALING IS NOT COSMETIC. GARCH is fitted on CLOSE-TO-CLOSE returns,
    which contain the overnight gap; the target here is a Parkinson range,
    measured inside the session and excluding that gap. They estimate related
    but not identical quantities, and measured on this panel GARCH runs about
    1.37x the target on average. QLIKE is a ratio loss, so a 37% level bias
    dominates it and the comparison would report a units mismatch as a
    modelling failure.

    The correction is a Mincer-Zarnowitz-style rescaling: the model's own
    IN-SAMPLE conditional volatility is compared with the realized volatility
    over the same training window, and the ratio is applied to the forecast.
    Both quantities come from training data only, so nothing about the
    forecast day enters its own correction. This is what the realized-
    volatility literature does when comparing models fitted on different
    proxies, precisely to separate scale from dynamics.
    """
    from arch import arch_model

    key = "gjr_garch" if asymmetric else "garch"
    y = ret_hist * RET_SCALE
    try:
        am = arch_model(y, vol="GARCH", p=1, o=1 if asymmetric else 0, q=1,
                        dist="normal", mean="Constant", rescale=False)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
        var_pct = float(res.forecast(horizon=1).variance.values[-1, 0])
        if not np.isfinite(var_pct) or var_pct <= 0:
            raise ValueError("non-positive variance forecast")
        cond = np.asarray(res.conditional_volatility, dtype=float)
    except Exception:                                   # noqa: BLE001
        FALLBACKS[key] += 1
        fb = float(np.std(ret_hist) * np.sqrt(TRADING_DAYS))
        return fb, fb

    raw = float(np.sqrt(max(var_pct / (RET_SCALE ** 2), 1e-12) * TRADING_DAYS))
    if rv_hist is None or len(rv_hist) == 0:
        return raw, raw
    cond_ann = cond / RET_SCALE * np.sqrt(TRADING_DAYS)
    cond_ann = cond_ann[np.isfinite(cond_ann) & (cond_ann > 0)]
    if len(cond_ann) == 0:
        return raw, raw
    scale = float(np.mean(rv_hist)) / float(np.mean(cond_ann))
    if not np.isfinite(scale) or scale <= 0:
        return raw, raw
    return raw, raw * scale


# --------------------------------------------------------- walk-forward
def walk_forward_models(frame: pd.DataFrame, returns: pd.DataFrame,
                        min_train: int = MIN_TRAIN, refit_every: int = REFIT_EVERY,
                        verbose: bool = True) -> pd.DataFrame:
    """GARCH, GJR-GARCH and CARR forecasts aligned to `frame`'s rows.

    Refitted on the same schedule as every other model, and the parameters
    fitted at a refit date are carried forward until the next one -- which is
    what makes this comparable rather than an optimiser run per day.
    """
    FALLBACKS["garch"] = FALLBACKS["gjr_garch"] = 0
    keys = ("garch", "garch_mz", "gjr_garch", "gjr_garch_mz", "carr")
    out = {k: np.full(len(frame), np.nan) for k in keys}
    ret_map = {t: g.set_index("date")["ret"] for t, g in returns.groupby("ticker")}

    t0 = time.time()
    for n, (ticker, g) in enumerate(frame.groupby("ticker"), 1):
        idx = g.index.to_numpy()
        dates = g["date"].to_numpy()
        rv = g["naive"].to_numpy(float)          # today's realized vol
        rser = ret_map.get(ticker)
        if rser is None or len(g) <= min_train:
            continue
        rvals = rser.reindex(pd.DatetimeIndex(dates)).to_numpy(float)
        rvals = np.nan_to_num(rvals, nan=0.0)

        cache = {}
        for i in range(min_train, len(g)):
            if (i - min_train) % refit_every == 0:
                hist_ret, hist_rv = rvals[:i], rv[:i]
                g_raw, g_mz = garch_forecast(hist_ret, False, hist_rv)
                j_raw, j_mz = garch_forecast(hist_ret, True, hist_rv)
                cache = {"garch": g_raw, "garch_mz": g_mz,
                         "gjr_garch": j_raw, "gjr_garch_mz": j_mz,
                         "carr": carr_forecast(hist_rv)}
            for k, v in cache.items():
                out[k][idx[i]] = v
        if verbose:
            print(f"    [{n:>2}] {ticker:<16} {time.time() - t0:6.0f}s", flush=True)

    if verbose:
        print(f"    GARCH fallbacks: {FALLBACKS['garch']} garch, "
              f"{FALLBACKS['gjr_garch']} gjr_garch", flush=True)
    res = pd.DataFrame(out, index=frame.index)
    # These cost ~83 minutes of MLE to produce. Cache them so any later
    # re-analysis -- a different loss, a recalibration, a new competitor --
    # is seconds rather than another full refit.
    cache_path = RESULTS / "garch_carr_forecasts.parquet"
    try:
        keep = res.copy()
        keep["date"] = frame["date"].to_numpy()
        keep["ticker"] = frame["ticker"].to_numpy()
        keep.to_parquet(cache_path, index=False)
        if verbose:
            print(f"    cached forecasts -> {cache_path.name}", flush=True)
    except Exception as exc:                            # noqa: BLE001
        print(f"    could not cache forecasts: {type(exc).__name__}: {exc}")
    return res


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="parkinson")
    ap.add_argument("--tickers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    from quant.mcs import model_confidence_set
    from quant.vol_models import FEATURE_SETS, build_features, walk_forward
    from quant.garch_carr import FALLBACKS
    from quant.volatility import download_ohlc, mse, qlike, qlike_series

    frame = build_features(args.estimator, verbose=True)
    if args.tickers:
        keep = sorted(frame["ticker"].unique())[: args.tickers]
        frame = frame[frame["ticker"].isin(keep)].reset_index(drop=True)

    ohlc = download_ohlc(verbose=False).sort_values(["ticker", "date"])
    ohlc["ret"] = ohlc.groupby("ticker")["close"].pct_change()
    returns = ohlc[["date", "ticker", "ret"]].dropna()

    print(f"  {len(frame):,} rows, {frame['ticker'].nunique()} tickers")
    print("  fitting the HAR-family models ...", flush=True)
    cols = {"naive": frame["naive"].to_numpy(float)}
    for model in FEATURE_SETS:
        cols[model] = walk_forward(frame, model)

    print("  fitting GARCH / GJR-GARCH / CARR (MLE per refit) ...", flush=True)
    extra = walk_forward_models(frame, returns)
    for k in extra.columns:
        cols[k] = extra[k].to_numpy()

    scored = frame.copy()
    for k, v in cols.items():
        scored[k] = v
    models = list(cols)
    scored = scored.dropna(subset=models).reset_index(drop=True)
    actual = scored["target"].to_numpy(float)

    summary = {m: {"qlike": qlike(actual, scored[m].to_numpy(float)),
                   "mse": mse(actual, scored[m].to_numpy(float))} for m in models}
    losses = {m: qlike_series(actual, scored[m].to_numpy(float)) for m in models}
    mcs = model_confidence_set(losses, verbose=True)

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "n_forecasts": int(len(scored)),
           "n_tickers": int(scored["ticker"].nunique()),
           "n_models": len(models), "summary": summary, "mcs": mcs,
           "garch_fallback_fits": dict(FALLBACKS),
           "note": ("GARCH forecasts return variance while the target is a range-based "
                    "realized volatility; both estimate the same latent quantity but "
                    "GARCH is scored on a proxy it was not fitted to. CARR has no such "
                    "mismatch, being fitted on the range itself.")}

    print(f"\n{'=' * 62}\nJOINT COMPARISON  {len(models)} models, "
          f"{len(scored):,} forecasts\n{'=' * 62}")
    print(f"  {'model':<13}{'QLIKE':>10}{'MSE':>11}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
        star = "  <- MCS" if m in mcs["mcs"] else ""
        print(f"  {m:<13}{v['qlike']:>10.5f}{v['mse']:>11.6f}{star}")
    print(f"\n  MCS (95%): {mcs['mcs']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"volatility_joint_{args.estimator}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
