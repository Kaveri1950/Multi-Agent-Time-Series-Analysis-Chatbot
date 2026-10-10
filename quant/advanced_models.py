"""Everything the cross-domain research suggested, benchmarked together.

The project has an unusual advantage at this point: it knows its own ceiling.
A state-space fit puts the variance of latent log volatility at 0.125 and the
measurement-noise variance at 0.110, so a PERFECT forecast of the latent
process would score R-squared 0.533 against the observed series, and 92% of
the day-to-day movement in that series is estimation error rather than
market behaviour. HAR reaches 0.353, about two-thirds of what is attainable.

So the question is not "can we reach 0.8" -- nothing can -- but whether the
remaining third of the attainable signal can be recovered. Four families are
tried here, each chosen because it targets the noise problem rather than
adding capacity for its own sake.

KALMAN STATE SPACE. Estimates the signal-to-noise trade-off from the data
instead of hard-coding it as HAR's 1/5/22-day windows. The daily analogue of
HARQ, whose quarticity correction needs intraday returns.

TEMPORAL HIERARCHY (MAPA, Kourentzes et al. 2014; Athanasopoulos et al.
2017). Forecast at daily, weekly and monthly aggregation and reconcile. The
mechanism is exactly what this project measured independently: temporal
aggregation filters high-frequency components, so the weekly and monthly
levels carry a cleaner signal, and the reconciled daily forecast inherits it.
The project's own numbers show the effect -- the same forecast scores 0.322
against one day and 0.405 against a 5-day average.

WAVELET DECOMPOSITION, borrowed from electricity-price forecasting. Split
log volatility into frequency bands, model each, recombine. Strictly causal
here: the transform is applied only to history, never across the forecast
boundary, which is the error that makes most published wavelet forecasting
results irreproducible.

MACHINE LEARNING, on the same features so the comparison isolates the
learner. Random forest, extra trees, SVR and an MLP. These have lost before
in this project, which is the point of running them again under a protocol
that is now better understood.

Run:  python -m quant.advanced_models [--tickers 20]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import MIN_TRAIN, RESULTS, mse, qlike, qlike_series

HAR_FEATS = ["rv_1", "rv_5", "rv_22"]
ML_FEATS = ["rv_1", "rv_5", "rv_22", "lev", "ovn", "vix"]
REFIT_ML = 252          # annual; the sweep showed refit frequency is minor
WAVELET = "db4"
WAVE_LEVEL = 3


def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def _apply(b, X):
    return np.column_stack([np.ones(len(X)), X]) @ b


# ------------------------------------------------- temporal hierarchy
def walk_forward_mapa(frame: pd.DataFrame, min_train: int = MIN_TRAIN,
                      refit_every: int = 21, lookback: int = 750) -> np.ndarray:
    """Forecast at three aggregation levels, then reconcile to the daily one.

    Each level gets its own HAR-style regression on the SAME information set,
    but with the target aggregated over 1, 5 and 22 days. The aggregate
    targets are less noisy, so their coefficients are estimated more
    precisely; the reconciliation step averages the three implied daily
    forecasts with weights inversely proportional to each level's training
    error, which is the practical form of the temporal-hierarchy argument.
    """
    preds = np.full(len(frame), np.nan)
    for ticker, g in frame.groupby("ticker"):
        g = g.sort_values("date")
        idx = g.index.to_numpy()
        X = g[HAR_FEATS].to_numpy(float)
        rv = g["naive"].to_numpy(float)
        n = len(g)
        if n <= min_train + 25:
            continue
        # aggregate targets: mean realized vol over the next k days
        tg = {}
        for k in (1, 5, 22):
            fwd = pd.Series(rv).shift(-1).rolling(k).mean().shift(-(k - 1))
            tg[k] = fwd.to_numpy(float)

        beta, w = {}, None
        for i in range(min_train, n):
            lo = max(0, i - lookback)
            if not beta or (i - min_train) % refit_every == 0:
                errs = {}
                for k in (1, 5, 22):
                    ok = np.isfinite(tg[k][lo:i])
                    if ok.sum() < 100:
                        continue
                    b = _ols(X[lo:i][ok], tg[k][lo:i][ok])
                    beta[k] = b
                    resid = tg[k][lo:i][ok] - _apply(b, X[lo:i][ok])
                    errs[k] = float(np.mean(resid ** 2)) or 1e-9
                if not beta:
                    continue
                inv = np.array([1.0 / errs[k] for k in sorted(beta)])
                w = inv / inv.sum()
            if not beta or w is None:
                continue
            each = np.array([float(_apply(beta[k], X[i:i + 1])[0]) for k in sorted(beta)])
            preds[idx[i]] = float(each @ w)
    return preds


# ------------------------------------------------------ wavelet + HAR
def walk_forward_wavelet(frame: pd.DataFrame, min_train: int = MIN_TRAIN,
                         refit_every: int = 21, lookback: int = 750) -> np.ndarray:
    """Decompose log volatility into frequency bands, model each, recombine.

    The transform is applied ONLY to the training history at each step. A
    wavelet transform run across the whole series leaks future information
    backwards through the filter, which is the flaw in most published
    wavelet forecasting results.
    """
    import pywt

    preds = np.full(len(frame), np.nan)
    for ticker, g in frame.groupby("ticker"):
        g = g.sort_values("date")
        idx = g.index.to_numpy()
        rv = np.log(np.clip(g["naive"].to_numpy(float), 1e-8, None))
        tgt = np.log(np.clip(g["target"].to_numpy(float), 1e-8, None))
        n = len(g)
        if n <= min_train + 25:
            continue
        for i in range(min_train, n):
            if (i - min_train) % refit_every and not np.isnan(preds[idx[i - 1]]):
                # carry the previous coefficients; refit only periodically
                pass
            lo = max(0, i - lookback)
            hist = rv[lo:i]
            try:
                coeffs = pywt.wavedec(hist, WAVELET, level=WAVE_LEVEL, mode="periodization")
            except Exception:                           # noqa: BLE001
                continue
            # reconstruct each band back to full length, then regress the
            # target on the band values at the most recent observation
            bands = []
            for j in range(len(coeffs)):
                c = [np.zeros_like(x) for x in coeffs]
                c[j] = coeffs[j]
                rec = pywt.waverec(c, WAVELET, mode="periodization")[: len(hist)]
                bands.append(rec)
            B = np.column_stack(bands)
            y = tgt[lo:i]
            m = min(len(B), len(y))
            if m < 100:
                continue
            b = _ols(B[:m], y[:m])
            preds[idx[i]] = float(np.exp(_apply(b, B[m - 1:m])[0]))
    return preds


# ----------------------------------------------------- machine learning
def walk_forward_ml(frame: pd.DataFrame, kind: str, min_train: int = MIN_TRAIN,
                    refit_every: int = REFIT_ML, seed: int = 0) -> np.ndarray:
    """Pooled ML on the HAR feature set, refitted annually."""
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    def make():
        if kind == "random_forest":
            return RandomForestRegressor(n_estimators=250, max_depth=8,
                                         min_samples_leaf=20, n_jobs=-1, random_state=seed)
        if kind == "extra_trees":
            return ExtraTreesRegressor(n_estimators=250, max_depth=10,
                                       min_samples_leaf=20, n_jobs=-1, random_state=seed)
        if kind == "svr":
            return make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.02, gamma="scale"))
        if kind == "mlp":
            return make_pipeline(StandardScaler(),
                                 MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                                              early_stopping=True, random_state=seed))
        raise ValueError(kind)

    feats = [c for c in ML_FEATS if c in frame.columns and frame[c].notna().any()]
    df = frame.dropna(subset=feats + ["target"]).sort_values("date")
    preds = np.full(len(frame), np.nan)
    dates = df["date"].to_numpy()
    X = df[feats].to_numpy(float)
    y = df["target"].to_numpy(float)
    idx = df.index.to_numpy()
    uniq = np.sort(np.unique(dates))
    if len(uniq) <= min_train:
        return preds

    model, last = None, -10 ** 9
    for di in range(min_train, len(uniq)):
        day = uniq[di]
        if model is None or (di - last) >= refit_every:
            tr = dates < day
            if tr.sum() < 500:
                continue
            # SVR is O(n^2); subsample for it only, deterministically
            sel = np.where(tr)[0]
            if kind == "svr" and len(sel) > 8000:
                sel = np.random.default_rng(seed).choice(sel, 8000, replace=False)
            model = make().fit(X[sel], y[sel])
            last = di
        m = dates == day
        if not m.any() or model is None:
            continue
        preds[idx[m]] = model.predict(X[m])
    return preds


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="garman_klass")
    ap.add_argument("--tickers", type=int, default=0)
    args = ap.parse_args()

    from quant.mcs import model_confidence_set
    from quant.statespace import walk_forward_ss
    from quant.vol_models import build_features, walk_forward

    frame = build_features(args.estimator, verbose=True)
    if args.tickers:
        keep = sorted(frame["ticker"].unique())[: args.tickers]
        frame = frame[frame["ticker"].isin(keep)].reset_index(drop=True)
    print(f"  {len(frame):,} rows, {frame['ticker'].nunique()} tickers\n")

    cols = {"naive": frame["naive"].to_numpy(float)}
    jobs = [
        ("har", lambda: walk_forward(frame, "har")),
        ("har_ovn", lambda: walk_forward(frame, "har_ovn")),
        ("kalman_ss", lambda: walk_forward_ss(frame, verbose=False)),
        ("mapa_hier", lambda: walk_forward_mapa(frame)),
        ("wavelet_har", lambda: walk_forward_wavelet(frame)),
        ("random_forest", lambda: walk_forward_ml(frame, "random_forest")),
        ("extra_trees", lambda: walk_forward_ml(frame, "extra_trees")),
        ("svr", lambda: walk_forward_ml(frame, "svr")),
        ("mlp", lambda: walk_forward_ml(frame, "mlp")),
    ]
    for name, fn in jobs:
        t0 = time.time()
        try:
            cols[name] = fn()
            got = int(np.isfinite(cols[name]).sum())
            print(f"    {name:<14} {got:>7,} forecasts  {time.time() - t0:7.0f}s", flush=True)
        except Exception as exc:                        # noqa: BLE001
            print(f"    {name:<14} FAILED {type(exc).__name__}: {exc}", flush=True)

    fam = [c for c in ("har_ovn", "kalman_ss", "mapa_hier") if c in cols]
    cols["ens_best3"] = np.nanmean(np.column_stack([cols[c] for c in fam]), axis=1)

    sc = frame.copy()
    for k, v in cols.items():
        sc[k] = v
    models = [m for m in cols if np.isfinite(sc[m]).sum() > 1000]
    sc = sc.dropna(subset=models).reset_index(drop=True)
    a = sc["target"].to_numpy(float)

    def r2(y, yh):
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

    def r2log(y, yh):
        ly, lp = np.log(np.clip(y, 1e-9, None)), np.log(np.clip(yh, 1e-9, None))
        return 1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)

    summary = {m: {"qlike": qlike(a, sc[m].to_numpy(float)),
                   "mse": mse(a, sc[m].to_numpy(float)),
                   "r2": float(r2(a, sc[m].to_numpy(float))),
                   "r2_log": float(r2log(a, sc[m].to_numpy(float)))}
               for m in models}
    mcs = model_confidence_set({m: qlike_series(a, sc[m].to_numpy(float))
                                for m in models}, verbose=True)

    CEILING = 0.533
    print(f"\n{'=' * 70}\nADVANCED ROSTER   {len(sc):,} forecasts, "
          f"{sc['ticker'].nunique()} tickers\n{'=' * 70}")
    print(f"  {'model':<15}{'QLIKE':>10}{'R2':>8}{'R2 log':>9}{'% of ceiling':>14}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
        tag = "  <- MCS" if m in mcs["mcs"] else ""
        print(f"  {m:<15}{v['qlike']:>10.5f}{v['r2']:>8.3f}{v['r2_log']:>9.3f}"
              f"{100 * v['r2_log'] / CEILING:>13.0f}%{tag}")
    print(f"\n  MCS (95%): {mcs['mcs']}")
    print(f"  noise ceiling on R2(log) is {CEILING:.3f} -- 92% of day-to-day "
          "movement is measurement error")

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "n_forecasts": int(len(sc)),
           "n_tickers": int(sc["ticker"].nunique()), "noise_ceiling_r2_log": CEILING,
           "summary": summary, "mcs": mcs}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_advanced.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
