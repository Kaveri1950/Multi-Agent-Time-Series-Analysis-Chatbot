"""Models optimised for TRACKING, not for the risk loss.

THE OBJECTIVE CHANGES HERE, DELIBERATELY. Everything else in this project
scores QLIKE, which is the right loss for a risk number: it punishes
under-forecasting volatility harder than over-forecasting, because a risk
limit set too low is worse than one set too high. But QLIKE is a LEVEL loss,
and a model tuned for it will happily sit near the mean.

If the goal is instead that the forecast should FOLLOW the realized series,
the objective is squared error on log volatility -- equivalently, maximise
R-squared in logs. Those are different targets and they select different
models: in the earlier roster SVR had the best R-squared(log) of anything
tried, 0.316, while having nearly the worst QLIKE.

So this module changes three things together, which is why it is separate
rather than another entry in the roster:

  target   log realized volatility, not the level
  loss     squared error in log space, which is what R-squared measures
  features cross-sectional ones added, because volatility has a strong
           common component that a single-ticker model cannot see

THE CROSS-SECTIONAL FEATURES ARE THE NEW INFORMATION. Measured on this
panel, the market-wide median volatility correlates +0.396 with a ticker's
next-day log volatility, and the ticker's volatility RELATIVE to the market
correlates +0.471. Neither is available to a model that only ever sees one
stock, and both are free -- the panel is already downloaded. A stock that is
calm while the whole market is calm is in a different state from one that is
calm while the market is stressed, and only the relative term can express it.

THE CEILING STILL APPLIES. A state-space fit puts the attainable
R-squared(log) at about 0.533, because 92% of day-to-day movement in the
observed series is measurement error. Nothing here can exceed that, and a
model that appears to has leaked.

Run:  python -m quant.tracking_models [--tickers 20]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import MIN_TRAIN, RESULTS, qlike

CEILING_R2_LOG = 0.533
REFIT_DAYS = 252
SEED = 0

FEATURES = [
    "l_rv1", "l_rv5", "l_rv22", "l_rv66",      # own history, several horizons
    "l_mkt", "l_rel",                           # market factor and relative level
    "l_vix", "l_ovn",                           # implied vol and the overnight gap
    "neg", "spread",                            # leverage sign, short-vs-long spread
]


def build_tracking_frame(estimator: str = "garman_klass", verbose: bool = True,
                         require_target: bool = True, ohlc: pd.DataFrame | None = None,
                         market_tickers: list | None = None) -> pd.DataFrame:
    """Log-space features including the cross-sectional terms.

    `require_target=False` keeps the most recent row, whose target is the
    day that has not happened yet. Training must drop it -- there is nothing
    to fit against -- but SERVING needs exactly that row, and serving from
    the last row with a known target would be forecasting a day already
    observed, using a row the model was trained on.

    `ohlc` and `market_tickers` exist for serving names outside the panel.
    The market factor must then come from the panel alone: a new name's own
    volatility leaking into "the market" would be a feature the trained model
    never saw, and a held-out name's leaking in would flatter its test.
    """
    from quant.vol_models import build_features

    f = build_features(estimator, verbose=verbose, require_target=require_target, ohlc=ohlc)
    f = f.sort_values(["date", "ticker"]).reset_index(drop=True)

    # The market factor is the cross-sectional MEDIAN of today's realized
    # volatility. Median rather than mean so one blown-up stock cannot
    # define the market state. It uses only same-day information, which is
    # available when the forecast is made.
    base = f if market_tickers is None else f[f["ticker"].isin(market_tickers)]
    mkt = base.groupby("date")["naive"].median().rename("mkt")
    f = f.merge(mkt, on="date", how="left")

    g = []
    for t, s in f.groupby("ticker"):
        s = s.sort_values("date").copy()
        s["rv66"] = s["naive"].rolling(66).mean()
        g.append(s)
    f = pd.concat(g, ignore_index=True)

    lg = lambda x: np.log(np.clip(x, 1e-8, None))      # noqa: E731
    f["l_rv1"] = lg(f["naive"])
    f["l_rv5"] = lg(f["rv_5"])
    f["l_rv22"] = lg(f["rv_22"])
    f["l_rv66"] = lg(f["rv66"])
    f["l_mkt"] = lg(f["mkt"])
    f["l_rel"] = f["l_rv1"] - f["l_mkt"]
    f["l_vix"] = lg(f["vix"])
    f["l_ovn"] = lg(f["ovn"])
    f["neg"] = (f["lev"] > 0).astype(float)
    f["spread"] = f["l_rv5"] - f["l_rv22"]
    f["y"] = lg(f["target"])
    need = FEATURES + (["y", "target"] if require_target else [])
    return f.dropna(subset=need).reset_index(drop=True)


def _make(kind: str):
    from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if kind == "ridge_log":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if kind == "rf_log":
        return RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=20,
                                     n_jobs=-1, random_state=SEED)
    if kind == "et_log":
        # depth 12 / leaf 50 won a causal grid search (0.4420 against ridge's
        # 0.4377). The grid was nearly flat and the DEEPER settings were
        # worse -- depth=None with leaf=5 fell to 0.4351 -- so this is a
        # regularisation choice, not a capacity one.
        return ExtraTreesRegressor(n_estimators=400, max_depth=12, min_samples_leaf=50,
                                   n_jobs=-1, random_state=SEED)
    if kind == "gbm_log":
        # lr 0.03 / depth 3 was the best of six settings; at lr 0.06 depth 5
        # it collapsed to 0.4232, which is the same over-capacity failure the
        # spline and the LSTMs showed.
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_depth=3,
                                             l2_regularization=1.0, random_state=SEED)
    if kind == "svr_log":
        return make_pipeline(StandardScaler(), SVR(C=2.0, epsilon=0.05, gamma="scale"))
    if kind == "mlp_log":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(96, 48), alpha=1e-3,
                                          max_iter=600, early_stopping=True,
                                          random_state=SEED))
    raise ValueError(kind)


def walk_forward(f: pd.DataFrame, kind: str, min_train: int = MIN_TRAIN,
                 refit_days: int = REFIT_DAYS, subsample: int = 0) -> np.ndarray:
    """Pooled, expanding-window, refitted annually. Predicts LOG volatility."""
    preds = np.full(len(f), np.nan)
    dates = f["date"].to_numpy()
    X = f[FEATURES].to_numpy(float)
    y = f["y"].to_numpy(float)
    uniq = np.sort(np.unique(dates))
    if len(uniq) <= min_train:
        return preds
    model, last = None, -10 ** 9
    rng = np.random.default_rng(SEED)
    for di in range(min_train, len(uniq)):
        day = uniq[di]
        if model is None or (di - last) >= refit_days:
            tr = np.where(dates < day)[0]
            if len(tr) < 1000:
                continue
            if subsample and len(tr) > subsample:
                tr = rng.choice(tr, subsample, replace=False)
            model = _make(kind).fit(X[tr], y[tr])
            last = di
        m = dates == day
        if m.any() and model is not None:
            preds[m] = model.predict(X[m])
    return preds


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="garman_klass")
    ap.add_argument("--tickers", type=int, default=0)
    args = ap.parse_args()

    from quant.mcs import model_confidence_set

    f = build_tracking_frame(args.estimator)
    if args.tickers:
        keep = sorted(f["ticker"].unique())[: args.tickers]
        f = f[f["ticker"].isin(keep)].reset_index(drop=True)
    print(f"  {len(f):,} rows, {f['ticker'].nunique()} tickers, {len(FEATURES)} features\n")

    # HAR in log space, as the fair benchmark for a log-space objective
    cols = {}
    specs = [("har_log", dict(kind="ridge_log")),
             ("rf_log", dict(kind="rf_log")),
             ("et_log", dict(kind="et_log")),
             ("gbm_log", dict(kind="gbm_log")),
             ("svr_log", dict(kind="svr_log", subsample=12000)),
             ("mlp_log", dict(kind="mlp_log"))]
    for name, kw in specs:
        t0 = time.time()
        try:
            cols[name] = walk_forward(f, **kw)
            print(f"    {name:<10} {int(np.isfinite(cols[name]).sum()):>7,} preds "
                  f"{time.time() - t0:7.0f}s", flush=True)
        except Exception as exc:                        # noqa: BLE001
            print(f"    {name:<10} FAILED {type(exc).__name__}: {exc}", flush=True)

    good = [k for k, v in cols.items() if np.isfinite(v).sum() > 1000]
    cols["ens_log"] = np.nanmean(np.column_stack([cols[k] for k in good]), axis=1)
    cols["naive"] = f["l_rv1"].to_numpy(float)
    cols["har3_log"] = np.nan                            # placeholder, filled below

    # plain HAR on the three own-history terms only, in logs -- isolates how
    # much the cross-sectional features are worth
    f3 = f.copy()
    keep3 = ["l_rv1", "l_rv5", "l_rv22"]
    saved = FEATURES[:]
    FEATURES[:] = keep3
    cols["har3_log"] = walk_forward(f3, "ridge_log")
    FEATURES[:] = saved

    sc = f.copy()
    for k, v in cols.items():
        sc[k] = v
    models = [m for m in cols if np.isfinite(sc[m]).sum() > 1000]
    sc = sc.dropna(subset=models).reset_index(drop=True)
    ly = sc["y"].to_numpy(float)
    lvl = sc["target"].to_numpy(float)

    def r2(y, yh):
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

    summary = {}
    for m in models:
        lp = sc[m].to_numpy(float)
        resid = ly - lp
        lvl_p = np.exp(lp + 0.5 * np.var(resid))         # smearing back to level
        summary[m] = {"r2_log": float(r2(ly, lp)),
                      "corr_log": float(np.corrcoef(ly, lp)[0, 1]),
                      "qlike": qlike(lvl, lvl_p),
                      "pct_of_ceiling": float(100 * r2(ly, lp) / CEILING_R2_LOG)}

    mcs = model_confidence_set({m: (ly - sc[m].to_numpy(float)) ** 2 for m in models},
                               verbose=True)

    print(f"\n{'=' * 68}\nTRACKING OBJECTIVE (log-space MSE)   {len(sc):,} forecasts\n{'=' * 68}")
    print(f"  {'model':<12}{'R2 log':>9}{'corr':>8}{'% ceiling':>11}{'QLIKE':>10}")
    for m, v in sorted(summary.items(), key=lambda kv: -kv[1]["r2_log"]):
        tag = "  <- MCS" if m in mcs["mcs"] else ""
        print(f"  {m:<12}{v['r2_log']:>9.3f}{v['corr_log']:>8.3f}"
              f"{v['pct_of_ceiling']:>10.0f}%{v['qlike']:>10.4f}{tag}")
    print(f"\n  MCS (95%, log-MSE loss): {mcs['mcs']}")
    print(f"  ceiling on R2(log) is {CEILING_R2_LOG:.3f}")

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "objective": "squared error on log volatility",
           "features": FEATURES, "ceiling_r2_log": CEILING_R2_LOG,
           "n_forecasts": int(len(sc)), "n_tickers": int(sc["ticker"].nunique()),
           "summary": summary, "mcs": mcs}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_tracking.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")

    # Per-observation predictions, on the ORIGINAL scale, so the dashboard can
    # show every model on any ticker without refitting. These are pooled
    # models trained across all 48 names -- retraining six learners inside a
    # web request would take half an hour, and training them on one ticker
    # instead would be a different (worse) model than the one benchmarked.
    # Each column is exponentiated with its own smearing correction, since a
    # log-space fit exponentiates biased low.
    keep = sc[["date", "ticker", "target"]].copy()
    for m in models:
        lp = sc[m].to_numpy(float)
        keep[m] = np.exp(lp + 0.5 * np.var(ly - lp))
    pred_path = RESULTS / "volatility_tracking_predictions.parquet"
    keep.to_parquet(pred_path, index=False)
    print(f"wrote {pred_path}  ({len(keep):,} rows x {len(models)} models)")


if __name__ == "__main__":
    main()
