"""Does the results-day gain survive in the full six-member ensemble?

evaluation/news_volatility.py found that the volatility model under-forecasts
on results days (about +0.31 to +0.38 in logs, every year 2021-2026, 49 of 49
stocks), and that adding calendar and filing features to the RIDGE member
raised within-stock R2(log) from 0.2391 to 0.2558. Ridge is one member of
six. Before anything changes in the served model, the same comparison has to
hold for the ensemble that is actually served.

Both arms use identical rows, members, refit schedule and seed; the only
difference is three extra columns. Scored against the same ceilings as
everything else: direction 74.1%, within-stock R2(log) 0.533.

THE THREE FEATURES, AND WHY EACH IS KNOWN IN TIME
  results_next_session   a results date on the calendar for tomorrow.
                         Board meetings for results are intimated to the
                         exchange in advance, so this is known today.
  results_today          a results date today. Results usually land after
                         the close, so tomorrow is the reaction day.
  log_ann_material       non-routine filings stamped today, by the close.

Run:  python -m evaluation.news_ensemble        (about 80 minutes)
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "news_ensemble.json"
MEMBERS = ("ridge_log", "rf_log", "et_log", "gbm_log", "svr_log", "mlp_log")
NEWS = ["results_next_session", "results_today", "log_ann_material"]
CEILING_DIRECTION, CEILING_R2 = 0.741, 0.533


def build() -> pd.DataFrame:
    from finance import news_history as nh
    from quant.tracking_models import build_tracking_frame

    f = build_tracking_frame("garman_klass", verbose=False, require_target=True)
    symbols = sorted(t.replace(".NS", "") for t in f["ticker"].unique())
    raw = nh.build(symbols, verbose=False)
    news = nh.daily_features(raw, pd.DatetimeIndex(f["date"].unique()), symbols)
    f = f.merge(news[["date", "ticker", "results_next_session", "results_today",
                      "ann_material"]], on=["date", "ticker"], how="left")
    f["results_next_session"] = f["results_next_session"].fillna(0.0)
    f["results_today"] = f["results_today"].fillna(0.0)
    f["log_ann_material"] = np.log1p(f["ann_material"].fillna(0.0))
    return f.sort_values(["date", "ticker"]).reset_index(drop=True)


def walk(f: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    """tracking_models.walk_forward with the feature list as an argument."""
    from quant.tracking_models import REFIT_DAYS, SEED, _make
    from quant.volatility import MIN_TRAIN

    X, y = f[cols].to_numpy(float), f["y"].to_numpy(float)
    dates = f["date"].to_numpy()
    uniq = np.sort(np.unique(dates))
    preds = np.full(len(f), np.nan)
    rng = np.random.default_rng(SEED)
    model, last = None, -10 ** 9
    for di in range(MIN_TRAIN, len(uniq)):
        if model is None or di - last >= REFIT_DAYS:
            tr = np.where(dates < uniq[di])[0]
            if len(tr) < 1000:
                continue
            if kind == "svr_log" and len(tr) > 12000:
                tr = rng.choice(tr, 12000, replace=False)
            model, last = _make(kind).fit(X[tr], y[tr]), di
        m = dates == uniq[di]
        preds[m] = model.predict(X[m])
    return preds


def score(d: pd.DataFrame, col: str) -> dict:
    from quant.volatility import qlike

    y, p = d["y"].to_numpy(float), d[col].to_numpy(float)
    v = float(np.var(y - p))
    dm = d.groupby("ticker")[["y", col]].transform("mean")
    yy, pp = (d["y"] - dm["y"]).to_numpy(), (d[col] - dm[col]).to_numpy()
    return {"r2_within": float(1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2)),
            "r2_pooled": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
            "direction": float(((p > d["l_rv1"]) == (y > d["l_rv1"])).mean()),
            "qlike": float(qlike(d["target"].to_numpy(float), np.exp(p + 0.5 * v))),
            "mse_log": float(np.mean((y - p) ** 2)), "n": int(len(d))}


def main():
    warnings.filterwarnings("ignore")
    from quant.mcs import model_confidence_set
    from quant.tracking_models import FEATURES
    from quant.volatility import diebold_mariano

    t0 = time.time()
    f = build()
    print(f"{len(f):,} rows, {f['ticker'].nunique()} tickers; results days "
          f"{int(f['results_today'].sum())}", flush=True)
    arms = {"base": list(FEATURES), "news": list(FEATURES) + NEWS}
    for arm, cols in arms.items():
        for kind in MEMBERS:
            f[f"{arm}:{kind}"] = walk(f, cols, kind)
            print(f"  {arm:<5} {kind:<9} {time.time() - t0:7.0f}s", flush=True)
        f[arm] = np.nanmean(np.column_stack([f[f"{arm}:{k}"] for k in MEMBERS]), axis=1)

    d = f.dropna(subset=["base", "news", "y"]).reset_index(drop=True)
    on_results = (d["results_today"] == 1) | (d["results_next_session"] == 1)
    loss = {a: (d["y"] - d[a]) ** 2 for a in arms}
    daily = {a: loss[a].groupby(d["date"].to_numpy()).mean() for a in arms}
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "news_features": NEWS, "members": list(MEMBERS),
        "all_days": {a: score(d, a) for a in arms},
        "results_days": {a: score(d[on_results], a) for a in arms},
        "other_days": {a: score(d[~on_results], a) for a in arms},
        "bias_on_results_days": {a: float((d["y"] - d[a])[on_results].mean()) for a in arms},
        "diebold_mariano_news_vs_base": diebold_mariano(daily["news"].to_numpy(),
                                                        daily["base"].to_numpy()),
        "model_confidence_set": model_confidence_set(
            {a: loss[a].to_numpy() for a in arms}, verbose=False).get("mcs"),
        "per_member_r2_within": {f"{arm}:{k}": score(d.dropna(subset=[f"{arm}:{k}"]),
                                                     f"{arm}:{k}")["r2_within"]
                                 for arm in arms for k in MEMBERS},
        "ceilings": {"direction": CEILING_DIRECTION, "r2_within": CEILING_R2},
        "seconds": round(time.time() - t0),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"\nFULL ENSEMBLE, WITH AND WITHOUT NEWS FEATURES  ({out['all_days']['base']['n']:,} "
          f"forecasts, {int(on_results.sum()):,} on results days)")
    for label, key in (("all days", "all_days"), ("results days", "results_days"),
                       ("other days", "other_days")):
        b, n = out[key]["base"], out[key]["news"]
        print(f"  {label:<13} within-R2 {b['r2_within']:.4f} -> {n['r2_within']:.4f}   "
              f"direction {b['direction']:.2%} -> {n['direction']:.2%}   "
              f"QLIKE {b['qlike']:.4f} -> {n['qlike']:.4f}")
    bb = out["bias_on_results_days"]
    print(f"  bias on results days (log)  {bb['base']:+.3f} -> {bb['news']:+.3f}")
    dm = out["diebold_mariano_news_vs_base"]
    print(f"  Diebold-Mariano (negative favours news) t={dm['stat']:.2f} p={dm['p_value']:.3g};"
          f"  confidence set {out['model_confidence_set']}")
    print(f"  ceilings: direction {CEILING_DIRECTION:.1%}, within-R2 {CEILING_R2}")
    print(f"\nwrote {OUT_PATH}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
