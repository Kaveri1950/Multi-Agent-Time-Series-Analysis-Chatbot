"""Can the pooled volatility model serve a stock it was never trained on?

Today it cannot try: `quant.production_model.forecast` refuses any name
outside its 50, and the dashboard falls back to a HAR fitted on that one
stock's history. Zhang, Zhang, Cucuringu & Qian (2024, J. Financial
Econometrics) report that pooled models of realized volatility transfer to
stocks outside the training set, which suggests the refusal is giving away
skill. This measures whether that holds here.

TWO TESTS, BECAUSE THEY ANSWER DIFFERENT QUESTIONS

  A  leave-10-out    the 50 split into five folds; each fold is forecast by a
                     model trained on the other 40, against the same names
                     forecast by a model that saw them. Measures what is lost
                     by not having trained on a stock, among similar stocks.

  B  genuinely new   thirty names from outside the panel, in three liquidity
                     tiers, forecast by the model trained on the 50, against
                     the per-stock HAR the dashboard uses today. Measures
                     whether the transfer holds on smaller, less liquid names,
                     which is what serving the whole price panel would mean.

THE MARKET FACTOR COMES FROM THE TRAINING NAMES ONLY. Two of the features are
cross-sectional. Letting a held-out name's volatility into "the market"
would flatter test A, and in serving, a new name's would be a feature the
model never saw.

THE ENSEMBLE HERE IS THREE MEMBERS, NOT SIX: ridge, extra trees and gradient
boosting. SVR and the MLP would multiply the run time several-fold. Every
comparison below is like for like, so the three-member version answers the
transfer question; it does not re-measure the served model's absolute skill.

Run:  python -m evaluation.out_of_panel        (about half an hour)
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
OUT_PATH = RESULTS / "out_of_panel.json"
MEMBERS = ("ridge_log", "et_log", "gbm_log")
TIERS = {"next_largest": (0, 10), "mid": (100, 110), "small": (250, 260)}
N_FOLDS = 5
MIN_ROWS = 600            # a new name needs this much history to be scored
HAR_WINDOW, HAR_REFIT = 750, 21


def outside_names() -> dict:
    """Thirty names outside the panel, ranked by traded value, in three tiers."""
    from quant.data import load_panel
    from quant.volatility import OHLC_PATH

    close, volume = load_panel()
    panel = set(pd.read_parquet(OHLC_PATH, columns=["ticker"])["ticker"].unique())
    tv = (close * volume).rolling(21, min_periods=10).mean().mean()
    ranked = [t for t in tv.sort_values(ascending=False).index if t not in panel]
    return {tier: ranked[a:b] for tier, (a, b) in TIERS.items()}


def download(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start="2019-01-01", progress=False, auto_adjust=True,
                      threads=True)
    close = raw["Close"].dropna(how="all")
    keep = close.index[close.notna().mean(axis=1) > 0.5]
    frames = [raw[f].loc[keep].stack().rename(f.lower()) for f in ("Open", "High", "Low", "Close")]
    df = pd.concat(frames, axis=1).reset_index()
    df.columns = ["date", "ticker", "open", "high", "low", "close"]
    return df.dropna()


def walk(f: pd.DataFrame, train_tickers: set) -> np.ndarray:
    """Pooled walk-forward, refitted annually, trained on `train_tickers` only.

    Predicts every row, so the same fit scores the training names and any
    name it never saw.
    """
    from quant.tracking_models import FEATURES, REFIT_DAYS, _make
    from quant.volatility import MIN_TRAIN

    X = f[FEATURES].to_numpy(float)
    y = f["y"].to_numpy(float)
    dates = f["date"].to_numpy()
    is_train = f["ticker"].isin(train_tickers).to_numpy()
    uniq = np.sort(np.unique(dates[is_train]))
    pred = np.full(len(f), np.nan)
    models, last = None, -10 ** 9
    for di in range(MIN_TRAIN, len(uniq)):
        if models is None or di - last >= REFIT_DAYS:
            tr = is_train & (dates < uniq[di]) & np.isfinite(y)
            models, last = [_make(k).fit(X[tr], y[tr]) for k in MEMBERS], di
        m = dates == uniq[di]
        if m.any():
            pred[m] = np.mean([mdl.predict(X[m]) for mdl in models], axis=0)
    return pred


def per_stock_har(f: pd.DataFrame) -> np.ndarray:
    """The dashboard's current fallback: HAR with overnight, fitted per stock.

    Same features and window as `quant.vol_models.forecast_ticker`, refitted
    every 21 days on the trailing 750, but on the Garman-Klass target so it
    is scored on the same quantity as the pooled model.
    """
    from quant.vol_models import FEATURE_SETS, _ols_apply, _ols_fit

    feats = FEATURE_SETS["har_ovn"]
    pred = np.full(len(f), np.nan)
    for _, g in f.groupby("ticker"):
        g = g.sort_values("date")
        idx = g.index.to_numpy()
        X, y = g[feats].to_numpy(float), g["target"].to_numpy(float)
        beta = None
        for i in range(HAR_WINDOW // 3, len(g)):
            if beta is None or i % HAR_REFIT == 0:
                lo = max(0, i - HAR_WINDOW)
                ok = np.isfinite(y[lo:i]) & np.isfinite(X[lo:i]).all(axis=1)
                if ok.sum() < 120:
                    continue
                beta = _ols_fit(X[lo:i][ok], y[lo:i][ok])
            if beta is not None and np.isfinite(X[i]).all():
                pred[idx[i]] = float(_ols_apply(beta, X[i:i + 1])[0])
    return np.log(np.clip(pred, 1e-4, None))


def score(g: pd.DataFrame, col: str, v: float) -> dict:
    from quant.volatility import qlike

    y, p = g["y"].to_numpy(float), g[col].to_numpy(float)
    dm = g.groupby("ticker")[["y", col]].transform("mean")
    yy, pp = (g["y"] - dm["y"]).to_numpy(), (g[col] - dm[col]).to_numpy()
    return {"qlike": float(qlike(g["target"].to_numpy(float), np.exp(p + 0.5 * v))),
            "r2_within": float(1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2)),
            "direction": float(((p > g["l_rv1"]) == (y > g["l_rv1"])).mean())}


def summarise(d: pd.DataFrame, cols: list[str]) -> dict:
    d = d.dropna(subset=cols + ["y"])
    if d.empty:
        return {"n": 0}
    out = {"n": int(len(d)), "n_tickers": int(d["ticker"].nunique())}
    for c in cols:
        v = float(np.var(d["y"] - d[c]))       # smearing variance, per model
        out[c] = score(d, c, v)
    if "pooled" in cols and "har" in cols:
        wins = 0
        for _, g in d.groupby("ticker"):
            vp, vh = float(np.var(g["y"] - g["pooled"])), float(np.var(g["y"] - g["har"]))
            wins += score(g, "pooled", vp)["qlike"] < score(g, "har", vh)["qlike"]
        out["tickers_where_pooled_beats_har_on_qlike"] = int(wins)
    return out


def main():
    warnings.filterwarnings("ignore")
    from quant.tracking_models import build_tracking_frame
    from quant.volatility import OHLC_PATH

    t0 = time.time()
    panel_ohlc = pd.read_parquet(OHLC_PATH)
    panel = sorted(panel_ohlc["ticker"].unique())
    tiers = outside_names()
    new_ohlc = download([t for names in tiers.values() for t in names])
    counts = new_ohlc.groupby("ticker").size()
    usable = set(counts.index[counts >= MIN_ROWS])
    tiers = {k: [t for t in v if t in usable] for k, v in tiers.items()}
    new_ohlc = new_ohlc[new_ohlc["ticker"].isin(usable)]
    print(f"outside names with >= {MIN_ROWS} days: "
          + ", ".join(f"{k} {len(v)}" for k, v in tiers.items()), flush=True)

    # ---- trained on all 50; scores the panel AND the new names
    both = pd.concat([panel_ohlc, new_ohlc], ignore_index=True)
    f = build_tracking_frame("garman_klass", verbose=False, ohlc=both, market_tickers=panel)
    f = f.sort_values(["date", "ticker"]).reset_index(drop=True)
    f["pooled"] = walk(f, set(panel))
    print(f"  pooled fit on 50: {time.time() - t0:.0f}s", flush=True)
    f["har"] = per_stock_har(f)
    print(f"  per-stock HAR: {time.time() - t0:.0f}s", flush=True)

    # ---- leave-10-out within the panel
    rng = np.random.default_rng(0)
    order = list(rng.permutation(panel))
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
    held = []
    for k, fold in enumerate(folds):
        train = [t for t in panel if t not in fold]
        fk = build_tracking_frame("garman_klass", verbose=False, ohlc=panel_ohlc,
                                  market_tickers=train)
        fk = fk.sort_values(["date", "ticker"]).reset_index(drop=True)
        fk["held_out"] = walk(fk, set(train))
        held.append(fk.loc[fk["ticker"].isin(fold), ["date", "ticker", "held_out"]])
        print(f"  fold {k + 1}/{N_FOLDS}: {time.time() - t0:.0f}s", flush=True)
    f = f.merge(pd.concat(held), on=["date", "ticker"], how="left")

    in_panel = f[f["ticker"].isin(panel)]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "members": list(MEMBERS), "tiers": tiers,
           "A_leave_10_out": summarise(in_panel, ["pooled", "held_out", "har"]),
           "B_new_names": {k: summarise(f[f["ticker"].isin(v)], ["pooled", "har"])
                           for k, v in tiers.items()},
           "B_new_names_all": summarise(f[f["ticker"].isin(usable)], ["pooled", "har"]),
           "seconds": round(time.time() - t0)}
    OUT_PATH.write_text(json.dumps(out, indent=2))

    def line(label, s, col):
        x = s[col]
        return (f"   {label:<34}QLIKE {x['qlike']:.4f}   within-R2 {x['r2_within']:.3f}   "
                f"direction {x['direction']:.1%}")

    a = out["A_leave_10_out"]
    print(f"\nA. LEAVE-10-OUT WITHIN THE PANEL  ({a['n']:,} forecasts, {a['n_tickers']} names)")
    print(line("trained WITH the stock", a, "pooled"))
    print(line("trained WITHOUT the stock", a, "held_out"))
    print(line("per-stock HAR (today's fallback)", a, "har"))
    print("\nB. NAMES OUTSIDE THE PANEL, model trained on the 50")
    for k, s in list(out["B_new_names"].items()) + [("all", out["B_new_names_all"])]:
        if not s.get("n"):
            continue
        print(f"  {k}  ({s['n']:,} forecasts, {s['n_tickers']} names; pooled beats HAR on "
              f"{s['tickers_where_pooled_beats_har_on_qlike']} of {s['n_tickers']})")
        print(line("pooled, never saw these names", s, "pooled"))
        print(line("per-stock HAR (today's fallback)", s, "har"))
    print(f"\nwrote {OUT_PATH}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
