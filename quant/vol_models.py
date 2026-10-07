"""The extended volatility roster: everything HAR has to beat.

HAR on its own is a baseline, not a finding. This module adds the challengers
the literature says should matter, so the Model Confidence Set arbitrates a
crowded field rather than a pair:

  naive       tomorrow = today. The null, present in every comparison.
  har         Corsi's daily/weekly/monthly averages. The incumbent.
  har_l       plus a LEVERAGE term. A fall raises tomorrow's volatility more
              than a rise of the same size, and the sign of the daily return
              is enough to capture that without intraday data.
  har_iv      plus INDIA VIX. The implied-volatility literature is blunt about
              this: a HAR augmented with implied vol beats any HAR without it.
              ^INDIAVIX is free and aligns with the panel, so the usual excuse
              for omitting it does not apply here.
  har_iv_l    both.
  har_ovn     plus the OVERNIGHT gap variance. A range estimator sees only the
              trading session, so it is structurally blind to the close-to-open
              move -- which for Indian equities absorbs a whole session of
              foreign news.
  pooled_har  one HAR fitted across every ticker at once: ~60,000 training
              rows instead of ~1,400. Tests whether volatility dynamics are
              common enough to share.
  gbm         gradient boosting on the same features. The ML-vs-HAR papers
              credit their gains to NON-LINEARITY, so this isolates that claim
              from the extra predictors those papers also had.

WHAT IS DELIBERATELY ABSENT. HARQ and SHAR are the two strongest published
extensions and both need intraday returns -- realized quarticity and
semivariances cannot be formed from a daily bar. They are named in the
limitations rather than approximated badly.

EVERY MODEL SEES THE SAME ROWS. Features that need more history start later,
so the frame is built once and inner-joined; a model is never flattered by
skipping the days it would have found hard.

Run:  python -m quant.vol_models [--estimator parkinson] [--sweep]
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.volatility import (DEFAULT_TOP_N, HAR_LAGS, MIN_TRAIN, REFIT_EVERY,
                              RESULTS, TRADING_DAYS, download_ohlc, mse, qlike,
                              qlike_series, realized_vol)

VIX_SYMBOL = "^INDIAVIX"
VIX_PATH = Path(__file__).parent.parent / "data" / "cache" / "quant" / "india_vix.parquet"

#: Feature columns each model is allowed to use. The target and the naive
#: forecast are handled separately.
FEATURE_SETS = {
    "har":        ["rv_1", "rv_5", "rv_22"],
    "har_l":      ["rv_1", "rv_5", "rv_22", "lev"],
    "har_iv":     ["rv_1", "rv_5", "rv_22", "vix"],
    "har_iv_l":   ["rv_1", "rv_5", "rv_22", "vix", "lev"],
    "har_ovn":    ["rv_1", "rv_5", "rv_22", "ovn"],
    "spline_har": ["rv_1", "rv_5", "rv_22"],
    "pooled_har": ["rv_1", "rv_5", "rv_22"],
    "gbm":        ["rv_1", "rv_5", "rv_22", "lev", "vix", "ovn"],
}
POOLED = {"pooled_har"}
#: Fitted on a natural-cubic-spline expansion of the same three features.
SPLINE = {"spline_har"}
#: Internal knot positions, as quantiles of the TRAINING data only.
SPLINE_KNOTS = (0.05, 0.275, 0.5, 0.725, 0.95)
NONLINEAR = {"gbm"}


# ------------------------------------------------------------------ data
def load_vix(force: bool = False, verbose: bool = True) -> pd.Series:
    """India VIX daily close, cached. Free, and already in the panel's range."""
    if VIX_PATH.exists() and not force:
        return pd.read_parquet(VIX_PATH)["vix"]
    import yfinance as yf

    if verbose:
        print(f"  downloading {VIX_SYMBOL} ...", flush=True)
    raw = yf.download(VIX_SYMBOL, start="2019-01-01", progress=False,
                      auto_adjust=True, threads=False)
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna().rename("vix")
    VIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(VIX_PATH)
    if verbose:
        print(f"  cached {len(s):,} VIX days -> {VIX_PATH.name}")
    return s


def build_features(estimator: str = "parkinson", top_n: int = DEFAULT_TOP_N,
                   include_overnight_in_target: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """One long frame: date, ticker, features, target.

    `include_overnight_in_target` changes what is being FORECAST, not which
    features are available, so a run with it on is not comparable with one
    without -- the loss is computed against a different quantity. It is
    therefore a separate run, never a separate model in the same set.
    """
    ohlc = download_ohlc(top_n=top_n, verbose=verbose)
    rv = realized_vol(ohlc, estimator, verbose=verbose)

    px = ohlc.sort_values(["ticker", "date"]).copy()
    px["prev_close"] = px.groupby("ticker")["close"].shift(1)
    px["ret"] = px["close"] / px["prev_close"] - 1.0
    # overnight gap variance, annualised onto the same scale as rv
    px["ovn_var"] = np.log(px["open"] / px["prev_close"]) ** 2 * TRADING_DAYS

    df = rv.merge(px[["date", "ticker", "ret", "ovn_var"]], on=["date", "ticker"],
                  how="left")
    if include_overnight_in_target:
        df["rv"] = np.sqrt(df["rv"] ** 2 + df["ovn_var"].fillna(0.0))

    vix = load_vix(verbose=verbose)
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    df = df.merge(vix.rename("vix_raw"), left_on="date", right_index=True, how="left")

    out = []
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        for lag in HAR_LAGS:
            g[f"rv_{lag}"] = g["rv"].rolling(lag).mean()
        # leverage: yesterday's volatility, kept only when yesterday fell
        g["lev"] = g["rv"] * (g["ret"] < 0).astype(float)
        g["ovn"] = np.sqrt(g["ovn_var"].clip(lower=0.0)).rolling(5).mean()
        # VIX is a percentage (13.6 == 13.6% a year); put it on rv's scale
        g["vix"] = g["vix_raw"].ffill() / 100.0
        g["target"] = g["rv"].shift(-1)
        g["naive"] = g["rv"]
        out.append(g)

    cols = (["date", "ticker", "target", "naive"]
            + sorted({c for s in FEATURE_SETS.values() for c in s}))
    frame = pd.concat(out, ignore_index=True)[cols].dropna()
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


# ------------------------------------------------------------- estimation
def _ncs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis for one column (Hastie & Tibshirani 5.2.1).

    Natural rather than unrestricted cubic: the fit is forced linear beyond
    the outer knots. Volatility is heavy-tailed, so an unrestricted cubic
    would swing wildly on the handful of extreme days at the edges and those
    days dominate the loss.

    Returns len(knots) - 1 columns: the linear term plus the curvature terms.
    """
    k = np.asarray(knots, dtype=float)
    K = len(k)
    if K < 3:
        return x.reshape(-1, 1)

    def d(j):
        denom = k[K - 1] - k[j]
        if denom <= 0:
            return np.zeros_like(x)
        return (np.maximum(x - k[j], 0.0) ** 3
                - np.maximum(x - k[K - 1], 0.0) ** 3) / denom

    cols = [x]
    dK = d(K - 2)
    for j in range(K - 2):
        cols.append(d(j) - dK)
    return np.column_stack(cols)


def _spline_design(X: np.ndarray, knot_sets: list) -> np.ndarray:
    """Expand every feature column into its own spline basis, additively."""
    return np.column_stack([_ncs_basis(X[:, j], knot_sets[j])
                            for j in range(X.shape[1])])


def _spline_knots(X: np.ndarray) -> list:
    """Knots from the training sample only, de-duplicated.

    Computed inside the walk-forward loop rather than once up front: knots
    placed using the whole sample would be look-ahead bias, since where the
    curve bends would then depend on days not yet observed.
    """
    sets = []
    for j in range(X.shape[1]):
        k = np.unique(np.quantile(X[:, j], SPLINE_KNOTS))
        sets.append(k if len(k) >= 3 else np.unique(np.quantile(X[:, j], (0.1, 0.5, 0.9))))
    return sets


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _ols_apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def walk_forward(frame: pd.DataFrame, model: str, min_train: int = MIN_TRAIN,
                 refit_every: int = REFIT_EVERY, seed: int = 0) -> np.ndarray:
    """Out-of-sample forecasts for one model over the whole frame.

    Per-ticker models are fitted on that ticker's own history; pooled models
    on every ticker's history up to the same calendar date. In both cases the
    cut is by DATE, so no model ever sees an observation from the day it is
    forecasting or later.
    """
    feats = FEATURE_SETS[model]
    preds = np.full(len(frame), np.nan)

    if model in POOLED or model in NONLINEAR:
        dates = np.sort(frame["date"].unique())
        if len(dates) <= min_train:
            return preds
        fitted, last_fit = None, -10 ** 9
        for di in range(min_train, len(dates)):
            day = dates[di]
            if fitted is None or (di - last_fit) >= refit_every:
                tr = frame[frame["date"] < day]
                if len(tr) < 50:
                    continue
                Xtr, ytr = tr[feats].to_numpy(float), tr["target"].to_numpy(float)
                if model in NONLINEAR:
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    fitted = HistGradientBoostingRegressor(
                        max_iter=200, learning_rate=0.06, max_depth=3,
                        random_state=seed).fit(Xtr, ytr)
                else:
                    fitted = _ols_fit(Xtr, ytr)
                last_fit = di
            m = (frame["date"] == day).to_numpy()
            if not m.any() or fitted is None:
                continue
            Xte = frame.loc[m, feats].to_numpy(float)
            preds[m] = (fitted.predict(Xte) if model in NONLINEAR
                        else _ols_apply(fitted, Xte))
        return preds

    is_spline = model in SPLINE
    for ticker, g in frame.groupby("ticker"):
        idx = g.index.to_numpy()
        X, y = g[feats].to_numpy(float), g["target"].to_numpy(float)
        if len(g) <= min_train:
            continue
        beta, knots = None, None
        for i in range(min_train, len(g)):
            if beta is None or (i - min_train) % refit_every == 0:
                if is_spline:
                    knots = _spline_knots(X[:i])
                    beta = _ols_fit(_spline_design(X[:i], knots), y[:i])
                else:
                    beta = _ols_fit(X[:i], y[:i])
            xi = _spline_design(X[i:i + 1], knots) if is_spline else X[i:i + 1]
            preds[idx[i]] = float(_ols_apply(beta, xi)[0])
    return preds


# ------------------------------------------------------------ evaluation
def evaluate(estimator: str = "parkinson", top_n: int = DEFAULT_TOP_N,
             overnight_target: bool = False, min_train: int = MIN_TRAIN,
             refit_every: int = REFIT_EVERY, verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    frame = build_features(estimator, top_n, overnight_target, verbose=verbose)
    if verbose:
        print(f"  {len(frame):,} ticker-days, {frame['ticker'].nunique()} tickers, "
              f"{frame['date'].min().date()} -> {frame['date'].max().date()}")

    cols = {"naive": frame["naive"].to_numpy(float)}
    for model in FEATURE_SETS:
        if verbose:
            print(f"    fitting {model} ...", flush=True)
        cols[model] = walk_forward(frame, model, min_train, refit_every)

    scored = frame.copy()
    for k, v in cols.items():
        scored[k] = v
    models = list(cols)
    scored = scored.dropna(subset=models).reset_index(drop=True)
    actual = scored["target"].to_numpy(float)

    summary = {m: {"qlike": qlike(actual, scored[m].to_numpy(float)),
                   "mse": mse(actual, scored[m].to_numpy(float))} for m in models}
    losses = {m: qlike_series(actual, scored[m].to_numpy(float)) for m in models}

    from quant.mcs import model_confidence_set
    mcs = model_confidence_set(losses, verbose=verbose)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estimator": estimator, "overnight_in_target": overnight_target,
        "protocol": {"min_train": min_train, "refit_every": refit_every,
                     "walk_forward": "expanding, cut by date, strictly causal"},
        "n_forecasts": int(len(scored)), "n_tickers": int(scored["ticker"].nunique()),
        "span": [str(scored["date"].min())[:10], str(scored["date"].max())[:10]],
        "summary": summary, "mcs": mcs,
    }

    if verbose:
        print(f"\n{'=' * 60}\n{estimator}"
              f"{'  (+overnight in target)' if overnight_target else ''}  "
              f"{len(scored):,} forecasts\n{'=' * 60}")
        print(f"  {'model':<12}{'QLIKE':>10}{'MSE':>11}")
        for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
            star = "  <- MCS" if m in mcs["mcs"] else ""
            print(f"  {m:<12}{v['qlike']:>10.5f}{v['mse']:>11.6f}{star}")
        print(f"\n  MCS (95%): {mcs['mcs']}")
    return out


def sweep(estimator: str = "parkinson", verbose: bool = True) -> dict:
    """Fitting-scheme sweep: does the WINDOW matter more than the MODEL?

    Published work argues that training-window length and refit frequency
    decide more than model class does, and that machine learning stops
    beating HAR once HAR's fitting scheme is specified properly. This project
    chose 500/21 by convention and never tested it, so the claim is checked
    here rather than assumed.
    """
    grid = [(250, 21), (500, 21), (750, 21), (1000, 21),
            (500, 5), (500, 63), (500, 252)]
    rows = []
    for mt, re_ in grid:
        if verbose:
            print(f"\n--- min_train={mt}, refit_every={re_} ---", flush=True)
        r = evaluate(estimator, min_train=mt, refit_every=re_, verbose=False)
        best = min(r["summary"], key=lambda m: r["summary"][m]["qlike"])
        rows.append({"min_train": mt, "refit_every": re_,
                     "n_forecasts": r["n_forecasts"],
                     "har_qlike": r["summary"]["har"]["qlike"],
                     "best_model": best,
                     "best_qlike": r["summary"][best]["qlike"],
                     "mcs": r["mcs"]["mcs"]})
        if verbose:
            print(f"    har {rows[-1]['har_qlike']:.5f}   best {best} "
                  f"{rows[-1]['best_qlike']:.5f}   MCS {rows[-1]['mcs']}")

    spread = max(r["har_qlike"] for r in rows) - min(r["har_qlike"] for r in rows)
    return {"grid": rows, "har_qlike_spread_across_schemes": spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="parkinson")
    ap.add_argument("--overnight-target", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        res = sweep(args.estimator)
        path = RESULTS / "volatility_fitting_sweep.json"
        path.write_text(json.dumps(res, indent=2, default=str))
        print(f"\nHAR QLIKE spread across fitting schemes: "
              f"{res['har_qlike_spread_across_schemes']:.5f}")
        print(f"wrote {path}")
        return

    res = evaluate(args.estimator, overnight_target=args.overnight_target)
    suffix = "_ovn" if args.overnight_target else ""
    path = RESULTS / f"volatility_roster_{args.estimator}{suffix}.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
