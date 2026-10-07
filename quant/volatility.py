"""Realized-volatility target and the HAR baselines it has to beat.

WHY THIS EXISTS. Every price forecast in this project loses to "tomorrow
equals today". That is not a bug in the models -- it is the textbook result.
Returns are approximately a random walk; their VOLATILITY is the documented
exception, strongly autocorrelated and genuinely forecastable. So the target
changes rather than the model: this module forecasts next-day realized
volatility, where a model can actually earn its skill value.

THE PROXY. True realized volatility needs 5-minute intraday bars. This
project uses free daily data only, so volatility is estimated from the daily
OHLC range instead. Range estimators are far more efficient than squared
close-to-close returns -- a day that opens at 100, runs to 110, falls to 95
and closes at 100 has zero close-to-close return and was obviously not a
quiet day.

    Parkinson      uses the high-low range only
    Garman-Klass   adds the open-close move, lower variance than Parkinson
    Rogers-Satchell handles a drifting price, which the other two assume away

THE BASELINES, PRE-REGISTERED BEFORE ANY MODEL WAS RUN. HAR (Corsi) is a
plain OLS regression of tomorrow's volatility on the average of the last
day, week and month. It is trivial and it is hard to beat; the published
comparisons that put foundation models against it find gains of a couple of
percent at best. Log-HAR fits the same regression in logs, which handles the
right skew of volatility. `naive` carries tomorrow's forecast as today's
value, exactly as the price log does, so the null model is present from the
start rather than added after the fact.

METRICS. MSE on a noisy volatility proxy is dominated by a handful of
extreme days, so QLIKE is reported alongside it. QLIKE is robust to noise in
the proxy and penalises under-forecasting of risk harder than
over-forecasting, which is the asymmetry a risk user actually cares about.

Run:  python -m quant.volatility --refresh      (download + cache OHLC)
      python -m quant.volatility                 (fit and evaluate)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "quant"
OHLC_PATH = CACHE_DIR / "ohlc_panel.parquet"
RESULTS = Path(__file__).parent.parent / "evaluation" / "results"

DEFAULT_TOP_N = 50          # matches the anomaly benchmark's equity universe
DEFAULT_START = "2019-01-01"
TRADING_DAYS = 252

#: HAR's three horizons, in trading days. Corsi's original choice: one day,
#: one week, one month. Not tuned -- tuning them per ticker is how a
#: reporting budget of 1 quietly becomes a budget of 50.
HAR_LAGS = (1, 5, 22)

#: Days of history before the first out-of-sample forecast, and how often the
#: model is refitted. Refitting daily is ~20x the cost for no measurable
#: change in a 3-parameter OLS.
MIN_TRAIN = 500
REFIT_EVERY = 21


# ------------------------------------------------------------------ data
def download_ohlc(top_n: int = DEFAULT_TOP_N, start: str = DEFAULT_START,
                  force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Open/High/Low/Close for the most liquid names, long-format, cached.

    Liquidity is measured on the EXISTING cached panel, so the universe here
    is the same one the rest of the project uses rather than a new choice.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if OHLC_PATH.exists() and not force:
        return pd.read_parquet(OHLC_PATH)

    import yfinance as yf

    from quant.data import load_panel, traded_value

    close, volume = load_panel()
    tv = traded_value(close, volume)
    symbols = list(tv.mean().nlargest(top_n).index)
    if verbose:
        print(f"  downloading OHLC for {len(symbols)} most liquid names "
              f"from {start} ...", flush=True)

    raw = yf.download(symbols, start=start, progress=False,
                      auto_adjust=True, threads=True)
    frames = []
    for field in ("Open", "High", "Low", "Close"):
        f = raw[field].stack().rename(field.lower())
        frames.append(f)
    df = pd.concat(frames, axis=1).reset_index()
    df.columns = ["date", "ticker", "open", "high", "low", "close"]
    df = df.dropna()
    df.to_parquet(OHLC_PATH, index=False)
    if verbose:
        print(f"  cached {len(df):,} ticker-days -> {OHLC_PATH.name}")
    return df


# ------------------------------------------------- volatility estimators
def parkinson(df: pd.DataFrame) -> pd.Series:
    """High-low range estimator. Variance, in daily units."""
    return (np.log(df["high"] / df["low"]) ** 2) / (4.0 * np.log(2.0))


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Range plus open-close. Lower variance than Parkinson."""
    hl = np.log(df["high"] / df["low"]) ** 2
    co = np.log(df["close"] / df["open"]) ** 2
    return 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co


def rogers_satchell(df: pd.DataFrame) -> pd.Series:
    """Drift-robust: unlike the other two it does not assume zero mean."""
    ho, lo = np.log(df["high"] / df["open"]), np.log(df["low"] / df["open"])
    hc, lc = np.log(df["high"] / df["close"]), np.log(df["low"] / df["close"])
    return ho * hc + lo * lc


ESTIMATORS = {"parkinson": parkinson, "garman_klass": garman_klass,
              "rogers_satchell": rogers_satchell}


def realized_vol(df: pd.DataFrame, estimator: str = "parkinson",
                 verbose: bool = False) -> pd.DataFrame:
    """Per-ticker daily realized volatility (annualised standard deviation).

    Returned as volatility rather than variance because that is the unit a
    reader can sanity-check: 0.25 is a 25%-a-year stock.

    ZERO-RANGE DAYS ARE DROPPED, NOT FLOORED. 383 of 86,094 ticker-days in
    this panel have high == low -- a circuit limit, an untraded name, or a
    vendor artifact. On those days the estimator is UNDEFINED, not small.
    Clipping them to a tiny positive number instead made the naive forecast
    carry ~0 volatility into the next day, and QLIKE, which divides by the
    forecast, exploded to 6.5e6 and swamped every honest comparison.
    """
    valid = df["high"] > df["low"]
    dropped = int((~valid).sum())
    d = df.loc[valid]
    var = ESTIMATORS[estimator](d)
    out = d[["date", "ticker"]].copy()
    out["rv"] = np.sqrt(var.clip(lower=0.0) * TRADING_DAYS)
    out = out[out["rv"] > 0.0]
    if verbose and dropped:
        print(f"  dropped {dropped:,} zero-range ticker-days "
              f"({dropped / len(df):.2%}) -- estimator undefined")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


# ------------------------------------------------------------------ HAR
def har_design(rv: pd.Series, lags: tuple = HAR_LAGS) -> pd.DataFrame:
    """Daily / weekly / monthly trailing averages, all strictly causal.

    Every column at row t uses observations up to and including t, and the
    target is t+1, so nothing about the forecast day enters its own features.
    """
    X = pd.DataFrame(index=rv.index)
    for lag in lags:
        X[f"rv_{lag}"] = rv.rolling(lag).mean()
    X["target"] = rv.shift(-1)
    return X


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _apply(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def walk_forward_ticker(rv: pd.Series, log_space: bool = False,
                        min_train: int = MIN_TRAIN,
                        refit_every: int = REFIT_EVERY) -> pd.DataFrame:
    """Expanding-window out-of-sample HAR forecasts for one ticker.

    The model never sees a day at or beyond the one it is forecasting: it is
    refitted on data up to day t and predicts t+1.
    """
    design = har_design(rv).dropna()
    if len(design) <= min_train + 1:
        return pd.DataFrame()

    feat_cols = [c for c in design.columns if c != "target"]
    Xall = design[feat_cols].to_numpy(dtype=float)
    yall = design["target"].to_numpy(dtype=float)
    dates = design.index.to_numpy()

    preds = np.full(len(design), np.nan)
    beta = None
    for i in range(min_train, len(design)):
        if beta is None or (i - min_train) % refit_every == 0:
            Xtr, ytr = Xall[:i], yall[:i]
            if log_space:
                beta = _ols(np.log(Xtr), np.log(ytr))
            else:
                beta = _ols(Xtr, ytr)
        xi = Xall[i:i + 1]
        if log_space:
            # Exponentiating a log-space fit is biased low; the standard
            # smearing correction uses the fitted residual variance.
            resid = np.log(yall[:i]) - _apply(beta, np.log(Xall[:i]))
            preds[i] = float(np.exp(_apply(beta, np.log(xi))[0] + 0.5 * resid.var()))
        else:
            preds[i] = float(_apply(beta, xi)[0])

    out = pd.DataFrame({"date": dates, "actual": yall, "pred": preds,
                        "naive": Xall[:, 0]})          # rv_1 == today's RV
    return out.dropna().reset_index(drop=True)


# -------------------------------------------------------------- metrics
def qlike(actual: np.ndarray, pred: np.ndarray) -> float:
    """Robust to a noisy proxy, and asymmetric: under-forecasting risk hurts
    more than over-forecasting it."""
    a = np.clip(actual, 1e-12, None) ** 2
    p = np.clip(pred, 1e-12, None) ** 2
    r = a / p
    return float(np.mean(r - np.log(r) - 1.0))


def mse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((actual - pred) ** 2))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> dict:
    """Equal predictive accuracy of A vs B, with the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic favours A."""
    from scipy import stats

    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 10:
        return {"n": n, "stat": None, "p_value": None}
    dbar = d.mean()
    gamma0 = d.var(ddof=0)
    acc = gamma0
    for lag in range(1, h):
        g = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        acc += 2.0 * g
    if acc <= 0:
        return {"n": n, "stat": None, "p_value": None}
    stat = dbar / np.sqrt(acc / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= corr
    p = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return {"n": n, "mean_loss_diff": float(dbar), "stat": float(stat),
            "p_value": float(p)}


__all__ = ["download_ohlc", "realized_vol", "walk_forward_ticker",
           "qlike", "mse", "diebold_mariano", "ESTIMATORS"]


# ----------------------------------------------------------- evaluation
def run(estimator: str = "parkinson", top_n: int = DEFAULT_TOP_N,
        verbose: bool = True) -> dict:
    """Walk-forward HAR, Log-HAR and naive on every ticker, then compare.

    Aggregated EQUAL-WEIGHTED across tickers as well as pooled. Pooled loss
    on a volatility proxy is dominated by whichever few names had the wildest
    days, so it answers "which model handled the outliers" rather than
    "which model is better", which is the question being asked.
    """
    df = download_ohlc(top_n=top_n, verbose=verbose)
    rv_all = realized_vol(df, estimator, verbose=verbose)

    rows, per_ticker = [], []
    for ticker, g in rv_all.groupby("ticker"):
        s = g.set_index("date")["rv"].dropna()
        har = walk_forward_ticker(s, log_space=False)
        if har.empty:
            continue
        log_har = walk_forward_ticker(s, log_space=True)
        if log_har.empty or len(log_har) != len(har):
            continue
        rec = har[["date", "actual", "naive"]].copy()
        rec["har"] = har["pred"].to_numpy()
        rec["log_har"] = log_har["pred"].to_numpy()
        rec["ticker"] = ticker
        rows.append(rec)
        per_ticker.append({
            "ticker": ticker, "n": int(len(rec)),
            **{f"qlike_{m}": qlike(rec["actual"].to_numpy(), rec[m].to_numpy())
               for m in ("naive", "har", "log_har")},
            **{f"mse_{m}": mse(rec["actual"].to_numpy(), rec[m].to_numpy())
               for m in ("naive", "har", "log_har")},
        })

    if not rows:
        raise SystemExit("no ticker produced enough history to evaluate")
    allrec = pd.concat(rows, ignore_index=True)
    pt = pd.DataFrame(per_ticker)
    models = ("naive", "har", "log_har")

    summary = {m: {
        "qlike_equal_weighted": float(pt[f"qlike_{m}"].mean()),
        "mse_equal_weighted": float(pt[f"mse_{m}"].mean()),
        "qlike_pooled": qlike(allrec["actual"].to_numpy(), allrec[m].to_numpy()),
        "mse_pooled": mse(allrec["actual"].to_numpy(), allrec[m].to_numpy()),
    } for m in models}

    # Diebold-Mariano on the pooled per-observation QLIKE loss series.
    def loss(m):
        a = np.clip(allrec["actual"].to_numpy(), 1e-12, None) ** 2
        p = np.clip(allrec[m].to_numpy(), 1e-12, None) ** 2
        r = a / p
        return r - np.log(r) - 1.0

    dm = {}
    for a, b in (("har", "naive"), ("log_har", "naive"), ("log_har", "har")):
        dm[f"{a}_vs_{b}"] = diebold_mariano(loss(a), loss(b))

    wins = {m: int((pt[f"qlike_{m}"] == pt[[f"qlike_{x}" for x in models]].min(axis=1)).sum())
            for m in models}

    out = {
        "estimator": estimator, "n_tickers": int(pt["ticker"].nunique()),
        "n_forecasts": int(len(allrec)),
        "span": [str(allrec["date"].min())[:10], str(allrec["date"].max())[:10]],
        "protocol": {"min_train": MIN_TRAIN, "refit_every": REFIT_EVERY,
                     "har_lags": list(HAR_LAGS), "horizon_days": 1,
                     "walk_forward": "expanding window, strictly causal"},
        "summary": summary, "diebold_mariano": dm,
        "per_ticker_qlike_wins": wins,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "per_ticker": per_ticker,
    }

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"REALIZED VOLATILITY, 1-day ahead  ({estimator}, {out['n_tickers']} tickers, "
              f"{out['n_forecasts']:,} forecasts)")
        print(f"{'=' * 72}")
        print(f"  {'model':<10}{'QLIKE (eq-wt)':>15}{'QLIKE (pooled)':>16}"
              f"{'MSE (eq-wt)':>14}{'best on':>10}")
        for m in models:
            print(f"  {m:<10}{summary[m]['qlike_equal_weighted']:>15.5f}"
                  f"{summary[m]['qlike_pooled']:>16.5f}"
                  f"{summary[m]['mse_equal_weighted']:>14.6f}"
                  f"{wins[m]:>7} /{out['n_tickers']:>3}")
        print("\n  Diebold-Mariano on pooled QLIKE (negative stat favours the first):")
        for k, v in dm.items():
            if v.get("stat") is None:
                print(f"    {k:<22} n/a")
            else:
                verdict = "significant" if v["p_value"] < 0.05 else "not significant"
                print(f"    {k:<22} stat {v['stat']:+8.3f}   p = {v['p_value']:.4g}   {verdict}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"volatility_har_{estimator}.json"
    path.write_text(json.dumps(out, indent=2))
    if verbose:
        print(f"\nwrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="parkinson", choices=list(ESTIMATORS))
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--refresh", action="store_true", help="re-download OHLC")
    args = ap.parse_args()
    if args.refresh:
        download_ohlc(top_n=args.top_n, force=True)
    run(estimator=args.estimator, top_n=args.top_n)


if __name__ == "__main__":
    main()
