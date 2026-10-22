"""Can the volatility forecast be used to predict PRICE?

Not the point, and this module measures that rather than assuming it: the
forecast says how big tomorrow's move will be and carries no information
about its sign, so the best point forecast of tomorrow's close stays
today's close.

What it can do is set the WIDTH of the range around that point. Every price
model already in the project publishes an interval, and an interval is only
as good as its volatility: a band built from a stale or constant volatility
is too wide on calm days and too narrow on wild ones, even when its average
coverage is perfect. So the comparison here is between price intervals that
share one point forecast and differ only in where the volatility comes from:

    constant      the stock's own long-run volatility
    hist60        trailing 60-day standard deviation of returns
    ewma          RiskMetrics, lambda 0.94
    garch         GARCH(1,1), the standard conditional-volatility model
    range_today   today's Garman-Klass range, carried forward
    ens_log       the served log-space ensemble

ONE CALIBRATION RECIPE FOR ALL SIX, so that none wins on calibration effort.
A volatility source s is turned into an interval for tomorrow's log return by

    r in [ q_lo * k * s ,  q_hi * k * s ]

where k is the stock's own scale -- needed because a range estimator sees
only the trading session and a close-to-close return also contains the
overnight gap -- and q are empirical quantiles of the standardised return
r / (k s), pooled across stocks. Both come ONLY from days before the one
being forecast. Every source therefore hits its nominal coverage on average
by construction, and what is left to compare is the thing that matters:
how narrow the band is for that coverage, and whether coverage holds on
calm and wild days separately. The Winkler interval score rewards both.

Run:  python -m quant.price_intervals
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import OHLC_PATH, RESULTS, TRADING_DAYS, diebold_mariano

#: har_stock was added after the first comparison had already been reported:
#: the per-stock HAR has a lower QLIKE than the ensemble, so leaving it out
#: meant the "narrowest band" claim was never tested against the strongest
#: rival on the risk loss.
SOURCES = ("constant", "hist60", "ewma", "garch", "range_today", "har_stock", "ens_log")
LEVELS = {"80": (0.10, 0.90), "95": (0.025, 0.975)}
MIN_HISTORY = 250          # a stock's own prior days before its scale is trusted
REFRESH_DAYS = 21          # how often the pooled quantiles are re-read
EWMA_LAMBDA = 0.94
OUT_PATH = RESULTS / "price_intervals.json"


def build_frame() -> pd.DataFrame:
    """One row per stock-day: today's close, tomorrow's return, six volatilities."""
    from quant.tracking_models import build_tracking_frame

    px = pd.read_parquet(OHLC_PATH).sort_values(["ticker", "date"])
    g = px.groupby("ticker")["close"]
    px["ret"] = np.log(px["close"] / g.shift(1))
    px["close_next"] = g.shift(-1)
    px["ret_next"] = np.log(px["close_next"] / px["close"])
    by = px.groupby("ticker")["ret"]
    px["hist60"] = by.transform(lambda r: r.rolling(60).std()) * np.sqrt(TRADING_DAYS)
    px["ewma"] = by.transform(
        lambda r: np.sqrt((r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False,
                                       min_periods=60).mean())) * np.sqrt(TRADING_DAYS)
    px["constant"] = 1.0

    pred = pd.read_parquet(RESULTS / "volatility_tracking_predictions.parquet")
    feat = build_tracking_frame("garman_klass", verbose=False, require_target=True)
    garch = pd.read_parquet(RESULTS / "garch_carr_forecasts.parquet")

    from evaluation.out_of_panel import per_stock_har

    feat = feat.reset_index(drop=True)
    feat["har_stock"] = np.exp(per_stock_har(feat))      # walk-forward, per stock
    d = (pred[["date", "ticker", "ens_log", "target"]]
         .merge(feat[["date", "ticker", "naive", "har_stock"]]
                .rename(columns={"naive": "range_today"}), on=["date", "ticker"])
         .merge(garch[["date", "ticker", "garch"]], on=["date", "ticker"], how="left")
         .merge(px[["date", "ticker", "close", "close_next", "ret_next",
                    "hist60", "ewma", "constant"]], on=["date", "ticker"]))
    d = d.dropna(subset=["ret_next", *SOURCES])
    return d.sort_values(["date", "ticker"]).reset_index(drop=True)


def calibrate(d: pd.DataFrame, source: str) -> pd.DataFrame:
    """Causal interval bounds, in log-return units, for one volatility source.

    A row dated t holds the return from t to t+1, which is known at t+1. So
    a forecast made on day t may use rows dated strictly before t, and that
    is the only rule this function has to keep.
    """
    z = (d["ret_next"] / d[source]).to_numpy(float)
    grp = d.groupby("ticker", sort=False)
    # the stock's own scale: root mean square of z over its earlier rows
    prior_n = grp.cumcount().to_numpy()
    prior_ss = (pd.Series(z ** 2, index=d.index).groupby(d["ticker"], sort=False).cumsum()
                .to_numpy() - z ** 2)
    k = np.sqrt(np.where(prior_n >= MIN_HISTORY, prior_ss / np.maximum(prior_n, 1), np.nan))
    zs = z / k                                     # standardised, causally

    dates = d["date"].to_numpy()
    uniq = np.sort(np.unique(dates))
    levels = sorted({q for pair in LEVELS.values() for q in pair})
    q = np.full((len(d), len(levels)), np.nan)
    for start in range(0, len(uniq), REFRESH_DAYS):
        block = uniq[start:start + REFRESH_DAYS]
        past = zs[(dates < block[0]) & np.isfinite(zs)]
        if len(past) < 2000:
            continue
        q[np.isin(dates, block)] = np.quantile(past, levels)

    out = pd.DataFrame(index=d.index)
    scale = k * d[source].to_numpy(float)
    for name, (lo, hi) in LEVELS.items():
        out[f"lo{name}"] = q[:, levels.index(lo)] * scale
        out[f"hi{name}"] = q[:, levels.index(hi)] * scale
    return out


def winkler(lo, hi, y, alpha):
    """Interval score: width, plus a penalty of 2/alpha per unit of miss."""
    return (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def score(d: pd.DataFrame, bounds: dict) -> dict:
    y = d["ret_next"].to_numpy(float) * 100            # percent
    # calm / middle / wild by how volatile the forecast day actually was
    bucket = pd.qcut(d["target"], 3, labels=["calm", "middle", "wild"]).to_numpy()
    out = {}
    for src, b in bounds.items():
        entry = {}
        for name, (qlo, qhi) in LEVELS.items():
            lo, hi = b[f"lo{name}"].to_numpy() * 100, b[f"hi{name}"].to_numpy() * 100
            inside = (y >= lo) & (y <= hi)
            w = winkler(lo, hi, y, 1 - (qhi - qlo))
            entry[name] = {
                "coverage": float(inside.mean()),
                "mean_width_pct": float((hi - lo).mean()),
                "winkler": float(w.mean()),
                "coverage_by_day_type": {k: float(inside[bucket == k].mean())
                                         for k in ("calm", "middle", "wild")},
                "width_tracks_move": float(pd.Series(hi - lo).corr(
                    pd.Series(np.abs(y)), method="spearman")),
            }
            entry[name]["_daily_winkler"] = pd.Series(w).groupby(d["date"].to_numpy()).mean()
        out[src] = entry
    return out


def point_evidence(d: pd.DataFrame) -> dict:
    """Does the volatility forecast say anything about WHERE the price goes?"""
    from scipy.stats import spearmanr

    r, v = d["ret_next"], np.log(d["ens_log"])
    daily = d.assign(v=v).groupby("date").apply(
        lambda g: pd.Series({"sign": spearmanr(g["v"], g["ret_next"])[0],
                             "size": spearmanr(g["v"], g["ret_next"].abs())[0]})
        if len(g) > 10 else pd.Series({"sign": np.nan, "size": np.nan})).dropna()
    t_sign = float(daily["sign"].mean() / (daily["sign"].std() / np.sqrt(len(daily))))
    high = v > v.groupby(d["date"]).transform("median")
    return {
        "corr_with_return": float(spearmanr(v, r)[0]),
        "corr_with_abs_return": float(spearmanr(v, r.abs())[0]),
        "daily_cross_sectional_corr_with_return": float(daily["sign"].mean()),
        "daily_cross_sectional_t_stat": t_sign,
        "daily_cross_sectional_corr_with_abs_return": float(daily["size"].mean()),
        "share_up_when_forecast_vol_high": float((r[high] > 0).mean()),
        "share_up_when_forecast_vol_low": float((r[~high] > 0).mean()),
        "mean_abs_move_pct_high_vs_low": [float(r[high].abs().mean() * 100),
                                          float(r[~high].abs().mean() * 100)],
    }


def against_forecast_log(d: pd.DataFrame, bounds: pd.DataFrame) -> dict:
    """The same days the GP and ARIMA price models were scored on.

    Small -- the log holds a few tickers over a few months -- so this is a
    sanity check on the same rows, not a ranking.
    """
    from finance.forecast_log import load_log

    log = load_log()
    if log.empty:
        return {"status": "forecast log is empty"}
    log = log[log["actual_close"].notna()].rename(columns={"as_of_date": "date"})
    mine = d[["date", "ticker", "close"]].join(bounds[["lo80", "hi80"]])
    m = log.merge(mine, on=["date", "ticker"], how="inner").dropna(subset=["lo80"])
    if m.empty:
        return {"status": "no overlapping rows"}
    actual = m["actual_close"].to_numpy(float)
    ref = m["close_today"].to_numpy(float)
    cand = {"volatility_interval": (ref * np.exp(m["lo80"]), ref * np.exp(m["hi80"])),
            "gpr": (m["gpr_lower80"], m["gpr_upper80"]),
            "arima": (m["arima_lower80"], m["arima_upper80"])}
    out = {"n": int(len(m)), "tickers": sorted(m["ticker"].unique()),
           "span": f"{m['date'].min().date()} -> {m['date'].max().date()}", "models": {}}
    for name, (lo, hi) in cand.items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        rel = 100 / ref[ok]
        out["models"][name] = {
            "n": int(ok.sum()),
            "coverage": float(((actual[ok] >= lo[ok]) & (actual[ok] <= hi[ok])).mean()),
            "mean_width_pct_of_price": float(((hi[ok] - lo[ok]) * rel).mean()),
            "winkler_pct_of_price": float((winkler(lo[ok], hi[ok], actual[ok], 0.2)
                                           * rel).mean())}
    out["point_mae_pct_of_price"] = {
        "today's close": float((np.abs(actual - ref) / ref).mean() * 100),
        "gpr": float((m["gpr_abs_err"] / ref).mean() * 100),
        "arima": float((m["arima_abs_err"] / ref).mean() * 100)}
    return out


def serving_calibration(source: str = "ens_log") -> dict:
    """Each stock's scale and the pooled quantiles, from every held-out day.

    Uses all of the history at once, which is causal for what it serves: a
    forecast made after the last of these days. Needs only the volatility
    forecast and the returns, so it covers every stock in the panel rather
    than the 45 that also have a GARCH series for the comparison.
    """
    px = pd.read_parquet(OHLC_PATH).sort_values(["ticker", "date"])
    px["ret_next"] = np.log(px.groupby("ticker")["close"].shift(-1) / px["close"])
    pred = pd.read_parquet(RESULTS / "volatility_tracking_predictions.parquet")
    d = pred[["date", "ticker", source]].merge(
        px[["date", "ticker", "ret_next"]], on=["date", "ticker"]).dropna()
    z = d["ret_next"] / d[source]
    pooled = float(np.sqrt((z ** 2).mean()))
    n = z.groupby(d["ticker"]).size()
    k = np.sqrt((z ** 2).groupby(d["ticker"]).mean()).where(n >= MIN_HISTORY, pooled)
    zs = z / d["ticker"].map(k)
    levels = sorted({q for pair in LEVELS.values() for q in pair})
    return {"source": source, "through": str(d["date"].max().date()), "n": int(len(d)),
            "scale": {t: float(v) for t, v in k.items()},
            "scale_is_pooled": sorted(n.index[n < MIN_HISTORY]),
            "quantiles": {str(q): float(v) for q, v in zip(levels, np.quantile(zs, levels))}}


def forecast(ticker: str) -> dict:
    """Next-session price range for one stock: today's close, and how far from it.

    The point is today's close and is labelled as such. The volatility
    forecast contributes the width and nothing else, because it carries no
    information about direction (measured: correlation with the next return
    +0.016, 50.0% up days whether the forecast is high or low).
    """
    from quant.production_model import forecast as vol_forecast

    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH.name} not found. Run `python -m quant.price_intervals`.")
    res = json.loads(OUT_PATH.read_text())
    cal = res["serving_calibration"]
    v = vol_forecast(ticker)
    if ticker not in cal["scale"]:
        raise ValueError(f"{ticker!r} has no calibration; re-run `python -m quant.price_intervals`.")
    px = pd.read_parquet(OHLC_PATH)
    row = px[(px["ticker"] == ticker) & (px["date"] == pd.Timestamp(v["as_of"]))]
    close = float(row["close"].iloc[0])
    width = cal["scale"][ticker] * v["forecast_annualised_vol"]
    q = cal["quantiles"]

    def band(lo, hi):
        return [close * float(np.exp(q[str(lo)] * width)), close * float(np.exp(q[str(hi)] * width))]

    best = res["sources"]["ens_log"]

    def next_best(level):
        return min(v[level]["winkler"] for s, v in res["sources"].items() if s != "ens_log")
    return {
        "ticker": ticker, "model": "ens_log price range", "as_of": v["as_of"],
        "close": close, "point": close,
        "point_is": "today's close -- no model here predicts direction",
        "interval_80": band(*LEVELS["80"]), "interval_95": band(*LEVELS["95"]),
        "forecast_annualised_vol": v["forecast_annualised_vol"],
        "is_live": v["is_live"], "horizon_sessions": 1,
        "scale_is_pooled": ticker in cal["scale_is_pooled"],
        "skill": {"coverage_80": best["80"]["coverage"], "coverage_95": best["95"]["coverage"],
                  "winkler_80": best["80"]["winkler"], "winkler_80_next_best": next_best("80"),
                  "winkler_95": best["95"]["winkler"], "winkler_95_next_best": next_best("95"),
                  "coverage_80_on_wild_days": best["80"]["coverage_by_day_type"]["wild"],
                  "coverage_95_on_wild_days": best["95"]["coverage_by_day_type"]["wild"],
                  "corr_with_next_return": res["point"]["corr_with_return"],
                  "share_up_when_vol_high": res["point"]["share_up_when_forecast_vol_high"],
                  "share_up_when_vol_low": res["point"]["share_up_when_forecast_vol_low"],
                  "n_heldout_forecasts": res["n_forecasts"]},
    }


def main():
    warnings.filterwarnings("ignore")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    args = ap.parse_args()
    if args.ticker:
        r = forecast(args.ticker)
        print(f"{r['ticker']}  as of {r['as_of']}  close {r['close']:.2f}  (live={r['is_live']})")
        print(f"  point          {r['point']:.2f}   ({r['point_is']})")
        for lv in ("80", "95"):
            lo, hi = r[f"interval_{lv}"]
            print(f"  {lv}% range      {lo:.2f} to {hi:.2f}   "
                  f"({(lo / r['close'] - 1):+.2%} to {(hi / r['close'] - 1):+.2%})")
        return
    d = build_frame()
    bounds = {s: calibrate(d, s) for s in SOURCES}
    ok = np.ones(len(d), bool)
    for b in bounds.values():
        ok &= b.notna().all(axis=1).to_numpy()
    full_bounds = bounds["ens_log"]
    e = d[ok].reset_index(drop=True)
    eb = {s: b[ok].reset_index(drop=True) for s, b in bounds.items()}
    sc = score(e, eb)

    dm = {}
    for name in LEVELS:
        mine = sc["ens_log"][name]["_daily_winkler"]
        dm[name] = {s: diebold_mariano(mine.to_numpy(), sc[s][name]["_daily_winkler"]
                                       .reindex(mine.index).to_numpy())
                    for s in SOURCES if s != "ens_log"}
    for s in sc.values():
        for lv in s.values():
            lv.pop("_daily_winkler")

    # Is the win broad, or carried by a few stocks or one year?
    y = e["ret_next"].to_numpy(float)
    robust = {}
    for name, (qlo, qhi) in LEVELS.items():
        w = pd.DataFrame({s: winkler(eb[s][f"lo{name}"].to_numpy(), eb[s][f"hi{name}"].to_numpy(),
                                     y, 1 - (qhi - qlo)) for s in SOURCES})
        per_ticker = w.groupby(e["ticker"].to_numpy()).mean()
        per_year = w.groupby(e["date"].dt.year.to_numpy()).mean()
        others = [s for s in SOURCES if s != "ens_log"]
        robust[name] = {
            "tickers_where_ens_log_beats": {
                s: int((per_ticker["ens_log"] < per_ticker[s]).sum()) for s in others},
            "tickers_where_ens_log_is_best": int(
                (per_ticker.idxmin(axis=1) == "ens_log").sum()),
            "n_tickers": int(len(per_ticker)),
            "years_where_ens_log_is_best": [int(yr) for yr in per_year.index
                                            if per_year.loc[yr].idxmin() == "ens_log"],
            "years": [int(yr) for yr in per_year.index]}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_forecasts": int(len(e)), "n_tickers": int(e["ticker"].nunique()),
           "n_days": int(e["date"].nunique()),
           "span": f"{e['date'].min().date()} -> {e['date'].max().date()}",
           "point_forecast": "today's close, for every source",
           "calibration": {"min_history": MIN_HISTORY, "refresh_days": REFRESH_DAYS},
           "sources": sc, "diebold_mariano_ens_log_vs": dm, "robustness": robust,
           "point": point_evidence(e), "serving_calibration": serving_calibration(),
           "forecast_log": against_forecast_log(d, full_bounds)}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"PRICE INTERVALS  {out['n_forecasts']:,} forecasts, {out['n_tickers']} tickers, "
          f"{out['span']}\n(next-session log return, percent; point = today's close)\n")
    for name in LEVELS:
        print(f"{name}% interval")
        print(f"  {'volatility from':<14}{'coverage':>9}{'width':>8}{'Winkler':>9}"
              f"{'calm':>8}{'middle':>8}{'wild':>8}{'tracks':>8}{'DM p':>9}")
        for s in sorted(SOURCES, key=lambda s: sc[s][name]["winkler"]):
            v = sc[s][name]
            c = v["coverage_by_day_type"]
            p = dm[name].get(s, {}).get("p_value")
            print(f"  {s:<14}{v['coverage']:>9.3f}{v['mean_width_pct']:>8.2f}"
                  f"{v['winkler']:>9.3f}{c['calm']:>8.3f}{c['middle']:>8.3f}{c['wild']:>8.3f}"
                  f"{v['width_tracks_move']:>8.3f}"
                  + (f"{p:>9.2g}" if p is not None else f"{'-':>9}"))
        rb = robust[name]
        print(f"  ens_log is the best of the {len(SOURCES)} on {rb['tickers_where_ens_log_is_best']} of "
              f"{rb['n_tickers']} stocks and in {len(rb['years_where_ens_log_is_best'])} of "
              f"{len(rb['years'])} years; beats constant on "
              f"{rb['tickers_where_ens_log_beats']['constant']} stocks; DM t vs constant "
              f"{dm[name]['constant']['stat']:.1f} on {dm[name]['constant']['n']} days\n")
    pt = out["point"]
    print("DOES IT PREDICT DIRECTION?")
    print(f"  correlation of forecast volatility with next return      "
          f"{pt['corr_with_return']:+.3f}   (daily cross-section "
          f"{pt['daily_cross_sectional_corr_with_return']:+.3f}, t = "
          f"{pt['daily_cross_sectional_t_stat']:+.2f})")
    print(f"  correlation with the SIZE of the next move               "
          f"{pt['corr_with_abs_return']:+.3f}")
    print(f"  share of up days when forecast vol is high / low         "
          f"{pt['share_up_when_forecast_vol_high']:.1%} / "
          f"{pt['share_up_when_forecast_vol_low']:.1%}")
    hi, lo = pt["mean_abs_move_pct_high_vs_low"]
    print(f"  mean absolute move when forecast vol is high / low       {hi:.2f}% / {lo:.2f}%")
    fl = out["forecast_log"]
    if "models" in fl:
        print(f"\nSAME ROWS AS THE GP AND ARIMA PRICE MODELS  (n={fl['n']}, "
              f"{', '.join(fl['tickers'])}, {fl['span']})")
        for k, v in fl["models"].items():
            print(f"  {k:<20} coverage {v['coverage']:.3f}  width "
                  f"{v['mean_width_pct_of_price']:.2f}% of price  Winkler "
                  f"{v['winkler_pct_of_price']:.3f}")
        print("  point MAE, % of price: " + ", ".join(
            f"{k} {v:.3f}" for k, v in fl["point_mae_pct_of_price"].items()))
    else:
        print(f"\nforecast log: {fl['status']}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
