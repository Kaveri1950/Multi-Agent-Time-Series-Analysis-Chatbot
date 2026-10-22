"""Do the chart patterns traders talk about predict tomorrow's price here?

Two claims, both common and both testable on the cached panel:

  REVERSAL      "if it went up yesterday, today it will most likely fall"
  RESISTANCE    "once it crosses resistance it falls back" -- and the
                mirror image, a break below support bounces

Each is turned into a rule that uses only data up to today, and scored the
way every price model in this project is scored: against "tomorrow equals
today". A pattern that exists but is too small to beat that baseline is not
a better prediction, however familiar it looks on a chart.

THE SIGNIFICANCE UNIT IS THE DAY, NOT THE STOCK-DAY. Four hundred stocks on
one date are mostly the market moving once, and counting them as 400
independent trials turns a nothing into p = 1e-50. Every test here first
averages across stocks within a day, then tests across days.

Run:  python -m evaluation.price_patterns
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "price_patterns.json"
LOOKBACK = 20          # "resistance" = highest close of the previous 20 sessions
YEAR_LOOKBACK = 250    # and the 52-week version
MIN_TRAIN_DAYS = 500   # before the causal rule is allowed to forecast


def _returns(universe: str) -> pd.DataFrame:
    from quant.data import load_panel
    from quant.volatility import OHLC_PATH

    close, _ = load_panel()
    if universe == "panel50":
        names = sorted(pd.read_parquet(OHLC_PATH, columns=["ticker"])["ticker"].unique())
        close = close[names]
    return close


def _day_test(flag: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 3) -> dict:
    """Mean forward return after the pattern minus the mean on all other days,
    computed within each date, then tested across dates."""
    from scipy import stats

    on = fwd.where(flag).mean(axis=1)
    off = fwd.where(~flag & fwd.notna()).mean(axis=1)
    n_on = flag.sum(axis=1)
    diff = (on - off)[n_on >= min_names].dropna()
    up_on = (fwd > 0).where(flag & fwd.notna()).mean(axis=1)[n_on >= min_names].dropna()
    t = stats.ttest_1samp(diff, 0.0)
    return {"n_days": int(len(diff)), "n_events": int(flag.sum().sum()),
            "mean_return_after_pct": float(fwd.where(flag).stack().mean() * 100),
            "mean_return_otherwise_pct": float(fwd.where(~flag).stack().mean() * 100),
            "daily_mean_difference_pct": float(diff.mean() * 100),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue),
            "share_up_after": float(up_on.mean())}


def reversal(close: pd.DataFrame) -> dict:
    r = close.pct_change(fill_method=None)
    nxt = r.shift(-1)
    sd = r.rolling(60, min_periods=40).std().shift(1)        # known before today
    out = {
        "lag1_autocorrelation_median_stock": float(
            r.apply(lambda s: s.autocorr(1)).median()),
        "after_up_day": _day_test(r > 0, nxt),
        "after_down_day": _day_test(r < 0, nxt),
        "after_big_up_day_2sd": _day_test(r > 2 * sd, nxt),
        "after_big_down_day_2sd": _day_test(r < -2 * sd, nxt),
    }
    # The literal claim: P(down tomorrow | up today)
    both = pd.concat([r.stack().rename("today"), nxt.stack().rename("tomorrow")], axis=1).dropna()
    up = both[both["today"] > 0]
    out["p_down_tomorrow_given_up_today"] = float((up["tomorrow"] < 0).mean())
    out["p_down_tomorrow_unconditional"] = float((both["tomorrow"] < 0).mean())
    return out


def support_resistance(close: pd.DataFrame) -> dict:
    nxt1 = close.pct_change(fill_method=None).shift(-1)
    nxt5 = close.shift(-5) / close - 1
    out = {}
    for label, n in (("20d", LOOKBACK), ("52w", YEAR_LOOKBACK)):
        hi = close.rolling(n, min_periods=n).max().shift(1)    # resistance, from earlier days
        lo = close.rolling(n, min_periods=n).min().shift(1)    # support
        prev = close.shift(1)
        breakout = (close > hi) & (prev <= hi.shift(1))        # first close above resistance
        breakdown = (close < lo) & (prev >= lo.shift(1))       # first close below support
        out[f"breakout_above_{label}_high"] = {"next_day": _day_test(breakout, nxt1),
                                               "next_5_days": _day_test(breakout, nxt5)}
        out[f"breakdown_below_{label}_low"] = {"next_day": _day_test(breakdown, nxt1),
                                               "next_5_days": _day_test(breakdown, nxt5)}
    return out


def as_a_forecast(close: pd.DataFrame) -> dict:
    """Turn every pattern into a next-day point forecast and score it.

    Each stock-day is put in a state -- today up or down, big or not, at a
    20-day breakout or breakdown -- and the forecast return for that state is
    its average next-day return over all EARLIER days, across all stocks.
    Then: does that beat "tomorrow equals today" on error, and does it call
    the direction better than a coin?
    """
    r = close.pct_change(fill_method=None)
    sd = r.rolling(60, min_periods=40).std().shift(1)
    hi = close.rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1)
    lo = close.rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1)
    z = r / sd
    state = pd.DataFrame("flat", index=r.index, columns=r.columns)
    state = state.mask(z > 0, "up").mask(z < 0, "down")
    state = state.mask(z > 2, "big_up").mask(z < -2, "big_down")
    state = state.mask(close > hi, state + "+breakout").mask(close < lo, state + "+breakdown")

    long = pd.concat([state.stack().rename("state"), r.shift(-1).stack().rename("y")],
                     axis=1).dropna().reset_index()
    long.columns = ["date", "ticker", "state", "y"]
    dates = np.sort(long["date"].unique())
    preds = np.full(len(long), np.nan)
    # causal: the state means are re-estimated monthly from earlier rows only
    # (an earlier row's target is known by the next date, hence `<`)
    for i in range(MIN_TRAIN_DAYS, len(dates), 21):
        block = dates[i:i + 21]
        means = long.loc[long["date"] < block[0]].groupby("state")["y"].mean()
        m = long["date"].isin(block)
        preds[m.to_numpy()] = long.loc[m, "state"].map(means).fillna(0.0).to_numpy()
    long["pred"] = preds
    s = long.dropna(subset=["pred"])

    y, p = s["y"].to_numpy(), s["pred"].to_numpy()
    nonzero = p != 0
    hit = (np.sign(p[nonzero]) == np.sign(y[nonzero]))
    daily_hit = pd.Series(hit, index=s.loc[nonzero, "date"].to_numpy()).groupby(level=0).mean()
    from scipy import stats

    t = stats.ttest_1samp(daily_hit, 0.5)
    by_state = s.groupby("state").agg(n=("y", "size"), mean_next_pct=("y", lambda v: v.mean() * 100),
                                      share_up=("y", lambda v: (v > 0).mean()))
    return {"n_forecasts": int(len(s)), "first_date": str(pd.Timestamp(s["date"].min()).date()),
            "mae_pct_pattern_rule": float(np.abs(y - p).mean() * 100),
            "mae_pct_tomorrow_equals_today": float(np.abs(y).mean() * 100),
            "rmse_ratio_rule_over_naive": float(np.sqrt(np.mean((y - p) ** 2) / np.mean(y ** 2))),
            "directional_accuracy": float(hit.mean()),
            "directional_accuracy_daily_t": float(t.statistic),
            "directional_accuracy_daily_p": float(t.pvalue),
            "by_state": {k: {c: (int(v) if c == "n" else float(v)) for c, v in row.items()}
                         for k, row in by_state.iterrows()}}


def main():
    import warnings

    warnings.filterwarnings("ignore")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for universe in ("all473", "panel50"):
        close = _returns(universe)
        out[universe] = {"n_tickers": int(close.shape[1]),
                         "span": f"{close.index[0].date()} -> {close.index[-1].date()}",
                         "reversal": reversal(close),
                         "support_resistance": support_resistance(close),
                         "as_a_forecast": as_a_forecast(close)}
    OUT_PATH.write_text(json.dumps(out, indent=2))

    for universe in ("all473", "panel50"):
        u = out[universe]
        rv, sr, fc = u["reversal"], u["support_resistance"], u["as_a_forecast"]
        print(f"\n{'=' * 78}\n{universe}: {u['n_tickers']} stocks, {u['span']}\n{'=' * 78}")
        print(f"REVERSAL  lag-1 autocorrelation of daily returns, median stock "
              f"{rv['lag1_autocorrelation_median_stock']:+.3f}")
        print(f"  P(down tomorrow | up today) {rv['p_down_tomorrow_given_up_today']:.1%}  "
              f"vs on any day {rv['p_down_tomorrow_unconditional']:.1%}")
        for k in ("after_up_day", "after_down_day", "after_big_up_day_2sd", "after_big_down_day_2sd"):
            v = rv[k]
            print(f"  {k:<24} next day {v['mean_return_after_pct']:+.3f}% vs "
                  f"{v['mean_return_otherwise_pct']:+.3f}%  up {v['share_up_after']:.1%}  "
                  f"t={v['t_stat']:+.2f} p={v['p_value']:.3f}  ({v['n_events']:,} events)")
        print("SUPPORT / RESISTANCE")
        for k, v in sr.items():
            for h in ("next_day", "next_5_days"):
                x = v[h]
                print(f"  {k:<26}{h:<12} {x['mean_return_after_pct']:+.3f}% vs "
                      f"{x['mean_return_otherwise_pct']:+.3f}%  up {x['share_up_after']:.1%}  "
                      f"t={x['t_stat']:+.2f} p={x['p_value']:.3f}  ({x['n_events']:,})")
        print(f"AS A FORECAST  ({fc['n_forecasts']:,} causal next-day forecasts from "
              f"{fc['first_date']})")
        print(f"  MAE {fc['mae_pct_pattern_rule']:.4f}% vs tomorrow=today "
              f"{fc['mae_pct_tomorrow_equals_today']:.4f}%   RMSE ratio "
              f"{fc['rmse_ratio_rule_over_naive']:.4f}")
        print(f"  direction right {fc['directional_accuracy']:.1%}  (daily t="
              f"{fc['directional_accuracy_daily_t']:+.2f}, p={fc['directional_accuracy_daily_p']:.3f})")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
