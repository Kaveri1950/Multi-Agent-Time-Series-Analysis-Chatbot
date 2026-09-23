"""One stock, one risk picture: "how risky is holding this, and why?"

Answers a question anomaly detection cannot. "TCS moved 6% on these days" is
description, and the user already knows it. "Holding TCS overnight, you lose
more than 1.8% once a fortnight, and in a turbulent regime that becomes 4.1%
-- and of last year's twelve unusual days, eight were the whole market moving
and four were TCS specifically" is a risk assessment.

FOUR LAYERS, EACH ANSWERING SOMETHING THE ONE ABOVE CANNOT

  LEVEL        volatility, drawdown           how much does it move
  SYSTEMATIC   beta, R^2, residual variance   is that the market or the firm
  TAIL         VaR, ES, Sortino, skew         how bad is the bad end
  REGIME       Markov state + transitions     and how bad is it RIGHT NOW

The regime layer is what makes this forward-looking. A single unconditional
VaR is an average over calm years and crisis months, too tight in one and
too loose in the other -- measured on the Nifty proxy, an unconditional 99%
VaR was REJECTED by a Kupiec test (breach rate 0.17% against a 1% target)
while the regime-conditional version passed at p=0.86.

WHAT THIS DELIBERATELY DOES NOT DO

It does not say whether to buy. It reports what the downside looks like and
where past shocks came from. "Should I invest" depends on horizon, on what
else is held, and on the investor's tolerance -- none of which this sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Flag a day when either its raw or its idiosyncratic move exceeds this many
#: standard deviations. Both, because they catch different things: a big raw
#: move can be pure market beta, and a modest raw move can hide a large
#: company-specific one that the market was masking.
ANOMALY_SIGMA = 2.5

#: Trading days per year, for annualising.
YEAR = 252


def _max_drawdown(returns: pd.Series) -> dict:
    """Worst peak-to-trough fall, and how long it took to recover.

    Recovery time matters as much as depth and is usually omitted. A 30% fall
    that recovers in two months is a different experience from a 30% fall
    that takes three years, and volatility cannot tell them apart.
    """
    curve = (1 + returns).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    trough = dd.idxmin()
    peak_date = curve.loc[:trough].idxmax()
    after = curve.loc[trough:]
    recovered = after[after >= curve.loc[peak_date]]
    return {
        "max_drawdown_pct": round(float(dd.min()) * 100, 2),
        "peak_date": str(peak_date.date()),
        "trough_date": str(trough.date()),
        "recovery_date": str(recovered.index[0].date()) if len(recovered) else None,
        "recovery_days": int((recovered.index[0] - trough).days) if len(recovered) else None,
        "still_underwater": bool(len(recovered) == 0),
    }


def _systematic_split(stock: pd.Series, market: pd.Series) -> dict:
    """Beta, and how much of the stock's risk the market explains.

    R-squared is the number that matters for interpreting everything else.
    At R^2 = 0.6 most moves are the market and a "stock anomaly" is usually a
    market anomaly wearing a ticker. At R^2 = 0.2 the stock mostly does its
    own thing and idiosyncratic flags deserve attention.
    """
    aligned = pd.concat([stock, market], axis=1).dropna()
    s, m = aligned.iloc[:, 0], aligned.iloc[:, 1]
    beta, alpha = np.polyfit(m, s, 1)
    resid = s - (beta * m + alpha)
    r2 = 1 - resid.var() / s.var() if s.var() > 0 else np.nan
    return {
        "beta": round(float(beta), 3),
        "alpha_annual_pct": round(float(alpha) * YEAR * 100, 2),
        "r_squared": round(float(r2), 3),
        "variance_from_market_pct": round(float(r2) * 100, 1),
        "variance_idiosyncratic_pct": round((1 - float(r2)) * 100, 1),
        "idio_vol_annual_pct": round(float(resid.std()) * np.sqrt(YEAR) * 100, 2),
        "_residuals": resid,
    }


def _tail_metrics(returns: pd.Series) -> dict:
    from quant.risk import var_es

    var95, es95 = var_es(returns.to_numpy(), 0.95)
    var99, es99 = var_es(returns.to_numpy(), 0.99)
    downside = returns[returns < 0]
    dd_dev = float(np.sqrt((downside ** 2).mean())) if len(downside) else np.nan
    ann_ret = float(returns.mean()) * YEAR
    return {
        "var_95_pct": round(var95 * 100, 2),
        "expected_shortfall_95_pct": round(es95 * 100, 2),
        "var_99_pct": round(var99 * 100, 2),
        "expected_shortfall_99_pct": round(es99 * 100, 2),
        "worst_day_pct": round(float(returns.min()) * 100, 2),
        "best_day_pct": round(float(returns.max()) * 100, 2),
        "skew": round(float(returns.skew()), 2),
        # Excess kurtosis above 3 means fatter tails than a normal
        # distribution -- i.e. a Gaussian VaR would understate the risk.
        "excess_kurtosis": round(float(returns.kurt()), 2),
        "downside_deviation_annual_pct": round(dd_dev * np.sqrt(YEAR) * 100, 2),
        # Sortino penalises only downside volatility, which is the half an
        # investor actually minds.
        "sortino": round(ann_ret / (dd_dev * np.sqrt(YEAR)), 2) if dd_dev else np.nan,
        "sharpe": round(ann_ret / (float(returns.std()) * np.sqrt(YEAR)), 2)
        if returns.std() else np.nan,
    }


#: The fitted regime model, cached across calls.
#:
#: The HMM is fitted on the MARKET, so it is identical for every ticker --
#: yet refitting it per query cost 28-46 seconds and made the whole risk
#: path unusable in a chat. Keyed on the panel's last date so it refreshes
#: when new data arrives and never goes stale silently.
_REGIME_CACHE: dict = {}


#: On-disk cache, so the cold start happens once ever rather than once per
#: process. The expensive part is not the HMM -- it is the turbulence index,
#: a Mahalanobis distance computed on a rolling 252-day covariance for every
#: one of ~1,660 days, which is a Python loop and takes most of the 50s.
_REGIME_CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "finance" / "regime_state.joblib"


def _fit_market_regime(market_panel: pd.DataFrame):
    from quant.regimes import (build_observations, filtered_states, fit_regimes,
                               turbulent_flag)

    market = market_panel.mean(axis=1).dropna()
    key = (str(market.index[-1].date()), len(market), market_panel.shape[1])
    if key in _REGIME_CACHE:
        return _REGIME_CACHE[key]

    if _REGIME_CACHE_PATH.exists():
        try:
            import joblib

            stored = joblib.load(_REGIME_CACHE_PATH)
            if stored.get("key") == key:
                _REGIME_CACHE[key] = stored["value"]
                return stored["value"]
        except Exception:
            pass   # a stale or unreadable cache is never fatal -- just refit

    obs = build_observations(market, panel_returns=market_panel)
    model, order = fit_regimes(obs)
    regime = turbulent_flag(model, order, obs, use_filtered=True)
    probs = filtered_states(model, obs).iloc[-1].to_numpy()
    _REGIME_CACHE.clear()          # only ever one panel in play
    _REGIME_CACHE[key] = (obs, model, order, regime, probs)
    try:
        import joblib

        _REGIME_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"key": key, "value": _REGIME_CACHE[key]}, _REGIME_CACHE_PATH)
    except Exception:
        pass   # failing to persist is a slow next start, not an error
    return _REGIME_CACHE[key]


def _regime_layer(stock: pd.Series, market_panel: pd.DataFrame) -> dict:
    """Markov regime state and what it implies for this stock specifically.

    The regime is inferred from the MARKET, not the stock -- turbulence is a
    property of the environment, and one stock does not have enough tail
    observations to fit a regime model on. But the VaR is then measured on
    THIS stock's returns within each regime, so the answer is stock-specific.
    """
    from quant.risk import markov_var, propagate_state, var_es

    obs, model, order, regime, probs = _fit_market_regime(market_panel)
    s = stock.reindex(obs.index).dropna()
    reg = regime.reindex(s.index)
    quiet_var, _ = var_es(s[reg == False].to_numpy(), 0.95)   # noqa: E712
    turb_var, turb_es = var_es(s[reg == True].to_numpy(), 0.95)   # noqa: E712

    collapsed = np.array([probs[int(order[0])],
                          probs[[int(x) for x in order[1:]]].sum()])
    trans = np.asarray(model.transmat_, dtype=float)
    qi, ti = int(order[0]), [int(x) for x in order[1:]]
    C = np.array([[trans[qi, qi], trans[qi, ti].sum()],
                  [trans[ti, qi].mean(), trans[np.ix_(ti, ti)].sum(axis=1).mean()]])
    C /= C.sum(axis=1, keepdims=True)

    out = {
        "current_regime": "turbulent" if collapsed[1] > 0.5 else "quiet",
        "p_turbulent_now": round(float(collapsed[1]), 4),
        "p_turbulent_tomorrow": round(float(propagate_state(C, collapsed, 1)[1]), 4),
        "p_turbulent_in_21d": round(float(propagate_state(C, collapsed, 21)[1]), 4),
        "long_run_turbulent_share": round(float(propagate_state(C, collapsed, 1000)[1]), 4),
        "var_95_quiet_pct": round(quiet_var * 100, 2),
        "var_95_turbulent_pct": round(turb_var * 100, 2),
        "turbulent_multiplier": round(turb_var / quiet_var, 2) if quiet_var else np.nan,
    }
    # Multi-day VaR through the chain rather than the sqrt(t) rule, which
    # assumes independent returns and so ignores volatility clustering.
    try:
        for h in (5, 10):
            mv = markov_var(s, reg, model, order, collapsed, horizon=h)
            out[f"var_95_{h}d_markov_pct"] = round(mv["var"] * 100, 2)
            out[f"var_95_{h}d_sqrt_rule_pct"] = round(mv.get("sqrt_rule_var", np.nan) * 100, 2)
    except Exception:
        pass
    return out


def _describe_event(raw: float, market: float, resid: float,
                    explained: float, sigma: float) -> str:
    """Why this day was flagged, as a phrase a reader can use.

    Derived from the same three numbers the flag itself came from, so the
    explanation cannot disagree with the detection. Deliberately plain: the
    reader wants "a sharp fall while the market barely moved", not a
    standardised residual.
    """
    direction = "fall" if raw < 0 else "rise"
    size = "sharp" if sigma > 4 else "notable"

    if abs(explained) > 0.6:
        return (f"a {size} {direction} of {abs(raw) * 100:.1f}%, but the market moved "
                f"{abs(market) * 100:.1f}% the same day -- this was the market, not the company")
    if abs(market) < 0.005:
        return (f"a {size} {direction} of {abs(raw) * 100:.1f}% while the market was "
                f"essentially flat, so it came from the company itself")
    if (raw > 0) != (market > 0):
        return (f"a {size} {direction} of {abs(raw) * 100:.1f}% in the OPPOSITE direction "
                f"to the market, which moved {market * 100:+.1f}% -- clearly company-specific")
    return (f"a {size} {direction} of {abs(raw) * 100:.1f}%, of which only "
            f"{abs(raw - resid) * 100:.1f}% is explained by the market")


def _anomalies(stock: pd.Series, market: pd.Series, residuals: pd.Series,
               beta: float, sigma: float = ANOMALY_SIGMA) -> list[dict]:
    """Unusual days, split by whether the market explains them."""
    z_raw = (stock - stock.mean()) / stock.std()
    z_res = (residuals - residuals.mean()) / residuals.std()
    flagged = (z_raw.abs() > sigma) | (z_res.abs() > sigma)

    rows = []
    for date in stock.index[flagged]:
        raw, mkt, res = float(stock[date]), float(market[date]), float(residuals[date])
        # What fraction of the move the market accounts for. Above ~0.6 the
        # stock essentially went where everything went.
        explained = float(np.clip(beta * mkt / raw, -1, 1)) if raw else 0.0
        kind = "market-wide" if abs(explained) > 0.6 else "company-specific"
        sigma = float(max(abs(z_raw[date]), abs(z_res[date])))
        rows.append({
            "date": str(date.date()),
            "return_pct": round(raw * 100, 2),
            "market_pct": round(mkt * 100, 2),
            "idiosyncratic_pct": round(res * 100, 2),
            "explained_by_market": round(explained, 2),
            "kind": kind,
            "sigma": round(sigma, 2),
            # WHY the day was flagged, in words. Without this the answer can
            # say a day was unusual but not what made it so, which is the
            # question a reader actually has.
            "reason": _describe_event(raw, mkt, res, explained, sigma),
        })
    return sorted(rows, key=lambda r: -abs(r["return_pct"]))


def risk_profile(ticker: str, lookback_days: int = 365, top_n: int = 8,
                 with_news: bool = True) -> dict:
    """The complete picture for one stock."""
    from quant.data import load_panel

    close, volume = load_panel()
    if ticker not in close.columns:
        raise ValueError(f"{ticker} not in the cached panel of {close.shape[1]} tickers")

    tv = (close * volume).rolling(21, min_periods=10).mean()
    panel = close[tv.mean().nlargest(100).index].pct_change(fill_method=None).dropna(how="all")
    market = panel.mean(axis=1)
    full = close[ticker].pct_change(fill_method=None).dropna()

    # Beta is estimated on the FULL history -- a year of data gives a noisy
    # beta -- while the reported risk is measured over the requested window.
    systematic = _systematic_split(full, market)
    residuals_full = systematic.pop("_residuals")

    end = full.index[-1]
    window = full.loc[end - pd.Timedelta(f"{lookback_days}D"):]
    mkt_w = market.reindex(window.index)
    res_w = residuals_full.reindex(window.index)

    profile = {
        "ticker": ticker,
        "window": {"from": str(window.index[0].date()), "to": str(window.index[-1].date()),
                   "trading_days": len(window)},
        "level": {
            "annual_return_pct": round(float(window.mean()) * YEAR * 100, 1),
            "annual_volatility_pct": round(float(window.std()) * np.sqrt(YEAR) * 100, 1),
        },
        "systematic": systematic,
        "tail": _tail_metrics(window),
        "drawdown": _max_drawdown(window),
    }
    try:
        profile["regime"] = _regime_layer(full, panel)
    except Exception as exc:  # noqa: BLE001
        profile["regime_error"] = f"{type(exc).__name__}: {exc}"

    events = _anomalies(window, mkt_w, res_w, systematic["beta"])
    profile["anomalies"] = {
        "n_flagged": len(events),
        "n_market_wide": sum(1 for e in events if e["kind"] == "market-wide"),
        "n_company_specific": sum(1 for e in events if e["kind"] == "company-specific"),
        "events": events[:top_n],
    }

    if with_news and events:
        profile["anomalies"]["events"] = _attach_news(ticker, events[:top_n])
    return profile


def _attach_news(ticker: str, events: list[dict]) -> list[dict]:
    """Headlines near each company-specific date, where the archive reaches.

    Marked explicitly as NEAR the date, never as the cause. Free RSS gives
    roughly five to nine months, so older events get an honest "unavailable"
    rather than a guess -- and a headline that merely reports the move
    ("shares slip as Reliance weighs") is not an explanation of it.
    """
    try:
        from finance.sentiment import fetch_headlines_rss

        news = fetch_headlines_rss(ticker)
        if news.empty:
            return events
        news["_d"] = pd.to_datetime(news["published"], errors="coerce", utc=True).dt.tz_localize(None)
        news = news.dropna(subset=["_d"])
        oldest = news["_d"].min()
    except Exception:
        return events

    for ev in events:
        if ev["kind"] != "company-specific":
            continue
        date = pd.Timestamp(ev["date"])
        if date < oldest:
            ev["news"] = None
            ev["news_status"] = f"unavailable -- predates the RSS archive ({oldest.date()})"
            continue
        near = news[(news["_d"] >= date - pd.Timedelta("3D"))
                    & (news["_d"] <= date + pd.Timedelta("3D"))]
        ev["news"] = near["headline"].head(3).tolist() or None
        ev["news_status"] = "headlines near this date" if len(near) else "none found within 3 days"
    return events


def to_fact_block(profile: dict) -> str:
    """Compact prose for the LLM prompt.

    Pre-formatted rather than handing over raw rows, for two reasons: it costs
    roughly 800 fewer tokens than the equivalent JSON, and it removes the
    chance of the model composing its own numbers -- the failure mode that
    produced a reported -3.3% gap where the real figure was -1.37%.
    """
    s, t, d = profile["systematic"], profile["tail"], profile["drawdown"]
    a, lv = profile["anomalies"], profile["level"]
    r = profile.get("regime", {})
    lines = [
        f"{profile['ticker']} risk profile, {profile['window']['from']} to "
        f"{profile['window']['to']} ({profile['window']['trading_days']} trading days).",
        f"Volatility {lv['annual_volatility_pct']}% annualised. Beta {s['beta']} "
        f"(market explains {s['variance_from_market_pct']}% of its variance; "
        f"{s['variance_idiosyncratic_pct']}% is company-specific).",
        f"95% one-day VaR {t['var_95_pct']}%, expected shortfall {t['expected_shortfall_95_pct']}%. "
        f"Worst single day {t['worst_day_pct']}%. Return skew {t['skew']}, "
        f"excess kurtosis {t['excess_kurtosis']} (fat tails: a Gaussian VaR would understate this).",
        f"Max drawdown {d['max_drawdown_pct']}% ({d['peak_date']} to {d['trough_date']}"
        + (f", recovered in {d['recovery_days']} days)." if d["recovery_days"]
           else ", not yet recovered)."),
    ]
    if r:
        lines.append(
            f"Market regime is {r['current_regime']} (P(turbulent tomorrow) = "
            f"{r['p_turbulent_tomorrow']:.1%}). In a quiet regime this stock's 95% VaR is "
            f"{r['var_95_quiet_pct']}%; in a turbulent one {r['var_95_turbulent_pct']}% "
            f"({r['turbulent_multiplier']}x wider)."
        )
        if "var_95_10d_markov_pct" in r:
            lines.append(
                f"10-day VaR via the Markov chain {r['var_95_10d_markov_pct']}%, against "
                f"{r['var_95_10d_sqrt_rule_pct']}% from the sqrt-of-time rule, which assumes "
                f"independent returns and so ignores volatility clustering."
            )
    lines.append(
        f"{a['n_flagged']} unusual days in the window: {a['n_market_wide']} market-wide, "
        f"{a['n_company_specific']} company-specific."
    )
    for ev in a["events"][:4]:
        line = (f"  {ev['date']}  {ev['return_pct']:+.2f}% (market {ev['market_pct']:+.2f}%, "
                f"idiosyncratic {ev['idiosyncratic_pct']:+.2f}%) -- {ev['kind']}")
        if ev.get("news"):
            line += f" | near this date: {ev['news'][0][:70]}"
        elif ev.get("news_status"):
            line += f" | news {ev['news_status']}"
        lines.append(line)
    lines.append("This describes risk only. It is not advice on whether to buy or sell.")
    return "\n".join(lines)



def to_user_facts(profile: dict) -> str:
    """Facts for the chat prompt: the real metrics, each with its meaning.

    An earlier version translated every quantity away -- "swings about 28% a
    year" with no number named. That reads pleasantly and is useless to
    anyone who wants to compare two stocks or check the work. The fix is not
    to hide the metric but to STATE it and say what it means, which is what a
    good analyst does and what a reader can actually act on.

    So each line carries the number, its standard name, and one clause of
    plain English. The model is told to keep all three.
    """
    from agents.answer_style import describe_drawdown

    s, t, d = profile["systematic"], profile["tail"], profile["drawdown"]
    a, lv, r = profile["anomalies"], profile["level"], profile.get("regime", {})
    name = profile["ticker"].replace(".NS", "")

    lines = [
        f"FACTS about {name}, {profile['window']['from']} to {profile['window']['to']} "
        f"({profile['window']['trading_days']} trading days).",
        "",
        "RISK METRICS -- state the number AND its plain meaning:",
        f"- Annualised volatility: {lv['annual_volatility_pct']:.1f}% "
        f"(how much it swings in a typical year)",
        f"- Beta vs the market: {s['beta']:.2f} "
        f"(a beta below 1 means it moves less than the market)",
        f"- Share of its movement explained by the market: {s['variance_from_market_pct']:.0f}% "
        f"-- so {s['variance_idiosyncratic_pct']:.0f}% is company-specific and will NOT be "
        f"cushioned by holding other Indian stocks",
        f"- Value at Risk, 95% one-day: {t['var_95_pct']:.2f}% "
        f"(on the worst 1 day in 20, expect to lose at least this)",
        f"- Expected Shortfall, 95%: {t['expected_shortfall_95_pct']:.2f}% "
        f"(the AVERAGE loss on those worst days -- what VaR alone hides)",
        f"- Value at Risk, 99% one-day: {t['var_99_pct']:.2f}% "
        f"(the worst 1 day in 100)",
        f"- Worst single day observed: {t['worst_day_pct']:.2f}%",
        f"- Maximum drawdown: {d['max_drawdown_pct']:.1f}% "
        f"({describe_drawdown(d['max_drawdown_pct'], d['recovery_days'], d['still_underwater'])})",
        f"- Excess kurtosis: {t['excess_kurtosis']:.2f} "
        f"(above 0 means fatter tails than a bell curve -- extreme days are more likely "
        f"than a normal model would predict)",
        f"- Skew: {t['skew']:.2f} "
        f"({'falls faster than it rises' if t['skew'] < 0 else 'rises faster than it falls'})",
        f"- Sortino ratio: {t['sortino']:.2f} "
        f"(return per unit of DOWNSIDE risk; unlike Sharpe it does not penalise upside)",
    ]

    if r:
        lines += [
            "",
            "MARKET REGIME (from a two-state hidden Markov model fitted on the market):",
            f"- Current state: {r['current_regime']}",
            f"- Probability it turns turbulent tomorrow: {r['p_turbulent_tomorrow']:.1%}",
            f"- Probability it is turbulent at some point within a month: "
            f"{r['p_turbulent_in_21d']:.1%}",
            f"- Long-run share of turbulent days: {r['long_run_turbulent_share']:.1%}",
            f"- This stock's 95% VaR in a QUIET regime: {r['var_95_quiet_pct']:.2f}%; "
            f"in a TURBULENT one: {r['var_95_turbulent_pct']:.2f}% "
            f"({r['turbulent_multiplier']:.1f}x wider)",
        ]
        if "var_95_10d_markov_pct" in r:
            lines.append(
                f"- 10-day VaR propagated through the Markov chain: "
                f"{r['var_95_10d_markov_pct']:.2f}%, against "
                f"{r['var_95_10d_sqrt_rule_pct']:.2f}% from the usual square-root-of-time "
                f"rule, which wrongly assumes each day is independent of the last")

    lines += [
        "",
        f"UNUSUAL DAYS: {a['n_flagged']} in this period -- {a['n_market_wide']} where the "
        f"whole market moved, {a['n_company_specific']} specific to the company.",
    ]
    for ev in a["events"][:6]:
        lines.append(f"  - {ev['date']}  |  {ev['return_pct']:+.1f}%  |  market that day: "
                     f"{ev['market_pct']:+.1f}%  |  WHY: {ev.get('reason', 'flagged as unusual')}")
        if ev.get("news"):
            lines.append(f"      headline around that date: \"{ev['news'][0][:110]}\"")
        elif ev.get("news_status", "").startswith("unavailable"):
            lines.append("      no news archive reaches back that far")
        else:
            lines.append("      no notable news found within three days")

    lines += [
        "",
        "RULES: describe risk, never recommend buying or selling. Headlines appeared "
        "around the same time -- never call them the cause.",
    ]
    return "\n".join(lines)



if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="RELIANCE.NS")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    prof = risk_profile(args.ticker, lookback_days=args.days)
    if args.json:
        print(json.dumps({k: v for k, v in prof.items()}, indent=2, default=str))
    else:
        print(to_fact_block(prof))

