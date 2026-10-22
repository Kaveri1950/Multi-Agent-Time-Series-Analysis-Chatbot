"""One question, five models, one answer: "how risky is this stock?"

The registry holds five finance models that each answer a different part of
that question, and until now nothing put them together:

    forecast_volatility   how big will tomorrow's move be
    forecast_value        where that puts tomorrow's close, as a range
    assess_risk           how much could be lost, and how bad is the bad end
    forecast_event_prob   is the market calm or turbulent, and will that change
    detect_anomaly        which recent days were unusual

SELECTION IS NOT HARD-CODED HERE. The question is fanned out into five
sub-questions and each one goes through `select_model`, the same selector
that scored 95% on the labelled query set. This module never names a model
when deciding what to run: it holds a table of RUNNERS keyed by model id, and
runs whichever id the selector returns. Replace a registry entry with a better
model for the same task and the composition picks it up, provided a runner
exists; if none does, the output says so instead of quietly running the old
one.

EVERY NUMBER CARRIES ITS VALIDATION STATUS. A component returns `figures`,
and a figure is never a bare value: it has a status and the evidence for it.

    passed       a named test was run on THIS number and it passed
    partial      one test passed and the other could not be computed
    failed       a named test was run on this number and it failed
    untested     no backtest of this quantity exists
    descriptive  a historical fact read off the data; nothing to validate

The distinction that matters most is between "the method passed somewhere"
and "this number passed". The regime-conditional VaR passes Kupiec and
Christoffersen on the equal-weighted market proxy, which is what the registry
records, and on individual stocks it passes both on a minority of names. So
each stock's VaR is reported with that stock's own walk-forward backtest.

Evidence that cannot be computed per query -- interval coverage, directional
accuracy, the regime's lift on event days -- is read from
evaluation/results/risk_validation.json, written by
`python -m evaluation.risk_validation`. Nothing here is typed in by hand
except what the registry and the model's own metadata already record.

Run:  python -m agents.risk_composition TCS.NS
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from agents.model_registry import (ASSESS_RISK, DETECT_ANOMALY, FORECAST_EVENT_PROBABILITY,
                                   FORECAST_VALUE, FORECAST_VOLATILITY)

EVIDENCE_PATH = Path(__file__).parent.parent / "evaluation" / "results" / "risk_validation.json"

PASSED, PARTIAL, FAILED = "passed", "partial", "failed"
UNTESTED, DESCRIPTIVE = "untested", "descriptive"
STATUSES = (PASSED, PARTIAL, FAILED, UNTESTED, DESCRIPTIVE)

#: The fan-out. Each sub-question is phrased the way a user would ask it, so
#: it travels the selector's real path -- task classification included --
#: rather than a side door that the 95% figure would not certify.
FAN_OUT = (
    {"role": "volatility", "task": FORECAST_VOLATILITY,
     "ask": "how volatile will {name} be tomorrow"},
    # `next_session` is what makes the one-day price range feasible. Without
    # it the selector returns the GP, which is the right answer to a question
    # with no stated horizon and the wrong one to this.
    {"role": "price_range", "task": FORECAST_VALUE,
     "ask": "what will {name} close at tomorrow", "available": {"ticker", "next_session"}},
    {"role": "tail_risk", "task": ASSESS_RISK,
     "ask": "how risky is {name}"},
    {"role": "regime", "task": FORECAST_EVENT_PROBABILITY,
     "ask": "what is the chance the market turns turbulent"},
    {"role": "anomalies", "task": DETECT_ANOMALY,
     "ask": "any unusual days in {name}"},
)

Z80, Z95 = 1.2816, 1.96

#: An interval counts as calibrated when measured coverage is within this of
#: nominal. A tolerance rather than a significance test, on purpose: with
#: 61,303 forecasts a binomial test rejects a 0.6-point miss that nobody
#: would act on, and the forecasts are correlated across tickers on the same
#: day, so the nominal sample size overstates the evidence anyway.
COVERAGE_TOLERANCE = 0.02

#: The same, for one stock's own coverage. Wider because one stock has about
#: 1,200 forecasts, where sampling error alone is a little over a point.
STOCK_COVERAGE_TOLERANCE = 0.04

#: Walk-forward VaR needs 500 days to train on, then enough test days for a
#: 5% breach rate to be countable.
VAR_MIN_TRAIN, VAR_MIN_TEST = 500, 250


def _fig(key: str, label: str, value, unit: str, status: str, evidence: str) -> dict:
    return {"key": key, "label": label, "value": value, "unit": unit,
            "status": status, "evidence": evidence}


@lru_cache(maxsize=1)
def _evidence() -> dict:
    try:
        return json.loads(EVIDENCE_PATH.read_text())
    except (OSError, ValueError):
        return {}


_NO_EVIDENCE = "not measured -- run `python -m evaluation.risk_validation`"

@lru_cache(maxsize=2)
def _cached_parquet(path: str, mtime: float):
    import pandas as pd

    return pd.read_parquet(path)


def _panel():
    """(close, volume), read once per refresh of the cache rather than per call."""
    from quant.data import PANEL_PATH, VOLUME_PATH

    return (_cached_parquet(str(PANEL_PATH), PANEL_PATH.stat().st_mtime),
            _cached_parquet(str(VOLUME_PATH), VOLUME_PATH.stat().st_mtime))


def _ohlc():
    from quant.volatility import OHLC_PATH

    return _cached_parquet(str(OHLC_PATH), OHLC_PATH.stat().st_mtime)



# --------------------------------------------------------------------------
# per-stock VaR backtest
# --------------------------------------------------------------------------

def stock_var_backtest(ticker: str, level: float = 0.95) -> dict:
    """Walk-forward VaR backtest on ONE stock, with and without the regime.

    The registry's Kupiec and Christoffersen p-values were measured on the
    equal-weighted market proxy. A single stock is a different series with
    its own tail, so the pass does not transfer and has to be re-run.

    The regime series is the served one, fitted on the market. Its states are
    filtered, so each day's label uses data up to that day only, but the HMM's
    PARAMETERS are fitted once on the full sample. That is a mild look-ahead
    which this backtest inherits and does not remove.
    """
    from finance.risk_profile import _fit_market_regime, market_panel
    from quant.risk import backtest

    close, volume = _panel()
    if ticker not in close.columns:
        return {"testable": False, "why": f"{ticker} is not in the cached price panel"}
    obs, _, _, regime, _ = _fit_market_regime(market_panel(close, volume))
    s = close[ticker].pct_change(fill_method=None).dropna().reindex(obs.index).dropna()
    out = {"panel_last_date": str(close.index[-1].date()), "level": level}
    if len(s) < VAR_MIN_TRAIN + VAR_MIN_TEST:
        return {**out, "testable": False,
                "why": f"only {len(s)} days of history; the backtest needs "
                       f"{VAR_MIN_TRAIN} to train on and {VAR_MIN_TEST} to test on"}

    def one(reg):
        b = backtest(s, reg, level=level, min_train=VAR_MIN_TRAIN)
        k, c = b["kupiec"], b["christoffersen"]
        return {"n_days": k["n"], "breach_rate": k["rate"],
                "kupiec_p": _num(k["p_value"]), "christoffersen_p": _num(c["p_value"])}

    return {**out, "testable": True, "conditional": one(regime), "unconditional": one(None)}


def _num(x):
    """NaN -> None, so the result survives a round trip through JSON."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _var_backtest(ticker: str) -> dict:
    """The stored backtest when it describes the current panel, else a fresh one."""
    stored = _evidence().get("var", {})
    hit = stored.get("per_ticker", {}).get(ticker)
    if hit and stored.get("panel_last_date") == str(_panel()[0].index[-1].date()):
        return hit
    return stock_var_backtest(ticker)


def _var_verdict(bt: dict, which: str = "conditional") -> tuple[str, str]:
    """(status, evidence) for a VaR figure, from that stock's own backtest."""
    if not bt.get("testable"):
        return UNTESTED, f"not backtested on this stock: {bt.get('why', 'no result')}"
    r = bt[which]
    target = 1 - bt["level"]
    k, c = r["kupiec_p"], r["christoffersen_p"]
    text = (f"walk-forward on this stock, {r['n_days']} days: breached on "
            f"{r['breach_rate']:.1%} of them against a {target:.0%} target. ")
    if k is None:
        return UNTESTED, text + "Kupiec could not be computed."
    problems = []
    if k <= 0.05:
        side = ("TOO MANY breaches -- this VaR understates the loss"
                if r["breach_rate"] > target else
                "too few breaches -- this VaR overstates the loss, so it errs on the safe side")
        problems.append(f"Kupiec p={k:.4f} REJECTS coverage: {side}")
    else:
        text += f"Kupiec p={k:.3f} passes. "
    if c is None:
        text += "Christoffersen not computable (no two breaches on consecutive days). "
    elif c <= 0.05:
        problems.append(f"Christoffersen p={c:.4f} REJECTS independence: breaches cluster, "
                        f"so bad days arrive together rather than evenly")
    else:
        text += f"Christoffersen p={c:.3f} passes. "
    if problems:
        return FAILED, text + "; ".join(problems) + "."
    return (PARTIAL if c is None else PASSED), text.strip()


# --------------------------------------------------------------------------
# runners -- one per model id, each returning a component
# --------------------------------------------------------------------------

def _run_volatility(ticker: str, ctx: dict) -> dict:
    from quant.production_model import forecast

    r = forecast(ticker)
    sk = r["skill"]
    ev = _evidence().get("volatility", {})
    lo, hi = r["interval_80"]
    # forecast() publishes the 80% band only. In logs its centre is the log
    # forecast and its half-width is 1.2816 sd, so both come back exactly and
    # the 95% band below is the same model's, not a second estimate.
    lp = 0.5 * (math.log(lo) + math.log(hi))
    sd = (math.log(hi) - math.log(lo)) / (2 * Z80)
    band95 = [math.exp(lp - Z95 * sd), math.exp(lp + Z95 * sd)]
    today = r["today_realised_vol"]
    direction = r["direction"]

    def coverage(level: str) -> tuple[str, str]:
        c = ev.get("interval", {}).get(level)
        if not c:
            return UNTESTED, _NO_EVIDENCE
        nominal = int(level) / 100
        ok = abs(c["coverage"] - nominal) <= COVERAGE_TOLERANCE
        mine = c.get("per_ticker", {}).get(ticker)
        text = (f"held-out coverage {c['coverage']:.1%} against a nominal {nominal:.0%} on "
                f"{ev['n_forecasts']:,} walk-forward forecasts across all stocks "
                f"(range across stocks {c['per_ticker_min']:.1%} to {c['per_ticker_max']:.1%}). ")
        # The band's width is one number shared by every stock, so it is
        # calibrated on average and can still be wrong for a particular name.
        if mine is not None and abs(mine - nominal) > STOCK_COVERAGE_TOLERANCE:
            ok = False
            text += (f"ON THIS STOCK it covered {mine:.1%}: the band is "
                     + ("TOO NARROW here, so the true range is wider than shown. "
                        if mine < nominal else "too wide here, so it overstates the range. "))
        elif mine is not None:
            text += f"On this stock it covered {mine:.1%}. "
        text += (f"Misses are lopsided: realised volatility lands ABOVE the band on "
                 f"{c['above']:.1%} of days and below it on {c['below']:.1%}, so the upper "
                 f"edge is the weaker one.")
        return (PASSED if ok else FAILED), text

    d = ev.get("direction", {})
    r2 = ev.get("r2", {})
    acc, ceil = sk["directional_accuracy"], sk["directional_ceiling"]
    w, wceil = sk["within_ticker_r2_log"], sk["within_ticker_ceiling"]
    figures = [
        _fig("forecast_vol_pct", "Forecast volatility, next session (annualised)",
             round(r["forecast_annualised_vol"] * 100, 2), "%", PASSED,
             f"QLIKE {sk['qlike']} on {sk['n_heldout_forecasts']:,} walk-forward forecasts; "
             f"beats each stock's own trailing average on {sk['beats_trailing_mean_on']}. "
             f"LIMIT: within one stock the forecast explains R-squared {w} of the day-to-day "
             f"level against an attainable {wceil} ({w / wceil:.0%}), so read it as a rough "
             f"level, not a precise one"
             + (f" (re-measured: {r2['within']:.3f})." if r2 else ".")),
        _fig("interval_80_pct", "80% interval",
             [round(lo * 100, 2), round(hi * 100, 2)], "%", *coverage("80")),
        _fig("interval_95_pct", "95% interval",
             [round(b * 100, 2) for b in band95], "%", *coverage("95")),
        _fig("direction", "Tomorrow against today", direction, "", PASSED,
             f"calls higher-or-lower correctly {acc:.1%} of the time against a measured "
             f"ceiling of {ceil:.1%}, so {1 - acc:.0%} of these calls are wrong and at most "
             f"{ceil - acc:.1%} of that is recoverable by any model"
             + (f" (re-measured: {d['log_rule']:.1%})." if d else ".")),
        _fig("today_vol_pct", "Realised volatility, latest session (annualised)",
             round(today * 100, 2), "%", DESCRIPTIVE,
             "Garman-Klass estimate from one day's open, high, low and close. A noisy "
             "measurement: about half of its day-to-day variance is estimation error."),
    ]
    limits = ["This is how BIG the move will be, never which way the price goes.",
              "One session ahead only. Nothing here speaks to next week."]
    if not r["is_live"]:
        limits.append(r["note"])
    return {"as_of": r["as_of"], "figures": figures, "limits": limits,
            "detail": {"trained_through": r["trained_through"],
                       "member_forecasts": r["member_forecasts"], "is_live": r["is_live"]}}


def _run_price_range(ticker: str, ctx: dict) -> dict:
    from quant.price_intervals import forecast

    r = forecast(ticker)
    sk = r["skill"]

    def band(level: str) -> tuple[str, str]:
        nominal = int(level) / 100
        cov = sk[f"coverage_{level}"]
        ok = abs(cov - nominal) <= COVERAGE_TOLERANCE
        return (PASSED if ok else FAILED), (
            f"held-out coverage {cov:.1%} against a nominal {nominal:.0%} on "
            f"{sk['n_heldout_forecasts']:,} stock-days. Interval score "
            f"{sk[f'winkler_{level}']:.3f} against {sk[f'winkler_{level}_next_best']:.3f} for "
            f"the best of six other volatility sources, a per-stock HAR -- effectively a tie "
            f"with that simple model, and clearly narrower than constant, trailing, EWMA or "
            f"GARCH volatility. LIMIT: "
            f"on the most volatile third of days it covers only "
            f"{sk[f'coverage_{level}_on_wild_days']:.0%}.")

    def pct(price):
        return round((price / r["close"] - 1) * 100, 2)

    lo80, hi80 = r["interval_80"]
    lo95, hi95 = r["interval_95"]
    figures = [
        _fig("point", "Best single guess for the next close", round(r["close"], 2), "",
             DESCRIPTIVE,
             f"this is today's close. The volatility forecast says nothing about direction: "
             f"its correlation with the next return is {sk['corr_with_next_return']:+.3f}, and "
             f"the share of up days is {sk['share_up_when_vol_high']:.1%} when it is high "
             f"against {sk['share_up_when_vol_low']:.1%} when it is low. No price model in "
             f"this project beats today's close as a point forecast."),
        _fig("range_80", "80% range for the next close",
             [round(lo80, 2), round(hi80, 2)], "", *band("80")),
        _fig("range_80_pct", "...as a move from today's close", [pct(lo80), pct(hi80)], "%",
             DESCRIPTIVE, "the line above, restated as a percentage"),
        _fig("range_95", "95% range for the next close",
             [round(lo95, 2), round(hi95, 2)], "", *band("95")),
    ]
    limits = ["A range, not a call. It is centred on no change by construction.",
              "One session ahead only."]
    if r["scale_is_pooled"]:
        limits.append("This stock has under 250 days of history, so its range uses the "
                      "panel's average scale, not its own.")
    return {"as_of": r["as_of"], "figures": figures, "limits": limits, "detail": {}}


def _run_tail_risk(ticker: str, ctx: dict) -> dict:
    from finance.risk_profile import risk_profile

    p = risk_profile(ticker, lookback_days=ctx["lookback_days"], top_n=10 ** 6,
                     with_news=ctx["with_news"])
    bt = _var_backtest(ticker)
    lv, s, t, d = p["level"], p["systematic"], p["tail"], p["drawdown"]
    reg = p.get("regime", {})
    win = f"{p['window']['from']} to {p['window']['to']}"
    fact = f"historical fact over {win} ({p['window']['trading_days']} trading days)"

    figures = []
    if reg:
        now = reg["current_regime"]
        other = "turbulent" if now == "quiet" else "quiet"
        status, why = _var_verdict(bt, "conditional")
        figures += [
            _fig("var_95_now_pct", f"95% one-day VaR in the current ({now}) regime",
                 reg[f"var_95_{now}_pct"], "%", status, why),
            _fig("var_95_other_pct", f"95% one-day VaR if the market turns {other}",
                 reg[f"var_95_{other}_pct"], "%", status,
                 "same estimator and same backtest as the line above"),
        ]
        if "var_95_10d_markov_pct" in reg:
            figures.append(_fig(
                "var_95_10d_pct", "95% ten-day VaR, propagated through the Markov chain",
                reg["var_95_10d_markov_pct"], "%", UNTESTED,
                "multi-day VaR has never been backtested in this project; only the one-day "
                "number has"))
    _, uncond = _var_verdict(bt, "unconditional")
    figures += [
        _fig("var_95_window_pct", "95% one-day VaR, trailing window, ignoring the regime",
             t["var_95_pct"], "%", UNTESTED,
             "a trailing-window quantile, which is not the estimator that was backtested. "
             "The nearest tested thing is the expanding-window unconditional VaR -- " + uncond),
        _fig("es_95_pct", "95% expected shortfall (average loss on the worst days)",
             t["expected_shortfall_95_pct"], "%", UNTESTED,
             "expected shortfall is not backtested anywhere in this project; Kupiec and "
             "Christoffersen test VaR only"),
        _fig("annual_vol_pct", "Annualised volatility", lv["annual_volatility_pct"], "%",
             DESCRIPTIVE, fact),
        _fig("max_drawdown_pct", "Maximum drawdown", d["max_drawdown_pct"], "%",
             DESCRIPTIVE, fact + ("; not yet recovered" if d["still_underwater"]
                                  else f"; recovered in {d['recovery_days']} days")),
        _fig("worst_day_pct", "Worst single day", t["worst_day_pct"], "%", DESCRIPTIVE, fact),
        _fig("beta", "Beta against the market", s["beta"], "", DESCRIPTIVE,
             f"least-squares fit on the full history; the market explains "
             f"{s['variance_from_market_pct']}% of this stock's variance and "
             f"{s['variance_idiosyncratic_pct']}% is company-specific"),
    ]
    limits = ["Historical tails describe only the shocks that have already happened."]
    if "regime_error" in p:
        limits.append(f"Regime-conditional figures are missing: {p['regime_error']}")
    # Attribution for the anomaly component, which has typed flags but no way
    # to say whether a day was the market or the company.
    ctx["attribution"] = {e["date"]: {"kind": e["kind"], "reason": e["reason"],
                                      **({"news": e["news"]} if e.get("news") else {})}
                          for e in p["anomalies"]["events"]}
    return {"as_of": p["window"]["to"], "figures": figures, "limits": limits,
            # the full profile rides along for the chatbot's charts; it is
            # dropped before the market snapshot is written to disk
            "detail": {"window": p["window"], "var_backtest": bt, "profile": p}}


def _run_regime(ticker: str, ctx: dict) -> dict:
    from finance.risk_profile import market_panel, market_regime_state

    s = market_regime_state(market_panel(*_panel()))
    ev = _evidence().get("regime", {})
    if ev:
        status = PASSED if ev["fisher_p"] < 0.05 and ev["lift"] > 1 else FAILED
        why = (f"turbulent on {ev['p_turbulent_event']:.1%} of {ev['n_event_days']} "
               f"pre-registered event days against {ev['p_turbulent_ordinary']:.1%} of ordinary "
               f"days, a lift of {ev['lift']:.2f} (Fisher exact p={ev['fisher_p']:.2g}; lift "
               f"{ev['excluding_2020']['lift']:.2f}, p={ev['excluding_2020']['fisher_p']:.2g} "
               f"with 2020 removed). States are filtered, so each day uses data up to that "
               f"day; the HMM's parameters are fitted once on the full sample.")
    else:
        status, why = UNTESTED, _NO_EVIDENCE
    forecast_gap = ("a transition probability read off the fitted chain. It has never been "
                    "scored as a forecast, and cannot honestly be scored against the model's "
                    "own state labels")
    figures = [
        _fig("current_regime", "Market regime now", s["current_regime"], "", status, why),
        _fig("p_turbulent_now", "P(market is turbulent now)", s["p_turbulent_now"], "prob",
             status, "same evidence as the line above"),
        _fig("p_turbulent_tomorrow", "P(turbulent tomorrow)", s["p_turbulent_tomorrow"],
             "prob", UNTESTED, forecast_gap),
        _fig("p_turbulent_in_21d", "P(turbulent on the 21st session from now)",
             s["p_turbulent_in_21d"], "prob", UNTESTED,
             forecast_gap + ". This is the chance of BEING turbulent on that day, not of a "
             "turbulent day occurring at some point in the month, which is higher"),
        _fig("p_quiet_to_turbulent", "Daily chance a quiet market turns turbulent",
             s["p_quiet_to_turbulent"], "prob", DESCRIPTIVE, "fitted transition probability"),
        _fig("p_turbulent_to_quiet", "Daily chance a turbulent market turns quiet",
             s["p_turbulent_to_quiet"], "prob", DESCRIPTIVE, "fitted transition probability"),
        _fig("turbulent_share", "Share of history spent turbulent",
             s["turbulent_share_of_history"], "prob", DESCRIPTIVE,
             "fraction of days the filtered state was turbulent"),
    ]
    return {"as_of": s["as_of"], "figures": figures,
            "limits": ["The regime is a property of the MARKET, identical for every stock.",
                       "Turbulent means moving, not falling: turbulent stretches in this "
                       "sample were often rallies."],
            "detail": {}}


def price_frame(ticker: str):
    """Daily OHLCV with indicators for one ticker, from the cached panels.

    The cache is preferred so a 50-name scan makes no network calls and gives
    the same answer twice. A name the OHLC cache lacks is downloaded.
    """
    import pandas as pd

    from finance.price_data import add_technical_indicators, get_price_history

    ohlc = _ohlc()
    g = ohlc[ohlc["ticker"] == ticker]
    if g.empty:
        return add_technical_indicators(get_price_history(ticker, start="2019-01-01"))
    df = g.set_index("date").sort_index()[["open", "high", "low", "close"]]
    df.columns = ["Open", "High", "Low", "Close"]
    df["Volume"] = _panel()[1][ticker].reindex(df.index)
    return add_technical_indicators(df)


_FLAG_NAMES = {"return_outlier": "return outlier", "volatility_spike": "volatility spike",
               "volume_spike": "volume spike", "gap_event": "overnight gap",
               "sentiment_divergence": "sentiment divergence"}


def _run_anomalies(ticker: str, ctx: dict) -> dict:
    import pandas as pd

    from finance.anomaly_detector import detect_all

    scored = detect_all(price_frame(ticker))
    last = scored.index[-1]
    recent = scored.loc[last - pd.Timedelta(days=ctx["recent_days"]):]
    attribution = ctx.get("attribution", {})
    days = []
    for date, row in recent[recent["is_anomaly"]].iterrows():
        key = str(date.date())
        days.append({"date": key,
                     "types": [n for c, n in _FLAG_NAMES.items() if bool(row.get(c))],
                     "severity": row["severity_label"],
                     "severity_score": round(float(row["severity_score"]), 2),
                     "reasons": row["reasons"],
                     "attribution": attribution.get(key)})
    days.sort(key=lambda r: -r["severity_score"])

    ev = _evidence().get("anomaly", {})
    if ev:
        dl, ex = ev["day_level"], ev["excluding_2020"]
        status = PASSED if dl["p"] < 0.05 and ev["lift"] > 1 else FAILED
        why = (f"flags {ev['flag_rate_event']:.1%} of stock-days on pre-registered event "
               f"dates against {ev['flag_rate_ordinary']:.1%} on ordinary ones across "
               f"{ev['n_tickers']} stocks, a lift of {ev['lift']:.2f} (Mann-Whitney on the "
               f"daily share flagged, {dl['n_event_days']} event days, p={dl['p']:.2g}; "
               f"lift {ex['lift']:.2f} with 2020 removed). The thresholds are fixed "
               f"constants, never tuned on those events. Most flagged days are still not "
               f"events: {ev['flag_rate_ordinary']:.0%} of ordinary days are flagged too.")
    else:
        status, why = UNTESTED, ("the registry records no ground-truth evaluation for this "
                                 "detector; " + _NO_EVIDENCE)
    n_attr = sum(1 for r in days if r["attribution"])
    figures = [
        _fig("n_recent_anomalies", f"Unusual days in the last {ctx['recent_days']} days",
             len(days), "days", status, why),
        _fig("n_market_wide", "...of which the market explains",
             sum(1 for r in days if (r["attribution"] or {}).get("kind") == "market-wide"),
             "days", DESCRIPTIVE,
             f"attribution splits a day's return into beta times the market and a remainder. "
             f"Only {n_attr} of the {len(days)} flagged days are large enough in return terms "
             f"to have one; a volume spike or a gap with a small return has none."),
        _fig("n_company_specific", "...of which are company-specific",
             sum(1 for r in days if (r["attribution"] or {}).get("kind") == "company-specific"),
             "days", DESCRIPTIVE, "same decomposition as the line above"),
    ]
    return {"as_of": str(last.date()), "figures": figures,
            "limits": ["Describes the past only. A flagged day is not a forecast.",
                       "Four of the five detectors ran. Sentiment divergence needs "
                       "headlines and was skipped."],
            "detail": {"days": days, "n_days_scanned": int(len(recent))}}


#: model id -> runner. Keyed by MODEL, not by task, so what runs is whatever
#: the selector returned.
RUNNERS = {
    "finance.volatility_forecast": _run_volatility,
    "finance.volatility_price_range": _run_price_range,
    "finance.risk_profile": _run_tail_risk,
    "finance.market_regime": _run_regime,
    "finance.anomaly_detectors": _run_anomalies,
}


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

def normalise_ticker(ticker: str) -> str:
    t = ticker.strip().upper()
    return t if "." in t else t + ".NS"


def assess_stock_risk(ticker: str, recent_days: int = 90, lookback_days: int = 365,
                      with_news: bool = False, strategy: str = "filter_only") -> dict:
    """Fan one risk question out to five models and return one assessment.

    A component that fails is reported as an error in its own slot; the
    others still answer. A report with a stated hole is more use than none.
    """
    from agents.model_selector import select_model

    ticker = normalise_ticker(ticker)
    name = ticker.split(".")[0]
    ctx = {"recent_days": recent_days, "lookback_days": lookback_days,
           "with_news": with_news}
    routing, components = [], {}

    for step in FAN_OUT:
        question = step["ask"].format(name=name)
        sel = select_model(question, domain="finance",
                           available=step.get("available", {"ticker"}), strategy=strategy)
        spec = sel.get("spec") or {}
        routing.append({"role": step["role"], "question": question, "task": sel["task"],
                        "model_id": sel["model_id"], "how": sel["how"],
                        "shortlist": sel["shortlist"],
                        "fallback_used": sel["fallback_used"]})
        base = {"model_id": sel["model_id"], "model_name": spec.get("name"),
                "registry_skill": spec.get("skill", {})}
        if sel["task"] != step["task"]:
            components[step["role"]] = {**base, "error": (
                f"the selector read {question!r} as {sel['task']}, not {step['task']}; "
                f"not run, because the answer would be to a different question")}
            continue
        runner = RUNNERS.get(sel["model_id"])
        if runner is None:
            components[step["role"]] = {**base, "error": (
                f"the selector chose {sel['model_id']}, which this layer has no runner for")}
            continue
        try:
            components[step["role"]] = {**base, **runner(ticker, ctx)}
        except Exception as exc:  # noqa: BLE001
            components[step["role"]] = {**base, "error": f"{type(exc).__name__}: {exc}"}

    out = {"ticker": ticker,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "routing": routing, "components": components,
           "freshness": _freshness(components)}
    out["summary"] = _summary(out)
    return out


def _freshness(components: dict) -> dict:
    """Which day each component describes, and whether they agree."""
    as_of = {role: c["as_of"] for role, c in components.items() if c.get("as_of")}
    out = {"as_of": as_of, "consistent": len(set(as_of.values())) <= 1}
    if as_of:
        newest, oldest = max(as_of.values()), min(as_of.values())
        today = datetime.now(timezone.utc).date()
        out["age_days"] = (today - datetime.fromisoformat(newest).date()).days
        if not out["consistent"]:
            lag = sorted(r for r, d in as_of.items() if d == oldest)
            out["warning"] = (
                f"The components do not describe the same day. The newest is as of {newest}; "
                f"{', '.join(lag)} stop at {oldest}, "
                f"{(datetime.fromisoformat(newest) - datetime.fromisoformat(oldest)).days} "
                f"days earlier, because their price data was last refreshed on a different "
                f"day. Anything that happened in between is missing from those parts.")
    return out


def figure(assessment: dict, role: str, key: str) -> dict | None:
    for f in assessment["components"].get(role, {}).get("figures", []):
        if f["key"] == key:
            return f
    return None


def _summary(a: dict) -> str:
    """Three or four sentences, built from the figures and nothing else."""
    name = a["ticker"].split(".")[0]
    parts = []
    v, band, d = (figure(a, "volatility", k) for k in
                  ("forecast_vol_pct", "interval_80_pct", "direction"))
    today = figure(a, "volatility", "today_vol_pct")
    if v:
        parts.append(
            f"{name}'s volatility for the next session is forecast at {v['value']:.1f}% "
            f"annualised (80% interval {band['value'][0]:.1f}% to {band['value'][1]:.1f}%), "
            f"{d['value']} than the latest session's {today['value']:.1f}%.")
    rng = figure(a, "price_range", "range_80_pct")
    if rng:
        parts.append(f"That puts the next close within {rng['value'][0]:+.1f}% to "
                     f"{rng['value'][1]:+.1f}% of today's four times in five, with no view "
                     f"on which side.")
    var = figure(a, "tail_risk", "var_95_now_pct")
    reg = figure(a, "regime", "current_regime")
    if var:
        verdict = {PASSED: "this stock's backtest passes both coverage tests",
                   PARTIAL: "this stock's backtest passes coverage; independence could not "
                            "be tested",
                   FAILED: "this stock's backtest FAILS, so treat the figure as indicative",
                   UNTESTED: "it could not be backtested on this stock"}[var["status"]]
        parts.append(f"On the worst one day in twenty it loses at least "
                     f"{abs(var['value']):.2f}% in the current regime; {verdict}.")
    if reg:
        p = figure(a, "regime", "p_turbulent_tomorrow")
        parts.append(f"The market is {reg['value']}, with a {p['value']:.1%} chance of being "
                     f"turbulent tomorrow.")
    n = figure(a, "anomalies", "n_recent_anomalies")
    if n:
        parts.append(f"{n['value']} unusual day(s) were flagged recently.")
    if a["freshness"].get("warning"):
        parts.append("These parts are not all as of the same day.")
    parts.append("This describes risk and is not advice to buy or sell.")
    return " ".join(parts)


def to_text(a: dict) -> str:
    """The assessment as plain text, every figure with its status beside it."""
    def show(f):
        v = f["value"]
        if isinstance(v, list):
            return f"{v[0]} to {v[1]}{f['unit']}"
        if f["unit"] == "prob":
            return f"{v:.1%}"
        return f"{v}{f['unit'] if f['unit'] == '%' else (' ' + f['unit']).rstrip()}"

    lines = [f"RISK ASSESSMENT  {a['ticker']}", "=" * 72, a["summary"], ""]
    if a["freshness"].get("warning"):
        lines += ["DATA: " + a["freshness"]["warning"], ""]
    for r in a["routing"]:
        c = a["components"][r["role"]]
        lines.append(f"{r['role'].upper()}  <- {r['model_id']}  "
                     f"(task {r['task']}, chosen from {len(r['shortlist'])} feasible; "
                     f"as of {c.get('as_of', 'n/a')})")
        if "error" in c:
            lines += [f"  NOT AVAILABLE: {c['error']}", ""]
            continue
        for f in c["figures"]:
            lines.append(f"  {f['label']}: {show(f)}   [{f['status'].upper()}]")
            lines.append(f"      {f['evidence']}")
        for day in c["detail"].get("days", [])[:6]:
            attr = day["attribution"]
            lines.append(f"  - {day['date']}  {day['severity']:<6} {', '.join(day['types'])}"
                         + (f"  ({attr['kind']})" if attr else ""))
        for lim in c["limits"]:
            lines.append(f"  LIMIT: {lim}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# for the chatbot: facts for the model to phrase, and a table it cannot alter
# --------------------------------------------------------------------------

#: How a status reads in a chat answer. Icon and word together, never colour.
STATUS_WORDS = {PASSED: "✅ passed its test", PARTIAL: "◐ partly tested",
                FAILED: "❌ failed its test", UNTESTED: "❔ not tested",
                DESCRIPTIVE: "· a historical fact"}

#: The figures a chat answer leads with, in order. The rest stay in the trace
#: and on the page; a reply that lists twenty numbers is not read.
CHAT_FIGURES = (("volatility", "forecast_vol_pct"), ("volatility", "direction"),
                ("price_range", "range_80"), ("tail_risk", "var_95_now_pct"),
                ("tail_risk", "var_95_other_pct"), ("tail_risk", "es_95_pct"),
                ("tail_risk", "max_drawdown_pct"), ("regime", "current_regime"),
                ("regime", "p_turbulent_tomorrow"), ("anomalies", "n_recent_anomalies"))


def _show_value(f: dict) -> str:
    v = f["value"]
    if isinstance(v, list):
        if f["key"] == "range_80_pct":
            # a MOVE from today's close, so it carries signs
            return f"{v[0]:+.1f}% to {v[1]:+.1f}% from today's close"
        if f["unit"] == "%":
            # a band of volatility LEVELS: no signs, or a model reads it as a move
            return f"{v[0]:.1f}% to {v[1]:.1f}% a year"
        return f"{v[0]} to {v[1]}"
    if f["unit"] == "prob":
        return f"{v:.1%}"
    if f["unit"] == "%":
        return f"{v}%"
    return f"{v} {f['unit']}".strip()


def _short(text: str, limit: int = 160) -> str:
    """The evidence up to its first full stop, so a table cell stays readable."""
    first = text.split(". ")[0].rstrip(".")
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "…"


#: Meanings the chat model got wrong when left to supply its own. Observed:
#: VaR called the "expected maximum loss", and a volatility band described as
#: a range of price moves.
_MEANINGS = [
    "Value at Risk (VaR) 95%: on the worst 1 day in 20 the loss is AT LEAST this much. "
    "It is a threshold, not a maximum; the average loss on those days is the expected "
    "shortfall, which is larger.",
    "Forecast volatility and its 80%/95% intervals are levels of volatility, in percent a "
    "year. They are NOT price moves. The price range for the next close is a separate line.",
]


def chat_facts(a: dict, context: str | None = None) -> str:
    """Everything the chat model may say, each figure with its status.

    `context` is a comparison with the rest of the market. Without one the
    model judged a 31% forecast "relatively low" when the panel median was
    21% -- an adjective it had no basis for.
    """
    lines = [f"FACTS about {a['ticker'].split('.')[0]}. Every figure is labelled with how "
             f"far it can be trusted.", "", "SUMMARY: " + a["summary"], ""]
    lines += ["MEANINGS:"] + [f"- {m}" for m in _MEANINGS] + [""]
    if context:
        lines += ["COMPARISON WITH THE MARKET: " + context, ""]
    if a["freshness"].get("warning"):
        lines += ["DATA DATES: " + a["freshness"]["warning"], ""]
    for role, comp in a["components"].items():
        if "error" in comp:
            lines.append(f"{role.upper()}: not available ({comp['error']})")
            continue
        lines.append(f"{role.upper()} (as of {comp.get('as_of')}):")
        for f in comp["figures"]:
            lines.append(f"- {f['label']}: {_show_value(f)} "
                         f"[{STATUS_WORDS[f['status']]}: {_short(f['evidence'], 220)}]")
        for day in comp["detail"].get("days", [])[:5]:
            attr = (day["attribution"] or {}).get("kind")
            lines.append(f"  unusual day {day['date']}: {', '.join(day['types'])}"
                         + (f" ({attr})" if attr else ""))
        for lim in comp["limits"]:
            lines.append(f"  limit: {lim}")
    lines += ["", "RULES: say how far each number can be trusted, in words. Never call a "
              "number tested or reliable when it is marked not tested or failed. Do not call "
              "a level high or low except against the COMPARISON given. This describes "
              "risk; never recommend buying or selling."]
    return "\n".join(lines)


def status_table(a: dict) -> str:
    """The key figures as a markdown table, built here rather than by the model.

    A small local model drops or softens qualifiers. The table is appended to
    its reply verbatim, so the status of every headline number reaches the
    reader whatever the prose did.
    """
    rows = ["| Measure | Value | How far to trust it |", "|---|---|---|"]
    for role, key in CHAT_FIGURES:
        f = figure(a, role, key)
        if f:
            rows.append(f"| {f['label']} | **{_show_value(f)}** | "
                        f"{STATUS_WORDS[f['status']]} — {_short(f['evidence'])} |")
    return "\n".join(rows)


def market_chat_facts(m: dict, top: int = 10) -> str:
    rows = m["ranking"]
    lines = ["FACTS about the market, all 50 names in the panel. Each figure is labelled "
             "with how far it can be trusted.", "", "SUMMARY: " + m["summary"], ""]
    for f in m["figures"]:
        lines.append(f"- {f['label']}: {_show_value(f)} "
                     f"[{STATUS_WORDS[f['status']]}: {_short(f['evidence'], 260)}]")
    reg = (m.get("regime") or {}).get("figures", [])
    for f in reg[:3]:
        lines.append(f"- {f['label']}: {_show_value(f)} [{STATUS_WORDS[f['status']]}]")
    lines += ["", f"MOST VOLATILE {top}, forecast for the next session:"]
    for r in rows[:top]:
        lines.append(f"  {r['rank']}. {r['ticker'].split('.')[0]}: {r['forecast_vol_pct']:.1f}% "
                     f"(VaR backtest on this stock: {r['var_status']})")
    lines += ["CALMEST 5:"] + [f"  {r['rank']}. {r['ticker'].split('.')[0]}: "
                              f"{r['forecast_vol_pct']:.1f}%" for r in rows[-5:]]
    lines += ["", "RULES: the order is reliable in broad strokes only; neighbours are not "
              "distinguishable. Never recommend buying or selling."]
    return "\n".join(lines)


def market_status_table(m: dict, top: int = 10) -> str:
    rows = ["| # | Stock | Forecast volatility | 80% band | VaR backtest on this stock |",
            "|---|---|---|---|---|"]
    for r in m["ranking"][:top]:
        rows.append(f"| {r['rank']} | {r['ticker'].split('.')[0]} | "
                    f"**{r['forecast_vol_pct']:.1f}%** | "
                    f"{r['band_80_low']:.1f}–{r['band_80_high']:.1f}% | "
                    f"{STATUS_WORDS.get(r['var_status'], str(r['var_status']))} |")
    rank = next((f for f in m["figures"] if f["key"] == "ranking"), None)
    if rank:
        rows.append("")
        rows.append(f"*The order {STATUS_WORDS[rank['status']].split(' ', 1)[1]}, but a plain "
                    f"66-day average ranks almost as well: read it in broad strokes.*")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# the whole market
# --------------------------------------------------------------------------

def panel_tickers() -> list[str]:
    """The names the volatility model can serve: those in the OHLC cache."""
    return sorted(_ohlc()["ticker"].unique())


def _value(a: dict, role: str, key: str, field: str = "value"):
    f = figure(a, role, key)
    return f[field] if f else None


def assess_market_risk(tickers: list[str] | None = None, recent_days: int = 90,
                       progress=None) -> dict:
    """The same five-model assessment for every name in the panel, ranked.

    RANKED BY FORECAST VOLATILITY, AND ONLY BY THAT. It is the one quantity
    here with measured skill at ordering stocks against each other. A blended
    score would need weights, and no weights have been validated; a rank that
    looks more sophisticated and has less evidence behind it is a step back.

    The other columns travel with each name so they can be read beside the
    rank, each carrying its own status. They do not move it.
    """
    tickers = tickers or panel_tickers()
    assessments, rows, unranked = {}, [], []
    for i, t in enumerate(tickers, 1):
        a = assess_stock_risk(t, recent_days=recent_days)
        assessments[a["ticker"]] = a
        if progress:
            progress(i, len(tickers), a["ticker"])
        vol = a["components"]["volatility"]
        if "error" in vol:
            unranked.append({"ticker": a["ticker"], "why": vol["error"]})
            continue
        days = a["components"]["anomalies"].get("detail", {}).get("days", [])
        band = _value(a, "volatility", "interval_80_pct")
        bt = a["components"]["tail_risk"].get("detail", {}).get("var_backtest", {})
        rows.append({
            "ticker": a["ticker"],
            "forecast_vol_pct": _value(a, "volatility", "forecast_vol_pct"),
            "band_80_low": band[0], "band_80_high": band[1],
            "band_status": _value(a, "volatility", "interval_80_pct", "status"),
            "today_vol_pct": _value(a, "volatility", "today_vol_pct"),
            "direction": _value(a, "volatility", "direction"),
            "var_95_now_pct": _value(a, "tail_risk", "var_95_now_pct"),
            "var_status": _value(a, "tail_risk", "var_95_now_pct", "status"),
            "var_breach_rate": (bt.get("conditional") or {}).get("breach_rate"),
            "annual_vol_pct": _value(a, "tail_risk", "annual_vol_pct"),
            "max_drawdown_pct": _value(a, "tail_risk", "max_drawdown_pct"),
            "beta": _value(a, "tail_risk", "beta"),
            "n_recent_anomalies": _value(a, "anomalies", "n_recent_anomalies"),
            "anomaly_dates": [d["date"] for d in days],
        })
    rows.sort(key=lambda r: -r["forecast_vol_pct"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    first = next(iter(assessments.values()), {"components": {}, "freshness": {}})
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "ranked_by": "forecast_vol_pct", "n_ranked": len(rows), "unranked": unranked,
           "ranking": rows, "regime": first["components"].get("regime"),
           "freshness": first["freshness"], "recent_days": recent_days,
           "figures": _market_figures(rows, recent_days),
           "unusual_dates": _unusual_dates(rows),
           "assessments": assessments}
    out["summary"] = _market_summary(out)
    return out


MARKET_SNAPSHOT_PATH = EVIDENCE_PATH.with_name("market_risk.json")


def _jsonable(x):
    """numpy scalars as plain numbers; anything else unexpected as text."""
    return x.item() if hasattr(x, "item") else str(x)


def save_market_snapshot(m: dict) -> Path:
    """Write the market assessment to disk, so a page can open without recomputing it."""
    import copy

    m = copy.deepcopy(m)
    for a in m.get("assessments", {}).values():
        a["components"].get("tail_risk", {}).get("detail", {}).pop("profile", None)
    MARKET_SNAPSHOT_PATH.write_text(json.dumps(m, indent=1, default=_jsonable))
    return MARKET_SNAPSHOT_PATH


def load_market_snapshot() -> dict | None:
    try:
        return json.loads(MARKET_SNAPSHOT_PATH.read_text())
    except (OSError, ValueError):
        return None


def _unusual_dates(rows: list[dict], top: int = 5) -> list[dict]:
    """Days on which many names were flagged at once: the market, not a stock."""
    from collections import Counter

    counts = Counter(d for r in rows for d in r["anomaly_dates"])
    return [{"date": d, "n_flagged": n, "share": round(n / len(rows), 3)}
            for d, n in counts.most_common(top)] if rows else []


def _market_figures(rows: list[dict], recent_days: int) -> list[dict]:
    """Panel-level numbers, each with its status, like any other figure."""
    import statistics as st

    if not rows:
        return []
    cs = _evidence().get("volatility", {}).get("cross_section", {})
    an = _evidence().get("anomaly", {})
    if cs:
        rank_status = PASSED if cs["gain_over_trailing66_t"] > 2 else FAILED
        rank_why = (
            f"on {cs['n_days']:,} held-out days the forecast's order across stocks correlates "
            f"{cs['spearman_mean']:.3f} (Spearman) with the order that then happened, and its "
            f"five most volatile names land in the realised top ten "
            f"{cs['top5_forecast_in_realised_top10']:.0%} of the time against "
            f"{cs['top5_chance_level']:.0%} by chance. LIMIT: most of that is the stocks' "
            f"persistent level, which needs no model. Ranking by a plain trailing 66-day "
            f"average scores {cs['spearman_trailing66_mean']:.3f} and "
            f"{cs['top5_trailing66_in_realised_top10']:.0%}. The model adds "
            f"{cs['gain_over_trailing66']:+.3f} (t = {cs['gain_over_trailing66_t']:.1f}; better "
            f"on {cs['share_of_days_better_than_trailing66']:.0%} of days) -- real, and small. "
            f"Read the order in broad strokes: neighbours in the table are not "
            f"distinguishable.")
    else:
        rank_status, rank_why = UNTESTED, _NO_EVIDENCE
    tally = lambda key: {s: sum(1 for r in rows if r[key] == s) for s in STATUSES}  # noqa: E731
    var, band = tally("var_status"), tally("band_status")
    med_f = st.median(r["forecast_vol_pct"] for r in rows)
    med_t = st.median(r["today_vol_pct"] for r in rows)
    higher = sum(1 for r in rows if r["direction"] == "higher")
    with_flag = sum(1 for r in rows if r["n_recent_anomalies"])
    figures = [
        _fig("ranking", "Order of the table", "forecast volatility, highest first", "",
             rank_status, rank_why),
        _fig("median_forecast_vol_pct", "Median forecast volatility across the panel",
             round(med_f, 2), "%", PASSED,
             f"median of {len(rows)} per-stock forecasts, each validated as on its own line; "
             f"the median of the latest session's realised values is {med_t:.2f}%"),
        _fig("share_called_higher", "Names forecast more volatile tomorrow than today",
             round(higher / len(rows), 3), "prob", PASSED,
             "each call is right about 70.9% of the time against a ceiling of 74.1%, so "
             "this share is a tilt, not a count to rely on name by name"),
        _fig("var_passed", "Names whose own VaR backtest passes both tests",
             var[PASSED], f"of {len(rows)}", DESCRIPTIVE,
             f"{var[FAILED]} fail, {var[PARTIAL]} pass coverage with independence not "
             f"testable, {var[UNTESTED]} have too little history. The method passes on the "
             f"market proxy; it does not pass on most single stocks."),
        _fig("band_passed", "Names whose own 80% volatility band is calibrated",
             band[PASSED], f"of {len(rows)}", DESCRIPTIVE,
             f"{band[FAILED]} are more than {STOCK_COVERAGE_TOLERANCE:.0%} off nominal on "
             f"their own history, because one band width is shared by every stock"),
        _fig("names_with_anomaly", f"Names with an unusual day in the last {recent_days} days",
             with_flag, f"of {len(rows)}",
             PASSED if an and an["day_level"]["p"] < 0.05 else UNTESTED,
             (f"the share of names flagged on a day is what separates event days from "
              f"ordinary ones: median {an['day_level']['median_share_event']:.0%} on "
              f"pre-registered event days against "
              f"{an['day_level']['median_share_ordinary']:.0%} otherwise "
              f"(p={an['day_level']['p']:.2g}). A day with many names flagged is a market "
              f"day; a name flagged alone is its own story.") if an else _NO_EVIDENCE),
    ]
    return figures


def _market_summary(m: dict) -> str:
    rows = m["ranking"]
    if not rows:
        return "No name in the panel could be ranked."
    f = {x["key"]: x for x in m["figures"]}
    top = ", ".join(f"{r['ticker'].split('.')[0]} ({r['forecast_vol_pct']:.0f}%)"
                    for r in rows[:5])
    low = ", ".join(f"{r['ticker'].split('.')[0]} ({r['forecast_vol_pct']:.0f}%)"
                    for r in rows[-3:])
    parts = [f"Of {len(rows)} names, the most volatile for the next session are forecast "
             f"to be {top}; the calmest are {low}. The panel median is "
             f"{f['median_forecast_vol_pct']['value']:.1f}% annualised, and "
             f"{f['share_called_higher']['value']:.0%} of names are forecast more volatile "
             f"than their latest session."]
    reg = m.get("regime") or {}
    if reg.get("figures"):
        cur = next(x for x in reg["figures"] if x["key"] == "current_regime")
        nxt = next(x for x in reg["figures"] if x["key"] == "p_turbulent_tomorrow")
        parts.append(f"The market is {cur['value']}, with a {nxt['value']:.1%} chance of "
                     f"being turbulent tomorrow.")
    if m["unusual_dates"] and m["unusual_dates"][0]["share"] >= 0.2:
        d = m["unusual_dates"][0]
        parts.append(f"{d['n_flagged']} names were flagged together on {d['date']}.")
    parts.append(f"The order is reliable in broad strokes only, and "
                 f"{f['var_passed']['value']} of {len(rows)} names have a VaR that passes "
                 f"its own backtest. This describes risk and is not advice.")
    if m["freshness"].get("warning"):
        parts.append("The components are not all as of the same day.")
    return " ".join(parts)


def market_to_text(m: dict) -> str:
    lines = [f"MARKET RISK  {m['n_ranked']} names, ranked by forecast volatility",
             "=" * 100, m["summary"], ""]
    if m["freshness"].get("warning"):
        lines += ["DATA: " + m["freshness"]["warning"], ""]
    for f in m["figures"]:
        v = f["value"]
        shown = (f"{v:.1%}" if f["unit"] == "prob" else
                 f"{v}%" if f["unit"] == "%" else f"{v} {f['unit']}".strip())
        lines += [f"{f['label']}: {shown}   [{f['status'].upper()}]", f"    {f['evidence']}"]
    lines += ["", f"{'#':>3} {'ticker':<15}{'forecast':>9}{'80% band':>16}{'band':>9}"
                  f"{'today':>8}{'dir':>8}{'VaR95':>8}{'VaR test':>10}{'breach':>8}"
                  f"{'beta':>6}{'unusual':>8}"]
    for r in m["ranking"]:
        br = f"{r['var_breach_rate']:.1%}" if r["var_breach_rate"] is not None else "-"
        var = f"{r['var_95_now_pct']:.2f}" if r["var_95_now_pct"] is not None else "-"
        lines.append(
            f"{r['rank']:>3} {r['ticker']:<15}{r['forecast_vol_pct']:>8.1f}%"
            f"{r['band_80_low']:>8.1f}-{r['band_80_high']:<7.1f}{r['band_status']:>9}"
            f"{r['today_vol_pct']:>7.1f}%{r['direction']:>8}{var:>8}"
            f"{str(r['var_status']):>10}{br:>8}{r['beta']:>6.2f}{r['n_recent_anomalies']:>8}")
    if m["unusual_dates"]:
        lines += ["", "Days with the most names flagged at once:"]
        lines += [f"  {d['date']}  {d['n_flagged']} names ({d['share']:.0%})"
                  for d in m["unusual_dates"]]
    for u in m["unranked"]:
        lines.append(f"NOT RANKED: {u['ticker']} -- {u['why']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="RELIANCE.NS")
    ap.add_argument("--market", action="store_true", help="rank the whole panel")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.market:
        result = assess_market_risk()
        saved_to = save_market_snapshot(result)
        if args.json:
            result = {k: v for k, v in result.items() if k != "assessments"}
        print(json.dumps(result, indent=2, default=str) if args.json
              else market_to_text(result))
        print(f"saved -> {saved_to}")
    else:
        result = assess_stock_risk(args.ticker)
        print(json.dumps(result, indent=2, default=str) if args.json else to_text(result))
