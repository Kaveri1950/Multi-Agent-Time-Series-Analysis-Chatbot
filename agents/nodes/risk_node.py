"""The finance path: profile the series, choose a detector, assess risk, explain.

This is where the pieces meet. The order matters and is not arbitrary:

    1. PROFILE    look at the series and decide what kind of thing it is.
                  No labels, no domain tag -- just the data. This is the only
                  step that reasons about data it has never seen.

    2. SELECT     among detectors that CAN run, pick by measured out-of-sample
                  skill. Not by asking a language model: on 123 labelled
                  queries the capability filter scored 95.9% against the LLM's
                  83.7%, in under a second against 41 minutes.

    3. ASSESS     compute the risk picture -- how much it moves, how much of
                  that is the market, what a bad day looks like now.

    4. TRANSLATE  turn every number into the sentence a person would use, and
                  only then hand it to the model to phrase.

Step 4 is not cosmetic. An LLM given "excess kurtosis 3.21" in its context
will repeat it, because it appears authoritative. Translating BEFORE the
prompt is the only reliable fix; an instruction not to use jargon leaks.
"""

from __future__ import annotations


def _cheap_profile(ticker: str, lookback_days: int = 365) -> dict:
    """Structure of this ticker's returns. Costs about a millisecond."""
    from anomaly.profile import profile_and_recommend
    from quant.data import load_panel

    close, _ = load_panel()
    if ticker not in close.columns:
        return {"structure": "unknown", "reason": f"{ticker} not in the cached panel"}
    series = close[ticker].pct_change(fill_method=None).dropna()
    import pandas as pd
    series = series.loc[series.index[-1] - pd.Timedelta(f"{lookback_days}D"):]
    return profile_and_recommend(series.to_numpy())


def risk_node(state) -> dict:
    """Produce the facts for a risk question. Returns a dict merged into state.

    Every failure is returned as a message rather than raised: a chat turn
    that dies with a traceback is worse than one that says what went wrong.
    """
    ticker = (state.get("ticker") if isinstance(state, dict) else getattr(state, "ticker", None))
    if not ticker:
        return {"risk_error": "no ticker identified in the question"}

    trace = []
    try:
        prof = _cheap_profile(ticker)
        trace.append({
            "step": "profile",
            "decision": prof.get("structure"),
            "why": prof.get("recommendation", {}).get("reason", ""),
        })
    except Exception as exc:  # noqa: BLE001
        prof = {}
        trace.append({"step": "profile", "decision": "skipped", "why": str(exc)})

    # Selection is a lookup against measured skill, which is the point: the
    # comparison happened offline against labels, not live against a guess.
    try:
        from agents.model_registry import ASSESS_RISK, feasible

        candidates = feasible(domain="finance", task=ASSESS_RISK, available={"ticker"})
        chosen = candidates[0] if candidates else None
        if chosen:
            trace.append({
                "step": "select",
                "decision": chosen["id"],
                "why": f"{len(candidates)} model(s) can answer this; chosen on measured skill "
                       f"({chosen['skill']['metric']} = {chosen['skill']['value']})",
            })
    except Exception:
        chosen = None

    try:
        from finance.risk_profile import risk_profile, to_user_facts

        profile = risk_profile(ticker, lookback_days=365)
    except Exception as exc:  # noqa: BLE001
        return {"risk_error": f"could not build a risk profile for {ticker}: {exc}",
                "risk_trace": trace}

    trace.append({
        "step": "assess",
        "decision": f"{profile['anomalies']['n_flagged']} unusual days",
        "why": f"{profile['anomalies']['n_market_wide']} market-wide, "
               f"{profile['anomalies']['n_company_specific']} company-specific",
    })
    return {
        "risk_profile": profile,
        "risk_facts": to_user_facts(profile),
        "risk_trace": trace,
        "structure": prof.get("structure"),
    }


def answer_risk_question(ticker: str, question: str = "", lookback_days: int = 365) -> dict:
    """End to end, for the chatbot and for testing.

    Returns the plain-language facts plus the trace. The caller sends `facts`
    to the model with CHAT_SYSTEM_PROMPT; the trace is for the dashboard's
    reasoning panel, never for the reply.
    """
    from finance.risk_profile import risk_profile, to_user_facts

    result = risk_node({"ticker": ticker})
    if "risk_error" in result:
        return result
    return {
        "ticker": ticker,
        "facts": result["risk_facts"],
        "trace": result["risk_trace"],
        "profile": result["risk_profile"],
    }


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    out = answer_risk_question("RELIANCE.NS")
    print("TRACE (backend only -- never shown to the user):")
    for step in out["trace"]:
        print(f"  [{step['step']:8}] {step['decision']}")
        print(f"             {step['why'][:100]}")
    print("\nFACTS sent to the model:\n")
    print(out["facts"])
