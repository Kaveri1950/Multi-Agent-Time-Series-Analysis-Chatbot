"""Solar flare forecasting agent node -- the third domain in the graph.

Structurally identical to astronomy_node / finance_node: read the state,
call the domain's own modules, write a result dict back, append to the
trace. The graph does not need to know that this domain forecasts a point
process while finance forecasts a continuous series -- that difference lives
entirely inside solar/, which is the whole point of the shared-contract
design.

Note the deliberate asymmetry with the other two nodes: this one has no
"key" to look up. Finance needs a ticker and astronomy needs a target, but
there is only one Sun, so the node takes no entity argument and the router
never has to extract one.
"""

from agents.forecast_shared import from_solar_result
from agents.state import AgentState


def solar_flare_node(state: AgentState) -> AgentState:
    """Produce the current M+ flare forecast and attach it to the state.

    Failures are recorded in the trace and returned as a result with an
    `error` key rather than raised, matching the other nodes -- a dead NOAA
    endpoint should degrade one section of an answer, not kill the whole
    multi-agent run.
    """
    trace = list(state.get("trace", []))

    try:
        from solar.forecast import current_forecast

        result = current_forecast(include_context=True)
    except Exception as e:
        trace.append(f"[solar_flare] FAILED: {type(e).__name__}: {e}")
        return {**state, "solar_result": {"error": f"{type(e).__name__}: {e}"}, "trace": trace}

    prob = result.get("forecast", {}).get("probability")
    if prob is None:
        trace.append("[solar_flare] no model available -- run `python -m solar.forecast --train`")
    else:
        n_baselines = len(result.get("baselines", []))
        trace.append(
            f"[solar_flare] P(M+ in 24h) = {prob:.1%} "
            f"({result.get('method')}), compared against {n_baselines} baseline(s)"
        )

    ctx = result.get("context", {})
    if ctx.get("has_delta_region"):
        trace.append("[solar_flare] a delta-configuration active region is on the disc "
                     "(context only -- not a fitted model input)")

    # The frames are useful to the Streamlit page but are noise inside an
    # LLM prompt, so the state carries the full result while the prompt
    # rendering (forecast_shared.format_for_prompt) sees only the summary.
    return {**state, "solar_result": result, "trace": trace}


def solar_forecast_block(state: AgentState) -> str:
    """The grounded text block for the explainer prompt."""
    from agents.forecast_shared import format_for_prompt

    result = state.get("solar_result") or {}
    if result.get("error"):
        return f"SOLAR FORECAST UNAVAILABLE: {result['error']}"

    block = format_for_prompt(from_solar_result(result))

    ctx = result.get("context", {})
    if ctx:
        extra = ["", "TODAY'S SOLAR STATE (context, not model inputs):"]
        if ctx.get("n_regions") is not None:
            extra.append(f"  active regions on disc: {ctx['n_regions']}")
        if ctx.get("radio_flux") is not None:
            extra.append(f"  F10.7 radio flux: {ctx['radio_flux']}")
        if ctx.get("sunspot_number") is not None:
            extra.append(f"  sunspot number: {ctx['sunspot_number']}")
        if ctx.get("sunspot_area") is not None:
            extra.append(f"  sunspot area: {ctx['sunspot_area']} (millionths of a hemisphere)")
        if ctx.get("has_delta_region") is not None:
            extra.append(f"  delta-configuration region present: {ctx['has_delta_region']}")
        if ctx.get("note"):
            extra.append(f"  note: {ctx['note']}")
        block += "\n" + "\n".join(extra)

    return block


if __name__ == "__main__":
    out = solar_flare_node({"domain": "solar", "trace": []})
    for line in out["trace"]:
        print(line)
    print()
    print(solar_forecast_block(out))
