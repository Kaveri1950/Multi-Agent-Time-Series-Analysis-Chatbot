"""Shared presentation contract for FORECAST results across domains.

Deliberately parallel to agents/anomaly_shared.py, and deliberately a
separate shape rather than an extension of it. An anomaly summary answers
"what already happened that was unusual"; a forecast answers "what is likely
to happen next, how confident are we, and compared to what". Those need
different fields, and pretending otherwise would have forced one of them to
carry meaningless keys.

As with the anomaly contract, this does NOT unify the modelling. Finance
forecasts a continuous series (ARIMA / XGBoost on price returns,
finance/price_model.py); solar forecasts a point process (Hawkes / XGBoost
on flare occurrence, solar/flare_model.py). Those are genuinely different
statistical objects and each keeps the method that suits it. What is shared
is the OUTPUT shape, so the chatbot has one consistent way to talk about a
prediction regardless of which domain produced it.

The `baseline` field is the part worth arguing for. A forecast reported on
its own invites the reader to assume it is good. Reported next to what a
trivial rule or an external operational product achieves, it is honest. The
solar domain has a real external baseline available (NOAA SWPC publishes its
own flare probabilities), so the contract makes room for it rather than
leaving it as an afterthought.
"""

from typing import TypedDict


class ForecastValue(TypedDict, total=False):
    label: str            # what is being predicted, in words
    probability: float    # 0-1, for event-occurrence forecasts
    value: float          # for continuous forecasts (e.g. a predicted price)
    horizon: str          # "next 24 hours", "30 days"
    confidence: str       # LOW | MEDIUM | HIGH
    basis: list[str]      # the actual drivers, for a grounded explanation


class BaselineComparison(TypedDict, total=False):
    name: str             # "NOAA SWPC operational", "persistence", "climatology"
    probability: float
    value: float
    note: str             # why this baseline is the relevant comparison


class ForecastSummary(TypedDict, total=False):
    domain: str                       # "solar" | "finance"
    key: str                          # what was forecast (e.g. "Sun", a ticker)
    method: str                       # the model that produced it
    methods_available: list[str]
    forecast: ForecastValue
    baselines: list[BaselineComparison]
    skill: dict                       # measured skill, e.g. {"tss": .., "brier": ..}
    caveats: list[str]                # stated limits -- never omitted
    as_of: str


def from_solar_result(solar_result: dict) -> ForecastSummary:
    """Normalise solar/forecast.py's output into the shared shape."""
    fc = solar_result.get("forecast", {})
    return ForecastSummary(
        domain="solar",
        key="Sun",
        method=solar_result.get("method", "hawkes"),
        methods_available=solar_result.get("methods_available", []),
        forecast=ForecastValue(
            label=fc.get("label", "M-class or larger flare"),
            probability=fc.get("probability"),
            horizon=fc.get("horizon", "next 24 hours"),
            confidence=fc.get("confidence", "MEDIUM"),
            basis=fc.get("basis", []),
        ),
        baselines=solar_result.get("baselines", []),
        skill=solar_result.get("skill", {}),
        caveats=solar_result.get("caveats", []),
        as_of=solar_result.get("as_of", ""),
    )


def summarize(domain: str, state: dict) -> ForecastSummary | None:
    """Convenience entry point for callers holding a full graph AgentState
    that just want "the forecast for whichever domain answered", mirroring
    anomaly_shared.summarize()."""
    if domain == "solar" and state.get("solar_result"):
        return from_solar_result(state["solar_result"])
    return None


def format_for_prompt(summary: ForecastSummary) -> str:
    """Render a forecast as the grounded text block handed to the LLM.

    Every number the model is allowed to state appears here, and the caveats
    travel WITH the numbers rather than being bolted on by the prompt -- a
    model that is shown "0.31" and separately told "be careful" will quote
    the number and drop the care.
    """
    fc = summary.get("forecast", {})
    lines = [
        f"DOMAIN: {summary.get('domain')}",
        f"AS OF: {summary.get('as_of')}",
        f"MODEL: {summary.get('method')}",
        "",
        "FORECAST:",
        f"  target      : {fc.get('label')}",
        f"  horizon     : {fc.get('horizon')}",
    ]
    p = fc.get("probability")
    if p is not None:
        lines.append(f"  probability : {p:.1%}")
    lines.append(f"  confidence  : {fc.get('confidence')}")

    if fc.get("basis"):
        lines.append("  driven by   :")
        lines += [f"    - {b}" for b in fc["basis"]]

    if summary.get("baselines"):
        lines.append("")
        lines.append("BASELINES FOR COMPARISON:")
        for b in summary["baselines"]:
            val = b.get("probability")
            shown = f"{val:.1%}" if isinstance(val, (int, float)) else "n/a"
            lines.append(f"  {b.get('name')}: {shown}"
                         + (f"  ({b['note']})" if b.get("note") else ""))

    if summary.get("skill"):
        lines.append("")
        lines.append("MEASURED SKILL (from backtesting, not this forecast):")
        for k, v in summary["skill"].items():
            lines.append(f"  {k}: {v}")

    if summary.get("caveats"):
        lines.append("")
        lines.append("CAVEATS -- these must be reflected in any answer:")
        lines += [f"  - {c}" for c in summary["caveats"]]

    return "\n".join(lines)
