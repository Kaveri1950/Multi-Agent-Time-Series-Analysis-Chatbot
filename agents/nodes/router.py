"""Orchestrator/Router agent: decides finance vs astronomy vs solar.

Heuristic-first (fast, free, no LLM round-trip needed for an unambiguous
input) with the domain simply passed through if the caller already set it
explicitly. Astronomy targets look like 'KIC 11446443' / 'TIC 123456789';
solar requests name the Sun or space weather rather than an entity id (there
is only one Sun, so there is no key to match on); anything else is treated as
a stock ticker.
"""

import re

from agents.state import AgentState

_ASTRO_PATTERN = re.compile(r"^(KIC|TIC|EPIC)\s*\d+", re.IGNORECASE)

# Solar has no entity id to key off, so it is matched on subject words. Kept
# narrow on purpose: "flare" alone is ambiguous (stellar flares are the
# astronomy domain), so it only counts alongside a solar/space-weather word.
_SOLAR_PATTERN = re.compile(
    r"(?:solar\s+(?:flare|activity|weather|cycle|wind|storm)s?|space\s+weather|"
    r"geomagnetic|sunspot|coronal|\bCME\b|the\s+sun\b|x-?class|m-?class)",
    re.IGNORECASE,
)


def route_node(state: AgentState) -> AgentState:
    if state.get("domain"):
        return state  # caller already specified it

    target_or_ticker = (state.get("target") or state.get("ticker") or "").strip()

    # Solar is checked FIRST and against the raw text, because it is the one
    # domain with no entity id: "solar flare activity" is a subject, not a
    # ticker, and left to the finance fallback it would be looked up as one.
    if _SOLAR_PATTERN.search(target_or_ticker):
        domain = "solar"
    elif _ASTRO_PATTERN.match(target_or_ticker):
        domain = "astronomy"
    else:
        domain = "finance"

    trace = state.get("trace", [])
    trace.append(f"[router] classified '{target_or_ticker}' as domain='{domain}'")
    return {**state, "domain": domain, "trace": trace}
