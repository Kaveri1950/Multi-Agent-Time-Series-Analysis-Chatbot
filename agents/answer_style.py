"""Turn measured facts into something a person would actually say.

THE PROBLEM THIS SOLVES

The analysis layer produces things like:

    beta 0.465, excess kurtosis 3.21, 95% VaR -2.90%, PR-AUC 0.314,
    detector latent_isoforest, periodicity ratio 107440.53

All true, all useless to someone asking "is TCS risky?". A user wants:

    "TCS has been swinging around 28% a year, and it is still 38% below its
     December peak. On a rough day you would be down about 3%."

So this module does two things, and keeps them apart:

  1. A SYSTEM PROMPT that forbids the jargon by name. Left to itself an LLM
     handed a fact block will happily repeat "excess kurtosis 3.21", because
     it appears in its context and looks important.

  2. TRANSLATION HELPERS that convert a number into the sentence a person
     would use for it -- volatility into "swings about X% a year", beta into
     "moves less than the market", kurtosis into "bad days are worse than
     you'd expect".

WHAT STAYS BACKEND

Which detector ran, what it scored, how it was selected, PR-AUC, lift,
held-out margins. All of it is real and all of it belongs in the report and
the trace panel, not in a chat reply. The one exception is when the user
asks HOW it works -- then the machinery is the answer.
"""

from __future__ import annotations

#: Words that must never reach the user unless they asked a methods question.
#: Kept explicit because a model will otherwise echo whatever jargon appears
#: in the facts it was given.
#: Standard risk measures are NOT banned. Hiding "Value at Risk" behind "you'd
#: be down about 3%" reads pleasantly and leaves a reader unable to compare
#: two stocks or check the work. Named measures are informative when the
#: number comes with its meaning, which the facts supply.
#:
#: What stays banned is the MACHINERY: which detector ran, what it scored,
#: how it was selected. That belongs in the trace and the report.
BANNED_IN_CHAT = (
    "PR-AUC", "ROC-AUC", "held-out", "in-sample", "prevalence",
    "isolation forest", "latent_isoforest", "hampel", "mahalanobis",
    "matrix profile", "Lomb-Scargle", "periodicity ratio", "SDE",
    "detector", "PR curve", "no-skill floor",
)

CHAT_SYSTEM_PROMPT = """You are a market and astronomy analyst writing for someone intelligent who is not a specialist. Think of how a good explainer writes: clear, structured, warm but never chatty.

You will be given a block of FACTS. Every number you write must come from it. Never invent, infer, or adjust a figure that is not there.

STRUCTURE YOUR ANSWER
Open with one or two sentences that answer the question directly. Then use short bold section headings to organise the rest. Typically:

  **The short answer**        one or two sentences
  **Risk metrics**            a table: measure, value, what it means
  **What actually happened**  a table of unusual days: date, move, what drove it
  **Market conditions**       the regime now, and what it does to the numbers

Use a markdown table when listing several dated events -- date, size of move, and what drove it. A table is far easier to scan than the same facts in prose. Use bold sparingly, for the numbers that matter most.

HOW TO WRITE
- GIVE THE NUMBERS. The facts contain named risk measures with their meanings
attached; keep both. Write "Value at Risk (95%, one day): **-2.90%** -- on the
worst day in twenty, expect to lose at least this much", not "you'd be down
about 3%" and not "-2.90%" alone. A reader who wants to compare two stocks
needs the figure; a reader who does not know the term needs the clause.
- Present the risk measures as a compact markdown table where there are several.
- Say what a number MEANS, not just what it is. "85% of its movement is its own"
matters because holding other Indian stocks will not cushion it -- say so.
- Do not name the DETECTOR or the selection machinery. Naming a standard risk
measure is informative; naming "latent isolation forest" is not.
- Where a headline is given, quote it and place it beside the date. Say it appeared around that time; never that it caused the move.
- If something is missing, say so plainly rather than filling the gap.

WHAT YOU MUST NOT DO
- Never advise buying, selling or holding. If asked, describe the risk and say the decision depends on their horizon and circumstances.
- Never claim a planet has been found. A candidate is a candidate.
- Do not pad. If the facts are thin, the answer is short.

Only if the user asks HOW the system works may you describe the machinery.
"""

#: Wording for the methods question, so the honest answer is available
#: without it leaking into every other reply.
METHODS_SYSTEM_PROMPT = CHAT_SYSTEM_PROMPT + """

The user has asked about the method itself, so this time you may name what ran \
and why it was chosen. Keep it concrete and short: what the system looked at, \
what it decided, and what evidence supports that choice. Still no unexplained \
jargon -- say "how far it moved compared with a typical day" rather than \
"z-score".
"""


def describe_volatility(annual_pct: float) -> str:
    if annual_pct < 15:
        return f"fairly steady, swinging about {annual_pct:.0f}% in a typical year"
    if annual_pct < 25:
        return f"moderately bumpy, swinging about {annual_pct:.0f}% a year"
    if annual_pct < 40:
        return f"quite volatile, swinging about {annual_pct:.0f}% a year"
    return f"very volatile, swinging about {annual_pct:.0f}% a year"


def describe_beta(beta: float, market_share_pct: float) -> str:
    """Beta in words, with the caveat that low beta is not low risk.

    Worth stating, because it is the most common misreading: TCS has a beta
    of 0.47 and is MORE volatile than Reliance at 0.76, because 85% of its
    movement is its own rather than the market's.
    """
    if beta < 0.7:
        move = "tends to move less than the market overall"
    elif beta < 1.1:
        move = "moves roughly in step with the market"
    else:
        move = "tends to swing harder than the market"
    if market_share_pct < 30:
        return (f"{move}, though most of what it does -- about "
                f"{100 - market_share_pct:.0f}% -- is its own story rather than the market's")
    return f"{move}; roughly {market_share_pct:.0f}% of its movement comes from the market"


def describe_tail(excess_kurtosis: float, skew: float) -> str:
    parts = []
    if excess_kurtosis > 1:
        parts.append("its really bad days are worse than you'd expect from the usual swings")
    if skew < -0.3:
        parts.append("and it tends to fall faster than it rises")
    elif skew > 0.3:
        parts.append("and its big moves are more often up than down")
    return ", ".join(parts) if parts else "its ups and downs are fairly evenly spread"


def describe_drawdown(pct: float, recovered_days: int | None, still_under: bool) -> str:
    if still_under:
        return f"it's still about {abs(pct):.0f}% below its peak and hasn't climbed back yet"
    if recovered_days:
        months = recovered_days / 30
        return (f"its worst stretch took it {abs(pct):.0f}% down, and it took about "
                f"{months:.0f} months to recover")
    return f"its worst stretch took it {abs(pct):.0f}% down"


def describe_regime(regime: str, p_tomorrow: float, multiplier: float) -> str:
    if regime == "quiet":
        return (f"markets are calm right now, and there's only about a "
                f"{p_tomorrow:.0%} chance that changes tomorrow. If things did turn "
                f"choppy, a rough day would be roughly {multiplier:.0f} times worse")
    return (f"markets are choppy at the moment, so a rough day right now is about "
            f"{multiplier:.0f} times worse than it would be in calmer times")


def leaked_jargon(text: str) -> list[str]:
    """Which forbidden terms slipped through. Used by tests, not at runtime."""
    low = text.lower()
    return [w for w in BANNED_IN_CHAT if w.lower() in low]


# ==========================================================================
# Numeric guard
# ==========================================================================
#
# A small local model rewrites numbers even when they are stated plainly in
# its context. Observed: the facts said "8 in total -- 0 where the whole
# market moved, and 8 that were about this company specifically", and the
# reply said "only one of them was when the whole market moved. The other
# seven were specifically related to TCS."
#
# That is not a prompting problem to be solved with firmer wording -- the
# instruction was already explicit. It is a property of the model, so it
# needs a check rather than a request. Every number the answer asserts must
# appear in the facts it was given; anything else is invented and the answer
# is repaired rather than shown.

import re as _re

#: Numbers that are ordinary English rather than claims. "one day in twenty"
#: and "about half" are phrasings, not figures pulled from the data.
_INNOCUOUS = {"0", "1", "2", "3", "4", "5", "10", "12", "20", "50", "100"}

_NUMBER = _re.compile(r"-?\d+(?:\.\d+)?")

#: Numbers written as words, which a model uses freely when paraphrasing a
#: count ("the other seven"). These are the dangerous ones, because they look
#: like prose rather than data.
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12,
}


def _numbers_in(text: str) -> set:
    out = {m.group(0).lstrip("-") for m in _NUMBER.finditer(text)}
    return {v.rstrip(".").rstrip("0").rstrip(".") if "." in v else v for v in out}


def unsupported_numbers(answer: str, facts: str, tolerance: float = 0.05) -> list[str]:
    """Figures asserted in `answer` that do not appear in `facts`.

    Matching is deliberately loose: a model that says "about 28%" when the
    facts say "28.5%" is rounding, not inventing, and flagging that would
    make the check useless. Anything outside `tolerance` of every number in
    the facts is reported.
    """
    fact_values = []
    for token in _numbers_in(facts):
        try:
            fact_values.append(abs(float(token)))
        except ValueError:
            continue

    def supported(value: float) -> bool:
        if value == 0:
            return any(v == 0 for v in fact_values)
        return any(abs(v - value) <= tolerance * max(abs(v), abs(value))
                   for v in fact_values)

    bad = []
    for token in _numbers_in(answer):
        if token in _INNOCUOUS:
            continue
        try:
            value = abs(float(token))
        except ValueError:
            continue
        if not supported(value):
            bad.append(token)

    # Word-numbers in a COUNTING context. The first version required a noun
    # immediately after ("seven days") and therefore missed the actual
    # observed failure, "The other seven were specifically related to TCS" --
    # where the noun is implied. Counting reads several ways in English and
    # the pattern has to cover them:
    #
    #   "eight unusual days"      number + noun
    #   "one of them was"         number + of + pronoun
    #   "the other seven were"    number + verb
    #
    # Digits are handled the same way, because "only 1 of them" is the same
    # claim as "only one of them" and 1 sits in the innocuous list.
    # "One was in June" is ordinary English meaning "one of the days" -- an
    # earlier version flagged it and stapled a correction onto an answer that
    # was entirely right. Singular "was" is therefore excluded; only plural
    # "were" counts, since "the other seven were..." IS a count claim.
    count_ctx = (r"(?:\s+(?:unusual\s+|bad\s+|other\s+)?(?:days?|events?|moves?|times?|"
                 r"occasions?)\b|\s+of\s+(?:them|these|those|the\s+\w+)|\s+were\b)")
    for word, value in _WORD_NUMBERS.items():
        if _re.search(rf"\b{word}\b{count_ctx}", answer, _re.I) and not supported(float(value)):
            bad.append(word)
    for match in _re.finditer(rf"\b(\d+){count_ctx}", answer, _re.I):
        token = match.group(1)
        try:
            value = float(token)
        except ValueError:
            continue
        if not supported(value):
            bad.append(token)
    return sorted(set(bad))


def guard_answer(answer: str, facts: str, correction: str | None = None) -> tuple[str, list[str]]:
    """Return the answer, with a SHORT correction appended if it invented figures.

    An earlier version pasted the whole fact block underneath the reply. It
    was accurate and horrible to read -- a wall of bullets stapled to a
    conversational answer, which defeats the point of having written the
    answer conversationally. The caller supplies one sentence naming the
    figures that actually matter; the full detail belongs in the trace, where
    someone debugging can find it and a reader is not confronted by it.
    """
    bad = unsupported_numbers(answer, facts)
    if not bad:
        return answer, []
    if correction:
        return f"{answer}\n\n*(Correction: {correction})*", bad
    return f"{answer}\n\n*(Some figures above didn't match the computed data.)*", bad
