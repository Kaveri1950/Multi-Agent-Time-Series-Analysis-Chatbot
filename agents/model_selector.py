"""Choose WHICH model answers a query.

The project already routes by domain. This adds the second axis -- the task
-- and turns "run the whole finance pipeline" into "run the one model that
answers this question".

THREE STRATEGIES, because which one is best is an empirical question the
thesis should answer rather than assume:

  filter_only : deterministic. Classify the task by pattern, keep the models
                that can physically answer, take the highest measured skill.
                No LLM call.
  llm_only    : hand the model the whole registry and let it choose freely.
  hybrid      : the capability filter narrows the field, the LLM picks
                within it, and a validator rejects anything off the list.

Keeping all three callable is the point. `evaluation/` can then measure
selection accuracy across them on one labelled query set, so "the agentic
layer earns its place" becomes a number instead of an assertion -- and if
the deterministic filter matches the hybrid, that is a finding too.

WHY THE FILTER COMES FIRST IN HYBRID: this project has repeatedly found the
3B local model unreliable at structured choice -- it invented tickers,
misread pronoun follow-ups, and picked the wrong target when asked to
compare 20 rows. Every one of those was fixed by constraining the model
rather than trusting it. Selection is the same shape of problem, so the
shortlist is built deterministically and the LLM only ever orders it.
"""

import json
import re

from agents.model_registry import (ASSESS_RISK, COMPARE_MODELS, DETECT_ANOMALY,
                                   EVALUATE_MODEL, EXTRACT_SERIES, FORECAST_EVENT_COUNT,
                                   FORECAST_EVENT_PROBABILITY, FORECAST_VALUE,
                                   LOOKUP_FACT, REGISTRY, SCORE_SENTIMENT,
                                   TASK_DESCRIPTIONS, by_id, describe_for_prompt, feasible)

# --------------------------------------------------------------------------
# task classification -- patterns first, LLM second
# --------------------------------------------------------------------------

# Order matters: the first match wins, and the more specific intents are
# listed above the more general ones. "how many flares should I expect" is a
# count question AND contains a probability word, so count must be tested
# first or it would be swallowed.
_TASK_PATTERNS: list[tuple[str, re.Pattern]] = [
    (FORECAST_EVENT_COUNT, re.compile(
        r"\bhow\s+many\b.*\b(expect|likely|will|should|flares?|events?)\b"
        r"|\b(?:number|count|tally)\s+of\s+(?:expected\s+)?(?:solar\s+)?(?:flares?|events?)\b"
        r"|\bhow\s+many\s+(flares?|events?)\b", re.IGNORECASE)),

    (COMPARE_MODELS, re.compile(
        r"\bdoes\s+(sentiment|news|adding)\b.*\b(help|improve|matter)\b"
        r"|\bwith\s+(?:and|or)\s+without\b|\bcompare[sd]?\s+(the\s+)?(models?|approaches?)\b"
        r"|\bis\s+\w+\s+better\s+than\b", re.IGNORECASE)),

    (EVALUATE_MODEL, re.compile(
        r"\bhow\s+(accurate|good|reliable|well)\b"
        r"|\b(rmse|error rate|backtest|skill score|model accuracy|how\s+much\s+can\s+i\s+trust)\b"
        r"|\bhow\s+well\s+does\s+(it|the model)\b", re.IGNORECASE)),

    (LOOKUP_FACT, re.compile(
        r"\b(does|do)\b.*\bhave\s+(any\s+)?planets?\b"
        r"|\bhow\s+many\s+planets?\b|\bknown\s+planets?\b"
        r"|\b(confirmed|false\s+positive)\b|\bwhat\s+planets?\b"
        r"|\b(?:find|found|discover(?:ed)?)\s+(?:a\s+)?planets?\b"
        r"|\borbital\s+period\b", re.IGNORECASE)),

    (DETECT_ANOMALY, re.compile(
        r"\b(anomal\w*|unusual|abnormal\w*|outliers?|suspicious|irregular\w*"
        r"|weird|strange|odd)\b", re.IGNORECASE)),

    # ASSESS_RISK must be tested BEFORE forecast_event_prob, because that
    # pattern claims the bare word "risk" and would otherwise swallow every
    # risk-assessment query. The two are genuinely different questions:
    #
    #     forecast_event_prob : "risk of a geomagnetic storm"  -- will an
    #                           EVENT occur?
    #     assess_risk         : "how risky is TCS"             -- how much
    #                           could I LOSE holding this?
    #
    # So this pattern deliberately never matches a bare "<noun> risk" or
    # "risk of <event>". It requires either an explicit risk-assessment term
    # (downside, drawdown, value at risk, expected shortfall), a "how
    # risky/safe" construction, or risk paired with a holding/investing verb.
    # Checked against the whole labelled set: it steals none of the existing
    # 123 queries, including "risk of a geomagnetic storm" and
    # "flare risk today", which stay on forecast_event_prob.
    #
    # Every "lose" is anchored with \b -- an unanchored one matches the "lose"
    # inside "close", which would hijack "what will RELIANCE close at in
    # 30 days" away from forecast_value.
    (ASSESS_RISK, re.compile(
        r"\bhow\s+risky\b|\briskiness\b"
        r"|\bhow\s+(?:safe|dangerous)\s+is\b"
        r"|\brisk\s+(?:profile|metrics?|assessment|exposure)\b"
        r"|\brisk\s+of\s+(?:invest\w*|buy\w*|hold\w*|own\w*|putting)\b"
        r"|\b(?:how\s+much|what)\s+(?:can|could|might)\s+i\s+lose\b"
        r"|\bcould\s+i\s+lose\b|\bhow\s+much\s+.{0,20}\blose\b"
        r"|\bdownside\b|\bdrawdown\b"
        r"|\bvalue\s+at\s+risk\b|\bVaR\b|\bexpected\s+shortfall\b"
        r"|\bworst[-\s]case\s+(?:loss|scenario)\b"
        # up to two words between the adjective and the verb, so "a safe
        # stock to hold" is caught as well as the bare "safe to hold"
        r"|\b(?:safe|risky)\s+(?:\w+\s+){0,2}to\s+(?:hold|buy|invest|own)\b"
        r"|\blos(?:e|ing)\s+money\b"
        r"|\bhow\s+much\s+(?:risk|downside)\b", re.IGNORECASE)),

    (FORECAST_EVENT_PROBABILITY, re.compile(
        r"\b(chance|probability|likelihood|odds|risk)\b"
        r"|\bwill\s+there\s+be\b|\bhow\s+likely\b", re.IGNORECASE)),

    (SCORE_SENTIMENT, re.compile(
        r"\b(sentiment|news|headlines?|mood|market\s+feeling)\b"
        r"|\bwhat\s+are\s+people\s+saying\b", re.IGNORECASE)),

    (FORECAST_VALUE, re.compile(
        # "close", "trade" and "open" belong here: "what will TCS close at
        # tomorrow" is an unmistakable forecast request that matched nothing
        # while the list stopped at "be|price|cost|worth|nav", and fell
        # through to the general pipeline, which answered with backtest RMSEs.
        r"\bwhat\s+will\b.*\b(be|price|cost|worth|nav|close|trade|open|reach|hit)\b"
        r"|\b(predict|forecast|project)\b"
        # A horizon clause. "in 30 days" was the only form accepted, so
        # "what is the price of TCS for the next 30 days" -- and three other
        # natural phrasings -- classified as nothing and fell through to the
        # general pipeline, which answered with backtest RMSEs instead of a
        # forecast. People say for/over/next at least as often as "in".
        # Up to two arbitrary words between the preposition and the number,
        # rather than a fixed "the next". That covers "for the next 30 days",
        # "over the coming 30 days", and typos like "for nect 30 days" --
        # which a literal pattern rejects even though the horizon is
        # unmistakable. The same trick is already used for spelled-out
        # horizons elsewhere in the project ("next THIRTY days").
        # Safe against anomaly queries because DETECT_ANOMALY is tested
        # first, so "anomalies in TCS over the last 30 days" still routes
        # to detection.
        r"|\b(?:in|for|over|within|during)\s+(?:\w+\s+){0,2}\d+\s+(?:days?|weeks?|months?)\b"
        r"|\bnext\s+\d+\s+(?:days?|weeks?|months?)\b"
        # ...and without a number at all: "price for next month".
        r"|\bprice\b.*\b(?:for|over)\s+(?:the\s+)?next\s+(?:day|week|month|quarter|year)\b"
        r"|\bprice\s+target\b|\bwhere\s+is\s+\w+\s+head", re.IGNORECASE)),
]


# Definitional phrasing: a question about a CONCEPT rather than about any
# particular star or stock.
_DEFINITIONAL = re.compile(
    r"^\s*(?:what\s+(?:is|are|does|do)\b|what'?s\s+(?:a|an|the)\b|explain\b"
    r"|how\s+(?:does|do)\b|why\s+(?:do|does|are|is)\b|define\b|tell me about\b)",
    re.IGNORECASE)

# Anything naming a specific entity disqualifies a question from being
# general: "what is the orbital period of KIC 8544996" is a catalogue lookup,
# not a definition, even though it opens the same way.
_ENTITY_MENTION = re.compile(
    r"\b(?:KIC|TIC|EPIC|KPLR)\s*\d+"            # a star
    r"|\b[A-Za-z]{1,10}\.(?:NS|BO|L|TO|HK|AX)\b"  # an exchange-suffixed ticker
    r"|\b(?:nifty|sensex|reliance|tcs|infy|hdfcbank|sbin|wipro|itc|ongc|tatamotors)\b",
    re.IGNORECASE)

# Domain vocabulary that LOOKS like a ticker but is a concept. Without this,
# "what does RSI mean" reads as a question about a company called RSI.
_CONCEPT_TERMS = {"RSI", "MACD", "NAV", "CME", "ARIMA", "AUC", "TSS", "SEBI",
                  "GOES", "NOAA", "KOI", "BLS", "IPO", "ETF", "PE", "EPS"}


def is_general_question(message: str) -> bool:
    """Should NO model run for this?

    Definitional questions are answerable from general knowledge, and firing
    a pipeline at one costs minutes to answer what a sentence would. Measured
    at 0% before this existed: every "what is a light curve" ran
    IsolationForest over a four-year light curve first.

    Deliberately conservative -- it requires definitional phrasing AND the
    absence of any named entity. Misclassifying a real data question as
    general would answer it from the model's own knowledge, which is exactly
    the hallucination the grounding prompts exist to prevent; the reverse
    error only wastes time.
    """
    if not _DEFINITIONAL.match(message):
        return False
    if _ENTITY_MENTION.search(message):
        return False
    # an unknown all-caps token is probably a ticker, not a concept
    for token in re.findall(r"\b[A-Z]{2,10}\b", message):
        if token not in _CONCEPT_TERMS:
            return False
    return True


def classify_task(message: str, has_image: bool = False) -> tuple[str | None, str]:
    """(task, how_it_was_decided). Returns (None, ...) when nothing matches."""
    if has_image:
        return EXTRACT_SERIES, "image attached"
    for task, pattern in _TASK_PATTERNS:
        if pattern.search(message):
            return task, "pattern"
    if is_general_question(message):
        return None, "general/definitional -- no model needed"
    return None, "no pattern matched"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

SELECTOR_SYSTEM_PROMPT = """You choose which analysis model should answer a user's question.

You are given a SHORTLIST of models that are already known to be capable of running for this
question, each with what it does, its MEASURED skill, and when it should be avoided.

Rules:
1. Reply with ONLY a JSON object: {"model_id": "<exact id from the shortlist>", "why": "<one short sentence>"}
2. The model_id MUST be copied exactly from the shortlist. Never invent one.
3. Read the "avoid when" line. A model that measured at chance, or that is documented as unsuitable
   for this kind of question, must not be chosen just because it sounds relevant.
4. Prefer a recorded fact over a model's inference when both could answer.
5. Prefer the model with better measured skill when two genuinely fit.
Output strictly valid JSON and nothing else."""


# Metrics where a HIGHER number is better. Anything else (RMSE, Brier, MAE)
# is better when lower, and metrics from different families are not
# comparable at all.
_HIGHER_IS_BETTER = {"tss", "roc-auc", "auc", "interval coverage", "accuracy",
                     "f1", "precision", "recall", "round-trip correlation"}
_LOWER_IS_BETTER = {"rmse", "brier", "mae", "rmse delta"}


def _skill_value(spec) -> float:
    """A sortable score, or -inf when the entry cannot be ranked.

    Comparing raw numbers across metrics is meaningless and was actively
    wrong here: `interval coverage 0.825` was being ranked against an RMSE,
    two quantities with opposite senses and different units. It chose the
    right model by luck; with a Brier score in the mix it would have chosen
    the worst one.

    Lower-is-better metrics are negated so one sort works for both, and an
    unmeasured or non-numeric entry sorts last rather than pretending to be
    zero.
    """
    skill = spec.get("skill", {}) or {}
    value = skill.get("value")
    if not isinstance(value, (int, float)):
        return float("-inf")
    metric = str(skill.get("metric", "")).strip().lower()
    if metric in _LOWER_IS_BETTER:
        return -float(value)
    if metric in _HIGHER_IS_BETTER:
        return float(value)
    return float("-inf")        # unknown metric: not rankable


def _pick_best(shortlist: list):
    """Highest measured skill, with a stable tiebreak.

    When nothing on the shortlist has a rankable score -- which is the normal
    case for finance, where accuracy is reported per-ticker at query time --
    fall back to registry order rather than to whichever model happened to
    sort first.
    """
    ranked = sorted(shortlist, key=lambda m: (_skill_value(m), -REGISTRY.index(m)),
                    reverse=True)
    return ranked[0]


def select_model(message: str, domain: str | None = None, available: set | None = None,
                 strategy: str = "hybrid", has_image: bool = False,
                 backend: str | None = None, model: str | None = None) -> dict:
    """Pick one model for this query.

    Returns {model_id, spec, task, strategy, how, shortlist, fallback_used, trace}.
    Never raises on an LLM failure -- an unreachable or confused model falls
    back to the best-measured feasible option, and says so in `how`, because
    a selector that crashes is worse than one that is predictable.
    """
    available = set(available or set())
    trace = []

    task, how_task = classify_task(message, has_image=has_image)
    trace.append(f"[selector] task={task or 'unknown'} ({how_task})")

    # Declining to run anything is a valid, and sometimes the only correct,
    # outcome. A definitional question needs no data, and answering it by
    # running a pipeline is minutes spent to say what a sentence would.
    if task is None and "general" in how_task:
        return {"model_id": None, "spec": None, "task": None, "strategy": strategy,
                "how": "general/definitional question -- answer from knowledge, run no model",
                "shortlist": [], "fallback_used": False, "trace": trace}

    shortlist = feasible(domain=domain, task=task, available=available)
    if not shortlist:
        # Task classification may be wrong or the task may be unsupported in
        # this domain; fall back to whatever the domain can do at all rather
        # than refusing outright.
        shortlist = feasible(domain=domain, available=available)
        trace.append(f"[selector] no model for that task in '{domain}' -- "
                     f"widened to {len(shortlist)} domain model(s)")

    if not shortlist:
        trace.append("[selector] nothing feasible")
        return {"model_id": None, "spec": None, "task": task, "strategy": strategy,
                "how": "no feasible model", "shortlist": [], "fallback_used": True,
                "trace": trace}

    if strategy == "filter_only":
        best = _pick_best(shortlist)
        trace.append(f"[selector] filter_only -> {best['id']} (best measured skill)")
        return _result(best, task, strategy, "capability filter + measured skill",
                       shortlist, False, trace)

    pool = REGISTRY if strategy == "llm_only" else shortlist
    if strategy == "llm_only":
        trace.append(f"[selector] llm_only -- unconstrained over all {len(REGISTRY)} models")

    chosen, note, fell_back = _ask_llm(message, pool, backend=backend, model=model)
    if chosen is None:
        best = _pick_best(shortlist)
        trace.append(f"[selector] LLM unusable ({note}) -- fell back to {best['id']}")
        return _result(best, task, strategy, f"fallback: {note}", shortlist, True, trace)

    if strategy == "hybrid" and chosen["id"] not in {m["id"] for m in shortlist}:
        # The validator. In hybrid mode an off-shortlist answer is rejected
        # outright rather than trusted, which is the whole reason the
        # shortlist is built first.
        best = _pick_best(shortlist)
        trace.append(f"[selector] REJECTED '{chosen['id']}' -- not on the shortlist; "
                     f"using {best['id']}")
        return _result(best, task, strategy, "validator rejected the LLM's choice",
                       shortlist, True, trace)

    trace.append(f"[selector] {strategy} -> {chosen['id']} ({note})")
    return _result(chosen, task, strategy, note, shortlist, fell_back, trace)


def _result(spec, task, strategy, how, shortlist, fallback_used, trace) -> dict:
    return {
        "model_id": spec["id"], "spec": spec, "task": task, "strategy": strategy,
        "how": how, "shortlist": [m["id"] for m in shortlist],
        "fallback_used": fallback_used, "trace": trace,
    }


def _ask_llm(message: str, pool: list, backend=None, model=None):
    """Returns (spec | None, note, fell_back)."""
    if not pool:
        return None, "empty pool", True
    try:
        from agents.llm_backend import call_llm

        prompt = (f"SHORTLIST:\n{describe_for_prompt(pool)}\n\n"
                  f"USER QUESTION: {message}\n\n"
                  f"Which model should answer? Reply with JSON only.")
        raw = call_llm(prompt, system=SELECTOR_SYSTEM_PROMPT, backend=backend)
    except Exception as e:
        return None, f"{type(e).__name__}", True

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, "no JSON in reply", True
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid JSON", True

    spec = by_id(str(parsed.get("model_id", "")).strip())
    if spec is None:
        return None, f"unknown id '{parsed.get('model_id')}'", True
    return spec, str(parsed.get("why", "LLM choice"))[:160], False


def explain_selection(result: dict) -> str:
    """Human-readable, for the dashboard trace and the report's appendix."""
    if not result.get("model_id"):
        return "No model could answer that question."
    spec = result["spec"]
    lines = [
        f"Selected: {spec['name']}  ({spec['id']})",
        f"  task      : {result['task']} -- {TASK_DESCRIPTIONS.get(result['task'], 'unclassified')}",
        f"  strategy  : {result['strategy']}",
        f"  reason    : {result['how']}",
        f"  considered: {len(result['shortlist'])} feasible model(s)",
    ]
    skill = spec.get("skill", {})
    if skill.get("value") is not None:
        lines.append(f"  skill     : {skill.get('metric')} {skill['value']}")
    if result.get("fallback_used"):
        lines.append("  note      : fell back to the best-measured option")
    return "\n".join(lines)


if __name__ == "__main__":
    EXAMPLES = [
        ("what's the chance of a solar flare tomorrow", "solar", set()),
        ("how many flares should I expect this week", "solar", {"horizon_hours"}),
        ("what will RELIANCE close at in 30 days", "finance", {"ticker", "horizon_days"}),
        ("how accurate is your price model", "finance", {"ticker"}),
        ("does news sentiment actually help prediction", "finance", {"ticker"}),
        ("any anomalies in RELIANCE", "finance", {"ticker"}),
        ("anything unusual in KIC 10874614", "astronomy", {"target"}),
        ("does KIC 10874614 have any confirmed planets", "astronomy", {"target"}),
    ]

    print("Task classification + capability filter (no LLM call)\n" + "=" * 68)
    for msg, domain, avail in EXAMPLES:
        r = select_model(msg, domain=domain, available=avail, strategy="filter_only")
        print(f"\n{msg!r}")
        print(f"   -> {r['model_id']}")
        print(f"      task={r['task']}  shortlist={r['shortlist']}")
