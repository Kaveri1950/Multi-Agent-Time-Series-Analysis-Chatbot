"""Chatbot intent layer: turns a free-form chat message into a domain +
ticker/target + question focus, then either answers directly (general
knowledge) or drives the existing LangGraph pipeline (agents/graph.py) for
a focused answer -- reusing the same Finance/Sentiment/Astronomy/Explainer
nodes the rest of the app uses, not a separate implementation.

Multi-turn behavior:
  - If the message doesn't name a ticker/target, the previous turn's is
    reused (passed in as last_ticker/last_target).
  - A per-session pipeline cache (keyed by domain+ticker/target) means a
    follow-up question about the same stock/target re-runs only the
    Explainer step (cheap, ~seconds) instead of re-fetching price/news/light
    curve data and re-running IsolationForest/XGBoost/FinBERT again
    (expensive, ~10-30s) -- this is what makes "what about its RSI instead"
    feel like a real conversation rather than a fresh query each time.
"""

import re

import pandas as pd

from agents.model_registry import FORECAST_VALUE
from agents.model_selector import classify_task, select_model

from agents.json_utils import extract_json
from agents.llm_backend import call_llm
from agents.prompts import ASTRONOMY_EXPLAINER_SYSTEM_PROMPT, FINANCE_EXPLAINER_SYSTEM_PROMPT
from finance.ticker_search import looks_like_explicit_ticker, resolve_ticker

# Detects a genuine "what will the price be" forecast request, as opposed
# to an RSI/sentiment/overview question -- these need forecast_future()
# (a real forward-looking ARIMA projection + chart), not the standard
# Finance/Sentiment/Explainer pipeline, which only backtests already-known
# history and returns text.
# \w* (not \d*) between "next" and "day/week/month" is deliberate --
# observed in testing: "next THIRTY days" (spelled out, not "30") is a very
# natural phrasing and a digit-only pattern silently missed it entirely.
_FORECAST_PATTERN = re.compile(
    r"\b(predict|prediction|forecast|future price|price target|"
    r"will.*price|price.*will|next\s+(?:\w+\s+){0,2}(day|days|week|weeks|month|months))\b", re.IGNORECASE,
)
_DAYS_PATTERN = re.compile(r"(\d+)\s*days?", re.IGNORECASE)

# Whether the USER's raw text (not the LLM's JSON output) already contains
# an explicit exchange-suffixed ticker. Observed in testing: asked about
# "reliance" (no suffix), the small model's own intent JSON silently
# auto-completed ticker_or_target to "RELIANCE.NS" -- which happens to be
# the intended company here, but the same model would silently guess wrong
# just as easily (Reliance Industries vs the unrelated Reliance
# Infrastructure, both real NSE listings). Trust an explicit suffix only
# when the *user* actually typed it; otherwise always re-derive from the
# bare name via a real search so genuine ambiguity surfaces instead of
# getting silently paved over by the model's autocomplete.
_EXPLICIT_SUFFIX_IN_TEXT = re.compile(r"\b[A-Za-z]{1,10}\.(NS|BO|L|TO|HK|AX|BA)\b", re.IGNORECASE)

# A question about the whole downloaded light-curve dataset (astronomy/
# dataset.py), not one specific KIC ID -- answered from the precomputed
# index, not a live per-target fetch. Checked before LLM intent parsing,
# same reasoning as the image-upload path: this is unambiguous from
# keywords alone, no need to spend an LLM call classifying it.
_DATASET_QUERY_PATTERN = re.compile(
    r"\b(my dataset|my (?:light curves?|targets?)|downloaded light curves?|"
    r"which targets?|how many (?:light curves?|targets?)|"
    r"across (?:all|my) targets?|list (?:my |the )?targets?|"
    r"most anomalous target|compare targets?)\b", re.IGNORECASE,
)

# Domain-agnostic anomaly-intent detector -- the same command surface
# ("any anomalies?", "anything unusual?") should work whether the current
# conversation is about a stock or a light curve. Matched against both the
# raw message AND used to force question_focus, so the small model's own
# focus-extraction (unreliable, see README design notes) can't silently
# drop the anomaly framing the user actually asked for.
# A risk question is not an anomaly question. "Which days were odd?" looks
# backwards at what already happened; "how risky is this?" looks forward at
# what could. They need different models and different answers, so they are
# separated before routing rather than after.
_RISK_QUERY_PATTERN = re.compile(
    r"\b(risk(y|s)?|volatil(e|ity)|how\s+safe|safe\s+to|downside|drawdown|"
    r"lose|loss|could\s+i\s+lose|worst\s+case|exposure|should\s+i\s+(invest|buy|put)|"
    # Nobody types "run anomaly detection". They ask what happened and why,
    # and the risk profile is the better answer to that than a list of
    # flagged dates: it reports the unusual days AND attributes each one to
    # the market or the company, which a bare detector cannot.
    r"what\s+(happened|went\s+(on|wrong))|why\s+did\s+\w+\s+(drop|fall|crash|jump|rise|spike|move|tank)|"
    r"unusual\s+(day|days|move|moves)|anything\s+(strange|odd|unusual)|"
    r"bad\s+days|worst\s+days|big\s+(move|moves|drop|drops))\b",
    re.IGNORECASE,
)

# A question about the MARKET'S risk, or about which names are riskiest. It
# must be tested before the single-stock risk pattern, which would otherwise
# match "riskiest" and answer about whatever ticker the previous turn left
# in the session.
_MARKET_RISK_PATTERN = re.compile(
    r"\b(?:riskiest|most\s+(?:risky|volatile)|least\s+(?:risky|volatile)|safest)\b"
    r"|\b(?:which|what)\s+(?:stocks?|shares?|names?|companies)\b.{0,40}"
    r"\b(?:risk\w*|volatil\w*|safe\w*)"
    r"|\b(?:market|whole\s+market|overall\s+market|nifty)\s+risk\b"
    r"|\bhow\s+(?:risky|volatile)\s+is\s+the\s+(?:market|nifty)\b"
    r"|\brank\w*\b.{0,30}\b(?:risk|volatil\w*)",
    re.IGNORECASE,
)

_ANOMALY_QUERY_PATTERN = re.compile(
    # anomal\w* (not a literal "anomal\b") is deliberate -- found in
    # testing: "anomal" followed by a required \b never matches "anomaly"
    # or "anomalies" at all, since there's no word boundary between "l" and
    # the "y"/"ies" that always follows it in real English. \w* consumes
    # the suffix so the trailing \b lands after it instead.
    r"\b(anomal\w*|unusual|abnormal\w*|outliers?|suspicious|irregular\w*|"
    r"weird (?:movement|behavior|activity)|strange (?:movement|behavior))\b", re.IGNORECASE,
)

# Explicit session-level domain pin ("switch to astronomy mode") so a bare
# follow-up with no named ticker/target ("any anomalies?") has somewhere to
# resolve to besides whatever the last turn happened to be about.
_MODE_PIN_PATTERN = re.compile(
    r"\b(?:switch|change|set)\s+(?:to\s+)?(finance|astronomy)\s*mode\b|"
    r"\blet'?s\s+(?:talk|do|focus on)\s+(finance|astronomy)\b", re.IGNORECASE,
)
# Common spelled-out horizons -- not a full number parser, just the round
# numbers people actually say when asking for a price forecast.
_WORD_DAY_COUNTS = {
    "week": 7, "fortnight": 14, "two weeks": 14, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fortyfive": 45, "fifty": 50,
    "sixty": 60, "ninety": 90, "month": 30, "quarter": 90, "three months": 90,
}


# Solar-domain safety net. The project already layers regex over the LLM's
# intent JSON for ticker extraction and resize requests (see README design
# notes on small-model reliability); solar needs the same treatment, and more
# so because it is the one domain with no entity id for the model to latch
# onto. Deliberately requires a solar/space-weather word: a bare "flare" is
# ambiguous, since stellar flares belong to the astronomy domain.
_SOLAR_QUERY_PATTERN = re.compile(
    r"\b(?:solar\s+(?:flare|activity|weather|cycle|wind|storm|event)s?"
    r"|space\s+weather|geomagnetic|sunspots?|coronal|\bCMEs?\b"
    r"|(?:x|m)-?class\s+flares?|flare\s+(?:forecast|probability|prediction)"
    r"|solar\s+prediction|will\s+there\s+be\s+a\s+flare)\b",
    re.IGNORECASE,
)

# "the sun" only counts when a KIC/TIC id is absent, so "compare the sun to
# KIC 8462852" stays in astronomy rather than being hijacked.
_SUN_MENTION = re.compile(r"\bthe\s+sun\b", re.IGNORECASE)
_STELLAR_ID = re.compile(r"\b(?:KIC|TIC|EPIC)\s*\d+", re.IGNORECASE)


# An entity that is clearly NOT a star: a bare all-caps ticker like TCS or
# HDFCBANK, or an exchange-suffixed one. Concept words (RSI, MACD) and the
# catalogue prefixes are excluded, so "what does RSI mean" and
# "anything unusual in KIC 5358624" both fall through untouched.
_NON_STELLAR_ENTITY = re.compile(
    r"\b(?!KIC\b|TIC\b|EPIC\b|KPLR\b|RSI\b|MACD\b|NAV\b|CME\b|ARIMA\b|AUC\b|TSS\b"
    r"|SEBI\b|GOES\b|NOAA\b|KOI\b|BLS\b|IPO\b|ETF\b|EPS\b|NSE\b|BSE\b|USD\b|INR\b)"
    r"[A-Z]{2,12}(?:\.(?:NS|BO|L|TO|HK|AX))?\b")

# A bare "flare" with no qualifier. Ambiguous in principle -- stellar flares
# belong to astronomy -- but not in practice: the astronomy domain always
# addresses a specific star by KIC/TIC id, so a flare question carrying no
# stellar id is about the Sun. Found by testing "how many flares should I
# expect this week", which silently routed to finance because it happens to
# contain no solar keyword at all.
_BARE_FLARE = re.compile(r"\bflares?\b", re.IGNORECASE)


# Vocabulary that identifies what a GENERAL question is about, when no
# entity is named. Kept to words that belong unambiguously to one field --
# "model" and "data" appear in both and are deliberately absent.
_FINANCE_WORDS = re.compile(
    r"\b(price|stock|share|ticker|market|trading|volatility|rsi|macd|"
    r"sentiment|news|earnings|dividend|portfolio|nifty|sensex|nav|"
    r"mutual\s+fund|equity|invest\w*|rupee|sebi|arima|forecast\w*)\b",
    re.IGNORECASE)

_ASTRO_WORDS = re.compile(
    r"\b(light\s*curves?|transits?|exoplanets?|planets?|stars?|stellar|kepler|"
    r"telescopes?|flux|orbit\w*|astronom\w*|brightness|eclips\w*|kic|tic)\b",
    re.IGNORECASE)


def _general_question_domain(message: str):
    """Which field is this definitional question about? None if unclear.

    Only decides when the evidence is one-sided. A question mentioning both
    fields ("does sentiment analysis work on light curves") gets None, so the
    caller falls back to conversation context rather than this guessing.
    """
    fin = len(_FINANCE_WORDS.findall(message))
    ast = len(_ASTRO_WORDS.findall(message))
    if fin and not ast:
        return "finance"
    if ast and not fin:
        return "astronomy"
    return None


def _looks_solar(message: str) -> bool:
    if _STELLAR_ID.search(message):
        return False
    if _SOLAR_QUERY_PATTERN.search(message) or _SUN_MENTION.search(message):
        return True
    return bool(_BARE_FLARE.search(message))


# Ways people say "one day ahead". Checked before the numeric and word
# parsers, because the default of 30 otherwise swallows them: "what will TCS
# close at tomorrow" was being treated as a 30-day horizon, which meant a
# next-session question never reached the pre-computed daily forecast log and
# instead refitted ARIMA for a month-ahead projection nobody asked for.
_NEXT_DAY_PATTERN = re.compile(
    r"\b(tomorrow|next\s+(?:session|trading\s+day|day)|"
    r"one\s+day|1\s+day|next\s+close|by\s+tomorrow)\b", re.IGNORECASE)


def _extract_forecast_days(message: str, default: int = 30) -> int:
    if _NEXT_DAY_PATTERN.search(message):
        return 1
    match = _DAYS_PATTERN.search(message)
    if match:
        return max(1, min(int(match.group(1)), 180))  # sane bounds -- ARIMA gets unreliable far out anyway
    lower = message.lower()
    for word, days in _WORD_DAY_COUNTS.items():
        if word in lower:
            return days
    return default


def _try_resolve_disambiguation(message: str, candidates: list[dict]) -> dict | None:
    """Does this reply look like a selection among previously-offered
    candidates? Checks a leading number ("2", "2.", "option 2"), then an
    explicit ticker symbol mention, then a company-name-first-word mention."""
    text = message.strip()

    num_match = re.match(r"^\W*(\d+)\b", text)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]

    text_upper = text.upper()
    for c in candidates:
        if c["symbol"].upper() in text_upper:
            return c
    for c in candidates:
        first_word = c["name"].split()[0].upper()
        if len(first_word) > 2 and first_word in text_upper:
            return c
    return None


def _format_disambiguation_answer(candidates: list[dict], original_query: str) -> str:
    lines = [f"{i + 1}. **{c['symbol']}** -- {c['name']} ({c['exchange']})" for i, c in enumerate(candidates)]
    return (f"I found multiple matches for \"{original_query}\":\n\n" + "\n".join(lines) +
            "\n\nWhich one did you mean? Reply with the number or the ticker.")

# Small local models (e.g. llama3.2:3b) are unreliable at resolving pronoun
# follow-ups ("what about ITS sentiment") -- observed in testing: the model
# classified such a message as domain="general" with zero context, and the
# general-knowledge fallback (no ticker/target in its prompt) hallucinated
# about an unrelated topic. If the model calls something "general" right
# after a finance/astronomy turn AND the message reads like a referential
# follow-up, trust the conversation state over the model's classification.
# "what about the RSI" (no pronoun, but still clearly a follow-up) was
# observed falling through this net in testing -- broadened to also catch
# the "what about the <metric>" phrasing generically.
_REFERENTIAL_PATTERN = re.compile(
    r"\b(it|its|it's|that|this|the stock|the target|the fund|instead|what about the)\b", re.IGNORECASE,
)

# Same lesson as agents/dashboard_agent.py: don't leave an explicit,
# unambiguous named entity (an Indian index name) entirely up to the small
# model's JSON output -- detect it directly and let it win.
_INDIA_INDEX_PATTERNS = [
    (re.compile(r"\bnifty\s*-?\s*500\b", re.IGNORECASE), "^CRSLDX"),
    (re.compile(r"\bnifty\s*-?\s*100\b", re.IGNORECASE), "^CNX100"),
    (re.compile(r"\bnifty\s*-?\s*50\b", re.IGNORECASE), "^NSEI"),
    (re.compile(r"\bsensex\b", re.IGNORECASE), "^BSESN"),
]


def _detect_india_index(message: str) -> str | None:
    for pattern, ticker in _INDIA_INDEX_PATTERNS:
        if pattern.search(message):
            return ticker
    return None


INTENT_SYSTEM_PROMPT = """You are the intent-parsing layer of a multi-agent finance+astronomy+solar assistant.
Given the user's message and the conversation's previous context, output ONLY a JSON object, no other text:
{
  "domain": "finance" | "astronomy" | "solar" | "mutual_fund" | "general",
  "ticker_or_target": string or null,
  "question_focus": short phrase describing exactly what they want to know (e.g. "RSI", "sentiment", "price forecast", "anomalies", "returns"),
  "needs_data": true or false
}
Rules:
- "finance" = questions about a stock/ticker/market/index, including Indian indices (Nifty 50/100/500, Sensex) and Indian stocks (e.g. "RELIANCE.NS"). "astronomy" = questions about a light curve / Kepler ID / TIC ID / stellar anomaly. "mutual_fund" = questions about a NAMED Indian mutual fund scheme (e.g. "SBI Large Cap fund", "HDFC Mid-Cap") -- for this domain, ticker_or_target should be the fund name text as typed, not a ticker. "solar" = questions about THE SUN and space weather: solar flares, flare probability/forecast, sunspots, active regions, solar cycle, CMEs, geomagnetic storms, X-class/M-class events. For this domain ticker_or_target must be null -- there is only one Sun, so there is no entity to name. IMPORTANT: a question about a *stellar* flare on another star (with a KIC/TIC id) is "astronomy", NOT "solar". "general" = conceptual/definitional questions not tied to a specific live ticker/target/fund (e.g. "what does RSI mean").
- needs_data should be false ONLY for "general" questions that need no live lookup.
- If the user doesn't name a ticker/target/fund this turn but one was discussed previously (given below), reuse it UNLESS the message clearly changes subject.
- ticker_or_target must be a clean symbol like "AAPL" or "RELIANCE.NS", a clean id like "KIC 11446443", or (for mutual_fund) the fund name text, or null if truly none.
Output strictly valid JSON and nothing else."""


def interpret_message(message: str, last_domain: str | None = None, last_ticker: str | None = None,
                       last_target: str | None = None, backend: str | None = None, model: str | None = None) -> dict:
    context = f"Previous domain: {last_domain}. Previous ticker: {last_ticker}. Previous astronomy target: {last_target}."
    prompt = f"{context}\n\nUser message: \"{message}\"\n\nJSON:"

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model

    raw = call_llm(prompt, system=INTENT_SYSTEM_PROMPT, **kwargs)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = {}

    parsed.setdefault("domain", last_domain or "general")
    parsed.setdefault("ticker_or_target", last_ticker if parsed.get("domain") == "finance" else last_target)
    parsed.setdefault("question_focus", message)
    parsed.setdefault("needs_data", bool(parsed.get("ticker_or_target")))
    return parsed


def _answer_dataset_query(message: str, backend: str | None = None, model: str | None = None) -> dict:
    """Dataset-level astronomy queries (e.g. "which of my targets has the
    most anomalies", "how many light curves do I have") -- answered from
    the precomputed index built by astronomy/dataset.py's build_dataset(),
    not a live per-target fetch."""
    from astronomy.dataset import load_index
    from dashboard.charts import dataset_anomaly_overview_chart

    try:
        index_df = load_index()
    except FileNotFoundError as exc:
        return {"answer": f"I don't have a light curve dataset built yet ({exc}). Ask your developer to run "
                           "`python -m astronomy.dataset` to download and index a batch of targets first.",
                "domain": "astronomy", "trace": ["[chat] no dataset index found"], "retrieved_context": [], "chart": None}

    ranked = index_df.sort_values("anomaly_rate", ascending=False)
    summary_cols = ["target", "source", "n_points", "n_windows", "n_flagged_anomalies", "anomaly_rate", "mean_reconstruction_error"]
    summary_rows = ranked[summary_cols].to_dict("records")

    # Precompute the two most common "which is highest" interpretations
    # deterministically rather than asking the model to scan and compare
    # ~20 rows itself. Observed in testing: asked "which target has the
    # most anomalies", the small 3B model picked a target with 84 flagged
    # windows while a different target (98) was the true max -- a real
    # numerical-comparison error, not a data problem. Also worth noting:
    # "highest rate" and "highest count" can disagree -- a light curve
    # with only 6 windows total can show a deceptively high anomaly RATE
    # (1/6 = 16.7%) purely from having so few windows to begin with, so
    # both are surfaced explicitly rather than picking one silently.
    by_rate = ranked.iloc[0]
    by_count = index_df.sort_values("n_flagged_anomalies", ascending=False).iloc[0]
    computed_facts = (
        f"COMPUTED FACTS (already correctly calculated -- use these directly, don't re-derive):\n"
        f"- Highest anomaly RATE: {by_rate['target']} at {by_rate['anomaly_rate']:.1%} "
        f"({by_rate['n_flagged_anomalies']} of {by_rate['n_windows']} windows -- note a small light curve "
        f"with few total windows can show a high rate from a small count alone).\n"
        f"- Highest anomaly COUNT: {by_count['target']} with {by_count['n_flagged_anomalies']} flagged windows "
        f"(out of {by_count['n_windows']} total)."
    )

    prompt = f"""Light curve dataset: {len(index_df)} targets, each with precomputed IsolationForest anomaly
detection (see astronomy/anomaly_detector.py).

{computed_facts}

Full per-target summary, sorted by anomaly rate (highest first):
{summary_rows}

The user asked: "{message}". Answer using only the data above -- prefer the COMPUTED FACTS for any
"which is highest/most" style question instead of re-comparing the full list yourself."""

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model
    answer = call_llm(prompt, system=ASTRONOMY_EXPLAINER_SYSTEM_PROMPT, **kwargs)

    chart = dataset_anomaly_overview_chart(ranked)
    return {"answer": answer, "domain": "astronomy",
            "trace": [f"[chat] answered dataset-level query over {len(index_df)} cached targets"],
            "retrieved_context": [], "chart": chart}


def _answer_image_lightcurve_query(image_bytes: bytes, image_name: str, focus: str, session_state: dict,
                                    backend: str | None = None, model: str | None = None,
                                    pre_trace: list | None = None) -> dict:
    """An uploaded light-curve plot image, run through the same
    IsolationForest + autoencoder pipeline as agents/nodes/astronomy_node.py
    (real KIC-ID targets) -- the only difference is the flux series comes
    from astronomy.image_lightcurve's pixel extraction instead of
    lightkurve/MAST. Builds the cached state in the exact shape
    explainer_node expects, so a text-only follow-up ("what about the
    depth of the biggest one") reuses it via the normal
    _answer_finance_or_astronomy cache-reuse path, no special-casing needed."""
    from astronomy.anomaly_detector import STRIDE, WINDOW_SIZE, extract_features, fit_isolation_forest, make_windows, score_anomalies
    from astronomy.image_lightcurve import extract_lightcurve_from_image
    from astronomy.ts_embedder import embed_windows, reconstruction_error, train_autoencoder
    from dashboard.charts import lightcurve_anomaly_chart

    trace = list(pre_trace or [])
    target_key = f"UPLOADED_IMAGE:{image_name}".upper()

    try:
        lc = extract_lightcurve_from_image(image_bytes)
    except ValueError as exc:
        trace.append(f"[chat] image extraction failed for '{image_name}': {exc}")
        return {"answer": f"I couldn't read a light curve out of this image: {exc}",
                "domain": "astronomy", "trace": trace, "retrieved_context": [], "chart": None}

    flux, time = lc["flux"].values, lc["time"].values
    windows = make_windows(flux)
    feats = extract_features(windows)
    iso_model = fit_isolation_forest(feats)
    scored = score_anomalies(iso_model, feats)
    anomaly_mask = scored["is_anomaly"].values

    pseudo_normal_mask = ~anomaly_mask
    embeddings, recon_errors = None, None
    if pseudo_normal_mask.sum() >= 10:
        ae_model = train_autoencoder(windows[pseudo_normal_mask])
        embeddings = embed_windows(ae_model, windows)
        recon_errors = reconstruction_error(ae_model, windows)

    top_anomalies = scored.sort_values("anomaly_score", ascending=False).head(5)
    astronomy_result = {
        "target": target_key,
        "data_source": (f"extracted from uploaded image '{image_name}' -- relative flux shape only, "
                         "time axis is a plain index, not real calendar dates"),
        "n_windows": len(windows),
        "n_flagged_anomalies": int(anomaly_mask.sum()),
        "top_anomaly_windows": top_anomalies.reset_index().rename(columns={"index": "window_id"}).to_dict("records"),
        "mean_reconstruction_error": float(recon_errors.mean()) if recon_errors is not None else None,
        "_embeddings": embeddings,
    }

    chart = lightcurve_anomaly_chart(time, flux, window_size=WINDOW_SIZE, stride=STRIDE, anomaly_mask=anomaly_mask)

    trace.append(f"[chat] extracted light curve from image '{image_name}': "
                 f"{len(windows)} windows, {int(anomaly_mask.sum())} flagged anomalous")

    # explainer_node's call_llm doesn't accept a model override (it always
    # falls back to BTP_OLLAMA_MODEL / "llama3:latest" otherwise) -- set it
    # via env var so an explicit `model` here (e.g. the faster 3B default)
    # is actually honored, same pattern used for `backend` just below.
    if backend or model:
        import os
        if backend:
            os.environ["BTP_LLM_BACKEND"] = backend
        if model:
            os.environ["BTP_OLLAMA_MODEL"] = model
    from agents.nodes.explainer_node import explainer_node
    state = explainer_node({"domain": "astronomy", "target": target_key, "astronomy_result": astronomy_result,
                             "question_focus": focus, "trace": trace})
    trace = state["trace"]

    session_state["last_domain"] = "astronomy"
    session_state["last_target"] = target_key
    cache = session_state.setdefault("_pipeline_cache", {})
    cache[f"astronomy:{target_key}"] = state

    return {"answer": state["report"], "domain": "astronomy", "trace": trace,
            "retrieved_context": state.get("retrieved_context", []), "chart": chart}


def _answer_price_forecast(ticker: str, focus: str, days: int, session_state: dict, backend: str | None = None,
                            model: str | None = None, pre_trace: list | None = None) -> dict:
    """Forward-looking price forecast + chart, via a Gaussian Process.

    Uses gpr_forecast_path(), not the ARIMA forecast_future() this used to
    call. The GP is fitted to RETURNS and the price path integrated from
    them, which is the formulation that behaves correctly on a near-random
    walk: the posterior mean over future returns is ~0, so the path stays
    flat at the last close, and the variance accumulates so the band widens
    as sqrt(t).

    Measured over 66 month-forecasts across 6 tickers: the 80% band contained
    the actual price 75.6% of the time, and the point forecast beat a naive
    "no change" baseline on exactly half of them. So the band is the useful
    output and the central line is not -- which the answer says out loud.
    """
    from dashboard.charts import future_forecast_chart
    from finance.price_data import get_price_history
    from finance.price_model import InsufficientDataError, gpr_forecast_path

    trace = list(pre_trace or [])
    cache = session_state.setdefault("_forecast_cache", {})
    cache_key = f"{ticker.upper()}:{days}"

    if cache_key in cache:
        data = cache[cache_key]
        trace.append(f"[chat] reusing cached {days}-day forecast for '{ticker}'")
    else:
        close = get_price_history(ticker, start="2023-01-01")["Close"]
        try:
            result = gpr_forecast_path(close, days=days)
        except InsufficientDataError as exc:
            # A real failure mode found in testing: a thinly-traded ticker
            # (e.g. a BSE listing Yahoo Finance barely covers) can return
            # only a handful of data points, too few to fit anything. Surface
            # it as an honest, conversational answer instead of a traceback.
            trace.append(f"[chat] insufficient data to forecast '{ticker}': {exc}")
            return {"answer": f"I can't forecast {ticker} -- {exc} Try a more actively-traded ticker "
                               "(e.g. the NSE listing, .NS, if this was a .BO one).",
                    "domain": "finance", "trace": trace, "retrieved_context": []}
        data = {"close": close, **result}
        cache[cache_key] = data
        trace.append(f"[chat] computed {days}-trading-day Gaussian Process forecast for '{ticker}': "
                      f"{data['last_known_close']:.2f} -> {data['final_predicted_close']:.2f}")

    session_state["last_domain"] = "finance"
    session_state["last_ticker"] = ticker

    chart = future_forecast_chart(data["close"], data["forecast"], ticker)

    lo = data["forecast"]["lower_80"].iloc[-1]
    hi = data["forecast"]["upper_80"].iloc[-1]
    prompt = f"""Ticker: {ticker}
Last known close ({data['last_known_date'].date()}): {data['last_known_close']:.2f}
Model: Gaussian Process fitted to daily RETURNS, price path integrated from them.
{days}-trading-day forecast, central estimate on {data['final_predicted_date'].date()}: {data['final_predicted_close']:.2f}
80% range on that final day: [{lo:.2f}, {hi:.2f}]

MEASURED TRACK RECORD (66 month-forecasts, 6 tickers, walk-forward):
- the 80% range contained the real price 75.6% of the time -- so the RANGE is roughly trustworthy
- the central estimate beat a naive "assume no change" baseline on only 50% of months -- a coin flip

The user asked: "{focus}". Answer in 2-4 sentences. Lead with the RANGE, not the central number, because
the range is the part that measures well. Say plainly that the central estimate is no better than assuming
no change, and that the forecast is near-flat because daily price direction is not predictable from past
prices. Not financial advice."""

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model
    answer = call_llm(prompt, system=FINANCE_EXPLAINER_SYSTEM_PROMPT, **kwargs)

    trace.append(f"[chat] generated forecast report ({len(answer)} chars)")
    return {"answer": answer, "domain": "finance", "trace": trace, "retrieved_context": [], "chart": chart}


def _answer_solar_query(focus: str, session_state: dict, backend: str | None = None,
                        model: str | None = None, pre_trace: list | None = None) -> dict:
    """Solar / space-weather questions.

    Simpler than the finance and astronomy paths for a structural reason:
    there is no entity to resolve, so no ticker search, no disambiguation and
    no per-key cache -- there is only one Sun. The cache is therefore keyed on
    the domain alone, and the underlying solar/forecast.current_forecast()
    does its own network caching.

    The answer is generated from the grounded forecast block (which already
    carries the NOAA baseline, the measured skill, and the caveats) under the
    strict solar prompt, so the model is never in a position to invent a
    probability or, worse, a flare time.
    """
    trace = list(pre_trace or [])

    if backend or model:
        import os
        if backend:
            os.environ["BTP_LLM_BACKEND"] = backend
        if model:
            os.environ["BTP_OLLAMA_MODEL"] = model

    from agents.nodes.solar_flare_node import solar_flare_node, solar_forecast_block

    state = solar_flare_node({"domain": "solar", "question_focus": focus, "trace": trace})
    trace = state["trace"]
    result = state.get("solar_result", {})

    if result.get("error"):
        return {"answer": f"I couldn't reach the space-weather data just now ({result['error']}). "
                           f"The NOAA endpoints are free and unauthenticated, so this is usually "
                           f"transient -- try again in a moment.",
                "domain": "solar", "trace": trace, "retrieved_context": [], "chart": None}

    # The chatbot quotes the GP classifier, not Hawkes. It is the better
    # forecaster of the two -- Brier 0.148 against 0.158, TSS 0.239 against
    # 0.163 over 57 days scored on NOAA's own outcomes -- and it beats NOAA's
    # operational TSS of 0.231. The Hawkes model stays on the dashboard
    # because its parameters are the interpretable result (branching ratio
    # 0.84, 22-hour excitation half-life), but a chat answer should lead with
    # whichever model actually predicts best.
    gp = (result.get("gp") or {})
    if gp.get("probability") is not None:
        result["forecast"]["probability"] = gp["probability"]
        result["method"] = "gaussian-process-classifier"
        skill = gp.get("holdout_skill") or {}
        result["forecast"]["basis"] = [
            f"GP classifier over 39 daily features (flare counts by class and window, "
            f"time since last event, fluence, Carrington-rotation lag)",
            f"trained on {gp.get('trained_on', '1996-2026')}, features as of {gp.get('as_of')}",
        ] + [b for b in result["forecast"].get("basis", []) if "GP classifier" not in b]
        if skill.get("brier") is not None:
            result["skill"] = {"brier": skill["brier"], "tss": skill.get("tss"),
                               "note": "held-out tail of the training period; scored against "
                                       "NOAA's outcomes over 57 archived days it reached "
                                       "Brier 0.148 / TSS 0.239 vs NOAA's 0.131 / 0.231"}
        trace.append(f"[solar] answering with the GP classifier ({gp['probability']:.1%}), "
                     f"the better-scoring of the two models")
    elif gp.get("error"):
        trace.append(f"[solar] GP unavailable ({gp['error']}) -- falling back to Hawkes")

    block = solar_forecast_block(state)

    from agents.llm_backend import call_llm
    from agents.prompts import SOLAR_FORECAST_SYSTEM_PROMPT

    prompt = (f"{block}\n\n"
              f"USER QUESTION: {focus}\n\n"
              f"Answer using only the forecast data above. Always pair the probability with its "
              f"horizon, mention NOAA's own figure alongside it, and never state a time at which a "
              f"flare will occur.")
    try:
        answer = call_llm(prompt, system=SOLAR_FORECAST_SYSTEM_PROMPT, backend=backend)
    except Exception as e:
        # A dead LLM should still leave the user with the real numbers rather
        # than nothing -- the computed block is the substance, the prose is a
        # convenience.
        trace.append(f"[solar] LLM unavailable ({type(e).__name__}) -- returning the computed block")
        fc = result.get("forecast", {})
        p = fc.get("probability")
        answer = (f"**P({fc.get('label')}) = {p:.1%}** over the {fc.get('horizon')}.\n\n"
                  f"```\n{block}\n```" if p is not None else f"```\n{block}\n```")

    session_state["last_domain"] = "solar"
    cache = session_state.setdefault("_pipeline_cache", {})
    cache["solar:sun"] = state

    chart = _build_solar_chart(result)
    return {"answer": answer, "domain": "solar", "trace": trace,
            "retrieved_context": [block], "chart": chart}


_HORIZON_PATTERN = re.compile(
    r"\b(?:next\s+)?(\d+)\s*(hour|hr|day|week)s?\b"
    r"|\b(today|tomorrow|this\s+week|next\s+week|this\s+month)\b", re.IGNORECASE)

_NAMED_HORIZONS = {"today": 24, "tomorrow": 24, "this week": 168,
                   "next week": 168, "this month": 720}


def _extract_horizon_hours(message: str, default: int = 168) -> int:
    """How far ahead the user is asking about, in hours.

    Defaults to a week rather than a day: "how many flares should I expect"
    with no stated window reads as a general-activity question, and a 24-hour
    answer to it would usually be a fraction below 1, which tells the user
    nothing.
    """
    m = _HORIZON_PATTERN.search(message)
    if not m:
        return default
    if m.group(1):
        n, unit = int(m.group(1)), m.group(2).lower()
        return n * {"hour": 1, "hr": 1, "day": 24, "week": 168}[unit]
    return _NAMED_HORIZONS.get(re.sub(r"\s+", " ", m.group(3).lower().strip()), default)


def _answer_solar_count_query(message: str, session_state: dict, backend: str | None = None,
                              model: str | None = None, pre_trace: list | None = None) -> dict:
    """"How many flares should I expect?" -- the expected-count model.

    A different question from "will there be one", and a different
    computation: probability_within() is exact and closed-form because
    conditioning on no events rules out self-excitation, while the expected
    COUNT needs simulation, since each flare excites further flares and that
    branching cascade has no closed form. This path existed in
    solar/flare_model.py from the start but nothing could reach it -- the
    chatbot's pipeline was fixed, so there was no way to ask for it.
    """
    trace = list(pre_trace or [])
    horizon = _extract_horizon_hours(message)

    try:
        import numpy as np
        import pandas as pd

        from solar.flare_data import fetch_recent_flares
        from solar.flare_model import catalog_to_hawkes_input
        from solar.forecast import load_model

        hawkes, meta = load_model()
        if hawkes is None:
            return {"answer": "The flare model hasn't been fitted yet -- run "
                              "`python -m solar.forecast --train` first.",
                    "domain": "solar", "trace": trace, "retrieved_context": [], "chart": None}

        flares = fetch_recent_flares()
        cat = flares.rename(columns={"max_class": "flare_class"})
        times, marks, _, t0 = catalog_to_hawkes_input(cat)

        if t0 is None or len(times) == 0:
            # A quiet Sun: no M+ flares in the 7-day window, so
            # catalog_to_hawkes_input returns t0=None. Computing now_hours
            # from that yields NaN, and NaN silently defeats the simulation's
            # `while t < end` guard -- the loop never runs and the answer
            # comes back as a confident 0.00, alongside a 26% probability
            # from the closed form, which cannot both be true. With no
            # history there is nothing to date the clock against and nothing
            # exciting anything, so the origin is arbitrary: start at 0 and
            # let the background rate do the work.
            times, marks = np.array([]), np.array([])
            now_hours = 0.0
            trace.append("[solar_count] no M+ flares in the last 7 days -- "
                         "expectation comes from the background rate alone")
        else:
            now_hours = (pd.Timestamp.utcnow().tz_localize(None)
                         - pd.Timestamp(t0)).total_seconds() / 3600.0

        expected = hawkes.expected_count(times, marks, now_hours, horizon_hours=horizon,
                                          n_sims=400, rng=np.random.default_rng(0))
        prob = hawkes.probability_within(times, marks, now_hours, horizon)
        trace.append(f"[solar_count] expected {expected:.2f} M+ flares in {horizon}h "
                     f"(simulated); P(at least one) = {prob:.1%}")
    except Exception as e:
        trace.append(f"[solar_count] FAILED {type(e).__name__}: {e}")
        return {"answer": f"I couldn't compute that just now ({type(e).__name__}).",
                "domain": "solar", "trace": trace, "retrieved_context": [], "chart": None}

    window = f"{horizon} hours" if horizon < 48 else f"{horizon // 24} days"
    block = (f"EXPECTED COUNT (simulated, marked Hawkes process)\n"
             f"  horizon            : next {window}\n"
             f"  expected M+ flares : {expected:.2f}\n"
             f"  P(at least one)    : {prob:.1%}\n"
             f"  recent history     : {len(times)} M+ flare(s) in the last 7 days\n"
             f"  model              : branching ratio {hawkes.alpha:.2f}, "
             f"half-life {np.log(2) / hawkes.beta:.1f}h\n"
             f"CAVEAT: an expected count is an average over simulations, not a promise of "
             f"that many events. Individual flare timing is not predictable.")

    answer = (f"**About {expected:.1f} M-class-or-larger flares** are expected over the next "
              f"{window}, with a **{prob:.0%} chance of at least one**. "
              f"That's based on {len(times)} M+ flare(s) in the past week and the fitted "
              f"self-excitation (half-life {np.log(2) / hawkes.beta:.0f}h). "
              f"It's an average over simulations, not a prediction of when any flare will occur.")

    session_state["last_domain"] = "solar"
    return {"answer": answer, "domain": "solar", "trace": trace,
            "retrieved_context": [block], "chart": None}


_KEPID_PATTERN = re.compile(r"\b(?:KIC|KPLR)\s*0*(\d{5,9})\b", re.IGNORECASE)


def _answer_koi_lookup(message: str, target: str | None, session_state: dict,
                       pre_trace: list | None = None) -> dict:
    """"Does this star have planets?" -- answered from NASA's catalogue.

    A recorded fact beats a detector's inference, and the project previously
    had no way to say so: asking whether a star had planets ran the anomaly
    pipeline and got back flagged windows, which is a different question and
    a weaker answer. No LLM call -- the archive either knows or it does not.
    """
    trace = list(pre_trace or [])
    m = _KEPID_PATTERN.search(message) or (_KEPID_PATTERN.search(target or "") if target else None)
    if not m:
        return {"answer": "Which star? Give me a KIC id, e.g. KIC 10874614.",
                "domain": "astronomy", "trace": trace, "retrieved_context": [], "chart": None}

    kepid = int(m.group(1))
    try:
        from astronomy.koi_catalog import target_summary, usable_kois

        info = target_summary(kepid, usable_kois())
    except Exception as e:
        trace.append(f"[koi_lookup] FAILED {type(e).__name__}: {e}")
        return {"answer": f"I couldn't reach the NASA catalogue just now ({type(e).__name__}).",
                "domain": "astronomy", "trace": trace, "retrieved_context": [], "chart": None}

    if not info.get("known"):
        trace.append(f"[koi_lookup] KIC {kepid}: not in the KOI catalogue")
        answer = (f"**KIC {kepid} is not in NASA's Kepler Object of Interest catalogue** with a "
                  f"decided disposition. That means no transit candidate has been catalogued and "
                  f"vetted for it -- not that it has no planets. I can still run anomaly detection "
                  f"on its light curve if you'd like.")
        return {"answer": answer, "domain": "astronomy", "trace": trace,
                "retrieved_context": [], "chart": None}

    names = [n for n in info.get("planet_names", []) if n]
    trace.append(f"[koi_lookup] KIC {kepid}: {info['n_kois']} KOI(s), "
                 f"{info['n_confirmed']} confirmed, label={info['label']}")

    if info["n_confirmed"]:
        named = ", ".join(names) if names else "unnamed in the catalogue"
        periods = ", ".join(f"{p:g} d" for p in info["periods_days"][:6])
        answer = (f"**KIC {kepid} has {info['n_confirmed']} confirmed planet"
                  f"{'s' if info['n_confirmed'] != 1 else ''}** "
                  f"out of {info['n_kois']} catalogued object(s) of interest.\n\n"
                  f"- Confirmed: {named}\n- Orbital periods: {periods}\n\n"
                  f"That's from NASA's Exoplanet Archive -- a published, vetted result, not "
                  f"something this system inferred.")
    else:
        answer = (f"**KIC {kepid} has no confirmed planets.** It has {info['n_kois']} catalogued "
                  f"object(s) of interest, all dispositioned **FALSE POSITIVE** -- transit-like "
                  f"signals that vetting attributed to something other than a planet, such as an "
                  f"eclipsing binary or instrumental artefact.")

    session_state["last_domain"] = "astronomy"
    session_state["last_target"] = f"KIC {kepid}"
    return {"answer": answer, "domain": "astronomy", "trace": trace,
            "retrieved_context": [str(info)], "chart": None}


def _answer_logged_forecast(ticker: str, session_state: dict, pre_trace: list | None = None) -> dict:
    """Next-day price from the daily forecast log.

    A lookup, not a fit. A Gaussian Process costs ~10-45s to fit and cannot
    sit inside a chat turn; the stored row is free. The log is written once a
    day BEFORE the outcome is known (finance/forecast_log.py), so the track
    record quoted here was never tuned against.

    The answer leads with the interval rather than the point estimate, because
    that is the part that measured well: 82.5% interval coverage against an
    80% target over 120 walk-forward days, while point accuracy loses to a
    naive "tomorrow equals today" rule (RMSE ratio 1.22) and directional
    accuracy is 0.44 -- below chance. Presenting the point forecast as the
    headline would be advertising the one thing that does not work.
    """
    trace = list(pre_trace or [])
    try:
        from finance.forecast_log import evaluate, latest

        row = latest(ticker)
        skill = evaluate(min_rows=5)
    except Exception as e:
        trace.append(f"[forecast_log] unavailable ({type(e).__name__}: {e})")
        return None

    if not row or row.get("gpr_pred") is None:
        trace.append(f"[forecast_log] no stored forecast for {ticker} -- run "
                     f"`python -m finance.forecast_log --log`")
        return None

    close = float(row["close_today"])
    pred = float(row["gpr_pred"])
    lo, hi = float(row["gpr_lower80"]), float(row["gpr_upper80"])
    move = pred - close
    target = pd.Timestamp(row["target_date"]).date() if row.get("target_date") is not None else "the next session"
    trace.append(f"[forecast_log] stored forecast for {ticker}: {pred:.2f} "
                 f"[{lo:.2f}, {hi:.2f}] as of {row.get('as_of_date')}")

    lines = [
        f"**{ticker} — next trading session ({target})**",
        "",
        f"- Last close: **{close:,.2f}**",
        f"- Model's central estimate: **{pred:,.2f}** ({move:+,.2f})",
        f"- 80% range: **{lo:,.2f} to {hi:,.2f}**",
        f"- Naive baseline (assume no change): **{close:,.2f}**",
        "",
    ]

    gpr = (skill or {}).get("gpr") or {}
    naive = (skill or {}).get("naive") or {}
    if gpr.get("interval_coverage") is not None:
        lines += [
            f"**How this model has actually performed** "
            f"({gpr.get('n', 0)} scored days):",
            "",
            f"- The 80% range contained the real close **{gpr['interval_coverage']:.1%}** of the "
            f"time — close to the 80% it promises, so the *range* is trustworthy.",
        ]
        if naive.get("rmse") and gpr.get("rmse"):
            lines.append(
                f"- The *central estimate* is **not** trustworthy: its average error "
                f"({gpr['rmse']:.1f}) is worse than simply assuming tomorrow equals today "
                f"({naive['rmse']:.1f}).")
        if isinstance(gpr.get("directional_accuracy"), (int, float)):
            lines.append(
                f"- It gets the direction right {gpr['directional_accuracy']:.0%} of the time — "
                f"no better than a coin flip.")
        lines.append("")

    lines += [
        "**Read this as a range, not a prediction.** Daily price direction is not predictable "
        "from past prices, and this model does not beat assuming no change. What it does do "
        "reliably is tell you how wide the plausible range is. Not investment advice — for "
        "decisions, consult a SEBI-registered adviser.",
    ]

    session_state["last_domain"] = "finance"
    session_state["last_ticker"] = ticker
    return {"answer": "\n".join(lines), "domain": "finance", "trace": trace,
            "retrieved_context": [str({k: v for k, v in row.items()
                                       if not k.startswith("_")})], "chart": None}


def _build_solar_chart(result: dict):
    """Recent flare peak-flux scatter on a log axis, with the M and X
    thresholds drawn in. Mirrors the anomaly charts: the chatbot shows the
    same visual the dashboard would, so a chat answer and the page agree."""
    flares = result.get("recent_flares")
    if flares is None or getattr(flares, "empty", True):
        return None
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        colors = {"A": "#94a3b8", "B": "#60a5fa", "C": "#fbbf24", "M": "#f97316", "X": "#dc2626"}
        for letter, color in colors.items():
            sub = flares[flares["class_letter"] == letter]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub["begin_time"], y=sub["peak_flux"], mode="markers",
                                      name=f"{letter}-class", marker=dict(size=8, color=color)))
        fig.add_hline(y=1e-5, line_dash="dash", line_color="#f97316",
                      annotation_text="M threshold")
        fig.update_yaxes(type="log", title="peak flux (W/m²)")
        fig.update_layout(height=300, margin=dict(t=30, b=20),
                          title="GOES flares, last 7 days")
        return fig
    except Exception:
        return None


def _window_time_range(result: dict, window_id: int):
    """Window index -> actual observation times.

    A window id is an artefact of how the series was sliced; it tells a
    reader nothing. "Day 1023.4, lasting 11.8 hours" is the same fact in
    terms of the observation itself, and it is what makes an anomaly
    checkable against a published ephemeris.
    """
    times = result.get("_time")
    stride = result.get("stride_points")
    window = result.get("window_points")
    if times is None or stride is None or window is None:
        return None, None, None
    start_idx = min(int(window_id) * stride, len(times) - 1)
    end_idx = min(start_idx + window, len(times) - 1)
    hours = float(times[end_idx] - times[start_idx]) * 24.0
    return float(times[start_idx]), float(times[end_idx]), hours


def _dedupe_overlapping(windows: list, result: dict) -> list:
    """Collapse flags that are the same event seen through adjacent windows.

    Windows overlap by design (stride 6, width 24 means four cover any given
    point), so one real event produces a run of near-identical flags. Listing
    "day 1134.90" and "day 1135.02" as the two most significant findings
    presents one event as two and wastes most of a top-5.
    """
    stride = result.get("stride_points")
    window = result.get("window_points")
    if not stride or not window:
        return windows
    span = max(1, window // stride)

    kept = []
    for w in windows:
        wid = w.get("window_id")
        if wid is None:
            kept.append(w)
            continue
        if any(abs(int(wid) - int(k.get("window_id", -10 ** 9))) < span for k in kept):
            continue
        kept.append(w)
    return kept


def _plain_shape(row: dict) -> str:
    """Describe what the brightness actually DID, in words.

    "skew -4.29" is a real measurement and meaningless to a reader. The sign
    carries the information: a long left tail means the brightness fell
    sharply and came back, a long right tail means it spiked. Saying that is
    the same fact in language someone can picture.
    """
    skew = float(row.get("skew", 0.0) or 0.0)
    depth = float(row.get("depth", 0.0) or 0.0)

    if depth > 0.008:
        pct = depth * 100
        if abs(skew) < 2:
            return (f"the star dimmed by about {pct:.1f}% and brightened back up again, "
                    f"in a smooth symmetric dip")
        return (f"the star dimmed by about {pct:.1f}%, but unevenly — it faded and "
                f"recovered at different rates")
    if skew < -2:
        return "the brightness dropped sharply and then recovered"
    if skew > 2:
        return "the brightness spiked upward briefly, then settled"
    return "the brightness wandered away from its usual level without a clear dip or spike"


def _plain_cause(cause: str) -> str:
    """The rule-based cause label, said in ordinary words."""
    mapping = {
        "Possible transit-like dip":
            "looks like something passing in front of the star",
        "Possible transit-like dip (asymmetric)":
            "looks like something passing in front of the star, though the shape is lopsided",
        "Possible flare or asymmetric irregularity":
            "looks more like a flare or a sudden flicker than a planet",
        "Possible stellar variability or noise":
            "looks like the star's own restlessness, or measurement noise",
        "Unclear -- statistical outlier only":
            "unusual, but with no single clear reason",
    }
    return mapping.get(cause, cause.lower())


def _plain_fraction(x) -> str:
    """0.567 -> 'a bit under 6 in 10'. Percentages invite false precision."""
    if x is None:
        return "an unknown share"
    n = round(float(x) * 10)
    near = abs(float(x) * 10 - n) < 0.15
    qualifier = "" if near else ("a bit over " if float(x) * 10 > n else "a bit under ")
    return f"{qualifier}{n} in 10"


def _astronomy_plain_summary(result: dict) -> str:
    """The opening paragraph: what this star is and what was found.

    This exists because the technical report, while accurate, was unreadable
    -- "3,317 of 8,619 windows flagged, skew -4.29, reconstruction error
    0.00056" is a correct description of nothing a person can picture. The
    numbers stay, further down; this says what they mean.
    """
    target = result.get("target", "this star")
    planets = result.get("known_planets")
    rec = result.get("transit_recovery") or {}
    baseline = result.get("baseline_days")

    bits = []
    if planets and planets.get("n_confirmed"):
        name = (planets.get("planet_names") or ["an unnamed planet"])[0]
        period = planets["periods_days"][0] if planets.get("periods_days") else None
        when = f"every {period:.2f} days" if period else "on a regular cycle"
        bits.append(f"**{target} has a confirmed planet, {name}**, which passes in front of "
                    f"the star {when}, dimming it slightly each time.")
    elif planets:
        bits.append(f"**{target} has no confirmed planets.** Astronomers catalogued "
                    f"{planets['n_kois']} transit-like signal(s) here and concluded each was "
                    f"something else — often two stars orbiting each other, or an instrument "
                    f"artefact.")
    else:
        bits.append(f"**{target} has never been catalogued** as having a planet candidate, so "
                    f"there is nothing published to compare these findings against.")

    if baseline:
        bits.append(f"I looked at about {baseline / 365:.1f} years of NASA brightness "
                    f"measurements, in 12-hour stretches.")

    if rec.get("recall") is not None:
        caught = _plain_fraction(rec["recall"])
        right = _plain_fraction(rec["precision"])
        bits.append(f"Of the times the planet was actually crossing, the detector spotted "
                    f"**{caught}**. And of everything it flagged as odd, **{right}** were "
                    f"genuine crossings — the rest were the star doing something else.")

    return " ".join(bits)


ASTRONOMY_CONVERSATIONAL_PROMPT = """You explain light-curve analysis results clearly and directly.

STRUCTURE your answer like this:
- Open by NAMING the planet if FINDINGS gives one, with its orbital period and how much it dims
  the star. "KIC 5358624 hosts Kepler-428 b, a gas giant that orbits every 3.53 days and dims the
  star by 2.23% each pass." Lead with the conclusion, not the process.
- ALWAYS state how well the detection did when FINDINGS gives a hit rate -- how many of the
  planet's crossings were caught, and how many flagged events were genuine. That is the actual
  result; an answer that lists the odd moments but never says whether the planet was found has
  omitted the point.
- Then a short bulleted list of the notable moments: what happened, when, and why it was flagged.
- Close with one or two sentences on what it means and what it does not prove -- a detected dip
  is a candidate, and confirming a planet needs the dip to repeat at a steady interval with a
  consistent depth and duration.

Use **bold** for key numbers and names. Keep it under 250 words.

REGISTER -- this matters:
- Do NOT narrate your own process. Never write "I analysed", "I looked at", "we found",
  "the goal was", "what I did". State the findings directly.
- Do NOT open with "So," or "Well," or "Sure enough". Do not use filler like "pretty cool",
  "fascinating", "interesting to note", "it's worth mentioning".
- Do NOT end by offering further help.
- Write as an informed reference would: direct, specific, unhurried. Not chatty, not a sales pitch.

STRICT FACTUAL RULES:
1. Every number must appear in FINDINGS. Never invent, never re-round.
1b. Do NOT convert a fraction into a percentage. "a bit over 7 in 10" must stay as written -- it is
   deliberately imprecise because the underlying figure does not support three significant digits.
   Writing "71%" manufactures precision, and "a false positive rate of 29%" additionally misuses a
   term that means something specific and different. Report hit rates in the words given.
2. Never mention windows, window numbers, anomaly scores, skew, kurtosis, reconstruction error,
   IsolationForest, or autoencoders. Talk about the star, its brightness, days, and the planet.
3. Do NOT describe the planet's size, mass, temperature, habitability, composition or significance
   unless FINDINGS states it. An earlier draft called a gas giant "a pretty small planet" because
   the size was not supplied. If FINDINGS does not say it, you do not know it.
4. A detected dip is a candidate, not proof of a planet.
5. If FINDINGS says the strongest signals were not the planet, explain why plainly: the detector
   ranks by how abrupt a change is, and a flare is far more abrupt than a planet's gradual dip."""


def _astronomy_facts_for_llm(result: dict) -> str:
    """The grounded fact sheet the model writes from.

    Deliberately NOT the raw result dict. Handed the raw fields, the explainer
    produced "the most severe are window 7017, 7914, 4578" -- accurate,
    unreadable, and repeating internals no reader can use. Given the same
    information already translated into observations about the star, it has
    something worth saying.
    """
    f = []
    target = result.get("target", "the star")
    planets = result.get("known_planets")
    rec = result.get("transit_recovery") or {}

    f.append(f"STAR: {target}")
    if planets and planets.get("n_confirmed"):
        name = (planets.get("planet_names") or ["an unnamed confirmed planet"])[0]
        period = planets["periods_days"][0] if planets.get("periods_days") else None
        radius = (planets.get("radii_earth") or [None])[0]
        depth = (planets.get("depths_ppm") or [None])[0]
        size = ""
        if radius:
            # Anchor to Earth and Jupiter so "11 Earth radii" means something.
            comparison = ("about Earth's size" if radius < 1.5 else
                          "a few times Earth's size" if radius < 4 else
                          "roughly Neptune-sized" if radius < 8 else
                          "a gas giant, comparable to Jupiter")
            size = f", radius {radius} times Earth's -- {comparison}"
        dim = f", dimming the star by {depth / 10000:.2f}% when it passes" if depth else ""
        f.append(f"KNOWN PLANET: {name}, confirmed by NASA"
                 + (f", orbits every {period:.2f} days" if period else "") + size + dim)
    elif planets:
        f.append(f"KNOWN PLANETS: none. {planets['n_kois']} transit-like signal(s) were "
                 f"catalogued here and all were judged false positives.")
    else:
        f.append("KNOWN PLANETS: this star has never been catalogued as a planet candidate.")

    if result.get("baseline_days"):
        f.append(f"OBSERVED: {result['baseline_days'] / 365:.1f} years of NASA brightness "
                 f"measurements, examined in 12-hour stretches")

    if rec.get("recall") is not None:
        f.append(f"HOW WELL IT DID: caught {_plain_fraction(rec['recall'])} of the times the "
                 f"planet was actually crossing; of everything flagged as odd, "
                 f"{_plain_fraction(rec['precision'])} were genuine crossings")

    tops = _dedupe_overlapping(result.get("top_anomaly_windows") or [], result)
    known = result.get("_known_labels")
    if tops:
        f.append("MOST STRIKING MOMENTS:")
        for w in tops[:5]:
            wid = w.get("window_id")
            t0, _t1, _h = _window_time_range(result, wid)
            matched = (known is not None and wid is not None
                       and int(wid) < len(known) and bool(known[int(wid)]))
            day = f"day {t0:.0f}" if t0 is not None else "an unrecorded time"
            f.append(f"  - {day}: {_plain_shape(w)}. "
                     + ("NASA's records confirm the planet was crossing at that moment."
                        if matched else
                        "No planet was crossing then; it "
                        f"{_plain_cause(w.get('possible_cause', ''))}."))

        if known is not None:
            hits = sum(1 for w in tops[:5] if w.get("window_id") is not None
                       and int(w["window_id"]) < len(known) and known[int(w["window_id"])])
            if hits == 0 and rec.get("recall", 0) > 0.3:
                f.append("NOTE: none of the strongest signals are the planet, even though most "
                         "of its crossings were found. The detector ranks by how extreme a "
                         "change is, and a sudden flare is far more extreme than a planet's "
                         "gentle dip.")
    return "\n".join(f)


def _astronomy_technical_block(result: dict) -> str:
    """Raw numbers, for the trace panel rather than the answer body."""
    rec = result.get("transit_recovery") or {}
    n_flagged = result.get("n_flagged_anomalies", 0)
    n_windows = result.get("n_windows", 1)
    lines = [
        f"data: {result.get('data_source')}",
        f"{n_windows:,} windows of {result.get('window_points')} points (~12 h) at "
        f"{result.get('cadence_minutes')} min cadence, {result.get('baseline_days')} day baseline",
        f"flagged: {n_flagged:,} ({n_flagged / max(n_windows, 1):.1%})",
        f"detectors: IsolationForest on 8 window statistics"
        + (f"; autoencoder trained on {result['autoencoder_trained_on']}"
           if result.get("autoencoder_trained_on") else ""),
    ]
    if rec.get("n_windows_with_known_transit"):
        lines.append(f"vs published ephemeris: {rec['n_flagged_on_known_transit']:,} of "
                     f"{rec['n_windows_with_known_transit']:,} transit windows recovered "
                     f"(recall {rec['recall']}, precision {rec['precision']})")
    for w in _dedupe_overlapping(result.get("top_anomaly_windows") or [], result)[:5]:
        t0, _t1, _h = _window_time_range(result, w.get("window_id"))
        lines.append(f"  day {t0:.2f}: score {w.get('anomaly_score', 0):.3f}, "
                     f"{w.get('severity_label')}, depth {float(w.get('depth', 0) or 0):.4f}, "
                     f"skew {float(w.get('skew', 0) or 0):.2f}")
    return "\n".join(lines)


def _astronomy_overview(result: dict) -> str:
    """The answer a person reads.

    Ordered so the first thing seen is what it MEANS, not what was measured.
    The measurements follow, then the raw numbers last, because a reader who
    wants "skew -4.29" can scroll and a reader who does not should never have
    to parse it to learn what happened.
    """
    target = result.get("target", "target")
    lines = [f"## {target}", "", _astronomy_plain_summary(result), ""]

    # --- the individual events, in words --------------------------------
    tops = _dedupe_overlapping(result.get("top_anomaly_windows") or [], result)
    known = result.get("_known_labels")
    if tops:
        lines += ["### What stood out", ""]
        for w in tops[:5]:
            wid = w.get("window_id")
            t0, t1, hours = _window_time_range(result, wid)
            when = f"**Day {t0:.0f}**" if t0 is not None else f"**Window {wid}**"

            matches = (known is not None and wid is not None
                       and int(wid) < len(known) and bool(known[int(wid)]))

            # When the published ephemeris says a planet was crossing at this
            # moment, that settles it. The rule-based cause is a guess from
            # window statistics, and letting it argue with the catalogue
            # produced lines that contradicted themselves -- "looks more like
            # a flare than a planet ... this is the planet crossing".
            if matches:
                planets = result.get("known_planets") or {}
                name = (planets.get("planet_names") or ["the known planet"])[0]
                verdict = f"✅ **{name} was crossing at this moment**, according to NASA's records"
            else:
                verdict = (f"⚠️ no planet was crossing then — it "
                           f"{_plain_cause(w.get('possible_cause', ''))}")

            lines.append(f"- {when} — {_plain_shape(w)}. {verdict}.")
        lines.append("")

        if known is not None:
            top_hits = sum(1 for w in tops[:5]
                           if w.get("window_id") is not None
                           and int(w["window_id"]) < len(known)
                           and known[int(w["window_id"])])
            rec = (result.get("transit_recovery") or {}).get("recall")
            if top_hits == 0 and rec and rec > 0.3:
                lines += [
                    "> The strongest signals here are *not* the planet. That is expected: the "
                    "detector ranks by how statistically extreme something is, and a sudden "
                    "flare is far more extreme than the gentle dip a planet makes. The planet "
                    "is still found — just not at the top of this list.",
                    "",
                ]

    # --- what to take from it -------------------------------------------
    lines += [
        "### What this does and doesn't tell you",
        "",
        "A flagged dip is a *candidate*, never proof of a planet. Confirming one needs the same "
        "dip to repeat at a steady interval with a consistent depth and duration. A single dip "
        "looks the same whether it is a planet, two stars eclipsing each other, a dark patch "
        "rotating across the star's face, or a glitch in the instrument.",
        "",
    ]

    # --- the numbers, last ----------------------------------------------
    n_flagged = result.get("n_flagged_anomalies", 0)
    n_windows = result.get("n_windows", 1)
    rec = result.get("transit_recovery") or {}
    lines += ["<details>", "<summary><b>Technical detail</b></summary>", ""]

    planets = result.get("known_planets")
    if planets:
        lines.append(f"- NASA catalogue: {planets['n_kois']} object(s) of interest, "
                     f"{planets['n_confirmed']} confirmed"
                     + (f" ({', '.join(planets['planet_names'])})" if planets.get("planet_names") else "")
                     + (f"; period(s) {', '.join(f'{p:g} d' for p in planets['periods_days'])}"
                        if planets.get("periods_days") else ""))
    lines += [
        f"- Data: {result.get('data_source')}",
        f"- {n_windows:,} windows of {result.get('window_points')} points (~12 h) "
        f"at {result.get('cadence_minutes')} min cadence, {result.get('baseline_days')} day baseline",
        f"- Flagged: {n_flagged:,} ({n_flagged / max(n_windows, 1):.1%})",
        f"- Detectors: IsolationForest on 8 window statistics"
        + (f"; autoencoder trained on {result['autoencoder_trained_on']}"
           if result.get("autoencoder_trained_on") else ""),
    ]
    if rec.get("n_windows_with_known_transit"):
        lines.append(f"- Against published ephemeris: "
                     f"{rec['n_flagged_on_known_transit']:,} of {rec['n_windows_with_known_transit']:,} "
                     f"transit windows recovered — recall {rec['recall']}, precision {rec['precision']}")
    if tops:
        lines.append("- Top regions (raw):")
        for w in tops[:5]:
            t0, _t1, _h = _window_time_range(result, w.get("window_id"))
            lines.append(f"    - day {t0:.2f}: score {w.get('anomaly_score', 0):.3f}, "
                         f"severity {w.get('severity_label')}, "
                         f"depth {float(w.get('depth', 0) or 0):.4f}, "
                         f"skew {float(w.get('skew', 0) or 0):.2f}")
    lines += ["", "</details>"]
    return "\n".join(lines)


def _astronomy_charts(result: dict) -> list:
    """Raw curve, flagged regions, and a zoom on the strongest flag."""
    charts = []
    flux, times = result.get("_flux"), result.get("_time")
    mask = result.get("_anomaly_mask")
    if flux is None or times is None:
        return charts
    try:
        from dashboard.charts import (lightcurve_anomaly_chart, lightcurve_zoom_chart,
                                      raw_lightcurve_chart)

        charts.append(raw_lightcurve_chart(times, flux))
        if mask is not None:
            charts.append(lightcurve_anomaly_chart(
                times, flux, window_size=result.get("window_points", 200),
                stride=result.get("stride_points", 50), anomaly_mask=mask))

        tops = result.get("top_anomaly_windows") or []
        if tops:
            wid = int(tops[0].get("window_id", 0))
            centre = min(wid * result.get("stride_points", 50), len(times) - 1)
            t0, _t1, _h = _window_time_range(result, wid)
            charts.append(lightcurve_zoom_chart(
                times, flux, centre, result.get("window_points", 200),
                title=f"Strongest flagged region — day {t0:.2f}" if t0 is not None
                else "Strongest flagged region"))
    except Exception:
        pass
    return charts


def _build_anomaly_chart(domain: str, state: dict):
    """Domain-appropriate anomaly chart built from data the pipeline already
    computed (finance_anomaly_node's/astronomy_node's private _-prefixed
    fields) -- no re-fetch, no re-detection. Returns None if the cached
    state predates these fields (e.g. resumed from an older cache entry)."""
    if domain == "finance":
        from dashboard.charts import finance_anomaly_chart

        anomr = state.get("finance_anomaly_result") or {}
        feats, scored = anomr.get("_features_df"), anomr.get("_scored_df")
        if feats is not None and scored is not None:
            return finance_anomaly_chart(feats, scored, anomr.get("ticker", ""))
    elif domain == "astronomy":
        from astronomy.anomaly_detector import STRIDE, WINDOW_SIZE
        from dashboard.charts import lightcurve_anomaly_chart

        ar = state.get("astronomy_result") or {}
        flux, time, mask = ar.get("_flux"), ar.get("_time"), ar.get("_anomaly_mask")
        if flux is not None and time is not None and mask is not None:
            # Read the window geometry the node ACTUALLY used, not the module
            # defaults. Windows are sized from each target's cadence now
            # (24 points at 30-minute long cadence, ~720 at 1-minute), so
            # assuming 200/50 here converts window indices to time indices
            # with the wrong stride and runs off the end of the array --
            # "index 64900 is out of bounds for axis 0 with size 64794".
            window = ar.get("window_points", WINDOW_SIZE)
            stride = ar.get("stride_points", STRIDE)
            return lightcurve_anomaly_chart(time, flux, window_size=window,
                                            stride=stride, anomaly_mask=mask)
    return None


def _answer_finance_or_astronomy(domain: str, key: str, focus: str, session_state: dict, backend: str | None = None,
                                  model: str | None = None, pre_trace: list | None = None,
                                  forecast_days: int | None = None, want_anomaly_chart: bool = False,
                                  message_for_task: str | None = None) -> dict:
    """Shared by the normal turn-by-turn flow and the post-disambiguation
    resume path -- both need to land here once domain+key+focus are settled."""
    pre_trace = pre_trace or []

    if domain == "finance" and forecast_days:
        # A next-session question is answered from the daily log: it is an
        # instant lookup instead of a fit, and it can quote a real track
        # record because the row was written before the outcome was known.
        # Longer horizons still fit ARIMA on demand -- the log stores one
        # step ahead only.
        if forecast_days <= 1:
            logged = _answer_logged_forecast(key, session_state, pre_trace=pre_trace)
            if logged is not None:
                return logged
        return _answer_price_forecast(key, focus, forecast_days, session_state, backend=backend, model=model,
                                       pre_trace=pre_trace)

    cache = session_state.setdefault("_pipeline_cache", {})
    cache_key = f"{domain}:{key.upper()}"

    if cache_key in cache:
        from agents.nodes.explainer_node import explainer_node
        state = dict(cache[cache_key])
        state["question_focus"] = focus
        state["trace"] = list(state.get("trace", [])) + pre_trace + [f"[chat] reusing cached {domain} data for '{key}', re-explaining for new focus"]
        state = explainer_node(state)
    else:
        from agents.graph import run
        if backend:
            import os
            os.environ["BTP_LLM_BACKEND"] = backend

        # The second routing axis. The graph skips sentiment and the anomaly
        # detectors for tasks that do not need them, so an RSI question stops
        # paying for FinBERT and a second XGBoost fit. An unclassified task
        # is passed through as None, which the graph treats as "run
        # everything" -- the original behaviour, kept as the safe default.
        available = {"ticker"} if domain == "finance" else {"target"}
        selection = select_model(message_for_task or focus or "", domain=domain,
                                  available=available, strategy="filter_only")
        pre_trace = list(pre_trace) + selection["trace"]

        state = run(ticker=key if domain == "finance" else None,
                     target=key if domain == "astronomy" else None,
                     question_focus=focus, task=selection.get("task"))
        state["trace"] = pre_trace + state["trace"]

    cache[cache_key] = state
    session_state["last_domain"] = domain
    if domain == "finance":
        session_state["last_ticker"] = key
    else:
        session_state["last_target"] = key

    result = {"answer": state["report"], "domain": domain, "trace": state["trace"],
              "retrieved_context": state.get("retrieved_context", [])}

    # Astronomy answers lead with the computed report and let the model's
    # narrative follow. Asked to summarise on its own, the LLM produced
    # "windows 7513, 7514, 1155 and 1156 were flagged" -- true, unreadable,
    # and silently discarding the catalogue facts and the transit-recovery
    # check the pipeline had already computed.
    if domain == "astronomy" and state.get("astronomy_result"):
        ar = state["astronomy_result"]
        facts = _astronomy_facts_for_llm(ar)

        # The model writes the answer, from facts already translated out of
        # window indices and z-scores. Two earlier attempts failed for
        # opposite reasons: handed the raw result it recited window numbers,
        # and with no model at all the reply read like a generated form
        # rather than someone explaining. Grounded facts plus a
        # conversational brief is what gets both.
        try:
            prose = call_llm(f"FINDINGS:\n{facts}\n\nUSER ASKED: {focus or 'tell me about this star'}",
                             system=ASTRONOMY_CONVERSATIONAL_PROMPT, backend=backend).strip()
        except Exception as e:
            state["trace"].append(f"[chat] LLM unavailable ({type(e).__name__}) -- "
                                  f"falling back to the computed summary")
            prose = _astronomy_plain_summary(ar)

        result["answer"] = prose or _astronomy_plain_summary(ar)
        # Raw numbers belong in the trace panel, not the reply. They were
        # rendering as literal "<details>" tags anyway, since st.write does
        # not process HTML.
        result["retrieved_context"] = [facts, _astronomy_technical_block(ar)]
        charts = _astronomy_charts(ar)
        if charts:
            result["charts"] = charts
            result["chart"] = charts[1] if len(charts) > 1 else charts[0]
        return result

    if want_anomaly_chart:
        from agents.anomaly_shared import summarize

        chart = _build_anomaly_chart(domain, state)
        if chart is not None:
            result["chart"] = chart
        summary = summarize(domain, state)
        if summary is not None:
            result["anomaly_summary"] = summary

    return result


def _answer_mutual_fund_query(fund_query: str, focus: str, session_state: dict, backend: str | None = None,
                               model: str | None = None, pre_trace: list | None = None) -> dict:
    """Mutual funds don't go through agents/graph.py (no Sentiment/Explainer
    node makes sense for a diversified fund -- see finance/mutual_funds.py's
    docstring) so this is a small, self-contained path, with its own
    per-session cache mirroring the finance/astronomy one in answer_query."""
    from finance.mutual_funds import compute_returns, forecast_nav, get_nav_history, get_scheme_meta, search_schemes

    trace = list(pre_trace or [])
    cache = session_state.setdefault("_mf_cache", {})
    cache_key = fund_query.lower().strip()

    if cache_key in cache:
        data = cache[cache_key]
        trace.append(f"[chat] reusing cached mutual fund data for '{fund_query}'")
    else:
        matches = search_schemes(fund_query, limit=1)
        if not matches:
            trace.append(f"[chat] no scheme matched '{fund_query}'")
            return {"answer": f"I couldn't find a mutual fund scheme matching '{fund_query}'. Try a more specific name.",
                    "domain": "mutual_fund", "trace": trace, "retrieved_context": []}

        scheme_code = matches[0]["schemeCode"]
        meta = get_scheme_meta(scheme_code)
        nav_df = get_nav_history(scheme_code)
        data = {"scheme_code": scheme_code, "meta": meta, "returns": compute_returns(nav_df),
                "rmse": forecast_nav(nav_df, test_size=30)["rmse"]}
        cache[cache_key] = data
        trace.append(f"[chat] resolved '{fund_query}' -> scheme {scheme_code} ({meta.get('scheme_name')})")

    session_state["last_domain"] = "mutual_fund"
    session_state["last_mutual_fund"] = fund_query

    prompt = f"""Scheme: {data['meta'].get('scheme_name')}
Fund house: {data['meta'].get('fund_house')}
Category: {data['meta'].get('scheme_category')}
Trailing returns: {data['returns']}
ARIMA baseline RMSE (last 30 NAV points): {data['rmse']}

The user asked: "{focus}". Answer that question directly in 2-4 sentences using only the numbers above."""

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model
    answer = call_llm(prompt, system=FINANCE_EXPLAINER_SYSTEM_PROMPT, **kwargs)

    trace.append(f"[chat] generated mutual fund report ({len(answer)} chars)")
    return {"answer": answer, "domain": "mutual_fund", "trace": trace, "retrieved_context": []}


def answer_query(message: str, session_state: dict, backend: str | None = None, model: str | None = None,
                  image_bytes: bytes | None = None, image_name: str | None = None) -> dict:
    """session_state is a plain dict the caller persists across turns
    (Streamlit's st.session_state works directly). Mutated in place with
    last_domain/last_ticker/last_target and the pipeline cache.

    image_bytes: an uploaded light-curve plot (JPG/PNG) -- unambiguously an
    astronomy request, so this bypasses intent parsing/disambiguation
    entirely and goes straight to image extraction + anomaly detection.
    `message` (if the user typed anything alongside the image) becomes the
    question focus; empty defaults to a general overview."""

    if image_bytes is not None:
        return _answer_image_lightcurve_query(
            image_bytes, image_name or "uploaded", message.strip() or "give an overview of this light curve and any anomalies",
            session_state, backend=backend, model=model,
        )

    if _DATASET_QUERY_PATTERN.search(message):
        return _answer_dataset_query(message, backend=backend, model=model)

    # Risk is checked early and only when a ticker is resolvable. It comes
    # before the anomaly path deliberately: "is TCS risky, any unusual days?"
    # matches both, and the risk answer already contains the anomalies --
    # attributed to market or company, which the anomaly path alone cannot do.
    if _MARKET_RISK_PATTERN.search(message) and not _EXPLICIT_SUFFIX_IN_TEXT.search(message):
        return _answer_market_risk_query(message, session_state, backend=backend, model=model)

    if _RISK_QUERY_PATTERN.search(message):
        risk_answer = _answer_risk_query(message, session_state, backend=backend, model=model)
        if risk_answer is not None:
            return risk_answer

    # Solar is checked before intent parsing, for the same reason the dataset
    # and mode-pin patterns are: it is unambiguous when it matches, and the
    # small local model is unreliable at spotting a domain that has no entity
    # id to anchor on. _looks_solar() already declines when a KIC/TIC id is
    # present, so a stellar-flare question is not stolen from astronomy.
    if _looks_solar(message):
        pre = ["[chat] detected a solar/space-weather question -- "
               "routing to the solar domain (no entity id needed)"]
        # Second axis: WHICH solar model. "will there be a flare" and "how
        # many flares" are different questions over the same data, and until
        # the selector existed only the first was reachable.
        sel = select_model(message, domain="solar", available={"horizon_hours"},
                           strategy="filter_only")
        pre += sel["trace"]
        if sel["model_id"] == "solar.hawkes_expected_count":
            return _answer_solar_count_query(message, session_state, backend=backend,
                                             model=model, pre_trace=pre)
        return _answer_solar_query(message, session_state, backend=backend, model=model,
                                   pre_trace=pre)

    # A catalogue question about a named star is answered from NASA's record
    # rather than by running a detector over its light curve -- "does it have
    # planets" is a lookup, and a published disposition beats an inference.
    if _KEPID_PATTERN.search(message):
        sel = select_model(message, domain="astronomy", available={"target"},
                           strategy="filter_only")
        if sel["model_id"] == "astronomy.koi_lookup":
            return _answer_koi_lookup(message, None, session_state,
                                      pre_trace=["[chat] catalogue question about a named KIC"]
                                      + sel["trace"])

    mode_match = _MODE_PIN_PATTERN.search(message)
    if mode_match:
        chosen_mode = (mode_match.group(1) or mode_match.group(2)).lower()
        session_state["pinned_domain"] = chosen_mode
        noun = "ticker" if chosen_mode == "finance" else "target"
        return {"answer": f"Got it -- I'll default to {chosen_mode} for anomaly/follow-up questions that don't "
                           f"name a specific {noun}, until you switch modes again.",
                "domain": chosen_mode, "trace": [f"[chat] pinned_domain set to '{chosen_mode}'"], "retrieved_context": []}

    pending = session_state.pop("_pending_disambiguation", None)
    if pending:
        chosen = _try_resolve_disambiguation(message, pending["candidates"])
        if chosen:
            trace_note = f"[chat] disambiguation resolved '{pending['original_query']}' -> {chosen['symbol']} ({chosen['name']})"

            # The user's SECOND message is just a ticker pick ("TCS.NS"), so
            # re-routing on it loses whatever they originally asked. Without
            # this, "is tata consultancy risky?" disambiguates and then comes
            # back with RSI and MACD, because the risk intent lived only in
            # the first message.
            original = f"{pending.get('original_query') or ''} {pending.get('focus') or ''}"
            if _RISK_QUERY_PATTERN.search(original):
                session_state["last_ticker"] = chosen["symbol"]
                risk_answer = _answer_risk_query(original, session_state,
                                                 backend=backend, model=model)
                if risk_answer is not None:
                    risk_answer["trace"] = [trace_note] + list(risk_answer.get("trace", []))
                    return risk_answer

            return _answer_finance_or_astronomy(pending["domain"], chosen["symbol"], pending["focus"], session_state,
                                                 backend=backend, model=model, pre_trace=[trace_note],
                                                 forecast_days=pending.get("forecast_days"),
                                                 want_anomaly_chart=bool(_ANOMALY_QUERY_PATTERN.search(pending["focus"] or "")),
                                                 message_for_task=pending.get("original_query") or pending["focus"])
        # Doesn't look like a selection reply (e.g. the user asked something
        # else entirely) -- pending is already popped, fall through to a
        # normal fresh-query interpretation instead of forcing a match.

    parsed = interpret_message(
        message,
        last_domain=session_state.get("last_domain"),
        last_ticker=session_state.get("last_ticker"),
        last_target=session_state.get("last_target"),
        backend=backend, model=model,
    )
    domain = parsed["domain"]
    focus = parsed["question_focus"]
    key = parsed.get("ticker_or_target")
    last_domain = session_state.get("last_domain")
    last_ticker = session_state.get("last_ticker")
    last_target = session_state.get("last_target")
    last_mutual_fund = session_state.get("last_mutual_fund")
    trace_notes = []
    is_anomaly_query = bool(_ANOMALY_QUERY_PATTERN.search(message))

    if is_anomaly_query:
        # Small local models are unreliable at extracting a clean
        # question_focus (see README design notes) -- force it here so the
        # explainer prompt reliably emphasizes the ANOMALY DETECTION block
        # regardless of what the model's own intent JSON said.
        focus = "anomaly detection: what was flagged as unusual, its severity, and why (cite the actual detected reasons)"
        trace_notes.append("[chat] detected anomaly-focused question -- forcing question_focus for a grounded, detector-based answer")

    # When the message's own vocabulary contradicts the parsed domain, the
    # previous turn is almost certainly dragging the classification. Measured:
    # "how accurate is your price model for TCS" asked after an astronomy turn
    # parsed as domain='astronomy', key=None on one run and
    # domain='finance', key='TCS.NS' on the next -- same message, same model.
    # With a clean context it was finance on every trial, so the fix is to
    # re-ask without the stale context rather than to second-guess the answer.
    # Astronomy in this app is ALWAYS addressed by a stellar id. So a parsed
    # domain of "astronomy" for a message that names some other entity is
    # wrong by construction. Measured: "any anomalies in TCS" asked after an
    # astronomy turn came back as domain='astronomy', key='KIC 5358624' --
    # the parser ignored TCS completely and reused the previous target, which
    # would have run light-curve detection and reported it as an answer about
    # a stock. Neither the vocabulary check nor the anomaly pattern catches
    # this: "anomalies" is deliberately domain-agnostic and a bare ticker is
    # in no word list.
    if domain == "astronomy" and not _STELLAR_ID.search(message) and _NON_STELLAR_ENTITY.search(message):
        stolen = _NON_STELLAR_ENTITY.search(message).group(0)
        reparsed = interpret_message(message, last_domain=None, last_ticker=None,
                                     last_target=None, backend=backend, model=model)
        trace_notes.append(
            f"[chat] parser said 'astronomy' for a message naming '{stolen}' and no stellar id "
            f"-- it was reusing the previous target; re-parsed cleanly as "
            f"'{reparsed.get('domain')}', key='{reparsed.get('ticker_or_target')}'")
        domain = reparsed.get("domain", "finance")
        key = reparsed.get("ticker_or_target") or stolen
        focus = reparsed.get("question_focus") or focus

    # A hard invariant, enforced rather than asked for: the astronomy domain
    # is addressed only by a KIC/TIC/EPIC id. A key that is not one cannot be
    # an astronomy target, whatever the parser says.
    #
    # This is the backstop because re-asking is not enough. Given "any
    # anomalies in TCS" with NO conversation context at all, the model still
    # answered 'astronomy' on every trial -- the re-parse recovered the right
    # entity ('TCS' instead of a stale 'KIC 5358624') but not the right
    # domain. Left alone it would run light-curve detection on a stock and
    # present the result as an answer.
    if domain == "astronomy" and key and not _STELLAR_ID.search(str(key)):
        trace_notes.append(f"[chat] '{key}' is not a KIC/TIC id, so this cannot be an astronomy "
                           f"target -- routing to finance")
        domain = "finance"

    vocab_domain = _general_question_domain(message)
    if vocab_domain and domain in ("general", "finance", "astronomy") and domain != vocab_domain:
        reparsed = interpret_message(message, last_domain=vocab_domain,
                                     last_ticker=None, last_target=None,
                                     backend=backend, model=model)
        if reparsed.get("domain") == vocab_domain:
            trace_notes.append(
                f"[chat] message vocabulary says '{vocab_domain}' but the parser said "
                f"'{domain}' -- re-parsed without the previous turn's context and got "
                f"'{reparsed['domain']}', key='{reparsed.get('ticker_or_target')}'")
            domain = reparsed["domain"]
            key = reparsed.get("ticker_or_target") or key
            focus = reparsed.get("question_focus") or focus

    india_index_ticker = _detect_india_index(message)
    if india_index_ticker:
        domain, key = "finance", india_index_ticker
        trace_notes.append(f"[chat] detected an Indian index name in the message -- routing to finance with ticker='{key}'")

    # A bare follow-up with no named entity ("any anomalies?") should
    # resolve to: an explicit session-level mode pin first, else whichever
    # domain the conversation was already in -- same fallback the
    # referential-pronoun override already used, broadened to also fire on
    # anomaly-intent messages (which often don't contain a pronoun at all).
    fallback_domain = session_state.get("pinned_domain") or last_domain
    if domain == "general" and fallback_domain in ("finance", "astronomy", "mutual_fund") and (
            _REFERENTIAL_PATTERN.search(message) or is_anomaly_query):
        domain = fallback_domain
        key = {"finance": last_ticker, "astronomy": last_target, "mutual_fund": last_mutual_fund}[domain]
        trace_notes.append(f"[chat] intent parser said 'general' but message looks referential/anomaly-focused -- "
                            f"overriding to domain='{domain}', key='{key}' from conversation context/pinned mode")

    if domain == "mutual_fund":
        return _answer_mutual_fund_query(key or message, focus, session_state, backend=backend, model=model, pre_trace=trace_notes)

    if domain == "general" or not key:
        kwargs = {}
        if backend:
            kwargs["backend"] = backend
        if model:
            kwargs["model"] = model
        context_hint = (f" (For context, the previous discussion was about '{last_ticker or last_target}' "
                         f"in the {last_domain} domain -- only use this if actually relevant to the question; "
                         f"ignore it entirely if this question is unrelated.)"
                         if last_domain else "")
        # Persona for a general question: the message's OWN vocabulary first,
        # the previous turn only as a tiebreak.
        #
        # Choosing purely on the previous turn stranded the conversation. After
        # a few questions about a light curve, "how accurate is your price
        # model" and "does news sentiment help prediction" were both answered
        # by the astronomy explainer, which then -- correctly, under its own
        # rules -- refused to discuss finance and lectured the user about
        # Kepler. The intent parser had returned "general" for both, so the
        # domain never got a chance to change, and the sticky persona made the
        # assistant look broken for the rest of the session.
        subject = _general_question_domain(message) or last_domain
        system_prompt = (ASTRONOMY_EXPLAINER_SYSTEM_PROMPT if subject == "astronomy"
                         else FINANCE_EXPLAINER_SYSTEM_PROMPT)
        trace_notes.append(f"[chat] general question answered with the "
                           f"{subject or 'finance'} persona "
                           f"(from the question's own wording)"
                           if _general_question_domain(message)
                           else f"[chat] general question -- no domain words, "
                                f"using previous domain '{last_domain}'")
        answer = call_llm(message + context_hint, system=system_prompt, **kwargs)
        return {"answer": answer, "domain": "general", "trace": ["[chat] answered as general knowledge, no pipeline run"], "retrieved_context": []}

    # A forecast request is whatever the TASK CLASSIFIER calls one, with the
    # older keyword pattern kept as a second chance.
    #
    # The pattern alone was too literal: it requires "price", "predict",
    # "forecast" or "next", so "what will TCS.NS be in 30 days" -- an
    # unmistakable forecast request -- matched nothing and fell through to
    # the general finance pipeline, which answered with backtest RMSEs
    # instead of a forecast. The selector had already classified it correctly
    # as forecast_value; there is no reason to ask a weaker regex to
    # re-decide something the selector knows.
    _forecast_task = (classify_task(message)[0] == FORECAST_VALUE)
    forecast_days = (_extract_forecast_days(message)
                     if domain == "finance" and (_forecast_task or _FORECAST_PATTERN.search(message))
                     else None)

    # Skip resolution entirely for: index tickers (already fully resolved by
    # _detect_india_index above); a key that already equals last_ticker
    # (by construction last_ticker is only ever set AFTER resolution
    # completed in a previous turn -- re-resolving it on every follow-up
    # would re-trigger disambiguation on an already-settled ticker, which
    # is exactly what happened in testing with "what about its RSI" right
    # after picking RELIANCE.NS from a disambiguation prompt); and anything
    # the user typed an explicit exchange suffix for themselves. Everything
    # else gets re-derived from the bare name via a real search,
    # specifically so the model's own ticker autocompletion can't paper
    # over real ambiguity.
    if (domain == "finance" and key and not key.startswith("^") and key != last_ticker
            and not _EXPLICIT_SUFFIX_IN_TEXT.search(message)):
        bare_query = key.split(".")[0]
        resolution = resolve_ticker(bare_query)

        if resolution["status"] == "ambiguous":
            session_state["_pending_disambiguation"] = {
                "domain": domain, "focus": focus, "candidates": resolution["candidates"],
                "original_query": bare_query, "forecast_days": forecast_days,
            }
            return {"answer": _format_disambiguation_answer(resolution["candidates"], bare_query), "domain": "finance",
                    "trace": [f"[chat] '{bare_query}' matched multiple companies -- asking user to choose"], "retrieved_context": []}

        if resolution["status"] == "not_found":
            return {"answer": f"I couldn't find a stock matching \"{bare_query}\". Try the exact ticker symbol, e.g. RELIANCE.NS.",
                    "domain": "finance", "trace": [f"[chat] no ticker match for '{bare_query}'"], "retrieved_context": []}

        if resolution["symbol"].upper() != key.upper():
            trace_notes.append(f"[chat] resolved '{bare_query}' -> {resolution['symbol']} ({resolution['name']}, {resolution['exchange']})")
        key = resolution["symbol"]

    return _answer_finance_or_astronomy(domain, key, focus, session_state, backend=backend, model=model,
                                         pre_trace=trace_notes, forecast_days=forecast_days,
                                         want_anomaly_chart=is_anomaly_query,
                                         message_for_task=message)


if __name__ == "__main__":
    session = {}
    for msg in [
        "what is the price for reliance in the next thirty days",  # ambiguous name -> should ask to disambiguate
        "1",  # picks RELIANCE.NS -> should now run the actual forecast + return a chart
        "what about its RSI",  # follow-up on the resolved ticker
    ]:
        print(f"\n>>> {msg}")
        result = answer_query(msg, session, backend="ollama", model="llama3.2:3b")
        print(result["answer"])
        print("has chart:", "chart" in result)
        print("trace:", result["trace"])


def _answer_risk_query(message: str, session_state: dict, backend: str | None = None,
                       model: str | None = None):
    """Forward-looking risk for one stock. Returns None if no ticker is found,
    so the caller falls through to its normal routing.

    Answered by the risk composition: five registry models chosen through the
    selector, every figure carrying its validation status. The model phrases
    the facts; the table of headline figures and their statuses is appended
    verbatim, because a small local model drops qualifiers such as "this
    backtest failed" more readily than it drops numbers.
    """
    from agents.answer_style import CHAT_SYSTEM_PROMPT, guard_answer
    from agents.risk_composition import (assess_stock_risk, chat_facts, figure,
                                         status_table)

    ticker = _resolve_ticker_from_message(message, session_state)
    if not ticker:
        return None

    try:
        a = assess_stock_risk(ticker, with_news=True)
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"I couldn't work out the risk picture for {ticker}: {exc}",
                "domain": "finance", "trace": []}
    working = [c for c in a["components"].values() if "error" not in c]
    if not working:
        reasons = "; ".join(c["error"] for c in a["components"].values())
        return {"answer": f"I couldn't work out the risk picture for {ticker}. {reasons}",
                "domain": "finance", "trace": []}

    facts = chat_facts(a, context=_market_context(ticker, a))
    prompt = (f"{facts}\n\nThe user asked: {message}\n\nAnswer conversationally in a few "
              f"short paragraphs, using only the facts above. A table of the headline numbers "
              f"will be shown under your answer, so do not repeat it as a table.")
    try:
        answer = call_llm(prompt, system=CHAT_SYSTEM_PROMPT, backend=backend, model=model)
    except Exception:
        answer = a["summary"]     # without a model the summary is still exact

    n = figure(a, "anomalies", "n_recent_anomalies")
    correction = (f"{n['value']} unusual days were flagged in the last "
                  f"{n['label'].rsplit(' ', 2)[-2]} days" if n else None)
    answer, invented = guard_answer(answer, facts, correction=correction)
    answer = f"{answer}\n\n**The numbers, and how far to trust each one**\n\n{status_table(a)}"

    trace = [f"[risk] {r['role']}: {r['question']!r} -> task {r['task']} -> {r['model_id']} "
             f"({len(r['shortlist'])} feasible)" for r in a["routing"]]
    if a["freshness"].get("warning"):
        trace.append("[risk] " + a["freshness"]["warning"])
    if invented:
        trace.append(f"[guard] corrected {len(invented)} figure(s) not in the computed facts: "
                     f"{', '.join(invented)}")

    profile = a["components"].get("tail_risk", {}).get("detail", {}).get("profile")
    charts = []
    if profile:
        try:
            from dashboard.charts import risk_attribution_chart, risk_price_chart

            charts = [c for c in (risk_price_chart(ticker, profile),
                                  risk_attribution_chart(profile)) if c is not None]
        except Exception:
            pass   # a missing chart should never cost the user their answer

    session_state["last_domain"] = "finance"
    session_state["last_ticker"] = ticker
    return {"answer": answer, "domain": "finance", "ticker": ticker, "trace": trace,
            "risk_profile": profile, "risk_assessment": a, "charts": charts,
            "retrieved_context": []}


def _market_context(ticker: str, a: dict) -> str | None:
    """Where this stock's forecast sits among the 50, from the saved snapshot.

    Only used when the snapshot describes the same day as the assessment; a
    comparison against another day's market is worse than none.
    """
    import statistics

    from agents.risk_composition import figure, load_market_snapshot

    m = load_market_snapshot()
    f = figure(a, "volatility", "forecast_vol_pct")
    if not m or not f or not m.get("ranking"):
        return None
    if (m.get("freshness") or {}).get("as_of", {}).get("volatility") != \
            a["components"]["volatility"].get("as_of"):
        return None
    vols = [r["forecast_vol_pct"] for r in m["ranking"]]
    rank = 1 + sum(v > f["value"] for v in vols)
    return (f"across the {len(vols)} stocks in the panel the median forecast is "
            f"{statistics.median(vols):.1f}% and the range is {min(vols):.1f}% to "
            f"{max(vols):.1f}%; this stock's {f['value']:.1f}% ranks {rank} of {len(vols)}, "
            f"where 1 is the most volatile")


def _answer_market_risk_query(message: str, session_state: dict, backend: str | None = None,
                              model: str | None = None) -> dict:
    """Which names are riskiest, from the saved 50-name assessment.

    The saved snapshot is used because a fresh pass takes about two minutes;
    its date is stated in the answer so a stale one cannot pass for today's.
    """
    from agents.answer_style import CHAT_SYSTEM_PROMPT, guard_answer
    from agents.risk_composition import (assess_market_risk, load_market_snapshot,
                                         market_chat_facts, market_status_table,
                                         save_market_snapshot)

    m = load_market_snapshot()
    trace = []
    if m is None:
        trace.append("[risk] no saved market assessment -- computing all 50 names")
        m = assess_market_risk()
        save_market_snapshot(m)
    trace.append(f"[risk] market assessment computed {m['generated_at']}, "
                 f"{m['n_ranked']} names ranked by forecast volatility")

    facts = market_chat_facts(m)
    prompt = (f"{facts}\n\nThe user asked: {message}\n\nAnswer conversationally in a few "
              f"short paragraphs using only the facts above. A ranked table will be shown "
              f"under your answer, so do not repeat it.")
    try:
        answer = call_llm(prompt, system=CHAT_SYSTEM_PROMPT, backend=backend, model=model)
    except Exception:
        answer = m["summary"]
    answer, invented = guard_answer(answer, facts)
    if invented:
        trace.append(f"[guard] corrected {len(invented)} figure(s): {', '.join(invented)}")
    as_of = (m.get("freshness") or {}).get("as_of", {}).get("volatility", "?")
    answer = (f"{answer}\n\n**Riskiest names for the next session** "
              f"(data as of {as_of})\n\n{market_status_table(m)}")
    session_state["last_domain"] = "finance"
    return {"answer": answer, "domain": "finance", "trace": trace,
            "market_risk": {k: v for k, v in m.items() if k != "assessments"},
            "retrieved_context": []}


def _resolve_ticker_from_message(message: str, session_state: dict) -> str | None:
    """Ticker from the message, or the one from the previous turn.

    Kept deliberately simple: an explicit suffix, then a name lookup against
    the cached panel, then whatever the last turn settled on. Anything more
    clever belongs in the existing resolution path, not duplicated here.
    """
    explicit = _EXPLICIT_SUFFIX_IN_TEXT.search(message)
    if explicit:
        return explicit.group(0).upper()
    try:
        from quant.data import load_panel

        close, _ = load_panel()
        upper = message.upper()
        for col in close.columns:
            base = col.replace(".NS", "")
            if len(base) >= 3 and re.search(rf"\b{re.escape(base)}\b", upper):
                return col
    except Exception:
        pass
    return session_state.get("last_ticker")
