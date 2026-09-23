"""Adaptive dashboard layer: an LLM call that turns a typed preference
("just show me sentiment and RSI for TCS.NS") into a layout config the Home
page (app.py) renders from. This is a genuine agent decision (the model
chooses which sections/size fit the request) rather than a fixed set of
toggle switches -- toggles in the UI are just a manual override/reset path.
"""

import re

from agents.json_utils import extract_json
from agents.llm_backend import call_llm

# Observed in testing: the small local model (llama3.2:3b) reliably picks
# the right *sections* but is unreliable on two explicit, unambiguous
# signals the user actually typed -- a resize keyword ("make it bigger")
# and a named ticker. It even claimed "adjusted layout for a bigger chart"
# in its own note while leaving layout_size="normal" unchanged. Rather than
# trust the model on things a plain keyword match already gets right,
# override its output when a clear signal is present in the raw text.
_GROW_PATTERN = re.compile(r"\b(bigger|larger|expand|zoom|enlarge)\b", re.IGNORECASE)
_SHRINK_PATTERN = re.compile(r"\b(smaller|shrink|minimal|compact)\b", re.IGNORECASE)
# Up to 10 letters (not 5) + optional .NS/.BO suffix -- plain {2,5} would
# silently miss common Indian tickers like RELIANCE, HDFCBANK, ICICIBANK,
# BAJFINANCE now that Indian stocks are this app's default focus.
_TICKER_PATTERN = re.compile(r"\b[A-Z]{2,10}(?:\.(?:NS|BO))?\b")
_TICKER_STOPWORDS = {"RSI", "MACD", "CAPM", "PE", "EPS", "ROE", "YTD", "USD", "AI", "LLM", "RAG",
                      "API", "CEO", "GDP", "IPO", "ETF", "SMA", "EMA", "NSE", "BSE", "NIFTY", "SENSEX"}

KNOWN_SECTIONS = ["stock_charts", "company_metrics", "prediction", "sentiment", "astronomy"]

DEFAULT_CONFIG = {
    "visible_sections": list(KNOWN_SECTIONS),
    "layout_size": "normal",
    "focus_ticker": "RELIANCE.NS",
    "note": "Default dashboard -- all sections shown.",
}

DASHBOARD_SYSTEM_PROMPT = f"""You control the layout of a Streamlit dashboard with these togglable sections:
- stock_charts: candlestick/line price chart + RSI/MACD/moving-average overlays
- company_metrics: market cap, beta, P/E, profit margins, quick ratio, revenue growth, CAPM
- prediction: ARIMA vs XGBoost vs XGBoost+sentiment price forecast comparison
- sentiment: recent headlines + FinBERT sentiment score
- astronomy: light curve anomaly detection (IsolationForest + autoencoder)

Given the user's typed preference and the current config, output ONLY a JSON object, no other text:
{{
  "visible_sections": [subset of {KNOWN_SECTIONS}],
  "layout_size": "compact" | "normal" | "large",
  "focus_ticker": string or null (only if they named a specific ticker to focus the finance sections on),
  "note": one short sentence confirming what you changed, to show the user
}}
If the request is unclear, or asks to "reset"/"show everything"/"default", return all 5 sections visible, layout_size "normal".
Output strictly valid JSON and nothing else."""


def parse_dashboard_preference(user_text: str, current_config: dict | None = None,
                                backend: str | None = None, model: str | None = None) -> dict:
    current_config = current_config or DEFAULT_CONFIG
    prompt = f"Current config: {current_config}\n\nUser preference: \"{user_text}\"\n\nJSON:"

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model

    raw = call_llm(prompt, system=DASHBOARD_SYSTEM_PROMPT, **kwargs)
    try:
        parsed = extract_json(raw)
    except Exception:
        return {**current_config, "note": "Could not parse that preference -- keeping the current layout."}

    visible = [s for s in parsed.get("visible_sections", []) if s in KNOWN_SECTIONS]
    if not visible:
        visible = list(KNOWN_SECTIONS)

    layout_size = parsed.get("layout_size") if parsed.get("layout_size") in ("compact", "normal", "large") else "normal"
    if _GROW_PATTERN.search(user_text):
        layout_size = "large"
    elif _SHRINK_PATTERN.search(user_text):
        layout_size = "compact"

    focus_ticker = parsed.get("focus_ticker") or current_config.get("focus_ticker", "RELIANCE.NS")
    ticker_candidates = [t for t in _TICKER_PATTERN.findall(user_text) if t not in _TICKER_STOPWORDS]
    if len(ticker_candidates) == 1:
        focus_ticker = ticker_candidates[0]

    return {
        "visible_sections": visible,
        "layout_size": layout_size,
        "focus_ticker": focus_ticker,
        "note": parsed.get("note", "Layout updated."),
    }


if __name__ == "__main__":
    for pref in ["just show me sentiment and RSI for HDFCBANK.NS", "hide the astronomy section, I only care about finance today",
                 "make the prediction chart bigger", "reset to default"]:
        cfg = parse_dashboard_preference(pref, backend="ollama", model="llama3.2:3b")
        print(f"\n>>> {pref}\n{cfg}")
