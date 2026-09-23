"""Adaptive layout layer for the Astronomy page, mirroring
agents/dashboard_agent.py's pattern (an LLM call turns a typed preference
into a layout config) but scoped to the Astronomy page's own five-step
flow instead of the Home page's cross-domain sections. Kept as a separate
agent rather than folded into dashboard_agent.py because the two control
genuinely different section vocabularies -- reusing one KNOWN_SECTIONS
list for both would let an Astronomy preference silently reference a
Home-only section name (or vice versa).
"""

import re

from agents.json_utils import extract_json
from agents.llm_backend import call_llm

# Same rationale as dashboard_agent.py's _GROW_PATTERN/_SHRINK_PATTERN and
# ticker-extraction safety net: the small local model reliably picks the
# right sections but is unreliable on explicit signals the user actually
# typed (a resize keyword, a specific count, a named target) -- override
# its output when a clear signal is present in the raw text rather than
# trusting it to have extracted the number/target correctly itself.
_GROW_PATTERN = re.compile(r"\b(bigger|larger|expand|zoom|enlarge)\b", re.IGNORECASE)
_SHRINK_PATTERN = re.compile(r"\b(smaller|shrink|minimal|compact)\b", re.IGNORECASE)
_COUNT_PATTERN = re.compile(r"\btop\s+(\d+)\b|\b(\d+)\s+anomal", re.IGNORECASE)
_TARGET_PATTERN = re.compile(r"\b(?:KIC|TIC|KOI|Kepler-\d+)\s*[\d.-]*\b", re.IGNORECASE)

KNOWN_SECTIONS = ["light_curve", "description", "anomaly_detection", "normalized_view", "explanations"]

DEFAULT_CONFIG = {
    "visible_sections": list(KNOWN_SECTIONS),
    "layout_size": "normal",
    "top_n_anomalies": 5,
    "focus_target": None,
    "note": "Default layout -- all five steps shown.",
}

ASTRONOMY_DASHBOARD_SYSTEM_PROMPT = f"""You control the layout of a Streamlit light-curve analysis dashboard with
these togglable steps, always shown in this order when visible:
- light_curve: the raw flux-vs-time chart for the target
- description: computed summary stats (duration, cadence, flux range, variability) in plain language
- anomaly_detection: IsolationForest anomaly detection results, flagged windows shaded on the raw curve
- normalized_view: the same flux re-expressed as a z-score with +-2/+-3 sigma reference lines
- explanations: per-anomaly cards with the computed reasons it was flagged and a rule-based possible cause

Given the user's typed preference and the current config, output ONLY a JSON object, no other text:
{{
  "visible_sections": [subset of {KNOWN_SECTIONS}],
  "layout_size": "compact" | "normal" | "large",
  "top_n_anomalies": integer 1-10 (how many flagged windows to explain in detail),
  "focus_target": string or null (only if they named a specific KIC/TIC/Kepler target to switch to),
  "note": one short sentence confirming what you changed, to show the user
}}
If the request is unclear, or asks to "reset"/"show everything"/"default", return all 5 sections visible,
layout_size "normal", top_n_anomalies 5. Output strictly valid JSON and nothing else."""


def parse_astronomy_preference(user_text: str, current_config: dict | None = None,
                                 backend: str | None = None, model: str | None = None) -> dict:
    current_config = current_config or DEFAULT_CONFIG
    prompt = f"Current config: {current_config}\n\nUser preference: \"{user_text}\"\n\nJSON:"

    kwargs = {}
    if backend:
        kwargs["backend"] = backend
    if model:
        kwargs["model"] = model

    raw = call_llm(prompt, system=ASTRONOMY_DASHBOARD_SYSTEM_PROMPT, **kwargs)
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

    top_n = parsed.get("top_n_anomalies")
    top_n = int(top_n) if isinstance(top_n, (int, float)) and 1 <= top_n <= 10 else current_config.get("top_n_anomalies", 5)
    count_match = _COUNT_PATTERN.search(user_text)
    if count_match:
        digits = count_match.group(1) or count_match.group(2)
        candidate = int(digits)
        if 1 <= candidate <= 10:
            top_n = candidate

    focus_target = parsed.get("focus_target") or current_config.get("focus_target")
    target_match = _TARGET_PATTERN.search(user_text)
    if target_match:
        focus_target = target_match.group(0).strip()

    return {
        "visible_sections": visible,
        "layout_size": layout_size,
        "top_n_anomalies": top_n,
        "focus_target": focus_target,
        "note": parsed.get("note", "Layout updated."),
    }


if __name__ == "__main__":
    for pref in ["just show me the anomaly explanations", "focus on KIC 8462852 and show top 3 anomalies",
                 "make the charts bigger", "reset to default"]:
        cfg = parse_astronomy_preference(pref, backend="ollama", model="llama3.2:3b")
        print(f"\n>>> {pref}\n{cfg}")
