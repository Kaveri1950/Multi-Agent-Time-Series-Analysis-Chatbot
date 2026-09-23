"""Shared presentation contract for anomaly-detection results across
domains. This does NOT unify the detection math -- finance keeps its 5
rolling-z-score detectors (finance/anomaly_detector.py) and astronomy keeps
IsolationForest + the autoencoder's reconstruction error
(astronomy/anomaly_detector.py, astronomy/ts_embedder.py), each tuned to
what actually differs about its signal. What's shared is the OUTPUT shape:
a normalized "what was flagged, how severe, and why" summary, so the
chatbot has one consistent way to talk about anomalies regardless of which
domain answered the question -- one command surface, two specialized
detectors underneath.
"""

from typing import TypedDict


class AnomalyItem(TypedDict):
    label: str          # a date (finance) or "window N" (astronomy)
    severity: str        # LOW | MEDIUM | HIGH | NONE
    score: float
    reasons: list[str]


class AnomalySummary(TypedDict):
    domain: str          # "finance" | "astronomy"
    key: str             # ticker or target
    methods: list[str]   # which detector(s) actually ran
    n_units_analyzed: int
    n_flagged: int
    flagged_rate: float
    top_anomalies: list[AnomalyItem]


def from_finance_result(finance_anomaly_result: dict) -> AnomalySummary:
    n_total = finance_anomaly_result.get("n_days_analyzed", 0)
    n_flagged = finance_anomaly_result.get("n_anomalous_days", 0)
    methods = ["return_outlier", "volatility_spike", "volume_spike", "gap_event"]
    if finance_anomaly_result.get("used_sentiment_divergence"):
        methods.append("sentiment_divergence")
    return AnomalySummary(
        domain="finance",
        key=finance_anomaly_result.get("ticker", ""),
        methods=methods,
        n_units_analyzed=n_total,
        n_flagged=n_flagged,
        flagged_rate=round(n_flagged / n_total, 4) if n_total else 0.0,
        top_anomalies=[
            AnomalyItem(label=a["date"], severity=a["severity"], score=a["severity_score"], reasons=a["reasons"])
            for a in finance_anomaly_result.get("top_anomalies", [])
        ],
    )


def from_astronomy_result(astronomy_result: dict) -> AnomalySummary:
    n_total = astronomy_result.get("n_windows", 0)
    n_flagged = astronomy_result.get("n_flagged_anomalies", 0)
    methods = ["isolation_forest"]
    if astronomy_result.get("mean_reconstruction_error") is not None:
        methods.append("autoencoder_reconstruction_error")
    return AnomalySummary(
        domain="astronomy",
        key=astronomy_result.get("target", ""),
        methods=methods,
        n_units_analyzed=n_total,
        n_flagged=n_flagged,
        flagged_rate=round(n_flagged / n_total, 4) if n_total else 0.0,
        top_anomalies=[
            AnomalyItem(
                label=f"window {w['window_id']}", severity=w.get("severity_label", "LOW"),
                score=w.get("anomaly_score", 0.0), reasons=w.get("reasons", []),
            )
            for w in astronomy_result.get("top_anomaly_windows", [])
        ],
    )


def summarize(domain: str, state: dict) -> AnomalySummary | None:
    """Convenience entry point for callers (e.g. agents/chat_agent.py) that
    have a full graph AgentState and just want "the anomaly summary for
    whichever domain this was", without caring which result key to read."""
    if domain == "finance" and state.get("finance_anomaly_result"):
        return from_finance_result(state["finance_anomaly_result"])
    if domain == "astronomy" and state.get("astronomy_result"):
        return from_astronomy_result(state["astronomy_result"])
    return None
