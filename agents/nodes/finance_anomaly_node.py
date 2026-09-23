"""Finance Anomaly Agent: multi-type statistical anomaly detection (return
outliers, volatility spikes, volume spikes, gap events, and -- when
sentiment data is available -- sentiment-price divergence) over the same
price+sentiment data finance_node/sentiment_node already fetched, fused
into one severity-scored, typed anomaly table per day. Runs after
sentiment_node so the sentiment-divergence detector has real daily
sentiment to compare against, not just price data alone."""

from agents.state import AgentState
from finance.anomaly_detector import detect_all, top_anomalies


def finance_anomaly_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    trace = state.get("trace", [])
    finance_result = dict(state["finance_result"])

    if finance_result.get("error"):
        trace.append(f"[finance_anomaly_agent] {ticker}: skipped, finance_agent reported insufficient data")
        return {**state, "finance_anomaly_result": {}, "trace": trace}

    features = finance_result.pop("_features_df", None)
    daily_sentiment = finance_result.pop("_daily_sentiment", None)
    if features is None:
        trace.append(f"[finance_anomaly_agent] {ticker}: skipped, no feature data available")
        return {**state, "finance_result": finance_result, "finance_anomaly_result": {}, "trace": trace}

    has_sentiment = daily_sentiment is not None and not daily_sentiment.empty
    scored = detect_all(features, sentiment_series=daily_sentiment if has_sentiment else None)
    top = top_anomalies(scored, n=5)

    result = {
        "ticker": ticker,
        "n_days_analyzed": len(scored),
        "n_anomalous_days": int(scored["is_anomaly"].sum()),
        "used_sentiment_divergence": has_sentiment,
        "top_anomalies": [
            {
                "date": str(idx.date()),
                "severity": row["severity_label"],
                "severity_score": round(float(row["severity_score"]), 3),
                "reasons": row["reasons"],
            }
            for idx, row in top.iterrows()
        ],
        # Both kept for the chatbot's shared anomaly-chart builder, not
        # passed to the LLM (explainer_node reads named fields explicitly,
        # never the whole dict).
        "_scored_df": scored,
        "_features_df": features,
    }
    trace.append(
        f"[finance_anomaly_agent] {ticker}: {result['n_anomalous_days']}/{result['n_days_analyzed']} "
        f"days flagged anomalous (sentiment divergence {'enabled' if has_sentiment else 'skipped, no headlines'})"
    )
    return {**state, "finance_result": finance_result, "finance_anomaly_result": result, "trace": trace}
