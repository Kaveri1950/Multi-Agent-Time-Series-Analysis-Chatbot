"""Sentiment Agent: free RSS/Reddit headlines -> local FinBERT -> daily
score, then re-runs the XGBoost model WITH sentiment so finance_result
carries the direct with/without-sentiment RMSE comparison end to end."""

from agents.state import AgentState
from finance.price_model import build_training_frame, train_xgboost_model
from finance.sentiment import get_ticker_sentiment


def sentiment_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    trace = state.get("trace", [])
    finance_result = dict(state["finance_result"])

    if finance_result.get("error"):
        # finance_node already hit InsufficientDataError -- nothing to
        # augment with sentiment, and no _features_df to pop.
        trace.append(f"[sentiment_agent] {ticker}: skipped, finance_agent reported insufficient data")
        return {**state, "finance_result": finance_result, "sentiment_result": {}, "trace": trace}

    features = finance_result.pop("_features_df")

    scored_headlines, daily_sentiment = get_ticker_sentiment(ticker)

    if not daily_sentiment.empty:
        frame_with_sentiment = build_training_frame(features, sentiment_series=daily_sentiment)
        xgb_with = train_xgboost_model(frame_with_sentiment, test_size=30)
        finance_result["xgboost_with_sentiment_rmse"] = xgb_with["rmse"]
    else:
        finance_result["xgboost_with_sentiment_rmse"] = None

    # Re-attach for finance_anomaly_node (runs next) -- it needs the full
    # features_df (for return/volatility/volume/gap detectors) and the full
    # daily_sentiment series (for the sentiment-divergence detector), not
    # just the summary stats below. finance_anomaly_node pops both when done.
    finance_result["_features_df"] = features
    finance_result["_daily_sentiment"] = daily_sentiment

    n_headlines = len(scored_headlines)
    avg_sentiment = float(scored_headlines["sentiment_score"].mean()) if n_headlines else 0.0

    sentiment_result = {
        "ticker": ticker,
        "n_headlines": n_headlines,
        "avg_sentiment_score": round(avg_sentiment, 3),
        "recent_headlines": scored_headlines[["headline", "sentiment_label"]].head(5).to_dict("records") if n_headlines else [],
    }

    trace.append(f"[sentiment_agent] {ticker}: {n_headlines} headlines, avg sentiment={avg_sentiment:.3f}")
    return {**state, "finance_result": finance_result, "sentiment_result": sentiment_result, "trace": trace}
