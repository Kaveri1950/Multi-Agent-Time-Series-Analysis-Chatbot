"""Finance Agent: pulls price history + technical indicators, runs the
ARIMA-baseline-vs-XGBoost-augmented comparison (sentiment plugged in
afterwards by sentiment_node, since sentiment_score defaults to 0 here)."""

from agents.state import AgentState
from finance.price_data import add_technical_indicators, get_price_history
from finance.price_model import InsufficientDataError, arima_baseline, build_training_frame, train_xgboost_model


def finance_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    trace = state.get("trace", [])

    raw = get_price_history(ticker, start="2023-01-01")
    features = add_technical_indicators(raw)

    try:
        arima = arima_baseline(features["Close"], test_size=30)
        frame = build_training_frame(features, sentiment_series=None)
        xgb_result = train_xgboost_model(frame, test_size=30)
    except InsufficientDataError as exc:
        # Found in testing: a thinly-traded/illiquid ticker (a candidate
        # from finance/ticker_search.py's disambiguation can be one) may
        # have too little history in Yahoo Finance for ARIMA/XGBoost to
        # fit at all. Surface this as data on the result (which
        # sentiment_node and explainer_node both check for) instead of
        # letting the exception propagate uncaught through the whole
        # LangGraph run -- that would crash agents/graph.py for ANY caller
        # (chatbot, CLI, dashboard), not just the price-forecast path.
        trace.append(f"[finance_agent] {ticker}: insufficient data -- {exc}")
        return {**state, "finance_result": {"ticker": ticker, "error": "insufficient_data", "error_message": str(exc)}, "trace": trace}

    result = {
        "ticker": ticker,
        "latest_close": float(features["Close"].iloc[-1]),
        "arima_rmse": arima["rmse"],
        "xgboost_rmse": xgb_result["rmse"],
        "latest_rsi": float(features["RSI_14"].iloc[-1]),
        "latest_macd": float(features["MACD"].iloc[-1]),
        "feature_importance": xgb_result["feature_importance"],
        "_features_df": features,  # kept for sentiment_node to reuse without re-downloading
    }
    trace.append(f"[finance_agent] {ticker}: ARIMA RMSE={arima['rmse']}, XGBoost RMSE={xgb_result['rmse']}")
    return {**state, "finance_result": result, "trace": trace}
