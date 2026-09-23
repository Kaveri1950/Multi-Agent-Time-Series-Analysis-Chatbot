import streamlit as st

from dashboard.sections import render_model_comparison, render_prediction, render_sentiment

st.title("🔮 Stock Prediction")
st.caption("Four models predicting the same held-out days: ARIMA, XGBoost, XGBoost + FinBERT sentiment, "
           "and a Gaussian Process -- all measured against the actual closes and against a naive "
           "\"assume no change\" baseline.")

ticker = st.text_input("Stock Ticker", "TCS.NS", key="prediction_ticker",
                        help="Any yfinance ticker works, e.g. TCS.NS, RELIANCE.NS, HDFCBANK.NS, INFY.NS").upper().strip()

if ticker:
    try:
        _, daily_sentiment = render_sentiment(ticker, key_prefix="prediction")
        st.divider()
        render_model_comparison(ticker, sentiment_daily=daily_sentiment, key_prefix="prediction")
        st.divider()
        render_prediction(ticker, sentiment_daily=daily_sentiment, key_prefix="prediction")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't run prediction for '{ticker}': {exc}")
