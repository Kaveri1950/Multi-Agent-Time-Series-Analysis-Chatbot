import streamlit as st

from dashboard.sections import render_company_metrics, render_finance_anomalies, render_stock_charts

st.title("📊 Stock Analysis")

ticker = st.text_input("Stock Ticker", "RELIANCE.NS", key="analysis_ticker",
                        help="Any yfinance ticker works, e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS").upper().strip()

if ticker:
    try:
        feats = render_stock_charts(ticker, key_prefix="analysis")
        st.divider()
        render_company_metrics(ticker, key_prefix="analysis")
        st.divider()
        render_finance_anomalies(ticker, features_df=feats, key_prefix="analysis")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Couldn't load data for '{ticker}': {exc}")
