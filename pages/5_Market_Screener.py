import streamlit as st

from dashboard.sections import render_market_screener

st.title("📈 Indian Market Screener")
st.caption("Nifty 50 / 100 / 500 and Sensex 30 constituents, ranked by 1-day return and RSI(14). "
           "Constituent lists come from NSE's public archive (free, no key); Sensex 30 is a static snapshot "
           "(BSE doesn't expose a free scrapable list) -- see finance/india_universe.py to refresh it.")

col1, col2 = st.columns([2, 1])
with col1:
    index_name = st.selectbox("Index", ["nifty50", "nifty100", "nifty500", "sensex"],
                               format_func=lambda s: s.upper(), key="screener_index")
with col2:
    max_stocks = st.slider("Max stocks to screen", min_value=10, max_value=100, value=30, step=10,
                            key="screener_max", help="Capped for responsiveness on CPU-only hardware.")

try:
    render_market_screener(index_name, max_stocks=max_stocks, key_prefix="screener")
except Exception as exc:  # noqa: BLE001
    st.error(f"Screener failed for '{index_name}': {exc}")
