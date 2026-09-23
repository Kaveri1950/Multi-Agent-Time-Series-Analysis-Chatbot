"""Entrypoint / page router. Run with:  streamlit run app.py

Streamlit's older "just drop files in a pages/ folder" auto-discovery does
NOT register pages in this Streamlit version -- pages must be declared
explicitly via st.navigation()/st.Page(). st.set_page_config() must be
called exactly once per app, here, before pg.run() (each page module must
NOT call it again).
"""

import streamlit as st

st.set_page_config(page_title="BTP Multi-Agent Dashboard", page_icon="🛰️", layout="wide")

pg = st.navigation([
    st.Page("views/home.py", title="Home", icon="🛰️", default=True),
    st.Page("pages/1_Stock_Analysis.py", title="Stock Analysis", icon="📊"),
    st.Page("pages/2_Stock_Prediction.py", title="Stock Prediction", icon="🔮"),
    st.Page("pages/3_Astronomy.py", title="Astronomy", icon="🔭"),
    st.Page("pages/5_Market_Screener.py", title="Market Screener", icon="📈"),
    st.Page("pages/6_Mutual_Funds.py", title="Mutual Funds", icon="💰"),
    st.Page("pages/7_Solar_Weather.py", title="Solar Weather", icon="☀️"),
    st.Page("pages/8_Anomaly_Benchmark.py", title="Anomaly Benchmark", icon="🔍"),
    st.Page("pages/9_Stock_Risk.py", title="Stock Risk", icon="⚖️"),
    st.Page("pages/4_Chatbot.py", title="Chatbot", icon="💬"),
])
pg.run()
