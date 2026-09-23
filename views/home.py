"""Adaptive Home dashboard body. Registered as a page by app.py via
st.navigation() -- st.set_page_config() is called once, centrally, in
app.py, not here (Streamlit only allows one call per app).
"""

import streamlit as st

from agents.dashboard_agent import DEFAULT_CONFIG, KNOWN_SECTIONS, parse_dashboard_preference
from dashboard.sections import render_astronomy, render_company_metrics, render_prediction, render_sentiment, render_stock_charts

st.title("🛰️📈 Multi-Agent Time-Series Dashboard")
st.caption("Astronomy anomaly detection + financial forecasting (with FinBERT sentiment), one shared multi-agent "
           "architecture. Free/open-source by default. Use the sidebar for the full Stock Analysis / Stock "
           "Prediction / Astronomy / Chatbot pages -- this Home view is a reconfigurable overview.")

if "dashboard_config" not in st.session_state:
    st.session_state.dashboard_config = dict(DEFAULT_CONFIG)
config = st.session_state.dashboard_config

with st.sidebar:
    st.header("Adaptive layout")
    pref_text = st.text_input("Describe how you want this dashboard to look",
                               placeholder='e.g. "just show sentiment and RSI for TCS.NS"', key="dash_pref")
    col_a, col_b = st.columns(2)
    apply_clicked = col_a.button("Apply", type="primary")
    reset_clicked = col_b.button("Reset")

    if apply_clicked and pref_text.strip():
        with st.spinner("Interpreting your preference..."):
            st.session_state.dashboard_config = parse_dashboard_preference(pref_text, current_config=config)
        st.rerun()
    if reset_clicked:
        st.session_state.dashboard_config = dict(DEFAULT_CONFIG)
        st.rerun()

    st.caption(f"💡 {config.get('note', '')}")

    st.divider()
    st.subheader("Manual override")
    manual_sections = st.multiselect("Visible sections", KNOWN_SECTIONS, default=config["visible_sections"], key="manual_sections")
    manual_size = st.selectbox("Layout size", ["compact", "normal", "large"],
                               index=["compact", "normal", "large"].index(config["layout_size"]), key="manual_size")
    if manual_sections != config["visible_sections"] or manual_size != config["layout_size"]:
        config["visible_sections"] = manual_sections
        config["layout_size"] = manual_size

    ticker = st.text_input("Focus ticker", value=config.get("focus_ticker", "RELIANCE.NS"), key="focus_ticker_input").upper().strip()
    config["focus_ticker"] = ticker
    astro_target = st.text_input("Astronomy target", value="KIC 11446443", key="focus_target_input")

visible = set(config["visible_sections"])
compact = config["layout_size"] == "compact"
wide = config["layout_size"] == "large"

if not visible:
    st.info("No sections selected -- pick some in the sidebar, or type a preference like \"show everything\".")

sentiment_cache = {}

if "stock_charts" in visible:
    try:
        render_stock_charts(ticker, key_prefix="home_charts", compact=compact)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Stock charts failed for '{ticker}': {exc}")
    st.divider()

if "company_metrics" in visible and not compact:
    try:
        render_company_metrics(ticker, key_prefix="home_metrics")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Company metrics failed for '{ticker}': {exc}")
    st.divider()

if "sentiment" in visible:
    try:
        scored, daily = render_sentiment(ticker, key_prefix="home_sentiment")
        sentiment_cache[ticker] = daily
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sentiment failed for '{ticker}': {exc}")
    st.divider()

if "prediction" in visible:
    try:
        daily = sentiment_cache.get(ticker)
        render_prediction(ticker, sentiment_daily=daily, key_prefix="home_prediction", wide=wide)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed for '{ticker}': {exc}")
    st.divider()

if "astronomy" in visible:
    try:
        render_astronomy(astro_target, key_prefix="home_astro")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Astronomy analysis failed for '{astro_target}': {exc}")

st.divider()
st.caption("Need a full-page deep dive, or a free-form chat? See **Stock Analysis**, **Stock Prediction**, "
           "**Astronomy**, and **Chatbot** in the sidebar.")
