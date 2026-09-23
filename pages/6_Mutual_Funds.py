import streamlit as st

from dashboard.sections import render_mutual_fund_analysis
from finance.mutual_funds import search_schemes

st.title("💰 Mutual Funds (India)")
st.caption("Search any of ~38,000 Indian mutual fund schemes by name (data via api.mfapi.in, free, no key, "
           "backed by AMFI's official daily NAV filings). ARIMA forecast only -- see the page footer for why "
           "sentiment/XGBoost augmentation isn't offered here.")

query = st.text_input("Search scheme name", "SBI Large Cap", key="mf_query",
                       help='Try "HDFC Mid-Cap", "ICICI Prudential Bluechip", "Axis Small Cap", "SBI Large Cap"')

if query:
    with st.spinner("Searching schemes..."):
        matches = search_schemes(query, limit=20)

    if not matches:
        st.info("No schemes matched. Try a shorter/different name fragment.")
    else:
        options = {f"{m['schemeName']} (code {m['schemeCode']})": m["schemeCode"] for m in matches}
        chosen_label = st.selectbox(f"{len(matches)} match(es) -- pick one", list(options.keys()), key="mf_pick")
        scheme_code = options[chosen_label]

        try:
            render_mutual_fund_analysis(scheme_code, key_prefix="mf")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't analyze scheme {scheme_code}: {exc}")

st.divider()
st.caption("Why no sentiment analysis here: a diversified mutual fund's NAV reflects dozens/hundreds of "
           "underlying holdings, so single-company-style news sentiment doesn't map onto it the way it does "
           "for one stock. See finance/mutual_funds.py for the full reasoning.")
