"""Risk profile for a single stock: how much could this lose, and why.

Answers the question anomaly detection cannot. "TCS moved 8% on these days"
is description the user already has; "on a rough day you would be down about
3%, markets are calm so that is the number in force, and of last year's
unusual days eight were about the company rather than the market" is an
assessment.

The page shows the technical numbers -- it is a dashboard, read by someone
who wants them. The CHATBOT gets the same facts translated into plain
language, because a person asking "is TCS risky" does not want to be told
about excess kurtosis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.title("⚖️ Stock Risk Profile")
st.caption("Volatility, downside estimates, drawdown and regime-conditional risk for one "
           "stock — with its unusual days attributed to the market or to the company.")


@st.cache_data(show_spinner=False)
def _tickers():
    from quant.data import load_panel

    close, volume = load_panel()
    tv = (close * volume).rolling(21, min_periods=10).mean()
    return list(tv.mean().sort_values(ascending=False).index)


@st.cache_data(show_spinner="Building risk profile…", max_entries=16)
def _profile(ticker: str, days: int):
    from finance.risk_profile import risk_profile

    return risk_profile(ticker, lookback_days=days)


try:
    tickers = _tickers()
except Exception as exc:  # noqa: BLE001
    st.error(f"No cached price panel found. Run `python -m quant.data` first. ({exc})")
    st.stop()

c1, c2 = st.columns([3, 2])
with c1:
    ticker = st.selectbox("Stock", tickers, index=tickers.index("RELIANCE.NS")
                          if "RELIANCE.NS" in tickers else 0,
                          help=f"{len(tickers)} tickers, ordered by traded value.")
with c2:
    days = st.select_slider("Look back", [180, 365, 730, 1095, 2000], value=365,
                            format_func=lambda d: f"{d/365:.0f} year" + ("s" if d > 400 else ""))

try:
    p = _profile(ticker, days)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not build a profile for {ticker}: {exc}")
    st.stop()

lv, s, t, d = p["level"], p["systematic"], p["tail"], p["drawdown"]
r, a = p.get("regime", {}), p["anomalies"]

# ==========================================================================
# 1. headline
# ==========================================================================
m = st.columns(4)
m[0].metric("Annual volatility", f"{lv['annual_volatility_pct']:.1f}%",
            help="How much it swings in a typical year.")
m[1].metric("95% one-day VaR", f"{t['var_95_pct']:.2f}%",
            delta=f"ES {t['expected_shortfall_95_pct']:.2f}%", delta_color="off",
            help="On about one day in twenty it falls at least this much. Expected "
                 "shortfall is the average loss on those days.")
m[2].metric("Beta", f"{s['beta']:.2f}",
            delta=f"{s['variance_from_market_pct']:.0f}% from market", delta_color="off",
            help="Low beta is NOT low risk: it only means the MARKET explains little of "
                 "the movement. The rest is company-specific and does not diversify away.")
m[3].metric("Max drawdown", f"{d['max_drawdown_pct']:.1f}%",
            delta="not recovered" if d["still_underwater"] else f"{d['recovery_days']}d to recover",
            delta_color="inverse" if d["still_underwater"] else "off")

if r:
    st.info(
        f"**Markets are {r['current_regime']} right now.** Chance of turning choppy tomorrow: "
        f"{r['p_turbulent_tomorrow']:.1%}; within a month, {r['p_turbulent_in_21d']:.1%}. "
        f"This stock's 95% VaR is **{r['var_95_quiet_pct']:.2f}%** in a quiet regime and "
        f"**{r['var_95_turbulent_pct']:.2f}%** in a turbulent one — **{r['turbulent_multiplier']:.1f}× wider**. "
        f"An unconditional VaR averages the two and is rejected by a Kupiec coverage test; "
        f"the regime-conditional version passes.",
        icon="📊",
    )

# ==========================================================================
# 2. where the unusual days came from
# ==========================================================================
st.divider()
st.subheader("Unusual days, attributed")

st.markdown(
    "A large move is only news if the market did not make it. Stripping out the market "
    "(via beta) leaves the **idiosyncratic** part — what happened to *this company*. "
    "The two lists genuinely differ: a stock can fall 6% on a day the whole market fell 6%, "
    "which is not a company event at all."
)

k = st.columns(3)
k[0].metric("Unusual days", a["n_flagged"])
k[1].metric("Market-wide", a["n_market_wide"], help="The index moved; this stock followed.")
k[2].metric("Company-specific", a["n_company_specific"],
            help="The move was not explained by the market.")

if a["events"]:
    ev = pd.DataFrame(a["events"])
    fig = go.Figure()
    for kind, colour in (("market-wide", "#9aa0a6"), ("company-specific", "#0b2545")):
        sub = ev[ev["kind"] == kind]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["date"], y=sub["return_pct"], name=kind, marker_color=colour,
            customdata=np.stack([sub["market_pct"], sub["idiosyncratic_pct"]], axis=-1),
            hovertemplate="%{x}<br>total %{y:.2f}%<br>market %{customdata[0]:.2f}%"
                          "<br>company %{customdata[1]:.2f}%<extra></extra>",
        ))
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=40, b=30),
                      plot_bgcolor="white", paper_bgcolor="white", barmode="group",
                      title="Largest unusual days — navy is the company, grey is the market",
                      yaxis_title="return (%)",
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True)

    show = ev[["date", "return_pct", "market_pct", "idiosyncratic_pct", "kind"]].copy()
    show.columns = ["date", "total %", "market %", "company %", "attribution"]
    st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander("News around these dates"):
        st.caption("Headlines that appeared NEAR each date — not necessarily the cause. "
                   "Free RSS reaches back only five to nine months, so older events are "
                   "marked unavailable rather than guessed at.")
        for e in a["events"]:
            if e["kind"] != "company-specific":
                continue
            st.markdown(f"**{e['date']}** ({e['return_pct']:+.2f}%)")
            if e.get("news"):
                for h in e["news"]:
                    st.markdown(f"- {h}")
            else:
                st.caption(f"  {e.get('news_status', 'no news attached')}")

# ==========================================================================
# 3. the detail
# ==========================================================================
st.divider()
with st.expander("Full numbers"):
    left, right = st.columns(2)
    with left:
        st.markdown("**Tail**")
        st.dataframe(pd.DataFrame([
            {"measure": "95% VaR", "value": f"{t['var_95_pct']:.2f}%"},
            {"measure": "95% expected shortfall", "value": f"{t['expected_shortfall_95_pct']:.2f}%"},
            {"measure": "99% VaR", "value": f"{t['var_99_pct']:.2f}%"},
            {"measure": "worst day", "value": f"{t['worst_day_pct']:.2f}%"},
            {"measure": "skew", "value": f"{t['skew']:.2f}"},
            {"measure": "excess kurtosis", "value": f"{t['excess_kurtosis']:.2f}"},
            {"measure": "Sortino", "value": f"{t['sortino']:.2f}"},
        ]), use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Systematic vs idiosyncratic**")
        st.dataframe(pd.DataFrame([
            {"measure": "beta", "value": f"{s['beta']:.3f}"},
            {"measure": "R²", "value": f"{s['r_squared']:.3f}"},
            {"measure": "variance from market", "value": f"{s['variance_from_market_pct']:.1f}%"},
            {"measure": "variance company-specific", "value": f"{s['variance_idiosyncratic_pct']:.1f}%"},
            {"measure": "idiosyncratic vol (annual)", "value": f"{s['idio_vol_annual_pct']:.2f}%"},
        ]), use_container_width=True, hide_index=True)
    if r and "var_95_10d_markov_pct" in r:
        st.markdown("**Multi-day, through the Markov chain**")
        st.caption("The √t rule assumes returns are independent. Volatility clustering is "
                   "precisely the statement that they are not, so the chain propagates the "
                   "regime forward instead.")
        st.dataframe(pd.DataFrame([
            {"horizon": f"{h} days",
             "Markov VaR": f"{r.get(f'var_95_{h}d_markov_pct', float('nan')):.2f}%",
             "√t rule": f"{r.get(f'var_95_{h}d_sqrt_rule_pct', float('nan')):.2f}%"}
            for h in (5, 10) if f"var_95_{h}d_markov_pct" in r
        ]), use_container_width=True, hide_index=True)

st.warning(
    "**This describes risk, not advice.** Whether to hold a stock depends on your horizon, "
    "what else you own and your own tolerance — none of which this sees. Historical tails "
    "also only describe shocks that have already happened.",
    icon="⚠️",
)
