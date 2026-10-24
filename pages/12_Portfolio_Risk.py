"""Portfolio risk: the whole mix, not one stock at a time.

Enter holdings and weights, or pick a model portfolio. The page shows how
much the portfolio moves, where its risk comes from, how it would have done
in real Indian market shocks, and whether its VaR passes a backtest. The
client summary is built from the numbers; an optional AI-written note is a
first draft that is checked number by number before it is shown.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.portfolio_risk import MODEL_PORTFOLIOS, ai_client_note, analyse

BLUE, NAVY, GREY = "#1f5fa8", "#0b2545", "#9aa0a6"
STATUS = {"passed": "✅ passed", "partial": "◐ partial", "failed": "❌ failed"}

st.title("🧺 Portfolio Risk")
st.caption("Volatility, VaR, risk contribution per holding, and stress tests that replay real "
           "Indian market shocks on your portfolio. Weights are held fixed through history.")


@st.cache_data(show_spinner="Analysing the portfolio…", max_entries=16)
def _analyse(items: tuple):
    return analyse(dict(items))


left, right = st.columns([1, 2])
with left:
    preset = st.selectbox("Start from", ["Custom"] + list(MODEL_PORTFOLIOS))
    default = MODEL_PORTFOLIOS.get(preset, MODEL_PORTFOLIOS["Balanced"])
    text = st.text_area("Holdings (one per line: TICKER weight)",
                        "\n".join(f"{t.replace('.NS', '')} {w:g}" for t, w in default.items()),
                        height=220, key=f"holdings_{preset}")
holdings = {}
for line in text.splitlines():
    parts = line.replace(",", " ").split()
    if len(parts) == 2:
        try:
            holdings[parts[0]] = float(parts[1])
        except ValueError:
            pass
if len(holdings) < 2:
    st.info("Enter at least two holdings.")
    st.stop()

try:
    a = _analyse(tuple(sorted(holdings.items())))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with right:
    st.markdown(a["summary"])

m = st.columns(4)
m[0].metric("Annual volatility", f"{a['volatility_pct']:.1f}%",
            delta=f"holdings alone {a['weighted_average_volatility_pct']:.1f}%",
            delta_color="off")
m[1].metric("95% one-day VaR", f"{a['var_95_pct']:.2f}%",
            delta=f"ES {a['es_95_pct']:.2f}%", delta_color="off",
            help="On the worst 1 day in 20 the portfolio loses at least this much.")
m[2].metric("Beta", f"{a['beta']:.2f}",
            help="Moves this many times as much as the market. The market here is an "
                 "equal-weighted index of the liquid NSE stocks in the panel, not the Nifty 50.")
m[3].metric("Diversification ratio", f"{a['diversification_ratio']:.2f}",
            help="Holdings' average volatility divided by the portfolio's. Above 1 means "
                 "holding them together removes risk.")

st.subheader("Where the risk comes from")
h = pd.DataFrame(a["holdings"])
fig = go.Figure()
fig.add_trace(go.Bar(y=h["ticker"], x=h["weight_pct"], name="share of money",
                     orientation="h", marker_color=GREY))
fig.add_trace(go.Bar(y=h["ticker"], x=h["risk_contribution_pct"], name="share of risk",
                     orientation="h", marker_color=BLUE))
fig.update_layout(barmode="group", height=60 + 36 * len(h), margin=dict(l=0, r=10, t=10, b=30),
                  xaxis_title="%", legend=dict(orientation="h", y=1.08, x=0),
                  yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
st.caption("Risk contributions are an Euler allocation: they add up exactly to the portfolio's "
           "volatility. A holding whose risk share is well above its money share is where the "
           "portfolio's risk is concentrated.")
st.dataframe(h.rename(columns={"ticker": "holding", "weight_pct": "weight %",
                               "volatility_pct": "own volatility %",
                               "risk_contribution_pct": "share of risk %"}),
             width="stretch", hide_index=True)

st.subheader("Risk by industry")
sec = pd.DataFrame(a.get("sectors", []))
if len(sec) > 1:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sec["sector"], x=sec["weight_pct"], name="share of money",
                         orientation="h", marker_color=GREY))
    fig.add_trace(go.Bar(y=sec["sector"], x=sec["risk_contribution_pct"], name="share of risk",
                         orientation="h", marker_color=NAVY))
    fig.update_layout(barmode="group", height=60 + 40 * len(sec), margin=dict(l=0, r=10, t=10, b=30),
                      xaxis_title="%", legend=dict(orientation="h", y=1.08, x=0),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Industries from NSE's Nifty 500 list. Because the risk contributions add up exactly, "
               "an industry's share of risk is the sum of its holdings' shares.")
elif len(sec) == 1:
    st.info(f"Every holding is in one industry: {sec.iloc[0]['sector']}. All the risk is concentrated there.")

st.subheader("Stress tests: real shocks replayed")
s = pd.DataFrame(a["stress"])
if not s.empty:
    s["covered_weight"] = (s["covered_weight"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(s.rename(columns={"scenario": "scenario", "portfolio_pct": "portfolio %",
                                   "worst_holding": "worst holding",
                                   "worst_holding_pct": "worst holding %",
                                   "covered_weight": "weight covered"}),
                 width="stretch", hide_index=True)
    st.caption("Holdings not yet listed at the time are left out and the rest re-weighted; "
               "'weight covered' says how much of the portfolio the scenario actually measures.")

bt = a.get("backtest")
if bt:
    st.subheader("Can this VaR be trusted?")
    rows = [{"VaR model": label, "result": STATUS[r["status"]],
             "breached on": f"{r['breach_rate']:.1%} of {r['n_days']} days (target 5%)",
             "Kupiec p": r["kupiec_p"], "Christoffersen p": r["christoffersen_p"]}
            for label, r in (("adjusted for market regime", bt["regime"]),
                             ("single, ignoring regime", bt["unconditional"]))]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.subheader("Client summary")
report = (f"PORTFOLIO RISK SUMMARY ({a['window']['to']})\n\n{a['summary']}\n\nHoldings:\n"
          + "\n".join(f"  {r['ticker']}: weight {r['weight_pct']}%, share of risk "
                      f"{r['risk_contribution_pct']}%"
                      for r in a["holdings"])
          + "\n\nStress tests:\n"
          + "\n".join(f"  {x['scenario']}: {x['portfolio_pct']}%" for x in a["stress"]
                      if x.get("portfolio_pct") is not None)
          + "\n\nThis describes risk, not a recommendation to buy or sell.\n")
st.download_button("Download summary (.txt)", report, file_name="portfolio_risk_summary.txt")
if st.button("Draft an AI client note (checked by the number guard)",
             help="Uses the configured LLM. On a CPU-only local model this takes a few minutes."):
    with st.spinner("Drafting…"):
        try:
            note = ai_client_note(a)
            st.markdown(note["note"])
            if note["flagged_numbers"]:
                st.warning(f"The guard flagged figures not in the computed facts: "
                           f"{', '.join(note['flagged_numbers'])}")
            else:
                st.caption("Every number in the note matches a computed figure.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not reach the LLM: {exc}")
