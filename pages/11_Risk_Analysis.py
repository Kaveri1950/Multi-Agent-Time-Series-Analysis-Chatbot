"""Risk analysis: five models composed into one answer, for a stock and the market.

Follows the shape of the volatility benchmark page -- headline, then the
evidence, then the honest limits -- with one difference that is the point of
this page: the evidence is not a section further down. Every number shown
carries its validation status beside it, because a risk figure that passed a
backtest and one that failed it look identical on a dashboard unless the
dashboard says which is which.

Nothing is computed here. The page renders agents/risk_composition.py, and
the evidence comes from evaluation/results/risk_validation.json.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).parent.parent / "evaluation" / "results"

#: A status is never shown as colour alone: icon and word travel together.
STATUS = {"passed": "✅ passed", "partial": "◐ partial", "failed": "❌ failed",
          "untested": "❔ untested", "descriptive": "· descriptive"}
ROLE_TITLES = {"volatility": "Forward risk — how big will the move be",
               "price_range": "Price range — where the next close is likely to land",
               "tail_risk": "Tail risk — how much could be lost",
               "regime": "Market regime — calm or turbulent, and will it change",
               "anomalies": "Recent unusual days"}
FORECAST, LATEST = "#2a78d6", "#898781"

st.title("🧭 Risk Analysis — Five Models, One Answer")
st.caption("One question fans out to the volatility forecast, the price range it implies, "
           "the risk profile, the market regime and the anomaly detectors. The registry's selector chooses each model; "
           "this page only shows what came back, with the validation status of every number.")


@st.cache_data(show_spinner="Assessing all 50 names — about two minutes…")
def _market(refresh: int, snapshot_mtime: float):
    """The saved assessment, or a fresh one when asked for or when none is saved.

    A full pass over the panel takes about two minutes, which is too long to
    sit behind a page load, so the last result is kept on disk and shown with
    its date. `refresh` only exists to give a recomputation its own cache key,
    and `snapshot_mtime` so that a snapshot rewritten from the command line is
    picked up: without it this process went on serving the copy it had read
    earlier, which lacked a component the page had since learned to show.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from agents.risk_composition import (assess_market_risk, load_market_snapshot,
                                         save_market_snapshot)

    saved = None if refresh else load_market_snapshot()
    if saved is not None:
        return saved
    fresh = assess_market_risk()
    save_market_snapshot(fresh)
    return load_market_snapshot()


@st.cache_data(show_spinner=False)
def _evidence():
    p = RESULTS / "risk_validation.json"
    return json.loads(p.read_text()) if p.exists() else None


def _show(f: dict) -> str:
    v = f["value"]
    if isinstance(v, list):
        return f"{v[0]} to {v[1]}{f['unit']}"
    if f["unit"] == "prob":
        return f"{v:.1%}"
    if f["unit"] == "%":
        return f"{v}%"
    return f"{v} {f['unit']}".strip()


def _figure_table(figures: list) -> None:
    st.dataframe(
        pd.DataFrame([{"measure": f["label"], "value": _show(f),
                       "status": STATUS[f["status"]], "evidence": f["evidence"]}
                      for f in figures]),
        width="stretch", hide_index=True,
        column_config={"evidence": st.column_config.TextColumn(width="large")})


ev = _evidence()
if ev is None:
    st.warning("No validation evidence found. Run `python -m evaluation.risk_validation` "
               "first; without it every figure would be shown as untested.")
    st.stop()

try:
    from agents.risk_composition import MARKET_SNAPSHOT_PATH

    m = _market(st.session_state.get("risk_refresh", 0),
                MARKET_SNAPSHOT_PATH.stat().st_mtime if MARKET_SNAPSHOT_PATH.exists() else 0.0)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not assess the panel: {type(exc).__name__}: {exc}")
    st.stop()

rows = m["ranking"]
if not rows:
    st.error("No name in the panel could be ranked.")
    st.stop()
fig_by_key = {f["key"]: f for f in m["figures"]}
regime = {f["key"]: f for f in (m.get("regime") or {}).get("figures", [])}

# ------------------------------------------------------------ the market now
st.header("1 · The market now")
left, right = st.columns([4, 1])
left.caption(f"Assessment computed {m['generated_at'][:16].replace('T', ' ')} UTC. Shown from "
             f"the saved result; it does not update itself when the price data does.")
if right.button("Recompute", help="Runs all five models on all 50 names. About two minutes."):
    st.session_state["risk_refresh"] = st.session_state.get("risk_refresh", 0) + 1
    st.rerun()
if m["freshness"].get("warning"):
    st.warning(m["freshness"]["warning"], icon="🗓️")

c = st.columns(4)
c[0].metric("Median forecast volatility", f"{fig_by_key['median_forecast_vol_pct']['value']:.1f}%",
            help="Annualised, next session, across the panel.")
c[1].metric("Names forecast more volatile", f"{fig_by_key['share_called_higher']['value']:.0%}",
            help="Than their latest session. Each call is right about 70.9% of the time.")
if regime:
    c[2].metric("Market regime", regime["current_regime"]["value"],
                help="From a hidden Markov model on the market, filtered states.")
    c[3].metric("P(turbulent tomorrow)", f"{regime['p_turbulent_tomorrow']['value']:.1%}",
                help="Read off the fitted chain. Never scored as a forecast.")
st.markdown(m["summary"])
_figure_table([f for f in m["figures"] if f["key"] != "ranking"])
if regime:
    with st.expander("Market regime, in full"):
        _figure_table(m["regime"]["figures"])
        for lim in m["regime"]["limits"]:
            st.caption("Limit: " + lim)

# ---------------------------------------------------------------- the ranking
st.header("2 · Which names are riskiest")
rank_fig = fig_by_key["ranking"]
st.markdown(f"**Ranked by forecast volatility, and only by that — {STATUS[rank_fig['status']]}.** "
            + rank_fig["evidence"])

df = pd.DataFrame(rows)
df["name"] = df["ticker"].str.replace(".NS", "", regex=False)
plot = df.iloc[::-1]                       # highest at the top of a horizontal chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot["forecast_vol_pct"], y=plot["name"], mode="markers", name="forecast, with 80% band",
    marker=dict(color=FORECAST, size=9),
    error_x=dict(type="data", symmetric=False, thickness=2, width=0, color=FORECAST,
                 array=plot["band_80_high"] - plot["forecast_vol_pct"],
                 arrayminus=plot["forecast_vol_pct"] - plot["band_80_low"]),
    customdata=plot[["band_80_low", "band_80_high", "today_vol_pct", "rank"]].to_numpy(),
    hovertemplate="<b>%{y}</b> · rank %{customdata[3]}<br>forecast %{x:.1f}%"
                  "<br>80% band %{customdata[0]:.1f}% to %{customdata[1]:.1f}%"
                  "<br>latest session %{customdata[2]:.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(
    x=plot["today_vol_pct"], y=plot["name"], mode="markers", name="latest session, realised",
    marker=dict(color=LATEST, size=8, symbol="circle-open", line=dict(width=2)),
    hoverinfo="skip"))
fig.update_layout(height=22 * len(plot) + 110, margin=dict(l=0, r=10, t=40, b=40),
                  xaxis_title="annualised volatility (%)", yaxis_title=None,
                  legend=dict(orientation="h", y=1.03, x=0))
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, width="stretch")
st.caption("The bands overlap heavily. That is the honest picture: the forecast separates "
           "the top of the table from the bottom, not one name from the next.")

table = df[["rank", "name", "forecast_vol_pct", "band_80_low", "band_80_high", "band_status",
            "today_vol_pct", "direction", "var_95_now_pct", "var_status", "var_breach_rate",
            "beta", "n_recent_anomalies"]].copy()
table["band_status"] = table["band_status"].map(STATUS)
table["var_status"] = table["var_status"].map(STATUS)
table["var_breach_rate"] = table["var_breach_rate"] * 100
table.columns = ["#", "stock", "forecast vol %", "80% low", "80% high", "band on this stock",
                 "latest vol %", "vs today", "95% VaR %", "VaR backtest", "VaR breach % (5 target)",
                 "beta", "unusual days"]
st.dataframe(table.style.format({
    "forecast vol %": "{:.1f}", "80% low": "{:.1f}", "80% high": "{:.1f}",
    "latest vol %": "{:.1f}", "95% VaR %": "{:.2f}", "VaR breach % (5 target)": "{:.1f}",
    "beta": "{:.2f}"}, na_rep="—"), width="stretch", hide_index=True)

if m["unusual_dates"]:
    with st.expander("Days when many names were flagged at once"):
        st.caption("A day with many names flagged is a market day; a name flagged alone is "
                   "its own story.")
        st.dataframe(pd.DataFrame(m["unusual_dates"]).assign(
            share=lambda d: (d["share"] * 100).round(0).astype(int).astype(str) + "%")
            .rename(columns={"n_flagged": "names flagged", "share": "share of panel"}),
            width="stretch", hide_index=True)
for u in m["unranked"]:
    st.caption(f"Not ranked: {u['ticker']} — {u['why']}")

# ------------------------------------------------------------------ one stock
st.header("3 · One stock, in full")
names = df["name"].tolist()
pick = st.selectbox("Stock", names, index=names.index("TCS") if "TCS" in names else 0,
                    help="Ordered as in the table above, riskiest first.")
a = m["assessments"][pick + ".NS"]
st.markdown(a["summary"])

with st.expander("How each model was chosen"):
    st.caption("Each sub-question went through the registry's selector. Nothing on this "
               "page names a model when deciding what to run.")
    st.dataframe(pd.DataFrame([
        {"sub-question": r["question"], "task": r["task"], "selected": r["model_id"],
         "feasible models": len(r["shortlist"]), "how": r["how"]} for r in a["routing"]]),
        width="stretch", hide_index=True)

for role, title in ROLE_TITLES.items():
    comp = a["components"].get(role)
    st.subheader(title)
    if comp is None:
        st.info("Not in the saved assessment, which predates this component. "
                "Press Recompute at the top of the page.")
        continue
    st.caption(f"`{comp['model_id']}` · as of {comp.get('as_of', 'n/a')}")
    if "error" in comp:
        st.error("Not available: " + comp["error"])
        continue
    _figure_table(comp["figures"])
    days = comp["detail"].get("days")
    if days:
        st.dataframe(pd.DataFrame([
            {"date": d["date"], "severity": d["severity"], "what": ", ".join(d["types"]),
             "why": "; ".join(d["reasons"]),
             "attribution": (d["attribution"] or {}).get("kind", "—")} for d in days]),
            width="stretch", hide_index=True)
    for lim in comp["limits"]:
        st.caption("Limit: " + lim)

# ------------------------------------------------- what the checking found
st.header("4 · What checking the components found")
var = ev["var"]
s, proxy = var["per_ticker_summary"], var["market_proxy_registry_top50"]["conditional_95"]
k = st.columns(3)
k[0].metric("VaR on the market proxy", "passes both",
            delta=f"Kupiec p={proxy['kupiec_p']}, Christoffersen p={proxy['christoffersen_p']}",
            delta_color="off")
k[1].metric("VaR on single stocks", f"{s['conditional']['both_pass']} of {s['n_testable']} pass",
            delta=f"unconditional: {s['unconditional']['both_pass']} of {s['n_testable']}",
            delta_color="off")
v80, v95 = ev["volatility"]["interval"]["80"], ev["volatility"]["interval"]["95"]
k[2].metric("Volatility band coverage", f"{v80['coverage']:.1%} / {v95['coverage']:.1%}",
            delta="nominal 80% / 95%", delta_color="off")
c_, cs, dr = s["conditional"], ev["volatility"]["cross_section"], ev["volatility"]["direction"]
st.markdown(
    f"- **The VaR backtest does not transfer from the market to a stock.** The registry's "
    f"p-values were measured on an equal-weighted proxy. Re-run per stock, the "
    f"regime-conditional VaR is rejected by Kupiec on {s['n_testable'] - c_['kupiec_pass']} "
    f"of {s['n_testable']} names: {c_['kupiec_fail_overstates_loss']} overstate the loss, "
    f"which errs on the safe side, and {c_['kupiec_fail_understates_loss']} understate it. "
    f"Conditioning on the regime still helps — without it only "
    f"{s['unconditional']['both_pass']} pass.\n"
    f"- **The volatility band is calibrated on average, not for every stock.** One width is "
    f"shared by all names, so per-stock coverage of the 80% band runs from "
    f"{v80['per_ticker_min']:.1%} to {v80['per_ticker_max']:.1%}. Misses are lopsided at 95%: "
    f"{v95['above']:.1%} above the band against {v95['below']:.1%} below.\n"
    f"- **The ranking is mostly persistence.** Daily rank correlation "
    f"{cs['spearman_mean']:.3f} for the model against {cs['spearman_trailing66_mean']:.3f} for "
    f"a trailing 66-day average. The model's contribution is "
    f"{cs['gain_over_trailing66']:+.3f}.\n"
    f"- **Direction is measured on one rule and served on another.** The log-space rule "
    f"scores {dr['log_rule']:.1%}; the served model's own field compares a mean and scores "
    f"{dr['served_rule']:.1%}, calling *higher* {dr['share_called_higher_served_rule']:.0%} "
    f"of the time when the truth is {dr['share_actually_higher']:.0%}. This page uses the "
    f"measured rule."
)

# --------------------------------------------------------------------- limits
st.header("5 · Honest limits")
st.markdown(
    "- **One session ahead.** Every forecast here is for the next trading day. The ten-day "
    "VaR and the 21-session regime probability are shown and are untested.\n"
    "- **Size, never direction.** Volatility says how big the move will be. Its correlation "
    "with the next return is +0.016.\n"
    "- **Expected shortfall is not backtested** anywhere in the project. Kupiec and "
    "Christoffersen test VaR only.\n"
    "- **The regime's parameters are fitted on the full sample.** Each day's state uses data "
    "up to that day, but the model that reads it has seen the whole history.\n"
    "- **Fifty Indian large caps, 2019–2026.** One market and one regime mix. Four names have "
    "too little history for any VaR backtest.\n"
    "- **This describes risk and is not advice** on whether to buy or sell anything."
)
st.caption(f"Composed live by agents/risk_composition.py · evidence read from "
           f"{RESULTS.name}/risk_validation.json, generated {ev['generated_at'][:10]}")
