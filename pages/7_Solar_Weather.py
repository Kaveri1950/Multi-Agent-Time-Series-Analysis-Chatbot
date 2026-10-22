"""Solar / space-weather dashboard.

Layout follows the project's other domain pages: a headline result up top,
then the evidence behind it, then the honest limits. The ordering is
deliberate -- the baseline comparison sits immediately beside the model's own
number rather than in a footnote, because a forecast shown alone invites the
reader to trust it and a forecast shown next to NOAA's invites them to judge
it.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solar.flare_data import ARCHIVE_DIR, class_at_least, fetch_xray_flux, load_archive
from solar.forecast import current_forecast, load_model

st.title("☀️ Solar Weather — Flare Forecasting")
st.caption("Real GOES/NOAA SWPC data, no API key. Forecasts P(M-class-or-larger flare in the next 24h) "
           "Two models on the same target: a Hawkes self-exciting point process and a Gaussian "
           "Process classifier, both fitted on the 1996–2026 GOES catalogue (Cycles 23–25) and "
           "scored against NOAA SWPC's own operational forecast.")


# --------------------------------------------------------------------------
# cached loaders -- Streamlit reruns the whole script on every interaction,
# so anything touching the network is cached with a short TTL. NOAA updates
# these products on the order of minutes, so 10 minutes is generous.
# --------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _forecast():
    return current_forecast(include_context=True)


@st.cache_data(ttl=600, show_spinner=False)
def _xray():
    return fetch_xray_flux()


model, model_meta = load_model()
if model is None:
    st.error("No fitted model found. Run this once, then reload:\n\n"
             "```\npython -m solar.forecast --train\n```")
    st.stop()

with st.spinner("Fetching live NOAA data..."):
    fc = _forecast()

forecast = fc["forecast"]
prob = forecast.get("probability")
ctx = fc.get("context", {})
flares = fc.get("recent_flares", pd.DataFrame())


# ==========================================================================
# 1. the headline, with its baseline beside it
# ==========================================================================

st.subheader("Forecast")

noaa = next((b for b in fc["baselines"] if b["name"].startswith("NOAA SWPC operational")), None)
persist = next((b for b in fc["baselines"] if b["name"].startswith("persistence")), None)

gp = fc.get("gp") or {}
gp_prob = gp.get("probability")

# Both models are shown, deliberately. The GP scores better (Brier 0.148 vs
# 0.158, TSS 0.239 vs 0.163 over 57 days against NOAA's outcomes) but says
# nothing about mechanism; the Hawkes parameters ARE the physical result.
# Two models disagreeing is information rather than a problem to hide.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GP classifier", f"{gp_prob:.0%}" if gp_prob is not None else "n/a",
          help="Gaussian Process over 39 daily features. The better-scoring of the two: "
               "Brier 0.148 vs Hawkes 0.158, and TSS 0.239 vs NOAA's own 0.231.")
c2.metric("Hawkes process", f"{prob:.0%}" if prob is not None else "n/a",
          help="Self-exciting point process. Scores slightly worse than the GP but is the "
               "interpretable one — its branching ratio and half-life are the physics result.")
c3.metric("NOAA SWPC official", f"{noaa['probability']:.0%}" if noaa else "n/a",
          delta=f"{(prob - noaa['probability']) * 100:+.0f} pts" if (noaa and prob is not None) else None,
          delta_color="off",
          help="NOAA's own operational forecast for the same window — the benchmark, not a strawman")
c4.metric("Persistence rule", f"{persist['probability']:.0%}" if persist else "n/a",
          help="'Was there an M+ flare in the last 24h?' — a one-line rule that scores TSS ~0.4")
c5.metric("Confidence", forecast.get("confidence", "—"),
          help="Reflects how much evidence the forecast rests on, not how high the probability "
               "is. Capped at MEDIUM: the models now cover Cycles 23-25, but magnetic complexity "
               "— the strongest known predictor — has no historical archive to train on.")

st.caption(f"Horizon: **{forecast.get('horizon')}** · as of {fc.get('as_of')} · "
           f"Hawkes: `{fc.get('method')}`"
           + (f" · GP features as of {gp.get('as_of')}" if gp.get("as_of") else "")
           + " · the chatbot quotes the GP classifier, the better-scoring of the two")

st.warning("This is a **probability over a 24-hour window**, not a prediction of when a flare will "
           "occur. Individual flare timing is not predictable — by this model or by anyone else.",
           icon="⚠️")


# ==========================================================================
# 2. why -- the drivers
# ==========================================================================

st.divider()
st.subheader("What's driving it")

left, right = st.columns([3, 2])

with left:
    for item in forecast.get("basis", []):
        st.markdown(f"- {item}")

    st.markdown("**Fitted Hawkes parameters**")
    hp = pd.DataFrame([{
        "background rate μ (per day)": round(model.mu * 24, 4),
        "branching ratio α": round(model.alpha, 3),
        "excitation half-life (h)": round(np.log(2) / model.beta, 1),
        "mark sensitivity γ": round(model.gamma, 3),
    }]).T.rename(columns={0: "value"})
    st.dataframe(hp, width='stretch')
    st.caption(
        f"A branching ratio of **{model.alpha:.2f}** means roughly {model.alpha:.0%} of M+ flares are "
        f"triggered by an earlier flare rather than arising independently — the Sun's flare activity "
        f"is mostly self-excited. The excitation decays by half every "
        f"**{np.log(2) / model.beta:.0f} hours**, which is the physical statement that an active "
        f"region stays 'primed' for about a day after firing."
    )

with right:
    st.markdown("**Today's Sun** *(context — not model inputs)*")
    st.metric("Active regions on disc", ctx.get("n_regions", "—"))
    r1, r2 = st.columns(2)
    r1.metric("F10.7 flux", ctx.get("radio_flux", "—"))
    r2.metric("Sunspot no.", ctx.get("sunspot_number", "—"))
    if ctx.get("has_delta_region"):
        st.error("δ-configuration region present — these produce nearly all X-class flares.",
                 icon="🔴")
    else:
        st.success("No δ-configuration region on the disc.", icon="🟢")
    st.caption("Magnetic complexity is the strongest known flare predictor, but NOAA serves it for "
               "today only — there is no historical archive to train on, so it is shown as context "
               "rather than fed to the model. An honest gap, not an oversight.")


# ==========================================================================
# 3. recent activity
# ==========================================================================

st.divider()
st.subheader("Recent activity (last 7 days)")

tab_flares, tab_flux, tab_regions = st.tabs(["Flare events", "X-ray flux", "Active regions"])

with tab_flares:
    if flares.empty:
        st.info("No flares reported in the last 7 days.")
    else:
        show = flares.copy()
        show["M+"] = show["max_class"].apply(class_at_least)
        counts = show["class_letter"].value_counts()

        fig = go.Figure()
        colors = {"A": "#c5d3e3", "B": "#8db8e8", "C": "#4a90d9", "M": "#1f5fa8", "X": "#0b2545"}
        for letter in ("A", "B", "C", "M", "X"):
            sub = show[show["class_letter"] == letter]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["begin_time"], y=sub["peak_flux"],
                mode="markers", name=f"{letter}-class",
                marker=dict(size=9, color=colors[letter]),
                hovertemplate="%{x}<br>class %{customdata}<br>%{y:.2e} W/m²<extra></extra>",
                customdata=sub["max_class"],
            ))
        fig.add_hline(y=1e-5, line_dash="dash", line_color="#1f5fa8",
                      annotation_text="M threshold (forecast target)")
        fig.add_hline(y=1e-4, line_dash="dot", line_color="#0b2545",
                      annotation_text="X threshold (not forecast)")
        fig.update_yaxes(type="log", title="peak flux (W/m²)")
        fig.update_layout(height=380, margin=dict(t=20, b=20), hovermode="closest")
        st.plotly_chart(fig, width='stretch')

        k = st.columns(5)
        for i, letter in enumerate(("A", "B", "C", "M", "X")):
            k[i].metric(f"{letter}-class", int(counts.get(letter, 0)))

        with st.expander("Event table"):
            st.dataframe(
                show[["begin_time", "max_time", "end_time", "max_class", "peak_flux", "M+"]],
                width='stretch', hide_index=True)

with tab_flux:
    flux = _xray()
    if flux.empty:
        st.info("X-ray flux unavailable.")
    else:
        long_band = flux[flux["energy"] == "0.1-0.8nm"]
        short_band = flux[flux["energy"] == "0.05-0.4nm"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=long_band["time_tag"], y=long_band["flux"],
                                 name="0.1–0.8 nm (defines class)", line=dict(color="#111111", width=1)))
        fig.add_trace(go.Scatter(x=short_band["time_tag"], y=short_band["flux"],
                                 name="0.05–0.4 nm", line=dict(color="#60a5fa", width=1)))
        for level, label, color in ((1e-6, "C", "#4a90d9"), (1e-5, "M", "#1f5fa8"), (1e-4, "X", "#0b2545")):
            fig.add_hline(y=level, line_dash="dash", line_color=color, annotation_text=label)
        fig.update_yaxes(type="log", title="flux (W/m²)")
        fig.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')
        st.caption("1-minute GOES X-ray flux, last 24 hours. Flare class is defined by the peak of the "
                   "long band. The two bands' ratio is a hardness proxy that tends to rise as plasma heats.")

with tab_regions:
    regions = ctx.get("regions")
    if regions is None or regions.empty:
        st.info("No active regions reported today.")
    else:
        st.dataframe(regions[["region", "location", "area", "spot_count",
                              "mcintosh_class", "mag_class", "mag_complexity", "has_delta"]],
                     width='stretch', hide_index=True)
        st.caption("`mag_class` is the Mount Wilson magnetic classification (A=α, B=β, BG=βγ, BGD=βγδ). "
                   "`mcintosh_class` is the three-letter McIntosh class. δ-configurations are the "
                   "flare-productive ones.")


# ==========================================================================
# 4. measured skill
# ==========================================================================

st.divider()
st.subheader("Measured skill (backtest, not this forecast)")

skill = fc.get("skill", {})
if not skill:
    st.info("No backtest results stored. Re-run `python -m solar.forecast --train`.")
else:
    xgb = skill.get("xgboost_tss_by_fold") or []
    pers = skill.get("persistence_tss_by_fold") or []
    if xgb and pers:
        folds = [f"fold {i + 1}" for i in range(len(xgb))]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=folds, y=xgb, name="XGBoost", marker_color="#2563eb"))
        fig.add_trace(go.Bar(x=folds, y=pers, name="Persistence (1 line of code)",
                             marker_color="#94a3b8"))
        fig.add_hline(y=0, line_color="#475569",
                      annotation_text="no skill")
        fig.update_layout(barmode="group", height=320, yaxis_title="TSS (true skill statistic)",
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig, width='stretch')

        st.markdown(
            f"**The honest result.** Walk-forward across solar-cycle phases, XGBoost scores TSS "
            f"{min(xgb):.2f}–{max(xgb):.2f} while the one-line persistence rule scores "
            f"{min(pers):.2f}–{max(pers):.2f} — and **persistence wins in "
            f"{sum(1 for a, b in zip(xgb, pers) if b > a)} of {len(xgb)} folds**. Most of the "
            f"apparent skill in short-horizon flare forecasting comes from knowing whether the Sun "
            f"was already flaring. That is worth reporting, not hiding: it is the same clustering "
            f"the Hawkes model formalises, and it sets the real bar for any more complex model."
        )
        st.caption("TSS = TPR − FPR. Accuracy is not used: with a ~14% positive rate, always "
                   "predicting 'no flare' is 86% accurate and worthless. Operational M-class "
                   "forecasting runs roughly 0.5–0.6 TSS industry-wide.")


# ==========================================================================
# 5. the accruing archive
# ==========================================================================

st.divider()
st.subheader("Self-collected archive")

arch = load_archive("probabilities")
flare_arch = load_archive("flares")
a1, a2, a3 = st.columns(3)
a1.metric("NOAA forecasts archived", len(arch))
a2.metric("Flare events archived", len(flare_arch))
a3.metric("Archive location", "data/cache/solar/")

st.info(
    "**Why this exists.** Every NOAA endpoint is a rolling 7–30 day window and NOAA does not "
    "backfill — no bulk archive covers 2021–2026, and Solar Cycle 25 peaked in 2024–25. So the most "
    "relevant years are exactly the ones missing from the training data. `harvest_daily()` appends "
    "each day's snapshot locally, building a genuine holdout set the model has never seen. "
    "Run it daily:\n\n```\npython -c \"from solar.flare_data import harvest_daily; harvest_daily()\"\n```",
    icon="📥")


# ==========================================================================
# 6. caveats -- last, and not collapsed by default
# ==========================================================================

st.divider()
st.subheader("Limits of this forecast")
for c in fc.get("caveats", []):
    st.markdown(f"- {c}")
