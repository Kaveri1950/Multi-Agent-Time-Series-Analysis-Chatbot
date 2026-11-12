"""Sunspot-number section of the Solar Weather page: predicted vs actual.

Reads what `python -m solar.sunspots --backtest` and `--forecast` saved; it
never fits a model on page load (the LSTM alone takes about a minute).
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solar.sunspots import (HORIZONS, LIVE_PATH, MODELS, PRED_PATH, RESULTS_PATH,
                            load_series, scored_log)

LABELS = {"persistence": "Persistence (tomorrow = today)", "recurrence27": "27-day recurrence",
          "mean27": "27-day mean", "arima": "ARIMA", "xgboost": "XGBoost", "lstm": "LSTM",
          "ensemble": "Ensemble (XGBoost + LSTM)"}
STYLE = {"persistence": dict(color="#94a3b8", dash="dash"),
         "recurrence27": dict(color="#64748b", dash="dot"),
         "mean27": dict(color="#cbd5e1", dash="dashdot"),
         "arima": dict(color="#93c5fd"), "xgboost": dict(color="#2563eb"),
         "lstm": dict(color="#1e3a8a"), "ensemble": dict(color="#0ea5e9")}
ACTUAL = dict(color="#111111", width=1.6)


@st.cache_data(ttl=3600, show_spinner=False)
def _results():
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None


@st.cache_data(ttl=3600, show_spinner=False)
def _predictions():
    return pd.read_parquet(PRED_PATH) if PRED_PATH.exists() else None


@st.cache_data(ttl=3600, show_spinner=False)
def _live():
    return json.loads(LIVE_PATH.read_text()) if LIVE_PATH.exists() else None


@st.cache_data(ttl=3600, show_spinner=False)
def _series():
    return load_series(with_eisn=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _log():
    return scored_log()


def render() -> None:
    st.subheader("🌞 Sunspot number forecast (SILSO daily, 1 to 27 days ahead)")
    st.caption("The official daily sunspot number from SIDC/SILSO (Royal Observatory of Belgium), "
               "1900 onward. Seven forecasters are compared walk-forward from 2010: three one-line "
               "baselines, ARIMA, XGBoost, an LSTM and an XGBoost+LSTM ensemble. Every forecast "
               "uses only data up to its own origin day.")

    res, pred = _results(), _predictions()
    if res is None or pred is None:
        st.info("No sunspot backtest saved yet. Run `python -m solar.sunspots --backtest`.")
        return

    by_h = res["by_horizon"]

    # ---- headline: the measured result, per horizon
    cols = st.columns(len(by_h))
    for c, (h, v) in zip(cols, by_h.items()):
        best = v["best_model"]
        skill = v["scores"][best]["mae_skill_vs_persistence"]
        p = v["dm_best_model_vs_best_baseline"].get("p_value")
        p_txt = f"{p:.3g}" if p is not None else "n/a"
        c.metric(f"{h}-day ahead: best model", LABELS[best].split(" (")[0],
                 (f"{skill:.1%} lower error than persistence" if skill >= 0
                  else f"-{-skill:.1%}: higher error than persistence"),
                 help=f"MAE {v['scores'][best]['mae']:.1f} sunspots against persistence "
                      f"{v['scores']['persistence']['mae']:.1f}. Diebold-Mariano against the best "
                      f"baseline ({LABELS[v['best_baseline']]}): p = {p_txt}")

    tab_bt, tab_day, tab_live, tab_table = st.tabs(["Predicted vs actual (backtest)",
                                                    "Pick a past day", "Next 27 days",
                                                    "All scores"])

    # ---- 1. backtest: predicted vs actual
    with tab_bt:
        c1, c2, c3 = st.columns([1, 2, 1])
        h = c1.radio("Horizon (days ahead)", list(HORIZONS), index=0, horizontal=True,
                     key="ss_h")
        default = ["persistence", by_h[str(h)]["best_model"]]
        default = list(dict.fromkeys(default + ["arima"]))
        chosen = c2.multiselect("Models", list(MODELS), default=default,
                                format_func=lambda m: LABELS[m], key="ss_models")
        span = c3.selectbox("Period", ["Last 12 months", "Last 3 years", "Cycle 25 (2020 on)",
                                       "All (2010 on)"], index=0, key="ss_span")

        p = pred[pred["h"] == h].sort_values("target")
        end = p["target"].max()
        lo = {"Last 12 months": end - pd.DateOffset(years=1),
              "Last 3 years": end - pd.DateOffset(years=3),
              "Cycle 25 (2020 on)": pd.Timestamp("2020-01-01"),
              "All (2010 on)": p["target"].min()}[span]
        p = p[p["target"] >= lo]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p["target"], y=p["actual"], name="Actual (SILSO)",
                                 line=ACTUAL))
        for m in chosen:
            fig.add_trace(go.Scatter(x=p["target"], y=p[m], name=LABELS[m],
                                     line=dict(width=1.2, **STYLE[m])))
        fig.update_layout(height=420, margin=dict(t=20, b=20), hovermode="x unified",
                          yaxis_title="daily sunspot number", legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")

        win = pd.DataFrame({LABELS[m]: [round(float((p[m] - p["actual"]).abs().mean()), 2)]
                            for m in chosen}, index=["MAE in this window"])
        st.dataframe(win, width="stretch")
        st.caption(f"Each point is a forecast made {h} day(s) before the plotted date. "
                   f"{len(p):,} days shown. The last ~4 months of SILSO data are provisional "
                   f"(from {res['data'].get('provisional_from')}) and may be revised slightly.")

        # error by horizon for the chosen models -- how fast skill decays
        fig2 = go.Figure()
        for m in chosen:
            fig2.add_trace(go.Scatter(x=[int(k) for k in by_h], y=[by_h[k]["scores"][m]["mae"] for k in by_h],
                                      name=LABELS[m], mode="lines+markers",
                                      line=dict(width=1.5, **STYLE[m])))
        fig2.update_layout(height=300, margin=dict(t=20, b=20), xaxis_title="days ahead",
                           yaxis_title="MAE (sunspots), 2010 on", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig2, width="stretch")

    # ---- 1b. one past origin day: what each model said then vs what happened
    with tab_day:
        origins = pred.groupby("origin")["h"].nunique()
        origins = origins[origins == len(HORIZONS)].index.sort_values()
        if len(origins) == 0:
            st.info("No origin day has all horizons scored yet.")
        else:
            c1, c2 = st.columns([1, 2])
            day = c1.date_input("Forecast made on", value=origins[-1].date(),
                                min_value=origins[0].date(), max_value=origins[-1].date(),
                                key="ss_day")
            day = origins[origins.searchsorted(pd.Timestamp(day), side="right") - 1]                 if pd.Timestamp(day) >= origins[0] else origins[0]
            pick = c2.multiselect("Models", list(MODELS),
                                  default=["persistence", "arima", "xgboost", "lstm", "ensemble"],
                                  format_func=lambda m: LABELS[m], key="ss_day_models")
            rows = pred[pred["origin"] == day].sort_values("h")
            s = _series()
            seen = s[(s["date"] > day - pd.Timedelta(days=60))
                     & (s["date"] <= rows["target"].max())]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=seen["date"], y=seen["sn"], name="Actual (SILSO)",
                                     line=ACTUAL))
            for m in pick:
                fig.add_trace(go.Scatter(x=rows["target"], y=rows[m], name=LABELS[m],
                                         mode="markers+lines",
                                         line=dict(width=1.2, **STYLE[m]),
                                         marker=dict(size=8)))
            fig.add_vline(x=day, line_color="#475569", line_dash="dot")
            fig.update_layout(height=420, margin=dict(t=20, b=20), hovermode="x unified",
                              yaxis_title="daily sunspot number",
                              legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")

            tbl = pd.DataFrame({"days ahead": rows["h"].values,
                                "target date": rows["target"].dt.date.values,
                                "actual": rows["actual"].round(0).values,
                                **{LABELS[m]: rows[m].round(1).values for m in pick}})
            st.dataframe(tbl, width="stretch", hide_index=True)
            st.caption(f"Forecast issued on {day.date()} using only data up to that day "
                       f"(walk-forward block {int(rows['block'].iloc[0])}). Dots are the 1, 7, 14 "
                       f"and 27-day-ahead forecasts; the black line is what SILSO recorded.")

    # ---- 2. live forecast
    with tab_live:
        live = _live()
        if live is None:
            st.info("No live forecast saved. Run `python -m solar.sunspots --forecast`.")
        else:
            s = _series()
            recent = s[s["date"] > pd.Timestamp(live["origin"]) - pd.Timedelta(days=120)]
            targets = pd.to_datetime(live["targets"])
            show = st.multiselect("Models", list(MODELS),
                                  default=["persistence", "arima", "xgboost", "lstm"],
                                  format_func=lambda m: LABELS[m], key="ss_live_models")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=recent["date"], y=recent["sn"], name="Actual (SILSO + EISN)",
                                     line=ACTUAL))
            for m in show:
                fig.add_trace(go.Scatter(x=[pd.Timestamp(live["origin"])] + list(targets),
                                         y=[live["origin_value"]] + live["paths"][m],
                                         name=LABELS[m], line=dict(width=1.5, **STYLE[m])))
            fig.add_vline(x=pd.Timestamp(live["origin"]), line_color="#475569", line_dash="dot")
            fig.update_layout(height=420, margin=dict(t=20, b=20), hovermode="x unified",
                              yaxis_title="daily sunspot number", legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Issued from {live['origin']} (value {live['origin_value']:.0f}, source: "
                       f"{live['origin_source']}). Values after the dotted line are forecasts; "
                       f"each run is logged so it can be checked against SILSO later.")

            lg = _log()
            if lg is not None and not lg.empty:
                summ = (lg.groupby("model")["abs_err"].agg(["mean", "count"])
                        .rename(columns={"mean": "MAE", "count": "days scored"})
                        .rename(index=LABELS).round(2))
                st.markdown("**Logged live forecasts checked against what happened**")
                st.dataframe(summ, width="stretch")
            else:
                st.caption("No logged forecast has reached its target day yet.")

    # ---- 3. full table
    with tab_table:
        rows = []
        for m in MODELS:
            row = {"model": LABELS[m]}
            for k, v in by_h.items():
                row[f"MAE {k}d"] = v["scores"][m]["mae"]
                row[f"vs persistence {k}d"] = f"{v['scores'][m]['mae_skill_vs_persistence']:+.1%}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        cyc = res.get("mae_by_cycle", {})
        if cyc:
            st.markdown("**MAE by solar cycle**")
            crow = []
            for k, v in cyc.items():
                for label, vals in v.items():
                    crow.append({"horizon": f"{k}d", "period": label.replace("_", " "),
                                 "days": vals["n"], **{LABELS[m]: vals[m] for m in MODELS}})
            st.dataframe(pd.DataFrame(crow), width="stretch", hide_index=True)

        d = res["design"]
        st.caption(f"Walk-forward from {d['test_start']} in {d['n_blocks']} blocks of "
                   f"{d['refit_years']} years; each block is trained only on days whose target "
                   f"precedes it. ARIMA{tuple(d['arima_order'])} on sqrt(SN); LSTM reads "
                   f"{d['lstm_window']} days. Data: {res['data']['source']}, {res['data']['span']}. "
                   f"Reproduce: `python -m solar.sunspots --backtest`.")
