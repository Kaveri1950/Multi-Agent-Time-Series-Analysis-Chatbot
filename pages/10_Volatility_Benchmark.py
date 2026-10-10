"""Volatility forecasting: the one target where a model beats its baseline.

Follows the shape of the anomaly benchmark page -- headline result, then the
evidence, then the honest limits -- with the same ordering argument. The
premise is checked BEFORE the leaderboard, because the whole reason this
target exists is that the previous one was unwinnable: if returns and
volatility were equally unpredictable there would be nothing here worth
reading, and that is a claim the reader should be able to check first.

The page deliberately shows the confidence SET rather than a winner. Four
models cannot be statistically separated, and presenting the lowest number
as "the best model" would be a coin flip dressed as a measurement -- the
exact failure the Model Confidence Set exists to prevent.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent.parent / "evaluation" / "results"

st.title("📉 Volatility Benchmark — Where a Model Finally Wins")
st.caption("Fourteen models, one walk-forward protocol, 59,709 strictly causal forecasts "
           "across 48 NSE names. Scored with QLIKE and arbitrated by a Model Confidence Set, "
           "which returns the models that cannot be statistically separated rather than a "
           "single winner.")


@st.cache_data(show_spinner=False)
def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


joint = _load("volatility_joint_parkinson.json")
sweep = _load("volatility_fitting_sweep.json")
nn = _load("volatility_nn.json")

if joint is None:
    st.warning("No volatility results committed yet. Run `python -m quant.garch_carr` first.")
    st.stop()

# ----------------------------------------------------------- the premise
st.header("1 · Why the target changed")
st.markdown(
    "Every price model in this project loses to *tomorrow equals today*. That is the "
    "textbook random-walk result, not a modelling failure — and it is checkable on this "
    "panel directly."
)
c1, c2 = st.columns(2)
c1.metric("Lag-1 autocorrelation, daily returns", "+0.010",
          help="Essentially zero. There is no persistence to forecast.")
c2.metric("Lag-1 autocorrelation, volatility", "+0.459",
          help="Still +0.17 at lag 22. Forty-five times more persistent.")
st.markdown(
    "So the **target** was changed rather than the model. This matters beyond finance: it is "
    "the project's positive control. A framework that only ever reports *nothing works* "
    "cannot be told apart from a broken measuring instrument."
)

# ------------------------------------------------------- the leaderboard
st.header("2 · Fourteen models, one protocol")
rows = [{"model": m, "QLIKE": v["qlike"], "MSE": v["mse"],
         "in confidence set": m in joint["mcs"]["mcs"]}
        for m, v in joint["summary"].items()]
df = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)

st.dataframe(
    df.style.format({"QLIKE": "{:.5f}", "MSE": "{:.6f}"})
      .apply(lambda r: ["background-color: #e8f4ea" if r["in confidence set"] else ""] * len(r),
             axis=1),
    width="stretch", hide_index=True)

best = df.iloc[0]
st.markdown(
    f"**Model Confidence Set (95%): `{'`, `'.join(joint['mcs']['mcs'])}`.** "
    "Four models that cannot be separated statistically. The registry picks `har_ovn` from "
    "that set — not because it has the lowest number, but because the tie is broken on "
    "**prerequisites**: it needs only the OHLC panel already cached, while `har_iv` needs a "
    "live India VIX feed for no measurable gain."
)

with st.expander("What lost, and why it is interesting"):
    st.markdown(
        "- **Gradient boosting** (0.41764) and a **natural cubic spline** (0.41015) both lose. "
        "The spline provably nests linear HAR — it reproduces a straight line to 1.8e-15 — so "
        "it *could* have matched HAR exactly and still forecast worse. That is overfitting, "
        "not a lack of capacity.\n"
        "- **Pooled LSTM** (0.41696) and **BiLSTM** (0.41729). Bidirectionality adds nothing, "
        "as expected: the window ends the day before the target, so the backward pass only "
        "re-reads days already observed.\n"
        "- **GARCH** (0.54608 raw) is partly mis-scaled: it is fitted on close-to-close returns "
        "that include the overnight gap, while the target is an intraday range. A causal "
        "Mincer–Zarnowitz rescaling brings it to 0.51955 — but that recovers only **18.6%** of "
        "the gap, so most of it is genuine.\n"
        "- **CARR** (0.51932) needs no such correction, being fitted on the range itself. Once "
        "GARCH is rescaled the two are effectively tied, so CARR's apparent edge was the units "
        "mismatch it never had."
    )

if nn:
    st.warning(
        "**The ranking depends on the metric, and that should be stated.** On MSE rather than "
        "QLIKE the two neural nets are the *best* models in the set "
        f"(lstm {nn['summary']['lstm']['mse']:.6f} against har {nn['summary']['har']['mse']:.6f}). "
        "QLIKE is the pre-registered primary metric because MSE on a noisy volatility proxy is "
        "dominated by a handful of extreme days — but the conclusion rests on that choice, not "
        "on an unambiguous fact."
    )

# ------------------------------------------------------- fitting scheme
if sweep:
    st.header("3 · The fitting scheme matters more than the model")
    g = pd.DataFrame(sweep["grid"])
    g["confidence set size"] = g["mcs"].apply(len)
    st.dataframe(
        g[["min_train", "refit_every", "har_qlike", "best_model", "best_qlike",
           "confidence set size"]]
        .style.format({"har_qlike": "{:.5f}", "best_qlike": "{:.5f}"}),
        width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.metric("HAR loss spread, window length 250→1000", "0.02826")
    b.metric("HAR loss spread, refit 5→252 days", "0.00121")
    st.markdown(
        "Window length moves the loss **23× more** than refit frequency, and **twice as far** "
        "as the gap between HAR and gradient boosting (0.01387). But it does not dwarf "
        "modelling itself: HAR beats naive by 0.37558, thirteen times the window effect. The "
        "defensible claim is that the window matters more than the choice *among reasonable "
        "models* — not more than having a reasonable model.\n\n"
        "The confidence set also changes **size** with the scheme: seven models at a 250-day "
        "window, four at 500, two at 500/63. With too little training data the test cannot "
        "separate anything, which is a statement about the evaluation rather than the data."
    )

# -------------------------------------------------------------- limits
st.header("4 · Honest limits")
st.markdown(
    "- Volatility is estimated from the daily **high–low range** (Parkinson), not from "
    "5-minute intraday returns. The two strongest published HAR extensions — **HARQ** and "
    "**SHAR** — need realized quarticity and semivariances, which a daily bar cannot form.\n"
    "- 383 of 86,094 ticker-days have `high == low`, where the estimator is **undefined**. "
    "They are dropped, not floored: clipping them to a small positive number made the naive "
    "forecast carry ~0 volatility into the next day and QLIKE, which divides by the forecast, "
    "reached 6.5e6.\n"
    "- One-day horizon only. Published ML gains over HAR grow with horizon, so a one-day "
    "comparison is where those methods look weakest.\n"
    "- All results are on Indian equities, 2019–2026, a single market and a single regime mix."
)
st.caption(f"Read live from {RESULTS.name}/volatility_joint_parkinson.json · "
           f"{joint['n_forecasts']:,} forecasts · generated {joint['generated_at'][:10]}")
