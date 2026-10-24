"""Audit view: data-quality controls, their exceptions, and the chatbot answer log."""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from audit.answer_log import DB_PATH

ROOT = Path(__file__).parent.parent
DQ = ROOT / "evaluation" / "results" / "data_quality.json"
EXPORTS = ROOT / "exports" / "audit"

st.title("🧾 Audit")
st.caption("Controls on the data every model depends on, and a record of every chatbot answer. "
           "Problems are reported here, never repaired silently.")

if not DQ.exists():
    st.warning("No data-quality run yet. Run `python -m audit.warehouse`.")
    st.stop()

dq = json.loads(DQ.read_text())
p = dq["panel"]
c = st.columns(4)
c[0].metric("Controls passed", f"{dq['passed']} of {dq['passed'] + dq['failed']}")
c[1].metric("Stocks checked", f"{p['stocks']}")
c[2].metric("OHLC bars checked", f"{p['ohlc_rows']:,}")
c[3].metric("Data runs to", pd.Timestamp(p["to"]).strftime("%d %b %Y"))
st.caption(f"Last run {dq['generated_at'][:16].replace('T', ' ')} UTC.")

st.subheader("Data-quality controls")
rows = [{"result": "❌ fail" if x["status"] == "fail" else "✅ pass", "control": x["check"],
         "rule": x["description"], "severity": x["severity"],
         "failed": f"{x['rows_failed']:,} of {x['rows_tested']:,}",
         "failure rate": f"{x['fail_rate']:.3%}", "limit": f"{x['threshold']:.2%}"}
        for x in sorted(dq["checks"], key=lambda x: (x["status"] != "fail", x["check"]))]
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.subheader("Exceptions to review")
pick = st.selectbox("Control", [x["check"] for x in dq["checks"] if x["examples"]])
ex = next(x for x in dq["checks"] if x["check"] == pick)
st.dataframe(pd.DataFrame(ex["examples"]), width="stretch", hide_index=True)
st.caption("The first few rows that broke the rule. The full lists come from `python -m audit.data_quality`.")

errors = EXPORTS / "routing_errors.csv"
if errors.exists():
    st.subheader("Where the model selector was wrong")
    err = pd.read_csv(errors)
    st.dataframe(err, width="stretch", hide_index=True)
    st.caption(f"The {len(err)} labelled questions the rule-based selector got wrong.")

st.subheader("Chatbot answer log")
if DB_PATH.exists():
    con = sqlite3.connect(DB_PATH)
    try:
        log = pd.read_sql("SELECT asked_at, question, domain, task, model_id, guard_flags, "
                          "guard_figures, seconds, status FROM chat_answers ORDER BY id DESC", con)
    except Exception:  # noqa: BLE001 -- table not created yet
        log = pd.DataFrame()
    con.close()
else:
    log = pd.DataFrame()
if log.empty:
    st.info("No answers logged yet. Every chatbot answer is recorded here.")
else:
    k = st.columns(4)
    k[0].metric("Answers logged", len(log))
    k[1].metric("Corrected by the number guard", int((log["guard_flags"] > 0).sum()))
    k[2].metric("Errors", int((log["status"] == "error").sum()))
    k[3].metric("Median time", f"{log['seconds'].median():.0f} s")
    st.dataframe(log, width="stretch", hide_index=True)
