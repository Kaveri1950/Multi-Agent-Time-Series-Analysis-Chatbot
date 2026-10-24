"""Search the NSE corporate filings of the 50 panel stocks by meaning."""

import json
from pathlib import Path

import streamlit as st

from finance.filing_search import INDEX_DIR, search

RESULTS = Path(__file__).parent.parent / "evaluation" / "results" / "filing_search.json"

st.title("🗂️ Filing Search")
st.caption("Search NSE corporate announcements in plain English. The search matches meaning, "
           "so 'a director stepped down' finds filings that say 'cessation'.")

if not (INDEX_DIR / "embeddings.npy").exists():
    st.warning("The search index is not built yet. Run `python -m finance.filing_search --build`.")
    st.stop()

c1, c2, c3 = st.columns([3, 1, 1])
query = c1.text_input("What are you looking for?", "credit rating agency revised the company's rating")
symbol = c2.text_input("Stock (optional)", "").strip() or None
method = c3.selectbox("Method", ["hybrid", "dense", "keyword"],
                      help="hybrid fuses keyword (TF-IDF) and meaning (MiniLM) rankings")
if query:
    with st.spinner("Searching…"):
        res = search(query, k=15, method=method, symbol=symbol)
    if res.empty:
        st.info("No filings found.")
    else:
        res["timestamp"] = res["timestamp"].astype(str).str[:16]
        st.dataframe(res.rename(columns={"timestamp": "filed", "type": "NSE category"}),
                     width="stretch", hide_index=True)

if RESULTS.exists():
    ev = json.loads(RESULTS.read_text())
    with st.expander("How well does it work?"):
        st.markdown(
            f"Tested on {ev['n_queries']} plain-English queries over {ev['index_size']:,} unique "
            f"filings. A result counts as right when its NSE category (assigned by the exchange, "
            f"never indexed) matches the query. A filing in the right category picked at random "
            f"would be right {ev['mean_base_rate']:.1%} of the time.")
        st.table({m: {"precision@10": f"{v['mean_p_at_10']:.1%}", "MRR": f"{v['mean_mrr']:.3f}"}
                  for m, v in ev["methods"].items()})
        st.caption("A lower bound: relevant filings filed under generic categories like "
                   "'Updates' count as misses.")
