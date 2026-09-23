"""Explainer/Report Agent: retrieves similar past context from the local
vector store (RAG) and asks the local LLM (Ollama, via agents/llm_backend)
to synthesize everything into one natural-language report.

Note on persistence: this builds a small VectorStore fresh each run,
seeded from the current query's own data (recent headlines for finance,
this light curve's own windows for astronomy) -- enough to demonstrate
real retrieval end to end. For the full BTP (TECHNICAL_PLAN.md Week 6),
persist a VectorStore across runs (rag/vector_store.py .save()/.load())
so retrieval finds genuinely *past* cases, not just this run's own data.
"""

from agents.llm_backend import call_llm
from agents.prompts import ASTRONOMY_EXPLAINER_SYSTEM_PROMPT, FINANCE_EXPLAINER_SYSTEM_PROMPT
from agents.state import AgentState
from rag.embed_text import embed_texts
from rag.vector_store import VectorStore


def _instruction_for(focus: str | None, generic: str) -> str:
    if focus:
        return (f'The user specifically asked: "{focus}". Answer THAT question directly in 2-4 sentences, '
                 f"using only the relevant numbers above. Don't recite unrelated metrics they didn't ask about.")
    return generic


def _explain_finance(state: AgentState) -> tuple[str, list]:
    fr = state["finance_result"]
    sr = state.get("sentiment_result", {})
    anomr = state.get("finance_anomaly_result", {})
    focus = state.get("question_focus")

    if fr.get("error") == "insufficient_data":
        prompt = (f"Ticker: {fr['ticker']}\nStatus: {fr['error_message']}\n\n"
                  f"Tell the user plainly, in 1-2 sentences, that you don't have enough historical data for "
                  f"{fr['ticker']} to analyze it, and suggest they try a more actively-traded ticker instead.")
        return prompt, []

    retrieved = []
    headlines = [h["headline"] for h in sr.get("recent_headlines", [])]
    if headlines:
        vecs = embed_texts(headlines)
        store = VectorStore(dim=vecs.shape[1])
        store.add(vecs, [{"kind": "news", "text": h} for h in headlines])
        query_vec = embed_texts([f"{fr['ticker']} price outlook sentiment"])[0]
        retrieved = store.search(query_vec, k=3, kind="news")

    instruction = _instruction_for(
        focus,
        "Explain what these numbers mean in plain language, whether sentiment improved or hurt the "
        "forecast, whether anything unusual was flagged recently, and one caveat a reader should keep in "
        "mind. (4-6 sentences)",
    )

    if anomr:
        n_flagged, n_total = anomr.get("n_anomalous_days", 0), anomr.get("n_days_analyzed", 0)
        top = anomr.get("top_anomalies", [])
        if top:
            anomaly_lines = "\n".join(
                f"  - {a['date']}: {a['severity']} severity (score {a['severity_score']}) -- {', '.join(a['reasons'])}"
                for a in top
            )
            anomaly_block = (
                f"ANOMALY DETECTION (already computed -- {n_flagged}/{n_total} trading days flagged across "
                f"return/volatility/volume/gap{'/ sentiment-divergence' if anomr.get('used_sentiment_divergence') else ''} "
                f"detectors). Most severe flagged days, most severe first (use these dates/reasons verbatim, do not "
                f"invent others or re-rank them yourself):\n{anomaly_lines}"
            )
        else:
            anomaly_block = f"ANOMALY DETECTION (already computed): none of the last {n_total} trading days were flagged as anomalous."
    else:
        anomaly_block = "ANOMALY DETECTION: not available for this query."

    prompt = f"""Ticker: {fr['ticker']}
Latest close: {fr['latest_close']:.2f}
RSI(14): {fr['latest_rsi']:.1f}
MACD: {fr['latest_macd']:.3f}
ARIMA baseline RMSE (no sentiment): {fr['arima_rmse']}
XGBoost RMSE (no sentiment): {fr['xgboost_rmse']}
XGBoost RMSE (with sentiment): {fr.get('xgboost_with_sentiment_rmse')}
News/social sentiment: {sr.get('n_headlines', 0)} items, average score {sr.get('avg_sentiment_score', 0):+.2f} (-1 very negative to +1 very positive)
Most relevant recent headlines (retrieved via local embedding search): {[r['text'] for r in retrieved]}

{anomaly_block}

{instruction}"""

    return prompt, retrieved


def _explain_astronomy(state: AgentState) -> tuple[str, list]:
    ar = state["astronomy_result"]
    focus = state.get("question_focus")

    retrieved = []
    embeddings = ar.get("_embeddings")
    if embeddings is not None and len(ar["top_anomaly_windows"]) > 0:
        top_window_ids = [w["window_id"] for w in ar["top_anomaly_windows"]]
        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, [{"kind": "lightcurve", "window_id": i} for i in range(len(embeddings))])
        query_vec = embeddings[top_window_ids[0]]
        retrieved = store.search(query_vec, k=4, kind="lightcurve")

    instruction = _instruction_for(
        focus,
        "Explain in plain language what was flagged and why it might be a transit-like dip or stellar "
        "variability, whether anything unusual was flagged recently, and one caveat about the small "
        "sample size or proxy labels. (4-6 sentences)",
    )

    top = ar["top_anomaly_windows"][:5]
    if top:
        anomaly_lines = "\n".join(
            f"  - window {w['window_id']}: {w.get('severity_label', 'LOW')} severity -- {', '.join(w.get('reasons', []))}"
            for w in top
        )
        anomaly_block = (
            f"ANOMALY DETECTION (already computed -- {ar['n_flagged_anomalies']}/{ar['n_windows']} windows flagged "
            f"by IsolationForest). Most severe flagged windows, most severe first (use these window ids/reasons "
            f"verbatim, do not invent others or re-rank them yourself):\n{anomaly_lines}"
        )
    else:
        anomaly_block = f"ANOMALY DETECTION (already computed): none of the {ar['n_windows']} windows were flagged as anomalous."

    prompt = f"""Target: {ar['target']}
Data source: {ar['data_source']}
Total windows analyzed: {ar['n_windows']}
Mean autoencoder reconstruction error: {ar.get('mean_reconstruction_error')}

{anomaly_block}

Windows retrieved as embedding-similar to the top anomaly (self-similarity check): window ids {[r.get('window_id') for r in retrieved]}

{instruction}"""

    return prompt, retrieved


def explainer_node(state: AgentState) -> AgentState:
    trace = state.get("trace", [])

    if state["domain"] == "finance":
        prompt, retrieved = _explain_finance(state)
        system_prompt = FINANCE_EXPLAINER_SYSTEM_PROMPT
    else:
        prompt, retrieved = _explain_astronomy(state)
        system_prompt = ASTRONOMY_EXPLAINER_SYSTEM_PROMPT

    report = call_llm(prompt, system=system_prompt)

    trace.append(f"[explainer_agent] generated report ({len(report)} chars), retrieved {len(retrieved)} context items")
    return {**state, "retrieved_context": retrieved, "report": report, "trace": trace}
