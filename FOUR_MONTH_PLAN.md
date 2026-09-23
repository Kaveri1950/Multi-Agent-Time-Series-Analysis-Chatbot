# Four-Month Plan: Multi-Agent Generative AI System for Time-Series Forecasting & Anomaly Detection
### Astronomy (Kepler light curves) + Indian Financial Markets (NSE/BSE stocks, indices, mutual funds)

**Student:** Kaveri Neeli · **Start:** 2026-07-27 · **Duration:** 4 months (16 weeks)

This supersedes the original 3-month plan (`TECHNICAL_PLAN.md`) — the project's scope has grown to include
full Indian-market coverage (Nifty 50/100/500, Sensex, mutual funds), an adaptive multi-page dashboard, and a
conversational agent with real ticker disambiguation and forward-looking price forecasting, which the original
3-month scope did not account for. Everything below stays on the same 100% free/open-source stack (Ollama,
HuggingFace, yfinance, lightkurve, NSE/AMFI public data) established in the original plan.

---

## Month 1 (Weeks 1–4): Foundations & Domain MVPs
Build the two domains as independent, working pipelines before any agent orchestration.

- **Week 1:** Environment setup (Python venv, Ollama + local LLM, HuggingFace `transformers`/`sentence-transformers`), repo scaffolding, literature review (light-curve anomaly detection, financial sentiment forecasting, multi-agent LLM frameworks).
- **Week 2:** Data pipelines — real Kepler/TESS light curves via `lightkurve`; Yahoo Finance OHLCV + technical indicators; Indian market universe (Nifty 50/100/500 via NSE's public archive, Sensex 30) and mutual fund NAV data (`api.mfapi.in`, ~38,000 schemes).
- **Week 3:** Astronomy anomaly detection v1 — statistical-feature IsolationForest baseline, evaluated against proxy transit labels.
- **Week 4:** Financial prediction v1 — ARIMA baseline vs. XGBoost, FinBERT sentiment scoring from free RSS/Reddit sources, with/without-sentiment RMSE comparison.

**Deliverable:** two standalone, working domain pipelines (astronomy + finance), each with real data and a first evaluation metric.

## Month 2 (Weeks 5–8): Multi-Agent System, RAG & Dashboard
Connect the two domains through a shared agent architecture and give them a real interface.

- **Week 5:** Custom embeddings — free local text embeddings (MiniLM) for news/reports; a self-trained 1D-CNN autoencoder for light-curve windows (no pretrained equivalent exists for this, unlike text).
- **Week 6:** Local FAISS vector store + retrieval; LangGraph multi-agent graph (Router → Finance/Sentiment or Astronomy → Explainer), all backed by a local Ollama LLM with a one-line swap to a paid API for the eventual demo.
- **Week 7:** Multi-page Streamlit dashboard — Stock Analysis (candlestick/RSI/MACD/CAPM), Stock Prediction (ARIMA vs. XGBoost vs. XGBoost+sentiment), Astronomy (flux chart with shaded anomalies).
- **Week 8:** Adaptive Home dashboard (a typed preference reconfigures which sections/layout render) + two India-specific pages: Market Screener (Nifty/Sensex constituents ranked by return/RSI) and Mutual Funds (scheme search, NAV history, ARIMA forecast).

**Deliverable:** a working multi-agent system with a full dashboard UI, covering both domains and the Indian market universe end to end.

## Month 3 (Weeks 9–12): Conversational Intelligence & Robustness
Turn the dashboard's backend into a genuinely usable chatbot, then find and fix its failure modes.

- **Week 9:** Chatbot intent layer — free-form multi-turn chat, domain/entity/question-focus extraction, per-session caching so follow-up questions reuse fetched data instead of re-running the whole pipeline.
- **Week 10:** Real-world ticker resolution — a plain company name (e.g. "reliance") is checked against a live search and, when genuinely ambiguous (e.g. Reliance Industries vs. the unrelated Reliance Infrastructure), the chatbot asks the user to choose rather than guessing; a genuine forward-looking price forecast (not backtesting) with a confidence interval and an inline chart in the chat.
- **Week 11:** Strict data-grounding system prompts for both domains — no hallucinated numbers, a clear line between static/definitional questions (answerable from general knowledge) and live/numeric ones (must come from retrieved data only), a SEBI-advisor disclaimer on investment suggestions, and no cross-domain leakage (astronomy context bleeding into a finance answer or vice versa).
- **Week 12:** Hardening pass — full regression suite (`pytest` + Streamlit's `AppTest` framework, which actually executes the app rather than just checking it starts) across every page and agent path; document where the small local model (3B parameters) still falls short of what a larger/cloud model would get right.

**Deliverable:** a chatbot that handles ambiguous input, gives real forecasts with charts, and is guarded against hallucination — with a documented test suite proving it.

## Month 4 (Weeks 13–16): Evaluation, Benchmarking & Demo Readiness
Turn the working system into a defensible BTP report and a rehearsed live demo.

- **Week 13:** Local-vs-lab-vs-cloud performance benchmarking — identical workloads on the laptop CPU, the lab's parallel-computing system, and a cloud LLM API, answering the open question from the original proposal about which setup suits deployment.
- **Week 14:** Full quantitative evaluation — anomaly-detection precision/recall/AUC, price-prediction RMSE with vs. without sentiment, embedding-quality visualization (t-SNE/UMAP), and a structured audit of chatbot answer accuracy against the strict-prompt rules.
- **Week 15:** RAG persistence (retrieval across sessions, not just within one run) and final polish; cloud-API demo path finalized and tested end to end.
- **Week 16:** BTP report writing, slide deck, and full live-demo rehearsal on the cloud-API path.

**Deliverable:** final BTP report, evaluation results, and a rehearsed live demonstration.

---

## Milestone Check-ins
| Milestone | Target date | Demo to professor |
|---|---|---|
| M1: Domain MVPs | ~2026-08-23 | Standalone astronomy + finance pipelines, each with an evaluation metric |
| M2: Multi-agent dashboard | ~2026-09-20 | Full dashboard covering both domains + Indian markets |
| M3: Robust chatbot | ~2026-10-18 | Chatbot demo: disambiguation, real forecast+chart, hallucination-guard walkthrough |
| M4: Final report + demo | ~2026-11-15 | Full evaluation results, live demo, final report |
