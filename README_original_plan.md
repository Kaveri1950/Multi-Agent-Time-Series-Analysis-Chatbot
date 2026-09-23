# Multi-Agent Generative AI System for Time-Series Forecasting & Anomaly Detection
### Astronomy (light curves) + Financial Markets (price + sentiment) — BTP Project

See [`TECHNICAL_PLAN.md`](TECHNICAL_PLAN.md) for the full 3-month build plan and architecture rationale,
and [`PROFESSOR_PROPOSAL.md`](PROFESSOR_PROPOSAL.md) for the professor-facing proposal.

## What's here

A working, 100% free/open-source-by-default implementation:

| Module | What it does | Key libraries |
|---|---|---|
| `finance/price_data.py` | Yahoo Finance OHLCV + technical indicators (RSI, MACD, SMA/EMA) | `yfinance` |
| `finance/sentiment.py` | Free RSS (Yahoo/Google News) + optional Reddit headlines → local FinBERT sentiment scoring | `feedparser`, `praw`, `transformers` |
| `finance/price_model.py` | ARIMA baseline vs. XGBoost (technical + sentiment features) — the key with/without-sentiment RMSE comparison | `statsmodels`, `xgboost` |
| `finance/anomaly_detector.py` | Multi-type, severity-scored anomaly detection on daily price/volume/sentiment: return outliers, volatility spikes, volume spikes, overnight gaps, and sentiment-price divergence — each with rolling z-score thresholds that adapt to the ticker's own regime | `numpy`, `pandas` |
| `astronomy/data_loader.py` | Real Kepler light curves via `lightkurve` (MAST archive), synthetic fallback for offline dev | `lightkurve` |
| `astronomy/anomaly_detector.py` | Statistical-feature IsolationForest baseline anomaly detector | `scikit-learn` |
| `astronomy/ts_embedder.py` | **Custom-trained** 1D-CNN autoencoder → light-curve embedding + reconstruction-error anomaly score | `torch` |
| `astronomy/image_lightcurve.py` | Extracts a light curve from a plot image (JPG/PNG) via pixel-column tracing, so the chatbot can analyze a screenshotted light curve | `Pillow` |
| `astronomy/dataset.py` | Downloads and caches a real batch of Kepler light curves + precomputes anomaly detection for each, so the chatbot can answer dataset-level questions ("which target has the most anomalies") from an index instead of a live fetch | `lightkurve`, `pyarrow` |
| `rag/embed_text.py` | Free local text embeddings (MiniLM) — no paid embedding API | `sentence-transformers` |
| `rag/vector_store.py` | Local FAISS vector store for retrieval | `faiss-cpu` |
| `agents/llm_backend.py` | One `call_llm()` function; swap `backend="ollama"` (free/local, dev) → `"openai"`/`"claude"` (paid, demo only) with zero code changes elsewhere | `ollama`, `langgraph` |
| `agents/graph.py` | The multi-agent system: Router → {Finance→Sentiment \| Astronomy} → Explainer, as a LangGraph state graph | `langgraph` |
| `agents/chat_agent.py` | Chatbot intent layer: free-form message → domain/ticker-or-target/question focus, multi-turn context + a per-session pipeline cache so follow-ups reuse fetched data instead of re-running everything; also the unified anomaly-detection command surface (mode pin + domain-agnostic anomaly-intent detection) over both domains' specialized detectors | — |
| `agents/anomaly_shared.py` | Shared presentation contract (`AnomalySummary`) both domains' anomaly results normalize into — one command surface, two fully specialized detectors underneath, not one generalized model | — |
| `agents/dashboard_agent.py` | Adaptive-layout agent: typed preference ("just show sentiment and RSI for TSLA") → which dashboard sections/size to render | — |
| `dashboard/charts.py` | Plotly chart builders (candlestick, RSI/MACD/MA overlays, prediction comparison, light-curve anomaly shading, NAV chart) | `plotly` |
| `dashboard/sections.py` | One render function per dashboard section, shared between the Home page and the dedicated pages | `streamlit` |
| `finance/india_universe.py` | Nifty 50/100/500 constituents (NSE's free public archive CSVs, cached) + Sensex 30 (hardcoded snapshot) + index-level tickers | `requests` |
| `finance/mutual_funds.py` | Indian mutual fund scheme search + real historical NAV (`api.mfapi.in`, free, ~38k schemes) + trailing returns + ARIMA forecast | `requests` |
| `finance/ticker_search.py` | Company name → ticker resolution (Yahoo's free search endpoint) with real disambiguation ("reliance" → asks Reliance Industries vs Reliance Infrastructure) instead of silently guessing | `requests` |
| `agents/prompts.py` | The strict system prompts (data-grounding rules, static-vs-live distinction, SEBI disclaimer, no cross-domain leakage) shared by every finance/astronomy-answering call site | — |
| `app.py` + `pages/*.py` | Multi-page Streamlit UI: adaptive Home dashboard, Stock Analysis, Stock Prediction, Astronomy, Market Screener, Mutual Funds, Chatbot | `streamlit` |
| `benchmarks/benchmark.py` | Laptop-vs-lab performance benchmark (LLM latency, embedding throughput, autoencoder training time) | — |

### Pages

- **Home** (`app.py`) — adaptive overview dashboard. Type a preference in the sidebar (e.g. "just show sentiment and RSI for TSLA", "hide astronomy", "make the prediction chart bigger") and an LLM call reconfigures which sections/size render; manual multiselect/size toggles are available as a fallback or reset.
- **Stock Analysis** (`pages/1_Stock_Analysis.py`) — candlestick/line chart, RSI/MACD/moving averages, company financial metrics + CAPM, plus a multi-type anomaly detection section (see `finance/anomaly_detector.py`) marking flagged days on the price chart with the actual detector reasons on hover. Ported from the original stock-prediction app's look and feel. Works for any yfinance ticker, including Indian ones (`RELIANCE.NS`, `TCS.BO`, ...).
- **Stock Prediction** (`pages/2_Stock_Prediction.py`) — ARIMA vs. XGBoost vs. XGBoost+FinBERT-sentiment forecast comparison, with the RMSE delta front and center.
- **Astronomy** (`pages/3_Astronomy.py`) — Kepler/TESS target ID or CSV upload, flux-vs-time chart with anomalous windows shaded, autoencoder reconstruction-error trace, plus a "downloaded dataset overview" section (see `astronomy/dataset.py`) once you've run a batch download.
- **Market Screener** (`pages/5_Market_Screener.py`) — pick Nifty 50 / 100 / 500 or Sensex 30, ranks constituents by 1-day return and RSI(14). Nifty 500 is capped to a configurable number of stocks for CPU-only responsiveness (explicitly stated on-screen, not silently truncated).
- **Mutual Funds** (`pages/6_Mutual_Funds.py`) — search any of ~38,000 Indian mutual fund schemes by name, NAV history chart, trailing 1M/6M/1Y/3Y returns, ARIMA forecast. No sentiment/XGBoost augmentation here by design (see `finance/mutual_funds.py`'s docstring) — a diversified fund's NAV doesn't have the same single-company news-sentiment signal a stock does.
- **Chatbot** (`pages/4_Chatbot.py`) — free-form multi-turn chat covering stocks, Indian indices (Nifty 50/100/500, Sensex), mutual funds, astronomy targets, and general finance/astronomy questions, with the agent trace and RAG-retrieved context visible per turn. Ambiguous company names ("reliance") trigger a real disambiguation prompt; genuine price-forecast requests ("what will the price be in 30 days") run an actual forward-looking ARIMA projection and render an inline chart, not just text. Attach a light-curve plot image (📎 button) to ask about it directly, or ask dataset-level questions ("which of my targets has the most anomalies") once a batch has been downloaded via `python -m astronomy.dataset`.

### Finance anomaly detection

`finance/anomaly_detector.py` runs after `sentiment_node` in the graph (`finance_node -> sentiment_node ->
finance_anomaly_node -> explainer_node`) and fuses five independent detectors into one severity-scored,
TYPED anomaly per trading day (not a single binary flag): return outliers, volatility spikes, volume
spikes, overnight gap events, and sentiment-price divergence (sentiment strongly one direction while price
is flat or moves opposite — only possible because this project already runs a parallel FinBERT sentiment
pipeline on the same ticker/date axis). Each detector is a rolling z-score against the series' own recent
history, so what counts as "anomalous" adapts as the ticker's volatility regime shifts, rather than using
one fixed global threshold. A day's severity (`LOW`/`MEDIUM`/`HIGH`) reflects how many detectors fired
together and how extreme the underlying z-scores/ratios were, and the explainer prompt is handed the
already-computed top anomalies with their concrete reasons (the same "computed facts, not model-derived"
pattern used to fix the light-curve dataset ranking bug below) so the LLM phrases them rather than
re-deriving or re-ranking them itself.

**Real-world validation, not just a proxy label**: run against RELIANCE.NS, the detector correctly flags
2024-06-04 — India's Lok Sabha election-result day, when Nifty/Sensex crashed intraday on the
exit-poll-vs-actual mismatch — as a HIGH-severity anomaly (return z-score -4.50, plus a 2.8x volume spike).
This is a genuine known event, not a proxy ground truth invented to make the detector look good, and it's
now a regression test (`test_finance_anomaly_detector`) so future changes can't silently break it.

### One chatbot, two specialized anomaly detectors

Deliberately *not* one generic "time-series anomaly model" shared across domains — flux windows and daily
price/volume series don't share enough statistical structure for a single algorithm to be strong at both;
forcing that would trade real detection power for a simplistic story. Instead, `agents/anomaly_shared.py`
defines a shared **presentation contract** (`AnomalySummary`: domain, key, which detector methods ran,
counts, and a normalized `top_anomalies` list with label/severity/score/reasons) that both domains'
already-separate, fully-specialized detectors get normalized into — finance keeps its 5 rolling-z-score
detectors, astronomy keeps IsolationForest + the autoencoder's reconstruction error (now also carrying a
`severity_label`/`reasons` per window, added so both sides can be explained the same way). One interface,
two specialized implementations underneath — a single chatbot command surface, not a single weak model.

On top of that shared contract, `agents/chat_agent.py` adds:
- **A domain-agnostic anomaly-intent detector** (`_ANOMALY_QUERY_PATTERN`) so "any anomalies in RELIANCE.NS"
  and "anything unusual in KIC 8866102" both work as the same natural phrasing, routed to whichever domain's
  detector applies. Found and fixed a real regex bug while building this: `\banomal\b` (word boundary
  immediately after the "anomal" stem) can never match "anomaly" or "anomalies" -- there's no boundary
  between "l" and the "y"/"ies" that always follows it in real English, so the literal word the feature
  exists to detect was silently never matched until switched to `anomal\w*`.
- **An explicit mode pin** ("switch to astronomy mode") so a bare, entity-free follow-up ("anything
  unusual lately?") has somewhere to resolve to besides re-guessing from the previous turn alone.
- **A shared anomaly chart builder** that reuses whichever detector already ran (`finance_anomaly_chart` or
  `lightcurve_anomaly_chart`) from data cached on the AgentState (`_scored_df`/`_features_df` for finance,
  `_flux`/`_time`/`_anomaly_mask` for astronomy) -- no re-fetch, no re-detection for a follow-up question.

### Building the light curve dataset

```bash
python -m astronomy.dataset
```

Downloads and caches ~20 real Kepler targets (`data/cache/lightcurves/*.parquet`), runs the same
IsolationForest + autoencoder pipeline on each, and writes a summary index
(`data/cache/lightcurve_dataset_index.json`) the chatbot and the Astronomy page both read from. Safe to
re-run (cached targets are skipped unless `force_refetch=True`); extend `astronomy/dataset.DEFAULT_TARGETS`
to scale toward the 100–300 targets `TECHNICAL_PLAN.md` Week 2 called for. In one real run, 18 of 20
targets fetched genuine MAST/Kepler data; the other 2 fell back to a clearly-labeled synthetic curve
(one MAST timeout, one invalid KIC ID) — the dataset index records which is which.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# then:
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) installed locally with at least one model pulled, e.g.:
```bash
ollama pull llama3.2:3b
```

## Run

```bash
# CLI, one query:
python -m agents.graph AAPL
python -m agents.graph "KIC 11446443"

# Full UI:
streamlit run app.py

# Benchmark (run once now, again on the lab machine later):
python benchmarks/benchmark.py --tag laptop-cpu
```

## Design notes / what to scale up for the full BTP

- **Astronomy data**: currently a handful of sample Kepler targets (`astronomy/data_loader.SAMPLE_TARGETS`). Scale to 100–300 targets per `TECHNICAL_PLAN.md` Week 2 once the pipeline is validated.
- **RAG persistence**: `explainer_node` currently builds a fresh, small vector store from each run's own data (real retrieval, but no cross-run memory yet). Persist a `VectorStore` across runs (`.save()`/`.load()`) once you have a real corpus of past reports to retrieve against.
- **LLM model size**: `llama3:latest`/`mistral:latest` (7B) take ~100s/response on CPU-only hardware — too slow for interactive multi-agent chains. Default dev model is `llama3.2:3b` (much faster); swap to the 7B models or the lab GPU for quality comparisons.
- **Autoencoder discrimination**: the reconstruction-error anomaly signal is real but modest at this small sample size — the classical IsolationForest baseline currently outperforms it. This is an honest, reportable finding, not a bug to "fix" by tuning until it looks better.
- **Small-model intent parsing**: `llama3.2:3b` is unreliable at a few things a bigger/cloud model would likely get right on the first try -- e.g. it misclassified a pronoun follow-up ("what about its sentiment") as an unrelated general question, and separately failed to extract an explicitly-named ticker or a "make it bigger" resize request into its own JSON output. It also silently auto-completed an ambiguous company name ("reliance") into a specific ticker inside its own intent JSON, which would have quietly bypassed disambiguation entirely if not caught -- `agents/chat_agent.py` now always re-derives the ticker from the bare name via a real search unless the user's own raw text contained an explicit exchange suffix, specifically so the model's autocompletion can't paper over genuine ambiguity. Both `agents/chat_agent.py` and `agents/dashboard_agent.py` have small keyword/regex safety nets layered on top of the LLM call for these reasons -- worth mentioning in the report as a concrete small-model-vs-cloud-model tradeoff, and worth re-testing once you swap to the 7B models or a cloud API for the final demo.
- **Strict data-grounding system prompts** (`agents/prompts.py`): tested against real data with adversarial questions the model isn't equipped to answer. Results were genuinely mixed, which is itself a useful report finding:
  - Astronomy prompt: performed very well. Correctly answered static/conceptual questions ("what is a light curve"), used only real retrieved numbers for target-specific questions, and -- notably -- correctly refused to confirm a planet detection from a flagged anomaly, explicitly naming what evidence (dip shape, depth, repeat count) would be needed first.
  - Finance prompt: partial compliance. It never hallucinated a live number it didn't have, and correctly attached the SEBI-advisor disclaimer to an investment-suggestion question -- but it incorrectly refused a static/historical question ("what happened in the Yes Bank crisis," which the prompt explicitly says to answer from general knowledge) as if it needed retrieved data, and in the investment-suggestion answer it gave two wrong ticker symbols (`HCL.NS`/`WIPR.NS` instead of the real `HCLTECH.NS`/`WIPRO.NS`) despite the rule against fabricating tickers.
  - Neither failure mode has a clean regex fix the way the ticker/resize ones did (they require actual language understanding of "is this question historical/definitional" and "is this the correct real ticker symbol"), so they're left as documented small-model limitations rather than papered over -- a legitimate discussion point for the report on why the final demo swaps to a larger model.
- **Small-model numerical comparison** (`agents/chat_agent.py`'s dataset-query path): asked "which of my targets has the most anomalies" over the real 20-target dataset, `llama3.2:3b` picked a target with 84 flagged windows while a different target actually had 98 -- it couldn't reliably scan and compare ~20 rows itself. Unlike the two findings above, this one DOES have a clean fix: `_answer_dataset_query` now precomputes the true highest-rate and highest-count targets deterministically in Python and hands them to the model as labeled "COMPUTED FACTS" instead of asking it to derive them, the same pattern already used for ticker/resize detection. Worth citing in the report as a general principle: push any comparison/aggregation a small local model does unreliably into code, and give the model the already-correct answer to phrase rather than to compute.
