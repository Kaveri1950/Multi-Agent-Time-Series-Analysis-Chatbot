# BTP Month 1 Plan: Foundations & Domain MVPs
### A Multi-Agent Generative AI System for Time-Series Forecasting and Anomaly Detection Across Astronomy and Financial Domains

**Student:** Kaveri Neeli
**Period:** Weeks 1–4, 2026-07-27 to 2026-08-23
**Part of:** a 4-month project plan (this document covers Month 1 only; a full roadmap is available on request)

---

## 1. Objective for This Month

Before any multi-agent orchestration is built, Month 1 establishes two independent, working pipelines —
one for astronomy anomaly detection, one for financial forecasting — each using real, free, public data,
so that later integration work (Months 2–4) builds on a validated foundation rather than untested pieces.

## 2. Resources Used (100% free / open-source)

| Component | Resource |
|---|---|
| Local LLM (for later agent work, set up now) | Ollama, running open-source models locally |
| Astronomy data | NASA Kepler/TESS light curves via the open-source `lightkurve` library (public MAST archive) |
| Financial data | Yahoo Finance (`yfinance`) — historical prices, technical indicators |
| Indian market universe | NSE's public archive (Nifty 50/100/500 constituent lists), Sensex 30 |
| Mutual fund data | AMFI-backed public NAV data via `api.mfapi.in` (~38,000 schemes) |
| Sentiment analysis | FinBERT (HuggingFace, run locally) on free RSS/Reddit headlines |

No paid API or subscription is used at any point in this month's work.

## 3. Weekly Plan

### Week 1 (Jul 27 – Aug 2): Environment & Foundations
- Set up Python environment, install Ollama and pull a local model, install HuggingFace `transformers` and `sentence-transformers`
- Scaffold the project repository with a clear module structure (astronomy / finance / agents / dashboard)
- Literature review: light-curve anomaly detection methods, financial sentiment-driven forecasting, multi-agent LLM architectures (3–5 papers per area)

**Deliverable:** reproducible development environment, versioned repository, one-page literature summary.

### Week 2 (Aug 3 – Aug 9): Data Pipelines
- Astronomy: download and clean a sample batch of real Kepler/TESS light curves
- Finance: build a historical OHLCV + technical-indicator pipeline (RSI, MACD, moving averages)
- Indian markets: pull live Nifty 50/100/500 constituent lists and Sensex 30, plus mutual fund scheme search and historical NAV

**Deliverable:** clean, versioned datasets for both domains, with loader scripts and an exploratory data summary.

### Week 3 (Aug 10 – Aug 16): Astronomy Anomaly Detection v1
- Extract statistical features from light-curve windows (mean, variance, skew, transit-depth proxy)
- Train an IsolationForest baseline anomaly detector
- Evaluate against known transit events as a proxy ground truth (precision, recall, ROC-AUC)

**Deliverable:** a working anomaly detector with a first quantitative evaluation.

### Week 4 (Aug 17 – Aug 23): Financial Prediction v1
- Extend the ARIMA baseline forecasting model
- Add an XGBoost model using technical indicators
- Integrate FinBERT sentiment scoring from free news/social sources, and compare forecast accuracy with vs. without sentiment as a feature

**Deliverable:** a working, sentiment-augmented financial forecasting pipeline with a documented accuracy comparison.

## 4. End-of-Month Checkpoint

By **2026-08-23**, both domain pipelines will run standalone on real data with a first evaluation metric each.
This is the checkpoint I'd propose demonstrating to you before Month 2 (multi-agent integration and the
dashboard interface) begins.

## 5. Looking Ahead (brief context)

This is the first of four planned months. Month 2 connects both pipelines through a shared multi-agent
architecture with a dashboard interface; Month 3 builds a conversational agent on top with safeguards against
incorrect or hallucinated answers; Month 4 is dedicated to evaluation, local-vs-lab benchmarking, and final
reporting. I'm happy to share the complete 4-month roadmap if useful.
