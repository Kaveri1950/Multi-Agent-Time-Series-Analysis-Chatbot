# Multi-Agent Time-Series Analysis

B.Tech Project, IIT Indore (Aug 2026 to present). Guide: Amit Shukla

A Streamlit app with a chatbot that answers questions about three kinds of time series: Indian stock prices, Kepler star light curves, and solar flares. You ask something like "any unusual days in RELIANCE?" or "how risky is TCS?", and the app works out what kind of question it is, picks a model, runs it, and explains the result.

The main idea is in how the model is picked. Every model is listed in a registry along with the inputs it needs and the score it got on held-out data. A question is mapped to a task (forecast, anomaly, risk, lookup, and so on), models that can't run on the available inputs are removed, and the one with the best measured score is used. A local LLM (Llama 3 through Ollama) writes the final answer from the computed numbers, and in the chatbot it also reads the question to pick out the stock or star name. It doesn't choose the model and doesn't calculate anything.

## What's in the app

| Page | What it does |
|---|---|
| Chatbot | Multi-turn chat across all three domains. You can also upload an image of a light-curve plot |
| Stock Analysis | Price charts, company metrics and unusual days for any ticker |
| Stock Prediction | ARIMA, XGBoost and XGBoost with FinBERT news sentiment, then the volatility forecast |
| Volatility Benchmark | Why the forecasting target was changed from price to volatility, and the model comparison |
| Stock Risk | Volatility, VaR/ES, beta, drawdown, market regime, and unusual days split into market vs company moves |
| Risk Analysis | Market risk today, 50 stocks ranked by forecast volatility, and each number marked passed / failed / untested |
| Market Screener | Nifty 50/100/500 or Sensex 30 ranked by 1-day return and RSI |
| Mutual Funds | Search about 38,000 Indian schemes; NAV history, returns and an ARIMA forecast |
| Anomaly Benchmark | The detector comparison on both datasets, with a live demo |
| Astronomy | Kepler light curve, anomaly detection and plain-language reasons |
| Solar Weather | Chance of an M-class or larger flare in the next 24 h, next to NOAA's own forecast |
| Home | A dashboard you can rearrange by typing what you want |

## How a question is answered

```
question
  -> task classifier (ordered regex rules)             agents/model_selector.py
  -> capability filter + rank by held-out score        agents/model_registry.py  (22 models, 11 tasks)
  -> LangGraph agents: router, finance, sentiment,     agents/graph.py
     anomaly, astronomy, solar, explainer
  -> LLM writes the answer from the computed facts     agents/llm_backend.py
  -> number check: any figure not in the facts is flagged   agents/answer_style.py
```

Questions like "what is RSI?" don't run any model. They are answered directly.

## Results

All of these are on held-out or walk-forward data. Result files are in `evaluation/results/`.

### 1. Picking the model: rules vs LLM

141 labelled questions (11 canonical, 102 paraphrases, 18 tricky near-misses, 10 that no model should answer).

| Method | Correct | Time per question |
|---|---|---|
| Rule-based filter | **95.0%** (134/141) | under 0.01 s |
| Filter, then LLM re-ranks the shortlist | 80.9% | 22.6 s |
| LLM picks from the whole registry | 65.2% | 26.7 s |

On the 18 tricky questions the filter got 77.8% and the LLM alone 33.3%. Adding the LLM on top of the filter made it worse, not better.

Limitation: I wrote both the rules and the questions, so 95% is an optimistic number. A fresh question set written by someone else would be a fairer test.

Earlier version: on the first 123-question set the numbers were 95.9% / 95.1% / 83.7% (`selection_all.json`). Some expected answers in that set pointed to an old model. After fixing the labels and adding risk and volatility questions, the current numbers are the ones above.

Reproduce: `python -m evaluation.run_selection --strategies filter_only`

### 2. Anomaly detection on two very different datasets

18 detector settings from 6 families (simple thresholds, rolling rules, density/distance, reconstruction, matrix profile, HMM/GARCH), each with one fixed setting. GARCH is equity-only, so Kepler has 17. Each is compared with two baselines that learn nothing (raw values and an untrained autoencoder) and with a supervised model given the labels.

| | Kepler | Indian equities |
|---|---|---|
| Data | 40 stars, 384,281 windows (12 h windows, 3 h stride) | 50-stock index, 1,911 days |
| Labels | Transit times from NASA ephemerides | A fixed list of market events, written before running any detector |
| Test split | By star | By time |
| Test positives | 29,168 of 124,782 (23.4%) | 8 of 568 (1.4%) |
| Baseline PR-AUC | 0.2528 | 0.0325 |
| Best detector | matrix profile, 0.2630 | spectral PCA, 0.0355 |

The best detectors beat "learn nothing" by only a small margin. On equities the margin is not significant (block-bootstrap 95% CI -0.040 to +0.137), because there are only 8 positive test windows.

Scores on training data overstate skill a lot: one-class SVM looks 6.4x better in-sample than on held-out data, while matrix profile is about the same (0.9x).

An LSTM autoencoder was also tried on Kepler only: PR-AUC 0.2591, just above the baseline.

The labels never come from the series being scored. An earlier version labelled transits by their own depth, which inflated the median AUC from 0.88 to 0.98 on 25 stars, an inflation of about 0.10 (`python -m astronomy.evaluate --inflation`).

### 3. Price forecasting doesn't work, so the target changed

On a 120-day walk-forward log for 3 stocks, nothing beats "tomorrow = today":

| Model | RMSE | Direction correct |
|---|---|---|
| Naive (no change) | 30.21 | n/a |
| ARIMA | 30.10 | 47.5% |
| Gaussian Process | 36.83 | 44.2% |

Daily returns have almost no autocorrelation (about +0.01), but volatility does (about +0.5). So the forecasting target became tomorrow's volatility (Garman-Klass).

### 4. Volatility forecasting

A pooled ensemble (ridge, random forest, extra trees, gradient boosting, SVR, MLP) trained in log space on 50 NSE stocks, walk-forward.

| Measure | Result |
|---|---|
| Says correctly whether volatility goes up or down | **70.9%** |
| Best possible, estimated from a Kalman fit of measurement noise | 74.1% |
| Beats each stock's own 60-day average | 50 of 50 stocks |
| Coverage of the 80% / 95% bands | 80.6% / 94.8% |

On the QLIKE loss a simple per-stock HAR model is slightly better (0.335 vs 0.353), so the "best model" depends on the metric.

### 5. Market regimes and Value-at-Risk

- Turbulence: Mahalanobis distance of daily returns from the past year's normal pattern.
- Regimes: a 2-state Gaussian HMM (quiet vs turbulent). Only forward-filtered states are used, since Viterbi states use future data.
- On the pre-registered event days, the market is in the turbulent state 47% of the time vs 10% on other days.

VaR backtest on the equal-weighted market proxy (walk-forward, data to 21 Sep 2026):

| Model | 95%: Kupiec p | 95%: Christoffersen p | 99%: Kupiec p |
|---|---|---|---|
| Unconditional | 0.0068 (rejected) | 0.563 | 0.0005 (rejected) |
| Regime-conditional | **0.273** (pass) | 0.237 (pass) | **0.861** (pass) |

The exact p-values move with new data: after the 9 Oct refresh they are 0.831 and 0.229, and on that data the pass holds under all 10 HMM random seeds. Tested stock by stock, only 12 of 46 stocks pass both tests, so the Risk Analysis page shows each stock's own result.

10-day VaR from the Markov chain is -4.89%, against -6.56% from the square-root-of-time rule.

### 6. Astronomy and solar

- A Lomb-Scargle check sorts series into periodic or one-off: 9 of 10 Kepler stars come out periodic and 10 of 10 stocks one-off.
- Box Least Squares period search is implemented (`astronomy/transit_search.py`) but doesn't have a committed result file yet.
- Solar flares: a Hawkes process (each flare raises the short-term chance of another) gets TSS 0.49 on the last test block. On a recent 61-day hindcast, NOAA's own forecast and simple persistence both did better.

## Running it

Python 3.12. Ollama with `llama3` is needed for chatbot answers. MongoDB is optional; it's used for chat history and the forecast log.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Data comes from free sources (yfinance, NSE, api.mfapi.in, NASA Exoplanet Archive, NOAA) and is cached in `data/cache/`, which is not in the repo. Settings are read from environment variables: `BTP_LLM_BACKEND` (`ollama`, `openai` or `claude`), `BTP_OLLAMA_MODEL`, `MONGODB_URI`.

Tests: `pytest -m "not slow"` (61 tests pass; some need internet). The slow tests call the LLM.

## Folder layout

```
agents/       registry, model selector, LangGraph nodes, risk composition, answer checks
anomaly/      detector interface, detectors, splits, metrics, data audit, labels
quant/        turbulence, HMM regimes, VaR and backtests, volatility models
finance/      price data, forecasting, sentiment, risk profile, screener, mutual funds
astronomy/    Kepler loading, KOI labels, transit search
solar/        NOAA data, flare features, Hawkes model
evaluation/   question set, experiments, results/
rag/          sentence embeddings + FAISS for the explainer
pages/ views/ dashboard/   Streamlit UI
tests/
```

## Known limitations

- The question set is small and I wrote it myself.
- The equity anomaly test has only 8 positive windows.
- The detector comparison uses 40 of the 440 downloaded Kepler stars.
- With the local LLM on CPU, a full chatbot answer takes about 5 to 7 minutes.
- Volatility jumps on results days are under-forecast. Calendar features help a little, but they aren't in the served model yet.
- Dependencies are not pinned, and there is no CI yet.

## Licence

MIT. Data belongs to its sources (NASA Exoplanet Archive, NOAA SWPC, NSE, AMFI, Yahoo Finance) and is not redistributed here.
