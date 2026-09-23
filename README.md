# A Multi-Agent System for Time-Series Modelling across Domains

B.Tech Project, IIT Indore. An agent that answers natural-language questions about time
series by **selecting a model from a registry of measured out-of-sample skill**, rather than
asking a language model to choose.

Two domains are covered end to end: **Kepler light curves** and **Indian equities**.

---

## The headline results

| Result | Measurement |
|---|---|
| Model selection beats an LLM | **95.9%** vs **83.7%** on 123 labelled queries, in <1s vs 41 min |
| ...and adding the LLM makes it worse | filter + LLM: 95.1% |
| No detector clears a no-skill floor | 22 detectors, 2 domains, all within **0.04 PR-AUC** |
| In-sample scores lie in proportion to capacity | **1.0x** (parameter-free) to **6.4x** (one-class SVM) |
| Regime-conditional VaR passes coverage tests | Kupiec **p=0.27**; the unconditional model is **rejected at p=0.007** |
| Price forecasting loses to "assume no change" | naive RMSE **36.2** vs XGBoost 50.5, GPR 65.8, ARIMA 123.7 |
| A period search recovers what detectors cannot | **60.9%** of orbital periods across 440 targets |

The central finding is negative and is reported as one. The apparatus that establishes it, non-circular
labels, a dataset audit, held-out scoring and declared reporting budgets, is the contribution.

---

## Layout

```
agents/       routing, model registry, capability filter, serving nodes
anomaly/      domain-neutral detection: contract, 22 detectors, labels, audit, splits
quant/        price panel, point-in-time universe, Markov regimes, VaR
finance/      price data, sentiment, forecasting, per-stock risk profile
astronomy/    light-curve ingestion, KOI catalogue, Box Least Squares period search
solar/        GOES flare catalogue, Hawkes point process, GP classifier
dashboard/    Plotly chart builders, chat persistence
pages/        Streamlit pages
evaluation/   query set, selection benchmark, report figures
report/       mid-semester report (LaTeX + PDF), slides, presentation script
```

---

## Getting started

```bash
python -m venv .venv && .venv/Scripts/activate       # Windows
pip install -r requirements.txt

python -m quant.data                                  # build the 473-ticker panel
python -m astronomy.bulk_download                     # fetch 440 light curves (~358 MB)

streamlit run app.py                                  # dashboard, 9 pages
```

No data is committed. Everything is reproducible from public sources: yfinance, the NASA
Exoplanet Archive, MAST and NOAA SWPC.

### Reproducing the results

```bash
python -m anomaly.grid --kepler-limit 40 --jobs 6     # detector grid, both scoring modes
python -m quant.risk                                  # VaR backtests at 95% and 99%
python -m astronomy.transit_search --limit 25         # BLS period search
python -m evaluation.run_selection                    # routing benchmark
python -m evaluation.report_figures                   # regenerate all figures
```

Results land in `evaluation/results/` as JSON and CSV. Report figures are generated from
those files rather than retyped, so a changed result cannot silently disagree with its figure.

---

## Design notes

**The scoring contract is deliberately narrow.** A detector receives
`windows (n, w)` and returns `scores (n,)`, higher meaning stranger. It is never told what
instrument produced the data or what normal looks like. Anything needing that context would be
solving an easier problem, and its column in the results table would not be comparable.

**Labels never come from the series being scored.** Astronomy labels come from NASA
ephemerides, where a transit falls is predicted by orbital mechanics. Finance labels come from a
pre-registered calendar of events committed before any detector ran. An earlier version labelled
windows by thresholding their own depth and measured **+0.11 AUC** of inflation as a result.

**Solar is excluded from the anomaly comparison** for the strongest version of the same problem:
a GOES flare class is *defined* by an X-ray flux threshold, so detecting flares from X-ray flux
cannot fail. The code remains, as a positive control.

**Both scoring modes are reported.** The difference between in-sample and held-out is itself a
result, and it is how the capacity/inflation relationship was found.

---

## Documentation

- [`report/midsem_report.pdf`](report/midsem_report.pdf) — mid-semester report
- [`report/presentation_script.md`](report/presentation_script.md) — 5-minute talk script
- [`README_original_plan.md`](README_original_plan.md) — the original build plan, kept for reference

---

## Known limitations

- Survivorship is only partly handled: yfinance serves no delisted tickers, so failed companies
  are absent from the panel entirely.
- The finance label set is 15 events. A detector penalised for flagging a real shock missing from
  the list is penalised for being right, so recall reads low by construction.
- Eight test events on the equity held-out window. Those figures are thin.
- Window length is a human choice, verified against the KOI duration distribution but not derived
  from it.
- The astronomy cache and period search are not yet reachable from the chatbot.

---

## Licence

MIT. Data belongs to its original providers: NASA, MAST, NOAA and Yahoo Finance.
