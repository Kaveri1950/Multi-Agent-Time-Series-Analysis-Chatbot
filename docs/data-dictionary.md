# Data dictionary

Every dataset the project reads or writes, with its source, columns, refresh command and known issues. All raw data is cached under `data/cache/` (not committed; rebuilt by the commands below). Committed outputs are under `evaluation/results/` and `exports/audit/`.

Sizes and counts are from the cache as of 24 Oct 2026.

## Market data (Indian equities)

| Dataset | Path | Source | Shape / grain | Columns | Refresh |
|---|---|---|---|---|---|
| Close prices | `data/cache/quant/price_panel.parquet` | Yahoo Finance (yfinance), Nifty 500 constituents | 1,921 days × 473 stocks, 2019-01-01 to 2026-10-09; one row per trading day, one column per ticker (`TCS.NS` …) | adjusted close | `python -m quant.data` |
| Volume | `data/cache/quant/volume_panel.parquet` | Yahoo Finance | same grid as close prices | shares traded | `python -m quant.data` |
| Panel metadata | `data/cache/quant/panel_meta.json` | written by the downloader | one record | index, start, downloaded_at, n_requested (501), n_kept (473), first/last date | with the panel |
| OHLC bars | `data/cache/quant/ohlc_panel.parquet` | Yahoo Finance, top 50 by traded value | 86,000 rows; one row per stock per day (long format) | `date`, `ticker`, `open`, `high`, `low`, `close` | `quant.volatility.download_ohlc()` |
| India VIX | `data/cache/quant/india_vix.parquet` | Yahoo Finance (`^INDIAVIX`) | 1,901 days | `vix` (index: date) | with the OHLC refresh |
| Index constituents | `data/cache/nifty{50,100,500}_constituents.json` | NSE index CSVs (archives.nseindia.com) | one list per index | symbols | `finance.india_universe` (re-fetched on use) |

**Known issues** (from the latest `python -m audit.data_quality` run, `evaluation/results/data_quality.json`):
- **Flat bars:** 134 of 86,000 (0.16%, limit 0.10%) have high = low, mostly circuit-limit days such as Yes Bank in 2020. **Failed.**
- **Zero volume:** 1,204 price points (0.15%, limit 0.10%) have a price but zero reported volume. **Failed.**
- **Freshness:** the latest bar is 9 Oct 2026, more than 10 days old. **Failed** until the next refresh.
- **Extreme moves:** 17 one-day moves beyond ±40%, under the review threshold. They include corporate actions the free data didn't adjust for (for example a −95% Patanjali move on relisting, and −75% moves on 1 Jan 2024).
- **Missing days:** 91 gaps inside listed history; 41 of them are one stock (FORCEMOT).
- Close prices are split-adjusted by Yahoo, but not always correctly (see above). The panel uses today's constituents, so there is survivorship bias; `quant.data.point_in_time_universe` reduces it for backtests.

## Company news and filings

| Dataset | Path | Source | Shape / grain | Columns | Refresh |
|---|---|---|---|---|---|
| NSE corporate announcements | `data/cache/news/nse_announcements/<SYMBOL>.parquet` | NSE API `/api/corporate-announcements` (needs a session cookie from the NSE homepage) | 60,905 rows across 50 stocks; one row per filing | `timestamp`, `type` (NSE's category, 195 distinct), `text`, `company` | `finance.news_history.build(symbols, refresh=True)` |
| NSE event calendar | `data/cache/news/nse_events/<SYMBOL>.parquet` | NSE API `/api/event-calendar` | one row per scheduled board meeting | `date`, `purpose`, `description` | same |
| Times of India headlines | `data/cache/news/india-news-headlines.csv` | Harvard Dataverse "India News Headlines" | about 3.3M headlines | `publish_date`, `headline_category`, `headline_text` | downloaded once |
| Company headlines, scored | `data/cache/news/toi_company_scored.parquet` | TOI headlines matched to company names, FinBERT tone | 2,451 rows | `date`, `category`, `headline`, `symbol`, `tone` | `finance.news_history.score_headlines` |
| Filing search index | `data/cache/filing_index/` | built from the announcements | 34,657 unique filings (deduplicated on company + text); embeddings 34,657 × 384 | `timestamp`, `type`, `text`, `company`, `symbol`; `embeddings.npy` (MiniLM, unit-normalised) | `python -m finance.filing_search --build` |

**Known issues:** 43% of filings are exact duplicates of another filing by the same company, mostly boilerplate. Many material filings sit under generic categories ("Updates", "Press Release"), which is why the filing-search score is a lower bound. GDELT tone is optional because the API rate-limits.

## Astronomy and solar

| Dataset | Path | Source | Shape / grain | Columns | Refresh |
|---|---|---|---|---|---|
| Kepler Objects of Interest | `data/cache/astronomy/koi_cumulative.parquet` | NASA Exoplanet Archive (TAP) | 9,564 KOIs | `kepid`, `kepoi_name`, `kepler_name`, `koi_disposition`, `koi_period`, `koi_time0bk`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_score` | `astronomy.koi_catalog` |
| Kepler light curves | `data/cache/lightcurves_bulk/KIC_<id>.parquet` | NASA MAST, via lightkurve | 441 stars, 358 MB; one row per 29.4-min cadence | `time` (BKJD days), `flux` | `python -m astronomy.bulk_download` |
| Solar flare history | `data/cache/solar/history/flares_<year>.parquet` | NASA/GSFC GOES event listings | one row per flare | `begin_time`, `peak_time`, `end_time`, `flare_class`, `class_letter`, `peak_flux`, `duration_min`, `position`, `active_region` | `python -m solar.flare_data` |
| Solar daily indices | `data/cache/solar/archive/daily_indices.parquet` | NOAA SWPC | one row per day | radio flux, sunspot number/area, flare counts by class … | `python -m solar.flare_data` |
| Solar forecast scores | `data/cache/solar/forecast_scores.parquet` | this project vs NOAA | 61 days | observed M/X flares, `noaa_m_prob`, `hawkes_prob`, `persistence_prob`, `gp_prob` … | `python -m solar.forecast_log` |

**Known issues:** the anomaly grid uses 40 of the 441 downloaded stars. The solar hindcast covers only 61 days.

## Outputs

| Dataset | Path | What it holds | Refresh |
|---|---|---|---|
| Forecast log | `data/cache/finance/forecast_log.parquet` | 120 logged price forecasts with naive, ARIMA and GP predictions and the actual close | the forecast pages, then `finance.forecast_log` |
| Audit database | `data/cache/audit/audit.db` (SQLite) | tables `data_quality_checks` (appended each run), `model_registry`, `routing_eval`, `var_backtests`, `chat_answers` (one row per chatbot answer: time, question, domain, task, model, guard corrections, seconds, status) | `python -m audit.warehouse`; `chat_answers` fills live |
| Audit query exports | `exports/audit/*.csv` | the result of each query in `audit/queries.sql` | `python -m audit.warehouse` |
| Evaluation results | `evaluation/results/*.json` | one file per experiment; every number in the README comes from one of these | the script named at the top of each `evaluation/*.py` |

## Conventions

- Tickers use Yahoo's NSE suffix (`TCS.NS`); filings and events use the bare NSE symbol (`TCS`).
- Dates are exchange-local (IST) calendar dates; filing timestamps are IST to the second.
- Returns are simple daily returns unless a module says log returns. Volatility figures are annualised with 252 trading days.
- Nothing in the pipeline overwrites raw data to "fix" it. Quality problems are reported by `audit.data_quality` and left for a person to decide on.
