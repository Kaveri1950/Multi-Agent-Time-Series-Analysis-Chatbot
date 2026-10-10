"""One render function per dashboard section, shared between the adaptive
Home page (app.py) and the dedicated deep-dive pages (pages/*.py) so the
UI logic lives in exactly one place. Each function does its own data
fetch/compute + st.* rendering; `key_prefix` keeps Streamlit widget/element
keys unique when the same section is rendered from two different pages in
the same session.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from agents.llm_backend import call_llm
from agents.prompts import ASTRONOMY_EXPLAINER_SYSTEM_PROMPT, FINANCE_EXPLAINER_SYSTEM_PROMPT
from astronomy.anomaly_detector import (
    STRIDE, WINDOW_SIZE, annotate_windows, extract_features, fit_isolation_forest, make_windows, score_anomalies,
)
from astronomy.data_loader import fetch_light_curve, generate_synthetic_light_curve
from astronomy.ts_embedder import reconstruction_error, train_autoencoder
from dashboard.charts import (
    candlestick_chart, close_line_chart, finance_anomaly_chart, lightcurve_anomaly_chart, macd_chart,
    moving_average_chart, nav_chart, normalized_lightcurve_chart, plotly_table, prediction_comparison_chart,
    raw_lightcurve_chart, reconstruction_error_chart, rmse_comparison_bar, rsi_chart,
)
from finance.anomaly_detector import detect_all, top_anomalies
from finance.india_universe import fetch_index_constituents, get_index_ticker
from finance.mutual_funds import compute_returns, forecast_nav, get_nav_history, get_scheme_meta, search_schemes
from finance.price_data import add_technical_indicators, get_price_history
from finance.price_model import compare_with_without_sentiment
from finance.sentiment import get_ticker_sentiment, headlines_near_date

def _default_fast_model() -> str:
    # Read fresh from the environment on every call, not baked in as a
    # module-level constant -- same rationale as agents/llm_backend.py's
    # call_llm reading BTP_OLLAMA_MODEL live.
    return os.environ.get("BTP_OLLAMA_MODEL", "llama3.2:3b")

PERIODS = ["5D", "1M", "6M", "YTD", "1Y", "5Y", "MAX"]


def render_stock_charts(ticker: str, key_prefix: str = "", compact: bool = False) -> pd.DataFrame:
    st.subheader(f"📊 {ticker} — Price & Technical Indicators")
    raw = get_price_history(ticker, start="2022-01-01")
    feats = add_technical_indicators(raw)

    if compact:
        period, chart_type = "1Y", "Candlestick"
    else:
        c1, c2 = st.columns([2, 1])
        period = c1.select_slider("Period", options=PERIODS, value="1Y", key=f"{key_prefix}_period")
        chart_type = c2.radio("Chart type", ["Candlestick", "Line"], horizontal=True, key=f"{key_prefix}_charttype")

    st.markdown("#### Price")
    st.caption(
        ("Daily candlesticks -- each bar's body/wick shows that day's open, high, low, and close." if chart_type == "Candlestick"
         else "Daily open/high/low/close plotted as lines.")
        + " Drag the range slider below the chart to zoom into a specific stretch."
    )
    price_fig = candlestick_chart(feats, period) if chart_type == "Candlestick" else close_line_chart(feats, period)
    st.plotly_chart(price_fig, use_container_width=True, key=f"{key_prefix}_price")

    if not compact:
        st.divider()
    st.markdown("#### RSI — Relative Strength Index")
    st.caption("A 0-100 momentum gauge for how fast/far price has moved over the last 14 days. Above 70 is "
               "generally read as overbought (may be due for a pullback); below 30 as oversold (may be due for "
               "a bounce) -- neither is a guaranteed reversal on its own.")
    st.plotly_chart(rsi_chart(feats, period), use_container_width=True, key=f"{key_prefix}_rsi")

    if not compact:
        st.divider()
    st.markdown("#### MACD — Moving Average Convergence Divergence")
    st.caption("Tracks the gap between a fast and slow moving average of price against its own signal line. "
               "MACD crossing above the signal line is typically read as strengthening bullish momentum; "
               "crossing below, weakening/bearish momentum.")
    st.plotly_chart(macd_chart(feats, period), use_container_width=True, key=f"{key_prefix}_macd")

    if not compact:
        st.divider()
        st.markdown("#### Moving Averages")
        st.caption("Close price with 7-day and 21-day simple moving averages overlaid to smooth out daily "
                   "noise. The shorter average crossing above the longer one is often read as a short-term "
                   "bullish signal, and crossing below as bearish.")
        st.plotly_chart(moving_average_chart(feats, period), use_container_width=True, key=f"{key_prefix}_ma")

    return feats


def render_company_metrics(ticker: str, key_prefix: str = "") -> None:
    import yfinance as yf

    st.subheader(f"🏢 {ticker} — Company & CAPM Metrics")
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not fetch company info for '{ticker}': {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        df1 = pd.DataFrame(index=["Market Cap", "Beta", "EPS", "P/E Ratio"])
        df1["Value"] = [info.get("marketCap"), info.get("beta"), info.get("trailingEps"), info.get("trailingPE")]
        st.plotly_chart(plotly_table(df1), use_container_width=True, key=f"{key_prefix}_metrics1")
    with col2:
        df2 = pd.DataFrame(index=["Quick Ratio", "Revenue/Share", "Profit Margin", "Debt/Equity", "ROE"])
        df2["Value"] = [info.get("quickRatio"), info.get("revenuePerShare"), info.get("profitMargins"),
                         info.get("debtToEquity"), info.get("returnOnEquity")]
        st.plotly_chart(plotly_table(df2), use_container_width=True, key=f"{key_prefix}_metrics2")

    beta = info.get("beta")
    if beta is not None:
        risk_free, market_return = 0.04, 0.10
        expected_return = risk_free + beta * (market_return - risk_free)
        st.caption(f"**CAPM expected return** (assumes {risk_free:.0%} risk-free rate, {market_return:.0%} market "
                    f"return): **{expected_return:.1%}** at beta={beta:.2f}")


def render_sentiment(ticker: str, key_prefix: str = "") -> tuple:
    st.subheader(f"📰 {ticker} — News/Social Sentiment (FinBERT)")
    with st.spinner("Fetching headlines + scoring sentiment locally..."):
        scored, daily = get_ticker_sentiment(ticker)

    if scored.empty:
        st.info("No recent headlines found for this ticker.")
        return scored, daily

    avg = float(scored["sentiment_score"].mean())
    st.metric("Average sentiment", f"{avg:+.2f}", help="-1 very negative to +1 very positive")
    for _, row in scored.head(8).iterrows():
        st.write(f"- **{row['sentiment_label']}** — {row['headline']}")

    return scored, daily


def render_prediction(ticker: str, sentiment_daily: pd.Series | None = None, key_prefix: str = "", wide: bool = False) -> dict:
    st.subheader(f"🔮 {ticker} — Price Prediction (ARIMA vs XGBoost vs XGBoost+Sentiment)")
    with st.spinner("Training ARIMA + XGBoost models..."):
        raw = get_price_history(ticker, start="2023-01-01")
        feats = add_technical_indicators(raw)
        comparison = compare_with_without_sentiment(feats, sentiment_series=sentiment_daily, test_size=30)

    labels = ["ARIMA baseline", "XGBoost (no sentiment)", "XGBoost (with sentiment)"]
    rmses = [comparison["arima_rmse"], comparison["xgboost_no_sentiment_rmse"], comparison["xgboost_with_sentiment_rmse"]]

    col1, col2 = st.columns([4, 1] if wide else [2, 1])
    with col1:
        actual = comparison["detail_without_sentiment"]["actual"]
        arima_pred = comparison["detail_arima"]["forecast"]
        xgb_pred = comparison["detail_without_sentiment"]["predictions"]
        xgb_sent_pred = comparison["detail_with_sentiment"]["predictions"]
        chart = prediction_comparison_chart(actual, arima_pred, xgb_pred, xgb_sent_pred)
        if wide:
            chart.update_layout(height=650)
        st.plotly_chart(chart, use_container_width=True, key=f"{key_prefix}_predchart")
    with col2:
        st.plotly_chart(rmse_comparison_bar(labels, rmses), use_container_width=True, key=f"{key_prefix}_rmsebar")

    delta = rmses[1] - rmses[2]
    if delta > 0:
        st.success(f"Sentiment **improved** RMSE by {delta:.2f} ({delta / rmses[1] * 100:.1f}%).")
    else:
        st.warning(f"Sentiment made RMSE **worse** by {abs(delta):.2f} ({abs(delta) / rmses[1] * 100:.1f}%) -- "
                    "an honest result to report, not necessarily a bug in the pipeline.")

    return comparison


def render_finance_anomalies(ticker: str, features_df: pd.DataFrame | None = None,
                              sentiment_daily: pd.Series | None = None, key_prefix: str = "") -> pd.DataFrame:
    """Multi-type statistical anomaly detection (finance/anomaly_detector.py)
    over daily price/volume, plus sentiment-price divergence when a daily
    sentiment series is available. `features_df` lets callers that already
    fetched+computed indicators (e.g. render_stock_charts) reuse them
    instead of re-downloading."""
    st.subheader(f"⚠️ {ticker} — Anomaly Detection")
    feats = features_df if features_df is not None else add_technical_indicators(get_price_history(ticker, start="2022-01-01"))

    with st.spinner("Scanning for return/volatility/volume/gap anomalies..."):
        scored = detect_all(feats, sentiment_series=sentiment_daily)

    n_total = len(scored)
    n_flagged = int(scored["is_anomaly"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Days analyzed", n_total)
    col2.metric("Flagged anomalous", n_flagged)
    col3.metric("Anomaly rate", f"{n_flagged / n_total * 100:.1f}%" if n_total else "N/A")
    if sentiment_daily is None or sentiment_daily.empty:
        st.caption("Sentiment-price divergence detector skipped -- no daily sentiment series supplied.")

    st.plotly_chart(finance_anomaly_chart(feats, scored, ticker), use_container_width=True, key=f"{key_prefix}_finanom")

    top = top_anomalies(scored, n=10)
    if top.empty:
        st.info("No anomalies flagged in this period.")
    else:
        table = pd.DataFrame({
            "Date": [idx.date() for idx in top.index],
            "Severity": top["severity_label"].values,
            "Severity score": top["severity_score"].round(2).values,
            "Reasons": [", ".join(r) for r in top["reasons"]],
        })
        st.dataframe(table, use_container_width=True, hide_index=True, key=f"{key_prefix}_finanomtable")

        st.markdown("###### Why were these flagged?")
        st.caption("Statistical reasons are always shown. \"Explain with news\" additionally checks free RSS "
                   "headlines published near that date -- this only finds real matches for RECENT anomalies "
                   "(Yahoo/Google News RSS serve current headlines, not a historical archive), and says so "
                   "honestly rather than guessing a cause for an older one.")

        badge_by_severity = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡"}
        headlines_cache_key = f"{key_prefix}_headlines_raw_{ticker}"

        for date_idx, row in top.iterrows():
            badge = badge_by_severity.get(row["severity_label"], "⚪")
            header = f"{badge} {date_idx.date()} — {row['severity_label']} severity — {', '.join(row['reasons'])}"
            with st.expander(header):
                st.write("**Computed reasons this day was flagged:**")
                for reason in row["reasons"]:
                    st.write(f"- {reason}")

                news_key = f"{key_prefix}_news_cause_{ticker}_{date_idx.date()}"
                if st.button("🤖 Explain with news", key=f"{key_prefix}_news_btn_{date_idx.date()}"):
                    with st.spinner("Fetching nearby headlines and asking the local LLM..."):
                        if headlines_cache_key not in st.session_state:
                            raw_headlines, _ = get_ticker_sentiment(ticker)
                            st.session_state[headlines_cache_key] = raw_headlines
                        raw_headlines = st.session_state[headlines_cache_key]

                        matched = headlines_near_date(raw_headlines, date_idx, window_days=3)
                        if matched.empty:
                            st.session_state[news_key] = (
                                "No news headlines found within 3 days of this date via the free Yahoo/Google "
                                "News RSS feeds -- these only carry current news, not a historical archive, so "
                                "this is most likely just outside their window rather than evidence nothing "
                                "happened. A real cause can't be attributed from this source for this date."
                            )
                        else:
                            headline_lines = "\n".join(
                                f"  - [{r['published']:%Y-%m-%d}] {r['headline']} ({r['sentiment_label']})"
                                for _, r in matched.iterrows()
                            )
                            prompt = (
                                f"Ticker: {ticker}\nDate: {date_idx.date()}\n"
                                f"Statistical anomaly signals already computed for this day (use verbatim, do "
                                f"not invent others): {', '.join(row['reasons'])}\n\n"
                                f"Real news headlines found within 3 days of this date, with FinBERT sentiment "
                                f"label:\n{headline_lines}\n\n"
                                "Using ONLY the headlines above, explain in 2-4 sentences what likely caused "
                                "this anomalous trading day. If none of the headlines plausibly explain the "
                                "statistical signals, say so explicitly rather than forcing a connection -- do "
                                "not invent a cause the headlines don't support."
                            )
                            st.session_state[news_key] = call_llm(prompt, system=FINANCE_EXPLAINER_SYSTEM_PROMPT,
                                                                    model=_default_fast_model())
                if news_key in st.session_state:
                    st.success(st.session_state[news_key])

    return scored


def _load_light_curve(target_input: str, uploaded_file=None) -> tuple[pd.DataFrame, str]:
    """Shared by render_astronomy and render_astronomy_pro so both fetch
    (or read an uploaded CSV/image) the exact same way instead of
    duplicating the real-fetch/synthetic-fallback logic twice. An image
    upload (JPG/PNG) is routed through astronomy.image_lightcurve's pixel
    extraction -- the same path agents/chat_agent.py already uses for a
    screenshotted light curve -- rather than requiring raw CSV data."""
    if uploaded_file is not None:
        name = getattr(uploaded_file, "name", "uploaded")
        mime = (getattr(uploaded_file, "type", "") or "")
        is_image = name.lower().endswith((".png", ".jpg", ".jpeg")) or mime.startswith("image/")

        if is_image:
            from astronomy.image_lightcurve import extract_lightcurve_from_image

            # More points than the module's own default (400) so there are
            # enough post-windowing samples (WINDOW_SIZE=200, STRIDE=50) for
            # the autoencoder's >=10-window "normal" training set to have a
            # real chance of being met, not just a couple of IsolationForest windows.
            lc = extract_lightcurve_from_image(uploaded_file.getvalue(), n_points=800)
            return lc, (f"extracted from uploaded image ({name}) -- recovers the plotted SHAPE only, not "
                        f"physical time/flux units (see astronomy/image_lightcurve.py)")

        lc = pd.read_csv(uploaded_file)
        if "flux" not in lc.columns:
            raise ValueError("Uploaded CSV must have a 'flux' column (and optionally a 'time' column).")
        if "time" not in lc.columns:
            lc["time"] = np.arange(len(lc))
        return lc, f"uploaded file ({name})"

    try:
        lc = fetch_light_curve(target_input)
        return lc, "real (MAST/Kepler via lightkurve)"
    except Exception as exc:  # noqa: BLE001
        seed = abs(hash(target_input)) % (2**32)
        lc = generate_synthetic_light_curve(seed=seed)
        return lc, f"synthetic fallback (real fetch failed: {exc})"


def render_astronomy(target_input: str, uploaded_file=None, key_prefix: str = "") -> tuple:
    st.subheader(f"🔭 {target_input} — Light Curve Anomaly Detection")

    with st.spinner("Fetching light curve (real Kepler/MAST data if reachable)..."):
        try:
            lc, source = _load_light_curve(target_input, uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return None, None

    st.caption(f"Data source: {source}")

    flux, time = lc["flux"].values, lc["time"].values
    windows = make_windows(flux)
    feats = extract_features(windows)
    iso_model = fit_isolation_forest(feats)
    scored = score_anomalies(iso_model, feats)
    anomaly_mask = scored["is_anomaly"].values

    st.plotly_chart(
        lightcurve_anomaly_chart(time, flux, window_size=WINDOW_SIZE, stride=STRIDE, anomaly_mask=anomaly_mask),
        use_container_width=True, key=f"{key_prefix}_lc",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Windows analyzed", len(windows))
    col2.metric("Flagged anomalous", int(anomaly_mask.sum()))
    col3.metric("Anomaly rate", f"{anomaly_mask.mean() * 100:.1f}%")

    pseudo_normal = windows[~anomaly_mask]
    if len(pseudo_normal) >= 10:
        with st.spinner("Training custom autoencoder embedding..."):
            ae_model = train_autoencoder(pseudo_normal)
            errors = reconstruction_error(ae_model, windows)
        st.plotly_chart(reconstruction_error_chart(range(len(errors)), errors),
                         use_container_width=True, key=f"{key_prefix}_recon")

    return scored, windows


def render_astronomy_pro(target_input: str, uploaded_file=None, key_prefix: str = "", config: dict | None = None) -> dict:
    """The Astronomy page's full analysis flow, one step at a time:
    1. the raw light curve, 2. a computed description of it, 3. anomaly
    detection flagged on the raw curve, 4. the same flux as a normalized
    z-score view, 5. per-anomaly cards with the computed reasons + a
    rule-based possible cause (astronomy.anomaly_detector.possible_cause).
    Which steps render, how big, and how many anomalies get explained in
    detail are all controlled by `config` -- see
    agents/astronomy_dashboard_agent.py for how a typed preference becomes
    that config, mirroring the Home dashboard's adaptive-layout pattern
    (agents/dashboard_agent.py) but scoped to this page's own five steps.

    LLM narrative generation (the description/explanation "Generate AI ..."
    buttons) is opt-in via a button rather than automatic, because
    Streamlit reruns this whole function on every widget interaction --
    auto-firing an LLM call on each rerun would make the page unusably slow.
    Results are cached in st.session_state per target so they survive
    reruns until the target actually changes.
    """
    from agents.astronomy_dashboard_agent import DEFAULT_CONFIG

    config = config or DEFAULT_CONFIG
    visible = set(config.get("visible_sections", DEFAULT_CONFIG["visible_sections"]))
    top_n = config.get("top_n_anomalies", 5)
    large = config.get("layout_size") == "large"
    compact = config.get("layout_size") == "compact"

    st.subheader(f"🔭 {target_input} — Light Curve Analysis")

    with st.spinner("Fetching light curve (real Kepler/MAST data if reachable)..."):
        try:
            lc, source = _load_light_curve(target_input, uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return {}

    st.caption(f"Data source: {source}")

    flux, time = lc["flux"].values, lc["time"].values
    windows = make_windows(flux)
    feats = extract_features(windows)
    iso_model = fit_isolation_forest(feats)
    scored = score_anomalies(iso_model, feats)
    anomaly_mask = scored["is_anomaly"].values

    recon_errors = None
    pseudo_normal = windows[~anomaly_mask]
    if len(pseudo_normal) >= 10:
        with st.spinner("Training custom autoencoder embedding..."):
            ae_model = train_autoencoder(pseudo_normal)
            recon_errors = reconstruction_error(ae_model, windows)
    scored = annotate_windows(scored, recon_errors)

    target_id = f"{target_input}_{len(flux)}"  # cache key: reset any cached AI text if the target/data actually changes

    if "light_curve" in visible:
        st.markdown("#### 1. Light curve")
        chart = raw_lightcurve_chart(time, flux)
        chart.update_layout(height=550 if large else (300 if compact else 420))
        st.plotly_chart(chart, use_container_width=True, key=f"{key_prefix}_raw")

    if "description" in visible:
        st.markdown("#### 2. Description")
        duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
        cadence = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
        overall_skew = float(pd.Series(flux).skew())
        overall_kurtosis = float(pd.Series(flux).kurtosis())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data points", len(flux))
        col2.metric("Time span", f"{duration:.1f}")
        col3.metric("Flux range", f"{flux.min():.4f} – {flux.max():.4f}")
        col4.metric("Variability (std)", f"{flux.std():.4f}")

        st.info(
            f"This light curve for **{target_input}** spans **{duration:.1f}** time units across **{len(flux)}** "
            f"points (median cadence ≈{cadence:.4f}). Flux is normalized around a median of 1.0, ranging from "
            f"{flux.min():.4f} to {flux.max():.4f}, with overall variability (std) of {flux.std():.4f} "
            f"({flux.std() * 100:.2f}%). Overall shape skew {overall_skew:.2f}, kurtosis {overall_kurtosis:.2f}."
        )

        desc_key = f"{key_prefix}_ai_desc_{target_id}"
        if st.button("🤖 Generate AI description", key=f"{key_prefix}_ai_desc_btn"):
            with st.spinner("Asking the local LLM to describe this light curve..."):
                prompt = (
                    f"Computed stats for light curve '{target_input}' (data source: {source}):\n"
                    f"- {len(flux)} points over {duration:.1f} time units, median cadence {cadence:.4f}\n"
                    f"- flux range {flux.min():.4f} to {flux.max():.4f}, std {flux.std():.4f}\n"
                    f"- overall skew {overall_skew:.2f}, kurtosis {overall_kurtosis:.2f}\n\n"
                    "Describe this light curve's overall shape and variability in plain language for someone new "
                    "to astronomy, in 2-3 sentences. Use only the numbers given above -- do not claim any specific "
                    "detection (transit/flare/planet) here, that comes later in the anomaly explanations."
                )
                st.session_state[desc_key] = call_llm(prompt, system=ASTRONOMY_EXPLAINER_SYSTEM_PROMPT,
                                                        model=_default_fast_model())
        if desc_key in st.session_state:
            st.success(st.session_state[desc_key])

    if "anomaly_detection" in visible:
        st.markdown("#### 3. Anomaly detection")
        col1, col2, col3 = st.columns(3)
        col1.metric("Windows analyzed", len(windows))
        col2.metric("Flagged anomalous", int(anomaly_mask.sum()))
        col3.metric("Anomaly rate", f"{anomaly_mask.mean() * 100:.1f}%")

        chart = lightcurve_anomaly_chart(time, flux, window_size=WINDOW_SIZE, stride=STRIDE, anomaly_mask=anomaly_mask)
        chart.update_layout(height=600 if large else (350 if compact else 450))
        st.plotly_chart(chart, use_container_width=True, key=f"{key_prefix}_lc")

        if recon_errors is not None and not compact:
            st.plotly_chart(reconstruction_error_chart(range(len(recon_errors)), recon_errors),
                             use_container_width=True, key=f"{key_prefix}_recon")
        elif recon_errors is None:
            st.caption("Not enough windows were classified 'normal' to train the autoencoder embedding for this target.")

    if "normalized_view" in visible:
        st.markdown("#### 4. Normalized view")
        chart = normalized_lightcurve_chart(time, flux, window_size=WINDOW_SIZE, stride=STRIDE, anomaly_mask=anomaly_mask)
        chart.update_layout(height=520 if large else (300 if compact else 380))
        st.plotly_chart(chart, use_container_width=True, key=f"{key_prefix}_norm")

    if "explanations" in visible:
        st.markdown("#### 5. Why these were flagged")
        top = scored.sort_values("anomaly_score", ascending=False).head(top_n)

        if top.empty:
            st.info("No anomalies flagged for this target.")
        else:
            badge_by_severity = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "NONE": "⚪"}
            for window_id, row in top.iterrows():
                start_t = time[window_id * STRIDE]
                end_t = time[min(window_id * STRIDE + WINDOW_SIZE, len(time) - 1)]
                badge = badge_by_severity.get(row["severity_label"], "⚪")
                header = f"{badge} Window {window_id} ({start_t:.1f}–{end_t:.1f}) — {row['severity_label']} severity — {row['possible_cause']}"
                with st.expander(header):
                    st.write("**Computed reasons this window was flagged:**")
                    for reason in row["reasons"]:
                        st.write(f"- {reason}")
                    st.write(f"**Possible cause:** {row['possible_cause']}")
                    st.caption(row["possible_cause_detail"])

            narrative_key = f"{key_prefix}_ai_narrative_{target_id}_{top_n}"
            if st.button("🤖 Generate AI narrative for these anomalies", key=f"{key_prefix}_ai_narrative_btn"):
                with st.spinner("Asking the local LLM to synthesize a narrative..."):
                    lines = "\n".join(
                        f"  - window {wid}: {row['severity_label']} severity, possible cause: {row['possible_cause']} "
                        f"-- {', '.join(row['reasons'])}"
                        for wid, row in top.iterrows()
                    )
                    prompt = (
                        f"Target: {target_input}\nData source: {source}\n"
                        f"Total windows analyzed: {len(windows)}, flagged: {int(anomaly_mask.sum())}\n\n"
                        f"Top flagged windows (already computed -- use these verbatim, do not invent others or "
                        f"re-rank them):\n{lines}\n\n"
                        "Write a short (4-6 sentence) narrative explaining what was flagged and why, in plain "
                        "language, naming the most plausible cause for the top window(s), and one caveat that these "
                        "are rule-based candidate causes, not confirmed classifications."
                    )
                    st.session_state[narrative_key] = call_llm(prompt, system=ASTRONOMY_EXPLAINER_SYSTEM_PROMPT,
                                                                 model=_default_fast_model())
            if narrative_key in st.session_state:
                st.success(st.session_state[narrative_key])

    return {"scored": scored, "windows": windows, "flux": flux, "time": time, "source": source}


def render_market_screener(index_name: str, max_stocks: int = 50, key_prefix: str = "") -> pd.DataFrame:
    """Nifty 50/100/500 or Sensex 30 constituents, ranked by 1-day return
    and RSI(14). Nifty 500 has 500 constituents -- screening all of them
    live on CPU-only hardware would be slow, so this caps to `max_stocks`
    and says so explicitly (no silent truncation), rather than quietly
    only ever showing "the index" while actually covering a fraction of it."""
    st.subheader(f"📈 {index_name.upper()} Screener")

    with st.spinner(f"Fetching {index_name.upper()} constituent list..."):
        all_symbols = fetch_index_constituents(index_name)
    total = len(all_symbols)
    symbols = all_symbols[:max_stocks]
    if max_stocks < total:
        st.caption(f"Showing {len(symbols)} of {total} {index_name.upper()} constituents "
                    "(capped for responsiveness on CPU-only hardware -- raise the slider for more).")

    with st.spinner(f"Fetching price data for {len(symbols)} stocks..."):
        raw = yf.download(symbols, period="3mo", progress=False, group_by="ticker", threads=True)

    rows, failed = [], []
    for sym in symbols:
        try:
            close = raw[sym]["Close"].dropna()
            if len(close) < 15:
                failed.append(sym)
                continue
            change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else None

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

            rows.append({
                "Ticker": sym, "Close": round(float(close.iloc[-1]), 2),
                "1D Return %": round(float(change_1d), 2) if change_1d is not None else None,
                "RSI(14)": round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else None,
            })
        except Exception:  # noqa: BLE001 -- one bad ticker (transient Yahoo hiccup) shouldn't sink the whole screener
            failed.append(sym)

    df = pd.DataFrame(rows)
    if failed:
        shown = ", ".join(failed[:10]) + ("..." if len(failed) > 10 else "")
        st.caption(f"Skipped {len(failed)} tickers with no/insufficient data this run: {shown}")

    if df.empty:
        st.warning("No data returned for any constituent -- try again (Yahoo Finance occasionally rate-limits bulk requests).")
        return df

    sort_col = st.selectbox("Sort by", ["1D Return %", "RSI(14)", "Close"], key=f"{key_prefix}_sort")
    df_sorted = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(df_sorted, use_container_width=True, hide_index=True, key=f"{key_prefix}_table")

    try:
        index_ticker = get_index_ticker(index_name)
        index_hist = get_price_history(index_ticker, start="2025-01-01")
        st.caption(f"{index_name.upper()} index level ({index_ticker}): {float(index_hist['Close'].iloc[-1]):.2f}")
    except Exception:  # noqa: BLE001
        pass

    return df_sorted


def render_mutual_fund_analysis(scheme_code: int, key_prefix: str = "") -> dict:
    """NAV history + trailing returns + ARIMA forecast for one Indian
    mutual fund scheme. No sentiment/XGBoost augmentation here by design --
    see finance/mutual_funds.py's module docstring for why."""
    meta = get_scheme_meta(scheme_code)
    st.subheader(f"💰 {meta.get('scheme_name', scheme_code)}")
    st.caption(f"{meta.get('fund_house', '')} — {meta.get('scheme_category', '')}")

    with st.spinner("Fetching NAV history..."):
        nav_df = get_nav_history(scheme_code)

    with st.spinner("Fitting ARIMA baseline..."):
        forecast_result = forecast_nav(nav_df, test_size=30)

    st.plotly_chart(nav_chart(nav_df.tail(500), forecast=forecast_result["forecast"]),
                     use_container_width=True, key=f"{key_prefix}_nav")

    returns = compute_returns(nav_df)
    cols = st.columns(len(returns))
    for col, (label, val) in zip(cols, returns.items()):
        col.metric(label, f"{val:+.2f}%" if val is not None else "N/A")

    st.caption(f"ARIMA baseline RMSE (last 30 NAV points): {forecast_result['rmse']}")

    return {"meta": meta, "nav_df": nav_df, "returns": returns, "forecast": forecast_result}


def render_model_comparison(ticker: str, sentiment_daily=None, key_prefix: str = "cmp"):
    """All four price models on one held-out window, plotted together.

    Exists because the project's earlier finance numbers were produced by
    separate calls with separate splits, which is not a comparison. Here
    every model predicts the same days from the same training data, and the
    naive "assume no change" baseline is drawn alongside -- measured across
    9 tickers it beat both ARIMA and XGBoost on every one, so omitting it
    would make the chart flattering rather than informative.
    """
    import streamlit as st

    from dashboard.charts import model_comparison_chart, model_error_bar
    from finance.price_data import add_technical_indicators, get_price_history
    from finance.price_model import InsufficientDataError, compare_all_models

    st.subheader("Model comparison — predictions vs reality")

    test_size = st.slider("Days to hold out and predict", 15, 60, 30, step=5,
                          key=f"{key_prefix}_cmp_days",
                          help="Each model is trained on everything BEFORE these days, "
                               "then predicts them. The actual closes are never seen during fitting.")

    with st.spinner("Fitting ARIMA, XGBoost, XGBoost+sentiment and a Gaussian Process..."):
        try:
            features = add_technical_indicators(get_price_history(ticker, start="2023-01-01"))
            comparison = compare_all_models(features, sentiment_daily, test_size=test_size)
        except InsufficientDataError as exc:
            st.warning(f"Not enough history for {ticker}: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Comparison failed: {type(exc).__name__}: {exc}")
            return None

    st.plotly_chart(model_comparison_chart(comparison, ticker), width="stretch")
    st.plotly_chart(model_error_bar(comparison), width="stretch")

    metrics = comparison["metrics"]
    rows = []
    for name, m in metrics.items():
        direction = m.get("directional_accuracy")
        rows.append({
            "Model": name,
            "RMSE": m["rmse"],
            "MAE": m["mae"],
            "Direction correct": "n/a" if direction is None else f"{direction:.0%}",
            "vs naive": f"{m['vs_naive']:.2f}x",
            "Beats naive": "yes" if m["beats_naive"] else "no",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    beaten = [n for n, m in metrics.items()
              if n != "Naive (no change)" and m["beats_naive"]]
    if beaten:
        st.success(f"Beat the naive baseline: {', '.join(beaten)}", icon="✅")
    else:
        st.warning(
            "**No model beat the naive baseline** on this window — predicting \"tomorrow equals "
            "today\" was more accurate than every fitted model. Directional accuracy sits near "
            "50%, i.e. a coin flip. This is the expected result for daily equity prices, which "
            "are close to a random walk: if past prices predicted tomorrow's direction, that "
            "edge would be traded away. Reported rather than hidden — it is the honest finding, "
            "and a comparison that omitted the baseline would have looked like a success.",
            icon="⚠️")

    # The sentiment rows are only interpretable alongside their coverage.
    # Free RSS reaches back weeks, not years, so the sentiment feature is
    # exactly 0.0 on almost every training row -- which means the two
    # sentiment models are not really testing sentiment at all. Saying so
    # here prevents the table being read as evidence that sentiment does
    # not matter, when what it actually shows is that the data is missing.
    cov = comparison.get("sentiment_coverage")
    if cov:
        st.info(
            f"**On the sentiment rows:** news sentiment is available for only "
            f"**{cov['rows_with_sentiment']} of {cov['total_rows']} training days "
            f"({cov['fraction']:.1%})** — free RSS feeds carry recent headlines, not history, so "
            f"the feature is 0.0 on the rest. Both sentiment models are therefore fitting a column "
            f"that is almost entirely zeros, and any difference they show is closer to noise than "
            f"to evidence about sentiment. Testing this properly needs sentiment collected daily "
            f"going forward, not reconstructed backwards.",
            icon="📰")

    if comparison.get("errors"):
        with st.expander("Models that failed to fit"):
            for name, err in comparison["errors"].items():
                st.write(f"**{name}** — {err}")

    return comparison


def render_volatility_forecast(ticker: str, key_prefix: str = "vol") -> dict | None:
    """Next-day volatility for one stock, with the baseline it has to beat.

    This is the forecasting task where a model actually wins. Price models
    here all lose to carry-forward -- that is the random-walk result, not a
    bug -- so the project forecasts volatility instead, where HAR cuts the
    loss 47% against naive over 59,709 held-out forecasts.

    The model is har_ovn, and the choice is worth surfacing: four models sit
    inside the 95% Model Confidence Set and cannot be separated statistically,
    so the tie was broken on PREREQUISITES rather than on the lowest number.
    har_ovn needs only the OHLC bars already downloaded; the tied alternative
    har_iv needs a live India VIX feed for no measurable gain.
    """
    from dashboard.charts import volatility_forecast_chart
    from quant.vol_models import forecast_ticker

    st.subheader(f"📉 {ticker} — Volatility Forecast (HAR + overnight)")
    st.caption("How BIG tomorrow's move will be, not which way. Returns are a random walk "
               "here (lag-1 autocorrelation +0.010); their volatility is not (+0.459).")

    try:
        with st.spinner("Fitting HAR on this ticker's own history..."):
            res = forecast_ticker(ticker)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Volatility forecast unavailable for {ticker}: {exc}")
        return None

    fc, nv = res["forecast_annualised_vol"], res["naive_annualised_vol"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast (next session)", f"{fc:.1%}",
              delta=f"{fc - nv:+.1%} vs naive",
              help="Annualised. 0.25 means a 25%-a-year pace of movement.")
    c2.metric("Naive carry-forward", f"{nv:.1%}",
              help="Yesterday's realized volatility, repeated. The baseline to beat.")
    c3.metric("22-day average", f"{res['last_22d_mean']:.1%}",
              help="HAR's longest component, and the one it weights most heavily.")

    st.plotly_chart(volatility_forecast_chart(res["history"], fc, nv, ticker),
                    width="stretch", key=f"{key_prefix}_vol_chart")

    st.caption(
        f"Fitted on {res['n_train']} days ending {res['as_of']}. "
        f"Measured skill: QLIKE {res['skill']['value']} against naive's 0.77935 over "
        "59,709 held-out forecasts across 48 NSE names — a 48% reduction, with naive "
        "the best model on none of them. Volatility says how large the move will be and "
        "never which direction; it is not a trading signal."
    )
    return res


def render_volatility_backtest(ticker: str, n_days: int = 63,
                               key_prefix: str = "vol") -> dict | None:
    """Did the forecast actually work? Walk-forward over the recent past.

    Every point is produced by a model fitted ONLY on days before it. The
    day being predicted and everything after is absent from the fit, so no
    forecast can have seen its own answer.
    """
    from dashboard.charts import volatility_backtest_chart
    from quant.vol_models import backtest_ticker

    st.subheader(f"🧪 {ticker} — Does it actually work?")
    st.caption("Walk-forward backtest. At every step the model is refitted on data strictly "
               "before that day, then asked for the next one — so nothing is ever trained on "
               "the day it predicts.")

    months = st.radio("Period", [3, 6, 12], index=0, horizontal=True,
                      format_func=lambda m: f"{m} months",
                      key=f"{key_prefix}_bt_months")
    try:
        with st.spinner(f"Refitting the model once per day over {months} months..."):
            r = backtest_ticker(ticker, n_days=int(months * 21))
    except Exception as exc:  # noqa: BLE001
        st.info(f"Backtest unavailable for {ticker}: {exc}")
        return None

    h, n = r["har_ovn"], r["naive"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("QLIKE — HAR", f"{h['qlike']:.4f}",
              delta=f"{-r['qlike_reduction_vs_naive']:.0%} vs naive", delta_color="inverse",
              help="Primary metric; lower is better. Robust to noise in the volatility proxy.")
    c2.metric("QLIKE — naive", f"{n['qlike']:.4f}", help="Yesterday repeated.")
    c3.metric("Mean abs error — HAR", f"{h['mae']:.4f}")
    c4.metric("Mean abs error — naive", f"{n['mae']:.4f}")

    verdict = "beats" if r["beats_naive"] else "does NOT beat"
    st.markdown(f"Over **{r['n_days']} trading days** ({r['span'][0]} to {r['span'][1]}), "
                f"the model **{verdict}** the naive baseline, cutting QLIKE by "
                f"**{r['qlike_reduction_vs_naive']:.0%}**.")

    st.plotly_chart(volatility_backtest_chart(r["backtest"], ticker),
                    width="stretch", key=f"{key_prefix}_bt_chart")

    d = r["dispersion"]
    st.info(
        f"**Why the orange line is smoother than the grey one.** The forecast varies about "
        f"**{d['ratio']:.0%}** as much as the realized series "
        f"(forecast {d['forecast_range'][0]:.1%}–{d['forecast_range'][1]:.1%}, "
        f"actual {d['actual_range'][0]:.1%}–{d['actual_range'][1]:.1%}). That is deliberate, "
        "not a failure to react: a single day's high–low range is a noisy estimate of a "
        "smoother underlying process, so a good forecaster tracks the level and declines to "
        "chase every spike. It is also why HAR weights the 5- and 22-day averages far more "
        "than yesterday — and why naive, which copies yesterday's noise wholesale, loses."
    )
    st.caption("Volatility says how large tomorrow's move will be, never which direction. "
               "Not a trading signal.")
    return r
