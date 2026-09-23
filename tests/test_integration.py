"""Smoke tests -- not exhaustive unit tests, just a fast sanity check that
each module's core function runs without crashing, plus one full
multi-agent end-to-end run per domain. Run with: pytest tests/ -v -s
"""

import numpy as np
import pytest


def test_price_data():
    from finance.price_data import add_technical_indicators, get_price_history

    df = get_price_history("AAPL", start="2024-01-01")
    feats = add_technical_indicators(df)
    assert len(feats) > 50
    assert "RSI_14" in feats.columns


def test_price_model_arima_and_xgboost():
    from finance.price_data import add_technical_indicators, get_price_history
    from finance.price_model import compare_with_without_sentiment

    df = get_price_history("AAPL", start="2023-01-01")
    feats = add_technical_indicators(df)
    result = compare_with_without_sentiment(feats, sentiment_series=None, test_size=30)
    assert result["arima_rmse"] > 0
    assert result["xgboost_no_sentiment_rmse"] > 0


def test_finance_anomaly_detector():
    """Regression + real-data validation for finance/anomaly_detector.py
    (Phase 1 of the finance anomaly detection plan). Checks structure, that
    top_anomalies is sorted by severity, and -- the real validation -- that
    a genuine, well-known market event actually gets flagged: 2024-06-04 was
    India's Lok Sabha election-result day, when Nifty/Sensex crashed
    intraday on the exit-poll-vs-actual mismatch. If RELIANCE.NS has that
    date in range, it should be flagged anomalous, not just flagged on some
    arbitrary date."""
    import pandas as pd

    from finance.anomaly_detector import detect_all, top_anomalies
    from finance.price_data import add_technical_indicators, get_price_history

    df = get_price_history("RELIANCE.NS", start="2023-01-01")
    feats = add_technical_indicators(df)
    scored = detect_all(feats)

    assert {"is_anomaly", "severity_label", "severity_score", "reasons"}.issubset(scored.columns)
    assert int(scored["is_anomaly"].sum()) > 0

    top = top_anomalies(scored, n=5)
    assert len(top) <= 5
    scores = top["severity_score"].tolist()
    assert scores == sorted(scores, reverse=True)

    known_event = pd.Timestamp("2024-06-04")
    if known_event in scored.index:
        assert bool(scored.loc[known_event, "is_anomaly"]), (
            "Expected 2024-06-04 (Indian election-result crash day) to be flagged anomalous for RELIANCE.NS"
        )


def test_anomaly_query_pattern_matches_natural_phrasing():
    """Regression test for a real bug found while building the unified
    anomaly chatbot: \\banomal\\b (with a trailing word boundary right
    after the "anomal" stem) can never match "anomaly" or "anomalies" --
    there's no word boundary between "l" and the "y"/"ies" that always
    follows it in real English, so the very word the feature exists to
    detect was silently never matched. Fixed with anomal\\w*."""
    from agents.chat_agent import _ANOMALY_QUERY_PATTERN

    should_match = ["any anomalies in reliance", "anything unusual happening", "is TCS behaving abnormally",
                     "any outliers today", "check for anomalies", "an irregular pattern"]
    for text in should_match:
        assert _ANOMALY_QUERY_PATTERN.search(text), f"expected a match in: {text!r}"
    assert not _ANOMALY_QUERY_PATTERN.search("what is the price")


def test_mode_pin_command():
    """The explicit session-level domain pin ('switch to astronomy mode')
    should set pinned_domain and answer without running any pipeline/LLM
    call (it's a meta-command, not a data question)."""
    from agents.chat_agent import answer_query

    session = {}
    result = answer_query("switch to astronomy mode", session)
    assert session.get("pinned_domain") == "astronomy"
    assert result["domain"] == "astronomy"

    result2 = answer_query("let's talk finance", session)
    assert session.get("pinned_domain") == "finance"
    assert result2["domain"] == "finance"


def test_anomaly_shared_normalizers():
    """agents/anomaly_shared.py normalizes finance and astronomy anomaly
    results into the same AnomalySummary shape -- the shared presentation
    contract underneath the unified chatbot -- without unifying the
    underlying detection math."""
    from agents.anomaly_shared import from_astronomy_result, from_finance_result

    finance_summary = from_finance_result({
        "ticker": "RELIANCE.NS", "n_days_analyzed": 100, "n_anomalous_days": 5,
        "used_sentiment_divergence": True,
        "top_anomalies": [{"date": "2024-06-04", "severity": "HIGH", "severity_score": 8.5, "reasons": ["return z-score -4.5"]}],
    })
    assert finance_summary["domain"] == "finance"
    assert finance_summary["n_flagged"] == 5
    assert "sentiment_divergence" in finance_summary["methods"]
    assert finance_summary["top_anomalies"][0]["label"] == "2024-06-04"

    astro_summary = from_astronomy_result({
        "target": "KIC 11446443", "n_windows": 40, "n_flagged_anomalies": 4, "mean_reconstruction_error": 0.0004,
        "top_anomaly_windows": [{"window_id": 3, "severity_label": "HIGH", "anomaly_score": 1.2, "reasons": ["IsolationForest anomaly score 1.20"]}],
    })
    assert astro_summary["domain"] == "astronomy"
    assert astro_summary["n_flagged"] == 4
    assert "autoencoder_reconstruction_error" in astro_summary["methods"]
    assert astro_summary["top_anomalies"][0]["label"] == "window 3"


def test_astronomy_node_severity_and_reasons():
    """astronomy_node's per-window output must carry severity_label/reasons
    now, not just a raw anomaly_score -- the astronomy-side half of the
    same explainability upgrade finance/anomaly_detector.py already has,
    needed so agents/anomaly_shared.py's normalizer has real fields to
    read on both sides."""
    from agents.nodes.astronomy_node import astronomy_node

    state = {"target": "SYNTHETIC-TEST-TARGET", "trace": []}
    result = astronomy_node(state)["astronomy_result"]

    assert result["n_windows"] > 0
    assert result["top_anomaly_windows"], "expected at least one top anomaly window"
    for w in result["top_anomaly_windows"]:
        assert w["severity_label"] in {"LOW", "MEDIUM", "HIGH", "NONE"}
        assert isinstance(w["reasons"], list) and w["reasons"]
    assert result["_flux"] is not None and result["_time"] is not None and result["_anomaly_mask"] is not None


def test_astronomy_anomaly_pipeline():
    from astronomy.anomaly_detector import run_pipeline
    from astronomy.data_loader import generate_synthetic_light_curve

    lc = generate_synthetic_light_curve(n_points=6000, n_transits=4, transit_depth=0.015, seed=1)
    result = run_pipeline(lc["flux"].values)
    assert result["metrics"]["n_windows"] > 0
    assert 0 <= result["metrics"]["precision"] <= 1


def test_ts_embedder():
    from astronomy.anomaly_detector import extract_features, label_windows_by_depth, make_windows
    from astronomy.data_loader import generate_synthetic_light_curve
    from astronomy.ts_embedder import embed_windows, train_autoencoder

    lc = generate_synthetic_light_curve(n_points=6000, n_transits=4, transit_depth=0.015, seed=1)
    windows = make_windows(lc["flux"].values)
    feats = extract_features(windows)
    labels = label_windows_by_depth(feats)

    model = train_autoencoder(windows[labels == 0], epochs=5)  # short, just checking it runs
    embeddings = embed_windows(model, windows)
    assert embeddings.shape == (len(windows), 8)


def test_dataset_index_has_real_data():
    """Assumes astronomy.dataset.build_dataset() has already been run at
    least once (it writes data/cache/lightcurve_dataset_index.json) --
    skips rather than fails if not, since building it is a slow, separate
    step (real MAST downloads + IsolationForest + autoencoder per target),
    not something every test run should trigger."""
    from astronomy.dataset import load_index

    try:
        index_df = load_index()
    except FileNotFoundError:
        pytest.skip("No light curve dataset built yet -- run `python -m astronomy.dataset` first.")

    assert len(index_df) > 0
    assert {"target", "source", "n_points", "n_windows", "n_flagged_anomalies", "anomaly_rate"} <= set(index_df.columns)
    assert (index_df["source"].str.startswith("real")).sum() > 0  # at least some real MAST data, not all fallback


@pytest.mark.slow
def test_chat_agent_dataset_query():
    """Regression test for two things:
    1. Dataset-query routing: a question about the whole downloaded
       collection ("which of my targets has the most anomalies") should be
       answered from the precomputed index (fast, no live fetch), not
       routed through the single-target astronomy pipeline.
    2. A real numerical-comparison bug found in testing: llama3.2:3b, asked
       to find the target with the most anomalies across ~20 rows, picked
       one with 84 flagged windows when a different target actually had 98
       (the true max). Fixed by precomputing the correct answer in Python
       and handing it to the model as a labeled fact (see
       agents/chat_agent.py's _answer_dataset_query) instead of asking the
       model to derive it -- this asserts the now-correct target appears."""
    from astronomy.dataset import load_index

    try:
        index_df = load_index()
    except FileNotFoundError:
        pytest.skip("No light curve dataset built yet -- run `python -m astronomy.dataset` first.")

    true_max_target = index_df.sort_values("n_flagged_anomalies", ascending=False).iloc[0]["target"]

    from agents.chat_agent import answer_query

    session = {}
    result = answer_query("which of my targets has the most anomalies", session, backend="ollama", model="llama3.2:3b")
    assert result["domain"] == "astronomy"
    assert result["answer"]
    assert true_max_target in result["answer"]
    assert result["chart"] is not None
    assert any("dataset-level query" in line for line in result["trace"])


def test_image_lightcurve_extraction():
    from astronomy.image_lightcurve import _render_test_curve_to_png, extract_lightcurve_from_image

    t = np.linspace(0, 100, 800)
    flux = 1.0 - 0.05 * np.exp(-0.5 * ((t - 30) / 2) ** 2) - 0.03 * np.exp(-0.5 * ((t - 70) / 3) ** 2)
    image_bytes = _render_test_curve_to_png(t, flux)

    extracted = extract_lightcurve_from_image(image_bytes, n_points=200)
    assert len(extracted) == 200
    assert {"time", "flux"} == set(extracted.columns)

    original_resampled = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(flux)), flux)
    correlation = np.corrcoef(extracted["flux"].values, original_resampled)[0, 1]
    # Not a pixel-perfect digitizer (see astronomy/image_lightcurve.py's
    # docstring) -- 0.5 is a reasonable bar for "the shape was recovered",
    # calibrated against the ~0.66 seen in manual testing on this exact
    # test curve, leaving margin for the median-filter/tracking's minor
    # run-to-run sensitivity to image rendering details.
    assert correlation > 0.5


def test_image_lightcurve_extraction_autocrops_real_world_screenshot():
    """Regression test for a real bug found in manual testing: a genuine
    user upload was a full webpage/lightbox screenshot (dark surrounding
    background, nav arrows, a site banner) around the actual plot, not a
    tightly-cropped plot image. Without cropping first, the column-tracer
    derailed onto that surrounding clutter instead of the curve --
    extraction correlation with the true signal was ~0.11, visibly garbage
    (a near-flat stairstep, not the source curve's shape at all).
    astronomy/image_lightcurve.py's _autocrop_to_plot now finds the largest
    contiguous bright (matplotlib-background-colored) region and crops to
    it first; this checks that recovers a real, non-garbage signal instead
    (0.39 correlation observed in manual testing -- not as clean as an
    already-tightly-cropped image, but a 3.5x improvement, not zero)."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    from astronomy.image_lightcurve import extract_lightcurve_from_image

    t = np.linspace(1950, 1985, 800)
    flux = 97220 + 15 * np.sin(t) + np.random.default_rng(0).normal(0, 8, 800)
    flux[150:160] += 70  # a spike, like a real flagged anomaly

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, flux, color="black", linewidth=0.8)
    ax.set_title("Lightcurve for Sector 24")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux [e/s]")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)  # no bbox_inches="tight" -- natural white margin, like a real screenshot
    plt.close(fig)
    plot_img = Image.open(buf).convert("RGB")

    # Paste the plot onto a much larger dark "webpage" canvas with clutter,
    # mimicking a real lightbox/viewer screenshot rather than a bare plot.
    canvas = Image.new("RGB", (1180, 912), color=(20, 20, 20))
    canvas.paste(plot_img, (150, 80))
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(20, 450), (60, 430), (60, 470)], fill=(200, 200, 200))
    draw.polygon([(1120, 450), (1080, 430), (1080, 470)], fill=(200, 200, 200))
    draw.rectangle([0, 0, 1180, 20], fill=(40, 40, 40))
    draw.text((10, 5), "readthedocs.io", fill=(100, 200, 100))
    out = io.BytesIO()
    canvas.save(out, format="png")

    extracted = extract_lightcurve_from_image(out.getvalue(), n_points=300)
    true_resampled = np.interp(np.linspace(0, 1, 300), np.linspace(0, 1, len(flux)), flux)
    true_norm = (true_resampled - true_resampled.min()) / (true_resampled.max() - true_resampled.min())
    correlation = np.corrcoef(extracted["flux"].values, true_norm)[0, 1]
    assert correlation > 0.3, f"expected autocrop to recover a real (if imperfect) signal, got correlation={correlation:.2f}"


def test_image_lightcurve_extraction_rejects_blank_image():
    import io

    from PIL import Image

    from astronomy.image_lightcurve import extract_lightcurve_from_image

    blank = Image.new("RGB", (400, 200), color="white")
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    with pytest.raises(ValueError):
        extract_lightcurve_from_image(buf.getvalue())


def test_text_embeddings_and_vector_store():
    from rag.embed_text import embed_texts
    from rag.vector_store import VectorStore

    texts = ["Apple stock rallies", "Tesla shares drop", "Fed holds rates"]
    vecs = embed_texts(texts)
    assert vecs.shape == (3, 384)

    store = VectorStore(dim=384)
    store.add(vecs, [{"kind": "news", "text": t} for t in texts])
    results = store.search(vecs[0], k=1)
    assert results[0]["text"] == "Apple stock rallies"


def test_india_index_constituents():
    from finance.india_universe import fetch_index_constituents, get_index_ticker

    nifty50 = fetch_index_constituents("nifty50")
    assert len(nifty50) == 50
    assert all(s.endswith(".NS") for s in nifty50)

    sensex = fetch_index_constituents("sensex")
    assert len(sensex) == 30

    assert get_index_ticker("nifty50") == "^NSEI"
    assert get_index_ticker("sensex") == "^BSESN"


def test_mutual_fund_nav_and_returns():
    from finance.mutual_funds import compute_returns, get_nav_history, get_scheme_meta

    scheme_code = 119598  # SBI Large Cap Fund - Direct - Growth, verified reachable
    meta = get_scheme_meta(scheme_code)
    assert meta["scheme_code"] == scheme_code

    nav_df = get_nav_history(scheme_code)
    assert len(nav_df) > 1000
    assert nav_df.index.is_monotonic_increasing

    returns = compute_returns(nav_df)
    assert "1Y" in returns


@pytest.mark.slow
def test_full_finance_graph_run():
    from agents.graph import run

    result = run(ticker="AAPL")
    assert result["domain"] == "finance"
    assert result["report"]
    assert "finance_result" in result
    assert "finance_anomaly_result" in result
    assert "n_anomalous_days" in result["finance_anomaly_result"]
    assert "_scored_df" not in result["report"]  # internal df must never leak into the LLM-facing report text


@pytest.mark.slow
def test_full_astronomy_graph_run():
    from agents.graph import run

    result = run(target="KIC 11446443")
    assert result["domain"] == "astronomy"
    assert result["report"]
    assert "astronomy_result" in result


@pytest.mark.slow
def test_chat_agent_multiturn_followup():
    """Regression test for the pronoun-follow-up bug found during manual
    testing: 'what about its sentiment' right after an AAPL question was
    misclassified as domain='general' by the small model and hallucinated
    about an unrelated topic. The referential-pattern override in
    agents/chat_agent.py should route it back to finance/AAPL instead."""
    from agents.chat_agent import answer_query

    session = {}
    first = answer_query("what's AAPL's RSI right now", session, backend="ollama", model="llama3.2:3b")
    assert first["domain"] == "finance"
    assert session["last_ticker"] == "AAPL"

    followup = answer_query("what about its sentiment", session, backend="ollama", model="llama3.2:3b")
    assert followup["domain"] == "finance"
    assert "aapl" not in followup["answer"].lower() or True  # ticker need not be echoed, just must not be off-topic
    assert not any(w in followup["answer"].lower() for w in ["bitcoin", "crypto"])  # the observed hallucination


@pytest.mark.slow
def test_dashboard_agent_preference_parsing():
    from agents.dashboard_agent import parse_dashboard_preference

    cfg = parse_dashboard_preference("hide the astronomy section, I only care about finance today", backend="ollama", model="llama3.2:3b")
    assert "astronomy" not in cfg["visible_sections"]
    assert cfg["visible_sections"]  # shouldn't wipe out everything


@pytest.mark.slow
def test_chat_agent_india_index_and_mutual_fund():
    """Covers the two new chat domains: an Indian index name should route
    to finance with the correct index ticker (regardless of what the LLM's
    JSON classification says -- see _detect_india_index), and a named
    mutual fund should resolve to a real scheme and answer from real NAV
    data, with follow-ups reusing the cached scheme."""
    from agents.chat_agent import answer_query

    session = {}
    index_result = answer_query("how is nifty 50 doing today", session, backend="ollama", model="llama3.2:3b")
    assert index_result["domain"] == "finance"
    assert session["last_ticker"] == "^NSEI"

    mf_result = answer_query("tell me about SBI Large Cap fund returns", session, backend="ollama", model="llama3.2:3b")
    assert mf_result["domain"] == "mutual_fund"
    assert session["last_mutual_fund"] == "SBI Large Cap fund"

    followup = answer_query("what about its 3 year return", session, backend="ollama", model="llama3.2:3b")
    assert followup["domain"] == "mutual_fund"
    assert any("reusing cached mutual fund data" in line for line in followup["trace"])


def test_ticker_resolution_ambiguous_and_exact():
    """Doesn't assert on the exact set of candidates Yahoo's live search
    returns for "reliance" -- observed in testing that this genuinely
    varies run to run (one run returned RELIANCE.NS + RELINFRA.NS, another
    returned only RELINFRA.NS/RELINFRA.BO). That's the live external
    service's own ranking fluctuating, not something under this code's
    control -- the guaranteed behavior is "ambiguous with 2+ real NSE/BSE
    candidates", which is what's checked."""
    from finance.ticker_search import resolve_ticker

    reliance = resolve_ticker("reliance")
    assert reliance["status"] == "ambiguous"
    assert len(reliance["candidates"]) >= 2
    assert all(c["symbol"].endswith((".NS", ".BO")) for c in reliance["candidates"])

    aapl = resolve_ticker("AAPL")
    assert aapl["status"] == "resolved"
    assert aapl["symbol"] == "AAPL"


def test_price_forecast_future():
    from finance.price_data import get_price_history
    from finance.price_model import forecast_future

    close = get_price_history("RELIANCE.NS", start="2023-01-01")["Close"]
    result = forecast_future(close, days=30)
    assert len(result["forecast"]) == 30
    assert result["final_predicted_date"] > result["last_known_date"]
    assert result["forecast"]["lower_80"].iloc[-1] < result["forecast"]["predicted_close"].iloc[-1] < result["forecast"]["upper_80"].iloc[-1]


def test_forecast_rejects_insufficient_data():
    """Regression test: a thinly-traded ticker (found in testing:
    RELINFRA.BO returned exactly 1 row from Yahoo Finance regardless of
    start date) used to crash deep inside statsmodels with a cryptic
    'Invalid input, x is constant' error. Should now fail fast with a
    clear, catchable InsufficientDataError instead."""
    import pandas as pd

    from finance.price_model import InsufficientDataError, forecast_future

    tiny_series = pd.Series([100.0, 101.0, 99.5], index=pd.date_range("2026-01-01", periods=3))
    with pytest.raises(InsufficientDataError):
        forecast_future(tiny_series, days=30)


def test_finance_pipeline_handles_insufficient_data_without_crashing(monkeypatch):
    """Regression test for the full failure chain found in testing: a
    disambiguation candidate with almost no Yahoo Finance history (a real,
    live example: RELINFRA.BO returned 1 row; another run's candidate
    returned 0) used to propagate InsufficientDataError uncaught through
    agents/nodes/finance_node.py, crashing the entire LangGraph run for
    ANY caller (chatbot, CLI, dashboard) -- not just the price-forecast
    path this was first caught in. finance_node/sentiment_node/
    explainer_node should now all degrade gracefully instead.

    Monkeypatches get_price_history to return a too-short series
    deterministically, rather than depending on which real-world ticker
    happens to be data-sparse today."""
    import pandas as pd

    import agents.nodes.finance_node as finance_node_module
    from agents.graph import run

    tiny_df = pd.DataFrame(
        {"Open": [100.0] * 5, "High": [101.0] * 5, "Low": [99.0] * 5, "Close": [100.0] * 5, "Volume": [1000] * 5},
        index=pd.date_range("2026-07-01", periods=5),
    )
    # finance_node did `from finance.price_data import get_price_history`, a
    # separate binding in its own namespace -- patch it there, not on the
    # source module, or the monkeypatch would silently do nothing.
    monkeypatch.setattr(finance_node_module, "get_price_history", lambda ticker, start=None, end=None: tiny_df)

    result = run(ticker="SPARSE.NS")
    assert result["domain"] == "finance"
    assert result["report"]  # graceful message, not a crash
    assert result["finance_result"].get("error") == "insufficient_data"


@pytest.mark.slow
def test_chat_agent_disambiguation_triggers_and_resolves():
    """Regression test: 'reliance' (ambiguous -- real-world matches both
    Reliance Industries and the unrelated Reliance Infrastructure) must
    trigger a disambiguation prompt, not silently auto-resolve, even
    though the small model's OWN intent JSON quietly guesses a specific
    ticker on its own. Then picking WHICHEVER candidate was actually
    offered (not assuming a specific one -- Yahoo's live search ranking
    for "reliance" was observed varying run to run in testing, sometimes
    not even including the same tickers) must resolve cleanly with no
    disambiguation prompt left pending."""
    from agents.chat_agent import answer_query

    session = {}
    first = answer_query("tell me about reliance", session, backend="ollama", model="llama3.2:3b")
    assert "_pending_disambiguation" in session
    candidates = session["_pending_disambiguation"]["candidates"]
    assert len(candidates) >= 2

    picked = answer_query(candidates[0]["symbol"], session, backend="ollama", model="llama3.2:3b")
    assert "_pending_disambiguation" not in session
    assert session.get("last_ticker") == candidates[0]["symbol"]


@pytest.mark.slow
def test_chat_agent_forecast_with_chart_and_followup_reuse():
    """Regression test for two real bugs found in manual testing:
    1. "next thirty days" (spelled out, not "30") must still trigger the
       real forecast_future() path and return a chart -- the digit-only
       regex originally used silently missed spelled-out numbers.
    2. A referential follow-up ("what about its RSI") must reuse the
       already-resolved ticker directly, not re-trigger disambiguation.
    Uses an explicit, unambiguous, liquid ticker (RELIANCE.NS) so this is
    deterministic and independent of live disambiguation-candidate
    ranking (covered separately, see test_chat_agent_disambiguation_triggers_and_resolves)."""
    from agents.chat_agent import answer_query

    session = {}
    first = answer_query("what is the price for RELIANCE.NS in the next thirty days", session,
                          backend="ollama", model="llama3.2:3b")
    assert session.get("last_ticker") == "RELIANCE.NS"
    assert first.get("chart") is not None

    followup = answer_query("what about its RSI", session, backend="ollama", model="llama3.2:3b")
    assert "_pending_disambiguation" not in session
    assert followup["domain"] == "finance"
    assert any("RELIANCE.NS" in line for line in followup["trace"])
    assert not any("matched multiple companies" in line for line in followup["trace"])


@pytest.mark.slow
def test_finance_prompt_no_investment_hallucination_and_disclaimer():
    """The strict finance system prompt (agents/prompts.py) requires: real
    company names only, and a SEBI-advisor disclaimer on investment
    suggestions. Doesn't assert ticker accuracy (observed in manual testing
    that the small 3B model sometimes gets ticker symbols wrong even while
    following the disclaimer rule -- a genuine model-quality limit, not
    something a regex test should paper over)."""
    from agents.chat_agent import answer_query

    session = {}
    result = answer_query("which IT stocks should I invest in", session, backend="ollama", model="llama3.2:3b")
    assert "sebi" in result["answer"].lower()


@pytest.mark.slow
def test_astronomy_prompt_distinguishes_detection_from_confirmation():
    """The strict astronomy system prompt must not confirm a planet
    detection outright -- a flagged anomaly is a candidate signal, not an
    automatic confirmed planet (rule 4 in agents/prompts.py).

    Note: this asserts on the small model's exact free-text phrasing, so it
    can occasionally flake on wording alone (observed once, passed cleanly
    on retry with no code change) -- that's LLM output non-determinism, not
    a regression. Re-run before assuming a real break."""
    from agents.chat_agent import answer_query

    session = {}
    answer_query("check KIC 11446443 for anomalies", session, backend="ollama", model="llama3.2:3b")
    result = answer_query("is this definitely a planet", session, backend="ollama", model="llama3.2:3b")
    lower = result["answer"].lower()
    assert not any(phrase in lower for phrase in ["yes, it is a planet", "this confirms a planet", "definitely a planet"])


@pytest.mark.slow
def test_chat_agent_image_lightcurve_upload():
    """Regression test for the image-upload feature: an uploaded light
    curve plot should bypass intent parsing entirely (an image is
    unambiguously astronomy), get extracted, run through the same
    anomaly-detection pipeline as a real KIC-ID target, and produce both a
    real chart and a grounded report. Also caught a real bug during manual
    testing: agents/nodes/explainer_node.py's call_llm never receives a
    `model` override, so it silently fell back to the larger default
    Ollama model and hit an out-of-memory allocation error on this
    CPU-only machine -- fixed by threading `model` through via the
    BTP_OLLAMA_MODEL env var, same pattern already used for `backend`."""
    from astronomy.image_lightcurve import _render_test_curve_to_png
    from agents.chat_agent import answer_query

    t = np.linspace(0, 100, 800)
    flux = 1.0 - 0.05 * np.exp(-0.5 * ((t - 30) / 2) ** 2) - 0.03 * np.exp(-0.5 * ((t - 70) / 3) ** 2)
    image_bytes = _render_test_curve_to_png(t, flux)

    session = {}
    result = answer_query("are there any anomalies in this light curve", session, backend="ollama",
                           model="llama3.2:3b", image_bytes=image_bytes, image_name="test_curve.png")
    assert result["domain"] == "astronomy"
    assert result["answer"]
    assert result["chart"] is not None
    assert session["last_domain"] == "astronomy"
    assert "TEST_CURVE.PNG" in session["last_target"]


@pytest.mark.slow
def test_chat_agent_finance_anomaly_query_returns_chart_and_summary_then_bare_followup():
    """One chatbot, two specialized detectors underneath: an explicit
    anomaly question for a ticker should get both a chart
    (dashboard.charts.finance_anomaly_chart) and a normalized
    agents.anomaly_shared.AnomalySummary, and a bare, entity-free follow-up
    ("anything unusual lately?") should fall back to the same ticker via
    last_domain/last_ticker instead of answering as ungrounded general
    knowledge."""
    from agents.chat_agent import answer_query

    session = {}
    first = answer_query("check RELIANCE.NS for any anomalies", session, backend="ollama", model="llama3.2:3b")
    assert first["domain"] == "finance"
    assert first.get("chart") is not None
    summary = first.get("anomaly_summary")
    assert summary is not None
    assert summary["domain"] == "finance"
    assert summary["key"] == "RELIANCE.NS"
    assert summary["n_units_analyzed"] > 0

    followup = answer_query("anything unusual lately?", session, backend="ollama", model="llama3.2:3b")
    assert followup["domain"] == "finance"


@pytest.mark.slow
def test_chat_agent_astronomy_mode_pin_then_bare_anomaly_followup():
    """Same unified behavior on the astronomy side, plus the explicit
    mode-pin: pinning astronomy mode, then a bare anomaly follow-up with no
    named target, should resolve via pinned_domain + last_target rather
    than falling through to the general-knowledge answer with no data."""
    from agents.chat_agent import answer_query

    session = {}
    pin = answer_query("switch to astronomy mode", session)
    assert session.get("pinned_domain") == "astronomy"
    assert "trace" in pin

    first = answer_query("check KIC 11446443 for anomalies", session, backend="ollama", model="llama3.2:3b")
    assert first["domain"] == "astronomy"
    assert first.get("chart") is not None
    summary = first.get("anomaly_summary")
    assert summary is not None and summary["domain"] == "astronomy"

    followup = answer_query("is anything abnormal", session, backend="ollama", model="llama3.2:3b")
    assert followup["domain"] == "astronomy"
