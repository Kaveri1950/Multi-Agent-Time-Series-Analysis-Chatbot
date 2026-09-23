"""Tests for the solar feature builder, the forecasting models, and the
agent/chatbot integration.

Split from test_solar.py (which covers the data layer) because these test
different risks. The data-layer tests guard against parse regressions; these
guard against the two failure modes that would silently invalidate every
result the project reports: label leakage, and a metric that rewards
exploiting class imbalance.

Run with: pytest tests/test_solar_model.py -v
"""

import numpy as np
import pandas as pd
import pytest


def _two_flare_catalog():
    """One X-class flare and one C-class, four days apart. Small enough that
    a leak is unmissable rather than merely detectable."""
    return pd.DataFrame([{
        "begin_time": pd.Timestamp("2000-01-10 12:00"),
        "peak_time": pd.Timestamp("2000-01-10 12:05"),
        "end_time": pd.Timestamp("2000-01-10 12:30"),
        "flare_class": "X2.0", "class_letter": "X", "peak_flux": 2e-4,
        "duration_min": 30.0, "position": None, "active_region": 12345,
    }, {
        "begin_time": pd.Timestamp("2000-01-14 12:00"),
        "peak_time": pd.Timestamp("2000-01-14 12:05"),
        "end_time": pd.Timestamp("2000-01-14 12:30"),
        "flare_class": "C1.0", "class_letter": "C", "peak_flux": 1e-6,
        "duration_min": 30.0, "position": None, "active_region": 12345,
    }])


# --------------------------------------------------------------------------
# features -- the leakage guard is the thing worth testing
# --------------------------------------------------------------------------

def test_features_do_not_leak_the_label():
    """THE test for this module.

    Day D's features must be built only from flares that began before
    D 00:00, or the model can see its own answer and every score the project
    reports becomes meaningless. Rolling windows include the current row by
    default, so this is easy to get wrong and invisible when you do.
    """
    from solar.features import build_daily_frame

    frame = build_daily_frame(_two_flare_catalog())

    labelled = frame[frame["label"] == 1]
    assert len(labelled) == 1
    assert labelled.iloc[0]["date"] == pd.Timestamp("2000-01-10")

    # the leak check: on its own labelled day, the flare must be invisible
    assert labelled.iloc[0]["n_X_1d"] == 0
    assert labelled.iloc[0]["n_target_1d"] == 0
    assert labelled.iloc[0]["max_flux_1d"] == 0

    # ...and must become visible the very next day
    assert frame.iloc[1]["n_X_1d"] == 1
    assert frame.iloc[1]["n_target_1d"] == 1

    # time-since-last-X walks forward one day at a time
    assert list(frame["hours_since_X"].iloc[1:4]) == [0.0, 24.0, 48.0]


def test_label_is_target_class_only():
    """The label is 'M-class or larger'. A C-class flare must not set it, or
    the model is being trained on a different question than it is asked."""
    from solar.features import build_daily_frame

    frame = build_daily_frame(_two_flare_catalog())
    c_day = frame[frame["date"] == pd.Timestamp("2000-01-14")]
    assert len(c_day) == 1
    assert c_day.iloc[0]["label"] == 0


def test_chronological_split_preserves_order():
    """A random split leaks the future into the past. Every training date
    must fall strictly before every test date."""
    from solar.features import chronological_split

    frame = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=100),
                          "label": [0, 1] * 50, "feat": range(100)})
    train, test = chronological_split(frame, test_fraction=0.2)

    assert len(train) == 80 and len(test) == 20
    assert train["date"].max() < test["date"].min()


# --------------------------------------------------------------------------
# metrics -- why TSS instead of accuracy
# --------------------------------------------------------------------------

def test_tss_gives_no_credit_to_constant_forecasts():
    """The reason the project reports TSS and not accuracy: at a ~14%
    positive rate, "never a flare" is 86% accurate and completely useless.
    TSS must score any constant forecast at exactly zero."""
    from solar.flare_model import tss

    y = np.array([0] * 86 + [1] * 14)
    assert tss(y, np.zeros(100), 0.5) == 0.0        # always "no flare"
    assert tss(y, np.ones(100), 0.5) == 0.0         # always "flare"
    assert tss(y, y.astype(float), 0.5) == 1.0      # a perfect forecast


def test_brier_rewards_calibration():
    from solar.flare_model import brier

    y = np.array([1, 1, 0, 0])
    assert brier(y, np.array([1.0, 1.0, 0.0, 0.0])) == 0.0
    # a confidently wrong forecast is the worst possible
    assert brier(y, np.array([0.0, 0.0, 1.0, 1.0])) == 1.0


# --------------------------------------------------------------------------
# Hawkes -- the closed form, and the physics it claims
# --------------------------------------------------------------------------

def _hand_built_model(mu=0.002, alpha=0.3, beta=0.05, gamma=0.0):
    from solar.flare_model import HawkesModel

    m = HawkesModel()
    m.mu, m.alpha, m.beta, m.gamma = mu, alpha, beta, gamma
    m.marks_ = np.array([0.0])
    return m


def test_hawkes_closed_form_agrees_with_simulation():
    """Cross-validates the analytic integral against the independent Monte
    Carlo path.

    probability_within() is exact and closed-form; expected_count() is
    simulated. They measure different quantities in general, but in a
    low-intensity window where two events are very unlikely, P(>=1) and E[N]
    must nearly coincide. If the compensator integral were wrong, these two
    would disagree -- and neither would flag it alone.
    """
    m = _hand_built_model()
    hist_t = np.array([0.0, 10.0])
    hist_m = np.array([0.0, 0.0])

    p = m.probability_within(hist_t, hist_m, now=50.0, horizon_hours=24.0)
    e = m.expected_count(hist_t, hist_m, now=50.0, horizon_hours=24.0,
                         n_sims=4000, rng=np.random.default_rng(7))

    assert 0.0 < p < 1.0
    assert abs(p - e) < 0.05, f"closed form {p:.4f} vs simulation {e:.4f}"


def test_hawkes_probability_rises_after_a_flare():
    """The model's entire physical claim: a recent flare raises the chance of
    the next one. If this fails, the self-excitation term is inert and the
    model is a Poisson process wearing a costume."""
    m = _hand_built_model(mu=0.002, alpha=0.8, beta=0.026, gamma=0.04)

    just_flared = m.probability_within(np.array([100.0]), np.array([0.0]),
                                       now=101.0, horizon_hours=24.0)
    long_quiet = m.probability_within(np.array([100.0]), np.array([0.0]),
                                      now=10_000.0, horizon_hours=24.0)

    assert just_flared > long_quiet
    # after a long quiet stretch the excitation has fully decayed, so the
    # forecast must fall back to the fitted background rate
    background = 1.0 - np.exp(-m.mu * 24.0)
    assert abs(long_quiet - background) < 0.01


def test_hawkes_bigger_flares_excite_more():
    """Marks must matter -- an X10 should excite more than an M1. A standard
    unmarked Hawkes process cannot express this, which is why the mark
    weighting is there."""
    m = _hand_built_model(mu=0.002, alpha=0.8, beta=0.026, gamma=0.5)

    after_m1 = m.probability_within(np.array([0.0]), np.array([0.0]), now=1.0)
    after_x10 = m.probability_within(np.array([0.0]), np.array([2.0]), now=1.0)
    assert after_x10 > after_m1


def test_hawkes_probability_is_monotone_in_horizon():
    """A longer window cannot be less likely to contain an event."""
    m = _hand_built_model(mu=0.002, alpha=0.5, beta=0.03, gamma=0.0)
    probs = [m.probability_within(np.array([0.0]), np.array([0.0]), now=5.0,
                                  horizon_hours=h) for h in (1, 6, 24, 72)]
    assert probs == sorted(probs)
    assert all(0.0 <= p <= 1.0 for p in probs)


@pytest.mark.slow
def test_hawkes_fit_recovers_self_excitation_on_real_data():
    """Fit on the real GOES catalogue and check the parameters are physically
    sensible, not merely finite: sub-critical but strongly self-exciting, with
    a decay on the order of a day, which is roughly how long an active region
    stays primed after firing."""
    from solar.flare_data import load_historical_catalog
    from solar.flare_model import HawkesModel, catalog_to_hawkes_input

    catalog = load_historical_catalog(2012, 2014)
    times, marks, horizon, _ = catalog_to_hawkes_input(catalog)
    assert len(times) > 100

    m = HawkesModel().fit(times, marks, horizon)
    assert m._converged
    assert 0.0 < m.alpha < 1.0       # sub-critical: not explosive
    assert m.alpha > 0.3             # genuinely self-exciting
    assert 1.0 < m.half_life_hours < 200.0


def test_persistence_baseline_is_the_stated_rule():
    """Persistence must be exactly 'was there an M+ flare in the last 24h'.
    It is the floor the report compares everything against, so it should not
    quietly become something cleverer."""
    from solar.flare_model import persistence_forecast

    test = pd.DataFrame({"n_target_1d": [0.0, 1.0, 3.0, 0.0]})
    assert list(persistence_forecast(test)) == [0.0, 1.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# integration -- routing and the shared contract
# --------------------------------------------------------------------------

def test_router_sends_solar_to_solar():
    """Solar is the one domain with no entity id, so it is matched on subject
    words -- and must not steal stellar-flare or ticker questions."""
    from agents.nodes.router import route_node

    def domain_of(text):
        return route_node({"target": text, "trace": []})["domain"]

    assert domain_of("solar flare activity") == "solar"
    assert domain_of("space weather") == "solar"
    assert domain_of("sunspot number") == "solar"
    assert domain_of("geomagnetic storm") == "solar"
    assert domain_of("KIC 11446443") == "astronomy"
    assert domain_of("AAPL") == "finance"
    assert domain_of("RELIANCE.NS") == "finance"


def test_chat_solar_detector_does_not_steal_stellar_questions():
    """A flare on another star is the astronomy domain. A KIC/TIC id in the
    message must veto solar routing even when 'the sun' also appears."""
    from agents.chat_agent import _looks_solar

    assert _looks_solar("will there be a solar flare tomorrow")
    assert _looks_solar("what is the flare probability")
    assert _looks_solar("geomagnetic storm risk")
    assert _looks_solar("will there be a CME")
    assert not _looks_solar("any anomalies in KIC 8462852")
    assert not _looks_solar("compare the sun to KIC 8462852")
    assert not _looks_solar("what is RELIANCE.NS doing")
    assert not _looks_solar("what does RSI mean")


def test_forecast_contract_always_carries_caveats_and_baseline():
    """The contract's job is to make honesty structural rather than a matter
    of remembering: a consumer must not be able to read the probability
    without also receiving the caveats and something to judge it against."""
    from agents.forecast_shared import format_for_prompt, from_solar_result

    raw = {
        "as_of": "2026-08-21T00:00:00+00:00",
        "method": "hawkes-exponential-marked",
        "forecast": {"label": "M-class or larger solar flare", "probability": 0.57,
                     "horizon": "next 24 hours", "confidence": "LOW",
                     "basis": ["4 M+ flares in the last 7 days"]},
        "baselines": [{"name": "NOAA SWPC operational forecast", "probability": 0.40,
                       "note": "the real benchmark"}],
        "skill": {"hawkes_tss_final_block": 0.572},
        "caveats": ["not a prediction of when a flare will occur"],
    }
    summary = from_solar_result(raw)
    assert summary["domain"] == "solar"
    assert summary["forecast"]["probability"] == 0.57

    text = format_for_prompt(summary)
    for expected in ("57.0%", "next 24 hours", "NOAA SWPC operational forecast",
                     "40.0%", "CAVEATS", "not a prediction of when a flare will occur"):
        assert expected in text, f"missing from prompt block: {expected}"


def test_solar_prompt_forbids_stating_a_flare_time():
    """The one rule that must never be relaxed. Checked as a property of the
    prompt text so a future edit cannot quietly drop it."""
    from agents.prompts import SOLAR_FORECAST_SYSTEM_PROMPT

    p = SOLAR_FORECAST_SYSTEM_PROMPT.lower()
    assert "never state or imply a specific time" in p
    assert "horizon" in p
    assert "x-class" in p          # and that X-class is not forecast


@pytest.mark.slow
def test_solar_node_produces_a_live_forecast():
    """The node both the graph and the chatbot call, end to end against live
    NOAA data. Asserts shape and plausibility, never a specific value -- the
    Sun does not hold still for a test suite."""
    from agents.nodes.solar_flare_node import solar_flare_node, solar_forecast_block

    state = solar_flare_node({"domain": "solar", "trace": []})
    result = state["solar_result"]
    assert "error" not in result, result.get("error")

    p = result["forecast"]["probability"]
    assert p is not None and 0.0 <= p <= 1.0
    assert result["forecast"]["horizon"] == "next 24 hours"
    assert result["caveats"], "caveats must never be empty"
    assert any(b["name"].startswith("NOAA") for b in result["baselines"])

    block = solar_forecast_block(state)
    assert "FORECAST:" in block and "CAVEATS" in block
