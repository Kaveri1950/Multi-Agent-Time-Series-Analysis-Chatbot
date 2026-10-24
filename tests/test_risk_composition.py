"""The composed risk assessment: routing is the selector's, and no figure is bare.

Runs offline against the cached panels and the persisted volatility model.
"""

import warnings

import pytest

from agents.risk_composition import (FAILED, FAN_OUT, PASSED, STATUSES, UNTESTED,
                                     _var_verdict, assess_stock_risk, figure, to_text)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def tcs():
    return assess_stock_risk("TCS")


def test_each_sub_question_reaches_its_task(tcs):
    assert [r["task"] for r in tcs["routing"]] == [s["task"] for s in FAN_OUT]
    assert all(r["model_id"] for r in tcs["routing"])
    assert not any(r["fallback_used"] for r in tcs["routing"])


@pytest.mark.needs_data
def test_every_figure_carries_a_status_and_its_evidence(tcs):
    assert tcs["ticker"] == "TCS.NS"
    for role, comp in tcs["components"].items():
        assert "error" not in comp, f"{role}: {comp.get('error')}"
        assert comp["figures"], role
        for f in comp["figures"]:
            assert f["status"] in STATUSES, f
            assert len(f["evidence"]) > 20, f
            assert f["value"] is not None, f


@pytest.mark.needs_data
def test_limits_are_in_the_output_not_a_footnote(tcs):
    text = to_text(tcs)
    for stated in ("70.9%", "74.1%", "0.247", "0.533", "never which way"):
        assert stated in text, stated


@pytest.mark.needs_data
def test_intervals_nest_and_contain_the_forecast(tcs):
    point = figure(tcs, "volatility", "forecast_vol_pct")["value"]
    lo80, hi80 = figure(tcs, "volatility", "interval_80_pct")["value"]
    lo95, hi95 = figure(tcs, "volatility", "interval_95_pct")["value"]
    assert lo95 < lo80 < point < hi80 < hi95


def test_var_verdict_is_the_stocks_own_not_the_proxys():
    row = {"n_days": 900, "breach_rate": 0.084, "kupiec_p": 0.0001, "christoffersen_p": 0.9}
    status, why = _var_verdict({"testable": True, "level": 0.95, "conditional": row})
    assert status == FAILED and "understates" in why
    row = {**row, "breach_rate": 0.051, "kupiec_p": 0.8}
    assert _var_verdict({"testable": True, "level": 0.95, "conditional": row})[0] == PASSED
    assert _var_verdict({"testable": False, "why": "too short"})[0] == UNTESTED


@pytest.mark.needs_data
def test_a_ticker_outside_the_volatility_panel_degrades_one_component():
    a = assess_stock_risk("PIDILITIND.NS")
    assert "error" in a["components"]["volatility"]
    assert "error" not in a["components"]["tail_risk"]
    assert "error" not in a["components"]["regime"]


def test_a_model_without_a_runner_is_reported_not_substituted(monkeypatch):
    import agents.risk_composition as rc

    monkeypatch.delitem(rc.RUNNERS, "finance.market_regime")
    a = rc.assess_stock_risk("TCS")
    assert "no runner" in a["components"]["regime"]["error"]


@pytest.fixture(scope="module")
def market():
    from agents.risk_composition import assess_market_risk

    return assess_market_risk(["TCS.NS", "IRFC.NS", "ZEEL.NS", "LT.NS", "PIDILITIND.NS"])


@pytest.mark.needs_data
def test_market_is_ranked_by_forecast_volatility_alone(market):
    vols = [r["forecast_vol_pct"] for r in market["ranking"]]
    assert vols == sorted(vols, reverse=True)
    assert [r["rank"] for r in market["ranking"]] == [1, 2, 3, 4]
    assert market["ranked_by"] == "forecast_vol_pct"


@pytest.mark.needs_data
def test_market_reports_what_it_could_not_rank(market):
    assert [u["ticker"] for u in market["unranked"]] == ["PIDILITIND.NS"]


@pytest.mark.needs_data
def test_market_states_how_little_the_model_adds_to_the_order(market):
    ranking = next(f for f in market["figures"] if f["key"] == "ranking")
    assert "trailing 66-day" in ranking["evidence"] and "small" in ranking["evidence"]
    for f in market["figures"]:
        assert f["status"] in STATUSES and len(f["evidence"]) > 20


@pytest.mark.needs_data
def test_each_row_carries_its_own_var_verdict(market):
    by = {r["ticker"]: r for r in market["ranking"]}
    assert by["TCS.NS"]["var_status"] == PASSED
    assert by["IRFC.NS"]["var_status"] == FAILED


@pytest.mark.parametrize("question, market", [
    ("which stocks are riskiest right now", True),
    ("what are the most volatile stocks", True),
    ("how risky is the market", True),
    ("which stock is safest to hold", True),
    ("is TCS risky", False),
    ("how volatile is reliance", False),
])
def test_market_questions_route_before_the_single_stock_path(question, market):
    from agents.chat_agent import _MARKET_RISK_PATTERN

    assert bool(_MARKET_RISK_PATTERN.search(question)) is market


@pytest.mark.needs_data
def test_chat_facts_state_what_var_means_and_unsign_volatility_bands(tcs):
    from agents.risk_composition import chat_facts, status_table

    facts = chat_facts(tcs, context="median 21.0%")
    assert "AT LEAST" in facts and "not a maximum" in facts
    band = next(l for l in facts.splitlines() if l.startswith("- 80% interval"))
    assert "+" not in band.split("[")[0]
    assert "COMPARISON WITH THE MARKET" in facts
    table = status_table(tcs)
    assert "80% range for the next close" in table and "...as a move" not in table
