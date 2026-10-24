"""Audit controls, the answer log, the SQL queries, and portfolio risk maths."""

import sqlite3

import numpy as np
import pytest

from audit import answer_log, data_quality, warehouse
from quant.portfolio_risk import normalise, risk_contributions, sector_breakdown


def test_check_status_follows_threshold():
    assert data_quality._check("x", "", 0, 100, 0.0, [])["status"] == "pass"
    assert data_quality._check("x", "", 1, 100, 0.0, [])["status"] == "fail"
    assert data_quality._check("x", "", 1, 1000, 0.001, [])["status"] == "pass"
    assert data_quality._check("x", "", 0, 0, 0.0, [])["fail_rate"] == 0.0


def test_parse_reads_selector_and_guard_from_trace():
    result = {"domain": "finance", "answer": "abc", "trace": [
        "[selector] task=risk_assessment (rule)",
        "[selector] filter_only -> finance.risk_composition (best measured skill)",
        "[guard] corrected 2 figure(s) not in the computed facts: 12.5, 3%"]}
    row = answer_log.parse(result)
    assert row["task"] == "risk_assessment"
    assert row["model_id"] == "finance.risk_composition"
    assert row["selector_strategy"] == "filter_only"
    assert row["guard_flags"] == 2 and row["guard_figures"] == "12.5, 3%"


def test_parse_records_every_model_in_a_risk_composition():
    row = answer_log.parse({"domain": "finance", "trace": [
        "[risk] volatility: 'how volatile' -> task forecast_volatility -> finance.volatility_forecast (2 feasible)",
        "[risk] tail_risk: 'what is the VaR' -> task assess_risk -> finance.risk_profile (1 feasible)",
        "[risk] anomalies: 'unusual days' -> task detect_anomaly -> finance.anomaly_detectors (3 feasible)"]})
    assert row["selector_strategy"] == "composition"
    assert row["model_id"] == ("finance.volatility_forecast + finance.risk_profile + "
                               "finance.anomaly_detectors")


def test_parse_labels_paths_that_skip_the_selector():
    gp = answer_log.parse({"trace": ["[chat] computed 10-trading-day Gaussian Process forecast for 'SBIN.NS': x"]})
    market = answer_log.parse({"trace": ["[risk] market assessment computed 2026-10-24, 50 names ranked"]})
    assert (gp["model_id"], gp["selector_strategy"]) == ("gaussian_process_forecast", "fixed path")
    assert market["model_id"] == "market_risk_ranking"


def test_tests_write_to_a_private_log():
    from pathlib import Path
    real = Path(__file__).parent.parent / "data" / "cache" / "audit"
    assert answer_log.DB_PATH.parent.resolve() != real.resolve()


def test_parse_handles_unrouted_and_missing_results():
    assert answer_log.parse({"trace": ["[chat] something"]})["model_id"] is None
    assert answer_log.parse(None)["guard_flags"] == 0


def test_record_writes_rows_and_never_raises(tmp_path):
    db = tmp_path / "a.db"
    answer_log.record("is TCS risky", {"domain": "finance", "answer": "hi", "trace": []}, 1.2, path=db)
    answer_log.record("boom", None, 0.1, error="RuntimeError()", path=db)
    rows = sqlite3.connect(db).execute("SELECT question, status, answer_chars FROM chat_answers").fetchall()
    assert rows == [("is TCS risky", "ok", 2), ("boom", "error", 0)]
    answer_log.record("x", {"trace": [object()]}, 0.0, path=tmp_path / "missing" / "dir" / "b.db")


def test_every_named_query_is_valid_sql():
    con = sqlite3.connect(":memory:")
    con.execute(answer_log.SCHEMA)
    con.executescript("""
        CREATE TABLE data_quality_checks (run_at, check_name, description, status, severity,
            rows_tested, rows_failed, fail_rate, threshold, examples);
        CREATE TABLE model_registry (id, name, domain, task, method_family, requires,
            skill_metric, skill_value);
        CREATE TABLE routing_eval (strategy, query, domain, category, expected, chosen, correct,
            task, fallback_used, seconds);
        CREATE TABLE var_backtests (portfolio, holdings, method, status, breach_rate, n_days,
            kupiec_p, christoffersen_p, volatility_pct, diversification_ratio);""")
    qs = warehouse.queries()
    assert len(qs) >= 9
    for name, sql in qs.items():
        con.execute(sql).fetchall()


def test_euler_contributions_sum_to_portfolio_volatility():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.normal(size=(200, 5))
        cov, w = np.cov(x, rowvar=False), rng.dirichlet(np.ones(5))
        assert risk_contributions(cov, w).sum() == pytest.approx(np.sqrt(w @ cov @ w), abs=1e-12)


def test_sector_breakdown_sums_holdings_and_keeps_unknowns():
    rows = [{"ticker": "TCS", "weight_pct": 30.0, "risk_contribution_pct": 35.0},
            {"ticker": "INFY", "weight_pct": 20.0, "risk_contribution_pct": 25.0},
            {"ticker": "NOT_A_STOCK", "weight_pct": 50.0, "risk_contribution_pct": 40.0}]
    out = {r["sector"]: r for r in sector_breakdown(rows)}
    assert out["Information Technology"]["weight_pct"] == 50.0
    assert out["Information Technology"]["risk_contribution_pct"] == 60.0
    assert out["Unclassified"]["holdings"] == ["NOT_A_STOCK"]
    assert sum(r["risk_contribution_pct"] for r in out.values()) == pytest.approx(100.0)


def test_normalise_adds_suffix_and_rescales():
    out = normalise({"tcs": 2, "INFY.NS": 2})
    assert set(out) == {"TCS.NS", "INFY.NS"}
    assert sum(out.values()) == pytest.approx(1.0)
