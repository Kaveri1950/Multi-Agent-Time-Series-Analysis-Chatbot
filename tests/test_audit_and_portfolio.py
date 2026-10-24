"""Audit controls, the answer log, the SQL queries, and portfolio risk maths."""

import sqlite3

import numpy as np
import pytest

from audit import answer_log, data_quality, warehouse
from quant.portfolio_risk import normalise, risk_contributions


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


def test_normalise_adds_suffix_and_rescales():
    out = normalise({"tcs": 2, "INFY.NS": 2})
    assert set(out) == {"TCS.NS", "INFY.NS"}
    assert sum(out.values()) == pytest.approx(1.0)
