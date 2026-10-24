-- Audit queries over data/cache/audit/audit.db (built by `python -m audit.warehouse`).
-- Each block starts with "-- name: <id>"; audit/warehouse.py runs every one and
-- writes its result to exports/powerbi/<id>.csv.

-- name: control_summary
-- Latest data-quality run: which controls failed, how badly, and how serious.
SELECT check_name, status, severity, rows_failed, rows_tested,
       ROUND(100.0 * fail_rate, 4) AS fail_pct, ROUND(100.0 * threshold, 4) AS limit_pct
FROM data_quality_checks
WHERE run_at = (SELECT MAX(run_at) FROM data_quality_checks)
ORDER BY status DESC, CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END;

-- name: control_history
-- Pass/fail of every control over every run, for a trend line.
SELECT run_at, check_name, status, fail_rate FROM data_quality_checks ORDER BY run_at, check_name;

-- name: routing_accuracy
-- How often each selection strategy picked the right model, overall and by category.
SELECT strategy, category, COUNT(*) AS questions, SUM(correct) AS right,
       ROUND(100.0 * AVG(correct), 2) AS accuracy_pct, ROUND(AVG(seconds), 3) AS avg_seconds
FROM routing_eval
GROUP BY strategy, category
UNION ALL
SELECT strategy, 'ALL', COUNT(*), SUM(correct), ROUND(100.0 * AVG(correct), 2), ROUND(AVG(seconds), 3)
FROM routing_eval GROUP BY strategy
ORDER BY strategy, category;

-- name: routing_errors
-- Every question the rule-based filter got wrong: the exceptions an auditor would sample.
SELECT domain, category, query, expected, chosen
FROM routing_eval WHERE strategy = 'filter_only' AND correct = 0
ORDER BY domain, category;

-- name: models_without_evidence
-- Models in the registry that carry no measured skill number: claims without evidence.
SELECT id, domain, task, method_family FROM model_registry WHERE skill_value IS NULL ORDER BY domain;

-- name: var_backtest_outcomes
-- Portfolio VaR backtests: pass rate of each VaR method across random portfolios.
SELECT method, COUNT(*) AS portfolios, SUM(status = 'passed') AS passed,
       SUM(status = 'failed') AS failed, ROUND(100.0 * AVG(status = 'passed'), 1) AS pass_pct,
       ROUND(100.0 * AVG(breach_rate), 2) AS avg_breach_pct
FROM var_backtests GROUP BY method;

-- name: var_backtest_detail
-- One row per portfolio and method, for drill-down.
SELECT portfolio, method, status, ROUND(100.0 * breach_rate, 2) AS breach_pct, n_days,
       kupiec_p, christoffersen_p, volatility_pct, diversification_ratio
FROM var_backtests ORDER BY portfolio, method;

-- name: chatbot_answers_by_model
-- Chatbot answers by model chosen: volume, speed, and how often the number guard intervened.
SELECT COALESCE(model_id, '(not routed)') AS model_id, COALESCE(domain, '-') AS domain,
       COUNT(*) AS answers, ROUND(AVG(seconds), 2) AS avg_seconds,
       SUM(guard_flags > 0) AS answers_corrected, SUM(status = 'error') AS errors
FROM chat_answers GROUP BY model_id, domain ORDER BY answers DESC;

-- name: chatbot_guard_events
-- Every answer where the guard caught figures not in the computed facts (question text omitted).
SELECT asked_at, domain, model_id, guard_flags, guard_figures
FROM chat_answers WHERE guard_flags > 0 ORDER BY asked_at DESC;
