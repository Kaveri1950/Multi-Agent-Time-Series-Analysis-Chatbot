# Audit exports

These CSVs are produced by `python -m audit.warehouse`, which loads the project's evidence into a SQLite database (`data/cache/audit/audit.db`) and runs every query in [audit/queries.sql](../../audit/queries.sql). Each file is the result of one named query. The same command then runs `audit/charts.py`, which draws the charts in `docs/figures/audit_*.png` from these files.

| File | What it answers |
|---|---|
| `control_summary.csv` | Latest data-quality run: each control, pass/fail, severity, failure rate against its limit |
| `control_history.csv` | Every control on every run, for a trend line |
| `routing_accuracy.csv` | How often each model-selection strategy picked the right model, by question category |
| `routing_errors.csv` | The questions the rule-based selector got wrong, which are the exceptions to sample |
| `models_without_evidence.csv` | Registered models that have no measured skill number |
| `var_backtest_outcomes.csv` | Pass rate of regime-conditional vs plain VaR across 40 random portfolios |
| `var_backtest_detail.csv` | One row per portfolio and VaR method |
| `chatbot_answers_by_model.csv` | Chatbot answers per model: count, speed, guard corrections, errors |
| `chatbot_guard_events.csv` | Every answer where the number guard caught a figure not in the computed facts |

The chatbot files fill up as the chatbot is used; every answer is logged by `audit/answer_log.py`. The committed copies come from 16 questions asked with `python -m evaluation.answer_log_run` (risk, forecasts, anomalies, solar, Kepler, and two that no model should answer). In those 16 answers the number guard didn't have to correct anything, so `chatbot_guard_events.csv` has only its header. That's a result, not a missing file. How often the guard catches invented numbers is measured separately in `evaluation/results/guard_eval.json`.

| Chart | Drawn from |
|---|---|
| `audit_dashboard.png` | control_summary, routing_accuracy, routing_errors, var_backtest_outcomes, var_backtest_detail |
| `audit_controls.png` | control_summary |
| `audit_routing.png` | routing_accuracy |
| `audit_var_backtest.png` | var_backtest_detail |
| `audit_var_outcomes.png` | var_backtest_outcomes |

The files open in Excel or any BI tool; every column is plain text or numbers.
