# Power BI exports

These CSVs are produced by `python -m audit.warehouse`, which loads the project's evidence into a SQLite database (`data/cache/audit/audit.db`) and runs every query in [audit/queries.sql](../../audit/queries.sql). Each file is the result of one named query.

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

The chatbot files fill up as the chatbot is used; every answer is logged by `audit/answer_log.py`.

## Building the report in Power BI Desktop

1. **Get data → Text/CSV**, select each file and click **Load**. Alternatively use **Get data → Folder** on this directory and combine.
2. Suggested pages:
   - **Controls:** a table of `control_summary` with conditional formatting on `status` (fail in dark blue, pass in light grey), plus a card counting failed controls.
   - **Model selection:** a clustered bar of `accuracy_pct` by `strategy`, with `category` as the legend. Use `routing_errors` as a drill-through table.
   - **VaR validation:** a stacked bar of passed/failed from `var_backtest_outcomes`, and a scatter of `breach_pct` against `volatility_pct` from `var_backtest_detail` with a constant line at 5.
   - **Chatbot audit:** a card for the answers corrected by the guard, and a table of `chatbot_guard_events`.
3. To refresh, re-run `python -m audit.warehouse` and click **Refresh** in Power BI.

The same files open in Excel; every column is plain text or numbers.
