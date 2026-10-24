"""Load the project's evidence into one SQLite database, run the audit queries,
and export each result as a CSV that Power BI (or Excel) can read directly.

Tables
  data_quality_checks  one row per control per run (appended, so history builds up)
  model_registry       every model, its task and its measured skill number
  routing_eval         one row per labelled question per selection strategy
  var_backtests        portfolio VaR backtests, per random portfolio and method
  chat_answers         the chatbot audit log (written live by audit/answer_log.py)

Run:  python -m audit.warehouse          (runs the data-quality checks first)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from audit.answer_log import DB_PATH, connect

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "evaluation" / "results"
QUERIES = Path(__file__).parent / "queries.sql"
EXPORTS = ROOT / "exports" / "powerbi"
ROUTING_FILE = RESULTS / "selection_with_volatility.json"


def load(con) -> dict:
    from agents.model_registry import REGISTRY
    from audit import data_quality

    counts = {}
    con.executescript("""
    CREATE TABLE IF NOT EXISTS data_quality_checks (run_at TEXT, check_name TEXT, description TEXT,
        status TEXT, severity TEXT, rows_tested INTEGER, rows_failed INTEGER, fail_rate REAL,
        threshold REAL, examples TEXT);
    DROP TABLE IF EXISTS model_registry;
    CREATE TABLE model_registry (id TEXT PRIMARY KEY, name TEXT, domain TEXT, task TEXT,
        method_family TEXT, requires TEXT, skill_metric TEXT, skill_value REAL);
    DROP TABLE IF EXISTS routing_eval;
    CREATE TABLE routing_eval (strategy TEXT, query TEXT, domain TEXT, category TEXT,
        expected TEXT, chosen TEXT, correct INTEGER, task TEXT, fallback_used INTEGER, seconds REAL);
    DROP TABLE IF EXISTS var_backtests;
    CREATE TABLE var_backtests (portfolio INTEGER, holdings TEXT, method TEXT, status TEXT,
        breach_rate REAL, n_days INTEGER, kupiec_p REAL, christoffersen_p REAL,
        volatility_pct REAL, diversification_ratio REAL);
    """)

    dq = data_quality.run()
    (RESULTS / "data_quality.json").write_text(json.dumps(dq, indent=2, default=str))
    con.executemany("INSERT INTO data_quality_checks VALUES (?,?,?,?,?,?,?,?,?,?)",
                    [(dq["generated_at"], c["check"], c["description"], c["status"], c["severity"],
                      c["rows_tested"], c["rows_failed"], c["fail_rate"], c["threshold"],
                      json.dumps(c["examples"], default=str)) for c in dq["checks"]])
    counts["data_quality_checks"] = len(dq["checks"])

    rows = []
    for m in REGISTRY:
        skill = m.get("skill") or {}
        rows.append((m["id"], m.get("name"), m.get("domain"), m.get("task"), m.get("method_family"),
                     ",".join(m.get("requires", [])), skill.get("metric"), skill.get("value")))
    con.executemany("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?)", rows)
    counts["model_registry"] = len(rows)

    sel = json.loads(ROUTING_FILE.read_text())
    rows = [(strategy, r["query"], r["domain"], r["category"], r["expected"], r["chosen"],
             int(bool(r["correct"])), r.get("task"), int(bool(r.get("fallback_used"))),
             r.get("seconds"))
            for strategy, items in sel["rows"].items() for r in items]
    con.executemany("INSERT INTO routing_eval VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    counts["routing_eval"] = len(rows)

    pv = json.loads((RESULTS / "portfolio_validation.json").read_text())
    rows = [(p["portfolio"], " ".join(p["holdings"]), method, p[key]["status"],
             p[key]["breach_rate"], p[key]["n_days"], p[key]["kupiec_p"],
             p[key]["christoffersen_p"], p["volatility_pct"], p["diversification_ratio"])
            for p in pv["portfolios"]
            for method, key in (("regime-conditional", "regime"), ("unconditional", "unconditional"))]
    con.executemany("INSERT INTO var_backtests VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    counts["var_backtests"] = len(rows)
    counts["chat_answers"] = con.execute("SELECT COUNT(*) FROM chat_answers").fetchone()[0]
    return counts


def queries() -> dict[str, str]:
    text = QUERIES.read_text(encoding="utf-8")
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def export(con) -> dict[str, int]:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sql in queries().items():
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        data = cur.fetchall()
        with open(EXPORTS / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            w.writerows(data)
        out[name] = len(data)
    return out


def main():
    con = connect(DB_PATH)
    with con:
        counts = load(con)
    print(f"Loaded into {DB_PATH}:")
    for t, n in counts.items():
        print(f"  {t:<20} {n:>5} rows")
    print(f"Queries -> {EXPORTS}:")
    for name, n in export(con).items():
        print(f"  {name:<26} {n:>4} rows")
    for row in con.execute(queries()["var_backtest_outcomes"]):
        print("  ", row)
    con.close()


if __name__ == "__main__":
    main()
