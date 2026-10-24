"""An audit trail for every chatbot answer.

For each answer: when, which domain and task, which model the selector chose,
whether the number guard had to correct invented figures, how long it took,
and whether it errored. The question text is kept (truncated) because an
audit trail that cannot say what was asked is not much of a trail; the log
lives under data/cache/, which is never committed.

Logging must never break the chatbot, so every write is wrapped.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "cache" / "audit" / "audit.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at TEXT NOT NULL,
    question TEXT,
    domain TEXT,
    task TEXT,
    model_id TEXT,
    selector_strategy TEXT,
    guard_flags INTEGER NOT NULL DEFAULT 0,
    guard_figures TEXT,
    seconds REAL,
    answer_chars INTEGER,
    status TEXT NOT NULL,
    trace TEXT
)"""

_TASK = re.compile(r"\[selector\] task=(\S+)")
_MODEL = re.compile(r"\[selector\] (filter_only|hybrid|llm_only) -> (\S+)")
_GUARD = re.compile(r"\[guard\] corrected (\d+) figure\(s\)[^:]*: (.*)")
# the risk composition routes five sub-questions through the selector, one line each
_RISK = re.compile(r"\[risk\] [^:]+: .* -> task (\S+) -> (\S+) \(")
# paths that do not go through the selector, logged as such rather than left blank
_FIXED = ((re.compile(r"\[risk\] market assessment"), "assess_market_risk", "market_risk_ranking"),
          (re.compile(r"Gaussian Process forecast|reusing cached \d+-day forecast"),
           "forecast_value", "gaussian_process_forecast"))


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def parse(result: dict | None) -> dict:
    """Pull task, model and guard outcome out of an answer's trace."""
    trace = "\n".join(str(t) for t in (result or {}).get("trace", []))
    task, model, guard = _TASK.search(trace), _MODEL.search(trace), _GUARD.search(trace)
    row = {"domain": (result or {}).get("domain"),
           "task": task.group(1) if task else None,
           "selector_strategy": model.group(1) if model else None,
           "model_id": (model.group(2) if model else None),
           "guard_flags": int(guard.group(1)) if guard else 0,
           "guard_figures": guard.group(2).strip() if guard else None,
           "trace": trace or None}
    risk = _RISK.findall(trace)
    if risk:
        row.update(task="assess_risk", selector_strategy="composition",
                   model_id=" + ".join(dict.fromkeys(m for _, m in risk)))
    elif row["model_id"] is None:
        for pattern, fixed_task, fixed_model in _FIXED:
            if pattern.search(trace):
                row.update(task=fixed_task, model_id=fixed_model, selector_strategy="fixed path")
                break
    return row


def record(question: str, result: dict | None, seconds: float, error: str | None = None,
           path: Path | None = None) -> None:
    try:
        row = parse(result)
        with connect(path or DB_PATH) as con:
            con.execute(
                "INSERT INTO chat_answers (asked_at, question, domain, task, model_id, "
                "selector_strategy, guard_flags, guard_figures, seconds, answer_chars, status, trace) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), (question or "")[:300],
                 row["domain"], row["task"], row["model_id"], row["selector_strategy"],
                 row["guard_flags"], row["guard_figures"], round(seconds, 3),
                 len((result or {}).get("answer") or ""), "error" if error else "ok",
                 (row["trace"] or "")[:20000] or (error or None)))
    except Exception:  # noqa: BLE001 -- the log must never take the chatbot down
        pass
