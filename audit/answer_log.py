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
    status TEXT NOT NULL
)"""

_TASK = re.compile(r"\[selector\] task=(\S+)")
_MODEL = re.compile(r"\[selector\] (filter_only|hybrid|llm_only) -> (\S+)")
_GUARD = re.compile(r"\[guard\] corrected (\d+) figure\(s\)[^:]*: (.*)")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def parse(result: dict | None) -> dict:
    """Pull task, model and guard outcome out of an answer's trace."""
    trace = "\n".join(str(t) for t in (result or {}).get("trace", []))
    task, model, guard = _TASK.search(trace), _MODEL.search(trace), _GUARD.search(trace)
    return {"domain": (result or {}).get("domain"),
            "task": task.group(1) if task else None,
            "selector_strategy": model.group(1) if model else None,
            "model_id": (model.group(2) if model else None),
            "guard_flags": int(guard.group(1)) if guard else 0,
            "guard_figures": guard.group(2).strip() if guard else None}


def record(question: str, result: dict | None, seconds: float, error: str | None = None,
           path: Path = DB_PATH) -> None:
    try:
        row = parse(result)
        with connect(path) as con:
            con.execute(
                "INSERT INTO chat_answers (asked_at, question, domain, task, model_id, "
                "selector_strategy, guard_flags, guard_figures, seconds, answer_chars, status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), (question or "")[:300],
                 row["domain"], row["task"], row["model_id"], row["selector_strategy"],
                 row["guard_flags"], row["guard_figures"], round(seconds, 3),
                 len((result or {}).get("answer") or ""), "error" if error else "ok"))
    except Exception:  # noqa: BLE001 -- the log must never take the chatbot down
        pass
