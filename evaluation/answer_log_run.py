"""Ask the chatbot a fixed set of real questions so the answer log has rows.

Every answer goes through agents.chat_agent.answer_query, which writes one row
per answer to the audit log (audit/answer_log.py). Errors are logged too.
The questions span all domains and include ones no model should answer.

Run:  python -m evaluation.answer_log_run          (slow: the LLM runs on CPU)
"""

from __future__ import annotations

import os
import time

QUESTIONS = [
    "is TCS risky right now?",
    "how risky is HDFCBANK?",
    "what is the VaR of RELIANCE.NS?",
    "which stocks are the riskiest in the market today?",
    "how volatile will INFY be tomorrow?",
    "were there any unusual trading days for ITC.NS recently?",
    "forecast TATASTEEL.NS for the next 10 days",
    "what will SBIN.NS close at in 30 days",
    "will there be a solar flare in the next 24 hours?",
    "how many flares should we expect this week?",
    "does KIC 11446443 have planets?",
    "find anomalies in the light curve of KIC 11446443",
    "what datasets do you use?",
    "is BAJFINANCE.NS risky?",
    "what's the weather in Mumbai?",
    "should I buy ADANIENT?",
]


def main():
    from agents.chat_agent import answer_query

    model = os.environ.get("BTP_OLLAMA_MODEL", "llama3.2:3b")
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.time()
        try:
            r = answer_query(q, {}, backend="ollama", model=model)
            status = f"ok, {len(r.get('answer') or '')} chars"
        except Exception as exc:  # noqa: BLE001 -- logged by answer_query; keep going
            status = f"error: {exc!r}"[:120]
        print(f"{i:>2}/{len(QUESTIONS)} {time.time() - t0:6.0f}s  {status}  | {q}", flush=True)


if __name__ == "__main__":
    main()
