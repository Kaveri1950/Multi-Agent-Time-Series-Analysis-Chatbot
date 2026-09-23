"""Company name -> ticker resolution via Yahoo Finance's free public
search/autocomplete endpoint (query1.finance.yahoo.com/v1/finance/search)
-- no API key, same backend yfinance itself talks to.

This is what lets the chatbot handle "reliance" instead of requiring the
exact ticker "RELIANCE.NS", and lets it ask for disambiguation when a name
is genuinely ambiguous -- e.g. "reliance" real-world matches BOTH Reliance
Industries (RELIANCE.NS) and the unrelated Reliance Infrastructure
(RELINFRA.NS), plus several foreign-exchange-listed companies with
"Reliance" in the name. Silently picking one would be a real (if
occasional) correctness bug, not a UX nicety to skip.
"""

import re

import requests

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"

# Only trust input as an already-correct ticker when it has an explicit
# exchange suffix (RELIANCE.NS, TCS.BO, BRK.B) or is an index/astronomy
# identifier (^NSEI, KIC 11446443) -- anything else (bare "TCS", "AAPL",
# "reliance") goes through search_ticker() so typos/ambiguity/missing
# suffixes are actually caught instead of silently mis-resolving.
_EXPLICIT_PATTERN = re.compile(r"^\^|\.[A-Za-z]{1,3}$|^(KIC|TIC|EPIC)\s*\d+", re.IGNORECASE)


def looks_like_explicit_ticker(text: str) -> bool:
    return bool(_EXPLICIT_PATTERN.search(text.strip()))


def search_ticker(query: str, limit: int = 6) -> list[dict]:
    """Returns [{'symbol', 'name', 'exchange'}, ...], equities only,
    Yahoo's own relevance ranking (most relevant first). One retry -- this
    free endpoint occasionally times out under rapid successive calls,
    same flakiness observed with yfinance itself earlier in this project."""
    params = {"q": query, "quotesCount": limit, "newsCount": 0}
    try:
        resp = requests.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=15)
    except requests.exceptions.RequestException:
        resp = requests.get(_SEARCH_URL, params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    quotes = resp.json().get("quotes", [])

    results = []
    for q in quotes:
        if q.get("quoteType") != "EQUITY" or not q.get("symbol"):
            continue
        results.append({
            "symbol": q["symbol"],
            "name": q.get("shortname") or q.get("longname") or q["symbol"],
            "exchange": q.get("exchange", "?"),
        })
    return results[:limit]


def resolve_ticker(query: str, prefer_exchanges: tuple[str, ...] = ("NSI", "BSE")) -> dict:
    """Resolve free text to a ticker.
    {'status': 'resolved', 'symbol', 'name', 'exchange'}     -- confident single match (or explicit input, trusted as-is)
    {'status': 'ambiguous', 'candidates': [...]}              -- caller should ask the user to pick one
    {'status': 'not_found'}                                   -- no equity matches at all
    """
    query = query.strip()
    if looks_like_explicit_ticker(query):
        return {"status": "resolved", "symbol": query.upper(), "name": query.upper(), "exchange": None}

    candidates = search_ticker(query)
    if not candidates:
        return {"status": "not_found", "candidates": []}
    if len(candidates) == 1:
        c = candidates[0]
        return {"status": "resolved", "symbol": c["symbol"], "name": c["name"], "exchange": c["exchange"]}

    # An exact symbol match (query "AAPL" -> candidate symbol "AAPL") beats
    # everything else -- it's the strongest possible signal, and matters
    # because well-known US tickers otherwise get falsely flagged
    # "ambiguous" by depositary-receipt duplicates on foreign exchanges
    # (AAPL also lists as an Argentina CEDEAR, a Thailand DR, etc. -- same
    # company, not a real choice for the user to make).
    exact = [c for c in candidates if c["symbol"].upper() == query.upper()]
    if len(exact) == 1:
        c = exact[0]
        return {"status": "resolved", "symbol": c["symbol"], "name": c["name"], "exchange": c["exchange"]}

    # If exactly one candidate is on a preferred (Indian) exchange, prefer
    # it without prompting -- but if two+ real companies both sit on NSE/BSE
    # (the Reliance Industries vs Reliance Infrastructure case), that's
    # genuine ambiguity the user needs to resolve, not something to guess.
    preferred = [c for c in candidates if c["exchange"] in prefer_exchanges]
    if len(preferred) == 1:
        c = preferred[0]
        return {"status": "resolved", "symbol": c["symbol"], "name": c["name"], "exchange": c["exchange"]}

    return {"status": "ambiguous", "candidates": preferred or candidates}


if __name__ == "__main__":
    for q in ["reliance", "TCS", "AAPL", "infosys", "tata motors"]:
        print(f"\n>>> {q}")
        print(resolve_ticker(q))
