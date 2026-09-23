"""Indian market universes: index constituents and index-level tickers,
all free/public sources, no API key.

- Nifty 50 / 100 / 500 constituents: NSE's public archive CSVs
  (archives.nseindia.com) -- real, current, refreshed on every call
  (cached to disk for a day to avoid hammering NSE and to survive their
  occasional rate-limiting/downtime).
- Sensex 30: NSE/BSE don't expose a free scrapable constituent list the
  way NSE does for Nifty, so this is a hardcoded snapshot -- Sensex only
  reconstitutes ~twice a year, so this is a reasonable static list for a
  BTP project. Verify against bseindia.com/sensex before using for
  anything beyond coursework, and update SENSEX_30_SYMBOLS if it drifts.
- Index-level price series (e.g. "how is Nifty 50 doing"): plain yfinance
  tickers, already free -- ^NSEI / ^CNX100 / ^CRSLDX / ^BSESN.
"""

import json
import time
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
CACHE_TTL_SECONDS = 24 * 3600

NSE_CSV_URLS = {
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

INDEX_TICKERS = {
    "nifty50": "^NSEI",
    "nifty100": "^CNX100",
    "nifty500": "^CRSLDX",
    "sensex": "^BSESN",
    "sensex30": "^BSESN",
}

# Snapshot as of 2026; BSE Sensex 30 constituents (NSE-listed symbols, since
# yfinance's NSE data tends to be more complete than its BSE data for the
# same dual-listed company).
SENSEX_30_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "LT", "KOTAKBANK", "AXISBANK",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "M&M", "NTPC", "POWERGRID", "TATASTEEL", "TATAMOTORS", "TECHM",
    "HCLTECH", "INDUSINDBK", "JSWSTEEL", "BAJAJFINSV", "WIPRO",
]


def _read_cache(name: str) -> list[str] | None:
    path = CACHE_DIR / f"{name}_constituents.json"
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
        return None
    return json.loads(path.read_text())


def _write_cache(name: str, symbols: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{name}_constituents.json").write_text(json.dumps(symbols))


def fetch_index_constituents(index: str) -> list[str]:
    """Returns yfinance-ready tickers (NSE symbol + '.NS'), e.g. ['RELIANCE.NS', ...].
    `index` is one of 'nifty50', 'nifty100', 'nifty500', 'sensex'/'sensex30'."""
    index = index.lower().strip()

    if index in ("sensex", "sensex30"):
        return [f"{s}.NS" for s in SENSEX_30_SYMBOLS]

    if index not in NSE_CSV_URLS:
        raise ValueError(f"Unknown index '{index}'. Use one of: nifty50, nifty100, nifty500, sensex.")

    cached = _read_cache(index)
    if cached is not None:
        return cached

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(NSE_CSV_URLS[index], headers=headers, timeout=15)
    resp.raise_for_status()

    lines = resp.text.splitlines()
    header = lines[0].split(",")
    symbol_col = header.index("Symbol")
    symbols = [f"{line.split(',')[symbol_col].strip()}.NS" for line in lines[1:] if line.strip()]

    _write_cache(index, symbols)
    return symbols


def get_index_ticker(index: str) -> str:
    index = index.lower().strip()
    if index not in INDEX_TICKERS:
        raise ValueError(f"Unknown index '{index}'. Use one of: {list(INDEX_TICKERS)}.")
    return INDEX_TICKERS[index]


if __name__ == "__main__":
    for idx in ["nifty50", "nifty100", "nifty500", "sensex"]:
        symbols = fetch_index_constituents(idx)
        print(f"{idx}: {len(symbols)} constituents, e.g. {symbols[:5]}  (index ticker: {get_index_ticker(idx)})")
