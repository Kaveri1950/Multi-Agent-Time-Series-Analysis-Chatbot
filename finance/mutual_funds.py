"""Indian mutual fund NAV data via api.mfapi.in -- free, public, no API
key, community-run but widely used, backed by AMFI's official daily NAV
filings. Two endpoints:
  - GET /mf            -> full scheme list (~38k schemes), cached to disk
  - GET /mf/{code}      -> full historical NAV series for one scheme

Note: unlike single-company stocks, a diversified mutual fund's NAV
doesn't have a clean sentiment signal the way one company's news does, so
prediction here is ARIMA-only (see mutual_funds.forecast_nav) -- no
XGBoost+sentiment augmentation is offered for funds, by design, not as an
oversight.
"""

import json
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning, module="finance.mutual_funds")

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
SCHEME_LIST_CACHE_TTL_SECONDS = 7 * 24 * 3600  # scheme list changes rarely; NAV history is always fetched fresh

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _load_scheme_list() -> list[dict]:
    cache_path = CACHE_DIR / "mf_scheme_list.json"
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < SCHEME_LIST_CACHE_TTL_SECONDS:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resp = requests.get("https://api.mfapi.in/mf", headers=_HEADERS, timeout=60)  # ~38k schemes, several MB -- genuinely slow, not a hang
    resp.raise_for_status()
    data = resp.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def search_schemes(query: str, limit: int = 10) -> list[dict]:
    """Case-insensitive substring search over scheme names. Returns
    [{'schemeCode': int, 'schemeName': str}, ...]."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    schemes = _load_scheme_list()
    matches = [s for s in schemes if query_lower in s["schemeName"].lower()]
    return matches[:limit]


def get_nav_history(scheme_code: int) -> pd.DataFrame:
    """Real historical NAV, oldest to newest. Columns: ['nav'], DatetimeIndex."""
    resp = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("data", [])
    if not rows:
        raise ValueError(f"No NAV data returned for scheme code {scheme_code}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = df["nav"].astype(float)
    df = df.set_index("date").sort_index()
    return df[["nav"]]


def get_scheme_meta(scheme_code: int) -> dict:
    resp = requests.get(f"https://api.mfapi.in/mf/{scheme_code}", headers=_HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json().get("meta", {})


def compute_returns(nav_df: pd.DataFrame) -> dict:
    """Trailing returns as of the latest NAV date -- 1M/6M/1Y/3Y % change,
    None where history doesn't go back far enough."""
    latest_date, latest_nav = nav_df.index[-1], nav_df["nav"].iloc[-1]
    windows = {"1M": 30, "6M": 182, "1Y": 365, "3Y": 365 * 3}

    out = {}
    for label, days in windows.items():
        cutoff = latest_date - pd.Timedelta(days=days)
        past = nav_df[nav_df.index <= cutoff]
        if past.empty:
            out[label] = None
        else:
            past_nav = past["nav"].iloc[-1]
            out[label] = round((latest_nav / past_nav - 1) * 100, 2)
    return out


def forecast_nav(nav_df: pd.DataFrame, test_size: int = 30) -> dict:
    """ARIMA baseline on the NAV series -- same approach as
    finance/price_model.arima_baseline, generalized to any price-like series."""
    from finance.price_model import arima_baseline

    return arima_baseline(nav_df["nav"], test_size=test_size)


if __name__ == "__main__":
    matches = search_schemes("HDFC Top 100")
    print(f"'HDFC Top 100' matches: {len(matches)}")
    for m in matches[:5]:
        print(" ", m)

    if matches:
        code = matches[0]["schemeCode"]
    else:
        code = 119598  # SBI Large Cap Fund - Direct - Growth, verified reachable

    meta = get_scheme_meta(code)
    print("\nScheme meta:", meta)

    nav_df = get_nav_history(code)
    print(f"\nNAV history: {len(nav_df)} points, {nav_df.index[0].date()} to {nav_df.index[-1].date()}")
    print(nav_df.tail())

    print("\nTrailing returns:", compute_returns(nav_df))

    result = forecast_nav(nav_df)
    print("\nARIMA RMSE on last 30 NAV points:", result["rmse"])
