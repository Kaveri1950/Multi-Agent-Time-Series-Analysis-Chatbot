"""Historical news for each stock, combined from every free source that works.

The FinBERT feature in the price model did nothing for a simple reason:
free RSS reaches back a few weeks, so on TCS only 9 of 886 training days had
a headline and the rest were filled with zeros. No model learns from that,
and no test can say whether news helps. This module assembles years of
dated, per-company news instead, from several sources, because each one
covers a different gap:

  nse_announcements   every filing a company makes to NSE: results, board
                      outcomes, press releases, dividends. Official, dated to
                      the second, 2019 onwards. Event information, not tone.

  nse_events          the exchange's calendar of board meetings and results.
                      These dates are intimated to the exchange in advance
                      (SEBI's LODR rules require prior notice of a results
                      meeting), so a results day is known BEFORE it happens --
                      the one kind of news a forecast can legitimately use
                      ahead of time.

  toi_headlines       3.3 million Times of India headlines (Harvard Dataverse,
                      CC0), filtered to those naming the company and scored
                      with FinBERT. Tone, not events; covers 2001 to 2022.

  rss                 the existing Yahoo / Google News feeds. Recent weeks only.

  gdelt               global news tone. Implemented but optional: on
                      2026-10-21 its API rate-limited this machine and then
                      reset the connection, so nothing downstream depends on
                      it. `gdelt_tone` returns an empty frame when it fails.

Every source records the span it actually covers, so a test can see which
days had news from which source rather than mistaking "no data" for "no news".

THE TIMING RULE. A feature for forecasting day t+1 may only use what was known
by the end of day t: announcements stamped on or before day t, headlines dated
on or before day t, and calendar events whose date was announced in advance.

Run:  python -m finance.news_history            (builds the cache for the panel)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent.parent / "data" / "cache" / "news"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9"}
START, END = "01-01-2019", None          # NSE date format dd-mm-yyyy
TOI_URL = ("https://dataverse.harvard.edu/api/access/datafile/:persistentId"
           "?persistentId=doi:10.7910/DVN/DPQMQH/P2Z4PM")

#: Filings that are routine paperwork rather than news about the business.
#: Counting them would make "lots of announcements" mean "lots of lost share
#: certificates", which is not information about risk.
ROUTINE = re.compile(r"loss of share|duplicate share|newspaper publication|trading window|"
                     r"certificate under reg|compliance certificate|reg\. 74|"
                     r"shareholding pattern|investor complaints|change in rta",
                     re.IGNORECASE)
#: Filings that are, or announce, financial results.
RESULTS = re.compile(r"financial result|outcome of board meeting|results", re.IGNORECASE)

#: How headlines refer to companies whose legal name is not how papers write
#: it. Everything else uses the cleaned NSE company name.
ALIASES = {
    "HDFCBANK": ["HDFC Bank"], "ICICIBANK": ["ICICI Bank"], "SBIN": ["SBI", "State Bank of India"],
    "BHARTIARTL": ["Airtel", "Bharti Airtel"], "M&M": ["Mahindra & Mahindra", "M&M"],
    "TCS": ["TCS", "Tata Consultancy"], "INFY": ["Infosys"], "HCLTECH": ["HCL Tech"],
    "KOTAKBANK": ["Kotak Mahindra Bank", "Kotak Bank"], "AXISBANK": ["Axis Bank"],
    "INDUSINDBK": ["IndusInd"], "BAJFINANCE": ["Bajaj Finance"], "BAJAJFINSV": ["Bajaj Finserv"],
    "HINDUNILVR": ["HUL", "Hindustan Unilever"], "LT": ["Larsen & Toubro", "L&T"],
    "SUNPHARMA": ["Sun Pharma"], "DRREDDY": ["Dr Reddy", "Dr. Reddy"],
    "TATASTEEL": ["Tata Steel"], "TATAPOWER": ["Tata Power"], "TMPV": ["Tata Motors"],
    "ETERNAL": ["Zomato", "Eternal"], "PAYTM": ["Paytm", "One97"], "IDEA": ["Vodafone Idea"],
    "ZEEL": ["Zee Entertainment"], "YESBANK": ["Yes Bank"], "BSE": ["BSE Ltd", "BSE share"],
    "HAL": ["Hindustan Aeronautics", "HAL"], "BEL": ["Bharat Electronics"],
    "BHEL": ["BHEL", "Bharat Heavy Electricals"], "IRFC": ["IRFC"], "IREDA": ["IREDA"],
    "JIOFIN": ["Jio Financial"], "MAZDOCK": ["Mazagon Dock"], "NETWEB": ["Netweb"],
    "OLAELEC": ["Ola Electric"], "SWIGGY": ["Swiggy"], "VEDL": ["Vedanta"],
    "WAAREEENER": ["Waaree"], "SAMMAANCAP": ["Indiabulls Housing", "Sammaan Capital"],
    "SHRIRAMFIN": ["Shriram Finance", "Shriram Transport"], "HINDALCO": ["Hindalco"],
    "ADANIENT": ["Adani Enterprises"], "ADANIPORTS": ["Adani Ports"], "ASIANPAINT": ["Asian Paints"],
    "ITC": ["ITC"], "TITAN": ["Titan"], "MARUTI": ["Maruti"], "WIPRO": ["Wipro"],
    "RELIANCE": ["Reliance Industries", "RIL"], "INDIGO": ["IndiGo", "InterGlobe"],
}


# --------------------------------------------------------------------------
# NSE: announcements and the event calendar
# --------------------------------------------------------------------------

def _nse_session():
    import requests

    s = requests.Session()
    s.headers.update(UA)
    s.get("https://www.nseindia.com/", timeout=20)
    return s


def _cached(path: Path, fetch, refresh: bool):
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df.to_parquet(path, index=False)
    return df


def nse_announcements(symbol: str, refresh: bool = False, session=None) -> pd.DataFrame:
    """Every NSE filing by one company since 2019: timestamp, type, text."""
    def fetch():
        s = session or _nse_session()
        end = END or pd.Timestamp.today().strftime("%d-%m-%Y")
        r = s.get("https://www.nseindia.com/api/corporate-announcements",
                  params={"index": "equities", "symbol": symbol, "from_date": START,
                          "to_date": end}, timeout=60)
        rows = r.json()
        rows = rows if isinstance(rows, list) else rows.get("data", [])
        df = pd.DataFrame([{"timestamp": x.get("sort_date") or x.get("an_dt"),
                            "type": x.get("desc") or "", "text": x.get("attchmntText") or "",
                            "company": x.get("sm_name") or ""} for x in rows])
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "type", "text", "company"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return _cached(CACHE / "nse_announcements" / f"{symbol}.parquet", fetch, refresh)


def nse_events(symbol: str, refresh: bool = False, session=None) -> pd.DataFrame:
    """Board meetings and results dates from the exchange's event calendar."""
    def fetch():
        s = session or _nse_session()
        end = END or (pd.Timestamp.today() + pd.Timedelta(days=60)).strftime("%d-%m-%Y")
        r = s.get("https://www.nseindia.com/api/event-calendar",
                  params={"index": "equities", "symbol": symbol, "from_date": START,
                          "to_date": end}, timeout=60)
        rows = r.json()
        rows = rows if isinstance(rows, list) else rows.get("data", [])
        df = pd.DataFrame([{"date": x.get("date"), "purpose": x.get("purpose") or "",
                            "description": x.get("bm_desc") or ""} for x in rows])
        if df.empty:
            return pd.DataFrame(columns=["date", "purpose", "description"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return _cached(CACHE / "nse_events" / f"{symbol}.parquet", fetch, refresh)


# --------------------------------------------------------------------------
# Times of India headlines
# --------------------------------------------------------------------------

def toi_headlines(refresh: bool = False) -> pd.DataFrame:
    """All 3.3 million headlines, downloaded once (about 200 MB, CC0)."""
    import requests

    raw = CACHE / "india-news-headlines.csv"
    if refresh or not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(TOI_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(raw, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    df = pd.read_csv(raw, dtype=str)
    df.columns = ["date", "category", "headline"]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date", "headline"])


def aliases_for(symbol: str, company: str = "") -> list[str]:
    """Names a headline would use for this company."""
    if symbol in ALIASES:
        return ALIASES[symbol]
    clean = re.sub(r"\b(limited|ltd\.?|corporation|company|india)\b", "", company,
                   flags=re.IGNORECASE).strip(" .,&")
    return [clean] if len(clean) >= 4 else [symbol]


def toi_company_headlines(symbols: list[str], companies: dict[str, str],
                          since: str = "2018-01-01") -> pd.DataFrame:
    """Headlines naming each company, matched on whole words."""
    df = toi_headlines()
    df = df[df["date"] >= since]
    out = []
    for sym in symbols:
        pats = [re.escape(a) for a in aliases_for(sym, companies.get(sym, ""))]
        rx = re.compile(r"(?<![A-Za-z])(?:" + "|".join(pats) + r")(?![A-Za-z])")
        hit = df[df["headline"].str.contains(rx, na=False)]
        if len(hit):
            out.append(hit.assign(symbol=sym))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score_headlines(df: pd.DataFrame, cache_name: str) -> pd.DataFrame:
    """FinBERT score per headline, cached so the model runs once per headline."""
    path = CACHE / f"{cache_name}_scored.parquet"
    if path.exists():
        done = pd.read_parquet(path)
        if len(done) == len(df):
            return done
    from finance.sentiment import _get_finbert

    clf = _get_finbert()
    texts = df["headline"].astype(str).str[:300].tolist()
    scores = []
    for i in range(0, len(texts), 64):
        for res in clf(texts[i:i + 64], truncation=True):
            sign = {"positive": 1.0, "negative": -1.0}.get(res["label"].lower(), 0.0)
            scores.append(sign * float(res["score"]))
    out = df.assign(tone=scores)
    out.to_parquet(path, index=False)
    return out


# --------------------------------------------------------------------------
# GDELT (optional)
# --------------------------------------------------------------------------

def gdelt_tone(query: str, start: str = "20190101000000", end: str | None = None,
               retries: int = 3) -> pd.DataFrame:
    """Daily average tone of global news matching `query`. Empty on failure.

    The API allows one request per five seconds and has refused this machine
    outright at times, so failure is an expected outcome, not an error.
    """
    import requests

    end = end or pd.Timestamp.today().strftime("%Y%m%d000000")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    for attempt in range(retries):
        try:
            r = requests.get(url, params={"query": query, "mode": "timelinetone",
                                          "startdatetime": start, "enddatetime": end,
                                          "format": "json"}, headers=UA, timeout=90)
            if r.status_code == 200:
                data = r.json()["timeline"][0]["data"]
                return pd.DataFrame({"date": pd.to_datetime([d["date"][:8] for d in data]),
                                     "gdelt_tone": [float(d["value"]) for d in data]})
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10 * (attempt + 1))
    return pd.DataFrame(columns=["date", "gdelt_tone"])


# --------------------------------------------------------------------------
# combining
# --------------------------------------------------------------------------

def build(symbols: list[str], refresh: bool = False, verbose: bool = True) -> dict:
    """Fetch every source for every symbol; return the raw tables and coverage."""
    session = _nse_session()
    anns, events, companies = [], [], {}
    for i, sym in enumerate(symbols, 1):
        try:
            a = nse_announcements(sym, refresh=refresh, session=session)
            e = nse_events(sym, refresh=refresh, session=session)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  {sym}: NSE failed ({type(exc).__name__}); retrying with a new session")
            time.sleep(3)
            session = _nse_session()
            a = nse_announcements(sym, refresh=refresh, session=session)
            e = nse_events(sym, refresh=refresh, session=session)
        anns.append(a.assign(symbol=sym))
        events.append(e.assign(symbol=sym))
        companies[sym] = (a["company"].dropna().iloc[0] if len(a) else "")
        if verbose:
            print(f"  [{i:>2}/{len(symbols)}] {sym:<12} {len(a):>5} announcements, "
                  f"{len(e):>3} calendar events", flush=True)
        time.sleep(0.4)                      # NSE throttles at about 3 requests a second
    ann = pd.concat(anns, ignore_index=True)
    ev = pd.concat(events, ignore_index=True)

    toi = toi_company_headlines(symbols, companies)
    if len(toi):
        toi = score_headlines(toi, "toi_company")

    coverage = {
        "nse_announcements": _span(ann["timestamp"]) | {"rows": int(len(ann))},
        "nse_events": _span(ev["date"]) | {"rows": int(len(ev))},
        "toi_headlines": (_span(toi["date"]) | {"rows": int(len(toi))}) if len(toi) else {"rows": 0},
        "gdelt": {"status": "optional; not fetched in bulk (API rate-limits this machine)"},
    }
    return {"announcements": ann, "events": ev, "toi": toi, "companies": companies,
            "coverage": coverage}


def _span(s: pd.Series) -> dict:
    s = pd.to_datetime(s).dropna()
    return {"first": str(s.min().date()), "last": str(s.max().date())} if len(s) else {}


def daily_features(raw: dict, dates: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """One row per (date, ticker): what the news said by the end of that day.

    Calendar events are the exception to "by the end of that day": a results
    date is known in advance, so `results_next_session` may flag tomorrow.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    nxt = pd.Series(dates[1:].tolist() + [pd.NaT], index=dates)
    rows = []
    ann, ev, toi = raw["announcements"], raw["events"], raw["toi"]
    for sym in symbols:
        a = ann[ann["symbol"] == sym].copy()
        a["day"] = a["timestamp"].dt.normalize()
        material = a[~a["type"].str.contains(ROUTINE, na=False)
                     & ~a["text"].str.contains(ROUTINE, na=False)]
        n_mat = material.groupby("day").size()
        res = material[material["type"].str.contains(RESULTS, na=False)
                       | material["text"].str.contains(RESULTS, na=False)].groupby("day").size()
        e = ev[(ev["symbol"] == sym)
               & ev["purpose"].str.contains("result", case=False, na=False)]["date"]
        e_days = set(pd.to_datetime(e).dt.normalize())
        t = toi[toi["symbol"] == sym] if len(toi) else pd.DataFrame(columns=["date", "tone"])
        t_n = t.groupby("date").size()
        t_tone = t.groupby("date")["tone"].mean()
        t_neg = t.assign(neg=t["tone"] < -0.5).groupby("date")["neg"].sum()
        f = pd.DataFrame(index=dates)
        f["ticker"] = sym + ".NS"
        f["ann_material"] = n_mat.reindex(dates).fillna(0).to_numpy()
        f["ann_results"] = res.reindex(dates).fillna(0).to_numpy()
        f["results_today"] = [float(d in e_days) for d in dates]
        f["results_next_session"] = [float(n in e_days) if pd.notna(n) else 0.0
                                     for n in nxt.reindex(dates)]
        f["toi_count"] = t_n.reindex(dates).fillna(0).to_numpy()
        f["toi_tone"] = t_tone.reindex(dates).to_numpy()            # NaN = no headline
        f["toi_negative"] = t_neg.reindex(dates).fillna(0).to_numpy()
        rows.append(f.rename_axis("date").reset_index())
    return pd.concat(rows, ignore_index=True)


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from quant.volatility import OHLC_PATH

    symbols = sorted(t.replace(".NS", "") for t in
                     pd.read_parquet(OHLC_PATH, columns=["ticker"])["ticker"].unique())
    raw = build(symbols)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "coverage.json").write_text(json.dumps(raw["coverage"], indent=2))
    print(json.dumps(raw["coverage"], indent=2))
    if len(raw["toi"]):
        per = raw["toi"].groupby("symbol").size().sort_values(ascending=False)
        print("Times of India headlines per company (top 10):", per.head(10).to_dict())
        print("companies with none:", sorted(set(symbols) - set(per.index)))


if __name__ == "__main__":
    main()
