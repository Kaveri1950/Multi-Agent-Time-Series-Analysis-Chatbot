"""Solar flare + space-weather data ingestion, free & public, no API key.

Two very different kinds of source, and the difference drives the whole
design:

1. NOAA SWPC's live JSON/text endpoints are all ROLLING WINDOWS (7 or 30
   days). They are the only source of recent data, and NOAA does not
   backfill them -- whatever you did not download, you cannot ever get.
   Hence harvest_daily(): run it once a day and it appends today's snapshot
   to a local parquet archive, so a genuine 2026 holdout set accrues over
   the project instead of having to be faked.

2. NASA/Hesperia's yearly GOES event lists are a real historical catalogue,
   1975 to the present, ~70k classified flares spanning five solar cycles.
   This is what the models actually train on.

CORRECTION, recorded because the wrong version shaped several decisions.
This module previously asserted that "no bulk archive covers 2021-2026" and
capped training at 2020. That was wrong. The claim came from reading a
directory listing whose visible subfolders stopped at 2020, without ever
requesting the event-list filenames directly. They are all there:

    2021: 2,065   2022: 3,050   2023: 2,887
    2024: 3,749   2025: 3,204   2026: 2,174 (ytd)

2024 is the Cycle 25 maximum, and its omission was measurable: scored against
NOAA over 57 archived days, the model trained on 1996-2020 predicted a mean
probability of 0.114 against an actual M+ rate of 0.228 -- systematically
under-confident by roughly half, because its background rate was anchored to
a quieter average.

The daily harvester is still worth running, but for the reason it was always
worth running: a prospectively logged forecast cannot be tuned against, which
a hindcast over known days can be.

All endpoint URLs and field names below were verified live against the real
services (2026-08-21), not guessed from documentation.
"""

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "solar"
HISTORY_DIR = CACHE_DIR / "history"       # one parquet per year, from Hesperia
ARCHIVE_DIR = CACHE_DIR / "archive"       # accumulated daily harvest snapshots

_SWPC = "https://services.swpc.noaa.gov"
_HESPERIA = "https://hesperia.gsfc.nasa.gov/goes/goes_event_listings"

# Verified live endpoints. Each is a rolling window -- see module docstring.
ENDPOINTS = {
    "flares":        f"{_SWPC}/json/goes/primary/xray-flares-7-day.json",
    "xray_flux":     f"{_SWPC}/json/goes/primary/xrays-1-day.json",
    "daily_indices": f"{_SWPC}/text/daily-solar-indices.txt",
    "probabilities": f"{_SWPC}/json/solar_probabilities.json",
    "regions":       f"{_SWPC}/text/solar-regions.txt",
}

# GOES soft X-ray class -> peak flux in W/m^2 at the class's "1.0" point.
# A flare's class letter plus its multiplier is just a log-scale flux, so
# "M1.5" is 1.5e-5 W/m^2. Converting to a single continuous number matters
# because a model should treat X1.0 as ten times M1.0, not as "the next
# category along".
CLASS_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

REQUEST_TIMEOUT = 30


class SolarDataError(RuntimeError):
    """A source was reachable but returned something unusable."""


# --------------------------------------------------------------------------
# flare class <-> flux
# --------------------------------------------------------------------------

_CLASS_RE = re.compile(r"^\s*([ABCMX])\s*([0-9]*\.?[0-9]+)?\s*$", re.IGNORECASE)


def parse_flare_class(class_str):
    """'M1.5' -> ('M', 1.5e-05). Returns (None, None) for missing or
    unparseable input rather than raising -- NOAA leaves these blank often
    enough that failing hard here would throw away otherwise-good rows."""
    if not class_str or not isinstance(class_str, str):
        return None, None
    m = _CLASS_RE.match(class_str)
    if not m:
        return None, None
    letter = m.group(1).upper()
    multiplier = float(m.group(2)) if m.group(2) else 1.0
    return letter, CLASS_BASE[letter] * multiplier


def class_at_least(class_str, threshold: str = "M") -> bool:
    """Is this flare at or above `threshold` class? The forecasting target is
    'M or bigger', which is a flux comparison, not a string comparison."""
    _, flux = parse_flare_class(class_str)
    if flux is None:
        return False
    return flux >= CLASS_BASE[threshold.upper()]


# --------------------------------------------------------------------------
# historical catalogue (Hesperia yearly GOES event lists, 1975-present)
# --------------------------------------------------------------------------

# " 3-Jan-2020 21:02   21:03   21:04   A1.0   N22E30   12756"
#   date        start   peak    end     class  position AR
# Position and active region are both optional and either may be absent, so
# the trailing fields are classified by shape rather than by position.
_HESP_ROW = re.compile(
    r"^\s*(\d{1,2}-[A-Za-z]{3}-\d{4})\s+"      # date
    r"(\d{2}:\d{2})\s+"                         # begin
    r"(\d{2}:\d{2})\s+"                         # peak
    r"(\d{2}:\d{2})\s+"                         # end
    r"([ABCMX][0-9.]*)"                         # class
    r"(?P<rest>.*)$"
)
_POSITION_RE = re.compile(r"\b([NS]\d{1,2}[EW]\d{1,3})\b")
_AR_RE = re.compile(r"\b(\d{4,5})\b")

HISTORY_FIRST_YEAR = 1975
HISTORY_LAST_YEAR = 2026


def _combine(day: datetime, hhmm: str, reference=None) -> datetime:
    """Attach an HH:MM to a date, rolling to the next day when the clock has
    gone backwards. Hesperia rows record only times of day, so a flare that
    begins 23:50 and ends 00:14 would otherwise appear to last -23 hours."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    stamp = day.replace(hour=hh, minute=mm)
    if reference is not None and stamp < reference:
        stamp += timedelta(days=1)
    return stamp


def parse_hesperia_year(text: str) -> pd.DataFrame:
    """Parse one Hesperia yearly event list into tidy rows."""
    rows = []
    for line in text.splitlines():
        m = _HESP_ROW.match(line)
        if not m:
            continue
        date_s, begin_s, peak_s, end_s, class_s = m.groups()[:5]
        rest = m.group("rest")

        day = datetime.strptime(date_s, "%d-%b-%Y")
        begin = _combine(day, begin_s)
        peak = _combine(day, peak_s, reference=begin)
        end = _combine(day, end_s, reference=peak)

        position = _POSITION_RE.search(rest)
        # strip the position before hunting for an AR number, so the digits
        # inside "N22E30" cannot be mistaken for a region id
        rest_no_pos = _POSITION_RE.sub(" ", rest)
        ar = _AR_RE.search(rest_no_pos)

        letter, flux = parse_flare_class(class_s)
        rows.append({
            "begin_time": begin,
            "peak_time": peak,
            "end_time": end,
            "flare_class": class_s,
            "class_letter": letter,
            "peak_flux": flux,
            "duration_min": (end - begin).total_seconds() / 60.0,
            "position": position.group(1) if position else None,
            "active_region": int(ar.group(1)) if ar else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("begin_time").reset_index(drop=True)
    return df


def fetch_historical_year(year: int, force_refetch: bool = False) -> pd.DataFrame:
    """One year of the GOES flare catalogue, cached to parquet.

    Available 1975 to the present. An earlier version of this docstring said
    coverage stopped at 2020 and described a 2021-2026 gap; that was wrong --
    see the correction in the module docstring.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = HISTORY_DIR / f"flares_{year}.parquet"

    if cache_file.exists() and not force_refetch:
        return pd.read_parquet(cache_file)

    url = f"{_HESPERIA}/goes_xray_event_list_{year}.txt"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise SolarDataError(f"{year}: HTTP {resp.status_code} from {url}")

    df = parse_hesperia_year(resp.text)
    if df.empty:
        raise SolarDataError(f"{year}: fetched {len(resp.text)} bytes but parsed 0 flares")

    df.to_parquet(cache_file)
    return df


def load_historical_catalog(start_year: int = HISTORY_FIRST_YEAR,
                            end_year: int = HISTORY_LAST_YEAR,
                            skip_failures: bool = True) -> pd.DataFrame:
    """The full training catalogue. Years that fail are reported and skipped
    rather than aborting a multi-decade download, but the caller is told
    exactly which are missing."""
    frames, failed = [], []
    for year in range(start_year, end_year + 1):
        try:
            frames.append(fetch_historical_year(year))
        except Exception as e:
            failed.append((year, f"{type(e).__name__}: {e}"))
            if not skip_failures:
                raise
    if failed:
        print(f"  [warn] {len(failed)} year(s) unavailable: "
              + ", ".join(str(y) for y, _ in failed))
    if not frames:
        raise SolarDataError("no historical years could be loaded")
    return pd.concat(frames, ignore_index=True).sort_values("begin_time").reset_index(drop=True)


# --------------------------------------------------------------------------
# live rolling-window endpoints
# --------------------------------------------------------------------------

def _get_json(name: str):
    url = ENDPOINTS[name]
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise SolarDataError(f"{name}: HTTP {resp.status_code} from {url}")
    try:
        return resp.json()
    except json.JSONDecodeError as e:
        raise SolarDataError(f"{name}: response was not JSON ({e})") from e


def _get_text(name: str) -> str:
    url = ENDPOINTS[name]
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise SolarDataError(f"{name}: HTTP {resp.status_code} from {url}")
    return resp.text


def fetch_recent_flares() -> pd.DataFrame:
    """Last ~7 days of classified flares. Real event times and real GOES
    classes -- genuine ground truth, not a proxy label."""
    df = pd.DataFrame(_get_json("flares"))
    if df.empty:
        return df
    for col in ("begin_time", "max_time", "end_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    parsed = df["max_class"].apply(parse_flare_class)
    df["class_letter"] = [p[0] for p in parsed]
    df["peak_flux"] = [p[1] for p in parsed]
    return df.sort_values("begin_time").reset_index(drop=True)


def fetch_noaa_probabilities() -> pd.DataFrame:
    """NOAA's OWN operational flare forecast: C/M/X probabilities at 1/2/3
    day horizons, last ~31 days. This is the benchmark the project is scored
    against -- a real human-plus-model operational product, not a strawman
    baseline invented to be beaten."""
    df = pd.DataFrame(_get_json("probabilities"))
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_xray_flux() -> pd.DataFrame:
    """1-minute GOES X-ray flux, last 24h, in two energy bands. The long band
    (0.1-0.8nm) is what defines flare class; the ratio of the two is a
    hardness proxy that tends to rise as plasma heats."""
    df = pd.DataFrame(_get_json("xray_flux"))
    if not df.empty:
        df["time_tag"] = pd.to_datetime(df["time_tag"], errors="coerce", utc=True)
        df = df.sort_values("time_tag").reset_index(drop=True)
    return df


# "2026 07 22  150     63      820      0    -999      *   7  1  0 10  0  0  0"
_DSD_COLUMNS = ["radio_flux", "sunspot_number", "sunspot_area", "new_regions",
                "stanford_mean_field", "xray_background",
                "c_flares", "m_flares", "x_flares",
                "optical_s", "optical_1", "optical_2", "optical_3"]


def _dsd_value(token: str):
    """DSD marks missing data as -999 or '*'."""
    if token in ("*", "-999", "-999.9"):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_daily_indices(text: str) -> pd.DataFrame:
    """Parse the Daily Solar Data product: the classic daily feature set --
    radio flux, sunspot number and area, new regions, and the day's own
    C/M/X flare counts."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ":")):
            continue
        parts = line.split()
        if len(parts) < 10 or not re.match(r"^\d{4}$", parts[0]):
            continue
        try:
            date = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        row = {"date": date}
        for name, token in zip(_DSD_COLUMNS, parts[3:]):
            row[name] = _dsd_value(token)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_daily_indices() -> pd.DataFrame:
    """Last 30 days of daily solar indices (rolling -- see harvest_daily)."""
    return parse_daily_indices(_get_text("daily_indices"))


# " 4506    N12W0*    216     30     7     DAO    4      B"
#  region   location  long   area extent McIntosh count MtWilson
_REGION_ROW = re.compile(
    r"^\s*(\d{4,5})\s+"                  # region number
    r"([NS]\d{1,2}[EW]\d{1,3}\*?)\s+"    # heliographic location (may carry a '*')
    r"(\d+)\s+"                           # Carrington longitude
    r"(\d+)\s+"                           # area
    r"(\d+)\s+"                           # extent
    r"([A-Z]{3})\s+"                      # McIntosh class
    r"(\d+)\s+"                           # spot count
    r"([A-Z]{1,4})\s*$"                   # Mount Wilson magnetic class
)

# Mount Wilson magnetic complexity, ordered. Delta configurations are where
# nearly all X-class flares come from, so this ordinal is one of the single
# most informative features available anywhere in this module.
MAG_COMPLEXITY = {"A": 1, "B": 2, "G": 3, "BG": 3, "BD": 4, "GD": 4, "BGD": 5}


def parse_solar_regions(text: str) -> pd.DataFrame:
    """Parse the Solar Region Summary: one row per active region on the disc,
    with its McIntosh and Mount Wilson magnetic classification.

    NOTE: this endpoint carries TODAY only. There is no verified historical
    archive for it yet, so magnetic-complexity features are available for
    live forecasting but not (currently) for training on 1975-2020. Flagged
    rather than quietly worked around -- it is a real limitation.
    """
    rows = []
    for line in text.splitlines():
        m = _REGION_ROW.match(line)
        if not m:
            continue
        region, location, longitude, area, extent, mcintosh, count, mag = m.groups()
        rows.append({
            "region": int(region),
            "location": location.rstrip("*"),
            "carrington_longitude": int(longitude),
            "area": int(area),
            "extent": int(extent),
            "mcintosh_class": mcintosh,
            "spot_count": int(count),
            "mag_class": mag,
            "mag_complexity": MAG_COMPLEXITY.get(mag, 0),
            "has_delta": "D" in mag,
        })
    return pd.DataFrame(rows)


def fetch_solar_regions() -> pd.DataFrame:
    """Today's active regions with magnetic classification."""
    return parse_solar_regions(_get_text("regions"))


# --------------------------------------------------------------------------
# the daily harvester -- the time-sensitive part
# --------------------------------------------------------------------------

_HARVEST = {
    "flares":        ("flares.parquet",        fetch_recent_flares,      ["begin_time", "satellite"]),
    "daily_indices": ("daily_indices.parquet", fetch_daily_indices,      ["date"]),
    "probabilities": ("probabilities.parquet", fetch_noaa_probabilities, ["date"]),
    "regions":       ("regions.parquet",       fetch_solar_regions,      ["observed_date", "region"]),
}


def harvest_daily(verbose: bool = True) -> dict:
    """Append today's snapshot of every rolling endpoint to the local archive.

    Idempotent: safe to run many times a day, deduped on natural keys, so a
    cron job that fires twice or a manual re-run costs nothing. Every run
    only ever adds rows -- NOAA's windows slide forward and old data becomes
    permanently unavailable, so this never deletes.

    Returns {source: rows_added}.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    added = {}

    for name, (filename, fetch, keys) in _HARVEST.items():
        path = ARCHIVE_DIR / filename
        try:
            fresh = fetch()
        except Exception as e:
            if verbose:
                print(f"  [warn] {name}: {type(e).__name__}: {e}")
            added[name] = 0
            continue

        if fresh.empty:
            added[name] = 0
            continue

        # the regions product is undated in its rows -- stamp it, or repeated
        # harvests would collapse into a single indistinguishable day
        if name == "regions":
            fresh.insert(0, "observed_date", pd.Timestamp(today))

        before = 0
        if path.exists():
            existing = pd.read_parquet(path)
            before = len(existing)
            combined = pd.concat([existing, fresh], ignore_index=True)
        else:
            combined = fresh

        usable_keys = [k for k in keys if k in combined.columns]
        if usable_keys:
            combined = combined.drop_duplicates(subset=usable_keys, keep="last")
        combined.to_parquet(path)

        added[name] = len(combined) - before
        if verbose:
            print(f"  {name:<14} +{added[name]:<5} (archive now {len(combined)} rows)")

    return added


def load_archive(name: str) -> pd.DataFrame:
    """Read back one harvested archive. Empty frame if never harvested."""
    filename = _HARVEST[name][0]
    path = ARCHIVE_DIR / filename
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    print("Solar data layer self-check\n" + "=" * 62)

    print("\n[1] live endpoints")
    for label, fn in (("recent flares", fetch_recent_flares),
                      ("noaa forecast", fetch_noaa_probabilities),
                      ("daily indices", fetch_daily_indices),
                      ("active regions", fetch_solar_regions),
                      ("xray flux", fetch_xray_flux)):
        try:
            df = fn()
            print(f"  {label:<16} {len(df):>6} rows   {list(df.columns)[:4]}")
        except Exception as e:
            print(f"  {label:<16} FAILED  {type(e).__name__}: {e}")

    print("\n[2] historical catalogue (one year, cached)")
    try:
        y = fetch_historical_year(2014)
        counts = y["class_letter"].value_counts().to_dict()
        m_plus = y[y["flare_class"].apply(class_at_least)]
        print(f"  2014: {len(y)} flares, by class {counts}")
        print(f"  M+ events: {len(m_plus)} on {m_plus['begin_time'].dt.date.nunique()} distinct days")
    except Exception as e:
        print(f"  FAILED  {type(e).__name__}: {e}")

    print("\n[3] daily harvest")
    harvest_daily()
    print("\ndone.")
