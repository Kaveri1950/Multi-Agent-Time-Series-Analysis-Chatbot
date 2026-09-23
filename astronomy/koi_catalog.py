"""KOI catalogue: real labels and transit ephemerides from NASA.

This module exists to replace the proxy ground truth in
astronomy/anomaly_detector.py's label_windows_by_depth(), which labels a
window anomalous when its flux dips more than a threshold -- using `depth`,
which is ALSO one of the eight features the detector sees. That makes the
evaluation partly circular: it measures whether IsolationForest can
rediscover a threshold on its own input.

The NASA Exoplanet Archive publishes, for 9,564 Kepler Objects of Interest,
a disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE) and a complete
transit ephemeris -- period, epoch, and duration. With those, a window can
be labelled by whether a KNOWN, PUBLISHED transit falls inside it. The label
then comes from astronomy rather than from the detector's own features, and
it works on every target rather than only those that happen to cross a depth
threshold.

Verified live 2026-09-16: 9,564 KOIs (2,748 CONFIRMED, 1,977 CANDIDATE,
4,839 FALSE POSITIVE), with zero nulls in period/epoch/duration.

TIME SCALE -- the detail that silently ruins this if you get it wrong:
`koi_time0bk` is in BKJD (Barycentric Kepler Julian Date, BJD - 2454833),
and lightkurve reports Kepler `lc.time.value` in BKJD as well. So the two
are directly comparable with no offset. Do not "helpfully" convert either.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "astronomy"
KOI_TABLE_PATH = CACHE_DIR / "koi_cumulative.parquet"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# The columns actually needed: identity, label, and the three ephemeris
# numbers that place every transit this object has ever shown.
KOI_COLUMNS = [
    "kepid",            # the KIC number -- joins to the light curve
    "kepoi_name",       # e.g. K00157.01
    "kepler_name",      # e.g. Kepler-11 b (null for unconfirmed)
    "koi_disposition",  # CONFIRMED | CANDIDATE | FALSE POSITIVE
    "koi_period",       # orbital period, days
    "koi_time0bk",      # epoch of first transit, BKJD
    "koi_duration",     # transit duration, HOURS
    "koi_depth",        # transit depth, parts per million
    "koi_prad",         # planet radius, Earth radii
    "koi_score",        # archive's own disposition confidence, 0-1
]

REQUEST_TIMEOUT = 180


class KOICatalogError(RuntimeError):
    """The archive was reachable but returned something unusable."""


def fetch_koi_table(force_refetch: bool = False) -> pd.DataFrame:
    """The full cumulative KOI table, cached to parquet.

    One HTTP request, no API key. ~9.5k rows, so it is small enough to hold
    entirely in memory and cheap enough to refresh whenever the archive
    updates its dispositions.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if KOI_TABLE_PATH.exists() and not force_refetch:
        return pd.read_parquet(KOI_TABLE_PATH)

    query = f"select {','.join(KOI_COLUMNS)} from cumulative"
    resp = requests.get(TAP_URL, params={"query": query, "format": "csv"},
                        timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise KOICatalogError(f"TAP HTTP {resp.status_code}: {resp.text[:200]}")

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "kepid" not in df.columns:
        raise KOICatalogError(f"unexpected response shape: {list(df.columns)[:8]}")

    df.to_parquet(KOI_TABLE_PATH)
    return df


def usable_kois(koi: pd.DataFrame | None = None) -> pd.DataFrame:
    """KOIs with a complete ephemeris and a decided disposition.

    CANDIDATEs are excluded from the labelled set on purpose: "we do not yet
    know" is not a class, and training or scoring against it would mean
    treating the archive's open questions as ground truth. They are still
    worth keeping for later -- an unlabelled pool to make predictions on is
    exactly what a candidate list is for.
    """
    koi = fetch_koi_table() if koi is None else koi
    ok = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
    return ok[ok["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()


def select_targets(n_targets: int = 500, balanced: bool = True, seed: int = 0,
                   koi: pd.DataFrame | None = None) -> pd.DataFrame:
    """Choose which stars to download, one row per KIC.

    A star is labelled CONFIRMED if ANY of its KOIs is confirmed -- a system
    with one confirmed planet and three false positives is still a star with
    real transits in its light curve, which is what the detector sees.

    `balanced` samples equally from confirmed and false-positive stars.
    Without it the sample inherits the archive's 2748/4839 skew, and a
    classifier can score well by leaning on the majority class rather than
    on the light curves.
    """
    ok = usable_kois(koi)

    per_star = ok.groupby("kepid").agg(
        n_kois=("kepoi_name", "count"),
        n_confirmed=("koi_disposition", lambda s: int((s == "CONFIRMED").sum())),
        min_period=("koi_period", "min"),
        max_depth=("koi_depth", "max"),
    ).reset_index()
    per_star["label"] = np.where(per_star["n_confirmed"] > 0, "CONFIRMED", "FALSE POSITIVE")

    rng = np.random.default_rng(seed)
    if not balanced:
        take = min(n_targets, len(per_star))
        idx = rng.choice(len(per_star), size=take, replace=False)
        return per_star.iloc[idx].reset_index(drop=True)

    out = []
    for label in ("CONFIRMED", "FALSE POSITIVE"):
        pool = per_star[per_star["label"] == label]
        take = min(n_targets // 2, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        out.append(pool.iloc[idx])
    return pd.concat(out).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------
# ephemeris -> per-point and per-window labels
# --------------------------------------------------------------------------

def transit_mask(times: np.ndarray, kepid: int, koi: pd.DataFrame | None = None,
                 duration_scale: float = 1.0) -> np.ndarray:
    """Boolean per time point: is a KNOWN transit happening then?

    Folds each KOI's ephemeris onto the observation times. A star with six
    confirmed planets contributes six independent transit trains, OR-ed
    together -- which is the physically correct thing, since the light curve
    really does contain all of them.

    `duration_scale` widens the in-transit window; >1 is useful when
    labelling coarse bins, so a transit shorter than one bin still marks the
    bin containing it.
    """
    koi = usable_kois(koi) if koi is None else koi
    rows = koi[koi["kepid"] == kepid]
    mask = np.zeros(len(times), dtype=bool)

    for _, r in rows.iterrows():
        period = float(r["koi_period"])
        epoch = float(r["koi_time0bk"])
        half_dur = (float(r["koi_duration"]) / 24.0) * duration_scale / 2.0
        if period <= 0 or half_dur <= 0:
            continue
        # signed time to the NEAREST transit centre, not to the first one
        dt = (times - epoch + period / 2.0) % period - period / 2.0
        mask |= np.abs(dt) <= half_dur

    return mask


def label_windows(times: np.ndarray, kepid: int, window_size: int, stride: int,
                  koi: pd.DataFrame | None = None, min_in_transit: int = 1,
                  duration_scale: float = 1.0) -> np.ndarray:
    """Per-window labels aligned with astronomy.anomaly_detector.make_windows.

    A window is positive when it contains at least `min_in_transit` points
    that fall inside a known transit. Kept deliberately in step with
    make_windows' own slicing so window i here is window i there -- an
    off-by-one between the two would silently scramble every metric.
    """
    point_mask = transit_mask(times, kepid, koi, duration_scale=duration_scale)
    n = len(times)
    labels = [
        int(point_mask[i:i + window_size].sum() >= min_in_transit)
        for i in range(0, n - window_size + 1, stride)
    ]
    return np.array(labels, dtype=int)


def target_summary(kepid: int, koi: pd.DataFrame | None = None) -> dict:
    """What the archive knows about one star -- for the chatbot and the
    dashboard, so an answer can cite the real planets rather than only the
    detector's opinion."""
    koi = usable_kois(koi) if koi is None else koi
    rows = koi[koi["kepid"] == kepid]
    if rows.empty:
        return {"kepid": kepid, "known": False}
    confirmed = rows[rows["koi_disposition"] == "CONFIRMED"]
    return {
        "kepid": kepid,
        "known": True,
        "n_kois": len(rows),
        "n_confirmed": len(confirmed),
        "label": "CONFIRMED" if len(confirmed) else "FALSE POSITIVE",
        "planet_names": [p for p in confirmed["kepler_name"].dropna().tolist()],
        "periods_days": [round(float(p), 4) for p in rows["koi_period"]],
        "depths_ppm": [None if pd.isna(d) else float(d) for d in rows["koi_depth"]],
        "radii_earth": [None if pd.isna(r) else round(float(r), 2) for r in rows["koi_prad"]],
    }


if __name__ == "__main__":
    print("Fetching the cumulative KOI table...")
    koi = fetch_koi_table()
    print(f"  {len(koi)} KOIs")
    print(f"  {koi['koi_disposition'].value_counts().to_dict()}")

    ok = usable_kois(koi)
    print(f"\nUsable (complete ephemeris, decided disposition): {len(ok)} KOIs "
          f"across {ok['kepid'].nunique()} stars")

    sel = select_targets(500)
    print(f"\nBalanced sample of {len(sel)} stars: {sel['label'].value_counts().to_dict()}")
    print(sel.head(6).to_string(index=False))

    # sanity check the ephemeris folding against a known multi-planet system
    kepid = 6541920
    times = np.linspace(120, 1580, 64797)     # a Kepler-like BKJD baseline
    mask = transit_mask(times, kepid, ok)
    info = target_summary(kepid, ok)
    print(f"\nKIC {kepid}: {info['n_kois']} KOIs, {info['n_confirmed']} confirmed")
    print(f"  in-transit fraction over a 4-year baseline: {mask.mean():.4%}")
    print(f"  periods: {info['periods_days']}")
