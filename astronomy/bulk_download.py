"""Bulk light-curve acquisition: many targets, uniform cadence, full mission.

Replaces the 20-target hand-picked sample in astronomy/dataset.py, and fixes
three problems that made the old cache unusable for real evaluation:

1. ARBITRARY PRODUCT SELECTION. data_loader.fetch_light_curve() calls
   `search[0].download()` -- whatever MAST happens to list first. In
   practice that returned Kepler Quarter 00 (a ~10-day commissioning run,
   469 points) for some stars and Quarter 02 at 1-minute cadence (41,816
   points) for others. Nobody chose these.

2. MIXED CADENCE. The old cache held 13 targets at ~1 min and 7 at ~30 min.
   A 200-point window therefore covered 3.3 hours on some targets and 98 on
   others, so the same feature meant different things per star and pooling
   them into one metric was meaningless.

3. ONE QUARTER PER STAR. Kepler observed for four years; the old cache held
   ~30 days per target, about 2% of what exists.

This module takes long cadence (1800s) for everything and stitches every
available quarter. Long cadence is the right uniform choice: it exists for
essentially all Kepler targets across all 17 quarters, whereas short cadence
was allocated selectively. One target becomes ~65,000 points spanning ~1,460
days instead of ~43,000 points spanning 30.

NO SYNTHETIC FALLBACK. data_loader falls back to a generated curve when a
fetch fails, which is right for keeping a demo alive and wrong for building
an evaluation set -- two synthetic curves in the old cache scored AUC 0.21
and silently dragged the reported mean down. Here a failure is recorded and
skipped, never substituted.

Resumable and parallel: already-downloaded targets are skipped, so an
interrupted run continues where it stopped and raising the target count
only fetches the new ones.
"""

import contextlib
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from astronomy.koi_catalog import select_targets, target_summary, usable_kois

BULK_DIR = Path(__file__).parent.parent / "data" / "cache" / "lightcurves_bulk"
MANIFEST_PATH = BULK_DIR / "manifest.parquet"
FAILURES_PATH = BULK_DIR / "failures.json"

CADENCE_SECONDS = 1800      # Kepler long cadence -- the uniform choice
MISSION = "Kepler"
AUTHOR = "Kepler"           # the official pipeline product, not a community one

# Politeness, learned the hard way. Each worker calls download_all(), which
# itself pulls ~17 quarter files -- so N workers means roughly 17N requests
# in flight. At 6 workers MAST dropped 312 of 498 connections outright. 3 is
# the setting that actually finishes faster, because nothing gets refused.
DEFAULT_WORKERS = 3

# Transient-fault handling for the same reason.
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 4.0

MIN_POINTS = 5000           # below this a "stitched" curve is a failed stitch


def _say(*args) -> None:
    """Print that cannot kill a long run.

    A multi-hour download reporting progress through a closed pipe (a `head`
    upstream, a terminated `timeout`, a detached background shell) otherwise
    dies on a print AFTER every file is safely on disk -- which is exactly
    what happened on the first trial run. The data write always precedes the
    reporting, so swallowing this is losing a log line, not losing work.
    """
    try:
        print(*args, flush=True)
    except (ValueError, OSError):
        pass


def _path_for(kepid: int) -> Path:
    return BULK_DIR / f"KIC_{kepid}.parquet"


def is_cached(kepid: int) -> bool:
    return _path_for(kepid).exists()


def _fetch_stitched(kepid: int):
    """The network part, with lightkurve's own console output muted.

    lightkurve prints a "Downloading URL ..." line per quarter file straight
    to stdout. In a detached background run those prints raise
    "ValueError: I/O operation on closed file" from inside the worker
    thread, which the pool then reports as a download failure -- 104 of them
    on the first 500-target run, none of which were actually network
    problems. Redirecting here silences the library itself, which guarding
    our own prints could never do.
    """
    import lightkurve as lk

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        search = lk.search_lightcurve(f"KIC {kepid}", mission=MISSION,
                                      exptime=CADENCE_SECONDS, author=AUTHOR)
        if len(search) == 0:
            raise ValueError(f"no long-cadence {MISSION} products for KIC {kepid}")
        lc = search.download_all().stitch().remove_nans().normalize()
    return search, lc


def download_target(kepid: int, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """One star: every long-cadence quarter, stitched, normalised, cached.

    Retries on transient network faults with exponential backoff. MAST
    answers a burst of parallel requests by simply dropping connections
    ("Remote end closed connection without response" -- 312 times on the
    first run), and those targets are perfectly downloadable a moment later,
    so treating the first refusal as permanent throws away most of the
    dataset.

    Returns a manifest row. Raises on genuine failure -- the caller decides
    whether that is fatal or just skipped.
    """
    started = time.time()

    last_error = None
    for attempt in range(max_attempts):
        try:
            search, lc = _fetch_stitched(kepid)
            break
        except ValueError:
            raise                       # "no products" is not transient
        except Exception as e:          # network faults: back off and retry
            last_error = e
            if attempt == max_attempts - 1:
                raise
            time.sleep(BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1.5))
    else:                                # pragma: no cover - loop always breaks or raises
        raise last_error

    time_v = np.asarray(lc.time.value, dtype=float)
    flux_v = np.asarray(lc.flux.value, dtype=float)

    keep = np.isfinite(time_v) & np.isfinite(flux_v)
    time_v, flux_v = time_v[keep], flux_v[keep]
    if len(time_v) < MIN_POINTS:
        raise ValueError(f"only {len(time_v)} usable points after stitching")

    BULK_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": time_v, "flux": flux_v}).to_parquet(_path_for(kepid))

    return {
        "kepid": kepid,
        "n_points": len(time_v),
        "span_days": round(float(time_v[-1] - time_v[0]), 2),
        "cadence_min": round(float(np.median(np.diff(time_v))) * 24 * 60, 2),
        "n_quarters": len(search),
        "fetch_seconds": round(time.time() - started, 1),
    }


def build_bulk_dataset(n_targets: int = 500, workers: int = DEFAULT_WORKERS,
                       balanced: bool = True, seed: int = 0,
                       verbose: bool = True) -> pd.DataFrame:
    """Download a labelled sample of Kepler targets in parallel.

    Safe to re-run and safe to interrupt: cached targets are skipped, so
    raising n_targets later fetches only the difference.
    """
    BULK_DIR.mkdir(parents=True, exist_ok=True)
    targets = select_targets(n_targets, balanced=balanced, seed=seed)
    koi = usable_kois()

    todo = [int(k) for k in targets["kepid"] if not is_cached(int(k))]
    cached = len(targets) - len(todo)
    if verbose:
        _say(f"{len(targets)} targets selected "
              f"({targets['label'].value_counts().to_dict()})")
        _say(f"  {cached} already cached, {len(todo)} to fetch "
              f"with {workers} workers")
        if todo:
            _say(f"  estimated ~{len(todo) * 80 / workers / 60:.0f} min\n")

    rows, failures = [], []
    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_target, k): k for k in todo}
            for fut in as_completed(futures):
                kepid = futures[fut]
                done += 1
                try:
                    rows.append(fut.result())
                    if verbose and done % 10 == 0:
                        _say(f"  [{done}/{len(todo)}] ok, {len(failures)} failed")
                except Exception as e:
                    failures.append({"kepid": kepid,
                                     "error": f"{type(e).__name__}: {e}"})

    # rebuild the manifest from what is actually on disk, so it always
    # reflects reality rather than the outcome of the most recent run
    manifest = _rebuild_manifest(targets, koi, verbose=verbose)

    if failures:
        existing = json.loads(FAILURES_PATH.read_text()) if FAILURES_PATH.exists() else []
        FAILURES_PATH.write_text(json.dumps(existing + failures, indent=2))
        if verbose:
            _say(f"\n  {len(failures)} failures recorded -> {FAILURES_PATH.name}")

    return manifest


def _rebuild_manifest(targets: pd.DataFrame, koi: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """One row per successfully cached target: shape, label, and what the
    archive says is really there."""
    label_by_kepid = dict(zip(targets["kepid"], targets["label"]))

    rows = []
    for path in sorted(BULK_DIR.glob("KIC_*.parquet")):
        kepid = int(path.stem.split("_")[1])
        df = pd.read_parquet(path)
        t = df["time"].values
        info = target_summary(kepid, koi)
        rows.append({
            "kepid": kepid,
            "label": label_by_kepid.get(kepid, info.get("label")),
            "n_points": len(df),
            "span_days": round(float(t[-1] - t[0]), 2),
            "cadence_min": round(float(np.median(np.diff(t))) * 24 * 60, 2),
            "n_kois": info.get("n_kois", 0),
            "n_confirmed": info.get("n_confirmed", 0),
            "file": path.name,
        })

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest.to_parquet(MANIFEST_PATH)
        if verbose:
            _say(f"\nmanifest: {len(manifest)} targets cached")
            _say(f"  {manifest['label'].value_counts().to_dict()}")
            _say(f"  median {manifest['n_points'].median():.0f} points, "
                  f"{manifest['span_days'].median():.0f} day span, "
                  f"{manifest['cadence_min'].median():.1f} min cadence")
            _say(f"  total {sum(p.stat().st_size for p in BULK_DIR.glob('*.parquet')) / 1e6:.0f} MB")
    return manifest


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()


def load_target(kepid: int) -> pd.DataFrame:
    path = _path_for(kepid)
    if not path.exists():
        raise FileNotFoundError(f"KIC {kepid} not in the bulk cache -- run build_bulk_dataset()")
    return pd.read_parquet(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Bulk-download labelled Kepler light curves.")
    ap.add_argument("-n", "--n-targets", type=int, default=500)
    ap.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    build_bulk_dataset(n_targets=args.n_targets, workers=args.workers, seed=args.seed)
