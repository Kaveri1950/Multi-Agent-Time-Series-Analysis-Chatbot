"""Run the detectors offline, store the results, let the chatbot read them.

WHY PRECOMPUTE

Measured cost of answering "any anomalies in KIC 10001368?" live:

    load cached light curve     0.01 s
    latent_isoforest            0.68 s      <- fine to run live
    hmm_3state                  9.36 s      <- too slow
    BLS transit search          7.95 s      <- too slow
    matrix_profile_discord     63.73 s      <- far too slow

The cheap detectors could run per query. The expensive ones cannot, and more
importantly they SHOULD not: a target's orbital period does not change
between questions. Computing it afresh each time is waste, not caution.

WHAT THE BOT DOES AND DOES NOT DO

It does NOT compare detector scores at query time. Those scores are not
comparable -- on one window Mahalanobis reports 44.22 where IsolationForest
reports 0.60, and neither is "more confident". The comparison already
happened offline, against labels, in `anomaly/grid.py`. The registry records
which detector won on which domain, and the bot looks that up.

What the bot CAN usefully read off several detectors is AGREEMENT: "9 of 12
flagged this window" is a confidence signal, and a genuinely different thing
from picking a winner. Both are stored.

STORAGE

MongoDB when configured, parquet otherwise, so the pipeline runs on a laptop
with no database and the chatbot path is identical either way.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGODB_DB", "btp_chat")
COLLECTION = "anomaly_results"

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "anomaly"
PARQUET_PATH = CACHE_DIR / "precomputed.parquet"

#: Detectors cheap enough to run live if the cache ever misses. Measured,
#: not guessed -- everything here came in under a second on one target.
LIVE_SAFE = {"raw_input", "moving_variance", "zscore", "hampel_mad",
             "spectral_pca", "isolation_forest", "latent_isoforest",
             "local_outlier_factor", "one_class_svm"}

#: How many flagged windows to keep per target. The chatbot shows a handful;
#: storing every window would turn a 440-target run into 2 million rows for
#: no benefit.
TOP_N = 20


def _mongo():
    """Collection handle, or None when no database is reachable."""
    try:
        from pymongo import MongoClient

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        coll = client[MONGO_DB][COLLECTION]
        coll.create_index([("domain", 1), ("key", 1)], unique=True)
        return coll
    except Exception:
        return None


def analyse_kepler_target(kepid: int, detectors: list[str] | None = None,
                          run_bls: bool = True) -> dict:
    """Everything the chatbot might be asked about one star, computed once.

    Deliberately stores BLS alongside the anomaly detectors even though they
    answer different questions. "Is anything odd here?" and "is there a
    planet here?" both arrive as questions about the same target, and the
    honest answer to the second is a period and an SDE, not an anomaly score.
    """
    from anomaly.core import windowize
    from anomaly.detectors import default_detectors
    from anomaly.explain import annotate_flagged
    from anomaly.grid import WINDOW_CONFIG
    from anomaly.labels import astronomy_window_labels
    from astronomy.bulk_download import load_target
    from astronomy.koi_catalog import fetch_koi_table

    cfg = WINDOW_CONFIG["kepler"]
    df = load_target(int(kepid))
    time_v, flux = df["time"].to_numpy(), df["flux"].to_numpy()
    ok = np.isfinite(time_v) & np.isfinite(flux)
    time_v, flux = time_v[ok], flux[ok]

    windows = windowize(flux, cfg["window"], cfg["stride"])
    koi = fetch_koi_table()
    try:
        y = astronomy_window_labels(time_v, int(kepid), cfg["window"], cfg["stride"], koi)
        n = min(len(windows), len(y))
        windows, y = windows[:n], y[:n]
    except Exception:
        y = None

    wanted = set(detectors) if detectors else LIVE_SAFE
    scores, timings = {}, {}
    for det in default_detectors(stride=cfg["stride"]):
        if det.is_ceiling or det.name not in wanted:
            continue
        import time as _t
        t0 = _t.time()
        try:
            scores[det.name] = det.fit_score(windows, y)
            timings[det.name] = round(_t.time() - t0, 3)
        except Exception:
            continue

    record: dict = {
        "domain": "kepler", "key": f"KIC {int(kepid)}", "kepid": int(kepid),
        "n_points": int(len(flux)), "n_windows": int(len(windows)),
        "window": cfg["window"], "stride": cfg["stride"],
        "time_start": float(time_v[0]), "time_end": float(time_v[-1]),
        "detector_seconds": timings,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Agreement: how many detectors independently put a window in their own
    # top 2%. This is a confidence signal, NOT a way of choosing a winner.
    if scores:
        flags = np.zeros(len(windows), dtype=int)
        for s in scores.values():
            flags += (s >= np.quantile(s, 0.98)).astype(int)
        record["n_detectors"] = len(scores)
        record["max_agreement"] = int(flags.max())
        record["n_unanimous"] = int((flags == len(scores)).sum())

        primary = "latent_isoforest" if "latent_isoforest" in scores else next(iter(scores))
        record["primary_detector"] = primary
        ann = annotate_flagged(windows, scores[primary], "kepler", y, top_n=TOP_N,
                               window=cfg["window"], stride=cfg["stride"])
        ann["agreement"] = [int(flags[int(i)]) for i in ann["window"]]
        ann["time"] = [float(time_v[min(int(i), len(time_v) - 1)]) for i in ann["start_index"]]
        record["detections"] = ann.to_dict("records")

    if y is not None and y.sum():
        from anomaly.metrics import evaluate_scores
        record["label_prevalence"] = round(float(y.mean()), 5)
        record["skill"] = {k: evaluate_scores(y, v)["pr_auc"] for k, v in scores.items()}

    if run_bls:
        try:
            from astronomy.transit_search import compare_with_truth, search
            bls = search(time_v, flux)
            cmp = compare_with_truth(bls, int(kepid), koi)
            record["bls"] = {
                "period": bls["period"], "epoch": bls["epoch"],
                "duration": bls["duration"], "depth_ppm": bls["depth"] * 1e6,
                "sde": bls["sde"],
                # SDE ~7 is the conventional threshold in the transit
                # literature for calling a periodogram peak a detection.
                "is_detection": bool(bls["sde"] >= 7.0),
                "matches_nasa": bool(cmp.get("matched", False)),
                "nasa_period": cmp.get("true_period"),
                "period_error_pct": cmp.get("period_error_pct"),
                "harmonic": cmp.get("harmonic"),
            }
        except Exception as exc:  # noqa: BLE001
            record["bls_error"] = f"{type(exc).__name__}: {exc}"

    return record


def _store(records: list[dict], verbose: bool = True) -> str:
    """Upsert to Mongo, falling back to parquet. Returns where it went."""
    coll = _mongo()
    if coll is not None:
        from pymongo import UpdateOne

        ops = [UpdateOne({"domain": r["domain"], "key": r["key"]},
                         {"$set": r}, upsert=True) for r in records]
        if ops:
            coll.bulk_write(ops, ordered=False)
        if verbose:
            print(f"  stored {len(records)} records -> MongoDB {MONGO_DB}.{COLLECTION}")
        return "mongodb"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    # Nested lists and dicts survive a parquet round-trip only as JSON.
    import json
    for col in ("detections", "skill", "detector_seconds", "bls"):
        if col in frame.columns:
            frame[col] = frame[col].apply(lambda v: json.dumps(v, default=str))
    if PARQUET_PATH.exists():
        old = pd.read_parquet(PARQUET_PATH)
        frame = pd.concat([old[~old["key"].isin(frame["key"])], frame], ignore_index=True)
    frame.to_parquet(PARQUET_PATH)
    if verbose:
        print(f"  no MongoDB reachable -- stored {len(records)} records -> {PARQUET_PATH}")
    return "parquet"


def fetch(domain: str, key: str) -> dict | None:
    """What the chatbot calls. One lookup, no computation."""
    coll = _mongo()
    if coll is not None:
        return coll.find_one({"domain": domain, "key": key}, {"_id": 0})
    if not PARQUET_PATH.exists():
        return None
    frame = pd.read_parquet(PARQUET_PATH)
    hit = frame[(frame["domain"] == domain) & (frame["key"] == key)]
    if hit.empty:
        return None
    import json
    row = hit.iloc[0].to_dict()
    for col in ("detections", "skill", "detector_seconds", "bls"):
        if isinstance(row.get(col), str):
            try:
                row[col] = json.loads(row[col])
            except json.JSONDecodeError:
                pass
    return row


def run_kepler(limit: int | None = None, n_jobs: int = 6, run_bls: bool = True,
               verbose: bool = True) -> int:
    """Precompute every cached target. Parallel across targets.

    Targets are independent, so this is embarrassingly parallel. Measured on
    12 logical cores: 5.61x at n_jobs=12 with bit-identical results. The
    default is deliberately lower -- sklearn and numpy already use threads
    internally, and oversubscribing runs SLOWER, so workers are pinned to one
    thread each below.
    """
    from joblib import Parallel, delayed
    from astronomy.bulk_download import load_manifest

    manifest = load_manifest()
    if limit:
        manifest = manifest.head(limit)
    kepids = [int(r["kepid"]) for _, r in manifest.iterrows()]
    if verbose:
        print(f"precomputing {len(kepids)} Kepler targets, n_jobs={n_jobs}, BLS={run_bls}")

    def one(kid):
        # Pin BLAS/OMP to one thread per worker: 6 processes each spawning 12
        # threads oversubscribes 12 cores and is slower than serial.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = "1"
        try:
            return analyse_kepler_target(kid, run_bls=run_bls)
        except Exception as exc:  # noqa: BLE001
            return {"domain": "kepler", "key": f"KIC {kid}", "kepid": kid,
                    "error": f"{type(exc).__name__}: {exc}"}

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=5 if verbose else 0)(
        delayed(one)(k) for k in kepids)
    ok = [r for r in records if "error" not in r]
    if verbose:
        print(f"  {len(ok)} succeeded, {len(records) - len(ok)} failed")
    _store(records, verbose=verbose)
    return len(ok)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--no-bls", action="store_true")
    args = ap.parse_args()

    n = run_kepler(limit=args.limit, n_jobs=args.jobs, run_bls=not args.no_bls)
    print(f"\ndone: {n} targets precomputed")
    sample = fetch("kepler", f"KIC {__import__('astronomy.bulk_download', fromlist=['x']).load_manifest().iloc[0]['kepid']:.0f}")
    if sample:
        print(f"lookup check: {sample['key']} -> "
              f"{len(sample.get('detections', []))} detections, "
              f"BLS SDE {sample.get('bls', {}).get('sde', 'n/a')}")
