"""Score one named detector on the committed anomaly split.

WHY A SEPARATE ENTRY POINT. `anomaly/grid.py` has no per-detector flag, so
adding a single detector means re-running all 21 and waiting on
matrix_profile_discord_raw's 11,698 s -- about 3.25 hours to obtain one row.
This reproduces `grid.run_domain`'s `evaluate_one` for exactly one detector:
same loader, same target limit, same make_split(seed=0), same
_score_dataset / score_held_out, same metric calls. The row it emits
therefore means the same thing as the committed rows.

It generalises evaluation/lstm_addendum.py, which did this for one hard-coded
detector. A detector that is in `default_detectors()` but has never produced
a committed number is the exact gap the LSTM sat in for months, and it should
be cheap to close rather than something that waits for a full grid run.

Run:  python -m evaluation.detector_addendum --detector cusum
      python -m evaluation.detector_addendum --detector cusum --domain both
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"
KEPLER_LIMIT = 40          # matches the committed anomaly_grid.json run
SEED = 0
TOP_K = 50


def committed_rows(domain: str) -> list:
    path = RESULTS / "anomaly_grid.json"
    if not path.exists():
        return []
    grid = json.loads(path.read_text())
    return [r for r in grid.get("rows", []) if r.get("domain") == domain]


def score_one(detector: str, domain: str, skip_in_sample: bool = False,
              verbose: bool = True) -> dict:
    warnings.filterwarnings("ignore")
    from anomaly.detectors import default_detectors
    from anomaly.grid import _score_dataset, load_equity, load_kepler
    from anomaly.metrics import evaluate_scores, precision_at_k
    from anomaly.splits import make_split, score_held_out

    if verbose:
        print(f"\nloading {domain} ...", flush=True)
    ds = load_equity() if domain == "equity" else load_kepler(limit=KEPLER_LIMIT)
    train_mask, test_mask = make_split(ds, seed=SEED)
    y_test = ds.y[test_mask]

    stride = int(ds.meta.get("stride", 1))
    pool = {d.name: d for d in default_detectors(
        include_finance_only=ds.finance_only, stride=stride)}
    if detector not in pool:
        raise SystemExit(f"{detector!r} is not in default_detectors(); "
                         f"have {sorted(pool)}")
    det = pool[detector]

    if verbose:
        print(f"  train {train_mask.sum():,} / test {test_mask.sum():,}"
              f"  ({int(y_test.sum())} positive, {y_test.mean():.4f})", flush=True)

    row = {"domain": domain, "detector": det.name, "group": det.group,
           "is_floor": bool(det.is_floor), "is_ceiling": bool(det.is_ceiling)}
    started = time.time()

    t0 = time.time()
    held = score_held_out(det, ds, train_mask, test_mask)
    metrics = evaluate_scores(y_test, held)
    row.update(metrics)
    row[f"precision_at_{TOP_K}"] = precision_at_k(y_test, held, TOP_K)
    if verbose:
        print(f"  held-out pr_auc {metrics['pr_auc']:.4f}"
              f"  roc_auc {metrics.get('roc_auc', float('nan')):.4f}"
              f"  lift {metrics.get('lift', float('nan')):.2f}x"
              f"  {time.time() - t0:.1f}s", flush=True)

    if skip_in_sample:
        row["in_sample_pr_auc"] = row["in_sample_lift"] = None
    else:
        t0 = time.time()
        in_sample = evaluate_scores(ds.y, _score_dataset(det, ds))
        row["in_sample_pr_auc"] = in_sample.get("pr_auc")
        row["in_sample_lift"] = in_sample.get("lift")
        if verbose:
            print(f"  in-sample pr_auc {in_sample.get('pr_auc'):.4f}"
                  f"  {time.time() - t0:.1f}s", flush=True)
    row["seconds"] = round(time.time() - started, 2)

    others = [r for r in committed_rows(domain) if r["detector"] != det.name]
    ranked = sorted([r for r in others if np.isfinite(r.get("pr_auc") or np.nan)]
                    + [row], key=lambda r: -(r.get("pr_auc") or 0))
    floor = next((r for r in others if r["detector"] == "raw_input"), None)
    best = max((r for r in others if not r.get("is_ceiling")
                and np.isfinite(r.get("pr_auc") or np.nan)),
               key=lambda r: r["pr_auc"], default=None)

    if verbose:
        print(f"\n{'=' * 72}\n{domain.upper()} held-out PR-AUC, {det.name} against "
              f"the committed grid\n{'=' * 72}")
        print(f"{'detector':<32}{'group':<16}{'pr_auc':>9}{'lift':>8}{'seconds':>11}")
        for r in ranked:
            mark = "  <-- NEW" if r["detector"] == det.name else ""
            tag = " FLOOR" if r.get("is_floor") else (" CEIL" if r.get("is_ceiling") else "")
            print(f"{r['detector']:<32}{r.get('group', ''):<16}"
                  f"{r.get('pr_auc', float('nan')):>9.4f}"
                  f"{r.get('lift', float('nan')):>8.2f}"
                  f"{r.get('seconds', float('nan')):>11.1f}{tag}{mark}")

    verdict = {}
    if floor:
        margin = row["pr_auc"] - floor["pr_auc"]
        verdict["vs_floor_raw_input"] = {"floor_pr_auc": floor["pr_auc"],
                                         "margin": margin, "beats_floor": bool(margin > 0)}
        if verbose:
            print(f"\nvs raw_input floor ({floor['pr_auc']:.4f}): {margin:+.4f}"
                  f"  -> {'BEATS' if margin > 0 else 'DOES NOT BEAT'} the floor")
    if best:
        verdict["vs_best_committed"] = {"detector": best["detector"],
                                        "pr_auc": best["pr_auc"],
                                        "margin": row["pr_auc"] - best["pr_auc"]}
        rank = [r["detector"] for r in ranked].index(det.name) + 1
        verdict["rank"] = f"{rank} of {len(ranked)}"
        if verbose:
            print(f"vs best committed ({best['detector']} {best['pr_auc']:.4f}): "
                  f"{row['pr_auc'] - best['pr_auc']:+.4f}")
            print(f"rank: {rank} of {len(ranked)}")

    return {"row": row, "verdict": verdict,
            "split": {"train_windows": int(train_mask.sum()),
                      "test_windows": int(test_mask.sum()),
                      "test_positive": int(y_test.sum()),
                      "test_positive_rate": float(y_test.mean())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--domain", default="both", choices=["kepler", "equity", "both"])
    ap.add_argument("--skip-in-sample", action="store_true")
    args = ap.parse_args()

    domains = ["equity", "kepler"] if args.domain == "both" else [args.domain]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "detector": args.detector, "kepler_limit": KEPLER_LIMIT, "seed": SEED,
           "note": ("Scored by evaluation/detector_addendum.py, which mirrors "
                    "grid.run_domain's evaluate_one for a single detector. "
                    "anomaly_grid.json is deliberately not modified, and ensemble "
                    "selection is not redone."),
           "domains": {}}
    for dom in domains:
        out["domains"][dom] = score_one(args.detector, dom,
                                        skip_in_sample=args.skip_in_sample)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"detector_addendum_{args.detector}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
