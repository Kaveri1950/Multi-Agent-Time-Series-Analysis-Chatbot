"""Score the LSTM autoencoder on the committed Kepler split.

WHY THIS EXISTS RATHER THAN A GRID RE-RUN

`LSTMAutoencoder` is implemented in `anomaly/detectors.py` but was never
listed in `default_detectors()`, so it has never appeared in
`anomaly_grid.json` and no performance claim has ever been backed by a
committed number for it. Closing that gap by re-running the grid is
disproportionate: `anomaly/grid.py` exposes only `--kepler-limit`,
`--no-save` and `--jobs`, so it re-runs all 21 Kepler detectors, and its
wall-clock is floored by `matrix_profile_discord_raw` at 11,698 s -- about
3.25 hours to obtain one missing row.

This module instead reproduces `grid.run_domain`'s `evaluate_one` for
exactly one detector: same loader, same target limit, same
`make_split(seed=0)`, same `_score_dataset` / `score_held_out`, same metric
calls. The row it emits therefore carries the same meaning as the committed
rows and can be read directly against them.

WHAT IT DOES NOT DO

It does not redo ensemble selection. `select_ensemble` needs every
detector's TRAINING-half scores, and those are not cached by the grid, so
whether the LSTM would have displaced an ensemble member is not answerable
without the full re-run. The committed `ensemble_4` row is left alone.

Run:  python -m evaluation.lstm_addendum [--skip-in-sample]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"

#: Must match the committed anomaly_grid.json Kepler run, whose meta records
#: n_targets=40 / n_windows=384,281. `anomaly/grid.py`'s own default is 60,
#: which pools 60 targets and yields a different split entirely.
KEPLER_LIMIT = 40
SEED = 0
TOP_K = 50


def committed_kepler_rows() -> list[dict]:
    """The Kepler rows already in anomaly_grid.json, for comparison."""
    path = RESULTS / "anomaly_grid.json"
    if not path.exists():
        return []
    grid = json.loads(path.read_text())
    return [r for r in grid.get("rows", []) if r.get("domain") == "kepler"]


def run(skip_in_sample: bool = False) -> dict:
    warnings.filterwarnings("ignore")
    from anomaly.detectors import LSTMAutoencoder
    from anomaly.grid import _score_dataset, load_kepler
    from anomaly.metrics import evaluate_scores, precision_at_k
    from anomaly.splits import make_split, score_held_out

    print("loading kepler ...", flush=True)
    ds = load_kepler(limit=KEPLER_LIMIT)
    train_mask, test_mask = make_split(ds, seed=SEED)
    y_test = ds.y[test_mask]
    print(f"  {ds.meta['n_targets']} targets, {ds.meta['n_windows']:,} windows", flush=True)
    print(f"  train {train_mask.sum():,} / test {test_mask.sum():,}"
          f"  ({int(y_test.sum())} positive, {y_test.mean():.4f})", flush=True)

    det = LSTMAutoencoder()
    row: dict = {"domain": "kepler", "detector": det.name, "group": det.group,
                 "is_floor": False, "is_ceiling": False}
    started = time.time()

    # HELD-OUT -- the headline. One fit on the training targets, then each
    # test target scored and rank-normalised within itself, exactly as
    # splits.score_held_out does for every other detector.
    print("\nheld-out pass (one fit on training targets) ...", flush=True)
    t0 = time.time()
    held = score_held_out(det, ds, train_mask, test_mask)
    metrics = evaluate_scores(y_test, held)
    row.update(metrics)
    row[f"precision_at_{TOP_K}"] = precision_at_k(y_test, held, TOP_K)
    print(f"  held-out pr_auc {metrics['pr_auc']:.4f}"
          f"  roc_auc {metrics.get('roc_auc', float('nan')):.4f}"
          f"  lift {metrics.get('lift', float('nan')):.2f}x"
          f"  {time.time() - t0:.1f}s", flush=True)

    # IN-SAMPLE -- 40 per-target fits. Kept because the in-sample/held-out
    # gap is the thing that separates a memorising detector from a useful
    # one, and an LSTM has far more capacity than most of the grid.
    if skip_in_sample:
        row["in_sample_pr_auc"] = None
        row["in_sample_lift"] = None
        print("\nin-sample pass SKIPPED (--skip-in-sample)")
    else:
        print(f"\nin-sample pass ({ds.meta['n_targets']} per-target fits) ...", flush=True)
        t0 = time.time()
        in_sample = evaluate_scores(ds.y, _score_dataset(det, ds))
        row["in_sample_pr_auc"] = in_sample.get("pr_auc")
        row["in_sample_lift"] = in_sample.get("lift")
        print(f"  in-sample pr_auc {in_sample.get('pr_auc'):.4f}"
              f"  lift {in_sample.get('lift', float('nan')):.2f}x"
              f"  {time.time() - t0:.1f}s", flush=True)

    row["seconds"] = round(time.time() - started, 2)

    # ---------------------------------------------------------- comparison
    others = committed_kepler_rows()
    floor = next((r for r in others if r["detector"] == "raw_input"), None)
    ceiling = next((r for r in others if r.get("is_ceiling")), None)
    ranked = sorted([r for r in others if np.isfinite(r.get("pr_auc") or np.nan)]
                    + [row], key=lambda r: -(r.get("pr_auc") or 0))

    print(f"\n{'=' * 78}\nKEPLER HELD-OUT PR-AUC, lstm_autoencoder against the committed grid"
          f"\n{'=' * 78}")
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
        verdict["vs_floor_raw_input"] = {
            "floor_pr_auc": floor["pr_auc"], "lstm_pr_auc": row["pr_auc"],
            "margin": margin, "beats_floor": bool(margin > 0),
        }
        print(f"\nvs raw_input floor ({floor['pr_auc']:.4f}): "
              f"{margin:+.4f}  -> {'BEATS' if margin > 0 else 'DOES NOT BEAT'} the floor")
    if ceiling:
        verdict["vs_supervised_ceiling"] = {
            "ceiling_pr_auc": ceiling["pr_auc"],
            "margin": row["pr_auc"] - ceiling["pr_auc"]}
    best = max((r for r in others if not r.get("is_ceiling")
                and np.isfinite(r.get("pr_auc") or np.nan)),
               key=lambda r: r["pr_auc"], default=None)
    if best:
        verdict["vs_best_committed"] = {
            "detector": best["detector"], "pr_auc": best["pr_auc"],
            "margin": row["pr_auc"] - best["pr_auc"]}
        print(f"vs best committed ({best['detector']} {best['pr_auc']:.4f}): "
              f"{row['pr_auc'] - best['pr_auc']:+.4f}")
        print(f"rank among Kepler detectors: "
              f"{[r['detector'] for r in ranked].index(det.name) + 1} of {len(ranked)}")

    out = {
        "note": ("LSTM autoencoder scored on the committed Kepler split "
                 "(40 targets, make_split seed=0). Produced by "
                 "evaluation/lstm_addendum.py, which mirrors grid.run_domain's "
                 "evaluate_one for a single detector. anomaly_grid.json is "
                 "deliberately not modified. Ensemble selection is not redone."),
        "kepler_limit": KEPLER_LIMIT, "seed": SEED,
        "split": {"train_windows": int(train_mask.sum()),
                  "test_windows": int(test_mask.sum()),
                  "test_positive": int(y_test.sum()),
                  "test_positive_rate": float(y_test.mean()),
                  "test_groups": int(len(np.unique(ds.groups[test_mask])))},
        "detector_config": {"hidden": det.hidden, "epochs": det.epochs,
                            "batch": det.batch, "lr": det.lr,
                            "seed": det.seed, "max_train": det.max_train},
        "row": row, "verdict": verdict,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "lstm_addendum.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-in-sample", action="store_true",
                    help="held-out only; skips the 40 per-target fits")
    args = ap.parse_args()
    run(skip_in_sample=args.skip_in_sample)
