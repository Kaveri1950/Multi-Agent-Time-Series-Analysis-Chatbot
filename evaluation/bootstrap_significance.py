"""Bootstrap confidence intervals on the anomaly-detection PR-AUC margins.

THE QUESTION. The committed grid reports two margins over the no-skill floor:

    Kepler   matrix_profile_discord_raw  0.2630  vs  raw_input  0.2528   (+0.0102)
    equity   spectral_pca                0.0355  vs  raw_input  0.0325   (+0.0030)

Both are small. A point estimate cannot say whether either is distinguishable
from sampling noise, and the report should not claim a detector "beats the
floor" without that. This module answers it by resampling the held-out set.

DESIGN -- three things that matter.

1. PAIRED resampling. The two detectors are scored on the SAME held-out
   windows, so each bootstrap replicate applies ONE index vector to both and
   the statistic is their difference. Resampling them independently would
   inflate the variance of the difference and understate significance.

2. THE WINDOWS ARE NOT INDEPENDENT, and a naive window-level bootstrap
   ignores that. Kepler windows are 24 points at stride 6, so neighbours
   share 18 of 24 points; equity windows are 21 days at stride 1, sharing 20
   of 21. Treating them as independent draws makes the effective sample size
   far larger than it really is and produces intervals that are too narrow.
   The naive interval is therefore reported as ANTI-CONSERVATIVE, alongside a
   second estimator that respects the dependence:

       Kepler  cluster bootstrap -- resample whole TARGETS with replacement.
               Windows within a star are dependent; different stars are not.
       equity  moving-block bootstrap -- resample contiguous blocks of length
               21 (one window span), which preserves local dependence in a
               single time series.

3. AP is undefined when a replicate contains no positives, so such replicates
   are discarded and counted rather than silently scored as zero.

Held-out scores are not saved by anomaly/grid.py, so they are recomputed here
and cached to .npz -- the Kepler matrix profile costs roughly an hour.

Run:  python -m evaluation.bootstrap_significance [--domain kepler|equity|both]
Out:  evaluation/results/bootstrap_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

RESULTS = Path(__file__).parent / "results"
OUT_JSON = RESULTS / "bootstrap_results.json"
CACHE = RESULTS / "bootstrap_scores_{domain}.npz"

N_BOOT = 2000
SEED = 0
ALPHA = 0.05            # 95% interval

# (detector under test, floor it is compared against), per domain
COMPARISON = {
    "kepler": ("matrix_profile_discord_raw", "raw_input"),
    "equity": ("spectral_pca", "raw_input"),
}
# Block length for the equity moving-block bootstrap: one window span, so a
# block covers exactly the dependence introduced by the sliding window.
EQUITY_BLOCK = 21

#: Targets pooled for Kepler. MUST match the committed anomaly_grid.json run,
#: whose meta records n_targets=40 / n_windows=384,281 -- otherwise this script
#: bootstraps a different dataset than the margin it claims to be testing.
#: At 60 (anomaly/grid.py's own default) the loader pools 60 targets and the
#: stratified group split hands back 20 test groups / 200,157 held-out windows,
#: against the committed 13 groups / 124,782. The interval would be real but it
#: would not belong to the 0.2630-vs-0.2528 margin anyone is citing.
KEPLER_LIMIT = 40


# --------------------------------------------------------------- held-out
def compute_held_out(domain: str, verbose: bool = True) -> dict:
    """Held-out labels and per-window scores for the two detectors.

    Reproduces anomaly/grid.py's pipeline exactly -- same loader, same
    WINDOW_CONFIG, same make_split(seed=0), same score_held_out -- so the
    scores here correspond to the committed PR-AUC values.
    """
    path = Path(str(CACHE).format(domain=domain))
    if path.exists():
        d = np.load(path, allow_pickle=True)
        if verbose:
            print(f"  loaded cached scores <- {path.name}")
        return {"y": d["y"], "groups": d["groups"] if d["groups"].ndim else None,
                "scores": {k: d[f"score_{k}"] for k in COMPARISON[domain]}}

    import warnings
    warnings.filterwarnings("ignore")
    from anomaly.detectors import default_detectors
    from anomaly.grid import load_equity, load_kepler
    from anomaly.splits import make_split, score_held_out

    if verbose:
        print(f"  loading {domain} ...", flush=True)
    ds = load_equity() if domain == "equity" else load_kepler(limit=KEPLER_LIMIT)
    train_mask, test_mask = make_split(ds, seed=SEED)
    y = ds.y[test_mask]
    groups = ds.groups[test_mask] if ds.groups is not None else None
    if verbose:
        print(f"  held out {test_mask.sum():,} windows, {int(y.sum())} positive"
              + (f", {len(np.unique(groups))} groups" if groups is not None else ""),
              flush=True)

    wanted = set(COMPARISON[domain])
    by_name = {d.name: d for d in default_detectors(
        include_finance_only=ds.finance_only, stride=int(ds.meta.get("stride", 1)))}
    missing = wanted - set(by_name)
    if missing:
        raise SystemExit(f"detector(s) not in default_detectors(): {missing}")

    scores = {}
    for name in COMPARISON[domain]:
        t0 = time.time()
        scores[name] = score_held_out(by_name[name], ds, train_mask, test_mask)
        ap = average_precision_score(y, scores[name])
        if verbose:
            print(f"    {name:<28} PR-AUC {ap:.4f}   {time.time()-t0:8.1f}s", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=y,
                        groups=groups if groups is not None else np.array(0),
                        **{f"score_{k}": v for k, v in scores.items()})
    if verbose:
        print(f"  cached -> {path.name}")
    return {"y": y, "groups": groups, "scores": scores}


# --------------------------------------------------------------- bootstrap
def _ap(y, s):
    """Average precision, or nan when the replicate has no positives."""
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def _summarise(diffs, observed, n_skipped, method, n_units):
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    lo, hi = np.percentile(d, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "method": method,
        "n_resamples_requested": N_BOOT,
        "n_resamples_used": int(len(d)),
        "n_resamples_skipped_no_positives": int(n_skipped),
        "resampling_unit": n_units,
        "observed_difference": round(float(observed), 6),
        "bootstrap_mean_difference": round(float(d.mean()), 6),
        "bootstrap_std": round(float(d.std(ddof=1)), 6),
        "ci95_low": round(float(lo), 6),
        "ci95_high": round(float(hi), 6),
        "ci95_excludes_zero": bool(lo > 0 or hi < 0),
        "p_difference_gt_zero": round(float((d > 0).mean()), 4),
    }


def bootstrap_windows(y, a, b, n=N_BOOT, seed=SEED):
    """Naive i.i.d. window resampling -- what was asked for, and
    ANTI-CONSERVATIVE here because neighbouring windows overlap."""
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    diffs, skipped = [], 0
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yy = y[idx]
        if not 0 < yy.sum() < len(yy):
            skipped += 1
            continue
        diffs.append(_ap(yy, a[idx]) - _ap(yy, b[idx]))
    return _summarise(diffs, _ap(y, a) - _ap(y, b), skipped,
                      "iid window bootstrap (anti-conservative: windows overlap)",
                      f"{n_obs} windows")


def bootstrap_clusters(y, a, b, groups, n=N_BOOT, seed=SEED):
    """Resample whole groups (Kepler targets) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs, skipped = [], 0
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        yy = y[idx]
        if not 0 < yy.sum() < len(yy):
            skipped += 1
            continue
        diffs.append(_ap(yy, a[idx]) - _ap(yy, b[idx]))
    return _summarise(diffs, _ap(y, a) - _ap(y, b), skipped,
                      "cluster bootstrap over targets (respects within-target dependence)",
                      f"{len(uniq)} targets")


def bootstrap_blocks(y, a, b, block=EQUITY_BLOCK, n=N_BOOT, seed=SEED):
    """Moving-block bootstrap -- contiguous runs, for one dependent series."""
    rng = np.random.default_rng(seed)
    n_obs = len(y)
    n_blocks = int(np.ceil(n_obs / block))
    starts = np.arange(max(n_obs - block + 1, 1))
    diffs, skipped = [], 0
    for _ in range(n):
        s = rng.choice(starts, n_blocks, replace=True)
        idx = np.concatenate([np.arange(x, min(x + block, n_obs)) for x in s])[:n_obs]
        yy = y[idx]
        if not 0 < yy.sum() < len(yy):
            skipped += 1
            continue
        diffs.append(_ap(yy, a[idx]) - _ap(yy, b[idx]))
    return _summarise(diffs, _ap(y, a) - _ap(y, b), skipped,
                      f"moving-block bootstrap, block={block} (respects serial dependence)",
                      f"{n_obs} windows in {n_blocks} blocks of {block}")


# --------------------------------------------------------------- driver
def run_domain(domain: str) -> dict:
    print(f"\n{'=' * 74}\n{domain.upper()}\n{'=' * 74}")
    det, floor = COMPARISON[domain]
    data = compute_held_out(domain)
    y, scores = data["y"], data["scores"]
    a, b = scores[det], scores[floor]

    out = {
        "domain": domain,
        "detector": det,
        "floor": floor,
        "n_held_out": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": round(float(y.mean()), 5),
        "pr_auc_detector": round(_ap(y, a), 4),
        "pr_auc_floor": round(_ap(y, b), 4),
        "observed_margin": round(_ap(y, a) - _ap(y, b), 6),
        "tests": [],
    }
    print(f"  {det} {out['pr_auc_detector']:.4f}  vs  {floor} "
          f"{out['pr_auc_floor']:.4f}   margin {out['observed_margin']:+.4f}")

    t0 = time.time()
    out["tests"].append(bootstrap_windows(y, a, b))
    print(f"  [{time.time()-t0:5.1f}s] iid window bootstrap")

    t0 = time.time()
    if data["groups"] is not None:
        out["tests"].append(bootstrap_clusters(y, a, b, data["groups"]))
        print(f"  [{time.time()-t0:5.1f}s] cluster bootstrap over targets")
    else:
        out["tests"].append(bootstrap_blocks(y, a, b))
        print(f"  [{time.time()-t0:5.1f}s] moving-block bootstrap")

    for t in out["tests"]:
        print(f"\n    {t['method']}")
        print(f"      unit               {t['resampling_unit']}")
        print(f"      resamples used     {t['n_resamples_used']} of "
              f"{t['n_resamples_requested']}"
              + (f"  ({t['n_resamples_skipped_no_positives']} skipped, no positives)"
                 if t["n_resamples_skipped_no_positives"] else ""))
        print(f"      observed margin    {t['observed_difference']:+.6f}")
        print(f"      bootstrap mean     {t['bootstrap_mean_difference']:+.6f}"
              f"   sd {t['bootstrap_std']:.6f}")
        print(f"      95% CI             [{t['ci95_low']:+.6f}, {t['ci95_high']:+.6f}]")
        print(f"      excludes zero      {t['ci95_excludes_zero']}")
        print(f"      P(margin > 0)      {t['p_difference_gt_zero']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both",
                    choices=["kepler", "equity", "both"])
    args = ap.parse_args()

    domains = ["equity", "kepler"] if args.domain == "both" else [args.domain]
    results = {"n_resamples": N_BOOT, "seed": SEED, "confidence": 1 - ALPHA,
               "domains": []}

    # merge into any existing file, so the two domains can be run separately
    if OUT_JSON.exists():
        prior = json.loads(OUT_JSON.read_text())
        results["domains"] = [d for d in prior.get("domains", [])
                              if d["domain"] not in domains]

    for d in domains:
        results["domains"].append(run_domain(d))

    results["domains"].sort(key=lambda d: d["domain"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
