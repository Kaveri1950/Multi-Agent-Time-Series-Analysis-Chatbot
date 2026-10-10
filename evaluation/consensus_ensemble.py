"""Consensus over ALL detectors, rather than a greedy-selected subset.

The committed grid already contains an ensemble, but it is not this one:
`anomaly.splits.select_ensemble` picks at most four members greedily on the
training half and averages their rank scores. This module instead uses every
unsupervised detector in the roster and combines them three ways:

  mean_rank      average of every detector's within-split rank, the natural
                 "average of their score across all models"
  vote_fraction  each detector flags its top-q windows (q = the held-out
                 positive rate); the score is the fraction of detectors that
                 flagged the window
  hard_majority  the binary decision "more than half the detectors flagged
                 it", reported as precision / recall / F1 rather than PR-AUC,
                 because it is a decision and not a ranking

The supervised ceiling is excluded: it is fitted on labels, so including it
in an "unsupervised consensus" would leak. Both floors (raw input, untrained
autoencoder) are kept, since they are legitimate unsupervised scorers and
excluding them would be a post-hoc choice.

Run:  python -m evaluation.consensus_ensemble --domain equity|kepler|both
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

RESULTS = Path(__file__).parent / "results"
CACHE = RESULTS / "consensus_scores_{domain}.npz"
KEPLER_LIMIT = 40          # matches the committed anomaly_grid.json run
SEED = 0


def held_out_scores(domain: str, verbose: bool = True) -> dict:
    """Per-window held-out rank scores for every unsupervised detector."""
    path = Path(str(CACHE).format(domain=domain))
    if path.exists():
        d = np.load(path, allow_pickle=True)
        names = [str(n) for n in d["names"]]
        if verbose:
            print(f"  loaded cached scores <- {path.name} ({len(names)} detectors)")
        return {"y": d["y"], "names": names,
                "S": d["S"], "floor": float(d["floor"]), "best": float(d["best"]),
                "best_name": str(d["best_name"])}

    warnings.filterwarnings("ignore")
    from anomaly.detectors import default_detectors
    from anomaly.grid import load_equity, load_kepler
    from anomaly.splits import make_split, rank01, score_held_out

    if verbose:
        print(f"  loading {domain} ...", flush=True)
    ds = load_equity() if domain == "equity" else load_kepler(limit=KEPLER_LIMIT)
    train_mask, test_mask = make_split(ds, seed=SEED)
    y = ds.y[test_mask]
    if verbose:
        print(f"  held out {test_mask.sum():,} windows, {int(y.sum())} positive "
              f"({y.mean():.4f})", flush=True)

    dets = [d for d in default_detectors(include_finance_only=ds.finance_only,
                                         stride=int(ds.meta.get("stride", 1)))
            if not d.is_ceiling]

    names, cols, aps = [], [], []
    for det in dets:
        t0 = time.time()
        try:
            raw = score_held_out(det, ds, train_mask, test_mask)
        except Exception as exc:               # noqa: BLE001
            print(f"    {det.name:<28} FAILED {type(exc).__name__}: {exc}")
            continue
        names.append(det.name)
        cols.append(rank01(raw))               # rank only for COMBINING
        aps.append(average_precision_score(y, raw))   # AP from the raw score
        if verbose:
            print(f"    {det.name:<28} PR-AUC {aps[-1]:.4f}"
                  f"   {time.time() - t0:8.1f}s", flush=True)

    S = np.column_stack(cols)
    aps = np.asarray(aps)
    fi = names.index("raw_input")
    bi = int(np.argmax(aps))
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y=y, names=np.array(names), S=S,
                        floor=aps[fi], best=aps[bi], best_name=names[bi])
    if verbose:
        print(f"  cached -> {path.name}")
    return {"y": y, "names": names, "S": S, "floor": float(aps[fi]),
            "best": float(aps[bi]), "best_name": names[bi]}


def consensus(domain: str) -> dict:
    d = held_out_scores(domain)
    y, S, names = d["y"], d["S"], d["names"]
    n_det = S.shape[1]
    q = float(y.mean())

    mean_rank = S.mean(axis=1)
    # Each detector flags its own top-q fraction. Using the held-out positive
    # rate as the budget is the fairest per-detector threshold available; it
    # is the same budget for every detector, so no detector is advantaged.
    thresholds = np.quantile(S, 1.0 - q, axis=0)
    flags = S >= thresholds
    vote_fraction = flags.mean(axis=1)
    hard = vote_fraction > 0.5

    ap_mean = float(average_precision_score(y, mean_rank))
    ap_vote = float(average_precision_score(y, vote_fraction))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, hard, average="binary", zero_division=0)

    # How much do the detectors actually agree? A vote can only help if they
    # make partly independent errors; if they barely correlate, the vote is
    # averaging noise, and if they agree completely it adds nothing.
    C = np.corrcoef(S, rowvar=False)
    iu = np.triu_indices(n_det, k=1)
    mean_rho = float(np.nanmean(C[iu]))
    votes = flags.sum(axis=1)
    jac = []
    for i in range(n_det):
        for j in range(i + 1, n_det):
            inter = np.logical_and(flags[:, i], flags[:, j]).sum()
            union = np.logical_or(flags[:, i], flags[:, j]).sum()
            if union:
                jac.append(inter / union)

    out = {
        "domain": domain, "n_detectors": n_det, "detectors": names,
        "agreement": {
            "mean_pairwise_rank_correlation": mean_rho,
            "mean_pairwise_jaccard_of_flags": float(np.mean(jac)) if jac else None,
            "max_votes_on_any_window": int(votes.max()),
            "votes_needed_for_majority": n_det // 2 + 1,
            "mean_votes_on_positive_windows": float(votes[y.astype(bool)].mean()),
            "mean_votes_on_negative_windows": float(votes[~y.astype(bool)].mean()),
        },
        "n_held_out": int(len(y)), "n_positive": int(y.sum()), "prevalence": q,
        "floor_raw_input": d["floor"],
        "best_single": {"detector": d["best_name"], "pr_auc": d["best"]},
        "mean_rank_all": {"pr_auc": ap_mean, "margin_vs_floor": ap_mean - d["floor"]},
        "vote_fraction_all": {"pr_auc": ap_vote, "margin_vs_floor": ap_vote - d["floor"]},
        "hard_majority": {
            "rule": "flagged by more than half of %d detectors" % n_det,
            "n_flagged": int(hard.sum()),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "precision_lift_vs_prevalence": float(prec / q) if q else None,
        },
    }

    print(f"\n{'=' * 66}\n{domain.upper()}  consensus over all {n_det} unsupervised detectors\n{'=' * 66}")
    print(f"  held out {len(y):,} windows, {int(y.sum())} positive (prevalence {q:.4f})")
    print(f"  raw-input floor              PR-AUC {d['floor']:.4f}")
    print(f"  best single ({d['best_name']:<26}) PR-AUC {d['best']:.4f}")
    print(f"  mean rank, all {n_det:>2} detectors    PR-AUC {ap_mean:.4f}"
          f"   ({ap_mean - d['floor']:+.4f} vs floor)")
    print(f"  vote fraction, all {n_det:>2} detectors PR-AUC {ap_vote:.4f}"
          f"   ({ap_vote - d['floor']:+.4f} vs floor)")
    print(f"  hard majority (>50% of {n_det}): flagged {int(hard.sum()):,} windows")
    print(f"     max votes on any window {int(votes.max())} "
          f"(majority needs {n_det // 2 + 1})")
    print(f"     mean votes  positives {votes[y.astype(bool)].mean():.2f}"
          f"   negatives {votes[~y.astype(bool)].mean():.2f}")
    print(f"     mean pairwise rank corr {mean_rho:.3f}"
          f"   mean flag Jaccard {np.mean(jac):.3f}")
    print(f"     precision {prec:.4f}  recall {rec:.4f}  f1 {f1:.4f}"
          f"   (precision lift {prec / q:.2f}x vs prevalence)" if q else "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="both", choices=["equity", "kepler", "both"])
    args = ap.parse_args()
    domains = ["equity", "kepler"] if args.domain == "both" else [args.domain]

    path = RESULTS / "consensus_results.json"
    merged = json.loads(path.read_text()) if path.exists() else {"domains": []}
    merged["domains"] = [d for d in merged["domains"] if d["domain"] not in domains]
    for dom in domains:
        merged["domains"].append(consensus(dom))
    merged["domains"].sort(key=lambda d: d["domain"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
