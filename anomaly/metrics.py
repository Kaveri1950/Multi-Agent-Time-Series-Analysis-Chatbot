"""Scoring, with the two traps in this literature avoided on purpose.

TRAP 1 -- POINT ADJUSTMENT. The dominant protocol in published time-series
anomaly detection marks an entire ground-truth segment correct if the
detector flags any single moment inside it. Kim et al. (AAAI 2022, "Towards
a Rigorous Evaluation of Time-Series Anomaly Detection") showed this lets a
RANDOM anomaly score reach state-of-the-art F1, because long segments are
almost certain to contain a spurious hit somewhere. Nothing here applies
point adjustment, and no point-adjusted F1 is reported.

TRAP 2 -- ROC-AUC ALONE UNDER IMBALANCE. ROC-AUC has a floor of 0.5 no
matter how rare the positive class is, so it cannot tell you whether 0.578
is a real edge or noise. Average precision (PR-AUC) is floored at the
PREVALENCE instead, which makes the comparison interpretable: at 3%
prevalence, a PR-AUC of 0.15 is five times chance. Both are reported, with
PR-AUC as the decision metric and `lift` making the comparison explicit.

ON BEST-OF-N. The 2026 adversarial stress-test of these metrics found that
ROC-based measures become gameable when a researcher runs many random
configurations and reports the best, but that at N=1 no metric is gameable.
Every number this module produces comes from a single pre-declared
configuration; `n_configs_tried` records that, because a metric without its
reporting budget is uninterpretable.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray,
                    n_configs_tried: int = 1) -> dict:
    """Ranking quality of `scores` against binary `y_true`.

    Threshold-free on purpose. Picking a threshold introduces a choice that
    can be tuned per detector, and tuning it per detector is precisely how
    comparisons stop being fair.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if len(y_true) != len(scores):
        raise ValueError(f"length mismatch: {len(y_true)} labels, {len(scores)} scores")

    n_pos = int(y_true.sum())
    prevalence = n_pos / len(y_true) if len(y_true) else float("nan")

    # Degenerate cases are reported as nan rather than raising: a target with
    # no transits is a legitimate row in the results table, not an error.
    if n_pos == 0 or n_pos == len(y_true):
        return {
            "roc_auc": float("nan"), "pr_auc": float("nan"), "lift": float("nan"),
            "prevalence": round(prevalence, 5), "n": len(y_true), "n_positive": n_pos,
            "n_configs_tried": n_configs_tried,
        }

    # Ties matter here. A detector returning a constant scores 0.5 ROC-AUC
    # and prevalence PR-AUC, which is correct -- it has no skill -- but it is
    # worth being able to see that it happened.
    pr_auc = float(average_precision_score(y_true, scores))
    return {
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(pr_auc, 4),
        # How many times better than chance, in the metric whose floor is meaningful.
        "lift": round(pr_auc / prevalence, 2) if prevalence > 0 else float("nan"),
        "prevalence": round(prevalence, 5),
        "n": len(y_true),
        "n_positive": n_pos,
        "n_constant_score": int(np.ptp(scores) == 0),
        "n_configs_tried": n_configs_tried,
    }


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Of the k highest-scoring items, what fraction are true positives?

    The metric that matches how these detectors actually get used: an
    astronomer follows up the top 50 candidates, an analyst reviews the 20
    strangest days. Nobody consumes a full ranking.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    k = min(k, len(scores))
    if k <= 0:
        return float("nan")
    top = np.argsort(-scores, kind="stable")[:k]
    return round(float(y_true[top].mean()), 4)


def summarise_against_floor(rows: list[dict], metric: str = "pr_auc") -> dict:
    """Did anything actually beat the no-skill floors?

    The headline the report needs. If the best real detector does not clear
    the best floor, that is the finding, and it should be stated rather than
    buried under a table of similar-looking numbers.
    """
    floors = [r for r in rows if r.get("is_floor")]
    real = [r for r in rows if not r.get("is_floor") and not r.get("is_ceiling")]
    ceilings = [r for r in rows if r.get("is_ceiling")]

    def best(group):
        vals = [r for r in group if np.isfinite(r.get(metric, float("nan")))]
        return max(vals, key=lambda r: r[metric]) if vals else None

    best_floor, best_real, best_ceiling = best(floors), best(real), best(ceilings)
    out = {
        "metric": metric,
        "best_floor": (best_floor or {}).get("detector"),
        "best_floor_value": (best_floor or {}).get(metric),
        "best_real": (best_real or {}).get("detector"),
        "best_real_value": (best_real or {}).get(metric),
        "ceiling": (best_ceiling or {}).get("detector"),
        "ceiling_value": (best_ceiling or {}).get(metric),
    }
    if best_floor and best_real:
        out["real_beats_floor"] = bool(best_real[metric] > best_floor[metric])
        out["margin_over_floor"] = round(best_real[metric] - best_floor[metric], 4)
    if best_real and best_ceiling and best_ceiling[metric] > 0:
        # What fraction of the label-aware ceiling the best unsupervised
        # detector recovered -- context for whether a modest number is close
        # to the limit of the data or nowhere near it.
        out["fraction_of_ceiling"] = round(best_real[metric] / best_ceiling[metric], 3)
    return out
