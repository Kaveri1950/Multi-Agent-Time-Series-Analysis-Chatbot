"""Audit a dataset against Wu & Keogh's four benchmark flaws.

From "Current Time Series Anomaly Detection Benchmarks are Flawed and are
Creating the Illusion of Progress" (IEEE TKDE), which examined the standard
Yahoo / Numenta / NASA / OMNI benchmarks and found most of their problems
suffer from at least one of:

  1. TRIVIALITY            solvable by a single line of arithmetic, so
                           reporting success on it means nothing
  2. UNREALISTIC DENSITY   so many labelled anomalies that the task is
                           really classification wearing a disguise
  3. MISLABELLED TRUTH     labels that visibly disagree with the data
  4. RUN-TO-FAILURE BIAS   anomalies concentrated at the end of the series,
                           so "flag the last point" scores well

Their conclusion is blunt: on such benchmarks "there is simply no level of
performance that would suggest the utility of a proposed algorithm."

Almost nobody applies this to their own data. Running it before reporting
any detector result is the difference between numbers that mean something
and numbers that merely exist.

A NOTE ON WHAT IS AND IS NOT CIRCULAR. Flaw 3 inspects labels against the
series they describe, which superficially resembles the circularity this
project is careful to avoid elsewhere. It is not the same thing: auditing
label QUALITY is a legitimate use of the data, whereas deriving the labels
themselves from it is not. The distinction is that nothing here feeds back
into how any detector is scored.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: Wu & Keogh observe that in real settings "the prior probability of an
#: anomaly is expected to be only slightly greater than zero". Benchmarks
#: where half the test data is a contiguous anomalous region blur the line
#: between anomaly detection and classification.
DENSITY_WARN = 0.10
DENSITY_FAIL = 0.25

#: Their triviality definition is a single line of standard library code
#: using primitives like mean, max, std, diff. The one-liner detectors in
#: `detectors.py` are exactly that, so their score IS the triviality test.
TRIVIAL_PR_AUC = 0.80
TRIVIAL_LIFT = 8.0


def check_density(y: np.ndarray) -> dict:
    """Flaw 2. What fraction of the data is labelled anomalous?"""
    y = np.asarray(y).astype(int).ravel()
    prevalence = float(y.mean()) if len(y) else float("nan")
    # The longest unbroken positive run: the specific pathology Wu & Keogh
    # call out, where a single contiguous region covers half the test data.
    longest, current = 0, 0
    for v in y:
        current = current + 1 if v else 0
        longest = max(longest, current)
    verdict = ("fail" if prevalence > DENSITY_FAIL
               else "warn" if prevalence > DENSITY_WARN else "pass")
    return {
        "flaw": "unrealistic_anomaly_density",
        "prevalence": round(prevalence, 4),
        "n_positive": int(y.sum()),
        "n_total": len(y),
        "longest_positive_run": longest,
        "longest_run_fraction": round(longest / len(y), 4) if len(y) else float("nan"),
        "verdict": verdict,
    }


def check_run_to_failure(y: np.ndarray) -> dict:
    """Flaw 4. Are the anomalies bunched at the end of the series?

    Tests the positions of positive labels against a uniform distribution.
    If they skew late, "flag the last point" is a strong strategy and any
    detector with a mild recency bias gets credit it has not earned.
    """
    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    pos = np.flatnonzero(y)
    if n == 0 or len(pos) < 3:
        return {"flaw": "run_to_failure_bias", "verdict": "insufficient_data",
                "n_positive": int(len(pos))}
    rel = pos / max(n - 1, 1)
    ks = stats.kstest(rel, "uniform")
    mean_pos = float(rel.mean())
    # Skew matters only when it is LATE; anomalies bunched early do not
    # reward the trivial "flag the end" strategy.
    late_skew = mean_pos > 0.5 and ks.pvalue < 0.01
    return {
        "flaw": "run_to_failure_bias",
        "mean_relative_position": round(mean_pos, 4),
        "fraction_in_last_decile": round(float((rel > 0.9).mean()), 4),
        "ks_pvalue": round(float(ks.pvalue), 6),
        "verdict": "fail" if late_skew else "pass",
    }


def check_triviality(detector_rows: list[dict]) -> dict:
    """Flaw 1. Does a one-line arithmetic rule already solve this?

    Consumes the grid's own results rather than recomputing, so the
    triviality verdict and the reported table can never disagree.
    """
    liners = [r for r in detector_rows if r.get("group") == "one_liner"]
    floors = [r for r in detector_rows if r.get("is_floor")]
    candidates = liners + floors
    finite = [r for r in candidates if np.isfinite(r.get("pr_auc", np.nan))]
    if not finite:
        return {"flaw": "triviality", "verdict": "insufficient_data"}
    best = max(finite, key=lambda r: r["pr_auc"])
    trivial = best["pr_auc"] >= TRIVIAL_PR_AUC or best.get("lift", 0) >= TRIVIAL_LIFT
    return {
        "flaw": "triviality",
        "best_trivial_detector": best["detector"],
        "best_trivial_pr_auc": best["pr_auc"],
        "best_trivial_lift": best.get("lift"),
        "verdict": "fail" if trivial else "pass",
    }


def check_label_quality(y: np.ndarray, series_effect: np.ndarray | None = None) -> dict:
    """Flaw 3. Do the labels visibly disagree with the data?

    `series_effect` is a per-item magnitude (an absolute return, a transit
    depth). Two failure modes are measured:

      quiet positives -- labelled anomalous but indistinguishable from
                         background, the Yahoo A1-Real47 problem where a
                         labelled region looks like dozens of unlabelled ones
      loud negatives  -- unlabelled but more extreme than most positives,
                         suggesting the label set is incomplete

    Reported as numbers rather than a pass/fail, because some of both is
    expected in any honest label set and the right response is disclosure,
    not a verdict.
    """
    y = np.asarray(y).astype(int).ravel()
    if series_effect is None or y.sum() == 0 or (1 - y).sum() == 0:
        return {"flaw": "mislabeled_ground_truth", "verdict": "not_assessable"}
    eff = np.abs(np.asarray(series_effect, dtype=float).ravel())
    pos, neg = eff[y == 1], eff[y == 0]
    neg_median = float(np.median(neg))
    pos_median = float(np.median(pos))
    # A positive is "quiet" if it is no more extreme than a typical negative.
    quiet = float((pos <= neg_median).mean())
    # A negative is "loud" if it exceeds the median positive.
    loud = float((neg > pos_median).mean())
    return {
        "flaw": "mislabeled_ground_truth",
        "median_effect_positive": round(pos_median, 6),
        "median_effect_negative": round(neg_median, 6),
        "effect_ratio": round(pos_median / neg_median, 3) if neg_median else float("inf"),
        "fraction_quiet_positives": round(quiet, 4),
        "fraction_loud_negatives": round(loud, 4),
        "verdict": "review" if quiet > 0.5 else "pass",
    }


def audit_dataset(y: np.ndarray, series_effect: np.ndarray | None = None,
                  detector_rows: list[dict] | None = None) -> dict:
    """All four checks. `detector_rows` may be supplied later for triviality."""
    checks = [
        check_density(y),
        check_run_to_failure(y),
        check_label_quality(y, series_effect),
    ]
    if detector_rows:
        checks.insert(0, check_triviality(detector_rows))
    failed = [c["flaw"] for c in checks if c.get("verdict") == "fail"]
    return {
        "checks": checks,
        "n_failed": len(failed),
        "failed_flaws": failed,
        "believable": not failed,
    }


def format_audit(audit: dict, title: str = "") -> str:
    """Human-readable audit block for the console and the report."""
    mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL",
            "review": "REVIEW", "insufficient_data": "N/A", "not_assessable": "N/A"}
    lines = [f"  {title}" if title else "", "  " + "-" * 62]
    for c in audit["checks"]:
        v = c.get("verdict", "?")
        detail = ", ".join(f"{k}={v2}" for k, v2 in c.items()
                           if k not in ("flaw", "verdict"))
        lines.append(f"  [{mark.get(v, v):6}] {c['flaw']}")
        if detail:
            lines.append(f"           {detail}")
    lines.append("  " + "-" * 62)
    lines.append(f"  believable: {audit['believable']}"
                 + (f"  (failed: {', '.join(audit['failed_flaws'])})"
                    if audit["failed_flaws"] else ""))
    return "\n".join(lines)
