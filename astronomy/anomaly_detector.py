"""Anomaly detection on light curve windows.

Two detectors, matching TECHNICAL_PLAN.md Section 3:
1. Baseline: statistical window features -> IsolationForest (fast, CPU, no
   training-data-only-normal assumption needed).
2. Learned: small autoencoder (astronomy/ts_embedder.py) -> reconstruction
   error as anomaly score. Its bottleneck layer doubles as the "custom
   light-curve embedding" for the RAG layer.

Evaluation here uses a proxy ground truth: for the sample/synthetic data,
"anomalous" windows are the ones containing an injected/real transit dip.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, roc_auc_score

WINDOW_SIZE = 200
STRIDE = 50


def make_windows(flux: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = STRIDE) -> np.ndarray:
    n = len(flux)
    windows = [flux[i:i + window_size] for i in range(0, n - window_size + 1, stride)]
    return np.array(windows)


# A Kepler transit lasts roughly 1-10 hours, so a window should be comparable
# to that rather than an order of magnitude larger.
WINDOW_HOURS = 12.0
STRIDE_HOURS = 3.0


def window_params_for_cadence(times: np.ndarray, window_hours: float = WINDOW_HOURS,
                               stride_hours: float = STRIDE_HOURS) -> tuple[int, int]:
    """Window size in POINTS for this target's actual cadence.

    The fixed 200-point window was chosen when the cache happened to hold
    1-minute data, where it spans 3.3 hours. On 29.4-minute long cadence the
    same 200 points span ~98 hours, so a transit occupies about 5% of the
    window it is meant to dominate and drowns in the statistics of everything
    around it. Specifying the window in TIME makes a feature mean the same
    thing on every target regardless of cadence.

    Shared by the evaluation (astronomy/evaluate.py) and the serving path
    (agents/nodes/astronomy_node.py) on purpose: if they computed windows
    differently, the numbers in the report would describe something other
    than what the chatbot actually runs.
    """
    cadence_hours = float(np.median(np.diff(times))) * 24.0
    if not np.isfinite(cadence_hours) or cadence_hours <= 0:
        return WINDOW_SIZE, STRIDE
    window = max(8, int(round(window_hours / cadence_hours)))
    stride = max(1, int(round(stride_hours / cadence_hours)))
    return window, stride


def extract_features(windows: np.ndarray) -> pd.DataFrame:
    """Hand-crafted statistical features per window -- the classical-ML
    baseline representation (as opposed to the learned autoencoder embedding)."""
    feats = {
        "mean": windows.mean(axis=1),
        "std": windows.std(axis=1),
        "min": windows.min(axis=1),
        "max": windows.max(axis=1),
        "skew": pd.DataFrame(windows.T).skew().values,
        "kurtosis": pd.DataFrame(windows.T).kurtosis().values,
        "depth": 1.0 - windows.min(axis=1),  # how far the deepest dip falls below normalized flux=1.0
        "range": windows.max(axis=1) - windows.min(axis=1),
    }
    return pd.DataFrame(feats)


def fit_isolation_forest(feature_df: pd.DataFrame, contamination: float = 0.1) -> IsolationForest:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(feature_df)
    return model


def score_anomalies(model: IsolationForest, feature_df: pd.DataFrame) -> pd.DataFrame:
    out = feature_df.copy()
    out["anomaly_score"] = -model.decision_function(feature_df)  # higher = more anomalous
    out["is_anomaly"] = model.predict(feature_df) == -1
    return out


def label_windows_by_depth(feature_df: pd.DataFrame, depth_threshold: float = 0.008) -> np.ndarray:
    """Proxy ground truth for evaluation: a window is a 'true' anomaly if
    its flux dips more than `depth_threshold` below the normalized baseline
    (i.e. it contains a transit-like dip)."""
    return (feature_df["depth"] > depth_threshold).astype(int).values


def evaluate(feature_df_with_scores: pd.DataFrame, true_labels: np.ndarray) -> dict:
    pred_labels = feature_df_with_scores["is_anomaly"].astype(int).values
    scores = feature_df_with_scores["anomaly_score"].values
    return {
        "precision": round(precision_score(true_labels, pred_labels, zero_division=0), 3),
        "recall": round(recall_score(true_labels, pred_labels, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(true_labels, scores), 3) if len(set(true_labels)) > 1 else float("nan"),
        "n_windows": len(true_labels),
        "n_true_anomalies": int(true_labels.sum()),
        "n_flagged": int(pred_labels.sum()),
    }


def severity_by_score_tertile(scored: pd.DataFrame) -> pd.Series:
    """Buckets flagged windows into LOW/MEDIUM/HIGH by where their
    anomaly_score falls among only the OTHER flagged windows -- mirrors the
    intent of finance/anomaly_detector.py's severity labeling (distinguish
    "barely over the line" from "extreme") using the mechanism that
    actually fits astronomy: a single continuous IsolationForest score
    rather than a count of independent detectors firing together."""
    flagged_scores = scored.loc[scored["is_anomaly"], "anomaly_score"]
    if flagged_scores.empty:
        return pd.Series(["NONE"] * len(scored), index=scored.index)
    high_cut, med_cut = flagged_scores.quantile(2 / 3), flagged_scores.quantile(1 / 3)

    def _label(row):
        if not row["is_anomaly"]:
            return "NONE"
        if row["anomaly_score"] >= high_cut:
            return "HIGH"
        if row["anomaly_score"] >= med_cut:
            return "MEDIUM"
        return "LOW"

    return scored.apply(_label, axis=1)


def window_reasons(row: pd.Series, recon_error: float | None) -> list[str]:
    reasons = [f"IsolationForest anomaly score {row['anomaly_score']:.2f}"]
    if row.get("depth", 0) > 0.008:
        reasons.append(f"flux dip depth {row['depth']:.4f}")
    if abs(row.get("skew", 0)) > 1.0:
        reasons.append(f"skew {row['skew']:.2f}")
    if recon_error is not None:
        reasons.append(f"autoencoder reconstruction error {recon_error:.5f}")
    return reasons


def possible_cause(row: pd.Series, recon_error: float | None = None, recon_error_median: float | None = None,
                    depth_threshold: float = 0.008, skew_threshold: float = 2.0) -> tuple[str, str]:
    """Rule-based (not LLM-derived) best-guess explanation for why a window
    was flagged, using only the already-computed features -- same
    "computed facts, not model-derived" principle as the rest of this
    project's anomaly explanations. Deliberately hedged: a flagged window
    is a candidate signal, never a confirmed classification (a single dip
    can't rule out an eclipsing binary or instrumental artifact, matching
    the caution required by ASTRONOMY_EXPLAINER_SYSTEM_PROMPT)."""
    depth = float(row.get("depth", 0.0))
    skew = float(row.get("skew", 0.0))
    kurtosis = float(row.get("kurtosis", 0.0))

    if depth > depth_threshold:
        if abs(skew) < skew_threshold:
            return (
                "Possible transit-like dip",
                f"A localized, roughly symmetric drop in flux (depth {depth:.4f}) is consistent with a transiting "
                f"object -- planet or eclipsing binary -- passing in front of the star. A single dip can't confirm "
                f"a planet on its own: that needs repeated detections at the same period with consistent "
                f"depth/duration.",
            )
        return (
            "Possible transit-like dip (asymmetric)",
            f"A flux dip of depth {depth:.4f} was found, but the window's shape is notably asymmetric "
            f"(skew {skew:.2f}) rather than a clean symmetric dip -- still consistent with a transit captured near "
            f"the window edge, but also raises the chance of an eclipsing binary (which can produce asymmetric "
            f"eclipse depths) or a partially-captured event. Depth is the dominant signal here; the asymmetry is a "
            f"caveat, not a reason to discount the dip.",
        )

    if recon_error is not None and recon_error_median is not None and recon_error_median > 0 \
            and recon_error > 1.5 * recon_error_median and depth <= depth_threshold:
        return (
            "Possible stellar variability or noise",
            f"The autoencoder reconstructed this window poorly (error {recon_error:.5f}, "
            f"{recon_error / recon_error_median:.1f}x the dataset median) even though no strong dip was found by "
            f"depth alone -- suggests an irregular shape rather than a clean transit, e.g. intrinsic stellar "
            f"variability (starspots), instrumental noise, or a signal too shallow for the depth heuristic.",
        )

    if abs(skew) >= skew_threshold or kurtosis >= 3.0:
        return (
            "Possible flare or asymmetric irregularity",
            f"A strongly skewed/peaked flux shape (skew {skew:.2f}, kurtosis {kurtosis:.2f}) suggests a brief, "
            f"asymmetric brightening or dimming rather than a periodic transit -- consistent with a stellar flare "
            f"or genuinely irregular variability (the kind seen in targets like Tabby's Star).",
        )

    return (
        "Unclear -- statistical outlier only",
        "This window was flagged by the IsolationForest's combined statistical profile without one dominant "
        "feature (dip depth, skew, or reconstruction error) standing out clearly. Worth a manual look at the raw "
        "window before drawing any conclusion.",
    )


def annotate_windows(scored: pd.DataFrame, recon_errors: np.ndarray | None = None) -> pd.DataFrame:
    """One-call convenience: adds severity_label, reasons, and
    possible_cause/possible_cause_detail columns to an already-scored
    window DataFrame (astronomy.anomaly_detector.score_anomalies's output).
    Shared by agents/nodes/astronomy_node.py and the dashboard so both read
    the same severity/reasons/cause logic instead of drifting apart."""
    out = scored.copy()
    out["severity_label"] = severity_by_score_tertile(out)

    recon_median = float(np.median(recon_errors)) if recon_errors is not None else None
    reasons_col, cause_col, cause_detail_col = [], [], []
    for i, row in out.iterrows():
        recon_error = float(recon_errors[i]) if recon_errors is not None else None
        reasons_col.append(window_reasons(row, recon_error))
        cause, detail = possible_cause(row, recon_error, recon_median)
        cause_col.append(cause)
        cause_detail_col.append(detail)
    out["reasons"] = reasons_col
    out["possible_cause"] = cause_col
    out["possible_cause_detail"] = cause_detail_col
    return out


def run_pipeline(flux: np.ndarray) -> dict:
    windows = make_windows(flux)
    feats = extract_features(windows)
    model = fit_isolation_forest(feats)
    scored = score_anomalies(model, feats)
    true_labels = label_windows_by_depth(feats)
    metrics = evaluate(scored, true_labels)
    return {"windows": windows, "features": scored, "metrics": metrics, "model": model}


if __name__ == "__main__":
    from data_loader import generate_synthetic_light_curve

    # Sparse, well-separated transits relative to window size, so anomalous
    # windows are a small minority (as in a real light curve) rather than
    # the majority class -- otherwise IsolationForest's contamination
    # assumption inverts (it flags the minority "normal" windows instead).
    lc = generate_synthetic_light_curve(n_points=6000, n_transits=4, transit_depth=0.015, seed=1)
    result = run_pipeline(lc["flux"].values)
    print("Metrics:", result["metrics"])
    print("\nTop 5 most anomalous windows:")
    print(result["features"].sort_values("anomaly_score", ascending=False).head(5))
