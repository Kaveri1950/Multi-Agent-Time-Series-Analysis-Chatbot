"""Honest evaluation of the light-curve anomaly detectors.

Replaces the evaluation path in astronomy/anomaly_detector.run_pipeline(),
which had three defects that together made its numbers unreportable:

1. CIRCULAR LABELS. label_windows_by_depth() calls a window anomalous when
   `depth > 0.008` -- and `depth` is one of the eight features the detector
   is given. The metric therefore partly measured whether IsolationForest
   could rediscover a threshold on its own input. Here labels come from
   published transit ephemerides instead (astronomy/koi_catalog.py), so the
   ground truth is astronomical fact rather than a restatement of the
   feature vector. quantify_label_inflation() measures how much difference
   that makes, because "our old numbers were too good" deserves a figure.

2. A CONFOUNDED SECOND DETECTOR. agents/nodes/astronomy_node.py trains the
   autoencoder on whatever IsolationForest called normal, so the two
   detectors were never independent and the observed "IsolationForest beats
   the autoencoder" could not mean anything. Here the autoencoder trains on
   windows with no KNOWN transit, so both detectors see the same honest
   split and can finally be compared.

3. A PHYSICALLY WRONG WINDOW. The 200-point window was chosen when the cache
   held 1-minute data (3.3 hours). On the uniform 29.4-minute long-cadence
   data it spans ~98 hours, while a real transit lasts 2-10. A transit then
   occupies ~5% of the window it is supposed to dominate, which buries the
   signal in the statistics of everything around it. Windows here are
   specified in HOURS and converted per target.

Everything is evaluated per target and then pooled, because pooling alone
hides the variation that matters -- a mean AUC over targets with wildly
different transit depths is a summary of a bimodal distribution.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from astronomy.anomaly_detector import (extract_features, fit_isolation_forest, make_windows,
                                        score_anomalies, window_params_for_cadence)
from astronomy.bulk_download import load_manifest, load_target
from astronomy.koi_catalog import label_windows, usable_kois

# A Kepler transit lasts roughly 1-10 hours. A window should be comparable
# to that, not an order of magnitude larger.
WINDOW_HOURS = 12.0
STRIDE_HOURS = 3.0

MIN_WINDOWS = 50
MIN_POSITIVES = 5


def _window_params(times: np.ndarray, window_hours: float, stride_hours: float):
    """Thin wrapper over the shared helper in anomaly_detector, so the
    evaluation and the serving path window identically by construction."""
    return window_params_for_cadence(times, window_hours, stride_hours)


def evaluate_target(kepid: int, koi: pd.DataFrame | None = None,
                    window_hours: float = WINDOW_HOURS,
                    stride_hours: float = STRIDE_HOURS,
                    with_autoencoder: bool = True) -> dict | None:
    """Both detectors on one star, scored against published transits.

    Returns None when the target cannot support a meaningful score -- too
    few windows, or no known transit inside the observed baseline. Returning
    None rather than a NaN-filled row keeps un-scoreable targets out of the
    pooled statistics instead of silently dragging them down.
    """
    koi = usable_kois() if koi is None else koi
    df = load_target(kepid)
    times, flux = df["time"].values, df["flux"].values

    window, stride = _window_params(times, window_hours, stride_hours)
    windows = make_windows(flux, window_size=window, stride=stride)
    if len(windows) < MIN_WINDOWS:
        return None

    y = label_windows(times, kepid, window_size=window, stride=stride, koi=koi)
    y = y[:len(windows)]
    if y.sum() < MIN_POSITIVES or y.sum() == len(y):
        return None

    feats = extract_features(windows)
    iso = fit_isolation_forest(feats, contamination=float(np.clip(y.mean(), 0.01, 0.5)))
    scored = score_anomalies(iso, feats)
    iso_scores = scored["anomaly_score"].values

    row = {
        "kepid": kepid,
        "n_windows": len(windows),
        "window_points": window,
        "n_transit_windows": int(y.sum()),
        "transit_rate": round(float(y.mean()), 4),
        "iso_auc": round(float(roc_auc_score(y, iso_scores)), 4),
        "iso_ap": round(float(average_precision_score(y, iso_scores)), 4),
        "iso_precision": round(float(precision_score(y, scored["is_anomaly"], zero_division=0)), 4),
        "iso_recall": round(float(recall_score(y, scored["is_anomaly"], zero_division=0)), 4),
    }

    if with_autoencoder:
        row.update(_autoencoder_scores(windows, y))
    return row


def _autoencoder_scores(windows: np.ndarray, y: np.ndarray) -> dict:
    """Autoencoder trained on genuinely transit-free windows.

    This is the fix for the confound: the training set is chosen by the
    published ephemerides, not by IsolationForest's opinion, so the two
    detectors are independent and a comparison between them means something.
    """
    from astronomy.ts_embedder import reconstruction_error, train_autoencoder

    normal = windows[y == 0]
    if len(normal) < 20:
        return {"ae_auc": None, "ae_ap": None}
    try:
        import contextlib
        import io as _io

        sink = _io.StringIO()
        with contextlib.redirect_stdout(sink):      # the trainer prints per-epoch loss
            model = train_autoencoder(normal, epochs=40)
        errors = reconstruction_error(model, windows)
        return {
            "ae_auc": round(float(roc_auc_score(y, errors)), 4),
            "ae_ap": round(float(average_precision_score(y, errors)), 4),
        }
    except Exception as e:
        return {"ae_auc": None, "ae_ap": None, "ae_error": f"{type(e).__name__}: {e}"}


def bls_window_scores(times: np.ndarray, flux: np.ndarray, window: int, stride: int,
                      max_period: float = 100.0, n_durations: int = 6) -> np.ndarray:
    """Box Least Squares, scored per window so it is directly comparable.

    This is the method professional transit searches actually use, and it is
    here to quantify what the project's single-window architecture costs.

    The difference is not the statistic but the STACKING. IsolationForest
    judges each window against the others in isolation, so a 200 ppm dip has
    to beat the noise on its own -- and cannot. BLS folds the light curve at
    thousands of trial periods, so every transit of the same planet lands on
    top of every other. Fifty stacked transits improve signal-to-noise by
    about sqrt(50), which is the difference between undetectable and obvious.

    Returns a per-window score: the fraction of each window that the fitted
    BLS ephemeris predicts is in transit. Scored against the same published
    labels as every other detector here, so the AUCs sit on one scale.
    """
    from astropy.timeseries import BoxLeastSquares

    baseline = float(times[-1] - times[0])
    # Durations 1.2h-7.2h covers real Kepler transits. The minimum period must
    # exceed the longest trial duration -- a transit cannot last longer than
    # the orbit it belongs to, and astropy rejects the grid outright if asked.
    durations = np.linspace(0.05, 0.3, n_durations)          # days: 1.2h - 7.2h

    # An EXPLICIT log-spaced period grid, not autopower(). autopower builds a
    # grid fine enough to resolve every possible transit phase across a
    # 4-year baseline, which on 30k points did not finish in nine minutes for
    # a single target. A 3000-point log grid runs in ~1.2s and still locates
    # the period to well within a window width, which is all that is needed
    # to score windows.
    periods = np.exp(np.linspace(np.log(1.0),
                                 np.log(min(max_period, baseline / 3.0)),
                                 3000))
    periodogram = BoxLeastSquares(times, flux).power(periods, durations, objective="snr")

    best = int(np.argmax(periodogram.power))
    period = float(periodogram.period[best])
    t0 = float(periodogram.transit_time[best])
    duration = float(periodogram.duration[best])

    # signed distance to the nearest predicted transit centre
    dt = (times - t0 + period / 2.0) % period - period / 2.0
    in_transit = (np.abs(dt) <= duration / 2.0).astype(float)

    n = len(times)
    return np.array([
        in_transit[i:i + window].mean()
        for i in range(0, n - window + 1, stride)
    ])


def compare_with_bls(limit: int = 15, window_hours: float = WINDOW_HOURS,
                     stride_hours: float = STRIDE_HOURS, verbose: bool = True) -> pd.DataFrame:
    """IsolationForest vs BLS on identical targets, windows, and labels."""
    manifest = load_manifest()
    koi = usable_kois()
    rows = []

    for i, kepid in enumerate(manifest["kepid"].tolist()[:limit], 1):
        kepid = int(kepid)
        try:
            df = load_target(kepid)
            times, flux = df["time"].values, df["flux"].values
            window, stride = _window_params(times, window_hours, stride_hours)
            windows = make_windows(flux, window_size=window, stride=stride)
            y = label_windows(times, kepid, window_size=window, stride=stride, koi=koi)[:len(windows)]
            if len(windows) < MIN_WINDOWS or y.sum() < MIN_POSITIVES or y.sum() == len(y):
                continue

            feats = extract_features(windows)
            iso = fit_isolation_forest(feats, contamination=float(np.clip(y.mean(), 0.01, 0.5)))
            iso_auc = roc_auc_score(y, score_anomalies(iso, feats)["anomaly_score"].values)

            bls = bls_window_scores(times, flux, window, stride)[:len(y)]
            bls_auc = roc_auc_score(y, bls) if len(set(bls)) > 1 else np.nan

            depth = koi[koi["kepid"] == kepid]["koi_depth"].max()
            rows.append({"kepid": kepid, "depth_ppm": float(depth) if pd.notna(depth) else np.nan,
                         "transit_rate": round(float(y.mean()), 4),
                         "iso_auc": round(float(iso_auc), 4),
                         "bls_auc": round(float(bls_auc), 4) if np.isfinite(bls_auc) else None})
            if verbose:
                print(f"  [{i}/{limit}] KIC {kepid}: iso {iso_auc:.3f}  bls {bls_auc:.3f}")
        except Exception as e:
            if verbose:
                print(f"  [{i}/{limit}] KIC {kepid} failed: {type(e).__name__}: {e}")

    return pd.DataFrame(rows)


def evaluate_all(limit: int | None = None, with_autoencoder: bool = True,
                 window_hours: float = WINDOW_HOURS, verbose: bool = True) -> pd.DataFrame:
    """Every cached bulk target, scored against real transits."""
    manifest = load_manifest()
    if manifest.empty:
        raise RuntimeError("no bulk cache -- run `python -m astronomy.bulk_download` first")

    koi = usable_kois()
    kepids = manifest["kepid"].tolist()[:limit]

    rows, skipped = [], 0
    for i, kepid in enumerate(kepids, 1):
        try:
            r = evaluate_target(int(kepid), koi, window_hours=window_hours,
                                with_autoencoder=with_autoencoder)
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(kepids)}] KIC {kepid} failed: {type(e).__name__}: {e}")
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        rows.append(r)
        if verbose and len(rows) % 10 == 0:
            print(f"  [{i}/{len(kepids)}] scored {len(rows)}, skipped {skipped}")

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\nscored {len(out)} targets, skipped {skipped}")
    return out


def quantify_label_inflation(limit: int = 25, window_hours: float = WINDOW_HOURS) -> pd.DataFrame:
    """How much did the circular labels flatter the detector?

    Scores the same targets twice -- once against the old depth-threshold
    proxy, once against published transits -- so the report can state the
    size of the effect instead of merely noting that it existed.
    """
    from astronomy.anomaly_detector import label_windows_by_depth

    manifest = load_manifest()
    koi = usable_kois()
    rows = []

    for kepid in manifest["kepid"].tolist()[:limit]:
        try:
            df = load_target(int(kepid))
        except Exception:
            continue
        times, flux = df["time"].values, df["flux"].values
        window, stride = _window_params(times, window_hours, STRIDE_HOURS)
        windows = make_windows(flux, window_size=window, stride=stride)
        if len(windows) < MIN_WINDOWS:
            continue

        feats = extract_features(windows)
        y_real = label_windows(times, int(kepid), window_size=window, stride=stride, koi=koi)[:len(windows)]
        y_proxy = label_windows_by_depth(feats)

        if len(set(y_real)) < 2 or len(set(y_proxy)) < 2:
            continue

        iso = fit_isolation_forest(feats, contamination=0.1)
        s = score_anomalies(iso, feats)["anomaly_score"].values
        rows.append({
            "kepid": int(kepid),
            "auc_proxy_labels": round(float(roc_auc_score(y_proxy, s)), 4),
            "auc_real_labels": round(float(roc_auc_score(y_real, s)), 4),
            "proxy_positive_rate": round(float(y_proxy.mean()), 4),
            "real_positive_rate": round(float(y_real.mean()), 4),
        })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> dict:
    """Pooled headline numbers, with spread -- a mean AUC alone over targets
    with very different transit depths describes nothing anyone can act on."""
    if results.empty:
        return {}
    out = {
        "n_targets": len(results),
        "median_windows": int(results["n_windows"].median()),
        "median_transit_rate": round(float(results["transit_rate"].median()), 4),
        "iso_auc_median": round(float(results["iso_auc"].median()), 4),
        "iso_auc_iqr": [round(float(results["iso_auc"].quantile(0.25)), 4),
                        round(float(results["iso_auc"].quantile(0.75)), 4)],
        "iso_ap_median": round(float(results["iso_ap"].median()), 4),
        "iso_beats_random_pct": round(float((results["iso_auc"] > 0.5).mean()) * 100, 1),
    }
    if "ae_auc" in results.columns and results["ae_auc"].notna().any():
        ae = results["ae_auc"].dropna()
        out["ae_auc_median"] = round(float(ae.median()), 4)
        out["ae_beats_random_pct"] = round(float((ae > 0.5).mean()) * 100, 1)
        both = results.dropna(subset=["ae_auc"])
        out["iso_beats_ae_pct"] = round(float((both["iso_auc"] > both["ae_auc"]).mean()) * 100, 1)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate light-curve detectors against real transits.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--window-hours", type=float, default=WINDOW_HOURS)
    ap.add_argument("--no-autoencoder", action="store_true")
    ap.add_argument("--inflation", action="store_true",
                    help="compare proxy labels vs real labels")
    args = ap.parse_args()

    if args.inflation:
        print("Proxy (circular) labels vs published transits\n" + "=" * 60)
        infl = quantify_label_inflation(limit=args.limit or 25, window_hours=args.window_hours)
        if infl.empty:
            print("  no comparable targets")
        else:
            print(infl.to_string(index=False))
            print(f"\n  median AUC with proxy labels : {infl['auc_proxy_labels'].median():.4f}")
            print(f"  median AUC with real  labels : {infl['auc_real_labels'].median():.4f}")
            print(f"  inflation                    : "
                  f"{infl['auc_proxy_labels'].median() - infl['auc_real_labels'].median():+.4f}")
        raise SystemExit(0)

    print(f"Evaluating against published transit ephemerides "
          f"(window {args.window_hours}h)\n" + "=" * 60)
    res = evaluate_all(limit=args.limit, with_autoencoder=not args.no_autoencoder,
                       window_hours=args.window_hours)
    if res.empty:
        print("nothing scoreable")
        raise SystemExit(1)

    print("\n=== summary ===")
    for k, v in summarize(res).items():
        print(f"  {k:<26}{v}")
    print("\n=== best and worst targets by IsolationForest AUC ===")
    cols = [c for c in ["kepid", "n_windows", "transit_rate", "iso_auc", "ae_auc"] if c in res.columns]
    print(res.nlargest(5, "iso_auc")[cols].to_string(index=False))
    print("  ...")
    print(res.nsmallest(5, "iso_auc")[cols].to_string(index=False))
