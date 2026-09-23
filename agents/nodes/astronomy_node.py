"""Astronomy Agent: light-curve anomaly detection, grounded in NASA's catalogue.

Three changes from the original, each fixing something that made the answers
weaker than they looked:

1. DATA SOURCE. It used to call data_loader.fetch_light_curve(), which does
   `search[0].download()` -- whatever MAST happens to list first. That
   returned Kepler Quarter 00 (a ~10-day commissioning run, 469 points) for
   some stars and Quarter 02 at 1-minute cadence for others, so the same
   question got a different kind of answer depending on the star, and the
   two most demo-worthy targets (Tabby's Star, Kepler-16) came back with six
   windows. The bulk cache is preferred now: uniform 29.4-minute long
   cadence, every quarter stitched, ~4 years per target. The live fetch
   remains as a fallback for anything not cached.

2. WINDOW SIZE. Fixed at 200 points, which is 3.3 hours at 1-minute cadence
   and ~98 hours at 30-minute. Now specified in time, via the helper shared
   with the evaluation so both window identically.

3. THE AUTOENCODER'S TRAINING SET. It used to train on whatever
   IsolationForest called normal, so the two detectors were never
   independent and comparing them meant nothing. Where NASA publishes
   transit ephemerides for the star, the autoencoder now trains on windows
   containing no KNOWN transit. The pseudo-label path survives for targets
   the catalogue does not cover.

And one addition: the result carries what the archive actually KNOWS about
the star, plus how many flagged windows land on published transits. "I
flagged 12 windows, and 9 of them fall on the transits of a confirmed planet
at 3.23 days" is a far better answer than a flag count, and it lets the
explainer distinguish a real recovery from a pile of false alarms.
"""

import numpy as np

from agents.state import AgentState
from astronomy.anomaly_detector import (annotate_windows, extract_features, fit_isolation_forest,
                                        make_windows, score_anomalies, window_params_for_cadence)
from astronomy.ts_embedder import embed_windows, reconstruction_error, train_autoencoder

_MIN_NORMAL_WINDOWS = 20


def _resolve_kepid(target: str):
    """'KIC 10874614' -> 10874614. None when the target is not a KIC id."""
    import re

    m = re.search(r"\b(?:KIC|KPLR)\s*0*(\d{5,9})\b", str(target), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _load_light_curve(target: str, trace: list):
    """Bulk cache first, live MAST second, synthetic last.

    Order matters: the bulk cache is the only source with a known, uniform
    cadence and full mission coverage, so preferring it makes answers
    comparable between targets. The synthetic fallback keeps a demo alive
    when the network is down, and is labelled loudly because two synthetic
    curves silently entering the old evaluation scored AUC 0.21 and dragged
    the reported mean down without anyone noticing.
    """
    kepid = _resolve_kepid(target)

    if kepid is not None:
        try:
            from astronomy.bulk_download import is_cached, load_target

            if is_cached(kepid):
                df = load_target(kepid)
                trace.append(f"[astronomy_agent] {target}: bulk cache "
                             f"({len(df)} points, stitched long cadence)")
                return df, "bulk cache (Kepler long cadence, all quarters stitched)", kepid
        except Exception as e:
            trace.append(f"[astronomy_agent] bulk cache unavailable ({type(e).__name__})")

    from astronomy.data_loader import fetch_light_curve, generate_synthetic_light_curve

    try:
        df = fetch_light_curve(target)
        trace.append(f"[astronomy_agent] {target}: live MAST fetch ({len(df)} points) -- "
                     f"single arbitrary quarter; cadence may differ from cached targets")
        return df, "real (MAST/Kepler via lightkurve, single quarter)", kepid
    except Exception as exc:
        df = generate_synthetic_light_curve(seed=abs(hash(target)) % (2 ** 32))
        trace.append(f"[astronomy_agent] {target}: SYNTHETIC fallback -- real fetch failed ({exc})")
        return df, f"synthetic fallback (real fetch failed: {exc})", kepid


def _known_transit_labels(times: np.ndarray, kepid, window: int, stride: int, trace: list):
    """Per-window labels from published ephemerides, or None if unavailable."""
    if kepid is None:
        return None, None
    try:
        from astronomy.koi_catalog import label_windows, target_summary, usable_kois

        koi = usable_kois()
        info = target_summary(kepid, koi)
        if not info.get("known"):
            trace.append(f"[astronomy_agent] KIC {kepid} is not in the KOI catalogue -- "
                         f"no published transits to compare against")
            return None, info
        labels = label_windows(times, kepid, window_size=window, stride=stride, koi=koi)
        trace.append(f"[astronomy_agent] KOI catalogue: {info['n_kois']} object(s) of interest, "
                     f"{info['n_confirmed']} confirmed, {int(labels.sum())} window(s) contain a "
                     f"published transit")
        return labels, info
    except Exception as e:
        trace.append(f"[astronomy_agent] catalogue lookup failed ({type(e).__name__}: {e})")
        return None, None


def astronomy_node(state: AgentState) -> AgentState:
    target = state["target"]
    trace = state.get("trace", [])

    lc, source, kepid = _load_light_curve(target, trace)
    flux = lc["flux"].values
    times = lc["time"].values

    window, stride = window_params_for_cadence(times)
    windows = make_windows(flux, window_size=window, stride=stride)
    cadence_min = float(np.median(np.diff(times))) * 24 * 60
    trace.append(f"[astronomy_agent] cadence {cadence_min:.1f} min -> window {window} points "
                 f"(~12h), stride {stride}; {len(windows)} windows")

    feats = extract_features(windows)
    known_labels, koi_info = _known_transit_labels(times, kepid, window, stride, trace)
    if known_labels is not None:
        known_labels = known_labels[:len(windows)]

    # Contamination from the real transit rate where it is known, rather than
    # a fixed 0.1 -- true rates across targets run from under 1% to over 15%,
    # and a fixed value makes the detector flag ~10% of windows regardless of
    # what is actually there.
    contamination = 0.1
    if known_labels is not None and 0 < known_labels.mean() < 0.5:
        contamination = float(np.clip(known_labels.mean(), 0.01, 0.4))
        trace.append(f"[astronomy_agent] contamination set from the published transit rate "
                     f"({contamination:.3f}) instead of the fixed 0.1")

    iso_model = fit_isolation_forest(feats, contamination=contamination)
    scored = score_anomalies(iso_model, feats)

    # The autoencoder's training set. With published transits, train on
    # windows that genuinely contain none -- which is what makes
    # reconstruction error an independent second opinion rather than an echo
    # of IsolationForest's own decisions.
    if known_labels is not None and (known_labels == 0).sum() >= _MIN_NORMAL_WINDOWS:
        normal_mask = known_labels == 0
        ae_basis = "windows with no published transit"
    else:
        normal_mask = ~scored["is_anomaly"].values
        ae_basis = "IsolationForest pseudo-labels (no catalogue coverage)"

    ae_model = train_autoencoder(windows[normal_mask]) if normal_mask.sum() >= _MIN_NORMAL_WINDOWS else None
    embeddings, recon_errors = None, None
    if ae_model is not None:
        embeddings = embed_windows(ae_model, windows)
        recon_errors = reconstruction_error(ae_model, windows)
        trace.append(f"[astronomy_agent] autoencoder trained on {int(normal_mask.sum())} "
                     f"{ae_basis}")

    scored = annotate_windows(scored, recon_errors)
    top_anomalies = scored.sort_values("anomaly_score", ascending=False).head(5)

    result = {
        "target": target,
        "kepid": kepid,
        "data_source": source,
        "cadence_minutes": round(cadence_min, 2),
        "baseline_days": round(float(times[-1] - times[0]), 1),
        "window_points": window,
        "stride_points": stride,
        "n_windows": len(windows),
        "n_flagged_anomalies": int(scored["is_anomaly"].sum()),
        "top_anomaly_windows": top_anomalies.reset_index().rename(
            columns={"index": "window_id"}).to_dict("records"),
        "mean_reconstruction_error": float(recon_errors.mean()) if recon_errors is not None else None,
        "autoencoder_trained_on": ae_basis if ae_model is not None else None,
        "_embeddings": embeddings,
        "_flux": flux,
        "_time": times,
        "_anomaly_mask": scored["is_anomaly"].values,
        # Per-window published-transit labels, so the report can say of each
        # individual flag whether it lands on a real transit -- not just the
        # aggregate recall.
        "_known_labels": known_labels,
    }

    # What the archive knows, and whether the detector found it. This is the
    # difference between "12 windows flagged" and "12 flagged, 9 of them on
    # the transits of a confirmed planet" -- the second is checkable.
    if koi_info and koi_info.get("known"):
        result["known_planets"] = {
            "n_kois": koi_info["n_kois"],
            "n_confirmed": koi_info["n_confirmed"],
            "disposition": koi_info["label"],
            "planet_names": [n for n in koi_info.get("planet_names", []) if n],
            "periods_days": koi_info["periods_days"][:6],
            "radii_earth": koi_info.get("radii_earth", [])[:6],
            "depths_ppm": koi_info.get("depths_ppm", [])[:6],
        }
    if known_labels is not None and len(known_labels) == len(scored):
        flagged = scored["is_anomaly"].values
        hits = int((flagged & (known_labels == 1)).sum())
        n_transit = int(known_labels.sum())
        result["transit_recovery"] = {
            "n_windows_with_known_transit": n_transit,
            "n_flagged_on_known_transit": hits,
            "recall": round(hits / n_transit, 3) if n_transit else None,
            "precision": round(hits / int(flagged.sum()), 3) if flagged.sum() else None,
        }
        trace.append(f"[astronomy_agent] of {int(flagged.sum())} flagged windows, {hits} fall on a "
                     f"published transit (recall {result['transit_recovery']['recall']}, "
                     f"precision {result['transit_recovery']['precision']})")

    trace.append(f"[astronomy_agent] {target} ({source}): "
                 f"{result['n_flagged_anomalies']}/{len(windows)} windows flagged anomalous")
    return {**state, "astronomy_result": result, "trace": trace}


if __name__ == "__main__":
    out = astronomy_node({"target": "KIC 10874614", "trace": []})
    for line in out["trace"]:
        print(line)
    r = out["astronomy_result"]
    print(f"\nknown planets  : {r.get('known_planets')}")
    print(f"recovery       : {r.get('transit_recovery')}")
