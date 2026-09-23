"""A small, real, persisted dataset of Kepler light curves -- built once
via build_dataset(), then queryable by the chatbot without a live fetch
per question. Extends astronomy/data_loader.py's single-target fetch
(with its existing real-fetch + synthetic-fallback behavior) to a batch,
caches each light curve to disk (data/cache/lightcurves/*.parquet), and
precomputes an anomaly-detection summary per target (data/cache/
lightcurve_dataset_index.json) so "which of my targets has the most
anomalies" is a lookup, not a re-run of the whole pipeline live.

TECHNICAL_PLAN.md's Week 2 goal was 100-300 targets; this ships a smaller
real, working batch (DEFAULT_TARGETS below) as a genuine starting dataset
-- call build_dataset() with a longer target list to scale it up later.
"""

import json
from pathlib import Path

import pandas as pd

from astronomy.anomaly_detector import STRIDE, WINDOW_SIZE, extract_features, fit_isolation_forest, make_windows, score_anomalies
from astronomy.data_loader import fetch_light_curve, generate_synthetic_light_curve
from astronomy.ts_embedder import reconstruction_error, train_autoencoder

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "lightcurves"
INDEX_PATH = Path(__file__).parent.parent / "data" / "cache" / "lightcurve_dataset_index.json"

# Real, publicly documented Kepler Objects of Interest -- confirmed
# exoplanet hosts (should show real transit-like anomalies), one famous
# irregular-dip target, and a couple of quieter comparison stars. Any
# target that fails to resolve via MAST falls back to a labeled synthetic
# curve automatically (astronomy/data_loader.py's existing behavior) --
# the dataset index records which is which, honestly.
DEFAULT_TARGETS = [
    "KIC 11446443",   # Kepler-1b / TrES-2b -- confirmed hot Jupiter
    "KIC 10666592",   # Kepler-6b
    "KIC 8462852",    # "Tabby's Star" -- famous irregular, unexplained dips
    "KIC 6922244",    # quiet-ish comparison target
    "KIC 11804465",   # Kepler-7b
    "KIC 10874614",   # Kepler-8b
    "KIC 5780885",    # Kepler-4b
    "KIC 6541920",    # Kepler-5b
    "KIC 9631995",    # Kepler-9 -- multi-planet system
    "KIC 11853905",   # Kepler-10 -- small rocky planet
    "KIC 8191672",    # Kepler-11 -- six-planet system
    "KIC 9941662",    # Kepler-12b
    "KIC 9651668",    # Kepler-14b
    "KIC 8395660",    # Kepler-15b
    "KIC 12365184",   # Kepler-16b -- circumbinary ("Tatooine") planet
    "KIC 4570949",    # Kepler-17b
    "KIC 9145861",    # Kepler-18 system
    "KIC 6774038",    # Kepler-19 system
    "KIC 11512246",   # Kepler-20 system
    "KIC 8866102",    # comparison target
]


def _cache_path(target: str) -> Path:
    safe_name = target.replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_name}.parquet"


def _analyze(flux) -> dict:
    windows = make_windows(flux, window_size=WINDOW_SIZE, stride=STRIDE)
    feats = extract_features(windows)
    iso_model = fit_isolation_forest(feats)
    scored = score_anomalies(iso_model, feats)
    anomaly_mask = scored["is_anomaly"].values

    recon_err_mean = None
    pseudo_normal = windows[~anomaly_mask]
    if pseudo_normal.shape[0] >= 10:
        ae_model = train_autoencoder(pseudo_normal)
        recon_err_mean = float(reconstruction_error(ae_model, windows).mean())

    return {
        "n_windows": len(windows),
        "n_flagged_anomalies": int(anomaly_mask.sum()),
        "anomaly_rate": round(float(anomaly_mask.mean()), 4) if len(windows) else 0.0,
        "top_anomaly_score": round(float(scored["anomaly_score"].max()), 4) if len(scored) else None,
        "mean_reconstruction_error": round(recon_err_mean, 6) if recon_err_mean is not None else None,
    }


def build_dataset(targets: list[str] | None = None, force_refetch: bool = False) -> pd.DataFrame:
    """Fetches (or loads from cache) each target's light curve, runs the
    same IsolationForest + autoencoder pipeline as a single live chatbot
    query, and writes both the raw light curve (parquet) and a summary
    index (JSON) to disk. Safe to re-run -- cached targets are skipped
    unless force_refetch=True."""
    targets = targets or DEFAULT_TARGETS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for target in targets:
        cache_file = _cache_path(target)
        if cache_file.exists() and not force_refetch:
            lc = pd.read_parquet(cache_file)
            source = json.loads(cache_file.with_suffix(".meta.json").read_text())["source"] if cache_file.with_suffix(".meta.json").exists() else "cached"
            print(f"[dataset] {target}: loaded from cache ({len(lc)} points)")
        else:
            try:
                lc = fetch_light_curve(target)
                source = "real (MAST/Kepler via lightkurve)"
            except Exception as exc:  # noqa: BLE001
                seed = abs(hash(target)) % (2**32)
                lc = generate_synthetic_light_curve(seed=seed)
                source = f"synthetic fallback ({exc})"
            lc.to_parquet(cache_file)
            cache_file.with_suffix(".meta.json").write_text(json.dumps({"source": source}))
            print(f"[dataset] {target}: {source} ({len(lc)} points)")

        analysis = _analyze(lc["flux"].values)
        rows.append({"target": target, "source": source, "n_points": len(lc), **analysis})
        print(f"           -> {analysis['n_windows']} windows, {analysis['n_flagged_anomalies']} flagged anomalous")

    index_df = pd.DataFrame(rows)
    INDEX_PATH.write_text(index_df.to_json(orient="records", indent=2))
    return index_df


def load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError("No light curve dataset built yet -- run astronomy.dataset.build_dataset() first.")
    return pd.read_json(INDEX_PATH)


def load_cached_light_curve(target: str) -> pd.DataFrame:
    cache_file = _cache_path(target)
    if not cache_file.exists():
        raise FileNotFoundError(f"No cached light curve for '{target}' -- run astronomy.dataset.build_dataset() first.")
    return pd.read_parquet(cache_file)


if __name__ == "__main__":
    df = build_dataset()
    print("\n=== Dataset summary ===")
    print(df[["target", "source", "n_points", "n_flagged_anomalies", "anomaly_rate"]].to_string(index=False))
    print(f"\n{(df['source'].str.startswith('real')).sum()} of {len(df)} targets used real MAST data.")
