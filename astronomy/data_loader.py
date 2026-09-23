"""Light curve ingestion, free & public.

Primary path: NASA Kepler/TESS light curves via `lightkurve` (queries the
public MAST archive, no API key). Falls back to a small bundled synthetic
set if the network/archive is unreachable, so the rest of the pipeline
(feature extraction, anomaly detection, embeddings) can always be developed
and tested offline.

BTP note: for the full 3-month project, replace SAMPLE_TARGETS with a
larger batch (100-300 targets per TECHNICAL_PLAN.md Week 2) once the
pipeline below is validated on this small sample.
"""

import numpy as np
import pandas as pd

# A handful of well-known Kepler objects of interest (KOIs) -- mix of
# confirmed exoplanet hosts (should show transit-like anomalies) and
# quiet targets (should look "normal"). Free, public target IDs.
SAMPLE_TARGETS = [
    "KIC 11446443",   # Kepler-1, confirmed hot Jupiter host -- clear transits
    "KIC 10666592",   # Kepler-6
    "KIC 8462852",    # "Tabby's Star" -- famous for irregular, unexplained dips
    "KIC 6922244",    # quiet-ish comparison target
]


def fetch_light_curve(target: str, quarter: int | None = None) -> pd.DataFrame:
    """Download and lightly clean one light curve. Returns a DataFrame with
    columns ['time', 'flux'] normalized to median 1.0."""
    import lightkurve as lk

    search = lk.search_lightcurve(target, mission="Kepler", quarter=quarter)
    if len(search) == 0:
        raise ValueError(f"No Kepler light curves found for target '{target}'")

    lc = search[0].download()
    lc = lc.remove_nans().normalize()

    df = pd.DataFrame({"time": lc.time.value, "flux": lc.flux.value})
    return df


def generate_synthetic_light_curve(n_points: int = 2000, n_transits: int = 6,
                                    transit_depth: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Offline fallback / fast-iteration dataset: realistic-looking stellar
    flux with periodic transit dips (the 'anomaly' signal) plus noise and
    slow stellar variability. Used when the network or MAST archive is
    unavailable, and for quick unit tests."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 90, n_points)  # ~90 days, Kepler-like cadence

    stellar_variability = 0.003 * np.sin(2 * np.pi * time / 13.0)
    noise = rng.normal(0, 0.0015, n_points)
    flux = 1.0 + stellar_variability + noise

    period = time[-1] / n_transits
    for k in range(n_transits):
        center = period * (k + 0.5)
        width = 0.6
        dip = transit_depth * np.exp(-0.5 * ((time - center) / (width / 2.355)) ** 2)
        flux -= dip

    return pd.DataFrame({"time": time, "flux": flux})


def load_sample_batch(targets: list[str] | None = None, use_synthetic_fallback: bool = True) -> dict[str, pd.DataFrame]:
    targets = targets or SAMPLE_TARGETS
    out = {}
    for t in targets:
        try:
            out[t] = fetch_light_curve(t)
            print(f"[data_loader] fetched real light curve for {t} ({len(out[t])} points)")
        except Exception as exc:  # noqa: BLE001
            print(f"[data_loader] could not fetch '{t}' from MAST ({exc}).", end=" ")
            if use_synthetic_fallback:
                seed = abs(hash(t)) % (2**32)
                out[t] = generate_synthetic_light_curve(seed=seed)
                print("using synthetic fallback for offline development.")
            else:
                print("skipping.")
    return out


if __name__ == "__main__":
    batch = load_sample_batch()
    for name, df in batch.items():
        print(f"{name}: {len(df)} points, flux range [{df.flux.min():.4f}, {df.flux.max():.4f}]")
