"""Box Least Squares transit search -- a model that knows what it is looking for.

WHY THIS EXISTS

Every detector in `anomaly/` scores windows in isolation, and on Kepler they
all land between 1.2x and 1.4x lift. Three independent measurements say the
same thing about why:

  1. 41.7% of labelled transits are no more extreme than a typical
     non-transit window, so window statistics cannot separate them.
  2. An FFT of a 12-hour window cannot see a 4.77-day period -- the
     information is not inside the window at all.
  3. The matrix profile scores transits BELOW CHANCE (ROC 0.303 on KIC
     10074466), because a transit recurring 221 times is the least unusual
     feature of the light curve. Inverted, the same numbers give 0.697.

A transit is not an anomaly. It is a MOTIF -- a periodic, box-shaped dip that
repeats. BLS is the matched filter for exactly that: it scans candidate
periods, folds the light curve on each, and measures how well a box fits.
Signal invisible in any single window becomes overwhelming once 221 of them
are stacked, because noise falls as sqrt(N) while the dip does not.

THE MISTAKE THIS FILE IS CAREFUL NOT TO REPEAT

An earlier attempt in this project was retracted because the period grid was
roughly 17x too coarse -- 0.113% spacing where 0.0068% was needed. BLS is
unforgiving about this. Too coarse and every trial period is slightly wrong,
the folded transits smear across phase, and the search silently returns
nothing while looking like it ran correctly.

The grid spacing here was therefore chosen by MEASUREMENT rather than by
taste: three spacings were run against a target with a known period, and
accuracy had already saturated at 0.02% (see PERIOD_SPACING_PCT). Note that
astropy's own `autoperiod` is not the answer either -- it asks for 75 million
trial periods on this data, which is correct and unusable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trial transit durations, in days. Spans the catalogue: measured KOI
#: durations run from 0.8h at the 1st percentile to 16h at the 95th, with a
#: median of 3.8h.
DEFAULT_DURATIONS = np.array([2.0, 4.0, 8.0]) / 24.0

#: Below three transits a "period" is a coincidence rather than a detection.
MIN_N_TRANSIT = 3

#: Relative spacing of the log-uniform period grid, in percent.
#:
#: astropy's `autoperiod` is mathematically correct and computationally
#: unusable here: over a 1052-day baseline with a 1-hour minimum duration it
#: asks for 75 MILLION trial periods (spacing 0.00001%), because its grid
#: scales as baseline^2 / min_duration. That is not wrong, it is simply far
#: finer than the signal needs.
#:
#: Measured on KIC 10001368, whose true period is 4.76831 d:
#:     0.020%   23,028 periods    31 s   error 0.004%   SDE 22.1
#:     0.010%   46,054 periods    60 s   error 0.005%   SDE 21.5
#:     0.005%   92,105 periods   134 s   error 0.000%   SDE 22.2
#:
#: Accuracy has already saturated at 0.02%, so finer grids buy nothing but
#: runtime. Note the direction of the earlier mistake: the retracted attempt
#: used 0.113%, five times COARSER than this, and found nothing.
PERIOD_SPACING_PCT = 0.02

#: Widest period searched. The KOI catalogue's median is 9.8 days and its
#: 75th percentile 40.7, so 50 days covers most of the population while
#: keeping the grid affordable. Longer-period planets exist and this will
#: miss them -- a stated limit, not an oversight.
MAX_PERIOD_DAYS = 50.0
MIN_PERIOD_DAYS = 0.5


def period_grid(minimum_period: float = MIN_PERIOD_DAYS,
                maximum_period: float = MAX_PERIOD_DAYS,
                spacing_pct: float = PERIOD_SPACING_PCT) -> np.ndarray:
    """Log-uniform period grid at a fixed RELATIVE spacing.

    Log-uniform rather than linear because what matters is fractional period
    error: being 0.01 days wrong on a 1-day period smears the fold
    completely, while the same absolute error on a 40-day period is
    harmless. A linear grid oversamples long periods and starves short ones.
    """
    step = 1.0 + spacing_pct / 100.0
    n = int(np.ceil(np.log(maximum_period / minimum_period) / np.log(step)))
    return minimum_period * np.power(step, np.arange(n + 1))


def search(time: np.ndarray, flux: np.ndarray,
           durations: np.ndarray = DEFAULT_DURATIONS,
           minimum_period: float = MIN_PERIOD_DAYS,
           maximum_period: float = MAX_PERIOD_DAYS,
           spacing_pct: float = PERIOD_SPACING_PCT) -> dict:
    """Run BLS and return the best-fitting periodic box.

    Returns the period, epoch, duration and depth of the strongest candidate,
    plus the full periodogram for plotting and the signal detection
    efficiency -- the peak's height measured in standard deviations of the
    periodogram itself, which is the number that says whether a peak is real.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    if len(time) < 100:
        raise ValueError(f"only {len(time)} usable points -- too few for a period search")

    baseline = float(time[-1] - time[0])
    model = BoxLeastSquares(time, flux)
    # A period longer than baseline/3 cannot show the three transits needed
    # to call anything periodic, so the cap is whichever is smaller.
    periods = period_grid(minimum_period,
                          min(maximum_period, baseline / MIN_N_TRANSIT),
                          spacing_pct)
    pg = model.power(periods, durations)
    best = int(np.argmax(pg.power))

    power = np.asarray(pg.power, dtype=float)
    finite = power[np.isfinite(power)]
    # Signal detection efficiency: how far the peak stands above the
    # periodogram's own scatter. A tall peak on a noisy periodogram is not a
    # detection; SDE is what distinguishes the two, and ~7 is the usual
    # threshold in the transit literature.
    sde = float((power[best] - finite.mean()) / finite.std()) if finite.std() > 0 else np.nan

    return {
        "period": float(pg.period[best]),
        "epoch": float(pg.transit_time[best]),
        "duration": float(pg.duration[best]),
        "depth": float(pg.depth[best]),
        "depth_snr": float(pg.depth_snr[best]) if hasattr(pg, "depth_snr") else np.nan,
        "power": float(power[best]),
        "sde": sde,
        "n_periods_tried": len(periods),
        "period_grid_spacing_pct": float(100 * np.median(np.diff(periods)) / np.median(periods)),
        "baseline_days": baseline,
        "n_points": len(time),
        "periodogram": {"period": np.asarray(pg.period), "power": power},
    }


def fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> tuple:
    """Phase-fold the light curve, phase measured in days from mid-transit.

    The picture that makes a detection convincing: 221 transits stacked on
    top of each other, where each one alone is invisible.
    """
    time = np.asarray(time, dtype=float)
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    order = np.argsort(phase)
    return phase[order], np.asarray(flux, dtype=float)[order]


def binned_fold(time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
                n_bins: int = 200) -> pd.DataFrame:
    """Phase-folded curve averaged into bins -- the stacked transit."""
    phase, f = fold(time, flux, period, epoch)
    edges = np.linspace(phase.min(), phase.max(), n_bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    frame = pd.DataFrame({"bin": idx, "phase": phase, "flux": f})
    out = frame.groupby("bin").agg(phase=("phase", "mean"), flux=("flux", "mean"),
                                   scatter=("flux", "std"), n=("flux", "size"))
    return out.reset_index(drop=True)


def compare_with_truth(result: dict, kepid: int, koi: pd.DataFrame | None = None,
                       tolerance_pct: float = 1.0) -> dict:
    """Did BLS recover the period NASA publishes?

    The only honest test. A high SDE means the search found SOMETHING
    periodic; only agreement with the published ephemeris says it found the
    planet. Harmonics are checked explicitly because recovering 2P or P/2 is
    a partial success, not a failure -- the signal is there, the fold is just
    at the wrong multiple.
    """
    from astronomy.koi_catalog import fetch_koi_table

    koi = fetch_koi_table() if koi is None else koi
    rows = koi[koi["kepid"] == kepid]
    if rows.empty:
        return {"kepid": kepid, "matched": False, "reason": "no KOI entry"}

    found = result["period"]
    best = None
    for _, r in rows.iterrows():
        true_p = float(r["koi_period"])
        for mult, label in ((1.0, "exact"), (2.0, "2x"), (0.5, "half"),
                            (3.0, "3x"), (1 / 3, "third")):
            err = abs(found - true_p * mult) / (true_p * mult) * 100
            if best is None or err < best["period_error_pct"]:
                best = {"true_period": true_p, "harmonic": label,
                        "period_error_pct": err,
                        "disposition": r["koi_disposition"],
                        "true_depth_ppm": float(r["koi_depth"]) if pd.notna(r["koi_depth"]) else np.nan}
    matched = best["period_error_pct"] <= tolerance_pct
    return {
        "kepid": kepid, "found_period": found, "sde": result["sde"],
        "matched": bool(matched),
        "exact_match": bool(matched and best["harmonic"] == "exact"),
        **best,
    }


def run_batch(limit: int = 20, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """BLS across many cached targets, scored against NASA's ephemerides."""
    from astronomy.bulk_download import load_manifest, load_target
    from astronomy.koi_catalog import fetch_koi_table

    manifest = load_manifest().sample(frac=1.0, random_state=seed).head(limit)
    koi = fetch_koi_table()
    rows = []
    for _, row in manifest.iterrows():
        kepid = int(row["kepid"])
        try:
            df = load_target(kepid)
            res = search(df["time"].to_numpy(), df["flux"].to_numpy())
            cmp = compare_with_truth(res, kepid, koi)
            cmp["label"] = row["label"]
            cmp["grid_spacing_pct"] = res["period_grid_spacing_pct"]
            rows.append(cmp)
            if verbose:
                mark = "MATCH" if cmp["matched"] else "miss "
                print(f"  [{mark}] KIC {kepid} ({row['label'][:4]})  "
                      f"found {res['period']:8.4f}d  true {cmp['true_period']:8.4f}d  "
                      f"err {cmp['period_error_pct']:7.3f}%  SDE {res['sde']:6.1f}  "
                      f"{cmp['harmonic']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [ERROR] KIC {kepid}: {type(exc).__name__}: {exc}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    print(f"BLS transit search on {args.limit} cached targets\n")
    res = run_batch(limit=args.limit)
    if res.empty:
        raise SystemExit("no results")

    print(f"\n{'=' * 64}")
    print(f"  period recovered within 1%:        {res['matched'].mean():.1%}")
    print(f"  exact period (not a harmonic):     {res['exact_match'].mean():.1%}")
    for lab in res["label"].unique():
        sub = res[res["label"] == lab]
        print(f"    {lab:16} {sub['matched'].mean():6.1%}  (n={len(sub)})")
    print(f"  median SDE, matched:   {res.loc[res['matched'], 'sde'].median():.1f}")
    print(f"  median SDE, unmatched: {res.loc[~res['matched'], 'sde'].median():.1f}")
    print(f"  median grid spacing:   {res['grid_spacing_pct'].median():.4f}%  "
          f"(the retracted attempt used 0.113%)")
