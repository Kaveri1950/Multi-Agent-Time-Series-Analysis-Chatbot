"""Look at a series and work out what KIND of thing it is, before choosing a detector.

This is the step that makes the system decide rather than look up. Every
other choice in the pipeline is made from a table of measured skill; this one
is made by inspecting data the agent has never seen, with no labels and no
domain tag attached.

WHY IT MATTERS, IN ONE MEASUREMENT

A Kepler transit recurs ~221 times with near-identical shape, so it is the
LEAST unusual feature of its light curve -- a motif. A market crash happens
once -- a discord. Scored the wrong way round on KIC 10074466, the matrix
profile gives ROC 0.303, below chance. Scored the right way round, 0.697.

The agent cannot be told which is which: a new star has no labels, and "it
came from a telescope" is not enough -- a gamma-ray burst is also a light
curve and behaves like a market crash, not like a transit. The distinction
is PERIODICITY, and periodicity is measurable.

    periodicity_ratio = height of the best Lomb-Scargle peak / median power

Measured over 10 Kepler targets and 10 Indian equities:

    one-off (equities)    median    12.4    range  9.4 - 15.0
    periodic (Kepler)     median 1,939.3    range 74.1 - 49,974.7

Two orders of magnitude apart, with no overlap; a threshold of 33 separated
all 20 correctly. The band below is deliberately wider than that, so that
anything ambiguous is reported as uncertain rather than guessed.

Lomb-Scargle rather than an FFT because Kepler light curves have gaps --
spacecraft rolls, safe modes, downlinks -- and Lomb-Scargle is built for
unevenly sampled data where an FFT silently assumes regularity.
"""

from __future__ import annotations

import numpy as np

#: Below this the series is treated as one-off; above it as periodic. The
#: gap between them is the uncertain band: observed samples sat at ~12 and
#: ~1,900, so anything landing between 30 and 60 is unlike both and the
#: honest answer is "not sure", not a coin flip.
ONE_OFF_MAX = 30.0
PERIODIC_MIN = 60.0

#: Subsample long series before the periodogram. A 65,000-point light curve
#: takes seconds to transform and the peak is unchanged by thinning, since
#: the periods of interest are thousands of samples long.
MAX_POINTS = 8000


def profile_series(values: np.ndarray, times: np.ndarray | None = None) -> dict:
    """Label-free description of a series.

    `times` is optional but matters for irregular sampling: passing real
    timestamps lets the periodogram use actual spacing rather than assuming
    one sample per unit.
    """
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    if times is None:
        times = np.arange(len(values), dtype=float)
    times = np.asarray(times, dtype=float).ravel()[: len(values)]
    values, times = values[finite], times[finite]
    if len(values) < 50:
        return {"n": len(values), "periodicity_ratio": np.nan,
                "structure": "unknown", "reason": "series too short to profile"}

    if len(values) > MAX_POINTS:
        step = len(values) // MAX_POINTS
        values, times = values[::step], times[::step]

    span = float(times[-1] - times[0]) or float(len(values))
    dt = float(np.median(np.diff(times))) or 1.0
    # Nyquist. Searching above it is searching for frequencies the sampling
    # cannot represent -- an earlier version used a fixed 2 cycles/day, which
    # is four times Nyquist for daily equity data.
    f_max = 0.5 / dt
    # A period longer than a third of the baseline cannot repeat enough times
    # to be called periodic at all.
    f_min = 3.0 / span

    ratio, peak_period, peak_power = np.nan, np.nan, np.nan
    if f_max > f_min:
        try:
            from astropy.timeseries import LombScargle

            freq, power = LombScargle(times, values).autopower(
                minimum_frequency=f_min, maximum_frequency=f_max, samples_per_peak=5)
            power = np.asarray(power, dtype=float)
            good = np.isfinite(power)
            if good.sum() > 10:
                best = int(np.argmax(np.where(good, power, -np.inf)))
                background = float(np.median(power[good]))
                peak_power = float(power[best])
                peak_period = float(1.0 / freq[best])
                ratio = peak_power / background if background > 0 else np.inf
        except Exception:
            pass

    centred = values - values.mean()
    acf = np.correlate(centred, centred, "full")[len(centred) - 1:]
    acf = acf / acf[0] if acf[0] else acf
    # Ignore the first few lags: everything is autocorrelated at lag 1, and
    # it says nothing about repetition.
    tail = acf[5: min(len(acf), 2000)]
    max_acf = float(np.max(np.abs(tail))) if len(tail) else np.nan

    structure = ("periodic" if ratio >= PERIODIC_MIN
                 else "one-off" if ratio <= ONE_OFF_MAX else "uncertain")
    return {
        "n": int(len(values)),
        "span": round(span, 3),
        "sampling_interval": round(dt, 6),
        "periodicity_ratio": round(float(ratio), 2) if np.isfinite(ratio) else np.nan,
        "dominant_period": round(peak_period, 4) if np.isfinite(peak_period) else np.nan,
        "max_autocorrelation": round(max_acf, 3) if np.isfinite(max_acf) else np.nan,
        "structure": structure,
    }


def recommend(profile: dict) -> dict:
    """Which detector family this structure calls for, and why.

    The `reason` is written for a human to read in a trace, because a
    decision the system cannot explain is one nobody can check.
    """
    structure = profile.get("structure")
    ratio = profile.get("periodicity_ratio")

    if structure == "periodic":
        return {
            "family": "periodic",
            "primary": "bls",
            "matrix_profile": {"motif": True, "normalize": True},
            "reason": (f"The signal repeats (periodicity ratio {ratio}, dominant period "
                       f"{profile.get('dominant_period')}). A repeating event is the LEAST "
                       f"unusual thing in its own series, so generic anomaly detection looks "
                       f"for the opposite of what is there. A period search is the right tool."),
        }
    if structure == "one-off":
        return {
            "family": "one-off",
            "primary": "hampel_mad",
            "matrix_profile": {"motif": False, "normalize": False},
            "reason": (f"No repeating structure (periodicity ratio {ratio}). Events here are "
                       f"unique, so they are discords, and their SIZE carries the signal -- "
                       f"which is why the distance is left un-normalised."),
        }
    return {
        "family": "uncertain",
        "primary": "hampel_mad",
        "matrix_profile": {"motif": False, "normalize": False},
        "reason": (f"Periodicity ratio {ratio} falls between the one-off and periodic ranges "
                   f"seen in calibration, so the structure is genuinely ambiguous. Defaulting "
                   f"to the robust threshold detector, which assumes least."),
    }


def profile_and_recommend(values: np.ndarray, times: np.ndarray | None = None) -> dict:
    prof = profile_series(values, times)
    return {**prof, "recommendation": recommend(prof)}


if __name__ == "__main__":
    import json
    import warnings

    warnings.filterwarnings("ignore")
    from astronomy.bulk_download import load_manifest, load_target
    from quant.data import load_panel

    print("Profiling series with no labels and no domain tag.\n")
    rows = []
    for _, r in load_manifest().sample(frac=1.0, random_state=11).head(5).iterrows():
        d = load_target(int(r["kepid"]))
        p = profile_and_recommend(d["flux"].to_numpy(), d["time"].to_numpy())
        rows.append((f"KIC {int(r['kepid'])}", "periodic", p))

    close, volume = load_panel()
    tv = (close * volume).rolling(21, min_periods=10).mean()
    for tk in list(tv.mean().nlargest(5).index):
        s = close[tk].pct_change(fill_method=None).dropna()
        rows.append((tk, "one-off", profile_and_recommend(s.to_numpy())))

    correct = 0
    for name, truth, p in rows:
        got = p["structure"]
        correct += got == truth
        print(f"  {name:16} ratio {p['periodicity_ratio']:>12}  -> {got:9} "
              f"(truth {truth})  primary={p['recommendation']['primary']}")
    print(f"\n  {correct}/{len(rows)} correct")
    print("\n  example reasoning:\n   ", rows[0][2]["recommendation"]["reason"])
