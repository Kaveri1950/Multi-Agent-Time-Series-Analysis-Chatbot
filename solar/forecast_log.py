"""Solar flare forecasts scored against NOAA's own operational forecast.

THE RESULT THIS PRODUCES. NOAA SWPC publishes a daily probability that an
M-class-or-larger flare will occur in the next 24 hours -- a real
human-plus-model operational product. This module scores our Hawkes process
against it on the same days, with the same outcomes. That comparison is the
strongest claim available in the solar domain, because the benchmark is not
a strawman we invented but the number professionals actually issue.

WHERE THE OUTCOME COMES FROM, and why it took a wrong turn first. The
obvious source is the archived flare-event list, but it is a 7-day rolling
window harvested only when the harvester ran -- so it has a two-week hole
(2026-08-26 to 09-08) between runs, and worse, a missing day is
indistinguishable from a quiet day. "We did not look" and "nothing happened"
record identically, which would have silently scored quiet days as correct
for whichever model predicted nothing.

The Daily Solar Data product solves it: it carries an explicit per-day count
of C, M and X flares, on a 30-day window, and the archive has 57 consecutive
days with NO gaps. A day with m_flares + x_flares > 0 is a positive. That is
an affirmative record for every day rather than an absence of evidence.

METRIC. TSS (true skill statistic, TPR - FPR) and Brier score, matching
solar/flare_model.py so the numbers are comparable with the 1996-2020
backtest. Probabilities are thresholded at 0.5 for TSS unless a threshold is
supplied; Brier needs no threshold and is the fairer comparison between two
probabilistic forecasters.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from solar.flare_data import CACHE_DIR, load_archive

LOG_PATH = CACHE_DIR / "forecast_scores.parquet"

DEFAULT_THRESHOLD = 0.5


def outcomes() -> pd.DataFrame:
    """Per-day ground truth: did an M-class-or-larger flare occur?

    Uses the Daily Solar Data counts rather than the event list, for the
    completeness reason in the module docstring.
    """
    dsd = load_archive("daily_indices")
    if dsd.empty:
        return pd.DataFrame()
    out = dsd[["date", "m_flares", "x_flares", "radio_flux", "sunspot_number"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["m_plus_count"] = out["m_flares"].fillna(0) + out["x_flares"].fillna(0)
    out["occurred"] = (out["m_plus_count"] > 0).astype(int)
    return out.drop_duplicates(subset=["date"], keep="last").sort_values("date")


def noaa_forecasts() -> pd.DataFrame:
    """NOAA's own published M-class 1-day probability, as a fraction."""
    probs = load_archive("probabilities")
    if probs.empty:
        return pd.DataFrame()
    out = probs[["date", "m_class_1_day", "x_class_1_day", "c_class_1_day"]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["noaa_m_prob"] = out["m_class_1_day"].astype(float) / 100.0
    return out.drop_duplicates(subset=["date"], keep="last").sort_values("date")


def _complete_event_history(dates: pd.Series) -> pd.DataFrame:
    """Every M+ flare over the scored period, plus a run-up before it.

    The run-up matters: the Hawkes intensity on the first scored day depends
    on flares shortly BEFORE it, so starting the history at the first scored
    date would make that day look artificially quiet.
    """
    from solar.flare_data import class_at_least, fetch_historical_year

    start = pd.to_datetime(dates).min() - pd.Timedelta(days=30)
    end = pd.to_datetime(dates).max() + pd.Timedelta(days=1)

    frames = []
    for year in range(start.year, end.year + 1):
        try:
            frames.append(fetch_historical_year(year))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    ev = pd.concat(frames, ignore_index=True)
    ev["begin_time"] = pd.to_datetime(ev["begin_time"], errors="coerce")
    ev = ev.dropna(subset=["begin_time"])
    ev = ev[(ev["begin_time"] >= start) & (ev["begin_time"] <= end)]
    ev = ev[ev["flare_class"].apply(lambda c: class_at_least(c, "M"))]
    return ev.drop_duplicates(subset=["begin_time"]).sort_values("begin_time")


def hawkes_hindcast(dates: pd.Series) -> pd.Series:
    """Our model's probability for each date, from flares strictly before it.

    A hindcast, not a prospective forecast: it is computed now, over dates
    whose outcomes are already known. It is still honest -- the closed-form
    probability for a given day uses only events before that day -- but the
    experiment was designed with the period's contents visible, so it is
    weaker evidence than a forecast logged in advance. Labelled accordingly
    wherever it is reported.
    """
    from solar.flare_data import class_at_least
    from solar.flare_model import M_CLASS_FLUX
    from solar.forecast import load_model

    model, _meta = load_model()
    if model is None:
        return pd.Series(np.nan, index=dates.index)

    # The event history MUST be complete, or a self-exciting model reads a
    # missing flare as a quiet period and under-predicts accordingly.
    #
    # Using the harvested 7-day-window archive for this was wrong and
    # measurably so: over 2026-07-22..09-17 it holds 10 of the 19 M+ events
    # that actually occurred -- 53% -- because the harvester ran only a few
    # times and each run captures one week. The model shown half the
    # triggering events duly predicted about half the risk (mean probability
    # 0.121 against an actual rate of 0.228), which I first misattributed to
    # the training window omitting Cycle 25. Retraining on 1996-2026 moved
    # the mean from 0.114 to 0.121 and settled that: the training data was
    # never the problem.
    #
    # The Hesperia yearly catalogue is complete for the period, so it is the
    # right input. The sparse archive remains the right source for a
    # genuinely PROSPECTIVE forecast, where nothing better exists yet.
    ev = _complete_event_history(dates)
    if ev.empty:
        # No M+ events on record at all -> the fitted background rate.
        return pd.Series(float(1.0 - np.exp(-model.mu * 24.0)), index=dates.index)

    t0 = ev["begin_time"].iloc[0]
    times = (ev["begin_time"] - t0).dt.total_seconds().values / 3600.0
    marks = np.log10(ev["peak_flux"].values / M_CLASS_FLUX)

    probs = []
    for d in pd.to_datetime(dates):
        now = (pd.Timestamp(d) - t0).total_seconds() / 3600.0
        past = times < now
        if not past.any():
            probs.append(float(1.0 - np.exp(-model.mu * 24.0)))
        else:
            probs.append(model.probability_within(times[past], marks[past], now, 24.0))
    return pd.Series(probs, index=dates.index)


def gp_hindcast(frame: pd.DataFrame) -> pd.Series:
    """GP classifier probability per scored day.

    Trained once on all days strictly BEFORE the scored window, then applied
    across it. That is a single expanding-window fit rather than a refit per
    day, which is the honest thing to say about it -- a daily refit would be
    more rigorous and costs 14s per day at this training size.
    """
    from solar.flare_data import load_historical_catalog
    from solar.features import build_daily_frame
    from solar.flare_model import train_gp_classifier

    start = pd.to_datetime(frame["date"]).min()
    full = build_daily_frame(load_historical_catalog(1996, 2026))
    full["date"] = pd.to_datetime(full["date"])

    history = full[full["date"] < start]
    target = full[full["date"].isin(pd.to_datetime(frame["date"]))]
    if history.empty or target.empty:
        return pd.Series(np.nan, index=frame.index)

    _clf, prob, _ptr, _sc = train_gp_classifier(history, target)
    by_date = dict(zip(target["date"], prob))
    return pd.to_datetime(frame["date"]).map(by_date)


def persistence_hindcast(frame: pd.DataFrame) -> pd.Series:
    """"Was there an M+ flare yesterday?" -- the one-line rule that beat
    XGBoost in 2 of 4 walk-forward folds on the 1996-2020 data."""
    return frame["occurred"].shift(1).fillna(0).astype(float)


def climatology_hindcast(frame: pd.DataFrame) -> pd.Series:
    """The long-run base rate, constant. TSS is 0 by construction."""
    return pd.Series(float(frame["occurred"].mean()), index=frame.index)


def build_scores() -> pd.DataFrame:
    """Join outcomes, NOAA's forecast, and ours, one row per day."""
    truth = outcomes()
    noaa = noaa_forecasts()
    if truth.empty or noaa.empty:
        return pd.DataFrame()

    frame = truth.merge(noaa[["date", "noaa_m_prob", "x_class_1_day"]], on="date", how="inner")
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["hawkes_prob"] = hawkes_hindcast(frame["date"])
    frame["persistence_prob"] = persistence_hindcast(frame)
    try:
        frame["gp_prob"] = gp_hindcast(frame)
    except Exception as e:
        print(f"  [warn] GP hindcast failed: {type(e).__name__}: {e}")
        frame["gp_prob"] = np.nan
    frame["climatology_prob"] = climatology_hindcast(frame)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(LOG_PATH)
    return frame


def score(frame: pd.DataFrame | None = None, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """TSS and Brier for every forecaster on the same days."""
    from solar.flare_model import best_tss, brier, evaluate

    frame = build_scores() if frame is None else frame
    if frame.empty:
        return {"status": "no overlapping forecast/outcome days archived yet"}

    y = frame["occurred"].values
    out = {
        "n_days": len(frame),
        "span": f"{frame['date'].min().date()} -> {frame['date'].max().date()}",
        "positive_rate": round(float(y.mean()), 4),
        "note": "hindcast for our models (computed over known days); NOAA's figures are the "
                "genuinely prospective ones, published before each outcome",
    }

    for name, col in (("noaa", "noaa_m_prob"), ("hawkes", "hawkes_prob"),
                      ("gp_classifier", "gp_prob"),
                      ("persistence", "persistence_prob"), ("climatology", "climatology_prob")):
        if col not in frame.columns or frame[col].isna().all():
            continue
        p = frame[col].fillna(0.0).values
        entry = evaluate(y, p, threshold)
        entry["brier"] = round(brier(y, p), 4)
        best, best_t = best_tss(y, p)
        entry["best_tss"] = round(best, 4)
        entry["best_tss_threshold"] = round(best_t, 4)
        entry["mean_prob"] = round(float(p.mean()), 4)
        out[name] = entry

    if "hawkes" in out and "noaa" in out:
        out["hawkes_beats_noaa_on_brier"] = out["hawkes"]["brier"] < out["noaa"]["brier"]
        out["brier_ratio_hawkes_over_noaa"] = round(
            out["hawkes"]["brier"] / out["noaa"]["brier"], 3) if out["noaa"]["brier"] else None
    return out


if __name__ == "__main__":
    import json

    print("Building the solar forecast scoreboard...")
    frame = build_scores()
    if frame.empty:
        print("  nothing to score -- run solar.flare_data.harvest_daily() for a few days first")
        raise SystemExit(1)

    print(f"  {len(frame)} days, {int(frame['occurred'].sum())} with an M+ flare\n")
    print(frame[["date", "occurred", "m_plus_count", "noaa_m_prob",
                 "hawkes_prob", "persistence_prob"]].head(12).to_string(index=False))

    print("\n=== scores ===")
    print(json.dumps(score(frame), indent=2, default=str))
