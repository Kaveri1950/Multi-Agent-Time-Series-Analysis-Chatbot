"""Flare catalogue -> daily feature rows with a next-24h label.

The forecasting target is deliberately the same one NOAA publishes, so the
project can be scored against a real operational product rather than a
strawman: **will an M-class-or-larger flare occur in the next 24 hours?**
A probability, not a timestamp -- individual flare timing is not predictable
and nobody in the field claims otherwise.

LEAKAGE IS THE WHOLE DESIGN PROBLEM HERE. Every feature for day D is
computed from flares that began strictly BEFORE D 00:00 UTC, and the label
is read from flares inside D. If a single feature is allowed to see its own
day's flares, the model scores beautifully and means nothing -- and this is
easy to do by accident with rolling windows, which by default include the
current row. build_daily_frame() therefore shifts every window explicitly
and test_solar.py pins the behaviour with a synthetic case where a leak
would be obvious.

Feature availability is uneven and the report should say so plainly:
magnetic complexity (Mount Wilson / McIntosh class) is the strongest known
flare predictor, but NOAA's Solar Region Summary endpoint serves TODAY only
and no historical archive for it has been verified. So the features below
are built purely from the flare catalogue -- which does span 1975-2020 --
and magnetic complexity is available at live-forecast time but not for
training. That asymmetry is a real limitation, not a modelling choice.
"""

import numpy as np
import pandas as pd

from solar.flare_data import CLASS_BASE, class_at_least

# Windows in days. 27 is not arbitrary: the Sun rotates once per ~27 days,
# so a flare-productive active region that leaves the visible disc comes
# back around. Recurrence at the Carrington period is real physical signal
# that a naive 1/3/7-day feature set misses entirely.
WINDOWS = (1, 3, 7, 27)

CARRINGTON_DAYS = 27

# Cap on "hours since last X-class". X-class flares can be absent for years
# at solar minimum, and an uncapped value lets that dominate every tree
# split. Capped at 60 days, i.e. "long enough ago to not matter".
MAX_QUIET_HOURS = 60 * 24


def _fluence(sub: pd.DataFrame) -> float:
    """Rough radiated-energy proxy: peak flux x duration, summed. Two C1
    flares and one C2 flare are not the same thing to an active region's
    energy budget, and a plain event count cannot tell them apart."""
    if sub.empty:
        return 0.0
    return float((sub["peak_flux"].fillna(0.0) * sub["duration_min"].fillna(0.0)).sum())


def build_daily_frame(catalog: pd.DataFrame, threshold: str = "M") -> pd.DataFrame:
    """One row per calendar day: history-only features + the next-24h label.

    `catalog` is a flare catalogue as returned by
    solar.flare_data.load_historical_catalog() or fetch_recent_flares().
    Returns a frame indexed 0..N with a 'date' column, sorted ascending.
    """
    if catalog.empty:
        return pd.DataFrame()

    cat = catalog.copy()
    # normalise away timezone so historical (naive) and live (UTC-aware)
    # catalogues can both be fed through this same function
    begin = pd.to_datetime(cat["begin_time"], utc=True, errors="coerce").dt.tz_localize(None)
    cat["begin_time"] = begin
    cat = cat.dropna(subset=["begin_time"]).sort_values("begin_time").reset_index(drop=True)
    cat["day"] = cat["begin_time"].dt.floor("D")
    cat["is_target"] = cat["flare_class"].apply(lambda c: class_at_least(c, threshold))

    # Daily aggregates first, then every feature is built by shifting these
    # so that day D only ever sees strictly-earlier days.
    days = pd.date_range(cat["day"].min(), cat["day"].max(), freq="D")
    daily = pd.DataFrame({"date": days}).set_index("date")

    for letter in ("B", "C", "M", "X"):
        counts = cat[cat["class_letter"] == letter].groupby("day").size()
        daily[f"n_{letter}"] = counts.reindex(days, fill_value=0).astype(float)

    daily["n_all"] = cat.groupby("day").size().reindex(days, fill_value=0).astype(float)
    daily["n_target"] = cat[cat["is_target"]].groupby("day").size().reindex(days, fill_value=0).astype(float)
    daily["max_flux"] = cat.groupby("day")["peak_flux"].max().reindex(days).fillna(0.0)
    daily["fluence"] = (
        cat.assign(f=cat["peak_flux"].fillna(0.0) * cat["duration_min"].fillna(0.0))
        .groupby("day")["f"].sum().reindex(days, fill_value=0.0)
    )
    daily["n_regions"] = cat.groupby("day")["active_region"].nunique().reindex(days, fill_value=0).astype(float)

    # ---- the label: did a target-class flare occur ON this day ----
    # This is what the model predicts for "the next 24 hours", so a row's
    # features must come from before the row's own day. Handled below.
    out = pd.DataFrame(index=days)
    out["label"] = (daily["n_target"] > 0).astype(int)

    # ---- history features: shift(1) FIRST, then roll ----
    # shift(1) is the leakage guard. Rolling without it would include the
    # current day, i.e. the answer.
    past = daily.shift(1)

    for w in WINDOWS:
        roll = past.rolling(w, min_periods=1)
        out[f"n_all_{w}d"] = roll["n_all"].sum()
        out[f"n_C_{w}d"] = roll["n_C"].sum()
        out[f"n_M_{w}d"] = roll["n_M"].sum()
        out[f"n_X_{w}d"] = roll["n_X"].sum()
        out[f"n_target_{w}d"] = roll["n_target"].sum()
        out[f"fluence_{w}d"] = roll["fluence"].sum()
        out[f"max_flux_{w}d"] = roll["max_flux"].max()
        out[f"n_regions_{w}d"] = roll["n_regions"].max()

    # rate acceleration: is activity ramping up or winding down? Same count
    # over 24h means something different depending on the day before.
    out["accel_1d"] = out["n_all_1d"] / out["n_all_3d"].replace(0, np.nan) * 3.0
    out["accel_7d"] = out["n_all_7d"] / out["n_all_27d"].replace(0, np.nan) * (27.0 / 7.0)

    # Carrington recurrence: what happened one solar rotation ago, when the
    # same longitudes were facing us.
    out["n_all_lag27"] = past["n_all"].shift(CARRINGTON_DAYS - 1).fillna(0.0)
    out["n_target_lag27"] = past["n_target"].shift(CARRINGTON_DAYS - 1).fillna(0.0)

    # ---- time since last event of each class ----
    for letter, col in (("C", "n_C"), ("M", "n_M"), ("X", "n_X")):
        occurred = past[col].fillna(0) > 0
        idx = np.arange(len(past))
        last_seen = np.where(occurred, idx, -1)
        last_seen = pd.Series(last_seen, index=past.index).replace(-1, np.nan).ffill()
        gap_days = pd.Series(idx, index=past.index) - last_seen
        out[f"hours_since_{letter}"] = (gap_days * 24.0).fillna(MAX_QUIET_HOURS).clip(upper=MAX_QUIET_HOURS)

    out = out.fillna(0.0)
    out.index.name = "date"
    return out.reset_index()


def feature_columns(frame: pd.DataFrame) -> list:
    """Every column that is a model input -- i.e. not the date or the label."""
    return [c for c in frame.columns if c not in ("date", "label")]


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Train/test split by TIME, never at random.

    Random k-fold on time series leaks the future into the past and produces
    beautiful, meaningless scores -- the single most common way flare
    forecasting papers overstate their results. The split point is returned
    so the report can state exactly which dates were held out.
    """
    n_test = max(1, int(len(frame) * test_fraction))
    train = frame.iloc[:-n_test].reset_index(drop=True)
    test = frame.iloc[-n_test:].reset_index(drop=True)
    return train, test


def describe_frame(frame: pd.DataFrame) -> dict:
    """Summary for the report / dashboard: size, span, class balance."""
    if frame.empty:
        return {}
    return {
        "n_days": len(frame),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "n_positive": int(frame["label"].sum()),
        "positive_rate": round(float(frame["label"].mean()), 4),
        "n_features": len(feature_columns(frame)),
    }


if __name__ == "__main__":
    from solar.flare_data import load_historical_catalog

    print("Building daily feature frame from the historical catalogue...")
    catalog = load_historical_catalog(1996, 2020)
    frame = build_daily_frame(catalog)

    info = describe_frame(frame)
    print(f"\n  {info['n_days']} days  {info['start']} -> {info['end']}")
    print(f"  {info['n_positive']} positive ({info['positive_rate']:.1%})  "
          f"{info['n_features']} features")

    train, test = chronological_split(frame)
    print(f"\n  train {len(train)} days ({train['date'].min().date()} -> {train['date'].max().date()})"
          f"  positive {train['label'].mean():.1%}")
    print(f"  test  {len(test)} days ({test['date'].min().date()} -> {test['date'].max().date()})"
          f"  positive {test['label'].mean():.1%}")

    print("\n  feature columns:")
    cols = feature_columns(frame)
    for i in range(0, len(cols), 4):
        print("    " + "  ".join(f"{c:<20}" for c in cols[i:i + 4]))
