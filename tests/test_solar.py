"""Tests for the solar / space-weather domain.

Split deliberately into two kinds. The parser tests are offline and exact --
they run on literal sample rows copied from the real products, so they catch
a regression without needing the network and without depending on whatever
the Sun happens to be doing today. The live tests hit NOAA and assert only
on structure and plausible ranges, because the actual values change hourly
and a test that asserts "there were 88 flares" would fail tomorrow by design.

Run with: pytest tests/test_solar.py -v
"""

import pytest


# --------------------------------------------------------------------------
# offline parser tests -- exact, no network
# --------------------------------------------------------------------------

def test_flare_class_to_flux():
    """Class strings are a log scale, and X1 must come out as ten times M1 --
    if this is wrong every model downstream silently mis-weights severity."""
    from solar.flare_data import parse_flare_class

    assert parse_flare_class("M1.0") == ("M", pytest.approx(1e-5))
    assert parse_flare_class("X1.0") == ("X", pytest.approx(1e-4))
    assert parse_flare_class("M5.0")[1] == pytest.approx(5e-5)
    assert parse_flare_class("C")[1] == pytest.approx(1e-6)   # bare letter -> 1.0
    assert parse_flare_class("x2.3")[1] == pytest.approx(2.3e-4)  # case-insensitive

    # missing/garbage must degrade to (None, None), not raise -- NOAA leaves
    # these blank often enough that raising would discard good rows
    for bad in (None, "", "garbage", 42):
        assert parse_flare_class(bad) == (None, None)


def test_class_at_least_is_a_flux_comparison():
    """Severity must be compared as flux, not as text. The letters happen to
    sort correctly (A<B<C<M<X), which is exactly what makes string comparison
    a tempting shortcut -- but the multiplier breaks it, so the shortcut fails
    on precisely the largest and most consequential events."""
    from solar.flare_data import class_at_least, parse_flare_class

    assert class_at_least("M1.0") is True
    assert class_at_least("X9.9") is True
    assert class_at_least("C9.9") is False
    assert class_at_least("B1.0") is False
    assert class_at_least(None) is False

    # the real ordering trap: multi-digit multipliers. X10 is ten times X1,
    # but sorts BELOW X9.9 as text. Real events live here -- the Nov 2003
    # flare was ~X28.
    assert "X10.0" < "X9.9"                                  # text order: wrong
    assert parse_flare_class("X10.0")[1] > parse_flare_class("X9.9")[1]
    assert parse_flare_class("X28.0")[1] == pytest.approx(2.8e-3)

    # and the boundary that defines the forecasting target: M9.9 is the last
    # sub-X value, and C9.9 sits just under the M threshold
    assert class_at_least("M9.9") is True
    assert class_at_least("C9.9") is False
    assert parse_flare_class("C9.9")[1] < parse_flare_class("M1.0")[1]


def test_hesperia_parser_handles_midnight_rollover():
    """Hesperia records only times of day, so a flare beginning 23:50 and
    ending 00:14 must roll to the next date -- otherwise its duration comes
    out as roughly minus 23 hours."""
    from solar.flare_data import parse_hesperia_year

    sample = (
        "GOES XRAY events for 2020\n"
        "10-Jan-2020 23:50   23:58   00:14   M1.1   N22E30   12756\n"
    )
    df = parse_hesperia_year(sample)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["begin_time"].day == 10
    assert row["end_time"].day == 11          # rolled forward
    assert row["duration_min"] == pytest.approx(24.0)
    assert row["duration_min"] > 0


def test_hesperia_parser_separates_position_from_active_region():
    """Both trailing fields are optional and a heliographic position contains
    digits ('N22E30'), so a naive number-grab would read '22' or '30' as the
    active region id."""
    from solar.flare_data import parse_hesperia_year

    sample = (
        "10-Jan-2020 11:52   11:59   12:24   M1.1   N22E30   12756\n"   # both
        "11-Jan-2020 11:52   11:59   12:24   X2.3      12757\n"          # AR only
        "12-Jan-2020 01:00   01:05   01:09   A1.0   S10W56\n"            # position only
        "13-Jan-2020 01:00   01:05   01:09   B1.0\n"                     # neither
    )
    df = parse_hesperia_year(sample)

    assert len(df) == 4
    assert list(df["position"]) == ["N22E30", None, "S10W56", None]
    assert df.iloc[0]["active_region"] == 12756
    assert df.iloc[1]["active_region"] == 12757
    assert df.iloc[2]["active_region"] is None or pd_isna(df.iloc[2]["active_region"])
    assert list(df["class_letter"]) == ["M", "X", "A", "B"]


def pd_isna(v):
    import pandas as pd
    return pd.isna(v)


def test_daily_indices_parser():
    """Real header + data rows from the DSD product, including its two
    different missing-value markers ('-999' and '*')."""
    from solar.flare_data import parse_daily_indices

    sample = (
        ":Product: Daily Solar Data            DSD.txt\n"
        "# comment line\n"
        "#  Date     10.7cm Number  Hemis. Regions Field  Flux   C  M  X  S  1  2  3\n"
        "2026 08 19  126     78      560      1    -999      *  18  1  0  5  0  0  0\n"
        "2026 08 20  125     72      530      1    -999      *  16  3  0  8  1  0  0\n"
    )
    df = parse_daily_indices(sample)

    assert len(df) == 2
    assert df.iloc[0]["radio_flux"] == 126
    assert df.iloc[0]["sunspot_number"] == 78
    assert df.iloc[0]["sunspot_area"] == 560
    assert df.iloc[1]["m_flares"] == 3
    assert df.iloc[1]["c_flares"] == 16
    # both missing-value markers must become None, not sentinel numbers that
    # would poison a model as if -999 were a real field strength
    assert df.iloc[0]["stanford_mean_field"] is None
    assert df.iloc[0]["xray_background"] is None


def test_solar_region_parser_and_delta_flag():
    """Magnetic complexity is the strongest single flare predictor available,
    so the delta flag and the complexity ordinal are worth pinning down."""
    from solar.flare_data import parse_solar_regions

    sample = (
        "# Region   Location         Sunspot Characteristics\n"
        " 4506    N12W0*    216     30     7     DAO    4      B\n"
        " 4507    N04W82    198    300    11     EHI   13     BG\n"
        " 4599    S10W56    172     20     4     CRO    5    BGD\n"
    )
    df = parse_solar_regions(sample)

    assert len(df) == 3
    assert list(df["region"]) == [4506, 4507, 4599]
    assert df.iloc[0]["location"] == "N12W0"      # trailing '*' stripped
    assert df.iloc[1]["mcintosh_class"] == "EHI"
    assert df.iloc[1]["area"] == 300

    # beta < beta-gamma < beta-gamma-delta, and only the last has a delta
    assert list(df["mag_complexity"]) == [2, 3, 5]
    assert list(df["has_delta"]) == [False, False, True]


# --------------------------------------------------------------------------
# live NOAA tests -- structure and plausible ranges only
# --------------------------------------------------------------------------

def test_live_noaa_probabilities_are_the_baseline():
    """The project is scored against NOAA's own forecast, so this endpoint
    working is load-bearing for the whole evaluation story."""
    from solar.flare_data import fetch_noaa_probabilities

    df = fetch_noaa_probabilities()
    assert len(df) > 0
    for col in ("date", "c_class_1_day", "m_class_1_day", "x_class_1_day"):
        assert col in df.columns
    # they are percentages
    assert df["m_class_1_day"].between(0, 100).all()
    assert df["x_class_1_day"].between(0, 100).all()


def test_live_recent_flares():
    from solar.flare_data import fetch_recent_flares

    df = fetch_recent_flares()
    assert len(df) > 0
    assert {"begin_time", "max_class", "class_letter", "peak_flux"} <= set(df.columns)
    assert df["class_letter"].dropna().isin(list("ABCMX")).all()
    assert df["begin_time"].is_monotonic_increasing


def test_live_daily_indices():
    from solar.flare_data import fetch_daily_indices

    df = fetch_daily_indices()
    assert len(df) > 0
    # F10.7 has a hard physical floor around 65 sfu and never approaches 1000
    flux = df["radio_flux"].dropna()
    assert flux.between(60, 400).all()


@pytest.mark.slow
def test_historical_year_matches_known_totals():
    """2014 was near solar maximum. Pinning the real totals catches a parser
    regression that silently drops or duplicates rows -- a class of bug that
    would otherwise show up only as mysteriously worse model scores."""
    from solar.flare_data import class_at_least, fetch_historical_year

    df = fetch_historical_year(2014)
    counts = df["class_letter"].value_counts().to_dict()

    assert len(df) == 2260
    assert counts["C"] == 1797
    assert counts["M"] == 207
    assert counts["X"] == 16
    assert df["duration_min"].min() >= 0          # no negative-duration rows
    assert df["begin_time"].is_monotonic_increasing
    assert df["flare_class"].apply(class_at_least).sum() == 223


@pytest.mark.slow
def test_harvest_is_idempotent():
    """The harvester runs daily and must tolerate double-firing: a second run
    in the same window should add nothing rather than duplicating the archive."""
    from solar.flare_data import harvest_daily

    harvest_daily(verbose=False)
    second = harvest_daily(verbose=False)
    assert all(added == 0 for added in second.values()), second
