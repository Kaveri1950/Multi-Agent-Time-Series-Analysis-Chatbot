"""Ground truth for both domains. The only module allowed to know about domains.

THE RULE THIS MODULE EXISTS TO ENFORCE: a label may never be derived from the
series being scored.

That sounds obvious and it is very easy to violate. This project already did
once -- astronomy windows were labelled "contains a transit" by thresholding
the window's own DEPTH, then scored with detectors that look for deep dips.
The detectors were being graded against their own inputs, and measured label
inflation came to +0.11 AUC.

The same trap in finance is labelling a day anomalous because the return was
large, then reporting that a return-based detector finds it. The same trap in
solar is labelling flares by X-ray threshold -- which is how GOES flare
classes are DEFINED -- then detecting them from X-ray flux. That is why solar
is not in this comparison.

So both label sets below come from outside their series:

  astronomy   NASA ephemerides (period, epoch, duration) from the Exoplanet
              Archive. Where a transit falls is predicted by orbital
              mechanics, not read off the brightness we score.

  finance     A calendar of events that happened for reasons of their own --
              budget sessions, election results, documented shocks. Their
              dates exist whether or not the market moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketEvent:
    date: str
    name: str
    kind: str          # "scheduled" | "shock"
    confidence: str    # "certain" | "approximate"


# ---------------------------------------------------------------------------
# PRE-REGISTERED finance event list.
#
# Committed before any detector was run against it, and deliberately short.
# A longer list assembled by scanning for big moves would be circular, and a
# longer list assembled from memory would be wrong -- both are worse than a
# small list that is defensible line by line.
#
# "approximate" marks events whose market impact spanned several sessions
# rather than one; `event_labels` widens those by `window_days`.
#
# KNOWN LIMITATION, to be stated in any write-up: this list is incomplete.
# Many real shocks are missing, and some listed events moved the market
# barely at all. So RECALL IS BOUNDED BELOW ITS TRUE VALUE by construction --
# a detector penalised for flagging a genuine event absent from this list is
# being penalised for being right. Precision-style metrics read more
# honestly here than recall-style ones, which is a further reason the
# reporting leans on PR-AUC and lift rather than raw recall.
# ---------------------------------------------------------------------------
MARKET_EVENTS: tuple[MarketEvent, ...] = (
    MarketEvent("2019-05-23", "Lok Sabha election results", "scheduled", "certain"),
    MarketEvent("2019-07-05", "Union Budget 2019-20 (full)", "scheduled", "certain"),
    MarketEvent("2019-09-20", "Corporate tax rate cut announcement", "shock", "certain"),
    MarketEvent("2020-03-06", "Yes Bank moratorium", "shock", "certain"),
    MarketEvent("2020-03-09", "COVID crash / oil price war", "shock", "certain"),
    MarketEvent("2020-03-12", "COVID selloff, circuit breaker", "shock", "certain"),
    MarketEvent("2020-03-23", "COVID trough, nationwide lockdown announced", "shock", "certain"),
    MarketEvent("2021-02-01", "Union Budget 2021-22", "scheduled", "certain"),
    MarketEvent("2022-02-24", "Russian invasion of Ukraine", "shock", "certain"),
    MarketEvent("2023-01-25", "Hindenburg report on Adani Group", "shock", "certain"),
    MarketEvent("2023-02-01", "Adani Enterprises FPO withdrawn", "shock", "certain"),
    MarketEvent("2023-03-13", "SVB collapse, global banking stress", "shock", "certain"),
    MarketEvent("2024-06-04", "Lok Sabha election results", "scheduled", "certain"),
    MarketEvent("2024-07-23", "Union Budget 2024-25 (full)", "scheduled", "certain"),
    MarketEvent("2025-04-07", "Global tariff announcement selloff", "shock", "approximate"),
)

#: Union Budget is presented on 1 February. Derivable from the calendar
#: rather than recalled, so it is generated rather than listed -- and snapped
#: to the next trading session, which matters because 1 Feb 2020 and 2025
#: fell on Saturdays (NSE ran special sessions, but the panel may not hold
#: them).
BUDGET_MONTH_DAY = (2, 1)


def budget_days(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Union Budget sessions falling inside `index`, snapped to trading days."""
    out = []
    for year in sorted({d.year for d in index}):
        target = pd.Timestamp(year=year, month=BUDGET_MONTH_DAY[0], day=BUDGET_MONTH_DAY[1])
        later = index[index >= target]
        if len(later) and (later[0] - target).days <= 4:
            out.append(later[0])
    return out


def event_dates(index: pd.DatetimeIndex, include_budgets: bool = True,
                window_days: int = 1) -> pd.DatetimeIndex:
    """All pre-registered event dates that exist in `index`.

    `window_days` widens each event, because a shock's market impact does not
    always land on the calendar day it is dated -- Hindenburg published after
    Indian market hours, so the reaction is the following session.
    """
    stamps: set[pd.Timestamp] = set()
    for ev in MARKET_EVENTS:
        target = pd.Timestamp(ev.date)
        span = window_days if ev.confidence == "certain" else window_days + 2
        near = index[(index >= target - pd.Timedelta(days=1)) &
                     (index <= target + pd.Timedelta(days=span))]
        stamps.update(near)
    if include_budgets:
        stamps.update(budget_days(index))
    return pd.DatetimeIndex(sorted(stamps))


def finance_day_labels(index: pd.DatetimeIndex, **kwargs) -> np.ndarray:
    """Binary label per trading day: was this an event day?"""
    events = event_dates(index, **kwargs)
    return index.isin(events).astype(int)


def windows_from_day_labels(day_labels: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Lift day-level labels to window level: positive if it contains an event.

    The same convention astronomy uses for transits, so the two domains'
    numbers mean the same thing and the comparison is not quietly between
    two different tasks.
    """
    day_labels = np.asarray(day_labels).astype(int).ravel()
    n_windows = 1 + (len(day_labels) - window) // stride
    return np.array([
        int(day_labels[i * stride: i * stride + window].any())
        for i in range(n_windows)
    ])


def astronomy_window_labels(times: np.ndarray, kepid: int, window: int, stride: int,
                            koi: pd.DataFrame | None = None) -> np.ndarray:
    """Transit labels from NASA ephemerides, via the existing KOI catalogue.

    Delegates rather than reimplementing so the evaluation and the serving
    path cannot drift apart -- if these disagreed, the reported numbers would
    describe something other than what the chatbot runs.
    """
    from astronomy.koi_catalog import label_windows
    return np.asarray(label_windows(times, kepid, window, stride, koi)).astype(int)


def describe_event_set(index: pd.DatetimeIndex) -> dict:
    """Provenance summary for the write-up."""
    labels = finance_day_labels(index)
    return {
        "n_trading_days": len(index),
        "n_event_days": int(labels.sum()),
        "prevalence": round(float(labels.mean()), 4),
        "n_registered_events": len(MARKET_EVENTS),
        "n_scheduled": sum(1 for e in MARKET_EVENTS if e.kind == "scheduled"),
        "n_shock": sum(1 for e in MARKET_EVENTS if e.kind == "shock"),
        "n_budget_days": len(budget_days(index)),
        "first": str(index[0].date()), "last": str(index[-1].date()),
    }
