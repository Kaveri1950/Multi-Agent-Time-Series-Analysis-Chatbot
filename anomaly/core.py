"""The scoring contract every detector obeys.

    windows: (n_windows, window_size) float array
        -> scores: (n_windows,) float array, higher = more anomalous

That is the whole interface. A detector gets a matrix of windows and returns
one number per window. It is not told what instrument produced them, what
the units are, or what "normal" looks like -- it has to work that out from
the data it is fitted on.

Keeping the contract this narrow is what makes the comparison honest. Any
detector that needed extra domain context would be solving an easier problem
than the others, and its column in the results table would not be comparable.

WHY WINDOWS AND NOT POINTS
Some detectors are naturally point-level (a z-score applies to one
observation) and some are naturally window-level (an autoencoder reconstructs
a segment). Scoring everything at window level puts them on the same footing.
Point-level detectors aggregate within the window by taking their maximum,
which is the standard choice: a window is as anomalous as its worst moment.

SIGN CONVENTION
Features are built so that both directions of deviation are visible
separately -- `dev_below` and `dev_above` rather than a single "depth". A
planetary transit is a dip, a solar or jet flare is a spike, and a market
shock can be either. A feature set that only measures dips would quietly
encode the astronomy problem into what is supposed to be a neutral core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Detector(ABC):
    """Base class. Subclasses implement `_fit` and `_score`.

    `fit` and `score` are separate so a detector can be fitted on one split
    and scored on another. Most of these are unsupervised and transductive --
    fitted and scored on the same windows -- which `fit_score` expresses.
    """

    name: str = "unnamed"
    group: str = "ungrouped"
    #: Set on detectors that exist only to be beaten. If a real detector
    #: cannot beat these, it has not learned anything -- the check proposed
    #: by Kim et al. (AAAI 2022), who found untrained models matching
    #: published methods on standard benchmarks.
    is_floor: bool = False
    #: Set on the supervised ceiling, which is allowed to see labels and so
    #: is not a competitor -- it reports what is achievable at all.
    is_ceiling: bool = False

    def fit(self, windows: np.ndarray, y: np.ndarray | None = None) -> "Detector":
        self._fit(np.asarray(windows, dtype=float), y)
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        scores = self._score(np.asarray(windows, dtype=float))
        scores = np.asarray(scores, dtype=float).ravel()
        # A detector that emits NaN would silently poison the ranking
        # metrics, so failures become "least anomalous" rather than errors.
        return np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max,
                             neginf=-np.finfo(float).max)

    def fit_score(self, windows: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(windows, y).score(windows)

    @abstractmethod
    def _fit(self, windows: np.ndarray, y: np.ndarray | None) -> None: ...

    @abstractmethod
    def _score(self, windows: np.ndarray) -> np.ndarray: ...

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"<{self.__class__.__name__} name={self.name!r} group={self.group!r}>"


def windowize(series: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Overlapping windows as a (n_windows, window) array.

    Uses a strided view rather than a Python loop -- a 1,900-day equity series
    at stride 1 is cheap either way, but a Kepler light curve can run to
    65,000 points and the loop version dominates the runtime of the whole grid.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    if window <= 0 or stride <= 0:
        raise ValueError(f"window and stride must be positive, got {window}, {stride}")
    if n < window:
        raise ValueError(f"series of length {n} is shorter than window {window}")
    n_windows = 1 + (n - window) // stride
    itemsize = series.strides[0]
    return np.lib.stride_tricks.as_strided(
        series, shape=(n_windows, window), strides=(stride * itemsize, itemsize),
        writeable=False,
    ).copy()


def window_start_indices(n_points: int, window: int, stride: int) -> np.ndarray:
    """Index into the original series of each window's first point.

    Needed to map window-level scores back onto dates or timestamps, which is
    how window scores become target-level or day-level predictions.
    """
    n_windows = 1 + (n_points - window) // stride
    return np.arange(n_windows) * stride


def window_features(windows: np.ndarray) -> pd.DataFrame:
    """Domain-neutral summary statistics, one row per window.

    Deliberately excludes anything that assumes a baseline value. The
    astronomy code had a `depth = 1 - min` feature, which silently encodes
    "flux is normalised to 1.0 and anomalies are dips" -- true for Kepler,
    meaningless for returns, and fatal to a generality claim.
    """
    windows = np.asarray(windows, dtype=float)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    mn = windows.min(axis=1)
    mx = windows.max(axis=1)
    med = np.median(windows, axis=1)
    diffs = np.diff(windows, axis=1)
    frame = pd.DataFrame({
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "range": mx - mn,
        "median": med,
        # Both directions kept separate, so dips and spikes are equally visible.
        "dev_below": med - mn,
        "dev_above": mx - med,
        "skew": pd.DataFrame(windows.T).skew().to_numpy(),
        "kurtosis": pd.DataFrame(windows.T).kurtosis().to_numpy(),
        "mean_abs_diff": np.abs(diffs).mean(axis=1),
        "max_abs_diff": np.abs(diffs).max(axis=1),
        # Crossings of the window's own median: a cheap proxy for whether the
        # window is oscillating or has made a one-way excursion.
        "median_crossings": (np.diff(np.signbit(windows - med[:, None]), axis=1) != 0).sum(axis=1),
    })
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def standardise(train: np.ndarray, apply_to: np.ndarray | None = None) -> np.ndarray:
    """Z-scale columns using statistics from `train`.

    Distance-based detectors are scale-sensitive, and these features are on
    wildly different scales (a kurtosis of 40 next to a std of 0.0002). Left
    unscaled, whichever feature happens to have the largest units dominates.
    """
    train = np.asarray(train, dtype=float)
    mu = train.mean(axis=0)
    sigma = train.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    target = train if apply_to is None else np.asarray(apply_to, dtype=float)
    return (target - mu) / sigma


def series_from_windows(windows: np.ndarray, stride: int) -> np.ndarray:
    """Rebuild the original series from its overlapping windows.

    Needed by detectors that cannot work window-by-window. The matrix profile
    compares every subsequence against every other, which requires the
    contiguous series -- handing it a pile of pre-cut windows would lose the
    adjacency it depends on.

    Exact whenever stride <= window: consecutive windows overlap by
    (window - stride) points, so taking the first `stride` points of each
    window and the whole of the last one reconstructs the original exactly.
    """
    windows = np.asarray(windows, dtype=float)
    if stride >= windows.shape[1]:
        return windows.reshape(-1)
    head = windows[:-1, :stride].reshape(-1)
    return np.concatenate([head, windows[-1]])
