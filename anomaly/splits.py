"""Held-out evaluation, because the transductive numbers were wrong.

THE PROBLEM THIS FIXES

Every score in the first grid was produced by `fit_score(windows, y)` --
fitted and scored on the same windows. That is the standard convention for
unsupervised anomaly detection and it is harmless for a detector with two
parameters: a z-score estimates a mean and a standard deviation, and cannot
meaningfully memorise 1,891 windows with them.

It is not harmless for a detector with hundreds. Measured on equities:

    one_class_svm    in-sample 0.1026 (4.62x)  ->  held out 0.0151 (1.07x)

The in-sample table also reported `fraction_of_ceiling = 1.724`, meaning the
best unsupervised detector beat a model that had been shown the labels. That
is impossible out-of-sample, and it is the tell that something is memorising.

So the gap between in-sample and held-out scales with model capacity. Both
numbers are reported: the difference between them is itself a result, and
arguably a more interesting one than the ranking.

HOW EACH DOMAIN IS SPLIT

    equity    TEMPORALLY. Fit on the earlier period, score the later one.
              A random split would let a detector learn from March 2020
              while being tested on February 2020, which is not a thing that
              can happen in deployment.

    kepler    BY TARGET. No star appears in both halves. Splitting windows
              randomly would put windows from the same transit -- overlapping
              by 18 of their 24 points -- on both sides, which is leakage
              wearing a disguise.
"""

from __future__ import annotations

import numpy as np

#: Fraction of the data used for fitting. 0.7 is conventional and leaves
#: enough events in the test half to be worth scoring: on equities that is
#: 34 train / 8 test, which is thin but honest.
TRAIN_FRACTION = 0.7


def temporal_split(n: int, train_fraction: float = TRAIN_FRACTION) -> tuple:
    """Earlier -> later. No shuffling, ever."""
    cut = int(n * train_fraction)
    train = np.zeros(n, dtype=bool)
    train[:cut] = True
    return train, ~train


def group_split(groups: np.ndarray, train_fraction: float = TRAIN_FRACTION,
                seed: int = 0, y: np.ndarray | None = None) -> tuple:
    """Split by series, so no series straddles the boundary.

    STRATIFIED BY PREVALENCE when labels are supplied, and that is not a
    refinement -- an unstratified split produced train at 17.9% positive and
    test at 42.8%. Kepler targets differ enormously in how much of their
    light curve is labelled transit: a short-period planet transits
    constantly (one target measured 99.9% positive windows) while a
    long-period one barely registers. Shuffled at random, a handful of the
    dense ones land together and skew the half they fall in.

    That breaks the comparison outright, because PR-AUC is floored at the
    base rate. A score of 0.50 at 42.8% prevalence is WORSE than 0.30 at
    24%, and comparing them as though they were the same quantity is
    meaningless. Stratifying keeps both halves near the same base rate so
    the numbers can be read side by side.

    Targets are still shuffled within each stratum, because the manifest is
    ordered by target id and id correlates with sky position.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)

    if y is None:
        shuffled = rng.permutation(unique)
        n_train = max(1, int(len(shuffled) * train_fraction))
        train_ids = set(shuffled[:n_train].tolist())
        return (lambda t: (t, ~t))(np.array([g in train_ids for g in groups]))

    # Order targets by their own prevalence, then deal them alternately into
    # train and test in blocks, so both halves span the full range.
    y = np.asarray(y).astype(int)
    prevalence = {g: float(y[groups == g].mean()) for g in unique}
    ordered = sorted(unique, key=lambda g: prevalence[g])

    train_ids, test_ids = [], []
    block = max(1, int(round(1 / (1 - train_fraction))))   # e.g. 0.7 -> every 3rd to test
    for i, g in enumerate(ordered):
        (test_ids if (i + 1) % block == 0 else train_ids).append(g)
    if not test_ids:                      # tiny sample -- give test at least one
        test_ids.append(train_ids.pop())

    train_set = set(train_ids)
    train = np.array([g in train_set for g in groups])
    return train, ~train


def make_split(ds, seed: int = 0) -> tuple:
    """Pick the right split for the dataset's structure.

    Labels are passed through to `group_split` so it can stratify. Without
    that the two halves ended up at 17.9% and 42.8% positive, which makes
    their PR-AUC values incomparable.
    """
    if ds.groups is not None:
        return group_split(ds.groups, seed=seed, y=ds.y)
    return temporal_split(len(ds.y))


def score_held_out(det, ds, train_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    """Fit on train, score test. Returns scores for the TEST rows only.

    With groups, each test series is scored by a detector fitted on the
    TRAINING series, then converted to within-series ranks -- one star's raw
    score is not comparable to another's, and the ranking metrics only need
    order.
    """
    if ds.groups is None:
        det.fit(ds.windows[train_mask], ds.y[train_mask])
        return det.score(ds.windows[test_mask])

    out = np.empty(int(test_mask.sum()), dtype=float)
    test_groups = ds.groups[test_mask]
    test_windows = ds.windows[test_mask]
    det.fit(ds.windows[train_mask], ds.y[train_mask])
    for gid in np.unique(test_groups):
        m = test_groups == gid
        scores = det.score(test_windows[m])
        order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
        out[m] = order / max(len(scores) - 1, 1)
    return out


def rank01(scores: np.ndarray) -> np.ndarray:
    """Scores as ranks in [0, 1].

    The only way to combine detectors: on one window Mahalanobis reports
    44.22 where IsolationForest reports 0.60, so averaging raw scores would
    simply return whichever detector has the largest units.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
    return order / max(len(scores) - 1, 1)


def select_ensemble(train_scores: dict, y_train: np.ndarray,
                    max_members: int = 4, metric: str = "pr_auc") -> list[str]:
    """Greedily build an ensemble, judged ONLY on the training half.

    Forward selection: start with the best single detector, then repeatedly
    add whichever remaining detector most improves the training score, and
    stop when nothing does.

    Greedy rather than exhaustive on purpose. Trying all 3,876 subsets of 20
    detectors and keeping the best is best-of-N selection, which is the exact
    thing that makes a metric uninterpretable -- and I did it that way once
    already and reported an inflated number. Greedy evaluates at most
    ~70 candidates and never looks at the test half.
    """
    from anomaly.metrics import evaluate_scores

    names = [k for k, v in train_scores.items()
             if np.isfinite(v).all() and np.ptp(v) > 0]
    if not names:
        return []

    def score_of(members):
        combined = np.mean([train_scores[m] for m in members], axis=0)
        value = evaluate_scores(y_train, combined).get(metric, np.nan)
        return value if np.isfinite(value) else -np.inf

    chosen = [max(names, key=lambda n: score_of([n]))]
    best = score_of(chosen)
    while len(chosen) < max_members:
        candidates = [(score_of(chosen + [n]), n) for n in names if n not in chosen]
        if not candidates:
            break
        value, name = max(candidates)
        if value <= best:
            break
        chosen.append(name)
        best = value
    return chosen
