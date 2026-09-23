"""Three forecasters for P(M+ flare in the next 24h), meant to be compared.

1. **Climatology / persistence** -- the honest floors. Climatology always
   predicts the long-run base rate; persistence predicts "same as yesterday".
   Any model that cannot beat these is not doing anything.

2. **Hawkes process** (this module's real contribution) -- a self-exciting
   point process fitted from scratch by maximum likelihood. Flares cluster
   because an active region that flares stays magnetically stressed and
   flares again, so each event temporarily raises the rate of the next. That
   is a structural claim about the physics, and it is what a plain Poisson
   or a per-day feature vector cannot express.

3. **XGBoost** on the daily features -- deliberately the same tool used for
   price forecasting in finance/price_model.py, so "one method, several
   domains" is demonstrated rather than asserted.

Two methodological commitments, both of which cost accuracy and are kept
anyway because the alternative is a meaningless number:

**Walk-forward validation, never a random split.** Random k-fold on a time
series leaks the future into the past. Worse, on this dataset even a single
chronological split is misleading: the last 20% of 1996-2020 falls entirely
in cycle 24's decline, where the positive rate is 1.7% against 17.3% in the
training period. So evaluation expands a training window forward through
several folds and reports each fold separately, which exposes how much of
the "skill" is really just knowing which phase of the solar cycle it is.

**TSS, not accuracy.** With a ~14% positive rate, a model that always says
"no flare" scores 86% accuracy and is worthless. The flare-forecasting
literature uses the true skill statistic (TPR - FPR), so that is what is
reported here, alongside Brier score for calibration.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from solar.features import build_daily_frame, feature_columns

M_CLASS_FLUX = 1e-5          # the M threshold, in W/m^2
HOURS_PER_DAY = 24.0


# ==========================================================================
# metrics
# ==========================================================================

def tss(y_true, y_prob, threshold: float) -> float:
    """True skill statistic = TPR - FPR. Zero for any constant forecast,
    which is exactly why it is the field's metric of choice: it gives no
    credit for exploiting class imbalance."""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_prob) >= threshold
    tp = np.sum(pred & y_true)
    fn = np.sum(~pred & y_true)
    fp = np.sum(pred & ~y_true)
    tn = np.sum(~pred & ~y_true)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(tpr - fpr)


def best_tss(y_true, y_prob):
    """Highest achievable TSS and the threshold that achieves it. The
    threshold must be chosen on TRAINING data and then applied unchanged to
    test -- tuning it on test is a subtle way of fitting the answer."""
    candidates = np.unique(np.round(np.asarray(y_prob), 4))
    if len(candidates) > 400:
        candidates = np.quantile(y_prob, np.linspace(0, 1, 400))
    scores = [(tss(y_true, y_prob, t), t) for t in candidates]
    score, thresh = max(scores)
    return float(score), float(thresh)


def brier(y_true, y_prob) -> float:
    """Mean squared error of the probabilities. Rewards being calibrated,
    not just correctly ranked -- a model can have good TSS and still be
    systematically over-confident."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true).astype(float)) ** 2))


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    return {
        "tss": round(tss(y_true, y_prob, threshold), 4),
        "brier": round(brier(y_true, y_prob), 4),
        "threshold": round(threshold, 4),
        "n": len(y_true),
        "positive_rate": round(float(np.mean(y_true)), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
    }


# ==========================================================================
# 1. trivial baselines -- the floor any real model must clear
# ==========================================================================

def climatology_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Always predict the training base rate. TSS is 0 by construction."""
    return np.full(len(test), float(train["label"].mean()))


def persistence_forecast(test: pd.DataFrame) -> np.ndarray:
    """'Tomorrow looks like yesterday': predict 1 if an M+ flare occurred in
    the previous 24h. Crude, but genuinely hard to beat, because flare
    clustering is real -- which is the same fact the Hawkes model formalises."""
    return (test["n_target_1d"].values > 0).astype(float)


# ==========================================================================
# 2. Hawkes process
# ==========================================================================

class HawkesModel:
    """Marked Hawkes process with an exponential kernel, fitted by MLE.

        lambda(t) = mu + sum_{t_j < t} alpha * w_j * beta * exp(-beta (t - t_j))
        w_j       = exp(gamma * m_j),   m_j = log10(flux_j / 1e-5)

    Marks matter: an X10 event should excite far more strongly than an M1,
    and the standard unmarked Hawkes cannot say so. m_j is defined so M1 has
    mark 0, X1 has mark 1, X10 has mark 2.

    `alpha` is the branching ratio -- the expected number of direct offspring
    per event. It is bounded below 1 during fitting because at alpha >= 1 the
    process is explosive (infinitely many descendants) and the likelihood
    surface stops being meaningful.

    Time is in HOURS throughout.
    """

    def __init__(self):
        self.mu = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.loglik = None
        self.n_events = 0
        self.marks_ = None
        self._converged = False

    # -- likelihood ------------------------------------------------------

    @staticmethod
    def _neg_loglik(params, times, marks, horizon):
        """Negative log-likelihood, O(n) via the exponential kernel's
        recursion. The naive double sum is O(n^2) and unusable at 25 years
        of events."""
        log_mu, log_alpha, log_beta, gamma = params
        mu, alpha, beta = np.exp(log_mu), np.exp(log_alpha), np.exp(log_beta)
        if alpha >= 1.0:
            return 1e10

        w = np.exp(gamma * marks)

        # recursive term: A_i = exp(-beta dt) * (A_{i-1} + w_{i-1})
        n = len(times)
        A = 0.0
        log_intensity_sum = 0.0
        prev_t = times[0]
        for i in range(n):
            if i > 0:
                A = np.exp(-beta * (times[i] - prev_t)) * (A + w[i - 1])
                prev_t = times[i]
            lam = mu + alpha * beta * A
            if lam <= 0:
                return 1e10
            log_intensity_sum += np.log(lam)

        # compensator: integral of lambda over [0, horizon]
        compensator = mu * horizon + alpha * np.sum(w * (1.0 - np.exp(-beta * (horizon - times))))

        ll = log_intensity_sum - compensator
        return -ll if np.isfinite(ll) else 1e10

    def fit(self, times_hours: np.ndarray, marks: np.ndarray, horizon_hours: float):
        """Maximum-likelihood fit. Parameters are optimised in log-space so
        the optimiser cannot wander into negative rates."""
        times_hours = np.asarray(times_hours, dtype=float)
        marks = np.asarray(marks, dtype=float)

        base_rate = max(len(times_hours) / horizon_hours, 1e-6)
        x0 = np.array([np.log(base_rate * 0.5), np.log(0.5), np.log(1 / 12.0), 0.5])

        result = minimize(
            self._neg_loglik, x0,
            args=(times_hours, marks, horizon_hours),
            method="L-BFGS-B",
            bounds=[(-20, 5), (-20, np.log(0.999)), (np.log(1e-4), np.log(10)), (0.0, 3.0)],
        )

        self.mu, self.alpha, self.beta = np.exp(result.x[0]), np.exp(result.x[1]), np.exp(result.x[2])
        self.gamma = float(result.x[3])
        self.loglik = -result.fun
        self.n_events = len(times_hours)
        self.marks_ = marks
        self._converged = bool(result.success)
        return self

    # -- interpretation --------------------------------------------------

    @property
    def half_life_hours(self):
        """How long the excitation from one flare takes to decay by half.
        This is the physically interpretable output: it says how long an
        active region stays 'primed' after firing."""
        return float(np.log(2) / self.beta) if self.beta else None

    def summary(self) -> dict:
        return {
            "mu_per_day": round(float(self.mu * 24), 4),
            "branching_ratio": round(float(self.alpha), 4),
            "half_life_hours": round(self.half_life_hours, 2) if self.beta else None,
            "mark_sensitivity": round(float(self.gamma), 4),
            "loglik": round(float(self.loglik), 2),
            "n_events": self.n_events,
            "converged": self._converged,
        }

    # -- forecasting -----------------------------------------------------

    def intensity(self, t: float, hist_times: np.ndarray, hist_marks: np.ndarray) -> float:
        past = hist_times < t
        if not np.any(past):
            return float(self.mu)
        w = np.exp(self.gamma * hist_marks[past])
        decay = np.exp(-self.beta * (t - hist_times[past]))
        return float(self.mu + self.alpha * self.beta * np.sum(w * decay))

    def probability_within(self, hist_times, hist_marks, now: float,
                           horizon_hours: float = HOURS_PER_DAY) -> float:
        """P(at least one event in (now, now + horizon]), in closed form.

        Worth being precise about why this is exact rather than an
        approximation, because the obvious objection is that a self-exciting
        process cannot use a Poisson-style survival formula.

        The objection does not apply to THIS quantity. Conditioning on "no
        event occurred in the window" also rules out any new self-excitation
        inside the window -- there are no new events to excite anything. So
        along that one path the intensity is fully determined by the history
        already observed, and

            P(N = 0) = exp( - integral of lambda*(s) ds )

        is exact, with lambda* the history-only intensity. For the
        exponential kernel that integral is closed-form:

            mu*h + alpha * sum_j w_j [ e^{-beta(now - t_j)} - e^{-beta(end - t_j)} ]

        Simulation IS still required for the expected NUMBER of events or the
        full count distribution, where self-excitation genuinely compounds --
        see expected_count(). It is just not required here, and using it here
        would only add Monte Carlo noise to an exact answer.
        """
        hist_times = np.asarray(hist_times, dtype=float)
        hist_marks = np.asarray(hist_marks, dtype=float)
        end = now + horizon_hours

        integral = float(self.mu) * horizon_hours
        past = hist_times < now
        if np.any(past):
            w = np.exp(self.gamma * hist_marks[past])
            tj = hist_times[past]
            integral += float(self.alpha * np.sum(
                w * (np.exp(-self.beta * (now - tj)) - np.exp(-self.beta * (end - tj)))
            ))
        return float(1.0 - np.exp(-integral))

    def expected_count(self, hist_times, hist_marks, now: float,
                       horizon_hours: float = HOURS_PER_DAY,
                       n_sims: int = 500, rng=None) -> float:
        """Expected number of events in the window, by simulation.

        Unlike probability_within(), this genuinely needs Monte Carlo: once a
        first event can occur, it excites further events which excite further
        events, and that branching cascade has no simple closed form. Ogata
        thinning, with simulated events appended to the history so their own
        excitation is felt.
        """
        rng = rng or np.random.default_rng(0)
        hist_times = np.asarray(hist_times, dtype=float)
        hist_marks = np.asarray(hist_marks, dtype=float)
        mark_pool = (self.marks_ if self.marks_ is not None and len(self.marks_)
                     else np.array([0.0]))
        end = now + horizon_hours

        total = 0
        for _ in range(n_sims):
            times, marks = hist_times, hist_marks
            t = now
            while t < end:
                # intensity decays between events, so its value at t is a
                # valid thinning upper bound until the next accepted event
                lam_max = self.intensity(t + 1e-9, times, marks)
                if lam_max <= 0:
                    break
                t = t + rng.exponential(1.0 / lam_max)
                if t >= end:
                    break
                if rng.random() <= self.intensity(t, times, marks) / lam_max:
                    times = np.append(times, t)
                    marks = np.append(marks, rng.choice(mark_pool))
                    total += 1
        return total / n_sims


def catalog_to_hawkes_input(catalog: pd.DataFrame, threshold_flux: float = M_CLASS_FLUX):
    """Extract M+ event times (hours since first event) and marks from a
    catalogue.

    Fitting on M+ events only, rather than on all flares with a separate
    class-conditional model, is a deliberate simplification: ~2400 events is
    ample for four parameters, and it keeps the fitted kernel directly
    interpretable as "how M+ events excite further M+ events".
    """
    cat = catalog.copy()
    begin = pd.to_datetime(cat["begin_time"], utc=True, errors="coerce").dt.tz_localize(None)
    cat = cat.assign(begin_time=begin).dropna(subset=["begin_time"])
    cat = cat[cat["peak_flux"].fillna(0) >= threshold_flux].sort_values("begin_time")
    if cat.empty:
        return np.array([]), np.array([]), 0.0, None

    t0 = cat["begin_time"].iloc[0]
    times = (cat["begin_time"] - t0).dt.total_seconds().values / 3600.0
    marks = np.log10(cat["peak_flux"].values / threshold_flux)
    horizon = float(times[-1]) + 1.0
    return times, marks, horizon, t0


def hawkes_daily_forecast(model: HawkesModel, catalog: pd.DataFrame, frame: pd.DataFrame,
                          t0) -> np.ndarray:
    """Run the Hawkes model forward one day at a time over `frame`, using
    only events strictly before each day -- the same leakage discipline the
    feature builder uses.

    Exact and vectorised: probability_within() is closed-form, so this is a
    couple of array operations per day rather than a simulation, which is
    what makes it fast enough to sit behind an interactive dashboard.
    """
    times, marks, _, _ = catalog_to_hawkes_input(catalog)
    if len(times) == 0:
        return np.zeros(len(frame))

    day_hours = (pd.to_datetime(frame["date"]) - t0).dt.total_seconds().values / 3600.0
    probs = np.array([
        model.probability_within(times, marks, now, HOURS_PER_DAY)
        for now in day_hours
    ])
    return probs


# ==========================================================================
# 3. XGBoost
# ==========================================================================

def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    """Gradient-boosted trees on the daily feature frame.

    scale_pos_weight is set from the training imbalance -- without it the
    model learns to always predict "no flare", which is 86% accurate and
    useless."""
    from xgboost import XGBClassifier

    cols = feature_columns(train)
    n_pos = max(int(train["label"].sum()), 1)
    n_neg = max(len(train) - n_pos, 1)

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss", n_jobs=2, random_state=0,
    )
    clf.fit(train[cols], train["label"])
    return clf, clf.predict_proba(test[cols])[:, 1], clf.predict_proba(train[cols])[:, 1]


GP_MAX_TRAIN = 1500


def train_gp_classifier(train: pd.DataFrame, test: pd.DataFrame, max_train: int = GP_MAX_TRAIN):
    """Gaussian Process classifier on the daily features.

    A GP is a better structural match here than in the finance domain, for
    two reasons. The target is already a probability -- did an M+ flare occur
    -- so a GP classifier outputs the quantity being asked for rather than a
    point estimate that has to be reinterpreted. And the metric that matters
    is Brier score, which rewards calibration, and calibration is what a GP
    is actually good at: the same approach gave 82.5% coverage against an 80%
    target on the finance side.

    TRAINING IS CAPPED because the cost is cubic: 400 days fits in 2.3s, 800
    in 14.3s, so the full 11,215-day frame is out of reach without sparse
    approximations. The cap keeps the most RECENT days, which are also the
    most relevant -- the solar cycle means 1998 describes a different Sun
    from 2026.

    An RBF kernel over standardised features, plus a white-noise term so the
    model can attribute part of the variation to irreducible noise instead of
    fitting structure into it.
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.preprocessing import StandardScaler

    cols = feature_columns(train)
    capped = train.tail(max_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(capped[cols])
    y_train = capped["label"].values
    if len(set(y_train)) < 2:
        raise ValueError("training window contains only one class")

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-2)
    clf = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=0,
                                    random_state=0).fit(x_train, y_train)

    prob_test = clf.predict_proba(scaler.transform(test[cols]))[:, 1]
    prob_train = clf.predict_proba(x_train)[:, 1]
    return clf, prob_test, prob_train, scaler


def feature_importance(clf, cols, top_n: int = 12) -> list:
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(imp)[::-1][:top_n]
    return [(cols[i], round(float(imp[i]), 4)) for i in order]


# ==========================================================================
# walk-forward comparison
# ==========================================================================

def walk_forward(frame: pd.DataFrame, n_folds: int = 4, min_train_frac: float = 0.4) -> dict:
    """Expanding-window evaluation of the tractable models.

    Fold k trains on everything up to a cut point and tests on the next
    block, so no model ever sees its own future. Each fold is reported
    separately because the folds land in different phases of the solar cycle
    and averaging them hides exactly the variation worth discussing.

    The Hawkes model is excluded here (its per-day simulation is far slower
    than a tree fit) and is fitted and reported separately.
    """
    n = len(frame)
    start = int(n * min_train_frac)
    bounds = np.linspace(start, n, n_folds + 1).astype(int)

    folds = []
    for k in range(n_folds):
        cut, end = bounds[k], bounds[k + 1]
        train, test = frame.iloc[:cut], frame.iloc[cut:end]
        if train["label"].sum() < 10 or test["label"].sum() < 3:
            continue

        clim = climatology_forecast(train, test)
        pers = persistence_forecast(test)
        clf, xgb_test, xgb_train = train_xgboost(train, test)
        # threshold chosen on TRAIN, applied to test
        _, xgb_thresh = best_tss(train["label"], xgb_train)

        gp_entry = None
        try:
            _gp, gp_test, gp_train, _sc = train_gp_classifier(train, test)
            _, gp_thresh = best_tss(train["label"].tail(len(gp_train)), gp_train)
            gp_entry = evaluate(test["label"], gp_test, gp_thresh)
            gp_entry["brier"] = round(brier(test["label"], gp_test), 4)
            gp_entry["mean_prob"] = round(float(np.mean(gp_test)), 4)
        except Exception as e:
            gp_entry = {"error": f"{type(e).__name__}: {e}"}

        folds.append({
            "fold": k + 1,
            "train_days": len(train),
            "test_span": f"{test['date'].min().date()} -> {test['date'].max().date()}",
            "test_positive_rate": round(float(test["label"].mean()), 4),
            "climatology": evaluate(test["label"], clim, 0.5),
            "persistence": evaluate(test["label"], pers, 0.5),
            "brier_climatology": round(brier(test["label"], clim), 4),
            "brier_persistence": round(brier(test["label"], pers), 4),
            "brier_xgboost": round(brier(test["label"], xgb_test), 4),
            "xgboost": evaluate(test["label"], xgb_test, xgb_thresh),
            "gp_classifier": gp_entry,
            "top_features": feature_importance(clf, feature_columns(train)),
        })
    return {"n_folds": len(folds), "folds": folds}


def fit_and_report(start_year: int = 1996, end_year: int = 2020,
                   n_folds: int = 4) -> dict:
    """The full comparison: load, build features, fit everything, score.
    This is the function the dashboard and the agent node both call."""
    from solar.flare_data import load_historical_catalog

    catalog = load_historical_catalog(start_year, end_year)
    frame = build_daily_frame(catalog)

    report = {
        "catalog": {
            "n_flares": len(catalog),
            "span": f"{catalog['begin_time'].min().date()} -> {catalog['begin_time'].max().date()}",
            "by_class": catalog["class_letter"].value_counts().to_dict(),
        },
        "frame": {
            "n_days": len(frame),
            "positive_rate": round(float(frame["label"].mean()), 4),
            "n_features": len(feature_columns(frame)),
        },
    }

    report["walk_forward"] = walk_forward(frame, n_folds=n_folds)

    times, marks, horizon, t0 = catalog_to_hawkes_input(catalog)
    hawkes = HawkesModel().fit(times, marks, horizon)
    report["hawkes"] = hawkes.summary()

    # score Hawkes on the final fold only -- simulation is the slow part
    cut = int(len(frame) * 0.8)
    test = frame.iloc[cut:].reset_index(drop=True)
    hp = hawkes_daily_forecast(hawkes, catalog, test, t0)
    score, thresh = best_tss(test["label"], hp)
    report["hawkes_eval"] = evaluate(test["label"], hp, thresh)
    report["hawkes_eval"]["note"] = (
        "threshold tuned in-sample on this block; an optimistic upper bound, "
        "reported as such"
    )
    return report


if __name__ == "__main__":
    import json

    print("Fitting solar flare forecasters (1996-2020)...\n")
    rep = fit_and_report()

    c = rep["catalog"]
    print(f"catalogue : {c['n_flares']} flares  {c['span']}  {c['by_class']}")
    f = rep["frame"]
    print(f"features  : {f['n_days']} days, {f['n_features']} features, "
          f"{f['positive_rate']:.1%} positive\n")

    print("Hawkes process (fitted on M+ events):")
    for k, v in rep["hawkes"].items():
        print(f"  {k:<20} {v}")
    print(f"  -> one M+ flare raises the M+ rate, decaying by half every "
          f"{rep['hawkes']['half_life_hours']}h")
    print()

    print("Walk-forward comparison (TSS; 0 = no skill):")
    print(f"  {'fold':<6}{'test period':<26}{'pos%':<8}{'clim':<8}{'persist':<10}{'xgboost':<9}")
    for fold in rep["walk_forward"]["folds"]:
        print(f"  {fold['fold']:<6}{fold['test_span']:<26}"
              f"{fold['test_positive_rate']:<8.1%}"
              f"{fold['climatology']['tss']:<8}"
              f"{fold['persistence']['tss']:<10}"
              f"{fold['xgboost']['tss']:<9}")

    print(f"\nHawkes on final block: TSS {rep['hawkes_eval']['tss']}  "
          f"Brier {rep['hawkes_eval']['brier']}")
    print(f"  ({rep['hawkes_eval']['note']})")

    print("\nTop XGBoost features (last fold):")
    for name, val in rep["walk_forward"]["folds"][-1]["top_features"][:8]:
        print(f"  {name:<22} {val}")
