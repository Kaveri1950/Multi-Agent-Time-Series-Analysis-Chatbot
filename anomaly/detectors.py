"""Twelve detectors, six theories of what "anomalous" means.

Grouped by assumption rather than by sophistication, because the comparison
is about which assumption holds on which data -- not about which model has
the most parameters.

  floor          no learning at all; exists to be beaten
  one_liner      a single arithmetic expression
  threshold      far from typical, in units of typical spread
  density        sits where few other windows sit
  reconstruction hard to rebuild from learned structure
  latent_state   does not fit the regime currently in force
  ceiling        allowed to see the labels; reports what is achievable

The floors are not a formality. Kim et al. (AAAI 2022) found untrained
models matching published detectors on standard benchmarks, and Wu & Keogh
(TKDE) found most benchmark problems yield to one line of code. A detector
that cannot clear both groups has not earned its place.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from anomaly.core import (Detector, series_from_windows, standardise,
                          window_features)

# ---------------------------------------------------------------- floors


class RawInput(Detector):
    """Score = how far the window's extreme lies from the global median.

    No fitting beyond one median and one scale. This is the baseline Kim et
    al. describe as "the anomaly score defined as the input itself" -- the
    extreme case of a model that has learned nothing whatsoever. Anything
    that fails to beat it is not detecting, it is just reading.
    """

    name, group, is_floor = "raw_input", "floor", True

    def _fit(self, windows, y=None):
        flat = windows.ravel()
        self._median = float(np.median(flat))
        scale = float(np.median(np.abs(flat - self._median))) or float(flat.std()) or 1.0
        self._scale = scale

    def _score(self, windows):
        return np.abs(windows - self._median).max(axis=1) / self._scale


class UntrainedAutoencoder(Detector):
    """Reconstruction error from a randomly-initialised linear autoencoder.

    A random projection to a bottleneck and back, never trained. If the
    TRAINED autoencoder cannot beat this, its training accomplished nothing
    -- which is the specific comparison the reconstruction assumption
    deserves, given the ICLR 2026 finding that anomalies are not reliably
    harder to reconstruct than normal data.
    """

    name, group, is_floor = "untrained_autoencoder", "floor", True

    def __init__(self, bottleneck: int = 8, seed: int = 0):
        self.bottleneck, self.seed = bottleneck, seed

    def _fit(self, windows, y=None):
        rng = np.random.default_rng(self.seed)
        w = windows.shape[1]
        k = min(self.bottleneck, w)
        # Orthonormal random projection, so the "encoder" is at least a
        # sane compression rather than a degenerate one.
        self._enc = np.linalg.qr(rng.standard_normal((w, k)))[0]
        self._mu = windows.mean(axis=0)
        self._sd = windows.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0

    def _score(self, windows):
        x = (windows - self._mu) / self._sd
        recon = (x @ self._enc) @ self._enc.T
        return ((x - recon) ** 2).mean(axis=1)


# ------------------------------------------------------------- one-liners


class MovingVariance(Detector):
    """Score = variance within the window.

    One of the two "one-liner" baselines that ICLR 2026 found time-series
    foundation models fail to significantly outperform.
    """

    name, group = "moving_variance", "one_liner"

    def _fit(self, windows, y=None):
        self._scale = float(windows.std()) or 1.0

    def _score(self, windows):
        return windows.var(axis=1) / (self._scale ** 2)


class SquaredDifference(Detector):
    """Score = largest squared step inside the window.

    The second one-liner. Catches abrupt level changes, which is what a
    surprising fraction of real anomalies actually are.
    """

    name, group = "squared_difference", "one_liner"

    def _fit(self, windows, y=None):
        self._scale = float(np.diff(windows, axis=1).std()) or 1.0

    def _score(self, windows):
        return (np.diff(windows, axis=1) ** 2).max(axis=1) / (self._scale ** 2)


# -------------------------------------------------------------- threshold


class RollingZScore(Detector):
    """Score = largest absolute deviation from the global mean, in sigmas.

    The textbook detector, and the one most likely to be quietly competitive.
    """

    name, group = "zscore", "threshold"

    def _fit(self, windows, y=None):
        flat = windows.ravel()
        self._mu = float(flat.mean())
        self._sd = float(flat.std()) or 1.0

    def _score(self, windows):
        return np.abs(windows - self._mu).max(axis=1) / self._sd


class HampelMAD(Detector):
    """Median/MAD version of the above.

    Matters more than it looks. A z-score computes its own sigma from data
    that CONTAINS the anomalies it is hunting, so a large outlier inflates
    the denominator and partially hides itself -- worse, several outliers
    hide each other. Median absolute deviation has a 50% breakdown point and
    does not. 1.4826 rescales MAD to be comparable to a Gaussian sigma.
    """

    name, group = "hampel_mad", "threshold"

    def _fit(self, windows, y=None):
        flat = windows.ravel()
        self._med = float(np.median(flat))
        mad = float(np.median(np.abs(flat - self._med)))
        self._scale = 1.4826 * mad or float(flat.std()) or 1.0

    def _score(self, windows):
        return np.abs(windows - self._med).max(axis=1) / self._scale


class CusumDetector(Detector):
    """Cumulative sum control chart (Page 1954), the classical change detector.

    Every other threshold detector here asks whether any SINGLE point is far
    from normal. CUSUM asks a different question: has a run of small
    deviations, none alarming on its own, accumulated into evidence that the
    level has shifted? That is the regime-change case, and a per-point
    threshold is blind to it by construction -- twenty days each 0.4 sigma
    high trip nothing, while their cumulative sum is unmistakable.

    Two one-sided sums are tracked, so an upward and a downward shift are
    both caught:

        S+ = max(0, S+ + (x - mu) / sigma - k)
        S- = max(0, S- - (x - mu) / sigma - k)

    `k` is the slack, in standard deviations. It is the drift the chart is
    willing to ignore, and the conventional choice is half the shift you want
    to detect; 0.5 therefore targets a one-sigma move. The window's score is
    the largest value either sum reaches.

    mu and sigma come from the TRAINING windows, not from the window being
    scored -- a chart calibrated on the data it is judging would absorb the
    shift into its own reference level and report nothing.
    """

    name, group = "cusum", "threshold"

    def __init__(self, k: float = 0.5):
        self.k = k

    def _fit(self, windows, y=None):
        flat = windows.ravel()
        self._mu = float(np.mean(flat))
        self._sigma = float(np.std(flat)) or 1.0

    def _score(self, windows):
        z = (windows - self._mu) / self._sigma
        pos = np.zeros(len(windows))
        neg = np.zeros(len(windows))
        worst = np.zeros(len(windows))
        for j in range(z.shape[1]):
            pos = np.maximum(0.0, pos + z[:, j] - self.k)
            neg = np.maximum(0.0, neg - z[:, j] - self.k)
            worst = np.maximum(worst, np.maximum(pos, neg))
        return worst


# ---------------------------------------------------------------- density


class IsolationForestDetector(Detector):
    """Random splits isolate an outlier in fewer cuts than an inlier.

    Operates on the domain-neutral window feature table rather than raw
    points, so "unusual" can mean an unusual COMBINATION -- moderate depth
    with unusually low variance, say -- rather than merely an extreme value.
    """

    name, group = "isolation_forest", "density"

    def __init__(self, n_estimators: int = 200, seed: int = 42):
        self.n_estimators, self.seed = n_estimators, seed

    def _fit(self, windows, y=None):
        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed,
        ).fit((feats - self._mu) / self._sd)

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        return -self._model.score_samples((feats - self._mu) / self._sd)


class MahalanobisDetector(Detector):
    """Distance from the feature centroid, accounting for correlation.

    The turbulence measure of Kritzman & Li (2010). Its distinguishing
    property is sensitivity to BROKEN RELATIONSHIPS rather than large values:
    a window whose features are individually ordinary but jointly
    inconsistent with their usual covariance scores highly, and nothing
    univariate can see that.

    Fitted with a robust covariance estimator, because a plain sample
    covariance is computed from data containing the anomalies and gets
    dragged towards them -- the same self-hiding problem as the z-score, one
    dimension up.
    """

    name, group = "mahalanobis", "density"

    def __init__(self, support_fraction: float = 0.85, seed: int = 42):
        self.support_fraction, self.seed = support_fraction, seed

    def _fit(self, windows, y=None):
        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        z = (feats - self._mu) / self._sd
        try:
            self._model = MinCovDet(
                support_fraction=self.support_fraction, random_state=self.seed,
            ).fit(z)
        except (ValueError, np.linalg.LinAlgError):
            # Degenerate covariance (constant features, or fewer windows than
            # features) -- fall back to a diagonal distance rather than
            # dropping the detector from the comparison entirely.
            self._model = None

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        z = (feats - self._mu) / self._sd
        if self._model is None:
            return (z ** 2).sum(axis=1)
        return self._model.mahalanobis(z)


# --------------------------------------------------------- reconstruction


class LinearAutoencoder(Detector):
    """PCA reconstruction error -- a trained linear autoencoder.

    Kept deliberately linear. The point of this row is to test the
    RECONSTRUCTION ASSUMPTION itself, not a particular architecture, and a
    linear bottleneck tests it with no training noise, no learning rate and
    no seed sensitivity. If reconstruction error carried signal, the optimal
    linear bottleneck would show it.
    """

    name, group = "pca_autoencoder", "reconstruction"

    def __init__(self, bottleneck: int = 8):
        self.bottleneck = bottleneck

    def _fit(self, windows, y=None):
        self._mu = windows.mean(axis=0)
        self._sd = windows.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        x = (windows - self._mu) / self._sd
        k = min(self.bottleneck, min(x.shape) - 1) or 1
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self._components = vt[:k].T

    def _score(self, windows):
        x = (windows - self._mu) / self._sd
        recon = (x @ self._components) @ self._components.T
        return ((x - recon) ** 2).mean(axis=1)


class LatentIsolationForest(Detector):
    """Bottleneck coordinates -> IsolationForest.

    The approach the astronomical transient literature actually uses
    (Muthukrishna et al., MCIF) rather than reconstruction error: throw away
    the decoder and ask whether the COMPRESSED representation is unusual.
    Same trained encoder as `pca_autoencoder`, different question asked of it,
    which isolates how much of that detector's result is the representation
    and how much is the reconstruction assumption.
    """

    name, group = "latent_isoforest", "reconstruction"

    def __init__(self, bottleneck: int = 8, n_estimators: int = 200, seed: int = 42):
        self.bottleneck, self.n_estimators, self.seed = bottleneck, n_estimators, seed

    def _fit(self, windows, y=None):
        self._mu = windows.mean(axis=0)
        self._sd = windows.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        x = (windows - self._mu) / self._sd
        k = min(self.bottleneck, min(x.shape) - 1) or 1
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self._components = vt[:k].T
        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed,
        ).fit(x @ self._components)

    def _score(self, windows):
        x = (windows - self._mu) / self._sd
        return -self._model.score_samples(x @ self._components)


# ---------------------------------------------------------- latent state


class HMMDetector(Detector):
    """Negative log-likelihood under a fitted Gaussian HMM.

    Asks a different question from every other detector here: not "is this
    point extreme" but "what regime is in force, and does this fit it?" A
    3% daily move is unremarkable inside a crisis regime and alarming inside
    a calm one -- a distinction no global threshold can draw.

    Fitted on window features rather than raw points so the state space
    stays small. Falls back to a single-Gaussian likelihood if hmmlearn is
    absent, which keeps the grid runnable without the optional dependency
    and is honestly labelled in `name` when it happens.
    """

    group = "latent_state"

    def __init__(self, n_states: int = 3, seed: int = 42):
        self.n_states, self.seed = n_states, seed
        self.name = f"hmm_{n_states}state"

    def _fit(self, windows, y=None):
        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        z = (feats - self._mu) / self._sd
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            self._model = None
            self.name = f"hmm_{self.n_states}state_unavailable"
            return
        try:
            self._model = GaussianHMM(
                n_components=self.n_states, covariance_type="diag",
                n_iter=100, random_state=self.seed,
            ).fit(z)
        except (ValueError, np.linalg.LinAlgError):
            self._model = None

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        z = (feats - self._mu) / self._sd
        if self._model is None:
            return (z ** 2).sum(axis=1)
        # Expected log-likelihood of each window under the posterior over
        # states: sum_k P(state=k | all data) * log P(window | state=k).
        #
        # The obvious alternative -- differencing the running total
        # log-likelihood of the first i windows -- is O(n^2) in forward
        # passes and took 35s on 600 windows, which would be hours on a
        # Kepler light curve. This is one forward-backward pass, O(n), and
        # measures the same thing: how well the window fits the regime the
        # model believes is in force.
        posteriors = self._model.predict_proba(z)
        frame_logprob = self._model._compute_log_likelihood(z)
        return -(posteriors * frame_logprob).sum(axis=1)


class GarchResidual(Detector):
    """Return standardised by its own conditional volatility.

    Finance-specific and labelled as such. Separates "large move" from
    "large move GIVEN how volatile the market already is", which is the
    distinction that makes March 2020 unremarkable and an identical move in
    a quiet July alarming.

    Uses an EWMA variance recursion rather than a fitted GARCH when `arch`
    is unavailable. That is RiskMetrics' estimator, not a hack -- GARCH(1,1)
    with alpha+beta pinned to 1 -- and it needs no optimiser.
    """

    group = "latent_state"

    def __init__(self, lam: float = 0.94):
        self.lam = lam
        self.name = "garch_residual"

    def _fit(self, windows, y=None):
        self._mu = float(windows.ravel().mean())

    def _score(self, windows):
        out = np.empty(len(windows))
        for i, w in enumerate(windows):
            x = w - self._mu
            var = x[0] ** 2 if x[0] else float(np.var(x)) or 1.0
            worst = 0.0
            for val in x[1:]:
                var = self.lam * var + (1 - self.lam) * val ** 2
                worst = max(worst, abs(val) / np.sqrt(max(var, 1e-18)))
            out[i] = worst
        return out


# ---------------------------------------------------------------- ceiling


class SupervisedCeiling(Detector):
    """A classifier allowed to see the labels. NOT a competitor.

    Reports what is achievable on these features at all, so an unsupervised
    score of 0.58 can be read as "most of the available signal" or "barely
    any of it" rather than floating free. Scored out-of-fold, because an
    in-sample ceiling is just memorisation and would set an unreachable bar.
    """

    name, group, is_ceiling = "supervised_ceiling", "ceiling", True

    def __init__(self, seed: int = 42, n_splits: int = 5):
        self.seed, self.n_splits = seed, n_splits

    def _fit(self, windows, y=None):
        if y is None:
            raise ValueError("SupervisedCeiling requires labels")
        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        y = np.asarray(y).astype(int).ravel()
        self._fit_n = len(feats)
        if len(np.unique(y)) < 2:
            self._model, self._oof = None, np.zeros(len(y))
            return
        z = (feats - self._mu) / self._sd
        # Out-of-fold predictions for the TRANSDUCTIVE case: an in-sample
        # ceiling is memorisation and would set an unreachable bar. A model
        # fitted on the whole training split then serves the held-out case,
        # where no such precaution is needed.
        n_splits = max(2, min(self.n_splits, int(np.bincount(y).min())))
        self._oof = cross_val_predict(
            GradientBoostingClassifier(random_state=self.seed), z, y,
            cv=n_splits, method="predict_proba")[:, 1]
        self._model = GradientBoostingClassifier(random_state=self.seed).fit(z, y)

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        if len(feats) == self._fit_n:
            return self._oof
        if self._model is None:
            return np.zeros(len(feats))
        return self._model.predict_proba((feats - self._mu) / self._sd)[:, 1]


def default_detectors(include_finance_only: bool = False,
                      stride: int = 1) -> list[Detector]:
    """The pre-declared configuration. One per detector, no tuning.

    Fixed deliberately: the moment detectors get individually tuned, the
    reporting budget stops being 1 and the metrics stop being interpretable.
    """
    dets: list[Detector] = [
        RawInput(), UntrainedAutoencoder(),
        MovingVariance(), SquaredDifference(),
        RollingZScore(), HampelMAD(), CusumDetector(),
        IsolationForestDetector(), MahalanobisDetector(),
        LocalOutlierFactorDetector(), EllipticEnvelopeDetector(), OneClassSVMDetector(),
        LinearAutoencoder(), LatentIsolationForest(), SpectralPCADetector(),
        HMMDetector(n_states=3),
        # All four matrix-profile configurations, because which one wins is
        # exactly the question -- and the answer differs by domain.
        MatrixProfileDetector(motif=False, normalize=True, stride=stride),
        MatrixProfileDetector(motif=False, normalize=False, stride=stride),
        MatrixProfileDetector(motif=True, normalize=True, stride=stride),
        MatrixProfileDetector(motif=True, normalize=False, stride=stride),
    ]
    if include_finance_only:
        dets.append(GarchResidual())
    dets.append(SupervisedCeiling())
    return dets


# ------------------------------------------------- additional density methods
# The classical outlier-detection family from scikit-learn's own comparison
# (User Guide 2.7). Their documentation notes IsolationForest and LOF "perform
# reasonably well", One-Class SVM "is known to be sensitive to outliers", and
# EllipticEnvelope assumes Gaussian data -- all three claims are testable here
# on real labelled data rather than on toy datasets.


class LocalOutlierFactorDetector(Detector):
    """Density relative to neighbours, rather than density in absolute terms.

    Asks how isolated a window is COMPARED WITH ITS OWN NEIGHBOURHOOD, so it
    can flag a point sitting in a sparse pocket even when denser clusters
    exist elsewhere. That matters here because window statistics are not
    uniformly distributed -- quiet stretches and active stretches form
    separate clumps, and a global density method judges them by one standard.

    novelty=False, so scores come from `negative_outlier_factor_` on the
    fitted data. Calling predict on training rows under novelty=True gives
    silently wrong answers, which sklearn warns about twice.
    """

    name, group = "local_outlier_factor", "density"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def _fit(self, windows, y=None):
        from sklearn.neighbors import LocalOutlierFactor

        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        z = (feats - self._mu) / self._sd
        n = max(2, min(self.n_neighbors, len(z) - 1))
        # TWO models on purpose. LOF was designed for outlier detection, where
        # scores come from `negative_outlier_factor_` on the fitted rows and
        # there is no predict method. Scoring UNSEEN rows is a different task
        # (sklearn calls it novelty detection) and needs novelty=True, which
        # in turn must never be used on the training rows -- sklearn warns
        # about this twice. Keeping both lets the same detector be evaluated
        # transductively and held-out without either path being wrong.
        self._fit_n = len(z)
        self._transductive = -LocalOutlierFactor(n_neighbors=n).fit(z).negative_outlier_factor_
        self._novelty = LocalOutlierFactor(n_neighbors=n, novelty=True).fit(z)

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        if len(feats) == self._fit_n:
            return self._transductive
        return -self._novelty.score_samples((feats - self._mu) / self._sd)


class EllipticEnvelopeDetector(Detector):
    """Robust Gaussian ellipse; outliers are what falls outside it.

    Effectively Mahalanobis distance with an explicit contamination
    assumption. Included because sklearn presents it as a standard option and
    because its Gaussian assumption is exactly the kind of thing worth
    testing rather than trusting: window features here are visibly skewed.
    """

    name, group = "elliptic_envelope", "density"

    def __init__(self, contamination: float = 0.1, seed: int = 42):
        self.contamination, self.seed = contamination, seed

    def _fit(self, windows, y=None):
        from sklearn.covariance import EllipticEnvelope

        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        try:
            self._model = EllipticEnvelope(contamination=self.contamination,
                                           random_state=self.seed,
                                           support_fraction=0.9)
            self._model.fit((feats - self._mu) / self._sd)
        except (ValueError, np.linalg.LinAlgError):
            self._model = None

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        z = (feats - self._mu) / self._sd
        if self._model is None:
            return (z ** 2).sum(axis=1)
        return -self._model.score_samples(z)


class OneClassSVMDetector(Detector):
    """RBF one-class SVM -- a learned frontier around the normal region.

    Uses SGDOneClassSVM with a Nystroem kernel approximation rather than the
    exact solver. The exact version is at best quadratic in samples, and
    these datasets run to hundreds of thousands of windows; sklearn documents
    this combination as the way to scale it.
    """

    name, group = "one_class_svm", "density"

    def __init__(self, nu: float = 0.1, n_components: int = 100, seed: int = 42):
        self.nu, self.n_components, self.seed = nu, n_components, seed

    def _fit(self, windows, y=None):
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.pipeline import make_pipeline

        feats = window_features(windows).to_numpy()
        self._mu, self._sd = feats.mean(axis=0), feats.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        z = (feats - self._mu) / self._sd
        comps = min(self.n_components, len(z))
        self._model = make_pipeline(
            Nystroem(gamma=1.0 / max(z.shape[1], 1), n_components=comps,
                     random_state=self.seed),
            SGDOneClassSVM(nu=self.nu, random_state=self.seed),
        ).fit(z)

    def _score(self, windows):
        feats = window_features(windows).to_numpy()
        return -self._model.decision_function((feats - self._mu) / self._sd)


# ----------------------------------------------------- frequency-domain route


class SpectralPCADetector(Detector):
    """FFT -> PCA -> distance in component space.

    The representation used for gamma-ray burst work: move each window to the
    frequency domain, compress with PCA, then judge unusualness there. The
    premise is that WHAT FREQUENCIES a window contains describes its shape
    better than its summary statistics do -- a smooth dip and a jagged noisy
    stretch can share a mean, a variance and a range while having entirely
    different spectra.

    Magnitudes only, phase discarded, so the representation does not care
    WHERE in the window an event sits -- only what it looked like. That is
    the right invariance for overlapping windows, where the same event
    appears at four different offsets.

    IMPORTANT LIMIT: this is an FFT of each WINDOW, so it can only see
    structure shorter than the window. A transit recurring every 4.77 days is
    invisible to a 12-hour window no matter what transform is applied. Finding
    that requires a periodogram over the FULL series (Lomb-Scargle, or the
    box-fitting BLS used for transits) -- a different computation, not a
    different feature set.
    """

    name, group = "spectral_pca", "reconstruction"

    def __init__(self, n_components: int = 8, seed: int = 42):
        self.n_components, self.seed = n_components, seed

    @staticmethod
    def _spectra(windows: np.ndarray) -> np.ndarray:
        centred = windows - windows.mean(axis=1, keepdims=True)
        mag = np.abs(np.fft.rfft(centred, axis=1))
        # Log-power: spectral magnitudes span orders of magnitude, and PCA on
        # raw magnitudes is dominated entirely by the lowest frequency bin.
        return np.log1p(mag)

    def _fit(self, windows, y=None):
        from sklearn.decomposition import PCA

        spec = self._spectra(windows)
        self._mu, self._sd = spec.mean(axis=0), spec.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        z = (spec - self._mu) / self._sd
        k = max(1, min(self.n_components, min(z.shape) - 1))
        self._pca = PCA(n_components=k, random_state=self.seed).fit(z)
        scores = self._pca.transform(z)
        self._centre = scores.mean(axis=0)
        self._scale = scores.std(axis=0)
        self._scale[self._scale < 1e-12] = 1.0

    def _score(self, windows):
        spec = self._spectra(windows)
        z = (spec - self._mu) / self._sd
        comp = (self._pca.transform(z) - self._centre) / self._scale
        return (comp ** 2).sum(axis=1)


# ------------------------------------------------------- matrix profile
# Keogh's own method -- the same Keogh whose benchmark critique this project
# relies on. It compares every subsequence against every other and reports
# the distance to each one's nearest neighbour.
#
# TWO AXES, AND BOTH FLIP BETWEEN DOMAINS. This is the most interesting
# measured result in the package.
#
#   DIRECTION   a DISCORD has no near neighbour: a one-off event.
#               a MOTIF has many: a recurring pattern.
#
#               A market shock is a discord -- COVID happened once. A
#               planetary transit is a MOTIF: it recurs ~221 times with
#               near-identical shape, making it the LEAST unusual feature of
#               the light curve. Scored as a discord it lands below chance
#               (measured: ROC 0.303 on KIC 10074466); inverted, 0.697.
#
#   NORMALISATION  z-normalised distance rescales each subsequence before
#               comparing, so only SHAPE counts. Right for transits, whose
#               depth varies by star. Catastrophic for returns, where the
#               SIZE of the move is the entire signal -- a -10% crash and a
#               +0.1% wiggle of the same profile become near-identical.
#               Measured on equities at m=10: z-normalised discord reaches
#               1.05x lift, non-normalised reaches 5.27x.


class MatrixProfileDetector(Detector):
    """Distance to each subsequence's nearest neighbour, via STUMPY.

    Operates on the reconstructed series rather than the window matrix,
    because nearest-neighbour search needs adjacency.
    """

    group = "matrix_profile"

    def __init__(self, motif: bool = False, normalize: bool = True,
                 subsequence: int | None = None, stride: int = 1):
        self.motif, self.normalize = motif, normalize
        self.subsequence, self.stride = subsequence, stride
        self.name = ("matrix_profile_"
                     + ("motif" if motif else "discord")
                     + ("" if normalize else "_raw"))

    def _fit(self, windows, y=None):
        # Nothing is learned. A matrix profile compares a series against
        # ITSELF, so there are no parameters to carry from a training split
        # to a test one -- the profile of the test series is computed from
        # the test series. Recording the geometry is all `fit` can do, and
        # scoring reconstructs the series it is actually given.
        self._window = windows.shape[1]

    def _score(self, windows):
        import stumpy

        m = self.subsequence or self._window
        n_windows = len(windows)
        series = series_from_windows(windows, self.stride)
        x = np.ascontiguousarray(series, dtype=np.float64)
        if len(x) <= m + 1:
            return np.zeros(n_windows)
        try:
            profile = (stumpy.stump(x, m=m) if self.normalize
                       else stumpy.aamp(x, m=m))[:, 0].astype(float)
        except Exception:
            return np.zeros(n_windows)
        # A motif is the ABSENCE of distance, so the sign flips. Everything
        # downstream assumes higher = more anomalous.
        profile = -profile if self.motif else profile
        # Map subsequence positions back to windows: window i begins at
        # index i * stride in the reconstructed series.
        idx = np.minimum(np.arange(n_windows) * self.stride, len(profile) - 1)
        return np.nan_to_num(profile[idx], nan=float(np.nanmin(profile)))


class LSTMAutoencoder(Detector):
    """Sequence-to-sequence autoencoder over the window, in PyTorch.

    Tests the one hypothesis nothing else here can: DOES THE ORDER OF POINTS
    INSIDE A WINDOW MATTER? Every other detector treats a window as an
    unordered bag -- `window_features` computes mean, spread, skew, all
    permutation-invariant, and even the PCA autoencoder learns a fixed linear
    map that has no notion of "before" and "after". An LSTM reads the window
    as a sequence, so if temporal shape carries signal it should find it.

    The prior is poor and worth stating up front. The linear autoencoder lost
    to an UNTRAINED one (0.265 vs 0.273 on Kepler), a non-linear MLP
    autoencoder barely matched random weights (2.01x vs 2.00x lift), and ICLR
    2026 explains why: "anomalies are not consistently harder to reconstruct
    or forecast". A negative result here is informative rather than
    disappointing -- it closes off the temporal-structure explanation.

    Deliberately small: one layer, 16 hidden units, 30 epochs. A larger model
    on CPU would take hours for a hypothesis the evidence says will fail, and
    capacity is exactly what inflates in-sample scores.
    """

    name, group = "lstm_autoencoder", "reconstruction"

    def __init__(self, hidden: int = 16, epochs: int = 30, batch: int = 256,
                 lr: float = 1e-2, seed: int = 0, max_train: int = 20000):
        self.hidden, self.epochs, self.batch = hidden, epochs, batch
        self.lr, self.seed, self.max_train = lr, seed, max_train

    def _build(self, window_len: int):
        import torch
        from torch import nn

        class Seq2Seq(nn.Module):
            def __init__(self, hidden, length):
                super().__init__()
                self.length = length
                self.encoder = nn.LSTM(1, hidden, batch_first=True)
                self.decoder = nn.LSTM(1, hidden, batch_first=True)
                self.out = nn.Linear(hidden, 1)

            def forward(self, x):
                _, (h, c) = self.encoder(x)
                # Decode from a zero input driven only by the encoder state,
                # so the bottleneck really is the hidden vector -- feeding the
                # input back in would let it copy rather than compress.
                zeros = torch.zeros(x.size(0), self.length, 1, device=x.device)
                seq, _ = self.decoder(zeros, (h, c))
                return self.out(seq)

        return Seq2Seq(self.hidden, window_len)

    def _fit(self, windows, y=None):
        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)          # workers are already parallel
        self._mu = windows.mean(axis=0)
        self._sd = windows.std(axis=0)
        self._sd[self._sd < 1e-12] = 1.0
        x = (windows - self._mu) / self._sd

        # Subsample for training only. 280,000 Kepler windows through an LSTM
        # on CPU is hours; 20,000 is minutes and the reconstruction target is
        # the same distribution.
        if len(x) > self.max_train:
            idx = np.random.default_rng(self.seed).choice(len(x), self.max_train, replace=False)
            x_train = x[idx]
        else:
            x_train = x

        tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
        self._model = self._build(windows.shape[1])
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        self._model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(tensor))
            for i in range(0, len(tensor), self.batch):
                batch = tensor[perm[i:i + self.batch]]
                opt.zero_grad()
                loss = loss_fn(self._model(batch), batch)
                loss.backward()
                opt.step()
        self._model.eval()

    def _score(self, windows):
        import torch

        x = (windows - self._mu) / self._sd
        tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        errors = np.empty(len(x), dtype=float)
        with torch.no_grad():
            for i in range(0, len(tensor), 1024):
                chunk = tensor[i:i + 1024]
                recon = self._model(chunk)
                errors[i:i + len(chunk)] = ((chunk - recon) ** 2).mean(dim=(1, 2)).numpy()
        return errors
