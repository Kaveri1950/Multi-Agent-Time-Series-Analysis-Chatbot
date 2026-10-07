"""Model Confidence Set (Hansen, Lunde & Nason 2011).

WHY THIS AND NOT A PILE OF PAIRWISE TESTS. The anomaly benchmark compared 21
detectors; this registry holds 21 models. Running Diebold-Mariano on every
pair and keeping whatever looks best is exactly the data-snooping the MCS was
invented to stop: with 21 models and a 5% test, some pair clears the bar by
luck alone. The MCS instead returns the SMALLEST SET that contains the best
model with a stated confidence, and makes no assumption that any particular
model is the benchmark.

That output is also the right shape for this project. A set of models that
cannot be statistically separated IS a feasible shortlist with a confidence
level attached -- the same object `feasible()` produces from capability
constraints, now produced from measured loss.

THE PROCEDURE, as implemented here (the T_max variant):

  1. For each model i, d_i = mean_t( L_i,t - Lbar_t ), its average loss
     relative to the surviving set.
  2. Studentise d_i by a bootstrap standard error, giving t_i.
  3. T_max = max_i t_i. If T_max exceeds its bootstrap critical value, the
     hypothesis that all surviving models are equally good is rejected, the
     single worst model is eliminated, and step 1 repeats.
  4. When the hypothesis is no longer rejected, whatever remains is the MCS.

A MOVING-BLOCK BOOTSTRAP is used throughout, not an i.i.d. one. Loss series
from overlapping forecasts are serially dependent, and resampling individual
days would understate the variance and return an MCS that is too small --
the same mistake the i.i.d. window bootstrap made on the Kepler margins,
where it reported P(margin>0)=1.0000 for a difference the cluster bootstrap
could not separate from zero.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BOOT = 2000
DEFAULT_ALPHA = 0.05
SEED = 0


def _block_indices(n: int, block: int, rng) -> np.ndarray:
    """One moving-block bootstrap resample of positions 0..n-1."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def model_confidence_set(losses: dict, alpha: float = DEFAULT_ALPHA,
                         n_boot: int = DEFAULT_BOOT, block: int | None = None,
                         seed: int = SEED, verbose: bool = False) -> dict:
    """Hansen-Lunde-Nason MCS over per-observation loss series.

    `losses` maps model name -> array of per-observation losses, all the same
    length and aligned on the same observations.
    """
    names = list(losses)
    L = np.column_stack([np.asarray(losses[k], dtype=float) for k in names])
    n, m = L.shape
    if m < 2:
        return {"mcs": names, "alpha": alpha, "eliminated": [], "n_obs": n}
    if block is None:
        # Politis-White style rule of thumb; one month of trading days is the
        # natural dependence scale for a daily loss series.
        block = max(2, min(22, int(round(n ** (1 / 3)))))

    rng = np.random.default_rng(seed)
    boot_idx = [_block_indices(n, block, rng) for _ in range(n_boot)]

    alive = list(range(m))
    eliminated = []
    while len(alive) > 1:
        sub = L[:, alive]
        # d_i,t : model i's loss relative to the mean of the surviving set
        d = sub - sub.mean(axis=1, keepdims=True)
        dbar = d.mean(axis=0)

        boot_means = np.empty((n_boot, len(alive)))
        for b, idx in enumerate(boot_idx):
            boot_means[b] = d[idx].mean(axis=0)
        se = boot_means.std(axis=0, ddof=1)
        se[se <= 0] = np.inf
        t = dbar / se

        # Null distribution of the studentised range statistic
        boot_t = (boot_means - dbar) / se
        boot_tmax = boot_t.max(axis=1)
        tmax = t.max()
        crit = float(np.quantile(boot_tmax, 1.0 - alpha))
        p_value = float((boot_tmax >= tmax).mean())

        if verbose:
            print(f"    alive={len(alive)}  T_max={tmax:+.3f}  crit={crit:+.3f}  p={p_value:.4f}")

        if p_value > alpha:
            break
        worst = int(np.argmax(t))
        eliminated.append({"model": names[alive[worst]], "t_stat": float(t[worst]),
                           "p_value": p_value, "n_alive_before": len(alive)})
        alive.pop(worst)

    mean_loss = {names[i]: float(L[:, i].mean()) for i in range(m)}
    return {
        "mcs": [names[i] for i in alive],
        "eliminated": eliminated,
        "alpha": alpha, "n_obs": int(n), "n_models": int(m),
        "block_length": int(block), "n_bootstrap": int(n_boot), "seed": seed,
        "mean_loss": mean_loss,
        "best_by_mean_loss": min(mean_loss, key=mean_loss.get),
    }


__all__ = ["model_confidence_set"]
