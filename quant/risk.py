"""Regime-conditional Value-at-Risk, with the standard backtests.

The point of this module is to turn a backward-looking question into a
forward-looking one. Anomaly detection asks "was yesterday strange?", which
the user already knows. Risk asks "how much could I lose tomorrow, given
what today looks like?", which they do not.

The chain, all of it causal:

    Mahalanobis turbulence across the cross-section   (correlation breakdown)
        -> HMM regime state, FILTERED not smoothed    (calm or turbulent)
            -> VaR / ES conditioned on that state     (how bad could it get)
                -> Kupiec + Christoffersen backtests  (is the number honest)

WHY CONDITIONING HELPS

Measured on the equal-weighted Nifty proxy, using the regime inferred from
data available on the day to predict the NEXT day:

    quiet       n=1486   ann vol 17.8%   95% VaR -1.91%   ES -2.86%
    turbulent   n= 172   ann vol 36.3%   95% VaR -3.49%   ES -6.03%

An unconditional VaR splits the difference: too tight in calm markets, too
loose in crises. It passes a coverage test anyway, because the errors cancel
-- which is exactly why coverage alone is not enough, and why the
independence test below matters.

WHAT DOES NOT WORK, RECORDED HERE RATHER THAN OMITTED

Using the same regime signal to SIZE POSITIONS made risk-adjusted returns
worse: Sharpe 0.67 -> 0.26, though maximum drawdown improved from -18.8% to
-8.6%. The reason is that turbulent regimes in this sample were frequently
rallies rather than crashes -- "turbulent" means moving, not falling.
Estimating risk and timing exposure are different problems, and this signal
is only good at the first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Standard regulatory confidence levels. 99% is the Basel number; 95% is
#: the one with enough breaches to test on a few years of data.
DEFAULT_LEVELS = (0.95, 0.99)

#: Minimum observations before a conditional quantile means anything. A 99%
#: VaR from 40 turbulent days is one order statistic and pure noise.
MIN_REGIME_OBS = 100


def var_es(returns: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Historical VaR and Expected Shortfall, both as negative numbers.

    Historical rather than parametric on purpose: a Gaussian VaR on equity
    returns understates the tail precisely when it matters, and the whole
    point here is the tail.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 20:
        return float("nan"), float("nan")
    var = float(np.percentile(returns, 100 * (1 - level)))
    tail = returns[returns <= var]
    return var, float(tail.mean()) if len(tail) else var


def kupiec_pof(breaches: np.ndarray, level: float = 0.95) -> dict:
    """Unconditional coverage: are there the RIGHT NUMBER of breaches?

    A 95% VaR should be exceeded on about 5% of days. Too many and risk is
    understated; too few and capital is being wasted on a limit nobody
    approaches. Kupiec's proportion-of-failures test is the likelihood-ratio
    version of that comparison, distributed chi-squared with 1 dof.
    """
    breaches = np.asarray(breaches).astype(bool)
    n, x = len(breaches), int(breaches.sum())
    p = 1 - level
    if n == 0 or x == 0 or x == n:
        return {"n": n, "breaches": x, "rate": x / n if n else np.nan,
                "expected_rate": p, "lr": np.nan, "p_value": np.nan, "pass": None}
    lr = (-2 * (np.log(1 - p) * (n - x) + np.log(p) * x)
          + 2 * (np.log(1 - x / n) * (n - x) + np.log(x / n) * x))
    pv = float(1 - stats.chi2.cdf(lr, 1))
    return {"n": n, "breaches": x, "rate": round(x / n, 4), "expected_rate": p,
            "lr": round(float(lr), 3), "p_value": round(pv, 4), "pass": bool(pv > 0.05)}


def christoffersen_independence(breaches: np.ndarray) -> dict:
    """Do breaches CLUSTER? The test coverage alone cannot see.

    A VaR model can produce exactly the right number of breaches and still be
    useless if they all arrive in the same fortnight -- that is a model which
    ignores the regime and gets rescued by averaging. The test compares the
    probability of a breach following a breach against one following a calm
    day; under a good model they should be equal.
    """
    b = np.asarray(breaches).astype(int)
    if len(b) < 10:
        return {"lr": np.nan, "p_value": np.nan, "pass": None}
    prev, curr = b[:-1], b[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"lr": np.nan, "p_value": np.nan, "pass": None,
                "counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11}}
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi0, pi1 = n01 / (n00 + n01), n11 / (n10 + n11)

    def ll(p0, p1):
        with np.errstate(divide="ignore", invalid="ignore"):
            return (n00 * np.log(1 - p0) + n01 * np.log(p0)
                    + n10 * np.log(1 - p1) + n11 * np.log(p1))

    lr = -2 * (ll(pi, pi) - ll(pi0, pi1))
    pv = float(1 - stats.chi2.cdf(lr, 1)) if np.isfinite(lr) else np.nan
    return {"lr": round(float(lr), 3), "p_value": round(pv, 4),
            "pass": bool(pv > 0.05) if np.isfinite(pv) else None,
            "counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
            "p_breach_after_calm": round(pi0, 4), "p_breach_after_breach": round(pi1, 4)}


def walk_forward_var(returns: pd.Series, regime: pd.Series | None = None,
                     level: float = 0.95, min_train: int = 500,
                     step: int = 21) -> pd.DataFrame:
    """VaR for each day, estimated only from days before it.

    Re-estimated every `step` days on an expanding window rather than once on
    a single split. A single split reports how one arbitrary training period
    happened to generalise; walk-forward reports how the METHOD behaves, and
    is the convention for VaR backtesting.

    `regime` must already be causal. Passing smoothed HMM states here would
    silently leak the future into every quantile.
    """
    returns = returns.dropna()
    idx = returns.index
    rows = []
    for start in range(min_train, len(returns), step):
        end = min(start + step, len(returns))
        hist = returns.iloc[:start]
        if regime is None:
            q, _ = var_es(hist.to_numpy(), level)
            levels = {None: q}
        else:
            hist_reg = regime.reindex(hist.index)
            levels = {}
            for state in (False, True):
                sub = hist[hist_reg == state]
                levels[state] = (var_es(sub.to_numpy(), level)[0]
                                 if len(sub) >= MIN_REGIME_OBS
                                 else var_es(hist.to_numpy(), level)[0])
        for i in range(start, end):
            state = None if regime is None else bool(regime.reindex([idx[i]]).iloc[0])
            rows.append({"date": idx[i], "actual": float(returns.iloc[i]),
                         "var": float(levels[state]), "regime": state})
    frame = pd.DataFrame(rows).set_index("date")
    frame["breach"] = frame["actual"] < frame["var"]
    return frame


def backtest(returns: pd.Series, regime: pd.Series | None = None,
             level: float = 0.95, **kwargs) -> dict:
    """Walk-forward VaR plus both coverage tests."""
    frame = walk_forward_var(returns, regime, level=level, **kwargs)
    breaches = frame["breach"].to_numpy()
    out = {
        "level": level,
        "conditional": regime is not None,
        "kupiec": kupiec_pof(breaches, level),
        "christoffersen": christoffersen_independence(breaches),
        "mean_var": round(float(frame["var"].mean()), 5),
        "worst_loss": round(float(frame["actual"].min()), 5),
    }
    # Both tests together: a model must get the COUNT and the TIMING right.
    kp = out["kupiec"].get("p_value")
    cp = out["christoffersen"].get("p_value")
    if kp is not None and cp is not None and np.isfinite(kp) and np.isfinite(cp):
        lr_cc = out["kupiec"]["lr"] + out["christoffersen"]["lr"]
        out["conditional_coverage"] = {
            "lr": round(lr_cc, 3),
            "p_value": round(float(1 - stats.chi2.cdf(lr_cc, 2)), 4),
        }
    if regime is not None:
        for state, label in ((False, "quiet"), (True, "turbulent")):
            sub = frame[frame["regime"] == state]
            if len(sub):
                out[label] = {
                    "n_days": len(sub), "mean_var": round(float(sub["var"].mean()), 5),
                    "breach_rate": round(float(sub["breach"].mean()), 4),
                    "realised_vol_ann": round(float(sub["actual"].std() * np.sqrt(252)), 4),
                }
    return out


def risk_summary(returns: pd.Series, regime: pd.Series, level: float = 0.95) -> dict:
    """Today's risk numbers, for the chatbot. No model fitting here.

    Returns what a user actually asks for -- the current state, the loss
    level that state implies, and how it compares with normal.
    """
    current = bool(regime.iloc[-1])
    hist_reg = regime.reindex(returns.index)
    quiet_var, quiet_es = var_es(returns[hist_reg == False].to_numpy(), level)   # noqa: E712
    turb_var, turb_es = var_es(returns[hist_reg == True].to_numpy(), level)      # noqa: E712
    var, es = (turb_var, turb_es) if current else (quiet_var, quiet_es)
    return {
        "as_of": str(returns.index[-1].date()),
        "regime": "turbulent" if current else "quiet",
        "confidence_level": level,
        "var_pct": round(var * 100, 2),
        "expected_shortfall_pct": round(es * 100, 2),
        "quiet_var_pct": round(quiet_var * 100, 2),
        "turbulent_var_pct": round(turb_var * 100, 2),
        "ratio": round(turb_var / quiet_var, 2) if quiet_var else np.nan,
        "share_of_days_turbulent": round(float(hist_reg.mean()), 4),
    }


if __name__ == "__main__":
    import json

    from quant.data import load_panel
    from quant.regimes import build_observations, fit_regimes, turbulent_flag

    close, volume = load_panel()
    tv = (close * volume).rolling(21, min_periods=10).mean()
    panel = close[tv.mean().nlargest(50).index].pct_change()
    returns = panel.mean(axis=1).dropna()

    print("fitting regimes...")
    obs = build_observations(returns, panel_returns=panel)
    model, order = fit_regimes(obs)
    regime = turbulent_flag(model, order, obs, use_filtered=True)
    aligned = returns.reindex(obs.index)

    for level in DEFAULT_LEVELS:
        print(f"\n{'=' * 66}\n  {level:.0%} VaR, walk-forward\n{'=' * 66}")
        for label, reg in (("UNCONDITIONAL", None), ("REGIME-CONDITIONAL", regime)):
            res = backtest(aligned, reg, level=level)
            k, c = res["kupiec"], res["christoffersen"]
            print(f"  {label:20} breach {k['rate']:.2%} (target {k['expected_rate']:.2%})  "
                  f"Kupiec p={k['p_value']}  Christoffersen p={c['p_value']}")
            if "conditional_coverage" in res:
                print(f"  {'':20} joint coverage p={res['conditional_coverage']['p_value']}")

    print(f"\n{'=' * 66}\n  today\n{'=' * 66}")
    print(json.dumps(risk_summary(aligned, regime), indent=2))


# ==========================================================================
# Using the MARKOV structure, not just the regime label
# ==========================================================================
#
# `backtest` above conditions on today's state and stops there, which throws
# away the part of the model that makes it a Markov chain: the transition
# matrix. Two things it buys.
#
# 1. TOMORROW IS NOT TODAY. A hard switch says "today is quiet, use the quiet
#    VaR", ignoring that there is a real probability of flipping overnight.
#    The chain gives that probability, so the loss distribution becomes a
#    MIXTURE weighted by where the state will be, not where it is.
#
# 2. MULTI-DAY HORIZONS WITHOUT THE SQUARE-ROOT LIE. Basel wants 10-day VaR.
#    The usual shortcut multiplies the 1-day number by sqrt(10), which
#    assumes returns are independent -- and volatility clustering is the
#    precise statement that they are not. A Markov chain propagates the state
#    distribution forward with matrix powers, so a 10-day horizon can account
#    for the chance of ENTERING a crisis partway through and staying there.


def propagate_state(transmat: np.ndarray, state_probs: np.ndarray,
                    horizon: int) -> np.ndarray:
    """Where will the regime be in `horizon` days? pi_t @ T^h."""
    return np.asarray(state_probs, dtype=float) @ np.linalg.matrix_power(
        np.asarray(transmat, dtype=float), horizon)


def mixture_var_es(returns_by_state: dict, weights: dict, level: float = 0.95,
                   grid: int = 2000) -> tuple[float, float]:
    """VaR and ES of a probability-weighted MIXTURE of state distributions.

    Not the weighted average of the per-state VaRs -- that is a common error.
    The quantile of a mixture is not the mixture of the quantiles, so the
    mixture CDF is evaluated directly and inverted on a grid.
    """
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in returns_by_state.values()])
    lo, hi = float(pooled.min()), float(pooled.max())
    xs = np.linspace(lo, hi, grid)
    cdf = np.zeros_like(xs)
    for state, sample in returns_by_state.items():
        sample = np.sort(np.asarray(sample, dtype=float))
        if len(sample) == 0:
            continue
        cdf += weights.get(state, 0.0) * np.searchsorted(sample, xs, side="right") / len(sample)
    target = 1 - level
    var = float(xs[np.searchsorted(cdf, target)]) if cdf[-1] >= target else lo
    # ES: probability-weighted mean of each state's losses beyond the shared VaR.
    num = den = 0.0
    for state, sample in returns_by_state.items():
        sample = np.asarray(sample, dtype=float)
        tail = sample[sample <= var]
        w = weights.get(state, 0.0)
        num += w * tail.sum() / max(len(sample), 1)
        den += w * len(tail) / max(len(sample), 1)
    return var, float(num / den) if den > 0 else var


def markov_var(returns: pd.Series, regime: pd.Series, model, order,
               state_probs: np.ndarray, horizon: int = 1,
               level: float = 0.95, n_paths: int = 20000,
               seed: int = 0) -> dict:
    """VaR over `horizon` days, propagating the regime through the chain.

    For one day this is the mixture above. For longer horizons the returns
    compound and the state can move partway through, so the honest route is
    simulation: sample a state PATH from the transition matrix, draw a return
    from that state's empirical distribution each day, compound, repeat.

    Empirical draws rather than Gaussian ones, so the fat tails that make
    equity VaR interesting survive the simulation.
    """
    rng = np.random.default_rng(seed)
    hist = regime.reindex(returns.index)
    by_state = {False: returns[hist == False].to_numpy(),   # noqa: E712
                True: returns[hist == True].to_numpy()}     # noqa: E712
    if min(len(v) for v in by_state.values()) < 20:
        var, es = var_es(returns.to_numpy(), level)
        return {"horizon": horizon, "var": var, "expected_shortfall": es,
                "method": "unconditional fallback -- too few observations in one state"}

    # Collapse the HMM's states onto quiet/turbulent, matching turbulent_flag.
    trans = np.asarray(model.transmat_, dtype=float)
    quiet_idx, turb_idx = int(order[0]), [int(s) for s in order[1:]]
    collapsed = np.array([
        [trans[quiet_idx, quiet_idx], trans[quiet_idx, turb_idx].sum()],
        [trans[turb_idx, quiet_idx].mean(), trans[np.ix_(turb_idx, turb_idx)].sum(axis=1).mean()],
    ])
    collapsed /= collapsed.sum(axis=1, keepdims=True)

    probs = np.asarray(state_probs, dtype=float)
    if horizon == 1:
        nxt = probs @ collapsed
        var, es = mixture_var_es(by_state, {False: nxt[0], True: nxt[1]}, level)
        return {"horizon": 1, "var": round(var, 5), "expected_shortfall": round(es, 5),
                "p_turbulent_next": round(float(nxt[1]), 4), "method": "markov mixture"}

    states = rng.random(n_paths) > probs[0]
    total = np.zeros(n_paths)
    for _ in range(horizon):
        p_turb = np.where(states, collapsed[1, 1], collapsed[0, 1])
        states = rng.random(n_paths) < p_turb
        draws = np.where(
            states,
            rng.choice(by_state[True], n_paths),
            rng.choice(by_state[False], n_paths),
        )
        total += np.log1p(draws)
    horizon_returns = np.expm1(total)
    var = float(np.percentile(horizon_returns, 100 * (1 - level)))
    es = float(horizon_returns[horizon_returns <= var].mean())
    one_day, _ = var_es(returns.to_numpy(), level)
    return {
        "horizon": horizon, "var": round(var, 5), "expected_shortfall": round(es, 5),
        "sqrt_rule_var": round(one_day * np.sqrt(horizon), 5),
        # How badly the independence assumption misprices this horizon.
        "sqrt_rule_error_pct": round(100 * (var / (one_day * np.sqrt(horizon)) - 1), 1),
        "p_turbulent_at_horizon": round(float(propagate_state(collapsed, probs, horizon)[1]), 4),
        "method": f"markov simulation, {n_paths:,} paths",
    }
