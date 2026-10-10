"""Does combining the models beat the best single one?

THE QUESTION. A Model Confidence Set leaves four HAR variants that cannot be
separated statistically. The obvious next move is to combine them, and the
volatility literature supports it -- the foundation-model study found an
equal-weight ensemble entering the confidence set for 98-100% of assets
where single models managed 86-98%. This project also has direct evidence
from its own anomaly work: averaging every detector's rank matched the best
single detector on Kepler WITHOUT needing to know which detector was best.

FOUR COMBINATIONS, IN INCREASING ORDER OF AMBITION.

  mean_mcs      equal weights over the confidence-set members only
  mean_all      equal weights over every model, good and bad alike
  inv_loss      weights proportional to 1/QLIKE measured on TRAINING data
  stack_ols     a meta-model: regress the target on the member forecasts,
                fitted walk-forward. This is real stacking rather than
                averaging, and it can learn that one member is systematically
                biased and correct it.

EVERY WEIGHT IS CAUSAL. The combination weights at day t are fitted only on
rows whose target falls on or before t. Equal weights need no fitting and
are therefore safe by construction; the inverse-loss and OLS weights are
refitted on the expanding window exactly like the base models, because
weights chosen with hindsight are the oldest way to make an ensemble look
good.

A NEGATIVE CONSTRAINT WORTH STATING: the stack is given only the base
forecasts, never the raw features. If it had the features it could simply
re-fit HAR and the comparison would be circular.

Run:  python -m quant.stacking
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import MIN_TRAIN, REFIT_EVERY, RESULTS, mse, qlike, qlike_series

MCS_MEMBERS = ["har", "har_l", "har_iv", "har_ovn"]
CACHE = RESULTS / "garch_carr_forecasts.parquet"


def _ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


def _apply(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def build_base(estimator: str = "parkinson", verbose: bool = True) -> pd.DataFrame:
    """Every base model's walk-forward forecast, on one aligned frame."""
    from quant.vol_models import FEATURE_SETS, build_features, walk_forward

    frame = build_features(estimator, verbose=verbose)
    out = frame[["date", "ticker", "target", "naive"]].copy()
    for m in FEATURE_SETS:
        if verbose:
            print(f"    base {m} ...", flush=True)
        out[m] = walk_forward(frame, m)

    if CACHE.exists():
        cached = pd.read_parquet(CACHE)
        keep = [c for c in ("garch_mz", "gjr_garch_mz", "carr") if c in cached]
        out = out.merge(cached[["date", "ticker"] + keep], on=["date", "ticker"], how="left")
        if verbose:
            print(f"    merged cached {keep} (saves ~83 min of MLE)", flush=True)
    return out


def combine(base: pd.DataFrame, members: list, min_train: int = MIN_TRAIN,
            refit_every: int = REFIT_EVERY, verbose: bool = True) -> pd.DataFrame:
    """Add the four combination forecasts, all fitted causally."""
    df = base.dropna(subset=members + ["target"]).sort_values("date").reset_index(drop=True)
    M = df[members].to_numpy(float)
    y = df["target"].to_numpy(float)
    dates = df["date"].to_numpy()

    df["mean_mcs"] = df[[m for m in MCS_MEMBERS if m in members]].mean(axis=1)
    df["mean_all"] = M.mean(axis=1)

    inv = np.full(len(df), np.nan)
    stk = np.full(len(df), np.nan)
    uniq = np.sort(np.unique(dates))
    if len(uniq) > min_train:
        w, beta, last = None, None, -10 ** 9
        for di in range(min_train, len(uniq)):
            day = uniq[di]
            if w is None or (di - last) >= refit_every:
                tr = dates < day
                if tr.sum() < 200:
                    continue
                # inverse-loss weights from training QLIKE only
                losses = np.array([qlike(y[tr], M[tr, j]) for j in range(M.shape[1])])
                losses = np.where(np.isfinite(losses) & (losses > 0), losses, np.inf)
                w = (1.0 / losses) / np.sum(1.0 / losses)
                beta = _ols(M[tr], y[tr])
                last = di
            m = dates == day
            if not m.any() or w is None:
                continue
            inv[m] = M[m] @ w
            stk[m] = _apply(beta, M[m])
        if verbose and w is not None:
            print("    final inverse-loss weights: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in zip(members, w)), flush=True)
            print("    final stack coefficients  : intercept=%.4f, " % beta[0]
                  + ", ".join(f"{k}={v:+.3f}" for k, v in zip(members, beta[1:])), flush=True)
    df["inv_loss"] = inv
    df["stack_ols"] = stk
    return df


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="parkinson")
    args = ap.parse_args()

    from quant.mcs import model_confidence_set

    base = build_base(args.estimator)
    members = [c for c in base.columns
               if c not in ("date", "ticker", "target")]
    members = [m for m in members if base[m].notna().sum() > 1000]
    print(f"  base models: {members}")

    df = combine(base, members)
    combos = ["mean_mcs", "mean_all", "inv_loss", "stack_ols"]
    allm = members + combos
    sc = df.dropna(subset=allm).reset_index(drop=True)
    a = sc["target"].to_numpy(float)

    summary = {m: {"qlike": qlike(a, sc[m].to_numpy(float)),
                   "mse": mse(a, sc[m].to_numpy(float))} for m in allm}
    mcs = model_confidence_set({m: qlike_series(a, sc[m].to_numpy(float)) for m in allm},
                               verbose=True)

    best_single = min((m for m in members), key=lambda m: summary[m]["qlike"])
    best_combo = min(combos, key=lambda m: summary[m]["qlike"])

    print(f"\n{'=' * 58}\nCOMBINATIONS vs SINGLE MODELS   {len(sc):,} forecasts\n{'=' * 58}")
    print(f"  {'model':<16}{'QLIKE':>10}{'MSE':>11}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
        tag = "  <- MCS" if m in mcs["mcs"] else ""
        kind = " [combo]" if m in combos else ""
        print(f"  {m:<16}{v['qlike']:>10.5f}{v['mse']:>11.6f}{tag}{kind}")
    gain = summary[best_single]["qlike"] - summary[best_combo]["qlike"]
    print(f"\n  best single : {best_single} {summary[best_single]['qlike']:.5f}")
    print(f"  best combo  : {best_combo} {summary[best_combo]['qlike']:.5f}")
    print(f"  combining {'HELPS' if gain > 0 else 'does NOT help'} by {gain:+.5f}")
    print(f"  MCS (95%): {mcs['mcs']}")

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "n_forecasts": int(len(sc)),
           "base_models": members, "combinations": combos,
           "summary": summary, "mcs": mcs,
           "best_single": best_single, "best_combination": best_combo,
           "combination_gain_qlike": gain}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_stacking.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
