"""Reversal and support/resistance in VOLATILITY: real, and already used?

Two trader intuitions, applied to volatility rather than price:

  REVERSAL      "volatility rose yesterday, so today it will fall"
  RESISTANCE    "volatility pushes through its recent high, then falls back"
                -- and a break below the recent low bounces

For price both are absent (evaluation/price_patterns.py). For volatility
they should be strong, for a reason this project has already measured: 92%
of the day-to-day change in observed volatility is estimation noise, and a
noisy series snaps back after every jump.

So the useful question is not whether the patterns exist but whether the
served model ALREADY knows them. Three tests, in increasing strength:

  1. the raw pattern       how often does volatility fall after it rose
  2. the model's residuals in each pattern state, is the out-of-sample
                           error biased? A bias means information left over
  3. an augmented model    add the pattern as features to the ridge member,
                           same walk-forward protocol, and compare

The ceiling applies throughout: direction 70.9% of an attainable 74.1%, and
within-ticker R2(log) 0.247 of 0.533. Any gain is measured against what is
left, not against zero.

Run:  python -m evaluation.volatility_patterns
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "volatility_patterns.json"
LOOKBACK = 20
PATTERN_FEATURES = ["d_rv1", "brk_hi", "brk_lo", "dist_hi", "dist_lo"]

#: The PRICE's position against its own 52-week range. Driessen, Lin and Van
#: Hemert (2013, US stocks 1963-2008) report idiosyncratic variance about 32%
#: lower as a stock approaches its 52-week high, and 46% / 111% higher the day
#: after it breaks the high / the low. Nothing in the served model sees price
#: levels, so this is information it could genuinely lack.
YEAR = 250
PRICE_LEVEL_FEATURES = ["near_hi52", "near_lo52", "breach_hi52", "breach_lo52"]


def build() -> pd.DataFrame:
    """Held-out ensemble forecasts joined to the features and pattern states."""
    from quant.tracking_models import build_tracking_frame

    f = build_tracking_frame("garman_klass", verbose=False, require_target=True)
    f = f.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = f.groupby("ticker")["l_rv1"]
    f["d_rv1"] = f["l_rv1"] - g.shift(1)                       # did it rise today
    hi = g.transform(lambda s: s.rolling(LOOKBACK).max().shift(1))
    lo = g.transform(lambda s: s.rolling(LOOKBACK).min().shift(1))
    f["brk_hi"] = (f["l_rv1"] > hi).astype(float)              # through "resistance"
    f["brk_lo"] = (f["l_rv1"] < lo).astype(float)              # through "support"
    f["dist_hi"] = f["l_rv1"] - hi
    f["dist_lo"] = f["l_rv1"] - lo
    from quant.volatility import OHLC_PATH

    px = pd.read_parquet(OHLC_PATH, columns=["date", "ticker", "close"])
    px = px.sort_values(["ticker", "date"])
    c = px.groupby("ticker")["close"]
    hi52 = c.transform(lambda s: s.rolling(YEAR, min_periods=YEAR).max().shift(1))
    lo52 = c.transform(lambda s: s.rolling(YEAR, min_periods=YEAR).min().shift(1))
    px["near_hi52"] = np.log(px["close"] / hi52)       # 0 at the high, negative below
    px["near_lo52"] = np.log(px["close"] / lo52)       # 0 at the low, positive above
    px["breach_hi52"] = (px["close"] > hi52).astype(float)
    px["breach_lo52"] = (px["close"] < lo52).astype(float)
    f = f.merge(px[["date", "ticker", *PRICE_LEVEL_FEATURES]], on=["date", "ticker"],
                how="left")
    p = pd.read_parquet(RESULTS / "volatility_tracking_predictions.parquet")
    d = f.merge(p[["date", "ticker", "ens_log"]], on=["date", "ticker"], how="left")
    # back out the log forecast: the parquet stores exp(lp + v/2)
    raw = d["y"] - np.log(d["ens_log"])
    d["lp"] = np.log(d["ens_log"]) - 0.5 * float(np.nanvar(raw))
    return d.dropna(subset=PATTERN_FEATURES + PRICE_LEVEL_FEATURES).reset_index(drop=True)


def _by_day(values: pd.Series, dates: pd.Series) -> dict:
    """Mean of a quantity, tested across days rather than stock-days."""
    from scipy import stats

    daily = values.groupby(dates.to_numpy()).mean().dropna()
    t = stats.ttest_1samp(daily, 0.0)
    return {"mean": float(values.mean()), "n_days": int(len(daily)),
            "t_stat": float(t.statistic), "p_value": float(t.pvalue)}


def raw_patterns(d: pd.DataFrame) -> dict:
    fell = d["y"] < d["l_rv1"]
    states = {"rose_today": d["d_rv1"] > 0, "fell_today": d["d_rv1"] < 0,
              "through_20d_high": d["brk_hi"] == 1, "through_20d_low": d["brk_lo"] == 1,
              "any_day": pd.Series(True, index=d.index)}
    return {k: {"n": int(m.sum()), "p_falls_tomorrow": float(fell[m].mean()),
                "mean_log_change_tomorrow": float((d["y"] - d["l_rv1"])[m].mean())}
            for k, m in states.items()}


def price_level_check(d: pd.DataFrame) -> dict:
    """The Driessen-Lin-Van Hemert claim on this panel, raw and against the model."""
    e = d.dropna(subset=["lp"]).copy()
    e["resid"] = e["y"] - e["lp"]
    e["chg"] = e["y"] - e["l_rv1"]
    near_hi = e["near_hi52"] > -0.02          # within 2% of the 52-week high
    states = {"breach_52w_high": e["breach_hi52"] == 1, "breach_52w_low": e["breach_lo52"] == 1,
              "within_2pct_of_high": near_hi & (e["breach_hi52"] == 0),
              "all": pd.Series(True, index=e.index)}
    out = {}
    for k, m in states.items():
        out[k] = {"n": int(m.sum()),
                  "mean_log_vol_vs_stock_norm": float((e["y"] - e["l_rv66"])[m].mean()),
                  "mean_log_change_tomorrow": float(e.loc[m, "chg"].mean()),
                  "mean_residual_log": _by_day(e.loc[m, "resid"], e.loc[m, "date"])}
    return out


def residual_check(d: pd.DataFrame) -> dict:
    """In each pattern state, is the model's held-out error biased?"""
    e = d.dropna(subset=["lp"]).copy()
    e["resid"] = e["y"] - e["lp"]
    right = (e["lp"] > e["l_rv1"]) == (e["y"] > e["l_rv1"])
    states = {"rose_today": e["d_rv1"] > 0, "fell_today": e["d_rv1"] < 0,
              "through_20d_high": e["brk_hi"] == 1, "through_20d_low": e["brk_lo"] == 1,
              "all": pd.Series(True, index=e.index)}
    out = {}
    for k, m in states.items():
        out[k] = {"n": int(m.sum()), "direction_accuracy": float(right[m].mean()),
                  "model_calls_fall": float((e["lp"] < e["l_rv1"])[m].mean()),
                  "actually_falls": float((e["y"] < e["l_rv1"])[m].mean()),
                  "mean_residual_log": _by_day(e.loc[m, "resid"], e.loc[m, "date"])}
    corr = {c: float(np.corrcoef(e[c], e["resid"])[0, 1]) for c in PATTERN_FEATURES}
    return {"n": int(len(e)), "states": out, "corr_residual_with_feature": corr}


def augmented_ridge(d: pd.DataFrame, extra: list | None = None) -> dict:
    """The ridge member with and without extra features, same protocol."""
    extra = extra or PATTERN_FEATURES
    from quant.mcs import model_confidence_set
    from quant.tracking_models import FEATURES, REFIT_DAYS, _make
    from quant.volatility import MIN_TRAIN, diebold_mariano

    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)
    dates = d["date"].to_numpy()
    uniq = np.sort(np.unique(dates))
    y = d["y"].to_numpy(float)

    def walk(cols):
        X = d[cols].to_numpy(float)
        pred = np.full(len(d), np.nan)
        model, last = None, -10 ** 9
        for di in range(MIN_TRAIN, len(uniq)):
            if model is None or di - last >= REFIT_DAYS:
                tr = dates < uniq[di]
                model, last = _make("ridge_log").fit(X[tr], y[tr]), di
            m = dates == uniq[di]
            pred[m] = model.predict(X[m])
        return pred

    base, aug = walk(list(FEATURES)), walk(list(FEATURES) + list(extra))
    ok = np.isfinite(base) & np.isfinite(aug)
    s = d[ok].copy()
    s["base"], s["aug"] = base[ok], aug[ok]

    def score(col):
        dm = s.groupby("ticker")[["y", col]].transform("mean")
        yy, pp = (s["y"] - dm["y"]).to_numpy(), (s[col] - dm[col]).to_numpy()
        return {"r2_within": float(1 - np.sum((yy - pp) ** 2) / np.sum(yy ** 2)),
                "direction": float(((s[col] > s["l_rv1"]) == (s["y"] > s["l_rv1"])).mean()),
                "mse_log": float(np.mean((s["y"] - s[col]) ** 2))}

    lb = ((s["y"] - s["base"]) ** 2).groupby(s["date"].to_numpy()).mean()
    la = ((s["y"] - s["aug"]) ** 2).groupby(s["date"].to_numpy()).mean()
    mcs = model_confidence_set({"base": (s["y"] - s["base"]).to_numpy() ** 2,
                                "augmented": (s["y"] - s["aug"]).to_numpy() ** 2},
                               verbose=False)
    return {"n_forecasts": int(len(s)), "base": score("base"), "augmented": score("aug"),
            "diebold_mariano_aug_vs_base": diebold_mariano(la.to_numpy(), lb.to_numpy()),
            "model_confidence_set": mcs.get("mcs")}


def main():
    import warnings

    warnings.filterwarnings("ignore")
    d = build()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_rows": int(len(d)), "raw": raw_patterns(d), "residuals": residual_check(d),
           "augmented_ridge": augmented_ridge(d),
           "price_levels": price_level_check(d),
           "augmented_ridge_price_levels": augmented_ridge(d, PRICE_LEVEL_FEATURES),
           "augmented_ridge_both": augmented_ridge(d, PATTERN_FEATURES + PRICE_LEVEL_FEATURES)}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"VOLATILITY PATTERNS  {out['n_rows']:,} stock-days\n")
    print("1. RAW PATTERN: how often does tomorrow's volatility come in below today's?")
    for k, v in out["raw"].items():
        print(f"   {k:<18} {v['p_falls_tomorrow']:6.1%}   mean log change "
              f"{v['mean_log_change_tomorrow']:+.3f}   (n={v['n']:,})")
    r = out["residuals"]
    print(f"\n2. DOES THE MODEL ALREADY KNOW?  ({r['n']:,} held-out forecasts)")
    print(f"   {'state':<18}{'model right':>12}{'calls fall':>12}{'falls':>8}"
          f"{'bias (log)':>12}{'t':>7}{'p':>8}")
    for k, v in r["states"].items():
        b = v["mean_residual_log"]
        print(f"   {k:<18}{v['direction_accuracy']:>12.1%}{v['model_calls_fall']:>12.1%}"
              f"{v['actually_falls']:>8.1%}{b['mean']:>+12.3f}{b['t_stat']:>7.1f}"
              f"{b['p_value']:>8.3f}")
    print("   correlation of held-out error with each pattern feature: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in r["corr_residual_with_feature"].items()))
    a = out["augmented_ridge"]
    print(f"\n3. RIDGE MEMBER WITH THE PATTERNS AS FEATURES  ({a['n_forecasts']:,} forecasts)")
    for k in ("base", "augmented"):
        v = a[k]
        print(f"   {k:<10} within-R2 {v['r2_within']:.4f}   direction {v['direction']:.2%}   "
              f"log-MSE {v['mse_log']:.5f}")
    dm = a["diebold_mariano_aug_vs_base"]
    print(f"   Diebold-Mariano (negative favours augmented) t={dm['stat']:.2f} "
          f"p={dm['p_value']:.3g}   confidence set {a['model_confidence_set']}")
    print("   ceilings: direction 74.1%, within-R2 0.533")
    print("\n4. PRICE AGAINST ITS 52-WEEK RANGE (Driessen, Lin & Van Hemert 2013)")
    print(f"   {'state':<22}{'n':>7}{'vol vs own norm':>17}{'change tmrw':>13}"
          f"{'model bias':>12}{'t':>7}{'p':>8}")
    for k, v in out["price_levels"].items():
        b = v["mean_residual_log"]
        print(f"   {k:<22}{v['n']:>7,}{v['mean_log_vol_vs_stock_norm']:>+17.3f}"
              f"{v['mean_log_change_tomorrow']:>+13.3f}{b['mean']:>+12.3f}{b['t_stat']:>7.1f}"
              f"{b['p_value']:>8.3f}")
    for key, label in (("augmented_ridge_price_levels", "52-week price levels"),
                       ("augmented_ridge_both", "both feature sets")):
        a = out[key]
        dm = a["diebold_mariano_aug_vs_base"]
        print(f"   ridge + {label:<20} within-R2 {a['base']['r2_within']:.4f} -> "
              f"{a['augmented']['r2_within']:.4f}   direction {a['base']['direction']:.2%} -> "
              f"{a['augmented']['direction']:.2%}   DM t={dm['stat']:.2f} p={dm['p_value']:.3g}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
