"""Does news explain where the volatility model goes wrong?

The served model reacts to spikes instead of anticipating them, which is the
expected behaviour of anything that sees only prices: most spikes are news.
This tests whether the combined news layer (finance/news_history.py) carries
information the model lacks, in the same three steps as the price-pattern
test in evaluation/volatility_patterns.py:

  1. raw         how much does volatility move on news days
  2. residuals   on news days, is the model's out-of-sample error biased? A
                 bias is information left on the table
  3. augmented   add the news as features to the ridge member, same
                 walk-forward protocol, and compare

TIMING. The forecast for day t+1 uses news known by the end of day t, plus
results dates, which are announced to the exchange in advance.

COVERAGE. NSE filings and the results calendar cover 2019 onwards; Times of
India headlines end in 2022. Every result names the rows it was measured on,
so a headline effect measured on 2021-2022 is not presented as 2019-2026.

Run:  python -m evaluation.news_volatility
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "news_volatility.json"
NEWS_FEATURES = ["results_next_session", "results_today", "log_ann_material",
                 "has_toi", "toi_tone_filled", "log_toi_count"]


def build() -> tuple[pd.DataFrame, dict]:
    from evaluation.volatility_patterns import build as build_vol
    from finance import news_history as nh

    d = build_vol()
    symbols = sorted(t.replace(".NS", "") for t in d["ticker"].unique())
    raw = nh.build(symbols, verbose=False)
    news = nh.daily_features(raw, pd.DatetimeIndex(d["date"].unique()), symbols)
    d = d.merge(news, on=["date", "ticker"], how="left")
    d["log_ann_material"] = np.log1p(d["ann_material"].fillna(0))
    d["log_toi_count"] = np.log1p(d["toi_count"].fillna(0))
    d["has_toi"] = d["toi_tone"].notna().astype(float)
    d["toi_tone_filled"] = d["toi_tone"].fillna(0.0)
    for c in ("results_next_session", "results_today"):
        d[c] = d[c].fillna(0.0)
    return d, raw["coverage"]


def states(d: pd.DataFrame) -> dict:
    toi_window = d["date"] <= pd.Timestamp("2022-12-31")
    return {
        "results_next_session": d["results_next_session"] == 1,
        "results_today": d["results_today"] == 1,
        "material_filing_today": d["ann_material"] > 0,
        "3+_material_filings": d["ann_material"] >= 3,
        "toi_headline_today": d["toi_count"] > 0,
        "toi_negative_headline": d["toi_negative"] > 0,
        "toi_window_no_headline": toi_window & (d["toi_count"] == 0),
        "all": pd.Series(True, index=d.index),
    }


def residuals(d: pd.DataFrame) -> dict:
    from evaluation.volatility_patterns import _by_day

    e = d.dropna(subset=["lp"]).copy()
    e["resid"] = e["y"] - e["lp"]
    e["chg"] = e["y"] - e["l_rv1"]
    out = {}
    for k, m in states(e).items():
        if m.sum() < 30:
            out[k] = {"n": int(m.sum())}
            continue
        out[k] = {"n": int(m.sum()),
                  "mean_log_change_tomorrow": float(e.loc[m, "chg"].mean()),
                  "mean_abs_log_change": float(e.loc[m, "chg"].abs().mean()),
                  "model_bias_log": _by_day(e.loc[m, "resid"], e.loc[m, "date"])}
    return out


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from evaluation.volatility_patterns import augmented_ridge

    d, coverage = build()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "coverage": coverage, "n_rows": int(len(d)), "residuals": residuals(d),
           "ridge_calendar_and_filings": augmented_ridge(
               d, ["results_next_session", "results_today", "log_ann_material"]),
           "ridge_all_news": augmented_ridge(d, NEWS_FEATURES)}
    toi = d[d["date"] <= pd.Timestamp("2022-12-31")]
    if toi["has_toi"].sum() > 200:
        out["ridge_headlines_toi_window"] = augmented_ridge(
            toi, ["has_toi", "toi_tone_filled", "log_toi_count"])
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print("NEWS AND VOLATILITY\ncoverage:", json.dumps(coverage))
    print(f"\n1-2. ON NEWS DAYS: how volatility moved, and the model's held-out error")
    print(f"   {'state':<26}{'n':>8}{'mean change':>13}{'mean |change|':>15}"
          f"{'model bias':>12}{'t':>7}{'p':>9}")
    for k, v in out["residuals"].items():
        if "model_bias_log" not in v:
            print(f"   {k:<26}{v['n']:>8}   too few")
            continue
        b = v["model_bias_log"]
        print(f"   {k:<26}{v['n']:>8,}{v['mean_log_change_tomorrow']:>+13.3f}"
              f"{v['mean_abs_log_change']:>15.3f}{b['mean']:>+12.3f}{b['t_stat']:>7.1f}"
              f"{b['p_value']:>9.3g}")
    print("\n3. RIDGE MEMBER WITH NEWS FEATURES (same walk-forward)")
    for key in ("ridge_calendar_and_filings", "ridge_all_news", "ridge_headlines_toi_window"):
        if key not in out:
            continue
        a = out[key]
        dm = a["diebold_mariano_aug_vs_base"]
        print(f"   {key:<28} within-R2 {a['base']['r2_within']:.4f} -> "
              f"{a['augmented']['r2_within']:.4f}   direction {a['base']['direction']:.2%} -> "
              f"{a['augmented']['direction']:.2%}   DM t={dm['stat']:.2f} p={dm['p_value']:.3g}"
              f"   ({a['n_forecasts']:,} forecasts)")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
