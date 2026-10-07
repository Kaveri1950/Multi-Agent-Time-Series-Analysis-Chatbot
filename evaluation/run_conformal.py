"""Conformal intervals against the model's own intervals, on the same days.

For every ticker in the prospective log, the GP and ARIMA point forecasts are
re-wrapped in an Adaptive Conformal interval and scored on the identical
observations the parametric interval was scored on. Coverage AND width are
reported for both, because coverage alone can be bought with width.

Run:  python -m evaluation.run_conformal
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"


def main():
    warnings.filterwarnings("ignore")
    from finance.forecast_log import load_log
    from quant.conformal import GAMMA, TARGET_ALPHA, WARMUP, aci_intervals

    log = load_log()
    log = log[log["actual_close"].notna()].sort_values(["ticker", "as_of_date"])

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "method": {"alpha_target": TARGET_ALPHA, "gamma": GAMMA, "warmup": WARMUP,
                      "reference": "Gibbs & Candes 2021, adaptive conformal inference"},
           "models": {}}

    for model, pcol, lo_col, hi_col in (
            ("gpr", "gpr_pred", "gpr_lower80", "gpr_upper80"),
            ("arima", "arima_pred", "arima_lower80", "arima_upper80")):
        conf_rows, para_rows = [], []
        for ticker, g in log.groupby("ticker"):
            g = g[g[pcol].notna()]
            if len(g) < WARMUP + 5:
                continue
            res = aci_intervals(g[pcol].to_numpy(), g["actual_close"].to_numpy())
            res["ticker"] = ticker
            # the parametric interval, scored on exactly the same rows
            par = pd.DataFrame({
                "covered": ((g["actual_close"] >= g[lo_col]) &
                            (g["actual_close"] <= g[hi_col])).to_numpy().astype(float),
                "width": (g[hi_col] - g[lo_col]).to_numpy(),
            })
            par.loc[res["covered"].isna().to_numpy(), :] = np.nan   # same warm-up
            conf_rows.append(res)
            para_rows.append(par)

        if not conf_rows:
            print(f"{model}: not enough history per ticker")
            continue
        conf = pd.concat(conf_rows, ignore_index=True).dropna(subset=["covered"])
        para = pd.concat(para_rows, ignore_index=True).dropna(subset=["covered"])

        entry = {
            "n": int(len(conf)),
            "conformal": {"coverage": float(conf["covered"].mean()),
                          "mean_width": float(conf["width"].mean())},
            "parametric": {"coverage": float(para["covered"].mean()),
                           "mean_width": float(para["width"].mean())},
            "target_coverage": 1.0 - TARGET_ALPHA,
        }
        entry["width_ratio_conformal_over_parametric"] = (
            entry["conformal"]["mean_width"] / entry["parametric"]["mean_width"])
        out["models"][model] = entry

        print(f"\n{model.upper()}  ({entry['n']} scored days, target coverage "
              f"{entry['target_coverage']:.0%})")
        print(f"  {'interval':<14}{'coverage':>10}{'mean width':>13}")
        print(f"  {'parametric':<14}{entry['parametric']['coverage']:>10.4f}"
              f"{entry['parametric']['mean_width']:>13.3f}")
        print(f"  {'conformal':<14}{entry['conformal']['coverage']:>10.4f}"
              f"{entry['conformal']['mean_width']:>13.3f}")
        print(f"  width ratio conformal/parametric: "
              f"{entry['width_ratio_conformal_over_parametric']:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "conformal_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
