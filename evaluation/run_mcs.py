"""Model Confidence Set over the forecasting models, in both targets.

Answers a question the project could not answer before: which forecasters are
statistically indistinguishable from the best one? That set, rather than a
single winner, is what the registry should really carry -- picking one model
out of several that cannot be separated is a coin flip dressed up as a
measurement.

Two targets are run, and the contrast is the point:

  volatility   naive / HAR / Log-HAR on realized volatility, QLIKE loss
  price        naive / GPR / ARIMA on next-day close, absolute-error loss

Run:  python -m evaluation.run_mcs
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"


def volatility_losses(estimator: str = "parkinson") -> dict:
    from quant.volatility import forecast_frame, qlike_series

    allrec, _ = forecast_frame(estimator=estimator)
    actual = allrec["actual"].to_numpy()
    from quant.volatility import MODELS
    return ({m: qlike_series(actual, allrec[m].to_numpy()) for m in MODELS},
            {"n_tickers": int(allrec["ticker"].nunique()),
             "span": [str(allrec["date"].min())[:10], str(allrec["date"].max())[:10]]})


def price_losses() -> dict:
    """Absolute error on next-day close, from the prospective log.

    Rows where ARIMA failed to fit are dropped for EVERY model, not just for
    ARIMA: the MCS compares models on the same observations, and letting one
    model skip its hard days would flatter it.
    """
    from finance.forecast_log import load_log

    df = load_log()
    df = df[df["actual_close"].notna() & df["arima_pred"].notna()]
    actual = df["actual_close"].to_numpy(dtype=float)
    losses = {}
    for m, col in (("naive", "naive_pred"), ("gpr", "gpr_pred"), ("arima", "arima_pred")):
        losses[m] = np.abs(actual - df[col].to_numpy(dtype=float))
    return losses, {"n_tickers": int(df["ticker"].nunique()),
                    "span": [str(df["as_of_date"].min())[:10],
                             str(df["as_of_date"].max())[:10]],
                    "n_dropped_arima_failed": 2}


def main():
    warnings.filterwarnings("ignore")
    from quant.mcs import model_confidence_set

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "targets": {}}

    for label, loader, loss_name in (
            ("volatility", volatility_losses, "QLIKE"),
            ("price", price_losses, "absolute error")):
        try:
            losses, meta = loader()
        except Exception as exc:                      # noqa: BLE001
            print(f"{label}: could not build losses -- {type(exc).__name__}: {exc}")
            continue

        print(f"\n{'=' * 70}\n{label.upper()}  ({loss_name}, "
              f"{len(next(iter(losses.values()))):,} observations)\n{'=' * 70}")
        res = model_confidence_set(losses, verbose=True)
        res["loss"] = loss_name
        res.update(meta)
        out["targets"][label] = res

        print(f"  mean loss: " + "  ".join(
            f"{k}={v:.5g}" for k, v in sorted(res["mean_loss"].items(), key=lambda kv: kv[1])))
        print(f"  block length {res['block_length']}, {res['n_bootstrap']} resamples")
        for e in res["eliminated"]:
            print(f"  eliminated {e['model']:<10} t={e['t_stat']:+.3f}  p={e['p_value']:.4f}")
        print(f"  MCS (95%): {res['mcs']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "mcs_results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
