"""Chronos-Bolt, zero-shot, against HAR on realized volatility.

THE CLAIM BEING TESTED. Time-series foundation models are advertised as
forecasting a new series with no training at all. The published comparison
closest to this task -- realized volatility across 50 assets -- found that of
nine such models only one beat a Log-HAR baseline, and by 1.3 to 1.8%. This
checks whether that holds on Indian equities with a range-based proxy.

HOW IT IS KEPT FAIR. Chronos sees exactly the context HAR's longest
component sees and nothing more: the last `CONTEXT` observations of the same
realized-volatility series, ending the day before the one being forecast. It
is scored on the same days, with the same QLIKE loss, and enters the same
Model Confidence Set. It is zero-shot by design -- no fine-tuning -- which is
the setting the claim is made in.

COST IS PART OF THE COMPARISON. HAR refits in microseconds. A transformer
forward pass per forecast is several orders of magnitude more expensive, and
the registry records runtime as a capability, so the comparison is reported
in both skill and seconds.

Run:  python -m evaluation.run_chronos_volatility [--tickers 12]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
MODEL = "amazon/chronos-bolt-small"
CONTEXT = 64          # observations of history fed to the model
BATCH = 256


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=12,
                    help="how many tickers to score (a forward pass per forecast "
                         "is slow; 12 still gives ~15k forecasts)")
    ap.add_argument("--estimator", default="parkinson")
    args = ap.parse_args()

    import torch
    from chronos import BaseChronosPipeline

    from quant.mcs import model_confidence_set
    from quant.volatility import MODELS, forecast_frame, qlike, qlike_series

    allrec, _ = forecast_frame(estimator=args.estimator, verbose=True)
    keep = sorted(allrec["ticker"].unique())[: args.tickers]
    allrec = allrec[allrec["ticker"].isin(keep)].reset_index(drop=True)
    print(f"  scoring {len(keep)} tickers, {len(allrec):,} forecast days")

    # Rebuild each ticker's full RV series so a context window can be cut
    # ending the day BEFORE each forecast target.
    from quant.volatility import download_ohlc, realized_vol
    rv_all = realized_vol(download_ohlc(verbose=False), args.estimator)

    print(f"  loading {MODEL} ...", flush=True)
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu",
                                               torch_dtype=torch.float32)
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    preds = np.full(len(allrec), np.nan)
    t0 = time.time()
    done = 0
    for ticker, g in allrec.groupby("ticker"):
        series = rv_all[rv_all["ticker"] == ticker].set_index("date")["rv"]
        pos = {d: i for i, d in enumerate(series.index)}
        contexts, slots = [], []
        for ridx, date in zip(g.index, g["date"]):
            i = pos.get(date)
            if i is None or i < CONTEXT:
                continue
            # history ends at i-1: the target day itself is never in context
            contexts.append(torch.tensor(series.to_numpy()[i - CONTEXT:i],
                                         dtype=torch.float32))
            slots.append(ridx)
        for s in range(0, len(contexts), BATCH):
            chunk = contexts[s:s + BATCH]
            _, mean = pipe.predict_quantiles(chunk, prediction_length=1,
                                             quantile_levels=[0.1, 0.5, 0.9])
            preds[slots[s:s + BATCH]] = mean[:, 0].numpy()
            done += len(chunk)
        print(f"    {ticker:<16} {done:,} forecasts, {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    allrec["chronos"] = preds
    scored = allrec.dropna(subset=["chronos"]).reset_index(drop=True)
    models = list(MODELS) + ["chronos"]
    actual = scored["actual"].to_numpy()

    summary = {m: {"qlike_pooled": qlike(actual, scored[m].to_numpy())} for m in models}
    losses = {m: qlike_series(actual, scored[m].to_numpy()) for m in models}
    mcs = model_confidence_set(losses, verbose=True)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL, "context_length": CONTEXT, "zero_shot": True,
        "estimator": args.estimator,
        "n_tickers": int(scored["ticker"].nunique()),
        "n_forecasts": int(len(scored)),
        "seconds_total": round(elapsed, 1),
        "seconds_per_1000_forecasts": round(1000 * elapsed / max(len(scored), 1), 2),
        "summary": summary, "mcs": mcs,
    }

    print(f"\n{'=' * 66}\nCHRONOS-BOLT zero-shot vs HAR  ({len(scored):,} forecasts, "
          f"{out['n_tickers']} tickers)\n{'=' * 66}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike_pooled"]):
        print(f"  {m:<12} QLIKE {v['qlike_pooled']:.5f}")
    print(f"\n  MCS (95%): {mcs['mcs']}")
    print(f"  chronos cost: {elapsed:.0f}s for {len(scored):,} forecasts "
          f"({out['seconds_per_1000_forecasts']}s per 1000)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "chronos_volatility.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
