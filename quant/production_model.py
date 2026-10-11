"""The model the system actually serves: fit once, predict any ticker.

WHY A SEPARATE MODULE. ens_log is an ensemble of six learners POOLED across
48 tickers. The benchmark fits it walk-forward, refitting annually, which is
right for measuring it and impossible inside a web request. Fitting it on a
single ticker instead would be a different and worse model than the one
measured -- the whole point of pooling is the ~60,000 observations.

So the trained ensemble is persisted and loaded for inference. Two things
follow, and both matter:

  the SKILL VALUE TRANSFERS, because this is the same specification,
  features and target as the benchmarked model rather than a re-implementation

  the FIT IS FROZEN at a known date, so a forecast made later is genuinely
  out of sample with respect to the training data, and the staleness is
  visible rather than hidden

WHAT IT PREDICTS AND WHAT IT DOES NOT. Log realized volatility for the next
session, exponentiated with a smearing correction. Measured on 61,303
held-out forecasts: 70.9% directional accuracy against a simulated ceiling
of 74.1%, within-ticker R-squared 0.247 against a ceiling of 0.533, and it
beats the stock's own trailing 60-day average on 50 of 50 tickers. It says
how BIG tomorrow's move will be, never which way the price goes.

THE MARKET FACTOR NEEDS THE PANEL. Two of the ten features are
cross-sectional -- the market median volatility and this ticker's level
relative to it -- so a forecast for any single name still requires the
panel. That is a real dependency and the reason this cannot be a pure
single-ticker function.

Run:  python -m quant.production_model --fit        (train and persist)
      python -m quant.production_model --ticker TCS.NS
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent.parent / "data" / "cache" / "models"
MODEL_PATH = MODEL_DIR / "volatility_ens_log.joblib"
META_PATH = MODEL_DIR / "volatility_ens_log.json"

#: The six learners the benchmark averaged. Equal weights: the Model
#: Confidence Set could not separate them, and an estimated weighting lost
#: to a flat one (the forecast-combination puzzle showed up here too --
#: stack_ols scored 0.42399 against inv_loss's 0.40275).
MEMBERS = ("ridge_log", "rf_log", "et_log", "gbm_log", "svr_log", "mlp_log")
SVR_SUBSAMPLE = 12000


def fit_production(estimator: str = "garman_klass", verbose: bool = True) -> dict:
    """Train every member on all available history and persist them."""
    import warnings

    import joblib

    warnings.filterwarnings("ignore")
    from quant.tracking_models import FEATURES, _make, build_tracking_frame

    f = build_tracking_frame(estimator, verbose=verbose)
    X = f[FEATURES].to_numpy(float)
    y = f["y"].to_numpy(float)
    if verbose:
        print(f"  training on {len(f):,} rows, {f['ticker'].nunique()} tickers, "
              f"{f['date'].min().date()} -> {f['date'].max().date()}")

    rng = np.random.default_rng(0)
    fitted, resid = {}, []
    for kind in MEMBERS:
        idx = np.arange(len(X))
        if kind == "svr_log" and len(idx) > SVR_SUBSAMPLE:
            idx = rng.choice(idx, SVR_SUBSAMPLE, replace=False)
        m = _make(kind).fit(X[idx], y[idx])
        fitted[kind] = m
        resid.append(y - m.predict(X))
        if verbose:
            print(f"    fitted {kind}", flush=True)

    # One smearing variance for the ensemble, from the ensemble's own
    # residuals -- exponentiating a log-space fit without it is biased low.
    ens_resid = y - np.mean([m.predict(X) for m in fitted.values()], axis=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"members": fitted, "features": list(FEATURES),
                 "smear_var": float(np.var(ens_resid))}, MODEL_PATH)

    meta = {"fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "estimator": estimator, "n_rows": int(len(f)),
            "n_tickers": int(f["ticker"].nunique()),
            "train_last_date": str(f["date"].max().date()),
            "members": list(MEMBERS), "features": list(FEATURES),
            "measured_skill": {
                "directional_accuracy": 0.709, "directional_ceiling": 0.741,
                "within_ticker_r2_log": 0.247, "within_ticker_ceiling": 0.533,
                "pooled_r2_log": 0.452, "qlike": 0.3517,
                "beats_trailing_mean_on": "50 of 50 tickers",
                "n_heldout_forecasts": 61303}}
    META_PATH.write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"  saved -> {MODEL_PATH.name}")
    return meta


def _load():
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} not found. Run `python -m quant.production_model --fit`.")
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


def forecast(ticker: str, estimator: str = "garman_klass") -> dict:
    """Next-session volatility for one ticker from the persisted ensemble."""
    import warnings

    warnings.filterwarnings("ignore")
    bundle, meta = _load()
    from quant.tracking_models import build_tracking_frame

    # require_target=False keeps the latest row, whose target is tomorrow and
    # therefore unknown. Serving from the last row with a KNOWN target would
    # forecast a day already observed, from a row the model trained on.
    f = build_tracking_frame(estimator, verbose=False, require_target=False)
    g = f[f["ticker"] == ticker].sort_values("date")
    if g.empty:
        raise ValueError(
            f"{ticker!r} is not in the panel. The model uses cross-sectional "
            "features, so it serves the 48 benchmarked names only.")

    latest = g.iloc[-1]
    is_live = bool(pd.isna(latest.get("target")))
    row = g[bundle["features"]].to_numpy(float)[-1:]
    preds = {k: float(m.predict(row)[0]) for k, m in bundle["members"].items()}
    lp = float(np.mean(list(preds.values())))
    point = float(np.exp(lp + 0.5 * bundle["smear_var"]))
    today = float(g["naive"].iloc[-1])

    sd = float(np.sqrt(bundle["smear_var"]))
    return {
        "ticker": ticker, "model": "ens_log",
        "as_of": str(pd.Timestamp(g["date"].iloc[-1]).date()),
        "forecast_annualised_vol": point,
        "today_realised_vol": today,
        "direction": "higher" if point > today else "lower",
        "interval_80": [float(np.exp(lp - 1.2816 * sd)), float(np.exp(lp + 1.2816 * sd))],
        "member_forecasts": {k: float(np.exp(v + 0.5 * bundle["smear_var"]))
                             for k, v in preds.items()},
        "trained_through": meta["train_last_date"],
        "is_live": is_live,
        "note": ("forecasting an unobserved day" if is_live else
                 "WARNING: the latest feature row already has a known outcome, "
                 "so this is not a live forecast -- refresh the price panel"),
        "skill": meta["measured_skill"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--ticker")
    ap.add_argument("--estimator", default="garman_klass")
    args = ap.parse_args()

    if args.fit:
        m = fit_production(args.estimator)
        print(json.dumps(m["measured_skill"], indent=2))
        return
    if args.ticker:
        r = forecast(args.ticker, args.estimator)
        print(f"{r['ticker']}  as of {r['as_of']}  (model trained through "
              f"{r['trained_through']}, live={r['is_live']})")
        print(f"  today realised      {r['today_realised_vol']:.2%}")
        print(f"  forecast next day   {r['forecast_annualised_vol']:.2%}  "
              f"-> {r['direction']}")
        print(f"  80% interval        {r['interval_80'][0]:.2%} to "
              f"{r['interval_80'][1]:.2%}")
        print("  members: " + ", ".join(f"{k.replace('_log','')} {v:.1%}"
                                        for k, v in r["member_forecasts"].items()))
        return
    ap.error("pass --fit or --ticker")


if __name__ == "__main__":
    main()
