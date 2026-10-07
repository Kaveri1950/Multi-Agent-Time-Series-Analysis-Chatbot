"""Prospective price forecasts, logged before the outcome is known.

WHY THIS EXISTS. Every finance number this project reported was a backtest:
fit on history, evaluate on a held-out tail, report RMSE. That is legitimate
but weak, because the evaluation window was already in the data when the
model was designed, and nothing stops a choice made with hindsight from
flattering the result. Worse, the original comparison (ARIMA vs XGBoost vs
XGBoost+sentiment) omitted a null model, and once one was added it turned out
that a naive "tomorrow equals today" rule beats both on all 9 tickers tested,
with directional accuracy at 49.6% -- chance.

A prospective log fixes the evaluation, not the accuracy. Predictions are
written with tomorrow's date and no outcome. The truth arrives on its own the
next day. Nothing can be tuned against it, so whatever it says is real.

THREE PREDICTORS ARE STORED FOR EVERY ROW, deliberately:
  naive  -- tomorrow = today. The null model. Storing it beside every
            prediction makes the original omission impossible to repeat.
  gpr    -- Gaussian Process, which gives a calibrated predictive INTERVAL
            rather than only a point estimate.
  arima  -- already used elsewhere in the project, and it also emits an 80%
            interval that has never been checked for calibration.

THE RESULT THIS IS ACTUALLY FOR. Point accuracy is a lost cause: daily
direction is not predictable from past prices, and no model here will beat
the naive baseline. Calibration is a different question and a winnable one --
over N days, does the actual close fall inside the 80% interval about 80% of
the time? A model that cannot predict the price but correctly knows how
uncertain it is, is genuinely useful, and nothing in this project has ever
tested whether its confidence intervals hold up.

Mirrors solar/flare_data.harvest_daily() on purpose: the same
log-before-you-know discipline, now in a second domain.
"""

import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "finance"
LOG_PATH = CACHE_DIR / "forecast_log.parquet"

# The universe logged daily. Small on purpose -- a GP fit is O(n^3) and this
# runs once a day, so the cost is roughly a minute per ticker.
#: The 20 most liquid names by trailing traded value, plus the index. Widened
#: from 3 because coverage precision scales with sqrt(n): at 3 tickers the
#: 95% interval on an 80% coverage rate is +/-7.2 points and reaching +/-3.5
#: takes about eight months, whereas at 20 tickers it takes about five weeks.
#: The universe is the same liquidity ranking the rest of the project uses,
#: so this introduces no new selection choice.
DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
                   "BAJFINANCE.NS", "INFY.NS", "AXISBANK.NS", "BHARTIARTL.NS",
                   "TCS.NS", "KOTAKBANK.NS", "TATASTEEL.NS", "MARUTI.NS",
                   "INDUSINDBK.NS", "LT.NS", "ADANIENT.NS", "ITC.NS",
                   "HINDUNILVR.NS", "SUNPHARMA.NS", "TITAN.NS", "NESTLEIND.NS",
                   "^NSEI"]

# GP training window. 500 points took ~44s to fit; the cubic cost means 2000
# would take nearly an hour, and older data describes a different market
# anyway (tested: extending to 2000-2025 changed the naive-vs-model ratio by
# noise, not by improvement).
GP_TRAIN_POINTS = 400

INTERVAL = 0.80          # store an 80% predictive interval
HISTORY_START = "2023-01-01"


def _next_trading_day_after(d) -> pd.Timestamp:
    """The next business day. Indian market holidays are not modelled, so
    reconciliation matches the first ACTUAL close after the prediction date
    rather than trusting this exactly -- a holiday just shifts the fill."""
    return pd.Timestamp(d) + pd.offsets.BDay(1)


# --------------------------------------------------------------------------
# the predictors
# --------------------------------------------------------------------------

def _gpr_forecast(features: pd.DataFrame, n_train: int = GP_TRAIN_POINTS,
                  extra_features: list | None = None) -> dict:
    """One-step-ahead GP forecast with a predictive interval.

    A Matern kernel plus an explicit white-noise term. The noise term is the
    important part: it lets the GP attribute most of the variation to
    irreducible noise instead of fitting structure into it. That is why GPR
    lands within ~5-7% of the naive baseline where XGBoost is 40-60% worse --
    a well-specified GP degrades gracefully when there is no signal, and a
    boosted tree does not.
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    from finance.price_model import FEATURE_COLUMNS

    # Callers can append columns already present on the frame -- used to
    # add FinBERT sentiment without duplicating the whole fit.
    columns = list(FEATURE_COLUMNS)
    for col in (extra_features or []):
        if col in features.columns and col not in columns:
            columns.append(col)

    frame = features.copy()
    frame["target_next_close"] = frame["Close"].shift(-1)
    train = frame.dropna().tail(n_train)
    if len(train) < 60:
        raise ValueError(f"only {len(train)} usable rows for a GP fit")

    sx, sy = StandardScaler(), StandardScaler()
    x_train = sx.fit_transform(train[columns])
    y_train = sy.fit_transform(train[["target_next_close"]]).ravel()

    kernel = (ConstantKernel(1.0) * Matern(length_scale=np.ones(len(columns)), nu=1.5)
              + WhiteKernel(1e-2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                      n_restarts_optimizer=0).fit(x_train, y_train)

    # predict from the LAST available row, whose target is genuinely unknown
    x_now = sx.transform(features[columns].tail(1))
    mu, sd = gp.predict(x_now, return_std=True)

    scale = float(sy.scale_[0])
    centre = float(sy.inverse_transform(mu.reshape(-1, 1))[0, 0])
    sigma = float(sd[0]) * scale
    z = float(norm.ppf(0.5 + INTERVAL / 2))

    return {"pred": centre, "lower": centre - z * sigma, "upper": centre + z * sigma,
            "sigma": sigma, "n_train": len(train), "columns": columns}


def _arima_forecast(close: pd.Series) -> dict:
    """One-step ARIMA forecast, reusing the project's existing model."""
    from finance.price_model import forecast_future

    result = forecast_future(close, days=1)
    row = result["forecast"].iloc[0]
    return {"pred": float(row["predicted_close"]),
            "lower": float(row["lower_80"]), "upper": float(row["upper_80"])}


def predict_one(ticker: str) -> dict:
    """All three predictors for one ticker, for the next trading day."""
    from finance.price_data import add_technical_indicators, get_price_history

    raw = get_price_history(ticker, start=HISTORY_START)
    features = add_technical_indicators(raw).dropna()
    if features.empty:
        raise ValueError(f"no usable history for {ticker}")

    as_of = features.index[-1]
    close_today = float(features["Close"].iloc[-1])

    row = {
        "predicted_at": pd.Timestamp(date.today()),
        "ticker": ticker,
        "as_of_date": pd.Timestamp(as_of),
        "close_today": close_today,
        "target_date": _next_trading_day_after(as_of),
        # the null model, stored on every row so it can never be omitted again
        "naive_pred": close_today,
    }

    try:
        gp = _gpr_forecast(features)
        row.update({"gpr_pred": gp["pred"], "gpr_lower80": gp["lower"],
                    "gpr_upper80": gp["upper"], "gpr_sigma": gp["sigma"],
                    "gpr_n_train": gp["n_train"]})
    except Exception as e:
        row["gpr_error"] = f"{type(e).__name__}: {e}"

    try:
        ar = _arima_forecast(features["Close"])
        row.update({"arima_pred": ar["pred"], "arima_lower80": ar["lower"],
                    "arima_upper80": ar["upper"]})
    except Exception as e:
        row["arima_error"] = f"{type(e).__name__}: {e}"

    return row


# --------------------------------------------------------------------------
# the daily log
# --------------------------------------------------------------------------

def log_predictions(tickers: list | None = None, verbose: bool = True) -> pd.DataFrame:
    """Predict the next close for each ticker and append to the log.

    Idempotent on (ticker, target_date): running twice in a day replaces the
    row rather than duplicating it, so a cron that double-fires is harmless.
    Outcome columns are left empty -- reconcile() fills them once the day has
    actually happened.
    """
    tickers = tickers or DEFAULT_TICKERS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for ticker in tickers:
        try:
            rows.append(predict_one(ticker))
            if verbose:
                r = rows[-1]
                got = f"gpr {r.get('gpr_pred', float('nan')):.2f}" if "gpr_pred" in r else "gpr FAILED"
                print(f"  {ticker:<14} close {r['close_today']:>9.2f}  {got}  "
                      f"-> target {r['target_date'].date()}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  {ticker:<14} FAILED {type(e).__name__}: {e}", flush=True)

    if not rows:
        return load_log()

    fresh = pd.DataFrame(rows)
    combined = pd.concat([load_log(), fresh], ignore_index=True) if LOG_PATH.exists() else fresh
    combined = combined.drop_duplicates(subset=["ticker", "target_date"], keep="last")
    combined.to_parquet(LOG_PATH)
    if verbose:
        pending = combined["actual_close"].isna().sum() if "actual_close" in combined else len(combined)
        print(f"\n  log now {len(combined)} rows, {pending} awaiting an outcome")
    return combined


def reconcile(verbose: bool = True) -> pd.DataFrame:
    """Fill in what actually happened, for predictions whose day has passed.

    Matches the first real close AFTER the prediction's as_of_date rather than
    requiring an exact date hit -- market holidays shift the next session, and
    demanding an exact match would silently strand those rows forever.
    """
    from finance.price_data import get_price_history

    log = load_log()
    if log.empty:
        if verbose:
            print("  nothing logged yet")
        return log

    if "actual_close" not in log.columns:
        log["actual_close"] = np.nan
        log["actual_date"] = pd.NaT

    filled = 0
    for ticker, group in log[log["actual_close"].isna()].groupby("ticker"):
        try:
            prices = get_price_history(ticker, start=HISTORY_START)["Close"]
        except Exception as e:
            if verbose:
                print(f"  {ticker}: price fetch failed ({type(e).__name__})")
            continue
        for idx, row in group.iterrows():
            later = prices[prices.index > pd.Timestamp(row["as_of_date"])]
            if later.empty:
                continue                      # the day has not happened yet
            log.at[idx, "actual_close"] = float(later.iloc[0])
            log.at[idx, "actual_date"] = pd.Timestamp(later.index[0])
            filled += 1

    # derived columns, recomputed from scratch so they always match the data
    for name in ("gpr", "naive", "arima"):
        col = f"{name}_pred"
        if col not in log.columns:
            continue
        log[f"{name}_abs_err"] = (log[col] - log["actual_close"]).abs()
        # Directional accuracy is UNDEFINED for the naive model. Its
        # prediction IS today's close, so its implied move is always exactly
        # zero and sign(0) matches nothing -- which produced a meaningless
        # "naive directional accuracy 0.025", really just the share of days
        # with no price change. A model that declines to forecast direction
        # should be reported as having no directional score, not a terrible
        # one.
        if name == "naive":
            continue
        log[f"{name}_dir_right"] = (
            np.sign(log[col] - log["close_today"])
            == np.sign(log["actual_close"] - log["close_today"]))
    for name in ("gpr", "arima"):
        lo, hi = f"{name}_lower80", f"{name}_upper80"
        if lo in log.columns:
            log[f"{name}_in_interval"] = ((log["actual_close"] >= log[lo])
                                          & (log["actual_close"] <= log[hi]))

    log.to_parquet(LOG_PATH)
    if verbose:
        print(f"  filled {filled} outcome(s); "
              f"{int(log['actual_close'].notna().sum())} of {len(log)} rows now scored")
    return log


def backfill(tickers: list | None = None, n_days: int = 40,
             gp_train_points: int = 250, verbose: bool = True) -> pd.DataFrame:
    """Walk-forward backtest over recent history. NOT prospective.

    Rows are tagged source="backfill" so they can never be confused with the
    genuinely prospective ones. The distinction matters: a backfill was run
    after the outcomes existed, so while nothing leaks (each fit sees only
    data up to its own as-of date), the experiment was designed knowing what
    the period contained. The prospective log is the stronger evidence; this
    is here because waiting six weeks for a first number is not a plan, and
    because a proper walk-forward evaluation was missing from the project
    entirely -- the original finance result used one fixed 30-day split.

    Each step fits only on `features.iloc[:i]`, so step i knows nothing about
    day i's close, which is what it is asked to predict.
    """
    from finance.price_data import add_technical_indicators, get_price_history

    tickers = tickers or DEFAULT_TICKERS
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ticker in tickers:
        try:
            features = add_technical_indicators(get_price_history(ticker, start=HISTORY_START)).dropna()
        except Exception as e:
            if verbose:
                print(f"  {ticker}: history fetch failed ({type(e).__name__})")
            continue
        if len(features) < gp_train_points + n_days + 5:
            if verbose:
                print(f"  {ticker}: only {len(features)} rows, need "
                      f"{gp_train_points + n_days + 5}")
            continue

        done = 0
        for offset in range(n_days, 0, -1):
            hist = features.iloc[:-offset]          # strictly before the target day
            target_idx = len(features) - offset
            actual_date = features.index[target_idx]
            actual_close = float(features["Close"].iloc[target_idx])
            close_today = float(hist["Close"].iloc[-1])

            row = {"predicted_at": pd.Timestamp(hist.index[-1]), "ticker": ticker,
                   "as_of_date": pd.Timestamp(hist.index[-1]), "close_today": close_today,
                   "target_date": pd.Timestamp(actual_date), "naive_pred": close_today,
                   "actual_close": actual_close, "actual_date": pd.Timestamp(actual_date),
                   "source": "backfill"}
            try:
                gp = _gpr_forecast(hist, n_train=gp_train_points)
                row.update({"gpr_pred": gp["pred"], "gpr_lower80": gp["lower"],
                            "gpr_upper80": gp["upper"], "gpr_sigma": gp["sigma"],
                            "gpr_n_train": gp["n_train"]})
            except Exception as e:
                row["gpr_error"] = f"{type(e).__name__}: {e}"
            try:
                ar = _arima_forecast(hist["Close"])
                row.update({"arima_pred": ar["pred"], "arima_lower80": ar["lower"],
                            "arima_upper80": ar["upper"]})
            except Exception as e:
                row["arima_error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            done += 1
            if verbose and done % 10 == 0:
                print(f"  {ticker}: {done}/{n_days}", flush=True)

    if not rows:
        return load_log()

    fresh = pd.DataFrame(rows)
    existing = load_log()
    if not existing.empty and "source" not in existing.columns:
        existing["source"] = "prospective"
    combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
    combined = combined.drop_duplicates(subset=["ticker", "target_date", "source"], keep="last")
    combined.to_parquet(LOG_PATH)
    if verbose:
        print(f"\n  backfilled {len(fresh)} rows; log now {len(combined)}")
    return combined


def load_log() -> pd.DataFrame:
    return pd.read_parquet(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame()


def latest(ticker: str) -> dict | None:
    """The most recent forecast for one ticker -- what the chatbot reads.

    A lookup, not a fit. This is the point of the log: a GP costs ~44s to fit
    and cannot sit in a chat turn, but yesterday's stored forecast is free.
    """
    log = load_log()
    if log.empty:
        return None
    rows = log[log["ticker"] == ticker].sort_values("predicted_at")
    return None if rows.empty else rows.iloc[-1].to_dict()


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def evaluate(min_rows: int = 5, source: str | None = None) -> dict:
    """Score every predictor on outcomes that have actually arrived.

    Reports point accuracy AND calibration, because they are different claims
    and only the second one is winnable here. "Beat the naive baseline" is the
    accuracy test; "was the actual inside the 80% interval about 80% of the
    time" is the calibration test, and a model can fail the first while
    passing the second -- which is a genuinely useful thing to be able to say.
    """
    log = load_log()
    if log.empty:
        return {"status": "no predictions logged"}
    scored = log[log["actual_close"].notna()]
    if source and "source" in scored.columns:
        scored = scored[scored["source"] == source]
    if len(scored) < min_rows:
        return {"status": f"only {len(scored)} scored row(s); need {min_rows}",
                "n_logged": len(log), "n_pending": int(log["actual_close"].isna().sum())
                if "actual_close" in log else len(log)}

    if "source" in scored.columns:
        out_src = scored["source"].value_counts().to_dict()
    else:
        out_src = {"prospective": len(scored)}

    out = {"n_scored": len(scored), "by_source": out_src,
           "n_pending": int(log["actual_close"].isna().sum()),
           "span": f"{scored['target_date'].min().date()} -> {scored['target_date'].max().date()}",
           "tickers": sorted(scored["ticker"].unique().tolist())}

    for name in ("naive", "gpr", "arima"):
        err_col, dir_col = f"{name}_abs_err", f"{name}_dir_right"
        if err_col not in scored.columns:
            continue
        sub = scored[scored[err_col].notna()]
        if sub.empty:
            continue
        entry = {"n": len(sub),
                 "mean_abs_err": round(float(sub[err_col].mean()), 3),
                 "rmse": round(float(np.sqrt((sub[err_col] ** 2).mean())), 3)}
        if name == "naive":
            entry["directional_accuracy"] = "n/a -- predicts no change, so has no direction"
        elif dir_col in sub.columns and sub[dir_col].notna().any():
            entry["directional_accuracy"] = round(float(sub[dir_col].mean()), 4)
        int_col = f"{name}_in_interval"
        if int_col in sub.columns and sub[int_col].notna().any():
            entry["interval_coverage"] = round(float(sub[int_col].mean()), 4)
            entry["coverage_target"] = INTERVAL
        out[name] = entry

    # the comparison that was missing from every earlier finance result
    if "naive" in out and "gpr" in out:
        out["gpr_beats_naive"] = out["gpr"]["rmse"] < out["naive"]["rmse"]
        out["gpr_vs_naive_ratio"] = round(out["gpr"]["rmse"] / out["naive"]["rmse"], 3)
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Prospective price-forecast log.")
    ap.add_argument("--log", action="store_true", help="predict and append today's forecasts")
    ap.add_argument("--reconcile", action="store_true", help="fill in outcomes that have arrived")
    ap.add_argument("--evaluate", action="store_true", help="score what has been scored")
    ap.add_argument("--backfill", type=int, default=0,
                    help="walk-forward backtest over the last N trading days (NOT prospective)")
    ap.add_argument("--source", default=None, help="score only 'prospective' or 'backfill'")
    ap.add_argument("--tickers", default=None, help="comma-separated override")
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    if not (args.log or args.reconcile or args.evaluate or args.backfill):
        args.log = args.reconcile = args.evaluate = True

    if args.backfill:
        print(f"Walk-forward backfill over the last {args.backfill} trading days "
              f"(backtest, not prospective)...")
        backfill(tickers, n_days=args.backfill)

    if args.log:
        print("Logging forecasts for the next trading day...")
        log_predictions(tickers)
    if args.reconcile:
        print("\nReconciling outcomes...")
        reconcile()
    if args.evaluate:
        print("\nScores:")
        print(json.dumps(evaluate(source=args.source), indent=2, default=str))
