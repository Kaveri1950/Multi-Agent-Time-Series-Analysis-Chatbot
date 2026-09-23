"""Daily price forecasts in MongoDB: one row per ticker per trading day.

THE SHAPE OF A ROW. Each row is one day's prediction for one ticker, made
using ONLY data available before that day, plus the actual close once it
exists. So a row for 21 Aug was produced from everything up to 20 Aug, and
its `actual_close` is filled in after 21 Aug closes. Nothing in a row can
have been influenced by the outcome it is scored against.

TWO WAYS ROWS ARRIVE, and they are tagged differently because they are
different strengths of evidence:

  source="backfill" -- walk-forward over history. Each fit sees only prior
      data, so nothing leaks, but the experiment was designed knowing what
      the period contained. Useful for getting a number today instead of in
      six weeks.
  source="daily"    -- written before the outcome exists. Genuinely
      prospective, and the only kind that cannot be tuned against.

WHY MONGODB rather than the parquet log this replaces: rows are updated in
place when the actual arrives, which parquet does badly (rewrite the whole
file) and a document store does natively. The chat history already lives in
Mongo, so there is one database to run rather than two storage systems.

DEGRADES TO NOTHING. Every function returns an empty result rather than
raising if Mongo is unreachable, matching dashboard/chat_store.py -- a dead
database should cost the forecast history, not the chatbot.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGODB_DB", "btp_chat")
COLLECTION = "price_forecasts"

_CONNECT_TIMEOUT_MS = 2500

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS"]
HISTORY_START = "2023-01-01"
GP_TRAIN_POINTS = 250          # the GP fit is O(n^3); 250 keeps a day's run sane

_client = None
_unavailable = None


def _collection():
    global _client, _unavailable
    if _client is not None:
        return _client[MONGO_DB][COLLECTION]
    if _unavailable is not None:
        return None
    try:
        from pymongo import ASCENDING, MongoClient

        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=_CONNECT_TIMEOUT_MS)
        client.server_info()
        col = client[MONGO_DB][COLLECTION]
        # One row per (ticker, target_date, source). The unique index is what
        # makes a re-run idempotent: a cron that fires twice, or a backfill
        # over an already-covered range, updates rather than duplicates.
        col.create_index([("ticker", ASCENDING), ("target_date", ASCENDING),
                          ("source", ASCENDING)], unique=True)
        col.create_index([("target_date", ASCENDING)])
        _client = client
        return col
    except Exception as e:
        _unavailable = f"{type(e).__name__}: {e}"
        return None


def is_available() -> tuple[bool, str]:
    return (True, "connected") if _collection() is not None else (False, _unavailable or "unavailable")


# --------------------------------------------------------------------------
# predicting one day
# --------------------------------------------------------------------------

def _predict_for(features: pd.DataFrame, target_date, ticker: str, source: str) -> dict:
    """One row: predict `target_date`'s close from data strictly before it.

    `features` must already be truncated to the day before the target. The
    truncation is the caller's job precisely so it is visible at the call
    site rather than buried here.
    """
    from finance.forecast_log import _gpr_forecast

    close_before = float(features["Close"].iloc[-1])
    row = {
        "ticker": ticker,
        "target_date": pd.Timestamp(target_date).normalize().to_pydatetime(),
        "as_of_date": pd.Timestamp(features.index[-1]).normalize().to_pydatetime(),
        "close_before": close_before,
        # The null model, stored on every row. Finance results in this
        # project were once reported without one, and a naive "tomorrow
        # equals today" rule turned out to beat both models on all 9 tickers
        # tested. Storing it here makes that omission impossible to repeat.
        "naive_pred": close_before,
        "source": source,
        "predicted_at": datetime.now(timezone.utc),
        "actual_close": None,
    }
    try:
        gp = _gpr_forecast(features, n_train=GP_TRAIN_POINTS)
        row.update({"gpr_pred": gp["pred"], "gpr_lower80": gp["lower"],
                    "gpr_upper80": gp["upper"], "gpr_sigma": gp["sigma"],
                    "gpr_n_train": gp["n_train"]})
    except Exception as e:
        row["gpr_error"] = f"{type(e).__name__}: {e}"
    return row


def _upsert(rows: list) -> int:
    col = _collection()
    if col is None or not rows:
        return 0
    from pymongo import UpdateOne

    ops = [UpdateOne({"ticker": r["ticker"], "target_date": r["target_date"],
                      "source": r["source"]},
                     {"$set": r}, upsert=True) for r in rows]
    result = col.bulk_write(ops, ordered=False)
    return (result.upserted_count or 0) + (result.modified_count or 0)


# --------------------------------------------------------------------------
# backfill and the daily job
# --------------------------------------------------------------------------

def backfill(tickers: list | None = None, n_days: int = 40, verbose: bool = True) -> int:
    """Walk-forward over the last `n_days` trading days.

    For each day D, fit on everything strictly before D and predict D's
    close, then store D's real close alongside. Tagged source="backfill".
    """
    from finance.price_data import add_technical_indicators, get_price_history

    tickers = tickers or DEFAULT_TICKERS
    if _collection() is None:
        if verbose:
            print(f"  MongoDB unavailable ({_unavailable})")
        return 0

    written = 0
    for ticker in tickers:
        try:
            feats = add_technical_indicators(get_price_history(ticker, start=HISTORY_START)).dropna()
        except Exception as e:
            if verbose:
                print(f"  {ticker}: history fetch failed ({type(e).__name__})")
            continue
        if len(feats) < GP_TRAIN_POINTS + n_days + 5:
            if verbose:
                print(f"  {ticker}: only {len(feats)} rows, need "
                      f"{GP_TRAIN_POINTS + n_days + 5}")
            continue

        rows = []
        for offset in range(n_days, 0, -1):
            history = feats.iloc[:-offset]                 # strictly before the target
            target_idx = len(feats) - offset
            row = _predict_for(history, feats.index[target_idx], ticker, "backfill")
            row["actual_close"] = float(feats["Close"].iloc[target_idx])
            rows.append(_score(row))
        written += _upsert(rows)
        if verbose:
            print(f"  {ticker}: {len(rows)} day(s) backfilled", flush=True)

    if verbose:
        print(f"\n  {written} row(s) written; collection now {count()} rows")
    return written


def log_today(tickers: list | None = None, verbose: bool = True) -> int:
    """Predict the NEXT trading day and store it with no outcome.

    This is the prospective half. The row is written before the close it
    predicts exists, so when reconcile() fills the actual in tomorrow, the
    comparison is genuinely out of sample.
    """
    from finance.price_data import add_technical_indicators, get_price_history

    tickers = tickers or DEFAULT_TICKERS
    if _collection() is None:
        if verbose:
            print(f"  MongoDB unavailable ({_unavailable})")
        return 0

    rows = []
    for ticker in tickers:
        try:
            feats = add_technical_indicators(get_price_history(ticker, start=HISTORY_START)).dropna()
            target = pd.Timestamp(feats.index[-1]) + pd.offsets.BDay(1)
            rows.append(_predict_for(feats, target, ticker, "daily"))
            if verbose:
                r = rows[-1]
                got = f"{r.get('gpr_pred', float('nan')):.2f}" if "gpr_pred" in r else "FAILED"
                print(f"  {ticker:<14} close {r['close_before']:>9.2f} -> predicts {got} "
                      f"for {pd.Timestamp(target).date()}")
        except Exception as e:
            if verbose:
                print(f"  {ticker:<14} FAILED {type(e).__name__}: {e}")
    return _upsert(rows)


def reconcile(verbose: bool = True) -> int:
    """Fill in actual closes for rows whose day has now happened.

    Matches the first real close on or after the target date rather than
    demanding an exact hit -- market holidays shift the session, and an
    exact-match rule would strand those rows permanently.
    """
    from finance.price_data import get_price_history

    col = _collection()
    if col is None:
        return 0

    pending = list(col.find({"actual_close": None}))
    if not pending:
        if verbose:
            print("  nothing awaiting an outcome")
        return 0

    filled = 0
    for ticker in sorted({r["ticker"] for r in pending}):
        try:
            closes = get_price_history(ticker, start=HISTORY_START)["Close"]
        except Exception as e:
            if verbose:
                print(f"  {ticker}: price fetch failed ({type(e).__name__})")
            continue
        for row in [r for r in pending if r["ticker"] == ticker]:
            later = closes[closes.index >= pd.Timestamp(row["target_date"])]
            if later.empty:
                continue                                   # not yet traded
            row["actual_close"] = float(later.iloc[0])
            row["actual_date"] = pd.Timestamp(later.index[0]).normalize().to_pydatetime()
            col.update_one({"_id": row["_id"]}, {"$set": _score(row)})
            filled += 1

    if verbose:
        print(f"  filled {filled} outcome(s)")
    return filled


def _score(row: dict) -> dict:
    """Derived columns, recomputed from scratch so they always match the data."""
    actual = row.get("actual_close")
    if actual is None:
        return row
    for name in ("gpr", "naive"):
        pred = row.get(f"{name}_pred")
        if pred is None:
            continue
        row[f"{name}_abs_err"] = abs(pred - actual)
        # Naive predicts no change, so its implied direction is always
        # exactly zero and matches nothing. Reporting a directional score
        # for it would be meaningless, so it does not get one.
        if name != "naive":
            row[f"{name}_dir_right"] = bool(
                np.sign(pred - row["close_before"]) == np.sign(actual - row["close_before"]))
    lo, hi = row.get("gpr_lower80"), row.get("gpr_upper80")
    if lo is not None and hi is not None:
        row["gpr_in_interval"] = bool(lo <= actual <= hi)
    return row


# --------------------------------------------------------------------------
# reading back
# --------------------------------------------------------------------------

def count() -> int:
    col = _collection()
    return col.count_documents({}) if col is not None else 0


def history(ticker: str | None = None, source: str | None = None,
            limit: int = 500) -> pd.DataFrame:
    """Rows as a DataFrame, oldest first -- what a chart reads."""
    col = _collection()
    if col is None:
        return pd.DataFrame()
    query = {}
    if ticker:
        query["ticker"] = ticker
    if source:
        query["source"] = source
    rows = list(col.find(query, {"_id": 0}).sort("target_date", 1).limit(limit))
    return pd.DataFrame(rows)


def latest(ticker: str) -> dict | None:
    """The most recent stored forecast for one ticker -- a lookup, not a fit."""
    col = _collection()
    if col is None:
        return None
    return col.find_one({"ticker": ticker}, {"_id": 0}, sort=[("target_date", -1)])


def evaluate(ticker: str | None = None, source: str | None = None) -> dict:
    """Score everything that has an outcome.

    Reports point accuracy AND interval coverage, because only the second is
    winnable: measured over 66 month-forecasts the GP's band held 75.6% of
    the time against an 80% target, while its central estimate beat the naive
    baseline on exactly half.
    """
    df = history(ticker=ticker, source=source, limit=100000)
    if df.empty or "actual_close" not in df.columns:
        return {"status": "nothing scored yet", "n_rows": len(df)}
    scored = df[df["actual_close"].notna()]
    if scored.empty:
        return {"status": "no outcomes yet", "n_pending": len(df)}

    out = {"n_scored": len(scored), "n_pending": int(df["actual_close"].isna().sum()),
           "span": f"{scored['target_date'].min().date()} -> {scored['target_date'].max().date()}",
           "tickers": sorted(scored["ticker"].unique().tolist()),
           "by_source": scored["source"].value_counts().to_dict()}

    for name in ("naive", "gpr"):
        col_err = f"{name}_abs_err"
        if col_err not in scored.columns:
            continue
        sub = scored[scored[col_err].notna()]
        if sub.empty:
            continue
        entry = {"n": len(sub),
                 "mean_abs_err": round(float(sub[col_err].mean()), 3),
                 "rmse": round(float(np.sqrt((sub[col_err] ** 2).mean())), 3)}
        if name == "naive":
            entry["directional_accuracy"] = "n/a -- predicts no change"
        elif f"{name}_dir_right" in sub.columns:
            entry["directional_accuracy"] = round(float(sub[f"{name}_dir_right"].mean()), 4)
        if name == "gpr" and "gpr_in_interval" in sub.columns:
            entry["interval_coverage"] = round(float(sub["gpr_in_interval"].mean()), 4)
            entry["coverage_target"] = 0.80
        out[name] = entry

    if "naive" in out and "gpr" in out:
        out["gpr_vs_naive_rmse_ratio"] = round(out["gpr"]["rmse"] / out["naive"]["rmse"], 3)
        out["gpr_beats_naive"] = out["gpr"]["rmse"] < out["naive"]["rmse"]
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Daily price forecasts in MongoDB.")
    ap.add_argument("--backfill", type=int, default=0, help="walk-forward over the last N days")
    ap.add_argument("--log", action="store_true", help="predict the next trading day")
    ap.add_argument("--reconcile", action="store_true", help="fill in outcomes that arrived")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--tickers", default=None)
    args = ap.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None
    ok, reason = is_available()
    print(f"MongoDB: {reason}  |  collection holds {count()} row(s)\n")
    if not ok:
        raise SystemExit(1)

    if args.backfill:
        print(f"Walk-forward backfill, last {args.backfill} trading days...")
        backfill(tickers, n_days=args.backfill)
    if args.log:
        print("\nPredicting the next trading day...")
        log_today(tickers)
    if args.reconcile:
        print("\nReconciling...")
        reconcile()
    if args.evaluate or not (args.backfill or args.log or args.reconcile):
        print("\nScores:")
        print(json.dumps(evaluate(), indent=2, default=str))
