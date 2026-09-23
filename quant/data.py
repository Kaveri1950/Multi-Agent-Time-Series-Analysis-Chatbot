"""Price panel and point-in-time universe for Indian equities.

SURVIVORSHIP BIAS IS THE FIRST THING THAT GOES WRONG HERE, so it is the
first thing this module deals with.

NSE publishes only TODAY's index membership. Backtesting 2020 with today's
Nifty 500 list means testing on companies that survived to 2026 -- the
failures were removed from the list, so the strategy is handed a universe
selected on an outcome it could not have known. That inflates returns, and
it is the single most common error in student equity backtests.

Two defences, and the honest limits of each:

1. The universe is reconstructed AT EACH DATE from rolling traded value,
   not from the current index list. On any given day the universe is the
   top-N most liquid names as of that day, which is information that
   genuinely existed then. Liquidity is also the right screen on its own
   terms: Avramov et al. (2023) found ML equity strategies lose
   significance once illiquid names are excluded, because that is where
   their apparent edge lives.

2. The residual bias is stated rather than hidden: yfinance does not serve
   delisted tickers at all, so companies that failed are missing from the
   panel entirely. Starting from the Nifty 500 (large and mid caps) limits
   the damage -- outright delisting is rare there -- but it does not
   eliminate it, and any result here should be read as a mild upper bound.

Everything downstream is built from prices and volumes only. Fundamental
ratios from yfinance (P/B, dividend yield) are CURRENT snapshots with no
history, so using them in a backtest would be look-ahead bias of the most
direct kind -- scoring 2020 with a number computed in 2026.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache" / "quant"
PANEL_PATH = CACHE_DIR / "price_panel.parquet"
VOLUME_PATH = CACHE_DIR / "volume_panel.parquet"
META_PATH = CACHE_DIR / "panel_meta.json"

DEFAULT_INDEX = "nifty500"
DEFAULT_START = "2019-01-01"

# A name needs enough history for a 12-month momentum signal plus a year of
# evaluation before it is usable at all.
MIN_HISTORY_DAYS = 400


def download_panel(index: str = DEFAULT_INDEX, start: str = DEFAULT_START,
                   force: bool = False, verbose: bool = True) -> tuple:
    """Adjusted closes and volumes for the index, cached as parquet.

    auto_adjust=True matters: it back-adjusts for splits and dividends, so a
    5:1 split does not appear as an 80% crash and get picked up as a
    momentum signal.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if PANEL_PATH.exists() and VOLUME_PATH.exists() and not force:
        return pd.read_parquet(PANEL_PATH), pd.read_parquet(VOLUME_PATH)

    import yfinance as yf

    from finance.india_universe import fetch_index_constituents

    symbols = fetch_index_constituents(index)
    if verbose:
        print(f"  downloading {len(symbols)} tickers from {start}...", flush=True)

    raw = yf.download(symbols, start=start, progress=False,
                      auto_adjust=True, threads=True)
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Drop names too sparse to compute a 12-month signal on.
    keep = close.notna().sum() >= MIN_HISTORY_DAYS
    close, volume = close.loc[:, keep], volume.loc[:, keep]

    close.to_parquet(PANEL_PATH)
    volume.to_parquet(VOLUME_PATH)
    META_PATH.write_text(pd.Series({
        "index": index, "start": start,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_requested": len(symbols), "n_kept": int(keep.sum()),
        "first_date": str(close.index[0].date()), "last_date": str(close.index[-1].date()),
    }).to_json(indent=2))

    if verbose:
        print(f"  kept {int(keep.sum())} of {len(symbols)} tickers "
              f"({close.index[0].date()} to {close.index[-1].date()})")
    return close, volume


def load_panel() -> tuple:
    if not PANEL_PATH.exists():
        raise FileNotFoundError("no panel cached -- run `python -m quant.data`")
    return pd.read_parquet(PANEL_PATH), pd.read_parquet(VOLUME_PATH)


def traded_value(close: pd.DataFrame, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Rolling average daily traded value (price x volume).

    Traded value rather than share volume, because a share count is not
    comparable across stocks priced differently -- 10,000 shares of a Rs 50
    stock is not the same market as 10,000 shares of a Rs 5,000 one.
    """
    return (close * volume).rolling(window, min_periods=window // 2).mean()


def point_in_time_universe(close: pd.DataFrame, volume: pd.DataFrame,
                           top_n: int = 200, window: int = 21) -> pd.DataFrame:
    """Boolean mask: was this stock in the tradable universe on this date?

    Membership is the top `top_n` names by trailing traded value, recomputed
    every day from data available on that day. Nothing about the future
    enters -- which is the whole point, and the difference from filtering on
    today's index list.
    """
    tv = traded_value(close, volume, window=window)
    ranks = tv.rank(axis=1, ascending=False, method="first")
    mask = ranks <= top_n
    # A name with no price that day cannot be traded regardless of rank.
    return mask & close.notna()


def universe_stats(mask: pd.DataFrame) -> dict:
    """How much the universe churns -- the number that shows the
    reconstruction is doing something."""
    daily_count = mask.sum(axis=1)
    entries_exits = mask.astype(int).diff().abs().sum(axis=1)
    return {
        "n_dates": len(mask),
        "median_universe_size": int(daily_count.median()),
        "n_names_ever_included": int(mask.any().sum()),
        "mean_daily_changes": round(float(entries_exits.iloc[1:].mean()), 2),
        "total_distinct_members": int(mask.any().sum()),
    }


if __name__ == "__main__":
    import json

    print("Building the Indian equity price panel...")
    close, volume = download_panel()

    print(f"\npanel: {close.shape[0]} trading days x {close.shape[1]} tickers")
    print(f"  {close.index[0].date()} to {close.index[-1].date()}")

    mask = point_in_time_universe(close, volume, top_n=200)
    stats = universe_stats(mask)
    print("\npoint-in-time universe (top 200 by 21-day traded value):")
    print(json.dumps(stats, indent=2))
    print(f"\n  {stats['n_names_ever_included']} distinct names appear at some point, "
          f"but only ~{stats['median_universe_size']} on any given day -- the gap is "
          f"membership churn, which a fixed 'today's index' list would have hidden.")
