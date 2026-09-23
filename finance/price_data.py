"""Free historical OHLCV data via yfinance, plus technical indicators.

Adapted from the existing stock-prediction Streamlit app
(pages/utils/model_train.py), extended to a ticker basket and
richer technical features for the augmented price model.
"""

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
                    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


def get_price_history(ticker: str, start: str = "2022-01-01", end: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV history for one ticker. Free, no API key."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "Date"
    return df


def get_basket(tickers: list[str] | None = None, start: str = "2022-01-01") -> dict[str, pd.DataFrame]:
    tickers = tickers or DEFAULT_TICKERS
    out = {}
    for t in tickers:
        try:
            out[t] = get_price_history(t, start=start)
        except Exception as exc:  # noqa: BLE001 - surface but keep pipeline going for the rest of the basket
            print(f"[price_data] skipping {t}: {exc}")
    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD, moving averages, rolling volatility -- same family of indicators
    already used in pages/Stock_Analysis.py, computed here as model features."""
    out = df.copy()
    close = out["Close"]

    out["SMA_7"] = close.rolling(7).mean()
    out["SMA_21"] = close.rolling(21).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    out["Volatility_21"] = close.pct_change().rolling(21).std()
    out["Return_1d"] = close.pct_change()

    return out.dropna()


if __name__ == "__main__":
    df = get_price_history("AAPL", start="2024-01-01")
    feats = add_technical_indicators(df)
    print(feats.tail())
    print(f"\nrows: {len(feats)}, columns: {list(feats.columns)}")
