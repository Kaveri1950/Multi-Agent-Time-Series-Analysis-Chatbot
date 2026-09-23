"""Multi-type anomaly detection for daily price/volume/sentiment series.

Mirrors astronomy/anomaly_detector.py's shape (per-row feature extraction ->
scored output -> top-N ranking) but tailored to what actually differs about
financial data: it's daily-frequency (not a fine-grained continuous flux
signal), it regime-shifts (a "normal" volatility level today can be
completely different from six months ago), and it comes enriched with the
technical indicators finance/price_data.py already computes.

Each detector below is a rolling z-score against the series' OWN recent
history rather than one fixed global threshold, so what counts as
"anomalous" adapts as the regime changes -- the closest finance analogue to
how astronomy's IsolationForest fits per-target instead of using one global
model.

Five independent detectors are fused into one severity-scored, TYPED
anomaly table -- not a single binary flag -- so the chatbot can explain
*why* a day was flagged instead of just that it was:
  1. return_outlier        -- a day's return is a statistical outlier
  2. volatility_spike      -- realized volatility jumps unusually high
  3. volume_spike          -- traded volume far above its trailing average
  4. gap_event             -- overnight open-vs-prior-close jump
  5. sentiment_divergence  -- sentiment strongly one direction while price
                               is flat or moves the opposite way (only
                               possible because this project already has a
                               parallel FinBERT sentiment pipeline -- no
                               astronomy-side analogue)
"""

import numpy as np
import pandas as pd

RETURN_WINDOW = 60
RETURN_Z_THRESHOLD = 2.5
VOLATILITY_WINDOW = 90
VOLATILITY_Z_THRESHOLD = 2.0
VOLUME_WINDOW = 20
VOLUME_RATIO_THRESHOLD = 2.5
GAP_THRESHOLD_PCT = 0.03
SENTIMENT_THRESHOLD = 0.3
SENTIMENT_FLAT_RETURN_THRESHOLD = 0.005


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    roll_std = series.rolling(window, min_periods=max(5, window // 3)).std()
    return (series - roll_mean) / roll_std.replace(0, np.nan)


def detect_return_outliers(df: pd.DataFrame, window: int = RETURN_WINDOW,
                            z_threshold: float = RETURN_Z_THRESHOLD) -> pd.DataFrame:
    z = _rolling_zscore(df["Return_1d"], window)
    return pd.DataFrame({"return_zscore": z, "return_outlier": z.abs() > z_threshold}, index=df.index)


def detect_volatility_spikes(df: pd.DataFrame, window: int = VOLATILITY_WINDOW,
                              z_threshold: float = VOLATILITY_Z_THRESHOLD) -> pd.DataFrame:
    # One-directional: only unusually HIGH volatility is a "spike" worth
    # flagging, unlike returns which can be outliers in either direction.
    z = _rolling_zscore(df["Volatility_21"], window)
    return pd.DataFrame({"volatility_zscore": z, "volatility_spike": z > z_threshold}, index=df.index)


def detect_volume_spikes(df: pd.DataFrame, window: int = VOLUME_WINDOW,
                          ratio_threshold: float = VOLUME_RATIO_THRESHOLD) -> pd.DataFrame:
    roll_mean = df["Volume"].rolling(window, min_periods=max(5, window // 3)).mean()
    ratio = df["Volume"] / roll_mean.replace(0, np.nan)
    return pd.DataFrame({"volume_ratio": ratio, "volume_spike": ratio > ratio_threshold}, index=df.index)


def detect_gap_events(df: pd.DataFrame, gap_threshold_pct: float = GAP_THRESHOLD_PCT) -> pd.DataFrame:
    prev_close = df["Close"].shift(1)
    gap_pct = (df["Open"] - prev_close) / prev_close
    return pd.DataFrame({"gap_pct": gap_pct, "gap_event": gap_pct.abs() > gap_threshold_pct}, index=df.index)


def detect_sentiment_divergence(df: pd.DataFrame, sentiment_series: pd.Series,
                                 sentiment_threshold: float = SENTIMENT_THRESHOLD,
                                 flat_return_threshold: float = SENTIMENT_FLAT_RETURN_THRESHOLD) -> pd.DataFrame:
    """Sentiment strongly one direction while the price return is flat or
    moves the opposite way. A cross-signal anomaly with no single-series
    equivalent -- it only exists because this project already runs both a
    price pipeline and a FinBERT news-sentiment pipeline on the same
    ticker/date axis."""
    sent = sentiment_series.reindex(df.index).ffill().fillna(0.0)
    ret = df["Return_1d"]
    opposite_or_flat = (ret.abs() < flat_return_threshold) | (np.sign(sent) != np.sign(ret))
    divergence = (sent.abs() > sentiment_threshold) & opposite_or_flat
    return pd.DataFrame({"sentiment_score": sent, "sentiment_divergence": divergence}, index=df.index)


_FLAG_COLUMNS = ["return_outlier", "volatility_spike", "volume_spike", "gap_event", "sentiment_divergence"]


def _severity_label(n_flagged: int) -> str:
    if n_flagged >= 3:
        return "HIGH"
    if n_flagged == 2:
        return "MEDIUM"
    if n_flagged == 1:
        return "LOW"
    return "NONE"


def _reasons(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("return_outlier"):
        reasons.append(f"return z-score {row['return_zscore']:.2f}")
    if row.get("volatility_spike"):
        reasons.append(f"volatility z-score {row['volatility_zscore']:.2f}")
    if row.get("volume_spike"):
        reasons.append(f"volume {row['volume_ratio']:.1f}x trailing average")
    if row.get("gap_event"):
        reasons.append(f"overnight gap {row['gap_pct'] * 100:.1f}%")
    if row.get("sentiment_divergence"):
        reasons.append(f"sentiment {row['sentiment_score']:+.2f} diverging from price move")
    return reasons


def detect_all(df: pd.DataFrame, sentiment_series: pd.Series | None = None) -> pd.DataFrame:
    """Runs every detector and fuses the results into one severity-scored,
    typed-anomaly table indexed the same as df (output of
    finance.price_data.add_technical_indicators -- needs Close, Open,
    Volume, Return_1d, Volatility_21)."""
    out = pd.concat([
        detect_return_outliers(df),
        detect_volatility_spikes(df),
        detect_volume_spikes(df),
        detect_gap_events(df),
    ], axis=1)

    if sentiment_series is not None and not sentiment_series.empty:
        out = pd.concat([out, detect_sentiment_divergence(df, sentiment_series)], axis=1)
    else:
        out["sentiment_score"] = np.nan
        out["sentiment_divergence"] = False

    out["n_types_flagged"] = out[_FLAG_COLUMNS].fillna(False).sum(axis=1).astype(int)
    out["is_anomaly"] = out["n_types_flagged"] > 0

    # Severity: sum of magnitudes actually contributing, so one extreme
    # signal can outrank two mild ones firing together, not just a count.
    severity = out[["return_zscore", "volatility_zscore"]].abs().sum(axis=1, skipna=True).fillna(0)
    severity += (out["volume_ratio"].fillna(1) - 1).clip(lower=0)
    severity += out["gap_pct"].abs().fillna(0) * 10
    out["severity_score"] = severity.where(out["is_anomaly"], 0.0)
    out["severity_label"] = out["n_types_flagged"].apply(_severity_label)
    out["reasons"] = out.apply(_reasons, axis=1)
    return out


def top_anomalies(scored_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return scored_df[scored_df["is_anomaly"]].sort_values("severity_score", ascending=False).head(n)


if __name__ == "__main__":
    from price_data import add_technical_indicators, get_price_history

    ticker = "RELIANCE.NS"
    raw = get_price_history(ticker, start="2023-01-01")
    feats = add_technical_indicators(raw)
    scored = detect_all(feats)

    print(f"{ticker}: {len(scored)} days analyzed, {int(scored['is_anomaly'].sum())} flagged anomalous")
    print("\nTop 5 most severe anomalies:")
    top = top_anomalies(scored, n=5)
    for idx, row in top.iterrows():
        print(f"  {idx.date()}  [{row['severity_label']}]  severity={row['severity_score']:.2f}  {row['reasons']}")
