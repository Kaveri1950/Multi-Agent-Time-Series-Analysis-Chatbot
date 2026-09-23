"""Two price-forecasting models, for a direct baseline-vs-augmented comparison:

1. ARIMA baseline -- same approach as the existing stock-prediction app
   (pages/utils/model_train.py), no sentiment, close price only.
2. XGBoost augmented model -- technical indicators + sentiment score as
   features, predicting next-day return. CPU-friendly, trains in seconds.

This comparison (RMSE with vs without sentiment) is the key financial-track
result for the BTP report (TECHNICAL_PLAN.md, Month 1 Week 4).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


class InsufficientDataError(ValueError):
    """A price series doesn't have enough history for a meaningful ARIMA
    fit -- e.g. a thinly-traded/illiquid ticker where Yahoo Finance only
    has a handful of rows. Found in testing: a BSE-listed ticker
    (RELINFRA.BO) returned exactly 1 row regardless of how far back the
    start date went, and statsmodels' adfuller() failed deep inside with a
    cryptic "Invalid input, x is constant" rather than a clear message
    about the real problem (not enough data)."""


def _require_min_history(close: pd.Series, min_points: int) -> None:
    if len(close) < min_points:
        raise InsufficientDataError(
            f"Only {len(close)} historical data point(s) available -- need at least {min_points} for a "
            "meaningful ARIMA fit. This ticker may be too thinly traded/illiquid for Yahoo Finance to have "
            "adequate history, or the requested date range is too short."
        )


# ---------------------------------------------------------------------------
# 1. ARIMA baseline (close price only)
# ---------------------------------------------------------------------------

def _differencing_order(close: pd.Series, max_d: int = 2) -> int:
    d = 0
    series = close.copy()
    while d < max_d:
        p_value = adfuller(series.dropna())[1]
        if p_value <= 0.05:
            break
        series = series.diff()
        d += 1
    return d


def arima_baseline(close: pd.Series, test_size: int = 30) -> dict:
    _require_min_history(close, min_points=test_size + 40)
    train, test = close[:-test_size], close[-test_size:]
    d = _differencing_order(train)

    model = ARIMA(train, order=(5, d, 5))
    fit = model.fit()
    forecast = fit.get_forecast(steps=test_size).predicted_mean
    forecast.index = test.index

    rmse = float(np.sqrt(mean_squared_error(test, forecast)))
    return {"rmse": round(rmse, 4), "forecast": forecast, "test": test}


def forecast_future(close: pd.Series, days: int = 30) -> dict:
    """Genuine forward-looking forecast beyond the last available trading
    day -- NOT the same thing as arima_baseline(), which backtests against
    the last `test_size` *already-known* days to compute an RMSE. This
    fits on the FULL history and projects `days` trading days into the
    future, which is what "what will the price be in 30 days" actually
    needs. Includes an 80% confidence interval since a single-point future
    price estimate this far out is inherently uncertain -- reporting only
    the point forecast would overstate precision."""
    _require_min_history(close, min_points=40)
    d = _differencing_order(close)
    model = ARIMA(close, order=(5, d, 5))
    fit = model.fit()

    forecast_result = fit.get_forecast(steps=days)
    predicted = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.20)  # 80% CI

    future_dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=days)
    predicted_df = pd.DataFrame({
        "predicted_close": predicted.values,
        "lower_80": conf_int.iloc[:, 0].values,
        "upper_80": conf_int.iloc[:, 1].values,
    }, index=future_dates)

    return {
        "forecast": predicted_df,
        "last_known_date": close.index[-1],
        "last_known_close": float(close.iloc[-1]),
        "final_predicted_close": float(predicted_df["predicted_close"].iloc[-1]),
        "final_predicted_date": future_dates[-1],
    }


# ---------------------------------------------------------------------------
# 2. XGBoost augmented model (technical indicators + sentiment)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "SMA_7", "SMA_21", "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "RSI_14", "Volatility_21", "Return_1d",
]


def build_training_frame(features_df: pd.DataFrame, sentiment_series: pd.Series | None = None) -> pd.DataFrame:
    """features_df: output of finance.price_data.add_technical_indicators.
    sentiment_series: daily sentiment score indexed by date (optional -- if
    omitted, the model degrades gracefully to technical-indicators-only,
    which is itself a useful ablation)."""
    df = features_df.copy()
    if sentiment_series is not None:
        df = df.join(sentiment_series.rename("sentiment"), how="left")
        df["sentiment"] = df["sentiment"].fillna(0.0)
    else:
        df["sentiment"] = 0.0

    df["target_next_close"] = df["Close"].shift(-1)
    return df.dropna()


def train_xgboost_model(training_frame: pd.DataFrame, test_size: int = 30):
    import xgboost as xgb

    cols = FEATURE_COLUMNS + ["sentiment"]
    train_df, test_df = training_frame[:-test_size], training_frame[-test_size:]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(train_df[cols], train_df["target_next_close"])

    preds = model.predict(test_df[cols])
    rmse = float(np.sqrt(mean_squared_error(test_df["target_next_close"], preds)))
    return {
        "rmse": round(rmse, 4),
        "model": model,
        "predictions": pd.Series(preds, index=test_df.index),
        "actual": test_df["target_next_close"],
        "feature_importance": dict(zip(cols, model.feature_importances_.round(4).tolist())),
    }


# ---------------------------------------------------------------------------
# 3. Gaussian Process over TIME -- multi-step forecasting with an interval
# ---------------------------------------------------------------------------

GP_HISTORY_DAYS = 250        # ~1 trading year; the GP fit is O(n^3)


def gpr_forecast_path(close: pd.Series, days: int = 30,
                      history_days: int = GP_HISTORY_DAYS) -> dict:
    """Forecast a price path `days` ahead with an 80% band, using a GP on time.

    WHY A GP ON TIME, and not the feature-based GP used for the one-day
    forecast. That model maps today's technical indicators to tomorrow's
    close, so stepping 30 days ahead would mean feeding its own predictions
    back in as inputs and recomputing indicators from predicted prices --
    error compounds, and by day 30 the inputs are largely fiction. A GP
    indexed by time extrapolates natively and, more usefully, its posterior
    variance grows with distance from the data, which is exactly the shape
    uncertainty really has.

    THE GP IS FITTED TO RETURNS, NOT PRICES, and the price path is
    integrated from them. Fitting the level directly was the obvious first
    attempt and it failed in a specific, instructive way: a stationary kernel
    (RBF, Matern) extrapolates the recent trend, so on a month held out from
    TCS, RELIANCE and INFY it forecast prices RISING while all three fell,
    and only 30-67% of actual days landed inside a band that promised 80%.

    Modelling returns fixes it because it encodes the right structure. Daily
    returns are close to zero-mean noise, so the GP's posterior mean over
    future returns is near zero and the integrated price path stays flat at
    the last close -- which is the correct forecast for a random walk. The
    variance accumulates across the horizon, so the band widens as sqrt(t),
    which is how uncertainty genuinely grows.

    EXPECT A NEARLY FLAT PATH. That is the answer, not a failure: if daily
    direction were predictable the prediction would be traded away. What the
    model contributes is the WIDTH of the band, which measured well elsewhere
    in this project (82.5% coverage against an 80% target on 120 walk-forward
    days).
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    series = close.dropna().tail(history_days)
    if len(series) < 60:
        raise InsufficientDataError(
            f"only {len(series)} closes available; a GP path needs at least 60.")

    returns = series.pct_change().dropna()
    t = np.arange(len(returns), dtype=float).reshape(-1, 1)
    r = returns.values.astype(float)
    r_mean, r_std = float(r.mean()), float(r.std() or 1e-6)
    r_scaled = (r - r_mean) / r_std

    # Short lengthscale only. Any long-range component would let the GP read
    # a trend into the returns and reintroduce the extrapolation this design
    # exists to avoid. The white-noise term is expected to dominate -- that
    # is the model correctly concluding returns are mostly noise.
    kernel = (ConstantKernel(0.1) * Matern(length_scale=5.0, nu=0.5)
              + WhiteKernel(noise_level=1.0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                  alpha=1e-6, n_restarts_optimizer=0).fit(t, r_scaled)

    future_t = np.arange(len(returns), len(returns) + days, dtype=float).reshape(-1, 1)
    mu, sd = gp.predict(future_t, return_std=True)
    # No drift term. Adding r_mean back would make the forecast converge to
    # the SAMPLE MEAN daily return far from the data, and that sample mean is
    # noise: over 250 days of TCS it was -0.113%/day with a standard error of
    # 0.114%/day, a t-statistic of -0.99. Compounded it produced a confident
    # -3.3% slope whose own error bar was +/-3.5%.
    #
    # Removing it is also better empirically, not just cleaner: across 66
    # month-forecasts on 6 tickers, RMSE improved from 68.58 to 66.31 and
    # coverage from 75.6% to 76.2%. A random walk has no drift to estimate,
    # and estimating one from a year of noisy returns adds error.
    mean_returns = mu * r_std
    # Per-step predictive sd, straight from the GP.
    #
    # Do NOT add the WhiteKernel's noise on top: sklearn already includes it
    # in return_std, because the noise term is part of the kernel. Adding it
    # again inflated every band by exactly sqrt(2), which showed up as 100%
    # coverage against an 80% target -- bands so wide that nothing could
    # fall outside them. Over-covering is a calibration failure too; it just
    # looks safe instead of wrong.
    step_sd = sd * r_std

    last = float(series.iloc[-1])
    centre = last * np.cumprod(1.0 + mean_returns)
    # Variance of a sum of returns accumulates, so the band on the LEVEL
    # grows as sqrt(horizon) -- the random-walk result.
    cum_sd = np.sqrt(np.cumsum(step_sd ** 2))
    z = float(norm.ppf(0.90))                     # 80% two-sided

    future_dates = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({"predicted_close": centre,
                             "lower_80": centre * (1.0 - z * cum_sd),
                             "upper_80": centre * (1.0 + z * cum_sd)}, index=future_dates)
    return {
        "forecast": forecast,
        "last_known_date": series.index[-1],
        "last_known_close": last,
        "final_predicted_close": float(forecast["predicted_close"].iloc[-1]),
        # Same key ARIMA's forecast_future() returns, so callers can swap
        # between the two without special-casing.
        "final_predicted_date": forecast.index[-1],
        "kernel": str(gp.kernel_),
        "n_train": len(series),
        "daily_vol": round(float(r_std), 5),
    }


def backtest_forecast_month(close: pd.Series, days: int = 30,
                            history_days: int = GP_HISTORY_DAYS) -> dict:
    """Hide the last `days` of data, forecast them, and return both.

    This is the honest version of "does the model work": the GP is fitted
    only on closes strictly BEFORE the withheld month, so the comparison is
    genuinely out of sample. The alternative -- fitting on everything and
    plotting the fit over the same period -- looks far more impressive and
    means nothing.

    Returns the actual closes, the forecast path, its band, and whether each
    actual day landed inside that band, so a chart can show all three and a
    caption can state coverage rather than implying it.
    """
    series = close.dropna()
    if len(series) < days + 80:
        raise InsufficientDataError(
            f"need at least {days + 80} closes to hold out {days}; have {len(series)}.")

    train, actual = series.iloc[:-days], series.iloc[-days:]
    result = gpr_forecast_path(train, days=days, history_days=history_days)

    # The forecast is indexed by business days, which can drift from the real
    # trading calendar over a month (holidays). Align positionally instead --
    # both are "the next N trading days", in order.
    forecast = result["forecast"].copy()
    n = min(len(forecast), len(actual))
    forecast = forecast.iloc[:n]
    forecast.index = actual.index[:n]
    actual = actual.iloc[:n]

    inside = ((actual.values >= forecast["lower_80"].values)
              & (actual.values <= forecast["upper_80"].values))
    err = np.abs(forecast["predicted_close"].values - actual.values)
    naive_err = np.abs(float(train.iloc[-1]) - actual.values)

    return {
        "actual": actual,
        "forecast": forecast,
        "train_end": train.index[-1],
        "last_train_close": float(train.iloc[-1]),
        "coverage": round(float(inside.mean()), 4),
        "inside": inside,
        "mae": round(float(err.mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "naive_mae": round(float(naive_err.mean()), 3),
        "naive_rmse": round(float(np.sqrt((naive_err ** 2).mean())), 3),
        "beats_naive": bool(np.sqrt((err ** 2).mean()) < np.sqrt((naive_err ** 2).mean())),
        "kernel": result["kernel"],
        "n_train": result["n_train"],
    }


def compare_all_models(features_df: pd.DataFrame, sentiment_series: pd.Series | None = None,
                       test_size: int = 30) -> dict:
    """Every price model on ONE held-out window, so they can be plotted together.

    All four predict the same days from the same training data, which is the
    only way a comparison means anything -- the project's earlier numbers
    came from separate calls with different splits.

    THE NAIVE BASELINE IS INCLUDED AND IS NOT DECORATION. Measured across 9
    tickers, "tomorrow equals today" beat both ARIMA and XGBoost on every
    one, and directional accuracy across all models sat at 49.6%. A chart
    showing three models converging near the actual price looks impressive
    until the flat line of "assume no change" is drawn on it too, at which
    point the reader can see what the models actually added. Which is why it
    is drawn.

    "FinBERT" is a sentiment scorer, not a price model. What is compared
    here is XGBoost WITH FinBERT sentiment as a feature against XGBoost
    without it -- the ablation, labelled accurately.
    """
    out = {"test_size": test_size, "models": {}}

    frame = build_training_frame(features_df, sentiment_series)
    if len(frame) < test_size + 60:
        raise InsufficientDataError(
            f"only {len(frame)} usable rows; need {test_size + 60} to hold out {test_size}.")
    test = frame.iloc[-test_size:]
    out["actual"] = test["target_next_close"]
    out["dates"] = test.index

    # The null model: tomorrow = today.
    out["models"]["Naive (no change)"] = pd.Series(test["Close"].values, index=test.index)

    try:
        arima = arima_baseline(features_df["Close"], test_size=test_size)
        pred = arima["forecast"]
        pred.index = test.index[:len(pred)]
        out["models"]["ARIMA"] = pred
    except Exception as e:
        out.setdefault("errors", {})["ARIMA"] = f"{type(e).__name__}: {e}"

    try:
        no_sent = build_training_frame(features_df, sentiment_series=None)
        res = train_xgboost_model(no_sent, test_size=test_size)
        out["models"]["XGBoost"] = pd.Series(res["predictions"].values, index=test.index)
    except Exception as e:
        out.setdefault("errors", {})["XGBoost"] = f"{type(e).__name__}: {e}"

    if sentiment_series is not None and not sentiment_series.empty:
        try:
            res = train_xgboost_model(frame, test_size=test_size)
            out["models"]["XGBoost + FinBERT sentiment"] = pd.Series(res["predictions"].values,
                                                                    index=test.index)
        except Exception as e:
            out.setdefault("errors", {})["XGBoost + FinBERT sentiment"] = f"{type(e).__name__}: {e}"

    # GPR, walked forward one day at a time: each prediction uses only data
    # before the day it predicts. The other models here are fitted once on
    # the training window, which is the weaker protocol -- noted rather than
    # hidden, because it slightly favours GPR on recency and slightly
    # disfavours it on fitting cost.
    try:
        gp_preds = []
        for i in range(test_size):
            history = frame.iloc[:len(frame) - test_size + i]
            gp_preds.append(_gpr_forecast(history, n_train=250)["pred"])
        out["models"]["GPR"] = pd.Series(gp_preds, index=test.index)
    except Exception as e:
        out.setdefault("errors", {})["GPR"] = f"{type(e).__name__}: {e}"

    # GPR with FinBERT sentiment as an extra feature. Included so the
    # comparison is complete, but read the coverage warning below before
    # drawing any conclusion from it: free RSS reaches back only weeks, so
    # sentiment is non-zero on roughly 1.3-1.9% of training rows and is
    # exactly 0.0 on the rest. Measured on TCS the difference was -0.08 RMSE,
    # about 0.2% -- noise. That is a statement about the DATA, not about
    # whether sentiment matters: a feature that is constant-zero on 98% of
    # rows cannot inform a kernel lengthscale.
    if sentiment_series is not None and not sentiment_series.empty:
        try:
            gp_preds = []
            for i in range(test_size):
                history = frame.iloc[:len(frame) - test_size + i]
                gp_preds.append(_gpr_forecast(history, n_train=250,
                                              extra_features=["sentiment"])["pred"])
            out["models"]["GPR + FinBERT sentiment"] = pd.Series(gp_preds, index=test.index)
            nonzero = int((frame["sentiment"] != 0).sum())
            out["sentiment_coverage"] = {
                "rows_with_sentiment": nonzero, "total_rows": len(frame),
                "fraction": round(nonzero / max(len(frame), 1), 4)}
        except Exception as e:
            out.setdefault("errors", {})["GPR + FinBERT sentiment"] = f"{type(e).__name__}: {e}"

    actual = out["actual"].values
    out["metrics"] = {}
    for name, series in out["models"].items():
        err = series.values - actual
        entry = {"rmse": round(float(np.sqrt((err ** 2).mean())), 3),
                 "mae": round(float(np.abs(err).mean()), 3)}
        if name != "Naive (no change)":
            pred_dir = np.sign(series.values - test["Close"].values)
            true_dir = np.sign(actual - test["Close"].values)
            entry["directional_accuracy"] = round(float((pred_dir == true_dir).mean()), 4)
        out["metrics"][name] = entry

    naive_rmse = out["metrics"]["Naive (no change)"]["rmse"]
    for name, m in out["metrics"].items():
        m["vs_naive"] = round(m["rmse"] / naive_rmse, 3) if naive_rmse else None
        m["beats_naive"] = bool(m["rmse"] < naive_rmse)
    return out


def _gpr_forecast(features: pd.DataFrame, n_train: int = 250,
                  extra_features: list | None = None) -> dict:
    """Re-exported from finance.forecast_log so the comparison does not
    duplicate the one-step GP definition."""
    from finance.forecast_log import _gpr_forecast as impl

    return impl(features, n_train=n_train, extra_features=extra_features)


def compare_with_without_sentiment(features_df: pd.DataFrame, sentiment_series: pd.Series | None, test_size: int = 30) -> dict:
    """The headline ablation: same XGBoost model, sentiment feature zeroed
    out vs populated. Returns both RMSEs plus the ARIMA baseline RMSE."""
    frame_with = build_training_frame(features_df, sentiment_series)
    frame_without = build_training_frame(features_df, sentiment_series=None)

    result_with = train_xgboost_model(frame_with, test_size=test_size)
    result_without = train_xgboost_model(frame_without, test_size=test_size)
    arima_result = arima_baseline(features_df["Close"], test_size=test_size)

    return {
        "arima_rmse": arima_result["rmse"],
        "xgboost_no_sentiment_rmse": result_without["rmse"],
        "xgboost_with_sentiment_rmse": result_with["rmse"],
        "detail_with_sentiment": result_with,
        "detail_without_sentiment": result_without,
        "detail_arima": arima_result,
    }


if __name__ == "__main__":
    from price_data import get_price_history, add_technical_indicators

    df = get_price_history("AAPL", start="2023-01-01")
    feats = add_technical_indicators(df)
    comparison = compare_with_without_sentiment(feats, sentiment_series=None, test_size=30)
    print("ARIMA baseline RMSE:            ", comparison["arima_rmse"])
    print("XGBoost (no sentiment) RMSE:    ", comparison["xgboost_no_sentiment_rmse"])
    print("XGBoost (with sentiment) RMSE:  ", comparison["xgboost_with_sentiment_rmse"], "(sentiment=0 placeholder until finance/sentiment.py wired in)")
