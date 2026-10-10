"""Plotly chart builders for the Streamlit dashboard.

Ported from the original stock-prediction app
(stock/pages/utils/plotly_figure.py), adapted to read pre-computed
indicator columns from finance.price_data.add_technical_indicators()
instead of recomputing RSI/MACD/etc inline -- one source of truth for the
indicator math, shared by the Stock Analysis page, the Home dashboard, and
anything the chatbot cites.
"""

import dateutil.relativedelta
import pandas as pd
import plotly.graph_objects as go

PERIOD_MAP = {
    "5D": dict(days=-5), "1M": dict(months=-1), "6M": dict(months=-6),
    "YTD": None, "1Y": dict(years=-1), "5Y": dict(years=-5), "MAX": None,
}


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """df must have a DatetimeIndex. period is one of PERIOD_MAP's keys."""
    if period == "YTD":
        cutoff = pd.Timestamp(year=df.index[-1].year, month=1, day=1)
    elif period == "MAX" or period not in PERIOD_MAP or PERIOD_MAP[period] is None:
        return df
    else:
        cutoff = df.index[-1] + dateutil.relativedelta.relativedelta(**PERIOD_MAP[period])
    return df[df.index > cutoff]


def plotly_table(df: pd.DataFrame) -> go.Figure:
    header_color, odd_color, even_color = "grey", "#e1efff", "#f8fafd"
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>" + str(c)[:16] + "</b>" for c in df.columns],
            fill_color="white", line_color="#0078ff", height=32, align="center",
            font=dict(color="black", size=13),
        ),
        cells=dict(
            values=[["<b>" + str(i) + "</b>" for i in df.index]] + [df[c] for c in df.columns],
            fill_color=[[odd_color, even_color] * len(df)],
            align="left", line_color="white", font=dict(color="black", size=13),
        ),
    )])
    fig.update_layout(height=min(400, 40 + 32 * len(df)), margin=dict(l=0, r=0, t=0, b=0))
    return fig


def candlestick_chart(df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    d = filter_period(df, period)
    fig = go.Figure(data=[go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"])])
    fig.update_layout(showlegend=False, height=450, margin=dict(l=0, r=20, t=20, b=0),
                       plot_bgcolor="white", paper_bgcolor="#e1efff", xaxis_rangeslider_visible=True)
    return fig


def close_line_chart(df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    d = filter_period(df, period)
    fig = go.Figure()
    for col, color in [("Open", "#5ab7ff"), ("Close", "black"), ("High", "#0078ff"), ("Low", "red")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], mode="lines", name=col, line=dict(width=2, color=color)))
    fig.update_layout(height=450, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor="white",
                       paper_bgcolor="#e1efff", xaxis_rangeslider_visible=True,
                       legend=dict(yanchor="top", xanchor="right"))
    return fig


def rsi_chart(features_df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    """features_df must already have an 'RSI_14' column (finance.price_data.add_technical_indicators)."""
    d = filter_period(features_df, period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI_14"], name="RSI(14)", line=dict(width=2, color="orange")))
    fig.add_trace(go.Scatter(x=d.index, y=[70] * len(d), name="Overbought", line=dict(width=2, color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=d.index, y=[30] * len(d), name="Oversold", line=dict(width=2, color="green", dash="dash")))
    fig.update_layout(yaxis_range=[0, 100], height=220, plot_bgcolor="white", paper_bgcolor="#e1efff",
                       margin=dict(l=0, r=0, t=10, b=0))
    return fig


def macd_chart(features_df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    """features_df must already have 'MACD'/'MACD_signal' columns."""
    d = filter_period(features_df, period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", line=dict(width=2, color="orange")))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_signal"], name="Signal", line=dict(width=2, color="red", dash="dash")))
    fig.update_layout(height=220, plot_bgcolor="white", paper_bgcolor="#e1efff", margin=dict(l=0, r=0, t=10, b=0))
    return fig


def moving_average_chart(features_df: pd.DataFrame, period: str = "1Y") -> go.Figure:
    """features_df must already have 'SMA_7'/'SMA_21' columns."""
    d = filter_period(features_df, period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(width=2, color="black")))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA_7"], name="SMA(7)", line=dict(width=2, color="#0078ff")))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA_21"], name="SMA(21)", line=dict(width=2, color="#ff7f0e")))
    fig.update_layout(height=450, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor="white",
                       paper_bgcolor="#e1efff", xaxis_rangeslider_visible=True)
    return fig


def prediction_comparison_chart(actual: pd.Series, arima_pred: pd.Series, xgb_pred: pd.Series,
                                 xgb_sentiment_pred: pd.Series | None = None) -> go.Figure:
    """Overlay actual close price vs each model's held-out-period predictions."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Actual", line=dict(width=3, color="black")))
    fig.add_trace(go.Scatter(x=arima_pred.index, y=arima_pred.values, name="ARIMA baseline", line=dict(width=2, color="#5ab7ff", dash="dot")))
    fig.add_trace(go.Scatter(x=xgb_pred.index, y=xgb_pred.values, name="XGBoost (no sentiment)", line=dict(width=2, color="#ff7f0e", dash="dot")))
    if xgb_sentiment_pred is not None:
        fig.add_trace(go.Scatter(x=xgb_sentiment_pred.index, y=xgb_sentiment_pred.values,
                                  name="XGBoost + sentiment", line=dict(width=2, color="#2ca02c")))
    fig.update_layout(height=450, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor="white", paper_bgcolor="#e1efff",
                       legend=dict(yanchor="top", xanchor="left"))
    return fig


def future_forecast_chart(historical_close: pd.Series, forecast_df: pd.DataFrame, ticker: str,
                           history_days: int = 90) -> go.Figure:
    """Recent history + a genuine forward-looking forecast beyond the last
    known date (finance.price_model.forecast_future), with an 80%
    confidence band -- distinct from prediction_comparison_chart, which
    shows backtested (already-known) held-out periods."""
    recent = historical_close.tail(history_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent.values, name="Historical close",
                              line=dict(width=2, color="black")))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["upper_80"], name="80% CI",
                              line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["lower_80"], name="80% confidence interval",
                              line=dict(width=0), fill="tonexty", fillcolor="rgba(44,160,44,0.15)"))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["predicted_close"], name="Forecast",
                              line=dict(width=2, color="#2ca02c", dash="dot")))
    # Give the x-axis labels room, and pad the y-range to the full extent of
    # the band. With b=0 the tick labels had nowhere to sit, so Plotly
    # squeezed the plot area and clipped the bottom of the confidence band --
    # a forecast whose stated range reached 1865 was drawn on an axis that
    # stopped at 2000, hiding exactly the downside the band exists to show.
    lo = float(min(recent.min(), forecast_df["lower_80"].min()))
    hi = float(max(recent.max(), forecast_df["upper_80"].max()))
    pad = (hi - lo) * 0.06

    fig.update_layout(height=460, margin=dict(l=0, r=20, t=40, b=40),
                      plot_bgcolor="white", paper_bgcolor="#f0fff4",
                      title=f"{ticker} -- {len(forecast_df)}-trading-day forecast",
                      yaxis=dict(range=[lo - pad, hi + pad], title="Close"),
                      xaxis=dict(title=""),
                      legend=dict(yanchor="top", xanchor="left"))
    return fig


def model_comparison_chart(comparison: dict, ticker: str) -> go.Figure:
    """Every model's predictions against the actual closes, on one axis.

    The actual price is drawn heaviest and the naive baseline as a dashed
    grey line, deliberately. Three model lines hugging the actual price look
    impressive until "assume no change" is drawn alongside and turns out to
    hug it just as closely -- which is what the measurements say happens.
    Leaving the baseline off would make the chart flattering rather than
    informative.
    """
    actual = comparison["actual"]
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Actual close",
                             line=dict(width=3, color="black"), mode="lines+markers",
                             marker=dict(size=5)))

    styles = {
        "Naive (no change)": dict(color="#9aa0a6", dash="dash", width=2),
        "ARIMA": dict(color="#d62728", dash="dot", width=2),
        "XGBoost": dict(color="#ff7f0e", dash="dot", width=2),
        "XGBoost + FinBERT sentiment": dict(color="#9467bd", dash="dot", width=2),
        "GPR": dict(color="#2ca02c", dash="solid", width=2),
        "GPR + FinBERT sentiment": dict(color="#17becf", dash="dashdot", width=2),
    }
    metrics = comparison.get("metrics", {})
    for name, series in comparison["models"].items():
        rmse = metrics.get(name, {}).get("rmse")
        label = f"{name} (RMSE {rmse:.0f})" if rmse is not None else name
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=label,
                                 line=styles.get(name, dict(width=2, dash="dot")),
                                 mode="lines"))

    fig.update_layout(height=470, margin=dict(l=0, r=20, t=50, b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      title=f"{ticker} -- model predictions vs actual, "
                            f"held-out {comparison['test_size']} trading days",
                      yaxis=dict(title="Close"), xaxis=dict(title=""),
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                  bgcolor="rgba(255,255,255,0.75)"))
    return fig


def model_error_bar(comparison: dict) -> go.Figure:
    """RMSE per model, with the naive baseline marked.

    A bar chart makes the ordering unmissable in a way the overlay cannot:
    on the overlay every line looks close to the actual price, because a
    stock that moves 2% a day never strays far from where it started.
    """
    metrics = comparison.get("metrics", {})
    names = list(metrics)
    rmses = [metrics[n]["rmse"] for n in names]
    naive = metrics.get("Naive (no change)", {}).get("rmse")

    colors = ["#9aa0a6" if n == "Naive (no change)"
              else ("#2ca02c" if naive and metrics[n]["rmse"] < naive else "#d62728")
              for n in names]

    fig = go.Figure(go.Bar(x=names, y=rmses, marker_color=colors,
                           text=[f"{v:.0f}" for v in rmses], textposition="outside"))
    if naive:
        fig.add_hline(y=naive, line_dash="dash", line_color="#9aa0a6",
                      annotation_text="naive baseline — anything above this is worse than "
                                      "assuming no change")
    fig.update_layout(height=340, margin=dict(l=0, r=20, t=50, b=60),
                      plot_bgcolor="white", paper_bgcolor="white",
                      title="Prediction error (RMSE) — lower is better",
                      yaxis=dict(title="RMSE"), showlegend=False)
    return fig


def finance_anomaly_chart(features_df: pd.DataFrame, scored_df: pd.DataFrame, ticker: str, period: str = "1Y") -> go.Figure:
    """Close price with flagged anomalous days marked as diamonds, color
    coded by severity (finance.anomaly_detector.detect_all). Hovering a
    marker shows the actual detector reasons (return/volatility/volume/gap/
    sentiment-divergence), not just that something was flagged."""
    d = filter_period(features_df, period)
    scored_d = scored_df.loc[scored_df.index.intersection(d.index)]
    anomalies = scored_d[scored_d["is_anomaly"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close", line=dict(width=2, color="black")))

    for severity, color in [("LOW", "#f4d35e"), ("MEDIUM", "#ee964b"), ("HIGH", "#d62728")]:
        subset = anomalies[anomalies["severity_label"] == severity]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset.index, y=d.loc[subset.index, "Close"], mode="markers", name=f"{severity} severity",
            marker=dict(size=11, color=color, symbol="diamond", line=dict(width=1, color="black")),
            text=[", ".join(r) for r in subset["reasons"]], hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))

    fig.update_layout(height=450, margin=dict(l=0, r=20, t=30, b=0), plot_bgcolor="white", paper_bgcolor="#fff5f5",
                       title=f"{ticker} -- flagged anomalous days", xaxis_rangeslider_visible=True,
                       legend=dict(yanchor="top", xanchor="left"))
    return fig


def rmse_comparison_bar(labels: list[str], rmses: list[float]) -> go.Figure:
    colors = ["#5ab7ff", "#ff7f0e", "#2ca02c"][: len(labels)]
    fig = go.Figure(data=[go.Bar(x=labels, y=rmses, marker_color=colors, text=[f"{r:.2f}" for r in rmses], textposition="outside")])
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white", paper_bgcolor="white",
                       yaxis_title="RMSE (lower is better)")
    return fig


def dataset_anomaly_overview_chart(index_df: pd.DataFrame) -> go.Figure:
    """Bar chart of anomaly rate per target across a whole downloaded
    light-curve dataset (astronomy/dataset.py), color-coded by whether the
    underlying data was real (MAST/Kepler) or a synthetic fallback."""
    colors = ["#2ca02c" if str(s).startswith("real") else "#d62728" for s in index_df["source"]]
    fig = go.Figure(data=[go.Bar(
        x=index_df["target"], y=index_df["anomaly_rate"], marker_color=colors,
        text=[f"{r:.0%}" for r in index_df["anomaly_rate"]], textposition="outside",
    )])
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=80), plot_bgcolor="white", paper_bgcolor="white",
                       yaxis_title="Anomaly rate", xaxis_tickangle=-45,
                       title="Dataset overview -- green = real MAST data, red = synthetic fallback")
    return fig


def raw_lightcurve_chart(time, flux) -> go.Figure:
    """Plain flux-vs-time, no shading -- step 1 of the Astronomy page's
    dashboard flow, before any detection has run."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=flux, mode="lines", name="Flux", line=dict(width=1, color="#0078ff")))
    fig.update_layout(height=380, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Time", yaxis_title="Normalized flux", title="Raw light curve")
    return fig


def _merge_flagged_spans(anomaly_mask, window_size: int, stride: int, n_times: int):
    """Contiguous flagged windows -> one span each.

    Windows overlap heavily (stride 6, width 24 means four windows cover any
    given point), so a single transit produces a run of consecutive flags
    that should be drawn as ONE band rather than four stacked translucent
    rectangles. Merging is both more honest visually and far cheaper.
    """
    spans, start = [], None
    for i, flagged in enumerate(anomaly_mask):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(anomaly_mask) - 1))

    return [(min(a * stride, n_times - 1), min(b * stride + window_size, n_times - 1))
            for a, b in spans]


MAX_SHADED_SPANS = 400


def lightcurve_anomaly_chart(time, flux, window_size: int, stride: int, anomaly_mask) -> go.Figure:
    """Raw flux-vs-time with each anomalous region's time range shaded.

    Shapes are assembled once and assigned in a single update_layout call.
    fig.add_vrect() re-validates the entire accumulated shapes tuple on every
    call, making it quadratic: 50 windows took 1.6s, 400 took 178.8s, and the
    1,799 that a four-year stitched light curve produces would have run for
    hours. This was invisible while windows were a fixed 200 points over ~30
    days, which never yielded more than a few dozen flags.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=flux, mode="lines", name="Flux", line=dict(width=1, color="#0078ff")))

    spans = _merge_flagged_spans(anomaly_mask, window_size, stride, len(time))
    truncated = len(spans) > MAX_SHADED_SPANS
    if truncated:
        spans = spans[:MAX_SHADED_SPANS]

    fig.update_layout(
        shapes=[dict(type="rect", xref="x", yref="paper",
                     x0=time[a], x1=time[b], y0=0, y1=1,
                     fillcolor="red", opacity=0.15, line_width=0, layer="below")
                for a, b in spans],
        height=450, margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor="white", paper_bgcolor="#fff5f5",
        xaxis_title="Time", yaxis_title="Normalized flux",
        title="Flux vs time (shaded = flagged anomalous region)"
        + (f" -- first {MAX_SHADED_SPANS} of {len(_merge_flagged_spans(anomaly_mask, window_size, stride, len(time)))} shown"
           if truncated else ""),
    )
    return fig


def lightcurve_zoom_chart(time, flux, centre_index: int, window_size: int,
                          context_factor: float = 6.0, title: str = "") -> go.Figure:
    """Zoom on one flagged region, with surrounding context.

    A four-year light curve plotted whole is 65,000 points in a few hundred
    pixels, so an individual transit is roughly one pixel deep and invisible.
    Showing WHY a window was flagged requires actually looking at it, which
    means a zoom -- with enough context either side to see that the dip is a
    departure from the local baseline rather than the baseline itself.
    """
    half = int(window_size * context_factor / 2)
    lo = max(0, centre_index - half)
    hi = min(len(time) - 1, centre_index + half)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time[lo:hi], y=flux[lo:hi], mode="lines+markers",
                             name="Flux", line=dict(width=1, color="#0078ff"),
                             marker=dict(size=3)))
    fig.add_vrect(x0=time[centre_index],
                  x1=time[min(centre_index + window_size, len(time) - 1)],
                  fillcolor="red", opacity=0.18, line_width=0)
    fig.update_layout(height=320, margin=dict(l=0, r=20, t=40, b=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="Time (BKJD days)", yaxis_title="Normalized flux",
                      title=title or "Flagged window (shaded), in context")
    return fig


def normalized_lightcurve_chart(time, flux, window_size: int, stride: int, anomaly_mask) -> go.Figure:
    """Flux re-expressed as a z-score (std deviations from the mean) rather
    than raw normalized-to-1.0 flux -- makes dip/flare magnitude directly
    comparable across targets and visually ties back to the rolling
    z-score thresholding style already used on the finance anomaly side
    (finance/anomaly_detector.py), instead of introducing a second,
    unrelated notion of "normalized" for this domain."""
    import numpy as np

    flux = np.asarray(flux, dtype=float)
    std = flux.std()
    z = (flux - flux.mean()) / std if std > 0 else flux * 0.0

    fig = go.Figure()
    for level, color in [(3, "#d62728"), (2, "#ee964b")]:
        fig.add_trace(go.Scatter(x=[time[0], time[-1]], y=[level, level], mode="lines", showlegend=False,
                                  line=dict(width=1, color=color, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[time[0], time[-1]], y=[-level, -level], mode="lines", showlegend=False,
                                  line=dict(width=1, color=color, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=time, y=z, mode="lines", name="Normalized flux (z-score)",
                              line=dict(width=1, color="#0078ff")))

    n_windows = len(anomaly_mask)
    for i in range(n_windows):
        if not anomaly_mask[i]:
            continue
        start_idx = i * stride
        end_idx = min(start_idx + window_size, len(time) - 1)
        fig.add_vrect(x0=time[start_idx], x1=time[end_idx], fillcolor="red", opacity=0.15, line_width=0)

    fig.update_layout(height=380, margin=dict(l=0, r=20, t=30, b=0), plot_bgcolor="white", paper_bgcolor="#f5f9ff",
                       xaxis_title="Time", yaxis_title="Flux (std deviations from mean)",
                       title="Normalized flux (dashed lines = ±2σ / ±3σ, shaded = flagged window)")
    return fig


def nav_chart(nav_df: pd.DataFrame, forecast: pd.Series | None = None) -> go.Figure:
    """Mutual fund NAV history, optionally with an ARIMA forecast overlay
    (finance.mutual_funds.forecast_nav's held-out-period prediction)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df["nav"], name="NAV", line=dict(width=2, color="black")))
    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="ARIMA forecast (held-out period)",
                                  line=dict(width=2, color="#2ca02c", dash="dot")))
    fig.update_layout(height=450, margin=dict(l=0, r=20, t=20, b=0), plot_bgcolor="white",
                       paper_bgcolor="#f0fff4", xaxis_rangeslider_visible=True)
    return fig


def reconstruction_error_chart(window_ids, errors) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(window_ids), y=list(errors), marker_color="#d62728"))
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white", paper_bgcolor="white",
                       xaxis_title="Window ID", yaxis_title="Autoencoder reconstruction error")
    return fig


# ==========================================================================
# Cross-domain anomaly detector comparison (anomaly/grid.py)
# ==========================================================================

#: Group colours, ordered by how much machinery each family brings. The
#: floors are deliberately grey: they are the thing to beat, not a competitor.
DETECTOR_GROUP_COLORS = {
    "floor": "#9aa0a6",
    "one_liner": "#1f77b4",
    "threshold": "#17a2b8",
    "density": "#6f42c1",
    "reconstruction": "#e8590c",
    "latent_state": "#2ca02c",
    "ceiling": "#343a40",
}


def detector_grid_bar(rows: list[dict], domain: str, metric: str = "pr_auc") -> go.Figure:
    """Detectors ranked on one domain, with the no-skill floor drawn across.

    The floor line is the whole point of the chart. A PR-AUC of 0.295 means
    nothing in isolation; drawn against the best score achievable with no
    learning at all, it either clears the line or it does not, and the reader
    can see which without being told.
    """
    subset = [r for r in rows if r.get("domain") == domain
              and isinstance(r.get(metric), (int, float)) and r.get(metric) == r.get(metric)]
    subset = sorted(subset, key=lambda r: r[metric])
    if not subset:
        return go.Figure()

    names = [r["detector"] for r in subset]
    values = [r[metric] for r in subset]
    colors = [DETECTOR_GROUP_COLORS.get(r.get("group"), "#888") for r in subset]

    floors = [r[metric] for r in subset if r.get("is_floor")]
    best_floor = max(floors) if floors else None
    prevalence = subset[0].get("prevalence")

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        hovertext=[f"{r['detector']}<br>group: {r.get('group')}<br>"
                   f"ROC-AUC {r.get('roc_auc')}<br>lift {r.get('lift')}x" for r in subset],
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    if best_floor is not None:
        fig.add_vline(x=best_floor, line_dash="dash", line_color="#d62728",
                      annotation_text="best no-skill floor — anything left of this "
                                      "has not learned anything",
                      annotation_position="top")
    if metric == "pr_auc" and prevalence:
        # PR-AUC's floor is the base rate, which is what makes it readable
        # under imbalance where ROC-AUC's fixed 0.5 floor is not.
        fig.add_vline(x=prevalence, line_dash="dot", line_color="#9aa0a6",
                      annotation_text=f"chance ({prevalence:.1%})",
                      annotation_position="bottom")

    label = {"pr_auc": "PR-AUC (average precision)", "roc_auc": "ROC-AUC",
             "lift": "lift over chance"}.get(metric, metric)
    fig.update_layout(
        height=max(360, 30 * len(names) + 110),
        margin=dict(l=0, r=90, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        title=f"{domain} — {label}, higher is better",
        xaxis=dict(title=label, gridcolor="#eee"), showlegend=False,
    )
    return fig


def detector_domain_scatter(rows: list[dict], metric: str = "pr_auc") -> go.Figure:
    """Each detector's score on one domain against the other.

    The chart that answers the generalisation question directly. A detector
    that genuinely captures something about "anomalous" sits high on both
    axes; one that happens to suit a single dataset sits off-diagonal. Scores
    are shown as lift so two domains with very different base rates (2.2% vs
    21.8%) can share a plot at all.
    """
    by_detector: dict[str, dict] = {}
    for r in rows:
        by_detector.setdefault(r["detector"], {})[r["domain"]] = r

    domains = sorted({r["domain"] for r in rows})
    if len(domains) < 2:
        return go.Figure()
    dx, dy = domains[0], domains[1]

    xs, ys, names, colors = [], [], [], []
    for name, per_domain in by_detector.items():
        a, b = per_domain.get(dx), per_domain.get(dy)
        if not a or not b:
            continue
        va, vb = a.get("lift"), b.get("lift")
        if va is None or vb is None or va != va or vb != vb:
            continue
        xs.append(va); ys.append(vb); names.append(name)
        colors.append(DETECTOR_GROUP_COLORS.get(a.get("group"), "#888"))

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=names, textposition="top center",
        textfont=dict(size=9),
        marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
        hovertemplate="%{text}<br>" + f"{dx}: " + "%{x:.2f}x<br>" + f"{dy}: " + "%{y:.2f}x<extra></extra>",
    ))
    # Lift of 1.0 is chance in either direction.
    fig.add_vline(x=1.0, line_dash="dot", line_color="#d62728")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#d62728")
    fig.update_layout(
        height=520, margin=dict(l=0, r=20, t=60, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        title="Does a detector that works on one domain work on the other?",
        xaxis=dict(title=f"{dx} — lift over chance", gridcolor="#eee"),
        yaxis=dict(title=f"{dy} — lift over chance", gridcolor="#eee"),
        showlegend=False,
    )
    return fig


def audit_verdict_table(audits: dict) -> go.Figure:
    """Wu & Keogh's four flaws, per dataset, as a coloured grid."""
    flaw_order = ["triviality", "unrealistic_anomaly_density",
                  "run_to_failure_bias", "mislabeled_ground_truth"]
    pretty = {"triviality": "Triviality",
              "unrealistic_anomaly_density": "Anomaly density",
              "run_to_failure_bias": "Run-to-failure bias",
              "mislabeled_ground_truth": "Label quality"}
    shade = {"pass": "#d4edda", "warn": "#fff3cd", "fail": "#f8d7da",
             "review": "#fff3cd", "insufficient_data": "#e9ecef",
             "not_assessable": "#e9ecef"}

    domains = list(audits)
    cells, fills = [], []
    for d in domains:
        verdicts = {c["flaw"]: c.get("verdict", "—") for c in audits[d]["checks"]}
        cells.append([verdicts.get(f, "—").upper() for f in flaw_order])
        fills.append([shade.get(verdicts.get(f), "#e9ecef") for f in flaw_order])

    fig = go.Figure(go.Table(
        header=dict(values=["<b>Flaw</b>"] + [f"<b>{d}</b>" for d in domains],
                    fill_color="white", line_color="#0078ff", height=32,
                    align="left", font=dict(color="black", size=13)),
        cells=dict(values=[[pretty[f] for f in flaw_order]] + cells,
                   fill_color=[["#f8fafd"] * len(flaw_order)] + fills,
                   align="left", line_color="white", height=30,
                   font=dict(color="black", size=13)),
    ))
    fig.update_layout(height=40 + 32 * (len(flaw_order) + 1), margin=dict(l=0, r=0, t=0, b=0))
    return fig


def series_anomaly_chart(x, y, window_size: int, stride: int, anomaly_mask,
                         y_true_mask=None, title: str = "", y_label: str = "value") -> go.Figure:
    """Any 1-D series with flagged regions shaded -- and, where known, the
    regions that were actually anomalous drawn underneath.

    The second band is what makes this an evidence chart rather than a
    demo. Red-only shading shows what the detector claimed; overlaying the
    ground truth shows where it was right and, more usefully, where it was
    not. A reader can count the misses without being told about them.

    Generic over x: works with floats (barycentric days) or a DatetimeIndex,
    so one chart serves both light curves and price series.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=y_label,
                             line=dict(width=1, color="#0078ff")))

    shapes = []
    if y_true_mask is not None:
        for a, b in _merge_flagged_spans(y_true_mask, window_size, stride, len(x))[:MAX_SHADED_SPANS]:
            shapes.append(dict(type="rect", xref="x", yref="paper",
                               x0=x[a], x1=x[b], y0=0, y1=0.06,
                               fillcolor="#2ca02c", opacity=0.55, line_width=0, layer="below"))
    flagged = _merge_flagged_spans(anomaly_mask, window_size, stride, len(x))
    truncated = len(flagged) > MAX_SHADED_SPANS
    for a, b in flagged[:MAX_SHADED_SPANS]:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=x[a], x1=x[b], y0=0, y1=1,
                           fillcolor="red", opacity=0.15, line_width=0, layer="below"))

    caption = title or "Series with flagged regions"
    if y_true_mask is not None:
        caption += "  —  red = detector flagged, green strip = actually anomalous"
    if truncated:
        caption += f"  (first {MAX_SHADED_SPANS} of {len(flagged)} shown)"

    fig.update_layout(shapes=shapes, height=430, margin=dict(l=0, r=20, t=50, b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis_title="", yaxis_title=y_label, title=caption, showlegend=False)
    return fig


def window_zoom_chart(x, y, start_index: int, window_size: int,
                      context_factor: float = 6.0, title: str = "",
                      y_label: str = "value") -> go.Figure:
    """One flagged window in close-up, with surrounding context for scale.

    A flagged region viewed alone always looks dramatic, because the y-axis
    rescales to it. Showing several window-widths either side is what lets a
    reader judge whether the excursion is actually unusual for this series.
    """
    n = len(x)
    pad = int(window_size * context_factor)
    lo = max(0, start_index - pad)
    hi = min(n - 1, start_index + window_size + pad)
    w_lo, w_hi = max(0, start_index), min(n - 1, start_index + window_size)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[lo:hi], y=y[lo:hi], mode="lines",
                             line=dict(width=1.2, color="#0078ff"), name=y_label))
    fig.add_trace(go.Scatter(x=x[w_lo:w_hi], y=y[w_lo:w_hi], mode="lines",
                             line=dict(width=2.4, color="#d62728"), name="flagged window"))
    fig.update_layout(
        shapes=[dict(type="rect", xref="x", yref="paper", x0=x[w_lo], x1=x[w_hi],
                     y0=0, y1=1, fillcolor="red", opacity=0.10, line_width=0, layer="below")],
        height=330, margin=dict(l=0, r=20, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        title=title or "Flagged window in context", yaxis_title=y_label,
        showlegend=True, legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def feature_deviation_bar(deviation_row, top_k: int = 8) -> go.Figure:
    """Which features drove this window's score, in robust sigmas.

    The visual form of the reason text: a reader can see at a glance whether
    one feature dominated (a clean, interpretable detection) or whether the
    window was flagged on a diffuse combination (much weaker evidence).
    """
    items = sorted(((k, float(v)) for k, v in deviation_row.items()),
                   key=lambda kv: -abs(kv[1]))[:top_k][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d62728" if abs(v) >= 2 else "#9aa0a6" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors,
                           text=[f"{v:+.1f}σ" for v in vals], textposition="outside"))
    fig.add_vline(x=2, line_dash="dot", line_color="#d62728")
    fig.add_vline(x=-2, line_dash="dot", line_color="#d62728")
    fig.update_layout(height=max(240, 26 * len(names) + 80),
                      margin=dict(l=0, r=60, t=40, b=30),
                      plot_bgcolor="white", paper_bgcolor="white",
                      title="Feature deviations (robust σ) — dotted lines mark ±2σ",
                      xaxis=dict(title="σ from population median", gridcolor="#eee"),
                      showlegend=False)
    return fig


def dual_series_anomaly_chart(x, real_y, model_y, window_size: int, stride: int,
                              anomaly_mask, y_true_mask=None,
                              real_label: str = "actual", model_label: str = "what the detector saw",
                              title: str = "") -> go.Figure:
    """Two stacked panels: the quantity a human recognises, and the quantity
    the detector was actually given.

    Worth the extra panel because the transformation that makes a series
    analysable also makes it unreadable. A detector operating on daily
    RETURNS sees the 2020 crash as one spike near -0.09; a person looking for
    the crash expects to see an index fall off a cliff. Showing only the
    model's view hides what happened; showing only the real view hides what
    was measured. The shading is identical on both, so a flagged region can
    be read in either language.

    Panels share an x-axis, so zooming or panning one moves the other.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.55, 0.45],
                        subplot_titles=(real_label, model_label))
    fig.add_trace(go.Scatter(x=x, y=real_y, mode="lines", name=real_label,
                             line=dict(width=1.3, color="#1f4e79")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=model_y, mode="lines", name=model_label,
                             line=dict(width=1, color="#0078ff")), row=2, col=1)

    shapes = []
    if y_true_mask is not None:
        for a, b in _merge_flagged_spans(y_true_mask, window_size, stride, len(x))[:MAX_SHADED_SPANS]:
            # yref "paper" would span both panels, so each band is drawn against
            # its own y-axis domain instead.
            shapes.append(dict(type="rect", xref="x", yref="y domain",
                               x0=x[a], x1=x[b], y0=0, y1=0.07,
                               fillcolor="#2ca02c", opacity=0.6, line_width=0, layer="below"))
    for a, b in _merge_flagged_spans(anomaly_mask, window_size, stride, len(x))[:MAX_SHADED_SPANS]:
        for axis in ("y", "y2"):
            shapes.append(dict(type="rect", xref="x", yref=f"{axis} domain",
                               x0=x[a], x1=x[b], y0=0, y1=1,
                               fillcolor="red", opacity=0.14, line_width=0, layer="below"))

    fig.update_layout(shapes=shapes, height=560, margin=dict(l=0, r=20, t=60, b=40),
                      plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                      title=title or "Flagged regions, in real units and as the detector saw them")
    fig.update_xaxes(gridcolor="#f0f0f0")
    fig.update_yaxes(gridcolor="#f0f0f0")
    return fig


def risk_price_chart(ticker: str, profile: dict, lookback_days: int = 365):
    """Price over the window, with unusual days marked and attributed.

    Two marker colours, not one: a day when the whole market fell is a
    different fact from a day when only this company did, and collapsing them
    loses the distinction the analysis exists to draw.
    """
    import pandas as pd

    from quant.data import load_panel

    close, _ = load_panel()
    if ticker not in close.columns:
        return None
    series = close[ticker].dropna()
    series = series.loc[series.index[-1] - pd.Timedelta(f"{lookback_days}D"):]
    if series.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             name=ticker.replace(".NS", ""),
                             line=dict(width=1.4, color="#1f4e79")))

    events = profile.get("anomalies", {}).get("events", [])
    for kind, colour, symbol in (("market-wide", "#9aa0a6", "circle"),
                                 ("company-specific", "#d62728", "diamond")):
        pts = [e for e in events if e.get("kind") == kind]
        if not pts:
            continue
        dates = [pd.Timestamp(e["date"]) for e in pts]
        prices, texts = [], []
        for e, d in zip(pts, dates):
            nearest = series.index[series.index.get_indexer([d], method="nearest")][0]
            prices.append(float(series.loc[nearest]))
            news = (e.get("news") or [None])[0]
            texts.append(f"{e['date']}<br>total {e['return_pct']:+.2f}%"
                         f"<br>market {e['market_pct']:+.2f}%"
                         f"<br>company {e['idiosyncratic_pct']:+.2f}%"
                         + (f"<br>{news[:60]}" if news else ""))
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="markers", name=kind,
            marker=dict(size=11, color=colour, symbol=symbol,
                        line=dict(width=1, color="white")),
            text=texts, hovertemplate="%{text}<extra></extra>",
        ))

    dd = profile.get("drawdown", {})
    title = f"{ticker.replace('.NS', '')} — closing price with unusual days marked"
    if dd.get("max_drawdown_pct"):
        title += f"  (worst fall {abs(dd['max_drawdown_pct']):.0f}%)"
    fig.update_layout(height=380, margin=dict(l=0, r=20, t=50, b=40),
                      plot_bgcolor="white", paper_bgcolor="white",
                      title=title, yaxis_title="price",
                      legend=dict(orientation="h", y=1.1, x=0))
    return fig


def risk_attribution_chart(profile: dict):
    """Each unusual day split into its market part and its company part.

    The bar that matters is the second one. A -6% day is unremarkable if the
    market fell 6%; it is news if the market did not move at all.
    """
    events = profile.get("anomalies", {}).get("events", [])
    if not events:
        return None
    events = sorted(events, key=lambda e: e["date"])
    dates = [e["date"] for e in events]
    market = [e["market_pct"] for e in events]
    idio = [e["idiosyncratic_pct"] for e in events]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=market, name="the market moved",
                         marker_color="#9aa0a6"))
    fig.add_trace(go.Bar(x=dates, y=idio, name="the company moved",
                         marker_color="#d62728"))
    fig.add_hline(y=0, line_color="#333", line_width=1)
    fig.update_layout(height=320, margin=dict(l=0, r=20, t=50, b=40),
                      plot_bgcolor="white", paper_bgcolor="white", barmode="relative",
                      title="What drove each unusual day",
                      yaxis_title="contribution to the day's return (%)",
                      legend=dict(orientation="h", y=1.14, x=0))
    return fig


def volatility_forecast_chart(history: pd.DataFrame, forecast: float, naive: float,
                              ticker: str) -> go.Figure:
    """Recent realized volatility, with tomorrow's forecast and the naive baseline.

    The baseline is drawn, not just mentioned. Every price model in this
    project loses to carry-forward, so a volatility forecast shown WITHOUT
    the number it has to beat invites exactly the misreading the rest of the
    work exists to prevent.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["date"], y=history["realized_vol"], mode="lines",
        name="realized volatility", line=dict(color="#4A5568", width=1.4)))

    last = history["date"].iloc[-1]
    nxt = last + pd.Timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[last, nxt], y=[history["realized_vol"].iloc[-1], naive], mode="lines+markers",
        name="naive (carry forward)", line=dict(color="#6B7482", width=1.6, dash="dash"),
        marker=dict(size=7)))
    fig.add_trace(go.Scatter(
        x=[last, nxt], y=[history["realized_vol"].iloc[-1], forecast], mode="lines+markers",
        name="HAR+overnight forecast", line=dict(color="#C8621E", width=2.2),
        marker=dict(size=9)))

    fig.update_layout(
        title=f"{ticker} — next-day volatility forecast",
        yaxis_title="annualised volatility", xaxis_title=None,
        height=380, margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    return fig


def volatility_backtest_chart(bt: pd.DataFrame, ticker: str) -> go.Figure:
    """Walk-forward backtest: what was actually realized, against both forecasts.

    The actual series is drawn thin and grey because it is the noisy one --
    a daily range estimate of a latent quantity -- and the forecast is drawn
    bold because it is the claim being tested. Naive sits between them, and
    is always present: the question a backtest answers is not "does the line
    look close" but "is it closer than doing nothing".
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], mode="lines",
                             name="actual realized volatility",
                             line=dict(color="#9AA3AF", width=1.2)))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["naive"], mode="lines",
                             name="naive (yesterday repeated)",
                             line=dict(color="#6B7482", width=1.1, dash="dot")))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["har_ovn"], mode="lines",
                             name="HAR+overnight forecast",
                             line=dict(color="#C8621E", width=2.4)))
    fig.update_layout(
        title=f"{ticker} — walk-forward backtest, refitted every day",
        yaxis_title="annualised volatility", xaxis_title=None, height=420,
        margin=dict(l=10, r=10, t=46, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_yaxes(tickformat=".0%")
    return fig


def volatility_models_chart(bt, models: list, labels: dict, ticker: str) -> go.Figure:
    """Several models against the same realized series.

    Actual is drawn thin and grey because it is the noisy quantity -- a
    single day's range is one draw from that day's true volatility, so a
    model that tracked it exactly would be fitting noise. Naive is dotted
    and always shown: the question is whether a model is closer than doing
    nothing, not whether it hugs the grey line.
    """
    palette = ["#C8621E", "#2B5F8A", "#1F7A5A", "#8A4FA0", "#B03A48",
               "#7A6A1F", "#3A6EA5", "#A0522D"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["actual"], mode="lines",
                             name="actual realized volatility",
                             line=dict(color="#9AA3AF", width=1.2)))
    if "naive" in bt:
        fig.add_trace(go.Scatter(x=bt["date"], y=bt["naive"], mode="lines",
                                 name=labels.get("naive", "naive"),
                                 line=dict(color="#6B7482", width=1.0, dash="dot")))
    i = 0
    for m in models:
        if m in ("naive",) or m not in bt:
            continue
        fig.add_trace(go.Scatter(x=bt["date"], y=bt[m], mode="lines",
                                 name=labels.get(m, m),
                                 line=dict(color=palette[i % len(palette)], width=2.0)))
        i += 1
    fig.update_layout(
        title=f"{ticker} — models against the same realized series",
        yaxis_title="annualised volatility", xaxis_title=None, height=440,
        margin=dict(l=10, r=10, t=46, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_yaxes(tickformat=".0%")
    return fig
