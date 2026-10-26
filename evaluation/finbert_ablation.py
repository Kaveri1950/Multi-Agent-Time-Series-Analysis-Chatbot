"""Does FinBERT news sentiment improve the XGBoost price model?

The ablation the README describes but never reported a number for. Same
XGBoost, same technical-indicator features, same rows; the only difference is
one extra column, the day's average FinBERT tone of Times of India headlines
about the company (`data/cache/news/toi_company_scored.parquet`, 2,451
headlines, Jan 2018 to Jun 2020). Days with no headline get 0 (neutral), as in
finance/price_model.build_training_frame.

Headlines dated day t predict the close of day t+1, so no feature comes from
the day being forecast. The headline archive ends in June 2020 and the price
panel starts in January 2019, so every test window lies in that overlap.

12 stocks with the most headline days x 4 thirty-day test windows = 48 cases.
Each case reports RMSE for: naive ("tomorrow = today"), XGBoost without
sentiment, XGBoost with sentiment, plus the sentiment feature's importance and
how many training days had any headline.

Run:  python -m evaluation.finbert_ablation        (about a minute)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
SCORED = ROOT / "data" / "cache" / "news" / "toi_company_scored.parquet"
OUT_PATH = Path(__file__).parent / "results" / "finbert_ablation.json"
N_STOCKS, TEST_SIZE = 12, 30
WINDOW_ENDS = ["2019-09-30", "2019-12-31", "2020-03-31", "2020-06-30"]


def main():
    from scipy.stats import wilcoxon

    from finance.price_data import add_technical_indicators
    from finance.price_model import build_training_frame, train_xgboost_model
    from quant.data import load_panel

    close, _ = load_panel()
    news = pd.read_parquet(SCORED)
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    days = news.groupby("symbol")["date"].nunique().sort_values(ascending=False)
    symbols = [s for s in days.index if f"{s}.NS" in close.columns][:N_STOCKS]

    rows = []
    for sym in symbols:
        tone = news[news["symbol"] == sym].groupby("date")["tone"].mean()
        px = close[f"{sym}.NS"].dropna()
        for end in WINDOW_ENDS:
            s = px[px.index <= end]
            feats = add_technical_indicators(s.to_frame("Close"))
            with_ = build_training_frame(feats, tone)
            without = build_training_frame(feats, None)
            if len(with_) < TEST_SIZE + 60:
                continue
            r_with = train_xgboost_model(with_, TEST_SIZE)
            r_without = train_xgboost_model(without, TEST_SIZE)
            test = without[-TEST_SIZE:]
            naive = float(np.sqrt(((test["target_next_close"] - test["Close"]) ** 2).mean()))
            train_idx = with_.index[:-TEST_SIZE]
            rows.append({
                "symbol": sym, "test_end": end,
                "rmse_naive": round(naive, 3),
                "rmse_xgb": r_without["rmse"], "rmse_xgb_finbert": r_with["rmse"],
                "sentiment_importance": r_with["feature_importance"]["sentiment"],
                "training_days": int(len(train_idx)),
                "training_days_with_news": int(tone.index.isin(train_idx).sum()),
            })

    d = pd.DataFrame(rows)
    diff = d["rmse_xgb_finbert"] - d["rmse_xgb"]
    rel = d["rmse_xgb_finbert"] / d["rmse_xgb"] - 1
    try:
        p = float(wilcoxon(d["rmse_xgb_finbert"], d["rmse_xgb"]).pvalue)
    except ValueError:
        p = float("nan")
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "headlines": int(len(news)), "headline_span": f"{news['date'].min().date()} -> "
                                                      f"{news['date'].max().date()}",
        "n_cases": int(len(d)), "symbols": symbols, "window_ends": WINDOW_ENDS,
        "test_size": TEST_SIZE,
        "finbert_lower_rmse_in": int((diff < 0).sum()),
        "median_relative_change_pct": round(float(rel.median() * 100), 2),
        "mean_relative_change_pct": round(float(rel.mean() * 100), 2),
        "wilcoxon_p": round(p, 4),
        "mean_sentiment_importance": round(float(d["sentiment_importance"].mean()), 4),
        "median_share_of_training_days_with_news": round(float(
            (d["training_days_with_news"] / d["training_days"]).median()), 4),
        "xgb_beats_naive_in": int((d["rmse_xgb"] < d["rmse_naive"]).sum()),
        "xgb_finbert_beats_naive_in": int((d["rmse_xgb_finbert"] < d["rmse_naive"]).sum()),
        "median_rmse_ratio_to_naive": {
            "xgb": round(float((d["rmse_xgb"] / d["rmse_naive"]).median()), 3),
            "xgb_finbert": round(float((d["rmse_xgb_finbert"] / d["rmse_naive"]).median()), 3)},
        "cases": rows,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"FinBERT ablation: {out['n_cases']} cases ({len(symbols)} stocks x "
          f"{len(WINDOW_ENDS)} windows of {TEST_SIZE} days)")
    print(f"  sentiment lowered XGBoost RMSE in {out['finbert_lower_rmse_in']}/{out['n_cases']}; "
          f"median change {out['median_relative_change_pct']:+.2f}% (Wilcoxon p={out['wilcoxon_p']})")
    print(f"  sentiment feature importance (mean) {out['mean_sentiment_importance']:.4f}; "
          f"training days with any headline (median) "
          f"{out['median_share_of_training_days_with_news']:.1%}")
    print(f"  beats naive: XGBoost {out['xgb_beats_naive_in']}/{out['n_cases']}, "
          f"XGBoost+FinBERT {out['xgb_finbert_beats_naive_in']}/{out['n_cases']}; "
          f"median RMSE vs naive {out['median_rmse_ratio_to_naive']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
