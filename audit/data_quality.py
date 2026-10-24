"""Data-quality controls on the market data every model depends on.

Each check is a rule with a threshold, run over the cached price, volume and
OHLC panels. The result is a pass/fail log: what was tested, how many rows
broke the rule, a few examples, and whether that breaches the threshold.
Nothing is repaired here. A control that silently fixes data hides the
problem; this one reports it and leaves the decision to whoever owns it.

Run:  python -m audit.data_quality
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
QUANT = ROOT / "data" / "cache" / "quant"
OUT_PATH = ROOT / "evaluation" / "results" / "data_quality.json"
STALE_DAYS = 10          # calendar days since the last bar before the cache counts as stale
JUMP = 0.40              # a one-day move beyond +/-40% is flagged for review, not deleted


def _check(name, description, broken, total, threshold, examples, severity="high"):
    rate = broken / total if total else 0.0
    return {"check": name, "description": description, "rows_tested": int(total),
            "rows_failed": int(broken), "fail_rate": round(rate, 6), "threshold": threshold,
            "status": "pass" if rate <= threshold else "fail", "severity": severity,
            "examples": examples[:5]}


def _ex(frame: pd.DataFrame, n=5):
    return [{k: (str(v.date()) if isinstance(v, pd.Timestamp) else
                 (round(float(v), 4) if isinstance(v, (float, np.floating)) else v))
             for k, v in r.items()} for r in frame.head(n).to_dict("records")]


def run(today: pd.Timestamp | None = None) -> dict:
    close = pd.read_parquet(QUANT / "price_panel.parquet")
    volume = pd.read_parquet(QUANT / "volume_panel.parquet")
    ohlc = pd.read_parquet(QUANT / "ohlc_panel.parquet")
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    today = today or pd.Timestamp.now().normalize()
    checks = []

    # 1. OHLC consistency: low <= open, close <= high
    bad = ohlc[(ohlc["low"] > ohlc[["open", "close"]].min(axis=1) + 1e-9)
               | (ohlc["high"] < ohlc[["open", "close"]].max(axis=1) - 1e-9)]
    checks.append(_check("ohlc_consistency", "low <= open, close <= high on every bar",
                         len(bad), len(ohlc), 0.0, _ex(bad[["date", "ticker", "open", "high",
                                                            "low", "close"]])))

    # 2. Non-positive prices
    long_close = close.stack().rename("close").reset_index()
    long_close.columns = ["date", "ticker", "close"]
    bad = long_close[long_close["close"] <= 0]
    checks.append(_check("positive_prices", "every closing price is above zero",
                         len(bad), len(long_close), 0.0, _ex(bad)))

    # 3. Duplicate dates / duplicate (date, ticker) rows
    dup = ohlc[ohlc.duplicated(["date", "ticker"], keep=False)]
    checks.append(_check("no_duplicates", "one bar per stock per day",
                         len(dup) + int(close.index.duplicated().sum()),
                         len(ohlc) + len(close), 0.0, _ex(dup[["date", "ticker"]])))

    # 4. Flat bars: high == low with volume reported (suspended or stale quotes)
    flat = ohlc[(ohlc["high"] == ohlc["low"])]
    checks.append(_check("flat_bars", "high equals low (no trading range): suspended or "
                         "stale quote", len(flat), len(ohlc), 0.001,
                         _ex(flat[["date", "ticker", "close"]]), severity="medium"))

    # 5. Extreme one-day moves (could be splits not adjusted, or a bad print)
    r = close.pct_change(fill_method=None)
    jumps = r.stack()
    jumps = jumps[jumps.abs() > JUMP].rename("return").reset_index()
    jumps.columns = ["date", "ticker", "return"]
    checks.append(_check("extreme_moves", f"one-day move beyond +/-{JUMP:.0%}: unadjusted "
                         "split or bad print, to review", len(jumps), int(r.notna().sum().sum()),
                         0.0005, _ex(jumps.sort_values("return")), severity="medium"))

    # 6. Missing days inside a stock's trading life (gaps, not pre-listing NaN)
    first, last = close.apply(pd.Series.first_valid_index), close.apply(pd.Series.last_valid_index)
    gaps, alive = [], 0
    for t in close.columns:
        if first[t] is None:
            continue
        s = close.loc[first[t]:last[t], t]
        alive += len(s)
        n = int(s.isna().sum())
        if n:
            gaps.append({"ticker": t, "missing_days": n})
    gaps.sort(key=lambda g: -g["missing_days"])
    checks.append(_check("missing_days", "no gaps inside a stock's listed history",
                         sum(g["missing_days"] for g in gaps), alive, 0.005, gaps,
                         severity="medium"))

    # 7. Zero volume on a day with a price
    zv = ((volume == 0) & close.notna()).stack()
    zv = zv[zv].reset_index().iloc[:, :2]
    zv.columns = ["date", "ticker"]
    checks.append(_check("zero_volume", "a traded price with zero volume reported",
                         len(zv), int(close.notna().sum().sum()), 0.001, _ex(zv),
                         severity="low"))

    # 8. Price and OHLC panels agree on the close
    m = ohlc.merge(long_close, on=["date", "ticker"], suffixes=("", "_panel"))
    diff = m[(m["close"] / m["close_panel"] - 1).abs() > 0.01]
    checks.append(_check("panels_agree", "OHLC close within 1% of the price-panel close",
                         len(diff), len(m), 0.001, _ex(diff[["date", "ticker", "close",
                                                             "close_panel"]])))

    # 9. Freshness
    lag = (today - close.index.max()).days
    checks.append(_check("freshness", f"latest bar no older than {STALE_DAYS} calendar days",
                         int(lag > STALE_DAYS), 1, 0.0,
                         [{"latest_bar": str(close.index.max().date()), "days_old": lag}]))

    passed = sum(c["status"] == "pass" for c in checks)
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "panel": {"stocks": int(close.shape[1]), "days": int(close.shape[0]),
                      "from": str(close.index.min().date()), "to": str(close.index.max().date()),
                      "ohlc_rows": int(len(ohlc))},
            "passed": passed, "failed": len(checks) - passed, "checks": checks}


def main():
    out = run()
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    p = out["panel"]
    print(f"Data quality: {p['stocks']} stocks x {p['days']} days ({p['from']} to {p['to']}), "
          f"{p['ohlc_rows']:,} OHLC bars")
    for c in out["checks"]:
        print(f"  [{c['status'].upper():<4}] {c['check']:<17} {c['rows_failed']:>6,} of "
              f"{c['rows_tested']:>9,} ({c['fail_rate']:.3%}, limit {c['threshold']:.2%})")
    print(f"{out['passed']} passed, {out['failed']} failed -> {OUT_PATH}")


if __name__ == "__main__":
    main()
