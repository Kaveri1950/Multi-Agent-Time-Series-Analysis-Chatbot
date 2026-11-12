"""Daily sunspot number: forecast 1 to 27 days ahead, scored against the record.

DATA. The official series is SILSO's daily total sunspot number (Royal
Observatory of Belgium, sidc.be/SILSO), 1818 to about two weeks ago, with no
missing day since 1900. The most recent ~4 months are flagged provisional and
may still be revised by a few units. Days after the file ends come from
SILSO's EISN (estimated international sunspot number), a quick-look value
published daily; it is used to start the live forecast and never scored.

MODELS, chosen from the literature on daily sunspot forecasting:
  persistence     tomorrow = today. The bar every model has to clear.
  recurrence27    the value one solar rotation (27 days) before the target day.
                  Active regions come back round as the Sun turns.
  mean27          the average of the last 27 days, one full rotation.
  arima           ARIMA on sqrt(SN), the classical statistical model
                  (SARIMA/ARIMA appear as the baseline in most papers).
  xgboost         gradient-boosted trees on lags, rolling means and the
                  rotation echoes, one model per horizon (direct). The
                  literature uses LightGBM for this; XGBoost is the same
                  method family and is what the project already ships.
  lstm            a two-layer LSTM reading the last 54 days and emitting all
                  27 horizons at once. LSTMs scored best at most horizons up to
                  14 days in a 2025 comparison with TCN and Transformer models
                  on Cycle 25.
  ensemble        the plain mean of xgboost and lstm.

All models work on sqrt(SN): the day-to-day spread grows with the level, and
the square root roughly evens it out. Predictions are squared back and
clipped at 0, which gives a median-type forecast (right for MAE).

EVALUATION. Walk-forward. Test origins run from 2010-01-01 (Cycle 24) to the
end of the SILSO file, in two-year blocks. Each block's models are trained
only on origins whose TARGET day falls before the block starts, so nothing in
a block is seen during its training. Scored by MAE and RMSE in sunspot units,
skill relative to persistence, and a Diebold-Mariano test of the best model
against the best baseline at each horizon.

Run:  python -m solar.sunspots --backtest     (about 10-15 minutes, CPU)
      python -m solar.sunspots --forecast     (live 27-day forecast, ~2 minutes)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from solar.flare_data import CACHE_DIR

SILSO_DAILY_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
SILSO_EISN_URL = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"

SUNSPOT_DIR = CACHE_DIR / "sunspots"
DAILY_PATH = SUNSPOT_DIR / "sn_daily.parquet"
EISN_PATH = SUNSPOT_DIR / "eisn_current.parquet"
PRED_PATH = SUNSPOT_DIR / "backtest_predictions.parquet"
LIVE_PATH = SUNSPOT_DIR / "live_forecast.json"
LOG_PATH = SUNSPOT_DIR / "forecast_log.parquet"
RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "results" / "sunspot_forecast.json"

START = "1900-01-01"            # complete daily record from here on
TEST_START = "2010-01-01"
REFIT_YEARS = 2
HORIZONS = (1, 7, 14, 27)
MAX_H = 27
ROTATION = 27
LSTM_WINDOW = 54                # two rotations
MAX_AGE_HOURS = 12
SEED = 42

MODELS = ("persistence", "recurrence27", "mean27", "arima", "xgboost", "lstm", "ensemble")
BASELINES = ("persistence", "recurrence27", "mean27")


# ---------------------------------------------------------------- data

def _stale(path: Path) -> bool:
    return (not path.exists()
            or time.time() - path.stat().st_mtime > MAX_AGE_HOURS * 3600)


def fetch_daily(force: bool = False) -> pd.DataFrame:
    """SILSO daily total sunspot number, cached for 12 hours."""
    SUNSPOT_DIR.mkdir(parents=True, exist_ok=True)
    if not force and not _stale(DAILY_PATH):
        return pd.read_parquet(DAILY_PATH)
    resp = requests.get(SILSO_DAILY_URL, timeout=60)
    resp.raise_for_status()
    d = pd.read_csv(io.StringIO(resp.text), sep=";", header=None,
                    names=["year", "month", "day", "decimal_year", "sn", "sd",
                           "n_obs", "definitive"])
    d["date"] = pd.to_datetime(d[["year", "month", "day"]])
    d["sn"] = d["sn"].where(d["sn"] >= 0)          # -1 marks a missing day
    d = d[["date", "sn", "sd", "n_obs", "definitive"]]
    d.to_parquet(DAILY_PATH)
    return d


def fetch_eisn(force: bool = False) -> pd.DataFrame:
    """SILSO's provisional daily estimate for the current month."""
    SUNSPOT_DIR.mkdir(parents=True, exist_ok=True)
    if not force and not _stale(EISN_PATH):
        return pd.read_parquet(EISN_PATH)
    resp = requests.get(SILSO_EISN_URL, timeout=60)
    resp.raise_for_status()
    rows = [r.split(",") for r in resp.text.strip().splitlines() if r.strip()]
    d = pd.DataFrame([[int(r[0]), int(r[1]), int(r[2]), float(r[4])] for r in rows],
                     columns=["year", "month", "day", "sn"])
    d["date"] = pd.to_datetime(d[["year", "month", "day"]])
    d = d[["date", "sn"]]
    d.to_parquet(EISN_PATH)
    return d


def load_series(with_eisn: bool = True) -> pd.DataFrame:
    """One row per day from 1900: date, sn, source (definitive/provisional/eisn)."""
    d = fetch_daily()
    d = d[d["date"] >= START].copy()
    d["source"] = np.where(d["definitive"] == 1, "definitive", "provisional")
    d = d[["date", "sn", "source"]]
    if with_eisn:
        try:
            e = fetch_eisn()
            e = e[e["date"] > d["date"].max()].assign(source="eisn")
            d = pd.concat([d, e], ignore_index=True)
        except Exception:                       # EISN is a bonus, never a blocker
            pass
    d = d.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    full = pd.DataFrame({"date": pd.date_range(d["date"].min(), d["date"].max())})
    d = full.merge(d, on="date", how="left")
    if d["sn"].isna().any():
        d["sn"] = d["sn"].interpolate(limit_direction="both")
        d["source"] = d["source"].fillna("interpolated")
    return d


# ---------------------------------------------------------------- features

def _features(z: pd.Series, h: int) -> pd.DataFrame:
    """Everything known at the origin day t, for a target h days later."""
    f = {f"lag{k}": z.shift(k) for k in range(0, ROTATION + 1)}
    for w in (7, 27, 81, 365):
        f[f"mean{w}"] = z.rolling(w).mean()
    f["std27"] = z.rolling(27).std()
    f["echo27"] = z.shift(ROTATION - h)          # one rotation before the target day
    f["echo54"] = z.shift(2 * ROTATION - h)      # two rotations before
    f["trend81"] = z.rolling(27).mean() - z.rolling(81).mean()
    return pd.DataFrame(f)


def _baseline(name: str, sn: np.ndarray, origins: np.ndarray, h: int) -> np.ndarray:
    if name == "persistence":
        return sn[origins]
    if name == "recurrence27":
        return sn[origins + h - ROTATION]
    if name == "mean27":
        c = np.concatenate([[0.0], np.cumsum(sn)])
        return (c[origins + 1] - c[origins + 1 - ROTATION]) / ROTATION
    raise ValueError(name)


# ---------------------------------------------------------------- models

def _fit_xgb(X: pd.DataFrame, y: pd.Series):
    from xgboost import XGBRegressor

    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=-1)
    m.fit(X, y)
    return m


class _LSTMNet:
    """Two-layer LSTM -> all 27 horizons in one output layer."""

    def __init__(self):
        import torch
        from torch import nn

        torch.manual_seed(SEED)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.1)
                self.head = nn.Linear(64, MAX_H)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        self.net = Net()
        self.mu, self.sd = 0.0, 1.0

    @staticmethod
    def windows(z: np.ndarray, origins: np.ndarray) -> np.ndarray:
        idx = origins[:, None] + np.arange(-LSTM_WINDOW + 1, 1)[None, :]
        return z[idx]

    def fit(self, z: np.ndarray, origins: np.ndarray, max_epochs: int = 25):
        import torch

        self.mu, self.sd = float(z[: origins.max() + 1].mean()), float(z[: origins.max() + 1].std())
        zs = (z - self.mu) / self.sd
        X = self.windows(zs, origins)[:, :, None]
        Y = zs[origins[:, None] + np.arange(1, MAX_H + 1)[None, :]]
        n_val = max(1, len(X) // 10)             # last 10% in time, for early stopping
        Xt, Yt = torch.tensor(X[:-n_val], dtype=torch.float32), torch.tensor(Y[:-n_val], dtype=torch.float32)
        Xv, Yv = torch.tensor(X[-n_val:], dtype=torch.float32), torch.tensor(Y[-n_val:], dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        gen = torch.Generator().manual_seed(SEED)
        best, best_state, bad = np.inf, None, 0
        for _ in range(max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[b]), Yt[b])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(loss_fn(self.net(Xv), Yv))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, z: np.ndarray, origins: np.ndarray) -> np.ndarray:
        """(n_origins, 27) predictions in sqrt space."""
        import torch

        zs = (z - self.mu) / self.sd
        X = torch.tensor(self.windows(zs, origins)[:, :, None], dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            out = self.net(X).numpy()
        return out * self.sd + self.mu


def _select_arima_order(z: np.ndarray) -> tuple:
    """Smallest AIC on the last 20 years before the test period, from a short list."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best, best_order = np.inf, (2, 1, 1)
    for order in ((1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1), (5, 1, 1)):
        try:
            res = SARIMAX(z, order=order).fit(disp=False)
            if res.aic < best:
                best, best_order = res.aic, order
        except Exception:
            continue
    return best_order


def _arima_paths(z: np.ndarray, train_end: int, origins: np.ndarray, order: tuple) -> np.ndarray:
    """(n_origins, 27) sqrt-space forecasts with parameters fixed at train_end.

    The parameters come from the 20 years before train_end only. They are then
    run through a Kalman filter over the test block, and each forecast starts
    from data up to its own origin (dynamic prediction), so no later
    observation reaches it.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit_lo = max(0, train_end - 20 * 365)
    res = SARIMAX(z[fit_lo:train_end], order=order).fit(disp=False)
    lo = max(0, origins.min() - 3 * 365)
    hi = min(len(z), origins.max() + 1)
    applied = res.apply(z[lo:hi])
    out = np.empty((len(origins), MAX_H))
    for j, t in enumerate(origins):
        s = t - lo + 1                       # first predicted day; data up to t is used
        pred = applied.get_prediction(start=s, end=s + MAX_H - 1, dynamic=True)
        out[j] = np.asarray(pred.predicted_mean)[:MAX_H]
    return out


def _to_sn(zhat: np.ndarray) -> np.ndarray:
    return np.clip(zhat, 0, None) ** 2


# ---------------------------------------------------------------- backtest

def backtest(verbose: bool = True) -> dict:
    from quant.volatility import diebold_mariano

    t_start = time.time()
    d = load_series(with_eisn=False)
    dates = d["date"].to_numpy()
    sn = d["sn"].to_numpy(float)
    z = np.sqrt(sn)
    zs = pd.Series(z)
    n = len(d)
    first_ok = max(365, 2 * ROTATION, LSTM_WINDOW)     # enough history for every feature

    feats = {h: _features(zs, h) for h in HORIZONS}
    test_start = int(np.searchsorted(dates, np.datetime64(TEST_START)))
    order = _select_arima_order(z[test_start - 20 * 365:test_start])
    if verbose:
        print(f"{n} days {pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}; "
              f"ARIMA order {order}")

    block_starts = []
    y0 = pd.Timestamp(TEST_START).year
    while True:
        i = int(np.searchsorted(dates, np.datetime64(f"{y0}-01-01")))
        if i >= n - 1:
            break
        block_starts.append(i)
        y0 += REFIT_YEARS
    block_starts.append(n)

    rows = []
    for b in range(len(block_starts) - 1):
        bs, be = block_starts[b], block_starts[b + 1]
        origins = np.arange(bs, min(be, n - 1))
        if len(origins) == 0:
            continue
        tb = time.time()
        # training origins: every target day strictly before the block
        train_lstm = np.arange(first_ok, bs - MAX_H)
        lstm = _LSTMNet().fit(z, train_lstm)
        lstm_all = lstm.predict(z, origins)
        arima_all = _arima_paths(z, bs, origins, order)

        for h in HORIZONS:
            ok = origins[origins + h < n]
            tr = np.arange(first_ok, bs - h)
            X, y = feats[h].iloc[tr], zs.shift(-h).iloc[tr]
            xgb = _fit_xgb(X, y)
            preds = {
                "xgboost": _to_sn(xgb.predict(feats[h].iloc[ok])),
                "lstm": _to_sn(lstm_all[: len(ok), h - 1]),
                "arima": _to_sn(arima_all[: len(ok), h - 1]),
            }
            preds["ensemble"] = 0.5 * (preds["xgboost"] + preds["lstm"])
            for name in BASELINES:
                preds[name] = _baseline(name, sn, ok, h)
            frame = pd.DataFrame({"origin": dates[ok], "target": dates[ok + h], "h": h,
                                  "actual": sn[ok + h], "block": b, **preds})
            rows.append(frame)
        if verbose:
            print(f"  block {b + 1}: {pd.Timestamp(dates[bs]).date()} -> "
                  f"{pd.Timestamp(dates[origins[-1]]).date()}  ({time.time() - tb:.0f}s)")

    pred = pd.concat(rows, ignore_index=True)
    SUNSPOT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_PATH)

    def _score(p: pd.DataFrame) -> dict:
        out = {}
        for m in MODELS:
            err = p[m] - p["actual"]
            out[m] = {"mae": round(float(err.abs().mean()), 3),
                      "rmse": round(float(np.sqrt((err ** 2).mean())), 3)}
        base = out["persistence"]["mae"]
        for m in MODELS:
            out[m]["mae_skill_vs_persistence"] = round(1 - out[m]["mae"] / base, 4)
        return out

    by_h, cycles = {}, {}
    for h in HORIZONS:
        p = pred[pred["h"] == h]
        s = _score(p)
        best_model = min((m for m in MODELS if m not in BASELINES), key=lambda m: s[m]["mae"])
        best_base = min(BASELINES, key=lambda m: s[m]["mae"])
        dm = diebold_mariano(np.abs(p[best_model] - p["actual"]).to_numpy(),
                             np.abs(p[best_base] - p["actual"]).to_numpy(), h=h)
        by_h[str(h)] = {"n": int(len(p)), "scores": s, "best_model": best_model,
                        "best_baseline": best_base,
                        "dm_best_model_vs_best_baseline": {k: (round(v, 6) if isinstance(v, float) else v)
                                                           for k, v in dm.items()}}
        cyc = {}
        for label, lo, hi in (("cycle24_2010_2019", "2010-01-01", "2020-01-01"),
                              ("cycle25_2020_on", "2020-01-01", "2100-01-01")):
            q = p[(p["origin"] >= lo) & (p["origin"] < hi)]
            cyc[label] = {"n": int(len(q)),
                          **{m: round(float((q[m] - q["actual"]).abs().mean()), 3) for m in MODELS}}
        cycles[str(h)] = cyc

    d_src = d.set_index("date")["source"]
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"source": "SILSO daily total sunspot number (sidc.be/SILSO)",
                 "span": f"{pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}",
                 "n_days": int(n),
                 "provisional_from": str(d_src[d_src == "provisional"].index.min().date())
                 if (d_src == "provisional").any() else None},
        "design": {"test_start": TEST_START, "refit_years": REFIT_YEARS,
                   "n_blocks": len(block_starts) - 1, "horizons": list(HORIZONS),
                   "arima_order": list(order), "lstm_window": LSTM_WINDOW,
                   "transform": "sqrt", "note": "walk-forward; each block trained only on "
                   "origins whose target precedes the block"},
        "by_horizon": by_h,
        "mae_by_cycle": cycles,
        "runtime_s": round(time.time() - t_start, 1),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    if verbose:
        print_summary(out)
        print(f"wrote {RESULTS_PATH}")
    return out


def print_summary(res: dict) -> None:
    print(f"\nMAE (sunspots), walk-forward {res['design']['test_start']} on")
    print(f"  {'model':<14}" + "".join(f"{'h=' + h:>10}" for h in res["by_horizon"]))
    for m in MODELS:
        print(f"  {m:<14}" + "".join(f"{res['by_horizon'][h]['scores'][m]['mae']:>10.2f}"
                                      for h in res["by_horizon"]))
    for h, v in res["by_horizon"].items():
        dm = v["dm_best_model_vs_best_baseline"]
        print(f"  h={h}: best {v['best_model']} vs {v['best_baseline']}, "
              f"skill {v['scores'][v['best_model']]['mae_skill_vs_persistence']:+.1%} vs persistence, "
              f"DM p={dm.get('p_value')}")


# ---------------------------------------------------------------- live

def live_forecast(verbose: bool = True) -> dict:
    """Fit on everything available and forecast the next 27 days.

    Uses the definitive + provisional file and EISN for the latest days. Every
    run is also appended to a forecast log, so the forecasts can later be
    checked against what SILSO actually records -- a prospective test, which
    the backtest cannot be.
    """
    d = load_series(with_eisn=True)
    dates = d["date"].to_numpy()
    sn = d["sn"].to_numpy(float)
    z = np.sqrt(sn)
    zs = pd.Series(z)
    n = len(d)
    t = n - 1
    first_ok = max(365, 2 * ROTATION, LSTM_WINDOW)
    origin = pd.Timestamp(dates[t])
    targets = pd.date_range(origin + pd.Timedelta(days=1), periods=MAX_H)

    lstm = _LSTMNet().fit(z, np.arange(first_ok, n - MAX_H))
    lstm_path = _to_sn(lstm.predict(z, np.array([t]))[0])

    order = tuple(json.loads(RESULTS_PATH.read_text())["design"]["arima_order"]) \
        if RESULTS_PATH.exists() else (2, 1, 1)
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    res = SARIMAX(z[max(0, n - 20 * 365):], order=order).fit(disp=False)
    arima_path = _to_sn(np.asarray(res.forecast(MAX_H)))

    xgb_path = np.empty(MAX_H)
    for h in range(1, MAX_H + 1):
        f = _features(zs, h)
        tr = np.arange(first_ok, n - h)
        m = _fit_xgb(f.iloc[tr], zs.shift(-h).iloc[tr])
        xgb_path[h - 1] = _to_sn(m.predict(f.iloc[[t]]))[0]

    paths = {
        "persistence": np.full(MAX_H, sn[t]),
        "recurrence27": np.array([sn[t + h - ROTATION] for h in range(1, MAX_H + 1)]),
        "mean27": np.full(MAX_H, sn[t - ROTATION + 1:t + 1].mean()),
        "arima": arima_path, "xgboost": xgb_path, "lstm": lstm_path,
        "ensemble": 0.5 * (xgb_path + lstm_path),
    }
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "origin": str(origin.date()), "origin_value": float(sn[t]),
        "origin_source": d["source"].iloc[t],
        "targets": [str(x.date()) for x in targets],
        "paths": {k: [round(float(v), 2) for v in p] for k, p in paths.items()},
    }
    SUNSPOT_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_PATH.write_text(json.dumps(out, indent=1))

    log = pd.DataFrame([{"issued": origin, "target": tg, "h": h + 1, "model": k, "pred": float(p[h])}
                        for k, p in paths.items() for h, tg in enumerate(targets)])
    if LOG_PATH.exists():
        log = pd.concat([pd.read_parquet(LOG_PATH), log], ignore_index=True)
    log = log.drop_duplicates(["issued", "target", "model"], keep="last")
    log.to_parquet(LOG_PATH)
    if verbose:
        print(f"origin {out['origin']} ({out['origin_source']}) SN {sn[t]:.0f}")
        for k in ("persistence", "arima", "xgboost", "lstm", "ensemble"):
            p = out["paths"][k]
            print(f"  {k:<12} +1d {p[0]:6.1f}   +7d {p[6]:6.1f}   +14d {p[13]:6.1f}   +27d {p[26]:6.1f}")
    return out


def scored_log() -> pd.DataFrame:
    """Logged live forecasts whose target day now has a recorded value."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    log = pd.read_parquet(LOG_PATH)
    d = load_series(with_eisn=True)[["date", "sn", "source"]].rename(columns={"date": "target"})
    m = log.merge(d, on="target", how="inner")
    m["abs_err"] = (m["pred"] - m["sn"]).abs()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--forecast", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        backtest()
    if a.forecast or not a.backtest:
        live_forecast()
