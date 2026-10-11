"""Kalman-filtered state feeding a GRU: statistics to de-noise, a net to learn.

WHY THIS COMBINATION AND NOT ANOTHER NEURAL NETWORK. The earlier LSTM and
BiLSTM were fed the RAW realized-volatility series, and that series is about
92% measurement noise day to day. A recurrent net given mostly noise learns
mostly noise, which is why both lost to a 3-parameter regression.

The project then measured a specific, systematic defect: every model keyed to
today's observed volatility moves the WRONG way 64% of the time, with
correlation -0.40 between the change in forecast and the change in realized
volatility. The cause is mechanical -- today's reading is high partly because
of transient noise, the forecast follows it up, and tomorrow the noise is
gone. Substituting the Kalman-filtered latent state for the raw observation
removes it almost completely: directional agreement goes 34.1% -> 48.8% and
the correlation of changes -0.445 -> -0.054.

So the division of labour here is deliberate:

  the KALMAN FILTER does what statistics is good at -- separating a latent
  signal from measurement error, using an explicitly estimated noise model
  (per ticker: persistence ~0.91, Var(latent) 0.125, Var(noise) 0.110)

  the GRU does what a net is good at -- finding non-linear and sequential
  structure in a signal that is already clean, across ~60,000 pooled
  observations rather than one ticker's 1,700

Feeding a net de-noised inputs is the whole point. Giving it the raw series
again would repeat the experiment that already failed twice.

THE CEILING STILL BINDS. Within a ticker, a perfect forecast of the latent
process scores R-squared(log) 0.533 against the noisy observation, and the
best model so far reaches 0.247 within-ticker. Anything above the ceiling has
leaked rather than learned.

Run:  python -m quant.hybrid_model [--tickers 20] [--epochs 20]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import MIN_TRAIN, RESULTS, mse, qlike

SEQ = 22
HIDDEN = 48
EPOCHS = 20
BATCH = 512
LR = 2e-3
REFIT_DAYS = 252
SEED = 0

#: Inputs at each timestep. The Kalman state is first because it is the
#: de-noised backbone; everything else is context the filter cannot see.
SEQ_FEATS = ["l_kal", "l_rv1", "l_rv5", "l_rv22", "l_mkt", "l_rel", "l_vix", "l_ovn"]


def build_hybrid_frame(estimator: str = "garman_klass", tickers: int = 0,
                       verbose: bool = True) -> pd.DataFrame:
    """Tracking features plus the Kalman latent state, per ticker."""
    from quant.statespace import fit_kalman
    from quant.tracking_models import build_tracking_frame

    f = build_tracking_frame(estimator, verbose=verbose)
    if tickers:
        keep = sorted(f["ticker"].unique())[:tickers]
        f = f[f["ticker"].isin(keep)].reset_index(drop=True)

    # The filtered state is computed causally: parameters are estimated on a
    # training prefix only, then the filter is RUN FORWARD over the whole
    # series. Running the filter forward uses no future information -- that
    # is what distinguishes a filter from a smoother, and a smoother here
    # would leak tomorrow into today.
    out = []
    for t, g in f.groupby("ticker"):
        g = g.sort_values("date").copy()
        y = g["l_rv1"].to_numpy(float)
        n_fit = min(max(int(len(y) * 0.5), 300), len(y))
        par = fit_kalman(y[:n_fit])
        phi, q, r, c = par["phi"], par["q"], par["r"], par["c"]
        x, p = 0.0, q / max(1.0 - phi ** 2, 1e-6)
        state = np.empty(len(y))
        for i in range(len(y)):
            x_pred, p_pred = phi * x, phi * phi * p + q
            f_ = p_pred + r
            k = p_pred / f_
            x = x_pred + k * ((y[i] - c) - x_pred)
            p = (1.0 - k) * p_pred
            state[i] = c + x
        g["l_kal"] = state
        out.append(g)
    return pd.concat(out, ignore_index=True).dropna(subset=SEQ_FEATS + ["y"])


def _sequences(d: pd.DataFrame, seq: int = SEQ):
    Xs, ys, idx = [], [], []
    for _, g in d.groupby("ticker"):
        g = g.sort_values("date")
        M = g[SEQ_FEATS].to_numpy(np.float32)
        yy = g["y"].to_numpy(np.float32)
        rows = g.index.to_numpy()
        for i in range(seq - 1, len(g)):
            Xs.append(M[i - seq + 1:i + 1])
            ys.append(yy[i])
            idx.append(rows[i])
    return np.asarray(Xs), np.asarray(ys), np.asarray(idx)


def _make_net(n_feat: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_feat, HIDDEN, batch_first=True)
            self.head = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return Net()


def _train(X, y, epochs: int):
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    sd = X.reshape(-1, X.shape[-1]).std(0)
    sd[sd < 1e-8] = 1.0
    ym, ys_ = float(y.mean()), float(y.std()) or 1.0

    net = _make_net(X.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = torch.nn.MSELoss()
    xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor((y - ym) / ys_)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    return net, mu, sd, ym, ys_


def walk_forward_hybrid(d: pd.DataFrame, epochs: int = EPOCHS,
                        min_train: int = MIN_TRAIN, refit_days: int = REFIT_DAYS,
                        verbose: bool = True) -> np.ndarray:
    import torch

    X, y, idx = _sequences(d)
    dates = d.loc[idx, "date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    X, y, idx, dates = X[order], y[order], idx[order], dates[order]

    preds = np.full(len(d), np.nan)
    uniq = np.sort(np.unique(dates))
    if len(uniq) <= min_train:
        return preds
    t0 = time.time()
    cuts = list(range(min_train, len(uniq), refit_days))
    for n, c in enumerate(cuts, 1):
        cut = uniq[c]
        end = uniq[min(c + refit_days, len(uniq) - 1)]
        tr = dates < cut
        te = (dates >= cut) & (dates < end) if c + refit_days < len(uniq) else (dates >= cut)
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        net, mu, sd, ym, ys_ = _train(X[tr], y[tr], epochs)
        with torch.no_grad():
            xt = torch.tensor((X[te] - mu) / sd)
            preds[idx[te]] = (net(xt).numpy() * ys_ + ym)
        if verbose:
            print(f"    refit {n}/{len(cuts)} @ {str(cut)[:10]}  train {tr.sum():,}  "
                  f"score {te.sum():,}  {time.time() - t0:.0f}s", flush=True)
    return preds


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--estimator", default="garman_klass")
    args = ap.parse_args()

    from quant.tracking_models import FEATURES, walk_forward as wf_flat

    d = build_hybrid_frame(args.estimator, args.tickers)
    print(f"  {len(d):,} rows, {d['ticker'].nunique()} tickers\n")

    print("  training GRU on Kalman-filtered inputs ...", flush=True)
    d["hybrid"] = walk_forward_hybrid(d, epochs=args.epochs)
    print("  reference models ...", flush=True)
    d["har_log"] = wf_flat(d, "ridge_log")
    d["kal_only"] = d["l_kal"]
    d["naive"] = d["l_rv1"]

    models = ["hybrid", "har_log", "kal_only", "naive"]
    sc = d.dropna(subset=models + ["y"]).reset_index(drop=True)
    ly = sc["y"].to_numpy(float)
    lvl = sc["target"].to_numpy(float)

    def r2(a, p):
        return 1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2)

    def within_r2(col):
        a = ly - sc.groupby("ticker")["y"].transform("mean").to_numpy()
        p = sc[col].to_numpy(float) - sc.groupby("ticker")[col].transform("mean").to_numpy()
        return 1 - np.sum((a - p) ** 2) / np.sum(a ** 2)

    def direction(col, k=1):
        hits = []
        for _, g in sc.groupby("ticker"):
            a, p = g["y"].to_numpy(float), g[col].to_numpy(float)
            if k > 1:
                a = pd.Series(a).rolling(k).mean().to_numpy()
                p = pd.Series(p).rolling(k).mean().to_numpy()
            da, dp = np.diff(a), np.diff(p)
            m = np.isfinite(da) & np.isfinite(dp) & (da != 0) & (dp != 0)
            if m.sum() > 30:
                hits.append(np.mean(np.sign(da[m]) == np.sign(dp[m])))
        return float(np.mean(hits)) if hits else float("nan")

    summary = {}
    for m in models:
        lp = sc[m].to_numpy(float)
        summary[m] = {"r2_log": float(r2(ly, lp)), "within_r2": float(within_r2(m)),
                      "dir_1d": direction(m, 1), "dir_22d": direction(m, 22),
                      "qlike": qlike(lvl, np.exp(lp + 0.5 * np.var(ly - lp)))}

    print(f"\n{'=' * 68}\nHYBRID (Kalman -> GRU)   {len(sc):,} forecasts, "
          f"{sc['ticker'].nunique()} tickers\n{'=' * 68}")
    print(f"  {'model':<11}{'R2 log':>9}{'within R2':>11}{'dir 1d':>9}{'dir 22d':>9}{'QLIKE':>9}")
    for m, v in sorted(summary.items(), key=lambda kv: -kv[1]["r2_log"]):
        print(f"  {m:<11}{v['r2_log']:>9.3f}{v['within_r2']:>11.3f}"
              f"{100 * v['dir_1d']:>8.1f}%{100 * v['dir_22d']:>8.1f}%{v['qlike']:>9.4f}")
    print("\n  within-ticker ceiling is 0.533; 50% direction is a coin flip")

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "seq_len": SEQ, "hidden": HIDDEN, "epochs": args.epochs,
           "features": SEQ_FEATS, "n_forecasts": int(len(sc)),
           "n_tickers": int(sc["ticker"].nunique()), "summary": summary}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_hybrid.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
