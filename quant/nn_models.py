"""Pooled LSTM and BiLSTM for realized volatility.

THE QUESTION. Every non-linear challenger so far has lost to a 3-parameter
linear fit: gradient boosting (0.41764), a natural cubic spline that provably
nests HAR (0.41015), and a pretrained transformer (Chronos, 4th of 5). A
recurrent net is the obvious remaining architecture, and it is the one that
gets proposed most often, so it is worth settling with a number rather than
an argument.

THREE DESIGN CHOICES, EACH WITH A REASON.

Pooled, not per-ticker. One stock has ~1,700 usable days, which is nothing
for a recurrent net; pooled there are ~84,000. Note that pooling HURT the
linear model (pooled_har 0.41364 against har 0.40377), so if the net also
suffers, that is a sample-size finding rather than an architecture one.

Refit annually, not every 21 days. This is affordable rather than
prohibitive, and the project's own fitting sweep shows it is fair: refit
frequency moved HAR's loss by 0.00121 across 5 to 252 days, against 0.02826
for window length. Refitting yearly therefore costs essentially nothing in
comparability.

Fitted in logs. Volatility is right-skewed and positive, and levels differ
across tickers by more than a factor of three. Working in log space turns
those level differences into offsets and keeps the target roughly symmetric.
Exponentiating back uses the same smearing correction as log_har, since a
naive exp() of a log-space fit is biased low.

ON BIDIRECTIONALITY. A BiLSTM reads the window forwards and backwards. That
is NOT leakage here: the window ends the day BEFORE the target, so the
backward pass only re-reads days already observed. It would be leakage if the
net were run across the whole series including the forecast day, which is a
common error in published implementations and worth checking before citing
one.

Run:  python -m quant.nn_models [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant.volatility import MIN_TRAIN, RESULTS

SEQ_LEN = 22            # same history HAR's longest component sees
HIDDEN = 32
EPOCHS = 15
BATCH = 512
LR = 3e-3
REFIT_DAYS = 252        # annual; justified by the fitting sweep
SEED = 0


def _build_sequences(frame: pd.DataFrame, seq_len: int = SEQ_LEN):
    """Per-row log-RV history ending the day BEFORE the target.

    Returns (X, y, row_index). X[i] holds log volatility for days t-seq_len+1
    through t; y[i] is log volatility at t+1. Nothing from the target day or
    later enters its own input.
    """
    Xs, ys, idx = [], [], []
    for _, g in frame.groupby("ticker"):
        g = g.sort_values("date")
        rv = np.log(np.clip(g["naive"].to_numpy(float), 1e-8, None))
        tgt = np.log(np.clip(g["target"].to_numpy(float), 1e-8, None))
        rows = g.index.to_numpy()
        for i in range(seq_len - 1, len(g)):
            Xs.append(rv[i - seq_len + 1:i + 1])
            ys.append(tgt[i])
            idx.append(rows[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(idx)


def _make_net(bidirectional: bool):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(1, HIDDEN, batch_first=True,
                               bidirectional=bidirectional)
            self.head = nn.Linear(HIDDEN * (2 if bidirectional else 1), 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return Net()


def _train(X: np.ndarray, y: np.ndarray, bidirectional: bool, epochs: int):
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))
    mu, sd = float(X.mean()), float(X.std()) or 1.0
    net = _make_net(bidirectional)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    xt = torch.tensor((X - mu) / sd).unsqueeze(-1)
    yt = torch.tensor((y - mu) / sd)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[b]), yt[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        resid = (yt - net(xt)).numpy()
    return net, mu, sd, float(np.var(resid) * sd * sd)


def _predict(net, X: np.ndarray, mu: float, sd: float, resid_var: float) -> np.ndarray:
    import torch

    out = np.empty(len(X), dtype=float)
    with torch.no_grad():
        for s in range(0, len(X), 4096):
            chunk = torch.tensor((X[s:s + 4096] - mu) / sd).unsqueeze(-1)
            out[s:s + len(chunk)] = net(chunk).numpy() * sd + mu
    # smearing correction: exp() of a log-space fit is biased low
    return np.exp(out + 0.5 * resid_var)


def walk_forward_nn(frame: pd.DataFrame, bidirectional: bool = False,
                    epochs: int = EPOCHS, min_train: int = MIN_TRAIN,
                    refit_days: int = REFIT_DAYS, verbose: bool = True) -> np.ndarray:
    """Expanding-window pooled forecasts, refitted annually."""
    X, y, idx = _build_sequences(frame)
    dates = frame.loc[idx, "date"].to_numpy()
    order = np.argsort(dates, kind="stable")
    X, y, idx, dates = X[order], y[order], idx[order], dates[order]

    preds = np.full(len(frame), np.nan)
    uniq = np.unique(dates)
    if len(uniq) <= min_train:
        return preds

    t0 = time.time()
    cuts = list(range(min_train, len(uniq), refit_days))
    for n, c in enumerate(cuts, 1):
        cut_date = uniq[c]
        end_date = uniq[min(c + refit_days, len(uniq) - 1)]
        tr = dates < cut_date
        te = (dates >= cut_date) & (dates < end_date) if c + refit_days < len(uniq) \
            else (dates >= cut_date)
        if tr.sum() < 500 or te.sum() == 0:
            continue
        net, mu, sd, rv_ = _train(X[tr], y[tr], bidirectional, epochs)
        preds[idx[te]] = _predict(net, X[te], mu, sd, rv_)
        if verbose:
            print(f"    refit {n}/{len(cuts)} at {str(cut_date)[:10]}  "
                  f"train {tr.sum():,}  score {te.sum():,}  {time.time() - t0:.0f}s",
                  flush=True)
    return preds


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--estimator", default="parkinson")
    args = ap.parse_args()

    from quant.mcs import model_confidence_set
    from quant.vol_models import FEATURE_SETS, build_features, walk_forward
    from quant.volatility import mse, qlike, qlike_series

    frame = build_features(args.estimator, verbose=True)
    print(f"  {len(frame):,} rows, {frame['ticker'].nunique()} tickers")

    cols = {"naive": frame["naive"].to_numpy(float)}
    print("  fitting the HAR family ...", flush=True)
    for model in FEATURE_SETS:
        cols[model] = walk_forward(frame, model)

    for label, bi in (("lstm", False), ("bilstm", True)):
        print(f"  training pooled {label} ({args.epochs} epochs/refit) ...", flush=True)
        cols[label] = walk_forward_nn(frame, bidirectional=bi, epochs=args.epochs)

    scored = frame.copy()
    for k, v in cols.items():
        scored[k] = v
    models = list(cols)
    scored = scored.dropna(subset=models).reset_index(drop=True)
    actual = scored["target"].to_numpy(float)

    summary = {m: {"qlike": qlike(actual, scored[m].to_numpy(float)),
                   "mse": mse(actual, scored[m].to_numpy(float))} for m in models}
    losses = {m: qlike_series(actual, scored[m].to_numpy(float)) for m in models}
    mcs = model_confidence_set(losses, verbose=True)

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "estimator": args.estimator, "seq_len": SEQ_LEN, "hidden": HIDDEN,
           "epochs": args.epochs, "refit_days": REFIT_DAYS, "pooled": True,
           "n_forecasts": int(len(scored)), "n_models": len(models),
           "summary": summary, "mcs": mcs}

    print(f"\n{'=' * 60}\nWITH POOLED LSTM / BiLSTM   {len(scored):,} forecasts\n{'=' * 60}")
    print(f"  {'model':<13}{'QLIKE':>10}{'MSE':>11}")
    for m, v in sorted(summary.items(), key=lambda kv: kv[1]["qlike"]):
        star = "  <- MCS" if m in mcs["mcs"] else ""
        print(f"  {m:<13}{v['qlike']:>10.5f}{v['mse']:>11.6f}{star}")
    print(f"\n  MCS (95%): {mcs['mcs']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "volatility_nn.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
