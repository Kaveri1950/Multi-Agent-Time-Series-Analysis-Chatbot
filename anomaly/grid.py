"""Run every detector on every domain and emit the comparison table.

The table IS the result. Not any single detector -- the question this
project asks is which assumption about "anomalous" survives contact with two
unrelated kinds of data, and only a grid answers that.

Every domain loader returns the same thing, a `Dataset`, so `run_domain`
never learns which one it is holding. Adding a third domain (Fermi blazar
light curves are the obvious next one) means writing one loader and
nothing else.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly.audit import audit_dataset, format_audit
from anomaly.core import windowize
from anomaly.detectors import default_detectors
from anomaly.metrics import evaluate_scores, precision_at_k, summarise_against_floor
from anomaly.splits import make_split, rank01, score_held_out, select_ensemble

RESULTS_DIR = Path(__file__).parent.parent / "evaluation" / "results"

#: Pre-declared per domain. Fixed before any result was seen -- tuning these
#: per detector is how a reporting budget of 1 silently becomes 12.
WINDOW_CONFIG = {
    "kepler": {"window": 24, "stride": 6},    # 12h / 3h at 29.4-min cadence
    "equity": {"window": 21, "stride": 1},    # one trading month, daily step
}


@dataclass
class Dataset:
    """A scored unit of one domain: windows, labels, and provenance.

    `groups` decides how detectors are fitted. When it is None the dataset is
    one series and every detector gets a single pooled fit. When it is an
    array of series ids, each series is fitted and scored SEPARATELY and the
    per-series metrics are aggregated.

    That distinction is not cosmetic. Kepler targets differ in intrinsic
    variability by more than two orders of magnitude -- one star in the first
    twelve has flux std 0.0344 against another's 0.0002. Pooled, every
    detector ranks the noisy star's windows above the quiet star's
    regardless of where the transits are, so the grid measures which star is
    noisiest rather than which window holds a transit. Per-target fitting is
    also what `astronomy/evaluate.py` already does, so the numbers stay
    comparable with the existing result.
    """

    name: str
    windows: np.ndarray
    y: np.ndarray
    effect: np.ndarray | None = None   # per-window magnitude, for the label audit
    finance_only: bool = False
    groups: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def load_equity(top_n: int = 1) -> Dataset:
    """Nifty index-level returns, labelled by the pre-registered event calendar.

    Index-level rather than per-stock on purpose: the event calendar records
    MARKET-wide events, so a single stock that ignored the 2024 election
    result is not a detector failure, it is a label that does not apply to it.
    """
    from anomaly.labels import finance_day_labels, windows_from_day_labels
    from quant.data import load_panel

    close, volume = load_panel()
    # Equal-weighted index of the most liquid names -- a market proxy built
    # from the panel rather than a separately downloaded index, so the dates
    # line up exactly with the label calendar.
    tv = (close * volume).rolling(21, min_periods=10).mean()
    liquid = tv.mean().nlargest(max(top_n, 50)).index
    returns = close[liquid].pct_change().mean(axis=1).dropna()

    cfg = WINDOW_CONFIG["equity"]
    day_labels = finance_day_labels(returns.index)
    windows = windowize(returns.to_numpy(), cfg["window"], cfg["stride"])

    # TRAILING windows: each window's label is its LAST day, not "any day
    # inside it". Two reasons, and both matter.
    #
    # Honesty: a window labelled by any of its 21 days turns 42 event days
    # into 384 positive windows -- 20% prevalence, which the audit flags as
    # Wu & Keogh's unrealistic-density flaw. The task stops being anomaly
    # detection and becomes near-balanced classification.
    #
    # Causality: the detector sees day t and the 20 days before it, never
    # after. A centred window would let tomorrow's crash explain today's
    # score, which is look-ahead bias with extra steps.
    y = day_labels[cfg["window"] - 1:][: len(windows)]
    windows = windows[: len(y)]
    effect = np.abs(windows[:, -1])
    return Dataset(
        name="equity", windows=windows, y=y, effect=effect, finance_only=True,
        meta={"n_days": len(returns), "n_names": len(liquid), "labelling": "trailing",
              "first": str(returns.index[0].date()), "last": str(returns.index[-1].date()),
              **cfg},
    )


def load_kepler(limit: int | None = 60, seed: int = 0) -> Dataset:
    """Kepler light curves, labelled by NASA ephemerides.

    Windows from many targets are pooled into one matrix. That is the honest
    framing of the question actually being asked -- "is this window unusual
    among all windows we have seen" -- and it matches how equity is scored.
    Per-target fitting would give each detector a private normalisation the
    equity column never gets.
    """
    from anomaly.labels import astronomy_window_labels
    from astronomy.bulk_download import load_manifest, load_target
    from astronomy.koi_catalog import fetch_koi_table

    manifest = load_manifest()
    koi = fetch_koi_table()
    cfg = WINDOW_CONFIG["kepler"]

    rng = np.random.default_rng(seed)
    rows = manifest.sample(frac=1.0, random_state=seed) if limit else manifest
    if limit:
        rows = rows.head(limit)

    all_w, all_y, kepids = [], [], []
    for _, row in rows.iterrows():
        try:
            df = load_target(int(row["kepid"]))
        except Exception:
            continue
        times = df["time"].to_numpy()
        flux = df["flux"].to_numpy()
        ok = np.isfinite(times) & np.isfinite(flux)
        times, flux = times[ok], flux[ok]
        if len(flux) < cfg["window"] * 4:
            continue
        w = windowize(flux, cfg["window"], cfg["stride"])
        try:
            y = astronomy_window_labels(times, int(row["kepid"]),
                                        cfg["window"], cfg["stride"], koi)
        except Exception:
            continue
        n = min(len(w), len(y))
        all_w.append(w[:n])
        all_y.append(y[:n])
        kepids.extend([int(row["kepid"])] * n)

    if not all_w:
        raise RuntimeError("no Kepler targets loaded -- is the bulk cache present?")
    windows = np.vstack(all_w)
    y = np.concatenate(all_y)
    med = np.median(windows, axis=1)
    effect = np.maximum(med - windows.min(axis=1), windows.max(axis=1) - med)
    return Dataset(
        name="kepler", windows=windows, y=y, effect=effect,
        groups=np.asarray(kepids),
        meta={"n_targets": len(set(kepids)), "n_windows": len(windows),
              "fitting": "per-target", **cfg},
    )


def _score_dataset(det, ds: Dataset) -> np.ndarray:
    """Detector scores for every window, fitted per group where groups exist.

    Scores from different groups are NOT comparable as raw numbers -- one
    star's "very anomalous" is another's background -- so each group's scores
    are converted to within-group ranks in [0, 1] before being concatenated.
    Ranking metrics only care about order, and this makes the order
    meaningful across groups instead of dominated by whichever series has
    the largest units.
    """
    if ds.groups is None:
        return det.fit_score(ds.windows, ds.y)

    out = np.empty(len(ds.windows), dtype=float)
    for gid in np.unique(ds.groups):
        mask = ds.groups == gid
        y_g = ds.y[mask] if ds.y is not None else None
        scores = det.fit_score(ds.windows[mask], y_g)
        order = np.argsort(np.argsort(scores, kind="stable"), kind="stable")
        denom = max(len(scores) - 1, 1)
        out[mask] = order / denom
    return out


def run_domain(ds: Dataset, top_k: int = 50, verbose: bool = True,
               n_jobs: int = 1, seed: int = 0) -> list[dict]:
    """Every detector against one dataset, scored BOTH ways.

    in_sample   fitted and scored on the same windows (the old number)
    pr_auc      fitted on the training split, scored on the held-out one

    Both are kept because their difference measures how much a detector
    memorises, which turned out to be the thing that separates the honest
    results from the flattering ones.
    """
    from joblib import Parallel, delayed

    stride = int(ds.meta.get("stride", 1))
    train_mask, test_mask = make_split(ds, seed=seed)
    y_train, y_test = ds.y[train_mask], ds.y[test_mask]
    if verbose:
        print(f"  split: train {train_mask.sum():,} windows / {int(y_train.sum())} pos"
              f"   test {test_mask.sum():,} / {int(y_test.sum())} pos")

    detectors = default_detectors(include_finance_only=ds.finance_only, stride=stride)

    def evaluate_one(det):
        start = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                in_sample = evaluate_scores(ds.y, _score_dataset(det, ds))
                held = score_held_out(det, ds, train_mask, test_mask)
                train_scores = rank01(score_held_out(det, ds, train_mask, train_mask))
            metrics = evaluate_scores(y_test, held)
            return {
                "domain": ds.name, "detector": det.name, "group": det.group,
                "is_floor": det.is_floor, "is_ceiling": det.is_ceiling,
                **metrics,
                "in_sample_pr_auc": in_sample.get("pr_auc"),
                "in_sample_lift": in_sample.get("lift"),
                f"precision_at_{top_k}": precision_at_k(y_test, held, top_k),
                "seconds": round(time.time() - start, 2),
            }, (det.name, rank01(held), train_scores)
        except Exception as exc:  # noqa: BLE001
            return ({"domain": ds.name, "detector": det.name, "group": det.group,
                     "is_floor": det.is_floor, "is_ceiling": det.is_ceiling,
                     "error": f"{type(exc).__name__}: {exc}",
                     "roc_auc": float("nan"), "pr_auc": float("nan"),
                     "seconds": round(time.time() - start, 2)}, None)

    # Detectors are independent, so this parallelises cleanly. Workers are
    # pinned to one BLAS thread each: sklearn and numpy already thread
    # internally, and oversubscribing 12 cores runs slower than serial.
    if n_jobs > 1:
        import os as _os
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            _os.environ.setdefault(var, "1")
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(evaluate_one)(d) for d in detectors)
    else:
        results = [evaluate_one(d) for d in detectors]

    rows, test_scores, train_scores = [], {}, {}
    for row, payload in results:
        rows.append(row)
        if payload is not None and not row.get("is_ceiling"):
            name, test_s, train_s = payload
            test_scores[name], train_scores[name] = test_s, train_s
        if verbose:
            if "error" in row:
                print(f"  {row['detector']:26} FAILED  {row['error'][:50]}")
            else:
                print(f"  {row['detector']:26} {row['group']:15} "
                      f"held-out pr={row['pr_auc']:.4f} lift={row.get('lift', float('nan')):5.2f}x"
                      f"   in-sample pr={row['in_sample_pr_auc']:.4f}"
                      f"   {row['seconds']:6.1f}s")

    # ENSEMBLE: members chosen on the TRAINING half only, then scored on test.
    members = select_ensemble(train_scores, y_train)
    if len(members) > 1:
        combined = np.mean([test_scores[m] for m in members], axis=0)
        metrics = evaluate_scores(y_test, combined)
        rows.append({
            "domain": ds.name, "detector": f"ensemble_{len(members)}",
            "group": "ensemble", "is_floor": False, "is_ceiling": False,
            "members": members, **metrics,
            f"precision_at_{top_k}": precision_at_k(y_test, combined, top_k),
            "seconds": 0.0,
        })
        if verbose:
            print(f"  {'ensemble':26} {'ensemble':15} "
                  f"held-out pr={metrics['pr_auc']:.4f} lift={metrics.get('lift', float('nan')):5.2f}x"
                  f"   <- {' + '.join(members)}")
    return rows


def main(kepler_limit: int | None = 60, save: bool = True, n_jobs: int = 1) -> dict:
    loaders = [("equity", load_equity), ("kepler", lambda: load_kepler(limit=kepler_limit))]
    all_rows, audits, summaries, metas = [], {}, {}, {}

    for name, loader in loaders:
        print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
        try:
            ds = loader()
        except Exception as exc:  # noqa: BLE001
            print(f"  could not load: {type(exc).__name__}: {exc}")
            continue
        print(f"  {ds.windows.shape[0]} windows x {ds.windows.shape[1]} points, "
              f"{int(ds.y.sum())} positive ({ds.y.mean():.2%})")
        print(f"  meta: {ds.meta}\n")

        rows = run_domain(ds, n_jobs=n_jobs)
        all_rows.extend(rows)
        metas[name] = ds.meta

        audit = audit_dataset(ds.y, ds.effect, rows)
        audits[name] = audit
        print(f"\n  DATASET AUDIT (Wu & Keogh four flaws)")
        print(format_audit(audit))

        summaries[name] = summarise_against_floor(rows)
        print(f"\n  {json.dumps(summaries[name], indent=2)}")

    out = {"rows": all_rows, "audits": audits, "summaries": summaries, "meta": metas}
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "anomaly_grid.json").write_text(json.dumps(out, indent=2, default=str))
        pd.DataFrame(all_rows).to_csv(RESULTS_DIR / "anomaly_grid.csv", index=False)
        print(f"\nwrote {RESULTS_DIR / 'anomaly_grid.json'}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kepler-limit", type=int, default=60,
                    help="targets to pool; None-like 0 means all 440")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    main(kepler_limit=args.kepler_limit or None, save=not args.no_save, n_jobs=args.jobs)
