"""The evidence the composed risk report quotes, measured rather than recalled.

agents/risk_composition.py attaches a validation status to every number it
shows. Some of that can be computed per query; the rest is measured here and
written to evaluation/results/risk_validation.json, so the report reads its
evidence from a file that a command reproduces.

FOUR QUESTIONS, EACH AN ATTEMPT TO BREAK A NUMBER THE REPORT LEANS ON

  volatility   Is the served interval calibrated out of sample? Its width
               comes from IN-SAMPLE residuals of a fit that includes random
               forests, which is the textbook way to get a band too narrow.
               And is direction measured on the rule that is served?

  VaR          The registry's Kupiec and Christoffersen p-values were measured
               on the equal-weighted market proxy. Does the pass survive on
               individual stocks, which is what the report prints?

  regime       Does the "lift on event days" in the registry reproduce
               against the pre-registered calendar in anomaly/labels.py?

  anomalies    The registry records no evaluation at all for the five typed
               detectors. Do they flag pre-registered event days more often
               than ordinary ones?

Run:  python -m evaluation.risk_validation        (about six minutes)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "risk_validation.json"
PREDICTIONS = RESULTS / "volatility_tracking_predictions.parquet"


def _r2(y, yh) -> float:
    return float(1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))


def volatility_evidence() -> dict:
    """Coverage, direction and cross-sectional skill on the held-out forecasts."""
    import joblib
    from scipy.stats import spearmanr

    from quant.production_model import MODEL_PATH
    from quant.tracking_models import build_tracking_frame

    p = pd.read_parquet(PREDICTIONS)
    f = build_tracking_frame("garman_klass", verbose=False, require_target=True)
    d = p.merge(f[["date", "ticker", "y", "l_rv1", "l_rv22", "l_rv66"]],
                on=["date", "ticker"], how="inner")

    # The parquet stores exp(log forecast + v/2), v being the variance of the
    # held-out log residual. A constant shift leaves a variance unchanged, so
    # v comes straight back, and with it the residual itself.
    raw = np.log(d["target"]) - np.log(d["ens_log"])
    v_oos = float(np.var(raw))
    d["resid"] = raw + 0.5 * v_oos
    d["lp"] = d["y"] - d["resid"]
    smear_var = float(joblib.load(MODEL_PATH)["smear_var"])
    sd = float(np.sqrt(smear_var))            # the sd the SERVED interval uses

    interval = {}
    for level, z in (("80", 1.2816), ("95", 1.96)):
        inside = d["resid"].abs() <= z * sd
        per = inside.groupby(d["ticker"]).mean()
        interval[level] = {
            "coverage": float(inside.mean()),
            "below": float((d["resid"] < -z * sd).mean()),
            "above": float((d["resid"] > z * sd).mean()),
            "per_ticker_min": float(per.min()), "per_ticker_max": float(per.max()),
            "per_ticker": {t: round(float(c), 4) for t, c in per.items()},
            "by_year": {int(y): round(float(c), 4)
                        for y, c in inside.groupby(d["date"].dt.year).mean().items()},
        }

    up = d["y"] > d["l_rv1"]
    served_up = d["lp"] + 0.5 * smear_var > d["l_rv1"]
    direction = {
        "log_rule": float(((d["lp"] > d["l_rv1"]) == up).mean()),
        "served_rule": float((served_up == up).mean()),
        "share_called_higher_log_rule": float((d["lp"] > d["l_rv1"]).mean()),
        "share_called_higher_served_rule": float(served_up.mean()),
        "share_actually_higher": float(up.mean()),
    }

    dm = d.groupby("ticker")[["y", "lp"]].transform("mean")
    tm = d.groupby("ticker")[["y", "lp"]].mean()
    r2 = {"pooled": _r2(d["y"].to_numpy(), d["lp"].to_numpy()),
          "within": _r2((d["y"] - dm["y"]).to_numpy(), (d["lp"] - dm["lp"]).to_numpy()),
          "between_ticker_means": _r2(tm["y"].to_numpy(), tm["lp"].to_numpy())}

    # What a market-wide ranking relies on: on a given day, does the forecast
    # order the stocks the way the realised volatility then orders them?
    def rank_corr(col):
        return d.groupby("date").apply(
            lambda g: spearmanr(g[col], g["y"])[0] if len(g) > 10 else np.nan).dropna()

    def top5_in_top10(col):
        return float(d.groupby("date").apply(
            lambda g: len(set(g.nlargest(5, col)["ticker"])
                          & set(g.nlargest(10, "y")["ticker"])) / 5
            if len(g) >= 40 else np.nan).dropna().mean())

    rk, rk_naive = rank_corr("lp"), rank_corr("l_rv1")
    # The comparison that could embarrass the model. Most of what separates
    # one stock's volatility from another's is a persistent level, and a
    # trailing average already knows that level without any model at all.
    rk_22, rk_66 = rank_corr("l_rv22"), rank_corr("l_rv66")
    gap = (rk - rk_66).dropna()
    cross = {"spearman_mean": float(rk.mean()), "spearman_p10": float(rk.quantile(0.10)),
             "spearman_naive_mean": float(rk_naive.mean()), "n_days": int(len(rk)),
             "spearman_trailing22_mean": float(rk_22.mean()),
             "spearman_trailing66_mean": float(rk_66.mean()),
             "gain_over_trailing66": float(gap.mean()),
             "gain_over_trailing66_t": float(gap.mean() / (gap.std() / np.sqrt(len(gap)))),
             "share_of_days_better_than_trailing66": float((gap > 0).mean()),
             "top5_forecast_in_realised_top10": top5_in_top10("lp"),
             "top5_trailing66_in_realised_top10": top5_in_top10("l_rv66"),
             "top5_chance_level": float(10 / d["ticker"].nunique())}

    return {"n_forecasts": int(len(d)), "n_tickers": int(d["ticker"].nunique()),
            "first_date": str(d["date"].min().date()), "last_date": str(d["date"].max().date()),
            "log_sd_served": sd, "log_sd_out_of_sample": float(np.sqrt(v_oos)),
            "interval": interval, "direction": direction, "r2": r2, "cross_section": cross}


def _proxy_backtests(returns: pd.Series, regime: pd.Series) -> dict:
    from quant.risk import DEFAULT_LEVELS, backtest

    out = {}
    for level in DEFAULT_LEVELS:
        for label, reg in (("unconditional", None), ("conditional", regime)):
            b = backtest(returns, reg, level=level)
            out[f"{label}_{int(level * 100)}"] = {
                "n_days": b["kupiec"]["n"], "breach_rate": b["kupiec"]["rate"],
                "kupiec_p": b["kupiec"]["p_value"],
                "christoffersen_p": b["christoffersen"]["p_value"],
                "joint_p": b.get("conditional_coverage", {}).get("p_value")}
    return out


def var_evidence(tickers: list[str]) -> dict:
    """The market-proxy backtest the registry quotes, then one per stock."""
    from agents.risk_composition import stock_var_backtest
    from finance.risk_profile import _fit_market_regime, market_panel
    from quant.data import load_panel
    from quant.regimes import build_observations, fit_regimes, turbulent_flag

    close, volume = load_panel()

    # (a) exactly what `python -m quant.risk` runs, the registry's source:
    #     top 50 names, its own regime fit.
    tv = (close * volume).rolling(21, min_periods=10).mean()
    panel50 = close[tv.mean().nlargest(50).index].pct_change()
    ret50 = panel50.mean(axis=1).dropna()
    obs = build_observations(ret50, panel_returns=panel50)
    model, order = fit_regimes(obs)
    registry_proxy = _proxy_backtests(ret50.reindex(obs.index),
                                      turbulent_flag(model, order, obs, use_filtered=True))

    # (b) the regime the system SERVES, which is fitted on the top 100.
    panel100 = market_panel(close, volume)
    obs100, _, _, regime100, _ = _fit_market_regime(panel100)
    served_proxy = _proxy_backtests(panel100.mean(axis=1).reindex(obs100.index), regime100)

    per = {t: stock_var_backtest(t) for t in tickers}
    tested = {t: r for t, r in per.items() if r.get("testable")}

    def count(which):
        k = [r[which]["kupiec_p"] for r in tested.values()]
        c = [r[which]["christoffersen_p"] for r in tested.values()]
        both = sum(1 for a, b in zip(k, c) if a is not None and b is not None
                   and a > 0.05 and b > 0.05)
        too_many = sum(1 for r in tested.values()
                       if r[which]["kupiec_p"] is not None and r[which]["kupiec_p"] <= 0.05
                       and r[which]["breach_rate"] > 0.05)
        too_few = sum(1 for r in tested.values()
                      if r[which]["kupiec_p"] is not None and r[which]["kupiec_p"] <= 0.05
                      and r[which]["breach_rate"] < 0.05)
        return {"kupiec_pass": sum(1 for a in k if a is not None and a > 0.05),
                "christoffersen_pass": sum(1 for b in c if b is not None and b > 0.05),
                "christoffersen_not_computable": sum(1 for b in c if b is None),
                "both_pass": both,
                "kupiec_fail_understates_loss": too_many,
                "kupiec_fail_overstates_loss": too_few,
                "median_breach_rate": float(np.median(
                    [r[which]["breach_rate"] for r in tested.values()]))}

    return {"panel_last_date": str(close.index[-1].date()),
            "market_proxy_registry_top50": registry_proxy,
            "market_proxy_served_top100": served_proxy,
            "per_ticker": per,
            "per_ticker_summary": {
                "n_tickers": len(per), "n_testable": len(tested),
                "untestable": sorted(set(per) - set(tested)),
                "conditional": count("conditional"),
                "unconditional": count("unconditional")}}


def _fisher(flag: np.ndarray, event: np.ndarray) -> dict:
    """Is `flag` more common on event days? One-sided Fisher exact."""
    from scipy.stats import fisher_exact

    flag, event = np.asarray(flag, bool), np.asarray(event, bool)
    a, b = int((flag & event).sum()), int((~flag & event).sum())
    c, d = int((flag & ~event).sum()), int((~flag & ~event).sum())
    pe, po = a / max(a + b, 1), c / max(c + d, 1)
    return {"rate_event": pe, "rate_ordinary": po,
            "lift": pe / po if po else float("nan"),
            "fisher_p": float(fisher_exact([[a, b], [c, d]], alternative="greater")[1]),
            "n_event": a + b, "n_ordinary": c + d}


def regime_evidence() -> dict:
    from anomaly.labels import event_dates
    from finance.risk_profile import _fit_market_regime, market_panel
    from quant.data import load_panel

    _, _, _, regime, _ = _fit_market_regime(market_panel(*load_panel()))
    event = regime.index.isin(event_dates(regime.index))
    r = _fisher(regime.to_numpy(), event)
    # Four of the fifteen registered events are March 2020, and a regime
    # persists, so those days are close to one observation counted many
    # times. Dropping the year is the cheapest way to see what is left.
    keep = regime.index.year != 2020
    ex = _fisher(regime.to_numpy()[keep], event[keep])
    return {"p_turbulent_event": r["rate_event"], "p_turbulent_ordinary": r["rate_ordinary"],
            "lift": r["lift"], "fisher_p": r["fisher_p"],
            "excluding_2020": {"lift": ex["lift"], "fisher_p": ex["fisher_p"],
                               "p_turbulent_event": ex["rate_event"],
                               "n_event_days": ex["n_event"]},
            "n_event_days": r["n_event"], "n_days": int(len(regime)),
            "turbulent_share": float(regime.mean()),
            "first_date": str(regime.index[0].date()),
            "last_date": str(regime.index[-1].date()),
            "note": "regime fitted on the top-100 panel, states filtered, parameters "
                    "fitted on the full sample"}


def regime_stability(n_seeds: int = 10) -> dict:
    """Do the regime's results survive a different random start?

    The HMM is fitted by EM from a random initialisation, and everything
    downstream -- the lift, the VaR backtest -- inherits whichever optimum it
    lands in. Refreshing the panel by nine days moved the proxy's Kupiec
    p-value from 0.273 to 0.831, which is either the data or the fit, and the
    two have different consequences. So: same data, ten seeds.
    """
    from anomaly.labels import event_dates
    from finance.risk_profile import market_panel
    from quant.data import load_panel
    from quant.regimes import build_observations, fit_regimes, turbulent_flag
    from quant.risk import backtest

    panel = market_panel(*load_panel())
    market = panel.mean(axis=1).dropna()
    obs = build_observations(market, panel_returns=panel)
    ret = market.reindex(obs.index)
    event = obs.index.isin(event_dates(obs.index))
    runs, base = [], None
    for seed in range(n_seeds):
        model, order = fit_regimes(obs, seed=seed)
        reg = turbulent_flag(model, order, obs, use_filtered=True)
        base = reg if base is None else base
        b = backtest(ret, reg, level=0.95)
        runs.append({"seed": seed, "turbulent_share": float(reg.mean()),
                     "agreement_with_seed_0": float((reg == base).mean()),
                     "lift": float(reg[event].mean() / reg[~event].mean()),
                     "breach_rate": b["kupiec"]["rate"],
                     "kupiec_p": b["kupiec"]["p_value"],
                     "christoffersen_p": b["christoffersen"]["p_value"]})

    def span(key):
        return [min(r[key] for r in runs), max(r[key] for r in runs)]

    return {"n_seeds": n_seeds, "runs": runs,
            "min_agreement_with_seed_0": min(r["agreement_with_seed_0"] for r in runs),
            "lift_range": span("lift"), "kupiec_p_range": span("kupiec_p"),
            "christoffersen_p_range": span("christoffersen_p"),
            "seeds_passing_both": sum(1 for r in runs if r["kupiec_p"] > 0.05
                                      and r["christoffersen_p"] > 0.05)}


def anomaly_evidence(tickers: list[str]) -> dict:
    """Do the five typed detectors flag pre-registered event days more often?

    Pooled over ticker-days. The events are market-wide and the detectors run
    per stock, so a stock that shrugged off an event counts against the
    detector; the lift is a floor, not a ceiling.
    """
    from agents.risk_composition import price_frame
    from anomaly.labels import event_dates
    from finance.anomaly_detector import _FLAG_COLUMNS, detect_all

    frames = []
    for t in tickers:
        s = detect_all(price_frame(t))
        s["event"] = s.index.isin(event_dates(s.index))
        frames.append(s[["is_anomaly", "event", *_FLAG_COLUMNS]])
    from scipy.stats import mannwhitneyu

    allf = pd.concat(frames)
    top = _fisher(allf["is_anomaly"].to_numpy(), allf["event"].to_numpy())

    # The p-value is NOT taken from ticker-days. Fifty stocks on one event
    # date are one observation of the market seen fifty times, and treating
    # them as independent produced p = 5e-191. The unit is the day: the share
    # of stocks flagged on each date, event days against ordinary ones.
    daily = allf.groupby(level=0).agg(share=("is_anomaly", "mean"), event=("event", "first"))

    def day_test(frame):
        e, o = frame.loc[frame["event"], "share"], frame.loc[~frame["event"], "share"]
        return {"n_event_days": int(len(e)), "median_share_event": float(e.median()),
                "median_share_ordinary": float(o.median()),
                "p": float(mannwhitneyu(e, o, alternative="greater")[1])}

    day = day_test(daily)
    ex = day_test(daily[daily.index.year != 2020])
    ex_t = allf[allf.index.year != 2020]
    ex["lift"] = _fisher(ex_t["is_anomaly"].to_numpy(), ex_t["event"].to_numpy())["lift"]

    by_type = {}
    for col in _FLAG_COLUMNS:
        r = _fisher(allf[col].fillna(False).to_numpy(), allf["event"].to_numpy())
        by_type[col] = {k: r[k] for k in ("rate_event", "rate_ordinary", "lift")}
    return {"flag_rate_event": top["rate_event"], "flag_rate_ordinary": top["rate_ordinary"],
            "lift": top["lift"], "day_level": day, "excluding_2020": ex,
            "n_event_ticker_days": top["n_event"], "n_ordinary_ticker_days": top["n_ordinary"],
            "n_tickers": len(tickers), "by_type": by_type,
            "note": "sentiment_divergence never fires here: no headlines are passed"}


def main():
    import argparse

    warnings.filterwarnings("ignore")
    from quant.tracking_models import build_tracking_frame

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated sections to re-measure; "
                                               "the rest are kept from the existing file")
    only = {s for s in ap.parse_args().only.split(",") if s}

    tickers = sorted(build_tracking_frame("garman_klass", verbose=False,
                                          require_target=False)["ticker"].unique())
    out = json.loads(OUT_PATH.read_text()) if only and OUT_PATH.exists() else {}
    out["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, fn in (("volatility", volatility_evidence),
                     ("var", lambda: var_evidence(tickers)),
                     ("regime", regime_evidence),
                     ("regime_stability", regime_stability),
                     ("anomaly", lambda: anomaly_evidence(tickers))):
        if only and name not in only:
            continue
        print(f"  measuring {name} ...", flush=True)
        out[name] = fn()

    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    v, var, rg, an = out["volatility"], out["var"], out["regime"], out["anomaly"]
    print(f"\n{'=' * 70}\nVOLATILITY   {v['n_forecasts']:,} held-out forecasts, "
          f"{v['n_tickers']} tickers\n{'=' * 70}")
    for lvl in ("80", "95"):
        c = v["interval"][lvl]
        print(f"  {lvl}% interval   coverage {c['coverage']:.4f}   below {c['below']:.4f}  "
              f"above {c['above']:.4f}   per-ticker {c['per_ticker_min']:.3f}"
              f"-{c['per_ticker_max']:.3f}")
    dr = v["direction"]
    print(f"  direction      log rule {dr['log_rule']:.4f}   served rule "
          f"{dr['served_rule']:.4f}   (served calls 'higher' "
          f"{dr['share_called_higher_served_rule']:.1%} of the time; truth "
          f"{dr['share_actually_higher']:.1%})")
    print(f"  R2(log)        pooled {v['r2']['pooled']:.4f}   within {v['r2']['within']:.4f}")
    cs = v["cross_section"]
    print(f"  cross-section  daily Spearman {cs['spearman_mean']:.3f} against "
          f"{cs['spearman_trailing66_mean']:.3f} for a trailing 66-day average and "
          f"{cs['spearman_naive_mean']:.3f} for today's value; top-5 in realised top-10 "
          f"{cs['top5_forecast_in_realised_top10']:.1%} against "
          f"{cs['top5_trailing66_in_realised_top10']:.1%} (chance "
          f"{cs['top5_chance_level']:.0%})")

    print(f"\n{'=' * 70}\nVaR, 95%, walk-forward\n{'=' * 70}")
    for label, key in (("market proxy (registry)", "market_proxy_registry_top50"),
                       ("market proxy (served)", "market_proxy_served_top100")):
        for kind in ("unconditional", "conditional"):
            r = var[key][f"{kind}_95"]
            print(f"  {label:<24}{kind:<15} breach {r['breach_rate']:.2%}  "
                  f"Kupiec p={r['kupiec_p']}  Christoffersen p={r['christoffersen_p']}")
    s = var["per_ticker_summary"]
    for kind in ("unconditional", "conditional"):
        c = s[kind]
        print(f"  {'individual stocks':<24}{kind:<15} both tests pass on {c['both_pass']} of "
              f"{s['n_testable']}  (Kupiec {c['kupiec_pass']}, Christoffersen "
              f"{c['christoffersen_pass']}; rejected for understating loss "
              f"{c['kupiec_fail_understates_loss']}, overstating "
              f"{c['kupiec_fail_overstates_loss']})")
    print(f"  not testable: {', '.join(s['untestable']) or 'none'}")

    print(f"\n{'=' * 70}\nREGIME and ANOMALIES against the pre-registered calendar\n{'=' * 70}")
    print(f"  regime     turbulent on {rg['p_turbulent_event']:.1%} of {rg['n_event_days']} "
          f"event days vs {rg['p_turbulent_ordinary']:.1%} ordinary   lift {rg['lift']:.2f}   "
          f"p={rg['fisher_p']:.2g}")
    x = rg["excluding_2020"]
    print(f"             excluding 2020: {x['p_turbulent_event']:.1%} of {x['n_event_days']} "
          f"event days   lift {x['lift']:.2f}   p={x['fisher_p']:.2g}")
    rs = out.get("regime_stability")
    if rs:
        print(f"  stability  across {rs['n_seeds']} random starts the regime labels agree on at "
              f"least {rs['min_agreement_with_seed_0']:.1%} of days; lift "
              f"{rs['lift_range'][0]:.2f}-{rs['lift_range'][1]:.2f}; proxy VaR passes both "
              f"tests on {rs['seeds_passing_both']} of {rs['n_seeds']} (Kupiec p "
              f"{rs['kupiec_p_range'][0]:.2f}-{rs['kupiec_p_range'][1]:.2f}, Christoffersen "
              f"{rs['christoffersen_p_range'][0]:.2f}-{rs['christoffersen_p_range'][1]:.2f})")
    dl, x = an["day_level"], an["excluding_2020"]
    print(f"  detectors  flag {an['flag_rate_event']:.1%} of event ticker-days vs "
          f"{an['flag_rate_ordinary']:.1%} ordinary   lift {an['lift']:.2f}   "
          f"day-level p={dl['p']:.2g} over {dl['n_event_days']} event days")
    print(f"             excluding 2020: lift {x['lift']:.2f}   day-level p={x['p']:.2g} "
          f"over {x['n_event_days']} event days")
    for col, r in an["by_type"].items():
        print(f"    {col:<22} {r['rate_event']:.1%} vs {r['rate_ordinary']:.1%}   "
              f"lift {r['lift']:.2f}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
