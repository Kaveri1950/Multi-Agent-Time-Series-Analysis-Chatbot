"""Risk of a whole portfolio, not one stock at a time.

Everything else in the project scores a single name. A wealth adviser holds
a portfolio, and its risk is not the average of its parts: two holdings that
move together add risk, two that offset each other remove it. So this module
answers the questions a portfolio review asks:

  how much does it move           annualised volatility, from the covariance
  how bad is a bad day            historical VaR and expected shortfall of the
                                  portfolio's own daily returns
  where does the risk come from   each holding's contribution, by Euler
                                  allocation, which sums exactly to the total
  how diversified is it           diversification ratio, effective number of
                                  holdings
  how does it ride the market     beta against the equal-weighted market proxy
  what if history repeats         the portfolio replayed through real Indian
                                  market shocks from the pre-registered event
                                  calendar (anomaly/labels.py)
  can the VaR be trusted          a walk-forward Kupiec and Christoffersen
                                  backtest on this portfolio, by regime

Weights are held fixed through history. That is the standard way to ask "how
risky is this mix", and it is not the same as how a real account that drifts
and rebalances would have done.

Run:  python -m quant.portfolio_risk
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR = 252
LOOKBACK_DAYS = 756          # three years for the covariance and the VaR

#: Historical shocks replayed on the portfolio, from the pre-registered
#: calendar. Windows run from the last close before the shock to its worst
#: point, so a multi-day selloff is measured as one move.
SCENARIOS = (
    ("COVID crash (Feb–Mar 2020)", "2020-02-19", "2020-03-23"),
    ("Russia invades Ukraine (24 Feb 2022)", "2022-02-23", "2022-02-24"),
    ("Adani–Hindenburg (Jan–Feb 2023)", "2023-01-24", "2023-02-01"),
    ("Election-result day (4 Jun 2024)", "2024-06-03", "2024-06-04"),
    ("Global tariff selloff (Apr 2025)", "2025-04-02", "2025-04-07"),
)

#: Illustrative model portfolios, in the spirit of the model portfolios
#: wealth platforms offer. Not recommendations.
MODEL_PORTFOLIOS = {
    "Conservative": {"HINDUNILVR.NS": 0.2, "ITC.NS": 0.2, "TCS.NS": 0.2,
                     "HDFCBANK.NS": 0.2, "SUNPHARMA.NS": 0.2},
    "Balanced": {"RELIANCE.NS": 0.15, "TCS.NS": 0.15, "HDFCBANK.NS": 0.15,
                 "ICICIBANK.NS": 0.15, "LT.NS": 0.1, "BHARTIARTL.NS": 0.1,
                 "ITC.NS": 0.1, "MARUTI.NS": 0.1},
    "Aggressive": {"ADANIENT.NS": 0.2, "TATASTEEL.NS": 0.2, "BAJFINANCE.NS": 0.2,
                   "IRFC.NS": 0.2, "BHEL.NS": 0.2},
}


def _returns(tickers: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Daily returns of the holdings and of the market proxy."""
    from finance.risk_profile import market_panel
    from quant.data import load_panel

    close, volume = load_panel()
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise ValueError(f"not in the price panel: {', '.join(missing)}")
    rets = close[tickers].pct_change(fill_method=None)
    market = market_panel(close, volume).mean(axis=1)
    return rets, market


def normalise(holdings: dict[str, float]) -> dict[str, float]:
    total = float(sum(holdings.values()))
    if total <= 0:
        raise ValueError("weights must be positive")
    return {t.upper() if "." in t else t.upper() + ".NS": w / total
            for t, w in holdings.items()}


def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Euler allocation: w_i (Cov w)_i / sigma_p. Sums exactly to sigma_p."""
    sigma = float(np.sqrt(w @ cov @ w))
    return w * (cov @ w) / sigma


def stress_tests(rets: pd.DataFrame, w: pd.Series) -> list[dict]:
    """The portfolio replayed through each historical shock.

    A holding not yet listed at the time is left out and the rest are
    re-weighted, and the row says how much of the portfolio was covered, so
    a scenario is never silently computed on a fraction of the holdings.
    """
    out = []
    growth = (1 + rets.fillna(0)).cumprod()
    for name, start, end in SCENARIOS:
        try:
            s = growth.index[growth.index.get_indexer([pd.Timestamp(start)], method="ffill")[0]]
            e = growth.index[growth.index.get_indexer([pd.Timestamp(end)], method="ffill")[0]]
        except (KeyError, IndexError):
            continue
        listed = rets.loc[:s].notna().sum() > 20
        cover = float(w[listed].sum())
        if cover < 0.5:
            out.append({"scenario": name, "covered_weight": round(cover, 3), "portfolio_pct": None})
            continue
        ww = w[listed] / w[listed].sum()
        move = (growth.loc[e, ww.index] / growth.loc[s, ww.index] - 1)
        worst = move.idxmin()
        out.append({"scenario": name, "from": str(s.date()), "to": str(e.date()),
                    "covered_weight": round(cover, 3),
                    "portfolio_pct": round(float((ww * move).sum()) * 100, 2),
                    "worst_holding": worst.replace(".NS", ""),
                    "worst_holding_pct": round(float(move[worst]) * 100, 2)})
    return out


def backtest_var(port: pd.Series, level: float = 0.95) -> dict:
    """Walk-forward VaR on this portfolio, with and without the market regime."""
    from finance.risk_profile import _fit_market_regime, market_panel
    from quant.data import load_panel
    from quant.risk import backtest

    obs, _, _, regime, _ = _fit_market_regime(market_panel(*load_panel()))
    p = port.reindex(obs.index).dropna()
    out = {}
    for label, reg in (("unconditional", None), ("regime", regime)):
        b = backtest(p, reg, level=level)
        k, c = b["kupiec"], b["christoffersen"]
        kp, cp = k.get("p_value"), c.get("p_value")
        ok_k = kp is not None and np.isfinite(kp) and kp > 0.05
        ok_c = cp is not None and np.isfinite(cp) and cp > 0.05
        out[label] = {"n_days": k["n"], "breach_rate": k["rate"], "kupiec_p": kp,
                      "christoffersen_p": cp,
                      "status": "passed" if ok_k and ok_c else
                                ("partial" if ok_k and not np.isfinite(cp or np.nan) else "failed")}
    out["regime_now"] = "turbulent" if bool(regime.iloc[-1]) else "calm"
    return out


def analyse(holdings: dict[str, float], with_backtest: bool = True) -> dict:
    """The full portfolio review."""
    from quant.risk import var_es

    weights = normalise(holdings)
    tickers = list(weights)
    rets, market = _returns(tickers)
    w = pd.Series(weights)
    recent = rets.iloc[-LOOKBACK_DAYS:]
    usable = recent.dropna(axis=1, thresh=int(0.8 * len(recent))).columns.tolist()
    if len(usable) < len(tickers):
        dropped = sorted(set(tickers) - set(usable))
        raise ValueError(f"too little history in the last three years for: {', '.join(dropped)}")

    r = recent.fillna(0.0)
    cov = r.cov().to_numpy() * YEAR
    wv = w[tickers].to_numpy()
    sigma = float(np.sqrt(wv @ cov @ wv))
    rc = risk_contributions(cov, wv)
    vols = np.sqrt(np.diag(cov))
    port = (r * wv).sum(axis=1)
    full_port = (rets.fillna(0.0) * wv).sum(axis=1).loc[rets.notna().any(axis=1)]
    var95, es95 = var_es(port.to_numpy(), 0.95)
    var99, es99 = var_es(port.to_numpy(), 0.99)
    m = market.reindex(port.index)
    beta = float(np.cov(port, m.fillna(0))[0, 1] / np.var(m.fillna(0)))
    corr = r.corr().to_numpy()
    avg_corr = float((corr.sum() - len(tickers)) / max(len(tickers) * (len(tickers) - 1), 1))

    holdings_rows = [{"ticker": t.replace(".NS", ""), "weight_pct": round(float(wv[i]) * 100, 1),
                      "volatility_pct": round(float(vols[i]) * 100, 1),
                      "risk_contribution_pct": round(float(rc[i] / sigma) * 100, 1)}
                     for i, t in enumerate(tickers)]
    holdings_rows.sort(key=lambda x: -x["risk_contribution_pct"])
    out = {
        "holdings": holdings_rows,
        "window": {"from": str(port.index[0].date()), "to": str(port.index[-1].date()),
                   "days": int(len(port))},
        "volatility_pct": round(sigma * 100, 2),
        "weighted_average_volatility_pct": round(float(wv @ vols) * 100, 2),
        "diversification_ratio": round(float(wv @ vols) / sigma, 2),
        "effective_holdings": round(float(1 / np.sum(wv ** 2)), 1),
        "average_correlation": round(avg_corr, 2),
        "beta": round(beta, 2),
        "var_95_pct": round(var95 * 100, 2), "es_95_pct": round(es95 * 100, 2),
        "var_99_pct": round(var99 * 100, 2), "es_99_pct": round(es99 * 100, 2),
        "stress": stress_tests(rets, w),
    }
    if with_backtest:
        out["backtest"] = backtest_var(full_port)
    out["summary"] = summary(out)
    return out


def summary(a: dict) -> str:
    """Plain-English review, built from the numbers and nothing else."""
    top = a["holdings"][0]
    worst = min((s for s in a["stress"] if s.get("portfolio_pct") is not None),
                key=lambda s: s["portfolio_pct"], default=None)
    parts = [
        f"This portfolio moves about {a['volatility_pct']:.1f}% a year. Its holdings average "
        f"{a['weighted_average_volatility_pct']:.1f}% on their own, so holding them together "
        f"removes some risk (diversification ratio {a['diversification_ratio']:.2f}).",
        f"On the worst one day in twenty it loses at least {abs(a['var_95_pct']):.2f}%, and "
        f"{abs(a['es_95_pct']):.2f}% on average on those days.",
        f"{top['ticker']} is {top['weight_pct']:.0f}% of the money but "
        f"{top['risk_contribution_pct']:.0f}% of the risk.",
        f"It moves about {a['beta']:.2f} times as much as the market.",
    ]
    if worst:
        parts.append(f"Replayed through history, its worst scenario was the {worst['scenario']}: "
                     f"{worst['portfolio_pct']:+.1f}%.")
    bt = a.get("backtest")
    if bt:
        rg = bt["regime"]
        verdict = {"passed": "passes both standard backtests",
                   "partial": "passes the count test; the clustering test could not be computed",
                   "failed": "fails at least one standard backtest, so treat it as indicative"}
        parts.append(f"Its VaR, adjusted for the market's current {bt['regime_now']} mood, "
                     f"{verdict[rg['status']]} (breached on {rg['breach_rate']:.1%} of "
                     f"{rg['n_days']} days against a 5% target).")
    parts.append("This describes risk, not a recommendation to buy or sell.")
    return " ".join(parts)


def facts_for_llm(a: dict) -> str:
    """Fact block for an AI-written client note, checked afterwards by the guard."""
    lines = ["PORTFOLIO FACTS (every number you write must come from here):",
             f"- annual volatility {a['volatility_pct']}%; holdings' average "
             f"{a['weighted_average_volatility_pct']}%",
             f"- 95% one-day VaR {a['var_95_pct']}% (the loss exceeded on the worst 1 day in 20, "
             f"AT LEAST this much); expected shortfall {a['es_95_pct']}%",
             f"- beta {a['beta']}; diversification ratio {a['diversification_ratio']}"]
    lines += [f"- {h['ticker']}: weight {h['weight_pct']}%, share of risk "
              f"{h['risk_contribution_pct']}%" for h in a["holdings"]]
    lines += [f"- scenario {s['scenario']}: {s['portfolio_pct']}%" for s in a["stress"]
              if s.get("portfolio_pct") is not None]
    return "\n".join(lines)


def ai_client_note(a: dict, backend: str | None = None) -> dict:
    """An LLM-drafted note for a client, checked number by number.

    The draft is a first draft: every figure in it is compared with the fact
    block, and anything not supported is flagged before it reaches anyone.
    """
    from agents.answer_style import guard_answer
    from agents.llm_backend import call_llm

    facts = facts_for_llm(a)
    prompt = (f"{facts}\n\nWrite a short, plain-English note (under 120 words) for a retail "
              f"client about this portfolio's risk. No advice to buy or sell.")
    draft = call_llm(prompt, backend=backend)
    checked, invented = guard_answer(draft, facts)
    return {"note": checked, "flagged_numbers": invented}


if __name__ == "__main__":
    import json
    import warnings

    warnings.filterwarnings("ignore")
    for name, h in MODEL_PORTFOLIOS.items():
        a = analyse(h)
        print(f"\n=== {name}\n{a['summary']}")
        print(json.dumps({k: a[k] for k in ("volatility_pct", "var_95_pct", "beta",
                                             "diversification_ratio")}))
        for s in a["stress"]:
            print("  ", s)
