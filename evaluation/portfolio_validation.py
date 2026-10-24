"""Can the portfolio VaR be trusted? Backtested on many random portfolios.

Three model portfolios are three anecdotes. This draws 40 random portfolios
of 8 liquid NSE stocks with random weights, and backtests each one's 95% VaR
walk-forward, with and without the market regime, using the same Kupiec and
Christoffersen tests as everything else. It also checks that the risk
contributions add up to the portfolio's volatility, which is what makes them
an honest attribution rather than a heuristic.

Run:  python -m evaluation.portfolio_validation      (a few minutes)
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
OUT_PATH = RESULTS / "portfolio_validation.json"
N_PORTFOLIOS, N_HOLDINGS, SEED = 40, 8, 0


def main():
    warnings.filterwarnings("ignore")
    from quant.data import load_panel
    from quant.portfolio_risk import MODEL_PORTFOLIOS, analyse, risk_contributions

    close, volume = load_panel()
    tv = (close * volume).rolling(21, min_periods=10).mean().mean()
    full = close.columns[close.notna().mean() > 0.98]
    universe = list(tv[full].nlargest(100).index)
    rng = np.random.default_rng(SEED)

    rows = []
    for i in range(N_PORTFOLIOS):
        picks = list(rng.choice(universe, N_HOLDINGS, replace=False))
        weights = rng.dirichlet(np.ones(N_HOLDINGS))
        a = analyse(dict(zip(picks, weights)))
        rows.append({"portfolio": i, "holdings": [p.replace(".NS", "") for p in picks],
                     "volatility_pct": a["volatility_pct"], "var_95_pct": a["var_95_pct"],
                     "diversification_ratio": a["diversification_ratio"],
                     "regime": a["backtest"]["regime"],
                     "unconditional": a["backtest"]["unconditional"]})
        print(f"  {i + 1:>2}/{N_PORTFOLIOS}  regime {a['backtest']['regime']['status']:<8} "
              f"unconditional {a['backtest']['unconditional']['status']}", flush=True)

    # Euler contributions must sum to the total: check on random covariance matrices
    errs = []
    for _ in range(200):
        x = rng.normal(size=(300, 6))
        cov, w = np.cov(x, rowvar=False), rng.dirichlet(np.ones(6))
        errs.append(abs(risk_contributions(cov, w).sum() - np.sqrt(w @ cov @ w)))

    models = {name: {k: v for k, v in analyse(h).items() if k != "holdings"}
              for name, h in MODEL_PORTFOLIOS.items()}

    def tally(key):
        st = [r[key]["status"] for r in rows]
        return {s: st.count(s) for s in ("passed", "partial", "failed")}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_portfolios": N_PORTFOLIOS, "n_holdings": N_HOLDINGS, "universe": "100 most "
           "liquid NSE names with full history", "seed": SEED,
           "regime_var": tally("regime"), "unconditional_var": tally("unconditional"),
           "median_breach_regime": float(np.median([r["regime"]["breach_rate"] for r in rows])),
           "median_breach_unconditional": float(np.median(
               [r["unconditional"]["breach_rate"] for r in rows])),
           "median_diversification_ratio": float(np.median(
               [r["diversification_ratio"] for r in rows])),
           "euler_max_abs_error": float(max(errs)),
           "model_portfolios": models, "portfolios": rows}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{N_PORTFOLIOS} random {N_HOLDINGS}-stock portfolios, 95% VaR walk-forward:")
    print(f"  regime-conditional: {out['regime_var']}  (median breach "
          f"{out['median_breach_regime']:.2%})")
    print(f"  unconditional:      {out['unconditional_var']}  (median breach "
          f"{out['median_breach_unconditional']:.2%})")
    print(f"  median diversification ratio {out['median_diversification_ratio']:.2f}; "
          f"Euler contributions sum to total within {out['euler_max_abs_error']:.1e}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
