"""Charts for the audit exports.

Reads the CSVs that audit/warehouse.py writes to exports/audit/ and draws
them as PNGs under docs/figures/: one overview dashboard plus one chart per
question. Nothing here computes a new number -- every bar and point is a row
of an exported query, so the pictures can never disagree with the SQL.

Run with `python -m audit.charts` (audit.warehouse also calls it at the end).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
EXPORTS = ROOT / "exports" / "audit"
FIGURES = ROOT / "docs" / "figures"

# Same blue and black palette as dashboard/charts.py.
BLUE, NAVY, LIGHT, GREY, PALE = "#0078FF", "#0B2545", "#5AB7FF", "#9AA5B1", "#E1EFFF"

STRATEGIES = {"filter_only": ("Rule-based filter", BLUE),
              "hybrid": ("Filter + LLM", LIGHT),
              "llm_only": ("LLM only", GREY)}
CATEGORIES = ["adversarial", "paraphrase", "canonical", "general"]


def load(folder: Path = EXPORTS) -> dict:
    names = ["control_summary", "routing_accuracy", "routing_errors",
             "var_backtest_outcomes", "var_backtest_detail"]
    return {n: pd.read_csv(folder / f"{n}.csv") for n in names}


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def controls(ax, cs: pd.DataFrame):
    """Failure rate of each control against its limit; failed controls in navy.

    Bars are capped at 1% so that `freshness` (a single 0/1 check, so 100%
    when it fails) does not flatten every other bar to nothing. The label
    always carries the real value.
    """
    cs = cs.copy()
    cs["ratio"] = [r.fail_pct / r.limit_pct if r.limit_pct > 0 else (2.0 if r.fail_pct > 0 else 0.0)
                   for r in cs.itertuples()]
    cs = cs.sort_values("ratio")
    ax.barh(cs.check_name, cs.fail_pct.clip(upper=1.0),
            color=[NAVY if s == "fail" else LIGHT for s in cs.status])
    for y, (fp, lp) in enumerate(zip(cs.fail_pct, cs.limit_pct)):
        ax.text(min(fp, 1.0) + 0.01, y, f"{fp:g}% (limit {lp:g}%)", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("failure rate %")
    ax.set_title("Data-quality controls: failure rate vs limit (navy = failed)", fontsize=10.5,
                 color=NAVY, loc="left")
    _clean(ax)


def routing(ax, ra: pd.DataFrame):
    x, w = np.arange(len(CATEGORIES)), 0.27
    for k, (strategy, (label, color)) in enumerate(STRATEGIES.items()):
        r = ra[ra.strategy == strategy].set_index("category").reindex(CATEGORIES)
        ax.bar(x + (k - 1) * w, r.accuracy_pct, w, color=color, label=label)
    n = int(ra[(ra.strategy == "filter_only") & (ra.category == "ALL")].questions.iloc[0])
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 112)
    ax.set_ylabel("correct model chosen %")
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper left")
    ax.set_title(f"Model selection accuracy by question type ({n} labelled questions)",
                 fontsize=10.5, color=NAVY, loc="left")
    _clean(ax)


def var_scatter(ax, vd: pd.DataFrame):
    """One point per portfolio and method; a failed backtest gets a navy outline."""
    for method, color in (("unconditional", GREY), ("regime-conditional", BLUE)):
        s = vd[vd.method == method]
        ax.scatter(s.volatility_pct, s.breach_pct, c=color, s=30, label=method,
                   edgecolor=[NAVY if st == "failed" else "none" for st in s.status])
    ax.axhline(5, color=NAVY, ls="--", lw=1)
    ax.text(ax.get_xlim()[1], 5.08, "target 5%", ha="right", fontsize=8.5, color=NAVY)
    ax.set_xlabel("portfolio volatility %")
    ax.set_ylabel("VaR breach rate %")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title(f"95% VaR backtest on {vd.portfolio.nunique()} random portfolios (outlined = failed)",
                 fontsize=10.5, color=NAVY, loc="left")
    _clean(ax)


def var_outcomes(ax, vo: pd.DataFrame):
    vo = vo.set_index("method").reindex(["unconditional", "regime-conditional"])
    other = vo.portfolios - vo.passed - vo.failed
    y = np.arange(len(vo))
    ax.barh(y, vo.passed, color=BLUE, label="passed")
    ax.barh(y, other, left=vo.passed, color=GREY, label="partial")
    ax.barh(y, vo.failed, left=vo.passed + other, color=NAVY, label="failed")
    for i, (p, n) in enumerate(zip(vo.passed, vo.portfolios)):
        ax.text(p / 2, i, f"{p} of {n}", ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(vo.index)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.02))
    ax.set_title("Which VaR method passes its backtest", fontsize=10.5, color=NAVY, loc="left")
    _clean(ax)


def exceptions(ax, re_: pd.DataFrame):
    ax.axis("off")
    ax.set_title(f"Routing exceptions to review ({len(re_)} questions the filter got wrong)",
                 fontsize=10.5, color=NAVY, loc="left")
    rows = [[r.domain, r.query if len(r.query) <= 45 else r.query[:44] + "...", r.chosen.split(".")[-1]]
            for r in re_.itertuples()]
    t = ax.table(cellText=rows, colLabels=["domain", "question", "model chosen"], loc="upper left",
                 cellLoc="left", colWidths=[0.16, 0.6, 0.24])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.35)
    for (i, _), cell in t.get_celld().items():
        cell.set_edgecolor("#DDE6F0")
        if i == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")


def dashboard(d: dict, path: Path):
    cs, ra, vo = d["control_summary"], d["routing_accuracy"], d["var_backtest_outcomes"]
    pick = lambda s: ra[(ra.strategy == s) & (ra.category == "ALL")].accuracy_pct.iloc[0]  # noqa: E731
    regime = vo[vo.method == "regime-conditional"].iloc[0]
    kpis = [(f"{(cs.status == 'fail').sum()} / {len(cs)}", "data-quality controls failed"),
            (f"{pick('filter_only'):.1f}%", "rule-based model selection accuracy"),
            (f"{pick('llm_only'):.1f}%", "LLM-only model selection accuracy"),
            (f"{regime.passed} / {regime.portfolios}", "portfolios where regime VaR passes")]

    fig = plt.figure(figsize=(16, 9.5), facecolor="white")
    fig.suptitle("Audit overview", x=0.02, ha="left", fontsize=16, color=NAVY, weight="bold")
    for i, (value, label) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i * 0.245, 0.80, 0.23, 0.11])
        ax.set_facecolor(PALE)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.05, 0.55, value, fontsize=26, color=BLUE, weight="bold", va="center")
        ax.text(0.05, 0.18, label, fontsize=10.5, color=NAVY)
    controls(fig.add_axes([0.11, 0.42, 0.36, 0.32]), cs)
    routing(fig.add_axes([0.55, 0.42, 0.42, 0.32]), ra)
    var_scatter(fig.add_axes([0.06, 0.06, 0.40, 0.28]), d["var_backtest_detail"])
    exceptions(fig.add_axes([0.55, 0.04, 0.42, 0.30]), d["routing_errors"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _single(draw, data, path: Path, size=(9, 4.8), left=0.08):
    fig = plt.figure(figsize=size, facecolor="white")
    draw(fig.add_axes([left, 0.12, 0.97 - left, 0.76]), data)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(folder: Path = EXPORTS, out: Path = FIGURES) -> list[Path]:
    d = load(folder)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit_dashboard.png": lambda p: dashboard(d, p),
        "audit_controls.png": lambda p: _single(controls, d["control_summary"], p, left=0.18),
        "audit_routing.png": lambda p: _single(routing, d["routing_accuracy"], p),
        "audit_var_backtest.png": lambda p: _single(var_scatter, d["var_backtest_detail"], p),
        "audit_var_outcomes.png": lambda p: _single(var_outcomes, d["var_backtest_outcomes"], p,
                                                    size=(9, 3), left=0.2),
    }
    written = []
    for name, draw in paths.items():
        draw(out / name)
        written.append(out / name)
    print(f"Charts -> {out}: " + ", ".join(p.name for p in written))
    return written


if __name__ == "__main__":
    main()
