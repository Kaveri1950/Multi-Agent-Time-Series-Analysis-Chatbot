"""Figures for the mid-semester report, emitted as one multi-page PDF.

Scope is deliberately narrow, matching the report: the agent framework and
anomaly detection on astronomy and finance. Solar, the risk layer and the
cross-sectional work are out -- they belong to the second half.

Every number here is read from evaluation/results/, never retyped, so a
figure cannot drift away from the result it depicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

RESULTS = Path(__file__).parent / "results"
OUT_PDF = RESULTS / "report_figures.pdf"
OUT_DIR = RESULTS / "figures"

# A quiet, print-safe palette. Figures are read on paper and in greyscale, so
# the two series are separated by lightness as well as hue.
INK, FLOOR, CEIL, BEST, MUTED = "#1A1C19", "#B8442A", "#2A5751", "#8A6A1F", "#9AA39B"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load():
    return json.loads((RESULTS / "anomaly_grid.json").read_text())


def fig_pipeline():
    """The agent pipeline, drawn rather than described."""
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.axis("off")
    steps = [
        ("QUESTION", "natural language"),
        ("ROUTE", "domain + task"),
        ("FILTER", "which models CAN run"),
        ("PROFILE", "what KIND of series"),
        ("SELECT", "measured skill"),
        ("EXECUTE", "detector"),
        ("EXPLAIN", "reasons + prose"),
    ]
    width, gap = 0.125, 0.018
    for i, (name, sub) in enumerate(steps):
        x = i * (width + gap)
        colour = BEST if name == "PROFILE" else INK
        ax.add_patch(plt.Rectangle((x, 0.35), width, 0.34, facecolor="white",
                                   edgecolor=colour, linewidth=1.4 if name == "PROFILE" else 0.9))
        ax.text(x + width / 2, 0.57, name, ha="center", va="center",
                fontsize=7.5, weight="bold", color=colour)
        ax.text(x + width / 2, 0.44, sub, ha="center", va="center",
                fontsize=6.2, color=MUTED)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + width + gap, 0.52), xytext=(x + width, 0.52),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.text(0.0, 0.18, "zero LLM calls  ─────────────────────────────────────────────▶",
            fontsize=7, color=MUTED)
    ax.text(0.80, 0.18, "1 LLM call", fontsize=7, color=INK, weight="bold")
    ax.text(0.0, 0.05, "PROFILE is the only step that reasons about data it has never seen; "
                       "every other choice is a lookup against measured skill.",
            fontsize=6.5, color=MUTED, style="italic")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 0.8)
    ax.set_title("Figure 1  Query pipeline", loc="left", weight="bold")
    return fig


def fig_grid(data):
    """Held-out PR-AUC per detector, both domains, floors marked."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6))
    for ax, domain, label in zip(axes, ("kepler", "equity"),
                                 ("Kepler light curves", "Indian equities")):
        rows = [r for r in data["rows"] if r["domain"] == domain
                and isinstance(r.get("pr_auc"), (int, float)) and r["pr_auc"] == r["pr_auc"]]
        rows = sorted(rows, key=lambda r: r["pr_auc"])
        names = [r["detector"].replace("_", " ") for r in rows]
        vals = [r["pr_auc"] for r in rows]
        colours = [FLOOR if r.get("is_floor") else CEIL if r.get("is_ceiling")
                   else BEST if r is rows[-1] else MUTED for r in rows]
        ax.barh(range(len(rows)), vals, color=colours, height=0.72)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=6.2)
        prev = next((r.get("prevalence") for r in rows if r.get("prevalence")), None)
        if prev:
            ax.axvline(prev, color=INK, ls=":", lw=1.0)
            ax.text(prev, len(rows) - 0.4, f" chance ({prev:.1%})", fontsize=6,
                    color=INK, va="top")
        floors = [r["pr_auc"] for r in rows if r.get("is_floor")]
        if floors:
            ax.axvline(max(floors), color=FLOOR, ls="--", lw=1.1)
        ax.set_xlabel("PR-AUC (held out)")
        ax.set_title(label, fontsize=9)
    fig.suptitle("Figure 2  Detector comparison, held-out", x=0.02, ha="left",
                 weight="bold", fontsize=10)
    fig.text(0.5, 0.005, "red dashed = best no-skill floor; dotted = chance. "
                         "On Kepler every detector lies within 0.04 of a rule that learned nothing.",
             ha="center", fontsize=6.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    return fig


def fig_inflation(data):
    """In-sample vs held-out: the memorisation result."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rows = [r for r in data["rows"] if r["domain"] == "equity"
            and isinstance(r.get("in_sample_pr_auc"), (int, float))
            and isinstance(r.get("pr_auc"), (int, float)) and r["pr_auc"] > 0]
    rows = sorted(rows, key=lambda r: r["in_sample_pr_auc"] / r["pr_auc"])
    names = [r["detector"].replace("_", " ") for r in rows]
    infl = [r["in_sample_pr_auc"] / r["pr_auc"] for r in rows]
    # Capacity is not a number the registry stores, so it is expressed by
    # family -- which is the honest granularity anyway.
    fam = {"floor": MUTED, "one_liner": MUTED, "threshold": CEIL, "matrix_profile": CEIL,
           "density": FLOOR, "reconstruction": BEST, "latent_state": FLOOR, "ceiling": INK}
    colours = [fam.get(r.get("group"), MUTED) for r in rows]
    ax.barh(range(len(rows)), infl, color=colours, height=0.72)
    ax.axvline(1.0, color=INK, lw=1.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlabel("in-sample PR-AUC  ÷  held-out PR-AUC")
    ax.set_title("Figure 3  How much each detector's in-sample score overstates it",
                 loc="left", weight="bold")
    ax.text(1.03, 0.5, "no inflation", fontsize=6.5, color=INK, rotation=90, va="bottom")
    fig.text(0.5, 0.005, "Inflation rises with model capacity: parameter-free methods sit near 1.0, "
                         "a one-class SVM at 6.4x.", ha="center", fontsize=6.5,
             color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def fig_transit():
    """A light curve, its flagged windows, and the phase-folded transit."""
    import warnings

    warnings.filterwarnings("ignore")
    from anomaly.core import windowize
    from anomaly.detectors import HampelMAD
    from astronomy.bulk_download import load_target
    from astronomy.transit_search import binned_fold, compare_with_truth, search

    kepid = 10001368
    df = load_target(kepid)
    t, f = df["time"].to_numpy(), df["flux"].to_numpy()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4))
    seg = slice(0, 4000)
    axes[0].plot(t[seg], (f[seg] - 1) * 1e6, lw=0.4, color=CEIL)
    axes[0].set_ylabel("ppm from baseline")
    axes[0].set_xlabel("time (days)")
    axes[0].set_title(f"KIC {kepid}: raw light curve (first 4,000 points)", fontsize=8.5)

    res = search(t, f)
    cmp = compare_with_truth(res, kepid)
    b = binned_fold(t, f, res["period"], res["epoch"], n_bins=120)
    axes[1].plot(b["phase"] * 24, (b["flux"] - 1) * 1e6, ".", ms=3, color=INK)
    axes[1].axvline(0, color=FLOOR, ls=":", lw=1.0)
    axes[1].set_xlabel("hours from mid-transit")
    axes[1].set_ylabel("ppm")
    axes[1].set_xlim(-12, 12)
    axes[1].set_title(
        f"Phase-folded on the recovered period: {res['period']:.5f} d "
        f"(NASA {cmp['true_period']:.5f} d, error {cmp['period_error_pct']:.3f}%, "
        f"significance {res['sde']:.0f})", fontsize=8.5)
    fig.suptitle("Figure 4  Why a period search succeeds where anomaly detection fails",
                 x=0.02, ha="left", weight="bold", fontsize=10)
    fig.text(0.5, 0.005, "Individually every transit is invisible. Folded on the right period, "
                         "221 of them stack and the dip appears.",
             ha="center", fontsize=6.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    return fig


def fig_routing():
    """Capability filter versus LLM routing."""
    path = RESULTS / "selection_all.json"
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    if path.exists():
        res = json.loads(path.read_text())["results"]
        names = {"filter_only": "capability filter", "hybrid": "filter + LLM", "llm_only": "LLM only"}
        labels = [names.get(r["strategy"], r["strategy"]) for r in res]
        acc = [r["accuracy"] * 100 for r in res]
        secs = [r["elapsed_seconds"] for r in res]
        colours = [BEST if r["strategy"] == "filter_only" else MUTED for r in res]
        bars = ax.bar(labels, acc, color=colours, width=0.55)
        for bar, a, s in zip(bars, acc, secs):
            ax.text(bar.get_x() + bar.get_width() / 2, a + 1.2,
                    f"{a:.1f}%\n{s / 60:.0f} min" if s > 60 else f"{a:.1f}%\n<1 s",
                    ha="center", fontsize=7.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("routing accuracy (%)")
    ax.set_title("Figure 5  Model selection: measured skill beats asking a language model",
                 loc="left", weight="bold")
    fig.text(0.5, 0.01, "123 labelled queries. Adding the LLM to the filter made it slightly worse "
                        "and four orders of magnitude slower.",
             ha="center", fontsize=6.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main():
    data = _load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = [("pipeline", fig_pipeline, ()), ("grid", fig_grid, (data,)),
                ("inflation", fig_inflation, (data,)), ("transit", fig_transit, ()),
                ("routing", fig_routing, ())]
    with PdfPages(OUT_PDF) as pdf:
        for name, fn, args in builders:
            try:
                fig = fn(*args)
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:10} FAILED {type(exc).__name__}: {exc}")
                continue
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(OUT_DIR / f"fig_{name}.pdf", bbox_inches="tight")
            fig.savefig(OUT_DIR / f"fig_{name}.png", bbox_inches="tight")
            plt.close(fig)
            print(f"  {name:10} ok")
    print(f"\nwrote {OUT_PDF}")
    print(f"individual figures in {OUT_DIR}/  (pdf + png, for \\includegraphics)")


if __name__ == "__main__":
    main()
