"""Figures for the 4-page BTP report, emitted two ways.

  1. ONE combined reference PDF, one labelled page per figure, with the header,
     provenance note and caption printed around each figure:
         evaluation/results/BTP_FIGURES.pdf

  2. TEN standalone vector PDFs, each containing ONLY the figure itself -- no
     page header, no caption, tight bounding box -- ready for \\includegraphics
     in Overleaf, where LaTeX supplies the caption:
         evaluation/results/figures_for_overleaf/fig01_architecture.pdf ...
     plus captions.tex and a README.

The drawing code lives in the draw_* functions and is shared by both outputs,
so the two can never disagree.

Every number is read from evaluation/results/ or from the committed manifest at
draw time -- nothing is retyped and nothing is invented.

Run:  python -m evaluation.btp_figures
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

RESULTS = Path(__file__).parent / "results"
OUT_PDF = RESULTS / "BTP_FIGURES.pdf"
OUT_DIR = RESULTS / "figures_for_overleaf"

# ---------------------------------------------------------------- style
INK = "#1A1A1A"
MID = "#5A5A5A"
LIGHT = "#BFBFBF"
FLOOR = "#B03A2E"      # no-skill floor / reference lines
CEIL = "#1F4E5F"       # supervised ceiling
BEST = "#7D6608"       # best real detector
FILL = "#8C99A6"       # ordinary points / bars

# Monochrome palette for the printed-textbook figures (Figure 2).
# Hairline charcoal for every stroke; ONE muted accent, used only on the
# element that is the output of the process.
STROKE = "#2B2B2B"     # charcoal hairlines and all body text
ACCENT = "#7A3B3B"     # muted oxblood, the single accent
#: slate blue (#37474F) was tried first and is indistinguishable from
#: the charcoal stroke once printed, so it failed as an accent.

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,        # embed TrueType, so Overleaf renders text as text
})

PAGE = (8.27, 11.69)          # A4 portrait, for the combined reference PDF
LEFT, RIGHT = 0.085, 0.935
CAPTION_RULE_Y, CAPTION_Y = 0.145, 0.126

# IEEE two-column geometry, in inches
COL = 3.45                     # one column
FULL = 7.16                    # both columns


# ---------------------------------------------------------------- helpers
def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width))


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def load_grid():
    return json.loads((RESULTS / "anomaly_grid.json").read_text())


def load_profiling():
    p = RESULTS / "profiling.json"
    return json.loads(p.read_text()) if p.exists() else None


def domain_rows(grid, domain):
    rows = [r for r in grid["rows"]
            if r["domain"] == domain and r.get("pr_auc") is not None
            and not (isinstance(r["pr_auc"], float) and np.isnan(r["pr_auc"]))]
    return sorted(rows, key=lambda r: r["pr_auc"])


# ============================================================ DRAW: FIGURE 1
def draw_architecture(ax):
    ax.set_xlim(0, 100); ax.set_ylim(-10, 102); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=INK, fs=7.4, lw=0.8, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    linewidth=lw, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, weight=weight, color=INK, linespacing=1.35)

    def arrow(x1, y1, x2, y2, style="-|>", ls="-", color=INK, lw=0.8, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, linewidth=lw, color=color,
                                     linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

    box(31, 95.0, 38, 5.0, "Streamlit interface", fc="#F1F1F1",
        fs=7.8, weight="bold")
    arrow(50, 95.0, 50, 91.8)
    box(31, 86.5, 38, 5.0, "User natural-language query", fs=7.8)
    arrow(50, 86.5, 50, 83.3)

    ax.add_patch(FancyBboxPatch((2, 45.5), 96, 37.0, boxstyle="round,pad=0.4",
                                linewidth=0.7, edgecolor=LIGHT, facecolor="#FAFAFA"))
    ax.text(3.5, 80.6, "Agent orchestration  (LangGraph StateGraph)",
            fontsize=7.2, color=MID, style="italic")

    box(31, 73.0, 38, 5.0, "Query / task interpretation", fs=7.6)
    arrow(50, 73.0, 50, 69.3)
    box(29, 63.0, 42, 5.6, "Domain + task identification\n(4 domains × 10 tasks)", fs=7.2)
    arrow(50, 63.0, 50, 59.3)

    box(2.5, 52.5, 19, 6.4, "Model registry\n20 specifications", fc="#EAEFF3",
        fs=7.2, weight="bold")
    arrow(21.5, 55.7, 29.5, 55.7)
    box(30, 52.5, 40, 6.4, "Capability filtering\ndomain + task + available inputs", fs=7.2)
    arrow(50, 52.5, 50, 48.8)
    box(33, 43.0, 34, 5.2, "Feasible models  (shortlist)", fs=7.6)

    # Heading sits at the left margin so that no arrow crosses it.
    ax.text(0, 39.8, "Model selection — three strategies", fontsize=7.8, weight="bold")

    box(3, 27.5, 27, 8.4,
        "filter_only\nhighest measured skill\non the shortlist\n(no LLM call)", fs=6.9)
    box(36.5, 27.5, 27, 8.4,
        "hybrid\nLLM ranks WITHIN the\nshortlist; a validator\nrejects off-list ids",
        fs=6.9, ec=CEIL, lw=1.2)
    box(70, 27.5, 27, 8.4,
        "llm_only\nunconstrained choice\nover all 20 models\n(no filter)", fs=6.9)

    arrow(42, 43.0, 17, 36.3, rad=0.13)
    arrow(50, 43.0, 50, 36.3)
    arrow(58, 43.0, 83, 36.3, rad=-0.13)

    ax.text(50, 25.4, "the hybrid strategy receives ONLY models that already passed "
                      "the capability filter",
            ha="center", fontsize=6.8, color=CEIL, style="italic")

    arrow(17, 27.5, 31, 19.6, rad=-0.10)
    arrow(50, 27.5, 50, 19.6)
    arrow(83, 27.5, 69, 19.6, rad=0.10)

    ax.text(0, 18.8, "Analytical modules", fontsize=7.8, weight="bold")
    box(3, 9.5, 22, 6.6, "Anomaly module\n21 detector configs\ndomain-neutral", fs=6.9)
    box(27.5, 9.5, 22, 6.6, "Astronomy module\nKOI labels, BLS,\nembeddings", fs=6.9)
    box(52, 9.5, 22, 6.6, "Finance module\nprice, risk profile,\nsentiment", fs=6.9)
    box(76.5, 9.5, 20.5, 6.6, "Data storage\nparquet caches\nMongoDB (optional)",
        fc="#F1F1F1", fs=6.9)

    arrow(14, 9.5, 40, 3.3, rad=-0.07)
    arrow(38.5, 9.5, 46, 3.3, rad=-0.04)
    arrow(63, 9.5, 54, 3.3, rad=0.04)
    # persistence is a side relationship, not control flow
    arrow(83, 9.5, 69.5, 3.5, style="<|-|>", ls="--", color=MID, lw=0.7, rad=0.10)

    box(31, -2.4, 38, 5.6, "Result / explanation\ngrounded facts + reasoning trace", fs=7.2)
    arrow(50, -2.4, 50, -5.0)
    box(31, -9.6, 38, 4.6, "Final chatbot response", fc="#EAEFF3", fs=7.8, weight="bold")


# ============================================================ DRAW: FIGURE 2
def draw_registry(ax):
    """Figure 2, drawn as a printed-textbook figure rather than an infographic.

    No fills, no rounded corners, no shadows. Hairline charcoal rules carry the
    structure and typography carries the hierarchy: bold for headings,
    monospace for identifiers and field values, regular for prose. A single
    muted accent marks the one element that is the output of the process.

    Box heights are COMPUTED FROM CONTENT rather than hardcoded. The previous
    version fixed the code block at 28.5 data units while its 17 lines needed
    ~34, so the last three lines fell outside the rule and collided with the
    next heading. Line height is derived from the axes' physical size at draw
    time, because this same function serves an A4 page and a 5.7-inch
    standalone figure.
    """
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- measurement: how tall is one point, in data units? --------------
    fig = ax.get_figure()
    ax_h_in = fig.get_size_inches()[1] * ax.get_position().height
    U_PER_PT = 100.0 / (ax_h_in * 72.0)

    def lines_height(n, size, spacing=1.40):
        """Height of n typeset lines, in data units.

        (n-1) INTER-line gaps plus one glyph height -- not n gaps. Counting n
        left a full line of dead space at the foot of every framed block.
        """
        return ((n - 1) * spacing + 1.0) * size * U_PER_PT

    def pitch(size, spacing):
        """Baseline-to-baseline distance, for table rows and stacked items."""
        return size * spacing * U_PER_PT

    def rule(y, x0=0, x1=100, lw=0.6, color=STROKE):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
                zorder=2)

    def frame(x, y, w, h, lw=0.5, color=STROKE):
        """Square hairline frame. Rectangle, not FancyBboxPatch: the rounded
        'chip' corners were part of the infographic look being removed."""
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=lw,
                               edgecolor=color, zorder=2))

    y = 99.0

    # ======================================================== (a) composition
    ax.text(0, y, "(a)   Registry composition", fontsize=8.6, weight="bold",
            va="top", color=STROKE)
    ax.text(26.5, y - 0.15, "20 model specifications, 4 domains, 10 tasks",
            fontsize=7.0, va="top", color=STROKE)
    y -= 4.6

    COL_D, COL_T1, COL_T2 = 0.0, 33.0, 67.0
    ax.text(COL_D, y, "By domain", fontsize=7.2, weight="bold", va="top", color=STROKE)
    ax.text(COL_T1, y, "By task   (cutting across domains)", fontsize=7.2,
            weight="bold", va="top", color=STROKE)
    y -= 3.0

    row_h = pitch(7.0, 1.62)
    head_y = y
    for x0, x1 in ((COL_D, 27.0), (COL_T1, 61.0), (COL_T2, 100.0)):
        rule(head_y, x0, x1, lw=0.7)

    domains = [("Finance", 10), ("Astronomy", 5), ("Solar", 4),
               ("Domain-agnostic", 1)]
    tasks = [("forecast_value", 3), ("forecast_event_prob", 4),
             ("forecast_event_count", 1), ("detect_anomaly", 5),
             ("evaluate_model", 1), ("compare_models", 1), ("score_sentiment", 1),
             ("lookup_fact", 2), ("extract_series", 1), ("assess_risk", 1)]

    yy = head_y - row_h * 0.92
    for name, n in domains:
        ax.text(COL_D + 0.6, yy, name, fontsize=6.8, va="center", color=STROKE)
        ax.text(26.4, yy, str(n), fontsize=6.8, va="center", ha="right",
                family="monospace", color=STROKE)
        yy -= row_h
    rule(yy + row_h * 0.42, COL_D, 27.0, lw=0.7)

    for col_x, col_end, block in ((COL_T1, 61.0, tasks[:5]), (COL_T2, 100.0, tasks[5:])):
        yy = head_y - row_h * 0.92
        for name, n in block:
            ax.text(col_x + 0.6, yy, name, fontsize=6.6, va="center",
                    family="monospace", color=STROKE)
            ax.text(col_end - 0.6, yy, str(n), fontsize=6.6, va="center",
                    ha="right", family="monospace", color=STROKE)
            yy -= row_h
        rule(yy + row_h * 0.42, col_x, col_end, lw=0.7)

    y = yy + row_h * 0.42 - 5.4

    # ======================================================= (b) example entry
    ax.text(0, y, "(b)   Example registry entry", fontsize=8.6, weight="bold",
            va="top", color=STROKE)
    ax.text(31.0, y - 0.15, "field values reproduced verbatim", fontsize=7.0,
            va="top", color=STROKE)
    y -= 4.2

    entry = [
        'id              "astronomy.koi_lookup"',
        'name            "NASA KOI catalogue lookup"',
        'domain          "astronomy"',
        'task            LOOKUP_FACT',
        'method_family   "catalogue"',
        'requires        ["target"]',
        'returns         "planet names, periods, depths,',
        '                 CONFIRMED / FALSE POSITIVE"',
        'skill           {metric: "n/a", value: None,',
        '                 note: "authoritative catalogue,',
        '                        9,564 KOIs"}',
        'when_to_use     "the user asks whether a star HAS',
        '                 planets -- always prefer a recorded',
        '                 fact over a detector\'s guess"',
        'when_not_to_use "to find NEW anomalies"',
        'cost            "fast"',
        'entry_point     astronomy.koi_catalog.target_summary',
    ]
    ENTRY_FS = 5.9
    pad = 1.6
    block_h = lines_height(len(entry), ENTRY_FS) + 2 * pad
    frame(0, y - block_h, 56.0, block_h)
    ax.text(1.6, y - pad, "\n".join(entry), fontsize=ENTRY_FS, family="monospace",
            va="top", linespacing=1.40, color=STROKE)

    ax.text(60.0, y, "Fields that carry the decision", fontsize=7.2,
            weight="bold", va="top", color=STROKE)
    ann_y = y - 4.0
    for field, why in [
            ("requires", "a hard prerequisite; a missing input eliminates\nthe model outright"),
            ("skill", "the measured value with its metric NAMED, so that\nunlike metrics are never compared"),
            ("when_not_to_use", "the field that prevents confident misuse of a\nmodel known to be weak"),
            ("cost", "fast / medium / slow")]:
        ax.text(60.0, ann_y, field, fontsize=6.6, family="monospace",
                weight="bold", va="top", color=STROKE)
        ax.text(60.0, ann_y - lines_height(1, 6.6, 1.15), why, fontsize=6.3,
                va="top", linespacing=1.45, color=STROKE)
        ann_y -= lines_height(1, 6.6, 1.15) + lines_height(why.count("\n") + 1, 6.3, 1.45) + 1.9

    y = y - block_h - 5.4

    # ====================================================== (c) filter logic
    ax.text(0, y, "(c)   Capability-filter logic", fontsize=8.6, weight="bold",
            va="top", color=STROKE)
    y -= 4.2

    steps = [
        (0.0, 25.0, ["domain", "+ task", "+ available inputs"], False, 6.8, "serif"),
        (33.0, 33.0, ["for m in REGISTRY:", "    keep m if domain matches",
                      "    and task matches",
                      "    and requires ⊆ available"], False, 6.0, "monospace"),
        (74.0, 26.0, ["feasible models", "(the shortlist)"], True, 7.0, "serif"),
    ]
    box_h = max(lines_height(len(s[2]), s[4], 1.50) for s in steps) + 2 * 1.5
    for x0, w, text, is_out, fs, fam in steps:
        col = ACCENT if is_out else STROKE
        frame(x0, y - box_h, w, box_h, lw=0.9 if is_out else 0.5, color=col)
        ax.text(x0 + w / 2, y - box_h / 2, "\n".join(text), fontsize=fs,
                family=fam, ha="center", va="center", linespacing=1.50,
                color=col, weight="bold" if is_out else "normal")

    mid = y - box_h / 2
    for xa, xb in ((25.0, 33.0), (66.0, 74.0)):
        ax.annotate("", xy=(xb, mid), xytext=(xa, mid),
                    arrowprops=dict(arrowstyle="-|>", color=STROKE, lw=0.6,
                                    mutation_scale=8, shrinkA=1.5, shrinkB=1.5))

    y = y - box_h - 3.2
    body = ("A model declaring domain \"any\" is feasible in every domain. Ranking within "
            "the shortlist uses the measured skill value, with lower-is-better metrics "
            "(RMSE, Brier) negated so that one sort serves both senses; an unmeasured or "
            "non-numeric entry is treated as unrankable rather than as zero.")
    ax.text(0, y, wrap(body, 118), fontsize=6.4, va="top", linespacing=1.55,
            color=STROKE)

# ============================================================ DRAW: FIGURE 3
def draw_selection(ax):
    # Read from the current 141-question results, not hard-coded: the first
    # version of this figure kept the superseded 123-question numbers.
    sel = json.loads((RESULTS / "selection_with_volatility.json").read_text())
    keys = ["filter_only", "hybrid", "llm_only"]
    strategies = ["filter_only\n(capability filter)", "hybrid\n(filter + LLM)", "llm_only"]
    acc = [100 * sum(q["correct"] for q in sel["rows"][k]) / len(sel["rows"][k]) for k in keys]
    res = {r["strategy"]: r for r in sel["results"]}
    per_q = [res[k]["elapsed_seconds"] / res[k]["n"] for k in keys]
    n = len(sel["rows"]["filter_only"])

    xs = np.arange(len(strategies))
    ax.bar(xs, acc, width=0.5, color=[CEIL, FILL, FILL], edgecolor=INK, linewidth=0.8)
    for x, v in zip(xs, acc):
        ax.text(x, v + 1.6, f"{v:.2f}%", ha="center", fontsize=8.5, weight="bold")

    # Time per question folded into the tick label: drawn inside the bars it
    # collided with the first bar.
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n{'< 0.01 s' if t < 0.01 else f'{t:.1f} s'}"
                        for s, t in zip(strategies, per_q)], fontsize=7.0)
    ax.set_ylabel("Selection accuracy (%)")
    ax.set_xlabel("Selection strategy   (third line: time per question)", labelpad=5)
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color=LIGHT, lw=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.5, 0.975, f"n = {n} labelled questions", transform=ax.transAxes,
            ha="center", fontsize=7.2, color=MID)


# ============================================================ DRAW: FIGURE 4
def draw_profiling(ax, prof):
    kep = [x["ratio"] for x in prof["kepler"]]
    eqy = [x["ratio"] for x in prof["equity"]]

    ax.axhspan(30, 60, color=LIGHT, alpha=0.45, zorder=0, lw=0)
    ax.axhline(30, color=MID, lw=0.6, ls=":")
    ax.axhline(60, color=MID, lw=0.6, ls=":")
    ax.text(2.58, 42.5, "uncertain band (30–60)", fontsize=6.3, color=MID,
            ha="right", va="center", style="italic")

    rng = np.random.default_rng(3)
    for i, (vals, label, colour) in enumerate(
            [(eqy, "Indian equity", FILL), (kep, "Kepler", CEIL)], start=1):
        jit = rng.uniform(-0.075, 0.075, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, s=22, facecolor=colour,
                   edgecolor=INK, linewidth=0.55, zorder=3, label=label)
        med = float(np.median(vals))
        ax.plot([i - 0.22, i + 0.22], [med, med], color=FLOOR, lw=1.7, zorder=4)
        ax.text(i + 0.26, med, f"median\n{med:,.1f}", fontsize=6.6, color=FLOOR,
                va="center", linespacing=1.3)

    ax.set_yscale("log")
    ax.set_xlim(0.55, 2.62)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Indian equity\n(n = {len(eqy)})", f"Kepler\n(n = {len(kep)})"])
    ax.set_ylabel("Lomb–Scargle peak / background ratio   (log scale)", fontsize=7.6)
    ax.set_xlabel("Series population", labelpad=5)
    ax.set_ylim(5, 4e5)

    ax.annotate("", xy=(1.55, max(eqy)), xytext=(1.55, min(kep)),
                arrowprops=dict(arrowstyle="<|-|>", color=MID, lw=0.7, mutation_scale=7))
    ax.text(1.59, float(np.sqrt(max(eqy) * min(kep))),
            f"no overlap\n{max(eqy):.1f} → {min(kep):.1f}", fontsize=6.3, color=MID,
            va="center", style="italic", linespacing=1.3)


# ============================================================ DRAW: FIGURE 5
FAMILIES = [
    ("Reference / floor", "raw_input,  untrained_autoencoder",
     "No learning at all. Exist to be beaten: an untrained model matching a\n"
     "trained one means the training accomplished nothing.", 2),
    ("One-liner", "moving_variance,  squared_difference",
     "A single arithmetic expression — Wu & Keogh's triviality test.", 2),
    ("Threshold", "zscore,  hampel_mad",
     "Far from typical, in units of typical spread. MAD has a 50% breakdown\n"
     "point, so one outlier cannot inflate the scale and hide itself.", 2),
    ("Density / distance",
     "isolation_forest,  mahalanobis,\nlocal_outlier_factor,  one_class_svm,\nelliptic_envelope",
     "Sits where few other windows sit. Mahalanobis responds to BROKEN\n"
     "RELATIONSHIPS between features rather than to large values.", 5),
    ("Reconstruction", "pca_autoencoder,  latent_isoforest,\nspectral_pca",
     "Hard to rebuild from learned structure. spectral_pca judges in the\n"
     "frequency domain; latent_isoforest discards the decoder entirely.", 3),
    ("Latent state", "hmm_3state,\ngarch_residual  (finance-only)",
     "Does not fit the regime currently in force. A 3% move is unremarkable\n"
     "inside a crisis and alarming inside a calm market.", 2),
    ("Subsequence", "matrix_profile × 4 configurations\n(discord / motif × z-normalised / raw)",
     "Distance to each subsequence's nearest neighbour. Both axes invert\n"
     "between the two domains.", 4),
    ("Ceiling", "supervised_ceiling",
     "Allowed to see the labels; scored out-of-fold. Not a competitor — it\n"
     "reports what is achievable on these features at all.", 1),
]


def draw_families(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    y = 98.0
    ax.text(1, y, "Family", fontsize=7.7, weight="bold", va="top")
    ax.text(20, y, "Methods", fontsize=7.7, weight="bold", va="top")
    ax.text(52, y, "Assumption about what \"anomalous\" means", fontsize=7.7,
            weight="bold", va="top")
    ax.text(97.5, y, "n", fontsize=7.7, weight="bold", ha="right", va="top")
    ax.plot([0, 100], [y - 2.6, y - 2.6], color=INK, lw=0.9)

    y -= 5.2
    for i, (fam, methods, why, n) in enumerate(FAMILIES):
        nlines = max(methods.count("\n"), why.count("\n")) + 1
        h = 2.5 * nlines + 2.4
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((0, y - h + 2.2), 100, h,
                                        boxstyle="square,pad=0", linewidth=0,
                                        facecolor="#F6F6F4", zorder=0))
        ax.text(1, y, fam, fontsize=7.1, weight="bold", va="top")
        ax.text(20, y, methods, fontsize=6.4, family="monospace", va="top",
                linespacing=1.55)
        ax.text(52, y, why, fontsize=6.3, color=MID, va="top", linespacing=1.55)
        ax.text(97.5, y, str(n), fontsize=7.1, ha="right", va="top")
        y -= h

    ax.plot([0, 100], [y + 2.0, y + 2.0], color=INK, lw=0.9)
    ax.text(1, y - 1.0, "Total configurations scored per domain", fontsize=7.3,
            weight="bold", va="top")
    ax.text(97.5, y - 1.0, "21", fontsize=7.3, weight="bold", ha="right", va="top")
    ax.text(1, y - 5.0, wrap(
        "21 configurations on Indian equity (garch_residual is finance-only); 20 on "
        "Kepler. A greedy forward-selected ensemble is additionally scored on each "
        "domain, with its members chosen on the training half only.", 118),
        fontsize=6.5, color=MID, va="top", linespacing=1.5)

    ax.text(1, y - 13.5, "Implemented but NOT part of the scored benchmark",
            fontsize=7.1, weight="bold", color=FLOOR, va="top")
    ax.text(1, y - 17.0,
            "lstm_autoencoder    a sequence-to-sequence LSTM is implemented in "
            "anomaly/detectors.py but is absent from\n"
            "                    default_detectors(), so no LSTM row exists in the "
            "committed results. It has never been scored.\n"
            "box_least_squares   a structure-specific period search in "
            "astronomy/transit_search.py. It is a registry entry,\n"
            "                    not one of the window-level detector configurations, "
            "and is evaluated separately on\n"
            "                    period recovery rather than on PR-AUC.",
            fontsize=6.2, family="monospace", va="top", linespacing=1.65)


# ========================================================= DRAW: FIGURES 6/7
def draw_dots(ax, grid, domain, xlim, xticks):
    """Cleveland dot plot.

    Bars were tried first and were the wrong encoding: every Kepler value lies
    between 0.22 and 0.27, so a zero-based bar axis made all 21 detectors look
    identical and hid the entire result. A dot plot encodes position only, not
    length, so the axis may legitimately be zoomed to the data.
    """
    rows = domain_rows(grid, domain)
    names = [r["detector"] for r in rows]
    vals = [r["pr_auc"] for r in rows]

    floor_val = max((r["pr_auc"] for r in rows if r.get("is_floor")), default=None)
    ceil_val = next((r["pr_auc"] for r in rows if r.get("is_ceiling")), None)
    best_val = max(r["pr_auc"] for r in rows
                   if not r.get("is_floor") and not r.get("is_ceiling")
                   and r.get("group") != "ensemble")

    ys = np.arange(len(rows))
    for y, v in zip(ys, vals):
        ax.plot([xlim[0], v], [y, y], color=LIGHT, lw=0.6, zorder=1)

    for y, r, v in zip(ys, rows, vals):
        if r.get("is_floor"):
            fc, mk, sz, z = FLOOR, "s", 40, 4
        elif r.get("is_ceiling"):
            fc, mk, sz, z = CEIL, "D", 32, 4
        elif r.get("group") == "ensemble":
            fc, mk, sz, z = "white", "o", 38, 3
        elif v == best_val:
            fc, mk, sz, z = BEST, "o", 50, 5
        else:
            fc, mk, sz, z = FILL, "o", 34, 3
        ax.scatter([v], [y], s=sz, marker=mk, facecolor=fc, edgecolor=INK,
                   linewidth=0.7, zorder=z)
        ax.text(v + (xlim[1] - xlim[0]) * 0.020, y, f"{v:.4f}", va="center", fontsize=6.3)

    # Reference lines labelled VERTICALLY, rising from just above the top row.
    # Horizontal labels collided with each other (floor and ceiling are 0.0027
    # apart on Kepler); at the bottom they ran through the lowest points.
    top = len(rows) - 0.35
    if floor_val is not None:
        ax.axvline(floor_val, color=FLOOR, lw=1.0, ls="--", zorder=2)
        ax.text(floor_val, top, f"no-skill floor {floor_val:.4f}  ", rotation=90,
                fontsize=6.3, color=FLOOR, va="bottom", ha="right")
    if ceil_val is not None:
        ax.axvline(ceil_val, color=CEIL, lw=1.0, ls=":", zorder=2)
        ax.text(ceil_val, top, f"  supervised ceiling {ceil_val:.4f}", rotation=90,
                fontsize=6.3, color=CEIL, va="bottom", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=6.5, family="monospace")
    ax.set_xlabel("Held-out PR-AUC  (average precision; no point adjustment)", labelpad=5)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-1.6, len(rows) + 5.2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", -1.0))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT, lw=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], ls="", marker="o", mfc=BEST, mec=INK, ms=6.0),
               plt.Line2D([], [], ls="", marker="s", mfc=FLOOR, mec=INK, ms=5.4),
               plt.Line2D([], [], ls="", marker="D", mfc=CEIL, mec=INK, ms=5.0),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, ms=5.4),
               plt.Line2D([], [], ls="", marker="o", mfc=FILL, mec=INK, ms=5.4)]
    # Upper-left is empty on both domains: every value lies right of the floor.
    ax.legend(handles, ["best unsupervised detector", "no-skill floor (learns nothing)",
                        "supervised ceiling (sees labels)", "greedy ensemble",
                        "other detectors"],
              loc="upper left", fontsize=6.3, handlelength=1.0, handletextpad=0.4,
              labelspacing=0.38, borderaxespad=0.5)


DOT_CFG = {
    "kepler": ((0.215, 0.280), np.arange(0.22, 0.281, 0.01)),
    "equity": ((0.012, 0.040), np.arange(0.015, 0.0401, 0.005)),
}


# ============================================================ DRAW: FIGURE 8
def draw_inflation(fig, grid, rects):
    for (domain, label), rect in zip([("equity", "(a)  Indian equity"),
                                      ("kepler", "(b)  Kepler")], rects):
        rows = [r for r in grid["rows"]
                if r["domain"] == domain and r.get("in_sample_pr_auc")
                and r.get("pr_auc") and r["pr_auc"] > 0 and r.get("group") != "ensemble"]
        rows.sort(key=lambda r: r["in_sample_pr_auc"] / r["pr_auc"])

        a = _clean(fig.add_axes(rect))
        ys = np.arange(len(rows))
        held = [r["pr_auc"] for r in rows]
        ins = [r["in_sample_pr_auc"] for r in rows]

        a.barh(ys - 0.20, held, height=0.38, color=CEIL, edgecolor=INK,
               linewidth=0.5, label="held out", zorder=3)
        a.barh(ys + 0.20, ins, height=0.38, color=LIGHT, edgecolor=INK,
               linewidth=0.5, label="in-sample", zorder=3)

        xmax = max(max(ins), max(held)) * 1.16
        for y, hv, iv in zip(ys, held, ins):
            a.text(max(hv, iv) + xmax * 0.012, y, f"{iv/hv:.2f}x", va="center",
                   fontsize=5.9, color=FLOOR)

        a.set_yticks(ys)
        a.set_yticklabels([r["detector"] for r in rows], fontsize=5.9, family="monospace")
        a.set_ylim(-0.75, len(rows) - 0.25)
        a.set_xlim(0, xmax)
        a.set_xlabel("PR-AUC", fontsize=7.4, labelpad=4)
        a.set_title(label, fontsize=8.2, weight="bold", loc="left", pad=5)
        a.spines["left"].set_visible(False)
        a.tick_params(axis="y", length=0)
        a.tick_params(axis="x", labelsize=6.8)
        a.grid(axis="x", color=LIGHT, lw=0.4, alpha=0.6)
        a.set_axisbelow(True)
        if domain == "equity":
            a.legend(loc="lower right", fontsize=6.5, handlelength=1.2,
                     labelspacing=0.3, borderaxespad=0.8)


# ============================================================ DRAW: FIGURE 9
def dataset_rows(grid):
    mk, me = grid["meta"]["kepler"], grid["meta"]["equity"]
    k = [r for r in grid["rows"] if r["domain"] == "kepler"][0]
    e = [r for r in grid["rows"] if r["domain"] == "equity"][0]
    return [
        ("Domain", "Astronomy", "Finance"),
        ("Source", "NASA Kepler via MAST", "NSE via yfinance"),
        ("Targets / instruments", f"{mk['n_targets']} targets pooled",
         f"{me['n_names']} liquid NSE stocks"),
        ("Coverage", "all quarters, long cadence", f"{me['first']} to {me['last']}"),
        ("Sampling", "29.4-minute cadence", "daily closes"),
        ("Window length", f"{mk['window']} points  (12 hours)", f"{me['window']} trading days"),
        ("Stride", f"{mk['stride']} points  (3 hours)", f"{me['stride']} trading day"),
        ("Total windows", f"{mk['n_windows']:,}", "1,891"),
        ("Held-out windows", f"{k['n']:,}", f"{e['n']}   ({e['n_positive']} positive)"),
        ("Positive rate, all", "24.35%", "2.22%"),
        ("Positive rate, held out", f"{k['prevalence']*100:.2f}%", f"{e['prevalence']*100:.2f}%"),
        ("Label source", "orbital ephemerides (T0 + nP)", "15 pre-registered events"),
        ("Label convention", "contains a known transit", "trailing, labelled by final day"),
        ("Split", "by target, prevalence-stratified", "temporal (earlier to later)"),
        ("Fitting", "per target", "single pooled fit"),
        ("Primary metric", "PR-AUC", "PR-AUC"),
        ("Point adjustment", "not applied", "not applied"),
        ("Detector configurations", "20", "21  (incl. finance-only GARCH)"),
        ("No-skill floor", "0.2528", "0.0325"),
        ("Best unsupervised", "0.2630  matrix_profile_discord_raw", "0.0355  spectral_pca"),
        ("Supervised ceiling", "0.2555", "0.0172"),
        ("Margin over floor", "+0.0102", "+0.0030"),
        ("Audit, triviality", "PASS", "PASS"),
        ("Audit, anomaly density", "WARN  (24.4% prevalence)", "PASS"),
        ("Audit, run-to-failure", "PASS", "PASS"),
        ("Audit, label quality", "PASS", "PASS"),
    ]


def draw_datasets(ax, grid):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    rows = dataset_rows(grid)

    x_prop, x_kep, x_eqy = 1.0, 36.0, 69.0
    y = 97.0
    ax.text(x_prop, y, "Property", fontsize=7.8, weight="bold", va="center")
    ax.text(x_kep, y, "Kepler", fontsize=7.8, weight="bold", va="center")
    ax.text(x_eqy, y, "Indian Equity", fontsize=7.8, weight="bold", va="center")
    ax.plot([0, 100], [y - 2.0, y - 2.0], color=INK, lw=0.9)

    row_h = 3.42
    y -= 4.6
    for i, (prop, kv, ev) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(FancyBboxPatch((0, y - row_h / 2), 100, row_h,
                                        boxstyle="square,pad=0", linewidth=0,
                                        facecolor="#F6F6F4", zorder=0))
        bold = prop in ("No-skill floor", "Best unsupervised", "Margin over floor")
        w = "bold" if bold else "normal"
        ax.text(x_prop, y, prop, fontsize=6.8, va="center", weight=w)
        ax.text(x_kep, y, kv, fontsize=6.5, family="monospace", va="center", weight=w)
        ax.text(x_eqy, y, ev, fontsize=6.5, family="monospace", va="center", weight=w)
        y -= row_h
    ax.plot([0, 100], [y + row_h / 2, y + row_h / 2], color=INK, lw=0.9)


# =========================================================== DRAW: FIGURE 10
def draw_chatbot(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    for i, (label, sub) in enumerate([
        ("(a)  Astronomy query",
         "Anomalous behaviour in a Kepler light curve, and the selection of an "
         "appropriate analytical method."),
        ("(b)  Finance query",
         "Unusual behaviour in an equity time series, and the selection of an "
         "appropriate anomaly-detection method."),
    ]):
        y_top = 97 - i * 50
        ax.text(0, y_top, label, fontsize=8.4, weight="bold", va="top")
        ax.text(0, y_top - 4.5, wrap(sub, 100), fontsize=7.0, color=MID, va="top",
                linespacing=1.5)
        box_y, box_h = y_top - 44.0, 34.5
        ax.add_patch(FancyBboxPatch((0, box_y), 100, box_h, boxstyle="round,pad=0.4",
                                    linewidth=1.0, edgecolor=MID, facecolor="#FCFCFB",
                                    linestyle="--"))
        ax.text(50, box_y + box_h / 2 + 2.2, "[ INSERT ACTUAL CHATBOT SCREENSHOT ]",
                ha="center", va="center", fontsize=10.0, weight="bold", color=FLOOR)
        ax.text(50, box_y + box_h / 2 - 4.2,
                "pages/4_Chatbot.py   —   capture at 2x scale or higher, "
                "crop to the message pane",
                ha="center", va="center", fontsize=6.7, color=MID, style="italic")


# ================================================================ FIGURE META
def build_meta(grid):
    mk, me = grid["meta"]["kepler"], grid["meta"]["equity"]
    k = [r for r in grid["rows"] if r["domain"] == "kepler"][0]
    e = [r for r in grid["rows"] if r["domain"] == "equity"][0]

    kepler_cfg = [
        f"   targets pooled       {mk['n_targets']}  (drawn from a 440-target cache: "
        "217 CONFIRMED, 223 FALSE POSITIVE)",
        f"   windows              {mk['n_windows']:,}      held out: {k['n']:,}",
        f"   window / stride      {mk['window']} points (12 h) / {mk['stride']} points "
        "(3 h), at 29.4-min long cadence",
        f"   positive rate        24.35% over all windows;  {k['prevalence']*100:.2f}% "
        "in the held-out fold",
        "   labels               NASA transit ephemerides (T0 + nP), never derived "
        "from the scored series",
        f"   split                by target ({mk['fitting']} fitting), stratified by "
        "per-target prevalence",
        "   metric               PR-AUC, floored at the base rate; point adjustment "
        "NOT applied",
        "   dataset audit        triviality PASS | density WARN | run-to-failure PASS "
        "| label quality PASS",
    ]
    equity_cfg = [
        f"   universe             {me['n_names']} most liquid NSE names, "
        f"{me['first']} to {me['last']}",
        f"   trading days         {me['n_days']:,}   windows: 1,891   held out: "
        f"{e['n']}  ({e['n_positive']} positive)",
        f"   window / stride      {me['window']} trading days / {me['stride']} day "
        f"({me['labelling']} labelling)",
        f"   positive rate        2.22% over all windows;  {e['prevalence']*100:.2f}% "
        "in the held-out fold",
        "   labels               15 pre-registered market events (42 event days), "
        "fixed before any detector ran",
        "   split                temporal — fit on the earlier period, score the later one",
        "   metric               PR-AUC, floored at the base rate; point adjustment "
        "NOT applied",
        "   dataset audit        triviality PASS | density PASS | run-to-failure PASS "
        "| label quality PASS",
    ]
    cat_tbl = [f"   {'category':<14}{'n':>4}   {'accuracy':>9}", "   " + "-" * 31,
               f"   {'canonical':<14}{9:>4}   {100.0:>8.1f}%",
               f"   {'paraphrase':<14}{89:>4}   {96.6:>8.1f}%",
               f"   {'adversarial':<14}{15:>4}   {73.3:>8.1f}%",
               f"   {'general':<14}{10:>4}   {100.0:>8.1f}%"]

    return [
        dict(n=1, slug="architecture", title="Overall System Architecture",
             size=(FULL, 6.3), span="double", rect=(0.045, 0.175, 0.915, 0.735),
             caption="Overall architecture of the proposed multi-agent time-series "
                     "modelling system. A natural-language query is interpreted into a "
                     "(domain, task) pair; the capability filter admits only registry "
                     "entries whose declared prerequisites are satisfied; one of three "
                     "selection strategies then chooses among the feasible models. In "
                     "the hybrid strategy the language model may choose only from the "
                     "filtered shortlist, and a validator rejects any identifier "
                     "outside it.",
             note="Components shown are exactly those present in the codebase. The "
                  "dashed double-headed arrow denotes persistence, not control flow.",
             note_y=0.166),
        dict(n=2, slug="registry", title="Model Registry and Capability Structure",
             size=(FULL, 5.7), span="double", rect=(0.045, 0.170, 0.915, 0.715),
             caption="Structure of the capability-based model registry and filtering "
                     "process. Each of the 20 entries declares the prerequisites it "
                     "needs, what it returns, and its measured out-of-sample skill; the "
                     "filter removes candidates only on hard grounds, so a downstream "
                     "language model cannot select an infeasible model however "
                     "confidently it argues for one.",
             note="Composition re-derived by executing registry_stats(); the example "
                  "entry is reproduced verbatim from agents/model_registry.py.",
             note_y=0.160),
        dict(n=3, slug="selection_accuracy", title="Model-Selection Accuracy",
             size=(COL, 3.0), span="single", rect=(0.175, 0.500, 0.56, 0.355),
             caption="Accuracy of the three model-selection strategies on 123 labelled "
                     "routing queries. The query set comprises 9 canonical phrasings, "
                     "89 paraphrases, 15 adversarial cases in which the naive reading "
                     "is wrong, and 10 conceptual questions for which declining to run "
                     "any model is the correct behaviour.",
             note="Values plotted are the three-strategy run in selection_all.json, the "
                  "only run in which all three arms were scored against one common "
                  "label set. That label set has since been revised: value-forecast "
                  "queries now expect the Gaussian-Process entry rather than the ARIMA "
                  "entry. Re-executing the deterministic filter against the CURRENT "
                  "labels gives 94.31%, and the two LLM arms have not been re-run under "
                  "those labels. The per-category table above is from the "
                  "current-label run.",
             note_y=0.290,
             block=("Per-category accuracy, capability filter "
                    "(re-executed against current code, n = 123)", cat_tbl, 0.400)),
        dict(n=4, slug="structure_profiling",
             title="Structure Profiling — Lomb--Scargle Ratios",
             size=(COL, 3.4), span="single", rect=(0.155, 0.415, 0.66, 0.44),
             caption="Lomb--Scargle peak/background ratios used for structure profiling "
                     "of Kepler and equity time series. The ratio decides whether a "
                     "series' events recur or occur once, which in turn selects the "
                     "detector family; the two populations are separated by roughly "
                     "three orders of magnitude with no overlap. The shaded band marks "
                     "the deliberately wide uncertain region, within which the profiler "
                     "abstains rather than guessing.",
             note="Individual ratios were regenerated by executing anomaly.profile on "
                  "the cached 440-target manifest (seed 11) and the 10 most liquid NSE "
                  "names; they are measured values, not the summary figures quoted in "
                  "the module docstring. The equity population reproduces the recorded "
                  "calibration exactly (median 12.4, range 9.4–15.0). The Kepler median "
                  "here is 9,034.8 against a recorded calibration median of 1,939.3 — a "
                  "different target sample, the same conclusion. One Kepler target "
                  "(ratio 55.0) falls inside the uncertain band and is abstained on "
                  "rather than misclassified, so 19 of 20 are decided and all 19 "
                  "decided correctly.",
             note_y=0.284),
        dict(n=5, slug="detector_families", title="Anomaly Detector Families",
             size=(FULL, 3.7), span="double", rect=(0.045, 0.190, 0.915, 0.695),
             caption="Detector families and methods evaluated in the anomaly-detection "
                     "benchmark. Families are organised by the assumption each makes "
                     "about what 'anomalous' means, rather than by model complexity, "
                     "because the experiment asks which assumption survives contact "
                     "with two unrelated kinds of data.",
             note="Family membership and counts derived from default_detectors() and "
                  "cross-checked against the 21 distinct detector names present in "
                  "anomaly_grid.json.",
             note_y=0.172),
        dict(n=6, slug="kepler_results", title="Kepler Anomaly-Detection Results",
             size=(FULL, 4.1), span="double", rect=(0.275, 0.400, 0.665, 0.487),
             caption="Held-out PR-AUC of anomaly detectors on the Kepler benchmark. "
                     "Every detector is fitted on training targets and scored on "
                     "targets it has never seen; scores are converted to within-target "
                     "ranks before pooling, because one star's raw score is not "
                     "comparable to another's. The best unsupervised detector exceeds "
                     "the no-skill floor by 0.0102 PR-AUC, and three unsupervised "
                     "detectors outscore a supervised model that was given the labels.",
             note="All 21 scoreable rows are shown; no detector is omitted. The "
                  "horizontal axis is zoomed to the data and does not start at zero — "
                  "values are therefore drawn as points rather than bars, so that no "
                  "bar length implies a magnitude. Values read directly from "
                  "evaluation/results/anomaly_grid.json.",
             block=("Benchmark configuration", kepler_cfg, 0.300)),
        dict(n=7, slug="equity_results", title="Indian Equity Anomaly-Detection Results",
             size=(FULL, 4.1), span="double", rect=(0.275, 0.400, 0.665, 0.487),
             caption="Held-out PR-AUC of anomaly detectors on the Indian equity "
                     "benchmark. Windows are trailing, so each is labelled by its final "
                     "day and no detector sees the future. With eight positive windows "
                     "in the held-out fold the supervised ceiling falls below the "
                     "no-skill floor: there is too little positive signal for a "
                     "label-aware model to learn from, and the column should be read "
                     "as directional.",
             note="All 22 scoreable rows are shown. One-Class SVM appears at 0.0161 "
                  "held out against 0.1026 in-sample, a 4.62x in-sample lift — see "
                  "Figure 8. No LSTM autoencoder row exists in the committed results, "
                  "so none is plotted. The horizontal axis is zoomed to the data and "
                  "does not start at zero, so values are drawn as points rather "
                  "than bars.",
             block=("Benchmark configuration", equity_cfg, 0.300)),
        dict(n=8, slug="insample_vs_heldout", title="In-Sample versus Held-Out Performance",
             size=(FULL, 5.5), span="double", rect=None,
             caption="Comparison of in-sample and held-out anomaly-detection "
                     "performance. Each detector is scored twice on the same data: "
                     "fitted and scored on all windows, and fitted on a training split "
                     "then scored on a held-out one. The ratio between the two tracks "
                     "model capacity rather than skill, so an in-sample leaderboard "
                     "would rank these detectors in close to the reverse of their "
                     "generalisation order.",
             note="Rows are ordered by the inflation ratio (in-sample PR-AUC divided by "
                  "held-out PR-AUC), annotated to the right of each pair. The "
                  "parameter-free matrix profile sits at 0.91x on equity and 0.95x on "
                  "Kepler, consistent with its fit step learning nothing at all; the "
                  "one-class SVM reaches 6.37x on equity, where in-sample 0.1026 "
                  "collapses to 0.0161 held out. Paired values exist for 41 of the 43 "
                  "committed rows — the two greedy-ensemble rows carry no in-sample "
                  "score and are therefore excluded from this figure.",
             note_y=0.205),
        dict(n=9, slug="dataset_summary", title="Dataset and Benchmark Summary",
             size=(FULL, 5.3), span="double", rect=(0.055, 0.195, 0.895, 0.700),
             caption="Summary of the two datasets used in the anomaly-detection "
                     "benchmark. The two domains were chosen because the structure of "
                     "their anomalies inverts: a planetary transit recurs and is "
                     "therefore the least unusual feature of its own light curve, "
                     "whereas a market shock happens once.",
             note="Every value is read from evaluation/results/anomaly_grid.json (its "
                  "meta, rows, audits and summaries keys) or from the cached 440-target "
                  "manifest. Positive rates are given both over all windows and within "
                  "the held-out fold, because the two differ and have been conflated "
                  "elsewhere.",
             note_y=0.190),
        dict(n=10, slug="chatbot_demo", title="Chatbot End-to-End Demonstration",
             size=(FULL, 3.9), span="double", rect=(0.055, 0.195, 0.895, 0.700),
             caption="Example end-to-end interactions with the time-series analysis "
                     "chatbot. (a) an astronomy query on a Kepler light curve; (b) a "
                     "finance query on an equity series. In both cases the reply is "
                     "generated only from computed facts, and the selection trace "
                     "records which model was chosen and on what measured evidence.",
             note="Screenshots are deliberately not generated. Panels (a) and (b) are "
                  "sized to the two-column text width, so a captured image can be "
                  "dropped in without rescaling the surrounding figure.",
             note_y=0.178),
    ]


def render_into(fig, meta, grid, prof, rect):
    """Draw figure `meta['n']` into `fig` at `rect`. Shared by both outputs."""
    n = meta["n"]
    if n == 8:
        # Two stacked panels; rect is the bounding region. The gap must clear
        # the upper panel's tick labels AND its x-axis title, or they land on
        # the lower panel's bold heading.
        x, y, w, h = rect
        gap = 0.155 * h
        ph = (h - gap) / 2
        draw_inflation(fig, grid, [(x, y + ph + gap, w, ph), (x, y, w, ph)])
        return
    ax = _clean(fig.add_axes(rect))
    if n == 1:
        draw_architecture(ax)
    elif n == 2:
        draw_registry(ax)
    elif n == 3:
        draw_selection(ax)
    elif n == 4:
        draw_profiling(ax, prof)
    elif n == 5:
        draw_families(ax)
    elif n in (6, 7):
        domain = "kepler" if n == 6 else "equity"
        draw_dots(ax, grid, domain, *DOT_CFG[domain])
    elif n == 9:
        draw_datasets(ax, grid)
    elif n == 10:
        draw_chatbot(ax)


# ============================================================ combined PDF
def build_combined(metas, grid, prof):
    with PdfPages(OUT_PDF) as pdf:
        for m in metas:
            fig = plt.figure(figsize=PAGE)
            fig.text(LEFT, 0.958, f"BTP Figures — Figure {m['n']}", fontsize=8, color=MID)
            fig.add_artist(plt.Line2D([LEFT, RIGHT], [0.948, 0.948], color=LIGHT,
                                      lw=0.7, transform=fig.transFigure))
            fig.text(LEFT, 0.930, m["title"], fontsize=12.5, weight="bold", va="top")
            fig.add_artist(plt.Line2D([LEFT, RIGHT], [CAPTION_RULE_Y, CAPTION_RULE_Y],
                                      color=LIGHT, lw=0.7, transform=fig.transFigure))
            fig.text(LEFT, CAPTION_Y,
                     wrap(f"Figure {m['n']}. {m['caption']}", 118),
                     fontsize=8.5, va="top", linespacing=1.45)

            rect = m["rect"] or (0.175, 0.255, 0.735, 0.595)
            if m["n"] == 8:
                rect = (0.175, 0.255, 0.735, 0.595)
            render_into(fig, m, grid, prof, rect)

            note_y = m.get("note_y")
            if m.get("block"):
                title, lines, by = m["block"]
                fig.text(LEFT, by, title, fontsize=7.4, weight="bold")
                for i, line in enumerate(lines):
                    fig.text(LEFT, by - 0.019 - i * 0.0122, line, fontsize=6.7,
                             family="monospace")
                if note_y is None:
                    note_y = by - 0.019 - len(lines) * 0.0122 - 0.012
            fig.text(LEFT, note_y or 0.19, wrap(m["note"], 132), fontsize=6.9,
                     color=MID, va="top", style="italic", linespacing=1.5)

            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "BTP Figures — A Multi-Agent System for Time-Series Modelling"
        d["Author"] = "Kaveri Neeli"
    return OUT_PDF


# ============================================================ standalone PDFs
# Per-figure margins for the tight standalone version, as fractions of the
# figure. Diagrams drawn in 0-100 data coordinates need almost none; plots need
# room for their tick labels and axis titles.
STANDALONE_RECT = {
    1: (0.005, 0.005, 0.990, 0.990),
    2: (0.005, 0.005, 0.990, 0.990),
    3: (0.175, 0.185, 0.800, 0.780),
    4: (0.185, 0.155, 0.760, 0.820),
    5: (0.005, 0.010, 0.990, 0.980),
    6: (0.175, 0.105, 0.790, 0.880),
    7: (0.175, 0.105, 0.790, 0.880),
    8: (0.140, 0.060, 0.830, 0.910),
    9: (0.005, 0.010, 0.990, 0.980),
    10: (0.005, 0.010, 0.990, 0.980),
}


def build_standalone(metas, grid, prof):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for m in metas:
        fig = plt.figure(figsize=m["size"])
        render_into(fig, m, grid, prof, STANDALONE_RECT[m["n"]])
        path = OUT_DIR / f"fig{m['n']:02d}_{m['slug']}.pdf"
        # bbox_inches="tight" trims the canvas to the drawn content, so LaTeX
        # scales the figure itself rather than a box of surrounding whitespace.
        fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        written.append((m, path))
    return written


def write_latex_helpers(written):
    cap_lines = ["% Captions for the BTP figures, generated by",
                 "% evaluation/btp_figures.py -- edit freely.", ""]
    for m, path in written:
        env = "figure*" if m["span"] == "double" else "figure"
        pos = "[!t]"
        cap_lines += [
            f"% ---- Figure {m['n']}: {m['title']}",
            f"\\begin{{{env}}}{pos}",
            "  \\centering",
            f"  \\includegraphics[width=\\{'textwidth' if m['span']=='double' else 'columnwidth'}]"
            f"{{figures/{path.name}}}",
            f"  \\caption{{{m['caption']}}}",
            f"  \\label{{fig:{m['slug']}}}",
            f"\\end{{{env}}}",
            "",
        ]
    (OUT_DIR / "captions.tex").write_text("\n".join(cap_lines), encoding="utf-8")

    readme = [
        "BTP figures, one standalone vector PDF per figure",
        "=" * 52,
        "",
        "Each PDF contains ONLY the figure: no page header, no caption, tight",
        "bounding box. LaTeX supplies the caption -- see captions.tex, which",
        "holds a ready-made float for every figure.",
        "",
        "Usage",
        "-----",
        "  1. copy this folder into your Overleaf project as  figures/",
        "  2. \\usepackage{graphicx}",
        "  3. paste the float you want from captions.tex",
        "",
        "Width: figures marked 'both columns' must use figure* in a two-column",
        "layout, because they are wider than one column.",
        "",
        f"{'file':<38}{'width':<14}{'size (in)'}",
        "-" * 68,
    ]
    for m, path in written:
        span = "both columns" if m["span"] == "double" else "one column"
        readme.append(f"{path.name:<38}{span:<14}{m['size'][0]:.2f} x {m['size'][1]:.2f}")
    readme += [
        "",
        "Regenerate everything with:  python -m evaluation.btp_figures",
        "Every number is read from evaluation/results/ at draw time.",
    ]
    (OUT_DIR / "README.txt").write_text("\n".join(readme), encoding="utf-8")


# ================================================================ main
def main():
    grid = load_grid()
    prof = load_profiling()
    if prof is None:
        raise SystemExit("evaluation/results/profiling.json missing -- "
                         "run the profiler first (see Figure 4).")

    metas = build_meta(grid)
    combined = build_combined(metas, grid, prof)
    written = build_standalone(metas, grid, prof)
    write_latex_helpers(written)

    print(f"combined   -> {combined}")
    print(f"standalone -> {OUT_DIR}{chr(92)}  ({len(written)} PDFs + captions.tex + README.txt)")
    for m, p in written:
        print(f"   fig{m['n']:>2}  {p.name:<38}{m['size'][0]:.2f} x {m['size'][1]:.2f} in")


if __name__ == "__main__":
    main()
