"""Vector PDF figures for the 4-page two-column BTP report.

Every number is read out of a committed result file. Nothing is typed in.
Each figure asserts the expected values supplied in the figure brief; a
mismatch raises rather than silently drawing a different number.

Run:  python -m evaluation.make_report_figures [fig01] [fig02] [fig04] ...
      (no arguments runs every implemented figure)

FIGURE 3 IS NOT IMPLEMENTED. Its panel (a) needs per-series Lomb-Scargle
ratios for 10 Kepler targets, and the two sources in the repository
disagree: anomaly/profile.py's docstring states median 1,939.3 over range
74.1-49,974.7 (the value the brief expects and the report cites), while
evaluation/results/profiling.json -- the only file holding the 10 individual
values a strip plot needs -- gives median 9,034.77 over range
55.02-107,440.53. The equity group agrees exactly between the two
(median 12.40, range 9.41-15.02), so the disagreement is Kepler-specific.
See check_fig03_sources() below, which prints the evidence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "evaluation" / "results"

# ----------------------------------------------------------------- style
C_BEST = "#1f5fa8"        # best detector / key element
C_SUP = "#2B5F8A"         # supervised reference
C_GREY = "#9AA3AF"        # ordinary detectors
C_RAW = "#4A5568"         # raw-input baseline
C_CHANCE = "#6B7482"      # chance line
C_GRID = "#E5E7EB"        # gridlines
C_KEY_FILL = "#16213A"    # key box fill (figure 1)
C_PANEL = "#F3F4F6"       # background panel (figure 1)
C_EDGE = "#9AA3AF"        # thin box borders

FS_LABEL = 7              # labels and ticks
FS_SMALL = 6              # small annotations
FS_TICK_Y = 6.5           # detector-name ticks

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_LABEL,
    "ytick.labelsize": FS_LABEL,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "hatch.linewidth": 0.45,
    "pdf.fonttype": 42,      # embed TrueType, keep text as text
    "ps.fonttype": 42,
})

SAVE = dict(bbox_inches="tight", pad_inches=0.02)

#: Set to a directory to also drop a high-dpi PNG of each figure there, for
#: visual inspection. The committed deliverable is always the vector PDF.
PREVIEW_DIR = os.environ.get("FIGURE_PREVIEW_DIR")


def save_figure(fig, stem: str) -> Path:
    out = ROOT / "docs" / "figures" / f"{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, **SAVE)
    if PREVIEW_DIR:
        d = Path(PREVIEW_DIR)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=260, **SAVE)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ data
def grid_rows(domain: str) -> list[dict]:
    d = json.loads((RESULTS / "anomaly_grid.json").read_text())
    return [r for r in d["rows"] if r["domain"] == domain]


def lstm_row() -> dict:
    return json.loads((RESULTS / "lstm_addendum.json").read_text())["row"]


def pretty(name: str) -> str:
    """Detector id -> reader-facing label, per the figure brief."""
    special = {
        "supervised_ceiling": "supervised reference",
        "raw_input": "raw-input baseline",
        "matrix_profile_discord_raw": "matrix profile discord (raw)",
        "lstm_autoencoder": "LSTM autoencoder (scored separately)",
    }
    return special.get(name, name.replace("_", " "))


def check(label: str, got, want, tol=1e-9):
    if abs(float(got) - float(want)) > tol:
        raise SystemExit(
            f"CHECK FAILED: {label}\n  computed {got!r}\n  expected {want!r}\n"
            f"  Stopping rather than drawing a value that differs from the brief."
        )


def panel_label(ax, text, x=0.0, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=FS_LABEL, fontweight="bold", color="#111827")


# ================================================================ FIGURE 1
def _measure(fig, text, fs, mono=False):
    """Rendered size of a string, in inches.

    Estimates are not reliable here -- DejaVu Sans is wider than a naive
    0.5-em guess, and a guessed width clipped the last box on the first pass.
    """
    t = fig.text(0.5, 0.5, text, fontsize=fs,
                 family="monospace" if mono else "sans-serif")
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def fig01():
    """Architecture diagram, laid out in inches from measured text widths."""
    W, H = 7.1, 2.35
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    PAD = 0.13
    MARGIN = 0.05

    flow = [
        dict(t="User\nquestion", s="Streamlit\nchat"),
        dict(t="Domain + task", s="4 domains ×\n10 tasks ·\npattern rules"),
        dict(t="Capability\nfilter", s="feasible(domain,\ntask, available)",
             mono=True, key=True),
        dict(t="Feasible\nshortlist", s=None),
        None,
        dict(t="Analytical\nmodule", s="anomaly · astronomy\nfinance · solar"),
        dict(t="Grounded\nanswer", s="facts + reasoning\ntrace"),
    ]
    minis = [
        ("filter_only: best\nmeasured skill,\nno LLM", C_BEST),
        ("hybrid: LLM ranks\nshortlist, validator\nrejects off-list", "#111827"),
        ("llm_only: LLM picks\nfrom all 20\n(no filter)", "#111827"),
    ]

    widths = []
    for b in flow:
        if b is None:
            mw = max(_measure(fig, m, FS_SMALL)[0] for m, _ in minis)
            widths.append(mw + 4 * 0.055)
        else:
            tw = _measure(fig, b["t"], FS_LABEL)[0]
            sw = _measure(fig, b["s"], FS_SMALL, b.get("mono", False))[0] if b["s"] else 0
            widths.append(max(tw, sw) + PAD)

    gap = (W - 2 * MARGIN - sum(widths)) / (len(flow) - 1)
    if gap < 0.04:
        raise SystemExit(
            "figure 1 does not fit: boxes total %.2f in of %.1f in, leaving "
            "%.3f in per gap. Shorten or wrap a label." % (sum(widths), W, gap))

    mini_h = _measure(fig, minis[0][0], FS_SMALL)[1] + 0.075
    h5 = 0.17 + 3 * mini_h + 2 * 0.032 + 0.055
    h1 = 0.56
    yc = 1.50
    yb = yc - h1 / 2
    y5 = yc - h5 / 2

    def rbox(x, y, w, h, face="white", ec=C_EDGE, lw=0.7, z=3, r=0.045):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=%s" % r,
            linewidth=lw, edgecolor=ec, facecolor=face, zorder=z))

    def arrow(p0, p1, dashed=False):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=5.5, linewidth=0.7,
            color="#6B7280", zorder=6, shrinkA=0, shrinkB=0,
            linestyle=(0, (2.2, 1.8)) if dashed else "solid"))

    xs, x = [], MARGIN
    for w in widths:
        xs.append(x)
        x += w + gap

    geom = {}
    for i, (b, x0, w) in enumerate(zip(flow, xs, widths)):
        if b is None:
            geom["sel"] = (x0, y5, w, h5)
            continue
        key = b.get("key", False)
        rbox(x0, yb, w, h1, face=C_KEY_FILL if key else "white")
        tcol = "white" if key else "#111827"
        scol = "#D7DDE8" if key else "#4B5563"
        if b["s"]:
            ax.text(x0 + w / 2, yb + h1 * 0.72, b["t"], ha="center", va="center",
                    fontsize=FS_LABEL, color=tcol, zorder=4, linespacing=1.15)
            ax.text(x0 + w / 2, yb + h1 * 0.28, b["s"], ha="center", va="center",
                    fontsize=FS_SMALL, color=scol, zorder=4, linespacing=1.2,
                    family="monospace" if b.get("mono") else "sans-serif")
        else:
            ax.text(x0 + w / 2, yb + h1 / 2, b["t"], ha="center", va="center",
                    fontsize=FS_LABEL, color=tcol, zorder=4, linespacing=1.15)
        geom[i] = (x0, yb, w, h1)

    sx, sy, sw, sh = geom["sel"]
    rbox(sx, sy, sw, sh)
    ax.text(sx + sw / 2, sy + sh - 0.09, "Selection", ha="center", va="center",
            fontsize=FS_LABEL, color="#111827", zorder=4)
    mini_geom = []
    top = sy + sh - 0.17
    for i, (txt, col) in enumerate(minis):
        my = top - (i + 1) * mini_h - i * 0.032
        rbox(sx + 0.055, my, sw - 0.11, mini_h, face="#FAFBFC",
             ec="#D1D5DB", lw=0.55, z=4, r=0.03)
        ax.text(sx + sw / 2, my + mini_h / 2, txt, ha="center", va="center",
                fontsize=FS_SMALL, color=col, zorder=5, linespacing=1.18)
        mini_geom.append((sx + 0.055, my, sw - 0.11, mini_h))

    px0 = geom[1][0] - 0.07
    px1 = sx + sw + 0.07
    py0, py1 = min(sy, yb) - 0.10, 2.28
    ax.add_patch(FancyBboxPatch(
        (px0, py0), px1 - px0, py1 - py0,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=0, facecolor=C_PANEL, zorder=1))
    ax.text(px0 + 0.07, py1 - 0.10, "LangGraph StateGraph", ha="left",
            va="center", fontsize=FS_SMALL, style="italic", color="#6B7280",
            zorder=2)

    for a, b in ((0, 1), (1, 2), (2, 3)):
        arrow((geom[a][0] + geom[a][2], yc), (geom[b][0], yc))
    arrow((geom[3][0] + geom[3][2], yc), (sx, yc))
    arrow((sx + sw, yc), (geom[5][0], yc))
    arrow((geom[5][0] + geom[5][2], yc), (geom[6][0], yc))

    b3x, _, b3w, _ = geom[2]
    rw = min(b3w, 1.05)
    rx = b3x + (b3w - rw) / 2
    rbox(rx, 0.14, rw, 0.40)
    ax.text(rx + rw / 2, 0.14 + 0.40 * 0.66, "Model registry", ha="center",
            va="center", fontsize=FS_LABEL, color="#111827", zorder=4)
    ax.text(rx + rw / 2, 0.14 + 0.40 * 0.26, "20 specs", ha="center",
            va="center", fontsize=FS_SMALL, color="#4B5563", zorder=4)
    arrow((rx + rw / 2, 0.54), (b3x + b3w / 2, yb))

    mx, my, mw, mh_ = mini_geom[2]
    run_y = my + mh_ / 2
    bx = geom[1][0] + geom[1][2] / 2
    ax.plot([bx, bx], [yb, run_y], color="#8A9099", lw=0.7,
            linestyle=(0, (2.2, 1.8)), zorder=5)
    ax.plot([bx, mx - 0.03], [run_y, run_y], color="#8A9099", lw=0.7,
            linestyle=(0, (2.2, 1.8)), zorder=5)
    arrow((mx - 0.03, run_y), (mx, run_y), dashed=True)
    ax.text(geom[3][0] + geom[3][2] / 2, run_y - 0.115, "bypasses the filter",
            ha="center", va="center", fontsize=FS_SMALL, style="italic",
            color="#8A9099", zorder=6)

    b6x, _, b6w, _ = geom[5]
    cw = _measure(fig, "MongoDB (optional)", FS_SMALL)[0] + 0.34
    cx = min(b6x + b6w / 2 - cw / 2, W - MARGIN - cw)
    cy, ch = 0.18, 0.30
    ax.add_patch(Ellipse((cx + cw / 2, cy + ch), cw, 0.12, linewidth=0.7,
                         edgecolor=C_EDGE, facecolor="white", zorder=4))
    ax.add_patch(Rectangle((cx, cy), cw, ch, linewidth=0, facecolor="white",
                           zorder=3))
    ax.add_patch(Ellipse((cx + cw / 2, cy), cw, 0.12, linewidth=0.7,
                         edgecolor=C_EDGE, facecolor="white", zorder=3))
    ax.plot([cx, cx], [cy, cy + ch], color=C_EDGE, lw=0.7, zorder=4)
    ax.plot([cx + cw, cx + cw], [cy, cy + ch], color=C_EDGE, lw=0.7, zorder=4)
    ax.text(cx + cw / 2, cy + ch * 0.48, "Parquet cache ·\nMongoDB (optional)",
            ha="center", va="center", fontsize=FS_SMALL, color="#4B5563",
            zorder=5, linespacing=1.2)
    ax.add_patch(FancyArrowPatch(
        (cx + cw / 2, cy + ch + 0.06), (b6x + b6w / 2, yb),
        arrowstyle="-", linewidth=0.7, color="#9CA3AF", zorder=2,
        linestyle=(0, (2.2, 1.8)), shrinkA=0, shrinkB=0))

    return save_figure(fig, "fig01_architecture")


# ================================================================ FIGURE 2
def fig02():
    W, H = 7.1, 2.65
    fig, axes = plt.subplots(1, 2, figsize=(W, H))

    kep = grid_rows("kepler")
    eq = grid_rows("equity")
    lstm = lstm_row()

    # --- checks against the brief
    kb = max((r for r in kep if not r.get("is_ceiling")), key=lambda r: r["pr_auc"])
    check("Kepler best = matrix_profile_discord_raw", kb["pr_auc"], 0.2630)
    if kb["detector"] != "matrix_profile_discord_raw":
        raise SystemExit(f"CHECK FAILED: Kepler best is {kb['detector']}")
    check("Kepler supervised reference",
          next(r for r in kep if r.get("is_ceiling"))["pr_auc"], 0.2555)
    check("Kepler LSTM", lstm["pr_auc"], 0.2591)
    eb = max((r for r in eq if not r.get("is_ceiling")), key=lambda r: r["pr_auc"])
    check("equity best = spectral_pca", eb["pr_auc"], 0.0355)
    if eb["detector"] != "spectral_pca":
        raise SystemExit(f"CHECK FAILED: equity best is {eb['detector']}")
    check("equity supervised reference",
          next(r for r in eq if r.get("is_ceiling"))["pr_auc"], 0.0172)

    kep_raw = next(r for r in kep if r["detector"] == "raw_input")["pr_auc"]
    eq_raw = next(r for r in eq if r["detector"] == "raw_input")["pr_auc"]
    check("Kepler raw-input baseline", kep_raw, 0.2528)
    check("equity raw-input baseline", eq_raw, 0.0325)
    kep_ch = round(kep[0]["prevalence"], 4)
    eq_ch = round(eq[0]["prevalence"], 4)
    check("Kepler chance (held-out positive rate)", kep_ch, 0.2338, tol=5e-5)
    check("equity chance (held-out positive rate)", eq_ch, 0.0141, tol=5e-5)

    # The y-tick labels are long ("LSTM autoencoder (scored separately)"), and
    # with fixed margins they spilled past the canvas -- bbox_inches="tight"
    # then grew the saved page to 7.84 in, 10% wider than the column. Measure
    # them and size the margins so the tight box lands back on the figsize.
    w_a = max(_measure(fig, pretty(r["detector"]), FS_TICK_Y)[0]
              for r in kep + [lstm])
    w_b = max(_measure(fig, pretty(r["detector"]), FS_TICK_Y)[0] for r in eq)
    left_in, right_pad = w_a + 0.06, 0.05
    gap_in = w_b + 0.10
    axes_w = (W - left_in - gap_in - right_pad) / 2
    if axes_w < 1.0:
        raise SystemExit("figure 2 panels would be under 1 in wide")
    fig.subplots_adjust(left=left_in / W, right=1 - right_pad / W,
                        wspace=gap_in / axes_w, top=0.875, bottom=0.135)

    panels = [
        ("(a)", axes[0], kep + [lstm], kep_raw, kep_ch, (0.21, 0.27), "%.4f",
         "matrix_profile_discord_raw"),
        ("(b)", axes[1], eq, eq_raw, eq_ch, (0.00, 0.040), "%.4f", "spectral_pca"),
    ]

    for tag, ax, rows, raw_v, chance_v, xlim, vfmt, best_id in panels:
        rows = sorted(rows, key=lambda r: r["pr_auc"])   # bottom-to-top ascending
        names = [pretty(r["detector"]) for r in rows]
        vals = [r["pr_auc"] for r in rows]
        ypos = np.arange(len(rows))

        for i, r in enumerate(rows):
            if r["detector"] == "lstm_autoencoder":
                kw = dict(color="#EDF0F3", edgecolor=C_GREY, hatch="///", linewidth=0.5)
            elif r["detector"] == best_id:
                kw = dict(color=C_BEST, linewidth=0)
            elif r.get("is_ceiling"):
                kw = dict(color=C_SUP, linewidth=0)
            elif r["detector"] == "raw_input":
                kw = dict(color=C_RAW, linewidth=0)
            else:
                kw = dict(color=C_GREY, linewidth=0)
            ax.barh(i, r["pr_auc"], height=0.72, zorder=3, **kw)

        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=FS_TICK_Y)
        ax.set_ylim(-0.75, len(rows) - 0.25)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Held-out PR-AUC", fontsize=FS_LABEL)

        for i, v in enumerate(vals):
            # white pad: several values sit on top of a reference line
            ax.text(v + (xlim[1] - xlim[0]) * 0.012, i, vfmt % v,
                    va="center", ha="left", fontsize=FS_SMALL, color="#374151",
                    zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.6, alpha=0.85))

        ax.axvline(raw_v, color=C_RAW, linestyle=(0, (1.2, 1.4)), linewidth=0.8, zorder=2)
        ax.axvline(chance_v, color=C_CHANCE, linestyle=(0, (3.2, 1.8)), linewidth=0.8, zorder=2)
        ax.text(raw_v, 1.075, f"raw-input {vfmt % raw_v}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=FS_SMALL, color=C_RAW)
        ax.text(chance_v, 1.012, f"chance {vfmt % chance_v}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=FS_SMALL, color=C_CHANCE)

        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.tick_params(axis="y", length=0, pad=1.5)
        panel_label(ax, tag, x=0.0, y=1.135)

    return save_figure(fig, "fig02_heldout_prauc")


# ================================================================ FIGURE 3
def fig03():
    """(a) Lomb-Scargle periodicity strip plot, (b) BLS periodogram.

    Panel (a) is drawn from evaluation/results/profiling.json. Those ten
    Kepler ratios were re-profiled with the current anomaly/profile.py and
    reproduce to the stored decimal, so this is the calibration the code
    actually produces today. anomaly/profile.py's docstring still quotes an
    older sample (median 1,939.3, range 74.1-49,974.7) that no committed file
    backs and that the current code does not reproduce; the report text
    repeating it needs updating to the values drawn here.
    """
    W, H = 3.3, 1.9
    fig, axes = plt.subplots(
        1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [0.45, 0.55]})
    fig.subplots_adjust(left=0.155, right=0.985, top=0.90, bottom=0.215,
                        wspace=0.62)

    from anomaly.profile import ONE_OFF_MAX, PERIODIC_MIN

    prof = json.loads((RESULTS / "profiling.json").read_text())
    groups = [("Equity", [r["ratio"] for r in prof["equity"]]),
              ("Kepler", [r["ratio"] for r in prof["kepler"]])]

    # ---------------------------------------------------------- panel (a)
    ax = axes[0]
    ax.axhspan(ONE_OFF_MAX, PERIODIC_MIN, color=C_GRID, zorder=1)
    ax.text(0.5, (ONE_OFF_MAX * PERIODIC_MIN) ** 0.5, "uncertain",
            ha="center", va="center", fontsize=FS_SMALL - 0.5,
            color="#6B7280", zorder=2)

    rng = np.random.default_rng(0)
    for i, (name, vals) in enumerate(groups):
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=7,
                   color=C_GREY if name == "Equity" else C_RAW,
                   zorder=4, linewidths=0)
        med = float(np.median(vals))
        ax.plot([i - 0.28, i + 0.28], [med, med], color=C_BEST, lw=1.3, zorder=5)
        txt = "%,.1f".replace("%,.1f", "{:,.1f}").format(med)
        ax.annotate(txt, (i + 0.30, med), textcoords="offset points",
                    xytext=(1, 0), fontsize=FS_SMALL, color=C_BEST,
                    ha="left", va="center", zorder=6)

    lo = min(min(v) for _, v in groups)
    hi = max(max(v) for _, v in groups)
    ax.set_yscale("log")
    ax.set_ylim(lo / 3, hi * 4)
    ax.set_xlim(-0.55, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=FS_LABEL)
    ax.set_ylabel("Peak / median power", fontsize=FS_LABEL, labelpad=1.5)
    ax.tick_params(axis="both", labelsize=FS_LABEL, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9CA3AF")
    panel_label(ax, "(a)", x=-0.42, y=1.02)

    # ---------------------------------------------------------- panel (b)
    ax = axes[1]
    d = np.load(RESULTS / "bls_example.npz", allow_pickle=True)
    period, power = d["pg_period"], d["pg_power"]
    best, truth = float(d["period"]), float(d["true_period"])
    kepid = int(d["kepid"])

    ax.plot(period, power, color=C_GREY, lw=0.45, zorder=3)
    peak = float(power[int(np.argmin(np.abs(period - best)))])
    ax.scatter([best], [peak], s=13, color=C_BEST, zorder=5, linewidths=0)
    ax.annotate("%.4f d\n(catalogue %.4f d)" % (best, truth),
                (best, peak), textcoords="offset points", xytext=(-2, 5),
                fontsize=FS_SMALL, ha="right", va="bottom", color=C_BEST,
                linespacing=1.15, zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Trial period (days)", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylabel("BLS power", fontsize=FS_LABEL, labelpad=1.5)
    ax.set_ylim(0, peak * 1.42)
    ax.tick_params(axis="both", labelsize=FS_LABEL, pad=1.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FS_SMALL)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#9CA3AF")
    panel_label(ax, "(b)", x=-0.30, y=1.02)
    axes[1].set_xlabel("Trial period (days), KIC %d" % kepid,
                       fontsize=FS_LABEL, labelpad=1.5)

    return save_figure(fig, "fig03_structure_bls")


# ================================================================ FIGURE 4
def fig04():
    W, H = 3.3, 1.9
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.subplots_adjust(wspace=0.52, left=0.135, right=0.985, top=0.90, bottom=0.21)

    kep = [r for r in grid_rows("kepler") if r.get("group") != "ensemble"]
    eq = [r for r in grid_rows("equity") if r.get("group") != "ensemble"]
    lstm = lstm_row()

    if len(eq) != 21:
        raise SystemExit(f"CHECK FAILED: equity detectors = {len(eq)}, expected 21")
    if len(kep) != 20:
        raise SystemExit(f"CHECK FAILED: Kepler detectors = {len(kep)}, expected 20")
    check("LSTM fit+score seconds", lstm["seconds"], 908.75)
    mv = next(r for r in kep if r["detector"] == "moving_variance")
    check("Kepler moving variance PR-AUC", mv["pr_auc"], 0.2595)
    if not (3.0 <= mv["seconds"] <= 4.0):
        raise SystemExit(f"CHECK FAILED: moving variance seconds {mv['seconds']}, expected ~3.5")

    for tag, ax, rows, extra in (("(a)", axes[0], eq, None),
                                 ("(b)", axes[1], kep, lstm)):
        raw = next(r for r in rows if r["detector"] == "raw_input")
        sup = next(r for r in rows if r.get("is_ceiling"))
        best = max((r for r in rows if not r.get("is_ceiling")),
                   key=lambda r: r["pr_auc"])

        for r in rows:
            if r is best:
                col, z, s = C_BEST, 5, 11
            elif r is sup:
                col, z, s = C_SUP, 5, 11
            elif r is raw:
                col, z, s = C_RAW, 5, 11
            else:
                col, z, s = C_GREY, 3, 8
            ax.scatter(r["seconds"], r["pr_auc"], s=s, color=col, zorder=z,
                       linewidths=0)

        if extra is not None:
            ax.scatter(extra["seconds"], extra["pr_auc"], s=13, facecolors="none",
                       edgecolors=C_BEST, linewidths=0.8, zorder=6)

        ax.axhline(raw["pr_auc"], color=C_RAW, linestyle=(0, (1.2, 1.4)),
                   linewidth=0.7, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Fit + score time (s)", fontsize=FS_LABEL, labelpad=1.5)
        ax.set_ylabel("Held-out PR-AUC", fontsize=FS_LABEL, labelpad=1.5)
        ax.tick_params(axis="both", labelsize=FS_LABEL, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9CA3AF")
        panel_label(ax, tag, x=-0.30, y=1.02)

        def lab(r, text, dx, dy, ha="left", va="bottom", col="#374151"):
            ax.annotate(text, (r["seconds"], r["pr_auc"]),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=FS_SMALL, ha=ha, va=va, color=col,
                        linespacing=1.15, zorder=7)

        # The raw-input floor is a LINE across the panel, so it is labelled at
        # the panel edge rather than at its marker -- labelling the marker put
        # the text on top of neighbouring points.
        edge = ax.get_yaxis_transform()
        if tag == "(a)":
            ax.set_ylim(0.011, 0.041)
            lab(best, "spectral PCA", 2, 4, col=C_BEST)
            lab(sup, "supervised\nreference", 6, -3, ha="right", va="top", col=C_SUP)
            ax.text(0.985, raw["pr_auc"] + 0.0004, "raw-input baseline",
                    transform=edge, ha="right", va="bottom",
                    fontsize=FS_SMALL, color=C_RAW, zorder=7)
        else:
            ax.set_ylim(0.218, 0.2755)
            lab(best, "matrix profile\ndiscord (raw)", -4, 3, ha="right", col=C_BEST)
            lab(sup, "supervised\nreference", 6, -3, ha="right", va="top", col=C_SUP)
            lab(mv, "moving\nvariance", 5, 3)
            lab(extra, "LSTM", -4, 0, ha="right", va="center", col=C_BEST)
            ax.text(0.03, raw["pr_auc"] + 0.0006, "raw-input baseline",
                    transform=edge, ha="left", va="bottom",
                    fontsize=FS_SMALL, color=C_RAW, zorder=7)

    return save_figure(fig, "fig04_performance_vs_cost")


# ============================================== FIGURE 3 source conflict
def check_fig03_sources():
    """Print the evidence for why figure 3 panel (a) is blocked."""
    import statistics as st
    p = RESULTS / "profiling.json"
    if not p.exists():
        print("profiling.json missing entirely")
        return
    d = json.loads(p.read_text())
    print("evaluation/results/profiling.json  (UNTRACKED -- not committed)")
    for g in ("equity", "kepler"):
        v = sorted(r["ratio"] for r in d[g])
        print(f"  {g:7} n={len(v)}  min {min(v):,.2f}  max {max(v):,.2f} "
              f" median {st.median(v):,.2f}")
    print("\nanomaly/profile.py docstring (COMMITTED), lines 24-25:")
    print("  one-off (equities)    median    12.4    range  9.4 - 15.0")
    print("  periodic (Kepler)     median 1,939.3    range 74.1 - 49,974.7")
    print("\nEquity agrees. Kepler does not. The brief expects the docstring "
          "values,\nbut only profiling.json carries the 10 per-series values a "
          "strip plot needs.")


FIGURES = {"fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for name in wanted:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; have {list(FIGURES)}")
        path = FIGURES[name]()
        print(f"wrote {path.name}  ({path.stat().st_size:,} bytes)")
    print()
    check_fig03_sources()
