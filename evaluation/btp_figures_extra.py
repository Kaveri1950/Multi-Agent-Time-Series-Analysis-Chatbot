"""Figures 11 and 12 for the BTP report, each as its own standalone PDF.

Shares the style of evaluation/btp_figures.py so the whole figure set is
typographically consistent. Each figure contains ONLY the plot -- no page
header, no caption, tight bounding box -- for \\includegraphics in Overleaf.

  fig011_bls_period_recovery.pdf   real BLS run on a cached Kepler target
  fig012_performance_vs_cost.pdf   measured runtime vs held-out PR-AUC

Every value is read from evaluation/results/ at draw time.

Run:  python -m evaluation.btp_figures_extra
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.btp_figures import (BEST, CEIL, COL, FILL, FLOOR, FULL, INK,
                                    LIGHT, MID, OUT_DIR, RESULTS, load_grid)

BLS_NPZ = RESULTS / "bls_example.npz"


# ================================================================ FIGURE 11
def fig11_bls():
    """Two panels: the phase-folded transit, and the BLS periodogram.

    Both come from one real search (astronomy.transit_search.search) on one
    cached target, so the periodogram peak and the fold are the same fit.
    """
    if not BLS_NPZ.exists():
        raise SystemExit(f"{BLS_NPZ} missing -- run the BLS example first.")
    d = np.load(BLS_NPZ, allow_pickle=True)

    kepid = int(d["kepid"])
    time, flux = d["time"], d["flux"]
    period, epoch = float(d["period"]), float(d["epoch"])
    duration, depth = float(d["duration"]), float(d["depth"])
    sde, true_period = float(d["sde"]), float(d["true_period"])
    err_pct, n_periods = float(d["err_pct"]), int(d["n_periods"])
    baseline = float(d["baseline"])
    true_depth_ppm = float(d["true_depth_ppm"])
    pg_p, pg_pow = d["pg_period"], d["pg_power"]

    n_transits = int(baseline // period)

    fig = plt.figure(figsize=(FULL, 3.05))
    axa = fig.add_axes([0.070, 0.165, 0.395, 0.760])
    axb = fig.add_axes([0.575, 0.165, 0.400, 0.760])
    for ax in (axa, axb):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- panel (a): phase-folded light curve ----------------
    phase = ((time - epoch + 0.5 * period) % period) - 0.5 * period
    phase_h = phase * 24.0                                  # hours from mid-transit
    keep = np.abs(phase_h) <= 9.0
    ph, fl = phase_h[keep], flux[keep]

    axa.scatter(ph, fl, s=0.6, color=LIGHT, alpha=0.45, linewidths=0, zorder=1,
                rasterized=True)

    # binned mean -- the stacked transit, which is the point of the method
    nb = 90
    edges = np.linspace(-9.0, 9.0, nb + 1)
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
    cent = 0.5 * (edges[:-1] + edges[1:])
    binned = np.array([fl[idx == i].mean() if (idx == i).any() else np.nan
                       for i in range(nb)])
    axa.plot(cent, binned, color=INK, lw=1.1, zorder=3)
    axa.scatter(cent, binned, s=7, color=INK, zorder=4)

    dur_h = duration * 24.0
    axa.axvspan(-dur_h / 2, dur_h / 2, color=CEIL, alpha=0.10, zorder=0, lw=0)
    axa.axvline(0, color=CEIL, lw=0.6, ls=":", zorder=2)

    # The y-axis MUST be scaled to the transit, not to the raw scatter. This
    # dip is ~230 ppm while the per-cadence noise spans ~2%, so an axis fitted
    # to the points makes the signal invisible -- the individual cadences are
    # deliberately allowed to clip.
    lo, hi = float(np.nanmin(binned)), float(np.nanmax(binned))
    dip = 1.0 - lo
    axa.set_ylim(lo - dip * 0.55, max(hi, 1.0) + dip * 0.90)

    axa.annotate("transit signal",
                 xy=(0, lo), xytext=(3.2, lo - dip * 0.30),
                 fontsize=6.8, color=CEIL, va="center",
                 arrowprops=dict(arrowstyle="-|>", color=CEIL, lw=0.8,
                                 mutation_scale=7,
                                 connectionstyle="arc3,rad=-0.25"))
    # Lower left. At the top it ran into the legend; the transit-signal
    # annotation occupies the lower right.
    axa.text(-8.6, lo - dip * 0.46,
             f"{n_transits} transits stacked\ndepth {depth * 1e6:.0f} ppm "
             f"(catalogue {true_depth_ppm:.0f} ppm)",
             fontsize=6.2, color=MID, va="bottom", linespacing=1.4)

    axa.set_xlabel("Phase from mid-transit (hours)", labelpad=3)
    axa.set_ylabel("Normalised flux")
    axa.set_xlim(-9, 9)
    axa.set_title(f"(a)  Phase-folded light curve, KIC {kepid}",
                  fontsize=8.0, weight="bold", loc="left", pad=5)
    axa.grid(color=LIGHT, lw=0.35, alpha=0.5)
    axa.set_axisbelow(True)
    axa.ticklabel_format(axis="y", useOffset=False, style="plain")

    handles = [plt.Line2D([], [], ls="", marker="o", mfc=LIGHT, mec=LIGHT, ms=3.4),
               plt.Line2D([], [], color=INK, lw=1.1, marker="o", ms=2.8)]
    # upper right: above the out-of-transit baseline, so it covers no data
    axa.legend(handles, ["individual cadences (clipped)", "binned mean"],
               loc="upper right", fontsize=6.0, handlelength=1.3, labelspacing=0.3,
               borderaxespad=0.4, frameon=False)

    # ---------------- panel (b): BLS periodogram -------------------------
    axb.plot(pg_p, pg_pow, color=MID, lw=0.45, zorder=2)
    peak = float(np.nanmax(pg_pow))
    axb.scatter([period], [peak], s=34, facecolor=BEST, edgecolor=INK,
                linewidth=0.7, zorder=5)

    # xytext in AXES coordinates: in data coordinates the label ran past the
    # right-hand limit, and bbox_inches="tight" then widened this panel alone,
    # so the two panels no longer matched in height.
    axb.annotate(f"recovered period\n{period:.5f} d",
                 xy=(period, peak), xycoords="data",
                 xytext=(0.965, 0.58), textcoords="axes fraction",
                 fontsize=6.8, color=INK, va="center", ha="right",
                 arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                 mutation_scale=7,
                                 connectionstyle="arc3,rad=-0.25"))

    axb.set_xscale("log")
    axb.set_xlabel("Trial orbital period (days, log scale)", labelpad=3)
    axb.set_ylabel("BLS power")
    axb.set_xlim(pg_p.min(), pg_p.max())
    axb.set_title("(b)  BLS periodogram", fontsize=8.0, weight="bold",
                  loc="left", pad=5)
    axb.grid(color=LIGHT, lw=0.35, alpha=0.5)
    axb.set_axisbelow(True)

    # Upper-left: the periodogram's low-power short-period region, so the box
    # covers no data. At lower-right it sat on top of the curve.
    lines = [f"catalogue period   {true_period:.5f} d",
             f"period error       {err_pct:.4f}%",
             f"detection signif.  SDE {sde:.1f}",
             f"trial periods      {n_periods:,} at 0.02% spacing"]
    axb.text(0.025, 0.965, "\n".join(lines), transform=axb.transAxes,
             fontsize=6.1, family="monospace", color=INK, ha="left", va="top",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F7F5",
                       edgecolor=LIGHT, linewidth=0.6))
    axb.set_ylim(0, peak * 1.38)

    path = OUT_DIR / "fig011_bls_period_recovery.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path, dict(kepid=kepid, period=period, true_period=true_period,
                      err_pct=err_pct, sde=sde, n_transits=n_transits,
                      depth_ppm=depth * 1e6, true_depth_ppm=true_depth_ppm)


# ================================================================ FIGURE 12
# Detectors worth naming on the scatter. Labelling all 21 per panel is
# unreadable; the full ranking is in Figures 6 and 7.
LABEL_EQUITY = {"spectral_pca", "raw_input", "matrix_profile_discord",
                "matrix_profile_motif", "supervised_ceiling", "one_class_svm",
                "pca_autoencoder", "hampel_mad"}
LABEL_KEPLER = {"matrix_profile_discord_raw", "latent_isoforest", "moving_variance",
                "raw_input", "supervised_ceiling", "matrix_profile_motif_raw",
                "local_outlier_factor", "pca_autoencoder"}

# Manual (dx, dy) offsets in points, to keep labels off their own markers and
# off each other. Chosen by inspection of the rendered figure.
OFFSETS = {
    # equity panel
    # raw_input and hampel_mad score identically and sit almost on top of each
    # other, so their labels are stacked rather than side by side.
    "spectral_pca": (6, 5), "raw_input": (8, -10), "hampel_mad": (8, 3),
    "matrix_profile_discord": (-7, -11), "matrix_profile_motif": (-8, 7),
    "supervised_ceiling": (8, -4), "one_class_svm": (-7, -10),
    "pca_autoencoder": (2, 7),
    # kepler panel
    "matrix_profile_discord_raw": (-8, 9), "latent_isoforest": (8, -3),
    "moving_variance": (5, 6), "matrix_profile_motif_raw": (-8, -11),
    "local_outlier_factor": (-6, -11),
}

# Per-domain offsets that differ from the shared table above, because the two
# panels place the same detector in very different neighbourhoods.
OFFSETS_KEPLER = {
    # both are near the left axis, so their labels must extend rightward or
    # they are clipped by the panel edge
    "raw_input": (8, -10), "pca_autoencoder": (7, -4),
    "supervised_ceiling": (-7, -11),
}

# Where the no-skill reference label sits, per panel, as (x in axes fraction,
# ha, va). On equity the right-hand end of the line is clear. On Kepler the
# supervised ceiling sits at the right-hand end at almost the same height, so
# the label goes to the left instead.
FLOOR_LABEL = {"equity": (0.985, "right", "bottom"),
               "kepler": (0.015, "left", "bottom")}


def fig12_cost():
    grid = load_grid()
    fig = plt.figure(figsize=(FULL, 3.25))
    axes = [fig.add_axes([0.070, 0.175, 0.395, 0.745]),
            fig.add_axes([0.575, 0.175, 0.400, 0.745])]

    for ax, (domain, label, labelset) in zip(
            axes, [("equity", "(a)  Indian equity", LABEL_EQUITY),
                   ("kepler", "(b)  Kepler", LABEL_KEPLER)]):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        rows = [r for r in grid["rows"]
                if r["domain"] == domain and r.get("group") != "ensemble"
                and r.get("seconds") and r["seconds"] > 0
                and r.get("pr_auc") is not None and r["pr_auc"] == r["pr_auc"]]

        floor_val = max(r["pr_auc"] for r in rows if r.get("is_floor"))
        best_val = max(r["pr_auc"] for r in rows
                       if not r.get("is_floor") and not r.get("is_ceiling"))

        ax.axhline(floor_val, color=FLOOR, lw=1.0, ls="--", zorder=1)
        fx, fha, fva = FLOOR_LABEL[domain]
        pad = " " if fha == "left" else ""
        ax.text(fx, floor_val, f"{pad}no-skill floor {floor_val:.4f}{pad}",
                transform=ax.get_yaxis_transform(), fontsize=6.2, color=FLOOR,
                va=fva, ha=fha)

        for r in rows:
            if r.get("is_floor"):
                fc, mk, sz, z = FLOOR, "s", 34, 4
            elif r.get("is_ceiling"):
                fc, mk, sz, z = CEIL, "D", 28, 4
            elif r["pr_auc"] == best_val:
                fc, mk, sz, z = BEST, "o", 44, 5
            else:
                fc, mk, sz, z = FILL, "o", 28, 3
            ax.scatter([r["seconds"]], [r["pr_auc"]], s=sz, marker=mk,
                       facecolor=fc, edgecolor=INK, linewidth=0.6, zorder=z)
            if r["detector"] in labelset:
                tbl = OFFSETS_KEPLER if domain == "kepler" else {}
                dx, dy = tbl.get(r["detector"], OFFSETS.get(r["detector"], (6, 5)))
                ax.annotate(r["detector"], (r["seconds"], r["pr_auc"]),
                            textcoords="offset points", xytext=(dx, dy),
                            fontsize=5.6, family="monospace", color=INK,
                            ha="left" if dx > 0 else "right", zorder=6)

        secs = [r["seconds"] for r in rows]
        ratio = max(secs) / min(secs)
        ax.set_xscale("log")
        ax.set_xlabel("Computational cost — measured fit + score time (s, log scale)",
                      labelpad=3, fontsize=7.6)
        ax.set_ylabel("Held-out PR-AUC")
        # The cost span goes in the title: as a corner box it landed on a
        # labelled point on the Kepler panel.
        ax.set_title(f"{label}   —   {len(rows)} detectors, cost spans "
                     f"{ratio:,.0f}$\\times$",
                     fontsize=8.0, weight="bold", loc="left", pad=5)
        ax.grid(color=LIGHT, lw=0.35, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(min(secs) * 0.35, max(secs) * 4.5)

        ymin = min(r["pr_auc"] for r in rows)
        ymax = max(r["pr_auc"] for r in rows)
        pad = (ymax - ymin) * 0.20
        ax.set_ylim(ymin - pad, ymax + pad * 1.35)

    handles = [plt.Line2D([], [], ls="", marker="o", mfc=BEST, mec=INK, ms=5.4),
               plt.Line2D([], [], ls="", marker="s", mfc=FLOOR, mec=INK, ms=4.8),
               plt.Line2D([], [], ls="", marker="D", mfc=CEIL, mec=INK, ms=4.4),
               plt.Line2D([], [], ls="", marker="o", mfc=FILL, mec=INK, ms=4.8)]
    fig.legend(handles, ["best unsupervised detector", "no-skill floor (learns nothing)",
                         "supervised ceiling (sees labels)", "other detectors"],
               loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=4,
               fontsize=6.3, handlelength=1.0, columnspacing=1.6,
               handletextpad=0.4, frameon=False)

    path = OUT_DIR / "fig012_performance_vs_cost.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p11, info = fig11_bls()
    p12 = fig12_cost()
    print(f"wrote {p11}")
    print(f"      KIC {info['kepid']}: recovered {info['period']:.5f} d vs catalogue "
          f"{info['true_period']:.5f} d, error {info['err_pct']:.4f}%, SDE {info['sde']:.1f}")
    print(f"wrote {p12}")


if __name__ == "__main__":
    main()
