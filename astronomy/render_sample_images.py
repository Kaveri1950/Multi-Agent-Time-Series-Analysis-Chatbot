"""Renders real, already-fetched light curves (data/cache/lightcurves/*.parquet,
built by astronomy/dataset.py from the free public lightkurve/MAST pipeline)
into plot images -- a local, legitimately-sourced set of light-curve
"photos" to test the Astronomy page's image-upload path
(dashboard.sections._load_light_curve -> astronomy.image_lightcurve)
against. Deliberately NOT scraped from the open web: light-curve images
found via generic image search would carry unclear copyright/licensing and
no reliable way to confirm they're genuine, unlabeled light curves rather
than an arbitrary plot -- rendering our own from data we already fetched
under NASA's public archive terms sidesteps both problems, and gives a
known ground truth (the source parquet) to check extraction accuracy against.

Two styles per target, on purpose:
- "clean": no axis/spines/labels -- the easy case, matching
  astronomy/image_lightcurve.py's own stated best-case assumptions.
- "annotated": axis labels, title, gridlines -- closer to a real screenshot
  a user might actually upload, and a more honest test of the extractor's
  documented limitations (see that module's docstring) than only ever
  feeding it the easy case.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astronomy.dataset import DEFAULT_TARGETS, load_cached_light_curve

IMAGE_DIR = Path(__file__).parent.parent / "data" / "cache" / "lightcurve_images"
BIN_POINTS = 300  # astronomy/image_lightcurve.py's own docstring: a raw multi-thousand-point cadence
                   # dump packs far more data than a plot has pixel columns for -- a binned/downsampled
                   # curve (standard practice for light curve plots anyway) is a better match for what
                   # the pixel-tracing extractor can resolve than plotting every raw point.


def _safe_name(target: str) -> str:
    return target.replace(" ", "_").replace("/", "_")


def _bin_light_curve(lc: pd.DataFrame, n_bins: int = BIN_POINTS) -> pd.DataFrame:
    if len(lc) <= n_bins:
        return lc
    bin_edges = np.linspace(0, len(lc), n_bins + 1).astype(int)
    times = [lc["time"].values[a:b].mean() for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    fluxes = [lc["flux"].values[a:b].mean() for a, b in zip(bin_edges[:-1], bin_edges[1:])]
    return pd.DataFrame({"time": times, "flux": fluxes})


def render_target_image(target: str, style: str = "clean", out_dir: Path = IMAGE_DIR) -> Path:
    lc = _bin_light_curve(load_cached_light_curve(target))  # raises FileNotFoundError if not cached -- see astronomy/dataset.py

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_name(target)}_{style}.png"

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(lc["time"], lc["flux"], color="black", linewidth=1.2)

    if style == "clean":
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{target} — Kepler light curve")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    # bbox_inches="tight" matters here, not just cosmetics: without it, the
    # saved image keeps whatever surrounding margin matplotlib left, and the
    # extractor scans the FULL image width as if it were all data -- padding
    # columns stretch/misalign the pixel-to-data-index mapping against the
    # real underlying series (astronomy/image_lightcurve.py's own test
    # helper, _render_test_curve_to_png, already does this for the same
    # reason). Verified in testing this omission alone was enough to sink
    # round-trip correlation on a real target to ~0.11 -- but note this
    # fixes only that specific failure mode: a genuinely subtle/noisy
    # underlying signal (near-flat flux, dominated by scatter rather than a
    # visually prominent feature) still extracts poorly regardless, per
    # this module's own documented limitations.
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(targets: list[str] | None = None, styles: tuple[str, ...] = ("clean", "annotated")) -> list[Path]:
    targets = targets or DEFAULT_TARGETS
    paths = []
    for target in targets:
        for style in styles:
            try:
                path = render_target_image(target, style=style)
                paths.append(path)
                print(f"[render_sample_images] {target} ({style}): saved -> {path.name}")
            except FileNotFoundError as exc:
                print(f"[render_sample_images] {target}: skipped ({exc})")
    return paths


def render_synthetic_demo_images(out_dir: Path = IMAGE_DIR) -> list[Path]:
    """A couple of synthetic curves with isolated, smooth, noise-free
    dips -- deliberately the SAME curve shape astronomy/image_lightcurve.py's
    own test suite already validates at 0.66 round-trip correlation
    (test_image_lightcurve_extraction), not astronomy/data_loader.py's
    generate_synthetic_light_curve (periodic dips + noise + stellar
    variability), which was tried here first and measured at 0.29/-0.00 --
    a real, verified finding that the pixel-tracing extractor handles
    isolated well-separated features far better than periodic/noisy ones.
    Real Kepler targets are real light curves either way (see
    render_target_image above); this function exists only to guarantee at
    least one upload example that reliably extracts well, for a first try."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (depth1, depth2) in enumerate([(0.05, 0.03), (0.08, 0.05)]):
        t = np.linspace(0, 100, 800)
        flux = (1.0 - depth1 * np.exp(-0.5 * ((t - 30) / 2) ** 2)
                - depth2 * np.exp(-0.5 * ((t - 70) / 3) ** 2))
        out_path = out_dir / f"synthetic_demo_{i}_depths{depth1}_{depth2}.png"

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(t, flux, color="black", linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        paths.append(out_path)
        print(f"[render_sample_images] synthetic demo (depths={depth1}/{depth2}): saved -> {out_path.name}")
    return paths


if __name__ == "__main__":
    paths = render_all()
    paths += render_synthetic_demo_images()
    print(f"\nRendered {len(paths)} image(s) to {IMAGE_DIR}")
