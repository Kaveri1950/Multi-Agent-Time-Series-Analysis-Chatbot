"""Extract a light curve (flux vs. index) from a plot image (JPG/PNG), so
the chatbot can analyze a light curve a user only has as a picture (e.g. a
screenshot from a paper or an exported chart) rather than raw data.

Honest scope limitation, stated up front: this recovers the RELATIVE SHAPE
of the plotted curve (normalized flux, baseline ~1.0), not absolute
physical time/flux units. Recovering true units would require OCR-ing the
axis tick labels and calibrating pixel-to-data coordinates -- a much
larger, error-prone computer-vision project on its own, out of scope here.
For anomaly detection this is sufficient: astronomy/anomaly_detector.py
and astronomy/ts_embedder.py both operate on flux *shape* (dips,
variability), not on true calendar time.

For best results the image should be reasonably tightly cropped to the
plot's data area (minimal surrounding whitespace/labels/legend), with the
curve visually distinct from the background (a single line/scatter color
against a light background is the assumed common case).

Tested against real Kepler data and synthetic curves during development.
Works well when the feature of interest (a dip, a clear anomaly) is
visually prominent in the plot -- the realistic case, since that's usually
*why* someone would screenshot and ask about a specific light curve. It is
markedly less reliable for: (a) very subtle signals near the noise floor
(a full-cadence real light curve where the interesting dip is only ~1-2%
of a visually near-flat plot can have the automatic column-tracing lock
onto the wrong pixel for a stretch and throw off a strict point-by-point
comparison, even though it still recovers the general dip location in
practice), and (b) very high-density plots packing far more raw data
points than the image has pixel columns (a plotted multi-thousand-point
raw light curve compresses many points per column; a paper figure or a
binned/downsampled plot is a much better match for what a per-column
pixel-tracing approach can resolve). This is a fundamental limit of
inferring a curve from pixels alone, not something a better heuristic
fully escapes -- true plot digitizers (e.g. WebPlotDigitizer) rely on
manual axis calibration and often manual correction for exactly this
reason.
"""

import io

import numpy as np
import pandas as pd
from PIL import Image


def _load_image_array(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def _autocrop_to_plot(arr: np.ndarray) -> np.ndarray:
    """Real-world uploads are often a whole webpage/lightbox screenshot
    (dark surrounding background, nav arrows, a site banner, an image-info
    overlay), not just the plot -- but everything below assumes the image
    IS the plot, and column-tracing derails onto whatever unrelated bright
    or dark clutter surrounds it instead of the actual curve. This finds
    the largest contiguous bright (near-white) region -- matplotlib's
    default figure background -- and crops down to it first.

    No-op (returns the array unchanged) if nothing confidently plot-shaped
    is found, or if the brightest region already covers most of the image
    (i.e. the input already IS just the plot) -- so an already-tightly-
    cropped image is unaffected, only genuinely cluttered screenshots get
    cropped."""
    from scipy import ndimage

    gray = arr.mean(axis=2)
    height, width = gray.shape
    bright = gray > 200  # matplotlib's default white/near-white figure background

    if bright.mean() < 0.15:
        return arr  # nothing plot-like found -- leave the image alone rather than guess

    labeled, n_components = ndimage.label(bright)
    if n_components == 0:
        return arr

    sizes = ndimage.sum(bright, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    component_mask = labeled == largest_label
    rows, cols = np.any(component_mask, axis=1), np.any(component_mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    crop_area_fraction = (bottom - top) * (right - left) / (height * width)
    if crop_area_fraction > 0.9:
        return arr  # already basically the whole image -- nothing meaningful to crop

    pad = 2
    top, left = max(0, top - pad), max(0, left - pad)
    bottom, right = min(height, bottom + pad), min(width, right + pad)
    return arr[top:bottom, left:right]


def extract_lightcurve_from_image(image_bytes: bytes, n_points: int = 400) -> pd.DataFrame:
    """Returns a DataFrame with columns ['time', 'flux']. `time` is a plain
    0..n_points-1 index (no physical units are recoverable from a plot
    image alone). `flux` is normalized so the curve's baseline (its
    brightest/plateau level, i.e. "out of transit") is close to 1.0 and
    dips read as values below it, matching the convention used elsewhere
    in this app (astronomy/data_loader.py's synthetic/real light curves).

    Raises ValueError if no plausible curve can be detected (e.g. a blank
    or all-background image)."""
    arr = _load_image_array(image_bytes)
    arr = _autocrop_to_plot(arr)
    gray = arr.mean(axis=2)
    height, width = gray.shape

    # Background = the most common pixel intensity (typically white/light
    # for a standard plot). "Data" pixels are whatever deviates meaningfully
    # from it -- this deliberately ignores faint gridlines/anti-aliasing via
    # the threshold below, while still catching a bold plotted line/points.
    hist, bin_edges = np.histogram(gray, bins=64)
    background_level = bin_edges[np.argmax(hist)]
    deviation = np.abs(gray - background_level)
    threshold = max(15.0, deviation.max() * 0.15)
    active = deviation > threshold

    # Axis border spines/frames are a near-continuous straight line spanning
    # almost the full width (horizontal spines) or height (vertical spines)
    # -- categorically different from a data curve, which only occupies one
    # row per column and varies across columns. Detected and zeroed out
    # before the per-column scan below; otherwise a border pixel gets
    # included in every column's weighted average and drags the whole
    # extracted curve toward a near-flat line (observed in testing: this
    # alone dropped round-trip correlation on a synthetic test curve from a
    # clean extraction to ~0.12).
    row_coverage = active.mean(axis=1)
    col_coverage = active.mean(axis=0)
    border_rows = row_coverage > 0.9
    border_cols = col_coverage > 0.9
    if border_rows.any():
        deviation[border_rows, :] = 0
    if border_cols.any():
        deviation[:, border_cols] = 0
    active = deviation > threshold

    # Track the curve left-to-right using continuity, not a per-column
    # weighted average of every active pixel. Reason (found in testing): a
    # connecting line segment between two data points with a steep local
    # slope (e.g. the sharp edge of a transit dip) renders as a near-
    # vertical stroke spanning many rows within a SINGLE pixel column.
    # Averaging that whole stroke doesn't represent either data point --
    # it lands near the stroke's midpoint, which varies erratically column
    # to column and destroyed the extracted signal's correlation with the
    # source curve (~0.12-0.2 in testing). Picking the active pixel closest
    # to the previous column's chosen position correctly resolves a steep
    # stroke to whichever endpoint continues the curve.
    y_positions = np.full(width, np.nan)
    prev_y = None
    for x in range(width):
        mask = active[:, x]
        if not mask.any():
            continue
        ys = np.nonzero(mask)[0]
        if prev_y is None:
            y_positions[x] = np.average(ys, weights=deviation[ys, x])
        else:
            y_positions[x] = ys[np.argmin(np.abs(ys - prev_y))]
        prev_y = y_positions[x]

    valid = ~np.isnan(y_positions)
    if valid.sum() < width * 0.3:
        raise ValueError(
            "Could not detect a plotted curve in this image (too little contrast against the "
            "background, or the plot area is mostly empty). Try a more tightly cropped, "
            "higher-contrast image of just the light curve."
        )

    x_idx = np.arange(width)
    y_positions = np.interp(x_idx, x_idx[valid], y_positions[valid])

    # Continuity-tracking occasionally locks onto the wrong pixel for a
    # column or two near a sharp reversal (found in testing: dip minima
    # were the main place this happened), producing a single-column
    # outlier. A real plotted curve has no genuine single-pixel-wide
    # spikes, so a small median filter cleans this up without flattening
    # actual features (which span many columns).
    half_win = 2
    padded = np.pad(y_positions, half_win, mode="edge")
    y_positions = np.array([
        np.median(padded[i:i + 2 * half_win + 1]) for i in range(width)
    ])

    # Image rows increase downward; flip so larger = brighter/higher flux.
    y_flux_proxy = -y_positions

    # Baseline = a high percentile of the proxy (the plateau/"out of
    # transit" level), assuming dips are the minority of points -- true for
    # typical transit-style light curves. Normalize the full observed range
    # to [~0, 1] rather than forcing a fixed depth scale, so the image's own
    # real relative dip depth is preserved instead of invented.
    baseline = np.percentile(y_flux_proxy, 90)
    floor = y_flux_proxy.min()
    span = baseline - floor
    if span <= 1e-6:
        raise ValueError("Detected curve has almost no variation -- check the image crop/contrast.")

    flux = (y_flux_proxy - floor) / span

    if width != n_points:
        src_idx = np.linspace(0, width - 1, width)
        dst_idx = np.linspace(0, width - 1, n_points)
        flux = np.interp(dst_idx, src_idx, flux)

    return pd.DataFrame({"time": np.arange(n_points, dtype=float), "flux": flux})


def _render_test_curve_to_png(time, flux) -> bytes:
    """Test helper: render a flux series to a clean PNG (no axis spines/ticks),
    matching what a well-cropped user upload should look like."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(time, flux, color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


if __name__ == "__main__":
    # A visually prominent, moderate-density curve -- representative of the
    # realistic case (someone screenshots a plot because a feature in it is
    # clearly visible), not a raw multi-thousand-point cadence dump.
    t = np.linspace(0, 100, 800)
    flux = 1.0 - 0.05 * np.exp(-0.5 * ((t - 30) / 2) ** 2) - 0.03 * np.exp(-0.5 * ((t - 70) / 3) ** 2)

    image_bytes = _render_test_curve_to_png(t, flux)
    extracted = extract_lightcurve_from_image(image_bytes, n_points=200)
    original_resampled = np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, len(flux)), flux)
    correlation = np.corrcoef(extracted["flux"].values, original_resampled)[0, 1]
    print(f"Round-trip correlation (clear-feature test curve): {correlation:.4f}")
    print(extracted.head())
