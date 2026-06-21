"""Boulder detection (shadow method) and density mapping from optical imagery.

Implemented by: **terrain agent**.

References
----------
* Pajola, M. et al. (2017); Watkins, R. N. et al. (2019) — the shadow-length
  method for boulder sizing: a boulder of height ``h`` casts a shadow of length
  ``L = h / tan(sun_elev)`` on level ground, so ``h = L * tan(sun_elev)``.
* Bickel, V. T. et al. (2020), "Automated detection of lunar rockfalls", *IEEE
  TGRS* — connected-component shadow detection / size-frequency mapping.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

__all__ = ["detect_boulders_shadow", "boulder_density_map"]


def _shadow_threshold(img: np.ndarray, dark_frac: float) -> float | None:
    """Intensity below which a pixel counts as shadow, or ``None`` if no shadow.

    Uses Otsu's bimodal split between the sunlit and shadowed populations
    (Bickel et al. 2020), but only accepts it when the image actually has
    shadow-like contrast: the candidate dark population must be both a small
    minority (``< 0.5``) and genuinely darker than the bright mode. A uniform /
    contrast-free image returns ``None`` (no boulders).
    """
    finite = img[np.isfinite(img)]
    if finite.size == 0 or np.ptp(finite) <= 0:
        return None
    try:
        thr = float(threshold_otsu(finite))
    except Exception:
        return None
    dark = finite <= thr
    frac = float(dark.mean())
    if frac <= 0.0 or frac >= 0.5:
        # not a clear minority shadow population
        return None
    # require the dark mode to sit clearly below the bright mode
    bright_med = float(np.median(finite[finite > thr]))
    dark_med = float(np.median(finite[dark]))
    if dark_med >= bright_med - 1e-9:
        return None
    return thr


def detect_boulders_shadow(
    img: np.ndarray,
    sun_elev_deg: float,
    gsd: float,
    min_area_px: int = 2,
    dark_frac: float = 0.5,
) -> np.ndarray:
    """Detect boulders from their cast shadows in an optical image.

    Shadows are the darkest pixels in the scene; an Otsu bimodal threshold
    (guarded against contrast-free images, see :func:`_shadow_threshold`)
    isolates the shadow population. Connected dark blobs are labelled
    (:func:`skimage.measure.label`, 8-connectivity) and, for each, the shadow
    length ``L`` is taken as the larger bounding-box extent (rows vs columns),
    which is exact for an axis-aligned cast shadow and robust to the
    ellipse-fit overshoot of a moment-based axis length. The boulder height
    then follows from level-ground shadow geometry (Pajola et al. 2017; Watkins
    et al. 2019):

        h = L * tan( radians(sun_elev_deg) )

    The boulder width is estimated from the smaller bounding-box extent.

    Parameters
    ----------
    img : np.ndarray
        Single-band optical image, shape ``(H, W)`` (higher = brighter).
    sun_elev_deg : float
        Solar elevation angle [deg].
    gsd : float
        Ground sample distance [m/px].
    min_area_px : int, default 2
        Minimum connected-component area [px] to accept as a boulder shadow.
    dark_frac : float, default 0.5
        Maximum fraction of the image allowed in the dark (shadow) population
        before the Otsu split is rejected as not shadow-like.

    Returns
    -------
    np.ndarray
        Detected boulders, shape ``(N, 4)`` = ``(row, col, width_m, height_m)``,
        one row per detection (empty ``(0, 4)`` array if none are found).
    """
    img = np.asarray(img, dtype=np.float64)
    if img.size == 0:
        return np.empty((0, 4), dtype=np.float64)

    thresh = _shadow_threshold(img, dark_frac)
    if thresh is None:
        return np.empty((0, 4), dtype=np.float64)

    shadow = img <= thresh
    labels = label(shadow, connectivity=2)
    tan_e = np.tan(np.radians(sun_elev_deg))

    rows = []
    for reg in regionprops(labels):
        if reg.area < min_area_px:
            continue
        minr, minc, maxr, maxc = reg.bbox
        ext_r = (maxr - minr)
        ext_c = (maxc - minc)
        # shadow length = longer bbox side; width = shorter side (in metres).
        L = float(max(ext_r, ext_c)) * gsd
        width = float(min(ext_r, ext_c)) * gsd
        if L <= 0.0:
            L = float(reg.equivalent_diameter) * gsd
        if width <= 0.0:
            width = gsd
        height = L * tan_e
        r, c = reg.centroid
        rows.append((float(r), float(c), float(width), float(height)))

    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def boulder_density_map(
    boulders: np.ndarray,
    shape: tuple[int, int],
    window: int,
    per_m2: bool = False,
    gsd: float = 1.0,
) -> np.ndarray:
    """Rasterise boulders into a spatial density (count per window) map.

    Each boulder centre is splatted onto an empty raster, then a boxcar moving
    sum over a ``window x window`` neighbourhood gives the local count
    (implemented as ``uniform_filter * window^2`` to obtain a sum from a mean).
    Optionally normalised to a number density per m^2.

    Parameters
    ----------
    boulders : np.ndarray
        Output of :func:`detect_boulders_shadow`, shape ``(N, >=2)`` with the
        first two columns ``(row, col)``.
    shape : tuple[int, int]
        Output raster shape ``(H, W)``.
    window : int
        Density kernel side length [px].
    per_m2 : bool, default False
        If True, divide counts by the window area in m^2 (using ``gsd``).
    gsd : float, default 1.0
        Ground sample distance [m/px] (only used when ``per_m2`` is True).

    Returns
    -------
    np.ndarray
        Boulder density [count/window] (or [count/m^2] if ``per_m2``), shape
        ``(H, W)``.
    """
    H, W = int(shape[0]), int(shape[1])
    counts = np.zeros((H, W), dtype=np.float64)
    boulders = np.asarray(boulders, dtype=np.float64)
    if boulders.ndim == 2 and boulders.shape[0] > 0:
        rr = np.clip(np.round(boulders[:, 0]).astype(int), 0, H - 1)
        cc = np.clip(np.round(boulders[:, 1]).astype(int), 0, W - 1)
        np.add.at(counts, (rr, cc), 1.0)

    win = max(int(window), 1)
    density = uniform_filter(counts, size=win, mode="constant") * (win * win)
    if per_m2:
        area_m2 = (win * gsd) ** 2
        density = density / area_m2
    return density
