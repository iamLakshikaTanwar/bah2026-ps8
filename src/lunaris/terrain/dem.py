"""DEM derivatives: slope, aspect, and multi-scale roughness.

Implemented by: **terrain agent**.

References
----------
* Horn, B. K. P. (1981), "Hill shading and the reflectance map",
  *Proc. IEEE* 69(1), 14-47 — the 3x3 weighted finite-difference gradient
  estimator used for slope and aspect.
* Rosenburg, M. A. et al. (2011), "Global surface slopes and roughness of the
  Moon from the Lunar Orbiter Laser Altimeter", *J. Geophys. Res.* 116, E02001
  — the deviogram / RMS-deviation roughness statistic
  ``nu(dx) = sqrt(<[z(x+dx) - z(x)]^2>)`` and its power-law (Hurst) scaling
  ``nu(dx) ~ dx^H`` used here for ``rms_roughness`` and ``hurst_exponent``.
* Kreslavsky, M. A. & Head, J. W. (2000), *J. Geophys. Res.* 105, 26695 —
  inter-quartile-range (median-based) roughness robust to boulder outliers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import correlate, percentile_filter, uniform_filter

__all__ = ["slope_horn", "aspect", "rms_roughness", "hurst_exponent", "iqr_roughness"]


# Horn (1981) 3x3 kernels. With pixels labelled
#     a b c
#     d e f
#     g h i
# Horn's gradients are
#     dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * res)
#     dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * res)
# ``scipy.ndimage.correlate`` slides the kernel as-is (no flip), so we encode
# the weights directly at the neighbour offsets. Row index increases downward
# (south), matching image convention; dz/dy is therefore the gradient toward
# increasing row (south).
_HORN_DZDX = np.array(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)
_HORN_DZDY = np.array(
    [[-1.0, -2.0, -1.0],
     [0.0, 0.0, 0.0],
     [1.0, 2.0, 1.0]]
)


def _horn_gradients(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Horn (1981) ``(dz/dx, dz/dy)`` finite-difference gradients.

    Edges use ``mode='nearest'`` (replicate the border row/column) so the
    estimator is defined everywhere without shrinking the array.
    """
    dem = np.asarray(dem, dtype=np.float64)
    denom = 8.0 * res
    dzdx = correlate(dem, _HORN_DZDX, mode="nearest") / denom
    dzdy = correlate(dem, _HORN_DZDY, mode="nearest") / denom
    return dzdx, dzdy


def slope_horn(dem: np.ndarray, res: float) -> np.ndarray:
    """Slope magnitude (degrees) via Horn's (1981) 3x3 method.

    With the 3x3 neighbourhood labelled ``a b c / d e f / g h i`` and pixel
    size ``res``::

        dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * res)
        dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * res)
        slope = degrees( arctan( hypot(dz/dx, dz/dy) ) )

    Implemented with :func:`scipy.ndimage.correlate` and Horn's weighting
    kernels; borders are handled with ``mode='nearest'`` (Horn 1981, eqs. for
    the reflectance-map gradient).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].

    Returns
    -------
    np.ndarray
        Slope [deg] in ``[0, 90)``, shape ``(H, W)``.
    """
    dzdx, dzdy = _horn_gradients(dem, res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))


def aspect(dem: np.ndarray, res: float) -> np.ndarray:
    """Aspect (downslope azimuth, degrees) via Horn's (1981) gradients.

        aspect = degrees( arctan2(dz/dy, -dz/dx) )  mod 360

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].

    Returns
    -------
    np.ndarray
        Aspect [deg] in ``[0, 360)``, shape ``(H, W)``.
    """
    dzdx, dzdy = _horn_gradients(dem, res)
    return np.mod(np.degrees(np.arctan2(dzdy, -dzdx)), 360.0)


def _deviogram_nu(dem: np.ndarray, lag_px: int) -> np.ndarray:
    """Per-pixel RMS deviation (deviogram amplitude) at a single baseline.

    Implements the Rosenburg et al. (2011) RMS-deviation statistic
    ``nu(dx) = sqrt(< [z(x + dx) - z(x)]^2 >)`` where the expectation is taken
    over the four axis-aligned directions (+x, -x, +y, -y) at lag ``dx = lag_px``
    pixels. The squared first differences are then smoothed over a window of
    side ``2*lag_px + 1`` to give a *local* RMS-deviation map rather than a
    single global scalar.
    """
    dem = np.asarray(dem, dtype=np.float64)
    lag = max(int(lag_px), 1)
    sq = np.zeros_like(dem)
    # four axis shifts; border replicated via np.roll then edge correction is
    # negligible at the small lags used here. Use shifted slices to avoid the
    # wrap-around bias of np.roll near the edges.
    for axis in (0, 1):
        fwd = np.empty_like(dem)
        # forward difference along +axis (replicate edge)
        sl_a = [slice(None), slice(None)]
        sl_b = [slice(None), slice(None)]
        sl_a[axis] = slice(0, dem.shape[axis] - lag)
        sl_b[axis] = slice(lag, dem.shape[axis])
        diff = dem[tuple(sl_b)] - dem[tuple(sl_a)]
        fwd[tuple(sl_a)] = diff
        # replicate the last valid difference into the trailing border band
        sl_fill = [slice(None), slice(None)]
        sl_fill[axis] = slice(dem.shape[axis] - lag, dem.shape[axis])
        sl_last = [slice(None), slice(None)]
        sl_last[axis] = slice(dem.shape[axis] - lag - 1, dem.shape[axis] - lag)
        fwd[tuple(sl_fill)] = fwd[tuple(sl_last)]
        sq += fwd ** 2
        sq += fwd ** 2  # symmetric: +axis and -axis contribute the same |diff|
    mean_sq = sq / 4.0
    win = 2 * lag + 1
    local = uniform_filter(mean_sq, size=win, mode="nearest")
    return np.sqrt(np.maximum(local, 0.0))


def rms_roughness(dem: np.ndarray, baseline_px: int) -> np.ndarray:
    """RMS-deviation roughness (deviogram amplitude) at a given baseline.

    Implements the Rosenburg et al. (2011) deviogram statistic

        nu(dx) = sqrt( mean_over_directions( [z(x + dx) - z(x)]^2 ) )

    evaluated at baseline ``dx = baseline_px`` pixels, averaged over the four
    axis-aligned directions and smoothed over a local ``(2*dx+1)`` window to
    yield a per-pixel roughness map (units of the DEM, i.e. metres). This is the
    self-affine roughness whose scale dependence the Hurst exponent captures.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    baseline_px : int
        Baseline (lag) length [px] at which to evaluate the deviogram.

    Returns
    -------
    np.ndarray
        RMS-deviation roughness [m], shape ``(H, W)``.
    """
    return _deviogram_nu(dem, baseline_px)


def hurst_exponent(
    dem: np.ndarray, baselines_px: Sequence[int]
) -> tuple[float, np.ndarray, np.ndarray]:
    """Global Hurst exponent ``H`` from deviogram scaling (Rosenburg 2011).

    A self-affine surface obeys ``nu(dx) ~ dx^H``; the Hurst exponent ``H`` is
    therefore the slope of ``log nu`` versus ``log dx``. For each baseline in
    ``baselines_px`` the *global* RMS deviation

        nu(dx) = sqrt( mean_over_pixels_and_directions( [z(x+dx) - z(x)]^2 ) )

    is computed, and ``H`` is recovered with :func:`numpy.polyfit` of
    ``log nu`` on ``log dx`` (degree 1).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    baselines_px : sequence of int
        Baseline (lag) lengths [px] to regress over (need >= 2 distinct values).

    Returns
    -------
    H : float
        Fitted Hurst exponent (deviogram log-log slope), typically in ``[0, 1]``
        for natural terrain (can exceed 1 for very smooth crater interiors).
    baselines : np.ndarray
        The baselines actually used [px].
    nu : np.ndarray
        Global RMS deviation at each baseline [m].
    """
    dem = np.asarray(dem, dtype=np.float64)
    bl = np.asarray(sorted({int(b) for b in baselines_px if int(b) >= 1}),
                    dtype=np.float64)
    if bl.size < 2:
        raise ValueError("hurst_exponent needs >= 2 distinct baselines")

    nu = np.empty(bl.size, dtype=np.float64)
    for k, lag in enumerate(bl.astype(int)):
        sq_sum = 0.0
        count = 0
        for axis in (0, 1):
            sl_a = [slice(None), slice(None)]
            sl_b = [slice(None), slice(None)]
            sl_a[axis] = slice(0, dem.shape[axis] - lag)
            sl_b[axis] = slice(lag, dem.shape[axis])
            diff = dem[tuple(sl_b)] - dem[tuple(sl_a)]
            sq_sum += float(np.sum(diff ** 2))
            count += diff.size
        nu[k] = np.sqrt(sq_sum / max(count, 1))

    # Guard against zero (perfectly flat) before taking logs.
    pos = nu > 0
    if pos.sum() < 2:
        return 0.0, bl, nu
    coeffs = np.polyfit(np.log(bl[pos]), np.log(nu[pos]), 1)
    H = float(coeffs[0])
    return H, bl, nu


def iqr_roughness(dem: np.ndarray, window: int) -> np.ndarray:
    """Inter-quartile-range roughness (robust to boulder outliers).

    Moving-window 75th-minus-25th percentile of elevation
    (Kreslavsky & Head 2000), evaluated with
    :func:`scipy.ndimage.percentile_filter`. Being median-based, the IQR is
    insensitive to a few extreme outliers (boulders, data spikes) that would
    inflate an RMS statistic.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    window : int
        Window side length [px].

    Returns
    -------
    np.ndarray
        IQR of elevation within the window [m], shape ``(H, W)``.
    """
    dem = np.asarray(dem, dtype=np.float64)
    win = max(int(window), 3)
    q75 = percentile_filter(dem, percentile=75, size=win, mode="nearest")
    q25 = percentile_filter(dem, percentile=25, size=win, mode="nearest")
    return q75 - q25
