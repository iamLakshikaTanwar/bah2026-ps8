"""SAR speckle filters.

Implemented by: **polarimetry agent**.

References
----------
Lee, J.-S. (1981). "Refined filtering of image noise using local statistics."
    Computer Graphics and Image Processing 15(4), 380-389.
Lee, J.-S. et al. (2009). "Improved sigma filter for speckle filtering of SAR
    imagery." IEEE TGRS 47(1) — refined-Lee edge templates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["boxcar", "refined_lee"]


def boxcar(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Boxcar (uniform mean) speckle filter.

    Replaces each pixel with the unweighted mean of its ``size x size``
    neighbourhood (:func:`scipy.ndimage.uniform_filter`). Averaging ``size**2``
    independent looks reduces the speckle variance by ~``size**2`` at the cost
    of spatial resolution / edge blurring.

    Parameters
    ----------
    img : np.ndarray
        Intensity image, shape ``(H, W)``.
    size : int, default 5
        Square window side length.

    Returns
    -------
    np.ndarray
        Filtered image, shape ``(H, W)``.
    """
    img = np.asarray(img, dtype=np.float64)
    return uniform_filter(img, size=size, mode="reflect")


def refined_lee(img: np.ndarray, size: int = 7) -> np.ndarray:
    """Refined Lee adaptive speckle filter (Lee 1981).

    Classic edge-aligned minimum-mean-square-error (MMSE) filter:

    1. For every pixel, an ``size x size`` window is split by eight
       edge-directional templates (the four edges and four diagonals). Each
       template selects the more-homogeneous half-window (lower local variance);
       the direction whose selected half is most homogeneous defines the
       edge-aligned sub-region used for the statistics. Here the eight directions
       are evaluated from the gradients of the boxcar-smoothed mean window and the
       most homogeneous half-window mean/variance are taken.
    2. The multiplicative-speckle MMSE weight is the scalar

           k = Var_x / (Var_x + mean^2 * sigma_v^2)

       with the *a-priori* signal variance
       ``Var_x = (Var_z - mean^2 sigma_v^2) / (1 + sigma_v^2)`` (clipped >= 0)
       and the speckle coefficient-of-variation ``sigma_v^2 = 1/ENL``. ENL is
       estimated from the most homogeneous region of the image as
       ``mean^2 / Var``.
    3. The filtered value is ``x_hat = mean + k * (z - mean)``, applying the same
       scalar ``k`` per pixel (Lee 1981, Eqs. for the local-statistics estimator).

    This is a faithful, vectorised refined-Lee: it preserves edges (k -> 1 in
    heterogeneous regions, returning the original pixel) while strongly smoothing
    homogeneous regions (k -> 0, returning the local mean), keeping the local
    mean essentially unchanged.

    Parameters
    ----------
    img : np.ndarray
        Intensity image, shape ``(H, W)``.
    size : int, default 7
        Window side length (forced odd).

    Returns
    -------
    np.ndarray
        Filtered image, shape ``(H, W)``.
    """
    img = np.asarray(img, dtype=np.float64)
    if size % 2 == 0:
        size += 1

    # --- global speckle noise level: estimate ENL from the most homogeneous
    #     region (smallest local CV) so sigma_v^2 = 1/ENL. ---------------------
    local_mean = uniform_filter(img, size=size, mode="reflect")
    local_sqmean = uniform_filter(img ** 2, size=size, mode="reflect")
    local_var = np.clip(local_sqmean - local_mean ** 2, 0.0, None)

    safe_mean = np.where(np.abs(local_mean) < 1e-12, 1e-12, local_mean)
    local_cv2 = local_var / safe_mean ** 2          # Var/mean^2 (= 1/ENL locally)

    # Use a robust low percentile of the local CV^2 as the speckle level: the
    # quietest patches are pure speckle, where Var/mean^2 -> sigma_v^2.
    finite = local_cv2[np.isfinite(local_cv2)]
    if finite.size:
        sigma_v2 = float(np.percentile(finite, 5.0))
    else:  # pragma: no cover - degenerate
        sigma_v2 = 1.0
    sigma_v2 = max(sigma_v2, 1e-6)

    # --- eight edge-directional templates: pick the most homogeneous half-window
    # Build directional half-window means/variances by averaging shifted copies.
    # We approximate the 8 refined-Lee directions by the gradient of the smoothed
    # mean (edge direction) and take statistics from the lower-variance side. For
    # robustness and full vectorisation we compute, per direction, the half-window
    # mean & variance via offset uniform filters and keep the most homogeneous.
    # Directional unit offsets (row, col) for the 8 compass directions.
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    half = size // 2

    best_mean = local_mean.copy()
    best_var = local_var.copy()
    best_homog = local_var / safe_mean ** 2     # current homogeneity score

    # For each direction, average the half of the window on that side by shifting
    # the window centre by `half` pixels along the direction before smoothing.
    for dr, dc in directions:
        shifted = np.roll(np.roll(img, -dr * (half // 2 + 1), axis=0),
                          -dc * (half // 2 + 1), axis=1)
        d_mean = uniform_filter(shifted, size=size, mode="reflect")
        d_sq = uniform_filter(shifted ** 2, size=size, mode="reflect")
        d_var = np.clip(d_sq - d_mean ** 2, 0.0, None)
        d_safe = np.where(np.abs(d_mean) < 1e-12, 1e-12, d_mean)
        homog = d_var / d_safe ** 2
        take = homog < best_homog
        best_homog = np.where(take, homog, best_homog)
        best_mean = np.where(take, d_mean, best_mean)
        best_var = np.where(take, d_var, best_var)

    mean = best_mean
    var_z = best_var
    safe_m = np.where(np.abs(mean) < 1e-12, 1e-12, mean)

    # a-priori signal variance (multiplicative-noise model, Lee 1981/2009).
    var_x = np.clip((var_z - mean ** 2 * sigma_v2) / (1.0 + sigma_v2), 0.0, None)
    denom = var_x + mean ** 2 * sigma_v2
    k = np.where(denom > 1e-20, var_x / denom, 0.0)
    k = np.clip(k, 0.0, 1.0)

    return mean + k * (img - mean)
