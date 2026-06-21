"""Boulder detection (shadow method) and density mapping from optical imagery.

Implemented by: **terrain agent**.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_boulders_shadow", "boulder_density_map"]


def detect_boulders_shadow(
    img: np.ndarray, sun_elev_deg: float, gsd: float
) -> np.ndarray:
    """Detect boulders from their cast shadows in an optical image.

    Thresholds dark shadow blobs, measures shadow length ``L``, and infers
    boulder height ``h = L * tan(sun_elev_deg)`` (and diameter from the bright
    sunlit cap). Returns a record array / Nx4 of ``(row, col, diameter_m,
    height_m)``.

    Parameters
    ----------
    img : np.ndarray
        Single-band optical image, shape ``(H, W)``.
    sun_elev_deg : float
        Solar elevation angle [deg].
    gsd : float
        Ground sample distance [m/px].

    Returns
    -------
    np.ndarray
        Detected boulders, shape ``(N, 4)`` = ``(row, col, diameter_m, height_m)``.
    """
    raise NotImplementedError("terrain agent")


def boulder_density_map(
    boulders: np.ndarray, shape: tuple[int, int], window: int
) -> np.ndarray:
    """Rasterise boulders into a spatial density (count per window) map.

    Parameters
    ----------
    boulders : np.ndarray
        Output of :func:`detect_boulders_shadow`, shape ``(N, >=2)``.
    shape : tuple[int, int]
        Output raster shape ``(H, W)``.
    window : int
        Density kernel side length [px].

    Returns
    -------
    np.ndarray
        Boulder density [count / window], shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")
