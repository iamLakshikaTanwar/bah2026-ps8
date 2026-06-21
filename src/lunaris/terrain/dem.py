"""DEM derivatives: slope, aspect, and multi-scale roughness.

Implemented by: **terrain agent**.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["slope_horn", "aspect", "rms_roughness", "hurst_exponent", "iqr_roughness"]


def slope_horn(dem: np.ndarray, res: float) -> np.ndarray:
    """Slope magnitude (degrees) via Horn's 3x3 method.

        slope = arctan( sqrt(dz/dx^2 + dz/dy^2) )

    where ``dz/dx, dz/dy`` are Horn's weighted finite differences over the 3x3
    neighbourhood with pixel size ``res``.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].

    Returns
    -------
    np.ndarray
        Slope [deg], shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def aspect(dem: np.ndarray, res: float) -> np.ndarray:
    """Aspect (downslope azimuth, degrees clockwise from north).

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
    raise NotImplementedError("terrain agent")


def rms_roughness(dem: np.ndarray, baseline_px: int) -> np.ndarray:
    """RMS height (roughness) at a given baseline.

        rms = sqrt( mean( (z - <z>)^2 ) )  over a ``baseline_px`` window.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    baseline_px : int
        Window side length [px] defining the baseline scale.

    Returns
    -------
    np.ndarray
        RMS roughness [m], shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def hurst_exponent(dem: np.ndarray, baselines_px: Sequence[int]) -> np.ndarray:
    """Hurst exponent ``H`` from RMS-deviation vs baseline scaling.

    Fits ``log(rms) = H * log(baseline) + c`` across ``baselines_px`` per pixel
    (self-affine roughness; ``H`` separates fractal rough surfaces from ice).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    baselines_px : sequence of int
        Baseline window sizes [px] to regress over.

    Returns
    -------
    np.ndarray
        Hurst exponent map, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def iqr_roughness(dem: np.ndarray, window: int) -> np.ndarray:
    """Inter-quartile-range roughness (robust to boulder outliers).

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
    raise NotImplementedError("terrain agent")
