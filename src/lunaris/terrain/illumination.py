"""Illumination, horizon, permanent-/double-shadow, Earth-visibility and
sky-view-factor modelling from a DEM.

Implemented by: **terrain agent**.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "horizon_map",
    "permanent_shadow_mask",
    "double_shadow_mask",
    "earth_visibility_map",
    "sky_view_factor",
]


def horizon_map(dem: np.ndarray, res: float, n_azimuth: int = 180) -> np.ndarray:
    """Per-pixel maximum horizon-elevation angle in each azimuth.

    Ray-traces the DEM outward in ``n_azimuth`` directions, recording the
    maximum elevation angle to the local horizon (used for shadowing & SVF).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    n_azimuth : int, default 180
        Number of azimuth samples.

    Returns
    -------
    np.ndarray
        Horizon elevation angle [deg], shape ``(H, W, n_azimuth)``.
    """
    raise NotImplementedError("terrain agent")


def permanent_shadow_mask(dem: np.ndarray, res: float, **kwargs) -> np.ndarray:
    """Boolean permanently-shadowed-region (PSR) mask.

    A pixel is PSR if the Sun never rises above the local horizon over a full
    year given the Moon's small obliquity (the Sun's max elevation at the pole
    is bounded by ``MOON_OBLIQUITY_DEG``).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    **kwargs
        Optional Sun-geometry overrides (max solar elevation, sub-solar lon).

    Returns
    -------
    np.ndarray
        Boolean PSR mask, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def double_shadow_mask(
    dem: np.ndarray, res: float, psr_mask: np.ndarray
) -> np.ndarray:
    """Boolean "doubly-shadowed" mask within a PSR.

    Doubly-shadowed pixels see neither direct sunlight nor secondary
    (scattered/thermal) illumination from sunlit terrain — the coldest traps.
    Computed as PSR pixels also shadowed from the surrounding illuminated rim.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    psr_mask : np.ndarray
        Boolean PSR mask from :func:`permanent_shadow_mask`, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Boolean doubly-shadowed mask, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def earth_visibility_map(dem: np.ndarray, res: float) -> np.ndarray:
    """Fraction of time Earth is above the local horizon (DTE comms).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].

    Returns
    -------
    np.ndarray
        Earth-visibility fraction in ``[0, 1]``, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def sky_view_factor(dem: np.ndarray, res: float) -> np.ndarray:
    """Sky-view factor (hemispherical sky fraction) per pixel.

        SVF = 1 - <sin^2(horizon_angle)>  averaged over azimuth.

    Drives radiative cooling / cold-trap temperature.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].

    Returns
    -------
    np.ndarray
        SVF in ``[0, 1]``, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")
