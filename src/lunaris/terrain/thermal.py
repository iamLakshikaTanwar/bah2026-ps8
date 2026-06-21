"""Thermal cold-trap analysis: cold-trap masks, sublimation rate, ice-stability
depth, and a 1-D regolith thermal profile.

Implemented by: **terrain agent**.
"""

from __future__ import annotations

import numpy as np

from ..constants import T_WATER_STABLE

__all__ = [
    "cold_trap_mask",
    "sublimation_rate",
    "ice_stability_depth",
    "regolith_thermal_profile",
]


def cold_trap_mask(tmax: np.ndarray, threshold: float = T_WATER_STABLE) -> np.ndarray:
    """Boolean water-ice cold-trap mask.

        cold_trap = tmax < threshold   (default 110 K for H2O)

    Parameters
    ----------
    tmax : np.ndarray
        Annual maximum surface temperature [K], shape ``(H, W)``.
    threshold : float, default :data:`T_WATER_STABLE`
        Stability ceiling [K].

    Returns
    -------
    np.ndarray
        Boolean cold-trap mask, shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def sublimation_rate(T: np.ndarray) -> np.ndarray:
    """Free-space water-ice sublimation mass-loss rate.

        E = P_sv(T) * sqrt( M / (2 pi R T) )   [kg m^-2 s^-1]

    with ``P_sv`` the saturation vapour pressure (Clausius-Clapeyron) and
    ``M = H2O_MOLAR_MASS``.

    Parameters
    ----------
    T : np.ndarray
        Surface temperature [K], shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Sublimation rate [kg m^-2 s^-1], shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def ice_stability_depth(tmax: np.ndarray, **kwargs) -> np.ndarray:
    """Depth to the ice-stable layer from the thermal regime.

    Where surface ``tmax`` exceeds the stability ceiling, ice is only stable
    below a burial depth set by the regolith thermal gradient; returns that
    depth (0 where ice is stable at the surface).

    Parameters
    ----------
    tmax : np.ndarray
        Annual maximum surface temperature [K], shape ``(H, W)``.
    **kwargs
        Optional thermophysical overrides (diffusivity, geothermal flux).

    Returns
    -------
    np.ndarray
        Ice-stability depth [m], shape ``(H, W)``.
    """
    raise NotImplementedError("terrain agent")


def regolith_thermal_profile(*args, **kwargs):
    """1-D regolith temperature-vs-depth profile.

    Solves the conductive heat equation with a surface boundary condition and
    the lunar geothermal flux to give ``T(z)`` for a column.
    """
    raise NotImplementedError("terrain agent")
