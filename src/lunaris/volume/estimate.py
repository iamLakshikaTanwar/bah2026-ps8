"""Ice volume / mass estimation with Monte-Carlo uncertainty.

Implemented by: **volume agent**.
"""

from __future__ import annotations

import numpy as np

from ..constants import RHO_ICE_KGM3

__all__ = ["ice_volume", "ice_mass", "monte_carlo_volume"]


def ice_volume(
    area_m2: np.ndarray | float,
    depth_m: np.ndarray | float,
    frac: np.ndarray | float,
) -> np.ndarray | float:
    """Bulk ice volume from footprint area, sampled depth and ice fraction.

        V = area_m2 * depth_m * frac   [m^3]

    Parameters
    ----------
    area_m2 : np.ndarray or float
        Ice-bearing footprint area [m^2].
    depth_m : np.ndarray or float
        Sampled subsurface depth (<= top 5 m) [m].
    frac : np.ndarray or float
        Ice volume fraction in ``[0, 1]``.

    Returns
    -------
    np.ndarray or float
        Ice volume [m^3].
    """
    raise NotImplementedError("volume agent")


def ice_mass(
    volume_m3: np.ndarray | float, rho: float = RHO_ICE_KGM3
) -> np.ndarray | float:
    """Ice mass from volume and density.

        M = rho * V   [kg]

    Parameters
    ----------
    volume_m3 : np.ndarray or float
        Ice volume [m^3].
    rho : float, default :data:`RHO_ICE_KGM3`
        Ice density [kg m^-3].

    Returns
    -------
    np.ndarray or float
        Ice mass [kg].
    """
    raise NotImplementedError("volume agent")


def monte_carlo_volume(
    area_m2: float,
    depth_m: float,
    frac: float,
    n: int = 10000,
    seed: int = 0,
    **uncertainty,
) -> dict:
    """Monte-Carlo ice-volume estimate with uncertainty propagation.

    Draws ``n`` samples of area, depth and ice fraction from their uncertainty
    distributions (passed via ``uncertainty`` as ``*_std`` keywords) and returns
    summary statistics.

    Parameters
    ----------
    area_m2, depth_m, frac : float
        Central estimates.
    n : int, default 10000
        Number of Monte-Carlo samples.
    seed : int, default 0
        RNG seed.
    **uncertainty
        Standard deviations, e.g. ``area_std``, ``depth_std``, ``frac_std``.

    Returns
    -------
    dict
        Keys ``"mean"``, ``"std"``, ``"ci"`` (2.5/97.5 percentile tuple),
        plus the raw ``"samples"`` array, for volume [m^3].
    """
    raise NotImplementedError("volume agent")
