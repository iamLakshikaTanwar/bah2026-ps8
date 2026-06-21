"""Dielectric mixing models and radar penetration depth.

Implemented by: **volume agent**.

Relates the effective permittivity of an ice-regolith mixture to its ice
volume fraction (Looyenga / Maxwell-Garnett) and computes the radar penetration
depth that bounds the sampled subsurface column.
"""

from __future__ import annotations

import numpy as np

from ..constants import EPS_ICE, EPS_REGOLITH_DEFAULT, TAN_DELTA_ICE

__all__ = ["looyenga_ice_fraction", "maxwell_garnett_eps", "penetration_depth"]


def looyenga_ice_fraction(
    eps_eff: np.ndarray | float,
    eps_ice: float = EPS_ICE,
    eps_reg: float = EPS_REGOLITH_DEFAULT,
) -> np.ndarray | float:
    """Ice volume fraction from effective permittivity (Looyenga-Landau-Lifshitz).

        eps_eff^(1/3) = f * eps_ice^(1/3) + (1 - f) * eps_reg^(1/3)
        =>  f = (eps_eff^(1/3) - eps_reg^(1/3)) / (eps_ice^(1/3) - eps_reg^(1/3))

    Parameters
    ----------
    eps_eff : np.ndarray or float
        Effective (measured) real permittivity.
    eps_ice : float, default :data:`EPS_ICE`
        Pure-ice permittivity.
    eps_reg : float, default :data:`EPS_REGOLITH_DEFAULT`
        Host-regolith permittivity.

    Returns
    -------
    np.ndarray or float
        Ice volume fraction (clipped to ``[0, 1]`` by the agent).
    """
    raise NotImplementedError("volume agent")


def maxwell_garnett_eps(
    f_ice: np.ndarray | float,
    eps_ice: float = EPS_ICE,
    eps_host: float = EPS_REGOLITH_DEFAULT,
) -> np.ndarray | float:
    """Effective permittivity of sparse ice inclusions (Maxwell-Garnett).

        eps_eff = eps_host * [1 + 3 f (eps_ice - eps_host) /
                  (eps_ice + 2 eps_host - f (eps_ice - eps_host))]

    Parameters
    ----------
    f_ice : np.ndarray or float
        Ice volume fraction in ``[0, 1]``.
    eps_ice : float, default :data:`EPS_ICE`
        Inclusion (ice) permittivity.
    eps_host : float, default :data:`EPS_REGOLITH_DEFAULT`
        Host (regolith) permittivity.

    Returns
    -------
    np.ndarray or float
        Effective permittivity.
    """
    raise NotImplementedError("volume agent")


def penetration_depth(
    wavelength: float,
    eps: np.ndarray | float,
    tan_delta: float = TAN_DELTA_ICE,
) -> np.ndarray | float:
    """1/e radar power penetration (skin) depth in a low-loss medium.

        delta = wavelength * sqrt(eps') / (2 pi * eps'' )  ~
                wavelength / (2 pi sqrt(eps') tan_delta)

    Parameters
    ----------
    wavelength : float
        Free-space radar wavelength [m] (e.g. L-band 0.2399 m).
    eps : np.ndarray or float
        Real permittivity of the medium.
    tan_delta : float, default :data:`TAN_DELTA_ICE`
        Loss tangent.

    Returns
    -------
    np.ndarray or float
        Penetration depth [m].
    """
    raise NotImplementedError("volume agent")
