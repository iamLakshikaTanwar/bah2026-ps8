"""Circular Polarisation Ratio (CPR) and Degree of Polarisation (DOP).

Implemented by: **polarimetry agent**.

These are the two scalars in the Sinha et al. (2026) ice criterion
(``CPR>1 & DOP<0.13``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["circular_polarization_ratio", "degree_of_polarization", "sc_oc_power"]


def circular_polarization_ratio(s0: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """Circular Polarisation Ratio from Stokes ``s0`` and ``s3``.

        CPR = SC / OC = (s0 - s3) / (s0 + s3)

    where SC/OC are the same-/opposite-sense circular received powers. Values
    ``>1`` indicate strong same-sense (CBOE) backscatter consistent with ice.

    Parameters
    ----------
    s0, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        CPR, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def degree_of_polarization(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> np.ndarray:
    """Degree of Polarisation from the full Stokes vector.

        DOP = sqrt(s1^2 + s2^2 + s3^2) / s0

    Low DOP (<0.13) distinguishes coherent ice volume-scatter from polarised
    rough-surface returns (high DOP) that can also exceed CPR=1.

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        DOP in ``[0, 1]``, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def sc_oc_power(s0: np.ndarray, s3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-sense (SC) and opposite-sense (OC) circular powers.

        SC = (s0 - s3) / 2 ;  OC = (s0 + s3) / 2

    Parameters
    ----------
    s0, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (sc, oc) : tuple[np.ndarray, np.ndarray]
        SC and OC powers, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")
