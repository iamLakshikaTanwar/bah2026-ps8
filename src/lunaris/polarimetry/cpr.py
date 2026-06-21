"""Circular Polarisation Ratio (CPR) and Degree of Polarisation (DOP).

Implemented by: **polarimetry agent**.

These are the two scalars in the Sinha et al. (2026) ice criterion
(``CPR>1 & DOP<0.13``).

References
----------
Raney, R. K. et al. (2012). m-chi decomposition, IGARSS — circular Stokes.
Campbell, B. A. (2012). "High circular polarization ratios in radar scattering
    from geologic targets." JGR Planets 117, E06008 — CPR = SC/OC definition.
"""

from __future__ import annotations

import numpy as np

__all__ = ["circular_polarization_ratio", "degree_of_polarization", "sc_oc_power"]

_EPS = 1e-12


def circular_polarization_ratio(s0: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """Circular Polarisation Ratio from Stokes ``s0`` and ``s3``.

        CPR = SC / OC = (s0 - s3) / (s0 + s3)

    (Campbell 2012, Eq. 1; SC/OC are the same-/opposite-sense circular received
    powers.) Values ``>1`` indicate strong same-sense (coherent-backscatter
    opposition effect) scattering consistent with subsurface ice.

    A small epsilon guards the ``s0 + s3 -> 0`` denominator (a physical Stokes
    vector has ``s0 >= |s3|`` so ``s0 + s3 >= 0``).

    Parameters
    ----------
    s0, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        CPR, shape ``(H, W)``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    denom = s0 + s3
    return (s0 - s3) / np.where(np.abs(denom) < _EPS, _EPS, denom)


def degree_of_polarization(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> np.ndarray:
    """Degree of Polarisation from the full Stokes vector.

        DOP = m = sqrt(s1^2 + s2^2 + s3^2) / s0

    (Raney 2012.) Low DOP (<0.13) distinguishes coherent ice volume-scatter
    from polarised rough-surface returns (high DOP) that can also exceed
    ``CPR = 1``. The result is clipped to ``[0, 1]`` (the physical range).

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        DOP in ``[0, 1]``, shape ``(H, W)``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    pol = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
    m = pol / np.where(np.abs(s0) < _EPS, _EPS, s0)
    return np.clip(m, 0.0, 1.0)


def sc_oc_power(s0: np.ndarray, s3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-sense (SC) and opposite-sense (OC) circular powers.

        SC = (s0 - s3) / 2 ;  OC = (s0 + s3) / 2

    (Raney 2012; for circular transmit the SC channel carries odd-bounce/volume
    same-sense power and OC the opposite-sense surface power.) By construction
    ``CPR = SC / OC``.

    Parameters
    ----------
    s0, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (sc, oc) : tuple[np.ndarray, np.ndarray]
        SC and OC powers, shape ``(H, W)``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    sc = (s0 - s3) / 2.0
    oc = (s0 + s3) / 2.0
    return sc, oc
