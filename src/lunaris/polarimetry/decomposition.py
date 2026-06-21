"""Hybrid-polarity (compact-pol) decompositions: m-chi, m-delta, CHILD, and
optional Cloude-Pottier.

Implemented by: **polarimetry agent**.

These separate the backscatter into even-bounce / volume / odd-bounce (or
double / volume / surface) power components, isolating the volume-scatter
channel that dominates over subsurface ice.
"""

from __future__ import annotations

import numpy as np

__all__ = ["m_chi", "m_delta", "child_parameters", "cloude_pottier"]


def m_chi(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m-chi decomposition (Raney et al. 2012).

    With degree of polarisation ``m = sqrt(s1^2+s2^2+s3^2)/s0`` and ellipticity
    ``chi = 0.5 * arcsin(-s3 / (m * s0))``:

        even   = sqrt(m * s0 * (1 + sin 2chi) / 2)   (double-bounce)
        volume = sqrt(s0 * (1 - m))                  (random volume)
        odd    = sqrt(m * s0 * (1 - sin 2chi) / 2)   (single-bounce / Bragg)

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (even, volume, odd) : tuple of np.ndarray
        Power components, shape ``(H, W)`` (RGB-ready: R=even, G=volume, B=odd).
    """
    raise NotImplementedError("polarimetry agent")


def m_delta(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m-delta decomposition (Raney et al. 2012).

    Uses the relative phase ``delta = arctan2(s3, s2)`` (or ``s3, s1`` per
    convention) instead of ellipticity to split double / volume / surface power.

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (double, volume, surface) : tuple of np.ndarray
        Power components, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def child_parameters(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> dict[str, np.ndarray]:
    """CHILD / hybrid-pol descriptors (chi, delta, psi, conformity).

        chi        = 0.5 * arcsin(-s3 / (m s0))      ellipticity
        delta      = arctan2(s3, s2)                 relative phase
        psi        = 0.5 * arctan2(s2, s1)           orientation
        conformity = 2 s2 / (s0 + s1)  (approx.)     surface/volume sign

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"chi"``, ``"delta"``, ``"psi"``, ``"conformity"``.
    """
    raise NotImplementedError("polarimetry agent")


def cloude_pottier(*args, **kwargs):
    """Cloude-Pottier H/A/alpha eigen-decomposition (optional, full-pol only).

    Returns entropy ``H``, anisotropy ``A`` and mean scattering angle
    ``alpha`` from the eigenvalues of the coherency matrix ``T3``.
    """
    raise NotImplementedError("polarimetry agent")
