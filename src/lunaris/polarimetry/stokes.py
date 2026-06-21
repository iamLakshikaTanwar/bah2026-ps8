"""Stokes-vector synthesis from the polarimetric scattering matrix.

Implemented by: **polarimetry agent**.

DFSAR transmits/receives in a linear (H/V) basis. For the CPR/DOP ice criterion
we work in the *circular* basis. The conversion of the complex scattering
matrix ``S = [[Shh, Shv], [Svh, Svv]]`` to circular receive components and the
4-vector Stokes parameters are defined here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["scattering_matrix_to_circular", "stokes_from_circular", "multilook"]


def scattering_matrix_to_circular(
    shh: np.ndarray, shv: np.ndarray, svh: np.ndarray, svv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a linear-basis scattering matrix to circular receive fields.

    For circular transmit (right-hand) the received right/left fields are

        E_RH = 0.5 * (Shh - Svv + j (Shv + Svh))
        E_RV = 0.5 * j * (Shh + Svv + j (Shv - Svh))

    (Raney 2007, hybrid-polarity convention.)

    Parameters
    ----------
    shh, shv, svh, svv : np.ndarray (complex)
        Complex scattering-matrix elements, shape ``(H, W)``.

    Returns
    -------
    (E_RH, E_RV) : tuple[np.ndarray, np.ndarray]
        Complex circular-basis receive fields, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def stokes_from_circular(
    E_RH: np.ndarray, E_RV: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 4 Stokes parameters from circular receive fields.

    With ``<.>`` an ``window x window`` spatial ensemble average:

        s0 = <|E_RH|^2> + <|E_RV|^2>
        s1 = <|E_RH|^2> - <|E_RV|^2>
        s2 =  2 Re<E_RH E_RV*>
        s3 = -2 Im<E_RH E_RV*>

    Parameters
    ----------
    E_RH, E_RV : np.ndarray (complex)
        Circular-basis receive fields, shape ``(H, W)``.
    window : int, default 5
        Side length of the boxcar averaging window.

    Returns
    -------
    (s0, s1, s2, s3) : tuple of np.ndarray
        Real Stokes parameters, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def multilook(arr: np.ndarray, looks: int = 2) -> np.ndarray:
    """Multi-look (block-average) a raster to reduce speckle.

    Parameters
    ----------
    arr : np.ndarray
        Input raster, shape ``(H, W)``.
    looks : int, default 2
        Block size; output shape is ``(H // looks, W // looks)``.

    Returns
    -------
    np.ndarray
        The block-averaged raster.
    """
    raise NotImplementedError("polarimetry agent")
