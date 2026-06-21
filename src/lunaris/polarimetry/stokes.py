"""Stokes-vector synthesis from the polarimetric scattering matrix.

Implemented by: **polarimetry agent**.

DFSAR transmits/receives in a linear (H/V) basis. For the CPR/DOP ice criterion
we work in the *circular* basis. The conversion of the complex scattering
matrix ``S = [[Shh, Shv], [Svh, Svv]]`` to circular receive components and the
4-vector Stokes parameters are defined here.

References
----------
Raney, R. K. (2007). "Hybrid-polarity SAR architecture." IEEE TGRS 45(11).
Raney, R. K. et al. (2012). "The m-chi decomposition of hybrid dual-polarimetric
    radar data." IGARSS.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["scattering_matrix_to_circular", "stokes_from_circular", "multilook"]


def scattering_matrix_to_circular(
    shh: np.ndarray, shv: np.ndarray, svh: np.ndarray, svv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a linear-basis scattering matrix to circular receive fields.

    CTLR hybrid-pol (transmit right-circular). For a right-circular transmit the
    received horizontal/vertical fields after the quarter-wave on transmit are

        E_RH = (Shh + j Shv) / sqrt(2)
        E_RV = (Svh + j Svv) / sqrt(2)

    i.e. the columns of ``S`` driven by the circular transmit vector
    ``(1, j)^T / sqrt(2)`` (Raney 2007 hybrid-polarity / CTLR architecture; the
    sqrt(2) keeps total transmitted power unit-normalised).

    Parameters
    ----------
    shh, shv, svh, svv : np.ndarray (complex)
        Complex scattering-matrix elements, shape ``(H, W)``.

    Returns
    -------
    (E_RH, E_RV) : tuple[np.ndarray, np.ndarray]
        Complex circular-basis receive fields, shape ``(H, W)``.
    """
    shh = np.asarray(shh, dtype=np.complex128)
    shv = np.asarray(shv, dtype=np.complex128)
    svh = np.asarray(svh, dtype=np.complex128)
    svv = np.asarray(svv, dtype=np.complex128)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    E_RH = (shh + 1j * shv) * inv_sqrt2
    E_RV = (svh + 1j * svv) * inv_sqrt2
    return E_RH, E_RV


def stokes_from_circular(
    E_RH: np.ndarray, E_RV: np.ndarray, window: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 4 Stokes parameters from circular receive fields.

    With ``<.>`` an ``window x window`` boxcar (multilook) spatial ensemble
    average implemented via :func:`scipy.ndimage.uniform_filter`:

        s0 = <|E_RH|^2> + <|E_RV|^2>
        s1 = <|E_RH|^2> - <|E_RV|^2>
        s2 =  2 Re<E_RH E_RV*>
        s3 = -2 Im<E_RH E_RV*>

    (Raney 2007, hybrid-polarity Stokes definition.)

    .. note::
        Multilooking is *mandatory*: without spatial averaging every pixel is a
        fully-polarised single look, the cross-products factorise and the degree
        of polarisation ``m`` collapses to 1 identically. The window introduces
        the partial-coherence statistics that make ``m < 1`` and the CPR/DOP ice
        criterion meaningful.

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
    E_RH = np.asarray(E_RH, dtype=np.complex128)
    E_RV = np.asarray(E_RV, dtype=np.complex128)

    def _avg(a: np.ndarray) -> np.ndarray:
        return uniform_filter(a, size=window, mode="reflect")

    p_rh = _avg(np.abs(E_RH) ** 2)          # <|E_RH|^2>
    p_rv = _avg(np.abs(E_RV) ** 2)          # <|E_RV|^2>
    cross = _avg(E_RH * np.conj(E_RV))      # <E_RH E_RV*> (complex)

    s0 = p_rh + p_rv
    s1 = p_rh - p_rv
    s2 = 2.0 * np.real(cross)
    s3 = -2.0 * np.imag(cross)
    return s0, s1, s2, s3


def multilook(arr: np.ndarray, looks: int = 2) -> np.ndarray:
    """Multi-look (block-average) a raster to reduce speckle.

    The input is averaged in non-overlapping ``looks x looks`` blocks. Block
    averaging of ``L`` independent single-look intensity samples raises the
    Equivalent Number of Looks by ``looks**2`` (ENL ~ ``looks**2`` for fully
    developed speckle), reducing the intensity coefficient-of-variation by
    ``1/looks`` while shrinking the raster by the same factor in each axis.

    Parameters
    ----------
    arr : np.ndarray
        Input raster, shape ``(H, W)`` (real or complex).
    looks : int, default 2
        Block size; output shape is ``(H // looks, W // looks)``.

    Returns
    -------
    np.ndarray
        The block-averaged raster, shape ``(H // looks, W // looks)``.
    """
    arr = np.asarray(arr)
    if looks <= 1:
        return arr.copy()
    h, w = arr.shape[:2]
    nh, nw = h // looks, w // looks
    if nh == 0 or nw == 0:
        # window larger than the array: fall back to a single mean cell.
        return np.asarray(arr.mean()).reshape(1, 1)
    cropped = arr[: nh * looks, : nw * looks]
    # reshape into (nh, looks, nw, looks) and average the two block axes.
    blocks = cropped.reshape(nh, looks, nw, looks)
    return blocks.mean(axis=(1, 3))
