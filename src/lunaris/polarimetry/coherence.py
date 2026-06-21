"""Polarimetric coherence.

Implemented by: **polarimetry agent**.

References
----------
Campbell, B. A. (2012). "High circular polarization ratios..." JGR Planets 117 —
    co-pol coherence / correlation in planetary radar.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["copol_coherence"]

_EPS = 1e-12


def copol_coherence(
    shh: np.ndarray, svv: np.ndarray, window: int = 5
) -> np.ndarray:
    """Co-polarised (HH-VV) complex-coherence magnitude.

        gamma_hhvv = |<Shh Svv*>| / sqrt(<|Shh|^2> <|Svv|^2>)

    with ``<.>`` an ``window x window`` boxcar spatial average
    (:func:`scipy.ndimage.uniform_filter`). By the Cauchy-Schwarz inequality
    the result lies in ``[0, 1]``. Low co-pol coherence accompanies volume
    scattering (ice / porous regolith); high coherence accompanies single
    surface scattering.

    Parameters
    ----------
    shh, svv : np.ndarray (complex)
        Co-pol scattering elements, shape ``(H, W)``.
    window : int, default 5
        Averaging window side length.

    Returns
    -------
    np.ndarray
        Coherence magnitude in ``[0, 1]``, shape ``(H, W)``.
    """
    shh = np.asarray(shh, dtype=np.complex128)
    svv = np.asarray(svv, dtype=np.complex128)

    def _avg(a: np.ndarray) -> np.ndarray:
        return uniform_filter(a, size=window, mode="reflect")

    # uniform_filter does not accept complex arrays directly -> filter the real
    # and imaginary parts of the cross-product separately.
    cross = shh * np.conj(svv)
    cross_avg = _avg(cross.real) + 1j * _avg(cross.imag)
    p_hh = _avg(np.abs(shh) ** 2)
    p_vv = _avg(np.abs(svv) ** 2)

    denom = np.sqrt(p_hh * p_vv)
    gamma = np.abs(cross_avg) / np.where(denom < _EPS, _EPS, denom)
    return np.clip(gamma, 0.0, 1.0)
