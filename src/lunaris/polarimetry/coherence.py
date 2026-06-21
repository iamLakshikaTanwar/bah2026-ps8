"""Polarimetric coherence.

Implemented by: **polarimetry agent**.
"""

from __future__ import annotations

import numpy as np

__all__ = ["copol_coherence"]


def copol_coherence(
    shh: np.ndarray, svv: np.ndarray, window: int = 5
) -> np.ndarray:
    """Co-polarised (HH-VV) complex-coherence magnitude.

        gamma_hhvv = |<Shh Svv*>| / sqrt(<|Shh|^2> <|Svv|^2>)

    averaged over an ``window x window`` boxcar. Low co-pol coherence
    accompanies volume scattering (ice/regolith), high coherence accompanies
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
    raise NotImplementedError("polarimetry agent")
