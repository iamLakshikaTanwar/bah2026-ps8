"""SAR speckle filters.

Implemented by: **polarimetry agent**.
"""

from __future__ import annotations

import numpy as np

__all__ = ["boxcar", "refined_lee"]


def boxcar(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Boxcar (uniform mean) speckle filter.

    Parameters
    ----------
    img : np.ndarray
        Intensity image, shape ``(H, W)``.
    size : int, default 5
        Square window side length.

    Returns
    -------
    np.ndarray
        Filtered image, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")


def refined_lee(img: np.ndarray, size: int = 7) -> np.ndarray:
    """Refined Lee adaptive speckle filter (Lee 1981; edge-aligned MMSE).

    Selects one of eight edge-aligned sub-windows per pixel and applies the
    minimum-mean-square-error Lee estimator using the local speckle statistics,
    preserving edges and the ice-patch boundary.

    Parameters
    ----------
    img : np.ndarray
        Intensity image, shape ``(H, W)``.
    size : int, default 7
        Window side length (odd).

    Returns
    -------
    np.ndarray
        Filtered image, shape ``(H, W)``.
    """
    raise NotImplementedError("polarimetry agent")
