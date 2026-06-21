"""Traversability cost-grid construction.

Implemented by: **planning agent**.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["build_cost_grid"]


def build_cost_grid(
    slope: np.ndarray,
    roughness: np.ndarray,
    illumination: np.ndarray,
    traversable: np.ndarray,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Combine hazard layers into a per-pixel traversal cost grid.

        cost = w_s * slope_n + w_r * roughness_n + w_i * (1 - illumination)

    normalised to ``[0, 1]`` per term; non-traversable pixels are set to ``inf``.

    Parameters
    ----------
    slope : np.ndarray
        Slope [deg], shape ``(H, W)``.
    roughness : np.ndarray
        Roughness [m], shape ``(H, W)``.
    illumination : np.ndarray
        Illuminated fraction in ``[0, 1]``, shape ``(H, W)``.
    traversable : np.ndarray
        Boolean traversability mask, shape ``(H, W)``.
    weights : mapping, optional
        Keys ``"slope"``, ``"roughness"``, ``"illumination"``; defaults applied
        if ``None``.

    Returns
    -------
    np.ndarray
        Cost grid (``inf`` = impassable), shape ``(H, W)``.
    """
    raise NotImplementedError("planning agent")
