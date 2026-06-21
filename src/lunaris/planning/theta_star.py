"""Theta* any-angle path planner.

Implemented by: **planning agent**.

Theta* (Nash et al. 2007) relaxes the grid-edge constraint by allowing parent
pointers to any visible ancestor (line-of-sight checks), yielding shorter, more
natural any-angle rover paths than grid-locked A*.
"""

from __future__ import annotations

import numpy as np

__all__ = ["theta_star"]


def theta_star(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Any-angle minimum-cost path via Theta* with line-of-sight smoothing.

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    obstacles : np.ndarray, optional
        Boolean obstacle mask blocking line-of-sight, shape ``(H, W)``. If
        ``None``, impassable cells of ``cost`` are used.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(row, col)`` any-angle waypoints (empty if no path).
    """
    raise NotImplementedError("planning agent")
