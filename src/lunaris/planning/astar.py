"""A* shortest-path planner on a cost grid.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import numpy as np

__all__ = ["astar"]


def astar(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """A* minimum-cost path on an 8-connected grid.

    Uses the octile distance as the admissible heuristic and accumulates the
    per-cell traversal ``cost`` (``inf`` cells are impassable).

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(row, col)`` path from ``start`` to ``goal`` (empty if none).
    """
    raise NotImplementedError("planning agent")
