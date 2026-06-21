"""RRT* sampling-based path planner.

Implemented by: **planning agent**.

RRT* (Karaman & Frazzoli 2011) is an asymptotically-optimal sampling planner,
useful for large continuous traverse spaces where grid search is costly.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rrt_star"]


def rrt_star(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    n_samples: int = 2000,
    step: float = 5.0,
    goal_bias: float = 0.05,
    seed: int = 0,
) -> list[tuple[int, int]]:
    """Asymptotically-optimal RRT* path over a cost grid.

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    n_samples : int, default 2000
        Maximum number of random samples.
    step : float, default 5.0
        Steering step length [px].
    goal_bias : float, default 0.05
        Probability of sampling the goal directly.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(row, col)`` path (empty if goal not reached).
    """
    raise NotImplementedError("planning agent")
