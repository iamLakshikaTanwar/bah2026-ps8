"""NSGA-II multi-objective traverse optimisation.

Implemented by: **planning agent**.

NSGA-II (Deb et al. 2002) finds the Pareto-optimal front trading off competing
traverse objectives (distance vs energy vs hazard vs science value), returning a
set of non-dominated candidate paths for operator selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["nsga2_paths"]


def nsga2_paths(
    objectives: Sequence[np.ndarray],
    start: tuple[int, int],
    goal: tuple[int, int],
    pop_size: int = 50,
    generations: int = 100,
    seed: int = 0,
) -> list[dict]:
    """Pareto-optimal traverse paths via NSGA-II.

    Parameters
    ----------
    objectives : sequence of np.ndarray
        Per-objective cost grids (e.g. distance, energy, hazard), each shape
        ``(H, W)``; all minimised.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    pop_size : int, default 50
        Population size.
    generations : int, default 100
        Number of generations.
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[dict]
        Non-dominated solutions, each ``{"path": [...], "objectives": (...)}``.
    """
    raise NotImplementedError("planning agent")
