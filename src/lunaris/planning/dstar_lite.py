"""D* Lite incremental replanner for dynamic cost grids.

Implemented by: **planning agent**.

D* Lite (Koenig & Likhachev 2002) efficiently repairs a shortest path when edge
costs change (e.g. newly detected hazards), without replanning from scratch —
ideal for online rover navigation as the map updates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DStarLite"]


class DStarLite:
    """Incremental D* Lite planner over a 2-D cost grid.

    Parameters
    ----------
    cost : np.ndarray
        Initial per-cell cost grid (``inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    """

    def __init__(
        self,
        cost: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        raise NotImplementedError("planning agent")

    def plan(self) -> list[tuple[int, int]]:
        """Compute / return the current shortest path ``start -> goal``."""
        raise NotImplementedError("planning agent")

    def update_cost(self, changes: dict[tuple[int, int], float]) -> None:
        """Apply ``{(row, col): new_cost}`` edge changes and repair the path."""
        raise NotImplementedError("planning agent")
