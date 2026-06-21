"""A* shortest-path planner on a traversability cost grid.

Classic A* (Hart, Nilsson & Raphael 1968) with a binary heap (``heapq``) open
list.  The per-cell ``cost`` grid gives the entry cost of stepping *into* a
cell; ``+inf`` cells are impassable.  The edge cost between two adjacent cells is
the geometric step length (1 for orthogonal, ``sqrt(2)`` for diagonal moves)
times the average of the two cells' costs, so the planner is sensitive both to
distance travelled and to the hazard of the terrain it crosses.

Heuristic
---------
The admissible, consistent heuristic is the **octile distance** (for
8-connectivity) or Manhattan/Euclidean distance (for 4-connectivity) multiplied
by the minimum finite cell cost present in the grid.  Multiplying by the minimum
cost guarantees the heuristic never over-estimates the true remaining cost
(which uses *average* cell costs ``>=`` the minimum), preserving optimality.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["astar"]

# Orthogonal and diagonal neighbour offsets with their geometric step lengths.
_ORTHO = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
_DIAG = (
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _octile(dr: int, dc: int) -> float:
    """Octile distance between two cells offset by ``(dr, dc)``."""
    dr, dc = abs(dr), abs(dc)
    return (max(dr, dc) - min(dr, dc)) + math.sqrt(2.0) * min(dr, dc)


def astar(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    connectivity: int = 8,
):
    """A* minimum-cost path on a grid.

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``np.inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints (must be in bounds and finite-cost).
    connectivity : int, default 8
        ``8`` for octile (king) moves, ``4`` for orthogonal moves only.

    Returns
    -------
    (path, total_cost)
        ``path`` is the ordered list of ``(row, col)`` cells from ``start`` to
        ``goal`` and ``total_cost`` the accumulated traversal cost.  Returns
        ``(None, inf)`` if no path exists (or an endpoint is blocked / out of
        bounds).
    """
    cost = np.asarray(cost, dtype=np.float64)
    h, w = cost.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    if not (in_bounds(*start) and in_bounds(*goal)):
        return None, math.inf
    if not np.isfinite(cost[start]) or not np.isfinite(cost[goal]):
        return None, math.inf

    neighbours = _ORTHO + _DIAG if connectivity == 8 else _ORTHO

    finite = cost[np.isfinite(cost)]
    min_cost = float(finite.min()) if finite.size else 0.0
    # Scale factor for the heuristic: minimum cell cost keeps it admissible.
    hscale = min_cost

    def heuristic(r: int, c: int) -> float:
        dr, dc = r - goal[0], c - goal[1]
        if connectivity == 8:
            return hscale * _octile(dr, dc)
        return hscale * (abs(dr) + abs(dc))

    if start == goal:
        return [start], 0.0

    # g-score: best known cost-to-reach. open heap holds (f, g, node).
    g = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap: list[tuple[float, float, tuple[int, int]]] = [
        (heuristic(*start), 0.0, start)
    ]
    closed: set[tuple[int, int]] = set()

    while open_heap:
        f, gc, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            # reconstruct
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            return path, gc
        closed.add(node)
        nr0, nc0 = node
        c_here = cost[node]
        for dr, dc, step in neighbours:
            nr, nc = nr0 + dr, nc0 + dc
            if not in_bounds(nr, nc):
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            c_nb = cost[nb]
            if not np.isfinite(c_nb):
                continue
            # edge cost = geometric length * average terrain cost of the two cells
            edge = step * 0.5 * (c_here + c_nb)
            tentative = gc + edge
            if tentative < g.get(nb, math.inf):
                g[nb] = tentative
                came_from[nb] = node
                heapq.heappush(open_heap, (tentative + heuristic(nr, nc), tentative, nb))

    return None, math.inf
