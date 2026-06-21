"""Theta* any-angle path planner.

Theta* (Daniel, Nash, Koenig & Felner, *J. Artif. Intell. Res.* 2010; orig.
Nash et al. 2007) is a variant of A* that produces **any-angle** paths.  It runs
the same best-first search over grid vertices, but when relaxing an edge it
tries to make the *parent of the parent* the new parent whenever there is an
unobstructed straight line (line of sight) between them.  Path segments are
therefore not restricted to the 8 grid directions, giving shorter, more natural
traverses than grid-locked A*.

Line of sight is tested with **Bresenham's** integer line algorithm; a segment
is blocked if any cell it passes through is an obstacle.  On obstacle-free
terrain the returned path length is provably ``<=`` the A* path length because a
straight line is the shortest connection between two free points.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["theta_star", "line_of_sight", "bresenham"]

_NEIGH = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)


def bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Integer cells on the straight segment ``(r0,c0) -> (r1,c1)``.

    Standard Bresenham line rasterisation (Bresenham 1965), endpoints included.
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    if dc >= dr:
        err = dc / 2.0
        while c != c1:
            cells.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
            c += sc
        cells.append((r1, c1))
    else:
        err = dr / 2.0
        while r != r1:
            cells.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
            r += sr
        cells.append((r1, c1))
    return cells


def line_of_sight(
    obstacles: np.ndarray, a: tuple[int, int], b: tuple[int, int]
) -> bool:
    """``True`` if no obstacle cell lies on the Bresenham line from ``a`` to ``b``."""
    for (r, c) in bresenham(a[0], a[1], b[0], b[1]):
        if obstacles[r, c]:
            return False
    return True


def _segment_cost(cost: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float:
    """Cost of a straight segment: Euclidean length times mean traversed cell cost.

    Returns ``inf`` if any traversed cell is impassable.
    """
    cells = bresenham(a[0], a[1], b[0], b[1])
    vals = np.array([cost[r, c] for (r, c) in cells], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return math.inf
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return length * float(vals.mean())


def theta_star(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacles: np.ndarray | None = None,
):
    """Any-angle minimum-cost path via Theta* with line-of-sight shortcutting.

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``np.inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    obstacles : np.ndarray, optional
        Boolean obstacle mask blocking line-of-sight.  Defaults to
        ``~np.isfinite(cost)`` (i.e. the impassable cells of ``cost``).

    Returns
    -------
    (path, total_cost)
        Ordered ``(row, col)`` any-angle waypoints and the accumulated cost, or
        ``(None, inf)`` if unreachable.
    """
    cost = np.asarray(cost, dtype=np.float64)
    h, w = cost.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if obstacles is None:
        obstacles = ~np.isfinite(cost)
    obstacles = np.asarray(obstacles, dtype=bool)

    def in_bounds(r, c):
        return 0 <= r < h and 0 <= c < w

    if not (in_bounds(*start) and in_bounds(*goal)):
        return None, math.inf
    if obstacles[start] or obstacles[goal]:
        return None, math.inf
    if start == goal:
        return [start], 0.0

    finite = cost[np.isfinite(cost)]
    hscale = float(finite.min()) if finite.size else 0.0

    def heuristic(node):
        return hscale * math.hypot(node[0] - goal[0], node[1] - goal[1])

    g = {start: 0.0}
    parent = {start: start}
    open_heap = [(heuristic(start), start)]
    closed = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node == goal:
            break
        if node in closed:
            continue
        closed.add(node)
        for dr, dc, _step in _NEIGH:
            nr, nc = node[0] + dr, node[1] + dc
            if not in_bounds(nr, nc):
                continue
            nb = (nr, nc)
            if obstacles[nb] or nb in closed:
                continue
            par = parent[node]
            # Path 2: try to connect the neighbour straight to node's parent.
            if line_of_sight(obstacles, par, nb):
                new_g = g[par] + _segment_cost(cost, par, nb)
                new_par = par
            else:
                # Path 1: ordinary grid edge from node to neighbour.
                new_g = g[node] + _segment_cost(cost, node, nb)
                new_par = node
            if new_g < g.get(nb, math.inf):
                g[nb] = new_g
                parent[nb] = new_par
                heapq.heappush(open_heap, (new_g + heuristic(nb), nb))

    if goal not in parent:
        return None, math.inf

    # reconstruct via parent pointers (any-angle waypoints)
    path = [goal]
    node = goal
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path, g[goal]
