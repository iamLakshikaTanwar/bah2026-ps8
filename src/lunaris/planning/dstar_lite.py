"""D* Lite incremental replanner for dynamic cost grids.

D* Lite (Koenig & Likhachev, *"D* Lite"*, AAAI 2002) repairs a shortest path
when edge costs change — e.g. when the rover's hazard cameras reveal a boulder
field that was unknown at launch — *without* replanning from scratch.  It is the
incremental search of choice for online navigation because each repair touches
only the cells whose costs actually changed plus their affected descendants.

Algorithm
---------
D* Lite is Lifelong Planning A* (LPA*) run *backwards* from the goal so that the
search tree is rooted at the goal and only the start needs to move.  Each vertex
``s`` keeps two values:

* ``g(s)``  — the cost-to-goal currently believed,
* ``rhs(s)`` — a one-step-lookahead estimate
  ``rhs(s) = min_{s' in succ(s)} (c(s, s') + g(s'))`` (``rhs(goal) = 0``).

A vertex is *locally consistent* when ``g == rhs``.  Inconsistent vertices are
held on a priority queue keyed by

    key(s) = [ min(g, rhs) + h(start, s) + k_m ,  min(g, rhs) ]

where ``h`` is an admissible heuristic and ``k_m`` accumulates the heuristic
shift as the start moves (here the start is fixed, so ``k_m`` stays 0 and only
grows if :meth:`update_cost` is given a moved start in future extensions).
``computeShortestPath`` pops vertices until the start is consistent and its key
is minimal, expanding the same way A* does but propagating both over- and
under-consistency.  On :meth:`update_cost`, only the changed edges' endpoints are
updated and re-queued, then ``computeShortestPath`` resumes — reusing all prior
``g`` values.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["DStarLite"]

_NEIGH = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
)


class DStarLite:
    """Incremental D* Lite planner over a 2-D cost grid.

    Parameters
    ----------
    cost : np.ndarray
        Initial per-cell cost grid (``np.inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.

    Notes
    -----
    Edge cost between adjacent cells ``u`` and ``v`` is the geometric step length
    times the average of the two cells' traversal costs — identical to the model
    used by :func:`lunaris.planning.astar.astar`, so an initial D* Lite plan and
    an A* plan on the same grid agree on cost.
    """

    def __init__(
        self,
        cost: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> None:
        self.cost = np.asarray(cost, dtype=np.float64).copy()
        self.h, self.w = self.cost.shape
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.km = 0.0

        finite = self.cost[np.isfinite(self.cost)]
        self._min_cost = float(finite.min()) if finite.size else 0.0

        # g and rhs default to +inf (lazily stored; absent == inf).
        self.g: dict[tuple[int, int], float] = {}
        self.rhs: dict[tuple[int, int], float] = {}
        self.U: list[tuple[tuple[float, float], tuple[int, int]]] = []
        self._in_queue: dict[tuple[int, int], tuple[float, float]] = {}

        self.rhs[self.goal] = 0.0
        self._push(self.goal, self._calc_key(self.goal))

    # -- low-level helpers --------------------------------------------------
    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w

    def _g(self, s) -> float:
        return self.g.get(s, math.inf)

    def _rhs(self, s) -> float:
        return self.rhs.get(s, math.inf)

    def _heuristic(self, s) -> float:
        # Octile distance from the (search) start, scaled by the min cell cost
        # so it under-estimates the true average-cost path (admissible).
        dr = abs(s[0] - self.start[0])
        dc = abs(s[1] - self.start[1])
        octile = (max(dr, dc) - min(dr, dc)) + math.sqrt(2.0) * min(dr, dc)
        return self._min_cost * octile

    def _edge(self, u, v) -> float:
        """Cost of traversing from ``u`` to ``v`` (``inf`` if either blocked)."""
        cu = self.cost[u]
        cv = self.cost[v]
        if not (np.isfinite(cu) and np.isfinite(cv)):
            return math.inf
        step = math.hypot(v[0] - u[0], v[1] - u[1])
        return step * 0.5 * (cu + cv)

    def _neighbours(self, s):
        r0, c0 = s
        for dr, dc, _ in _NEIGH:
            r, c = r0 + dr, c0 + dc
            if self._in_bounds(r, c):
                yield (r, c)

    def _calc_key(self, s) -> tuple[float, float]:
        k2 = min(self._g(s), self._rhs(s))
        if not math.isfinite(k2):
            return (math.inf, math.inf)
        return (k2 + self._heuristic(s) + self.km, k2)

    # -- priority queue (lazy-deletion binary heap) ------------------------
    def _push(self, s, key) -> None:
        self._in_queue[s] = key
        heapq.heappush(self.U, (key, s))

    def _top_key(self) -> tuple[float, float]:
        self._clean_top()
        if not self.U:
            return (math.inf, math.inf)
        return self.U[0][0]

    def _clean_top(self) -> None:
        # Drop stale heap entries (those whose stored key no longer matches).
        while self.U:
            key, s = self.U[0]
            cur = self._in_queue.get(s)
            if cur is None or cur != key:
                heapq.heappop(self.U)
            else:
                return

    def _pop(self):
        self._clean_top()
        key, s = heapq.heappop(self.U)
        self._in_queue.pop(s, None)
        return s

    def _update_vertex(self, u) -> None:
        if u != self.goal:
            best = math.inf
            for sp in self._neighbours(u):
                val = self._edge(u, sp) + self._g(sp)
                if val < best:
                    best = val
            self.rhs[u] = best
        # remove any existing queue entry, re-insert if inconsistent
        self._in_queue.pop(u, None)
        if self._g(u) != self._rhs(u):
            self._push(u, self._calc_key(u))

    def _compute_shortest_path(self) -> None:
        start = self.start
        while (
            self._top_key() < self._calc_key(start)
            or self._rhs(start) != self._g(start)
        ):
            if not self.U:
                break
            k_old = self._top_key()
            u = self._pop()
            k_new = self._calc_key(u)
            if k_old < k_new:
                # key was stale (cost grew) -> re-queue with the correct key
                self._push(u, k_new)
            elif self._g(u) > self._rhs(u):
                # over-consistent: relax
                self.g[u] = self._rhs(u)
                for s in self._neighbours(u):
                    self._update_vertex(s)
            else:
                # under-consistent: raise to inf and re-evaluate u and preds
                self.g[u] = math.inf
                self._update_vertex(u)
                for s in self._neighbours(u):
                    self._update_vertex(s)

    # -- public API ---------------------------------------------------------
    def plan(self):
        """Compute / return the current shortest path ``start -> goal``.

        Returns
        -------
        (path, total_cost)
            Ordered ``(row, col)`` cells and accumulated cost, or
            ``(None, inf)`` if the goal is unreachable from the start.
        """
        self._compute_shortest_path()
        if not math.isfinite(self._g(self.start)):
            return None, math.inf

        # Greedy descent: from start, repeatedly step to the successor that
        # minimises edge(s, s') + g(s') until the goal is reached.
        path = [self.start]
        s = self.start
        total = 0.0
        visited = {self.start}
        guard = self.h * self.w + 5
        while s != self.goal and guard > 0:
            guard -= 1
            best = None
            best_val = math.inf
            best_edge = math.inf
            for sp in self._neighbours(s):
                e = self._edge(s, sp)
                val = e + self._g(sp)
                if val < best_val:
                    best_val = val
                    best = sp
                    best_edge = e
            if best is None or not math.isfinite(best_val) or best in visited:
                return None, math.inf
            total += best_edge
            s = best
            visited.add(s)
            path.append(s)
        if s != self.goal:
            return None, math.inf
        return path, total

    def update_cost(self, changes) -> None:
        """Apply edge-cost changes and repair the search tree.

        Parameters
        ----------
        changes : iterable
            Either a mapping ``{(row, col): new_cost}`` or an iterable of
            ``((row, col), new_cost)`` pairs.  Each updates the cost of *entering*
            that cell; the affected vertices (the cell and its neighbours) are
            re-evaluated and :meth:`plan` will reuse all unaffected ``g`` values.
        """
        if hasattr(changes, "items"):
            items = list(changes.items())
        else:
            items = list(changes)

        # NB: with a moving start one would add k_m += h(last_start, start) and
        # reset last_start here. The start is fixed in this implementation, so
        # k_m remains 0; the structure is kept for online-replanning extensions.
        affected: set[tuple[int, int]] = set()
        for cell, new_cost in items:
            cell = (int(cell[0]), int(cell[1]))
            self.cost[cell] = float(new_cost)
            affected.add(cell)
            affected.update(self._neighbours(cell))

        # refresh the cached minimum finite cost (affects the heuristic scale)
        finite = self.cost[np.isfinite(self.cost)]
        self._min_cost = float(finite.min()) if finite.size else 0.0

        for v in affected:
            self._update_vertex(v)
