"""RRT* sampling-based path planner.

RRT* (Karaman & Frazzoli, *"Sampling-based Algorithms for Optimal Motion
Planning"*, Int. J. Robotics Research 2011) augments the Rapidly-exploring
Random Tree with two rewiring steps that make it **asymptotically optimal**:

* *ChooseParent* — a new node is connected not to its nearest tree node but to
  the node within a ball of radius ``r`` that minimises its cost-to-come along a
  collision-free edge;
* *Rewire* — existing nodes inside that ball are re-parented through the new node
  whenever that lowers their cost.

The connection radius shrinks with tree size as

    r = gamma * (log(n) / n) ** (1 / d)

(here ``d = 2`` for the planar grid), the rate that guarantees almost-sure
convergence to the optimum.  Nearest- and near-neighbour queries use a
``scipy.spatial.cKDTree`` rebuilt as the tree grows.

The planner works in continuous ``(row, col)`` coordinates restricted to free
cells (``cost < inf``); an edge is collision-free iff every grid cell it crosses
(sampled with Bresenham) is free.  Edge cost is the Euclidean length times the
mean traversed cell cost, matching the other planners.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .theta_star import bresenham

__all__ = ["rrt_star"]


def _collision_free(cost: np.ndarray, a, b) -> bool:
    """``True`` iff every cell on the Bresenham segment ``a -> b`` is finite-cost."""
    ar, ac = int(round(a[0])), int(round(a[1]))
    br, bc = int(round(b[0])), int(round(b[1]))
    h, w = cost.shape
    for (r, c) in bresenham(ar, ac, br, bc):
        if not (0 <= r < h and 0 <= c < w) or not np.isfinite(cost[r, c]):
            return False
    return True


def _edge_cost(cost: np.ndarray, a, b) -> float:
    """Euclidean length of ``a -> b`` times the mean cost of cells it crosses."""
    ar, ac = int(round(a[0])), int(round(a[1]))
    br, bc = int(round(b[0])), int(round(b[1]))
    cells = bresenham(ar, ac, br, bc)
    vals = np.array([cost[r, c] for (r, c) in cells], dtype=np.float64)
    if not np.all(np.isfinite(vals)):
        return math.inf
    return math.hypot(b[0] - a[0], b[1] - a[1]) * float(vals.mean())


def _steer(frm, to, step: float):
    """Return a point at most ``step`` from ``frm`` in the direction of ``to``."""
    dr = to[0] - frm[0]
    dc = to[1] - frm[1]
    dist = math.hypot(dr, dc)
    if dist <= step or dist == 0.0:
        return (to[0], to[1])
    f = step / dist
    return (frm[0] + dr * f, frm[1] + dc * f)


def rrt_star(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    n_samples: int = 2000,
    step: float = 5.0,
    goal_bias: float = 0.05,
    seed: int = 0,
):
    """Asymptotically-optimal RRT* path over a cost grid.

    Parameters
    ----------
    cost : np.ndarray
        Per-cell cost grid (``np.inf`` = impassable), shape ``(H, W)``.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints (must be free cells).
    n_samples : int, default 2000
        Maximum number of random samples (tree nodes attempted).
    step : float, default 5.0
        Steering step length [px].
    goal_bias : float, default 0.05
        Probability of sampling the goal directly.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    (path, total_cost)
        Ordered ``(row, col)`` waypoints from ``start`` to ``goal`` and the
        accumulated cost, or ``(None, inf)`` if the goal was not connected.
    """
    cost = np.asarray(cost, dtype=np.float64)
    h, w = cost.shape
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    rng = np.random.default_rng(seed)

    si = (int(round(start[0])), int(round(start[1])))
    gi = (int(round(goal[0])), int(round(goal[1])))
    if not (0 <= si[0] < h and 0 <= si[1] < w and np.isfinite(cost[si])):
        return None, math.inf
    if not (0 <= gi[0] < h and 0 <= gi[1] < w and np.isfinite(cost[gi])):
        return None, math.inf

    # Tree storage. nodes[i] = (row, col); parent[i], cost_to_come[i].
    nodes = [start]
    parent = [-1]
    g_cost = [0.0]
    free_min = float(cost[np.isfinite(cost)].min())

    # RRT* radius constant; generous so neighbourhoods are non-trivial.
    gamma = 2.0 * step * math.sqrt(3.0)
    goal_node = -1
    goal_thresh = max(step, 1.5)

    for it in range(int(n_samples)):
        # --- sample (with goal bias) --------------------------------------
        if rng.random() < goal_bias:
            sample = goal
        else:
            sample = (rng.uniform(0, h - 1), rng.uniform(0, w - 1))

        pts = np.asarray(nodes)
        tree = cKDTree(pts)

        # --- nearest -> steer ---------------------------------------------
        _, nn = tree.query(sample)
        nn = int(nn)
        new_pt = _steer(nodes[nn], sample, step)
        ni = (int(round(new_pt[0])), int(round(new_pt[1])))
        if not (0 <= ni[0] < h and 0 <= ni[1] < w) or not np.isfinite(cost[ni]):
            continue
        if not _collision_free(cost, nodes[nn], new_pt):
            continue

        # --- near set & ChooseParent --------------------------------------
        n = len(nodes)
        radius = min(gamma * math.sqrt(math.log(n + 1) / (n + 1)), 5.0 * step)
        near = tree.query_ball_point(new_pt, radius)
        if nn not in near:
            near.append(nn)

        best_parent = nn
        best_cost = g_cost[nn] + _edge_cost(cost, nodes[nn], new_pt)
        for j in near:
            j = int(j)
            ec = _edge_cost(cost, nodes[j], new_pt)
            if not math.isfinite(ec):
                continue
            cand = g_cost[j] + ec
            if cand < best_cost and _collision_free(cost, nodes[j], new_pt):
                best_cost = cand
                best_parent = j
        if not math.isfinite(best_cost):
            continue

        new_idx = len(nodes)
        nodes.append(new_pt)
        parent.append(best_parent)
        g_cost.append(best_cost)

        # --- Rewire neighbours through the new node -----------------------
        for j in near:
            j = int(j)
            if j == best_parent:
                continue
            ec = _edge_cost(cost, new_pt, nodes[j])
            if not math.isfinite(ec):
                continue
            through = best_cost + ec
            if through < g_cost[j] and _collision_free(cost, new_pt, nodes[j]):
                parent[j] = new_idx
                g_cost[j] = through

        # --- goal connection ----------------------------------------------
        d_goal = math.hypot(new_pt[0] - goal[0], new_pt[1] - goal[1])
        if d_goal <= goal_thresh and _collision_free(cost, new_pt, goal):
            cand = best_cost + _edge_cost(cost, new_pt, goal)
            if goal_node == -1:
                nodes.append(goal)
                parent.append(new_idx)
                g_cost.append(cand)
                goal_node = len(nodes) - 1
            elif cand < g_cost[goal_node]:
                parent[goal_node] = new_idx
                g_cost[goal_node] = cand

    if goal_node == -1:
        return None, math.inf

    # reconstruct path from goal back to start
    path_pts = []
    i = goal_node
    guard = len(nodes) + 5
    while i != -1 and guard > 0:
        guard -= 1
        path_pts.append((int(round(nodes[i][0])), int(round(nodes[i][1]))))
        i = parent[i]
    path_pts.reverse()
    return path_pts, float(g_cost[goal_node])
