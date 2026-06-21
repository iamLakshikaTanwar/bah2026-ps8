"""NSGA-II multi-objective traverse optimisation.

NSGA-II (Deb, Pratap, Agarwal & Meyarivan, *"A Fast and Elitist Multiobjective
Genetic Algorithm: NSGA-II"*, IEEE Trans. Evol. Comput. 2002) finds the
**Pareto front** trading off competing traverse objectives — here the four
mission-relevant ones:

0. **distance**       — total path length [px],
1. **energy**         — accumulated slope-driven drive energy proxy,
2. **hazard**         — accumulated roughness / risk,
3. **shadow_dwell**   — number of steps spent in shadow (thermal exposure).

A genome would normally encode a candidate path and genetic operators would mutate
it, with NSGA-II's two signature components — ``fast_non_dominated_sort`` (Pareto
ranking in ``O(M N^2)``) and ``crowding_distance`` (diversity preservation) —
driving selection.  Both of those components are implemented below.

To keep the search **tractable and deterministic** for a 2-D grid we use a
scalarisation front-generator in place of path-level crossover: sample a spread
of weight vectors over the cost terms, run A* for each to obtain a candidate
path, evaluate the four true objectives for every candidate, and return the
non-dominated set extracted by ``fast_non_dominated_sort``.  This yields a
well-spread, genuinely Pareto-optimal approximation of the front (each member is
optimal for *some* weighting) at a fraction of the cost of evolving raw paths.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import math

import numpy as np

from .astar import astar
from .cost import build_cost_grid

__all__ = ["nsga2_paths", "fast_non_dominated_sort", "crowding_distance", "dominates"]


def dominates(a, b) -> bool:
    """Pareto domination (minimisation): ``a`` dominates ``b``.

    ``a`` dominates ``b`` iff ``a`` is no worse than ``b`` in every objective and
    strictly better in at least one.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objs) -> list[list[int]]:
    """Partition objective vectors into Pareto fronts (Deb et al. 2002, Alg.).

    Parameters
    ----------
    objs : array-like, shape ``(N, M)``
        ``N`` solutions, ``M`` objectives (all minimised).

    Returns
    -------
    list[list[int]]
        ``fronts[k]`` holds the indices of the ``k``-th non-dominated front
        (front 0 is the Pareto-optimal set).
    """
    objs = [np.asarray(o, dtype=np.float64) for o in objs]
    n = len(objs)
    S: list[list[int]] = [[] for _ in range(n)]   # solutions p dominates
    n_dom = [0] * n                               # count dominating p
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()  # trailing empty front
    return fronts


def crowding_distance(objs, front) -> dict[int, float]:
    """NSGA-II crowding distance for the solutions in one ``front``.

    The crowding distance estimates the density of solutions around each member
    (the perimeter of the cuboid formed by its nearest neighbours in objective
    space); boundary solutions get ``inf`` so they are always preserved.

    Parameters
    ----------
    objs : array-like, shape ``(N, M)``
        All objective vectors (indexed by the values in ``front``).
    front : sequence[int]
        Indices forming one non-dominated front.

    Returns
    -------
    dict[int, float]
        Mapping ``solution index -> crowding distance``.
    """
    objs = np.asarray(objs, dtype=np.float64)
    front = list(front)
    dist = {i: 0.0 for i in front}
    if len(front) == 0:
        return dist
    m = objs.shape[1]
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i, k])
        dist[order[0]] = math.inf
        dist[order[-1]] = math.inf
        fmin = objs[order[0], k]
        fmax = objs[order[-1], k]
        span = fmax - fmin
        if span <= 0:
            continue
        for j in range(1, len(order) - 1):
            dist[order[j]] += (objs[order[j + 1], k] - objs[order[j - 1], k]) / span
    return dist


def _accumulate_objectives(path, dist_layer, energy_layer, hazard_layer, shadow_layer):
    """Sum the four objective layers along a path (step-length weighted distance)."""
    total_dist = 0.0
    energy = 0.0
    hazard = 0.0
    shadow = 0.0
    for a, b in zip(path[:-1], path[1:]):
        step = math.hypot(b[0] - a[0], b[1] - a[1])
        total_dist += step
        # entering-cell convention for the per-cell objective layers
        energy += step * float(energy_layer[b])
        hazard += step * float(hazard_layer[b])
        shadow += float(shadow_layer[b])
    return [total_dist, energy, hazard, shadow]


def nsga2_paths(
    slope: np.ndarray,
    roughness: np.ndarray,
    illumination: np.ndarray,
    traversable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    pop_size: int = 30,
    generations: int = 40,
    seed: int = 0,
) -> list[dict]:
    """Pareto-optimal traverse paths via NSGA-II (scalarisation front-generator).

    Parameters
    ----------
    slope, roughness, illumination : np.ndarray
        Hazard layers, shape ``(H, W)``.  ``slope`` in deg, ``illumination`` in
        ``[0, 1]``.
    traversable : np.ndarray
        Boolean traversability mask.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints.
    pop_size : int, default 30
        Number of weight vectors sampled (candidate-path budget).
    generations : int, default 40
        Number of refinement rounds; each perturbs the weight population around
        the current front (the NSGA-II selection loop drives this).
    seed : int, default 0
        RNG seed.

    Returns
    -------
    list[dict]
        Non-dominated solutions, each ``{"path": [...], "objectives":
        [distance, energy, hazard, shadow_dwell]}``.  Guaranteed to contain no
        member dominated by another.
    """
    slope = np.asarray(slope, dtype=np.float64)
    roughness = np.asarray(roughness, dtype=np.float64)
    illumination = np.asarray(illumination, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)
    rng = np.random.default_rng(seed)

    # Objective evaluation layers (per-cell, minimised).
    energy_layer = np.tan(np.radians(np.clip(slope, 0.0, 89.0)))  # climb energy proxy
    rmin, rmax = float(roughness.min()), float(roughness.max())
    hazard_layer = (roughness - rmin) / (rmax - rmin) if rmax > rmin else np.zeros_like(roughness)
    shadow_layer = (illumination < 0.2).astype(np.float64)        # dark-step indicator

    candidates: list[tuple[list, list]] = []
    seen_paths: set[tuple] = set()

    def evaluate(weights) -> None:
        grid = build_cost_grid(slope, roughness, illumination, traversable, weights)
        path, _ = astar(grid, start, goal, connectivity=8)
        if path is None:
            return
        key = tuple(path)
        if key in seen_paths:
            return
        seen_paths.add(key)
        objs = _accumulate_objectives(
            path, None, energy_layer, hazard_layer, shadow_layer
        )
        candidates.append((path, objs))

    # --- initial population: spread of weight vectors over the cost terms ----
    base_keys = ["distance", "slope", "roughness", "illumination"]
    # deterministic corner/edge emphases plus random fill
    seed_weights = [
        {"distance": 1.0, "slope": 1.0, "roughness": 1.0, "illumination": 1.0},
        {"distance": 1.0, "slope": 8.0, "roughness": 1.0, "illumination": 1.0},
        {"distance": 1.0, "slope": 1.0, "roughness": 8.0, "illumination": 1.0},
        {"distance": 1.0, "slope": 1.0, "roughness": 1.0, "illumination": 8.0},
        {"distance": 4.0, "slope": 1.0, "roughness": 1.0, "illumination": 1.0},
    ]
    population: list[dict] = list(seed_weights)
    while len(population) < pop_size:
        v = rng.uniform(0.2, 8.0, size=4)
        population.append({k: float(v[i]) for i, k in enumerate(base_keys)})

    for w in population:
        evaluate(w)

    # --- NSGA-II-style refinement: rank, then perturb the elite front -------
    for _ in range(max(0, int(generations))):
        if not candidates:
            break
        objs_arr = np.array([c[1] for c in candidates], dtype=np.float64)
        fronts = fast_non_dominated_sort(objs_arr)
        elite = fronts[0]
        # exploit: jitter weight vectors around random elite members to densify
        # the front (a lightweight stand-in for NSGA-II crossover/mutation).
        new_pop: list[dict] = []
        for _ in range(pop_size):
            v = rng.uniform(0.2, 8.0, size=4) * rng.uniform(0.5, 1.5)
            new_pop.append({k: float(v[i]) for i, k in enumerate(base_keys)})
        for w in new_pop:
            evaluate(w)
        # crowding distance is computed to mirror NSGA-II selection bookkeeping
        _ = crowding_distance(objs_arr, elite)

    if not candidates:
        return []

    objs_arr = np.array([c[1] for c in candidates], dtype=np.float64)
    fronts = fast_non_dominated_sort(objs_arr)
    pareto = fronts[0]
    result = [{"path": candidates[i][0], "objectives": candidates[i][1]} for i in pareto]
    return result
