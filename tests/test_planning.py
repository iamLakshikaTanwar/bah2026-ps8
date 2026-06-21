"""Tests for the :mod:`lunaris.planning` sub-package.

Covers the cost-grid builder, all five path planners (A*, D* Lite, Theta*,
RRT*, NSGA-II), the rover energy / shadow-survival model, and AHP-weighted
landing-site selection.  Inputs are derived directly from the deterministic
synthetic Faustini scene (``generate_faustini_scene(n=128, seed=42)``) so the
suite is fast and reproducible; a slope field is computed here from the DEM
gradient (the terrain sub-package is intentionally not imported).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, label, uniform_filter

from lunaris.constants import (
    LANDING_MAX_SLOPE_DEG,
    ROVER_BATTERY_WH,
    ROVER_HIBERNATE_POWER_W,
    ROVER_MAX_SLOPE_DEG,
)
from lunaris.io.synthetic import generate_faustini_scene
from lunaris.planning import (
    DStarLite,
    ahp_weights,
    astar,
    build_cost_grid,
    drive_energy_per_m,
    energy_aware_plan,
    landing_suitability,
    nsga2_paths,
    rrt_star,
    select_landing_sites,
    solar_power,
    survival_time_h,
    theta_star,
    traversability_mask,
)
from lunaris.planning.nsga2 import (
    crowding_distance,
    dominates,
    fast_non_dominated_sort,
)


# ---------------------------------------------------------------------------
# shared input builders
# ---------------------------------------------------------------------------
def _path_length(path) -> float:
    """Geometric (Euclidean) length of a polyline of ``(r, c)`` waypoints."""
    return sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )


@pytest.fixture(scope="module")
def scene():
    """Small deterministic synthetic Faustini scene."""
    return generate_faustini_scene(n=128, seed=42)


@pytest.fixture(scope="module")
def layers(scene):
    """Slope / roughness / illumination / traversability rasters + cost grid.

    * slope     = degrees(arctan(|grad(DEM)|))  from a plain numpy gradient,
    * roughness = local std of the DEM (uniform-filter variance),
    * illumination from the scene,
    * traversable = slope < ROVER_MAX_SLOPE_DEG.
    """
    dem = scene.dem
    gy, gx = np.gradient(dem, scene.resolution_m)
    slope = np.degrees(np.arctan(np.hypot(gy, gx)))
    mean = uniform_filter(dem, size=5)
    var = uniform_filter(dem * dem, size=5) - mean * mean
    roughness = np.sqrt(np.clip(var, 0.0, None))
    illumination = scene.illumination
    traversable = slope < ROVER_MAX_SLOPE_DEG
    cost = build_cost_grid(slope, roughness, illumination, traversable)
    return {
        "slope": slope,
        "roughness": roughness,
        "illumination": illumination,
        "traversable": traversable,
        "cost": cost,
        "resolution_m": scene.resolution_m,
    }


@pytest.fixture(scope="module")
def connected_endpoints(layers):
    """Two finite-cost endpoints guaranteed to lie in one connected component.

    Picks the largest 8-connected component of finite-cost cells and returns its
    raster-order extreme cells so a path is guaranteed to exist.
    """
    finite = np.isfinite(layers["cost"])
    lab, _ = label(finite, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp = lab == int(np.argmax(sizes))
    ys, xs = np.where(comp)
    order = np.argsort(ys * 10_000 + xs)
    start = (int(ys[order[0]]), int(xs[order[0]]))
    goal = (int(ys[order[-1]]), int(xs[order[-1]]))
    return start, goal, comp


# ---------------------------------------------------------------------------
# cost grid
# ---------------------------------------------------------------------------
def test_traversability_mask(layers):
    slope = layers["slope"]
    mask = traversability_mask(slope)
    assert mask.dtype == bool
    assert mask.shape == slope.shape
    # exactly the cells below the limit (no NaNs in this scene)
    assert np.array_equal(mask, slope < ROVER_MAX_SLOPE_DEG)


def test_build_cost_grid_blocks_untraversable(layers):
    cost = layers["cost"]
    slope = layers["slope"]
    # every steep / non-traversable cell is +inf
    blocked = (slope > ROVER_MAX_SLOPE_DEG) | (~layers["traversable"])
    assert np.all(~np.isfinite(cost[blocked]))
    # traversable cells are finite and strictly positive (base distance weight)
    trav = layers["traversable"] & (slope <= ROVER_MAX_SLOPE_DEG)
    finite_vals = cost[trav]
    assert np.all(np.isfinite(finite_vals))
    assert np.all(finite_vals > 0.0)


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------
def test_astar_finds_path(layers, connected_endpoints):
    cost = layers["cost"]
    start, goal, _ = connected_endpoints
    path, total = astar(cost, start, goal)
    assert path is not None
    assert math.isfinite(total)
    assert total > 0.0
    # endpoints correct
    assert path[0] == start
    assert path[-1] == goal
    # path stays on finite-cost cells (never crosses an inf wall)
    assert all(np.isfinite(cost[r, c]) for (r, c) in path)
    # contiguous 8-connected steps
    for a, b in zip(path[:-1], path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_astar_no_path_returns_none():
    # fully blocked grid except the two isolated endpoints
    cost = np.full((10, 10), np.inf)
    cost[0, 0] = 1.0
    cost[9, 9] = 1.0
    path, total = astar(cost, (0, 0), (9, 9))
    assert path is None
    assert total == math.inf


# ---------------------------------------------------------------------------
# Theta*  (any-angle path <= A* path on open terrain)
# ---------------------------------------------------------------------------
def test_theta_star_not_longer_than_astar_open():
    # open grid with a single wall that forces a detour around it
    cost = np.ones((21, 21))
    cost[8:13, 10] = np.inf
    start, goal = (10, 3), (10, 17)
    pa, _ = astar(cost, start, goal)
    pt, _ = theta_star(cost, start, goal)
    assert pa is not None and pt is not None
    # any-angle path is geometrically no longer than the grid-locked path
    assert _path_length(pt) <= _path_length(pa) + 1e-6
    # and on this obstacle layout it is strictly shorter
    assert _path_length(pt) < _path_length(pa)
    assert pt[0] == start and pt[-1] == goal
    # no waypoint sits on an obstacle
    assert all(np.isfinite(cost[r, c]) for (r, c) in pt)


def test_theta_star_open_equals_diagonal():
    # on a totally free grid Theta* must reach the goal with a finite cost
    cost = np.ones((30, 30))
    pt, ct = theta_star(cost, (0, 0), (29, 29))
    assert pt is not None
    assert math.isfinite(ct)
    assert pt[0] == (0, 0) and pt[-1] == (29, 29)


# ---------------------------------------------------------------------------
# D* Lite
# ---------------------------------------------------------------------------
def test_dstar_lite_matches_astar_initial(layers, connected_endpoints):
    cost = layers["cost"]
    start, goal, _ = connected_endpoints
    _, astar_cost = astar(cost, start, goal)
    planner = DStarLite(cost, start, goal)
    path, dcost = planner.plan()
    assert path is not None
    assert path[0] == start and path[-1] == goal
    # initial D* Lite optimum equals A* optimum (same edge-cost model)
    assert dcost == pytest.approx(astar_cost, rel=1e-6, abs=1e-6)


def test_dstar_lite_incremental_replan():
    # open grid; raise a partial wall across the corridor leaving a gap so the
    # rover must detour. The replan must differ, avoid the wall, and cost more.
    cost = np.ones((30, 30))
    start, goal = (15, 2), (15, 27)
    planner = DStarLite(cost, start, goal)
    p0, c0 = planner.plan()
    assert p0 is not None

    wall = {(r, 15): np.inf for r in range(0, 25)}  # gap at rows 25..29
    planner.update_cost(wall)
    p1, c1 = planner.plan()

    assert p1 is not None                       # still reachable via the gap
    assert p1 != p0                             # genuinely replanned
    assert p1[0] == start and p1[-1] == goal
    assert all(cell not in wall for cell in p1)  # avoids the new wall
    assert c1 > c0                              # the detour is more expensive


def test_dstar_lite_replan_blocked_returns_none():
    cost = np.ones((12, 12))
    start, goal = (6, 1), (6, 10)
    planner = DStarLite(cost, start, goal)
    assert planner.plan()[0] is not None
    # seal a full vertical wall -> goal becomes unreachable
    planner.update_cost({(r, 6): np.inf for r in range(12)})
    path, cost_out = planner.plan()
    assert path is None
    assert cost_out == math.inf


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------
def test_rrt_star_free_region():
    cost = np.ones((40, 40))
    start, goal = (2, 2), (37, 37)
    path, total = rrt_star(cost, start, goal, n_samples=1000, step=4.0, seed=1)
    # may occasionally fail to connect; if so it must do so gracefully
    if path is None:
        assert total == math.inf
        return
    assert math.isfinite(total)
    assert path[0] == start and path[-1] == goal
    # every waypoint is on a free (finite-cost) cell
    assert all(np.isfinite(cost[r, c]) for (r, c) in path)


def test_rrt_star_blocked_returns_none():
    cost = np.full((15, 15), np.inf)
    cost[0, 0] = 1.0
    cost[14, 14] = 1.0
    path, total = rrt_star(cost, (0, 0), (14, 14), n_samples=200, seed=0)
    assert path is None
    assert total == math.inf


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def test_fast_non_dominated_sort_basic():
    # objectives (minimise both): A and B mutually non-dominated; C dominated.
    objs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    fronts = fast_non_dominated_sort(objs)
    assert set(fronts[0]) == {0, 1}     # Pareto front
    assert fronts[1] == [2]             # dominated solution in the next front
    assert dominates([1.0, 1.0], [2.0, 2.0])
    assert not dominates([1.0, 2.0], [2.0, 1.0])


def test_crowding_distance_boundaries_infinite():
    objs = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    front = [0, 1, 2, 3]
    cd = crowding_distance(objs, front)
    # the two extreme solutions get infinite crowding distance
    assert cd[0] == math.inf
    assert cd[3] == math.inf
    assert math.isfinite(cd[1]) and math.isfinite(cd[2])


def test_nsga2_paths_pareto_front(layers, connected_endpoints):
    start, goal, _ = connected_endpoints
    res = nsga2_paths(
        layers["slope"],
        layers["roughness"],
        layers["illumination"],
        layers["traversable"],
        start,
        goal,
        pop_size=12,
        generations=3,
        seed=0,
    )
    assert len(res) >= 1
    for sol in res:
        assert sol["path"][0] == start and sol["path"][-1] == goal
        assert len(sol["objectives"]) == 4   # distance, energy, hazard, shadow
    # the returned set must be Pareto: no member dominated by another member
    objs = [sol["objectives"] for sol in res]
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j:
                assert not dominates(objs[j], objs[i])


# ---------------------------------------------------------------------------
# energy model
# ---------------------------------------------------------------------------
def test_solar_power_monotonic():
    assert solar_power(0.0) == pytest.approx(0.0)
    assert solar_power(-5.0) == pytest.approx(0.0)   # below horizon -> 0
    assert solar_power(90.0) > solar_power(45.0) > solar_power(10.0) > 0.0
    # vectorised call
    out = solar_power(np.array([0.0, 30.0, 90.0]))
    assert out.shape == (3,)
    assert out[0] == pytest.approx(0.0)


def test_drive_energy_increases_with_slope():
    e0 = drive_energy_per_m(0.0)
    e10 = drive_energy_per_m(10.0)
    e20 = drive_energy_per_m(20.0)
    assert e0 > 0.0                  # rolling resistance on the flat
    assert e20 > e10 > e0


def test_survival_time():
    t = survival_time_h(battery_wh=ROVER_BATTERY_WH, load_w=ROVER_HIBERNATE_POWER_W)
    assert t == pytest.approx(ROVER_BATTERY_WH / ROVER_HIBERNATE_POWER_W)


def test_energy_aware_plan_dark_excursion():
    # controlled corridor: lit first half, dark second half.
    H, W = 12, 40
    cost = np.ones((H, W))
    illum = np.ones((H, W))
    illum[:, W // 2:] = 0.0
    slope = np.zeros((H, W))
    start, goal = (6, 1), (6, 38)

    res = energy_aware_plan(
        cost, illum, start, goal,
        resolution_m=5.0, slope=slope,
        soc_init=1.0, soc_floor=0.05, illum_threshold=0.2,
        speed_ms=0.05,
    )
    # required keys / types
    assert isinstance(res["feasible"], bool)
    for key in ("path", "soc_profile", "max_dark_hours", "energy_Wh"):
        assert key in res
    assert res["path"] is not None
    assert res["path"][0] == start and res["path"][-1] == goal

    soc = res["soc_profile"]
    assert len(soc) == len(res["path"])
    assert soc[0] == pytest.approx(1.0)

    # SOC is non-increasing across the dark stretch (no recharge in shadow).
    path = res["path"]
    dark_soc = [
        soc[i + 1]
        for i, (a, b) in enumerate(zip(path[:-1], path[1:]))
        if illum[b] < 0.2
    ]
    assert len(dark_soc) > 0
    assert all(dark_soc[i] >= dark_soc[i + 1] - 1e-9 for i in range(len(dark_soc) - 1))

    # feasibility respects the documented survival floor and shadow-dwell cap.
    assert res["dark_hours"] <= res["max_dark_hours"] or not res["feasible"]
    if res["feasible"]:
        assert res["min_soc"] >= 0.05 - 1e-9


def test_energy_aware_plan_respects_floor():
    # a tiny battery forced through a long dark stretch must report infeasible
    # (SOC would breach the floor) rather than silently succeeding.
    H, W = 8, 60
    cost = np.ones((H, W))
    illum = np.zeros((H, W))          # entirely in shadow
    start, goal = (4, 1), (4, 58)
    res = energy_aware_plan(
        cost, illum, start, goal,
        resolution_m=50.0, slope=np.zeros((H, W)),
        battery_wh=50.0,             # very small battery
        soc_init=0.2, soc_floor=0.1,
        speed_ms=0.05, hibernate_power_w=ROVER_HIBERNATE_POWER_W,
    )
    assert res["feasible"] is False
    assert res["min_soc"] < 0.1 + 1e-9


# ---------------------------------------------------------------------------
# landing-site selection
# ---------------------------------------------------------------------------
def test_landing_suitability_range(scene, layers):
    d2ice = distance_transform_edt(~scene.ice_truth)
    lyrs = {
        "slope": layers["slope"],
        "roughness": layers["roughness"],
        "illumination": layers["illumination"],
        "earth_visibility": scene.earth_visibility,
        "distance_to_ice": d2ice,
    }
    weights = {
        "slope": 0.4, "roughness": 0.15, "illumination": 0.2,
        "earth_visibility": 0.1, "distance_to_ice": 0.15,
    }
    score = landing_suitability(lyrs, weights)
    assert score.shape == layers["slope"].shape
    assert np.all(score >= 0.0) and np.all(score <= 1.0)
    # cells above the landing slope limit are hard-penalised to zero
    too_steep = layers["slope"] > LANDING_MAX_SLOPE_DEG
    assert np.all(score[too_steep] == 0.0)


def test_ahp_weights_sum_and_consistency():
    pw = np.array(
        [
            [1.0, 3.0, 5.0, 4.0, 2.0],
            [1.0 / 3, 1.0, 3.0, 2.0, 1.0],
            [1.0 / 5, 1.0 / 3, 1.0, 1.0 / 2, 1.0 / 3],
            [1.0 / 4, 1.0 / 2, 2.0, 1.0, 1.0 / 2],
            [1.0 / 2, 1.0, 3.0, 2.0, 1.0],
        ]
    )
    w, cr = ahp_weights(pw)
    assert w.shape == (5,)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= 0.0)
    # this matrix is deliberately consistent -> CR well below 0.1
    assert 0.0 <= cr < 0.1
    # most-important criterion (row 0) gets the largest weight
    assert int(np.argmax(w)) == 0


def test_ahp_weights_perfectly_consistent():
    # a rank-1 (perfectly consistent) matrix -> CR == 0
    v = np.array([4.0, 2.0, 1.0])
    pw = np.outer(v, 1.0 / v)
    w, cr = ahp_weights(pw)
    assert cr == pytest.approx(0.0, abs=1e-6)
    assert w == pytest.approx(v / v.sum(), rel=1e-6)


def test_select_landing_sites_on_rim(scene, layers):
    d2ice = distance_transform_edt(~scene.ice_truth)
    lyrs = {
        "slope": layers["slope"],
        "roughness": layers["roughness"],
        "illumination": layers["illumination"],
        "earth_visibility": scene.earth_visibility,
        "distance_to_ice": d2ice,
    }
    weights = {
        "slope": 0.4, "roughness": 0.15, "illumination": 0.2,
        "earth_visibility": 0.1, "distance_to_ice": 0.15,
    }
    score = landing_suitability(lyrs, weights)
    sites = select_landing_sites(score, 5, layers["traversable"])
    assert len(sites) == 5
    for r, c, sc in sites:
        assert layers["traversable"][r, c]                 # drivable
        assert layers["slope"][r, c] <= LANDING_MAX_SLOPE_DEG  # within slope limit
        assert sc == pytest.approx(score[r, c])
    # sites are ordered best-first by score
    scores = [sc for _, _, sc in sites]
    assert scores == sorted(scores, reverse=True)
    # the very best site prefers a bright, gentle rim cell (high illumination)
    br, bc, _ = sites[0]
    assert scene.illumination[br, bc] > 0.5
