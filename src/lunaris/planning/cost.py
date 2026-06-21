"""Traversability cost-grid construction for rover path planning.

The planners in this sub-package all operate on a single scalar *cost grid*: a
2-D ``float`` array whose finite entries give the per-cell traversal penalty and
whose ``+inf`` entries mark impassable terrain.  This module turns the raw
hazard layers (slope, surface roughness, illumination) into that grid.

Cost model
----------
For every cell the composite traversal cost is a weighted linear combination of
normalised hazard terms (a standard multi-criteria / weighted-overlay model, cf.
the VIPER traversability cost formulation, Tompkins et al., arXiv:2401.08558)::

    cost = w_d
         + w_s * tan(radians(slope))          # energetic climb penalty
         + w_r * roughness_norm               # ride-quality / hazard penalty
         + w_I * (1 - illumination)           # darkness / thermal penalty

``w_d`` is a constant base distance weight so that, on perfectly flat, smooth,
fully-lit terrain, every step still costs a positive amount (otherwise A*/D*Lite
heuristics degenerate).  A cell is *blocked* (``cost = +inf``) where the slope
exceeds the rover climbing limit (:data:`ROVER_MAX_SLOPE_DEG`) or the supplied
``traversable`` mask is ``False``.

The ``tan(slope)`` term is deliberate: on a slope ``theta`` the extra path length
per unit horizontal advance grows like ``1/cos(theta)`` and the gravitational
work like ``sin(theta)``; ``tan(theta) = sin/cos`` captures the
super-linear blow-up as the rover approaches its climbing limit, matching the
slope-energy model in :mod:`lunaris.planning.energy`.

Implemented by: **planning agent**.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..constants import ROVER_MAX_SLOPE_DEG

__all__ = ["build_cost_grid", "traversability_mask", "DEFAULT_WEIGHTS"]


# Default weighting of the four cost terms.  Tuned so slope dominates (it maps
# directly to drive energy and roll-over risk), illumination is a strong
# secondary driver (thermal survival near the pole), and roughness a tertiary
# ride-quality penalty.  ``distance`` is the constant per-step base cost.
DEFAULT_WEIGHTS: dict[str, float] = {
    "distance": 1.0,
    "slope": 4.0,
    "roughness": 1.0,
    "illumination": 2.0,
}


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a finite array to ``[0, 1]`` (constant -> all zeros).

    Non-finite entries are treated as the array maximum so they map to ``1`` and
    never lower the cost of a blocked cell.
    """
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros_like(a)
    out = (a - lo) / (hi - lo)
    out[~np.isfinite(a)] = 1.0
    return np.clip(out, 0.0, 1.0)


def traversability_mask(
    slope: np.ndarray,
    max_slope: float = ROVER_MAX_SLOPE_DEG,
) -> np.ndarray:
    """Boolean mask of cells the rover is permitted to drive on.

    A cell is traversable iff its terrain slope is below the rover's absolute
    climbing limit and the slope value itself is finite (NaN slopes — e.g. data
    voids — are treated as untraversable).

    Parameters
    ----------
    slope : np.ndarray
        Terrain slope [deg], shape ``(H, W)``.
    max_slope : float, default :data:`ROVER_MAX_SLOPE_DEG`
        Maximum climbable slope [deg].

    Returns
    -------
    np.ndarray
        Boolean array, ``True`` where drivable.
    """
    s = np.asarray(slope, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(s) & (s < float(max_slope))
    return ok


def build_cost_grid(
    slope: np.ndarray,
    roughness: np.ndarray,
    illumination: np.ndarray,
    traversable: np.ndarray,
    weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Combine hazard layers into a per-cell traversal cost grid.

    See the module docstring for the cost model.  Blocked cells (slope above the
    climbing limit *or* ``traversable`` ``False``) are set to ``+inf``.

    Parameters
    ----------
    slope : np.ndarray
        Slope [deg], shape ``(H, W)``.
    roughness : np.ndarray
        Surface roughness (arbitrary units; min-max normalised internally),
        shape ``(H, W)``.
    illumination : np.ndarray
        Illuminated fraction in ``[0, 1]``, shape ``(H, W)``.
    traversable : np.ndarray
        Boolean traversability mask, shape ``(H, W)``.
    weights : mapping, optional
        Keys ``"distance"``, ``"slope"``, ``"roughness"``, ``"illumination"``.
        Missing keys fall back to :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    np.ndarray
        Cost grid (``np.inf`` = impassable), ``float64``, shape ``(H, W)``.
    """
    slope = np.asarray(slope, dtype=np.float64)
    roughness = np.asarray(roughness, dtype=np.float64)
    illumination = np.asarray(illumination, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)

    w = dict(DEFAULT_WEIGHTS)
    if weights is not None:
        w.update(weights)

    # --- normalised hazard terms ------------------------------------------
    # Slope term: tan(theta) clipped at the climbing limit so the term stays
    # finite even for absurd slopes (blocked cells are overwritten below).
    theta = np.radians(np.clip(slope, 0.0, 89.0))
    slope_term = np.tan(theta)
    slope_term = slope_term / (np.tan(np.radians(ROVER_MAX_SLOPE_DEG)) + 1e-9)
    slope_term = np.clip(slope_term, 0.0, 1.0)

    rough_term = _normalise(roughness)

    illum = np.clip(illumination, 0.0, 1.0)
    dark_term = 1.0 - illum

    cost = (
        float(w["distance"])
        + float(w["slope"]) * slope_term
        + float(w["roughness"]) * rough_term
        + float(w["illumination"]) * dark_term
    )

    # --- block impassable cells -------------------------------------------
    blocked = (~traversable) | (slope > ROVER_MAX_SLOPE_DEG) | (~np.isfinite(slope))
    cost = cost.astype(np.float64, copy=True)
    cost[blocked] = np.inf
    return cost
