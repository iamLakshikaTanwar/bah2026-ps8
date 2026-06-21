"""Rover energy & survival modelling.

Couples solar-power generation, slope-dependent drive energy, and battery
survival time into an energy-aware traverse plan.  The numbers feed the
shadow-survival constraint that governs whether the rover can dip into a
permanently-shadowed region (PSR) to sample ice and climb back into sunlight
before its battery is exhausted.

Physics
-------
* **Solar power.** A body-mounted flat array of area ``A`` and efficiency ``eta``
  facing a Sun at elevation ``e`` above the local horizon collects
  ``P = S * A * eta * sin(e)`` (cosine/Lambert projection of the
  beam onto the panel), with ``S = SOLAR_CONSTANT_WM2``.  Near the lunar south
  pole the Sun grazes the horizon (``e`` small), so ``sin(e)`` is tiny and only
  topographic high points stay lit — hence the survival problem.
* **Drive energy.** Per metre of travel on an along-track slope ``theta`` the
  mechanical work is ``E = m g (sin(theta) + Crr cos(theta))`` (gravity term plus
  rolling resistance ``Crr``), in joules per metre.
* **Survival time.** At a constant hotel/hibernation load ``P`` a battery of
  capacity ``E_batt`` lasts ``t = E_batt / P`` hours.

State-augmentation note
-----------------------
The rigorous way to plan into shadow is to search an augmented state
``<cell, state-of-charge>``: a node is the grid cell *and* the discretised
battery SOC on arrival, with edges that debit drive energy and credit solar
recharge, pruning any state whose SOC drops below the survival floor.  That makes
feasibility a hard constraint inside the search.  Here we implement the tractable
*simulate-along-path* surrogate: plan the geometric/energy-optimal path with A*,
then integrate the battery SOC forward along it and report feasibility — adequate
for a fixed start/goal traverse and far cheaper, while documenting the augmented
formulation for the online case.

Implemented by: **planning agent**.
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import (
    MOON_GRAVITY,
    ROVER_BATTERY_WH,
    ROVER_DRIVE_POWER_W,
    ROVER_HIBERNATE_POWER_W,
    ROVER_MASS_KG,
    ROVER_ROLLING_RESISTANCE,
    ROVER_SOLAR_AREA_M2,
    ROVER_SOLAR_EFF,
    ROVER_SPEED_MS,
    ROVER_SURVIVE_SHADOW_H,
    SOLAR_CONSTANT_WM2,
)
from .astar import astar

__all__ = ["solar_power", "drive_energy_per_m", "survival_time_h", "energy_aware_plan"]


def solar_power(
    elev_deg: np.ndarray | float,
    area: float = ROVER_SOLAR_AREA_M2,
    eff: float = ROVER_SOLAR_EFF,
) -> np.ndarray | float:
    """Electrical power from a body-mounted solar array.

        P = SOLAR_CONSTANT_WM2 * area * eff * sin(radians(clip(elev, 0, 90)))

    The clip floors the Sun at the horizon (no negative / below-horizon power)
    and caps it at zenith.  At the pole ``elev`` is a few degrees, so ``P`` is a
    small fraction of the flat-on maximum (the "grazing sun" regime).

    Parameters
    ----------
    elev_deg : np.ndarray or float
        Solar elevation angle above the local horizon [deg].
    area : float, default :data:`ROVER_SOLAR_AREA_M2`
        Array area [m^2].
    eff : float, default :data:`ROVER_SOLAR_EFF`
        Conversion efficiency [-].

    Returns
    -------
    np.ndarray or float
        Electrical power [W] (same type/shape as ``elev_deg``).
    """
    elev = np.clip(np.asarray(elev_deg, dtype=np.float64), 0.0, 90.0)
    p = SOLAR_CONSTANT_WM2 * area * eff * np.sin(np.radians(elev))
    if np.isscalar(elev_deg) or (np.ndim(elev_deg) == 0):
        return float(p)
    return p


def drive_energy_per_m(
    slope_deg: np.ndarray | float,
    mass: float = ROVER_MASS_KG,
    rolling_resistance: float = ROVER_ROLLING_RESISTANCE,
    g: float = MOON_GRAVITY,
) -> np.ndarray | float:
    """Mechanical drive energy per metre travelled on a slope.

        E/m = m g ( sin(slope) + Crr cos(slope) )   [J m^-1]

    Monotonically increasing in ``slope`` over ``[0, 90] deg`` (the ``sin`` term
    grows, the resistance term shrinks more slowly), so climbing always costs
    more than driving on the flat.

    Parameters
    ----------
    slope_deg : np.ndarray or float
        Along-track slope [deg].
    mass : float, default :data:`ROVER_MASS_KG`
        Rover mass [kg].
    rolling_resistance : float, default :data:`ROVER_ROLLING_RESISTANCE`
        Rolling-resistance coefficient ``Crr`` [-].
    g : float, default :data:`MOON_GRAVITY`
        Gravity [m s^-2].

    Returns
    -------
    np.ndarray or float
        Energy per metre [J m^-1].
    """
    theta = np.radians(np.asarray(slope_deg, dtype=np.float64))
    e = mass * g * (np.sin(theta) + rolling_resistance * np.cos(theta))
    if np.isscalar(slope_deg) or (np.ndim(slope_deg) == 0):
        return float(e)
    return e


def survival_time_h(
    battery_wh: float = ROVER_BATTERY_WH,
    load_w: float = ROVER_HIBERNATE_POWER_W,
) -> float:
    """Hours the rover survives on battery at a constant load.

        t = battery_wh / load_w

    Parameters
    ----------
    battery_wh : float, default :data:`ROVER_BATTERY_WH`
        Usable battery capacity [W h].
    load_w : float, default :data:`ROVER_HIBERNATE_POWER_W`
        Power draw [W] (e.g. the hibernate load in shadow).

    Returns
    -------
    float
        Survival time [h].
    """
    if load_w <= 0:
        return math.inf
    return float(battery_wh) / float(load_w)


def energy_aware_plan(
    cost: np.ndarray,
    illumination: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    resolution_m: float,
    slope: np.ndarray | None = None,
    battery_wh: float = ROVER_BATTERY_WH,
    soc_init: float = 1.0,
    soc_floor: float = 0.10,
    illum_threshold: float = 0.2,
    sun_elev_deg: float = 3.0,
    speed_ms: float = ROVER_SPEED_MS,
    drive_power_w: float = ROVER_DRIVE_POWER_W,
    hibernate_power_w: float = ROVER_HIBERNATE_POWER_W,
) -> dict:
    """Plan a sunlit-to-dark traverse that respects the shadow-survival limit.

    An A* path is found on ``cost`` (so it already prefers lit, low-slope, smooth
    terrain), then the battery state-of-charge (SOC) is integrated forward along
    the path: each segment debits drive energy (``drive_power_w`` for the time at
    ``speed_ms``) and, where the cell illumination is at or above
    ``illum_threshold``, credits solar recharge at :func:`solar_power`
    (``sun_elev_deg``).  In shadow the rover is assumed to hibernate, drawing
    ``hibernate_power_w`` while it traverses the dark stretch.

    Feasibility requires that the SOC never drops below ``soc_floor`` and that the
    cumulative time spent in shadow does not exceed
    :data:`ROVER_SURVIVE_SHADOW_H`.

    Parameters
    ----------
    cost : np.ndarray
        Traversal cost grid (``np.inf`` = impassable).
    illumination : np.ndarray
        Illuminated fraction in ``[0, 1]`` on the same grid.
    start, goal : tuple[int, int]
        ``(row, col)`` endpoints; ``start`` should be lit, ``goal`` in shadow.
    resolution_m : float
        Ground sample distance [m] — converts pixel steps to metres.
    slope : np.ndarray, optional
        Slope [deg] grid; if given, drive energy uses the local slope, otherwise
        a flat-drive assumption is used for the hotel-load accounting.
    battery_wh : float, default :data:`ROVER_BATTERY_WH`
        Usable battery capacity [W h].
    soc_init : float, default 1.0
        Initial SOC fraction in ``[0, 1]``.
    soc_floor : float, default 0.10
        Minimum allowed SOC fraction (survival floor).
    illum_threshold : float, default 0.2
        Illumination at/above which the cell counts as "lit" (recharging).
    sun_elev_deg : float, default 3.0
        Sun elevation used for the solar-power estimate [deg].
    speed_ms : float, default :data:`ROVER_SPEED_MS`
        Drive speed [m s^-1].
    drive_power_w, hibernate_power_w : float
        Electrical loads while driving (lit) and hibernating (dark) [W].

    Returns
    -------
    dict
        ``{"path", "soc_profile", "feasible", "max_dark_hours", "energy_Wh",
        "dark_hours", "min_soc"}``.  ``path`` is ``None`` if no route exists.
    """
    cost = np.asarray(cost, dtype=np.float64)
    illumination = np.asarray(illumination, dtype=np.float64)
    cap_wh = float(battery_wh)

    path, _ = astar(cost, start, goal, connectivity=8)
    if path is None:
        return {
            "path": None,
            "soc_profile": [],
            "feasible": False,
            "max_dark_hours": ROVER_SURVIVE_SHADOW_H,
            "energy_Wh": math.inf,
            "dark_hours": math.inf,
            "min_soc": 0.0,
        }

    p_solar_lit = float(solar_power(sun_elev_deg))  # W when illuminated

    soc_wh = soc_init * cap_wh
    soc_profile = [soc_wh / cap_wh]
    dark_hours = 0.0
    energy_drawn_wh = 0.0  # gross electrical energy consumed (drive + hibernate)
    min_soc = soc_profile[0]

    for a, b in zip(path[:-1], path[1:]):
        seg_px = math.hypot(b[0] - a[0], b[1] - a[1])
        seg_m = seg_px * float(resolution_m)
        # time to drive the segment
        dt_s = seg_m / speed_ms if speed_ms > 0 else 0.0
        dt_h = dt_s / 3600.0

        lit = illumination[b] >= illum_threshold

        if lit:
            load_w = drive_power_w
            gen_w = p_solar_lit
        else:
            # hibernating crawl through shadow: low hotel load, no generation
            load_w = hibernate_power_w
            gen_w = 0.0
            dark_hours += dt_h

        # mechanical drive energy (slope-aware) is added on top of the hotel load
        if slope is not None:
            e_drive_j = float(drive_energy_per_m(np.asarray(slope)[b])) * seg_m
        else:
            e_drive_j = 0.0
        e_drive_wh = e_drive_j / 3600.0

        net_wh = (gen_w - load_w) * dt_h - e_drive_wh
        soc_wh = min(cap_wh, soc_wh + net_wh)
        energy_drawn_wh += load_w * dt_h + e_drive_wh
        frac = soc_wh / cap_wh
        soc_profile.append(frac)
        min_soc = min(min_soc, frac)

    feasible = (min_soc >= soc_floor) and (dark_hours <= ROVER_SURVIVE_SHADOW_H)

    return {
        "path": path,
        "soc_profile": soc_profile,
        "feasible": bool(feasible),
        "max_dark_hours": ROVER_SURVIVE_SHADOW_H,
        "dark_hours": float(dark_hours),
        "energy_Wh": float(energy_drawn_wh),
        "min_soc": float(min_soc),
    }
