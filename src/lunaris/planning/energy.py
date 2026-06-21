"""Rover energy & survival modelling.

Implemented by: **planning agent**.

Couples solar-power generation, slope-dependent drive energy, and battery
survival time into an energy-aware traverse plan.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    MOON_GRAVITY,
    ROVER_BATTERY_WH,
    ROVER_MASS_KG,
    ROVER_ROLLING_RESISTANCE,
    ROVER_SOLAR_AREA_M2,
    ROVER_SOLAR_EFF,
)

__all__ = ["solar_power", "drive_energy_per_m", "survival_time_h", "energy_aware_plan"]


def solar_power(
    elev_deg: np.ndarray | float,
    area: float = ROVER_SOLAR_AREA_M2,
    eff: float = ROVER_SOLAR_EFF,
) -> np.ndarray | float:
    """Electrical power from a body-mounted solar array.

        P = S * area * eff * max(sin(elev), 0)

    with ``S = SOLAR_CONSTANT_WM2``.

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
        Electrical power [W].
    """
    raise NotImplementedError("planning agent")


def drive_energy_per_m(
    slope_deg: np.ndarray | float,
    mass: float = ROVER_MASS_KG,
    rolling_resistance: float = ROVER_ROLLING_RESISTANCE,
    g: float = MOON_GRAVITY,
) -> np.ndarray | float:
    """Mechanical drive energy per metre travelled on a slope.

        E/m = m g ( sin(slope) + Crr cos(slope) )   [J m^-1]

    Parameters
    ----------
    slope_deg : np.ndarray or float
        Along-track slope [deg].
    mass : float, default :data:`ROVER_MASS_KG`
        Rover mass [kg].
    rolling_resistance : float, default :data:`ROVER_ROLLING_RESISTANCE`
        Rolling-resistance coefficient [-].
    g : float, default :data:`MOON_GRAVITY`
        Gravity [m s^-2].

    Returns
    -------
    np.ndarray or float
        Energy per metre [J m^-1].
    """
    raise NotImplementedError("planning agent")


def survival_time_h(
    battery_wh: float = ROVER_BATTERY_WH,
    load_w: float = 30.0,
) -> float:
    """Hours the rover survives on battery at a constant load.

        t = battery_wh / load_w

    Parameters
    ----------
    battery_wh : float, default :data:`ROVER_BATTERY_WH`
        Usable battery capacity [W h].
    load_w : float, default 30.0
        Power draw [W] (e.g. hibernate load in shadow).

    Returns
    -------
    float
        Survival time [h].
    """
    raise NotImplementedError("planning agent")


def energy_aware_plan(*args, **kwargs):
    """Plan a traverse that respects battery / solar / shadow-dwell limits.

    Combines a path planner with the energy model so the rover never depletes
    its battery beyond the survivable shadow-dwell limit, scheduling charging
    stops in illuminated waypoints.
    """
    raise NotImplementedError("planning agent")
