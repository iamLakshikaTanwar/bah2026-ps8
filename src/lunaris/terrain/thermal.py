"""Thermal cold-trap analysis: cold-trap masks, sublimation rate, ice-stability
depth, and a 1-D regolith thermal profile.

Implemented by: **terrain agent**.

References
----------
* Murphy, D. M. & Koop, T. (2005), "Review of the vapour pressures of ice and
  supercooled water for atmospheric applications", *Q. J. R. Meteorol. Soc.*
  131, 1539-1565 — the ice saturation-vapour-pressure expression
  ``ln p_ice(T)`` used here (valid down to ~110 K, extrapolated below).
* Schorghofer, N. & Williams, J.-P. (2020), "Mapping of ice storage processes
  on the Moon", *J. Geophys. Res. Planets* / Schorghofer (2008), *ApJ* 682,
  697 — Hertz-Knudsen free-sublimation flux and ice-stability burial depth.
* Paige, D. A. et al. (2010), *Science* 330, 479 — Diviner cold-trap
  temperature ceilings (110 K for H2O over Gyr).
* Hayne, P. O. et al. (2017), "Global regolith thermophysical properties of the
  Moon from the Diviner Lunar Radiometer Experiment", *J. Geophys. Res.
  Planets* 122, 2371-2400 — depth- and temperature-dependent regolith thermal
  conductivity ``k(T, z) = k_d [1 + chi (T/350)^3]`` used in the conduction
  solver.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    GAS_CONST,
    GEOTHERMAL_FLUX_WM2,
    H2O_MOLAR_MASS,
    T_WATER_STABLE,
)

__all__ = [
    "cold_trap_mask",
    "vapor_pressure_ice",
    "sublimation_rate",
    "ice_stability_depth",
    "regolith_thermal_profile",
]

# Seconds in 1 Gyr (for converting a mass-loss threshold to a rate).
_SECONDS_PER_GYR = 1.0e9 * 365.25 * 24.0 * 3600.0


def cold_trap_mask(tmax: np.ndarray, threshold: float = T_WATER_STABLE) -> np.ndarray:
    """Boolean water-ice cold-trap mask (Paige et al. 2010).

        cold_trap = tmax < threshold     (default 110 K for H2O, Gyr stability)

    Parameters
    ----------
    tmax : np.ndarray
        Annual maximum surface temperature [K], shape ``(H, W)``.
    threshold : float, default :data:`T_WATER_STABLE` (110 K)
        Stability ceiling [K] (use :data:`T_CO2_STABLE` etc. for other species).

    Returns
    -------
    np.ndarray
        Boolean cold-trap mask, shape ``(H, W)``.
    """
    return np.asarray(tmax, dtype=np.float64) < threshold


def vapor_pressure_ice(T: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water ice (Murphy & Koop 2005).

    Murphy & Koop (2005), eq. (7) for hexagonal ice::

        ln p_ice = 9.550426 - 5723.265 / T + 3.53068 * ln T - 0.00728332 * T

    with ``p_ice`` in pascals and ``T`` in kelvin. Valid for ``T > ~110 K`` and
    smoothly extrapolated to the colder PSR regime.

    Parameters
    ----------
    T : np.ndarray
        Temperature [K].

    Returns
    -------
    np.ndarray
        Ice saturation vapour pressure [Pa].
    """
    T = np.asarray(T, dtype=np.float64)
    Tc = np.maximum(T, 1.0)  # guard against div-by-zero / log of non-positive
    ln_p = (
        9.550426
        - 5723.265 / Tc
        + 3.53068 * np.log(Tc)
        - 0.00728332 * Tc
    )
    return np.exp(ln_p)


def sublimation_rate(T: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Free-space water-ice sublimation mass-loss rate (Hertz-Knudsen).

    The Hertz-Knudsen free-sublimation flux into vacuum (Schorghofer & Williams
    2020; Schorghofer 2008) is

        E = alpha * p_s(T) * sqrt( M / (2 * pi * R * T) )   [kg m^-2 s^-1]

    where ``p_s(T)`` is the ice saturation vapour pressure (Murphy & Koop 2005,
    :func:`vapor_pressure_ice`), ``M = H2O_MOLAR_MASS``, ``R = GAS_CONST`` and
    the sticking/evaporation coefficient ``alpha`` defaults to 1. ``E`` rises
    steeply and monotonically with ``T``.

    Parameters
    ----------
    T : np.ndarray
        Surface temperature [K].
    alpha : float, default 1.0
        Evaporation coefficient [-].

    Returns
    -------
    np.ndarray
        Sublimation rate [kg m^-2 s^-1], same shape as ``T``.
    """
    T = np.asarray(T, dtype=np.float64)
    Tc = np.maximum(T, 1.0)
    p_s = vapor_pressure_ice(Tc)
    return alpha * p_s * np.sqrt(H2O_MOLAR_MASS / (2.0 * np.pi * GAS_CONST * Tc))


def ice_stability_depth(
    tmax: np.ndarray,
    threshold: float = T_WATER_STABLE,
    diffusion_length_m: float = 100e-6,
    loss_threshold_kg_m2_gyr: float = 100.0,
) -> np.ndarray:
    """Burial depth to the ice-stable layer from the thermal regime.

    Where the surface ``tmax`` is at or below ``threshold`` (110 K), ice is
    stable at the surface and the depth is 0. Where it is warmer, surface ice
    sublimes; buried ice survives only below a depth ``z`` at which Fickian
    vapour diffusion through the regolith throttles the surface free-sublimation
    flux below a tolerable Gyr-integrated loss (Schorghofer & Williams 2020):

        E_subsurface ~= (l / z) * E_surface(tmax)
        stable when  E_subsurface <= E_threshold
        => z = l * E_surface(tmax) / E_threshold

    with diffusion length scale ``l = diffusion_length_m`` (~100 um pore scale)
    and ``E_threshold`` the rate equivalent to ``loss_threshold_kg_m2_gyr`` over
    1 Gyr. Returns ``z`` in metres (0 where stable at the surface).

    Parameters
    ----------
    tmax : np.ndarray
        Annual maximum surface temperature [K], shape ``(H, W)``.
    threshold : float, default 110 K
        Surface-stability ceiling [K].
    diffusion_length_m : float, default 100e-6
        Pore-scale vapour diffusion length ``l`` [m].
    loss_threshold_kg_m2_gyr : float, default 100.0
        Tolerable ice loss over 1 Gyr [kg m^-2].

    Returns
    -------
    np.ndarray
        Ice-stability depth [m], shape ``(H, W)`` (0 where surface-stable).
    """
    tmax = np.asarray(tmax, dtype=np.float64)
    e_surface = sublimation_rate(tmax)
    e_thresh = loss_threshold_kg_m2_gyr / _SECONDS_PER_GYR  # kg m^-2 s^-1
    depth = diffusion_length_m * e_surface / e_thresh
    # ice already stable at the surface -> zero burial depth required.
    depth = np.where(tmax <= threshold, 0.0, depth)
    return np.maximum(depth, 0.0)


def regolith_thermal_profile(
    surface_temp: float,
    depth_max: float = 1.0,
    nz: int = 50,
    k_d: float = 3.4e-3,
    chi: float = 2.7,
    geothermal_flux: float = GEOTHERMAL_FLUX_WM2,
) -> tuple[np.ndarray, np.ndarray]:
    """1-D steady-state regolith temperature-vs-depth profile.

    Solves the steady conductive heat equation ``d/dz( k(T) dT/dz ) = 0`` for a
    column with a fixed surface temperature and a basal lunar geothermal flux,
    using the Hayne et al. (2017) temperature-dependent conductivity

        k(T) = k_d * ( 1 + chi * (T / 350)^3 )

    (radiative-in-pores term; ``k_d`` the contact conductivity, ``chi`` the
    radiative-to-contact ratio). Steady state requires a constant heat flux
    ``Q = -k(T) dT/dz = geothermal_flux`` at every depth, so the profile is
    obtained by upward integration from the surface:

        dT/dz = Q / k(T),   T(0) = surface_temp,

    marched with an implicit (backward-Euler in depth) update that is
    unconditionally stable for the chosen ``nz``. With a positive geothermal
    flux the temperature *rises* with depth; the returned column is finite and
    monotonic.

    Parameters
    ----------
    surface_temp : float
        Surface (z=0) temperature [K].
    depth_max : float, default 1.0
        Column depth [m].
    nz : int, default 50
        Number of depth nodes.
    k_d : float, default 3.4e-3
        Contact (dust) thermal conductivity [W m^-1 K^-1] (Hayne 2017 scale).
    chi : float, default 2.7
        Radiative-conduction ratio [-] (Hayne 2017).
    geothermal_flux : float, default :data:`GEOTHERMAL_FLUX_WM2`
        Basal heat flux [W m^-2].

    Returns
    -------
    z : np.ndarray
        Depth grid [m], shape ``(nz,)`` (0 at the surface, ``depth_max`` at base).
    T : np.ndarray
        Temperature [K] at each depth, shape ``(nz,)``.
    """
    nz = max(int(nz), 2)
    z = np.linspace(0.0, depth_max, nz)
    dz = z[1] - z[0]
    T = np.empty(nz, dtype=np.float64)
    T[0] = float(surface_temp)

    def k_of_T(t: float) -> float:
        return k_d * (1.0 + chi * (max(t, 1.0) / 350.0) ** 3)

    # Upward integration: dT/dz = Q / k(T). Backward-Euler in depth (evaluate k
    # at the new node via one fixed-point pass) for stability.
    Q = geothermal_flux
    for i in range(1, nz):
        t_prev = T[i - 1]
        t_new = t_prev + Q / k_of_T(t_prev) * dz  # explicit predictor
        # one implicit corrector using k at the predicted node
        t_new = t_prev + Q / k_of_T(t_new) * dz
        T[i] = t_new
    return z, T
