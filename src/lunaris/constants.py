"""Physical, instrument, and mission constants for the ``lunaris`` platform.

This is the single source of truth for every numeric constant used across the
package. Downstream modules MUST import from here rather than hard-coding
values, so that a single review of this file validates the physics of the
whole pipeline.

References
----------
* Sinha et al. (2026), *npj Space Exploration* — refined CPR>1 & DOP<0.13
  subsurface-ice criterion for lunar PSRs.
* Fa & Wieczorek (2012), *Icarus* — regolith dielectric / density relations.
* Heggy et al. (2012); Carrier, Olhoeft & Mendell (1991) — loss-tangent model.
* Colaprete et al. (2010), *Science* — LCROSS water abundance (5.6 +/- 2.9 wt%).
* Paige et al. (2010), *Science* — Diviner PSR temperatures / cold-trap species.

All values are SI unless the symbol name states otherwise (``_KM``, ``_DEG``,
``_GCM3``, ``_WH``, ``_PCT``, ``_H`` suffixes carry their unit).
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Moon — bulk body parameters
# ---------------------------------------------------------------------------
MOON_RADIUS_M: float = 1737400.0          # volumetric mean radius [m]
MOON_GRAVITY: float = 1.62                 # surface gravity [m s^-2]
MOON_OBLIQUITY_DEG: float = 1.543          # obliquity to the ecliptic [deg]
GEOTHERMAL_FLUX_WM2: float = 0.018         # mean heat flux [W m^-2]
SOLAR_CONSTANT_WM2: float = 1361.0         # solar irradiance at 1 AU [W m^-2]
STEFAN_BOLTZMANN: float = 5.670374419e-8   # sigma [W m^-2 K^-4]

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
# Lunar south-polar stereographic on a sphere of radius MOON_RADIUS_M.
SOUTH_POLAR_STEREO_PROJ4: str = (
    "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+R=1737400 +units=m +no_defs +type=crs"
)
# ESRI authority code for the equivalent Moon 2000 south-polar stereographic.
SOUTH_POLAR_STEREO_ESRI: str = "ESRI:103878"

# ---------------------------------------------------------------------------
# Chandrayaan-2 DFSAR (Dual-Frequency Synthetic Aperture Radar)
# ---------------------------------------------------------------------------
L_BAND_FREQ_HZ: float = 1.25e9             # L-band centre frequency [Hz]
L_BAND_WAVELENGTH_M: float = 0.2399        # L-band wavelength [m]
S_BAND_FREQ_HZ: float = 2.5e9              # S-band centre frequency [Hz]
S_BAND_WAVELENGTH_M: float = 0.1199        # S-band wavelength [m]
DFSAR_INCIDENCE_DEG: tuple[float, float] = (9.6, 36.9)   # incidence-angle range [deg]
DFSAR_RES_M: tuple[float, float] = (2.0, 75.0)           # resolution range (slant..scan) [m]

# ---------------------------------------------------------------------------
# Subsurface-ice detection criterion (Sinha et al. 2026)
# ---------------------------------------------------------------------------
# Ice candidate iff CPR > CPR_ICE_THRESHOLD  AND  DOP < DOP_ICE_THRESHOLD.
CPR_ICE_THRESHOLD: float = 1.0             # circular polarisation ratio [-]
DOP_ICE_THRESHOLD: float = 0.13            # degree of polarisation [-]

# ---------------------------------------------------------------------------
# Dielectric properties (radar volume / depth modelling)
# ---------------------------------------------------------------------------
EPS_ICE: float = 3.15                      # real permittivity of water ice [-]
TAN_DELTA_ICE: float = 0.005               # loss tangent of cold water ice [-]
RHO_ICE_KGM3: float = 920.0                # density of water ice [kg m^-3]
EPS_REGOLITH_DEFAULT: float = 3.0          # default regolith permittivity [-]
RHO_REGOLITH_GCM3: float = 1.8             # default regolith bulk density [g cm^-3]


def eps_regolith(density_gcm3: float) -> float:
    """Real permittivity of lunar regolith from bulk density.

    Empirical relation (Olhoeft & Strangway 1975; Fa & Wieczorek 2012):

        eps' = 1.919 ** rho

    Parameters
    ----------
    density_gcm3 : float
        Bulk density of the regolith [g cm^-3].

    Returns
    -------
    float
        Real part of the relative permittivity [-].
    """
    return 1.919 ** density_gcm3


def loss_tangent_regolith(density: float, feo_tio2_pct: float) -> float:
    """Loss tangent of lunar regolith (Carrier, Olhoeft & Mendell 1991).

        tan(delta) = 10 ** (0.038 * (%FeO + %TiO2) + 0.312 * rho - 3.260)

    Parameters
    ----------
    density : float
        Bulk density [g cm^-3].
    feo_tio2_pct : float
        Combined weight percent of FeO + TiO2 [%].

    Returns
    -------
    float
        Loss tangent tan(delta) [-].
    """
    return 10.0 ** (0.038 * feo_tio2_pct + 0.312 * density - 3.260)


# ---------------------------------------------------------------------------
# Thermal cold-trap stability thresholds (max-temperature ceilings) [K]
# ---------------------------------------------------------------------------
T_WATER_STABLE: float = 110.0              # H2O ice stable for Gyr below this [K]
T_CO2_STABLE: float = 60.0                 # CO2 ice stability [K]
T_NH3_STABLE: float = 66.0                 # NH3 ice stability [K]
T_SO2_STABLE: float = 60.0                 # SO2 ice stability [K]
T_SUPERVOLATILE: float = 40.0             # super-volatile (CO, N2, CH4) trap [K]

H2O_MOLAR_MASS: float = 0.018015           # molar mass of water [kg mol^-1]
GAS_CONST: float = 8.314                   # universal gas constant [J mol^-1 K^-1]

# ---------------------------------------------------------------------------
# Rover (Pragyan-class chassis with VIPER-class planning defaults)
# ---------------------------------------------------------------------------
ROVER_MASS_KG: float = 27.0                # rover mass [kg]
ROVER_SPEED_MS: float = 0.05               # nominal drive speed [m s^-1]
ROVER_DRIVE_POWER_W: float = 110.0         # power while driving [W]
ROVER_IDLE_POWER_W: float = 80.0           # power while idle (awake) [W]
ROVER_HIBERNATE_POWER_W: float = 30.0      # power while hibernating [W]
ROVER_BATTERY_WH: float = 7000.0           # usable battery capacity [W h]
ROVER_SOLAR_AREA_M2: float = 1.5           # solar-panel area [m^2]
ROVER_SOLAR_EFF: float = 0.30              # solar conversion efficiency [-]
ROVER_MAX_SLOPE_DEG: float = 20.0          # absolute climbing limit [deg]
ROVER_SAFE_SLOPE_DEG: float = 15.0         # planning-safe slope limit [deg]
ROVER_SURVIVE_SHADOW_H: float = 70.0       # max survivable shadow dwell [h]
ROVER_ROLLING_RESISTANCE: float = 0.2      # rolling-resistance coefficient [-]
LANDING_MAX_SLOPE_DEG: float = 10.0        # landing-pad slope limit [deg]

# ---------------------------------------------------------------------------
# Target — Faustini crater & its nested doubly-shadowed crater
# ---------------------------------------------------------------------------
FAUSTINI_LAT: float = -87.3                # crater-centre latitude [deg]
FAUSTINI_LON: float = 77.0                 # crater-centre longitude [deg E]
FAUSTINI_DIAMETER_KM: float = 39.0         # host-crater diameter [km]
FAUSTINI_FLOOR_ELEV_M: float = -2700.0     # floor elevation rel. datum [m]

# ---------------------------------------------------------------------------
# Ground-truth references for reporting / validation
# ---------------------------------------------------------------------------
# LCROSS Centaur-impact ejecta water abundance (mean, std) in weight percent.
LCROSS_WT_PCT: tuple[float, float] = (5.6, 2.9)

# ---------------------------------------------------------------------------
# Derived convenience constants
# ---------------------------------------------------------------------------
DEG2RAD: float = math.pi / 180.0
RAD2DEG: float = 180.0 / math.pi

__all__ = [name for name in dir() if not name.startswith("_") and name != "math"]
