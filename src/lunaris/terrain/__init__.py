"""Terrain analysis: DEM derivatives, illumination/shadow modelling, thermal
cold-trap stability, and boulder detection.

(Function bodies are filled in by the terrain agent.)
"""

from __future__ import annotations

from .boulders import boulder_density_map, detect_boulders_shadow
from .dem import aspect, hurst_exponent, iqr_roughness, rms_roughness, slope_horn
from .illumination import (
    double_shadow_mask,
    earth_visibility_map,
    horizon_map,
    permanent_shadow_mask,
    sky_view_factor,
)
from .thermal import (
    cold_trap_mask,
    ice_stability_depth,
    regolith_thermal_profile,
    sublimation_rate,
)

__all__ = [
    "slope_horn",
    "aspect",
    "rms_roughness",
    "hurst_exponent",
    "iqr_roughness",
    "horizon_map",
    "permanent_shadow_mask",
    "double_shadow_mask",
    "earth_visibility_map",
    "sky_view_factor",
    "cold_trap_mask",
    "sublimation_rate",
    "ice_stability_depth",
    "regolith_thermal_profile",
    "detect_boulders_shadow",
    "boulder_density_map",
]
