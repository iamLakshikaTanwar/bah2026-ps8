"""Ice-volume estimation: dielectric mixing models (Looyenga, Maxwell-Garnett),
radar penetration depth, and Monte-Carlo volume/mass with uncertainty.

Implemented by the **volume agent**.
"""

from __future__ import annotations

from .dielectric import (
    cpr_to_ice_likelihood,
    looyenga_eps,
    looyenga_ice_fraction,
    maxwell_garnett_eps,
    penetration_depth,
)
from .estimate import (
    estimate_scene_ice,
    ice_mass,
    ice_volume,
    monte_carlo_volume,
)

__all__ = [
    "looyenga_ice_fraction",
    "maxwell_garnett_eps",
    "looyenga_eps",
    "penetration_depth",
    "cpr_to_ice_likelihood",
    "ice_volume",
    "ice_mass",
    "monte_carlo_volume",
    "estimate_scene_ice",
]
