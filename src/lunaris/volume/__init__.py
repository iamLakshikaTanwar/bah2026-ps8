"""Ice-volume estimation: dielectric mixing models (Looyenga, Maxwell-Garnett),
radar penetration depth, and Monte-Carlo volume/mass with uncertainty.

(Function bodies are filled in by the volume agent.)
"""

from __future__ import annotations

from .dielectric import looyenga_ice_fraction, maxwell_garnett_eps, penetration_depth
from .estimate import ice_mass, ice_volume, monte_carlo_volume

__all__ = [
    "looyenga_ice_fraction",
    "maxwell_garnett_eps",
    "penetration_depth",
    "ice_volume",
    "ice_mass",
    "monte_carlo_volume",
]
