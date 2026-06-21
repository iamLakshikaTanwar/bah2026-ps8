"""lunaris — Lunar Subsurface-Ice Detection & Mission-Planning Platform.

ISRO BAH 2026, Problem Statement 8. Detects subsurface water-ice in a lunar
South-Pole *doubly-shadowed crater* (Faustini PSR) from Chandrayaan-2 DFSAR
radar polarimetry (CPR>1 & DOP<0.13) fused with 30+ multi-sensor datasets,
then plans a landing site, an energy-aware rover traverse, and a top-5 m
ice-volume estimate.

The public surface kept here is intentionally small and stable; deep
functionality lives in the sub-packages (``io``, ``polarimetry``, ``terrain``,
``classify``, ``planning``, ``volume``, ``viz``).
"""

from __future__ import annotations

from . import constants
from .config import Settings, load_config
from .scene import LAYER_NAMES, LunarScene

__version__ = "0.1.0"

__all__ = [
    "LunarScene",
    "LAYER_NAMES",
    "Settings",
    "load_config",
    "constants",
    "__version__",
]
