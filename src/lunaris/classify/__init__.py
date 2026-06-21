"""Ice classification: the O(1) precomputed CPR-DOP LUT, supervised
multi-sensor fusion, and Bayesian / Dempster-Shafer evidence combination.

(Function bodies are filled in by the classify agent.)
"""

from __future__ import annotations

from .evidence import bayesian_fusion, dempster_shafer
from .fusion import build_feature_stack, predict_ice, train_ice_classifier
from .ice_lut import build_ice_lut, classify_ice_lut, classify_ice_threshold

__all__ = [
    "build_ice_lut",
    "classify_ice_lut",
    "classify_ice_threshold",
    "build_feature_stack",
    "train_ice_classifier",
    "predict_ice",
    "bayesian_fusion",
    "dempster_shafer",
]
