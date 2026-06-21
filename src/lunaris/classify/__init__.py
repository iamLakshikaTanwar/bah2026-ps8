"""Ice classification: the O(1) precomputed CPR-DOP LUT, supervised
multi-sensor fusion, and Bayesian / Dempster-Shafer evidence combination.
"""

from __future__ import annotations

from .evidence import bayesian_fusion, dempster_shafer, multi_evidence_ice
from .fusion import (
    bake_lut_from_model,
    build_feature_stack,
    feature_importance,
    predict_ice,
    train_ice_classifier,
)
from .ice_lut import (
    build_ice_lut,
    classify_ice_lut,
    classify_ice_threshold,
    default_edges,
)

__all__ = [
    "build_ice_lut",
    "classify_ice_lut",
    "classify_ice_threshold",
    "default_edges",
    "build_feature_stack",
    "train_ice_classifier",
    "predict_ice",
    "feature_importance",
    "bake_lut_from_model",
    "bayesian_fusion",
    "dempster_shafer",
    "multi_evidence_ice",
]
