"""Supervised multi-sensor ice classification (feature stack + model).

Implemented by: **classify agent**.

Stacks every co-registered evidence layer of a :class:`~lunaris.scene.LunarScene`
into a per-pixel feature matrix and trains a classifier (e.g. random forest /
gradient boosting) to predict ice probability, fusing radar, thermal,
illumination and optical-frost evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..scene import LunarScene

__all__ = ["build_feature_stack", "train_ice_classifier", "predict_ice"]


def build_feature_stack(scene: LunarScene) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-pixel feature matrix from a scene.

    Stacks radar (cpr_L, dop_L, cpr_S, dop_S, Stokes-derived), thermal
    (temperature_max), illumination, and optical-frost (albedo_1064,
    lamp_ratio) layers, plus any terrain derivatives present in ``scene.meta``.

    Parameters
    ----------
    scene : LunarScene
        Source scene of shape ``(H, W)``.

    Returns
    -------
    (X, names) : tuple[np.ndarray, list[str]]
        ``X`` has shape ``(H*W, n_features)``; ``names`` lists the columns.
    """
    raise NotImplementedError("classify agent")


def train_ice_classifier(X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a supervised ice classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary ice labels, shape ``(n_samples,)``.

    Returns
    -------
    Any
        A fitted scikit-learn-compatible estimator with ``predict_proba``.
    """
    raise NotImplementedError("classify agent")


def predict_ice(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict per-sample ice probability with a trained model.

    Parameters
    ----------
    model : Any
        Estimator from :func:`train_ice_classifier`.
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.

    Returns
    -------
    np.ndarray
        Ice probability in ``[0, 1]``, shape ``(n_samples,)``.
    """
    raise NotImplementedError("classify agent")
