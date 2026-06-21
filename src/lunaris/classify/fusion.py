"""Supervised multi-sensor ice classification (feature stack + model).

Implemented by: **classify agent**.

Stacks every co-registered evidence layer of a :class:`~lunaris.scene.LunarScene`
into a per-pixel feature matrix and trains a random-forest classifier to predict
ice probability, fusing radar (CPR/DOP, m-chi volume), thermal, illumination and
optical-frost evidence. Where the O(1) LUT applies the hard physical rule, this
module is the learned, multi-sensor corroboration that hardens the result
against rock/roughness false positives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..scene import LunarScene

__all__ = [
    "build_feature_stack",
    "train_ice_classifier",
    "predict_ice",
    "feature_importance",
    "bake_lut_from_model",
]


# Canonical feature order. Kept as a module constant so callers (and
# :func:`bake_lut_from_model`) can locate the CPR/DOP columns deterministically.
FEATURE_NAMES: list[str] = [
    "cpr_L",
    "dop_L",
    "cpr_S",
    "dop_S",
    "mchi_volume",
    "temperature_max",
    "illumination",
    "albedo_1064",
    "lamp_ratio",
    "earth_visibility",
]


def _mchi_volume(s0: np.ndarray, dop: np.ndarray) -> np.ndarray:
    """m-chi *volume* (depolarised) power component.

    The m-chi decomposition splits total power ``s0`` into a polarised fraction
    ``m = DOP`` and a depolarised remainder ``(1 - m)``. The depolarised power,
    ``sqrt(s0 * (1 - DOP))``, is the classic randomly-polarised / volume-scatter
    return that dominates over coherent subsurface ice (and is the discriminator
    behind CPR>1). Computed inline here so this module does not depend on the
    ``polarimetry`` subpackage.
    """
    return np.sqrt(np.clip(s0 * (1.0 - dop), 0.0, None))


def build_feature_stack(scene: LunarScene) -> tuple[np.ndarray, list[str]]:
    """Assemble the per-pixel feature matrix from a scene.

    Stacks radar (``cpr_L``, ``dop_L``, ``cpr_S``, ``dop_S`` and the inline
    m-chi volume ``sqrt(clip(s0_L * (1 - dop_L), 0, None))``), thermal
    (``temperature_max``), illumination, and optical-frost (``albedo_1064``,
    ``lamp_ratio``) layers, plus ``earth_visibility``, into a column matrix.
    Non-finite values (NaN/inf) are replaced with finite numbers so any
    downstream estimator trains cleanly.

    Parameters
    ----------
    scene : LunarScene
        Source scene of shape ``(H, W)``.

    Returns
    -------
    (X, names) : tuple[np.ndarray, list[str]]
        ``X`` has shape ``(H * W, n_features)`` in the order :data:`FEATURE_NAMES`;
        ``names`` is a fresh copy of that ordering.
    """
    mchi = _mchi_volume(np.asarray(scene.s0_L), np.asarray(scene.dop_L))
    columns = {
        "cpr_L": scene.cpr_L,
        "dop_L": scene.dop_L,
        "cpr_S": scene.cpr_S,
        "dop_S": scene.dop_S,
        "mchi_volume": mchi,
        "temperature_max": scene.temperature_max,
        "illumination": scene.illumination,
        "albedo_1064": scene.albedo_1064,
        "lamp_ratio": scene.lamp_ratio,
        "earth_visibility": scene.earth_visibility,
    }
    names = list(FEATURE_NAMES)
    stacked = [np.asarray(columns[name], dtype=np.float64).ravel() for name in names]
    X = np.column_stack(stacked)
    # Defend against any NaN/inf leaking in from upstream products.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def train_ice_classifier(
    X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> Any:
    """Fit a balanced random-forest ice classifier.

    Uses :class:`sklearn.ensemble.RandomForestClassifier` with
    ``n_estimators=200`` and ``class_weight="balanced"`` (subsurface ice is a
    small fraction of any scene, so balancing prevents the majority "no-ice"
    class from swamping the trees).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Binary ice labels, shape ``(n_samples,)`` (``scene.ice_truth.ravel()``).
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        A fitted estimator exposing ``predict`` / ``predict_proba`` /
        ``feature_importances_``.
    """
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(np.asarray(X), np.asarray(y).ravel())
    return model


def predict_ice(
    model: Any, X: np.ndarray, shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predict ice labels and ice probability with a trained model.

    Parameters
    ----------
    model : Any
        Estimator from :func:`train_ice_classifier`.
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    shape : tuple[int, int], optional
        If given, ``labels`` and ``proba`` are reshaped to this ``(H, W)`` raster
        shape; otherwise both are returned flat with shape ``(n_samples,)``.

    Returns
    -------
    (labels, proba) : tuple[np.ndarray, np.ndarray]
        ``labels`` is the boolean ice prediction; ``proba`` is ``P(ice)`` in
        ``[0, 1]``. Both are 1-D unless ``shape`` is supplied.
    """
    X = np.asarray(X)
    labels = model.predict(X).astype(bool)

    # Locate the column of predict_proba that corresponds to the positive
    # (ice == True) class; robust to single-class training data.
    classes = list(model.classes_)
    proba_all = model.predict_proba(X)
    if True in classes:
        pos = classes.index(True)
        proba = proba_all[:, pos]
    elif 1 in classes:
        proba = proba_all[:, classes.index(1)]
    elif len(classes) == 1:
        # Degenerate: model saw a single class. Probability is that class value.
        proba = np.full(X.shape[0], float(bool(classes[0])))
    else:  # pragma: no cover - defensive
        proba = proba_all[:, -1]

    if shape is not None:
        labels = labels.reshape(shape)
        proba = proba.reshape(shape)
    return labels, proba


def feature_importance(model: Any, names: list[str]) -> list[tuple[str, float]]:
    """Return ``(feature_name, importance)`` pairs sorted most-important first.

    Reads the random forest's mean impurity-decrease importances and pairs them
    with the supplied column names.

    Parameters
    ----------
    model : Any
        A fitted estimator exposing ``feature_importances_``.
    names : list[str]
        Column names in the same order used to train ``model``.

    Returns
    -------
    list[tuple[str, float]]
        Descending-importance ``(name, importance)`` tuples.
    """
    importances = np.asarray(model.feature_importances_, dtype=float)
    pairs = list(zip(names, importances.tolist(), strict=True))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs


def bake_lut_from_model(
    model: Any,
    cpr_edges: np.ndarray,
    dop_edges: np.ndarray,
    fixed_features: dict[str, float] | None = None,
) -> np.ndarray:
    """Collapse a trained model into an O(1) ``(CPR, DOP)`` LUT.

    Demonstrates how the learned, multi-feature classifier can be *baked* into
    the same flat lookup table the rule-based path uses: evaluate the model once
    on the Cartesian grid of CPR/DOP bin centres (all other features held at
    representative constants) and store the boolean predictions. Thereafter a
    scene is classified by the very same O(1) digitize+gather as
    :func:`~lunaris.classify.ice_lut.classify_ice_lut` — no model inference at
    runtime.

    Parameters
    ----------
    model : Any
        Estimator trained on the :data:`FEATURE_NAMES` column order.
    cpr_edges, dop_edges : np.ndarray
        Bin edges, shapes ``(nc + 1,)`` / ``(nd + 1,)``.
    fixed_features : dict[str, float], optional
        Constant values for every non-CPR/DOP feature (``cpr_S``, ``dop_S`` are
        tied to the L-band centres unless overridden). Missing features default
        to ``0.0``.

    Returns
    -------
    np.ndarray
        Boolean LUT, shape ``(nc, nd)``, ``True`` where the model predicts ice.
    """
    cpr_edges = np.asarray(cpr_edges, dtype=np.float64)
    dop_edges = np.asarray(dop_edges, dtype=np.float64)
    cpr_centers = 0.5 * (cpr_edges[:-1] + cpr_edges[1:])
    dop_centers = 0.5 * (dop_edges[:-1] + dop_edges[1:])
    nc, nd = cpr_centers.size, dop_centers.size

    fixed = dict(fixed_features or {})

    # Build the full feature grid (nc*nd rows) in canonical column order.
    cc, dd = np.meshgrid(cpr_centers, dop_centers, indexing="ij")
    cc = cc.ravel()
    dd = dd.ravel()
    n_rows = cc.size

    grid_columns: dict[str, np.ndarray] = {
        "cpr_L": cc,
        "dop_L": dd,
        # tie S-band to L-band unless the caller pins it explicitly
        "cpr_S": np.full(n_rows, fixed.get("cpr_S", np.nan)),
        "dop_S": np.full(n_rows, fixed.get("dop_S", np.nan)),
        "mchi_volume": np.full(n_rows, fixed.get("mchi_volume", 0.0)),
        "temperature_max": np.full(n_rows, fixed.get("temperature_max", 0.0)),
        "illumination": np.full(n_rows, fixed.get("illumination", 0.0)),
        "albedo_1064": np.full(n_rows, fixed.get("albedo_1064", 0.0)),
        "lamp_ratio": np.full(n_rows, fixed.get("lamp_ratio", 0.0)),
        "earth_visibility": np.full(n_rows, fixed.get("earth_visibility", 0.0)),
    }
    # Default S-band to the L-band centres where the caller did not override.
    if np.isnan(grid_columns["cpr_S"]).all():
        grid_columns["cpr_S"] = cc
    if np.isnan(grid_columns["dop_S"]).all():
        grid_columns["dop_S"] = dd

    grid = np.column_stack([grid_columns[name] for name in FEATURE_NAMES])
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

    preds = model.predict(grid).astype(bool)
    lut = preds.reshape(nc, nd)
    return np.ascontiguousarray(lut)
