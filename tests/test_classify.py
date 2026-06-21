"""Tests for the ``lunaris.classify`` subpackage.

Covers the flagship O(1) CPR-DOP look-up-table classifier (exactness vs the
direct rule, detection quality vs ground truth, the rock/roughness false-positive
advantage of DOP, and the constant-time micro-benchmark), the supervised
random-forest multi-sensor fusion, and the Bayesian / Dempster-Shafer evidence
combination.

All tests run against the deterministic synthetic Faustini scene
(``generate_faustini_scene(n=128, seed=42)``) so results are reproducible.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from lunaris.classify import (
    bake_lut_from_model,
    bayesian_fusion,
    build_feature_stack,
    build_ice_lut,
    classify_ice_lut,
    classify_ice_threshold,
    default_edges,
    dempster_shafer,
    feature_importance,
    multi_evidence_ice,
    predict_ice,
    train_ice_classifier,
)
from lunaris.constants import CPR_ICE_THRESHOLD, DOP_ICE_THRESHOLD
from lunaris.io.synthetic import generate_faustini_scene


# Radar columns we expect a sensible ice classifier to lean on.
_RADAR_FEATURES = {"cpr_L", "dop_L", "cpr_S", "dop_S", "mchi_volume"}


@pytest.fixture(scope="module")
def scene():
    """Deterministic synthetic scene shared across the module's tests."""
    return generate_faustini_scene(n=128, seed=42)


def _confusion(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` counts for two boolean masks."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# LUT exactness
# ---------------------------------------------------------------------------
def test_lut_matches_threshold_exactly(scene):
    """The precomputed LUT reproduces the direct CPR&DOP rule bit-for-bit."""
    cpr_edges, dop_edges = default_edges()
    lut = build_ice_lut(cpr_edges, dop_edges)

    ref = classify_ice_threshold(scene.cpr_L, scene.dop_L)
    lut_mask = classify_ice_lut(scene.cpr_L, scene.dop_L, lut, cpr_edges, dop_edges)

    assert lut_mask.dtype == bool
    assert lut_mask.shape == scene.cpr_L.shape
    # Exactness is the whole point of baking the rule into the table.
    assert np.array_equal(lut_mask, ref)


def test_build_ice_lut_bakes_the_rule():
    """LUT cell truth == rule evaluated at the bin centre (sanity on baking)."""
    cpr_edges, dop_edges = default_edges(nbins=64)
    lut = build_ice_lut(cpr_edges, dop_edges)

    cpr_centers = 0.5 * (cpr_edges[:-1] + cpr_edges[1:])
    dop_centers = 0.5 * (dop_edges[:-1] + dop_edges[1:])
    expected = (cpr_centers[:, None] > CPR_ICE_THRESHOLD) & (
        dop_centers[None, :] < DOP_ICE_THRESHOLD
    )
    assert np.array_equal(lut, expected)


def test_classify_ice_lut_handles_non_uniform_edges(scene):
    """Non-uniform edges fall back to digitize and stay correct."""
    cpr_edges = np.array([0.0, 0.5, 1.0, 1.0001, 1.5, 2.0, 4.0])
    dop_edges = np.array([0.0, 0.1, 0.13, 0.1301, 0.3, 0.6, 1.0])
    lut = build_ice_lut(cpr_edges, dop_edges)
    got = classify_ice_lut(scene.cpr_L, scene.dop_L, lut, cpr_edges, dop_edges)

    nc, nd = lut.shape
    ci = np.clip(np.digitize(scene.cpr_L, cpr_edges) - 1, 0, nc - 1)
    di = np.clip(np.digitize(scene.dop_L, dop_edges) - 1, 0, nd - 1)
    assert np.array_equal(got, lut[ci, di])


# ---------------------------------------------------------------------------
# Detection quality + DOP false-positive rejection
# ---------------------------------------------------------------------------
def test_detection_precision_and_recall(scene):
    """CPR&DOP achieves precision > 0.9 AND recall > 0.9 vs ground truth."""
    pred = classify_ice_threshold(scene.cpr_L, scene.dop_L)
    tp, fp, fn, _ = _confusion(pred, scene.ice_truth)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\nCPR&DOP detection: precision={precision:.4f} recall={recall:.4f} "
          f"(tp={tp} fp={fp} fn={fn})")

    assert precision > 0.9
    assert recall > 0.9


def test_cpr_only_has_more_false_positives(scene):
    """CPR-only thresholding admits many more rock/roughness false positives.

    The synthetic scene seeds rough-rim "decoy" pixels with CPR>1 *and* high DOP;
    the DOP cut rejects them, so CPR&DOP has strictly fewer false positives than
    CPR alone. This is the multi-parameter advantage the platform claims.
    """
    cpr_only = scene.cpr_L > CPR_ICE_THRESHOLD
    cpr_dop = classify_ice_threshold(scene.cpr_L, scene.dop_L)

    _, fp_cpr_only, _, _ = _confusion(cpr_only, scene.ice_truth)
    _, fp_cpr_dop, _, _ = _confusion(cpr_dop, scene.ice_truth)

    print(f"\nFalse positives: CPR-only={fp_cpr_only}  CPR&DOP={fp_cpr_dop}")
    assert fp_cpr_only > fp_cpr_dop
    # The DOP cut should remove essentially all of them.
    assert fp_cpr_dop < fp_cpr_only // 2


# ---------------------------------------------------------------------------
# O(1) micro-benchmark
# ---------------------------------------------------------------------------
def test_lut_classify_is_fast():
    """Classifying a large (2000x2000) array is O(1)/pixel and well under 0.5 s.

    The random inputs are allocated *outside* the timed region so the
    measurement reflects only the digitize+gather classification cost. The bound
    is deliberately generous (0.5 s vs a typical ~0.1 s) to avoid CI flakiness.
    """
    cpr_edges, dop_edges = default_edges()
    lut = build_ice_lut(cpr_edges, dop_edges)

    rng = np.random.default_rng(0)
    big_cpr = rng.uniform(0.0, 3.0, size=(2000, 2000))
    big_dop = rng.uniform(0.0, 1.0, size=(2000, 2000))

    # Warm up (first call may pay one-off allocation / dispatch costs).
    classify_ice_lut(big_cpr, big_dop, lut, cpr_edges, dop_edges)

    t0 = time.perf_counter()
    mask = classify_ice_lut(big_cpr, big_dop, lut, cpr_edges, dop_edges)
    elapsed = time.perf_counter() - t0

    print(f"\nO(1) LUT benchmark: classified {big_cpr.size:,} pixels in "
          f"{elapsed * 1e3:.1f} ms")
    assert mask.shape == big_cpr.shape
    assert elapsed < 0.5


# ---------------------------------------------------------------------------
# Supervised random-forest fusion
# ---------------------------------------------------------------------------
def test_random_forest_cross_val_accuracy(scene):
    """5-fold CV accuracy > 0.9 for the multi-sensor random forest."""
    from sklearn.model_selection import cross_val_score

    X, names = build_feature_stack(scene)
    y = scene.ice_truth.ravel()

    assert X.shape[0] == scene.ice_truth.size
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all()  # NaN/inf handling

    model = train_ice_classifier(X, y, random_state=42)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"\nRandom-forest 5-fold CV accuracy: mean={scores.mean():.4f} "
          f"min={scores.min():.4f}")
    assert scores.mean() > 0.9


def test_feature_importance_ranks_radar_highly(scene):
    """Radar (CPR/DOP/volume) features rank among the most important."""
    X, names = build_feature_stack(scene)
    model = train_ice_classifier(X, scene.ice_truth.ravel(), random_state=42)

    ranking = feature_importance(model, names)
    assert [n for n, _ in ranking] and len(ranking) == len(names)
    # sorted descending
    importances = [imp for _, imp in ranking]
    assert importances == sorted(importances, reverse=True)

    order = [name for name, _ in ranking]
    best_radar_rank = min(order.index(f) for f in _RADAR_FEATURES)
    radar_total = sum(imp for name, imp in ranking if name in _RADAR_FEATURES)
    print(f"\nTop features: {[(n, round(v, 3)) for n, v in ranking[:5]]}")
    print(f"best radar rank={best_radar_rank} radar_total_importance={radar_total:.3f}")

    # At least one radar feature lands in the top half of the ranking, and the
    # radar group as a whole carries substantial importance.
    assert best_radar_rank < len(names) // 2
    assert radar_total > 0.2


def test_predict_ice_shapes_and_range(scene):
    """predict_ice returns boolean labels and [0, 1] probabilities, reshapable."""
    X, _ = build_feature_stack(scene)
    model = train_ice_classifier(X, scene.ice_truth.ravel(), random_state=42)

    labels, proba = predict_ice(model, X)
    assert labels.dtype == bool
    assert labels.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0],)
    assert np.isfinite(proba).all()
    assert proba.min() >= 0.0 and proba.max() <= 1.0

    labels2d, proba2d = predict_ice(model, X, shape=scene.shape)
    assert labels2d.shape == scene.shape
    assert proba2d.shape == scene.shape


def test_bake_lut_from_model_runs(scene):
    """A trained model collapses into a boolean (CPR, DOP) LUT for O(1) reuse."""
    X, _ = build_feature_stack(scene)
    model = train_ice_classifier(X, scene.ice_truth.ravel(), random_state=42)

    cpr_edges, dop_edges = default_edges(nbins=64)
    lut = bake_lut_from_model(model, cpr_edges, dop_edges)
    assert lut.shape == (cpr_edges.size - 1, dop_edges.size - 1)
    assert lut.dtype == bool
    # The model learned an ice region, so the baked table must contain some.
    assert lut.any()


# ---------------------------------------------------------------------------
# Evidence fusion
# ---------------------------------------------------------------------------
def test_bayesian_fusion_in_range(scene):
    """Bayesian fusion returns a finite probability map in [0, 1]."""
    layer_a = multi_evidence_ice(scene)
    layer_b = (scene.albedo_1064 - scene.albedo_1064.min()) / (
        np.ptp(scene.albedo_1064) + 1e-9
    )
    fused = bayesian_fusion([layer_a, layer_b], weights=[1.5, 0.5])

    assert fused.shape == scene.shape
    assert np.isfinite(fused).all()
    assert fused.min() >= 0.0 and fused.max() <= 1.0


def test_bayesian_fusion_default_weights(scene):
    """Bayesian fusion works with default (equal) weights."""
    layer = multi_evidence_ice(scene)
    fused = bayesian_fusion([layer, layer])
    assert np.isfinite(fused).all()
    assert fused.min() >= 0.0 and fused.max() <= 1.0


def test_dempster_shafer_in_range(scene):
    """Dempster-Shafer combination returns a finite belief map in [0, 1]."""
    radar = (scene.cpr_L > CPR_ICE_THRESHOLD) & (scene.dop_L < DOP_ICE_THRESHOLD)
    thermal = scene.temperature_max < 110.0
    belief = dempster_shafer(
        [radar.astype(float), thermal.astype(float)], masses=[0.9, 0.8]
    )

    assert belief.shape == scene.shape
    assert np.isfinite(belief).all()
    assert belief.min() >= 0.0 and belief.max() <= 1.0
    # Corroborating evidence should raise belief inside the true ice region.
    assert belief[scene.ice_truth].mean() > belief[~scene.ice_truth].mean()


def test_dempster_shafer_default_masses(scene):
    """Dempster-Shafer works with default masses and a single source."""
    radar = (scene.cpr_L > CPR_ICE_THRESHOLD) & (scene.dop_L < DOP_ICE_THRESHOLD)
    belief = dempster_shafer([radar.astype(float)])
    assert np.isfinite(belief).all()
    assert belief.min() >= 0.0 and belief.max() <= 1.0


def test_multi_evidence_ice_separates_classes(scene):
    """Fused confidence is higher inside the ice truth than outside."""
    conf = multi_evidence_ice(scene)

    assert conf.shape == scene.shape
    assert np.isfinite(conf).all()
    assert conf.min() >= 0.0 and conf.max() <= 1.0

    mean_in = float(conf[scene.ice_truth].mean())
    mean_out = float(conf[~scene.ice_truth].mean())
    print(f"\nmulti_evidence_ice: mean confidence inside={mean_in:.4f} "
          f"outside={mean_out:.4f}")
    assert mean_in > mean_out
