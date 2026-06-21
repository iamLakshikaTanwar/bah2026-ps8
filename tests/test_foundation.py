"""Foundation tests: package importability, the synthetic-scene contract,
constants, and the dataset registry.

These guard the shared infrastructure that all six downstream agents build on.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.constants import (
    CPR_ICE_THRESHOLD,
    DOP_ICE_THRESHOLD,
    EPS_ICE,
    FAUSTINI_LAT,
    MOON_RADIUS_M,
    eps_regolith,
    loss_tangent_regolith,
)
from lunaris.io.registry import get_dataset, list_datasets
from lunaris.io.synthetic import generate_faustini_scene
from lunaris.scene import LAYER_NAMES, LunarScene


# --------------------------------------------------------------------------
# importability
# --------------------------------------------------------------------------
def test_import_lunaris():
    import lunaris

    assert lunaris.__version__
    assert hasattr(lunaris, "LunarScene")
    assert hasattr(lunaris, "load_config")


def test_subpackages_import():
    # Every sub-package must import (stubs included) so downstream agents can
    # resolve the contract without ImportError.
    import lunaris.classify  # noqa: F401
    import lunaris.planning  # noqa: F401
    import lunaris.polarimetry  # noqa: F401
    import lunaris.terrain  # noqa: F401
    import lunaris.viz  # noqa: F401
    import lunaris.volume  # noqa: F401
    from lunaris import cli, pipeline  # noqa: F401

    assert hasattr(cli, "app")


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
def test_constants_values():
    assert MOON_RADIUS_M == 1737400.0
    assert CPR_ICE_THRESHOLD == 1.0
    assert DOP_ICE_THRESHOLD == 0.13
    assert EPS_ICE == 3.15
    assert FAUSTINI_LAT == -87.3


def test_constants_helpers():
    # eps_regolith(1.8) = 1.919**1.8 ; loss tangent is finite & positive.
    assert eps_regolith(1.8) == pytest.approx(1.919 ** 1.8, rel=1e-9)
    lt = loss_tangent_regolith(1.8, 10.0)
    assert lt > 0.0 and np.isfinite(lt)


# --------------------------------------------------------------------------
# synthetic scene — shapes & layers
# --------------------------------------------------------------------------
def test_scene_shapes(faustini_scene: LunarScene):
    s = faustini_scene
    assert s.shape == (128, 128)
    layers = s.layers()
    # every declared layer present, correct shape, finite.
    for name in LAYER_NAMES:
        arr = layers[name]
        assert arr.shape == (128, 128), name
        assert np.all(np.isfinite(arr.astype(float))), name
    assert s.crs
    assert s.resolution_m == 20.0
    assert s.transform is not None


def test_scene_determinism():
    a = generate_faustini_scene(n=96, seed=7)
    b = generate_faustini_scene(n=96, seed=7)
    for name in LAYER_NAMES:
        assert np.array_equal(a.layers()[name], b.layers()[name]), name
    # different seed -> different radar field
    c = generate_faustini_scene(n=96, seed=8)
    assert not np.array_equal(a.cpr_L, c.cpr_L)


def test_ice_truth_nonempty(faustini_scene: LunarScene):
    assert faustini_scene.ice_truth.dtype == bool
    assert int(faustini_scene.ice_truth.sum()) > 0


# --------------------------------------------------------------------------
# synthetic scene — encodes the ice criterion
# --------------------------------------------------------------------------
def test_ice_criterion_encoded(faustini_scene: LunarScene):
    s = faustini_scene
    ice = s.ice_truth
    bg = ~ice

    # Inside the ice patch the criterion holds (median).
    assert np.median(s.cpr_L[ice]) > CPR_ICE_THRESHOLD
    assert np.median(s.dop_L[ice]) < DOP_ICE_THRESHOLD

    # Background mature regolith is clearly below the CPR threshold.
    assert np.median(s.cpr_L[bg]) < 0.7


def test_threshold_classifier_recovers_truth(faustini_scene: LunarScene):
    # CPR>1 & DOP<0.13 should recover the ice patch with high precision/recall,
    # i.e. the DOP gate rejects the rough-rim CPR>1 decoys.
    s = faustini_scene
    ice = s.ice_truth
    pred = (s.cpr_L > CPR_ICE_THRESHOLD) & (s.dop_L < DOP_ICE_THRESHOLD)
    tp = int((pred & ice).sum())
    fp = int((pred & ~ice).sum())
    fn = int((~pred & ice).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert precision > 0.9
    assert recall > 0.9


def test_decoys_present(faustini_scene: LunarScene):
    # There must exist high-CPR-but-high-DOP decoy pixels OUTSIDE the ice patch
    # (roughness false positives) so the DOP disambiguation actually matters.
    s = faustini_scene
    decoy = (s.cpr_L > CPR_ICE_THRESHOLD) & (s.dop_L > 0.2) & (~s.ice_truth)
    assert int(decoy.sum()) > 0


def test_sband_weaker_than_lband(faustini_scene: LunarScene):
    # S-band ice signature should be a bit weaker (lower CPR) than L-band.
    s = faustini_scene
    ice = s.ice_truth
    assert np.median(s.cpr_S[ice]) <= np.median(s.cpr_L[ice]) + 1e-9


# --------------------------------------------------------------------------
# dataset registry
# --------------------------------------------------------------------------
def test_registry_has_many_datasets():
    keys = list_datasets()
    assert len(keys) >= 20
    # spot-check a few essential sources exist
    for required in ("ch2_dfsar", "ch2_ohrc", "lro_lola", "lro_diviner"):
        assert required in keys
    ds = get_dataset("ch2_dfsar")
    assert ds.mission == "Chandrayaan-2"
    assert ds.access_url.startswith("https://pradan.issdc.gov.in")


def test_registry_unknown_key_raises():
    with pytest.raises(KeyError):
        get_dataset("does_not_exist")
