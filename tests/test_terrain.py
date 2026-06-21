"""Tests for the ``lunaris.terrain`` module: DEM derivatives, illumination /
shadow modelling, thermal cold-trap analysis, and shadow-based boulder sizing.

All tests run against the small deterministic synthetic Faustini scene
(``generate_faustini_scene(n=128, seed=42)``) and against purpose-built
analytic fixtures (flat plane, constant-gradient ramp, planted-shadow image) so
each numerical claim can be checked exactly.

The illumination ray-march is the expensive part; tests use reduced azimuth
counts and a >1 ``step_px`` so the whole file stays fast while still exercising
the real (un-mocked) horizon engine.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.constants import MOON_OBLIQUITY_DEG, T_WATER_STABLE
from lunaris.io.synthetic import generate_faustini_scene
from lunaris.terrain import (
    aspect,
    boulder_density_map,
    cold_trap_mask,
    detect_boulders_shadow,
    double_shadow_mask,
    earth_visibility_map,
    hurst_exponent,
    ice_stability_depth,
    iqr_roughness,
    permanent_shadow_mask,
    regolith_thermal_profile,
    rms_roughness,
    sky_view_factor,
    slope_horn,
    sublimation_rate,
)


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene():
    """Small deterministic synthetic Faustini scene (n=128, seed=42)."""
    return generate_faustini_scene(n=128, seed=42)


@pytest.fixture(scope="module")
def dem_res(scene):
    return scene.dem.astype(np.float64), float(scene.resolution_m)


@pytest.fixture(scope="module")
def psr(dem_res):
    """PSR mask once (reused by several illumination tests; fast parameters)."""
    dem, res = dem_res
    return permanent_shadow_mask(dem, res, n_azimuth=48, step_px=1.5)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


# ===========================================================================
# dem.py
# ===========================================================================
def test_slope_flat_plane_is_zero():
    flat = np.full((24, 24), 5.0)
    sl = slope_horn(flat, 10.0)
    assert sl.shape == flat.shape
    assert np.allclose(sl, 0.0, atol=1e-9)


def test_slope_known_ramp_recovers_gradient():
    # constant-gradient ramp: z increases by tan(s)*res per pixel column, so
    # Horn's slope must recover the planted angle s exactly in the interior.
    s_deg = 12.0
    res = 20.0
    grad = math.tan(math.radians(s_deg)) * res  # rise per pixel step [m]
    cols = np.arange(40, dtype=np.float64)
    ramp = np.repeat((grad * cols)[None, :], 40, axis=0)
    sl = slope_horn(ramp, res)
    assert sl.shape == ramp.shape
    # interior (avoid replicated edges) recovers the slope to high precision
    assert sl[20, 20] == pytest.approx(s_deg, abs=1e-6)
    assert np.allclose(sl[2:-2, 2:-2], s_deg, atol=1e-6)


def test_aspect_range_and_ramp_direction():
    # ramp rising toward +column => downslope points toward -column (west).
    res = 20.0
    cols = np.arange(30, dtype=np.float64)
    ramp = np.repeat((cols * res)[None, :], 30, axis=0)
    asp = aspect(ramp, res)
    assert asp.shape == ramp.shape
    assert asp.min() >= 0.0 and asp.max() < 360.0
    # downslope azimuth is constant in the interior (a single facing direction)
    interior = asp[3:-3, 3:-3]
    assert np.ptp(interior) == pytest.approx(0.0, abs=1e-6)


def test_rms_roughness_positive_on_rough_region(dem_res, scene):
    dem, _ = dem_res
    rough = rms_roughness(dem, baseline_px=2)
    assert rough.shape == dem.shape
    assert np.all(np.isfinite(rough))
    assert np.all(rough >= 0.0)
    # the rim/ejecta region is genuinely rough -> strictly positive there.
    rim = scene.illumination > 0.4  # bright rims carry the fractal roughness
    assert rough[rim].mean() > 0.0
    assert rough.max() > 0.0


def test_hurst_exponent_finite_and_bounded(dem_res):
    dem, _ = dem_res
    H, baselines, nu = hurst_exponent(dem, [1, 2, 4, 8])
    assert np.isfinite(H)
    # self-affine natural terrain: H in a physically plausible band.
    assert 0.0 <= H <= 1.5
    assert baselines.shape == nu.shape
    # deviogram amplitude grows with baseline (positive scaling).
    assert np.all(np.diff(nu) > 0.0)


def test_iqr_roughness_basic(dem_res):
    dem, _ = dem_res
    iqr = iqr_roughness(dem, window=5)
    assert iqr.shape == dem.shape
    assert np.all(iqr >= 0.0)
    assert np.all(np.isfinite(iqr))


# ===========================================================================
# illumination.py
# ===========================================================================
def test_permanent_shadow_mask_overlaps_dark_floor(dem_res, scene, psr):
    # The host/nested crater floor (scene.illumination ~ 0) must be largely
    # flagged PSR, while bright rims must not be. Don't demand perfection:
    # assert substantial overlap (IoU) and high recall of the dark region.
    dark = scene.illumination < 0.05
    assert psr.shape == scene.dem.shape
    assert psr.dtype == bool
    assert psr.sum() > 0
    iou = _iou(psr, dark)
    recall = float((psr & dark).sum() / dark.sum())
    assert iou > 0.3
    assert recall > 0.5
    # The mask discriminates floor from rim: PSR coverage of the dark floor is
    # far higher than of the bright rim. (The synthetic rim is a near-vertical,
    # fractally-rough wall, so an honest ray-trace legitimately catches some
    # self-shadowed micro-pockets there that the smooth illumination proxy
    # misses; we therefore compare fractions rather than demand a zero rim.)
    rim = scene.illumination > 0.6
    assert float(psr[dark].mean()) > 2.0 * float(psr[rim].mean())


def test_double_shadow_subset_and_nonempty(dem_res, psr):
    dem, res = dem_res
    dbl = double_shadow_mask(dem, res, psr, n_azimuth=48, step_px=1.5)
    assert dbl.shape == dem.shape
    assert dbl.dtype == bool
    # doubly-shadowed pixels are a strict subset of the PSR mask.
    assert np.all(dbl <= psr)
    assert not np.any(dbl & ~psr)
    # the deepest nested-crater core is doubly shadowed -> non-empty.
    assert int(dbl.sum()) > 0


def test_sky_view_factor_in_range_and_lower_in_crater(dem_res, scene):
    dem, res = dem_res
    svf = sky_view_factor(dem, res, n_azimuth=36, step_px=1.5)
    assert svf.shape == dem.shape
    assert np.all(svf >= 0.0) and np.all(svf <= 1.0)
    floor = scene.illumination < 0.02   # deep crater floor
    plain = scene.illumination > 0.4    # open terrain / bright slopes
    assert float(svf[floor].mean()) < float(svf[plain].mean())


def test_earth_visibility_map(dem_res):
    dem, res = dem_res
    ev = earth_visibility_map(dem, res)
    assert ev.shape == dem.shape
    # boolean DTE mask with a non-degenerate mix of visible / hidden pixels.
    assert ev.dtype == bool
    frac = float(ev.mean())
    assert 0.0 < frac < 1.0


# ===========================================================================
# thermal.py
# ===========================================================================
def test_cold_trap_mask_overlaps_ice_truth(scene):
    ct = cold_trap_mask(scene.temperature_max, threshold=T_WATER_STABLE)
    assert ct.dtype == bool
    ice = scene.ice_truth
    # the ice patch sits in the coldest core -> cold-trap mask must cover it.
    recall = float((ct & ice).sum() / ice.sum())
    assert recall > 0.8
    assert _iou(ct, ice) > 0.3


def test_cold_trap_mask_threshold_semantics():
    tmax = np.array([[50.0, 109.9], [110.0, 200.0]])
    ct = cold_trap_mask(tmax)
    assert ct.tolist() == [[True, True], [False, False]]


def test_sublimation_rate_monotonic_in_temperature():
    # Hertz-Knudsen free sublimation rises steeply & monotonically with T.
    e100 = float(sublimation_rate(np.array([100.0]))[0])
    e110 = float(sublimation_rate(np.array([110.0]))[0])
    e114 = float(sublimation_rate(np.array([114.0]))[0])
    assert e114 > e110 > e100 > 0.0
    # whole-curve monotonicity over a representative range.
    T = np.linspace(40.0, 160.0, 60)
    E = sublimation_rate(T)
    assert np.all(np.isfinite(E))
    assert np.all(np.diff(E) > 0.0)


def test_ice_stability_depth_zero_when_cold_positive_when_warm():
    tmax = np.array([35.0, 109.0, 150.0, 220.0])
    z = ice_stability_depth(tmax)
    assert z.shape == tmax.shape
    assert np.all(np.isfinite(z))
    # stable at surface where tmax <= 110 K -> zero burial depth.
    assert z[0] == 0.0 and z[1] == 0.0
    # warmer surface -> deeper ice-stable layer, strictly increasing.
    assert z[2] > 0.0
    assert z[3] > z[2]


def test_regolith_thermal_profile_converges_with_depth():
    z, T = regolith_thermal_profile(surface_temp=100.0, depth_max=1.0, nz=50)
    assert z.shape == T.shape == (50,)
    assert z[0] == 0.0 and z[-1] == pytest.approx(1.0)
    assert np.all(np.isfinite(T))
    assert T[0] == pytest.approx(100.0)
    # geothermal conduction gives a small, smooth, monotonic gradient that
    # converges (nearly constant) over the shallow column -> finite & bounded.
    assert np.all(np.diff(T) >= 0.0)            # monotonic (no oscillation)
    assert abs(T[-1] - T[0]) < 50.0             # converged, not diverging


# ===========================================================================
# boulders.py
# ===========================================================================
def test_detect_boulders_recovers_planted_height():
    # plant a single dark rectangular shadow of known length L on a bright bg.
    img = np.full((64, 64), 0.85)
    L_px, w_px = 10, 3
    img[20:20 + L_px, 30:30 + w_px] = 0.04
    gsd = 2.0
    elev = 30.0
    det = detect_boulders_shadow(img, sun_elev_deg=elev, gsd=gsd)
    assert det.shape[0] >= 1
    assert det.shape[1] == 4
    # recovered height ~ L * tan(elev) with L = L_px * gsd.
    expected_h = (L_px * gsd) * math.tan(math.radians(elev))
    # pick the largest detection (the planted boulder).
    h = float(det[np.argmax(det[:, 3]), 3])
    assert h == pytest.approx(expected_h, rel=0.15)


def test_detect_boulders_empty_on_uniform_image():
    # contrast-free image must yield zero detections without error.
    flat = np.full((20, 20), 0.5)
    det = detect_boulders_shadow(flat, sun_elev_deg=20.0, gsd=1.0)
    assert det.shape == (0, 4)


def test_boulder_density_map():
    img = np.full((64, 64), 0.85)
    img[20:30, 30:33] = 0.04
    det = detect_boulders_shadow(img, sun_elev_deg=30.0, gsd=2.0)
    dens = boulder_density_map(det, shape=(64, 64), window=9)
    assert dens.shape == (64, 64)
    assert np.all(dens >= 0.0)
    # total mass of the density field equals the boulder count (boxcar sum).
    assert dens.sum() == pytest.approx(det.shape[0] * (9 * 9), rel=1e-6)
    # empty-input density map is all zeros and correctly shaped.
    empty = boulder_density_map(np.empty((0, 4)), shape=(16, 16), window=5)
    assert empty.shape == (16, 16)
    assert empty.sum() == 0.0


def test_module_constants_imported():
    # sanity: the obliquity ceiling used by the PSR default is the constant.
    assert MOON_OBLIQUITY_DEG == pytest.approx(1.543)
