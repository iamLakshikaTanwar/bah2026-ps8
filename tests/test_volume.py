"""Tests for the ``lunaris.volume`` ice-volume estimation subpackage.

Covers the dielectric mixing models (Looyenga-Landau-Lifshitz round-trip,
Maxwell-Garnett monotonicity / bounds), radar penetration depth, the
volume/mass arithmetic, Monte-Carlo uncertainty propagation, and the end-to-end
scene estimate with its LCROSS gravimetric cross-check.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.constants import (
    EPS_ICE,
    EPS_REGOLITH_DEFAULT,
    L_BAND_WAVELENGTH_M,
    RHO_ICE_KGM3,
    S_BAND_WAVELENGTH_M,
)
from lunaris.scene import LunarScene
from lunaris.volume import (
    cpr_to_ice_likelihood,
    estimate_scene_ice,
    ice_mass,
    ice_volume,
    looyenga_eps,
    looyenga_ice_fraction,
    maxwell_garnett_eps,
    monte_carlo_volume,
    penetration_depth,
)


# ---------------------------------------------------------------------------
# Looyenga-Landau-Lifshitz mixing (inverse consistency)
# ---------------------------------------------------------------------------
def test_looyenga_round_trip():
    # forward (looyenga_eps) then inverse (looyenga_ice_fraction) recovers f.
    f = 0.3
    eps_eff = looyenga_eps(f, EPS_ICE, EPS_REGOLITH_DEFAULT)
    recovered = looyenga_ice_fraction(eps_eff, EPS_ICE, EPS_REGOLITH_DEFAULT)
    assert recovered == pytest.approx(f, rel=1e-6)


def test_looyenga_endpoints():
    # f = 0 when eps_eff == eps_reg ; f = 1 when eps_eff == eps_ice.
    assert looyenga_ice_fraction(EPS_REGOLITH_DEFAULT) == pytest.approx(0.0, abs=1e-9)
    assert looyenga_ice_fraction(EPS_ICE) == pytest.approx(1.0, abs=1e-9)


def test_looyenga_fraction_clipped():
    # Below-regolith and above-ice permittivities clip into [0, 1].
    assert looyenga_ice_fraction(2.0) == 0.0
    assert looyenga_ice_fraction(10.0) == 1.0


def test_looyenga_round_trip_array():
    f = np.array([0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    eps_eff = looyenga_eps(f)
    recovered = looyenga_ice_fraction(eps_eff)
    assert np.allclose(recovered, f, atol=1e-9)


# ---------------------------------------------------------------------------
# Maxwell-Garnett mixing (forward)
# ---------------------------------------------------------------------------
def test_maxwell_garnett_monotonic_and_bounded():
    f = np.linspace(0.0, 1.0, 21)
    eps = maxwell_garnett_eps(f, EPS_ICE, EPS_REGOLITH_DEFAULT)
    # strictly increasing in f_ice
    assert np.all(np.diff(eps) > 0.0)
    # bounded by [eps_host, eps_ice], hitting the endpoints exactly
    assert np.all(eps >= EPS_REGOLITH_DEFAULT - 1e-12)
    assert np.all(eps <= EPS_ICE + 1e-12)
    assert eps[0] == pytest.approx(EPS_REGOLITH_DEFAULT)
    assert eps[-1] == pytest.approx(EPS_ICE)


# ---------------------------------------------------------------------------
# Radar penetration depth
# ---------------------------------------------------------------------------
def test_penetration_depth_lband_magnitude():
    # L-band, eps' = 3, tan_delta = 0.005 -> a few metres (top ~5 m column).
    delta = penetration_depth(L_BAND_WAVELENGTH_M, 3.0, 0.005)
    assert 2.0 < delta < 8.0


def test_penetration_depth_decreases_with_loss():
    lo = penetration_depth(L_BAND_WAVELENGTH_M, 3.0, 0.005)
    hi = penetration_depth(L_BAND_WAVELENGTH_M, 3.0, 0.02)
    assert hi < lo


def test_penetration_depth_sband_shallower_than_lband():
    # Shorter S-band wavelength -> shallower penetration than L-band.
    d_l = penetration_depth(L_BAND_WAVELENGTH_M, 3.0, 0.005)
    d_s = penetration_depth(S_BAND_WAVELENGTH_M, 3.0, 0.005)
    assert d_s < d_l


# ---------------------------------------------------------------------------
# Volume / mass arithmetic
# ---------------------------------------------------------------------------
def test_ice_volume_arithmetic():
    assert ice_volume(1000.0, 5.0, 0.2) == pytest.approx(1000.0 * 5.0 * 0.2)


def test_ice_volume_array_fraction_uses_mean():
    frac = np.array([0.1, 0.3])  # mean 0.2
    assert ice_volume(1000.0, 5.0, frac) == pytest.approx(1000.0 * 5.0 * 0.2)


def test_ice_mass_arithmetic():
    v = 1234.5
    assert ice_mass(v) == pytest.approx(v * RHO_ICE_KGM3)
    assert ice_mass(v) == pytest.approx(v * 920.0)


# ---------------------------------------------------------------------------
# Monte-Carlo uncertainty propagation
# ---------------------------------------------------------------------------
def test_monte_carlo_volume_unbiased_and_bracketed():
    area, depth, frac, n = 580400.0, 5.0, 0.1, 10000
    mc = monte_carlo_volume(area, depth, frac, n=n, seed=0)
    analytic = ice_volume(area, depth, frac)

    # mean within 10 % of the analytic point estimate
    assert mc["mean"] == pytest.approx(analytic, rel=0.10)
    # the 95 % CI brackets the mean
    lo, hi = mc["ci"]
    assert lo < mc["mean"] < hi
    # positive spread and a full samples array
    assert mc["std"] > 0.0
    assert len(mc["samples"]) == n
    # derived mass statistics present and positive
    assert mc["mass_mean"] > 0.0
    assert mc["mass_std"] > 0.0


def test_monte_carlo_volume_deterministic_with_seed():
    a = monte_carlo_volume(1000.0, 5.0, 0.2, n=2000, seed=42)
    b = monte_carlo_volume(1000.0, 5.0, 0.2, n=2000, seed=42)
    assert a["mean"] == b["mean"]
    assert np.array_equal(a["samples"], b["samples"])


def test_monte_carlo_dielectric_inversion_mode_runs():
    # The physically-motivated (biased) dielectric path still returns a valid
    # distribution with positive spread.
    mc = monte_carlo_volume(
        1000.0, 5.0, 0.2, n=3000, seed=1, use_dielectric_inversion=True
    )
    assert mc["std"] > 0.0
    assert mc["ci"][0] < mc["ci"][1]
    assert len(mc["samples"]) == 3000


# ---------------------------------------------------------------------------
# cpr_to_ice_likelihood — relative index, not absolute abundance
# ---------------------------------------------------------------------------
def test_cpr_to_ice_likelihood_relative_index():
    # ice-like (high CPR, low DOP) -> near 1 ; decoy (low CPR or high DOP) -> ~0
    ice_like = cpr_to_ice_likelihood(1.3, 0.05)
    rough_decoy = cpr_to_ice_likelihood(1.3, 0.40)
    low_cpr = cpr_to_ice_likelihood(0.4, 0.05)
    assert 0.0 <= ice_like <= 1.0
    assert ice_like > rough_decoy
    assert ice_like > low_cpr
    # vectorised and bounded
    out = cpr_to_ice_likelihood(np.array([0.4, 1.0, 1.5]), np.array([0.3, 0.13, 0.02]))
    assert np.all((out >= 0.0) & (out <= 1.0))


# ---------------------------------------------------------------------------
# End-to-end scene estimate
# ---------------------------------------------------------------------------
def test_estimate_scene_ice(faustini_scene: LunarScene):
    res = estimate_scene_ice(faustini_scene)

    # positive ice-bearing area (ice_truth is non-empty)
    assert res["area_m2"] > 0.0
    # ice fraction is a physical fraction in (0, 1]
    assert 0.0 < res["ice_fraction"] <= 1.0
    # positive volume and mass
    assert res["volume_m3"] > 0.0
    assert res["mass_kg"] > 0.0
    # Monte-Carlo summary present and well-formed
    assert "mc" in res
    mc = res["mc"]
    assert mc["ci"][0] < mc["ci"][1]
    assert mc["std"] > 0.0
    # LCROSS gravimetric cross-check present and positive
    assert "lcross_cross_check_kg" in res
    assert res["lcross_cross_check_kg"] > 0.0
    # units documentation present
    assert "units" in res

    # area consistency: n_ice_px * pixel_area
    n_ice = int(np.asarray(faustini_scene.ice_truth).sum())
    pixel_area = faustini_scene.resolution_m ** 2
    assert res["area_m2"] == pytest.approx(n_ice * pixel_area)

    # report the headline numbers
    volume_m3 = res["volume_m3"]
    mass_tonnes = res["mass_kg"] / 1000.0
    ci_lo_t = ice_mass(mc["ci"][0]) / 1000.0
    ci_hi_t = ice_mass(mc["ci"][1]) / 1000.0
    lcross_t = res["lcross_cross_check_kg"] / 1000.0
    print(
        f"\n[volume] area = {res['area_m2']:.0f} m^2  "
        f"ice_fraction = {res['ice_fraction']:.3f}\n"
        f"[volume] volume = {volume_m3:,.0f} m^3  "
        f"mass = {mass_tonnes:,.0f} t\n"
        f"[volume] MC 95% CI (mass) = [{ci_lo_t:,.0f}, {ci_hi_t:,.0f}] t\n"
        f"[volume] LCROSS gravimetric cross-check = {lcross_t:,.0f} t"
    )
