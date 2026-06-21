"""Tests for :mod:`lunaris.polarimetry`.

Exercises the CTLR hybrid-pol Stokes synthesis, CPR/DOP ice criterion,
m-chi / m-delta / CHILD decompositions, Cloude-Pottier H/A/alpha, the boxcar
and refined-Lee speckle filters, and co-pol coherence — all driven by the
deterministic synthetic Faustini scene.

Reference equations (CTLR, transmit right-circular):
    Raney (2012), Kumar et al. (2015), Campbell (2012).
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.io.synthetic import generate_faustini_scene
from lunaris.polarimetry import (
    boxcar,
    child_parameters,
    circular_polarization_ratio,
    cloude_pottier,
    copol_coherence,
    degree_of_polarization,
    m_chi,
    m_delta,
    multilook,
    refined_lee,
    sc_oc_power,
    scattering_matrix_to_circular,
    stokes_from_circular,
)

N = 128
SEED = 42


@pytest.fixture(scope="module")
def scene():
    """A single deterministic 128x128 synthetic scene for all tests."""
    return generate_faustini_scene(n=N, seed=SEED)


@pytest.fixture(scope="module")
def stokes_L(scene):
    """Raw L-band Stokes arrays (s0,s1,s2,s3) from the scene."""
    return scene.s0_L, scene.s1_L, scene.s2_L, scene.s3_L


# ---------------------------------------------------------------------------
# CPR / DOP — the ice criterion
# ---------------------------------------------------------------------------
def test_cpr_reproduces_scene(scene, stokes_L):
    """CPR from scene Stokes reproduces the scene's stored cpr_L layer."""
    s0, _, _, s3 = stokes_L
    cpr = circular_polarization_ratio(s0, s3)
    assert np.allclose(cpr, scene.cpr_L, atol=1e-6)


def test_dop_reproduces_scene(scene, stokes_L):
    """DOP from scene Stokes reproduces the scene's stored dop_L layer."""
    s0, s1, s2, s3 = stokes_L
    dop = degree_of_polarization(s0, s1, s2, s3)
    assert np.allclose(dop, scene.dop_L, atol=1e-6)


def test_ice_criterion_inside_truth(scene, stokes_L):
    """Inside the ice mask: median CPR > 1 and median DOP < 0.13."""
    s0, s1, s2, s3 = stokes_L
    ice = np.asarray(scene.ice_truth, dtype=bool)
    cpr = circular_polarization_ratio(s0, s3)
    dop = degree_of_polarization(s0, s1, s2, s3)
    cpr_ice = float(np.median(cpr[ice]))
    dop_ice = float(np.median(dop[ice]))
    assert cpr_ice > 1.0, f"ice CPR median {cpr_ice} not > 1"
    assert dop_ice < 0.13, f"ice DOP median {dop_ice} not < 0.13"


def test_background_cpr_low(scene, stokes_L):
    """Background (non-ice) median CPR stays below 0.7."""
    s0, _, _, s3 = stokes_L
    ice = np.asarray(scene.ice_truth, dtype=bool)
    cpr = circular_polarization_ratio(s0, s3)
    bg = float(np.median(cpr[~ice]))
    assert bg < 0.7, f"background CPR median {bg} not < 0.7"


def test_dop_in_unit_range(stokes_L):
    """DOP is clipped into [0, 1] everywhere."""
    s0, s1, s2, s3 = stokes_L
    dop = degree_of_polarization(s0, s1, s2, s3)
    assert dop.min() >= 0.0
    assert dop.max() <= 1.0


def test_sc_oc_consistency(stokes_L):
    """CPR equals SC/OC from sc_oc_power."""
    s0, _, _, s3 = stokes_L
    sc, oc = sc_oc_power(s0, s3)
    cpr = circular_polarization_ratio(s0, s3)
    assert np.allclose(cpr, sc / oc, atol=1e-9)
    # SC + OC reconstructs total power s0.
    assert np.allclose(sc + oc, s0, atol=1e-9)


# ---------------------------------------------------------------------------
# m-chi / m-delta decomposition
# ---------------------------------------------------------------------------
def test_mchi_power_conservation(stokes_L):
    """m-chi: even^2 + odd^2 == m*s0 and volume^2 == (1-m)*s0."""
    s0, s1, s2, s3 = stokes_L
    even, volume, odd = m_chi(s0, s1, s2, s3)
    m = degree_of_polarization(s0, s1, s2, s3)
    assert np.allclose(even ** 2 + odd ** 2, m * s0, atol=1e-9)
    assert np.allclose(volume ** 2, (1.0 - m) * s0, atol=1e-9)


def test_mchi_volume_dominant_over_ice(scene, stokes_L):
    """Inside the ice mask the m-chi volume channel dominates even & odd."""
    s0, s1, s2, s3 = stokes_L
    ice = np.asarray(scene.ice_truth, dtype=bool)
    even, volume, odd = m_chi(s0, s1, s2, s3)
    v = float(np.median(volume[ice]))
    e = float(np.median(even[ice]))
    o = float(np.median(odd[ice]))
    assert v > e, f"volume {v} !> even {e}"
    assert v > o, f"volume {v} !> odd {o}"


def test_mchi_nonnegative(stokes_L):
    """All m-chi power channels are real and non-negative (clip before sqrt)."""
    s0, s1, s2, s3 = stokes_L
    for ch in m_chi(s0, s1, s2, s3):
        assert np.all(np.isfinite(ch))
        assert np.all(ch >= 0.0)


def test_mdelta_volume_conserved(stokes_L):
    """m-delta volume^2 == (1-m)*s0 and all channels finite/non-negative."""
    s0, s1, s2, s3 = stokes_L
    double, volume, surface = m_delta(s0, s1, s2, s3)
    m = degree_of_polarization(s0, s1, s2, s3)
    assert np.allclose(volume ** 2, (1.0 - m) * s0, atol=1e-9)
    for ch in (double, volume, surface):
        assert np.all(np.isfinite(ch))
        assert np.all(ch >= 0.0)


# ---------------------------------------------------------------------------
# Stokes physical bound after multilook
# ---------------------------------------------------------------------------
def test_stokes_physical_bound_after_multilook():
    """s0^2 >= s1^2 + s2^2 + s3^2 everywhere after spatial multilook.

    Build circular fields from a random scattering matrix, form multilooked
    Stokes, and verify the partial-polarisation inequality holds (it must, since
    a windowed coherency matrix is positive semi-definite).
    """
    rng = np.random.default_rng(SEED)
    shape = (N, N)

    def cplx():
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    shh, shv, svh, svv = cplx(), cplx(), cplx(), cplx()
    E_RH, E_RV = scattering_matrix_to_circular(shh, shv, svh, svv)
    s0, s1, s2, s3 = stokes_from_circular(E_RH, E_RV, window=5)
    pol2 = s1 ** 2 + s2 ** 2 + s3 ** 2
    assert np.all(s0 ** 2 >= pol2 - 1e-9)
    # and DOP <= 1 follows directly.
    assert degree_of_polarization(s0, s1, s2, s3).max() <= 1.0 + 1e-9


def test_multilook_shape_and_mean():
    """multilook block-averages: shrinks by factor & preserves the global mean."""
    rng = np.random.default_rng(1)
    img = rng.gamma(6.0, 1.0 / 6.0, (N, N))
    out = multilook(img, looks=2)
    assert out.shape == (N // 2, N // 2)
    # block mean preserves the overall mean.
    assert out.mean() == pytest.approx(img.mean(), rel=1e-9)
    # variance is reduced by multilooking.
    assert out.var() < img.var()


# ---------------------------------------------------------------------------
# CHILD parameters
# ---------------------------------------------------------------------------
def test_child_parameters_finite_and_chi_range(stokes_L):
    """child_parameters returns finite arrays; chi within [-45, 45] degrees."""
    s0, s1, s2, s3 = stokes_L
    cp = child_parameters(s0, s1, s2, s3)
    for key in ("chi", "delta", "psi", "conformity", "chi_deg", "delta_deg"):
        assert key in cp
        assert np.all(np.isfinite(cp[key])), f"{key} has non-finite values"
    chi_deg = cp["chi_deg"]
    assert chi_deg.min() >= -45.0 - 1e-9
    assert chi_deg.max() <= 45.0 + 1e-9


# ---------------------------------------------------------------------------
# Cloude-Pottier
# ---------------------------------------------------------------------------
def test_cloude_pottier_ranges():
    """Entropy & anisotropy in [0,1]; alpha in [0,90] deg; all finite."""
    rng = np.random.default_rng(3)
    shape = (64, 64)

    def cplx():
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    shh, shv, svh, svv = cplx(), cplx(), cplx(), cplx()
    E_RH, E_RV = scattering_matrix_to_circular(shh, shv, svh, svv)
    out = cloude_pottier(E_RH, E_RV, window=5)
    for key in ("entropy", "anisotropy", "alpha"):
        assert np.all(np.isfinite(out[key]))
    assert out["entropy"].min() >= 0.0 and out["entropy"].max() <= 1.0
    assert out["anisotropy"].min() >= 0.0 and out["anisotropy"].max() <= 1.0
    assert out["alpha_deg"].min() >= 0.0
    assert out["alpha_deg"].max() <= 90.0 + 1e-6


def test_cloude_pottier_accepts_stokes(stokes_L):
    """The Stokes-input path returns the same-shaped, in-range outputs."""
    s0, s1, s2, s3 = stokes_L
    out = cloude_pottier(s0=s0, s1=s1, s2=s2, s3=s3, window=5)
    assert out["entropy"].shape == s0.shape
    assert out["entropy"].min() >= 0.0 and out["entropy"].max() <= 1.0


# ---------------------------------------------------------------------------
# Speckle filters
# ---------------------------------------------------------------------------
def test_boxcar_reduces_variance():
    """Boxcar reduces variance on a homogeneous speckle patch."""
    rng = np.random.default_rng(5)
    patch = rng.gamma(6.0, 1.0 / 6.0, (96, 96))  # homogeneous, mean ~1
    out = boxcar(patch, size=5)
    assert out.var() < patch.var()


def test_refined_lee_reduces_variance_preserves_mean():
    """Refined Lee reduces variance vs raw while preserving the mean within a few %."""
    rng = np.random.default_rng(9)
    patch = rng.gamma(6.0, 1.0 / 6.0, (96, 96))  # homogeneous speckle field
    out = refined_lee(patch, size=7)
    assert out.var() < patch.var(), "refined_lee did not reduce variance"
    # mean preserved within a few percent.
    rel = abs(out.mean() - patch.mean()) / patch.mean()
    assert rel < 0.03, f"refined_lee mean drifted {rel:.4f}"
    assert out.shape == patch.shape
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Co-pol coherence
# ---------------------------------------------------------------------------
def test_coherence_in_unit_range():
    """Co-pol coherence magnitude lies in [0, 1]."""
    rng = np.random.default_rng(13)
    shape = (96, 96)
    shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    # partially correlated svv plus independent component.
    svv = 0.6 * shh + 0.4 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    gamma = copol_coherence(shh, svv, window=5)
    assert np.all(np.isfinite(gamma))
    assert gamma.min() >= 0.0
    assert gamma.max() <= 1.0


def test_coherence_perfectly_correlated():
    """Identical co-pol channels give coherence == 1."""
    rng = np.random.default_rng(17)
    shape = (48, 48)
    shh = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    gamma = copol_coherence(shh, shh.copy(), window=5)
    assert np.allclose(gamma, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Round-trip sanity: known scattering matrix -> Stokes -> CPR
# ---------------------------------------------------------------------------
def test_roundtrip_dihedral_high_cpr():
    """A dihedral (double-bounce) scatterer round-trips to same-sense CPR > 1.

    Shh = +1, Svv = -1 is a canonical dihedral: in the CTLR circular basis it
    backscatters into the same-sense channel, so CPR = (s0 - s3)/(s0 + s3) >> 1.
    Independent additive speckle keeps the ratio finite.
    """
    rng = np.random.default_rng(11)
    shape = (48, 48)

    def noise():
        return 0.15 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    shh = np.ones(shape, dtype=complex) + noise()
    svv = -np.ones(shape, dtype=complex) + noise()
    shv = noise()
    svh = noise()
    E_RH, E_RV = scattering_matrix_to_circular(shh, shv, svh, svv)
    s0, s1, s2, s3 = stokes_from_circular(E_RH, E_RV, window=7)
    cpr = circular_polarization_ratio(s0, s3)
    assert np.all(np.isfinite(cpr))
    assert float(np.median(cpr)) > 1.0


def test_roundtrip_trihedral_low_cpr():
    """A trihedral (single-bounce surface) round-trips to CPR < 1.

    Shh = Svv = +1 is a canonical odd-bounce surface: opposite-sense dominated,
    so CPR < 1.
    """
    rng = np.random.default_rng(23)
    shape = (48, 48)

    def noise():
        return 0.15 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    shh = np.ones(shape, dtype=complex) + noise()
    svv = np.ones(shape, dtype=complex) + noise()
    shv = noise()
    svh = noise()
    E_RH, E_RV = scattering_matrix_to_circular(shh, shv, svh, svv)
    s0, s1, s2, s3 = stokes_from_circular(E_RH, E_RV, window=7)
    cpr = circular_polarization_ratio(s0, s3)
    assert float(np.median(cpr)) < 1.0
