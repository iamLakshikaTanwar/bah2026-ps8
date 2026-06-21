"""Deterministic synthetic Faustini-scene generator — the data backbone.

Until the real Chandrayaan-2 DFSAR / OHRC granules are downloaded and
co-registered, every pipeline stage is developed and tested against a
physically-plausible *synthetic* lunar South-Pole scene produced here.

The generator builds, in lunar south-polar stereographic CRS:

1. A bowl-shaped **host crater** (Faustini) with a raised rim and a deep floor
   (~ -2700 m), and a **nested "doubly-shadowed" crater** on that floor with its
   own rim and floor — plus fractal roughness and a few rim boulders.
2. Full **L- and S-band Stokes vectors** synthesised from per-pixel scattering
   mechanism fractions (surface / double-bounce / volume). The nested-crater
   floor carries a coherent **ICE patch** engineered so that
   ``CPR = (s0 - s3) / (s0 + s3) > 1`` and
   ``DOP = sqrt(s1^2 + s2^2 + s3^2) / s0 < 0.13`` (the Sinha et al. 2026
   criterion), while rough rim/ejecta produce **decoy** pixels with high CPR
   *and* high DOP (roughness false positives). Multiplicative speckle is added.
3. Diviner-like **maximum temperature**, annual **illumination**, 1064-nm
   **albedo**, **LAMP ratio**, and **Earth-visibility** layers, all correlated
   with topography (and with the ice patch where physically appropriate).
4. The boolean **ice ground-truth** mask.

Determinism: a single ``numpy.random.default_rng(seed)`` drives all stochastic
content, so identical ``(n, resolution_m, seed)`` yields bit-identical arrays.
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from ..constants import (
    FAUSTINI_FLOOR_ELEV_M,
    SOUTH_POLAR_STEREO_PROJ4,
)
from ..scene import LunarScene

__all__ = ["generate_faustini_scene"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _value_noise(rng: np.random.Generator, shape: tuple[int, int],
                 octaves: int = 5, persistence: float = 0.55) -> np.ndarray:
    """Seeded multi-octave smoothed value-noise in roughly ``[-1, 1]``.

    Each octave draws white noise, smooths it with a Gaussian whose width
    halves per octave, and accumulates with geometrically decaying amplitude.
    Deterministic for a given ``rng`` state.
    """
    h, w = shape
    out = np.zeros(shape, dtype=np.float64)
    amp = 1.0
    total = 0.0
    sigma = max(h, w) / 8.0
    for _ in range(octaves):
        white = rng.standard_normal(shape)
        smoothed = gaussian_filter(white, sigma=max(sigma, 0.6), mode="reflect")
        # normalise this octave to unit std before weighting
        s = smoothed.std()
        if s > 0:
            smoothed /= s
        out += amp * smoothed
        total += amp
        amp *= persistence
        sigma *= 0.5
    out /= total
    # squash to ~[-1, 1]
    m = np.abs(out).max()
    if m > 0:
        out /= m
    return out


def _radial(shape: tuple[int, int], cy: float, cx: float) -> np.ndarray:
    """Euclidean distance (in pixels) from ``(cy, cx)`` for every pixel."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return np.hypot(yy - cy, xx - cx)


def _bowl(r: np.ndarray, radius: float, depth: float, rim_h: float,
          rim_w: float) -> np.ndarray:
    """Idealised crater profile: paraboloid floor + raised Gaussian rim.

    Returns an elevation offset (m) relative to the surrounding plain that is
    negative inside the crater (down to ``depth``) and positive on the rim.
    """
    rn = r / radius
    # smooth paraboloid bowl, flattening just outside the rim
    floor = depth * np.clip(1.0 - rn ** 2, 0.0, 1.0)
    # raised rim ring centred on r == radius
    rim = rim_h * np.exp(-((r - radius) ** 2) / (2.0 * rim_w ** 2))
    elev = floor + rim
    # outside influence radius -> 0
    elev[r > radius + 4.0 * rim_w] = 0.0
    return elev


def _stokes_from_fractions(
    rng: np.random.Generator,
    f_surface: np.ndarray,
    f_double: np.ndarray,
    f_volume: np.ndarray,
    total_power: np.ndarray,
    looks: int,
    cpr_target_volume: float,
    dop_volume: float,
    f_decoy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a Stokes vector ``(s0, s1, s2, s3)`` from mechanism fractions.

    Sign / magnitude conventions (circular-basis Stokes for radar):

    * **Volume / ice** scattering -> ``s3 < 0`` with small ``|s1|, |s2|`` so that
      ``CPR = (s0 - s3)/(s0 + s3) > 1`` while ``DOP < dop_volume`` (coherent
      backscatter opposition effect from clean subsurface ice).
    * **Surface** (Bragg) scattering -> ``s3 > 0`` (CPR < 1), moderately
      polarised, partly randomised azimuth.
    * **Double-bounce** scattering -> ``s3 > 0`` (CPR < 1) but *highly* polarised
      with large ``s1``.
    * **Decoy roughness** (``f_decoy``) -> ``s3 < 0`` (so CPR > 1, mimicking ice)
      but with a large *linear* polarised component (high ``s1``) so the
      **DOP stays high** (~0.3-0.6). This is the wavelength-scale rough-rim
      false positive that pure CPR thresholding cannot reject — DOP does.

    Speckle: a multiplicative Gamma(L, 1/L) (``L`` = ``looks``) field scales the
    total power, mimicking ``L``-look SAR intensity speckle.
    """
    shape = total_power.shape
    if f_decoy is None:
        f_decoy = np.zeros(shape)
    # --- per-mechanism polarised contributions -------------------------
    # Volume: s3 negative, |s3|/s0 == dop_volume -> CPR>1 with low DOP.
    # Drive d (= |s3|/s0) from the requested CPR; clip so DOP stays < dop cap.
    d_from_cpr = (cpr_target_volume - 1.0) / (cpr_target_volume + 1.0)
    d_vol = min(max(d_from_cpr, 0.02), dop_volume)
    s3_vol = -d_vol * f_volume          # negative -> high CPR
    s1_vol = 0.01 * f_volume * rng.standard_normal(shape)
    s2_vol = 0.01 * f_volume * rng.standard_normal(shape)

    # Surface: s3 positive (CPR<1), partial linear polarisation.
    s3_surf = 0.45 * f_surface
    ang = rng.uniform(0.0, np.pi, size=shape)
    pol_s = 0.35 * f_surface
    s1_surf = pol_s * np.cos(2 * ang)
    s2_surf = pol_s * np.sin(2 * ang)

    # Double-bounce: s3 positive (CPR<1) but strongly polarised (high DOP).
    s3_db = 0.30 * f_double
    s1_db = 0.55 * f_double
    s2_db = 0.10 * f_double * rng.standard_normal(shape)

    # Decoy roughness: s3 negative (CPR>1) but strong linear pol -> high DOP.
    s3_decoy = -0.22 * f_decoy          # negative -> CPR > 1 (looks like ice)
    s1_decoy = 0.42 * f_decoy           # large linear pol -> DOP ~0.3-0.6
    s2_decoy = 0.18 * f_decoy * rng.standard_normal(shape)

    s1 = s1_vol + s1_surf + s1_db + s1_decoy
    s2 = s2_vol + s2_surf + s2_db + s2_decoy
    s3 = s3_vol + s3_surf + s3_db + s3_decoy

    # --- speckle on total power ---------------------------------------
    if looks > 0:
        speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=shape)
    else:  # pragma: no cover
        speckle = np.ones(shape)
    s0 = total_power * speckle

    # Scale polarised parts by the same speckle so DOP is stable, then
    # enforce the physical constraint s0 >= sqrt(s1^2+s2^2+s3^2).
    s1 = s1 * total_power * speckle
    s2 = s2 * total_power * speckle
    s3 = s3 * total_power * speckle
    pol_mag = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
    s0 = np.maximum(s0, pol_mag * 1.0001) + 1e-6
    return s0, s1, s2, s3


def _cpr(s0: np.ndarray, s3: np.ndarray) -> np.ndarray:
    return (s0 - s3) / (s0 + s3 + 1e-12)


def _dop(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> np.ndarray:
    return np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2) / (s0 + 1e-12)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def generate_faustini_scene(n: int = 512, resolution_m: float = 20.0,
                            seed: int = 42) -> LunarScene:
    """Generate a deterministic synthetic Faustini South-Pole scene.

    Parameters
    ----------
    n : int, default 512
        Grid edge length in pixels (scene is ``n x n``).
    resolution_m : float, default 20.0
        Ground sample distance per pixel [m].
    seed : int, default 42
        Seed for ``numpy.random.default_rng`` — fully determines the output.

    Returns
    -------
    LunarScene
        A populated scene whose ``ice_truth`` mask marks a subsurface-ice patch
        on the nested doubly-shadowed crater floor, with radar layers obeying
        ``CPR>1 & DOP<0.13`` inside that patch.
    """
    rng = np.random.default_rng(seed)
    shape = (n, n)
    cy = cx = (n - 1) / 2.0

    # ---- geometry of the two craters (pixels) ------------------------
    host_radius = 0.40 * n
    host_rim_w = max(0.03 * n, 3.0)
    host_rim_h = 900.0           # raised rim [m]
    host_depth = FAUSTINI_FLOOR_ELEV_M  # floor offset [m]

    # nested doubly-shadowed crater ~1.1 km diameter -> radius 550 m.
    nested_radius_px = 550.0 / resolution_m
    # clamp so it comfortably fits inside the host floor
    nested_radius_px = float(np.clip(nested_radius_px, 4.0, 0.22 * n))
    nested_rim_w = max(0.4 * nested_radius_px, 1.5)
    nested_rim_h = 120.0
    nested_depth = -350.0        # extra depth below host floor [m]
    # place the nested crater slightly off-centre on the host floor
    off = 0.12 * n
    ncy = cy + off * 0.3
    ncx = cx - off * 0.5

    r_host = _radial(shape, cy, cx)
    r_nested = _radial(shape, ncy, ncx)

    # ---- DEM ----------------------------------------------------------
    plain = 0.0
    dem = np.full(shape, plain, dtype=np.float64)
    dem += _bowl(r_host, host_radius, host_depth, host_rim_h, host_rim_w)
    dem += _bowl(r_nested, nested_radius_px, nested_depth, nested_rim_h, nested_rim_w)

    # fractal roughness — stronger on rims/ejecta, gentle on smooth floors
    rough = _value_noise(rng, shape, octaves=6, persistence=0.55)
    rim_emphasis = (
        np.exp(-((r_host - host_radius) ** 2) / (2.0 * (2.0 * host_rim_w) ** 2))
        + 0.6 * np.exp(-((r_nested - nested_radius_px) ** 2)
                       / (2.0 * (2.0 * nested_rim_w) ** 2))
    )
    dem += rough * (8.0 + 45.0 * rim_emphasis)

    # boulders: a handful of small Gaussian bumps near the host rim
    n_boulders = 12
    for _ in range(n_boulders):
        ang = rng.uniform(0, 2 * np.pi)
        rad = host_radius + rng.uniform(-1.5, 1.5) * host_rim_w
        by = cy + rad * np.sin(ang)
        bx = cx + rad * np.cos(ang)
        if 0 <= by < n and 0 <= bx < n:
            bump_r = _radial(shape, by, bx)
            sigma = rng.uniform(1.0, 2.5)
            height = rng.uniform(6.0, 20.0)
            dem += height * np.exp(-(bump_r ** 2) / (2.0 * sigma ** 2))

    # ---- region masks -------------------------------------------------
    host_floor = r_host < (host_radius - host_rim_w)
    nested_floor = r_nested < (nested_radius_px - 0.5 * nested_rim_w)
    nested_rim = (np.abs(r_nested - nested_radius_px) < 1.5 * nested_rim_w)
    host_rim = (np.abs(r_host - host_radius) < 2.0 * host_rim_w)

    # ICE ground truth: the central, deepest part of the nested-crater floor.
    ice_truth = r_nested < (0.78 * nested_radius_px)
    ice_truth &= nested_floor

    # ---- scattering-mechanism fractions ------------------------------
    # baseline mature regolith: mostly surface scatter, low volume.
    f_surface = np.full(shape, 0.62)
    f_double = np.full(shape, 0.10)
    f_volume = np.full(shape, 0.28)

    # rough rims / ejecta: more double-bounce + surface roughness (decoys).
    rim_mask = host_rim | nested_rim
    f_double = np.where(rim_mask, 0.45, f_double)
    f_surface = np.where(rim_mask, 0.40, f_surface)
    f_volume = np.where(rim_mask, 0.15, f_volume)

    # decoy roughness hotspots: scattered very-rough pixels on the rim that
    # produce high CPR (s3<0) *with* high DOP -> roughness false positives that
    # only DOP can reject. Carried in a dedicated f_decoy mechanism channel.
    decoy_seed = rng.random(shape)
    decoy = rim_mask & (~ice_truth) & (decoy_seed > 0.80)
    f_decoy = np.where(decoy, 0.65, 0.0)
    # the remaining (non-decoy) mechanisms shrink where decoy dominates
    f_surface = np.where(decoy, 0.20, f_surface)
    f_double = np.where(decoy, 0.10, f_double)
    f_volume = np.where(decoy, 0.05, f_volume)

    # ICE patch: dominated by coherent volume scattering.
    f_volume = np.where(ice_truth, 0.88, f_volume)
    f_surface = np.where(ice_truth, 0.08, f_surface)
    f_double = np.where(ice_truth, 0.04, f_double)

    # renormalise the four mechanism fractions to sum to 1
    fsum = f_surface + f_double + f_volume + f_decoy
    f_surface = f_surface / fsum
    f_double = f_double / fsum
    f_volume = f_volume / fsum
    f_decoy = f_decoy / fsum

    # total radar power (arbitrary linear units): ice patch a bit brighter
    total_power_L = np.full(shape, 1.0)
    total_power_L = np.where(ice_truth, 1.6, total_power_L)
    total_power_L = np.where(rim_mask, 1.3, total_power_L)
    total_power_L *= (1.0 + 0.05 * rough)
    total_power_S = total_power_L * 0.9

    # ---- L-band Stokes -----------------------------------------------
    s0_L, s1_L, s2_L, s3_L = _stokes_from_fractions(
        rng, f_surface, f_double, f_volume, total_power_L,
        looks=6, cpr_target_volume=1.22, dop_volume=0.085, f_decoy=f_decoy,
    )
    # ---- S-band Stokes (shallower -> slightly weaker ice signature) ---
    # Lower the requested volume-CPR and raise the DOP cap for S-band so its
    # ice signature is a touch weaker than L (penetration is shallower at S).
    s0_S, s1_S, s2_S, s3_S = _stokes_from_fractions(
        rng, f_surface, f_double, f_volume, total_power_S,
        looks=6, cpr_target_volume=1.14, dop_volume=0.105, f_decoy=f_decoy,
    )

    # ---- derived radar products --------------------------------------
    cpr_L = _cpr(s0_L, s3_L)
    dop_L = _dop(s0_L, s1_L, s2_L, s3_L)
    cpr_S = _cpr(s0_S, s3_S)
    dop_S = _dop(s0_S, s1_S, s2_S, s3_S)

    # ---- temperature_max (Diviner-like) ------------------------------
    # warm plain ~210 K; host floor cold; nested floor coldest; rims warmest.
    temperature_max = np.full(shape, 210.0)
    # cool with depth inside host
    host_cool = np.clip((host_radius - r_host) / host_radius, 0.0, 1.0)
    temperature_max -= 120.0 * host_cool
    # nested crater extra-cold core (down to ~20-40 K)
    nested_cool = np.clip((nested_radius_px - r_nested) / nested_radius_px, 0.0, 1.0)
    temperature_max = np.where(nested_floor,
                               60.0 - 30.0 * nested_cool,
                               temperature_max)
    temperature_max = np.where(ice_truth,
                               np.minimum(temperature_max, 35.0),
                               temperature_max)
    # warm rims
    temperature_max = np.where(host_rim, 220.0, temperature_max)
    temperature_max += 4.0 * rough
    temperature_max = np.clip(temperature_max, 18.0, 250.0)

    # ---- illumination (annual fraction, [0,1]) -----------------------
    illumination = np.full(shape, 0.45)
    illumination = np.where(host_floor, 0.02, illumination)
    illumination = np.where(nested_floor, 0.0, illumination)
    # bright pole-facing rims
    rim_bright = np.exp(-((r_host - host_radius) ** 2) / (2.0 * host_rim_w ** 2))
    illumination = illumination + 0.85 * rim_bright
    illumination += 0.03 * rough
    illumination = np.clip(illumination, 0.0, 1.0)

    # ---- albedo_1064 & lamp_ratio (frost proxies) --------------------
    albedo_1064 = np.full(shape, 0.12) + 0.01 * rough
    albedo_1064 = np.where(ice_truth, 0.30, albedo_1064)
    albedo_1064 += 0.02 * rng.standard_normal(shape)
    albedo_1064 = np.clip(albedo_1064, 0.0, 1.0)

    lamp_ratio = np.full(shape, 1.0) + 0.03 * rng.standard_normal(shape)
    lamp_ratio = np.where(ice_truth, 1.25, lamp_ratio)
    lamp_ratio = np.where(nested_floor & ~ice_truth, 1.10, lamp_ratio)
    lamp_ratio = np.clip(lamp_ratio, 0.7, 1.6)

    # ---- earth_visibility ([0,1]) ------------------------------------
    # high on pole-facing rims, ~0 deep in floors.
    earth_visibility = np.clip(0.30 + 0.9 * rim_bright, 0.0, 1.0)
    earth_visibility = np.where(host_floor, 0.05, earth_visibility)
    earth_visibility = np.where(nested_floor, 0.0, earth_visibility)
    earth_visibility += 0.02 * rough
    earth_visibility = np.clip(earth_visibility, 0.0, 1.0)

    # ---- geo-referencing ---------------------------------------------
    # centre the south pole near the grid centre.
    half = (n * resolution_m) / 2.0
    transform = from_origin(-half, half, resolution_m, resolution_m)

    meta = {
        "target": "Faustini (synthetic)",
        "synthetic": True,
        "seed": int(seed),
        "n": int(n),
        "host_radius_px": float(host_radius),
        "nested_center_px": [float(ncy), float(ncx)],
        "nested_radius_px": float(nested_radius_px),
        "ice_pixels": int(ice_truth.sum()),
        "generator": "lunaris.io.synthetic.generate_faustini_scene",
    }

    return LunarScene(
        dem=dem,
        s0_L=s0_L, s1_L=s1_L, s2_L=s2_L, s3_L=s3_L,
        s0_S=s0_S, s1_S=s1_S, s2_S=s2_S, s3_S=s3_S,
        cpr_L=cpr_L, dop_L=dop_L, cpr_S=cpr_S, dop_S=dop_S,
        temperature_max=temperature_max,
        illumination=illumination,
        albedo_1064=albedo_1064,
        lamp_ratio=lamp_ratio,
        earth_visibility=earth_visibility,
        ice_truth=ice_truth,
        transform=transform,
        crs=SOUTH_POLAR_STEREO_PROJ4,
        resolution_m=float(resolution_m),
        meta=meta,
    )
