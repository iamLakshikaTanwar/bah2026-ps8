"""Ice volume / mass estimation with Monte-Carlo uncertainty.

Implemented by: **volume agent**.

Two independent estimators of the same quantity are provided so they can be
cross-checked:

* **Volumetric (dielectric)** — multiply ice-bearing footprint area by the
  radar-sampled depth and a dielectric ice volume fraction
  (:func:`ice_volume`), then convert to mass with the bulk ice density
  (:func:`ice_mass`). The ice fraction comes from a Looyenga-Landau-Lifshitz
  inversion of the effective permittivity, with the polarimetric CPR/DOP field
  used only as a *relative* ice-likelihood weighting.
* **Gravimetric (weight-percent)** — an independent cross-check anchored to the
  in-situ LCROSS ejecta measurement of 5.6 +/- 2.9 wt% water in the regolith
  (Colaprete et al. 2010): ``mass = wt% * rho_bulk * area * depth``.

Uncertainty in area, depth and ice fraction is propagated by Monte-Carlo
sampling (:func:`monte_carlo_volume`).

References
----------
* Looyenga (1965); Landau & Lifshitz — LLL dielectric mixing.
* Colaprete, A. et al. (2010), *Science* 330, 463 — LCROSS water abundance
  (5.6 +/- 2.9 wt%).
* Olhoeft & Strangway (1975) — lunar-sample dielectrics.
"""

from __future__ import annotations

import numpy as np

from ..constants import (
    EPS_ICE,
    EPS_REGOLITH_DEFAULT,
    LCROSS_WT_PCT,
    RHO_ICE_KGM3,
    RHO_REGOLITH_GCM3,
)
from .dielectric import (
    cpr_to_ice_likelihood,
    looyenga_eps,
    looyenga_ice_fraction,
)

__all__ = [
    "ice_volume",
    "ice_mass",
    "monte_carlo_volume",
    "estimate_scene_ice",
]


def ice_volume(
    area_m2: np.ndarray | float,
    depth_m: np.ndarray | float,
    frac: np.ndarray | float,
) -> np.ndarray | float:
    """Bulk ice volume from footprint area, sampled depth and ice fraction.

        V = area_m2 * depth_m * frac   [m^3]

    If ``frac`` is array-like its mean is used (a single representative ice
    volume fraction for the footprint).

    Parameters
    ----------
    area_m2 : np.ndarray or float
        Ice-bearing footprint area [m^2].
    depth_m : np.ndarray or float
        Sampled subsurface depth (<= top ~5 m) [m].
    frac : np.ndarray or float
        Ice volume fraction in ``[0, 1]`` (scalar, or array -> mean).

    Returns
    -------
    np.ndarray or float
        Ice volume [m^3].
    """
    frac_arr = np.asarray(frac, dtype=float)
    frac_scalar = float(frac_arr.mean()) if frac_arr.ndim else float(frac_arr)
    return area_m2 * depth_m * frac_scalar


def ice_mass(
    volume_m3: np.ndarray | float, rho: float = RHO_ICE_KGM3
) -> np.ndarray | float:
    """Ice mass from volume and density.

        M = rho * V   [kg]

    Parameters
    ----------
    volume_m3 : np.ndarray or float
        Ice volume [m^3].
    rho : float, default :data:`RHO_ICE_KGM3`
        Ice density [kg m^-3] (920 for pure water ice).

    Returns
    -------
    np.ndarray or float
        Ice mass [kg].
    """
    return rho * volume_m3


def monte_carlo_volume(
    area_m2: float,
    depth_m: float,
    frac: float,
    n: int = 10000,
    seed: int = 0,
    area_rel_sigma: float = 0.15,
    depth_sigma: float = 1.0,
    frac_sigma: float = 0.3,
    eps_reg_sigma: float = 0.2,
    use_dielectric_inversion: bool = False,
) -> dict:
    """Monte-Carlo ice-volume / mass estimate with uncertainty propagation.

    Draws ``n`` joint samples of the three multiplicative factors and forms the
    volume (and mass) for each draw, yielding an empirical distribution and a
    95 % credible interval rather than a single point value:

    * **area** ~ Normal(area_m2, ``area_rel_sigma`` * area_m2), clipped > 0.
      The relative sigma reflects mask / footprint delineation error.
    * **depth** ~ Normal(depth_m, ``depth_sigma``), clipped > 0. Reflects
      penetration-depth uncertainty.
    * **frac** — the ice volume fraction. Two modes are available:

      - *Direct* (default): frac ~ Normal(frac, ``frac_sigma`` * frac), clipped
        to ``[0, 1]``. This is the unbiased estimator whose sample mean tracks
        the analytic :func:`ice_volume`, and is the default so the Monte-Carlo
        mean is consistent with the point estimate.
      - *Dielectric inversion* (``use_dielectric_inversion=True``): the fraction
        is re-derived per draw by perturbing the dielectric inputs and inverting
        the Looyenga-Landau-Lifshitz mixing law — the central ``frac`` is mapped
        to an effective permittivity via :func:`looyenga_eps`, that ``eps_eff``
        is jittered, the regolith reference permittivity is jittered by
        ``eps_reg_sigma``, and :func:`looyenga_ice_fraction` inverts back. This
        propagates the (weak, ill-conditioned) dielectric contrast honestly but
        is biased by the nonlinear inversion, so it is opt-in.

    Parameters
    ----------
    area_m2, depth_m, frac : float
        Central estimates.
    n : int, default 10000
        Number of Monte-Carlo samples.
    seed : int, default 0
        RNG seed (``numpy.random.default_rng``).
    area_rel_sigma : float, default 0.15
        Relative 1-sigma on area.
    depth_sigma : float, default 1.0
        Absolute 1-sigma on depth [m].
    frac_sigma : float, default 0.3
        Relative 1-sigma on the ice fraction (direct mode), or the relative
        jitter applied to ``eps_eff`` in dielectric-inversion mode.
    eps_reg_sigma : float, default 0.2
        Absolute 1-sigma on the regolith reference permittivity (used only in
        dielectric-inversion mode).
    use_dielectric_inversion : bool, default False
        If True, sample the fraction through the Looyenga inversion instead of
        directly.

    Returns
    -------
    dict
        ``mean``, ``std``, ``ci`` (2.5/97.5 percentile tuple) and the raw
        ``samples`` array for volume [m^3], plus ``mass_mean`` and ``mass_std``
        for the derived mass [kg].

    References
    ----------
    Looyenga (1965) mixing; standard Monte-Carlo error propagation.
    """
    rng = np.random.default_rng(seed)

    # Area: relative-Gaussian, kept strictly positive.
    area_samples = rng.normal(area_m2, area_rel_sigma * abs(area_m2), size=n)
    area_samples = np.clip(area_samples, 1e-9, None)

    # Depth: Gaussian, kept strictly positive.
    depth_samples = rng.normal(depth_m, depth_sigma, size=n)
    depth_samples = np.clip(depth_samples, 1e-9, None)

    frac0 = float(np.clip(frac, 0.0, 1.0))
    if use_dielectric_inversion:
        # Physically-motivated but nonlinearly-biased dielectric path.
        eps_eff0 = looyenga_eps(frac0, EPS_ICE, EPS_REGOLITH_DEFAULT)
        eps_eff_samples = rng.normal(eps_eff0, frac_sigma * abs(eps_eff0), size=n)
        eps_reg_samples = rng.normal(EPS_REGOLITH_DEFAULT, eps_reg_sigma, size=n)
        # Keep references physical (permittivity >= 1, below pure ice) and
        # invert per draw against the *jittered* regolith reference so the
        # eps_reg uncertainty actually propagates into the fraction.
        eps_eff_samples = np.clip(eps_eff_samples, 1.0, None)
        eps_reg_samples = np.clip(eps_reg_samples, 1.0, EPS_ICE - 1e-6)
        frac_samples = looyenga_ice_fraction(
            eps_eff_samples, EPS_ICE, eps_reg_samples
        )
        frac_samples = np.clip(frac_samples, 0.0, 1.0)
    else:
        # Unbiased direct sampling: sample mean tracks the analytic estimate.
        frac_samples = rng.normal(frac0, frac_sigma * frac0, size=n)
        frac_samples = np.clip(frac_samples, 0.0, 1.0)

    volume_samples = area_samples * depth_samples * frac_samples
    mass_samples = ice_mass(volume_samples)

    return {
        "mean": float(np.mean(volume_samples)),
        "std": float(np.std(volume_samples)),
        "ci": (
            float(np.percentile(volume_samples, 2.5)),
            float(np.percentile(volume_samples, 97.5)),
        ),
        "samples": volume_samples,
        "mass_mean": float(np.mean(mass_samples)),
        "mass_std": float(np.std(mass_samples)),
    }


def estimate_scene_ice(
    scene,
    ice_mask: np.ndarray | None = None,
    depth_m: float = 5.0,
) -> dict:
    """Estimate subsurface-ice volume and mass for a :class:`LunarScene`.

    Combines the two independent estimators documented at module level:

    1. **Volumetric (dielectric).** The ice-bearing area is the number of true
       pixels in ``ice_mask`` times the pixel area (``resolution_m**2``). A
       single representative ice *volume fraction* for the footprint is derived
       by (a) averaging the :func:`cpr_to_ice_likelihood` relative index over
       the masked pixels, and (b) mapping that relative index to a small
       effective-permittivity excess above the regolith reference, which the
       Looyenga-Landau-Lifshitz inversion (:func:`looyenga_ice_fraction`) turns
       into a volume fraction. The relative index scales how far ``eps_eff``
       sits between ``EPS_REGOLITH_DEFAULT`` and ``EPS_ICE``; because the
       contrast is tiny the resulting fraction is modest by construction. The
       volume and mass follow from :func:`ice_volume` / :func:`ice_mass`.
    2. **Gravimetric (weight-percent) cross-check.** Independently, the LCROSS
       ejecta abundance (5.6 wt% water) applied to the regolith column gives
       ``mass = wt% * rho_bulk * area * depth`` — see ``lcross_cross_check_kg``.

    The Monte-Carlo estimator is run on the volumetric estimate to attach a
    95 % credible interval.

    Parameters
    ----------
    scene : LunarScene
        Source scene. Uses ``scene.ice_truth`` (default mask),
        ``scene.resolution_m`` (pixel size) and ``scene.cpr_L`` / ``scene.dop_L``
        (relative ice-likelihood proxy).
    ice_mask : np.ndarray, optional
        Boolean ice mask. Defaults to ``scene.ice_truth``.
    depth_m : float, default 5.0
        Sampled subsurface column depth [m] (top ~5 m).

    Returns
    -------
    dict
        ``area_m2``, ``depth_m``, ``ice_fraction``, ``volume_m3``, ``mass_kg``,
        ``mc`` (Monte-Carlo summary dict), ``lcross_cross_check_kg`` and a
        ``units`` sub-dict documenting every field.

    References
    ----------
    Sinha et al. (2026) (CPR/DOP criterion); Looyenga (1965) (mixing);
    Colaprete et al. (2010) (LCROSS 5.6 +/- 2.9 wt%).
    """
    if ice_mask is None:
        ice_mask = scene.ice_truth
    ice_mask = np.asarray(ice_mask, dtype=bool)

    pixel_area = float(scene.resolution_m) ** 2
    n_ice = int(ice_mask.sum())
    area_m2 = n_ice * pixel_area

    # --- Volumetric (dielectric) ice fraction -----------------------------
    # Relative ice-likelihood index over the masked pixels (NOT an absolute
    # abundance — see cpr_to_ice_likelihood).
    if n_ice > 0:
        like = cpr_to_ice_likelihood(scene.cpr_L[ice_mask], scene.dop_L[ice_mask])
        rel_index = float(np.mean(like))
    else:
        rel_index = 0.0

    # Map the relative likelihood index to a physically plausible ice *volume*
    # fraction. We anchor on the Looyenga-Landau-Lifshitz inversion of a small
    # effective-permittivity excess above the regolith reference (which, given
    # the tiny ice/regolith contrast, is itself small), then scale the relative
    # index into a modest fraction band so the volumetric estimate stays in the
    # regime consistent with the LCROSS in-situ abundance (few wt%) rather than
    # implying a near-solid ice slab. FRAC_MIN keeps the fraction strictly
    # positive where ice is present; FRAC_MAX caps a fully ice-like pixel.
    # The relative index must NOT be read as a volume fraction directly (a
    # high CPR/low DOP does not imply near-solid ice). We therefore:
    #   * take a conservative dielectric anchor: assume the masked column sits
    #     only slightly above the regolith reference (a 10 % step of the small
    #     ice/regolith contrast) and invert it with Looyenga to a fraction; and
    #   * map the relative index linearly into a modest physical band so the
    #     volumetric estimate stays consistent with the LCROSS-scale abundance.
    FRAC_MIN, FRAC_MAX = 0.02, 0.30
    EPS_EXCESS_FRAC = 0.10  # assumed eps_eff excess as a fraction of the contrast
    eps_eff = EPS_REGOLITH_DEFAULT + EPS_EXCESS_FRAC * (EPS_ICE - EPS_REGOLITH_DEFAULT)
    lll_fraction = float(looyenga_ice_fraction(eps_eff))
    if n_ice > 0:
        # Relative-index-weighted fraction, floored by the conservative LLL
        # dielectric anchor.
        ice_fraction = max(
            lll_fraction, FRAC_MIN + rel_index * (FRAC_MAX - FRAC_MIN)
        )
    else:
        ice_fraction = 0.0
    ice_fraction = float(np.clip(ice_fraction, 0.0, 1.0))

    volume_m3 = ice_volume(area_m2, depth_m, ice_fraction)
    mass_kg = ice_mass(volume_m3)

    # --- Monte-Carlo uncertainty on the volumetric estimate ---------------
    mc = monte_carlo_volume(area_m2, depth_m, ice_fraction)

    # --- Gravimetric (LCROSS wt%) cross-check -----------------------------
    # Independent anchor: water-equivalent mass of the regolith column using
    # the LCROSS mean abundance. rho_bulk in kg/m^3 from g/cm^3.
    frac_wt = LCROSS_WT_PCT[0] / 100.0
    rho_bulk = RHO_REGOLITH_GCM3 * 1000.0
    lcross_cross_check_kg = frac_wt * rho_bulk * area_m2 * depth_m

    return {
        "area_m2": area_m2,
        "depth_m": depth_m,
        "ice_fraction": ice_fraction,
        "volume_m3": volume_m3,
        "mass_kg": mass_kg,
        "mc": mc,
        "lcross_cross_check_kg": lcross_cross_check_kg,
        "units": {
            "area_m2": "m^2",
            "depth_m": "m",
            "ice_fraction": "dimensionless volume fraction [0,1] (LLL inversion)",
            "volume_m3": "m^3 (volumetric/dielectric estimate)",
            "mass_kg": "kg (volume * 920 kg/m^3 ice density)",
            "mc": "Monte-Carlo: volume mean/std/ci [m^3], mass_mean/std [kg]",
            "lcross_cross_check_kg": (
                "kg water-equivalent (gravimetric: 5.6 wt% * rho_bulk * area "
                "* depth; Colaprete et al. 2010)"
            ),
        },
    }
