"""Dielectric mixing models and radar penetration depth.

Implemented by: **volume agent**.

Relates the effective permittivity of an ice-regolith mixture to its ice
volume fraction and computes the radar penetration depth that bounds the
sampled subsurface column.

Two complementary mixing laws are provided:

* **Looyenga-Landau-Lifshitz (LLL)** power-law mixing
  (Looyenga 1965; Landau & Lifshitz, *Electrodynamics of Continuous Media*).
  Symmetric in its constituents and accurate for the moderate ice fractions
  expected in a regolith-ice mixture; used here as the *invertible* model that
  maps a measured effective permittivity back to an ice volume fraction.
* **Maxwell-Garnett (MG)** (Maxwell Garnett 1904). A dilute-inclusion
  formula appropriate for sparse, isolated ice grains embedded in a regolith
  host; used as a forward model / cross-check at low ice fractions.

A critical caveat for lunar PSRs: the real permittivity of cold water ice
(``EPS_ICE`` = 3.15) is barely distinguishable from that of dry regolith
(``EPS_REGOLITH_DEFAULT`` = 3.0; Olhoeft & Strangway 1975). The dielectric
contrast is therefore weak, so an *absolute* ice weight-percent is effectively
not retrievable from permittivity alone (Heggy et al. 2012). Polarimetric
radar (CPR/DOP) is used only as a *relative* ice-likelihood index, never as an
absolute abundance — see :func:`cpr_to_ice_likelihood`.

References
----------
* Looyenga, H. (1965), *Physica* 31, 401 — power-law dielectric mixing.
* Landau & Lifshitz, *Electrodynamics of Continuous Media* (LLL mixing).
* Maxwell Garnett, J. C. (1904), *Phil. Trans. R. Soc.* — colour of metal
  glasses (dilute-inclusion mixing).
* Olhoeft, G. R. & Strangway, D. W. (1975), *EPSL* — lunar-sample dielectrics.
* Heggy, E. et al. (2012), *Icarus* — radar sounding of ice-poor regolith.
* Colaprete, A. et al. (2010), *Science* — LCROSS water abundance.
"""

from __future__ import annotations

import numpy as np

from ..constants import EPS_ICE, EPS_REGOLITH_DEFAULT, TAN_DELTA_ICE

__all__ = [
    "looyenga_ice_fraction",
    "maxwell_garnett_eps",
    "looyenga_eps",
    "penetration_depth",
    "cpr_to_ice_likelihood",
]


def looyenga_ice_fraction(
    eps_eff: np.ndarray | float,
    eps_ice: float = EPS_ICE,
    eps_reg: float = EPS_REGOLITH_DEFAULT,
) -> np.ndarray | float:
    """Ice volume fraction from effective permittivity (Looyenga-Landau-Lifshitz).

    Inverts the LLL power-law mixing rule for a two-component (ice + regolith)
    medium:

        eps_eff**(1/3) = f * eps_ice**(1/3) + (1 - f) * eps_reg**(1/3)
        =>  f = (eps_eff**(1/3) - eps_reg**(1/3))
                / (eps_ice**(1/3) - eps_reg**(1/3))

    The result is clipped to the physical range ``[0, 1]``. Because the
    ice/regolith dielectric contrast is small (3.15 vs 3.0), the inversion is
    ill-conditioned and amplifies measurement noise — this is propagated
    explicitly by the Monte-Carlo estimator rather than ignored.

    Parameters
    ----------
    eps_eff : np.ndarray or float
        Effective (measured) real permittivity.
    eps_ice : float, default :data:`EPS_ICE`
        Pure-ice permittivity.
    eps_reg : float, default :data:`EPS_REGOLITH_DEFAULT`
        Host-regolith permittivity.

    Returns
    -------
    np.ndarray or float
        Ice volume fraction, clipped to ``[0, 1]``.

    References
    ----------
    Looyenga (1965); Landau & Lifshitz (LLL mixing).
    """
    eps_eff = np.asarray(eps_eff, dtype=float)
    eps_reg = np.asarray(eps_reg, dtype=float)
    denom = eps_ice ** (1.0 / 3.0) - eps_reg ** (1.0 / 3.0)
    # Guard the (degenerate) zero-contrast case where eps_reg == eps_ice; the
    # fraction is undefined there and is set to 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(
            denom == 0.0,
            0.0,
            (eps_eff ** (1.0 / 3.0) - eps_reg ** (1.0 / 3.0)) / denom,
        )
    f = np.clip(np.nan_to_num(f, nan=0.0), 0.0, 1.0)
    return f if f.ndim else float(f)


def maxwell_garnett_eps(
    f_ice: np.ndarray | float,
    eps_ice: float = EPS_ICE,
    eps_host: float = EPS_REGOLITH_DEFAULT,
) -> np.ndarray | float:
    """Effective permittivity of sparse ice inclusions (Maxwell-Garnett).

    Forward dilute-inclusion mixing for spherical ice grains in a regolith
    host (Maxwell Garnett 1904):

        eps_eff = eps_host * (eps_ice + 2 eps_host + 2 f (eps_ice - eps_host))
                            / (eps_ice + 2 eps_host -   f (eps_ice - eps_host))

    For ``f_ice`` in ``[0, 1]`` this is monotonically increasing in ``f_ice``
    and bounded by ``[eps_host, eps_ice]`` (it returns ``eps_host`` at
    ``f_ice = 0`` and ``eps_ice`` at ``f_ice = 1``).

    Parameters
    ----------
    f_ice : np.ndarray or float
        Ice volume fraction in ``[0, 1]``.
    eps_ice : float, default :data:`EPS_ICE`
        Inclusion (ice) permittivity.
    eps_host : float, default :data:`EPS_REGOLITH_DEFAULT`
        Host (regolith) permittivity.

    Returns
    -------
    np.ndarray or float
        Effective permittivity.

    References
    ----------
    Maxwell Garnett (1904).
    """
    f = np.asarray(f_ice, dtype=float)
    contrast = eps_ice - eps_host
    num = eps_ice + 2.0 * eps_host + 2.0 * f * contrast
    den = eps_ice + 2.0 * eps_host - f * contrast
    eps_eff = eps_host * num / den
    return eps_eff if eps_eff.ndim else float(eps_eff)


def looyenga_eps(
    f_ice: np.ndarray | float,
    eps_ice: float = EPS_ICE,
    eps_reg: float = EPS_REGOLITH_DEFAULT,
) -> np.ndarray | float:
    """Forward Looyenga-Landau-Lifshitz effective permittivity.

    The exact inverse of :func:`looyenga_ice_fraction` (within ``[0, 1]``):

        eps_eff = (f * eps_ice**(1/3) + (1 - f) * eps_reg**(1/3)) ** 3

    Provided so callers can round-trip an assumed ice fraction to an effective
    permittivity and back, and to seed the Monte-Carlo dielectric inversion.

    Parameters
    ----------
    f_ice : np.ndarray or float
        Ice volume fraction in ``[0, 1]``.
    eps_ice : float, default :data:`EPS_ICE`
        Pure-ice permittivity.
    eps_reg : float, default :data:`EPS_REGOLITH_DEFAULT`
        Host-regolith permittivity.

    Returns
    -------
    np.ndarray or float
        Effective permittivity.

    References
    ----------
    Looyenga (1965); Landau & Lifshitz (LLL mixing).
    """
    f = np.asarray(f_ice, dtype=float)
    root = f * eps_ice ** (1.0 / 3.0) + (1.0 - f) * eps_reg ** (1.0 / 3.0)
    eps_eff = root ** 3
    return eps_eff if eps_eff.ndim else float(eps_eff)


def penetration_depth(
    wavelength: float,
    eps: np.ndarray | float,
    tan_delta: float = TAN_DELTA_ICE,
) -> np.ndarray | float:
    """1/e radar power penetration (skin) depth in a low-loss medium.

    For a low-loss dielectric (``tan_delta << 1``) the attenuation-limited
    penetration depth of the radar field is

        delta = wavelength / (2 * pi * sqrt(eps') * tan_delta)   [m]

    With L-band (lambda = 0.2399 m), eps' = 3 and tan_delta = 0.005 this gives
    ~4.4 m, consistent with sounding the top few metres of a PSR regolith
    column. The depth shrinks with increasing loss tangent and with the shorter
    S-band wavelength.

    Parameters
    ----------
    wavelength : float
        Free-space radar wavelength [m] (e.g. L-band 0.2399 m, S-band 0.1199 m).
    eps : np.ndarray or float
        Real permittivity of the medium.
    tan_delta : float, default :data:`TAN_DELTA_ICE`
        Loss tangent.

    Returns
    -------
    np.ndarray or float
        Penetration depth [m].

    References
    ----------
    Standard low-loss skin-depth relation; loss tangent from Carrier, Olhoeft
    & Mendell (1991).
    """
    eps = np.asarray(eps, dtype=float)
    delta = wavelength / (2.0 * np.pi * np.sqrt(eps) * tan_delta)
    return delta if delta.ndim else float(delta)


def cpr_to_ice_likelihood(
    cpr: np.ndarray | float,
    dop: np.ndarray | float,
    cpr_threshold: float = 1.0,
    dop_threshold: float = 0.13,
    cpr_scale: float = 0.05,
    dop_scale: float = 0.05,
) -> np.ndarray | float:
    """Map (CPR, DOP) to a *relative* subsurface-ice likelihood index in [0, 1].

    .. important::
       This is a **relative likelihood index**, *not* an absolute ice
       weight-percent or volume fraction. Because the real permittivity of cold
       water ice (~3.15) is nearly identical to that of dry regolith (~3.0),
       absolute abundance is not retrievable from radar permittivity
       (Heggy et al. 2012). The polarimetric Sinha et al. (2026) criterion
       (CPR > 1 *and* DOP < 0.13) discriminates *coherent backscatter from
       subsurface ice* against rough-surface decoys, so we use it only to rank
       relative ice likelihood across the scene.

    The index is a product of two logistic gates that smoothly reproduce the
    hard criterion:

        L = sigma((CPR - cpr_threshold) / cpr_scale)
            * sigma((dop_threshold - DOP) / dop_scale)

    so ``L -> 1`` for high-CPR / low-DOP (ice-like coherent backscatter) and
    ``L -> 0`` for low-CPR or high-DOP (rough-surface / volume-scatter decoys).

    Parameters
    ----------
    cpr : np.ndarray or float
        Circular polarisation ratio.
    dop : np.ndarray or float
        Degree of polarisation.
    cpr_threshold : float, default 1.0
        CPR decision boundary (Sinha et al. 2026).
    dop_threshold : float, default 0.13
        DOP decision boundary (Sinha et al. 2026).
    cpr_scale, dop_scale : float
        Logistic softness of the CPR / DOP gates.

    Returns
    -------
    np.ndarray or float
        Relative ice-likelihood index in ``[0, 1]`` (NOT an absolute wt%).

    References
    ----------
    Sinha et al. (2026), *npj Space Exploration*; Heggy et al. (2012),
    *Icarus*.
    """
    cpr = np.asarray(cpr, dtype=float)
    dop = np.asarray(dop, dtype=float)
    cpr_gate = 1.0 / (1.0 + np.exp(-(cpr - cpr_threshold) / cpr_scale))
    dop_gate = 1.0 / (1.0 + np.exp(-(dop_threshold - dop) / dop_scale))
    likelihood = cpr_gate * dop_gate
    likelihood = np.clip(likelihood, 0.0, 1.0)
    return likelihood if likelihood.ndim else float(likelihood)
