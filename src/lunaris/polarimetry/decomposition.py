"""Hybrid-polarity (compact-pol) decompositions: m-chi, m-delta, CHILD, and
optional Cloude-Pottier.

Implemented by: **polarimetry agent**.

These separate the backscatter into even-bounce / volume / odd-bounce (or
double / volume / surface) power components, isolating the volume-scatter
channel that dominates over subsurface ice.

References
----------
Raney, R. K. et al. (2012). "The m-chi decomposition of hybrid dual-polarimetric
    radar data." IGARSS — m-chi and m-delta.
Kumar, S. et al. (2015). CHILD / hybrid-pol child parameters.
Cloude, S. R. & Pottier, E. (1997). "An entropy based classification scheme for
    land applications of polarimetric SAR." IEEE TGRS 35(1) — H/A/alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ["m_chi", "m_delta", "child_parameters", "cloude_pottier"]

_EPS = 1e-12


def _dop(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> np.ndarray:
    """Degree of polarisation ``m = sqrt(s1^2+s2^2+s3^2)/s0`` clipped to [0,1]."""
    pol = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
    m = pol / np.where(np.abs(s0) < _EPS, _EPS, s0)
    return np.clip(m, 0.0, 1.0)


def m_chi(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m-chi decomposition (Raney et al. 2012).

    With degree of polarisation ``m = sqrt(s1^2+s2^2+s3^2)/s0`` and ellipticity
    ``chi = 0.5 arcsin(-s3/(m s0))`` (so that ``m s0 sin 2chi = -s3``):

        even   = sqrt( (m s0 - s3) / 2 )   = sqrt(m s0 (1 + sin 2chi) / 2)
        volume = sqrt( s0 (1 - m) )
        odd    = sqrt( (m s0 + s3) / 2 )   = sqrt(m s0 (1 - sin 2chi) / 2)

    (Raney 2012, Eqs. 6-8.) Each radicand is clipped to ``[0, inf)`` before the
    square root to absorb speckle-induced negative excursions. Over subsurface
    ice the same-sense ``s3 < 0`` drives the **volume** channel (RGB green)
    dominant — the diagnostic ice signature.

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (even, volume, odd) : tuple of np.ndarray
        Power components, shape ``(H, W)`` (RGB-ready: R=even, G=volume, B=odd).
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    m = _dop(s0, s1, s2, s3)
    ms0 = m * s0
    even = np.sqrt(np.clip((ms0 - s3) / 2.0, 0.0, None))
    volume = np.sqrt(np.clip(s0 * (1.0 - m), 0.0, None))
    odd = np.sqrt(np.clip((ms0 + s3) / 2.0, 0.0, None))
    return even, volume, odd


def m_delta(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m-delta decomposition (Raney et al. 2012).

    Uses the relative phase ``delta = arctan2(s3, s2)`` between the circular
    Stokes components instead of ellipticity:

        double  = sqrt( m s0 (1 + sin delta) / 2 )
        volume  = sqrt( s0 (1 - m) )
        surface = sqrt( m s0 (1 - sin delta) / 2 )

    (Raney 2012, Eqs. 9-11.) Radicands clipped to ``[0, inf)`` before sqrt.

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    (double, volume, surface) : tuple of np.ndarray
        Power components, shape ``(H, W)``.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    m = _dop(s0, s1, s2, s3)
    ms0 = m * s0
    delta = np.arctan2(s3, s2)
    sin_d = np.sin(delta)
    double = np.sqrt(np.clip(ms0 * (1.0 + sin_d) / 2.0, 0.0, None))
    volume = np.sqrt(np.clip(s0 * (1.0 - m), 0.0, None))
    surface = np.sqrt(np.clip(ms0 * (1.0 - sin_d) / 2.0, 0.0, None))
    return double, volume, surface


def child_parameters(
    s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray
) -> dict[str, np.ndarray]:
    """CHILD / hybrid-pol descriptors (Kumar et al. 2015).

        chi        = 0.5 arcsin( clip(-s3 / (m s0), -1, 1) )   ellipticity
        delta      = arctan2(s3, s2)                           relative phase
        psi        = 0.5 arctan2(s2, s1)                       orientation
        conformity = -s3 / s0                                  surface/volume sign

    The ``arcsin`` argument is clipped to ``[-1, 1]`` so ``chi`` stays inside
    ``[-pi/4, pi/4]`` ([-45, 45] deg) even where speckle pushes ``|s3|`` past
    ``m s0``. Angles are returned in radians, with degree convenience copies
    ``chi_deg`` and ``delta_deg``.

    Parameters
    ----------
    s0, s1, s2, s3 : np.ndarray
        Stokes parameters, shape ``(H, W)``.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"chi"``, ``"delta"``, ``"psi"``, ``"conformity"`` (radians) plus
        ``"chi_deg"`` and ``"delta_deg"`` (degrees).
    """
    s0 = np.asarray(s0, dtype=np.float64)
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    s3 = np.asarray(s3, dtype=np.float64)
    m = _dop(s0, s1, s2, s3)
    ms0 = m * s0
    safe_ms0 = np.where(np.abs(ms0) < _EPS, _EPS, ms0)
    safe_s0 = np.where(np.abs(s0) < _EPS, _EPS, s0)

    chi = 0.5 * np.arcsin(np.clip(-s3 / safe_ms0, -1.0, 1.0))
    delta = np.arctan2(s3, s2)
    psi = 0.5 * np.arctan2(s2, s1)
    conformity = -s3 / safe_s0

    return {
        "chi": chi,
        "delta": delta,
        "psi": psi,
        "conformity": conformity,
        "chi_deg": np.degrees(chi),
        "delta_deg": np.degrees(delta),
    }


def cloude_pottier(
    E_RH: np.ndarray | None = None,
    E_RV: np.ndarray | None = None,
    window: int = 5,
    *,
    s0: np.ndarray | None = None,
    s1: np.ndarray | None = None,
    s2: np.ndarray | None = None,
    s3: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Cloude-Pottier H / A / alpha eigen-decomposition (dual-pol, 2x2).

    Builds the per-pixel 2x2 Hermitian covariance (Wishart) matrix of the
    circular receive vector ``k = [E_RH, E_RV]^T`` averaged over a boxcar window

        J = <k k^H> = [[<|E_RH|^2>,      <E_RH E_RV*>],
                       [<E_RV E_RH*>,    <|E_RV|^2>  ]]

    and eigen-decomposes it. From the two non-negative eigenvalues
    ``lambda_1 >= lambda_2`` and pseudo-probabilities ``p_i = lambda_i / sum``:

        entropy     H = -sum_i p_i log2 p_i / log2(2)      in [0, 1]
        anisotropy  A = (lambda_1 - lambda_2)/(lambda_1 + lambda_2)
        alpha       = sum_i p_i * alpha_i  (mean scattering angle, radians)

    where ``alpha_i = arccos(|v_i[0]|)`` from the dominant component of each
    unit eigenvector (Cloude & Pottier 1997, adapted to the 2x2 hybrid-pol
    coherency matrix). The eigenproblems are solved batched with
    :func:`numpy.linalg.eigh` over the stacked ``(H*W, 2, 2)`` array.

    Either pass the circular fields ``E_RH, E_RV`` (preferred) or pre-averaged
    Stokes parameters ``s0, s1, s2, s3`` (the covariance is then reconstructed
    as ``J = 0.5 [[s0+s1, s2+j s3],[s2-j s3, s0-s1]]``).

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"entropy"``, ``"anisotropy"``, ``"alpha"`` (alpha in radians,
        with ``"alpha_deg"`` in degrees), each shape ``(H, W)``.
    """
    if E_RH is not None and E_RV is not None:
        E_RH = np.asarray(E_RH, dtype=np.complex128)
        E_RV = np.asarray(E_RV, dtype=np.complex128)

        def _avg(a: np.ndarray) -> np.ndarray:
            return uniform_filter(a, size=window, mode="reflect")

        j00 = _avg(np.abs(E_RH) ** 2)
        j11 = _avg(np.abs(E_RV) ** 2)
        cross = _avg(E_RH * np.conj(E_RV))   # j01
        shape = E_RH.shape
    elif s0 is not None and s1 is not None and s2 is not None and s3 is not None:
        s0 = np.asarray(s0, dtype=np.float64)
        s1 = np.asarray(s1, dtype=np.float64)
        s2 = np.asarray(s2, dtype=np.float64)
        s3 = np.asarray(s3, dtype=np.float64)
        # Inverse of the circular-Stokes definitions (J Hermitian, 2x2).
        j00 = 0.5 * (s0 + s1)
        j11 = 0.5 * (s0 - s1)
        cross = 0.5 * (s2 + 1j * s3)
        shape = s0.shape
    else:  # pragma: no cover - usage guard
        raise ValueError(
            "cloude_pottier: provide either (E_RH, E_RV) or (s0, s1, s2, s3)."
        )

    n = int(np.prod(shape)) if shape else 1
    J = np.empty((n, 2, 2), dtype=np.complex128)
    J[:, 0, 0] = j00.reshape(-1)
    J[:, 1, 1] = j11.reshape(-1)
    J[:, 0, 1] = cross.reshape(-1)
    J[:, 1, 0] = np.conj(cross).reshape(-1)

    # Hermitian eigendecomposition; eigh returns ascending eigenvalues and the
    # corresponding orthonormal eigenvectors in columns.
    evals, evecs = np.linalg.eigh(J)            # evals (n,2) ascending
    evals = np.clip(evals.real, 0.0, None)
    total = evals.sum(axis=1, keepdims=True)
    safe_total = np.where(total < _EPS, _EPS, total)
    p = evals / safe_total                      # (n, 2)

    # Shannon entropy normalised by log2(2)=1 -> H in [0,1].
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0.0, np.log2(p), 0.0)
    H = -(p * logp).sum(axis=1) / np.log2(2.0)

    lam2 = evals[:, 0]                           # smaller
    lam1 = evals[:, 1]                           # larger
    denom = lam1 + lam2
    anisotropy = (lam1 - lam2) / np.where(denom < _EPS, _EPS, denom)

    # alpha_i from the leading component magnitude of each eigenvector
    # (column j of evecs is the eigenvector for evals[:, j]).
    comp0 = np.abs(evecs[:, 0, :])              # |v_i[0]| for both eigenvectors
    alpha_i = np.arccos(np.clip(comp0, 0.0, 1.0))   # (n, 2)
    alpha = (p * alpha_i).sum(axis=1)

    H = np.clip(H, 0.0, 1.0).reshape(shape)
    anisotropy = np.clip(anisotropy, 0.0, 1.0).reshape(shape)
    alpha = alpha.reshape(shape)

    return {
        "entropy": H,
        "anisotropy": anisotropy,
        "alpha": alpha,
        "alpha_deg": np.degrees(alpha),
    }
