"""Landing-site suitability scoring and selection (AHP-weighted MCDA).

Landing-site selection is a multi-criteria decision-analysis (MCDA) problem:
several conflicting raster criteria (terrain slope, roughness, illumination for
power, Earth-visibility for direct-to-Earth comms, proximity to the science
target / ice) are normalised to a common ``[0, 1]`` "goodness" scale and combined
into a single weighted suitability score.  The criterion weights are derived with
the **Analytic Hierarchy Process** (Saaty 1980): the analyst supplies a
reciprocal pairwise-comparison matrix and the weights are its principal
eigenvector, with a *consistency ratio* flagging incoherent judgements.

Implemented by: **planning agent**.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..constants import LANDING_MAX_SLOPE_DEG

__all__ = ["landing_suitability", "ahp_weights", "select_landing_sites"]

# Saaty's Random Consistency Index (RI) by matrix order n (1..10).
_RANDOM_INDEX = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}

# Criteria where a *higher* raw value is *worse* and must be inverted before
# combining (everything else is treated as higher = better).
_LOWER_IS_BETTER = {"slope", "roughness", "temperature", "hazard", "distance_to_ice"}


def _norm01(arr: np.ndarray, lower_is_better: bool) -> np.ndarray:
    """Min-max normalise to a ``[0, 1]`` goodness raster (1 == most desirable)."""
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        out = np.zeros_like(a)
    else:
        out = (a - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if lower_is_better:
        out = 1.0 - out
    return out


def landing_suitability(
    layers: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Weighted multi-criteria landing-suitability score in ``[0, 1]``.

    Each named criterion raster is normalised to a ``[0, 1]`` *goodness* scale
    (criteria in :data:`_LOWER_IS_BETTER` — e.g. ``"slope"``, ``"roughness"`` —
    are inverted so that smaller raw values score higher) and combined as a
    weight-normalised linear sum.  Cells whose slope exceeds
    :data:`LANDING_MAX_SLOPE_DEG` are hard-penalised to a near-zero score so they
    can never be selected, regardless of their other merits.

    Parameters
    ----------
    layers : mapping[str, np.ndarray]
        Criterion rasters (e.g. ``"slope"``, ``"roughness"``, ``"illumination"``,
        ``"earth_visibility"``, ``"distance_to_ice"``), each shape ``(H, W)``.
    weights : mapping[str, float]
        Per-criterion weights (need not pre-sum to 1 — they are renormalised over
        the criteria actually present).

    Returns
    -------
    np.ndarray
        Suitability score in ``[0, 1]``, shape ``(H, W)``.
    """
    if not layers:
        raise ValueError("landing_suitability requires at least one criterion layer")

    shape = next(iter(layers.values())).shape
    score = np.zeros(shape, dtype=np.float64)

    wsum = sum(float(weights.get(name, 0.0)) for name in layers)
    if wsum <= 0:
        # fall back to equal weighting if none of the named weights apply
        eff = {name: 1.0 / len(layers) for name in layers}
    else:
        eff = {name: float(weights.get(name, 0.0)) / wsum for name in layers}

    for name, raster in layers.items():
        g = _norm01(raster, lower_is_better=(name in _LOWER_IS_BETTER))
        score += eff[name] * g

    score = np.clip(score, 0.0, 1.0)

    # Hard slope penalty: drive cells above the landing slope limit to ~0.
    if "slope" in layers:
        too_steep = np.asarray(layers["slope"], dtype=np.float64) > LANDING_MAX_SLOPE_DEG
        score = np.where(too_steep, 0.0, score)

    return score


def ahp_weights(pairwise: np.ndarray):
    """Analytic-Hierarchy-Process weights from a pairwise-comparison matrix.

    Computes the normalised principal (largest-eigenvalue) eigenvector of the
    reciprocal matrix ``pairwise`` (Saaty 1980) and the **consistency ratio**

        CR = (lambda_max - n) / ((n - 1) * RI)

    where ``RI`` is Saaty's random-index for order ``n``.  ``CR < 0.1`` indicates
    acceptably consistent judgements.

    Parameters
    ----------
    pairwise : np.ndarray
        Positive reciprocal matrix, shape ``(k, k)`` (``a_ij = 1 / a_ji``).

    Returns
    -------
    (weights, CR)
        ``weights`` is a non-negative vector summing to 1, shape ``(k,)``;
        ``CR`` is the consistency ratio (``0.0`` for ``k <= 2``).
    """
    A = np.asarray(pairwise, dtype=np.float64)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("pairwise comparison matrix must be square")

    eigvals, eigvecs = np.linalg.eig(A)
    # principal eigenvalue = the one with the largest real part
    k = int(np.argmax(eigvals.real))
    lam_max = float(eigvals[k].real)
    w = np.abs(eigvecs[:, k].real)
    total = w.sum()
    weights = w / total if total > 0 else np.full(n, 1.0 / n)

    ri = _RANDOM_INDEX.get(n, 1.49)
    if n <= 2 or ri == 0.0:
        cr = 0.0
    else:
        ci = (lam_max - n) / (n - 1)
        cr = ci / ri
    return weights, float(cr)


def select_landing_sites(
    score: np.ndarray,
    n: int,
    traversable: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Pick the top-``n`` landing sites (local suitability maxima).

    Sites are restricted to ``traversable`` cells and returned best-first.  Local
    maxima are preferred (via :func:`skimage.feature.peak_local_max` with a small
    exclusion radius so the returned sites are spatially distinct); if too few
    distinct peaks exist the remainder is filled from the globally highest-scoring
    traversable cells.

    Parameters
    ----------
    score : np.ndarray
        Suitability score, shape ``(H, W)``.
    n : int
        Number of candidate sites to return.
    traversable : np.ndarray
        Boolean mask of acceptable (drivable/safe) pixels.

    Returns
    -------
    list[tuple[int, int, float]]
        ``(row, col, score)`` of the selected sites, best first.
    """
    score = np.asarray(score, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)
    masked = np.where(traversable & np.isfinite(score), score, -np.inf)

    sites: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    # 1) spatially-separated local maxima
    try:
        from skimage.feature import peak_local_max

        finite = masked[np.isfinite(masked)]
        thresh = float(np.percentile(finite, 50)) if finite.size else 0.0
        coords = peak_local_max(
            np.where(np.isfinite(masked), masked, 0.0),
            min_distance=max(2, min(score.shape) // 16),
            threshold_abs=thresh,
            exclude_border=False,
        )
        peaks = [(int(r), int(c)) for r, c in coords if traversable[r, c]]
        peaks.sort(key=lambda rc: masked[rc], reverse=True)
        for (r, c) in peaks:
            if (r, c) in seen:
                continue
            sites.append((r, c, float(score[r, c])))
            seen.add((r, c))
            if len(sites) >= n:
                return sites
    except Exception:
        pass

    # 2) top-up from the globally highest-scoring traversable cells
    flat = masked.ravel()
    order = np.argsort(flat)[::-1]
    for idx in order:
        if not np.isfinite(flat[idx]):
            break
        r, c = int(idx // score.shape[1]), int(idx % score.shape[1])
        if (r, c) in seen:
            continue
        sites.append((r, c, float(score[r, c])))
        seen.add((r, c))
        if len(sites) >= n:
            break
    return sites
