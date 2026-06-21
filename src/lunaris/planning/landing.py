"""Landing-site suitability scoring and selection (AHP-weighted MCDA).

Implemented by: **planning agent**.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["landing_suitability", "ahp_weights", "select_landing_sites"]


def landing_suitability(
    layers: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    """Weighted multi-criteria landing-suitability score in ``[0, 1]``.

    Each named criterion layer (slope, roughness, illumination, Earth-visibility,
    proximity-to-ice, ...) is normalised and combined with ``weights`` (e.g. from
    :func:`ahp_weights`).

    Parameters
    ----------
    layers : mapping[str, np.ndarray]
        Criterion rasters, each shape ``(H, W)``.
    weights : mapping[str, float]
        Per-criterion weights (should sum to 1).

    Returns
    -------
    np.ndarray
        Suitability score in ``[0, 1]``, shape ``(H, W)``.
    """
    raise NotImplementedError("planning agent")


def ahp_weights(pairwise: np.ndarray) -> np.ndarray:
    """Analytic-Hierarchy-Process criterion weights from a pairwise matrix.

    Returns the principal-eigenvector (normalised) weights of the reciprocal
    pairwise-comparison matrix ``pairwise`` (Saaty 1980); the consistency ratio
    should be checked < 0.1.

    Parameters
    ----------
    pairwise : np.ndarray
        Positive reciprocal matrix, shape ``(k, k)``.

    Returns
    -------
    np.ndarray
        Weight vector summing to 1, shape ``(k,)``.
    """
    raise NotImplementedError("planning agent")


def select_landing_sites(
    score: np.ndarray,
    n: int,
    traversable: np.ndarray,
) -> list[tuple[int, int]]:
    """Pick the top-``n`` non-adjacent landing sites from a suitability score.

    Parameters
    ----------
    score : np.ndarray
        Suitability score, shape ``(H, W)``.
    n : int
        Number of candidate sites to return.
    traversable : np.ndarray
        Boolean mask of acceptable (e.g. low-slope, reachable) pixels.

    Returns
    -------
    list[tuple[int, int]]
        ``(row, col)`` of the selected sites, best first.
    """
    raise NotImplementedError("planning agent")
