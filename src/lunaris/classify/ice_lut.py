"""O(1) precomputed CPR-DOP look-up-table ice classifier.

Implemented by: **classify agent**.

The flagship "fastest-platform" classifier. Instead of evaluating the
``CPR>1 & DOP<0.13`` rule (plus any learned refinements) per pixel at runtime,
we precompute a 2-D boolean LUT over discretised ``(CPR, DOP)`` bins once, then
classify any scene by a single vectorised ``np.digitize`` + fancy-index lookup —
O(1) per pixel, no branches.
"""

from __future__ import annotations

import numpy as np

from ..constants import CPR_ICE_THRESHOLD, DOP_ICE_THRESHOLD

__all__ = ["build_ice_lut", "classify_ice_lut", "classify_ice_threshold"]


def build_ice_lut(cpr_edges: np.ndarray, dop_edges: np.ndarray) -> np.ndarray:
    """Precompute the boolean ice LUT over a ``(CPR, DOP)`` bin grid.

    Cell ``[i, j]`` is ``True`` iff a pixel whose CPR falls in
    ``cpr_edges[i:i+1]`` and DOP in ``dop_edges[j:j+1]`` should be classified as
    ice (default rule: bin centre satisfies ``CPR>1 & DOP<0.13``; the agent may
    encode a smoother learned decision boundary).

    Parameters
    ----------
    cpr_edges : np.ndarray
        Monotonic CPR bin edges, shape ``(nc+1,)``.
    dop_edges : np.ndarray
        Monotonic DOP bin edges, shape ``(nd+1,)``.

    Returns
    -------
    np.ndarray
        Boolean LUT, shape ``(nc, nd)``.
    """
    raise NotImplementedError("classify agent")


def classify_ice_lut(
    cpr: np.ndarray,
    dop: np.ndarray,
    lut: np.ndarray,
    cpr_edges: np.ndarray,
    dop_edges: np.ndarray,
) -> np.ndarray:
    """Classify a scene by O(1)-per-pixel LUT lookup.

    Digitises ``cpr``/``dop`` into the LUT bins and indexes ``lut`` — a single
    vectorised gather with no per-pixel branching.

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP, shape ``(H, W)``.
    lut : np.ndarray
        Boolean LUT from :func:`build_ice_lut`, shape ``(nc, nd)``.
    cpr_edges, dop_edges : np.ndarray
        The same edges used to build ``lut``.

    Returns
    -------
    np.ndarray
        Boolean ice mask, shape ``(H, W)``.
    """
    raise NotImplementedError("classify agent")


def classify_ice_threshold(
    cpr: np.ndarray,
    dop: np.ndarray,
    cpr_thr: float = CPR_ICE_THRESHOLD,
    dop_thr: float = DOP_ICE_THRESHOLD,
) -> np.ndarray:
    """Direct rule-based ice mask (reference for the LUT).

        ice = (cpr > cpr_thr) & (dop < dop_thr)

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP, shape ``(H, W)``.
    cpr_thr : float, default :data:`CPR_ICE_THRESHOLD`
        CPR lower bound.
    dop_thr : float, default :data:`DOP_ICE_THRESHOLD`
        DOP upper bound.

    Returns
    -------
    np.ndarray
        Boolean ice mask, shape ``(H, W)``.
    """
    raise NotImplementedError("classify agent")
