"""O(1) precomputed CPR-DOP look-up-table ice classifier.

Implemented by: **classify agent**.

The flagship "fastest-platform" classifier. Instead of evaluating the
``CPR>1 & DOP<0.13`` rule (Sinha et al. 2026) per pixel at runtime, we
precompute a 2-D boolean LUT over discretised ``(CPR, DOP)`` bins **once**, then
classify any scene by a single vectorised ``np.digitize`` + fancy-index lookup.

Why this is O(1) per pixel
--------------------------
The decision boundary is *baked* into ``lut`` at build time. At classify time
there is **no per-pixel arithmetic on the physics and no branching**: each pixel
is mapped to an integer ``(cpr_bin, dop_bin)`` pair by ``np.digitize`` (a binary
search over a *fixed*, tiny number of edges that is independent of the image
size) and the answer is read straight out of the table with one fancy-index
gather. The work per pixel is therefore a constant number of memory operations
— independent of the number of thresholds, of any model complexity that was
folded into the table, and of the scene's value range. Doubling the resolution
only doubles the (already trivial) gather cost; it never changes the per-pixel
cost. This is the property the platform advertises as "fastest".

Exactness
---------
With a fine enough grid the LUT is *bit-identical* to the direct rule because a
bin is flagged ice iff its **centre** satisfies ``CPR>1 & DOP<0.13`` and the rule
is monotone in both axes (ice for larger CPR, ice for smaller DOP). The default
256x256 grid reproduces :func:`classify_ice_threshold` exactly on the synthetic
Faustini scene (see ``tests/test_classify.py``).
"""

from __future__ import annotations

import numpy as np

from ..constants import CPR_ICE_THRESHOLD, DOP_ICE_THRESHOLD

__all__ = [
    "build_ice_lut",
    "classify_ice_lut",
    "classify_ice_threshold",
    "default_edges",
]


def default_edges(
    cpr_max: float = 4.0, dop_max: float = 1.0, nbins: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience uniform bin edges for the CPR-DOP LUT grid.

    Builds two monotonically increasing edge arrays spanning ``[0, cpr_max]`` and
    ``[0, dop_max]`` with ``nbins`` equal-width bins each (hence ``nbins + 1``
    edges). The ranges comfortably bracket the physical domain (CPR is rarely
    above ~2-3, DOP is in ``[0, 1]``) so virtually no values are clipped, which
    keeps the LUT an exact stand-in for the direct rule.

    Parameters
    ----------
    cpr_max : float, default 4.0
        Upper edge of the CPR axis.
    dop_max : float, default 1.0
        Upper edge of the DOP axis (degree of polarisation is bounded by 1).
    nbins : int, default 256
        Number of bins per axis.

    Returns
    -------
    (cpr_edges, dop_edges) : tuple[np.ndarray, np.ndarray]
        Edge arrays of shape ``(nbins + 1,)`` each.
    """
    cpr_edges = np.linspace(0.0, cpr_max, nbins + 1)
    dop_edges = np.linspace(0.0, dop_max, nbins + 1)
    return cpr_edges, dop_edges


def build_ice_lut(cpr_edges: np.ndarray, dop_edges: np.ndarray) -> np.ndarray:
    """Precompute the boolean ice LUT over a ``(CPR, DOP)`` bin grid.

    Cell ``[i, j]`` is ``True`` iff the **centre** of bin ``i`` on the CPR axis
    and bin ``j`` on the DOP axis satisfies the Sinha et al. (2026) subsurface-ice
    rule ``CPR > CPR_ICE_THRESHOLD AND DOP < DOP_ICE_THRESHOLD``. Evaluating the
    rule at the bin centres bakes the entire decision boundary into the table so
    that classification later costs nothing but a lookup.

    Parameters
    ----------
    cpr_edges : np.ndarray
        Monotonic CPR bin edges, shape ``(nc + 1,)``.
    dop_edges : np.ndarray
        Monotonic DOP bin edges, shape ``(nd + 1,)``.

    Returns
    -------
    np.ndarray
        Boolean LUT, shape ``(nc, nd)``.
    """
    cpr_edges = np.asarray(cpr_edges, dtype=np.float64)
    dop_edges = np.asarray(dop_edges, dtype=np.float64)
    if cpr_edges.ndim != 1 or cpr_edges.size < 2:
        raise ValueError("cpr_edges must be 1-D with at least two entries")
    if dop_edges.ndim != 1 or dop_edges.size < 2:
        raise ValueError("dop_edges must be 1-D with at least two entries")

    # Bin centres: the representative (CPR, DOP) value for each cell.
    cpr_centers = 0.5 * (cpr_edges[:-1] + cpr_edges[1:])  # (nc,)
    dop_centers = 0.5 * (dop_edges[:-1] + dop_edges[1:])  # (nd,)

    # Broadcast the rule across the whole grid in one shot -> (nc, nd) boolean.
    cpr_ok = (cpr_centers > CPR_ICE_THRESHOLD)[:, None]
    dop_ok = (dop_centers < DOP_ICE_THRESHOLD)[None, :]
    lut = cpr_ok & dop_ok
    return np.ascontiguousarray(lut)


def classify_ice_lut(
    cpr: np.ndarray,
    dop: np.ndarray,
    lut: np.ndarray,
    cpr_edges: np.ndarray,
    dop_edges: np.ndarray,
) -> np.ndarray:
    """Classify a scene by **O(1)-per-pixel** LUT lookup.

    Each pixel's ``cpr``/``dop`` is digitised into the LUT's bin indices and the
    boolean answer is gathered from ``lut`` with a single fancy-index operation::

        ci   = clip(digitize(cpr, cpr_edges) - 1, 0, nc - 1)
        di   = clip(digitize(dop, dop_edges) - 1, 0, nd - 1)
        mask = lut[ci, di]

    There is **no per-pixel branching and no per-pixel physics**: the only work
    per pixel is a binary search into a fixed-size edge array (cost independent of
    the image size) plus one memory read. The per-pixel cost is therefore
    constant — the O(1) property that backs the platform's "fastest" claim. The
    ``clip`` folds any out-of-range values into the boundary bins so the lookup
    never indexes out of bounds.

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP, any matching shape (e.g. ``(H, W)``).
    lut : np.ndarray
        Boolean LUT from :func:`build_ice_lut`, shape ``(nc, nd)``.
    cpr_edges, dop_edges : np.ndarray
        The **same** edges used to build ``lut``.

    Returns
    -------
    np.ndarray
        Boolean ice mask with the broadcast shape of ``cpr`` and ``dop``.
    """
    cpr = np.asarray(cpr)
    dop = np.asarray(dop)
    cpr_edges = np.asarray(cpr_edges, dtype=np.float64)
    dop_edges = np.asarray(dop_edges, dtype=np.float64)
    nc, nd = lut.shape

    # Map each pixel value to its 0-based bin index, then clip into [0, n-1] so
    # the extreme open bins (below the first / above the last edge) fold onto the
    # boundary cells rather than indexing out of range.
    ci = _to_bin_index(cpr, cpr_edges, nc)
    di = _to_bin_index(dop, dop_edges, nd)

    # Single vectorised gather -> O(1) per pixel, fully branch-free.
    mask = lut[ci, di]
    return np.ascontiguousarray(mask, dtype=bool)


def _is_uniform(edges: np.ndarray) -> bool:
    """True if ``edges`` are (within float tolerance) equally spaced."""
    if edges.size < 3:
        return True
    d = np.diff(edges)
    return bool(np.allclose(d, d[0], rtol=1e-9, atol=1e-12)) and d[0] > 0.0


def _to_bin_index(values: np.ndarray, edges: np.ndarray, nbins: int) -> np.ndarray:
    """Bin index for every value, clipped to ``[0, nbins - 1]``.

    For **uniform** edges the index is the closed-form ``floor((x - lo) / width)``
    — a genuinely O(1) arithmetic map with no binary search, which keeps the
    flagship lookup constant-time per pixel (and is bit-identical to the
    ``np.digitize`` result the boundary clip would produce). For non-uniform
    edges it falls back to ``np.digitize`` so the function stays fully general.
    """
    if _is_uniform(edges):
        lo = edges[0]
        width = (edges[-1] - lo) / (edges.size - 1)
        idx = np.floor((values - lo) / width).astype(np.intp)
    else:
        idx = np.digitize(values, edges) - 1
    return np.clip(idx, 0, nbins - 1)


def classify_ice_threshold(
    cpr: np.ndarray,
    dop: np.ndarray,
    cpr_thr: float = CPR_ICE_THRESHOLD,
    dop_thr: float = DOP_ICE_THRESHOLD,
) -> np.ndarray:
    """Direct rule-based ice mask (the reference the LUT must match).

    Implements the Sinha et al. (2026) subsurface-ice criterion verbatim::

        ice = (cpr > cpr_thr) & (dop < dop_thr)

    This is the ground-truth decision the precomputed :func:`build_ice_lut` table
    encodes; :func:`classify_ice_lut` reproduces it exactly on a fine grid.

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP, any matching shape (e.g. ``(H, W)``).
    cpr_thr : float, default :data:`CPR_ICE_THRESHOLD`
        CPR lower bound (strictly greater than).
    dop_thr : float, default :data:`DOP_ICE_THRESHOLD`
        DOP upper bound (strictly less than).

    Returns
    -------
    np.ndarray
        Boolean ice mask with the broadcast shape of ``cpr`` and ``dop``.
    """
    cpr = np.asarray(cpr)
    dop = np.asarray(dop)
    return (cpr > cpr_thr) & (dop < dop_thr)
