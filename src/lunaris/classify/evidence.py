"""Probabilistic evidence fusion: Bayesian and Dempster-Shafer.

Implemented by: **classify agent**.

Combines independent ice-evidence layers (radar criterion, thermal cold-trap
stability, permanent shadow, UV/NIR frost proxies) into a single posterior ice
probability / belief map. Multi-evidence corroboration is what lets the platform
*reject* CPR rock/roughness false positives that a single sensor cannot: a rough
rim pixel may shout "ice" on CPR alone, but it is warm, sunlit and frost-free, so
the fused confidence stays low.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..scene import LunarScene

__all__ = ["bayesian_fusion", "dempster_shafer", "multi_evidence_ice"]


# Probabilities are clamped this far inside (0, 1) before any log-odds transform
# so logits stay finite (a layer asserting p == 0 or p == 1 would otherwise send
# the posterior to +/-inf and veto every other layer).
_EPS = 1e-6


def bayesian_fusion(
    evidence_layers: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted Bayesian (log-odds) fusion of evidence probability layers.

    Under a **naive-Bayes** assumption — the evidence layers are conditionally
    independent given the ice / no-ice class, with a uniform prior ``P(ice)=0.5``
    — the posterior log-odds is the (weighted) sum of the per-layer log-odds::

        logit(P) = sum_k w_k * logit(p_k)
        P        = sigmoid(logit(P))

    The weights let stronger sensors count more; equal weights reproduce plain
    naive-Bayes accumulation. Each ``p_k`` is clipped to ``[eps, 1 - eps]`` so the
    logit is finite. The result is a per-pixel posterior probability in ``[0, 1]``.

    Parameters
    ----------
    evidence_layers : sequence of np.ndarray
        Per-layer ice probabilities in ``[0, 1]``, each broadcastable to a common
        shape ``(H, W)``. Must be non-empty.
    weights : sequence of float, optional
        Per-layer weights (same length as ``evidence_layers``). Defaults to all
        ones (unweighted naive-Bayes).

    Returns
    -------
    np.ndarray
        Posterior ice probability in ``[0, 1]``, shape ``(H, W)``.
    """
    layers = [np.asarray(layer, dtype=np.float64) for layer in evidence_layers]
    if not layers:
        raise ValueError("evidence_layers must contain at least one layer")
    if weights is None:
        weights = np.ones(len(layers), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != len(layers):
            raise ValueError("weights must match the number of evidence_layers")

    logit_sum = np.zeros(np.broadcast_shapes(*[layer.shape for layer in layers]))
    for layer, w in zip(layers, weights, strict=True):
        p = np.clip(np.nan_to_num(layer, nan=0.5), _EPS, 1.0 - _EPS)
        logit_sum = logit_sum + w * np.log(p / (1.0 - p))

    posterior = 1.0 / (1.0 + np.exp(-logit_sum))
    return np.clip(posterior, 0.0, 1.0)


def dempster_shafer(
    masks: Sequence[np.ndarray],
    masses: Sequence[float] | None = None,
) -> np.ndarray:
    """Dempster-Shafer combination of belief masks (2-hypothesis frame).

    Works on the frame of discernment ``{ice, not-ice}`` with an explicit
    *uncertainty* mass on ``{ice, not-ice}`` (the whole frame). Each source ``k``
    contributes a basic belief assignment derived from its evidence map
    ``e_k in [0, 1]`` and a reliability ``mass_k in [0, 1]``::

        m_k(ice)     = mass_k * e_k
        m_k(not-ice) = mass_k * (1 - e_k)
        m_k(theta)   = 1 - mass_k          # residual / "don't know"

    Sources are combined pairwise with **Dempster's rule of combination**,
    normalising away the conflict mass ``K = m_a(ice)*m_b(not-ice) +
    m_a(not-ice)*m_b(ice)``. The returned map is the combined **belief in ice**,
    ``Bel(ice) = m(ice)``, in ``[0, 1]``. The uncertainty mass keeps the rule
    robust when sources disagree (no division-by-zero: total conflict ``K==1`` is
    handled by falling back to the uninformative ``0.5``).

    Parameters
    ----------
    masks : sequence of np.ndarray
        Per-source evidence in ``[0, 1]`` (boolean masks are accepted and read as
        0/1), each broadcastable to a common shape ``(H, W)``. Must be non-empty.
    masses : sequence of float, optional
        Per-source reliability mass in ``[0, 1]`` (how much total belief the
        source commits vs. leaves to uncertainty). Defaults to ``0.9`` each.

    Returns
    -------
    np.ndarray
        Combined belief in ice, shape ``(H, W)``, values in ``[0, 1]``.
    """
    arrays = [np.asarray(m, dtype=np.float64) for m in masks]
    if not arrays:
        raise ValueError("masks must contain at least one mask")
    if masses is None:
        masses = [0.9] * len(arrays)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape[0] != len(arrays):
        raise ValueError("masses must match the number of masks")

    shape = np.broadcast_shapes(*[a.shape for a in arrays])

    # Initialise the accumulator with the first source's BBA.
    e0 = np.clip(np.nan_to_num(arrays[0], nan=0.5), 0.0, 1.0)
    w0 = float(np.clip(masses[0], 0.0, 1.0))
    m_ice = np.broadcast_to(w0 * e0, shape).astype(np.float64).copy()
    m_not = np.broadcast_to(w0 * (1.0 - e0), shape).astype(np.float64).copy()
    m_theta = np.broadcast_to(1.0 - w0, shape).astype(np.float64).copy()

    for arr, mass in zip(arrays[1:], masses[1:], strict=True):
        e = np.clip(np.nan_to_num(arr, nan=0.5), 0.0, 1.0)
        w = float(np.clip(mass, 0.0, 1.0))
        b_ice = w * e
        b_not = w * (1.0 - e)
        b_theta = 1.0 - w

        # Conflict mass (ice vs not-ice intersect to the empty set).
        K = m_ice * b_not + m_not * b_ice
        denom = 1.0 - K

        # Unnormalised combined masses (theta only survives theta∩theta plus
        # the cross terms with theta).
        c_ice = m_ice * b_ice + m_ice * b_theta + m_theta * b_ice
        c_not = m_not * b_not + m_not * b_theta + m_theta * b_not
        c_theta = m_theta * b_theta

        # Normalise; where the sources totally conflict (denom == 0) fall back to
        # a flat, fully-uncertain assignment so the output stays finite.
        safe = denom > _EPS
        inv = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)
        m_ice = np.where(safe, c_ice * inv, 0.5)
        m_not = np.where(safe, c_not * inv, 0.5)
        m_theta = np.where(safe, c_theta * inv, 0.0)

    return np.clip(m_ice, 0.0, 1.0)


def _normalize01(x: np.ndarray) -> np.ndarray:
    """Min-max scale a finite array into ``[0, 1]`` (flat input -> zeros)."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo <= _EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def multi_evidence_ice(scene: LunarScene) -> np.ndarray:
    """Build per-pixel evidence layers from a scene and fuse them to confidence.

    Constructs five independent, physically-motivated ice-evidence maps directly
    from the scene (no dependency on the ``terrain`` subpackage) and fuses them
    with :func:`bayesian_fusion` into a single confidence map in ``[0, 1]``:

    * **Permanent shadow (PSR)** — ``illumination < 0.05`` (ice survives only
      where the Sun never reaches).
    * **Cold trap** — ``temperature_max < 110 K`` (the water-ice cold-trap
      threshold), strength scaled by how far below 110 K the pixel sits.
    * **Frost albedo** — high normalised ``albedo_1064`` (bright 1064-nm frost).
    * **LAMP frost** — high normalised ``lamp_ratio`` (UV off/on-band frost
      signature).
    * **Radar criterion** — the Sinha et al. (2026) ``cpr_L > 1 & dop_L < 0.13``
      rule, the single strongest layer (given a heavier weight).

    Because a CPR rock/roughness false positive lights up *only* the radar layer
    while remaining warm, sunlit and frost-free, naive-Bayes fusion drags its
    confidence back down — the corroboration that rejects roughness decoys.

    Parameters
    ----------
    scene : LunarScene
        Source scene of shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        Per-pixel ice confidence in ``[0, 1]``, shape ``(H, W)``.
    """
    illum = np.asarray(scene.illumination, dtype=np.float64)
    temp = np.asarray(scene.temperature_max, dtype=np.float64)
    albedo = np.asarray(scene.albedo_1064, dtype=np.float64)
    lamp = np.asarray(scene.lamp_ratio, dtype=np.float64)
    cpr = np.asarray(scene.cpr_L, dtype=np.float64)
    dop = np.asarray(scene.dop_L, dtype=np.float64)

    # 1) Permanent shadow: soft probability that the pixel is in shadow.
    psr = np.clip((0.05 - illum) / 0.05, 0.0, 1.0)

    # 2) Cold trap: 1 well below 110 K, ramping to 0 at the threshold.
    cold_trap = np.clip((110.0 - temp) / 110.0, 0.0, 1.0)

    # 3) Frost albedo (normalised across the scene).
    frost_albedo = _normalize01(albedo)

    # 4) LAMP frost signature (normalised across the scene).
    frost_lamp = _normalize01(lamp)

    # 5) Radar criterion (hard rule -> high/low probability with a small margin
    #    so it never single-handedly vetoes the fusion).
    radar = np.where((cpr > 1.0) & (dop < 0.13), 0.95, 0.05)

    layers = [psr, cold_trap, frost_albedo, frost_lamp, radar]
    # Radar is the most decisive single sensor; weight it above the proxies.
    weights = [1.0, 1.0, 1.0, 1.0, 2.0]
    return bayesian_fusion(layers, weights=weights)
