"""Statistical charts (matplotlib).

Implemented by: **viz agent**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["cpr_dop_scatter", "volume_histogram"]


def cpr_dop_scatter(
    cpr: np.ndarray,
    dop: np.ndarray,
    ice_mask: np.ndarray | None = None,
    out: str | Path | None = None,
    **kwargs,
) -> Any:
    """CPR-vs-DOP scatter with the ``CPR>1 & DOP<0.13`` decision box drawn.

    Visualises how the ice criterion separates the ice cluster (low DOP, high
    CPR) from rough-surface decoys (high CPR but high DOP).

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP (flattened or 2-D), same shape.
    ice_mask : np.ndarray, optional
        Boolean ground-truth / predicted mask to colour the points.
    out : str or Path, optional
        Save path.
    **kwargs
        Styling overrides.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError("viz agent")


def volume_histogram(
    samples: np.ndarray,
    ci: tuple[float, float] | None = None,
    out: str | Path | None = None,
    **kwargs,
) -> Any:
    """Histogram of Monte-Carlo ice-volume samples with a CI band.

    Parameters
    ----------
    samples : np.ndarray
        Monte-Carlo volume samples [m^3], shape ``(n,)``.
    ci : tuple[float, float], optional
        Confidence-interval bounds to annotate.
    out : str or Path, optional
        Save path.
    **kwargs
        Styling overrides.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError("viz agent")
