"""2-D map visualisations (matplotlib).

Implemented by: **viz agent**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["plot_layer", "plot_ice_map", "mchi_rgb"]


def plot_layer(
    layer: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    out: str | Path | None = None,
    **kwargs,
) -> Any:
    """Render a single raster layer as a georeferenced image.

    Parameters
    ----------
    layer : np.ndarray
        Raster to display, shape ``(H, W)``.
    title : str, optional
        Figure title.
    cmap : str, default "viridis"
        Matplotlib colormap.
    out : str or Path, optional
        If given, save the figure to this path.
    **kwargs
        Forwarded to ``imshow`` (e.g. ``vmin``, ``vmax``, ``extent``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError("viz agent")


def plot_ice_map(
    ice_mask: np.ndarray,
    background: np.ndarray | None = None,
    out: str | Path | None = None,
    **kwargs,
) -> Any:
    """Overlay a detected-ice mask on an optional background raster.

    Parameters
    ----------
    ice_mask : np.ndarray
        Boolean / probability ice mask, shape ``(H, W)``.
    background : np.ndarray, optional
        Grayscale backdrop (e.g. DEM hillshade or radar power).
    out : str or Path, optional
        Save path.
    **kwargs
        Styling overrides.

    Returns
    -------
    matplotlib.figure.Figure
    """
    raise NotImplementedError("viz agent")


def mchi_rgb(
    even: np.ndarray, volume: np.ndarray, odd: np.ndarray
) -> np.ndarray:
    """Compose an m-chi RGB image (R=even, G=volume, B=odd).

    Stretches each decomposition channel to ``[0, 1]`` and stacks them so ice
    volume-scatter appears green.

    Parameters
    ----------
    even, volume, odd : np.ndarray
        m-chi power channels, each shape ``(H, W)``.

    Returns
    -------
    np.ndarray
        RGB image, shape ``(H, W, 3)`` in ``[0, 1]``.
    """
    raise NotImplementedError("viz agent")
