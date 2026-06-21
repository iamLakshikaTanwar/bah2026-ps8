"""2-D map visualisations (matplotlib).

Implemented by: **viz agent**.

Publication-quality static maps for the LUNARIS deliverables: single-layer
imshow panels, the subsurface-ice overlay, the m-chi polarimetric RGB
composite, a hillshade helper for terrain backdrops, and the rover-traverse
overlay.

All rendering is head-less: the matplotlib *Agg* backend is selected at import
time so the module is safe to import on a server / CI with no display, no GPU
and no VTK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # head-less, offline rendering — MUST precede pyplot

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "plot_layer",
    "plot_ice_map",
    "mchi_rgb",
    "plot_mchi",
    "hillshade",
    "plot_traverse",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _percentile_stretch(
    arr: np.ndarray, lo: float = 2.0, hi: float = 98.0
) -> np.ndarray:
    """Contrast-stretch ``arr`` to ``[0, 1]`` using percentile clipping.

    Robust to outliers/speckle: values below the ``lo`` percentile map to 0 and
    values above the ``hi`` percentile map to 1. NaNs are treated as the low
    bound so they render dark rather than blowing out the stretch.

    Parameters
    ----------
    arr : np.ndarray
        Input raster.
    lo, hi : float
        Lower / upper percentiles (0-100) defining the stretch endpoints.

    Returns
    -------
    np.ndarray
        Float array in ``[0, 1]`` with the same shape as ``arr``.
    """
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    vlo, vhi = np.percentile(finite, [lo, hi])
    if vhi <= vlo:
        vhi = vlo + 1e-9
    out = (a - vlo) / (vhi - vlo)
    out = np.clip(out, 0.0, 1.0)
    out[~np.isfinite(a)] = 0.0
    return out


def _save(fig: Figure, out: str | Path | None, dpi: int = 150) -> None:
    """Persist ``fig`` to ``out`` (PNG) if a path is given, else no-op."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


def hillshade(
    dem: np.ndarray,
    res: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Compute a normalised hillshade of a DEM for use as a terrain backdrop.

    Uses matplotlib's :class:`~matplotlib.colors.LightSource` with the standard
    cartographic convention (light from the NW, 45 deg above the horizon). The
    vertical exaggeration is derived from the pixel resolution so the shading
    looks consistent across grids of different sample distances.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel ground sample distance [m]; used to scale the surface slopes.
    azimuth : float, default 315.0
        Illumination azimuth in degrees (0 = North, clockwise).
    altitude : float, default 45.0
        Illumination altitude angle in degrees above the horizon.

    Returns
    -------
    np.ndarray
        Hillshade intensity in ``[0, 1]``, shape ``(H, W)``.
    """
    dem = np.asarray(dem, dtype=np.float64)
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    res = float(res) if res else 1.0
    # dx/dy are the pixel spacings; vert_exag emphasises subtle lunar relief.
    shade = ls.hillshade(dem, vert_exag=2.0, dx=res, dy=res)
    return np.clip(shade, 0.0, 1.0)


# ---------------------------------------------------------------------------
# single-layer map
# ---------------------------------------------------------------------------
def plot_layer(
    layer: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    out: str | Path | None = None,
    **kwargs: Any,
) -> Figure:
    """Render a single raster layer as an annotated image with a colorbar.

    Parameters
    ----------
    layer : np.ndarray
        Raster to display, shape ``(H, W)``.
    title : str, optional
        Figure title.
    cmap : str, default "viridis"
        Matplotlib colormap (e.g. ``"cividis"``/``"magma"`` for temperature,
        ``"Blues_r"`` for ice/frost proxies).
    out : str or Path, optional
        If given, save the figure to this path as a PNG.
    **kwargs
        Forwarded to ``imshow`` (e.g. ``vmin``, ``vmax``, ``extent``). A
        ``cbar_label`` keyword (popped here) labels the colorbar.

    Returns
    -------
    matplotlib.figure.Figure
    """
    layer = np.asarray(layer)
    cbar_label = kwargs.pop("cbar_label", "")
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    im = ax.imshow(layer, cmap=cmap, origin="upper", **kwargs)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    fig.tight_layout()
    _save(fig, out)
    return fig


# ---------------------------------------------------------------------------
# ice overlay
# ---------------------------------------------------------------------------
def plot_ice_map(
    ice_mask: np.ndarray,
    background: np.ndarray | None = None,
    out: str | Path | None = None,
    title: str = "Subsurface ice (CPR>1 & DOP<0.13)",
    **kwargs: Any,
) -> Figure:
    """Overlay a detected-ice mask on an optional grayscale background.

    The background (typically a DEM hillshade or radar ``s0`` power) is drawn in
    grayscale; ice pixels are painted in semi-transparent **cyan** on top so the
    detected region stands out against the terrain.

    Parameters
    ----------
    ice_mask : np.ndarray
        Boolean / probability ice mask, shape ``(H, W)``. Non-zero == ice.
    background : np.ndarray, optional
        Grayscale backdrop (e.g. DEM hillshade or radar power). If ``None`` a
        flat dark backdrop is used.
    out : str or Path, optional
        Save path (PNG).
    title : str, optional
        Figure title.
    **kwargs
        Styling overrides (``alpha`` for the overlay opacity, ``bg_cmap`` for
        the background colormap).

    Returns
    -------
    matplotlib.figure.Figure
    """
    ice_mask = np.asarray(ice_mask)
    alpha = float(kwargs.pop("alpha", 0.55))
    bg_cmap = kwargs.pop("bg_cmap", "gray")

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    if background is not None:
        bg = _percentile_stretch(background)
        ax.imshow(bg, cmap=bg_cmap, origin="upper")
    else:
        ax.imshow(np.zeros(ice_mask.shape), cmap="gray", vmin=0, vmax=1,
                  origin="upper")

    # cyan, transparent where there is no ice.
    cyan = ListedColormap([(0.0, 1.0, 1.0, 1.0)])
    overlay = np.ma.masked_where(np.asarray(ice_mask) <= 0, ice_mask)
    ax.imshow(overlay, cmap=cyan, alpha=alpha, origin="upper",
              interpolation="nearest")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")

    # legend proxy for the ice colour.
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor="cyan", edgecolor="k", label="detected ice")],
        loc="upper right",
        framealpha=0.85,
    )
    fig.tight_layout()
    _save(fig, out)
    return fig


# ---------------------------------------------------------------------------
# m-chi RGB composite
# ---------------------------------------------------------------------------
def mchi_rgb(
    even: np.ndarray, volume: np.ndarray, odd: np.ndarray
) -> np.ndarray:
    """Compose an m-chi RGB image (R=even, G=volume, B=odd).

    Each decomposition channel is independently percentile-stretched to
    ``[0, 1]`` and stacked, so double-bounce/even-bounce appears red, ice
    volume-scatter appears green, and single/odd-bounce surface appears blue.

    Parameters
    ----------
    even, volume, odd : np.ndarray
        m-chi power channels (double-bounce, volume, surface), each ``(H, W)``.

    Returns
    -------
    np.ndarray
        RGB image, shape ``(H, W, 3)`` with all values in ``[0, 1]``.
    """
    r = _percentile_stretch(even)
    g = _percentile_stretch(volume)
    b = _percentile_stretch(odd)
    rgb = np.dstack([r, g, b]).astype(np.float64)
    return np.clip(rgb, 0.0, 1.0)


def plot_mchi(
    even: np.ndarray,
    volume: np.ndarray,
    odd: np.ndarray,
    out: str | Path | None = None,
    title: str = "m-chi decomposition (R: double-bounce, G: volume, B: surface)",
) -> Figure:
    """Render the m-chi RGB composite with a channel legend.

    Parameters
    ----------
    even, volume, odd : np.ndarray
        m-chi power channels, each ``(H, W)``.
    out : str or Path, optional
        Save path (PNG).
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rgb = mchi_rgb(even, volume, odd)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    ax.imshow(rgb, origin="upper")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="red", label="double-bounce"),
            Patch(facecolor="lime", label="volume (ice)"),
            Patch(facecolor="blue", label="surface"),
        ],
        loc="upper right",
        framealpha=0.85,
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, out)
    return fig


# ---------------------------------------------------------------------------
# traverse overlay
# ---------------------------------------------------------------------------
def plot_traverse(
    dem: np.ndarray,
    path: Sequence[tuple[int, int]] | np.ndarray | None,
    landing: tuple[int, int] | None = None,
    target: tuple[int, int] | None = None,
    out: str | Path | None = None,
    res: float = 20.0,
    title: str = "Rover traverse (hillshade)",
) -> Figure:
    """Plot a rover traverse as a polyline over a DEM hillshade.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    path : sequence of (row, col) or ndarray, optional
        Ordered traverse waypoints (pixel coordinates). Drawn as a polyline.
    landing : (row, col), optional
        Landing-site marker (green star).
    target : (row, col), optional
        Ice-target marker (cyan star).
    out : str or Path, optional
        Save path (PNG).
    res : float, default 20.0
        Pixel resolution [m], passed to the hillshade helper.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dem = np.asarray(dem, dtype=np.float64)
    shade = hillshade(dem, res=res)

    fig, ax = plt.subplots(figsize=(7.0, 6.0), facecolor="white")
    ax.imshow(shade, cmap="gray", origin="upper", vmin=0, vmax=1)
    # faint colour wash of elevation to give context without hiding the shade.
    ax.imshow(dem, cmap="terrain", origin="upper", alpha=0.30)

    if path is not None and len(path) > 0:
        pts = np.asarray(path, dtype=float)
        rows, cols = pts[:, 0], pts[:, 1]
        ax.plot(cols, rows, "-", color="#ff6f00", linewidth=2.5,
                label="traverse", zorder=4)
        ax.plot(cols, rows, ".", color="#ffd180", markersize=4, zorder=5)

    if landing is not None:
        ax.plot(landing[1], landing[0], "*", color="#00e676", markersize=20,
                markeredgecolor="k", label="landing", zorder=6)
    if target is not None:
        ax.plot(target[1], target[0], "*", color="#00e5ff", markersize=20,
                markeredgecolor="k", label="ice target", zorder=6)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("column (px)")
    ax.set_ylabel("row (px)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    _save(fig, out)
    return fig
