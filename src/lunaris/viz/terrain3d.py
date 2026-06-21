"""Interactive 3-D terrain visualisation (plotly only — no pyvista/VTK).

Implemented by: **viz agent**.

Renders the DEM as a plotly ``Surface`` with an optional draped overlay (e.g.
ice probability) and an optional traverse path polyline, writing a single
self-contained HTML file (plotly.js inlined) that opens in any browser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["terrain3d_html"]


def terrain3d_html(
    dem: np.ndarray,
    overlay: np.ndarray | None = None,
    path: list[tuple[int, int]] | None = None,
    out: str | Path = "terrain3d.html",
) -> Path:
    """Write a self-contained interactive 3-D terrain HTML (plotly Surface).

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)`` — the ``z`` surface.
    overlay : np.ndarray, optional
        Per-vertex scalar (e.g. ice probability) coloured onto the surface,
        shape ``(H, W)``. If ``None``, elevation colours the surface.
    path : list[tuple[int, int]], optional
        Rover-traverse ``(row, col)`` waypoints drawn as a 3-D polyline.
    out : str or Path, default "terrain3d.html"
        Output HTML path (plotly.js inlined; no external assets / VTK).

    Returns
    -------
    Path
        The written HTML file path.
    """
    raise NotImplementedError("viz agent")
