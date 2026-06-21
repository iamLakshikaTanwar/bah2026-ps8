"""Interactive 3-D terrain visualisation (plotly only — no pyvista/VTK).

Implemented by: **viz agent**.

Renders the DEM as a plotly ``Surface`` with an optional draped overlay (e.g.
ice probability or temperature) and an optional rover-traverse polyline,
writing a single **self-contained** HTML file (plotly.js inlined) that opens in
any browser offline — no internet, no GPU, no VTK. This is the showcase 3-D
view of the LUNARIS report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

__all__ = ["terrain3d_html"]


def _downsample(arr: np.ndarray, max_dim: int) -> tuple[np.ndarray, int]:
    """Stride-decimate a 2-D array so its largest dimension <= ``max_dim``.

    Returns the decimated array and the stride used (so coordinates can be
    rescaled consistently).
    """
    h, w = arr.shape
    step = max(1, int(np.ceil(max(h, w) / float(max_dim))))
    return arr[::step, ::step], step


def terrain3d_html(
    dem: np.ndarray,
    overlay: np.ndarray | None = None,
    path: Sequence[tuple[int, int]] | np.ndarray | None = None,
    landing: tuple[int, int] | None = None,
    target: tuple[int, int] | None = None,
    out: str | Path = "terrain3d.html",
    res: float = 20.0,
    max_dim: int = 220,
) -> Path:
    """Write a self-contained interactive 3-D terrain HTML (plotly Surface).

    The DEM is drawn as a 3-D surface with realistic horizontal scaling derived
    from ``res`` (so x/y are in metres). If ``overlay`` is supplied it colours
    the surface via ``surfacecolor`` (e.g. ice likelihood or temperature);
    otherwise elevation colours the surface. A rover ``path`` is drawn as a 3-D
    line riding on the surface, and landing/target markers are placed at their
    sampled elevations.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)`` — the ``z`` surface.
    overlay : np.ndarray, optional
        Per-vertex scalar (e.g. ice probability / temperature) coloured onto the
        surface, shape ``(H, W)``. If ``None``, elevation colours the surface.
    path : sequence of (row, col) or ndarray, optional
        Rover-traverse waypoints drawn as a 3-D polyline.
    landing : (row, col), optional
        Landing-site marker.
    target : (row, col), optional
        Ice-target marker.
    out : str or Path, default "terrain3d.html"
        Output HTML path (plotly.js inlined; no external assets / VTK).
    res : float, default 20.0
        Pixel ground sample distance [m]; sets the surface x/y scale.
    max_dim : int, default 220
        Cap on the surface grid edge after decimation (keeps the HTML light).

    Returns
    -------
    Path
        The written HTML file path.
    """
    dem = np.asarray(dem, dtype=np.float64)
    res = float(res) if res else 1.0

    dem_ds, step = _downsample(dem, max_dim)
    hh, ww = dem_ds.shape
    # metric axes (row -> y, col -> x); full-resolution pixel -> metre factor.
    xs = np.arange(ww) * step * res
    ys = np.arange(hh) * step * res

    surf_kwargs = dict(
        x=xs, y=ys, z=dem_ds,
        colorbar=dict(title="elev [m]"),
        colorscale="Cividis",
        showscale=True,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.15, roughness=0.9),
        contours=dict(z=dict(show=True, usecolormap=True, width=1,
                             project=dict(z=False))),
    )
    if overlay is not None:
        ov = np.asarray(overlay, dtype=np.float64)
        ov_ds, _ = _downsample(ov, max_dim)
        # guard against an overlay of a different native size than the dem
        if ov_ds.shape != dem_ds.shape:
            from numpy import interp  # rare path; nearest-resize via indices

            ri = np.linspace(0, ov.shape[0] - 1, hh).astype(int)
            ci = np.linspace(0, ov.shape[1] - 1, ww).astype(int)
            ov_ds = ov[np.ix_(ri, ci)]
        surf_kwargs["surfacecolor"] = ov_ds
        surf_kwargs["colorscale"] = "Viridis"
        surf_kwargs["colorbar"] = dict(title="overlay")

    fig = go.Figure(data=[go.Surface(**surf_kwargs)])

    def _z_at(rc: tuple[int, int]) -> float:
        r = int(np.clip(rc[0], 0, dem.shape[0] - 1))
        c = int(np.clip(rc[1], 0, dem.shape[1] - 1))
        return float(dem[r, c])

    def _xy(rc: tuple[int, int]) -> tuple[float, float]:
        return float(rc[1]) * res, float(rc[0]) * res

    if path is not None and len(path) > 0:
        pts = np.asarray(path, dtype=float)
        px = pts[:, 1] * res
        py = pts[:, 0] * res
        pz = np.array([_z_at((r, c)) for r, c in pts]) + 8.0  # lift above surf
        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz, mode="lines+markers", name="traverse",
            line=dict(color="#ff6f00", width=6),
            marker=dict(size=2, color="#ffd180"),
        ))

    if landing is not None:
        lx, ly = _xy(landing)
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[_z_at(landing) + 15.0], mode="markers",
            name="landing",
            marker=dict(size=7, color="#00e676", symbol="diamond"),
        ))
    if target is not None:
        tx, ty = _xy(target)
        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[_z_at(target) + 15.0], mode="markers",
            name="ice target",
            marker=dict(size=7, color="#00e5ff", symbol="diamond"),
        ))

    fig.update_layout(
        title=dict(text="LUNARIS — 3-D terrain & rover traverse", x=0.5),
        paper_bgcolor="#0b1021",
        font=dict(color="#e6ecff", family="Inter, Segoe UI, sans-serif"),
        scene=dict(
            xaxis=dict(title="x [m]", backgroundcolor="#0b1021",
                       gridcolor="#2a3354"),
            yaxis=dict(title="y [m]", backgroundcolor="#0b1021",
                       gridcolor="#2a3354"),
            zaxis=dict(title="elevation [m]", backgroundcolor="#0b1021",
                       gridcolor="#2a3354"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.45),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs=True -> plotly.js embedded -> opens fully offline.
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    return out
