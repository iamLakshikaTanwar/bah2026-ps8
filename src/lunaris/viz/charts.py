"""Statistical charts (plotly, interactive & self-contained).

Implemented by: **viz agent**.

Interactive analytics for the LUNARIS deliverables. Every figure is a
``plotly.graph_objects.Figure``; when an ``out`` path is given the figure is
written as a **self-contained** HTML file (``include_plotlyjs=True``) so it
opens in any browser with no internet connection and no external assets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go

__all__ = ["cpr_dop_scatter", "volume_histogram", "feature_importance_bar"]


# Dark "space" theme shared across the interactive charts.
_BG = "#0b1021"
_PANEL = "#141a31"
_FONT = "#e6ecff"
_GRID = "#2a3354"


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the LUNARIS dark space theme to a plotly figure in place."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(color=_FONT, family="Inter, Segoe UI, sans-serif"),
        title=dict(text=title, x=0.5, xanchor="center"),
        margin=dict(l=70, r=40, t=70, b=60),
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID)
    return fig


def _write(fig: go.Figure, out: str | Path | None) -> None:
    """Write a self-contained HTML (plotly.js inlined) if ``out`` is given."""
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)


# ---------------------------------------------------------------------------
# CPR vs DOP scatter
# ---------------------------------------------------------------------------
def cpr_dop_scatter(
    cpr: np.ndarray,
    dop: np.ndarray,
    ice_mask: np.ndarray | None = None,
    out: str | Path | None = None,
    sample: int = 20000,
    cpr_thresh: float = 1.0,
    dop_thresh: float = 0.13,
    **kwargs: Any,
) -> go.Figure:
    """CPR-vs-DOP scatter with the ``CPR>1 & DOP<0.13`` decision box drawn.

    Visualises how the dual CPR/DOP criterion separates the ice cluster (high
    CPR, low DOP) from rough-surface decoys (high CPR but high DOP). The points
    are subsampled for rendering speed and coloured by ``ice_mask`` when given.

    Parameters
    ----------
    cpr, dop : np.ndarray
        Per-pixel CPR and DOP (flattened or 2-D); must share a shape.
    ice_mask : np.ndarray, optional
        Boolean ground-truth / predicted mask used to colour the points.
    out : str or Path, optional
        Self-contained HTML save path.
    sample : int, default 20000
        Maximum number of points to plot (random subsample for speed).
    cpr_thresh : float, default 1.0
        CPR decision threshold (vertical line).
    dop_thresh : float, default 0.13
        DOP decision threshold (horizontal line).
    **kwargs
        Ignored / reserved for styling.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cpr = np.asarray(cpr, dtype=np.float64).ravel()
    dop = np.asarray(dop, dtype=np.float64).ravel()
    n = cpr.size

    finite = np.isfinite(cpr) & np.isfinite(dop)
    idx = np.flatnonzero(finite)
    if idx.size > sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=sample, replace=False)

    cs, ds = cpr[idx], dop[idx]
    fig = go.Figure()

    if ice_mask is not None:
        mask = np.asarray(ice_mask).astype(bool).ravel()[idx]
        fig.add_trace(go.Scattergl(
            x=cs[~mask], y=ds[~mask], mode="markers", name="other",
            marker=dict(size=3, color="#5a6488", opacity=0.45),
        ))
        fig.add_trace(go.Scattergl(
            x=cs[mask], y=ds[mask], mode="markers", name="ice",
            marker=dict(size=4, color="#00e5ff", opacity=0.85),
        ))
    else:
        fig.add_trace(go.Scattergl(
            x=cs, y=ds, mode="markers", name="pixels",
            marker=dict(size=3, color="#7aa2ff", opacity=0.5),
        ))

    # threshold lines
    fig.add_vline(x=cpr_thresh, line=dict(color="#ff5252", dash="dash"))
    fig.add_hline(y=dop_thresh, line=dict(color="#ffd54f", dash="dash"))

    # shade + annotate the ice quadrant (CPR > thresh, DOP < thresh)
    xmax = float(np.nanmax(cs)) if cs.size else cpr_thresh + 1.0
    fig.add_shape(
        type="rect", x0=cpr_thresh, x1=max(xmax, cpr_thresh + 0.1),
        y0=0.0, y1=dop_thresh,
        fillcolor="rgba(0,229,255,0.10)", line=dict(width=0), layer="below",
    )
    fig.add_annotation(
        x=cpr_thresh + 0.02 * max(xmax - cpr_thresh, 0.1), y=dop_thresh * 0.5,
        text="ICE quadrant<br>CPR>1 & DOP<0.13",
        showarrow=False, xanchor="left",
        font=dict(color="#00e5ff", size=12),
    )

    _apply_theme(fig, title=f"CPR vs DOP  (n={n:,}; shown={idx.size:,})")
    fig.update_xaxes(title="Circular Polarisation Ratio (CPR)")
    fig.update_yaxes(title="Degree of Polarisation (DOP)")
    _write(fig, out)
    return fig


# ---------------------------------------------------------------------------
# Monte-Carlo volume histogram
# ---------------------------------------------------------------------------
def volume_histogram(
    samples: np.ndarray,
    ci: tuple[float, float] | None = None,
    out: str | Path | None = None,
    units: str = "m³",
    **kwargs: Any,
) -> go.Figure:
    """Histogram of Monte-Carlo ice-volume samples with CI + mean lines.

    Parameters
    ----------
    samples : np.ndarray
        Monte-Carlo volume samples, shape ``(n,)``.
    ci : tuple[float, float], optional
        Confidence-interval bounds to annotate (vertical lines + shaded band).
    out : str or Path, optional
        Self-contained HTML save path.
    units : str, default "m³"
        Unit label for the volume axis.
    **kwargs
        Ignored / reserved for styling.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    samples = np.asarray(samples, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    mean = float(np.mean(samples)) if samples.size else 0.0

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=60, name="MC samples",
        marker=dict(color="#5a7bff", line=dict(color="#0b1021", width=0.4)),
        opacity=0.85,
    ))

    if ci is not None:
        lo, hi = float(ci[0]), float(ci[1])
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(0,229,255,0.10)",
                      line=dict(width=0), layer="below")
        for x, lbl, col in ((lo, "CI low", "#00e5ff"), (hi, "CI high", "#00e5ff")):
            fig.add_vline(x=x, line=dict(color=col, dash="dot"),
                          annotation_text=lbl, annotation_position="top")
    fig.add_vline(x=mean, line=dict(color="#ff8a65", dash="dash"),
                  annotation_text=f"mean = {mean:,.3g}",
                  annotation_position="top right")

    _apply_theme(fig, title="Monte-Carlo ice-volume estimate")
    fig.update_xaxes(title=f"Ice volume [{units}]")
    fig.update_yaxes(title="count")
    _write(fig, out)
    return fig


# ---------------------------------------------------------------------------
# feature-importance bar
# ---------------------------------------------------------------------------
def feature_importance_bar(
    names: Sequence[str],
    importances: Sequence[float],
    out: str | Path | None = None,
    title: str = "Feature importance",
    **kwargs: Any,
) -> go.Figure:
    """Horizontal bar chart of feature importances, sorted ascending.

    Parameters
    ----------
    names : sequence of str
        Feature names.
    importances : sequence of float
        Importance score for each feature (same length as ``names``).
    out : str or Path, optional
        Self-contained HTML save path.
    title : str, optional
        Figure title.
    **kwargs
        Ignored / reserved for styling.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    names = list(names)
    importances = list(map(float, importances))
    order = np.argsort(importances)  # smallest first -> largest on top
    names = [names[i] for i in order]
    importances = [importances[i] for i in order]

    fig = go.Figure(go.Bar(
        x=importances, y=names, orientation="h",
        marker=dict(
            color=importances, colorscale="Cividis",
            line=dict(color="#0b1021", width=0.5),
        ),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    _apply_theme(fig, title=title)
    fig.update_xaxes(title="importance")
    fig.update_yaxes(title="")
    _write(fig, out)
    return fig
