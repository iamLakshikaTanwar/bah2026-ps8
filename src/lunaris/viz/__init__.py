"""Visualisation: 2-D layer/ice maps, CPR-DOP & volume charts, interactive
3-D terrain HTML (plotly, no VTK), and the final self-contained HTML report.

All rendering is offline-safe: the matplotlib *Agg* backend is selected inside
:mod:`lunaris.viz.maps` / :mod:`lunaris.viz.report` and every plotly artefact
inlines plotly.js, so figures and the report open with no internet, no GPU and
no VTK.
"""

from __future__ import annotations

from .charts import cpr_dop_scatter, feature_importance_bar, volume_histogram
from .maps import (
    hillshade,
    mchi_rgb,
    plot_ice_map,
    plot_layer,
    plot_mchi,
    plot_traverse,
)
from .report import build_report, demo_report
from .terrain3d import terrain3d_html

__all__ = [
    "plot_layer",
    "plot_ice_map",
    "mchi_rgb",
    "plot_mchi",
    "hillshade",
    "plot_traverse",
    "cpr_dop_scatter",
    "volume_histogram",
    "feature_importance_bar",
    "terrain3d_html",
    "build_report",
    "demo_report",
]
