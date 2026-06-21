"""Visualisation: 2-D layer/ice maps, CPR-DOP & volume charts, interactive
3-D terrain HTML (plotly, no VTK), and the final HTML report.

(Function bodies are filled in by the viz agent.)
"""

from __future__ import annotations

from .charts import cpr_dop_scatter, volume_histogram
from .maps import mchi_rgb, plot_ice_map, plot_layer
from .report import build_report
from .terrain3d import terrain3d_html

__all__ = [
    "plot_layer",
    "plot_ice_map",
    "mchi_rgb",
    "cpr_dop_scatter",
    "volume_histogram",
    "terrain3d_html",
    "build_report",
]
