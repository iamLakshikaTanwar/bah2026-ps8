"""Tests for the ``lunaris.viz`` module.

Exercises the static maps (matplotlib), interactive charts + 3-D terrain
(plotly) and the final self-contained HTML report. Every artefact is checked
for offline self-containment (no external http(s) plotly asset) and a sane file
size. All scenes use the small deterministic synthetic Faustini grid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # head-less, mirror module behaviour

import numpy as np
import pytest
from matplotlib.figure import Figure

from lunaris.io.synthetic import generate_faustini_scene
from lunaris.viz import charts, maps, report, terrain3d


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scene():
    """Small deterministic synthetic scene shared by the viz tests."""
    return generate_faustini_scene(n=128, seed=42)


@pytest.fixture(scope="module")
def arrays(scene):
    """Convenience bundle of the arrays used across the tests."""
    dem = np.asarray(scene.dem)
    cpr = np.asarray(scene.cpr_L)
    dop = np.asarray(scene.dop_L)
    s0 = np.asarray(scene.s0_L)
    s3 = np.asarray(scene.s3_L)
    ice = (cpr > 1.0) & (dop < 0.13)
    # inline m-chi-ish channels (no other subpackage imported)
    vol = np.sqrt(np.clip(s0 * (1.0 - dop), 0.0, None))
    even = np.sqrt(np.clip(s0 * np.clip(dop, 0, 1), 0.0, None)) * (s3 > 0)
    odd = np.sqrt(np.clip(s0 * np.clip(dop, 0, 1), 0.0, None)) * (s3 <= 0)
    return {
        "dem": dem, "cpr": cpr, "dop": dop, "s0": s0, "s3": s3,
        "ice": ice, "even": even, "vol": vol, "odd": odd,
    }


def _size(p: Path) -> int:
    assert p.exists(), f"{p} was not written"
    return p.stat().st_size


# ---------------------------------------------------------------------------
# maps.py (matplotlib)
# ---------------------------------------------------------------------------
def test_plot_layer_returns_figure_and_writes_png(arrays, tmp_path):
    out = tmp_path / "layer.png"
    fig = maps.plot_layer(arrays["dem"], title="DEM", cmap="cividis", out=out)
    assert isinstance(fig, Figure)
    assert _size(out) > 1024


def test_plot_ice_map_returns_figure_and_writes_png(arrays, tmp_path):
    out = tmp_path / "ice.png"
    bg = maps.hillshade(arrays["dem"], res=20.0)
    fig = maps.plot_ice_map(arrays["ice"], background=bg, out=out)
    assert isinstance(fig, Figure)
    assert _size(out) > 1024


def test_plot_ice_map_without_background(arrays):
    fig = maps.plot_ice_map(arrays["ice"], background=None)
    assert isinstance(fig, Figure)


def test_mchi_rgb_shape_and_range(arrays):
    rgb = maps.mchi_rgb(arrays["even"], arrays["vol"], arrays["odd"])
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert rgb.shape[:2] == arrays["dem"].shape
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert np.isfinite(rgb).all()


def test_plot_mchi_returns_figure_and_writes_png(arrays, tmp_path):
    out = tmp_path / "mchi.png"
    fig = maps.plot_mchi(arrays["even"], arrays["vol"], arrays["odd"], out=out)
    assert isinstance(fig, Figure)
    assert _size(out) > 1024


def test_hillshade_range(arrays):
    hs = maps.hillshade(arrays["dem"], res=20.0)
    assert hs.shape == arrays["dem"].shape
    assert hs.min() >= 0.0 and hs.max() <= 1.0


def test_plot_traverse_returns_figure_and_writes_png(arrays, tmp_path):
    out = tmp_path / "traverse.png"
    path = [(10, 10), (40, 50), (80, 90)]
    fig = maps.plot_traverse(arrays["dem"], path=path, landing=(10, 10),
                             target=(80, 90), out=out)
    assert isinstance(fig, Figure)
    assert _size(out) > 1024


# ---------------------------------------------------------------------------
# charts.py (plotly -> self-contained HTML)
# ---------------------------------------------------------------------------
def test_cpr_dop_scatter_writes_html(arrays, tmp_path):
    import plotly.graph_objects as go

    out = tmp_path / "scatter.html"
    fig = charts.cpr_dop_scatter(arrays["cpr"], arrays["dop"],
                                 ice_mask=arrays["ice"], out=out, sample=5000)
    assert isinstance(fig, go.Figure)
    assert _size(out) > 1024


def test_volume_histogram_writes_html(tmp_path):
    import plotly.graph_objects as go

    rng = np.random.default_rng(0)
    samples = rng.normal(1e6, 1.5e5, size=3000)
    out = tmp_path / "hist.html"
    fig = charts.volume_histogram(samples, ci=(7.5e5, 1.25e6), out=out)
    assert isinstance(fig, go.Figure)
    assert _size(out) > 1024


def test_feature_importance_bar_writes_html(tmp_path):
    import plotly.graph_objects as go

    out = tmp_path / "fi.html"
    fig = charts.feature_importance_bar(
        ["CPR", "DOP", "temp", "illum"], [0.4, 0.3, 0.2, 0.1], out=out)
    assert isinstance(fig, go.Figure)
    assert _size(out) > 1024


# ---------------------------------------------------------------------------
# terrain3d.py (plotly, self-contained)
# ---------------------------------------------------------------------------
def test_terrain3d_html_self_contained(arrays, tmp_path):
    out = tmp_path / "terrain3d.html"
    path = [(10, 10), (40, 50), (80, 90)]
    p = terrain3d.terrain3d_html(
        arrays["dem"], overlay=arrays["vol"], path=path,
        landing=(10, 10), target=(80, 90), out=out, res=20.0)
    assert isinstance(p, Path)
    text = out.read_text(encoding="utf-8")
    assert "Plotly" in text  # plotly.js present
    # >100KB confirms plotly.js is embedded (self-contained / offline)
    assert _size(out) > 100 * 1024


# ---------------------------------------------------------------------------
# report.py — the deliverable
# ---------------------------------------------------------------------------
def test_demo_report_self_contained(tmp_path):
    out = tmp_path / "outputs" / "lunaris_report.html"
    p = report.demo_report(out_html=out, n=128, seed=42)
    assert isinstance(p, Path) and p.exists()
    text = out.read_text(encoding="utf-8")

    # > 50KB standalone report
    assert _size(out) > 50 * 1024

    # branding + every section title present (titles are HTML-escaped in the
    # document, so compare against the escaped form, e.g. "&" -> "&amp;").
    import html as _html

    assert "LUNARIS" in text
    for _, title in report._SECTIONS:
        assert _html.escape(title) in text, f"missing section: {title}"

    # offline self-containment: plotly.js is inlined at least once (a large
    # inline <script> block, NOT an external <script src="...cdn..."> tag).
    # NB: the *bundled* plotly.js itself contains the literal string
    # "cdn.plot.ly" as a default topojson config value — that is harmless and
    # is never fetched for our charts — so we must assert on the *script tag*,
    # not on any occurrence of the substring.
    import re

    assert "Plotly.newPlot" in text or "Plotly" in text
    assert re.search(r'<script[^>]+src=["\'][^"\']*cdn\.plot', text) is None, \
        "report references an external plotly CDN script"
    assert re.search(r'<script[^>]+src=["\'][^"\']*plotly', text) is None, \
        "report references an external plotly script instead of inlining it"
    # plotly.js must actually be embedded inline -> document is large.
    assert "Plotly" in text and _size(out) > 50 * 1024
    # the report should ship its own terrain3d.html sidecar too
    assert (out.parent / "terrain3d.html").exists()


def test_build_report_tolerates_missing_keys(tmp_path):
    # only a couple of keys present -> should still render without exceptions
    out = tmp_path / "min.html"
    p = report.build_report({"meta": {"target": "test"}}, out_html=out)
    assert p.exists()
    text = out.read_text(encoding="utf-8")
    assert "LUNARIS" in text
    assert _size(out) > 1024


def test_build_report_with_arrays(arrays, tmp_path):
    out = tmp_path / "full.html"
    results = {
        "dem": arrays["dem"],
        "ice_mask": arrays["ice"],
        "cpr": arrays["cpr"],
        "dop": arrays["dop"],
        "mchi": (arrays["even"], arrays["vol"], arrays["odd"]),
        "metrics": {"precision": 0.9, "recall": 0.8, "F1": 0.85},
        "feature_importance": (["CPR", "DOP"], [0.6, 0.4]),
        "landing": (10, 10),
        "target": (80, 90),
        "path": [(10, 10), (45, 50), (80, 90)],
        "volume_samples": np.random.default_rng(0).normal(1e6, 1e5, 2000),
        "volume_ci": (8e5, 1.2e6),
        "volume_mean": 1e6,
    }
    p = report.build_report(results, out_html=out)
    assert p.exists()
    assert _size(out) > 50 * 1024
