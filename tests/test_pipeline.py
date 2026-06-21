"""Integration smoke tests for the end-to-end :func:`lunaris.pipeline.run_pipeline`.

Runs the full nine-stage pipeline on a small deterministic synthetic Faustini
scene (``n=64``) and asserts the results bundle, detection quality, and the
written artefacts (HTML report + ``results_summary.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lunaris.config import Settings
from lunaris.pipeline import run_pipeline


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the pipeline once at n=64 into a temp dir; shared across tests."""
    out = tmp_path_factory.mktemp("pipeline_out")
    settings = Settings(grid_size=64, seed=42, outputs_dir=out)
    return run_pipeline(settings)


EXPECTED_KEYS = [
    "cpr", "dop", "ice_mask", "precision", "recall", "slope", "cold_trap",
    "landing_site", "traverse_path", "volume_m3", "mass_kg", "report_path",
]


def test_returns_all_expected_keys(pipeline_result: dict) -> None:
    for key in EXPECTED_KEYS:
        assert key in pipeline_result, f"missing result key: {key}"


def test_summary_is_json_serialisable(pipeline_result: dict) -> None:
    summary = pipeline_result["summary"]
    # round-trips through json without raising
    text = json.dumps(summary)
    assert isinstance(json.loads(text), dict)
    # no numpy arrays leaked into the headline summary
    for v in summary.values():
        assert not isinstance(v, np.ndarray)


def test_detection_quality(pipeline_result: dict) -> None:
    assert pipeline_result["precision"] > 0.9
    assert pipeline_result["recall"] > 0.9
    # the O(1) LUT must reproduce the direct threshold rule exactly
    assert np.array_equal(
        pipeline_result["ice_mask"], pipeline_result["ice_mask_threshold"]
    )


def test_terrain_and_volume_layers(pipeline_result: dict) -> None:
    scene = pipeline_result["scene"]
    h, w = scene.shape
    assert pipeline_result["slope"].shape == (h, w)
    assert pipeline_result["cold_trap"].shape == (h, w)
    assert pipeline_result["cold_trap"].dtype == bool
    assert pipeline_result["volume_m3"] > 0.0
    assert pipeline_result["mass_kg"] > 0.0


def test_landing_and_traverse(pipeline_result: dict) -> None:
    scene = pipeline_result["scene"]
    h, w = scene.shape
    lr, lc = pipeline_result["landing_site"]
    assert 0 <= lr < h and 0 <= lc < w
    # a connected traverse path from landing to the ice-access point exists
    path = pipeline_result["traverse_path"]
    assert path is not None and len(path) >= 1
    assert tuple(path[0]) == tuple(pipeline_result["landing_site"])
    assert tuple(path[-1]) == tuple(pipeline_result["ice_access"])
    assert pipeline_result["summary"]["traverse_length_m"] > 0.0


def test_report_html_written(pipeline_result: dict) -> None:
    report = Path(pipeline_result["report_path"])
    assert report.exists()
    assert report.stat().st_size > 50_000  # > 50 KB self-contained report


def test_results_summary_json_written(pipeline_result: dict) -> None:
    summary_path = Path(pipeline_result["summary_path"])
    assert summary_path.exists()
    data = json.loads(summary_path.read_text())
    assert data["precision"] > 0.9
    assert data["recall"] > 0.9
    assert "ice_volume_m3" in data and "ice_mass_t" in data
