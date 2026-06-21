"""Shared pytest fixtures for the lunaris test-suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from lunaris.io.synthetic import generate_faustini_scene
from lunaris.scene import LunarScene


@pytest.fixture(scope="session")
def faustini_scene() -> LunarScene:
    """A small deterministic synthetic Faustini scene (fast: n=128, seed=42)."""
    return generate_faustini_scene(n=128, seed=42)


@pytest.fixture()
def tmp_outputs(tmp_path: Path) -> Path:
    """A clean temporary outputs directory for artefact-writing tests."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out
