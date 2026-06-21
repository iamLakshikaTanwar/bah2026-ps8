"""Data ingestion: synthetic scene generation, real-raster readers, the
multi-sensor dataset registry, and the on-disk cache."""

from __future__ import annotations

from .cache import cached, memory
from .readers import read_cog_window, read_raster, reproject_to_south_polar
from .registry import DATASETS, get_dataset, list_datasets
from .synthetic import generate_faustini_scene

__all__ = [
    "generate_faustini_scene",
    "read_raster",
    "read_cog_window",
    "reproject_to_south_polar",
    "DATASETS",
    "list_datasets",
    "get_dataset",
    "memory",
    "cached",
]
