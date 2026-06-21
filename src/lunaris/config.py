"""Typed configuration for the ``lunaris`` pipeline (pydantic-settings).

:class:`Settings` centralises every tunable: filesystem paths, the ice-detection
thresholds (defaulting to :mod:`lunaris.constants`), the working grid size, and
the target name. Settings may be supplied via environment variables
(prefix ``LUNARIS_``) or loaded from a YAML file with :func:`load_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import (
    CPR_ICE_THRESHOLD,
    DOP_ICE_THRESHOLD,
)

__all__ = ["Settings", "load_config"]


class Settings(BaseSettings):
    """Runtime configuration for a pipeline invocation.

    Values resolve from (in order) explicit kwargs, ``LUNARIS_*`` environment
    variables, then the defaults below. :func:`load_config` overlays a YAML file
    on top of these defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="LUNARIS_",
        extra="ignore",
        validate_assignment=True,
    )

    # --- identity ------------------------------------------------------
    target_name: str = Field(
        default="Faustini",
        description="Name of the target crater / region.",
    )

    # --- filesystem paths ---------------------------------------------
    data_dir: Path = Field(default=Path("data"), description="Root data dir.")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw inputs.")
    interim_dir: Path = Field(default=Path("data/interim"),
                              description="Intermediate artefacts.")
    cache_dir: Path = Field(default=Path("data/cache"), description="joblib cache.")
    outputs_dir: Path = Field(default=Path("outputs"),
                              description="Figures / reports / GIS exports.")

    # --- grid ----------------------------------------------------------
    grid_size: int = Field(default=512, ge=16,
                           description="Working raster edge length [px].")
    resolution_m: float = Field(default=20.0, gt=0.0,
                                description="Ground sample distance [m].")
    seed: int = Field(default=42, description="Master RNG seed.")

    # --- ice-detection thresholds (default to constants) ---------------
    cpr_threshold: float = Field(
        default=CPR_ICE_THRESHOLD,
        description="Circular-polarisation-ratio lower bound for ice.",
    )
    dop_threshold: float = Field(
        default=DOP_ICE_THRESHOLD,
        description="Degree-of-polarisation upper bound for ice.",
    )

    # --- pipeline toggles ---------------------------------------------
    use_synthetic: bool = Field(
        default=True,
        description="Use the synthetic Faustini scene when no real data present.",
    )

    def ensure_dirs(self) -> "Settings":
        """Create all configured directories (idempotent). Returns ``self``."""
        for p in (self.data_dir, self.raw_dir, self.interim_dir,
                  self.cache_dir, self.outputs_dir):
            Path(p).mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None) -> Settings:
    """Load :class:`Settings`, overlaying a YAML file if ``path`` is given.

    Parameters
    ----------
    path : str or Path, optional
        YAML file whose keys override the defaults. If ``None`` (or the file is
        absent), pure defaults + environment variables are returned.

    Returns
    -------
    Settings
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            loaded = yaml.safe_load(p.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config {p} must be a YAML mapping, got {type(loaded)}")
            data = loaded
    return Settings(**data)
