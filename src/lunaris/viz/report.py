"""Final HTML report assembly.

Implemented by: **viz agent**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

__all__ = ["build_report"]


def build_report(results: Mapping[str, Any], out_html: str | Path) -> Path:
    """Assemble the end-to-end results into a single HTML report.

    Embeds the ice map, CPR-DOP scatter, m-chi RGB, terrain 3-D view, landing
    site, traverse and ice-volume estimate (with uncertainty) into one
    shareable, self-contained HTML deliverable.

    Parameters
    ----------
    results : mapping
        The dict returned by :func:`lunaris.pipeline.run_pipeline` (figures,
        masks, volume stats, paths, metadata).
    out_html : str or Path
        Output HTML path.

    Returns
    -------
    Path
        The written report path.
    """
    raise NotImplementedError("viz agent")
