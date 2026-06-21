"""End-to-end orchestration of the lunaris ice-detection & planning pipeline.

Implemented by: **integration agent**.

The pipeline runs nine stages, each delegating to a sub-package:

1. **Ingest** — load real Chandrayaan-2 DFSAR/OHRC granules (``lunaris.io``) or,
   absent data, the deterministic synthetic Faustini scene
   (:func:`lunaris.io.synthetic.generate_faustini_scene`).
2. **Stokes / CPR / DOP** — synthesise Stokes vectors and compute the circular
   polarisation ratio and degree of polarisation (``lunaris.polarimetry``).
3. **m-chi decomposition** — split surface / double-bounce / volume scatter to
   isolate the ice volume-scatter channel (``lunaris.polarimetry.decomposition``).
4. **Ice LUT classification** — apply the O(1) precomputed ``CPR>1 & DOP<0.13``
   LUT classifier (``lunaris.classify.ice_lut``).
5. **Terrain** — slope, roughness, illumination / shadow, and thermal cold-trap
   analysis (``lunaris.terrain``).
6. **Multi-sensor fusion** — combine radar, thermal, neutron and optical-frost
   evidence (Bayesian / Dempster-Shafer / learned) into an ice posterior
   (``lunaris.classify.fusion`` / ``lunaris.classify.evidence``).
7. **Landing** — AHP-weighted landing-site suitability and selection
   (``lunaris.planning.landing``).
8. **Traverse** — energy-aware rover path from the landing site to the ice
   target (``lunaris.planning``: A* / D* Lite / Theta* / RRT* / NSGA-II +
   ``energy``).
9. **Volume + report** — dielectric-mixing ice fraction, top-5 m Monte-Carlo
   ice volume/mass (``lunaris.volume``), then the HTML report (``lunaris.viz``).
"""

from __future__ import annotations

from typing import Any

from .config import Settings

__all__ = ["run_pipeline"]


def run_pipeline(config: Settings) -> dict[str, Any]:
    """Run the full nine-stage pipeline and return a results dictionary.

    Parameters
    ----------
    config : Settings
        Resolved configuration (see :func:`lunaris.config.load_config`).

    Returns
    -------
    dict
        Aggregated results: scene, ice masks, terrain layers, fusion posterior,
        landing sites, traverse path, ice-volume statistics, and figure paths.
    """
    raise NotImplementedError("integration agent")
