"""Radar polarimetry: Stokes synthesis, CPR/DOP, m-chi/m-delta decompositions,
speckle filtering, and co-pol coherence.

(Function bodies are filled in by the polarimetry agent.)
"""

from __future__ import annotations

from .coherence import copol_coherence
from .cpr import circular_polarization_ratio, degree_of_polarization, sc_oc_power
from .decomposition import child_parameters, cloude_pottier, m_chi, m_delta
from .speckle import boxcar, refined_lee
from .stokes import multilook, scattering_matrix_to_circular, stokes_from_circular

__all__ = [
    "scattering_matrix_to_circular",
    "stokes_from_circular",
    "multilook",
    "circular_polarization_ratio",
    "degree_of_polarization",
    "sc_oc_power",
    "m_chi",
    "m_delta",
    "child_parameters",
    "cloude_pottier",
    "boxcar",
    "refined_lee",
    "copol_coherence",
]
