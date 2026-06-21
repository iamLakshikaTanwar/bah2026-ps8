"""The common in-memory data structure shared by every pipeline stage.

A :class:`LunarScene` bundles the co-registered raster layers (DEM, the full
L- and S-band Stokes vectors, derived radar parameters, thermal / illumination
/ optical layers, and the ice ground-truth mask) together with their geo-spatial
metadata (rasterio ``Affine`` transform, CRS string, pixel resolution).

Every downstream module consumes ``LunarScene`` and (optionally) attaches new
layers to ``scene.meta`` or returns plain numpy arrays. Persisting a scene
writes each layer as a Cloud-Optimized GeoTIFF (COG) in the lunar south-polar
stereographic projection so the artefacts open directly in QGIS/ArcGIS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

# Layer names that hold a 2-D float (or bool) raster. Order matters for I/O.
_FLOAT_LAYERS: tuple[str, ...] = (
    "dem",
    "s0_L", "s1_L", "s2_L", "s3_L",
    "s0_S", "s1_S", "s2_S", "s3_S",
    "cpr_L", "dop_L", "cpr_S", "dop_S",
    "temperature_max",
    "illumination",
    "albedo_1064",
    "lamp_ratio",
    "earth_visibility",
)
_BOOL_LAYERS: tuple[str, ...] = ("ice_truth",)
_ALL_LAYERS: tuple[str, ...] = _FLOAT_LAYERS + _BOOL_LAYERS


@dataclass
class LunarScene:
    """Co-registered multi-sensor lunar scene.

    Attributes
    ----------
    dem : np.ndarray
        Digital elevation model [m], shape ``(H, W)``.
    s0_L, s1_L, s2_L, s3_L : np.ndarray
        L-band Stokes parameters (total power and the three polarisation
        components) on the same grid.
    s0_S, s1_S, s2_S, s3_S : np.ndarray
        S-band Stokes parameters.
    cpr_L, dop_L, cpr_S, dop_S : np.ndarray
        Convenience radar products: circular polarisation ratio and degree of
        polarisation for each band.
    temperature_max : np.ndarray
        Annual maximum surface temperature [K] (Diviner-like).
    illumination : np.ndarray
        Annual average illuminated fraction in ``[0, 1]``.
    albedo_1064 : np.ndarray
        Normal albedo at 1064 nm (LOLA-like) — frost proxy.
    lamp_ratio : np.ndarray
        LAMP off-/on-band Lyman-alpha ratio — surface-frost proxy.
    earth_visibility : np.ndarray
        Fraction of time Earth is above the local horizon in ``[0, 1]``
        (direct-to-Earth comms availability).
    ice_truth : np.ndarray
        Boolean ground-truth mask of the subsurface-ice region.
    transform : affine.Affine
        rasterio affine transform mapping pixel -> projected (x, y).
    crs : str
        CRS as a PROJ4 / WKT / authority string.
    resolution_m : float
        Pixel size [m] (square pixels assumed).
    meta : dict
        Free-form metadata bag (target name, seed, provenance, extra layers).
    """

    dem: np.ndarray
    s0_L: np.ndarray
    s1_L: np.ndarray
    s2_L: np.ndarray
    s3_L: np.ndarray
    s0_S: np.ndarray
    s1_S: np.ndarray
    s2_S: np.ndarray
    s3_S: np.ndarray
    cpr_L: np.ndarray
    dop_L: np.ndarray
    cpr_S: np.ndarray
    dop_S: np.ndarray
    temperature_max: np.ndarray
    illumination: np.ndarray
    albedo_1064: np.ndarray
    lamp_ratio: np.ndarray
    earth_visibility: np.ndarray
    ice_truth: np.ndarray
    transform: Any  # affine.Affine
    crs: str
    resolution_m: float
    meta: dict = field(default_factory=dict)

    # -- geometry -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` of the raster grid."""
        return tuple(self.dem.shape)  # type: ignore[return-value]

    def layers(self) -> dict[str, np.ndarray]:
        """Return a mapping ``name -> 2-D array`` for every raster layer."""
        return {name: getattr(self, name) for name in _ALL_LAYERS}

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Write every layer as a COG GeoTIFF plus a ``scene.json`` sidecar.

        Float layers are stored as ``float32``; ``ice_truth`` as ``uint8``.
        The lunar south-polar CRS and affine transform are embedded in each
        GeoTIFF so the outputs are directly GIS-loadable.

        Parameters
        ----------
        directory : str or Path
            Output directory (created if absent).

        Returns
        -------
        Path
            The directory that was written to.
        """
        import rasterio
        from rasterio.crs import CRS

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        try:
            rio_crs = CRS.from_string(self.crs)
        except Exception:
            rio_crs = CRS.from_proj4(self.crs)

        h, w = self.shape
        for name, arr in self.layers().items():
            is_bool = name in _BOOL_LAYERS
            dtype = "uint8" if is_bool else "float32"
            data = arr.astype(dtype)
            profile = {
                "driver": "GTiff",
                "height": h,
                "width": w,
                "count": 1,
                "dtype": dtype,
                "crs": rio_crs,
                "transform": self.transform,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "compress": "deflate",
                # COG-friendly layout; overviews added below.
                "interleave": "band",
            }
            path = out / f"{name}.tif"
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")

        sidecar = {
            "crs": self.crs,
            "resolution_m": float(self.resolution_m),
            "transform": list(self.transform)[:6],
            "shape": list(self.shape),
            "layers": list(self.layers().keys()),
            "meta": _jsonify(self.meta),
        }
        (out / "scene.json").write_text(json.dumps(sidecar, indent=2))
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "LunarScene":
        """Reconstruct a :class:`LunarScene` previously written by :meth:`save`.

        Parameters
        ----------
        directory : str or Path
            Directory containing the per-layer GeoTIFFs and ``scene.json``.

        Returns
        -------
        LunarScene
        """
        import rasterio

        src_dir = Path(directory)
        sidecar = json.loads((src_dir / "scene.json").read_text())

        data: dict[str, Any] = {}
        transform = None
        crs = sidecar["crs"]
        for name in _ALL_LAYERS:
            path = src_dir / f"{name}.tif"
            with rasterio.open(path) as src:
                band = src.read(1)
                transform = src.transform
            if name in _BOOL_LAYERS:
                band = band.astype(bool)
            else:
                band = band.astype(np.float64)
            data[name] = band

        return cls(
            transform=transform,
            crs=crs,
            resolution_m=float(sidecar["resolution_m"]),
            meta=sidecar.get("meta", {}),
            **data,
        )

    # -- niceties -----------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        h, w = self.shape
        tgt = self.meta.get("target", "?")
        return (
            f"LunarScene(target={tgt!r}, shape=({h},{w}), "
            f"res={self.resolution_m} m, crs={self.crs!r}, "
            f"ice_px={int(np.asarray(self.ice_truth).sum())})"
        )


def _jsonify(obj: Any) -> Any:
    """Best-effort conversion of metadata values to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


__all__ = ["LunarScene"]

# Re-export the canonical layer ordering for downstream feature stacking.
LAYER_NAMES: tuple[str, ...] = _ALL_LAYERS

# Sanity: dataclass fields must cover all declared layers (+ geo metadata).
_declared = {f.name for f in fields(LunarScene)}
assert set(_ALL_LAYERS).issubset(_declared), "LunarScene missing a declared layer"
