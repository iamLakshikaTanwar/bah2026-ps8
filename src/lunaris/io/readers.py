"""Real-data raster I/O helpers built on rasterio / GDAL.

These functions back the "real granule" path of the pipeline once Chandrayaan-2
DFSAR/OHRC and the supporting LRO/Kaguya products are downloaded. They favour
**O(1) windowed reads** of Cloud-Optimized GeoTIFFs (COGs) via HTTP-range
requests so a tiny crater AOI never forces a full-image download.

Notes
-----
* PDS4 archives (e.g. Chandrayaan-2 PRADAN, LRO PDS) can be opened directly with
  GDAL's PDS4 driver or with the ``pds4_tools`` package; convert to COG once and
  then use :func:`read_cog_window` for fast AOI access.
* The GDAL environment toggles set below make ``/vsicurl/`` remote reads cheap:
  ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`` avoids sibling-file listing and
  ``CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif`` restricts range-request probing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

from ..constants import SOUTH_POLAR_STEREO_PROJ4

__all__ = [
    "read_raster",
    "read_cog_window",
    "reproject_to_south_polar",
    "GDAL_COG_ENV",
]

# GDAL configuration recommended for fast remote-COG windowed reads.
GDAL_COG_ENV: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "VSI_CACHE": "TRUE",
}


def read_raster(path: str | Path) -> tuple[np.ndarray, Any, str]:
    """Read a single-band raster fully into memory.

    Parameters
    ----------
    path : str or Path
        Local path or GDAL-readable URI.

    Returns
    -------
    (array, transform, crs) : (np.ndarray, affine.Affine, str)
        Band-1 array as ``float64``, the affine transform, and the CRS as a
        string (WKT). ``crs`` is ``""`` if the source is unreferenced.
    """
    with rasterio.Env(**GDAL_COG_ENV):
        with rasterio.open(str(path)) as src:
            arr = src.read(1).astype(np.float64)
            transform = src.transform
            crs = src.crs.to_wkt() if src.crs else ""
    return arr, transform, crs


def read_cog_window(
    url_or_path: str | Path,
    bounds: tuple[float, float, float, float] | None = None,
    band: int = 1,
) -> tuple[np.ndarray, Any, str]:
    """Windowed (O(1)) read of a COG over an optional ``bounds`` AOI.

    Uses :func:`rasterio.windows.from_bounds` so that only the byte ranges
    overlapping the requested projected bounding box are fetched — ideal for
    extracting a single doubly-shadowed crater from a continent-scale mosaic.

    Parameters
    ----------
    url_or_path : str or Path
        Local path or remote ``/vsicurl/`` / ``https://`` COG URL.
    bounds : (left, bottom, right, top), optional
        AOI in the *source* CRS. If ``None`` the full raster is read.
    band : int, default 1
        1-based band index.

    Returns
    -------
    (array, transform, crs) : (np.ndarray, affine.Affine, str)
        The windowed array (``float64``), the window's affine transform, and the
        source CRS as WKT.
    """
    with rasterio.Env(**GDAL_COG_ENV):
        with rasterio.open(str(url_or_path)) as src:
            if bounds is None:
                arr = src.read(band).astype(np.float64)
                transform = src.transform
            else:
                window = from_bounds(*bounds, transform=src.transform)
                arr = src.read(band, window=window).astype(np.float64)
                transform = src.window_transform(window)
            crs = src.crs.to_wkt() if src.crs else ""
    return arr, transform, crs


def reproject_to_south_polar(
    array: np.ndarray,
    src_transform: Any,
    src_crs: str,
    dst_crs: str = SOUTH_POLAR_STEREO_PROJ4,
    resolution_m: float | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, Any, str]:
    """Reproject a raster into the lunar south-polar stereographic CRS.

    Parameters
    ----------
    array : np.ndarray
        Source band, shape ``(H, W)``.
    src_transform : affine.Affine
        Source affine transform.
    src_crs : str
        Source CRS (PROJ4 / WKT / authority string).
    dst_crs : str, default :data:`SOUTH_POLAR_STEREO_PROJ4`
        Target CRS.
    resolution_m : float, optional
        Target pixel size [m]. If ``None``, GDAL chooses one preserving the
        source pixel count.
    resampling : rasterio.warp.Resampling, default bilinear
        Resampling kernel.

    Returns
    -------
    (array, transform, crs) : (np.ndarray, affine.Affine, str)
        The reprojected band, its transform, and ``dst_crs``.
    """
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    h, w = array.shape
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, src_transform)
    res = (resolution_m, resolution_m) if resolution_m else None
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src, dst, w, h, left=left, bottom=bottom, right=right, top=top,
        resolution=res,
    )
    dst_arr = np.empty((dst_h, dst_w), dtype=np.float64)
    reproject(
        source=array,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src,
        dst_transform=dst_transform,
        dst_crs=dst,
        resampling=resampling,
    )
    return dst_arr, dst_transform, dst_crs
