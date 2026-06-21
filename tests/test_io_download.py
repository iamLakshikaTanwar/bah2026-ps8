"""Network-free unit tests for :mod:`lunaris.io.download`.

These tests must run offline so CI stays green: instead of hitting the live
LOLA endpoints, they synthesise a small local Cloud-Optimized-GeoTIFF in
``tmp_path`` and exercise :func:`fetch_south_polar_dem` against it via the
``source_url=`` override (a local file path, never the network). They verify the
windowed read returns a correct AOI sub-window, that reprojection is a no-op
when the source already matches the target CRS, that a differing source CRS is
reprojected, and that :func:`list_dem_candidates` returns the candidate URLs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from lunaris.constants import SOUTH_POLAR_STEREO_PROJ4
from lunaris.io.download import (
    DEM_CANDIDATES,
    fetch_south_polar_dem,
    gdal_cog_env,
    last_fetch_provenance,
    list_dem_candidates,
)

# A south-polar-stereographic-like grid centred on the pole at (0, 0):
# 400x400 px at 50 m -> 20 km box spanning [-10000, 10000] m in x and y.
_N = 400
_RES = 50.0
_HALF = _N * _RES / 2.0  # 10 000 m


def _ramp_dem() -> np.ndarray:
    """A smooth, unique-per-pixel elevation field (so windows are identifiable)."""
    yy, xx = np.mgrid[0:_N, 0:_N].astype(np.float64)
    # distinct integer-ish elevations: row-major ramp plus a radial bowl.
    bowl = -0.01 * ((xx - _N / 2) ** 2 + (yy - _N / 2) ** 2)
    return yy * 1000.0 + xx + bowl


def _write_cog(path: Path, crs: CRS, dem: np.ndarray) -> Path:
    """Write ``dem`` as a small tiled GeoTIFF (COG-like) at the given CRS."""
    transform = from_origin(-_HALF, _HALF, _RES, _RES)  # top-left origin
    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 128,
        "blockysize": 128,
        "compress": "deflate",
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(dem.astype("float32"), 1)
    return path


@pytest.fixture()
def local_polar_cog(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A local synthetic COG already in the south-polar stereographic CRS."""
    dem = _ramp_dem()
    crs = CRS.from_proj4(SOUTH_POLAR_STEREO_PROJ4)
    path = _write_cog(tmp_path / "local_polar.tif", crs, dem)
    return path, dem


def test_list_dem_candidates_returns_urls() -> None:
    urls = list_dem_candidates()
    assert isinstance(urls, list) and len(urls) >= 2
    assert all(isinstance(u, str) and u.startswith("https://") for u in urls)
    # at least one tiled-COG (.tif) candidate and one JP2 fallback are present
    assert any(u.lower().endswith(".tif") for u in urls)
    assert any(u.lower().endswith(".jp2") for u in urls)
    # the public list matches the internal candidate table
    assert urls == [u for _l, u, _k in DEM_CANDIDATES]


def test_gdal_cog_env_keys() -> None:
    env = gdal_cog_env()
    assert env["GDAL_HTTP_VERSION"] == "1.1"  # proxy-resilient
    assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert ".tif" in env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"]
    # unsafe fallback can be forced
    assert gdal_cog_env(unsafe_ssl=True)["GDAL_HTTP_UNSAFESSL"] == "YES"


def test_fetch_reads_full_window_from_local_cog(
    local_polar_cog: tuple[Path, np.ndarray],
) -> None:
    """Reading the whole extent from a local COG returns the source DEM."""
    path, dem = local_polar_cog
    arr, transform, crs = fetch_south_polar_dem(
        source_url=path,
        extent_km=100.0,  # larger than the 20 km raster -> clipped to full extent
        max_aoi_px=_N,  # no decimation
        use_cache=False,
        reproject_to=SOUTH_POLAR_STEREO_PROJ4,
    )
    assert arr.shape == (_N, _N)
    # source already in target CRS -> values preserved (float32 round-trip)
    np.testing.assert_allclose(arr, dem.astype("float32"), rtol=0, atol=1e-3)
    assert crs  # non-empty WKT
    assert abs(transform.a - _RES) < 1e-6


def test_fetch_reads_correct_subwindow(
    local_polar_cog: tuple[Path, np.ndarray],
) -> None:
    """An explicit ``bounds`` AOI returns exactly that geographic sub-window."""
    path, dem = local_polar_cog
    # AOI = upper-right quadrant in projected coords: x in [0, HALF], y in [0, HALF]
    bounds = (0.0, 0.0, _HALF, _HALF)
    arr, transform, _crs = fetch_south_polar_dem(
        source_url=path,
        bounds=bounds,
        max_aoi_px=_N,  # no decimation so we can compare pixel-for-pixel
        use_cache=False,
        reproject_to=SOUTH_POLAR_STEREO_PROJ4,
    )
    # that quadrant is the top-right NxN/2 block of the array (rows 0..N/2,
    # cols N/2..N) given a top-left origin transform.
    half = _N // 2
    expected = dem[0:half, half:_N].astype("float32")
    assert arr.shape == expected.shape
    np.testing.assert_allclose(arr, expected, rtol=0, atol=1e-3)
    # the returned window transform's top-left maps to (x=0, y=HALF)
    x0, y0 = transform * (0, 0)
    assert abs(x0 - 0.0) < _RES and abs(y0 - _HALF) < _RES


def test_fetch_decimates_large_window(
    local_polar_cog: tuple[Path, np.ndarray],
) -> None:
    """A small ``max_aoi_px`` decimates the window (O(1)-style coarse read)."""
    path, _dem = local_polar_cog
    arr, transform, _crs = fetch_south_polar_dem(
        source_url=path,
        extent_km=100.0,
        max_aoi_px=100,  # force decimation from 400 -> 100 px
        use_cache=False,
    )
    assert max(arr.shape) <= 100
    # effective resolution coarsened by ~4x
    assert abs(transform.a) > _RES


def test_fetch_reprojects_when_source_crs_differs(tmp_path: Path) -> None:
    """A source in a different (same-body) CRS is reprojected to the target.

    The source is tagged in a *lunar north*-polar stereographic CRS (same Moon
    sphere, so PROJ can transform it) which differs from the south-polar target;
    this exercises the reprojection branch with a valid coordinate operation.
    """
    dem = _ramp_dem()
    # Lunar NORTH-polar stereographic on the same Moon sphere: differs from the
    # south-polar target (lat_0=+90 vs -90) but is transformable (same datum).
    north_proj4 = (
        "+proj=stere +lat_0=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
        "+R=1737400 +units=m +no_defs +type=crs"
    )
    other = CRS.from_proj4(north_proj4)
    path = _write_cog(tmp_path / "other_crs.tif", other, dem)
    arr, transform, crs = fetch_south_polar_dem(
        source_url=path,
        extent_km=100.0,
        max_aoi_px=_N,
        use_cache=False,
        reproject_to=SOUTH_POLAR_STEREO_PROJ4,
    )
    assert arr.ndim == 2 and arr.size > 0
    assert np.isfinite(arr).any()
    # target CRS WKT mentions a polar stereographic projection on the Moon radius
    assert "Stereographic" in crs or "stere" in crs.lower()


def test_fetch_caches_geotiff(local_polar_cog: tuple[Path, np.ndarray],
                              tmp_path: Path) -> None:
    """``use_cache`` writes a readable GeoTIFF with provenance tags."""
    path, _dem = local_polar_cog
    cache = tmp_path / "cache" / "aoi.tif"
    arr, _t, _c = fetch_south_polar_dem(
        source_url=path,
        extent_km=100.0,
        max_aoi_px=_N,
        out_path=cache,
        use_cache=True,
    )
    assert cache.exists()
    with rasterio.open(cache) as src:
        assert src.read(1).shape == arr.shape
        tags = src.tags()
        assert tags.get("source_url") == str(path)
    prov = last_fetch_provenance()
    assert prov.get("source_url") == str(path)
    assert prov.get("shape") == list(arr.shape)


def test_fetch_raises_when_all_sources_fail(tmp_path: Path) -> None:
    """A non-existent local source raises a clear RuntimeError (caller fallback)."""
    missing = tmp_path / "does_not_exist.tif"
    with pytest.raises(RuntimeError, match="all candidate|failed"):
        fetch_south_polar_dem(source_url=missing, use_cache=False)
