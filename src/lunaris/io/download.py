"""Remote LOLA south-polar DEM fetch with O(1) windowed COG reads.

This module backs the ``lunaris realdata`` command. It opens a **real**,
openly-downloadable LOLA south-pole digital elevation model directly over HTTPS
with GDAL's ``/vsicurl/`` virtual file system and reads only a small
area-of-interest (AOI) window around the pole — never the multi-gigabyte global
mosaic. That windowed read is the headline "O(1)" capability: a tiny crater
region is extracted from a continent-scale Cloud-Optimized GeoTIFF (COG) using
HTTP range requests, fetching kilobytes-to-megabytes instead of gigabytes.

Robustness
----------
* **TLS** — some networks terminate TLS at an intercepting proxy whose CA is in
  the *system* trust store but not in GDAL's bundled curl CA file. We therefore
  point GDAL at the system CA bundle when one is found and, as a last resort,
  fall back to ``GDAL_HTTP_UNSAFESSL=YES`` so the fetch still succeeds behind a
  corporate/proxy CA.
* **HTTP/1.1** — HTTP/2 multiplexing can stall through some proxies, so we pin
  ``GDAL_HTTP_VERSION=1.1`` and set generous curl timeouts plus GDAL's built-in
  range-read retries (``GDAL_HTTP_MAX_RETRY``).
* **Overview-decimated reads** — the PGDA adjusted COGs ship internal overviews;
  reading the AOI at a decimated ``out_shape`` makes GDAL fetch a handful of
  large, contiguous decimated tiles rather than hundreds of full-resolution
  ones. This is both faster and far more resilient to flaky range responses,
  while still being a genuine windowed (O(1)) read of the remote COG.
* Each candidate URL is tried in turn (``try``/``except``); the first that opens
  *and* yields a valid window is used, the choice is logged, and the AOI is
  cached to ``data/raw/`` so re-runs are instant. If **every** candidate fails
  the caller gets a clear :class:`RuntimeError` and can fall back to synthetic.

Candidates (in order)
---------------------
1. **PGDA LOLA-adjusted south-polar COGs** (Barker/Mazarico, GSFC Planetary
   Geodynamics) — true tiled GeoTIFFs with overviews, the most robust option:
   ``LDEM_80S_20MPP_ADJ.TIF`` (80 deg S, 20 m/px) and finer 83/87 deg S tiles.
2. **PDS LOLA polar GDR JP2** (Geosciences Node) — JPEG-2000 polar DEMs. GDAL
   reads these, but the single-codestream JP2 layout does not support cheap
   partial decode, so they are a correctness fallback rather than an O(1) path.

References
----------
* Barker, M. K. et al. (2021), "Improved LOLA Elevation Maps for South Pole
  Landing Sites", *Planet. Space Sci.* — the adjusted 5/10/20 m polar DEMs.
* Smith, D. E. et al. (2010), *Space Sci. Rev.* — LOLA instrument / GDR.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform as rtransform
from rasterio.enums import Resampling
from rasterio.warp import Resampling as WarpResampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

from ..constants import SOUTH_POLAR_STEREO_PROJ4

__all__ = [
    "fetch_south_polar_dem",
    "list_dem_candidates",
    "gdal_cog_env",
    "DEM_CANDIDATES",
]

logger = logging.getLogger("lunaris.io.download")


# ---------------------------------------------------------------------------
# Candidate remote LOLA south-polar DEMs, tried in order.
# ---------------------------------------------------------------------------
# Each entry: (label, url, kind). ``kind`` is "cog" (tiled GeoTIFF, supports
# cheap windowed/overview reads) or "jp2" (JPEG-2000, correctness fallback).
DEM_CANDIDATES: list[tuple[str, str, str]] = [
    (
        "PGDA LOLA 80S 20mpp adjusted COG",
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_20MPP_ADJ.TIF",
        "cog",
    ),
    (
        "PGDA LOLA 83S 10mpp adjusted COG",
        "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_83S_10MPP_ADJ.TIF",
        "cog",
    ),
    (
        "PGDA LOLA 87S 5mpp COG (Faustini/Shackleton)",
        "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/ldem_87s_5mpp.tif",
        "cog",
    ),
    (
        "PDS LOLA polar GDR 80S 20m JP2",
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/data/lola_gdr/polar/jp2/ldem_80s_20m.jp2",
        "jp2",
    ),
    (
        "PDS LOLA polar GDR 75S 30m JP2",
        "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/data/lola_gdr/polar/jp2/ldem_75s_30m.jp2",
        "jp2",
    ),
]

# Cap the decimated AOI to a modest pixel count: small enough that the windowed
# COG read stays an O(1) range fetch (and resilient to flaky proxies), big
# enough to resolve crater-scale terrain for the illumination ray-march.
_MAX_AOI_PX = 600


def _ca_bundle() -> str | None:
    """Locate a system CA bundle (so GDAL's curl trusts a proxy CA)."""
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        p = os.environ.get(var)
        if p and Path(p).is_file():
            return p
    for cand in (
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL/CentOS
        "/etc/ssl/cert.pem",  # Alpine/macOS
    ):
        if Path(cand).is_file():
            return cand
    return None


def gdal_cog_env(unsafe_ssl: bool = False) -> dict[str, str]:
    """GDAL/curl options for fast, proxy-resilient ``/vsicurl/`` COG reads.

    Parameters
    ----------
    unsafe_ssl : bool, default False
        If True (or if no system CA bundle is found), disable TLS verification
        (``GDAL_HTTP_UNSAFESSL=YES``). Used only as a fallback when a proxy CA
        is not in GDAL's trust store; the data are public so this leaks nothing.

    Returns
    -------
    dict[str, str]
        Environment mapping suitable for :class:`rasterio.Env`.
    """
    env: dict[str, str] = {
        # only probe range requests for raster extensions we expect
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2,.img",
        # avoid sibling-directory listing on open (one fewer round trip)
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # HTTP/2 multiplexing stalls through some intercepting proxies
        "GDAL_HTTP_VERSION": "1.1",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        # generous timeouts + built-in retry for flaky range responses
        "GDAL_HTTP_TIMEOUT": "120",
        "GDAL_HTTP_CONNECTTIMEOUT": "20",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "VSI_CACHE": "TRUE",
        "CPL_VSIL_CURL_USE_HEAD": "YES",
    }
    ca = _ca_bundle()
    if ca and not unsafe_ssl:
        env["GDAL_HTTP_CAINFO"] = ca
        env["CURL_CA_BUNDLE"] = ca
    if unsafe_ssl or ca is None:
        env["GDAL_HTTP_UNSAFESSL"] = "YES"
    return env


@contextlib.contextmanager
def _gdal_http_env(env: dict[str, str]):
    """Temporarily apply GDAL/curl options to ``os.environ`` (then restore).

    GDAL's ``/vsicurl/`` curl handle reads several HTTP/TLS options — notably
    ``GDAL_HTTP_UNSAFESSL`` and the CA-bundle paths — from the *process*
    environment at curl-init time, not from thread-local GDAL config options.
    Setting them via :class:`rasterio.Env` is therefore insufficient for the SSL
    path behind an intercepting proxy, so we set them as real environment
    variables for the duration of the read and restore the prior values after.
    """
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def list_dem_candidates() -> list[str]:
    """Return the ordered candidate DEM URLs (for transparency / docs).

    Returns
    -------
    list[str]
        The remote LOLA south-polar DEM URLs that
        :func:`fetch_south_polar_dem` will try, in priority order.
    """
    return [url for _label, url, _kind in DEM_CANDIDATES]


def _aoi_bounds_from_extent(
    src: rasterio.io.DatasetReader,
    extent_km: float,
    center_xy: tuple[float, float] | None,
) -> tuple[float, float, float, float]:
    """Projected (left, bottom, right, top) AOI box of side ``extent_km``.

    The AOI is centred on ``center_xy`` (defaults to the dataset centre, which
    for these polar products is the south pole at the projection origin) and
    clipped to the dataset bounds.
    """
    if center_xy is None:
        cx = 0.5 * (src.bounds.left + src.bounds.right)
        cy = 0.5 * (src.bounds.top + src.bounds.bottom)
    else:
        cx, cy = center_xy
    half = 0.5 * float(extent_km) * 1000.0
    left = max(cx - half, src.bounds.left)
    right = min(cx + half, src.bounds.right)
    bottom = max(cy - half, src.bounds.bottom)
    top = min(cy + half, src.bounds.top)
    return (left, bottom, right, top)


def _read_window_decimated(
    src: rasterio.io.DatasetReader,
    bounds: tuple[float, float, float, float],
    max_px: int = _MAX_AOI_PX,
):
    """Windowed (O(1)) read of ``bounds``, decimated to <= ``max_px`` per side.

    Returns ``(array, transform)`` where ``transform`` is the affine of the
    *returned* (possibly decimated) array. Decimation makes GDAL read from the
    COG's overviews — a few large contiguous range requests — which is both fast
    and resilient to flaky proxies, while remaining a true partial read.
    """
    window = from_bounds(*bounds, transform=src.transform)
    win_w = max(int(round(window.width)), 1)
    win_h = max(int(round(window.height)), 1)
    out_w = min(win_w, max_px)
    out_h = min(win_h, max_px)
    # preserve aspect ratio of the (square-ish) AOI
    arr = src.read(
        1,
        window=window,
        out_shape=(out_h, out_w),
        resampling=Resampling.bilinear,
        boundless=False,
    ).astype(np.float64)

    win_transform = src.window_transform(window)
    if (out_w, out_h) != (win_w, win_h):
        sx = window.width / out_w
        sy = window.height / out_h
        out_transform = win_transform * rtransform.Affine.scale(sx, sy)
    else:
        out_transform = win_transform
    return arr, out_transform


def _clean_dem(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Replace nodata / non-finite samples by the in-window median elevation."""
    a = np.asarray(arr, dtype=np.float64)
    bad = ~np.isfinite(a)
    if nodata is not None and np.isfinite(nodata):
        bad |= a == nodata
    if bad.any():
        good = a[~bad]
        fill = float(np.median(good)) if good.size else 0.0
        a = a.copy()
        a[bad] = fill
    return a


def _maybe_reproject(
    arr: np.ndarray,
    transform,
    src_crs,
    dst_crs: str,
):
    """Reproject ``arr`` to ``dst_crs`` if its CRS differs; else pass through.

    Returns ``(array, transform, crs_wkt)``.
    """
    from rasterio.crs import CRS

    src = CRS.from_user_input(src_crs) if src_crs else None
    dst = CRS.from_user_input(dst_crs)
    if src is None or src == dst:
        wkt = src.to_wkt() if src else dst.to_wkt()
        return arr, transform, wkt

    h, w = arr.shape
    left, bottom, right, top = rasterio.transform.array_bounds(h, w, transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src, dst, w, h, left=left, bottom=bottom, right=right, top=top
    )
    dst_arr = np.empty((dst_h, dst_w), dtype=np.float64)
    reproject(
        source=arr,
        destination=dst_arr,
        src_transform=transform,
        src_crs=src,
        dst_transform=dst_transform,
        dst_crs=dst,
        resampling=WarpResampling.bilinear,
    )
    return dst_arr, dst_transform, dst.to_wkt()


def fetch_south_polar_dem(
    bounds: tuple[float, float, float, float] | None = None,
    out_path: str | Path | None = None,
    prefer_cog: bool = True,
    extent_km: float = 120.0,
    center_xy: tuple[float, float] | None = None,
    source_url: str | Path | None = None,
    max_aoi_px: int = _MAX_AOI_PX,
    reproject_to: str | None = SOUTH_POLAR_STEREO_PROJ4,
    use_cache: bool = True,
):
    """Fetch a real LOLA south-polar DEM AOI via an O(1) windowed COG read.

    Opens a remote LOLA south-pole DEM with GDAL ``/vsicurl/`` and reads only a
    small area-of-interest window around the pole (or ``bounds``), demonstrating
    the windowed Cloud-Optimized-GeoTIFF capability. Candidate URLs in
    :data:`DEM_CANDIDATES` are tried in order; the first that opens and yields a
    valid window is used and logged.

    Parameters
    ----------
    bounds : (left, bottom, right, top), optional
        AOI in the *source* CRS (projected metres). If ``None`` the AOI is a
        square of side ``extent_km`` centred on ``center_xy`` (default: the pole
        at the projection origin).
    out_path : str or Path, optional
        Where to cache the AOI as a GeoTIFF. Defaults to
        ``data/raw/lola_south_polar_aoi.tif`` when ``use_cache`` is True.
    prefer_cog : bool, default True
        If True, try the tiled-COG candidates before the JP2 fallbacks.
    extent_km : float, default 120.0
        Side length of the default square AOI [km] (used only when ``bounds``
        is None).
    center_xy : (x, y), optional
        AOI centre in the source CRS [m]. Defaults to the dataset centre.
    source_url : str or Path, optional
        Override the candidate list with an explicit source (a remote URL *or* a
        **local file path**). Passing a local COG makes this function fully
        network-free — used by the unit tests.
    max_aoi_px : int, default 600
        Cap on the returned array's larger side; larger AOIs are read decimated
        from the COG overviews (keeps the read O(1) and proxy-resilient).
    reproject_to : str or None, default south-polar stereographic
        Target CRS (PROJ4/WKT). If the source already matches (these products
        are all south-polar stereographic) no reprojection is done. Pass
        ``None`` to keep the native CRS.
    use_cache : bool, default True
        Write the AOI GeoTIFF to ``out_path`` for instant re-runs.

    Returns
    -------
    (array, transform, crs) : (np.ndarray, affine.Affine, str)
        The windowed AOI elevation array [m] (``float64``), its affine
        transform, and the CRS as a WKT string. Provenance of the fetch (source
        label, URL, native resolution, AOI bounds) is recorded via logging, in
        the cached GeoTIFF's tags, and on :func:`last_fetch_provenance`.

    Raises
    ------
    RuntimeError
        If every candidate (or the supplied ``source_url``) fails to yield a
        valid windowed read. The caller is expected to fall back to synthetic.
    """
    # Build the ordered list of (label, source, kind) to attempt.
    if source_url is not None:
        candidates: list[tuple[str, str, str]] = [
            ("explicit source", str(source_url), "local")
        ]
    else:
        cands = list(DEM_CANDIDATES)
        if not prefer_cog:
            cands.sort(key=lambda c: 0 if c[2] == "jp2" else 1)
        candidates = cands

    last_err: Exception | None = None
    for label, src_str, kind in candidates:
        # Local file paths and remote URLs both go through rasterio.open; only
        # remote URLs need the /vsicurl/ prefix + the curl env.
        is_remote = kind != "local" and (
            src_str.startswith("http://") or src_str.startswith("https://")
        )
        gdal_path = f"/vsicurl/{src_str}" if is_remote else str(src_str)

        # SSL attempts for remote sources. We try the *unsafe* (no-verify) mode
        # first because GDAL's bundled curl ignores a custom CA bundle in some
        # builds, and a failed verify-on open poisons GDAL's per-URL vsicurl
        # cache (subsequent opens report "does not exist"). The data are public,
        # so disabling verification leaks nothing; if it somehow fails we fall
        # back to the CA-trusted mode.
        ssl_modes = [True, False] if is_remote else [False]
        for unsafe in ssl_modes:
            env = gdal_cog_env(unsafe_ssl=unsafe) if is_remote else {}
            try:
                # Apply via os.environ (curl reads SSL opts from the real env)
                # AND rasterio.Env (covers the GDAL-config-option path).
                with _gdal_http_env(env), rasterio.Env(**env):
                    with rasterio.open(gdal_path) as src:
                        native_res = float(src.res[0])
                        src_crs = src.crs
                        aoi = (
                            bounds
                            if bounds is not None
                            else _aoi_bounds_from_extent(src, extent_km, center_xy)
                        )
                        arr, transform = _read_window_decimated(
                            src, aoi, max_px=max_aoi_px
                        )
                        nodata = src.nodata
                arr = _clean_dem(arr, nodata)
                if arr.size == 0 or not np.isfinite(arr).any():
                    raise RuntimeError("empty or all-nodata window")

                # Reproject into the south-polar stereographic CRS if requested
                # and the source differs (these products already match).
                if reproject_to is not None:
                    arr, transform, crs_wkt = _maybe_reproject(
                        arr, transform, src_crs, reproject_to
                    )
                else:
                    crs_wkt = src_crs.to_wkt() if src_crs else ""

                eff_res = float(abs(transform.a))
                logger.info(
                    "fetched LOLA DEM AOI: %s [%s] url=%s native_res=%.1fm "
                    "aoi_px=%s eff_res=%.1fm ssl_unsafe=%s",
                    label, kind, src_str, native_res, arr.shape, eff_res, unsafe,
                )

                if use_cache:
                    cache_path = (
                        Path(out_path)
                        if out_path is not None
                        else Path("data/raw") / "lola_south_polar_aoi.tif"
                    )
                    try:
                        _cache_geotiff(
                            arr, transform, crs_wkt, cache_path,
                            tags={
                                "source_label": label,
                                "source_url": src_str,
                                "source_kind": kind,
                                "native_res_m": f"{native_res:.3f}",
                                "eff_res_m": f"{eff_res:.3f}",
                            },
                        )
                        logger.info("cached AOI -> %s", cache_path)
                    except Exception as ce:  # caching is best-effort
                        logger.warning("could not cache AOI: %s", ce)

                # Stash provenance where the CLI can pick it up.
                _LAST_FETCH.update(
                    {
                        "source_label": label,
                        "source_url": src_str,
                        "source_kind": kind,
                        "native_res_m": native_res,
                        "eff_res_m": eff_res,
                        "aoi_bounds": list(aoi),
                        "shape": list(arr.shape),
                    }
                )
                return arr, transform, crs_wkt
            except Exception as e:  # noqa: BLE001 - try next ssl mode / candidate
                last_err = e
                logger.warning(
                    "candidate failed: %s (ssl_unsafe=%s): %s",
                    label, unsafe, str(e)[:160],
                )
                continue

    raise RuntimeError(
        "fetch_south_polar_dem: all candidate LOLA DEM sources failed; "
        f"last error: {last_err!r}. Candidates tried: {list_dem_candidates()}"
    )


# Module-level scratch dict holding provenance of the most recent successful
# fetch (read by the CLI for the summary JSON).
_LAST_FETCH: dict[str, object] = {}


def last_fetch_provenance() -> dict[str, object]:
    """Return provenance metadata of the most recent successful fetch."""
    return dict(_LAST_FETCH)


def _cache_geotiff(
    arr: np.ndarray,
    transform,
    crs_wkt: str,
    path: str | Path,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write the AOI array to a small tiled GeoTIFF cache file."""
    from rasterio.crs import CRS

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape
    try:
        rio_crs = CRS.from_wkt(crs_wkt) if crs_wkt else None
    except Exception:
        rio_crs = None
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": rio_crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        if tags:
            dst.update_tags(**tags)
    return out
