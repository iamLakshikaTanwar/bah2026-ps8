"""Illumination, horizon, permanent-/double-shadow, Earth-visibility and
sky-view-factor modelling from a DEM.

Implemented by: **terrain agent**.

All horizon geometry is computed by an honest outward ray-march over the DEM.
For a target pixel at elevation ``z0`` and a sample at horizontal distance ``d``
with elevation ``z(d)``, the *terrain* elevation angle to that sample is

    theta = arctan( ( z(d) - z0 - d^2 / (2 R) ) / d )

where the ``d^2 / (2 R)`` term (R = lunar radius) is the planetary-curvature
drop that lowers a distant horizon below the local tangent plane (Mazarico et
al. 2011, eq. for the curvature correction to topographic horizons). The
per-azimuth horizon is the maximum of ``theta`` along the ray.

Because the Moon's obliquity is only ``MOON_OBLIQUITY_DEG ~ 1.543 deg``, the
Sun's centre never rises more than ~1.5 deg above the horizontal at the south
pole; a point is permanently shadowed (PSR) when its horizon exceeds that solar
elevation in *every* azimuth from which the Sun can appear (Mazarico et al.
2011; Paige et al. 2010).

References
----------
* Mazarico, E. et al. (2011), "Illumination conditions of the lunar polar
  regions using LOLA topography", *Icarus* 211, 1066-1081 — horizon ray-tracing
  with curvature correction and the PSR definition.
* Paige, D. A. et al. (2010), "Diviner Lunar Radiometer observations of cold
  traps in the Moon's south polar region", *Science* 330, 479-482 — PSR /
  cold-trap context.
* Mahanti, P. et al. / Dozier & Frew (1990) — sky-view-factor definition
  ``SVF = 1 - mean_az( sin(max(horizon, 0)) )`` for radiative cooling.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..constants import MOON_OBLIQUITY_DEG, MOON_RADIUS_M

__all__ = [
    "horizon_map",
    "permanent_shadow_mask",
    "double_shadow_mask",
    "earth_visibility_map",
    "sky_view_factor",
]


def _ray_horizons(
    dem: np.ndarray,
    res: float,
    azimuths_rad: np.ndarray,
    step_px: float = 1.0,
    max_range_px: int | None = None,
) -> np.ndarray:
    """Per-azimuth maximum terrain-elevation angle for every pixel.

    Honest outward ray-march: for each azimuth the ray is advanced in
    ``step_px``-pixel increments and the elevation profile is sampled by bilinear
    interpolation (:func:`scipy.ndimage.map_coordinates`, ``order=1``). At range
    ``d = step * step_px * res`` the terrain elevation angle is

        theta = arctan( (z(d) - z0 - d^2/(2 R)) / d )

    (Mazarico et al. 2011, curvature-corrected horizon), and the running maximum
    over the ray gives that azimuth's horizon angle.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    azimuths_rad : np.ndarray
        Azimuth directions [rad]. ``az = 0`` points toward ``-y`` (north, i.e.
        decreasing row); azimuth increases clockwise toward ``+x`` (east).
    step_px : float, default 1.0
        Ray step in pixels (larger = faster, coarser).
    max_range_px : int, optional
        Maximum march length in steps; defaults to the grid diagonal.

    Returns
    -------
    np.ndarray
        Horizon elevation angles [rad], shape ``(n_az, H, W)``.
    """
    dem = np.asarray(dem, dtype=np.float64)
    H, W = dem.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z0 = dem
    if max_range_px is None:
        max_range_px = int(np.hypot(H, W)) + 1
    n_steps = max(int(max_range_px / step_px), 1)

    out = np.empty((azimuths_rad.size, H, W), dtype=np.float64)
    inv_2R = 1.0 / (2.0 * MOON_RADIUS_M)

    for ai, az in enumerate(azimuths_rad):
        # az=0 -> north (-y); clockwise. dy is the row increment, dx the col.
        dy = -np.cos(az)
        dx = np.sin(az)
        best = np.full((H, W), -np.inf, dtype=np.float64)
        for step in range(1, n_steps + 1):
            dist_px = step * step_px
            sy = yy + dy * dist_px
            sx = xx + dx * dist_px
            valid = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
            if not valid.any():
                break
            z = map_coordinates(
                dem, np.vstack([sy.ravel(), sx.ravel()]), order=1, mode="nearest"
            ).reshape(H, W)
            d = dist_px * res
            drop = d * d * inv_2R
            ang = np.arctan2(z - z0 - drop, d)
            np.maximum(best, np.where(valid, ang, -np.inf), out=best)
        # pixels whose ray immediately leaves the grid keep a flat horizon of 0
        best[~np.isfinite(best)] = 0.0
        out[ai] = best
    return out


def horizon_map(
    dem: np.ndarray,
    res: float,
    n_azimuth: int = 180,
    step_px: float = 1.0,
    max_range_px: int | None = None,
) -> np.ndarray:
    """Per-pixel horizon-elevation angle in each azimuth (degrees).

    Ray-traces the DEM outward in ``n_azimuth`` equally spaced directions,
    recording for each the maximum curvature-corrected terrain-elevation angle
    (Mazarico et al. 2011). Complexity is ``O(n_azimuth * range * H * W)``;
    ``range`` is the grid diagonal in steps, so cost scales as the cube of the
    grid edge. For large scenes raise ``step_px`` (coarser march) and/or lower
    ``n_azimuth`` to trade accuracy for speed.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    n_azimuth : int, default 180
        Number of azimuth samples (uniform over ``[0, 2*pi)``).
    step_px : float, default 1.0
        Ray step length in pixels.
    max_range_px : int, optional
        Maximum march length [steps]; defaults to the grid diagonal.

    Returns
    -------
    np.ndarray
        Horizon elevation angle [deg], shape ``(n_azimuth, H, W)``.
    """
    az = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    return np.degrees(_ray_horizons(dem, res, az, step_px=step_px,
                                    max_range_px=max_range_px))


def permanent_shadow_mask(
    dem: np.ndarray,
    res: float,
    sun_elev_deg: float = MOON_OBLIQUITY_DEG,
    n_azimuth: int = 72,
    step_px: float = 1.0,
) -> np.ndarray:
    """Boolean permanently-shadowed-region (PSR) mask (Mazarico et al. 2011).

    The Sun's centre at the south pole never exceeds ``sun_elev_deg`` above the
    horizontal (bounded by the lunar obliquity ``MOON_OBLIQUITY_DEG`` plus the
    pixel's polar latitude offset, here taken as the obliquity ceiling). A pixel
    is *potentially lit* if in **any** azimuth its horizon falls below
    ``sun_elev_deg`` (the Sun can peek over the lowest horizon); it is a PSR
    otherwise — i.e. its horizon meets or exceeds the solar elevation in every
    direction, so the Sun is never visible.

        lit  = ANY_az( horizon(az) <  sun_elev )
        PSR  = NOT lit = ALL_az( horizon(az) >= sun_elev )

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    sun_elev_deg : float, default ``MOON_OBLIQUITY_DEG`` (1.543)
        Maximum solar-centre elevation at the pole [deg].
    n_azimuth : int, default 72
        Number of azimuth samples for the horizon ray-march.
    step_px : float, default 1.0
        Ray step length in pixels.

    Returns
    -------
    np.ndarray
        Boolean PSR mask, shape ``(H, W)``. (Cross-checks strongly against the
        scene ``illumination < 0.05`` region.)
    """
    az = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    horizons = _ray_horizons(dem, res, az, step_px=step_px)  # (n_az, H, W) [rad]
    sun = np.radians(sun_elev_deg)
    lit = np.any(horizons < sun, axis=0)
    return ~lit


def double_shadow_mask(
    dem: np.ndarray,
    res: float,
    psr_mask: np.ndarray,
    n_azimuth: int = 72,
    secondary_range_m: float = 800.0,
    step_px: float = 1.0,
) -> np.ndarray:
    """Boolean "doubly-shadowed" mask within a PSR (O'Brien & Byrne 2022).

    A permanently shadowed pixel still receives weak *secondary* illumination
    (single-scattered sunlight + thermal re-radiation) if it has a direct line
    of sight to nearby *directly-lit* terrain — typically an illuminated crater
    wall that rises above its local horizon. A pixel is **doubly shadowed** when
    it can see *no* such sunlit surface within the range over which secondary
    illumination matters: every nearby skyline point is itself permanently
    shadowed (O'Brien & Byrne 2022; Mazarico et al. 2011).

    Because scattered/thermal flux falls off with distance (~1/d^2), only the
    *local* skyline contributes meaningfully; the line-of-sight search is
    therefore capped at ``secondary_range_m``. A distant sunlit rim seen at
    grazing range across a wide crater floor does not rescue a deep nested
    sub-crater whose own walls are dark.

    Ray-cast implementation (documented, robust approximate view-shed):

    1. March each azimuth ray outward only to ``secondary_range_m``.
    2. Track the running-maximum terrain-elevation angle (the local skyline) and
       whether the silhouette point that set it is *lit* (not in ``psr_mask``).
    3. A PSR pixel "sees a lit rim" if, in any azimuth, the local skyline
       silhouette is lit terrain rising above the horizontal.
    4. Doubly-shadowed = PSR pixels that see **no** lit rim within range.

    The result is always a subset of ``psr_mask`` and isolates the coldest
    deepest nested-crater floors.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    psr_mask : np.ndarray
        Boolean PSR mask from :func:`permanent_shadow_mask`, shape ``(H, W)``.
    n_azimuth : int, default 72
        Number of azimuth samples.
    secondary_range_m : float, default 800.0
        Maximum line-of-sight range over which a lit rim can supply secondary
        illumination [m].
    step_px : float, default 1.0
        Ray step length in pixels.

    Returns
    -------
    np.ndarray
        Boolean doubly-shadowed mask (subset of ``psr_mask``), shape ``(H, W)``.
    """
    dem = np.asarray(dem, dtype=np.float64)
    psr_mask = np.asarray(psr_mask, dtype=bool)
    H, W = dem.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z0 = dem
    inv_2R = 1.0 / (2.0 * MOON_RADIUS_M)
    az = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    # cap the march at the secondary-illumination range (in steps)
    range_px = max(secondary_range_m / res, 2.0)
    n_steps = int(range_px / step_px) + 1

    # lit terrain = potentially-illuminated (not PSR). float lookup for the
    # interpolated "is the silhouette point lit?" test.
    lit_field = (~psr_mask).astype(np.float64)

    # For each pixel: does ANY azimuth's local skyline silhouette land on lit
    # terrain (above the horizontal)?
    sees_lit_rim = np.zeros((H, W), dtype=bool)

    for a in az:
        dy = -np.cos(a)
        dx = np.sin(a)
        best = np.full((H, W), -np.inf)
        best_lit = np.zeros((H, W), dtype=bool)  # is current skyline point lit?
        for step in range(1, n_steps + 1):
            dist_px = step * step_px
            sy = yy + dy * dist_px
            sx = xx + dx * dist_px
            valid = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
            if not valid.any():
                break
            coords = np.vstack([sy.ravel(), sx.ravel()])
            z = map_coordinates(dem, coords, order=1, mode="nearest").reshape(H, W)
            litv = map_coordinates(
                lit_field, coords, order=1, mode="nearest"
            ).reshape(H, W) > 0.5
            d = dist_px * res
            drop = d * d * inv_2R
            ang = np.where(valid, np.arctan2(z - z0 - drop, d), -np.inf)
            newmax = ang > best
            best = np.where(newmax, ang, best)
            # whenever this sample sets a new local skyline, remember if it's lit
            best_lit = np.where(newmax, litv, best_lit)
        # a lit skyline rising above the local horizontal => sees a sunlit rim.
        sees_lit_rim |= best_lit & (best > 0.0)

    # doubly shadowed = PSR pixel that sees no lit rim within range.
    return psr_mask & (~sees_lit_rim)


def earth_visibility_map(
    dem: np.ndarray,
    res: float,
    earth_elev_deg: float = 0.0,
    earth_azimuth_deg: float = 0.0,
    tol_deg: float = 0.5,
    step_px: float = 1.0,
) -> np.ndarray:
    """Boolean direct-to-Earth (DTE) visibility map.

    Seen from the lunar south pole, Earth librates about the horizon near
    longitude 0 (toward ``earth_azimuth_deg``). A pixel has a direct line to
    Earth when its horizon *in the Earth direction* lies below Earth's elevation
    (plus a small libration tolerance):

        DTE = horizon(earth_azimuth) <= earth_elev + tol

    (Mazarico et al. 2011 use the same horizon test for Earth visibility / comms
    windows.)

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    earth_elev_deg : float, default 0.0
        Earth-centre elevation above the horizontal [deg].
    earth_azimuth_deg : float, default 0.0
        Azimuth toward Earth [deg] (0 = toward ``-y``/north of the grid).
    tol_deg : float, default 0.5
        Libration tolerance added to Earth's elevation [deg].
    step_px : float, default 1.0
        Ray step length in pixels.

    Returns
    -------
    np.ndarray
        Boolean DTE-visibility mask, shape ``(H, W)``.
    """
    az = np.array([np.radians(earth_azimuth_deg)])
    horizon = _ray_horizons(dem, res, az, step_px=step_px)[0]  # (H, W) [rad]
    return np.degrees(horizon) <= (earth_elev_deg + tol_deg)


def sky_view_factor(
    dem: np.ndarray,
    res: float,
    n_azimuth: int = 72,
    step_px: float = 1.0,
) -> np.ndarray:
    """Sky-view factor (visible-sky fraction) per pixel.

    For an isotropic sky the fraction of the upward hemisphere not blocked by
    terrain is (Dozier & Frew 1990; Mahanti et al.)::

        SVF = 1 - mean_az( sin( max(horizon(az), 0) ) )

    where negative (below-horizontal) horizons are clamped to 0 (the flat sky
    is fully open). ``SVF -> 1`` on an open plain and decreases inside craters,
    driving their radiative cooling.

    Parameters
    ----------
    dem : np.ndarray
        Elevation [m], shape ``(H, W)``.
    res : float
        Pixel size [m].
    n_azimuth : int, default 72
        Number of azimuth samples.
    step_px : float, default 1.0
        Ray step length in pixels.

    Returns
    -------
    np.ndarray
        Sky-view factor in ``[0, 1]``, shape ``(H, W)``.
    """
    az = np.linspace(0.0, 2.0 * np.pi, int(n_azimuth), endpoint=False)
    horizons = _ray_horizons(dem, res, az, step_px=step_px)  # (n_az, H, W) [rad]
    svf = 1.0 - np.mean(np.sin(np.clip(horizons, 0.0, np.pi / 2.0)), axis=0)
    return np.clip(svf, 0.0, 1.0)
