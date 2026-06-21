"""``lunaris`` command-line interface (Typer).

Exposes the console-script entry point ``lunaris`` (see ``pyproject.toml``).
``demo`` runs the full nine-stage pipeline on the deterministic synthetic
Faustini scene and prints a rich mission summary; ``run`` does the same from a
YAML config; ``version`` prints the package version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Settings, load_config

app = typer.Typer(
    add_completion=False,
    help="Lunar Subsurface-Ice Detection & Mission-Planning Platform (BAH 2026 PS8).",
)
console = Console()


def _fmt(x: Any, nd: int = 3) -> str:
    """Format a number compactly, or ``-`` for ``None``/NaN."""
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:  # NaN
        return "-"
    if abs(xf) >= 1e6 or (0 < abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:,.{nd}f}"


def _print_summary(s: dict[str, Any]) -> None:
    """Render the headline mission-summary table from a pipeline summary dict."""
    console.print(Panel.fit(
        f"[bold cyan]LUNARIS[/] — {s['target']} "
        f"([{'green' if s['ingest_source'] == 'real-data' else 'yellow'}]"
        f"{s['ingest_source']}[/], grid {s['grid'][0]}×{s['grid'][1]} @ "
        f"{_fmt(s['resolution_m'], 1)} m)",
        border_style="cyan",
    ))

    t = Table(title="Mission summary", show_lines=False, header_style="bold")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    t.add_row("[bold]— Ice detection —", "")
    t.add_row("O(1) LUT == direct rule", "yes" if s["lut_matches_rule"] else "NO")
    t.add_row("Precision", _fmt(s["precision"]))
    t.add_row("Recall", _fmt(s["recall"]))
    t.add_row("F1", _fmt(s["f1"]))
    t.add_row("Detected ice pixels", _fmt(s["ice_pixels"], 0))
    t.add_row("RF fusion CV-F1", f"{_fmt(s['rf_cv_f1_mean'])} ± {_fmt(s['rf_cv_f1_std'])}")
    t.add_row("Top fusion feature", str(s["top_feature"]))

    t.add_row("[bold]— Terrain —", "")
    t.add_row("Hurst exponent", _fmt(s["hurst_exponent"]))
    t.add_row("Cold-trap pixels (<110 K)", _fmt(s["cold_trap_pixels"], 0))
    t.add_row("Median slope [deg]", _fmt(s["median_slope_deg"]))

    t.add_row("[bold]— Landing & traverse —", "")
    t.add_row("Landing site (r, c)", f"{tuple(s['landing_site'])}")
    t.add_row("Landing suitability", _fmt(s["landing_score"]))
    t.add_row("AHP consistency ratio", _fmt(s["ahp_consistency_ratio"]))
    t.add_row("Ice access (r, c)", f"{tuple(s['ice_access'])}")
    t.add_row("Traverse length", f"{_fmt(s['traverse_length_m'], 1)} m "
                                  f"({_fmt(s['traverse_length_km'])} km)")
    t.add_row("Energy feasible", "[green]yes[/]" if s["energy_feasible"]
                                 else "[red]no[/]")
    t.add_row("Dark dwell [h] (≤70)", _fmt(s["dark_hours"]))
    t.add_row("Min SOC", _fmt(s["min_soc"]))

    t.add_row("[bold]— Ice inventory (top 5 m) —", "")
    t.add_row("Ice area [m²]", _fmt(s["ice_area_m2"], 0))
    t.add_row("Ice volume fraction", _fmt(s["ice_fraction"]))
    t.add_row("Ice volume [m³]", _fmt(s["ice_volume_m3"], 0))
    ci = s["ice_mass_ci_t"]
    t.add_row("Ice mass [t]", f"{_fmt(s['ice_mass_t'], 1)} "
                              f"(95% CI {_fmt(ci[0], 1)}–{_fmt(ci[1], 1)})")
    t.add_row("LCROSS cross-check [t]",
              _fmt(s["lcross_cross_check_kg"] / 1000.0, 1))
    console.print(t)


def _run_and_report(settings: Settings) -> None:
    """Execute the pipeline and print outputs + the summary table."""
    from .pipeline import run_pipeline  # local import: heavy deps

    console.print(f"[bold cyan]lunaris[/] v{__version__} — target "
                  f"[bold]{settings.target_name}[/] → "
                  f"[dim]{settings.outputs_dir}[/]")
    with console.status("[cyan]running 9-stage pipeline…", spinner="dots"):
        results = run_pipeline(settings)

    _print_summary(results["summary"])

    out = Path(settings.outputs_dir)
    console.print(
        f"\n[green]✓ outputs written[/] → "
        f"[bold]{out / 'lunaris_report.html'}[/]\n"
        f"  • report HTML, 3-D terrain, figures/*.png, *.tif (COG), "
        f"results_summary.json"
    )


@app.command()
def run(
    config: str = typer.Option(
        "configs/faustini.yaml", "--config", "-c", help="Path to a YAML config."
    ),
) -> None:
    """Run the full ice-detection & mission-planning pipeline from a config."""
    settings = load_config(config)
    _run_and_report(settings)


@app.command()
def demo(
    n: int = typer.Option(256, "--n", help="Grid edge length [px]."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    out: str = typer.Option("outputs/faustini", "--out", help="Output directory."),
) -> None:
    """Run the full pipeline on the synthetic Faustini scene and report results.

    A self-contained, deterministic end-to-end demo: detection → terrain →
    fusion → landing → traverse → ice inventory → HTML report. No external data
    required.
    """
    settings = Settings(grid_size=n, seed=seed, outputs_dir=Path(out))
    _run_and_report(settings)


def _real_terrain_pipeline(
    out_dir: Path,
    extent_km: float,
    max_aoi_px: int,
    bounds: tuple[float, float, float, float] | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    """Run the REAL-terrain validation pipeline on a live LOLA south-pole DEM.

    Fetches a real LOLA south-polar DEM AOI via an O(1) windowed COG read, then
    runs the genuine terrain / illumination / planning algorithms on it (no
    synthetic topography). The radar ICE/CPR-DOP stage stays on the synthetic
    DFSAR field because real Chandrayaan-2 DFSAR is PRADAN-login-restricted — a
    limitation stated explicitly in the summary. Writes a report, figures and
    ``realdata_summary.json`` into ``out_dir`` and returns the summary dict.
    """
    import json
    import time

    import numpy as np

    from .constants import (
        MOON_OBLIQUITY_DEG,
        RHO_REGOLITH_GCM3,
        SOUTH_POLAR_STEREO_PROJ4,
        T_WATER_STABLE,
    )
    from .io.download import fetch_south_polar_dem, last_fetch_provenance
    from .planning.astar import astar
    from .planning.cost import build_cost_grid, traversability_mask
    from .planning.landing import (
        ahp_weights,
        landing_suitability,
        select_landing_sites,
    )
    from .planning.theta_star import theta_star
    from .terrain import dem as dem_mod
    from .terrain import illumination as illum_mod
    from .volume import estimate as vol_mod

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # -- Stage 1: FETCH real LOLA south-pole DEM (windowed COG read) ---------
    console.print(
        f"[cyan]Fetching real LOLA south-polar DEM AOI[/] "
        f"(extent {extent_km:g} km, <= {max_aoi_px}px) via /vsicurl/ …"
    )
    fetch_status = "ok"
    fetch_error: str | None = None
    try:
        dem, transform, crs = fetch_south_polar_dem(
            bounds=bounds,
            extent_km=extent_km,
            max_aoi_px=max_aoi_px,
            source_url=source_url,
            reproject_to=SOUTH_POLAR_STEREO_PROJ4,
        )
        prov = last_fetch_provenance()
    except Exception as e:  # genuine network failure after all fallbacks
        fetch_status = "network-unavailable"
        fetch_error = f"{type(e).__name__}: {e}"
        console.print(f"[red]Live DEM fetch failed:[/] {fetch_error}")
        summary = {
            "status": fetch_status,
            "error": fetch_error,
            "note": (
                "All remote LOLA south-polar DEM candidates failed (see "
                "lunaris.io.download.list_dem_candidates). Loader + CLI are "
                "committed; re-run with network egress to fetch real data."
            ),
        }
        (out_dir / "realdata_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    H, W = dem.shape
    eff_res = float(abs(transform.a))
    elev_min, elev_max = float(np.nanmin(dem)), float(np.nanmax(dem))
    elev_mean = float(np.nanmean(dem))
    console.print(
        f"[green]Fetched[/] {prov.get('source_label')} — "
        f"{H}x{W}px @ {eff_res:.1f} m  "
        f"(native {float(prov.get('native_res_m', eff_res)):.1f} m); "
        f"elev {elev_min:.0f}…{elev_max:.0f} m"
    )

    # -- Stage 2: TERRAIN derivatives on the REAL DEM -----------------------
    slope = dem_mod.slope_horn(dem, eff_res)
    baseline_px = max(2, min(H, W) // 16)
    roughness = dem_mod.rms_roughness(dem, baseline_px)
    baselines = sorted({max(1, baseline_px // 4), max(2, baseline_px // 2),
                        baseline_px, baseline_px * 2})
    hurst_H, _bl, _nu = dem_mod.hurst_exponent(dem, baselines)
    slope_mean = float(np.mean(slope))
    slope_median = float(np.median(slope))
    slope_p95 = float(np.percentile(slope, 95))

    # -- Stage 3: ILLUMINATION — honest horizon ray-march on real topography -
    console.print("[cyan]Ray-marching horizons on real topography "
                  "(PSR + sky-view factor)…[/]")
    psr_mask = illum_mod.permanent_shadow_mask(
        dem, eff_res, sun_elev_deg=MOON_OBLIQUITY_DEG, n_azimuth=48, step_px=1.5
    )
    svf = illum_mod.sky_view_factor(dem, eff_res, n_azimuth=48, step_px=1.5)
    psr_fraction = float(np.mean(psr_mask))
    # A pseudo-illumination proxy from the sky-view factor (1 = open sky) drives
    # the cost grid and landing illumination criterion on the real DEM.
    illumination = np.clip(svf, 0.0, 1.0)

    # -- Stage 4: LANDING suitability + selection on the real terrain -------
    traversable = traversability_mask(slope)
    # proximity-to-PSR criterion (science target = nearest cold/shadowed ground)
    from scipy import ndimage as ndi

    if psr_mask.any():
        dist_to_psr = ndi.distance_transform_edt(~psr_mask)
    else:
        dist_to_psr = np.full((H, W), float(max(H, W)))

    cost_grid = build_cost_grid(slope, roughness, illumination, traversable)
    drivable = traversable & np.isfinite(cost_grid)

    landing_layers = {
        "slope": slope,
        "illumination": illumination,
        "roughness": roughness,
        "distance_to_ice": dist_to_psr,
    }
    criteria = ["slope", "roughness", "illumination", "distance_to_ice"]
    pairwise = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1 / 2, 1.0, 2.0, 3.0],
        [1 / 3, 1 / 2, 1.0, 2.0],
        [1 / 4, 1 / 3, 1 / 2, 1.0],
    ])
    ahp_w, ahp_cr = ahp_weights(pairwise)
    weights = {name: float(w) for name, w in zip(criteria, ahp_w)}
    suitability = landing_suitability(landing_layers, weights)
    sites = select_landing_sites(suitability, n=5, traversable=drivable)
    if not sites:
        raise RuntimeError("no safe landing site found on real terrain")
    landing_r, landing_c, landing_score = sites[0]
    landing_site = (int(landing_r), int(landing_c))

    # Restrict the science target to the SAME connected drivable component as the
    # landing pad so an A*/Theta* traverse between them is guaranteed to exist
    # (the rover cannot teleport across an impassable crater wall). Pick, within
    # that component, the cell that comes closest to a permanent-shadow region.
    labels, _n_comp = ndi.label(drivable, structure=np.ones((3, 3)))
    land_label = int(labels[landing_site])
    operating_region = labels == land_label
    region_rc = np.argwhere(operating_region)
    region_dist = dist_to_psr[operating_region]
    target = tuple(int(v) for v in region_rc[int(np.argmin(region_dist))])

    # -- Stage 5: TRAVERSE on the real cost grid (A* + Theta*) --------------
    console.print("[cyan]Planning rover traverse on the real cost grid "
                  "(A* + Theta*)…[/]")
    path, path_cost = astar(cost_grid, landing_site, target, connectivity=8)
    theta_path, theta_cost = theta_star(cost_grid, landing_site, target)
    if path is not None and len(path) > 1:
        seg = np.diff(np.asarray(path, dtype=float), axis=0)
        path_len_px = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    else:
        path_len_px = 0.0
    traverse_len_m = path_len_px * eff_res
    if theta_path is not None and len(theta_path) > 1:
        tseg = np.diff(np.asarray(theta_path, dtype=float), axis=0)
        theta_len_m = float(np.sum(np.hypot(tseg[:, 0], tseg[:, 1]))) * eff_res
    else:
        theta_len_m = float("nan")

    # -- Stage 6: ICE VOLUME on the flagged cold/low-illumination region ----
    # Real DFSAR radar is PRADAN-login-restricted; we estimate a regolith
    # water-equivalent budget on the genuinely-derived PSR region using the
    # LCROSS gravimetric anchor (no synthetic radar needed for this number).
    psr_area_m2 = float(psr_mask.sum()) * eff_res * eff_res
    ice_depth_m = 5.0
    frac_wt = 0.056  # LCROSS mean 5.6 wt% water
    rho_bulk = RHO_REGOLITH_GCM3 * 1000.0
    psr_ice_mass_kg = frac_wt * rho_bulk * psr_area_m2 * ice_depth_m
    mc = vol_mod.monte_carlo_volume(
        area_m2=psr_area_m2, depth_m=ice_depth_m, frac=frac_wt, seed=7
    )
    psr_ice_volume_ci_m3 = [float(mc["ci"][0]), float(mc["ci"][1])]

    runtime_s = time.time() - t_start

    # -- Assemble the real-data summary -------------------------------------
    summary: dict[str, Any] = {
        "status": "ok",
        "target": "Lunar South Pole (real LOLA DEM AOI)",
        "source_label": prov.get("source_label"),
        "source_url": prov.get("source_url"),
        "source_kind": prov.get("source_kind"),
        "native_resolution_m": float(prov.get("native_res_m", eff_res)),
        "effective_resolution_m": eff_res,
        "aoi_bounds_proj_m": prov.get("aoi_bounds"),
        "crs": SOUTH_POLAR_STEREO_PROJ4,
        "grid": [int(H), int(W)],
        "windowed_read": (
            "O(1) windowed COG read via rasterio /vsicurl/ "
            "(overview-decimated AOI from a multi-GB global mosaic)"
        ),
        # terrain
        "elev_min_m": elev_min,
        "elev_max_m": elev_max,
        "elev_mean_m": elev_mean,
        "elev_range_m": elev_max - elev_min,
        "slope_mean_deg": slope_mean,
        "slope_median_deg": slope_median,
        "slope_p95_deg": slope_p95,
        "rms_roughness_mean_m": float(np.mean(roughness)),
        "hurst_exponent": float(hurst_H),
        # illumination
        "psr_fraction": psr_fraction,
        "psr_pixels": int(psr_mask.sum()),
        "sky_view_factor_mean": float(np.mean(svf)),
        "cold_trap_threshold_K": float(T_WATER_STABLE),
        # landing & traverse
        "landing_site_rc": [int(landing_r), int(landing_c)],
        "landing_score": float(landing_score),
        "ahp_consistency_ratio": float(ahp_cr),
        "science_target_rc": [int(target[0]), int(target[1])],
        "traverse_length_m": float(traverse_len_m),
        "traverse_length_km": float(traverse_len_m / 1000.0),
        "astar_cost": float(path_cost) if path is not None else None,
        "theta_star_length_m": theta_len_m,
        "theta_star_cost": float(theta_cost) if theta_path is not None else None,
        # ice budget on the real PSR region
        "psr_area_m2": psr_area_m2,
        "ice_depth_m": ice_depth_m,
        "psr_ice_mass_kg": float(psr_ice_mass_kg),
        "psr_ice_mass_t": float(psr_ice_mass_kg / 1000.0),
        "psr_ice_volume_ci_m3": psr_ice_volume_ci_m3,
        "dfsar_note": (
            "Real Chandrayaan-2 DFSAR L-band radar is PRADAN-login-restricted; "
            "the CPR/DOP ICE classification stage therefore stays on the "
            "synthetic DFSAR field. All terrain, illumination (horizon "
            "ray-march), landing and traverse results above are computed on the "
            "REAL LOLA DEM."
        ),
        "runtime_s": float(runtime_s),
    }

    # -- Figures + report ----------------------------------------------------
    _write_realdata_outputs(
        out_dir, dem, eff_res, slope, roughness, illumination, psr_mask, svf,
        suitability, cost_grid, path, theta_path, landing_site, target, summary,
    )

    (out_dir / "realdata_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_realdata_outputs(
    out_dir: Path,
    dem,
    res: float,
    slope,
    roughness,
    illumination,
    psr_mask,
    svf,
    suitability,
    cost_grid,
    path,
    theta_path,
    landing_site,
    target,
    summary: dict[str, Any],
) -> None:
    """Render figures + an HTML report for the real-terrain validation run."""
    import numpy as np

    from .viz import maps as maps_mod
    from .viz import report as report_mod

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dem = np.asarray(dem, dtype=np.float64)

    maps_mod.plot_layer(dem, title="Real LOLA DEM elevation [m]", cmap="terrain",
                        out=fig_dir / "dem.png")
    maps_mod.plot_layer(slope, title="Slope [deg] (Horn, real DEM)", cmap="magma",
                        out=fig_dir / "slope.png")
    maps_mod.plot_layer(np.asarray(svf), title="Sky-view factor (real horizons)",
                        cmap="cividis", out=fig_dir / "sky_view_factor.png")
    maps_mod.plot_ice_map(
        np.asarray(psr_mask),
        background=maps_mod.hillshade(dem, res=res),
        out=fig_dir / "psr_mask.png",
        title="Permanent shadow (PSR) on real terrain",
    )
    maps_mod.plot_layer(np.asarray(suitability), title="Landing suitability (AHP)",
                        cmap="viridis", out=fig_dir / "landing_suitability.png")
    try:
        maps_mod.plot_traverse(dem, path, landing_site, target,
                               out=fig_dir / "traverse.png", res=res)
    except TypeError:
        maps_mod.plot_traverse(dem, path, out=fig_dir / "traverse.png")

    # HTML report (tolerant of missing keys); reuse the real layers.
    results = {
        "resolution_m": res,
        "dem": dem,
        "ice_mask": np.asarray(psr_mask),
        "background": maps_mod.hillshade(dem, res=res),
        "slope": slope,
        "metrics": {
            "elev range (m)": summary["elev_range_m"],
            "mean slope (deg)": summary["slope_mean_deg"],
            "Hurst H": summary["hurst_exponent"],
            "PSR fraction": summary["psr_fraction"],
            "traverse (km)": summary["traverse_length_km"],
        },
        "landing": landing_site,
        "target": target,
        "path": path,
        "datasets": [
            (str(summary.get("source_label")), "Real LOLA south-pole DEM AOI",
             str(summary.get("source_url"))),
            ("Chandrayaan-2 DFSAR L-band", "CPR/DOP ice (synthetic — PRADAN "
             "login-restricted)", "ISRO PRADAN"),
        ],
        "meta": {
            "target": summary["target"],
            "grid": f"{summary['grid'][0]}x{summary['grid'][1]} @ {res:g} m",
            "ingest": "real-data (LOLA)",
            "source": str(summary.get("source_url")),
        },
    }
    report_mod.build_report(results, out_html=out_dir / "realdata_report.html")


@app.command()
def realdata(
    out: str = typer.Option("outputs/realdata", "--out",
                            help="Output directory."),
    extent_km: float = typer.Option(120.0, "--extent-km",
                                    help="Square AOI side length [km]."),
    max_aoi_px: int = typer.Option(300, "--max-aoi-px",
                                   help="Cap on AOI array side [px] (bounds the "
                                        "horizon ray-march runtime)."),
) -> None:
    """Validate the pipeline on a REAL LOLA south-polar DEM AOI (live fetch).

    Fetches a real, openly-downloadable LOLA south-pole DEM window via an O(1)
    ``/vsicurl/`` Cloud-Optimized-GeoTIFF read, then runs the genuine terrain,
    illumination (horizon ray-march), landing and traverse algorithms on that
    real topography, and writes a report + figures + ``realdata_summary.json``.
    The radar CPR/DOP ice stage stays synthetic (real DFSAR is PRADAN-restricted).
    """
    out_dir = Path(out)
    console.print(f"[bold cyan]lunaris[/] v{__version__} — REAL-data validation "
                  f"→ [dim]{out_dir}[/]")
    summary = _real_terrain_pipeline(out_dir, extent_km, max_aoi_px)

    if summary.get("status") != "ok":
        console.print(Panel.fit(
            f"[red]Live fetch unavailable[/] — wrote "
            f"{out_dir / 'realdata_summary.json'} (status: "
            f"{summary.get('status')}). Loader + CLI are committed.",
            border_style="red",
        ))
        return

    t = Table(title="Real LOLA south-pole terrain validation",
              show_lines=False, header_style="bold")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Source", str(summary["source_label"]))
    t.add_row("Native / eff. resolution",
              f"{summary['native_resolution_m']:.1f} / "
              f"{summary['effective_resolution_m']:.1f} m")
    t.add_row("AOI grid", f"{summary['grid'][0]}×{summary['grid'][1]}")
    t.add_row("Elevation min…max", f"{summary['elev_min_m']:.0f} … "
                                   f"{summary['elev_max_m']:.0f} m")
    t.add_row("Mean / median slope", f"{summary['slope_mean_deg']:.2f} / "
                                     f"{summary['slope_median_deg']:.2f}°")
    t.add_row("Hurst exponent", f"{summary['hurst_exponent']:.3f}")
    t.add_row("PSR fraction", f"{summary['psr_fraction']:.3f} "
                              f"({summary['psr_pixels']:,} px)")
    t.add_row("Sky-view factor (mean)", f"{summary['sky_view_factor_mean']:.3f}")
    t.add_row("Landing site (r,c)", f"{tuple(summary['landing_site_rc'])}")
    t.add_row("Traverse length", f"{summary['traverse_length_m']:.1f} m "
                                 f"({summary['traverse_length_km']:.3f} km)")
    t.add_row("PSR ice budget (5 m)", f"{summary['psr_ice_mass_t']:.1f} t")
    t.add_row("Runtime", f"{summary['runtime_s']:.1f} s")
    console.print(t)
    console.print(
        f"\n[green]✓ real-data outputs written[/] → "
        f"[bold]{out_dir / 'realdata_report.html'}[/]\n"
        f"  • report HTML, figures/*.png, realdata_summary.json\n"
        f"  • source: [dim]{summary['source_url']}[/]"
    )


@app.command()
def version() -> None:
    """Print the lunaris version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
