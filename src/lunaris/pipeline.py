"""End-to-end orchestration of the lunaris ice-detection & planning pipeline.

Implemented by: **integration agent**.

The pipeline runs nine stages, each delegating to a sub-package:

1. **Ingest** — load real Chandrayaan-2 DFSAR/OHRC granules (``lunaris.io``) or,
   absent data, the deterministic synthetic Faustini scene
   (:func:`lunaris.io.synthetic.generate_faustini_scene`).
2. **Stokes / CPR / DOP** — compute the circular polarisation ratio and degree
   of polarisation from the scene Stokes vectors (``lunaris.polarimetry``), and
   demonstrate :func:`~lunaris.polarimetry.stokes.stokes_from_circular` on a
   synthesised circular field.
3. **m-chi decomposition** — split surface / double-bounce / volume scatter to
   isolate the ice volume-scatter channel (``lunaris.polarimetry.decomposition``).
4. **Ice LUT classification** — apply the O(1) precomputed ``CPR>1 & DOP<0.13``
   LUT classifier and assert it is bit-identical to the direct threshold rule
   (``lunaris.classify.ice_lut``).
5. **Terrain** — slope, roughness, illumination / shadow, and thermal cold-trap
   analysis (``lunaris.terrain``).
6. **Multi-sensor fusion** — combine radar, thermal, neutron and optical-frost
   evidence (random-forest + Dempster-Shafer / Bayesian) into an ice posterior
   (``lunaris.classify.fusion`` / ``lunaris.classify.evidence``).
7. **Landing** — AHP-weighted landing-site suitability and selection
   (``lunaris.planning.landing``).
8. **Traverse** — energy-aware rover path from the landing site to the ice
   access point (``lunaris.planning``: A* / Theta* + ``energy``).
9. **Volume + report** — dielectric-mixing ice fraction, top-5 m Monte-Carlo
   ice volume/mass (``lunaris.volume``), then the HTML report (``lunaris.viz``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings

__all__ = ["run_pipeline"]


# Datasets table shown in the report's methods section (name, role, source).
_DATASETS: list[tuple[str, str, str]] = [
    ("Chandrayaan-2 DFSAR L-band", "Subsurface ice CPR/DOP", "ISRO PRADAN"),
    ("Mini-RF / Mini-SAR S-band", "Cross-band CPR confirmation", "LRO / Chandrayaan-1"),
    ("LRO Diviner T_max", "Cold-trap (<110 K) detection", "NASA PDS"),
    ("LOLA DEM (south pole)", "Slope / roughness / shadow", "NASA PDS"),
    ("LRO LAMP off/on ratio", "Surface-frost UV proxy", "NASA PDS"),
    ("LOLA 1064 nm albedo", "Surface-frost optical proxy", "NASA PDS"),
    ("LCROSS ejecta abundance", "Gravimetric ice cross-check", "Colaprete et al. 2010"),
    ("M3 / OHRC imagery", "Optical context", "ISRO / NASA"),
]


def _jsonify(obj: Any) -> Any:
    """Recursively coerce numpy / path values into JSON-serialisable types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _precision_recall(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall and F1 of a boolean prediction vs ground truth."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_pipeline(config: Settings | None = None, **overrides: Any) -> dict[str, Any]:
    """Run the full nine-stage pipeline and return a results dictionary.

    Parameters
    ----------
    config : Settings, optional
        Resolved configuration (see :func:`lunaris.config.load_config`). When
        ``None`` a default :class:`~lunaris.config.Settings` is built and any
        keyword ``overrides`` (e.g. ``grid_size``, ``seed``, ``outputs_dir``)
        are applied on top.
    **overrides
        Convenience keyword overrides applied to ``config`` (handy for tests /
        the CLI, e.g. ``run_pipeline(grid_size=64)``).

    Returns
    -------
    dict
        Aggregated results: scene, ice masks, terrain layers, fusion posterior,
        landing site, traverse path, ice-volume statistics, figure paths, and a
        JSON-serialisable ``summary`` sub-dict.
    """
    # Lazy module imports keep ``import lunaris`` cheap and isolate heavy deps.
    from .classify import evidence as ev
    from .classify import fusion as fus
    from .classify import ice_lut as lut
    from .io.synthetic import generate_faustini_scene
    from .planning.astar import astar
    from .planning.cost import build_cost_grid, traversability_mask
    from .planning.energy import energy_aware_plan
    from .planning.landing import (
        ahp_weights,
        landing_suitability,
        select_landing_sites,
    )
    from .planning.theta_star import theta_star
    from .polarimetry import cpr as cpr_mod
    from .polarimetry import decomposition as decomp
    from .polarimetry import stokes as stokes_mod
    from .terrain import dem as dem_mod
    from .terrain import thermal as thermal_mod
    from .volume import estimate as vol_mod

    if config is None:
        config = Settings(**overrides)
    elif overrides:
        config = config.model_copy(update=overrides)

    n = int(config.grid_size)
    seed = int(config.seed)
    res = float(config.resolution_m)
    out_dir = Path(config.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Stage 1: INGEST -----------------------------------------------------
    # Real-data ingestion via ``lunaris.io.readers`` is wired for granules under
    # ``config.raw_dir``; absent those, fall back to the deterministic synthetic
    # Faustini scene (the default, fully reproducible from ``seed``).
    raw_dir = Path(config.raw_dir)
    use_real = (not config.use_synthetic) and raw_dir.exists() and any(raw_dir.glob("*.tif"))
    if use_real:  # pragma: no cover - exercised only with real granules present
        from .scene import LunarScene

        scene = LunarScene.load(raw_dir)
        ingest_source = "real-data"
    else:
        scene = generate_faustini_scene(n=n, resolution_m=res, seed=seed)
        ingest_source = "synthetic"
    scene.meta.setdefault("target", config.target_name)
    H, W = scene.shape

    # -- Stage 2: POLARIMETRY (Stokes -> CPR / DOP) --------------------------
    cpr = cpr_mod.circular_polarization_ratio(scene.s0_L, scene.s3_L)
    dop = cpr_mod.degree_of_polarization(scene.s0_L, scene.s1_L, scene.s2_L, scene.s3_L)
    sc, oc = cpr_mod.sc_oc_power(scene.s0_L, scene.s3_L)

    # Demonstrate the full hybrid-polarity Stokes synthesis from circular fields:
    # build RH/RV receive fields whose multilooked Stokes reproduce the scene's
    # same-sense ice signature, then recover (s0..s3) -> CPR/DOP off the synth.
    rng = np.random.default_rng(seed + 1)
    amp = np.sqrt(np.clip(scene.s0_L, 0.0, None))
    phase = rng.uniform(0.0, 2 * np.pi, size=scene.s0_L.shape)
    e_rh = amp * np.exp(1j * phase)
    # opposite-sense fraction from OC/SC keeps the synthetic field ice-consistent
    frac_rv = np.sqrt(np.clip(oc / np.clip(sc + oc, 1e-9, None), 0.0, 1.0))
    e_rv = amp * frac_rv * np.exp(1j * (phase + rng.uniform(0.0, 2 * np.pi, size=phase.shape)))
    s0c, s1c, s2c, s3c = stokes_mod.stokes_from_circular(e_rh, e_rv, window=5)
    cpr_synth = cpr_mod.circular_polarization_ratio(s0c, s3c)

    # -- Stage 3: m-chi DECOMPOSITION ---------------------------------------
    mchi_even, mchi_vol, mchi_odd = decomp.m_chi(
        scene.s0_L, scene.s1_L, scene.s2_L, scene.s3_L
    )

    # -- Stage 4: ICE LUT CLASSIFICATION (O(1) path) ------------------------
    cpr_edges, dop_edges = lut.default_edges()
    ice_lut_table = lut.build_ice_lut(cpr_edges, dop_edges)
    ice_mask = lut.classify_ice_lut(cpr, dop, ice_lut_table, cpr_edges, dop_edges)
    ice_mask_thr = lut.classify_ice_threshold(
        cpr, dop, config.cpr_threshold, config.dop_threshold
    )
    # The precomputed table must be a bit-exact stand-in for the direct rule.
    lut_matches_rule = bool(np.array_equal(ice_mask, ice_mask_thr))
    assert lut_matches_rule, "O(1) LUT classifier diverged from the direct rule"

    precision, recall, f1 = _precision_recall(ice_mask, scene.ice_truth)

    # -- Stage 5: TERRAIN ----------------------------------------------------
    slope = dem_mod.slope_horn(scene.dem, res)
    baseline_px = max(2, min(H, W) // 32)
    roughness = dem_mod.rms_roughness(scene.dem, baseline_px)
    baselines = sorted({max(1, baseline_px // 2), baseline_px, baseline_px * 2,
                        baseline_px * 4})
    hurst_H, _hurst_bl, _hurst_nu = dem_mod.hurst_exponent(scene.dem, baselines)
    cold_trap = thermal_mod.cold_trap_mask(scene.temperature_max)

    # Illumination: use the scene's pre-computed annual-average field for the
    # demo (avoids the expensive horizon ray-march). For real DEMs the
    # platform derives the same field with
    # ``lunaris.terrain.illumination.permanent_shadow_mask`` / ``horizon_map``;
    # we only run that ground-truthing ray-march on small scenes to keep the
    # demo runtime bounded.
    illumination = np.asarray(scene.illumination, dtype=np.float64)
    psr_mask: np.ndarray | None = None
    if n <= 128:
        from .terrain import illumination as illum_mod

        psr_mask = illum_mod.permanent_shadow_mask(
            scene.dem, res, n_azimuth=48, step_px=1.5
        )

    # -- Stage 6: MULTI-SENSOR FUSION ---------------------------------------
    X, feat_names = fus.build_feature_stack(scene)
    y = np.asarray(scene.ice_truth, dtype=bool).ravel()
    rf_model = fus.train_ice_classifier(X, y, random_state=seed)
    rf_labels, rf_proba = fus.predict_ice(rf_model, X, shape=(H, W))
    importances = fus.feature_importance(rf_model, feat_names)
    rf_precision, rf_recall, rf_f1 = _precision_recall(rf_labels, scene.ice_truth)

    # 5-fold cross-validated F1 of the random-forest fusion classifier.
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        cv_scores = cross_val_score(
            RandomForestClassifier(
                n_estimators=120, class_weight="balanced",
                random_state=seed, n_jobs=-1,
            ),
            X, y, cv=3, scoring="f1",
        )
        cv_f1_mean = float(np.mean(cv_scores))
        cv_f1_std = float(np.std(cv_scores))
    except Exception:  # pragma: no cover - sklearn always present here
        cv_f1_mean = cv_f1_std = float("nan")

    confidence = ev.multi_evidence_ice(scene)

    # -- Stage 7: LANDING ----------------------------------------------------
    # Geodesic distance (px) from the ice mask -> proximity-to-ice criterion.
    from scipy import ndimage as ndi

    if ice_mask.any():
        dist_to_ice = ndi.distance_transform_edt(~ice_mask)
    else:  # pragma: no cover - synthetic scene always has ice
        dist_to_ice = np.full((H, W), float(max(H, W)))

    traversable = traversability_mask(slope)
    cost_grid = build_cost_grid(slope, roughness, illumination, traversable)
    drivable = traversable & np.isfinite(cost_grid)

    # The ice patch floor is typically ringed by >20 deg walls, so the rover can
    # never stand *on* the ice. Both the landing pad and the sampling approach
    # must therefore live in one connected drivable region. Label the drivable
    # cells (8-connectivity, matching the planner) and pick the **largest
    # component that comes closest to the ice** as the operating region — this
    # guarantees the landing site and the ice-access goal are mutually
    # reachable.
    labels, n_comp = ndi.label(drivable, structure=np.ones((3, 3)))
    ice_rows, ice_cols = np.nonzero(ice_mask)
    if ice_rows.size:
        ice_centroid = (float(ice_rows.mean()), float(ice_cols.mean()))
    else:  # pragma: no cover
        ice_centroid = (H / 2.0, W / 2.0)

    # Rank candidate components by (min distance to ice, then size) and keep the
    # best sizeable one: closest approach to the ice that is still a real region.
    min_region_px = max(16, (H * W) // 400)
    best_comp = 0
    best_key: tuple[float, int] | None = None
    for comp in range(1, n_comp + 1):
        comp_mask = labels == comp
        size = int(comp_mask.sum())
        if size < min_region_px:
            continue
        comp_min_dist = float(dist_to_ice[comp_mask].min())
        key = (comp_min_dist, -size)
        if best_key is None or key < best_key:
            best_key = key
            best_comp = comp
    if best_comp == 0:  # pragma: no cover - drivable region always non-trivial
        best_comp = int(np.argmax(np.bincount(labels.ravel())[1:]) + 1)
    operating_region = labels == best_comp

    # Ice access / sampling point: the operating-region cell that gets closest to
    # the ice mask (overlooking the patch from drivable ground).
    region_rc = np.argwhere(operating_region)
    region_dist = dist_to_ice[operating_region]
    ice_access = tuple(int(v) for v in region_rc[int(np.argmin(region_dist))])

    landing_layers = {
        "slope": slope,
        "illumination": illumination,
        "earth_visibility": np.asarray(scene.earth_visibility, dtype=np.float64),
        "roughness": roughness,
        "distance_to_ice": dist_to_ice,
    }
    # AHP pairwise judgements (safety-first): slope > roughness > illumination
    # > earth-visibility > proximity-to-ice. Order matches ``criteria`` below.
    criteria = ["slope", "roughness", "illumination", "earth_visibility",
                "distance_to_ice"]
    pairwise = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1 / 2, 1.0, 2.0, 3.0, 4.0],
        [1 / 3, 1 / 2, 1.0, 2.0, 3.0],
        [1 / 4, 1 / 3, 1 / 2, 1.0, 2.0],
        [1 / 5, 1 / 4, 1 / 3, 1 / 2, 1.0],
    ])
    ahp_w, ahp_cr = ahp_weights(pairwise)
    weights = {name: float(w) for name, w in zip(criteria, ahp_w)}
    suitability = landing_suitability(landing_layers, weights)
    # Restrict landing candidates to the operating region so the chosen pad can
    # actually reach the sampling point.
    sites = select_landing_sites(suitability, n=5, traversable=operating_region)
    if not sites:  # pragma: no cover - operating region always has a max
        raise RuntimeError("no safe landing site found")
    landing_r, landing_c, landing_score = sites[0]
    landing_site = (int(landing_r), int(landing_c))

    # -- Stage 8: TRAVERSE ---------------------------------------------------
    path, path_cost = astar(cost_grid, landing_site, ice_access, connectivity=8)
    theta_path, theta_cost = theta_star(cost_grid, landing_site, ice_access)

    if path is not None:
        seg = np.diff(np.asarray(path, dtype=float), axis=0)
        path_len_px = float(np.sum(np.hypot(seg[:, 0], seg[:, 1]))) if seg.size else 0.0
    else:  # pragma: no cover - access point is chosen traversable & connected
        path_len_px = float("inf")
    traverse_len_m = path_len_px * res

    energy = energy_aware_plan(
        cost_grid, illumination, landing_site, ice_access, resolution_m=res,
        slope=slope,
    )

    # -- Stage 9a: VOLUME ----------------------------------------------------
    vol = vol_mod.estimate_scene_ice(scene, ice_mask=ice_mask, depth_m=5.0)
    mc = vol["mc"]
    vol_ci = (float(mc["ci"][0]), float(mc["ci"][1]))

    # ---- Assemble the results bundle --------------------------------------
    summary: dict[str, Any] = {
        "target": config.target_name,
        "ingest_source": ingest_source,
        "grid": [int(H), int(W)],
        "resolution_m": res,
        "seed": seed,
        # detection
        "lut_matches_rule": lut_matches_rule,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ice_pixels": int(ice_mask.sum()),
        "ice_truth_pixels": int(np.asarray(scene.ice_truth).sum()),
        # fusion
        "rf_precision": float(rf_precision),
        "rf_recall": float(rf_recall),
        "rf_f1": float(rf_f1),
        "rf_cv_f1_mean": cv_f1_mean,
        "rf_cv_f1_std": cv_f1_std,
        "top_feature": importances[0][0] if importances else None,
        "feature_importance": [[k, float(v)] for k, v in importances],
        # terrain
        "hurst_exponent": float(hurst_H),
        "cold_trap_pixels": int(np.asarray(cold_trap).sum()),
        "median_slope_deg": float(np.median(slope)),
        # landing
        "landing_site": [int(landing_r), int(landing_c)],
        "landing_score": float(landing_score),
        "ahp_consistency_ratio": float(ahp_cr),
        # traverse
        "ice_access": [int(ice_access[0]), int(ice_access[1])],
        "traverse_length_m": float(traverse_len_m),
        "traverse_length_km": float(traverse_len_m / 1000.0),
        "astar_cost": float(path_cost) if path is not None else None,
        "theta_star_cost": float(theta_cost) if theta_path is not None else None,
        "energy_feasible": bool(energy["feasible"]),
        "dark_hours": (float(energy["dark_hours"])
                       if np.isfinite(energy["dark_hours"]) else None),
        "min_soc": float(energy["min_soc"]),
        "energy_Wh": (float(energy["energy_Wh"])
                      if np.isfinite(energy["energy_Wh"]) else None),
        # volume
        "ice_area_m2": float(vol["area_m2"]),
        "ice_fraction": float(vol["ice_fraction"]),
        "ice_depth_m": float(vol["depth_m"]),
        "ice_volume_m3": float(vol["volume_m3"]),
        "ice_mass_kg": float(vol["mass_kg"]),
        "ice_mass_t": float(vol["mass_kg"] / 1000.0),
        "ice_volume_ci_m3": [vol_ci[0], vol_ci[1]],
        "ice_mass_ci_t": [float(vol_ci[0] * 920.0 / 1000.0),
                          float(vol_ci[1] * 920.0 / 1000.0)],
        "lcross_cross_check_kg": float(vol["lcross_cross_check_kg"]),
    }

    results: dict[str, Any] = {
        # core scene + geo
        "scene": scene,
        "resolution_m": res,
        "dem": scene.dem,
        # polarimetry
        "cpr": cpr,
        "dop": dop,
        "cpr_synth": cpr_synth,
        "sc": sc,
        "oc": oc,
        "mchi": (mchi_even, mchi_vol, mchi_odd),
        # classification
        "ice_mask": ice_mask,
        "ice_mask_threshold": ice_mask_thr,
        "ice_lut": ice_lut_table,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        # fusion
        "rf_labels": rf_labels,
        "rf_proba": rf_proba,
        "confidence": confidence,
        "feature_importance": ([k for k, _ in importances],
                               [v for _, v in importances]),
        # terrain
        "slope": slope,
        "roughness": roughness,
        "hurst_exponent": float(hurst_H),
        "cold_trap": cold_trap,
        "illumination": illumination,
        "psr_mask": psr_mask,
        # landing
        "suitability": suitability,
        "landing_sites": sites,
        "landing_site": landing_site,
        "landing": landing_site,
        # traverse
        "cost_grid": cost_grid,
        "traverse_path": path,
        "path": path,
        "theta_path": theta_path,
        "ice_access": ice_access,
        "target": ice_access,
        "traverse_length_m": float(traverse_len_m),
        "energy": energy,
        # volume
        "volume": vol,
        "volume_m3": float(vol["volume_m3"]),
        "mass_kg": float(vol["mass_kg"]),
        "volume_samples": mc["samples"],
        "volume_ci": vol_ci,
        "volume_mean": float(mc["mean"]),
        # metadata
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(f1),
            "RF CV-F1": cv_f1_mean,
            "ice px": int(ice_mask.sum()),
            "ice area (km²)": float(vol["area_m2"] / 1e6),
        },
        "datasets": _DATASETS,
        "meta": {
            "target": config.target_name,
            "grid": f"{H}×{W} @ {res:g} m",
            "ingest": ingest_source,
            "seed": seed,
        },
        "summary": summary,
    }

    # -- Stage 9b: REPORT + figures + GeoTIFFs ------------------------------
    _write_outputs(results, out_dir)

    return results


def _write_outputs(results: dict[str, Any], out_dir: Path) -> None:
    """Render the HTML report, key figures, summary JSON and COG GeoTIFFs."""
    from .viz import charts as charts_mod
    from .viz import maps as maps_mod
    from .viz import report as report_mod
    from .viz import terrain3d as t3d_mod

    out_dir.mkdir(parents=True, exist_ok=True)
    scene = results["scene"]
    res = float(results["resolution_m"])

    # 3-D terrain (interactive, self-contained) — link it from the report.
    t3d_path = out_dir / "terrain3d.html"
    t3d_mod.terrain3d_html(
        scene.dem,
        overlay=results["confidence"],
        path=results["traverse_path"],
        landing=results["landing_site"],
        target=results["ice_access"],
        out=t3d_path,
        res=res,
    )
    results["terrain3d_html"] = str(t3d_path)

    # HTML report (all 5 deliverables, offline-renderable).
    report_path = out_dir / "lunaris_report.html"
    report_mod.build_report(results, out_html=report_path)
    results["report_path"] = str(report_path)

    # Static key figures (PNG / interactive HTML charts).
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        bg = maps_mod.hillshade(np.asarray(scene.dem), res=res)
        maps_mod.plot_ice_map(results["ice_mask"], background=bg,
                              out=fig_dir / "ice_map.png")
        maps_mod.plot_mchi(*results["mchi"], out=fig_dir / "mchi_rgb.png")
        maps_mod.plot_layer(results["slope"], title="Slope [deg]", cmap="magma",
                            out=fig_dir / "slope.png")
        maps_mod.plot_layer(results["confidence"], title="Ice confidence",
                            cmap="cividis", out=fig_dir / "confidence.png")
        maps_mod.plot_traverse(
            scene.dem, results["traverse_path"], results["landing_site"],
            results["ice_access"], out=fig_dir / "traverse.png", res=res,
        )
    except TypeError:
        # plot_traverse signature variants: fall back to positional dem+path.
        maps_mod.plot_traverse(scene.dem, results["traverse_path"],
                               out=fig_dir / "traverse.png")
    charts_mod.cpr_dop_scatter(results["cpr"], results["dop"],
                               ice_mask=results["ice_mask"],
                               out=fig_dir / "cpr_dop_scatter.html")
    charts_mod.volume_histogram(results["volume_samples"], ci=results["volume_ci"],
                                out=fig_dir / "volume_hist.html")
    charts_mod.feature_importance_bar(*results["feature_importance"],
                                      out=fig_dir / "feature_importance.html")

    # COG GeoTIFFs of the headline rasters (GIS-loadable).
    _save_geotiffs(scene, results, out_dir)

    # Headline numbers as JSON (no numpy arrays).
    summary_path = out_dir / "results_summary.json"
    summary_path.write_text(json.dumps(_jsonify(results["summary"]), indent=2))
    results["summary_path"] = str(summary_path)


def _save_geotiffs(scene: Any, results: dict[str, Any], out_dir: Path) -> None:
    """Write ice mask / slope / CPR as COG GeoTIFFs into ``out_dir``."""
    try:
        import rasterio
        from rasterio.crs import CRS
    except Exception:  # pragma: no cover - rasterio is a hard dependency
        return

    try:
        rio_crs = CRS.from_string(scene.crs)
    except Exception:  # pragma: no cover
        rio_crs = CRS.from_proj4(scene.crs)

    h, w = scene.shape
    layers = {
        "ice_mask": (np.asarray(results["ice_mask"]).astype("uint8"), "uint8"),
        "slope": (np.asarray(results["slope"]).astype("float32"), "float32"),
        "cpr": (np.asarray(results["cpr"]).astype("float32"), "float32"),
    }
    for name, (data, dtype) in layers.items():
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": dtype,
            "crs": rio_crs,
            "transform": scene.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
            "interleave": "band",
        }
        path = out_dir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
