# lunaris — System Architecture

**Lunar Subsurface-Ice Detection & Mission-Planning Platform**
ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 8

> Detect and characterize subsurface water-ice in a lunar South-Pole *doubly
> shadowed crater* (Faustini PSR) from Chandrayaan-2 DFSAR radar polarimetry,
> fuse it with 30+ multi-sensor datasets, then deliver a landing site, an
> energy-aware rover traverse, and a top-5 m ice-volume estimate.

---

## Table of contents

1. [Executive summary & the five PS8 deliverables](#1-executive-summary--the-five-ps8-deliverables)
2. [System architecture — the nine-stage pipeline](#2-system-architecture--the-nine-stage-pipeline)
3. [Module map](#3-module-map)
4. [The "fastest platform / O(1)" engineering](#4-the-fastest-platform--o1-engineering)
5. [The science, with equations](#5-the-science-with-equations)
6. [Multi-sensor data-fusion strategy](#6-multi-sensor-data-fusion-strategy)
7. [Data flow & formats](#7-data-flow--formats)
8. [Tech stack & dependencies](#8-tech-stack--dependencies)
9. [Reproducibility, testing, performance, limitations & future work](#9-reproducibility-testing-performance-limitations--future-work)
10. [References](#10-references)

---

## 1. Executive summary & the five PS8 deliverables

`lunaris` is a `src`-layout Python package that turns Chandrayaan-2 Dual-Frequency
Synthetic Aperture Radar (DFSAR) observations of a lunar South-Pole doubly
shadowed crater into a complete, defensible exploration plan. It is engineered as
a **single, reproducible, offline-capable pipeline**: from a YAML config it
ingests real PDS4/COG granules (or a deterministic synthetic Faustini scene when
no data are present), computes the polarimetric subsurface-ice criterion of
**Sinha et al. (2026)** — `CPR > 1 ∧ DOP < 0.13` — hardens that detection with
30+ independent multi-sensor datasets, and emits a self-contained HTML report
with every map, figure, number and uncertainty bound a reviewer needs.

Two design commitments distinguish the platform:

* **Scientific rigour over hand-waving.** Every numeric constant lives in one
  audited file (`lunaris.constants`); every algorithm carries its governing
  equation and a peer-reviewed citation in its docstring; and we are explicit
  about what radar polarimetry *can* and *cannot* retrieve (see §5 and §9 — we
  report a *relative ice-likelihood index*, not an absolute ice weight-percent,
  because the ice/regolith dielectric contrast is weak).
* **"Fastest platform" engineering that is honest.** We separate genuine `O(1)`
  operations (the precomputed `(CPR, DOP)` LUT classifier; COG windowed reads;
  H3 bitwise indexing; `np.memmap` page access; joblib cache hits) from
  `O(log n)` spatial search and from mere constant-factor vectorisation. The
  precision of that claim (§4) is itself a differentiator.

### The five PS8 deliverables and how lunaris produces each

| # | PS8 deliverable | lunaris stage(s) | Primary module(s) | Key method |
|---|-----------------|------------------|-------------------|------------|
| **D1** | **Identify & map subsurface-ice regions** in the doubly shadowed crater | 2-4, 6 | `polarimetry`, `classify` | Stokes → CPR/DOP → m-χ volume → **O(1) LUT** `CPR>1 ∧ DOP<0.13` → Bayesian/Dempster-Shafer fusion |
| **D2** | A **validated radar-based detection framework** | 2-4, 6 | `polarimetry`, `classify` | m-χ/m-δ decomposition, Cloude-Pottier H/A/α, co-pol coherence, multi-sensor cross-verification to defeat roughness false positives |
| **D3** | A **feasible landing site** near the target | 5, 7 | `terrain`, `planning.landing` | slope/roughness/illumination/Earth-visibility criteria → AHP-weighted MCDA → top-`n` site selection |
| **D4** | An **optimized rover traverse** (terrain hazards + solar power) | 5, 8 | `terrain`, `planning` | cost grid → A\* / D\* Lite / Theta\* / RRT\* / NSGA-II → energy-aware `⟨cell, SOC⟩` planner with 70-h shadow survival |
| **D5** | A **top-5 m ice-volume estimate** | 9 | `volume` | Looyenga dielectric-mixing inversion → `V = A·d·f` → Monte-Carlo uncertainty (mean, σ, 95% CI) |

All five are assembled by `lunaris.pipeline.run_pipeline(config)` and rendered by
`lunaris.viz.report.build_report` into one shareable HTML artefact.

---

## 2. System architecture — the nine-stage pipeline

`lunaris` is a linear, side-effect-light pipeline. Each stage consumes the shared
`LunarScene` dataclass (co-registered raster layers + geo-metadata) and either
attaches derived layers to `scene.meta` or returns plain NumPy arrays. The
orchestrator is `lunaris.pipeline.run_pipeline`.

```mermaid
flowchart TD
    A["① INGEST<br/>lunaris.io<br/>PDS4/COG granules or<br/>deterministic synthetic Faustini"] --> B
    B["② DFSAR POLARIMETRY<br/>lunaris.polarimetry<br/>scattering matrix → circular basis<br/>Stokes S0..S3 → CPR, DOP"] --> C
    C["③ m-χ / m-δ DECOMPOSITION<br/>lunaris.polarimetry.decomposition<br/>even / volume / odd power split<br/>+ Cloude-Pottier H/A/α"] --> D
    D["④ O(1) LUT ICE CLASSIFY<br/>lunaris.classify.ice_lut<br/>precomputed (CPR,DOP) boolean table<br/>digitize + single fancy-index gather"] --> E
    E["⑤ TERRAIN<br/>lunaris.terrain<br/>slope · roughness · Hurst<br/>illumination/shadow · thermal cold-trap"] --> F
    F["⑥ MULTI-SENSOR FUSION<br/>lunaris.classify.fusion / evidence<br/>radar ∧ thermal ∧ neutron ∧ UV/IR ∧ albedo<br/>RandomForest · Bayesian · Dempster-Shafer"] --> G
    G["⑦ LANDING-SITE MCDA<br/>lunaris.planning.landing<br/>AHP weights → WLC suitability<br/>→ top-n site selection"] --> H
    H["⑧ ENERGY-AWARE TRAVERSE<br/>lunaris.planning<br/>cost grid → A*/D*Lite/Theta*/RRT*/NSGA-II<br/>⟨cell,SOC⟩ energy model · 70-h survival"] --> I
    I["⑨ ICE VOLUME + UNCERTAINTY<br/>lunaris.volume<br/>Looyenga f → V = A·d·f<br/>Monte-Carlo CI"] --> J
    J["HTML REPORT<br/>lunaris.viz.report<br/>maps · CPR-DOP scatter · m-χ RGB<br/>3-D terrain · traverse · volume PDF"]

    subgraph EV["fusion evidence inputs"]
      direction LR
      D -. "radar criterion" .-> F
      E -. "thermal / illumination" .-> F
      A -. "neutron · UV · albedo · IIRS" .-> F
    end
```

### ASCII fallback (the same nine stages, end to end)

```
  CONFIG (YAML) ─► lunaris.config.Settings
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ ① INGEST            io.readers (COG windowed) / io.synthetic (fallback)    │
 │                     → LunarScene{dem, s0..s3 (L,S), thermal, illum, ...}   │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ② POLARIMETRY       polarimetry.stokes  : S = [[Shh,Shv],[Svh,Svv]]       │
 │                     → circular E_RH,E_RV → Stokes S0,S1,S2,S3 (multilook) │
 │                     polarimetry.cpr      : CPR=(S0-S3)/(S0+S3); DOP=m      │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ③ DECOMPOSITION     polarimetry.decomposition : m-χ (even/vol/odd),       │
 │                     m-δ (double/vol/surface), CHILD, Cloude-Pottier H/A/α  │
 │                     polarimetry.speckle / coherence (refined-Lee, γ_hhvv)  │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ④ ICE LUT           classify.ice_lut : build_ice_lut() once,              │
 │   (TRUE O(1)/px)    classify_ice_lut() = digitize → lut[ci,di] gather      │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ⑤ TERRAIN           terrain.dem   : Horn slope/aspect, deviogram roughness,│
 │                     Hurst H, IQR roughness                                 │
 │                     terrain.illumination : horizon, PSR, double-shadow,    │
 │                       Earth-visibility, sky-view factor                    │
 │                     terrain.thermal : cold-trap, sublimation, stability    │
 │                     terrain.boulders : shadow-length boulder detection     │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ⑥ FUSION            classify.fusion   : RandomForest (10-feature stack)    │
 │                     classify.evidence : Bayesian log-odds, Dempster-Shafer │
 │                     radar ∧ thermal ∧ neutron ∧ UV/IR ∧ albedo → posterior │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ⑦ LANDING           planning.landing : AHP weights → WLC suitability →     │
 │                       select top-n non-adjacent safe sites                 │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ⑧ TRAVERSE          planning.cost   : weighted hazard cost grid           │
 │                     planning.{astar,dstar_lite,theta_star,rrt_star,nsga2}  │
 │                     planning.energy : solar/battery/SOC, 70-h survival     │
 ├──────────────────────────────────────────────────────────────────────────┤
 │ ⑨ VOLUME            volume.dielectric : Looyenga f(ε_eff), penetration δ   │
 │                     volume.estimate   : V=A·d·f, Monte-Carlo CI            │
 ├──────────────────────────────────────────────────────────────────────────┤
 │    REPORT           viz.{maps,charts,terrain3d,report} → outputs/*.html    │
 └──────────────────────────────────────────────────────────────────────────┘
```

### The shared data structure: `LunarScene`

Every stage operates on one in-memory object, `lunaris.scene.LunarScene`
(`@dataclass`), bundling co-registered float/bool raster layers with their
geo-spatial metadata (rasterio `Affine` transform, CRS string, pixel resolution,
free-form `meta` bag). The canonical layer set is:

| Group | Layers |
|-------|--------|
| Topography | `dem` |
| L-band Stokes | `s0_L, s1_L, s2_L, s3_L` |
| S-band Stokes | `s0_S, s1_S, s2_S, s3_S` |
| Radar products | `cpr_L, dop_L, cpr_S, dop_S` |
| Thermal | `temperature_max` |
| Illumination | `illumination`, `earth_visibility` |
| Optical frost proxies | `albedo_1064`, `lamp_ratio` |
| Ground truth | `ice_truth` (bool) |

`LunarScene.save(dir)` writes every layer as a **Cloud-Optimized GeoTIFF** (COG;
tiled 256×256, deflate-compressed, with `[2,4,8]` overviews) in the lunar
south-polar stereographic CRS plus a `scene.json` sidecar, so all artefacts open
directly in QGIS/ArcGIS. `LunarScene.load(dir)` reconstructs it. The module
asserts at import that the dataclass fields cover every declared layer — a single
structural invariant guarding the whole pipeline.

---

## 3. Module map

`lunaris` is organised into seven sub-packages under `src/lunaris/`, plus the
top-level `constants`, `config`, `scene`, `pipeline`, and `cli` modules.

```
src/lunaris/
├── constants.py        single source of truth for all physical/instrument constants
├── config.py           pydantic-settings Settings + load_config(yaml)
├── scene.py            LunarScene dataclass (+ COG save/load), LAYER_NAMES
├── pipeline.py         run_pipeline(config) — nine-stage orchestrator
├── cli.py              Typer CLI: `lunaris run | demo | version`
├── io/                 data ingestion & caching
├── polarimetry/        DFSAR Stokes/CPR/DOP, decompositions, speckle, coherence
├── classify/           O(1) LUT, RandomForest fusion, Bayesian/Dempster-Shafer
├── terrain/            DEM derivatives, illumination/shadow, thermal, boulders
├── planning/           cost grid, A*/D*Lite/Theta*/RRT*/NSGA-II, energy, landing
├── volume/             dielectric mixing & penetration, Monte-Carlo volume
└── viz/                maps, charts, 3-D terrain, HTML report
```

### `lunaris.io` — ingestion, real & synthetic

Responsibility: get co-registered rasters into a `LunarScene`, fast.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `io.readers` | `read_raster(path)`, `read_cog_window(url, bounds, band)`, `reproject_to_south_polar(arr, transform, crs, …)`, `GDAL_COG_ENV` | rasterio/GDAL real-data I/O; **O(1) windowed COG reads** via `rasterio.windows.from_bounds` + `/vsicurl/` HTTP range requests; reprojection to the lunar south-polar CRS |
| `io.synthetic` | `generate_faustini_scene(n, resolution_m, seed)` | deterministic, physically-plausible synthetic Faustini scene (bowl crater + nested doubly-shadowed crater, full L/S Stokes with an engineered ICE patch and roughness *decoys*, thermal/illumination/albedo layers, ground-truth mask) |
| `io.registry` | `Dataset`, `DATASETS`, `list_datasets()`, `get_dataset(key)`, `as_table()` | the **30+ dataset fusion catalogue** (26 curated records spanning radar/optical/IR/thermal/UV/neutron/topography + LCROSS ground truth) |
| `io.cache` | `cached` (decorator), `memory`, `clear_cache()`, `CACHE_DIR` | joblib on-disk memoisation: **O(1) re-access** of expensive deterministic artefacts (horizon maps, SVF, LUT builds, MC runs) |

### `lunaris.polarimetry` — DFSAR radar polarimetry

Responsibility: convert the DFSAR scattering matrix into the CPR/DOP criterion
and decomposition channels that diagnose ice.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `polarimetry.stokes` | `scattering_matrix_to_circular(shh, shv, svh, svv)`, `stokes_from_circular(E_RH, E_RV, window=5)`, `multilook(arr, looks=2)` | CTLR hybrid-pol transform to the circular basis; **multilooked** Stokes S0..S3 (multilooking is mandatory — a single look has DOP ≡ 1) |
| `polarimetry.cpr` | `circular_polarization_ratio(s0, s3)`, `degree_of_polarization(s0,s1,s2,s3)`, `sc_oc_power(s0, s3)` | the two scalars of the Sinha et al. (2026) criterion: `CPR=(S0-S3)/(S0+S3)`, `DOP=√(S1²+S2²+S3²)/S0`, and SC/OC powers |
| `polarimetry.decomposition` | `m_chi(s0..s3)`, `m_delta(s0..s3)`, `child_parameters(s0..s3)`, `cloude_pottier(E_RH, E_RV, …)` | m-χ (even/volume/odd) & m-δ (double/volume/surface) power splits; CHILD child parameters (χ, δ, ψ, conformity); Cloude-Pottier H/A/α eigen-decomposition (batched `np.linalg.eigh`) |
| `polarimetry.speckle` | `boxcar(img, size=5)`, `refined_lee(img, …)` | SAR speckle filters (Lee 1981; refined-Lee 2009) |
| `polarimetry.coherence` | `copol_coherence(shh, svv, window=5)` | HH-VV complex-coherence magnitude `γ_hhvv ∈ [0,1]` — low over volume scatter (ice) |

### `lunaris.classify` — ice classification & fusion

Responsibility: turn radar (and multi-sensor) evidence into ice masks and an ice
posterior — fast and false-positive-hardened.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `classify.ice_lut` | `default_edges(cpr_max=4, dop_max=1, nbins=256)`, `build_ice_lut(cpr_edges, dop_edges)`, `classify_ice_lut(cpr, dop, lut, …)`, `classify_ice_threshold(cpr, dop, …)` | the **flagship O(1)-per-pixel** classifier: a precomputed boolean `(CPR, DOP)` table classified by `digitize` + single fancy-index gather; provably bit-identical to the direct rule on a fine grid |
| `classify.fusion` | `build_feature_stack(scene)`, `train_ice_classifier(X, y)`, `predict_ice(model, X, shape)`, `feature_importance(model, names)`, `bake_lut_from_model(model, …)` | 10-feature multi-sensor stack → balanced RandomForest ice probability; can **bake** the learned model back into the same O(1) LUT |
| `classify.evidence` | `bayesian_fusion(layers, weights)`, `dempster_shafer(masks)` | probabilistic fusion of independent evidence into an ice posterior / belief |

The 10-feature canonical order (`FEATURE_NAMES`) is:
`cpr_L, dop_L, cpr_S, dop_S, mchi_volume, temperature_max, illumination,
albedo_1064, lamp_ratio, earth_visibility`.

### `lunaris.terrain` — terrain, illumination & thermal

Responsibility: derive the safety, illumination and cold-trap context that turns
a detection into a *site*.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `terrain.dem` | `slope_horn(dem, res)`, `aspect(dem, res)`, `rms_roughness(dem, baseline_px)`, `hurst_exponent(dem, baselines_px)`, `iqr_roughness(dem, window)` | Horn (1981) slope/aspect; Rosenburg (2011) deviogram RMS roughness & self-affine **Hurst exponent**; Kreslavsky & Head (2000) boulder-robust IQR roughness |
| `terrain.illumination` | `horizon_map(dem, res, n_azimuth)`, `permanent_shadow_mask(dem, res, …)`, `double_shadow_mask(dem, res, psr_mask)`, `earth_visibility_map(dem, res)`, `sky_view_factor(dem, res)` | horizon raytracing; PSR & **doubly-shadowed** masks; DTE comms windows; sky-view factor (drives radiative cooling) |
| `terrain.thermal` | `cold_trap_mask(tmax, threshold=110K)`, `sublimation_rate(T)`, `ice_stability_depth(tmax, …)`, `regolith_thermal_profile(…)` | Diviner-style cold-trap masks; Hertz-Knudsen sublimation; ice-stability burial depth; 1-D conductive regolith profile |
| `terrain.boulders` | `detect_boulders_shadow(img, sun_elev_deg, gsd)`, `boulder_density_map(boulders, shape, window)` | shadow-length boulder detection (`h = L·tan(elev)`) and density rasterisation for hazard mapping |

### `lunaris.planning` — landing & traverse

Responsibility: convert hazard + illumination layers into a safe landing site and
an energy-feasible traverse.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `planning.cost` | `build_cost_grid(slope, roughness, illumination, traversable, weights)`, `traversability_mask(slope, max_slope)`, `DEFAULT_WEIGHTS` | weighted-overlay traversability cost grid (`tan(slope)` energy term + roughness + darkness); `+inf` on impassable cells |
| `planning.astar` | `astar(cost, start, goal, connectivity=8)` | A\* (Hart-Nilsson-Raphael 1968) with octile heuristic on the cost grid |
| `planning.dstar_lite` | `DStarLite(cost, start, goal)` with `.plan()`, `.update_cost(changes)` | D\* Lite (Koenig & Likhachev 2002) incremental replanner for dynamic hazards |
| `planning.theta_star` | `theta_star(cost, start, goal, obstacles)` | Theta\* any-angle planner (Nash et al. 2007) with line-of-sight smoothing |
| `planning.rrt_star` | `rrt_star(cost, start, goal, n_samples, step, …)` | asymptotically-optimal sampling planner (Karaman & Frazzoli 2011) |
| `planning.nsga2` | `nsga2_paths(objectives, start, goal, …)` | NSGA-II (Deb et al. 2002) multi-objective Pareto front (distance vs energy vs hazard vs science) |
| `planning.energy` | `solar_power(elev_deg, …)`, `drive_energy_per_m(slope_deg, …)`, `survival_time_h(battery_wh, load_w)`, `energy_aware_plan(…)` | rover power/energy/SOC model; **70-h shadow survival** constraint; charging-stop scheduling |
| `planning.landing` | `landing_suitability(layers, weights)`, `ahp_weights(pairwise)`, `select_landing_sites(score, n, traversable)` | AHP-weighted MCDA suitability and top-`n` non-adjacent site selection |

### `lunaris.volume` — dielectric inversion & ice volume

Responsibility: bound the sampled column and estimate top-5 m ice volume with
honest uncertainty.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `volume.dielectric` | `looyenga_ice_fraction(eps_eff, …)`, `looyenga_eps(f_ice, …)`, `maxwell_garnett_eps(f_ice, …)`, `penetration_depth(wavelength, eps, tan_delta)`, `cpr_to_ice_likelihood(cpr, dop, …)` | Looyenga-Landau-Lifshitz invertible mixing; Maxwell-Garnett forward cross-check; 1/e penetration depth; **relative** ice-likelihood index (not absolute wt%) |
| `volume.estimate` | `ice_volume(area, depth, frac)`, `ice_mass(volume, rho)`, `monte_carlo_volume(area, depth, frac, n, seed, **σ)` | `V = A·d·f`, `M = ρV`, and **Monte-Carlo** uncertainty (mean, σ, 2.5/97.5 percentile CI) |

### `lunaris.viz` — visualisation & report

Responsibility: produce publication-quality, head-less, self-contained outputs.

| Module | Key public functions | Responsibility |
|--------|----------------------|----------------|
| `viz.maps` | `plot_layer`, `plot_ice_map`, `mchi_rgb`, `plot_mchi`, `hillshade`, `plot_traverse` | static matplotlib maps (Agg backend, no display needed): ice overlay, m-χ RGB composite, hillshade, traverse polyline |
| `viz.charts` | `cpr_dop_scatter`, `volume_histogram` | the CPR-DOP decision-space scatter and the Monte-Carlo volume histogram |
| `viz.terrain3d` | `terrain3d_html` | interactive Plotly 3-D terrain view |
| `viz.report` | `build_report(results, out_html)` | assembles all of the above into one self-contained HTML deliverable |

---

## 4. The "fastest platform / O(1)" engineering

The platform advertises itself as the *fastest* lunar ice-detection pipeline. We
back that with a precise, honest accounting of algorithmic complexity. The
discipline here — distinguishing true `O(1)` from `O(log n)` from
constant-factor speedups — is a deliberate differentiator: it is the difference
between a marketing claim and an engineering one.

### Complexity ledger

| Operation | Where | Cost | Class | Why |
|-----------|-------|------|-------|-----|
| COG byte-offset windowed read of an AOI tile | `io.readers.read_cog_window` | constant per tile (independent of mosaic size) | **O(1)** | `rasterio.windows.from_bounds` computes the overlapping window from the affine transform; only the intersecting COG tiles' byte ranges are fetched via HTTP range requests — extracting one crater from a continent-scale mosaic never reads the whole image |
| `(CPR, DOP)` LUT ice classification, per pixel | `classify.ice_lut.classify_ice_lut` | constant per pixel | **TRUE O(1)/px** | the decision boundary is *baked* into a fixed-size boolean table at build time; at classify time each pixel is a `digitize` (binary search over a *fixed, tiny* edge array — cost independent of image size) plus **one** fancy-index gather. No per-pixel physics, no per-pixel branching. Doubling resolution only doubles the (trivial) gather count; it never changes the per-pixel cost |
| H3 64-bit hex cell ops (index / neighbour / parent) | (H3 integration) | constant | **O(1)** | H3 cell IDs are 64-bit integers; resolution, parent/child and neighbour traversal are bitwise/arithmetic operations on the ID, independent of dataset size |
| `np.memmap` page access | array-backed reads | constant per page | **O(1)** | the OS maps file pages on demand; touching an element faults in one page — random access cost is independent of file length |
| joblib cache hit | `io.cache.cached` | constant (hash → load) | **O(1)** | a previously computed deterministic artefact is re-accessed by hashing the call signature and loading the pickle — no recomputation |
| R-tree / cKDTree spatial query (k-NN, range) | spatial joins, boulder/site search | `O(log n)` | **O(log n)** | balanced spatial trees answer a nearest-neighbour or range query in logarithmic time in the number of indexed features — *not* constant, and we say so |
| A\* / Theta\* path search | `planning.astar`, `theta_star` | `O(E log V)` | super-linear | priority-queue graph search over the cost grid |
| numba JIT / NumPy vectorisation | hot loops | constant-factor speedup | **NOT asymptotic** | SIMD/JIT shrink the constant in front of the same `O(n)` (or `O(n·m)`) work; they make `O(n)` *faster*, they do not make it `O(1)` |

### Why the LUT classifier is genuinely O(1) per pixel (the proof)

The Sinha et al. (2026) rule `ice = (CPR > 1) ∧ (DOP < 0.13)` is **monotone** in
both axes (ice for *larger* CPR, ice for *smaller* DOP). `build_ice_lut`
discretises the physical domain into `nbins²` cells and flags a cell `True` iff
its **centre** satisfies the rule:

```python
cpr_ok = (cpr_centers > CPR_ICE_THRESHOLD)[:, None]   # (nc, 1)
dop_ok = (dop_centers < DOP_ICE_THRESHOLD)[None, :]    # (1, nd)
lut    = cpr_ok & dop_ok                               # (nc, nd) boolean
```

Classification then performs, per pixel, only:

```python
ci   = clip(digitize(cpr, cpr_edges) - 1, 0, nc - 1)   # binary search, fixed edges
di   = clip(digitize(dop, dop_edges) - 1, 0, nd - 1)
mask = lut[ci, di]                                      # single fancy-index gather
```

The per-pixel work is a constant number of memory operations — **independent of
the image size, of the number of thresholds, and of any model complexity folded
into the table**. Because the rule is monotone and we evaluate it at bin centres,
on a sufficiently fine grid the LUT is *bit-identical* to the direct rule; the
default 256×256 grid reproduces `classify_ice_threshold` exactly on the synthetic
Faustini scene (see `tests/test_classify.py`). Crucially, `bake_lut_from_model`
shows the *learned* RandomForest classifier can be collapsed into the **same**
O(1) table — multi-sensor inference at LUT speed, with zero model inference at
runtime.

> **Honesty note.** `np.digitize` is a binary search, which is `O(log k)` in the
> number of *edges* `k`. But `k` is a fixed, tiny constant (257) chosen at build
> time and **independent of the image size `n`**. So with respect to the data —
> the quantity that actually scales — the classifier is `O(1)` per pixel and
> `O(n)` for an `n`-pixel scene with the smallest possible constant. We are
> precise about this rather than hiding it.

---

## 5. The science, with equations

### 5.1 Chandrayaan-2 DFSAR — instrument

DFSAR is a dual-frequency, full- and hybrid-polarimetric SAR
(Bhiravarasu et al. 2021):

| Parameter | L-band | S-band |
|-----------|--------|--------|
| Centre frequency | 1.25 GHz | 2.5 GHz |
| Wavelength λ | 0.2399 m (~24 cm) | 0.1199 m (~12 cm) |
| Polarimetry | full-pol + hybrid (compact) | full-pol + hybrid (compact) |
| Incidence-angle range | 9.6°–36.9° | 9.6°–36.9° |
| Spatial resolution | 2–75 m (mode-dependent) | 2–75 m |
| Nominal penetration | ~3–5 m (regolith, low-loss) | ~1–2 m |

The longer L-band wavelength penetrates several metres into dry, low-loss
regolith, making it the workhorse for *subsurface* ice; S-band adds a shallower,
shorter-wavelength cross-check.

### 5.2 From scattering matrix to Stokes vector (circular basis)

DFSAR transmits/receives in a linear (H/V) basis; the ice criterion is defined in
the *circular* basis (Raney 2007 hybrid-polarity / CTLR architecture). For a
right-circular transmit `(1, j)ᵀ/√2`, the received horizontal/vertical fields are

```
E_RH = (Shh + j·Shv) / √2
E_RV = (Svh + j·Svv) / √2
```

The four Stokes parameters follow from a *multilooked* (boxcar `⟨·⟩`) ensemble
average (Raney 2007):

```
S0 = ⟨|E_RH|²⟩ + ⟨|E_RV|²⟩      (total power)
S1 = ⟨|E_RH|²⟩ − ⟨|E_RV|²⟩
S2 =  2·Re⟨E_RH·E_RV*⟩
S3 = −2·Im⟨E_RH·E_RV*⟩
```

> **Multilooking is mandatory.** Without spatial averaging every pixel is a
> fully-polarised single look, the cross-products factorise, and the degree of
> polarisation collapses to `m ≡ 1` identically. The averaging window introduces
> the partial-coherence statistics that make `m < 1` (and therefore the CPR/DOP
> criterion) meaningful.

### 5.3 The two ice scalars: CPR and DOP

```
CPR = SC / OC = (S0 − S3) / (S0 + S3)          (Campbell 2012, Eq. 1)
DOP = m = √(S1² + S2² + S3²) / S0   ∈ [0, 1]    (Raney 2012)
```

where `SC = (S0−S3)/2` and `OC = (S0+S3)/2` are the same-/opposite-sense circular
received powers. **CPR > 1** indicates strong same-sense scattering consistent
with the coherent-backscatter opposition effect (CBOE) of buried ice; **DOP <
0.13** indicates a depolarised, randomly-polarised return (volume scatter)
rather than a polarised surface return.

### 5.4 m-χ and m-δ decompositions

The hybrid-pol decompositions (Raney et al. 2012) split total power into
physically interpretable channels using the degree of polarisation `m` and either
the ellipticity `χ` (m-χ) or the relative phase `δ` (m-δ):

m-χ, with `χ = ½·arcsin(−S3/(m·S0))`:

```
even   = √( m·S0·(1 + sin 2χ) / 2 )
volume = √( S0·(1 − m) )
odd    = √( m·S0·(1 − sin 2χ) / 2 )
```

m-δ, with `δ = arctan2(S3, S2)`:

```
double  = √( m·S0·(1 + sin δ) / 2 )
volume  = √( S0·(1 − m) )
surface = √( m·S0·(1 − sin δ) / 2 )
```

Over subsurface ice, the same-sense `S3 < 0` drives the **volume** channel
(rendered green in the m-χ RGB) dominant — the diagnostic ice signature. The
CHILD parameters add ellipticity `χ`, relative phase `δ`, orientation
`ψ = ½·arctan2(S2, S1)` and conformity `−S3/S0`. The Cloude-Pottier
decomposition adds entropy `H`, anisotropy `A`, and mean scattering angle `α`
from the eigenvalues of the 2×2 circular coherency matrix
`J = ⟨k·kᴴ⟩` with `k = [E_RH, E_RV]ᵀ`.

### 5.5 The CPR-ambiguity problem and how lunaris resolves it

A high CPR is **not by itself** proof of ice. Two competing physical mechanisms
both raise CPR:

* **Coherent Backscatter Opposition Effect (CBOE):** multiple scattering within
  a low-loss volume of buried ice produces same-sense returns and `CPR > 1`
  (the ice interpretation; Spudis et al. 2013 argued anomalous polar craters are
  ice-bearing on this basis).
* **Surface/wavelength-scale roughness (rocks, blocky ejecta, fresh crater
  rims):** dihedral and multiple surface bounces *also* raise CPR with no ice
  present (Fa & Cahill 2013; Eke et al. 2014 showed that many high-CPR craters
  are simply rough, not icy — the Spudis-vs-Fa/Eke debate).

This is the central scientific risk of any radar-only ice claim. `lunaris`
resolves it on **two** complementary fronts:

1. **The DOP gate (the refinement of Sinha et al. 2026).** Rough-surface returns
   are *polarised* (high DOP); genuine ice volume-scatter is *depolarised* (low
   DOP). Requiring `DOP < 0.13` *in addition to* `CPR > 1` rejects the
   roughness decoys that pass a CPR-only test. The synthetic Faustini scene is
   engineered to contain exactly such decoys (rough rim/ejecta with **high CPR
   and high DOP**) to validate that the joint criterion rejects them while
   accepting the coherent ice patch.
2. **Independent-physics corroboration (multi-sensor fusion, §6).** A detection
   is only promoted to high confidence where *independent* sensors agree: the
   pixel is also a thermal cold trap (`Tmax < 110 K`, Diviner), shows neutron
   hydrogen suppression (LEND/LP-NS), and/or optical-frost brightening
   (LOLA 1064 nm albedo, LAMP UV). Roughness has no reason to correlate with all
   of these; ice does.

### 5.6 Dielectric mixing and ice fraction (Looyenga)

The effective real permittivity of an ice+regolith mixture follows the
symmetric power-law Looyenga-Landau-Lifshitz (LLL) rule, inverted for the ice
volume fraction `f`:

```
ε_eff^(1/3) = f·ε_ice^(1/3) + (1 − f)·ε_reg^(1/3)
        ⇒  f = (ε_eff^(1/3) − ε_reg^(1/3)) / (ε_ice^(1/3) − ε_reg^(1/3))
```

with `ε_ice ≈ 3.15` (cold water ice) and `ε_reg ≈ 3.0` (dry regolith;
`ε_reg' = 1.919^ρ`, Olhoeft & Strangway 1975). A dilute Maxwell-Garnett forward
model provides a low-fraction cross-check. **Critical caveat:** because
`ε_ice ≈ ε_reg`, the dielectric contrast is weak and absolute ice weight-percent
is *not* retrievable from permittivity alone (Heggy et al. 2012). `lunaris`
therefore reports a **relative ice-likelihood index**, never an absolute
abundance (see `cpr_to_ice_likelihood` and §9).

### 5.7 Radar penetration depth

For a low-loss dielectric (`tan δ ≪ 1`) the 1/e power penetration (skin) depth is

```
δ_pen = λ / (2π · √ε · tan δ)     [m]
```

With L-band (λ = 0.2399 m), `ε = 3`, `tan δ = 0.005`, this is **≈ 4.4 m** —
consistent with sounding the top ~5 m PSR column required by deliverable D5. The
depth shrinks with increasing loss tangent and with the shorter S-band
wavelength. The regolith loss tangent itself scales with composition
(Carrier, Olhoeft & Mendell 1991):
`tan δ = 10^(0.038·(%FeO + %TiO₂) + 0.312·ρ − 3.260)`.

### 5.8 Illumination, PSRs and double shadows

Because the Moon's obliquity is only **1.543°**, the Sun's maximum elevation at
the pole is bounded by ~1.54°. A pixel is a **Permanently Shadowed Region (PSR)**
if the Sun never rises above its local horizon over a full year; a **doubly
shadowed** pixel additionally receives no secondary (scattered/thermal)
illumination from surrounding sunlit terrain — the coldest traps. `lunaris`
raytraces the DEM horizon in `n_azimuth` directions, derives the PSR mask, then
the doubly-shadowed mask within it, plus the sky-view factor
`SVF = 1 − ⟨sin²(horizon_angle)⟩` that governs radiative cooling.

### 5.9 Cold traps, sublimation and ice stability

Volatiles are retained where the *annual maximum* temperature stays below the
species' stability ceiling (Paige et al. 2010):

| Species | Stability ceiling |
|---------|-------------------|
| H₂O | **≤ 110 K** (stable over Gyr) |
| SO₂ / CO₂ | ≤ 60 K |
| NH₃ | ≤ 66 K |
| super-volatiles (CO, N₂, CH₄) | ≤ 40 K |

Free-space sublimation mass loss follows Hertz-Knudsen:

```
E = P_sv(T) · √( M / (2π·R·T) )     [kg m⁻² s⁻¹]
```

with `P_sv` the Clausius-Clapeyron saturation vapour pressure and
`M = 0.018015 kg mol⁻¹`. Where surface `Tmax` exceeds 110 K, ice is only stable
below a burial depth set by the regolith thermal gradient — the `ice_stability_depth`
that bounds the top-5 m volume model (Schörghofer & Williams 2020).

### 5.10 Rover energy model and 70-h shadow survival

Solar generation, slope-dependent drive energy and battery survival are coupled
(VIPER-class planning defaults on a Pragyan-class 27 kg chassis):

```
P_solar  = S · A · η · max(sin(elev), 0)                      [W]
E_drive  = m·g·( sin(slope) + Crr·cos(slope) ) per metre      [J m⁻¹]
t_survive = battery_Wh / load_W                               [h]
```

with `S = 1361 W m⁻²`, `A = 1.5 m²`, `η = 0.30`, `m = 27 kg`,
`g = 1.62 m s⁻²`, `Crr = 0.2`, usable battery `7000 Wh`. The hard mission
constraint is **70 h** of survivable shadow dwell: the energy-aware planner must
never let the state-of-charge fall below what is needed to ride out a 70-h
shadow at hibernate load, scheduling charging stops in illuminated waypoints.

### 5.11 Traverse cost function

The planners operate on a single scalar cost grid built as a weighted overlay
(cf. the VIPER traversability formulation, Tompkins et al. arXiv:2401.08558):

```
cost = w_d
     + w_s · tan(radians(slope))     # super-linear climb-energy penalty
     + w_r · roughness_norm          # ride-quality / hazard penalty
     + w_I · (1 − illumination)      # darkness / thermal-survival penalty
```

with default weights `{distance:1, slope:4, roughness:1, illumination:2}` (slope
dominates because it maps directly to drive energy and roll-over risk). Cells are
blocked (`+inf`) where `slope > 20°` (`ROVER_MAX_SLOPE_DEG`) or the traversability
mask is `False`. The `tan(slope)` form is deliberate: per unit horizontal
advance the extra path length grows like `1/cos θ` and the gravitational work
like `sin θ`, so `tan θ = sin/cos` captures the super-linear blow-up as the rover
nears its climbing limit, matching the slope-energy model in `planning.energy`.

---

## 6. Multi-sensor data-fusion strategy

A radar-only ice claim is fragile (§5.5). `lunaris` therefore treats DFSAR as
*one* line of evidence among **30+ multi-sensor datasets** (catalogued in
`io.registry`, 26 curated records; see `docs/DATA_SOURCES.md`) and fuses them so
that **independent physics defeats false positives**.

### The cross-verification logic

A pixel is a high-confidence subsurface-ice site only where *physically
independent* indicators corroborate one another:

```
ICE_confidence ↑  ⇔   radar(CPR>1 ∧ DOP<0.13)          (Chandrayaan-2 DFSAR)
                 ∧   thermal cold trap (Tmax < 110 K)   (LRO Diviner)
                 ∧   neutron H suppression (WEH)         (LRO LEND, LP-NS)
                 ∧   optical/UV frost brightening        (LOLA 1064 nm, LAMP)
                 ∧   (no rock-abundance decoy)            (Diviner rock abundance)
```

The power of the fusion is that the failure modes are **uncorrelated**:
wavelength-scale roughness raises CPR but has no reason to be a thermal cold
trap, suppress epithermal neutrons, *and* brighten in the UV — whereas ice
does all four. The evidence layers span complementary depths and physics: radar
(metres-deep dielectric/structure), neutrons (top ~1 m hydrogen), thermal
(surface stability), and UV/NIR/albedo (top-microns frost).

### Evidence groups (independent physics)

| Evidence | Sensors | Physics | Defeats |
|----------|---------|---------|---------|
| Radar polarimetry | Ch-2 DFSAR, Ch-1 Mini-SAR, LRO Mini-RF, Arecibo | CBOE / volume scatter | — (the primary detector) |
| Thermal cold trap | LRO Diviner | `Tmax < 110 K` Gyr stability | rough-but-warm decoys |
| Neutron hydrogen | LRO LEND, LP-NS | epithermal suppression → WEH | dry rough surfaces |
| Optical/UV frost | LOLA 1064 nm albedo, LAMP UV, ShadowCam, M3, IIRS | surface frost brightening / 3 µm OH/H₂O | dry regolith |
| Roughness context | Diviner rock abundance, LOLA roughness | rock fraction → radar-decoy flag | mislabelled rough pixels |

### Formal fusion (Bayesian / Dempster-Shafer / learned)

* **Bayesian (log-odds) fusion** (`classify.evidence.bayesian_fusion`): combine
  independent evidence probabilities by summing weighted logits,
  `logit(P) = Σ_k w_k·logit(p_k)`, then `P = σ(logit)`. This is the canonical way
  to pool conditionally-independent likelihood ratios into a posterior.
* **Dempster-Shafer** (`classify.evidence.dempster_shafer`): treat each source as
  a basic belief assignment toward the "ice" hypothesis and combine with
  Dempster's rule — robust when sources are uncertain or partially conflicting,
  and it represents *ignorance* explicitly (unlike a single point probability).
* **Supervised RandomForest** (`classify.fusion`): a balanced 200-tree forest on
  the 10-feature multi-sensor stack provides a learned, non-linear corroboration
  and ranks feature importance. It can be **baked back** into the O(1) LUT
  (`bake_lut_from_model`), so the multi-sensor decision runs at lookup speed.

---

## 7. Data flow & formats

### Formats and standards

* **PDS4 / PDS3** — the native archive formats for Chandrayaan-2 (PRADAN) and
  LRO/Lunar-Prospector (PDS Geosciences). Opened directly with GDAL's PDS
  drivers or `pds4_tools`; converted once to COG for fast AOI access.
* **Cloud-Optimized GeoTIFF (COG)** — the internal working format. Tiled
  (256×256), deflate-compressed, with `[2,4,8]` overviews, enabling **O(1)
  windowed reads** of an AOI over HTTP range requests (`/vsicurl/`).
* **STAC** — SpatioTemporal Asset Catalog (e.g. USGS Astrogeology) for
  discovery/indexing of the COG mosaics.

### Coordinate reference system

All layers are co-registered in the **lunar south-polar stereographic** CRS on a
sphere of radius `R = 1 737 400 m`:

```
+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs
```

equivalently the authority code **ESRI:103878** (Moon 2000 south-polar
stereographic). Reprojection of any source raster into this CRS is handled by
`io.readers.reproject_to_south_polar`.

### Real-data ingestion with deterministic synthetic fallback

The pipeline's ingest stage (§2, ①) first attempts the **real** path: windowed
COG reads of the Chandrayaan-2 DFSAR/OHRC AOI plus supporting LRO/Kaguya
products, reprojected and stacked into a `LunarScene`. When no data are present
(`Settings.use_synthetic = True`, the default), it falls back to
`io.synthetic.generate_faustini_scene(n, resolution_m, seed)` — a deterministic,
physically-plausible scene seeded by a single `np.random.default_rng(seed)`, so
identical `(n, resolution_m, seed)` yields **bit-identical** arrays. This makes
the entire pipeline (and every test) runnable end-to-end with zero external
dependencies, while keeping the real-data path a drop-in replacement.

---

## 8. Tech stack & dependencies

`lunaris` targets Python ≥ 3.11 and depends only on mature, open scientific
libraries (full list in `pyproject.toml`):

| Dependency | Role in lunaris | Why it (and not an alternative) |
|------------|-----------------|---------------------------------|
| **rasterio / GDAL** | raster I/O, COG windowed reads, reprojection | the de-facto standard for geospatial raster + the only practical route to O(1) `/vsicurl/` windowed COG reads and PDS drivers |
| **numpy** | every array operation, the O(1) LUT, Stokes math | the vectorised numerical core; the LUT's single fancy-index gather is a NumPy primitive |
| **scipy** | `ndimage` filters (multilook, roughness, speckle), linear algebra | boxcar/percentile/uniform filters and batched `eigh` for Cloude-Pottier |
| **scikit-image** | image-processing primitives (boulder/blob, morphology) | robust, well-tested CV ops for shadow-boulder detection |
| **scikit-learn** | RandomForest multi-sensor classifier | balanced, interpretable, feature-importance-ready ensemble that bakes into the LUT |
| **spiceypy** | SPICE ephemerides / illumination geometry | NAIF SPICE is the authoritative source for Sun/Earth geometry used in illumination & comms modelling |
| **h3** | hex spatial indexing | 64-bit bitwise cell ops give O(1) neighbour/parent/aggregation |
| **plotly** | interactive 3-D terrain & HTML figures | self-contained, offline-embeddable interactive graphics for the report |
| **networkx** | graph structures for planners | clean graph abstractions backing the traverse search |
| **numba** | JIT for hot numeric loops | constant-factor acceleration of raytracing / per-cell kernels (explicitly *not* an asymptotic change, §4) |
| **pyproj** | CRS transforms | rigorous projection handling for the south-polar stereographic CRS |
| **pydantic / pydantic-settings** | typed `Settings`, YAML/env config | validated, environment-overridable configuration |
| **typer / rich** | CLI (`lunaris run/demo/version`) and pretty tables | ergonomic, self-documenting CLI |
| **joblib** | on-disk memoisation (O(1) artefact re-access) | transparent caching of expensive deterministic computations |
| **matplotlib** | static publication maps (Agg, head-less) | reliable server/CI rendering with no display |
| **pandas / pyyaml** | tabular reporting, config parsing | dataset catalogue tables and YAML configs |

---

## 9. Reproducibility, testing, performance, limitations & future work

### Reproducibility

* **Single source of truth for physics.** Every numeric constant lives in
  `lunaris.constants`; downstream modules import from it rather than hard-coding,
  so one review of that file validates the physics of the whole pipeline.
* **Deterministic by construction.** The synthetic scene, the RandomForest
  (`random_state`), the Monte-Carlo volume (`seed`), and the planners (`seed`)
  are all seeded; identical inputs yield identical outputs.
* **Typed, file-overridable config.** `Settings` (pydantic-settings) resolves
  from explicit kwargs → `LUNARIS_*` env vars → defaults, with a YAML overlay via
  `load_config`. The `configs/faustini.yaml` pins the target run.
* **GIS-loadable artefacts.** `LunarScene.save` writes COGs + a JSON sidecar in
  the documented CRS, so every intermediate opens in QGIS/ArcGIS.

### Testing

The package ships with a `pytest` suite (`tests/`) covering the data backbone
(`test_foundation.py`), the polarimetry math, and — importantly — the assertion
that the **O(1) LUT is bit-identical to the direct rule** on the synthetic scene
(`test_classify.py`). `ruff` lints the codebase (`E,F,I,W,UP,B`). The `lunaris
demo` CLI command is itself a real, working smoke test that exercises ingest →
CPR/DOP end-to-end with no external data.

### Performance

The per-pixel ice decision is O(1) (§4); COG windowed reads keep ingest
proportional to AOI rather than mosaic size; joblib caches make repeated runs of
deterministic stages O(1); and the heavy per-cell kernels (raytracing,
filtering) are vectorised (and JIT-compilable with numba) for the smallest
practical constant factor.

### Limitations (stated honestly)

1. **Relative, not absolute, ice abundance.** Because `ε_ice ≈ 3.15 ≈ ε_reg ≈
   3.0`, the ice/regolith dielectric contrast is weak and absolute ice
   weight-percent is *not* retrievable from radar permittivity (Heggy et al.
   2012). We therefore report a **relative ice-likelihood index** and propagate
   the ill-conditioned Looyenga inversion explicitly through the Monte-Carlo
   estimator rather than ignoring it. The LCROSS `5.6 ± 2.9 wt%` ground truth
   (Colaprete et al. 2010) is used only as a sanity anchor, not a calibration.
2. **Residual DFSAR polarimetric distortion.** Channel imbalance and
   cross-talk in real DFSAR products bias CPR/DOP; full polarimetric calibration
   (and faraday-free L-band assumptions) are required before the criterion is
   applied to operational data. The synthetic path sidesteps this; the real path
   must not.
3. **Synthetic-data caveat.** The default scene is *engineered* to contain a
   coherent ice patch and roughness decoys to validate the algorithms; it is a
   testbed, not an observation. Quantitative results carry meaning only once real
   Chandrayaan-2 granules are ingested.
4. **CPR ambiguity is mitigated, not eliminated.** The DOP gate + multi-sensor
   fusion strongly suppress roughness false positives, but radar remote sensing
   of an `ε ≈ 3` contrast in a doubly shadowed crater remains inherently
   uncertain; we communicate confidence, not certainty.

### Future work

* Ingest and co-register the supplied **real Chandrayaan-2 DFSAR/OHRC granules**;
  add full polarimetric calibration (channel imbalance, cross-talk).
* Validate the detection framework against **independent ground truth** (LCROSS,
  cross-mission Mini-RF/Arecibo CPR maps).
* Wire `spiceypy` SPICE kernels for time-resolved illumination/thermal histories
  feeding the cold-trap and rover-survival models.
* Extend NSGA-II objectives with explicit **science-value** and **comms-window**
  terms; close the loop with D\* Lite online replanning on simulated rover telemetry.
* Calibrate the regolith loss tangent per-pixel from Ch-2 CLASS / Kaguya MI
  composition to sharpen the penetration-depth and volume estimates.

---

## 10. References

1. **Sinha, R. K. et al. (2026).** Refined CPR>1 & DOP<0.13 subsurface-ice
   criterion for lunar PSRs. *npj Space Exploration*.
   DOI: [10.1038/s44453-026-00038-9](https://doi.org/10.1038/s44453-026-00038-9)
2. **Bhiravarasu, S. S. et al. (2021).** Chandrayaan-2 Dual-Frequency Synthetic
   Aperture Radar (DFSAR): performance characterization and initial results.
   *The Planetary Science Journal* 2, 134.
   DOI: [10.3847/PSJ/ac0244](https://doi.org/10.3847/PSJ/ac0244)
3. **Raney, R. K. (2007).** Hybrid-polarity SAR architecture.
   *IEEE Trans. Geosci. Remote Sens.* 45(11), 3397-3404.
   DOI: [10.1109/TGRS.2007.895883](https://doi.org/10.1109/TGRS.2007.895883)
4. **Raney, R. K., Cahill, J. T. S., Patterson, G. W., Bussey, D. B. J. (2012).**
   The m-chi decomposition of hybrid dual-polarimetric radar data.
   *IGARSS 2012*, 5093-5096.
   DOI: [10.1109/IGARSS.2012.6352465](https://doi.org/10.1109/IGARSS.2012.6352465)
5. **Spudis, P. D. et al. (2013).** Evidence for water ice on the Moon: Results
   for anomalous polar craters from the LRO Mini-RF imaging radar.
   *J. Geophys. Res. Planets* 118, 2016-2029.
   DOI: [10.1002/jgre.20156](https://doi.org/10.1002/jgre.20156)
6. **Fa, W. & Cahill, J. T. S. (2013).** Coherent backscatter opposition effect
   in radar observations of the lunar polar regions.
   *J. Geophys. Res. Planets* 118.
   DOI: [10.1002/jgre.20133](https://doi.org/10.1002/jgre.20133)
7. **Eke, V. R. et al. (2014).** The effect of craters on the lunar neutron and
   radar ice signal. *Icarus* 241, 66-78.
   DOI: [10.1016/j.icarus.2014.06.021](https://doi.org/10.1016/j.icarus.2014.06.021)
8. **Campbell, B. A. (2012).** High circular polarization ratios in radar
   scattering from geologic targets. *J. Geophys. Res. Planets* 117, E06008.
   DOI: [10.1029/2012JE004061](https://doi.org/10.1029/2012JE004061)
9. **Mazarico, E. et al. (2011).** Illumination conditions of the lunar polar
   regions using LOLA topography. *Icarus* 211, 1066-1081.
   DOI: [10.1016/j.icarus.2010.10.030](https://doi.org/10.1016/j.icarus.2010.10.030)
10. **O'Brien, P. & Byrne, S. (2022).** Stability and detectability of buried
    polar ice. *J. Geophys. Res. Planets* 127.
    DOI: [10.1029/2021JE006918](https://doi.org/10.1029/2021JE006918)
11. **Paige, D. A. et al. (2010).** Diviner Lunar Radiometer observations of cold
    traps in the Moon's south polar region. *Science* 330, 479-482.
    DOI: [10.1126/science.1187726](https://doi.org/10.1126/science.1187726)
12. **Williams, J.-P. et al. (2024).** Temperature and illumination of Faustini
    and other South-Pole craters. (Faustini thermal/illumination characterization.)
13. **Hayne, P. O. et al. (2017).** Global regolith thermophysical properties of
    the Moon from Diviner. *J. Geophys. Res. Planets* 122, 2371-2400.
    DOI: [10.1002/2017JE005387](https://doi.org/10.1002/2017JE005387)
14. **Schörghofer, N. & Williams, J.-P. (2020).** Mapping of ice storage
    processes on the Moon. *Astrophys. J.* / Planetary thermal-stability modelling.
15. **Koenig, S. & Likhachev, M. (2002).** D* Lite. *AAAI-02*, 476-483.
16. **Karaman, S. & Frazzoli, E. (2011).** Sampling-based algorithms for optimal
    motion planning (RRT*). *Int. J. Robotics Research* 30(7), 846-894.
    DOI: [10.1177/0278364911406761](https://doi.org/10.1177/0278364911406761)
17. **Tompkins, P. et al. (2024).** VIPER traversability and route planning.
    arXiv: [2401.08558](https://arxiv.org/abs/2401.08558)
18. **Hart, P. E., Nilsson, N. J., Raphael, B. (1968).** A formal basis for the
    heuristic determination of minimum cost paths (A*).
    *IEEE Trans. Syst. Sci. Cybern.* 4(2), 100-107.
19. **Nash, A., Daniel, K., Koenig, S., Felner, A. (2007).** Theta*: any-angle
    path planning on grids. *AAAI-07*.
20. **Deb, K. et al. (2002).** A fast and elitist multiobjective genetic
    algorithm: NSGA-II. *IEEE Trans. Evol. Comput.* 6(2), 182-197.
    DOI: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)
21. **Horn, B. K. P. (1981).** Hill shading and the reflectance map.
    *Proc. IEEE* 69(1), 14-47.
22. **Rosenburg, M. A. et al. (2011).** Global surface slopes and roughness of
    the Moon from LOLA. *J. Geophys. Res.* 116, E02001.
    DOI: [10.1029/2010JE003716](https://doi.org/10.1029/2010JE003716)
23. **Kreslavsky, M. A. & Head, J. W. (2000).** Kilometer-scale roughness of
    Mars. *J. Geophys. Res.* 105, 26695-26711.
24. **Cloude, S. R. & Pottier, E. (1997).** An entropy-based classification
    scheme for land applications of polarimetric SAR.
    *IEEE Trans. Geosci. Remote Sens.* 35(1), 68-78.
25. **Looyenga, H. (1965).** Dielectric constants of heterogeneous mixtures.
    *Physica* 31(3), 401-406.
26. **Olhoeft, G. R. & Strangway, D. W. (1975).** Dielectric properties of the
    first 100 m of the Moon. *Earth Planet. Sci. Lett.* 24, 394-404.
27. **Carrier, W. D., Olhoeft, G. R., Mendell, W. (1991).** Physical properties
    of the lunar surface. In *Lunar Sourcebook*, Cambridge Univ. Press.
28. **Heggy, E. et al. (2012).** Radar properties of comparative planetary
    regoliths and implications for ice detection. *Icarus*.
29. **Colaprete, A. et al. (2010).** Detection of water in the LCROSS ejecta
    plume. *Science* 330, 463-468.
    DOI: [10.1126/science.1186986](https://doi.org/10.1126/science.1186986)
30. **Lee, J.-S. (1981).** Refined filtering of image noise using local
    statistics. *Comput. Graph. Image Process.* 15(4), 380-389.
31. **Saaty, T. L. (1980).** *The Analytic Hierarchy Process.* McGraw-Hill.
32. **Fa, W. & Wieczorek, M. A. (2012).** Regolith thickness over the lunar
    nearside from Chang'E-1 and LRO. *Icarus* 218, 771-787.

---

*lunaris — engineered for the ISRO BAH 2026 South-Pole subsurface-ice challenge.
For the analysis-method catalogue see [METHODS.md](METHODS.md); for the data
sources see [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md); for the deeper science
narrative see [docs/SCIENCE.md](docs/SCIENCE.md).*
