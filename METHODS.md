# lunaris — Analysis Methods Catalogue

A numbered, citation-backed catalogue of **every** analysis method implemented (or
specified) in `lunaris` for ISRO BAH 2026 Problem Statement 8. Methods are grouped
into four families. Each entry gives a one-line description, the governing
equation or algorithm, and a peer-reviewed citation. The total is **40 methods**
(comfortably ≥ 30).

| Family | Methods | Count |
|--------|---------|-------|
| A. Radar polarimetry | M1-M12 | 12 |
| B. Illumination & thermal | M13-M22 | 10 |
| C. Terrain, planning & volume | M23-M37 | 15 |
| D. Multi-sensor fusion | M38-M40 | 3 |
| **Total** | | **40** |

---

## A. Radar polarimetry (M1-M12)

Module: `lunaris.polarimetry`, `lunaris.classify`, `lunaris.volume.dielectric`

### M1. Circular Polarisation Ratio (CPR)
The primary same-/opposite-sense backscatter ratio; `> 1` flags candidate ice
(CBOE).
**Equation:** `CPR = SC/OC = (S0 − S3)/(S0 + S3)`.
*Campbell (2012), JGR Planets 117, E06008.* `polarimetry.cpr.circular_polarization_ratio`

### M2. Degree of Polarisation (DOP)
The depolarisation scalar; `< 0.13` rejects polarised rough-surface decoys.
**Equation:** `DOP = m = √(S1² + S2² + S3²)/S0 ∈ [0,1]`.
*Raney et al. (2012), IGARSS.* `polarimetry.cpr.degree_of_polarization`

### M3. Joint CPR-DOP ice criterion (Sinha et al. 2026)
The refined two-scalar subsurface-ice test that resolves the CPR ambiguity.
**Algorithm:** `ice = (CPR > 1) ∧ (DOP < 0.13)`, applied either directly
(`classify_ice_threshold`) or via the O(1) LUT (`classify_ice_lut`).
*Sinha et al. (2026), npj Space Exploration, DOI 10.1038/s44453-026-00038-9.*
`classify.ice_lut`

### M4. m-χ decomposition
Splits power into even/volume/odd channels using ellipticity; the volume channel
dominates over ice.
**Equation:** with `χ = ½·arcsin(−S3/(m·S0))`,
`even = √(m·S0·(1+sin2χ)/2)`, `volume = √(S0·(1−m))`, `odd = √(m·S0·(1−sin2χ)/2)`.
*Raney et al. (2012), IGARSS.* `polarimetry.decomposition.m_chi`

### M5. m-δ decomposition
Double/volume/surface split using the relative phase δ instead of ellipticity.
**Equation:** with `δ = arctan2(S3, S2)`,
`double = √(m·S0·(1+sinδ)/2)`, `volume = √(S0·(1−m))`, `surface = √(m·S0·(1−sinδ)/2)`.
*Raney et al. (2012), IGARSS.* `polarimetry.decomposition.m_delta`

### M6. Stokes child parameters (CHILD)
Hybrid-pol descriptors that characterise the scattering ellipse.
**Equation:** `χ = ½·arcsin(−S3/(m·S0))`, `δ = arctan2(S3, S2)`,
`ψ = ½·arctan2(S2, S1)`, `conformity = −S3/S0`.
*Kumar et al. (2015), CHILD parameters.* `polarimetry.decomposition.child_parameters`

### M7. SC / OC same-/opposite-sense powers
The constituent circular powers behind CPR.
**Equation:** `SC = (S0 − S3)/2`, `OC = (S0 + S3)/2`; `CPR = SC/OC`.
*Raney et al. (2012), IGARSS.* `polarimetry.cpr.sc_oc_power`

### M8. Cloude-Pottier H/A/α eigen-decomposition
Entropy/anisotropy/mean-scattering-angle from the 2×2 circular coherency matrix.
**Algorithm:** eigen-decompose `J = ⟨k·kᴴ⟩`, `k = [E_RH, E_RV]ᵀ`; with
`pᵢ = λᵢ/Σλ`: `H = −Σpᵢ·log₂pᵢ`, `A = (λ₁−λ₂)/(λ₁+λ₂)`, `α = Σpᵢ·αᵢ`.
*Cloude & Pottier (1997), IEEE TGRS 35(1).* `polarimetry.decomposition.cloude_pottier`

### M9. Speckle-filtered backscatter σ⁰ (boxcar & refined-Lee)
Reduce multiplicative SAR speckle before parameter estimation.
**Algorithm:** boxcar = uniform `n×n` mean (ENL↑ by `n²`); refined-Lee uses
edge-aligned local-statistics minimum-MSE weighting.
*Lee (1981), CGIP 15(4); Lee et al. (2009), IEEE TGRS 47(1).*
`polarimetry.speckle.boxcar`, `polarimetry.speckle.refined_lee`

### M10. Two-layer / CBOE scattering model
Interprets `CPR > 1` as the coherent-backscatter opposition effect of a buried
low-loss ice volume versus a surface-roughness decoy.
**Model:** same-sense enhancement from multiple scattering in a low-loss volume;
discriminated from roughness by the DOP gate (M2-M3).
*Spudis et al. (2013), JGR 118; Fa & Cahill (2013), JGR 118.*
(physical basis behind `classify`)

### M11. Dielectric mixing & penetration depth
Map effective permittivity to ice fraction and bound the sampled column.
**Equation (Looyenga):** `f = (ε_eff^⅓ − ε_reg^⅓)/(ε_ice^⅓ − ε_reg^⅓)`,
`ε_ice ≈ 3.15`; **penetration:** `δ = λ/(2π·√ε·tanδ)` (≈ 4.4 m at L-band).
*Looyenga (1965), Physica 31; Olhoeft & Strangway (1975), EPSL.*
`volume.dielectric.looyenga_ice_fraction`, `volume.dielectric.penetration_depth`

### M12. Polarimetric coherence + L-S contrast
Co-pol coherence (low over volume scatter) and the dual-frequency L-vs-S
signature differencing.
**Equation:** `γ_hhvv = |⟨Shh·Svv*⟩| / √(⟨|Shh|²⟩·⟨|Svv|²⟩) ∈ [0,1]`; the L−S CPR
contrast separates deep (L) from shallow (S) scattering.
*Campbell (2012), JGR Planets 117.* `polarimetry.coherence.copol_coherence`

---

## B. Illumination & thermal (M13-M22)

Module: `lunaris.terrain.illumination`, `lunaris.terrain.thermal`

### M13. Horizon PSR mapping
Per-pixel maximum horizon-elevation angle by DEM raytracing; basis for all
shadowing.
**Algorithm:** trace `n_azimuth` rays outward, record max elevation angle to the
local horizon → `horizon(x, azimuth)`.
*Mazarico et al. (2011), Icarus 211.* `terrain.illumination.horizon_map`

### M14. Permanently-shadowed-region (PSR) mask
A pixel never lit if the Sun stays below its horizon all year.
**Algorithm:** `PSR = max_azimuth(horizon_angle) > sun_max_elev`, with
`sun_max_elev ≤ MOON_OBLIQUITY_DEG = 1.543°`.
*Mazarico et al. (2011), Icarus 211.* `terrain.illumination.permanent_shadow_mask`

### M15. Double-shadow raycast
Within a PSR, pixels also shadowed from the surrounding sunlit rim (no secondary
light) — the coldest traps.
**Algorithm:** PSR pixels for which every ray to an illuminated rim cell is also
horizon-blocked.
*Paige et al. (2010), Science 330.* `terrain.illumination.double_shadow_mask`

### M16. Secondary-illumination view-factor
Sky-view factor quantifying the hemispherical sky fraction that drives radiative
cooling and scattered-light budgets.
**Equation:** `SVF = 1 − ⟨sin²(horizon_angle)⟩` averaged over azimuth.
*Mazarico et al. (2011), Icarus 211.* `terrain.illumination.sky_view_factor`

### M17. Diviner cold-trap mask
Flags water-ice-stable pixels by annual maximum temperature.
**Equation:** `cold_trap = (Tmax < 110 K)` for H₂O (60/66/40 K for SO₂-CO₂/NH₃/
super-volatiles).
*Paige et al. (2010), Science 330.* `terrain.thermal.cold_trap_mask`

### M18. Hertz-Knudsen sublimation
Free-space water-ice mass-loss rate bounding ice longevity at a given T.
**Equation:** `E = P_sv(T)·√(M/(2π·R·T))` [kg m⁻² s⁻¹], `P_sv` from
Clausius-Clapeyron, `M = 0.018015 kg mol⁻¹`.
*Schörghofer & Williams (2020); standard kinetic theory.*
`terrain.thermal.sublimation_rate`

### M19. Ice-stability depth
Burial depth below which ice is stable where the surface is too warm.
**Algorithm:** invert the regolith thermal gradient for the depth at which
`T(z) < T_WATER_STABLE`; 0 where stable at the surface.
*O'Brien & Byrne (2022), JGR Planets 127.* `terrain.thermal.ice_stability_depth`

### M20. 1-D regolith thermal profile
Conductive `T(z)` for a column under a surface boundary condition and the lunar
geothermal flux.
**Equation:** solve `∂T/∂t = ∂/∂z(κ ∂T/∂z)` with basal flux
`GEOTHERMAL_FLUX = 0.018 W m⁻²` → `T(z)`.
*Hayne et al. (2017), JGR Planets 122.* `terrain.thermal.regolith_thermal_profile`

### M21. LOLA 1064 nm albedo frost proxy
Anomalously high normal albedo at 1064 nm in PSRs indicates surface frost.
**Indicator:** elevated `albedo_1064` co-located with cold traps → frost. Used as
a fusion feature.
*Fisher et al. (2017) / LOLA reflectance; Mazarico et al. (2011).*
(`scene.albedo_1064` feature in `classify.fusion`)

### M22. LAMP off/on-band UV frost
Far-UV Lyman-α off-/on-band ratio and Lyman-α dimming diagnose surface water
frost.
**Indicator:** `lamp_ratio` (off/on-band) anomaly → frost; complements albedo and
sky-view. Used as a fusion feature.
*Gladstone et al. (2012) LAMP; LRO LAMP PSR studies.*
(`scene.lamp_ratio` feature in `classify.fusion`)

### (also) Earth-visibility / comms mapping
Fraction of time Earth is above the local horizon (direct-to-Earth comms
windows), folded into landing/traverse criteria.
**Algorithm:** raytrace toward Earth's bounded libration position → visible
fraction ∈ [0,1].
*Mazarico et al. (2011), Icarus 211.* `terrain.illumination.earth_visibility_map`

---

## C. Terrain, planning & volume (M23-M37)

Module: `lunaris.terrain.dem`, `lunaris.terrain.boulders`, `lunaris.planning`,
`lunaris.volume`

### M23. Horn slope (and aspect)
3×3 finite-difference terrain gradient → slope magnitude and downslope azimuth.
**Equation:** `dz/dx = ((c+2f+i)−(a+2d+g))/(8·res)`,
`dz/dy = ((g+2h+i)−(a+2b+c))/(8·res)`, `slope = atan(hypot(dz/dx, dz/dy))`.
*Horn (1981), Proc. IEEE 69(1).* `terrain.dem.slope_horn`, `terrain.dem.aspect`

### M24. Multi-scale RMS roughness (deviogram)
Per-pixel RMS-deviation roughness at a chosen baseline.
**Equation:** `ν(dx) = √⟨[z(x+dx) − z(x)]²⟩` over four axis directions, smoothed
over `(2dx+1)`.
*Rosenburg et al. (2011), JGR 116, E02001.* `terrain.dem.rms_roughness`

### M25. Hurst exponent (self-affine scaling)
Global self-affine roughness exponent from deviogram log-log scaling.
**Equation:** `ν(dx) ∝ dx^H`; `H = slope of log ν vs log dx` (polyfit).
*Rosenburg et al. (2011), JGR 116, E02001.* `terrain.dem.hurst_exponent`

### M26. IQR roughness (boulder-robust)
Median-based roughness insensitive to boulder/spike outliers.
**Equation:** `IQR = Q₇₅(z) − Q₂₅(z)` in a moving window.
*Kreslavsky & Head (2000), JGR 105.* `terrain.dem.iqr_roughness`

### M27. Shadow-length boulder detection
Detect boulders from cast shadows in OHRC/NAC imagery and infer their height.
**Equation:** threshold dark shadow blob of length `L` → `h = L·tan(sun_elev)`;
diameter from the sunlit cap.
*Pajola et al. / standard shadow-photoclinometry.*
`terrain.boulders.detect_boulders_shadow`

### M28. CNN / blob boulder & density mapping
Rasterise detected boulders into a spatial hazard-density map (count per window),
extensible to a learned detector.
**Algorithm:** kernel-density of boulder centroids over an `n×n` window →
`boulder_density`.
*Bickel et al. (2020) deep boulder mapping (extension path).*
`terrain.boulders.boulder_density_map`

### M29. Crater / rim morphology
Bowl-profile crater & rim characterisation feeding double-shadow geometry and
hazard context.
**Model:** paraboloid floor + raised Gaussian rim; rim/floor extraction from the
DEM. (Synthetic generator encodes this; real path fits it.)
*Pike (1977) crater morphometry.* (`io.synthetic`, terrain context)

### M30. MCDA landing suitability (WLC + AHP)
Weighted-linear-combination multi-criteria scoring of landing safety/value.
**Equation:** `suitability = Σ_k w_k·normalise(layer_k)`; weights `w_k` from the
AHP principal eigenvector of a pairwise matrix (consistency ratio < 0.1).
*Saaty (1980), Analytic Hierarchy Process.*
`planning.landing.landing_suitability`, `planning.landing.ahp_weights`,
`planning.landing.select_landing_sites`

### M31. A\* shortest-path planner
Optimal grid path with an admissible heuristic on the hazard cost grid.
**Algorithm:** binary-heap best-first search; edge = geometric step × mean cell
cost; octile heuristic × min-cell-cost (admissible).
*Hart, Nilsson & Raphael (1968), IEEE TSSC 4(2).* `planning.astar.astar`

### M32. D\* Lite incremental replanner
Repairs the shortest path when edge costs change (newly detected hazards) without
full replanning.
**Algorithm:** maintain `g`/`rhs` values and a priority queue keyed by
`[min(g,rhs)+h+k_m, min(g,rhs)]`; locally repair on cost updates.
*Koenig & Likhachev (2002), AAAI-02.* `planning.dstar_lite.DStarLite`

### M33. Theta\* / Field D\* any-angle planner
Shorter, more natural any-angle paths by allowing line-of-sight parent
shortcutting.
**Algorithm:** A\* variant where a node may take its grandparent as parent if
line-of-sight is clear (Bresenham/LOS check).
*Nash et al. (2007), AAAI-07.* `planning.theta_star.theta_star`

### M34. RRT\* sampling-based planner
Asymptotically-optimal sampling planner for large continuous traverse spaces.
**Algorithm:** incremental random tree with rewiring within a shrinking radius
`r(n) ∝ (log n / n)^{1/d}` → converges to the optimum.
*Karaman & Frazzoli (2011), IJRR 30(7).* `planning.rrt_star.rrt_star`

### M35. NSGA-II multi-objective optimisation
Pareto-front of competing traverse objectives (distance vs energy vs hazard vs
science).
**Algorithm:** non-dominated sorting + crowding-distance selection over evolving
candidate paths → non-dominated set.
*Deb et al. (2002), IEEE TEC 6(2).* `planning.nsga2.nsga2_paths`

### M36. Energy-aware ⟨cell, SOC⟩ planner
Couples the planner with the rover energy model so battery state-of-charge never
violates the 70-h shadow-survival limit.
**Equations:** `P_solar = S·A·η·max(sin elev, 0)`;
`E_drive/m = m·g·(sin θ + Crr·cos θ)`; `t_survive = battery_Wh/load_W`; plan over
the augmented state `⟨cell, SOC⟩` with charging stops in lit waypoints.
*Tompkins et al. (2024), arXiv:2401.08558 (VIPER traversability).*
`planning.energy.energy_aware_plan` (+ `solar_power`, `drive_energy_per_m`,
`survival_time_h`); cost grid `planning.cost.build_cost_grid`

### M37. Dielectric-mixing inversion + volumetric integration + Monte-Carlo
Top-5 m ice volume/mass from area × depth × fraction, with full uncertainty
propagation.
**Equation:** `f` from Looyenga (M11); `V = A·d·f`; `M = ρ·V`; Monte-Carlo draws
`(A, d, f)` from their `*_std` distributions → `mean, σ, 95% CI`.
*Looyenga (1965), Physica 31; Colaprete et al. (2010), Science 330 (ground truth).*
`volume.estimate.ice_volume`, `volume.estimate.ice_mass`,
`volume.estimate.monte_carlo_volume`

---

## D. Multi-sensor fusion (M38-M40)

Module: `lunaris.classify.fusion`, `lunaris.classify.evidence`

### M38. Supervised RandomForest fusion
Learned, non-linear ice probability from the 10-feature multi-sensor stack;
bakeable into the O(1) LUT.
**Algorithm:** balanced 200-tree `RandomForestClassifier` on
`[cpr_L, dop_L, cpr_S, dop_S, mchi_volume, temperature_max, illumination,
albedo_1064, lamp_ratio, earth_visibility]` → `P(ice)` + feature importance.
*Breiman (2001), Random Forests.*
`classify.fusion.train_ice_classifier`, `classify.fusion.predict_ice`,
`classify.fusion.bake_lut_from_model`

### M39. Bayesian (log-odds) evidence fusion
Pool conditionally-independent evidence probabilities into a posterior.
**Equation:** `logit(P) = Σ_k w_k·logit(p_k)`; `P = σ(logit)`.
*Bayesian inference / log-odds pooling.* `classify.evidence.bayesian_fusion`

### M40. Dempster-Shafer evidence combination
Combine belief masses toward "ice" across uncertain/partially-conflicting
sources, representing ignorance explicitly.
**Algorithm:** Dempster's rule of combination on basic belief assignments toward
the ice hypothesis → combined belief.
*Shafer (1976), A Mathematical Theory of Evidence.*
`classify.evidence.dempster_shafer`

---

## Method-to-deliverable traceability

| PS8 deliverable | Methods |
|-----------------|---------|
| D1 — map ice regions | M1-M12, M38-M40 |
| D2 — validated detection framework | M3-M12, M38-M40 |
| D3 — landing site | M13-M30 |
| D4 — rover traverse | M23-M28, M31-M36 |
| D5 — top-5 m ice volume | M11, M17-M20, M37 |

*See [ARCHITECTURE.md](ARCHITECTURE.md) for how these methods compose into the
nine-stage pipeline, [docs/SCIENCE.md](docs/SCIENCE.md) for the scientific
narrative, and [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for the datasets each
method consumes.*
