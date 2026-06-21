<h1 align="center">🌑 lunaris</h1>

<p align="center">
  <b>Lunar Subsurface-Ice Detection &amp; Mission-Planning Platform</b><br/>
  <i>ISRO Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement 8</i>
</p>

<p align="center">
  <code>Python 3.11+</code> · <code>MIT</code> · <code>src-layout</code> ·
  <code>Chandrayaan-2 DFSAR</code> · <code>30+ datasets</code> ·
  <code>40 methods</code> · <code>O(1) ice classifier</code>
</p>

<p align="center">
  <a href="https://github.com/divyamohan1993/bah2026-ps8/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/divyamohan1993/bah2026-ps8/actions/workflows/ci.yml/badge.svg"/>
  </a>
</p>

---

## The problem

Find water-ice beneath a lunar **South-Pole doubly shadowed crater** — the coldest,
best-preserved volatile traps on the Moon — using Chandrayaan-2 **Dual-Frequency
Synthetic Aperture Radar (DFSAR)**, then turn that detection into an actionable
exploration plan. PS8 demands five things: map the ice, validate the detection,
propose a landing site, plan a rover traverse under terrain and solar-power
constraints, and estimate the top-5 m ice volume. The target is a doubly shadowed
crater on the floor of **Faustini** (87.3° S, 77.0° E — the Moon's third-largest PSR).

## What lunaris does — the five deliverables

| # | Deliverable | How |
|---|-------------|-----|
| **1** | **Map subsurface-ice regions** | DFSAR Stokes → **CPR > 1 ∧ DOP < 0.13** (Sinha et al. 2026) via an **O(1) look-up-table** classifier |
| **2** | **Validated detection framework** | m-χ/m-δ decomposition, Cloude-Pottier H/A/α, multi-sensor cross-verification that defeats roughness false positives |
| **3** | **Feasible landing site** | slope / roughness / illumination / Earth-visibility → AHP-weighted MCDA |
| **4** | **Energy-aware rover traverse** | hazard cost grid → A\* / D\* Lite / Theta\* / RRT\* / NSGA-II + a solar/battery model with **70-h shadow survival** |
| **5** | **Top-5 m ice-volume estimate** | Looyenga dielectric inversion → `V = A·d·f` → Monte-Carlo uncertainty |

Everything is produced by one command and rendered into a **single self-contained
HTML report**.

## Architecture at a glance

```
CONFIG ─► ① INGEST ─► ② STOKES→CPR/DOP ─► ③ m-χ/m-δ ─► ④ O(1) ICE LUT ─► ⑤ TERRAIN
                                                                              │
   HTML REPORT ◄─ ⑨ VOLUME+CI ◄─ ⑧ TRAVERSE ◄─ ⑦ LANDING MCDA ◄─ ⑥ FUSION ◄──┘
```

A nine-stage pipeline over a shared, co-registered `LunarScene`. Full design,
equations and module map in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quickstart

```bash
# 1. create an isolated environment and install (editable)
python -m venv .venv
.venv/bin/pip install -e .

# 2. run the self-contained demo — synthesises the Faustini scene and prints
#    its ice-criterion summary (no external data needed)
.venv/bin/lunaris demo

# 3. run the full pipeline on the Faustini config
.venv/bin/lunaris run --config configs/faustini.yaml

# 4. (optional) run the test suite
.venv/bin/python -m pytest -q

# 5. (optional, needs network) validate on a REAL LOLA south-pole DEM —
#    O(1) windowed read of a polar Cloud-Optimized GeoTIFF, then the genuine
#    terrain / illumination / landing / traverse algorithms on real topography
.venv/bin/lunaris realdata --out outputs/realdata --extent-km 120
```

The **HTML report** lands in `outputs/faustini/` (the `outputs_dir` set in
`configs/faustini.yaml`), alongside GIS-loadable Cloud-Optimized GeoTIFFs of every
scene layer. With no Chandrayaan-2 granules present, the pipeline runs fully
offline on a **deterministic synthetic Faustini scene**; drop real DFSAR/OHRC data
into `data/raw/faustini/` to switch to the real-data path.

## The O(1) / "fastest platform" highlights

`lunaris` is engineered to be the *fastest* lunar ice-detection pipeline — and it
is **precise and honest** about what that means (full ledger in
[ARCHITECTURE.md](ARCHITECTURE.md) §4):

* **TRUE O(1) per pixel** — the `(CPR, DOP)` ice classifier bakes the decision
  boundary into a precomputed boolean table; each pixel is one `digitize` +
  **one** fancy-index gather, *provably bit-identical* to the direct rule.
* **O(1) windowed COG reads** — extract one crater from a continent-scale mosaic
  via HTTP byte-range requests, never the whole image.
* **O(1)** H3 64-bit bitwise hex indexing · `np.memmap` page access · joblib cache
  hits.
* **O(log n)** spatial search (R-tree / cKDTree) — clearly distinguished, not
  conflated.
* **Constant-factor** numba/vectorisation — explicitly *not* asymptotic.

## 40 methods · 30+ datasets

* **40 analysis methods** across radar polarimetry, illumination/thermal, terrain,
  planning, volume and multi-sensor fusion — each with its governing equation and a
  peer-reviewed citation: **[METHODS.md](METHODS.md)**.
* **30+ multi-sensor datasets** (26 curated, fusion-ready) spanning radar, optical,
  IR, thermal, UV, neutron and topography — so *independent physics defeats radar
  false positives*: **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)**.

## Repository layout

```
bah2026-ps8/
├── ARCHITECTURE.md          deep design, pipeline, equations, references
├── METHODS.md               the 40-method catalogue
├── README.md                this file
├── docs/
│   ├── SCIENCE.md           PSRs, Faustini, cold traps, the CPR controversy
│   └── DATA_SOURCES.md      the dataset catalogue, portals, CRS, DEMs
├── configs/                 faustini.yaml, default.yaml
├── src/lunaris/
│   ├── constants.py         single source of truth for all physics
│   ├── config.py            typed Settings (+ YAML/env)
│   ├── scene.py             LunarScene dataclass (+ COG save/load)
│   ├── pipeline.py          run_pipeline() — nine-stage orchestrator
│   ├── cli.py               `lunaris run | demo | version`
│   ├── io/                  readers (COG windowed) · synthetic · registry · cache
│   ├── polarimetry/         stokes · cpr · decomposition · speckle · coherence
│   ├── classify/            ice_lut (O(1)) · fusion (RandomForest) · evidence
│   ├── terrain/             dem · illumination · thermal · boulders
│   ├── planning/            cost · astar · dstar_lite · theta_star · rrt_star · nsga2 · energy · landing
│   ├── volume/              dielectric · estimate
│   └── viz/                 maps · charts · terrain3d · report
└── tests/                   pytest suite (foundation, polarimetry, classify, ...)
```

## Documentation

| Document | What's inside |
|----------|---------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | executive summary, nine-stage pipeline, module map, the O(1) engineering, the science with equations, fusion strategy, data flow, tech stack, limitations, 32 references |
| **[METHODS.md](METHODS.md)** | the 40 analysis methods, each with equation + citation, traced to deliverables |
| **[docs/SCIENCE.md](docs/SCIENCE.md)** | PSRs & doubly shadowed craters, Faustini, cold-trap physics, the CPR ambiguity and how fusion resolves it, ISRU/Artemis/LUPEX relevance |
| **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** | the 26-dataset catalogue, access portals, south-polar CRS, recommended DEMs |

## Data-source credits

Built on open planetary archives: **ISRO/ISSDC PRADAN** (Chandrayaan-1/2),
**NASA PDS Geosciences Node** (LRO, Lunar Prospector, GRAIL), **NASA GSFC PGDA**
(LOLA DEMs & illumination), **ASU LROC**, **KPLO/Danuri ShadowCam**, **JAXA DARTS**
(Kaguya/SELENE), **USGS Astrogeology** (STAC/COG), and the **LCROSS** impact
ground truth. See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for per-mission
links and citation terms.

## License

MIT (see `pyproject.toml`). Dataset products remain subject to their respective
mission/archive citation and acknowledgement terms.

---

<p align="center"><i>lunaris — engineered for the ISRO BAH 2026 South-Pole subsurface-ice challenge.</i></p>
