# lunaris — Data Sources

The multi-sensor evidence base for ISRO BAH 2026 Problem Statement 8. `lunaris`
fuses **30+ datasets** (26 curated records in `src/lunaris/io/registry.py`,
extensible) so that *independent physics defeats radar false positives*
(see [docs/SCIENCE.md](SCIENCE.md) and [ARCHITECTURE.md](../ARCHITECTURE.md) §6).

All products are co-registered into the lunar **south-polar stereographic** CRS
on a sphere of radius `R = 1 737 400 m` (ESRI:103878):

```
+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs
```

---

## 1. The dataset catalogue (26 curated records)

Sourced verbatim from `lunaris.io.registry.DATASETS`. Every entry contributes an
ice indicator and/or a planning constraint. *Mission / instrument / measurement /
ice indicator / resolution / format / access*.

### Chandrayaan-2 (primary mission for this PS)

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `ch2_dfsar` | DFSAR L&S-band polarimetric SAR | DFSAR | L 1.25 GHz & S 2.5 GHz full-pol SAR | **CPR>1 & DOP<0.13 → subsurface ice**; m-χ volume scatter | 2-75 m | PDS4 | open | [PRADAN](https://pradan.issdc.gov.in/ch2/) |
| `ch2_ohrc` | Orbiter High-Resolution Camera | OHRC | Panchromatic optical | Boulders/roughness & shadow geometry for double-shadow ID | 0.25 m | PDS4 | open | [PRADAN](https://pradan.issdc.gov.in/ch2/) |
| `ch2_iirs` | Imaging Infrared Spectrometer | IIRS | 0.8-5.0 µm hyperspectral | 3 µm OH/H₂O absorption → surface hydration | ~80 m | PDS4 | open | [PRADAN](https://pradan.issdc.gov.in/ch2/) |
| `ch2_xsm_class` | CLASS X-ray fluorescence | CLASS | Elemental (Na,Mg,Al,Si,Ca,Fe) | FeO/TiO₂ → regolith loss-tangent for depth model | ~12.5 km | PDS4 | open | [PRADAN](https://pradan.issdc.gov.in/ch2/) |
| `ch2_tmc2` | Terrain Mapping Camera-2 DEM | TMC-2 | Stereo optical → DEM | High-res topography for PSR & slope mapping | 5 m | PDS4 | open | [PRADAN](https://pradan.issdc.gov.in/ch2/) |

### Chandrayaan-1

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `ch1_minisar` | Mini-SAR S-band radar | Mini-SAR | S 2.38 GHz hybrid-pol SAR | Anomalous high-CPR craters → candidate polar ice | ~150 m | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/chandrayaan1/) |
| `ch1_m3` | Moon Mineralogy Mapper (M3) | M3 | 0.43-3.0 µm imaging spectroscopy | 3.0 µm ice absorption → exposed surface ice in PSRs | 140 m | PDS3 | open | [PDS Imaging](https://pds-imaging.jpl.nasa.gov/volumes/m3.html) |

### Lunar Reconnaissance Orbiter (LRO)

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `lro_lola` | LOLA DEM | LOLA | 1064 nm altimetry / topo & normal albedo | High 1064 nm albedo in PSRs → frost; DEM → PSR | 5-20 m (polar) | GeoTIFF | open | [PGDA](https://pgda.gsfc.nasa.gov/products/90) |
| `lro_lola_illum` | LOLA illumination / PSR maps | LOLA | Average solar illumination & permanent-shadow fraction | PSR & doubly-shadowed delineation | 240 m / 20 m | GeoTIFF | open | [PGDA](https://pgda.gsfc.nasa.gov/products/69) |
| `lro_diviner` | Diviner Lunar Radiometer | Diviner | Thermal IR; max/annual T | **Tmax < 110 K → H₂O cold trap** (Gyr stability) | ~240 m | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lro/diviner.htm) |
| `lro_minirf` | Mini-RF S/X-band radar | Mini-RF | S 2.38 GHz hybrid-pol SAR | Elevated CPR in PSR crater interiors → ice | 15-30 m | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lro/mrf.htm) |
| `lro_lroc_nac` | LROC Narrow-Angle Camera | LROC NAC | Panchromatic optical | Boulder mapping, slope context, secondary-light PSR imaging | 0.5 m | PDS3 | open | [LROC](https://wms.lroc.asu.edu/lroc) |
| `lro_lroc_wac` | LROC Wide-Angle Camera | LROC WAC | VIS-UV multispectral | Regional photometry & illumination context | 100 m | PDS3 | open | [LROC](https://wms.lroc.asu.edu/lroc) |
| `lro_lamp` | Lyman-Alpha Mapping Project | LAMP | Far-UV (Lyman-α) albedo & off/on-band ratio | Off-/on-band UV ratio & Lyman-α dimming → surface frost | ~240 m | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lro/lamp.htm) |
| `lro_lend` | Lunar Exploration Neutron Detector | LEND | Epithermal neutron flux | Epithermal suppression → buried hydrogen (WEH) | ~10 km (collimated) | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lro/lend.htm) |
| `lola_slope` | LOLA-derived slope & roughness | LOLA (derived) | Slope, RMS roughness, Hurst exponent | Landing/traverse safety; roughness for CPR false-positive ID | 20-60 m | GeoTIFF | open | [PGDA](https://pgda.gsfc.nasa.gov/products/90) |
| `lola_earthvis` | LOLA Earth-visibility / comms maps | LOLA (derived) | Fraction of time Earth above horizon | Direct-to-Earth comms windows for rover ops | 240 m | GeoTIFF | open | [PGDA](https://pgda.gsfc.nasa.gov/products/78) |
| `diviner_rocka` | Diviner rock abundance & regolith T | Diviner (derived) | Rock-abundance fraction & regolith Tbol | Surface rock fraction → radar roughness-decoy context | ~240 m | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lro/diviner.htm) |
| `lola_psr_temp` | LOLA+Diviner thermal-stability model | LOLA+Diviner | Modelled ice-stability depth | Depth to ice-stable layer (top-5 m volume modelling) | 240 m | GeoTIFF | open | [PGDA](https://pgda.gsfc.nasa.gov/products/69) |

### Lunar Prospector

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `lp_ns` | Neutron Spectrometer | NS | Thermal/epithermal/fast neutron flux | Polar epithermal suppression → first global H/ice evidence | ~45 km | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/lunarp/) |

### Kaguya / SELENE

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `kaguya_tc` | Terrain Camera DEM | TC | Stereo optical → 10 m DEM | High-res topography for slope/illumination | 10 m | PDS3 | open | [JAXA DARTS](https://darts.isas.jaxa.jp/planet/pdap/selene/) |
| `kaguya_mi` | Multiband Imager | MI | VIS-NIR 9-band reflectance | Maturity & FeO/TiO₂ → regolith dielectric / loss tangent | 20 m | PDS3 | open | [JAXA DARTS](https://darts.isas.jaxa.jp/planet/pdap/selene/) |

### GRAIL

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `grail_gravity` | Lunar gravity field | LGRS | Spherical-harmonic gravity / crustal density | Crustal porosity/density context for subsurface modelling | ~5 km | PDS3 | open | [PDS Geosciences](https://pds-geosciences.wustl.edu/missions/grail/) |

### Danuri (KPLO)

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `danuri_shadowcam` | ShadowCam (PSR-optimised imager) | ShadowCam | High-sensitivity panchromatic (secondary illumination) | Direct PSR-floor imaging of frost brightness & morphology | 1.7 m | PDS4 | open | [ShadowCam PDS](https://pds.shadowcam.im-ldi.com/) |

### Earth-based radar

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `arecibo_radar` | Arecibo 12.6 cm lunar radar | S-band radar | 12.6 cm dual-pol backscatter & CPR | CPR mapping of polar craters; rough-vs-ice disambiguation | 20-80 m | FITS | open | [Arecibo lunar](https://www.naic.edu/~smarshal/lunar.html) |

### Ground truth

| Key | Product | Instrument | Measurement | Ice indicator | Resolution | Format | Access | URL |
|-----|---------|-----------|-------------|---------------|-----------|--------|--------|-----|
| `lcross` | LCROSS Centaur impact volatiles | Shepherding Spacecraft suite | Impact-ejecta NIR/UV spectroscopy | **Direct detection: 5.6 ± 2.9 wt% water in ejecta** (ground truth) | point (Cabeus) | PDS3 | open | [PDS LCROSS](https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LCROSS) |

**Total: 26 curated datasets.** The catalogue is the programmatic backbone of the
"30+ datasets" claim and is queryable via `list_datasets()`, `get_dataset(key)`
and `as_table()`.

### Coverage by evidence type

| Evidence class | Datasets |
|----------------|----------|
| Radar polarimetry | `ch2_dfsar`, `ch1_minisar`, `lro_minirf`, `arecibo_radar` (4) |
| Optical / high-res imaging | `ch2_ohrc`, `lro_lroc_nac`, `lro_lroc_wac`, `danuri_shadowcam` (4) |
| IR / NIR spectroscopy | `ch2_iirs`, `ch1_m3`, `kaguya_mi` (3) |
| Topography / DEM | `lro_lola`, `ch2_tmc2`, `kaguya_tc`, `lola_slope` (4) |
| Illumination / comms | `lro_lola_illum`, `lola_earthvis`, `lola_psr_temp` (3) |
| Thermal | `lro_diviner`, `diviner_rocka` (2) |
| UV frost | `lro_lamp` (1) |
| Neutron / hydrogen | `lro_lend`, `lp_ns` (2) |
| Composition (dielectric) | `ch2_xsm_class`, `kaguya_mi` (2) |
| Gravity / structure | `grail_gravity` (1) |
| Ground truth | `lcross` (1) |

---

## 2. Access portals

| Portal | Holds | URL |
|--------|-------|-----|
| **PRADAN** (ISSDC) | Chandrayaan-1/2 archives (DFSAR, OHRC, IIRS, TMC-2, CLASS) | https://pradan.issdc.gov.in/ch2/ |
| **PDS Geosciences Node** | LRO (Diviner, Mini-RF, LAMP, LEND), Lunar Prospector, GRAIL, Chandrayaan-1 | https://pds-geosciences.wustl.edu/ |
| **PDS ODE REST API** | Programmatic query/download across PDS Geosciences products | https://oderest.rsl.wustl.edu/ |
| **PGDA** (Planetary Geodynamics, NASA GSFC) | LOLA polar DEMs, illumination/PSR, Earth-visibility, thermal-stability | https://pgda.gsfc.nasa.gov/ |
| **LROC** (ASU) | LROC NAC/WAC imagery & WMS | https://wms.lroc.asu.edu/lroc |
| **ShadowCam PDS** (KPLO) | ShadowCam PSR-floor imagery | https://pds.shadowcam.im-ldi.com/ |
| **JAXA DARTS** | Kaguya/SELENE TC, MI, and other SELENE products | https://darts.isas.jaxa.jp/planet/pdap/selene/ |
| **USGS Astrogeology STAC** | Cloud-Optimized GeoTIFF mosaics with a STAC index | https://stac.astrogeology.usgs.gov/ |
| **Moon Trek** (NASA/JPL) | Browse/visualise & export lunar mosaics and DEMs | https://trek.nasa.gov/moon/ |

---

## 3. Coordinate reference system

| Property | Value |
|----------|-------|
| Projection | Lunar south-polar stereographic |
| Latitude of origin | −90° |
| Central meridian | 0° |
| Sphere radius `R` | 1 737 400 m |
| Authority code | **ESRI:103878** (Moon 2000 south-polar stereographic) |
| PROJ4 | `+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs` |

Reprojection into this CRS is performed by
`lunaris.io.readers.reproject_to_south_polar` (constant `SOUTH_POLAR_STEREO_PROJ4`
/ `SOUTH_POLAR_STEREO_ESRI` in `lunaris.constants`).

---

## 4. Recommended South-Pole DEMs

| DEM | Source | Resolution | Format | Notes |
|-----|--------|-----------|--------|-------|
| **LOLA polar DEM (PGDA #90)** | NASA GSFC PGDA | **20 m** (COG) | Cloud-Optimized GeoTIFF | Primary regional topography; supports O(1) windowed reads |
| **LOLA polar DEM (high-res)** | NASA GSFC PGDA | **5 m** (COG, where available) | Cloud-Optimized GeoTIFF | Focused crater-scale slope/roughness |
| **Chandrayaan-2 TMC-2 DEM** | PRADAN | 5 m | PDS4 → COG | Mission-native high-res stereo DEM |
| **Kaguya TC DEM** | JAXA DARTS | 10 m | PDS3 → COG | Independent stereo cross-check |
| **LOLA illumination / PSR (PGDA #69)** | NASA GSFC PGDA | 240 m / 20 m | GeoTIFF | PSR & doubly-shadowed delineation inputs |

> **Working-format recommendation.** Convert any PDS DEM to **Cloud-Optimized
> GeoTIFF (COG)** once, then access AOIs with `read_cog_window(url, bounds)` for
> O(1) windowed reads (see [ARCHITECTURE.md](../ARCHITECTURE.md) §4 & §7).

---

## 5. Data access & licensing notes

* All catalogued products are **open** (NASA PDS, ISSDC PRADAN, JAXA DARTS, USGS
  are public archives). Confirm the per-mission citation/acknowledgement terms.
* `lunaris` runs fully **offline** on a deterministic synthetic Faustini scene
  (`io.synthetic.generate_faustini_scene`) when no granules are present, so the
  pipeline (and tests) need no network access to demonstrate end to end.
* Chandrayaan-2 DFSAR/OHRC granules supplied for the hackathon drop into
  `data/raw/faustini/` and are read by the real-data path
  (`io.readers.read_cog_window` + `reproject_to_south_polar`).

*See [ARCHITECTURE.md](../ARCHITECTURE.md) §6-7 for how these sources are fused,
and [docs/SCIENCE.md](SCIENCE.md) for the physics each indicator encodes.*
