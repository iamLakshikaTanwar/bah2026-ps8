"""Multi-sensor dataset catalogue — the "30+ datasets" fusion registry.

Each entry is a :class:`Dataset` describing one remote-sensing product that
contributes evidence to lunar South-Pole subsurface-ice detection and
mission-planning. The catalogue spans radar (Chandrayaan-2 DFSAR,
Chandrayaan-1 Mini-SAR, LRO Mini-RF, Arecibo), optical/IR imaging (OHRC, LROC,
Kaguya, ShadowCam, M3, IIRS), topography & illumination (LOLA, GRAIL, Kaguya
TC), thermal (Diviner), UV (LAMP) and neutron (LEND, Lunar Prospector NS)
sensors, plus the LCROSS impact ground truth.

Use :func:`list_datasets` for keys and :func:`get_dataset` for a single record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["Dataset", "DATASETS", "list_datasets", "get_dataset", "as_table"]


@dataclass(frozen=True)
class Dataset:
    """A single remote-sensing product in the fusion catalogue.

    Attributes
    ----------
    name : str
        Human-readable product name.
    mission : str
        Parent mission / observatory.
    instrument : str
        Instrument or sensor.
    measurement : str
        Band(s) or measured quantity.
    resolution : str
        Native spatial resolution (free text, units included).
    ice_indicator : str
        One-line note on how the product indicates / constrains ice.
    access_url : str
        Where the data can be obtained.
    fmt : str
        Typical distribution format (PDS4, GeoTIFF, IMG, ...).
    access : str
        ``"open"`` or ``"restricted"``.
    """

    name: str
    mission: str
    instrument: str
    measurement: str
    resolution: str
    ice_indicator: str
    access_url: str
    fmt: str
    access: str = "open"


# ---------------------------------------------------------------------------
# The catalogue. Keys are short, stable identifiers.
# ---------------------------------------------------------------------------
DATASETS: dict[str, Dataset] = {
    # --- Chandrayaan-2 (primary mission for this PS) -------------------
    "ch2_dfsar": Dataset(
        name="Chandrayaan-2 DFSAR L&S-band polarimetric SAR",
        mission="Chandrayaan-2",
        instrument="DFSAR",
        measurement="L-band 1.25 GHz & S-band 2.5 GHz full-pol SAR",
        resolution="2-75 m",
        ice_indicator="CPR>1 & DOP<0.13 -> subsurface ice; m-chi volume scatter",
        access_url="https://pradan.issdc.gov.in/ch2/",
        fmt="PDS4",
        access="open",
    ),
    "ch2_ohrc": Dataset(
        name="Chandrayaan-2 Orbiter High-Resolution Camera",
        mission="Chandrayaan-2",
        instrument="OHRC",
        measurement="Panchromatic optical imagery",
        resolution="0.25 m",
        ice_indicator="Boulders/roughness & shadow geometry for double-shadow ID",
        access_url="https://pradan.issdc.gov.in/ch2/",
        fmt="PDS4",
        access="open",
    ),
    "ch2_iirs": Dataset(
        name="Chandrayaan-2 Imaging Infrared Spectrometer",
        mission="Chandrayaan-2",
        instrument="IIRS",
        measurement="0.8-5.0 um hyperspectral reflectance",
        resolution="~80 m",
        ice_indicator="3 um OH/H2O absorption -> surface hydration",
        access_url="https://pradan.issdc.gov.in/ch2/",
        fmt="PDS4",
        access="open",
    ),
    # --- Chandrayaan-1 -------------------------------------------------
    "ch1_minisar": Dataset(
        name="Chandrayaan-1 Mini-SAR S-band radar",
        mission="Chandrayaan-1",
        instrument="Mini-SAR",
        measurement="S-band 2.38 GHz hybrid-pol SAR",
        resolution="~150 m",
        ice_indicator="Anomalous high-CPR craters -> candidate polar ice",
        access_url="https://pds-geosciences.wustl.edu/missions/chandrayaan1/",
        fmt="PDS3",
        access="open",
    ),
    "ch1_m3": Dataset(
        name="Moon Mineralogy Mapper (M3)",
        mission="Chandrayaan-1",
        instrument="M3",
        measurement="0.43-3.0 um imaging spectroscopy",
        resolution="140 m",
        ice_indicator="3.0 um ice absorption -> exposed surface water ice in PSRs",
        access_url="https://pds-imaging.jpl.nasa.gov/volumes/m3.html",
        fmt="PDS3",
        access="open",
    ),
    # --- Lunar Reconnaissance Orbiter ---------------------------------
    "lro_lola": Dataset(
        name="LRO Lunar Orbiter Laser Altimeter DEM",
        mission="LRO",
        instrument="LOLA",
        measurement="1064 nm laser altimetry / topography & normal albedo",
        resolution="5-20 m (polar)",
        ice_indicator="High 1064 nm albedo in PSRs -> surface frost; DEM -> PSR",
        access_url="https://pgda.gsfc.nasa.gov/products/90",
        fmt="GeoTIFF",
        access="open",
    ),
    "lro_lola_illum": Dataset(
        name="LRO LOLA illumination / PSR maps",
        mission="LRO",
        instrument="LOLA",
        measurement="Average solar illumination & permanent-shadow fraction",
        resolution="240 m / 20 m",
        ice_indicator="Permanently shadowed & doubly-shadowed region delineation",
        access_url="https://pgda.gsfc.nasa.gov/products/69",
        fmt="GeoTIFF",
        access="open",
    ),
    "lro_diviner": Dataset(
        name="LRO Diviner Lunar Radiometer (bolometric T)",
        mission="LRO",
        instrument="Diviner",
        measurement="Thermal IR; max/annual surface temperature",
        resolution="~240 m",
        ice_indicator="Tmax < 110 K -> H2O cold trap (stability over Gyr)",
        access_url="https://pds-geosciences.wustl.edu/missions/lro/diviner.htm",
        fmt="PDS3",
        access="open",
    ),
    "lro_minirf": Dataset(
        name="LRO Mini-RF S/X-band radar",
        mission="LRO",
        instrument="Mini-RF",
        measurement="S-band 2.38 GHz hybrid-pol SAR",
        resolution="15-30 m",
        ice_indicator="Elevated CPR in PSR crater interiors -> ice signature",
        access_url="https://pds-geosciences.wustl.edu/missions/lro/mrf.htm",
        fmt="PDS3",
        access="open",
    ),
    "lro_lroc_nac": Dataset(
        name="LRO LROC Narrow-Angle Camera",
        mission="LRO",
        instrument="LROC NAC",
        measurement="Panchromatic optical imagery",
        resolution="0.5 m",
        ice_indicator="Boulder mapping, slope context, secondary-light PSR imaging",
        access_url="https://wms.lroc.asu.edu/lroc",
        fmt="PDS3",
        access="open",
    ),
    "lro_lroc_wac": Dataset(
        name="LRO LROC Wide-Angle Camera",
        mission="LRO",
        instrument="LROC WAC",
        measurement="VIS-UV multispectral imagery",
        resolution="100 m",
        ice_indicator="Regional photometry & illumination context",
        access_url="https://wms.lroc.asu.edu/lroc",
        fmt="PDS3",
        access="open",
    ),
    "lro_lamp": Dataset(
        name="LRO Lyman-Alpha Mapping Project",
        mission="LRO",
        instrument="LAMP",
        measurement="Far-UV (Lyman-alpha) albedo & off/on-band ratio",
        resolution="~240 m",
        ice_indicator="Off-/on-band UV ratio & Lyman-alpha dimming -> surface frost",
        access_url="https://pds-geosciences.wustl.edu/missions/lro/lamp.htm",
        fmt="PDS3",
        access="open",
    ),
    "lro_lend": Dataset(
        name="LRO Lunar Exploration Neutron Detector",
        mission="LRO",
        instrument="LEND",
        measurement="Epithermal neutron flux",
        resolution="~10 km (collimated)",
        ice_indicator="Epithermal neutron suppression -> buried hydrogen (WEH)",
        access_url="https://pds-geosciences.wustl.edu/missions/lro/lend.htm",
        fmt="PDS3",
        access="open",
    ),
    # --- Lunar Prospector ---------------------------------------------
    "lp_ns": Dataset(
        name="Lunar Prospector Neutron Spectrometer",
        mission="Lunar Prospector",
        instrument="NS",
        measurement="Thermal/epithermal/fast neutron flux",
        resolution="~45 km",
        ice_indicator="Polar epithermal suppression -> first global H/ice evidence",
        access_url="https://pds-geosciences.wustl.edu/missions/lunarp/",
        fmt="PDS3",
        access="open",
    ),
    # --- Kaguya / SELENE ----------------------------------------------
    "kaguya_tc": Dataset(
        name="Kaguya Terrain Camera DEM",
        mission="Kaguya/SELENE",
        instrument="TC",
        measurement="Stereo optical -> 10 m DEM",
        resolution="10 m",
        ice_indicator="High-res topography for slope/illumination modelling",
        access_url="https://darts.isas.jaxa.jp/planet/pdap/selene/",
        fmt="PDS3",
        access="open",
    ),
    "kaguya_mi": Dataset(
        name="Kaguya Multiband Imager",
        mission="Kaguya/SELENE",
        instrument="MI",
        measurement="VIS-NIR 9-band reflectance",
        resolution="20 m",
        ice_indicator="Maturity & FeO/TiO2 -> regolith dielectric / loss tangent",
        access_url="https://darts.isas.jaxa.jp/planet/pdap/selene/",
        fmt="PDS3",
        access="open",
    ),
    # --- GRAIL --------------------------------------------------------
    "grail_gravity": Dataset(
        name="GRAIL lunar gravity field",
        mission="GRAIL",
        instrument="LGRS",
        measurement="Spherical-harmonic gravity / crustal density",
        resolution="~5 km",
        ice_indicator="Crustal porosity/density context for subsurface modelling",
        access_url="https://pds-geosciences.wustl.edu/missions/grail/",
        fmt="PDS3",
        access="open",
    ),
    # --- LCROSS impact ground truth -----------------------------------
    "lcross": Dataset(
        name="LCROSS Centaur impact volatiles",
        mission="LCROSS",
        instrument="Shepherding Spacecraft suite",
        measurement="Impact-ejecta NIR/UV spectroscopy",
        resolution="point (Cabeus)",
        ice_indicator="Direct detection: 5.6 +/- 2.9 wt% water in ejecta (ground truth)",
        access_url="https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LCROSS",
        fmt="PDS3",
        access="open",
    ),
    # --- Danuri (KPLO) ShadowCam --------------------------------------
    "danuri_shadowcam": Dataset(
        name="Danuri ShadowCam (PSR-optimised imager)",
        mission="Danuri/KPLO",
        instrument="ShadowCam",
        measurement="High-sensitivity panchromatic (secondary illumination)",
        resolution="1.7 m",
        ice_indicator="Direct PSR-floor imaging of frost brightness & morphology",
        access_url="https://pds.shadowcam.im-ldi.com/",
        fmt="PDS4",
        access="open",
    ),
    # --- Arecibo Earth-based radar ------------------------------------
    "arecibo_radar": Dataset(
        name="Arecibo 12.6 cm Earth-based lunar radar",
        mission="Arecibo Observatory",
        instrument="S-band radar",
        measurement="12.6 cm dual-pol radar backscatter & CPR",
        resolution="20-80 m",
        ice_indicator="CPR mapping of polar craters; rough vs ice disambiguation",
        access_url="https://www.naic.edu/~smarshal/lunar.html",
        fmt="FITS",
        access="open",
    ),
    # --- Supporting derived / model products --------------------------
    "lola_slope": Dataset(
        name="LOLA-derived slope & roughness",
        mission="LRO",
        instrument="LOLA (derived)",
        measurement="Slope, RMS roughness, Hurst exponent",
        resolution="20-60 m",
        ice_indicator="Landing/traverse safety; roughness for CPR false-positive ID",
        access_url="https://pgda.gsfc.nasa.gov/products/90",
        fmt="GeoTIFF",
        access="open",
    ),
    "lola_earthvis": Dataset(
        name="LOLA Earth-visibility / comms maps",
        mission="LRO",
        instrument="LOLA (derived)",
        measurement="Fraction of time Earth above horizon",
        resolution="240 m",
        ice_indicator="Direct-to-Earth comms windows for rover ops planning",
        access_url="https://pgda.gsfc.nasa.gov/products/78",
        fmt="GeoTIFF",
        access="open",
    ),
    "diviner_rocka": Dataset(
        name="Diviner rock abundance & regolith temperature",
        mission="LRO",
        instrument="Diviner (derived)",
        measurement="Rock abundance fraction & regolith Tbol",
        resolution="~240 m",
        ice_indicator="Surface rock fraction -> radar roughness decoy context",
        access_url="https://pds-geosciences.wustl.edu/missions/lro/diviner.htm",
        fmt="PDS3",
        access="open",
    ),
    "ch2_xsm_class": Dataset(
        name="Chandrayaan-2 CLASS X-ray fluorescence",
        mission="Chandrayaan-2",
        instrument="CLASS",
        measurement="Elemental abundance (Na, Mg, Al, Si, Ca, Fe)",
        resolution="~12.5 km",
        ice_indicator="FeO/TiO2 abundance -> regolith loss-tangent for depth model",
        access_url="https://pradan.issdc.gov.in/ch2/",
        fmt="PDS4",
        access="open",
    ),
    "ch2_tmc2": Dataset(
        name="Chandrayaan-2 Terrain Mapping Camera-2 DEM",
        mission="Chandrayaan-2",
        instrument="TMC-2",
        measurement="Stereo optical -> DEM",
        resolution="5 m",
        ice_indicator="High-res topography for PSR & slope mapping",
        access_url="https://pradan.issdc.gov.in/ch2/",
        fmt="PDS4",
        access="open",
    ),
    "lola_psr_temp": Dataset(
        name="LOLA+Diviner thermal-stability model",
        mission="LRO (derived)",
        instrument="LOLA+Diviner",
        measurement="Modelled ice-stability depth",
        resolution="240 m",
        ice_indicator="Depth to ice-stable layer (top-5 m volume modelling)",
        access_url="https://pgda.gsfc.nasa.gov/products/69",
        fmt="GeoTIFF",
        access="open",
    ),
}


def list_datasets() -> list[str]:
    """Return all catalogue keys (sorted)."""
    return sorted(DATASETS.keys())


def get_dataset(key: str) -> Dataset:
    """Return the :class:`Dataset` for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not in the catalogue (message lists valid keys).
    """
    try:
        return DATASETS[key]
    except KeyError as exc:  # pragma: no cover - error path
        raise KeyError(
            f"Unknown dataset {key!r}. Available: {', '.join(list_datasets())}"
        ) from exc


def as_table() -> list[dict]:
    """Return the catalogue as a list of plain dicts (for pandas / reporting)."""
    return [{"key": k, **asdict(v)} for k, v in sorted(DATASETS.items())]
