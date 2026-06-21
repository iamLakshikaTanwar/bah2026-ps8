# lunaris — The Science

The scientific narrative behind ISRO BAH 2026 Problem Statement 8: *why*
doubly shadowed craters are the Moon's best volatile reservoirs, *how* radar
polarimetry detects buried ice, *why* that detection is famously ambiguous, and
*how* `lunaris` earns confidence through multi-evidence fusion. For the equations
and code references see [ARCHITECTURE.md](../ARCHITECTURE.md) §5 and
[METHODS.md](../METHODS.md).

---

## 1. Permanently shadowed regions and doubly shadowed craters

The Moon's spin axis is tilted only **1.543°** from the ecliptic normal — almost
nothing. Near the poles, this means the Sun never climbs more than ~1.5° above the
horizon, and the floors of deep impact craters can lie in shadow *forever*. These
**Permanently Shadowed Regions (PSRs)** are among the coldest places in the Solar
System, with floor temperatures that can fall below **40 K**.

A PSR floor still receives a trickle of *secondary* illumination — sunlight and
thermal infrared scattered and re-emitted from the surrounding sunlit crater rim
and walls. A **doubly shadowed crater** is a smaller crater nested *inside* a PSR
whose own rim blocks even that secondary light. Such a floor sees neither direct
sunlight nor scattered light from any sunlit terrain. It is therefore colder,
more thermally stable, and a more perfect volatile **cold trap** than the parent
PSR — the single best place on the Moon to look for ancient, well-preserved
water-ice. PS8 targets exactly this: a doubly shadowed crater on the floor of
**Faustini**.

---

## 2. Faustini — the target

| Property | Value | Note |
|----------|-------|------|
| Centre | **87.3° S, 77.0° E** | deep South-Pole highlands |
| Diameter | **~39-41 km** | host crater |
| Floor elevation | ~ −2700 m (rel. datum) | deep, well-shadowed bowl |
| PSR area | **~664 km²** | **third-largest PSR** on the Moon |
| Floor temperature | **< 110 K**, locally **~20 K** | robust H₂O cold trap (and colder species) |
| Detection | DFSAR sub-crater **CPR** anomaly | the doubly shadowed crater on the floor |

Faustini is a flagship cold trap: its enormous, ultra-cold PSR floor has long been
a prime ISRU candidate, and the nested doubly shadowed crater concentrates the
coldest, most stable conditions. Chandrayaan-2 DFSAR observations resolve a
**circular-polarisation-ratio anomaly** over the sub-crater floor consistent with
buried ice — the detection `lunaris` is built to characterise. (Faustini's
thermal and illumination environment is characterised in Williams et al. 2024 and
in the LOLA/Diviner PSR products; see [docs/DATA_SOURCES.md](DATA_SOURCES.md).)

---

## 3. Cold-trap physics — how ice survives for an aeon

Water ice on an airless body is held by **thermal stability**, not by pressure.
At the relevant temperatures the loss process is **sublimation** directly to
vapour, and its rate is set exponentially by temperature through the saturation
vapour pressure. The free-space mass-loss rate (Hertz-Knudsen) is

```
E = P_sv(T) · √( M / (2π·R·T) )     [kg m⁻² s⁻¹]
```

with `P_sv` the Clausius-Clapeyron saturation vapour pressure and `M` the molar
mass of water. Because `P_sv(T)` plunges by orders of magnitude as `T` falls, a
threshold emerges: below roughly **110 K**, water ice sublimes so slowly that it
survives over **billions of years** (Gyr). Colder ceilings trap progressively
more volatile species:

| Species | Stability ceiling | Where |
|---------|-------------------|-------|
| **H₂O** | **≤ 110 K** | most PSR floors |
| SO₂ / CO₂ | ≤ 60 K | colder traps |
| NH₃ | ≤ 66 K | colder traps |
| CO, N₂, CH₄ (super-volatiles) | ≤ 40 K | the very coldest doubly shadowed floors |

(Thresholds: Paige et al. 2010, Diviner.) Where the *annual maximum* surface
temperature exceeds the ceiling, ice is unstable at the surface but can still be
stable **buried** below a depth where the diurnal/seasonal thermal wave is
damped — the **ice-stability depth**, set by the regolith thermal gradient and
the basal geothermal flux (`0.018 W m⁻²`). This is precisely why a *subsurface*
detector (penetrating L-band radar) is the right tool, and why the top ~5 m
column is the volume of interest (deliverable D5). `lunaris` models all of this
in `terrain.thermal` (cold-trap mask, sublimation, stability depth, 1-D profile)
and the illumination physics in `terrain.illumination` (PSR, double-shadow,
sky-view factor).

---

## 4. Detecting buried ice with radar polarimetry

A radar transmits a circularly polarised wave and listens in two senses. The key
observable is the **Circular Polarisation Ratio**,

```
CPR = same-sense / opposite-sense = (S0 − S3)/(S0 + S3).
```

A smooth surface flips the handedness of the reflected wave (low CPR). But a
low-loss, transparent **volume of ice** lets the wave scatter multiple times
internally; constructive interference along time-reversed paths — the
**Coherent Backscatter Opposition Effect (CBOE)** — preserves the same sense and
drives `CPR > 1`. High CPR over a cold-trap floor is therefore the classic radar
signature of buried ice (Spudis et al. 2013; Campbell 2012). Chandrayaan-2 DFSAR
adds two crucial capabilities: a **long L-band wavelength (24 cm)** that
penetrates several metres of dry regolith to *sound the subsurface*, and a
**shorter S-band (12 cm)** cross-check, both in full/hybrid polarimetry so the
full Stokes vector — and hence the m-χ/m-δ decompositions and the degree of
polarisation — can be recovered.

---

## 5. The ice-detection controversy — and how lunaris answers it

Here is the hard part, and the reason naïve radar ice claims are contested.

**A high CPR is not, by itself, proof of ice.** Wavelength-scale surface
**roughness** — blocky ejecta, fresh crater rims, angular boulders — *also*
produces multiple bounces and same-sense returns, raising CPR with **no ice
present**. This is the **CPR ambiguity**:

* **Spudis et al. (2013)** interpreted anomalous high-CPR polar craters (from LRO
  Mini-RF) as ice-bearing.
* **Fa & Cahill (2013)** and **Eke et al. (2014)** showed that many high-CPR
  craters are simply *rough* — young, blocky impact craters — and that CPR alone
  cannot distinguish ice from roughness. Eke et al. specifically demonstrated how
  crater morphology biases both the radar and neutron ice signals.

This debate is the central scientific risk of the whole problem. `lunaris`
resolves it on two complementary fronts:

### Front 1 — the DOP gate (the Sinha et al. 2026 refinement)

The **Degree of Polarisation**,

```
DOP = m = √(S1² + S2² + S3²)/S0  ∈ [0, 1],
```

measures how *organised* the returned wave is. Rough-surface multiple bounces
remain **polarised** (high DOP); genuine ice volume-scatter is **depolarised**
(low DOP). Requiring

> **CPR > 1  AND  DOP < 0.13**

(Sinha et al. 2026) keeps the coherent ice returns while **rejecting the
roughness decoys** that pass a CPR-only test. This is the joint criterion at the
heart of `lunaris` (`classify.ice_lut`). The synthetic Faustini testbed is
deliberately seeded with *both* a coherent ice patch (high CPR, **low** DOP) and
rough rim/ejecta **decoys** (high CPR, **high** DOP) so the algorithm's ability
to separate them is verifiable.

### Front 2 — independent-physics corroboration (multi-sensor fusion)

Even the joint criterion is radar physics alone. The decisive move is to demand
agreement from **physically independent** sensors whose failure modes are
*uncorrelated* with roughness:

```
ICE confidence ↑  ⇔  radar(CPR>1 ∧ DOP<0.13)        (Ch-2 DFSAR)
                 ∧  thermal cold trap (Tmax<110 K)   (LRO Diviner)
                 ∧  neutron H suppression (WEH)       (LRO LEND, LP-NS)
                 ∧  UV/NIR/albedo frost               (LAMP, LOLA 1064 nm, M3/IIRS)
```

A rough patch of rock has no reason to *also* be a 110 K cold trap, *also*
suppress epithermal neutrons (which respond to hydrogen, not roughness), and
*also* brighten in the far-UV. **Ice does all four.** Because these indicators
sample different depths and different physics — radar (metres-deep
dielectric/structure), neutrons (top ~1 m hydrogen), thermal (surface
stability), UV/NIR (top-microns frost) — their conjunction is far harder to fake
than any single channel. `lunaris` formalises this as Bayesian log-odds pooling,
Dempster-Shafer belief combination, and a learned RandomForest over a 10-feature
multi-sensor stack (`classify.evidence`, `classify.fusion`; see
[ARCHITECTURE.md](../ARCHITECTURE.md) §6).

The honest bottom line (also stated in [ARCHITECTURE.md](../ARCHITECTURE.md) §9):
the CPR ambiguity is **mitigated, not eliminated**, and — because the real
permittivity of cold ice (≈ 3.15) is barely above that of dry regolith (≈ 3.0) —
we report a **relative ice-likelihood index**, never an absolute ice
weight-percent. The LCROSS direct detection of **5.6 ± 2.9 wt%** water in the
Cabeus ejecta (Colaprete et al. 2010) is the one *in-situ* anchor that the
remote-sensing picture must be consistent with.

---

## 6. Why this matters — ISRU, Artemis, LUPEX

Confidently locating and quantifying accessible South-Pole ice is foundational to
the next era of lunar exploration:

* **In-Situ Resource Utilisation (ISRU).** Water-ice is propellant (H₂ + O₂),
  drinking water and radiation shielding. Mining it *in place* is the difference
  between a visited Moon and an inhabited one.
* **ISRO LUPEX (Lunar Polar Exploration).** The ISRO-JAXA LUPEX mission will land
  a rover at the South Pole to prospect for and characterise water-ice in situ —
  exactly the detection-to-traverse problem `lunaris` plans end to end.
* **Artemis.** NASA's crewed programme targets the South Pole specifically for its
  sustained-shadow ice; ice maps and safe-traverse plans directly inform landing
  and surface operations.
* **VIPER-class prospecting.** Energy-aware rover traverses across PSR margins —
  balancing science value against the brutal 70-h shadow-survival constraint — are
  the operational core of polar ice prospecting (Tompkins et al. 2024). `lunaris`
  plans precisely such traverses (`planning.energy`, `planning.*`).

By turning Chandrayaan-2 DFSAR observations of a doubly shadowed Faustini crater
into a fused, uncertainty-quantified ice map, a feasible landing site, an
energy-aware traverse, and a top-5 m ice-volume estimate, `lunaris` advances all
of these objectives — and the broader science of planetary radar remote sensing.

---

*For the governing equations and module-level detail see
[ARCHITECTURE.md](../ARCHITECTURE.md); for the full method catalogue see
[METHODS.md](../METHODS.md); for the datasets and access portals see
[docs/DATA_SOURCES.md](DATA_SOURCES.md).*
