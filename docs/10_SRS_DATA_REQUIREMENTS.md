# Document 10 — Data Requirements Specification

**Owner: Chetan Mahajan** · Brief reference document. Every dataset, where it comes from, what it is for, what is wrong with it, and what to do if it fails.

---

## 1. Summary

| Class | Count | All free? |
|---|---|---|
| Event sources | 6 | Yes |
| Meteorological | 3 | Yes |
| Terrain and static | 5 | Yes |
| Hydrological | 3 | Yes |
| Corroboration | 4 | Yes |

No paid, licensed or access-restricted data is required. Three sources need a free account: Copernicus Climate Data Store, NASA Earthdata, and Google Earth Engine.

---

## 2. Event data

| ID | Dataset | Source | Provides | Limitation | Fallback |
|---|---|---|---|---|---|
| **E1** | ESSD 2018 Kerala landslide inventory (Hao et al. 2020) | `essd.copernicus.org`, doi 10.5194/essd-12-2899-2020 | Thousands of mapped polygons with LULC, 2018 monsoon | Single event year | E2, E3 |
| **E2** | NRSC satellite inventory (Martha et al. 2019) | *Current Science* 117(5):804–812 | ~5,191 polygons, Western Ghats, Aug 2018 | Same event | E1 |
| **E3** | GSI Bhukosh national landslide inventory | `bhukosh.gsi.gov.in/Bhukosh/Public` | Dated, located landslides, ongoing | Attribute completeness varies by state and field season | E4, published studies |
| **E4** | GSI Bhusanket field reports (FIRs) | `bhusanket.gsi.gov.in` | Authoritative detail on major events; the Mundakkai FIR gives source elevation, slope angle, overburden thickness, susceptibility zone, flow path | PDFs, manual transcription | Published event studies |
| **E5** | NASA COOLR / Global Landslide Catalog | `landslides.nasa.gov` | Global rainfall-triggered landslide reports | **Coverage ~2007–2018. Does NOT contain the 2024 Wayanad event** | Use as pre-2018 cross-check only |
| **E6** | Published event studies | journals | Puthumala 2019, Kavalappara 2019, Pettimudi 2020, Kokkayar/Plappally/Kavali 2021, Mundakkai 2024 | Individual events only | — |

---

## 3. Meteorological data

| ID | Dataset | Source | Resolution | Purpose | Limitation |
|---|---|---|---|---|---|
| **M1** | IMD gridded daily rainfall | `imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html` | 0.25° (~27 km), daily, 1901–2024 | Long-record climatology, antecedent windows | **Too coarse for a single hillslope**. Gauge-based, so more reliable than satellite for totals |
| **M2** | NASA GPM IMERG V07 | NASA Earthdata (free login). Early Run at `jsimpsonhttps.pps.eosdis.nasa.gov/imerg/early/` | 0.1°, half-hourly, 2000– | Event-scale intensity; **all real-time work** | **Early Run 4 h latency, Late 14 h, Final 3.5 months.** Satellite underestimates extreme orographic rainfall in the Ghats |
| **M3** | ERA5-Land soil moisture | Copernicus Climate Data Store | ~9 km, hourly | Antecedent wetness for **fitting only** | **Multi-day latency — unusable in real time.** Substitute a calibrated API from M2 at run time |

---

## 4. Terrain and static data

| ID | Dataset | Source | Resolution | Purpose | Note |
|---|---|---|---|---|---|
| **T1** | Copernicus DEM (preferred) or SRTM | Copernicus Data Space; OpenTopography | 30 m | Slope units, channel network, flow routing, **travel time** | **Fill sinks before flow routing** or the projection operator breaks silently |
| **T2** | HydroSHEDS / HydroATLAS | `hydrosheds.org` | — | Sub-basin boundaries, drainage attributes | Saves a week of DEM work; makes spatial units defensible |
| **T3** | SoilGrids | `soilgrids.org` | 250 m | Texture, depth, organic carbon | Global model, coarse locally |
| **T4** | ESA WorldCover | ESA / Copernicus | 10 m | Land cover | Specific years only |
| **T5** | Sentinel-2 L2A | Copernicus Data Space | 10–20 m | Land cover change 2011 vs 2024; optional turbidity proxy | Heavy monsoon cloud in Kerala |

---

## 5. Hydrological data

| ID | Dataset | Source | Purpose | Limitation |
|---|---|---|---|---|
| **H1** | GloFAS v4.0 hydrological reanalysis | Copernicus CDS, dataset `cems-glofas-historical` | **Primary flood event definition.** Daily discharge, 0.05°, 1980– | **Skill lowest in small catchments**; operational reporting points restricted to catchments above 250 km². Our 50 km² research threshold must be sensitivity-tested |
| **H2** | GloFAS return-period thresholds | Derived by us using GloFAS's own method: Gumbel fit to annual maxima by L-moments → `Q₂, Q₅, Q₂₀` | Threshold definition | Trends in the ERA5-driven reanalysis can affect extreme-value fits — note it |
| **H3** | CWC discharge and water level | National Water Data Portal, `nwdp.nwic.gov.in` — daily manual discharge and hourly telemetry level, per state including Kerala | Ground truth for H1 | Sparse; gaps. The older India-WRIS portal has documented gaps to ~2020 and login requirements for classified rivers |

---

## 6. Corroboration data

| ID | Dataset | Source | Purpose |
|---|---|---|---|
| **C1** | Sentinel-1 GRD SAR | Google Earth Engine | **Flood-definition precision test.** Otsu thresholding of VV backscatter; published Kerala method with ~94% reported accuracy. Works through monsoon cloud |
| **C2** | Copernicus EMS / UNOSAT rapid mapping | respective portals | Official activation footprints for the largest events; recall test |
| **C3** | KSDMA reports, NDMA situation reports | state and national portals | Human confirmation of flooding at named places |
| **C4** | News archives and ReliefWeb | ReliefWeb, news outlets | Timing and impact detail for major events; **lowest confidence tier — never the sole source for a catalog row** |

---

## 7. Accounts to register in Week 1

| Account | Needed for | Approval time |
|---|---|---|
| Copernicus Climate Data Store + `cdsapi` key | GloFAS, ERA5-Land | usually immediate |
| NASA Earthdata | IMERG | usually immediate |
| Google Earth Engine (research/non-commercial) | Sentinel-1 processing | **can take days — register first** |
| Copernicus Data Space | Sentinel-2, DEM | immediate |

**Register all four on Day 4 of Week 1.** Waiting for approval in Week 6 is an avoidable delay.

---

## 8. Storage and provenance rules

```
data/
  raw/<source_code>/<YYYY-MM-DD_downloaded>/
      <files>
      PROVENANCE.md          ← written AT DOWNLOAD TIME
  interim/                   ← cleaned, reprojected, still recognisable
  processed/                 ← model-ready tables and tensors
  catalog/
      events.csv
      cascade_pairs.csv
      events_CHANGELOG.md
```

**`PROVENANCE.md` template:**

```
Source:            GloFAS v4.0 hydrological reanalysis
URL:               https://cds.climate.copernicus.eu/...
Exact query:       area [8,74,13,78], 1980-01-01 to 2024-12-31, variable river_discharge
Downloaded:        2026-09-18
By:                Raj Choudhari
Licence:           Copernicus, free and open
Citation:          Harrigan et al. (2020), ESSD 12:2043-2060
File checksum:     sha256:...
Notes:             10 GB. Subset to study bbox before committing anything to interim/.
```

### Rules

1. `data/raw/` is **never edited**. Ever.
2. No CSV is ever opened and saved in Excel — dates and coordinate precision are silently corrupted.
3. All spatial data reprojected to **EPSG:32643** before analysis. Distance and area cannot be computed in degrees.
4. `data/` is gitignored except for `PROVENANCE.md` files and the catalog.
5. Large downloads are subset to the study bounding box before anything moves to `interim/`.

---

## 9. Data volume estimate

| Item | Approximate size |
|---|---|
| DEM tiles + derivatives | 5 GB |
| GloFAS subset, 1980–2024 | 3 GB |
| IMERG subset, 2018–2024 | 8 GB |
| IMD gridded, full record | 2 GB |
| ERA5-Land subset | 6 GB |
| Sentinel-1 (processed in GEE, exports only) | 1 GB |
| Sentinel-2 scenes | 10 GB |
| Precomputed travel-time and reachability tables | 2 GB |
| **Total** | **~35–40 GB** |

Plan storage before Week 2. One external drive or one cloud bucket, shared, with the folder structure above.

---

## 10. Known data risks and responses

| Risk | Response |
|---|---|
| Catalog too small after deduplication | **Hard gate at Week 3.** Widen the region to the Western Ghats across three states |
| GloFAS unreliable at our catchment sizes | Sensitivity analysis at 250 km²; corroborate with SAR; state the limitation for headwater reaches |
| IMERG underestimates the Wayanad rainfall peak | Use IMD for totals, IMERG for timing; document the discrepancy quantitatively |
| Bhukosh portal down or restricted | Fall back to published inventories E1, E2 and event studies E6 |
| GEE approval delayed | Sentinel-1 processing is a corroboration step, not a dependency. Proceed and backfill |
| ERA5-Land download too slow | Reduce to the 4 layers actually used, and to the study bounding box only |
