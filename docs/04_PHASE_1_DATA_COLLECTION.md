# Document 04 — Phase 1: Data Collection and Event Catalog

**Weeks 1–3 · Owner: Chetan Mahajan · Reviewer: Ronit Dahiwal**

**Goal:** one file, `data/catalog/events.csv`, containing every landslide and flood event in the study region and period, each fully sourced.

**Why this phase comes first:** everything else is built on this table. A point process learns only from events. If the catalog is thin, wrong, or badly sourced, no amount of clever mathematics later will rescue the project.

---

## 1. Study region and period — fixed here, used everywhere

| Item | Value |
|---|---|
| Region | Western Ghats of Kerala: Wayanad, Malappuram, Kozhikode, Idukki districts at minimum |
| Boundary definition | HydroSHEDS sub-basin polygons intersecting those districts, **not** administrative boundaries |
| Period | 1 June 2018 – 31 December 2024 |
| Analysis CRS | UTM Zone 43N, **EPSG:32643** |
| Storage CRS | WGS 84, EPSG:4326 (catalog only) |
| Held out entirely | 1 June 2024 – 31 December 2024 |

**Why hydrological boundaries:** a landslide in Wayanad district can flood a reach in Malappuram district. A model built on administrative boundaries would cut the cascade in half.

---

## 2. The catalog schema

One row per event. These columns exactly, in this order.

```
event_id            unique, format: L-0001 / F-0001
hazard_type         L | F
date                YYYY-MM-DD
time_known          Y | N
time                HH:MM IST, blank if unknown
lat, lon            EPSG:4326, decimal degrees, 6 dp
geometry_type       point | polygon_centroid | polygon_crown | reach
area_m2             mapped area, blank for floods
elevation_m         from DEM at the recorded point
mark_raw            landslide: area_m2 | flood: peak discharge m3/s
mark_value          landslide: log(area_m2) | flood: log(Q_peak/Q2)
severity_class      landslide: small/medium/large | flood: minor/major/extreme
spatial_unit_id     slope unit id (L) or reach id (F), filled in Phase 2
source              short code, e.g. GSI_BHUKOSH, ESSD_HAO2020, GLOFAS_V4
source_url          full URL
source_date_accessed YYYY-MM-DD
accessed_by         member name
confidence          high | medium | low
duplicate_of        event_id if merged, else blank
notes               free text
```

**Rules that are not negotiable:**

- Every row has a `source_url` and an `accessed_by`. No exceptions.
- `data/raw/` is never edited. Ever.
- No CSV is ever opened and saved in Excel — it silently mangles dates and truncates coordinate precision.
- Every raw download folder contains a `PROVENANCE.md` written **at download time**, recording the URL, the exact query or filter used, the date, the person, and the licence.

---

## 3. Landslide events — sources and procedure

### 3.1 Source priority

| Priority | Source | Coverage | Notes |
|---|---|---|---|
| 1 | **Published Kerala inventories** — ESSD 2018 monsoon dataset (Hao et al. 2020); NRSC satellite inventory (Martha et al. 2019, ~5,191 polygons) | 2018 event | **Bulk of the catalog.** Thousands of polygons with area. Peer-reviewed and citable |
| 2 | **GSI Bhukosh** — `bhukosh.gsi.gov.in/Bhukosh/Public` | ongoing | National inventory layer, free download |
| 3 | **GSI Bhusanket** — `bhusanket.gsi.gov.in` | major events | Field First Information Reports. The Mundakkai FIR gives source elevation, slope angle, overburden thickness, susceptibility zone |
| 4 | **Published event studies** | specific events | Puthumala 2019, Kavalappara 2019, Pettimudi 2020, Kokkayar/Plappally/Kavali 2021, Mundakkai 2024 |
| 5 | **NASA COOLR / Global Landslide Catalog** | ~2007–2018 only | Cross-check for the pre-2018 period. **It does not contain the 2024 Wayanad event** |

### 3.2 Polygon to point conversion

Inventories give polygons. The model needs an initiation point.

> **Use the highest-elevation point of the polygon, not the centroid.**

A landslide starts at the crown and runs downhill. The centroid of a long-runout polygon sits in the middle of the runout path, which is physically wrong and will corrupt both the terrain covariates and the flow-path projection.

Record `geometry_type = polygon_crown` and keep the original polygon in `data/interim/` so the area and the runout path can be recovered.

### 3.3 Deduplication rule — write it down before you start

> Two landslide records are the **same event** if they are within **500 m** and within **1 day** of each other.
>
> When merging: keep the record with the higher `confidence`; if tied, keep the one with a mapped polygon over a point; if still tied, keep the earlier-published source. Record the discarded `event_id` in `duplicate_of`.

Two inventories will list the same landslide with coordinates a few hundred metres apart. Without a written rule, four people will merge differently and the catalog will be inconsistent.

### 3.4 Completeness cutoff — a required analysis

Plot the **size–frequency distribution** of mapped landslide areas on log–log axes. Real landslide inventories follow an inverse power law above a rollover point, and are systematically incomplete below it.

1. Identify the rollover area `A_min`.
2. Report it.
3. **Model only events above `A_min`**, and state this in the paper.

Fitting a point process on an inventory that is incomplete at small sizes, without acknowledging it, produces biased background rates. This one plot pre-empts a serious reviewer objection.

---

## 4. Flood events — the detection procedure

This follows Document 02, §3 exactly. It is a procedure, not a judgement call.

### 4.1 Step by step

**Step 1 — download.** GloFAS v4.0 hydrological reanalysis, daily river discharge, 0.05°, from the Copernicus Climate Data Store (dataset `cems-glofas-historical`). Requires a free CDS account and the `cdsapi` client. Pull 1980–2024 for the study region bounding box.

**Step 2 — eligibility.** Keep only river cells with upstream contributing area **> 50 km²**. Record how many reaches this leaves. Run a sensitivity check later at 250 km².

**Step 3 — thresholds.** For each eligible cell: extract the annual maximum daily discharge for each year, fit a **Gumbel distribution by the L-moments method**, and compute `Q₂`, `Q₅`, `Q₂₀`.

This is GloFAS's own threshold procedure. Adopting it deliberately means our flood definition matches an operational system.

**Step 4 — detect.** A flood event begins on the first day discharge exceeds `Q₂`.

**Step 5 — decluster.** The episode ends when discharge stays below `Q₂` for **3 consecutive days**. Only then may a new event be recorded at that reach.

**Step 6 — record.** Event date = first exceedance day. Location = the reach. `mark_value = log(Q_peak / Q₂)`. `severity_class` from `Q₂/Q₅/Q₂₀` bands.

### 4.2 The corroboration experiment — required deliverable

The flood definition is model-derived, so it must be checked against observations.

**Precision test.**
1. Take all detected flood events with Sentinel-1 coverage within ±3 days.
2. Map inundation using **Otsu automatic thresholding of VV backscatter in Google Earth Engine** — a published method for Kerala with reported overall accuracy around 94%.
3. Compare against a dry-season reference date.
4. Report: what fraction of detected events show a real inundation signal.

**Recall test.**
1. Assemble an independent list of documented major Kerala floods from KSDMA reports, Copernicus EMS / UNOSAT activations, and news archives.
2. Report: what fraction were detected by the GloFAS rule.

**Both numbers go in the paper.** If precision at `Q₂` is poor, raise to `Q₅` and repeat. This converts a definitional choice into a measured quantity — which is exactly what the reviewer asked for.

### 4.3 The limitation to state now, not later

The Punnapuzha at Chooralmala has a contributing area far below the eligibility threshold. **GloFAS cannot see the Wayanad flood at the debris-flow site.**

For the Wayanad backtest, the downstream flood is evidenced by field reports, the destroyed bridge, SAR imagery, and the Chaliyar discharge signal further downstream where the catchment is large enough. Write this into `LIMITATIONS.md` in Week 3, not in December.

---

## 5. Cascade pairs — the linking table

Separate file: `data/catalog/cascade_pairs.csv`.

```
pair_id, landslide_event_id, flood_event_id,
lag_hours, network_distance_km, evidence_level, evidence_source, notes
```

`evidence_level`:
- **confirmed** — a published source or government report explicitly states the landslide caused the downstream flooding
- **probable** — correct spatial and temporal relationship, plus at least one supporting report
- **candidate** — the flood is downstream and within the plausible time window, but no independent evidence

Only `confirmed` and `probable` pairs are used to *evaluate* cascade detection. All events are used to *fit* the model — the model discovers pairs on its own; the table is for validation.

### Starter list of cascade candidates in Kerala

| Event | Date | Role |
|---|---|---|
| Kerala monsoon landslides + floods | 16 Aug 2018 | Bulk training; concurrent flood and landslide |
| Kavalappara, Malappuram | Aug 2019 | Test event |
| Puthumala, Wayanad | Aug 2019 | Test event; **2–3 km from the 2024 crown** |
| Pettimudi, Idukki | 6 Aug 2020 | Test event |
| Kokkayar / Plappally / Kavali | Oct 2021 | Test events, three different causes |
| **Mundakkai–Chooralmala** | **30 Jul 2024** | **Final held-out test. Never in training** |

**Provenance warning.** Published sources disagree on the exact date of the Puthumala 2019 landslide — some give 8 August, others 29 August. Resolve it from GSI or the primary field report, and record which source you trusted in `notes`. Treat this as the team's training exercise in provenance discipline: it is exactly the kind of small discrepancy that becomes an embarrassing question in a review.

---

## 6. Quality gates — the phase is not done until all five pass

| # | Gate | Test |
|---|---|---|
| 1 | **Volume** | ≥ 200 landslide events after deduplication and the completeness cutoff. Target ≥ 1,000 |
| 2 | **Provenance** | 100% of rows have `source_url` and `accessed_by`. Automated check |
| 3 | **Seasonality** | Events per month follow the monsoon. If they do not, something is wrong with dates or timezones |
| 4 | **Size–frequency** | Log–log plot shows a power law above a rollover; `A_min` identified and recorded |
| 5 | **No duplicates** | Automated check: no two same-type events within 500 m and 1 day |

### Hard stop

> **If Gate 1 fails at the end of Week 3, stop and widen the region** to the full Western Ghats across Kerala, Karnataka and Tamil Nadu before doing anything else. Do not proceed with a thin catalog and hope.

---

## 7. Four-person split

| Member | Task |
|---|---|
| **Chetan** | GSI Bhukosh and Bhusanket downloads; transcribe the Mundakkai FIR; build and enforce the schema; write the provenance checker |
| **Yugant** | Published inventories: retrieve the ESSD 2018 dataset and the NRSC inventory; polygon → crown-point conversion; area extraction; size–frequency plot |
| **Raj** | GloFAS: CDS download, Gumbel/L-moments threshold fitting, event detection and declustering, sensitivity to `Q₂` vs `Q₅`; document the threshold choice |
| **Ronit** | Deduplication and merge rules; the corroboration experiment (Sentinel-1 in GEE); cascade pair table; all summary plots |

---

## 8. Deliverables

| Deliverable | Format |
|---|---|
| `data/catalog/events.csv` | the catalog |
| `data/catalog/cascade_pairs.csv` | validation pairs |
| `data/catalog/events_CHANGELOG.md` | every change, dated, with reason |
| `figures/phase1_events_per_year.png` | bar chart |
| `figures/phase1_event_map.png` | map with landslides and flood reaches |
| `figures/phase1_size_frequency.png` | log–log with `A_min` marked |
| `reports/phase1_flood_definition.md` | the rule, plus measured precision and recall |
| `notebooks/01_*.ipynb` | exploratory work, numbered |

**Show the guide:** the events-per-year chart, the map, and the measured precision of the flood definition. That third item is what separates this from a student data-gathering exercise.
