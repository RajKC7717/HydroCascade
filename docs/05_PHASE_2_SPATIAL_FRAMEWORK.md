# Document 05 — Phase 2: Spatial Framework and Covariate Stack

**Weeks 3–5 · Owner: Chetan Mahajan · Reviewer: Raj Choudhari**

**Goal:** build the two spatial supports, the operator that connects them, and every precomputed quantity the cascade model needs — plus the daily covariate stack for Part 1.

**Why this is its own phase:** the answer to "what does *location* mean in your model?" is built here, once. Everything downstream depends on it. Building the travel-time and reachability tables now also de-risks the novel part of the model, because they are expensive to compute and are needed in Phase 4.

---

## 1. What this phase produces

| Artefact | Description | Used by |
|---|---|---|
| `slope_units.gpkg` | ~80,000 hillslope polygons, the domain **H** | Parts 1 and 2 |
| `channel_network.gpkg` | ~15,000 river reaches, the domain **N** | Part 2 |
| `projection_table.parquet` | `π(u)` for every slope unit → injection reach | Part 2 |
| `downstream_index.npz` | sparse reachability: which reaches lie downstream of each node | Part 2 |
| `travel_time.npz` | `τ(p → s)` for all downstream pairs within the truncation radius | Part 2 |
| `static_covariates.parquet` | terrain, soil, land cover per unit | Part 1 |
| `daily_covariates.parquet` | rainfall windows and soil moisture per unit per day | Parts 1 and 2 |
| `get_covariates(lat, lon, date)` | one function returning everything | everything |

---

## 2. Building the hillslope domain H

### 2.1 Why slope units, not a grid

A landslide belongs to a hillside, not to a square. Slope units — polygons bounded by ridges and drainage lines — are the standard mapping unit in landslide susceptibility work and have been used in operational regional early-warning models based on slope units. They also mean each unit has a well-defined single flow direction, which the projection operator needs.

### 2.2 Procedure

1. **DEM.** Copernicus DEM 30 m (preferred) or SRTM 30 m. Download tiles covering the study bounding box plus a 10 km buffer, so flow routing does not terminate at the study boundary.
2. **Reproject** to EPSG:32643.
3. **Fill sinks.** Use `pysheds`, `richdem` or WhiteboxTools. **Skipping this step causes flow paths to terminate mid-hillside and will silently break the projection operator.**
4. **Flow direction (D8)** and **flow accumulation**.
5. **Half-basin partition:** delineate sub-basins, then split each along its main drainage line into left-bank and right-bank half-basins. These are the slope units.
6. **Merge and split** to a target median area of 0.25 km². Discard slivers below 0.02 km².
7. **Record** for each unit: id, area, centroid, mean and max slope, mean elevation.

### 2.3 Quality check

Overlay the landslide catalog on the slope units. Every landslide crown must fall inside exactly one unit. Count units containing at least one landslide — this is your positive class size for Part 1 and must be reported.

---

## 3. Building the channel network N

1. **Channel definition:** cells with flow accumulation corresponding to a contributing area above **1 km²** are channel cells.
2. **Vectorise** into a connected line network.
3. **Split into reaches** of nominal 500 m. Always split at confluences.
4. **Build the graph:** nodes at reach endpoints, directed edges pointing downstream.
5. **Attributes per reach:** id, length, mean bed slope `S`, upstream contributing area `A`, Strahler order, downstream neighbour id, from/to node ids.

### 3.1 Validation

Overlay the derived network on OpenStreetMap or Bhuvan river layers. The Punnapuzha and Chaliyar must appear, correctly connected, with flow directed downstream. **Check specifically that a path exists from the Mundakkai crown injection point to the Chaliyar at Nilambur.** If it does not, the DEM processing is wrong and everything in Phase 4 will fail silently.

---

## 4. The projection operator π : H → N

This is the physical bridge between the two domains.

**For every slope unit `u`:**

1. Take the unit's representative point (its highest point, matching the crown convention used for landslide records).
2. Follow the D8 flow path downhill, cell by cell.
3. Stop at the first channel cell reached. That cell's reach is the **injection reach**, `π(u)`.
4. Record the path length, `d_slope(u)`, in km.

**Store:** `unit_id, injection_reach_id, d_slope_km, injection_node_id`.

**Check:** every slope unit must have an injection reach. Units that do not are either outside the network's drainage or hit an unfilled sink — investigate, do not discard silently.

*In plain English:* for every hillside in the study area, this table records where debris from that hillside would first enter a river, and how far it would have to travel to get there.

---

## 5. Downstream reachability and travel time

These two tables are the computational backbone of the cascade model.

### 5.1 Channel celerity

For each reach `s`, wave speed:

> `v(s) = v₀ · S(s)^0.3`

with `S` the bed slope. `v₀` is a model parameter fitted later, but for building the precomputed table we use a **reference** `v₀` chosen so speeds fall in the physical range 1–5 m·s⁻¹. The table stores **the path integral of `1/S^0.3` per unit length**, so that the actual travel time for any fitted `v₀` is obtained by dividing — meaning the table never has to be recomputed during optimisation.

> `T_ref(p → s) = Σ over reaches on the path of ( length / S^0.3 )`, and `τ(p → s) = T_ref(p → s) / v₀`

**This design decision is important.** It means `v₀` can be a learned parameter without invalidating the precomputed table.

### 5.2 Downstream reachability

For every network node `p`, compute the set of reaches downstream of it, truncated at a **maximum reference travel time** corresponding to about 48 hours at the slowest plausible celerity.

Store as a sparse structure (CSR): for each source node, an array of downstream reach ids and the corresponding `T_ref` values.

### 5.3 Contributing area ratio

For each (source node, downstream reach) pair, precompute `A(p)/A(s)`. This feeds the dilution term `Ω^ψ`. Store alongside `T_ref`.

### 5.4 Cost and check

This is the most expensive computation in the project. Budget up to four hours. Run it once, cache to disk, version it, and never recompute inside an optimiser loop.

**Check:** pick the Mundakkai injection point and list its downstream reaches with their reference travel times. Verify that Chooralmala appears within a few kilometres and the Chaliyar at Nilambur appears further along, in the right order. Print this list and put it in the Phase 2 report — it is a convincing, concrete demonstration that the spatial framework works.

---

## 6. Static covariates

Computed once per slope unit and per reach.

| Variable | Source | Notes |
|---|---|---|
| elevation, slope, aspect | DEM | mean and max per unit |
| plan and profile curvature | DEM | |
| topographic wetness index | DEM | `ln(a / tan β)` |
| flow accumulation | DEM | |
| drainage density | network | channel length per unit area in the neighbourhood |
| distance to nearest stream, `w(x)` | network | needed for the flood → landslide corridor |
| `d_slope`, injection reach | §4 | |
| contributing area | DEM | for reaches |
| soil texture, depth, organic carbon | SoilGrids 250 m | resampled to unit |
| land cover class | ESA WorldCover 10 m | majority class and fractions per unit |
| land cover change 2011 → 2024 | Sentinel-2 | forest loss, built-up gain |
| lithology, geomorphology | GSI Bhukosh layers | if available; optional |

---

## 7. Daily covariates

Computed per unit, per day, for the full period.

| Variable | Source | Notes |
|---|---|---|
| `R1, R3, R7, R15, R30` | IMD 0.25° daily; IMERG 0.1° for event-scale | Rolling sums ending on the day |
| catchment-averaged rainfall for reaches | same, weighted by contributing area | **Rain on the hills is what floods the river.** For a reach, average rainfall over its upstream catchment, not point rainfall at the reach |
| soil moisture, layers 1–4 | ERA5-Land | ~9 km, hourly, aggregated to daily |
| antecedent precipitation index (API) | derived from IMERG | **Required.** ERA5-Land has multi-day latency and cannot be used in real time. Build an API from rainfall alone, calibrate it against ERA5-Land offline, and use the API at run time. See Document 09, §1 |
| discharge | GloFAS | per reach |

### 7.1 Rainfall reconciliation — a required check

IMD (0.25°, gauge-based) and IMERG (0.1°, satellite) will disagree, particularly for extreme orographic rainfall in the Ghats, where satellite estimates are known to underestimate.

1. Extract both for the Wayanad cell, July 2024.
2. Confirm you can reproduce roughly **204 mm on 29 July and 373 mm on 30 July**.
3. **If you cannot reproduce those numbers, your extraction is wrong. Fix it before proceeding.** This is the single best sanity check available in the whole project.
4. Compare the two products across the region and document which you use where. Recommended: IMD for long-record climatology and antecedent windows, IMERG for event-scale intensity and for all real-time work.

---

## 8. The one function everything uses

```
get_covariates(unit_id | reach_id, date) → dict of all static and daily variables
```

Written once, tested once, imported everywhere. Both Part 1 and Part 2 call it. If two parts of the project compute rainfall differently, results will not be comparable and the bug will be found in November.

---

## 9. Quality gates

| # | Gate | Test |
|---|---|---|
| 1 | Every slope unit has an injection reach | count of nulls = 0 |
| 2 | Network is connected and correctly directed | path exists Mundakkai → Chooralmala → Chaliyar at Nilambur, in that order |
| 3 | Every catalog event maps to exactly one spatial unit | join produces no nulls, no duplicates |
| 4 | Rainfall reproduces the published Wayanad totals | within ~15% for IMD; document any IMERG shortfall |
| 5 | All layers in EPSG:32643, aligned | automated CRS check on load |
| 6 | Travel-time table built and cached | file exists, spot-check printed |

---

## 10. Four-person split

| Member | Task |
|---|---|
| **Chetan** | DEM processing, sink filling, flow routing, slope units, channel network, projection operator |
| **Raj** | Downstream reachability, travel-time table, contributing area ratios, sparse storage format. **This is the computational backbone of your Phase 4 model — build it yourself so you understand it** |
| **Yugant** | Static covariates: soil, land cover, land-cover change, terrain derivatives |
| **Ronit** | Daily covariates: rainfall windows, catchment averaging, ERA5-Land, the API proxy, and the rainfall reconciliation check |

---

## 11. Deliverables

- All artefacts in §1, cached and versioned
- `figures/phase2_slope_units.png`, `phase2_network.png`
- `figures/phase2_wayanad_flowpath.png` — the Mundakkai crown, its D8 path, the injection point, and the downstream reaches coloured by travel time. **This is your best Review 1 slide.** It makes the whole novel idea visible in one picture
- `reports/phase2_rainfall_reconciliation.md`
- `reports/phase2_spatial_framework.md` — the written specification, matching Document 02 §2
