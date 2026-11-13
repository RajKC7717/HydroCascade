# Document 01 — Project Definition and System Architecture

---

## 1. The project in one paragraph

Disasters arrive in chains. Heavy rain soaks a hillside, the hillside fails, the failed material becomes a fast debris flow, and that debris flow causes flooding and sediment damage many kilometres downstream. Every warning system in use today watches one link of that chain at a time, so the second and third disasters arrive with no warning. HydroCascade builds one model that watches the whole chain. It learns from real historical data which physical conditions cause hazards in a region, then uses a continuously updating mathematical model in which one type of hazard raises the risk of a *different* type of hazard downstream, along the actual river network, after the actual travel time of the flow. We test it by hiding a real disaster from the model and asking whether it would have raised an alarm in time.

---

## 2. The disaster we chose, and why

### The event

**Mundakkai–Chooralmala, Wayanad district, Kerala. 30 July 2024, between 02:00 and 04:30 IST.**

| Fact | Value |
|---|---|
| Rainfall | ~204 mm in the first 24 h, ~373 mm in the next 24 h; about 572 mm in 48 h |
| Landslide crown | 11.4654 °N, 76.1348 °E, elevation ~1,550–1,600 m |
| Slope angle at source | ~25–40°, overburden thickness under 5 m |
| Mechanism | Planar failure transitioning into a long-runout debris flow |
| Runout | About 6–8 km |
| Flow path | Head of the Punnapuzha stream → Chooralmala → Chaliyar river |
| Downstream damage | Bridge destroyed near 11.4992 °N, 76.1601 °E; flash floods, bank erosion, sediment deposition; debris carried tens of kilometres to Pothukal in Malappuram |
| Deaths | Over 400 |

### Why this event

1. **The whole chain is documented.** Crown coordinates, minute-level timing, rainfall totals, runout distance, downstream river, and a government field report published within days.
2. **It is a genuine cascade, not a single hazard.** Rainfall → landslide → debris flow → flash flood → sediment. Four links.
3. **The gap is stated by others, about this event.** A January 2026 *Scientific Reports* paper used Wayanad as one of two Indian case studies and called for real-time threshold monitoring that its own method could not provide.
4. **A near-identical event happened 2–3 km away in 2019** (Puthumala). So the region has repeat behaviour we can learn from.

### What we actually study

Not one event. **The Western Ghats of Kerala, June 2018 to December 2024.** A statistical model needs thousands of events to learn from. Kerala's published inventories contain thousands of mapped landslides — the August 2018 monsoon alone produced roughly 4,700.

Wayanad 2024 is deliberately **removed from training** and used as the final unseen test.

---

## 3. What goes in

Two categories of input.

### Events — what happened, when, where, how big

| Event type | Source | What we record |
|---|---|---|
| Landslides | GSI Bhukosh inventory, GSI Bhusanket field reports, published Kerala inventories (ESSD 2018 dataset, NRSC satellite inventory) | date, time if known, initiation point, mapped area, source |
| Floods | GloFAS v4.0 discharge reanalysis, with corroboration from Sentinel-1 SAR inundation and CWC gauges | date, river reach, peak discharge relative to the 2-year return-period threshold |

The exact rule that turns a discharge time series into a "flood event" is specified in Document 02, §3. This is not left vague.

### Conditions — the physical situation everywhere, every day

| Category | Source | Variables |
|---|---|---|
| Rainfall | IMD 0.25° gauge-based grid; GPM IMERG 0.1° half-hourly | 1, 3, 7, 15, 30-day accumulations |
| Terrain | Copernicus DEM 30 m | elevation, slope, curvature, topographic wetness index, flow direction, flow accumulation, drainage density, distance to stream |
| Soil | SoilGrids 250 m | texture, depth, organic carbon |
| Land cover | ESA WorldCover 10 m | class, and 2011 vs 2024 change |
| Wetness | ERA5-Land | volumetric soil water by layer |
| Hydrology | GloFAS, HydroSHEDS | discharge, sub-basin boundaries, contributing area |

Everything is free and public. No paid or restricted data.

---

## 4. What we do

### Part 1 — Root cause analysis

Find out which physical conditions actually cause hazards in this region, using real data instead of general description.

Method: assemble features, remove collinear ones, rank with permutation importance and SHAP, then train Random Forest and XGBoost under **spatial block cross-validation** (holding out whole sub-basins, not random pixels, because neighbouring pixels are nearly identical and random splitting inflates the score).

**Two outputs, and both go directly into Part 2:**

- `S_i(x)` — a calibrated susceptibility score for every location, for each hazard type
- `ẽ(x)` — the top two to four ranked drivers, standardised

### Part 2 — Cross-hazard cascade model

Take the self-exciting point process from earthquake science. It says risk now = background risk + a push from every past event, fading with time and distance. Extend it in four ways:

1. **Cross-type excitation.** A landslide raises flood risk. The standard model cannot do this.
2. **A hybrid spatial support.** Landslides are points on a 2D hillslope surface; floods are points on a 1D river network. A projection operator maps a landslide to the point where its flow path first meets a channel.
3. **Wave propagation, not instant spreading.** A downstream reach feels the effect after the *travel time* of the flow, not immediately. The excitation arrives late and dilutes with catchment size.
4. **Mechanism-separated timing.** Three named physical mechanisms with three different timescales, and the model estimates which dominates.

Part 1's susceptibility surface becomes Part 2's background term. That is what makes this one project rather than two.

---

## 5. What comes out

### The primary output

A continuously updating **conditional intensity** for each hazard type at each location, converted to an **occurrence probability** over a stated time window, with an **uncertainty band** and an **exact attribution** of what is driving it.

Because the intensity is a sum of terms by construction, attribution is exact rather than approximated:

> **Chooralmala reach — flood probability, next 24 h: 0.61 [0.44 – 0.75]**
> - 48% from the landslide at Punchirimattam 4.2 h ago, 2.1 km upstream (expected arrival window 3–7 h)
> - 27% from 15-day antecedent rainfall (99.4th percentile locally)
> - 14% from 1-day rainfall
> - 11% background susceptibility

No existing hazard system produces that fourth column. It is the point of the whole project.

### The secondary outputs

- A ranked, defensible profile of what drives landslides and floods in the Kerala Western Ghats
- A baseline susceptibility model with honestly reported accuracy
- A four-level alert system (Normal / Watch / Warning / Emergency) with thresholds chosen on validation data, never on test data
- A replay-mode dashboard
- A lead-time versus false-alarm trade-off curve

### The single figure the project exists to produce

A two-panel chart sharing one time axis:

- **Top:** daily rainfall bars, the landslide risk for the Mundakkai unit with its uncertainty band, the alert thresholds as horizontal lines, and a vertical marker at 30 July 2024 02:17 IST. The gap is annotated as the lead time.
- **Bottom:** flood risk for the downstream Chaliyar reaches, with the model's predicted arrival window shaded, observed discharge overlaid, and the observed flood peak marked.

If that figure works, we have a paper. If it does not, we have an honest paper about why lead time is hard at this data resolution.

---

## 6. System architecture

### 6.1 The five layers

```
┌─── LAYER 1 · DATA INGESTION ────────────────────────────────────────┐
│  Event sources          Condition sources                           │
│  ─────────────          ─────────────────                           │
│  GSI Bhukosh            IMD gridded rainfall                        │
│  GSI Bhusanket FIRs     GPM IMERG (half-hourly)                     │
│  ESSD 2018 inventory    Copernicus DEM 30 m                         │
│  NRSC inventory         SoilGrids · ESA WorldCover                  │
│  GloFAS discharge       ERA5-Land soil moisture                     │
│  Sentinel-1 SAR         HydroSHEDS sub-basins                       │
│         │                        │                                  │
│         ▼                        ▼                                  │
│   events.csv              raw raster / netCDF store                 │
│   (one row per event,     (each folder has PROVENANCE.md)           │
│    fully sourced)                                                   │
└──────────────────┬──────────────────────────────────────────────────┘
                   ▼
┌─── LAYER 2 · SPATIAL FRAMEWORK ─────────────────────────────────────┐
│                                                                     │
│   DEM ──► fill sinks ──► flow direction ──► flow accumulation       │
│                                    │                                │
│              ┌─────────────────────┴────────────────────┐           │
│              ▼                                          ▼           │
│    HILLSLOPE DOMAIN  H                        CHANNEL NETWORK  N    │
│    2D, slope units (~0.25 km²)                1D metric graph,      │
│    landslide events live here                 500 m reaches,        │
│                                               flood events live here│
│              └──────────► π : H → N ◄─────────┘                     │
│                  projection operator: follow the D8 flow path       │
│                  from a hillslope point to the first channel cell   │
│                  it reaches — the INJECTION POINT                   │
│                                                                     │
│   Also precomputed here: downstream reachability lists, network     │
│   distance, channel celerity, travel time, contributing area        │
└──────────────────┬──────────────────────────────────────────────────┘
                   ▼
┌─── LAYER 3 · ROOT CAUSE ENGINE (Part 1) ────────────────────────────┐
│                                                                     │
│   feature table  ──►  collinearity removal  ──►  ranking            │
│   (unit × day)        (VIF, correlation)         (permutation       │
│                                                   importance, SHAP) │
│                              │                                      │
│                              ▼                                      │
│         Random Forest / XGBoost, spatial block CV                   │
│                              │                                      │
│              ┌───────────────┴──────────────┐                       │
│              ▼                              ▼                       │
│      S_i(x) susceptibility          ẽ(x) top 2–4 drivers            │
│      (calibrated to a               (standardised)                  │
│       probability)                                                  │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │ becomes μ_i (background)     │ becomes g_i (receptivity)
               ▼                              ▼
┌─── LAYER 4 · CROSS-HAZARD CASCADE ENGINE (Part 2) ──────────────────┐
│                                                                     │
│   λ_i(u,t) = μ_i(u,t)  +  Σ over past events k of any type:         │
│                            α_ij · exp(η_j m_k) · g_i(u)             │
│                            · CONNECTIVITY(x_k → u)                  │
│                            · φ_ij( t − t_k − τ(x_k → u) )           │
│                                          ▲                          │
│                                          │                          │
│                      travel time: the excitation ARRIVES,           │
│                      it does not appear instantly                   │
│                                                                     │
│   φ_ij = mixture of three mechanism components                      │
│          fast (bulking) · delayed (dam breach) · slow (aggradation) │
│                                                                     │
│   fitted by Maximum Likelihood Estimation, PyTorch autodiff,        │
│   on a daily × spatial-unit quadrature grid                         │
└──────────────────┬──────────────────────────────────────────────────┘
                   ▼
┌─── LAYER 5 · VALIDATION, ALERT AND OUTPUT ──────────────────────────┐
│                                                                     │
│   λ ──► P(≥1 event in Δt) ──► uncertainty band ──► alert level      │
│    │                                                                │
│    └──► exact attribution (the intensity is additive, so each       │
│         term's share is computed, not estimated)                    │
│                                                                     │
│   Validation: external baselines · ablations · temporal split ·     │
│   leave-one-cascade-out · spatial transfer to Maharashtra           │
│                                                                     │
│   Delivery: replay-mode dashboard · lead-time vs false-alarm curve  │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Why the architecture is shaped this way

**Layer 2 exists as its own layer because the spatial framework is a scientific decision, not plumbing.** The question "what does *location* mean in this model?" is answered here, once, and everything above depends on it. Building the projection operator, the downstream reachability lists and the travel-time table early also de-risks the novel part of the model, because those artefacts are needed in Layer 4 and are expensive to compute.

**Layers 3 and 4 are joined by two named objects, not by a sentence.** `S_i(x)` and `ẽ(x)` are files. If Layer 3 changes, Layer 4 changes. That is what "the root cause study feeds the cascade model" actually means in practice.

**Layer 4 separates connectivity from timing.** Whether a landslide can affect a reach at all (is it downstream? did the debris actually reach the channel?) is a different question from when it does (travel time) and how strongly (mechanism weights). Keeping these as three separate factors makes each one independently testable by ablation.

**Layer 5 does the arithmetic that converts a rate into a decision.** The model's native output is an intensity, which is not a probability and not an alert. Converting intensity → probability → uncertainty band → alert level is three explicit, documented steps, each of which a reviewer can check.

### 6.3 What runs when

| Mode | Purpose | Cadence | Deliverable this semester |
|---|---|---|---|
| **Fit mode** | Estimate all parameters from the historical catalog | Once, offline, minutes to hours | Yes — core |
| **Replay mode** | Roll the fitted model forward over a historical period as if live, for backtesting | Once per experiment | Yes — core |
| **Live mode** | Ingest new rainfall and events and update risk continuously | Hourly, with ~4–5 h data latency | Design document + working replay demo. **Not** a running 24/7 service |

We state that last row plainly. A semester project should deliver a validated model and a credible deployment design, not an unmonitored production service. Document 09 specifies the live-mode architecture fully so that it is implementable, without claiming it is deployed.

---

## 7. Scope boundaries

### In scope

- Two hazard types: landslide (`L`) and flood (`F`)
- One study region: Western Ghats of Kerala
- One time window: June 2018 – December 2024
- Prediction target: **hazard occurrence**, expressed as intensity and probability

### Explicitly out of scope

- **Loss, damage, casualties, vulnerability, exposure.** We predict whether a hazard occurs, not what it costs. This distinction is stated in Document 02, §1 and must be stated in the paper's abstract, because "risk" in disaster science normally includes consequence.
- **Water contamination as a third cascade stage.** Deferred; no validatable dataset. An optional sediment/turbidity proxy is specified.
- **Sub-hourly warning.** Data latency is about four hours at best. This is a nowcast layer, not a siren.
- **Replacing physics-based models.** This is a fast statistical layer that would sit alongside them, exactly as ETAS sits alongside physical models in earthquake agencies.

---

## 8. What would make this project fail, and the guard against each

| Failure mode | Guard |
|---|---|
| Not enough events to fit the model | Hard gate at end of Phase 1: fewer than 200 landslide events means widen the region before proceeding. Document 04 |
| Flood events defined vaguely, so cross-hazard results are meaningless | A single, written, testable definition with a measured corroboration rate. Document 02, §3 |
| Cross-excitation not statistically detectable | Planned for. Report the negative result with a power analysis, and let the lead-time analysis carry the paper. Document 08, §6 |
| The model is tuned on Wayanad and tested on Wayanad | Three-way temporal split plus leave-one-cascade-out over six events plus spatial transfer to Maharashtra. Document 08, §4 |
| Optimiser produces numbers that look fine but are meaningless | Parameter recovery on synthetic data before touching real data; multi-start; profile likelihood; branching-ratio check. Documents 07 and 08 |
| The system is too slow to run | Complexity budget and eight named optimisations, with a stated wall-clock target. Document 02, §10 |
