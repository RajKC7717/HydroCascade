# HydroCascade

**Root cause analysis and a cross-hazard point process for rainfall-triggered
disaster chains in the Kerala Western Ghats.**

Rain falls. A hillslope fails. The debris enters a stream. Hours later a river
floods somewhere else. Most hazard models treat those as three unrelated
problems. This one treats the chain as the object of study, and asks a question
you can be wrong about: *given a landslide here, at this hour, how much does
flood risk rise **there**, and when?*

> **All six phases are built.** Catalog, spatial framework, root-cause study,
> the fitted cascade model, the experiment suite and the alert layer all run
> end to end. Four datasets are still account-gated and running on labelled
> synthetic stand-ins, so many numbers are rehearsals rather than results —
> see [LIMITATIONS.md](LIMITATIONS.md), which is honest to the point of being
> uncomfortable, by design.

---

## The one idea worth reading first

Landslides and floods do not live on the same kind of space.

- Landslides happen on a **2D hillslope surface** `H`, partitioned into 17,138
  slope units.
- Floods happen on a **1D directed river network** `N`, cut into 9,038 reaches
  of ~500 m.
- A **projection operator** `π: H → N` connects them: follow the D8 flow path
  from a hillslope point until it hits the first channel cell.

Excitation then travels as a **wave, not a field**. A landslide at `x` raises
flood intensity at reach `s` only if `s` is downstream of `π(x)`, only after the
travel time `τ = T_ref/v₀`, and diluted by how much the catchment has grown:
`Ω = (A(π(x))/A(s))^ψ`.

The travel-time table stores the path integral `T_ref = Σ length/S^0.3`, which
does **not** depend on the fitted celerity coefficient `v₀`. So Phase 4's
optimiser divides instead of rebuilding a 716,519-pair table at every step.
Doc 05 §5.4 budgets four hours for that build; it takes **2.3 seconds**.

---

## Reproduce it from an empty `data/` directory *(NFR-07)*

Nothing below needs an account. The DEM, the basin boundary and the rainfall are
all free and login-free.

```powershell
git clone <repo> && cd edisem1
python -m venv .venv --system-site-packages
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -e .
.venv\Scripts\Activate.ps1

$env:HYDROCASCADE_USER = 'your name'    # goes in every provenance record

hydrocascade aoi          # study boundary from HydroBASINS
hydrocascade dem          # Copernicus GLO-30, mosaicked and reprojected
hydrocascade flow         # sink fill, D8 directions, flow accumulation
hydrocascade network      # channel network, 500 m reaches, directed graph
hydrocascade units        # slope units — the hillslope domain H
hydrocascade projection   # the operator pi: H -> N
hydrocascade traveltime   # downstream reachability + reference travel time
hydrocascade rainfall     # CHIRPS download, daily windows, API
hydrocascade catalog      # data/catalog/events.csv
hydrocascade covariates   # terrain covariates per unit and reach
hydrocascade figures      # every figure named in Docs 04 and 05
hydrocascade report       # reports/phase1_flood_definition.md

hydrocascade rootcause    # Phase 3: drivers, SHAP, susceptibility, B1-B3
hydrocascade cascade      # Phase 4: verify S1/S2, then fit S3-S5
hydrocascade experiments  # Phase 5: ladder, ablations, backtest, money figure
hydrocascade realtime     # Phase 6: alert thresholds and the architecture

hydrocascade check        # the quality gates, all six phases
hydrocascade app          # the Explorer
```

Every stage is idempotent and cached; add `--force` to recompute one. Every
threshold, date, CRS and path comes from `config/default.yaml`, so an experiment
is reproducible from a config file alone (Doc 11 §5 rule 3, generalised to the
whole pipeline).

**NFR-07 is not met until a team member who did not write this runs the block
above on a clean machine and gets the same figures.** That has not happened yet.

### Environment

Python **3.11** with `venv` + `pip`, not the 3.12 + conda of Doc 11 §3. That pin
exists only for `tick`, which Doc 07 §2 wanted as an independent cross-check of
the Hawkes likelihood. `tick` does not install here and the same paragraph of
Doc 07 already flags it as version-fragile, so the cross-check is **substituted**:
three structurally different implementations of the likelihood — an O(n²) double
sum, the O(n) Ozaki recursion, and the vectorised torch version — must agree to
machine precision. They do, exactly. That is real evidence and weaker than an
external library, so the gate records `BLOCKED`, not `PASS`.

3.11 is also what the geospatial wheels are most stable on today.

`numpy` is pinned below 2.0. A 2.x install in this venv shadows the system
1.26 and breaks the OpenCV ABI (`_ARRAY_API not found`) on this machine.

---

## What you should look at

| | |
|---|---|
| **`hydrocascade app`** | The Explorer. Nine tabs; start with *The cascade path* |
| `figures/money_figure.png` | **The figure the project exists to produce** — the Wayanad backtest with the predicted downstream arrival window shaded |
| `figures/phase2_wayanad_flowpath.png` | The Review-1 slide: crown → injection → Chooralmala → Nilambur, with travel times |
| `reports/phase1_flood_definition.md` | The flood rule, its sensitivity, the blind spot, and two honest non-results |
| `reports/quality_gates.md` | Every gate across all six phases, with `BLOCKED` reported separately from `FAIL` |
| `LIMITATIONS.md` | Read this before quoting any number |

---

## Where it stands

Measured on the Chaliyar pilot basin, 1 Jun 2018 – 31 Dec 2024:

| | |
|---|---|
| study area | 2,969 km² (published Chaliyar ≈ 2,923 km², so within 2%) |
| DEM | 2,622 × 3,458 at 30 m; 4 cells of 9.07 M still undrained |
| channel network | 9,038 reaches · 4,358 km · density 1.47 km/km² · Strahler 5 |
| slope units | 17,138, median 0.194 km² |
| travel time | 716,519 (source, downstream) pairs · median celerity 1.94 m/s |
| catalog | 9,765 events — **1 real, 9,764 synthetic stand-ins** |
| tests | 147 passing |
| gates | 9 pass · 1 fail · 2 blocked |

Two independent checks that the geometry is right, neither of which was tuned:

- The Mundakkai crown samples at **1,586.9 m** against the GSI field report's
  ~1,550–1,600 m, and the elevation sequence 1,587 → 835 → 47 m confirms correct
  downhill ordering.
- The routed channel distance from the injection reach to the Chooralmala bridge
  is **6.21 km**, against the GSI FIR's documented 6–8 km runout.

And the three that do not pass, stated plainly:

- **Gate P2.4 FAILS.** CHIRPS captures 71 mm of the published 577 mm at the
  crown. Diagnosed as orographic displacement, not an extraction bug; needs IMD
  or IMERG.
- **Gate P1.1 is BLOCKED.** 9,764 of 9,765 catalog rows are synthetic, so the
  ≥ 200 real-landslide volume gate cannot be evaluated at all. `BLOCKED` is
  reported separately from `FAIL` precisely so "we have not got the data" can
  never be read as "the check passed".
- **Gate P1.6 is BLOCKED — FR-05 is unmet.** The flood rule's precision against
  Sentinel-1 inundation was not measured, because Otsu thresholding of VV
  backscatter needs a Google Earth Engine account. The report prints the exact
  recipe and refuses to print a number. A plausible-looking precision nobody
  computed would be worse than an honest gap.

---

## Rules the code enforces, not just documents

From Doc 04 §2 and Doc 10 §8:

- **`data/raw/` is never edited. Ever.** Downloads land in
  `data/raw/<SOURCE>/<YYYY-MM-DD>/` with a `PROVENANCE.md` written *at download
  time*.
- **No CSV is ever opened and saved in Excel.** It silently mangles dates and
  truncates coordinate precision. Coordinates are written at fixed 6 dp.
- **Every catalog row carries a `source_url` and an `accessed_by` — no
  exceptions (FR-04).** Documented events whose coordinates are not yet sourced
  live in `events_PENDING.csv`, so `events.csv` is provenance-clean *by
  construction* and the automated check keeps meaning something.
- **A stand-in can never pass for real data.** Synthetic rows are labelled in
  the catalog, watermarked on figures, banner-flagged in the Explorer, and cause
  the affected gate to report `BLOCKED`.
- **`data/reference/` is scored against, never fitted to.**

---

## Layout

```
config/default.yaml         every tunable knob; nothing is hard-coded in src/
data/
  raw/                      never edited; one PROVENANCE.md per download
  interim/  processed/      derived; processed/spatial holds the framework
  catalog/                  events.csv, cascade_pairs.csv, events_CHANGELOG.md
  reference/                documented floods — scored against, never fitted to
src/hydrocascade/
  ingest/                   HydroSHEDS, Copernicus DEM, CHIRPS, discharge,
                            landslide inventories, adapters, corroboration
  spatial/                  flow routing, network, slope units, pi, travel time
  features/                 rainfall windows and the antecedent index
  catalog/                  schema, dedup, completeness, cascade pairs, build
  qa/                       executable quality gates, the flood-definition report
  viz/                      every figure Docs 04 and 05 name
  dashboard/                the Explorer
tests/                      147 tests
docs/                       the 13 specification documents this is built from
```

`src/hydrocascade/spatial/flow.py` is the one module to read if you read only
one: depression filling, Barnes-style flat resolution, D8 by steepest *slope*
(not steepest drop), ordered-scan accumulation and a single-pass trace-to-channel,
all from first principles with the failure modes documented in the comments.

---

## Data

Free and used now: **Copernicus DEM GLO-30** (AWS), **HydroBASINS** (HydroSHEDS),
**CHIRPS** (added as dataset M4 to Doc 10 §3).

Currently synthetic: the **landslide inventory** (until one is placed in
`data/raw/`) and **GloFAS** discharge. Optional upgrades: **IMD** or **IMERG**
rainfall in place of CHIRPS. Each has a reader for the provider's files and a
labelled stand-in, so the pipeline never blocks on an approval email — and never
reports the stand-in as though it were the real thing. ERA5-Land is on no
pipeline path (the rainfall-based API index replaces it).

**Step-by-step: [docs/15_HOW_TO_RUN.md](docs/15_HOW_TO_RUN.md)** — what to
download, which folder it goes in, what to run, and where the results appear.

---

## Tests

```powershell
.venv\Scripts\python -m pytest tests -q
```

147 tests across 8 files. The ones worth knowing about: `test_flow.py` verifies routing on
hand-computed grids (a plane, a pit, a flat, nodata escapes, mass conservation);
`test_floods.py` verifies L-moments and declustering on hydrographs whose right
answers are worked out in the docstrings; `test_corroboration.py` asserts that
the SAR precision test *raises* rather than returning a plausible number; and
`test_dashboard.py` renders the whole Explorer and fails if any tab throws.
