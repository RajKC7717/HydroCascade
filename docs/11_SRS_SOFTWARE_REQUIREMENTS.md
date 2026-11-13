# Document 11 — Software Requirements Specification

**Owner: Ronit Dahiwal** · Brief reference document.

---

## 1. Functional requirements

Each requirement has an ID, a verification method, and the phase that delivers it.

### Data layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-01 | Ingest landslide inventories from at least three independent sources into one schema | Catalog row count by source | 1 |
| FR-02 | Detect flood events from discharge using the `Q₂` exceedance rule with 3-day declustering | Unit test on a synthetic hydrograph with known exceedances | 1 |
| FR-03 | Deduplicate events within 500 m and 1 day using a written merge rule | Automated check: zero remaining duplicates | 1 |
| FR-04 | Every catalog row carries source URL, access date and accessor | Automated provenance check, 100% pass | 1 |
| FR-05 | Report the flood-definition precision against SAR corroboration | Number reported in `phase1_flood_definition.md` | 1 |

### Spatial layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-06 | Partition the region into slope units, median area ~0.25 km² | Unit count and area histogram | 2 |
| FR-07 | Derive a directed channel network of 500 m reaches from the DEM | Path exists Mundakkai → Chooralmala → Chaliyar, in order | 2 |
| FR-08 | Compute the projection operator `π(u)` for every slope unit | Zero null injection reaches | 2 |
| FR-09 | Precompute downstream reachability, reference travel time and area ratios | Cached files exist; Mundakkai spot-check printed | 2 |
| FR-10 | Provide `get_covariates(unit, date)` returning all static and daily variables | Unit test; used by both Part 1 and Part 2 | 2 |
| FR-11 | Reproduce the published Wayanad rainfall totals from the ingested data | Within ~15% for IMD | 2 |

### Root cause layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-12 | Rank drivers using correlation, permutation importance and SHAP | Three-column ranked table | 3 |
| FR-13 | Train susceptibility models under **spatial block** cross-validation | Grouped CV by sub-basin; both random and spatial numbers reported | 3 |
| FR-14 | Export **calibrated** susceptibility `S_i` in (0,1) | Reliability diagram before and after calibration | 3 |
| FR-15 | Export the top 2–4 drivers as `ẽ(u)` | JSON with exactly 2–4 entries | 3 |
| FR-16 | Implement baselines B1, B2, B3 on the same splits and metrics | Baseline metrics table | 3 |

### Cascade layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-17 | Compute the multitype spatio-temporal log-likelihood on the quadrature grid | Parameter recovery on simulated data | 4 |
| FR-18 | Support isotropic 2D, network-directed, and travel-time-shifted kernels, switchable by config | Each ablation runs from a config flag | 4 |
| FR-19 | Enforce parameter constraints and compute the branching ratio after every fit | Branching ratio < 1 asserted | 4 |
| FR-20 | Support a mechanism mixture kernel with 1, 2 or 3 components | Config-selectable | 4 |
| FR-21 | Fit by MLE with multi-start; persist every fit as a JSON record with seed and git commit | Fit records exist for every reported number | 4 |

### Validation layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-22 | Enforce the train/validation/test split with a **leakage assertion that raises on violation** | Deliberate violation test raises the error | 5 |
| FR-23 | Run all nine ablations A1–A9 and produce the comparison table | Nine-row table | 5 |
| FR-24 | Run leave-one-cascade-out across six events | Lead-time distribution reported | 5 |
| FR-25 | Compute uncertainty via Laplace, profile likelihood and bootstrap | Intervals for headline parameters | 5 |
| FR-26 | Produce reliability diagram, Brier decomposition and CRPS | Figures and numbers | 5 |
| FR-27 | Report lead time **and** false-alarm rate together, never separately | Enforced in the reporting function | 5 |

### Output layer

| ID | Requirement | Verified by | Phase |
|---|---|---|---|
| FR-28 | Convert intensity to occurrence probability over a stated area and horizon | Unit test against the analytic void probability | 6 |
| FR-29 | Produce **exact** additive attribution, shares summing to 1 | Unit test: sum of shares = 1.0 within tolerance | 6 |
| FR-30 | Assign four alert levels using local percentiles, with hysteresis | Threshold sweep and trade-off curve | 6 |
| FR-31 | Dashboard replay mode with a time slider over any historical period | Wayanad replay demonstrated | 6 |
| FR-32 | Flag stale data rather than showing "Normal" | Simulated feed failure test | 6 |

---

## 2. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | One full likelihood + gradient on the complete catalog | **< 60 s**, laptop CPU |
| NFR-02 | One full fit, single start | < 2 h |
| NFR-03 | Incremental update after a new event | **< 2 s** |
| NFR-04 | Hourly full recompute in live mode | < 3 min |
| NFR-05 | Precomputation of travel-time and reachability tables | One-off, < 4 h |
| NFR-06 | Total disk footprint | < 50 GB |
| NFR-07 | Reproducibility: clone → money figure | documented in README, tested by a member who did not write it |
| NFR-08 | Determinism: fixed seeds everywhere | two runs with the same seed produce identical numbers |
| NFR-09 | Numerical stability | `float64`; log-sum-exp; no `nan` in any reported output |
| NFR-10 | Traceability | every reported number links to a fit record with a git commit |

---

## 3. Technology stack

| Layer | Technology | Justification |
|---|---|---|
| Language | **Python 3.12** | Pinned at 3.12 because the `tick` library's fitting classes break on 3.13 |
| Environment | `conda` / `micromamba`, pinned `environment.yml` | Geospatial binaries have complex dependencies; reproducibility is a stated goal |
| Raster I/O | `rasterio`, `rioxarray`, `xarray` | Rasters, netCDF, GRIB |
| Vector | `geopandas`, `shapely`, `fiona`, `pyproj` | Polygons, networks, reprojection |
| Terrain & hydrology | `pysheds` / `richdem` / `WhiteboxTools` | Sink filling, flow routing, **flow-path distance** |
| Graph | `networkx` (build), `scipy.sparse` (runtime) | Network build in `networkx`; runtime lookups from sparse CSR arrays |
| Numerics | `numpy`, `scipy`, `pandas`, `pyarrow` | Standard; Parquet for large tables |
| ML | `scikit-learn`, `xgboost` | Part 1 baselines |
| Interpretability | `shap` | Transparent driver ranking |
| **Model fitting** | **PyTorch** | Autodiff for MLE. **Our intensity function does not exist in any library** — a hybrid-support kernel with travel-time shift and covariate-dependent background must be written by hand |
| Point-process cross-check | `tick`, isolated env, Python 3.12 | **Simulation only**, to verify our fitter recovers known parameters. Never a runtime dependency |
| Statistics | `scipy.stats`, `statsmodels` | Likelihood-ratio tests, profile likelihood, Gumbel/L-moments |
| Cloud processing | Google Earth Engine Python API | Sentinel-1 SAR flood corroboration |
| Data access | `cdsapi` (Copernicus), `earthaccess` (NASA) | Programmatic, reproducible downloads |
| Visualisation | `matplotlib`, `contextily` | Publication figures and maps |
| Dashboard | `streamlit` | One page, fast to build. No auth, no database |
| Version control | Git, GitHub, public repository | |
| Testing | `pytest` | Unit tests for every FR marked "unit test" |

---

## 4. Module structure

```
hydrocascade/
  README.md
  LIMITATIONS.md
  environment.yml
  data/                        gitignored except PROVENANCE and catalog
  src/
    ingest/
      landslides.py            E1–E6
      floods.py                FR-02 flood detection
      rainfall.py              M1, M2
      hydro.py                 H1–H3
    spatial/
      slope_units.py           FR-06
      network.py               FR-07
      projection.py            FR-08  π: H → N
      travel_time.py           FR-09  the precomputed tables
    features/
      covariates.py            FR-10  get_covariates()
      rainfall_windows.py
      api_index.py             the real-time wetness proxy
    rootcause/
      sampling.py              negative sampling
      ranking.py               FR-12
      models.py                FR-13
      calibration.py           FR-14
      baselines.py             FR-16  B1, B2, B3
    cascade/
      likelihood.py            FR-17  the log-likelihood
      kernels.py               FR-18  isotropic / network / shifted
      mechanisms.py            FR-20  the mixture kernel
      fit.py                   FR-21  optimiser, multi-start, records
      diagnostics.py           FR-19  branching ratio, residuals
    validate/
      splits.py                FR-22  leakage assertion
      ablations.py             FR-23
      loco.py                  FR-24
      uncertainty.py           FR-25
      metrics.py               FR-26, FR-27
      backtest.py              the Wayanad protocol
    output/
      probability.py           FR-28
      attribution.py           FR-29
      alerts.py                FR-30
    dashboard/
      app.py                   FR-31, FR-32
  tests/
  notebooks/                   numbered, exploratory, never imported from
  figures/
  reports/
  experiments/register.csv
  paper/
  docs/                        this document suite
```

---

## 5. Coding rules

1. **Notebooks are never imported from.** Production logic lives in `src/`.
2. **No Python loop inside the likelihood.** Assemble a flat tensor, evaluate once.
3. **Every kernel variant is a config flag**, not a code edit. Ablations must be reproducible from a config file alone.
4. **Fixed seeds everywhere.** Record the seed in the fit record.
5. **Every reported number has a fit record** containing parameters, intervals, log-likelihood, seed, git commit and wall-clock time.
6. **One owner, one reviewer** per module. Never four owners.
7. **Every FR marked "unit test" has a `pytest` test** that would fail if the requirement were broken.

---

## 6. Verification matrix

| Verification method | Requirements |
|---|---|
| Unit test | FR-02, FR-03, FR-10, FR-17, FR-19, FR-22, FR-28, FR-29 |
| Automated check on data | FR-04, FR-08, FR-15 |
| Figure or report | FR-05, FR-11, FR-12, FR-14, FR-23, FR-24, FR-25, FR-26 |
| Timed measurement | NFR-01 to NFR-05 |
| Independent reproduction by another member | NFR-07 |
| Deliberate failure injection | FR-22 (leakage), FR-32 (stale data) |

---

## 7. Definition of done, for the whole system

The system is done when a person who has never seen the project can:

1. clone the repository,
2. create the environment from `environment.yml`,
3. follow the README,
4. and regenerate the money figure —

with no undocumented step, no manual file placement, and no number that cannot be traced to a fit record.
