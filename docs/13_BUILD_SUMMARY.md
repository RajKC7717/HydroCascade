# Document 13 — Build Summary: what actually exists

**Status:** implementation record, not a specification. Documents 00–12 say what
*should* be built. This one says what *has* been built, what data went into it,
what came out, and what does not work.

**Generated against the repository state of 17 September 2026.** Every number
here was read from a built artefact or a passing test. Where a number would
drift as the pipeline is re-run, this document points at the generated report
that owns it rather than copying it.

---

## 0. The short version

| | |
|---|---|
| Scope built | **All six phases.** Catalog, spatial framework, root-cause study, cascade model, experiment suite, alert layer |
| Study area | Chaliyar pilot basin, **2,969 km²**, derived hydrologically |
| Period | 1 Jun 2018 – 31 Dec 2024 (holdout opens 2024-06-01) |
| Code | **15,543 lines** across 73 modules in `src/hydrocascade/` |
| Tests | **147 passing** across 8 files (1,891 lines) |
| Pipeline | **18 CLI stages**, each idempotent and cached |
| Real datasets downloaded | 3 (Copernicus DEM, HydroBASINS, CHIRPS) — all free, no login |
| Synthetic stand-ins | 4 (GloFAS, ERA5-Land, IMERG, IMD) — labelled everywhere |
| Visible output | Streamlit Explorer (9 tabs) + 14 figures + 14 reports |

**The honest headline.** The machinery is complete and tested end to end. The
*science* is not, because four datasets need accounts and the landslide catalog
holds one real event. Every number derived from a stand-in is labelled at
source, watermarked on figures, banner-flagged in the dashboard, and reported as
`BLOCKED` rather than `PASS` by the gate that depends on it.

---

## 1. Scope, and how it changed

The build ran in two commissions.

**First:** Phases 1–2 only, built deeply, on the Chaliyar pilot basin, with
adapters and synthetic generators for the account-gated data, and a Streamlit
Explorer as the demo. Phase 3 was explicitly out of scope.

**Then:** complete the whole project as the documents specify. Two decisions
were confirmed before starting, because they changed what would be built:

| Question | Decision | Consequence |
|---|---|---|
| Doc 08 §4.3 spatial transfer to Maharashtra | **Build the machinery, defer the region** | `transfer.py` and `config/region_maharashtra.yaml` exist and are tested; the second region's DEM is not built (1.5–2.5 GB of disk). It is item 2 on Doc 09 §8's own de-scope ladder |
| Doc 07 §2 cross-check against the `tick` library | **Substitute an in-repo independent check** | `tick` needs Python 3.12 in a throwaway environment and does not install here. Three structurally different implementations of the same likelihood must agree to machine precision instead. Gate 2 is recorded **BLOCKED**, not PASS, because that is weaker evidence than an external library |

### One deviation from the documents, deliberately

Doc 04 §3.3 states the 500 m / 1 day deduplication rule inside the *landslide*
section; Doc 04 §6 gate 5 restates it for "same-type events" generally. Applied
to floods it is actively harmful: reaches are 500 m long by construction, so
every pair of adjacent reaches flooding on the same day merges into one event,
erasing the downstream propagation the project exists to measure. Measured: it
cut 3,921 flood events to 2,463.

Resolved with `catalog.dedup.hazard_types: ["L"]` and the reasoning in the
config beside it. Floods are declustered instead — three consecutive
sub-threshold days at the same reach, Doc 02 §3.1 step 3.

---

## 2. The data — everything that was actually taken

### 2.1 Downloaded and used (free, no account)

Every download landed in `data/raw/<SOURCE>/<folder>/` with a `PROVENANCE.md`
written **at download time**, recording URL, exact query, date, person, licence,
citation and SHA-256 per file (Doc 10 §8).

| # | Dataset | Source | What was taken |
|---|---|---|---|
| **D1** | Copernicus DEM GLO-30 | AWS Open Data | 2 one-degree COG tiles: `N11_E075`, `N11_E076` |
| **D2** | HydroBASINS level 8 (v1c) | `data.hydrosheds.org` | Continental archive, subset to `MAIN_BAS = 4080029900` |
| **M4** | CHIRPS v2.0 daily p25 | Climate Hazards Center, UCSB | Global yearly netCDF 2018–2024, clipped to the study bbox; **global files deleted after clipping** |

CHIRPS is **dataset M4, added to Doc 10 §3 by this build.** It was not in the
original plan. It is gauge-blended like IMD, free and login-free, so it unblocks
Phases 2–3 while the IMD / IMERG / ERA5 accounts are pending.

### 2.2 Real, but one record

The **GSI Bhusanket FIR** gives the Mundakkai crown — 30 Jul 2024, 02:17 IST,
11.4654 N 76.1348 E. **This is the only real event in the catalog.**

### 2.3 Attempted and failed

GSI Bhukosh (host unreachable), NASA COOLR (404), the ESSD 2018 PANGAEA DOI
(404). The inventory readers accept CSV, SHP and GPKG with column synonyms and
convert polygons to **crown points** (Doc 04 §3.2 — highest elevation, not
centroid), so a manual download drops straight in.

### 2.4 Synthetic stand-ins, and the five things that stop them passing for real

*(Updated 2026-09-21: IMD and IMERG are now selected with `rainfall.primary`
and read from `data/raw/IMD|IMERG/`; GloFAS `real` reads `data/raw/GLOFAS/` or
downloads with a CDS key; ERA5-Land is marked `unused`. The landslide inventory
counts as synthetic whenever the catalog holds SYNTHETIC landslide rows. See
`docs/15_HOW_TO_RUN.md`.)*

`data_mode` was `synthetic` for **GloFAS, ERA5-Land, IMERG and IMD**, and
`unavailable` for Sentinel-1 — which has no generator *on purpose*, because it
exists only to **score** the flood definition and a synthetic corroboration
number would look like evidence.

1. Every synthetic catalog row carries `source = SYNTHETIC`, `confidence = low`.
2. The affected quality gate reports **BLOCKED**, never PASS.
3. The dashboard shows a red sidebar banner naming the active stand-ins.
4. Figures carry a diagonal `SYNTHETIC` watermark.
5. Every generated report opens with a warning block.

---

## 3. The pipeline — 18 stages

Every stage is idempotent, cached, and reads its knobs from
`config/default.yaml` (176 lines). No threshold, date, CRS or path is hard-coded
anywhere in `src/`.

```
aoi  dem  flow  network  units  projection  traveltime      # Phase 2 framework
rainfall  catalog  covariates                               # Phase 1-2 inputs
rootcause                                                   # Phase 3
cascade                                                     # Phase 4
experiments                                                 # Phase 5
realtime                                                    # Phase 6
figures  report  check  app                                 # outputs
```

### Phase 2 — the spatial framework

| Stage | Output | Verified |
|---|---|---|
| `aoi` | **2,969.4 km²** from 9 HydroBASINS polygons, buffered to 5,930.7 km² | Published Chaliyar ≈ 2,923 km² — **within 2%**, from an entirely different starting point |
| `dem` | 2,622 × 3,458 at 30 m = **9,066,876 cells**, 6,589,652 valid | Mundakkai crown samples at **1,586.9 m** against the field report's ~1,550–1,600 m; the landmark sequence 1,587 → 835 → 47 m confirms correct downhill ordering |
| `flow` | D8 routing written from scratch; **4 cells still stalled** of 9.07 M | Outlet accumulation **2,907.5 km²** of a 2,969 km² basin |
| `network` | **9,038 reaches**, 4,357.5 km, density **1.47 km/km²**, Strahler 1–5 | Topology exact by construction — built from the receiver forest, not inferred from line geometry |
| `units` | **17,138 slope units**, median **0.194 km²** | Doc 05 §2.2 target ~0.25 km² |
| `projection` | π: H → N, **0 null injection reaches** | Doc 05 §9 gate 1 |
| `traveltime` | **716,519** (source, downstream) pairs, median celerity **1.94 m/s** | Built in **2.3 s** against Doc 05 §5.4's four-hour budget |
| `rainfall` | 7 × 9 grid × 2,406 days; R1…R30 and API | Only **14 pixels** touch the basin's 17,138 units |
| `covariates` | 16 static covariates per unit, 10 per reach | TWI 4.2–18.3, curvature ±0.007 m⁻¹ — physically sensible ranges |

The travel-time table stores the **path integral** `Σ length/S^0.3` rather than a
time, so fitting the celerity coefficient `v₀` is a division and the table never
needs rebuilding. That is the single optimisation that makes Phase 4 tractable.

---

## 4. Phase 1 — the event catalog

`data/catalog/events.csv` — **9,765 rows × 21 columns**, in the exact Doc 04 §2
order, coordinates at fixed 6 dp.

| Hazard | Source | Rows |
|---|---|---|
| L | `GSI_BHUSANKET_FIR` | **1** |
| L | `SYNTHETIC` | 759 |
| F | `SYNTHETIC` | 9,005 |

**The flood rule** (Doc 02 §3.1) is a procedure with no judgement calls: annual
maxima → **Gumbel by L-moments** → Q2/Q5/Q20 → first day above Q2 → episode
closes after 3 sub-threshold days → mark `log(Q_peak/Q2)`. 1,811 of 9,038
reaches clear the 50 km² eligibility bar; 9,005 events fall in the study period
at a rate of **0.72 per reach-year** against the **0.69** a Q2 threshold implies.

**Completeness:** A_min = **863 m²**, α = **2.02 ± 0.04**, 717 of 759 above the
rollover. The detector was validated — the synthetic generator's true minimum is
800 m² and the rollover finder independently recovered 863 m².

**The blind spot, verified rather than quoted.** Doc 02 §3.4 asserts the
Punnapuzha at Chooralmala is below the eligibility threshold. Measured on our own
network: the injection reach drains **2.3 km²**, the bridge reach **21.4 km²** —
2.3× below our research-mode bar and **12× below** the 250 km² bar operational
GloFAS reporting points use. The rule regains sight 14 reaches and 0.9 h
downstream.

**An independent corroboration fell out of the cascade-pair table.** The routed
channel distance from the Mundakkai injection reach to the Chooralmala bridge is
**6.21 km**, against the GSI field report's documented 6–8 km runout. Nothing was
tuned to produce it.

Full detail: `reports/phase1_flood_definition.md`.

---

## 5. Phase 3 — root cause and susceptibility

**Negative sampling** (Doc 06 §2) is the decision that shapes the answer, and all
four rules are enforced: a 500 m same-day spatial buffer, a ±3-day same-unit
window, elevation stratification, and all three ratios reported. At 1:5 the
landslide matrix is 760 positives and 3,800 negatives, with 2,017 rows excluded
by the spatial buffer and 4,552 by the temporal window.

**Collinearity** (Doc 06 §3): 16 features kept; 5 dropped for |r| > 0.85, 1 for
VIF > 10. Which of a correlated pair survives is a **physical** judgement encoded
once in `INTERPRETABILITY` — slope beats elevation because a hillslope fails on
its gradient, not its height.

**Driver ranking** — three methods, reported together:

| # | driver | point-biserial r | permutation ΔAUC-PR | mean \|SHAP\| |
|---|---|---|---|---|
| 1 | `R7` | +0.303 | +0.294 | 0.903 |
| 2 | `slope_max_deg` | +0.305 | +0.131 | 0.891 |
| 3 | `twi` | **−0.233** | +0.183 | 0.709 |
| 4 | `R3` | +0.251 | +0.059 | 0.155 |

`twi` is exactly why Doc 06 §4 asks for three methods. Its univariate
correlation is **negative** while both model-based methods rank it third — the
blind spot of correlation ("ignores interactions entirely") made visible.

**Stability:** the top 3 are the same set across all three sampling ratios, with
the order changing between 1:5 and 1:20. Doc 06 §2's required sentence, in its
honest form.

**Calibration** (Doc 06 §6, a hard requirement): isotonic, fitted on the
held-out spatial folds, and the reported "after" error is **cross-fitted**.
Landslide S-dynamic ECE **0.084 → 0.0096**.

> Isotonic regression scored on its own fit reports *exactly* zero error, every
> time, for any model however badly calibrated. An ECE of 0.0000 in a report is
> a red flag, not a triumph. The first version of this code printed one.

**`ẽ(u)` exported:** `R7`, `slope_max_deg`, `twi`, `R3` — four drivers, inside
Doc 06 §8 gate 5's limit of two to four.

**A finding worth stating:** flood S-static reaches AUC-ROC **0.498** — chance —
and the exported `S_F` is nearly constant at 0.137–0.172. Flood susceptibility is
not learnable from terrain alone in this data. On synthetic discharge that is
expected; it is reported rather than hidden.

Full detail: `reports/phase3_features.md`, `phase3_drivers.md`,
`phase3_calibration.md`, `phase3_baselines.md`.

---

## 6. Phase 4 — the cascade model

### 6.1 Verification first, because Doc 07 says so

> *Never add a component until the previous one has been verified on data where
> you already know the answer.*

| Check | Result |
|---|---|
| Three implementations agree | **PASS** — O(n²) double sum, O(n) Ozaki recursion and the vectorised torch version differ by **0.00e+00** |
| S1 temporal recovery, 20 catalogs | **PASS** — μ +1.5%, α −4.4%, θ +3.7% |
| S2 space-time recovery, 8 catalogs | **PASS** — μ +4.4%, α +1.2%, θ −2.4%, σ **+0.1%** |
| S2 with events against the boundary | **PASS** — α −12.8% (1.6 SE), the parameter the `erf` boundary correction protects |

The pass criterion compares each bias against its own **Monte-Carlo standard
error**, not a flat percentage. A flat threshold makes running *fewer* catalogs
more likely to pass — exactly backwards.

Figure: `figures/phase4_parameter_recovery.png`.

### 6.2 Stability is a bound, not a hope

Doc 07 §7 rule 7 makes a branching ratio ≥ 1 an automatic invalid fit. The first
S4 fit came back at **1.569** — explosive — and was rejected.

`α_LF` is not a count of offspring: it is the excitation applied to **each**
downstream reach, and a landslide here reaches a few hundred. The quantity that
must stay below 1 is `α_LF × (mean downstream set size)`, so the bound on `α_LF`
**scales with the network**. Bounding it at a fixed number would be far too
generous the moment the network switches on.

### 6.3 Stage results

Live numbers: `reports/phase4_stages.md`. The shape of the result:

- **S4 beats S3** on held-out log-likelihood. The network formulation — the
  reachability gate, the runout gate and dilution — earns its place.
- **S5 does not beat S4**, and the fitted `v₀` pins at its lower bound.

The cause is diagnosed in the report and is not the model. The synthetic
discharge generator builds each reach's hydrograph independently, so **there is
no wave travelling between reaches in it at all**. A travel-time shift fitted
against data with no travel time in it can only hurt, and the optimiser responds
by pushing `v₀` to its floor to move the shifted kernel out of the way. **This
comparison becomes informative only when `data_mode.glofas` is `real`**, and it
is the strongest single argument for getting the CDS account.

### 6.4 Gate 6 fails at S4, and that is the gate doing its job

Doc 07 §5 asks a specific question: *with the fitted `c` and `γ`, does a
landslide of Mundakkai's mapped area at its `d_slope` pass the runout gate? If
it does not, the gate parameterisation is wrong.*

| stage | fitted runout reach | Mundakkai crown `d_slope` | verdict |
|---|---|---|---|
| S4 | 203 m (`c` = 203, `γ` = 0.000) | 684 m | **EXCLUDED — gate 6 FAILS** |
| S5 | 2,173 m (`c` = 2,173, `γ` = 0.040) | 684 m | passes |

At S4 the fit drives `c` to a value that excludes the one event the project is
built around. With `γ` at zero the gate has also stopped depending on event size
at all, which is the other half of its purpose gone. Reported as a gate failure
rather than a footnote.

The check reads the **crown cell's** `d_slope` from the projection, not the slope
unit's representative point — a landslide starts at its crown, and the unit's
representative cell sits 1,101 m from the channel where the crown sits at 684 m.
Using the wrong one would make the gate look harder to pass than it is.

---

## 7. Phase 5 — experiments and validation

Everything is recorded in `experiments/register.csv`. Doc 08 §1: *no number
appears in the paper unless it has a register entry.*

| Deliverable | Report |
|---|---|
| Nested ladder M0–M5 with likelihood-ratio tests | `reports/phase5_ablations.md` |
| Ablations A1–A9 | `reports/phase5_ablations.md` |
| Baselines B1–B5 versus M5 | `reports/phase5_baselines.md` |
| Leave-one-cascade-out | `reports/phase5_loco.md` |
| Sensitivity, six variations | `reports/phase5_sensitivity.md` |
| The Wayanad backtest and negative controls | `reports/phase5_backtest.md` |

**The leakage guard raises, it does not warn**, and `tests/test_experiments.py`
is the independent audit Doc 08 §4.1 asks for — five tests that deliberately try
to slip post-cutoff data past it, including the case where the guard is handed an
empty frame and must refuse to congratulate itself.

**It caught a real mistake in this code.** The backtest was applying the
*parameter* cutoff to the *history*, forbidding June 2024 events from the
history while rolling through July. Those are different things: on 15 July an
operator would of course know about an event from June. The two cutoffs are now
separate and both are stated in the report.

### The Wayanad backtest, and what it shows

The roll produced **no alert crossing**, and the report says why rather than
leaving it at "never". Three causes, chained:

1. **The rainfall product cannot see the trigger.** CHIRPS gives 71 mm of the
   published 577 mm over the 48 h before the failure — 12%. The background rises
   with rainfall and there is no rainfall in the input to rise on.
2. **The landslide catalog is synthetic**, and its events are placed
   independently of rainfall at any particular unit, so the background learned no
   relationship between a wet Mundakkai and a Mundakkai failure.
3. The shape of the failure confirms it: Warning-level alarms *before* the roll
   window and none during it — not a threshold set too high, but a model driven
   by something other than the conditions that preceded the landslide.

**This is reported rather than fixed by lowering the threshold.** A lead time
manufactured by moving the alert line down would be worth nothing. It also makes
gate P2.4's failure concrete: the rainfall gap is not an abstract data-quality
note, it is the reason the headline deliverable has no lead time.

The predicted downstream **arrival window** — the timing claim that distinguishes
this figure from anything else in the literature — is still produced and shaded
on `figures/money_figure.png`.

---

## 8. Phase 6 — alerts and delivery

**The honest scope statement** (Doc 09 §1.1) governs everything here: a validated
model, a fully specified architecture, and a working replay dashboard —
**not** a running 24/7 service.

**The latency budget** decides the rest. GPM IMERG Early Run arrives at
**T + 4 h**, so rain-to-updated-risk is about **4 h 15 min**. This is a nowcast
layer, not a siren. Wayanad failed at 02:17 IST; rainfall from 22:00 the previous
night would have reached the system around 02:00.

**Four alert levels, scored against each unit's own 2018–2023 distribution**, so
a permanently steep hillside and a gentle coastal slope are judged on the same
footing. Thresholds are optimised on the validation period at cost ratios 1:1,
5:1 and 20:1 — and **we do not choose the ratio.** That is a policy decision for
a disaster management authority. We recommend 5:1 and say so.

**Persistence costs lead time, and the cost is measured.** A level is raised only
after 2 consecutive updates and lowered only after 4 — asymmetric on purpose, and
the lead time it costs is reported rather than absorbed.

Reports: `reports/phase6_architecture.md`, `reports/phase6_alerts.md`.

---

## 9. What can be seen

**`hydrocascade app`** — the Explorer, nine tabs, verified by an automated render
test rather than by "it imports": the cascade path, the event catalog, the
spatial framework, rainfall, drivers and susceptibility, the model, backtest and
alerts, gates and reports, how to reproduce.

**Fourteen figures.** The two that matter most:

| | |
|---|---|
| `money_figure.png` | The Wayanad backtest — rainfall, P(landslide) with its band, the alert levels, the moment of failure, and the shaded downstream arrival window |
| `phase2_wayanad_flowpath.png` | Crown → injection → Chooralmala → Nilambur on hillshade, with travel times and dilution |

Every figure uses a palette **validated for colour-vision deficiency** rather
than chosen by eye — the six checks are recorded in `viz/palette.py`, including
the two candidates that failed and why.

---

## 10. Quality gates

Executable, so "the gates pass" is a command anyone can run.
`hydrocascade check` covers all six phases and exits non-zero while any gate
fails.

**Three states, and the third is the point.** `BLOCKED` means the gate *could not
be evaluated* because an input is missing or synthetic. It is reported separately
from `FAIL` so that "we have not got the data yet" can never be read as "the
check passed". A two-state scheme forces that distinction to be lost, and losing
it is how a synthetic number reaches a slide.

Live results: `reports/quality_gates.md`.

---

## 11. Tests — 147 passing

| File | Tests | What it guards |
|---|---|---|
| `test_flow.py` | 17 | Routing on hand-computed grids: a plane, a pit, a flat, nodata escapes, mass conservation |
| `test_floods.py` | 15 | L-moments and declustering on hydrographs whose answers are in the docstrings |
| `test_catalog.py` | 19 | Schema, FR-04 provenance, deduplication including the transitive chain |
| `test_corroboration.py` | 15 | The Poisson exceedance rate; **that the SAR precision test raises rather than returning a plausible number** |
| `test_cascade.py` | 22 | Compensator truncation, self-excitation, spectral radius vs max entry, signed parameters, the kernels |
| `test_experiments.py` | 24 | The leakage guard under deliberate attack; ablation bookkeeping; alert hysteresis |
| `test_dashboard.py` | 5 | The whole dashboard renders, nine tabs, populated metrics, synthetic banner visible |

**NFR-08 determinism verified:** two consecutive `hydrocascade catalog` runs
produce a byte-identical `events.csv`.

---

## 12. What does not work

1. **Gate P2.4 FAILS — rainfall at the crown.** CHIRPS captures 71 mm of the
   published 577 mm (12%). Diagnosed, not tuned away: CHIRPS *does* show a
   240–290 mm band over 29–31 July, but it peaks ~50 km **west** of the crown —
   the documented orographic displacement of satellite-blended products in the
   Western Ghats, worsened by 0.25° cells averaging over ~27 km. Needs IMD or
   IMERG. **This is the failure that propagates furthest**: it is why the
   backtest has no lead time.
2. **Gate P1.1 BLOCKED — one real landslide.** 9,764 of 9,765 rows are synthetic.
   Doc 04 §6's hard stop applies if this is still true at the end of Week 3:
   widen the region before proceeding. That decision has not been made.
3. **Gate P1.6 BLOCKED — FR-05 unmet.** SAR precision was not measured; Otsu
   thresholding needs Google Earth Engine. The report prints the exact recipe and
   a `BLOCKED` status instead of a figure.
4. **S5 does not beat S4**, for the diagnosed reason in §6.3.
5. **`α_FL` sits at its floor.** Only 199 flood→landslide pairs survive the
   corridor restriction against 147,846 total, so the direction is not
   detectable in this catalog. That is a finding about the data, reported as one.
6. **The spatial transfer is not run** — machinery built, region deferred.
7. **Leave-one-cascade-out scores one event of six.** The other five have
   unsourced coordinates and live in `events_PENDING.csv`. A lead-time
   distribution over one event is not a distribution.
8. **NFR-07 is unverified.** Nobody who did not write this has reproduced it from
   a clean clone.

---

## 13. Problems found and fixed

The fixes are more informative than the successes.

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | 6,403 interior cells stalled on the DEM | Copernicus GLO-30 flattens water bodies to *exactly* constant elevation — 248,810 cells at 0.0000 m, 3,314 at 41.5000 m. They do drain, off the grid edge or into nodata, which the flat resolver did not recognise | Added `escapes_model = border ∪ nodata-adjacent`. **6,403 → 4**; outlet accumulation 2,836 → **2,907 km²** |
| 2 | Median celerity 0.43 m/s against a required 1–5 | **`v₀` is a coefficient, not a speed.** `v = v₀·S^0.3` and median `S^0.3 ≈ 0.215`, so "1–5 m/s" means `v₀ ≈ 9` | `v0_reference_ms: 9.0` plus a guard. → median **1.94 m/s** |
| 3 | Nilambur reported NOT DOWNSTREAM | Nearest-line snapping picked a Strahler-1 ditch 528 m away over the Chaliyar at 588 m | `Network.snap()` prefers the largest catchment within 750 m — a radius that is load-bearing at both landmarks |
| 4 | Flood dedup erased the propagation signal | The 500 m rule against 500 m reaches | `catalog.dedup.hazard_types: ["L"]` |
| 5 | The exceedance-rate guard fired on healthy Q5 data | It assumed ~1 event/reach-year for any return period. The correct value is `−ln(1 − 1/T)` — **0.69** for Q2, **0.22** for Q5 | `expected_exceedance_rate()` with a multiplicative tolerance band |
| 6 | `scikit-image` imported but absent from `requirements.txt` | It worked only because the venv uses `--system-site-packages`. A clean install would fail at `hydrocascade flow`, breaking NFR-07 for exactly the person NFR-07 exists to test | Added |
| 7 | Curvature and aspect NaN at the AOI boundary | Every derivative is a 3×3 stencil, so a valid cell beside nodata reads a nodata neighbour. The AOI is an irregular basin inside a rectangular grid, so this is the whole boundary | Nearest-valid fill before differencing, results masked back, plus an assertion that no covariate is NaN |
| 8 | **Calibration reported ECE 0.0000** | Isotonic regression fitted *and scored* on the same predictions is perfectly calibrated by construction, for any model however bad | K-fold cross-fitted "after" error. The deployed calibrator is still refitted on everything; we simply never quote its in-sample error |
| 9 | Catchment rainfall 23.8% missing, deleting all five accumulation windows | CHIRPS is a land product and 15 of 63 pixels are Arabian Sea. `0 × NaN = NaN`, so a plain dot product returned NaN for **every** reach | Weighted mean over pixels that have data, weights renormalised per reach. 23.8% → **0.12%** |
| 10 | Doc 06 §2 rule 2a silently skipped for floods | `xy=None` was passed for reaches | Reach endpoints supplied; 11,944 reach-days now excluded where 0 were before |
| 11 | S4 branching ratio **1.569** — explosive | `α_LF` unconstrained; it multiplies a few-hundred-reach downstream set | Data-dependent bounds so every entry of the mean-offspring matrix stays ≤ 0.45 |
| 12 | Regression coefficients could not go negative | The optimiser parameterised *every* unbounded parameter as `exp(z)`. A risk-reducing driver was inexpressible, and the fit answered with `exp(−40)` — which reads as "does not matter" and is a different statement | `REAL_VALUED` prefixes get identity parameterisation. Coefficients now come back negative where the data says so |
| 13 | **Flood background used rainfall pixel 0 — an ocean pixel** | The compensator used the catchment average and the log-intensity term used a single hardcoded pixel, so the two halves of the likelihood described different models and `base_F` absorbed the mismatch | Per-event catchment-weighted rainfall, matching the compensator |
| 14 | **P(landslide) saturated at 1.00 on every day** | The forward predictor applied `α` to every past event in the basin with no spatial kernel, while the likelihood restricted and normalised it. Five orders of magnitude apart | Predictor applies the same normalised Gaussian and corridor the likelihood does. λ now 3.3–4.7e-05 against an empirical 1.8e-05 |
| 15 | **No flood→landslide pairs existed at all** | `α_FL` appeared only in the compensator, never in the log-intensity term, so it could only ever cost likelihood and was driven to zero. Half the cross-hazard claim was silently untestable | Corridor-restricted F→L pairs added; all four kernel directions now present |
| 16 | The leakage guard rejected the backtest | The *parameter* cutoff was being applied to the *history* | The two cutoffs separated, both stated in the report. **The guard working as designed** |
| 17 | **Ablation A6 crashed after 40 minutes of fitting**: `KeyError: 'beta_dyn0'` | `initial_params` created the rainfall coefficients only when the *background* was on; the objective read them whenever *rainfall* was on. A6 is the one spec that separates those two flags, so the mismatch was invisible everywhere else | Rainfall coefficients gated on `rainfall`, where they belong |
| 18 | **`sigma_m` was reported as a fitted length scale in three fit records** (1.2 km, 2.1 km, 2.1 km) | It was drawn by `initial_params` and read by nothing — the landslide→landslide kernel width is the fixed `LL_SIGMA_M` constant the likelihood and the predictor share. The optimiser left it wherever the random draw put it | Parameter removed. A regression test now fails if any created parameter has zero gradient |
| 19 | M0–M2 carried a `theta1` (and two mixture weights) nothing read | The timescales belong to the triggering kernel, and those rungs have no triggering. The unused parameters inflated their AIC and put the M2→M3 likelihood-ratio test one degree of freedom out — penalising exactly the reduced models the ladder exists to compare | Timescales gated on excitation; the objective returns early for the Poisson rungs |
| 20 | **A1 was charged +4 AIC for two parameters it could not use** | The runout gate multiplies landslide→flood terms only, and A1 is the ablation that removes them. A1 is the row that tests *"landslides raise flood risk"*, so the penalty biased the headline comparison **towards the project's own hypothesis** | Gate parameters gated on the cross terms |

Items 8, 13, 14 and 15 share a shape worth naming: **two pieces of code computing
what should be the same quantity, differently.** None of them crashes; each
produces a confident, wrong number. They are the reason the verification stage
exists.

Items 17–20 share a different shape: **the set of parameters created and the set
the likelihood reads had drifted apart.** Only one of the four crashed. The other
three quietly published a number nobody had fitted (18) or charged a parameter
penalty to a model that never used the parameter (19, 20) — and 20 did it to the
single comparison the project most wants to be true. A parameterised ablation
framework makes these cheap to introduce, so `tests/test_cascade.py` now
evaluates **every** rung and **every** ablation on a toy dataset and asserts the
two sets match exactly, in both directions.

---

## 14. What unblocks what

| Blocker | Unblocked by | Turns into |
|---|---|---|
| One real landslide | A manual ESSD / NRSC / Bhukosh download into `data/raw/` with a `PROVENANCE.md` | Gate P1.1, and every driver ranking and fitted parameter becomes meaningful |
| Synthetic discharge | GloFAS files in `data/raw/GLOFAS/<date>/` (or a CDS/EWDS key) + `data_mode.glofas: real` | The S5-vs-S4 comparison, ablation A3, and every flood count |
| Rainfall at the crown | IMD NetCDF files (or IMERG via NASA Earthdata) + `rainfall.primary` | Gate P2.4 — **and the backtest's lead time** |
| No SAR precision | A Google Earth Engine account | FR-05 and gate P1.6 |
| Circular recall denominator | An afternoon in the KSDMA, EMS and CWC archives | A publishable recall number |
| Transfer region not built | 1.5–2.5 GB of disk and sourced Maharashtra coordinates | Doc 08 §4.3 |
| NFR-07 unverified | One team member, one clean machine | NFR-07 |
| `git_commit: not-a-git-repo` in every fit record | `git init && git commit` | Doc 07 §7 rule 8 traceability |

**None of the first six needs a code change.** That was the design intent of
`ingest/adapters.py`: the pipeline never blocks on an approval email, and it
never quietly reports the stand-in as though it were the real thing.

---

## 15. File inventory

```
README.md                 clone -> figures, NFR-07
LIMITATIONS.md            read before quoting any number
requirements.txt          pinned; numpy < 2.0 for the OpenCV ABI
config/default.yaml       176 lines; every knob in the project
config/region_maharashtra.yaml   the Doc 08 §4.3 transfer region, ready to run

data/
  raw/{COP_DEM_GLO30,HYDROSHEDS,CHIRPS}/   each with PROVENANCE.md
  processed/spatial/      18 artefacts, ~161 MB
  catalog/                events.csv (9,765 x 21), events_PENDING.csv,
                          cascade_pairs.csv, events_CHANGELOG.md
  reference/              documented_floods.csv - scored against, never fitted to

src/hydrocascade/         15,543 lines, 73 modules
  ingest/     spatial/    features/   catalog/
  rootcause/  cascade/    experiments/ realtime/
  qa/         viz/        dashboard/
tests/                    1,545 lines, 117 tests
notebooks/01_catalog_exploration.ipynb    21 cells, every one executed
figures/                  14 PNGs
reports/                  14 generated reports
models/fits/              one JSON per fit: parameters, intervals, logL, AIC,
                          seed, git commit, wall clock
experiments/register.csv  every experiment ever run
docs/00..12               the specification suite
docs/13                   this document
```
