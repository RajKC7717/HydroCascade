# Limitations

Doc 04 §4.3 asks for this file **in Week 3, not in December**. The reason is
tactical as much as ethical: a limitation you write down early is a design
constraint, and one you discover late is a crisis. Everything here is known now.

Status as of Phases 1–2. Phases 3–6 are not built.

---

## 1. The flood rule cannot see the flood the project is about

This is the most important limitation in the project and it is structural, not
fixable by better code.

The Q₂-exceedance rule is only applied to reaches above 50 km² of contributing
area, because reanalysis discharge is not trustworthy below that. Measured on
our own network:

| | reach | contributing area |
|---|---|---|
| Mundakkai injection point | `R-02503` | **2.3 km²** |
| Punnapuzha at the Chooralmala bridge | `R-01977` | **21.4 km²** |
| first reach the rule can see | `R-01788` | 73 km², 14 reaches and 0.9 h downstream |
| first reach *operational* GloFAS would see (250 km²) | `R-02674` | 258 km², 44 reaches and 2.7 h downstream |

So the channel that actually received the 30 July 2024 debris flow is 2.3× below
our research-mode threshold and 12× below the threshold operational GloFAS
reporting points use. Doc 02 §3.4 stated this as a claim; the numbers above
confirm it against the network we built.

**Consequence for the Wayanad backtest.** The landslide is evidenced by the GSI
field report. The downstream flood is evidenced by field reports, the destroyed
bridge and SAR imagery — *not* by our own detection rule. The model's predicted
flood intensity at the headwater reaches can therefore be validated only
qualitatively (did risk rise in the right place at the right time?), and
quantitatively only from `R-01788` downstream.

We say this in the paper. Hiding it would be the fastest way to lose a reviewer.

---

## 2. Four datasets are currently synthetic stand-ins

`config/default.yaml → data_mode` currently reads `synthetic` for **GloFAS,
ERA5-Land, IMERG and IMD**. Each has a real adapter written and a synthetic
generator behind the same function, so no pipeline stage blocks on an approval
email — but nothing derived from them is a result.

Enforcement, so this cannot be forgotten:

- every synthetic catalog row carries `source = SYNTHETIC` and `confidence = low`;
- `hydrocascade check` reports the volume gate as **BLOCKED**, never PASS, while
  any synthetic row is present;
- the Explorer shows a red banner listing the active stand-ins;
- `reports/phase1_flood_definition.md` opens with a warning block and marks its
  recall figure *not publishable*;
- figures generated from synthetic data carry a diagonal `SYNTHETIC` watermark.

**What it costs.** Every event count, exceedance rate and severity split in
§2–§3 and §5.2 of the flood-definition report describes a generator, not Kerala.
The spatial framework, the travel-time table and the blind-spot analysis do not
depend on the discharge source and are unaffected.

---

## 3. The landslide catalog is one real event and 9,764 stand-ins

Doc 04 §6 gate 1 wants ≥ 200 real landslides after deduplication, ideally
≥ 1,000. We have **one**: the Mundakkai crown from the GSI Bhusanket FIR.

No inventory downloaded automatically. Bhukosh was unreachable, the NASA COOLR
endpoint returned 404, and the PANGAEA DOI for the ESSD 2018 dataset returned
404. This is exactly the situation the adapter design anticipated, and the
readers in `ingest/landslides.py` accept CSV, SHP and GPKG with column synonyms
already wired, so a manual download drops straight in.

Doc 04 §6 also specifies the **hard stop**: if the volume gate still fails at
the end of Week 3, widen the region to the full Western Ghats across Kerala,
Karnataka and Tamil Nadu *before doing anything else*. That decision is not
made yet and must not be quietly skipped.

Five further documented events (Puthumala, Kavalappara, Pettimudi, Kokkayar and
the 2018 monsoon bulk) are held in `data/catalog/events_PENDING.csv` rather than
in the catalog, because their coordinates are not sourced and Doc 04 §2 admits
no exceptions to the source-URL rule. **No coordinate was invented to make a
row look complete.**

---

## 4. Rainfall at the crown is wrong, and it is not a bug in our extraction

Doc 05 §9 gate 4 asks the pipeline to reproduce the published Wayanad totals:
204 mm on 29 July and 373 mm on 30 July 2024, 577 mm over 48 h.

CHIRPS p25 gives **71 mm of that 577 mm — 12%.** The gate **FAILS**.

Diagnosed rather than tuned away: CHIRPS does contain a 240–290 mm band over
29–31 July, but it peaks roughly **50 km west of the crown** (241.7 mm at
lon 75.625 against 121.6 mm at the crown cell at lon 76.125, on lat 11.375).
That is the documented orographic displacement of satellite-blended products in
the Western Ghats, made worse by 0.25° cells averaging over ~27 km.

Reproducing the published totals needs **IMD** (dataset M1, gauge-based) or
**IMERG** (M2, 0.1°). Both require an account. Until then, no rainfall-triggered
result at hillslope scale should be quoted from CHIRPS.

A related resolution problem: at 0.25° the whole 2,969 km² basin is covered by
about **14 distinct rainfall pixels** for **17,138 slope units**. Doc 10 §3
already warns this is "too coarse for a single hillslope", and the rainfall
windows are therefore stored per pixel, not per unit — the model must not be
allowed to imply spatial precision the input does not have.

---

## 5. The corroboration experiment has no number yet

Doc 11 FR-05 requires the flood definition's precision against SAR corroboration
to be reported in `reports/phase1_flood_definition.md`. It is not, because it
was not measured: Otsu thresholding of Sentinel-1 VV backscatter runs in Google
Earth Engine, which needs an approved account.

The report prints the exact recipe and a **BLOCKED** status instead of a figure,
and gate **P1.6** in `hydrocascade check` records the same thing so the gap is
visible without reading the report. **FR-05 is unmet.** A plausible-looking
precision nobody computed would be worse than an honest gap.

Recall is in a similar state one step further along: the harness runs and the
reference list exists, but every row in it is still sourced to the project
brief rather than to a primary KSDMA / Copernicus EMS / CWC record, so
`corroboration.recall()` refuses to mark the result publishable.

---

## 6. Choices that are defensible but are still choices

**Study area.** The pilot is the Chaliyar basin (2,969 km²), not the four
districts Doc 04 §1 names as the minimum. It was chosen because it contains the
entire Mundakkai → Chooralmala → Punnapuzha → Chaliyar → Nilambur cascade, so
every Phase-2 gate is testable end to end. Scaling up is a `region.id` and
`seed_bbox` edit with no code change — but the wider region has not been run,
and its runtimes are unmeasured.

**Flood deduplication is not applied to floods.** Doc 04 §3.3 states the
500 m / 1 day merge rule inside the landslide section; Doc 04 §6 gate 5 restates
it for "same-type events" generally. Applied to floods it is actively harmful:
reaches are 500 m long by construction, so every pair of adjacent reaches
flooding on the same day merges into one event, erasing the downstream
propagation the project exists to measure (it cut 3,921 flood events to 2,463).
Floods are declustered instead — three consecutive sub-threshold days at the
same reach, Doc 02 §3.1 step 3. The choice is recorded in
`catalog.dedup.hazard_types` with the reasoning beside it.

**`v₀` is a coefficient, not a speed.** Celerity is `v = v₀·S^0.3`, and the
median reach slope gives `S^0.3 ≈ 0.215`. Doc 05 §5.1 asks for a reference `v₀`
"chosen so speeds fall in the physical range 1–5 m/s", which means `v₀ ≈ 9`, not
`v₀ ≈ 2`. Setting `v₀ = 2` directly gives a median flood wave of 0.43 m/s, which
is not physical. The build stage now asserts the resulting celerity distribution
(median 1.94 m/s, 5th–95th 0.57–5.53 m/s). `v₀` is fitted in Phase 4 regardless;
the cached table stores the path integral so it never needs rebuilding.

**Landmark snapping prefers the main stem.** Nearest-line snapping picks a
Strahler-1 roadside ditch 528 m from Nilambur over the Chaliyar itself at 588 m.
`Network.snap()` therefore takes the largest contributing area within 750 m. The
radius is load-bearing: widen it to 1.5 km and the rule flips the wrong way at
Chooralmala, preferring a 40 km² river 1,492 m away over the Punnapuzha running
under the bridge. 750 m is correct at both landmarks — but it *is* tuned to two
landmarks, and a third could break it.

---

## 7. Four cells still do not drain

The DEM has 4 interior cells (of 9.07 million) with no downhill neighbour after
sink filling and flat resolution — 0.00004%. They arise where Copernicus GLO-30
flattens water bodies to exactly constant elevation.

This was 6,403 cells until the flat resolver learned that a flat can also escape
by leaving the model entirely, off the grid edge or into nodata. Outlet
accumulation went from 2,836 km² to 2,907 km² of a 2,969 km² basin. The
remaining 4 are recorded, not silently discarded, per Doc 05 §4.

---

## 8. The travel-time wave cannot be detected in synthetic discharge — and that
is a property of the stand-in, not a verdict on the idea

This is the most important thing to understand before reading any Phase 4 or 5
number.

Stage S4 — which adds the network, the runout gate and dilution — **beats S3**
on held-out log-likelihood. The network formulation earns its place.

Stage S5, which adds the travel-time shift and the mechanism mixture, **does
not beat S4**, and the fitted `v₀` sits pinned at its lower bound. The nested
ladder shows the same shape: M0 → M4 improve monotonically, then M5 collapses.

The cause is diagnosable and it is not the model. The synthetic discharge
generator builds each reach's hydrograph **independently** — a per-reach annual
factor and its own linear reservoir — so **there is no wave travelling between
reaches in it at all**. A kernel that shifts excitation by a travel time, fitted
against data containing no travel time, can only hurt; and the optimiser
responds exactly as you would expect, pushing `v₀` to its floor to move the
shifted kernel as far out of the way as possible.

So ablation A3 — remove the travel-time shift — is expected to show the model
*improving*, and it is reported that way rather than quietly dropped. **This
comparison becomes informative only when `data_mode.glofas` is `real`.** It is
the single strongest argument for getting the CDS account.

---

## 9. The excitation parameters are bounded, and one sits on its bound

Doc 07 §7 rule 7 makes a branching ratio at or above 1 an automatic invalid fit.
The first S4 fit came back at **1.569** — explosive — because `α_LF` was
unconstrained.

`α_LF` is not a count of offspring: it is the excitation applied to *each*
downstream reach, and a landslide here reaches a few hundred of them. The
quantity that must stay below 1 is therefore `α_LF × (mean downstream set
size)`, and the bound on `α_LF` now scales with the network. Every fit since is
stationary.

But the branching ratio then sits **exactly at its bound** (0.450) in most fits,
which means the data wants more self-excitation than stationarity permits. On
synthetic data that is unsurprising — the generator clusters events without any
branching structure to justify it. It is reported rather than fixed by raising
the bound, because raising the bound to make a number look better is how a
model becomes explosive in a way nobody notices.

---

## 10. Phase 5 experiments that did not fully run

- **The spatial transfer to Maharashtra (Doc 08 §4.3) is not run.** The
  machinery exists — `experiments/transfer.py` applies a fitted model to any
  assembled region and refuses to refit — and `config/region_maharashtra.yaml`
  is written and ready. What is missing is the second region's DEM, network and
  slope units (roughly 1.5–2.5 GB of disk) and sourced coordinates for Malin,
  Taliye and Irshalwadi. Neither is a code change.
- **Two of the six sensitivity variations are pending.** The slope-unit
  resolution checks at 0.1 km² and 1 km² rebuild every downstream artefact, so
  they are reported as PENDING with the exact command rather than skipped.
- **Leave-one-cascade-out scores one event of six.** The other five have
  unsourced coordinates and live in `events_PENDING.csv`. A lead-time
  distribution over one event is not a distribution.
- **The `tick` cross-check (Doc 07 §10 gate 2) was substituted, not run.** It
  needs Python 3.12 in a throwaway environment and does not install here. The
  substitute is three structurally independent implementations of the same
  likelihood agreeing to machine precision — real evidence, but weaker than an
  external library, so the gate is recorded as **BLOCKED**, not PASS.

---

## 11. What the delivery layer is, and is not

Doc 09 §1.1, repeated because it governs how the dashboard should be read:

> **What we deliver:** a validated model, a fully specified deployment
> architecture, and a working replay-mode dashboard that reproduces the Wayanad
> event as if live.
>
> **What we do not deliver:** a running, monitored, 24/7 production service.

And the constraint that shapes it: GPM IMERG Early Run arrives at **T + 4
hours**, so rain-to-updated-risk is about 4 h 15 min. **This is a nowcast layer,
not a siren.** Wayanad failed at 02:17 IST; rainfall from 22:00 the previous
night would have reached the system around 02:00. The value is in the hours to
days of rising signal beforehand, not in the final minutes.

The alert thresholds are optimised on the validation period at three cost
ratios and the trade-off curve is published, but **we do not choose the ratio**.
That is a policy decision for a disaster management authority. We recommend 5:1
and say so.
