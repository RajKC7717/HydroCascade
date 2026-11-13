# Document 09 — Phase 6: Real-Time System, Alerts and Delivery

**Weeks 12–14 · Owner: Ronit Dahiwal (delivery), Chetan Mahajan (system) · Ends at Review 2**

**Goal:** a defensible real-time architecture, a justified alert framework, a working replay-mode dashboard, and the written outputs.

---

## 1. The real-time architecture

### 1.1 The honest scope statement

> **What we deliver:** a validated model, a fully specified deployment architecture, and a working replay-mode dashboard that reproduces the Wayanad event as if live.
>
> **What we do not deliver:** a running, monitored, 24/7 production service.

Say this plainly in the report. A semester project that claims a live operational system and cannot show one loses more credibility than it gains.

### 1.2 The latency budget — the constraint that shapes everything

| Stage | Latency | Source |
|---|---|---|
| Rain falls | T + 0 | — |
| **GPM IMERG Early Run available** | **T + 4 h** | NASA GPM: Early Run 4 h, Late Run 14 h, Final Run 3.5 months |
| Ingest, regrid, update accumulations | + 10 min | our pipeline |
| Recompute intensity across all units | + 2 min | §1.5 |
| Alert evaluation and dashboard refresh | + 1 min | |
| **Total: rain to updated risk** | **≈ 4 h 15 min** | |

**Therefore: this is a nowcast layer with roughly four hours of latency. It is not a siren.** Wayanad failed at 02:17 IST; rainfall from 22:00 the previous night would have been reflected in the system by about 02:00. The system's value is in the *hours to days* of rising signal beforehand, not in the final minutes.

State this explicitly. It is what an operational agency would ask first.

### 1.3 The ERA5-Land problem, and the fix

ERA5-Land soil moisture is the physically correct wetness variable, but it has multi-day latency and **cannot be used in real time**.

**Fix:** build an **antecedent precipitation index (API)** from IMERG rainfall alone — a recursively decayed rainfall accumulation. Calibrate it offline against ERA5-Land soil moisture across the training period, reporting the correlation. Use ERA5-Land in fitting; use the calibrated API at run time.

Document the calibration and the correlation achieved. This is exactly the kind of practical detail that separates a designed system from a sketched one.

### 1.4 The processing loop

```
┌── SCHEDULED: hourly ─────────────────────────────────────────┐
│ 1. Fetch IMERG Early half-hourly for the last hour           │
│ 2. Regrid to spatial units; append to the rainfall store     │
│ 3. Update R1, R3, R7, R15, R30 and the API per unit          │
│ 4. Recompute background μ_i for all units                    │
│ 5. Decay all active excitation terms; drop expired ones      │
│ 6. Recompute λ_i, then P_i, then uncertainty bands           │
│ 7. Evaluate alert levels; compute attribution                │
│ 8. Write state snapshot; refresh dashboard                   │
└──────────────────────────────────────────────────────────────┘

┌── SCHEDULED: daily ──────────────────────────────────────────┐
│ 1. Fetch GloFAS discharge                                    │
│ 2. Run flood event detection (Q2 exceedance + declustering)  │
│ 3. Append confirmed flood events to the live history         │
│ 4. Refresh long-window rainfall from IMD where available     │
└──────────────────────────────────────────────────────────────┘

┌── EVENT-DRIVEN: on new hazard event ─────────────────────────┐
│ Trigger: flood detected, or a landslide reported/confirmed   │
│ 1. Validate and append to history with its mark              │
│ 2. Look up the affected set:                                 │
│      landslide → Down(π(x)) within max travel time           │
│      flood     → downstream reaches + the corridor of slopes │
│ 3. Recompute λ ONLY for affected units  ← incremental        │
│ 4. Re-evaluate alerts; push notification if a level rises    │
│ Target: under 2 seconds                                      │
└──────────────────────────────────────────────────────────────┘
```

### 1.5 Why the hourly recompute is fast enough

Temporal truncation means only a few tens to a few hundred events are active at any time. Spatial and network truncation limit each event to a few hundred affected units. Static factors are cached. The whole recompute is a handful of vectorised tensor operations. See Document 02, §10.

### 1.6 Failure handling

| Failure | Response |
|---|---|
| IMERG feed unavailable | Persist last known accumulations, flag the dashboard as STALE with the age, do not silently extrapolate |
| GloFAS unavailable | Skip flood detection for the day, log it |
| A unit produces `nan` | Mark as UNKNOWN, never as Normal. Log and alert the operator |
| Model state corrupted | Rebuild from the event history, which is the single source of truth |

**Rule: the system never shows "Normal" when it means "I don't know."**

---

## 2. Alert thresholds

"Raise an alert when risk exceeds a threshold" is not a specification. This is.

### 2.1 Four levels

| Level | Meaning | Starting definition |
|---|---|---|
| **Normal** | Background conditions | below the unit's 95th percentile |
| **Watch** | Conditions developing | 95th – 99th percentile |
| **Warning** | Elevated, action advisable | 99th – 99.9th percentile |
| **Emergency** | Severe | above 99.9th percentile |

### 2.2 Why local percentiles, not absolute intensity

A steep, thin-soiled hillside above Mundakkai and a gentle slope near the coast have background intensities differing by orders of magnitude. An absolute threshold would alarm permanently in one place and never in the other.

Each unit is therefore scored against **its own 2018–2023 distribution**. This makes units comparable and makes the alert interpretable: *"conditions here are more extreme than 99.9% of days on record for this location."*

### 2.3 How the thresholds are chosen

**Not by us picking numbers.** By optimisation on the **validation period only** — never the test period.

1. Define the cost function: `Cost = C_miss × (missed events) + C_FA × (false alarms)`.
2. Sweep thresholds over a grid.
3. Minimise cost at several cost ratios: **1:1, 5:1, 20:1** (missing an event is 1, 5 or 20 times worse than a false alarm).
4. Report the chosen threshold at each ratio, with the resulting lead time and false-alarm rate.

**We do not choose the cost ratio.** That is a policy decision belonging to a disaster management authority, not to a student project. We supply the trade-off curve and a recommended default at 5:1, and we say so.

### 2.4 The required figure

**Lead-time versus false-alarm-rate curve**, generated by sweeping the Warning threshold. Each point is a threshold; the curve shows what you buy and what you pay.

This is the most operationally meaningful figure in the entire project, more so than any accuracy number.

### 2.5 Persistence and hysteresis

To prevent alert flapping:
- a level must be sustained for **2 consecutive updates** before it is raised
- once raised, a level is only lowered after **4 consecutive updates** below the threshold
- report how much these rules change the lead time (they will shorten it slightly — that is the honest cost of stability)

---

## 3. The dashboard

**One page. Streamlit. No login, no database, no user accounts.**

### Layout

1. **Map** of the study region, units coloured by alert level, with the river network drawn and the currently active event history marked.
2. **Ranked table** — the highest-risk units, with columns: unit, hazard type, probability with interval, alert level, and **top driver**.
3. **Detail panel** on click: the full attribution breakdown from Document 02, §9, plus a 72-hour intensity trace with uncertainty band.
4. **Time slider** for replay mode — drag through July 2024 and watch the Wayanad cascade develop.
5. **Status bar**: data freshness, last update time, and a STALE flag if a feed is late.

### Modes

| Mode | Status |
|---|---|
| **Replay** | **Core deliverable.** Steps through historical periods using the fitted model |
| **Live** | Optional stretch, only if Phases 1–5 are complete and clean |

The replay mode showing the Wayanad cascade unfold, with the alert level rising hours before 02:17 and the downstream arrival window appearing after the landslide is inserted, is the demonstration. It is more convincing than a live feed showing nothing happening.

---

## 4. Written outputs

| Output | Length | Owner |
|---|---|---|
| Final project report | as per VIT format | Ronit, all contribute |
| Paper draft | 8–10 pages | Ronit |
| Mathematics appendix | 3–4 pages | Raj |
| `README.md` — clone to money figure | 1 page | Yugant |
| `LIMITATIONS.md` | 1 page | Ronit |
| Two-minute dashboard screen recording | — | Chetan |
| One-page project summary PDF | 1 page | Ronit |

### Paper structure

1. Introduction — the chain problem, Wayanad, the gap named by others
2. Related work — the eight themes, condensed, with the closest prior art cited first
3. Study area and data — including the measured flood-definition precision
4. The spatial framework — hillslope, network, projection operator
5. Part 1: root cause analysis and susceptibility
6. Model formulation — the intensity, the mechanisms, the wave kernel
7. Fitting and identifiability
8. Results: baselines, ablations, parameter estimates with intervals
9. The Wayanad backtest and the lead-time/false-alarm trade-off
10. Limitations
11. Conclusion and future work

---

## 5. Review 2 — order of presentation

1. **The money figure, first slide.** Not motivation, not architecture. The result.
2. The baseline and ablation tables.
3. The parameter estimates: `α_FL′` with its interval, the L→F half-life in days, `ψ`, the mechanism weights, and the fitted channel celerity with a note that it is physically plausible.
4. **Lead time and false-alarm rate, together.** Then the trade-off curve.
5. Dashboard replay of Wayanad, thirty seconds, live.
6. Limitations, stated by you before anyone asks.
7. "Here is the repository. Here is the preprint. Here is what six more months would add."

---

## 6. After the semester

### December deliverables checklist

- [ ] Public GitHub repository, reproducible from clone
- [ ] arXiv preprint, 8–10 pages
- [ ] The money figure as a standalone PNG with a self-contained caption
- [ ] One-page summary PDF
- [ ] Two-minute demo recording
- [ ] `LIMITATIONS.md`, written by you, unflinching

### Where to submit

- **Climate Change AI workshops** at NeurIPS or ICLR — realistic for strong undergraduate work, and reviewers there value honest negative results
- **Natural Hazards and Earth System Sciences (NHESS)** — Copernicus, open peer review. Genuinely good for students: every review is public and becomes a citable record
- **Landslides**, **Natural Hazards**, **npj Natural Hazards** — a higher bar; worth aiming at if the backtest is clean

### Collaboration targets, in priority order

1. **Amrita Vishwa Vidyapeetham, Amritapuri — Prof. M. V. Ramesh's group.** They co-authored the January 2026 *Scientific Reports* paper using Wayanad and have long-running landslide early-warning work in Kerala. **The single most relevant group in India for this project.** Write in Week 4, once the catalog exists and you can say "we have assembled N events and are fitting a cross-hazard point process." Cite their paper, quote the limitation they name, describe what you are building in three sentences. Attach nothing on the first email.
2. **Geological Survey of India** — Bhukosh and Bhusanket teams, for inventory data and field reports.
3. **Kerala State Disaster Management Authority** and **NRSC**.
4. **Your own civil engineering department** — a fifteen-minute conversation about whether the terrain and celerity terms are physically sensible. Costs one email, materially improves credibility.

### For MEXT

The mathematics you are extending is Japanese in origin: the spatio-temporal ETAS model comes from Ogata at the Institute of Statistical Mathematics in Tokyo and is used operationally for aftershock forecasting in Japan. Your research plan can state exactly which equation you extend and how. Relevant places to name: **Institute of Statistical Mathematics (Tokyo)** for the statistical half, **Disaster Prevention Research Institute, Kyoto University** for sediment and debris-flow hazards. Read one or two papers from a specific lab and reference them.

### For Mitacs Globalink

Be the obviously right applicant for a posted geohazard, hydrology or spatial-statistics project. What wins: a public repository a professor can open in two minutes, one figure showing a real result, and a one-page PDF. Check the current application cycle directly.

---

## 7. Four-person split

| Member | Task |
|---|---|
| **Chetan** | Dashboard; the replay engine; the incremental update path; the demo recording |
| **Ronit** | Alert framework, threshold optimisation, the trade-off curve; the paper; both reviews |
| **Raj** | Mathematics appendix: full derivation, likelihood, the quadrature-grid argument with citations, identifiability discussion |
| **Yugant** | Reproducibility: README from clone to figure, pinned versions, fixed seeds, `LIMITATIONS.md` support, the API calibration against ERA5-Land |

---

## 8. De-scope ladder

If you are behind, cut from the top. Never from the bottom.

1. Live mode → replay only
2. Spatial transfer to Maharashtra
3. Land-cover change analysis
4. Mechanism mixture → single exponential
5. Dashboard → three static figures
6. Network kernel → isotropic, with the network formulation presented as specified-but-unfitted

**Never cut:** the event catalog, the flood-definition corroboration, the driver ranking, the baselines, the ablation table, the Wayanad backtest, the false-alarm analysis, or the limitations section. Those eight are the project.
