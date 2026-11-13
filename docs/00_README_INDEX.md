# HydroCascade — Document Suite v2

**Group CS-C1-04 · Computer Engineering · VIT Pune · Semester V, 2026–27**
**Project:** HydroCascade — A Root Cause Analysis and Cross-Hazard Point Process Framework for Predicting Rainfall-Triggered Disaster Chains
**Guide:** Dr. Rakhi Joshi Bharadwaj

---

## Why this suite exists

The project idea was reviewed and found to be sound at its core, but with several components that existed as **good ideas without exact definitions**. A reviewer could ask thirteen specific questions that we could not yet answer precisely.

This suite answers all thirteen. Nothing in the core idea has been abandoned. Three things have been made *sharper*, and in doing so the project became more novel, not less:

1. **The landslide → flood mechanism** is now three named physical mechanisms with three different timescales, and the model estimates which one dominates. This was our weakest point. It is now our strongest scientific contribution.
2. **The spatial framework** is now a hybrid support: landslides live on a 2D hillslope surface, floods live on a 1D river network, and a physically defined projection operator connects them. This is grounded in an established branch of spatial statistics.
3. **Excitation propagates as a wave, not as a field.** A downstream reach feels the effect of an upstream landslide *after the travel time of the flow*, not instantly. No existing hazard point process does this.

---

## Reading order

| # | File | What it is | Read when |
|---|---|---|---|
| 00 | `00_README_INDEX.md` | This file | First |
| 01 | `01_PROJECT_DEFINITION.md` | What the project is, in plain English, with the full system architecture explained | First, all four members |
| 02 | `02_MATHEMATICAL_AND_DATA_SPECIFICATION.md` | The exact definitions. Risk, flood events, spatial framework, mechanisms, kernels, uncertainty, attribution | Before Phase 3. Model Lead owns it |
| 03 | `03_LITERATURE_REVIEW_PLAN.md` | Every paper, where to find it, who reads it, how to record it | Week 1 |
| 04 | `04_PHASE_1_DATA_COLLECTION.md` | Build the event catalog | Weeks 1–3 |
| 05 | `05_PHASE_2_SPATIAL_FRAMEWORK.md` | Build the spatial units, river network and covariate stack | Weeks 3–5 |
| 06 | `06_PHASE_3_ROOT_CAUSE_AND_BASELINES.md` | Driver ranking, susceptibility model, all external baselines | Weeks 5–7 |
| 07 | `07_PHASE_4_CASCADE_MODEL.md` | Implement and fit the cross-hazard model | Weeks 7–10 |
| 08 | `08_PHASE_5_EXPERIMENTS_AND_VALIDATION.md` | Baselines, ablations, validation splits, backtests, uncertainty | Weeks 10–12 |
| 09 | `09_PHASE_6_REALTIME_AND_DELIVERY.md` | Alert system, dashboard, paper, packaging | Weeks 12–14 |
| 10 | `10_SRS_DATA_REQUIREMENTS.md` | Every dataset, source, licence, fallback | Reference |
| 11 | `11_SRS_SOFTWARE_REQUIREMENTS.md` | Functional and non-functional requirements, tech stack | Reference |
| 12 | `12_DATA_FLOW_SPECIFICATION.md` | Data flow diagrams, levels 0–2, and the real-time loop | Reference |
| 13 | `13_BUILD_SUMMARY.md` | **Implementation record, all six phases.** What has actually been built, what data went in, what came out, and what does not work | Before any review |
| 14 | `14_BUILD_TIMELINE.md` | **How long it takes.** Measured duration of every stage, a clean-build total, the publishable-run estimate, and what is left | Before running anything end to end |
| 15 | `15_HOW_TO_RUN.md` | **The manual.** Every command in order, which datasets to download and exactly where they go, what gets trained, where results land, and what to re-run after each change | Before touching anything |
| 16 | `16_DEMO_FLOW.md` | **Showing it to someone.** A 3-minute and a 12-minute walkthrough of the nine-tab app, plus the questions a reviewer asks and the honest answers | Before a demo or review |
| 17 | `17_TABS_EXPLAINED_SIMPLY.md` | **The app in plain words.** What each of the nine tabs shows and why it matters, with no maths and no jargon | If you are new, start here |
| 18 | `18_JURY_QA.md` | **42 questions a jury will ask, with two-line answers.** Including the uncomfortable ones, and the four questions to be careful with | The night before the viva |

Documents 00–12 say what *should* be built. Document 13 says what *has* been
built, 15 is how to run it and 16 is how to show it. Where they disagree, 13 is
the state of the repository and 00–12 are the plan — the gap between them is the work remaining, and Doc 13 §12 lists it.

---

## The thirteen review questions and where each is answered

| # | Review question | Answered in |
|---|---|---|
| 1 | Define "risk" precisely | Doc 02, §1 |
| 2 | Define flood events precisely | Doc 02, §3 · Doc 04, §4 |
| 3 | What does "location" actually mean? | Doc 02, §2 |
| 4 | What physical mechanism does the cross-hazard kernel represent? | Doc 02, §4 |
| 5 | Give downstream propagation a measurable rule | Doc 02, §5 |
| 6 | Baselines | Doc 08, §2 |
| 7 | Ablation experiments | Doc 08, §3 |
| 8 | Validation beyond Wayanad | Doc 08, §4 |
| 9 | Uncertainty estimation | Doc 02, §8 · Doc 08, §5 |
| 10 | Explainability in Part 2 | Doc 02, §9 |
| 11 | Real-time architecture | Doc 09, §1 · Doc 12, §4 |
| 12 | Alert thresholds | Doc 09, §2 |
| 13 | Computational scalability | Doc 02, §10 · Doc 11, §4 |

---

## The four roles

| Member | Role | Owns |
|---|---|---|
| Chetan Mahajan | Data and Geospatial Lead | Docs 04, 05, 10, 12 |
| Yugant Chincholikar | Root Cause and ML Lead | Doc 06, and §6–7 of Doc 02 |
| Raj Choudhari | Mathematics and Model Lead | Docs 02, 07 |
| Ronit Dahiwal | Validation, Delivery and Documentation Lead | Docs 03, 08, 09, 11 |

Swap the names if the fit is wrong. Keep the roles. Every deliverable has one owner and one reviewer, never four owners and never zero reviewers.

---

## Version note

This suite supersedes the earlier Documents 1–3 wherever they conflict. Two substantive corrections carried forward from the earlier review:

- **The cascade direction is rainfall → landslide → flood**, not flood → landslide. Rainfall is a continuous driver, not a discrete event.
- **Water contamination as a third stage is deferred**, because no dated, located, regional dataset exists to validate it. A measurable sediment proxy is defined as an optional extension.
