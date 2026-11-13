# Document 06 — Phase 3: Root Cause Analysis and Baseline Models

**Weeks 5–7 · Owner: Yugant Chincholikar · Reviewer: Chetan Mahajan**
**Ends at Review 1**

**Goal:** produce the ranked driver profile, the calibrated susceptibility surfaces `S_L(x)` and `S_F(s)`, the top drivers `ẽ(x)`, and every external baseline the cascade model must beat.

**Why baselines are built here and not later:** if you build baselines only after the novel model, you will unconsciously tune the baseline to lose. Build them first, honestly, while you have no result to protect.

---

## 1. The target variable, stated precisely

Part 1 predicts, for each spatial unit:

> **the probability that at least one event of type `i` occurs in this unit during a given day, given the physical conditions**

This is a **binary classification with time-varying covariates**, at unit × day resolution. It is not a susceptibility map in the classical time-independent sense, though the static component of it is exactly that.

Two model variants are produced, and both are needed:

| Variant | Covariates | Output | Used as |
|---|---|---|---|
| **S-static** | terrain, soil, land cover only | `S_i(u)` — time-independent susceptibility | the `S_i(u)` multiplier in `μ_i` |
| **S-dynamic** | static + rainfall windows + soil moisture | daily probability | **baseline B3**, the ML model the cascade must beat |

---

## 2. Negative sampling — the decision that shapes the answer

There are no "non-events" in the catalog, so they must be constructed. This choice materially changes the result, so it is documented and tested.

### Rules

1. **Positives:** unit × day pairs where an event occurred.
2. **Negatives:** unit × day pairs where no event occurred, drawn at random, **excluding**:
   - any unit within 500 m of a recorded event on that day (buffer against inventory incompleteness)
   - any unit × day within ±3 days of an event in the same unit
3. **Ratio:** primary analysis at **1:5** positives to negatives. Sensitivity check at 1:1 and 1:20.
4. **Stratification:** negatives are sampled to match the elevation distribution of positives, so the model does not simply learn "hills versus coastal plain."
5. **Class weighting** in the model, and **AUC-PR** as the headline metric, because ROC is optimistic under heavy imbalance.

### Required reporting

> "The top three drivers were unchanged across all three sampling ratios. The fourth and fifth exchanged places at 1:20."

That sentence, or its honest equivalent, must appear in the report. A driver ranking that flips with the sampling ratio is not a finding.

---

## 3. Feature engineering and collinearity

**Candidate features:** everything from Document 05, §6 and §7.

**Removal procedure, in order:**

1. Drop features with more than 20% missing values in the study area.
2. Compute the correlation matrix. Where |r| > 0.85, keep the more physically interpretable of the pair and record the decision.
3. Compute the Variance Inflation Factor. Drop features with VIF > 10, iteratively.
4. Record the final feature list and every removal in `reports/phase3_features.md`.

Slope, elevation and topographic wetness index are typically strongly correlated. Deciding which to keep is a judgement call — make it, write down the reason, and be ready to defend it.

---

## 4. Driver ranking

Three methods, reported together. Agreement between them is the evidence.

| Method | What it gives | Caveat |
|---|---|---|
| **Correlation / point-biserial** | Simple univariate association | Ignores interactions entirely |
| **Permutation importance** | Drop in performance when a feature is shuffled | Must be computed on the **held-out** set, not training |
| **SHAP** | Per-prediction contributions, with direction and magnitude | The primary method. Report the summary plot and dependence plots for the top 5 |

**Deliverable:** a ranked table with all three columns side by side, plus a short physical explanation for each of the top five. Not statistics alone — say *why* that factor matters physically.

**Feeding Part 2:** the top **two to four** standardised drivers become `ẽ(u)`, the argument of the receptivity term `g_i`. Not more. Every additional driver here costs identifiability in the cascade model that we cannot afford.

---

## 5. Models and validation

### 5.1 Models

- Random Forest
- XGBoost
- Logistic regression as a transparent floor

Modest hyperparameter tuning only. This is a baseline, not the contribution.

### 5.2 Spatial block cross-validation — mandatory

> Hold out **whole sub-basins**, not random pixels.

Neighbouring pixels are nearly identical. Random splitting lets the model see almost the same location in both training and test, which inflates accuracy substantially. Use 5-fold grouped CV with sub-basin as the group.

Report **both** random-split and spatial-block numbers, side by side, and state the gap. That gap is itself a finding worth a sentence in the paper.

### 5.3 Temporal integrity

For S-dynamic, the split must also respect time: train on 2018–2022, validate on 2023–May 2024. Never train on data after a test period.

### 5.4 Metrics

| Metric | Why |
|---|---|
| AUC-ROC | comparability with the literature |
| **AUC-PR** | headline metric, honest under imbalance |
| Brier score | probability quality, not just ranking |
| Reliability diagram | **required**, because `S_i` is used as a multiplier downstream |

### 5.5 Targets

| Metric | Minimum | Target |
|---|---|---|
| AUC-ROC, landslide | 0.80 | ≥ 0.87 |
| AUC-PR, landslide | 0.50 | ≥ 0.65 |
| AUC-ROC, flood | 0.80 | ≥ 0.87 |
| Brier | ≤ 0.15 | ≤ 0.10 |

**Note on the published benchmark.** A 2026 study reports flood susceptibility AUC-ROC around 0.905 for Kerala. Our number will likely be **lower**, because we use spatial block CV. Report ours honestly and explain the difference rather than switching to random splits to get a bigger number. Explaining a lower-but-more-honest number is a strength signal; a suspiciously high number is not.

---

## 6. Calibration — a hard requirement

`S_i(u)` is used as a **multiplier** in `μ_i(u,t)`. A miscalibrated score corrupts the entire cascade model.

1. Plot the reliability diagram on held-out data.
2. If predictions deviate from the diagonal, apply **isotonic regression** (or Platt scaling if data is scarce), fitted on the validation set only.
3. Re-check the reliability diagram after calibration.
4. Export `S_L(u)` and `S_F(s)` as calibrated probabilities in (0,1).

Document the calibration in `reports/phase3_calibration.md`. This is the interface contract between Part 1 and Part 2.

---

## 7. External baselines — built now

These are the models the cascade engine must beat. They are separate from the nested ladder M0–M5 in Document 08.

| ID | Baseline | Definition | What it establishes |
|---|---|---|---|
| **B1** | Climatology | Mean event rate per unit per calendar month, from training data | The floor. Everything must beat this |
| **B2** | Rainfall intensity–duration threshold | Classical I–D threshold fitted to the training catalog; alarm when the (intensity, duration) point crosses the line | **What is actually used operationally today.** Beating this is the practical claim |
| **B3** | ML susceptibility × rainfall | S-dynamic XGBoost, predicting daily probability per unit | The strongest non-point-process competitor |
| **B4** | Independent single-type Hawkes | Two separate ETAS-like models, one for landslides, one for floods, no cross terms | Isolates the value of cross-hazard coupling |
| **B5** | Multitype Hawkes, isotropic Euclidean kernel | Cross terms present, but no network, no travel time, no dilution | **Isolates the value of the network formulation** — the single most important comparison in the paper |

B4 and B5 require the Phase 4 code, so they are *specified* here and *run* in Phase 5. B1, B2 and B3 are built and evaluated in this phase.

### Building B2 properly

Do not build a strawman. Fit the I–D threshold the way the landslide literature does: extract (rainfall intensity, duration) pairs for triggering events, fit a lower-envelope power law, and tune the threshold on the validation set to a stated false-alarm rate. Then report its lead time and false-alarm rate on the same test period as everything else.

A well-built B2 that our model beats is worth far more than a badly built one that it beats easily.

---

## 8. Quality gates

| # | Gate |
|---|---|
| 1 | Driver ranking stable across three negative-sampling ratios (top 3 unchanged) |
| 2 | Spatial block CV used, and both random and spatial numbers reported |
| 3 | Reliability diagram produced; calibration applied if needed; re-checked |
| 4 | `S_L(u)` and `S_F(s)` exported as calibrated rasters/tables |
| 5 | `ẽ(u)` chosen — **between 2 and 4 features**, no more |
| 6 | B1, B2, B3 built, evaluated, and their metrics recorded in a table |

---

## 9. Four-person split

| Member | Task |
|---|---|
| **Yugant** | Feature engineering, collinearity removal, driver ranking, SHAP, RF and XGBoost training |
| **Chetan** | Spatial block CV splits; prediction over the full grid; export and calibration of `S_i` |
| **Raj** | Negative sampling design and the three sensitivity runs; specification of B4 and B5 |
| **Ronit** | Baselines B1, B2, B3; the metrics table; the Review 1 deck |

---

## 10. Review 1 — what to present

1. **The problem**, one sentence.
2. **The two corrections we found ourselves**: cascade direction, and the deferred contamination stage. Lead with these. Catching your own errors is a strength signal.
3. **The catalog**: events per year, the map, and the *measured precision* of the flood definition.
4. **The spatial framework**: the Wayanad flow-path figure from Phase 2. One picture, whole idea.
5. **Part 1 results**: the ranked driver table, the SHAP plot, one physical sentence per top driver.
6. **The baseline**: AUC-PR under spatial CV, against the published benchmark, with the gap explained.
7. **The bridge**: one slide showing `μ_i` and `g_i` with arrows from the Part 1 outputs into them. This is the slide that proves the project is one project.
8. **Next, and the risk**: "Phase 4 is the cross-hazard fit. The main risk is that cross-excitation is not statistically detectable with this many cascade pairs. If so, we will report that with a power analysis rather than force a result."

That last sentence is the most impressive thing you can say to a research supervisor.

---

## 11. Deliverables

- `models/susceptibility_landslide.pkl`, `susceptibility_flood.pkl` — calibrated
- `data/processed/S_landslide.tif`, `S_flood.parquet`
- `data/processed/top_drivers.json` — the `ẽ(u)` definition
- `reports/phase3_drivers.md` — ranked table with physical explanations
- `reports/phase3_calibration.md`
- `reports/phase3_baselines.md` — B1, B2, B3 metrics
- `figures/phase3_shap_summary.png`, `phase3_reliability.png`, `phase3_susceptibility_map.png`
