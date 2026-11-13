# Document 08 — Phase 5: Experiments, Ablations and Validation

**Weeks 10–12 · Owner: Ronit Dahiwal · Reviewer: Raj Choudhari**

**Goal:** prove — or honestly fail to prove — that each component of the model earns its place, and that the system would have warned before a real disaster.

This is the phase that turns a working model into a research result. It is also the phase most often skipped, which is why so many student projects end with one number and no argument.

---

## 1. The experimental register

Every experiment is recorded in `experiments/register.csv` with: id, description, model config, data split, seed, git commit, wall-clock time, and results file. No number appears in the paper unless it has a register entry.

---

## 2. External baselines

The models the cascade engine must beat. Specified in Document 06, §7; run here on the same splits and the same metrics.

| ID | Model | Established in | What beating it proves |
|---|---|---|---|
| **B1** | Climatology (monthly mean rate) | Phase 3 | The floor |
| **B2** | Rainfall intensity–duration threshold | Phase 3 | **Beating current operational practice** |
| **B3** | XGBoost susceptibility × rainfall | Phase 3 | Beating the strongest ML competitor |
| **B4** | Two independent single-type Hawkes | Phase 4 code | The value of cross-hazard coupling |
| **B5** | Multitype Hawkes, isotropic Euclidean kernel | Phase 4 code | **The value of the network formulation** |
| **M5** | Full model | Phase 4 | — |

B5 versus M5 is the single most important comparison in the paper. It isolates exactly the contribution we claim.

### Nested ladder, for likelihood-ratio testing

| Model | Includes |
|---|---|
| M0 | constant rate per type |
| M1 | + susceptibility background, no time |
| M2 | + rainfall windows — **the honest baseline** |
| M3 | + self-excitation only (`α_LF′ = α_FL′ = 0`) |
| M4 | + cross-excitation free |
| M5 | + network kernel, travel time, mechanism mixture |

M0–M5 are nested, so likelihood-ratio tests apply. B1–B3 are not nested, so they are compared on predictive metrics only.

---

## 3. Ablations

Each removes one component from the full model. Report ΔHeld-out log-likelihood, ΔAIC, Δlead time and Δfalse-alarm rate for each.

| ID | Removes | Tests the claim |
|---|---|---|
| **A1** | cross terms (`α_LF′ = α_FL′ = 0`) | landslides raise flood risk |
| **A2** | network kernel → isotropic Euclidean | influence follows the drainage network |
| **A3** | travel-time shift (`τ = 0`) | **influence arrives as a wave, not instantly** |
| **A4** | runout connectivity gate | debris must physically reach a channel |
| **A5** | dilution term (`ψ = 0`) | effect weakens as catchment grows |
| **A6** | Part-1 background (`μ` constant in space) | the root-cause study improves the cascade model |
| **A7** | rainfall windows | rainfall forcing matters |
| **A8** | mark term (`η = 0`) | event size matters |
| **A9** | mechanism mixture → single exponential | mechanisms have distinguishable timescales |

**Required output:** one table, nine rows, four numeric columns. This table is the scientific core of the paper. It tells a reviewer exactly which of our ideas are real.

Expect some ablations to show **no** degradation. Report those too. An idea that turns out not to matter is a finding, and saying so builds far more credibility than quietly dropping it.

---

## 4. Validation design

### 4.1 Three-way temporal split

| Split | Period | Used for |
|---|---|---|
| **Train** | 1 Jun 2018 – 31 Dec 2022 | fitting all parameters |
| **Validation** | 1 Jan 2023 – 31 May 2024 | model selection, alert thresholds, calibration |
| **Test** | 1 Jun 2024 – 31 Dec 2024 | **touched once, at the end.** Contains Wayanad |

**Leakage guard:** an assertion in the fitting code that raises an error if any training event is dated after the cutoff. Written by Ronit, independently audited by Raj. One person writes it, a different person tries to break it.

### 4.2 Leave-one-cascade-out (LOCO)

This is the answer to *"you designed the model around Wayanad and then tested it on Wayanad."*

For each of the six documented cascade events:

1. Refit the model excluding a ±30-day window around that event, across the whole region.
2. Roll forward through the event.
3. Record lead time, alert level reached, and false alarms in the preceding 30 days.

**Output: a distribution of lead times across six events**, not one number. Report median and range.

| Event | Date | Notes |
|---|---|---|
| Kerala monsoon | 16 Aug 2018 | Concurrent floods and thousands of landslides |
| Kavalappara, Malappuram | Aug 2019 | |
| Puthumala, Wayanad | Aug 2019 | 2–3 km from the 2024 crown |
| Pettimudi, Idukki | 6 Aug 2020 | |
| Kokkayar / Plappally / Kavali | Oct 2021 | Three different causal mechanisms |
| **Mundakkai–Chooralmala** | **30 Jul 2024** | Held out entirely, not just LOCO |

### 4.3 Spatial transfer test

Fit on Kerala. Apply, **without refitting**, to the Maharashtra Western Ghats: Malin (2014), Taliye (2021), Irshalwadi (2023).

Report the degradation. "Our model fitted on Kerala also behaves sensibly in Maharashtra" is a far stronger sentence than "we added three more rows to the catalog."

This also gives the project a Maharashtra angle, which matters for engagement with the Maharashtra State Disaster Management Authority.

### 4.4 Negative controls — required

Take heavy-rainfall days that produced **no** landslide and **no** flood, matched to event days on rainfall percentile. Run the model.

**If the model raises an alarm on every wet day, its lead time is worthless.** This experiment is what makes the lead-time claim meaningful. It is not optional.

---

## 5. The Wayanad backtest

The protocol, followed exactly.

1. **Refit** the winning model on data strictly before **1 June 2024**. The leakage assertion must pass.
2. **Roll forward** day by day from 1 July 2024, then hourly from 28 July, feeding only observed rainfall. No future information of any kind.
3. **Record** `λ_L` and `P_L` for the Mundakkai slope unit at every step, with uncertainty bands.
4. **Express as a local percentile** against that unit's own 2018–2023 distribution. A relative alarm level is interpretable; a raw intensity is not.
5. **Record** the time each alert threshold was crossed. The gap to **30 July 2024, 02:17 IST** is the lead time.
6. **Insert** the observed landslide as an event, with its mapped area as the mark.
7. **Record** `λ_F` and `P_F` for downstream Chaliyar reaches over the following 72 hours, **including the model's predicted arrival window** from the travel-time term.
8. **Overlay** observed downstream discharge and the documented timing of downstream impacts.
9. **Report** lead time and false-alarm rate together, always.

### The money figure

Two panels, shared time axis:

- **Top:** daily rainfall bars; `P_L` for the Mundakkai unit with its 5–95% band; the four alert thresholds as horizontal lines; a vertical marker at 02:17 IST 30 July; the lead time annotated.
- **Bottom:** `P_F` for downstream Chaliyar reaches with band; **the predicted arrival window shaded**; observed discharge overlaid; the observed flood peak marked.

The shaded arrival window in the bottom panel is what distinguishes this figure from anything else in the literature. It is a *timing* prediction, checkable against the hydrograph.

### The limitation to state in the caption

The Punnapuzha at Chooralmala is below GloFAS's reliable catchment size. The downstream flood is therefore validated quantitatively only at the larger Chaliyar reaches, and qualitatively at the headwater reaches against field reports and SAR imagery. Put this in the caption, not in a footnote.

---

## 6. Uncertainty experiments

| Experiment | Method | Reports |
|---|---|---|
| Parameter intervals, all | Laplace (inverse Hessian) | quick reference table |
| Parameter intervals, headline | **Profile likelihood** for `α_FL′`, `ψ`, `w₁w₂w₃` | the intervals quoted in the abstract |
| Full uncertainty | Parametric bootstrap, 200 replicates | the version in the paper if time allows |
| Predictive bands | 200 parameter draws pushed through `λ → P` | the dashboard bands |
| **Probability quality** | Reliability diagram, Brier decomposition, CRPS on the validation period | **proof that our probabilities mean something** |
| Multi-start spread | 20 restarts | evidence about local optima |

Hessian-based intervals are typically optimistic for point processes. Anything we make a claim about gets a profile-likelihood interval.

---

## 7. Metrics — the full set

### Point-process quality
Held-out log-likelihood (time-forward split) · AIC · BIC · likelihood-ratio tests on nested pairs · time-rescaling residual QQ · spatial residual map · branching ratio.

### Forecast quality
AUC-ROC · **AUC-PR** · Brier score and decomposition · CRPS · reliability diagram.

### Early-warning quality
**Lead time and false-alarm rate, always reported together** · precision–recall over alert thresholds · **the lead-time versus false-alarm trade-off curve** — the operationally meaningful figure · hit rate at fixed false-alarm rate.

---

## 8. Sensitivity analyses — required, not optional

Repeat the headline result under each variation and report whether the conclusion changes.

| Variation | Tests |
|---|---|
| Flood threshold `Q₅` instead of `Q₂` | dependence on the flood definition |
| Declustering window 5 d instead of 3 d | dependence on declustering |
| Eligibility 250 km² instead of 50 km² | dependence on GloFAS reliability limits |
| Negative sampling 1:1 and 1:20 | dependence on Part 1's sampling choice |
| Slope units 0.1 km² and 1 km² | dependence on spatial resolution |
| Completeness cutoff `A_min` raised | dependence on inventory incompleteness |

If the sign or significance of `α_FL′` changes under any of these, **say so prominently**. A result that survives six sensitivity analyses is worth ten times a result that was never tested.

---

## 9. The negative-result plan

Written now, so that a null finding cannot derail the project in December.

**If A1 shows no degradation** — cross-excitation is not detectable in this catalog:

1. Report the likelihood-ratio test result with its p-value.
2. Compute a **power analysis**: given the observed effect size and variance, how many cascade pairs would be needed for 80% power? State that number.
3. Reframe the paper's contribution as: (a) the first multitype spatio-temporal Hawkes model for rainfall-triggered hazards on a hybrid hillslope–network support, (b) a rigorous demonstration of the data requirements for detecting cross-hazard coupling, (c) the lead-time analysis for landslides, which does not depend on cross-excitation.
4. Retitle accordingly.

That paper is publishable. Many good papers are shaped exactly like it. Planning for it in Week 10 rather than discovering it in Week 14 is the difference between a controlled result and a panic.

---

## 10. Four-person split

| Member | Task |
|---|---|
| **Ronit** | Owns the register; runs all ablations A1–A9; the Wayanad backtest; the money figure |
| **Raj** | Independently audits the leakage assertion; profile likelihood; bootstrap; multi-start analysis; power analysis if needed |
| **Chetan** | LOCO refits; spatial transfer to Maharashtra; all maps and spatial residual figures |
| **Yugant** | Negative controls; calibration and reliability diagrams; all six sensitivity analyses; the baseline comparison table |

---

## 11. Deliverables

- `experiments/register.csv` — every run
- `reports/phase5_baselines.md` — B1–B5 vs M5, one table
- `reports/phase5_ablations.md` — the nine-row ablation table
- `reports/phase5_loco.md` — lead-time distribution across six events
- `reports/phase5_sensitivity.md` — six variations
- `reports/phase5_backtest.md` — the Wayanad protocol and result
- `figures/money_figure.png` — **the figure the project exists to produce**
- `figures/leadtime_vs_falsealarm.png`
- `figures/ablation_table.png`
- `figures/reliability_diagram.png`
- `figures/residual_qq.png`, `residual_map.png`
