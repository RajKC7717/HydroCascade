# Document 07 — Phase 4: The Cross-Hazard Cascade Model

**Weeks 7–10 · Owner: Raj Choudhari · Reviewer: Ronit Dahiwal**

**Goal:** implement, test and fit the model specified in Document 02, in five stages, where each stage is verified before the next begins.

**The rule that governs this whole phase:** never add a component until the previous one has been verified on data where you already know the answer. If you go straight to the full model and it produces nonsense, you will not know whether the bug is in the mathematics, the code, the spatial framework, or the catalog. That uncertainty is what kills projects in week thirteen.

---

## 1. The five stages

| Stage | What is added | Verification before proceeding |
|---|---|---|
| **S1** | 1-D Hawkes in time only | Recover known parameters from simulated data |
| **S2** | Space added (2D isotropic), single hazard type | Recover known parameters from simulated space-time data |
| **S3** | Two hazard types, isotropic kernels, real data | Fits, converges, branching ratio < 1 |
| **S4** | Hybrid support: network, projection, dilution | Beats S3 on held-out log-likelihood, or the reason is understood |
| **S5** | Travel-time shift and mechanism mixture | Beats S4, or components are honestly dropped |

Each stage is a git tag. Each stage's results are recorded before the next begins.

---

## 2. Stage S1 — temporal Hawkes, and the correctness test

### Build

Implement the 1-D Hawkes log-likelihood in PyTorch:

> `log L = Σ_k log λ(t_k) − ∫₀^T λ(t) dt`, with `λ(t) = μ + Σ_{t_k<t} α·exp(−(t−t_k)/θ)`

The integral is analytic: `∫ = μT + Σ_k α·θ·(1 − exp(−(T−t_k)/θ))`.

### Verify — this is the whole point of the stage

1. Simulate a Hawkes process with **known** `μ, α, θ` using Ogata's thinning algorithm.
2. Fit with your own code.
3. **Recover the known values** to within a few percent, across at least 20 simulated catalogs of realistic size.
4. Plot recovered vs true for each parameter.

**If you cannot recover known parameters, do not proceed.** Fix the likelihood first.

### Independent cross-check

In a throwaway environment on **Python 3.12**, simulate with the `tick` library and fit with your own code. Two independent implementations agreeing is real evidence. Note that `tick`'s *fitting* classes are known to break on Python 3.13 — its simulator still works, which is all we need. `tick` is a correctness check, never a runtime dependency: our model does not exist in any library.

### Also build here

The **diagnostics module**, because it is easier to write when nothing is on fire:
- time-rescaling residuals and the QQ plot
- branching ratio computation
- convergence trace plotting

---

## 3. Stage S2 — adding space

Add an isotropic 2D Gaussian kernel and a spatially varying background on the quadrature grid.

The integral now splits: background over (unit × day), triggering as (analytic time) × (spatial integral ≈ `2πσ²` away from boundaries, numerical near them).

**Verify:** simulate a space-time Hawkes with known `σ`, fit, recover. Include a case where events cluster near the domain boundary, to confirm the boundary correction works.

---

## 4. Stage S3 — two hazard types, real data, isotropic

First contact with the real catalog.

### Build
- the `α_ij` matrix (4 entries), per-pair `θ_ij`, mark terms `η_i`
- the full background from Document 02, §6, using the calibrated `S_i` and `ẽ` from Phase 3
- the quadrature grid: (slope unit × day) for landslides, (reach × day) for floods

### Expect problems here, and know what they look like

| Symptom | Likely cause |
|---|---|
| Log-likelihood is `nan` | exponentiating a large positive number; missing log-sum-exp; a zero in a denominator |
| Branching ratio ≥ 1 | `α` unconstrained, or the catalog has undeclustered clusters |
| `α` collapses to zero | background too flexible, absorbing everything; or truncation radius too small |
| Wildly different results across restarts | multimodal likelihood — real, and must be reported |
| Fit takes hours per iteration | truncation not applied; kernel evaluated inside a Python loop |

### Verify
- converges from 20 random starts; report the spread
- branching ratio strictly below 1
- beats a Poisson model (M2) on held-out log-likelihood
- residual QQ plot near the diagonal

**Milestone:** this is a working multitype spatio-temporal Hawkes model on real Kerala data. Even if nothing else works, this is a legitimate result. Tag it.

---

## 5. Stage S4 — the hybrid support

Now the model becomes ours.

### Build

Replace the flood-side spatial kernel with the network formulation from Document 02, §5:

1. **Connectivity gate 1 — runout reachability:** `1[d_slope(x_k) ≤ c·exp(γ·m_k)]`
2. **Connectivity gate 2 — downstream:** `1[s ∈ Down(π(x_k))]`, from the precomputed reachability index
3. **Dilution:** `(A(π(x_k)) / A(s))^ψ`
4. **Flood → landslide corridor:** restrict to `w(x) < w_max`

The reachability index and area ratios are already precomputed from Phase 2. **Do not recompute them here.**

### Implementation note that matters

Assemble all active (event, target unit) pairs into **one flat tensor** per time step, then evaluate the kernel in a single vectorised operation. No Python loops inside the likelihood. Apply temporal truncation (drop terms older than five time constants) and network truncation (only reaches within the precomputed downstream window) *before* assembling the tensor.

### Verify
- beats S3 on held-out log-likelihood, or you can explain precisely why not
- `ψ` is identified with a sensible interval
- the runout gate does not exclude the Wayanad event — check explicitly that with the fitted `c, γ`, a landslide of Mundakkai's mapped area at `d_slope` from the channel passes the gate. If it does not, the gate parameterisation is wrong

---

## 6. Stage S5 — wave propagation and mechanism mixture

The two ideas that make this novel.

### 6.1 The travel-time shift

Replace `φ(t − t_k)` with `φ(t − t_k − τ(π(x_k) → s))`, where `τ = T_ref / v₀` and `T_ref` comes from the Phase 2 table. `φ` is defined to be zero for negative arguments.

`v₀` is learned, constrained to give physical speeds of 1–5 m·s⁻¹.

**Verify:** compare held-out log-likelihood against S4 (unshifted). Also produce the direct physical check: for the Wayanad event, the model's predicted arrival window at the downstream Chaliyar reaches should be consistent with the observed timing of downstream impacts.

### 6.2 The mechanism mixture

Stage this according to the data, per Document 02, §4.4:

| Version | Components | Proceed only if |
|---|---|---|
| v1 | single exponential | always |
| v2 | fast + slow exponential | ≥ 30 identified cascade pairs |
| v3 | add the delayed gamma | ≥ 100 pairs, and v2 stable |

Bound each timescale to its physical range: `θ₁ ∈ [1 h, 24 h]`, `θ₂ ∈ [1 d, 14 d]`, `θ₃ ∈ [2 w, 6 mo]`.

**If a component's weight is not significantly different from zero, report that and drop it.** "In this catalog we could not detect a delayed dam-breach component, and here is the sample size that would be needed" is a real, publishable statement.

### 6.3 The headline numbers

At the end of S5 you should be able to state, each with an uncertainty interval:

| Quantity | Meaning |
|---|---|
| `α_FL′` | how strongly a landslide raises downstream flood intensity |
| `θ` for L→F, as a **half-life in days** | how long the elevated risk lasts |
| `ψ` | how fast the effect dilutes as the catchment grows |
| `v₀` | the fitted channel celerity — check it is physically plausible |
| `w₁, w₂, w₃` | which mechanism dominates in the Western Ghats |
| branching ratio | model stability |

**`α_FL′` with a 95% interval that excludes zero is the single most important number in the project.**

---

## 7. Fitting protocol

Applies at every stage from S3 onward.

1. Adam, 2,000 steps, learning rate decayed; then L-BFGS to polish.
2. **20 random restarts minimum.** Report the spread of final log-likelihoods.
3. Gradient clipping.
4. Early stopping on validation log-likelihood.
5. `float64` throughout.
6. Save the optimiser trajectory for the convergence plot.
7. After every fit: compute the branching ratio. **If ≥ 1, the fit is invalid regardless of likelihood.**
8. Every fit writes a JSON record: parameters, intervals, log-likelihood, AIC, seed, git commit, wall-clock time.

That last item makes every number in the paper traceable to a specific run.

---

## 8. Performance targets

| Operation | Target |
|---|---|
| One likelihood + gradient on the full catalog | under 60 s, laptop CPU |
| One full fit, single start | under 2 h |
| 20 restarts | overnight |

If the first target is missed, **reduce spatial resolution first** — larger slope units, 1 km reaches — before reducing scientific ambition. Losing resolution costs precision; losing a component costs the contribution.

---

## 9. Four-person split

| Member | Task |
|---|---|
| **Raj** | S1, S2, S3: the likelihood, the optimiser, the parameter recovery tests. Owns `cascade/likelihood.py` and `cascade/fit.py` |
| **Chetan** | S4: the network kernel, wiring the Phase 2 precomputed tables into the model, the flat-tensor assembly and truncation. Owns `cascade/kernels.py` |
| **Yugant** | The background: plug in calibrated `S_i` and `ẽ`, fit the `θ` rainfall coefficients, verify the interface with Phase 3 |
| **Ronit** | S5: mechanism mixture, the significance test on `w₁,w₂,w₃`, profile likelihood for `α_FL′` and `ψ`, the diagnostics suite |

---

## 10. Quality gates

| # | Gate |
|---|---|
| 1 | Parameter recovery verified on synthetic data at S1 and S2 |
| 2 | Independent cross-check against `tick` on simulated data passes |
| 3 | Branching ratio < 1 at every stage |
| 4 | Residual QQ plot near diagonal for the chosen model |
| 5 | 20-restart spread reported |
| 6 | Wayanad event passes the runout connectivity gate under the fitted parameters |
| 7 | Fitted `v₀` corresponds to a physically plausible channel celerity |
| 8 | Every reported number traceable to a saved fit record |

---

## 11. If it does not work

Two failure modes, both survivable, both planned for.

**The model fits but cross-excitation is not significant.** Report it. Compute how many cascade pairs would be needed for adequate power. Fall back to the lead-time analysis as the paper's headline. This is a legitimate finding about data limits and it is publishable.

**The model will not fit at all.** Reduce, in this order: drop the gamma component → drop the mechanism mixture entirely → fix `θ` from literature values instead of fitting → fix `c, γ` for the runout gate → reduce to isotropic (S3) and present the network formulation as a specified-but-unfitted extension.

Falling back to S3 with an honest explanation still leaves you with a working multitype spatio-temporal Hawkes model for Kerala hazards, which nobody has published. Do not treat S5 as all-or-nothing.
