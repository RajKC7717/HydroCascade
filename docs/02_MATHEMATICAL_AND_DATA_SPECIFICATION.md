# Document 02 — Mathematical and Data Specification

**Owner:** Raj Choudhari (Mathematics and Model Lead), with §6–7 co-owned by Yugant Chincholikar
**Status:** This is the authoritative specification. If any other document conflicts with this one, this one wins.

This document answers the questions a reviewer will ask. Each section states a definition precisely, then translates it into plain English, then says how we will test that the definition was a good choice.

---

## 1. What exactly are we predicting?

This must be stated first, because the words *hazard*, *susceptibility*, *intensity* and *risk* mean different things and we have used them loosely before.

### 1.1 Three distinct quantities

**(a) Conditional intensity** — the model's native output.

> λ_i(u, t | ℋ_t) = the expected number of type-`i` events per unit space per unit time, at location `u`, at time `t`, given everything that has happened up to time `t`.

Units:
- for landslides, `i = L`, `u` on the hillslope surface: **events · day⁻¹ · km⁻²**
- for floods, `i = F`, `u` on the river network: **events · day⁻¹ · km⁻¹** (per kilometre of channel)

λ is **not** a probability. It is a rate. It can exceed 1.

**(b) Occurrence probability** — what the dashboard shows.

For a spatial unit `A` and a forecast horizon `Δ`:

> P_i(A, t, Δ) = 1 − exp( − ∫ₜ^(t+Δ) ∫_A λ_i(u, s | ℋ_t) du ds )

This is bounded in [0, 1] and is the standard void-probability conversion for a point process. Note the conditioning is frozen at `ℋ_t` — we do not know about future events, so this is the probability under the current history.

**(c) Alert level** — an ordinal decision.

An integer in {Normal, Watch, Warning, Emergency}, obtained by thresholding either `P_i` or the local percentile of `λ_i`. Defined in Document 09, §2.

### 1.2 What we do NOT predict

We predict **hazard occurrence**. We do **not** predict:

- loss, damage, casualties, economic cost
- exposure or vulnerability of people and buildings
- the physical extent of inundation or the runout footprint
- severity as an outcome variable (severity enters as an input mark, not a prediction target, in version 1)

This must appear in the paper's abstract. In disaster science, "risk" conventionally means hazard × exposure × vulnerability. **We model only the hazard term.** Saying so plainly removes the single most likely reviewer objection.

### 1.3 Vocabulary, fixed for the whole project

| Term | Fixed meaning here |
|---|---|
| **Susceptibility** | A time-independent score, `S_i(x)`, of how prone a place is. A map. Output of Part 1 |
| **Intensity** | `λ_i(u,t)`, a rate that changes with time and history. Output of Part 2 |
| **Probability** | `P_i(A,t,Δ)`, derived from λ over a stated area and horizon |
| **Hazard** | Susceptibility combined with a time dimension, i.e. the intensity field |
| **Risk** | Reserved. We do not use this word for our output. If we must, we say "hazard occurrence probability" |
| **Mark** | A number describing an event's size, `m` |

Everyone on the team uses these words this way, in every document and every presentation.

---

## 2. What does "location" mean? The hybrid spatial framework

### 2.1 The problem

A landslide happens on a hillside — a two-dimensional surface. A flood happens in a river — a one-dimensional channel. Saying "the model is continuous in space" without saying *which* space is not a specification.

### 2.2 The definition

The model is defined on a **hybrid support** with two connected components.

**Hillslope domain H**
A bounded region of ℝ², in UTM Zone 43N (EPSG:32643). Landslide events are points `x ∈ H`.
Computational representation: **slope units** — irregular polygons derived from the DEM by half-basin partitioning, target median area 0.25 km². Roughly 80,000 units over the study region. Slope units are used rather than square pixels because a landslide belongs to a hillside, not to a grid square.

**Channel network N**
A 1D metric graph embedded in ℝ², derived from the DEM: cells with contributing area above 1 km² are channel; the resulting network is split into **reaches** of nominal length 500 m. Flood events are points `s ∈ N`, located by their along-network coordinate. Roughly 15,000 reaches.

**The projection operator π : H → N**
For a hillslope point `x`, follow the D8 flow path downhill until it first enters a channel cell. That cell is the **injection point** `π(x) ∈ N`.

*In plain English:* if debris leaves this hillside, this is the first place in the river it would land.

This operator is the physical bridge between the two domains. It is computed once, for every slope unit, and stored.

### 2.3 Precomputed network quantities

Computed once in Phase 2 and cached. All are needed by the kernel in §5.

| Symbol | Name | Definition | Units |
|---|---|---|---|
| `d_slope(x)` | slope-to-channel path length | length of the D8 path from `x` to `π(x)` | km |
| `Down(p)` | downstream set | all network locations reachable from `p` by travelling downstream | — |
| `d_N(p → s)` | network distance | along-channel distance from `p` to `s`, defined only if `s ∈ Down(p)` | km |
| `A(s)` | contributing area | upstream catchment area draining to `s` | km² |
| `v(s)` | channel celerity | wave speed in reach `s` (see §5.3) | km · h⁻¹ |
| `τ(p → s)` | travel time | ∫ along the path from `p` to `s` of `ds / v(s)` | h |
| `w(x)` | channel proximity | Euclidean distance from `x` to the nearest channel | km |

### 2.4 Why this is defensible, not invented

Point processes on **linear networks** are an established branch of spatial statistics, with geometrically corrected summary statistics (Ang, Baddeley and Nair, *Scandinavian Journal of Statistics*, 2012), a standard software implementation (`spatstat.linnet` in R), and published **multitype** analyses on networks (Baddeley, Jammalamadaka and Nair, *JRSS-C*, 2014). Cox processes on linear networks are also an active topic.

So we are not inventing a strange space. We are **combining two established supports with a physically defined operator between them**, which — as far as our search found — has not been done for natural hazards.

### 2.5 Discretisation

| Axis | Research mode | Real-time mode |
|---|---|---|
| Time | 1 day | 1 hour |
| Hillslope | slope unit | slope unit |
| Channel | 500 m reach | 500 m reach |

The model is *specified* in continuous time and space. It is *fitted* by evaluating the likelihood on a quadrature grid, which is standard: a point-process likelihood approximated on a quadrature grid becomes a weighted Poisson regression (the Berman–Turner device; see Baddeley and Turner). We state this in the paper rather than apologising for it.

**Why one day and not one hour for fitting:** many inventory records give a date but not a time, and a single storm can produce thousands of landslides all sharing one date. A continuous-time likelihood assumes no exact ties. Daily discretisation handles this honestly. Where timestamps exist (Wayanad: 02:17 IST), they are retained in the catalog and used in the backtest at hourly resolution.

---

## 3. What exactly is a flood event?

This was the weakest definition in the earlier plan. It is now fully specified.

### 3.1 Primary definition — discharge exceedance

We use GloFAS v4.0 hydrological reanalysis (daily, 0.05°, 1980 onward), which is free from the Copernicus Climate Data Store.

**Step 1 — thresholds.**
For each river cell, extract annual maxima of daily discharge from the reanalysis, fit a Gumbel distribution by the L-moments method, and read off the discharge magnitudes corresponding to the **2-, 5- and 20-year return periods**: `Q₂(r)`, `Q₅(r)`, `Q₂₀(r)`.

This is exactly the procedure GloFAS itself uses to define its own operational flood thresholds. We adopt it deliberately so our definition matches an operational system rather than being invented for this project.

**Step 2 — event detection.**

> A **flood event** at reach `r` occurs on the first day `t` of an episode in which daily discharge `Q(r,t)` exceeds `Q₂(r)`.

**Step 3 — declustering.**
An episode ends when `Q(r,·)` stays below `Q₂(r)` for **3 consecutive days**. Only then can a new event be recorded at that reach. Without this rule, a single week-long flood would be counted as five events and would badly corrupt the self-excitation parameters.

**Step 4 — mark and class.**

- Mark: `m_F = log( Q_peak / Q₂(r) )`, dimensionless, zero at threshold.
- Class: minor (`Q₂`–`Q₅`), major (`Q₅`–`Q₂₀`), extreme (above `Q₂₀`).

**Step 5 — eligibility filter.**
Only reaches with contributing area **above 50 km²** are eligible for flood event detection. Below that, GloFAS is not reliable — its published evaluation shows lowest skill for catchments under 10,000 km², and operational GloFAS reporting points are restricted to catchments above 250 km². We use 50 km² as a research-mode compromise and report a sensitivity check at 250 km².

### 3.2 Corroboration layer

The primary definition is **model-derived**, so it must be checked against observations. The corroboration layer is **not used for fitting**. It is used to measure how good the definition is.

| Source | What it gives | Coverage |
|---|---|---|
| **Sentinel-1 SAR inundation** | Actual water extent, through monsoon cloud. Otsu automatic thresholding of VV backscatter in Google Earth Engine is a published method for Kerala with reported overall accuracy around 94% | Every ~6–12 days per orbit; good for major events |
| **CWC gauge water level and discharge** | Ground observation | Sparse; via the National Water Data Portal |
| **Copernicus EMS / UNOSAT rapid mapping** | Official activation footprints | Only for the largest events |
| **KSDMA reports, GSI field reports, news archives** | Human confirmation of flooding at named places | Major events only |

### 3.3 The corroboration experiment — a required deliverable

This converts a definitional choice into a measured number.

1. Take all GloFAS-detected flood events in the study period.
2. Restrict to those with usable Sentinel-1 coverage within ±3 days.
3. For each, compute inundated area from SAR and compare with a non-flood reference date.
4. Report **precision**: what fraction of GloFAS-flagged events show a real inundation signal.
5. Separately, take the set of independently documented major floods (KSDMA, EMS, news) and report **recall**: what fraction did GloFAS flag.

Report both numbers in the paper. If precision is poor at `Q₂`, raise the threshold to `Q₅` and repeat. This is a defensible, testable data-definition process rather than an assumption.

### 3.4 The honest limitation, stated up front

The Punnapuzha at Chooralmala — the channel the Wayanad debris flow entered — has a contributing area far **below** the GloFAS eligibility threshold. **GloFAS cannot see that flood.**

Therefore, for the Wayanad backtest specifically:

- the *landslide* is evidenced by the GSI field report and published studies;
- the *downstream flood* is evidenced by field reports, the destroyed bridge, SAR imagery, and the Chaliyar discharge signal further downstream where the catchment is large enough;
- the model's **predicted** flood intensity at the small headwater reaches is therefore validated qualitatively (did risk rise in the right places at the right time?) and quantitatively only at the larger downstream reaches.

We say this in the paper. Hiding it would be the fastest way to lose a reviewer.

### 3.5 Sensitivity analysis — required

Repeat the headline analysis with:
- threshold `Q₅` instead of `Q₂`
- declustering window 5 days instead of 3
- eligibility 250 km² instead of 50 km²

Report whether the sign and significance of the cross-hazard coupling change. If they do, the result depends on a definitional choice and must be reported as such.

---

## 4. What physical mechanism does the cross-hazard kernel represent?

A landslide does not automatically produce a flood. Three distinct mechanisms exist, they operate on very different timescales, and the model is built to tell them apart.

### 4.1 The three mechanisms

**Mechanism A — bulking and direct debris discharge. Timescale: minutes to hours.**
Sediment delivered by the landslide bulks the flow, increasing its bulk density and therefore its power, which is why debris flows and debris floods are more erosive than clear-water floods. The mass arrives downstream almost immediately, moving at flow speed. This is what happened at Wayanad: a long-runout debris flow travelled 6–8 km and produced flash flooding, not an impounded lake.

**Mechanism B — damming, impoundment and breach. Timescale: hours to weeks.**
The landslide partially or completely blocks the channel, water impounds behind it, and the dam later breaches, producing an outburst flood downstream. Whether a dam forms at all depends on the momentum of the mass relative to the river flow: a low-momentum mass is deflected and only partially blocks the channel, while a high-momentum mass can span the full width. **The critical property for our model is that this mechanism has a delayed peak** — risk may be low immediately and high a day later.

**Mechanism C — aggradation and reduced conveyance. Timescale: weeks to years.**
Sediment deposited in the channel raises the bed. Documented cases show bed elevation rising by metres. A channel that could previously convey a given flood can no longer do so, and later, ordinary floods spill onto land that was previously safe. This raises **background** flood risk persistently, long after the landslide.

### 4.2 How the model separates them

The temporal excitation kernel for the landslide → flood pair is a **three-component mixture**:

> φ_LF(Δt) = w₁ · Exp(Δt ; θ₁) + w₂ · Gamma(Δt ; k, θ₂) + w₃ · Exp(Δt ; θ₃)
> with w₁ + w₂ + w₃ = 1, and θ₁ ≪ θ₂ ≪ θ₃

| Component | Mechanism | Shape | Expected timescale |
|---|---|---|---|
| `w₁, θ₁` | A — bulking | exponential, peaks immediately | hours |
| `w₂, k, θ₂` | B — dam breach | **gamma, peaks later** | 1–14 days |
| `w₃, θ₃` | C — aggradation | exponential, long tail | weeks to months |

The gamma component matters because **a pure exponential cannot peak later than zero**, and Mechanism B does exactly that. Gamma-density excitation functions in Hawkes processes applied to natural disaster data have precedent (Lesage et al., 2022), so this is an established rather than invented choice.

### 4.3 Why this is a scientific result, not just a modelling detail

The fitted weights `(w₁, w₂, w₃)` answer a real question:

> **In the Western Ghats, does the landslide → flood link operate mainly by immediate debris bulking, by delayed dam breach, or by persistent channel aggradation?**

Nobody has estimated this from a regional event catalog. Whatever the answer is, it is publishable, and it is directly useful to disaster managers deciding how long to keep a downstream valley evacuated after an upslope failure.

### 4.4 Practical staging

Fitting three components with shape parameters needs data. Stage it:

| Version | Components | Condition to proceed |
|---|---|---|
| v1 | single exponential | always |
| v2 | fast + slow exponential (`w₁, w₃`) | ≥ 30 identified cascade pairs |
| v3 | add the gamma component (`w₂`) | ≥ 100 pairs, and v2 converged stably |

Fix `θ` to physically plausible bounds (θ₁ ∈ [1 h, 24 h], θ₂ ∈ [1 d, 14 d], θ₃ ∈ [2 w, 6 mo]) rather than leaving them unbounded. If a component's weight is not significantly different from zero, report that and drop it.

---

## 5. The measurable rule for downstream propagation

This section converts "influence follows the river" from a conceptual claim into a mathematical contribution.

### 5.1 The three factors

Excitation from a landslide at `x_k` to a channel location `s` is the product of three independent factors, each separately ablatable:

> **CONNECTIVITY** × **DILUTION** × **ARRIVAL TIMING**

### 5.2 Connectivity — can it affect this place at all?

Two binary gates.

**Gate 1 — did the debris reach a channel?**
A landslide only injects into the network if its runout is long enough to get there:

> `reach_k = 1[ d_slope(x_k) ≤ L_max(m_k) ]` where `L_max(m) = c · exp(γ · m)`

`m_k` is the log mapped area. `c` and `γ` are learned, initialised from the observed range: typical shallow Western Ghats slides run a few hundred metres; Wayanad ran 6–8 km.

*In plain English:* a small slip 3 km from the nearest stream cannot cause a flood. A large debris flow 3 km from a stream can.

**Gate 2 — is the target downstream?**

> `down_k(s) = 1[ s ∈ Down(π(x_k)) ]`

*In plain English:* a landslide cannot flood a valley that is uphill of it.

### 5.3 Dilution — how much does the effect weaken as the river grows?

A sediment and water pulse that dominates a 5 km² headwater stream is negligible in a 2,000 km² river. This is captured by the **contributing area ratio**:

> `Ω_k(s) = ( A(π(x_k)) / A(s) )^ψ`, with `ψ ≥ 0` learned

`Ω = 1` at the injection point and decreases downstream. `ψ = 0` would mean no dilution; a fitted `ψ` significantly above zero is a testable physical finding.

### 5.4 Arrival timing — the wave, not the field

**This is the core novel idea.**

In a standard Hawkes process, excitation appears everywhere instantly and decays. Physically, that is wrong for a river: a flood wave *travels*. A reach 20 km downstream feels the effect after the travel time, not immediately.

**Channel celerity.** For each reach, wave speed is estimated from channel slope:

> `v(s) = v₀ · S(s)^0.3`, with `v₀` learned, `S(s)` the reach slope

(An exponent near 0.3 follows from Manning-type kinematic wave scaling. `v₀` is initialised to give physically sensible speeds of roughly 1–5 m·s⁻¹ and is constrained to that range.)

**Travel time** is then `τ(π(x_k) → s) = Σ over reaches on the path of ( length / v )`.

**The shifted kernel.** The excitation from event `k` at location `s` and time `t` is:

> `φ_ij( t − t_k − τ(π(x_k) → s) )`

with `φ` defined to be zero for negative arguments.

*In plain English:* the risk at a downstream reach does not rise the instant the landslide happens. It rises when the water and debris get there. And it rises for a while, then falls.

**Why this matters, in one sentence for the paper:** *the model predicts not only that a downstream flood is more likely, but when it should arrive* — and that arrival window is directly checkable against the observed Chaliyar hydrograph.

### 5.5 The complete kernel, by hazard pair

| Pair | Support | Connectivity | Timing | Physical meaning |
|---|---|---|---|---|
| **L → L** | H → H | isotropic Gaussian, `exp(−‖x−x_k‖² / 2σ_LL²)` | unshifted | same storm cell, same slope system, retrogressive failure |
| **L → F** | H → N | Gate 1 × Gate 2 × `Ω^ψ` | **shifted by τ** | the three mechanisms of §4 |
| **F → L** | N → H | `1[w(x) < w_max] · exp(−d(x, s)² / 2σ_FL²)` | short lag | bank undercutting and toe erosion destabilising near-channel slopes |
| **F → F** | N → N | Gate 2 × `Ω^ψ` | **shifted by τ** | flood wave routing downstream |

Note `F → L` is deliberately restricted to a corridor of width `w_max` around the channel. A flood cannot undercut a slope 5 km from the river.

### 5.6 The full intensity function

**Landslides**, at hillslope location `x`, time `t`:

> λ_L(x,t) = μ_L(x,t) + Σ over past landslides k: α_LL · e^(η_L m_k) · g_L(x) · e^(−‖x−x_k‖²/2σ_LL²) · φ_LL(t−t_k)
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Σ over past floods k: α_LF′ · e^(η_F m_k) · g_L(x) · 1[w(x)<w_max] · e^(−d(x,s_k)²/2σ_FL²) · φ_FL(t−t_k)

**Floods**, at network location `s`, time `t`:

> λ_F(s,t) = μ_F(s,t) + Σ over past landslides k: α_FL′ · e^(η_L m_k) · g_F(s) · reach_k · down_k(s) · Ω_k(s)^ψ · φ_LF(t − t_k − τ_k(s))
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+ Σ over past floods k: α_FF · e^(η_F m_k) · g_F(s) · down_k(s) · Ω_k(s)^ψ · φ_FF(t − t_k − τ_k(s))

*(Notation note: `α_LF′` denotes flood-excites-landslide, `α_FL′` denotes landslide-excites-flood. Fix one convention in code and comment it, because sign and subscript confusion here is the most likely source of a silent bug.)*

---

## 6. The background term — where Part 1 lives

> μ_i(u, t) = μ_i⁰ · S_i(u) · exp( θ_i¹·R₁(u,t) + θ_i³·R₃(u,t) + θ_i⁷·R₇(u,t) + θ_i¹⁵·R₁₅(u,t) )

| Symbol | Meaning | Source |
|---|---|---|
| `μ_i⁰` | base rate per unit space per day | fitted |
| `S_i(u)` | calibrated susceptibility score at `u`, in (0,1) | **Part 1 output** |
| `R_d(u,t)` | rainfall accumulated over the previous `d` days, standardised | IMD + IMERG |
| `θ_i^d` | rainfall response coefficients | fitted |

For a flood reach, `R_d(s,t)` is the **catchment-averaged** rainfall over the area draining to `s`, not point rainfall at the reach. Rain falling on the hills is what floods the river.

**Why multiple rainfall windows rather than an intensity–duration threshold.** A 2025 two-part review of rainfall-induced landslide prediction found that antecedent rainfall is a more robust indicator than classical intensity–duration thresholds, and that fixed I–D thresholds suffer from high false-alarm rates because they ignore antecedent soil wetness. Independently, a 2025 NHESS study of 22 tropical hazard cascades found that only a few were linked to the most extreme rainfall on record, implying that other controls — including antecedent wetness and sediment availability — matter. Our multi-window formulation is the statistically honest version of both findings.

### 6.1 The receptivity term

> g_i(u) = exp( η_i · ẽ(u) )

where `ẽ(u)` is a vector of the **top two to four** standardised drivers identified by Part 1's SHAP ranking. Not more. Every extra parameter here costs identifiability we cannot afford.

`g_i` answers: *given that something is pushing this place, how easily does this place actually fail?*

### 6.2 Calibration requirement

`S_i(u)` is used as a **multiplier**, so a badly calibrated susceptibility score corrupts the whole cascade model. Part 1 must therefore deliver a *calibrated* probability, checked with a reliability diagram and corrected by isotonic or Platt scaling if needed. This is a hard requirement on Phase 3, not a nice-to-have.

---

## 7. Fitting

### 7.1 The likelihood

For a multitype point process observed on `[0, T]` over supports `H` and `N`:

> log L = Σ over all observed events k of log λ_(type k)(u_k, t_k)
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;− ∫₀^T ∫_H λ_L(x,t) dx dt − ∫₀^T ∫_N λ_F(s,t) ds dt

The first term is easy. The two integrals are the work.

### 7.2 How the integrals are computed

**Split the integral into background and triggering.**

*Background:* `μ` is piecewise constant on (spatial unit × day), so its integral is a simple weighted sum over the quadrature grid — units multiplied by their area (or length), summed over days.

*Triggering, time part — analytic.* For the exponential components:

> ∫_{t_k+τ}^{T} exp(−(t − t_k − τ)/θ) dt = θ · ( 1 − exp( −(T − t_k − τ)/θ ) )

For the gamma component, use the regularised lower incomplete gamma function, available in `scipy.special` and differentiable in PyTorch via `torch.special`.

*Triggering, space part.* For the L→L isotropic Gaussian, the integral over `H` is approximately `2πσ_LL²` when far from the boundary, and evaluated numerically on the grid near boundaries. For network kernels, the integral is a finite sum over reaches in `Down(π(x_k))` weighted by reach length — there is no continuous integral to approximate, because the network is a finite set of segments.

**This decomposition is the single most useful implementation note in this document.** It is where student implementations usually get stuck.

### 7.3 Parameterisation and constraints

| Parameter | Constraint | Transform |
|---|---|---|
| `α_ij` | ≥ 0 | `softplus` |
| `θ` (all timescales) | > 0, bounded | `sigmoid` scaled to the physical range in §4.4 |
| `σ, τ₀, v₀` | > 0 | `softplus`, `v₀` bounded to 1–5 m·s⁻¹ |
| `ψ, γ, c` | ≥ 0 | `softplus` |
| `w₁, w₂, w₃` | simplex | `softmax` |
| `η, θ_i^d` | unconstrained | identity |

Use `float64`. Use log-sum-exp. Never exponentiate a large positive number directly.

### 7.4 Stability check — mandatory after every fit

Compute the **branching ratio**: the spectral radius of the matrix whose `(i,j)` entry is the expected number of type-`i` offspring produced by one type-`j` event. It must be **strictly below 1**. If it is not, the fitted process is explosive and the fit is invalid regardless of how good the likelihood looks.

### 7.5 Optimisation protocol

1. Adam with a decaying learning rate for 2,000 steps, then L-BFGS to polish.
2. **20 random restarts minimum.** Report the spread of the final log-likelihoods. A wide spread means many local optima and must be disclosed.
3. Gradient clipping.
4. Early stopping on validation log-likelihood.
5. Save the full optimiser trajectory so convergence can be plotted in the paper.

---

## 8. Uncertainty

An early warning system that reports a confident wrong number is worse than one that reports honest doubt.

### 8.1 Parameter uncertainty — three methods, increasing rigour

| Method | Cost | Use for |
|---|---|---|
| **Laplace approximation** — invert the Hessian at the optimum | seconds | routine reporting of all parameters |
| **Profile likelihood** — refit with one parameter fixed across a grid | minutes per parameter | the headline parameters: `α_FL′` (landslide→flood coupling), `ψ` (dilution), `w₁,w₂,w₃` (mechanism weights) |
| **Parametric bootstrap** — simulate 200 catalogs from the fitted model, refit each | hours | the version reported in the paper, if time allows |

The Hessian-based interval is usually **optimistic** for point processes. Report profile-likelihood intervals for anything we make a claim about.

### 8.2 Predictive uncertainty

Draw `B = 200` parameter vectors from the Laplace or bootstrap distribution. Push each through the intensity to obtain `B` values of `P_i(A,t,Δ)`. Report the **median and the 5th–95th percentiles**.

The dashboard shows, for example: `0.34 [0.19 – 0.52]`.

### 8.3 Are the probabilities meaningful? — a required check

A probability is only useful if it is calibrated. On the validation period, produce:

- a **reliability diagram**: bin predicted probabilities, plot observed frequency against predicted probability, and check it lies near the diagonal
- the **Brier score** and its decomposition into reliability, resolution and uncertainty
- the **continuous ranked probability score (CRPS)** for the count forecasts

If the reliability diagram is far off the diagonal, apply post-hoc calibration on the validation set and say so.

---

## 9. Explainability — why did the risk go up *right now*?

### 9.1 Attribution is exact, not approximated

The intensity is a **sum of terms by construction**:

> λ_i(u,t) = μ_i(u,t) + c₁ + c₂ + … + c_K

so the share of each contributor is simply `c_k / λ_i(u,t)`. These shares sum to 1 exactly. There is no approximation, no sampling, no surrogate model.

**This is stronger than SHAP.** SHAP estimates attributions for a black-box model. Here the attribution *is* the model. Say this explicitly in the paper — it is a genuine advantage of the point-process formulation over a neural approach.

### 9.2 Required output format

For any (location, time) the system returns:

```
Chooralmala reach R-4471 — flood, next 24 h
Probability      0.61   [0.44 – 0.75]
Alert level      WARNING
Intensity        2.8 events · day⁻¹ · km⁻¹   (99.6th percentile for this reach)

Why:
  48%  landslide, Punchirimattam, 4.2 h ago, 2.1 km upstream
       expected arrival window 3–7 h · mechanism: fast bulking (w1)
  27%  15-day antecedent rainfall — 99.4th percentile locally
  14%  1-day rainfall — 98.1st percentile locally
  11%  background susceptibility (slope, drainage density)
```

The background term is further decomposed into its rainfall windows and the susceptibility factor, which is itself explained by Part 1's SHAP values. So the chain of explanation runs all the way from the alert down to the physical driver.

### 9.3 Part 1 keeps SHAP

Attribution inside the susceptibility model `S_i(x)` still uses SHAP, because that model *is* a black box. The two explanation systems compose cleanly: Part 2 says how much of the alert came from susceptibility, and Part 1 says why the susceptibility is high there.

---

## 10. Computational feasibility

### 10.1 The problem

| Quantity | Approximate value |
|---|---|
| Study region | ~20,000 km² |
| Slope units | ~80,000 |
| River reaches | ~15,000 |
| Events in catalog | ~5,000 |
| Days | ~2,400 |

Naive evaluation is every unit against every event on every day: `95,000 × 5,000 × 2,400 ≈ 10¹²` kernel evaluations per likelihood call. Infeasible, and it must be called thousands of times during optimisation.

### 10.2 Eight optimisations

1. **Temporal truncation.** Drop any excitation term older than five time constants; the omitted contribution is under 1%. At any given day only a few tens to a few hundred events are "active".
2. **Spatial truncation for L→L.** Only slope units within `3σ_LL` of an event contribute. Query with an R-tree or KD-tree.
3. **Network truncation for L→F and F→F.** Only reaches in `Down(π(x_k))` within a maximum travel time (say 48 h). Precompute the downstream reachability list per network node **once**, store as a sparse CSR structure.
4. **Precomputed travel-time table.** `τ(p → s)` for all downstream pairs within the truncation radius, computed once in Phase 2. This is the single biggest saving and is why Phase 2 builds it early.
5. **Flat-tensor batching.** Assemble all active (event, unit) pairs for a time step into one flat tensor and evaluate the kernel in a single vectorised PyTorch operation. No Python loops inside the likelihood.
6. **Cached static factors.** `S_i(u)`, unit areas, reach lengths, contributing areas and `Ω` do not change during optimisation. Compute once, hold as tensors.
7. **Incremental update in live mode.** When one new event arrives, only recompute the units it can reach. Cost is independent of catalog size.
8. **GPU-ready by construction.** The flat-tensor design moves to GPU with a device change and no code rewrite. Report CPU timings; treat GPU as optional.

### 10.3 Complexity after optimisation

Per day step: `O(N_active × k̄)` where `k̄` is the mean number of units an event can affect — a few hundred. That is on the order of `10⁴–10⁵` operations per day, roughly `10⁷–10⁸` per full likelihood evaluation.

### 10.4 Performance targets — to be measured and reported

| Operation | Target |
|---|---|
| One full likelihood + gradient evaluation on the complete catalog | **under 60 seconds** on a laptop CPU |
| Full fit, single start | under 2 hours |
| 20 restarts | overnight |
| One incremental real-time update after a new event | **under 2 seconds** |
| Precomputation of the travel-time and reachability tables | one-off, under 4 hours |

If the first target is missed, reduce the spatial resolution (larger slope units, 1 km reaches) before reducing scientific ambition.

---

## 11. Complete parameter inventory

Everything the model estimates, so identifiability can be assessed honestly.

| Group | Parameters | Count (2 hazard types) |
|---|---|---|
| Background base rates | `μ_L⁰, μ_F⁰` | 2 |
| Rainfall response | `θ_i^d`, 4 windows × 2 types | 8 |
| Receptivity | `η_i`, up to 4 drivers × 2 types | ≤ 8 |
| Coupling strengths | `α_LL, α_FF, α_LF′, α_FL′` | 4 |
| Marks | `η_L, η_F` | 2 |
| Temporal kernels | `θ` per pair; up to 3 components for L→F | 6–10 |
| Mechanism weights | `w₁, w₂, w₃` (L→F only) | 2 free |
| Spatial spreads | `σ_LL, σ_FL` | 2 |
| Network parameters | `ψ` (dilution), `v₀` (celerity), `c, γ` (runout gate) | 4 |
| **Total** | | **≈ 40–44** |

With a catalog of 1,000+ events this is estimable, though the mechanism weights and the runout-gate parameters will be the least well constrained. Report their intervals prominently, and be prepared to fix `c, γ` from the literature if they will not identify.

---

## 12. Summary — the four claims this specification makes testable

| Claim | Parameter that tests it | Ablation that isolates it |
|---|---|---|
| Landslides raise downstream flood risk | `α_FL′` interval excludes 0 | A1 |
| Influence propagates downstream along the network, not radially | Network model beats isotropic in held-out log-likelihood | A2 |
| Influence *arrives* after a travel time rather than instantly | Shifted kernel beats unshifted; predicted arrival window matches observed hydrograph | A3 |
| Root-cause findings improve the cascade model | Full `μ` beats constant `μ` | A5 |

Every claim has a number attached and an experiment that could refute it. That is the difference between a good idea and a research contribution.
