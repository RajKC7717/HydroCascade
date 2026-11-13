# Document 03 — Literature Review Plan

**Owner:** Ronit Dahiwal
**Target:** 58 papers across 8 themes, divided among 4 members
**Deadline:** all Tier-1 papers read and recorded by end of Week 4

---

## 1. How this review is organised

The earlier survey had 42 papers in 6 themes. Sharpening the model created three new requirements that the old list did not cover:

- point processes on **linear networks** (because floods now live on a river network)
- the **physical mechanisms** of landslide-to-flood transition (because the kernel now represents a named mechanism)
- **verification, calibration and alerting** for probabilistic forecasts (because we now report uncertainty and alert levels)

Themes G, H and I below are new. Themes A–F are carried forward.

### Priority tiers

| Tier | Meaning | Requirement |
|---|---|---|
| **T1** | Load-bearing. The project's design depends on it | Read in full. Write a half-page note |
| **T2** | Supporting. Cited for context or method | Read abstract, methods, and limitations |
| **T3** | Background. Cited in one line | Read abstract only |

---

## 2. Theme A — Foundational point-process mathematics

*Question: what are we extending, and does it work?*

| # | Paper | Tier | Where |
|---|---|---|---|
| A1 | Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika* 58(1):83–90 | T1 | JSTOR / university library |
| A2 | Ogata, Y. (1988). *JASA* 83(401):9–27; Ogata, Y. (1998). *Ann. Inst. Stat. Math.* 50(2):379–402 | T1 | The 1998 space-time paper is the direct ancestor. Library |
| A3 | Laub, Taimre & Pollett (2015). Hawkes processes. arXiv:1507.02822 | T1 | arXiv, free |
| A4 | Reinhart, A. (2018). A review of self-exciting spatio-temporal point processes. *Statistical Science* 33(3):299–318 | T1 | arXiv:1708.02647, free. **Read the residual-diagnostics section especially** |
| A5 | Lesage et al. (2022). Hawkes process with a gamma density excitation function, applied to natural disasters for insurance. *Methodol. Comput. Appl. Probab.* | T1 | **Our precedent for the gamma kernel component** |
| A6 | Bayesian estimation of the ETAS model. arXiv:2109.05894 | T2 | arXiv |
| A7 | Baddeley & Turner — the Berman–Turner device; point-process likelihood as weighted Poisson regression. See also Baddeley et al. (2010), *Electronic Journal of Statistics* 4:1151–1201, spatial logistic regression and change of support | T1 | **Our justification for quadrature-grid fitting** |
| A8 | Hawkes models and their applications. *Annual Review of Statistics and Its Application* (2025) | T2 | Library |

---

## 3. Theme B — Point processes on linear networks  ⭐ NEW

*Question: is our hybrid 2D + 1D spatial support legitimate?*

This theme did not exist in the earlier survey. It is now essential, because it is what makes the river-network formulation defensible instead of ad hoc.

| # | Paper | Tier | Why |
|---|---|---|---|
| B1 | Ang, Q. W., Baddeley, A. & Nair, G. (2012). Geometrically corrected second-order analysis of events on a linear network. *Scandinavian Journal of Statistics* 39(4):591–617 | T1 | **The foundational paper for point processes on networks.** Establishes that Euclidean intuition fails on networks |
| B2 | Baddeley, A., Jammalamadaka, A. & Nair, G. (2014). Multitype point process analysis of spines on the dendrite network of a neuron. *JRSS-C* 63(5):673–694 | T1 | **Multitype** point processes on a network — the closest structural precedent for what we are doing |
| B3 | Baddeley, A., Nair, G., Rakshit, S. & McSwiggan, G. (2017). "Stationary" point processes are uncommon on linear networks. *Stat* 6(1):68–78 | T1 | Warns that standard stationarity assumptions break on networks. We must engage with this |
| B4 | `spatstat.linnet` package documentation, CRAN | T2 | Reference implementation and vocabulary. Useful even though we implement in Python |
| B5 | Rakshit et al. (2021). Variable selection using penalised likelihoods for point patterns on a linear network. *ANZ Journal of Statistics* | T2 | Method for covariate selection on networks |
| B6 | Cox processes driven by transformed Gaussian processes on linear networks — a review. arXiv:2212.08402 | T3 | Context for alternative network process models |

**What we take from this theme:** the vocabulary (linear network, geometrically corrected), the warning about non-stationarity, and the citation base that makes our hybrid support a recognised construction rather than an invention.

---

## 4. Theme C — Physical mechanisms of the landslide → flood link  ⭐ NEW

*Question: what does our cross-hazard kernel actually represent?*

| # | Paper | Tier | Mechanism |
|---|---|---|---|
| C1 | Landslide–channel feedbacks amplify channel widening during floods. *npj Natural Hazards* (2025), doi 10.1038/s44304-025-00059-6. Case: 2013 Colorado flood | T1 | **Names all three mechanisms in one paper: bulking, damming, and long-term widening.** Our §4 is built on this |
| C2 | Peng, M., Ji, S., Sun, R., Zhang, L., Zhang, S. & Bai, Z. (2026). Assessing the probabilistic evolution of cascading hazards of landslide–river blockage–dam breaching–flood. *Reliability Engineering & System Safety* 269:112097 | T1 | Mechanism B in full, with a Bayesian network. Our closest cascade-model competitor |
| C3 | Channel aggradation triggered by dam failure amplifies the damage of outburst flood. *Landslides* (2023) | T1 | **Mechanism C.** Documents bed elevation rising 10–15 m and the resulting loss of channel capacity |
| C4 | Role of sediment entrainment in the flash-flood to debris-flow transition during cascading landslide dam failures. *Engineering Geology* (2026) | T2 | Flume and numerical work on how floods transform into debris flows |
| C5 | Mechanisms of river dam formation by debris flows, compared with dry avalanches. *Engineering Geology* (2025) | T2 | **Why some landslides dam a river and others do not** — momentum versus river flow. Justifies our connectivity gate |
| C6 | Arango-Carmona, M. I., Voit, P., Hürlimann, M., Aristizábal, E. & Korup, O. (2025). Hillslope torrential hazard cascades in tropical mountains. *NHESS* 25:3641–3663 | T1 | Database of 22 tropical cascades, 2009–2024. **Finds extreme rainfall alone often is not the cause**; sediment availability and antecedent wetness matter. Justifies our multi-window rainfall term |
| C7 | Debris flow and debris flood hazard assessment in mountain catchments. *Geomorphology* (2024) | T2 | Terminology and process distinctions we must get right |
| C8 | Physically based modelling of co-seismic landslide, debris flow and flood cascade. *NHESS* (2022), Wenchuan | T2 | The heavy-physics alternative to our approach |
| C9 | Probabilistic cascade modeling, La Liboriana River Basin. *Water* (2024) | T2 | Coupled physical models for one basin |
| C10 | Combining rainfall-induced shallow landslides and subsequent debris flows for hazard chain prediction. *NHESS* (2025), Cyclone Idai | T2 | Static exposure mapping along the chain |

**What we take from this theme:** the three named mechanisms, their timescales, and the physical justification for the connectivity gate and dilution term.

---

## 5. Theme D — Machine learning for single-hazard prediction (Part 1 method)

| # | Paper | Tier | Why |
|---|---|---|---|
| D1 | Flash flood susceptibility mapping using Random Forest regression and GIS. PMC (2024) | T2 | Standard causative factor list |
| D2 | Flood susceptibility assessment comparing three ML techniques. *Scientific Reports* (Feb 2026) | T2 | RF vs GB vs XGBoost head to head |
| D3 | Interpretable ML for urban flash flood hotspots using Geographically Weighted RF + SHAP. ScienceDirect (2024) | T1 | **Our SHAP methodology** |
| D4 | Predicting road flooding risk with crowdsourced reports. arXiv:2108.13265 | T2 | Finds precipitation intensity is the top factor. A sanity check |
| D5 | Achu, A. L. et al. (2023). ML-based landslide susceptibility modelling with emphasis on uncertainty analysis. *Geoscience Frontiers* 14:101657 | T1 | Kerala-specific; honest uncertainty treatment |
| D6 | ML and DL landslide susceptibility mapping in Wayanad, Kerala. ScienceDirect (2024) | T1 | **Its future-work section asks for exactly our project** |
| D7 | RF + RBF ensemble for regional flood risk, Yangtze Delta. PMC (2020) | T3 | How to include human and infrastructure factors |
| D8 | Lombardo, L. et al. Point-process and log-Gaussian Cox process approaches to landslide intensity. arXiv:1807.08513 | T1 | **Closest existing use of point processes for landslides.** Spatial only — we add time and triggering |

---

## 6. Theme E — Cascade and multi-hazard models, and the reviews naming the gap

| # | Paper | Tier | Why |
|---|---|---|---|
| E1 | Reichstein, M. et al. (2025). Early warning of complex climate risk with integrated artificial intelligence. *Nature Communications* 16:2564, doi 10.1038/s41467-025-57640-w | T1 | Perspective naming hazard interaction as a central challenge; argues for causal models |
| E2 | Harnessing ML for climate multi-hazard and multi-risk assessment. *NHESS / EGUsphere* review (2025) | T1 | States hazard interaction and spatiotemporal validation are top open gaps |
| E3 | Exploring ML potential for climate change risk assessment. ScienceDirect review | T2 | Current methods treat hazards singularly |
| E4 | AI in extreme weather prediction: systematic topic-model review 2015–2024. *Frontiers* (2025) | T3 | Landscape dominated by single-hazard work |
| E5 | Deep learning in remote sensing for climate-induced disaster resilience. *Springer* (2024) | T3 | No cross-hazard mechanism |
| E6 | Mundayatt, A. & Sreevalsan-Nair, J. (2026). FL-MHSM: spatially-adaptive fusion and ensemble learning for flood–landslide multi-hazard susceptibility mapping. arXiv:2604.16265 | T1 | **Direct competitor.** Kerala + Nepal, flood AUC-ROC ~0.905. Static maps, no time. Know it better than they do |
| E7 | Statistical dependence models for multi-hazard events: an application to the Danube Region | T1 | **Closest prior art to our cross-type claim.** Poisson-type processes, non-spatial. Cite it ourselves, first |
| E8 | Graph neural networks for seismic vulnerability of transportation networks. ScienceDirect (2026) | T3 | Names multi-hazard interaction as future work |
| E9 | Lightweight model for power grid cascading failures, graph physics-informed attention network. ScienceDirect (2025) | T2 | Methodological parallel: physics + learnable model, different domain |
| E10 | Hybrid learning-based fault prediction in multi-network energy systems. *Scientific Reports* (2025) | T3 | Another cross-domain cousin |

---

## 7. Theme F — Real cascade case studies and validation targets

| # | Paper | Tier | Role |
|---|---|---|---|
| F1 | Ekkirala, H. C. & Ramesh, M. V. (2026). Spatiotemporal assessment of multi-hazard risk using graph-based analysis for case studies in India. *Scientific Reports* 16:5837, doi 10.1038/s41598-026-35925-4 | T1 | **Anchor paper.** Sikkim GLOF + Wayanad. Rainfall and discharge thresholds, graph metrics, cascade depth. Calls for real-time threshold monitoring |
| F2 | Mundakkai–Chooralmala landslide: assessment of initiation, progression and impact. *Scientific Reports* (2025), doi 10.1038/s41598-025-07828-3 | T1 | **Validation ground truth.** Confirmed sequence, intensity–duration thresholds, bridge location |
| F3 | Preliminary analysis of the July 30, 2024 Wayanad landslide disaster. ScienceDirect (2025) | T1 | Independent timing and mechanism confirmation |
| F4 | Geological Survey of India (2024). First Information Report, Mundakkai–Chooralmala. Bhusanket portal | T1 | Government field record: source elevation, slope angles, overburden thickness, susceptibility zone, Punnapuzha flow path |
| F5 | Landslide hazard mapping of Wayanad District using copula-based joint probability of rainfall. *Proc. IAHS* 387:79 (2024) | T1 | **Precedent for adding time in our exact district**, using copulas rather than a point process |
| F6 | Hao, L. et al. (2020). Constructing a complete landslide inventory dataset for the 2018 monsoon disaster in Kerala. *ESSD* 12:2899–2911 | T1 | **Bulk training data** |
| F7 | Martha, T. R. et al. (2019). Landslides mapped using satellite data in the Western Ghats after August 2018. *Current Science* 117(5):804–812 | T1 | NRSC inventory, ~5,191 polygons |
| F8 | Martha, T. R. et al. (2021). Geospatial landslide inventory of India. *Landslides* 18:2125–2141 | T2 | National inventory context |
| F9 | The tale of three landslides in the Western Ghats. *Geoenvironmental Disasters* (2022) | T2 | Kokkayar, Plappally, Kavali 2021 — three of our test events, with different causes |
| F10 | Achu, A. L. et al. (2021). Preliminary analysis of the Pettimudi landslide, 6 August 2020. *Landslides* 18:1459–1463 | T2 | Another test event |
| F11 | Compound disaster from extreme rainfall and landslides, Sumatra 2025. ScienceDirect (2026) | T3 | Optional external backtest |
| F12 | Cascading hazards, Hongzhuangzi Gully, Gansu, August 2025. *Landslides* (2026) | T3 | Optional external backtest, four stages |
| F13 | Sharma, S. et al. (2023). Increasing risk of cascading hazards in the Central Himalayas. *Natural Hazards* | T3 | Regional motivation |

---

## 8. Theme G — Cross-domain evidence that this mathematics generalises

| # | Paper | Tier |
|---|---|---|
| G1 | Mohler, G. O. et al. (2011). Self-exciting point process modeling of crime. *JASA* 106(493):100–108 | T2 |
| G2 | Chiang, W.-H., Liu, X. & Mohler, G. (2022). Hawkes process modeling of COVID-19 with mobility leading indicators and spatial covariates. *International Journal of Forecasting* 38(2):505–520 | T1 — **closest methodological cousin: external covariates driving a Hawkes intensity** |
| G3 | Assessing the contagiousness of mass shootings with nonparametric Hawkes processes. arXiv:2009.08007 | T3 |
| G4 | Interpreting financial market crashes as earthquakes. ScienceDirect (2015) | T3 |

---

## 9. Theme H — Rainfall thresholds, verification, calibration and alerting  ⭐ NEW

*Question: how do we set alert levels and prove our probabilities mean something?*

| # | Paper | Tier | Why |
|---|---|---|---|
| H1 | Rainfall-induced landslide prediction models, Part I: empirical–statistical and physically based causative thresholds. *Water* 17(22):3273 (2025) | T1 | **Finds antecedent rainfall more robust than intensity–duration thresholds**, and documents the false-alarm problem with fixed I–D lines. Justifies our background term |
| H2 | Rainfall-induced landslide prediction models, Part II: deterministic physical and phenomenological models. *Bull. Eng. Geol. Environ.* (2024) | T2 | The companion review |
| H3 | Guzzetti et al., rainfall thresholds for landslide initiation (the standard I–D threshold literature) | T1 | **Our external baseline B2** must be built from this |
| H4 | Regional early warning model for rainfall-induced landslides based on slope units, Chongqing. *Engineering Geology* (2024) | T1 | Uses **slope units** and an effective antecedent rainfall index; directly relevant to our spatial units and background term |
| H5 | Forecast verification literature: Brier score decomposition, reliability diagrams, CRPS. Standard meteorological verification references (e.g. Jolliffe & Stephenson, *Forecast Verification*) | T1 | **How we prove our probabilities are calibrated** |
| H6 | Ogata (1988) residual analysis for point processes; time-rescaling theorem | T1 | Our diagnostic method |
| H7 | Bray et al., Voronoi residuals for spatial point process models | T2 | Spatial residual diagnostics |

---

## 10. Theme I — Data and product documentation (must be cited, not just used)

| # | Source | Tier | Cited for |
|---|---|---|---|
| I1 | Harrigan, S. et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979–present. *ESSD* 12:2043–2060 | T1 | Flood definition; **skill-by-catchment-size limits** |
| I2 | GloFAS threshold methodology: Gumbel fit to annual maxima by L-moments for 2/5/20-year return periods (CEMS documentation) | T1 | Our flood threshold procedure |
| I3 | Pai, D. S. et al. (2014). IMD 0.25° daily gridded rainfall. *MAUSAM* 65(1):1–18 | T1 | Rainfall data citation |
| I4 | Huffman et al., IMERG algorithm; NASA GPM documentation. **Early Run 4-h latency, Late 14-h, Final 3.5-month; 0.1°, half-hourly** | T1 | Real-time architecture latency budget |
| I5 | Tiwari, V. et al. (2020). Flood inundation mapping, Kerala 2018: SAR, Otsu thresholding and Google Earth Engine. *PLOS ONE*, doi 10.1371/journal.pone.0237324 | T1 | **Our SAR corroboration method**, ~94% reported accuracy in Kerala |
| I6 | Kirschbaum, D. & Stanley, T. (2018). Satellite-based assessment of rainfall-triggered landslide hazard. *Earth's Future* 6(3); and the Global Landslide Catalog documentation | T2 | COOLR coverage limits (~2007–2018) |
| I7 | HydroSHEDS documentation | T3 | Sub-basin boundaries |
| I8 | Copernicus DEM product handbook | T3 | Terrain source |
| I9 | ESA WorldCover product user manual | T3 | Land cover |
| I10 | SoilGrids documentation | T3 | Soil |
| I11 | ERA5-Land documentation, including **data latency** | T2 | Why ERA5-Land cannot be used in real time |

---

## 11. Assignment

| Member | Themes | Papers | T1 count |
|---|---|---|---|
| **Chetan Mahajan** | B (networks), I (data) | B1–B6, I1–I11 | 8 |
| **Yugant Chincholikar** | D (ML), H (thresholds & verification) | D1–D8, H1–H7 | 8 |
| **Raj Choudhari** | A (point processes), G (cross-domain) | A1–A8, G1–G4 | 6 |
| **Ronit Dahiwal** | C (mechanisms), E (gap), F (cases) | C1–C10, E1–E10, F1–F13 | 13 |

Ronit's load is heaviest; Themes E and F overlap heavily with the report writing he already owns, so much of that reading is reused directly.

**Cross-reading requirement.** Every member also reads these four, regardless of assignment, because the whole project rests on them:

- **F1** — Ekkirala & Ramesh 2026, the anchor paper
- **C1** — the three mechanisms
- **B2** — multitype point processes on a network
- **E6** — FL-MHSM, the direct competitor

---

## 12. Where to find papers, in order of what to try

1. **Open access first.** arXiv, NHESS and ESSD (Copernicus, fully open), *Scientific Reports* and *Nature Communications* (open), PMC, PLOS ONE, *Current Science* (free PDF).
2. **Government portals.** GSI Bhukosh and Bhusanket publish reports and data free.
3. **Your institute library / DELNET / N-LIST access** for Elsevier, Springer and Wiley content.
4. **Author preprints.** Check the author's institutional page or ResearchGate before assuming a paper is inaccessible.
5. **Email the author.** A short, specific message from an undergraduate research group is answered more often than you would expect. Say what you are building in two sentences and ask for a copy.

Never use a paper you have not opened. Never cite from an abstract alone if it is Tier 1.

---

## 13. Recording protocol — one file per paper

Create `docs/literature/<theme><number>_<shortname>.md`. Every note has exactly these fields:

```markdown
# A4 — Reinhart 2018, Review of self-exciting spatio-temporal point processes

Full citation:  Reinhart, A. (2018). Statistical Science 33(3):299-318. arXiv:1708.02647
Accessed:       2026-09-14   |   Read by: Raj   |   Tier: 1
Verified exists: YES  (opened PDF, DOI resolves)

## What it does (3 sentences max)

## Method

## What we take from it
(concrete: an equation, a diagnostic, a threshold, a dataset, a phrase for the paper)

## Its stated limitation, in the authors' own words (paraphrased)

## Where it appears in our report
(section number)
```

**The `Verified exists` field is mandatory.** In the earlier document set, two references had incomplete or wrong metadata. A reference list containing a paper that does not exist is the single most damaging thing that can appear in a project report. Every citation must have been opened by a named person on a named date.

---

## 14. Search strategy for finding anything we have missed

Run these queries once in Week 2 and again in Week 10, on Google Scholar and on arXiv listings for `stat.AP`, `physics.geo-ph` and `cs.LG`:

```
"Hawkes process" AND (landslide OR flood OR "debris flow")
"point process" AND "linear network" AND (river OR stream OR hydrology)
"multi-hazard" AND ("triggering" OR "cascading") AND ("point process" OR "self-exciting")
"cross-hazard" AND (excitation OR coupling OR triggering)
"landslide" AND "flood" AND (Kerala OR "Western Ghats") AND (susceptibility OR cascade)
"early warning" AND "lead time" AND "false alarm" AND landslide
"travel time" AND "flood wave" AND "landslide" AND downstream
"branching ratio" AND "natural hazard"
```

Set up **Google Scholar alerts** for the first four. This is a fast-moving area — a directly related Kerala flood–landslide paper appeared in April 2026, after our first survey was written. Set the alerts in Week 2, not in Week 12.

Also set alerts on the two most relevant author groups: **M. V. Ramesh (Amrita Vishwa Vidyapeetham)** and **A. L. Achu / Kerala landslide susceptibility authors**.

---

## 15. Deliverables from this document

| Deliverable | Due | Owner |
|---|---|---|
| All T1 notes written and committed | end Week 4 | each member for their theme |
| All T2 notes | end Week 6 | as assigned |
| Verified reference list, every entry opened | end Week 6 | Ronit compiles, all verify |
| Related-work section draft, 1,200 words | end Week 8 | Ronit |
| Gap statement paragraph, final | end Week 8 | Ronit |
| Repeat literature search, second pass | Week 10 | Ronit |
