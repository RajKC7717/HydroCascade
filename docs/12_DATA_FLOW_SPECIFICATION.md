# Document 12 — Data Flow Specification

**Owner: Chetan Mahajan** · Brief reference document. Read with Document 11.

---

## 1. Level 0 — context diagram

```
   ┌──────────────────┐         ┌──────────────────┐
   │ EXTERNAL DATA    │         │ USER             │
   │ PROVIDERS        │         │ (analyst /       │
   │ GSI · NASA · IMD │         │  disaster mgr /  │
   │ Copernicus · CWC │         │  reviewer)       │
   │ ESA · ESSD       │         │                  │
   └────────┬─────────┘         └────────▲─────────┘
            │ raw data                   │ risk table, map,
            │ (files, APIs)              │ attribution, alerts
            ▼                            │
   ┌─────────────────────────────────────┴─────────┐
   │                                               │
   │            H Y D R O C A S C A D E            │
   │                                               │
   │   root cause analysis  +  cascade model       │
   │                                               │
   └───────────────────────┬───────────────────────┘
                           │ figures, tables,
                           │ fitted parameters,
                           ▼ preprint
                    ┌──────────────┐
                    │ RESEARCH     │
                    │ OUTPUTS      │
                    └──────────────┘
```

---

## 2. Level 1 — the five processes

```
 [D1 raw store]              [D2 event catalog]         [D3 spatial framework]
      │                            │                            │
      ▼                            ▼                            ▼
┌───────────┐   events   ┌──────────────┐   units,   ┌──────────────────┐
│ P1        │───────────►│ P2           │───────────►│ P3               │
│ INGEST    │            │ SPATIAL      │  network,  │ ROOT CAUSE       │
│           │  rasters   │ FRAMEWORK    │  travel    │ ANALYSIS         │
│ download, │───────────►│              │  time      │                  │
│ clean,    │            │ slope units, │            │ ranking, SHAP,   │
│ decluster,│            │ network,     │            │ RF / XGBoost,    │
│ dedupe,   │            │ projection π,│            │ calibration      │
│ provenance│            │ covariates   │            │                  │
└───────────┘            └──────┬───────┘            └────────┬─────────┘
                                │                             │
                                │  covariates                 │ S_i(u), ẽ(u)
                                │                             │ [D4]
                                ▼                             ▼
                         ┌──────────────────────────────────────────┐
                         │ P4  CASCADE ENGINE                       │
                         │                                          │
                         │  build λ_i(u,t)                          │
                         │   = μ_i (from S_i, ẽ, rainfall)          │
                         │   + Σ triggering (network, travel time,  │
                         │                   mechanism mixture)     │
                         │                                          │
                         │  fit by MLE → [D5 fitted parameters]     │
                         └───────────────────┬──────────────────────┘
                                             │ λ, parameters,
                                             │ uncertainty
                                             ▼
                         ┌──────────────────────────────────────────┐
                         │ P5  VALIDATE, ALERT AND PRESENT          │
                         │                                          │
                         │  ablations · LOCO · backtest             │
                         │  λ → P → uncertainty band → alert level  │
                         │  exact additive attribution              │
                         │  dashboard, figures, report              │
                         └──────────────────────────────────────────┘
```

### Data stores

| ID | Store | Contents | Written by | Read by |
|---|---|---|---|---|
| **D1** | `data/raw/` | Untouched downloads + `PROVENANCE.md` | P1 | P1 only |
| **D2** | `data/catalog/` | `events.csv`, `cascade_pairs.csv` | P1 | P2, P3, P4, P5 |
| **D3** | `data/processed/spatial/` | Slope units, network, `π`, travel time, reachability, covariates | P2 | P3, P4, P5 |
| **D4** | `data/processed/rootcause/` | `S_L`, `S_F`, `ẽ`, driver ranking | P3 | P4, P5 |
| **D5** | `models/` + `experiments/register.csv` | Fitted parameters, intervals, fit records | P4 | P5 |

---

## 3. Level 2 — the cascade engine (P4)

```
                     [D2 events]        [D3 spatial]        [D4 S_i, ẽ]
                          │                  │                   │
      ┌───────────────────┼──────────────────┼───────────────────┤
      ▼                   ▼                  ▼                   ▼
┌───────────┐    ┌────────────────┐   ┌─────────────┐   ┌────────────────┐
│ P4.1      │    │ P4.2           │   │ P4.3        │   │ P4.4           │
│ BUILD     │    │ ACTIVE EVENT   │   │ CONNECT-    │   │ BACKGROUND     │
│ QUADRATURE│    │ SELECTION      │   │ IVITY       │   │                │
│ GRID      │    │                │   │             │   │ μ = μ⁰·S_i(u)  │
│           │    │ temporal       │   │ gate 1:     │   │   ·exp(Σθ·R_d) │
│ unit×day  │    │ truncation:    │   │  runout     │   │                │
│ reach×day │    │ drop terms     │   │  reaches    │   │ g_i = exp(η·ẽ) │
│           │    │ older than     │   │  a channel  │   │                │
│           │    │ 5 time consts  │   │ gate 2:     │   │                │
│           │    │                │   │  downstream │   │                │
└─────┬─────┘    └───────┬────────┘   └──────┬──────┘   └───────┬────────┘
      │                  │                    │                  │
      └──────────────────┴────────┬───────────┴──────────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │ P4.5  FLAT TENSOR ASSEMBLY   │
                    │                              │
                    │ all surviving (event, unit)  │
                    │ pairs → one tensor           │
                    │ columns: α_ij, m_k, g_i(u),  │
                    │   Ω^ψ, T_ref, Δt             │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ P4.6  KERNEL EVALUATION      │
                    │                              │
                    │ τ = T_ref / v₀               │
                    │ φ(Δt − τ)   ← the wave shift │
                    │ mechanism mixture w₁,w₂,w₃   │
                    │ single vectorised operation  │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ P4.7  INTENSITY ASSEMBLY     │
                    │  λ = μ + Σ contributions     │
                    │  (contributions retained     │
                    │   individually → exact       │
                    │   attribution downstream)    │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ P4.8  LOG-LIKELIHOOD         │
                    │  Σ log λ at events           │
                    │  − ∫∫ λ   (time analytic,    │
                    │            space on grid)    │
                    └──────────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ P4.9  OPTIMISE               │
                    │  Adam → L-BFGS, 20 restarts  │
                    │  constraints via softplus /  │
                    │  sigmoid / softmax           │
                    │  branching ratio check       │
                    └──────────────┬───────────────┘
                                   ▼
                             [D5 parameters]
```

**Note on P4.7:** individual contributions are kept, not summed away. This is what makes attribution in Document 02 §9 exact rather than estimated. It costs memory and is worth it.

---

## 4. The real-time loop

```
┌── HOURLY ────────────────────────────────────────────────────────┐
│                                                                  │
│  IMERG Early (T+4h)                                              │
│        │                                                         │
│        ▼                                                         │
│  regrid to units ──► update R1,R3,R7,R15,R30 and API             │
│        │                                                         │
│        ▼                                                         │
│  recompute μ_i  ──►  decay active excitation, drop expired       │
│        │                                                         │
│        ▼                                                         │
│  λ_i ──► P_i ──► uncertainty band ──► alert level                │
│        │                                                         │
│        ▼                                                         │
│  attribution ──► dashboard refresh ──► state snapshot            │
│                                                                  │
│  budget: < 3 min                                                 │
└──────────────────────────────────────────────────────────────────┘

┌── DAILY ─────────────────────────────────────────────────────────┐
│  GloFAS discharge ──► Q₂ exceedance + declustering               │
│                  ──► append confirmed flood events to history    │
│  IMD long-window rainfall refresh where available                │
└──────────────────────────────────────────────────────────────────┘

┌── EVENT-DRIVEN ──────────────────────────────────────────────────┐
│  new landslide reported / flood detected                         │
│        │                                                         │
│        ▼                                                         │
│  validate ──► append to history with mark                        │
│        │                                                         │
│        ▼                                                         │
│  look up affected set from the PRECOMPUTED reachability index    │
│    landslide → Down(π(x)) within max travel time                 │
│    flood     → downstream reaches + near-channel slope corridor  │
│        │                                                         │
│        ▼                                                         │
│  recompute λ for affected units ONLY   ← incremental             │
│        │                                                         │
│        ▼                                                         │
│  re-evaluate alerts; push if a level rises                       │
│                                                                  │
│  budget: < 2 s                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Why the incremental path is cheap:** the affected set is a lookup, not a computation, because reachability and reference travel times were precomputed in Phase 2. Cost is independent of catalog size.

---

## 5. Interface contracts between phases

These are the points where one phase's output becomes another's input. Each is a file with a fixed schema. If a contract changes, both sides must be updated and the change recorded.

| Contract | Producer | Consumer | Format | Breaks if |
|---|---|---|---|---|
| `events.csv` | P1 | P2, P3, P4 | CSV, schema in Doc 04 §2 | a column is renamed or dropped |
| `projection_table.parquet` | P2 | P4 | `unit_id → injection_reach_id, d_slope_km` | slope units are regenerated without regenerating this |
| `travel_time.npz` | P2 | P4 | sparse `(node, reach) → T_ref, area_ratio` | the network is regenerated without regenerating this |
| `S_landslide.tif`, `S_flood.parquet` | P3 | P4 | **calibrated** probability in (0,1) | calibration is skipped — `μ` is then wrong everywhere |
| `top_drivers.json` | P3 | P4 | **2–4** standardised feature names | more than 4 are supplied — identifiability fails |
| fit record JSON | P4 | P5 | parameters + intervals + seed + commit | a number is reported without one |

**The most fragile contract is `S_i`.** It is a *multiplier* in the background term, so an uncalibrated score silently corrupts every intensity in the system without producing any error. Document 06 §6 makes calibration a hard gate for this reason.

---

## 6. Error propagation

```
uncertainty in data
   └─► uncertainty in the event catalog (incompleteness, dedup, flood definition)
        └─► uncertainty in S_i (model error, calibration error)
             └─► uncertainty in μ
                  └─► uncertainty in fitted parameters (Laplace / profile / bootstrap)
                       └─► uncertainty in λ
                            └─► uncertainty in P  ──► reported as a 5–95% band
```

Not every source is quantified. What we quantify:
- parameter uncertainty (three methods, Document 08 §6)
- predictive uncertainty from parameter draws
- sensitivity to the six definitional choices (Document 08 §8)

What we do **not** quantify, and must say so in `LIMITATIONS.md`:
- inventory incompleteness below `A_min`
- error in the GloFAS discharge reanalysis itself
- DEM vertical error propagating into slope, flow routing and travel time

Listing what you did not quantify is part of quantifying uncertainty honestly.
