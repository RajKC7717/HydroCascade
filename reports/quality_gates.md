# Quality gates — all six phases

Region `chaliyar_pilot` · period 2018-06-01 to 2024-12-31 · analysis CRS EPSG:32643

Gates are from Doc 04 §6, Doc 05 §9, Doc 06 §8, Doc 07 §10, Doc 08 and Doc 09. `BLOCKED` means the gate could not be evaluated because an input is missing or synthetic — it is never counted as a pass.

| Gate | Status | Detail |
|---|---|---|
| P1.1 volume >= 200 landslides | **BLOCKED** | 760 landslide rows but 9764 of 9765 catalog rows are SYNTHETIC. Doc 04 §6 gate 1 counts real events only. |
| P1.2 provenance on 100% of rows | **PASS** | all 9765 rows carry source, source_url and accessed_by |
| P1.3 seasonality follows the monsoon | **PASS** | 96% of landslides in Jun-Sep |
| P1.4 size-frequency rollover identified | **PASS** | A_min = 863 m2, alpha = 2.02 +/- 0.04 (estimated but NOT applied — set catalog.completeness.min_area_m2) |
| P1.5 no duplicates within 500 m / 1 day | **PASS** | 0 among hazard type(s) L; floods are declustered instead (see catalog.dedup.hazard_types) |
| P1.6 flood definition corroborated (FR-05) | **BLOCKED** | precision not measured: The Sentinel-1 precision test (Doc 02 §3.3 step 4) needs Google Earth Engine. Recall harness runs (4/6 matched) but is not publishable. See reports/phase1_flood_definition.md §5. |
| P2.1 every slope unit has an injection reach | **PASS** | 0 nulls across 17,138 units |
| P2.2 Mundakkai -> Chooralmala -> Nilambur | **PASS** | R-02503 -> chooralmala_bridge at 0.8 h, nilambur at 12.0 h |
| P2.3 every event joins to one spatial unit | **PASS** | all 760 landslide rows joined; 9,005 flood rows carry a reach id |
| P2.4 Wayanad rainfall totals reproduced | **FAIL** | CHIRPS V2.0 p25 gives 71 mm vs published 577 mm (12%). Not an extraction bug: the 29-31 Jul band peaks ~50 km west of the crown, the known orographic displacement at 0.25 deg. Needs IMD (M1) or IMERG (M2) — see docs/15_HOW_TO_RUN.md. |
| P2.5 all layers in EPSG:32643 | **PASS** | rasters and vectors all aligned |
| P2.6 travel-time table built and cached | **PASS** | 4.9 MB, 716,519 (source, downstream) pairs, median celerity 1.94 m/s |
| P3.4 S_L(u) and S_F(s) exported as calibrated tables | **PASS** | S_landslide.parquet, S_flood.parquet |
| P3.5 e~(u) chosen, between 2 and 4 drivers | **PASS** | 4: R7, slope_max_deg, twi, R3 |
| P3.1 driver ranking reported | **PASS** | reports/phase3_drivers.md |
| P3.3 calibration reported | **PASS** | reports/phase3_calibration.md |
| P3.6 B1-B3 built and evaluated | **PASS** | reports/phase3_baselines.md |
| P4.1 parameter recovery verified at S1 and S2 | **PASS** | S1 mu 0.5076 vs 0.5 (+1.5%, +0.5 SE); alpha 0.3825 vs 0.4 (-4.4%, -1.8 SE; S2 mu 0.005221 vs 0.005 (+4.4%, +1.2 SE); alpha 0.3037 vs 0.3 (+1.2%, +0. |
| P4.2 independent cross-check of the likelihood | **BLOCKED** | 434 events; direct -369.806780, recursive -369.806780, torch -369.806780; max |d — SUBSTITUTE for the Doc 07 §2 `tick` check, which does not install here. Three implementations by one author is weaker than an independent library, so this is not recorded as a pass. |
| P4.3 branching ratio < 1 at every stage | **PASS** | all 3 recorded fits are stationary |
| P4.6 landslide passes the fitted runout gate | **PASS** | passes in S4_S4[notravel_time,nomixture], S5_S5[full] |
| P4.7 fitted v0 gives a plausible channel celerity | **FAIL** | outside 1-5 m/s in S5_S5[full] |
| P4.8 every number traceable to a saved fit record | **PASS** | 3 fit records in models/fits/, stage report written |
| P5.0 experiment register populated | **PASS** | 67 runs recorded |
| P5.1 ablation table A1-A9 produced | **PASS** | reports/phase5_ablations.md |
| P5.2 baselines B1-B5 vs M5 reported | **PASS** | reports/phase5_baselines.md |
| P5.3 leave-one-cascade-out run | **PASS** | reports/phase5_loco.md |
| P5.4 sensitivity analyses reported | **PASS** | reports/phase5_sensitivity.md |
| P5.5 Wayanad backtest reported | **PASS** | reports/phase5_backtest.md |
| P5.6 money figure produced | **PASS** | figures/money_figure.png |
| P5.7 spatial transfer to a second region | **BLOCKED** | the second region's DEM, network and slope units are not built (roughly 1.5-2.5 GB of disk), and none of the three Maharashtra events has sourced coordinates. Both are one afternoon of work; neither is a code change. |
| P6.1 real-time architecture documented | **PASS** | reports/phase6_architecture.md |
| P6.2 alert thresholds optimised and reported | **PASS** | reports/phase6_alerts.md |
| P6.3 lead-time versus false-alarm curve produced | **PASS** | figures/leadtime_vs_falsealarm.png |
| P6.4 replay dashboard exists | **PASS** | `hydrocascade app` — replay mode only; Doc 09 §1.1 is explicit that no 24/7 production service is delivered |

**29 passed, 2 failed, 4 blocked.**

> **Synthetic stand-ins active:** glofas, landslides. No number derived from these may be reported. See `config/default.yaml` -> `data_mode`.
