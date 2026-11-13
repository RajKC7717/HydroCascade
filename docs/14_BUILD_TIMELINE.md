# Document 14 — Build Timeline: how long it takes, and what is left

**Written 21 September 2026, 11:40, while the Phase 5 run was in progress.**
Every duration below marked *measured* was read from a log line of the form
`stage 'x' done in N s`. Durations marked *estimated* are extrapolated from
measured per-step costs and say so.

Machine: Windows 11 laptop, 7.2 GB RAM, CPU only, no GPU.

---

## 1. The short answer

| Question | Answer |
|---|---|
| **When does the current run finish?** | about **12:15 today** (±10 min) |
| **What is left after that?** | about **15 minutes** of my work: gate check, full test suite, updating the docs with the final numbers |
| **A clean build from an empty `data/` folder** | about **1 h 15 min** at the settings used here |
| **A publishable run** (Doc 07's full fitting protocol) | about **8 hours overnight** for the headline model and the key ablations; ~1.5 days for every model at full protocol |
| **Work only a human can do** | accounts, data downloads, one reproduction — none of it compute-bound; see §5 |

---

## 2. Where the current run is

The pipeline runs as separate processes so the operating system reclaims memory
between phases. The previous attempt was killed by Claude Code's memory reaper
with 0.2 GB free; this run adds a cache that stops the validation dataset being
rebuilt for every fit.

| Step | Status | Started | Duration |
|---|---|---|---|
| Phase 4 — stages S3, S4, S5 | ✅ **done** | 11:27:32 | **5 m 49 s** *measured* |
| Phase 5 — nested ladder M0–M5 | ✅ **done** | 11:34:04 | **5 m 33 s** *measured* |
| Phase 5 — ablations FULL + A1–A9 (10 fits) | ⏳ running | 11:39:37 | ~13 min *estimated* |
| Phase 5 — profile likelihood, `α_LF` and `ψ` | waiting | | ~13 min *estimated* |
| Phase 5 — LOCO, Wayanad backtest, money figure, negative controls | waiting | | ~2 min *estimated* |
| Phase 5 — sensitivity, four variants | waiting | | ~6 min *estimated* |
| Phase 5 — reports and figures | waiting | | < 1 min |
| Phase 6 — alert thresholds, architecture report | waiting | | ~1 min *estimated* |
| **Expected finish** | | | **~12:15** |

How the estimates are built: an S5-type fit at 3 restarts × 160 steps measured
**93 s**; S3/S4-type fits measured **~45 s**. Nine of the ten ablations are
S5-type. A profile grid point is one 80-step re-optimisation, about half a fit.

---

## 3. A clean build, stage by stage

From an empty `data/` directory to every report and figure, at the settings this
build used (3 restarts × 160 steps for the cascade fits).

### Phases 1–2 — about 15 minutes, dominated by two downloads

| Stage | Time | What dominates |
|---|---|---|
| `aoi` | 9.7 s *measured* | reading HydroBASINS |
| `dem` | 2 m 23 s *measured* | **downloading** two Copernicus tiles |
| `flow` | 2 m 21 s – 3 m 18 s *measured* | depression filling and D8 over 9.07 M cells |
| `network` | 8.4 s *measured* | |
| `units` | 35.4 s *measured* | vectorising 17,138 slope units |
| `projection` | 5.2 s *measured* | |
| `traveltime` | 2.3 s *measured* | 716,519 pairs — Doc 05 budgets 4 hours for this |
| `rainfall` | 7 m 12 s *measured* | **downloading** seven years of global CHIRPS |
| `catalog` | 21 s *measured* | |
| `covariates` | ~10 s *measured* | |
| `figures` | 22 s *measured* | |
| `report` | 43 s *measured* | synthesising discharge five times for the sensitivity table |
| **Subtotal** | **~15 min** | about 9½ minutes of it is network I/O |

### Phases 3–6 — about an hour

| Stage | Time | Notes |
|---|---|---|
| `rootcause` | **8 m 8 s** *measured* | One later run took ~100 min on the same code. The machine was swapping for memory; the 1:20 flood sampling run builds a 189,105-row matrix. With ~2 GB free, expect 8–15 min |
| `cascade --verify` | **5 m 6 s** *measured* | S1 and S2 parameter recovery, 36 simulated catalogs |
| `cascade` stages | **5 m 49 s** *measured* | S3, S4, S5 |
| `experiments` | **~40 min** *estimated* | confirmed against the live run in §2 |
| `realtime` | ~1 min *estimated* | |
| `check` | 36 s *measured* | all six phases |
| **Subtotal** | **~1 h** | |

### Total

| | |
|---|---|
| **Compute, empty folder to every output** | **≈ 1 h 15 min** |
| Plus the test suite | a few minutes |
| If the machine runs short of memory | up to **~3 h**, almost all of it in `rootcause` swapping |

**Free memory matters more than CPU here.** The pipeline was killed outright
twice at 0.2 GB free. Close editors and browsers before a clean build; about 2 GB
free is comfortable.

---

## 4. The publishable run is a different order of magnitude

The settings above are enough to prove every stage works. They are **not**
enough to quote. Doc 07 §7 specifies the fitting protocol for any reported
number: Adam for **2,000 steps** then L-BFGS, with **20 random restarts**. This
build used 160 steps and 3 restarts.

Fit cost scales with restarts × steps. From the measured 93 s for an S5 fit at
3 × 160:

| What | Estimated time | Against Doc 07 §8's target |
|---|---|---|
| One likelihood + gradient | **~0.15 s** *measured* | target < 60 s — **met by 400×** |
| One S5 fit, single start, 2,000 steps | **~6–7 min** | target < 2 h — **met** |
| One S5 fit, 20 restarts | **~2 h** | target "overnight" — **met** |
| Stages S3 + S4 + S5 at full protocol | **~4 h** | |
| Every ladder and ablation model at full protocol (16 models) | **~1.5 days** | beyond overnight |

**Recommendation:** run the full protocol on the headline model and the three
ablations that carry the paper's claims — **A1** (cross terms), **A2** (the
network) and **A3** (the travel-time wave). That is about **8 hours**, one night.
Run the remaining ablations at a reduced setting and say so in the report.

There is no reason to spend that night **yet**. Every number the full protocol
would produce is currently a number about the synthetic stand-ins, and more
optimiser steps do not make a stand-in more real. Run it once the real
landslide inventory and GloFAS discharge are in.

---

## 5. How much is left

### Compute that is left — about 50 minutes, all automatic

| Item | Time | Who |
|---|---|---|
| Finish the Phase 5 and Phase 6 run in progress | ~35 min | running now |
| `hydrocascade check` — final gate counts, all six phases | < 1 min | me |
| Full test suite | a few min | me |
| Update `README.md`, `docs/13`, and the two artefacts with the final numbers | ~10 min | me |
| *Optional:* re-run `rootcause` so `phase3_baselines.md` picks up the new explanation of the near-zero spatial-CV gap | 8–15 min | me, if wanted |

### Build status by phase

| Phase | Code | Run on current data | Blocked on |
|---|---|---|---|
| 1 — catalog | ✅ complete | ✅ | real landslide inventory; Earth Engine for SAR precision |
| 2 — spatial framework | ✅ complete | ✅ | IMD or IMERG for the rainfall gate |
| 3 — root cause | ✅ complete | ✅ | real inventory, so drivers describe Kerala rather than the generator |
| 4 — cascade model | ✅ complete | ✅ | GloFAS, so the travel-time wave has something to find |
| 5 — experiments | ✅ complete | ⏳ ~35 min | second-region DEM for the transfer test; sourced coordinates for LOCO |
| 6 — alerts and delivery | ✅ complete | ⏳ after Phase 5 | nothing further in code |

**The code is finished.** What remains is compute that is already running, and
data that only a person can fetch.

### Work only a person can do — none of it is compute

These do not shorten with a faster machine. They are forms, downloads and
approvals, and they are what turns the rehearsal numbers into results.

| Task | Effort | Waiting time | What it unblocks |
|---|---|---|---|
| Download the ESSD and NRSC landslide inventories into `data/raw/` | an afternoon | none | the volume gate, and every driver and parameter becoming real |
| Register for the Copernicus CDS | 5 minutes | minutes to a day | GloFAS discharge; the S5 test; ablation A3 |
| Request IMD gridded rainfall | 10 minutes | days to a week | the rainfall gate, and **the backtest's lead time** |
| Register for NASA Earthdata | 10 minutes | minutes | IMERG rainfall |
| Register for Google Earth Engine | 10 minutes | a day or two | SAR precision, requirement FR-05 |
| Re-source the documented-floods list from KSDMA / EMS / CWC | an afternoon | none | a publishable recall number |
| `git init` and first commit | 1 minute | none | every fit record traceable to a commit |
| A teammate reproduces the build from a clean clone | ~1 h 15 min of compute | none | NFR-07 |

**Start the registrations today, even though they pay off last.** IMD in
particular is an email exchange, not an instant key, and it is the dataset that
decides whether the Wayanad backtest can show a lead time at all.

### After the data arrives

Once the real data is in, the complete sequence is a clean build plus the
publishable fits:

| | |
|---|---|
| Clean rebuild of every stage on real data | ~1 h 15 min |
| Publishable fits for the headline model and A1–A3 | ~8 h, overnight |
| **Total, data-in to paper numbers** | **about one working day plus one night** |

**Correction (2026-09-21).** An earlier version of this page said switching to
real data needed no code change. That was wrong for three datasets: the
`imd`/`imerg` switches were read by nothing, and `glofas: real` stopped at a
deliberate `NotImplementedError`. All three now have working readers
(`ingest/rainfall_sources.py`, `ingest/adapters.py`), with tests. The human
part is now: get the files, put them in the right folder, change one line,
re-run. `docs/15_HOW_TO_RUN.md` gives every step.
