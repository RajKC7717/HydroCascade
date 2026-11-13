# 15 — How to run it, what data to add, and where the results come out

Everything a human has to do by hand, in order. Nothing here needs you to write
code: you download files, put them in a named folder, change at most one line of
config, and run a command.

**The short version**

| Question | Answer |
|---|---|
| Can I run it right now, with no accounts and no downloads? | **Yes.** Every gated dataset has a labelled stand-in. Nothing is blocked. |
| Do I have to train any model by hand? | **No.** Training happens inside `hydrocascade rootcause`, `cascade` and `experiments`. There is no separate training script and no notebook to babysit. |
| What must I add before a number is publishable? | A **real landslide inventory** (§3.1). Everything else improves the result; without this one, every fitted number is exercising the pipeline on synthetic events. |
| Where do I look at the results? | `hydrocascade app` — nine tabs in a browser (§6). Files land in `reports/`, `figures/`, `models/` and `experiments/`. |
| How long does a full run take? | About 1 h 15 min clean from an empty repo. See [14_BUILD_TIMELINE.md](14_BUILD_TIMELINE.md). |

---

## 1. One-time setup

From the repo root (`C:\Users\choud\Desktop\TY\edisem1`):

```bash
# the virtualenv already exists in this checkout; if you start fresh:
python -m venv .venv
.venv\Scripts\activate                 # PowerShell / cmd
pip install -e .

hydrocascade --help                    # should list the stages
pytest -q                              # 147 tests, all should pass
```

`hydrocascade` and `python -m hydrocascade` are the same thing. Every command
takes `--config`, `--force` (recompute even if cached) and `-v`.

---

## 2. Run everything, as it stands

In this order. Each stage writes to disk and the next one reads it, so you can
stop and resume anywhere.

```bash
hydrocascade aoi          # study boundary from HydroBASINS          ~10 s
hydrocascade dem          # Copernicus DEM, mosaicked and reprojected ~2.5 min
hydrocascade flow         # sink fill, D8 direction, accumulation     ~3 min
hydrocascade network      # the channel network N                     ~10 s
hydrocascade units        # the slope units H                         ~35 s
hydrocascade projection   # pi: H -> N                                 ~5 s
hydrocascade traveltime   # downstream sets, T_ref, area ratios        ~3 s
hydrocascade rainfall     # rainfall windows R1..R30 and the API      ~7 min
hydrocascade catalog      # data/catalog/events.csv                   ~20 s
hydrocascade covariates   # static terrain covariates                 ~10 s
hydrocascade figures      # every Phase 1-2 figure                    ~25 s
hydrocascade report       # the flood-definition prose                ~45 s

hydrocascade rootcause    # Phase 3: drivers, SHAP, susceptibility    ~8 min
hydrocascade cascade      # Phase 4: fits S3, S4, S5                  ~6 min
hydrocascade experiments  # Phase 5: ladder, ablations, backtest     ~35 min
hydrocascade realtime     # Phase 6: alert thresholds, trade-off       ~1 min

hydrocascade check        # all six phases' quality gates            ~35 s
hydrocascade app          # the browser UI
```

On a machine with ~7 GB of RAM, run `cascade`, `experiments` and `realtime` one
at a time rather than together — `scripts/run_phases_4_to_6.sh` does exactly
that, one OS process per phase.

---

## 3. The data you may want to add

Each subsection is self-contained: where to get it, what to select, which folder
it goes in, what to change, what to re-run.

**Two rules that apply to every download** (Doc 10 §8):

1. Every raw folder carries a `PROVENANCE.md`, written when you download, not
   later. The readers **refuse** a folder without one.
2. Never edit anything under `data/raw/`, and never open a catalog CSV in Excel
   — it silently rewrites dates and coordinates.

The template — the three capitalised keys are parsed, the rest is for humans:

```markdown
# <dataset name>

URL: <the exact page or query you downloaded from>
By: <your name or email>
Downloaded: 2026-09-24
Licence: <e.g. CC-BY 4.0, Copernicus licence>
Notes: <selections you made: variables, date range, bbox, version>
```

### 3.1 Landslide inventory — the one that matters most

Without this, `data/catalog/events.csv` is filled by a **synthetic** inventory:
the pipeline runs, the gates report `BLOCKED`, and no fitted number means
anything about the real world.

Where to get it (all free):

| Source | What it gives | Access |
|---|---|---|
| **Kerala 2018 monsoon inventory** (Hao et al., *ESSD* 12, 2899–2918, 2020) | ~4,700 mapped polygons for the August 2018 event | open download, no login |
| **NASA COOLR / Global Landslide Catalog** | dated points, global, sparse in Kerala | open |
| **GSI Bhukosh** (`bhukosh.gsi.gov.in`) | Indian landslide inventory, polygons | free registration |
| **NRSC / ISRO Wayanad 2024 mapping** | the 30 July 2024 event itself | free registration |

What to do:

1. Make a folder named after the source, then a dated subfolder:
   `data/raw/ESSD_KERALA_2018/2026-09-24/`
2. Put the `.csv`, `.shp` (with its `.dbf`/`.shx`/`.prj`), `.gpkg` or
   `.geojson` in it, plus `PROVENANCE.md`.
3. Check it has **latitude, longitude and a date**. The reader finds columns
   named any of `lat/latitude/y/crown_lat`, `lon/long/longitude/x/crown_lon`,
   `date/event_date/occurrence_date/ls_date`, and area from
   `area_m2/area/area_sqm/shape_area`. If the inventory has no date column —
   common for a single-event inventory mapped from imagery — add one holding
   that event's date, and say so in `PROVENANCE.md`. Events with no date cannot
   be placed in time and are dropped.
4. Polygons are reduced to the **crown point** (highest elevation in the
   polygon), not the centroid — Doc 04 §3.2. Nothing for you to do; just know
   that is what the map will show.

Then:

```bash
hydrocascade catalog --force
hydrocascade covariates --force
hydrocascade rootcause && hydrocascade cascade && hydrocascade experiments && hydrocascade realtime
hydrocascade check
```

Only events inside the study period (**2018-06-01 → 2024-12-31**) and the study
bbox (75.55–76.65 E, 10.90–11.85 N) are kept; the rest are counted and logged.

### 3.2 GloFAS river discharge — turns flood events real

Flood events are currently detected from a **synthetic** discharge series.

**Option A — download the files yourself (no key needed in the repo).**

1. Go to `https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical`
   (GloFAS moved from CDS to EWDS). Free account.
2. Select: system version **4.0**, model **LISFLOOD**, product type
   **consolidated**, variable **river discharge in the last 24 hours**, the
   years you want (1980–2024 gives the Gumbel fit a long record), all months,
   all days, format **NetCDF**, and sub-area **N 12.0, W 75.0, S 10.4, E 77.2**.
3. Also grab the **upstream area** auxiliary file from the same dataset page if
   it is offered. With it, each reach is matched to the GloFAS cell whose
   catchment area is closest to its own; without it the fallback is the wettest
   cell nearby, which is biased to the main stem and is logged as such.
4. Put everything in `data/raw/GLOFAS/2026-09-24/` with a `PROVENANCE.md`.

**Option B — let it download.** Create `~/.cdsapirc`:

```
url: https://ewds.climate.copernicus.eu/api
key: <your EWDS key>
```

then `pip install cdsapi`. With no files present, the adapter downloads them
year by year into `data/raw/GLOFAS/<today>/` and writes the `PROVENANCE.md`
itself. The CDS queue is the slow part; already-downloaded years are skipped, so
an interrupted run resumes.

Either way, flip the switch in `config/default.yaml`:

```yaml
data_mode:
  glofas: real
```

then `hydrocascade catalog --force` and the Phase 3–6 commands again.

### 3.3 Rainfall: IMD or IMERG instead of CHIRPS

This is the fix for the project's most conspicuous result. CHIRPS sees **71 mm**
of the published **577 mm** that fell at Mundakkai in the 48 h before the
failure — gate P2.4 fails on it, and it is *why the Wayanad backtest shows no
lead time*: the model's background rises with rainfall, and CHIRPS has no
rainfall to rise on. IMD is gauge-based; IMERG is 0.1°. Either should see it.

**IMD 0.25° gridded rainfall** (free, no login):

1. `https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html`
2. Download the NetCDF file for each year 2018–2024
   (e.g. `RF25_ind2024_rfp25.nc`).
3. Put them in `data/raw/IMD/2026-09-24/` with a `PROVENANCE.md`.

**GPM IMERG daily** (free NASA Earthdata account):

1. `https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary`
2. Use *Subset / Get Data* → region **75.0, 10.4, 77.2, 12.0**, date range
   2018-06-01 to 2024-12-31, variable `precipitation`, format **netCDF**.
3. Put the files in `data/raw/IMERG/2026-09-24/` with a `PROVENANCE.md`.

Then in `config/default.yaml`:

```yaml
rainfall:
  primary: imd        # or: imerg, or: chirps
```

and

```bash
hydrocascade rainfall --force
hydrocascade rootcause && hydrocascade cascade && hydrocascade experiments && hydrocascade realtime
hydrocascade check      # P2.4 should now pass, and the backtest gets a real test
```

The reader handles the variations these products actually ship: upper-case
`TIME/LATITUDE/LONGITUDE`, `(time, lon, lat)` axis order, mm/hr or metres
instead of mm/day, and −999 / −9999.9 missing markers.

### 3.4 Optional extras

| What | Why | How |
|---|---|---|
| `data/reference/documented_floods.csv` | a **recall** number that isn't circular — scored against floods documented independently of the detector | fill the existing file; columns are listed in `ingest/corroboration.py` (`flood_id, date_start, date_end, place, district, lat, lon, in_pilot_basin, evidence_type, source, source_url, source_date_accessed, accessed_by, notes`). Use a closed date interval rather than inventing a precise day. |
| Sentinel-1 via Google Earth Engine | gate P1.6 / FR-05: scoring the flood definition against observed inundation | **the one piece still not wired.** `hydrocascade report` prints the exact GEE recipe; the gate reports `BLOCKED` until someone runs it. Deliberate: a fabricated corroboration statistic would look like evidence. |
| ERA5-Land soil moisture | nothing on the critical path | not used. Doc 09 §1.3: it has multi-day latency and cannot be used in real time, so the rainfall-derived API index replaces it everywhere. `data_mode.era5_land: unused`. |

---

## 4. Training: what is trained, by what, and where it lands

There is **no manual training step**. Three commands train everything:

| Command | What it trains | Where it lands |
|---|---|---|
| `hydrocascade rootcause` | Random Forest, XGBoost and logistic susceptibility models, with spatial block cross-validation by sub-basin and cross-fitted isotonic calibration; SHAP and permutation rankings; baselines B1–B3 | `models/susceptibility_landslide.pkl`, `models/susceptibility_flood.pkl`, `data/processed/S_*.parquet`, `reports/phase3_*.md` |
| `hydrocascade cascade` | the cross-hazard Hawkes model, stages S3 → S4 → S5, each by multi-restart Adam then L-BFGS | `models/fits/*.json`, `reports/phase4_stages.md` |
| `hydrocascade experiments` | refits the nested ladder M0–M5 and ablations A1–A9, profile likelihoods, LOCO, the Wayanad backtest | `experiments/register.csv`, `reports/phase5_*.md`, `figures/money_figure.png` |
| `hydrocascade realtime` | alert thresholds per unit and the cost-ratio trade-off | `reports/phase6_*.md`, `figures/leadtime_vs_falsealarm.png` |

Fit quality is a speed/quality dial, not a code change:

```bash
hydrocascade cascade --restarts 3  --steps 160    # ~6 min   — the development setting
hydrocascade cascade --restarts 8  --steps 400    # ~30 min  — the default
hydrocascade cascade --restarts 20 --steps 2000   # ~2 h/fit — the publishable setting, run overnight
```

`hydrocascade cascade --verify` runs the simulate-and-recover check (S1/S2
parameter recovery plus three independent likelihood implementations agreeing)
without refitting anything.

---

## 5. Testing what you have

```bash
pytest -q                     # 147 unit tests, seconds
hydrocascade cascade --verify # the model recovers parameters it generated
hydrocascade check            # the quality gates for all six phases
hydrocascade check --phase 4  # just one phase
```

Read `hydrocascade check` as three states, not two:

* **PASS** — checked and satisfied.
* **FAIL** — checked and violated. Fix it.
* **BLOCKED** — *could not be checked*, because an input is missing or is a
  stand-in. Never count it as a pass. Most BLOCKEDs clear when §3.1 and §3.2 do.

The report is written to `reports/quality_gates.md`.

---

## 6. Where the results are

**Start here:** `hydrocascade app` opens the Explorer in a browser (nine tabs —
cascade path, event catalog, spatial framework, rainfall, drivers &
susceptibility, the model, backtest & alerts, gates & reports, how to
reproduce). A red banner names every synthetic stand-in currently active, so a
screenshot can never quietly present synthetic numbers as real.

Files, if you want them directly:

| Path | What |
|---|---|
| `reports/phase1_flood_definition.md` | the flood definition, its sensitivity, its blind spot |
| `reports/phase3_drivers.md`, `phase3_features.md`, `phase3_calibration.md`, `phase3_baselines.md` | driver ranking, feature selection, calibration, baselines B1–B3 |
| `reports/phase4_stages.md` | S3 → S4 → S5, held-out log-likelihood, branching ratio, the runout-gate check |
| `reports/phase5_ablations.md` | the nested ladder and the nine ablations — the scientific core |
| `reports/phase5_backtest.md` | the Wayanad backtest: lead time **and** false-alarm rate, together |
| `reports/phase5_loco.md`, `phase5_sensitivity.md` | leave-one-cluster-out, sensitivity |
| `reports/phase6_alerts.md`, `phase6_architecture.md` | alert levels, cost ratios, latency budget |
| `reports/quality_gates.md` | every gate, PASS / FAIL / BLOCKED |
| `figures/money_figure.png` | the Wayanad money figure |
| `figures/leadtime_vs_falsealarm.png` | the operating trade-off |
| `experiments/register.csv` | one row per fit ever run: config hash, seed, likelihood, branching ratio |
| `models/fits/*.json` | fitted parameters per stage, with provenance |

---

## 7. What to re-run after you change something

| You changed | Re-run |
|---|---|
| a landslide inventory in `data/raw/` | `catalog --force` → `covariates --force` → `rootcause` → `cascade` → `experiments` → `realtime` |
| `data_mode.glofas` | `catalog --force` → same tail as above |
| `rainfall.primary` | `rainfall --force` → same tail as above |
| `region.seed_bbox` / `region.id` | everything, from `aoi` |
| fit settings (`--restarts`, `--steps`) | `cascade` → `experiments` → `realtime` |
| nothing (just want the pictures again) | `figures` |

---

## 8. When something goes wrong

| Symptom | Cause and fix |
|---|---|
| `RainfallSourceMissing: ... no .nc / .nc4 files found under data/raw/IMD/...` | `rainfall.primary` is `imd`/`imerg` but the files are not in a dated subfolder. Move them into `data/raw/IMD/<YYYY-MM-DD>/`. |
| `... has rainfall files but no PROVENANCE.md` | Write one (§3, template above). This is deliberate and not bypassable. |
| `CredentialsMissing: data_mode.glofas is 'real' but there are no GloFAS files and no CDS key` | Either add the files (§3.2 option A) or `~/.cdsapirc` (option B), or set `glofas: synthetic` back. |
| `no reach could be matched to a GloFAS cell` | The downloaded sub-area does not cover the basin. Re-download with N 12.0, W 75.0, S 10.4, E 77.2. |
| `could not read <file>: ValueError: could not find latitude/longitude columns` | The inventory uses unusual column names. Rename them to `lat`/`lon` (and `date`, `area_m2`) — it is logged as a warning and that file is skipped, not fatal. |
| Gate P1.1 says `BLOCKED ... rows are SYNTHETIC` | Expected until §3.1 is done. Gate 1 counts real events only. |
| A run dies silently on a 7–8 GB machine | Memory. Run one phase per process: `bash scripts/run_phases_4_to_6.sh`. |
| `git_commit: not-a-git-repo` in fit records | `git init && git add -A && git commit -m "baseline"`. Doc 07 §7 rule 8 wants each fit traceable to a commit. |

---

## 9. Checklist

- [ ] `pip install -e .` and `pytest -q` pass
- [ ] Stages `aoi` → `report` run clean
- [ ] A real landslide inventory is in `data/raw/<SOURCE>/<date>/` with `PROVENANCE.md` (§3.1)
- [ ] GloFAS files or `~/.cdsapirc`, and `data_mode.glofas: real` (§3.2)
- [ ] IMD or IMERG files, and `rainfall.primary` set (§3.3)
- [ ] `rootcause`, `cascade`, `experiments`, `realtime` all completed
- [ ] `hydrocascade check` — read every BLOCKED, and know why each one is blocked
- [ ] `hydrocascade app` opens and the synthetic banner says what you expect
