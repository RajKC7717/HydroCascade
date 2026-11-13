# `data/reference/` — things we score against, never fit to

This directory holds independently documented events. Nothing in it ever enters
the model. It exists so that a definitional choice can be turned into a measured
number, which is what Doc 02 §3.3 requires of the flood definition.

The distinction matters enough to state twice:

- `data/catalog/` is **what the model learns from**.
- `data/reference/` is **what the model is judged against**.

Mixing them is the classic way to report a number that means nothing.

---

## `documented_floods.csv`

The denominator of the recall test in `reports/phase1_flood_definition.md`:
*of the floods that are independently documented, what fraction did our
Q2-exceedance rule flag?*

### Columns

| Column | Meaning |
|---|---|
| `flood_id` | `DF-0001`, unique |
| `date_start`, `date_end` | closed interval. If a source says only "August 2019", write `2019-08-01` to `2019-08-31` — do **not** sharpen a month into a guessed day |
| `place`, `district` | as named by the source |
| `lat`, `lon` | EPSG:4326, blank if the source names only a town. Blank is fine; invented is not |
| `in_pilot_basin` | `yes` / `no` / `unknown`. Rows marked `no` are reported but not scored — the Chaliyar pilot cannot be blamed for missing a flood in Idukki |
| `evidence_type` | `ems_activation`, `unosat`, `ksdma_report`, `cwc_gauge`, `news_archive`, `field_report` — or `project_brief` for a row that still needs re-sourcing |
| `source`, `source_url` | FR-04 applies here too. A row with no `source_url` is not scored |
| `source_date_accessed`, `accessed_by` | who fetched it and when |
| `notes` | anything the next person needs, especially disagreements between sources |

### How the file gets created

`hydrocascade report` writes it once, seeded from the Doc 04 §5 starter list,
and then never touches it again — the same rule as `cascade_pairs.csv`. Once a
human has edited it, it is theirs.

### The one thing to do before quoting a recall number

Every seeded row carries `evidence_type = project_brief`, meaning its date and
place come from `docs/04_PHASE_1_DATA_COLLECTION.md` §5 rather than from a
primary record. **The project brief is not independent evidence about the
project's own study region.** A recall number computed over those rows is
circular, and `corroboration.recall()` refuses to mark the result publishable
while any scored row is still `project_brief`.

Re-source each row from:

- **KSDMA** — Kerala State Disaster Management Authority annual disaster reports
  and post-event memoranda
- **Copernicus EMS / UNOSAT** — rapid-mapping activation footprints, which give
  a dated, official flood extent
- **CWC** — gauge water level and discharge from the National Water Data Portal,
  the only ground observation in the list
- **Dated news archives** — acceptable as corroboration, not as the sole source

Then set `evidence_type`, fill `source_url`, resolve `in_pilot_basin` against
`data/processed/spatial/aoi.gpkg`, and re-run `hydrocascade report`.

### Known provenance problem to resolve first

Doc 04 §5 flags it explicitly: published sources disagree on the date of the
Puthumala 2019 landslide — some give 8 August, others 29 August. Resolve it from
GSI or the primary field report and record **which source you trusted** in
`notes`. Doc 04 calls this the team's training exercise in provenance discipline,
and it is exactly the kind of small discrepancy that becomes an awkward question
in a review.
