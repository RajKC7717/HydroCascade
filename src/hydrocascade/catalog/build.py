"""Build ``data/catalog/events.csv`` — the Phase 1 deliverable (Doc 04 §8).

Order of operations:

1. landslides: every inventory the team has downloaded, plus the documented
   seed events, plus (only if nothing real is available) a labelled synthetic
   stand-in so the rest of the pipeline can run;
2. floods: the Q2-exceedance procedure of Doc 02 §3 over the discharge table;
3. deduplicate at 500 m / 1 day with the written merge rule (FR-03);
4. attach the spatial unit and reach ids from Phase 2 (Doc 05 §9 gate 3);
5. sample the DEM for elevation;
6. size-frequency completeness analysis and the ``A_min`` cutoff;
7. validate against the schema and write, with a dated CHANGELOG entry.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..provenance import default_accessor
from . import completeness, dedup, schema

log = get("catalog.build")

EVENTS_PATH = paths.CATALOG / "events.csv"
CHANGELOG_PATH = paths.CATALOG / "events_CHANGELOG.md"
CASCADE_PAIRS_PATH = paths.CATALOG / "cascade_pairs.csv"


def run(cfg: Node, *, skip_floods: bool = False, force: bool = False) -> Path:
    paths.CATALOG.mkdir(parents=True, exist_ok=True)
    accessor = default_accessor()

    frames = []
    notes: list[str] = []

    # ---------------------------------------------------------- landslides ---
    from ..ingest import landslides

    real = landslides.discover_inventories(cfg, accessor)
    if not real.empty:
        frames.append(real)
        notes.append(f"{len(real)} landslide records from downloaded inventories")
    seed = landslides.seed_events(cfg, accessor)
    # Doc 04 §2 admits no exceptions: every catalog row carries a source_url and
    # an accessed_by. The Doc 04 §5 events whose coordinates are not yet sourced
    # therefore belong in a pending list, not in events.csv — otherwise the
    # catalog fails FR-04 by construction and the check stops meaning anything.
    pending = seed[seed["source"] == "NEEDS_SOURCING"]
    sourced = seed[seed["source"] != "NEEDS_SOURCING"]
    frames.append(sourced)
    notes.append(f"{len(sourced)} documented seed events with full provenance (Doc 04 §5)")
    if not pending.empty:
        _write_pending(pending)
        notes.append(f"{len(pending)} documented events held in events_PENDING.csv "
                     "awaiting coordinates and a source URL")

    if real.empty:
        log.warning("no real landslide inventory found in data/raw/. Falling back "
                    "to a SYNTHETIC inventory so the pipeline runs end to end. "
                    "Drop ESSD/NRSC/Bhukosh downloads into data/raw/<SOURCE>/ with "
                    "a PROVENANCE.md and re-run to replace it.")
        try:
            from ..spatial.slope_units import SlopeUnits
            synth = landslides.synthetic_inventory(cfg, SlopeUnits().gdf)
            frames.append(synth)
            notes.append(f"{len(synth)} SYNTHETIC landslide records")
        except FileNotFoundError:
            log.error("slope units not built; run `hydrocascade units` first")

    # -------------------------------------------------------------- floods ---
    if not skip_floods:
        try:
            frames.append(_flood_events(cfg, accessor, notes))
        except Exception as exc:  # noqa: BLE001
            log.error("flood detection failed: %s: %s", type(exc).__name__, exc)

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = schema.coerce(df)
    df = schema.make_event_ids(df)
    log.info("assembled %d raw records", len(df))

    # ---------------------------------------------------------- deduplicate ---
    kept, discarded = dedup.deduplicate(
        df,
        distance_m=float(cfg.catalog.dedup.distance_m),
        days=int(cfg.catalog.dedup.days),
        crs=str(cfg.crs.analysis),
        hazard_types=tuple(cfg.catalog.dedup.hazard_types),
    )
    notes.append(f"{len(discarded)} records merged away by the 500 m / 1 day rule")

    # ------------------------------------------------------- spatial join ---
    kept = _attach_spatial(cfg, kept)

    # ------------------------------------------------------- completeness ---
    areas = pd.to_numeric(kept.loc[kept["hazard_type"] == "L", "area_m2"],
                          errors="coerce").dropna()
    comp = None
    if len(areas) >= 30:
        try:
            comp = completeness.find_rollover(
                areas.to_numpy(), n_bins=int(cfg.catalog.completeness.n_bins))
            notes.append(comp.summary())
            configured = cfg.catalog.completeness.get("min_area_m2")
            if configured:
                kept = completeness.apply_cutoff(kept, float(configured))
                notes.append(f"applied configured cutoff {float(configured):,.0f} m2")
            else:
                log.info("A_min estimated but NOT applied. Set "
                         "catalog.completeness.min_area_m2: %.0f in the config to "
                         "model only events above the rollover (Doc 04 §3.4 step 3).",
                         comp.a_min_m2)
        except ValueError as exc:
            log.warning("completeness analysis skipped: %s", exc)
    else:
        log.warning("only %d landslides carry a mapped area; the size-frequency "
                    "analysis needs more before A_min is meaningful", len(areas))

    # ------------------------------------------------------------ validate ---
    kept = schema.coerce(kept)
    problems = schema.validate(kept, strict=False)
    if problems:
        log.warning("catalog has %d schema problem(s):", len(problems))
        for p in problems:
            log.warning("  - %s", p)
    else:
        log.info("catalog passes the Doc 04 §2 schema")

    schema.write(kept, EVENTS_PATH)
    _write_changelog(cfg, kept, discarded, notes, comp)
    _ensure_cascade_pairs(cfg)

    _summarise(kept)
    return EVENTS_PATH


def _flood_events(cfg: Node, accessor: str, notes: list[str]) -> pd.DataFrame:
    """Run the Doc 02 §3 procedure over the discharge table."""
    from ..ingest import adapters, floods
    from ..spatial.network import Network

    net = Network()
    reaches = net.gdf[["reach_id", "area_dn_km2"]].copy()
    disch = adapters.discharge(cfg, reaches)
    synthetic = bool(disch.attrs.get("synthetic", False))

    areas = reaches.set_index("reach_id")["area_dn_km2"]
    events = floods.detect_for_all_reaches(disch, areas, cfg)
    if events.empty:
        return pd.DataFrame()

    # keep only events inside the study period; the long record exists purely to
    # fit the Gumbel thresholds
    start, end = str(cfg.period.start), str(cfg.period.end)
    events = events[(events["date"] >= start) & (events["date"] <= end)]
    log.info("%d flood events fall inside the study period %s..%s",
             len(events), start, end)

    # locate each event at the downstream end of its reach
    geom = net.gdf.set_index("reach_id").geometry
    import geopandas as gpd
    pts = gpd.GeoSeries([geom.loc[r].interpolate(1.0, normalized=True)
                         for r in events["reach_id"]], crs=net.gdf.crs).to_crs("EPSG:4326")

    src = "GLOFAS_V4" if not synthetic else "SYNTHETIC"
    url = ("https://cds.climate.copernicus.eu/datasets/cems-glofas-historical"
           if not synthetic else "synthetic://hydrocascade/discharge")
    out = pd.DataFrame({
        "hazard_type": "F",
        "date": events["date"].to_numpy(),
        "time_known": "N", "time": "",
        "lat": pts.y.to_numpy(), "lon": pts.x.to_numpy(),
        "geometry_type": "reach",
        "area_m2": np.nan,
        "mark_raw": events["q_peak"].to_numpy(),
        "mark_value": events["mark_value"].to_numpy(),
        "severity_class": events["severity_class"].to_numpy(),
        "spatial_unit_id": events["reach_id"].astype(str).to_numpy(),
        "source": src,
        "source_url": url,
        "source_date_accessed": date.today().isoformat(),
        "accessed_by": accessor,
        "confidence": "low" if synthetic else "medium",
        "duplicate_of": "",
        "notes": ("SYNTHETIC discharge — not a real flood record. " if synthetic else "")
                 + f"Q2 exceedance, {cfg.floods.decluster_days}-day declustering, "
                   f"peak {events['q_peak'].round(1).astype(str)} m3/s",
    })
    notes.append(f"{len(out)} flood events from {'synthetic' if synthetic else 'GloFAS'} "
                 f"discharge by the Q{cfg.floods.threshold_return_period} rule")
    return out


def _attach_spatial(cfg: Node, df: pd.DataFrame) -> pd.DataFrame:
    """Doc 05 §9 gate 3: every event maps to exactly one spatial unit."""
    try:
        from ..spatial.slope_units import SlopeUnits
        from ..spatial.terrain import Routing
    except ImportError:
        return df
    try:
        rt = Routing()
        units = SlopeUnits()
    except FileNotFoundError:
        log.warning("Phase 2 artefacts missing; spatial_unit_id left blank")
        return df

    out = df.copy()
    import rasterio
    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        dem = src.read(1)

    unit_ids: list[str] = []
    elevations: list[float] = []
    n_outside = 0
    for _, row in out.iterrows():
        if row["hazard_type"] == "F":
            unit_ids.append(str(row.get("spatial_unit_id", "")))
            elevations.append(np.nan)
            continue
        try:
            cell = rt.index_of_lonlat(float(row["lon"]), float(row["lat"]))
        except (ValueError, TypeError):
            unit_ids.append("")
            elevations.append(np.nan)
            n_outside += 1
            continue
        uid = units.unit_of_cell[cell]
        unit_ids.append(str(int(uid)) if uid >= 0 else "")
        r, c = divmod(cell, rt.cols)
        elevations.append(float(dem[r, c]))

    out["spatial_unit_id"] = unit_ids
    out["elevation_m"] = np.where(np.isfinite(elevations), elevations,
                                  pd.to_numeric(out["elevation_m"], errors="coerce"))

    is_l = out["hazard_type"] == "L"
    unjoined = int((is_l & (out["spatial_unit_id"] == "")).sum())
    if n_outside:
        log.warning("%d landslide records fall outside the study grid", n_outside)
    if unjoined:
        log.warning("%d landslide records did not join to a slope unit", unjoined)
    else:
        log.info("every landslide record joined to exactly one slope unit")
    return out


def _write_changelog(cfg, kept, discarded, notes, comp) -> None:
    """Doc 04 §8: every change, dated, with a reason."""
    stamp = date.today().isoformat()
    lines = [f"\n## {stamp} — rebuild by `hydrocascade catalog`\n",
             f"- region `{cfg.region.id}`, period {cfg.period.start} to {cfg.period.end}",
             f"- **{len(kept)} events** ({int((kept['hazard_type']=='L').sum())} landslide, "
             f"{int((kept['hazard_type']=='F').sum())} flood)"]
    lines += [f"- {n}" for n in notes]
    if comp is not None:
        lines.append(f"- size-frequency: {comp.summary()}")
    by_source = kept.groupby("source").size().sort_values(ascending=False)
    lines.append("- by source: " + ", ".join(f"`{s}`={c}" for s, c in by_source.items()))
    if "SYNTHETIC" in by_source.index:
        lines.append("- **WARNING: this catalog contains synthetic records and must "
                     "not be used for any reported number.**")

    header = ("# events.csv changelog\n\nEvery change to the catalog, dated, with a "
              "reason (Doc 04 §8).\n")
    prev = CHANGELOG_PATH.read_text(encoding="utf-8") if CHANGELOG_PATH.is_file() else header
    CHANGELOG_PATH.write_text(prev + "\n".join(lines) + "\n", encoding="utf-8")
    log.info("appended to %s", CHANGELOG_PATH.name)


def _write_pending(pending: pd.DataFrame) -> None:
    """Documented events that still need coordinates and a citable source.

    Doc 04 §5 makes this an explicit team exercise — the Puthumala date is
    disputed between 8 and 29 August 2019 and must be resolved from GSI or the
    primary field report, recording which source was trusted.
    """
    out = paths.CATALOG / "events_PENDING.csv"
    schema.write(schema.coerce(pending), out)
    log.info("%d documented events await sourcing -> %s", len(pending), out.name)


def _ensure_cascade_pairs(cfg: Node) -> None:
    """Seed the Doc 04 §5 linking table, leaving an already-edited one alone."""
    from . import cascade_pairs
    try:
        cascade_pairs.build(cfg)
    except Exception as exc:  # noqa: BLE001 — never fail the whole catalog over this
        log.error("cascade pair table not written: %s: %s", type(exc).__name__, exc)


def _summarise(df: pd.DataFrame) -> None:
    log.info("catalog: %d events", len(df))
    for htype, label in (("L", "landslide"), ("F", "flood")):
        sel = df[df["hazard_type"] == htype]
        if sel.empty:
            continue
        years = pd.to_datetime(sel["date"], errors="coerce").dt.year
        log.info("  %-9s %5d events, %s-%s, sources: %s", label, len(sel),
                 int(years.min()) if years.notna().any() else "?",
                 int(years.max()) if years.notna().any() else "?",
                 ", ".join(sorted(sel["source"].unique())))
