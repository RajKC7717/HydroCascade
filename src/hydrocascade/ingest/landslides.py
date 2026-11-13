"""Landslide inventories (datasets E1-E6, Doc 10 §2).

Doc 04 §3.1 ranks the sources: published Kerala inventories first (ESSD 2018,
NRSC), then GSI Bhukosh, then GSI Bhusanket field reports, then published event
studies, with NASA COOLR as a pre-2018 cross-check only.

**None of those is fetchable without either an account or a manual portal
download**, and as of this build the Bhukosh portal does not respond, the COOLR
Socrata endpoint returns 404 and the PANGAEA DOI for the ESSD dataset 404s. So
this module provides:

``read_inventory()``
    A generic reader for whatever the team downloads by hand — CSV, shapefile
    or GeoPackage — mapping it into the Doc 04 §2 schema, including the
    polygon-to-crown conversion. Drop a file into ``data/raw/<SOURCE>/`` with a
    ``PROVENANCE.md`` beside it and it is picked up.

``seed_events()``
    The documented events from Doc 04 §5's starter list, with only the fields
    the document suite actually states. Rows whose coordinates are not yet
    sourced are emitted with ``confidence=low`` and a ``NEEDS_SOURCING`` note
    rather than invented, because Doc 04 §2 says every row carries a real
    ``source_url`` and Doc 04 §5 makes provenance discipline an explicit
    training exercise.

``synthetic_inventory()``
    A labelled stand-in so the dedup, completeness and figure code can be
    exercised end to end before the real inventories land.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("ingest.landslides")

# Doc 04 §5 starter list. Coordinates are filled in only where the document
# suite states them; the rest are left blank on purpose.
SEED_EVENTS = [
    dict(name="Mundakkai-Chooralmala", date="2024-07-30", time="02:17", time_known="Y",
         lat=11.4654, lon=76.1348, area_m2=np.nan, confidence="high",
         source="GSI_BHUSANKET_FIR",
         source_url="https://bhusanket.gsi.gov.in",
         notes="Doc 01 §2. Crown 11.4654N 76.1348E, elevation ~1550-1600 m, slope "
               "25-40 deg, overburden <5 m. Planar failure into long-runout debris "
               "flow, 6-8 km runout. FINAL HELD-OUT TEST EVENT: never in training."),
    dict(name="Puthumala", date="", time="", time_known="N",
         lat=np.nan, lon=np.nan, area_m2=np.nan, confidence="low",
         source="NEEDS_SOURCING", source_url="",
         notes="Doc 04 §5 provenance warning: published sources disagree on the "
               "date (8 vs 29 August 2019). Resolve from GSI or the primary field "
               "report and record which source was trusted. 2-3 km from the 2024 crown."),
    dict(name="Kavalappara", date="2019-08-08", time="", time_known="N",
         lat=np.nan, lon=np.nan, area_m2=np.nan, confidence="low",
         source="NEEDS_SOURCING", source_url="",
         notes="Doc 04 §5 test event, Malappuram. Coordinates to be sourced from GSI."),
    dict(name="Pettimudi", date="2020-08-06", time="", time_known="N",
         lat=np.nan, lon=np.nan, area_m2=np.nan, confidence="low",
         source="NEEDS_SOURCING", source_url="",
         notes="Doc 04 §5 test event, Idukki. Coordinates to be sourced from GSI."),
    dict(name="Kokkayar", date="2021-10-16", time="", time_known="N",
         lat=np.nan, lon=np.nan, area_m2=np.nan, confidence="low",
         source="NEEDS_SOURCING", source_url="",
         notes="Doc 04 §5 test event, October 2021 cluster. Coordinates to be sourced."),
]

# Column name synonyms seen across GSI / ESSD / NRSC / COOLR exports.
LAT_KEYS = ("lat", "latitude", "y", "ycoord", "y_coord", "crown_lat", "lat_dd")
LON_KEYS = ("lon", "long", "longitude", "x", "xcoord", "x_coord", "crown_lon", "lon_dd")
DATE_KEYS = ("date", "event_date", "eventdate", "occurrence_date", "date_", "ls_date")
AREA_KEYS = ("area_m2", "area", "area_sqm", "shape_area", "st_area(shape)", "area_m")


def _first_match(columns, keys) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def read_inventory(path: Path, *, source: str, source_url: str, accessed_by: str,
                   accessed: str, confidence: str = "medium",
                   dem_path: Path | None = None) -> pd.DataFrame:
    """Read one downloaded inventory into the Doc 04 §2 schema.

    Polygons are converted to a **crown point** — Doc 04 §3.2:

        Use the highest-elevation point of the polygon, not the centroid.

    A landslide starts at the crown and runs downhill; the centroid of a
    long-runout polygon sits in the middle of the runout path, which is
    physically wrong and corrupts both the terrain covariates and the flow-path
    projection. The original polygon is kept in ``data/interim/`` so the area
    and runout path can be recovered.
    """
    path = Path(path)
    if path.suffix.lower() in (".shp", ".gpkg", ".geojson", ".json"):
        return _read_vector(path, source, source_url, accessed_by, accessed,
                            confidence, dem_path)
    if path.suffix.lower() in (".csv", ".txt"):
        return _read_table(path, source, source_url, accessed_by, accessed, confidence)
    raise ValueError(f"unsupported inventory format: {path.suffix}")


def _read_table(path, source, source_url, accessed_by, accessed, confidence):
    df = pd.read_csv(path)
    lat_c = _first_match(df.columns, LAT_KEYS)
    lon_c = _first_match(df.columns, LON_KEYS)
    date_c = _first_match(df.columns, DATE_KEYS)
    if not (lat_c and lon_c):
        raise ValueError(f"{path.name}: could not find latitude/longitude columns "
                         f"among {list(df.columns)[:12]}")
    area_c = _first_match(df.columns, AREA_KEYS)

    out = pd.DataFrame({
        "hazard_type": "L",
        "date": pd.to_datetime(df[date_c], errors="coerce").dt.strftime("%Y-%m-%d")
                if date_c else "",
        "lat": pd.to_numeric(df[lat_c], errors="coerce"),
        "lon": pd.to_numeric(df[lon_c], errors="coerce"),
        "geometry_type": "point",
        "area_m2": pd.to_numeric(df[area_c], errors="coerce") if area_c else np.nan,
    })
    return _finalise(out, source, source_url, accessed_by, accessed, confidence)


def _read_vector(path, source, source_url, accessed_by, accessed, confidence, dem_path):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    log.info("%s: %d features, geometry %s", path.name, len(gdf),
             gdf.geom_type.value_counts().to_dict())

    date_c = _first_match(gdf.columns, DATE_KEYS)
    is_poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"])

    # keep the originals for area and runout recovery (Doc 04 §3.2)
    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    keep = paths.INTERIM / f"{source}_original{path.suffix}"
    if not keep.exists():
        gdf.to_file(keep)
        log.info("kept original geometry at %s", keep)

    area_m2 = np.full(len(gdf), np.nan)
    if is_poly.any():
        metric = gdf.to_crs("EPSG:32643")
        area_m2 = metric.geometry.area.to_numpy()

    if is_poly.any() and dem_path is not None:
        lat, lon, gtype = _crown_points(gdf, is_poly, dem_path)
    else:
        if is_poly.any():
            log.warning("no DEM supplied: falling back to polygon centroids. Doc 04 "
                        "§3.2 requires the crown; re-run with the DEM prepared.")
        pts = gdf.geometry.centroid.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" \
            else gdf.geometry.centroid
        lat, lon = pts.y.to_numpy(), pts.x.to_numpy()
        gtype = np.where(is_poly, "polygon_centroid", "point")

    out = pd.DataFrame({
        "hazard_type": "L",
        "date": pd.to_datetime(gdf[date_c], errors="coerce").dt.strftime("%Y-%m-%d")
                if date_c else "",
        "lat": lat, "lon": lon, "geometry_type": gtype, "area_m2": area_m2,
    })
    return _finalise(out, source, source_url, accessed_by, accessed, confidence)


def _crown_points(gdf, is_poly, dem_path):
    """Highest-elevation point inside each polygon (Doc 04 §3.2)."""
    import rasterio
    from rasterio.mask import mask as rio_mask

    src_crs = gdf.crs or "EPSG:4326"
    lat = np.full(len(gdf), np.nan)
    lon = np.full(len(gdf), np.nan)
    gtype = np.where(is_poly, "polygon_crown", "point").astype(object)

    with rasterio.open(dem_path) as src:
        g = gdf.to_crs(src.crs)
        for i, geom in enumerate(g.geometry):
            if not is_poly.iloc[i]:
                p = geom if geom.geom_type == "Point" else geom.centroid
                lon[i], lat[i] = _to_wgs84(p.x, p.y, src.crs)
                continue
            try:
                arr, tr = rio_mask(src, [geom], crop=True, nodata=np.nan, filled=True)
            except ValueError:
                p = geom.centroid
                lon[i], lat[i] = _to_wgs84(p.x, p.y, src.crs)
                gtype[i] = "polygon_centroid"
                continue
            band = arr[0]
            if not np.isfinite(band).any():
                p = geom.centroid
                lon[i], lat[i] = _to_wgs84(p.x, p.y, src.crs)
                gtype[i] = "polygon_centroid"
                continue
            r, c = np.unravel_index(np.nanargmax(band), band.shape)
            x, y = tr * (c + 0.5, r + 0.5)
            lon[i], lat[i] = _to_wgs84(x, y, src.crs)
    return lat, lon, gtype


def _to_wgs84(x, y, crs):
    import rasterio.warp
    lons, lats = rasterio.warp.transform(crs, "EPSG:4326", [x], [y])
    return lons[0], lats[0]


def _finalise(out, source, source_url, accessed_by, accessed, confidence):
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    out["time_known"] = "N"
    out["time"] = ""
    out["source"] = source
    out["source_url"] = source_url
    out["accessed_by"] = accessed_by
    out["source_date_accessed"] = accessed
    out["confidence"] = confidence
    out["duplicate_of"] = ""
    out["notes"] = ""
    out["mark_raw"] = out["area_m2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mark_value"] = np.log(out["area_m2"].to_numpy(dtype=float))
    out["severity_class"] = severity_from_area(out["area_m2"].to_numpy(dtype=float))
    return out


def severity_from_area(area_m2: np.ndarray) -> np.ndarray:
    """small / medium / large landslide classes by mapped area."""
    a = np.asarray(area_m2, dtype=float)
    out = np.full(a.shape, "", dtype=object)
    out[a < 5_000] = "small"
    out[(a >= 5_000) & (a < 50_000)] = "medium"
    out[a >= 50_000] = "large"
    return out


def discover_inventories(cfg: Node, accessed_by: str) -> pd.DataFrame:
    """Read every inventory the team has downloaded into ``data/raw/``.

    A folder qualifies if it contains a ``PROVENANCE.md`` (Doc 10 §8) and at
    least one readable inventory file.
    """
    from ..ingest.dem import DEM_PATH

    frames = []
    if not paths.RAW.is_dir():
        return pd.DataFrame()

    for prov in paths.RAW.rglob("PROVENANCE.md"):
        folder = prov.parent
        source = folder.relative_to(paths.RAW).parts[0]
        if source in {"HYDROSHEDS", "COP_DEM_GLO30", "CHIRPS", "IMD", "IMERG", "GLOFAS"}:
            continue
        meta = _parse_provenance(prov)
        for f in folder.iterdir():
            if f.suffix.lower() not in (".csv", ".shp", ".gpkg", ".geojson"):
                continue
            try:
                df = read_inventory(
                    f, source=source, source_url=meta.get("URL", ""),
                    accessed_by=meta.get("By", accessed_by),
                    accessed=meta.get("Downloaded", ""),
                    dem_path=DEM_PATH if DEM_PATH.is_file() else None)
                log.info("read %s from %s: %d records", f.name, source, len(df))
                frames.append(df)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not read %s: %s: %s", f, type(exc).__name__, exc)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_provenance(path: Path) -> dict[str, str]:
    meta = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta


def seed_events(cfg: Node, accessed_by: str) -> pd.DataFrame:
    """The documented events from Doc 04 §5, with nothing invented."""
    rows = []
    for ev in SEED_EVENTS:
        rows.append({
            "hazard_type": "L",
            "date": ev["date"],
            "time_known": ev["time_known"],
            "time": ev["time"],
            "lat": ev["lat"], "lon": ev["lon"],
            "geometry_type": "polygon_crown" if np.isfinite(ev["lat"]) else "point",
            "area_m2": ev["area_m2"],
            "mark_raw": ev["area_m2"],
            "mark_value": np.log(ev["area_m2"]) if np.isfinite(ev["area_m2"]) else np.nan,
            "severity_class": "large" if ev["name"].startswith("Mundakkai") else "",
            "source": ev["source"],
            "source_url": ev["source_url"],
            "source_date_accessed": "2026-09-04",
            "accessed_by": accessed_by,
            "confidence": ev["confidence"],
            "duplicate_of": "",
            "notes": f"{ev['name']}. {ev['notes']}",
        })
    df = pd.DataFrame(rows)
    unsourced = int((df["source"] == "NEEDS_SOURCING").sum())
    log.info("seed events: %d documented, %d still need coordinates and a source URL",
             len(df), unsourced)
    return df


def synthetic_inventory(cfg: Node, units_gdf, n_events: int = 1200) -> pd.DataFrame:
    """A labelled stand-in inventory for exercising the pipeline.

    Landslides are placed on steep, high slope units with probability rising in
    slope angle, dated within the monsoon seasons of the study period, and given
    areas from an inverse power law with a rollover — so the size-frequency
    analysis in :mod:`hydrocascade.catalog.completeness` has something with a
    known answer to find.

    **Never a source for a reported number.** Every row is stamped
    ``source=SYNTHETIC`` and the quality gates fail a catalog containing them.
    """
    rng = np.random.default_rng(int(cfg.project.seed))

    slope = units_gdf["slope_mean_deg"].to_numpy(dtype=float)
    elev = units_gdf["elev_mean_m"].to_numpy(dtype=float)
    ok = np.isfinite(slope) & np.isfinite(elev)
    weight = np.where(ok, np.clip(slope - 8.0, 0, None) ** 2.2, 0.0)
    weight *= np.where(elev > 250, 1.0, 0.05)
    if weight.sum() <= 0:
        raise RuntimeError("no steep units to place synthetic landslides on")
    weight = weight / weight.sum()

    pick = rng.choice(len(units_gdf), size=n_events, p=weight, replace=True)
    sel = units_gdf.iloc[pick]

    start = pd.Timestamp(str(cfg.period.start))
    end = pd.Timestamp(str(cfg.period.end))
    days = pd.date_range(start, end, freq="D")
    doy = days.dayofyear.to_numpy()
    season = np.exp(-0.5 * ((doy - 205) / 30.0) ** 2) + 1e-4
    season /= season.sum()
    dates = days[rng.choice(len(days), size=n_events, p=season)]

    # inverse power law above a rollover, so completeness analysis has a target
    a_min_true, alpha = 800.0, 2.4
    u = rng.random(n_events)
    area = a_min_true * (1 - u) ** (-1.0 / (alpha - 1))
    # censor small events the way a real inventory is incomplete below rollover
    observed = rng.random(n_events) < np.clip(area / (a_min_true * 3.0), 0.05, 1.0)

    import geopandas as gpd
    reps = gpd.GeoSeries(gpd.points_from_xy(sel["rep_x"], sel["rep_y"]),
                         crs=units_gdf.crs).to_crs("EPSG:4326")

    df = pd.DataFrame({
        "hazard_type": "L",
        "date": dates.strftime("%Y-%m-%d"),
        "time_known": "N", "time": "",
        "lat": reps.y.to_numpy(), "lon": reps.x.to_numpy(),
        "geometry_type": "polygon_crown",
        "area_m2": area,
        "mark_raw": area,
        "mark_value": np.log(area),
        "severity_class": severity_from_area(area),
        "source": "SYNTHETIC",
        "source_url": "synthetic://hydrocascade/landslides",
        "source_date_accessed": "2026-09-04",
        "accessed_by": "hydrocascade.ingest.landslides.synthetic_inventory",
        "confidence": "low",
        "duplicate_of": "",
        "notes": "SYNTHETIC placeholder — not a real landslide record. "
                 f"True rollover area {a_min_true:.0f} m2, power-law exponent {alpha}.",
    })[observed]

    log.warning("generated %d SYNTHETIC landslide records (of %d drawn). These "
                "exercise the pipeline and must never appear in a reported number.",
                len(df), n_events)
    return df.reset_index(drop=True)
