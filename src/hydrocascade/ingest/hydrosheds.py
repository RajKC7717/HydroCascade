"""HydroSHEDS / HydroBASINS — the study boundary (dataset T2, Doc 10 §4).

Doc 04 §1 fixes the boundary definition: *hydrological sub-basin polygons, not
administrative boundaries*, because a landslide in Wayanad district can flood a
reach in Malappuram district and an administrative cut would sever the cascade.

The area of interest is therefore derived, not drawn:

1. find the HydroBASINS polygon containing the seed point (the Mundakkai crown);
2. read its ``MAIN_BAS`` — the id of the most downstream basin of that whole
   river system;
3. take every polygon sharing that ``MAIN_BAS``. That set *is* the river basin,
   from the ridge above Mundakkai to the sea.

The individual sub-basins are kept as well: they become the grouping variable
for the spatial block cross-validation Doc 06 §5.2 makes mandatory.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from .. import paths
from ..config import Node, bbox
from ..logging_setup import get
from ..provenance import Provenance
from .fetch import download, unzip

log = get("ingest.hydrosheds")

SOURCE_CODE = "HYDROSHEDS"
BASE = "https://data.hydrosheds.org/file/hydrobasins/standard"
CITATION = "Lehner, B. and Grill, G. (2013), Hydrological Processes 27(15):2171-2186"
LICENCE = "HydroSHEDS licence, free for non-commercial and research use"


def _url(level: int, continent: str = "as") -> str:
    return f"{BASE}/hybas_{continent}_lev{level:02d}_v1c.zip"


def fetch(cfg: Node, *, force: bool = False) -> Path:
    """Download and unzip the HydroBASINS shapefile. Returns the .shp path."""
    level = int(cfg.region.hydrobasins_level)
    url = _url(level)
    folder = paths.raw_dir(SOURCE_CODE, "hybas_as_lev%02d" % level)
    zip_path = folder / Path(url).name

    download(url, zip_path, force=force)
    files = unzip(zip_path, folder)
    shp = next((f for f in files if f.suffix.lower() == ".shp"), None)
    if shp is None:
        raise FileNotFoundError(f"no .shp inside {zip_path}")

    Provenance(
        source=f"HydroBASINS standard level {level}, Asia (v1c)",
        url=url,
        query=f"whole-continent archive; subset to MAIN_BAS of the basin containing "
              f"{cfg.region.seed_point.lat}, {cfg.region.seed_point.lon}",
        licence=LICENCE,
        citation=CITATION,
        notes="Used to define the study boundary hydrologically (Doc 04 §1) and to "
              "supply sub-basin groups for spatial block CV (Doc 06 §5.2).",
    ).write(folder)
    return shp


def build_aoi(cfg: Node, *, force: bool = False) -> dict[str, Path]:
    """Derive the area of interest. Writes three files to ``data/processed/spatial``.

    Returns a dict of ``{name: path}`` for ``subbasins``, ``aoi``, ``aoi_buffered``.
    """
    shp = fetch(cfg, force=force)
    analysis_crs = cfg.crs.analysis
    seed = cfg.region.seed_point
    minlon, minlat, maxlon, maxlat = bbox(cfg)

    # Step 1 — locate the basin containing the seed point, reading only the
    # neighbourhood rather than all of Asia.
    near = gpd.read_file(shp, bbox=(minlon, minlat, maxlon, maxlat))
    if near.empty:
        raise RuntimeError(f"no HydroBASINS polygons inside region.seed_bbox {bbox(cfg)}")

    pt = Point(float(seed.lon), float(seed.lat))
    hit = near[near.geometry.contains(pt)]
    if hit.empty:  # tolerate a point exactly on a shared edge
        hit = near.iloc[[near.geometry.distance(pt).idxmin()]]
        log.warning("seed point not strictly inside any polygon; using nearest basin")
    main_bas = int(hit.iloc[0]["MAIN_BAS"])
    log.info("seed point %s sits in HYBAS_ID %d, river system MAIN_BAS %d",
             seed.name, int(hit.iloc[0]["HYBAS_ID"]), main_bas)

    # Step 2 — the whole river system, which may extend past the seed bbox.
    sub = gpd.read_file(shp, where=f"MAIN_BAS = {main_bas}")
    if sub.empty:
        raise RuntimeError(f"MAIN_BAS {main_bas} returned no polygons")
    sub = sub.to_crs(analysis_crs)
    sub["area_km2"] = sub.geometry.area / 1e6
    total = float(sub["area_km2"].sum())
    log.info("river system: %d sub-basins, %.0f km2 total", len(sub), total)

    # Step 3 — dissolve, then buffer so flow routing does not terminate at the
    # study boundary (Doc 05 §2.2 step 1).
    aoi = gpd.GeoDataFrame(
        {"region_id": [cfg.region.id], "main_bas": [main_bas], "area_km2": [total]},
        geometry=[sub.geometry.union_all()],
        crs=analysis_crs,
    )
    buf_m = float(cfg.region.buffer_km) * 1000.0
    aoi_buf = aoi.copy()
    aoi_buf["geometry"] = aoi.geometry.buffer(buf_m)
    aoi_buf["area_km2"] = aoi_buf.geometry.area / 1e6

    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    out = {
        "subbasins": paths.SPATIAL / "subbasins.gpkg",
        "aoi": paths.SPATIAL / "aoi.gpkg",
        "aoi_buffered": paths.SPATIAL / "aoi_buffered.gpkg",
    }
    sub.to_file(out["subbasins"], driver="GPKG")
    aoi.to_file(out["aoi"], driver="GPKG")
    aoi_buf.to_file(out["aoi_buffered"], driver="GPKG")

    log.info("AOI %.0f km2, buffered %.0f km2 -> %s",
             total, float(aoi_buf["area_km2"].iloc[0]), paths.SPATIAL)
    return out


def load_aoi(buffered: bool = False) -> gpd.GeoDataFrame:
    """Load the derived AOI. Raises a useful error if it was never built."""
    p = paths.SPATIAL / ("aoi_buffered.gpkg" if buffered else "aoi.gpkg")
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade aoi` first")
    return gpd.read_file(p)


def load_subbasins() -> gpd.GeoDataFrame:
    p = paths.SPATIAL / "subbasins.gpkg"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade aoi` first")
    return gpd.read_file(p)
