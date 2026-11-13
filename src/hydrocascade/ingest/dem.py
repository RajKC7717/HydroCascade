"""Copernicus DEM GLO-30 (dataset T1, Doc 10 §4).

Fetched from the AWS Open Data mirror, which needs no account. Tiles are 1x1
degree COGs at 1 arcsec (~30 m at this latitude).

Doc 05 §2.2 steps 1-2: download tiles covering the study bbox **plus a 10 km
buffer** so flow routing does not terminate at the study boundary, then
reproject to EPSG:32643. Distances and areas cannot be computed in degrees, so
everything downstream of this module is metric.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import calculate_default_transform, reproject

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..provenance import Provenance
from .fetch import download
from .hydrosheds import load_aoi

log = get("ingest.dem")

SOURCE_CODE = "COP_DEM_GLO30"
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
CITATION = "European Space Agency (2024), Copernicus DEM GLO-30, doi:10.5270/ESA-c5d3d65"
LICENCE = "Copernicus DEM open licence (free, attribution required)"

DEM_PATH = paths.SPATIAL / "dem.tif"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """1-degree tiles covering ``(minlon, minlat, maxlon, maxlat)`` in EPSG:4326."""
    minlon, minlat, maxlon, maxlat = bounds
    lats = range(math.floor(minlat), math.floor(maxlat) + 1)
    lons = range(math.floor(minlon), math.floor(maxlon) + 1)
    return [(la, lo) for la in lats for lo in lons]


def fetch_tiles(bounds: tuple[float, float, float, float], *, force: bool = False) -> list[Path]:
    """Download every GLO-30 tile intersecting ``bounds``."""
    folder = paths.raw_dir(SOURCE_CODE, "glo30_tiles")
    got: list[Path] = []
    wanted = tiles_for_bounds(bounds)
    log.info("DEM tiles needed: %s", ", ".join(f"{la}/{lo}" for la, lo in wanted))

    for lat, lon in wanted:
        name = tile_name(lat, lon)
        url = f"{BUCKET}/{name}/{name}.tif"
        dest = folder / f"{name}.tif"
        try:
            got.append(download(url, dest, force=force))
        except Exception as exc:  # ocean-only tiles legitimately do not exist
            log.warning("tile %s unavailable (%s) — skipping; this is normal for "
                        "ocean-only tiles", name, type(exc).__name__)

    if not got:
        raise RuntimeError(f"no DEM tiles could be downloaded for bounds {bounds}")

    Provenance(
        source="Copernicus DEM GLO-30 (COG), AWS Open Data mirror",
        url=f"{BUCKET}/<tile>/<tile>.tif",
        query=f"1-degree tiles covering {bounds} (study AOI + "
              f"buffer): {', '.join(p.stem for p in got)}",
        licence=LICENCE,
        citation=CITATION,
        notes="Heights are orthometric (EGM2008). Reprojected to EPSG:32643 at 30 m "
              "by hydrocascade.ingest.dem.prepare before any analysis.",
    ).write(folder, files=got)
    return got


def prepare(cfg: Node, *, force: bool = False) -> Path:
    """Mosaic, reproject to the analysis CRS and clip to the buffered AOI.

    Writes ``data/processed/spatial/dem.tif``.
    """
    if DEM_PATH.is_file() and not force:
        with rasterio.open(DEM_PATH) as src:
            log.info("cached  dem.tif %dx%d %s", src.height, src.width, src.crs)
        return DEM_PATH

    aoi_buf = load_aoi(buffered=True)
    aoi_ll = aoi_buf.to_crs("EPSG:4326")
    bounds = tuple(aoi_ll.total_bounds)  # minx, miny, maxx, maxy
    log.info("buffered AOI bounds (EPSG:4326): %.4f %.4f %.4f %.4f", *bounds)

    tiles = fetch_tiles(bounds, force=force)

    srcs = [rasterio.open(p) for p in tiles]
    try:
        mosaic, transform = rio_merge(srcs, bounds=bounds)
        profile = srcs[0].profile.copy()
        src_crs = srcs[0].crs
        src_nodata = srcs[0].nodata
    finally:
        for s in srcs:
            s.close()

    mosaic = mosaic.astype("float32")
    if src_nodata is not None:
        mosaic[mosaic == src_nodata] = np.nan
    log.info("mosaic  %d x %d in %s", mosaic.shape[1], mosaic.shape[2], src_crs)

    # Reproject to the metric analysis CRS at the configured resolution.
    res = float(cfg.spatial.dem.resolution_m)
    dst_crs = cfg.crs.analysis
    dst_transform, width, height = calculate_default_transform(
        src_crs, dst_crs, mosaic.shape[2], mosaic.shape[1],
        left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3],
        resolution=res,
    )
    dst = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=mosaic[0], destination=dst,
        src_transform=transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    log.info("warped  %d x %d at %.0f m in %s", height, width, res, dst_crs)

    profile.update(driver="GTiff", dtype="float32", nodata=np.nan, count=1,
                   crs=dst_crs, transform=dst_transform, width=width, height=height,
                   compress="deflate", predictor=2, tiled=True,
                   blockxsize=256, blockysize=256)

    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    tmp = DEM_PATH.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(dst, 1)

    # Clip to the buffered AOI so we do not carry the whole bbox rectangle.
    with rasterio.open(tmp) as src:
        clipped, clip_transform = rio_mask(src, aoi_buf.geometry, crop=True,
                                           nodata=np.nan, filled=True)
        clip_profile = src.profile.copy()
    clip_profile.update(height=clipped.shape[1], width=clipped.shape[2],
                        transform=clip_transform)
    with rasterio.open(DEM_PATH, "w", **clip_profile) as out:
        out.write(clipped[0], 1)
    tmp.unlink(missing_ok=True)

    valid = np.isfinite(clipped[0])
    log.info("dem.tif %d x %d, %.1f%% valid, elevation %.0f-%.0f m",
             clipped.shape[1], clipped.shape[2], 100 * valid.mean(),
             float(np.nanmin(clipped[0])), float(np.nanmax(clipped[0])))
    return DEM_PATH


def load(path: Path | None = None):
    """Return ``(array, profile)`` for the prepared DEM, with nodata as NaN."""
    p = Path(path) if path else DEM_PATH
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade dem` first")
    with rasterio.open(p) as src:
        arr = src.read(1, masked=False).astype("float32")
        profile = src.profile.copy()
    return arr, profile


def check_landmark_elevations(cfg: Node) -> dict[str, float]:
    """Sample the DEM at each configured landmark.

    The Mundakkai crown should come back at roughly 1,550-1,600 m per the GSI
    field report (Doc 01 §2). A wildly different number means the DEM, the CRS
    or the landmark coordinates are wrong, and it is far cheaper to learn that
    here than in Phase 4.
    """
    import rasterio.warp

    out: dict[str, float] = {}
    with rasterio.open(DEM_PATH) as src:
        for name, pt in cfg.landmarks.items():
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs,
                                             [float(pt["lon"])], [float(pt["lat"])])
            val = next(src.sample([(xs[0], ys[0])]))[0]
            out[name] = float(val)
            log.info("landmark %-20s %8.1f m", name, val)
    return out
