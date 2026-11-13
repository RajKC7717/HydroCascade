"""The hillslope domain H — slope units (Doc 02 §2.2, Doc 05 §2).

Doc 05 §2.1: *"A landslide belongs to a hillside, not to a square."* Slope units
are polygons bounded by ridges and drainage lines, the standard mapping unit in
landslide susceptibility work, and each one has a well-defined single flow
direction — which is exactly what the projection operator needs.

How they are built here
-----------------------
Doc 05 §2.2 step 5 asks for a half-basin partition: delineate sub-basins, then
split each along its main drainage line into left-bank and right-bank halves.
The flow routing already gives us both halves for free:

* every cell knows the first channel cell on its flow path (the projection
  operator), so the **local catchment of each reach** is just the set of cells
  injecting into it;
* the **bank** is the sign of the cross product between the channel's local
  direction and the vector from the injection point to the cell.

So ``unit = (injection reach, bank)``. The units are bounded by drainage lines
by construction, and by ridges wherever two reaches' catchments meet.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from .network import Network
from .terrain import Routing

log = get("spatial.slope_units")

UNITS_PATH = paths.SPATIAL / "slope_units.gpkg"
UNIT_INDEX_PATH = paths.SPATIAL / "unit_index.npz"


def _bank_side(rt: Routing, net: Network, injection_cell: np.ndarray,
               cells: np.ndarray) -> np.ndarray:
    """+1 / -1 for the left or right bank of the channel at the injection point.

    Sign of the 2D cross product between the channel's local flow direction and
    the vector from the injection cell to the hillslope cell. Cells lying on the
    channel line itself get 0 and are folded into the left bank.
    """
    inj = injection_cell[cells]
    r_c, c_c = rt.rowcol(cells)
    r_i, c_i = rt.rowcol(inj)

    # channel direction at the injection cell, from it to its receiver
    nxt = rt.receiver[inj]
    nxt = np.where(nxt >= 0, nxt, inj)
    r_n, c_n = rt.rowcol(nxt)
    dx = (c_n - c_i).astype(np.float64)
    dy = (r_n - r_i).astype(np.float64)

    vx = (c_c - c_i).astype(np.float64)
    vy = (r_c - r_i).astype(np.float64)

    cross = dx * vy - dy * vx
    return np.where(cross >= 0, 1, -1).astype(np.int8)


def build(cfg: Node, *, force: bool = False) -> dict[str, Path]:
    if UNITS_PATH.is_file() and UNIT_INDEX_PATH.is_file() and not force:
        log.info("cached  %s", UNITS_PATH.name)
        return {"units": UNITS_PATH, "index": UNIT_INDEX_PATH}

    from .projection import cell_level

    rt = Routing()
    net = Network()
    proj = cell_level(rt, net, force=force)

    min_area_km2 = float(cfg.spatial.slope_units.min_area_km2)
    target_km2 = float(cfg.spatial.slope_units.target_median_area_km2)

    inj_cell = proj["injection_cell"]
    inj_reach = proj["injection_reach"]
    n_grid = inj_cell.shape[0]

    assigned = np.flatnonzero(inj_reach >= 0)
    log.info("cells with an injection reach: %d", assigned.size)

    side = np.zeros(n_grid, dtype=np.int8)
    side[assigned] = _bank_side(rt, net, inj_cell, assigned)
    # A channel cell belongs to the network N, not to a hillslope; fold it into
    # the left bank so the tessellation stays complete and every event can be
    # joined to exactly one unit.
    side[net.is_channel] = 1

    key = np.full(n_grid, -1, dtype=np.int64)
    key[assigned] = inj_reach[assigned] * 2 + (side[assigned] > 0)

    uniq, inverse, counts = np.unique(key[assigned], return_inverse=True, return_counts=True)
    cell_area_km2 = rt.cell_area_km2
    areas = counts * cell_area_km2
    log.info("raw half-basins: %d, median area %.3f km2", uniq.size, float(np.median(areas)))

    # Doc 05 §2.2 step 6: discard slivers.
    keep = areas >= min_area_km2
    log.info("dropping %d units below %.3f km2 (%.2f%% of area)",
             int((~keep).sum()), min_area_km2,
             100 * areas[~keep].sum() / max(areas.sum(), 1e-9))

    remap = np.full(uniq.size, -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    unit_of_cell = np.full(n_grid, -1, dtype=np.int32)
    unit_of_cell[assigned] = remap[inverse]

    n_units = int(keep.sum())
    kept_areas = areas[keep]
    log.info("slope units: %d, area median %.3f km2 (target %.2f), "
             "IQR %.3f-%.3f, max %.2f",
             n_units, float(np.median(kept_areas)), target_km2,
             float(np.percentile(kept_areas, 25)), float(np.percentile(kept_areas, 75)),
             float(kept_areas.max()))

    gdf = _vectorise(rt, unit_of_cell, n_units)
    gdf = _attach_attributes(cfg, rt, net, gdf, unit_of_cell, proj, n_units)

    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    gdf.to_file(UNITS_PATH, driver="GPKG", layer="slope_units")
    np.savez_compressed(UNIT_INDEX_PATH, unit_of_cell=unit_of_cell, n_units=np.int64(n_units))
    log.info("wrote   %s (%d units)", UNITS_PATH.name, len(gdf))
    return {"units": UNITS_PATH, "index": UNIT_INDEX_PATH}


def _vectorise(rt: Routing, unit_of_cell: np.ndarray, n_units: int) -> gpd.GeoDataFrame:
    """Raster labels -> one dissolved polygon per unit."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    log.info("vectorising %d units ...", n_units)
    grid = unit_of_cell.reshape(rt.shape)
    mask = grid >= 0
    polys: dict[int, list] = {}
    for geom, val in shapes(grid.astype(np.int32), mask=mask, transform=rt.transform,
                            connectivity=8):
        polys.setdefault(int(val), []).append(shape(geom))

    from shapely import union_all
    rows = []
    for uid in range(n_units):
        parts = polys.get(uid)
        if not parts:
            continue
        rows.append({"unit_id": uid,
                     "geometry": parts[0] if len(parts) == 1 else union_all(parts)})
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=rt.crs)
    log.info("        %d polygons", len(gdf))
    return gdf


def _attach_attributes(cfg: Node, rt: Routing, net: Network, gdf: gpd.GeoDataFrame,
                       unit_of_cell: np.ndarray, proj: dict, n_units: int
                       ) -> gpd.GeoDataFrame:
    """Doc 05 §2.2 step 7: id, area, centroid, mean/max slope, mean elevation."""
    import rasterio

    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        z = src.read(1).ravel().astype(np.float64)
    with rasterio.open(paths.SPATIAL / "slope_deg.tif") as src:
        slope = src.read(1).ravel().astype(np.float64)

    valid = unit_of_cell >= 0
    lab = unit_of_cell[valid]
    order = np.argsort(lab, kind="stable")
    lab_sorted = lab[order]
    cells_sorted = np.flatnonzero(valid)[order]
    bounds = np.searchsorted(lab_sorted, np.arange(n_units + 1))

    stats = {k: np.full(n_units, np.nan) for k in
             ("area_km2", "elev_mean_m", "elev_max_m", "slope_mean_deg", "slope_max_deg")}
    rep_cell = np.full(n_units, -1, dtype=np.int64)

    for uid in range(n_units):
        cells = cells_sorted[bounds[uid]:bounds[uid + 1]]
        if cells.size == 0:
            continue
        zc = z[cells]
        sc = slope[cells]
        stats["area_km2"][uid] = cells.size * rt.cell_area_km2
        stats["elev_mean_m"][uid] = np.nanmean(zc)
        stats["elev_max_m"][uid] = np.nanmax(zc)
        stats["slope_mean_deg"][uid] = np.nanmean(sc)
        stats["slope_max_deg"][uid] = np.nanmax(sc)
        rep_cell[uid] = int(cells[np.nanargmax(zc)])   # the crown

    attrs = pd.DataFrame({"unit_id": np.arange(n_units), **stats,
                          "representative_cell": rep_cell,
                          "injection_reach_id": proj["injection_reach"][np.maximum(rep_cell, 0)],
                          "d_slope_km": proj["d_slope_m"][np.maximum(rep_cell, 0)] / 1000.0})
    attrs.loc[rep_cell < 0, ["injection_reach_id", "d_slope_km"]] = np.nan

    gdf = gdf.merge(attrs, on="unit_id", how="left")
    cx, cy = rt.xy(np.maximum(rep_cell, 0))
    lookup = pd.DataFrame({"unit_id": np.arange(n_units), "rep_x": cx, "rep_y": cy})
    return gdf.merge(lookup, on="unit_id", how="left")


class SlopeUnits:
    """Loaded slope units with the cell index."""

    def __init__(self):
        if not UNITS_PATH.is_file():
            raise FileNotFoundError(f"{UNITS_PATH} missing — run `hydrocascade units`")
        self.gdf = gpd.read_file(UNITS_PATH, layer="slope_units")
        z = np.load(UNIT_INDEX_PATH, allow_pickle=False)
        self.unit_of_cell = z["unit_of_cell"].astype(np.int64)
        self.n_units = int(z["n_units"])

    def __len__(self) -> int:
        return len(self.gdf)

    def representative_cell(self) -> np.ndarray:
        """Flat index of each unit's highest cell — the crown convention."""
        return self.gdf["representative_cell"].to_numpy().astype(np.int64)

    def unit_at(self, flat_cell: int) -> int:
        return int(self.unit_of_cell[flat_cell])
