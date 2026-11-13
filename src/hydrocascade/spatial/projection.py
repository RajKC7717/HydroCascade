"""The projection operator pi : H -> N (Doc 02 §2.2, Doc 05 §4).

*In plain English:* for every hillside in the study area, where would debris
leaving that hillside first enter a river, and how far would it travel to get
there?

Two levels:

``cell_level()``
    For **every** DEM cell, the first channel cell on its D8 flow path and the
    length of that path. Computed as one ascending-elevation scan rather than a
    per-unit walk, which turns an O(units x path) traversal into O(cells).
    Also the raw material for the half-basin split in
    :mod:`hydrocascade.spatial.slope_units`.

``build()``
    Aggregates to ``projection_table.parquet``: one row per slope unit, giving
    ``unit_id -> injection_reach_id, d_slope_km, injection_node``. This is the
    interface contract Doc 12 §5 names, consumed by the Phase 4 kernel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import flow
from .network import Network
from .terrain import Routing

log = get("spatial.projection")

CELL_PROJ_PATH = paths.SPATIAL / "cell_projection.npz"
TABLE_PATH = paths.SPATIAL / "projection_table.parquet"


def cell_level(rt: Routing | None = None, net: Network | None = None,
               *, force: bool = False) -> dict[str, np.ndarray]:
    """Per-cell injection cell, injection reach and path length. Cached."""
    if CELL_PROJ_PATH.is_file() and not force:
        z = np.load(CELL_PROJ_PATH, allow_pickle=False)
        return {"injection_cell": z["injection_cell"].astype(np.int64),
                "injection_reach": z["injection_reach"].astype(np.int64),
                "d_slope_m": z["d_slope_m"].astype(np.float64)}

    rt = rt or Routing()
    net = net or Network()

    log.info("tracing every cell to its first channel cell ...")
    injection, d_slope = flow.trace_to_channel(
        rt.receiver, rt.step_len, net.is_channel, rt.order_asc)

    reach = np.full(injection.shape, -1, dtype=np.int64)
    hit = injection >= 0
    reach[hit] = net.cell_to_reach[injection[hit]]

    # Denominator must be cells that are in the DEM at all, not the whole
    # rectangle: the clip to the buffered AOI leaves most of the grid as nodata.
    import rasterio
    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        in_dem = np.isfinite(src.read(1)).ravel()
    n_valid = int(hit.sum())
    n_total = int(in_dem.sum())
    log.info("        %d of %d cells in the DEM reach a channel (%.2f%%); the rest "
             "drain off the grid edge", n_valid, n_total, 100 * n_valid / max(n_total, 1))
    ok = hit & (d_slope > 0)
    log.info("        d_slope: median %.0f m, 95th pct %.0f m, max %.0f m",
             float(np.median(d_slope[ok])), float(np.percentile(d_slope[ok], 95)),
             float(np.nanmax(d_slope[ok])))

    np.savez_compressed(CELL_PROJ_PATH,
                        injection_cell=injection.astype(np.int32),
                        injection_reach=reach.astype(np.int32),
                        d_slope_m=d_slope.astype(np.float32))
    return {"injection_cell": injection, "injection_reach": reach, "d_slope_m": d_slope}


def build(cfg: Node, *, force: bool = False) -> Path:
    """Aggregate the cell-level projection to one row per slope unit."""
    from .slope_units import SlopeUnits

    if TABLE_PATH.is_file() and not force:
        log.info("cached  %s", TABLE_PATH.name)
        return TABLE_PATH

    rt = Routing()
    net = Network()
    units = SlopeUnits()
    proj = cell_level(rt, net)

    # Doc 05 §4 step 1: the unit's representative point is its HIGHEST point,
    # matching the crown convention used for landslide records (Doc 04 §3.2).
    # A landslide starts at the crown and runs downhill; using the centroid of
    # a long-runout polygon would place it mid-runout and corrupt both the
    # terrain covariates and the flow-path projection.
    rep = units.representative_cell()
    unit_ids = units.gdf["unit_id"].to_numpy()

    inj_reach = proj["injection_reach"][rep]
    d_slope_km = proj["d_slope_m"][rep] / 1000.0
    inj_cell = proj["injection_cell"][rep]

    missing = int((inj_reach < 0).sum())
    if missing:
        log.warning("%d of %d slope units have no injection reach (%.3f%%) — "
                    "these drain off the grid without meeting a channel",
                    missing, len(unit_ids), 100 * missing / len(unit_ids))
    else:
        log.info("every slope unit has an injection reach")

    df = pd.DataFrame({
        "unit_id": unit_ids,
        "injection_reach_id": inj_reach,
        "d_slope_km": d_slope_km,
        "injection_cell": inj_cell,
        "injection_node_id": inj_reach,     # reach id doubles as the network node
    })
    df.to_parquet(TABLE_PATH, index=False)

    ok = df[df["injection_reach_id"] >= 0]
    log.info("projection table: %d units, d_slope median %.2f km, max %.2f km",
             len(df), float(ok["d_slope_km"].median()), float(ok["d_slope_km"].max()))
    return TABLE_PATH


def load_table() -> pd.DataFrame:
    if not TABLE_PATH.is_file():
        raise FileNotFoundError(f"{TABLE_PATH} missing — run `hydrocascade projection`")
    return pd.read_parquet(TABLE_PATH)
