"""Rainfall accumulation windows and the antecedent precipitation index (Doc 05 §7).

Produces ``R1, R3, R7, R15, R30`` — rolling sums ending on the day — plus the
API, the real-time wetness proxy of Doc 09 §1.3.

**Why this is stored per rainfall pixel, not per slope unit.** There are 17,138
slope units and 2,405 days; a unit x day x window table would be 41 million rows
before any covariate is added. The rainfall grid over this basin is a few
hundred pixels at most, and thousands of units share each one, so the windows
are computed on the rainfall grid and joined to units through a fixed lookup.
Identical numbers, three orders of magnitude less storage.

Doc 05 §7 also requires **catchment-averaged** rainfall for reaches rather than
point rainfall at the reach: *"Rain on the hills is what floods the river."*
That average is taken over the reach's upstream contributing area, using the
flow accumulation already computed in Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("features.rainfall")

PIXEL_RAIN_PATH = paths.SPATIAL / "rainfall_pixels.nc"
UNIT_PIXEL_PATH = paths.SPATIAL / "unit_rain_pixel.parquet"
REACH_PIXEL_PATH = paths.SPATIAL / "reach_rain_weights.npz"


def build(cfg: Node, *, force: bool = False) -> dict[str, Path]:
    """Load the active rainfall source, compute the windows and the API.

    The source is `rainfall.primary` in the config: chirps, imd or imerg.
    """
    from ..ingest import rainfall_sources

    if PIXEL_RAIN_PATH.is_file() and UNIT_PIXEL_PATH.is_file() and not force:
        log.info("cached  %s", PIXEL_RAIN_PATH.name)
        return {"pixels": PIXEL_RAIN_PATH, "unit_lookup": UNIT_PIXEL_PATH}

    da = rainfall_sources.load(cfg, force=force)
    da = da.where(da >= 0)  # missing values are negative in every source
    windows = [int(w) for w in cfg.rainfall.windows_days]

    log.info("rainfall grid %d lat x %d lon, %d days",
             da.sizes["latitude"], da.sizes["longitude"], da.sizes["time"])

    ds = xr.Dataset({"R1": da})
    for w in windows:
        if w == 1:
            continue
        ds[f"R{w}"] = da.rolling(time=w, min_periods=1).sum()

    ds["API"] = _api(da, float(cfg.rainfall.api.decay_k))

    start = str(cfg.period.start)
    end = str(cfg.period.end)
    ds = ds.sel(time=slice(start, end))
    paths.SPATIAL.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(PIXEL_RAIN_PATH)
    log.info("wrote   %s (%.1f MB, %d days)", PIXEL_RAIN_PATH.name,
             PIXEL_RAIN_PATH.stat().st_size / 1e6, ds.sizes["time"])

    lookup = _unit_pixel_lookup(cfg, da)
    lookup.to_parquet(UNIT_PIXEL_PATH, index=False)
    log.info("wrote   %s (%d units -> %d distinct rainfall pixels)",
             UNIT_PIXEL_PATH.name, len(lookup), lookup["pixel_id"].nunique())

    _reach_catchment_weights(cfg, da)
    return {"pixels": PIXEL_RAIN_PATH, "unit_lookup": UNIT_PIXEL_PATH}


def _api(da: xr.DataArray, k: float) -> xr.DataArray:
    """Antecedent precipitation index: ``API_t = k * API_{t-1} + P_t``.

    Doc 09 §1.3: ERA5-Land soil moisture is the physically correct wetness
    variable but has multi-day latency and cannot be used in real time. The API
    is built from rainfall alone, so the same number is available at run time as
    at fit time. Calibrating it against ERA5-Land is a Phase 3 task.
    """
    vals = da.values
    out = np.zeros_like(vals, dtype=np.float32)
    prev = np.zeros(vals.shape[1:], dtype=np.float32)
    for t in range(vals.shape[0]):
        today = np.nan_to_num(vals[t], nan=0.0)
        prev = k * prev + today
        out[t] = prev
    return xr.DataArray(out, coords=da.coords, dims=da.dims, name="API")


def _unit_pixel_lookup(cfg: Node, da: xr.DataArray) -> pd.DataFrame:
    """Nearest rainfall pixel for each slope unit's representative point."""
    from ..spatial.slope_units import SlopeUnits
    import geopandas as gpd

    units = SlopeUnits()
    g = units.gdf
    pts = gpd.GeoSeries(gpd.points_from_xy(g["rep_x"], g["rep_y"]),
                        crs=g.crs).to_crs("EPSG:4326")

    lats = da["latitude"].values
    lons = da["longitude"].values
    iy = np.abs(pts.y.to_numpy()[:, None] - lats[None, :]).argmin(axis=1)
    ix = np.abs(pts.x.to_numpy()[:, None] - lons[None, :]).argmin(axis=1)

    return pd.DataFrame({
        "unit_id": g["unit_id"].to_numpy(),
        "lat_index": iy.astype(np.int32),
        "lon_index": ix.astype(np.int32),
        "pixel_id": (iy * len(lons) + ix).astype(np.int32),
    })


def _reach_catchment_weights(cfg: Node, da: xr.DataArray) -> Path:
    """Fraction of each reach's upstream catchment falling in each rainfall pixel.

    Doc 05 §7: for a reach, average rainfall over its **upstream catchment**,
    not point rainfall at the reach.
    """
    import geopandas as gpd

    from ..spatial.network import Network
    from ..spatial.projection import cell_level
    from ..spatial.terrain import Routing

    rt = Routing()
    net = Network()
    proj = cell_level(rt, net)

    # Every cell injects into some reach; a reach's catchment is the union of
    # its own local catchment and everything upstream. We approximate the
    # catchment-average by weighting each reach's *local* catchment pixels and
    # accumulating downstream, which reuses the routing rather than rasterising
    # thousands of polygons.
    inj = proj["injection_reach"]
    valid = inj >= 0
    cells = np.flatnonzero(valid)

    x, y = rt.xy(cells)
    tf = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=rt.crs).to_crs("EPSG:4326")
    lats = da["latitude"].values
    lons = da["longitude"].values
    iy = np.abs(tf.y.to_numpy()[:, None] - lats[None, :]).argmin(axis=1)
    ix = np.abs(tf.x.to_numpy()[:, None] - lons[None, :]).argmin(axis=1)
    pixel = (iy * len(lons) + ix).astype(np.int32)

    reach_of_cell = inj[cells].astype(np.int64)
    n_reach = len(net)
    n_pixel = len(lats) * len(lons)

    from scipy.sparse import coo_matrix
    local = coo_matrix((np.ones(cells.size, dtype=np.float32),
                        (reach_of_cell, pixel)), shape=(n_reach, n_pixel)).tocsr()

    # Accumulate downstream so each reach carries its full upstream catchment.
    # Contributing area increases monotonically downstream, so visiting reaches
    # in ascending area order guarantees every upstream reach is added before
    # the reach it drains into.
    order = np.argsort(net.gdf["area_dn_km2"].to_numpy())
    ds = net.downstream
    rows = [local[i].toarray().ravel() for i in range(n_reach)]
    for i in order:
        d = int(ds[i])
        if d >= 0:
            rows[d] += rows[i]
    acc = np.vstack(rows)
    totals = acc.sum(axis=1, keepdims=True)
    weights = np.divide(acc, totals, out=np.zeros_like(acc), where=totals > 0)

    np.savez_compressed(REACH_PIXEL_PATH, weights=weights.astype(np.float32),
                        n_lat=np.int32(len(lats)), n_lon=np.int32(len(lons)))
    log.info("wrote   %s (%d reaches x %d rainfall pixels)",
             REACH_PIXEL_PATH.name, n_reach, n_pixel)
    return REACH_PIXEL_PATH
