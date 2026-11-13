"""Static covariates per slope unit and per reach — Doc 05 §6.

Everything here is derived from the DEM and the channel network, so it needs no
account and no download. The two covariate families Doc 05 §6 lists that are
*not* derivable that way — SoilGrids texture/depth/organic-carbon and ESA
WorldCover land cover — have adapters in :mod:`hydrocascade.ingest.adapters`
and are reported missing rather than silently imputed. A susceptibility model
that quietly invents soil depth is worse than one that admits it has none.

The curvature formulation is Zevenbergen–Thorne (1987), which fits a partial
quartic to the 3x3 neighbourhood. It is the standard in the landslide
susceptibility literature, so our numbers are comparable to published ones.

Aspect is **decomposed into northness and eastness** rather than reported as a
bearing. A model given raw degrees learns that 359 deg and 1 deg are maximally
different, which is the opposite of the truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("features.terrain_covariates")

COVARIATE_PATH = paths.SPATIAL / "unit_covariates.parquet"
REACH_COVARIATE_PATH = paths.SPATIAL / "reach_covariates.parquet"

#: Covariates computed here, in the order they enter the feature table.
STATIC_COVARIATES = (
    "elev_mean_m", "elev_max_m", "relief_m",
    "slope_mean_deg", "slope_max_deg",
    "northness", "eastness",
    "curv_plan", "curv_prof",
    "twi", "log_flow_acc",
    "dist_stream_m", "d_slope_km",
    "drainage_density_km_km2",
    "area_km2", "injection_area_km2",
)


# ------------------------------------------------------------------ kernels ---

def _zevenbergen_thorne(dem: np.ndarray, cellsize: float):
    """Plan and profile curvature by the Zevenbergen-Thorne partial quartic.

    Returns ``(plan, profile)`` in units of 1/m. Positive profile curvature is
    convex (accelerating flow); positive plan curvature is divergent (a nose).
    Landslides concentrate where profile curvature is concave and plan
    curvature is convergent — the hollow that collects both water and debris.

    The grid is padded **once** and the nine neighbours are taken as views into
    that one copy. Padding per neighbour instead would allocate nine full grids,
    which on a 9-million-cell DEM is about 650 MB of pure waste.
    """
    L = float(cellsize)
    R, C = dem.shape
    p = np.pad(dem.astype(np.float32), 1, mode="edge")

    def w(dr: int, dc: int) -> np.ndarray:          # a view, not a copy
        return p[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]

    z5 = w(0, 0)
    D = ((w(0, -1) + w(0, 1)) * 0.5 - z5) / L ** 2
    E = ((w(-1, 0) + w(1, 0)) * 0.5 - z5) / L ** 2
    F = (-w(-1, -1) + w(-1, 1) + w(1, -1) - w(1, 1)) / (4.0 * L ** 2)
    G = (-w(0, -1) + w(0, 1)) / (2.0 * L)
    H = (w(-1, 0) - w(1, 0)) / (2.0 * L)

    denom = G ** 2 + H ** 2
    safe = np.maximum(denom, 1e-12)
    # Zevenbergen-Thorne: plan = 2(DH^2 + EG^2 - FGH) / (G^2 + H^2)
    #                     prof = -2(DG^2 + EH^2 + FGH) / (G^2 + H^2)
    plan = 2.0 * (D * H ** 2 + E * G ** 2 - F * G * H) / safe
    prof = -2.0 * (D * G ** 2 + E * H ** 2 + F * G * H) / safe
    flat = denom < 1e-12          # a perfectly flat cell has no curvature
    plan[flat] = 0.0
    prof[flat] = 0.0
    del p, D, E, F, G, H, denom, safe, flat
    return plan, prof


def _aspect_components(dem: np.ndarray, cellsize: float):
    """Northness and eastness: cos(aspect) and sin(aspect), weighted by slope.

    On a flat cell aspect is undefined, so both components are zero — which is
    the honest encoding, and it is what a bearing cannot express.
    """
    gy, gx = np.gradient(dem.astype(np.float32), cellsize, cellsize)
    # aspect measured clockwise from north; dz/dy points south-to-north
    aspect = np.arctan2(-gx, gy)
    mag = np.hypot(gx, gy)
    flat = mag < 1e-9
    north = np.cos(aspect)
    east = np.sin(aspect)
    north[flat] = 0.0
    east[flat] = 0.0
    return north, east


def _twi(flow_acc_cells: np.ndarray, slope_deg: np.ndarray, cellsize: float):
    """``ln(a / tan beta)`` with ``a`` the specific catchment area.

    ``a = A / L`` where ``A`` is the upslope contributing area and ``L`` the
    contour length, taken as one cell width. Slope is floored at 0.1 deg: a
    literal zero makes the index infinite, and a cell that flat drains through
    the flat-resolution gradient anyway.
    """
    a = np.maximum(flow_acc_cells, 1.0) * cellsize          # m2 per m of contour
    tan_beta = np.tan(np.radians(np.maximum(slope_deg, 0.1)))
    return np.log(a / tan_beta)


# -------------------------------------------------------------------- build ---

def build(cfg: Node, *, force: bool = False) -> Path:
    """Compute per-unit and per-reach static covariates."""
    if COVARIATE_PATH.is_file() and REACH_COVARIATE_PATH.is_file() and not force:
        log.info("cached  %s", COVARIATE_PATH.name)
        return COVARIATE_PATH

    import rasterio

    from ..spatial.network import Network
    from ..spatial.slope_units import SlopeUnits
    from ..spatial.terrain import Routing

    rt = Routing()
    net = Network()
    units = SlopeUnits()
    cell = float(rt.cellsize)

    # float32 throughout: this grid has 9.07 million cells, so every float64
    # intermediate is 72 MB. Covariates feed a tree ensemble, not a likelihood,
    # so single precision is far more than the model can use.
    with rasterio.open(paths.SPATIAL / "dem_filled.tif") as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
    with rasterio.open(paths.SPATIAL / "slope_deg.tif") as src:
        slope = src.read(1).astype(np.float32)

    valid = np.isfinite(dem) if nodata is None else (dem != nodata) & np.isfinite(dem)

    # Every derivative here is a 3x3 stencil, so a valid cell sitting next to
    # nodata reads a nodata neighbour and produces garbage — or NaN, which then
    # poisons the unit mean. The AOI is an irregular basin polygon inside a
    # rectangular grid, so this is the whole boundary, not a rare edge case.
    #
    # Nearest-valid fill before differencing is the standard treatment: it
    # makes the boundary behave like the grid edge already does under
    # mode="edge", and the results are masked back to valid cells before any
    # aggregation, so the fill never enters a covariate itself.
    if not valid.all():
        nearest = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True)
        dem = dem[tuple(nearest)]
        slope = slope[tuple(nearest)]
        del nearest

    log.info("curvature, aspect and TWI over %d x %d cells", *dem.shape)
    plan, prof = _zevenbergen_thorne(dem, cell)
    north, east = _aspect_components(dem, cell)
    acc_cells = np.asarray(rt.acc, dtype=np.float32).reshape(dem.shape)
    twi = _twi(acc_cells, slope, cell)

    # Euclidean distance to the nearest channel cell: Doc 05 §6 calls this w(x)
    # and Doc 02 §5 uses it for the flood -> landslide corridor. It is NOT
    # d_slope, which is measured along the flow path and is always longer.
    channel = net.is_channel.reshape(dem.shape)
    dist_stream = ndimage.distance_transform_edt(
        ~channel, sampling=cell).astype(np.float32)

    # Drainage density in a 2 km neighbourhood: channel length per unit area.
    # Each channel cell contributes about one cell width of channel.
    #
    # `uniform_filter` is separable and runs in O(N) regardless of window size.
    # A direct convolution with the equivalent 133x133 kernel would be roughly
    # 1.6e11 multiply-adds over this grid — minutes of CPU for a box mean.
    size = 2 * max(int(round(2000.0 / cell)), 1) + 1
    chan_frac = ndimage.uniform_filter(channel.astype(np.float32), size=size,
                                       mode="constant", cval=0.0)
    valid_frac = ndimage.uniform_filter(valid.astype(np.float32), size=size,
                                        mode="constant", cval=0.0)
    # length km / area km2, with the window cell-count cancelling out
    dens = np.divide(chan_frac * cell / 1000.0,
                     valid_frac * cell ** 2 / 1e6,
                     out=np.zeros_like(chan_frac), where=valid_frac > 1e-6)
    del chan_frac, valid_frac

    # --- aggregate to slope units -------------------------------------------
    unit_of_cell = units.unit_of_cell            # flat, -1 outside any unit
    n_units = len(units.gdf)
    flat_valid = valid.ravel() & (unit_of_cell >= 0)
    idx = unit_of_cell[flat_valid]

    def per_unit(field: np.ndarray, how: str = "mean") -> np.ndarray:
        v = field.ravel()[flat_valid]
        if how == "mean":
            s = np.bincount(idx, weights=v, minlength=n_units)
            n = np.bincount(idx, minlength=n_units)
            return np.divide(s, n, out=np.zeros(n_units), where=n > 0)
        if how == "max":
            out = np.full(n_units, -np.inf)
            np.maximum.at(out, idx, v)
            return np.where(np.isfinite(out), out, 0.0)
        raise ValueError(how)

    g = units.gdf
    table = pd.DataFrame({
        "unit_id": g["unit_id"].to_numpy(),
        "elev_mean_m": g["elev_mean_m"].to_numpy(),
        "elev_max_m": g["elev_max_m"].to_numpy(),
        "slope_mean_deg": g["slope_mean_deg"].to_numpy(),
        "slope_max_deg": g["slope_max_deg"].to_numpy(),
        "area_km2": g["area_km2"].to_numpy(),
        "d_slope_km": g["d_slope_km"].to_numpy(),
        "northness": per_unit(north),
        "eastness": per_unit(east),
        "curv_plan": per_unit(plan),
        "curv_prof": per_unit(prof),
        "twi": per_unit(twi),
        "log_flow_acc": per_unit(np.log10(np.maximum(acc_cells, 1.0))),
        "dist_stream_m": per_unit(dist_stream),
        "drainage_density_km_km2": per_unit(dens),
    })
    table["relief_m"] = table["elev_max_m"] - table["elev_mean_m"]

    inj = g["injection_reach_id"].to_numpy()
    reach_area = net.gdf.set_index("reach_id")["area_dn_km2"]
    table["injection_area_km2"] = [
        float(reach_area.get(int(r), np.nan)) if r >= 0 else np.nan for r in inj]

    table = table[["unit_id", *STATIC_COVARIATES]]

    # A NaN covariate is a boundary bug, not a missing measurement: every unit
    # is inside the grid by construction. Fail loudly rather than let an
    # imputer downstream quietly invent a curvature.
    bad = table[list(STATIC_COVARIATES)].isna().sum()
    if bad.any():
        raise RuntimeError(
            "NaN in static covariates after the nearest-valid fill: "
            + ", ".join(f"{k}={int(v)}" for k, v in bad[bad > 0].items()))

    table.to_parquet(COVARIATE_PATH, index=False)
    log.info("wrote %s: %d units x %d covariates",
             COVARIATE_PATH.name, len(table), len(STATIC_COVARIATES))

    # --- reach-side covariates ----------------------------------------------
    # A reach's terrain is the terrain of the catchment that drains into it, so
    # the unit-level means are aggregated over each reach's injecting units.
    by_reach = table.assign(reach=inj)
    by_reach = by_reach[by_reach["reach"] >= 0]
    agg = by_reach.groupby("reach")[list(STATIC_COVARIATES)].mean()

    r = net.gdf.set_index("reach_id")
    rtab = pd.DataFrame(index=r.index)
    rtab["area_dn_km2"] = r["area_dn_km2"]
    rtab["length_m"] = r["length_m"]
    rtab["slope"] = r["slope"]
    rtab["strahler"] = r["strahler"]
    rtab["z_dn_m"] = r["z_dn_m"]
    for c in ("slope_mean_deg", "twi", "curv_prof", "elev_mean_m",
              "drainage_density_km_km2"):
        rtab[f"catch_{c}"] = agg[c].reindex(rtab.index)
    rtab = rtab.reset_index().rename(columns={"index": "reach_id"})
    rtab.to_parquet(REACH_COVARIATE_PATH, index=False)
    log.info("wrote %s: %d reaches x %d covariates",
             REACH_COVARIATE_PATH.name, len(rtab), rtab.shape[1] - 1)

    _warn_about_missing_sources(cfg)
    return COVARIATE_PATH


def _warn_about_missing_sources(cfg: Node) -> None:
    """Doc 05 §6 lists two covariate families we cannot derive from the DEM."""
    log.warning("soil texture/depth/organic carbon (SoilGrids 250 m) and land "
                "cover (ESA WorldCover 10 m) are NOT in the covariate stack. "
                "Both are free and login-free; see ingest/adapters.py. Until "
                "they are downloaded the susceptibility model is terrain- and "
                "rainfall-only, and reports/phase3_features.md says so.")


def load() -> pd.DataFrame:
    if not COVARIATE_PATH.is_file():
        raise FileNotFoundError(
            f"{COVARIATE_PATH} missing — run `hydrocascade covariates`")
    return pd.read_parquet(COVARIATE_PATH)


def load_reaches() -> pd.DataFrame:
    if not REACH_COVARIATE_PATH.is_file():
        raise FileNotFoundError(
            f"{REACH_COVARIATE_PATH} missing — run `hydrocascade covariates`")
    return pd.read_parquet(REACH_COVARIATE_PATH)
