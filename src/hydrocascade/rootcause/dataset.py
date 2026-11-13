"""Assemble the (unit x day) design matrix — Doc 06 §1.

The target is *"the probability that at least one event of type i occurs in
this unit during a given day, given the physical conditions"* — a binary
classification with time-varying covariates, not a classical time-independent
susceptibility map.

Two variants come out of the same assembly, and both are needed (Doc 06 §1):

``S-static``
    terrain only. Produces ``S_i(u)``, the multiplier used inside the Phase 4
    background. It must not see rainfall, or the "susceptibility" it learns is
    really a weather forecast and the background double-counts the rain.
``S-dynamic``
    terrain + rainfall windows + the antecedent index. This is **baseline B3**,
    the strongest non-point-process competitor the cascade model must beat.

Building the full matrix is never necessary: 17,138 units x 2,406 days is 41
million rows to describe a few thousand events. Negative sampling reduces it to
a few thousand, which is the whole point of doing the sampling first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("rootcause.dataset")

RAIN_COLUMNS = ("R1", "R3", "R7", "R15", "R30", "API")


def _rain_cube():
    """Rainfall as ``(var -> [time, lat, lon])`` plus the index lookups."""
    import xarray as xr

    p = paths.SPATIAL / "rainfall_pixels.nc"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade rainfall`")
    ds = xr.open_dataset(p).load()
    cube = {v: ds[v].transpose("time", "latitude", "longitude").to_numpy()
            for v in RAIN_COLUMNS if v in ds}
    day_of = {pd.Timestamp(t).normalize(): i
              for i, t in enumerate(pd.DatetimeIndex(ds.time.to_numpy()))}
    ds.close()
    return cube, day_of


def _unit_pixel() -> pd.DataFrame:
    p = paths.SPATIAL / "unit_rain_pixel.parquet"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade rainfall`")
    return pd.read_parquet(p)


def attach_rainfall(frame: pd.DataFrame, *, unit_column: str = "unit_id"
                    ) -> pd.DataFrame:
    """Add R1..R30 and API for each (unit, day) row.

    Rows whose day falls outside the rainfall record get NaN rather than a
    forward-filled value. A silently carried-forward rainfall total is a
    covariate that lies about a day nobody measured.
    """
    cube, day_of = _rain_cube()
    lut = _unit_pixel().set_index("unit_id")

    lat_i = lut["lat_index"].reindex(frame[unit_column]).to_numpy()
    lon_i = lut["lon_index"].reindex(frame[unit_column]).to_numpy()
    days = pd.to_datetime(frame["date"]).dt.normalize()
    t_i = np.array([day_of.get(d, -1) for d in days])

    ok = (t_i >= 0) & np.isfinite(lat_i) & np.isfinite(lon_i)
    out = frame.copy()
    for name, arr in cube.items():
        col = np.full(len(frame), np.nan)
        if ok.any():
            col[ok] = arr[t_i[ok], lat_i[ok].astype(int), lon_i[ok].astype(int)]
        out[name] = col

    missing = int((~ok).sum())
    if missing:
        log.warning("%d of %d rows fall outside the rainfall record and carry "
                    "NaN rainfall", missing, len(frame))
    return out


def build_landslide(cfg: Node, *, ratio: int = 5, seed: int | None = None,
                    dynamic: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """The landslide design matrix, plus the covariate names to model on."""
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..features import terrain_covariates as tc
    from . import sampling

    seed = int(cfg.project.seed) if seed is None else int(seed)
    cov = tc.load()

    events = schema.read(EVENTS_PATH)
    land = events[events["hazard_type"] == "L"].copy()
    land["unit_id"] = pd.to_numeric(land["spatial_unit_id"], errors="coerce")
    land = land.dropna(subset=["unit_id"])
    land["unit_id"] = land["unit_id"].astype(int)

    dates = pd.date_range(str(cfg.period.start), str(cfg.period.end), freq="D")

    import geopandas as gpd
    g = gpd.read_file(paths.SPATIAL / "slope_units.gpkg",
                      columns=["unit_id", "rep_x", "rep_y"], ignore_geometry=True)
    g = g.set_index("unit_id").reindex(cov["unit_id"])
    xy = (g["rep_x"].to_numpy(), g["rep_y"].to_numpy())

    s = sampling.sample(land[["unit_id", "date"]], cov, dates,
                        ratio=ratio, seed=seed, xy=xy)

    df = s.frame.merge(cov, on="unit_id", how="left")
    features = list(tc.STATIC_COVARIATES)
    if dynamic:
        df = attach_rainfall(df)
        features += [c for c in RAIN_COLUMNS if c in df.columns]

    df.attrs["sampling"] = s.summary()
    df.attrs["n_positive"] = s.n_positive
    log.info("landslide matrix: %d rows x %d features (%s)",
             len(df), len(features), "S-dynamic" if dynamic else "S-static")
    return df, features


def build_flood(cfg: Node, *, ratio: int = 5, seed: int | None = None,
                dynamic: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """The flood design matrix, on reaches rather than slope units.

    Doc 05 §7 is emphatic that a reach's rainfall covariate is the **catchment
    average**, not the rain falling on the reach itself: *"rain on the hills is
    what floods the river."* The catchment weights were precomputed in Phase 2.
    """
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..features import terrain_covariates as tc
    from . import sampling

    seed = int(cfg.project.seed) if seed is None else int(seed)
    cov = tc.load_reaches().rename(columns={"reach_id": "unit_id"})

    events = schema.read(EVENTS_PATH)
    flood = events[events["hazard_type"] == "F"].copy()
    flood["unit_id"] = pd.to_numeric(flood["spatial_unit_id"], errors="coerce")
    flood = flood.dropna(subset=["unit_id"])
    flood["unit_id"] = flood["unit_id"].astype(int)

    # Only reaches the flood rule can actually see are candidates; sampling
    # negatives from ineligible headwaters would teach the model that small
    # catchments never flood, which is an artefact of the detector.
    min_area = float(cfg.floods.min_contributing_area_km2)
    cov = cov[cov["area_dn_km2"] > min_area].reset_index(drop=True)
    flood = flood[flood["unit_id"].isin(set(cov["unit_id"]))]

    dates = pd.date_range(str(cfg.period.start), str(cfg.period.end), freq="D")

    # Doc 06 §2 rule 2a applies to reaches too. A reach 500 m from one that
    # flooded is not a clean negative: the detector's eligibility filter and
    # the 500 m reach length together mean the neighbour was very likely wet on
    # the same day and simply did not clear its own Q2. Each reach is located
    # at its downstream end, which is where the flood event is recorded.
    import geopandas as gpd
    net = gpd.read_file(paths.SPATIAL / "channel_network.gpkg", layer="reaches")
    ends = net.set_index("reach_id").geometry.interpolate(1.0, normalized=True)
    ends = ends.reindex(cov["unit_id"])
    xy = (ends.x.to_numpy(), ends.y.to_numpy())

    s = sampling.sample(flood[["unit_id", "date"]], cov, dates,
                        ratio=ratio, seed=seed,
                        elevation_column="z_dn_m", xy=xy)

    df = s.frame.merge(cov, on="unit_id", how="left")
    features = [c for c in cov.columns if c != "unit_id"]
    if dynamic:
        df = _attach_catchment_rainfall(df)
        features += [c for c in RAIN_COLUMNS if c in df.columns]

    df.attrs["sampling"] = s.summary()
    df.attrs["n_positive"] = s.n_positive
    log.info("flood matrix: %d rows x %d features (%s)",
             len(df), len(features), "S-dynamic" if dynamic else "S-static")
    return df, features


def _attach_catchment_rainfall(frame: pd.DataFrame) -> pd.DataFrame:
    """Catchment-averaged rainfall per reach, from the Phase 2 weights."""
    p = paths.SPATIAL / "reach_rain_weights.npz"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run `hydrocascade rainfall`")
    z = np.load(p, allow_pickle=False)
    # Row index IS the reach id; the column index is the rainfall pixel
    # flattened as lat_index * n_lon + lon_index, matching how Phase 2 built it.
    weights = z["weights"].astype(np.float64)        # (n_reach, n_lat * n_lon)
    n_lat, n_lon = int(z["n_lat"]), int(z["n_lon"])
    n_reach = weights.shape[0]

    cube, day_of = _rain_cube()

    days = pd.to_datetime(frame["date"]).dt.normalize()
    t_i = np.array([day_of.get(d, -1) for d in days])
    r_i = pd.to_numeric(frame["unit_id"], errors="coerce").fillna(-1).to_numpy().astype(int)
    ok = (t_i >= 0) & (r_i >= 0) & (r_i < n_reach)

    out = frame.copy()
    for name, arr in cube.items():
        flat = arr.reshape(arr.shape[0], n_lat * n_lon)        # (time, pixel)

        # CHIRPS is a land product: 15 of these 63 pixels are Arabian Sea and
        # are NaN for every day. A catchment is entirely on land, so those
        # pixels carry weight zero — but `0 * NaN` is NaN, so a plain dot
        # product returns NaN for *every* reach. That silently pushed all five
        # accumulation windows over the 20% missingness threshold and deleted
        # them from the flood model, which is how a land/sea mask turns into a
        # missing covariate.
        #
        # The fix is a weighted mean over the pixels that have data, with the
        # weights renormalised per reach so the average stays an average.
        valid = np.isfinite(flat)
        filled = np.where(valid, flat, 0.0)

        col = np.full(len(frame), np.nan)
        if ok.any():
            w = weights[r_i[ok]]                               # (rows, pixel)
            v = valid[t_i[ok]]
            num = np.einsum("ij,ij->i", w, filled[t_i[ok]])
            den = np.einsum("ij,ij->i", w, v.astype(np.float64))
            col[ok] = np.divide(num, den, out=np.full(num.shape, np.nan),
                                where=den > 1e-12)
        out[name] = col

    if (~ok).any():
        log.warning("%d reach-days have no catchment rainfall", int((~ok).sum()))
    dropped = {n: float(pd.isna(out[n]).mean()) for n in cube}
    worst = max(dropped, key=dropped.get)
    if dropped[worst] > 0.01:
        log.warning("catchment rainfall still %.1f%% missing (worst: %s) — a "
                    "reach whose whole catchment falls on no-data pixels",
                    100 * dropped[worst], worst)
    return out
