"""Adapters for the four login-gated datasets (Doc 10 §7).

GloFAS and ERA5-Land need a Copernicus CDS account, IMERG needs NASA Earthdata,
and IMD needs a registration form. Doc 10 §7 says to register on day 4 of week 1
precisely because waiting for approval in week 6 is an avoidable delay — but the
pipeline should not be *blocked* on an approval email either.

So each gated source has two implementations behind one function:

``real``
    The actual download, written and ready. Raises a clear, actionable error if
    the credential file is absent.
``synthetic``
    A generator that produces data of the right shape, units and statistical
    character, clearly labelled as synthetic everywhere it lands.

Which one runs is ``data_mode.<source>`` in the config. For GloFAS, ``real``
reads files from ``data/raw/GLOFAS/`` (or downloads them with a CDS key).
IMD and IMERG are chosen by ``rainfall.primary`` instead
(ingest/rainfall_sources.py). See docs/15_HOW_TO_RUN.md.

**The synthetic data is for exercising the pipeline, never for a reported
number.** Every artefact derived from it carries ``synthetic=True`` and the
quality gates refuse to pass a catalog built on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("ingest.adapters")


class CredentialsMissing(RuntimeError):
    """A real-mode adapter was asked to run without its credentials."""


def _require_cdsapirc() -> Path:
    p = Path.home() / ".cdsapirc"
    if not p.is_file():
        raise CredentialsMissing(
            "GloFAS/ERA5-Land need a Copernicus CDS account.\n"
            "  1. register at https://cds.climate.copernicus.eu\n"
            "  2. copy your key into ~/.cdsapirc\n"
            "  3. pip install cdsapi\n"
            "Until then set data_mode.glofas: synthetic in config/default.yaml.")
    return p


# --------------------------------------------------------------- discharge ---

def discharge(cfg: Node, reaches: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Daily discharge, indexed by date, one column per eligible reach.

    ``reaches`` needs columns ``reach_id`` and ``area_dn_km2``.
    """
    mode = str(cfg.data_mode.glofas)
    if mode == "real":
        return _discharge_glofas(cfg, reaches, force=force)
    if mode == "synthetic":
        return _discharge_synthetic(cfg, reaches)
    raise ValueError(f"data_mode.glofas must be 'real' or 'synthetic', got {mode!r}")


#: Where GloFAS files are read from, and the variable names the CDS has used.
GLOFAS_RAW = "GLOFAS"
GLOFAS_VARS = ("dis24", "river_discharge_in_the_last_24_hours", "dis")
GLOFAS_UPAREA_VARS = ("uparea", "upArea", "upstream_area")
#: A GloFAS cell is accepted for a reach only if its upstream area is within
#: this factor of the reach's own. Outside it, the cell is on another river.
GLOFAS_AREA_FACTOR = 2.0
#: Search radius for the matching cell: 0.05 deg grid, so ~3 cells each way.
GLOFAS_SEARCH_DEG = 0.15


def _discharge_glofas(cfg: Node, reaches: pd.DataFrame, *, force: bool = False):
    """GloFAS v4.0 historical reanalysis (dataset H1, Doc 10 §5).

    Reads netCDF files from the newest ``data/raw/GLOFAS/<YYYY-MM-DD>/``
    folder that has them. If there are none and ``~/.cdsapirc`` exists, it
    downloads them there first (``glofas_download``); otherwise it stops with
    the instructions. Either way the result is the same ``date x reach`` table
    the synthetic generator returns, so ``ingest/floods.py`` is untouched.

    **Matching a reach to a GloFAS cell.** GloFAS is a 0.05 deg (~5 km) grid,
    so the cell nearest a reach's outlet is often on a neighbouring river. If
    the files carry an upstream-area field (``uparea``, the CDS auxiliary file)
    the cell within ``GLOFAS_SEARCH_DEG`` whose area is closest to the reach's
    is used, and a reach with no cell within ``GLOFAS_AREA_FACTOR`` is dropped.
    Without the field the fallback is the wettest cell in the window — biased
    towards the main stem, and logged as such.
    """
    cache = paths.INTERIM / "glofas_discharge.parquet"
    if cache.is_file() and not force:
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        df.columns = [int(c) for c in df.columns]
        df.attrs["synthetic"] = False
        log.info("GloFAS discharge from cache: %d days x %d reaches", *df.shape)
        return df

    folder = _glofas_folder(cfg)
    src = _glofas_open(folder, cfg)
    df = _glofas_sample(src, cfg, reaches)
    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(cache)
    df.attrs["synthetic"] = False
    return df


def _glofas_folder(cfg: Node) -> Path:
    root = paths.RAW / GLOFAS_RAW
    dated = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True) if root.is_dir() else []
    for d in dated:
        if any(d.glob("*.nc")) or any(d.glob("*.nc4")):
            if not (d / "PROVENANCE.md").is_file():
                raise FileNotFoundError(
                    f"{d} has GloFAS files but no PROVENANCE.md. Write one (template in "
                    f"docs/15_HOW_TO_RUN.md) — the catalog refuses unsourced data.")
            return d
    if (Path.home() / ".cdsapirc").is_file():
        return glofas_download(cfg)
    raise CredentialsMissing(
        "data_mode.glofas is 'real' but there are no GloFAS files and no CDS key.\n"
        f"  either put the netCDF files in {root}/<YYYY-MM-DD>/ with a PROVENANCE.md\n"
        "  or create ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api)\n"
        "     and run `hydrocascade catalog` again — it will download them.\n"
        "See docs/15_HOW_TO_RUN.md, step 3.")


def glofas_download(cfg: Node) -> Path:
    """Fetch GloFAS v4 daily discharge for the study bbox, one file per year.

    Request per Doc 04 §4.1 step 1. Each year is a few MB for this bbox; the
    CDS queue is the slow part (minutes to hours per request). Years already
    on disk are skipped, so an interrupted run resumes.
    """
    _require_cdsapirc()
    try:
        import cdsapi
    except ImportError as exc:
        raise CredentialsMissing("pip install cdsapi to download GloFAS") from exc
    from datetime import date

    from ..config import bbox
    w, s, e, n = bbox(cfg)
    area = [round(n + 0.1, 2), round(w - 0.1, 2), round(s - 0.1, 2), round(e + 0.1, 2)]
    folder = paths.RAW / GLOFAS_RAW / date.today().isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    first = 1980
    last = pd.Timestamp(str(cfg.period.end)).year
    client = cdsapi.Client()
    for year in range(first, last + 1):
        target = folder / f"glofas_dis24_{year}.nc"
        if target.is_file():
            continue
        log.info("GloFAS: requesting %d (the CDS queue can take a while)", year)
        client.retrieve("cems-glofas-historical", {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": [str(year)],
            "hmonth": [f"{m:02d}" for m in range(1, 13)],
            "hday": [f"{d:02d}" for d in range(1, 32)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": area,
        }, str(target))
    prov = folder / "PROVENANCE.md"
    if not prov.is_file():
        prov.write_text(
            "# GloFAS v4.0 historical reanalysis\n\n"
            "- dataset: cems-glofas-historical (Copernicus CDS/EWDS)\n"
            "- url: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical\n"
            f"- variable: river_discharge_in_the_last_24_hours, {first}-{last}\n"
            f"- area N/W/S/E: {area}\n"
            f"- downloaded: {date.today().isoformat()} by hydrocascade (glofas_download)\n"
            "- licence: Copernicus licence, attribution required\n",
            encoding="utf-8")
    return folder


def _glofas_open(folder: Path, cfg: Node) -> dict:
    import xarray as xr

    from ..config import bbox
    files = sorted(list(folder.glob("*.nc")) + list(folder.glob("*.nc4")))
    parts, uparea = [], None
    w, s, e, n = bbox(cfg)
    for f in files:
        ds = xr.open_dataset(f)
        ren = {}
        for std, names in (("latitude", ("latitude", "lat")),
                           ("longitude", ("longitude", "lon")),
                           ("time", ("time", "valid_time"))):
            for nm in names:
                if nm in ds.dims or nm in ds.coords:
                    if nm != std:
                        ren[nm] = std
                    break
        ds = ds.rename(ren)
        lat_slice = (slice(n + 0.2, s - 0.2) if ds.latitude[0] > ds.latitude[-1]
                     else slice(s - 0.2, n + 0.2))
        ds = ds.sel(latitude=lat_slice, longitude=slice(w - 0.2, e + 0.2))
        up = next((v for v in GLOFAS_UPAREA_VARS if v in ds.data_vars), None)
        if up is not None:
            ua = ds[up]
            if "time" in ua.dims:
                ua = ua.isel(time=0, drop=True)
            uparea = ua.load()
        var = next((v for v in GLOFAS_VARS if v in ds.data_vars), None)
        if var is None:
            if up is None:
                log.warning("GloFAS: %s has none of %s, skipped", f.name, GLOFAS_VARS)
            continue
        da = ds[var]
        extra = [d for d in da.dims if d not in ("time", "latitude", "longitude")]
        if extra:
            da = da.isel({d: 0 for d in extra}, drop=True)
        parts.append(da.load())
    if not parts:
        raise FileNotFoundError(f"no discharge variable {GLOFAS_VARS} in {folder}")
    dis = xr.concat(parts, dim="time").sortby("time")
    dis = dis.isel(time=~dis.indexes["time"].duplicated())
    dis["time"] = pd.to_datetime(dis["time"].values).normalize()
    if uparea is not None and float(uparea.max()) > 1e6:     # m2 -> km2
        uparea = uparea / 1e6
    log.info("GloFAS: %d days %s..%s, grid %dx%d, upstream area %s",
             dis.sizes["time"], str(dis.time.values[0])[:10], str(dis.time.values[-1])[:10],
             dis.sizes["latitude"], dis.sizes["longitude"],
             "present" if uparea is not None else "ABSENT (fallback: wettest cell)")
    return {"dis": dis, "uparea": uparea}


def _reach_outlets_lonlat(reaches: pd.DataFrame) -> pd.DataFrame:
    """Downstream end of each reach in lon/lat. Geometry runs upstream -> downstream."""
    import geopandas as gpd
    if isinstance(reaches, gpd.GeoDataFrame):
        g = reaches
    else:
        from ..spatial.network import Network
        g = (Network().gdf.set_index("reach_id")
             .loc[reaches["reach_id"].to_numpy()].reset_index())
    ends = gpd.GeoSeries(gpd.points_from_xy([geom.coords[-1][0] for geom in g.geometry],
                                            [geom.coords[-1][1] for geom in g.geometry]),
                         crs=g.crs).to_crs(4326)
    return pd.DataFrame({"reach_id": g["reach_id"].to_numpy(),
                         "lon": ends.x.to_numpy(), "lat": ends.y.to_numpy()})


def _glofas_sample(src: dict, cfg: Node, reaches: pd.DataFrame) -> pd.DataFrame:
    dis, uparea = src["dis"], src["uparea"]
    min_area = float(cfg.floods.min_contributing_area_km2)
    elig = reaches[reaches["area_dn_km2"] > min_area].copy()
    outlets = _reach_outlets_lonlat(elig).set_index("reach_id")
    lats, lons = dis.latitude.values, dis.longitude.values
    ua = uparea.values if uparea is not None else None
    mean_q = dis.mean("time").values if ua is None else None

    out, dropped = {}, 0
    for rid, area in zip(elig["reach_id"].to_numpy(), elig["area_dn_km2"].to_numpy()):
        lon, lat = outlets.loc[rid, "lon"], outlets.loc[rid, "lat"]
        iy = np.where(np.abs(lats - lat) <= GLOFAS_SEARCH_DEG)[0]
        ix = np.where(np.abs(lons - lon) <= GLOFAS_SEARCH_DEG)[0]
        if iy.size == 0 or ix.size == 0:
            dropped += 1
            continue
        if ua is not None:
            win = ua[np.ix_(iy, ix)].astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.abs(np.log(np.where(win > 0, win, np.nan) / area))
            if not np.isfinite(ratio).any() or np.nanmin(ratio) > np.log(GLOFAS_AREA_FACTOR):
                dropped += 1
                continue
            k = int(np.nanargmin(ratio))
        else:
            win = mean_q[np.ix_(iy, ix)]
            if not np.isfinite(win).any():
                dropped += 1
                continue
            k = int(np.nanargmax(win))
        j, i = np.unravel_index(k, win.shape)
        out[int(rid)] = dis.isel(latitude=int(iy[j]), longitude=int(ix[i])).values.astype(np.float32)
    if not out:
        raise RuntimeError("no reach could be matched to a GloFAS cell — check the files' bbox")
    df = pd.DataFrame(out, index=pd.DatetimeIndex(dis.time.values))
    log.info("GloFAS discharge table: %d days x %d reaches (%d had no matching cell)",
             df.shape[0], df.shape[1], dropped)
    return df


def _discharge_synthetic(cfg: Node, reaches: pd.DataFrame) -> pd.DataFrame:
    """A physically shaped stand-in for GloFAS discharge.

    Not a hydrological model and not calibrated to anything. It exists so the
    Q2/declustering machinery, the catalog schema and the figures can be
    exercised end to end before the CDS account arrives. The construction is:

    * a regional daily storm series with south-west monsoon seasonality
      (June-September) and heavy-tailed magnitudes;
    * a **per-reach annual factor**, so catchments do not all run wet or dry in
      the same year. Without it a single shared storm series makes entire years
      empty across all 1,800 reaches at once, which is not how a 3,000 km2 basin
      behaves;
    * **elevated weights on the monsoon years the record documents as extreme**
      — 2018, 2019 and 2024 — so the stand-in is not silent in exactly the years
      the project is about. Doc 01 §2 notes the August 2018 monsoon alone
      produced roughly 4,700 mapped landslides, and 30 July 2024 is the
      held-out test event;
    * routed through a per-reach linear reservoir whose memory grows with
      catchment area, so big rivers have smoother, later hydrographs;
    * scaled as ``A^0.85``, the usual empirical discharge-area exponent.

    The result has realistic annual maxima, which is all the Gumbel fit needs.
    """
    seed = int(cfg.project.seed)
    rng = np.random.default_rng(seed)

    min_area = float(cfg.floods.min_contributing_area_km2)
    elig = reaches[reaches["area_dn_km2"] > min_area].copy()
    if elig.empty:
        raise RuntimeError(f"no reach above {min_area} km2 to synthesise discharge for")

    # A long record so the Gumbel fit has enough annual maxima, matching the
    # 1980- span of the real GloFAS reanalysis.
    dates = pd.date_range("1980-01-01", str(cfg.period.end), freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    # south-west monsoon: broad June-September peak, plus a weaker NE monsoon
    season = (np.exp(-0.5 * ((doy - 200) / 38.0) ** 2)
              + 0.35 * np.exp(-0.5 * ((doy - 300) / 25.0) ** 2))
    season = 0.05 + season

    # Years the Kerala record documents as exceptional monsoons. Weighting them
    # keeps the stand-in from being silent in the years the project studies.
    year_weight = {2018: 1.45, 2019: 1.35, 2020: 1.05, 2021: 1.15, 2024: 1.40}
    years = dates.year.to_numpy()
    day_weight = np.array([year_weight.get(int(y), 1.0) for y in years])

    # Storm occurrence and magnitude. The year weight is applied at its square
    # root to occurrence and in full to magnitude: applying it fully to both
    # compounds into roughly twice the exceedance rate a Q2 threshold should
    # produce (~1 event per reach-year).
    storms = rng.random(n) < (0.26 * season * np.sqrt(day_weight))
    magnitude = np.zeros(n)
    k = int(storms.sum())
    magnitude[storms] = rng.lognormal(mean=0.0, sigma=1.05, size=k)
    magnitude *= day_weight

    # the held-out event itself, so replay has something to show on the day
    wayanad = pd.Timestamp(str(cfg.wayanad_event.date))
    if wayanad in dates:
        pos = dates.get_loc(wayanad)
        magnitude[max(pos - 1, 0):pos + 2] = np.maximum(
            magnitude[max(pos - 1, 0):pos + 2], np.array([6.0, 9.0, 5.0])[:pos + 2 - max(pos - 1, 0)])

    uniq_years = np.unique(years)
    year_index = np.searchsorted(uniq_years, years)

    log.info("synthetic discharge: %d days, %d reaches above %.0f km2, %d storm days",
             n, len(elig), min_area, k)

    out = {}
    areas = elig["area_dn_km2"].to_numpy(dtype=np.float64)
    for reach_id, area in zip(elig["reach_id"].to_numpy(), areas):
        # bigger catchments respond more slowly and remember longer
        memory = np.clip(0.55 + 0.09 * np.log10(max(area, 1.0)), 0.55, 0.93)
        # per-reach annual factor: decorrelates catchments year to year
        reach_year = rng.lognormal(0.0, 0.45, size=uniq_years.size)
        local = magnitude * rng.lognormal(0.0, 0.30, size=n) * reach_year[year_index]

        routed = np.empty(n)
        acc = 0.0
        for t in range(n):
            acc = memory * acc + (1.0 - memory) * local[t]
            routed[t] = acc

        base = 0.02 * season
        q = (area ** 0.85) * (base + 3.2 * routed)
        out[int(reach_id)] = q.astype(np.float32)

    df = pd.DataFrame(out, index=dates)
    df.attrs["synthetic"] = True
    log.info("synthetic discharge table: %d x %d, median annual max %.1f m3/s",
             df.shape[0], df.shape[1],
             float(df.groupby(df.index.year).max().median().median()))
    return df


# ------------------------------------------------------------ soil moisture ---

def soil_moisture(cfg: Node, *args, **kwargs):
    """ERA5-Land volumetric soil water (dataset M3, Doc 10 §3).

    Doc 09 §1.3: ERA5-Land is the physically correct wetness variable but has
    multi-day latency and **cannot be used in real time**. The pipeline
    therefore uses the API proxy from rainfall at run time and ERA5-Land only
    for fitting, so this adapter is not on the critical path for Phases 1-2.
    """
    mode = str(cfg.data_mode.era5_land)
    if mode == "real":
        _require_cdsapirc()
        raise NotImplementedError(
            "ERA5-Land request: dataset 'reanalysis-era5-land', variables "
            "volumetric_soil_water_layer_1..4, subset to the study bbox. "
            "Used for fitting only; see Doc 09 §1.3 for why the run-time path "
            "uses the calibrated API instead.")
    raise NotImplementedError(
        "Soil moisture is not required for Phases 1-2. The antecedent "
        "precipitation index in features/api_index.py covers the wetness term "
        "and is what real-time mode would use anyway.")


def is_synthetic(cfg: Node) -> dict[str, bool]:
    """Which sources are currently standing in. Reported by the quality gates.

    Config flags alone are not enough: the landslide inventory falls back to a
    synthetic one whenever ``data/raw/`` holds no real inventory, whatever the
    flag says. So the catalog itself is asked as well.
    """
    out = {k: str(v) == "synthetic" for k, v in cfg.data_mode.items()}
    out["landslides"] = bool(out.get("landslides")) or _catalog_has_synthetic_landslides()
    return out


_SYNTH_CACHE: dict = {}


def _catalog_has_synthetic_landslides() -> bool:
    from ..catalog.build import EVENTS_PATH
    if not EVENTS_PATH.is_file():
        return False
    key = (str(EVENTS_PATH), EVENTS_PATH.stat().st_mtime_ns)
    if key not in _SYNTH_CACHE:
        df = pd.read_csv(EVENTS_PATH, usecols=["hazard_type", "source"], dtype=str)
        _SYNTH_CACHE.clear()
        _SYNTH_CACHE[key] = bool(((df["hazard_type"] == "L")
                                  & (df["source"] == "SYNTHETIC")).any())
    return _SYNTH_CACHE[key]


def synthetic_sources(cfg: Node) -> list[str]:
    return sorted(k for k, v in is_synthetic(cfg).items() if v)


def catalog_synthetic_counts() -> tuple[int, int]:
    """``(synthetic rows, total rows)`` in the catalog, counted now.

    The report banners used to carry these two numbers as literal text. That is
    fine until the day someone drops a real inventory into ``data/raw/`` — the
    moment the sentence becomes wrong is precisely the moment it matters, and it
    would have gone on declaring 9,764 synthetic rows to every reader.
    """
    from ..catalog.build import EVENTS_PATH
    if not EVENTS_PATH.is_file():
        return 0, 0
    df = pd.read_csv(EVENTS_PATH, usecols=["source"], dtype=str)
    return int((df["source"] == "SYNTHETIC").sum()), int(len(df))
