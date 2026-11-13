"""CHIRPS daily rainfall — dataset **M4**, added to the Doc 10 §3 list.

Doc 10 §3 names three rainfall products, and **all three need an account**: IMD
requires a registration form, IMERG requires NASA Earthdata, ERA5-Land requires
Copernicus CDS. That leaves Phases 2 and 3 blocked on paperwork.

CHIRPS (Climate Hazards Group InfraRed Precipitation with Stations) is free,
needs no login, is gauge-blended like IMD rather than satellite-only, runs
1981-present daily, and comes in 0.05 and 0.25 degree grids. The 0.25 degree
product matches IMD's native resolution exactly, which is the resolution Doc 05
§7 uses for the antecedent windows anyway.

So CHIRPS is the development backbone and the third product in the Doc 05 §7.1
reconciliation. It does **not** replace IMD or IMERG:

* IMD stays the reference for long-record climatology and totals;
* IMERG stays the product for event-scale intensity and all real-time work,
  because it is half-hourly with 4 h latency;
* CHIRPS is daily and published with a lag of about three weeks, so it is a
  fitting-time product, not a run-time one.

Storage: the global yearly file is downloaded, clipped to the study bounding box
and deleted, so the repository keeps a few megabytes rather than a few gigabytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import paths
from ..config import Node, bbox
from ..logging_setup import get
from ..provenance import Provenance
from .fetch import download

log = get("ingest.chirps")

SOURCE_CODE = "CHIRPS"
BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf"
CITATION = ("Funk et al. (2015), 'The climate hazards infrared precipitation with "
            "stations', Scientific Data 2:150066")
LICENCE = "CHIRPS is public domain (Climate Hazards Center, UC Santa Barbara)"

CLIPPED_PATH = paths.INTERIM / "chirps_aoi.nc"


def _year_url(year: int, resolution: str) -> str:
    return f"{BASE}/{resolution}/chirps-v2.0.{year}.days_{resolution}.nc"


def fetch(cfg: Node, *, force: bool = False) -> Path:
    """Download, clip and concatenate CHIRPS over the study period."""
    if CLIPPED_PATH.is_file() and not force:
        log.info("cached  %s", CLIPPED_PATH.name)
        return CLIPPED_PATH

    resolution = str(cfg.rainfall.get("chirps_resolution", "p25"))
    minlon, minlat, maxlon, maxlat = bbox(cfg)
    pad = 0.5
    start = pd.Timestamp(str(cfg.period.start))
    end = pd.Timestamp(str(cfg.period.end))

    # The longest rainfall window needs a run-up before the period starts.
    max_window = int(max(cfg.rainfall.windows_days))
    first_year = (start - pd.Timedelta(days=max_window + 5)).year
    years = list(range(first_year, end.year + 1))
    log.info("CHIRPS %s, years %d-%d, bbox %.2f %.2f %.2f %.2f",
             resolution, years[0], years[-1], minlon, minlat, maxlon, maxlat)

    folder = paths.raw_dir(SOURCE_CODE, f"chirps_{resolution}")
    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    pieces = []
    urls = []

    for year in years:
        url = _year_url(year, resolution)
        urls.append(url)
        small = paths.INTERIM / f"chirps_{resolution}_{year}_aoi.nc"
        if small.is_file() and not force:
            pieces.append(xr.open_dataarray(small))
            continue

        tmp = folder / f"chirps-v2.0.{year}.days_{resolution}.nc"
        download(url, tmp, force=force, timeout=600)
        with xr.open_dataset(tmp) as ds:
            var = "precip" if "precip" in ds else list(ds.data_vars)[0]
            da = ds[var].sel(longitude=slice(minlon - pad, maxlon + pad),
                             latitude=slice(minlat - pad, maxlat + pad))
            da = da.load()
        da.to_netcdf(small)
        pieces.append(xr.open_dataarray(small))
        # global file is ~83 MB (p25) or ~1.4 GB (p05); do not keep it
        tmp.unlink(missing_ok=True)
        log.info("clipped %d -> %s (%.2f MB)", year, small.name,
                 small.stat().st_size / 1e6)

    combined = xr.concat(pieces, dim="time").sortby("time")
    combined = combined.sel(time=slice(str(first_year), str(end.date())))
    combined.name = "precip"
    combined.attrs.update(units="mm/day", source="CHIRPS v2.0", resolution=resolution)
    combined.to_netcdf(CLIPPED_PATH)
    for p in pieces:
        p.close()

    Provenance(
        source=f"CHIRPS v2.0 global daily precipitation, {resolution} grid",
        url=f"{BASE}/{resolution}/chirps-v2.0.<year>.days_{resolution}.nc",
        query=f"years {years[0]}-{years[-1]}, clipped to lon "
              f"[{minlon-pad:.2f}, {maxlon+pad:.2f}], lat [{minlat-pad:.2f}, "
              f"{maxlat+pad:.2f}]. Global files deleted after clipping.",
        licence=LICENCE,
        citation=CITATION,
        notes="Dataset M4, added to Doc 10 §3. Free and login-free, so it unblocks "
              "Phases 2-3 while IMD/IMERG/ERA5 accounts are pending. Gauge-blended "
              "like IMD. Not a real-time product: published with ~3 weeks lag.",
    ).write(folder, files=[CLIPPED_PATH])

    log.info("CHIRPS ready: %s, %d days, grid %d x %d", CLIPPED_PATH.name,
             combined.sizes["time"], combined.sizes["latitude"],
             combined.sizes["longitude"])
    return CLIPPED_PATH


def load(cfg: Node) -> xr.DataArray:
    if not CLIPPED_PATH.is_file():
        fetch(cfg)
    da = xr.open_dataarray(CLIPPED_PATH)
    return da


def wayanad_reconciliation(cfg: Node) -> dict:
    """Doc 05 §7.1: *the single best sanity check available in the whole project*.

    The published totals for the Wayanad cell are about 204 mm on 29 July 2024
    and 373 mm on 30 July. Doc 05 §7.1 step 3 is blunt about what a mismatch
    means: *"If you cannot reproduce those numbers, your extraction is wrong.
    Fix it before proceeding."*

    That instruction is written for IMD. CHIRPS is a different product, and
    satellite-blended products are documented to underestimate extreme
    orographic rainfall in the Western Ghats, so a shortfall here is a
    **product** result to report rather than an extraction bug. The function
    returns the numbers either way; interpreting them is Doc 05 §7.1 step 4.
    """
    da = load(cfg)
    lat = float(cfg.landmarks.mundakkai_crown["lat"])
    lon = float(cfg.landmarks.mundakkai_crown["lon"])
    cell = da.sel(latitude=lat, longitude=lon, method="nearest")

    ev = cfg.wayanad_event
    d1 = pd.Timestamp(str(ev.date)) - pd.Timedelta(days=1)
    d2 = pd.Timestamp(str(ev.date))

    got = {}
    for label, day, published in (("prior_24h", d1, float(ev.rainfall_mm_24h_prior)),
                                  ("event_24h", d2, float(ev.rainfall_mm_24h_event))):
        try:
            val = float(cell.sel(time=day.strftime("%Y-%m-%d"), method="nearest").values)
        except Exception:  # noqa: BLE001
            val = float("nan")
        got[label] = {"date": day.date().isoformat(), "published_mm": published,
                      "extracted_mm": val,
                      "ratio": val / published if published else float("nan")}

    tol = float(ev.rainfall_tolerance_frac)
    got["window_48h_mm"] = float(
        cell.sel(time=slice(d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d"))).sum())
    got["published_48h_mm"] = (float(ev.rainfall_mm_24h_prior)
                               + float(ev.rainfall_mm_24h_event))
    got["tolerance_frac"] = tol
    got["within_tolerance"] = bool(
        abs(got["window_48h_mm"] - got["published_48h_mm"]) / got["published_48h_mm"] <= tol)
    got["grid_cell"] = {"lat": float(cell.latitude), "lon": float(cell.longitude),
                        "resolution": da.attrs.get("resolution", "?")}

    log.info("Wayanad reconciliation (CHIRPS %s):", got["grid_cell"]["resolution"])
    for k in ("prior_24h", "event_24h"):
        r = got[k]
        log.info("  %-10s %s  published %6.1f mm   CHIRPS %6.1f mm   ratio %.2f",
                 k, r["date"], r["published_mm"], r["extracted_mm"], r["ratio"])
    log.info("  48 h total: published %.0f mm, CHIRPS %.0f mm -> %s",
             got["published_48h_mm"], got["window_48h_mm"],
             "WITHIN tolerance" if got["within_tolerance"] else "SHORTFALL")
    return got
