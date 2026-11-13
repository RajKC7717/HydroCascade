"""Daily rainfall from CHIRPS, IMD or IMERG, behind one switch.

``config/default.yaml -> rainfall.primary`` picks the source::

    chirps   free, downloads itself                     (dataset M4)
    imd      IMD 0.25 deg gridded, from local files     (dataset M1)
    imerg    GPM IMERG daily, from local files          (dataset M2)

Every source is returned the same way — a daily ``(time, latitude, longitude)``
DataArray in **mm/day**, clipped to the study area — so nothing downstream knows
or cares which one it came from.

**Why IMD and IMERG are read from local files rather than downloaded.** Both
need an account, and code written against a server nobody here can log into is
code nobody can test. A reader for a file in a published format *can* be tested,
so the reliable path is: download through the provider's own web tool, drop the
files into ``data/raw/<SOURCE>/<YYYY-MM-DD>/`` with a ``PROVENANCE.md``, run.
``docs/15_HOW_TO_RUN.md`` gives the exact form selections.

**Why this matters more than any other data upgrade.** CHIRPS captures 71 mm of
the published 577 mm that fell on Mundakkai in the 48 hours before the failure.
That is gate P2.4, and it is also the reason the Wayanad backtest shows no lead
time: the model's background rises with rainfall, and CHIRPS contains no
rainfall to rise on. IMD is gauge-based and IMERG is 0.1 deg; either is expected
to see the storm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import paths
from ..config import Node, bbox
from ..logging_setup import get

log = get("ingest.rainfall_sources")

SOURCES = ("chirps", "imd", "imerg")

#: Folder under data/raw/ each local source is read from.
RAW_FOLDER = {"imd": "IMD", "imerg": "IMERG"}

#: Variable names seen in the wild, most specific first.
VAR_CANDIDATES = {
    "imd": ("RAINFALL", "rainfall", "rf", "rain", "RF"),
    "imerg": ("precipitation", "precipitationCal", "precipitation_cal"),
}

#: Nominal grid spacing, recorded on the output for the reports.
RESOLUTION = {"chirps": "0.25 deg", "imd": "0.25 deg", "imerg": "0.1 deg"}

_LAT_NAMES = ("latitude", "lat", "LATITUDE", "LAT", "y")
_LON_NAMES = ("longitude", "lon", "LONGITUDE", "LON", "x")
_TIME_NAMES = ("time", "TIME", "valid_time", "date")


class RainfallSourceMissing(FileNotFoundError):
    """A local source was selected but its files are not where they belong."""


def active(cfg: Node) -> str:
    src = str(cfg.rainfall.get("primary", "chirps")).lower()
    if src not in SOURCES:
        raise ValueError(f"rainfall.primary must be one of {SOURCES}, got {src!r}")
    return src


def cache_path(source: str) -> Path:
    return paths.INTERIM / f"rainfall_{source}_aoi.nc"


# ------------------------------------------------------------ the one API ---

def load(cfg: Node, *, force: bool = False) -> xr.DataArray:
    """Daily rainfall for the active source, mm/day, clipped to the study area."""
    src = active(cfg)
    if src == "chirps":
        from . import rainfall_chirps
        da = rainfall_chirps.load(cfg)
        da = da.where(da >= 0)
        da.attrs.setdefault("source", "chirps")
        da.attrs.setdefault("resolution", RESOLUTION["chirps"])
        return da

    out = cache_path(src)
    if out.is_file() and not force:
        return xr.open_dataarray(out)

    da = read_local(src, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(out)
    log.info("cached %s rainfall -> %s (%d days, %d x %d cells)", src,
             out.relative_to(paths.ROOT), da.sizes["time"],
             da.sizes["latitude"], da.sizes["longitude"])
    return da


# ----------------------------------------------------------- local readers ---

def _raw_folder(src: str) -> Path:
    base = paths.RAW / RAW_FOLDER[src]
    folders = sorted((p for p in base.iterdir() if p.is_dir()), reverse=True) \
        if base.is_dir() else []
    # Prefer the newest dated folder that actually holds grids.
    for f in folders:
        if any(f.glob("*.nc")) or any(f.glob("*.nc4")):
            return f
    if base.is_dir() and (any(base.glob("*.nc")) or any(base.glob("*.nc4"))):
        return base
    raise RainfallSourceMissing(
        f"rainfall.primary is '{src}' but no .nc / .nc4 files were found under "
        f"{base.relative_to(paths.ROOT)}/<YYYY-MM-DD>/.\n"
        f"Download them as described in docs/15_HOW_TO_RUN.md, put them in a "
        f"dated folder there with a PROVENANCE.md, and re-run "
        f"`hydrocascade rainfall --force`.")


def _pick(names, available, what):
    for n in names:
        if n in available:
            return n
    raise ValueError(f"no {what} found; looked for {names}, file has {sorted(available)}")


def _standardise(ds: xr.Dataset, src: str) -> xr.DataArray:
    """Rename to (time, latitude, longitude), mm/day, missing values as NaN."""
    names = set(ds.variables) | set(ds.dims)
    lat = _pick(_LAT_NAMES, names, "latitude")
    lon = _pick(_LON_NAMES, names, "longitude")
    tim = _pick(_TIME_NAMES, names, "time")

    data_vars = list(ds.data_vars)
    try:
        var = _pick(VAR_CANDIDATES[src], set(data_vars), "rainfall variable")
    except ValueError:
        # One data variable and it is not a bounds array: that is the rainfall.
        cand = [v for v in data_vars if "bnds" not in v and "bounds" not in v]
        if len(cand) != 1:
            raise
        var = cand[0]

    da = ds[var].rename({lat: "latitude", lon: "longitude", tim: "time"})
    extra = [d for d in da.dims if d not in ("time", "latitude", "longitude")]
    if extra:
        da = da.isel({d: 0 for d in extra})
    # IMERG daily files are stored (time, lon, lat); name-based transpose makes
    # the axis order irrelevant.
    da = da.transpose("time", "latitude", "longitude").astype("float32")

    units = str(da.attrs.get("units", "")).lower().replace(" ", "")
    if units in ("mm/hr", "mmhr-1", "mm/h", "mmh-1"):
        # A daily file stored as a mean hourly rate.
        da = da * 24.0
        log.info("%s: converted mean hourly rate to mm/day", src)
    elif units in ("m", "metre", "meter", "m/day"):
        da = da * 1000.0
        log.info("%s: converted metres to mm", src)

    # IMD marks missing as -999 and ocean cells as missing; IMERG uses
    # -9999.9. Anything negative is not rainfall.
    da = da.where(da >= 0)
    if "time" in da.indexes:
        da = da.assign_coords(time=pd.DatetimeIndex(da.indexes["time"]).normalize())
    return da


def read_local(src: str, cfg: Node) -> xr.DataArray:
    """Read every grid in the newest ``data/raw/<SOURCE>/<date>/`` folder."""
    folder = _raw_folder(src)
    if not (folder / "PROVENANCE.md").is_file():
        raise RainfallSourceMissing(
            f"{folder.relative_to(paths.ROOT)} has rainfall files but no "
            f"PROVENANCE.md. Doc 10 §8 makes it non-negotiable: every raw "
            f"download carries one, written at download time. The template is "
            f"in docs/15_HOW_TO_RUN.md.")

    files = sorted(list(folder.glob("*.nc")) + list(folder.glob("*.nc4")))
    minlon, minlat, maxlon, maxlat = bbox(cfg)
    pad = 0.5                                    # a full cell of margin at 0.25 deg

    pieces = []
    for f in files:
        with xr.open_dataset(f) as ds:
            da = _standardise(ds, src)
            lat_asc = bool(da.latitude[0] < da.latitude[-1])
            da = da.sel(
                latitude=slice(minlat - pad, maxlat + pad) if lat_asc
                else slice(maxlat + pad, minlat - pad),
                longitude=slice(minlon - pad, maxlon + pad)).load()
        if da.sizes["latitude"] == 0 or da.sizes["longitude"] == 0:
            log.warning("%s does not cover the study area; skipped", f.name)
            continue
        pieces.append(da)
    if not pieces:
        raise RainfallSourceMissing(
            f"none of the {len(files)} files in {folder.relative_to(paths.ROOT)} "
            f"cover the study area {minlon}-{maxlon} E, {minlat}-{maxlat} N")

    da = xr.concat(pieces, dim="time").sortby("time")
    _, keep = np.unique(da.time.values, return_index=True)
    da = da.isel(time=np.sort(keep)).sortby("latitude")
    da.name = "precip"
    da.attrs.update(source=src, resolution=RESOLUTION[src],
                    folder=str(folder.relative_to(paths.ROOT)), units="mm/day")

    start, end = pd.Timestamp(str(cfg.period.start)), pd.Timestamp(str(cfg.period.end))
    have = pd.DatetimeIndex(da.time.values)
    if have.min() > start or have.max() < end:
        log.warning("%s covers %s to %s; the study period is %s to %s. Days "
                    "outside the files will be missing, not zero.", src,
                    have.min().date(), have.max().date(), start.date(), end.date())
    log.info("read %d %s files: %d days, %d x %d cells", len(pieces), src,
             da.sizes["time"], da.sizes["latitude"], da.sizes["longitude"])
    return da


# -------------------------------------------------- Doc 05 §7.1, any source ---

def wayanad_reconciliation(cfg: Node) -> dict:
    """Doc 05 §7.1: *the single best sanity check available in the whole project*.

    The published totals for the Mundakkai cell are about 204 mm on 29 July 2024
    and 373 mm on 30 July. Doc 05 §7.1 is blunt: *"If you cannot reproduce
    those numbers, your extraction is wrong."* That instruction is written for
    IMD. For a satellite-blended product a shortfall is a product result to
    report — for IMD it would be a bug to fix.
    """
    da = load(cfg)
    src = str(da.attrs.get("source", active(cfg)))
    lat = float(cfg.landmarks.mundakkai_crown["lat"])
    lon = float(cfg.landmarks.mundakkai_crown["lon"])
    cell = da.sel(latitude=lat, longitude=lon, method="nearest")

    ev = cfg.wayanad_event
    d1 = pd.Timestamp(str(ev.date)) - pd.Timedelta(days=1)
    d2 = pd.Timestamp(str(ev.date))

    got: dict = {"source": src}
    for label, day, published in (("prior_24h", d1, float(ev.rainfall_mm_24h_prior)),
                                  ("event_24h", d2, float(ev.rainfall_mm_24h_event))):
        try:
            val = float(cell.sel(time=day.strftime("%Y-%m-%d"), method="nearest").values)
        except Exception:                           # noqa: BLE001
            val = float("nan")
        got[label] = {"date": day.date().isoformat(), "published_mm": published,
                      "extracted_mm": val,
                      "ratio": val / published if published else float("nan")}

    tol = float(ev.rainfall_tolerance_frac)
    window = cell.sel(time=slice(d1.strftime("%Y-%m-%d"), d2.strftime("%Y-%m-%d")))
    got["window_48h_mm"] = float(window.sum()) if window.sizes.get("time", 0) else float("nan")
    got["published_48h_mm"] = (float(ev.rainfall_mm_24h_prior)
                               + float(ev.rainfall_mm_24h_event))
    got["tolerance_frac"] = tol
    got["within_tolerance"] = bool(
        np.isfinite(got["window_48h_mm"])
        and abs(got["window_48h_mm"] - got["published_48h_mm"])
        / got["published_48h_mm"] <= tol)
    got["grid_cell"] = {"lat": float(cell.latitude), "lon": float(cell.longitude),
                        "resolution": da.attrs.get("resolution", RESOLUTION.get(src, "?"))}

    log.info("Wayanad reconciliation (%s %s):", src.upper(), got["grid_cell"]["resolution"])
    for k in ("prior_24h", "event_24h"):
        r = got[k]
        log.info("  %-10s %s  published %6.1f mm   %s %6.1f mm   ratio %.2f",
                 k, r["date"], r["published_mm"], src.upper(), r["extracted_mm"],
                 r["ratio"])
    log.info("  48 h total: published %.0f mm, %s %.0f mm -> %s",
             got["published_48h_mm"], src.upper(), got["window_48h_mm"],
             "WITHIN tolerance" if got["within_tolerance"] else "SHORTFALL")
    return got
