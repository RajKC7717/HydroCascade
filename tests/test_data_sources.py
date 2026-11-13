"""Readers for the data a human drops into data/raw/ (docs/15_HOW_TO_RUN.md).

Each test writes a tiny netCDF in the layout the provider actually ships and
checks the reader turns it into what the pipeline expects. No network, no
credentials, and nothing under the real data/raw/ is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrocascade import paths
from hydrocascade.config import bbox, load_config
from hydrocascade.ingest import adapters, rainfall_sources


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW", tmp_path / "raw")
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "INTERIM", tmp_path / "interim")
    return tmp_path / "raw"


def _grid(cfg, step):
    minlon, minlat, maxlon, maxlat = bbox(cfg)
    lat = np.arange(minlat - 0.5, maxlat + 0.5, step)
    lon = np.arange(minlon - 0.5, maxlon + 0.5, step)
    return lat, lon


def _drop(folder, name, ds, provenance=True):
    folder.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(folder / name)
    if provenance:
        (folder / "PROVENANCE.md").write_text("# test\n", encoding="utf-8")


# ------------------------------------------------------------------ IMD ---

def test_imd_netcdf_layout_is_read(cfg, raw):
    """IMD Pune ships upper-case TIME/LATITUDE/LONGITUDE and RAINFALL, -999 missing."""
    lat, lon = _grid(cfg, 0.25)
    t = pd.date_range("2024-07-28", periods=4, freq="D")
    rf = np.full((t.size, lat.size, lon.size), 10.0, dtype="float32")
    rf[0, 0, 0] = -999.0
    ds = xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"), rf)},
                    coords={"TIME": t, "LATITUDE": lat, "LONGITUDE": lon})
    _drop(raw / "IMD" / "2026-09-21", "RF25_ind2024_rfp25.nc", ds)

    da = rainfall_sources.read_local("imd", cfg)
    assert da.dims == ("time", "latitude", "longitude")
    assert da.attrs["source"] == "imd" and da.attrs["units"] == "mm/day"
    assert np.isnan(da.isel(time=0).values).sum() == 1          # -999 masked
    assert float(da.isel(time=1).mean()) == pytest.approx(10.0)


def test_imerg_hourly_rate_and_axis_order(cfg, raw):
    """IMERG daily files are (time, lon, lat) in mm/hr for some versions."""
    lat, lon = _grid(cfg, 0.1)
    t = pd.date_range("2024-07-30", periods=1, freq="D")
    pr = np.full((t.size, lon.size, lat.size), 2.0, dtype="float32")
    da = xr.DataArray(pr, dims=("time", "lon", "lat"),
                      coords={"time": t, "lon": lon, "lat": lat},
                      attrs={"units": "mm/hr"})
    _drop(raw / "IMERG" / "2026-09-21", "3B-DAY.MS.MRG.3IMERG.20240730.nc4",
          xr.Dataset({"precipitation": da}))

    out = rainfall_sources.read_local("imerg", cfg)
    assert out.dims == ("time", "latitude", "longitude")
    assert float(out.mean()) == pytest.approx(48.0)             # 2 mm/hr x 24


def test_missing_provenance_is_refused(cfg, raw):
    lat, lon = _grid(cfg, 0.25)
    t = pd.date_range("2024-07-30", periods=1, freq="D")
    ds = xr.Dataset({"RAINFALL": (("TIME", "LATITUDE", "LONGITUDE"),
                                  np.ones((1, lat.size, lon.size), "float32"))},
                    coords={"TIME": t, "LATITUDE": lat, "LONGITUDE": lon})
    _drop(raw / "IMD" / "2026-09-21", "x.nc", ds, provenance=False)
    with pytest.raises(rainfall_sources.RainfallSourceMissing, match="PROVENANCE"):
        rainfall_sources.read_local("imd", cfg)


def test_no_files_gives_instructions(cfg, raw):
    with pytest.raises(rainfall_sources.RainfallSourceMissing, match="15_HOW_TO_RUN"):
        rainfall_sources.read_local("imerg", cfg)


def test_unknown_primary_rejected():
    from types import SimpleNamespace
    fake = SimpleNamespace(rainfall={"primary": "trmm"})
    with pytest.raises(ValueError):
        rainfall_sources.active(fake)


# --------------------------------------------------------------- GloFAS ---

def _glofas_ds(cfg, *, with_uparea):
    lat, lon = _grid(cfg, 0.05)
    lat = lat[::-1]                                  # GloFAS stores north first
    t = pd.date_range("2020-01-01", periods=5, freq="D")
    q = np.broadcast_to(np.arange(lon.size, dtype="float32"),
                        (t.size, lat.size, lon.size)).copy()
    ds = xr.Dataset({"dis24": (("valid_time", "latitude", "longitude"), q)},
                    coords={"valid_time": t, "latitude": lat, "longitude": lon})
    if with_uparea:
        ua = np.broadcast_to((np.arange(lon.size, dtype="float64") + 1) * 100e6,
                             (lat.size, lon.size)).copy()        # m2, grows eastward
        ds["uparea"] = (("latitude", "longitude"), ua)
    return ds


def _centre_area(cfg, opened):
    """Upstream area of the GloFAS cell nearest the synthetic reach outlet."""
    minlon, minlat, maxlon, maxlat = bbox(cfg)
    ua = opened["uparea"].sel(latitude=(minlat + maxlat) / 2,
                              longitude=(minlon + maxlon) / 2, method="nearest")
    return float(ua)


def _reaches(cfg, areas_km2):
    import geopandas as gpd
    from shapely.geometry import LineString
    minlon, minlat, maxlon, maxlat = bbox(cfg)
    cx, cy = (minlon + maxlon) / 2, (minlat + maxlat) / 2
    geoms = [LineString([(cx - 0.01, cy), (cx, cy)]) for _ in areas_km2]
    return gpd.GeoDataFrame({"reach_id": range(len(areas_km2)),
                             "area_dn_km2": areas_km2}, geometry=geoms, crs=4326)


def test_glofas_matches_by_upstream_area(cfg, raw):
    folder = raw / "GLOFAS" / "2026-09-21"
    _drop(folder, "glofas_2020.nc", _glofas_ds(cfg, with_uparea=True))
    opened = adapters._glofas_open(folder, cfg)
    assert opened["uparea"] is not None and float(opened["uparea"].max()) < 1e6  # now km2

    target = _centre_area(cfg, opened) * 1.1           # a little off any cell
    assert target > float(cfg.floods.min_contributing_area_km2)
    df = adapters._glofas_sample(opened, cfg, _reaches(cfg, [target]))
    assert list(df.columns) == [0] and len(df) == 5
    # discharge = column index and area = (index + 1) x 100 km2 by construction,
    # so the matched cell is the one whose area is closest to the target
    matched_area = (int(df.iloc[0, 0]) + 1) * 100.0
    best = (np.round(target / 100.0 - 1) + 1) * 100.0
    assert matched_area == pytest.approx(best)


def test_glofas_drops_reach_with_no_plausible_cell(cfg, raw):
    folder = raw / "GLOFAS" / "2026-09-21"
    _drop(folder, "glofas_2020.nc", _glofas_ds(cfg, with_uparea=True))
    opened = adapters._glofas_open(folder, cfg)
    reaches = _reaches(cfg, [_centre_area(cfg, opened), 1e7])   # 2nd is absurd
    df = adapters._glofas_sample(opened, cfg, reaches)
    assert list(df.columns) == [0]


def test_glofas_without_files_or_key_explains(cfg, raw, monkeypatch, tmp_path):
    monkeypatch.setattr(adapters.Path, "home", lambda: tmp_path / "nohome")
    with pytest.raises(adapters.CredentialsMissing, match="15_HOW_TO_RUN"):
        adapters._glofas_folder(cfg)


def test_config_switches_are_all_read(cfg):
    """Every data_mode key must mean something, or the synthetic flag lies."""
    assert set(cfg.data_mode.keys()) >= {"glofas", "sentinel1"}
    assert "imd" not in cfg.data_mode and "imerg" not in cfg.data_mode
    assert rainfall_sources.active(cfg) in rainfall_sources.SOURCES
