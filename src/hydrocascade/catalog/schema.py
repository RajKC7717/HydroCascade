"""The catalog schema — Doc 04 §2, these columns exactly, in this order.

Three rules from Doc 04 §2 are enforced in code rather than trusted to
discipline:

* every row has a ``source_url`` and an ``accessed_by`` — no exceptions (FR-04);
* no CSV is ever opened and saved in Excel, which silently mangles dates and
  truncates coordinate precision — so dates are written as strings and
  coordinates at a fixed 6 decimal places;
* ``data/raw/`` is never edited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("catalog.schema")

COLUMNS: tuple[str, ...] = (
    "event_id",             # unique, format L-0001 / F-0001
    "hazard_type",          # L | F
    "date",                 # YYYY-MM-DD
    "time_known",           # Y | N
    "time",                 # HH:MM IST, blank if unknown
    "lat", "lon",           # EPSG:4326, decimal degrees, 6 dp
    "geometry_type",        # point | polygon_centroid | polygon_crown | reach
    "area_m2",              # mapped area, blank for floods
    "elevation_m",          # from DEM at the recorded point
    "mark_raw",             # L: area_m2 | F: peak discharge m3/s
    "mark_value",           # L: log(area_m2) | F: log(Q_peak/Q2)
    "severity_class",       # L: small/medium/large | F: minor/major/extreme
    "spatial_unit_id",      # slope unit id (L) or reach id (F)
    "source",               # short code, e.g. GSI_BHUKOSH
    "source_url",
    "source_date_accessed",  # YYYY-MM-DD
    "accessed_by",
    "confidence",           # high | medium | low
    "duplicate_of",         # event_id if merged, else blank
    "notes",
)

HAZARD_TYPES = {"L", "F"}
GEOMETRY_TYPES = {"point", "polygon_centroid", "polygon_crown", "reach"}
CONFIDENCE = {"high", "medium", "low"}
SEVERITY_L = {"small", "medium", "large"}
SEVERITY_F = {"minor", "major", "extreme"}

STRING_COLUMNS = (
    "event_id", "hazard_type", "date", "time_known", "time", "geometry_type",
    "severity_class", "source", "source_url", "source_date_accessed",
    "accessed_by", "confidence", "duplicate_of", "notes", "spatial_unit_id",
)


def empty() -> pd.DataFrame:
    """An empty catalog with the right columns in the right order."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Put a frame into schema order, adding missing columns as blanks."""
    out = df.copy()
    for c in COLUMNS:
        if c not in out.columns:
            out[c] = ""
    extra = [c for c in out.columns if c not in COLUMNS]
    if extra:
        log.debug("dropping non-schema columns: %s", ", ".join(extra))
    return out[list(COLUMNS)]


def make_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign L-0001 / F-0001 style ids, stable under sort by (type, date)."""
    out = df.copy()
    ids = []
    counters = {"L": 0, "F": 0}
    for htype in out["hazard_type"]:
        counters[htype] = counters.get(htype, 0) + 1
        ids.append(f"{htype}-{counters[htype]:04d}")
    out["event_id"] = ids
    return out


def validate(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Return a list of schema violations. Empty list means the catalog is clean."""
    problems: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")
        return problems
    if list(df.columns) != list(COLUMNS):
        problems.append("columns are present but not in the Doc 04 §2 order")

    if df.empty:
        problems.append("catalog is empty")
        return problems

    # --- identity
    dup_ids = df["event_id"][df["event_id"].duplicated()].tolist()
    if dup_ids:
        problems.append(f"duplicate event_id: {dup_ids[:5]}")
    if df["event_id"].eq("").any():
        problems.append("blank event_id")

    # --- controlled vocabularies
    bad = set(df["hazard_type"].unique()) - HAZARD_TYPES
    if bad:
        problems.append(f"hazard_type not in {{L, F}}: {sorted(bad)}")
    bad = set(df["geometry_type"].unique()) - GEOMETRY_TYPES - {""}
    if bad:
        problems.append(f"unknown geometry_type: {sorted(bad)}")
    bad = set(df["confidence"].unique()) - CONFIDENCE - {""}
    if bad:
        problems.append(f"unknown confidence: {sorted(bad)}")

    for htype, allowed in (("L", SEVERITY_L), ("F", SEVERITY_F)):
        sel = df[df["hazard_type"] == htype]
        bad = set(sel["severity_class"].unique()) - allowed - {""}
        if bad:
            problems.append(f"severity_class for {htype} not in {sorted(allowed)}: {sorted(bad)}")

    # --- dates
    parsed = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    n_bad = int(parsed.isna().sum())
    if n_bad:
        problems.append(f"{n_bad} rows have a date that is not YYYY-MM-DD")

    # --- coordinates
    for col, lo, hi in (("lat", -90, 90), ("lon", -180, 180)):
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            problems.append(f"{int(vals.isna().sum())} rows have a non-numeric {col}")
        outside = vals.dropna()
        if not outside.between(lo, hi).all():
            problems.append(f"{col} outside [{lo}, {hi}]")

    # --- FR-04: provenance is not optional
    for col in ("source_url", "accessed_by", "source"):
        blank = df[col].astype(str).str.strip().eq("") | df[col].isna()
        if blank.any():
            problems.append(f"FR-04 violation: {int(blank.sum())} rows have no {col}")

    # --- time_known consistency
    tk = df["time_known"].astype(str)
    bad_tk = set(tk.unique()) - {"Y", "N", ""}
    if bad_tk:
        problems.append(f"time_known not in {{Y, N}}: {sorted(bad_tk)}")
    says_known = tk.eq("Y")
    has_time = df["time"].astype(str).str.strip().ne("")
    if (says_known & ~has_time).any():
        problems.append(f"{int((says_known & ~has_time).sum())} rows say time_known=Y "
                        "but carry no time")

    if strict and problems:
        for p in problems:
            log.error("schema: %s", p)
    return problems


def write(df: pd.DataFrame, path) -> None:
    """Write the catalog with fixed coordinate precision and no index."""
    out = coerce(df).copy()
    for col in ("lat", "lon"):
        out[col] = pd.to_numeric(out[col], errors="coerce").map(
            lambda v: "" if pd.isna(v) else f"{v:.6f}")
    for col in ("area_m2", "elevation_m", "mark_raw", "mark_value"):
        out[col] = pd.to_numeric(out[col], errors="coerce").map(
            lambda v: "" if pd.isna(v) else f"{v:.6g}")
    for col in STRING_COLUMNS:
        out[col] = out[col].fillna("").astype(str)
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    log.info("wrote   %s (%d rows)", path, len(out))


def read(path) -> pd.DataFrame:
    """Read the catalog back with everything as strings except the numerics."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for col in ("lat", "lon", "area_m2", "elevation_m", "mark_raw", "mark_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
