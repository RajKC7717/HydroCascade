"""FR-03 (deduplication) and FR-04 (provenance), on hand-built catalogs."""

from __future__ import annotations

import pandas as pd
import pytest

from hydrocascade.catalog import dedup, schema

CRS = "EPSG:32643"


def _row(**kw):
    base = dict(
        event_id="L-0001", hazard_type="L", date="2018-08-16", time_known="N", time="",
        lat=11.4654, lon=76.1348, geometry_type="point", area_m2=5000.0,
        elevation_m=1500.0, mark_raw=5000.0, mark_value=8.517, severity_class="medium",
        spatial_unit_id="", source="TEST", source_url="https://example.org/x",
        source_date_accessed="2026-09-04", accessed_by="tester", confidence="medium",
        duplicate_of="", notes="",
    )
    base.update(kw)
    return base


def _frame(rows):
    return schema.coerce(pd.DataFrame(rows))


# ------------------------------------------------------------------ schema ---

def test_valid_catalog_passes():
    df = _frame([_row(), _row(event_id="L-0002", lat=11.9, lon=76.9)])
    assert schema.validate(df, strict=False) == []


def test_missing_source_url_is_an_fr04_violation():
    df = _frame([_row(source_url="")])
    problems = schema.validate(df, strict=False)
    assert any("FR-04" in p and "source_url" in p for p in problems)


def test_missing_accessor_is_an_fr04_violation():
    df = _frame([_row(accessed_by="")])
    assert any("FR-04" in p and "accessed_by" in p for p in problems_of(df))


def problems_of(df):
    return schema.validate(df, strict=False)


def test_bad_date_format_is_caught():
    assert any("YYYY-MM-DD" in p for p in problems_of(_frame([_row(date="16/08/2018")])))


def test_duplicate_event_ids_are_caught():
    df = _frame([_row(), _row(lat=12.0)])
    assert any("duplicate event_id" in p for p in problems_of(df))


def test_wrong_severity_vocabulary_for_hazard_type():
    """A flood cannot be 'large'; that is the landslide vocabulary."""
    df = _frame([_row(hazard_type="F", severity_class="large", area_m2="")])
    assert any("severity_class for F" in p for p in problems_of(df))


def test_time_known_without_a_time_is_caught():
    assert any("time_known=Y" in p for p in problems_of(_frame([_row(time_known="Y")])))


def test_roundtrip_preserves_coordinate_precision(tmp_path):
    """Doc 04 §2: coordinate precision must survive a write/read cycle."""
    df = _frame([_row(lat=11.465432, lon=76.134876)])
    p = tmp_path / "events.csv"
    schema.write(df, p)
    back = schema.read(p)
    assert back["lat"].iloc[0] == pytest.approx(11.465432, abs=1e-6)
    assert back["lon"].iloc[0] == pytest.approx(76.134876, abs=1e-6)


def test_event_ids_are_assigned_per_hazard_type():
    df = _frame([_row(hazard_type="L"), _row(hazard_type="F"), _row(hazard_type="L")])
    out = schema.make_event_ids(df)
    assert list(out["event_id"]) == ["L-0001", "F-0001", "L-0002"]


# --------------------------------------------------------------- dedup ---

def test_records_within_500m_and_1_day_merge():
    """Two inventories listing the same slide ~300 m apart on the same day."""
    df = _frame([
        _row(event_id="L-0001", source="INV_A", confidence="medium"),
        _row(event_id="L-0002", source="INV_B", lat=11.4681, confidence="high"),
    ])
    kept, gone = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 1
    assert len(gone) == 1
    # higher confidence wins
    assert kept["event_id"].iloc[0] == "L-0002"
    assert gone["duplicate_of"].iloc[0] == "L-0002"


def test_records_beyond_500m_stay_separate():
    df = _frame([
        _row(event_id="L-0001"),
        _row(event_id="L-0002", lat=11.4754),      # ~1.1 km north
    ])
    kept, gone = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 2 and len(gone) == 0


def test_records_more_than_a_day_apart_stay_separate():
    df = _frame([
        _row(event_id="L-0001", date="2018-08-16"),
        _row(event_id="L-0002", date="2018-08-18"),
    ])
    kept, _ = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 2


def test_exactly_one_day_apart_still_merges():
    df = _frame([
        _row(event_id="L-0001", date="2018-08-16"),
        _row(event_id="L-0002", date="2018-08-17", confidence="high"),
    ])
    kept, _ = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 1


def test_polygon_beats_point_when_confidence_ties():
    df = _frame([
        _row(event_id="L-0001", geometry_type="point", confidence="medium"),
        _row(event_id="L-0002", geometry_type="polygon_crown", confidence="medium",
             lat=11.4670),
    ])
    kept, _ = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert kept["event_id"].iloc[0] == "L-0002"


def test_earlier_source_wins_when_confidence_and_geometry_tie():
    df = _frame([
        _row(event_id="L-0001", source_date_accessed="2026-09-10"),
        _row(event_id="L-0002", source_date_accessed="2026-09-01", lat=11.4670),
    ])
    kept, _ = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert kept["event_id"].iloc[0] == "L-0002"


def test_landslides_and_floods_are_never_merged():
    """Different hazard types at the same place and time are two events."""
    df = _frame([
        _row(event_id="L-0001", hazard_type="L"),
        _row(event_id="F-0001", hazard_type="F", area_m2="", severity_class="major"),
    ])
    kept, gone = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 2 and len(gone) == 0


def test_chain_of_nearby_records_becomes_one_event():
    """Transitive closure: A-B and B-C within 500 m makes one event, not two."""
    df = _frame([
        _row(event_id="L-0001", lat=11.4654),
        _row(event_id="L-0002", lat=11.4681),      # ~300 m from L-0001
        _row(event_id="L-0003", lat=11.4708),      # ~300 m from L-0002, ~600 from L-0001
    ])
    kept, gone = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 1 and len(gone) == 2


def test_rows_without_coordinates_survive_dedup():
    """Doc 04 §5 emits documented events whose coordinates are not yet sourced.

    Those rows carry NaN lat/lon on purpose rather than invented values, so
    dedup must keep each of them as its own event instead of failing.
    """
    df = _frame([
        _row(event_id="L-0001"),
        _row(event_id="L-0002", lat=float("nan"), lon=float("nan"),
             source="NEEDS_SOURCING", confidence="low"),
        _row(event_id="L-0003", lat=float("nan"), lon=float("nan"),
             source="NEEDS_SOURCING", confidence="low"),
    ])
    kept, gone = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert len(kept) == 3, "unlocatable rows must not be merged into each other"
    assert len(gone) == 0


def test_assert_no_duplicates_agrees_with_deduplicate():
    df = _frame([
        _row(event_id="L-0001"),
        _row(event_id="L-0002", lat=11.4681, confidence="high"),
    ])
    assert dedup.assert_no_duplicates(df, crs=CRS) == 1
    kept, _ = dedup.deduplicate(df, distance_m=500, days=1, crs=CRS)
    assert dedup.assert_no_duplicates(kept, crs=CRS) == 0
