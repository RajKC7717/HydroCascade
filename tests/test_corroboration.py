"""The corroboration layer — Doc 02 §3.3, Doc 04 §4.2.

These tests guard two things that are easy to get quietly wrong:

1. **The exceedance-rate expectation.** The intuitive answer ("a 2-year event
   happens every other year, so about one a year") is wrong, and a sanity check
   built on it fires on healthy data. The correct value is ``-ln(1 - 1/T)``.
2. **The refusal to invent a corroboration number.** A test asserting that a
   function *raises* looks odd until you remember what the alternative is: a
   plausible-looking precision figure that nobody computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrocascade.config import Node
from hydrocascade.ingest import corroboration, floods
from hydrocascade.ingest.adapters import CredentialsMissing


# ------------------------------------------------- the rate a threshold implies ---

def test_expected_rate_is_the_poisson_value_not_one():
    """Q2 implies ln 2 events per reach-year, not 1.0.

    ``P(annual max <= Q_T) = exp(-lam) = 1 - 1/T``, so ``lam = -ln(1 - 1/T)``.
    """
    assert floods.expected_exceedance_rate(2) == pytest.approx(np.log(2.0))
    assert floods.expected_exceedance_rate(5) == pytest.approx(-np.log(0.8))
    assert floods.expected_exceedance_rate(20) == pytest.approx(-np.log(0.95))


def test_expected_rate_falls_with_return_period():
    rates = [floods.expected_exceedance_rate(t) for t in (2, 5, 20, 100)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < 0.02


def test_expected_rate_rejects_a_return_period_of_one():
    with pytest.raises(ValueError):
        floods.expected_exceedance_rate(1.0)


# ------------------------------------------------------------- config variants ---

def _cfg(**flood_overrides) -> Node:
    base = {
        "project": {"seed": 1},
        "region": {"id": "test"},
        "period": {"start": "2000-01-01", "end": "2009-12-31"},
        "floods": {"return_periods": [2, 5, 20], "threshold_return_period": 2,
                   "decluster_days": 3, "min_contributing_area_km2": 50,
                   "min_years_for_gumbel": 5},
    }
    base["floods"].update(flood_overrides)
    return Node(base)


def test_variant_config_changes_only_the_flood_knobs():
    cfg = _cfg()
    v = corroboration._variant_config(cfg, 5, 5, 250.0)
    assert v.floods.threshold_return_period == 5
    assert v.floods.decluster_days == 5
    assert v.floods.min_contributing_area_km2 == 250.0
    assert v.floods.min_years_for_gumbel == 5          # untouched
    assert v.period.start == "2000-01-01"              # untouched


def test_variant_config_does_not_mutate_the_original():
    """A sensitivity run that quietly rewrote the headline config would make
    every later stage report the variant's numbers."""
    cfg = _cfg()
    corroboration._variant_config(cfg, 5, 5, 250.0)
    assert cfg.floods.threshold_return_period == 2
    assert cfg.floods.min_contributing_area_km2 == 50


# ------------------------------------------------------------------ sensitivity ---

def _discharge(n_reaches=3, years=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=365 * years, freq="D")
    data = {}
    for r in range(n_reaches):
        base = rng.lognormal(3.0, 0.6, size=len(dates))
        base[rng.random(len(dates)) < 0.02] *= 8.0     # storm days
        data[r] = base
    return pd.DataFrame(data, index=dates)


def test_sensitivity_returns_one_row_per_variant_and_q5_is_a_subset():
    """Raising the threshold can only remove events, never add them."""
    cfg = _cfg()
    disch = _discharge()
    areas = pd.Series({0: 100.0, 1: 400.0, 2: 60.0})
    out = corroboration.sensitivity(cfg, disch, areas)

    assert len(out) == 4
    headline = out.iloc[0]
    q5 = out[out["threshold_rp"] == 5].iloc[0]
    assert q5["n_events"] <= headline["n_events"]

    # 250 km2 leaves only the one reach above it
    strict = out[out["min_area_km2"] == 250.0].iloc[0]
    assert strict["n_eligible_reaches"] == 1


def test_sensitivity_survives_a_variant_with_no_eligible_reach():
    """Eligibility at 250 km2 can empty the basin. That is a row of zeros, not
    a crash — the report still has to render."""
    cfg = _cfg()
    areas = pd.Series({0: 60.0, 1: 70.0})
    out = corroboration.sensitivity(
        cfg, _discharge(n_reaches=2), areas,
        variants=[("strict", 2, 3, 250.0)])
    assert len(out) == 1
    assert out.iloc[0]["n_events"] == 0


# ----------------------------------------------------------------------- recall ---

def _reference(rows):
    full = []
    for r in rows:
        full.append({c: r.get(c, "") for c in corroboration.REFERENCE_COLUMNS})
    df = pd.DataFrame(full, columns=list(corroboration.REFERENCE_COLUMNS))
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def test_recall_matches_inside_the_window_and_misses_outside_it():
    detected = pd.DataFrame({"date": ["2019-08-10", "2020-01-05"]})
    ref = _reference([
        dict(flood_id="A", date_start="2019-08-12", date_end="2019-08-12",
             source_url="http://example.org/a", evidence_type="ksdma_report"),
        dict(flood_id="B", date_start="2021-06-01", date_end="2021-06-01",
             source_url="http://example.org/b", evidence_type="ksdma_report"),
    ])
    res = corroboration.recall(detected, ref, window_days=3)
    assert res.status == "measured"
    assert res.n_scored == 2 and res.n_matched == 1
    assert res.value == pytest.approx(0.5)
    got = res.detail.set_index("flood_id")["matched"]
    assert bool(got["A"]) and not bool(got["B"])


def test_recall_honours_a_date_range():
    """A source that says only 'August 2019' must not be sharpened into a day."""
    detected = pd.DataFrame({"date": ["2019-08-24"]})
    ref = _reference([dict(flood_id="A", date_start="2019-08-01",
                           date_end="2019-08-31",
                           source_url="u", evidence_type="ems_activation")])
    assert corroboration.recall(detected, ref).n_matched == 1


def test_rows_outside_the_pilot_basin_are_reported_but_not_scored():
    detected = pd.DataFrame({"date": ["2020-08-06"]})
    ref = _reference([
        dict(flood_id="IN", date_start="2020-08-06", in_pilot_basin="yes",
             source_url="u", evidence_type="cwc_gauge"),
        dict(flood_id="OUT", date_start="2015-01-01", in_pilot_basin="no",
             source_url="u", evidence_type="cwc_gauge"),
    ])
    res = corroboration.recall(detected, ref)
    assert res.n_reference == 2 and res.n_scored == 1
    assert res.value == pytest.approx(1.0)
    assert "OUT" in set(res.detail["flood_id"])          # reported, just not scored


def test_unsourced_rows_are_ignored_on_the_fr04_rule():
    detected = pd.DataFrame({"date": ["2019-08-10"]})
    ref = _reference([
        dict(flood_id="A", date_start="2019-08-10", source_url="u",
             evidence_type="ksdma_report"),
        dict(flood_id="B", date_start="2019-08-10", source_url="",
             evidence_type="news_archive"),
        dict(flood_id="C", date_start="2019-08-10", source_url="NEEDS_SOURCING",
             evidence_type="news_archive"),
    ])
    res = corroboration.recall(detected, ref)
    assert res.n_reference == 3 and res.n_sourced == 1 and res.n_scored == 1


def test_an_empty_reference_list_blocks_rather_than_scoring_zero():
    """Recall of 0% and 'we have no list' are different statements."""
    res = corroboration.recall(pd.DataFrame({"date": []}), _reference([]))
    assert res.status == "blocked"
    assert np.isnan(res.value)
    assert not res.publishable


def test_a_result_is_not_publishable_while_anything_is_a_stand_in():
    detected = pd.DataFrame({"date": ["2019-08-10"]})
    brief = _reference([dict(flood_id="A", date_start="2019-08-10",
                             source_url="docs/04_PHASE_1_DATA_COLLECTION.md",
                             evidence_type="project_brief")])
    primary = _reference([dict(flood_id="A", date_start="2019-08-10",
                               source_url="http://example.org/ems",
                               evidence_type="ems_activation")])

    assert not corroboration.recall(detected, brief).publishable      # circular source
    assert not corroboration.recall(detected, primary, synthetic=True).publishable
    assert corroboration.recall(detected, primary).publishable


# -------------------------------------------------------------------- precision ---

def test_the_sar_precision_test_refuses_rather_than_guessing():
    """Doc 02 §3.3 wants a measured precision. Without Earth Engine there is no
    measurement, and a plausible-looking number would be worse than none."""
    cfg = Node({"data_mode": {"sentinel1": "unavailable"}})
    with pytest.raises(CredentialsMissing) as exc:
        corroboration.sentinel1_precision(cfg, pd.DataFrame())
    msg = str(exc.value)
    assert "earthengine" in msg.lower()
    assert "Otsu" in msg          # the recipe travels with the refusal


def test_the_precision_recipe_names_the_whole_procedure():
    for token in ("COPERNICUS/S1_GRD", "VV", "Otsu", "reference"):
        assert token in corroboration.GEE_RECIPE
