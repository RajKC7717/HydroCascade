"""Phase 5-6 — the guards, the ablation bookkeeping, and the alert logic.

Doc 08 §4.1 asks for the leakage assertion to be *"written by Ronit,
independently audited by Raj... One person writes it, a different person tries
to break it."* The tests below are that audit: each one is a deliberate attempt
to slip post-cutoff data past the guard.

A guard that only gets tested on clean input has not been tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrocascade.experiments import ablations, splits
from hydrocascade.experiments.splits import LeakageError
from hydrocascade.realtime import alerts


# ------------------------------------------------------- the leakage guard ---

def test_clean_training_data_passes_and_reports_what_it_saw():
    dates = pd.date_range("2018-06-01", "2022-12-31", freq="30D")
    n = splits.assert_no_leakage(dates, splits.TRAIN_END)
    assert n == len(dates)


def test_one_day_past_the_cutoff_is_caught():
    """Off-by-one is the realistic failure, not a year of leakage."""
    dates = list(pd.date_range("2018-06-01", "2022-12-31", freq="30D"))
    dates.append(pd.Timestamp("2023-01-01"))
    with pytest.raises(LeakageError, match="after the"):
        splits.assert_no_leakage(dates, splits.TRAIN_END)


def test_the_wayanad_event_cannot_reach_a_training_set():
    """The single most important thing the guard exists to prevent."""
    dates = list(pd.date_range("2018-06-01", "2022-12-31", freq="60D"))
    dates.append(pd.Timestamp("2024-07-30"))
    with pytest.raises(LeakageError):
        splits.assert_holdout_untouched(dates)


def test_an_empty_check_is_not_a_passing_check():
    """A guard handed nothing must raise, not congratulate itself.

    This is the quiet failure mode: a filter upstream returns an empty frame,
    the assertion sees no violations, and the run reports 'leakage guard clean'
    having verified nothing at all.
    """
    with pytest.raises(LeakageError, match="not a passing check"):
        splits.assert_no_leakage([], splits.TRAIN_END)


def test_unparseable_dates_do_not_silently_pass():
    with pytest.raises(LeakageError):
        splits.assert_no_leakage(["not-a-date", None], splits.TRAIN_END)


def test_the_three_splits_do_not_overlap_and_leave_no_gap():
    assert splits.TRAIN_END < splits.VALID_START <= splits.VALID_END < splits.TEST_START
    assert (pd.Timestamp(splits.VALID_START)
            - pd.Timestamp(splits.TRAIN_END)).days == 1
    assert (pd.Timestamp(splits.TEST_START)
            - pd.Timestamp(splits.VALID_END)).days == 1


def test_describe_counts_events_into_the_right_split():
    ev = pd.DataFrame({
        "date": ["2019-07-01", "2023-06-01", "2024-07-30"],
        "hazard_type": ["L", "F", "L"]})
    d = splits.describe(ev).set_index("split")
    assert d.loc["train", "events"] == 1
    assert d.loc["validation", "events"] == 1
    assert d.loc["test", "events"] == 1
    assert d.loc["test", "landslides"] == 1


# --------------------------------------------------------------- ablations ---

def test_every_ablation_removes_exactly_one_thing():
    """An ablation that changes two components cannot attribute the difference."""
    for key, (_removes, _claim, overrides) in ablations.ABLATIONS.items():
        assert len(overrides) == 1, f"{key} changes {len(overrides)} components"


def test_the_ladder_is_monotone_in_components():
    """M0 through M5 must only ever add, never swap — or nesting fails and the
    likelihood-ratio test between consecutive rows is invalid."""
    flags = ["background", "rainfall", "marks", "cross", "network",
             "travel_time", "runout_gate", "dilution"]
    prev = None
    for key in ("M0", "M1", "M2", "M3", "M4", "M5"):
        kw = ablations.LADDER[key][1]
        on = {f for f in flags if kw.get(f)}
        if prev is not None:
            assert prev <= on, f"{key} removed something the previous model had"
        prev = on


def test_likelihood_ratio_is_positive_for_a_better_nested_model():
    simple = ablations.Run("M3", "", None, train_ll=-1000.0, n_params=5)
    rich = ablations.Run("M4", "", None, train_ll=-980.0, n_params=7)
    stat, df, p = ablations.likelihood_ratio(simple, rich)
    assert stat == pytest.approx(40.0)
    assert df == 2
    assert 0.0 <= p <= 1.0


def test_a_richer_model_fitting_worse_is_reported_not_hidden():
    """Impossible at the true optimum, so it is an optimisation failure."""
    simple = ablations.Run("M3", "", None, train_ll=-980.0, n_params=5)
    rich = ablations.Run("M4", "", None, train_ll=-1000.0, n_params=7)
    stat, _, p = ablations.likelihood_ratio(simple, rich)
    assert stat < 0
    assert np.isnan(p), "a negative statistic must not be given a p-value"


def test_no_difference_is_reported_as_no_difference():
    """Doc 08 §3: an idea that turns out not to matter is a finding."""
    assert "no detectable difference" in ablations.interpret(0.4)
    assert "earns its place" in ablations.interpret(-25.0)
    assert "hurts" in ablations.interpret(25.0)


# ------------------------------------------------------------- alert logic ---

def test_levels_follow_the_unit_s_own_distribution():
    rng = np.random.default_rng(0)
    base = rng.lognormal(0, 1, 5000)
    thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
    thr.fit(np.zeros(base.size, dtype=int), base)

    assert thr.level(0, float(np.percentile(base, 50))) == alerts.NORMAL
    assert thr.level(0, float(np.percentile(base, 97))) == alerts.WATCH
    assert thr.level(0, float(np.percentile(base, 99.5))) == alerts.WARNING
    assert thr.level(0, float(base.max()) * 10) == alerts.EMERGENCY


def test_two_units_with_different_scales_get_comparable_levels():
    """The whole reason for percentiles instead of absolute intensity.

    A steep hillside and a coastal slope differ by orders of magnitude in
    background rate. Under an absolute threshold one alarms permanently and the
    other never does.
    """
    rng = np.random.default_rng(1)
    steep = rng.lognormal(0, 1, 3000) * 1000.0
    gentle = rng.lognormal(0, 1, 3000) * 0.001
    thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
    thr.fit(np.concatenate([np.zeros(3000, int), np.ones(3000, int)]),
            np.concatenate([steep, gentle]))

    assert thr.level(0, float(np.percentile(steep, 99.5))) == alerts.WARNING
    assert thr.level(1, float(np.percentile(gentle, 99.5))) == alerts.WARNING


def test_an_unknown_unit_is_normal_not_a_crash():
    thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
    assert thr.level(9999, 1e9) == alerts.NORMAL


def test_hysteresis_needs_two_updates_to_raise():
    raw = np.array([0, 2, 0, 0, 0, 0])            # a single spike
    out = alerts.apply_hysteresis(raw, raise_after=2, lower_after=4)
    assert out.max() == 0, "one reading must not raise a level"


def test_hysteresis_raises_on_a_sustained_signal():
    raw = np.array([0, 2, 2, 2, 2])
    out = alerts.apply_hysteresis(raw, raise_after=2, lower_after=4)
    assert out[-1] == 2


def test_hysteresis_is_slower_to_stand_down_than_to_raise():
    """Asymmetric on purpose: staying alert too long is cheap; standing down
    too early is not."""
    raw = np.array([2, 2, 2, 2, 0, 0, 0, 0, 0, 0])
    out = alerts.apply_hysteresis(raw, raise_after=2, lower_after=4)

    raised = alerts.first_crossing(out, alerts.WARNING)
    assert raised == 1, "two consecutive readings should raise on the second"

    # It must not fall back on the first sub-threshold reading. Readings 4, 5, 6
    # and 7 are the four consecutive ones the rule needs, so the level holds
    # through 6 and drops at 7 — three updates of extra caution bought by the
    # asymmetry.
    fell = np.flatnonzero((out == 0) & (np.arange(out.size) > raised))
    assert fell.size and fell[0] == 7
    assert out[4] == alerts.WARNING and out[6] == alerts.WARNING


def test_hysteresis_costs_lead_time_and_the_cost_is_measured():
    """Doc 09 §2.5 requires the cost to be reported, not absorbed quietly."""
    raw = np.array([0, 2, 2, 2, 2, 2])
    times = np.arange(6, dtype=float)
    c = alerts.hysteresis_cost(raw, times, event_time=6.0, level=alerts.WARNING)
    assert c["raw"] > c["with_hysteresis"]
    assert c["cost"] > 0


def test_a_higher_miss_cost_buys_a_lower_threshold():
    """The cost ratio is a policy dial, and it must actually move the answer."""
    rng = np.random.default_rng(2)
    score = rng.uniform(0, 1, 2000)
    y = rng.uniform(0, 1, 2000) < (0.02 + 0.2 * score)
    chosen = alerts.optimise(score, y, ratios=(1.0, 20.0))
    assert chosen[20.0].threshold <= chosen[1.0].threshold
    assert chosen[20.0].hit_rate >= chosen[1.0].hit_rate


def test_the_tradeoff_curve_is_monotone_in_the_right_direction():
    rng = np.random.default_rng(3)
    score = rng.uniform(0, 1, 1500)
    y = rng.uniform(0, 1, 1500) < (0.05 + 0.3 * score)
    curve = alerts.tradeoff_curve(score, y).sort_values("false_alarm_rate")
    assert curve["hit_rate"].is_monotonic_increasing


def test_hit_rate_and_false_alarm_rate_are_both_defined():
    """Doc 08 §7: they are reported together, always. A sweep that produced
    only one of them would let a lead-time claim stand alone."""
    rng = np.random.default_rng(4)
    score = rng.uniform(0, 1, 500)
    y = rng.uniform(0, 1, 500) < 0.1
    for p in alerts.sweep_thresholds(score, y):
        assert np.isfinite(p.hit_rate)
        assert np.isfinite(p.false_alarm_rate)


# ------------------------------------------------------------- transfer ---

def test_transfer_reports_itself_as_not_built():
    """The one experiment that is deferred must say so rather than look done."""
    from hydrocascade.experiments import transfer
    s = transfer.status()
    assert s["region_built"] is False
    assert s["events_sourced"] == 0
    assert "disk" in s["blocked_because"]


def test_transfer_degradation_names_a_verdict():
    from hydrocascade.experiments import transfer
    home = transfer.TransferResult("kerala", -1000.0, 1000, -1.0, 0.5)
    away = transfer.TransferResult("maharashtra", -600.0, 500, -1.2, 0.5)
    d = transfer.degradation(home, away)
    assert "degrades" in d["verdict"] or "transfers" in d["verdict"]
    assert d["delta_per_event"] == pytest.approx(-0.2)
