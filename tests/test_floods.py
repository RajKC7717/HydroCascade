"""FR-02: flood detection from discharge, verified on synthetic hydrographs.

Doc 11 §1 names the verification method for FR-02 as *"unit test on a synthetic
hydrograph with known exceedances"*. Each hydrograph below has its correct
answer worked out by hand in the docstring.

The declustering rule is the part worth guarding hardest: Doc 02 §3 warns that
without it "a single week-long flood would be counted as five events and would
badly corrupt the self-excitation parameters".
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrocascade.ingest import floods

Q2 = 100.0


# ------------------------------------------------------------- L-moments ---

def test_lmoments_of_a_known_sample():
    """L-moments of 1..5: l1 is the mean, l2 the expected linear combination."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    l1, l2 = floods.lmoments_1_2(x)
    assert l1 == pytest.approx(3.0)
    # b1 = mean_i[(i-1)/(n-1) * x_i] = (0*1 + .25*2 + .5*3 + .75*4 + 1*5)/5 = 2.0
    assert l2 == pytest.approx(2 * 2.0 - 3.0)


def test_gumbel_recovers_its_own_parameters():
    """Fit a large sample drawn from a known Gumbel and get the truth back."""
    rng = np.random.default_rng(42)
    xi_true, alpha_true = 250.0, 60.0
    sample = xi_true - alpha_true * np.log(-np.log(rng.uniform(size=200_000)))
    xi, alpha = floods.fit_gumbel_lmoments(sample)
    assert xi == pytest.approx(xi_true, rel=0.01)
    assert alpha == pytest.approx(alpha_true, rel=0.01)


def test_gumbel_quantiles_are_ordered_and_correct():
    """Q2 < Q5 < Q20, at the analytic reduced variates."""
    xi, alpha = 100.0, 25.0
    q2 = floods.gumbel_quantile(xi, alpha, 2)
    q5 = floods.gumbel_quantile(xi, alpha, 5)
    q20 = floods.gumbel_quantile(xi, alpha, 20)
    assert q2 < q5 < q20
    # reduced variate -ln(-ln(1 - 1/T))
    assert q2 == pytest.approx(xi + alpha * 0.3665129, rel=1e-5)
    assert q5 == pytest.approx(xi + alpha * 1.4999394, rel=1e-5)
    assert q20 == pytest.approx(xi + alpha * 2.9702389, rel=1e-5)


def test_return_period_must_exceed_one_year():
    with pytest.raises(ValueError):
        floods.gumbel_quantile(100.0, 25.0, 1.0)


# ------------------------------------------------------ event detection ---

def test_no_exceedance_gives_no_event():
    q = np.full(30, 50.0)
    assert floods.detect_events(q, Q2) == []


def test_single_episode_dates_from_the_first_exceedance():
    """Days 5-9 above threshold: one event starting on day 5, peak on day 7."""
    q = np.full(20, 50.0)
    q[5:10] = [120, 180, 260, 150, 110]
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_index == 5
    assert ev.peak_index == 7
    assert ev.end_index == 9
    assert ev.q_peak == pytest.approx(260.0)
    assert ev.n_days_above == 5


def test_gap_shorter_than_the_window_stays_one_event():
    """Two peaks separated by only two sub-threshold days are ONE flood.

    This is the rule that stops a single week-long flood being counted five
    times over.
    """
    q = np.full(30, 50.0)
    q[5:8] = 200.0        # above
    q[8:10] = 50.0        # two days below — not enough to close the episode
    q[10:13] = 300.0      # above again
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert len(events) == 1
    assert events[0].start_index == 5
    assert events[0].end_index == 12
    assert events[0].q_peak == pytest.approx(300.0)
    assert events[0].n_days_above == 6      # the two below days are not counted


def test_gap_of_exactly_the_window_splits_the_event():
    """Three consecutive sub-threshold days do close the episode."""
    q = np.full(30, 50.0)
    q[5:8] = 200.0
    q[8:11] = 50.0        # exactly three days below
    q[11:14] = 300.0
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert len(events) == 2
    assert events[0].start_index == 5
    assert events[1].start_index == 11
    assert events[1].q_peak == pytest.approx(300.0)


def test_decluster_window_is_configurable():
    """The same series gives one event at 5 days and two at 3 (Doc 02 §3.5)."""
    q = np.full(30, 50.0)
    q[5:8] = 200.0
    q[8:12] = 50.0        # four days below
    q[12:15] = 300.0
    assert len(floods.detect_events(q, Q2, decluster_days=3)) == 2
    assert len(floods.detect_events(q, Q2, decluster_days=5)) == 1


def test_episode_already_in_progress_on_the_first_day():
    q = np.full(15, 50.0)
    q[0:4] = 400.0
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert len(events) == 1
    assert events[0].start_index == 0
    assert events[0].q_peak == pytest.approx(400.0)


def test_episode_running_past_the_end_of_the_record():
    q = np.full(10, 50.0)
    q[7:] = 500.0
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert len(events) == 1
    assert events[0].start_index == 7
    assert events[0].end_index == 9


def test_exceedance_is_strict():
    """Discharge exactly at Q2 is not an exceedance."""
    q = np.full(10, Q2)
    assert floods.detect_events(q, Q2) == []


def test_three_separate_floods_in_one_monsoon():
    q = np.full(60, 40.0)
    q[5:8] = 150.0
    q[20:24] = 220.0
    q[40:42] = 900.0
    events = floods.detect_events(q, Q2, decluster_days=3)
    assert [e.start_index for e in events] == [5, 20, 40]
    assert [e.q_peak for e in events] == pytest.approx([150.0, 220.0, 900.0])


# ------------------------------------------------------- marks and classes ---

def test_mark_is_zero_at_the_threshold_and_grows_with_the_peak():
    assert floods.mark_value(Q2, Q2) == pytest.approx(0.0)
    assert floods.mark_value(Q2 * np.e, Q2) == pytest.approx(1.0)
    assert floods.mark_value(300.0, Q2) > floods.mark_value(200.0, Q2)


def test_severity_bands():
    q2, q5, q20 = 100.0, 180.0, 300.0
    assert floods.severity_class(120.0, q2, q5, q20) == "minor"
    assert floods.severity_class(180.0, q2, q5, q20) == "major"
    assert floods.severity_class(250.0, q2, q5, q20) == "major"
    assert floods.severity_class(300.0, q2, q5, q20) == "extreme"
    assert floods.severity_class(1200.0, q2, q5, q20) == "extreme"
