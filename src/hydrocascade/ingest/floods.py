"""Flood event detection from daily discharge (Doc 02 §3, Doc 04 §4).

This was the weakest definition in the earlier plan, so it is now a procedure
with no judgement calls in it:

1. **Thresholds.** For each river cell, take the annual maxima of daily
   discharge, fit a Gumbel distribution **by the method of L-moments**, and read
   off Q2, Q5 and Q20. This is GloFAS's own threshold procedure — adopted
   deliberately so our flood definition matches an operational system rather
   than being invented for this project.
2. **Detection.** A flood event occurs on the first day of an episode in which
   daily discharge exceeds Q2.
3. **Declustering.** An episode ends only after discharge stays below Q2 for
   three consecutive days. Without this, one week-long flood counts as five
   events and badly corrupts the self-excitation parameters.
4. **Mark and class.** ``m_F = log(Q_peak / Q2)``; minor / major / extreme from
   the Q2-Q5-Q20 bands.
5. **Eligibility.** Only reaches above 50 km2 contributing area (Doc 02 §3.1
   step 5), with a sensitivity check at 250 km2.

Every threshold above is a config value, so the Doc 02 §3.5 sensitivity analysis
(Q5 instead of Q2, 5-day declustering, 250 km2 eligibility) is a config edit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("ingest.floods")

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------- thresholds ---

def lmoments_1_2(x: np.ndarray) -> tuple[float, float]:
    """First two sample L-moments, via unbiased probability weighted moments.

    ``l1`` is the mean; ``l2`` is a robust measure of scale. L-moments are used
    rather than maximum likelihood because they are far less sensitive to a
    single outlying annual maximum, which is exactly the situation in a record
    containing 2018 and 2024.
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n < 2:
        raise ValueError("need at least two values for L-moments")
    i = np.arange(1, n + 1, dtype=np.float64)
    b0 = x.mean()
    b1 = float(np.sum((i - 1.0) / (n - 1.0) * x) / n)
    return b0, 2.0 * b1 - b0


def fit_gumbel_lmoments(annual_maxima: np.ndarray) -> tuple[float, float]:
    """Gumbel (EV1) location and scale by L-moments.

    ``alpha = l2 / ln 2`` and ``xi = l1 - gamma * alpha``.
    """
    l1, l2 = lmoments_1_2(annual_maxima)
    if l2 <= 0:
        raise ValueError("non-positive L-scale; the record is degenerate")
    alpha = l2 / np.log(2.0)
    xi = l1 - EULER_GAMMA * alpha
    return float(xi), float(alpha)


def gumbel_quantile(xi: float, alpha: float, return_period: float) -> float:
    """Discharge with the given return period in years."""
    if return_period <= 1:
        raise ValueError("return period must exceed 1 year")
    f = 1.0 - 1.0 / return_period
    return float(xi - alpha * np.log(-np.log(f)))


def return_period_thresholds(annual_maxima: np.ndarray,
                             periods=(2, 5, 20)) -> dict[str, float]:
    """``{'Q2': ..., 'Q5': ..., 'Q20': ...}`` for one reach."""
    xi, alpha = fit_gumbel_lmoments(annual_maxima)
    return {f"Q{int(p)}": gumbel_quantile(xi, alpha, p) for p in periods}


# ------------------------------------------------------------ event detection ---

@dataclass
class FloodEvent:
    start_index: int
    peak_index: int
    end_index: int
    q_peak: float
    n_days_above: int

    def as_dict(self) -> dict:
        return asdict(self)


def detect_events(discharge: np.ndarray, q2: float,
                  decluster_days: int = 3) -> list[FloodEvent]:
    """Q2-exceedance episodes with declustering.

    An episode opens on the first day above ``q2`` and stays open until
    discharge has been below ``q2`` for ``decluster_days`` **consecutive** days.
    A gap shorter than that does not split the episode.

    Returns events in chronological order. ``start_index`` is the first
    exceedance day — the event date. ``end_index`` is the last day above the
    threshold, not the day the decluster window closes.
    """
    q = np.asarray(discharge, dtype=np.float64)
    above = q > q2

    events: list[FloodEvent] = []
    i = 0
    n = q.size
    while i < n:
        if not above[i]:
            i += 1
            continue

        start = i
        last_above = i
        below_run = 0
        j = i
        while j < n:
            if above[j]:
                last_above = j
                below_run = 0
            else:
                below_run += 1
                if below_run >= decluster_days:
                    break
            j += 1

        window = q[start:last_above + 1]
        peak_off = int(np.argmax(window))
        events.append(FloodEvent(
            start_index=start,
            peak_index=start + peak_off,
            end_index=last_above,
            q_peak=float(window.max()),
            n_days_above=int(above[start:last_above + 1].sum()),
        ))
        i = last_above + decluster_days + 1

    return events


def expected_exceedance_rate(return_period: float) -> float:
    """Independent episodes per reach-year expected above ``Q_T``.

    If episodes above ``Q_T`` arrive as a Poisson process with rate ``lam``, the
    annual maximum stays below ``Q_T`` with probability ``exp(-lam)``. The
    return period fixes that at ``1 - 1/T``, so::

        lam = -ln(1 - 1/T)

    which is 0.69 for Q2 and 0.22 for Q5 — *not* 1.0 and 1.0. The obvious
    intuition ("a 2-year event happens every other year, so about one a year")
    is wrong by 30% at Q2 and by a factor of four at Q5, which is exactly the
    kind of slip that makes a sanity check fire on healthy data.
    """
    if return_period <= 1:
        raise ValueError("return period must exceed 1 year")
    return float(-np.log1p(-1.0 / return_period))


def severity_class(q_peak: float, q2: float, q5: float, q20: float) -> str:
    """Doc 02 §3.1 step 4: minor (Q2-Q5), major (Q5-Q20), extreme (above Q20)."""
    if q_peak >= q20:
        return "extreme"
    if q_peak >= q5:
        return "major"
    return "minor"


def mark_value(q_peak: float, q2: float) -> float:
    """``m_F = log(Q_peak / Q2)`` — dimensionless, zero at the threshold."""
    return float(np.log(max(q_peak, 1e-12) / max(q2, 1e-12)))


# ---------------------------------------------------------------- the stage ---

def detect_for_all_reaches(discharge: pd.DataFrame, areas: pd.Series, cfg,
                           ) -> pd.DataFrame:
    """Run the whole procedure over a (date x reach) discharge table.

    ``discharge`` is indexed by date with one column per reach id; ``areas`` maps
    reach id to contributing area in km2. Returns one row per detected event.
    """
    fcfg = cfg.floods
    periods = tuple(int(p) for p in fcfg.return_periods)
    thr_rp = int(fcfg.threshold_return_period)
    decluster = int(fcfg.decluster_days)
    min_area = float(fcfg.min_contributing_area_km2)
    min_years = int(fcfg.min_years_for_gumbel)

    eligible = areas[areas > min_area].index
    eligible = [r for r in eligible if r in discharge.columns]
    log.info("eligibility: %d of %d reaches above %.0f km2",
             len(eligible), len(areas), min_area)
    if not eligible:
        raise RuntimeError(f"no reach exceeds the {min_area} km2 eligibility threshold")

    dates = pd.DatetimeIndex(discharge.index)
    years = dates.year.to_numpy()
    rows = []
    n_short = 0

    for reach in eligible:
        q = discharge[reach].to_numpy(dtype=np.float64)
        # annual maxima over the full available record
        amax = pd.Series(q, index=years).groupby(level=0).max().to_numpy()
        if amax.size < min_years:
            n_short += 1
            continue
        try:
            thr = return_period_thresholds(amax, periods)
        except ValueError:
            continue

        q_thr = thr[f"Q{thr_rp}"]
        for ev in detect_events(q, q_thr, decluster):
            rows.append({
                "reach_id": int(reach),
                "date": dates[ev.start_index].date().isoformat(),
                "peak_date": dates[ev.peak_index].date().isoformat(),
                "q_peak": ev.q_peak,
                "q2": thr["Q2"], "q5": thr["Q5"], "q20": thr["Q20"],
                "mark_value": mark_value(ev.q_peak, thr["Q2"]),
                "severity_class": severity_class(ev.q_peak, thr["Q2"], thr["Q5"], thr["Q20"]),
                "n_days_above": ev.n_days_above,
                "area_km2": float(areas[reach]),
            })

    if n_short:
        log.warning("%d reaches skipped: fewer than %d years of record", n_short, min_years)
    df = pd.DataFrame(rows)
    n_reaches = df["reach_id"].nunique() if len(df) else 0
    log.info("detected %d flood events on %d reaches (threshold Q%d, %d-day declustering)",
             len(df), n_reaches, thr_rp, decluster)

    # The exceedance rate is the first diagnostic to read: it is fixed by the
    # return period alone, so a rate far from the theoretical one indicts the
    # discharge series rather than the detection rule. The tolerance is
    # multiplicative because the expectation itself changes by a factor of three
    # between Q2 and Q5.
    if n_reaches:
        n_years = max(dates.year.max() - dates.year.min() + 1, 1)
        rate = len(df) / n_reaches / n_years
        expected = expected_exceedance_rate(thr_rp)
        log.info("exceedance rate %.2f events per reach-year (Q%d implies %.2f)",
                 rate, thr_rp, expected)
        if not 0.4 * expected <= rate <= 2.5 * expected:
            log.warning("exceedance rate %.2f is far from the %.2f per reach-year a "
                        "Q%d threshold implies — check the discharge series",
                        rate, expected, thr_rp)
    return df
