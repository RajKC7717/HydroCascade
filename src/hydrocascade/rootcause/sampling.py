"""Negative sampling — Doc 06 §2, "the decision that shapes the answer".

There are no non-events in an event catalog, so they have to be constructed,
and the construction materially changes what the model learns. Doc 06 §2 fixes
four rules and requires all three ratios to be reported:

1. **Positives** are (unit, day) pairs where an event occurred.
2. **Negatives** are (unit, day) pairs drawn at random, excluding
   * any unit within 500 m of a recorded event *on that day* — a buffer against
     inventory incompleteness, because "no mapped landslide" is not the same as
     "no landslide";
   * any (unit, day) within ±3 days of an event **in the same unit**, because a
     hillslope that failed on Tuesday was already primed on Monday.
3. **Ratio** 1:5 for the headline, with 1:1 and 1:20 as sensitivity runs.
4. **Stratification** on elevation, so the model cannot win by learning
   "hills versus coastal plain" — which is real, but is not a landslide model.

Rule 4 is the one that does the most work. Without it a classifier reaches a
deceptively high AUC by discovering that events happen where the ground is
steep and high, which every reviewer already knows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Node
from ..logging_setup import get

log = get("rootcause.sampling")


@dataclass
class Sample:
    """One sampled design matrix index: which unit, which day, and the label."""
    frame: pd.DataFrame          # unit_id, date, y
    ratio: int
    n_positive: int
    n_negative: int
    n_excluded_spatial: int
    n_excluded_temporal: int

    def summary(self) -> str:
        return (f"1:{self.ratio} — {self.n_positive:,} positives, "
                f"{self.n_negative:,} negatives; excluded "
                f"{self.n_excluded_spatial:,} (500 m same-day buffer) and "
                f"{self.n_excluded_temporal:,} (+/-3 day same-unit window)")


def _elevation_strata(elev: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Quantile bins of elevation, used to match negatives to positives."""
    finite = elev[np.isfinite(elev)]
    if finite.size == 0:
        return np.zeros(elev.size, dtype=int)
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return np.zeros(elev.size, dtype=int)
    return np.clip(np.digitize(elev, edges[1:-1]), 0, edges.size - 2)


def sample(events: pd.DataFrame, units: pd.DataFrame, dates: pd.DatetimeIndex,
           *, ratio: int = 5, buffer_m: float = 500.0, day_window: int = 3,
           elevation_column: str = "elev_mean_m", seed: int = 0,
           xy: tuple[np.ndarray, np.ndarray] | None = None) -> Sample:
    """Draw ``ratio`` negatives per positive under the Doc 06 §2 rules.

    ``events`` needs ``unit_id`` and ``date``. ``units`` needs ``unit_id`` and
    the elevation column. ``xy`` is the projected coordinate of each unit, in
    the same row order as ``units``; without it the 500 m spatial buffer is
    skipped and the count is reported as zero so nobody mistakes "not applied"
    for "nothing excluded".
    """
    rng = np.random.default_rng(seed)

    unit_ids = units["unit_id"].to_numpy()
    n_units = unit_ids.size
    pos_of_unit = {int(u): i for i, u in enumerate(unit_ids)}

    day_index = {d: i for i, d in enumerate(dates)}
    n_days = len(dates)

    ev = events.dropna(subset=["unit_id"]).copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    ev = ev[ev["date"].isin(day_index)]
    ev["_u"] = [pos_of_unit.get(int(u), -1) for u in ev["unit_id"]]
    ev = ev[ev["_u"] >= 0]
    if ev.empty:
        raise ValueError("no event joins to a unit inside the date range")

    ev["_d"] = [day_index[d] for d in ev["date"]]
    pos_pairs = set(zip(ev["_u"].to_numpy().tolist(), ev["_d"].to_numpy().tolist()))
    n_pos = len(pos_pairs)

    # --- rule 2b: +/- day_window around an event in the SAME unit -----------
    blocked: set[tuple[int, int]] = set()
    for u, d in pos_pairs:
        lo, hi = max(d - day_window, 0), min(d + day_window, n_days - 1)
        for dd in range(lo, hi + 1):
            blocked.add((u, dd))
    n_temporal = len(blocked) - n_pos

    # --- rule 2a: units within buffer_m of an event on that same day --------
    n_spatial = 0
    if xy is not None:
        from scipy.spatial import cKDTree

        x, y = xy
        ok = np.isfinite(x) & np.isfinite(y)
        tree = cKDTree(np.column_stack([x[ok], y[ok]]))
        rows_ok = np.flatnonzero(ok)
        by_day: dict[int, list[int]] = {}
        for u, d in pos_pairs:
            by_day.setdefault(d, []).append(u)
        for d, us in by_day.items():
            src = np.array([u for u in us if ok[u]], dtype=int)
            if src.size == 0:
                continue
            near = tree.query_ball_point(np.column_stack([x[src], y[src]]), r=buffer_m)
            for group in near:
                for j in group:
                    key = (int(rows_ok[j]), d)
                    if key not in pos_pairs and key not in blocked:
                        blocked.add(key)
                        n_spatial += 1
    else:
        log.warning("no unit coordinates supplied — the 500 m same-day buffer "
                    "(Doc 06 §2 rule 2a) was NOT applied")

    # --- rule 4: stratify negatives on elevation ----------------------------
    elev = pd.to_numeric(units[elevation_column], errors="coerce").to_numpy()
    strata = _elevation_strata(elev)
    pos_strata = np.array([strata[u] for u, _ in pos_pairs])
    want = np.bincount(pos_strata, minlength=strata.max() + 1) * ratio

    units_in_stratum = {s: np.flatnonzero(strata == s) for s in range(want.size)}

    negatives: list[tuple[int, int]] = []
    for s, target in enumerate(want):
        if target == 0 or units_in_stratum[s].size == 0:
            continue
        pool = units_in_stratum[s]
        got = 0
        # Rejection sampling: the blocked set is tiny next to units x days, so
        # a draw almost never collides. 50x the target is a generous ceiling.
        for _ in range(int(target) * 50):
            if got >= target:
                break
            u = int(pool[rng.integers(pool.size)])
            d = int(rng.integers(n_days))
            if (u, d) in blocked:
                continue
            blocked.add((u, d))          # never draw the same negative twice
            negatives.append((u, d))
            got += 1
        if got < target:
            log.warning("elevation stratum %d: wanted %d negatives, drew %d",
                        s, target, got)

    rows = [(int(unit_ids[u]), dates[d], 1) for u, d in sorted(pos_pairs)]
    rows += [(int(unit_ids[u]), dates[d], 0) for u, d in negatives]
    frame = pd.DataFrame(rows, columns=["unit_id", "date", "y"])
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    s = Sample(frame, ratio, n_pos, len(negatives), n_spatial, n_temporal)
    log.info("negative sampling %s", s.summary())
    return s
