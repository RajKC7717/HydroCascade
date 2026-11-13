"""Alert levels and threshold optimisation — Doc 09 §2.

    *"Raise an alert when risk exceeds a threshold" is not a specification.
    This is.*

**Four levels, keyed to a unit's own history.** A steep thin-soiled hillside
above Mundakkai and a gentle coastal slope differ in background intensity by
orders of magnitude, so an absolute threshold alarms permanently in one place
and never in the other. Each unit is therefore scored against **its own
2018-2023 distribution**, which also makes the alert interpretable: *conditions
here are more extreme than 99.9% of days on record for this location.*

**Thresholds are not chosen by us.** They are optimised on the validation
period against an explicit cost function, at several cost ratios, and the
trade-off curve is published. Doc 09 §2.3: *"We do not choose the cost ratio.
That is a policy decision belonging to a disaster management authority, not to
a student project. We supply the trade-off curve and a recommended default at
5:1, and we say so."*

**Persistence and hysteresis** stop the level flapping, and they cost lead time.
Doc 09 §2.5 requires that cost to be reported rather than absorbed quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("realtime.alerts")

NORMAL, WATCH, WARNING, EMERGENCY = 0, 1, 2, 3
LEVEL_NAMES = {NORMAL: "Normal", WATCH: "Watch",
               WARNING: "Warning", EMERGENCY: "Emergency"}

#: Doc 09 §2.1 starting definitions, as percentiles of a unit's own history.
DEFAULT_PERCENTILES = {WATCH: 95.0, WARNING: 99.0, EMERGENCY: 99.9}

#: Doc 09 §2.3: report the chosen threshold at each ratio. 5:1 is the
#: recommended default and is labelled as a recommendation, not a finding.
COST_RATIOS = (1.0, 5.0, 20.0)
RECOMMENDED_RATIO = 5.0


@dataclass
class Thresholds:
    """Per-unit intensity cut-points, derived from that unit's own baseline."""
    percentiles: dict[int, float]
    baseline: dict[int, np.ndarray] = field(default_factory=dict)
    cuts: dict[int, dict[int, float]] = field(default_factory=dict)

    def fit(self, unit_ids: np.ndarray, intensity: np.ndarray) -> "Thresholds":
        df = pd.DataFrame({"u": np.asarray(unit_ids), "x": np.asarray(intensity)})
        for u, g in df.groupby("u"):
            v = g["x"].to_numpy()
            v = v[np.isfinite(v)]
            if v.size < 30:
                continue
            self.baseline[int(u)] = v
            self.cuts[int(u)] = {lvl: float(np.percentile(v, p))
                                 for lvl, p in self.percentiles.items()}
        log.info("alert baselines fitted for %d units", len(self.cuts))
        return self

    def level(self, unit_id: int, value: float) -> int:
        c = self.cuts.get(int(unit_id))
        if c is None or not np.isfinite(value):
            return NORMAL
        if value >= c[EMERGENCY]:
            return EMERGENCY
        if value >= c[WARNING]:
            return WARNING
        if value >= c[WATCH]:
            return WATCH
        return NORMAL

    def percentile_of(self, unit_id: int, value: float) -> float:
        """Where this value sits in the unit's own history, as a percentile.

        This is what the dashboard shows, because a raw intensity in
        events per km² per day means nothing to anyone.
        """
        b = self.baseline.get(int(unit_id))
        if b is None or not np.isfinite(value):
            return float("nan")
        return float(100.0 * (b < value).mean())


def apply_hysteresis(levels: np.ndarray, *, raise_after: int = 2,
                     lower_after: int = 4) -> np.ndarray:
    """Doc 09 §2.5 — a level must persist before it is acted on.

    A level is raised only after ``raise_after`` consecutive updates at or above
    it, and lowered only after ``lower_after`` consecutive updates below. The
    asymmetry is deliberate: it is cheap to stay alert a little too long and
    expensive to stand down a little too early.

    **This shortens lead time**, because the first crossing no longer counts.
    Doc 09 §2.5 requires that cost to be measured and reported, which
    :func:`hysteresis_cost` does.
    """
    lv = np.asarray(levels, dtype=int)
    out = np.zeros_like(lv)
    current = NORMAL
    up = down = 0
    for i, want in enumerate(lv):
        if want > current:
            up += 1
            down = 0
            if up >= raise_after:
                current = want
                up = 0
        elif want < current:
            down += 1
            up = 0
            if down >= lower_after:
                current = want
                down = 0
        else:
            up = down = 0
        out[i] = current
    return out


def first_crossing(levels: np.ndarray, level: int) -> int | None:
    """Index of the first time ``level`` is reached, or None."""
    hit = np.flatnonzero(np.asarray(levels) >= level)
    return int(hit[0]) if hit.size else None


def hysteresis_cost(raw_levels: np.ndarray, times: np.ndarray, event_time: float,
                    level: int = WARNING) -> dict:
    """How much lead time the persistence rule costs, in the same units as ``times``."""
    smoothed = apply_hysteresis(raw_levels)
    out = {}
    for name, lv in (("raw", raw_levels), ("with_hysteresis", smoothed)):
        i = first_crossing(lv, level)
        out[name] = (float(event_time - times[i]) if i is not None
                     and times[i] <= event_time else float("nan"))
    out["cost"] = out["raw"] - out["with_hysteresis"]
    return out


# ------------------------------------------------- threshold optimisation ---

@dataclass
class Sweep:
    """One threshold's operating point."""
    threshold: float
    hits: int
    misses: int
    false_alarms: int
    n_negative: int

    @property
    def hit_rate(self) -> float:
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.n_negative if self.n_negative else float("nan")

    def cost(self, ratio: float) -> float:
        """``C_miss * misses + C_FA * false_alarms`` with ``C_FA = 1``."""
        return ratio * self.misses + self.false_alarms


def sweep_thresholds(score: np.ndarray, y: np.ndarray, *, n: int = 120) -> list[Sweep]:
    """Operating points across the full range of the score."""
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y).astype(bool)
    ok = np.isfinite(s)
    s, yy = s[ok], yy[ok]
    if s.size == 0 or not yy.any():
        return []
    qs = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n)))
    out = []
    for thr in qs:
        alarm = s >= thr
        out.append(Sweep(
            threshold=float(thr),
            hits=int((alarm & yy).sum()),
            misses=int((~alarm & yy).sum()),
            false_alarms=int((alarm & ~yy).sum()),
            n_negative=int((~yy).sum())))
    return out


def optimise(score: np.ndarray, y: np.ndarray, *,
             ratios=COST_RATIOS) -> dict[float, Sweep]:
    """Minimise ``C_miss * misses + C_FA * false_alarms`` at each cost ratio.

    Doc 09 §2.3 step 3. The answer is reported at every ratio because the choice
    between them is a policy question, not a statistical one.
    """
    points = sweep_thresholds(score, y)
    if not points:
        return {}
    out = {}
    for r in ratios:
        best = min(points, key=lambda p: p.cost(r))
        out[float(r)] = best
        log.info("cost ratio %4.0f:1 → threshold %.4g, hit rate %.3f, "
                 "false-alarm rate %.3f", r, best.threshold,
                 best.hit_rate, best.false_alarm_rate)
    return out


def tradeoff_curve(score: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """The Doc 09 §2.4 figure's data: what you buy and what you pay."""
    return pd.DataFrame([{
        "threshold": p.threshold,
        "hit_rate": p.hit_rate,
        "false_alarm_rate": p.false_alarm_rate,
        "hits": p.hits, "misses": p.misses, "false_alarms": p.false_alarms,
    } for p in sweep_thresholds(score, y)])
