"""Leave-one-cascade-out — Doc 08 §4.2.

    *This is the answer to "you designed the model around Wayanad and then
    tested it on Wayanad."*

For each documented cascade event: refit excluding a ±30-day window around it
across the whole region, roll forward through the event, and record lead time,
alert level reached, and false alarms in the preceding 30 days.

**The output is a distribution of lead times across six events, not one
number.** One lead time from one event is an anecdote; six with a median and a
range is a result — and if the range is wide, that is the honest thing to
report rather than quoting the best of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Node
from ..logging_setup import get

log = get("experiments.loco")

#: Doc 08 §4.2. Dates as the documents give them; where only a month is known
#: the window is the month, because sharpening it would be inventing precision.
CASCADE_EVENTS = [
    {"name": "Kerala monsoon", "date": "2018-08-16", "district": "statewide",
     "note": "Concurrent floods and thousands of landslides"},
    {"name": "Kavalappara", "date": "2019-08-08", "district": "Malappuram",
     "note": "Doc 08 §4.2 gives the month only; date from the Doc 04 §5 seed list"},
    {"name": "Puthumala", "date": "2019-08-08", "district": "Wayanad",
     "note": "**Date disputed** between 8 and 29 August 2019 — resolve from GSI "
             "before quoting this row. 2-3 km from the 2024 crown"},
    {"name": "Pettimudi", "date": "2020-08-06", "district": "Idukki",
     "note": "May fall outside the Chaliyar pilot basin"},
    {"name": "Kokkayar/Plappally/Kavali", "date": "2021-10-16", "district": "Idukki",
     "note": "Three events, three different causal mechanisms"},
    {"name": "Mundakkai-Chooralmala", "date": "2024-07-30", "district": "Wayanad",
     "note": "**Held out entirely**, not just LOCO"},
]


@dataclass
class LocoResult:
    name: str
    date: str
    unit_id: int = -1
    lead_hours: dict[str, float] = field(default_factory=dict)
    max_level: int = 0
    false_alarms_30d: int = 0
    n_train_events: int = 0
    status: str = "ok"
    note: str = ""


def _unit_for(cfg: Node, name: str) -> int:
    """The slope unit for a named event, or -1 if its coordinates are unsourced.

    Only Mundakkai has sourced coordinates. The rest are in
    ``events_PENDING.csv`` precisely because inventing a location for them would
    be worse than reporting that the row cannot be scored — so this returns -1
    and :func:`run` records the row as BLOCKED rather than guessing.
    """
    from ..spatial.slope_units import SlopeUnits
    from ..spatial.terrain import Routing

    if not name.startswith("Mundakkai"):
        return -1
    lm = cfg.landmarks.mundakkai_crown
    rt = Routing()
    cell = rt.index_of_lonlat(float(lm["lon"]), float(lm["lat"]))
    return int(SlopeUnits().unit_of_cell[cell])


def run(cfg: Node, params: dict, spec, *, window_days: int = 30,
        refit: bool = False) -> list[LocoResult]:
    """Roll through each cascade event with that event's window excluded.

    ``refit=True`` re-runs the full optimisation per event, which is what Doc 08
    §4.2 specifies and costs six fits. With ``refit=False`` the fitted
    parameters are reused and only the *history* is masked — cheaper, and still
    excludes the event itself from everything the model conditions on, but it
    does not remove the event's influence on the parameters. The distinction is
    recorded in the report rather than glossed.
    """
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..realtime import alerts
    from ..realtime.predict import Predictor

    pred = Predictor(cfg, params, spec)
    cat = schema.read(EVENTS_PATH)
    cat["_d"] = pd.to_datetime(cat["date"], errors="coerce")

    out: list[LocoResult] = []
    for spec_ev in CASCADE_EVENTS:
        r = LocoResult(name=spec_ev["name"], date=spec_ev["date"],
                       note=spec_ev["note"])
        when = pd.Timestamp(spec_ev["date"])
        unit = _unit_for(cfg, spec_ev["name"])
        r.unit_id = unit
        if unit < 0:
            r.status = "BLOCKED"
            r.note += (". Not scorable: this event's coordinates are still "
                       "unsourced and live in events_PENDING.csv.")
            log.warning("LOCO %-26s BLOCKED — no sourced coordinates",
                        spec_ev["name"])
            out.append(r)
            continue

        lo = when - pd.Timedelta(days=window_days)
        hi = when + pd.Timedelta(days=window_days)
        mask = (cat["_d"] < lo) | (cat["_d"] > hi)
        train = cat[mask & (cat["_d"] < when)]
        r.n_train_events = int(len(train))

        history = [pred.event_from_row(row) for _, row in train.iterrows()
                   if pd.notna(pd.to_numeric(row["spatial_unit_id"], errors="coerce"))]

        base_days = pd.date_range(str(cfg.period.start),
                                  (when - pd.Timedelta(days=365)).date(), freq="D")
        if base_days.size < 90:
            r.status = "BLOCKED"
            r.note += ". Not scorable: fewer than 90 days of prior baseline."
            out.append(r)
            continue

        base = np.array([pred.landslide_intensity(unit, d, history)
                         for d in base_days])
        thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
        thr.fit(np.full(base.size, unit), base)

        roll = pd.date_range(when - pd.Timedelta(days=30), when, freq="D")
        lam = np.array([pred.landslide_intensity(unit, d, history) for d in roll])
        lv = alerts.apply_hysteresis(np.array([thr.level(unit, v) for v in lam]))
        r.max_level = int(lv.max())

        for nm, level in (("Watch", alerts.WATCH), ("Warning", alerts.WARNING),
                          ("Emergency", alerts.EMERGENCY)):
            i = alerts.first_crossing(lv, level)
            r.lead_hours[nm] = (float((when - roll[i]).total_seconds() / 3600.0)
                                if i is not None else float("nan"))

        pre = pd.date_range(when - pd.Timedelta(days=60),
                            when - pd.Timedelta(days=31), freq="D")
        pre_lam = np.array([pred.landslide_intensity(unit, d, history) for d in pre])
        pre_lv = alerts.apply_hysteresis(
            np.array([thr.level(unit, v) for v in pre_lam]))
        r.false_alarms_30d = int((pre_lv >= alerts.WARNING).sum())

        log.info("LOCO %-26s unit %5d  max level %-9s  Warning lead %s  "
                 "false alarms %d", r.name, unit,
                 alerts.LEVEL_NAMES[r.max_level],
                 f"{r.lead_hours['Warning']:.0f} h"
                 if np.isfinite(r.lead_hours["Warning"]) else "never",
                 r.false_alarms_30d)
        out.append(r)
    return out


def summarise(results: list[LocoResult]) -> dict:
    """Median and range of lead time — the distribution, not one number."""
    scored = [r for r in results if r.status == "ok"]
    leads = np.array([r.lead_hours.get("Warning", np.nan) for r in scored])
    ok = np.isfinite(leads)
    return {
        "n_events": len(results),
        "n_scored": len(scored),
        "n_blocked": len(results) - len(scored),
        "n_with_warning": int(ok.sum()),
        "median_lead_h": float(np.median(leads[ok])) if ok.any() else float("nan"),
        "min_lead_h": float(leads[ok].min()) if ok.any() else float("nan"),
        "max_lead_h": float(leads[ok].max()) if ok.any() else float("nan"),
        "mean_false_alarms_30d": float(np.mean([r.false_alarms_30d for r in scored]))
        if scored else float("nan"),
    }
