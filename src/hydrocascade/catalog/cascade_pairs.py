"""``data/catalog/cascade_pairs.csv`` — the Doc 04 §5 linking table.

This table is not training data. Doc 04 §5 is explicit:

    Only `confirmed` and `probable` pairs are used to *evaluate* cascade
    detection. All events are used to *fit* the model — the model discovers
    pairs on its own; the table is for validation.

That one sentence decides what may go in each column.

``network_distance_km`` is **geometry**: how far apart two places are along the
channel network. It describes the basin, not the event, so computing it from our
own DEM is fine and it is filled in here.

``lag_hours`` is **evidence**: how long the downstream flooding actually took to
arrive. We could compute a travel time from the same network in one line — and
that is exactly why we must not. Writing a model-derived lag into the table the
model is scored against would quietly turn a prediction into its own ground
truth. It stays blank until someone sources it from a field report, and the
reference travel time is recorded in ``notes`` where nobody can mistake it for
an observation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("catalog.cascade_pairs")

PAIRS_PATH = paths.CATALOG / "cascade_pairs.csv"

COLUMNS = ("pair_id", "landslide_event_id", "flood_event_id", "lag_hours",
           "network_distance_km", "evidence_level", "evidence_source", "notes")

#: Doc 04 §5's starter list of cascade candidates, one row each. ``landslide``
#: is matched against ``events.csv`` by date; where the landslide is still
#: unsourced the row is written with a blank id and becomes a work item rather
#: than being dropped.
_STARTER = [
    dict(pair_id="P-001", date="2018-08-16", place="Kerala monsoon",
         evidence_level="candidate", evidence_source="DOC04_S5",
         notes="Bulk training period: concurrent flood and landslide across the "
               "state. Not a single identifiable pair — resolve into individual "
               "pairs once the 2018 inventory is downloaded."),
    dict(pair_id="P-002", date="2019-08-08", place="Kavalappara, Malappuram",
         evidence_level="candidate", evidence_source="DOC04_S5",
         notes="Doc 04 §5 test event. Needs landslide coordinates before the "
               "downstream reach can be identified."),
    dict(pair_id="P-003", date="", place="Puthumala, Wayanad",
         evidence_level="candidate", evidence_source="DOC04_S5",
         notes="Doc 04 §5 provenance warning: sources disagree between 8 and 29 "
               "August 2019. Resolve from GSI or the primary field report and "
               "record which source was trusted. 2-3 km from the 2024 crown."),
    dict(pair_id="P-004", date="2020-08-06", place="Pettimudi, Idukki",
         evidence_level="candidate", evidence_source="DOC04_S5",
         notes="Doc 04 §5 test event. Idukki may fall outside the Chaliyar pilot "
               "basin; confirm against aoi.gpkg before scoring."),
    dict(pair_id="P-005", date="2021-10-16", place="Kokkayar / Plappally / Kavali",
         evidence_level="candidate", evidence_source="DOC04_S5",
         notes="Doc 04 §5 lists three events with three different causes. Split "
               "into three pairs once each is sourced."),
    dict(pair_id="P-006", date="2024-07-30", place="Mundakkai-Chooralmala",
         evidence_level="probable", evidence_source="DOC04_S5; GSI_BHUSANKET_FIR",
         downstream_landmark="chooralmala_bridge",
         notes="FINAL HELD-OUT TEST. Correct spatial and temporal relationship "
               "plus a supporting report (the bridge at Chooralmala was destroyed "
               "the same night). Upgrade to `confirmed` once the GSI Bhusanket FIR "
               "is transcribed and states the causation explicitly."),
]


def _network_distance_km(cfg: Node, lat: float, lon: float,
                         landmark: str) -> tuple[float, float, str]:
    """Channel distance and reference travel time from a crown to a landmark.

    Returns ``(distance_km, reference_hours, injection_reach_label)``. Any
    missing Phase 2 artefact raises ``FileNotFoundError``, which the caller
    turns into a blank cell rather than a crash.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from ..spatial.network import Network
    from ..spatial.projection import cell_level
    from ..spatial.terrain import Routing
    from ..spatial.travel_time import TravelTime

    rt, net, tt = Routing(), Network(), TravelTime()
    proj = cell_level(rt, net)

    crown = rt.index_of_lonlat(float(lon), float(lat))
    inj = int(proj["injection_reach"][crown])
    ids, t_ref, _ = tt.downstream_of(inj)

    lm = cfg.landmarks[landmark]
    pt = gpd.GeoSeries([Point(float(lm["lon"]), float(lm["lat"]))],
                       crs="EPSG:4326").to_crs(net.gdf.crs).iloc[0]
    target, _ = net.snap(pt)

    k = np.flatnonzero(ids == target)
    if not k.size:
        raise ValueError(f"{landmark} is not downstream of R-{inj:05d}")
    k = int(k[0])

    # Channel distance is the sum of reach lengths along the path, which is the
    # geometry; T_ref is the slope-weighted path integral, which is not.
    length = net.gdf.set_index("reach_id")["length_m"]
    chain = net.downstream_chain(inj)
    upto = chain[:chain.index(target) + 1]
    dist_km = float(length.loc[upto].sum()) / 1000.0
    hours = float(tt.hours(t_ref[k]))
    return dist_km, hours, f"R-{inj:05d}"


def build(cfg: Node, *, force: bool = False) -> Path:
    """Seed the linking table. Never overwrites a table someone has filled in."""
    paths.CATALOG.mkdir(parents=True, exist_ok=True)

    if PAIRS_PATH.is_file() and not force:
        existing = pd.read_csv(PAIRS_PATH, dtype=str, keep_default_na=False)
        if len(existing):
            log.info("%s already has %d rows — left untouched",
                     PAIRS_PATH.name, len(existing))
            return PAIRS_PATH

    from .build import EVENTS_PATH
    from . import schema

    events = schema.read(EVENTS_PATH) if EVENTS_PATH.is_file() else pd.DataFrame()
    land = events[events["hazard_type"] == "L"] if len(events) else pd.DataFrame()

    rows = []
    for spec in _STARTER:
        spec = dict(spec)
        landmark = spec.pop("downstream_landmark", None)
        date_ = spec.pop("date", "")
        place = spec.pop("place", "")

        # Match the landslide by date, and only when exactly one real record
        # claims that date. Two candidates means a human has to choose.
        event_id, lat, lon = "", np.nan, np.nan
        if date_ and len(land):
            hit = land[(land["date"] == date_) & (land["source"] != "SYNTHETIC")]
            if len(hit) == 1:
                event_id = str(hit.iloc[0]["event_id"])
                lat = pd.to_numeric(hit.iloc[0]["lat"], errors="coerce")
                lon = pd.to_numeric(hit.iloc[0]["lon"], errors="coerce")
            elif len(hit) > 1:
                log.warning("%s: %d landslide records share %s; leaving the id blank",
                            spec["pair_id"], len(hit), date_)

        dist, note_extra = "", ""
        if landmark and np.isfinite(lat) and np.isfinite(lon):
            try:
                d_km, hours, inj = _network_distance_km(cfg, lat, lon, landmark)
                dist = f"{d_km:.2f}"
                note_extra = (f" Injection reach {inj}; channel distance to "
                              f"{landmark} {d_km:.2f} km. Reference travel time at "
                              f"the un-fitted v0 is {hours:.2f} h — that is model "
                              f"geometry, NOT the observed lag, which is why "
                              f"lag_hours is blank.")
            except (FileNotFoundError, ValueError) as exc:
                log.warning("%s: no network distance (%s)", spec["pair_id"], exc)

        rows.append({
            "pair_id": spec["pair_id"],
            "landslide_event_id": event_id,
            "flood_event_id": "",       # never a synthetic id; see the docstring
            "lag_hours": "",            # evidence, not geometry
            "network_distance_km": dist,
            "evidence_level": spec["evidence_level"],
            "evidence_source": spec["evidence_source"],
            "notes": f"{place}. {spec['notes']}{note_extra}",
        })

    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df.to_csv(PAIRS_PATH, index=False, encoding="utf-8")

    n_linked = int((df["landslide_event_id"] != "").sum())
    log.info("wrote %s: %d pairs, %d linked to a catalog event, %d awaiting a "
             "sourced landslide", PAIRS_PATH.name, len(df), n_linked,
             len(df) - n_linked)
    log.info("evidence levels: %s",
             ", ".join(f"{k}={v}" for k, v in df["evidence_level"]
                       .value_counts().items()))
    return PAIRS_PATH
