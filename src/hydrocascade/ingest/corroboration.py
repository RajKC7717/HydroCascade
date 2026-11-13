"""The corroboration layer — Doc 02 §3.2-§3.3, Doc 04 §4.2.

The flood definition in :mod:`hydrocascade.ingest.floods` is **model-derived**:
a reach floods when reanalysis discharge crosses a fitted Gumbel quantile. That
is a definitional choice, and Doc 02 §3.3 is explicit that a definitional choice
has to be converted into a *measured* number before it goes in a paper:

* **precision** — of the events the rule flags, what fraction show a real
  inundation signal in Sentinel-1 SAR?
* **recall** — of the floods that are independently documented, what fraction
  did the rule flag?

Neither number may be invented. This module therefore does three things and
refuses to do a fourth:

1. :func:`sensitivity` runs the whole detection procedure under the Doc 02 §3.5
   variants (Q5 instead of Q2, 5-day declustering, 250 km² eligibility) and
   returns a comparison table. This needs no external data and always runs.
2. :func:`blind_spot` checks the Doc 02 §3.4 claim — that the Punnapuzha at
   Chooralmala is below the eligibility threshold — against the network we
   actually built, and reports where downstream the rule regains sight.
3. :func:`recall` scores detections against ``data/reference/documented_floods.csv``,
   a list the team fills in from KSDMA / Copernicus EMS / UNOSAT / CWC. Rows
   without a source URL are not scored, on the same FR-04 rule the catalog uses.
4. :func:`sentinel1_precision` does **not** silently return a number. Sentinel-1
   Otsu thresholding runs in Google Earth Engine, which needs an account, so the
   function raises with the exact recipe rather than fabricating a result.

Nothing here is ever used to *fit*. It is used only to score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from .adapters import CredentialsMissing

log = get("ingest.corroboration")

REFERENCE_PATH = paths.REFERENCE / "documented_floods.csv"

#: Schema of ``data/reference/documented_floods.csv``. ``date_start``/``date_end``
#: are a closed interval so a source that says only "August 2019" can be recorded
#: honestly as 2019-08-01..2019-08-31 instead of being sharpened into a false
#: precision.
REFERENCE_COLUMNS = (
    "flood_id", "date_start", "date_end", "place", "district",
    "lat", "lon", "in_pilot_basin", "evidence_type",
    "source", "source_url", "source_date_accessed", "accessed_by", "notes",
)


# ------------------------------------------------------------- sensitivity ---

@dataclass
class Variant:
    """One row of the Doc 02 §3.5 sensitivity table."""
    label: str
    threshold_rp: int
    decluster_days: int
    min_area_km2: float
    n_eligible_reaches: int = 0
    n_events: int = 0
    n_reaches_with_events: int = 0
    rate_per_reach_year: float = float("nan")
    n_minor: int = 0
    n_major: int = 0
    n_extreme: int = 0
    median_days_above: float = float("nan")


def _variant_config(cfg: Node, threshold_rp: int, decluster_days: int,
                    min_area_km2: float) -> Node:
    """A shallow copy of ``cfg`` with only the three flood knobs changed.

    Doc 11 §5 rule 3 says a variant is a config edit, not a code edit — so the
    sensitivity analysis edits a config rather than reaching into the detector.
    """
    raw = cfg.to_dict()
    raw["floods"] = dict(raw["floods"])
    raw["floods"]["threshold_return_period"] = int(threshold_rp)
    raw["floods"]["decluster_days"] = int(decluster_days)
    raw["floods"]["min_contributing_area_km2"] = float(min_area_km2)
    return Node(raw)


def sensitivity(cfg: Node, discharge: pd.DataFrame, areas: pd.Series,
                variants: list[tuple[str, int, int, float]] | None = None,
                ) -> pd.DataFrame:
    """Re-run detection under each Doc 02 §3.5 variant.

    ``discharge`` is the (date × reach) table and ``areas`` maps reach id to
    contributing area, exactly as :func:`floods.detect_for_all_reaches` wants
    them. The discharge table is built once and reused, so the cost here is the
    detection loop only.
    """
    from . import floods

    base = (int(cfg.floods.threshold_return_period),
            int(cfg.floods.decluster_days),
            float(cfg.floods.min_contributing_area_km2))
    if variants is None:
        variants = [
            (f"headline (Q{base[0]}, {base[1]} d, {base[2]:.0f} km²)", *base),
            ("threshold Q5 instead of Q2", 5, base[1], base[2]),
            ("declustering 5 d instead of 3 d", base[0], 5, base[2]),
            ("eligibility 250 km² instead of 50 km²", base[0], base[1], 250.0),
        ]

    start, end = str(cfg.period.start), str(cfg.period.end)
    rows = []
    for label, rp, dc, area in variants:
        v = Variant(label, rp, dc, area)
        vcfg = _variant_config(cfg, rp, dc, area)
        v.n_eligible_reaches = int((areas > area).sum())
        try:
            ev = floods.detect_for_all_reaches(discharge, areas, vcfg)
        except RuntimeError as exc:          # no reach clears the eligibility bar
            log.warning("variant %r produced nothing: %s", label, exc)
            rows.append(v)
            continue

        # Score only the study period; the long record exists to fit thresholds.
        ev = ev[(ev["date"] >= start) & (ev["date"] <= end)]
        v.n_events = len(ev)
        if v.n_events:
            v.n_reaches_with_events = int(ev["reach_id"].nunique())
            years = pd.to_datetime(ev["date"]).dt.year
            span = max(int(years.max()) - int(years.min()) + 1, 1)
            v.rate_per_reach_year = v.n_events / v.n_reaches_with_events / span
            counts = ev["severity_class"].value_counts()
            v.n_minor = int(counts.get("minor", 0))
            v.n_major = int(counts.get("major", 0))
            v.n_extreme = int(counts.get("extreme", 0))
            v.median_days_above = float(ev["n_days_above"].median())
        rows.append(v)
        log.info("variant %-42s %6d events on %5d reaches",
                 label, v.n_events, v.n_reaches_with_events)

    return pd.DataFrame([r.__dict__ for r in rows])


# --------------------------------------------------------------- blind spot ---

def blind_spot(cfg: Node) -> dict:
    """Verify the Doc 02 §3.4 admission against the network we built.

    Doc 02 §3.4 states that the Punnapuzha at Chooralmala sits *"far below"* the
    GloFAS eligibility threshold, so the rule cannot see the flood at the
    debris-flow site. That is an assertion about a real catchment, and we now
    have the catchment, so it can be checked rather than repeated.

    Returns the injection reach, its catchment area, and the first reach
    downstream that clears each eligibility threshold, with the travel time to
    get there.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from ..spatial.network import Network
    from ..spatial.projection import cell_level
    from ..spatial.terrain import Routing
    from ..spatial.travel_time import TravelTime

    rt, net, tt = Routing(), Network(), TravelTime()
    proj = cell_level(rt, net)
    lm = cfg.landmarks

    crown = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                               float(lm.mundakkai_crown["lat"]))
    inj = int(proj["injection_reach"][crown])
    ids, t_ref, _ = tt.downstream_of(inj)
    hours = tt.hours(t_ref)
    area = tt.reach_area_km2[ids]

    bridge = gpd.GeoSeries(
        [Point(float(lm.chooralmala_bridge["lon"]), float(lm.chooralmala_bridge["lat"]))],
        crs="EPSG:4326").to_crs(net.gdf.crs).iloc[0]
    bridge_reach, bridge_dist = net.snap(bridge)
    bridge_area = float(net.gdf.loc[net.gdf["reach_id"] == bridge_reach,
                                    "area_dn_km2"].iloc[0])

    out = {
        "injection_reach": inj,
        "injection_area_km2": float(tt.reach_area_km2[inj]),
        "chooralmala_reach": int(bridge_reach),
        "chooralmala_area_km2": bridge_area,
        "chooralmala_snap_dist_m": float(bridge_dist),
        "n_downstream": int(ids.size),
        "thresholds": {},
    }
    for thr in (float(cfg.floods.min_contributing_area_km2), 250.0):
        ok = np.flatnonzero(area >= thr)
        if ok.size:
            k = int(ok[0])                      # the chain is ordered downstream
            out["thresholds"][thr] = {
                "visible": True,
                "first_reach": int(ids[k]),
                "first_area_km2": float(area[k]),
                "hours_from_injection": float(hours[k]),
                "reaches_blind": k,
            }
        else:
            out["thresholds"][thr] = {"visible": False, "reaches_blind": int(ids.size)}

    log.info("blind spot: injection R-%05d has %.1f km2; Chooralmala R-%05d has %.1f km2",
             inj, out["injection_area_km2"], bridge_reach, bridge_area)
    for thr, d in out["thresholds"].items():
        if d["visible"]:
            log.info("  at %.0f km2 the rule regains sight %d reaches / %.1f h downstream "
                     "(R-%05d, %.0f km2)", thr, d["reaches_blind"],
                     d["hours_from_injection"], d["first_reach"], d["first_area_km2"])
        else:
            log.info("  at %.0f km2 the whole %d-reach chain is invisible",
                     thr, d["reaches_blind"])
    return out


# ------------------------------------------------------- the reference list ---

#: The Doc 04 §5 starter list, verbatim: the date each row carries is the date
#: that table gives and nothing sharper. "Aug 2019" becomes a whole month rather
#: than a guessed day, and no coordinate is invented — the only two that appear
#: are read from ``config/default.yaml -> landmarks`` at write time, where they
#: already carry their own provenance note.
#:
#: These rows are a **starting point, not the finished reference list.** Their
#: source is the project brief, which is a secondary source; a recall number is
#: only publishable once each row is re-sourced to a primary record (KSDMA
#: annual report, Copernicus EMS or UNOSAT activation, CWC gauge series, or a
#: dated news archive entry). That is why ``evidence_type`` says ``project_brief``.
_STARTER_ROWS = [
    dict(flood_id="DF-001", date_start="2018-08-16", date_end="2018-08-16",
         place="Kerala monsoon floods", district="statewide",
         in_pilot_basin="unknown",
         notes="Doc 04 §5 starter list: 'Kerala monsoon landslides + floods, "
               "16 Aug 2018'. Bulk training event; concurrent flood and landslide."),
    dict(flood_id="DF-002", date_start="2019-08-01", date_end="2019-08-31",
         place="Kavalappara", district="Malappuram", in_pilot_basin="unknown",
         notes="Doc 04 §5 gives the month only. Narrow to a day from the primary "
               "source before scoring."),
    dict(flood_id="DF-003", date_start="2019-08-01", date_end="2019-08-31",
         place="Puthumala", district="Wayanad", in_pilot_basin="unknown",
         notes="Doc 04 §5 provenance warning: sources disagree between 8 and 29 "
               "August 2019. Resolve from GSI or the primary field report and "
               "record which source was trusted."),
    dict(flood_id="DF-004", date_start="2020-08-06", date_end="2020-08-06",
         place="Pettimudi", district="Idukki", in_pilot_basin="unknown",
         notes="Idukki drains east and south of the Chaliyar; confirm with a "
               "basin check before scoring, then set in_pilot_basin."),
    dict(flood_id="DF-005", date_start="2021-10-01", date_end="2021-10-31",
         place="Kokkayar / Plappally / Kavali", district="",
         in_pilot_basin="unknown",
         notes="Doc 04 §5 gives 'Oct 2021' and no district. Three different "
               "causes; treat as three rows once sourced."),
    dict(flood_id="DF-006", date_start="2024-07-30", date_end="2024-07-30",
         place="Chooralmala, Punnapuzha", district="Wayanad",
         in_pilot_basin="yes", landmark="chooralmala_bridge",
         notes="The held-out test event. Coordinates are the config landmark "
               "'Bridge destroyed 30 Jul 2024'. Below the GloFAS eligibility "
               "threshold — see Doc 02 §3.4 and the blind-spot section."),
]


def ensure_reference_list(cfg: Node) -> pd.DataFrame:
    """Create ``documented_floods.csv`` from the Doc 04 §5 starter list if absent.

    Written once and then owned by the team, exactly like
    ``cascade_pairs.csv``: the pipeline never overwrites a file a human has
    edited. ``accessed_by`` is stamped with whoever ran it, so the row is
    honestly attributed rather than carrying an invented name.
    """
    if REFERENCE_PATH.is_file():
        return load_reference()

    from datetime import date as _date

    from ..provenance import default_accessor

    lm = cfg.landmarks
    rows = []
    for spec in _STARTER_ROWS:
        spec = dict(spec)
        mark = spec.pop("landmark", None)
        lat = float(lm[mark]["lat"]) if mark else ""
        lon = float(lm[mark]["lon"]) if mark else ""
        rows.append({
            **spec, "lat": lat, "lon": lon,
            "evidence_type": "project_brief",
            "source": "DOC04_S5",
            "source_url": "docs/04_PHASE_1_DATA_COLLECTION.md",
            "source_date_accessed": _date.today().isoformat(),
            "accessed_by": default_accessor(),
        })
    df = pd.DataFrame(rows)[list(REFERENCE_COLUMNS)]
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_PATH, index=False, encoding="utf-8")
    log.info("seeded %s with the %d Doc 04 §5 starter rows — re-source each one "
             "to a primary record before publishing a recall number",
             REFERENCE_PATH.name, len(df))
    return load_reference()


def load_reference() -> pd.DataFrame:
    """Independently documented floods. Empty (but typed) if the file is absent."""
    if not REFERENCE_PATH.is_file():
        log.warning("no reference flood list at %s — recall cannot be measured",
                    REFERENCE_PATH)
        return pd.DataFrame(columns=list(REFERENCE_COLUMNS))
    df = pd.read_csv(REFERENCE_PATH, dtype=str, keep_default_na=False)
    missing = [c for c in REFERENCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{REFERENCE_PATH.name} is missing columns: {missing}")
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@dataclass
class RecallResult:
    status: str                       # measured | exercise | blocked
    reason: str = ""
    window_days: int = 3
    n_reference: int = 0
    n_sourced: int = 0
    n_scored: int = 0
    n_matched: int = 0
    n_provisional: int = 0            # scored rows still sourced to the brief
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def value(self) -> float:
        return self.n_matched / self.n_scored if self.n_scored else float("nan")

    @property
    def publishable(self) -> bool:
        """A recall number goes in the paper only if nothing about it is a stand-in."""
        return (self.status == "measured" and self.n_scored > 0
                and self.n_provisional == 0)


def recall(detected: pd.DataFrame, reference: pd.DataFrame | None = None, *,
           window_days: int = 3, synthetic: bool = False) -> RecallResult:
    """What fraction of independently documented floods did the rule flag?

    ``detected`` needs a ``date`` column (the catalog's flood rows will do) and,
    optionally, ``spatial_unit_id`` holding the reach id.

    A reference flood counts as detected if **any** reach in the basin recorded
    an event within ``window_days`` of its documented interval. That is
    deliberately a *basin-scale* recall: most published accounts name a town, not
    a reach, and inventing a reach id for them would be worse than measuring a
    coarser quantity honestly. Rows flagged ``in_pilot_basin = no`` are reported
    but not scored — the pilot basin cannot be blamed for missing a flood in
    Idukki.
    """
    ref = load_reference() if reference is None else reference.copy()
    n_ref = len(ref)
    if n_ref == 0:
        return RecallResult("blocked", n_reference=0, window_days=window_days,
                            reason=f"{REFERENCE_PATH.relative_to(paths.ROOT)} has no "
                                   f"rows. Fill it from KSDMA annual reports, "
                                   f"Copernicus EMS / UNOSAT activations and CWC "
                                   f"gauge records; every row needs a source_url.")

    # FR-04 applies here too: a row nobody can trace is not evidence.
    sourced = ref[ref["source_url"].astype(str).str.strip().ne("")
                  & ~ref["source_url"].astype(str).str.upper().str.contains("NEEDS_SOURCING")]
    if sourced.empty:
        return RecallResult("blocked", n_reference=n_ref, n_sourced=0,
                            window_days=window_days,
                            reason="no reference row carries a source_url (FR-04)")

    dates = pd.to_datetime(detected["date"], errors="coerce").dropna()
    dates = pd.DatetimeIndex(sorted(dates.unique()))

    rows = []
    for _, r in sourced.iterrows():
        d0 = pd.to_datetime(r["date_start"], errors="coerce")
        d1 = pd.to_datetime(r["date_end"] or r["date_start"], errors="coerce")
        if pd.isna(d0):
            rows.append({**r.to_dict(), "scored": False, "matched": False,
                         "nearest_detection": "", "lag_days": np.nan,
                         "skip_reason": "unparsable date_start"})
            continue
        lo = d0 - pd.Timedelta(days=window_days)
        hi = (d1 if pd.notna(d1) else d0) + pd.Timedelta(days=window_days)
        hit = dates[(dates >= lo) & (dates <= hi)]
        scored = str(r["in_pilot_basin"]).strip().lower() != "no"
        if len(hit):
            k = int(np.argmin(np.abs((hit - d0).days)))
            nearest, lag = hit[k].date().isoformat(), int((hit[k] - d0).days)
        else:
            nearest, lag = "", np.nan
        rows.append({**r.to_dict(), "scored": scored, "matched": bool(len(hit)),
                     "nearest_detection": nearest, "lag_days": lag,
                     "skip_reason": "" if scored else "outside the pilot basin"})

    detail = pd.DataFrame(rows)
    scored = detail[detail["scored"]]
    res = RecallResult(
        status="exercise" if synthetic else "measured",
        n_reference=n_ref, n_sourced=len(sourced), n_scored=len(scored),
        n_matched=int(scored["matched"].sum()), window_days=window_days,
        n_provisional=int((scored["evidence_type"] == "project_brief").sum()),
        detail=detail)
    reasons = []
    if synthetic:
        reasons.append("the discharge is SYNTHETIC, so this measures the harness "
                       "rather than GloFAS")
    if res.n_provisional:
        reasons.append(f"{res.n_provisional} of {res.n_scored} scored rows are still "
                       f"sourced to the project brief rather than to a primary "
                       f"KSDMA / EMS / CWC record")
    res.reason = ("not publishable: " + "; and ".join(reasons) + ".") if reasons else ""
    log.info("recall: %d of %d scorable documented floods matched within +/-%d days",
             res.n_matched, res.n_scored, window_days)
    return res


# ------------------------------------------------------------ the SAR check ---

GEE_RECIPE = """\
Sentinel-1 GRD, IW mode, VV polarisation, from the COPERNICUS/S1_GRD collection.
For each detected flood event:
  1. flood image  = median composite of scenes within +/-3 days of the event date
  2. reference    = median composite over the driest month of the same relative
                    orbit (January-February for Kerala)
  3. threshold    = Otsu on the histogram of the flood-image VV backscatter,
                    computed inside the reach buffer, not globally
  4. water mask   = VV < threshold, with slope > 5 deg masked out (radar shadow)
  5. signal       = flooded area(flood) - flooded area(reference), inside a
                    1 km buffer of the reach
An event 'shows a real inundation signal' when that difference is positive and
exceeds the reference-date noise level. Reported accuracy of this method for
Kerala is around 94% overall (Doc 02 §3.2).
"""


def sentinel1_precision(cfg: Node, events: pd.DataFrame):
    """Precision of the flood rule against Sentinel-1 inundation.

    Deliberately raises instead of returning a number. Otsu thresholding of VV
    backscatter runs in Google Earth Engine, which needs an approved account,
    and the one thing this project must never do is report a corroboration
    statistic it did not compute.
    """
    mode = str(cfg.data_mode.get("sentinel1", "unavailable"))
    if mode != "real":
        raise CredentialsMissing(
            "The Sentinel-1 precision test (Doc 02 §3.3 step 4) needs Google "
            "Earth Engine.\n"
            "  1. register at https://code.earthengine.google.com\n"
            "  2. pip install earthengine-api && earthengine authenticate\n"
            "  3. set data_mode.sentinel1: real in config/default.yaml\n\n"
            "The recipe is fixed, so wiring it up is mechanical:\n\n" + GEE_RECIPE)
    raise NotImplementedError(
        "Earth Engine credentials are configured but the export is not wired up "
        "yet. Implement the recipe below against the events table:\n\n" + GEE_RECIPE)
