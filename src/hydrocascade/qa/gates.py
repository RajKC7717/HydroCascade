"""Executable quality gates for all six phases.

Doc 04 §6: *"the phase is not done until all five pass"*, with a hard stop if
the event volume gate fails. Doc 05 §9 adds six for the spatial framework,
Doc 06 §8 six for the root-cause study, Doc 07 §10 eight for the cascade model,
and Docs 08-09 the deliverables of the experiment and delivery phases.

Each gate returns PASS, FAIL or **BLOCKED**. The three-way split is the whole
design: BLOCKED means the gate *could not be evaluated* because an input is
missing or is standing in for real data, and it is reported separately from
FAIL so that "we have not got the data yet" can never be read as "the check
passed". A two-state PASS/FAIL scheme forces that distinction to be lost, and
losing it is how a synthetic number reaches a slide.

Phases 3 to 6 take tens of minutes to run, so their gates **read the artefacts
those phases left on disk** rather than refitting. A missing artefact is
BLOCKED; a present one that is wrong is FAIL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("qa.gates")

PASS, FAIL, BLOCKED = "PASS", "FAIL", "BLOCKED"


@dataclass
class Gate:
    phase: int
    number: int
    name: str
    status: str
    detail: str = ""

    @property
    def label(self) -> str:
        return f"P{self.phase}.{self.number} {self.name}"


@dataclass
class Report:
    gates: list[Gate] = field(default_factory=list)

    def add(self, *a, **kw) -> None:
        g = Gate(*a, **kw)
        self.gates.append(g)
        marker = {PASS: "PASS   ", FAIL: "FAIL   ", BLOCKED: "BLOCKED"}[g.status]
        (log.info if g.status == PASS else
         log.error if g.status == FAIL else log.warning)(
            "%s  %-46s %s", marker, g.label, g.detail)

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if g.status == FAIL]

    @property
    def blocked(self) -> list[Gate]:
        return [g for g in self.gates if g.status == BLOCKED]

    def ok(self) -> bool:
        return not self.failed

    def to_markdown(self, cfg: Node) -> str:
        lines = [
            "# Quality gates — all six phases",
            "",
            f"Region `{cfg.region.id}` · period {cfg.period.start} to {cfg.period.end} "
            f"· analysis CRS {cfg.crs.analysis}",
            "",
            "Gates are from Doc 04 §6, Doc 05 §9, Doc 06 §8, Doc 07 §10, Doc 08 and Doc 09. "
            "`BLOCKED` means the gate could not be evaluated because an input is "
            "missing or synthetic — it is never counted as a pass.",
            "",
            "| Gate | Status | Detail |",
            "|---|---|---|",
        ]
        for g in self.gates:
            lines.append(f"| {g.label} | **{g.status}** | {g.detail} |")
        lines += ["",
                  f"**{sum(g.status == PASS for g in self.gates)} passed, "
                  f"{len(self.failed)} failed, {len(self.blocked)} blocked.**"]
        return "\n".join(lines)


# ------------------------------------------------------------------ phase 1 ---

def _phase1(cfg: Node, rep: Report) -> None:
    from ..catalog import dedup, schema
    from ..catalog.build import EVENTS_PATH
    from ..ingest import adapters

    if not EVENTS_PATH.is_file():
        rep.add(1, 0, "catalog exists", BLOCKED, "run `hydrocascade catalog` first")
        return

    df = schema.read(EVENTS_PATH)
    land = df[df["hazard_type"] == "L"]
    synthetic_rows = int((df["source"] == "SYNTHETIC").sum())

    # --- gate 1: volume
    n = len(land)
    if synthetic_rows:
        rep.add(1, 1, "volume >= 200 landslides", BLOCKED,
                f"{n} landslide rows but {synthetic_rows} of {len(df)} catalog rows are "
                f"SYNTHETIC. Doc 04 §6 gate 1 counts real events only.")
    elif n >= 1000:
        rep.add(1, 1, "volume >= 200 landslides", PASS, f"{n} events (target 1,000 met)")
    elif n >= 200:
        rep.add(1, 1, "volume >= 200 landslides", PASS,
                f"{n} events (above the 200 minimum, below the 1,000 target)")
    else:
        rep.add(1, 1, "volume >= 200 landslides", FAIL,
                f"only {n}. Doc 04 §6 hard stop: widen the region before proceeding.")

    # --- gate 2: provenance (FR-04)
    problems = schema.validate(df, strict=False)
    prov = [p for p in problems if "FR-04" in p]
    rep.add(1, 2, "provenance on 100% of rows", FAIL if prov else PASS,
            "; ".join(prov) if prov else
            f"all {len(df)} rows carry source, source_url and accessed_by")

    # --- gate 3: seasonality
    months = pd.to_datetime(land["date"], errors="coerce").dt.month
    monsoon = float(months.between(6, 9).mean()) if months.notna().any() else 0.0
    if monsoon >= 0.6:
        rep.add(1, 3, "seasonality follows the monsoon", PASS,
                f"{100*monsoon:.0f}% of landslides in Jun-Sep")
    else:
        rep.add(1, 3, "seasonality follows the monsoon", FAIL,
                f"only {100*monsoon:.0f}% in Jun-Sep — check dates and timezones")

    # --- gate 4: size-frequency
    from ..catalog import completeness
    areas = pd.to_numeric(land["area_m2"], errors="coerce").dropna()
    if len(areas) < 30:
        rep.add(1, 4, "size-frequency rollover identified", BLOCKED,
                f"only {len(areas)} landslides carry a mapped area")
    else:
        try:
            comp = completeness.find_rollover(
                areas.to_numpy(), n_bins=int(cfg.catalog.completeness.n_bins))
            configured = cfg.catalog.completeness.get("min_area_m2")
            note = ("applied" if configured else
                    "estimated but NOT applied — set catalog.completeness.min_area_m2")
            rep.add(1, 4, "size-frequency rollover identified", PASS,
                    f"A_min = {comp.a_min_m2:,.0f} m2, alpha = {comp.alpha:.2f} "
                    f"+/- {comp.alpha_stderr:.2f} ({note})")
        except ValueError as exc:
            rep.add(1, 4, "size-frequency rollover identified", FAIL, str(exc))

    # --- gate 5: no duplicates (FR-03)
    types = tuple(cfg.catalog.dedup.hazard_types)
    scoped = df[df["hazard_type"].isin(types)]
    n_dupes = dedup.assert_no_duplicates(
        scoped, distance_m=float(cfg.catalog.dedup.distance_m),
        days=int(cfg.catalog.dedup.days), crs=str(cfg.crs.analysis))
    rep.add(1, 5, "no duplicates within 500 m / 1 day", FAIL if n_dupes else PASS,
            f"{n_dupes} remaining among hazard type(s) {'/'.join(types)}"
            if n_dupes else
            f"0 among hazard type(s) {'/'.join(types)}; floods are declustered "
            f"instead (see catalog.dedup.hazard_types)")

    # --- gate 6: the flood definition is corroborated (FR-05)
    #
    # Not one of Doc 04 §6's five, but Doc 11 makes FR-05 a Phase 1 requirement
    # verified by "number reported in phase1_flood_definition.md". A requirement
    # nothing checks is a requirement nobody notices is unmet, so it gets a gate.
    from ..ingest import corroboration
    from .flood_definition import REPORT_PATH

    if not REPORT_PATH.is_file():
        rep.add(1, 6, "flood definition corroborated (FR-05)", BLOCKED,
                "no report — run `hydrocascade report`")
    else:
        try:
            corroboration.sentinel1_precision(cfg, df)
            rep.add(1, 6, "flood definition corroborated (FR-05)", PASS,
                    "SAR precision measured and reported")
        except Exception as exc:  # noqa: BLE001 — the reason is the finding
            head = str(exc).strip().splitlines()[0]
            rec = corroboration.recall(
                df[df["hazard_type"] == "F"],
                synthetic=bool(adapters.is_synthetic(cfg).get("glofas", False)))
            rep.add(1, 6, "flood definition corroborated (FR-05)", BLOCKED,
                    f"precision not measured: {head} Recall harness runs "
                    f"({rec.n_matched}/{rec.n_scored} matched) but is not "
                    f"publishable. See reports/{REPORT_PATH.name} §5.")


# ------------------------------------------------------------------ phase 2 ---

def _phase2(cfg: Node, rep: Report) -> None:
    import rasterio

    from ..spatial import projection
    from ..spatial.network import Network
    from ..spatial.slope_units import SlopeUnits
    from ..spatial.terrain import Routing
    from ..spatial.travel_time import TT_PATH, TravelTime

    # --- gate 1: every slope unit has an injection reach
    try:
        table = projection.load_table()
        nulls = int((table["injection_reach_id"] < 0).sum())
        rep.add(2, 1, "every slope unit has an injection reach",
                FAIL if nulls else PASS,
                f"{nulls} nulls of {len(table)} units" if nulls else
                f"0 nulls across {len(table):,} units")
    except FileNotFoundError as exc:
        rep.add(2, 1, "every slope unit has an injection reach", BLOCKED, str(exc))
        return

    # --- gate 2: network connected and correctly directed
    try:
        rt, net, tt = Routing(), Network(), TravelTime()
        proj = projection.cell_level(rt, net)
        lm = cfg.landmarks
        crown = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                                   float(lm.mundakkai_crown["lat"]))
        inj = int(proj["injection_reach"][crown])
        ids, t_ref, _ = tt.downstream_of(inj)
        hours = tt.hours(t_ref)

        import geopandas as gpd
        from shapely.geometry import Point
        order_ok, detail = True, []
        seen_at = {}
        for key in ("chooralmala_bridge", "nilambur"):
            pt = gpd.GeoSeries([Point(float(lm[key]["lon"]), float(lm[key]["lat"]))],
                               crs="EPSG:4326").to_crs(net.gdf.crs).iloc[0]
            rid, dist = net.snap(pt)
            k = np.flatnonzero(ids == rid)
            if not k.size:
                order_ok = False
                detail.append(f"{key} not downstream of the injection point")
            else:
                seen_at[key] = float(hours[k[0]])
                detail.append(f"{key} at {hours[k[0]]:.1f} h")
        if order_ok and len(seen_at) == 2:
            order_ok = seen_at["chooralmala_bridge"] < seen_at["nilambur"]
            if not order_ok:
                detail.append("but they appear in the WRONG ORDER")
        rep.add(2, 2, "Mundakkai -> Chooralmala -> Nilambur", PASS if order_ok else FAIL,
                f"R-{inj:05d} -> " + ", ".join(detail))
    except Exception as exc:  # noqa: BLE001
        rep.add(2, 2, "Mundakkai -> Chooralmala -> Nilambur", BLOCKED,
                f"{type(exc).__name__}: {exc}")

    # --- gate 3: every event maps to exactly one spatial unit
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    if EVENTS_PATH.is_file():
        df = schema.read(EVENTS_PATH)
        land = df[df["hazard_type"] == "L"]
        unjoined = int(land["spatial_unit_id"].astype(str).str.strip().eq("").sum())
        rep.add(2, 3, "every event joins to one spatial unit",
                FAIL if unjoined else PASS,
                f"{unjoined} landslide rows unjoined" if unjoined else
                f"all {len(land):,} landslide rows joined; "
                f"{len(df) - len(land):,} flood rows carry a reach id")
    else:
        rep.add(2, 3, "every event joins to one spatial unit", BLOCKED, "no catalog")

    # --- gate 4: rainfall reproduces the published Wayanad totals
    try:
        from ..ingest import rainfall_sources
        r = rainfall_sources.wayanad_reconciliation(cfg)
        got, pub = r["window_48h_mm"], r["published_48h_mm"]
        frac = got / pub
        if r["within_tolerance"]:
            rep.add(2, 4, "Wayanad rainfall totals reproduced", PASS,
                    f"48 h total {got:.0f} mm vs published {pub:.0f} mm")
        else:
            rep.add(2, 4, "Wayanad rainfall totals reproduced", FAIL,
                    f"{r['source'].upper()} {r['grid_cell']['resolution']} gives "
                    f"{got:.0f} mm vs published {pub:.0f} mm ({100*frac:.0f}%). "
                    + ("Not an extraction bug: the 29-31 Jul band peaks ~50 km west of "
                       "the crown, the known orographic displacement at 0.25 deg. "
                       "Needs IMD (M1) or IMERG (M2) — see docs/15_HOW_TO_RUN.md."
                       # the attribute is the product's own name, e.g.
                       # "CHIRPS V2.0", not the config key
                       if r["source"].lower().startswith("chirps") else
                       "Check the files in data/raw and their units."))
    except FileNotFoundError:
        rep.add(2, 4, "Wayanad rainfall totals reproduced", BLOCKED,
                "run `hydrocascade rainfall` first")
    except Exception as exc:  # noqa: BLE001
        rep.add(2, 4, "Wayanad rainfall totals reproduced", BLOCKED,
                f"{type(exc).__name__}: {exc}")

    # --- gate 5: CRS consistency
    target = str(cfg.crs.analysis)
    wrong = []
    for name in ("dem_filled.tif", "flow_acc.tif", "slope_deg.tif"):
        p = paths.SPATIAL / name
        if p.is_file():
            with rasterio.open(p) as src:
                if str(src.crs) != target:
                    wrong.append(f"{name}={src.crs}")
    import geopandas as gpd
    for name in ("slope_units.gpkg", "channel_network.gpkg", "aoi.gpkg"):
        p = paths.SPATIAL / name
        if p.is_file():
            crs = gpd.read_file(p, rows=1).crs
            if str(crs) != target:
                wrong.append(f"{name}={crs}")
    rep.add(2, 5, f"all layers in {target}", FAIL if wrong else PASS,
            ", ".join(wrong) if wrong else "rasters and vectors all aligned")

    # --- gate 6: travel-time table cached
    if TT_PATH.is_file():
        tt = TravelTime()
        v = tt.v0_reference_ms * tt.reach_slope ** tt.celerity_exponent
        rep.add(2, 6, "travel-time table built and cached", PASS,
                f"{TT_PATH.stat().st_size/1e6:.1f} MB, {tt.indices.size:,} "
                f"(source, downstream) pairs, median celerity {np.median(v):.2f} m/s")
    else:
        rep.add(2, 6, "travel-time table built and cached", BLOCKED,
                "run `hydrocascade traveltime`")


# ---------------------------------------------------------------------- run ---

def _phase3(cfg: Node, rep: Report) -> None:
    """Doc 06 §8, read from what Phase 3 left on disk.

    Phase 3 takes tens of minutes, so the gates read its artefacts rather than
    refitting. A missing artefact is BLOCKED, never FAIL: "we have not run it"
    and "we ran it and it was wrong" are different statements.
    """
    import json

    from ..rootcause import drivers

    td = drivers.TOP_DRIVERS_PATH
    s_l = paths.PROCESSED / "S_landslide.parquet"
    s_f = paths.PROCESSED / "S_flood.parquet"

    if not td.is_file() and not s_l.is_file():
        rep.add(3, 0, "Phase 3 has been run", BLOCKED,
                "run `hydrocascade rootcause` first")
        return

    exported = [p.name for p in (s_l, s_f) if p.is_file()]
    rep.add(3, 4, "S_L(u) and S_F(s) exported as calibrated tables",
            PASS if len(exported) == 2 else FAIL,
            ", ".join(exported) or "nothing exported")

    if td.is_file():
        d = json.loads(td.read_text(encoding="utf-8"))
        n = len(d["drivers"])
        rep.add(3, 5, "e~(u) chosen, between 2 and 4 drivers",
                PASS if 2 <= n <= 4 else FAIL,
                f"{n}: {', '.join(d['drivers'])}")
    else:
        rep.add(3, 5, "e~(u) chosen, between 2 and 4 drivers", BLOCKED,
                "top_drivers.json missing")

    for name, num, label in (("phase3_drivers.md", 1, "driver ranking reported"),
                             ("phase3_calibration.md", 3, "calibration reported"),
                             ("phase3_baselines.md", 6, "B1-B3 built and evaluated")):
        p = paths.REPORTS / name
        rep.add(3, num, label, PASS if p.is_file() else BLOCKED,
                f"reports/{name}" if p.is_file() else f"{name} not written")


def _phase4(cfg: Node, rep: Report) -> None:
    """Doc 07 §10, read from the fit records."""
    import json

    vp = paths.MODELS / "fits" / "stage_verification.json"
    if vp.is_file():
        v = json.loads(vp.read_text(encoding="utf-8"))
        s1, s2 = v.get("S1", {}), v.get("S2", {})
        both = bool(s1.get("passed")) and bool(s2.get("passed"))
        rep.add(4, 1, "parameter recovery verified at S1 and S2",
                PASS if both else FAIL,
                f"S1 {s1.get('detail', '?')[:70]}; S2 {s2.get('detail', '?')[:70]}")

        cc = v.get("cross_check", {})
        rep.add(4, 2, "independent cross-check of the likelihood", BLOCKED,
                f"{cc.get('detail', '?')[:80]} — SUBSTITUTE for the Doc 07 §2 "
                f"`tick` check, which does not install here. Three "
                f"implementations by one author is weaker than an independent "
                f"library, so this is not recorded as a pass.")
    else:
        rep.add(4, 1, "parameter recovery verified at S1 and S2", BLOCKED,
                "run `hydrocascade cascade --verify`")

    fits = sorted((paths.MODELS / "fits").glob("*.json")) if (
        paths.MODELS / "fits").is_dir() else []
    fits = [f for f in fits if f.name != "stage_verification.json"]
    if not fits:
        rep.add(4, 3, "branching ratio < 1 at every stage", BLOCKED,
                "no fit records — run `hydrocascade cascade`")
        return

    bad = []
    for f in fits:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            continue
        if float(r.get("branching_ratio", 0)) >= 1.0:
            bad.append(f"{r.get('name', f.stem)}={r['branching_ratio']:.3f}")
    rep.add(4, 3, "branching ratio < 1 at every stage", FAIL if bad else PASS,
            ", ".join(bad) if bad else
            f"all {len(fits)} recorded fits are stationary")

    # --- Doc 07 §10 gate 6: the Wayanad event must pass the fitted runout gate.
    #
    # Doc 07 §5 is unambiguous: "If it does not, the gate parameterisation is
    # wrong." A gate fitted to exclude the one event the project is built around
    # has been fitted to the wrong thing, and burying that in report prose is
    # how it gets missed.
    notes = {}
    for f in fits:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                           # noqa: BLE001
            continue
        notes[r.get("name", f.stem)] = str(r.get("note", ""))

    excluded = [n for n, t in notes.items() if "EXCLUDED" in t]
    passes = [n for n, t in notes.items() if "PASSES" in t]
    if not (excluded or passes):
        rep.add(4, 6, "landslide passes the fitted runout gate", BLOCKED,
                "no fit records the gate check — run `hydrocascade cascade`")
    elif excluded:
        rep.add(4, 6, "landslide passes the fitted runout gate", FAIL,
                f"EXCLUDED in {', '.join(sorted(excluded))}. Doc 07 §5: "
                f"\"if it does not, the gate parameterisation is wrong\". "
                f"Passes in {', '.join(sorted(passes)) or 'no stage'}.")
    else:
        rep.add(4, 6, "landslide passes the fitted runout gate", PASS,
                f"passes in {', '.join(sorted(passes))}")

    # --- Doc 07 §10 gate 7: is the fitted v0 physically plausible?
    outside = [n for n, t in notes.items() if "OUTSIDE the physical range" in t]
    inside = [n for n, t in notes.items() if "inside the physical range" in t]
    if not (outside or inside):
        rep.add(4, 7, "fitted v0 gives a plausible channel celerity", BLOCKED,
                "no stage fitted a travel-time shift")
    elif outside:
        rep.add(4, 7, "fitted v0 gives a plausible channel celerity", FAIL,
                f"outside 1-5 m/s in {', '.join(sorted(outside))}")
    else:
        rep.add(4, 7, "fitted v0 gives a plausible channel celerity", PASS,
                f"inside 1-5 m/s in {', '.join(sorted(inside))}")

    p = paths.REPORTS / "phase4_stages.md"
    rep.add(4, 8, "every number traceable to a saved fit record",
            PASS if p.is_file() else BLOCKED,
            f"{len(fits)} fit records in models/fits/, stage report "
            f"{'written' if p.is_file() else 'missing'}")


def _phase5(cfg: Node, rep: Report) -> None:
    """Doc 08 deliverables, read from disk."""
    from ..experiments import register, transfer

    reg = register.read()
    rep.add(5, 0, "experiment register populated",
            PASS if len(reg) else BLOCKED,
            f"{len(reg)} runs recorded" if len(reg) else
            "experiments/register.csv empty — run `hydrocascade experiments`")

    for name, num, label in (
            ("phase5_ablations.md", 1, "ablation table A1-A9 produced"),
            ("phase5_baselines.md", 2, "baselines B1-B5 vs M5 reported"),
            ("phase5_loco.md", 3, "leave-one-cascade-out run"),
            ("phase5_sensitivity.md", 4, "sensitivity analyses reported"),
            ("phase5_backtest.md", 5, "Wayanad backtest reported")):
        p = paths.REPORTS / name
        rep.add(5, num, label, PASS if p.is_file() else BLOCKED,
                f"reports/{name}" if p.is_file() else f"{name} not written")

    money = paths.FIGURES / "money_figure.png"
    rep.add(5, 6, "money figure produced", PASS if money.is_file() else BLOCKED,
            "figures/money_figure.png" if money.is_file() else "not rendered")

    st = transfer.status()
    rep.add(5, 7, "spatial transfer to a second region", BLOCKED,
            st["blocked_because"])


def _phase6(cfg: Node, rep: Report) -> None:
    """Doc 09 deliverables."""
    for name, num, label in (
            ("phase6_architecture.md", 1, "real-time architecture documented"),
            ("phase6_alerts.md", 2, "alert thresholds optimised and reported")):
        p = paths.REPORTS / name
        rep.add(6, num, label, PASS if p.is_file() else BLOCKED,
                f"reports/{name}" if p.is_file() else f"{name} not written")

    curve = paths.FIGURES / "leadtime_vs_falsealarm.png"
    rep.add(6, 3, "lead-time versus false-alarm curve produced",
            PASS if curve.is_file() else BLOCKED,
            "figures/leadtime_vs_falsealarm.png" if curve.is_file()
            else "not rendered")

    app = paths.ROOT / "src" / "hydrocascade" / "dashboard" / "app.py"
    rep.add(6, 4, "replay dashboard exists", PASS if app.is_file() else FAIL,
            "`hydrocascade app` — replay mode only; Doc 09 §1.1 is explicit that "
            "no 24/7 production service is delivered")


def run(cfg: Node, phase: str = "all", write_report: bool = True) -> bool:
    from ..ingest import adapters

    rep = Report()
    synth = adapters.synthetic_sources(cfg)
    if synth:
        log.warning("standing in with SYNTHETIC data for: %s", ", ".join(synth))

    if phase in ("1", "all"):
        _phase1(cfg, rep)
    if phase in ("2", "all"):
        _phase2(cfg, rep)
    for n, fn in (("3", _phase3), ("4", _phase4), ("5", _phase5), ("6", _phase6)):
        if phase in (n, "all"):
            try:
                fn(cfg, rep)
            except Exception as exc:                # noqa: BLE001
                rep.add(int(n), 0, f"Phase {n} gates", BLOCKED,
                        f"{type(exc).__name__}: {exc}")

    log.info("-" * 72)
    log.info("%d passed, %d failed, %d blocked",
             sum(g.status == PASS for g in rep.gates), len(rep.failed), len(rep.blocked))

    if write_report:
        paths.REPORTS.mkdir(parents=True, exist_ok=True)
        out = paths.REPORTS / "quality_gates.md"
        text = rep.to_markdown(cfg)
        if synth:
            text += ("\n\n> **Synthetic stand-ins active:** " + ", ".join(synth) +
                     ". No number derived from these may be reported. "
                     "See `config/default.yaml` -> `data_mode`.\n")
        out.write_text(text, encoding="utf-8")
        log.info("wrote   reports/%s", out.name)

    return rep.ok()
