"""HydroCascade Explorer — the whole project on one page.

Run with ``hydrocascade app`` (or ``streamlit run src/hydrocascade/dashboard/app.py``).

This is not the Doc 09 §3 dashboard. That one shows alert levels, a replay
slider and the attribution breakdown, none of which exist before the cascade
model is fitted in Phase 4. This one answers the questions a reviewer asks at
Review 1: what is in the catalog, what does the spatial framework look like,
does the Wayanad cascade path actually work, and which gates pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow `streamlit run path/to/app.py` without the package being installed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hydrocascade import paths                      # noqa: E402
from hydrocascade.config import load_config         # noqa: E402

st.set_page_config(page_title="HydroCascade Explorer", page_icon="🌧", layout="wide")

ACCENT = "#d1341f"
WATER = "#3b7ea1"


# ------------------------------------------------------------------ loaders ---

@st.cache_resource(show_spinner=False)
def get_cfg():
    return load_config()


@st.cache_data(show_spinner="loading catalog…")
def load_events():
    from hydrocascade.catalog import schema
    from hydrocascade.catalog.build import EVENTS_PATH
    if not EVENTS_PATH.is_file():
        return None
    df = schema.read(EVENTS_PATH)
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner="loading spatial framework…")
def load_spatial():
    """Attribute tables only — **no geometry**.

    This page never draws these layers: the maps it shows are the pre-rendered
    PNGs in `figures/`, and every use of this data is a count, a median or a
    sum over an attribute column. Reading the geometry as well meant building
    17,138 slope-unit polygons and 9,038 reach linestrings, then letting
    `cache_data` pickle them and hand back a copy — several hundred megabytes
    for numbers that never needed a shape. On a 7 GB machine that was the
    difference between the app staying up and being killed.

    `ignore_geometry=True` returns a plain pandas DataFrame, which is all the
    code below ever used.
    """
    import geopandas as gpd
    out = {}
    for key, name in (("units", "slope_units.gpkg"), ("network", "channel_network.gpkg"),
                      ("aoi", "aoi.gpkg")):
        p = paths.SPATIAL / name
        out[key] = gpd.read_file(p, ignore_geometry=True) if p.is_file() else None
    return out


@st.cache_data(show_spinner="tracing the cascade path…")
def load_cascade_chain():
    """The Mundakkai downstream chain with travel times and dilution."""
    from hydrocascade.spatial.network import Network
    from hydrocascade.spatial.projection import cell_level
    from hydrocascade.spatial.terrain import Routing
    from hydrocascade.spatial.travel_time import TravelTime

    cfg = get_cfg()
    rt, net, tt = Routing(), Network(), TravelTime()
    proj = cell_level(rt, net)
    lm = cfg.landmarks
    crown = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                               float(lm.mundakkai_crown["lat"]))
    inj = int(proj["injection_reach"][crown])
    ids, t_ref, ratio = tt.downstream_of(inj)
    geom = net.gdf.set_index("reach_id")
    chain = pd.DataFrame({
        "reach_id": ids,
        "hours": tt.hours(t_ref),
        "T_ref_km": t_ref / 1000.0,
        "dilution": ratio,
        "catchment_km2": tt.reach_area_km2[ids],
        "strahler": geom.loc[ids, "strahler"].to_numpy(),
        "slope": geom.loc[ids, "slope"].to_numpy(),
    })
    meta = {"injection_reach": inj,
            "d_slope_km": float(proj["d_slope_m"][crown]) / 1000.0,
            "n_downstream": len(ids),
            "v0_reference": tt.v0_reference_ms}
    return chain, meta


@st.cache_data(show_spinner=False)
def load_rainfall():
    import xarray as xr
    p = paths.SPATIAL / "rainfall_pixels.nc"
    if not p.is_file():
        return None
    return xr.open_dataset(p).load()


def figure(name: str):
    p = paths.FIGURES / name
    return p if p.is_file() else None


# ------------------------------------------------------------------ sidebar ---

cfg = get_cfg()
events = load_events()
spatial = load_spatial()

st.sidebar.title("HydroCascade")
st.sidebar.caption("Root cause analysis and cross-hazard point process framework "
                   "for rainfall-triggered disaster chains")
st.sidebar.markdown(
    f"**Region** `{cfg.region.id}`  \n"
    f"**Period** {cfg.period.start} → {cfg.period.end}  \n"
    f"**Analysis CRS** {cfg.crs.analysis}")

from hydrocascade.ingest import adapters  # noqa: E402
synthetic = adapters.synthetic_sources(cfg)
if synthetic:
    st.sidebar.error("**Synthetic stand-ins active**\n\n"
                     + ", ".join(f"`{s}`" for s in synthetic)
                     + "\n\nNo number on this page derived from these sources may "
                       "be reported. Switch `data_mode` in `config/default.yaml` "
                       "once the accounts exist.")

st.sidebar.markdown("---")
st.sidebar.caption("All six phases built. Any number derived from a synthetic "
                   "stand-in is labelled as such and blocked from the gates.")


# -------------------------------------------------------------------- header ---

st.title("HydroCascade Explorer")
st.caption("The event catalog, the spatial framework, the root-cause study, the "
           "fitted cascade model, and the Wayanad backtest.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("events", f"{len(events):,}" if events is not None else "—")
c2.metric("slope units", f"{len(spatial['units']):,}" if spatial["units"] is not None else "—")
c3.metric("river reaches", f"{len(spatial['network']):,}" if spatial["network"] is not None else "—")
c4.metric("basin area",
          f"{float(spatial['aoi']['area_km2'].iloc[0]):,.0f} km²"
          if spatial["aoi"] is not None else "—")
c5.metric("channel length",
          f"{spatial['network']['length_m'].sum()/1000:,.0f} km"
          if spatial["network"] is not None else "—")

tabs = st.tabs(["The cascade path", "Event catalog", "Spatial framework",
                "Rainfall", "Drivers & susceptibility", "The model",
                "Backtest & alerts", "Gates & reports", "How to reproduce"])


# ------------------------------------------------------- 1. the cascade path ---

with tabs[0]:
    st.subheader("Excitation arrives as a wave, not a field")
    st.markdown(
        "A landslide at Mundakkai does not raise flood risk everywhere at once. "
        "It raises risk **at the reaches its debris can actually reach**, "
        "**after the travel time of the flow**, and **diluted by how much the "
        "catchment has grown**. Those three factors are what Doc 02 §5 makes "
        "measurable, and this page is the evidence that the spatial framework "
        "computes them.")

    fig = figure("phase2_wayanad_flowpath.png")
    if fig:
        st.image(str(fig), width="stretch")
    else:
        st.info("Run `hydrocascade figures` to generate the flow-path figure.")

    try:
        chain, meta = load_cascade_chain()
        a, b, c, d = st.columns(4)
        a.metric("injection reach", f"R-{meta['injection_reach']:05d}")
        b.metric("d_slope, crown → river", f"{meta['d_slope_km']:.2f} km")
        c.metric("downstream reaches", f"{meta['n_downstream']:,}")
        d.metric("max travel time", f"{chain['hours'].max():.1f} h")

        st.markdown("**Doc 05 §5.4 spot check** — the reaches downstream of the "
                    "injection point, with reference travel time and dilution. "
                    "Travel time increases monotonically and the contributing area "
                    "grows, which is what a correct network must do.")
        show = chain.head(400).copy()
        show["reach"] = show["reach_id"].map(lambda r: f"R-{int(r):05d}")
        st.dataframe(
            show[["reach", "hours", "T_ref_km", "catchment_km2", "dilution", "strahler"]]
            .rename(columns={"hours": "travel time (h)", "T_ref_km": "T_ref (km)",
                             "catchment_km2": "catchment (km²)",
                             "dilution": "A(p)/A(s)"}),
            width="stretch", hide_index=True, height=300)

        st.markdown("**Dilution and arrival along the chain**")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.line_chart(chain.set_index("hours")[["catchment_km2"]],
                          y_label="catchment area (km²)", x_label="travel time (h)")
        with cc2:
            st.line_chart(chain.set_index("hours")[["dilution"]],
                          y_label="A(p)/A(s)", x_label="travel time (h)")
        st.caption("Left: the pulse enters a 3.8 km² headwater stream and ends in a "
                   "river draining hundreds of km². Right: the dilution factor Ω "
                   "that Doc 02 §5.3 raises to a fitted power ψ.")
    except FileNotFoundError as exc:
        st.warning(f"Spatial framework not built yet: {exc}")


# ------------------------------------------------------------ 2. the catalog ---

with tabs[1]:
    if events is None:
        st.info("No catalog yet. Run `hydrocascade catalog`.")
    else:
        st.subheader("Event catalog")
        n_synth = int((events["source"] == "SYNTHETIC").sum())
        if n_synth:
            st.warning(f"{n_synth:,} of {len(events):,} rows are **synthetic "
                       f"stand-ins**, not real events. They exist so the pipeline "
                       f"runs end to end while the inventory downloads are pending.")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Events by year**")
            by_year = (events.assign(year=events["_date"].dt.year)
                       .pivot_table(index="year", columns="hazard_type",
                                    aggfunc="size", fill_value=0))
            st.bar_chart(by_year)
        with right:
            st.markdown("**Seasonality** — Doc 04 §6 gate 3")
            by_month = (events.assign(month=events["_date"].dt.month)
                        .pivot_table(index="month", columns="hazard_type",
                                     aggfunc="size", fill_value=0)
                        .reindex(range(1, 13), fill_value=0))
            st.bar_chart(by_month)
            land = events[events["hazard_type"] == "L"]
            m = pd.to_datetime(land["date"], errors="coerce").dt.month
            st.caption(f"{100*float(m.between(6,9).mean()):.0f}% of landslides fall "
                       f"in June–September, the south-west monsoon.")

        f1, f2 = st.columns(2)
        with f1:
            htype = st.multiselect("hazard type", ["L", "F"], default=["L", "F"])
        with f2:
            sources = sorted(events["source"].unique())
            picked = st.multiselect("source", sources, default=sources)
        sel = events[events["hazard_type"].isin(htype) & events["source"].isin(picked)]
        st.caption(f"{len(sel):,} of {len(events):,} rows")

        st.map(sel.rename(columns={"lat": "latitude", "lon": "longitude"})
               .dropna(subset=["latitude", "longitude"])[["latitude", "longitude"]],
               size=60, color="#a4552f")

        st.dataframe(
            sel[["event_id", "hazard_type", "date", "lat", "lon", "area_m2",
                 "mark_value", "severity_class", "spatial_unit_id", "source",
                 "confidence"]].head(2000),
            width="stretch", hide_index=True, height=320)

        sf = figure("phase1_size_frequency.png")
        if sf:
            st.markdown("**Inventory completeness** — Doc 04 §3.4")
            st.image(str(sf), width="stretch")

        pending = paths.CATALOG / "events_PENDING.csv"
        if pending.is_file():
            st.markdown("**Documented events still awaiting provenance**")
            st.caption("Doc 04 §2 admits no exceptions: every catalog row carries a "
                       "source URL and an accessor. These Doc 04 §5 events are held "
                       "out of `events.csv` until someone sources their coordinates.")
            st.dataframe(pd.read_csv(pending)[["date", "notes"]],
                         width="stretch", hide_index=True)


# -------------------------------------------------- 3. the spatial framework ---

with tabs[2]:
    st.subheader("Two supports and the operator between them")
    st.markdown(
        "Landslides are points on a 2D hillslope surface **H**; floods are points "
        "on a 1D river network **N**. The projection operator **π: H → N** follows "
        "the D8 flow path from a hillslope point to the first channel cell it "
        "reaches. Doc 02 §2 calls this the answer to *what does location actually "
        "mean in your model?*")

    g1, g2 = st.columns(2)
    with g1:
        f = figure("phase2_network.png")
        if f:
            st.image(str(f), width="stretch")
    with g2:
        f = figure("phase2_slope_units.png")
        if f:
            st.image(str(f), width="stretch")

    if spatial["units"] is not None:
        u = spatial["units"]
        st.markdown("**Slope units — the hillslope domain H**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("units", f"{len(u):,}")
        m2.metric("median area", f"{u['area_km2'].median():.3f} km²")
        m3.metric("median slope", f"{u['slope_mean_deg'].median():.1f}°")
        m4.metric("elevation range",
                  f"{u['elev_mean_m'].min():.0f}–{u['elev_mean_m'].max():.0f} m")

    if spatial["network"] is not None:
        n = spatial["network"]
        st.markdown("**Channel network — the flood domain N**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("reaches", f"{len(n):,}")
        m2.metric("total length", f"{n['length_m'].sum()/1000:,.0f} km")
        m3.metric("max Strahler", f"{int(n['strahler'].max())}")
        m4.metric("median bed slope", f"{n['slope'].median():.4f}")
        st.markdown("**Reaches by Strahler order**")
        st.bar_chart(n.groupby("strahler").size().rename("reaches"))


# ------------------------------------------------------------- 4. rainfall ---

with tabs[3]:
    st.subheader("Rainfall windows and the Wayanad reconciliation")
    ds = load_rainfall()
    if ds is None:
        st.info("No rainfall built yet. Run `hydrocascade rainfall`.")
    else:
        lat = float(cfg.landmarks.mundakkai_crown["lat"])
        lon = float(cfg.landmarks.mundakkai_crown["lon"])
        cell = ds.sel(latitude=lat, longitude=lon, method="nearest")
        df = cell.to_dataframe().reset_index()
        cols = [c for c in df.columns if c.startswith("R") or c == "API"]

        yr = st.select_slider("year", options=sorted(df["time"].dt.year.unique()),
                              value=2024)
        sub = df[df["time"].dt.year == yr].set_index("time")
        pick = st.multiselect("series", cols, default=["R1", "R15"])
        if pick:
            st.line_chart(sub[pick], y_label="mm")

        st.markdown("**Doc 05 §7.1 — the single best sanity check in the project**")
        try:
            from hydrocascade.ingest import rainfall_sources
            r = rainfall_sources.wayanad_reconciliation(cfg)
            k1, k2, k3 = st.columns(3)
            k1.metric("published 48 h", f"{r['published_48h_mm']:.0f} mm")
            k2.metric("CHIRPS 48 h", f"{r['window_48h_mm']:.0f} mm",
                      delta=f"{r['window_48h_mm']-r['published_48h_mm']:.0f} mm")
            k3.metric("captured", f"{100*r['window_48h_mm']/r['published_48h_mm']:.0f}%")
            if not r["within_tolerance"]:
                st.error(
                    "**Gate 4 fails on CHIRPS, and it is not an extraction bug.** "
                    "CHIRPS does show a 240–290 mm band over 29–31 July 2024, but it "
                    "peaks roughly 50 km **west** of the crown — the documented "
                    "orographic displacement of satellite-blended products in the "
                    "Western Ghats, made worse by the 0.25° grid averaging over "
                    "~27 km. Reproducing the published 204 mm / 373 mm needs IMD "
                    "(dataset M1, gauge-based) or IMERG (M2, 0.1°). Both require an "
                    "account; see Doc 10 §7.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"reconciliation unavailable: {exc}")

        st.caption(f"Rainfall grid: {ds.sizes['latitude']}×{ds.sizes['longitude']} "
                   f"cells over the study bbox. At 0.25° the whole basin is covered "
                   f"by ~14 distinct pixels, which is the resolution Doc 10 §3 warns "
                   f"is 'too coarse for a single hillslope'.")


# ----------------------------------------- 5. drivers and susceptibility ---

with tabs[4]:
    st.subheader("Why this slope and not that one")
    st.markdown(
        "Part 1 answers the root-cause question and hands two things to the "
        "cascade model: a **calibrated** susceptibility surface `S_i`, and the "
        "two-to-four standardised drivers `e~(u)` that feed the receptivity "
        "term. Calibration is a hard requirement rather than a nicety — `S_i` "
        "multiplies the background intensity, so a score that ranks perfectly "
        "but sits 40% too high corrupts every number downstream.")

    _td = paths.PROCESSED / "top_drivers.json"
    if _td.is_file():
        import json as _json
        _d = _json.loads(_td.read_text(encoding="utf-8"))
        st.success("**e~(u) selected:** "
                   + "  ·  ".join(f"`{x}`" for x in _d["drivers"]))

    _g1, _g2 = st.columns(2)
    with _g1:
        _f = figure("phase3_shap_summary.png")
        if _f:
            st.image(str(_f), width="stretch")
    with _g2:
        _f = figure("phase3_reliability.png")
        if _f:
            st.image(str(_f), width="stretch")

    _f = figure("phase3_susceptibility_map.png")
    if _f:
        st.image(str(_f), width="stretch")

    for _name, _label in (("phase3_drivers.md", "Driver ranking"),
                          ("phase3_features.md", "Feature selection and collinearity"),
                          ("phase3_calibration.md", "Calibration"),
                          ("phase3_baselines.md", "Baselines B1-B3")):
        _p = paths.REPORTS / _name
        if _p.is_file():
            with st.expander(_label, expanded=False):
                st.markdown(_p.read_text(encoding="utf-8"))


# ------------------------------------------------------------ 6. the model ---

with tabs[5]:
    st.subheader("The cascade model, stage by stage")
    st.markdown(
        "Doc 07's governing rule: *never add a component until the previous one "
        "has been verified on data where you already know the answer.* So the "
        "first thing to look at is not a fit to Kerala — it is whether the "
        "likelihood can recover parameters from simulated catalogs whose truth "
        "we set ourselves.")

    _vp = paths.MODELS / "fits" / "stage_verification.json"
    if _vp.is_file():
        import json as _json
        _v = _json.loads(_vp.read_text(encoding="utf-8"))
        _cols = st.columns(4)
        for _col, _key in zip(_cols, ("cross_check", "S1", "S2", "S2-boundary")):
            _d = _v.get(_key, {})
            _col.metric(_key, "PASS" if _d.get("passed") else "FAIL")
        for _key in ("S1", "S2", "S2-boundary"):
            if _key in _v:
                st.caption(f"**{_key}** — {_v[_key].get('detail', '')}")
        st.info("The Doc 07 §2 cross-check against the `tick` library could not "
                "run — it needs Python 3.12 in a separate environment and does "
                "not install here. The substitute is three independent "
                "implementations of the same likelihood agreeing to machine "
                "precision. That is real evidence, but weaker than an external "
                "library, so the gate is recorded as BLOCKED rather than PASS.")

    _f = figure("phase4_parameter_recovery.png")
    if _f:
        st.image(str(_f), width="stretch")

    _fits = sorted((paths.MODELS / "fits").glob("*.json")) if (
        paths.MODELS / "fits").is_dir() else []
    _fits = [f for f in _fits if f.name != "stage_verification.json"]
    if _fits:
        import json as _json
        _rows = []
        for _f in _fits:
            try:
                _r = _json.loads(_f.read_text(encoding="utf-8"))
            except Exception:
                continue
            _rows.append({"fit": _r.get("name", _f.stem),
                          "logL": _r.get("log_likelihood"),
                          "AIC": _r.get("aic"),
                          "branching": _r.get("branching_ratio"),
                          "params": _r.get("n_params"),
                          "valid": _r.get("valid"),
                          "seed": _r.get("seed"),
                          "commit": _r.get("git_commit")})
        st.markdown("**Every fit on record** — Doc 07 §7 rule 8 makes each "
                    "number traceable to a specific run.")
        st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True,
                     height=260)

    for _name, _label in (("phase4_stages.md", "Stage-by-stage results"),
                          ("phase5_ablations.md", "Ablations and the nested ladder"),
                          ("phase5_baselines.md", "Baselines B1-B5 versus M5")):
        _p = paths.REPORTS / _name
        if _p.is_file():
            with st.expander(_label, expanded=False):
                st.markdown(_p.read_text(encoding="utf-8"))

    _g1, _g2 = st.columns(2)
    with _g1:
        _f = figure("ablation_table.png")
        if _f:
            st.image(str(_f), width="stretch")
    with _g2:
        _f = figure("phase5_ladder.png")
        if _f:
            st.image(str(_f), width="stretch")


# --------------------------------------------------- 7. backtest and alerts ---

with tabs[6]:
    st.subheader("Would it have called Wayanad?")
    st.markdown(
        "The model is refitted on data **strictly before 1 June 2024** — the "
        "leakage assertion raises rather than warns, and the test suite tries "
        "to break it — then rolled forward on observed rainfall only.")

    _f = figure("money_figure.png")
    if _f:
        st.image(str(_f), width="stretch")
    else:
        st.info("Run `hydrocascade experiments` to produce the money figure.")

    st.markdown("**Lead time and false-alarm rate are shown together, always.** "
                "A system that alarms every day has a magnificent lead time and "
                "is worthless; neither number means anything alone.")

    _f = figure("leadtime_vs_falsealarm.png")
    if _f:
        st.image(str(_f), width="stretch")

    for _name, _label in (("phase5_backtest.md", "The Wayanad backtest protocol"),
                          ("phase5_loco.md", "Leave-one-cascade-out"),
                          ("phase5_sensitivity.md", "Sensitivity analyses"),
                          ("phase6_alerts.md", "Alert levels and thresholds"),
                          ("phase6_architecture.md", "Real-time architecture")):
        _p = paths.REPORTS / _name
        if _p.is_file():
            with st.expander(_label, expanded=False):
                st.markdown(_p.read_text(encoding="utf-8"))


# --------------------------------------------------------- 5. quality gates ---

with tabs[7]:
    st.subheader("Quality gates — Doc 04 §6 and Doc 05 §9")
    st.caption("A phase is not done until its gates pass. `BLOCKED` means a gate "
               "could not be evaluated because an input is missing or synthetic — "
               "it is reported separately from `FAIL` so that *we have not got the "
               "data yet* can never be read as *the check passed*.")
    report = paths.REPORTS / "quality_gates.md"
    if report.is_file():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("Run `hydrocascade check` to generate the gate report.")

    st.markdown("---")
    st.subheader("Phase 1 — the flood definition")
    flood_def = paths.REPORTS / "phase1_flood_definition.md"
    if flood_def.is_file():
        st.caption("The Doc 04 §8 deliverable: the rule, its sensitivity to every "
                   "Doc 02 §3.5 variant, the eligibility blind spot at Chooralmala "
                   "verified against our own network, and the corroboration status. "
                   "Regenerate with `hydrocascade report`.")
        with st.expander("Read the full report", expanded=False):
            st.markdown(flood_def.read_text(encoding="utf-8"))
    else:
        st.info("Run `hydrocascade report` to generate the flood-definition report.")

    st.markdown("---")
    st.subheader("Cascade pairs — the validation table")
    st.caption("Doc 04 §5. Only `confirmed` and `probable` pairs evaluate cascade "
               "detection; the model is never fitted to this table. `lag_hours` is "
               "left blank on purpose — a lag computed from our own travel-time "
               "table would turn a prediction into its own ground truth.")
    pairs = paths.CATALOG / "cascade_pairs.csv"
    if pairs.is_file():
        st.dataframe(pd.read_csv(pairs, dtype=str, keep_default_na=False),
                     width="stretch", hide_index=True)
    else:
        st.info("Run `hydrocascade catalog` to seed the cascade pair table.")


# ------------------------------------------------------------ 6. reproduce ---

with tabs[8]:
    st.subheader("Reproducing this from an empty data directory")
    st.code(
        "python -m venv .venv --system-site-packages\n"
        ".venv\\Scripts\\pip install -r requirements.txt\n"
        ".venv\\Scripts\\pip install -e .\n"
        "$env:HYDROCASCADE_USER = 'your name'\n\n"
        "hydrocascade aoi          # study boundary from HydroBASINS\n"
        "hydrocascade dem          # Copernicus GLO-30, mosaicked and reprojected\n"
        "hydrocascade flow         # sink fill, D8, flow accumulation\n"
        "hydrocascade network      # channel network, 500 m reaches\n"
        "hydrocascade units        # slope units (the hillslope domain H)\n"
        "hydrocascade projection   # the operator pi: H -> N\n"
        "hydrocascade traveltime   # downstream reachability + travel time\n"
        "hydrocascade rainfall     # CHIRPS + rainfall windows + API\n"
        "hydrocascade catalog      # the event catalog\n"
        "hydrocascade covariates   # static terrain covariates\n"
        "hydrocascade figures      # every figure in Docs 04 and 05\n"
        "hydrocascade report       # reports/phase1_flood_definition.md\n\n"
        "hydrocascade rootcause    # Phase 3: drivers, SHAP, susceptibility\n"
        "hydrocascade cascade      # Phase 4: fits S3, S4, S5\n"
        "hydrocascade experiments  # Phase 5: ladder, ablations, backtest\n"
        "hydrocascade realtime     # Phase 6: alert thresholds\n\n"
        "hydrocascade check        # the quality gates, all six phases\n"
        "hydrocascade app          # this page",
        language="powershell")
    st.markdown(
        "Every threshold, date, CRS and path comes from `config/default.yaml`, so "
        "an experiment is reproducible from a config file alone (Doc 11 §5 rule 3). "
        "Nothing above needs an account: the DEM, basin boundary and rainfall are "
        "all free and login-free.")
    st.markdown("**All six phases are built.** What is still missing is data, "
                "not code:")
    st.markdown(
        "- a **real landslide inventory** in `data/raw/` — until then the "
        "catalog uses a labelled synthetic one and gate P1.1 is `BLOCKED`\n"
        "- **GloFAS** discharge files or a CDS key, so flood events come from "
        "observed discharge rather than a stand-in\n"
        "- **IMD or IMERG** rainfall — CHIRPS sees 71 mm of the 577 mm that "
        "fell at Mundakkai, which is why gate P2.4 fails and the backtest "
        "shows no lead time\n"
        "- **Sentinel-1 via Earth Engine** — the one piece still unwired; the "
        "SAR corroboration gate stays `BLOCKED` and prints its recipe\n\n"
        "`docs/15_HOW_TO_RUN.md` gives the download links, the exact folder "
        "each file goes in, and what to re-run afterwards.")
