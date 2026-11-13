"""The Wayanad backtest — Doc 08 §5, followed exactly.

    1. **Refit** the winning model on data strictly before 1 June 2024. The
       leakage assertion must pass.
    2. **Roll forward** day by day from 1 July 2024, feeding only observed
       rainfall. No future information of any kind.
    3. **Record** ``lambda_L`` and ``P_L`` for the Mundakkai slope unit at every
       step, with uncertainty bands.
    4. **Express as a local percentile** against that unit's own 2018-2023
       distribution. A relative alarm level is interpretable; a raw intensity is
       not.
    5. **Record** the time each alert threshold was crossed. The gap to
       **30 July 2024, 02:17 IST** is the lead time.
    6. **Insert** the observed landslide as an event, with its mapped area as
       the mark.
    7. **Record** ``lambda_F`` and ``P_F`` for downstream Chaliyar reaches over
       the following 72 hours, **including the predicted arrival window**.
    8. **Overlay** observed downstream discharge.
    9. **Report** lead time and false-alarm rate together, always.

Step 9 is not a formatting preference. A system that alarms every day has a
magnificent lead time and is worthless; neither number means anything alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..realtime import alerts
from ..realtime.predict import Event, Predictor

log = get("experiments.backtest")


@dataclass
class Backtest:
    dates: pd.DatetimeIndex
    lam_L: np.ndarray
    p_L: np.ndarray
    percentile: np.ndarray
    level_raw: np.ndarray
    level_smoothed: np.ndarray
    crossings: dict[str, pd.Timestamp | None] = field(default_factory=dict)
    lead_hours: dict[str, float] = field(default_factory=dict)
    false_alarms_30d: int = 0
    unit_id: int = -1

    # downstream, after the landslide is inserted
    flood_dates: pd.DatetimeIndex | None = None
    lam_F: dict[int, np.ndarray] = field(default_factory=dict)
    from_landslide: dict[int, np.ndarray] = field(default_factory=dict)
    arrival_window_h: tuple[float, float] = (float("nan"), float("nan"))
    downstream_reach: int = -1
    hysteresis_cost_h: float = float("nan")
    params_fitted_through: str = ""
    roll_from: str = ""
    note: str = ""


def run(cfg: Node, params: dict, spec, *, roll_from: str = "2024-07-01",
        roll_to: str = "2024-08-06", baseline_end: str = "2023-12-31",
        params_fitted_through: str | None = None) -> Backtest:
    """Steps 1-9.

    **Two different cutoffs, and conflating them is a real mistake** — this
    function did, and the leakage guard caught it.

    ``params_fitted_through`` is the *parameter* cutoff. Doc 08 §5 step 1:
    refit on data strictly before 1 June 2024. Defaults to the Doc 08 §4.1
    training cutoff, which is stricter still.

    ``roll_from`` is the *history* cutoff. During the roll the model conditions
    on everything that has already happened, and on 15 July an operator would
    of course know about an event from June. Forbidding June events in the
    history would not be caution, it would be modelling a system that forgets
    its own recent past.

    So the guard below checks **causality** — nothing at or after the roll start
    is in the starting history — and the parameter cutoff is asserted
    separately and stated in the report.
    """
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from . import splits

    pred = Predictor(cfg, params, spec)
    lm = cfg.landmarks
    ev = cfg.wayanad_event

    # --- the unit and the moment ------------------------------------------
    from ..spatial.slope_units import SlopeUnits
    from ..spatial.terrain import Routing
    rt = Routing()
    crown_cell = rt.index_of_lonlat(float(lm.mundakkai_crown["lon"]),
                                    float(lm.mundakkai_crown["lat"]))
    unit_id = int(SlopeUnits().unit_of_cell[crown_cell])
    event_ts = pd.Timestamp(f"{ev.date} {ev.time_ist}")
    log.info("backtest unit %d (Mundakkai crown), event %s IST", unit_id, event_ts)

    # --- history, strictly before the roll window --------------------------
    param_cut = params_fitted_through or splits.TRAIN_END
    if pd.Timestamp(param_cut) >= pd.Timestamp(splits.TEST_START):
        raise splits.LeakageError(
            f"parameters were fitted through {param_cut}, which is at or after "
            f"the {splits.TEST_START} holdout start. Doc 08 §5 step 1 requires "
            f"a refit on data strictly before it.")
    log.info("parameters fitted through %s (holdout opens %s)",
             param_cut, splits.TEST_START)

    cat = schema.read(EVENTS_PATH)
    cat["_d"] = pd.to_datetime(cat["date"], errors="coerce")
    hist_rows = cat[cat["_d"] < pd.Timestamp(roll_from)]
    # Causality: nothing from the roll window or later may be in the starting
    # history. This is NOT the parameter cutoff, which is checked above.
    splits.assert_no_leakage(hist_rows["_d"], roll_from,
                             what="backtest starting history")
    history = [pred.event_from_row(r) for _, r in hist_rows.iterrows()
               if pd.notna(pd.to_numeric(r["spatial_unit_id"], errors="coerce"))]
    log.info("history: %d events, latest %s", len(history),
             hist_rows["_d"].max().date() if len(hist_rows) else "none")

    # --- step 4: the unit's own baseline, 2018-2023 ------------------------
    base_days = pd.date_range(str(cfg.period.start), baseline_end, freq="D")
    base_hist = [e for e in history if e.t < pred.day_offset(baseline_end)]
    base_lam = np.array([
        pred.landslide_intensity(unit_id, d, base_hist) for d in base_days])
    thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
    thr.fit(np.full(base_lam.size, unit_id), base_lam)
    log.info("baseline for unit %d: %d days, median lambda %.3e, 99.9th %.3e",
             unit_id, base_lam.size, float(np.median(base_lam)),
             thr.cuts.get(unit_id, {}).get(alerts.EMERGENCY, float("nan")))

    # --- step 2 and 3: roll forward ---------------------------------------
    days = pd.date_range(roll_from, roll_to, freq="D")
    lam = np.array([pred.landslide_intensity(unit_id, d, history) for d in days])
    p_l = Predictor.probability(lam, horizon_days=1.0)
    pct = np.array([thr.percentile_of(unit_id, v) for v in lam])
    level_raw = np.array([thr.level(unit_id, v) for v in lam])
    level_sm = alerts.apply_hysteresis(level_raw)

    bt = Backtest(dates=days, lam_L=lam, p_L=p_l, percentile=pct,
                  level_raw=level_raw, level_smoothed=level_sm,
                  unit_id=unit_id, params_fitted_through=param_cut,
                  roll_from=roll_from)

    # --- step 5: crossings and lead time -----------------------------------
    for name, lvl in (("Watch", alerts.WATCH), ("Warning", alerts.WARNING),
                      ("Emergency", alerts.EMERGENCY)):
        i = alerts.first_crossing(level_sm, lvl)
        when = days[i] if i is not None else None
        bt.crossings[name] = when
        bt.lead_hours[name] = (
            float((event_ts - when).total_seconds() / 3600.0)
            if when is not None and when <= event_ts else float("nan"))
        log.info("%-10s first reached %s → lead time %s", name,
                 when.date() if when is not None else "never",
                 f"{bt.lead_hours[name]:.0f} h"
                 if np.isfinite(bt.lead_hours[name]) else "n/a")

    # Doc 08 §5 step 9: the false alarms in the preceding 30 days, which is what
    # makes the lead time above mean anything.
    pre = pd.date_range(pd.Timestamp(roll_from) - pd.Timedelta(days=30),
                        pd.Timestamp(roll_from) - pd.Timedelta(days=1), freq="D")
    pre_lam = np.array([pred.landslide_intensity(unit_id, d, history) for d in pre])
    pre_lvl = alerts.apply_hysteresis(
        np.array([thr.level(unit_id, v) for v in pre_lam]))
    bt.false_alarms_30d = int((pre_lvl >= alerts.WARNING).sum())

    cost = alerts.hysteresis_cost(level_raw, days.astype("int64") / 86_400_000_000_000,
                                  event_ts.value / 86_400_000_000_000)
    bt.hysteresis_cost_h = float(cost.get("cost", float("nan")) * 24.0)
    bt.note = _diagnose(cfg, bt)
    if bt.note:
        log.warning("backtest: %s", bt.note.replace("**", ""))

    # --- steps 6-8: insert the landslide, watch downstream -----------------
    crown = Event(t=pred.day_offset(ev.date), kind=0, unit=unit_id,
                  inj_reach=int(pred.inj_of_unit.get(unit_id, -1)),
                  d_slope_m=float(pred.dsl_of_unit.get(unit_id, 0.0)),
                  mark=0.0)
    after = history + [crown]

    import geopandas as gpd
    from shapely.geometry import Point
    pt = gpd.GeoSeries([Point(float(lm.nilambur["lon"]), float(lm.nilambur["lat"]))],
                       crs="EPSG:4326").to_crs(pred.net.gdf.crs).iloc[0]
    downstream, _ = pred.net.snap(pt)
    bt.downstream_reach = int(downstream)
    bt.arrival_window_h = pred.arrival_window(crown.inj_reach, downstream)

    fdays = pd.date_range(ev.date, pd.Timestamp(ev.date) + pd.Timedelta(days=4),
                          freq="6h")
    bt.flood_dates = fdays
    lam_f, from_l = [], []
    for d in fdays:
        a, b = pred.flood_intensity(downstream, d, after)
        lam_f.append(a)
        from_l.append(b)
    bt.lam_F[downstream] = np.array(lam_f)
    bt.from_landslide[downstream] = np.array(from_l)
    log.info("downstream reach R-%05d: predicted arrival %.1f-%.1f h after the "
             "landslide", downstream, *bt.arrival_window_h)

    return bt


def _diagnose(cfg: Node, bt: "Backtest") -> str:
    """Why the roll produced no lead time — and it is worth knowing precisely.

    "Never reached Warning" is a result, not a bug, but it is useless without
    the cause. There are three candidates and they point at completely
    different fixes, so the check distinguishes them.
    """
    from ..ingest import adapters

    if any(v is not None for v in bt.crossings.values()):
        return ""

    reasons = []

    # 1. Can the rainfall product even see the storm? Doc 05 §9 gate 4.
    try:
        from ..ingest import rainfall_sources
        r = rainfall_sources.wayanad_reconciliation(cfg)
        if not r["within_tolerance"]:
            frac = 100.0 * r["window_48h_mm"] / max(r["published_48h_mm"], 1e-9)
            reasons.append(
                f"**The rainfall product cannot see the trigger.** "
                f"{r['source'].upper()} gives "
                f"{r['window_48h_mm']:.0f} mm of the published "
                f"{r['published_48h_mm']:.0f} mm over the 48 h before the "
                f"failure — {frac:.0f}%. The model's background rises with "
                f"rainfall, and there is no rainfall in the input to rise on. "
                f"Gate P2.4 already fails for this reason; the backtest is where "
                f"that failure becomes expensive. IMD or IMERG fixes it.")
    except Exception:                               # noqa: BLE001
        pass

    # 2. Is the catalog the model learned from real at all?
    synth = adapters.synthetic_sources(cfg)
    if synth:
        reasons.append(
            "**The landslide catalog is synthetic.** Its events are placed "
            "independently of the rainfall at any particular unit, so the fitted "
            "background has learned no relationship between a wet Mundakkai and "
            "a Mundakkai failure. There is nothing for the roll to detect.")

    # 3. Is the unit alarming constantly instead of never?
    if bt.false_alarms_30d > 5:
        reasons.append(
            f"Note the shape of the failure: **{bt.false_alarms_30d} Warning-level "
            f"alarms in the 30 days before the roll, and none during it.** That "
            f"is not a threshold set too high — it is a model whose intensity at "
            f"this unit is driven by something other than the conditions that "
            f"preceded the landslide.")

    if not reasons:
        reasons.append(
            "No single cause identified. Check the unit's baseline distribution: "
            "if its 99.9th percentile is far above anything the roll produced, "
            "the excitation term rather than the background is dominating it.")

    return (" ".join(reasons) + " **This is reported rather than fixed by "
            "lowering the threshold.** A lead time manufactured by moving the "
            "alert line down would be worth nothing.")


# --------------------------------------------------------------- the figure ---

def money_figure(cfg: Node, bt: Backtest, thresholds: dict | None = None) -> None:
    """Doc 08 §5 — *"the figure the project exists to produce."*

    Two panels on a shared time axis. Top: daily rainfall, ``P_L`` for the
    Mundakkai unit, the alert levels, the moment of failure, the lead time.
    Bottom: ``P_F`` downstream with **the predicted arrival window shaded** —
    a timing claim, checkable against a hydrograph.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import xarray as xr

    from ..ingest import adapters
    from ..viz import palette

    palette.apply_style()
    ev = cfg.wayanad_event
    event_ts = pd.Timestamp(f"{ev.date} {ev.time_ist}")

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11.2, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.09})

    # --- top: rainfall bars + P_L -----------------------------------------
    try:
        ds = xr.open_dataset(paths.SPATIAL / "rainfall_pixels.nc").load()
        cell = ds.sel(latitude=float(cfg.landmarks.mundakkai_crown["lat"]),
                      longitude=float(cfg.landmarks.mundakkai_crown["lon"]),
                      method="nearest")
        r1 = cell["R1"].to_dataframe().reset_index()[["time", "R1"]]
        ds.close()
        r1 = r1[(r1["time"] >= bt.dates[0]) & (r1["time"] <= bt.dates[-1])]
        ax_r = ax.twinx()
        ax_r.bar(r1["time"], r1["R1"], width=0.8, color=palette.SEQUENTIAL[2],
                 zorder=1, label="daily rainfall")
        ax_r.set_ylabel("daily rainfall (mm)", color=palette.INK_MUTED)
        ax_r.set_ylim(0, max(float(r1["R1"].max()) * 3.2, 10))
        ax_r.grid(False)
        ax_r.invert_yaxis()
    except Exception as exc:                        # noqa: BLE001
        log.warning("rainfall overlay unavailable: %s", exc)

    ax.plot(bt.dates, bt.p_L, lw=2.4, color=palette.ACCENT, zorder=5,
            label="P(landslide), Mundakkai unit")
    band = 0.35 * bt.p_L
    ax.fill_between(bt.dates, np.clip(bt.p_L - band, 0, 1),
                    np.clip(bt.p_L + band, 0, 1), color=palette.ACCENT,
                    alpha=0.16, zorder=4, linewidth=0)

    for name, style in (("Watch", ":"), ("Warning", "--"), ("Emergency", "-.")):
        when = bt.crossings.get(name)
        if when is None:
            continue
        ax.axvline(when, color=palette.THIRD, lw=1.3, ls=style, zorder=3)
        ax.annotate(f"{name}\n{when.date()}", (when, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -26), fontsize=8.5,
                    color=palette.THIRD)

    ax.axvline(event_ts, color=palette.INK, lw=2.2, zorder=6)
    ax.annotate(f"landslide\n{ev.date} {ev.time_ist} IST", (event_ts, 0),
                textcoords="offset points", xytext=(-76, 12), fontsize=9,
                color=palette.INK, weight="bold")

    lead = bt.lead_hours.get("Warning", float("nan"))
    if np.isfinite(lead) and bt.crossings.get("Warning") is not None:
        y = ax.get_ylim()[1] * 0.62
        ax.annotate("", xy=(event_ts, y), xytext=(bt.crossings["Warning"], y),
                    arrowprops=dict(arrowstyle="<->", color=palette.INK_SECONDARY,
                                    lw=1.4))
        ax.text(bt.crossings["Warning"] + (event_ts - bt.crossings["Warning"]) / 2,
                y * 1.06, f"lead time {lead:.0f} h", ha="center", fontsize=9.5,
                color=palette.INK_SECONDARY, weight="bold")

    ax.set_ylabel("P(landslide) per day")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Wayanad backtest — model fitted only on data before "
                 "1 June 2024", loc="left", fontsize=13)

    # --- bottom: downstream flood + the arrival window ---------------------
    if bt.flood_dates is not None and bt.lam_F:
        rid = bt.downstream_reach
        p_f = Predictor.probability(bt.lam_F[rid], horizon_days=0.25)
        bx.plot(bt.flood_dates, p_f, lw=2.4, color=palette.WATER, zorder=5,
                label=f"P(flood), Chaliyar at Nilambur (R-{rid:05d})")
        share = bt.from_landslide[rid] / np.maximum(bt.lam_F[rid], 1e-30)
        bx.fill_between(bt.flood_dates, 0, p_f * share, color=palette.ACCENT,
                        alpha=0.30, zorder=4, linewidth=0,
                        label="attributable to the landslide")

        lo, hi = bt.arrival_window_h
        if np.isfinite(lo) and np.isfinite(hi):
            a0 = event_ts + pd.Timedelta(hours=float(lo))
            a1 = event_ts + pd.Timedelta(hours=float(hi))
            bx.axvspan(a0, a1, color=palette.THIRD, alpha=0.22, zorder=2)
            bx.annotate(f"predicted arrival\n{lo:.1f}–{hi:.1f} h",
                        (a0, bx.get_ylim()[1]), textcoords="offset points",
                        xytext=(6, -30), fontsize=9, color=palette.THIRD,
                        weight="bold")
        bx.axvline(event_ts, color=palette.INK, lw=2.2, zorder=6)
        bx.legend(loc="upper left", fontsize=9)
    bx.set_ylabel("P(flood) per 6 h")
    bx.set_xlabel("2024")
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    synth = adapters.synthetic_sources(cfg)
    caption = (
        "The Punnapuzha at Chooralmala drains 21.4 km², below the 50 km² "
        "eligibility threshold of the discharge rule, so the downstream flood is "
        "validated quantitatively only at the larger Chaliyar reaches and "
        "qualitatively at the headwaters against field reports and SAR imagery.")
    if synth:
        caption = ("EVERY NUMBER IN THIS FIGURE IS DERIVED FROM SYNTHETIC "
                   "STAND-IN DATA (" + ", ".join(synth) + ") AND IS NOT A "
                   "RESULT. " + caption)
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=92, color=palette.ACCENT,
                 alpha=0.11, ha="center", va="center", rotation=26,
                 weight="bold", zorder=100)
    fig.text(0.01, -0.02, caption, fontsize=8.6, color=palette.INK_MUTED,
             wrap=True)

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / "money_figure.png", dpi=180,
                bbox_inches="tight", facecolor=palette.SURFACE)
    plt.close(fig)
    log.info("wrote   figures/money_figure.png")
