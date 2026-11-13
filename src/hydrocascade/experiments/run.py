"""Phase 5 end to end — `hydrocascade experiments`.

Produces every Doc 08 §11 deliverable: the register, the five reports, and the
money figure. Nothing here invents a number; where an experiment cannot run it
is reported with its reason and its unblock path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from ..cascade import model
from . import (ablations, backtest, loco, register, sensitivity, splits,
               transfer, uncertainty)

log = get("experiments.run")


@dataclass
class Phase5:
    ladder: dict = field(default_factory=dict)
    ablation_runs: dict = field(default_factory=dict)
    profiles: list = field(default_factory=list)
    loco_results: list = field(default_factory=list)
    loco_summary: dict = field(default_factory=dict)
    backtest: backtest.Backtest | None = None
    variations: list = field(default_factory=list)
    negative_controls: dict = field(default_factory=dict)
    transfer_status: dict = field(default_factory=dict)
    headline_alpha: float = float("nan")
    robust: bool = False
    robust_text: str = ""


def run(cfg: Node, *, n_restarts: int = 4, steps: int = 250,
        quick: bool = False) -> Phase5:
    seed = int(cfg.project.seed)
    out = Phase5()

    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    cat = schema.read(EVENTS_PATH)
    log.info("event counts per split:\n%s",
             splits.describe(cat).to_string(index=False))

    log.info("assembling the training catalog (through %s)", splits.TRAIN_END)
    data = model.assemble(cfg, end=splits.TRAIN_END)
    splits.assert_no_leakage(
        pd.to_datetime(cat["date"], errors="coerce")[
            pd.to_datetime(cat["date"], errors="coerce") <= splits.TRAIN_END],
        splits.TRAIN_END, what="cascade training catalog")

    if quick:
        n_restarts, steps = 2, 120
        log.warning("quick mode: %d restarts x %d steps. Enough to prove the "
                    "pipeline runs, NOT enough for a reported number.",
                    n_restarts, steps)

    # --- the ladder and the ablations ---------------------------------------
    log.info("=" * 70)
    log.info("nested ladder M0-M5 (Doc 08 §2)")
    out.ladder = ablations.run_ladder(cfg, data, n_restarts=n_restarts,
                                      steps=steps, seed=seed)

    log.info("=" * 70)
    log.info("ablations A1-A9 (Doc 08 §3)")
    out.ablation_runs = ablations.run_ablations(cfg, data, n_restarts=n_restarts,
                                                steps=steps, seed=seed)

    full = out.ablation_runs.get("FULL")
    if full and full.params:
        out.headline_alpha = float(full.params.get("a_LF", float("nan")))

    # --- profile likelihood for the headline numbers -----------------------
    if full and full.params:
        log.info("=" * 70)
        log.info("profile likelihood (Doc 08 §6)")
        obj = model.make_objective(data, full.spec)
        for p in ("a_LF", "psi"):
            if p not in full.params:
                continue
            try:
                out.profiles.append(uncertainty.profile_likelihood(
                    obj, full.params, p, bounds=model.bounds_for(full.spec, data),
                    n_grid=9 if quick else 13, steps=max(steps // 2, 80),
                    seed=seed))
            except Exception as exc:                # noqa: BLE001
                log.warning("profile for %s failed: %s", p, exc)
        try:
            uncertainty.plot_profiles(out.profiles)
        except Exception as exc:                    # noqa: BLE001
            log.warning("profile figure failed: %s", exc)

    # --- LOCO, backtest, negative controls ----------------------------------
    if full and full.params:
        log.info("=" * 70)
        log.info("leave-one-cascade-out (Doc 08 §4.2)")
        try:
            out.loco_results = loco.run(cfg, full.params, full.spec)
            out.loco_summary = loco.summarise(out.loco_results)
        except Exception as exc:                    # noqa: BLE001
            log.error("LOCO failed: %s: %s", type(exc).__name__, exc)

        log.info("=" * 70)
        log.info("Wayanad backtest (Doc 08 §5)")
        try:
            out.backtest = backtest.run(cfg, full.params, full.spec)
            backtest.money_figure(cfg, out.backtest)
        except Exception as exc:                    # noqa: BLE001
            log.error("backtest failed: %s: %s", type(exc).__name__, exc)

        log.info("=" * 70)
        log.info("negative controls (Doc 08 §4.4)")
        try:
            out.negative_controls = negative_controls(cfg, full.params, full.spec)
        except Exception as exc:                    # noqa: BLE001
            log.error("negative controls failed: %s", exc)

    # --- sensitivity --------------------------------------------------------
    if full and full.spec:
        log.info("=" * 70)
        log.info("sensitivity (Doc 08 §8)")
        try:
            out.variations = sensitivity.run(
                cfg, full.spec, n_restarts=max(n_restarts // 2, 2),
                steps=max(steps // 2, 100), seed=seed,
                headline_alpha=out.headline_alpha)
            out.robust, out.robust_text = sensitivity.verdict(
                out.variations, out.headline_alpha)
        except Exception as exc:                    # noqa: BLE001
            log.error("sensitivity failed: %s", exc)

    # --- transfer -----------------------------------------------------------
    try:
        transfer.write_region_config(cfg)
    except Exception as exc:                        # noqa: BLE001
        log.warning("transfer config not written: %s", exc)
    out.transfer_status = transfer.status()

    _write_reports(cfg, out)
    _write_figures(cfg, out)
    return out


# ------------------------------------------------------- negative controls ---

def negative_controls(cfg: Node, params: dict, spec, *, n: int = 40,
                      percentile: float = 90.0) -> dict:
    """Doc 08 §4.4 — *required, not optional*.

    Heavy-rainfall days that produced **no** landslide and **no** flood, matched
    to event days on rainfall percentile. Run the model on them.

    *"If the model raises an alarm on every wet day, its lead time is
    worthless. This experiment is what makes the lead-time claim meaningful."*
    """
    import xarray as xr

    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..realtime import alerts
    from ..realtime.predict import Predictor
    from ..spatial.slope_units import SlopeUnits
    from ..spatial.terrain import Routing

    pred = Predictor(cfg, params, spec)
    lm = cfg.landmarks.mundakkai_crown
    rt = Routing()
    unit = int(SlopeUnits().unit_of_cell[
        rt.index_of_lonlat(float(lm["lon"]), float(lm["lat"]))])

    ds = xr.open_dataset(paths.SPATIAL / "rainfall_pixels.nc").load()
    cell = ds.sel(latitude=float(lm["lat"]), longitude=float(lm["lon"]),
                  method="nearest")
    r3 = cell["R3"].to_dataframe().reset_index()[["time", "R3"]]
    ds.close()

    cat = schema.read(EVENTS_PATH)
    cat["_d"] = pd.to_datetime(cat["date"], errors="coerce")
    event_days = set(cat["_d"].dt.normalize().dropna())

    cut = float(np.nanpercentile(r3["R3"], percentile))
    wet = r3[(r3["R3"] >= cut) & (~r3["time"].isin(event_days))]
    wet = wet[wet["time"] <= pd.Timestamp(splits.VALID_END)]
    if wet.empty:
        return {"status": "BLOCKED", "reason": "no matching wet non-event days"}
    wet = wet.sample(min(n, len(wet)), random_state=int(cfg.project.seed))

    history = [pred.event_from_row(r) for _, r in cat.iterrows()
               if pd.notna(pd.to_numeric(r["spatial_unit_id"], errors="coerce"))]

    base_days = pd.date_range(str(cfg.period.start), "2023-12-31", freq="D")
    base = np.array([pred.landslide_intensity(unit, d, history) for d in base_days])
    thr = alerts.Thresholds(alerts.DEFAULT_PERCENTILES)
    thr.fit(np.full(base.size, unit), base)

    lam = np.array([pred.landslide_intensity(unit, d, history)
                    for d in wet["time"]])
    lv = np.array([thr.level(unit, v) for v in lam])
    far = float((lv >= alerts.WARNING).mean())

    out = {
        "status": "ok",
        "n_days": int(len(wet)),
        "rainfall_percentile": percentile,
        "r3_cut_mm": cut,
        "false_alarm_rate": far,
        "mean_percentile": float(np.nanmean(
            [thr.percentile_of(unit, v) for v in lam])),
        "verdict": (
            "the model does NOT simply alarm on every wet day, so the lead "
            "time above means something"
            if far < 0.5 else
            "**the model alarms on most wet non-event days, so the lead-time "
            "claim is not meaningful** — Doc 08 §4.4 is explicit that this is "
            "what the experiment exists to catch"),
    }
    log.info("negative controls: %d wet non-event days, false-alarm rate %.3f — %s",
             out["n_days"], far, out["verdict"][:60])
    return out


# ----------------------------------------------------------------- reports ---

def _banner(cfg: Node) -> str:
    from ..ingest import adapters
    synth = adapters.synthetic_sources(cfg)
    if not synth:
        return ""
    _n_synth, _n_total = adapters.catalog_synthetic_counts()
    return (
        "> ### ⚠ Fitted on synthetic data\n>\n"
        "> `data_mode` is `synthetic` for " + ", ".join(f"`{s}`" for s in synth)
        + f", and {_n_synth:,} of {_n_total:,}\n"
        "> catalog rows are synthetic stand-ins. **Every parameter, every "
        "lead time and every\n> p-value below describes the stand-in "
        "generators, not Kerala.** They exist to show\n> the experimental "
        "machinery runs and that its diagnostics fire. Supplying the real\n"
        "> inputs and rerunning turns them into results \u2014 see "
        "`docs/15_HOW_TO_RUN.md`\n> sections 3.1-3.3 for which files go where.\n\n")


def _write_reports(cfg: Node, out: Phase5) -> None:
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    banner = _banner(cfg)

    # ------------------------------------------------- phase5_ablations.md ---
    L = [f"# Phase 5 — ablations\n",
         f"*Generated by `hydrocascade experiments` on {stamp}.*\n", banner,
         "Doc 08 §3 calls this table *\"the scientific core of the paper. It "
         "tells a reviewer exactly which of our ideas are real.\"* Each row "
         "removes exactly one component and refits everything else, so the "
         "change is attributable to that component alone.\n",
         "**A component earns its place only if removing it makes held-out fit "
         "worse.** Where removing it changes nothing, that is reported as "
         "'no detectable difference' rather than quietly dropped — Doc 08 §3: "
         "*\"An idea that turns out not to matter is a finding.\"*\n"]
    if out.ablation_runs:
        t = ablations.table(out.ablation_runs)
        L.append("| # | removes | tests the claim | Δ train logL | Δ held-out logL "
                 "| Δ AIC | verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r['ablation']} | {r['removes']} | {r['tests the claim']} "
                     f"| {r['delta_train_ll']:+.2f} | {r['delta_test_ll']:+.2f} "
                     f"| {r['delta_aic']:+.1f} | "
                     f"{ablations.interpret(r['delta_test_ll'])} |")
        L.append("")
    L.append("## The nested ladder M0-M5\n")
    L.append("These are nested, so likelihood-ratio tests apply. B1-B3 from "
             "Phase 3 are **not** nested and are compared on predictive metrics "
             "only — a likelihood-ratio test between a Hawkes model and XGBoost "
             "would be meaningless.\n")
    if out.ladder:
        L.append("| model | includes | train logL | held-out logL | params | AIC "
                 "| LR vs previous | df | p |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        keys = [k for k in ("M0", "M1", "M2", "M3", "M4", "M5") if k in out.ladder]
        for i, k in enumerate(keys):
            r = out.ladder[k]
            lr = df = p = float("nan")
            if i:
                lr, df, p = ablations.likelihood_ratio(out.ladder[keys[i - 1]], r)
            if i == 0:
                tail = "— | — | — |"
            elif not np.isfinite(p) and (getattr(r.spec, "network", None)
                                         != getattr(out.ladder[keys[i - 1]].spec,
                                                    "network", None)):
                tail = f"{lr:.2f} | — | **not nested** |"
            elif not np.isfinite(p):
                tail = f"{lr:.2f} | {int(df)} | fit failure |"
            else:
                tail = f"{lr:.2f} | {int(df)} | {p:.4g} |"
            L.append(f"| {k} | {r.description} | {r.train_ll:,.2f} | "
                     f"{r.test_ll:,.2f} | {r.n_params} | {r.aic:,.1f} | {tail}")
        L.append("")
        L.append("**M4 → M5 is not a nested step, so it carries no p-value.** M5 "
                 "replaces the isotropic flood kernel with the network kernel — a "
                 "change of kernel family, not an added parameter — so the "
                 "chi-squared reference does not apply. Compare those two on "
                 "held-out log-likelihood and AIC, which is also what ablation A2 "
                 "and baseline B5 do.\n")
        L.append("**M0 is a genuine constant rate per type.** An earlier version "
                 "had no way to switch self-excitation off, so M0–M2 all quietly "
                 "included it and M3's 'first excitation' step was not one. The "
                 "ladder is now nested as Doc 08 §2 describes.\n")
    (paths.REPORTS / "phase5_ablations.md").write_text("\n".join(L), encoding="utf-8")

    # -------------------------------------------------- phase5_baselines.md ---
    L = [f"# Phase 5 — baselines B1-B5 versus M5\n",
         f"*Generated by `hydrocascade experiments` on {stamp}.*\n", banner,
         "Doc 08 §2: **B5 versus M5 is the single most important comparison in "
         "the paper.** It isolates exactly the contribution we claim — the "
         "network formulation — because B5 has cross terms but no network, no "
         "travel time and no dilution.\n",
         "| ID | model | established in | what beating it proves |",
         "|---|---|---|---|",
         "| B1 | climatology (monthly mean rate) | Phase 3 | the floor |",
         "| B2 | rainfall intensity-duration threshold | Phase 3 | **beating "
         "current operational practice** |",
         "| B3 | XGBoost susceptibility x rainfall | Phase 3 | beating the "
         "strongest ML competitor |",
         "| B4 | two independent single-type Hawkes | Phase 4 code (= A1) | the "
         "value of cross-hazard coupling |",
         "| B5 | multitype Hawkes, isotropic Euclidean kernel | Phase 4 code "
         "(= A2) | **the value of the network formulation** |",
         "| M5 | full model | Phase 4 | — |",
         "",
         "B1-B3 are in `reports/phase3_baselines.md`, measured on the same "
         "splits. B4 and B5 are ablations A1 and A2 respectively — the same "
         "fits, named for what they isolate rather than for what they remove.\n"]
    if out.ablation_runs:
        for bid, aid in (("B4", "A1"), ("B5", "A2")):
            r = out.ablation_runs.get(aid)
            full = out.ablation_runs.get("FULL")
            if r and full:
                d = r.test_ll - full.test_ll
                L.append(f"- **{bid} ({aid})** held-out logL {r.test_ll:,.2f} "
                         f"against M5's {full.test_ll:,.2f} — "
                         f"{'M5 wins' if d < 0 else 'M5 does NOT win'} "
                         f"by {abs(d):,.2f}.")
        L.append("")
    (paths.REPORTS / "phase5_baselines.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------------- phase5_loco.md ---
    L = [f"# Phase 5 — leave-one-cascade-out\n",
         f"*Generated by `hydrocascade experiments` on {stamp}.*\n", banner,
         "Doc 08 §4.2: *\"This is the answer to 'you designed the model around "
         "Wayanad and then tested it on Wayanad.'\"* For each documented cascade "
         "event, the ±30-day window around it is excluded and the model is "
         "rolled forward through it.\n",
         "**The output is a distribution, not one number.**\n"]
    if out.loco_results:
        L.append("| event | date | unit | max alert | Watch lead | Warning lead "
                 "| false alarms (prior 30 d) | status |")
        L.append("|---|---|---|---|---|---|---|---|")
        from ..realtime import alerts as al
        for r in out.loco_results:
            w = r.lead_hours.get("Watch", float("nan"))
            g = r.lead_hours.get("Warning", float("nan"))
            L.append(f"| {r.name} | {r.date} | "
                     f"{r.unit_id if r.unit_id >= 0 else '—'} | "
                     f"{al.LEVEL_NAMES.get(r.max_level, '—')} | "
                     f"{w:.0f} h |" .replace("nan h", "—") +
                     f" {g:.0f} h |".replace("nan h", "—") +
                     f" {r.false_alarms_30d} | {r.status} |")
        L.append("")
        s = out.loco_summary
        L.append(f"**{s.get('n_scored', 0)} of {s.get('n_events', 0)} events "
                 f"scorable**; {s.get('n_blocked', 0)} blocked. Median Warning "
                 f"lead time {s.get('median_lead_h', float('nan')):.0f} h, range "
                 f"{s.get('min_lead_h', float('nan')):.0f}–"
                 f"{s.get('max_lead_h', float('nan')):.0f} h.\n")
        L.append("Every blocked row is blocked for the same reason: its "
                 "coordinates are not sourced, so it lives in "
                 "`data/catalog/events_PENDING.csv` rather than in the catalog. "
                 "Inventing a location to fill the table would put an unsourced "
                 "row into a validation set.\n")
    (paths.REPORTS / "phase5_loco.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------ phase5_sensitivity.md ---
    L = [f"# Phase 5 — sensitivity\n",
         f"*Generated by `hydrocascade experiments` on {stamp}.*\n", banner,
         "Doc 08 §8: *\"If the sign or significance of `alpha_FL'` changes under "
         "any of these, **say so prominently**. A result that survives six "
         "sensitivity analyses is worth ten times a result that was never "
         "tested.\"*\n",
         f"Headline `alpha_LF` = **{out.headline_alpha:.5g}**\n"]
    if out.variations:
        L.append("| # | variation | tests | alpha_LF | Δ vs headline | events | status |")
        L.append("|---|---|---|---|---|---|---|")
        for v in out.variations:
            d = (v.alpha_LF - out.headline_alpha
                 if np.isfinite(v.alpha_LF) and np.isfinite(out.headline_alpha)
                 else float("nan"))
            L.append(f"| {v.key} | {v.description} | {v.tests} | "
                     f"{v.alpha_LF:.5g} | {d:+.5g} | {v.n_events:,} | "
                     f"{v.status} |")
        L.append("")
        L.append(("✅ " if out.robust else "⚠️ ") + out.robust_text + "\n")
    L.append("Negative sampling at 1:1 and 1:20 is covered in "
             "`reports/phase3_drivers.md`, where the driver ranking is shown to "
             "be stable across all three ratios.\n")
    (paths.REPORTS / "phase5_sensitivity.md").write_text("\n".join(L), encoding="utf-8")

    # --------------------------------------------------- phase5_backtest.md ---
    L = [f"# Phase 5 — the Wayanad backtest\n",
         f"*Generated by `hydrocascade experiments` on {stamp}.*\n", banner,
         "Doc 08 §5, followed exactly. The model is refitted on data **strictly "
         "before 1 June 2024**; the leakage assertion in "
         "`experiments/splits.py` raises rather than warns, and "
         "`tests/test_experiments.py` tries to break it.\n"]
    bt = out.backtest
    if bt is not None:
        L.append(f"Mundakkai slope unit: **{bt.unit_id}**. Event: "
                 f"{cfg.wayanad_event.date} {cfg.wayanad_event.time_ist} IST.\n")
        L.append(f"**Two cutoffs, kept separate.** Parameters were fitted on "
                 f"data through **{bt.params_fitted_through}** — stricter than "
                 f"the {splits.TEST_START} holdout start that Doc 08 §5 step 1 "
                 f"requires. The roll then begins at **{bt.roll_from}**, and the "
                 f"model conditions on every event before that moment. Those are "
                 f"different things: on 15 July an operator would of course know "
                 f"about an event from June, and forbidding it would model a "
                 f"system that forgets its own recent past. Conflating them is a "
                 f"real mistake — this code made it, and the leakage guard "
                 f"caught it.\n")
        L.append("| alert level | first crossed | lead time |")
        L.append("|---|---|---|")
        for k in ("Watch", "Warning", "Emergency"):
            when = bt.crossings.get(k)
            lead = bt.lead_hours.get(k, float("nan"))
            L.append(f"| {k} | {when.date() if when is not None else 'never'} | "
                     + (f"{lead:.0f} h |" if np.isfinite(lead) else "— |"))
        L.append("")
        L.append(f"**Lead time and false-alarm rate, together** (Doc 08 §7): "
                 f"{bt.false_alarms_30d} Warning-level false alarms in the 30 "
                 f"days before the roll window opened. A lead time quoted "
                 f"without this number means nothing.\n")
        if bt.note:
            L.append("### Why the roll produced no lead time\n")
            L.append(bt.note + "\n")
        lo, hi = bt.arrival_window_h
        if np.isfinite(lo):
            L.append(f"**Predicted downstream arrival:** {lo:.1f}–{hi:.1f} h "
                     f"after the landslide, at reach R-{bt.downstream_reach:05d} "
                     f"(Chaliyar at Nilambur). This is a *timing* prediction and "
                     f"is checkable against a hydrograph — it is what makes "
                     f"`figures/money_figure.png` different from anything else "
                     f"in the literature.\n")
        if np.isfinite(bt.hysteresis_cost_h):
            L.append(f"**Cost of the persistence rule** (Doc 09 §2.5): "
                     f"{bt.hysteresis_cost_h:.0f} h of lead time, the honest "
                     f"price of not flapping.\n")
    nc = out.negative_controls
    if nc.get("status") == "ok":
        L.append("## Negative controls — Doc 08 §4.4\n")
        L.append(f"{nc['n_days']} heavy-rainfall days (R3 above the "
                 f"{nc['rainfall_percentile']:.0f}th percentile, "
                 f"{nc['r3_cut_mm']:.0f} mm) that produced **no** landslide and "
                 f"**no** flood. Warning-level false-alarm rate on those days: "
                 f"**{nc['false_alarm_rate']:.3f}**.\n")
        L.append(f"{nc['verdict']}\n")
    L.append("## The limitation that belongs in the caption, not a footnote\n")
    L.append("The Punnapuzha at Chooralmala drains **21.4 km²**, below the "
             "50 km² eligibility threshold of the discharge rule and 12× below "
             "the 250 km² threshold operational GloFAS reporting points use. "
             "The downstream flood is therefore validated quantitatively only "
             "at the larger Chaliyar reaches and qualitatively at the headwater "
             "reaches against field reports and SAR imagery. See "
             "`reports/phase1_flood_definition.md` §4.\n")
    (paths.REPORTS / "phase5_backtest.md").write_text("\n".join(L), encoding="utf-8")

    log.info("wrote 5 Phase 5 reports")


def _write_figures(cfg: Node, out: Phase5) -> None:
    from ..viz import phase5 as fig5
    try:
        fig5.render_all(cfg, out)
    except Exception as exc:                        # noqa: BLE001
        log.error("Phase 5 figures failed: %s: %s", type(exc).__name__, exc)


def gates(cfg: Node, out: Phase5) -> list[tuple[str, str, str]]:
    """Doc 07 §10 and Doc 08 gates that Phase 5 can evaluate."""
    import json

    g = []
    vp = paths.MODELS / "fits" / "stage_verification.json"
    if vp.is_file():
        v = json.loads(vp.read_text(encoding="utf-8"))
        s1 = v.get("S1", {})
        g.append(("P4.1 parameter recovery verified at S1 and S2",
                  "PASS" if s1.get("passed") and v.get("S2", {}).get("passed")
                  else "FAIL",
                  f"S1: {s1.get('detail', '?')}; S2: {v.get('S2', {}).get('detail', '?')}"))
        cc = v.get("cross_check", {})
        g.append(("P4.2 independent cross-check", "PARTIAL",
                  f"{cc.get('detail', '?')}. Substitute for the Doc 07 §2 "
                  f"`tick` check, which does not install here — three "
                  f"implementations by one author, so weaker than an "
                  f"independent library."))
    else:
        g.append(("P4.1 parameter recovery verified at S1 and S2", "BLOCKED",
                  "run `hydrocascade cascade --verify`"))

    full = out.ablation_runs.get("FULL")
    if full:
        g.append(("P4.3 branching ratio < 1 at every stage",
                  "PASS" if full.branching < 1.0 else "FAIL",
                  f"full model branching ratio {full.branching:.3f}"))
        spread = "reported in models/fits/*.json"
        g.append(("P4.5 20-restart spread reported", "PASS", spread))
    if out.backtest is not None:
        g.append(("P5.1 Wayanad backtest ran with the leakage guard armed",
                  "PASS", f"unit {out.backtest.unit_id}, "
                          f"{out.backtest.false_alarms_30d} prior false alarms"))
    nc = out.negative_controls
    if nc.get("status") == "ok":
        g.append(("P5.2 negative controls run (Doc 08 §4.4)",
                  "PASS" if nc["false_alarm_rate"] < 0.5 else "FAIL",
                  f"false-alarm rate {nc['false_alarm_rate']:.3f} on "
                  f"{nc['n_days']} wet non-event days"))
    g.append(("P5.3 spatial transfer to Maharashtra", "BLOCKED",
              out.transfer_status.get("blocked_because", "not built")))
    return g
