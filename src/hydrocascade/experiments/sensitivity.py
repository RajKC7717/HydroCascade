"""Sensitivity analyses — Doc 08 §8, *"required, not optional"*.

    *Repeat the headline result under each variation and report whether the
    conclusion changes. If the sign or significance of ``alpha_FL'`` changes
    under any of these, **say so prominently**. A result that survives six
    sensitivity analyses is worth ten times a result that was never tested.*

The headline result is ``alpha_LF`` — how strongly a landslide raises
downstream flood intensity. Each variation below rebuilds the catalog under a
different definitional choice and refits, so what moves is the *definition*,
not the model.

Four of the six are catalog-level and run here in memory, **never writing over
``events.csv``**: a sensitivity run that mutates the catalog it is testing is
how a variant quietly becomes the headline. Two need a Phase 2 rebuild at a
different spatial resolution; they are reported as pending with the exact
command, not silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Node
from ..logging_setup import get

log = get("experiments.sensitivity")


@dataclass
class Variation:
    key: str
    description: str
    tests: str
    alpha_LF: float = float("nan")
    train_ll: float = float("nan")
    branching: float = float("nan")
    n_events: int = 0
    status: str = "ok"
    note: str = ""


#: Doc 08 §8. The last two need a spatial rebuild and are reported as pending.
VARIATIONS = {
    "S1": ("flood threshold Q5 instead of Q2", "dependence on the flood definition",
           {"threshold_return_period": 5}),
    "S2": ("declustering 5 days instead of 3", "dependence on declustering",
           {"decluster_days": 5}),
    "S3": ("eligibility 250 km2 instead of 50", "dependence on GloFAS reliability limits",
           {"min_contributing_area_km2": 250.0}),
    "S4": ("completeness cutoff A_min applied", "dependence on inventory incompleteness",
           {"_apply_amin": True}),
    "S5": ("slope units 0.1 km2", "dependence on spatial resolution", {"_rebuild": 0.1}),
    "S6": ("slope units 1 km2", "dependence on spatial resolution", {"_rebuild": 1.0}),
}


def _variant_catalog(cfg: Node, overrides: dict) -> tuple[pd.DataFrame, Node]:
    """Rebuild the flood half of the catalog under a different definition."""
    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH
    from ..config import Node as _Node
    from ..ingest import adapters, floods
    from ..spatial.network import Network

    base = schema.read(EVENTS_PATH)
    raw = cfg.to_dict()
    raw["floods"] = {**raw["floods"],
                     **{k: v for k, v in overrides.items() if not k.startswith("_")}}
    vcfg = _Node(raw)

    if not any(k in overrides for k in
               ("threshold_return_period", "decluster_days",
                "min_contributing_area_km2")):
        return base, vcfg

    net = Network()
    reaches = net.gdf[["reach_id", "area_dn_km2"]].copy()
    areas = reaches.set_index("reach_id")["area_dn_km2"]
    disch = adapters.discharge(vcfg, reaches)
    ev = floods.detect_for_all_reaches(disch, areas, vcfg)

    start, end = str(cfg.period.start), str(cfg.period.end)
    ev = ev[(ev["date"] >= start) & (ev["date"] <= end)]

    land = base[base["hazard_type"] == "L"].copy()
    flood = pd.DataFrame({
        "event_id": [f"F-{i:05d}" for i in range(len(ev))],
        "hazard_type": "F",
        "date": ev["date"].to_numpy(),
        "time_known": "N", "time": "",
        "lat": np.nan, "lon": np.nan,
        "geometry_type": "reach", "area_m2": np.nan,
        "elevation_m": np.nan,
        "mark_raw": ev["q_peak"].to_numpy(),
        "mark_value": ev["mark_value"].to_numpy(),
        "severity_class": ev["severity_class"].to_numpy(),
        "spatial_unit_id": ev["reach_id"].astype(str).to_numpy(),
        "source": "SYNTHETIC", "source_url": "synthetic://variant",
        "source_date_accessed": "", "accessed_by": "sensitivity",
        "confidence": "low", "duplicate_of": "", "notes": "sensitivity variant",
    })
    return pd.concat([land, flood], ignore_index=True), vcfg


def run(cfg: Node, spec, *, n_restarts: int = 3, steps: int = 250,
        seed: int = 0, headline_alpha: float = float("nan")) -> list[Variation]:
    """Refit the headline model under each definitional variation."""
    from ..cascade import fit, model
    from . import register, splits

    out: list[Variation] = []
    for key, (desc, tests, overrides) in VARIATIONS.items():
        v = Variation(key=key, description=desc, tests=tests)

        if "_rebuild" in overrides:
            v.status = "PENDING"
            v.note = (
                f"needs a Phase 2 rebuild at "
                f"{overrides['_rebuild']} km2 median slope-unit area: set "
                f"`spatial.slope_units.target_median_area_km2: "
                f"{overrides['_rebuild']}` and re-run `hydrocascade units "
                f"projection traveltime --force`, then `hydrocascade cascade`. "
                f"Not run here because it rebuilds every downstream artefact.")
            log.warning("%s %s — PENDING (%s)", key, desc, v.note[:60])
            out.append(v)
            continue

        try:
            events, vcfg = _variant_catalog(cfg, overrides)
            if overrides.get("_apply_amin"):
                from ..catalog import completeness
                areas = pd.to_numeric(
                    events.loc[events["hazard_type"] == "L", "area_m2"],
                    errors="coerce").dropna()
                comp = completeness.find_rollover(areas.to_numpy())
                events = completeness.apply_cutoff(events, comp.a_min_m2)
                v.note = f"A_min = {comp.a_min_m2:,.0f} m2 applied"

            data = model.assemble(vcfg, end=splits.TRAIN_END, events=events)
            v.n_events = data.n_events
            obj = model.make_objective(data, spec)
            best, val, values, _ = fit.multi_start(
                obj, lambda rng: model.initial_params(spec, data, rng),
                n_restarts=n_restarts, seed=seed,
                bounds=model.bounds_for(spec, data), steps=steps, lr=0.08)
            v.alpha_LF = float(best.get("a_LF", float("nan")))
            v.train_ll = float(val)
            v.branching = model.branching(best, data, spec)
            register.append(f"sensitivity {key}: {desc}",
                            model_config=spec.label(), data_split="train",
                            seed=seed, log_likelihood=val,
                            branching_ratio=v.branching,
                            notes=f"alpha_LF = {v.alpha_LF:.5g}")
            log.info("%s %-44s alpha_LF %.5g (headline %.5g), %d events",
                     key, desc, v.alpha_LF, headline_alpha, v.n_events)
        except Exception as exc:                    # noqa: BLE001
            v.status = "FAILED"
            v.note = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s", key, exc)
        out.append(v)
    return out


def verdict(variations: list[Variation], headline: float) -> tuple[bool, str]:
    """Does the conclusion survive? Doc 08 §8's actual question."""
    scored = [v for v in variations
              if v.status == "ok" and np.isfinite(v.alpha_LF)]
    if not scored or not np.isfinite(headline):
        return False, ("no variation produced a comparable estimate, so the "
                       "headline has not been tested for robustness")

    signs = {np.sign(v.alpha_LF) for v in scored} | {np.sign(headline)}
    same_sign = len(signs) == 1
    vals = np.array([v.alpha_LF for v in scored])
    spread = float(np.max(vals) - np.min(vals))
    rel = spread / abs(headline) if headline else float("inf")

    if same_sign and rel < 1.0:
        return True, (f"alpha_LF keeps the same sign across all {len(scored)} "
                      f"variations that ran, with a spread of {spread:.4g} "
                      f"({100*rel:.0f}% of the headline {headline:.4g}). The "
                      f"conclusion does not depend on the definitional choices "
                      f"tested.")
    if not same_sign:
        return False, (f"**alpha_LF changes sign** across the variations "
                       f"({', '.join(f'{v.key} {v.alpha_LF:+.4g}' for v in scored)}). "
                       f"Doc 08 §8 requires this to be said prominently: the "
                       f"result depends on a definitional choice and must be "
                       f"reported as such.")
    return False, (f"alpha_LF keeps its sign but varies by {spread:.4g}, "
                   f"{100*rel:.0f}% of the headline — the magnitude is not "
                   f"robust to the flood definition even though the direction is.")
