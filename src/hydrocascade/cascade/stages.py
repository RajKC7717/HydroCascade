"""Stages S3-S5 on the real catalog, run in order — Doc 07 §1.

    *Each stage is a git tag. Each stage's results are recorded before the
    next begins.*

S3 is first contact with the real data: two hazard types, isotropic kernels,
no network. S4 replaces the flood-side spatial kernel with the hybrid support —
reachability, the runout gate, dilution. S5 adds the travel-time shift and the
mechanism mixture.

Each stage must beat the previous one on **held-out** log-likelihood, or the
reason must be understood and written down. Doc 07 §11 is explicit that falling
back is survivable: *"Falling back to S3 with an honest explanation still leaves
you with a working multitype spatio-temporal Hawkes model for Kerala hazards,
which nobody has published. Do not treat S5 as all-or-nothing."*
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import fit, kernels, model

log = get("cascade.stages")

#: Doc 07 §1. Each entry switches on exactly what that stage adds.
STAGE_SPECS = {
    "S3": model.ModelSpec(stage="S3", network=False, travel_time=False,
                          runout_gate=False, dilution=False, mixture_version=1),
    "S4": model.ModelSpec(stage="S4", network=True, travel_time=False,
                          runout_gate=True, dilution=True, mixture_version=1),
    "S5": model.ModelSpec(stage="S5", network=True, travel_time=True,
                          runout_gate=True, dilution=True, mixture_version=2),
}


@dataclass
class StageResult:
    stage: str
    params: dict
    train_ll: float
    test_ll: float
    branching: float
    record: fit.FitRecord
    trace: list[float] = field(default_factory=list)
    restarts: list[float] = field(default_factory=list)
    celerity_ok: bool | None = None
    celerity_note: str = ""
    gate_passes_wayanad: bool | None = None
    gate_note: str = ""
    note: str = ""


_HELD_OUT_CACHE: dict[str, model.CascadeData] = {}


def _held_out_ll(params: dict, cfg: Node, spec: model.ModelSpec,
                 test_start: str, test_end: str) -> float:
    """Log-likelihood on the untouched test window, at fixed parameters.

    The assembled validation catalog is cached per process. It depends only on
    the catalog and the window, never on the parameters or the spec, yet the
    experiment suite scores sixteen-odd fits against it — and rebuilding it each
    time reloads the full slope-unit geometry, the network and the rainfall cube.
    On an 8 GB machine that churn was enough to get the whole run killed by the
    memory reaper, so it is built once.
    """
    import torch

    d = _HELD_OUT_CACHE.get(test_end)
    if d is None:
        try:
            d = model.assemble(cfg, end=test_end)
        except Exception as exc:                    # noqa: BLE001
            log.warning("held-out assembly failed: %s", exc)
            return float("nan")
        _HELD_OUT_CACHE[test_end] = d
    obj = model.make_objective(d, spec)
    q = {k: torch.tensor(float(v), dtype=torch.float64) for k, v in params.items()}
    try:
        return float(obj(q).detach())
    except Exception as exc:                        # noqa: BLE001
        log.warning("held-out evaluation failed: %s", exc)
        return float("nan")


def run_stage(cfg: Node, stage: str, data: model.CascadeData, *,
              n_restarts: int = 8, steps: int = 400, seed: int = 0
              ) -> StageResult:
    """Fit one stage with the Doc 07 §7 protocol."""
    import torch

    spec = STAGE_SPECS[stage]
    obj = model.make_objective(data, spec)
    bounds = model.bounds_for(spec, data)
    rng_sampler = lambda rng: model.initial_params(spec, data, rng)   # noqa: E731

    log.info("-" * 70)
    log.info("stage %s — %s", stage, spec.label())
    t0 = time.perf_counter()
    best, val, values, trace = fit.multi_start(
        obj, rng_sampler, n_restarts=n_restarts, seed=seed, bounds=bounds,
        steps=steps, lr=0.08)
    wall = time.perf_counter() - t0

    br = model.branching(best, data, spec)
    intervals = fit.laplace_intervals(obj, best)

    # --- Doc 07 §10 gate 7: is the fitted v0 physically plausible? ----------
    cel_ok, cel_note = None, ""
    if spec.travel_time and "v0" in best:
        from ..spatial.travel_time import TravelTime
        cel_ok, cel_note = kernels.check_physical_celerity(
            best["v0"], TravelTime().reach_slope)
        (log.info if cel_ok else log.warning)("celerity: %s", cel_note)

    # --- Doc 07 §10 gate 6: does the Wayanad event pass the runout gate? ----
    gate_ok, gate_note = None, ""
    if spec.runout_gate and "gate_c" in best:
        gate_ok, gate_note = _wayanad_gate_check(cfg, best, data)
        (log.info if gate_ok else log.error)("runout gate: %s", gate_note)

    rec = fit.record(
        f"{stage}_{spec.label().replace(' ', '')}", best, val, data.n_events,
        branching=br, seed=seed, restarts=values, intervals=intervals,
        wall=wall, note=f"{spec.label()}; {cel_note}; {gate_note}")

    return StageResult(stage=stage, params=best, train_ll=val,
                       test_ll=float("nan"), branching=br, record=rec,
                       trace=trace, restarts=values, celerity_ok=cel_ok,
                       celerity_note=cel_note, gate_passes_wayanad=gate_ok,
                       gate_note=gate_note)


def _wayanad_gate_check(cfg: Node, params: dict,
                        data: model.CascadeData) -> tuple[bool, str]:
    """Doc 07 §5: *check explicitly that the Wayanad event passes the gate.*

    *"If it does not, the gate parameterisation is wrong."* The Mundakkai crown
    sits 0.68 km from its injection reach; a gate that excludes the one event
    the project is built around has been fitted to the wrong thing.
    """
    import torch

    from ..catalog import schema
    from ..catalog.build import EVENTS_PATH

    ev = schema.read(EVENTS_PATH)
    real = ev[(ev["hazard_type"] == "L") & (ev["source"] != "SYNTHETIC")]
    if real.empty:
        return False, "no real landslide in the catalog to test the gate on"

    # Doc 07 §5 asks a specific question: *with the fitted c and gamma, would a
    # landslide of Mundakkai's mapped area at its d_slope pass the gate?*
    #
    # That question does not depend on whether Mundakkai is inside this fit's
    # window — and it never is, because the training cutoff is 2022 and the
    # event is 2024. Reading its geometry from the catalog and the slope units
    # rather than from the fitted `data` is what makes the check test the event
    # the project is built around, instead of a median stand-in for it.
    which, d_slope, mark = "", float("nan"), 0.0
    try:
        from ..spatial.network import Network
        from ..spatial.projection import cell_level
        from ..spatial.terrain import Routing

        rt = Routing()
        proj = cell_level(rt, Network())
        lm = cfg.landmarks.mundakkai_crown
        crown = rt.index_of_lonlat(float(lm["lon"]), float(lm["lat"]))
        # The crown cell, not the slope unit's representative cell. A landslide
        # starts at its crown, and that is where the runout begins — the unit's
        # representative point sits further from the channel and would make the
        # gate look harder to pass than it is.
        d_slope = float(proj["d_slope_m"][crown])
        which = "Mundakkai crown"
        # The mark is centred within its type, and the Mundakkai FIR carries no
        # mapped area, so its centred mark is 0 — the neutral case, which is the
        # conservative one to test the gate at.
        mark = 0.0
    except Exception as exc:                        # noqa: BLE001
        log.debug("crown geometry unavailable: %s", exc)

    if not np.isfinite(d_slope):
        idx = np.flatnonzero((data.kind == model.L) & (data.d_slope_m > 0))
        if idx.size == 0:
            return False, "no landslide carries a d_slope to test"
        d_slope = float(np.median(data.d_slope_m[idx]))
        mark = float(np.median(data.mark[idx]))
        which = "median landslide (Mundakkai's geometry unavailable)"

    passes = float(kernels.runout_gate(
        torch.tensor(d_slope, dtype=torch.float64),
        torch.tensor(mark, dtype=torch.float64),
        torch.tensor(float(params["gate_c"]), dtype=torch.float64),
        torch.tensor(float(params["gate_gamma"]), dtype=torch.float64),
        hard=True))
    reach = params["gate_c"] * np.exp(params["gate_gamma"] * mark)
    ok = passes > 0.5
    return ok, (f"{which}: d_slope {d_slope:.0f} m vs fitted runout reach "
                f"{reach:.0f} m (c = {params['gate_c']:.0f} m, "
                f"gamma = {params['gate_gamma']:.3f}) — "
                f"{'PASSES' if ok else 'EXCLUDED, so the gate parameterisation is wrong'}")


def run(cfg: Node, *, stages=("S3", "S4", "S5"), n_restarts: int = 8,
        steps: int = 400, seed: int | None = None) -> dict[str, StageResult]:
    """Run the stages in order, recording each before the next begins."""
    seed = int(cfg.project.seed) if seed is None else int(seed)

    from ..experiments import splits
    train_end = splits.TRAIN_END

    log.info("assembling the training catalog (through %s)", train_end)
    data = model.assemble(cfg, end=train_end)

    results: dict[str, StageResult] = {}
    traces: dict[str, list[float]] = {}
    for stage in stages:
        r = run_stage(cfg, stage, data, n_restarts=n_restarts, steps=steps,
                      seed=seed)
        r.test_ll = _held_out_ll(r.params, cfg, STAGE_SPECS[stage],
                                 splits.VALID_START, splits.VALID_END)
        results[stage] = r
        traces[stage] = r.trace
        log.info("%s: train logL %.2f, validation logL %.2f, branching %.3f",
                 stage, r.train_ll, r.test_ll, r.branching)

    try:
        from . import diagnostics
        diagnostics.convergence_plot(traces)
    except Exception as exc:                        # noqa: BLE001
        log.warning("convergence plot failed: %s", exc)

    _compare(results)
    _write_report(cfg, results)
    return results


ADDS = {
    "S3": "two hazard types, isotropic kernels, real data",
    "S4": "the hybrid support: reachability, runout gate, dilution",
    "S5": "the travel-time wave and the mechanism mixture",
}


def _diagnose(stage: str, r: StageResult, prev: StageResult) -> str:
    """Why a stage failed to beat the previous one — Doc 07 §1 and §11.

    Doc 07 requires each stage to beat the last *or the reason to be understood*.
    An unexplained regression is what stalls a project in week thirteen, so the
    plausible causes are checked here rather than left to whoever reads the log.
    """
    reasons = []

    if r.celerity_ok is False and "v0" in r.params:
        lo, _hi = kernels.V0_BOUNDS
        if abs(r.params["v0"] - lo) / lo < 0.02:
            reasons.append(
                "`v0` is pinned at its lower bound, meaning the optimiser wants "
                "an even longer arrival delay than the physical celerity range "
                "allows. That is what it does when the data contains **no "
                "propagation structure to find**: the best available fit is to "
                "push the shifted kernel as far out of the way as possible.")
        else:
            reasons.append(f"the fitted celerity is unphysical — {r.celerity_note}")

    if stage == "S5":
        reasons.append(
            "Check the discharge source before blaming the model. The synthetic "
            "stand-in generates each reach's hydrograph independently — a "
            "per-reach annual factor and its own linear reservoir — so **there "
            "is no wave travelling between reaches in it at all**. A "
            "travel-time shift fitted against data with no travel time in it "
            "can only hurt, and this comparison becomes informative only once "
            "`data_mode.glofas` is `real`.")

    if r.restarts and len(r.restarts) > 1:
        spread = float(np.max(r.restarts) - np.min(r.restarts))
        if spread > 0.001 * abs(r.train_ll):
            reasons.append(
                f"The restart spread is {spread:.1f} log-likelihood units, so "
                f"the extra parameters made the surface multimodal and the "
                f"optimiser may have landed in a worse mode. More restarts "
                f"would settle whether that is the whole story.")

    if not reasons:
        reasons.append(
            "No single cause identified. Check the restart spread in the fit "
            "record and the convergence trace before concluding the component "
            "does not help.")
    return " ".join(reasons)


def _compare(results: dict[str, StageResult]) -> None:
    """Doc 07 §1: each stage must beat the previous, or the reason is recorded."""
    order = [s for s in ("S3", "S4", "S5") if s in results]
    for a, b in zip(order, order[1:]):
        ra, rb = results[a], results[b]
        d_train = rb.train_ll - ra.train_ll
        d_test = rb.test_ll - ra.test_ll
        beats = bool(np.isfinite(d_test) and d_test > 0)
        (log.info if beats else log.warning)(
            "%s %s %s on held-out log-likelihood (train %+.2f, validation %+.2f)",
            b, "beats" if beats else "does NOT beat", a, d_train, d_test)
        if not beats:
            rb.note = _diagnose(b, rb, ra)
            log.warning("  why: %s", rb.note.replace("**", ""))


def _write_report(cfg: Node, results: dict[str, StageResult]) -> None:
    """``reports/phase4_stages.md`` — Doc 07 §1 records each stage before the next."""
    from datetime import date

    from ..ingest import adapters

    L: list[str] = []
    A = L.append

    A("# Phase 4 — the cascade model, stage by stage")
    A("")
    A(f"*Generated by `hydrocascade cascade` on {date.today().isoformat()}.*")
    A("")

    synth = adapters.synthetic_sources(cfg)
    if synth:
        A("> ### ⚠ Fitted on synthetic data")
        A(">")
        A("> `data_mode` is `synthetic` for " + ", ".join(f"`{s}`" for s in synth)
          + ". Every parameter below")
        A("> describes the stand-in generators, not Kerala. They show that the "
          "fitting machinery")
        A("> works and that its diagnostics fire; they are not results.")
        A("")

    A("Doc 07 §1: *never add a component until the previous one has been "
      "verified on data where you already know the answer.* Each stage must beat "
      "the previous on **held-out** log-likelihood, or the reason is recorded "
      "here rather than left in a log.")
    A("")
    A("| stage | what it adds | train logL | validation logL | branching | verdict |")
    A("|---|---|---|---|---|---|")

    order = [s for s in ("S3", "S4", "S5") if s in results]
    for i, k in enumerate(order):
        r = results[k]
        if i == 0:
            verdict = "baseline"
        else:
            d = r.test_ll - results[order[i - 1]].test_ll
            verdict = (f"**beats {order[i-1]}** ({d:+,.0f})" if d > 0
                       else f"**does not beat {order[i-1]}** ({d:+,.0f})")
        A(f"| {k} | {ADDS.get(k, '')} | {r.train_ll:,.2f} | {r.test_ll:,.2f} "
          f"| {r.branching:.3f} | {verdict} |")
    A("")

    for k in order:
        r = results[k]
        if r.note:
            A(f"### Why {k} did not beat the previous stage")
            A("")
            A(r.note)
            A("")
            A("Doc 07 §11 plans for exactly this: *\"Falling back to S3 with an "
              "honest explanation still leaves you with a working multitype "
              "spatio-temporal Hawkes model for Kerala hazards, which nobody has "
              "published. Do not treat S5 as all-or-nothing.\"*")
            A("")
        if r.celerity_note:
            A(f"**{k} — celerity check** (Doc 07 §10 gate 7): {r.celerity_note}")
            A("")
        if r.gate_note:
            A(f"**{k} — runout gate** (Doc 07 §10 gate 6): {r.gate_note}")
            A("")

    A("## Stability, and why alpha is bounded by the network")
    A("")
    A("Doc 07 §7 rule 7 makes a branching ratio at or above 1 an automatic "
      "invalid fit, whatever the likelihood says. The excitation parameters are "
      "therefore bounded **by the size of the set each source can reach**.")
    A("")
    A("`alpha_LF` is not a count of offspring — it is the excitation applied to "
      "*each* reach downstream, and a landslide in this basin reaches a few "
      "hundred of them. So the quantity that must stay below 1 is "
      "`alpha_LF × (mean downstream set size)`, and the bound on `alpha_LF` "
      "scales down accordingly. Bounding `alpha_LF` itself at some fixed number "
      "would be far too generous the moment the network switches on at S4 — "
      "which is exactly how the first S4 fit came back at **1.569** and was "
      "rejected as explosive.")
    A("")
    A("A branching ratio sitting exactly at its bound means the data wants more "
      "self-excitation than stationarity permits. That is reported, not fixed by "
      "raising the bound.")
    A("")
    A("Every fit writes a JSON record to `models/fits/` carrying parameters, "
      "intervals, log-likelihood, AIC, seed, git commit and wall-clock time, so "
      "every number here traces back to a specific run (Doc 07 §7 rule 8).")
    A("")

    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "phase4_stages.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote   reports/phase4_stages.md")
