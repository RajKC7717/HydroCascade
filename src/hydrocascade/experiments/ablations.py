"""Ablations A1-A9 and the nested ladder M0-M5 — Doc 08 §2-§3.

Doc 08 §3 calls the nine-row ablation table *"the scientific core of the
paper. It tells a reviewer exactly which of our ideas are real."*

Each ablation removes exactly one component and refits everything else, so the
change in held-out log-likelihood is attributable to that component and nothing
else. Because every component is a flag on :class:`~hydrocascade.cascade.model.ModelSpec`,
an ablation is a spec, not a code branch — there is no second implementation of
the model that could drift away from the first.

**Expect some ablations to show no degradation, and report those too.** Doc 08
§3: *"An idea that turns out not to matter is a finding, and saying so builds
far more credibility than quietly dropping it."*

The ladder M0-M5 is **nested**, so likelihood-ratio tests apply to consecutive
pairs. B1-B3 are not nested and are compared on predictive metrics only — a
likelihood-ratio test between a Hawkes model and XGBoost would be meaningless.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Node
from ..logging_setup import get
from ..cascade import model, stages

log = get("experiments.ablations")

#: Doc 08 §3. Each entry is (id, what it removes, the claim it tests).
ABLATIONS = {
    "A1": ("cross terms (alpha_LF = alpha_FL = 0)",
           "landslides raise flood risk", {"cross": False}),
    "A2": ("network kernel -> isotropic Euclidean",
           "influence follows the drainage network", {"network": False}),
    "A3": ("travel-time shift (tau = 0)",
           "influence arrives as a wave, not instantly", {"travel_time": False}),
    "A4": ("runout connectivity gate",
           "debris must physically reach a channel", {"runout_gate": False}),
    "A5": ("dilution term (psi = 0)",
           "effect weakens as the catchment grows", {"dilution": False}),
    "A6": ("Part-1 background (mu constant in space)",
           "the root-cause study improves the cascade model", {"background": False}),
    "A7": ("rainfall windows",
           "rainfall forcing matters", {"rainfall": False}),
    "A8": ("mark term (eta = 0)",
           "event size matters", {"marks": False}),
    "A9": ("mechanism mixture -> single exponential",
           "mechanisms have distinguishable timescales", {"mixture_version": 1}),
}

#: Doc 08 §2. Nested, so likelihood-ratio tests apply between consecutive rows.
LADDER = {
    "M0": ("constant rate per type",
           dict(self_excite=False, cross=False, network=False, travel_time=False, runout_gate=False,
                dilution=False, background=False, rainfall=False, marks=False,
                mixture_version=1)),
    "M1": ("+ susceptibility background, no time",
           dict(self_excite=False, cross=False, network=False, travel_time=False, runout_gate=False,
                dilution=False, background=True, rainfall=False, marks=False,
                mixture_version=1)),
    "M2": ("+ rainfall windows — the honest baseline",
           dict(self_excite=False, cross=False, network=False, travel_time=False, runout_gate=False,
                dilution=False, background=True, rainfall=True, marks=False,
                mixture_version=1)),
    "M3": ("+ self-excitation only (alpha_LF = alpha_FL = 0)",
           dict(self_excite=True, cross=False, network=False, travel_time=False, runout_gate=False,
                dilution=False, background=True, rainfall=True, marks=True,
                mixture_version=1)),
    "M4": ("+ cross-excitation free",
           dict(self_excite=True, cross=True, network=False, travel_time=False, runout_gate=False,
                dilution=False, background=True, rainfall=True, marks=True,
                mixture_version=1)),
    "M5": ("+ network kernel, travel time, mechanism mixture",
           dict(self_excite=True, cross=True, network=True, travel_time=True, runout_gate=True,
                dilution=True, background=True, rainfall=True, marks=True,
                mixture_version=2)),
}


@dataclass
class Run:
    key: str
    description: str
    spec: model.ModelSpec
    params: dict = field(default_factory=dict)
    train_ll: float = float("nan")
    test_ll: float = float("nan")
    branching: float = float("nan")
    n_params: int = 0
    wall_s: float = 0.0
    experiment_id: str = ""

    @property
    def aic(self) -> float:
        return float(2 * self.n_params - 2 * self.train_ll)


def _fit_one(cfg: Node, key: str, description: str, spec: model.ModelSpec,
             data: model.CascadeData, *, n_restarts: int, steps: int,
             seed: int) -> Run:
    from ..cascade import fit
    from . import register

    obj = model.make_objective(data, spec)
    t0 = time.perf_counter()
    try:
        best, val, values, _ = fit.multi_start(
            obj, lambda rng: model.initial_params(spec, data, rng),
            n_restarts=n_restarts, seed=seed, bounds=model.bounds_for(spec, data),
            steps=steps, lr=0.08)
    except Exception as exc:                        # noqa: BLE001
        log.error("%s failed to fit: %s", key, exc)
        return Run(key, description, spec)
    wall = time.perf_counter() - t0

    r = Run(key, description, spec, params=best, train_ll=val,
            branching=model.branching(best, data, spec),
            n_params=len(best), wall_s=wall)
    r.test_ll = stages._held_out_ll(best, cfg, spec,
                                    splits_valid_start(), splits_valid_end())
    r.experiment_id = register.append(
        description, model_config=spec.label(), data_split="train/validation",
        seed=seed, wall_clock_s=wall, log_likelihood=val, aic=r.aic,
        branching_ratio=r.branching,
        results_file=f"models/fits/{key}.json",
        notes=f"{len(values)} restarts converged")
    log.info("%-3s %-52s train %+12.2f  valid %+12.2f  branching %.3f",
             key, description[:52], r.train_ll, r.test_ll, r.branching)
    # Each fit builds a closure holding every pair tensor; release it before the
    # next one is built so two never coexist on a memory-tight machine.
    del obj
    import gc
    gc.collect()
    return r


def splits_valid_start() -> str:
    from . import splits
    return splits.VALID_START


def splits_valid_end() -> str:
    from . import splits
    return splits.VALID_END


def run_ablations(cfg: Node, data: model.CascadeData, *, n_restarts: int = 4,
                  steps: int = 300, seed: int = 0) -> dict[str, Run]:
    """Full model plus A1-A9, each removing exactly one component."""
    out: dict[str, Run] = {}
    full = model.ModelSpec(stage="S5")
    out["FULL"] = _fit_one(cfg, "FULL", "full model (S5)", full, data,
                           n_restarts=n_restarts, steps=steps, seed=seed)
    for key, (removes, _claim, overrides) in ABLATIONS.items():
        # Start from the full spec and flip exactly one flag. `stage` is set
        # inside the merged dict rather than passed separately, or it arrives
        # twice and ModelSpec rejects it.
        kw = {**full.__dict__, **overrides}
        kw["stage"] = "S5"
        spec = model.ModelSpec(**kw)
        out[key] = _fit_one(cfg, key, f"ablation {key}: remove {removes}", spec,
                            data, n_restarts=n_restarts, steps=steps, seed=seed)
    return out


def run_ladder(cfg: Node, data: model.CascadeData, *, n_restarts: int = 4,
               steps: int = 300, seed: int = 0) -> dict[str, Run]:
    """M0-M5, nested so likelihood-ratio tests apply."""
    out: dict[str, Run] = {}
    for key, (desc, kw) in LADDER.items():
        spec = model.ModelSpec(stage=key, **kw)
        out[key] = _fit_one(cfg, key, desc, spec, data,
                            n_restarts=n_restarts, steps=steps, seed=seed)
    return out


def likelihood_ratio(simpler: Run, richer: Run) -> tuple[float, int, float]:
    """``(statistic, df, p)`` for a nested pair.

    ``2 * (logL_rich - logL_simple)`` is asymptotically chi-squared with degrees
    of freedom equal to the difference in parameter count. The test is only
    valid for **nested** models, which is why the ladder exists separately from
    the baselines: comparing M5 against XGBoost this way would be nonsense.

    A negative statistic means the richer model fitted *worse*, which is
    impossible at the true optimum and therefore evidence that one of the two
    optimisations did not converge. It is reported rather than clipped.
    """
    from scipy import stats

    stat = 2.0 * (richer.train_ll - simpler.train_ll)
    df = max(richer.n_params - simpler.n_params, 1)

    # Swapping the isotropic flood kernel for the network kernel is a change of
    # kernel family, not the addition of a parameter — the isotropic model is
    # not a special case of the network one. The chi-squared reference then does
    # not hold, and M4 -> M5 is exactly that step. Compare it on held-out
    # likelihood and AIC instead, and say why.
    s_net = getattr(simpler.spec, "network", None)
    r_net = getattr(richer.spec, "network", None)
    if s_net is not None and r_net is not None and s_net != r_net:
        return float(stat), df, float("nan")
    if not np.isfinite(stat):
        return float("nan"), df, float("nan")
    if stat < 0:
        log.warning("%s fits worse than the nested %s (statistic %.2f) — an "
                    "optimisation failure, not a model result",
                    richer.key, simpler.key, stat)
        return float(stat), df, float("nan")
    return float(stat), df, float(stats.chi2.sf(stat, df))


def table(runs: dict[str, Run], reference: str = "FULL") -> "pd.DataFrame":
    """The Doc 08 §3 nine-row table: delta held-out logL and delta AIC."""
    import pandas as pd

    ref = runs.get(reference)
    rows = []
    for key, r in runs.items():
        if key == reference:
            continue
        claim = ABLATIONS.get(key, ("", "", {}))[1]
        rows.append({
            "ablation": key,
            "removes": r.description.replace(f"ablation {key}: remove ", ""),
            "tests the claim": claim,
            "delta_train_ll": r.train_ll - ref.train_ll if ref else np.nan,
            "delta_test_ll": r.test_ll - ref.test_ll if ref else np.nan,
            "delta_aic": r.aic - ref.aic if ref else np.nan,
            "branching": r.branching,
        })
    return pd.DataFrame(rows)


def interpret(delta_test_ll: float, tol: float = 2.0) -> str:
    """Turn a delta into a sentence that does not overclaim.

    A component "earns its place" only if removing it makes held-out fit
    clearly worse. Removing it and seeing no change means the data cannot
    distinguish the two models — which is a finding about the data, not a
    verdict that the physics is wrong.
    """
    if not np.isfinite(delta_test_ll):
        return "not evaluable"
    if delta_test_ll < -tol:
        return "**earns its place** — removing it makes held-out fit worse"
    if delta_test_ll > tol:
        return "**hurts** — the model is better without it"
    return "no detectable difference — the data cannot distinguish these"
