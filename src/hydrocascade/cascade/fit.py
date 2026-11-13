"""The fitting protocol — Doc 07 §7, applied at every stage from S3 onward.

1. Adam for 2,000 steps with a decayed learning rate, then L-BFGS to polish.
   Adam is robust to a bad start and sloppy near the optimum; L-BFGS is the
   opposite. Using both is not indecision, it is the standard remedy.
2. **20 random restarts minimum**, with the spread of final log-likelihoods
   reported. A point-process likelihood is frequently multimodal, and one
   restart tells you nothing about which mode you landed in.
3. Gradient clipping, because an early step can put a parameter somewhere the
   exponential overflows.
4. Early stopping on validation log-likelihood.
5. ``float64`` throughout (NFR-09).
6. The optimiser trajectory is kept for the convergence plot.
7. **After every fit, the branching ratio.** At or above 1 the fit is invalid
   regardless of its likelihood, and :class:`FitRecord` refuses to call it
   converged.
8. Every fit writes a JSON record — parameters, intervals, log-likelihood, AIC,
   seed, git commit, wall-clock time. Doc 07: *"that last item makes every
   number in the paper traceable to a specific run."*

**Three parameterisations, chosen by what the parameter means.**

*Positive* — rates, timescales, widths. Stored as a logarithm and exponentiated
inside the likelihood, so the optimiser works unconstrained and no step can
propose a negative ``theta``.

*Bounded* — anything with a physical range, such as the Doc 07 §6.2 timescale
bounds. A scaled sigmoid rather than a clamp: a clamp puts a hard zero in the
gradient and the optimiser silently stops moving that parameter.

*Real* — regression coefficients and mixture logits. These are **signed**, and
forcing them positive is not a harmless simplification: a driver that *lowers*
risk becomes inexpressible, and the fit answers by pushing the coefficient to
``exp(-40)``, which looks like "this driver does not matter" and is not the
same statement at all. :data:`REAL_VALUED` names the prefixes that get identity
parameterisation.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from .. import paths
from ..logging_setup import get

log = get("cascade.fit")

FITS_DIR = paths.MODELS / "fits"

#: Parameter-name prefixes that are genuinely signed and must NOT be forced
#: positive: regression coefficients on the drivers, and the softmax logits of
#: the mechanism mixture.
REAL_VALUED = ("beta_", "w_logit")


def is_real_valued(name: str) -> bool:
    return name.startswith(REAL_VALUED)


def git_commit() -> str:
    """Short commit hash, or a marker that the tree is not under git."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=paths.ROOT, capture_output=True, text=True,
                             timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:                               # noqa: BLE001
        pass
    return "not-a-git-repo"


@dataclass
class FitRecord:
    """One fit, fully traceable (Doc 07 §7 rule 8)."""
    name: str
    params: dict[str, float]
    log_likelihood: float
    n_events: int
    n_params: int
    branching_ratio: float
    seed: int
    n_restarts: int
    restart_spread: list[float] = field(default_factory=list)
    intervals: dict[str, list[float]] = field(default_factory=dict)
    wall_clock_s: float = 0.0
    git_commit: str = ""
    timestamp: str = ""
    converged: bool = False
    note: str = ""

    @property
    def aic(self) -> float:
        return float(2 * self.n_params - 2 * self.log_likelihood)

    @property
    def bic(self) -> float:
        n = max(self.n_events, 1)
        return float(self.n_params * np.log(n) - 2 * self.log_likelihood)

    def valid(self) -> tuple[bool, str]:
        """Doc 07 §7 rule 7 and §10 gate 3."""
        if not np.isfinite(self.log_likelihood):
            return False, "log-likelihood is not finite"
        if self.branching_ratio >= 1.0:
            return False, (f"branching ratio {self.branching_ratio:.3f} >= 1 — "
                           f"the fitted process is explosive, so the fit is "
                           f"invalid regardless of its likelihood")
        return True, "ok"

    def save(self) -> None:
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.update(aic=self.aic, bic=self.bic)
        ok, why = self.valid()
        d.update(valid=ok, validity_note=why)
        path = FITS_DIR / f"{self.name}.json"
        path.write_text(json.dumps(d, indent=2, default=float), encoding="utf-8")
        log.info("fit record -> models/fits/%s", path.name)

    def summary(self) -> str:
        ok, why = self.valid()
        return (f"{self.name}: logL {self.log_likelihood:,.2f}, AIC {self.aic:,.1f}, "
                f"branching {self.branching_ratio:.3f}, "
                f"{'VALID' if ok else 'INVALID — ' + why}")


# ----------------------------------------------------------------- engine ---

def optimise(objective, init: dict[str, float], *, bounds: dict = None,
             steps: int = 2000, lr: float = 0.05, seed: int = 0,
             clip: float = 5.0, verbose: bool = False):
    """Maximise ``objective(params) -> scalar tensor`` over positive parameters.

    ``init`` maps a name to a positive starting value. ``bounds`` optionally
    maps a name to ``(lo, hi)``, enforced by a scaled sigmoid rather than by
    clamping — a clamp puts a hard zero in the gradient and the optimiser
    silently stops moving that parameter.

    Returns ``(best_params, best_value, trace)``.
    """
    import torch

    torch.manual_seed(seed)
    bounds = bounds or {}
    names = list(init)

    raw = {}
    for k in names:
        v = float(init[k])
        if k in bounds:
            lo, hi = bounds[k]
            v = min(max(v, lo + 1e-9), hi - 1e-9)
            z = np.log((v - lo) / (hi - v))         # inverse scaled sigmoid
        elif is_real_valued(k):
            z = v                                   # identity: signed parameter
        else:
            z = np.log(max(v, 1e-12))               # inverse exp
        raw[k] = torch.tensor(z, dtype=torch.float64, requires_grad=True)

    def unpack():
        out = {}
        for k in names:
            if k in bounds:
                lo, hi = bounds[k]
                out[k] = lo + (hi - lo) * torch.sigmoid(raw[k])
            elif is_real_valued(k):
                out[k] = raw[k]
            else:
                out[k] = torch.exp(raw[k])
        return out

    params = list(raw.values())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    trace: list[float] = []
    best_val, best = -np.inf, None

    for step in range(steps):
        opt.zero_grad()
        val = objective(unpack())
        loss = -val
        if not torch.isfinite(loss):
            log.warning("non-finite objective at step %d; stopping this start", step)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        sched.step()

        v = float(val.detach())
        trace.append(v)
        if v > best_val:
            best_val = v
            best = {k: float(t.detach()) for k, t in unpack().items()}
        if verbose and step % 400 == 0:
            log.debug("  step %4d  logL %.3f", step, v)

    # --- L-BFGS polish -------------------------------------------------------
    try:
        # 50, not 150: each L-BFGS iteration costs several closure evaluations
        # under strong-Wolfe line search, so it can quietly dominate the whole
        # fit. Adam has already done the exploring; this is a polish.
        lbfgs = torch.optim.LBFGS(params, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            loss = -objective(unpack())
            if torch.isfinite(loss):
                loss.backward()
            return loss

        lbfgs.step(closure)
        v = float(objective(unpack()).detach())
        if np.isfinite(v) and v > best_val:
            best_val = v
            best = {k: float(t.detach()) for k, t in unpack().items()}
            trace.append(v)
    except Exception as exc:                        # noqa: BLE001
        log.debug("L-BFGS polish skipped: %s", exc)

    return best, best_val, trace


def multi_start(objective, sampler, *, n_restarts: int = 20, seed: int = 0,
                bounds: dict = None, steps: int = 2000, lr: float = 0.05):
    """Doc 07 §7 rule 2. Returns ``(best_params, best_value, all_values, trace)``.

    The spread of final log-likelihoods across restarts is reported, not
    averaged away: if the restarts disagree the likelihood is multimodal, and
    Doc 07 §4 lists that as a real finding to report rather than a bug to hide.
    """
    rng = np.random.default_rng(seed)
    best, best_val, best_trace = None, -np.inf, []
    values = []

    for r in range(n_restarts):
        init = sampler(rng)
        p, v, trace = optimise(objective, init, bounds=bounds, steps=steps,
                               lr=lr, seed=seed + r)
        if p is None or not np.isfinite(v):
            continue
        values.append(v)
        if v > best_val:
            best, best_val, best_trace = p, v, trace

    if not values:
        raise RuntimeError("every restart failed to produce a finite likelihood")

    spread = float(np.max(values) - np.min(values))
    log.info("%d/%d restarts converged; best logL %.3f, spread %.3f",
             len(values), n_restarts, best_val, spread)
    if spread > 0.01 * abs(best_val):
        log.warning("restart spread is %.1f%% of the best log-likelihood — the "
                    "surface is multimodal and that must be reported, not "
                    "averaged away", 100 * spread / max(abs(best_val), 1e-9))
    return best, best_val, values, best_trace


def laplace_intervals(objective, params: dict[str, float], *,
                      level: float = 0.95) -> dict[str, list[float]]:
    """Wald intervals from the inverse observed information.

    Doc 08 §6 is explicit that these are **optimistic** for point processes and
    that anything we make a claim about gets a profile-likelihood interval
    instead (see :mod:`hydrocascade.experiments.uncertainty`). They are here as
    the quick reference table, nothing more.

    The Hessian is taken with respect to the **log** parameters and then
    transformed by the delta method, which keeps every interval positive — a
    Wald interval on a rate that includes zero is an artefact of the
    parameterisation, not a statement about the data.
    """
    import torch
    from scipy import stats

    # Signed parameters stay on their own scale; positive ones are transformed
    # to logs so the delta-method interval below cannot cross zero.
    names = list(params)
    real = [is_real_valued(k) for k in names]
    z = torch.tensor(
        [params[k] if r else np.log(max(params[k], 1e-12))
         for k, r in zip(names, real)],
        dtype=torch.float64, requires_grad=True)

    def f(zz):
        return objective({k: (zz[i] if r else torch.exp(zz[i]))
                          for i, (k, r) in enumerate(zip(names, real))})

    try:
        H = torch.autograd.functional.hessian(f, z).detach().numpy()
        cov = np.linalg.inv(-H)
    except Exception as exc:                        # noqa: BLE001
        log.warning("Laplace intervals unavailable: %s", exc)
        return {}

    crit = float(stats.norm.ppf(0.5 + level / 2))
    out = {}
    for i, (k, r) in enumerate(zip(names, real)):
        var = cov[i, i]
        if not np.isfinite(var) or var <= 0:
            out[k] = [float("nan"), float("nan")]
            continue
        se = float(np.sqrt(var))
        if r:                                       # already on the natural scale
            out[k] = [float(params[k] - crit * se), float(params[k] + crit * se)]
        else:                                       # log scale, back-transformed
            out[k] = [float(params[k] * np.exp(-crit * se)),
                      float(params[k] * np.exp(+crit * se))]
    return out


def record(name: str, params: dict, value: float, n_events: int, *,
           branching: float, seed: int, restarts: list[float],
           intervals: dict | None = None, wall: float = 0.0,
           note: str = "") -> FitRecord:
    r = FitRecord(
        name=name, params={k: float(v) for k, v in params.items()},
        log_likelihood=float(value), n_events=int(n_events),
        n_params=len(params), branching_ratio=float(branching), seed=int(seed),
        n_restarts=len(restarts), restart_spread=[float(v) for v in restarts],
        intervals=intervals or {}, wall_clock_s=float(wall),
        git_commit=git_commit(), timestamp=datetime.now().isoformat(timespec="seconds"),
        note=note)
    r.converged = r.valid()[0]
    r.save()
    log.info("%s", r.summary())
    return r
