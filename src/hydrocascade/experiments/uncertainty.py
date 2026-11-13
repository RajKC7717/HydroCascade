"""Uncertainty — Doc 08 §6.

    *Hessian-based intervals are typically optimistic for point processes.
    Anything we make a claim about gets a profile-likelihood interval.*

That sentence decides the design. :func:`~hydrocascade.cascade.fit.laplace_intervals`
produces the quick reference table for every parameter; **the numbers quoted in
an abstract come from here instead.**

**Why the Laplace interval is optimistic.** It assumes the log-likelihood is a
symmetric parabola around the optimum. For a Hawkes model it usually is not:
``alpha`` is bounded below by zero and the surface is skewed, so a symmetric
interval is too narrow on the side that matters and can even include negative
values, which the parameterisation forbids.

**What a profile interval does instead.** Fix the parameter of interest at a
grid of values; at each one, re-maximise over *every other parameter*; the
interval is where twice the drop in log-likelihood stays below the chi-squared
critical value. It costs one optimisation per grid point and makes no shape
assumption at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..logging_setup import get

log = get("experiments.uncertainty")


@dataclass
class Profile:
    parameter: str
    grid: np.ndarray
    log_likelihood: np.ndarray
    mle: float
    mle_ll: float
    interval: tuple[float, float] = (float("nan"), float("nan"))
    level: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        lo, hi = self.interval
        return bool(np.isfinite(lo) and lo > 0)

    def summary(self) -> str:
        lo, hi = self.interval
        s = (f"{self.parameter} = {self.mle:.4g} "
             f"[{lo:.4g}, {hi:.4g}] ({100*self.level:.0f}% profile)")
        if self.parameter.startswith("a_"):
            s += (" — **excludes zero**" if self.excludes_zero
                  else " — includes zero, so no detectable effect")
        return s


def profile_likelihood(objective, mle: dict[str, float], parameter: str, *,
                       bounds: dict | None = None, n_grid: int = 15,
                       span: float = 3.0, level: float = 0.95,
                       steps: int = 200, seed: int = 0) -> Profile:
    """Re-maximise over every other parameter at each fixed value.

    ``span`` is the multiplicative half-width of the grid around the MLE: the
    grid runs from ``mle / span`` to ``mle * span``, log-spaced, because these
    are all positive scale parameters and a linear grid would waste most of its
    points on one side.
    """
    from scipy import stats

    from ..cascade import fit

    if parameter not in mle:
        raise KeyError(f"{parameter} is not in the fitted parameters")

    centre = float(mle[parameter])
    if not np.isfinite(centre) or centre <= 0:
        raise ValueError(f"cannot profile {parameter} at {centre}")

    lo_b, hi_b = (bounds or {}).get(parameter, (centre / span, centre * span))
    grid = np.unique(np.clip(
        np.geomspace(max(centre / span, lo_b * 1.001),
                     min(centre * span, hi_b * 0.999), n_grid),
        lo_b * 1.001, hi_b * 0.999))

    others = {k: v for k, v in mle.items() if k != parameter}
    other_bounds = {k: v for k, v in (bounds or {}).items() if k != parameter}

    lls = []
    for g in grid:
        import torch

        fixed = torch.tensor(float(g), dtype=torch.float64)

        def obj(q, _fixed=fixed):
            return objective({**q, parameter: _fixed})

        if others:
            best, val, _ = fit.optimise(obj, others, bounds=other_bounds,
                                        steps=steps, lr=0.06, seed=seed)
            lls.append(val if best is not None else -np.inf)
        else:
            lls.append(float(obj({}).detach()))
        log.debug("  profile %s = %.4g → logL %.3f", parameter, g, lls[-1])

    ll = np.asarray(lls, dtype=np.float64)
    mle_ll = float(np.nanmax(ll))
    crit = float(stats.chi2.ppf(level, df=1)) / 2.0
    inside = ll >= mle_ll - crit

    if inside.any():
        lo = float(grid[inside][0])
        hi = float(grid[inside][-1])
        if inside[0]:
            log.warning("%s profile interval is open at the lower end of the "
                        "grid — widen `span`", parameter)
        if inside[-1]:
            log.warning("%s profile interval is open at the upper end of the "
                        "grid — widen `span`", parameter)
    else:
        lo = hi = float("nan")

    p = Profile(parameter=parameter, grid=grid, log_likelihood=ll,
                mle=centre, mle_ll=mle_ll, interval=(lo, hi), level=level)
    log.info("profile: %s", p.summary())
    return p


def parametric_bootstrap(fit_fn, simulate_fn, n: int = 200, *, seed: int = 0
                         ) -> dict[str, np.ndarray]:
    """Refit on ``n`` catalogs simulated from the fitted model.

    Doc 08 §6 calls this *"the version in the paper if time allows"*. It makes
    no distributional assumption about the estimator at all, which is why it is
    the reference the other two methods are judged against.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for i in range(n):
        try:
            cat = simulate_fn(rng)
            est = fit_fn(cat)
        except Exception as exc:                    # noqa: BLE001
            log.debug("bootstrap replicate %d failed: %s", i, exc)
            continue
        for k, v in est.items():
            draws.setdefault(k, []).append(float(v))
    out = {k: np.asarray(v) for k, v in draws.items()}
    log.info("parametric bootstrap: %d/%d replicates succeeded",
             len(next(iter(out.values()), [])), n)
    return out


def power_analysis(effect: float, se: float, *, alpha: float = 0.05,
                   power: float = 0.80, n_current: int = 0) -> dict:
    """Doc 08 §9 step 2 — the negative-result plan, made concrete.

    *"Compute a power analysis: given the observed effect size and variance, how
    many cascade pairs would be needed for 80% power? State that number."*

    The standard error of a rate parameter scales as ``1/sqrt(n)``, so the
    sample size needed to detect the observed effect is the current size times
    the squared ratio of the required to the observed z-statistic. Saying "we
    would have needed 340 pairs and we have 6" is a real, publishable statement
    about data requirements — and it is a far stronger position than forcing a
    result out of six.
    """
    from scipy import stats

    if not np.isfinite(effect) or not np.isfinite(se) or se <= 0:
        return {"feasible": False,
                "reason": "effect or standard error not estimable"}
    z_obs = abs(effect) / se
    z_need = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    factor = (z_need / z_obs) ** 2 if z_obs > 0 else float("inf")
    return {
        "feasible": True,
        "observed_z": float(z_obs),
        "required_z": z_need,
        "n_current": int(n_current),
        "n_required": int(np.ceil(n_current * factor)) if n_current else None,
        "scale_factor": float(factor),
        "already_powered": bool(z_obs >= z_need),
    }


def plot_profiles(profiles: list[Profile]) -> None:
    """Profile curves with the chi-squared cut-off drawn on."""
    import matplotlib.pyplot as plt
    from scipy import stats

    from .. import paths
    from ..viz import palette

    profiles = [p for p in profiles if np.isfinite(p.log_likelihood).any()]
    if not profiles:
        return
    palette.apply_style()
    fig, axes = plt.subplots(1, len(profiles), figsize=(4.2 * len(profiles), 3.6))
    axes = np.atleast_1d(axes)

    for ax, p in zip(axes, profiles):
        crit = float(stats.chi2.ppf(p.level, df=1)) / 2.0
        ax.plot(p.grid, p.log_likelihood, lw=2, color=palette.WATER, zorder=3)
        ax.axhline(p.mle_ll - crit, ls="--", lw=1.3, color=palette.ACCENT,
                   zorder=2, label=f"{100*p.level:.0f}% cut-off")
        ax.axvline(p.mle, ls=":", lw=1.2, color=palette.INK_MUTED, zorder=2)
        lo, hi = p.interval
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(lo, hi, color=palette.WATER, alpha=0.12, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(p.parameter)
        ax.set_ylabel("profile log-likelihood")
        ax.set_title(p.summary().split("—")[0].strip(), fontsize=9.5, loc="left")
        ax.legend(fontsize=8)

    fig.suptitle("Profile-likelihood intervals — Doc 08 §6", fontsize=12,
                 x=0.01, ha="left")
    fig.text(0.01, -0.04,
             "Each curve re-maximises over every other parameter at each fixed "
             "value, so it makes no assumption that the surface is a symmetric "
             "parabola — which for a bounded rate parameter it is not.",
             fontsize=8.5, color=palette.INK_MUTED)
    fig.tight_layout()
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / "phase5_profile_likelihood.png", dpi=170,
                bbox_inches="tight", facecolor=palette.SURFACE)
    plt.close(fig)
    log.info("wrote   figures/phase5_profile_likelihood.png")
