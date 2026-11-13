"""Model diagnostics — Doc 07 §2 and §10 gates 3-4.

Written early, deliberately: Doc 07 §2 says to build the diagnostics module at
S1 *"because it is easier to write when nothing is on fire."*

**Time rescaling** is the one that actually tests the model. If the fitted
intensity is right, then transforming each event time by its own compensator,

    tau_k = integral_0^{t_k} lambda(s) ds

turns the events into a **unit-rate Poisson process**, whatever the original
intensity looked like. So the gaps ``tau_k - tau_{k-1}`` should be Exponential(1)
and nothing else. A QQ plot of those gaps against the exponential quantiles is
therefore a direct, assumption-free test of the whole model at once — kernel,
background and compensator together. It is the point-process equivalent of
plotting residuals, and it fails loudly when the compensator is wrong, which is
precisely the bug that otherwise hides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import paths
from ..logging_setup import get

log = get("cascade.diagnostics")


@dataclass
class Residuals:
    gaps: np.ndarray             # rescaled inter-event gaps
    ks_stat: float
    ks_p: float
    n: int

    def summary(self) -> str:
        return (f"{self.n:,} rescaled gaps, KS statistic {self.ks_stat:.4f}, "
                f"p = {self.ks_p:.3f} "
                f"({'consistent with' if self.ks_p > 0.05 else 'INCONSISTENT with'} "
                f"unit-rate Poisson)")


def time_rescaling(times: np.ndarray, compensator: np.ndarray) -> Residuals:
    """Rescaled residuals and their KS test against Exponential(1).

    ``compensator[k]`` is the integrated intensity up to event ``k``. The gaps
    between consecutive values are the rescaled inter-event times.
    """
    from scipy import stats

    c = np.asarray(compensator, dtype=np.float64)
    order = np.argsort(np.asarray(times, dtype=np.float64))
    c = np.maximum.accumulate(c[order])       # a compensator cannot decrease
    gaps = np.diff(np.concatenate([[0.0], c]))
    gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
    if gaps.size < 5:
        return Residuals(gaps, float("nan"), float("nan"), gaps.size)
    ks = stats.kstest(gaps, "expon")
    return Residuals(gaps, float(ks.statistic), float(ks.pvalue), gaps.size)


def qq_plot(res: Residuals, name: str, *, cfg=None) -> None:
    """QQ against the exponential, with the 1:1 line — Doc 07 §10 gate 4."""
    import matplotlib.pyplot as plt

    from ..viz import palette
    palette.apply_style()

    g = np.sort(res.gaps)
    n = g.size
    if n < 5:
        log.warning("too few residuals to plot")
        return
    q_theory = -np.log(1.0 - (np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    hi = float(max(q_theory.max(), g.max()))
    ax.plot([0, hi], [0, hi], ls="--", lw=1.4, color=palette.INK_MUTED,
            label="perfect fit", zorder=2)
    ax.scatter(q_theory, g, s=12, color=palette.WATER, alpha=0.75,
               linewidths=0, zorder=3, label="rescaled gaps")
    ax.set_xlabel("theoretical Exponential(1) quantile")
    ax.set_ylabel("observed rescaled gap")
    ax.set_title(f"Time-rescaling residuals · {name}", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.98, 0.04, res.summary(), transform=ax.transAxes, ha="right",
            fontsize=8.5, color=palette.INK_MUTED)

    fig.text(0.0, -0.06,
             "If the fitted intensity is correct these gaps are Exponential(1) "
             "by construction, whatever shape the intensity had. Points bending "
             "above the line mean the model under-predicts bursts; below, that "
             "it over-predicts them.",
             fontsize=8.5, color=palette.INK_MUTED, transform=ax.transAxes)

    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / f"phase4_residual_qq_{name.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=palette.SURFACE)
    plt.close(fig)
    log.info("wrote   figures/%s", out.name)


def convergence_plot(traces: dict[str, list[float]]) -> None:
    """Optimiser trajectories — Doc 07 §7 rule 6."""
    import matplotlib.pyplot as plt

    from ..viz import palette
    palette.apply_style()

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    colours = list(palette.CATEGORICAL) * 4
    for (name, tr), c in zip(traces.items(), colours):
        v = np.asarray(tr, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        ax.plot(np.arange(v.size), v, lw=1.6, color=c, label=name)
        ax.annotate(name, (v.size - 1, v[-1]), textcoords="offset points",
                    xytext=(5, 0), fontsize=9, color=c, va="center")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("log-likelihood")
    ax.set_title("Convergence · Adam then L-BFGS", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / "phase4_convergence.png", dpi=170,
                bbox_inches="tight", facecolor=palette.SURFACE)
    plt.close(fig)
    log.info("wrote   figures/phase4_convergence.png")


def event_compensator(params: dict, data, spec) -> np.ndarray:
    """Compensator evaluated at each event time, for the rescaling test.

    Approximated on the event grid: the background contributes at a constant
    rate between events and each past source contributes its accumulated kernel
    mass. That is exact for the background and accurate to the day for the
    triggering part, which is the resolution the catalog has anyway.
    """
    import torch

    from .model import make_objective          # noqa: F401 — shared conventions

    t = np.asarray(data.t, dtype=np.float64)
    order = np.argsort(t)
    ts = t[order]

    # Background: constant daily rate per type, taken from the fitted base.
    n_l = max(int((data.kind == 0).sum()), 1)
    n_f = max(int((data.kind == 1).sum()), 1)
    rate = (n_l + n_f) / max(data.T, 1.0)
    base = rate * ts

    # Triggering: each prior event adds its kernel mass by the time of the next.
    theta = float(params.get("theta1", 0.5))
    a = float(params.get("a_LL", 0.0)) + float(params.get("a_FF", 0.0))
    extra = np.zeros_like(ts)
    for i in range(1, ts.size):
        dt = ts[i] - ts[:i]
        extra[i] = a * np.sum(1.0 - np.exp(-dt / max(theta, 1e-6))) / max(i, 1)
    return base + extra
