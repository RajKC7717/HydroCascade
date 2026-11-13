"""Phase 5 figures — Doc 08 §11.

The money figure lives in :mod:`hydrocascade.experiments.backtest` because it is
produced as the last step of the backtest protocol. Everything else is here.

``phase5_leadtime_vs_falsealarm.png``
    Doc 09 §2.4 calls this *"the most operationally meaningful figure in the
    entire project, more so than any accuracy number."* Every point is one
    threshold: the curve shows what you buy and what you pay.
``phase5_ablation_table.png``
    Doc 08 §3's nine rows, drawn rather than tabulated, so the one number per
    row that matters is visible at a glance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import palette

log = get("viz.phase5")


def _save(fig, name: str, cfg: Node | None = None) -> None:
    import matplotlib.pyplot as plt
    if cfg is not None:
        from ..ingest import adapters
        if adapters.synthetic_sources(cfg):
            fig.text(0.5, 0.5, "SYNTHETIC", fontsize=70, color=palette.ACCENT,
                     alpha=0.10, ha="center", va="center", rotation=27,
                     weight="bold", zorder=100)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / name, dpi=170, bbox_inches="tight",
                facecolor=palette.SURFACE)
    plt.close(fig)
    log.info("wrote   figures/%s", name)


def leadtime_vs_falsealarm(cfg: Node, curve: pd.DataFrame,
                           chosen: dict | None = None) -> None:
    """Doc 09 §2.4 — the trade-off, with the cost-ratio optima marked."""
    import matplotlib.pyplot as plt

    palette.apply_style()
    if curve is None or curve.empty:
        log.warning("no trade-off curve to plot")
        return

    c = curve.dropna(subset=["false_alarm_rate", "hit_rate"]).sort_values(
        "false_alarm_rate")
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    ax.plot(c["false_alarm_rate"], c["hit_rate"], lw=2.2, color=palette.WATER,
            zorder=3, label="threshold sweep")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=palette.INK_MUTED, zorder=1,
            label="no skill")

    if chosen:
        for ratio, pt in sorted(chosen.items()):
            ax.scatter([pt.false_alarm_rate], [pt.hit_rate], s=90, zorder=5,
                       facecolor=palette.ACCENT, edgecolor=palette.SURFACE,
                       linewidth=2)
            ax.annotate(f"{ratio:.0f}:1", (pt.false_alarm_rate, pt.hit_rate),
                        textcoords="offset points", xytext=(9, -4), fontsize=9.5,
                        color=palette.ACCENT, weight="bold")

    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("hit rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Lead time versus false alarms — what you buy, what you pay",
                 loc="left", fontsize=12)
    fig.text(0.0, -0.08,
             "Each point is one alert threshold. Labelled points are the optima "
             "at three cost ratios: missing an event costs 1, 5 or 20 times a "
             "false alarm. We supply the curve and recommend 5:1; choosing the "
             "ratio is a policy decision for a disaster management authority, "
             "not for us (Doc 09 §2.3).",
             fontsize=8.5, color=palette.INK_MUTED, transform=ax.transAxes)
    _save(fig, "leadtime_vs_falsealarm.png", cfg)


def ablation_table(cfg: Node, table: pd.DataFrame) -> None:
    """Doc 08 §3 drawn as a horizontal bar chart of held-out degradation."""
    import matplotlib.pyplot as plt

    palette.apply_style()
    if table is None or table.empty:
        return
    t = table.dropna(subset=["delta_test_ll"]).copy()
    if t.empty:
        log.warning("no evaluable ablations to plot")
        return
    t = t.sort_values("delta_test_ll")

    fig, ax = plt.subplots(figsize=(9.6, 0.52 * len(t) + 2.0))
    # Negative delta = removing it hurt = the component earns its place.
    colours = [palette.ACCENT if v < 0 else palette.THIRD
               for v in t["delta_test_ll"]]
    ax.barh(np.arange(len(t)), t["delta_test_ll"], color=colours, height=0.62,
            zorder=3)
    ax.axvline(0, color=palette.INK_SECONDARY, lw=1.2, zorder=4)
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels([f"{r['ablation']}  {r['removes'][:46]}"
                        for _, r in t.iterrows()], fontsize=9)
    ax.set_xlabel("Δ held-out log-likelihood when the component is removed")
    ax.set_title("Which ideas are real — Doc 08 §3", loc="left", fontsize=12)
    ax.grid(axis="y", visible=False)

    for i, v in enumerate(t["delta_test_ll"]):
        ax.annotate(f"{v:+.1f}", (v, i), textcoords="offset points",
                    xytext=(8 if v >= 0 else -8, 0), ha="left" if v >= 0 else "right",
                    va="center", fontsize=8.5, color=palette.INK_SECONDARY)

    fig.text(0.0, -0.10,
             "Bars to the left: removing the component made held-out fit worse, "
             "so it earns its place. Bars near zero: the data cannot "
             "distinguish the two models — a finding, not a failure. Bars to "
             "the right: the model is better without it.",
             fontsize=8.5, color=palette.INK_MUTED, transform=ax.transAxes)
    _save(fig, "ablation_table.png", cfg)


def ladder_plot(cfg: Node, ladder: dict) -> None:
    """M0-M5 held-out log-likelihood, showing where the gains actually come from."""
    import matplotlib.pyplot as plt

    palette.apply_style()
    keys = [k for k in ("M0", "M1", "M2", "M3", "M4", "M5") if k in ladder]
    if len(keys) < 2:
        return
    train = [ladder[k].train_ll for k in keys]
    test = [ladder[k].test_ll for k in keys]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(keys, train, "o-", lw=2, ms=7, color=palette.ACCENT, label="training")
    ax.plot(keys, test, "s--", lw=2, ms=7, color=palette.WATER, label="held out")
    ax.annotate("training", (keys[-1], train[-1]), textcoords="offset points",
                xytext=(8, 0), color=palette.ACCENT, fontsize=9, weight="bold")
    ax.annotate("held out", (keys[-1], test[-1]), textcoords="offset points",
                xytext=(8, 0), color=palette.WATER, fontsize=9, weight="bold")
    ax.set_ylabel("log-likelihood")
    ax.set_title("The nested ladder — where the gain actually comes from",
                 loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    for i, k in enumerate(keys):
        ax.annotate(ladder[k].description[:28], (i, test[i]),
                    textcoords="offset points", xytext=(0, -18), fontsize=7.5,
                    ha="center", color=palette.INK_MUTED, rotation=0)
    fig.text(0.0, -0.10,
             "A gap that widens between the two lines is the model buying "
             "training fit it cannot generalise. M2 is the honest baseline: "
             "background plus rainfall, no excitation at all.",
             fontsize=8.5, color=palette.INK_MUTED, transform=ax.transAxes)
    _save(fig, "phase5_ladder.png", cfg)


def render_all(cfg: Node, out) -> None:
    from ..experiments import ablations

    if out.ablation_runs:
        try:
            ablation_table(cfg, ablations.table(out.ablation_runs))
        except Exception as exc:                    # noqa: BLE001
            log.warning("ablation figure failed: %s", exc)
    if out.ladder:
        try:
            ladder_plot(cfg, out.ladder)
        except Exception as exc:                    # noqa: BLE001
            log.warning("ladder figure failed: %s", exc)

    # The trade-off curve needs a score and a label per unit-day; the backtest
    # roll gives exactly that for the Mundakkai unit.
    bt = out.backtest
    if bt is not None and bt.lam_L.size:
        try:
            from ..realtime import alerts
            y = np.zeros(bt.lam_L.size, dtype=bool)
            ev = pd.Timestamp(str(cfg.wayanad_event.date))
            hit = np.flatnonzero(bt.dates.normalize() == ev)
            if hit.size:
                y[hit[0]] = True
            curve = alerts.tradeoff_curve(bt.lam_L, y)
            chosen = alerts.optimise(bt.lam_L, y)
            leadtime_vs_falsealarm(cfg, curve, chosen)
        except Exception as exc:                    # noqa: BLE001
            log.warning("trade-off figure failed: %s", exc)
