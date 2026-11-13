"""Phase 3 figures — Doc 06 §11.

Three figures, each answering one question a reviewer will actually ask:

``phase3_shap_summary.png``
    *Which drivers matter, and in which direction?* The beeswarm is the Doc 06
    §4 primary method — it shows magnitude, sign and spread per feature rather
    than collapsing everything to one bar.
``phase3_reliability.png``
    *Do the probabilities mean anything?* Doc 06 §6 makes this required because
    ``S_i`` is a multiplier: a curve above the diagonal means every downstream
    intensity is inflated by the same factor.
``phase3_susceptibility_map.png``
    *Where?* The calibrated ``S_L(u)`` over the slope units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import palette

log = get("viz.phase3")


def _save(fig, name: str) -> None:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    out = paths.FIGURES / name
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=palette.SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("wrote   figures/%s", name)


def _synthetic_stamp(fig, cfg: Node) -> None:
    """A diagonal watermark, so a synthetic figure cannot be screenshotted clean."""
    from ..ingest import adapters
    if not adapters.synthetic_sources(cfg):
        return
    fig.text(0.5, 0.5, "SYNTHETIC", fontsize=54, color="#d1341f", alpha=0.10,
             ha="center", va="center", rotation=28, weight="bold", zorder=100)


# ------------------------------------------------------------ SHAP summary ---

def shap_summary(cfg: Node, ranking, hazard: str = "landslide") -> None:
    """Beeswarm of per-prediction SHAP contributions for the top features."""
    import matplotlib.pyplot as plt

    palette.apply_style()
    vals, X = ranking.shap_values, ranking.shap_features
    if vals is None or X is None:
        log.warning("no SHAP values to plot")
        return

    order = ranking.table["feature"].tolist()
    order = [f for f in order if f in X.columns][:12][::-1]
    idx = [X.columns.get_loc(f) for f in order]

    fig, ax = plt.subplots(figsize=(8.6, 0.46 * len(order) + 1.9))
    rng = np.random.default_rng(0)
    cmap = palette.sequential_cmap()

    for row, (f, j) in enumerate(zip(order, idx)):
        sv = np.asarray(vals)[:, j]
        fv = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=np.float64)
        lo, hi = np.nanpercentile(fv, [2, 98])
        norm = np.clip((fv - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
        # Vertical jitter proportional to local density, so overlapping points
        # spread instead of hiding each other.
        jitter = rng.normal(0.0, 0.085, size=sv.size)
        ax.scatter(sv, row + jitter, c=norm, cmap=cmap, s=9, alpha=0.7,
                   linewidths=0, rasterized=True)

    ax.axvline(0.0, color=palette.INK_MUTED, lw=1.0, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"`{f}`".strip("`") for f in order],
                       fontsize=9, color=palette.INK_SECONDARY)
    ax.set_xlabel("SHAP value — contribution to log-odds of an event")
    ax.set_title(f"Driver contributions, {hazard} · "
                 f"Doc 06 §4 primary method", loc="left", fontsize=12)
    ax.grid(axis="y", visible=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, pad=0.015, aspect=30)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value", fontsize=9, color=palette.INK_SECONDARY)
    cb.outline.set_visible(False)

    fig.text(0.0, -0.035,
             "Each point is one held-out unit-day. Positive values push toward "
             "an event. Colour is the feature's own value, so a band of dark "
             "points on the right means high values raise risk.",
             fontsize=8.5, color=palette.INK_MUTED, transform=ax.transAxes)
    _synthetic_stamp(fig, cfg)
    _save(fig, "phase3_shap_summary.png")


# --------------------------------------------------------------- reliability ---

def reliability(cfg: Node, calibrator, hazard: str = "landslide") -> None:
    """Predicted versus observed frequency, before and after calibration."""
    import matplotlib.pyplot as plt

    palette.apply_style()
    before, after = calibrator.before, calibrator.after

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(6.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=palette.INK_MUTED,
            label="perfect calibration", zorder=1)
    ax.plot(before.predicted, before.observed, "o-", ms=6, lw=2,
            color=palette.ACCENT, label=f"before  (ECE {before.ece:.3f})",
            zorder=3)
    if calibrator.applied:
        ax.plot(after.predicted, after.observed, "s-", ms=6, lw=2,
                color=palette.WATER,
                label=f"after {calibrator.method}  (ECE {after.ece:.3f})",
                zorder=4)

    # Direct labels as well as the legend: identity never rests on colour alone.
    if before.predicted.size:
        ax.annotate("before", (before.predicted[-1], before.observed[-1]),
                    textcoords="offset points", xytext=(6, -10),
                    color=palette.ACCENT, fontsize=9, weight="bold")
    if calibrator.applied and after.predicted.size:
        ax.annotate("after", (after.predicted[-1], after.observed[-1]),
                    textcoords="offset points", xytext=(6, 6),
                    color=palette.WATER, fontsize=9, weight="bold")

    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Reliability, {hazard} susceptibility · held-out spatial folds",
                 loc="left", fontsize=12)

    axh.bar(before.bin_centre, before.count, width=0.085,
            color=palette.SEQUENTIAL[3], edgecolor=palette.SURFACE, linewidth=2)
    axh.set_yscale("log")
    axh.set_xlabel("predicted probability")
    axh.set_ylabel("count")

    fig.text(0.0, -0.12,
             "Above the diagonal means the model under-predicts; below means it "
             "over-predicts. Because S_i multiplies the background intensity, a "
             "systematic offset here scales every downstream intensity, "
             "probability and alert threshold by the same factor.",
             fontsize=8.5, color=palette.INK_MUTED, transform=axh.transAxes,
             wrap=True)
    _synthetic_stamp(fig, cfg)
    _save(fig, "phase3_reliability.png")


# ---------------------------------------------------------- susceptibility ---

def susceptibility_map(cfg: Node, hazard: str = "landslide") -> None:
    """Calibrated ``S_L(u)`` over the slope units, with the network drawn."""
    import geopandas as gpd
    import matplotlib.pyplot as plt

    palette.apply_style()
    s_path = paths.PROCESSED / f"S_{hazard}.parquet"
    if not s_path.is_file():
        log.warning("%s missing; skipping the susceptibility map", s_path.name)
        return

    s = pd.read_parquet(s_path)
    units = gpd.read_file(paths.SPATIAL / "slope_units.gpkg")
    units = units.merge(s, on="unit_id", how="left")
    net = gpd.read_file(paths.SPATIAL / "channel_network.gpkg", layer="reaches")

    minx, miny, maxx, maxy = units.total_bounds
    aspect = (maxy - miny) / max(maxx - minx, 1.0)
    fig, ax = plt.subplots(figsize=(9.0, 9.0 * aspect + 1.2))

    units.plot(column="S", ax=ax, cmap=palette.sequential_cmap(),
               linewidth=0, legend=True,
               legend_kwds={"label": "calibrated susceptibility S_L(u)",
                            "shrink": 0.55, "pad": 0.012})
    net[net["strahler"] >= 3].plot(ax=ax, color=palette.WATER, linewidth=0.55,
                                   alpha=0.85, zorder=3)

    lm = cfg.landmarks
    import shapely.geometry as sg
    pts = gpd.GeoSeries(
        [sg.Point(float(lm[k]["lon"]), float(lm[k]["lat"]))
         for k in ("mundakkai_crown", "chooralmala_bridge", "nilambur")],
        crs="EPSG:4326").to_crs(units.crs)
    ax.scatter(pts.x, pts.y, s=46, facecolor="none",
               edgecolor=palette.ACCENT, linewidth=1.8, zorder=5)
    for (x, y), name in zip(zip(pts.x, pts.y),
                            ("Mundakkai crown", "Chooralmala", "Nilambur")):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(9, 5),
                    fontsize=9, color=palette.INK, zorder=6,
                    path_effects=None)

    ax.set_axis_off()
    ax.set_title(f"Calibrated {hazard} susceptibility · {len(units):,} slope units",
                 loc="left", fontsize=12)
    fig.text(0.01, 0.015,
             "S is time-independent by construction: it is fitted on terrain "
             "only, never rainfall, so the Phase 4 background does not count "
             "the weather twice.",
             fontsize=8.5, color=palette.INK_MUTED)
    _synthetic_stamp(fig, cfg)
    _save(fig, "phase3_susceptibility_map.png")


def render_all(cfg: Node, results: dict) -> None:
    land = results.get("landslide")
    if land is not None:
        from ..rootcause.run import HEADLINE_RATIO
        rk = land.ranking.get(HEADLINE_RATIO)
        if rk is not None:
            shap_summary(cfg, rk, "landslide")
        if land.calibrator is not None:
            reliability(cfg, land.calibrator, "landslide")
    susceptibility_map(cfg, "landslide")
