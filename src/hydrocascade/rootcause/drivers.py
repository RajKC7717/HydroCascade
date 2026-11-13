"""Driver ranking — Doc 06 §4. Three methods, reported together.

*"Agreement between them is the evidence."* Each method is wrong in a different
way, so a driver that all three put near the top is a driver, whereas one that
only SHAP likes may be an artefact of the tree structure.

| Method | What it gives | Its blind spot |
|---|---|---|
| point-biserial correlation | univariate association | ignores interactions entirely |
| permutation importance | drop in skill when shuffled | **must** be computed out-of-fold |
| SHAP | per-prediction contribution, with sign | can inherit the model's own bias |

The permutation step is the one that is easy to get wrong: computed on training
data it measures how much the model memorised, not how much the feature
matters. Here it is computed on the held-out spatial blocks.

The output feeds Phase 4: the top **two to four** standardised drivers become
``e~(u)``, the argument of the receptivity term. Doc 06 §4 caps it there
deliberately — *"every additional driver here costs identifiability in the
cascade model that we cannot afford."*
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import paths
from ..logging_setup import get

log = get("rootcause.drivers")

TOP_DRIVERS_PATH = paths.PROCESSED / "top_drivers.json"

#: One physical sentence per covariate. Doc 06 §4: *"Not statistics alone — say
#: why that factor matters physically."* A ranked table without these is a
#: correlation list, and a reviewer will say so.
PHYSICAL_MEANING = {
    "slope_mean_deg": "Gravity's driving stress scales with sin(slope); above "
                      "roughly 25 deg the shear stress on a thin regolith "
                      "approaches its strength even when dry.",
    "slope_max_deg": "The steepest facet within a unit is where failure "
                     "initiates, so the maximum matters more than the mean for "
                     "triggering, and the mean more for runout.",
    "twi": "ln(a/tan B) is a steady-state index of how much upslope water "
           "converges here per unit gradient. High TWI means saturation, which "
           "raises pore pressure and cancels frictional strength.",
    "curv_prof": "Concave profile curvature decelerates flow and lets water "
                 "and colluvium accumulate; it marks the hollows where "
                 "shallow landslides repeatedly initiate.",
    "curv_plan": "Convergent plan curvature funnels subsurface flow into a "
                 "line, concentrating pore pressure along it.",
    "relief_m": "Local relief is the potential energy available to a failure — "
                "it sets how far debris can travel once moving.",
    "d_slope_km": "Flow-path distance to the channel. Short d_slope means "
                  "debris reaches a stream and becomes a flood problem; long "
                  "d_slope means it stops on the hillside.",
    "dist_stream_m": "Euclidean distance to a channel, the corridor width for "
                     "flood-undercut bank failures.",
    "drainage_density_km_km2": "Dense drainage indicates erodible, "
                               "weathering-limited terrain and short hillslopes.",
    "R1": "Same-day rainfall — the proximate trigger for shallow failures.",
    "R3": "Three-day rainfall, the window most Kerala I-D thresholds use.",
    "R7": "A week of rain raises the water table rather than the wetting front.",
    "R15": "Fortnightly total; separates a wet monsoon from an intense burst.",
    "R30": "Monthly total, the seasonal wetness the slope failed under.",
    "API": "Antecedent precipitation index, a decayed rainfall memory standing "
           "in for soil moisture. Doc 09 §1.3: usable in real time, unlike "
           "ERA5-Land.",
    "elev_mean_m": "Elevation proxies orographic rainfall and regolith age; it "
                   "is a correlate of the cause rather than a cause.",
    "elev_max_m": "As elevation, for the highest ground in the unit.",
    "northness": "North-facing slopes in the Ghats stay wetter and more "
                 "vegetated; aspect modulates evapotranspiration and root "
                 "cohesion.",
    "eastness": "East-west aspect separates windward from leeward of the "
                "south-west monsoon.",
    "area_km2": "Unit size. Larger units contain more candidate failure sites, "
                "so this is partly an exposure term, not a physical driver.",
    "log_flow_acc": "Upslope contributing area, the numerator of TWI.",
    "injection_area_km2": "Catchment of the reach this unit drains into — "
                          "sets how diluted its debris becomes downstream.",
    "area_dn_km2": "Contributing area at the reach outlet; the first-order "
                   "control on discharge magnitude.",
    "slope": "Channel bed slope, which sets flood-wave celerity through "
             "v = v0 * S^0.3.",
    "strahler": "Stream order, a proxy for channel size and confinement.",
    "length_m": "Reach length; mostly a constant by construction here.",
    "z_dn_m": "Reach outlet elevation.",
    "catch_slope_mean_deg": "Mean hillslope steepness of the contributing "
                            "catchment — steep catchments deliver runoff fast.",
    "catch_twi": "Catchment wetness index; saturated catchments convert "
                 "rainfall to runoff almost one-for-one.",
    "catch_curv_prof": "Catchment profile curvature, a proxy for how much "
                       "storage the hillslopes provide.",
    "catch_elev_mean_m": "Mean catchment elevation, an orographic rainfall proxy.",
    "catch_drainage_density_km_km2": "Drainage density of the contributing "
                                     "catchment; dense networks shorten the "
                                     "time to peak.",
}


@dataclass
class Ranking:
    table: pd.DataFrame          # feature, corr, perm, shap, rank_* columns
    top: list[str]
    shap_values: np.ndarray | None = None
    shap_features: pd.DataFrame | None = None

    def markdown(self, n: int = 12) -> str:
        t = self.table.head(n)
        lines = ["| # | driver | point-biserial r | permutation ΔAUC-PR | "
                 "mean \\|SHAP\\| | physical meaning |",
                 "|---|---|---|---|---|---|"]
        for i, (_, r) in enumerate(t.iterrows(), start=1):
            lines.append(
                f"| {i} | `{r['feature']}` | {r['corr']:+.3f} | "
                f"{r['permutation']:+.4f} | {r['shap']:.4f} | "
                f"{PHYSICAL_MEANING.get(r['feature'], '—')} |")
        return "\n".join(lines)


def point_biserial(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Univariate association. Equivalent to Pearson r with a binary variable."""
    out = {}
    yv = np.asarray(y, dtype=np.float64)
    for c in X.columns:
        v = X[c].to_numpy(dtype=np.float64)
        ok = np.isfinite(v)
        if ok.sum() < 3 or np.std(v[ok]) == 0:
            out[c] = 0.0
            continue
        out[c] = float(np.corrcoef(v[ok], yv[ok])[0, 1])
    return pd.Series(out)


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, *,
                           n_repeats: int = 10, seed: int = 0) -> pd.Series:
    """Drop in AUC-PR when a column is shuffled, on **held-out** rows."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for c in X.columns:
        drops = []
        col = X[c].to_numpy().copy()
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[c] = rng.permutation(col)
            drops.append(base - average_precision_score(
                y, model.predict_proba(Xp)[:, 1]))
        out[c] = float(np.mean(drops))
    return pd.Series(out)


def shap_importance(model, X: pd.DataFrame, *, max_rows: int = 2000,
                    seed: int = 0):
    """Mean absolute SHAP value per feature, plus the raw values for plotting.

    TreeExplainer is exact for tree ensembles and fast; anything else falls back
    to the permutation explainer on a subsample.
    """
    import shap

    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        Xs = X.iloc[np.sort(idx)]
    else:
        Xs = X

    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xs, check_additivity=False)
    except Exception as exc:                      # noqa: BLE001 — model-dependent
        log.info("TreeExplainer unavailable (%s); using the permutation "
                 "explainer on %d rows", type(exc).__name__, min(len(Xs), 300))
        Xs = Xs.iloc[:300]
        explainer = shap.Explainer(lambda d: model.predict_proba(d)[:, 1], Xs)
        vals = explainer(Xs).values

    vals = np.asarray(vals)
    if vals.ndim == 3:                            # (rows, features, classes)
        vals = vals[..., -1]
    return pd.Series(np.abs(vals).mean(axis=0), index=Xs.columns), vals, Xs


def rank(fitted, df: pd.DataFrame, *, seed: int = 0, n_top: int = 4) -> Ranking:
    """Combine the three methods into one ranked table."""
    X = df[fitted.features]
    X = X.fillna(X.median(numeric_only=True))
    y = df["y"].to_numpy().astype(int)

    corr = point_biserial(X, y)

    # Permutation importance on the held-out spatial blocks, never on training
    # rows: on training data it measures memorisation, not importance.
    oof = getattr(fitted, "oof_prob", None)
    if oof is not None and np.isfinite(oof).any():
        held = np.isfinite(oof)
        perm = permutation_importance(fitted.model, X[held], y[held], seed=seed)
    else:
        log.warning("no out-of-fold predictions; permutation importance falls "
                    "back to the full sample and is optimistic")
        perm = permutation_importance(fitted.model, X, y, seed=seed)

    shap_mean, shap_vals, shap_X = shap_importance(fitted.model, X, seed=seed)

    table = pd.DataFrame({
        "feature": fitted.features,
        "corr": corr.reindex(fitted.features).fillna(0.0).to_numpy(),
        "permutation": perm.reindex(fitted.features).fillna(0.0).to_numpy(),
        "shap": shap_mean.reindex(fitted.features).fillna(0.0).to_numpy(),
    })
    for c in ("corr", "permutation", "shap"):
        key = table[c].abs() if c == "corr" else table[c]
        table[f"rank_{c}"] = key.rank(ascending=False, method="min").astype(int)
    # SHAP is the primary method (Doc 06 §4), so it breaks ties in the mean rank.
    table["rank_mean"] = table[["rank_corr", "rank_permutation", "rank_shap"]].mean(axis=1)
    table = table.sort_values(["rank_mean", "rank_shap"]).reset_index(drop=True)

    top = table["feature"].head(n_top).tolist()
    log.info("top %d drivers: %s", n_top, ", ".join(top))
    return Ranking(table=table, top=top, shap_values=shap_vals,
                   shap_features=shap_X)


def stability(rankings: dict[int, Ranking], n_check: int = 3) -> tuple[bool, str]:
    """Doc 06 §2: is the top-3 unchanged across the negative-sampling ratios?

    Doc 06 requires this sentence, or its honest equivalent, in the report:
    *"The top three drivers were unchanged across all three sampling ratios.
    The fourth and fifth exchanged places at 1:20."* A ranking that flips with
    the sampling ratio is not a finding.
    """
    tops = {r: rk.table["feature"].head(n_check).tolist()
            for r, rk in sorted(rankings.items())}
    sets = [tuple(v) for v in tops.values()]
    same_order = len(set(sets)) == 1
    same_set = len({frozenset(s) for s in sets}) == 1

    detail = "; ".join(f"1:{r} → {', '.join(v)}" for r, v in tops.items())
    if same_order:
        return True, (f"The top {n_check} drivers were unchanged, in the same "
                      f"order, across all {len(tops)} sampling ratios ({detail}).")
    if same_set:
        return True, (f"The top {n_check} drivers were the same set across all "
                      f"{len(tops)} sampling ratios, with the order changing "
                      f"({detail}).")
    return False, (f"**The top {n_check} drivers changed with the sampling "
                   f"ratio** ({detail}). Doc 06 §2 is explicit that a driver "
                   f"ranking which flips with the sampling ratio is not a "
                   f"finding, so it is reported as unstable rather than "
                   f"quoted.")


def write_top_drivers(top: list[str], stats: dict) -> None:
    """``e~(u)`` — the interface Phase 4 reads. Between 2 and 4 features."""
    if not 2 <= len(top) <= 4:
        raise ValueError(f"Doc 06 §8 gate 5 requires 2-4 drivers, got {len(top)}")
    TOP_DRIVERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOP_DRIVERS_PATH.write_text(json.dumps(
        {"drivers": top,
         "standardisation": stats,
         "note": "e~(u) for the receptivity term g_i in Doc 02 §6. Standardised "
                 "with the mean and standard deviation recorded here, which must "
                 "be reused at prediction time — refitting them on new data "
                 "silently changes the meaning of the fitted coefficients."},
        indent=2), encoding="utf-8")
    log.info("wrote %s: %s", TOP_DRIVERS_PATH.name, ", ".join(top))
