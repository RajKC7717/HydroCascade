"""Collinearity removal — Doc 06 §3, in the order the doc specifies.

1. Drop anything more than 20% missing in the study area.
2. Where ``|r| > 0.85``, keep the **more physically interpretable** of the pair
   and record the decision.
3. Drop VIF > 10, iteratively.
4. Write every removal to ``reports/phase3_features.md``.

Step 2 is a judgement call and Doc 06 says so: *"slope, elevation and
topographic wetness index are typically strongly correlated. Deciding which to
keep is a judgement call — make it, write down the reason, and be ready to
defend it."* The judgement is encoded once, here, in :data:`INTERPRETABILITY`,
so that it is visible, reviewable and identical on every run — rather than
being whatever ``pandas`` happened to put first in the column order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("rootcause.features")

#: Higher score = keep in preference. The ordering is physical, not statistical.
#:
#: Slope beats elevation because a hillslope fails on its gradient, not its
#: height above sea level — elevation is a proxy that happens to correlate with
#: slope in a mountain range and would not transfer to Maharashtra. TWI beats
#: raw flow accumulation because it already combines upslope area with gradient,
#: which is the quantity that actually governs pore pressure. Distance to stream
#: beats d_slope for the *flood corridor*, but d_slope beats it for landslides
#: because debris travels along the flow path, not as the crow flies — so both
#: are kept unless they genuinely collide.
INTERPRETABILITY = {
    "slope_mean_deg": 100, "slope_max_deg": 95,
    "twi": 90, "curv_prof": 85, "curv_plan": 84,
    "d_slope_km": 80, "dist_stream_m": 78,
    "relief_m": 70, "drainage_density_km_km2": 68,
    "R1": 66, "R3": 65, "R7": 64, "R15": 63, "R30": 62, "API": 61,
    "area_km2": 50, "northness": 45, "eastness": 44,
    "elev_mean_m": 40, "elev_max_m": 35,
    "log_flow_acc": 30, "injection_area_km2": 25,
    # reach-side
    "catch_slope_mean_deg": 100, "catch_twi": 90, "catch_curv_prof": 85,
    "slope": 82, "area_dn_km2": 60, "strahler": 55, "length_m": 20,
    "catch_drainage_density_km_km2": 68, "catch_elev_mean_m": 40, "z_dn_m": 38,
}


@dataclass
class Selection:
    kept: list[str]
    dropped_missing: list[tuple[str, float]] = field(default_factory=list)
    dropped_correlated: list[tuple[str, str, float]] = field(default_factory=list)
    dropped_vif: list[tuple[str, float]] = field(default_factory=list)
    vif_final: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{len(self.kept)} features kept; dropped "
                f"{len(self.dropped_missing)} for missingness, "
                f"{len(self.dropped_correlated)} for |r| > 0.85, "
                f"{len(self.dropped_vif)} for VIF > 10")


def _rank(name: str) -> int:
    return INTERPRETABILITY.get(name, 0)


def variance_inflation(X: pd.DataFrame) -> dict[str, float]:
    """VIF per column, computed from the inverse correlation matrix.

    ``VIF_j = 1 / (1 - R_j^2)`` and the diagonal of the inverse correlation
    matrix is exactly that, which avoids fitting one regression per feature.
    A singular matrix means perfect collinearity, so that column gets infinite
    VIF and is removed first.
    """
    Z = X.astype(np.float64)
    Z = Z.loc[:, Z.std(ddof=0) > 0]                    # constants have no VIF
    if Z.shape[1] < 2:
        return {c: 1.0 for c in Z.columns}
    corr = np.corrcoef(Z.to_numpy(), rowvar=False)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    out = {}
    for i, c in enumerate(Z.columns):
        v = float(inv[i, i])
        out[c] = float("inf") if not np.isfinite(v) or v <= 0 else v
    for c in X.columns:
        out.setdefault(c, float("inf"))                # constant or all-NaN
    return out


def select(df: pd.DataFrame, features: list[str], *,
           max_missing: float = 0.20, max_corr: float = 0.85,
           max_vif: float = 10.0) -> Selection:
    """Run the Doc 06 §3 procedure and return what survived, with reasons."""
    sel = Selection(kept=list(features))

    # --- step 1: missingness ------------------------------------------------
    frac = df[sel.kept].isna().mean()
    for c in list(sel.kept):
        if frac[c] > max_missing:
            sel.dropped_missing.append((c, float(frac[c])))
            sel.kept.remove(c)
    if sel.dropped_missing:
        log.info("dropped %d features for >%.0f%% missing",
                 len(sel.dropped_missing), 100 * max_missing)

    X = df[sel.kept].astype(np.float64)
    X = X.fillna(X.median(numeric_only=True))

    # --- step 2: pairwise correlation --------------------------------------
    corr = X.corr().abs()
    pairs = []
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and r > max_corr:
                pairs.append((float(r), a, b))
    # Strongest correlation first, so the most flagrant duplication is resolved
    # before weaker ones that may disappear once it is gone.
    for r, a, b in sorted(pairs, reverse=True):
        if a not in sel.kept or b not in sel.kept:
            continue
        loser = b if _rank(a) >= _rank(b) else a
        winner = a if loser == b else b
        sel.kept.remove(loser)
        sel.dropped_correlated.append((loser, winner, r))
        log.info("|r| = %.3f: dropped %s, kept %s", r, loser, winner)

    # --- step 3: VIF, iteratively ------------------------------------------
    X = X[sel.kept]
    for _ in range(len(sel.kept)):
        vif = variance_inflation(X)
        worst = max(vif, key=lambda k: vif[k])
        if vif[worst] <= max_vif:
            sel.vif_final = {k: round(v, 2) for k, v in sorted(
                vif.items(), key=lambda kv: -kv[1])}
            break
        sel.dropped_vif.append((worst, float(vif[worst])))
        sel.kept.remove(worst)
        X = X[sel.kept]
        log.info("VIF = %.1f: dropped %s", vif[worst], worst)
        if len(sel.kept) < 2:
            break

    log.info("collinearity selection: %s", sel.summary())
    return sel
