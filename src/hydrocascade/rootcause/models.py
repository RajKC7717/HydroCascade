"""Susceptibility models and their validation — Doc 06 §5.

Three models, because Doc 06 §5.1 asks for a transparent floor alongside the
ensembles: logistic regression, random forest, XGBoost. Modest tuning only —
this is a baseline, not the contribution.

**Spatial block cross-validation is mandatory and is the reason these numbers
are lower than published ones.** Neighbouring pixels on a hillside are nearly
identical; a random split puts one in train and its twin in test, and the model
scores well by memorising the location rather than learning the physics. Doc 06
§5.2 therefore holds out **whole sub-basins**. Both numbers are reported side by
side and the gap between them is itself a finding.

Doc 06 §5.5 anticipates the comparison a reviewer will make: *"a 2026 study
reports flood susceptibility AUC-ROC around 0.905 for Kerala. Our number will
likely be lower, because we use spatial block CV. Report ours honestly and
explain the difference rather than switching to random splits to get a bigger
number."*
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get

log = get("rootcause.models")

TRAIN_END = "2022-12-31"
VALID_END = "2024-05-31"


@dataclass
class Scores:
    auc_roc: float = float("nan")
    auc_pr: float = float("nan")
    brier: float = float("nan")
    n: int = 0
    n_positive: int = 0

    def row(self) -> str:
        return (f"AUC-ROC {self.auc_roc:.3f} · AUC-PR {self.auc_pr:.3f} · "
                f"Brier {self.brier:.3f} (n={self.n:,}, pos={self.n_positive:,})")


@dataclass
class Fitted:
    name: str
    model: object
    features: list[str]
    spatial: Scores = field(default_factory=Scores)
    random: Scores = field(default_factory=Scores)
    temporal: Scores = field(default_factory=Scores)
    oof_prob: np.ndarray | None = None      # out-of-fold, spatial blocks
    oof_y: np.ndarray | None = None


def score(y: np.ndarray, p: np.ndarray) -> Scores:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    s = Scores(n=y.size, n_positive=int(y.sum()))
    if s.n_positive == 0 or s.n_positive == s.n:
        return s                                   # a single class scores nothing
    s.auc_roc = float(roc_auc_score(y, p))
    s.auc_pr = float(average_precision_score(y, p))
    s.brier = float(brier_score_loss(y, p))
    return s


def subbasin_groups(unit_ids: np.ndarray, *, kind: str = "unit") -> np.ndarray:
    """Sub-basin id per unit or reach — the grouping for spatial block CV.

    Doc 06 §5.2 wants whole sub-basins held out, and HydroBASINS already
    provides them: ``subbasins.gpkg`` carries the level-8 polygons the study
    boundary was derived from, so the blocks are hydrological rather than an
    arbitrary grid laid over the region.
    """
    import geopandas as gpd

    subs = gpd.read_file(paths.SPATIAL / "subbasins.gpkg")
    layer = "slope_units.gpkg" if kind == "unit" else "channel_network.gpkg"
    id_col = "unit_id" if kind == "unit" else "reach_id"
    g = gpd.read_file(paths.SPATIAL / layer)
    g = g[[id_col, "geometry"]].copy()
    g["geometry"] = g.geometry.representative_point()
    joined = gpd.sjoin(g.to_crs(subs.crs), subs[["HYBAS_ID", "geometry"]],
                       how="left", predicate="within")
    joined = joined.drop_duplicates(subset=[id_col]).set_index(id_col)
    codes = pd.factorize(joined["HYBAS_ID"])[0]
    lookup = pd.Series(codes, index=joined.index)
    out = lookup.reindex(unit_ids).to_numpy()
    # A representative point can fall in a sliver outside every polygon; give
    # those their own block rather than silently merging them into block 0.
    bad = ~np.isfinite(out.astype(np.float64))
    if bad.any():
        out = out.astype(np.float64)
        out[bad] = np.nanmax(out) + 1 + np.arange(bad.sum())
    return out.astype(int)


def _make(name: str, seed: int, n_pos: int, n_neg: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Doc 06 §2 rule 5: class weighting, because negatives outnumber positives
    # by the sampling ratio and an unweighted fit optimises the wrong thing.
    w = max(n_neg / max(n_pos, 1), 1.0)

    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
            scale_pos_weight=w, eval_metric="aucpr", n_jobs=-1,
            random_state=seed, tree_method="hist")
    raise ValueError(f"unknown model {name!r}")


def _impute(X: pd.DataFrame, medians: pd.Series | None = None):
    med = X.median(numeric_only=True) if medians is None else medians
    return X.fillna(med), med


def cross_validate(df: pd.DataFrame, features: list[str], *, name: str,
                   groups: np.ndarray, seed: int = 0, n_splits: int = 5
                   ) -> Fitted:
    """Fit under both spatial-block and random CV, and report both."""
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    X_raw = df[features]
    y = df["y"].to_numpy().astype(int)
    X, med = _impute(X_raw)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    def run(splitter, split_args) -> tuple[Scores, np.ndarray]:
        oof = np.full(y.size, np.nan)
        for tr, te in splitter.split(*split_args):
            if y[tr].sum() == 0 or y[tr].sum() == tr.size:
                continue
            m = _make(name, seed, int(y[tr].sum()), int((1 - y[tr]).sum()))
            m.fit(X.iloc[tr], y[tr])
            oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        ok = np.isfinite(oof)
        return score(y[ok], oof[ok]), oof

    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        log.warning("only %d spatial block(s) — spatial CV is not meaningful",
                    n_groups)
        spatial, oof = Scores(), np.full(y.size, np.nan)
    else:
        spatial, oof = run(GroupKFold(n_splits=k), (X, y, groups))

    random_s, _ = run(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        (X, y))

    final = _make(name, seed, n_pos, n_neg)
    final.fit(X, y)

    f = Fitted(name=name, model=final, features=list(features),
               spatial=spatial, random=random_s, oof_prob=oof, oof_y=y)
    f.model_medians = med                                    # type: ignore[attr-defined]
    log.info("%-14s spatial  %s", name, spatial.row())
    log.info("%-14s random   %s", name, random_s.row())
    if np.isfinite(spatial.auc_roc) and np.isfinite(random_s.auc_roc):
        log.info("%-14s optimism from random splitting: %+.3f AUC-ROC",
                 name, random_s.auc_roc - spatial.auc_roc)
    return f


def temporal_holdout(df: pd.DataFrame, features: list[str], *, name: str,
                     seed: int = 0) -> Scores:
    """Doc 06 §5.3: train on 2018-2022, validate on 2023 - May 2024.

    Never train on data after a test period. For ``S-dynamic`` this matters more
    than the spatial split: rainfall is autocorrelated in time, so a random
    split over days leaks the weather.
    """
    d = pd.to_datetime(df["date"])
    tr = d <= TRAIN_END
    va = (d > TRAIN_END) & (d <= VALID_END)
    if tr.sum() == 0 or va.sum() == 0:
        log.warning("temporal split is empty on one side; skipping")
        return Scores()

    X, med = _impute(df[features])
    y = df["y"].to_numpy().astype(int)
    if y[tr.to_numpy()].sum() == 0 or y[va.to_numpy()].sum() == 0:
        log.warning("temporal split has no positives on one side; skipping")
        return Scores()

    m = _make(name, seed, int(y[tr.to_numpy()].sum()),
              int((1 - y[tr.to_numpy()]).sum()))
    m.fit(X[tr.to_numpy()], y[tr.to_numpy()])
    p = m.predict_proba(X[va.to_numpy()])[:, 1]
    s = score(y[va.to_numpy()], p)
    log.info("%-14s temporal %s", name, s.row())
    return s
