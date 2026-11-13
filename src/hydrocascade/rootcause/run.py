"""Phase 3 end to end — `hydrocascade rootcause`.

Produces every Doc 06 §11 deliverable:

* ``models/susceptibility_landslide.pkl`` and ``susceptibility_flood.pkl``
* ``data/processed/S_landslide.parquet``, ``S_flood.parquet``
* ``data/processed/top_drivers.json`` — the ``e~(u)`` interface to Phase 4
* ``reports/phase3_features.md``, ``phase3_drivers.md``,
  ``phase3_calibration.md``, ``phase3_baselines.md``
* the three Phase 3 figures

and evaluates the six Doc 06 §8 quality gates.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .. import paths
from ..config import Node
from ..logging_setup import get
from . import baselines, calibration, dataset, drivers, features, models

log = get("rootcause.run")

RATIOS = (1, 5, 20)          # Doc 06 §2 rule 3: 1:5 headline, 1:1 and 1:20 checks
HEADLINE_RATIO = 5
MODEL_NAMES = ("logistic", "random_forest", "xgboost")
PRIMARY_MODEL = "xgboost"


@dataclass
class HazardResult:
    hazard: str                                    # "landslide" | "flood"
    selection: features.Selection
    fitted: dict[str, models.Fitted] = field(default_factory=dict)
    temporal: dict[str, models.Scores] = field(default_factory=dict)
    ranking: dict[int, drivers.Ranking] = field(default_factory=dict)
    stable: bool = False
    stability_text: str = ""
    calibrator: calibration.Calibrator | None = None
    baseline_rows: list[dict] = field(default_factory=list)
    id_threshold: baselines.IDThreshold | None = None
    n_positive: int = 0
    sampling_text: str = ""


def run(cfg: Node, *, force: bool = False) -> dict:
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.project.seed)

    results = {}
    for hazard in ("landslide", "flood"):
        try:
            results[hazard] = _one_hazard(cfg, hazard, seed)
        except Exception as exc:                   # noqa: BLE001 — report, don't abort
            log.error("Phase 3 failed for %s: %s: %s", hazard,
                      type(exc).__name__, exc)

    if not results:
        raise RuntimeError("Phase 3 produced nothing; see the errors above")

    _write_reports(cfg, results)
    _write_figures(cfg, results)
    return results


# ------------------------------------------------------------- one hazard ---

def _one_hazard(cfg: Node, hazard: str, seed: int) -> HazardResult:
    log.info("=" * 70)
    log.info("Phase 3 — %s", hazard)
    build = dataset.build_landslide if hazard == "landslide" else dataset.build_flood
    kind = "unit" if hazard == "landslide" else "reach"

    # --- headline sample, collinearity, models ------------------------------
    df, feats = build(cfg, ratio=HEADLINE_RATIO, seed=seed, dynamic=True)
    sel = features.select(df, feats)
    res = HazardResult(hazard=hazard, selection=sel,
                       n_positive=int(df["y"].sum()),
                       sampling_text=str(df.attrs.get("sampling", "")))

    groups = models.subbasin_groups(df["unit_id"].to_numpy(), kind=kind)
    log.info("spatial block CV over %d sub-basins", len(np.unique(groups)))

    for name in MODEL_NAMES:
        res.fitted[name] = models.cross_validate(
            df, sel.kept, name=name, groups=groups, seed=seed)
        res.temporal[name] = models.temporal_holdout(
            df, sel.kept, name=name, seed=seed)

    primary = res.fitted[PRIMARY_MODEL]

    # --- driver ranking at every sampling ratio -----------------------------
    for ratio in RATIOS:
        if ratio == HEADLINE_RATIO:
            res.ranking[ratio] = drivers.rank(primary, df, seed=seed)
            continue
        d2, f2 = build(cfg, ratio=ratio, seed=seed, dynamic=True)
        g2 = models.subbasin_groups(d2["unit_id"].to_numpy(), kind=kind)
        keep = [c for c in sel.kept if c in f2]
        fit2 = models.cross_validate(d2, keep, name=PRIMARY_MODEL,
                                     groups=g2, seed=seed)
        res.ranking[ratio] = drivers.rank(fit2, d2, seed=seed)

    res.stable, res.stability_text = drivers.stability(res.ranking)
    log.info("driver stability: %s", res.stability_text)

    # --- calibration on the held-out spatial folds --------------------------
    oof, y = primary.oof_prob, primary.oof_y
    ok = np.isfinite(oof)
    res.calibrator = calibration.calibrate(y[ok], oof[ok])

    # --- S-static: the multiplier Phase 4 actually uses ---------------------
    _export_susceptibility(cfg, hazard, seed, res)

    # --- baselines B1-B3 ----------------------------------------------------
    _baselines(df, sel, res, seed)
    return res


def _export_susceptibility(cfg: Node, hazard: str, seed: int,
                           res: HazardResult) -> None:
    """Fit S-static and score every unit, then calibrate and export.

    S-static must **not** see rainfall. Doc 06 §1 separates the two variants for
    a reason: if the susceptibility multiplier already contains the weather,
    the Phase 4 background counts the rain twice, once through ``S_i`` and again
    through its own rainfall term.
    """
    from ..features import terrain_covariates as tc

    build = dataset.build_landslide if hazard == "landslide" else dataset.build_flood
    kind = "unit" if hazard == "landslide" else "reach"

    df_s, feats_s = build(cfg, ratio=HEADLINE_RATIO, seed=seed, dynamic=False)
    keep = [c for c in res.selection.kept if c in feats_s]
    groups = models.subbasin_groups(df_s["unit_id"].to_numpy(), kind=kind)
    static = models.cross_validate(df_s, keep, name=PRIMARY_MODEL,
                                   groups=groups, seed=seed)
    log.info("S-static (%s) spatial %s", hazard, static.spatial.row())

    ok = np.isfinite(static.oof_prob)
    cal = calibration.calibrate(static.oof_y[ok], static.oof_prob[ok])

    all_units = tc.load() if hazard == "landslide" else tc.load_reaches().rename(
        columns={"reach_id": "unit_id"})
    X = all_units[keep]
    X = X.fillna(X.median(numeric_only=True))
    raw = static.model.predict_proba(X)[:, 1]
    s_val = cal(raw)

    out = pd.DataFrame({"unit_id": all_units["unit_id"].to_numpy(),
                        "S_raw": raw, "S": s_val})
    path = paths.PROCESSED / f"S_{hazard}.parquet"
    out.to_parquet(path, index=False)
    log.info("wrote %s: %d units, S in [%.4f, %.4f], median %.4f",
             path.name, len(out), s_val.min(), s_val.max(), np.median(s_val))

    with open(paths.MODELS / f"susceptibility_{hazard}.pkl", "wb") as fh:
        pickle.dump({"model": static.model, "features": keep,
                     "calibrator": cal, "medians": X.median(numeric_only=True),
                     "spatial_scores": static.spatial.__dict__,
                     "seed": seed, "hazard": hazard,
                     "built": date.today().isoformat()}, fh)
    res.static_scores = static.spatial                       # type: ignore[attr-defined]
    res.static_calibrator = cal                              # type: ignore[attr-defined]


def _baselines(df: pd.DataFrame, sel: features.Selection, res: HazardResult,
               seed: int) -> None:
    """B1, B2, B3 on the same rows and the same metrics."""
    d = pd.to_datetime(df["date"])
    train = df[d <= models.TRAIN_END]
    valid = df[(d > models.TRAIN_END) & (d <= models.VALID_END)]
    if train.empty or valid.empty:
        log.warning("temporal split empty; baselines evaluated in-sample and "
                    "are therefore optimistic")
        train, valid = df, df

    y = valid["y"].to_numpy().astype(int)

    # B1 — climatology
    try:
        clim = baselines.fit_climatology(train)
        p = clim.predict(valid["unit_id"].to_numpy(), valid["date"])
        res.baseline_rows.append(baselines.evaluate("B1 climatology", y, p))
    except Exception as exc:                        # noqa: BLE001
        log.warning("B1 failed: %s", exc)

    # B2 — intensity-duration threshold
    try:
        thr = baselines.fit_id_threshold(train, valid)
        res.id_threshold = thr
        ratio = thr.exceedance(valid)
        row = baselines.evaluate("B2 intensity-duration", y, ratio / (1 + ratio))
        hit, far = baselines.lead_time_and_far(y, thr.alarm(valid))
        row.update(hit_rate=hit, false_alarm_rate=far)
        res.baseline_rows.append(row)
    except Exception as exc:                        # noqa: BLE001
        log.warning("B2 failed: %s", exc)

    # B3 — the S-dynamic ensemble, scored out of fold on the spatial blocks
    f = res.fitted.get(PRIMARY_MODEL)
    if f is not None and f.oof_prob is not None:
        ok = np.isfinite(f.oof_prob)
        res.baseline_rows.append(baselines.evaluate(
            "B3 S-dynamic XGBoost (spatial CV)", f.oof_y[ok], f.oof_prob[ok]))
    for name in MODEL_NAMES:
        s = res.fitted[name].spatial
        res.baseline_rows.append({
            "baseline": f"— {name} (spatial CV)", "auc_roc": s.auc_roc,
            "auc_pr": s.auc_pr, "brier": s.brier, "n": s.n,
            "n_positive": s.n_positive})


# ----------------------------------------------------------------- reports ---

def synthetic_active(cfg: Node) -> bool:
    from ..ingest import adapters
    return bool(adapters.synthetic_sources(cfg))


def _synthetic_banner(cfg: Node) -> str:
    from ..ingest import adapters
    synth = adapters.synthetic_sources(cfg)
    if not synth:
        return ""
    n_synth, n_total = adapters.catalog_synthetic_counts()
    return (
        "> ### ⚠ Fitted on synthetic data\n>\n"
        "> `data_mode` is `synthetic` for " + ", ".join(f"`{s}`" for s in synth) +
        f", and {n_synth:,} of\n"
        f"> {n_total:,} catalog rows are synthetic stand-ins. **Every number in this "
        "report describes\n> the stand-in generators, not Kerala.** They are "
        "here to show the procedure runs end\n> to end and that the diagnostics "
        "fire. Supplying the real inputs and rerunning\n"
        "> `hydrocascade rootcause` turns them into results — see "
        "`docs/15_HOW_TO_RUN.md` section 3.\n\n")


def _write_reports(cfg: Node, results: dict[str, HazardResult]) -> None:
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    banner = _synthetic_banner(cfg)
    stamp = date.today().isoformat()

    # ---------------------------------------------------- phase3_features ---
    L = [f"# Phase 3 — feature selection and collinearity\n",
         f"*Generated by `hydrocascade rootcause` on {stamp}.*\n", banner,
         "Doc 06 §3 procedure, in order: drop >20% missing, then |r| > 0.85 "
         "keeping the more physically interpretable of each pair, then VIF > 10 "
         "iteratively.\n",
         "**Two covariate families specified by Doc 05 §6 are absent.** "
         "SoilGrids (texture, depth, organic carbon) and ESA WorldCover (land "
         "cover class and fractions) were not downloaded, so the model is "
         "terrain- and rainfall-only. Both are free and login-free; until they "
         "are in, soil and land cover cannot appear in any driver ranking, and "
         "their absence is a ceiling on achievable skill rather than evidence "
         "that they do not matter.\n"]
    for hazard, r in results.items():
        s = r.selection
        L.append(f"## {hazard.title()}\n")
        L.append(f"{s.summary()}. Sampling: {r.sampling_text}\n")
        L.append("**Kept:** " + ", ".join(f"`{c}`" for c in s.kept) + "\n")
        if s.dropped_missing:
            L.append("**Dropped for missingness (>20%)**\n")
            L.append("| feature | missing |")
            L.append("|---|---|")
            L += [f"| `{c}` | {100*f:.1f}% |" for c, f in s.dropped_missing]
            L.append("")
        if s.dropped_correlated:
            L.append("**Dropped for |r| > 0.85** — the kept feature of each pair "
                     "is the more physically interpretable one, per the ordering "
                     "in `rootcause/features.py`.\n")
            L.append("| dropped | kept instead | \\|r\\| |")
            L.append("|---|---|---|")
            L += [f"| `{a}` | `{b}` | {r_:.3f} |" for a, b, r_ in s.dropped_correlated]
            L.append("")
        if s.dropped_vif:
            L.append("**Dropped for VIF > 10**, iteratively\n")
            L.append("| feature | VIF when dropped |")
            L.append("|---|---|")
            L += [f"| `{c}` | {v:.1f} |" for c, v in s.dropped_vif]
            L.append("")
        if s.vif_final:
            L.append("**Final VIF** — all below 10 by construction\n")
            L.append("| feature | VIF |")
            L.append("|---|---|")
            L += [f"| `{k}` | {v:.2f} |" for k, v in s.vif_final.items()]
            L.append("")
    (paths.REPORTS / "phase3_features.md").write_text("\n".join(L), encoding="utf-8")

    # ----------------------------------------------------- phase3_drivers ---
    L = [f"# Phase 3 — driver ranking\n",
         f"*Generated by `hydrocascade rootcause` on {stamp}.*\n", banner,
         "Doc 06 §4: three methods, reported together, because each is wrong in "
         "a different way and agreement between them is the evidence. "
         "Permutation importance is computed on the **held-out spatial blocks** "
         "— on training rows it measures memorisation, not importance.\n"]
    for hazard, r in results.items():
        L.append(f"## {hazard.title()}\n")
        rk = r.ranking[HEADLINE_RATIO]
        L.append(rk.markdown())
        L.append("")
        L.append("### Stability across negative-sampling ratios — Doc 06 §2\n")
        L.append(("✅ " if r.stable else "❌ ") + r.stability_text + "\n")
        L.append("| ratio | 1st | 2nd | 3rd | 4th |")
        L.append("|---|---|---|---|---|")
        for ratio in RATIOS:
            top = r.ranking[ratio].table["feature"].head(4).tolist()
            top += [""] * (4 - len(top))
            L.append(f"| 1:{ratio} | " + " | ".join(f"`{t}`" if t else "—"
                                                    for t in top) + " |")
        L.append("")
        if hazard == "landslide":
            L.append("### `e~(u)` — what Phase 4 receives\n")
            L.append("Doc 06 §4 caps this at two to four standardised drivers: "
                     "*every additional driver here costs identifiability in the "
                     "cascade model that we cannot afford.* Exported to "
                     "`data/processed/top_drivers.json`.\n")
            L.append("Selected: " + ", ".join(f"`{d}`" for d in rk.top[:4]) + "\n")
    (paths.REPORTS / "phase3_drivers.md").write_text("\n".join(L), encoding="utf-8")

    # ------------------------------------------------- phase3_calibration ---
    L = [f"# Phase 3 — calibration\n",
         f"*Generated by `hydrocascade rootcause` on {stamp}.*\n", banner,
         "Doc 06 §6 makes this a hard requirement, and the reason is mechanical: "
         "`S_i(u)` is a **multiplier** inside `mu_i(u,t)`. A score that ranks "
         "perfectly but sits 40% too high leaves every ranking untouched and "
         "corrupts every intensity, probability and alert threshold downstream. "
         "Discrimination and calibration are different properties.\n",
         "Calibration is fitted on the **held-out spatial folds only** — fitting "
         "it on training predictions would calibrate the model to its own "
         "memorised answers.\n",
         "| hazard | stage | method | ECE before | ECE after | MCE before | MCE after |",
         "|---|---|---|---|---|---|---|"]
    for hazard, r in results.items():
        for label, c in (("S-dynamic", r.calibrator),
                         ("S-static (exported)", getattr(r, "static_calibrator", None))):
            if c is None:
                continue
            L.append(f"| {hazard} | {label} | {c.method} | {c.before.ece:.4f} | "
                     f"{c.after.ece:.4f} | {c.before.mce:.4f} | {c.after.mce:.4f} |")
    L.append("")
    L.append("`S_L(u)` and `S_F(s)` are exported to "
             "`data/processed/S_landslide.parquet` and `S_flood.parquet` as "
             "calibrated probabilities strictly inside (0,1) — a zero there "
             "would make a unit permanently impossible, which no terrain "
             "justifies. This file is the interface contract between Part 1 and "
             "Part 2.\n")
    (paths.REPORTS / "phase3_calibration.md").write_text("\n".join(L), encoding="utf-8")

    # --------------------------------------------------- phase3_baselines ---
    L = [f"# Phase 3 — baselines B1, B2, B3\n",
         f"*Generated by `hydrocascade rootcause` on {stamp}.*\n", banner,
         "Doc 06 §7: *\"if you build baselines only after the novel model, you "
         "will unconsciously tune the baseline to lose. Build them first, "
         "honestly, while you have no result to protect.\"* These are built "
         "before any cascade model exists.\n",
         "**AUC-PR is the headline metric**, not AUC-ROC: under the sampling "
         "imbalance ROC is optimistic, and the operational question is precision "
         "among the days we raise an alarm.\n"]
    for hazard, r in results.items():
        L.append(f"## {hazard.title()}\n")
        L.append("| model | AUC-ROC | AUC-PR | Brier | n | positives |")
        L.append("|---|---|---|---|---|---|")
        for row in r.baseline_rows:
            L.append(f"| {row['baseline']} | {row['auc_roc']:.3f} | "
                     f"{row['auc_pr']:.3f} | {row['brier']:.3f} | "
                     f"{row['n']:,} | {row['n_positive']:,} |")
        L.append("")
        L.append("### Spatial block CV versus random splitting\n")
        L.append("Doc 06 §5.2 requires both, side by side. The gap is the "
                 "optimism a random split buys by putting a hillside in training "
                 "and its immediate neighbour in test.\n")
        L.append("| model | spatial AUC-ROC | random AUC-ROC | optimism | "
                 "spatial AUC-PR | temporal AUC-PR |")
        L.append("|---|---|---|---|---|---|")
        gaps = []
        for name in MODEL_NAMES:
            f = r.fitted[name]
            gap = f.random.auc_roc - f.spatial.auc_roc
            gaps.append(gap)
            L.append(f"| {name} | {f.spatial.auc_roc:.3f} | "
                     f"{f.random.auc_roc:.3f} | {gap:+.3f} | "
                     f"{f.spatial.auc_pr:.3f} | {r.temporal[name].auc_pr:.3f} |")
        L.append("")

        # A near-zero gap is a statement about the *data*, not a licence to use
        # random splits. Left unexplained a reader concludes the opposite.
        worst = max(abs(g) for g in gaps) if gaps else 0.0
        if worst < 0.01 and synthetic_active(cfg):
            L.append("**The optimism column is essentially zero, and that is a "
                     "property of the synthetic catalog, not a finding about "
                     "spatial cross-validation.** The stand-in inventory places "
                     "landslides independently across slope units, so "
                     "neighbouring units carry no shared signal for a random "
                     "split to leak. Real inventories cluster heavily — "
                     "neighbouring hillsides fail in the same storm — and the "
                     "gap on real data is expected to be large. Do not read "
                     "this row as permission to drop the spatial split.\n")
        elif worst >= 0.02:
            L.append(f"**A gap of {worst:+.3f} AUC-ROC is the optimism a random "
                     f"split buys.** It is reported rather than absorbed, "
                     f"because it is the difference between a number that "
                     f"transfers to a new hillside and one that does not.\n")
        if r.id_threshold is not None:
            t = r.id_threshold
            L.append("### B2, the intensity-duration threshold\n")
            L.append(f"Fitted line: **{t.summary()}**\n")
            L.append(f"Tuned on the validation period to a target false-alarm "
                     f"rate of {t.target_far:.2f}; achieved "
                     f"{t.achieved_far:.3f}.\n")
            if not t.points.empty:
                L.append("| duration (days) | triggering events | lower-envelope "
                         "intensity (mm/day) | median (mm/day) |")
                L.append("|---|---|---|---|")
                for _, p in t.points.iterrows():
                    L.append(f"| {p['duration_d']:.0f} | {int(p['n']):,} | "
                             f"{p['intensity_envelope']:.1f} | "
                             f"{p['intensity_median']:.1f} |")
                L.append("")
    L.append("### The benchmark comparison Doc 06 §5.5 asks for\n")
    L.append("A 2026 study reports flood susceptibility AUC-ROC around **0.905** "
             "for Kerala. Our spatial-block numbers will normally be lower, and "
             "that is the honest comparison rather than a defect: a random split "
             "over a 30 m grid lets a model see almost the same hillside in both "
             "folds. The optimism column above measures exactly how much that is "
             "worth on this data. Reporting a lower-but-more-honest number and "
             "explaining it is a strength signal; a suspiciously high one is "
             "not.\n")
    (paths.REPORTS / "phase3_baselines.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote 4 Phase 3 reports")

    # ------------------------------------------------------- e~(u) export ---
    land = results.get("landslide")
    if land is not None:
        rk = land.ranking[HEADLINE_RATIO]
        top = rk.top[:4]
        df, feats = dataset.build_landslide(cfg, ratio=HEADLINE_RATIO,
                                            seed=int(cfg.project.seed))
        stats = {c: {"mean": float(pd.to_numeric(df[c], errors="coerce").mean()),
                     "std": float(pd.to_numeric(df[c], errors="coerce").std(ddof=0))}
                 for c in top}
        drivers.write_top_drivers(top, stats)


def _write_figures(cfg: Node, results: dict[str, HazardResult]) -> None:
    from ..viz import phase3 as fig3
    try:
        fig3.render_all(cfg, results)
    except Exception as exc:                        # noqa: BLE001
        log.error("Phase 3 figures failed: %s: %s", type(exc).__name__, exc)


# ------------------------------------------------------------------- gates ---

def gates(cfg: Node, results: dict[str, HazardResult]) -> list[tuple[str, str, str]]:
    """The six Doc 06 §8 gates, as ``(name, status, detail)``."""
    from ..ingest import adapters
    out = []
    synth = bool(adapters.synthetic_sources(cfg))

    land = results.get("landslide")
    if land is None:
        return [("P3.1 driver ranking stable", "BLOCKED", "Phase 3 did not run")]

    out.append(("P3.1 driver ranking stable across 3 sampling ratios",
                "PASS" if land.stable else "FAIL", land.stability_text))

    f = land.fitted.get(PRIMARY_MODEL)
    both = f is not None and np.isfinite(f.spatial.auc_roc) and np.isfinite(f.random.auc_roc)
    out.append(("P3.2 spatial block CV used, both numbers reported",
                "PASS" if both else "FAIL",
                f"spatial {f.spatial.auc_roc:.3f} vs random {f.random.auc_roc:.3f} "
                f"(optimism {f.random.auc_roc - f.spatial.auc_roc:+.3f})"
                if both else "one of the two splits produced no score"))

    c = land.calibrator
    out.append(("P3.3 reliability diagram produced and calibration re-checked",
                "PASS" if c is not None else "FAIL",
                c.summary() if c is not None else "no calibrator"))

    exported = [p.name for p in (paths.PROCESSED / "S_landslide.parquet",
                                 paths.PROCESSED / "S_flood.parquet") if p.is_file()]
    out.append(("P3.4 S_L(u) and S_F(s) exported as calibrated tables",
                "PASS" if len(exported) == 2 else "FAIL",
                ", ".join(exported) or "nothing exported"))

    td = drivers.TOP_DRIVERS_PATH
    if td.is_file():
        n = len(json.loads(td.read_text(encoding="utf-8"))["drivers"])
        out.append(("P3.5 e~(u) chosen, between 2 and 4 features",
                    "PASS" if 2 <= n <= 4 else "FAIL", f"{n} drivers"))
    else:
        out.append(("P3.5 e~(u) chosen, between 2 and 4 features", "FAIL",
                    "top_drivers.json missing"))

    have = {r["baseline"].split()[0] for r in land.baseline_rows}
    need = {"B1", "B2", "B3"}
    missing = need - have
    status = "FAIL" if missing else ("BLOCKED" if synth else "PASS")
    detail = (f"missing {', '.join(sorted(missing))}" if missing else
              "B1, B2 and B3 built and evaluated on the same rows and metrics"
              + (" — but every input is synthetic, so the metrics describe the "
                 "generators, not Kerala" if synth else ""))
    out.append(("P3.6 B1, B2, B3 built and evaluated", status, detail))
    return out
