"""External baselines B1-B3 — Doc 06 §7.

These are the models the cascade engine must beat, and they are built **now**,
before the novel model exists. Doc 06's reasoning is worth repeating: *"if you
build baselines only after the novel model, you will unconsciously tune the
baseline to lose. Build them first, honestly, while you have no result to
protect."*

| ID | Baseline | What beating it proves |
|---|---|---|
| **B1** | Climatology — mean rate per unit per calendar month | the floor; everything must beat this |
| **B2** | Rainfall intensity-duration threshold | **beating what is actually used operationally today** |
| **B3** | ML susceptibility x rainfall (S-dynamic XGBoost) | beating the strongest non-point-process competitor |

B4 and B5 need the Phase 4 code and are run in Phase 5.

Doc 06 §7 is emphatic about B2: *"Do not build a strawman... A well-built B2
that our model beats is worth far more than a badly built one that it beats
easily."* So B2 here is the real thing — a lower-envelope power law in
(intensity, duration) space, fitted to triggering rainfall the way the landslide
literature does it, with the offset tuned on the validation period only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..logging_setup import get

log = get("rootcause.baselines")

#: Accumulation windows available as covariates, and their durations in days.
WINDOWS = {"R1": 1.0, "R3": 3.0, "R7": 7.0, "R15": 15.0, "R30": 30.0}


# ------------------------------------------------------------ B1 climatology ---

@dataclass
class Climatology:
    """Mean event rate per unit per calendar month, from training data only."""
    rate: dict[tuple[int, int], float]      # (unit_id, month) -> rate
    month_rate: dict[int, float]            # fallback when a unit is unseen
    global_rate: float

    def predict(self, unit_ids: np.ndarray, dates: pd.Series) -> np.ndarray:
        months = pd.to_datetime(dates).dt.month.to_numpy()
        out = np.empty(len(unit_ids), dtype=np.float64)
        for i, (u, m) in enumerate(zip(np.asarray(unit_ids), months)):
            out[i] = self.rate.get((int(u), int(m)),
                                   self.month_rate.get(int(m), self.global_rate))
        return out


def fit_climatology(train: pd.DataFrame) -> Climatology:
    """B1. Deliberately the simplest thing that is not zero.

    Rates are estimated on the sampled design matrix, so they are rates *within
    the sample*, not per unit-day of the real world. That is the right scale for
    comparing against the other baselines on the same rows, and the wrong scale
    for quoting as an absolute hazard — which is why nothing quotes it.
    """
    d = train.copy()
    d["month"] = pd.to_datetime(d["date"]).dt.month
    by_unit = d.groupby(["unit_id", "month"])["y"].mean()
    by_month = d.groupby("month")["y"].mean()
    c = Climatology(
        rate={(int(u), int(m)): float(v) for (u, m), v in by_unit.items()},
        month_rate={int(m): float(v) for m, v in by_month.items()},
        global_rate=float(d["y"].mean()))
    log.info("B1 climatology: %d (unit, month) cells, global rate %.4f",
             len(c.rate), c.global_rate)
    return c


# ----------------------------------------------- B2 intensity-duration line ---

@dataclass
class IDThreshold:
    """``I = alpha * D^(-beta)`` — the classical landslide triggering line.

    ``alpha`` is fitted to the **lower envelope** of triggering rainfall, which
    is what the literature means by an I-D threshold: the line below which
    landslides are essentially never observed. ``scale`` then shifts that line
    to hit a stated false-alarm rate on the validation period.
    """
    alpha: float
    beta: float
    scale: float = 1.0
    envelope_quantile: float = 0.05
    target_far: float = float("nan")
    achieved_far: float = float("nan")
    points: pd.DataFrame = field(default_factory=pd.DataFrame)

    def line(self, duration_days: np.ndarray) -> np.ndarray:
        return self.scale * self.alpha * np.power(duration_days, -self.beta)

    def exceedance(self, df: pd.DataFrame) -> np.ndarray:
        """Largest ratio of observed intensity to the threshold, over windows.

        A ratio above 1 means at least one accumulation window crossed the line.
        Returning the ratio rather than a bare flag gives the baseline a score
        to be ranked by, so AUC is defined for it and the comparison against the
        other baselines is like for like.
        """
        best = np.zeros(len(df), dtype=np.float64)
        for col, D in WINDOWS.items():
            if col not in df.columns:
                continue
            intensity = pd.to_numeric(df[col], errors="coerce").to_numpy() / D
            ratio = intensity / self.line(np.array(D))
            best = np.fmax(best, np.nan_to_num(ratio, nan=0.0))
        return best

    def alarm(self, df: pd.DataFrame) -> np.ndarray:
        return self.exceedance(df) >= 1.0

    def summary(self) -> str:
        return (f"I = {self.scale * self.alpha:.2f} * D^-{self.beta:.3f} "
                f"(mm/day vs days; envelope q{self.envelope_quantile:.2f}, "
                f"scale {self.scale:.3f})")


def fit_id_threshold(train: pd.DataFrame, valid: pd.DataFrame | None = None, *,
                     envelope_quantile: float = 0.05,
                     target_far: float = 0.10) -> IDThreshold:
    """B2, fitted properly rather than as a strawman.

    1. For every triggering (positive) row, form one (duration, intensity)
       point per accumulation window: ``I_D = R_D / D`` in mm/day.
    2. At each duration take the ``envelope_quantile`` of triggering
       intensities — the lower envelope, not the mean.
    3. Fit ``log I = log alpha - beta log D`` through those points by least
       squares.
    4. Slide the line on the **validation** period until the false-alarm rate
       hits ``target_far``. Never on the test period.
    """
    pos = train[train["y"] == 1]
    if pos.empty:
        raise ValueError("no positives to fit an I-D threshold to")

    rows = []
    for col, D in WINDOWS.items():
        if col not in pos.columns:
            continue
        v = pd.to_numeric(pos[col], errors="coerce").dropna().to_numpy() / D
        v = v[v > 0]
        if v.size < 5:
            continue
        rows.append({"duration_d": D, "n": int(v.size),
                     "intensity_envelope": float(np.quantile(v, envelope_quantile)),
                     "intensity_median": float(np.median(v))})
    pts = pd.DataFrame(rows)
    if len(pts) < 2:
        raise ValueError("need at least two durations to fit an I-D line")

    x = np.log(pts["duration_d"].to_numpy())
    y = np.log(pts["intensity_envelope"].to_numpy())
    beta_neg, log_alpha = np.polyfit(x, y, 1)
    thr = IDThreshold(alpha=float(np.exp(log_alpha)), beta=float(-beta_neg),
                      envelope_quantile=envelope_quantile, points=pts)

    # --- step 4: tune the offset on validation to a stated false-alarm rate --
    tune = valid if valid is not None and not valid.empty else train
    neg = tune[tune["y"] == 0]
    if not neg.empty:
        thr.target_far = target_far
        ratios = thr.exceedance(neg)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size:
            # Raising `scale` raises the line and lowers the alarm rate. The
            # scale that gives exactly `target_far` is the (1 - far) quantile
            # of the negatives' exceedance ratios.
            thr.scale = float(np.quantile(ratios, 1.0 - target_far))
            thr.scale = max(thr.scale, 1e-6)
            thr.achieved_far = float(thr.alarm(neg).mean())
    log.info("B2 %s", thr.summary())
    log.info("B2 false-alarm rate on the tuning set: %.3f (target %.3f)",
             thr.achieved_far, thr.target_far)
    return thr


# ---------------------------------------------------------------- reporting ---

def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    from .models import score
    s = score(y, p)
    return {"baseline": name, "auc_roc": s.auc_roc, "auc_pr": s.auc_pr,
            "brier": s.brier, "n": s.n, "n_positive": s.n_positive}


def lead_time_and_far(y: np.ndarray, alarm: np.ndarray) -> tuple[float, float]:
    """Hit rate and false-alarm rate — always reported together (Doc 08 §7).

    An alarm system with a 100% hit rate that fires every day has a lead time
    of 'always' and is worthless. Neither number means anything alone.
    """
    y = np.asarray(y).astype(bool)
    a = np.asarray(alarm).astype(bool)
    hit = float(a[y].mean()) if y.any() else float("nan")
    far = float(a[~y].mean()) if (~y).any() else float("nan")
    return hit, far
