"""Calibration — Doc 06 §6, *"a hard requirement"*.

``S_i(u)`` is used as a **multiplier** inside the background intensity
``mu_i(u,t)``. A score that ranks perfectly but is systematically 1.4x too large
leaves the ranking untouched and corrupts every intensity, every probability and
every alert threshold downstream. Discrimination and calibration are different
properties and only one of them is being asked for here.

Procedure:

1. Reliability diagram on **held-out** predictions (the spatial out-of-fold
   ones, not the training fit).
2. If the curve departs from the diagonal, fit isotonic regression on the
   held-out predictions only, never on training data.
3. Re-check the diagram after calibration.
4. Export ``S_L(u)`` and ``S_F(s)`` as calibrated probabilities in (0,1).

Isotonic regression is the default rather than Platt scaling because the
distortion from a class-weighted tree ensemble is monotone but not sigmoid;
Platt would impose a shape the data does not have. With very few positives
isotonic overfits, so :func:`calibrate` falls back to Platt below a threshold
and says which it used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_setup import get

log = get("rootcause.calibration")


@dataclass
class Reliability:
    bin_centre: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    count: np.ndarray
    ece: float                 # expected calibration error
    mce: float                 # maximum calibration error

    def summary(self) -> str:
        return f"ECE {self.ece:.4f}, MCE {self.mce:.4f}"


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Reliability:
    """Binned reliability with expected and maximum calibration error.

    Bins are equal-width in probability, which is what a reliability diagram
    conventionally shows. Empty bins are dropped rather than plotted at zero —
    a bin with no predictions is not evidence of perfect calibration.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    centres, pred, obs, cnt = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        pred.append(float(p[m].mean()))
        obs.append(float(y[m].mean()))
        cnt.append(int(m.sum()))

    pred_a, obs_a, cnt_a = np.array(pred), np.array(obs), np.array(cnt)
    gap = np.abs(pred_a - obs_a)
    ece = float((cnt_a * gap).sum() / max(cnt_a.sum(), 1))
    mce = float(gap.max()) if gap.size else float("nan")
    return Reliability(np.array(centres), pred_a, obs_a, cnt_a, ece, mce)


@dataclass
class Calibrator:
    method: str
    model: object
    before: Reliability
    after: Reliability
    applied: bool

    def __call__(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if not self.applied:
            return np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            out = self.model.predict(p)
        else:
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        # Strictly inside (0,1): S_i multiplies an intensity, and a zero there
        # makes a unit permanently impossible, which no terrain justifies.
        return np.clip(out, 1e-6, 1 - 1e-6)

    def summary(self) -> str:
        if not self.applied:
            return (f"already calibrated ({self.before.summary()}); no "
                    f"transform applied")
        return (f"{self.method}: {self.before.summary()} → "
                f"{self.after.summary()}")


def calibrate(y: np.ndarray, p: np.ndarray, *, ece_tolerance: float = 0.02,
              min_positives_for_isotonic: int = 100, seed: int = 0,
              n_folds: int = 5) -> Calibrator:
    """Fit a calibration map on held-out predictions, if one is needed.

    **The "after" number is cross-fitted, and it has to be.** Isotonic
    regression is a free-form monotone step function: fit it on a set of
    predictions and score it on those same predictions and the expected
    calibration error is *exactly* zero, every time, for any model, however
    badly calibrated. That is not a result, it is the definition of the
    estimator — and an ECE of 0.0000 in a report is a red flag, not a triumph.

    So the reported "after" error comes from K-fold cross-fitting: each fold's
    predictions are calibrated by a map fitted on the other folds only. The
    calibrator that is finally *returned* is refitted on everything, which is
    the right thing to deploy — we just never quote its in-sample error.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    before = reliability(y, p)

    if before.ece <= ece_tolerance:
        log.info("calibration not needed: %s is within the %.3f tolerance",
                 before.summary(), ece_tolerance)
        return Calibrator("none", None, before, before, applied=False)

    n_pos = int(y.sum())
    use_isotonic = n_pos >= min_positives_for_isotonic
    method = "isotonic" if use_isotonic else "platt"
    if not use_isotonic:
        log.info("only %d positives — isotonic would overfit, using Platt", n_pos)

    def _fit(pp: np.ndarray, yy: np.ndarray):
        if use_isotonic:
            m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            m.fit(pp, yy)
            return m
        m = LogisticRegression(max_iter=1000)
        m.fit(pp.reshape(-1, 1), yy)
        return m

    def _apply(m, pp: np.ndarray) -> np.ndarray:
        return (m.predict(pp) if use_isotonic
                else m.predict_proba(pp.reshape(-1, 1))[:, 1])

    # --- cross-fitted "after", the number that is honest to report ----------
    q = np.full(y.size, np.nan)
    k = min(n_folds, max(int(y.sum()), 2), max(int((1 - y).sum()), 2))
    if k >= 2:
        for tr, te in StratifiedKFold(n_splits=k, shuffle=True,
                                      random_state=seed).split(p, y):
            q[te] = _apply(_fit(p[tr], y[tr]), p[te])
    ok = np.isfinite(q)
    after = reliability(y[ok], q[ok]) if ok.any() else before

    # --- the calibrator we keep: refitted on everything ---------------------
    model = _fit(p, y)
    c = Calibrator(method, model, before, after, applied=True)
    log.info("calibration %s (after = %d-fold cross-fitted, not in-sample)",
             c.summary(), k)
    if after.ece > before.ece:
        log.warning("calibration made the fit WORSE out of sample "
                    "(%.4f -> %.4f). Usually too few held-out positives. "
                    "Reported rather than hidden.", before.ece, after.ece)
    return c
