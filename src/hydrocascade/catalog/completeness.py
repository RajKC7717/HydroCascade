"""Inventory completeness — Doc 04 §3.4, a required analysis.

    Plot the size-frequency distribution of mapped landslide areas on log-log
    axes. Real landslide inventories follow an inverse power law above a
    rollover point, and are systematically incomplete below it.

    1. Identify the rollover area ``A_min``.
    2. Report it.
    3. **Model only events above ``A_min``**, and state this in the paper.

Doc 04 §3.4 explains the stakes: fitting a point process on an inventory that is
incomplete at small sizes, without acknowledging it, produces biased background
rates. This one plot pre-empts a serious reviewer objection.

The rollover is found where the empirical frequency density stops rising and
starts falling as area increases — the modal bin of the log-binned density. The
power-law exponent above it is then estimated by the Clauset-Shalizi-Newman
maximum-likelihood estimator, which is the standard for this and does not depend
on the binning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_setup import get

log = get("catalog.completeness")


@dataclass
class Completeness:
    a_min_m2: float
    alpha: float
    alpha_stderr: float
    n_total: int
    n_above: int
    bin_centres: np.ndarray
    density: np.ndarray

    @property
    def fraction_kept(self) -> float:
        return self.n_above / max(self.n_total, 1)

    def summary(self) -> str:
        return (f"rollover A_min = {self.a_min_m2:,.0f} m2; power-law exponent "
                f"alpha = {self.alpha:.2f} +/- {self.alpha_stderr:.2f}; "
                f"{self.n_above:,} of {self.n_total:,} events above the rollover "
                f"({100 * self.fraction_kept:.1f}%)")


def frequency_density(areas: np.ndarray, n_bins: int = 40):
    """Log-binned frequency density dN/dA, the standard landslide presentation.

    Dividing the count in each bin by the bin *width* is what makes the power
    law a straight line; plotting raw counts against log-spaced bins does not.
    """
    a = np.asarray(areas, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 10:
        raise ValueError(f"need at least 10 mapped areas, got {a.size}")

    edges = np.logspace(np.log10(a.min()), np.log10(a.max()), n_bins + 1)
    counts, _ = np.histogram(a, bins=edges)
    widths = np.diff(edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    density = counts / widths / a.size
    return centres, density, counts


def estimate_alpha(areas: np.ndarray, a_min: float) -> tuple[float, float]:
    """Clauset-Shalizi-Newman MLE for a continuous power law above ``a_min``.

    ``alpha = 1 + n / sum(ln(a_i / a_min))``, standard error ``(alpha-1)/sqrt(n)``.
    """
    a = np.asarray(areas, dtype=float)
    a = a[np.isfinite(a) & (a >= a_min)]
    n = a.size
    if n < 10:
        raise ValueError(f"only {n} events above A_min = {a_min}; cannot fit an exponent")
    s = float(np.sum(np.log(a / a_min)))
    alpha = 1.0 + n / s
    return alpha, (alpha - 1.0) / np.sqrt(n)


def find_rollover(areas: np.ndarray, n_bins: int = 40) -> Completeness:
    """Locate the rollover and fit the exponent above it."""
    a = np.asarray(areas, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    centres, density, counts = frequency_density(a, n_bins=n_bins)

    # The rollover is the modal bin of the frequency density: below it the
    # inventory is losing small events, above it the power law takes over.
    positive = density > 0
    if not positive.any():
        raise ValueError("empty frequency density")
    modal = int(np.argmax(np.where(positive, density, -np.inf)))
    a_min = float(centres[modal])

    # Do not let the rollover swallow the whole catalog.
    while a_min > np.percentile(a, 60) and modal > 0:
        modal -= 1
        a_min = float(centres[modal])

    alpha, err = estimate_alpha(a, a_min)
    result = Completeness(
        a_min_m2=a_min, alpha=alpha, alpha_stderr=err,
        n_total=int(a.size), n_above=int((a >= a_min).sum()),
        bin_centres=centres, density=density,
    )
    log.info("completeness: %s", result.summary())
    if result.fraction_kept < 0.25:
        log.warning("the rollover discards %.0f%% of the inventory — check the "
                    "binning and the mapped areas before adopting it",
                    100 * (1 - result.fraction_kept))
    return result


def apply_cutoff(df, a_min_m2: float):
    """Keep only landslide rows at or above ``A_min``; floods pass through."""
    import pandas as pd

    if df.empty:
        return df
    area = pd.to_numeric(df.get("area_m2"), errors="coerce")
    is_l = df["hazard_type"] == "L"
    drop = is_l & area.notna() & (area < a_min_m2)
    log.info("completeness cutoff at %.0f m2 removes %d of %d landslide rows",
             a_min_m2, int(drop.sum()), int(is_l.sum()))
    return df[~drop].reset_index(drop=True)
