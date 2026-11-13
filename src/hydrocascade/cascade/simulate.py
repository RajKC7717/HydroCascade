"""Hawkes simulation by Ogata's thinning algorithm — Doc 07 §2.

This module exists so that the likelihood can be tested against data whose
answer is known. It is written from scratch rather than taken from a library,
and that is deliberate: Doc 07 §10 gate 2 asks for an **independent**
cross-check, and a simulator and a likelihood that share an author still share
no code, no parameterisation and no derivation. If the fit recovers the
simulator's parameters, two separately-derived pieces of mathematics agree.

Doc 07 §2 originally specified the ``tick`` library for this. ``tick`` needs
Python 3.12 in a throwaway environment and is documented in the same paragraph
as version-fragile; it does not install here. The substitute is stronger than
nothing and weaker than an external library, and the quality-gate report says
exactly that rather than claiming gate 2 passes.

**The algorithm.** Ogata (1981) thinning:

1. At current time ``t``, compute an upper bound ``M >= lambda(s)`` for all
   ``s >= t``. For an exponential kernel the intensity only decays between
   events, so ``lambda(t+)`` itself is a valid bound.
2. Draw the next candidate at ``t + Exp(M)``.
3. Accept it with probability ``lambda(candidate) / M``; otherwise reject and
   repeat from the new time.

Rejected candidates are not events; they are the price of sampling an
inhomogeneous process with a homogeneous proposal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..logging_setup import get

log = get("cascade.simulate")


@dataclass
class TemporalParams:
    """``lambda(t) = mu + sum_k alpha * exp(-(t - t_k) / theta)``.

    The branching ratio — the expected number of direct offspring per event —
    is the kernel integral ``alpha * theta``. It must be **strictly below 1**
    or the process is explosive and the likelihood is unbounded.
    """
    mu: float
    alpha: float
    theta: float

    @property
    def branching_ratio(self) -> float:
        return float(self.alpha * self.theta)

    def validate(self) -> None:
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.alpha < 0 or self.theta <= 0:
            raise ValueError("alpha must be >= 0 and theta > 0")
        if self.branching_ratio >= 1.0:
            raise ValueError(
                f"branching ratio {self.branching_ratio:.3f} >= 1: the process "
                f"is explosive and will not terminate")


def temporal(params: TemporalParams, T: float, *, seed: int = 0,
             max_events: int = 200_000) -> np.ndarray:
    """Simulate a 1-D Hawkes process on ``[0, T]``. Returns sorted times."""
    params.validate()
    rng = np.random.default_rng(seed)
    mu, alpha, theta = params.mu, params.alpha, params.theta

    events: list[float] = []
    t = 0.0
    # `excitation` is sum_k alpha * exp(-(t - t_k)/theta) evaluated at the
    # current time t, carried forward by the recursion
    #     S(t + w) = S(t) * exp(-w / theta)
    # so the intensity costs O(1) per step rather than O(n). The likelihood
    # uses the same identity, but derived from the other direction — this one
    # is a forward simulation step, that one is a backward sum.
    excitation = 0.0

    while t < T and len(events) < max_events:
        # Between events lambda only decays, so its value right now is a valid
        # upper bound for every later time until the next accepted event.
        bound = mu + excitation
        if bound <= 0:
            break

        w = rng.exponential(1.0 / bound)
        t = t + w
        if t >= T:
            break

        excitation *= np.exp(-w / theta)           # decay to the candidate time
        intensity = mu + excitation
        if rng.random() <= intensity / bound:
            events.append(t)
            excitation += alpha                    # the new event starts exciting

    if len(events) >= max_events:
        log.warning("hit the %d-event ceiling at t = %.1f of %.1f; the "
                    "branching ratio is %.3f", max_events, t, T,
                    params.branching_ratio)
    return np.asarray(events, dtype=np.float64)


@dataclass
class SpaceTimeParams:
    """``lambda(t, x) = mu + sum_k alpha * exp(-dt/theta) * N(x - x_k; sigma^2)``.

    The spatial kernel is an isotropic 2-D Gaussian normalised to integrate to
    one over the plane, so the branching ratio is still ``alpha * theta``:
    space redistributes offspring, it does not create them.
    """
    mu: float                    # background rate per unit area per unit time
    alpha: float
    theta: float
    sigma: float                 # kernel width, same units as the domain

    @property
    def branching_ratio(self) -> float:
        return float(self.alpha * self.theta)


def space_time(params: SpaceTimeParams, T: float,
               domain: tuple[float, float, float, float], *, seed: int = 0,
               max_events: int = 100_000):
    """Simulate a space-time Hawkes process. Returns ``(times, xy)``.

    ``domain`` is ``(xmin, ymin, xmax, ymax)``. Offspring that land outside the
    domain are **discarded, not reflected** — that is what makes the boundary
    correction in the likelihood necessary and therefore testable. Doc 07 §3
    asks specifically for a case where events cluster near the boundary.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = domain
    area = (x1 - x0) * (y1 - y0)
    mu, alpha, theta, sigma = params.mu, params.alpha, params.theta, params.sigma
    if params.branching_ratio >= 1.0:
        raise ValueError(
            f"branching ratio alpha * theta = {params.branching_ratio:.3f} >= 1: "
            f"the process is explosive. Note that alpha and theta are each "
            f"individually harmless here ({alpha} and {theta}) — it is their "
            f"product that has to stay below 1.")

    # Immigrants: a homogeneous Poisson process in space and time.
    n_immigrants = rng.poisson(mu * area * T)
    times = list(rng.uniform(0.0, T, n_immigrants))
    xs = list(rng.uniform(x0, x1, n_immigrants))
    ys = list(rng.uniform(y0, y1, n_immigrants))

    # Offspring, generation by generation. This is the cluster (branching)
    # representation, which is exact for a Hawkes process and avoids thinning
    # entirely in space-time.
    frontier = list(zip(times, xs, ys))
    generation = 0
    while frontier and len(times) < max_events:
        nxt = []
        for t, x, y in frontier:
            n_child = rng.poisson(alpha * theta)
            for _ in range(n_child):
                ct = t + rng.exponential(theta)
                if ct >= T:
                    continue
                cx = x + rng.normal(0.0, sigma)
                cy = y + rng.normal(0.0, sigma)
                if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                    continue                        # left the domain, and stays gone
                nxt.append((ct, cx, cy))
        times += [a for a, _, _ in nxt]
        xs += [b for _, b, _ in nxt]
        ys += [c for _, _, c in nxt]
        frontier = nxt
        generation += 1
        if generation > 100:
            log.warning("branching did not die out after 100 generations")
            break

    order = np.argsort(times)
    t_arr = np.asarray(times, dtype=np.float64)[order]
    xy = np.column_stack([np.asarray(xs)[order], np.asarray(ys)[order]])
    return t_arr, xy


def multitype(mu: np.ndarray, alpha: np.ndarray, theta: np.ndarray, T: float,
              *, seed: int = 0, max_events: int = 200_000):
    """Multitype temporal Hawkes by the branching representation.

    ``alpha[i, j]`` is the excitation of type ``j`` by an event of type ``i``,
    matching the convention in :mod:`hydrocascade.cascade.likelihood`. The
    stability condition is that the spectral radius of ``alpha * theta`` is
    below 1 — for a multitype process it is the largest eigenvalue that matters,
    not the individual entries, and a matrix with every entry below 1 can still
    be explosive.

    Returns ``(times, types)`` sorted by time.
    """
    mu = np.asarray(mu, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    n_types = mu.size

    radius = float(np.max(np.abs(np.linalg.eigvals(alpha * theta))))
    if radius >= 1.0:
        raise ValueError(f"spectral radius {radius:.3f} >= 1: explosive")

    rng = np.random.default_rng(seed)
    times: list[float] = []
    types: list[int] = []
    frontier: list[tuple[float, int]] = []

    for i in range(n_types):
        n = rng.poisson(mu[i] * T)
        for t in rng.uniform(0.0, T, n):
            frontier.append((float(t), i))
    times += [t for t, _ in frontier]
    types += [i for _, i in frontier]

    generation = 0
    while frontier and len(times) < max_events:
        nxt = []
        for t, i in frontier:
            for j in range(n_types):
                for _ in range(rng.poisson(alpha[i, j] * theta[i, j])):
                    ct = t + rng.exponential(theta[i, j])
                    if ct < T:
                        nxt.append((ct, j))
        times += [t for t, _ in nxt]
        types += [j for _, j in nxt]
        frontier = nxt
        generation += 1
        if generation > 200:
            break

    order = np.argsort(times)
    return (np.asarray(times, dtype=np.float64)[order],
            np.asarray(types, dtype=np.int64)[order])
