"""Hawkes log-likelihoods, stages S1-S3 — Doc 07 §2-§4.

The log-likelihood of a point process observed on ``[0, T]`` is

    log L = sum_k log lambda(t_k)  -  integral_0^T lambda(t) dt

The first term rewards putting intensity where events happened; the second
penalises putting it anywhere else. Getting the **compensator** (the integral)
wrong is the classic failure, because the fit still converges — to the wrong
answer — and nothing looks broken.

For the exponential kernel the integral is analytic, which is why it is the
kernel of choice:

    integral = mu * T + sum_k alpha * theta * (1 - exp(-(T - t_k) / theta))

Each event contributes its full kernel mass ``alpha * theta``, less the tail
that falls beyond the observation window. Forgetting that truncation biases
``alpha`` upward, and does so most for events near the end of the record —
which in this project means 2024, the year the whole thing is about.

**Two implementations, on purpose.** :func:`temporal_direct` is an O(n^2)
double sum written straight from the definition. :func:`temporal_recursive`
uses the Ozaki recursion and is O(n). They share no code path, and
``tests/test_cascade.py`` requires them to agree to 1e-10. Doc 07 §10 gate 2
asks for an independent cross-check; with ``tick`` unavailable this is the
substitute, and the gate report says so rather than claiming a pass.
"""

from __future__ import annotations

import numpy as np

from ..logging_setup import get

log = get("cascade.likelihood")

LOG_FLOOR = 1e-300          # log(0) guard; never reached by a positive mu


# ------------------------------------------------------------------- S1 ----

def temporal_direct(times: np.ndarray, mu: float, alpha: float, theta: float,
                    T: float) -> float:
    """O(n^2) log-likelihood, written straight from the definition.

    Slow and obviously correct. Its only job is to be the thing the fast
    implementation is checked against.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return float(-mu * T)

    total = 0.0
    for i in range(t.size):
        past = t[:i]
        excitation = float(np.sum(alpha * np.exp(-(t[i] - past) / theta)))
        total += float(np.log(max(mu + excitation, LOG_FLOOR)))

    compensator = mu * T + float(np.sum(alpha * theta * (1.0 - np.exp(-(T - t) / theta))))
    return total - compensator


def temporal_recursive(times: np.ndarray, mu: float, alpha: float,
                       theta: float, T: float) -> float:
    """O(n) log-likelihood via the Ozaki recursion.

    With ``R_i = sum_{k<i} exp(-(t_i - t_k)/theta)``,

        R_i = exp(-(t_i - t_{i-1}) / theta) * (1 + R_{i-1}),   R_0 = 0

    so the whole history collapses into one running scalar. This is what makes
    fitting a 10,000-event catalog tractable.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return float(-mu * T)

    total = 0.0
    r = 0.0
    prev = t[0]
    for i in range(t.size):
        if i > 0:
            r = np.exp(-(t[i] - prev) / theta) * (1.0 + r)
            prev = t[i]
        total += float(np.log(max(mu + alpha * r, LOG_FLOOR)))

    compensator = mu * T + float(np.sum(alpha * theta * (1.0 - np.exp(-(T - t) / theta))))
    return total - compensator


def temporal_torch(times, mu, alpha, theta, T):
    """Differentiable S1 log-likelihood for the optimiser.

    Vectorised rather than recursive: the recursion is sequential and would
    build an n-deep autograd graph, which is both slow and a gradient-stability
    risk. For the catalog sizes here the n x n matrix is small enough, and it is
    masked to the strictly-lower triangle so an event can never excite itself —
    the single most common sign error in a Hawkes implementation.
    """
    import torch

    t = times if torch.is_tensor(times) else torch.as_tensor(times, dtype=torch.float64)
    if t.numel() == 0:
        return -mu * T

    dt = t.unsqueeze(1) - t.unsqueeze(0)                  # dt[i, j] = t_i - t_j
    past = torch.tril(torch.ones_like(dt), diagonal=-1)   # strictly j < i
    excite = alpha * torch.exp(-dt.clamp(min=0.0) / theta) * past
    intensity = mu + excite.sum(dim=1)

    compensator = mu * T + (alpha * theta * (1.0 - torch.exp(-(T - t) / theta))).sum()
    return torch.log(intensity.clamp_min(1e-300)).sum() - compensator


# ------------------------------------------------------------------- S2 ----

def space_time_torch(times, xy, mu, alpha, theta, sigma, T, domain,
                     *, n_quad: int = 40):
    """S2: isotropic 2-D Gaussian kernel with a boundary correction.

    The triggering integral over the domain is ``2 pi sigma^2`` only when the
    parent sits far inside it. Near an edge part of the kernel mass falls
    outside and is never observed, so charging the likelihood for it biases
    ``alpha`` downward — and Doc 07 §3 asks specifically for a boundary case in
    the recovery test.

    The correction is exact for a rectangle, because a 2-D isotropic Gaussian
    factorises: the mass inside is the product of two 1-D normal CDF
    differences, and ``erf`` gives those in closed form. No quadrature needed.
    """
    import torch

    t = times if torch.is_tensor(times) else torch.as_tensor(times, dtype=torch.float64)
    p = xy if torch.is_tensor(xy) else torch.as_tensor(xy, dtype=torch.float64)
    x0, y0, x1, y1 = domain
    area = (x1 - x0) * (y1 - y0)

    if t.numel() == 0:
        return -mu * area * T

    dt = t.unsqueeze(1) - t.unsqueeze(0)
    past = torch.tril(torch.ones_like(dt), diagonal=-1)
    d2 = torch.cdist(p, p).pow(2)

    spatial = torch.exp(-d2 / (2.0 * sigma ** 2)) / (2.0 * np.pi * sigma ** 2)
    excite = alpha * torch.exp(-dt.clamp(min=0.0) / theta) * spatial * past
    intensity = mu + excite.sum(dim=1)

    # Fraction of each parent's spatial kernel that lands inside the rectangle.
    root2 = float(np.sqrt(2.0))
    def _mass(lo, hi, c):
        return 0.5 * (torch.erf((hi - c) / (sigma * root2))
                      - torch.erf((lo - c) / (sigma * root2)))
    inside = _mass(x0, x1, p[:, 0]) * _mass(y0, y1, p[:, 1])

    compensator = (mu * area * T
                   + (alpha * theta * (1.0 - torch.exp(-(T - t) / theta)) * inside).sum())
    return torch.log(intensity.clamp_min(1e-300)).sum() - compensator


# ------------------------------------------------------------------- S3 ----

def multitype_torch(times, types, mu, alpha, theta, T, *, marks=None, eta=None):
    """S3: two hazard types with a full 2x2 excitation matrix.

    ``alpha[i, j]`` is how strongly type ``i`` excites type ``j``; ``theta[i, j]``
    is that pair's timescale. The four entries are the scientific content of
    the model — ``alpha[L, F]`` is the landslide-to-flood coupling the whole
    project exists to measure.

    ``marks`` and ``eta`` implement the Doc 02 §6 mark term: a bigger event
    excites more, by ``exp(eta_i * m_k)``. Marks are expected already centred,
    so ``eta = 0`` reduces exactly to the unmarked model.
    """
    import torch

    t = times if torch.is_tensor(times) else torch.as_tensor(times, dtype=torch.float64)
    k = types if torch.is_tensor(types) else torch.as_tensor(types, dtype=torch.long)
    n_types = mu.numel()
    if t.numel() == 0:
        return -(mu * T).sum()

    dt = t.unsqueeze(1) - t.unsqueeze(0)
    past = torch.tril(torch.ones_like(dt), diagonal=-1)

    # a_ij / th_ij broadcast to (target i, source j) by the two type vectors
    a = alpha[k.unsqueeze(0), k.unsqueeze(1)].T        # [i, j] -> alpha[k_j, k_i]
    th = theta[k.unsqueeze(0), k.unsqueeze(1)].T
    boost = torch.ones_like(a)
    if marks is not None and eta is not None:
        m = marks if torch.is_tensor(marks) else torch.as_tensor(marks, dtype=torch.float64)
        boost = torch.exp(eta[k].unsqueeze(0) * m.unsqueeze(0)).expand_as(a)

    excite = a * boost * torch.exp(-dt.clamp(min=0.0) / th) * past
    intensity = mu[k] + excite.sum(dim=1)

    # Compensator: every event contributes its kernel mass to every target type.
    comp = (mu * T).sum()
    for j in range(n_types):
        a_j = alpha[k, j]
        th_j = theta[k, j]
        tail = 1.0 - torch.exp(-(T - t) / th_j)
        b = torch.exp(eta[k] * (marks if marks is not None else 0.0)) \
            if (marks is not None and eta is not None) else 1.0
        comp = comp + (a_j * th_j * tail * b).sum()

    return torch.log(intensity.clamp_min(1e-300)).sum() - comp


# -------------------------------------------------------------- stability ---

def branching_ratio(alpha, theta) -> float:
    """Expected direct offspring per event.

    For one type it is ``alpha * theta``. For several it is the **spectral
    radius** of ``alpha * theta`` — the largest eigenvalue, not the largest
    entry. A matrix whose every entry is below 1 can still be explosive, and
    Doc 07 §7 rule 7 makes a ratio at or above 1 an automatic invalid fit
    regardless of how good the likelihood looks.
    """
    a = np.asarray(alpha, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    if a.ndim == 0:
        return float(a * th)
    return float(np.max(np.abs(np.linalg.eigvals(a * th))))
