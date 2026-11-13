"""Stage S4/S5 kernel components — Doc 07 §5-§6, Doc 02 §4-§5.

This is where the model stops being a textbook Hawkes process and becomes the
thing the project exists to test. Four ideas, each separable so Phase 5 can
ablate it:

**A1 — cross-excitation.** A landslide raises flood intensity, not just
landslide intensity. ``alpha[L, F] > 0``.

**A2 — the network, not the plane.** Influence travels down the drainage
network, so a reach 2 km away over a ridge feels nothing while one 20 km
downstream feels a lot. Implemented as a hard reachability gate,
``1[s in Down(pi(x_k))]``, read from the Phase 2 table.

**A3 — a wave, not a field.** The excitation arrives at ``t_k + tau``, not at
``t_k``. ``tau = T_ref / v0`` with ``T_ref`` precomputed and ``v0`` fitted, so
the timing prediction is checkable against a hydrograph. Doc 08 §5 calls the
shaded arrival window *"what distinguishes this figure from anything else in
the literature."*

**A4 — debris has to reach the channel.** ``1[d_slope(x_k) <= c * exp(gamma *
m_k)]``: a big landslide runs further, a small one stops on the hillside. Doc
07 §5 requires an explicit check that the fitted gate does not exclude the
Wayanad event itself.

**A5 — dilution.** ``(A(pi(x_k)) / A(s))^psi``. A debris pulse entering a
3 km² headwater is a catastrophe there and a rounding error 2,000 km²
downstream.

Everything below is a pure function of tensors. Nothing loads a file, nothing
loops in Python over events, and every gate is a multiplicative factor so
setting it to a constant 1 is exactly the corresponding ablation.
"""

from __future__ import annotations

import numpy as np

SECONDS_PER_DAY = 86400.0


# --------------------------------------------------------- spatial densities ---
#
# These live here, once, because they were once written twice — in the
# likelihood and again in the forward predictor — and the two copies drifted.
# Four separate bugs in this project had that exact shape: two pieces of code
# computing what should be the same quantity, differently. None crashed; each
# produced a confident, wrong number. Anything that both fits and predicts must
# call these functions rather than restate them.

#: Bank corridor for the flood -> landslide kernel, in metres (Doc 02 §5's
#: ``w_max``). A flood undercuts the bank it touches; a hillslope 3 km away does
#: not care that the river rose.
CORRIDOR_M = 2000.0

#: Width of the landslide -> landslide spatial kernel, in metres.
LL_SIGMA_M = 1500.0


def ll_spatial_density(d2_m2):
    """Normalised isotropic 2-D Gaussian at squared distance ``d2``, per km².

    Integrates to exactly 1 over the plane, which is what the compensator
    assumes when it charges each source only ``alpha x (temporal mass)``. Works
    on a float, a numpy array or a torch tensor, so the fitting code and the
    predictor evaluate literally the same expression.
    """
    s2 = LL_SIGMA_M ** 2
    norm = 1.0e6 / (2.0 * np.pi * s2)               # per m^2 -> per km^2
    try:
        import torch
        if torch.is_tensor(d2_m2):
            return torch.exp(-d2_m2 / (2.0 * s2)) * norm
    except ImportError:                              # pragma: no cover
        pass
    return np.exp(-np.asarray(d2_m2, dtype=np.float64) / (2.0 * s2)) * norm


def corridor_density() -> float:
    """Uniform density over the corridor disc, per km² — integrates to 1."""
    return 1.0e6 / (np.pi * CORRIDOR_M ** 2)


#: Minimum lag, in days, before any event can excite another.
#:
#: The catalog is daily. Two events on the same day have no knowable order — the
#: sort puts one first arbitrarily — yet without this lag the model let the
#: first-sorted event "trigger" the other at lag zero, where the kernel sits at
#: its peak. Measured on the training catalog, 48% of all triggering pairs and
#: 53% of flood->flood pairs were same-day; one synthetic storm day held 988
#: events. That is co-occurrence under a shared storm, which belongs to the
#: background, being booked as triggering — and it is why the branching ratio
#: sat pinned at its stability bound in every fit.
#:
#: With a one-day lag the kernel keeps its full mass (it is shifted, not
#: truncated), same-day pairs contribute nothing, and the compensator charges
#: exactly what the intensity can collect. The cost is honest and worth stating:
#: at daily resolution a sub-daily response — the debris pulse, a travel time of
#: a few hours — cannot be resolved. Testing those needs hourly data.
DAILY_LAG_DAYS = 1.0


#: The isotropic alternative to the network kernel — ablation A2, baseline B5.
#: Flood-side excitation spreads to every reach within ISO_RADIUS_M by
#: straight-line distance, weighted by a Gaussian of width ISO_SIGMA_M and
#: normalised over the reaches that fall inside the radius. No downstream gate,
#: no travel time, no dilution: exactly Doc 06 §7's "cross terms present, but no
#: network". It is what the network formulation has to beat, and Doc 08 §2 calls
#: that comparison the single most important in the paper.
ISO_SIGMA_M = 2000.0
ISO_RADIUS_M = 3.0 * ISO_SIGMA_M


# ------------------------------------------------------------ connectivity ---

def runout_gate(d_slope_m, mark, c, gamma, *, hard: bool = False, sharpness=40.0):
    """``1[d_slope <= c * exp(gamma * m)]`` — can the debris reach a channel?

    The hard indicator has zero gradient everywhere, so the optimiser cannot
    learn ``c`` or ``gamma`` from it. During fitting we therefore use a logistic
    relaxation that approaches the indicator as ``sharpness`` grows; at
    evaluation time ``hard=True`` gives the true gate.

    This is not a fudge — it is the standard treatment for a learnable
    threshold, and the relaxation is only ever used to compute a gradient. Every
    reported number uses the hard gate.
    """
    import torch

    reach = c * torch.exp(gamma * mark)
    if hard:
        return (d_slope_m <= reach).to(d_slope_m.dtype)
    # logistic in log-space, so the transition width scales with the threshold
    # rather than being 50 m wide whether the threshold is 100 m or 10 km
    z = torch.log(reach.clamp_min(1e-6)) - torch.log(d_slope_m.clamp_min(1e-6))
    return torch.sigmoid(sharpness * z)


def dilution(area_source_km2, area_target_km2, psi):
    """``(A(pi(x_k)) / A(s))^psi`` — Doc 02 §5.3.

    ``psi = 0`` switches dilution off exactly (ablation A5); ``psi = 1`` makes
    the effect inversely proportional to catchment growth. The ratio is clamped
    at 1 from above because a target can never drain less than its own source —
    if it does, the network is wrong and we would rather see a flat 1 than a
    silently amplifying kernel.
    """
    import torch

    ratio = (area_source_km2 / area_target_km2.clamp_min(1e-9)).clamp(1e-12, 1.0)
    return torch.pow(ratio, psi)


def corridor_gate(dist_stream_m, w_max_m):
    """Flood → landslide: only slopes within ``w_max`` of the channel.

    A flood undercuts the bank it touches. A hillslope 3 km away does not care
    that the river rose, so the F→L kernel is restricted to the corridor rather
    than being allowed to reach the whole basin (Doc 02 §5, Doc 07 §5 item 4).
    """
    import torch
    return torch.sigmoid((w_max_m - dist_stream_m) / 50.0)


# ----------------------------------------------------------- time kernels ---

def exponential_kernel(dt, theta):
    """``exp(-dt / theta) / theta``, zero for ``dt < 0``.

    Normalised to unit mass so that ``alpha`` alone carries the strength and
    ``theta`` alone carries the timescale. Without the ``1/theta`` the two are
    confounded and neither is interpretable on its own.
    """
    import torch
    out = torch.exp(-dt.clamp_min(0.0) / theta) / theta
    return torch.where(dt >= 0.0, out, torch.zeros_like(out))


def gamma_kernel(dt, shape, scale):
    """A delayed, humped response — the landslide-dam breach mechanism.

    An exponential peaks at ``dt = 0``: maximum risk the instant the landslide
    happens. A dam breach does the opposite, building for days or weeks before
    releasing. The gamma with ``shape > 1`` rises then falls, and is the only
    component in the mixture that can express "later, not sooner".
    """
    import torch

    d = dt.clamp_min(1e-9)
    logpdf = ((shape - 1.0) * torch.log(d) - d / scale
              - torch.lgamma(shape) - shape * torch.log(scale))
    out = torch.exp(logpdf.clamp(-60.0, 60.0))
    return torch.where(dt > 0.0, out, torch.zeros_like(out))


def mechanism_mixture(dt, params, version: int = 1):
    """Doc 02 §4.4 / Doc 07 §6.2 — the three physical mechanisms.

    ==== =============================== =======================================
    v    components                      physical reading
    ==== =============================== =======================================
    1    one exponential                 "something happens, then decays"
    2    fast + slow exponential          immediate debris pulse; then weeks of
                                          elevated sediment supply
    3    + delayed gamma                  plus a landslide-dam breach, which
                                          arrives *later*, not sooner
    ==== =============================== =======================================

    Doc 07 §6.2 gates the version on the evidence available: v2 needs at least
    30 identified cascade pairs, v3 at least 100 and a stable v2. Fitting three
    timescales to twelve pairs produces three confident numbers that mean
    nothing.

    Weights are a softmax over free logits, so they sum to one by construction
    and the optimiser never has to be told about the constraint.
    """
    import torch

    if version == 1:
        return exponential_kernel(dt, params["theta1"])

    w = torch.softmax(torch.stack([params["w_logit1"], params["w_logit2"]]
                                  + ([params["w_logit3"]] if version >= 3 else [])), 0)
    out = w[0] * exponential_kernel(dt, params["theta1"])
    out = out + w[1] * exponential_kernel(dt, params["theta2"])
    if version >= 3:
        out = out + w[2] * gamma_kernel(dt, params["gamma_shape"], params["theta3"])
    return out


#: Doc 07 §6.2 bounds each timescale to its physical range, in **days**. A fit
#: that wants theta1 = 40 days is not describing a debris pulse, and letting it
#: go there makes the mixture unidentifiable because two components collapse
#: onto the same timescale.
TIMESCALE_BOUNDS = {
    "theta1": (1.0 / 24.0, 1.0),        # 1 hour to 1 day   — debris pulse
    "theta2": (1.0, 14.0),              # 1 to 14 days      — sediment supply
    "theta3": (14.0, 180.0),            # 2 weeks to 6 mo   — dam breach
}


# ------------------------------------------------------------ travel time ---

def arrival_shift(t_ref_m, v0_ms):
    """``tau = T_ref / v0``, in **days** — Doc 05 §5.1.

    ``T_ref`` is the stored path integral ``sum(length / S^0.3)`` in metres, so
    dividing by ``v0`` gives seconds and the table never has to be rebuilt when
    the optimiser moves ``v0``. That is the entire reason Phase 2 stored the
    integral rather than a time.
    """
    return t_ref_m / v0_ms / SECONDS_PER_DAY


#: Doc 02 §5.4 / Doc 07 §6.1: v0 is a *coefficient*, not a speed. The celerity
#: is v = v0 * S^0.3 and the median reach slope gives S^0.3 ~ 0.215, so the
#: 1-5 m/s physical range corresponds to v0 in roughly 4.6-23. Bounding v0 at
#: [1, 5] instead — the obvious mistake — would force every flood wave below
#: 1.1 m/s and silently break the timing prediction.
V0_BOUNDS = (4.0, 24.0)


def celerity_from_v0(v0_ms: float, reach_slope: np.ndarray,
                     exponent: float = 0.3) -> np.ndarray:
    """The actual wave speeds a given ``v0`` implies, for the physical check."""
    return float(v0_ms) * np.power(np.asarray(reach_slope, dtype=np.float64),
                                   exponent)


def check_physical_celerity(v0_ms: float, reach_slope: np.ndarray,
                            lo: float = 1.0, hi: float = 5.0) -> tuple[bool, str]:
    """Doc 07 §10 gate 7: is the fitted ``v0`` physically plausible?"""
    v = celerity_from_v0(v0_ms, reach_slope)
    med = float(np.median(v))
    p5, p95 = (float(x) for x in np.percentile(v, [5, 95]))
    ok = lo <= med <= hi
    return ok, (f"v0 = {v0_ms:.2f} gives median celerity {med:.2f} m/s "
                f"(5-95th {p5:.2f}-{p95:.2f}); "
                f"{'inside' if ok else 'OUTSIDE'} the physical range "
                f"{lo}-{hi} m/s")
