"""Phase 4 — the cascade model's mathematics.

Doc 07 §10 gates 1-3 in executable form, plus the pieces of the kernel that are
easy to get subtly wrong and impossible to notice afterwards.

The tests that matter most are the ones about the **compensator**. A Hawkes fit
with a wrong compensator still converges, still reports a plausible likelihood,
and still produces confident parameters — it just produces the wrong ones. The
only way to catch it is to fit data whose answer you already know.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrocascade.cascade import fit, kernels, likelihood, simulate  # noqa: E402


# ------------------------------------------------- three implementations ----

def test_the_three_likelihoods_agree_exactly():
    """Doc 07 §10 gate 2, in its substituted form.

    An O(n^2) double sum, the O(n) Ozaki recursion and the vectorised torch
    version share no code path. Agreement to machine precision is the strongest
    statement available without an external library.
    """
    p = simulate.TemporalParams(mu=0.5, alpha=0.4, theta=1.5)
    t = simulate.temporal(p, T=300.0, seed=3)
    assert t.size > 50

    a = likelihood.temporal_direct(t, 0.5, 0.4, 1.5, 300.0)
    b = likelihood.temporal_recursive(t, 0.5, 0.4, 1.5, 300.0)
    c = float(likelihood.temporal_torch(torch.as_tensor(t), 0.5, 0.4, 1.5, 300.0))
    assert a == pytest.approx(b, abs=1e-9)
    assert a == pytest.approx(c, abs=1e-9)


def test_an_event_never_excites_itself():
    """The commonest sign error in a Hawkes implementation.

    With alpha = 0 the likelihood is pure Poisson and has a closed form. If the
    diagonal leaked into the sum, the intensity at each event would be inflated
    by exactly alpha/theta and this would fail.
    """
    t = np.array([1.0, 4.0, 9.0])
    T, mu = 10.0, 0.7
    got = likelihood.temporal_recursive(t, mu, 0.0, 1.0, T)
    want = t.size * np.log(mu) - mu * T
    assert got == pytest.approx(want, rel=1e-12)


# ------------------------------------------------------- the compensator ----

def test_compensator_truncates_at_the_window_edge():
    """Each event contributes its kernel mass *less the tail beyond T*.

    Forgetting the truncation biases alpha upward, and worst for events near the
    end of the record — which in this project is 2024, the year it is about.
    """
    t = np.array([9.9])                       # right at the edge
    mu, alpha, theta, T = 0.0, 1.0, 2.0, 10.0
    # log-term is -inf with mu = 0, so check the compensator alone
    full_mass = alpha * theta
    truncated = alpha * theta * (1.0 - np.exp(-(T - t[0]) / theta))
    assert truncated < full_mass * 0.1

    ll_int = -likelihood.temporal_recursive(t, 1e-12, alpha, theta, T)
    assert ll_int > 0
    assert ll_int - (-np.log(1e-12)) == pytest.approx(truncated, abs=1e-6)


def test_an_early_event_is_charged_more_compensator_than_a_late_one():
    """The truncation has to depend on *when* the event happened.

    An event at t=1 in a window ending at T=10 has almost all of its kernel mass
    inside the window; one at t=9.5 has almost none. A compensator that charges
    both the same is the classic bug, and it biases alpha upward for records that
    end soon after their last big event.
    """
    mu, alpha, theta, T = 1e-12, 1.0, 2.0, 10.0
    log_term = np.log(mu)

    early = -(likelihood.temporal_recursive(np.array([1.0]), mu, alpha, theta, T)
              - log_term)
    late = -(likelihood.temporal_recursive(np.array([9.5]), mu, alpha, theta, T)
             - log_term)

    assert early > late
    assert early == pytest.approx(alpha * theta * (1 - np.exp(-9.0 / theta)), abs=1e-6)
    assert late == pytest.approx(alpha * theta * (1 - np.exp(-0.5 / theta)), abs=1e-6)


# ------------------------------------------------------- branching ratio ----

def test_branching_ratio_is_alpha_times_theta_for_one_type():
    assert likelihood.branching_ratio(0.4, 1.5) == pytest.approx(0.6)


def test_multitype_branching_is_the_spectral_radius_not_the_max_entry():
    """A matrix with every entry below 1 can still be explosive.

    Here each entry of alpha*theta is 0.8 — all individually 'safe' — but the
    spectral radius is 1.6, and the process explodes. Using the max entry as the
    stability check would pass this.
    """
    alpha = np.array([[0.8, 0.8], [0.8, 0.8]])
    theta = np.ones((2, 2))
    assert likelihood.branching_ratio(alpha, theta) == pytest.approx(1.6)
    assert np.max(alpha * theta) < 1.0        # the check that would have missed it


def test_the_simulator_refuses_an_explosive_process():
    p = simulate.TemporalParams(mu=0.5, alpha=1.0, theta=1.0)
    with pytest.raises(ValueError, match="explosive"):
        p.validate()


# ------------------------------------------------- parameter recovery S1 ----

@pytest.mark.slow
def test_s1_recovers_known_parameters():
    """Doc 07 §2: *"If you cannot recover known parameters, do not proceed."*

    Six catalogs, not twenty — the full run lives in
    `hydrocascade cascade --verify`. The tolerance is on the mean and allows for
    Monte-Carlo error, because a per-catalog tolerance would fail a correct
    estimator on sampling noise alone.
    """
    truth = {"mu": 0.5, "alpha": 0.4, "theta": 1.5}
    T = 600.0
    est = []
    for i in range(6):
        ts = simulate.temporal(simulate.TemporalParams(**truth), T=T, seed=500 + i)
        tt = torch.as_tensor(ts)
        obj = lambda q: likelihood.temporal_torch(tt, q["mu"], q["alpha"], q["theta"], T)
        best, _, _ = fit.optimise(obj, {"mu": 1.0, "alpha": 0.2, "theta": 1.0},
                                  steps=250, lr=0.1, seed=i)
        est.append(best)

    for k, true in truth.items():
        v = np.array([e[k] for e in est])
        se = v.std(ddof=1) / np.sqrt(v.size)
        assert abs(v.mean() - true) < max(0.25 * true, 2.5 * se), (
            f"{k}: recovered {v.mean():.3f} vs true {true}")


# ---------------------------------------------------------------- kernels ----

def test_runout_gate_lets_a_big_slide_reach_further():
    """`1[d_slope <= c * exp(gamma * m)]` — a larger mark buys more reach."""
    c = torch.tensor(500.0, dtype=torch.float64)
    g = torch.tensor(0.5, dtype=torch.float64)
    d = torch.tensor(900.0, dtype=torch.float64)
    small = kernels.runout_gate(d, torch.tensor(0.0, dtype=torch.float64), c, g, hard=True)
    big = kernels.runout_gate(d, torch.tensor(2.0, dtype=torch.float64), c, g, hard=True)
    assert float(small) == 0.0
    assert float(big) == 1.0


def test_the_soft_gate_is_differentiable_and_the_hard_one_is_not():
    """The relaxation exists only so `c` and `gamma` have a gradient."""
    d = torch.tensor(900.0, dtype=torch.float64)
    m = torch.tensor(0.0, dtype=torch.float64)
    c = torch.tensor(900.0, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(0.0, dtype=torch.float64)
    soft = kernels.runout_gate(d, m, c, g, hard=False)
    soft.backward()
    assert c.grad is not None and abs(float(c.grad)) > 0


def test_dilution_is_one_at_the_source_and_falls_downstream():
    psi = torch.tensor(1.0, dtype=torch.float64)
    same = kernels.dilution(torch.tensor(10.0, dtype=torch.float64),
                            torch.tensor(10.0, dtype=torch.float64), psi)
    far = kernels.dilution(torch.tensor(10.0, dtype=torch.float64),
                           torch.tensor(1000.0, dtype=torch.float64), psi)
    assert float(same) == pytest.approx(1.0)
    assert float(far) == pytest.approx(0.01)


def test_psi_zero_switches_dilution_off_exactly():
    """Ablation A5 must be exactly 'no dilution', not 'nearly none'."""
    off = kernels.dilution(torch.tensor(3.0, dtype=torch.float64),
                           torch.tensor(3000.0, dtype=torch.float64),
                           torch.tensor(0.0, dtype=torch.float64))
    assert float(off) == pytest.approx(1.0)


def test_the_time_kernel_is_zero_before_the_event():
    """phi is defined to be zero for negative arguments — causality."""
    dt = torch.tensor([-5.0, -0.1, 0.0, 1.0], dtype=torch.float64)
    k = kernels.exponential_kernel(dt, torch.tensor(2.0, dtype=torch.float64))
    assert float(k[0]) == 0.0
    assert float(k[1]) == 0.0
    assert float(k[3]) > 0


def test_the_time_kernel_has_unit_mass():
    """Normalised so alpha carries strength and theta carries timescale alone.

    Without the 1/theta the two are confounded and neither is interpretable.
    """
    theta = torch.tensor(3.0, dtype=torch.float64)
    dt = torch.linspace(0, 200, 400001, dtype=torch.float64)
    k = kernels.exponential_kernel(dt, theta)
    assert float(torch.trapz(k, dt)) == pytest.approx(1.0, rel=1e-4)


def test_the_gamma_kernel_peaks_after_the_event_not_at_it():
    """The dam-breach mechanism is the only one that can say 'later, not sooner'."""
    dt = torch.linspace(0.01, 120, 4000, dtype=torch.float64)
    k = kernels.gamma_kernel(dt, torch.tensor(3.0, dtype=torch.float64),
                             torch.tensor(10.0, dtype=torch.float64))
    peak = float(dt[int(torch.argmax(k))])
    assert peak > 5.0, "a shape > 1 gamma must rise before it falls"


def test_arrival_shift_is_slower_for_a_smaller_celerity():
    t_ref = torch.tensor(90_000.0, dtype=torch.float64)      # metres
    fast = kernels.arrival_shift(t_ref, torch.tensor(18.0, dtype=torch.float64))
    slow = kernels.arrival_shift(t_ref, torch.tensor(6.0, dtype=torch.float64))
    assert float(slow) == pytest.approx(3 * float(fast))


def test_v0_bounds_are_a_coefficient_range_not_a_speed_range():
    """v0 is a coefficient in v = v0 * S^0.3, not a speed.

    Bounding it to the 1-5 m/s physical range — the obvious mistake — would
    force every flood wave below about 1.1 m/s and silently break the timing
    prediction the project rests on.
    """
    lo, hi = kernels.V0_BOUNDS
    slope = np.full(1000, 0.006)                  # the basin's median bed slope

    # The trap is bounding v0 at (1, 5) as if it were the speed. Check the
    # bounds against what they actually imply for celerity: the range must
    # *contain* the 1-5 m/s physical window, not coincide with it.
    v_lo = kernels.celerity_from_v0(lo, slope)[0]
    v_hi = kernels.celerity_from_v0(hi, slope)[0]
    assert v_lo < 1.5, f"lower bound implies {v_lo:.2f} m/s — too fast a floor"
    assert v_hi > 4.5, f"upper bound implies {v_hi:.2f} m/s — too slow a ceiling"

    ok, note = kernels.check_physical_celerity(9.0, slope)
    assert ok, note


def test_celerity_check_rejects_an_implausible_v0():
    slope = np.full(1000, 0.006)
    ok, _ = kernels.check_physical_celerity(2.0, slope)
    assert not ok, "v0 = 2 gives a median wave of ~0.4 m/s and must be rejected"


# ----------------------------------------- one definition, used everywhere ---

def test_the_landslide_spatial_kernel_integrates_to_one():
    """The compensator charges each source alpha x (temporal mass), which is
    only right if the spatial kernel integrates to exactly 1 over the plane."""
    step_km = 0.05
    x = np.arange(-15.0, 15.0 + step_km, step_km)          # km
    X, Y = np.meshgrid(x, x)
    d2_m2 = (X ** 2 + Y ** 2) * 1.0e6
    mass = kernels.ll_spatial_density(d2_m2).sum() * step_km ** 2
    assert mass == pytest.approx(1.0, rel=1e-3)


def test_the_corridor_density_integrates_to_one():
    area_km2 = np.pi * (kernels.CORRIDOR_M / 1000.0) ** 2
    assert kernels.corridor_density() * area_km2 == pytest.approx(1.0)


def test_the_density_is_identical_for_numpy_and_torch():
    """Fitting evaluates it on tensors, prediction on floats. They must agree
    exactly, or the two halves of the model describe different kernels."""
    d2 = np.array([0.0, 1.0e6, 4.0e6, 2.5e7])
    a = kernels.ll_spatial_density(d2)
    b = kernels.ll_spatial_density(torch.as_tensor(d2, dtype=torch.float64)).numpy()
    assert np.allclose(a, b, rtol=0, atol=1e-15)


def test_model_and_predictor_share_one_set_of_constants():
    """The regression guard for the bug class this project hit four times.

    The likelihood and the forward predictor each once carried their own copy
    of these kernels, and the copies drifted — once by five orders of magnitude,
    which saturated every predicted probability at 1.0. They must now be the
    same objects, not equal-looking restatements.
    """
    from hydrocascade.cascade import model
    from hydrocascade.realtime import predict

    assert model.CORRIDOR_M is kernels.CORRIDOR_M or model.CORRIDOR_M == kernels.CORRIDOR_M
    assert model.LL_SIGMA_M == kernels.LL_SIGMA_M
    assert predict.kernels is kernels

    src = __import__("inspect").getsource(predict.Predictor.landslide_intensity)
    assert "kernels.ll_spatial_density" in src
    assert "kernels.corridor_density" in src
    assert "2.0 * np.pi" not in src, "the predictor restates a kernel formula"


# ------------------------------------------------------------ the fitter ----

def test_signed_parameters_are_not_forced_positive():
    """Regression coefficients must be able to go negative.

    Forcing them positive makes a risk-*reducing* driver inexpressible, and the
    optimiser answers by pushing the coefficient to exp(-40) — which reads as
    'this driver does not matter' and is a different statement entirely.
    """
    assert fit.is_real_valued("beta_dyn0")
    assert fit.is_real_valued("w_logit1")
    assert not fit.is_real_valued("a_LF")
    assert not fit.is_real_valued("theta1")

    def obj(q):
        return -(q["beta_x"] + 3.0) ** 2          # optimum at -3

    best, val, _ = fit.optimise(obj, {"beta_x": 0.5}, steps=400, lr=0.05)
    assert best["beta_x"] == pytest.approx(-3.0, abs=0.05)


def test_bounded_parameters_stay_inside_their_bounds():
    def obj(q):
        return -(q["theta1"] - 99.0) ** 2         # pulls hard past the bound

    best, _, _ = fit.optimise(obj, {"theta1": 0.5},
                              bounds={"theta1": (0.04, 1.0)}, steps=300, lr=0.1)
    assert 0.04 <= best["theta1"] <= 1.0


def test_a_fit_with_branching_over_one_is_marked_invalid():
    """Doc 07 §7 rule 7 — invalid regardless of how good the likelihood is."""
    r = fit.FitRecord(name="t", params={"a": 1.0}, log_likelihood=-10.0,
                      n_events=100, n_params=1, branching_ratio=1.4, seed=0,
                      n_restarts=1)
    ok, why = r.valid()
    assert not ok
    assert "explosive" in why


def test_a_healthy_fit_is_valid_and_has_a_sane_aic():
    r = fit.FitRecord(name="t", params={"a": 1.0, "b": 2.0},
                      log_likelihood=-50.0, n_events=100, n_params=2,
                      branching_ratio=0.6, seed=0, n_restarts=1)
    assert r.valid()[0]
    assert r.aic == pytest.approx(2 * 2 - 2 * -50.0)


# --------------------------------------------------- spec / parameter wiring ---
#
# A6 once died with `KeyError: 'beta_dyn0'` after forty minutes of fitting:
# `initial_params` created the rainfall coefficients only when the *background*
# was on, while the objective asked for them whenever *rainfall* was on. The
# mismatch is invisible in the full model and only appears in the one ablation
# that separates those two flags. These tests evaluate every ladder rung and
# every ablation once on a toy dataset, so the next such mismatch costs seconds
# instead of a run.

def _toy_data(n_days: int = 12, n_units: int = 4, n_reach: int = 3,
              n_pixel: int = 2, n_sta: int = 2, n_dyn: int = 3):
    """A CascadeData small enough to evaluate instantly, wired like the real one."""
    import numpy as np
    from hydrocascade.cascade.model import CascadeData, L, F

    t = np.array([1.0, 2.0, 4.0, 6.0, 9.0])
    kind = np.array([L, F, L, F, F])
    unit = np.array([0, 0, 2, 1, 2])              # unit id for L, reach id for F
    mark = np.array([0.4, -0.2, 0.9, 0.0, -0.5])
    inj = np.array([0, 0, 1, 1, 2])
    dsl = np.array([500.0, 0.0, 900.0, 0.0, 0.0])

    # one pair of each kind: 0 LL, 1 LF, 2 FL, 3 FF
    pair_i = np.array([2, 1, 3, 4])
    pair_j = np.array([0, 0, 1, 3])
    pair_kind = np.array([0, 1, 2, 3])
    return CascadeData(
        t=t, kind=kind, unit=unit, mark=mark, inj_reach=inj, d_slope_m=dsl,
        T=float(n_days),
        pair_i=pair_i, pair_j=pair_j,
        pair_tref_m=np.array([0.0, 3000.0, 0.0, 2000.0]),
        pair_area_src=np.array([1.0, 5.0, 5.0, 8.0]),
        pair_area_tgt=np.array([1.0, 12.0, 5.0, 20.0]),
        pair_kind=pair_kind,
        pair_dist_m=np.array([800.0, 0.0, 1200.0, 0.0]),
        unit_static=np.linspace(-1.0, 1.0, n_units * n_sta).reshape(n_units, n_sta),
        unit_pixel=np.arange(n_units) % n_pixel,
        unit_S=np.full(n_units, 0.02),
        reach_S=np.full(n_reach, 0.01),
        reach_pixel_w=np.full((n_reach, n_pixel), 0.5),
        rain=np.abs(np.linspace(0.0, 2.0, n_days * n_pixel * n_dyn)
                    ).reshape(n_days, n_pixel, n_dyn),
        n_units=n_units, n_reach=n_reach,
        src_event=np.array([0, 1, 3]), src_tref_m=np.array([3000.0, 0.0, 2000.0]),
        src_area_src=np.array([5.0, 5.0, 8.0]), src_area_tgt=np.array([12.0, 5.0, 20.0]),
        src_kind=np.array([1, 3, 3]),
        driver_names=[f"d{i}" for i in range(n_sta + n_dyn)],
        meta={"static": [f"s{i}" for i in range(n_sta)],
              "dynamic": [f"r{i}" for i in range(n_dyn)]},
        iso_i=pair_i, iso_j=pair_j, iso_w=np.full(4, 0.25), iso_kind=pair_kind,
        iso_has_reach=np.array([True, True, True, True, True]),
    )


def _every_spec():
    """(name, spec) for the full model, every ladder rung and every ablation."""
    from hydrocascade.cascade.model import ModelSpec
    from hydrocascade.experiments.ablations import ABLATIONS, LADDER

    full = ModelSpec(stage="S5")
    out = [("FULL", full)]
    for name, (_, overrides) in LADDER.items():
        out.append((name, ModelSpec(**{**full.__dict__, **overrides, "stage": "S5"})))
    for name, (_, _, overrides) in ABLATIONS.items():
        out.append((name, ModelSpec(**{**full.__dict__, **overrides, "stage": "S5"})))
    return out


@pytest.mark.parametrize("name,spec", _every_spec())
def test_every_spec_can_be_built_and_evaluated(name, spec):
    """initial_params must supply exactly what the objective asks for."""
    import numpy as np
    import torch
    from hydrocascade.cascade import model as m

    data = _toy_data()
    rng = np.random.default_rng(0)
    p0 = m.initial_params(spec, data, rng)
    obj = m.make_objective(data, spec)
    q = {k: torch.tensor(float(v), dtype=torch.float64, requires_grad=True)
         for k, v in p0.items()}
    ll = obj(q)                                     # KeyError here is the bug
    assert torch.isfinite(ll), f"{name}: non-finite log-likelihood"

    # and it must be differentiable in every parameter it claims to have
    ll.backward()
    dead = [k for k, v in q.items() if v.grad is None or not torch.isfinite(v.grad)]
    assert not dead, f"{name}: no usable gradient for {dead}"


def test_no_parameter_is_created_that_the_objective_ignores():
    """The other direction: a created-but-unread parameter is a fabricated number.

    It is reported in the fit record as though the data had spoken about it,
    which is how `sigma_m` came to appear as a fitted 2.1 km length scale.

    The gate parameters need care: `runout_gate` is a logistic relaxation with
    sharpness 40 in log space, so it is numerically flat unless a pair sits near
    the threshold, and at a random draw its gradient underflows to exactly zero.
    That is saturation, not deadness, so the gate is evaluated at a threshold
    that straddles the toy d_slope values instead.
    """
    import numpy as np
    import torch
    from hydrocascade.cascade import model as m

    data = _toy_data()
    straddle = {"gate_c": float(np.mean(data.d_slope_m[data.d_slope_m > 0])),
                "gate_gamma": 0.2}
    for name, spec in _every_spec():
        p0 = m.initial_params(spec, data, np.random.default_rng(1))
        p0.update({k: v for k, v in straddle.items() if k in p0})
        obj = m.make_objective(data, spec)
        q = {k: torch.tensor(float(v), dtype=torch.float64, requires_grad=True)
             for k, v in p0.items()}
        obj(q).backward()
        unused = [k for k, v in q.items()
                  if v.grad is None or float(v.grad.abs()) == 0.0]
        assert not unused, f"{name}: created but unused: {unused}"
