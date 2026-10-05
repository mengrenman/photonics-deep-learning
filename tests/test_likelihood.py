"""Tests for bayes_scatter.likelihood: SNR definition, noise statistics, misfit and the Target.

The Target tests use a small problem (8 x 8 grid, side 1, the book's antenna rings with 6 transmitters and
10 receivers) with the full forward model, the log_eps link and a Matern prior with a nonzero mean, so that
every link in the chain xi -> theta -> chi -> d -> Phi is exercised.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import Target, add_noise, misfit, noise_std_from_snr
from bayes_scatter.priors import GaussianField

# ---------------------------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------------------------


def _random_data(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("snr_db", [-20.0, 0.0, 17.3, 40.0])
def test_noise_std_is_power_snr_not_amplitude_snr(snr_db):
    """s^2 = mean|d|^2 10^(-SNR/10): at 20 dB the noise power is 1% of the signal power (an amplitude
    definition would give 10%, a factor-2 slip in the exponent 0.1 or 0.0316 ...)."""
    d = 0.3 * _random_data((7, 11), seed=1)
    s = noise_std_from_snr(d, snr_db)
    assert isinstance(s, float)
    np.testing.assert_allclose(s**2, np.mean(np.abs(d) ** 2) * 10.0 ** (-snr_db / 10.0), rtol=1e-13)
    # scale equivariance: doubling the data doubles s
    np.testing.assert_allclose(noise_std_from_snr(2.0 * d, snr_db), 2.0 * s, rtol=1e-13)
    np.testing.assert_allclose(noise_std_from_snr(d, 20.0) ** 2 / np.mean(np.abs(d) ** 2), 0.01, rtol=1e-13)


def test_add_noise_empirical_power_matches_snr_and_is_circular():
    """Re and Im parts are independent N(0, s^2/2): E|e|^2 = s^2, so the empirical SNR is the requested one."""
    d = 0.5 * _random_data((400, 500), seed=2)
    snr_db = 12.0
    s = noise_std_from_snr(d, snr_db)
    e = np.asarray(add_noise(jax.random.key(3), d, s)) - d
    n = e.size
    assert e.shape == d.shape and np.iscomplexobj(e)
    assert abs(np.mean(np.abs(e) ** 2) / s**2 - 1.0) < 0.02  # 2e5 samples: relative sd 0.2%
    assert abs(np.var(e.real) / (0.5 * s**2) - 1.0) < 0.02
    assert abs(np.var(e.imag) / (0.5 * s**2) - 1.0) < 0.02
    assert abs(np.mean(e.real * e.imag)) < 0.02 * 0.5 * s**2  # uncorrelated
    assert abs(np.mean(e.real)) < 5 * s / np.sqrt(n) and abs(np.mean(e.imag)) < 5 * s / np.sqrt(n)
    empirical_snr = 10.0 * np.log10(np.mean(np.abs(d) ** 2) / np.mean(np.abs(e) ** 2))
    assert abs(empirical_snr - snr_db) < 0.05
    # Re e is Gaussian: kurtosis ~ 3 and ~5% beyond 1.96 sd
    z = e.real / np.sqrt(0.5 * s**2)
    assert abs(np.mean(np.abs(z) > 1.96) - 0.05) < 0.005


def test_add_noise_is_deterministic_in_the_key():
    d = _random_data((5, 6), seed=4)
    a = add_noise(jax.random.key(0), d, 0.1)
    b = add_noise(jax.random.key(0), d, 0.1)
    c = add_noise(jax.random.key(1), d, 0.1)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert np.max(np.abs(np.asarray(a) - np.asarray(c))) > 1e-3
    assert np.max(np.abs(np.asarray(a) - d)) > 1e-3  # noise was actually added


# ---------------------------------------------------------------------------------------------
# Misfit
# ---------------------------------------------------------------------------------------------


def test_misfit_is_the_negative_log_likelihood_up_to_a_constant():
    """-log p(d_obs | d_model) for CN(0, s^2) noise, written as a density of the stacked real and imaginary
    parts (each N(0, s^2/2)), equals misfit + N log(pi s^2) with N complex entries."""
    d_model = _random_data((6, 9), seed=5)
    d_obs = _random_data((6, 9), seed=6)
    s = 0.37
    sd = s / np.sqrt(2.0)
    logpdf = (norm.logpdf(d_obs.real, loc=d_model.real, scale=sd).sum()
              + norm.logpdf(d_obs.imag, loc=d_model.imag, scale=sd).sum())
    nll = -logpdf
    np.testing.assert_allclose(float(misfit(d_model, d_obs, s)) + d_obs.size * np.log(np.pi * s**2), nll,
                               rtol=1e-12)
    # the constant does not depend on d_model: differences of misfits are differences of NLLs
    d_model2 = _random_data((6, 9), seed=7)
    logpdf2 = (norm.logpdf(d_obs.real, loc=d_model2.real, scale=sd).sum()
               + norm.logpdf(d_obs.imag, loc=d_model2.imag, scale=sd).sum())
    np.testing.assert_allclose(float(misfit(d_model2, d_obs, s) - misfit(d_model, d_obs, s)), logpdf - logpdf2,
                               rtol=1e-11)


def test_misfit_scaling_and_zero_and_gradient_at_zero_residual():
    d = _random_data((4, 5), seed=8)
    assert float(misfit(d, d, 0.2)) == 0.0
    r = _random_data((4, 5), seed=9)
    np.testing.assert_allclose(float(misfit(d + r, d, 0.5)), np.sum(np.abs(r) ** 2) / 0.25, rtol=1e-13)
    np.testing.assert_allclose(float(misfit(d + r, d, 1.0)), 4.0 * float(misfit(d + r, d, 2.0)), rtol=1e-13)
    g = jax.grad(lambda x: misfit(x, d, 0.3), argnums=0, holomorphic=False)(jnp.asarray(d.real))
    assert np.all(np.isfinite(np.asarray(g)))
    g0 = jax.grad(lambda c: misfit(c * jnp.ones((3,)), jnp.zeros((3,)), 1.0))(jnp.asarray(0.0))
    assert float(g0) == 0.0  # finite (and zero) gradient at a zero residual, where |r| has a kink


# ---------------------------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------------------------

N_GRID = 8


@pytest.fixture(scope="module")
def problem():
    setup = ch15_setup(n=N_GRID, side=1.0, n_tx=6, n_rx=10)
    fwd = Forward(setup, tol=1e-12)
    field = GaussianField(setup.grid, "matern", sigma=0.4, ell=0.3, nu=1.5, mean=0.2)
    xi_true = field.sample_xi(jax.random.key(0))
    chi_true = jnp.exp(field.theta(xi_true)) - 1.0
    d_clean, _ = fwd(chi_true)
    s = noise_std_from_snr(d_clean, 20.0)
    noise = add_noise(jax.random.key(1), d_clean, s) - d_clean
    d_obs = d_clean + noise
    xi = 0.7 * xi_true + 0.3 * field.sample_xi(jax.random.key(2))  # a point away from the truth
    return {"setup": setup, "fwd": fwd, "field": field, "xi_true": xi_true, "xi": xi, "d_clean": d_clean,
            "d_obs": d_obs, "s": s, "noise": noise}


def _target(p, link="log_eps", use_born=False):
    return Target(p["fwd"], p["field"], link, p["d_obs"], p["s"], use_born=use_born)


def test_potential_equals_explicit_forward_and_misfit(problem):
    """Phi(xi) computed by hand: chi = exp(theta) - 1 with theta = mu + C^(1/2) xi, d = F(chi), then misfit."""
    p = problem
    tgt = _target(p)
    chi = jnp.exp(p["field"].theta(p["xi"])) - 1.0
    d, _ = p["fwd"](chi)
    expected = float(misfit(d, p["d_obs"], p["s"]))
    phi, aux = tgt.potential(p["xi"], tgt.initial_aux())
    np.testing.assert_allclose(float(phi), expected, rtol=1e-11)
    phi2, grad, aux2 = tgt.potential_and_grad(p["xi"], tgt.initial_aux())
    np.testing.assert_allclose(float(phi2), expected, rtol=1e-11)
    np.testing.assert_allclose(float(tgt.phi_fn(p["xi"], tgt.initial_aux())[0]), expected, rtol=1e-11)
    np.testing.assert_allclose(np.asarray(tgt.chi(p["xi"])), np.asarray(chi), atol=1e-13)
    np.testing.assert_allclose(np.asarray(tgt.eps_r(p["xi"])), np.asarray(1.0 + chi), atol=1e-13)
    assert grad.shape == p["field"].xi_shape and aux.shape == (6, N_GRID, N_GRID)
    # Born variant uses forward.born
    tgt_b = _target(p, use_born=True)
    expected_b = float(misfit(p["fwd"].born(chi), p["d_obs"], p["s"]))
    np.testing.assert_allclose(float(tgt_b.potential(p["xi"], tgt_b.initial_aux())[0]), expected_b, rtol=1e-11)
    assert abs(expected_b - expected) > 0.05 * expected  # nonlinear and Born models really differ here


def test_phi_at_truth_is_the_noise_energy_over_s_squared(problem):
    """At the true xi the model reproduces the clean data, so Phi = sum |e|^2 / s^2 (mean N = 60 complex
    entries; a factor 1/2 or 2 in Phi would not give this identity)."""
    p = problem
    tgt = _target(p)
    phi = float(tgt.potential(p["xi_true"], tgt.initial_aux())[0])
    np.testing.assert_allclose(phi, np.sum(np.abs(np.asarray(p["noise"])) ** 2) / p["s"] ** 2, rtol=1e-8)


@pytest.mark.parametrize("use_born", [False, True], ids=["full", "born"])
def test_gradient_matches_central_finite_differences(problem, use_born):
    """grad Phi along 4 random unit directions against central differences (forward tol 1e-12)."""
    p = problem
    tgt = _target(p, use_born=use_born)
    aux = tgt.initial_aux()
    _, grad, aux = tgt.potential_and_grad(p["xi"], aux)
    rng = np.random.default_rng(3)
    eps = 1e-5
    for _ in range(4):
        v = rng.standard_normal(p["field"].xi_shape)
        v /= np.linalg.norm(v)
        v = jnp.asarray(v)
        fd = (tgt.potential(p["xi"] + eps * v, aux)[0] - tgt.potential(p["xi"] - eps * v, aux)[0]) / (2 * eps)
        ad = jnp.vdot(grad, v)
        assert abs(float(fd) - float(ad)) < 1e-5 * max(abs(float(fd)), abs(float(ad))), (float(fd), float(ad))
    # not a trivial gradient
    assert float(jnp.linalg.norm(grad)) > 1.0


def test_gradient_through_all_links_with_identity_link_too(problem):
    p = problem
    tgt = _target(p, link="identity")
    aux = tgt.initial_aux()
    xi = 0.3 * p["xi"]
    _, grad, aux = tgt.potential_and_grad(xi, aux)
    v = jnp.asarray(np.random.default_rng(4).standard_normal(p["field"].xi_shape))
    v = v / jnp.linalg.norm(v)
    eps = 1e-5
    fd = (tgt.potential(xi + eps * v, aux)[0] - tgt.potential(xi - eps * v, aux)[0]) / (2 * eps)
    assert abs(float(fd) - float(jnp.vdot(grad, v))) < 1e-5 * abs(float(fd))


def test_warm_start_does_not_change_phi_or_gradient_beyond_tolerance(problem):
    p = problem
    tgt = _target(p)
    phi0, g0, aux_here = tgt.potential_and_grad(p["xi"], tgt.initial_aux())
    # a stale aux from a quite different point (the truth) and a zero aux
    _, _, aux_stale = tgt.potential_and_grad(p["xi_true"], tgt.initial_aux())
    for aux in (aux_stale, aux_here, jnp.zeros_like(aux_here)):
        phi1, g1, _ = tgt.potential_and_grad(p["xi"], aux)
        assert abs(float(phi1) - float(phi0)) < 1e-7 * float(phi0)
        assert float(jnp.linalg.norm(g1 - g0)) < 1e-7 * float(jnp.linalg.norm(g0))
    # the warm start really differs from the cold start (otherwise the test proves nothing)
    assert float(jnp.max(jnp.abs(aux_stale - aux_here))) > 1e-3


def test_returned_aux_is_the_total_field_of_this_solve(problem):
    p = problem
    tgt = _target(p)
    chi = tgt.chi(p["xi"])
    _, _, aux = tgt.potential_and_grad(p["xi"], tgt.initial_aux())
    assert aux.shape == tgt.initial_aux().shape and aux.dtype == jnp.complex128
    assert float(jnp.max(p["fwd"].residual(chi, aux))) < 1e-10
    assert tgt.solver_residual(p["xi"], aux) < 1e-10
    # a field that does not solve the equation (the incident field) has a large residual
    assert tgt.solver_residual(p["xi"], tgt.initial_aux()) > 1e-3
    # the aux of one solve is the field the forward model returns
    np.testing.assert_allclose(np.asarray(aux), np.asarray(p["fwd"](chi)[1]), atol=1e-9)


def test_born_target_passes_aux_through_unchanged(problem):
    p = problem
    tgt = _target(p, use_born=True)
    marker = jnp.full(tgt.initial_aux().shape, 0.25 + 0.5j)
    _, aux_new = tgt.potential(p["xi"], marker)
    _, _, aux_new2 = tgt.potential_and_grad(p["xi"], marker)
    np.testing.assert_array_equal(np.asarray(aux_new), np.asarray(marker))
    np.testing.assert_array_equal(np.asarray(aux_new2), np.asarray(marker))
    assert tgt.solver_residual(p["xi"], marker) == 0.0


def test_gradient_free_phi_fn_and_record(problem):
    p = problem
    tgt = _target(p)
    phi_a, none_grad, aux = tgt.phi_fn_no_grad(p["xi"], tgt.initial_aux())
    assert none_grad is None
    np.testing.assert_allclose(float(phi_a), float(tgt.potential(p["xi"], tgt.initial_aux())[0]), rtol=1e-12)
    rec = tgt.record(p["xi"])
    assert rec.shape == (2, N_GRID, N_GRID)
    np.testing.assert_allclose(np.asarray(rec[0]), np.asarray(p["field"].theta(p["xi"])), atol=1e-13)
    np.testing.assert_allclose(np.asarray(rec[1]), np.exp(np.asarray(rec[0])), atol=1e-12)  # eps_r = e^theta


def test_input_validation(problem):
    p = problem
    with pytest.raises(ValueError):
        Target(p["fwd"], p["field"], "log_eps", p["d_obs"][:, :5], p["s"])
    with pytest.raises(ValueError):
        Target(p["fwd"], p["field"], "log_eps", p["d_obs"], 0.0)
    with pytest.raises(ValueError):
        Target(p["fwd"], p["field"], "sigmoid_box", p["d_obs"], p["s"])
    other = GaussianField(ch15_setup(n=6, side=1.0, n_tx=6, n_rx=10).grid, "matern")
    with pytest.raises(ValueError):
        Target(p["fwd"], other, "log_eps", p["d_obs"], p["s"])


@pytest.mark.parametrize("use_born", [False, True], ids=["full", "born"])
def test_misfit_of_chi_is_the_potential_of_the_contrast_map(problem, use_born):
    """misfit_of_chi(chi(xi)) is Phi(xi): it lets a run that sampled in other coordinates (Laplace-whitened
    zeta, whose own potential is Phi' = Phi + |xi|^2/2 - |zeta|^2/2) recover the physical misfit of its draws."""
    p = problem
    tgt = _target(p, use_born=use_born)
    phi_xi, aux_xi = tgt.potential(p["xi"], tgt.initial_aux())
    phi_chi, aux_chi = tgt.misfit_of_chi(tgt.chi(p["xi"]), tgt.initial_aux())
    np.testing.assert_allclose(float(phi_chi), float(phi_xi), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(aux_chi), np.asarray(aux_xi), atol=1e-12)
    # a different contrast map gives a different misfit, and it is not the potential at the same xi
    phi_other, _ = tgt.misfit_of_chi(0.5 * tgt.chi(p["xi"]), tgt.initial_aux())
    assert abs(float(phi_other) - float(phi_xi)) > 0.1 * float(phi_xi)
    # at the true contrast the misfit is the realized noise energy over s^2 (the identity of the test above)
    chi_true = tgt.chi(p["xi_true"])
    if not use_born:
        phi_t, _ = tgt.misfit_of_chi(chi_true, tgt.initial_aux())
        np.testing.assert_allclose(float(phi_t), np.sum(np.abs(np.asarray(p["noise"])) ** 2) / p["s"] ** 2,
                                   rtol=1e-8)
