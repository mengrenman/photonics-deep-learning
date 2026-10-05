"""Tests for bayes_scatter.inference: error metrics, exact Born posterior, MAP, sampler driver, Gaussian
comparison statistics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import samplers as sm
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ch15_setup, ring
from bayes_scatter.inference import (
    born_gaussian_posterior,
    compare_to_gaussian,
    err_chi,
    err_data,
    err_model,
    map_estimate,
    n_phi_evals,
    operating_point,
    physical_phi_of_draws,
    sample,
)
from bayes_scatter.likelihood import Target, add_noise, noise_std_from_snr
from bayes_scatter.phantoms import coverage, cylinder
from bayes_scatter.priors import GaussianField, identity, log_eps
from bayes_scatter.validation import make_born_problem

# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def test_error_metrics_known_values():
    rng = np.random.default_rng(0)
    m = 1.0 + rng.random((5, 5))
    assert err_model(m, m) == 0.0
    np.testing.assert_allclose(err_model(2.0 * m, m), 1.0, rtol=1e-14)  # eq. 15.21: ||m_R - m_T|| / ||m_T||
    np.testing.assert_allclose(err_model(0.5 * m, m), 0.5, rtol=1e-14)
    np.testing.assert_allclose(err_model(np.ones((2, 2)), 3.0 * np.ones((2, 2))), 2.0 / 3.0, rtol=1e-14)
    d = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    e = 0.1 * (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)))
    np.testing.assert_allclose(err_data(d + e, d), np.linalg.norm(e) / np.linalg.norm(d), rtol=1e-13)
    chi = m - 1.0
    # the same reconstruction error looks four times smaller in eps_r than in the contrast for chi = 0.25
    chi_t = 0.25 * np.ones((4, 4))
    np.testing.assert_allclose(err_chi(np.zeros((4, 4)), chi_t), 1.0, rtol=1e-14)
    np.testing.assert_allclose(err_model(np.ones((4, 4)), 1.0 + chi_t), 0.25 / 1.25, rtol=1e-14)
    assert err_chi(chi, chi) == 0.0


# ---------------------------------------------------------------------------------------------
# Exact Born posterior
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def born():
    return make_born_problem()


def test_born_posterior_matches_complex_formulas(born):
    """Independent derivation in complex arithmetic: Lambda = I + (2/s^2) Re(B^H B) and
    m = (2/s^2) Lambda^-1 Re(B^H y) (from the stacked real form with sigma_r^2 = s^2/2)."""
    p = born
    A = np.asarray(p.forward.born_matrix())
    L = p.field.sqrt_matrix()
    mu = np.asarray(p.field.mean).ravel()
    B = A @ L
    y = np.asarray(p.d_obs).ravel() - A @ mu
    lam = np.eye(B.shape[1]) + 2.0 / p.s**2 * (B.conj().T @ B).real
    m = np.linalg.solve(lam, 2.0 / p.s**2 * (B.conj().T @ y).real)
    ex = p.exact
    np.testing.assert_allclose(ex.precision_xi, lam, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(ex.mean_xi, m, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ex.cov_xi @ lam, np.eye(lam.shape[0]), atol=1e-9)
    np.testing.assert_allclose(ex.cov_xi, ex.cov_xi.T, atol=1e-14)
    np.testing.assert_allclose(ex.mean_chi, mu + L @ m, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ex.cov_chi, L @ ex.cov_xi @ L.T, rtol=1e-9, atol=1e-13)
    # it is a posterior: tighter than the prior, and strictly positive definite
    prior_var = p.field.sigma**2
    assert np.all(np.diag(ex.cov_chi) < 0.5 * prior_var)
    assert np.linalg.eigvalsh(ex.cov_xi).min() > 0.0


def test_born_posterior_prior_and_data_limits(born):
    p = born
    # s -> infinity: no information, the posterior is the prior
    ex = born_gaussian_posterior(p.forward, p.field, p.d_obs, 1e8)
    np.testing.assert_allclose(ex.mean_chi, np.asarray(p.field.mean).ravel(), atol=1e-8)
    np.testing.assert_allclose(ex.cov_chi, p.field.covariance_matrix(), atol=1e-8)
    np.testing.assert_allclose(ex.cov_xi, np.eye(ex.cov_xi.shape[0]), atol=1e-8)
    # s -> 0 with noise-free data: the posterior mean reproduces the data and the pixels are well determined
    A = np.asarray(p.forward.born_matrix())
    d0 = np.asarray(p.d_clean)
    s_small = 1e-3 * np.sqrt(np.mean(np.abs(d0) ** 2))
    ex = born_gaussian_posterior(p.forward, p.field, d0, s_small)
    d_post = (A @ ex.mean_chi).reshape(d0.shape)
    assert np.linalg.norm(d_post - d0) < 1e-6 * np.linalg.norm(d0)
    assert np.sqrt(np.diag(ex.cov_chi)).max() < 0.2 * p.field.sigma


def test_born_posterior_uses_the_prior_mean(born):
    """With a nonzero prior mean the answer is the one for the shifted data y = d - A mu."""
    p = born
    field = GaussianField(p.setup.grid, "matern", sigma=0.3, ell=0.3, nu=1.5, mean=0.1)
    ex = born_gaussian_posterior(p.forward, field, p.d_obs, p.s)
    A = np.asarray(p.forward.born_matrix())
    shifted = np.asarray(p.d_obs) - (A @ (0.1 * np.ones(p.n * p.n))).reshape(p.d_obs.shape)
    ex0 = born_gaussian_posterior(p.forward, p.field, shifted, p.s)
    np.testing.assert_allclose(ex.mean_xi, ex0.mean_xi, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(ex.mean_chi, ex0.mean_chi + 0.1, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("link", ["log_eps", log_eps])
def test_born_posterior_rejects_the_log_eps_link(born, link):
    p = born
    with pytest.raises(ValueError, match="identity"):
        born_gaussian_posterior(p.forward, p.field, p.d_obs, p.s, link=link)
    # the identity link is accepted in both spellings
    for ok in ("identity", identity):
        born_gaussian_posterior(p.forward, p.field, p.d_obs, p.s, link=ok)


def test_born_posterior_rejects_wrong_data_size(born):
    p = born
    with pytest.raises(ValueError):
        born_gaussian_posterior(p.forward, p.field, np.asarray(p.d_obs)[:, :10], p.s)


def test_target_energy_is_the_exact_gaussian_quadratic(born):
    """U(xi) = Phi + |xi|^2/2 for the Born/identity target equals U(m) + (xi - m)^T Lambda (xi - m)/2 with
    the exact precision and mean: ties Target (physics, noise, link) to born_gaussian_posterior."""
    p = born
    ex = p.exact
    tgt = p.target
    aux = tgt.initial_aux()
    m = jnp.asarray(ex.mean_xi).reshape(p.field.xi_shape)

    def energy(xi):
        return float(tgt.potential(xi, aux)[0]) + 0.5 * float(jnp.sum(xi**2))

    u_m = energy(m)
    rng = np.random.default_rng(0)
    for scale in (0.05, 1.0):
        delta = scale * rng.standard_normal(ex.mean_xi.size)
        quad = 0.5 * delta @ ex.precision_xi @ delta
        got = energy(m + jnp.asarray(delta).reshape(m.shape)) - u_m
        np.testing.assert_allclose(got, quad, rtol=1e-8)
    # the gradient of U vanishes at the posterior mean
    _, grad, _ = tgt.potential_and_grad(m, aux)
    g = np.asarray(grad) + np.asarray(m)
    _, grad0, _ = tgt.potential_and_grad(jnp.zeros_like(m), aux)
    assert np.linalg.norm(g) < 1e-8 * np.linalg.norm(np.asarray(grad0))


# ---------------------------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------------------------


def test_map_equals_exact_posterior_mean_in_the_born_linear_case(born):
    """For a Gaussian posterior the mode is the mean, so L-BFGS-B must land on born_gaussian_posterior."""
    p = born
    xi_map, chi_map, info = map_estimate(p.target, jnp.zeros(p.field.xi_shape), maxiter=500, gtol=1e-9)
    m = p.exact.mean_xi
    assert info["success"], info["message"]
    # optimizer tolerance: U is about 2400, so function values resolve only ~5e-13, which limits xi to ~1e-6
    assert np.linalg.norm(np.asarray(xi_map).ravel() - m) < 1e-5 * np.linalg.norm(m)
    np.testing.assert_allclose(chi_map, p.mean_chi, atol=1e-5 * np.abs(p.mean_chi).max())
    assert info["grad_inf"] < 1e-5 and info["nfev"] >= info["nit"] > 0
    # U decreased from the start and the reported U is the energy at the returned point
    assert info["U_history"][0] > info["U"] and info["U"] == pytest.approx(info["U_history"][-1], rel=1e-6)
    assert info["phi"] > 0 and info["time_s"] > 0
    assert chi_map.shape == (p.n, p.n) and xi_map.shape == p.field.xi_shape


def test_map_reduces_err_model_for_a_weak_contrast_cylinder_with_log_eps_link():
    """Nonlinear full forward model, log_eps link, data on a 2x finer grid (no inverse crime), prior mean 0
    (vacuum). The vacuum start theta = 0 has err_model 0.2 (and err_chi = 1); the MAP must remove most of it."""
    n, side = 16, 2.0
    setup = Setup(Grid(n, side), Antennas(ring(16, 4.0), ring(24, 4.5)))
    fine = Setup(Grid(2 * n, side), setup.antennas)
    radius, eps = 0.5, 1.5
    chi_data = cylinder(fine.grid, radius, eps)
    d_clean, _ = Forward(fine, tol=1e-10)(jnp.asarray(chi_data))
    s = noise_std_from_snr(d_clean, 30.0)
    d_obs = add_noise(jax.random.key(0), d_clean, s)

    forward = Forward(setup, tol=1e-10)
    field = GaussianField(setup.grid, "matern", sigma=0.5, ell=0.4, nu=1.5, mean=0.0)
    target = Target(forward, field, "log_eps", d_obs, s)
    eps_true = 1.0 + cylinder(setup.grid, radius, eps)
    chi_true = eps_true - 1.0

    xi0 = jnp.zeros(field.xi_shape)  # theta = 0: vacuum
    err0 = err_model(np.ones((n, n)), eps_true)
    assert 0.15 < err0 < 0.25
    xi_map, chi_map, info = map_estimate(target, xi0, maxiter=200, gtol=1e-5)
    eps_map = 1.0 + chi_map
    err_map = err_model(eps_map, eps_true)
    assert err_map < 0.5 * err0, (err0, err_map, info["message"])
    assert err_chi(chi_map, chi_true) < 0.6  # vacuum gives 1
    assert info["U"] < float(target.potential(xi0, target.initial_aux())[0])
    # the data are fitted far better than by the vacuum, whose misfit is the whole signal energy
    d_map, _ = forward(jnp.asarray(chi_map))
    assert err_data(d_map, d_clean) < 0.25  # vacuum: d = 0 gives 1
    # eps_r from the MAP is positive by construction of the log_eps link
    assert eps_map.min() > 0.0


def test_map_info_and_validation(born):
    p = born
    xi, chi, info = map_estimate(p.target, jnp.zeros(p.field.xi_shape), maxiter=3)
    assert info["nit"] <= 3 and not info["success"]  # capped before convergence
    assert set(info) >= {"success", "message", "nit", "nfev", "U", "phi", "grad_inf", "time_s", "U_history"}


# ---------------------------------------------------------------------------------------------
# Sampler driver
# ---------------------------------------------------------------------------------------------


def test_n_phi_evals_matches_the_samplers_actual_call_count():
    """Count real phi_fn calls (host callback) in a tiny pCN run and an HMC run."""
    counter = {"n": 0}

    def bump():
        counter["n"] += 1

    def phi_fn(xi, aux):
        jax.debug.callback(bump)
        return 0.5 * jnp.sum(xi**2), xi, aux

    for name, fn, kw in [("pcn", sm.pcn, {}), ("mala", sm.mala, {}), ("hmc", sm.hmc, {"n_leapfrog": 4})]:
        counter["n"] = 0
        res = fn(jax.random.key(0), phi_fn, jnp.ones(3), jnp.zeros(1), n_warmup=7, n_samples=11, step0=0.1, **kw)
        jax.block_until_ready(res.phi)
        jax.effects_barrier()
        assert counter["n"] == n_phi_evals(name, 7, 11, kw.get("n_leapfrog", 10)), name


@pytest.fixture(scope="module")
def tiny_born():
    return make_born_problem(n=6, side=1.0, snr_db=-10.0, sigma=0.3, ell=0.3, n_tx=6, n_rx=8)


@pytest.mark.parametrize("name", ["pcn", "pcn_langevin", "mala", "hmc"])
def test_sample_vmap_equals_sequential_and_reports_counts(tiny_born, name):
    p = tiny_born
    xi0s = p.field.sample_xi(jax.random.key(0), 3)
    kw = dict(n_warmup=30, n_samples=40, thin=4, step0=0.05, record_fn=p.target.chi, n_leapfrog=3)
    r1, i1 = sample(p.target, name, jax.random.key(1), xi0s, chain_method="vmap", **kw)
    r2, i2 = sample(p.target, name, jax.random.key(1), xi0s, chain_method="sequential", **kw)
    assert r1.samples.shape == (3, 10, p.n, p.n) and r1.phi.shape == (3, 40) and r1.step.shape == (3,)
    for a, b in zip(jax.tree.leaves(r1), jax.tree.leaves(r2)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-7, atol=1e-9)
    evals = n_phi_evals(name, 30, 40, 3)
    assert i1["n_phi_evals"] == evals and i1["n_forward_solves"] == 3 * evals
    assert i1["n_adjoint_solves"] == (0 if name == "pcn" else 3 * evals)
    assert i1["wall_s"] > 0 and i1["compile_s"] > 0
    # the final aux of a Born target is the unchanged initial aux
    np.testing.assert_array_equal(np.asarray(r1.aux[0]), np.asarray(p.target.initial_aux()))


def test_sample_default_record_and_validation(tiny_born):
    p = tiny_born
    xi0s = p.field.sample_xi(jax.random.key(0), 2)
    res, _ = sample(p.target, "mala", jax.random.key(1), xi0s, n_warmup=5, n_samples=6, step0=0.01)
    assert res.samples.shape == (2, 6, 2, p.n, p.n)  # default record: (theta, eps_r) stacked
    with pytest.raises(ValueError):
        sample(p.target, "nuts", jax.random.key(1), xi0s, n_warmup=5, n_samples=6, step0=0.01)
    with pytest.raises(ValueError):
        sample(p.target, "mala", jax.random.key(1), xi0s, n_warmup=5, n_samples=6, step0=0.01,
               chain_method="threads")


# ---------------------------------------------------------------------------------------------
# compare_to_gaussian
# ---------------------------------------------------------------------------------------------


def _ar1(rng, shape, phi):
    """Stationary AR(1) series with unit marginal variance along axis 1."""
    x = np.zeros(shape)
    x[:, 0] = rng.standard_normal(shape[0:1] + shape[2:])
    e = rng.standard_normal(shape) * np.sqrt(1 - phi**2)
    for t in range(1, shape[1]):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def test_compare_to_gaussian_is_calibrated_for_correlated_chains():
    """On AR(1) chains with phi = 0.9 (tau = 19) the standardized errors are N(0, 1): the MCSE must use the
    effective sample size (ignoring the autocorrelation would inflate the rms z to about 4)."""
    rng = np.random.default_rng(0)
    x = 2.0 + 3.0 * _ar1(rng, (4, 20000, 40), 0.9)
    st = compare_to_gaussian(x, np.full(40, 2.0), np.full(40, 9.0))
    assert 0.75 < st["z_mean_rms"] < 1.35 and st["z_mean_max"] < 4.0
    assert 0.75 < st["z_var_rms"] < 1.4 and st["z_var_max"] < 4.5
    assert 0.9 < st["var_ratio_mean"] < 1.1 and st["rhat_max"] < 1.05
    naive_sd = np.std(x.mean(axis=1).mean(axis=0)) / (3.0 / np.sqrt(x.shape[0] * x.shape[1]))
    assert naive_sd > 2.0  # the naive iid error would be too small by sqrt(tau)
    assert 800 < st["ess_min"] and st["ess"].max() < 80000 / 5  # ess ~ 80000 / 19 = 4200


def test_compare_to_gaussian_detects_wrong_mean_and_wrong_variance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5000, 12))
    ok = compare_to_gaussian(x, np.zeros(12), np.ones(12))
    assert ok["z_mean_rms"] < 1.6 and ok["z_var_rms"] < 1.6
    shifted = compare_to_gaussian(x, np.full(12, 0.1), np.ones(12))  # 0.1 sd off = 14 MCSE
    assert shifted["z_mean_rms"] > 8.0
    wide = compare_to_gaussian(x, np.zeros(12), np.full(12, 1.2))  # variance off by 20%
    assert wide["z_var_rms"] > 8.0 and abs(wide["var_ratio_mean"] - 1.0 / 1.2) < 0.03
    # trailing shape is preserved in the flattening order
    st = compare_to_gaussian(x.reshape(4, 5000, 3, 4), np.zeros((3, 4)), np.ones((3, 4)))
    np.testing.assert_allclose(st["z_mean"], ok["z_mean"])


# ---------------------------------------------------------------------------------------------
# Operating point (grid model error against the noise) and the physical misfit of stored draws
# ---------------------------------------------------------------------------------------------


def _phantom_problem(n, n_data, snr_db=30.0, seed=0):
    """Weak cylinder on an n grid, data on an n_data grid (n_data = n is the inverse-crime control)."""
    setup = ch15_setup(n=n, side=1.0, n_tx=8, n_rx=12)
    data_setup = Setup(Grid(n_data, 1.0), setup.antennas)
    d_clean, _ = Forward(data_setup, tol=1e-12)(jnp.asarray(cylinder(data_setup.grid, 0.3, 1.6)))
    s = noise_std_from_snr(d_clean, snr_db)
    d_obs = add_noise(jax.random.key(seed), d_clean, s)
    return Forward(setup, tol=1e-12), jnp.asarray(cylinder(setup.grid, 0.3, 1.6)), d_clean, d_obs, s


def test_operating_point_same_grid_control_has_no_model_error():
    fwd, chi, d_clean, d_obs, s = _phantom_problem(8, 8)
    op = operating_point(fwd, chi, d_clean, d_obs, s)
    assert op["rel_model_error"] < 1e-9 and abs(op["phi_excess_sd"]) < 1e-6
    assert op["n_data"] == 96 and op["phi_truth"] == pytest.approx(op["phi_clean"], rel=1e-8)
    # N complex entries of noise: phi_clean = sum |e|^2 / s^2 is about N, and rel_noise = s sqrt(N) / ||d||
    assert 0.5 * 96 < op["phi_clean"] < 1.6 * 96
    np.testing.assert_allclose(op["rel_noise"], s * np.sqrt(96) / np.linalg.norm(np.asarray(d_clean)), rtol=1e-13)


def test_operating_point_excess_of_phi_is_the_squared_model_to_noise_ratio_times_n():
    """Phi(truth) - Phi(clean) = ||delta||^2/s^2 - 2 Re<delta, e>/s^2 for the grid error delta, so its mean is
    (model_to_noise)^2 N. Checked over noise seeds on a deliberately coarse grid; a missing square or a
    factor 2 in the noise variance would change the ratio by 2x or more."""
    ratios, excess = [], []
    for seed in range(12):
        fwd, chi, d_clean, d_obs, s = _phantom_problem(6, 12, snr_db=35.0, seed=seed)
        op = operating_point(fwd, chi, d_clean, d_obs, s)
        ratios.append(op["model_to_noise"])
        excess.append(op["phi_truth"] - op["phi_clean"])
    n_data = 96
    expected = np.mean(ratios) ** 2 * n_data
    # the 12 seeds share delta (same d_clean up to the noise level), so the spread comes from the cross term
    sd_mean = np.sqrt(2.0 * expected) / np.sqrt(12)
    assert expected > 20.0  # the grid is coarse enough that the excess is large
    assert abs(np.mean(excess) - expected) < 4.0 * sd_mean


def test_physical_phi_of_draws_matches_the_potential_of_each_draw():
    setup = ch15_setup(n=8, side=1.0, n_tx=6, n_rx=8)
    fwd = Forward(setup, tol=1e-12)
    field = GaussianField(setup.grid, "matern", sigma=0.4, ell=0.3, nu=1.5, mean=0.1)
    d_true, _ = fwd(jnp.asarray(cylinder(setup.grid, 0.3, 1.4)))
    s = noise_std_from_snr(d_true, 25.0)
    tgt = Target(fwd, field, "log_eps", add_noise(jax.random.key(1), d_true, s), s)
    xis = field.sample_xi(jax.random.key(2), 6).reshape(2, 3, *field.xi_shape)
    eps = np.stack([[np.asarray(tgt.eps_r(x)) for x in row] for row in xis])  # (chains, draws, n, n)
    got = physical_phi_of_draws(tgt, eps)
    want = np.array([[float(tgt.potential(x, tgt.initial_aux())[0]) for x in row] for row in xis])
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got, want, rtol=1e-9)
    assert want.std() > 0.1 * want.mean() or want.std() > 10.0  # draws differ, so the check is not constant
