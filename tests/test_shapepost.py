"""Tests for bayes_scatter.shapepost: MAP, Laplace evidence, HMC, multimodality, thermodynamic integration and predictive checks.

The algorithms are tested on problems with known answers (a linear-Gaussian toy, a banana, a bimodal target); the scattering problem is tested for its gradient and
Fisher matrix against finite differences, and (slow tests) end to end on a small grid.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from bayes_scatter import diagnostics as dg
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr


# ----------------------------------------------------------------------------------------------
# Toy problems
# ----------------------------------------------------------------------------------------------


def make_toy(seed=0, m=12, d=3, noise=0.5, prior=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, d))
    u_true = rng.standard_normal(d) * prior
    y = A @ u_true + noise * rng.standard_normal(m)
    return sp.GaussianToy(A, y, noise_sd=noise, prior_sd=prior)


class FunctionProblem:
    """A problem from a scalar potential phi(u) (JAX) with a Gaussian prior N(0, prior_sd^2 I): the interface of ``shapepost`` for tests (Gauss-Newton information
    = the positive part of the Hessian of phi)."""

    def __init__(self, phi, dim, prior_sd=3.0, lik_const=0.0):
        self.dim, self.lik_const, self.names = dim, lik_const, tuple(f"u{k}" for k in range(dim))
        self.data = jnp.zeros(())
        vg = jax.value_and_grad(lambda u: phi(u))
        self.phi = jax.jit(lambda u, aux, data: (phi(u), aux))
        self.phi_and_grad = jax.jit(lambda u, aux, data: (*vg(u), aux))
        self.log_prior = jax.jit(lambda u: -0.5 * jnp.sum(u * u) / prior_sd**2 - 0.5 * dim * jnp.log(2 * jnp.pi * prior_sd**2))
        self._hess = jax.jit(jax.hessian(phi))
        self._U_and_grad = self._prior_hessian = None

    def initial_aux(self):
        return jnp.zeros(())

    def fisher(self, u):
        w, V = np.linalg.eigh(np.asarray(self._hess(jnp.asarray(u))))
        return (V * np.maximum(w, 0.0)) @ V.T


CORE_HALO = (0.6, 0.15, 0.4, 1.2)  # weights and standard deviations of the two isotropic components
CORE_HALO_PRIOR_SD = 3.0


def core_halo_phi(u):
    """phi = -log(0.6 N(u; 0, 0.15^2 I) + 0.4 N(u; 0, 1.2^2 I)) in two dimensions (lik_const = 0): a narrow core inside a wide halo. Unimodal, finite moments, and
    non-Gaussian: the Laplace approximation sees only the core curvature."""
    w1, s1, w2, s2 = CORE_HALO
    r2 = jnp.sum(u * u)
    a = jnp.log(w1) - jnp.log(2 * jnp.pi * s1**2) - 0.5 * r2 / s1**2
    b = jnp.log(w2) - jnp.log(2 * jnp.pi * s2**2) - 0.5 * r2 / s2**2
    return -jnp.logaddexp(a, b)


def core_halo_evidence():
    """log integral exp(-phi) N(u; 0, 3^2 I) du = log(sum_i w_i N(0; 0, (s_i^2 + 3^2) I)) (a convolution of Gaussians)."""
    w1, s1, w2, s2 = CORE_HALO
    v = CORE_HALO_PRIOR_SD**2
    return float(np.log(w1 / (2 * np.pi * (s1**2 + v)) + w2 / (2 * np.pi * (s2**2 + v))))


def bimodal_phi(shift=0.0):
    m1, m2 = jnp.array([4.0, 0.0]), jnp.array([-4.0, 0.0])

    def phi(u):
        a = -0.5 * jnp.sum((u - m1) ** 2) / 0.25
        b = -0.5 * jnp.sum((u - m2) ** 2) / 0.25 - shift
        return -jnp.logaddexp(a, b)

    return phi


# ----------------------------------------------------------------------------------------------
# MAP, Laplace, HMC on the linear-Gaussian toy
# ----------------------------------------------------------------------------------------------


def test_map_and_laplace_evidence_are_exact_for_a_linear_gaussian_problem():
    toy = make_toy()
    search = sp.map_estimate(toy, np.vstack([np.zeros(3), 3.0 * np.ones(3)]))
    mean, cov = toy.exact_posterior()
    assert all(c["converged"] for c in search.candidates)
    np.testing.assert_allclose(search.u_hat, mean, atol=1e-7)
    fit = sp.laplace_fit(toy, search.u_hat)
    np.testing.assert_allclose(fit.cov, cov, atol=1e-12)
    assert sp.laplace_log_evidence(toy, fit) == pytest.approx(toy.exact_log_evidence(), abs=1e-9)
    fit_fd = sp.laplace_fit(toy, search.u_hat, hessian="fd")
    assert sp.laplace_log_evidence(toy, fit_fd) == pytest.approx(toy.exact_log_evidence(), abs=1e-6)
    # the Laplace map whitens the posterior exactly
    amap = fit.affine_map()
    np.testing.assert_allclose(amap.covariance, cov, atol=1e-12)
    with pytest.raises(ValueError):
        sp.laplace_log_evidence(toy, sp.laplace_fit(toy, search.u_hat, beta=0.5))


def test_hmc_in_the_laplace_coordinates_recovers_the_posterior_and_a_wrong_prior_does_not_pass_the_same_test():
    toy = make_toy(seed=1)
    mean, cov = toy.exact_posterior()
    fit = sp.laplace_fit(toy, mean)
    ps = sp.sample_posterior(toy, fit, jax.random.key(0), n_chains=4, n_warmup=100, n_samples=1500)
    u = ps.batch.u
    pooled = u.reshape(-1, 3)
    ess = dg.ess_bulk(u)
    se_mean = np.sqrt(np.diag(cov) / ess)
    assert np.all(np.abs(pooled.mean(axis=0) - mean) < 4.0 * se_mean)
    assert np.all(np.abs(np.cov(pooled.T) / cov - 1.0).diagonal() < 0.15)
    assert 0.5 < ps.batch.acceptance <= 1.0 and ps.convergence["worst_ess_bulk"] > 200 and ps.convergence["max_rhat"] < 1.05
    np.testing.assert_allclose(ps.batch.phi[0, :5], [float(toy.phi(jnp.asarray(x), jnp.zeros(()), toy.data)[0]) for x in u[0, :5]], rtol=1e-12)  # phi is the physical misfit at the draws
    # negative control: the same comparison against the exact posterior of a different prior (sd 0.3 instead of 1) must fail, so the test above is not vacuous
    wrong_mean, wrong_cov = sp.GaussianToy(toy.A, np.asarray(toy.data), noise_sd=toy.noise_sd, prior_sd=0.3).exact_posterior()
    assert np.any(np.abs(pooled.mean(axis=0) - wrong_mean) > 4.0 * se_mean) or np.any(np.abs(np.cov(pooled.T) / wrong_cov - 1.0).diagonal() > 0.15)


def test_continuing_a_chain_from_its_final_state_is_a_continuation_with_the_frozen_step():
    toy = make_toy(seed=2)
    mean, cov = toy.exact_posterior()
    amap = sp.laplace_fit(toy, mean).affine_map()
    zeta0 = np.array([[1.0, -1.0, 0.5]])
    a = sp.run_hmc(toy, amap, zeta0, jax.random.key(3), n_warmup=60, n_samples=100)
    b = sp.run_hmc(toy, amap, a.final_zeta, jax.random.key(4), n_warmup=0, n_samples=100, step0=a.step, aux0=a.final_aux)
    assert b.step[0] == pytest.approx(a.step[0])  # the step is the frozen step of the first segment
    assert np.linalg.norm(b.zeta[0, 0] - a.final_zeta[0]) < 10.0  # the continuation starts where the segment ended (one trajectory away at most)
    assert b.zeta.shape == (1, 100, 3) and np.isfinite(b.zeta).all()
    # the recorded u are center + chol zeta
    np.testing.assert_allclose(a.u[0], amap.center_np + a.zeta[0] @ amap.chol_np.T, atol=1e-12)


def test_prior_truncation_rejects_proposals_outside_the_box_without_solving():
    toy = make_toy()
    val, g, _ = sp.potential(toy, np.array([sp.U_BOUND + 1.0, 0.0, 0.0]))
    assert val == np.inf and np.all(g == 0.0)
    val_in, _, _ = sp.potential(toy, np.zeros(3))
    assert np.isfinite(val_in)


# ----------------------------------------------------------------------------------------------
# Thermodynamic integration
# ----------------------------------------------------------------------------------------------


def exact_ti_integrand(toy, betas):
    """E_beta[phi] of the linear-Gaussian toy in closed form (the power posterior is Gaussian)."""
    A, y, sd, ps = toy.A, np.asarray(toy.data), toy.noise_sd, toy.prior_sd
    out = []
    for b in betas:
        prec = b * A.T @ A / sd**2 + np.eye(toy.dim) / ps**2
        cov = np.linalg.inv(prec)
        mu = cov @ (b * A.T @ y) / sd**2
        out.append(0.5 * (np.sum((y - A @ mu) ** 2) + np.trace(A @ cov @ A.T)) / sd**2)
    return np.array(out)


def test_ti_quadrature_uses_the_euler_maclaurin_correction_correctly():
    """With the exact E_beta[phi] and Var_beta[phi] of the toy (no Monte Carlo error) the corrected rule is closer to the exact evidence than the trapezoid, and the
    difference of the two is the quantity ``discretization`` reports."""
    toy = make_toy(seed=3, m=30, d=4, noise=0.3)
    betas = sp.ti_schedule(12)
    eps = 1e-5
    mean = exact_ti_integrand(toy, betas)
    # d/dbeta E_beta[phi] by a small central difference of the closed form (one-sided at beta = 0)
    var = np.array([(exact_ti_integrand(toy, [b + eps])[0] - exact_ti_integrand(toy, [b - eps])[0]) / (2 * eps) if b > 0 else
                    (exact_ti_integrand(toy, [eps])[0] - mean[0]) / eps for b in betas])
    # d E_beta[phi] / d beta = -Var_beta[phi] (phi = -log L + const), so the variance is minus the derivative
    stats_ = [{"beta": b, "mean_phi": m, "var_phi": -v, "ess_phi": 1e12, "rhat_phi": 1.0, "rhat_phi_rank": 1.0, "acceptance": 0.8} for b, m, v in zip(betas, mean, var)]
    res = sp.ti_combine(toy, stats_)
    exact = toy.exact_log_evidence()
    assert abs(res["log_z"] - exact) < abs(res["log_z_trapezoid"] - exact)
    assert abs(res["log_z"] - exact) < 0.25 * abs(res["log_z_trapezoid"] - exact)  # the end correction removes most of the O(Delta^2) error (measured: 7 times less)
    assert res["discretization"]["corrected_minus_trapezoid"] == pytest.approx(res["log_z"] - res["log_z_trapezoid"])
    assert res["monotone"]["ok"] and res["mc_se"] < 1e-5


def test_ti_matches_the_exact_evidence_of_the_linear_gaussian_problem_within_its_monte_carlo_error():
    toy = make_toy(seed=4)
    exact = toy.exact_log_evidence()
    rng = np.random.default_rng(0)
    res, temps = sp.thermodynamic_integration(toy, jax.random.key(5), n_temps=12, n_chains=4, n_pilot=30, n_warmup=50, n_samples=300, prior_starts=rng.standard_normal((4, 3)))
    assert res["n_temps"] == 12 and res["betas"][0] == 0.0 and res["betas"][-1] == 1.0 and res["betas"][1] == pytest.approx((1 / 11) ** 5)
    assert res["converged"] and res["monotone"]["ok"]
    assert 0 < res["mc_se"] < 0.3
    assert abs(res["log_z"] - exact) < 3.0 * res["mc_se"] + abs(res["discretization"]["richardson_error_estimate"])
    assert abs(res["log_z"] - exact) < 0.5
    assert len(temps) == 12 and all(t.phi.shape == (4, 300) for t in temps)


def test_ti_checks_the_laplace_evidence_where_the_posterior_is_not_gaussian():
    """A core inside a halo: the Laplace evidence is off by more than the TI error, and TI agrees with the exact evidence."""
    problem = FunctionProblem(core_halo_phi, 2, prior_sd=CORE_HALO_PRIOR_SD)
    exact = core_halo_evidence()
    search = sp.map_estimate(problem, np.array([[0.05, -0.03]]))
    fit = sp.laplace_fit(problem, search.u_hat)
    laplace = sp.laplace_log_evidence(problem, fit)
    assert abs(laplace - exact) > 0.3  # the Laplace evidence is not exact here
    rng = np.random.default_rng(1)
    res, _ = sp.thermodynamic_integration(problem, jax.random.key(6), n_temps=14, n_chains=4, n_pilot=40, n_warmup=60, n_samples=400, prior_starts=3.0 * rng.standard_normal((4, 2)))
    assert abs(res["log_z"] - exact) < 3.0 * res["mc_se"] + abs(res["discretization"]["richardson_error_estimate"]) + 0.05
    assert abs(res["log_z"] - exact) < abs(laplace - exact)
    # the scale mixing of the core / halo transition is slow at beta ~ 0.7 to 1: the diagnostics say so, and the estimate is still within
    # its error. Over 40 seeds (20 on the Mac, 20 on the workstation, 2026-11-08) the worst R-hat had mean 1.11, sd 0.06 and maximum 1.29
    # (seed 0 on the workstation), and the two error assertions above never failed; 1.35 is about 4 sd above the mean
    assert res["worst_rhat"] < 1.35


def test_ti_diagnostics_flag_a_crude_schedule():
    """With 4 temperatures on a linear schedule the trapezoid error is large and the coarse-fine difference says so, while the 12-temperature design schedule does not."""
    toy = make_toy(seed=5, m=40, d=4, noise=0.2)
    exact = toy.exact_log_evidence()
    out = {}
    for label, n, power in (("crude", 5, 1.0), ("design", 12, 5.0)):
        betas = sp.ti_schedule(n, power)
        mean = exact_ti_integrand(toy, betas)
        var = np.array([(exact_ti_integrand(toy, [b + 1e-5])[0] - exact_ti_integrand(toy, [max(b - 1e-5, 0.0)])[0]) / (b + 1e-5 - max(b - 1e-5, 0.0)) for b in betas])
        st = [{"beta": b, "mean_phi": m, "var_phi": -v, "ess_phi": 1e12, "rhat_phi": 1.0, "rhat_phi_rank": 1.0, "acceptance": 0.8} for b, m, v in zip(betas, mean, var)]
        out[label] = sp.ti_combine(toy, st)
    err_crude = abs(out["crude"]["log_z_trapezoid"] - exact)
    err_design = abs(out["design"]["log_z_trapezoid"] - exact)
    assert err_crude > 5.0 * err_design
    assert abs(out["crude"]["discretization"]["richardson_error_estimate"]) > 5.0 * abs(out["design"]["discretization"]["richardson_error_estimate"])
    with pytest.raises(ValueError):
        sp.ti_schedule(2)
    with pytest.raises(ValueError):
        sp.ti_combine(toy, [{"beta": 0.2, "mean_phi": 0, "var_phi": 1, "ess_phi": 10, "rhat_phi": 1, "acceptance": 0.8}] * 3)


def test_ti_convergence_flag_needs_every_temperature_to_pass_and_lists_monotonicity_violations():
    toy = make_toy(seed=5, m=40, d=4, noise=0.2)
    betas = sp.ti_schedule(12)
    mean = exact_ti_integrand(toy, betas)
    good = [{"beta": b, "mean_phi": m, "var_phi": 1.0, "ess_phi": 500.0, "rhat_phi": 1.0, "rhat_phi_rank": 1.0, "acceptance": 0.8} for b, m in zip(betas, mean)]
    assert sp.ti_combine(toy, good)["converged"] is True
    for key, bad_value in (("rhat_phi", 1.2), ("rhat_phi_rank", 1.2), ("ess_phi", 5.0), ("acceptance", 0.1), ("acceptance", 0.999)):
        broken = [dict(g) for g in good]
        broken[6][key] = bad_value
        assert sp.ti_combine(toy, broken)["converged"] is False, key
    jumpy = [dict(g) for g in good]
    jumpy[4]["mean_phi"] = jumpy[3]["mean_phi"] + 50.0  # E_beta[phi] must not increase with beta
    rep = sp.ti_combine(toy, jumpy)
    assert rep["monotone"]["ok"] is False and 3 in rep["monotone"]["violations_at"]


# ----------------------------------------------------------------------------------------------
# Multimodality
# ----------------------------------------------------------------------------------------------


def test_two_equal_modes_are_detected_from_the_map_search_and_from_the_chains():
    problem = FunctionProblem(bimodal_phi(), 2)
    starts = np.array([[3.0, 1.0], [-3.0, -1.0], [0.1, 3.0]])
    search = sp.map_estimate(problem, starts)
    fit = sp.laplace_fit(problem, search.u_hat)
    # one chain per mode (started in the mode, in the Laplace coordinates of the first one): the means are far apart in within-chain standard deviations
    amap = fit.affine_map()
    zeta = np.stack([np.linalg.solve(amap.chol_np, s - amap.center_np) for s in (np.array([4.0, 0.0]), np.array([4.0, 0.0]), np.array([-4.0, 0.0]), np.array([-4.0, 0.0]))])
    batch = sp.run_hmc(problem, amap, zeta, jax.random.key(7), n_warmup=100, n_samples=300, n_leapfrog=6)
    report = sp.detect_multimodality(problem, search, fit, batch)
    assert report["multimodal"] is True
    assert any("distinct mode" in r for r in report["reasons"]) and any("chain means" in r for r in report["reasons"])
    assert len(report["modes"]) == 2 and report["modes"][1]["relative_weight"] == pytest.approx(1.0, rel=0.05)
    assert report["chain_separation"] > 5.0
    assert "u0" in report["bimodality"]["flagged"]


def test_a_distant_mode_of_negligible_weight_and_a_unimodal_target_are_not_flagged():
    heavy = FunctionProblem(bimodal_phi(shift=25.0), 2)  # the second mode is e^-25 lighter
    search = sp.map_estimate(heavy, np.array([[3.0, 1.0], [-3.0, -1.0]]))
    fit = sp.laplace_fit(heavy, search.u_hat)
    rep = sp.detect_multimodality(heavy, search, fit)
    assert rep["multimodal"] is False and len(rep["modes"]) == 2 and rep["modes"][1]["relative_weight"] < 1e-9
    toy = make_toy()
    search = sp.map_estimate(toy, np.vstack([np.zeros(3), 2.0 * np.ones(3)]))
    fit = sp.laplace_fit(toy, search.u_hat)
    ps = sp.sample_posterior(toy, fit, jax.random.key(8), n_chains=4, n_warmup=100, n_samples=400)
    rep = sp.detect_multimodality(toy, search, fit, ps.batch)
    assert rep["multimodal"] is False and len(rep["modes"]) == 1 and rep["chain_separation"] < 1.0 and rep["bimodality"]["flagged"] == []


def test_chain_separation_and_bimodality_coefficient_basics():
    rng = np.random.default_rng(0)
    same = rng.standard_normal((4, 500, 2))
    assert sp.chain_separation(same) < 0.5
    shifted = same.copy()
    shifted[2:, :, 0] += 6.0
    assert sp.chain_separation(shifted) > 5.0
    assert sp.bimodality_coefficient(rng.standard_normal(5000)) == pytest.approx(1.0 / 3.0, abs=0.04)
    both = np.concatenate([rng.standard_normal(2500) - 3.0, rng.standard_normal(2500) + 3.0])
    assert sp.bimodality_coefficient(both) > 0.6
    assert np.isnan(sp.bimodality_coefficient(np.ones(100)))


# ----------------------------------------------------------------------------------------------
# Posterior-predictive check
# ----------------------------------------------------------------------------------------------


def exact_draws(toy, y, rng, n):
    mean, cov = toy.exact_posterior(y)
    return rng.multivariate_normal(mean, cov, n), mean


def test_refit_predictive_p_values_are_uniform_for_a_correct_model_and_small_for_a_misspecified_one():
    """Correct model: data sets drawn from the prior predictive of the toy give p-values that pass a Kolmogorov-Smirnov test against the uniform law. Negative control: data
    with a component the model cannot represent (an extra regressor) give p-values near 0."""
    toy = make_toy(seed=6, m=20, d=3, noise=0.5)
    rng = np.random.default_rng(1)
    pv = []
    for i in range(100):
        u_true = rng.standard_normal(3)
        y = toy.A @ u_true + 0.5 * rng.standard_normal(20)
        draws, mean = exact_draws(toy, y, rng, 60)
        post = toy.with_data(y)
        pv.append(sp.predictive_check(post, draws, jax.random.key(i), n_rep=40, mode="refit", u_hat=mean)["p_value"])
    assert stats.kstest(pv, "uniform").pvalue > 0.01
    assert 0.35 < np.mean(pv) < 0.65
    bad = []
    for i in range(10):
        u_true = rng.standard_normal(3)
        extra = 3.0 * rng.standard_normal(20)  # the model has no regressor for this
        y = toy.A @ u_true + extra + 0.5 * rng.standard_normal(20)
        draws, mean = exact_draws(toy, y, rng, 60)
        bad.append(sp.predictive_check(toy.with_data(y), draws, jax.random.key(1000 + i), n_rep=40, mode="refit", u_hat=mean)["p_value"])
    assert max(bad) < 0.05 and np.mean(bad) < 0.01


def test_draw_predictive_p_value_is_central_for_a_correct_model_and_small_for_a_misspecified_one():
    toy = make_toy(seed=7, m=20, d=3, noise=0.5)
    rng = np.random.default_rng(2)
    pv = []
    for i in range(60):
        y = toy.A @ rng.standard_normal(3) + 0.5 * rng.standard_normal(20)
        draws, _ = exact_draws(toy, y, rng, 60)
        pv.append(sp.predictive_check(toy.with_data(y), draws, jax.random.key(i), n_rep=40, mode="draw")["p_value"])
    pv = np.array(pv)
    assert 0.4 < pv.mean() < 0.6  # the classical p-value of a correct model sits around the middle (the data are used twice, which makes it conservative)
    y = toy.A @ rng.standard_normal(3) + 4.0 * rng.standard_normal(20)
    draws, _ = exact_draws(toy, y, rng, 60)
    assert sp.predictive_check(toy.with_data(y), draws, jax.random.key(3), n_rep=40, mode="draw")["p_value"] < 0.01
    with pytest.raises(ValueError):
        sp.predictive_check(toy, draws, jax.random.key(0), mode="refit")
    with pytest.raises(ValueError):
        sp.predictive_check(toy, draws, jax.random.key(0), mode="nope")


def test_gap_summary_reports_interval_coverage_and_the_detection_probability():
    rng = np.random.default_rng(0)
    g = 0.05 + 3e-4 * rng.standard_normal(4000)
    out = sp.gap_summary(g, g_true=0.05, g_det=7.5e-4)
    assert out["covers_true"] is True and out["p_below_g_det"] == 0.0 and out["ci"][0] < 0.05 < out["ci"][1] and abs(out["z_true"]) < 0.2
    out0 = sp.gap_summary(np.abs(rng.standard_normal(4000)) * 3e-4, g_true=0.05, g_det=7.5e-4)
    assert out0["covers_true"] is False and out0["p_below_g_det"] > 0.9


# ----------------------------------------------------------------------------------------------
# The scattering problem
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small():
    """n = 16 grid, 6 transmitters, 8 receivers, tolerance 1e-11, data of the F2 truth on the same grid at 50 dB (an inverse crime on purpose: the model is exactly right)."""
    setup = ch15_setup(n=16, side=1.5, n_tx=6, n_rx=8)
    fwd = Forward(setup, tol=1e-11)
    rend = el.EllipseRenderer(setup.grid)
    truth = sh.TwoDisks(g=0.2, r=0.2, eps_r=2.0)
    d_clean, _ = fwd(jnp.asarray(truth.chi(setup.grid)))
    s = noise_std_from_snr(d_clean, 50.0)
    d_obs = add_noise(jax.random.key(0), d_clean, s)
    return {"setup": setup, "fwd": fwd, "rend": rend, "truth": truth, "s": s, "d_obs": d_obs, "d_clean": d_clean}


@pytest.mark.parametrize("name", ("F2", "F2u", "Fe"))
def test_phi_gradient_in_the_unconstrained_space_matches_finite_differences(small, name):
    model = sm.make_model(name)
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    rng = np.random.default_rng(10)
    aux0 = post.initial_aux()
    for _ in range(2):
        u = model.start_points_u(rng, 1)[0]
        _, g, _ = post.phi_and_grad(jnp.asarray(u), aux0, post.data)
        for _ in range(2):
            v = rng.standard_normal(model.dim)
            v /= np.linalg.norm(v)
            h = 1e-6
            fp = float(post.phi(jnp.asarray(u + h * v), aux0, post.data)[0])
            fm = float(post.phi(jnp.asarray(u - h * v), aux0, post.data)[0])
            fd = (fp - fm) / (2 * h)
            assert abs(float(np.asarray(g) @ v) - fd) < 1e-5 * max(abs(fd), 1.0)


def test_log_posterior_is_log_likelihood_plus_log_prior_with_its_gradient(small):
    model = sm.F2u()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    u = np.linspace(-0.4, 0.5, 7)
    lp, g = sp.log_posterior(post, u)
    phi = float(post.phi(jnp.asarray(u), post.initial_aux(), post.data)[0])
    log_lik = -post.n_data * np.log(np.pi * small["s"] ** 2) - phi  # the normalized complex Gaussian likelihood
    assert lp == pytest.approx(log_lik + float(model.log_prior_u(jnp.asarray(u))), rel=1e-12)
    h = 1e-6
    fd = np.array([(sp.log_posterior(post, u + h * e)[0] - sp.log_posterior(post, u - h * e)[0]) / (2 * h) for e in np.eye(7)])
    assert np.linalg.norm(g - fd) / np.linalg.norm(fd) < 1e-5


def test_potential_at_beta_zero_is_the_prior_alone_and_beta_scales_the_data_term(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    u = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
    v0, g0, _ = sp.potential(post, u, beta=0.0)
    assert v0 == pytest.approx(-float(model.log_prior_u(jnp.asarray(u))), abs=1e-12)
    np.testing.assert_allclose(g0, -np.asarray(jax.grad(model.log_prior_u)(jnp.asarray(u))), atol=1e-12)
    v1, g1, _ = sp.potential(post, u, beta=1.0)
    vh, gh, _ = sp.potential(post, u, beta=0.5)
    assert vh - v0 == pytest.approx(0.5 * (v1 - v0), rel=1e-12) and np.allclose(gh - g0, 0.5 * (g1 - g0), rtol=1e-10, atol=1e-8)


def test_fisher_matrix_matches_the_finite_difference_jacobian_and_is_positive_definite(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    u = np.array([0.2, 0.1, -0.1, 0.3, -0.2])
    J = np.asarray(post.data_jacobian(u))
    h = 1e-6
    cols = []
    for k in range(5):
        e = np.zeros(5)
        e[k] = h
        dp = np.asarray(post.clean_data(u + e)) - np.asarray(post.clean_data(u - e))
        cols.append((dp / (2 * h)).reshape(-1))
    assert np.linalg.norm(J - np.stack(cols, axis=1)) / np.linalg.norm(J) < 1e-6
    F = post.fisher(u)
    assert np.allclose(F, F.T) and np.linalg.eigvalsh(F).min() > 0
    np.testing.assert_allclose(F, (2.0 / small["s"] ** 2) * np.real(J.conj().T @ J), rtol=1e-12)


def test_map_finds_the_true_two_disk_object_and_the_gn_and_exact_hessians_agree_at_the_optimum(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    rng = np.random.default_rng(1)
    starts = np.vstack([np.zeros(5), model.start_points_u(rng, 2)])
    search = sp.map_estimate(post, starts, lbfgs_iter=200)
    best = search.best_candidate
    assert best["converged"] and best["grad_inf"] < 1e-2
    p_hat = post.physical(search.u_hat)
    truth = np.array([0.2, 0.2, 2.0, 0.0, 0.0])
    fit = sp.laplace_fit(post, search.u_hat)
    assert np.all(np.abs(p_hat - truth)[:3] < np.array([0.02, 0.01, 0.05]))  # g, r, eps_r (50 dB, the model is exactly right on this grid)
    assert best["phi"] < 2.0 * post.n_data  # a model-consistent data set: phi about N
    fit_fd = sp.laplace_fit(post, search.u_hat, hessian="fd")
    assert np.linalg.norm(fit_fd.H - fit.H) / np.linalg.norm(fit.H) < 0.05
    ev, ev_fd = sp.laplace_log_evidence(post, fit), sp.laplace_log_evidence(post, fit_fd)
    assert abs(ev - ev_fd) < 0.5


def test_model_symmetry_images_give_starts_that_return_to_the_same_mode_for_a_symmetric_object(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    search = sp.map_estimate(post, np.zeros((1, 5)), lbfgs_iter=150)
    p_hat = post.physical(search.u_hat)
    labels, images = zip(*model.symmetry_images(p_hat))
    starts = np.stack([np.asarray(model.u_from_p(jnp.asarray(q))) for q in images])
    search2 = sp.map_estimate(post, np.vstack([search.u_hat, starts]), labels=["mode", *labels], lbfgs_iter=150)
    fit = sp.laplace_fit(post, search2.u_hat)
    rep = sp.detect_multimodality(post, search2, fit)
    assert rep["multimodal"] is False  # the object is mirror symmetric (cx ~ 0): the mirror start converges back to the same mode


@pytest.mark.slow
def test_hmc_on_the_scattering_posterior_agrees_with_the_laplace_approximation(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    search = sp.map_estimate(post, np.zeros((1, 5)), lbfgs_iter=150)
    fit = sp.laplace_fit(post, search.u_hat)
    ps = sp.sample_posterior(post, fit, jax.random.key(0), n_chains=4, n_warmup=60, n_samples=250)
    sd_mcmc = ps.batch.u.reshape(-1, 5).std(axis=0)
    assert np.all(np.abs(sd_mcmc / fit.sd() - 1.0) < 0.25)
    assert ps.convergence["max_rhat"] < 1.1 and 0.5 < ps.batch.acceptance <= 1.0
    # the stored phi is the physical misfit at the draws, not the sampler's potential
    assert abs(float(ps.batch.phi.mean()) - (search.best_candidate["phi"] + 2.5)) < 6.0
    gap = sp.gap_summary(ps.p[..., 0], g_true=0.2)
    assert gap["covers_true"] or abs(gap["z_true"]) < 3.0


@pytest.mark.slow
def test_ti_on_the_scattering_posterior_matches_the_laplace_evidence_where_the_posterior_is_gaussian(small):
    model = sm.F2()
    post = sp.ShapePosterior(model, small["fwd"], small["d_obs"], small["s"], small["rend"])
    search = sp.map_estimate(post, np.zeros((1, 5)), lbfgs_iter=150)
    fit = sp.laplace_fit(post, search.u_hat, hessian="fd")
    laplace = sp.laplace_log_evidence(post, fit)
    # the geometric schedule from beta = 1e-6 (ratio 2.9 between neighbors): on the power schedule beta_k = (k / 13)^5 the variance of phi falls by 100 to 1000 times between the first
    # nonzero temperatures, the end correction is invalid there and ``ti_combine`` skips it, which leaves a trapezoid LOWER bound (33 nats below the Laplace value for this problem)
    res, _ = sp.thermodynamic_integration(post, jax.random.key(1), n_temps=14, beta_min=1e-6, n_chains=3, n_pilot=30, n_warmup=40, n_samples=100)
    # the first intervals of the schedule have E_beta[phi] ~ 1e5 to 1e6, so a short run has a Monte Carlo error of several nats (res["mc_se"]); the estimate must be consistent with it
    assert np.isfinite(res["log_z"]) and res["mc_se"] > 0 and res["monotone"]["ok"] and res["worst_rhat"] < 1.5
    assert abs(res["log_z"] - laplace) < 5.0 * res["mc_se"] + 5.0  # measured: 909.06 against 913.91 (mc_se 1.1)
    # a ratio of 2.9 between neighbors is borderline by design (variance ratio ~ 8.4 against the limit 10, and 100 draws per chain make the variance noisy: 7 of 13 intervals exceed it
    # here), so the guard skips some end corrections; the bookkeeping must be consistent
    ec = res["end_correction"]
    assert ec["n_applied"] + len(ec["skipped_intervals"]) == res["n_temps"] - 1 and res["schedule_ok"] == (not ec["skipped_intervals"])


# ----------------------------------------------------------------------------------------------
# Review follow-up (DESIGN 14.7): TI end correction, importance-sampling evidence, analytic misfit flag, ESS-aware R-hat
# ----------------------------------------------------------------------------------------------


def sharp_gaussian_ti_stats(betas, t=1.0e6):
    """Exact E_beta[phi] and Var_beta[phi] of a one-dimensional posterior that is sqrt(t) times narrower than its N(0, 1) prior: phi = a u^2 / 2 with a = t, so the power posterior
    is N(0, v) with v = 1 / (1 + beta a), E[phi] = a v / 2, Var[phi] = (a v)^2 / 2, and log Z = -log(1 + t) / 2 (lik_const = 0)."""
    v = 1.0 / (1.0 + np.asarray(betas) * t)
    return [{"beta": float(b), "mean_phi": 0.5 * t * x, "var_phi": 0.5 * (t * x) ** 2, "ess_phi": 1e12, "rhat_phi": 1.0, "rhat_phi_rank": 1.0, "acceptance": 0.8}
            for b, x in zip(betas, v)], -0.5 * np.log1p(t)


class _ZeroConst:
    lik_const = 0.0


def test_ti_end_correction_is_skipped_where_the_variance_changes_by_more_than_a_factor_of_ten():
    """The defect found in review: on a power schedule the first intervals of a sharp posterior have Var_beta[phi] falling by 20 to 350 times, the one-term Euler-Maclaurin
    correction is invalid there and (applied everywhere) put log Z ABOVE the exact value, by 10.8 nats here. Skipped there, the estimate is a trapezoid lower bound, flagged."""
    betas = sp.ti_schedule(16)
    st, exact = sharp_gaussian_ti_stats(betas)
    always = sp.ti_combine(_ZeroConst, st, var_ratio_max=np.inf)  # the old behavior
    guarded = sp.ti_combine(_ZeroConst, st)
    assert always["log_z"] - exact > 5.0  # the defect: an overshoot of several nats, far outside any claimed error
    assert always["end_correction"]["skipped_intervals"] == [] and always["schedule_ok"] is True
    ec = guarded["end_correction"]
    assert ec["var_ratio_max"] == 10.0 and ec["skipped_intervals"] == [1, 2, 3] and guarded["schedule_ok"] is False
    assert ec["n_applied"] == 15 - 3 and ec["skipped_would_add"] > 5.0  # what the skipped corrections would have added
    assert guarded["log_z"] - exact < 0.5 and abs(guarded["log_z"] - exact) < abs(always["log_z"] - exact)
    assert guarded["log_z_trapezoid"] == always["log_z_trapezoid"] and all(r is None or r > 0 for r in ec["var_ratio"])
    # the standard error does not carry the weight of a correction that was not applied
    assert guarded["mc_se"] <= always["mc_se"]


def test_geometric_ti_schedule_keeps_the_variance_ratio_valid_and_the_estimate_close():
    betas = sp.ti_schedule(16, beta_min=1e-7)
    assert betas[0] == 0.0 and betas[1] == pytest.approx(1e-7) and betas[-1] == 1.0 and len(betas) == 16
    ratios = betas[2:] / betas[1:-1]
    assert np.allclose(ratios, ratios[0]) and ratios[0] < 3.2  # 10^(7 / 14) = 3.16
    st, exact = sharp_gaussian_ti_stats(betas)
    res = sp.ti_combine(_ZeroConst, st)
    assert res["schedule_ok"] is True and res["end_correction"]["skipped_intervals"] == []
    assert abs(res["log_z"] - exact) < 0.5 < abs(res["log_z_trapezoid"] - exact)  # the correction helps where it is valid
    with pytest.raises(ValueError):
        sp.ti_schedule(16, beta_min=1.5)
    with pytest.raises(ValueError):
        sp.ti_schedule(16, beta_min=0.0)
    # a zero or negative variance estimate is never an interval where the correction can be judged valid
    broken = [dict(s) for s in st]
    broken[5]["var_phi"] = 0.0
    assert 4 in sp.ti_combine(_ZeroConst, broken)["end_correction"]["skipped_intervals"]


def test_importance_sampling_evidence_is_exact_for_the_gaussian_toy_and_good_for_a_heavy_tailed_posterior():
    toy = make_toy(seed=2, m=15, d=3, noise=0.4)
    mean, cov = toy.exact_posterior()
    res = sp.importance_log_evidence(toy, mean, cov, n_draws=400, df=5.0, scale=1.2, seed=3)
    assert abs(res["log_z"] - toy.exact_log_evidence()) < 4.0 * res["se"] + 1e-3 and res["se"] < 0.05
    assert res["weight_ess"] > 0.7 * res["n_draws"] and res["n_outside_box"] == 0
    # a core inside a halo: the Laplace evidence is off by more than 0.3, importance sampling at the Laplace fit is not
    problem = FunctionProblem(core_halo_phi, 2, prior_sd=CORE_HALO_PRIOR_SD)
    search = sp.map_estimate(problem, np.array([[0.05, -0.03]]))
    fit = sp.laplace_fit(problem, search.u_hat)
    exact = core_halo_evidence()
    assert abs(sp.laplace_log_evidence(problem, fit) - exact) > 0.3
    res = sp.importance_log_evidence(problem, fit.u_hat, fit.cov, n_draws=4000, df=5.0, scale=2.0, seed=1)
    assert abs(res["log_z"] - exact) < 4.0 * res["se"] + 0.01  # the box |u| <= 10 removes 4e-3 of the N(0, 3^2) prior mass in two dimensions
    # draws outside the box get weight zero without a solve
    wide = sp.importance_log_evidence(toy, mean, 1e4 * cov, n_draws=200, seed=0)
    assert wide["n_outside_box"] > 0 and np.isfinite(wide["log_z"])


def test_misfit_p_value_is_the_exact_null_of_the_best_fit_misfit_of_a_linear_complex_model():
    """Phi_MAP = sum |d - A theta|^2 / s^2 with N complex data and ``dim`` real parameters has the law Gamma(N - dim / 2): checked by simulation (Kolmogorov-Smirnov), and the
    replicate-free p-value is uniform; this is the null against which 30 replicates (resolution 1/31) cannot resolve p < 0.03."""
    rng = np.random.default_rng(0)
    n, dim, reps = 80, 5, 3000
    A = rng.standard_normal((n, dim)) + 1j * rng.standard_normal((n, dim))
    Ar = np.vstack([A.real, A.imag])
    Q, _ = np.linalg.qr(Ar)
    phi = np.empty(reps)
    for k in range(reps):
        e = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2.0)  # CN(0, 1): s = 1
        r = np.concatenate([e.real, e.imag])
        r = r - Q @ (Q.T @ r)  # the residual of the least-squares fit of the real parameters
        phi[k] = np.sum(r * r)
    null = sp.misfit_null(n, dim)
    assert null.mean() == n - dim / 2 and np.sqrt(null.var()) == pytest.approx(np.sqrt(n - dim / 2))
    assert phi.mean() == pytest.approx(n - dim / 2, abs=3 * np.sqrt((n - dim / 2) / reps))
    assert stats.kstest(phi, null.cdf).pvalue > 0.01
    p = np.array([sp.misfit_p_value(x, n, dim)["p_value"] for x in phi])
    assert stats.kstest(p, "uniform").pvalue > 0.01
    flag = sp.misfit_p_value(2400 + 5.0 * np.sqrt(2397.5), 2400, 5)
    assert flag["mean"] == 2397.5 and flag["z"] == pytest.approx((2400 + 5.0 * np.sqrt(2397.5) - 2397.5) / np.sqrt(2397.5)) and flag["p_value"] < 1e-5
    assert sp.misfit_p_value(2456.1, 2400, 5)["p_value"] == pytest.approx(0.116, abs=0.001)  # the 0.5 % ellipticity rung of the ladder: not a flag


def test_noise_generator_has_the_gamma_null_misfit_of_the_noise_alone():
    """add_noise at noise level s gives sum |e|^2 / s^2 ~ Gamma(N, 1): mean N, sd sqrt(N) (the reviewers' check of the generator over 1000 keys: sd 48.0 against 49.0)."""
    d = jnp.zeros((20, 30), dtype=jnp.complex128)
    s = 0.7
    phis = np.array([float(jnp.sum(jnp.abs(add_noise(jax.random.key(k), d, s)) ** 2)) / s**2 for k in range(300)])
    n = 600
    assert phis.mean() == pytest.approx(n, abs=4 * np.sqrt(n / 300))
    assert phis.std(ddof=1) == pytest.approx(np.sqrt(n), rel=0.15)


def test_expected_split_rhat_matches_simulated_ar1_chains_and_makes_the_phi_criterion_ess_aware():
    """The split R-hat of well-mixed chains with ESS_c per split sequence is 1 + (1 / ESS_c - 1 / n) / 2 (tau = 8 in 4 x 500 draws: 1.014, above 1.01 although nothing is wrong)."""
    tau, n_chains, n_draws = 8.0, 4, 500
    sim = dg.ar1_rhat_null(n_chains, n_draws, tau, n_rep=150, rank=False, seed=2)["classic"]
    ess_total = n_chains * n_draws / tau
    expected = sp.expected_split_rhat(ess_total, n_chains, n_draws)
    assert expected == pytest.approx(1.014, abs=1e-4) and sim.mean() == pytest.approx(expected, abs=0.004)
    assert np.isnan(sp.expected_split_rhat(0.0, 4, 500)) and sp.expected_split_rhat(1e12, 4, 500) == pytest.approx(1.0 - 0.5 / 250)
    # convergence_report: AR(1) chains of one distribution pass the ESS-aware rule even when the fixed limit 1.01 fails; chains that sit at different places fail both
    rng = np.random.default_rng(5)
    rho = (tau - 1) / (tau + 1)
    e = rng.standard_normal((n_chains, n_draws + 100, 3))
    x = np.zeros_like(e)
    for t_ in range(1, e.shape[1]):
        x[:, t_] = rho * x[:, t_ - 1] + np.sqrt(1 - rho**2) * e[:, t_]
    x = x[:, 100:]
    phi = x[..., 0]
    rep = sp.convergence_report(phi, x)
    assert rep["rhat_phi_expected"] == pytest.approx(sp.expected_split_rhat(rep["ess_bulk_phi"], n_chains, n_draws))
    assert rep["rhat_phi_limit"] == pytest.approx(rep["rhat_phi_expected"] + 0.01) and rep["rhat_phi_expected"] > 1.01
    assert rep["converged_strict"] == (rep["rhat_phi"] <= 1.01 and rep["rhat_phi_rank"] <= 1.01 and rep["max_rhat"] <= 1.05 and rep["worst_ess_bulk"] >= 200.0)
    assert rep["converged"] == (rep["rhat_phi"] <= rep["rhat_phi_limit"] and rep["rhat_phi_rank"] <= rep["rhat_phi_limit"] and rep["max_rhat"] <= 1.05 and rep["worst_ess_bulk"] >= 200.0)
    assert rep["converged"] is True and rep["max_rhat_rank"] >= 1.0 and 0.0 < rep["max_rhat_rank"] < 1.1
    # a chain with a different mean is not excused by the allowance
    shifted = x.copy()
    shifted[0] += 1.5
    bad = sp.convergence_report(shifted[..., 0], shifted)
    assert bad["converged"] is False and bad["converged_strict"] is False and bad["rhat_phi"] > bad["rhat_phi_limit"]
    # the bulk ESS of unmixed chains collapses, which would make the expected R-hat (and an uncapped allowance) enormous: the limit is evaluated at ESS >= ess_min
    assert bad["rhat_phi_expected"] > 1.1 and bad["rhat_phi_limit"] == pytest.approx(sp.expected_split_rhat(200.0, n_chains, n_draws) + 0.01) and bad["rhat_phi_limit"] < 1.03
    # with a large ESS the limit is the plain 1.01
    iid = rng.standard_normal((4, 2000, 2))
    big = sp.convergence_report(iid[..., 0], iid)
    assert big["rhat_phi_limit"] == pytest.approx(1.01, abs=0.002) and big["converged"] is True
