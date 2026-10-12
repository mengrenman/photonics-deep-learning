"""Tests for bayes_scatter.thetaspace: U and its gradient, the Laplace mass matrix, the zeta reparametrization, the
MAP, and an end-to-end HMC run against an exactly known Gaussian posterior (with negative controls)."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import adapt as ad
from bayes_scatter import inference as inf
from bayes_scatter import jacobian as jb
from bayes_scatter import mrf
from bayes_scatter import thetaspace as ts
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.likelihood import Target, add_noise, noise_std_from_snr
from bayes_scatter.phantoms import cylinder
from bayes_scatter.priors import GaussianField

N_PIX = 36


@pytest.fixture(scope="module")
def prob():
    """6 x 6 grid, 5 transmitters, 7 receivers, a cylinder of eps_r 1.8, 30 dB noise, tight solver tolerance."""
    grid = Grid(6, 1.2)
    setup = Setup(grid, Antennas(ring(5, 2.0), ring(7, 2.4, offset=0.2)))
    fwd = Forward(setup, tol=1e-12)
    chi_true = cylinder(grid, 0.4, 1.8)
    d_clean, _ = fwd(jnp.asarray(chi_true))
    s = noise_std_from_snr(d_clean, 30.0)
    d_obs = add_noise(jax.random.key(0), d_clean, s)
    field = GaussianField(grid, "matern", sigma=0.5, ell=0.3, nu=1.5)
    priors = {"qggmrf": mrf.QGGMRF(6, q=1.2, T=1.0, sigma_g=0.1, sigma_0=1.0),
              "matern": mrf.GaussianEnergy(field)}
    targets = {k: ts.ThetaTarget(fwd, p, d_obs, s) for k, p in priors.items()}
    theta0 = np.log(1.0 + chi_true) + 0.05 * np.random.default_rng(3).standard_normal((6, 6))
    return {"grid": grid, "setup": setup, "fwd": fwd, "chi_true": chi_true, "d_obs": d_obs, "s": s, "field": field,
            "priors": priors, "targets": targets, "theta0": jnp.asarray(theta0)}


# ---------------------------------------------------------------------------------------------
# U, its gradient and the misfit
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["qggmrf", "matern"])
def test_gradient_of_u_matches_central_differences(prob, name):
    t, th = prob["targets"][name], prob["theta0"]
    aux = t.initial_aux()
    u0, g, _ = t.potential_and_grad(th, aux)
    np.testing.assert_allclose(float(t.potential(th, aux)[0]), float(u0), rtol=1e-13)
    rng = np.random.default_rng(7)
    for _ in range(3):
        d = jnp.asarray(rng.standard_normal((6, 6)))
        d = d / jnp.linalg.norm(d)
        h = 1e-6
        fd = (float(t.potential(th + h * d, aux)[0]) - float(t.potential(th - h * d, aux)[0])) / (2 * h)
        np.testing.assert_allclose(float(jnp.sum(g * d)), fd, rtol=1e-5)


def test_u_is_the_physical_misfit_plus_the_prior_energy(prob):
    t, th = prob["targets"]["qggmrf"], prob["theta0"]
    aux = t.initial_aux()
    parts = t.energies(th, aux)
    np.testing.assert_allclose(parts["U"], parts["phi"] + parts["R"], rtol=1e-14)
    np.testing.assert_allclose(parts["U"], float(t.potential(th, aux)[0]), rtol=1e-12)
    np.testing.assert_allclose(parts["R"], float(prob["priors"]["qggmrf"].energy(th)), rtol=1e-14)
    # the misfit is that of the existing whitened Target for the same contrast map (same forward, data, noise)
    old = Target(prob["fwd"], prob["field"], "log_eps", prob["d_obs"], prob["s"])
    chi = jnp.exp(th) - 1.0
    np.testing.assert_allclose(float(t.misfit(th)[0]), float(old.misfit_of_chi(chi, aux)[0]), rtol=1e-12)
    np.testing.assert_allclose(float(t.misfit_of_chi(chi, aux)[0]), float(t.misfit(th)[0]), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(t.eps_r(th)), np.exp(np.asarray(th)), rtol=1e-14)


# ---------------------------------------------------------------------------------------------
# Laplace mass
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["qggmrf", "matern"])
def test_laplace_mass_is_gauss_newton_plus_prior_hessian_and_spd(prob, name):
    t, th, fwd, s = prob["targets"][name], prob["theta0"], prob["fwd"], prob["s"]
    mass = t.laplace_mass(th, verify_columns=6)
    assert mass.info["jacobian_check_rel_err"] < 1e-8
    # independent assembly: data part from jvp columns of the forward model, prior part from jax.hessian / precision
    chi = jnp.exp(th) - 1.0
    cols = np.asarray(jb.jvp_columns(fwd, chi, np.arange(N_PIX), np.exp(np.asarray(th))))
    Jr = np.vstack([cols.real, cols.imag])
    h_gn = (2.0 / s**2) * Jr.T @ Jr
    if name == "qggmrf":
        pr = prob["priors"][name]
        h_prior = np.asarray(jax.hessian(lambda x: pr._energy(x.reshape(6, 6)))(th.ravel()))
    else:
        h_prior = prob["priors"][name].precision
    scale = np.abs(h_gn).max()
    np.testing.assert_allclose(mass.H, h_gn + h_prior, rtol=1e-7, atol=1e-8 * scale)
    # SPD, symmetric, and the Cholesky factor reproduces H
    assert mass.info["asymmetry"] < 1e-12
    np.testing.assert_allclose(mass.chol @ mass.chol.T, mass.H, rtol=1e-12, atol=1e-12 * scale)
    assert np.all(np.triu(mass.chol, 1) == 0.0) and np.linalg.eigvalsh(mass.H).min() > 0.0
    np.testing.assert_allclose(mass.theta_variance().ravel(), np.diag(np.linalg.inv(mass.H)), rtol=1e-8)
    np.testing.assert_allclose(mass.theta_std() ** 2, mass.theta_variance())
    assert mass.theta_hat.shape == (6, 6) and mass.info["n_solves"] == fwd.n_tx + fwd.n_rx


def test_laplace_mass_rejects_a_matrix_that_is_not_positive_definite(prob, monkeypatch):
    t = prob["targets"]["qggmrf"]
    monkeypatch.setattr(t.prior, "hessian", lambda theta: -1e12 * np.eye(N_PIX))
    with pytest.raises(ValueError, match="not positive definite"):
        t.laplace_mass(prob["theta0"])


# ---------------------------------------------------------------------------------------------
# Reparametrization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["qggmrf", "matern"])
def test_reparametrized_potential_plus_half_zeta_norm_is_u_of_theta(prob, name):
    t = prob["targets"][name]
    mass = t.laplace_mass(prob["theta0"])
    zt = t.zeta_target(mass)
    aux = t.initial_aux()
    rng = np.random.default_rng(8)
    for _ in range(3):
        z = jnp.asarray(0.7 * rng.standard_normal((6, 6)))
        phi_p, g_z, _ = zt.phi_fn(z, aux)
        theta = zt.to_theta(z)
        u_val = float(t.potential(theta, aux)[0])
        np.testing.assert_allclose(float(phi_p) + 0.5 * float(jnp.sum(z * z)), u_val, rtol=1e-12)
        np.testing.assert_allclose(float(zt.phi_fn_no_grad(z, aux)[0]), float(phi_p), rtol=1e-13)
        # theta = theta_hat + L_H^-T zeta  <=>  L_H^T (theta - theta_hat) = zeta
        np.testing.assert_allclose(mass.chol.T @ (np.asarray(theta) - mass.theta_hat).ravel(),
                                   np.asarray(z).ravel(), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(np.asarray(zt.to_zeta(theta)), np.asarray(z), atol=1e-11)
        # gradient: autodiff of the composition, and central differences
        auto = jax.grad(lambda zz: t.potential(zt.to_theta(zz), aux)[0] - 0.5 * jnp.sum(zz * zz))(z)
        np.testing.assert_allclose(np.asarray(g_z), np.asarray(auto), rtol=1e-8, atol=1e-8)
        d = jnp.asarray(rng.standard_normal((6, 6)))
        d = d / jnp.linalg.norm(d)
        fd = (float(zt.phi_fn(z + 1e-6 * d, aux)[0]) - float(zt.phi_fn(z - 1e-6 * d, aux)[0])) / 2e-6
        np.testing.assert_allclose(float(jnp.sum(g_z * d)), fd, rtol=1e-5, atol=1e-6)
    rec = zt.record(z)
    np.testing.assert_allclose(np.asarray(rec[0]), np.asarray(theta), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(rec[1]), np.exp(np.asarray(theta)), rtol=1e-13)
    assert zt.overdispersed_starts(jax.random.key(0), 3, 2.0).shape == (3, 6, 6)


def test_a_custom_center_changes_the_coordinates_but_not_the_target(prob):
    t = prob["targets"]["qggmrf"]
    mass = t.laplace_mass(prob["theta0"])
    shifted = t.zeta_target(mass, center=prob["theta0"] + 0.1)
    aux = t.initial_aux()
    z = jnp.asarray(np.random.default_rng(9).standard_normal((6, 6)))
    theta = shifted.to_theta(z)
    np.testing.assert_allclose(float(shifted.phi_fn(z, aux)[0]) + 0.5 * float(jnp.sum(z * z)),
                               float(t.potential(theta, aux)[0]), rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------------------------


def test_smooth_start_lowers_the_misfit_and_map_theta_finds_a_stationary_point(prob):
    t, fwd = prob["targets"]["qggmrf"], prob["fwd"]
    th0, info0 = ts.smooth_start(fwd, prob["d_obs"], prob["s"], prob["grid"], maxiter=30)
    assert th0.shape == (6, 6) and np.all(np.isfinite(np.asarray(th0))) and info0["nit"] <= 30
    assert float(t.misfit(th0)[0]) < float(t.misfit(jnp.zeros((6, 6)))[0])
    th, info = ts.map_theta(t, th0, maxiter=500, gtol=1e-7)
    # L-BFGS-B stops on its relative-reduction test with the gradient max-norm at about 3e-4 here, while the
    # gradient components of the data term alone are ~10^3
    assert info["U"] <= t.energies(th0)["U"] + 1e-9 and info["grad_inf"] < 2e-3
    np.testing.assert_allclose(info["U"], info["phi"] + info["R"], rtol=1e-12)
    g = np.asarray(t.potential_and_grad(th, t.initial_aux())[1])
    np.testing.assert_allclose(np.abs(g).max(), info["grad_inf"], rtol=1e-3)  # warm-start noise
    # a MAP of a convex-prior problem: a random perturbation does not lower U
    rng = np.random.default_rng(10)
    for _ in range(3):
        assert t.energies(th + 1e-3 * rng.standard_normal((6, 6)))["U"] >= info["U"] - 1e-6


# ---------------------------------------------------------------------------------------------
# Linear-Gaussian end-to-end: Born model, identity link, cropped Gaussian prior => exact Gaussian posterior
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gauss(prob):
    fwd, field = prob["fwd"], prob["field"]
    prior = mrf.GaussianEnergy(field)
    A = np.asarray(fwd.born_matrix())
    chi_true = np.asarray(prob["chi_true"])
    d = A @ chi_true.ravel()
    s = noise_std_from_snr(d.reshape(fwd.n_tx, fwd.n_rx), 15.0)  # moderate SNR: informative but not stiff
    d_obs = np.asarray(add_noise(jax.random.key(4), d.reshape(fwd.n_tx, fwd.n_rx), s))
    tgt = ts.ThetaTarget(fwd, prior, d_obs, s, link="identity", use_born=True)
    Ar, y = np.vstack([A.real, A.imag]), np.concatenate([d_obs.real.ravel(), d_obs.imag.ravel()])
    Lam = prior.precision + (2.0 / s**2) * Ar.T @ Ar
    mean = np.linalg.solve(Lam, (2.0 / s**2) * Ar.T @ y)  # prior mean 0
    cov = np.linalg.inv(Lam)
    return {"prior": prior, "target": tgt, "Lam": Lam, "mean": mean, "cov": cov, "s": s}


def test_laplace_mass_of_a_linear_gaussian_problem_is_the_exact_posterior_precision(gauss):
    mass = gauss["target"].laplace_mass(jnp.zeros((6, 6)))
    np.testing.assert_allclose(mass.H, gauss["Lam"], rtol=1e-10, atol=1e-10 * np.abs(gauss["Lam"]).max())
    assert mass.info["n_solves"] == 0
    th, info = ts.map_theta(gauss["target"], jnp.zeros((6, 6)), maxiter=2000, gtol=1e-9)
    sd = np.sqrt(np.diag(gauss["cov"]))
    assert np.max(np.abs(np.asarray(th).ravel() - gauss["mean"]) / sd) < 0.05  # MAP = posterior mean (Gaussian)


def _agrees(cmp):
    """The pass criterion for a run against the exact Gaussian: z-scores of mean and variance O(1), variance ratio ~1,
    converged chains."""
    return (cmp["z_mean_rms"] < 2.0 and cmp["z_mean_max"] < 5.5 and cmp["z_var_rms"] < 2.0 and cmp["z_var_max"] < 5.5
            and 0.93 < cmp["var_ratio_mean"] < 1.07 and cmp["rhat_max"] < 1.03 and cmp["ess_min"] > 200)


def _run_hmc(zt, n_warmup=300, n_samples=3000, seed=11):
    z0 = 1.5 * jax.random.normal(jax.random.key(seed), (4, 6, 6), dtype=jnp.float64)
    res, _ = inf.sample(zt, "hmc", jax.random.key(seed + 1), z0, n_warmup=n_warmup, n_samples=n_samples,
                        step0=0.3, thin=1, n_leapfrog=10, jitter=0.2)
    return res


def _distorted_mass(gauss, theta_hat):
    """A deliberately wrong mass matrix: the exact precision with eigenvalues scaled by factors in [1/4, 4]."""
    w, V = np.linalg.eigh(gauss["Lam"])
    rng = np.random.default_rng(12)
    H = (V * (w * np.exp(rng.uniform(-np.log(4), np.log(4), w.size)))) @ V.T
    H = 0.5 * (H + H.T)
    return ts.LaplaceMass(np.asarray(theta_hat), H, np.linalg.cholesky(H), {})


def test_hmc_in_zeta_recovers_the_exact_posterior_and_a_broken_target_fails(gauss):
    """Exactness for any center and any SPD mass: an off-center, distorted mass still gives the right posterior."""
    tgt = gauss["target"]
    sd = np.sqrt(np.diag(gauss["cov"]))
    theta_off = gauss["mean"] + 3.0 * sd * np.random.default_rng(13).standard_normal(N_PIX)
    mass = _distorted_mass(gauss, theta_off.reshape(6, 6))
    zt = tgt.zeta_target(mass)
    res = _run_hmc(zt)
    th = np.asarray(res.samples[:, :, 0])  # (chains, N, 6, 6)
    cmp = inf.compare_to_gaussian(th.reshape(4, -1, N_PIX), gauss["mean"], np.diag(gauss["cov"]))
    assert _agrees(cmp)
    # pooled Welford moments of the run agree with the exact posterior covariance (diagonal)
    mean_c, var_c = np.asarray(res.mean[:, 0]), np.asarray(res.var[:, 0])
    pooled = var_c.mean(axis=0) + mean_c.var(axis=0)
    np.testing.assert_allclose(pooled.ravel(), np.diag(gauss["cov"]), rtol=0.12)

    # negative control: forget the -|zeta|^2/2 of Phi'. The sampler then targets exp(-U - |zeta|^2/2), a posterior
    # pulled toward the (deliberately off-center) center of the map and with too small variances, so the same
    # criterion must fail by a wide margin.
    def broken_phi(z, aux):
        u_val, g, aux_new = tgt._potential_and_grad(zt.wmap.to_theta(z), aux)
        return u_val, zt.wmap.grad_to_zeta(g), aux_new

    broken = types.SimpleNamespace(phi_fn=jax.jit(broken_phi), phi_fn_no_grad=None, record=zt.record,
                                   initial_aux=zt.initial_aux, use_born=True)
    bad = inf.compare_to_gaussian(np.asarray(_run_hmc(broken).samples[:, :, 0]).reshape(4, -1, N_PIX),
                                  gauss["mean"], np.diag(gauss["cov"]))
    assert not _agrees(bad)
    assert bad["z_mean_rms"] > 5.0 and bad["z_var_rms"] > 5.0


# ---------------------------------------------------------------------------------------------
# Milestone 5: ZetaTarget in the coordinates of an arbitrary affine map (adapt.AffineMap)
# ---------------------------------------------------------------------------------------------


def test_zeta_target_with_an_affine_map_is_u_of_theta_and_flat_potential_matches(prob):
    t = prob["targets"]["qggmrf"]
    rng = np.random.default_rng(21)
    A = rng.standard_normal((N_PIX, N_PIX))
    cov = 0.01 * (A @ A.T + N_PIX * np.eye(N_PIX))
    amap = ad.AffineMap.from_covariance(np.asarray(prob["theta0"]), cov)
    zt = t.zeta_target_with_map(amap)
    aux = t.initial_aux()
    assert zt.mass is None and zt.center.shape == (6, 6) and zt.wmap is amap
    z = jnp.asarray(0.6 * rng.standard_normal((6, 6)))
    phi_p, g_z, _ = zt.phi_fn(z, aux)
    theta = zt.to_theta(z)
    np.testing.assert_allclose(float(phi_p) + 0.5 * float(jnp.sum(z * z)), float(t.potential(theta, aux)[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(zt.to_zeta(theta)), np.asarray(z), atol=1e-10)
    auto = jax.grad(lambda zz: t.potential(zt.to_theta(zz), aux)[0] - 0.5 * jnp.sum(zz * zz))(z)
    np.testing.assert_allclose(np.asarray(g_z), np.asarray(auto), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(np.asarray(zt.record(z)[0]), np.asarray(theta), rtol=1e-13)
    # the flat potential of adapt: same value, the gradient raveled
    u_flat, g_flat, _ = t.potential_flat()(theta.ravel(), aux)
    u_ref, g_ref, _ = t.potential_and_grad(theta, aux)
    np.testing.assert_allclose(float(u_flat), float(u_ref), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(g_flat), np.asarray(g_ref).ravel(), rtol=1e-12)
    # the same map through adapt.zeta_phi_fn gives the same sampler potential and gradient
    phi_flat = ad.zeta_phi_fn(t.potential_flat(), amap.center.ravel(), amap.chol)
    p2, g2, _ = phi_flat(z.ravel(), aux)
    np.testing.assert_allclose(float(p2), float(phi_p), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(g2), np.asarray(g_z).ravel(), rtol=1e-9, atol=1e-9)


def test_exact_posterior_map_makes_the_zeta_potential_flat_and_a_wrong_map_still_samples_the_exact_posterior(gauss):
    """Linear-Gaussian problem: the map with the exact posterior mean and covariance gives Phi' = const (zeta ~ N(0, I) exactly);
    a distorted, off-center map gives the same posterior from adapt.sample_chain (variable-length HMC), whatever the map."""
    tgt = gauss["target"]
    exact = ad.AffineMap.from_covariance(gauss["mean"].reshape(6, 6), gauss["cov"])
    zt = tgt.zeta_target_with_map(exact)
    aux = tgt.initial_aux()
    z = jnp.asarray(np.random.default_rng(22).standard_normal((6, 6)))
    p0 = float(zt.phi_fn(jnp.zeros((6, 6)), aux)[0])
    p1, g1, _ = zt.phi_fn(z, aux)
    np.testing.assert_allclose(float(p1), p0, rtol=1e-9, atol=1e-7)  # U(mean + A z) = U_min + |z|^2/2: Phi' is constant
    np.testing.assert_allclose(np.asarray(g1), 0.0, atol=1e-6)
    sd = np.sqrt(np.diag(gauss["cov"]))
    theta_off = gauss["mean"] + 3.0 * sd * np.random.default_rng(13).standard_normal(N_PIX)
    mass = _distorted_mass(gauss, theta_off.reshape(6, 6))
    wrong = ad.AffineMap.from_covariance(theta_off.reshape(6, 6), np.linalg.inv(mass.H))
    ztw = tgt.zeta_target_with_map(wrong)
    run = jax.jit(lambda k, x0: ad.sample_chain(k, ztw.phi_fn, x0, aux, eps=0.3, traj_length=1.75, max_leapfrog=40, n_samples=3000,
                                                record_fn=ztw.record))
    keys = jax.random.split(jax.random.key(31), 4)
    starts = 1.5 * jax.random.normal(jax.random.key(32), (4, 6, 6), dtype=jnp.float64)
    outs = [run(keys[c], starts[c]) for c in range(4)]
    th = np.stack([np.asarray(o["samples"][:, 0]) for o in outs]).reshape(4, -1, N_PIX)
    cmp = inf.compare_to_gaussian(th, gauss["mean"], np.diag(gauss["cov"]))
    assert _agrees(cmp)
