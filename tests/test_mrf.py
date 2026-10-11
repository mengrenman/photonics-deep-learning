"""Tests for bayes_scatter.mrf: q-GGMRF energy, cropped Gaussian energies, whitening and the prior-only sampler."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from bayes_scatter import mrf
from bayes_scatter.geometry import Grid
from bayes_scatter.priors import GaussianField

# ---------------------------------------------------------------------------------------------
# Neighborhood
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 5])
def test_neighbor_pairs_are_the_unordered_8_neighbor_pairs_with_free_boundary(n):
    i, j, b = mrf.neighbor_pairs(n)
    assert len(i) == 2 * n * (n - 1) + 2 * (n - 1) ** 2
    pairs = {frozenset((int(a), int(c))) for a, c in zip(i, j)}
    assert len(pairs) == len(i) and not np.any(i == j)  # each unordered pair once
    deg = np.bincount(np.concatenate([i, j]), minlength=n * n).reshape(n, n)
    if n >= 3:
        assert deg[0, 0] == 3 and deg[0, 1] == 5 and deg[1, 1] == 8  # corner, edge, interior: no wrap-around
    for a, c, w in zip(i, j, b):  # b = 1 for axis-aligned pairs, 1/sqrt(2) for diagonals
        (ai, aj), (ci, cj) = divmod(int(a), n), divmod(int(c), n)
        assert max(abs(ai - ci), abs(aj - cj)) == 1
        np.testing.assert_allclose(w, 1.0 if (ai == ci or aj == cj) else 1.0 / np.sqrt(2.0))


# ---------------------------------------------------------------------------------------------
# q-GGMRF energy
# ---------------------------------------------------------------------------------------------


def _direct_energy(theta, mu, q, T, sg, s0):
    """Loop implementation of the DESIGN 11.2 formula with the exact |D|^(2-q) (no smoothing)."""
    n = theta.shape[0]
    e = 0.0
    for a, c in itertools.product(range(n), range(n)):
        for da, dc, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1 / np.sqrt(2)), (1, -1, 1 / np.sqrt(2))):
            a2, c2 = a + da, c + dc
            if a2 < n and 0 <= c2 < n:
                d = theta[a, c] - theta[a2, c2]
                e += w * (abs(d) ** 2 / (2 * sg**2)) / (1.0 + abs(d / (T * sg)) ** (2.0 - q))
    return e + np.sum((theta - mu) ** 2) / (2 * s0**2)


@pytest.mark.parametrize("q", [1.0, 1.2, 1.7, 2.0])
def test_qggmrf_energy_matches_the_design_formula(q):
    rng = np.random.default_rng(1)
    theta = 0.4 * rng.standard_normal((5, 5))
    prior = mrf.QGGMRF(5, q=q, T=0.8, sigma_g=0.15, sigma_0=0.9, mu=0.1)
    np.testing.assert_allclose(float(prior.energy(theta)), _direct_energy(theta, 0.1, q, 0.8, 0.15, 0.9),
                               rtol=1e-12)


def test_qggmrf_value_gradient_hvp_and_hessian_are_finite_at_zero_difference():
    n = 6
    prior = mrf.QGGMRF(n, q=1.2, T=1.0, sigma_g=0.05, sigma_0=1.0, mu=0.3)
    for theta in (jnp.full((n, n), 0.3), jnp.full((n, n), 0.3) + 1e-9 * jnp.arange(n * n).reshape(n, n)):
        assert np.isfinite(float(prior.energy(theta)))
        g = np.asarray(prior.grad(theta))
        assert np.all(np.isfinite(g))
        v = jnp.asarray(np.random.default_rng(0).standard_normal((n, n)))
        assert np.all(np.isfinite(np.asarray(prior.hvp(theta, v))))
        H = prior.hessian(theta)
        assert np.all(np.isfinite(H)) and np.all(np.isfinite(np.asarray(jax.hessian(
            lambda x: prior._energy(x.reshape(n, n)))(theta.ravel()))))
    # at exactly Delta = 0 the gradient vanishes and the Hessian is the weighted graph Laplacian / sigma_g^2 + I
    const = jnp.full((n, n), 0.3)
    np.testing.assert_allclose(np.asarray(prior.grad(const)), 0.0, atol=1e-12)
    i, j, b = mrf.neighbor_pairs(n)
    lap = np.zeros((n * n, n * n))
    for a, c, w in zip(i, j, b):
        lap[a, a] += w
        lap[c, c] += w
        lap[a, c] -= w
        lap[c, a] -= w
    np.testing.assert_allclose(prior.hessian(const), lap / 0.05**2 + np.eye(n * n), rtol=1e-6)


def test_pair_potential_limits_quadratic_for_small_and_power_q_for_large_differences():
    q, T, sg = 1.2, 1.0, 0.1
    small = np.array([1e-6, 1e-4, 1e-3]) * sg
    r = np.asarray(mrf.rho(jnp.asarray(small), q, T, sg))
    quad = small**2 / (2 * sg**2)
    rel = np.abs(r / quad - 1.0)
    assert np.all(rel <= (small / (T * sg)) ** (2 - q) * 1.01)  # first correction is |D/(T sg)|^(2-q)
    assert rel[-1] < 0.02
    big = np.array([1e2, 1e4, 1e6]) * sg
    r = np.asarray(mrf.rho(jnp.asarray(big), q, T, sg))
    power = big**q * (T * sg) ** (2 - q) / (2 * sg**2)  # ~ |D|^q for |D| >> T sg
    assert np.all(np.abs(r / power - 1.0) <= 1.01 * (T * sg / big) ** (2 - q))  # rho/power = 1/(1 + x^-(2-q))
    # growth exponent: d log rho / d log |D| -> q far in the tail, 2 in the quadratic region
    d = jnp.array([1e5 * sg, 2e5 * sg])
    slope = np.log(float(mrf.rho(d[1], q, T, sg)) / float(mrf.rho(d[0], q, T, sg))) / np.log(2.0)
    np.testing.assert_allclose(slope, q, atol=2e-3)
    # q = 2: the denominator is 1 + 1, so rho = D^2 / (4 sg^2) at every D
    x = jnp.array([1e-3, 0.1, 5.0])
    np.testing.assert_allclose(np.asarray(mrf.rho(x, 2.0, T, sg)), np.asarray(x) ** 2 / (4 * sg**2), rtol=1e-9)


@pytest.mark.parametrize("q,T,sg", [(1.0, 1.0, 0.05), (1.2, 1.0, 0.1), (1.2, 0.3, 0.1), (1.5, 3.0, 0.1), (1.9, 1.0, 0.1)])
def test_pair_potential_is_convex_and_so_is_the_energy(q, T, sg):
    d = jnp.asarray(np.linspace(-4.0, 4.0, 40001))
    rpp = jax.vmap(jax.grad(jax.grad(lambda x: mrf.rho(x, q, T, sg))))(d)
    assert float(rpp.min()) >= 0.0
    prior = mrf.QGGMRF(6, q=q, T=T, sigma_g=sg, sigma_0=1.0)
    rng = np.random.default_rng(3)
    for _ in range(20):  # midpoint convexity R(tx + (1-t)y) <= t R(x) + (1-t) R(y)
        x, y = rng.standard_normal((2, 6, 6)) * rng.uniform(0.02, 1.0)
        t = rng.uniform()
        lhs = float(prior.energy(t * x + (1 - t) * y))
        rhs = t * float(prior.energy(x)) + (1 - t) * float(prior.energy(y))
        assert lhs <= rhs + 1e-9 * abs(rhs)
    eig = np.linalg.eigvalsh(prior.hessian(rng.standard_normal((6, 6)) * 0.2))
    assert eig.min() >= 1.0 / prior.sigma_0**2 - 1e-9  # Hessian >= I / sigma_0^2


def test_qggmrf_closed_form_hessian_and_hvp_match_autodiff():
    n = 4
    prior = mrf.QGGMRF(n, q=1.3, T=0.7, sigma_g=0.2, sigma_0=0.8, mu=-0.1)
    rng = np.random.default_rng(4)
    for scale in (0.0, 0.05, 0.6):
        theta = jnp.asarray(scale * rng.standard_normal((n, n)))
        auto = np.asarray(jax.hessian(lambda x: prior._energy(x.reshape(n, n)))(theta.ravel()))
        np.testing.assert_allclose(prior.hessian(theta), auto, rtol=1e-9, atol=1e-9 * np.abs(auto).max())
        v = jnp.asarray(rng.standard_normal((n, n)))
        np.testing.assert_allclose(np.asarray(prior.hvp(theta, v)).ravel(), auto @ np.asarray(v).ravel(),
                                   rtol=1e-9, atol=1e-9 * np.abs(auto).max())


def test_qggmrf_rejects_bad_parameters():
    for kw in (dict(q=0.9), dict(q=2.1), dict(T=0.0), dict(sigma_g=-1.0), dict(sigma_0=0.0)):
        with pytest.raises(ValueError):
            mrf.QGGMRF(4, **kw)


# ---------------------------------------------------------------------------------------------
# Cropped Gaussian energies
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["matern", "gmrf"])
def test_cropped_gaussian_energy_is_minus_log_density_up_to_a_constant(kind):
    grid = Grid(6, 1.2)
    field = GaussianField(grid, kind, sigma=0.5, ell=0.3, nu=1.5, mean=0.2)
    prior = mrf.GaussianEnergy(field)
    C = field.covariance_matrix()
    mu = np.full(36, 0.2)
    dist = multivariate_normal(mean=mu, cov=C)
    rng = np.random.default_rng(5)
    thetas = [0.2 + 0.4 * rng.standard_normal((6, 6)) for _ in range(4)]
    e = np.array([float(prior.energy(t)) for t in thetas])
    lp = np.array([dist.logpdf(t.ravel()) for t in thetas])
    np.testing.assert_allclose(e + lp, e[0] + lp[0], rtol=0, atol=1e-8)  # constant offset
    np.testing.assert_allclose(e + lp, -prior.log_norm_const, atol=1e-8)  # R = -log N(theta) - log_norm_const
    P = np.linalg.inv(C)
    t = thetas[0]
    np.testing.assert_allclose(np.asarray(prior.grad(t)).ravel(), P @ (t.ravel() - mu), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(prior.hessian(t), P, rtol=1e-7, atol=1e-8 * np.abs(P).max())
    np.testing.assert_allclose(np.asarray(prior.hvp(t, jnp.ones((6, 6)))).ravel(), P @ np.ones(36), rtol=1e-7,
                               atol=1e-8)
    np.testing.assert_allclose(prior.prior_std(), 0.5, rtol=1e-9)  # stationary marginal std = sigma
    assert prior.describe()["kind"] == kind


def test_cropped_gaussian_prior_draws_have_the_covariance():
    grid = Grid(5, 1.0)
    prior = mrf.GaussianEnergy(GaussianField(grid, "matern", sigma=0.4, ell=0.3, mean=0.1))
    th = np.asarray(prior.sample(jax.random.key(0), 20000)).reshape(20000, -1)
    C = prior.chol @ prior.chol.T
    np.testing.assert_allclose(th.mean(axis=0), 0.1, atol=4 * 0.4 / np.sqrt(20000))
    np.testing.assert_allclose(np.cov(th.T), C, atol=0.02)
    assert np.asarray(prior.sample(jax.random.key(1))).shape == (5, 5)


# ---------------------------------------------------------------------------------------------
# Whitening and the prior-only sampler
# ---------------------------------------------------------------------------------------------


def test_whitened_map_is_the_inverse_square_root_of_the_mass():
    rng = np.random.default_rng(6)
    n = 3
    A = rng.standard_normal((9, 9))
    H = A @ A.T + 9 * np.eye(9)
    L = np.linalg.cholesky(H)
    center = rng.standard_normal((n, n))
    w = mrf.WhitenedMap(center, L)
    z = jnp.asarray(rng.standard_normal((n, n)))
    np.testing.assert_allclose(np.asarray(w.to_zeta(w.to_theta(z))), np.asarray(z), atol=1e-12)
    # theta - center = L^-T zeta, so Cov(theta) = L^-T L^-1 = H^-1 for zeta ~ N(0, I)
    M = np.stack([np.asarray(w.to_theta(jnp.asarray(e.reshape(n, n)))).ravel() - center.ravel() for e in np.eye(9)],
                 axis=1)
    np.testing.assert_allclose(M @ M.T, np.linalg.inv(H), rtol=1e-10, atol=1e-12)
    # chain rule for the gradient: d/dzeta f(theta(zeta)) = L^-1 grad f
    f = lambda th: jnp.sum(jnp.sin(th) * jnp.arange(9.0).reshape(n, n))
    g_direct = jax.grad(lambda zz: f(w.to_theta(zz)))(z)
    np.testing.assert_allclose(np.asarray(w.grad_to_zeta(jax.grad(f)(w.to_theta(z)))), np.asarray(g_direct),
                               rtol=1e-10, atol=1e-12)


def test_prior_sampler_recovers_the_std_of_a_gaussian_prior():
    grid = Grid(4, 1.0)
    prior = mrf.GaussianEnergy(GaussianField(grid, "matern", sigma=0.5, ell=0.3, mean=0.2))
    out = mrf.sample_prior(prior, jax.random.key(0), n_chains=4, n_warmup=100, n_samples=600)
    assert out["ess_min"] > 400 and out["rhat_max"] < 1.02
    np.testing.assert_allclose(out["theta_std"], 0.5, rtol=0.08)
    np.testing.assert_allclose(out["theta_mean"], 0.2, atol=5 * 0.5 / np.sqrt(out["ess_min"]))
    # eps_r = exp(theta) is log-normal: mean exp(mu + sigma^2/2)
    np.testing.assert_allclose(out["eps_mean"].mean(), np.exp(0.2 + 0.5**2 / 2), rtol=0.08)


def _quadrature_std(prior, half_width=4.0, m=41):
    """Exact marginal std of pixel 0 of the 2 x 2 q-GGMRF prior (mean 0) by trapezoidal quadrature in 4-D.

    The integrand is smooth and decays like exp(-theta^2 / (2 sigma_0^2)), so the trapezoid rule on [-4, 4]^4 is
    accurate to far better than the 3% the test asks for.
    """
    g = np.linspace(-half_width, half_width, m)
    t0, t1, t2 = np.meshgrid(g, g, g, indexing="ij")
    energy = jax.jit(jax.vmap(lambda x: prior._energy(x.reshape(2, 2))))
    mass = second = 0.0
    for v in g:  # slab by slab in theta_3
        th = np.stack([t0, t1, t2, np.full_like(t0, v)], axis=-1).reshape(-1, 4)
        w = np.exp(-np.asarray(energy(jnp.asarray(th))))
        mass += w.sum()
        second += (w * th[:, 0] ** 2).sum()
    return float(np.sqrt(second / mass))


def test_prior_sampler_agrees_with_exact_quadrature_for_a_non_gaussian_qggmrf():
    """2 x 2 q-GGMRF, q = 1.2: HMC on R alone against 4-D quadrature of the marginal variance, and the Gaussian
    approximation at the mean is visibly off (so the check is not vacuous)."""
    prior = mrf.QGGMRF(2, q=1.2, T=1.0, sigma_g=0.5, sigma_0=1.0)
    exact = _quadrature_std(prior)
    out = mrf.sample_prior(prior, jax.random.key(1), n_chains=4, n_warmup=200, n_samples=3000, thin=2)
    assert out["rhat_max"] < 1.02
    np.testing.assert_allclose(out["theta_std"], exact, rtol=0.03)
    gaussian = np.sqrt(np.diag(np.linalg.inv(prior.hessian(prior.mean))))
    assert abs(gaussian[0] / exact - 1.0) > 0.05  # the Laplace/Gaussian approximation would fail the 3% check
