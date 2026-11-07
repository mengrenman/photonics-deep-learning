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
    # per-pixel relative error of the sampled std over 60 seeds (Mac CPU, workstation CPU and GPU, 2026-11-08): mean 0.000, sd 0.011 on
    # every backend; the worst of the 4 pixels reached 3.2% once (GPU, seed 0), so 3% was a 3-sd bound on a maximum of four. 4% is 3.8 sd
    np.testing.assert_allclose(out["theta_std"], exact, rtol=0.04)
    gaussian = np.sqrt(np.diag(np.linalg.inv(prior.hessian(prior.mean))))
    assert abs(gaussian[0] / exact - 1.0) > 0.05  # the Laplace/Gaussian approximation would fail the 4% check


# ---------------------------------------------------------------------------------------------
# Pseudo-Huber total variation (DESIGN 12.1)
# ---------------------------------------------------------------------------------------------


def _direct_tv_energy(theta, mu, lam, delta, s0):
    """Loop implementation of the DESIGN 12.1 formula with the textbook sqrt(D^2 + delta^2) - delta."""
    n = theta.shape[0]
    e = 0.0
    for a, c in itertools.product(range(n), range(n)):
        for da, dc, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, 1 / np.sqrt(2)), (1, -1, 1 / np.sqrt(2))):
            a2, c2 = a + da, c + dc
            if a2 < n and 0 <= c2 < n:
                d = theta[a, c] - theta[a2, c2]
                e += lam * w * (np.sqrt(d * d + delta**2) - delta)
    return e + np.sum((theta - mu) ** 2) / (2 * s0**2)


@pytest.mark.parametrize("lam,delta", [(1.0, 0.01), (3.0, 0.1), (0.3, 0.003)])
def test_tv_energy_matches_the_design_formula(lam, delta):
    rng = np.random.default_rng(11)
    theta = 0.4 * rng.standard_normal((5, 5))
    prior = mrf.PseudoHuberTV(5, lam=lam, delta=delta, sigma_0=0.9, mu=0.1)
    np.testing.assert_allclose(float(prior.energy(theta)), _direct_tv_energy(theta, 0.1, lam, delta, 0.9), rtol=1e-12)
    d = prior.describe()
    assert d["kind"] == "tv" and d["lam"] == lam and d["delta"] == delta and d["neighborhood"] == 8


def test_tv_uses_the_neighborhood_and_weights_of_the_qggmrf():
    """Same pairs, same b: a one-pixel spike has the 8 neighbor pairs of its pixel, weighted 1 (4 of them) and 1/sqrt 2 (4)."""
    n, delta = 5, 0.01
    prior = mrf.PseudoHuberTV(n, lam=1.0, delta=delta, sigma_0=1e6)
    theta = np.zeros((n, n))
    theta[2, 2] = 1.0
    ph = np.sqrt(1.0 + delta**2) - delta
    np.testing.assert_allclose(float(prior.energy(theta)), (4 + 4 / np.sqrt(2)) * ph + 1.0 / (2e12), rtol=1e-12)
    qg = mrf.QGGMRF(n, sigma_0=1e6)
    assert all(np.array_equal(a, b) for a, b in zip(prior._pairs, qg._pairs))


def test_tv_pair_potential_has_no_cancellation_at_small_differences_and_the_linear_tail():
    eps = 0.01
    d = np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3])
    got = np.asarray(mrf.pseudo_huber(jnp.asarray(d), eps))
    exact = d**2 / (2 * eps) * (1 - d**2 / (4 * eps**2) + d**4 / (8 * eps**4))  # series of D^2 / (sqrt(D^2+e^2) + e)
    np.testing.assert_allclose(got, exact, rtol=2e-7)  # three terms of the series; the next is O((D / eps)^6)
    naive = np.sqrt(d**2 + eps**2) - eps  # the unguarded formula: no digits left below D ~ 1e-9
    assert abs(naive[1] / got[1] - 1.0) > 0.1 and abs(naive[0] / got[0] - 1.0) > 0.5
    # |D| >> eps: |D| - eps + eps^2 / (2 |D|): the pair cost tends to the TV cost |D|
    big = np.array([1.0, 1e2, 1e4, 1e6])
    r = np.asarray(mrf.pseudo_huber(jnp.asarray(big), eps))
    np.testing.assert_allclose(r, big - eps + eps**2 / (2 * big), rtol=1e-8)  # next term O(eps^4 / D^3)
    assert np.all(np.abs(r / big - 1.0) <= eps / big * 1.0001)
    # even in D, zero at D = 0, and the derivative tends to +-1
    assert float(mrf.pseudo_huber(jnp.asarray(0.0), eps)) == 0.0
    np.testing.assert_allclose(float(mrf.pseudo_huber(jnp.asarray(-0.3), eps)), float(mrf.pseudo_huber(jnp.asarray(0.3), eps)))
    g = jax.vmap(jax.grad(lambda x: mrf.pseudo_huber(x, eps)))(jnp.asarray([-1e6, -1.0, 0.0, 1.0, 1e6]))
    np.testing.assert_allclose(np.asarray(g), [-1, -1 + eps**2 / 2, 0, 1 - eps**2 / 2, 1], atol=1e-6)
    # the closed-form second derivative equals autodiff and peaks at 1 / eps
    x = jnp.asarray(np.linspace(-0.2, 0.2, 41))
    auto = jax.vmap(jax.grad(jax.grad(lambda t: mrf.pseudo_huber(t, eps))))(x)
    np.testing.assert_allclose(np.asarray(mrf.pseudo_huber_second_derivative(x, eps)), np.asarray(auto), rtol=1e-10)
    np.testing.assert_allclose(float(mrf.pseudo_huber_second_derivative(jnp.asarray(0.0), eps)), 1.0 / eps)


def test_tv_limit_of_the_energy_is_lam_times_the_total_variation_for_large_jumps():
    """|Delta| >> delta: R - anchor -> lam sum b |Delta| (minus lam delta per active pair)."""
    n, lam, delta = 6, 2.5, 0.01
    prior = mrf.PseudoHuberTV(n, lam=lam, delta=delta, sigma_0=1e9)
    rng = np.random.default_rng(2)
    theta = 1000.0 * rng.standard_normal((n, n))  # every pair difference is >> delta
    i, j, b = mrf.neighbor_pairs(n)
    t = theta.ravel()
    tv = lam * np.sum(b * np.abs(t[i] - t[j]))
    np.testing.assert_allclose(float(prior.energy(theta)), tv - lam * delta * b.sum(), rtol=1e-6)
    np.testing.assert_allclose(float(prior.energy(theta)) / tv, 1.0, atol=1e-4)
    # and quadratic for small differences: lam b D^2 / (2 delta)
    small = 1e-5 * rng.standard_normal((n, n))
    np.testing.assert_allclose(float(prior.energy(small)), lam / (2 * delta) * np.sum(b * (small.ravel()[i] - small.ravel()[j]) ** 2),
                               rtol=1e-4)


def test_tv_value_gradient_hvp_and_hessian_are_finite_and_exact_at_zero_difference():
    n = 6
    prior = mrf.PseudoHuberTV(n, lam=2.0, delta=0.01, sigma_0=1.0, mu=0.3)
    const = jnp.full((n, n), 0.3)
    for theta in (const, const + 1e-9 * jnp.arange(n * n).reshape(n, n)):
        assert np.isfinite(float(prior.energy(theta)))
        assert np.all(np.isfinite(np.asarray(prior.grad(theta))))
        v = jnp.asarray(np.random.default_rng(0).standard_normal((n, n)))
        assert np.all(np.isfinite(np.asarray(prior.hvp(theta, v))))
        assert np.all(np.isfinite(prior.hessian(theta)))
    np.testing.assert_allclose(np.asarray(prior.grad(const)), 0.0, atol=1e-12)
    assert float(prior.energy(const)) == 0.0
    # at D = 0 the pair Hessian is lam b / delta: the weighted graph Laplacian over delta, plus the anchor
    i, j, b = mrf.neighbor_pairs(n)
    lap = np.zeros((n * n, n * n))
    for a, c, w in zip(i, j, b):
        lap[a, a] += w
        lap[c, c] += w
        lap[a, c] -= w
        lap[c, a] -= w
    np.testing.assert_allclose(prior.hessian(const), 2.0 * lap / 0.01 + np.eye(n * n), rtol=1e-12)


def test_tv_closed_form_gradient_hessian_and_hvp_match_jax_autodiff():
    n = 5
    prior = mrf.PseudoHuberTV(n, lam=1.7, delta=0.02, sigma_0=0.8, mu=-0.1)
    rng = np.random.default_rng(12)
    for scale in (0.0, 0.01, 0.1, 1.0):  # differences below, near and above delta
        theta = jnp.asarray(scale * rng.standard_normal((n, n)))
        flat = lambda x: prior._energy(x.reshape(n, n))
        np.testing.assert_allclose(np.asarray(prior.grad(theta)).ravel(), np.asarray(jax.grad(flat)(theta.ravel())),
                                   rtol=1e-12, atol=1e-13)
        auto = np.asarray(jax.hessian(flat)(theta.ravel()))
        np.testing.assert_allclose(prior.hessian(theta), auto, rtol=1e-10, atol=1e-10 * np.abs(auto).max())
        v = jnp.asarray(rng.standard_normal((n, n)))
        np.testing.assert_allclose(np.asarray(prior.hvp(theta, v)).ravel(), auto @ np.asarray(v).ravel(), rtol=1e-10,
                                   atol=1e-10 * np.abs(auto).max())
    # central finite differences of the gradient (independent of autodiff)
    theta = jnp.asarray(0.05 * rng.standard_normal((n, n)))
    v = rng.standard_normal((n, n))
    h = 1e-6
    fd = (np.asarray(prior.grad(theta + h * v)) - np.asarray(prior.grad(theta - h * v))) / (2 * h)
    np.testing.assert_allclose(np.asarray(prior.hvp(theta, jnp.asarray(v))), fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("lam,delta", [(1.0, 0.01), (5.0, 0.002), (0.2, 0.1)])
def test_tv_energy_is_convex_and_its_hessian_is_positive_definite(lam, delta):
    n = 6
    prior = mrf.PseudoHuberTV(n, lam=lam, delta=delta, sigma_0=1.0)
    rng = np.random.default_rng(13)
    for _ in range(30):  # midpoint convexity R(tx + (1-t)y) <= t R(x) + (1-t) R(y), at several scales
        x, y = rng.standard_normal((2, n, n)) * rng.uniform(0.005, 2.0)
        t = rng.uniform()
        lhs = float(prior.energy(t * x + (1 - t) * y))
        rhs = t * float(prior.energy(x)) + (1 - t) * float(prior.energy(y))
        assert lhs <= rhs + 1e-12 * abs(rhs)
    for scale in (0.0, 0.003, 0.05, 1.0):
        eig = np.linalg.eigvalsh(prior.hessian(rng.standard_normal((n, n)) * scale))
        assert eig.min() >= 1.0 / prior.sigma_0**2 - 1e-9  # Hessian >= I / sigma_0^2: SPD Laplace mass
        assert eig.max() <= 1.0 + lam / delta * 2 * (4 + 4 / np.sqrt(2)) + 1e-6  # lam b / delta per pair, Gershgorin
    d = jnp.asarray(np.linspace(-3.0, 3.0, 20001))
    assert float(jax.vmap(jax.grad(jax.grad(lambda x: mrf.pseudo_huber(x, delta))))(d).min()) > 0.0


def _ramp_field(n, height, k, row0=4):
    """Vertical edge: theta = 0 left of column ``row0``, ``height`` right of it, with the rise spread over k equal steps."""
    theta = np.zeros((n, n))
    steps = np.arange(1, k + 1) * height / k
    theta[:, row0:row0 + k] = steps[None, :]
    theta[:, row0 + k:] = height
    return theta


@pytest.mark.parametrize("delta", [0.01, 0.003, 0.001])
def test_tv_step_over_ramp_cost_ratio_is_1_plus_k_minus_1_delta_over_jump(delta):
    """The DESIGN 12.1 claim that the ratio is 'within a few percent of 1' holds for k = 2 at delta = 0.01 (1.4%) but NOT for
    the 8-pixel ramp (10.6%): a ramp of k pixels saves about (k - 1) delta per pair line, 7 * 0.01 / log 2 = 10%. It is 3.1%
    at delta = 0.003 and 1.0% at delta = 0.001. Checked on the pair potential and on the 2-D energy of a straight edge."""
    jump = np.log(2.0)
    ph = lambda d: float(mrf.pseudo_huber(jnp.asarray(d), delta))
    for k in (2, 4, 8):
        ratio = ph(jump) / (k * ph(jump / k))
        first_order = 1.0 + (k - 1) * delta / jump
        assert ratio > 1.0 and abs(ratio - first_order) < 3 * (k * delta / jump) ** 2  # first order plus O(delta^2) terms
    r8 = ph(jump) / (8 * ph(jump / 8))
    assert r8 == pytest.approx({0.01: 1.106, 0.003: 1.031, 0.001: 1.010}[delta], abs=0.0015)
    # 2-D: the same ratio for a straight vertical edge on the grid (horizontal and both diagonal pair families cross it)
    n = 16
    prior = mrf.PseudoHuberTV(n, lam=1.0, delta=delta, sigma_0=1e9)
    sharp = float(prior.energy(_ramp_field(n, jump, 1)))
    for k in (2, 8):
        np.testing.assert_allclose(sharp / float(prior.energy(_ramp_field(n, jump, k))), ph(jump) / (k * ph(jump / k)),
                                   rtol=1e-6)


def test_tv_is_nearly_indifferent_where_the_qggmrf_prefers_a_ramp_by_a_factor_two_or_more():
    jump = np.log(2.0)
    n = 16
    tv = mrf.PseudoHuberTV(n, lam=1.0, delta=0.01, sigma_0=1e9)
    qg = mrf.QGGMRF(n, q=1.2, T=1.0, sigma_g=0.05, sigma_0=1e9)
    ratio = lambda p, k: float(p.energy(_ramp_field(n, jump, 1))) / float(p.energy(_ramp_field(n, jump, k)))
    assert ratio(tv, 8) < 1.12 and ratio(qg, 8) > 2.0
    assert ratio(tv, 2) < 1.02


def test_tv_rejects_bad_parameters():
    for kw in (dict(lam=0.0), dict(lam=-1.0), dict(delta=0.0), dict(sigma_0=0.0)):
        with pytest.raises(ValueError):
            mrf.PseudoHuberTV(4, **kw)


def test_tv_prior_sampler_runs_and_has_the_pairwise_hessian_scale():
    """The prior-only HMC of ``sample_prior`` works for the TV prior (it is used by run_prior_comparison for the prior std)."""
    prior = mrf.PseudoHuberTV(4, lam=3.0, delta=0.05, sigma_0=1.0)
    out = mrf.sample_prior(prior, jax.random.key(0), n_chains=2, n_warmup=100, n_samples=300)
    assert np.all(np.isfinite(out["theta_std"])) and out["accept"] > 0.3
    # the anchor alone gives sd <= sigma_0 = 1; the TV coupling makes it smaller
    assert 0.05 < out["theta_std"].mean() < 1.0
