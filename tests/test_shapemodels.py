"""Tests for bayes_scatter.shapemodels: the unconstrained parametrizations, the priors with their Jacobians, and prior-only HMC."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter.geometry import Grid

MODELS = ("F2", "F2u", "Fe")


def random_u(model, rng, scale=2.0):
    return rng.uniform(-scale, scale, model.dim)


@pytest.mark.parametrize("name", MODELS)
def test_u_and_p_are_inverse_bijections_and_p_lies_in_the_prior_support(name):
    model = sm.make_model(name)
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = random_u(model, rng, 3.0)
        p = np.asarray(model.p_from_u(jnp.asarray(u)))
        np.testing.assert_allclose(np.asarray(model.u_from_p(jnp.asarray(p))), u, atol=1e-9)
        assert np.isfinite(float(model.log_prior_p(jnp.asarray(p))))  # inside the support
    draws = model.sample_prior_p(rng, 50)
    assert np.all(np.isfinite(np.asarray(jax.vmap(model.log_prior_p)(jnp.asarray(draws)))))
    np.testing.assert_allclose(np.asarray(jax.vmap(model.p_from_u)(jax.vmap(model.u_from_p)(jnp.asarray(draws)))), draws, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("name", MODELS)
def test_the_u_prior_is_the_p_prior_times_the_jacobian(name):
    """log pi_u(u) = log pi_p(p(u)) + log |det dp/du|: the closed form of ``log_prior_u`` equals the sum of its two parts, and the Jacobian is the autodiff one."""
    model = sm.make_model(name)
    rng = np.random.default_rng(1)
    for _ in range(10):
        u = jnp.asarray(random_u(model, rng, 3.0))
        parts = float(model.log_prior_p(model.p_from_u(u)) + model.log_jac_u(u))
        assert float(model.log_prior_u(u)) == pytest.approx(parts, abs=1e-10)
        jac = np.asarray(jax.jacfwd(model.p_from_u)(u))
        assert float(model.log_jac_u(u)) == pytest.approx(np.log(abs(np.linalg.det(jac))), abs=1e-9)


@pytest.mark.parametrize("name", MODELS)
def test_the_u_prior_is_separable_and_normalized(name):
    """log pi_u is a sum of one-dimensional terms (checked by mixed differences), so its integral is the product of the 1-D integrals: log Z = 0."""
    model = sm.make_model(name)
    lp = lambda u: float(model.log_prior_u(jnp.asarray(u)))
    base = np.linspace(-0.4, 0.5, model.dim)
    d = model.dim
    for i in range(d):
        for j in range(i + 1, d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i], ej[j] = 0.7, -0.9
            assert abs(lp(base + ei + ej) - lp(base + ei) - lp(base + ej) + lp(base)) < 1e-12
    xs = np.linspace(-60.0, 60.0, 24001)
    lp_many = jax.jit(jax.vmap(model.log_prior_u))
    total = lp(base)
    for k in range(d):
        pts = np.tile(base, (xs.size, 1))
        pts[:, k] = xs
        h = np.exp(np.asarray(lp_many(jnp.asarray(pts))) - lp(base))
        total += np.log(np.trapezoid(h, xs))
    assert total == pytest.approx(0.0, abs=2e-6)


class PriorOnly:
    """Problem with phi = 0: the posterior is the prior of ``model`` in u. ``jacobian=False`` drops the Jacobian (a wrong prior: the negative control)."""

    def __init__(self, model, jacobian=True):
        self.model, self.dim, self.lik_const, self.names = model, model.dim, 0.0, model.param_names
        self.data = jnp.zeros(())
        self.log_prior = (jax.jit(model.log_prior_u) if jacobian else jax.jit(lambda u: model.log_prior_p(model.p_from_u(u))))
        self.phi = jax.jit(lambda u, aux, data: (jnp.zeros(()), aux))
        self.phi_and_grad = jax.jit(lambda u, aux, data: (jnp.zeros(()), jnp.zeros_like(u), aux))
        self._U_and_grad = self._prior_hessian = None
        self.physical_many = jax.jit(jax.vmap(model.p_from_u))

    def initial_aux(self):
        return jnp.zeros(())

    def fisher(self, u):
        return np.zeros((self.dim, self.dim))


def prior_cdfs(name):
    w = sm.AX_HI - sm.AX_LO
    unif = lambda lo, hi: (lambda x: np.clip((x - lo) / (hi - lo), 0, 1))
    eps = lambda x: stats.norm.cdf(np.log(np.maximum(x - 1.0, 1e-300)) / sm.EPS_SIGMA)
    if name == "F2":
        return [unif(0, sm.G_MAX), unif(sm.R_LO, sm.R_HI), eps, unif(-sm.C_MAX, sm.C_MAX), unif(-sm.C_MAX, sm.C_MAX)]
    if name == "F2u":
        return [unif(0, sm.G_MAX), unif(sm.R_LO, sm.R_HI), unif(sm.R_LO, sm.R_HI), eps, eps, unif(-sm.C_MAX, sm.C_MAX), unif(-sm.C_MAX, sm.C_MAX)]
    return [unif(-sm.C_MAX, sm.C_MAX), unif(-sm.C_MAX, sm.C_MAX), lambda x: np.clip((x - sm.AX_LO) / w, 0, 1) ** 2,  # P(a <= x) = ((x - lo) / w)^2 (maximum of two uniforms)
            lambda x: 1.0 - np.clip((sm.AX_HI - x) / w, 0, 1) ** 2,  # P(b <= x) = 1 - ((hi - x) / w)^2 (minimum of two uniforms)
            unif(-0.5 * np.pi, 0.5 * np.pi), eps]


def prior_hmc_pvalues(problem, name, key, n_samples=1500, thin=6):
    amap = sp.prior_map(problem)
    starts = np.asarray(jax.vmap(problem.model.u_from_p)(jnp.asarray(problem.model.sample_prior_p(np.random.default_rng(7), 4))))
    zeta0 = np.stack([np.linalg.solve(amap.chol_np, s) for s in starts])
    batch = sp.run_hmc(problem, amap, zeta0, key, beta=1.0, n_warmup=200, n_samples=n_samples, n_leapfrog=8)
    p = sp.physical_draws(problem, batch.u)[:, ::thin].reshape(-1, problem.dim)
    cdfs = prior_cdfs(name)
    return [stats.kstest(p[:, k], cdfs[k]).pvalue for k in range(problem.dim)], batch


@pytest.mark.parametrize("name", MODELS)
def test_prior_only_hmc_in_u_reproduces_the_prior_marginals_in_p(name):
    """HMC on log pi_u (Jacobians included) gives draws whose p-space marginals are the specified priors (Kolmogorov-Smirnov against the exact CDFs of the
    uniform / log-normal / ordered-pair priors); this is what makes the u-space prior and the evidence right."""
    problem = PriorOnly(sm.make_model(name))
    pvals, batch = prior_hmc_pvalues(problem, name, jax.random.key(11))
    assert batch.acceptance > 0.5
    assert min(pvals) > 1e-3, dict(zip(problem.names, pvals))


def test_negative_control_without_the_jacobian_hmc_samples_the_wrong_marginals():
    """Dropping the Jacobian (a uniform density in p evaluated at p(u), flat in u) makes the draws pile up at the bounds: the same test rejects it decisively."""
    problem = PriorOnly(sm.make_model("F2"), jacobian=False)
    pvals, _ = prior_hmc_pvalues(problem, "F2", jax.random.key(12), n_samples=600)
    assert min(pvals) < 1e-10 or not np.isfinite(min(pvals))


def test_scenes_and_ordering_constraints():
    f2u = sm.F2u()
    p = np.array([0.1, 0.18, 0.22, 2.0, 2.4, 0.03, -0.02])
    ell, eps = f2u.scene(jnp.asarray(p))
    ell, eps = np.asarray(ell), np.asarray(eps)
    assert eps.tolist() == [2.0, 2.4]
    # the facing edges are g apart whatever the radii, and disk 1 is the left one
    assert (ell[1, 0] - ell[1, 2]) - (ell[0, 0] + ell[0, 2]) == pytest.approx(0.1, abs=1e-14)
    assert ell[0, 0] < ell[1, 0] and 0.5 * (ell[0, 0] + ell[1, 0]) == pytest.approx(0.03)
    # F2 is F2u with tied parameters
    f2 = sm.F2()
    p2 = np.array([0.1, 0.2, 2.0, 0.03, -0.02])
    np.testing.assert_allclose(np.asarray(f2.scene(jnp.asarray(p2))[0]), np.asarray(f2u.scene(jnp.asarray(f2.to_f2u(p2)))[0]), atol=1e-15)
    # the Fe ordering: a >= b for every u (also at extreme u), and the symmetry images
    fe = sm.Fe()
    rng = np.random.default_rng(3)
    for _ in range(50):
        pp = np.asarray(fe.p_from_u(jnp.asarray(rng.uniform(-8, 8, 6))))
        assert pp[2] >= pp[3] and sm.AX_LO <= pp[3] and pp[2] <= sm.AX_HI and -np.pi / 2 <= pp[4] <= np.pi / 2
    start = fe.u_from_p(jnp.asarray(f2.to_fe(p2)))
    assert np.all(np.isfinite(np.asarray(start)))
    labels = [k for k, _ in f2u.symmetry_images(p)]
    assert len(labels) == 2 and f2u.symmetry_images(p)[1][1][5] == -0.03 and f2u.symmetry_images(p)[0][1][5] == 0.03
    assert sm.F2().gap(p2) == 0.1 and fe.gap(np.zeros(6)) is None
    with pytest.raises(ValueError):
        sm.make_model("F3")


def test_models_render_through_the_exact_renderer_with_matching_dimensions():
    grid = Grid(20, 1.5)
    rend = el.EllipseRenderer(grid)
    rng = np.random.default_rng(5)
    for name in MODELS:
        model = sm.make_model(name)
        assert model.dim == len(model.param_names)
        chi = np.asarray(model.chi(rend, jnp.asarray(random_u(model, rng, 1.0))))
        assert chi.shape == (20, 20) and chi.min() >= 0.0
        d = model.describe()
        assert d["dim"] == model.dim and d["name"] == name
