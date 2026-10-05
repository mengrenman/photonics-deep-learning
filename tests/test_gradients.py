"""Gradient and warm-start tests for the differentiable solve (docs/DESIGN.md 2.4, 8.3)."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring

N = 16


def _setup(n=N):
    # coarse DOI of side 2 with the book's ring radii, fewer antennas for speed
    return Setup(Grid(n, 2.0), Antennas(ring(8, 4.0), ring(12, 4.5)))


def _phantom(n=N):
    chi = np.zeros((n, n))
    chi[n // 4: 3 * n // 4, n // 3: 2 * n // 3] = 1.5  # contrast 1.5 (eps_r = 2.5)
    return chi


def _random_unit(rng, shape):
    v = rng.standard_normal(shape)
    return v / np.linalg.norm(v)


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b))


def _misfit_fn(fwd, d_obs):
    def misfit(chi):
        d, _ = fwd(chi)
        return jnp.sum(jnp.abs(d - d_obs) ** 2)
    return misfit


@pytest.mark.parametrize("kwargs", [dict(), dict(batch="vmap"), dict(solver="bicgstab", maxiter=300)],
                         ids=["gmres-stack", "gmres-vmap", "bicgstab"])
def test_grad_matches_central_finite_differences(kwargs):
    """jax.grad of sum |F(chi) - d|^2 against central differences along 3 random unit directions.
    Measured relative errors are 1e-10 to 1e-7 (limited by finite-difference roundoff), against the
    required 1e-5; an adjoint solve with the wrong operator (conjugate transpose) or a wrong
    tolerance would give O(1e-2..1) errors."""
    fwd = Forward(_setup(), tol=1e-12, **kwargs)
    d_obs = fwd(_phantom())[0]
    misfit = _misfit_fn(fwd, d_obs)
    rng = np.random.default_rng(0)
    chi = jnp.asarray(0.4 + 0.25 * rng.standard_normal((N, N)))  # heterogeneous, with eps_r in ~(0.3, 2)
    g = np.asarray(jax.grad(misfit)(chi))
    assert g.shape == (N, N) and np.all(np.isfinite(g))
    eps = 1e-5
    for _ in range(3):
        v = _random_unit(rng, (N, N))
        fd = float((misfit(chi + eps * v) - misfit(chi - eps * v)) / (2 * eps))
        an = float(np.sum(g * v))
        assert abs(fd) > 1e-6 * np.linalg.norm(g)  # the direction is informative
        assert _rel(fd, an) < 1e-5, (fd, an)


def test_grad_through_log_eps_link_and_value_and_grad_with_aux():
    """Chain rule through chi = exp(theta) - 1 (the default link), using value_and_grad with aux fields."""
    fwd = Forward(_setup(), tol=1e-12)
    d_obs = fwd(_phantom())[0]
    rng = np.random.default_rng(5)
    theta = jnp.asarray(0.3 * rng.standard_normal((N, N)))

    def phi(th):
        d, u = fwd(jnp.expm1(th))
        return jnp.sum(jnp.abs(d - d_obs) ** 2), u

    (val, u), g = jax.value_and_grad(phi, has_aux=True)(theta)
    assert u.shape == (8, N, N) and np.isfinite(float(val))
    eps = 1e-5
    for _ in range(3):
        v = _random_unit(rng, (N, N))
        fd = float((phi(theta + eps * v)[0] - phi(theta - eps * v)[0]) / (2 * eps))
        assert _rel(fd, float(jnp.sum(g * v))) < 1e-5


def test_grad_with_data_through_jit_and_misfit_on_born_and_forward_differ():
    """jit(grad) composes; the Born gradient (linear model) differs from the full gradient."""
    fwd = Forward(_setup(), tol=1e-12)
    d_obs = fwd(_phantom())[0]
    chi = jnp.asarray(_phantom() * 0.5)
    g_full = jax.jit(jax.grad(_misfit_fn(fwd, d_obs)))(chi)
    g_born = jax.grad(lambda c: jnp.sum(jnp.abs(fwd.born(c) - d_obs) ** 2))(chi)
    # Born gradient is exact for the linear model: A^H (A chi - d)
    A = np.asarray(fwd.born_matrix())
    expected = 2 * np.real(A.conj().T @ (A @ np.asarray(chi).reshape(-1) - np.asarray(d_obs).reshape(-1)))
    np.testing.assert_allclose(np.asarray(g_born).reshape(-1), expected, rtol=1e-9, atol=1e-12)
    assert np.linalg.norm(np.asarray(g_full) - np.asarray(g_born)) > 0.05 * np.linalg.norm(np.asarray(g_born))


def test_warm_start_gives_same_fields_with_fewer_operator_applications():
    """u0 is a pure accelerator: same answer to the solver tolerance, fewer operator applications.
    Operator applications (G products, including the residual evaluations) are counted by
    Forward.matvec_counts. Measured at tol = 1e-8 on this problem, per transmitter: cold 22 to 23; warm
    from the solution at chi perturbed by 1e-3: 18 to 19 (5 instead of 8 orders of residual reduction);
    perturbed by 1e-6: 12; warm from the converged solution: 2."""
    fwd = Forward(_setup(), tol=1e-8)
    rng = np.random.default_rng(7)
    chi = jnp.asarray(_phantom() + 0.05 * rng.standard_normal((N, N)))
    noise = jnp.asarray(rng.standard_normal((N, N)))
    u_prev = fwd.solve_fields(chi)
    chi_new = chi + 1e-3 * noise
    u_cold = fwd.solve_fields(chi_new)
    u_warm = fwd.solve_fields(chi_new, u0=u_prev)
    ref = np.asarray(Forward(_setup(), tol=1e-13).solve_fields(chi_new))
    scale = np.max(np.abs(ref))
    for u in (u_cold, u_warm):  # both within the tolerance of the converged solution
        assert np.max(np.abs(np.asarray(u) - ref)) < 1e-7 * scale
    assert np.max(np.abs(np.asarray(u_warm) - np.asarray(u_cold))) < 2e-7 * scale
    d_warm, d_cold = np.asarray(fwd(chi_new, u0=u_prev)[0]), np.asarray(fwd(chi_new)[0])
    assert np.linalg.norm(d_warm - d_cold) < 1e-7 * np.linalg.norm(d_cold)

    cold = fwd.matvec_counts(chi_new)
    near = fwd.matvec_counts(chi_new, u0=u_prev)
    chi_close = chi + 1e-6 * noise
    close = fwd.matvec_counts(chi_close, u0=u_prev)
    exact = fwd.matvec_counts(chi_new, u0=u_cold)
    print("\nmatvec counts per transmitter  cold:", cold, " warm(1e-3):", near, " warm(1e-6):", close,
          " warm(converged):", exact)
    assert np.all(near < cold) and near.sum() < 0.9 * cold.sum()
    assert close.sum() < 0.6 * cold.sum()
    assert exact.max() <= 3  # a converged start costs only the initial residual check


def test_warm_start_does_not_change_the_gradient():
    """The gradient is the same with or without a warm start, whether the start is the converged solution
    at the evaluation point or a stale one from a different chi (tol = 1e-10, relative misfit ~ 1e-3)."""
    n = 12
    fwd = Forward(_setup(n), tol=1e-10)
    ref = Forward(_setup(n), tol=1e-13)
    rng = np.random.default_rng(9)
    chi = jnp.asarray(_phantom(n) + 0.05 * rng.standard_normal((n, n)))
    d_obs = ref(chi + 1e-3 * jnp.asarray(rng.standard_normal((n, n))))[0]
    g_ref = np.asarray(jax.grad(_misfit_fn(ref, d_obs))(chi))
    nrm = np.linalg.norm(g_ref)

    def grad_with(u0):
        return np.asarray(jax.grad(lambda c: jnp.sum(jnp.abs(fwd(c, u0)[0] - d_obs) ** 2))(chi))

    for u0 in (None, ref.solve_fields(chi), ref.solve_fields(chi + 0.1)):
        assert np.linalg.norm(grad_with(u0) - g_ref) / nrm < 1e-5


def test_adjoint_is_more_accurate_than_reusing_the_forward_solve():
    """Why Forward wraps lax.custom_linear_solve itself. jax.scipy.sparse.linalg.gmres uses one solve for
    both directions: the adjoint solve inherits the forward warm start x0 and the absolute tolerance
    tol * ||b_forward||. The comparator below reproduces exactly that, with our own GMRES (so the forward
    solves are identical) and ``transpose_solve = solve``. At tol = 1e-8 and a relative misfit of 1e-2 the
    gradient error against a tol = 1e-13 reference is, measured: comparator 2.8e-5 (cold) and
    2.6e-5 (warm); ours 9.8e-7 (cold) and 3.8e-9 (warm)."""
    from jax import lax

    from bayes_scatter.forward import _apply_G
    from bayes_scatter.krylov import gmres_stacked

    n = 16
    setup = Setup(Grid(n, 2.0), Antennas(ring(4, 4.0), ring(6, 4.5)))
    tol = 1e-8
    fwd, ref = Forward(setup, tol=tol), Forward(setup, tol=1e-13)
    rng = np.random.default_rng(1)
    chi = jnp.asarray(0.5 + 0.3 * rng.standard_normal((n, n)))
    d_obs = ref(chi + 0.01 * jnp.asarray(rng.standard_normal((n, n))))[0]

    def black_box_loss(c, u0):
        def one(b, x0):
            mv = lambda x: x - _apply_G(fwd.g_hat, c * x)
            atol = tol * jnp.linalg.norm(b)  # fixed by the forward right-hand side
            solve = lambda mv_, rhs: gmres_stacked(lambda v: mv_(v[0])[None], rhs[None], x0[None], tol=0.0,
                                                   atol=atol, restart=100, maxiter=30)[0]
            return lax.custom_linear_solve(mv, b, solve=solve, transpose_solve=solve)
        u = jax.vmap(one)(fwd.u_inc, u0)
        return jnp.sum(jnp.abs(fwd.scattered(c, u) - d_obs) ** 2)

    def ours_loss(c, u0):
        return jnp.sum(jnp.abs(fwd(c, u0)[0] - d_obs) ** 2)

    g_ref = np.asarray(jax.grad(_misfit_fn(ref, d_obs))(chi))
    u_exact = ref.solve_fields(chi)
    out = {}
    for name, u0 in (("cold", jnp.zeros_like(u_exact)), ("warm", u_exact)):
        e_bb = np.linalg.norm(np.asarray(jax.grad(black_box_loss)(chi, u0)) - g_ref) / np.linalg.norm(g_ref)
        e_ours = np.linalg.norm(np.asarray(jax.grad(ours_loss)(chi, u0)) - g_ref) / np.linalg.norm(g_ref)
        out[name] = (e_bb, e_ours)
        assert e_ours < e_bb / 3, (name, e_ours, e_bb)
        assert e_ours < 1e-6
    print("\ngradient rel. error  comparator/ours  cold: %.1e / %.1e   warm: %.1e / %.1e"
          % (*out["cold"], *out["warm"]))


def test_gradient_does_not_flow_into_u0():
    fwd = Forward(_setup(), tol=1e-10)
    chi = jnp.asarray(_phantom())
    u0 = fwd.solve_fields(chi)
    g = jax.grad(lambda u: jnp.sum(jnp.abs(fwd(chi, u)[0]) ** 2), argnums=0)(u0)
    assert np.max(np.abs(np.asarray(g))) == 0.0
