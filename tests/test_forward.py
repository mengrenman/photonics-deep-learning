"""Tests for forward.py and mie.py (docs/DESIGN.md 2.3-2.4, 8.2)."""

import numpy as np
import pytest
from scipy.special import hankel1, jv, jvp, h1vp

import jax
import jax.numpy as jnp

from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.greens import dense_green_matrix
from bayes_scatter.krylov import gmres_stacked
from bayes_scatter.mie import mie_coefficients, mie_scattered
from bayes_scatter.phantoms import cylinder

K0 = 2 * np.pi
TX = ring(6, 3.0)
RX = ring(8, 3.5, offset=np.pi / 8)


def _rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b)


def _mie_error(n, eps_r=2.0, radius=0.6, tol=1e-10):
    grid = Grid(n, 2.0)
    fwd = Forward(Setup(grid, Antennas(TX, RX)), tol=tol)
    d, u = fwd(cylinder(grid, radius, eps_r))
    return _rel(d, mie_scattered(TX, RX, radius, eps_r))


# ----------------------------------------------------------------------------- Mie series


def test_mie_coefficients_satisfy_boundary_conditions():
    """Continuity of u and du/drho at rho = R, with independent Bessel derivatives (scipy jvp, h1vp)."""
    R, eps = 0.6, 2.0 + 0.3j
    k1 = K0 * np.sqrt(eps)
    n = np.arange(12)
    a = mie_coefficients(11, R, eps, K0)
    A = 1.0  # amplitude of the regular (incident) partial wave J_n(k0 rho)
    b = a * A  # outgoing amplitude
    c = (A * jv(n, K0 * R) + b * hankel1(n, K0 * R)) / jv(n, k1 * R)  # interior, from continuity of u
    lhs = K0 * (A * jvp(n, K0 * R) + b * h1vp(n, K0 * R))
    rhs = k1 * c * jvp(n, k1 * R)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_mie_no_scattering_for_vacuum_and_reciprocity():
    assert np.max(np.abs(mie_scattered(TX, RX, 0.6, 1.0))) < 1e-14
    u_ab = mie_scattered(TX, RX, 0.6, 3.0)
    u_ba = mie_scattered(RX, TX, 0.6, 3.0)
    np.testing.assert_allclose(u_ab, u_ba.T, rtol=1e-13, atol=1e-16)
    assert np.max(np.abs(u_ab)) > 1e-3


def test_mie_truncation_is_converged_and_center_shift_works():
    u, nmax = mie_scattered(TX, RX, 1.2, 4.0, return_nmax=True)
    # adding many more modes changes nothing
    from bayes_scatter.mie import _series
    more = 0.25j * _series(TX, RX, 1.2, 4.0, K0, nmax + 15).sum(axis=0)
    np.testing.assert_allclose(u, more, rtol=1e-12, atol=1e-15)
    assert nmax >= int(K0 * 2 * 1.2)  # at least k1 R modes
    shifted = mie_scattered(TX + [0.3, -0.2], RX + [0.3, -0.2], 1.2, 4.0, center=(0.3, -0.2))
    np.testing.assert_allclose(shifted, u, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        mie_scattered([[0.5, 0.0]], RX, 1.2, 4.0)  # transmitter inside the cylinder


def test_mie_matches_born_at_low_contrast():
    """Independent of the Lippmann-Schwinger solver: Mie for eps = 1 + 1e-3 against the Born data of
    the discretized cylinder (R and u_inc only). Agreement is limited by the pixelization (~1e-3)."""
    grid = Grid(96, 2.0)
    fwd = Forward(Setup(grid, Antennas(TX, RX)))
    delta = 1e-3
    d_born = fwd.born(cylinder(grid, 0.6, 1.0 + delta))
    u_mie = mie_scattered(TX, RX, 0.6, 1.0 + delta)
    assert _rel(d_born, u_mie) < 6e-3
    # first-order scattering is linear in the contrast: doubling it doubles the field to O(delta)
    assert abs(_rel(mie_scattered(TX, RX, 0.6, 1.0 + 2 * delta), 2 * u_mie)) < 2e-3


# ----------------------------------------------------------------------------- forward vs Mie


def test_forward_vs_mie_converges_under_refinement():
    """Centered cylinder, radius 0.6, eps_r = 2, DOI side 2, 6 transmitters on r = 3, 8 receivers on
    r = 3.5. Measured relative errors (6 tx x 8 rx, tol 1e-10): n = 24: 3.07e-2, n = 48: 7.99e-3,
    n = 96: 2.45e-3, i.e. error ~ h^1.8 (area-averaged staircase boundary plus Richmond cell error).

    The n = 96 threshold 4e-3 is 1.6x above the measured value. A wrong constant in the kernel (a missing
    '-1' in the self term, a factor 2, a sign) changes the field by tens of percent and fails this."""
    errs = [_mie_error(n) for n in (24, 48, 96)]
    print("\nMie relative error n=24,48,96:", errs)
    assert errs[0] > errs[1] > errs[2]
    assert errs[0] / errs[1] > 2.0 and errs[1] / errs[2] > 2.0  # better than first order
    assert errs[2] < 4e-3


def test_forward_vs_mie_high_contrast_and_other_radius():
    """eps_r = 4, radius 0.8 (k1 R = 10) is harder: measured 4.79e-2 (n = 48) and 1.38e-2 (n = 96).
    The error is the integral-equation discretization, not the phantom: with sub-sampling ss = 64 in the
    coverage the errors at n = 48, 96, 192 are 5.08e-2, 1.32e-2, 3.34e-3, i.e. clean second order."""
    errs = [_mie_error(n, eps_r=4.0, radius=0.8) for n in (48, 96)]
    print("\nMie relative error eps_r=4, R=0.8, n=48,96:", errs)
    assert errs[1] < errs[0] / 2.5 and errs[1] < 2e-2


def test_forward_vs_mie_lossy_cylinder():
    """Complex permittivity: the forward model accepts complex chi, Mie accepts complex eps_r."""
    eps = 2.0 + 0.5j
    errs = [_mie_error(n, eps_r=eps) for n in (24, 48, 96)]
    print("\nMie relative error eps_r=2+0.5i, n=24,48,96:", errs)
    assert errs[0] > errs[1] > errs[2] and errs[2] < 6e-3


# ----------------------------------------------------------------------------- basic solve properties


def _small_setup(n=24, side=2.0):
    return Setup(Grid(n, side), Antennas(TX, RX))


def test_zero_contrast_gives_incident_field_and_no_data():
    fwd = Forward(_small_setup(16))
    d, u = fwd(jnp.zeros((16, 16)))
    np.testing.assert_allclose(np.asarray(u), np.asarray(fwd.u_inc), rtol=1e-12, atol=1e-15)
    assert np.max(np.abs(np.asarray(d))) == 0.0
    assert d.shape == (6, 8) and u.shape == (6, 16, 16) and d.dtype == jnp.complex128


def test_fields_satisfy_discrete_lippmann_schwinger_equation():
    grid = Grid(24, 2.0)
    fwd = Forward(Setup(grid, Antennas(TX, RX)), tol=1e-10)
    chi = jnp.asarray(cylinder(grid, 0.7, 5.0) + 0.3 * cylinder(grid, 0.3, 2.0, center=(0.4, 0.2)))
    u = fwd.solve_fields(chi)
    resid = np.asarray(fwd.residual(chi, u))
    assert resid.shape == (6,) and resid.max() < 1.05e-10  # tol = 1e-10, relative, per transmitter
    direct = np.asarray(u) - np.asarray(fwd.apply_G(chi * u)) - np.asarray(fwd.u_inc)
    ratio = np.linalg.norm(direct.reshape(6, -1), axis=1) / np.linalg.norm(np.asarray(fwd.u_inc).reshape(6, -1), axis=1)
    np.testing.assert_allclose(ratio, resid, rtol=1e-6)  # Forward.residual is that quantity
    # data from scattered(chi, u) equal the first output of __call__
    d1 = fwd.scattered(chi, u)
    d2, _ = fwd(chi)
    np.testing.assert_allclose(np.asarray(d1), np.asarray(d2), rtol=1e-8, atol=1e-12)


def test_solution_matches_dense_direct_solve():
    """Independent oracle: build I - G diag(chi) densely (n = 12) and solve with LAPACK."""
    n = 12
    grid = Grid(n, 2.0)
    setup = Setup(grid, Antennas(TX, RX))
    rng = np.random.default_rng(3)
    chi = 0.5 + 2.0 * rng.uniform(size=(n, n))  # eps_r between 1.5 and 3.5, heterogeneous
    fwd = Forward(setup, tol=1e-12)
    A = np.eye(n * n) - dense_green_matrix(setup) * chi.reshape(1, -1)
    u_inc = np.asarray(fwd.u_inc).reshape(6, -1)
    u_ref = np.linalg.solve(A, u_inc.T).T.reshape(6, n, n)
    u = np.asarray(fwd.solve_fields(chi))
    assert np.max(np.abs(u - u_ref)) < 1e-9 * np.max(np.abs(u_ref))
    d_ref = np.asarray(fwd.R) @ (chi * u_ref).reshape(6, -1).T  # (Nr, Nt)
    np.testing.assert_allclose(np.asarray(fwd.scattered(chi, u)), d_ref.T, rtol=1e-8, atol=1e-12)


def test_solver_variants_agree():
    grid = Grid(24, 2.0)
    setup = Setup(grid, Antennas(TX, RX))
    chi = jnp.asarray(cylinder(grid, 0.7, 4.0))
    ref, _ = Forward(setup, tol=1e-12)(chi)
    for kw in (dict(batch="vmap"), dict(restart=12, maxiter=200), dict(solver="bicgstab", maxiter=500)):
        d, _ = Forward(setup, tol=1e-11, **kw)(chi)
        assert _rel(d, ref) < 1e-8, kw


# ----------------------------------------------------------------------------- GMRES (krylov.py)


def _random_system(rng, B=3, N=40, shift=0.0):
    """Complex non-Hermitian matrices with eigenvalues spread around 1 (an I - G chi-like spectrum)."""
    A = np.eye(N)[None] + 0.25 * (rng.standard_normal((B, N, N)) + 1j * rng.standard_normal((B, N, N))) / np.sqrt(N)
    A = A + shift
    b = rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))
    return jnp.asarray(A), jnp.asarray(b)


def test_gmres_matches_direct_solve_and_meets_true_residual_tolerance():
    A, b = _random_system(np.random.default_rng(0))
    mv = lambda x: jnp.einsum("bij,bj->bi", A, x)
    x_ref = np.stack([np.linalg.solve(np.asarray(A[i]), np.asarray(b[i])) for i in range(3)])
    for tol in (1e-4, 1e-8, 1e-12):
        x = gmres_stacked(mv, b, tol=tol, restart=10, maxiter=100)
        r = np.asarray(b - mv(x))
        rel = np.linalg.norm(r, axis=1) / np.linalg.norm(np.asarray(b), axis=1)
        assert rel.max() <= tol, (tol, rel)  # the TRUE residual, not an estimate
        assert rel.max() > 0.05 * tol  # and it stops once the tolerance is met instead of overshooting
    assert np.abs(np.asarray(x) - x_ref).max() < 1e-9 * np.abs(x_ref).max()


def test_gmres_tolerance_is_relative_to_each_lanes_own_norm():
    A, b = _random_system(np.random.default_rng(1))
    scale = jnp.asarray([1.0, 1e-8, 1e6])[:, None]
    b = b * scale
    mv = lambda x: jnp.einsum("bij,bj->bi", A, x)
    x = gmres_stacked(mv, b, tol=1e-10, restart=10, maxiter=100)
    rel = np.linalg.norm(np.asarray(b - mv(x)), axis=1) / np.linalg.norm(np.asarray(b), axis=1)
    assert rel.max() <= 1e-10, rel  # a global tolerance would leave the 1e-8 lane unconverged


def test_gmres_warm_start_exact_solution_costs_one_operator_application():
    A, b = _random_system(np.random.default_rng(2), B=2)
    x_exact = jnp.asarray(np.stack([np.linalg.solve(np.asarray(A[i]), np.asarray(b[i])) for i in range(2)]))
    calls = []

    def mv(x):
        jax.debug.callback(lambda: calls.append(1))
        return jnp.einsum("bij,bj->bi", A, x)

    cold = gmres_stacked(mv, b, tol=1e-6, restart=10, maxiter=50)
    jax.effects_barrier()
    n_cold = len(calls)
    calls.clear()
    # warm start at the converged solution: one residual evaluation, no Arnoldi steps, x unchanged
    warm = gmres_stacked(mv, b, cold, tol=1e-6, restart=10, maxiter=50)
    jax.effects_barrier()
    assert len(calls) == 1 and n_cold > 10
    np.testing.assert_array_equal(np.asarray(warm), np.asarray(cold))
    # a start that is already close needs fewer applications than a cold start
    calls.clear()
    near = gmres_stacked(mv, b, x_exact + 1e-3 * (1 + 1j), tol=1e-6, restart=10, maxiter=50)
    jax.effects_barrier()
    assert 1 < len(calls) < n_cold
    assert np.abs(np.asarray(near - x_exact)).max() < 1e-4


def test_gmres_zero_rhs_lane_and_maxiter_cap_stay_finite():
    A, b = _random_system(np.random.default_rng(4))
    b = b.at[1].set(0.0)
    mv = lambda x: jnp.einsum("bij,bj->bi", A, x)
    x = gmres_stacked(mv, b, tol=1e-10, restart=10, maxiter=100)
    assert np.all(np.isfinite(np.asarray(x))) and np.all(np.asarray(x[1]) == 0.0)
    capped = gmres_stacked(mv, b, tol=1e-14, restart=2, maxiter=1)  # cannot converge: returns best iterate
    assert np.all(np.isfinite(np.asarray(capped)))
    r0 = np.linalg.norm(np.asarray(b[0]))
    assert np.linalg.norm(np.asarray(b[0] - mv(capped)[0])) < r0  # still reduced the residual


def test_gmres_real_dtype_and_restart_longer_than_system():
    rng = np.random.default_rng(5)
    A = jnp.asarray(np.eye(6)[None] + 0.2 * rng.standard_normal((1, 6, 6)))
    b = jnp.asarray(rng.standard_normal((1, 6)))
    x = gmres_stacked(lambda v: jnp.einsum("bij,bj->bi", A, v), b, tol=1e-13, restart=50, maxiter=3)
    np.testing.assert_allclose(np.asarray(x[0]), np.linalg.solve(np.asarray(A[0]), np.asarray(b[0])), rtol=1e-10)


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        Forward(_small_setup(8), solver="lu")
    with pytest.raises(ValueError):
        Forward(_small_setup(8), batch="loop")


# ----------------------------------------------------------------------------- Born


def test_born_matches_forward_at_tiny_contrast_and_not_at_large():
    grid = Grid(48, 2.0)
    fwd = Forward(Setup(grid, Antennas(TX, RX)), tol=1e-12)
    shape = cylinder(grid, 0.6, 2.0)  # contrast 1 on the disk
    err_small = _rel(fwd(1e-4 * shape)[0], fwd.born(1e-4 * shape))
    err_large = _rel(fwd(shape)[0], fwd.born(shape))
    print(f"\nBorn vs forward relative difference: chi=1e-4: {err_small:.2e}, chi=1: {err_large:.2e}")
    assert err_small < 1e-3  # measured ~1e-4 (second-order scattering ~ chi)
    assert err_large > 0.05  # the test is not vacuous: Born is badly off at contrast 1
    assert err_large > 100 * err_small


def test_born_equals_born_matrix_times_chi():
    grid = Grid(16, 2.0)
    fwd = Forward(Setup(grid, Antennas(TX, RX)))
    A = fwd.born_matrix()
    assert A.shape == (6 * 8, 256) and A.dtype == jnp.complex128
    rng = np.random.default_rng(1)
    chi = rng.uniform(-0.5, 2.0, size=(16, 16))
    b1 = np.asarray(fwd.born(chi)).reshape(-1)  # (t, r) row-major: row index t * Nr + r
    b2 = np.asarray(A) @ chi.reshape(-1)
    np.testing.assert_allclose(b1, b2, rtol=1e-12, atol=1e-15)
    # explicit entry: A[(t, r), m] = R[r, m] u_inc,t[m]
    t, r, m = 3, 5, 7 * 16 + 2
    np.testing.assert_allclose(np.asarray(A)[t * 8 + r, m],
                               np.asarray(fwd.R)[r, m] * np.asarray(fwd.u_inc).reshape(6, -1)[t, m], rtol=1e-14)
    # linearity in chi, including complex chi
    chi2 = chi + 1j * rng.standard_normal((16, 16))
    np.testing.assert_allclose(np.asarray(fwd.born(chi2)).reshape(-1), np.asarray(A) @ chi2.reshape(-1),
                               rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------- reciprocity


def test_reciprocity_swapping_transmitters_and_receivers():
    """The scattered field at B from a source at A equals the field at A from a source at B, for a
    heterogeneous, non-symmetric object. Setup 1 has sources A and receivers B; setup 2 swaps them."""
    n = 20
    grid = Grid(n, 2.0)
    A = ring(5, 3.0, offset=0.2)
    B = ring(7, 3.6, offset=0.9)
    rng = np.random.default_rng(2)
    chi = jnp.asarray(cylinder(grid, 0.5, 3.0, center=(0.3, 0.2)) + 0.4 * rng.uniform(0, 1, (n, n)))
    d_ab, _ = Forward(Setup(grid, Antennas(A, B)), tol=1e-12)(chi)  # (5, 7)
    d_ba, _ = Forward(Setup(grid, Antennas(B, A)), tol=1e-12)(chi)  # (7, 5)
    d_ab, d_ba = np.asarray(d_ab), np.asarray(d_ba)
    assert d_ab.shape == (5, 7) and d_ba.shape == (7, 5)
    scale = np.max(np.abs(d_ab))
    assert np.max(np.abs(d_ab - d_ba.T)) < 1e-9 * scale
    # the object is genuinely non-trivial: data are not symmetric in an unrelated pairing
    assert np.max(np.abs(d_ab - d_ab[::-1, ::-1])) > 1e-2 * scale
    # co-located transmitters and receivers give a symmetric data matrix
    P = ring(6, 3.2, offset=0.1)
    d_pp = np.asarray(Forward(Setup(grid, Antennas(P, P)), tol=1e-12)(chi)[0])
    assert np.max(np.abs(d_pp - d_pp.T)) < 1e-9 * np.max(np.abs(d_pp))
