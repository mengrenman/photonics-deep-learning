"""Tests for bayes_scatter.jacobian: the reciprocity Jacobian against jax.jvp columns, and its consequences."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

from bayes_scatter import jacobian as jb
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.greens import dense_green_matrix


@pytest.fixture(scope="module")
def tiny():
    """8 x 8 grid on a 1.5 x 1.5 DOI, 6 transmitters, 9 receivers, tight solver tolerance."""
    setup = Setup(Grid(8, 1.5), Antennas(ring(6, 2.0), ring(9, 2.4, offset=0.2)))
    fwd = Forward(setup, tol=1e-13)
    rng = np.random.default_rng(1)
    chi = 0.2 + 1.4 * rng.random((8, 8))  # eps_r between 1.2 and 2.6
    return {"setup": setup, "fwd": fwd, "chi": jnp.asarray(chi), "rng": rng}


def test_green_matrix_is_symmetric_but_the_system_matrix_is_not(tiny):
    G = dense_green_matrix(tiny["setup"])
    np.testing.assert_allclose(G, G.T, atol=0.0)  # g(m - m') depends on the distance only: the plain transpose
    A = np.eye(64) - G @ np.diag(np.asarray(tiny["chi"]).ravel())
    assert np.abs(A - A.T).max() > 1e-2  # I - G X is NOT symmetric (so the derivation uses A^T = I - X G)


def test_solve_rhs_matches_a_dense_solve(tiny):
    G = dense_green_matrix(tiny["setup"])
    chi = np.asarray(tiny["chi"]).ravel()
    A = np.eye(64) - G @ np.diag(chi)
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((5, 8, 8)) + 1j * rng.standard_normal((5, 8, 8))
    x = np.asarray(tiny["fwd"].solve_rhs(tiny["chi"], rhs))
    ref = np.linalg.solve(A, rhs.reshape(5, 64).T).T.reshape(5, 8, 8)
    np.testing.assert_allclose(x, ref, rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        tiny["fwd"].solve_rhs(tiny["chi"], rhs[0])


def test_jacobian_matches_jvp_columns_to_1e_8(tiny):
    fwd, chi = tiny["fwd"], tiny["chi"]
    J = jb.jacobian_chi(fwd, chi)
    assert J.shape == (fwd.n_tx * fwd.n_rx, 64) and J.dtype == jnp.complex128
    cols = np.arange(0, 64, 5)
    assert jb.verify_jacobian(fwd, chi, J, cols) < 1e-8
    # ... and every column of the whole matrix against the one-hot jvp directly
    ref = jb.jvp_columns(fwd, chi, np.arange(64))
    assert float(jnp.max(jnp.linalg.norm(J - ref, axis=0) / jnp.linalg.norm(ref, axis=0))) < 1e-8


def test_jacobian_holds_for_complex_contrast(tiny):
    """Lossy medium: only the plain transpose is used, so the formula is valid for complex chi."""
    fwd = tiny["fwd"]
    chi = tiny["chi"] * (1.0 + 0.25j)
    J = jb.jacobian_chi(fwd, chi)
    assert jb.verify_jacobian(fwd, chi, J, np.arange(0, 64, 7)) < 1e-8


def test_jacobian_theta_is_the_chi_jacobian_times_the_link_derivative(tiny):
    fwd = tiny["fwd"]
    theta = jnp.log(1.0 + tiny["chi"])
    from bayes_scatter.priors import log_eps

    J = jb.jacobian_theta(fwd, theta, "log_eps")
    scale = jb.link_derivative("log_eps", theta)
    np.testing.assert_allclose(np.asarray(scale), np.exp(np.asarray(theta)), rtol=1e-14)  # eps_r
    np.testing.assert_allclose(np.asarray(J), np.asarray(jb.jacobian_chi(fwd, log_eps(theta)))
                               * np.exp(np.asarray(theta)).reshape(1, -1), rtol=1e-12)
    assert jb.verify_jacobian(fwd, log_eps(theta), J, np.arange(1, 64, 6), scale) < 1e-8
    np.testing.assert_allclose(np.asarray(jb.jacobian_theta(fwd, theta, "identity")),
                               np.asarray(jb.jacobian_chi(fwd, theta)), rtol=1e-12)


def test_jacobian_at_zero_contrast_is_the_born_matrix(tiny):
    fwd = tiny["fwd"]
    A = np.asarray(fwd.born_matrix())
    np.testing.assert_allclose(np.asarray(jb.jacobian_chi(fwd, jnp.zeros((8, 8)))), A, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(np.asarray(jb.jacobian_chi(fwd, tiny["chi"], born=True)), A, rtol=1e-13)


def test_reciprocal_fields_are_the_fields_of_sources_at_the_receivers(tiny):
    """w_r = 2 pi k0 a J1(k0 a) u^(r), u^(r) the total field of a unit line source at receiver r (reciprocity)."""
    setup, fwd, chi = tiny["setup"], tiny["fwd"], tiny["chi"]
    u, w = jb.jacobian_factors(fwd, chi)
    swapped = Setup(setup.grid, Antennas(setup.antennas.rx, setup.antennas.rx))  # transmitters at the receivers
    u_rx = Forward(swapped, tol=1e-13).solve_fields(chi)
    k0a = setup.k0 * setup.grid.a
    np.testing.assert_allclose(np.asarray(w), 2.0 * np.pi * k0a * jv(1, k0a) * np.asarray(u_rx), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(u), np.asarray(fwd.solve_fields(chi)), rtol=1e-12)


def test_gauss_newton_hessian_from_gram_products_equals_the_dense_formula(tiny):
    fwd = tiny["fwd"]
    theta = jnp.log(1.0 + tiny["chi"])
    s = 3e-3
    u, w = jb.jacobian_factors(fwd, tiny["chi"])
    scale = jb.link_derivative("log_eps", theta)
    H = np.asarray(jb.gauss_newton_hessian(u, w, s, scale))
    J = jb.jacobian_theta(fwd, theta)
    Hd = (2.0 / s**2) * np.real(np.asarray(J).conj().T @ np.asarray(J))
    atol = 1e-13 * np.abs(Hd).max()
    np.testing.assert_allclose(H, Hd, rtol=1e-12, atol=atol)
    np.testing.assert_allclose(np.asarray(jb.gauss_newton_hessian_dense(J, s)), Hd, rtol=1e-12, atol=atol)
    np.testing.assert_allclose(H, H.T, atol=1e-9 * np.abs(H).max())
    assert np.linalg.eigvalsh(H).min() > -1e-9 * np.abs(H).max()  # positive semidefinite
    # independent: the real Jacobian of [Re d; Im d] from jax.jvp columns
    ref = np.asarray(jb.jvp_columns(fwd, tiny["chi"], np.arange(64), np.asarray(scale)))
    Jr = np.vstack([ref.real, ref.imag])
    np.testing.assert_allclose(H, (2.0 / s**2) * Jr.T @ Jr, rtol=1e-8, atol=1e-8 * np.abs(H).max())
