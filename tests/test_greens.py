"""Tests for greens.py and the FFT operator in forward.py (docs/DESIGN.md 2.2, 2.3, 8.1)."""

import numpy as np
import pytest
from scipy import integrate
from scipy.special import hankel1

from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.greens import (build_operators, check_clearance, dense_green_matrix, embed_kernel,
                                  incident_fields, kernel_fft, kernel_table, offcell_kernel,
                                  receiver_matrix, self_term)

K0 = 2 * np.pi
# Two pixel sizes: h = 5/96 = 0.052 (k0 a = 0.19) and h = 5/24 = 0.208 (k0 a = 0.74).
SETUPS = [Setup(Grid(96, 5.0), Antennas(ring(2, 4.0), ring(2, 4.5))),
          Setup(Grid(24, 5.0), Antennas(ring(2, 4.0), ring(2, 4.5)))]


def _quad_complex(f, *args, **kw):
    re = integrate.quad(lambda x: f(x).real, *args, **kw)[0]
    im = integrate.quad(lambda x: f(x).imag, *args, **kw)[0]
    return re + 1j * im


def _disk_integral_offcell(a, rho):
    """k0^2 int_disk(radius a, centered at origin) (i/4) H0(k0 |(rho, 0) - r'|) dA' by 2-D quadrature."""

    def integrand(part):
        def f(rp, ph):
            v = hankel1(0, K0 * np.hypot(rho - rp * np.cos(ph), rp * np.sin(ph))) * rp
            return v.real if part == "re" else v.imag
        return f

    kw = dict(epsabs=1e-13, epsrel=1e-12)
    re = integrate.dblquad(integrand("re"), 0, 2 * np.pi, 0, a, **kw)[0]
    im = integrate.dblquad(integrand("im"), 0, 2 * np.pi, 0, a, **kw)[0]
    return K0**2 * 0.25j * (re + 1j * im)


def _disk_integral_self(a):
    """k0^2 int_disk (i/4) H0(k0 |r'|) dA' with the field point at the disk center (polar, 1-D)."""
    radial = _quad_complex(lambda r: hankel1(0, K0 * r) * r, 0, a, epsabs=1e-14, epsrel=1e-13)
    return K0**2 * 0.25j * 2 * np.pi * radial


@pytest.mark.parametrize("setup", SETUPS, ids=["h=0.052", "h=0.208"])
def test_self_term_matches_quadrature(setup):
    n, a = setup.grid.n, setup.grid.a
    ref = _disk_integral_self(a)
    np.testing.assert_allclose(self_term(K0, a), ref, rtol=1e-11, atol=0)
    np.testing.assert_allclose(kernel_table(setup)[n - 1, n - 1], ref, rtol=1e-11, atol=0)


@pytest.mark.parametrize("setup", SETUPS, ids=["h=0.052", "h=0.208"])
def test_offcell_kernel_matches_quadrature(setup):
    n, h, a = setup.grid.n, setup.grid.h, setup.grid.a
    tab = kernel_table(setup)
    for di, dj in [(1, 0), (0, 1), (1, 1), (3, 2), (10, 0)]:
        rho = h * np.hypot(di, dj)
        ref = _disk_integral_offcell(a, rho)
        np.testing.assert_allclose(tab[n - 1 + di, n - 1 + dj], ref, rtol=1e-11, atol=0)
    far = 40.0 * h
    np.testing.assert_allclose(offcell_kernel(K0, a, far), _disk_integral_offcell(a, far), rtol=1e-11)


def test_offcell_kernel_requires_outside_cell():
    with pytest.raises(ValueError):
        offcell_kernel(K0, 0.1, 0.1)


@pytest.mark.parametrize("setup", SETUPS, ids=["h=0.052", "h=0.208"])
def test_equal_area_disk_approximates_square_pixel(setup):
    """Independent physical check of the Richmond approximation: the disk self term is within about 1%
    of the exact square-pixel integral (0.4% and 0.9% measured). A missing '-1' or a factor 2 in the
    self term would change it by 10% or more, so this guards the constants and not only the algebra."""
    h, n = setup.grid.h, setup.grid.n

    def wedge(ph):  # one of 8 congruent wedges of the square, radial integral in closed form by quad
        rmax = 0.5 * h / np.cos(ph)
        return _quad_complex(lambda r: hankel1(0, K0 * r) * r, 0, rmax, epsabs=1e-14, epsrel=1e-12)

    square = K0**2 * 0.25j * 8 * _quad_complex(wedge, 0, np.pi / 4, epsabs=1e-13, epsrel=1e-11)
    disk = kernel_table(setup)[n - 1, n - 1]
    assert abs(disk - square) / abs(square) < 0.015
    assert abs(disk - square) / abs(square) > 1e-4  # the approximation is not exact; test is meaningful


def test_kernel_table_symmetry_and_embedding():
    setup = SETUPS[1]
    n = setup.grid.n
    g = kernel_table(setup)
    assert g.shape == (2 * n - 1, 2 * n - 1)
    np.testing.assert_array_equal(g, g[::-1, :])  # g(-di, dj) = g(di, dj)
    np.testing.assert_array_equal(g, g[:, ::-1])
    np.testing.assert_array_equal(g, g.T)
    K = embed_kernel(g)
    assert K.shape == (2 * n, 2 * n)
    assert K[n, :].max() == 0 and K[:, n].max() == 0  # the unused wrap-around row/column
    assert K[0, 0] == g[n - 1, n - 1]
    assert K[-1, 0] == g[n - 2, n - 1]  # di = -1 wraps to index 2n - 1
    np.testing.assert_allclose(kernel_fft(setup), np.fft.fft2(K))


def test_fft_apply_G_equals_dense_matvec_n6():
    grid = Grid(6, 1.5)
    setup = Setup(grid, Antennas(ring(3, 3.0), ring(4, 3.0, offset=0.4)))
    fwd = Forward(setup)
    G = dense_green_matrix(setup)
    assert G.shape == (36, 36)
    np.testing.assert_array_equal(G, G.T)  # the kernel is symmetric
    rng = np.random.default_rng(0)
    w = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    out = np.asarray(fwd.apply_G(w))
    ref = (G @ w.reshape(-1)).reshape(6, 6)
    assert np.max(np.abs(out - ref)) < 1e-12 * np.max(np.abs(ref))
    # real input, and batches of shape (..., n, n)
    wb = rng.standard_normal((2, 3, 6, 6))
    outb = np.asarray(fwd.apply_G(wb))
    refb = (wb.reshape(6, 36) @ G.T).reshape(2, 3, 6, 6)
    assert outb.shape == (2, 3, 6, 6)
    assert np.max(np.abs(outb - refb)) < 1e-12 * np.max(np.abs(refb))
    # a delta at pixel (1, 4) reproduces the matching column of G (index conventions: k = i n + j)
    e = np.zeros((6, 6)); e[1, 4] = 1.0
    np.testing.assert_allclose(np.asarray(fwd.apply_G(e)).reshape(-1), G[:, 1 * 6 + 4], atol=1e-14)


def test_dense_G_is_block_toeplitz_with_correct_diagonal():
    grid = Grid(5, 1.0)
    G = dense_green_matrix(Setup(grid, Antennas(ring(2, 3.0), ring(2, 3.5))))
    assert np.allclose(np.diag(G), self_term(K0, grid.a))
    # block-Toeplitz structure: G[(i,j),(i',j')] depends only on (i - i', j - j')
    G4 = G.reshape(5, 5, 5, 5)
    np.testing.assert_allclose(G4[1, 2, 0, 1], G4[2, 3, 1, 2])  # offset (1, 1)
    np.testing.assert_allclose(G4[1, 2, 0, 3], G4[3, 2, 2, 3])  # offset (1, -1)
    # the kernel depends on distance only: offsets (1, 0), (0, 1), (-1, 0), (0, -1) are equal
    ref = G4[2, 2, 3, 2]
    for val in (G4[2, 2, 2, 3], G4[3, 2, 2, 2], G4[2, 3, 2, 2]):
        np.testing.assert_allclose(val, ref)
    assert not np.isclose(G4[2, 2, 3, 3], ref)  # (1, 1) is farther away


def test_receiver_matrix_matches_quadrature():
    grid = Grid(12, 2.0)
    ant = Antennas(ring(2, 3.0), np.array([[2.5, 1.0], [-1.0, -3.0], [0.0, 3.2]]))
    setup = Setup(grid, ant)
    R = receiver_matrix(setup)
    assert R.shape == (3, 144)
    X, Y = grid.centers()
    for r, m in [(0, 0), (1, 5 * 12 + 7), (2, 143)]:
        cx, cy = X.ravel()[m], Y.ravel()[m]
        rho = np.hypot(ant.rx[r, 0] - cx, ant.rx[r, 1] - cy)
        ref = _disk_integral_offcell(grid.a, rho)  # same kernel by translation invariance
        np.testing.assert_allclose(R[r, m], ref, rtol=1e-11)


def test_incident_fields_are_line_source_fields():
    grid = Grid(10, 2.0)
    tx = np.array([[3.0, 0.5], [-2.0, 2.5], [0.0, -4.0]])
    setup = Setup(grid, Antennas(tx, ring(2, 4.0)))
    u = incident_fields(setup)
    assert u.shape == (3, 10, 10) and u.dtype == np.complex128
    X, Y = grid.centers()
    for t in range(3):
        rho = np.hypot(X - tx[t, 0], Y - tx[t, 1])
        np.testing.assert_allclose(u[t], 0.25j * hankel1(0, K0 * rho), rtol=1e-14)
    # the field of a line source is outgoing: at large distance |u| ~ (1/4) sqrt(2 / (pi k0 rho))
    far = 0.25 * np.sqrt(2 / (np.pi * K0 * np.hypot(X - tx[0, 0], Y - tx[0, 1])))
    assert np.max(np.abs(np.abs(u[0]) - far) / far) < 0.01


def test_clearance_check_rejects_antenna_inside_a_cell():
    grid = Grid(8, 2.0)
    X, Y = grid.centers()
    bad_rx = np.array([[X[3, 4] + 0.5 * grid.a, Y[3, 4]]])
    with pytest.raises(ValueError, match="receiver"):
        check_clearance(Setup(grid, Antennas(ring(2, 3.0), bad_rx)))
    with pytest.raises(ValueError, match="transmitter"):
        build_operators(Setup(grid, Antennas(np.array([[X[0, 0], Y[0, 0]]]), ring(2, 3.0))))
    with pytest.raises(ValueError):
        Forward(Setup(grid, Antennas(ring(2, 3.0), bad_rx)))
    ok = check_clearance(Setup(grid, Antennas(ring(2, 3.0), ring(3, 3.5))))
    assert ok > grid.a
