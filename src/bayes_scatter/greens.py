"""Geometry-only precomputation for the Lippmann-Schwinger forward model (docs/DESIGN.md 2.2-2.3).

Everything here is NumPy/SciPy (``hankel1``, ``jv``) and returns plain ndarrays. The pixel kernel is
the Richmond equal-area-disk integral K(r, cell m') = k0^2 int_{cell m'} G(r, r') dr' with
G = (i/4) H0^(1)(k0 |r - r'|) and, with rho = |r - center of m'| and a = h / sqrt(pi):

* off-cell (rho > a):  K = (i pi / 2) k0 a J1(k0 a) H0^(1)(k0 rho)
* self term:           K = (i pi / 2) k0 a H1^(1)(k0 a) - 1

Arrays over pixels use "ij" indexing, [i, j] = [x, y], flattened in C order (k = i n + j).
"""

from typing import NamedTuple

import numpy as np
from scipy.special import hankel1, jv


class Operators(NamedTuple):
    """Geometry-only operators of one setup (all complex128).

    g_hat : (2n, 2n) FFT of the circularly embedded pixel kernel.
    R     : (Nr, n*n) receiver matrix, R[r, m] = off-cell kernel from pixel m to receiver r.
    u_inc : (Nt, n, n) incident field of each unit line source.
    """

    g_hat: np.ndarray
    R: np.ndarray
    u_inc: np.ndarray


def offcell_coefficient(k0, a):
    """The constant (i pi / 2) k0 a J1(k0 a) multiplying H0^(1)(k0 rho) in the off-cell kernel."""
    return 0.5j * np.pi * k0 * a * jv(1, k0 * a)


def self_term(k0, a):
    """Pixel self term (i pi / 2) k0 a H1^(1)(k0 a) - 1 (the field point is the cell center)."""
    return 0.5j * np.pi * k0 * a * hankel1(1, k0 * a) - 1.0


def offcell_kernel(k0, a, rho):
    """Off-cell kernel k0^2 int_disk G dA for a field point at distance rho > a from the disk center."""
    rho = np.asarray(rho, dtype=np.float64)
    if np.any(rho <= a):
        raise ValueError("off-cell kernel requires rho > a")
    return offcell_coefficient(k0, a) * hankel1(0, k0 * rho)


def kernel_table(setup):
    """Pixel kernel g(di, dj) for offsets di, dj in [-(n-1), n-1], shape (2n-1, 2n-1).

    Entry [di + n - 1, dj + n - 1] is the self term at di = dj = 0 and otherwise the off-cell kernel at
    rho = h sqrt(di^2 + dj^2), which is at least h > a.
    """
    grid, k0 = setup.grid, setup.k0
    n, h, a = grid.n, grid.h, grid.a
    d = np.arange(-(n - 1), n)
    di, dj = np.meshgrid(d, d, indexing="ij")
    rho = h * np.hypot(di, dj)
    g = np.empty(rho.shape, dtype=np.complex128)
    nonzero = rho > 0
    g[nonzero] = offcell_kernel(k0, a, rho[nonzero])
    g[~nonzero] = self_term(k0, a)
    return g


def embed_kernel(g):
    """Circular embedding of the (2n-1, 2n-1) kernel table into (2n, 2n): K[di mod 2n, dj mod 2n] = g."""
    n = (g.shape[0] + 1) // 2
    d = np.arange(-(n - 1), n)
    K = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    K[np.ix_(d % (2 * n), d % (2 * n))] = g
    return K


def kernel_fft(setup):
    """FFT (2n x 2n) of the embedded kernel; ``G w = ifft2(g_hat * fft2(pad(w)))[:n, :n]``."""
    return np.fft.fft2(embed_kernel(kernel_table(setup)))


def dense_green_matrix(setup):
    """Dense (n^2, n^2) matrix G, G[(i, j), (i', j')] = g(i - i', j - j'). For small grids and tests."""
    n = setup.grid.n
    g = kernel_table(setup)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    return g[(i[:, None] - i[None, :]) + n - 1, (j[:, None] - j[None, :]) + n - 1]


def _pixel_points(grid):
    X, Y = grid.centers()
    return np.stack([X.ravel(), Y.ravel()], axis=-1)


def _distances(points, grid):
    """Distances (len(points), n*n) from points to all pixel centers."""
    pix = _pixel_points(grid)
    return np.linalg.norm(points[:, None, :] - pix[None, :, :], axis=-1)


def check_clearance(setup):
    """Raise ValueError unless every transmitter and receiver is farther than a from every pixel center.

    The off-cell kernel (Graf's addition theorem) is valid only outside the equal-area disk.
    Returns the smallest antenna-to-pixel-center distance.
    """
    a = setup.grid.a
    smallest = np.inf
    for name, pts in (("transmitter", setup.antennas.tx), ("receiver", setup.antennas.rx)):
        dmin = _distances(pts, setup.grid).min()
        if dmin <= a:
            raise ValueError(
                f"a {name} lies within a = {a:.4g} of a pixel center (distance {dmin:.4g}); "
                "the off-cell kernel is invalid there"
            )
        smallest = min(smallest, dmin)
    return smallest


def receiver_matrix(setup):
    """R (Nr, n*n): R[r, m] = (i pi/2) k0 a J1(k0 a) H0^(1)(k0 |r_r - c_m|)."""
    grid = setup.grid
    rho = _distances(setup.antennas.rx, grid)
    return offcell_kernel(setup.k0, grid.a, rho)


def incident_fields(setup):
    """Incident fields (Nt, n, n) of unit line sources: u_inc,t(r) = (i/4) H0^(1)(k0 |r - r_t|)."""
    grid = setup.grid
    rho = _distances(setup.antennas.tx, grid)
    return (0.25j * hankel1(0, setup.k0 * rho)).reshape(-1, grid.n, grid.n)


def build_operators(setup):
    """Validate the layout and return the ``Operators`` (g_hat, R, u_inc) of a setup."""
    check_clearance(setup)
    return Operators(
        g_hat=kernel_fft(setup), R=receiver_matrix(setup), u_inc=incident_fields(setup)
    )
