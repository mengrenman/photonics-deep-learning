"""Static Rayleigh multipole method for the square array of circular disks (scalar conductivity or permittivity).

Used by notebooks 01 to 03 as the exact reference for the disk array and as the exact spectral measure.

Conventions
-----------
Unit period (lengths in units of the period), disk of radius R = sqrt(f/pi) centered in the cell, area fraction f
(non-touching: f < pi/4). The host has conductivity 1, the disk has conductivity ``kappa`` (real or complex, so a
complex permittivity ratio is allowed). beta = (kappa - 1)/(kappa + 1). Time dependence exp(-i omega t) is irrelevant
in the static limit; complex kappa simply continues the formulas analytically.

The multipole expansion keeps the odd orders l = 1, 3, ..., 2L - 1 (C4 symmetry removes the others). The real
symmetric Rayleigh matrix W is

    W_lm = sqrt(l/m) R^(l+m) [ pi delta_l1 delta_m1 + C(l+m-1, l) S_(l+m) 1(l+m >= 4) ],

with S_n = sum' (p + i q)^-n over the unit square lattice (origin excluded). The pi in the (1,1) slot is the
periodic-field (S_2) term; with it W_11 = pi R^2 + O(R^6) and the truncation L = 1 is exactly Maxwell Garnett.

Public functions and their contracts
------------------------------------
``square_lattice_sums(nmax)`` -> dict {n: S_n} for even n = 4, ..., nmax. Exact up to rounding: S_4 =
    Gamma(1/4)^8/(960 pi^2) = 3.1512120021539, S_n = 0 unless n = 0 mod 4 (in particular S_6 = 0), and the
    recursion (2k+1)(k-3)(2k-1) S_2k = 3 sum_{j=2}^{k-2} (2j-1)(2k-2j-1) S_2j S_(2k-2j) for k >= 4 (so S_8 = 3/7 S_4^2).
``rayleigh_matrix(f, L)`` -> W (L, L), real symmetric. Guarantee: W_11 = pi R^2 = f exactly (the l + m = 2 slot
    carries only the pi term), so the sum rules of the measure hold at every truncation.
``sigma_eff(f, kappa, L=20)`` -> sigma*/sigma_host = 1 + 2 f [(I/beta - W)^-1]_11. Same shape as ``kappa``
    (real or complex). Exponentially convergent in L (about 1e-15 relative by L = 10 for f <= 0.5; slower near
    touching, f -> pi/4, where the convergence in L degrades). Approximate only through the truncation L; L = 1 is
    exactly Maxwell Garnett.
``rayleigh_measure(f, L=20)`` -> (z, w): atoms z_k = (1 - mu_k)/2 (ascending) and weights w_k = f V[0, k]^2, where
    W = V diag(mu) V^T. Guarantee: all w_k > 0 and the eigenvalues mu_k lie in (-1, 1); the identity
    1 - sigma_eff(f, kappa, L) = sum_k w_k/(s - z_k) with s = 1/(1 - kappa) holds exactly at every L; the sum rules
    m0 = sum w = f and m1 = sum w z = f (1 - f)/2 hold to rounding at every L (V is orthogonal, W_11 = f).
"""

from math import comb, gamma, pi

import numpy as np


def square_lattice_sums(nmax):
    """Eisenstein sums S_n = sum'_{(p,q) != 0} (p + i q)^-n of the unit square lattice, n = 4, 6, ..., nmax.

    Returns {n: S_n} (floats). S_4 = Gamma(1/4)^8/(960 pi^2); S_n = 0 unless n is a multiple of 4; higher values by
    the Weierstrass recursion (2k+1)(k-3)(2k-1) S_2k = 3 sum_{j=2}^{k-2} (2j-1)(2k-2j-1) S_2j S_(2k-2j).
    """
    G = {4: gamma(0.25) ** 8 / (960.0 * pi ** 2), 6: 0.0}
    for k in range(4, nmax // 2 + 1):
        s = sum((2 * j - 1) * (2 * k - 2 * j - 1) * G[2 * j] * G[2 * k - 2 * j] for j in range(2, k - 1))
        G[2 * k] = 3.0 * s / ((2 * k + 1) * (k - 3) * (2 * k - 1))
    return {n: v for n, v in G.items() if n <= max(nmax, 6)}


def rayleigh_matrix(f, L):
    """Real symmetric Rayleigh matrix W (L, L) for area fraction f, odd orders 1, 3, ..., 2L-1 (see module docstring)."""
    R = np.sqrt(f / pi)
    ls = np.arange(1, 2 * L, 2)
    S = square_lattice_sums(4 * L + 4)
    W = np.zeros((L, L))
    for a, l in enumerate(ls):
        for b, m in enumerate(ls):
            n = int(l + m)
            t = pi if (l == 1 and m == 1) else 0.0
            if n >= 4:
                t += float(comb(n - 1, int(l))) * S[n]
            W[a, b] = np.sqrt(l / m) * R ** n * t
    return W


def sigma_eff(f, kappa, L=20):
    """Effective conductivity sigma*/sigma_host = 1 + 2 f [(I/beta - W)^-1]_11 of the square disk array.

    ``kappa`` = sigma_disk/sigma_host, scalar or array, real or complex; the result has the shape of ``kappa``.
    kappa = 1 gives 1. Computed by a batched linear solve (independent of the eigen-decomposition used by
    ``rayleigh_measure``).
    """
    W = rayleigh_matrix(f, L)
    k = np.asarray(kappa)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    out = np.ones(k.shape, dtype=np.result_type(k.dtype, np.float64))
    live = k != 1
    if np.any(live):
        invbeta = (k[live] + 1.0) / (k[live] - 1.0)
        A = invbeta[:, None, None] * np.eye(L)[None] - W[None]
        rhs = np.zeros((A.shape[0], L, 1), dtype=A.dtype)
        rhs[:, 0, 0] = 1.0
        x = np.linalg.solve(A, rhs)[:, 0, 0]
        out[live] = 1.0 + 2.0 * f * x
    return out[0] if scalar else out


def rayleigh_measure(f, L=20):
    """Exact spectral measure (z, w) of the disk array: z_k = (1 - mu_k)/2 ascending, w_k = f V[0, k]^2.

    Here W = V diag(mu) V^T. Then 1 - sigma_eff(f, kappa, L) = sum_k w_k/(s - z_k) with s = 1/(1 - kappa), all weights
    are positive, and the sum rules m0 = f, m1 = f (1 - f)/2 hold at every truncation (V orthogonal, W_11 = f).
    """
    W = rayleigh_matrix(f, L)
    mu, V = np.linalg.eigh(W)
    z = (1.0 - mu) / 2.0
    w = f * V[0] ** 2
    order = np.argsort(z)
    return z[order], w[order]
