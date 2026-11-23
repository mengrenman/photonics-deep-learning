"""Spectral (Bergman-Milton) measures of two-phase cells: three discretizations, measure utilities, closed forms.

Setting
-------
A two-phase cell with phase 1 (indicator chi1, area fraction f) in a host phase 2, conductivities (permittivities)
sigma1 and sigma2, contrast h = sigma1/sigma2. The scalar function

    F(s) = 1 - sigma*/sigma2,    s = 1/(1 - h),

is a Stieltjes function, F(s) = sum_k w_k / (s - z_k) = integral dmu(z) / (s - z), with a positive measure mu on
[0, 1] (atoms z_k in ascending order, weights w_k >= 0). Sum rules: m0 = sum w = f; m1 = sum w z = f (1 - f)/2 for
cells with C4 symmetry (isotropy of the first correction). In terms of the contrast,
sigma*/sigma2 = 1 - F(1/(1 - h)) = 1 - sum_k w_k (1 - h) / (1 - z_k (1 - h)); this form is also valid at h = 1.

Conventions
-----------
* Static problem, so the time convention exp(-i omega t) enters only through the Drude helper ``drude_s``
  (eps1 = 1 - omega_p^2/(omega (omega + i gamma)) in vacuum, which gives s = omega (omega + i gamma)/omega_p^2).
  "TM means E along z" is not used here: the scalar potential is the one of the cell problem of ``cellsolver``
  (with eps in the role of sigma).
* Lengths are in units of the period. Arrays a[i, j] have i as the x index and j as the y index, as in
  ``cellsolver`` (pixel (i, j) is the square [i/N, (i+1)/N] x [j/N, (j+1)/N], node id i*N + j).
* ``direction`` is the macroscopic field direction (0 = x, 1 = y). For cells with C4 or mirror symmetry both
  directions give the same measure.
* ``chi1`` is a 0/1 array (bool or float), True/1 on phase 1.

Discrete measures (each is a finite sum of atoms)
-------------------------------------------------
``q1_measure(chi1, direction=0, zero_tol=1e-10)`` -> (z, w). Dense generalized eigenproblem K1 v = t K v on the pinned periodic Q1
    space (K the unit-coefficient stiffness, K1 that of the phase-1 pixels, V^T K V = I, b the Q1 load of chi1).
    F_h(s) = f/s + (1/s) sum_k w_k/(s - t_k), partial fractions put the weight w_k/t_k at t_k (0 < t_k <= 1) and the
    remainder f - sum w_k/t_k at z = 0. Exact algebraic identity: ``eps_ratio_from_measure`` at h equals the
    q1_effective entry [direction, direction] of the field a = 1 + (h - 1) chi1 to rounding (about 1e-12). The weight
    at z = 0 equals lim U_h/sigma1 as sigma1 -> infinity and is nonzero exactly when phase 1 percolates in the discrete
    space (pixels that touch only at a corner are connected in Q1). Guarantees: m0 = f exactly (by construction),
    z in [0, 1], m1 = f (1 - f)/2 is an upper bound for C4-symmetric cells of the variational type (m1 is a
    maximum over nested spaces). The Q1 measure of a non-percolating pixel set has no atom at 0 (to rounding).
    ``zero_tol``: pencil eigenvalues t <= zero_tol (the numerical kernel of K1, where the load vanishes) are not
    returned as atoms; their weight w/t (at most the mass of an atom at z <= zero_tol) is merged into the atom at 0,
    so atoms below zero_tol are moved to 0 (a position error below 1e-10). The atom at 0 is therefore the
    weight on z <= zero_tol. Cost: dense eigh of size N^2 - 1 (N = 32: 0.1 s, N = 48: 0.7 s).
``fg_measure(shape, K, direction=0, zero_tol=1e-10, **p)`` -> (z, w). Fourier-Galerkin with EXACT coefficients of the analytic
    phase-1 indicator (``cellsolver.fg_coefficients`` with s_in = 1 and s_out = 0, or s_a = 1 and s_b = 0 for the
    checkerboard; any s_in, s_out, s_a, s_b in ``p`` are overridden), trig degree K per coordinate (modes
    |kx|, |ky| <= K, k != 0): the Hermitian eigenproblem of H[l, k] = (k.l)/(|k||l|) chihat(l - k) with load
    beta_l = (l_d/|l|) chihat(l). Same partial fractions as Q1. Exact algebraic identity with
    ``cellsolver.fg_bracket`` upper bound U (the same trial space and quadratic form; the CG residual of fg_bracket
    sets the agreement, about 1e-12). The compression to trig polynomials smears a single dominant atom over a
    cluster of atoms, so atom counts grow with K (disk R = 0.3: 99% of f needs 18, 25, 36, 42 atoms at K = 8, 12,
    16, 20, although the exact measure has one atom with 97.7%). No trigonometric polynomial is affine on a disk,
    but one can be almost affine on it, so the weight at z <= zero_tol is positive yet tiny and falls very fast with K (measured for the
    R = 0.3 disk: 5.9e-5, 1.7e-9, 1.2e-11 at K = 4, 8, 12, of f = 0.28, with a roundoff-limited floor near 1e-11:
    not the O(1/K) of the design). m0 = f exactly, m1 <= f (1 - f)/2 for C4-symmetric shapes. Cost: complex
    eigh of size (2K + 1)^2 - 1 (K = 16: 0.5 s, K = 20: 1.6 s).
``lanczos_measure(chi1, direction=0, steps=150, return_jacobi=False)`` -> (z, w) or (z, w, alpha, beta).
    Sampled (Moulinec-Suquet) scheme. T = chi1 Gamma chi1 on vector fields, Gamma the projector on spectral gradients
    with the Nyquist-zeroed symbol of ``cellsolver.spectral_wavenumbers`` (the same Gamma as
    ``cellsolver.spectral_sampled``), u = chi1 e_direction, F(s) = <u, (s - T)^-1 u> with the mean inner product
    (<u, u> = f). Lanczos with full reorthogonalization (done twice) from u/|u|, restricted to phase-1 pixels
    (T acts there); the Gauss rule of the tridiagonal matrix gives atoms z = Ritz values and weights
    f * (first eigenvector components)^2. Guarantees: m0 = f to rounding; for cells symmetric under x <-> y,
    m1 = f (1 - f)/2 minus half the Nyquist energy of chi1, exactly, where the Nyquist energy is sum |chihat(k)|^2
    over the modes k != 0 whose zeroed wavenumber vanishes ((N/2, 0), (0, N/2), (N/2, N/2) for even N; none for odd
    N; chihat = fft2(chi1)/N^2); the first Jacobi coefficient is alpha_0 = m1/m0; the m-step Gauss rule reproduces the
    first 2m moments of the exact sampled measure and F(s) converges geometrically in m for s outside [0, 1]
    (slowest for s close to 0 or 1, e.g. h = 100 where s = -0.0101). Approximate: the atoms are Ritz values, not the
    full spectrum of T, and the measure is the sampled one (not certified, no sign guarantee on sigma*). Ritz values
    can leave [0, 1] by about 1e-14. Stops early on an invariant subspace (then fewer atoms). Cost: steps * (2 rfft2
    + 2 irfft2 of the (2, N, N) field) plus steps^2 * 2 n1 for the reorthogonalization (N = 256, 300 steps: about
    1.5 s and 160 MB for a half-filled cell).
``lanczos_jacobi(chi1, direction=0, steps=150)`` -> (alpha, beta, f). The Jacobi matrix of the same Lanczos run:
    diagonal alpha[0..m-1], off-diagonal beta[0..m-2] with beta[k] = J[k, k+1] (so beta[k-1] is the "beta_k" of the
    orthogonal-polynomial recurrence, k >= 1), for the normalized measure mu/f; f is the area fraction. ``alpha`` and
    ``beta`` are the same arrays returned by ``lanczos_measure(..., return_jacobi=True)``.
``jacobi_measure(alpha, beta, mass=1.0)`` -> (z, w). Gauss rule of a Jacobi matrix: eigenvalues and
    mass * (first eigenvector components)^2.

Measure utilities
-----------------
``stieltjes(z, w, s)`` -> sum_k w_k/(s - z_k), s scalar or array (real or complex), shape of s preserved.
``eps_ratio_from_measure(z, w, h)`` -> 1 - F(1/(1 - h)) = sigma*/sigma2 (h scalar or array, real or complex).
``moments(z, w, kmax)`` -> array m_0..m_kmax, m_k = sum w z^k.
``cdf(z, w, grid)`` -> right-continuous cumulative weight sum_{z_k <= x} w_k at each grid point.
``wasserstein1(z1, w1, z2, w2, normalize=False)`` -> exact integral of |CDF1 - CDF2| over the span of the union of
    atoms (the Wasserstein-1 distance when the total masses agree; if they differ, the tail beyond the last atom,
    where the difference is the constant mass gap, is not counted; ``normalize=True`` divides each measure by its
    mass first).
``n_atoms_for(z, w, frac)`` -> the smallest number of atoms (largest weights first) carrying at least
    frac * (total weight).
``drude_s(omega, omega_p, gamma)`` -> omega (omega + i gamma)/omega_p^2.

Closed forms for the checkerboard (exact; sqrt(eps1 eps2) of Keller-Dykhne)
---------------------------------------------------------------------------
dmu = (1/pi) sqrt((1 - z)/z) dz on (0, 1), no atoms, because 1 - sqrt(1 - 1/s) = F(s) gives eps*/eps2 = sqrt(h).
``checkerboard_density(z)``, ``checkerboard_cdf(z)`` = M(z) = (arcsin sqrt z + sqrt(z (1 - z)))/pi on [0, 1]
    (0 below, 1/2 above). ``checkerboard_moments(kmax)`` -> m_k = C_k/(2 4^k) exactly (C_k the Catalan numbers:
    1/2, 1/8, 1/16, 5/128, ...). ``checkerboard_jacobi(n)`` -> (alpha, beta): alpha_0 = 1/4, alpha_k = 1/2,
    off-diagonal 1/4 (all), for the normalized measure mu/f (f = 1/2); the orthogonal polynomials are Chebyshev
    polynomials of the fourth kind on [0, 1]. ``checkerboard_gauss(n)`` -> (z, w): the n-point Gauss rule of mu
    (total mass 1/2), the reference for the CDF of discrete measures; it reproduces the first 2n moments exactly.
"""
import math

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla

import cellsolver as cs

__all__ = [
    "q1_measure", "fg_measure", "lanczos_measure", "lanczos_jacobi", "jacobi_measure",
    "stieltjes", "eps_ratio_from_measure", "moments", "cdf", "wasserstein1", "n_atoms_for", "drude_s",
    "checkerboard_density", "checkerboard_cdf", "checkerboard_moments", "checkerboard_jacobi", "checkerboard_gauss",
]

_TZERO = 1e-10  # default zero_tol: pencil eigenvalues t <= zero_tol are merged into the atom at z = 0


# ----------------------------------------------------------------------------- helpers

def _check_indicator(chi1):
    chi = np.asarray(chi1)
    if chi.ndim != 2 or chi.shape[0] != chi.shape[1]:
        raise ValueError("chi1 must be a square 2-D array chi1[i, j]")
    chi = chi.astype(float)
    if not np.all((chi == 0.0) | (chi == 1.0)):
        raise ValueError("chi1 must contain only 0 and 1")
    return chi


def _atoms_from_pencil(t, w, f, zero_tol):
    """Partial fractions of F = f/s + (1/s) sum w/(s - t): atoms w/t at t > zero_tol and f - sum w/t at 0."""
    pos = t > zero_tol
    z = np.concatenate([[0.0], t[pos]])
    wt = np.concatenate([[0.0], w[pos] / t[pos]])
    wt[0] = f - wt[1:].sum()
    order = np.argsort(z, kind="stable")
    return z[order], wt[order]


# ----------------------------------------------------------------------------- Q1

def q1_measure(chi1, direction=0, zero_tol=_TZERO):
    """Spectral measure (z ascending, w) of the periodic Q1 discretization of the pixel cell ``chi1`` (see module)."""
    chi = _check_indicator(chi1)
    N = chi.shape[0]
    conn = cs._conn(N, True)
    nn = N * N
    K1 = cs._assemble(chi, conn, nn)
    K = cs._assemble(np.ones_like(chi), conn, nn)
    b = cs._loads(chi, conn, nn)[direction]
    t, V = sla.eigh(K1[1:, 1:].toarray(), K[1:, 1:].toarray(), driver="gvd")
    w = (V.T @ b[1:]) ** 2
    return _atoms_from_pencil(t, w, chi.mean(), zero_tol)


# ----------------------------------------------------------------------------- Fourier-Galerkin

def _phase1(shape, p):
    q = dict(p)
    for key in ("s_in", "s_a"):
        q.pop(key, None)
    for key in ("s_out", "s_b"):
        q.pop(key, None)
    if shape == "checkerboard":
        q.update(s_a=1.0, s_b=0.0)
    else:
        q.update(s_in=1.0, s_out=0.0)
    return q


def fg_measure(shape, K, direction=0, zero_tol=_TZERO, **p):
    """Spectral measure (z ascending, w) of the Fourier-Galerkin discretization (trig degree K) of an analytic shape.

    ``shape`` and ``p`` as in ``cellsolver.fg_coefficients`` (phase 1 is the inclusion for 'disk', 'square' and
    'laminate', and the s_a tiles for 'checkerboard'; the phase values in ``p`` are ignored).
    """
    K = int(K)
    c = cs.fg_coefficients(shape, 2 * K, **_phase1(shape, p))
    m = np.arange(-K, K + 1)
    MX, MY = np.meshgrid(m, m, indexing="ij")
    keep = (MX != 0) | (MY != 0)
    kx, ky = MX[keep], MY[keep]
    nrm = np.hypot(kx, ky)
    ux, uy = kx / nrm, ky / nrm
    dx = kx[:, None] - kx[None, :] + 2 * K  # index of (l - k): rows l, columns k
    dy = ky[:, None] - ky[None, :] + 2 * K
    H = (ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]) * c[dx, dy]
    H = 0.5 * (H + H.conj().T)
    beta = (ux if direction == 0 else uy) * c[kx + 2 * K, ky + 2 * K]
    t, Q = sla.eigh(H, driver="evd")
    w = np.abs(Q.conj().T @ beta) ** 2
    return _atoms_from_pencil(t, w, float(c[2 * K, 2 * K].real), zero_tol)


# ----------------------------------------------------------------------------- Lanczos (sampled scheme)

def _lanczos(chi1, direction, steps):
    chi = _check_indicator(chi1)
    N = chi.shape[0]
    mask = chi > 0
    n1 = int(mask.sum())
    f = n1 / N ** 2
    if n1 == 0:
        raise ValueError("phase 1 is empty")
    KX, KY = cs.spectral_wavenumbers(N)
    h = N // 2 + 1
    KXr, KYr = KX[:, :h], KY[:, :h]
    k2 = KXr ** 2 + KYr ** 2
    inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    G = np.stack([KXr * KXr * inv, KXr * KYr * inv, KYr * KYr * inv])  # Gamma_xx, Gamma_xy, Gamma_yy
    fw = {"workers": 4}

    def matvec(v):  # v (2, n1) on phase-1 pixels
        full = np.zeros((2, N, N))
        full[:, mask] = v
        Vh = sfft.rfft2(full, axes=(-2, -1), **fw)
        out = np.stack([G[0] * Vh[0] + G[1] * Vh[1], G[1] * Vh[0] + G[2] * Vh[1]])
        return sfft.irfft2(out, s=(N, N), axes=(-2, -1), **fw)[:, mask]

    m = int(steps)
    dim = 2 * n1
    Q = np.zeros((m, dim))
    q = np.zeros((2, n1))
    q[direction] = 1.0 / math.sqrt(n1)
    alpha, beta = [], []
    for j in range(m):
        Q[j] = q.ravel()
        w = matvec(q).ravel()
        a = float(Q[j] @ w)
        alpha.append(a)
        w -= a * Q[j]
        if j > 0:
            w -= beta[-1] * Q[j - 1]
        for _ in range(2):  # full reorthogonalization, twice
            w -= Q[: j + 1].T @ (Q[: j + 1] @ w)
        b = float(np.linalg.norm(w))
        if j == m - 1:
            break
        if b < 1e-12:  # invariant subspace: the measure is exactly represented
            break
        beta.append(b)
        q = (w / b).reshape(2, n1)
    return np.array(alpha), np.array(beta), f


def jacobi_measure(alpha, beta, mass=1.0):
    """Gauss rule of the Jacobi matrix (alpha diagonal, beta off-diagonal): (z ascending, mass * V[0, k]^2)."""
    alpha, beta = np.asarray(alpha, float), np.asarray(beta, float)
    z, V = sla.eigh_tridiagonal(alpha, beta)
    return z, mass * V[0] ** 2


def lanczos_jacobi(chi1, direction=0, steps=150):
    """(alpha, beta, f): Jacobi coefficients of the Lanczos run of ``lanczos_measure`` (see module docstring)."""
    return _lanczos(chi1, direction, steps)


def lanczos_measure(chi1, direction=0, steps=150, return_jacobi=False):
    """Gauss measure (z ascending, w) of the sampled operator T = chi1 Gamma chi1 from u = chi1 e_direction.

    With ``return_jacobi=True`` returns (z, w, alpha, beta) (see ``lanczos_jacobi`` for the layout of alpha, beta).
    """
    alpha, beta, f = _lanczos(chi1, direction, steps)
    z, w = jacobi_measure(alpha, beta, f)
    if return_jacobi:
        return z, w, alpha, beta
    return z, w


# ----------------------------------------------------------------------------- measure utilities

def _chunks(x, size=2000):
    flat = x.reshape(-1)
    for i in range(0, flat.size, size):
        yield i, flat[i:i + size]


def stieltjes(z, w, s):
    """F(s) = sum_k w_k/(s - z_k); s scalar or array (real or complex), the shape of s is preserved."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    s = np.asarray(s)
    out = np.empty(s.shape, dtype=np.result_type(s.dtype, float))
    flat = out.reshape(-1)
    for i, chunk in _chunks(s):
        flat[i:i + chunk.size] = (w[None, :] / (chunk[:, None] - z[None, :])).sum(axis=1)
    return out


def eps_ratio_from_measure(z, w, h):
    """sigma*/sigma2 = 1 - F(1/(1 - h)) = 1 - sum_k w_k (1 - h)/(1 - z_k (1 - h)); h scalar or array, may be complex."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    h = np.asarray(h)
    out = np.empty(h.shape, dtype=np.result_type(h.dtype, float))
    flat = out.reshape(-1)
    for i, chunk in _chunks(h):
        g = 1.0 - chunk[:, None]
        flat[i:i + chunk.size] = 1.0 - (w[None, :] * g / (1.0 - z[None, :] * g)).sum(axis=1)
    return out


def moments(z, w, kmax):
    """Array [m_0, ..., m_kmax] with m_k = sum_j w_j z_j^k."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    return np.array([np.sum(w * z ** k) for k in range(int(kmax) + 1)])


def cdf(z, w, grid):
    """Right-continuous cumulative weight at the points of ``grid``: sum of w_k over z_k <= x."""
    z, w = np.asarray(z, float), np.asarray(w, float)
    o = np.argsort(z, kind="stable")
    c = np.concatenate([[0.0], np.cumsum(w[o])])
    return c[np.searchsorted(z[o], np.asarray(grid, float), side="right")]


def wasserstein1(z1, w1, z2, w2, normalize=False):
    """Exact integral of |CDF1 - CDF2| over the span of the union of atoms (Wasserstein-1 for equal masses)."""
    w1, w2 = np.asarray(w1, float), np.asarray(w2, float)
    if normalize:
        w1, w2 = w1 / w1.sum(), w2 / w2.sum()
    pts = np.unique(np.concatenate([np.asarray(z1, float), np.asarray(z2, float)]))
    d = np.abs(cdf(z1, w1, pts) - cdf(z2, w2, pts))
    return float(np.sum(d[:-1] * np.diff(pts)))


def n_atoms_for(z, w, frac):
    """Smallest number of atoms (largest weights first) whose total weight is at least frac * sum(w)."""
    w = np.sort(np.asarray(w, float))[::-1]
    c = np.cumsum(w)
    return int(np.searchsorted(c, frac * c[-1] * (1.0 - 1e-15), side="left") + 1)


def drude_s(omega, omega_p, gamma):
    """s = omega (omega + i gamma)/omega_p^2 for Drude inclusions in vacuum (time dependence exp(-i omega t))."""
    omega = np.asarray(omega, float)
    return omega * (omega + 1j * gamma) / omega_p ** 2


# ----------------------------------------------------------------------------- checkerboard closed forms

def checkerboard_density(z):
    """(1/pi) sqrt((1 - z)/z) on 0 < z < 1, 0 elsewhere (the density of the checkerboard measure, mass 1/2)."""
    z = np.asarray(z, float)
    inside = (z > 0) & (z < 1)
    zz = np.where(inside, z, 0.5)
    return np.where(inside, np.sqrt((1.0 - zz) / zz) / np.pi, 0.0)


def checkerboard_cdf(z):
    """M(z) = (arcsin sqrt z + sqrt(z (1 - z)))/pi on [0, 1]; 0 for z <= 0 and 1/2 for z >= 1."""
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    return (np.arcsin(np.sqrt(z)) + np.sqrt(z * (1.0 - z))) / np.pi


def checkerboard_moments(kmax):
    """Exact moments m_k = C_k/(2 4^k), k = 0..kmax (C_k Catalan): 1/2, 1/8, 1/16, 5/128, ..."""
    return np.array([math.comb(2 * k, k) // (k + 1) / (2.0 * 4.0 ** k) for k in range(int(kmax) + 1)])


def checkerboard_jacobi(n):
    """(alpha, beta) of the normalized checkerboard measure: alpha_0 = 1/4, alpha_k = 1/2; beta (n - 1 entries) = 1/4."""
    alpha = np.full(int(n), 0.5)
    alpha[0] = 0.25
    return alpha, np.full(max(int(n) - 1, 0), 0.25)


def checkerboard_gauss(n):
    """n-point Gauss rule (z ascending, w) of the checkerboard measure (total mass 1/2); exact for 2n moments."""
    a, b = checkerboard_jacobi(n)
    return jacobi_measure(a, b, 0.5)
