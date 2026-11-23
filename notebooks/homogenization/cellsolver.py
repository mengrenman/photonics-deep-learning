"""Periodic and window cell problems for scalar conductivity (permittivity) in 2D.

Shared by notebooks 01 to 04 of this folder. Three discretizations of the same cell problem
"effective tensor of a two-phase periodic medium" are provided:

* Q1 finite elements on a pixel grid (``q1_*``). Conforming in H^1_per, so the primal value is a certified UPPER
  bound (Loewner order) of the effective tensor of the pixel medium, and the dual value is a certified LOWER bound.
* The sampled-coefficient spectral scheme of Moulinec and Suquet, solved by preconditioned CG (``spectral_sampled``).
  Not conforming: no sign of the error is guaranteed, nothing is certified.
* Fourier-Galerkin with EXACT Fourier coefficients of analytic shapes (``fg_*``). Conforming trigonometric trial
  spaces and an exactly evaluated quadratic form, so the primal value is a rigorous upper bound and the dual a
  rigorous lower bound for the TRUE geometry (no staircase).

Conventions
-----------
* Unit period, so lengths are in units of the period. The scalar field is a potential u with flux sigma grad u;
  "sigma" can be a conductivity or a permittivity (the TE static problem, with eps in the role of sigma).
* Pixel arrays ``a[i, j]``: ``i`` is the x index and ``j`` the y index (``indexing="ij"``). Pixel (i, j) is the
  square [i/N, (i+1)/N] x [j/N, (j+1)/N] with center ((i+1/2)/N, (j+1/2)/N). Node (i, j) sits at (i/N, j/N)
  and has id ``i*N + j`` (periodic cell: N^2 nodes, indices wrap; window: (N+1)^2 nodes, no wrap).
* Element (pixel) node order: (i, j), (i+1, j), (i+1, j+1), (i, j+1). The element matrix of a pixel with value a_e
  is a_e/6 * [[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]], independent of h in 2D. The integrals of the
  shape-function gradients over a pixel are h*g with g_x = (-1/2, 1/2, 1/2, -1/2) and g_y = (-1/2, -1/2, 1/2, 1/2).
* The effective tensor U (2x2, symmetric) is defined by U_ij = <a (e_i + grad chi_i) . (e_j + grad chi_j)>, the
  minimum of the Dirichlet energy over periodic correctors, with chi_i the nodal corrector of direction i. The
  correctors solve K chi_i = -f_i, with load f_i = sum_e a_e h g_i (the sum over the pixels sharing a node), and are
  returned with zero mean (the pinned node 0 only fixes the constant in the solve).
* The dual (flux, Thomson) principle is the primal problem of 1/a rotated: with R = [[0,-1],[1,0]],
  L = inv(R^T U[1/a] R). For a 2x2 symmetric U this equals U / det(U), so for diagonal U it is the componentwise
  reciprocal of the swapped diagonal entries.
* Time dependence plays no role here (static problem).

Public functions and their contracts
------------------------------------
``q1_stiffness(a)`` -> (K csr, conn). The periodic stiffness matrix (symmetric, rows sum to 0) and the element
    connectivity (N^2, 4).
``q1_effective(a, return_fields=False)`` -> U (2,2) [, chi (2,N,N)]. Guarantee: U is symmetric to rounding and
    U >= sigma*(pixel medium) in the Loewner order (conforming Q1 space); exact (to rounding) for laminates whose
    layers are aligned with the grid, because the exact potential is piecewise linear. Approximate: nothing else.
``q1_dual(a)`` -> L (2,2); ``q1_bracket(a)`` -> (L, U). Guarantee: L <= sigma*(pixel medium) <= U in the Loewner
    order, up to rounding (about 1e-12). The bracket is monotone under 2x2 pixel replication (nested spaces).
``q1_energy_density(a, chi)`` -> E (2,N,N): E[i, p, q] = dU_ii/da_(p,q), the energy of e_i + grad chi_i in each
    pixel (envelope theorem). Guarantee: sum_e a_e E[i, e] = U_ii to rounding.
``q1_center_field(chi, direction)`` -> (2,N,N): e_direction + grad chi at the pixel centers (the element-mean of
    the Q1 gradient).
``q1_window(a, bc)`` -> (2,2), bc in {"affine", "flux"}: apparent tensor of a non-periodic window of N x N pixels
    (area 1). "affine": corrector zero on the boundary nodes (upper bound of the periodic value of the same window).
    "flux": stream function zero on the boundary, inv(R^T U_affine[1/a] R) (lower bound). Guarantee:
    U_aff >= U_per >= L_per >= L_flux in the Loewner order.
``spectral_wavenumbers(N)`` -> (KX, KY): full-grid (fft order) angular wavenumbers with the Nyquist component set to
    zero (the symbol of the derivative used by ``spectral_sampled``).
``spectral_sampled(a, tol=1e-11, maxit=5000, directions=(0, 1))`` -> (S (2,2), iterations): Moulinec-Suquet scheme
    with spectral derivatives, PCG on the potential. Approximate and uncertified. S is the discrete energy of the
    converged iterate; the entries of directions not requested are NaN.
``fg_coefficients(shape, K2, **p)`` -> complex array (2*K2+1, 2*K2+1) with index
    [kx + K2, ky + K2]: the exact Fourier coefficients c_k = int sigma e^(-i k.x) dx of an analytic shape, k = 2 pi n.
``fg_bracket(shape, K, tol=1e-12, maxit=3000, direction=0, **p)`` -> (L, U, info): scalar bounds on the diagonal
    entry sigma*_dd of the TRUE geometry. Guarantee (rigorous, even before the CG has converged, since the energy of
    the iterate is evaluated exactly): L <= sigma*_dd <= U; U is non-increasing and L non-decreasing in K;
    U is exact for a laminate in the direction parallel to the layers (and L in the direction across them).
    The bracket closes only like 1/K for a smooth curved interface (the gradient jumps across it).
Generators: ``checkerboard``, ``disk_pixels``, ``square_pixels``, ``laminate_pixels``, ``random_checkerboard``,
    ``swap_phases``, ``gaussian_threshold`` (see their docstrings).

Typical timings (4 threads): q1_bracket N=64 about 0.1 s, N=256 about 1.5 s, N=512 about 12 s; spectral_sampled
N=256 about 3 s; fg_bracket K=64 about 1 s, K=128 about 5 s.
"""

import numpy as np
import scipy.fft as sfft
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import j1

# ----------------------------------------------------------------------------- Q1 element data

KE = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]], dtype=float) / 6.0
GX = np.array([-0.5, 0.5, 0.5, -0.5])
GY = np.array([-0.5, -0.5, 0.5, 0.5])
G = np.stack([GX, GY])  # G[d] = integral of d N / d x_d over a pixel, in units of h
ROT = np.array([[0.0, -1.0], [1.0, 0.0]])  # R of the plan: J = e' + R grad psi


def _check_pixels(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square 2-D array a[i, j]")
    if not np.all(a > 0):
        raise ValueError("pixel values must be positive")
    return a


def _conn(N, periodic):
    """Element connectivity (N*N, 4), nodes ordered (i,j), (i+1,j), (i+1,j+1), (i,j+1), element index i*N + j."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    if periodic:
        nn, ip, jp = N, (i + 1) % N, (j + 1) % N
    else:
        nn, ip, jp = N + 1, i + 1, j + 1
    c = np.stack([i * nn + j, ip * nn + j, ip * nn + jp, i * nn + jp], axis=-1)
    return c.reshape(-1, 4)


def _assemble(a, conn, nnodes):
    s = a.ravel()
    rows = np.repeat(conn, 4, axis=1).ravel()
    cols = np.tile(conn, (1, 4)).ravel()
    vals = (s[:, None, None] * KE[None]).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(nnodes, nnodes))


def _loads(a, conn, nnodes):
    """f[d, node] = sum_e a_e h G[d, local node]; shape (2, nnodes)."""
    N = a.shape[0]
    h = 1.0 / N
    s = a.ravel()
    return np.stack([np.bincount(conn.ravel(), weights=(s[:, None] * (h * G[d])[None, :]).ravel(), minlength=nnodes)
                     for d in range(2)])


def q1_stiffness(a):
    """Periodic Q1 stiffness matrix of the pixel field ``a``: returns (K csr (N^2, N^2), conn int (N^2, 4))."""
    a = _check_pixels(a)
    N = a.shape[0]
    c = _conn(N, True)
    return _assemble(a, c, N * N), c


def q1_effective(a, return_fields=False):
    """Q1 (conforming) upper bound U of the effective tensor of the periodic pixel medium ``a`` (N x N).

    Returns U (2,2) symmetric; with ``return_fields`` also chi (2, N, N), the nodal correctors (chi[d] is the
    corrector of the macroscopic field e_d, zero mean, so the field is e_d + grad chi[d]). Guarantee: U >= sigma*
    in the Loewner order; exact for grid-aligned laminates. One sparse LU (node 0 pinned) serves both loads.
    """
    a = _check_pixels(a)
    N = a.shape[0]
    K, c = q1_stiffness(a)
    f = _loads(a, c, N * N)
    lu = spla.splu(K[1:, 1:].tocsc(), permc_spec="COLAMD")
    chi = np.zeros((2, N * N))
    chi[:, 1:] = lu.solve(-f[:, 1:].T).T
    U = a.mean() * np.eye(2) + chi @ f.T  # U_ij = <a> delta_ij + f_j . chi_i
    U = 0.5 * (U + U.T)
    if not return_fields:
        return U
    chi -= chi.mean(axis=1, keepdims=True)
    return U, chi.reshape(2, N, N)


def q1_dual(a):
    """Lower bound L = inv(R^T U[1/a] R) of the effective tensor (flux principle), with R = [[0,-1],[1,0]]."""
    a = _check_pixels(a)
    U = q1_effective(1.0 / a)
    return np.linalg.inv(ROT.T @ U @ ROT)


def q1_bracket(a):
    """(L, U): certified Loewner bracket L <= sigma*(pixel medium) <= U, both (2,2)."""
    a = _check_pixels(a)
    return q1_dual(a), q1_effective(a)


def q1_energy_density(a, chi):
    """E (2,N,N): dU_ii/da_e = h^2 + 2h g.chi_e + chi_e^T KE chi_e per pixel e, for chi from q1_effective.

    E[d] is the energy density (per pixel, area h^2 included) of the field e_d + grad chi[d]; sum_e a_e E[d, e] = U_dd.
    By the envelope theorem E[d] is the exact gradient of U_dd with respect to the pixel values.
    """
    a = _check_pixels(a)
    N = a.shape[0]
    h = 1.0 / N
    c = _conn(N, True)
    ch = np.asarray(chi, dtype=float).reshape(2, N * N)
    E = np.empty((2, N * N))
    for d in range(2):
        ce = ch[d][c]  # (N^2, 4)
        E[d] = h * h + 2.0 * h * (ce @ G[d]) + np.einsum("ei,ij,ej->e", ce, KE, ce)
    return E.reshape(2, N, N)


def q1_center_field(chi, direction):
    """e_direction + grad chi at the pixel centers, shape (2, N, N); ``chi`` is the (2,N,N) output of q1_effective."""
    ch = np.asarray(chi, dtype=float)
    N = ch.shape[-1]
    h = 1.0 / N
    ce = ch[direction].reshape(N * N)[_conn(N, True)]  # (N^2, 4)
    out = np.stack([ce @ GX, ce @ GY], axis=0) / h  # mean gradient over the pixel = (h g . chi) / h^2
    out[direction] += 1.0
    return out.reshape(2, N, N)


def _window_affine(a):
    a = _check_pixels(a)
    N = a.shape[0]
    nn = (N + 1) ** 2
    c = _conn(N, False)
    K = _assemble(a, c, nn)
    f = _loads(a, c, nn)
    ii, jj = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    inner = ((ii > 0) & (ii < N) & (jj > 0) & (jj < N)).ravel()
    lu = spla.splu(K[inner][:, inner].tocsc(), permc_spec="COLAMD")
    chi = np.zeros((2, nn))
    chi[:, inner] = lu.solve(-f[:, inner].T).T
    U = a.mean() * np.eye(2) + chi @ f.T
    return 0.5 * (U + U.T)


def q1_window(a, bc="affine"):
    """Apparent tensor (2,2) of a window of N x N pixels (area 1, nodes (N+1)^2, no periodicity).

    bc="affine": fluctuation zero on the boundary nodes (upper bound of the periodic tensor of the same window).
    bc="flux": stream function zero on the boundary, inv(R^T U_affine[1/a] R) (a lower bound of the periodic
    dual). Together: U_aff >= U_per >= L_per >= L_flux (Loewner). Flux bounds are inverses of 2x2 tensors,
    not componentwise reciprocals.
    """
    if bc == "affine":
        return _window_affine(a)
    if bc == "flux":
        a = _check_pixels(a)
        return np.linalg.inv(ROT.T @ _window_affine(1.0 / a) @ ROT)
    raise ValueError("bc must be 'affine' or 'flux'")


# ----------------------------------------------------------------------------- sampled spectral (Moulinec-Suquet)

def spectral_wavenumbers(N):
    """(KX, KY): angular wavenumbers 2 pi n on the full N x N fft grid (axis 0 is x), Nyquist set to zero."""
    k = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi
    if N % 2 == 0:
        k[N // 2] = 0.0
    return np.meshgrid(k, k, indexing="ij")


def spectral_sampled(a, tol=1e-11, maxit=5000, directions=(0, 1)):
    """Moulinec-Suquet (sampled coefficient) effective tensor by PCG: returns (S (2,2), iterations).

    Minimizes the sampled energy mean(a |e + D phi|^2) over real grid functions phi, where D is the spectral
    gradient with the Nyquist component of k set to zero. Preconditioner: (-mean(a) Laplacian)^-1 on the modes with
    nonzero symbol (zero-symbol modes excluded). Stops when |r| <= tol |b|. ``iterations`` is the larger count of the
    requested directions. S is the energy of the converged iterate (symmetrized); entries of directions not
    requested are NaN. No sign of S - sigma* is guaranteed (not conforming), nothing is certified.
    """
    a = _check_pixels(a)
    N = a.shape[0]
    kx = np.fft.fftfreq(N, d=1.0 / N) * 2.0 * np.pi
    ky = np.fft.rfftfreq(N, d=1.0 / N) * 2.0 * np.pi
    if N % 2 == 0:
        kx[N // 2] = 0.0
        ky[-1] = 0.0
    KX, KY = kx[:, None], ky[None, :]
    K2 = KX ** 2 + KY ** 2
    zero = K2 == 0.0
    inv_k2 = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, K2)) / a.mean()
    kw = {"s": (N, N), "workers": 4}
    fw = {"workers": 4}

    def grad(phi):
        P = sfft.rfft2(phi, **fw)
        return sfft.irfft2(1j * KX * P, **kw), sfft.irfft2(1j * KY * P, **kw)

    def div(gx, gy):
        return sfft.irfft2(1j * KX * sfft.rfft2(gx, **fw) + 1j * KY * sfft.rfft2(gy, **fw), **kw)

    def A(phi):
        gx, gy = grad(phi)
        return -div(a * gx, a * gy)

    S = np.full((2, 2), np.nan)
    fields = {}
    its = 0
    for d in directions:
        b = div(a if d == 0 else 0.0 * a, a if d == 1 else 0.0 * a)
        x = np.zeros_like(a)
        r = b.copy()
        z = sfft.irfft2(inv_k2 * sfft.rfft2(r, **fw), **kw)
        p = z.copy()
        rz = np.sum(r * z)
        nb = np.linalg.norm(b)
        it = 0
        if nb <= 1e-13 * np.linalg.norm(a):  # no driving force (e.g. field parallel to the layers of a laminate): phi = 0
            maxit_d = 0
        else:
            maxit_d = maxit
        for it in range(1, maxit_d + 1):
            Ap = A(p)
            al = rz / np.sum(p * Ap)
            x += al * p
            r -= al * Ap
            if np.linalg.norm(r) <= tol * nb:
                break
            z = sfft.irfft2(inv_k2 * sfft.rfft2(r, **fw), **kw)
            rzn = np.sum(r * z)
            p = z + (rzn / rz) * p
            rz = rzn
        its = max(its, it)
        gx, gy = grad(x)
        E = np.stack([gx, gy])
        E[d] += 1.0
        fields[d] = E
    for i in fields:
        for j in fields:
            S[i, j] = np.mean(a * (fields[i][0] * fields[j][0] + fields[i][1] * fields[j][1]))
    if len(fields) == 2:
        S = 0.5 * (S + S.T)
    return S, its


# ----------------------------------------------------------------------------- Fourier-Galerkin, exact coefficients

def fg_coefficients(shape, K2, **p):
    """Exact Fourier coefficients c[kx + K2, ky + K2] = int_cell sigma e^(-i 2 pi (kx x + ky y)) dx dy, |k| <= K2.

    shape (parameters, all in units of the period):
      "disk": R, s_in, s_out, center=(.5, .5): s_out delta_k0 + (s_in - s_out) f 2 J1(|g|R)/(|g|R) e^(-i g.c),
              g = 2 pi k, f = pi R^2 (R < 1/2 for a non-overlapping disk; the formula is for the periodic copy).
      "square": w, s_in, s_out, center=(.5, .5): (s_in - s_out) w^2 sinc(g_x w/2) sinc(g_y w/2) e^(-i g.c) + s_out.
      "checkerboard": s_a, s_b: tiles of side 1/2, s_a on [0,1/2)^2 and [1/2,1)^2, s_b elsewhere.
      "laminate": f, s_in, s_out, axis=0: stripe of width f (value s_in, centered at 1/2) normal to the axis.
    Returns a complex (2 K2 + 1, 2 K2 + 1) array, Hermitian in k (c[-k] = conj(c[k])).
    """
    K2 = int(K2)
    m = np.arange(-K2, K2 + 1)
    MX, MY = np.meshgrid(m, m, indexing="ij")
    gx, gy = 2.0 * np.pi * MX, 2.0 * np.pi * MY
    if shape == "disk":
        R, s_in, s_out = p["R"], p["s_in"], p["s_out"]
        cx, cy = p.get("center", (0.5, 0.5))
        g = np.hypot(gx, gy)
        gs = np.where(g == 0, 1.0, g * R)
        form = np.where(g == 0, 1.0, 2.0 * j1(gs) / gs)
        c = (s_in - s_out) * np.pi * R ** 2 * form * np.exp(-1j * (gx * cx + gy * cy))
        c = c.astype(complex)
        c[K2, K2] += s_out
    elif shape == "square":
        w, s_in, s_out = p["w"], p["s_in"], p["s_out"]
        cx, cy = p.get("center", (0.5, 0.5))
        c = ((s_in - s_out) * w ** 2 * np.sinc(gx * w / (2 * np.pi)) * np.sinc(gy * w / (2 * np.pi))
             * np.exp(-1j * (gx * cx + gy * cy))).astype(complex)
        c[K2, K2] += s_out
    elif shape == "checkerboard":
        s_a, s_b = p["s_a"], p["s_b"]
        # square wave +1 on [0,1/2), -1 on [1/2,1): coefficients -2i/(pi n) for odd n, 0 otherwise
        sq = np.where(m % 2 != 0, -2j / (np.pi * np.where(m == 0, 1, m)), 0.0)
        c = 0.5 * (s_a - s_b) * np.outer(sq, sq)
        c[K2, K2] += 0.5 * (s_a + s_b)
    elif shape == "laminate":
        f, s_in, s_out, axis = p["f"], p["s_in"], p["s_out"], p.get("axis", 0)
        n = m.astype(float)
        line = (s_in - s_out) * f * np.sinc(n * f) * np.exp(-1j * np.pi * n)
        c = np.zeros((2 * K2 + 1, 2 * K2 + 1), dtype=complex)
        if axis == 0:
            c[:, K2] = line
        else:
            c[K2, :] = line
        c[K2, K2] += s_out
    else:
        raise ValueError("unknown shape " + repr(shape))
    return c


def _reciprocal(shape, p):
    q = dict(p)
    for key in ("s_in", "s_out", "s_a", "s_b"):
        if key in q:
            q[key] = 1.0 / q[key]
    return q


def _fg_upper(c, K, direction, tol, maxit):
    """Galerkin minimum of <sigma |e_d + grad phi|^2> over real trig polys of degree K per coordinate.

    ``c`` holds the (4K+1)^2 coefficients of sigma (degree <= 2K). Returns (energy of the final iterate, iterations,
    relative residual). The energy is evaluated exactly on an alias-free M = 4K+2 grid with the true sigma
    coefficients up to degree 2K (the quadratic form needs no more), so it is an upper bound for ANY iterate.
    """
    K2, M = 2 * K, 4 * K + 2
    S = np.zeros((M, M), dtype=complex)
    m = np.arange(-K2, K2 + 1)
    S[np.ix_(m % M, m % M)] = c
    sig = sfft.ifft2(S).real * (M * M)
    nx = np.fft.fftfreq(M, d=1.0 / M)
    ny = np.fft.rfftfreq(M, d=1.0 / M)
    mask = (np.abs(nx)[:, None] <= K) & (ny[None, :] <= K)
    KX, KY = 2 * np.pi * nx[:, None], 2 * np.pi * ny[None, :]
    k2 = KX ** 2 + KY ** 2
    sbar = float(c[K2, K2].real)
    inv = np.where(mask & (k2 > 0), 1.0 / (sbar * np.where(k2 > 0, k2, 1.0)), 0.0)
    kw = {"s": (M, M), "workers": 4}
    fw = {"workers": 4}

    def grad(phi):
        P = sfft.rfft2(phi, **fw) * mask
        return sfft.irfft2(1j * KX * P, **kw), sfft.irfft2(1j * KY * P, **kw)

    def div(fx, fy):
        return sfft.irfft2(mask * (1j * KX * sfft.rfft2(fx, **fw) + 1j * KY * sfft.rfft2(fy, **fw)), **kw)

    def A(phi):
        gx, gy = grad(phi)
        return -div(sig * gx, sig * gy)

    ones = np.ones((M, M))
    b = div(sig * ones if direction == 0 else 0 * sig, sig * ones if direction == 1 else 0 * sig)
    x = np.zeros((M, M))
    r = b.copy()
    z = sfft.irfft2(inv * sfft.rfft2(r, **fw), **kw)
    pdir = z.copy()
    rz = np.sum(r * z)
    nb = np.linalg.norm(b)
    it, res = 0, 0.0
    if nb <= 1e-13 * np.linalg.norm(sig):  # no driving force (field parallel to the layers of a laminate): phi = 0
        maxit = 0
    for it in range(1, maxit + 1):
        Ap = A(pdir)
        al = rz / np.sum(pdir * Ap)
        x += al * pdir
        r -= al * Ap
        res = np.linalg.norm(r) / nb
        if res <= tol:
            break
        z = sfft.irfft2(inv * sfft.rfft2(r, **fw), **kw)
        rzn = np.sum(r * z)
        pdir = z + (rzn / rz) * pdir
        rz = rzn
    gx, gy = grad(x)
    gx = gx + (1.0 if direction == 0 else 0.0)
    gy = gy + (1.0 if direction == 1 else 0.0)
    return float(np.mean(sig * (gx ** 2 + gy ** 2))), it, float(res)


def fg_bracket(shape, K, tol=1e-12, maxit=3000, direction=0, **p):
    """Rigorous bounds (L, U, info) on sigma*_dd of the true analytic geometry, trig trial fields of degree K.

    U: primal Galerkin value with the exact coefficients of sigma (direction d). L: reciprocal of the primal value of
    the dual problem, which uses the exact coefficients of 1/sigma (same shape, reciprocal phase values) in the
    swapped direction 1-d; this equals the flux principle for shapes with a diagonal effective tensor (all four
    built-in shapes, since translation does not change the tensor). Guarantee: L <= sigma*_dd <= U, monotone in K,
    even when the CG is not converged (the energy of the iterate is exact); ``info["converged"]`` records whether
    both residuals reached ``tol``. ``info`` has keys iterations (primal, dual), residuals, converged, M.
    """
    K = int(K)
    cU = fg_coefficients(shape, 2 * K, **p)
    cL = fg_coefficients(shape, 2 * K, **_reciprocal(shape, p))
    U, itU, rU = _fg_upper(cU, K, direction, tol, maxit)
    Ud, itL, rL = _fg_upper(cL, K, 1 - direction, tol, maxit)
    info = {"iterations": (itU, itL), "residuals": (rU, rL), "converged": bool(rU <= tol and rL <= tol), "M": 4 * K + 2}
    return 1.0 / Ud, U, info


# ----------------------------------------------------------------------------- geometry generators

def checkerboard(N, a, b):
    """2x2-tile checkerboard of N x N pixels (N even): ``a`` on the tiles [0,1/2)^2 and [1/2,1)^2, ``b`` elsewhere."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.where((i < N // 2) ^ (j < N // 2), float(b), float(a))


def _pixel_centers(N):
    x = (np.arange(N) + 0.5) / N
    return np.meshgrid(x, x, indexing="ij")


def disk_pixels(N, R, a_in, a_out, center=(0.5, 0.5)):
    """Center-sampled pixel disk: a_in where the pixel center is within R of ``center`` (periodic minimal image)."""
    X, Y = _pixel_centers(N)
    dx = (X - center[0] + 0.5) % 1.0 - 0.5
    dy = (Y - center[1] + 0.5) % 1.0 - 0.5
    return np.where(dx ** 2 + dy ** 2 < R ** 2, float(a_in), float(a_out))


def square_pixels(N, side_px, a_in, a_out):
    """Centered pixel-aligned square of ``side_px`` pixels per side (start index (N - side_px)//2)."""
    out = np.full((N, N), float(a_out))
    s0 = (N - side_px) // 2
    out[s0:s0 + side_px, s0:s0 + side_px] = a_in
    return out


def laminate_pixels(N, n_in, a_in, a_out, axis=0):
    """Laminate with a centered stripe of ``n_in`` pixels of value a_in; ``axis`` is the layer normal (0: x, 1: y)."""
    line = np.full(N, float(a_out))
    s0 = (N - n_in) // 2
    line[s0:s0 + n_in] = a_in
    return np.repeat(line[:, None], N, axis=1) if axis == 0 else np.repeat(line[None, :], N, axis=0)


def random_checkerboard(L, m, p, a, b, rng):
    """L x L iid tiles of m x m pixels; each tile is ``b`` with probability p and ``a`` otherwise.

    ``rng`` is a seed or a numpy Generator. Returns an (L m, L m) array. The tile draw is
    ``rng.random((L, L)) < p`` so seeds reproduce.
    """
    rng = np.random.default_rng(rng)
    tiles = rng.random((L, L)) < p
    return np.where(np.kron(tiles, np.ones((m, m), dtype=bool)), float(b), float(a))


def swap_phases(s):
    """Phase-swapped field of a two-valued field: min + max - s (equals a b / s). Same law as s when p = 1/2."""
    s = np.asarray(s, dtype=float)
    return s.min() + s.max() - s


def gaussian_threshold(N, ell, theta, rng, values):
    """Thresholded Gaussian random field on a periodic N x N grid.

    White noise is filtered in Fourier space by exp(-(k ell)^2/2) (k in radians per pixel, correlation length ell in
    pixels), then the round(theta N^2) pixels with the LOWEST filtered value get values[0] and the rest values[1].
    So ``theta`` is the fraction of values[0] up to rounding to a whole pixel. ``rng`` is a seed or a Generator.
    """
    rng = np.random.default_rng(rng)
    k = np.fft.fftfreq(N) * 2.0 * np.pi
    KX, KY = np.meshgrid(k, k, indexing="ij")
    field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * np.exp(-0.5 * ell ** 2 * (KX ** 2 + KY ** 2))).real
    n0 = int(round(theta * N * N))
    order = np.argsort(field.ravel(), kind="stable")
    out = np.full(N * N, float(values[1]))
    out[order[:n0]] = values[0]
    return out.reshape(N, N)
