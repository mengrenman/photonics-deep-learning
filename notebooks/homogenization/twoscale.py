"""Finite-domain two-scale computations for -div(a(x/eps) grad u) = f on the unit square (notebook 04).

Everything here is static and scalar. The cell is a pixel array ``cell[i, j]`` (``i`` the x index, as in
``cellsolver``), Q1 finite elements throughout, so every comparison is between discrete objects built from the SAME
pixel cell (the staircase is part of the problem, not a source of error). Lengths are in units of the domain
(the unit square) for the fine problems and in units of the period for the cell and strip problems.

Conventions
-----------
* Cell: m x m pixels, period 1, node (i, j) at (i/m, j/m), node id ``i*m + j`` (cellsolver conventions). The cell
  correctors ``chi[d]`` live on the m x m periodic nodes with zero mean.
* Fine medium: N cells per side, n = N*m pixels per side, ``(n+1) x (n+1)`` nodes ``u[i, j]`` at (i/n, j/n). The
  cell pattern can be SHIFTED by s pixels: pixel i of the fine medium carries cell pixel ``(i - s) mod m`` and node
  i the corrector node ``(i - s) mod m``. A shift of s = m/4 pixels cuts the domain boundary at 3/4 of a cell.
* Boundary strips (``strip_tail``): pixels ``(depth*m, m)``, x index along the depth, periodic in y (m nodes),
  Dirichlet data on node column 0, natural (Neumann) condition at the far end.
* Edges are ordered ('left' x=0, 'right' x=1, 'bottom' y=0, 'top' y=1), index 0..3 in ``EDGES``.
* ``eps = 1/N``. The first-order two-scale expansion is w = u0 + eps * sum_d chi_d(x/eps - s) d_d u0, with u0 the
  homogenized solution.

Public functions and their contracts
------------------------------------
``cell_correctors(cell)`` -> (chi (2,m,m), sigma (2,2)). Q1 periodic correctors (zero mean) and effective tensor;
    identical to ``cellsolver.q1_effective(cell, return_fields=True)`` (this is the same call). sigma is a
    certified upper bound (Loewner) of the effective tensor of the pixel medium.
``corrector_gram(chi)`` -> M (2,2), M_ij = <(e_i + grad chi_i) . (e_j + grad chi_j)> over the cell, from the EXACT
    Q1 element integrals with unit coefficient (symmetric positive definite). This is the weight of the
    two-scale limit of the plain gradient energy.
``periodic_medium(cell, N, shift=0)`` -> a (n,n); ``periodic_nodal(field, N, shift=0)`` -> (n+1, n+1): tile the
    cell pixels or the cell nodal field N times and shift the pattern by ``shift`` pixels (int, or a pair for
    x and y). Exact copies, no interpolation.
``fine_dirichlet(a, rhs=1.0)`` -> u (n+1, n+1): Q1 solution of -div(a grad u) = rhs on [0,1]^2, u = 0 on the
    boundary. ``rhs`` is a constant or an (n,n) pixel array; the load of a node is the integral of rhs times the
    hat function (h^2 rhs for a constant at an interior node). Guarantee: the Galerkin solution (energy-norm
    best approximation in the Q1 space); nodal values are not bounded above or below.
``torsion(x, y, nterms=301)`` -> (u, u_x, u_y), arrays of shape (len(x), len(y)): the double sine series of
    -Lap u = 1 on the unit square with u = 0 on the boundary, odd p, q <= nterms, coefficient
    16/(pi^4 p q (p^2+q^2)). Exact up to truncation: the sup error of u is about 3 / nterms^3 (1.2e-7 at
    301, 1e-8 at 601; the plan's 1e-10 for 301 against 601 does NOT hold), the gradients converge more slowly
    (about nterms^-2). The center value 0.0736713533 is reproduced to 5e-9 by 601 terms.
``harmonic_dirichlet(g)`` -> u (n+1, n+1): discrete (Q1, unit coefficient) harmonic function with the boundary
    values of ``g`` (only the boundary ring of ``g`` is read). Solved by a fast sine transform; same operator
    as an assembled Q1 Laplacian (checked to rounding).
``open_stiffness(a, periodic_y=False)`` -> (K csr, nodes_per_row): Q1 stiffness on a rectangle of pixels
    a (nx, ny), nodes (nx+1) x (ny+1), or (nx+1) x ny with wrap in y; node id i*nyn + j.
``strip_tail(strip_a, data)`` -> (U, tail): solve the strip problem; ``data`` is (m,) or (k, m) (k right-hand sides
    share one factorization); U is (nx+1, m) (or (k, nx+1, m)); tail is the mean of the last node column
    (scalar or (k,)). Approximate: the strip is finite, so the tail differs from the semi-infinite value by
    about e^(-depth * kappa_1), kappa_1 about 6.8 at the notebook's cell, below 1e-17 at depth 6.
``edge_tails(a, chi_nodal, m, depth=6, offset=0, return_strips=False)`` -> tails (2,4) [corrector j, edge]
    (+ dict of strip solutions). Cuts the strips next to the four edges of the fine medium ``a`` (left: ``a[:depth*m,
    j0:j0+m]``; right: reversed in x; bottom and top: transposed) with Dirichlet data -chi_j taken from the
    fine nodal correctors ``chi_nodal`` (2, n+1, n+1) (from ``periodic_nodal`` per component) and returns the
    far-column means. Exactly zero (to rounding) wherever the correctors are odd on the cut.
``edge_data(tails, ux, uy)`` -> g (n+1, n+1): boundary values sum_j tail_j(edge) d_j u0 on each edge (the
    tail of corrector j is multiplied by d_j u0 evaluated on that edge); ``ux, uy`` are nodal gradients of u0.
    Corners: bottom and top edges are written last (they are zero there for the torsion field anyway).
``strip_decay_rates(U, m, tail)`` -> array of ln(r_k / r_{k+1}), r_k = L2 norm over the node column at depth k
    cells of (U - tail); k = 0, 1, ... up to where r_k is above 1e-13 times r_0. Rates between depths 1 and 2 cells
    are entries [1]. Approximate (finite strip, a mixture of modes of the excited parity at small depth).
``evanescent_exponents(cell, nkeep=8, method="reduced", return_details=False)`` -> (kappa, parity): the static
    evanescent Bloch exponents of div(a grad u) = 0 on one Q1 cell (open in x, periodic in y), u = e^(-kappa x)
    p(x, y), lambda = e^(-kappa) the Bloch factor of the QEP lambda^2 A1 + lambda A0 + A_-1 = 0 (A0 = K_aa +
    S^T K_bb S, A1 = K_ab S, A_-1 = S^T K_ba, S picks node column 0). Returns the ``nkeep`` exponents with
    Re kappa > 1e-6 and |lambda| < 1 - 1e-9, ascending in Re kappa, and their parity under y -> 1 - y
    (+1 even, -1 odd, 0 if not definite within 1e-6; degenerate pairs are rotated to definite parity; always 0 and meaningless for cells
    that are not mirror symmetric in y). The exponents come in pairs +-kappa (palindromic QEP); only the decaying
    half is returned. "reduced" eliminates the interior node columns exactly (an m x m QEP, milliseconds);
    "full" is the dense 2n x 2n linearization with n = m^2 (same eigenvalues to rounding, used as the
    cross-check). Exponents with Re kappa >= ``kmax`` (30) are dropped: the Bloch factor e^-kappa is then below the
    rounding floor of the eigensolver. Accuracy: the decaying modes are computed as the LARGE eigenvalues e^kappa
    of the reversed pencil, good to about 1e-12 at kappa < 15, 1e-9 near 20 and 1e-7 near 23; the two methods
    agree to those levels. ``return_details`` adds a dict with the eigenvector (node column 0), the relative QEP residual
    and the parity expectation value.
``interior_box(lo=0.25, hi=0.75)`` is just the tuple used as ``region`` below.
``norms(e, region=None)`` -> (Linf, L2, H1): for a nodal field e on the unit square (n+1, n+1). Linf is the max over
    nodes in the box ``region = (lo, hi)`` (all nodes by default); L2 is the exact L2 norm of the Q1 interpolant
    and H1 the exact H1 seminorm (unit stiffness), both over the elements whose centers are inside the box.
``first_order_expansion(u0, ux, uy, chi_nodal, N)`` -> w: u0 + eps (chi_1 u0x + chi_2 u0y) at the nodes.
``element_gradients(u)`` -> (gx, gy) (n,n): the mean gradient of the Q1 field over each pixel.
``weak_integrals(u, a, phi, exact_energy=False)`` -> (I_flux, I_grad): sum over pixels of phi a |g|^2 h^2 and
    phi |g|^2 h^2, phi an (n,n) array at the pixel centers. Default g = element-mean gradient (midpoint rule, as in
    the plan); its fixed-cell limits are ``midpoint_tensors``, NOT sigma and M (an O(h_cell^2) offset, about 1%
    at m = 16, that makes the error against the exact-tensor limit stall near 5e-5 at N = 32). With
    ``exact_energy=True`` the exact element energy is used and the errors against the limits built from sigma and
    ``corrector_gram`` fall at rate 2 (as eps^2) at s = 0.
``midpoint_tensors(cell, chi)`` -> (S, M) (2,2): the cell tensors that the default (midpoint) ``weak_integrals``
    converge to, from the element-mean fields.
``weak_limits(phi, ux, uy, sigma, M)`` -> (lim_flux, lim_grad, naive): sum phi sigma_h grad u0 . grad u0 h^2 with
    ``sigma`` the effective tensor (2,2), sum phi grad u0 . M grad u0 h^2, and sum phi |grad u0|^2 h^2 with u0
    gradients at the pixel centers (n,n). The div-curl lemma gives the first, two-scale convergence the second.

All sparse systems are solved by SuperLU; the heavy fine solves (n = 512, 262k unknowns) take a few seconds.
"""
import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import cellsolver as cs

KE = cs.KE
G = cs.G
EDGES = ("left", "right", "bottom", "top")
# exact Q1 element mass matrix divided by h^2
ME = np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=float) / 36.0


def _shift_pair(shift):
    s = np.broadcast_to(np.asarray(shift, dtype=int), (2,))
    return int(s[0]), int(s[1])


# ----------------------------------------------------------------------------- cell problem


def cell_correctors(cell):
    """(chi (2,m,m), sigma (2,2)): Q1 periodic correctors (zero mean) and effective tensor (same as cellsolver)."""
    U, chi = cs.q1_effective(cell, return_fields=True)
    return chi, U


def corrector_gram(chi):
    """M (2,2): <(e_i + grad chi_i) . (e_j + grad chi_j)> from exact Q1 element integrals (unit coefficient)."""
    chi = np.asarray(chi, dtype=float)
    m = chi.shape[-1]
    h = 1.0 / m
    c = cs._conn(m, True)
    ce = [chi[d].reshape(-1)[c] for d in range(2)]  # (m^2, 4)
    M = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            e = (h * h) * (i == j) + h * (ce[i] @ G[j] + ce[j] @ G[i]) + np.einsum("ek,kl,el->e", ce[i], KE, ce[j])
            M[i, j] = e.sum()
    return M


def periodic_medium(cell, N, shift=0):
    """Fine pixel medium a (n,n), n = N m: pixel i carries cell pixel (i - s) mod m (shift s in pixels)."""
    cell = np.asarray(cell)
    m = cell.shape[0]
    n = N * m
    sx, sy = _shift_pair(shift)
    return cell[np.ix_((np.arange(n) - sx) % m, (np.arange(n) - sy) % m)]


def periodic_nodal(field, N, shift=0):
    """Nodal field (n+1, n+1) from a periodic cell nodal field (m,m): node i carries (i - s) mod m."""
    field = np.asarray(field)
    m = field.shape[0]
    n = N * m
    sx, sy = _shift_pair(shift)
    return field[np.ix_((np.arange(n + 1) - sx) % m, (np.arange(n + 1) - sy) % m)]


# ----------------------------------------------------------------------------- fine problems


def open_stiffness(a, periodic_y=False):
    """Q1 stiffness (csr) of pixels a (nx, ny); nodes (nx+1) x (ny+1), or (nx+1) x ny when periodic in y.

    Returns (K, nyn) with nyn the number of nodes per row; node id i*nyn + j.
    """
    a = np.asarray(a, dtype=float)
    nx, ny = a.shape
    nyn = ny if periodic_y else ny + 1
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    jp = (j + 1) % nyn if periodic_y else j + 1
    jj = j % nyn if periodic_y else j
    c = np.stack([i * nyn + jj, (i + 1) * nyn + jj, (i + 1) * nyn + jp, i * nyn + jp], axis=-1).reshape(-1, 4)
    rows = np.repeat(c, 4, axis=1).ravel()
    cols = np.tile(c, (1, 4)).ravel()
    vals = (a.ravel()[:, None, None] * KE[None]).ravel()
    nn = (nx + 1) * nyn
    return sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn)), nyn


def _boundary_mask(n):
    I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    return ((I == 0) | (I == n) | (J == 0) | (J == n)).ravel()


def fine_dirichlet(a, rhs=1.0):
    """Q1 solution u (n+1, n+1) of -div(a grad u) = rhs on [0,1]^2 with u = 0 on the boundary (pixels a (n,n))."""
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    if a.ndim != 2 or a.shape[1] != n or not np.all(a > 0):
        raise ValueError("a must be a square positive pixel array")
    h = 1.0 / n
    K, _ = open_stiffness(a)
    r = np.broadcast_to(np.asarray(rhs, dtype=float), (n, n))
    c = cs._conn(n, False)
    f = np.bincount(c.ravel(), weights=np.repeat(r.ravel() * h * h / 4.0, 4), minlength=(n + 1) ** 2)
    interior = ~_boundary_mask(n)
    u = np.zeros((n + 1) ** 2)
    Kii = K[interior][:, interior].tocsc()
    u[interior] = spla.splu(Kii, permc_spec="COLAMD").solve(f[interior])
    return u.reshape(n + 1, n + 1)


def harmonic_dirichlet(g):
    """Discrete (Q1, unit coefficient) harmonic function with the boundary values of g ((n+1) x (n+1))."""
    g = np.asarray(g, dtype=float)
    n = g.shape[0] - 1
    if g.shape != (n + 1, n + 1):
        raise ValueError("g must be square")
    h = 1.0 / n
    u = np.zeros_like(g)
    u[0, :], u[n, :], u[:, 0], u[:, n] = g[0, :], g[n, :], g[:, 0], g[:, n]
    # Q1 unit stiffness = Kx (x) Mz + Mx (x) Kz with the 1D stiffness/mass; residual of the lifted boundary data
    def k1(v, axis):
        out = 2.0 * v
        sl = [slice(None)] * 2
        a1, a2 = list(sl), list(sl)
        a1[axis], a2[axis] = slice(1, None), slice(None, -1)
        out[tuple(a1)] -= v[tuple(a2)]
        out[tuple(a2)] -= v[tuple(a1)]
        return out / h

    def m1(v, axis):
        out = 4.0 * v
        sl = [slice(None)] * 2
        a1, a2 = list(sl), list(sl)
        a1[axis], a2[axis] = slice(1, None), slice(None, -1)
        out[tuple(a1)] += v[tuple(a2)]
        out[tuple(a2)] += v[tuple(a1)]
        return out * h / 6.0

    r = -(m1(k1(u, 0), 1) + k1(m1(u, 0), 1))[1:-1, 1:-1]
    # sine eigen-decomposition on the interior: K1 -> (2 - 2 cos t)/h, M1 -> h (4 + 2 cos t)/6
    t = np.pi * np.arange(1, n) / n
    lk, lm = (2.0 - 2.0 * np.cos(t)) / h, h * (4.0 + 2.0 * np.cos(t)) / 6.0
    lam = lk[:, None] * lm[None, :] + lm[:, None] * lk[None, :]
    rh = sfft.dstn(r, type=1, workers=4)
    sol = sfft.idstn(rh / lam, type=1, workers=4)
    u[1:-1, 1:-1] = sol
    return u


def torsion(x, y, nterms=301):
    """(u, u_x, u_y) of -Lap u = 1 on the unit square, u = 0 on the boundary, on the grid x (outer) y."""
    p = np.arange(1, nterms + 1, 2)
    P, Q = np.meshgrid(p, p, indexing="ij")
    coef = 16.0 / (np.pi ** 4 * P * Q * (P ** 2 + Q ** 2))
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    sx, cx = np.sin(np.pi * np.outer(x, p)), np.cos(np.pi * np.outer(x, p)) * np.pi * p
    sy, cy = np.sin(np.pi * np.outer(y, p)), np.cos(np.pi * np.outer(y, p)) * np.pi * p
    return sx @ coef @ sy.T, cx @ coef @ sy.T, sx @ coef @ cy.T


# ----------------------------------------------------------------------------- boundary layers


def strip_tail(strip_a, data):
    """Strip problem. ``strip_a`` (L*m, m) pixels, periodic in y, Dirichlet ``data`` ((m,) or (k, m)) on node
    column 0, natural BC at the far end. Returns (U (nx+1, m) [or (k, nx+1, m)], tail = mean of last column)."""
    strip_a = np.asarray(strip_a, dtype=float)
    data = np.asarray(data, dtype=float)
    single = data.ndim == 1
    data = np.atleast_2d(data)
    K, nyn = open_stiffness(strip_a, periodic_y=True)
    nx = strip_a.shape[0]
    if data.shape[1] != nyn:
        raise ValueError("data must have one value per node of the first column")
    free = np.ones((nx + 1) * nyn, bool)
    free[:nyn] = False
    Kff = K[free][:, free].tocsc()
    rhs = -(K[free][:, ~free] @ data.T)  # (nfree, k)
    sol = spla.splu(Kff).solve(np.asarray(rhs))
    U = np.zeros((data.shape[0], (nx + 1) * nyn))
    U[:, ~free] = data
    U[:, free] = sol.T
    U = U.reshape(data.shape[0], nx + 1, nyn)
    tail = U[:, -1, :].mean(axis=1)
    return (U[0], tail[0]) if single else (U, tail)


def edge_tails(a, chi_nodal, m, depth=6, offset=0, return_strips=False):
    """Tails (2,4) [corrector j, edge in EDGES] from strips of ``depth`` cells next to the four edges of ``a``."""
    a = np.asarray(a, dtype=float)
    chi_nodal = np.asarray(chi_nodal, dtype=float)
    n = a.shape[0]
    L = depth * m
    j0 = offset
    sl = slice(j0, j0 + m)
    strips = {
        "left": (a[:L, sl], chi_nodal[:, 0, sl]),
        "right": (a[::-1][:L, sl], chi_nodal[:, n, sl]),
        "bottom": (a[sl, :L].T, chi_nodal[:, sl, 0]),
        "top": (a[sl, ::-1][:, :L].T, chi_nodal[:, sl, n]),
    }
    tails = np.zeros((2, 4))
    fields = {}
    for e, name in enumerate(EDGES):
        s, dat = strips[name]
        U, t = strip_tail(s, -dat)  # dat (2, m): both correctors share one factorization
        tails[:, e] = t
        fields[name] = U
    return (tails, fields) if return_strips else tails


def edge_data(tails, ux, uy):
    """g (n+1, n+1): boundary values sum_j tail_j(edge) d_j u0 on each edge (ux, uy nodal gradients of u0)."""
    ux, uy = np.asarray(ux, float), np.asarray(uy, float)
    n = ux.shape[0] - 1
    g = np.zeros((n + 1, n + 1))
    g[0, :] = tails[0, 0] * ux[0, :] + tails[1, 0] * uy[0, :]
    g[n, :] = tails[0, 1] * ux[n, :] + tails[1, 1] * uy[n, :]
    g[:, 0] = tails[0, 2] * ux[:, 0] + tails[1, 2] * uy[:, 0]
    g[:, n] = tails[0, 3] * ux[:, n] + tails[1, 3] * uy[:, n]
    return g


def strip_decay_rates(U, m, tail):
    """ln(r_k / r_(k+1)) with r_k the L2 norm over the node column at depth k cells of (U - tail)."""
    U = np.asarray(U, float)
    cols = U[::m][: U.shape[0] // m + 1]
    r = np.sqrt(np.mean((cols - tail) ** 2, axis=1))
    keep = r > 1e-13 * r[0]
    r = r[keep]
    return np.log(r[:-1] / r[1:])


def evanescent_exponents(cell, nkeep=8, method="reduced", return_details=False, kmax=30.0):
    """(kappa, parity) of the decaying static Bloch waves of one Q1 cell (see module docstring)."""
    cell = np.asarray(cell, dtype=float)
    m = cell.shape[0]
    K = open_stiffness(cell, periodic_y=True)[0].toarray()  # nodes (m+1) x m, id i*m + j
    n = m * m
    if method == "reduced":
        ia, ii, ib = slice(0, m), slice(m, n), slice(n, n + m)
        Kaa, KaI, Kab = K[ia, ia], K[ia, ii], K[ia, ib]
        KIa, KII, KIb = K[ii, ia], K[ii, ii], K[ii, ib]
        Kba, KbI, Kbb = K[ib, ia], K[ib, ii], K[ib, ib]
        X = sla.solve(KII, np.hstack([KIa, KIb]), assume_a="pos")
        X0, X1 = -X[:, :m], -X[:, m:]
        A1 = Kab + KaI @ X1
        A0 = Kaa + Kbb + KaI @ X0 + KbI @ X1
        Am1 = Kba + KbI @ X0
        dim = m
    elif method == "full":
        S = np.zeros((m, n))
        S[np.arange(m), np.arange(m)] = 1.0
        Kaa, Kab, Kba, Kbb = K[:n, :n], K[:n, n:], K[n:, :n], K[n:, n:]
        A0 = Kaa + S.T @ Kbb @ S
        A1 = Kab @ S
        Am1 = S.T @ Kba
        dim = n
    else:
        raise ValueError("method must be 'reduced' or 'full'")
    # Solve the REVERSED problem mu^2 A_-1 + mu A0 + A1 = 0 with mu = 1/lambda = e^kappa: the decaying modes are the
    # LARGE eigenvalues of this pencil, which LAPACK computes with high relative accuracy (the small eigenvalues
    # lambda of the direct problem have only absolute accuracy ~1e-16, i.e. kappa = 25 is lost).
    Z, I = np.zeros((dim, dim)), np.eye(dim)
    wh, V = sla.eig(np.block([[Z, I], [-A1, -A0]]), np.block([[I, Z], [Z, Am1]]), homogeneous_eigvals=True)
    alpha, beta = wh
    V = V[:dim, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = alpha / beta
    ok = np.isfinite(mu) & (np.abs(mu) > 1.0 + 1e-9)
    mu, V = mu[ok], V[:, ok]
    w = 1.0 / mu
    kap = np.log(mu.astype(complex))
    sel = (kap.real > 1e-6) & (kap.real < kmax)
    w, V, kap = w[sel], V[:, sel], kap[sel]
    order = np.argsort(kap.real, kind="stable")
    w, V, kap = w[order], V[:, order].copy(), kap[order]
    # degenerate clusters (e.g. +-k_y pairs of a homogeneous cell): rotate the vectors to definite parity
    perm = (-np.arange(m)) % m
    k = 0
    while k < len(kap):
        j = k + 1
        while j < len(kap) and abs(kap[j] - kap[k]) < 1e-6 * abs(kap[k]):
            j += 1
        if j - k > 1:
            B, _ = np.linalg.qr(V[:m, k:j])
            Pb = B.conj().T @ B[perm, :]
            ev, R = np.linalg.eigh(0.5 * (Pb + Pb.conj().T))
            if np.all(np.minimum(abs(ev - 1), abs(ev + 1)) < 1e-6):
                C = np.linalg.lstsq(V[:m, k:j], B, rcond=None)[0]
                V[:, k:j] = V[:, k:j] @ C @ R
        k = j
    w, V, kap = w[:nkeep], V[:, :nkeep], kap[:nkeep]
    # parity of node column 0 under j -> -j mod m (the mirror y -> 1 - y)
    col0 = V[:m, :]
    mirror = col0[(-np.arange(m)) % m, :]
    expect = np.einsum("ik,ik->k", col0.conj(), mirror) / np.einsum("ik,ik->k", col0.conj(), col0)
    parity = np.where(np.abs(expect - 1) < 1e-6, 1, np.where(np.abs(expect + 1) < 1e-6, -1, 0)).astype(int)
    if not return_details:
        return kap, parity
    res = np.array([np.linalg.norm((w[k] ** 2 * A1 + w[k] * A0 + Am1) @ V[:, k]) / (np.linalg.norm(V[:, k]) * np.linalg.norm(A0))
                    for k in range(len(w))])
    return kap, parity, {"vector": col0, "residual": res, "parity_expectation": expect, "lambda": w}


# ----------------------------------------------------------------------------- error norms and weak limits


def interior_box(lo=0.25, hi=0.75):
    return (lo, hi)


def _corners(e):
    return np.stack([e[:-1, :-1], e[1:, :-1], e[1:, 1:], e[:-1, 1:]])  # (4, n, n)


def _element_energy(c):
    """Exact integral of |grad u|^2 over each pixel (h independent) from the corner values c (4, n, n), written with
    differences (no cancellation): (dx0^2 + dx0 dx1 + dx1^2 + dy0^2 + dy0 dy1 + dy1^2) / 3."""
    dx0, dx1 = c[1] - c[0], c[2] - c[3]
    dy0, dy1 = c[3] - c[0], c[2] - c[1]
    return (dx0 * dx0 + dx0 * dx1 + dx1 * dx1 + dy0 * dy0 + dy0 * dy1 + dy1 * dy1) / 3.0


def norms(e, region=None):
    """(Linf, L2, H1 seminorm) of the nodal field e; see the module docstring for the region convention."""
    e = np.asarray(e, dtype=float)
    n = e.shape[0] - 1
    h = 1.0 / n
    xs = np.arange(n + 1) * h
    xc = (np.arange(n) + 0.5) * h
    if region is None:
        nodes = np.ones((n + 1, n + 1), bool)
        w = np.ones((n, n))
    else:
        lo, hi = region
        tol = 1e-12
        inn = (xs >= lo - tol) & (xs <= hi + tol)
        nodes = np.outer(inn, inn)
        ine = (xc >= lo - tol) & (xc <= hi + tol)
        w = np.outer(ine, ine).astype(float)
    c = _corners(e)
    l2 = h * h * np.einsum("kij,kl,lij,ij->", c, ME, c, w)
    h1 = np.sum(_element_energy(c) * w)
    return float(np.abs(e[nodes]).max()), float(np.sqrt(l2)), float(np.sqrt(max(h1, 0.0)))


def first_order_expansion(u0, ux, uy, chi_nodal, N):
    """w = u0 + eps (chi_1 u0x + chi_2 u0y) at the nodes, eps = 1/N."""
    return u0 + (chi_nodal[0] * ux + chi_nodal[1] * uy) / N


def element_gradients(u):
    """(gx, gy) (n,n): mean gradient of the Q1 field u over each pixel."""
    n = u.shape[0] - 1
    gx = (u[1:, :-1] - u[:-1, :-1] + u[1:, 1:] - u[:-1, 1:]) * (n / 2.0)
    gy = (u[:-1, 1:] - u[:-1, :-1] + u[1:, 1:] - u[1:, :-1]) * (n / 2.0)
    return gx, gy


def weak_integrals(u, a, phi, exact_energy=False):
    """(I_flux, I_grad) = sum phi a |g|^2 h^2 and sum phi |g|^2 h^2 (phi at the pixel centers, (n,n)).

    Default: g is the element-mean gradient (midpoint rule). ``exact_energy=True`` replaces h^2 |g|^2 by the exact
    element integral of |grad u|^2 (u^T KE u), which is the consistent partner of ``corrector_gram`` and of the
    effective tensor ``sigma`` (their cell sums are exact element integrals).
    """
    n = u.shape[0] - 1
    if exact_energy:
        c = _corners(u)
        g2 = _element_energy(c) * n ** 2
    else:
        gx, gy = element_gradients(u)
        g2 = gx * gx + gy * gy
    return float(np.sum(phi * a * g2) / n ** 2), float(np.sum(phi * g2) / n ** 2)


def midpoint_tensors(cell, chi):
    """(S, M) (2,2): cell averages of a g_i.g_j and g_i.g_j with g_d = e_d + grad chi_d the element-MEAN fields.

    These are the limits of the midpoint-rule integrals of ``weak_integrals`` (default mode); they differ from
    ``sigma`` and ``corrector_gram`` by the O(h^2) twist energy of the Q1 elements (about 1% at m = 16).
    """
    f = [cs.q1_center_field(chi, d) for d in range(2)]
    a = np.asarray(cell, float)
    S = np.array([[np.mean(a * np.sum(f[i] * f[j], 0)) for j in range(2)] for i in range(2)])
    M = np.array([[np.mean(np.sum(f[i] * f[j], 0)) for j in range(2)] for i in range(2)])
    return S, M


def weak_limits(phi, ux, uy, sigma, M):
    """(lim_flux, lim_grad, naive) of the weak limits; ux, uy are the u0 gradients at the pixel centers (n,n)."""
    n = phi.shape[0]
    s = np.asarray(sigma, float)
    flux = s[0, 0] * ux * ux + 2 * s[0, 1] * ux * uy + s[1, 1] * uy * uy
    Mq = M[0, 0] * ux * ux + 2 * M[0, 1] * ux * uy + M[1, 1] * uy * uy
    return (float(np.sum(phi * flux) / n ** 2), float(np.sum(phi * Mq) / n ** 2), float(np.sum(phi * (ux * ux + uy * uy)) / n ** 2))
