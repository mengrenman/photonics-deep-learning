"""Plane-wave Bloch solver for TM rod lattices, little-group tools, k.p theory, Berry phases, 1D stacks.

Purpose
-------
Serves notebook 05 (band edges, little groups, k.p, Berry phases, finite 1D stacks) and the TM second-order
homogenization of notebook 06 (``eps_hat`` is imported by ``scatter.py``).

Conventions
-----------
* Time dependence exp(-i omega t); units c = 1, so omega = k0. TM means E along z: u = E_z obeys
  ``-(grad + i k)^2 u = lam eps u`` for the periodic part u of a Bloch wave exp(i k.r) u, with ``lam = (omega/c)^2``.
* Lengths are in units of the period ``a`` of ``Lattice`` (default a = 1). Frequencies ``f = omega a / (2 pi) = a/lambda``.
  Wave vectors are physical (2 pi/a is the size of the reciprocal lattice).
* Arrays ``a[i, j]`` have i the x index and j the y index, as in PLAN.md. Plane waves carry the reciprocal lattice
  vector ``G = i b1 + j b2``; ``G`` arrays have shape (n, 2) (Cartesian), integer index arrays have shape (n, 2).
* Fourier convention ``eps(r) = sum_G eps_G exp(i G.r)``, ``eps_G = (1/A) int_cell eps(r) exp(-i G.r) d^2r``.
* Band indices in code are 0-based (index 0 is the lowest band; "band 1" in the notebooks is index 0).
* The generalized Hermitian problem ``A c = lam B c`` with ``A = diag |k+G|^2`` and ``B[G, G'] = eps_{G-G'}``
  (positive definite) is solved with ``scipy.linalg.eigh``; eigenvectors are B-orthonormal, ``C^H B C = 1``.
  The matrix ``B`` is the Toeplitz ("Laurent") matrix of eps itself, which is the right factorization for TM (E_z is
  continuous, so ``eps E_z`` is a product of a piecewise constant and a continuous function).

Symmetry-preserving cutoff
--------------------------
``gvectors`` keeps ``|center + G| <= gmax`` (a disk around a high-symmetry point ``center``). The set is closed under every
point-group operation about the origin that fixes ``center`` modulo a reciprocal vector, so the truncated problem has the
exact little-group symmetry and degeneracies are exact to roundoff (about 1e-13 relative). Parallelogram cutoffs
(``|i|, |j| <= M``) split such degeneracies by about 1e-6. Use one G set (built around the loop center) for all k of a loop
or finite-difference stencil.

Public functions (contracts)
----------------------------
``Lattice.square(a)``, ``Lattice.honeycomb(a)``
    Lattice vectors ``a1, a2``, reciprocal ``b1, b2`` (a_i . b_j = 2 pi delta_ij), ``area``, ``points`` (dict of
    high-symmetry points as Cartesian arrays: square G, X = (pi/a, 0), M = (pi/a, pi/a); honeycomb G, K, Kp, M), and for
    the honeycomb ``sites`` (the two rod positions (a1+a2)/3 and 2(a1+a2)/3, C6v about the origin, the hexagon center).
    The triangular lattice has a1 = (a, 0), a2 = (a/2, sqrt(3) a/2); K = (b1 + 2 b2)/3 at angle 60 degrees, Kp = (2 b1 + b2)/3.
``disk(center, R, eps)``, ``ellipse(center, rx, ry, eps, angle=0)``
    Rod dictionaries ``{center, rx, ry, eps, angle}`` (the background is eps = 1 unless ``eps_bg`` is passed).
``gvectors(lat, gmax, center=0)``
    Returns ``(G, ij)`` with ``|center + G| <= gmax (1 + 1e-9)`` sorted by ``|center + G|``. The tolerance keeps complete
    shells. ``gmax_for_count(lat, n, center)`` returns the smallest gmax whose set has at least n plane waves
    and consists of complete shells.
``eps_hat(lat, rods, dG, eps_bg=1.0)``
    Fourier coefficients ``eps_G`` for reciprocal vectors ``dG`` of shape (..., 2); returns complex array of shape
    dG.shape[:-1]. Exact for non-overlapping rods inside one cell: ``eps_0`` is the area average; for an ellipse the form
    factor is ``2 J1(rho)/rho`` with ``rho = sqrt((Gx rx)^2 + (Gy ry)^2)`` in the ellipse axes. ``dG`` must be reciprocal
    lattice vectors (this is not checked) for the coefficients to describe a periodic function. Rods that overlap
    each other are summed (eps differences add), which is not the union.
``eps_matrix(lat, rods, G, eps_bg=1.0)``
    The Hermitian Toeplitz matrix ``B[G, G'] = eps_{G-G'}``, exact in the truncated basis. Real dtype if the rod set is
    inversion symmetric about the origin (imaginary part below 1e-13), complex otherwise.
``tm_solve(lat, rods, k, G, nbands=None, eps_bg=1.0)``
    Returns ``(f, C, B)``: frequencies ``f = sqrt(lam) a/(2 pi)`` (ascending, lam clipped at 0), B-orthonormal eigenvectors
    ``C[:, m]`` (n, nbands) and ``B``. ``nbands=None`` gives all n states. Guaranteed: ``C^H B C = 1`` to 1e-12;
    ``f`` are exact eigenvalues of the truncated problem (the truncation, not the method, is the approximation; for
    disk rods the error in f decays roughly as 1/n_pw, see the test).
``bands_on_path(lat, rods, kpath, G, nbands)``
    Frequencies for an array of k points (m, 2) with one G set: ``(m, nbands)``.
``point_group_ops(lat, k0=None, rods=None)``
    Returns ``(ops, name)``: the point-group operations about the origin (the lattice holohedry: C4v or C6v) that fix
    ``k0`` modulo a reciprocal vector (if given) and leave the rods invariant (if given). ``ops`` is a list of dicts
    ``{label, matrix, cls}`` (2x2 Cartesian matrix acting on vectors; ``cls`` the conjugacy-class label in the subgroup);
    ``name`` is the subgroup name (``C4v, C6v, C3v, C2v, C3, C2, Cs, C1``).
``rep_matrices(lat, C, G, k0, ops, B)``
    List of representation matrices ``D(g) = C^H B P_g C`` (nb x nb) of the operations on the chosen eigenvectors, where
    ``(P_g u)(r) = u(g^-1 r)`` permutes the plane-wave coefficients (g(k0+G) = k0+G'). Guaranteed: homomorphism and
    unitarity on a complete degenerate set (``rep_error`` quantifies). Raises ValueError if g is not in the little group.
``degenerate_sets(f, tol=1e-8)``
    Lists of consecutive band indices whose frequencies differ by less than ``tol`` (relative).
``irreps_of_bands(lat, f, C, G, k0, B, ops, tol=1e-8)``
    For each degenerate set: bands, frequency, characters (per op), irrep multiplicities (dict irrep -> float) and the label
    of the irrep when the decomposition is a single irrep to 1e-6, else ``None``. Character tables exist for C4v, C6v, C3v, C2v,
    C3, C2, Cs, C1 (``CHARACTER_TABLES``; conventions in the table comments).
``hom_dimension(ops, chi, kind)``
    ``dim Hom_G(Sym^2 V, End E)`` (kind ``"quadratic"``) or ``dim Hom_G(V, End E)`` (kind ``"linear"``), V the plane, E the
    representation with characters ``chi`` (per op), computed as an exact character inner product (returns a float that
    is an integer to roundoff). Time reversal is not imposed.
``kp_second_order(lat, rods, k0, G, bands, solution=None, eps_bg=1.0)``
    k.p data for the set ``bands`` (a complete degenerate set; indices 0-based) at ``k0``. Returns a tuple subclass
    ``(H1, H2)`` with attributes ``lam0`` (mean lam of the set), ``lam_set`` (lam of each band of the set),
    ``C`` (B-orthonormal vectors of the set), ``lam`` (all lam), ``f0``. In lam = (omega/c)^2 units the effective
    Hamiltonian is ``H(q) = diag(lam_set) + sum_i q_i H1[i] + sum_ij q_i q_j H2[i, j]`` (d x d Hermitian matrices; H2 is
    not a Hessian: ``q.H2.q`` with H2 symmetric in (i, j), so the Hessian is 2 H2; for a nondegenerate band
    ``lam(q) = lam0 + H1.q + q.H2.q``). It
    is the exact second-order Loewdin expansion of the truncated problem (sum over all other computed states): the eigenvalues
    of H(q) differ from the PWE eigenvalues in the same basis by O(q^3) (O(q^4) when odd orders vanish by symmetry).
    ``solution=(f, C, B)`` from ``tm_solve(nbands=None)`` avoids a second diagonalization. The set must be complete.
``kp_hamiltonian(kp, q)``
    The d x d matrix ``diag(lam of the set) + q.H1 + q.H2.q`` in the basis ``kp.C``, for a vector q. The Loewdin denominators
    are the symmetrized ``(1/2)(1/(l_a - l_n) + 1/(l_b - l_n))``, which reduce to ``1/(lam0 - l_n)`` for an exactly degenerate
    set and keep the expansion consistent for a nearly degenerate pair (a gapped Dirac point).
``kp_berry_phase(kp, radius, band, nk=64, order=2)``
    Berry phase of a k.p eigenvector on a circle (discrete product, as ``berry_phase``); ``order=1`` keeps only the linear
    term, ``order=2`` includes H2. Compare with ``berry_phase`` of the PWE: the difference is the effect of terms beyond
    the expansion order.
``adapt_doublet(kp, D_mx)``, ``c4v_constants(H2)``
    For a C4v doublet: ``adapt_doublet`` rotates the two states into an eigenbasis of the mirror x -> -x
    (index 0 odd, like x; index 1 even, like y) with the relative phase fixed so that the q_x q_y coefficient is real
    and non-negative. Returns ``(U, H1a, H2a)``. ``c4v_constants(H2a)`` returns ``(A, B, C, residual)`` for
    ``H2a[i,j] q_i q_j = (A q^2 + B (qx^2 - qy^2) sz + C qx qy sx)`` (``residual`` is the Frobenius norm of the part of
    H2a that is not of that form; it vanishes by symmetry for a true E doublet).
``degenerate_first_order(lam0, C, dB)``
    ``-lam0 C^H dB C``: the first-order shift matrix of a degenerate set for a perturbation ``B -> B + dB`` at fixed
    ``A`` (generalized problem). Exact to first order in dB.
``berry_phase(lat, rods, center, radius, bands, nk=64, gmax=None, G=None)``
    Berry phase ``gamma = -Im log prod_j det <u_j | B | u_{j+1}>`` over a counterclockwise circle of ``nk`` points around
    ``center`` with a fixed G set (gauge invariant; the B-weighted overlap is the natural inner product). ``bands`` is an
    int or a list (Wilson loop of a band group). Returns gamma in (-pi, pi]. For a loop of finite nk the value is exact for the discrete
    path (quantization to pi holds to roundoff for real-gauge problems, i.e. inversion-symmetric structures);
    for smooth bands the error vs the continuum phase is O(1/nk^2). Sign: counterclockwise loop, gamma = i oint <u|B du>.
``pseudospin_curve(lat, rods, center, radius, band, nk=64, gmax=None, G=None, doublet=None, U=None, doublet_bands=None)``
    Bloch vector ``(<sx>, <sy>, <sz>)`` (nk, 3) of band ``band`` (int) of the PWE on a circle, projected on the two states
    ``doublet`` (n, 2; B-orthonormal on the same G set, e.g. ``kp.C``) and normalized, in the basis ``doublet @ U`` (U optional,
    e.g. from ``adapt_doublet``). Alternatively pass ``doublet_bands`` (two band indices at the center). Returns ``(phi, n)``.
    The sphere's orientation depends on the basis: use the adapted basis for the (z, x)-plane statement.
``winding_number(n, axes=(2, 0))``
    Signed winding number (float) of ``atan2(n[:, axes[1]], n[:, axes[0]])`` around the closed curve; ``axes=None`` picks the
    plane of the best-fit great circle (sign then fixed so that the result is non-negative).
``transfer_1d(layers, omega, deriv=False)``
    2x2 transfer matrix mapping ``(u, u')`` at the start to the end of a stack of ``layers = [(eps, d), ...]``
    for ``u'' + omega^2 eps u = 0`` (u and u' continuous, the ordinary normal-incidence wave). ``omega`` may be an array;
    result shape (..., 2, 2). With ``deriv=True`` also returns the exact omega derivative. det = 1 to roundoff.
``bloch_k_1d(layers, omega)``
    Complex Bloch wave number k (per length) with ``cos(k a) = tr M / 2``, a = sum of d; Re(k) a in [0, 2 pi), Im(k) >= 0
    (Im k > 0 in gaps; Re k a = 0 or pi there).
``band_edges_1d(layers, omega_window, nscan=4000)``
    Sorted band-edge frequencies in the window as ``(omega, s)`` with ``s = +1`` (k = 0 edge) or ``-1`` (k a = pi edge),
    where ``tr M / 2 = s``; found by scan and brentq (accurate to 1e-13). Touching bands (|tr M/2| = 1 with a
    double root) are missed.
``edge_curvature_1d(layers, omega_edge, s)``
    ``T = (1/2) d^2(omega^2)/dk^2`` at the band edge (exact formula ``T = - s omega a^2 / (d tr(M)/2 / d omega)``).
``finite_stack_modes(layers, N, wall_offset=0.0, omega_window=..., nscan=4000)``
    Roots of ``u(end)`` for Dirichlet walls (u = 0) at both ends of N cells whose start lies ``wall_offset`` inside the
    cell. Returns ``(omega, k)``: sorted roots in the window (simple roots, scan resolution ``nscan``; a pair of roots closer than the
    scan step could be missed) and the complex Bloch k of the infinite crystal at each.

Approximations: none of the exact statements above (degeneracy, orthonormality, k.p identities, Wilson-loop quantization)
involves an approximation beyond the plane-wave truncation; band frequencies converge with the number of plane waves.
"""

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq
from scipy.special import j1

__all__ = [
    "Lattice", "disk", "ellipse", "gvectors", "gmax_for_count", "eps_hat", "eps_matrix", "tm_solve", "bands_on_path",
    "point_group_ops", "rep_matrices", "degenerate_sets", "irreps_of_bands", "hom_dimension", "kp_second_order",
    "kp_hamiltonian", "kp_berry_phase", "adapt_doublet", "c4v_constants", "degenerate_first_order", "berry_phase", "pseudospin_curve",
    "winding_number", "transfer_1d", "bloch_k_1d", "band_edges_1d", "edge_curvature_1d", "finite_stack_modes",
    "CHARACTER_TABLES", "rep_error",
]

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------- lattices and rods

class Lattice:
    """Two-dimensional lattice: ``a1, a2`` (rows of ``A``), reciprocal ``b1, b2``, ``area``, ``points``, ``name``."""

    def __init__(self, name, a1, a2, points, sites=None):
        self.name = name
        self.a1 = np.asarray(a1, float)
        self.a2 = np.asarray(a2, float)
        self.area = abs(self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0])
        Amat = np.array([self.a1, self.a2])                    # rows
        Bmat = TWO_PI * np.linalg.inv(Amat).T                  # rows b1, b2 with a_i . b_j = 2 pi delta_ij
        self.b1, self.b2 = Bmat[0], Bmat[1]
        self.points = {k: np.asarray(v(self) if callable(v) else v, float) for k, v in points.items()}
        self.sites = sites
        self.a = float(np.linalg.norm(self.a1))

    @classmethod
    def square(cls, a=1.0):
        lat = cls("square", (a, 0.0), (0.0, a), {})
        lat.points = {"G": np.zeros(2), "X": np.array([np.pi / a, 0.0]), "M": np.array([np.pi / a, np.pi / a])}
        return lat

    @classmethod
    def honeycomb(cls, a=1.0):
        s3 = np.sqrt(3.0)
        lat = cls("honeycomb", (a, 0.0), (a / 2, s3 * a / 2), {})
        lat.points = {"G": np.zeros(2), "K": (lat.b1 + 2 * lat.b2) / 3, "Kp": (2 * lat.b1 + lat.b2) / 3, "M": lat.b1 / 2}
        lat.sites = [(lat.a1 + lat.a2) / 3, 2 * (lat.a1 + lat.a2) / 3]
        return lat

    def index(self, G):
        """Integer indices (i, j) of reciprocal vectors G (n, 2): G = i b1 + j b2 (rounded; use ``index_error``)."""
        G = np.asarray(G, float)
        return np.stack([G @ self.a1, G @ self.a2], axis=-1) / TWO_PI

    def __repr__(self):
        return f"Lattice({self.name}, a={self.a:g})"


def disk(center, R, eps):
    """Circular rod ``{center, rx = ry = R, eps, angle = 0}``."""
    return dict(center=np.asarray(center, float), rx=float(R), ry=float(R), eps=float(eps), angle=0.0)


def ellipse(center, rx, ry, eps, angle=0.0):
    """Elliptical rod with semi-axes rx (along direction ``angle`` radians from x) and ry."""
    return dict(center=np.asarray(center, float), rx=float(rx), ry=float(ry), eps=float(eps), angle=float(angle))


def _rod_fields(rod):
    c = np.asarray(rod["center"], float)
    rx = rod["rx"] if "rx" in rod else rod["R"]
    ry = rod.get("ry", rx)
    return c, rx, ry, rod["eps"], rod.get("angle", 0.0)


def gvectors(lat, gmax, center=0.0):
    """Reciprocal vectors with ``|center + G| <= gmax (1 + 1e-9)``, sorted by |center + G|. Returns (G (n,2), ij (n,2))."""
    center = np.zeros(2) if np.isscalar(center) else np.asarray(center, float)
    rad = gmax * (1 + 1e-9) + np.linalg.norm(center)
    ni = int(np.ceil(rad * np.linalg.norm(lat.a1) / TWO_PI)) + 2
    nj = int(np.ceil(rad * np.linalg.norm(lat.a2) / TWO_PI)) + 2
    I, J = np.meshgrid(np.arange(-ni, ni + 1), np.arange(-nj, nj + 1), indexing="ij")
    I, J = I.ravel(), J.ravel()
    G = I[:, None] * lat.b1[None, :] + J[:, None] * lat.b2[None, :]
    d = np.linalg.norm(G + center[None, :], axis=1)
    keep = d <= gmax * (1 + 1e-9)
    order = np.argsort(np.round(d[keep], 9), kind="stable")
    return G[keep][order], np.stack([I[keep], J[keep]], axis=1)[order]


def gmax_for_count(lat, n, center=0.0):
    """Smallest cutoff giving at least n plane waves with complete shells (the n-th distance, a hair above)."""
    center = np.zeros(2) if np.isscalar(center) else np.asarray(center, float)
    rad = 2.0 * np.sqrt(4 * np.pi * n / lat.area) + 4 * np.linalg.norm(center) + 1.0
    G, _ = gvectors(lat, rad, center)
    d = np.sort(np.round(np.linalg.norm(G + center[None, :], axis=1), 9))
    return float(d[n - 1]) + 1e-8


def eps_hat(lat, rods, dG, eps_bg=1.0):
    """Fourier coefficients ``eps_G`` of the periodic permittivity at reciprocal vectors ``dG`` (..., 2).

    ``eps_G = eps_bg delta_G0 + sum_rods (eps - eps_bg) f 2 J1(rho)/rho exp(-i G.c)``, ``f = pi rx ry / A``,
    ``rho = sqrt((Gx' rx)^2 + (Gy' ry)^2)`` with G' the vector in the ellipse axes (rotated by ``angle``).
    """
    dG = np.asarray(dG, float)
    shp = dG.shape[:-1]
    g = dG.reshape(-1, 2)
    out = np.where(np.einsum("ij,ij->i", g, g) < 1e-18, eps_bg, 0.0).astype(complex)
    for rod in rods:
        c, rx, ry, eps, ang = _rod_fields(rod)
        ca, sa = np.cos(ang), np.sin(ang)
        gx = g[:, 0] * ca + g[:, 1] * sa
        gy = -g[:, 0] * sa + g[:, 1] * ca
        rho = np.hypot(gx * rx, gy * ry)
        small = rho < 1e-12
        form = np.where(small, 1.0, 2.0 * j1(np.where(small, 1.0, rho)) / np.where(small, 1.0, rho))
        f = np.pi * rx * ry / lat.area
        out += (eps - eps_bg) * f * form * np.exp(-1j * (g @ c))
    return out.reshape(shp)


def _indices(lat, G):
    ij = np.rint(lat.index(G)).astype(int)
    if np.abs(lat.index(G) - ij).max() > 1e-6:
        raise ValueError("G is not a set of reciprocal lattice vectors")
    return ij


def eps_matrix(lat, rods, G, eps_bg=1.0):
    """Toeplitz matrix ``B[G, G'] = eps_{G - G'}`` for the plane-wave set G (n, 2)."""
    ij = _indices(lat, G)
    di = ij[:, 0][:, None] - ij[:, 0][None, :]
    dj = ij[:, 1][:, None] - ij[:, 1][None, :]
    Di, Dj = int(np.abs(di).max()), int(np.abs(dj).max())
    ii, jj = np.meshgrid(np.arange(-Di, Di + 1), np.arange(-Dj, Dj + 1), indexing="ij")
    tab = eps_hat(lat, rods, ii[..., None] * lat.b1 + jj[..., None] * lat.b2, eps_bg)
    B = tab[di + Di, dj + Dj]
    if np.abs(B.imag).max() < 1e-13:
        B = B.real.copy()
    return B


def tm_solve(lat, rods, k, G, nbands=None, eps_bg=1.0, B=None):
    """Lowest ``nbands`` TM states at wave vector k: returns (f = sqrt(lam) a/2pi, C B-orthonormal, B)."""
    if B is None:
        B = eps_matrix(lat, rods, G, eps_bg)
    A = np.sum((np.asarray(k, float)[None, :] + G) ** 2, axis=1)
    n = len(A)
    kw = {} if nbands is None or nbands >= n else dict(subset_by_index=[0, nbands - 1])
    # generalized problem with diagonal A: scipy reduces with a Cholesky factor of B
    lam, C = sla.eigh(np.diag(A).astype(B.dtype), B, **kw)
    return np.sqrt(np.maximum(lam, 0.0)) * lat.a / TWO_PI, C, B


def bands_on_path(lat, rods, kpath, G, nbands, eps_bg=1.0):
    """Frequencies ``(m, nbands)`` at the k points ``kpath`` (m, 2) with the common plane-wave set G."""
    B = eps_matrix(lat, rods, G, eps_bg)
    return np.array([tm_solve(lat, rods, k, G, nbands, B=B)[0] for k in np.atleast_2d(kpath)])


# ----------------------------------------------------------------------------- point groups

def _rot(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _mir(deg):
    t = 2 * np.deg2rad(deg)
    return np.array([[np.cos(t), np.sin(t)], [np.sin(t), -np.cos(t)]])


def _holohedry(lat):
    """All operations of the lattice point group about the origin: (label, matrix, kind, angle)."""
    n = 4 if lat.name == "square" else 6
    ops = []
    for m in range(n):
        ang = 360.0 * m / n
        lab = "E" if m == 0 else f"C{n}^{m}"
        ops.append(dict(label=lab, matrix=_rot(ang), kind="rot", angle=ang))
    for m in range(n):
        ax = 180.0 * m / n
        ops.append(dict(label=f"m{ax:g}", matrix=_mir(ax), kind="mir", angle=ax))
    return ops


def _classes(mats):
    """Conjugacy classes (lists of indices) of the matrix group ``mats`` (list of 2x2 arrays)."""
    n = len(mats)
    inv = [m.T for m in mats]                              # orthogonal

    def find(M):
        for i, m in enumerate(mats):
            if np.allclose(m, M, atol=1e-8):
                return i
        raise ValueError("not a group")
    seen, out = set(), []
    for i in range(n):
        if i in seen:
            continue
        cl = sorted({find(mats[h] @ mats[i] @ inv[h]) for h in range(n)})
        seen.update(cl)
        out.append(cl)
    return out


def _name_group(ops):
    nrot = sum(o["kind"] == "rot" for o in ops)
    nmir = len(ops) - nrot
    return {(4, 4): "C4v", (6, 6): "C6v", (3, 3): "C3v", (2, 2): "C2v", (1, 1): "Cs", (3, 0): "C3", (2, 0): "C2",
            (1, 0): "C1", (4, 0): "C4", (6, 0): "C6"}.get((nrot, nmir), f"unnamed({nrot},{nmir})")


def _class_labels(ops, name):
    """Assign class labels used by ``CHARACTER_TABLES`` to every operation (sets ``op['cls']``)."""
    for o in ops:
        if o["kind"] == "rot":
            ang = o["angle"] % 360.0
            if abs(ang) < 1e-9:
                o["cls"] = "E"
            elif abs(ang - 180) < 1e-9:
                o["cls"] = "C2"
            else:
                n = int(round(360.0 / min(ang, 360 - ang)))
                o["cls"] = f"C{n}" if name in ("C4v", "C6v", "C3v") else f"C{n}{'+' if ang < 180 else '-'}"
        else:
            ax = o["angle"] % 180.0
            if name in ("C4v", "C6v"):
                per = 90.0 if name == "C4v" else 60.0
                r = ax % per
                o["cls"] = "sv" if (r < 1e-6 or per - r < 1e-6) else "sd"
            elif name == "C3v":
                o["cls"] = "sv"
            elif name == "C2v":                           # smaller axis angle -> 'my' (axis like x), the other 'mx'
                axes = sorted(q["angle"] % 180.0 for q in ops if q["kind"] == "mir")
                o["cls"] = "my" if abs(ax - axes[0]) < 1e-6 else "mx"
            else:
                o["cls"] = "m"
    return ops


def point_group_ops(lat, k0=None, rods=None):
    """Point-group operations about the origin that fix k0 (mod G) and the rods: returns (ops, name)."""
    ops = []
    rng = np.random.default_rng(0)
    ijs = rng.integers(-3, 4, size=(40, 2))
    Gs = ijs[:, :1] * lat.b1 + ijs[:, 1:] * lat.b2
    base = None if rods is None else eps_hat(lat, rods, Gs)
    for o in _holohedry(lat):
        R = o["matrix"]
        if k0 is not None:
            dk = R @ np.asarray(k0, float) - np.asarray(k0, float)
            if np.abs(lat.index(dk) - np.rint(lat.index(dk))).max() > 1e-8:
                continue
        if rods is not None and np.abs(eps_hat(lat, rods, Gs @ R.T) - base).max() > 1e-9:
            continue
        ops.append(dict(o))
    name = _name_group(ops)
    # the identity first, then by kind
    ops.sort(key=lambda o: (o["kind"] != "rot", o["angle"]))
    return _class_labels(ops, name), name


_w = np.exp(2j * np.pi / 3)
CHARACTER_TABLES = {
    # columns: classes in the order of the keys of each inner dict; conventions: C4v/C6v B1 is even under 'sv'
    # (mirror planes along x and y, resp. the mirrors at 0 degrees), B2 even under 'sd'; C2v: B1 odd under mx (x -> -x,
    # like x), B2 odd under my (like y); Cs: A' even, A'' odd under the mirror.
    "C4v": {"A1": dict(E=1, C2=1, C4=1, sv=1, sd=1), "A2": dict(E=1, C2=1, C4=1, sv=-1, sd=-1),
            "B1": dict(E=1, C2=1, C4=-1, sv=1, sd=-1), "B2": dict(E=1, C2=1, C4=-1, sv=-1, sd=1),
            "E": dict(E=2, C2=-2, C4=0, sv=0, sd=0)},
    "C6v": {"A1": dict(E=1, C2=1, C3=1, C6=1, sv=1, sd=1), "A2": dict(E=1, C2=1, C3=1, C6=1, sv=-1, sd=-1),
            "B1": dict(E=1, C2=-1, C3=1, C6=-1, sv=1, sd=-1), "B2": dict(E=1, C2=-1, C3=1, C6=-1, sv=-1, sd=1),
            "E1": dict(E=2, C2=-2, C3=-1, C6=1, sv=0, sd=0), "E2": dict(E=2, C2=2, C3=-1, C6=-1, sv=0, sd=0)},
    "C3v": {"A1": dict(E=1, C3=1, sv=1), "A2": dict(E=1, C3=1, sv=-1), "E": dict(E=2, C3=-1, sv=0)},
    "C2v": {"A1": dict(E=1, C2=1, mx=1, my=1), "A2": dict(E=1, C2=1, mx=-1, my=-1),
            "B1": dict(E=1, C2=-1, mx=-1, my=1), "B2": dict(E=1, C2=-1, mx=1, my=-1)},
    "C3": {"A": {"E": 1, "C3+": 1, "C3-": 1}, "E1": {"E": 1, "C3+": _w, "C3-": np.conj(_w)},
           "E2": {"E": 1, "C3+": np.conj(_w), "C3-": _w}},
    "C2": {"A": dict(E=1, C2=1), "B": dict(E=1, C2=-1)},
    "Cs": {"A'": dict(E=1, m=1), "A''": dict(E=1, m=-1)},
    "C1": {"A": dict(E=1)},
}


def rep_matrices(lat, C, G, k0, ops, B):
    """``D(g) = C^H B P_g C`` for every op: the operators permute plane-wave coefficients (g(k0 + G) = k0 + G')."""
    ij = _indices(lat, G)
    lo = ij.min(axis=0)
    shape = ij.max(axis=0) - lo + 1
    lut = -np.ones(shape, int)
    lut[ij[:, 0] - lo[0], ij[:, 1] - lo[1]] = np.arange(len(G))
    k0 = np.asarray(k0, float)
    out = []
    for o in ops:
        R = o["matrix"]
        Gp = (G + k0[None, :]) @ R.T - k0[None, :]
        fij = lat.index(Gp)
        pij = np.rint(fij).astype(int)
        if np.abs(fij - pij).max() > 1e-6:
            raise ValueError(f"operation {o['label']} is not in the little group of k0")
        p = pij - lo
        if (p < 0).any() or (p >= shape).any() or (lut[p[:, 0], p[:, 1]] < 0).any():
            raise ValueError("the plane-wave set is not closed under the operation (use a circular cutoff about k0)")
        perm = lut[p[:, 0], p[:, 1]]
        PC = np.empty_like(C)
        PC[perm] = C
        out.append(C.conj().T @ (B @ PC))
    return out


def rep_error(Ds, ops):
    """Largest deviation from unitarity, and from the homomorphism property, of the matrices ``Ds`` (complete degenerate set)."""
    mats = [o["matrix"] for o in ops]
    d = Ds[0].shape[0]
    e_u = max(np.abs(D.conj().T @ D - np.eye(d)).max() for D in Ds)
    e_h = 0.0
    for i, Ri in enumerate(mats):
        for j, Rj in enumerate(mats):
            M = Ri @ Rj
            k = next(m for m, Rk in enumerate(mats) if np.allclose(Rk, M, atol=1e-8))
            e_h = max(e_h, np.abs(Ds[i] @ Ds[j] - Ds[k]).max())
    return e_u, e_h


def degenerate_sets(f, tol=1e-8):
    """Groups of consecutive band indices with ``|f_{n+1} - f_n| <= tol (1 + |f_n|)``."""
    sets, cur = [], [0]
    for n in range(1, len(f)):
        if abs(f[n] - f[n - 1]) <= tol * (1 + abs(f[n - 1])):
            cur.append(n)
        else:
            sets.append(cur)
            cur = [n]
    sets.append(cur)
    return sets


def irreps_of_bands(lat, f, C, G, k0, B, ops, tol=1e-8):
    """Characters and irrep decomposition of every degenerate set among the computed bands (see module docstring).

    A multiplet cut by the last computed band cannot be detected: compute one band more than needed and ignore the last set.
    Returns a list of dicts with keys ``bands, f, chars, mult, irrep, error`` (``error`` = deviation of the multiplicities
    from integers plus the unitarity error of the block; ``mult`` maps irrep names to multiplicities).
    """
    name = _name_group(ops)
    table = CHARACTER_TABLES.get(name)
    out = []
    for s in degenerate_sets(f, tol):
        Cs = C[:, s]
        Ds = rep_matrices(lat, Cs, G, k0, ops, B)
        chi = np.array([np.trace(D) for D in Ds])
        err = rep_error(Ds, ops)[0]
        mult, label = {}, None
        if table is not None:
            for ir, tab in table.items():
                mult[ir] = np.mean([np.conj(tab[o["cls"]]) * c for o, c in zip(ops, chi)])
            tot = sum(abs(m - round(m.real)) for m in mult.values())
            err += float(tot)
            hits = [ir for ir, m in mult.items() if abs(m - 1) < 1e-6]
            if len(hits) == 1 and all(abs(m) < 1e-6 for ir, m in mult.items() if ir != hits[0]):
                label = hits[0]
        out.append(dict(bands=s, f=float(np.mean(f[s])), chars=chi, mult=mult, irrep=label, error=err))
    return out


def hom_dimension(ops, chi, kind="quadratic"):
    """``dim Hom_G(Sym^2 V, End E)`` or ``dim Hom_G(V, End E)`` from characters of E (per op), V the plane."""
    chi = np.asarray(chi)
    chiV = np.array([np.trace(o["matrix"]) for o in ops])
    if kind == "linear":
        w = chiV
    elif kind == "quadratic":
        w = 0.5 * (chiV ** 2 + np.array([np.trace(o["matrix"] @ o["matrix"]) for o in ops]))
    else:
        raise ValueError("kind must be 'linear' or 'quadratic'")
    return float(np.mean(w * np.abs(chi) ** 2).real)


# ----------------------------------------------------------------------------- k.p

class KPResult(tuple):
    """Tuple ``(H1, H2)`` with extra attributes ``lam0, C, lam, f0, lam_set``."""

    def __new__(cls, H1, H2, lam0, C, lam, f0, lam_set):
        obj = super().__new__(cls, (H1, H2))
        obj.H1, obj.H2, obj.lam0, obj.C, obj.lam, obj.f0, obj.lam_set = H1, H2, lam0, C, lam, f0, lam_set
        return obj


def kp_second_order(lat, rods, k0, G, bands, solution=None, eps_bg=1.0):
    """Second-order Loewdin k.p expansion of the set ``bands`` at k0 (see module docstring)."""
    k0 = np.asarray(k0, float)
    if solution is None:
        solution = tm_solve(lat, rods, k0, G, None, eps_bg)
    f, Call, B = solution
    lam = (TWO_PI * f / lat.a) ** 2
    bands = np.atleast_1d(bands).astype(int)
    d = len(bands)
    Cd = Call[:, bands]
    lam0 = lam[bands].mean()
    rest = np.setdiff1d(np.arange(len(lam)), bands)
    if np.min(np.abs(lam[rest] - lam0)) < 1e-9 * max(lam0, 1.0):
        raise ValueError("bands is not a complete degenerate set (another state has the same lam)")
    w = 2 * (k0[None, :] + G)                              # H1_i = diag(w_i)
    H1 = np.empty((2, d, d), complex)
    Wn = []
    for i in range(2):
        W = Call.conj().T @ (w[:, i:i + 1] * Cd)           # (N, d): <n|H1_i|a>
        H1[i] = W[bands]
        Wn.append(W[rest])
    ls = lam[bands]
    # Loewdin energy denominators 1/2 (1/(l_a - l_n) + 1/(l_b - l_n)); equals 1/(lam0 - l_n) for a degenerate set
    inv = 1.0 / (ls[None, :] - lam[rest][:, None])                      # (N-d, d)
    Dn = 0.5 * (inv[:, :, None] + inv[:, None, :])                      # (N-d, d, d)
    H2 = np.zeros((2, 2, d, d), complex)
    gram = Cd.conj().T @ Cd
    for i in range(2):
        for j in range(2):
            T = np.einsum("na,nb,nab->ab", Wn[i].conj(), Wn[j], Dn)
            H2[i, j] = 0.5 * (T + T.conj().T)
        H2[i, i] += gram
    return KPResult(H1, H2, lam0, Cd, lam, float(np.mean(f[bands])), ls)


def kp_hamiltonian(kp, q):
    """``diag(lam of the set) + q.H1 + q.H2.q`` as a (d, d) matrix in the basis ``kp.C`` (lam0 I for a degenerate set)."""
    q = np.asarray(q, float)
    H1, H2 = kp
    return np.diag(kp.lam_set).astype(complex) + np.einsum("i,iab->ab", q, H1) + np.einsum("i,j,ijab->ab", q, q, H2)


def adapt_doublet(kp, D_mx):
    """Rotate a C4v doublet into the mirror-adapted basis (index 0 odd under x -> -x). Returns (U, H1a, H2a)."""
    D = 0.5 * (D_mx + D_mx.conj().T)
    w, V = np.linalg.eigh(D)                               # ascending: -1 (odd) first, then +1 (even)
    U = V.astype(complex)
    H2a = np.einsum("ba,ijbc,cd->ijad", U.conj(), kp.H2, U)
    cross = H2a[0, 1] + H2a[1, 0]
    ph = np.angle(cross[0, 1])
    U[:, 1] *= np.exp(-1j * ph)
    H1a = np.einsum("ba,ibc,cd->iad", U.conj(), kp.H1, U)
    H2a = np.einsum("ba,ijbc,cd->ijad", U.conj(), kp.H2, U)
    return U, H1a, H2a


def c4v_constants(H2a):
    """Constants (A, B, C) of ``q.H2a.q = A q^2 + B (qx^2 - qy^2) sz + C qx qy sx`` and the residual norm of the fit."""
    sz = np.diag([1.0, -1.0])
    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    A = 0.25 * np.trace(H2a[0, 0] + H2a[1, 1]).real
    Bc = 0.25 * np.trace(sz @ (H2a[0, 0] - H2a[1, 1])).real
    Cc = 0.5 * np.trace(sx @ (H2a[0, 1] + H2a[1, 0])).real
    rec = np.zeros_like(H2a)
    rec[0, 0] = A * np.eye(2) + Bc * sz
    rec[1, 1] = A * np.eye(2) - Bc * sz
    rec[0, 1] = rec[1, 0] = 0.5 * Cc * sx
    return A, Bc, Cc, float(np.linalg.norm(H2a - rec))


def degenerate_first_order(lam0, C, dB):
    """First-order shift matrix ``-lam0 C^H dB C`` of a degenerate set for ``B -> B + dB`` (A fixed)."""
    M = -lam0 * (C.conj().T @ (dB @ C))
    return 0.5 * (M + M.conj().T)


# ----------------------------------------------------------------------------- Berry phases

def kp_berry_phase(kp, radius, band, nk=64, order=2):
    """Berry phase of eigenvector ``band`` (0-based within the set) of the k.p matrix on a circle of radius ``radius``
    around the expansion point (counterclockwise). ``order=1`` keeps only ``q.H1`` (and the set's own lam), ``order=2``
    adds ``q.H2.q``. Same sign convention and discrete-product definition as ``berry_phase``.
    """
    vs = []
    for t in TWO_PI * np.arange(nk) / nk:
        q = radius * np.array([np.cos(t), np.sin(t)])
        H = kp_hamiltonian(kp, q) if order == 2 else (
            np.diag(kp.lam_set).astype(complex) + np.einsum("i,iab->ab", q, kp.H1))
        vs.append(np.linalg.eigh(0.5 * (H + H.conj().T))[1][:, band])
    prod = 1.0 + 0j
    for j in range(nk):
        prod *= np.vdot(vs[j], vs[(j + 1) % nk])
    g = -np.angle(prod)
    return float(np.pi if g <= -np.pi + 1e-15 else g)


def _loop_vectors(lat, rods, center, radius, nmax, nk, G, gmax, eps_bg):
    center = np.asarray(center, float)
    if G is None:
        G, _ = gvectors(lat, gmax, center)
    B = eps_matrix(lat, rods, G, eps_bg)
    th = TWO_PI * np.arange(nk) / nk
    Cs = []
    for t in th:
        k = center + radius * np.array([np.cos(t), np.sin(t)])
        Cs.append(tm_solve(lat, rods, k, G, nmax, B=B)[1])
    return th, np.array(Cs), B, G


def berry_phase(lat, rods, center, radius, bands, nk=64, gmax=None, G=None, eps_bg=1.0):
    """Berry phase of the band(s) around a counterclockwise circle (module docstring). Returns gamma in (-pi, pi]."""
    bands = np.atleast_1d(bands).astype(int)
    th, Cs, B, G = _loop_vectors(lat, rods, center, radius, int(bands.max()) + 1, nk, G, gmax, eps_bg)
    prod = 1.0 + 0j
    for j in range(nk):
        Cj, Cn = Cs[j][:, bands], Cs[(j + 1) % nk][:, bands]
        prod *= np.linalg.det(Cj.conj().T @ (B @ Cn))
    g = -np.angle(prod)
    return float(np.pi if g <= -np.pi + 1e-15 else g)


def pseudospin_curve(lat, rods, center, radius, band, nk=64, gmax=None, G=None, doublet=None, U=None, doublet_bands=None,
                     eps_bg=1.0):
    """Bloch vector of band ``band`` projected on the doublet at the center; returns (phi (nk,), n (nk, 3))."""
    center = np.asarray(center, float)
    if G is None:
        G, _ = gvectors(lat, gmax, center)
    B = eps_matrix(lat, rods, G, eps_bg)
    if doublet is None:
        if doublet_bands is None:
            raise ValueError("pass doublet (n, 2) or doublet_bands")
        _, C0, _ = tm_solve(lat, rods, center, G, int(max(doublet_bands)) + 1, B=B)
        doublet = C0[:, list(doublet_bands)]
    if U is not None:
        doublet = doublet @ U
    th = TWO_PI * np.arange(nk) / nk
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1.0, -1.0]).astype(complex)
    nvec = np.empty((nk, 3))
    for j, t in enumerate(th):
        k = center + radius * np.array([np.cos(t), np.sin(t)])
        C = tm_solve(lat, rods, k, G, int(band) + 1, B=B)[1][:, int(band)]
        psi = doublet.conj().T @ (B @ C)
        psi = psi / np.linalg.norm(psi)
        nvec[j] = [np.vdot(psi, s @ psi).real for s in (sx, sy, sz)]
    return th, nvec


def winding_number(n, axes=(2, 0)):
    """Winding number of the planar angle ``atan2(n[:, axes[1]], n[:, axes[0]])`` around the closed curve (float)."""
    n = np.asarray(n, float)
    if axes is None:
        _, _, vt = np.linalg.svd(n, full_matrices=False)    # best-fit plane through the origin (great circle)
        e1, e2 = vt[0], vt[1]
        ang = np.arctan2(n @ e2, n @ e1)
        w = np.sum(np.angle(np.exp(1j * np.diff(np.r_[ang, ang[0]])))) / TWO_PI
        return float(abs(w))
    ang = np.arctan2(n[:, axes[1]], n[:, axes[0]])
    dang = np.angle(np.exp(1j * np.diff(np.r_[ang, ang[0]])))
    return float(np.sum(dang) / TWO_PI)


# ----------------------------------------------------------------------------- 1D stacks

def _layer_matrix(eps, d, omega, deriv):
    n = np.sqrt(eps + 0j).real if np.isreal(eps) else np.sqrt(eps)
    k = omega * n
    phi = k * d
    c, s = np.cos(phi), np.sin(phi)
    small = np.abs(phi) < 1e-6
    sk = np.where(small, d * (1 - phi ** 2 / 6), s / np.where(small, 1.0, k))          # sin(phi)/k
    M = np.empty(np.shape(omega) + (2, 2))
    M[..., 0, 0] = c
    M[..., 0, 1] = sk
    M[..., 1, 0] = -k * s
    M[..., 1, 1] = c
    if not deriv:
        return M, None
    dphi = n * d
    sp = np.where(small, -phi / 3, (phi * c - s) / np.where(small, 1.0, phi ** 2))     # d/dphi of sin(phi)/phi
    dM = np.empty_like(M)
    dM[..., 0, 0] = -s * dphi
    dM[..., 0, 1] = d * sp * dphi
    dM[..., 1, 0] = -n * s - k * c * dphi
    dM[..., 1, 1] = -s * dphi
    return M, dM


def transfer_1d(layers, omega, deriv=False):
    """Transfer matrix of ``layers = [(eps, d), ...]`` for ``u'' + omega^2 eps u = 0`` (see module docstring)."""
    omega = np.asarray(omega, float)
    M = np.broadcast_to(np.eye(2), omega.shape + (2, 2)).copy()
    dM = np.zeros_like(M)
    for eps, d in layers:
        Ml, dMl = _layer_matrix(eps, d, omega, deriv)
        if deriv:
            dM = dMl @ M + Ml @ dM
        M = Ml @ M
    return (M, dM) if deriv else M


def bloch_k_1d(layers, omega):
    """Complex Bloch wave number with ``cos(k a) = tr M / 2``, Re(k a) in [0, 2 pi), Im k >= 0."""
    a = sum(d for _, d in layers)
    t = 0.5 * np.trace(transfer_1d(layers, omega), axis1=-2, axis2=-1)
    z = np.arccos(np.asarray(t) + 0j)
    z = np.where(z.imag < 0, -z, z)
    z = np.where(z.real < 0, z + TWO_PI, z)
    return z / a


def band_edges_1d(layers, omega_window, nscan=4000):
    """Edges in the window: returns (omega (m,), s (m,)) with ``tr M/2 = s`` (+1 for k = 0, -1 for k a = pi)."""
    w = np.linspace(omega_window[0], omega_window[1], nscan)
    t = 0.5 * np.trace(transfer_1d(layers, w), axis1=-2, axis2=-1)
    res = []
    for s in (1.0, -1.0):
        g = t - s
        idx = np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]
        for i in idx:
            fn = lambda x: 0.5 * np.trace(transfer_1d(layers, np.float64(x))) - s
            res.append((brentq(fn, w[i], w[i + 1], xtol=1e-14, rtol=1e-14), s))
    res.sort()
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def edge_curvature_1d(layers, omega_edge, s):
    """``T = (1/2) d^2(omega^2)/dk^2`` at a band edge with ``tr M/2 = s``: ``T = -s omega a^2 / (d(tr M/2)/d omega)``."""
    a = sum(d for _, d in layers)
    M, dM = transfer_1d(layers, np.float64(omega_edge), deriv=True)
    dt = 0.5 * np.trace(dM)
    return float(-s * omega_edge * a ** 2 / dt)


def _shifted_cell(layers, x):
    """Layers of one period starting at position x inside the cell (cyclic shift, a layer may be split)."""
    a = sum(d for _, d in layers)
    x = x % a
    out, pos, first = [], 0.0, []
    for eps, d in layers:
        lo, hi = pos, pos + d
        if hi <= x + 1e-15:
            first.append((eps, d))
        elif lo >= x - 1e-15:
            out.append((eps, d))
        else:
            first.append((eps, x - lo))
            out.append((eps, hi - x))
        pos = hi
    return out + first


def _truncate(layers, length):
    """The first ``length`` (< sum d) of a layer list."""
    out, used = [], 0.0
    for eps, d in layers:
        if used >= length - 1e-15:
            break
        out.append((eps, min(d, length - used)))
        used += d
    return out


def finite_stack_modes(layers, N, wall_offset=0.0, omega_window=(0.0, 1.0), nscan=4000, extra=0.0):
    """Dirichlet modes of N cells (plus a further ``extra`` of length) whose left wall lies ``wall_offset`` inside the cell.

    Returns (omega, k complex). ``extra`` in [0, a) continues the periodic structure beyond the N cells, so the right
    wall sits ``wall_offset + N a + extra`` from the cell start (``extra = 0``: both walls at the same cell phase).
    """
    cell = _shifted_cell(layers, wall_offset)
    tail = _truncate(cell, extra) if extra > 0 else []
    w = np.linspace(omega_window[0], omega_window[1], nscan)
    if w[0] == 0.0:
        w[0] = 1e-9 * (w[1] - w[0])

    def total(x):
        M = np.linalg.matrix_power(transfer_1d(cell, x), N)
        return transfer_1d(tail, x) @ M if tail else M

    def u_end(x):
        return total(np.float64(x))[..., 0, 1]
    M = total(w)[:, 0, 1]
    roots = []
    for i in np.nonzero(np.sign(M[:-1]) * np.sign(M[1:]) < 0)[0]:
        roots.append(brentq(u_end, w[i], w[i + 1], xtol=1e-14, rtol=1e-14))
    roots = np.array(roots)
    k = bloch_k_1d(layers, roots) if len(roots) else np.array([], complex)
    return roots, k
