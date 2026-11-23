"""Multiple scattering for a square lattice with a basis of several circular rods per cell, in two dimensions.

Extends ``rodlattice`` (one rod per cell, mu = 1 in the driven response) to N rods per cell with
different centers, radii, permittivities and permeabilities, and adds the T-matrix of an isolated
cluster ("molecule") of such rods. Validated by ``validate_basis.py`` (same folder as the other
scratch scripts of the "more cases" work; see the report that accompanied this file).

Conventions (same as ``rodlattice``)
------------------------------------
Time dependence exp(-i w t), outgoing waves H_l = H_l^(1). Units eps0 = mu0 = c = 1 (w = k0) and, by
default, lattice period a = 1. ``pol="TM"``: u = E_z, interface weight q = 1/mu. ``pol="TE"``: u = H_z,
interface weight q = 1/eps. All quantities are 2D: z is along the rods.

Basis. ``Rod(center=(x, y), radius=R, eps=..., mu=1.0)``; centers are absolute positions (they need not
lie in [0, a)^2, but the rods of the reference cell are the ones listed). Rods of the cell translated by
a lattice vector R_n have centers d_j + R_n. Rods may not overlap, images included (``check_rods``).

Phase reference of the multipole coefficients. About the center d_j + R_n of rod j in cell R_n the
outgoing coefficients are b^(j)_l exp(i kB . R_n) (Bloch condition), so b^(j) is referred to the actual
center d_j of the rod in the reference cell, not to the origin of the cell. In ``driven`` the source
exp(i k . r) is the absolute plane wave, hence b^(j) contains the factor exp(i k . d_j); the macroscopic
quantities do not depend on that reference.

Lattice sums between sublattices. G^(ij)_n, n = -L..L, is the coefficient of J_n(k0 r) exp(i n phi) (r,
phi about the center of rod i) in sum_R exp(i kB.R) H_0(k0 |r + d_i - d_j - R|), R = 0 excluded when
i = j. The regular field incident on rod i is a^(i)_l = sum_j sum_m G^(ij)_{l-m} b^(j)_m, so a Bloch mode
is a zero of I - T G, T = diag(t) the Mie coefficients of the rods. For real k0 and kB below the
diffraction threshold G^(ii) = -I + i N^(ii) and G^(ij) = i N^(ij), with N^(ij)_n = conj(N^(ji)_{-n}).
"""

import dataclasses
import os
import sys

import numpy as np
from scipy.special import hankel1, h1vp, jv, jvp

try:
    import rodlattice as rl
except ImportError:                                   # imported from another folder
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rodlattice as rl

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------- basis description

@dataclasses.dataclass(frozen=True)
class Rod:
    """A circular rod: center (x, y), radius, relative permittivity and permeability (complex allowed)."""
    center: tuple
    radius: float
    eps: complex
    mu: complex = 1.0

    @property
    def d(self):
        return np.asarray(self.center, float)

    @property
    def is_vacuum(self):
        return self.eps == 1 and self.mu == 1


def min_image_distance(d, a=1.0):
    """Distance from the vector d to the nearest lattice point of the square lattice of period a."""
    d = np.asarray(d, float)
    return np.hypot(*(d - a * np.round(d / a)))


def check_rods(rods, a=1.0):
    """Raise ValueError if two rods (or a rod and an image of itself) overlap. Returns the smallest gap."""
    gap = np.inf
    for i, ri in enumerate(rods):
        for j in range(i, len(rods)):
            rj = rods[j]
            dist = a if i == j else min_image_distance(ri.d - rj.d, a)
            gap = min(gap, dist - ri.radius - rj.radius)
    if gap <= 0:
        raise ValueError("rods overlap (including images in neighboring cells)")
    return gap


# ----------------------------------------------------------------------------- lattice sums

def field_sum(k0, kB, pts, a=1.0, L=30, pmax=40, G=None):
    """Phi(x) = sum_R exp(i kB.R) H_0(k0 |x - R|) at points x (n, 2), none on a lattice point."""
    pts = np.atleast_2d(np.asarray(pts, float))
    n = np.round(pts / a)
    xr = pts - n * a
    phase = np.exp(1j * a * (kB[0] * n[:, 0] + kB[1] * n[:, 1]))
    out = np.empty(len(pts), complex)
    r = np.hypot(xr[:, 0], xr[:, 1])
    near = r < 0.4 * a
    if near.any():
        G = rl.lattice_sums(k0, kB, L, a) if G is None else G
        nn = np.arange(-L, L + 1)
        ph = np.arctan2(xr[near, 1], xr[near, 0])
        out[near] = hankel1(0, k0 * r[near]) + (G[None, :] * jv(nn[None, :], k0 * r[near][:, None])
                                                * np.exp(1j * nn[None, :] * ph[:, None])).sum(1)
    far = ~near
    if far.any():
        x, y = xr[far, 0], xr[far, 1]
        cols = np.abs(x) >= np.abs(y)
        F = np.empty(far.sum(), complex)
        if cols.any():
            F[cols] = rl._stack_field(k0, kB[1], kB[0], x[cols], y[cols], a, pmax)
        if (~cols).any():
            F[~cols] = rl._stack_field(k0, kB[0], kB[1], y[~cols], x[~cols], a, pmax)
        out[far] = F
    return out * phase


def basis_sums(k0, kB, di, dj, L, a=1.0, nphi=64):
    """G^(ij)_n, n = -L..L (index n + L): regular coefficients about rod i of the field of sublattice j.

    The field sum_R exp(i kB.R) H_0(k0 |r + di - dj - R|) is sampled on a circle of radius
    min(dmin/2, 2/|k0|) about r = 0 (dmin: distance from di - dj to the nearest lattice point), which
    contains no singularity, and projected on exp(i n phi). For di = dj it is ``rodlattice.lattice_sums``
    (R = 0 excluded).
    """
    di, dj = np.asarray(di, float), np.asarray(dj, float)
    if np.allclose(di, dj):
        return rl.lattice_sums(k0, kB, L, a, nphi)
    dmin = min_image_distance(di - dj, a)
    xi = min(0.5 * dmin, 2.0 / abs(k0))
    phi = (np.arange(nphi) + 0.5) * TWO_PI / nphi
    pts = (di - dj)[None, :] + xi * np.stack([np.cos(phi), np.sin(phi)], 1)
    F = field_sum(k0, kB, pts, a)
    n = np.arange(-L, L + 1)
    c = (F[None, :] * np.exp(-1j * n[:, None] * phi[None, :])).mean(1)
    return c / jv(n, k0 * xi)


def direct_basis_sums(k0, kB, di, dj, L, a=1.0, nmax=90):
    """Brute-force G^(ij)_n for complex k0 (Im k0 > 0), for checking only."""
    j = np.arange(-nmax, nmax + 1)
    X, Y = [v.ravel() for v in np.meshgrid(j * a, j * a, indexing="ij")]
    S = np.stack([X, Y], 1) + (np.asarray(dj) - np.asarray(di))[None, :]
    keep = np.hypot(S[:, 0], S[:, 1]) > 1e-12
    S, R = S[keep], np.stack([X, Y], 1)[keep]
    rr, ph = np.hypot(S[:, 0], S[:, 1]), np.arctan2(S[:, 1], S[:, 0])
    n = np.arange(-L, L + 1)[:, None]
    return (hankel1(n, k0 * rr[None]) * np.exp(-1j * n * ph[None]) * np.exp(1j * (R @ np.asarray(kB)))[None]).sum(1)


def sums_table(k0, kB, rods, L, a=1.0):
    """All G^(ij)_n as an array (N, N, 2L+1). The diagonal is the same for every rod and is computed once."""
    N = len(rods)
    out = np.empty((N, N, 2 * L + 1), complex)
    diag = None
    for i in range(N):
        for j in range(N):
            if i == j:
                if diag is None:
                    diag = rl.lattice_sums(k0, kB, L, a)
                out[i, j] = diag
            else:
                out[i, j] = basis_sums(k0, kB, rods[i].d, rods[j].d, L, a)
    return out


def _toeplitz_block(Gtab, M):
    """Block matrix [[toeplitz(G^(ij))]] of size N(2M+1) from Gtab (N, N, 4M+1)."""
    N = Gtab.shape[0]
    idx = np.arange(-M, M + 1)
    T = Gtab[:, :, (idx[:, None] - idx[None, :]) + 2 * M]            # (N, N, 2M+1, 2M+1) [i, j, l, m]
    return T.transpose(0, 2, 1, 3).reshape(N * (2 * M + 1), N * (2 * M + 1))


def _mie(rods, k0, pol, M):
    """t, as an (N, 2M+1) array."""
    l = np.arange(-M, M + 1)
    return np.array([rl.mie_t(l, k0, r.radius, r.eps, pol, r.mu) for r in rods])


# ----------------------------------------------------------------------------- Bloch modes

def kkr_matrix(k0, kB, rods, pol="TM", M=3, a=1.0):
    """I - T G of size N(2M+1); a Bloch mode exists where it is singular. Index: rod * (2M+1) + (l + M)."""
    Gb = _toeplitz_block(sums_table(k0, kB, rods, 2 * M, a), M)
    t = _mie(rods, k0, pol, M).ravel()
    return np.eye(len(t)) - t[:, None] * Gb


def kkr_hermitian(k0, kB, rods, pol="TM", M=3, a=1.0, scaled=False):
    """Hermitian matrix whose zero eigenvalues mark the Bloch modes (lossless rods, real k0 and kB).

    With t_l = -p_l/(p_l + i q_l) and G = -I + i N (N Hermitian as a block matrix) the mode condition
    I - T G = i (P + iQ)^-1 (Q + P N) is singular where diag(q/p) + N is. Vacuum rods (eps = mu = 1) have
    t = 0, p = 0, decouple from the others and are dropped.

    With ``scaled=True`` the matrix is congruent to the one above, Lam (diag(q/p) + N) Lam with the positive
    diagonal Lam_l = sqrt(|p_l| / (|p_l| + |q_l|)). Sylvester's law keeps the inertia and the null space,
    so the modes are the same, but the entries stay of order one: the unscaled form has q/p of order 1e17
    in the high orders of a small rod and its small eigenvalues are then lost in rounding.
    """
    rods = [r for r in rods if not r.is_vacuum]
    Gtab = sums_table(k0, kB, rods, 2 * M, a)
    N = len(rods)
    delta = (np.arange(-2 * M, 2 * M + 1) == 0)
    for i in range(N):
        Gtab[i, i] = Gtab[i, i] + delta
    Nb = _toeplitz_block(Gtab / 1j, M)
    l = np.arange(-M, M + 1)
    qp = []
    for r in rods:
        if abs(complex(r.eps).imag) > 0 or abs(complex(r.mu).imag) > 0:
            raise ValueError("kkr_hermitian needs lossless (real eps, mu) rods")
        p, q = rl.mie_pq(l, k0, r.radius, r.eps.real if isinstance(r.eps, complex) else r.eps, pol,
                         r.mu.real if isinstance(r.mu, complex) else r.mu)
        qp.append((q, p))
    qv = np.concatenate([x[0] for x in qp])
    pv = np.concatenate([x[1] for x in qp])
    if scaled:
        lam = np.sqrt(np.abs(pv) / (np.abs(pv) + np.abs(qv)))
        H = lam[:, None] * (np.diag(qv / pv) + Nb) * lam[None, :]
    else:
        H = np.diag(qv / pv) + Nb
    return 0.5 * (H + H.conj().T)


def bloch_frequencies(kB, rods, pol="TM", M=3, a=1.0, fmax=0.8, nscan=400, fmin=1e-3):
    """Normalized frequencies a/lambda0 of the Bloch modes at real kB in (fmin, fmax), with multiplicity.

    Inertia counting of ``kkr_hermitian`` along a frequency scan, bisection on each eigenvalue that
    changes sign; roots where an eigenvalue jumps through infinity are discarded (as in
    ``rodlattice.bloch_frequencies``). Scan points where the matrix is not finite are skipped.
    """
    from scipy.optimize import brentq
    nl = sum(not r.is_vacuum for r in rods) * (2 * M + 1)

    def ev(f):
        with np.errstate(all="ignore"):
            Hm = kkr_hermitian(TWO_PI * f / a, kB, rods, pol, M, a, scaled=True)
        if not np.isfinite(Hm).all():
            return np.full(nl, np.nan)
        return np.linalg.eigvalsh(Hm)

    f = np.linspace(fmin, fmax, nscan)
    E = np.array([ev(fi) for fi in f])
    ok = np.isfinite(E).all(axis=1)
    f, E = f[ok], E[ok]
    nneg = (E < 0).sum(axis=1)
    roots = []
    for i in np.nonzero(nneg[1:] != nneg[:-1])[0]:
        for j in range(min(nneg[i], nneg[i + 1]), max(nneg[i], nneg[i + 1])):
            g = lambda x: ev(x)[j]
            if g(f[i]) * g(f[i + 1]) >= 0:
                continue
            fr = brentq(g, f[i], f[i + 1], xtol=1e-13)
            if max(abs(g(fr - 1e-8)), abs(g(fr + 1e-8))) < 1e-3 * min(abs(E[i, j]), abs(E[i + 1, j])) + 1e-6:
                roots.append(fr)
    return np.sort(np.array(roots))


# ----------------------------------------------------------------------------- source-driven response

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


def driven(k0, kvec, rods, pol="TM", M=4, a=1.0, nphi=64):
    """Nonlocal effective permittivity eps_eff(w, k) of the square lattice with a basis of rods.

    The lattice is driven by an external current J0 exp(i k.r) flowing everywhere (inside the rods too),
    with k free of any dispersion relation. Macroscopic fields are the exp(i k.r) Fourier components
    over a unit cell, <X> = (1/A) int_cell X exp(-i k.r). eps_eff is the Landau-Lifshitz tensor,
    D_LL = E + P_LL with P_LL = <P> - (k x <M>)/k0, <P> = (1/A) sum_j (eps_j - 1) int_rod j E exp(-i k.r)
    and <M> = (1/A) sum_j (mu_j - 1) int_rod j H exp(-i k.r). It is a scalar for TM and a 2 x 2 in-plane
    matrix for TE (the E-plane response, with H = H_z). kvec may be complex.

    TE accepts mu_j != 1; TM with mu_j != 1 raises NotImplementedError.

    Returns (eps_eff, info). ``info``: b (N, 2M+1) outgoing coefficients about each rod center (referred
    to the rod center, so they contain exp(i k . d_j)), a (incident regular), c (interior), P (= P_LL
    for the source), E (macroscopic field); TM adds e0, e1 (particular amplitudes, e1 per rod); TE gives
    P and E as 2 x 2 matrices (columns: source along k, source across k) and ``sources`` with b, a, c,
    h0, h1 for each of the two sources.
    """
    if pol == "TM" and any(r.mu != 1 for r in rods):
        raise NotImplementedError("driven: TM with mu != 1 is not implemented")
    kx, ky = complex(kvec[0]), complex(kvec[1])
    k2 = kx * kx + ky * ky
    k = np.sqrt(k2)
    ek = (kx + 1j * ky) / k
    A = a * a
    N = len(rods)
    nl = 2 * M + 1
    l = np.arange(-M, M + 1)
    al = np.abs(l)
    sgn = np.where(l < 0, (-1.0) ** al, 1.0)
    w = (1j ** l) * ek ** (-l.astype(float))              # exp(i k.r) = sum_l w_l J_l(k r) exp(i l phi)

    Gb = _toeplitz_block(sums_table(k0, (kx, ky), rods, 2 * M, a), M)
    Gfull = Gb.reshape(N, nl, N * nl)                      # rows of rod i
    r_ = 0.5 * _GL_X + 0.5                                 # Gauss-Legendre nodes on (0, 1)
    phi = (np.arange(nphi) + 0.5) * TWO_PI / nphi
    cph, sph = np.cos(phi), np.sin(phi)
    eil = np.exp(1j * l[:, None] * phi[None, :])           # (nl, nphi)

    # per-rod arrays
    rod = []
    for j, rd in enumerate(rods):
        R = rd.radius
        NJ, DH, k1, q1 = rl.rod_terms(l, k0, R, rd.eps, pol, rd.mu)
        x0, x1, xk = k0 * R, k1 * R, k * R
        Jx1, dJx1 = sgn * jv(al, x1), sgn * jvp(al, x1)
        Jx0, dJx0 = sgn * jv(al, x0), sgn * jvp(al, x0)
        Hx0, dHx0 = sgn * hankel1(al, x0), sgn * h1vp(al, x0)
        DHs = q1 * k1 * dJx1 * Hx0 - k0 * Jx1 * dHx0
        rr = R * r_
        wr = 0.5 * R * _GL_W * rr                          # radial weight: dr = R/2 dx, times r
        ph_k = np.exp(-1j * (kx * rr[:, None] * cph[None, :] + ky * rr[:, None] * sph[None, :]))
        rod.append(dict(R=R, eps=complex(rd.eps), mu=complex(rd.mu), t=-NJ / DH, k1=k1, q1=q1,
                        Jx1=Jx1, dJx1=dJx1, Jx0=Jx0, dJx0=dJx0, Hx0=Hx0, dHx0=dHx0, DHs=DHs,
                        Jk=lambda n, xk=xk: jv(n, xk), dJk=lambda n, xk=xk: jvp(n, xk),
                        rr=rr, ph_k=ph_k, dA=wr[:, None] * (TWO_PI / nphi),
                        Jin=jv(l[:, None], k1 * rr[None, :]), dJin=jvp(l[:, None], k1 * rr[None, :]),
                        s=np.exp(1j * (kx * rd.d[0] + ky * rd.d[1]))))
    tvec = np.concatenate([r["t"] for r in rod])
    Mmat = np.eye(N * nl) - tvec[:, None] * Gb

    def solve(alphas, rhos):
        """alphas, rhos: (N, nl). Returns b, a, c (N, nl)."""
        sig = np.array([(rhos[j] * r["Jx1"] - alphas[j] * r["q1"] * r["k1"] * r["dJx1"]) / r["DHs"]
                        for j, r in enumerate(rod)])
        b = np.linalg.solve(Mmat, sig.ravel())
        aa = (Gb @ b).reshape(N, nl)
        b = b.reshape(N, nl)
        c = np.empty((N, nl), complex)
        for j, r in enumerate(rod):
            c1 = aa[j] * r["Jx0"] + b[j] * r["Hx0"] + alphas[j]
            c2 = k0 * (aa[j] * r["dJx0"] + b[j] * r["dHx0"]) + rhos[j]
            use1 = np.abs(r["Jx1"]) >= np.abs(r["dJx1"])
            c[j] = np.where(use1, c1 / np.where(use1, r["Jx1"], 1), c2 / np.where(use1, 1, r["q1"] * r["k1"] * r["dJx1"]))
        return b, aa, c

    if pol == "TM":
        s0 = 1j * k0                                       # J0 = 1 along z
        e0 = s0 / (k2 - k0 ** 2)
        alphas, rhos, e1s = [], [], []
        for r in rod:
            e1 = s0 * r["mu"] / (k2 - k0 ** 2 * r["eps"] * r["mu"])
            e1s.append(e1)
            alphas.append(r["s"] * (e0 - e1) * w * r["Jk"](l))
            rhos.append(r["s"] * (e0 - r["q1"] * e1) * w * k * r["dJk"](l))
        b, aa, c = solve(np.array(alphas), np.array(rhos))
        P = 0.0
        for j, r in enumerate(rod):
            hom = np.einsum("l,lr,lp->rp", c[j], r["Jin"], eil)
            P += (r["eps"] - 1) / A * (e1s[j] * np.pi * r["R"] ** 2 + (hom * r["ph_k"] * r["dA"]).sum() / r["s"])
        E = (-s0 - k0 ** 2 * P) / (k0 ** 2 - k2)
        return 1 + P / E, dict(b=b, a=aa, c=c, P=P, E=E, e0=e0, e1=np.array(e1s))

    # TE: two sources, along and across k (see rodlattice.driven)
    Pm, Em, extra = np.zeros((2, 2), complex), np.zeros((2, 2), complex), []
    Kmat = (k0 ** 2 - k2) * np.eye(2) + np.array([[kx * kx, kx * ky], [kx * ky, ky * ky]])
    for i, J0 in enumerate((np.array([kx, ky]) / k, np.array([-ky, kx]) / k)):
        kxJ = kx * J0[1] - ky * J0[0]
        h0 = 1j * kxJ / (k2 - k0 ** 2)
        alphas, rhos, h1s = [], [], []
        for r in rod:
            h1 = 1j * kxJ / (k2 - k0 ** 2 * r["eps"] * r["mu"])
            h1s.append(h1)
            jl = 0.5 * ((J0[1] + 1j * J0[0]) * (1j ** (l - 1)) * ek ** (-(l - 1.0)) * r["Jk"](l - 1)
                        + (J0[1] - 1j * J0[0]) * (1j ** (l + 1)) * ek ** (-(l + 1.0)) * r["Jk"](l + 1))
            alphas.append(r["s"] * (h0 - h1) * w * r["Jk"](l))
            rhos.append(r["s"] * ((h0 - r["q1"] * h1) * w * k * r["dJk"](l) + (1 - r["q1"]) * jl))
        b, aa, c = solve(np.array(alphas), np.array(rhos))
        Pvec = np.zeros(2, complex)
        Mz = 0j
        for j, r in enumerate(rod):
            dHr = np.einsum("l,lr,lp->rp", c[j], r["k1"] * r["dJin"], eil)
            dHp = np.einsum("l,lr,lp->rp", c[j] * 1j * l, r["Jin"] / r["rr"][None, :], eil)
            cx = dHp * cph[None, :] + dHr * sph[None, :]
            cy = dHp * sph[None, :] - dHr * cph[None, :]
            pref = (r["eps"] - 1) / A * 1j / (k0 * r["eps"])
            part = np.pi * r["R"] ** 2 * (1j * h1s[j] * np.array([ky, -kx]) - J0)
            Pvec += pref * (part + np.array([(cx * r["ph_k"] * r["dA"]).sum(), (cy * r["ph_k"] * r["dA"]).sum()]) / r["s"])
            if r["mu"] != 1:
                hom = np.einsum("l,lr,lp->rp", c[j], r["Jin"], eil)
                Mz += (r["mu"] - 1) / A * (h1s[j] * np.pi * r["R"] ** 2 + (hom * r["ph_k"] * r["dA"]).sum() / r["s"])
        Pvec = Pvec - np.array([ky * Mz, -kx * Mz]) / k0          # P_LL = <P> - (k x <M>)/k0, k x z M = (ky M, -kx M)
        Pm[:, i] = Pvec
        Em[:, i] = np.linalg.solve(Kmat, -1j * k0 * J0 - k0 ** 2 * Pm[:, i])
        extra.append(dict(b=b, a=aa, c=c, h0=h0, h1=np.array(h1s)))
    return np.eye(2) + Pm @ np.linalg.inv(Em), dict(P=Pm, E=Em, sources=extra)


# ----------------------------------------------------------------------------- isolated molecule

def regular_translation(k0, d, L, M):
    """R[l, m], l = -M..M (about the new center), m = -L..L (about the old): regular wave translation.

    J_m(k0 r) exp(i m phi) = sum_l R[l, m] J_l(k0 r') exp(i l phi'), r = d + r' (new center at d).
    R[l, m] = J_{m-l}(k0 |d|) exp(i (m - l) phi_d). Exact when l runs over all integers.
    """
    d = np.asarray(d, float)
    dist, ph = np.hypot(*d), np.arctan2(d[1], d[0])
    n = np.arange(-L, L + 1)[None, :] - np.arange(-M, M + 1)[:, None]
    return jv(n, k0 * dist) * np.exp(1j * n * ph)


def outgoing_translation(k0, d, L, M):
    """O[m, l], m = -L..L (about the origin), l = -M..M (about the source at d): outgoing wave translation.

    H_l(k0 r') exp(i l phi') = sum_m O[m, l] H_m(k0 r) exp(i m phi), r' = r - d, valid for r > |d|.
    O[m, l] = J_{m-l}(k0 |d|) exp(-i (m - l) phi_d).
    """
    d = np.asarray(d, float)
    dist, ph = np.hypot(*d), np.arctan2(d[1], d[0])
    n = np.arange(-L, L + 1)[:, None] - np.arange(-M, M + 1)[None, :]
    return jv(n, k0 * dist) * np.exp(-1j * n * ph)


def outgoing_to_regular(k0, D, M):
    """C[l, m], l, m = -M..M: field of an outgoing wave centered at D (about the target center) as regular
    waves about the target center: H_m(k0 r') exp(i m phi') = sum_l C[l, m] J_l(k0 r) exp(i l phi), r' = r - D,
    valid for r < |D|. C[l, m] = H_{l-m}(k0 |D|) exp(-i (l - m) phi_D)."""
    D = np.asarray(D, float)
    dist, ph = np.hypot(*D), np.arctan2(D[1], D[0])
    n = np.arange(-M, M + 1)[:, None] - np.arange(-M, M + 1)[None, :]
    return hankel1(n, k0 * dist) * np.exp(-1j * n * ph)


def molecule_t_matrix(k0, rods, pol="TM", M=6, Lmol=12):
    """T-matrix of a finite cluster of rods about the origin (no lattice), T[m, m'], m, m' = -Lmol..Lmol.

    Incident regular field sum_m' a0_m' J_m'(k0 r) exp(i m' phi) gives the scattered field outside the
    circumscribing circle sum_m B_m H_m(k0 r) exp(i m phi), B = T a0. Built from the multiple scattering
    of the rods (orders |l| <= M each, coupled by ``outgoing_to_regular``), the regular translation of the
    incident wave to each rod and the outgoing translation back to the origin. ``Lmol`` must exceed
    k0 times the cluster radius plus the number of significant orders. Lossless rods give a unitary
    S = I + 2T up to truncation. ``rods`` may include rods of any radius, permittivity and permeability
    (centers are positions relative to the origin; no lattice and no overlap check beyond ``check_cluster``).
    """
    N = len(rods)
    nl = 2 * M + 1
    check_cluster(rods)
    t = _mie(rods, k0, pol, M)
    big = np.eye(N * nl, dtype=complex)
    Rreg = np.zeros((N * nl, 2 * Lmol + 1), complex)
    Oout = np.zeros((2 * Lmol + 1, N * nl), complex)
    for i, ri in enumerate(rods):
        si = slice(i * nl, (i + 1) * nl)
        Rreg[si] = regular_translation(k0, ri.d, Lmol, M)
        Oout[:, si] = outgoing_translation(k0, ri.d, Lmol, M)
        for j, rj in enumerate(rods):
            if i != j:                                     # target i, source j; D = d_j - d_i
                big[si, j * nl:(j + 1) * nl] -= t[i][:, None] * outgoing_to_regular(k0, rj.d - ri.d, M)
    tvec = t.ravel()
    return Oout @ np.linalg.solve(big, tvec[:, None] * Rreg)


def check_cluster(rods):
    """Raise ValueError if two rods of a finite cluster overlap."""
    for i, ri in enumerate(rods):
        for rj in rods[i + 1:]:
            if np.hypot(*(ri.d - rj.d)) <= ri.radius + rj.radius:
                raise ValueError("rods of the cluster overlap")
