"""Multiple scattering for a square lattice of circular rods in two dimensions.

Shared by the notebooks in this folder; notebook 02 derives and validates everything here.

Conventions
-----------
Time dependence exp(-i w t), so outgoing waves are H_l = H_l^(1). Units eps0 = mu0 = c = 1 (w = k0)
and, by default, lattice period a = 1. ``pol="TM"``: u = E_z, interface weight q = 1/mu.
``pol="TE"``: u = H_z, q = 1/eps. Around a rod the field outside is

    u = sum_l [a_l J_l(k0 r) + b_l H_l(k0 r)] exp(i l phi),

and inside u = sum_l c_l J_l(k1 r) exp(i l phi), k1 = k0 sqrt(eps mu). The conditions at r = R are
"u and q du/dr continuous", which give b_l = t_l a_l.

Lattice sums are stored as arrays over n = -L..L (index n + L) and are defined as the coefficients
of the regular part of the quasi-periodic field of monopoles,

    sum_{R != 0} exp(i kB.R) H_0(k0 |r - R|) = sum_n G_n J_n(k0 r) exp(i n phi),

so that G_n = sum_{R != 0} H_n(k0 R) exp(-i n phi_R) exp(i kB.R) and the field incident on a rod from
all the others is a_l = sum_m G_{l-m} b_m.
"""

import numpy as np
from scipy.special import hankel1, h1vp, jv, jvp

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------- single rod

def rod_terms(l, k0, R, eps, pol="TM", mu=1.0):
    """N_J and D_H of order l for a rod (eps, mu) in vacuum; t_l = -N_J / D_H. Also returns (k1, q1)."""
    l = np.abs(np.asarray(l))
    k1 = k0 * np.sqrt(complex(eps) * complex(mu))
    q1 = 1.0 / complex(mu) if pol == "TM" else 1.0 / complex(eps)
    x0, x1 = k0 * R, k1 * R
    NJ = q1 * k1 * jvp(l, x1) * jv(l, x0) - k0 * jv(l, x1) * jvp(l, x0)
    DH = q1 * k1 * jvp(l, x1) * hankel1(l, x0) - k0 * jv(l, x1) * h1vp(l, x0)
    return NJ, DH, k1, q1


def mie_t(l, k0, R, eps, pol="TM", mu=1.0):
    """Scattering coefficients t_l (b_l = t_l a_l) of one rod, for integer orders l (array)."""
    NJ, DH, _, _ = rod_terms(l, k0, R, eps, pol, mu)
    return -NJ / DH


def mie_pq(l, k0, R, eps, pol="TM", mu=1.0):
    """Real p_l, q_l with t_l = -p_l / (p_l + i q_l), for a lossless rod at real k0."""
    l = np.abs(np.asarray(l))
    from scipy.special import yv, yvp
    k1 = k0 * np.sqrt(eps * mu)
    q1 = 1.0 / mu if pol == "TM" else 1.0 / eps
    x0, x1 = k0 * R, k1 * R
    p = q1 * k1 * jvp(l, x1) * jv(l, x0) - k0 * jv(l, x1) * jvp(l, x0)
    q = q1 * k1 * jvp(l, x1) * yv(l, x0) - k0 * jv(l, x1) * yvp(l, x0)
    return p, q


# ----------------------------------------------------------------------------- lattice sums

def _upper(z):
    """Square root with non-negative imaginary part (decaying or outgoing branch)."""
    s = np.sqrt(np.asarray(z, complex))
    return np.where(s.imag < 0, -s, s)


def _circle(k0, a, nphi):
    xi = min(0.5 * a, 2.0 / abs(k0))                 # J_n(k0 xi) has no zero for k0 xi < 2.4
    phi = (np.arange(nphi) + 0.5) * TWO_PI / nphi
    return xi, phi


def _project(F, phi, k0, xi, L):
    """Coefficients G_n, n = -L..L, from samples F of the full quasi-periodic monopole field on r = xi."""
    n = np.arange(-L, L + 1)
    c = (F[None, :] * np.exp(-1j * n[:, None] * phi[None, :])).mean(axis=1)
    c[L] -= hankel1(0, k0 * xi)                      # remove the monopole at the origin itself
    return c / jv(n, k0 * xi)


def _stack_field(k0, kpar, kperp, s, t, a, pmax):
    """Field of the square lattice of monopoles written as a stack of line arrays.

    The arrays run along the coordinate t (Bloch component kpar) and are stacked along s (kperp).
    The plane-wave series converges like exp(-2 pi p (a - |s|) / a) and exp(-2 pi p |s| / a).
    """
    p = np.arange(-pmax, pmax + 1)
    kt = kpar + TWO_PI * p / a
    ks = _upper(k0 ** 2 - kt ** 2)                   # the sum is even in ks, so the branch is free
    s, t = s[:, None], t[:, None]
    wp, wm = np.exp(1j * (ks + kperp) * a), np.exp(1j * (ks - kperp) * a)
    term = (np.exp(1j * ks * np.abs(s))
            + np.exp(1j * (ks * (a - s) + kperp * a)) / (1 - wp)
            + np.exp(1j * (ks * (a + s) - kperp * a)) / (1 - wm))
    return (2.0 / a) * (np.exp(1j * kt * t) / ks * term).sum(axis=1)


def lattice_sums(k0, kB, L, a=1.0, nphi=64, pmax=30):
    """G_n, n = -L..L, of the square lattice at (possibly complex) k0 and Bloch vector kB = (kx, ky).

    The monopole field is sampled on a circle of radius xi < a/2 around the origin and projected on
    exp(i n phi). Each sample uses the stacking direction in which it is farther from the line
    through the origin, so every plane-wave series converges at least like exp(-sqrt(2) pi p xi / a).
    """
    xi, phi = _circle(k0, a, nphi)
    x, y = xi * np.cos(phi), xi * np.sin(phi)
    use_cols = np.abs(x) >= np.abs(y)
    F = np.empty(nphi, complex)
    if use_cols.any():
        F[use_cols] = _stack_field(k0, kB[1], kB[0], x[use_cols], y[use_cols], a, pmax)
    if (~use_cols).any():
        F[~use_cols] = _stack_field(k0, kB[0], kB[1], y[~use_cols], x[~use_cols], a, pmax)
    return _project(F, phi, k0, xi, L)


def column_sums(k0, ky, L, a=1.0, nphi=64, tol=36.0):
    """G_n of ONE line array (rods at y = n a, phases exp(i ky n a)); outgoing branch for the orders."""
    xi, phi = _circle(k0, a, nphi)
    x, y = xi * np.cos(phi), xi * np.sin(phi)
    pmax = int(np.ceil(tol * a / (TWO_PI * np.abs(x).min()))) + 2
    p = np.arange(-pmax, pmax + 1)
    kyp = ky + TWO_PI * p / a
    kxp = _upper(k0 ** 2 - kyp ** 2)
    F = (2.0 / a) * (np.exp(1j * (kyp * y[:, None] + kxp * np.abs(x[:, None]))) / kxp).sum(axis=1)
    return _project(F, phi, k0, xi, L)


def column_translation(k0, ky, X, L, a=1.0, tol=36.0):
    """T_n(X), n = -L..L: regular coefficients at the origin from a line array displaced by X != 0 along x.

    T_n = sum_{n'} exp(i ky n' a) H_n(k0 |R|) exp(-i n phi_R), R = (X, n' a).
    """
    pmax = int(np.ceil(tol * a / (TWO_PI * abs(X)))) + 2
    p = np.arange(-pmax, pmax + 1)
    kyp = ky + TWO_PI * p / a
    kxp = _upper(k0 ** 2 - kyp ** 2)
    n = np.arange(-L, L + 1)[:, None]
    base = -1j * (kxp + 1j * kyp) / k0 if X < 0 else 1j * (kxp - 1j * kyp) / k0
    return (2.0 / a) * ((np.exp(1j * kxp * abs(X)) / kxp)[None, :] * base[None, :] ** (-n)).sum(axis=1)


def direct_lattice_sums(k0, kB, L, a=1.0, nmax=80, column=False):
    """Brute-force G_n for complex k0 (Im k0 > 0), for checking only."""
    j = np.arange(-nmax, nmax + 1)
    if column:
        X, Y = np.zeros_like(j, float), j * a
    else:
        X, Y = [v.ravel() for v in np.meshgrid(j * a, j * a, indexing="ij")]
    keep = (X != 0) | (Y != 0)
    X, Y = X[keep], Y[keep]
    Rr, ph = np.hypot(X, Y), np.arctan2(Y, X)
    n = np.arange(-L, L + 1)[:, None]
    return (hankel1(n, k0 * Rr[None, :]) * np.exp(-1j * n * ph[None, :])
            * np.exp(1j * (kB[0] * X + kB[1] * Y))[None, :]).sum(axis=1)


def _toeplitz(G, M):
    """Matrix [G_{l-m}] for l, m = -M..M from G stored over n = -2M..2M."""
    idx = np.arange(-M, M + 1)
    return G[(idx[:, None] - idx[None, :]) + 2 * M]


# ----------------------------------------------------------------------------- Bloch modes

def kkr_matrix(k0, kB, R, eps, pol="TM", M=3, a=1.0, mu=1.0):
    """I - T G for the square lattice; a Bloch mode exists where it is singular."""
    l = np.arange(-M, M + 1)
    t = mie_t(l, k0, R, eps, pol, mu)
    G = _toeplitz(lattice_sums(k0, kB, 2 * M, a), M)
    return np.eye(2 * M + 1) - t[:, None] * G


def kkr_hermitian(k0, kB, R, eps, pol="TM", M=3, a=1.0, mu=1.0):
    """Hermitian matrix whose zero eigenvalues mark the Bloch modes (lossless rod, real k0 and kB).

    With t_l = -p_l/(p_l + i q_l) and G_n = -delta_n0 + i N_n, where [N_{l-m}] is Hermitian,
    I - T G = i (P + i Q)^{-1} (Q + P N), so a mode exists where diag(q/p) + N is singular.
    """
    l = np.arange(-M, M + 1)
    p, q = mie_pq(l, k0, R, eps, pol, mu)
    G = lattice_sums(k0, kB, 2 * M, a)
    N = _toeplitz((G + (np.arange(-2 * M, 2 * M + 1) == 0)) / 1j, M)
    H = np.diag(q / p) + N
    return 0.5 * (H + H.conj().T)


def bloch_frequencies(kB, R, eps, pol="TM", M=3, a=1.0, fmax=0.8, nscan=400, fmin=1e-3, mu=1.0):
    """Normalized frequencies a/lambda of the Bloch modes at real kB in (fmin, fmax), with multiplicity.

    The number of negative eigenvalues of ``kkr_hermitian`` is tracked along a frequency scan; each
    sorted eigenvalue that changes sign is refined by bisection and kept if it passes through zero
    (it is discarded if it jumps through infinity, which happens at poles of q/p or of the lattice sums).
    """
    from scipy.optimize import brentq
    def ev(f):
        with np.errstate(all="ignore"):
            Hm = kkr_hermitian(TWO_PI * f / a, kB, R, eps, pol, M, a, mu)
        if not np.isfinite(Hm).all():                # exactly on a light cone or on a zero of p_l
            return np.full(2 * M + 1, np.nan)
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


def driven(k0, kvec, R, eps, pol="TM", M=4, a=1.0, nphi=64):
    """Nonlocal effective permittivity eps_eff(w, k) of the square lattice of rods (mu = 1 inside).

    The lattice is driven by an external current J0 exp(i k.r) flowing everywhere, with k free of
    any dispersion relation. Macroscopic fields are the exp(i k.r) Fourier components,
    <P> = (1/A) int_cell P exp(-i k.r), and eps_eff = 1 + <P>/<E> (a scalar for TM, where E is along
    z, and a 2 x 2 in-plane matrix for TE). kvec may be complex. Returns (eps_eff, info).
    """
    kx, ky = complex(kvec[0]), complex(kvec[1])
    k2 = kx * kx + ky * ky
    k = np.sqrt(k2)
    ek = (kx + 1j * ky) / k                              # exp(i phi_k), also for complex k
    A = a * a
    l = np.arange(-M, M + 1)
    al = np.abs(l)
    NJ, DH, k1, q1 = rod_terms(l, k0, R, eps, pol)
    t = -NJ / DH
    x0, x1, xk = k0 * R, k1 * R, k * R
    sgn = np.where(l < 0, (-1.0) ** al, 1.0)             # J_l = sgn * J_|l|
    Jx1, dJx1 = sgn * jv(al, x1), sgn * jvp(al, x1)
    Jx0, dJx0 = sgn * jv(al, x0), sgn * jvp(al, x0)
    Hx0, dHx0 = sgn * hankel1(al, x0), sgn * h1vp(al, x0)
    DHs = q1 * k1 * dJx1 * Hx0 - k0 * Jx1 * dHx0        # same D_H, with signs for negative orders
    w = (1j ** l) * ek ** (-l.astype(float))             # exp(i k.r) = sum_l w_l J_l(k r) exp(i l phi)
    Jk = lambda n: jv(n, xk)
    dJk = lambda n: jvp(n, xk)
    Gm = _toeplitz(lattice_sums(k0, (kx, ky), 2 * M, a), M)
    Mmat = np.eye(2 * M + 1) - t[:, None] * Gm

    # quadrature nodes in the rod
    r = 0.5 * R * (_GL_X + 1.0)
    wr = 0.5 * R * _GL_W * r
    phi = (np.arange(nphi) + 0.5) * TWO_PI / nphi
    cph, sph = np.cos(phi), np.sin(phi)
    rr, pp = r[:, None], phi[None, :]
    ph_k = np.exp(-1j * (kx * rr * np.cos(pp) + ky * rr * np.sin(pp)))    # exp(-i k.r)
    Jin = jv(l[:, None], k1 * r[None, :])                                  # (2M+1, nr)
    dJin = jvp(l[:, None], k1 * r[None, :])
    eil = np.exp(1j * l[:, None] * phi[None, :])                           # (2M+1, nphi)
    dA = wr[:, None] * (TWO_PI / nphi)

    def solve(alpha, rho):
        sigma = (rho * Jx1 - alpha * q1 * k1 * dJx1) / DHs
        b = np.linalg.solve(Mmat, sigma)
        aa = Gm @ b
        c1 = (aa * Jx0 + b * Hx0 + alpha)
        c2 = (k0 * (aa * dJx0 + b * dHx0) + rho)
        use1 = np.abs(Jx1) >= np.abs(dJx1)               # avoid dividing by a Bessel zero
        c = np.where(use1, c1 / np.where(use1, Jx1, 1), c2 / np.where(use1, 1, q1 * k1 * dJx1))
        return b, aa, c

    if pol == "TM":
        s = 1j * k0                                      # J0 = 1 along z
        e0, e1 = s / (k2 - k0 ** 2), s / (k2 - k0 ** 2 * eps)
        alpha = (e0 - e1) * w * Jk(l)
        rho = (e0 - e1) * w * k * dJk(l)
        b, aa, c = solve(alpha, rho)
        hom = np.einsum("l,lr,lp->rp", c, Jin, eil)      # interior field without the particular part
        P = (eps - 1) / A * (e1 * np.pi * R ** 2 + (hom * ph_k * dA).sum())
        E = (-s - k0 ** 2 * P) / (k0 ** 2 - k2)
        return 1 + P / E, dict(b=b, a=aa, c=c, P=P, E=E, e0=e0, e1=e1)

    # Two sources, along and across k: near a Bloch mode only one of them is resonant, which keeps the
    # 2 x 2 inversion below well conditioned (two Cartesian sources would both excite the same mode).
    Pm, Em, extra = np.zeros((2, 2), complex), np.zeros((2, 2), complex), []
    Kmat = (k0 ** 2 - k2) * np.eye(2) + np.array([[kx * kx, kx * ky], [kx * ky, ky * ky]])
    for i, J0 in enumerate((np.array([kx, ky]) / k, np.array([-ky, kx]) / k)):   # along k, then across k
        kxJ = kx * J0[1] - ky * J0[0]
        h0, h1 = 1j * kxJ / (k2 - k0 ** 2), 1j * kxJ / (k2 - k0 ** 2 * eps)
        jl = 0.5 * ((J0[1] + 1j * J0[0]) * (1j ** (l - 1)) * ek ** (-(l - 1.0)) * Jk(l - 1)
                    + (J0[1] - 1j * J0[0]) * (1j ** (l + 1)) * ek ** (-(l + 1.0)) * Jk(l + 1))
        alpha = (h0 - h1) * w * Jk(l)
        rho = (h0 - h1 / eps) * w * k * dJk(l) + (1 - 1 / eps) * jl
        b, aa, c = solve(alpha, rho)
        dHr = np.einsum("l,lr,lp->rp", c, k1 * dJin, eil)                  # d/dr of the homogeneous part
        dHp = np.einsum("l,lr,lp->rp", c * 1j * l, Jin / r[None, :], eil)  # (1/r) d/dphi
        cx = dHp * cph[None, :] + dHr * sph[None, :]                       # curl(H z) = r^ (1/r)dH/dphi - phi^ dH/dr
        cy = dHp * sph[None, :] - dHr * cph[None, :]
        pref = (eps - 1) / A * 1j / (k0 * eps)
        part = np.pi * R ** 2 * (1j * h1 * np.array([ky, -kx]) - J0)
        Pm[:, i] = pref * (part + np.array([(cx * ph_k * dA).sum(), (cy * ph_k * dA).sum()]))
        Em[:, i] = np.linalg.solve(Kmat, -1j * k0 * J0 - k0 ** 2 * Pm[:, i])
        extra.append(dict(b=b, a=aa, c=c, h0=h0, h1=h1))
    return np.eye(2) + Pm @ np.linalg.inv(Em), dict(P=Pm, E=Em, sources=extra)


# ----------------------------------------------------------------------------- finite slab

def slab(k0, ky, nlayers, R, eps, pol="TM", M=4, a=1.0, mu=1.0):
    """Plane wave exp(i(kx x + ky y)) on a slab of `nlayers` line arrays of rods at x = 0, a, ..., (nlayers-1) a.

    Returns the amplitudes (r, t) of the zeroth diffraction order, referred to the faces of the slab at
    x = -a/2 and x = (nlayers - 1/2) a, the full lists (r_p, t_p, kxp) for the orders p = -2..2, and the
    multipole coefficients b[layer, l].
    """
    l = np.arange(-M, M + 1)
    nl = 2 * M + 1
    t = mie_t(l, k0, R, eps, pol, mu)
    kx0 = _upper(k0 ** 2 - ky ** 2 + 0j)
    w = (1j ** l) * ((kx0 + 1j * ky) / k0) ** (-l.astype(float))
    Gself = _toeplitz(column_sums(k0, ky, 2 * M, a), M)
    Tcache = {}
    big = np.zeros((nlayers * nl, nlayers * nl), complex)
    rhs = np.zeros(nlayers * nl, complex)
    for j in range(nlayers):
        sj = slice(j * nl, (j + 1) * nl)
        rhs[sj] = t * np.exp(1j * kx0 * j * a) * w
        for i in range(nlayers):
            si = slice(i * nl, (i + 1) * nl)
            if i == j:
                big[sj, si] = np.eye(nl) - t[:, None] * Gself
            else:
                d = i - j
                if d not in Tcache:
                    Tcache[d] = _toeplitz(column_translation(k0, ky, d * a, 2 * M, a), M)
                big[sj, si] = -t[:, None] * Tcache[d]
    b = np.linalg.solve(big, rhs).reshape(nlayers, nl)
    orders = np.arange(-2, 3)
    kyp = ky + TWO_PI * orders / a
    kxp = _upper(k0 ** 2 - kyp ** 2 + 0j)
    xj = np.arange(nlayers) * a
    fwd = (-1j * (kxp[:, None] + 1j * kyp[:, None]) / k0) ** l[None, :]     # (-i e^{i theta_p})^l
    bwd = (1j * (kxp[:, None] - 1j * kyp[:, None]) / k0) ** l[None, :]      # (i e^{-i theta_p})^l
    tp = (2 / (a * kxp)) * np.einsum("pl,jl,pj->p", fwd, b, np.exp(-1j * kxp[:, None] * xj[None, :]))
    rp = (2 / (a * kxp)) * np.einsum("pl,jl,pj->p", bwd, b, np.exp(1j * kxp[:, None] * xj[None, :]))
    tp[2] += 1.0
    d = nlayers * a
    r0 = rp[2] * np.exp(1j * kx0 * a)                    # reflection referred to x = -a/2
    t0 = tp[2] * np.exp(1j * kx0 * d)                    # transmission from x = -a/2 to x = d - a/2
    return r0, t0, (rp, tp, kxp), b


# ----------------------------------------------------------------------------- finite cluster

def cluster(k0, centers, R, eps, pol="TM", M=3, mu=1.0, phi_inc=0.0):
    """Multiple scattering of a plane wave by a finite set of identical rods. Returns b[rod, l] (l = -M..M)."""
    centers = np.asarray(centers, float)
    nrod, nl = len(centers), 2 * M + 1
    l = np.arange(-M, M + 1)
    t = mie_t(l, k0, R, eps, pol, mu)
    d = centers[None, :, :] - centers[:, None, :]        # d[j, i] = r_i - r_j (source relative to target)
    dist = np.hypot(d[..., 0], d[..., 1])
    ang = np.arctan2(d[..., 1], d[..., 0])
    np.fill_diagonal(dist, 1.0)
    n = l[:, None] - l[None, :]                          # (target order, source order)
    blocks = hankel1(n[None, None], k0 * dist[:, :, None, None]) * np.exp(-1j * n[None, None] * ang[:, :, None, None])
    blocks[np.arange(nrod), np.arange(nrod)] = 0.0
    big = np.eye(nrod * nl) - (t[None, None, :, None] * blocks).transpose(0, 2, 1, 3).reshape(nrod * nl, nrod * nl)
    kdir = np.array([np.cos(phi_inc), np.sin(phi_inc)])
    inc = np.exp(1j * k0 * centers @ kdir)[:, None] * ((1j ** l) * np.exp(-1j * l * phi_inc))[None, :]
    b = np.linalg.solve(big, (t[None, :] * inc).ravel()).reshape(nrod, nl)
    return b


def cluster_far_field(k0, centers, b, angles):
    """Far-field amplitude f(theta) with u_sc ~ sqrt(2/(pi k0 r)) exp(i(k0 r - pi/4)) f(theta)."""
    centers = np.asarray(centers, float)
    M = (b.shape[1] - 1) // 2
    l = np.arange(-M, M + 1)
    ang = np.asarray(angles, float)
    rhat = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    phase = np.exp(-1j * k0 * rhat @ centers.T)                       # (nang, nrod)
    harm = ((-1j) ** l)[None, :] * np.exp(1j * ang[:, None] * l[None, :])
    return np.einsum("aj,jl,al->a", phase, b, harm)
