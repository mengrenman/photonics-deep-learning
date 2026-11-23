"""Spatial dispersion of a periodic stack of layers, exactly.

Shared by notebook 05. The layers are stacked along x with period a = 1; everything is uniform in y and z. The wave
vector k = (kx, ky) lies in the x-y plane. ``layers`` is a list of (thickness, permittivity) pairs from x = 0 to
x = a, with the thicknesses adding up to one and the permittivities possibly complex (and possibly negative).

Conventions (as in notebooks 01 to 04): time dependence exp(-i w t), eps0 = mu0 = c = 1 so w = k0, mu = 1 everywhere.
TM: E = E_z (parallel to every layer, s polarization). TE: H = H_z, E in the x-y plane (p polarization), with E_x
normal to the layers.

The source-driven problem. An external current J0 exp(i(kx x + ky y)) fills all space. With the field written as
exp(i kx x) times a periodic function u(x), each layer obeys a linear ordinary differential equation with constant
coefficients and a constant source, so the whole cell is a product of matrix exponentials of an augmented matrix
(the field, its flux, the running integrals that give the macroscopic averages, and the constant 1). The Bloch
condition is a 2 x 2 linear system. Nothing is truncated: the result is exact to rounding, also for complex k0, kx, ky.

    TM  state (u, u', int u, 1) with E_z = u exp(i kx x),  u'' + 2 i kx u' + (k0^2 eps - k^2) u = -i k0 J0.
    TE  state (u, g, int u, int g, 1) with H_z = u exp(i kx x) and g exp(i kx x) = (H_z' + J_y)/eps, which is
        proportional to E_y. H_z'' + q^2 H_z = i (ky J_x - kx J_y) exp(i kx x), q^2 = k0^2 eps - ky^2.
        Interface conditions: H_z and (H_z' + J_y)/eps continuous (E_y tangential).

Macroscopic fields are the exp(i k.r) Fourier components over one cell, <F> = int_cell F exp(-i k.r) dx / a.
eps(w, k) = 1 + <P>/<E>, with <P> = int (eps(x) - 1) E exp(-i k.r) dx. For TE two sources, along and across k, give
the 2 x 2 tensor in x-y components, as ``rodlattice.driven``.

``driven_layered_modes`` solves the same problem as a particular plane wave plus exp(+-i q x) in every layer, with a
2N x 2N linear system for the amplitudes. It is the independent check of ``driven_layered`` and is not used otherwise.
"""

import numpy as np
from scipy.linalg import expm

TWO_PI = 2.0 * np.pi


def _split(layers):
    d = np.array([l[0] for l in layers], float)
    e = np.array([l[1] for l in layers], complex)
    if abs(d.sum() - 1.0) > 1e-12:
        raise ValueError("layer thicknesses must add up to the period a = 1")
    return d, e


def local_eps(layers):
    """Static effective-medium values (eps along the layers, eps normal to the layers) = (<eps>, 1/<1/eps>)."""
    d, e = _split(layers)
    return (d * e).sum(), 1.0 / (d / e).sum()


def taylor(fun, rho, n=16, orders=(0, 1, 2, 3, 4)):
    """Taylor coefficients at 0 of an analytic function (array valued), from n samples on the circle |z| = rho."""
    th = (np.arange(n) + 0.5) * TWO_PI / n
    vals = np.array([np.atleast_1d(fun(rho * np.exp(1j * t))) for t in th])
    return np.array([(vals * np.exp(-1j * m * th)[:, None]).mean(axis=0) / rho ** m for m in orders])


# ----------------------------------------------------------------------------- source-driven response (matrix exponentials)

def driven_layered(k0, kx, ky, layers, pol="TM", info=False):
    """Exact eps(w, k) of the stack: a complex number for TM (eps_zz), a 2 x 2 matrix in x, y components for TE.

    With ``info`` also returns a dict holding <E> from direct averaging ("E_direct") and from the averaged Maxwell
    equation ("E_maxwell"), and <P>; for TE the entries are 2 x 2 matrices whose columns belong to the two sources.
    """
    d, e = _split(layers)
    kx, ky, k0 = complex(kx), complex(ky), complex(k0)
    k2 = kx * kx + ky * ky
    if pol == "TM":
        J0 = 1.0
        As = []
        for ej in e:
            A = np.zeros((4, 4), complex)
            A[0, 1] = 1.0
            A[1, 0] = -(k0 ** 2 * ej - k2)
            A[1, 1] = -2j * kx
            A[1, 3] = -1j * k0 * J0
            A[2, 0] = 1.0
            As.append(A)
        Xs = [expm(A * dj) for A, dj in zip(As, d)]
        T = np.eye(4, dtype=complex)
        for X in Xs:
            T = X @ T
        s0 = np.linalg.solve(T[:2, :2] - np.eye(2), -T[:2, 3])
        s = np.array([s0[0], s0[1], 0.0, 1.0], complex)
        Iu = []
        for X in Xs:
            sn = X @ s
            Iu.append(sn[2])
            s = np.array([sn[0], sn[1], 0.0, 1.0], complex)
        Iu = np.array(Iu)
        E_dir, P = Iu.sum(), ((e - 1.0) * Iu).sum()
        eps_eff = 1.0 + P / E_dir
        if not info:
            return eps_eff
        E_max = (-1j * k0 * J0 - k0 ** 2 * P) / (k0 ** 2 - k2)
        return eps_eff, dict(E_direct=E_dir, E_maxwell=E_max, P=P)

    # TE: two sources, along k and across k
    k = np.sqrt(k2)
    sources = [np.array([kx, ky]) / k, np.array([-ky, kx]) / k] if abs(k) > 0 else [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    Pm, Ed, Em = np.zeros((2, 2), complex), np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    Kmat = (k0 ** 2 - k2) * np.eye(2) + np.array([[kx * kx, kx * ky], [kx * ky, ky * ky]])
    for i, J in enumerate(sources):
        Jx, Jy = J
        Xs = []
        for ej, dj in zip(e, d):
            q2 = k0 ** 2 * ej - ky ** 2
            A = np.zeros((5, 5), complex)
            A[0, 0] = -1j * kx
            A[0, 1] = ej
            A[0, 4] = -Jy
            A[1, 0] = -q2 / ej
            A[1, 1] = -1j * kx
            A[1, 4] = 1j * ky * Jx / ej
            A[2, 0] = 1.0
            A[3, 1] = 1.0
            Xs.append(expm(A * dj))
        T = np.eye(5, dtype=complex)
        for X in Xs:
            T = X @ T
        s0 = np.linalg.solve(T[:2, :2] - np.eye(2), -T[:2, 4])
        s = np.array([s0[0], s0[1], 0.0, 0.0, 1.0], complex)
        Ex = Ey = Px = Py = 0.0
        for X, ej, dj in zip(Xs, e, d):
            sn = X @ s
            Iu_j, Ig_j = sn[2], sn[3]
            ex = (1j / k0) * (1j * ky * Iu_j - Jx * dj) / ej
            ey = (-1j / k0) * Ig_j
            Ex, Ey, Px, Py = Ex + ex, Ey + ey, Px + (ej - 1.0) * ex, Py + (ej - 1.0) * ey
            s = np.array([sn[0], sn[1], 0.0, 0.0, 1.0], complex)
        Ed[:, i] = (Ex, Ey)
        Pm[:, i] = (Px, Py)
        Em[:, i] = np.linalg.solve(Kmat, -1j * k0 * np.array(J) - k0 ** 2 * Pm[:, i])
    eps_eff = np.eye(2) + Pm @ np.linalg.inv(Ed)
    if not info:
        return eps_eff
    return eps_eff, dict(E_direct=Ed, E_maxwell=Em, P=Pm)


# ----------------------------------------------------------------------------- independent solution: particular wave + exp(+-i q x)

def _phi(z, d):
    """int_0^d exp(i z s) ds, safe at z = 0."""
    return d if abs(z * d) < 1e-9 else np.expm1(1j * z * d) / (1j * z)


def driven_layered_modes(k0, kx, ky, layers, pol="TM", info=False):
    """Same quantity as ``driven_layered`` from a particular plane wave plus exp(+-i q_j x) in each layer.

    Unknown amplitudes (a_j, b_j) per layer, 2N conditions: continuity of the field and of its flux across every interface
    and the Bloch condition across the cell boundary. Fails (division by zero) if some q_j is exactly zero.
    """
    d, e = _split(layers)
    N = len(d)
    x0 = np.concatenate([[0.0], np.cumsum(d)])
    kx, ky, k0 = complex(kx), complex(ky), complex(k0)
    k2 = kx * kx + ky * ky
    q = np.sqrt(k0 ** 2 * e - ky ** 2 + 0j)

    def one_source(J):
        # field F: E_z (TM) or H_z (TE); flux: E_z' (TM) or (H_z' + J_y)/eps (TE); both at the right end of layer j
        if pol == "TM":
            h = 1j * k0 * J / (k2 - k0 ** 2 * e)
            w = np.ones(N, complex)                   # flux = w (F' + jy)
            jy = 0.0
        else:
            Jx, Jy = J
            h = 1j * (kx * Jy - ky * Jx) / (k2 - k0 ** 2 * e)
            w = 1.0 / e
            jy = Jy
        # F_j(s) = h_j exp(i kx (x0_j + s)) + a_j exp(i q s) + b_j exp(-i q s), s from 0 to d_j
        def end_values(j, s):
            ph = np.exp(1j * kx * (x0[j] + s))
            Fp = h[j] * ph
            dFp = 1j * kx * h[j] * ph
            cs, sn = np.exp(1j * q[j] * s), np.exp(-1j * q[j] * s)
            Fa, Fb = cs, sn
            dFa, dFb = 1j * q[j] * cs, -1j * q[j] * sn
            flux_p = w[j] * (dFp + jy * ph)
            return Fp, flux_p, (Fa, w[j] * dFa), (Fb, w[j] * dFb)
        Amat = np.zeros((2 * N, 2 * N), complex)
        rhs = np.zeros(2 * N, complex)
        for j in range(N):                           # right end of layer j against left end of layer j+1 (Bloch for the last)
            jn = (j + 1) % N
            Fr, fr, (Fa, fa), (Fb, fb) = end_values(j, d[j])
            Fl, fl, (Ga, ga), (Gb, gb) = end_values(jn, 0.0)
            fac = np.exp(1j * kx) if j == N - 1 else 1.0
            for row, (vr, ar, br, vl, al, bl) in enumerate([(Fr, Fa, Fb, Fl, Ga, Gb), (fr, fa, fb, fl, ga, gb)]):
                r = 2 * j + row
                Amat[r, 2 * j], Amat[r, 2 * j + 1] = ar, br
                Amat[r, 2 * jn] -= fac * al
                Amat[r, 2 * jn + 1] -= fac * bl
                rhs[r] = -(vr - fac * vl)
        ab = np.linalg.solve(Amat, rhs)
        # averages of F e^{-i kx x} and of the flux
        IF, Iflux = np.zeros(N, complex), np.zeros(N, complex)
        for j in range(N):
            a, b = ab[2 * j], ab[2 * j + 1]
            ph0 = np.exp(-1j * kx * x0[j])
            IF[j] = h[j] * d[j] + ph0 * (a * _phi(q[j] - kx, d[j]) + b * _phi(-q[j] - kx, d[j]))
            dF = ph0 * (1j * q[j] * a * _phi(q[j] - kx, d[j]) - 1j * q[j] * b * _phi(-q[j] - kx, d[j])) + 1j * kx * h[j] * d[j]
            Iflux[j] = w[j] * (dF + jy * d[j])
        return IF, Iflux

    if pol == "TM":
        IE, _ = one_source(1.0)
        E_dir, P = IE.sum(), ((e - 1.0) * IE).sum()
        eps_eff = 1.0 + P / E_dir
        return (eps_eff, dict(E_direct=E_dir, P=P)) if info else eps_eff
    k = np.sqrt(k2)
    sources = [np.array([kx, ky]) / k, np.array([-ky, kx]) / k]
    Pm, Ed = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    for i, J in enumerate(sources):
        IH, Ig = one_source(J)
        ex = (1j / k0) * (1j * ky * IH - J[0] * d) / e
        ey = (-1j / k0) * Ig
        Ed[:, i] = (ex.sum(), ey.sum())
        Pm[:, i] = (((e - 1) * ex).sum(), ((e - 1) * ey).sum())
    eps_eff = np.eye(2) + Pm @ np.linalg.inv(Ed)
    return (eps_eff, dict(E_direct=Ed, P=Pm)) if info else eps_eff


# ----------------------------------------------------------------------------- Bloch waves of the stack

def _layer_transfer(k0, ky, ej, dj, pol):
    """Transfer matrix of one layer for the state (F, flux): E_z and E_z' (TM), H_z and H_z'/eps (TE)."""
    q = np.sqrt(complex(k0) ** 2 * ej - complex(ky) ** 2)
    c = np.cos(q * dj)
    s_over_q = dj if abs(q * dj) < 1e-12 else np.sin(q * dj) / q
    qs = -q * np.sin(q * dj)                          # = -q^2 sin(q d)/q
    if pol == "TM":
        return np.array([[c, s_over_q], [qs, c]])
    return np.array([[c, ej * s_over_q], [qs / ej, c]])


def cell_transfer(k0, ky, layers, pol="TM"):
    d, e = _split(layers)
    M = np.eye(2, dtype=complex)
    for dj, ej in zip(d, e):
        M = _layer_transfer(k0, ky, ej, dj, pol) @ M
    return M


def bloch_kx(k0, ky, layers, pol="TM"):
    """Bloch wavenumber kx a in the first zone from cos(kx a) = tr M / 2 (real in a pass band; complex in a gap)."""
    return np.arccos(0.5 * np.trace(cell_transfer(k0, ky, layers, pol)))


def symmetric_cell(layers, split):
    """Cell that is symmetric about its center: layer number ``split`` cut in half at the two ends of the cell."""
    L = list(layers[split:]) + list(layers[:split])
    d0, e0 = L[0]
    return [(d0 / 2, e0)] + L[1:] + [(d0 / 2, e0)]


def bloch_admittance(k0, ky, layers, pol="TM", cut=0):
    """Admittance Y = <H_tangential>/<E_tangential> of the Bloch wave at the boundary of the cell cut at layer ``cut``.

    Defined so that a plane wave traveling in +x in a uniform medium has Y = q/k0 (TM, tangential H_y over E_z, sign
    flipped) or Y = eps k0/q (TE, H_z over E_y); at normal incidence both are the index n. Returns (Y, kx a).
    """
    cell = symmetric_cell(layers, cut)
    M = cell_transfer(k0, ky, cell, pol)
    kxa = np.arccos(0.5 * np.trace(M))
    ratio = (np.exp(1j * kxa) - M[0, 0]) / M[0, 1]     # flux / F of the Bloch wave
    Y = -1j * ratio / k0 if pol == "TM" else 1j * k0 / ratio
    return Y, kxa


# ----------------------------------------------------------------------------- second-order coefficients

def tm_second_order(k0, layers, rho):
    """Taylor coefficients of eps_zz: (eps at k = 0, kx^2 coefficient, ky^2 coefficient)."""
    cx = taylor(lambda kap: driven_layered(k0, kap, 0.0, layers, "TM"), rho, orders=(0, 2))
    cy = taylor(lambda kap: driven_layered(k0, 0.0, kap, layers, "TM"), rho, orders=(0, 2))
    return dict(eps0=cx[0, 0], cx=cx[1, 0], cy=cy[1, 0])


def te_second_order(k0, layers, rho):
    """The five second-order coefficients of the TE tensor of a stack (symmetry of the stack: mirrors x -> -x, y -> -y).

    eps_xx = e_xx + A kx^2 + B ky^2, eps_yy = e_yy + C kx^2 + D ky^2, eps_xy = E kx ky. Also returns a redundancy check:
    the kx ky coefficient of eps_xy from an oblique direction and the residuals of the structure the mirrors impose.
    """
    cx = taylor(lambda kap: driven_layered(k0, kap, 0.0, layers, "TE").ravel(), rho, orders=(0, 1, 2))
    cy = taylor(lambda kap: driven_layered(k0, 0.0, kap, layers, "TE").ravel(), rho, orders=(0, 1, 2))
    s2 = 1 / np.sqrt(2)
    co = taylor(lambda kap: driven_layered(k0, s2 * kap, s2 * kap, layers, "TE").ravel(), rho, orders=(0, 1, 2))
    # raveled order: xx, xy, yx, yy
    out = dict(e_xx=cx[0, 0], e_yy=cx[0, 3], A=cx[2, 0], C=cx[2, 3], B=cy[2, 0], D=cy[2, 3], E=2 * co[2, 1])
    out["resid"] = max(abs(cx[0, 1]), abs(cx[2, 1]), abs(cy[0, 1]), abs(cy[2, 1]), abs(cx[1]).max(), abs(cy[1]).max(),
                       abs(co[2, 0] - 0.5 * (out["A"] + out["B"])), abs(co[2, 3] - 0.5 * (out["C"] + out["D"])),
                       abs(co[2, 1] - co[2, 2]))
    return out


def te_model_kx(k0, ky, co):
    """Roots kx^2 of det[k0^2 eps(k) - (k^2 I - k k^T)] = 0 for the second-order tensor of ``te_second_order``."""
    P = np.polynomial.polynomial
    kx2 = np.array([0.0, 1.0])                                       # kx^2 as a polynomial in the variable t = kx^2
    exx = P.polyadd([co["e_xx"] + co["B"] * ky ** 2], co["A"] * kx2)
    eyy = P.polyadd([co["e_yy"] + co["D"] * ky ** 2], co["C"] * kx2)
    mxx = P.polysub(k0 ** 2 * exx, [ky ** 2])
    myy = P.polysub(k0 ** 2 * eyy, kx2)
    # (k0^2 E kx ky + kx ky)^2 = ky^2 (k0^2 E + 1)^2 kx^2
    mxy2 = ky ** 2 * (k0 ** 2 * co["E"] + 1.0) ** 2 * kx2
    det = np.asarray(P.polysub(P.polymul(mxx, myy), mxy2), complex)
    det = P.polytrim(np.where(abs(det) > 1e-10 * abs(det).max(), det, 0.0))   # A = 0 exactly; drop its rounding residue
    return np.sort_complex(P.polyroots(det))
