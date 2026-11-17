"""Planar multilayer solvers in several matrix formulations, written to compare their numerical stability.

Shared by the notebooks in notebooks/omega_d. Nothing here is tuned for speed.

Conventions
-----------
Time dependence exp(-i omega t). Lengths are in free-space wavelengths, so k0 = 2 pi and a thickness d
is the product "Omega d" in units of 2 pi c. Layers are stacked along z; light travels in the x-z plane
with in-plane wavenumber kx = k0 beta. Medium "a" is on the entrance side (z < 0), medium "b" on the
exit side.

One channel (isotropic, nonmagnetic layers, one polarization). The field is u (E_y for s, H_y for p) and
v = u'/w with w = 1 for s and w = eps for p; both are continuous at interfaces. In a layer
u'' + q^2 u = 0 with q = k0 sqrt(eps - beta^2), Im q >= 0, and eta = q/w, so that a forward wave has
v = i eta u. The 2x2 matrices of one-channel functions are tuples (a, b, c, d) of arrays that broadcast
together, standing for [[a, b], [c, d]].

Many channels. A z-invariant region is described by its modes,
    Psi(z) = Fp exp(i qp z) a+ + Fm exp(i qm z) a-,
with Psi the 2n field components that are continuous at interfaces, Fp and Fm of shape (2n, n), and
"forward" modes (qp) either decaying or carrying power toward +z.

Reference values use mpmath, with K0 taken as the double 2*pi so that the reference solves exactly the
problem the double-precision code is given.
"""
import numpy as np

TWO_PI = 2 * np.pi
K0 = TWO_PI                      # lambda_0 = 1
EPS_MACH = np.finfo(float).eps


# --------------------------------------------------------------------------------------------------
# One channel
# --------------------------------------------------------------------------------------------------

def kz(eps, beta, k0=K0):
    """q = k0 sqrt(eps - beta^2), on the branch Im q >= 0 (Re q >= 0 when q is real)."""
    q = k0 * np.sqrt(np.asarray(eps, complex) - np.asarray(beta) ** 2 + 0j)
    return np.where(q.imag < 0, -q, q)


class Stack:
    """Isotropic multilayer: half-space eps_a, layers [(eps, d), ...] in order from a to b, half-space eps_b."""

    def __init__(self, eps_a, layers, eps_b):
        self.eps_a, self.eps_b = eps_a, eps_b
        self.layers = [(e, d) for e, d in layers]

    def params(self, beta, pol="s", k0=K0):
        """(eta_a, [(q, eta, d), ...], eta_b) for the polarization pol ('s' or 'p')."""
        def eta(eps):
            q = kz(eps, beta, k0)
            return q, (q if pol == "s" else q / eps)
        return eta(self.eps_a)[1], [eta(e) + (d,) for e, d in self.layers], eta(self.eps_b)[1]

    def interfaces(self):
        """z of the interfaces, starting at 0."""
        return np.concatenate([[0.0], np.cumsum([d for _, d in self.layers])])


def mm(A, B):
    """Product of two 2x2 matrices given as tuples."""
    a, b, c, d = A
    e, f, g, h = B
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def T_layer(q, eta, d):
    """Transfer matrix of a layer for (u, v), entrance face to exit face."""
    c, s = np.cos(q * d), np.sin(q * d)
    return (c, s / eta, -eta * s, c)


def T_layer_inv(q, eta, d):
    """Inverse of T_layer (exit face to entrance face), written out; no numerical inversion."""
    c, s = np.cos(q * d), np.sin(q * d)
    return (c, -s / eta, eta * s, c)


def T_forward(layers):
    """M with psi(exit) = M psi(entrance)."""
    M = (1.0, 0.0, 0.0, 1.0)
    for q, eta, d in layers:
        M = mm(T_layer(q, eta, d), M)
    return M


def T_backward(layers):
    """N with psi(entrance) = N psi(exit); this is the characteristic matrix of Born and Wolf."""
    N = (1.0, 0.0, 0.0, 1.0)
    for q, eta, d in layers:
        N = mm(N, T_layer_inv(q, eta, d))
    return N


def rt_from_forward(M, eta_a, eta_b, use_det=False):
    """(r, t, r', t') from the entrance-to-exit matrix.

    With use_det=False the transmissions are obtained by substituting r back, t = A (1 + r) + i eta_a B (1 - r),
    which is the one-channel form of the Schur complement K11 - K12 K22^-1 K21. With use_det=True they use
    det M = 1 analytically, t = 2 i eta_a / den.
    """
    A, B, C, D = M
    x, y = 1j * eta_a * D + eta_a * eta_b * B, 1j * eta_b * A - C
    den = x + y
    r = (x - y) / den
    rp = (1j * eta_b * A + eta_a * eta_b * B + C - 1j * eta_a * D) / den
    if use_det:
        t, tp = 2j * eta_a / den, 2j * eta_b / den
    else:
        t = A * (1 + r) + 1j * eta_a * B * (1 - r)
        tp = (1 + rp) / (A - 1j * eta_a * B)
    return r, t, rp, tp


def rt_from_backward(N, eta_a, eta_b):
    """(r, t) from the exit-to-entrance matrix: only its action on the outgoing vector (1, i eta_b) is used."""
    A, B, C, D = N
    x = A + 1j * eta_b * B
    y = (C + 1j * eta_b * D) / (1j * eta_a)
    return (x - y) / (x + y), 2 / (x + y)


def _pow2(x):
    """Binary exponent e with 0.5 <= |x| 2^-e < 1 (e = 0 where x is zero), as an integer array."""
    return np.frexp(np.where(x == 0, 1.0, x))[1]


def _scale(z, e):
    """z 2^-e, exact."""
    z = np.asarray(z, complex)
    return np.ldexp(z.real, -e) + 1j * np.ldexp(z.imag, -e)


def rt_vector(layers, eta_a, eta_b, modal=False):
    """(r, tm, te) with t = tm 2^te, by carrying one vector from the exit to the entrance.

    The vector is rescaled by a power of two after every layer and the factor exp(i q d) of each layer is
    split into a mantissa and a binary exponent, so nothing overflows or underflows and the scale is exact.
    With modal=False each layer is applied as the matrix of cosines and sines: the arithmetic of
    rt_from_backward(T_backward(...)) on one column. With modal=True the vector is first split into the
    forward and backward waves of the layer, each is multiplied by its own exponential, and the fields are
    put back together; in exact arithmetic the two are the same."""
    u = 1.0 + 0j
    v = 1j * eta_b * u
    tm, te = 1.0 + 0j, 0
    for q, eta, d in reversed(layers):
        x = q * d
        k = np.floor(np.imag(x) / np.log(2)).astype(int)
        phm = np.exp(1j * np.real(x) - (np.imag(x) - k * np.log(2)))      # exp(i q d) = phm 2^-k, 0.5 < |phm| <= 1
        ph2 = np.exp(2j * x)                                              # may underflow to zero, which is harmless
        if modal:
            a, b = (u + v / (1j * eta)) / 2, (u - v / (1j * eta)) / 2     # forward and backward amplitudes at the exit face
            b = b * ph2                                                   # entrance face, common factor exp(-i q d) left out
            u, v = a + b, 1j * eta * (a - b)
        else:
            c, s = (1 + ph2) / 2, (ph2 - 1) / 2j                          # cos and sin, each times exp(i q d)
            u, v = c * u - s / eta * v, eta * s * u + c * v
        e = _pow2(np.maximum(abs(u), abs(v)))
        u, v = _scale(u, e), _scale(v, e)
        tm = tm * phm
        e2 = _pow2(abs(tm))
        tm, te = _scale(tm, e2), te - k - e + e2
    x, y = u, v / (1j * eta_a)
    return (x - y) / (x + y), 2 / (x + y) * tm, te


def star(A, B):
    """Redheffer star product of two-ports A (nearer the entrance) and B, one channel.

    Each is (X11, X12, X21, X22) with [out_left; out_right] = X [in_left; in_right]."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    D = 1 / (1 - b11 * a22)
    return (a11 + a12 * b11 * a21 * D, a12 * b12 * D, b21 * a21 * D, b22 + b21 * a22 * b12 * D)


def S_interface(eta1, eta2):
    """Scattering matrix of an interface for the amplitudes of u: (r, t', t, r')."""
    s = eta1 + eta2
    return ((eta1 - eta2) / s, 2 * eta2 / s, 2 * eta1 / s, (eta2 - eta1) / s)


def S_stack(layers, eta_a, eta_b):
    """Scattering matrix (r, t', t, r') of the stack between the half-spaces, by star products.

    Amplitudes are those of u in the outer media, taken at the outer faces. Only decaying exponentials
    exp(i q d) appear."""
    S = None
    eta_prev = eta_a
    for q, eta, d in layers:
        I = S_interface(eta_prev, eta)
        S = I if S is None else star(S, I)
        p = np.exp(1j * q * d)
        S = star(S, (0.0, p, p, 0.0))
        eta_prev = eta
    I = S_interface(eta_prev, eta_b)
    return I if S is None else star(S, I)


def E_layer(q, eta, d):
    """Stiffness matrix of a layer, [v_L; v_R] = E [u_L; u_R], from decaying exponentials only."""
    p2 = np.exp(2j * q * d)
    cot = -1j * (1 + p2) / (1 - p2)
    csc = -2j * np.exp(1j * q * d) / (1 - p2)
    return (-eta * cot, eta * csc, -eta * csc, eta * cot)


def E_compose(A, B):
    """Stiffness matrix of A followed by B (the midpoint field is eliminated)."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    D = 1 / (b11 - a22)
    return (a11 + a12 * a21 * D, -a12 * b12 * D, b21 * a21 * D, b22 - b21 * b12 * D)


def rt_from_E(E, eta_a, eta_b):
    """(r, t) from the stiffness matrix of the whole stack and the two half-spaces."""
    e11, e12, e21, e22 = E
    a, b, c, d = e11 + 1j * eta_a, e12, e21, e22 - 1j * eta_b
    det = a * d - b * c
    u0, uN = 2j * eta_a * d / det, -2j * eta_a * c / det
    return u0 - 1, uN


def H_layer(q, eta, d):
    """Hybrid matrix of a layer, [u_L; v_R] = H [v_L; u_R], from decaying exponentials only."""
    p2 = np.exp(2j * q * d)
    tan = 1j * (1 - p2) / (1 + p2)
    sec = 2 * np.exp(1j * q * d) / (1 + p2)
    return (-tan / eta, sec, sec, -eta * tan)


def rt_from_H(H, eta_a, eta_b):
    """(r, t) from the hybrid matrix of the whole stack and the two half-spaces."""
    h11, h12, h21, h22 = H
    a, b, c, d = h11 + 1 / (1j * eta_a), h12, h21, h22 - 1j * eta_b
    det = a * d - b * c
    v0, uN = 2 * d / det, -2 * c / det
    return 1 - v0 / (1j * eta_a), uN


def chain(mats, compose):
    out = mats[0]
    for X in mats[1:]:
        out = compose(out, X)
    return out


def rt_E(layers, eta_a, eta_b):
    return rt_from_E(chain([E_layer(*L) for L in layers], E_compose), eta_a, eta_b)


def rt_H(layers, eta_a, eta_b):
    return rt_from_H(chain([H_layer(*L) for L in layers], star), eta_a, eta_b)


def partial_S(layers, eta_a, eta_b):
    """Reflection seen from inside each layer, for field reconstruction without growing exponentials.

    Returns (rho_right, rho_left, etas): rho_right[j] is the reflection coefficient of everything beyond the
    exit face of layer j, seen from inside layer j at that face (for u amplitudes); rho_left[j] the same
    for everything before its entrance face."""
    n = len(layers)
    rho_right, rho_left = [None] * n, [None] * n
    rho = 0.0
    eta_next = eta_b
    for j in range(n - 1, -1, -1):
        q, eta, d = layers[j]
        r = (eta - eta_next) / (eta + eta_next)
        rho_right[j] = (r + rho) / (1 + r * rho)
        rho = rho_right[j] * np.exp(2j * q * d)
        eta_next = eta
    rho = 0.0
    eta_prev = eta_a
    for j in range(n):
        q, eta, d = layers[j]
        r = (eta - eta_prev) / (eta + eta_prev)
        rho_left[j] = (r + rho) / (1 + r * rho)
        rho = rho_left[j] * np.exp(2j * q * d)
        eta_prev = eta
    return rho_right, rho_left


def field_S(stack, beta, pol, z, k0=K0):
    """u(z) inside the stack for unit incidence from a, built from forward amplitudes at entrance faces
    and backward amplitudes at exit faces, so that every exponential decays."""
    eta_a, layers, eta_b = stack.params(beta, pol, k0)
    rho_right, _ = partial_S(layers, eta_a, eta_b)
    zi = stack.interfaces()
    z = np.asarray(z, float)
    u = np.zeros(z.shape, complex)
    # forward amplitude at the entrance face of each layer, layer by layer
    q, eta, d = layers[0]
    rho_in = rho_right[0] * np.exp(2j * q * d)             # reflection seen at the entrance face of layer 0
    r01 = (eta_a - eta) / (eta_a + eta)
    a = (2 * eta_a / (eta_a + eta)) / (1 + r01 * rho_in)       # forward amplitude just inside layer 0
    for j, (q, eta, d) in enumerate(layers):
        m = (z >= zi[j]) & (z <= zi[j + 1])
        s = z[m] - zi[j]
        b = rho_right[j] * a * np.exp(1j * q * d)              # backward amplitude at the exit face
        u[m] = a * np.exp(1j * q * s) + b * np.exp(1j * q * (d - s))
        if j + 1 < len(layers):
            qn, etan, dn = layers[j + 1]
            rho_n = rho_right[j + 1] * np.exp(2j * qn * dn)
            rjn = (eta - etan) / (eta + etan)
            a = a * np.exp(1j * q * d) * (2 * eta / (eta + etan)) / (1 + rjn * rho_n)   # generalized transmission
    return u


def field_T(stack, beta, pol, z, direction="forward", k0=K0):
    """u(z) by marching the transfer matrix: 'forward' from the entrance with the computed r,
    'backward' from the exit with the computed t."""
    eta_a, layers, eta_b = stack.params(beta, pol, k0)
    r, t = rt_from_backward(T_backward(layers), eta_a, eta_b)
    zi = stack.interfaces()
    z = np.asarray(z, float)
    u = np.zeros(z.shape, complex)
    if direction == "forward":
        psi = (1 + r, 1j * eta_a * (1 - r))
        for j, (q, eta, d) in enumerate(layers):
            m = (z >= zi[j]) & (z <= zi[j + 1])
            s = z[m] - zi[j]
            u[m] = np.cos(q * s) * psi[0] + np.sin(q * s) / eta * psi[1]
            A, B, C, D = T_layer(q, eta, d)
            psi = (A * psi[0] + B * psi[1], C * psi[0] + D * psi[1])
    else:
        psi = (t, 1j * eta_b * t)
        for j in range(len(layers) - 1, -1, -1):
            q, eta, d = layers[j]
            m = (z >= zi[j]) & (z <= zi[j + 1])
            s = zi[j + 1] - z[m]
            u[m] = np.cos(q * s) * psi[0] - np.sin(q * s) / eta * psi[1]
            A, B, C, D = T_layer_inv(q, eta, d)
            psi = (A * psi[0] + B * psi[1], C * psi[0] + D * psi[1])
    return u


# ---- one channel, multiprecision reference -------------------------------------------------------

def _mp():
    import mpmath
    return mpmath


def ref_digits(stack, beta, extra=40):
    """Decimal digits that make the transfer-matrix products exact to about `extra` digits."""
    growth = sum(abs(kz(e, beta).imag) * d for e, d in stack.layers)
    return int(extra + 2 * growth / np.log(10))


def ref_layers(stack, beta, pol, mp, k0=K0):
    k0 = mp.mpf(k0)
    b2 = mp.mpf(beta) ** 2

    def eta(eps):
        e = mp.mpc(complex(eps))
        q = k0 * mp.sqrt(e - b2)
        if mp.im(q) < 0 or (mp.im(q) == 0 and mp.re(q) < 0):
            q = -q
        return q, (q if pol == "s" else q / e)
    return eta(stack.eps_a)[1], [eta(e) + (mp.mpmathify(d),) for e, d in stack.layers], eta(stack.eps_b)[1]


def ref_matrix(stack, beta, pol="s", dps=None, k0=K0):
    """The exit-to-entrance matrix N and the admittances, as mpmath numbers at dps digits."""
    mp = _mp()
    with mp.workdps(dps or ref_digits(stack, beta)):
        eta_a, layers, eta_b = ref_layers(stack, beta, pol, mp, k0)
        N = (mp.mpf(1), mp.mpf(0), mp.mpf(0), mp.mpf(1))
        for q, eta, d in layers:
            c, s = mp.cos(q * d), mp.sin(q * d)
            N = mm(N, (c, -s / eta, eta * s, c))
        return N, eta_a, eta_b


def ref_rt(stack, beta, pol="s", dps=None, k0=K0):
    """(r, t, r', t') as mpmath numbers: the transfer-matrix formulas evaluated with enough digits."""
    mp = _mp()
    dps = dps or ref_digits(stack, beta)
    with mp.workdps(dps):
        (A, B, C, D), eta_a, eta_b = ref_matrix(stack, beta, pol, dps, k0)
        j = mp.mpc(0, 1)
        x, y = A + j * eta_b * B, (C + j * eta_b * D) / (j * eta_a)
        r, t = (x - y) / (x + y), 2 / (x + y)
        # incidence from b: psi(entrance) = t' (1, -i eta_a) = N psi(exit), psi(exit) = (1 + r', i eta_b (r' - 1))
        # => the forward matrix M = N^-1 = (D, -B, -C, A) and the formulas of rt_from_forward
        Mf = (D, -B, -C, A)
        den = j * eta_a * Mf[3] + eta_a * eta_b * Mf[1] + j * eta_b * Mf[0] - Mf[2]
        rp = (j * eta_b * Mf[0] + eta_a * eta_b * Mf[1] + Mf[2] - j * eta_a * Mf[3]) / den
        tp = 2 * j * eta_b / den
        return r, t, rp, tp


def ref_field(stack, beta, pol, z, dps=None, k0=K0):
    """u(z) inside the stack for unit incidence from a, by backward marching in multiprecision.
    Returns log10|u| and the phase as float arrays (|u| itself may underflow a double)."""
    mp = _mp()
    dps = dps or ref_digits(stack, beta)
    zi = stack.interfaces()
    z = np.asarray(z, float)
    logabs, phase = np.zeros(z.shape), np.zeros(z.shape)
    with mp.workdps(dps):
        eta_a, layers, eta_b = ref_layers(stack, beta, pol, mp, k0)
        t = ref_rt(stack, beta, pol, dps, k0)[1]
        psi = (t, mp.mpc(0, 1) * eta_b * t)
        for jl in range(len(layers) - 1, -1, -1):
            q, eta, d = layers[jl]
            for i in np.nonzero((z >= zi[jl]) & (z <= zi[jl + 1]))[0]:
                s = mp.mpf(zi[jl + 1]) - mp.mpf(z[i])
                u = mp.cos(q * s) * psi[0] - mp.sin(q * s) / eta * psi[1]
                logabs[i] = float(mp.log10(abs(u))) if u != 0 else -np.inf
                phase[i] = float(mp.arg(u)) if u != 0 else 0.0
            c, s = mp.cos(q * d), mp.sin(q * d)
            psi = (c * psi[0] - s / eta * psi[1], eta * s * psi[0] + c * psi[1])
    return logabs, phase


def relerr(x, x_ref):
    """|x - x_ref| / |x_ref| with x a double (or nan/inf) and x_ref an mpmath number; inf if x is not finite."""
    mp = _mp()
    x = complex(x)
    if not np.isfinite(x.real) or not np.isfinite(x.imag):
        return np.inf
    with mp.workdps(60):
        return float(abs(mp.mpc(x) - x_ref) / abs(x_ref))


def relerr_scaled(tm, te, x_ref):
    """Relative error of tm 2^te against an mpmath number."""
    mp = _mp()
    with mp.workdps(60):
        x = mp.mpc(complex(tm)) * mp.mpf(2) ** int(te)
        return float(abs(x - x_ref) / abs(x_ref))


# --------------------------------------------------------------------------------------------------
# Many channels
# --------------------------------------------------------------------------------------------------

class Modes:
    """Modes of a z-invariant region: Psi(z) = Fp exp(i qp z) a+ + Fm exp(i qm z) a-."""

    def __init__(self, Fp, qp, Fm, qm):
        self.Fp, self.Fm = np.asarray(Fp, complex), np.asarray(Fm, complex)
        self.qp, self.qm = np.asarray(qp, complex), np.asarray(qm, complex)
        self.n = self.Fp.shape[1]

    @property
    def F(self):
        return np.hstack([self.Fp, self.Fm])

    def T(self, d):
        """Field transfer matrix over a distance d (negative d steps back toward the entrance)."""
        ph = np.exp(1j * np.concatenate([self.qp, self.qm]) * d)
        return (self.F * ph) @ np.linalg.inv(self.F)

    def generator(self):
        """G with Psi' = G Psi."""
        return (self.F * (1j * np.concatenate([self.qp, self.qm]))) @ np.linalg.inv(self.F)


def modes_scalar(eps, beta, pol="s", k0=K0):
    """One-channel modes in the variables (u, v = u'/w) of the scalar functions, as a Modes object."""
    q = complex(kz(eps, beta, k0))
    eta = q if pol == "s" else q / eps
    return Modes([[1], [1j * eta]], [q], [[1], [-1j * eta]], [-q])


def modes_from_generator(G, flux, tol=1e-9):
    """Modes of Psi' = G Psi, split into forward and backward.

    A mode with Im q > 0 is forward; a propagating mode (real q) is forward if its power flux
    Psi^dagger flux Psi is positive."""
    lam, F = np.linalg.eig(G)
    q = -1j * lam
    scale = max(1.0, np.abs(q).max())
    fl = np.real(np.einsum("ij,ik,kj->j", F.conj(), flux, F))
    fwd = np.where(np.abs(q.imag) > tol * scale, q.imag > 0, fl > 0)
    n = G.shape[0] // 2
    assert fwd.sum() == n, "could not split the modes into forward and backward halves"
    ip = np.nonzero(fwd)[0]
    im = np.nonzero(~fwd)[0]
    ip = ip[np.argsort(q[ip].imag)]            # least evanescent first
    im = im[np.argsort(-q[im].imag)]
    return Modes(F[:, ip], q[ip], F[:, im], q[im])


def modes_sl(A, w=None):
    """Modes of u'' + A u = 0 (matrix A), Psi = (u, v) with v = u': W exp(+-i Q z), A W = W Q^2."""
    A = np.atleast_2d(np.asarray(A, complex))
    if np.allclose(A, np.diag(np.diag(A))):
        lam, W = np.diag(A).copy(), np.eye(A.shape[0], dtype=complex)
    elif np.allclose(A, A.conj().T):
        lam, W = np.linalg.eigh(A)
        lam = lam.astype(complex)
    else:
        lam, W = np.linalg.eig(A)
    q = np.sqrt(lam + 0j)
    q = np.where(q.imag < 0, -q, q)
    V = 1j * W * q
    return Modes(np.vstack([W, V]), q, np.vstack([W, -V]), -q)


FLUX_SL = lambda n: np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]]) / 2j


def bstar(A, B):
    """Redheffer star product for n-channel two-ports given as (X11, X12, X21, X22) of n x n blocks."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    I = np.eye(a22.shape[0])
    D1 = np.linalg.solve(I - b11 @ a22, np.hstack([b11 @ a21, b12]))
    D2 = np.linalg.solve(I - a22 @ b11, np.hstack([a21, a22 @ b12]))
    n = a21.shape[1]
    return (a11 + a12 @ D1[:, :n], a12 @ D1[:, n:], b21 @ D2[:, :n], b22 + b21 @ D2[:, n:])


def bS_interface(L, R):
    """Scattering matrix of the interface between regions with modes L (entrance side) and R."""
    n = L.n
    X = np.linalg.solve(np.hstack([L.Fm, -R.Fp]), np.hstack([-L.Fp, R.Fm]))
    return (X[:n, :n], X[:n, n:], X[n:, :n], X[n:, n:])


def block_S(Ma, layers, Mb):
    """Scattering matrix (R, T', T, R') of the stack; layers is [(Modes, d), ...]."""
    S, prev = None, Ma
    for M, d in layers:
        I = bS_interface(prev, M)
        S = I if S is None else bstar(S, I)
        Z = np.zeros((M.n, M.n))
        S = bstar(S, (Z, np.diag(np.exp(-1j * M.qm * d)), np.diag(np.exp(1j * M.qp * d)), Z))
        prev = M
    I = bS_interface(prev, Mb)
    return I if S is None else bstar(S, I)


def block_T(Ma, layers, Mb, convention="backward"):
    """(R, T) from the product of field transfer matrices.

    'backward': N = prod T_j^-1 carries the exit fields to the entrance, K = Fa^-1 N Fb, T = K11^-1,
    R = K21 K11^-1.  'forward': M = prod T_j, K = Fb^-1 M Fa, R = -K22^-1 K21, T = K11 + K12 R."""
    n = Ma.n
    P = np.eye(2 * n, dtype=complex)
    if convention == "backward":
        for M, d in layers:
            P = P @ M.T(-d)
        K = np.linalg.solve(Ma.F, P @ Mb.F)
        T = np.linalg.inv(K[:n, :n])
        return K[n:, :n] @ T, T
    for M, d in layers:
        P = M.T(d) @ P
    K = np.linalg.solve(Mb.F, P @ Ma.F)
    R = -np.linalg.solve(K[n:, n:], K[n:, :n])
    return R, K[:n, :n] + K[:n, n:] @ R


def block_ETM(Ma, layers, Mb):
    """(R, T) by the enhanced transmittance matrix recursion of Moharam et al. (1995), in mode form.

    The frame Phi of the fields at the entrance face of each layer is built from the exit side; the
    factor exp(-i q d) that a plain transfer matrix would need is moved into the transmitted amplitudes."""
    n = Ma.n
    Phi = Mb.Fp.copy()
    pending = []                                    # (a, Pp) per layer, last layer first
    for M, d in reversed(layers):
        ab = np.linalg.solve(M.F, Phi)
        a, b = ab[:n], ab[n:]
        Pp, Pm = np.exp(1j * M.qp * d), np.exp(-1j * M.qm * d)
        ainv_P = np.linalg.solve(a, np.diag(Pp))
        Phi = M.Fp + (M.Fm * Pm) @ b @ ainv_P
        pending.append(ainv_P)
    X = np.linalg.solve(np.hstack([-Ma.Fm, Phi]), Ma.Fp)
    R, T = X[:n], X[n:]
    for ainv_P in reversed(pending):                # first layer first
        T = ainv_P @ T
    return R, T


def block_QR(Ma, layers, Mb, max_growth=5.0):
    """(R, T) by carrying an orthonormal frame of the outgoing solutions from exit to entrance.

    Each layer is cut into steps whose largest growth exponent is at most max_growth; after each step
    the frame is re-orthonormalized and the triangular factor goes into the transmission."""
    n = Ma.n
    Phi, Rfac = np.linalg.qr(Mb.Fp)
    Tacc = np.linalg.inv(Rfac)          # exit amplitudes t = Tacc c, with the exit fields equal to Phi c
    for M, d in reversed(layers):
        growth = max(np.abs(np.concatenate([M.qp, M.qm]).imag).max() * d, 1e-300)
        steps = max(1, int(np.ceil(growth / max_growth)))
        Tstep = M.T(-d / steps)
        for _ in range(steps):
            Phi, Rk = np.linalg.qr(Tstep @ Phi)
            Tacc = Tacc @ np.linalg.inv(Rk)
    Y = np.linalg.solve(Ma.F, Phi)
    Y1inv = np.linalg.inv(Y[:n])
    return Y[n:] @ Y1inv, Tacc @ Y1inv


def block_global(Ma, layers, Mb):
    """(R, T) from one linear system for all amplitudes (the global matrix), solved by LU with pivoting.

    Forward amplitudes are referred to the entrance face of their layer and backward ones to the exit
    face, so the matrix holds decaying exponentials only. Also returns the layer amplitudes."""
    n, L = Ma.n, len(layers)
    size = 2 * n * (L + 1)
    A = np.zeros((size, size), complex)
    rhs = np.zeros((size, n), complex)
    col = lambda j: slice(n + 2 * n * j, n + 2 * n * (j + 1))      # amplitudes (c+, c-) of layer j
    # unknowns: R (n), then per layer (c+, c-), then T (n)
    for i in range(L + 1):
        row = slice(2 * n * i, 2 * n * (i + 1))
        if i == 0:
            A[row, :n] = Ma.Fm
            rhs[row] = -Ma.Fp
        else:
            M, d = layers[i - 1]
            A[row, col(i - 1)] = np.hstack([M.Fp * np.exp(1j * M.qp * d), M.Fm])
        if i == L:
            A[row, size - n:] = -Mb.Fp
        else:
            M, d = layers[i]
            A[row, col(i)] = -np.hstack([M.Fp, M.Fm * np.exp(-1j * M.qm * d)])
    X = np.linalg.solve(A, rhs)
    return X[:n], X[size - n:], [X[col(j)] for j in range(L)]


# ---- stiffness and hybrid matrices, many channels ----------------------------------------------------

def _wv(M, d):
    n = M.n
    Pp, Pm = np.exp(1j * M.qp * d), np.exp(-1j * M.qm * d)
    return M.Fp[:n], M.Fp[n:], M.Fm[:n], M.Fm[n:], Pp, Pm


def _blocks(X, n):
    return (X[:n, :n], X[:n, n:], X[n:, :n], X[n:, n:])


def bE_layer(M, d):
    """Stiffness matrix of a layer, [v_L; v_R] = E [u_L; u_R], with u the first n field components and
    v the last n. Built from the amplitudes of forward waves at the entrance face and backward waves at
    the exit face, so only decaying exponentials enter."""
    Wp, Vp, Wm, Vm, Pp, Pm = _wv(M, d)
    U = np.block([[Wp, Wm * Pm], [Wp * Pp, Wm]])
    V = np.block([[Vp, Vm * Pm], [Vp * Pp, Vm]])
    return _blocks(np.linalg.solve(U.T, V.T).T, M.n)


def bH_layer(M, d):
    """Hybrid matrix of a layer, [u_L; v_R] = H [v_L; u_R]."""
    Wp, Vp, Wm, Vm, Pp, Pm = _wv(M, d)
    A = np.block([[Wp, Wm * Pm], [Vp * Pp, Vm]])
    B = np.block([[Vp, Vm * Pm], [Wp * Pp, Wm]])
    return _blocks(np.linalg.solve(B.T, A.T).T, M.n)


def bE_compose(A, B):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    X = np.linalg.solve(b11 - a22, np.hstack([a21, b12]))
    n = a21.shape[1]
    return (a11 + a12 @ X[:, :n], -a12 @ X[:, n:], b21 @ X[:, :n], b22 - b21 @ X[:, n:])


def block_E(Ma, layers, Mb):
    """(R, T) from the stiffness matrix of the stack and the two half-spaces."""
    n = Ma.n
    e11, e12, e21, e22 = chain([bE_layer(M, d) for M, d in layers], bE_compose)
    Wpa, Vpa, Wma, Vma = Ma.Fp[:n], Ma.Fp[n:], Ma.Fm[:n], Ma.Fm[n:]
    Wpb, Vpb = Mb.Fp[:n], Mb.Fp[n:]
    lhs = np.block([[Vma - e11 @ Wma, -e12 @ Wpb], [-e21 @ Wma, Vpb - e22 @ Wpb]])
    X = np.linalg.solve(lhs, np.vstack([e11 @ Wpa - Vpa, e21 @ Wpa]))
    return X[:n], X[n:]


def block_H(Ma, layers, Mb):
    """(R, T) from the hybrid matrix of the stack (composed with the star product) and the two half-spaces."""
    n = Ma.n
    h11, h12, h21, h22 = chain([bH_layer(M, d) for M, d in layers], bstar)
    Wpa, Vpa, Wma, Vma = Ma.Fp[:n], Ma.Fp[n:], Ma.Fm[:n], Ma.Fm[n:]
    Wpb, Vpb = Mb.Fp[:n], Mb.Fp[n:]
    lhs = np.block([[Wma - h11 @ Vma, -h12 @ Wpb], [-h21 @ Vma, Vpb - h22 @ Wpb]])
    X = np.linalg.solve(lhs, np.vstack([h11 @ Vpa - Wpa, h21 @ Vpa]))
    return X[:n], X[n:]


# ---- counting bound modes with the stiffness form ---------------------------------------------------

Z_REAL = np.array([[0, 0, 0, 1j], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, -1j, 0]])   # (Ex, Ey, Hx, Hy) -> (i Hy, Ey; -Ex, -i Hx)


def transformed(M, Z):
    """The same modes in the field variables Z Psi."""
    return Modes(Z @ M.Fp, M.qp, Z @ M.Fm, M.qm)


def count_modes(Ma, layers, Mb, safety=0.5, return_pivots=False):
    """Number of negative directions of the stiffness form of a lossless stack between evanescent half-spaces.

    The field vector must be (u; v) with u^T v the boundary term of the quadratic form, that is
    (u, u'/w) for one channel and Z_REAL Psi for the 4-vector of berreman_G with eps_xz = eps_yz = 0.
    Each layer is cut into sublayers thinner than `safety` times half the shortest wavelength along z,
    so that no sublayer has a mode of its own with u = 0 on both faces; the count is then the number
    of negative pivots of the block-tridiagonal stiffness form (Wittrick and Williams). For dielectric
    layers this is the number of bound modes whose beta is larger than the one the modes were built for."""
    n = Ma.n
    herm = lambda X: (X + X.conj().T) / 2
    P = Ma.Fm[n:] @ np.linalg.inv(Ma.Fm[:n])                 # v = Y_a u for the solutions decaying toward -infinity
    count, worst_asym, pivots = 0, 0.0, []
    for M, d in layers:
        qmax = np.abs(np.concatenate([M.qp, M.qm]).real).max()
        m = max(1, int(np.ceil(d * qmax / (safety * np.pi))))
        e11, e12, e21, e22 = bE_layer(M, d / m)
        for _ in range(m):
            P = P - e11
            worst_asym = max(worst_asym, np.abs(P - P.conj().T).max() / max(np.abs(P).max(), 1e-300))
            ev = np.linalg.eigvalsh(herm(P))
            count += int((ev < 0).sum())
            pivots.append(ev)
            P = e22 + e21 @ np.linalg.solve(P, e12)
    P = P - Mb.Fp[n:] @ np.linalg.inv(Mb.Fp[:n])
    ev = np.linalg.eigvalsh(herm(P))
    count += int((ev < 0).sum())
    pivots.append(ev)
    return (count, pivots, worst_asym) if return_pivots else count


def slice_modes(count, lo, hi, tol=1e-13, maxit=200):
    """All jumps of the non-increasing integer function count(beta) in (lo, hi), by bisection.
    Returns [(beta, multiplicity), ...] with beta decreasing."""
    out = []

    def rec(a, b, na, nb, depth):
        if na == nb:
            return
        if b - a <= tol * max(abs(a), abs(b)) or depth > maxit:
            out.append(((a + b) / 2, na - nb))
            return
        m = (a + b) / 2
        nm = count(m)
        rec(m, b, nm, nb, depth + 1)
        rec(a, m, na, nm, depth + 1)
    rec(lo, hi, count(lo), count(hi), 0)
    return out


# ---- compound (second exterior power) matrices for two channels -----------------------------------

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def compound2(X):
    """Second compound of a matrix with 4 rows: all 2x2 minors, rows and columns in the order PAIRS."""
    X = np.asarray(X)
    cols = PAIRS if X.shape[1] == 4 else [(0, 1)]
    C = np.empty((6, len(cols)), X.dtype)
    for a, (i, j) in enumerate(PAIRS):
        for b, (k, l) in enumerate(cols):
            C[a, b] = X[i, k] * X[j, l] - X[i, l] * X[j, k]
    return C


def compound_R(Ma, layers, Mb):
    """Reflection matrix of a two-channel stack from the Plucker coordinates of the outgoing plane.

    The plane is carried from exit to entrance as one vector of the six 2x2 minors. In the mode basis
    of a layer the compound transfer matrix is diagonal with exponents q_a + q_b; the largest growth,
    that of the two forward modes, is divided out. Returns (R, p, logscale) with p the six coordinates
    in the mode basis of medium a, normalized, and logscale the (real) logarithm of the removed factors,
    so that p[0] exp(logscale) is the determinant of the block K11 of the transfer-matrix method: the
    secular function of the bound modes."""
    p = compound2(Mb.Fp)[:, 0]
    logscale = 0.0
    for M, d in reversed(layers):
        C = compound2(M.F)
        qq = np.concatenate([M.qp, M.qm])
        expo = np.array([qq[i] + qq[j] for i, j in PAIRS])
        c = np.linalg.solve(C, p)
        c = c * np.exp(-1j * expo * d - expo[0].imag * d)    # the growth of the pair of forward modes is divided out
        logscale += expo[0].imag * d
        p = C @ c
        nrm = np.linalg.norm(p)
        p, logscale = p / nrm, logscale + np.log(nrm)
    pm = np.linalg.solve(compound2(Ma.F), p)
    R = np.array([[-pm[3], pm[1]], [-pm[4], pm[2]]]) / pm[0]
    return R, pm, logscale


# ---- anisotropic layers (Berreman 4x4) ------------------------------------------------------------

FLUX_B = np.array([[0, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0]], float) / 2   # Re(Ex Hy* - Ey Hx*)/2


def uniaxial(eps_o, eps_e, theta, phi):
    """Permittivity tensor of a uniaxial medium with its axis at polar angle theta from z, azimuth phi from x."""
    c = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return eps_o * np.eye(3) + (eps_e - eps_o) * np.outer(c, c)


def berreman_G(eps, beta, k0=K0):
    """G with Psi' = G Psi for Psi = (Ex, Ey, Hx, Hy), H in units that make the vacuum impedance 1."""
    e = np.asarray(eps, complex)
    ezz = e[2, 2]
    D = np.array([
        [-beta * e[2, 0] / ezz, -beta * e[2, 1] / ezz, 0, 1 - beta ** 2 / ezz],
        [0, 0, -1, 0],
        [-(e[1, 0] - e[1, 2] * e[2, 0] / ezz), -(e[1, 1] - e[1, 2] * e[2, 1] / ezz - beta ** 2), 0, beta * e[1, 2] / ezz],
        [e[0, 0] - e[0, 2] * e[2, 0] / ezz, e[0, 1] - e[0, 2] * e[2, 1] / ezz, 0, -beta * e[0, 2] / ezz]])
    return 1j * k0 * D


def modes_aniso(eps, beta, k0=K0):
    return modes_from_generator(berreman_G(eps, beta, k0), FLUX_B)


def modes_iso4(eps, beta, k0=K0):
    """Modes of an isotropic medium in the 4-vector (Ex, Ey, Hx, Hy), in the order (s, p).
    Amplitudes are E_y for s and H_y for p, as in the one-channel functions."""
    g = np.sqrt(complex(eps) - beta ** 2)
    if g.imag < 0:
        g = -g
    Fp = np.array([[0, g / eps], [1, 0], [-g, 0], [0, 1]], complex)
    Fm = np.array([[0, -g / eps], [1, 0], [g, 0], [0, 1]], complex)
    return Modes(Fp, k0 * np.array([g, g]), Fm, -k0 * np.array([g, g]))


# ---- lamellar grating, E along the grooves (s polarization), Fourier modal method -------------------

def grating_A(eps_ridge, eps_groove, fill, period, beta0, M, k0=K0):
    """A = k0^2 [[eps]] - Kx^2 for orders -M..M, so that u'' + A u = 0 for the Fourier amplitudes of E_y.
    The ridge occupies the fraction `fill` of the period, centered at x = 0."""
    m = np.arange(-M, M + 1)
    kx = k0 * beta0 + TWO_PI * m / period
    dm = m[:, None] - m[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        coef = np.where(dm == 0, fill, np.sin(np.pi * dm * fill) / (np.pi * dm))
    E = (eps_ridge - eps_groove) * coef + eps_groove * (dm == 0)
    return k0 ** 2 * E - np.diag(kx ** 2), kx


def uniform_A(eps, kx, k0=K0):
    return np.diag(k0 ** 2 * eps - kx ** 2 + 0j)


# ---- many channels, multiprecision reference -------------------------------------------------------

def ref_block_rt(Ma, gens, Mb, dps):
    """(R, T) as mpmath matrices from N = prod expm(-G_j d_j) at dps digits.

    gens is [(G, d), ...] with G the double-precision generator of each layer, taken as exact; the
    mode matrices of the two half-spaces are also taken as exact. This is the problem the
    double-precision routines are given, apart from their own eigendecompositions of G."""
    mp = _mp()
    n = Ma.n
    with mp.workdps(dps):
        tomp = lambda X: mp.matrix([[mp.mpc(complex(x)) for x in row] for row in np.asarray(X)])
        P = mp.eye(2 * n)
        for G, d in gens:
            P = P * mp.expm(tomp(G) * (-mp.mpf(d)), method="taylor")
        K = mp.inverse(tomp(Ma.F)) * (P * tomp(Mb.F))
        T = mp.inverse(K[:n, :n])
        return K[n:, :n] * T, T


def ref_plane(gens, Fb, dps):
    """Orthonormal basis, as doubles, of the plane of outgoing solutions carried from the exit to the
    entrance: the column space of prod expm(-G_j d_j) Fb, by Gram-Schmidt at dps digits."""
    mp = _mp()
    Fb = np.asarray(Fb)
    with mp.workdps(dps):
        tomp = lambda X: mp.matrix([[mp.mpc(complex(x)) for x in row] for row in np.asarray(X)])
        P = mp.eye(Fb.shape[0])
        for G, d in gens:
            P = P * mp.expm(tomp(G) * (-mp.mpf(d)), method="taylor")
        Y = P * tomp(Fb)
        cols = []
        for j in range(Y.cols):
            v = Y[:, j]
            for c in cols:
                v = v - c * (c.H * v)[0]
            cols.append(v / mp.sqrt(mp.re((v.H * v)[0])))
        return np.array([[complex(c[i]) for c in cols] for i in range(Y.rows)])


def plane_distance(Q, Phi):
    """Sine of the largest principal angle between the column spaces of Q (orthonormal) and Phi."""
    if not np.all(np.isfinite(Phi)):
        return np.inf
    Qc, _ = np.linalg.qr(Phi)
    return np.linalg.norm(Qc - Q @ (Q.conj().T @ Qc), 2)


def mp_err(X, X_ref, relative_to=None):
    """Largest |X - X_ref| over the entries, divided by the largest |X_ref| (or by relative_to)."""
    mp = _mp()
    X = np.asarray(X)
    if not np.all(np.isfinite(X)):
        return np.inf
    with mp.workdps(60):
        num = max(abs(mp.mpc(complex(X[i, j])) - X_ref[i, j]) for i in range(X.shape[0]) for j in range(X.shape[1]))
        den = relative_to or max(abs(X_ref[i, j]) for i in range(X.shape[0]) for j in range(X.shape[1]))
        return float(num / den)
