"""Thin-wire medium: a square lattice of wires along z, wave vector k = (k_x, k_y, beta) with a component along the wires.

Used by notebook 07. Conventions as in the other notebooks of this folder: exp(-i w t), eps0 = mu0 = c = 1 (w = k0),
period a = 1 by default.

Model
-----
Each wire carries an axial current I exp(i (kperp . R_j + beta z)) spread uniformly over a cylinder of radius r0
(order m = 0 only). With k_t = sqrt(k0^2 - beta^2) the field of one wire outside it is

    E_z = -(k_t^2 / (4 k0)) I J_0(k_t r0) H_0(k_t rho),

which follows from (Laplacian_perp + k_t^2) E_z = -i (k_t^2 / k0) J_z (Maxwell, with the charge density of the
current in the divergence term). The condition on wire 0 is that the surface average of the total E_z equals
Z_w I. With the lattice sum G_0 = G_0(k_t, kperp) of ``rodlattice.lattice_sums`` it reads

    I [ Z_w + (k_t^2 / (4 k0)) J_0(k_t r0) ( H_0(k_t r0) + J_0(k_t r0) G_0 ) ] = s0 J_0(kperp r0),

with s0 = i k_t^2 / (k0 (kperp^2 - k_t^2)) the free-space E_z of a unit source current J0 z exp(i k.r). The
polarization is P_z = (i / k0) <J> with <J> = I J_0(kperp r0) / a^2, and the averaged Maxwell equation gives

    <E_z> = (k_t^2 / k0^2) (k0^2 P_z + i k0) / (kperp^2 - k_t^2),     eps_zz = 1 + P_z / <E_z>,   eps = diag(1, 1, eps_zz).

``S = H_0(k_t r0) + J_0(k_t r0) G_0`` is an analytic function of k_t^2: the logarithm in k_t of H_0 cancels
that of G_0. Hence complex beta (Cauchy integrals) is allowed. For real k0, beta and kperp below the first
diffraction threshold S is purely imaginary, which makes the mode condition real (``mode_function``).

Wire impedance Z_w (field at the surface per unit total current): 0 for a perfect conductor (eps_m = None),
``Zw_thin = i / (k0 (eps_m - 1) pi r0^2)`` for uniform current, ``Zw_exact = k_m J_0(k_m r0) / (2 pi r0 sigma J_1(k_m r0))``
with k_m = k0 sqrt(eps_m), sigma = -i k0 (eps_m - 1) for a solid round wire (it neglects beta^2 against k_m^2 inside
the metal). eps_m may be a number or a function of k0.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import hankel1, jv

import rodlattice as rl


def kt_of(k0, beta):
    """k_t = sqrt(k0^2 - beta^2) with Re >= 0 (Im >= 0 if purely imaginary); continuous for |beta| < k0, also complex."""
    s = np.sqrt(complex(k0) ** 2 - complex(beta) ** 2)
    if s.real < 0 or (s.real == 0 and s.imag < 0):
        s = -s
    if abs(s) < 1e-8:                                # exactly on the TEM line: the lattice sums need k_t != 0
        s = 1e-8 + 0j
    return s


def G0_of(kt, kperp, a=1.0):
    """Lattice sum G_0(k_t, kperp) of the square lattice (k_t may be complex or imaginary)."""
    return rl.lattice_sums(kt, (kperp[0], kperp[1]), 0, a)[0]


def _eps_value(eps_m, k0):
    return eps_m(k0) if callable(eps_m) else eps_m


def Zw_thin(k0, r0, eps_m):
    """Internal impedance of a thin wire with uniform current."""
    return 1j / (k0 * (eps_m - 1) * np.pi * r0 ** 2)


def Zw_exact(k0, r0, eps_m):
    """Internal impedance of a solid round wire (skin effect included), surface field per total current."""
    km = k0 * np.sqrt(complex(eps_m))
    sigma = -1j * k0 * (eps_m - 1)
    x = km * r0
    return km * jv(0, x) / (2 * np.pi * r0 * sigma * jv(1, x))


def Zw(k0, r0, eps_m, exact=False):
    if eps_m is None:
        return 0.0
    e = _eps_value(eps_m, k0)
    return Zw_exact(k0, r0, e) if exact else Zw_thin(k0, r0, e)


def drude(f, f_p, f_gamma=0.0):
    """eps_m = 1 - f_p^2 / (f (f + i f_gamma)) for frequency f = a / lambda0 (exp(-i w t): Im eps_m > 0)."""
    return 1 - f_p ** 2 / (f * (f + 1j * f_gamma))


def drude_k0(f_p, f_gamma=0.0):
    """The same as a function of k0 = 2 pi f, to pass as eps_m."""
    return lambda k0: drude(k0 / (2 * np.pi), f_p, f_gamma)


def _core(k0, kperp, beta, r0, eps_m, exact, a):
    kt = kt_of(k0, beta)
    kp2 = complex(kperp[0]) ** 2 + complex(kperp[1]) ** 2
    kp = np.sqrt(kp2)
    x = kt * r0
    Jt = jv(0, x)
    S = hankel1(0, x) + Jt * G0_of(kt, kperp, a)
    D = Zw(k0, r0, eps_m, exact) + kt ** 2 / (4 * k0) * Jt * S
    q = kp2 - kt ** 2
    return kt, kp2, kp, Jt, S, D, q


def eps_zz(k0, kperp, beta, r0, eps_m=None, exact=False, a=1.0, full=False):
    """eps_zz(omega, k) of the wire lattice for k = (kperp, beta), driven by a source J0 z exp(i k.r)."""
    kt, kp2, kp, Jt, S, D, q = _core(k0, kperp, beta, r0, eps_m, exact, a)
    Jp = jv(0, kp * r0)
    s0 = 1j * kt ** 2 / (k0 * q)
    I = s0 * Jp / D
    P = 1j * I * Jp / (k0 * a * a)
    E = (kt ** 2 / (k0 ** 2 * q)) * (k0 ** 2 * P + 1j * k0)
    e = 1 + P / E
    return (e, dict(I=I, P=P, E=E, S=S, D=D)) if full else e


def mode_condition(k0, kperp, beta, r0, eps_m=None, exact=False, a=1.0):
    """D = Z_w + (k_t^2/4k0) J_0 (H_0 + J_0 G_0): zero for a Bloch mode (complex in general)."""
    return _core(k0, kperp, beta, r0, eps_m, exact, a)[5]


def mode_function(k0, kperp, beta, r0, eps_m=None, exact=False, a=1.0):
    """Real function whose zeros are the Bloch modes (lossless wires, real k0, beta, kperp below the diffraction onset).

    It is q D / i with q = kperp^2 - k_t^2. D has a pole at q = 0 (the free-space light cone of the lattice
    sums), which q D removes; D itself is imaginary, so no sign change except at modes. The factor k_t^2 in D
    makes beta = k0 a zero for a perfect conductor: the TEM branch.
    """
    kt, kp2, kp, Jt, S, D, q = _core(k0, kperp, beta, r0, eps_m, exact, a)
    return (q * D / 1j).real


def mode_imag(k0, kperp, beta, r0, eps_m=None, exact=False, a=1.0):
    """Relative real part of q D (should vanish in the lossless case, which is what makes mode_function sufficient)."""
    kt, kp2, kp, Jt, S, D, q = _core(k0, kperp, beta, r0, eps_m, exact, a)
    return abs((q * D).real) / (abs(q * D) + 1e-300)


def diffraction_onset(kperp, a=1.0, nmax=3):
    """Smallest |kperp + G|, G != 0 a reciprocal-lattice vector: the real form of the mode condition needs k_t below it."""
    g = 2 * np.pi / a * np.arange(-nmax, nmax + 1)
    return min(np.hypot(kperp[0] + gx, kperp[1] + gy) for gx in g for gy in g if (gx, gy) != (0, 0))


def find_modes_k0(beta, kperp, r0, k0_grid, eps_m=None, exact=False, a=1.0):
    """Frequencies k0 (not f) at which mode_function changes sign on k0_grid, at fixed beta and kperp.

    Only the part of the grid with k_t below the diffraction onset is used (beyond it G_0 is complex and the real
    form is invalid).
    """
    k0_grid = np.asarray(k0_grid)
    kt_grid = np.sqrt(np.maximum(k0_grid ** 2 - beta ** 2, 0.0))
    k0_grid = k0_grid[kt_grid < 0.999 * diffraction_onset(kperp, a)]
    vals = np.array([mode_function(k, kperp, beta, r0, eps_m, exact, a) for k in k0_grid])
    roots = []
    for i in np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]:
        roots.append(brentq(lambda k: mode_function(k, kperp, beta, r0, eps_m, exact, a), k0_grid[i], k0_grid[i + 1], xtol=1e-13))
    return np.array(roots)


# ------------------------------------------------------------------------- closed forms for comparison

def kp2_belov(r0, a=1.0):
    """Belov et al. (2003): k_p^2 = 2 pi / (a^2 (ln(a / (2 pi r0)) + 0.5275))."""
    return 2 * np.pi / (a * a * (np.log(a / (2 * np.pi * r0)) + 0.5275))


def kp2_model(r0, k=1e-3, a=1.0):
    """k_p^2 of the model, from the small-(k0, beta, kperp) limit: -(k0^2 - beta^2) (eps_zz - 1) for a perfect conductor.

    Exactly, eps_zz - 1 = 4 i / (k_t^2 a^2 (H_0(k_t r0) + G_reg)) with G_reg = G_0 + 4 i / (a^2 (k_t^2 - kperp^2)) regular;
    the sum H_0 + G_reg tends to a constant, so the result is independent of k to O(k^2 a^2).
    """
    k0, beta, kperp = 1.0 * k, 0.3 * k, (0.2 * k, 0.1 * k)
    e = eps_zz(k0, kperp, beta, r0, None, a=a)
    return (-(k0 ** 2 - beta ** 2) * (e - 1)).real


def eps_silveirinha(k0, beta, r0, eps_m, kp2, exact=False, a=1.0):
    """Silveirinha (2006) form: eps_zz = 1 + 1 / (1/((eps_m - 1) eta_w) - (k0^2 - beta^2) / k_p^2).

    With exact=True the kinetic term 1/((eps_m - 1) eta_w) is replaced by -i a^2 k0 Z_w (equal for uniform current).
    """
    eta_w = np.pi * r0 ** 2 / (a * a)
    T = -1j * a * a * k0 * Zw(k0, r0, eps_m, True) if exact else 1 / ((eps_m - 1) * eta_w)
    return 1 + 1 / (T - (k0 ** 2 - beta ** 2) / kp2)


def taylor(fun, rho, n=16, orders=(0, 1, 2, 3, 4)):
    """Taylor coefficients at 0 of an analytic function of one complex variable, from n samples on |z| = rho."""
    th = (np.arange(n) + 0.5) * 2 * np.pi / n
    vals = np.array([fun(rho * np.exp(1j * t)) for t in th])
    return np.array([(vals * np.exp(-1j * m * th)).mean() / rho ** m for m in orders])
