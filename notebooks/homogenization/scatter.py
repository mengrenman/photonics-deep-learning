"""Bridge between homogenization and the Bayesian scattering testbed (notebook 06).

Purpose
-------
* Exact scattered data of concentric dielectric rings (multilayer Mie series) and of a homogeneous disk anywhere in the
  plane, with analytic derivatives with respect to (eps, R, cx, cy).
* Ring geometries with a period cut or any cut phase; the Rytov and TM second-order effective-permittivity formulas.
* Homogeneous-disk fits (Gauss-Newton on whitened data), the profile foot point, whitened Jacobians for
  ``bayes_scatter.detectability.analyze_error``.
* The C20 isotypic projectors of the ch15 acquisition (40 transmitters, 60 receivers on circles), the lattice-disk
  painter, and the Born gain of a Gaussian-windowed grating.

Conventions
-----------
* Time dependence exp(-i omega t). TM means E along z, u = E_z, line sources u_inc = (i/4) H0^(1)(k0 |r - r_t|).
* Lengths are in free-space wavelengths (k0 = 2 pi), as in ``bayes_scatter``, except in the Bloch-side functions
  (``tm_second_order_sum`` and friends), which use the units of the ``bloch.Lattice`` (period a).
* Data arrays have shape (Nt, Nr) = (transmitter, receiver). Pixel arrays ``a[i, j]`` have i the x index.
* Parameters of a disk: ``p = (eps, R)`` or ``(eps, R, cx, cy)``; eps is the relative permittivity.
* Whitening: a complex (Nt, Nr) array d becomes the real vector sqrt(2)/s [Re d.ravel(); Im d.ravel()] of length 2 Nt Nr
  (``bayes_scatter.detectability.stack_real``), so that the misfit Phi = sum |r|^2/s^2 = (1/2) ||r_w||^2 (in nats).

Attribution
-----------
``log_ratios``, ``multilayer`` and ``exact_data`` are copied from
notebooks/cloaking/04_partial_waves_and_scattering_cancellation.ipynb (cell 4: ``log_ratios``, ``multilayer``; cell 32:
``exact_data``), where they were validated against ``bayes_scatter.mie.mie_scattered``. Changes: the TM-only signature
(``pol`` moved to the last keyword), the default ``M = 60`` of ``exact_data`` (the notebook used 70), and these docstrings.

Public functions and contracts
------------------------------
``log_ratios(x, M)``, ``multilayer(radii, eps, k0=2 pi, M=None, pol="TM")``, ``exact_data(tx, rx, radii, eps, M=60)``
    Copied code. ``multilayer`` returns the Mie coefficients s_m, m = 0..M, of concentric layers with OUTER radii
    ``radii`` (increasing) and permittivities ``eps`` (core first), vacuum outside. ``exact_data`` returns the exact
    scattered field (Nt, Nr) of line sources ``tx`` at receivers ``rx``, centered layers only. Exact up to the series
    truncation; M = 60 and 90 agree to 1e-12 for k0 R <= 6.
``mie_disk(eps, R, center, tx, rx, M=None)`` / ``disk_data``
    Exact data of ONE homogeneous disk at ``center`` (any place that keeps all antennas outside), by the closed-form Mie
    series with the centered-disk coefficients of ``bayes_scatter.mie``. Agrees with ``mie.mie_scattered`` to roundoff.
    ``M`` defaults to ceil(max(|k1|, k0) R) + 25, far past the last significant order. 
``disk_tangents(p, tx, rx, names=None, jac="analytic", step=1e-6)``
    Data tangents d(data)/d(parameter) (k, Nt, Nr), complex. ``jac="analytic"`` (default) uses closed-form derivatives of
    the Mie coefficients (Bessel ODE) and of the polar coordinates: exact to roundoff, so symmetry properties of the
    tangents hold to 1e-14. ``jac="fd"`` is a central difference with ``step`` (accurate to about 1e-10 relative; it
    breaks exact symmetries at that level).
``disk_jacobian(p, tx, rx, s, names=None)``
    Whitened Jacobian D (2 Nt Nr, k) of the disk model.
``rings(R_D, n_p, f, eps1, eps2, cut="period")`` -> ``(radii, eps)``
    Concentric rings of period p = R_D/n_p (n_p may be non-integer), eps1 on the first fraction f of each period, eps2 on
    the rest. The cut phase c in [0, 1) is the phase of the period at r = R_D: ``"period"`` (c = 0) gives core eps1 and
    outermost layer eps2 ending exactly at R_D (n_p integer: n_p complete periods); ``"mid"`` (c = (1+f)/2) puts R_D in the
    middle of an eps2 layer (core eps2 of thickness (1-f)p/2, outermost eps2 of half thickness); a float is c.
    Layer radii are outer radii, the last one is R_D exactly; layers have no zero thickness.
``ring_mean_eps(f, eps1, eps2)``, ``rytov_tm(f, eps1, eps2, p, k0)``, ``rytov_eps_eff``
    epsbar = f eps1 + (1-f) eps2; Rytov's second-order correction of the ordinary wave,
    k0^2 p^2 f^2 (1-f)^2 (eps1 - eps2)^2 / 12 (= (pi^2/3)(p/lambda)^2 f^2 (1-f)^2 d_eps^2). Valid to O(k0^4).
``tm_second_order_1d(f, eps1, eps2, p, k0, nterms=100000)``
    The second-order sum k0^2 sum_{G != 0} |eps_G|^2/|G|^2 of the 1D laminate by direct summation with the n^-4 tail;
    equals ``rytov_tm`` to 1e-10 (it is the same number).
``tm_second_order_sum(lat, rods, gmax, eps_bg=1.0, units="a_over_lambda")``
    S = sum_{0 < |G| <= gmax} |eps_G|^2/|G|^2 over the plane waves of a ``bloch.Lattice`` and rods (eps_hat from
    ``bloch.py``). With ``units="a_over_lambda"`` (default) the value is (2 pi)^2 S/a^2, the coefficient c in
    eps_2nd = <eps> + c (a/lambda)^2 (the plan's 1.2874 and 17.309); with ``units="k0"`` it is S itself, the coefficient
    in <eps> + S k0^2 (k0 in 1/length of the lattice). Exact in the contrast; truncation error about 8e-5 relative at 600 plane waves.
``eps_eff_pwe(lat, rods, K, G, B=None)`` -> ``(eps_eff, k0)``
    Lowest-band PWE (bloch.tm_solve) at wave vector K: k0 = 2 pi f/a and eps_eff = |K|^2/k0^2. ``tm_residual_k04``
    returns (eps_eff - <eps> - k0^2 S)/k0^4 (O(1) if the expansion is right).
``whiten(d, s)``, ``noise_level(d_clean, snr_db=30.0)``, ``whitened_delta(d_true, d_model, s)``
    Stacked whitened real vectors (see Conventions); s^2 = mean|d_clean|^2 10^(-snr/10) as in ``bayes_scatter.likelihood``.
``flag_nats(n_data, nsigma=3)``, ``loglog_crossing(p, y, level)``
    nsigma sqrt(n_data) nats (the chi^2 flag; 147 for 2400 complex data); power-law fit through (p, y) and the p where it
    reaches ``level`` -> (p_star, slope), for the detectability windows.
``fit_disk(d, tx, rx, p0, s, free=("eps", "R"), iters=10, jac="analytic")`` -> ``(p, D, r)``
    Gauss-Newton with step halving of the whitened misfit of the exact disk model; ``p0`` has length 2 or 4, the result
    has the same length; ``D`` is the whitened Jacobian at the result for ALL parameters of ``p`` (columns in p order, so
    it can go straight into ``analyze_error``); ``r`` is the whitened residual (Phi = r.r/2). Parameters not in ``free``
    stay at their ``p0`` values. A local minimizer, not a global one. A noise-free homogeneous disk is recovered to 1e-9.
``foot_point_R(d, tx, rx, eps_fixed, R0, s, iters=10)`` -> ``(p, D, r)``
    The profile foot point of DESIGN 15.1: eps fixed at ``eps_fixed``, R fitted from ``R0``; same return as ``fit_disk``.
``c20_projector(k, order=20)``
    Callable Q_k on data of shape (..., Nt, Nr) with Nt, Nr multiples of ``order``: Q_k d = (c_k/order) sum_j
    cos(2 pi j k/order) roll(d, (j Nt/order, j Nr/order)), c_0 = c_{order/2} = 1 else 2, k = 0..order/2. The roll is the
    rotation by 2 pi/order of object and acquisition (ch15: 40 tx, 60 rx, shifts (2, 3)). Guaranteed: sum_k Q_k = I,
    Q_k Q_l = delta_kl Q_k, real-linear and commuting with complex conjugation to roundoff.
``c20_fractions(d, order=20)``
    Squared-norm fractions ||Q_k d||^2/||d||^2, k = 0..order/2 (sum to 1).
``lattice_disk_rods(R_D, p, rod_frac=0.3)`` / ``lattice_disk_chi(grid, R_D, p, rod_frac=0.3, eps_rod=4.0, ss=8)``
    Centers (n, 2) of the square lattice (spacing p, one rod at the origin) with |center| <= R_D, and the contrast
    chi = eps - 1 of rods of radius rod_frac p at those centers on a ``bayes_scatter.geometry.Grid``, painted with
    ``phantoms.coverage`` (ss x ss subsamples). The result is made exactly C4-invariant (average over np.rot90) when
    the grid is even and centered at the origin, so boundary-pixel roundoff cannot break the symmetry.
    ``lattice_disk_eps_bar`` is the area average of eps over the disk |r| < R_D of the rod set, 1 + (eps_rod - 1) N rod_area/(pi R_D^2).
``born_grating_gain(forward, kappa_g, width=0.5, amp=0.05)``
    ||Born data|| / ||chi||_L2 for chi = amp w(r) cos(2 pi kappa_g x), w = exp(-r^2/(2 width^2)), ||chi||_L2 = h ||chi||_F;
    ``kappa_g`` scalar or array (cycles per wavelength). Born data sample the object spectrum for |q| <= 2 k0, that is
    kappa_g <= 2: the gain falls steeply above 2 (Gaussian leakage only).
"""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("NPROC", "4")

_FOLDER = Path(__file__).resolve().parent
# the testbed of THIS checkout (never an installed copy): <repo>/.../src
for _p in [_FOLDER, *_FOLDER.parents]:
    if (_p / "src" / "bayes_scatter").is_dir():
        if str(_p / "src") not in sys.path:
            sys.path.insert(0, str(_p / "src"))
        break
if str(_FOLDER) not in sys.path:
    sys.path.insert(0, str(_FOLDER))

import numpy as np
from scipy.special import hankel1, hankel1e, jv, jve

import bloch as _bloch
from bayes_scatter import detectability as _det
from bayes_scatter import likelihood as _lik
from bayes_scatter import phantoms as _ph
from bayes_scatter.mie import mie_coefficients as _mie_coefficients

K0 = 2 * np.pi
PARAM_NAMES = ("eps", "R", "cx", "cy")


# ======================================================================================================================
# Multilayer Mie (copied from notebooks/cloaking/04, cells 4 and 32; see the module docstring)
# ======================================================================================================================
def log_ratios(x, M):
    """J_m'/J_m, H_m'/H_m and ln(H_m/J_m) for m = 0..M at complex arguments x (Re x >= 0), via recurrences on ratios."""
    x = np.asarray(x, complex)
    start = int(max(M + 2, np.abs(x).max())) + 40            # start of the downward (Miller) recurrence
    R = np.zeros((M + 2, x.size), complex)                    # R[m] = J_m / J_{m-1}
    r = np.zeros(x.size, complex)
    for k in range(start, 0, -1):
        r = 1.0 / (2.0 * k / x - r)
        if k <= M + 1:
            R[k] = r
    Hr = np.zeros((M + 2, x.size), complex)                   # Hr[m] = H_m / H_{m-1}, upward recurrence
    h0e = hankel1e(0, x)
    Hr[1] = hankel1e(1, x) / h0e
    for k in range(1, M + 1):
        Hr[k + 1] = 2.0 * k / x - 1.0 / Hr[k]
    mm = np.arange(M + 1)[:, None]
    with np.errstate(all="ignore"):
        ln0 = np.log(h0e / jve(0, x)) + 1j * x - np.abs(x.imag)   # ln(H_0/J_0) from the exponentially scaled forms
        lnHJ = ln0 + np.concatenate([np.zeros((1, x.size)), np.cumsum(np.log(Hr[1:M + 1]) - np.log(R[1:M + 1]), axis=0)])
    return mm / x - R[1:M + 2], mm / x - Hr[1:M + 2], lnHJ


def multilayer(radii, eps, k0=K0, M=None, pol="TM"):
    """Coefficients s_m (m = 0..M) of concentric layers with outer radii `radii`, permittivities `eps`, vacuum outside."""
    r, e = np.asarray(radii, float), np.asarray(eps, complex)
    N = r.size
    M = int(np.ceil(k0 * r[-1])) + 20 if M is None else M
    k = k0 * np.sqrt(e)
    q = np.ones(N, complex) if pol == "TM" else 1.0 / e
    xs = np.concatenate([[k[0] * r[0]], k[1:] * r[:-1], k[1:] * r[1:], [k0 * r[-1]]])   # core, inner and outer layer arguments, vacuum
    Jl, Hl, lnHJ = log_ratios(xs, M)
    L = q[0] * k[0] * Jl[:, 0]
    for j in range(1, N):
        a, b, qk = j, N - 1 + j, q[j] * k[j]
        w1 = (qk * Jl[:, a] - L) / (L - qk * Hl[:, a])
        with np.errstate(all="ignore"):
            w2 = w1 * np.exp(lnHJ[:, b] - lnHJ[:, a])
        L = qk * (Jl[:, b] + w2 * Hl[:, b]) / (1.0 + w2)
    w0 = (k0 * Jl[:, -1] - L) / (L - k0 * Hl[:, -1])
    with np.errstate(all="ignore"):
        return w0 * np.exp(-lnHJ[:, -1])


def exact_data(tx, rx, radii, eps, M=60):
    """Exact scattered line-source data (Nt, Nr) of centered concentric layers; M = 60 orders is far past the last significant one."""
    s, m = multilayer(radii, eps, M=M), np.arange(M + 1)
    rt, rr = np.linalg.norm(tx, axis=1), np.linalg.norm(rx, axis=1)
    pt, pr = np.arctan2(tx[:, 1], tx[:, 0]), np.arctan2(rx[:, 1], rx[:, 0])
    cosm = np.cos(m[:, None, None] * (pr[None, None, :] - pt[None, :, None]))
    return 0.25j * np.einsum("m,mt,mr,mtr->tr", np.where(m == 0, 1.0, 2.0) * s, hankel1(m[:, None], K0 * rt[None]), hankel1(m[:, None], K0 * rr[None]), cosm)


# ======================================================================================================================
# Homogeneous disk anywhere, analytic in (eps, R, cx, cy)
# ======================================================================================================================
def _disk_field(tx, rx, R, eps, center, M):
    """Mie series of one homogeneous disk at ``center`` (real parameters), shape (Nt, Nr)."""
    n = np.arange(M + 1)
    a_n = _mie_coefficients(M, R, eps, K0)
    xt, yt = tx[:, 0] - center[0], tx[:, 1] - center[1]
    xr, yr = rx[:, 0] - center[0], rx[:, 1] - center[1]
    rt, rr = np.hypot(xt, yt), np.hypot(xr, yr)
    c = (xt[:, None] * xr[None, :] + yt[:, None] * yr[None, :]) / (rt[:, None] * rr[None, :])   # cos(phi_r - phi_t)
    T = np.empty((M + 1,) + c.shape)                                                          # Chebyshev T_n(c) = cos(n dphi)
    T[0] = 1.0
    if M >= 1:
        T[1] = c
    for k in range(2, M + 1):
        T[k] = 2.0 * c * T[k - 1] - T[k - 2]
    w = np.where(n == 0, 1.0, 2.0) * a_n
    return 0.25j * np.einsum("m,mt,mr,mtr->tr", w, hankel1(n[:, None], K0 * rt[None]), hankel1(n[:, None], K0 * rr[None]), T)


def _coeff_derivs(M, R, eps):
    """(a_n, d a_n/d eps, d a_n/d R) for n = 0..M, analytic (Bessel ODE for the second derivatives)."""
    n = np.arange(M + 1)
    k0, k1 = K0, K0 * np.sqrt(complex(eps))
    x1, x0 = k1 * R, k0 * R

    def fam(f, x):
        f0, fm, fp = f(n, x), f(n - 1, x), f(n + 1, x)
        f1 = 0.5 * (fm - fp)
        return f0, f1, -f1 / x - (1.0 - n**2 / x**2) * f0

    j1, dj1, d2j1 = fam(jv, x1)
    j0, dj0, d2j0 = fam(jv, x0)
    h0, dh0, d2h0 = fam(hankel1, x0)
    num = k1 * dj1 * j0 - k0 * j1 * dj0
    den = k0 * j1 * dh0 - k1 * dj1 * h0
    out = []
    for dk1, dx1, dx0 in ((k1 / (2 * eps), R * k1 / (2 * eps), 0.0), (0.0, k1, k0)):     # eps, then R
        dnum = (dk1 * dj1 * j0 + k1 * d2j1 * dx1 * j0 + k1 * dj1 * dj0 * dx0 - k0 * dj1 * dx1 * dj0 - k0 * j1 * d2j0 * dx0)
        dden = (k0 * dj1 * dx1 * dh0 + k0 * j1 * d2h0 * dx0 - dk1 * dj1 * h0 - k1 * d2j1 * dx1 * h0 - k1 * dj1 * dh0 * dx0)
        out.append((dnum * den - num * dden) / den**2)
    return num / den, out[0], out[1]


def _default_M(R, eps):
    return int(np.ceil(max(abs(K0 * np.sqrt(complex(eps))), K0) * abs(R))) + 25


def mie_disk(eps, R, center, tx, rx, M=None):
    """Exact scattered data (Nt, Nr) of a homogeneous disk (eps, R) centered at ``center`` (antennas must lie outside)."""
    tx, rx = np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float))
    center = np.asarray(center, float)
    return _disk_field(tx, rx, R, eps, center, _default_M(R, eps) if M is None else M)


def disk_data(eps, R, center, tx, rx, M=None):
    """Alias of ``mie_disk`` with the argument order of the plan: ``disk_data(eps, R, center, tx, rx)``."""
    return mie_disk(eps, R, center, tx, rx, M)


def _unpack(p, names=None):
    p = np.asarray(p, float)
    if p.size not in (2, 4):
        raise ValueError("p must be (eps, R) or (eps, R, cx, cy)")
    p4 = np.concatenate([p, np.zeros(4 - p.size)])
    return p4, (PARAM_NAMES[:p.size] if names is None else tuple(names))


def _model(p4, tx, rx, M=None):
    return mie_disk(p4[0], p4[1], p4[2:], tx, rx, M)


def _analytic_tangents(p4, tx, rx, names, M):
    eps, R = p4[0], p4[1]
    n = np.arange(M + 1)
    a_n, da_eps, da_R = _coeff_derivs(M, R, eps)
    out = {}
    wgt = np.where(n == 0, 1.0, 2.0)
    xt, yt = tx[:, 0] - p4[2], tx[:, 1] - p4[3]
    xr, yr = rx[:, 0] - p4[2], rx[:, 1] - p4[3]
    rt, rr = np.hypot(xt, yt), np.hypot(xr, yr)
    kt, kr = K0 * rt, K0 * rr
    Ht, Hr = hankel1(n[:, None], kt[None]), hankel1(n[:, None], kr[None])
    dphi = np.arctan2(yr, xr)[None, :] - np.arctan2(yt, xt)[:, None]
    cosn, sinn = np.cos(n[:, None, None] * dphi[None]), np.sin(n[:, None, None] * dphi[None])
    for nm, da in (("eps", da_eps), ("R", da_R)):
        if nm in names:
            out[nm] = 0.25j * np.einsum("m,mt,mr,mtr->tr", wgt * da, Ht, Hr, cosn)
    if "cx" in names or "cy" in names:
        def dH(Hf, x):                                   # d/dx H_n(k0 x) / k0 = (H_{n-1} - H_{n+1})/2
            Hm = np.vstack([-hankel1(1, x)[None], Hf[:-1]])
            Hp = hankel1(n[:, None] + 1, x[None])
            return 0.5 * (Hm - Hp)
        Dt, Dr = K0 * dH(Ht, kt), K0 * dH(Hr, kr)
        ct, st_, cr, sr = xt / rt, yt / rt, xr / rr, yr / rr
        ww = (wgt * a_n)[:, None, None]
        nn = n[:, None, None]
        # derivatives with respect to the positions of the t-th transmitter and the r-th receiver (relative to the disk)
        dxr = np.einsum("mtr->tr", ww * Ht[:, :, None] * (Dr[:, None, :] * cr[None, None, :] * cosn + Hr[:, None, :] * nn * sinn * sr[None, None, :] / rr[None, None, :]))
        dyr = np.einsum("mtr->tr", ww * Ht[:, :, None] * (Dr[:, None, :] * sr[None, None, :] * cosn - Hr[:, None, :] * nn * sinn * cr[None, None, :] / rr[None, None, :]))
        dxt = np.einsum("mtr->tr", ww * Hr[:, None, :] * (Dt[:, :, None] * ct[None, :, None] * cosn - Ht[:, :, None] * nn * sinn * st_[None, :, None] / rt[None, :, None]))
        dyt = np.einsum("mtr->tr", ww * Hr[:, None, :] * (Dt[:, :, None] * st_[None, :, None] * cosn + Ht[:, :, None] * nn * sinn * ct[None, :, None] / rt[None, :, None]))
        out["cx"] = -0.25j * (dxr + dxt)                 # moving the disk by +dc moves every antenna by -dc relative to it
        out["cy"] = -0.25j * (dyr + dyt)
    return out


def disk_tangents(p, tx, rx, names=None, jac="analytic", step=1e-6):
    """Tangents (k, Nt, Nr) of the disk data at ``p`` with respect to ``names`` (default: all of p).

    ``jac="analytic"`` (default; ``"cs"`` is accepted as an alias): closed-form derivatives of the Mie coefficients and of
    the position dependence; exact to roundoff, so symmetry properties of the tangents hold to 1e-14.
    ``jac="fd"``: central difference with ``step`` (about 1e-10 relative accuracy; it breaks exact symmetries at that level).
    """
    p4, names = _unpack(p, names)
    tx, rx = np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float))
    M = _default_M(p4[1], p4[0])
    if jac in ("analytic", "cs"):
        d = _analytic_tangents(p4, tx, rx, names, M)
        return np.array([d[nm] for nm in names])
    if jac != "fd":
        raise ValueError("jac must be 'analytic' or 'fd'")
    out = []
    for nm in names:
        i = PARAM_NAMES.index(nm)
        pp, pm = p4.copy(), p4.copy()
        pp[i] += step
        pm[i] -= step
        out.append((_model(pp, tx, rx, M) - _model(pm, tx, rx, M)) / (2 * step))
    return np.array(out)


def disk_jacobian(p, tx, rx, s, names=None, jac="analytic"):
    """Whitened Jacobian D (2 Nt Nr, k) of the disk model at p (columns in the order of ``names``)."""
    return _det.stack_real(disk_tangents(p, tx, rx, names, jac), s).T


# ======================================================================================================================
# Ring geometries and the effective-permittivity formulas
# ======================================================================================================================
def ring_mean_eps(f, eps1, eps2):
    """Area-fraction average f eps1 + (1 - f) eps2 of the layers (the TM leading order)."""
    return f * eps1 + (1.0 - f) * eps2


def rings(R_D, n_p, f, eps1, eps2, cut="period"):
    """Concentric two-phase rings of period p = R_D/n_p; returns (outer radii, eps), core first, last radius R_D. See the module docstring."""
    c = {"period": 0.0, "mid": 0.5 * (1.0 + f)}[cut] if isinstance(cut, str) else float(cut) % 1.0
    p = R_D / n_p
    ph0 = (c - n_p) % 1.0                      # phase of the profile at r = 0, so that the phase at R_D is c
    k = np.arange(-1, int(np.ceil(n_p)) + 2)
    cand = np.concatenate([(k - ph0) * p, (k + f - ph0) * p])
    tol = 1e-12 * R_D
    inner = np.sort(cand[(cand > tol) & (cand < R_D - tol)])
    edges = np.concatenate([[0.0], inner, [R_D]])
    mid = 0.5 * (edges[:-1] + edges[1:])
    phase = (mid / p + ph0) % 1.0
    eps = np.where(phase < f, eps1, eps2).astype(float)
    return edges[1:].copy(), eps


def rytov_tm(f, eps1, eps2, p, k0=K0):
    """Rytov's second-order correction of the TM ordinary wave in a laminate (also the O(k0^2) term for ring microstructure): k0^2 p^2 f^2 (1-f)^2 (eps1-eps2)^2 / 12."""
    return k0**2 * p**2 * f**2 * (1 - f) ** 2 * (eps1 - eps2) ** 2 / 12.0


def rytov_eps_eff(f, eps1, eps2, p, k0=K0):
    """<eps> + Rytov correction (accurate to O(k0^4))."""
    return ring_mean_eps(f, eps1, eps2) + rytov_tm(f, eps1, eps2, p, k0)


def tm_second_order_1d(f, eps1, eps2, p, k0=K0, nterms=100000):
    """k0^2 sum_{G != 0} |eps_G|^2/|G|^2 for the 1D laminate (G = 2 pi n/p, eps_n = (eps1-eps2) sin(pi n f)/(pi n)), by direct summation plus the n^-4 tail."""
    n = np.arange(1, nterms + 1, dtype=float)
    terms = np.sin(np.pi * n * f) ** 2 / n**4
    tail = 0.5 / (3.0 * nterms**3)                      # mean of sin^2 is 1/2; integral of n^-4 from N
    s = 2.0 * (terms.sum() + tail)                       # n and -n
    return k0**2 * (eps1 - eps2) ** 2 * s * p**2 / (np.pi**2 * 4 * np.pi**2)


def tm_second_order_sum(lat, rods, gmax, eps_bg=1.0, units="a_over_lambda"):
    """Sum over reciprocal vectors 0 < |G| <= gmax of |eps_G|^2/|G|^2 (eps_G from ``bloch.eps_hat``); see the module docstring for the units."""
    G, _ = _bloch.gvectors(lat, gmax, 0.0)
    g2 = np.sum(G**2, axis=1)
    nz = g2 > 1e-18
    eG = _bloch.eps_hat(lat, rods, G[nz], eps_bg)
    S = float(np.sum(np.abs(eG) ** 2 / g2[nz]))
    if units == "k0":
        return S
    if units == "a_over_lambda":
        return S * (2 * np.pi) ** 2 / lat.a**2        # k0^2 S = (2 pi a/lambda)^2 S / a^2
    raise ValueError("units must be 'a_over_lambda' or 'k0'")


def eps_eff_pwe(lat, rods, K, G, B=None, eps_bg=1.0):
    """(eps_eff, k0): lowest TM band at the physical wave vector K: k0 = 2 pi f/a, eps_eff = |K|^2/k0^2."""
    K = np.asarray(K, float)
    f = _bloch.tm_solve(lat, rods, K, G, 1, eps_bg=eps_bg, B=B)[0][0]
    k0 = 2 * np.pi * f / lat.a
    return float(K @ K / k0**2), float(k0)


def tm_residual_k04(lat, rods, K, G, S, B=None, eps_bg=1.0):
    """(eps_eff - <eps> - k0^2 S)/k0^4 with S = ``tm_second_order_sum(..., units='k0')`` (O(1) in k0 if the expansion is right)."""
    eps_eff, k0 = eps_eff_pwe(lat, rods, K, G, B, eps_bg)
    avg = float(np.real(_bloch.eps_hat(lat, rods, np.zeros((1, 2)), eps_bg)[0]))
    return (eps_eff - avg - k0**2 * S) / k0**4


# ======================================================================================================================
# Whitening, noise, fits
# ======================================================================================================================
def whiten(d, s):
    """sqrt(2)/s [Re d.ravel(); Im d.ravel()] over the last two axes (``detectability.stack_real``)."""
    return _det.stack_real(d, s)


def noise_level(d_clean, snr_db=30.0):
    """s = sqrt(mean|d_clean|^2 10^(-snr/10)) (``likelihood.noise_std_from_snr``)."""
    return _lik.noise_std_from_snr(d_clean, snr_db)


def whitened_delta(d_true, d_model, s):
    """Whitened model error whiten(d_true - d_model, s), the ``delta`` of ``detectability.analyze_error``."""
    return whiten(np.asarray(d_true) - np.asarray(d_model), s)


def flag_nats(n_data, nsigma=3.0):
    """Misfit excess (nats) of the nsigma chi^2 flag for ``n_data`` complex data: Phi = sum |r|^2/s^2 has mean n_data and sd sqrt(n_data), so nsigma sqrt(n_data) (147 for 2400 data, nsigma = 3)."""
    return float(nsigma * np.sqrt(n_data))


def loglog_crossing(p, y, level):
    """(p_star, slope): least-squares power law y = c p^slope through the points (p, y) (all > 0) and the p_star where it equals ``level``."""
    slope, icpt = np.polyfit(np.log(p), np.log(y), 1)
    return float(np.exp((np.log(level) - icpt) / slope)), float(slope)


def fit_disk(d, tx, rx, p0, s, free=("eps", "R"), iters=10, jac="analytic"):
    """Gauss-Newton fit of the exact disk model to complex data ``d``; returns (p, D (2N, len(p)), r (2N,)); see the module docstring."""
    p, names = _unpack(p0)
    npar = len(names)
    free = tuple(nm for nm in names if nm in free)
    cols = [PARAM_NAMES.index(nm) for nm in free]
    tx, rx = np.atleast_2d(np.asarray(tx, float)), np.atleast_2d(np.asarray(rx, float))
    yw = whiten(d, s)

    def resid(q):
        return yw - whiten(_model(q, tx, rx), s)

    r = resid(p)
    phi = 0.5 * float(r @ r)
    for _ in range(iters):
        D = disk_jacobian(p, tx, rx, s, free, jac)
        dp = np.linalg.lstsq(D, r, rcond=None)[0]
        t = 1.0
        while t > 1e-4:
            q = p.copy()
            q[cols] += t * dp
            try:
                rq = resid(q)
                phq = 0.5 * float(rq @ rq)
            except Exception:
                phq = np.inf
            if np.isfinite(phq) and phq <= phi * (1 + 1e-14):
                break
            t *= 0.5
        else:
            break
        p, r, phi = q, rq, phq
        if np.max(np.abs(t * dp)) <= 1e-14 * max(1.0, np.max(np.abs(p))):
            break
    D = disk_jacobian(p, tx, rx, s, names, jac)
    return p[:npar].copy(), D, r


def foot_point_R(d, tx, rx, eps_fixed, R0, s, iters=10):
    """Profile foot point: eps fixed at ``eps_fixed``, R least squares from ``R0``; returns (p = (eps_fixed, R), D, r) like ``fit_disk``."""
    return fit_disk(d, tx, rx, [eps_fixed, R0], s, free=("R",), iters=iters)


# ======================================================================================================================
# C20 isotypic projectors of the ch15 acquisition
# ======================================================================================================================
def c20_projector(k, order=20):
    """Callable Q_k on data (..., Nt, Nr): the real isotypic projector of the cyclic group of order ``order`` acting by rolls."""
    if not 0 <= k <= order // 2:
        raise ValueError("k must be in 0..order/2")
    ck = 1.0 if k in (0, order // 2) else 2.0

    def Q(d):
        d = np.asarray(d)
        nt, nr = d.shape[-2:]
        if nt % order or nr % order:
            raise ValueError("data dimensions must be multiples of the group order")
        st, sr = nt // order, nr // order
        out = 0.0
        for j in range(order):
            out = out + np.cos(2 * np.pi * j * k / order) * np.roll(d, (st * j, sr * j), axis=(-2, -1))
        return (ck / order) * out

    return Q


def c20_fractions(d, order=20):
    """Squared-norm fractions of d in the isotypic components Q_k, k = 0..order/2."""
    d = np.asarray(d)
    tot = np.sum(np.abs(d) ** 2)
    return np.array([np.sum(np.abs(c20_projector(k, order)(d)) ** 2) / tot for k in range(order // 2 + 1)])


# ======================================================================================================================
# Lattice-filled disk and Born grating gain
# ======================================================================================================================
def lattice_disk_rods(R_D, p, rod_frac=0.3):
    """Centers (N, 2) of the square lattice of spacing p (one rod at the origin) with |center| <= R_D; ``rod_frac`` is only checked (< 0.5)."""
    if not 0 < rod_frac < 0.5:
        raise ValueError("rod_frac must be in (0, 0.5)")
    m = int(np.floor(R_D / p * (1 + 1e-9)))
    i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1), indexing="ij")
    keep = i**2 + j**2 <= (R_D / p) ** 2 * (1 + 1e-9)
    return np.stack([i[keep], j[keep]], axis=1) * p


def lattice_disk_eps_bar(R_D, p, rod_frac=0.3, eps_rod=4.0):
    """Area average of eps over the disk of radius R_D of the rod set: 1 + (eps_rod - 1) N pi (rod_frac p)^2/(pi R_D^2)."""
    N = len(lattice_disk_rods(R_D, p, rod_frac))
    return 1.0 + (eps_rod - 1.0) * N * (rod_frac * p) ** 2 / R_D**2


def lattice_disk_chi(grid, R_D, p, rod_frac=0.3, eps_rod=4.0, ss=8):
    """Contrast chi = eps - 1 (n, n) of the lattice-filled disk on ``grid`` (see the module docstring)."""
    if not 0 < rod_frac < 0.5:
        raise ValueError("rod_frac must be in (0, 0.5)")
    r2, lim = (rod_frac * p) ** 2, (R_D / p) ** 2 * (1 + 1e-9)

    def inside(X, Y):
        mi, mj = np.rint(X / p), np.rint(Y / p)
        return ((X - p * mi) ** 2 + (Y - p * mj) ** 2 <= r2) & (mi**2 + mj**2 <= lim)

    chi = _ph.paint(np.zeros((grid.n, grid.n)), _ph.coverage(grid, inside, ss), eps_rod - 1.0)
    if tuple(grid.center) == (0.0, 0.0):
        chi = 0.25 * (chi + np.rot90(chi) + np.rot90(chi, 2) + np.rot90(chi, 3))
    return chi


def born_grating_gain(forward, kappa_g, width=0.5, amp=0.05):
    """||Born data||/||chi||_L2 of the Gaussian-windowed grating(s); scalar or array ``kappa_g`` in cycles per wavelength."""
    grid = forward.grid
    X, Y = grid.centers()
    w = np.exp(-(X**2 + Y**2) / (2 * width**2))
    ks = np.atleast_1d(np.asarray(kappa_g, float))
    out = []
    for kg in ks:
        chi = amp * w * np.cos(2 * np.pi * kg * X)
        d = np.asarray(forward.born(chi))
        out.append(np.linalg.norm(d) / (grid.h * np.linalg.norm(chi)))
    out = np.array(out)
    return float(out[0]) if np.ndim(kappa_g) == 0 else out
