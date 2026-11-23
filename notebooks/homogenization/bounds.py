"""Closed forms and bound tests for two-phase conductivity in 2D (isotropic phases s1, s2, fraction f1 of phase 1).

Used by notebooks 01 to 03 (and by 02 for the Bergman-Milton support functions).

Conventions
-----------
Phase 1 has conductivity ``s1`` and area fraction ``f1``, phase 2 has ``s2`` and ``f2 = 1 - f1``. Where a function
needs the weak phase it determines it itself (the formulas are symmetric in the phase labels), except
``trace_slacks`` and ``certified_violation``, which require s1 < s2 (phase 1 is the weak phase, f1 its fraction,
as in notebook 03 with theta1 = 0.7). Effective tensors are 2x2 symmetric. Direction arguments are angles in radians
(x axis = 0) or unit vectors of shape (n, 2).

Public functions and their contracts
------------------------------------
``wiener(s1, s2, f1)`` -> (harmonic, arithmetic) means; exact bounds, attained by laminates.
``hs2d(s1, s2, f1)`` -> (lo, hi) Hashin-Shtrikman bounds of an isotropic composite, lo = s_min + f_max/(1/(s_max -
    s_min) + f_min/(2 s_min)), hi the mirror image. Exact (attained by coated disks, rank-2 laminates). Vectorized.
``maxwell_garnett2d(s_inc, s_host, f_inc)`` -> sigma_host (1 + b f)/(1 - b f) with b = (s_inc - s_host)/(s_inc +
    s_host). Equals the HS lower bound when the host is the weaker phase and the HS upper bound when it is the
    stronger one. Approximate model for a dilute disk array (first correction O(f^4)).
``bruggeman2d(s1, s2, f1)`` -> the positive root of f1 (s1 - s)/(s1 + s) + f2 (s2 - s)/(s2 + s) = 0. Approximate.
``laminate_seq(sA, sB, thetaB, dirs, weights)`` -> (2,2) tensor of the sequential laminate with matrix A and core B
    (fraction thetaB), layer normals ``dirs`` and weights m_k (sum 1): thetaB inv(A - sA I) = (sB - sA)^-1 I +
    (thetaA/sA) sum_k m_k e_k e_k^T. Exact (Francfort-Murat / Tartar formula, quoted from memory and checked here
    against the Wiener means and HS values). Layer normal e_k = x gives the harmonic mean along x.
``rank2_curve(s1, s2, f1, m, matrix="weak")`` -> (n, 2) eigenvalues (lam_x, lam_y) of the rank-2 laminate with
    normals x (weight m) and y (weight 1 - m): matrix = "weak" (phase 1 is the matrix) traces the lower trace curve,
    "strong" the upper.
``trace_slacks(l1, l2, s1, s2, f1)`` -> (lower_slack, upper_slack): right side minus left side of
    1/(l1 - s1) + 1/(l2 - s1) <= 1/(lh - s1) + 1/(la - s1) and
    1/(s2 - l1) + 1/(s2 - l2) <= 1/(s2 - lh) + 1/(s2 - la), with lh, la the Wiener means; >= 0 inside the G-closure
    (quoted from memory, validated here by equality on the rank-2 laminates).
``certified_violation(L, U, s1, s2, f1, tol=1e-12)`` -> bool: True only if EVERY tensor with L <= sigma <= U violates
    the Wiener box or a trace inequality (eigenvalues of U for the lower bounds, of L for the upper ones, using the
    monotonicity of the eigenvalues in the Loewner order). A True result for a certified bracket is a bug (or a false
    inequality). False means "not certified", not "inside".
``bm_support_f(s, f, phis)`` -> h(phi) = f max_{z in [0,1]} Re(e^(-i phi)/(s - z)): support function of the set of
    values F(s) = int dmu/(s - z) of positive measures with mass f, s outside [0, 1]. Exact (grid scan plus golden
    refinement to 1e-14 in z).
``bm_support_iso(s, f, phis, nz=4001)`` -> (h, weights): the same with the extra moment m1 = f (1 - f)/2, by LP on
    z_j = j/(nz - 1) (highs-ds). weights (nphi, nz) are the basic optimal measures (at most two nonzero entries).
    Approximate: the grid makes h a lower bound of the continuum support function, off by at most about
    f Dz/dist(s, [0,1])^2 with Dz = 1/(nz - 1).
"""

import numpy as np
from scipy.optimize import linprog


def wiener(s1, s2, f1):
    """(harmonic, arithmetic) means of two phases with fraction f1 of phase 1."""
    s1, s2, f1 = np.broadcast_arrays(np.asarray(s1, float), np.asarray(s2, float), np.asarray(f1, float))
    f2 = 1.0 - f1
    return 1.0 / (f1 / s1 + f2 / s2), f1 * s1 + f2 * s2


def hs2d(s1, s2, f1):
    """Hashin-Shtrikman bounds (lo, hi) for an isotropic two-phase composite in 2D."""
    s1, s2, f1 = np.broadcast_arrays(np.asarray(s1, float), np.asarray(s2, float), np.asarray(f1, float))
    f2 = 1.0 - f1
    swap = s1 > s2
    smin, smax = np.where(swap, s2, s1), np.where(swap, s1, s2)
    fmin, fmax = np.where(swap, f2, f1), np.where(swap, f1, f2)
    same = smax == smin
    d = np.where(same, 1.0, smax - smin)
    lo = smin + fmax / (1.0 / d + fmin / (2.0 * smin))
    hi = smax + fmin / (-1.0 / d + fmax / (2.0 * smax))
    lo, hi = np.where(same, smin, lo), np.where(same, smin, hi)
    return (lo[()], hi[()]) if lo.ndim == 0 else (lo, hi)


def maxwell_garnett2d(s_inc, s_host, f_inc):
    """Maxwell Garnett effective conductivity of circular inclusions (2D)."""
    b = (s_inc - s_host) / (s_inc + s_host)
    return s_host * (1.0 + b * f_inc) / (1.0 - b * f_inc)


def bruggeman2d(s1, s2, f1):
    """Bruggeman effective medium (2D, symmetric): the positive root of s^2 - (f1 - f2)(s1 - s2) s - s1 s2 = 0."""
    f2 = 1.0 - np.asarray(f1, float)
    c = (f1 - f2) * (s1 - s2)
    return 0.5 * (c + np.sqrt(c * c + 4.0 * s1 * s2))


def _unit_vectors(dirs):
    d = np.asarray(dirs, float)
    if d.ndim == 1:
        return np.stack([np.cos(d), np.sin(d)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def laminate_seq(sA, sB, thetaB, dirs, weights):
    """Tensor (2,2) of a sequential laminate: matrix sA I (fraction 1 - thetaB), core sB I (fraction thetaB).

    thetaB inv(A - sA I) = (sB - sA)^-1 I + (thetaA/sA) sum_k m_k e_k e_k^T, thetaA = 1 - thetaB, with layer normals
    e_k (``dirs``: angles or an (n,2) array of vectors) and weights m_k >= 0, sum 1.
    """
    e = _unit_vectors(dirs)
    m = np.asarray(weights, float)
    if abs(m.sum() - 1.0) > 1e-12 or np.any(m < 0):
        raise ValueError("weights must be nonnegative and sum to 1")
    thetaA = 1.0 - thetaB
    rhs = np.eye(2) / (sB - sA) + (thetaA / sA) * np.einsum("k,ki,kj->ij", m, e, e)
    return sA * np.eye(2) + np.linalg.inv(rhs / thetaB)


def rank2_curve(s1, s2, f1, m, matrix="weak"):
    """Eigenvalues (lam_x, lam_y), shape (n, 2), of rank-2 laminates with normals x (weight m) and y (1 - m).

    matrix="weak": phase 1 (weak, fraction f1) is the matrix, phase 2 the core: the lower trace curve.
    matrix="strong": phase 2 is the matrix, phase 1 the core: the upper trace curve.
    """
    out = []
    for mi in np.atleast_1d(np.asarray(m, float)):
        if matrix == "weak":
            A = laminate_seq(s1, s2, 1.0 - f1, [0.0, np.pi / 2], [mi, 1.0 - mi])
        elif matrix == "strong":
            A = laminate_seq(s2, s1, f1, [0.0, np.pi / 2], [mi, 1.0 - mi])
        else:
            raise ValueError("matrix must be 'weak' or 'strong'")
        out.append([A[0, 0], A[1, 1]])
    return np.array(out)


def trace_slacks(l1, l2, s1, s2, f1):
    """(lower_slack, upper_slack) of the two trace inequalities (>= 0 inside the G-closure); needs s1 < s2.

    Division by zero (an eigenvalue equal to a phase value) gives inf or nan, as a signed infinite slack.
    """
    lh, la = wiener(s1, s2, f1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = 1.0 / (lh - s1) + 1.0 / (la - s1) - 1.0 / (l1 - s1) - 1.0 / (l2 - s1)
        upper = 1.0 / (s2 - lh) + 1.0 / (s2 - la) - 1.0 / (s2 - l1) - 1.0 / (s2 - l2)
    return lower, upper


def certified_violation(L, U, s1, s2, f1, tol=1e-12):
    """True if every tensor T with L <= T <= U (Loewner) violates the Wiener box or a trace bound (needs s1 < s2).

    Eigenvalues are monotone in the Loewner order: the true sorted eigenvalues satisfy eig(L) <= eig(T) <= eig(U).
    The lower trace left side decreases in each eigenvalue, so it is evaluated at eig(U); the upper one increases,
    so it is evaluated at eig(L). ``tol`` is the margin against rounding.
    """
    lh, la = wiener(s1, s2, f1)
    eL = np.linalg.eigvalsh(0.5 * (np.asarray(L) + np.asarray(L).T))
    eU = np.linalg.eigvalsh(0.5 * (np.asarray(U) + np.asarray(U).T))
    if eU[0] < lh - tol or eL[1] > la + tol:  # the sorted eigenvalue bounds already leave the Wiener box
        return True
    lower, _ = trace_slacks(eU[0], eU[1], s1, s2, f1)
    _, upper = trace_slacks(eL[0], eL[1], s1, s2, f1)
    return bool(lower < -tol or upper < -tol)


def _kernel(s, phis, z):
    """Re(e^(-i phi)/(s - z)) on the outer grid (nphi, nz)."""
    return np.real(np.exp(-1j * np.asarray(phis, float))[:, None] / (s - np.asarray(z, float)[None, :]))


def bm_support_f(s, f, phis, nscan=4001):
    """Support function f max_{z in [0,1]} Re(e^(-i phi)/(s - z)) of the f-only Bergman-Milton set, for each phi."""
    phis = np.atleast_1d(np.asarray(phis, float))
    z = np.linspace(0.0, 1.0, nscan)
    r = _kernel(s, phis, z)
    j = np.argmax(r, axis=1)
    lo = z[np.maximum(j - 1, 0)]
    hi = z[np.minimum(j + 1, nscan - 1)]
    ph = np.exp(-1j * phis)

    def g(x):
        return np.real(ph / (s - x))

    a, b = lo.copy(), hi.copy()
    for _ in range(100):  # vectorized ternary search for the maximum of a smooth function on [a, b]
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        left = g(m1) < g(m2)
        a = np.where(left, m1, a)
        b = np.where(left, b, m2)
    best = np.maximum.reduce([g(0.5 * (a + b)), r.max(axis=1), g(np.zeros_like(a)), g(np.ones_like(a))])
    return f * best


def bm_support_iso(s, f, phis, nz=4001, method="lp"):
    """Support function of the isotropic Bergman-Milton set (m0 = f, m1 = f (1 - f)/2): returns (h, weights).

    Variables mu_j >= 0 on z_j = j/(nz - 1) with sum mu = f and sum z mu = f (1 - f)/2, maximizing
    sum mu_j Re(e^(-i phi)/(s - z_j)). ``weights`` has shape (nphi, nz) and holds a basic optimal measure (at most
    two nonzero entries, located at z_l <= (1 - f)/2 <= z_r).
    method="lp": scipy linprog(method="highs-ds"), about 15 ms per direction.
    method="dual": the same discrete problem through its one-dimensional dual (bisection on the multiplier of the
    first-moment constraint, vectorized over directions, about 1 s for 720 directions); the optimal pair is read off
    from the two argmaxes. Agrees with the LP to about 1e-10 (tested).
    """
    phis = np.atleast_1d(np.asarray(phis, float))
    z = np.linspace(0.0, 1.0, nz)
    r = _kernel(s, phis, z)
    zbar = 0.5 * (1.0 - f)
    h = np.empty(len(phis))
    w = np.zeros((len(phis), nz))
    if method == "lp":
        Aeq = np.vstack([np.ones(nz), z])
        beq = np.array([1.0, zbar])  # p = mu/f: sum p = 1, sum z p = zbar
        for i in range(len(phis)):
            res = linprog(-r[i], A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs-ds")
            if res.status != 0:
                raise RuntimeError("LP failed at direction %d: %s" % (i, res.message))
            w[i] = f * res.x
            h[i] = f * float(r[i] @ res.x)
        return h, w
    if method != "dual":
        raise ValueError("method must be 'lp' or 'dual'")
    dz = z - zbar
    B = 4.0 * (r.max(axis=1) - r.min(axis=1)) / min(zbar, 1.0 - zbar) + 1.0
    lo, hi = -B, B.copy()
    for _ in range(90):  # phi(lam) = max_j (r_j - lam dz_j) is convex; its slope is -dz at the argmax
        lam = 0.5 * (lo + hi)
        jstar = np.argmax(r - lam[:, None] * dz[None, :], axis=1)
        go_up = dz[jstar] > 0  # slope negative: the minimizer lies at larger lam
        lo = np.where(go_up, lam, lo)
        hi = np.where(go_up, hi, lam)
    lam = 0.5 * (lo + hi)
    v = r - lam[:, None] * dz[None, :]
    left = z <= zbar + 1e-15
    jl = np.argmax(np.where(left[None, :], v, -np.inf), axis=1)
    jr = nz - 1 - np.argmax(np.where(~left[None, :], v, -np.inf)[:, ::-1], axis=1)
    jr = np.where(np.any(~left), jr, jl)
    for i in range(len(phis)):
        zl, zr = z[jl[i]], z[jr[i]]
        pr = 0.0 if zr == zl else (zbar - zl) / (zr - zl)
        w[i, jl[i]] += f * (1.0 - pr)
        w[i, jr[i]] += f * pr
        h[i] = f * ((1.0 - pr) * r[i, jl[i]] + pr * r[i, jr[i]])
    return h, w
