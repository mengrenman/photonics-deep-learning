"""Validation of notebooks/spatial_dispersion/basislattice.py (checks a to k of spec_basis.md).

Run from the folder notebooks/spatial_dispersion, in the photonics-dl environment:
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 python _build/validate_basis.py
Prints a table of checks and exits with status 1 if any check fails. The plane-wave reference code
(independent of basislattice) is in the first part of this file.
"""
import sys, time
import pathlib
ND = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, ND)
import numpy as np
import scipy.linalg as sla
from scipy.special import j1, jv, hankel1
import rodlattice as rl
import basislattice as bl
from basislattice import Rod
import numpy as np
from scipy.special import j1
import scipy.linalg as sla

def gvecs(n):
    m = np.arange(-n, n + 1)
    M1, M2 = np.meshgrid(m, m, indexing="ij")
    return 2 * np.pi * np.stack([M1.ravel(), M2.ravel()], 1)

def fourier(rods, D, key):
    """Fourier coefficients (1/A) int v(r) exp(-i D.r) of v = 1 + sum_j (v_j - 1) disk_j, at the vectors D (..., 2)."""
    g = np.hypot(D[..., 0], D[..., 1])
    out = np.where(g < 1e-12, 1.0, 0.0).astype(complex)
    for r in rods:
        x = g * r.radius
        form = np.where(x < 1e-9, 1.0, 2 * j1(np.where(x < 1e-9, 1.0, x)) / np.where(x < 1e-9, 1.0, x))
        eta = np.pi * r.radius ** 2
        out = out + (getattr(r, key) - 1) * eta * form * np.exp(-1j * (D @ np.asarray(r.center)))
    return out

def matrix(rods, G, key):
    D = G[:, None, :] - G[None, :, :]
    return fourier(rods, D, key)

def bands_tm(rods, kB, n, fmax):
    G = gvecs(n)
    E = matrix(rods, G, "eps")
    kg = np.asarray(kB)[None, :] + G
    A = np.diag((kg ** 2).sum(1))
    w = sla.eigh(A, E, eigvals_only=True)
    f = np.sqrt(np.abs(w)) / (2 * np.pi)
    return np.sort(f[f < fmax])

def driven_tm(rods, k0, kv, n):
    G = gvecs(n)
    E = matrix(rods, G, "eps")
    kg = np.asarray(kv)[None, :] + G
    A = -np.diag((kg ** 2).sum(1)) + k0 ** 2 * E
    rhs = np.zeros(len(G), complex)
    i0 = len(G) // 2
    rhs[i0] = -1j * k0
    Ef = np.linalg.solve(A, rhs)
    return (E[i0] @ Ef) / Ef[i0]

def driven_te(rods, k0, kv, n):
    """TE, H_z formulation with the inverse rule for 1/eps and the Laurent rule for mu.

    Returns dict(epsT, tensor): epsT is the transverse eps from <B_z> = |k| E_perp / k0 (Faraday) and the
    averaged Ampere law, for a source across k, assuming no longitudinal-transverse mixing (use only when
    k lies in a mirror plane of the basis); tensor is the full 2 x 2 tensor from the two sources, with
    <E> = (i/k0) [eps]^-1 (curl H - J) at G = 0 and <D> from the averaged Ampere law.
    """
    G = gvecs(n)
    i0 = len(G) // 2
    X = np.linalg.inv(matrix(rods, G, "eps"))
    Mu = matrix(rods, G, "mu")
    k = np.asarray(kv, float)
    kg = k[None, :] + G
    Lop = (kg @ kg.T) * X - k0 ** 2 * Mu
    kn = np.hypot(*k)
    Es, Ds, epsT = [], [], None
    for src, J0 in enumerate((k / kn, np.array([-k[1], k[0]]) / kn)):   # along k, across k
        cross = kg[:, 0] * J0[1] - kg[:, 1] * J0[0]
        u = np.linalg.solve(Lop, 1j * cross * X[:, i0])
        W = 1j * np.stack([kg[:, 1], -kg[:, 0]], 1) * u[:, None]
        F = W.copy()
        F[i0] -= J0
        E0 = (1j / k0) * (X[i0] @ F)
        Es.append(E0)
        Ds.append((kn ** 2 * E0 - k * (k @ E0) - 1j * k0 * J0) / k0 ** 2)
        if src == 1:
            Eperp = k0 * (Mu[i0] @ u) / kn
            epsT = kn ** 2 / k0 ** 2 - 1j / (k0 * Eperp)
    return dict(epsT=epsT, tensor=np.array(Ds).T @ np.linalg.inv(np.array(Es).T))


# =============================================================================== checks
RESULTS = []


def rec(name, value, tol, note=""):
    ok = bool(abs(value) <= tol)
    RESULTS.append((name, float(abs(value)), tol, ok, note))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {abs(value):.3e} (tol {tol:.0e}) {note}", flush=True)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-300)


def extrap(ns, vals):
    """Linear extrapolation of vals(n) in 1/(2n+1) from the last two points."""
    x = 1.0 / (2 * np.asarray(ns[-2:]) + 1)
    return vals[-1] + (vals[-1] - vals[-2]) / (x[1] - x[0]) * (0 - x[1])


TWO = [Rod((0.1, 0.05), 0.12, 9), Rod((0.6, 0.45), 0.15, 9)]
FOUR = [Rod((0.2, 0.2), 0.1, 9), Rod((0.7, 0.25), 0.12, 9), Rod((0.3, 0.75), 0.15, 9), Rod((0.8, 0.8), 0.1, 9)]
KB = (1.1, 0.5)
t_start = time.time()

# ---- a. lattice sums against brute force
cases = [((0.0, 0.0), (0.3, 0.0)), ((0.0, 0.0), (0.12, 0.05)), ((0.1, -0.2), (-0.35, 0.4)),
         ((0.2, 0.2), (-0.2, -0.2)), ((0.45, 0.0), (-0.45, 0.0)), ((0.075, 0.0), (-0.075, 0.0))]
worst = 0.0
for k0, kB in [(2.0 + 0.5j, (0.7, 0.3)), (3.5 + 0.4j, (1.9, -0.6)), (1.0 + 0.6j, (0.4, 0.4))]:
    for di, dj in cases:
        worst = max(worst, rel(bl.basis_sums(k0, kB, di, dj, 8), bl.direct_basis_sums(k0, kB, di, dj, 8)))
rec("a. basis_sums vs brute force (3 complex k0, 6 pairs incl. 0.15 apart), max relative", worst, 1e-9)

# ---- b. Hermitian block N
k0, kB = 2.3, (0.9, 0.4)
Gt = bl.sums_table(k0, kB, FOUR, 6)
for i in range(4):
    Gt[i, i] = Gt[i, i] + (np.arange(-6, 7) == 0)
Nb = bl._toeplitz_block(Gt / 1j, 3)
rec("b. block N Hermitian (4 rods, real k0, kB below threshold), relative", np.abs(Nb - Nb.conj().T).max() / np.abs(Nb).max(), 1e-9)

# ---- c. one rod at the origin reproduces rodlattice
for pol in ("TM", "TE"):
    f1 = rl.bloch_frequencies(KB, 0.3, 12.0, pol, 3, fmax=0.8, nscan=400)
    f2 = bl.bloch_frequencies(KB, [Rod((0, 0), 0.3, 12.0)], pol, 3, fmax=0.8, nscan=400)
    ok = len(f1) == len(f2)
    rec(f"c. Bloch frequencies, one rod vs rodlattice, {pol}", np.abs(f1 - f2).max() if ok else 1.0, 1e-9, f"({len(f1)} vs {len(f2)} bands)")
worst = {"TM": 0.0, "TE": 0.0}
for pol in ("TM", "TE"):
    for eps in (12.0, 12 + 0.5j, -20 + 0.3j):
        for f, kv in [(0.2, (0.9, 0.4)), (0.31, (1.7, -0.6))]:
            e1 = rl.driven(2 * np.pi * f, kv, 0.3, eps, pol, 4)[0]
            e2 = bl.driven(2 * np.pi * f, kv, [Rod((0, 0), 0.3, eps)], pol, 4)[0]
            worst[pol] = max(worst[pol], np.abs(np.asarray(e1) - np.asarray(e2)).max())
for pol in ("TM", "TE"):
    rec(f"c. driven, one rod vs rodlattice (eps 12, 12+0.5i, -20+0.3i; 2 points), {pol}", worst[pol], 1e-10)

# ---- d. a vacuum rod changes nothing
vac_places = [(0.5, 0.5), (0.4, 0.0), (0.0, 0.6)]
for pol in ("TM", "TE"):
    f1 = rl.bloch_frequencies(KB, 0.3, 12.0, pol, 3, fmax=0.8, nscan=400)
    f2 = bl.bloch_frequencies(KB, [Rod((0, 0), 0.3, 12.0), Rod((0.5, 0.5), 0.1, 1.0)], pol, 3, fmax=0.8, nscan=400)
    rec(f"d. Bloch frequencies with a vacuum rod, {pol}", np.abs(f1 - f2).max() if len(f1) == len(f2) else 1.0, 1e-10)
k0 = 2.1 + 0.3j
for pol in ("TM", "TE"):
    A1 = bl.kkr_matrix(k0, KB, [Rod((0, 0), 0.3, 12.0)], pol, 3)
    A2 = bl.kkr_matrix(k0, KB, [Rod((0, 0), 0.3, 12.0), Rod((0.5, 0.5), 0.1, 1.0)], pol, 3)
    s1, d1 = np.linalg.slogdet(A1)
    s2, d2 = np.linalg.slogdet(A2)
    rec(f"d. log|det(I - TG)| with a vacuum rod included (full matrix), {pol}", d1 - d2, 1e-10)
worst = 0.0
for pol in ("TM", "TE"):
    for eps in (12.0, 12 + 0.5j):
        for pl in vac_places:
            e1 = bl.driven(2 * np.pi * 0.27, (1.1, 0.4), [Rod((0, 0), 0.3, eps)], pol, 4)[0]
            e2 = bl.driven(2 * np.pi * 0.27, (1.1, 0.4), [Rod((0, 0), 0.3, eps), Rod(pl, 0.08 if pl == (0.5, 0.5) else 0.05, 1.0)], pol, 4)[0]
            worst = max(worst, np.abs(np.asarray(e1) - np.asarray(e2)).max())
rec("d. driven with a vacuum rod (TM, TE; 2 eps; 3 places)", worst, 1e-10)

# ---- e. translation of the whole basis
for pol in ("TM", "TE"):
    cvec = np.array([0.23, -0.31])
    moved = [Rod(tuple(r.d + cvec), r.radius, r.eps, r.mu) for r in TWO]
    fa = bl.bloch_frequencies(KB, TWO, pol, 3, fmax=0.55, nscan=500)
    fb = bl.bloch_frequencies(KB, moved, pol, 3, fmax=0.55, nscan=500)
    rec(f"e. Bloch frequencies after translating the basis, {pol} ({len(fa)} bands)", np.abs(fa - fb).max() if len(fa) == len(fb) else 1.0, 1e-10)
    worst = 0.0
    for f, kv in [(0.2, (0.9, 0.4)), (0.3, (1.5 + 0.2j, -0.7))]:
        ea = bl.driven(2 * np.pi * f, kv, TWO, pol, 4)[0]
        eb = bl.driven(2 * np.pi * f, kv, moved, pol, 4)[0]
        worst = max(worst, np.abs(np.asarray(ea) - np.asarray(eb)).max())
    rec(f"e. driven after translating the basis (one complex k), {pol}", worst, 1e-10)

# ---- f. plane waves, TM
print("\n-- f. plane-wave convergence, TM", flush=True)
NS = [6, 10, 14, 18]
for label, rods in (("2-rod", TWO), ("4-rod", FOUR)):
    check_gap = bl.check_rods(rods)
    worst_band, worst_eps = 0.0, 0.0
    for kB in [(0.9, 0.4), (2.0, 1.0), (0.3, 2.9)]:
        fb = bl.bloch_frequencies(kB, rods, "TM", 4, fmax=0.45, nscan=500)
        errs = []
        for n in NS:
            fp = bands_tm(rods, kB, n, 0.45)
            errs.append(np.abs(fp - fb).max() if len(fp) == len(fb) else np.nan)
        print(f"   {label} kB={kB}: {len(fb)} bands, max |f_PW - f_KKR| for n = {NS}: " + ", ".join(f"{e:.1e}" for e in errs), flush=True)
        worst_band = max(worst_band, errs[-1])
    for f, kv in [(0.2, (0.9, 0.4)), (0.3, (1.5, -0.7)), (0.12, (0.5, 0.3))]:
        e = bl.driven(2 * np.pi * f, kv, rods, "TM", 4)[0]
        vals = [driven_tm(rods, 2 * np.pi * f, kv, n) for n in NS]
        errs = [abs(v - e) / abs(e) for v in vals]
        print(f"   {label} f={f} k={kv}: eps_zz = {e.real:.6f}; relative error vs PW n = {NS}: " + ", ".join(f"{x:.1e}" for x in errs), flush=True)
        worst_eps = max(worst_eps, errs[-1])
    rec(f"f. TM Bloch frequencies vs plane waves (n = {NS[-1]}), {label}, 3 Bloch vectors", worst_band, 2e-4, "(a/lambda0)")
    rec(f"f. TM driven eps_zz vs plane waves (n = {NS[-1]}), {label}, 3 points", worst_eps, 1e-3, "(relative)")

# ---- g. plane waves, TE
print("\n-- g. plane-wave convergence, TE (inverse rule)", flush=True)
for mu in (1.0, 3.0):
    sym = [Rod((0.25, 0.0), 0.15, 4, mu), Rod((-0.2, 0.0), 0.1, 4, mu)]
    gen = [Rod((0.1, 0.05), 0.12, 4, mu), Rod((0.6, 0.45), 0.15, 4, mu)]
    for label, rods, kv, f in [("mirror basis, k along x", sym, (0.9, 0.0), 0.1), ("mirror basis, k along x", sym, (1.8, 0.0), 0.2),
                               ("generic basis, generic k", gen, (0.9, 0.4), 0.2)]:
        k0 = 2 * np.pi * f
        e = bl.driven(k0, kv, rods, "TE", 4)[0]
        kn = np.hypot(*kv)
        ek = np.array(kv) / kn
        ep = np.array([-ek[1], ek[0]])
        eT = (ep @ e @ ep).real
        outs = [driven_te(rods, k0, kv, n) for n in NS]
        vT = [o["epsT"].real for o in outs]
        vF = [(ep @ o["tensor"] @ ep).real for o in outs]
        eE = extrap(NS, vT)
        errs = [abs(v - eT) / abs(eT) for v in vT]
        print(f"   mu={mu} {label}, f={f}, k={kv}: eps_T(MS) = {eT:.6f}; PW (Faraday) for n={NS}: " + ", ".join(f"{v:.5f}" for v in vT)
              + f"; extrapolated {eE:.5f}; rel. err " + ", ".join(f"{x:.1e}" for x in errs), flush=True)
        if label.startswith("mirror"):
            rec(f"g. TE eps_T vs plane waves, mu={mu}, {label}, f={f}, k={kv[0]}, n={NS[-1]}", errs[-1], 1e-2, "(relative)")
            rec(f"g. TE eps_T vs plane waves, extrapolated n -> infinity", abs(eE - eT) / abs(eT), 2e-3, "(relative)")
        else:
            Tm = outs[-1]["tensor"]
            te = np.array([extrap(NS, [o["tensor"][a, b].real for o in outs]) for a in (0, 1) for b in (0, 1)]).reshape(2, 2)
            rec(f"g. TE full tensor vs plane waves, mu={mu}, {label}, f={f} (all elements, relative to the largest element; n={NS[-1]})",
                np.abs(Tm.real - e.real).max() / np.abs(e.real).max(), 1e-2)
            rec(f"g. TE full tensor, extrapolated n -> infinity", np.abs(te - e.real).max() / np.abs(e.real).max(), 2e-3)
            print(f"      MS tensor {np.round(e.real, 5).tolist()}  PW extrapolated {np.round(te, 5).tolist()}", flush=True)

# ---- h. static magnetism
print("\n-- h. static limit", flush=True)
R0, eta = 0.3, np.pi * 0.3 ** 2
f = 1e-3
k0 = 2 * np.pi * f
for eps in (1.0, 4.0):
    for mu in (5.0, 3.0) if eps == 4.0 else (5.0,):
        target = 1 - 1 / (1 + eta * (mu - 1))
        for phi_deg in (0.0, 45.0):
            ph = np.deg2rad(phi_deg)
            ek = np.array([np.cos(ph), np.sin(ph)])
            ep = np.array([-ek[1], ek[0]])
            ks = (0.3 * k0, 0.6 * k0)
            et = [(ep @ bl.driven(k0, k * ek, [Rod((0, 0), R0, eps, mu)], "TE", 4)[0] @ ep).real for k in ks]
            # eps_T(k) = e0 + c k^2 + d k^4, eliminate d with the two k values and the k -> 0 value from a third point
            e00 = (ep @ bl.driven(k0, 1e-3 * k0 * ek, [Rod((0, 0), R0, eps, mu)], "TE", 4)[0] @ ep).real
            c1 = (et[0] - e00) / ks[0] ** 2
            c2 = (et[1] - e00) / ks[1] ** 2
            c = c1 - (c2 - c1) / (ks[1] ** 2 - ks[0] ** 2) * ks[0] ** 2      # removes the k^4 term
            rec(f"h. k^2 coefficient x k0^2 of eps_T, eps={eps}, mu={mu}, k at {phi_deg:.0f} deg vs 1 - 1/(1 + eta (mu - 1))", c * k0 ** 2 - target, 1e-4,
                f"(coefficient x k0^2 = {c * k0 ** 2:.6f}, formula {target:.6f}; eps_T(0) = {e00:.6f})")

# also: eps_T(0) for eps = 4 against plane waves (static Rayleigh value), mu = 1
e00 = (bl.driven(k0, (1e-3 * k0, 0), [Rod((0, 0), R0, 4.0)], "TE", 4)[0][1, 1]).real
vals = [driven_te([Rod((0, 0), R0, 4.0)], k0, (1e-3 * k0, 0.0), n)["epsT"].real for n in NS]
print(f"   static eps_T(0), eps = 4, R = 0.3: MS {e00:.6f}; PW n={NS}: " + ", ".join(f"{v:.5f}" for v in vals) + f"; extrapolated {extrap(NS, vals):.5f}")
rec("h. static eps_T(0) of eps = 4 vs plane waves extrapolated", abs(extrap(NS, vals) - e00) / e00, 2e-3, "(relative)")

# ---- i. reciprocity and reality
print("\n-- i. reciprocity", flush=True)
for mu in (1.0, 3.0):
    rods = [Rod((0.1, 0.05), 0.12, 4.0, mu), Rod((0.6, 0.45), 0.15, 4.0, mu)]
    worst, worst_h = 0.0, 0.0
    for f, kv in [(0.2, (0.9, 0.4)), (0.1, (-1.3, 0.8)), (0.27, (0.4, -2.0))]:
        e1 = bl.driven(2 * np.pi * f, kv, rods, "TE", 4)[0]
        e2 = bl.driven(2 * np.pi * f, (-kv[0], -kv[1]), rods, "TE", 4)[0]
        worst = max(worst, np.abs(e1 - e2.T).max())
        worst_h = max(worst_h, np.abs(e1 - e1.conj().T).max())
    rec(f"i. TE reciprocity |eps(k) - eps(-k)^T|, two rods of different radii, mu={mu}", worst, 1e-9)
    rec(f"i. TE lossless eps Hermitian for real k (eps = eps^dagger), mu={mu}", worst_h, 1e-9)
    asym = np.abs(e1 - e1.T).max()
    print(f"      size of the antisymmetric part of eps(k) at the last point (not real): {asym:.2e}; "
          f"largest imaginary part {np.abs(e1.imag).max():.2e}  (nonzero: the tensor is Hermitian, not real, without inversion symmetry)")
# with inversion symmetry (rods at +-d, equal radii) the tensor is real and symmetric
inv = [Rod((0.2, 0.1), 0.12, 4.0, 3.0), Rod((-0.2, -0.1), 0.12, 4.0, 3.0)]
e1 = bl.driven(2 * np.pi * 0.2, (0.9, 0.4), inv, "TE", 4)[0]
rec("i. TE lossless eps real and symmetric for an inversion-symmetric basis", max(np.abs(e1.imag).max(), np.abs(e1 - e1.T).max()), 1e-9)
rodsTM = [Rod((0.1, 0.05), 0.12, 9.0), Rod((0.6, 0.45), 0.15, 9.0)]
rec("i. TM reciprocity |eps(k) - eps(-k)|, two rods", abs(bl.driven(2 * np.pi * 0.2, (0.9, 0.4), rodsTM, "TM", 4)[0] - bl.driven(2 * np.pi * 0.2, (-0.9, -0.4), rodsTM, "TM", 4)[0]), 1e-9)

# ---- j. molecule
print("\n-- j. molecule", flush=True)


def regular_numeric_C(k0, D, M, nphi=96):
    """Independent derivation of the outgoing -> regular translation: sample and project (no Graf formula)."""
    D = np.asarray(D, float)
    dist = np.hypot(*D)
    rho = 0.5 * dist
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    P = np.stack([rho * np.cos(phi), rho * np.sin(phi)], 1) - D[None, :]
    rr = np.hypot(P[:, 0], P[:, 1])
    th = np.arctan2(P[:, 1], P[:, 0])
    l = np.arange(-M, M + 1)
    C = np.zeros((2 * M + 1, 2 * M + 1), complex)
    for jm, m in enumerate(l):
        F = hankel1(m, k0 * rr) * np.exp(1j * m * th)
        C[:, jm] = (F[None, :] * np.exp(-1j * l[:, None] * phi[None, :])).mean(1) / jv(l, k0 * rho)
    return C


def direct_cluster(k0, rods, pol, M, theta_inc):
    """Plane wave on a cluster of rods; direct multiple scattering with numerically projected translations."""
    N, nl = len(rods), 2 * M + 1
    l = np.arange(-M, M + 1)
    t = np.array([rl.mie_t(l, k0, r.radius, r.eps, pol, r.mu) for r in rods])
    big = np.eye(N * nl, dtype=complex)
    for i in range(N):
        for j in range(N):
            if i != j:
                big[i * nl:(i + 1) * nl, j * nl:(j + 1) * nl] -= t[i][:, None] * regular_numeric_C(k0, rods[j].d - rods[i].d, M)
    kd = k0 * np.array([np.cos(theta_inc), np.sin(theta_inc)])
    inc = np.array([np.exp(1j * kd @ r.d) * (1j ** l) * np.exp(-1j * l * theta_inc) for r in rods])
    b = np.linalg.solve(big, (t * inc).ravel()).reshape(N, nl)
    return b


def far_direct(k0, rods, b, ths):
    l = np.arange(-(b.shape[1] // 2), b.shape[1] // 2 + 1)
    out = np.zeros(len(ths), complex)
    for j, r in enumerate(rods):
        ph = np.exp(-1j * k0 * (np.cos(ths) * r.d[0] + np.sin(ths) * r.d[1]))
        out += ph * (((-1j) ** l)[None, :] * np.exp(1j * ths[:, None] * l[None, :]) @ b[j])
    return out


# j0. translation formulas at sample points
k0 = 2.7
d = np.array([0.31, -0.22])
dist, phd = np.hypot(*d), np.arctan2(d[1], d[0])
rng = np.random.default_rng(3)
Mt = 40
worst_reg = worst_out = worst_c = 0.0
for kk in (k0, 2.2 + 0.4j):
    for _ in range(5):
        # regular: J_m(k r) e^{i m phi} at r = d + r'
        rp = 0.4 * rng.random() * np.array([np.cos(a := 2 * np.pi * rng.random()), np.sin(a)])
        r = d + rp
        Rm = bl.regular_translation(kk, d, 4, Mt)
        lp = np.arange(-Mt, Mt + 1)
        for m in (-3, 0, 2, 4):
            direct = jv(m, kk * np.hypot(*r)) * np.exp(1j * m * np.arctan2(r[1], r[0]))
            trans = (Rm[:, m + 4] * jv(lp, kk * np.hypot(*rp)) * np.exp(1j * lp * np.arctan2(rp[1], rp[0]))).sum()
            worst_reg = max(worst_reg, abs(direct - trans) / max(abs(direct), 1e-3))
        # outgoing: H_l(k |r - d|) e^{i l phi'} at |r| > |d|
        r = (1.0 + rng.random()) * np.array([np.cos(a := 2 * np.pi * rng.random()), np.sin(a)])
        rr_ = r - d
        Om = bl.outgoing_translation(kk, d, Mt, 4)
        mm = np.arange(-Mt, Mt + 1)
        for l0 in (-3, 0, 2, 4):
            direct = hankel1(l0, kk * np.hypot(*rr_)) * np.exp(1j * l0 * np.arctan2(rr_[1], rr_[0]))
            trans = (Om[:, l0 + 4] * hankel1(mm, kk * np.hypot(*r)) * np.exp(1j * mm * np.arctan2(r[1], r[0]))).sum()
            worst_out = max(worst_out, abs(direct - trans) / abs(direct))
    # outgoing -> regular
    Dv = np.array([0.45, 0.2])
    Cm = bl.outgoing_to_regular(kk, Dv, 40)
    Cn = regular_numeric_C(kk, Dv, 8)
    worst_c = max(worst_c, np.abs(Cm[32:49, 32:49] - Cn).max() / np.abs(Cn).max())
rec("j. regular-wave translation at sample points (real and complex k0)", worst_reg, 1e-12, "(relative)")
rec("j. outgoing-wave translation at sample points (real and complex k0)", worst_out, 1e-12, "(relative)")
rec("j. outgoing -> regular matrix vs numerically projected field", worst_c, 1e-12, "(relative to the largest entry)")

# j1. one rod at the origin and one displaced
ths = np.linspace(0, 2 * np.pi, 7, endpoint=False) + 0.2
for pol in ("TM", "TE"):
    for mu in (1.0, 2.0) if pol == "TE" else (1.0,):
        k0 = 2 * np.pi * 0.25
        rod0 = Rod((0, 0), 0.3, 5.0, mu)
        Tm = bl.molecule_t_matrix(k0, [rod0], pol, M=6, Lmol=10)
        t = rl.mie_t(np.arange(-10, 11), k0, 0.3, 5.0, pol, mu)
        rec(f"j. one rod at the origin: T = diag(Mie t), {pol} mu={mu}", np.abs(Tm - np.diag(t)).max(), 1e-12)
        rodd = Rod((0.35, -0.2), 0.25, 5.0, mu)
        Tm = bl.molecule_t_matrix(k0, [rodd], pol, M=6, Lmol=16)
        a0 = (1j ** np.arange(-16, 17)) * np.exp(-1j * np.arange(-16, 17) * 0.7)
        far_m = ((-1j) ** np.arange(-16, 17))[None, :] * np.exp(1j * ths[:, None] * np.arange(-16, 17)[None, :]) @ (Tm @ a0)
        far_d = far_direct(k0, [rodd], direct_cluster(k0, [rodd], pol, 6, 0.7), ths)
        rec(f"j. one rod displaced: far field vs direct, {pol} mu={mu}", np.abs(far_m - far_d).max() / np.abs(far_d).max(), 1e-8, "(relative)")

cluster = [Rod((0.25, 0.1), 0.12, 6.0), Rod((-0.2, 0.2), 0.15, 4.0 + 0.0j), Rod((0.0, -0.3), 0.1, 9.0), Rod((-0.15, -0.1), 0.08, 3.0)]
cluster_mu = [Rod(r.center, r.radius, r.eps, 2.0 if i % 2 else 1.0) for i, r in enumerate(cluster)]
for pol, cl, tag in (("TM", cluster, "mu=1"), ("TE", cluster, "mu=1"), ("TE", cluster_mu, "mu=2 on two rods")):
    k0 = 2 * np.pi * 0.3
    Lm = 18
    Tm = bl.molecule_t_matrix(k0, cl, pol, M=8, Lmol=Lm)
    mm = np.arange(-Lm, Lm + 1)
    worst = 0.0
    for th0 in (0.7, 2.5):
        a0 = (1j ** mm) * np.exp(-1j * mm * th0)
        far_m = ((-1j) ** mm)[None, :] * np.exp(1j * ths[:, None] * mm[None, :]) @ (Tm @ a0)
        far_d = far_direct(k0, cl, direct_cluster(k0, cl, pol, 8, th0), ths)
        worst = max(worst, np.abs(far_m - far_d).max() / np.abs(far_d).max())
    rec(f"j. 4-rod cluster: far field vs direct sum of rod fields, {pol} {tag}", worst, 1e-8, "(relative)")
    S = np.eye(2 * Lm + 1) + 2 * Tm
    rec(f"j. 4-rod cluster: S = I + 2T unitary, {pol} {tag}", np.abs(S.conj().T @ S - np.eye(2 * Lm + 1)).max(), 1e-10)

# ---- k. timing
print("\n-- k. timing", flush=True)
for pol in ("TM", "TE"):
    ts = []
    for _ in range(5):
        t0 = time.time(); bl.driven(2 * np.pi * 0.2, (0.9, 0.4), FOUR, pol, 4); ts.append(time.time() - t0)
    print(f"   driven, N = 4, M = 4, {pol}: median {1e3 * np.median(ts):.1f} ms")
ts = []
for _ in range(20):
    t0 = time.time(); bl.basis_sums(2.3, (0.9, 0.4), (0.1, -0.05), (-0.2, 0.25), 8); ts.append(time.time() - t0)
print(f"   basis_sums (one pair, L = 8): median {1e3 * np.median(ts):.2f} ms")
ts = []
for _ in range(5):
    t0 = time.time(); bl.sums_table(2.3, (0.9, 0.4), FOUR, 8); ts.append(time.time() - t0)
print(f"   sums_table, N = 4, L = 8 (all 16 blocks): median {1e3 * np.median(ts):.1f} ms")


# ---- l. the same lattice described two ways (added after the review of 2026-11-23)
# Four rods at (+-0.25, +-0.25) in a cell of period 1 are a square lattice of period 0.5 with one rod;
# nine rods at spacing 1/3 are one of period 1/3. TE, metal rods in the plasmonic regime, narrow gaps.
worst4 = 0.0
for R in (0.08, 0.10):
    for eps in (-1.5 + 0.03j, -3.0 + 0.02j, -1.3 + 0.003j):
        for f in (0.12, 0.2):
            for kv in ((0.3, 0.0), (0.2, 0.15), (0.05 + 0.02j, -0.03 + 0.01j)):
                rods4 = [Rod((sx * 0.25, sy * 0.25), R, eps) for sx in (1, -1) for sy in (1, -1)]
                worst4 = max(worst4, rel(bl.driven(2 * np.pi * f, kv, rods4, "TE", M=4)[0],
                                         rl.driven(2 * np.pi * f, kv, R, eps, "TE", M=4, a=0.5)[0]))
rec("l. TE driven eps, 4 metal rods per cell (a = 1) vs 1 rod (a = 0.5), 36 cases", worst4, 1e-9)
worst9 = 0.0
for R in (0.08, 0.14):
    for eps in (-1.85 + 0.0j, -1.3 + 0.003j):
        for kv in ((0.03, 0.0), (0.4, 0.25)):
            rods9 = [Rod((i / 3 + 0.05, j / 3 - 0.02), R, eps) for i in range(3) for j in range(3)]
            worst9 = max(worst9, rel(bl.driven(2 * np.pi * 0.148, kv, rods9, "TE", M=8)[0],
                                     rl.driven(2 * np.pi * 0.148, kv, R, eps, "TE", M=8, a=1 / 3)[0]))
rec("l. TE driven eps, 9 metal rods per cell (a = 1, gaps down to 0.053) vs 1 rod (a = 1/3), M = 8", worst9, 1e-6)


# ---- guards
def raises(exc, fn):
    try:
        fn()
    except exc:
        return 0.0
    return 1.0


rec("guard: driven TM with mu != 1 raises NotImplementedError",
    raises(NotImplementedError, lambda: bl.driven(1.0, (0.5, 0.1), [Rod((0, 0), 0.2, 4.0, 2.0)], "TM", 3)), 0.0)
rec("guard: check_rods raises ValueError for overlapping rods and for a rod overlapping its own image",
    raises(ValueError, lambda: bl.check_rods([Rod((0, 0), 0.2, 4.0), Rod((0.3, 0), 0.2, 4.0)]))
    + raises(ValueError, lambda: bl.check_rods([Rod((0, 0), 0.6, 4.0)])), 0.0)

# ---- summary
print("\n==== SUMMARY ====")
for name, v, tol, ok, note in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {v:.3e} | tol {tol:.0e} {note}")
nfail = sum(not r[3] for r in RESULTS)
print(f"{len(RESULTS)} checks, {nfail} failed, total run time {time.time() - t_start:.0f} s")
sys.exit(1 if nfail else 0)
