"""Standalone test of cellsolver.py: prints one PASS/FAIL/DISCREPANCY line per item, exits nonzero on any FAIL.

Set HOM_FULL=1 to add the N = 512 runs (Q1 local rates 256 -> 512 for kappa = 4, 10, 100; about 40 s more).
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

import cellsolver as cs
import rayleigh as ry

T0 = time.perf_counter()
FAILS = []
FULL = os.environ.get("HOM_FULL") == "1"
R90 = np.array([[0.0, -1.0], [1.0, 0.0]])


def check(name, value, tol, target=0.0):
    err = abs(value - target)
    ok = bool(err <= tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: value {value:.10g}, target {target:.10g}, |diff| {err:.2e} <= tol {tol:.1e}")
    if not ok:
        FAILS.append(name)


def check_true(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILS.append(name)


def random_field(N, seed, lo=1.0, hi=10.0, p=0.5):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((N, N)) < p, hi, lo)


# =============================================================== Q1: laminates and symmetry conventions
for axis in (0, 1):
    a = cs.laminate_pixels(64, 32, 10.0, 1.0, axis)
    L, U = cs.q1_bracket(a)
    harm, arith = 1 / (0.5 / 1 + 0.5 / 10), 5.5
    exact = np.diag([harm, arith]) if axis == 0 else np.diag([arith, harm])
    check(f"laminate N = 64, stripes normal to {'xy'[axis]}: U equals Wiener means (max entry error)", np.abs(U - exact).max(), 1e-12)
    check(f"laminate N = 64, stripes normal to {'xy'[axis]}: L equals Wiener means (max entry error)", np.abs(L - exact).max(), 1e-12)

a = random_field(16, 1)
U = cs.q1_effective(a)
check("U symmetric (random N = 16)", abs(U[0, 1] - U[1, 0]), 1e-13)
check("transposed field gives the axis-swapped tensor: U[a.T] = P U[a] P", np.abs(cs.q1_effective(a.T) - U[::-1, ::-1]).max(), 1e-12)
check("x-reflection i -> N-1-i: U_xx, U_yy unchanged, U_xy changes sign",
      np.abs(cs.q1_effective(a[::-1]) - U * np.array([[1, -1], [-1, 1]])).max(), 1e-12)
check("translation invariance: U[roll(a)] = U[a]", np.abs(cs.q1_effective(np.roll(a, (3, 5), (0, 1))) - U).max(), 1e-12)

# stiffness assembled by an explicit element loop (node order (i,j),(i+1,j),(i+1,j+1),(i,j+1), id i*N+j)
N = 4
a = random_field(N, 5, 1.0, 7.0)
Kc, conn = cs.q1_stiffness(a)
Kref = np.zeros((N * N, N * N))
ke = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]]) / 6.0
for i in range(N):
    for j in range(N):
        nodes = [i * N + j, ((i + 1) % N) * N + j, ((i + 1) % N) * N + (j + 1) % N, i * N + (j + 1) % N]
        for p in range(4):
            for q in range(4):
                Kref[nodes[p], nodes[q]] += a[i, j] * ke[p, q]
check("q1_stiffness equals the explicit element loop", np.abs(Kc.toarray() - Kref).max(), 1e-14)
check("q1_stiffness symmetric, rows sum to 0", max(abs(Kc - Kc.T).max(), np.abs(Kc @ np.ones(N * N)).max()), 1e-13)
check("conn of element (1, 2) at N = 4 is nodes (1,2), (2,2), (2,3), (1,3)", float(np.abs(conn[1 * N + 2] - np.array([6, 10, 11, 7])).max()), 0.0)

# =============================================================== Q1: Loewner order and the interchange identity
worst = np.inf
worst_int = 0.0
for seed in range(10):
    a = random_field(32, seed)
    L, U = cs.q1_bracket(a)
    worst = min(worst, np.linalg.eigvalsh(U - L).min())
    asw = cs.swap_phases(a)
    lhs = cs.q1_effective(asw)
    rhs = 10.0 * R90 @ np.linalg.inv(L) @ R90.T
    worst_int = max(worst_int, np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs))
check_true("Loewner order: min eig(U - L) >= -1e-12 on 10 random fields (N = 32)", worst >= -1e-12, f"min eigenvalue {worst:.3e}")
check("discrete interchange identity U[swap] = ab R inv(L[sigma]) R^T (relative Frobenius, 10 fields)", worst_int, 1e-10)
check("swap_phases(a) equals ab/a for a two-valued field", np.abs(cs.swap_phases(a) - 10.0 / a).max(), 1e-14)

# monotone under refinement, and replication equals the generator
chk = {N: cs.checkerboard(N, 1.0, 10.0) for N in (16, 32, 64)}
check("checkerboard(2N) is the 2x2 replication of checkerboard(N)", float(np.abs(cs.checkerboard(32, 1.0, 10.0) - np.kron(chk[16], np.ones((2, 2)))).max()), 0.0)
Us = {N: cs.q1_effective(chk[N])[0, 0] for N in chk}
Ls = {N: cs.q1_dual(chk[N])[0, 0] for N in chk}
check_true("monotone: U(16) >= U(32) >= U(64)", Us[16] >= Us[32] - 1e-12 and Us[32] >= Us[64] - 1e-12, f"{Us[16]:.8f} {Us[32]:.8f} {Us[64]:.8f}")
check_true("monotone: L(16) <= L(32) <= L(64)", Ls[16] <= Ls[32] + 1e-12 and Ls[32] <= Ls[64] + 1e-12, f"{Ls[16]:.8f} {Ls[32]:.8f} {Ls[64]:.8f}")
# monotone for a random fine geometry replicated 2x2 (nested spaces), full tensors
a = random_field(16, 7)
L1, U1 = cs.q1_bracket(a)
L2, U2 = cs.q1_bracket(np.kron(a, np.ones((2, 2))))
check_true("monotone under 2x2 replication of a random field: eig(U1 - U2) >= 0 and eig(L2 - L1) >= 0",
           np.linalg.eigvalsh(U1 - U2).min() >= -1e-12 and np.linalg.eigvalsh(L2 - L1).min() >= -1e-12)

# =============================================================== Keller-Dykhne on the grid
for kappa in (4.0, 10.0, 100.0):
    worst = 0.0
    for N in (16, 32, 64, 128):
        L, U = cs.q1_bracket(cs.checkerboard(N, 1.0, kappa))
        worst = max(worst, abs(L[0, 0] * U[0, 0] / kappa - 1), abs(L[1, 1] * U[1, 1] / kappa - 1))
    check(f"discrete Dykhne |L U/ab - 1| (kappa = {kappa:g}, N = 16..128, xx and yy)", worst, 1e-10)
check("  tighter, kappa = 10, N = 32 and 64: 1e-12 (plan's module validation)",
      max(abs(cs.q1_dual(chk[N])[0, 0] * cs.q1_effective(chk[N])[0, 0] / 10 - 1) for N in (32, 64)), 1e-12)
a = chk[64]
Uxx_sw = cs.q1_effective(cs.swap_phases(a))[1, 1]
check_true("primal-primal product U_xx[sigma] U_yy[swap]/ab - 1 is >= 1e-3 (kappa = 10, N = 64)",
          abs(Us[64] * Uxx_sw / 10 - 1) >= 1e-3, f"{abs(Us[64] * Uxx_sw / 10 - 1):.3e} (relative primal error {Us[64] / np.sqrt(10) - 1:.3e})")

# =============================================================== corner rates and sampled spectral
lam1 = lambda k: 4 / np.pi * np.arctan(k ** -0.5)
rates = {}
for kappa in (4.0, 10.0, 100.0):
    Ns = (16, 32, 64, 128, 256) + ((512,) if FULL else ())
    err = np.array([cs.q1_effective(cs.checkerboard(N, 1.0, kappa))[0, 0] / np.sqrt(kappa) - 1 for N in Ns])
    rates[kappa] = np.log2(err[:-1] / err[1:])
    print(f"INFO  Q1 local rates kappa = {kappa:g}, N = {Ns[0]}..{Ns[-1]}: {np.round(rates[kappa], 4)}; 2 lambda_1 = {2 * lam1(kappa):.4f}")
check_true("kappa = 4, 10: local rates decrease toward 2 lambda_1 from above (rate(128->256) > 2 lambda_1, decreasing)",
          all(rates[k][-1] > 2 * lam1(k) and np.all(np.diff(rates[k]) < 0.002) for k in (4.0, 10.0)))
check("kappa = 4: rate(128 -> 256) vs 2 lambda_1 (printed trend; plan quotes rate(256->512) within 0.01)", rates[4.0][-1], 0.01, 2 * lam1(4.0))
check_true("kappa = 100 pre-asymptotic: rates strictly decreasing and > 2 lambda_1 + 0.03", np.all(np.diff(rates[100.0]) < 0) and rates[100.0][-1] - 2 * lam1(100.0) >= 0.03,
           f"last {rates[100.0][-1]:.4f} vs 2 lambda_1 {2 * lam1(100.0):.4f}")
if FULL:
    check("kappa = 4: rate(256 -> 512) within 0.01 of 2 lambda_1", rates[4.0][-1], 0.01, 2 * lam1(4.0))
    check("kappa = 10: rate(256 -> 512) within 0.01 of 2 lambda_1", rates[10.0][-1], 0.01, 2 * lam1(10.0))
    check_true("kappa = 100: rate(256 -> 512) - 2 lambda_1 >= 0.03", rates[100.0][-1] - 2 * lam1(100.0) >= 0.03, f"{rates[100.0][-1] - 2 * lam1(100.0):.4f}")

t = time.perf_counter()
S, it = cs.spectral_sampled(chk[64])
rel = S[0, 0] / np.sqrt(10) - 1
check("spectral_sampled checkerboard kappa = 10, N = 64: relative error +2.5e-3", rel, 2e-4, 2.5e-3)
check_true("  at most 60 iterations", it <= 60, f"{it} iterations ({time.perf_counter() - t:.2f} s)")
check("  tensor symmetric and isotropic (|S_xx - S_yy|, |S_xy|)", max(abs(S[0, 0] - S[1, 1]), abs(S[0, 1])), 1e-9)
S1, it1 = cs.spectral_sampled(chk[64], directions=(0,))
check_true("  directions=(0,) gives the same S_xx and NaN for the entries not computed", abs(S1[0, 0] - S[0, 0]) < 1e-12 and np.isnan(S1[1, 1]))
Sr, _ = cs.spectral_sampled(random_field(32, 2))
check("  random field: S symmetric", abs(Sr[0, 1] - Sr[1, 0]), 1e-12)
# sampled errors: positive, decreasing, much smaller than Q1 at kappa = 10
errs = []
its = {}
for N in (64, 128, 256):
    S, its[N] = cs.spectral_sampled(cs.checkerboard(N, 1.0, 10.0), directions=(0,))
    errs.append(S[0, 0] / np.sqrt(10) - 1)
q1err256 = cs.q1_effective(cs.checkerboard(256, 1.0, 10.0))[0, 0] / np.sqrt(10) - 1
print(f"INFO  sampled kappa = 10 errors N = 64, 128, 256: {errs}; local rates {np.round(np.log2(np.array(errs[:-1]) / np.array(errs[1:])), 3)}; iterations {its}; Q1 error at 256 {q1err256:.3e}")
check_true("sampled kappa = 10: error positive and decreasing in N", all(e > 0 for e in errs) and errs[0] > errs[1] > errs[2])
check_true("sampled error at N = 256 is < 0.1 x the Q1 error", errs[2] < 0.1 * q1err256, f"{errs[2]:.3e} vs {q1err256:.3e}")
S, it100 = cs.spectral_sampled(cs.checkerboard(128, 1.0, 100.0), directions=(0,))
print(f"INFO  sampled kappa = 100, N = 128: error {S[0, 0] / 10 - 1:+.4f}, {it100} iterations")
check_true("sampled kappa = 100, N = 128: error positive, about 0.17 (plan 0.17), iterations in 60..150",
           0.1 < S[0, 0] / 10 - 1 < 0.25 and 60 <= it100 <= 150)
# the sampled scheme is exact for a grid-aligned laminate in the parallel direction
Sl, _ = cs.spectral_sampled(cs.laminate_pixels(32, 16, 10.0, 1.0, 0))
check("sampled scheme, laminate normal to x: S_yy = arithmetic mean", Sl[1, 1], 1e-10, 5.5)
KX, KY = cs.spectral_wavenumbers(8)
check("spectral_wavenumbers: Nyquist zero, integer multiples of 2 pi", float(np.abs(KX[:, 0] - 2 * np.pi * np.array([0, 1, 2, 3, 0, -3, -2, -1])).max()), 1e-14)

# =============================================================== energy density (envelope theorem)
N = 16
a = random_field(N, 11, 1.0, 5.0)
U, chi = cs.q1_effective(a, return_fields=True)
E = cs.q1_energy_density(a, chi)
check("sum_e a_e E_e = U_ii (both directions)", max(abs(np.sum(a * E[0]) - U[0, 0]), abs(np.sum(a * E[1]) - U[1, 1])), 1e-12)
check("chi has zero mean", np.abs(chi.mean(axis=(1, 2))).max(), 1e-13)
Uchk = cs.q1_effective(a)
check("return_fields does not change U", np.abs(Uchk - U).max(), 1e-14)
h = 1e-6
worst = 0.0
for (i, j) in ((0, 0), (5, 9), (12, 3)):
    for d in (0, 1):
        ap, am = a.copy(), a.copy()
        ap[i, j] += h
        am[i, j] -= h
        fd = (cs.q1_effective(ap)[d, d] - cs.q1_effective(am)[d, d]) / (2 * h)
        worst = max(worst, abs(fd - E[d, i, j]) / abs(E[d, i, j]))
check("q1_energy_density vs central finite difference (step 1e-6; 3 pixels x 2 directions), relative", worst, 1e-6)
F = cs.q1_center_field(chi, 0)
check("q1_center_field: mean of e_x + grad chi is e_x", np.abs(F.mean(axis=(1, 2)) - np.array([1.0, 0.0])).max(), 1e-13)
check_true("  pixel energy density h^2 |field at center|^2 is a lower estimate of E (Jensen: E >= h^2 |mean field|^2)",
           bool(np.all(E[0] >= (1.0 / N) ** 2 * (F[0] ** 2 + F[1] ** 2) - 1e-15)))

# =============================================================== windows
a = random_field(16, 21)
Ua, Uf = cs.q1_window(a, "affine"), cs.q1_window(a, "flux")
L, U = cs.q1_bracket(a)
ord_ok = [np.linalg.eigvalsh(Ua - U).min(), np.linalg.eigvalsh(U - L).min(), np.linalg.eigvalsh(L - Uf).min()]
check_true("q1_window ordering U_aff >= U_per >= L_per >= L_flux (min eigenvalues of the three differences >= -1e-10)", min(ord_ok) >= -1e-10, str(np.round(ord_ok, 5)))
worst = np.inf
for seed in range(6):
    a = random_field(12, 30 + seed)
    Ua, Uf, (L, U) = cs.q1_window(a, "affine"), cs.q1_window(a, "flux"), cs.q1_bracket(a)
    worst = min(worst, np.linalg.eigvalsh(Ua - U).min(), np.linalg.eigvalsh(U - L).min(), np.linalg.eigvalsh(L - Uf).min())
check_true("  six more windows (N = 12)", worst >= -1e-10, f"min {worst:.2e}")
a = cs.laminate_pixels(16, 8, 10.0, 1.0, 0)
check("window of a grid-aligned laminate across the layers: affine xx = harmonic mean is NOT attained (affine > harmonic) but yy = arithmetic",
      abs(cs.q1_window(a, "affine")[1, 1] - 5.5) + 0.0, 1e-12)
check("window flux bound of the laminate across the layers is exact: L_flux xx = harmonic", cs.q1_window(a, "flux")[0, 0], 1e-12, 1 / (0.5 + 0.05))
check("q1_window affine is symmetric", abs(Ua[0, 1] - Ua[1, 0]), 1e-13)
try:
    cs.q1_window(a, "other")
    check_true("q1_window rejects unknown bc", False)
except ValueError:
    check_true("q1_window rejects unknown bc", True)

# =============================================================== random checkerboard windows (notebook 01 section 5)
worst_ord, worst_dyk = np.inf, 0.0
for seed in range(4):
    a = cs.random_checkerboard(8, 4, 0.5, 1.0, 10.0, seed)
    asw = cs.swap_phases(a)
    Ua, Uf = cs.q1_window(a, "affine"), cs.q1_window(a, "flux")
    L, U = cs.q1_bracket(a)
    worst_ord = min(worst_ord, np.linalg.eigvalsh(Ua - U).min(), np.linalg.eigvalsh(U - L).min(), np.linalg.eigvalsh(L - Uf).min())
    Lsw = cs.q1_dual(asw)
    worst_dyk = max(worst_dyk, abs(np.log(np.linalg.det(U)) + np.log(np.linalg.det(Lsw)) - 2 * np.log(10.0)))
check_true("random checkerboard windows (L = 8, m = 4): ordering U_aff >= U_per >= L_per >= L_flux", worst_ord >= -1e-10, f"min eigenvalue {worst_ord:.2e}")
check("antithetic Dykhne identity |log det U[s] + log det L[swap s] - 2 log(ab)| (4 windows)", worst_dyk, 1e-10)
t = time.perf_counter()
cs.q1_window(cs.random_checkerboard(32, 4, 0.5, 1.0, 10.0, 0), "flux")
print(f"INFO  q1_window flux at N = 128: {time.perf_counter() - t:.2f} s")

# =============================================================== Fourier-Galerkin
R = np.sqrt(0.5 / np.pi)
Lb, Ub, info = cs.fg_bracket("disk", 16, R=R, s_in=10.0, s_out=1.0)
ray = ry.sigma_eff(0.5, 10.0) * 1.0
check_true("FG disk (0.5, 10), K = 16 contains the Rayleigh value 2.4154673802", Lb <= 2.4154673802 <= Ub and Lb <= ray <= Ub, f"[{Lb:.6f}, {Ub:.6f}]")
check("  lower bound equals the scratch value 2.408174", Lb, 1e-5, 2.408174)
check("  upper bound equals the scratch value 2.494377", Ub, 1e-5, 2.494377)
check_true("  converged, residuals below 1e-12", info["converged"], f"iterations {info['iterations']}, residuals {info['residuals']}")
prev_U, prev_L = np.inf, 0.0
mono = True
contains = True
t = time.perf_counter()
cases = {"(0.5, 10)": (np.sqrt(0.5 / np.pi), 10.0), "(0.5, 100)": (np.sqrt(0.5 / np.pi), 100.0), "(0.2827, 12)": (0.3, 12.0)}
brk = {}
for nm, (Rr, kap) in cases.items():
    f = np.pi * Rr ** 2
    refv = ry.sigma_eff(f, kap)
    pu, pl = np.inf, 0.0
    brk[nm] = []
    for K in (8, 16, 32):
        Lk, Uk, inf = cs.fg_bracket("disk", K, R=Rr, s_in=kap, s_out=1.0)
        mono &= (Uk <= pu + 1e-12) and (Lk >= pl - 1e-12)
        contains &= Lk <= refv + 1e-12 and refv <= Uk + 1e-12 and inf["converged"]
        pu, pl = Uk, Lk
        brk[nm].append((K, Lk, Uk))
check_true("FG brackets contain Rayleigh for three cases, K = 8, 16, 32 (all converged)", contains)
check_true("FG monotone in K: U non-increasing, L non-decreasing", mono)
# closure rate between K = 16 and 32 for (0.5, 10)
(K16, L16, U16), (K32, L32, U32) = brk["(0.5, 10)"][1], brk["(0.5, 10)"][2]
slope = np.log((U32 - L32) / (U16 - L16)) / np.log(2.0)
print(f"INFO  FG width slope K = 16 -> 32, (0.5, 10): {slope:.3f} (plan: [-1.15, -0.85] over K = 16-128)")
check_true("FG bracket width closes like 1/K (slope in [-1.15, -0.85] between K = 16 and 32)", -1.15 <= slope <= -0.85)
print(f"INFO  FG disk timing: all cases {time.perf_counter() - t:.2f} s")
Lk, Uk, inf = cs.fg_bracket("disk", 64, R=R, s_in=10.0, s_out=1.0)
check("FG K = 64 (0.5, 10) upper bound equals scratch 2.435077", Uk, 1e-5, 2.435077)
check("FG K = 64 (0.5, 10) lower bound equals scratch 2.413617", Lk, 1e-5, 2.413617)
check_true("FG K = 64 midpoint is biased high (relative >= 1e-3; plan scratch 3.7e-3)", (0.5 * (Lk + Uk) - ray) / ray >= 1e-3, f"{(0.5 * (Lk + Uk) - ray) / ray:.3e}")
ext = lambda K1, K2_, v1, v2: (K2_ * v2 - K1 * v1) / (K2_ - K1)
Lk32, Uk32 = brk["(0.5, 10)"][2][1], brk["(0.5, 10)"][2][2]
check_true("linear-in-1/K extrapolation from K = 32, 64 lands within 5e-4 relative of Rayleigh", abs(ext(32, 64, Uk32, Uk) - ray) / ray < 5e-4 and abs(ext(32, 64, Lk32, Lk) - ray) / ray < 5e-4,
           f"U: {ext(32, 64, Uk32, Uk):.5f}, L: {ext(32, 64, Lk32, Lk):.5f}, Rayleigh {ray:.5f}")

# FG laminates: exact in the parallel direction (U), and across (L); checkerboard contains sqrt(kappa)
for axis in (0, 1):
    Ll, Ul, inf = cs.fg_bracket("laminate", 6, f=0.5, s_in=10.0, s_out=1.0, axis=axis, direction=axis)
    Lp, Up, _ = cs.fg_bracket("laminate", 6, f=0.5, s_in=10.0, s_out=1.0, axis=axis, direction=1 - axis)
    check(f"FG laminate normal {'xy'[axis]}: U parallel to the layers equals arithmetic mean", Up, 1e-10, 5.5)
    check(f"FG laminate normal {'xy'[axis]}: L across the layers equals harmonic mean", Ll, 1e-10, 1 / (0.5 + 0.05))
    check_true(f"  consistent: L_across <= U_across and L_par <= 5.5 <= U_par", Ll <= Ul and Lp <= 5.5 + 1e-10 <= Up + 1e-10)
Lc, Uc, infc = cs.fg_bracket("checkerboard", 16, s_a=1.0, s_b=10.0)
check_true("FG checkerboard K = 16 contains sqrt(10) (Dykhne)", Lc <= np.sqrt(10) <= Uc and infc["converged"], f"[{Lc:.5f}, {Uc:.5f}]")
check("  FG checkerboard keeps the discrete-like Dykhne product L U = ab (exact by symmetry: swap of the checkerboard is a translate and rotate)",
      Lc * Uc, 1e-9, 10.0)
Lsq, Usq, infs = cs.fg_bracket("square", 16, w=0.5, s_in=10.0, s_out=1.0)
Lsq2, Usq2, _ = cs.fg_bracket("square", 24, w=0.5, s_in=10.0, s_out=1.0)
check_true("FG square: brackets nested between K = 16 and 24", Lsq <= Lsq2 + 1e-12 and Usq2 <= Usq + 1e-12, f"[{Lsq:.4f}, {Usq:.4f}] -> [{Lsq2:.4f}, {Usq2:.4f}]")
check_true("FG square: direction 1 gives the same bracket as direction 0 (C4)", abs(cs.fg_bracket("square", 8, w=0.5, s_in=10.0, s_out=1.0, direction=1)[1] - cs.fg_bracket("square", 8, w=0.5, s_in=10.0, s_out=1.0)[1]) < 1e-10)
# a bracket from the Q1 pixel square (same geometry, pixel aligned) must be consistent: FG is conforming for the true square, Q1 for the same square
Lq, Uq = cs.q1_bracket(cs.square_pixels(64, 32, 10.0, 1.0))
check_true("pixel-aligned square: FG (K = 24) and Q1 (N = 64) brackets overlap (same true geometry)", max(Lq[0, 0], Lsq2) <= min(Uq[0, 0], Usq2), f"Q1 [{Lq[0,0]:.4f}, {Uq[0,0]:.4f}], FG [{Lsq2:.4f}, {Usq2:.4f}]")

# Fourier coefficients against the FFT of pixel-aligned fields (box-filtered exactly)
Nf, K2 = 64, 10
def pixel_coeffs(a, K2):
    N = a.shape[0]
    n = np.arange(-K2, K2 + 1)
    F = np.fft.fft2(a) / N ** 2
    c = F[np.ix_(n % N, n % N)]
    ph = np.exp(-1j * np.pi * (n[:, None] + n[None, :]) / N)  # pixel-center phase
    return c * ph * np.sinc(n[:, None] / N) * np.sinc(n[None, :] / N)
for nm, shape, kw, pix in (
    ("square", "square", dict(w=0.5, s_in=10.0, s_out=1.0), cs.square_pixels(Nf, 32, 10.0, 1.0)),
    ("checkerboard", "checkerboard", dict(s_a=1.0, s_b=10.0), cs.checkerboard(Nf, 1.0, 10.0)),
    ("laminate x", "laminate", dict(f=0.5, s_in=10.0, s_out=1.0, axis=0), cs.laminate_pixels(Nf, 32, 10.0, 1.0, 0)),
    ("laminate y", "laminate", dict(f=0.25, s_in=10.0, s_out=1.0, axis=1), cs.laminate_pixels(Nf, 16, 10.0, 1.0, 1)),
):
    c = cs.fg_coefficients(shape, K2, **kw)
    check(f"fg_coefficients('{nm}') equals the box-filtered FFT of the pixel-aligned field (max |diff|, |k| <= {K2})", np.abs(c - pixel_coeffs(pix, K2)).max(), 1e-12)
    check(f"  Hermitian symmetry ({nm})", np.abs(c - np.conj(c[::-1, ::-1])).max(), 1e-14)
# disk: against a 4096^2 supersampled mask (exact to O(1/N) in the boundary), and mean = f (s_in - s_out) + s_out
c = cs.fg_coefficients("disk", 6, R=0.3, s_in=10.0, s_out=1.0)
check("fg_coefficients('disk') mean = s_out + (s_in - s_out) pi R^2", c[6, 6].real, 1e-14, 1.0 + 9.0 * np.pi * 0.09)
x = (np.arange(4096) + 0.5) / 4096
X, Y = np.meshgrid(x, x, indexing="ij")
mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.09) * 9.0 + 1.0
cm = pixel_coeffs(mask, 6)
check("fg_coefficients('disk') vs a 4096^2 supersampled mask (max |diff|, |k| <= 6)", np.abs(c - cm).max(), 2e-4)
c2 = cs.fg_coefficients("disk", 6, R=0.3, s_in=10.0, s_out=1.0, center=(0.2, 0.7))
kk = 2 * np.pi * np.arange(-6, 7)
phase = np.exp(-1j * (kk[:, None] * (0.2 - 0.5) + kk[None, :] * (0.7 - 0.5)))
check("disk center shift multiplies the coefficients by exp(-i k.dc) (off-center terms)", np.abs(np.delete((c2 - c * phase).ravel(), 6 * 13 + 6)).max(), 1e-13)

# =============================================================== generators
a = cs.disk_pixels(64, 0.3, 10.0, 1.0)
check("disk_pixels pixel fraction near pi R^2 (within 1e-2)", (a == 10.0).mean(), 1e-2, np.pi * 0.09)
check("disk_pixels C4v symmetric (transpose, flips)", float(np.abs(a - a.T).max() + np.abs(a - a[::-1]).max() + np.abs(a - a[:, ::-1]).max()), 0.0)
a2 = cs.disk_pixels(64, 0.3, 10.0, 1.0, center=(0.0, 0.0))
check("disk_pixels with a corner center wraps periodically (same pixel count)", float((a2 == 10.0).sum() - (a == 10.0).sum()), 8.0)
check("square_pixels: side 32 of 64 centered, fraction 1/4", (cs.square_pixels(64, 32, 10.0, 1.0) == 10.0).mean(), 1e-14, 0.25)
l = cs.laminate_pixels(8, 4, 10.0, 1.0, 0)
check("laminate_pixels axis = 0 varies along i (rows) only", float(np.abs(l - l[:, :1]).max()), 0.0)
check("laminate_pixels centered stripe: rows 2..5", float(l[2:6, 0].min() - 10.0), 0.0)
r1 = cs.random_checkerboard(4, 3, 0.5, 1.0, 10.0, 0)
r2 = cs.random_checkerboard(4, 3, 0.5, 1.0, 10.0, np.random.default_rng(0))
check("random_checkerboard reproducible by seed and shape (12, 12), constant on tiles",
      float(np.abs(r1 - r2).max()) + (r1.shape != (12, 12)) + float(np.abs(r1 - np.kron(r1[::3, ::3], np.ones((3, 3)))).max()), 0.0)
check("random_checkerboard fraction of b close to p over 64 x 64 tiles", (cs.random_checkerboard(64, 1, 0.3, 1.0, 10.0, 1) == 10.0).mean(), 0.03, 0.3)
check("swap_phases is an involution", float(np.abs(cs.swap_phases(cs.swap_phases(r1)) - r1).max()), 0.0)
g = cs.gaussian_threshold(32, 3.0, 0.7, 4, (1.0, 10.0))
g2 = cs.gaussian_threshold(32, 3.0, 0.7, 4, (1.0, 10.0))
check("gaussian_threshold: weak fraction is round(0.7 N^2)/N^2, reproducible, two-valued", abs((g == 1.0).sum() - round(0.7 * 1024)) + float(np.abs(g - g2).max()) + float(np.sum((g != 1.0) & (g != 10.0))), 0.0)
def corr_len(ell):
    g_ = cs.gaussian_threshold(64, ell, 0.5, 0, (0.0, 1.0)) - 0.5
    return np.mean(g_ * np.roll(g_, 2, 0)) / np.mean(g_ * g_)
check_true("gaussian_threshold: spatial correlation grows with ell", corr_len(1.0) < corr_len(4.0) < corr_len(8.0), f"{corr_len(1.0):.3f} < {corr_len(4.0):.3f} < {corr_len(8.0):.3f}")

# =============================================================== input validation
for bad in (np.ones((4, 5)), -np.ones((4, 4))):
    try:
        cs.q1_effective(bad)
        check_true("q1_effective rejects non-square or non-positive input", False)
    except ValueError:
        pass
check_true("q1_effective rejects non-square or non-positive input", True)

dt = time.perf_counter() - T0
print(f"{'FAILED' if FAILS else 'ALL PASSED'}: {len(FAILS)} failures, {dt:.1f} s")
sys.exit(1 if FAILS else 0)
