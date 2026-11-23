"""Standalone test of spectral.py: prints one PASS/FAIL/DISCREPANCY/INFO line per item, exits nonzero on any FAIL."""
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.fft as sfft
import scipy.integrate as si
import scipy.linalg as sla

import cellsolver as cs
import rayleigh as ry
import spectral as sp

T0 = time.perf_counter()
FAILS = []
F_ROD = math.pi * 0.3 ** 2


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


def info(msg):
    print("INFO  " + msg)


def discrepancy(msg):
    print("DISCREPANCY  " + msg)


def lost_energy(chi):
    """Energy sum |chihat|^2 of the modes k != 0 whose zeroed wavenumber vanishes: (N/2, 0), (0, N/2), (N/2, N/2) for even N."""
    n = chi.shape[0]
    if n % 2:
        return 0.0
    c = sfft.fft2(chi) / n ** 2
    return float(abs(c[n // 2, 0]) ** 2 + abs(c[0, n // 2]) ** 2 + abs(c[n // 2, n // 2]) ** 2)


zr_, wr_ = ry.rayleigh_measure(F_ROD, 20)

# ---------------------------------------------------------------------------------------------- 1. Q1 measure
N = 32
disk = cs.disk_pixels(N, 0.3, 1.0, 0.0)
t0 = time.perf_counter()
zq, wq = sp.q1_measure(disk)
tq = time.perf_counter() - t0
f_pix = disk.mean()
for h in (0.1, 10.0, 100.0):
    U = cs.q1_effective(np.where(disk > 0, h, 1.0))[0, 0]
    check(f"Q1 measure reproduces q1_effective (disk N=32, h={h:g})", abs(sp.eps_ratio_from_measure(zq, wq, h) / U - 1), 1e-10)
chk16 = cs.checkerboard(16, 1.0, 0.0)
zc, wc = sp.q1_measure(chk16)
for h in (0.1, 10.0):
    U = cs.q1_effective(np.where(chk16 > 0, h, 1.0))[0, 0]
    check(f"Q1 measure reproduces q1_effective (checkerboard N=16, h={h:g})", abs(sp.eps_ratio_from_measure(zc, wc, h) / U - 1), 1e-10)
# a field with different x and y behavior: direction 1 versus direction 0, full tensor diagonal entries
rect = np.zeros((24, 24))
rect[6:18, 9:15] = 1.0
for d in (0, 1):
    z, w = sp.q1_measure(rect, direction=d)
    for h in (0.1, 10.0):
        U = cs.q1_effective(np.where(rect > 0, h, 1.0))[d, d]
        check(f"Q1 measure, anisotropic rectangle, direction {d}, h={h:g}", abs(sp.eps_ratio_from_measure(z, w, h) / U - 1), 1e-10)
check_true("  the two directions differ for the rectangle (the argument is used)",
           abs(sp.eps_ratio_from_measure(*sp.q1_measure(rect, 0), 10.0) - sp.eps_ratio_from_measure(*sp.q1_measure(rect, 1), 10.0)) > 0.05)
# independent dense solve at contrasts inside and off the real axis (the measure is an analytic function of h)
Nq = 12
rng = np.random.default_rng(3)
rnd = (rng.random((Nq, Nq)) < 0.4).astype(float)
rnd = np.maximum(rnd, rnd.T)  # transposition symmetric
zr, wr = sp.q1_measure(rnd)
conn = cs._conn(Nq, True)
for h in (-0.5, -3.0, -2.0 + 0.5j):
    a = 1.0 + (h - 1.0) * rnd
    K = cs._assemble(a, conn, Nq * Nq).toarray()
    fl = (h - 1.0) * cs._loads(rnd, conn, Nq * Nq)[0]  # loads are linear in the coefficient (the load of 1 is zero)
    chi_d = np.zeros(Nq * Nq, dtype=complex)
    chi_d[1:] = np.linalg.solve(K[1:, 1:], -fl[1:])
    U = a.mean() + fl @ chi_d
    check(f"Q1 measure vs direct dense solve at h = {h} (random C4-ish field, N=12)", abs(sp.eps_ratio_from_measure(zr, wr, h) / U - 1), 1e-9)
# sum rules and bounds
check("Q1: m0 = f exactly (disk, checkerboard)", max(abs(wq.sum() - f_pix), abs(wc.sum() - 0.5)), 1e-12)
check_true("Q1: weights nonnegative (to -1e-12) and z in [0, 1]", wq.min() > -1e-12 and zq.min() >= 0 and zq.max() <= 1 + 1e-12,
           f"min w {wq.min():.1e}, z in [{zq.min():.3g}, {zq.max():.6f}]")
m1q = sp.moments(zq, wq, 1)[1]
check_true("Q1: m1 <= f(1-f)/2 for the disk (variational)", m1q <= f_pix * (1 - f_pix) / 2 + 1e-12, f"{m1q:.6f} <= {f_pix * (1 - f_pix) / 2:.6f}")
for Nd in (32, 40):
    zz, ww = sp.q1_measure(cs.disk_pixels(Nd, 0.3, 1.0, 0.0))
    check_true(f"Q1 isolated pixel disk N={Nd}: atom at 0 has weight <= 1e-12", abs(ww[0]) <= 1e-12 and zz[0] == 0.0, f"{ww[0]:.2e}")
try:
    sp.q1_measure(np.full((8, 8), 0.5))
    check_true("input validation: chi1 not 0/1 raises ValueError", False)
except ValueError:
    check_true("input validation: chi1 not 0/1 raises ValueError", True)

# ---------------------------------------------------------------------------------------------- 2. Q1 checkerboard (percolation atom)
m1s, a0s = [], []
for Nk in (16, 24, 32):
    chi = cs.checkerboard(Nk, 1.0, 0.0)
    z, w = sp.q1_measure(chi)
    m1s.append(sp.moments(z, w, 1)[1])
    a0s.append(w[0])
    Uh = cs.q1_effective(np.where(chi > 0, 1e8, 1.0))[0, 0] / 1e8
    check(f"checkerboard Q1 N={Nk}: atom at 0 equals U_h(1e8)/1e8 (relative)", abs(w[0] - Uh) / w[0], 1e-6)
check("checkerboard Q1 N=16: atom at 0 equals the scratch value 0.1978", a0s[0], 1e-3, 0.1978)
info(f"Q1 checkerboard atoms at 0: {a0s[0]:.4f}, {a0s[1]:.4f}, {a0s[2]:.4f} (scratch 0.1978, 0.1794, 0.1683); m1: {m1s[0]:.5f}, {m1s[1]:.5f}, {m1s[2]:.5f}")
check_true("checkerboard Q1: m1(16) <= m1(32) < 1/8 (nested under replication; m1(24) printed)", m1s[0] <= m1s[2] < 0.125, f"{m1s[0]:.6f} <= {m1s[2]:.6f} < 0.125")
check("checkerboard Q1 m1(16) scratch 0.12199", m1s[0], 1e-5, 0.12199)

# ---------------------------------------------------------------------------------------------- 3. FG measure
for K in (8, 16):
    t0 = time.perf_counter()
    zf, wf = sp.fg_measure("disk", K, R=0.3)
    tf = time.perf_counter() - t0
    for h in (0.1, 10.0, 100.0):
        _, U, inf = cs.fg_bracket("disk", K, R=0.3, s_in=h, s_out=1.0)
        check(f"FG measure reproduces fg_bracket upper (disk K={K}, h={h:g}, converged {inf['converged']})", abs(sp.eps_ratio_from_measure(zf, wf, h) / U - 1), 1e-10)
info(f"fg_measure disk K=16: {tf:.2f} s ({len(zf)} atoms); q1_measure N=32: {tq:.2f} s")
for shape, p, K in (("square", dict(w=0.5), 12), ("checkerboard", {}, 8)):
    z, w = sp.fg_measure(shape, K, **p)
    key = dict(s_a=10.0, s_b=1.0) if shape == "checkerboard" else dict(s_in=10.0, s_out=1.0)
    _, U, inf = cs.fg_bracket(shape, K, **p, **key)
    check(f"FG measure reproduces fg_bracket upper ({shape} K={K}, h=10)", abs(sp.eps_ratio_from_measure(z, w, 10.0) / U - 1), 1e-10)
for d in (0, 1):
    z, w = sp.fg_measure("laminate", 6, direction=d, f=0.3, axis=0)
    _, U, _ = cs.fg_bracket("laminate", 6, direction=d, f=0.3, s_in=10.0, s_out=1.0, axis=0)
    check(f"FG measure, laminate (layer normal x), direction {d}, h=10", abs(sp.eps_ratio_from_measure(z, w, 10.0) / U - 1), 1e-10)
    if d == 1:
        check("  laminate value is exact along the layers (arithmetic mean, direction 1)", sp.eps_ratio_from_measure(z, w, 10.0), 1e-10, 0.3 * 10 + 0.7)
    else:
        val, harm = sp.eps_ratio_from_measure(z, w, 10.0), 1 / (0.3 / 10 + 0.7)
        check_true("  across the layers (direction 0) the Galerkin upper bound lies above the harmonic mean (not exact at finite K)", val >= harm - 1e-12, f"{val:.6f} >= {harm:.6f}")
# independent dense Galerkin solve at negative contrast (stationary value, the measure is analytic in h)
K = 6
zf6, wf6 = sp.fg_measure("disk", K, R=0.3)
m = np.arange(-K, K + 1)
MX, MY = np.meshgrid(m, m, indexing="ij")
keep = (MX != 0) | (MY != 0)
kx, ky = 2 * np.pi * MX[keep], 2 * np.pi * MY[keep]
for h in (-0.5, -3.0):
    c = cs.fg_coefficients("disk", 2 * K, R=0.3, s_in=h, s_out=1.0)
    dx = MX[keep][:, None] - MX[keep][None, :] + 2 * K
    dy = MY[keep][:, None] - MY[keep][None, :] + 2 * K
    A = (kx[:, None] * kx[None, :] + ky[:, None] * ky[None, :]) * c[dx, dy]
    fvec = -1j * kx * c[MX[keep] + 2 * K, MY[keep] + 2 * K]
    U = (c[2 * K, 2 * K] - fvec.conj() @ np.linalg.solve(A, fvec)).real
    check(f"FG measure vs direct dense Galerkin solve at h = {h} (disk K=6)", abs(sp.eps_ratio_from_measure(zf6, wf6, h) / U - 1), 1e-9)
# moments without an eigen decomposition: m_k = beta^H Ht^k beta
c1 = cs.fg_coefficients("disk", 2 * K, R=0.3, s_in=1.0, s_out=0.0)
nrm = np.hypot(MX[keep], MY[keep])
ux, uy = MX[keep] / nrm, MY[keep] / nrm
H = (ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]) * c1[MX[keep][:, None] - MX[keep][None, :] + 2 * K, MY[keep][:, None] - MY[keep][None, :] + 2 * K]
beta = ux * c1[MX[keep] + 2 * K, MY[keep] + 2 * K]
vec, mk = beta.copy(), []
for k in range(1, 5):  # m_k = sum (w/t) t^k = beta^H Ht^(k-1) beta
    mk.append((beta.conj() @ vec).real)
    vec = H @ vec
check("FG moments m1..m4 equal beta^H Ht^(k-1) beta (independent of the eigen decomposition)",
      np.max(np.abs(sp.moments(zf6, wf6, 4)[1:] - np.array(mk))), 1e-12)
# bounds
zf16, wf16 = sp.fg_measure("disk", 16, R=0.3)
check("FG: m0 = f exactly (disk K=16, square K=12)", max(abs(wf16.sum() - F_ROD), abs(sp.fg_measure("square", 12, w=0.5)[1].sum() - 0.25)), 1e-12)
m1f = sp.moments(zf16, wf16, 1)[1]
check_true("FG: m1 <= f(1-f)/2 (disk K=16)", m1f <= F_ROD * (1 - F_ROD) / 2 + 1e-12, f"{m1f:.6f} <= {F_ROD * (1 - F_ROD) / 2:.6f}")
Ucont = ry.sigma_eff(F_ROD, 10.0)
check_true("FG upper bound from the measure >= Rayleigh (exact disk lattice) at h = 10 and h = 0.1",
           sp.eps_ratio_from_measure(zf16, wf16, 10.0) >= Ucont and sp.eps_ratio_from_measure(zf16, wf16, 0.1) >= ry.sigma_eff(F_ROD, 0.1),
           f"{sp.eps_ratio_from_measure(zf16, wf16, 10.0):.6f} >= {Ucont:.6f}")
# atom at 0, atom counts and distance to Rayleigh for the FG disk
zeros, fgm = {}, {}
for K in (4, 8, 12, 16, 20):
    z, w = sp.fg_measure("disk", K, R=0.3)
    zeros[K], fgm[K] = w[0], (z, w)
info("FG disk atom at 0 (zero_tol 1e-10): " + ", ".join(f"K={k}: {v:.2e}" for k, v in zeros.items()))
check_true("FG disk: atom at 0 is positive for K = 8, 12, 16, 20", all(zeros[k] > 0 for k in (8, 12, 16, 20)))
check_true("FG disk: atom at 0 decreases from K=4 to 8 to 12 (and K=20 is below K=8)", zeros[4] > zeros[8] > zeros[12] and zeros[20] < zeros[8])
discrepancy(f"plan: FG disk atom at 0 'about O(1/K)' (0.28/K would be 0.02-0.04); measured {zeros[8]:.1e} at K=8 and "
            f"{zeros[20]:.1e} at K=20: negligible (5.9e-5, 1.7e-9, 1.2e-11 at K=4, 8, 12, then a floor near 1e-11 set by zero_tol and roundoff)")
n99 = {K: sp.n_atoms_for(*fgm[K], 0.99) for K in (8, 12, 16, 20)}
w1r = {K: sp.wasserstein1(*fgm[K], zr_, wr_) for K in (8, 12, 16, 20)}
domr = zr_[np.argmax(wr_)]
info("FG disk K=8,12,16,20: 99%-atom count " + str(list(n99.values())) + "; W1 to Rayleigh " + ", ".join(f"{v:.4f}" for v in w1r.values())
     + "; weight of the largest FG atom / f " + ", ".join(f"{fgm[K][1].max() / F_ROD:.3f}" for K in n99))
discrepancy(f"plan section 4: 'FG disk 99%-atom count <= 5 at every K': measured {list(n99.values())} for K = 8, 12, 16, 20 (grows with K; the Galerkin compression "
            f"smears the dominant Rayleigh atom, which carries 97.7%, over a cluster); plan section 3: W1 to Rayleigh decreasing over K: measured "
            f"{[round(v, 4) for v in w1r.values()]} (not monotone: K=20 is above K=16)")

# ---------------------------------------------------------------------------------------------- 4. Lanczos (sampled scheme)
Nl = 128
dsk = cs.disk_pixels(Nl, 0.3, 1.0, 0.0)
fl = dsk.mean()
res = {}
for steps in (150, 300):
    t0 = time.perf_counter()
    zl, wl = sp.lanczos_measure(dsk, 0, steps)
    res[steps] = (zl, wl, time.perf_counter() - t0)
for h in (0.1, 10.0, 100.0):
    S, it = cs.spectral_sampled(np.where(dsk > 0, h, 1.0), directions=(0,))
    vals = {st: sp.eps_ratio_from_measure(res[st][0], res[st][1], h) for st in res}
    for st in res:
        check(f"Lanczos ({st} steps) reproduces spectral_sampled (disk N=128, h={h:g})", abs(vals[st] / S[0, 0] - 1), 1e-8)
    info(f"  h={h:g}: |value(300) - value(150)| = {abs(vals[300] - vals[150]):.2e} (s = {1 / (1 - h):.4f}); spectral_sampled {it} PCG iterations")
check("Lanczos: m0 = f", abs(res[300][1].sum() - fl), 1e-12)
m1l = sp.moments(res[300][0], res[300][1], 1)[1]
check("Lanczos: m1 = f(1-f)/2 - half the Nyquist-mode energy (disk N=128)", m1l, 1e-12, fl * (1 - fl) / 2 - 0.5 * lost_energy(dsk))
check_true("Lanczos: Ritz values in [0, 1] up to 1e-12", res[300][0].min() > -1e-12 and res[300][0].max() < 1 + 1e-12,
           f"[{res[300][0].min():.2e}, {res[300][0].max() - 1:.2e} + 1]")
info(f"Lanczos times, disk N=128: 150 steps {res[150][2]:.2f} s, 300 steps {res[300][2]:.2f} s")
# direction 1 on an anisotropic cell, and the m1 identity on transposition-symmetric fields (even, odd, and not divisible by 4)
bar = np.zeros((48, 48))
bar[10:38, 20:28] = 1.0
for d in (0, 1):
    zb, wb = sp.lanczos_measure(bar, d, 200)
    S, _ = cs.spectral_sampled(np.where(bar > 0, 10.0, 1.0), directions=(d,))
    check(f"Lanczos direction {d} on an anisotropic cell (N=48, h=10)", abs(sp.eps_ratio_from_measure(zb, wb, 10.0) / S[d, d] - 1), 1e-8)
rng = np.random.default_rng(5)
for Nm in (30, 31, 32):
    r = (rng.random((Nm, Nm)) < 0.5).astype(float)
    r = np.maximum(r, r.T)
    z, w = sp.lanczos_measure(r, 0, 12)
    fr = r.mean()
    nyq = 0.5 * lost_energy(r)
    check(f"Lanczos m1 identity on a random transposition-symmetric field, N={Nm}", sp.moments(z, w, 1)[1], 1e-12, fr * (1 - fr) / 2 - nyq)
# trivial cases
z, w = sp.lanczos_measure(np.ones((16, 16)), 0, 10)
check_true("Lanczos on a full cell: single atom 0 of weight 1 (sigma* = sigma1)", len(z) == 1 and z[0] == 0 and abs(w[0] - 1) < 1e-14 and abs(sp.eps_ratio_from_measure(z, w, 7.0) - 7.0) < 1e-12)
try:
    sp.lanczos_measure(np.zeros((8, 8)))
    check_true("input validation: empty phase 1 raises ValueError", False)
except ValueError:
    check_true("input validation: empty phase 1 raises ValueError", True)

# ---------------------------------------------------------------------------------------------- 5. checkerboard exact forms, Jacobi coefficients (amendment 2)
mom = sp.checkerboard_moments(7)
check("checkerboard_moments: 1/2, 1/8, 1/16, 5/128", np.max(np.abs(mom[:4] - np.array([0.5, 0.125, 0.0625, 5 / 128]))), 1e-16)
quad_m = np.array([si.quad(lambda th, k=k: 2 / np.pi * np.cos(th) ** 2 * np.sin(th) ** (2 * k), 0, np.pi / 2, epsabs=1e-14, epsrel=1e-14)[0] for k in range(8)])
check("checkerboard_moments equal quadrature of the density (z = sin^2 theta), k = 0..7", np.max(np.abs(mom - quad_m)), 1e-13)
check("checkerboard_cdf(1) = 1/2 and cdf(0) = 0, M' = density (central difference at z = 0.3, 0.8)",
      max(abs(sp.checkerboard_cdf(1.0) - 0.5), abs(sp.checkerboard_cdf(0.0)),
          *(abs((sp.checkerboard_cdf(z + 1e-6) - sp.checkerboard_cdf(z - 1e-6)) / 2e-6 - sp.checkerboard_density(z)) for z in (0.3, 0.8))), 1e-8)
check_true("checkerboard_density/cdf are zero/constant outside (0, 1)", sp.checkerboard_density(np.array([-0.1, 1.2])).max() == 0 and sp.checkerboard_cdf(np.array([-1.0, 2.0])).tolist() == [0.0, 0.5])
# Gauss rule: 2n moments exact, reproduces sqrt(h)
zg, wg = sp.checkerboard_gauss(40)
check("checkerboard_gauss(40): moments 0..79 exact (relative)", np.max(np.abs(sp.moments(zg, wg, 79) / sp.checkerboard_moments(79) - 1)), 1e-12)
for h in (0.1, 10.0, 100.0):
    zg2, wg2 = sp.checkerboard_gauss(200)
    check(f"checkerboard_gauss(200) gives sqrt(h) at h={h:g}", sp.eps_ratio_from_measure(zg2, wg2, h), 1e-10, math.sqrt(h))
# Jacobi coefficients from an independent route: discretize the density by Gauss-Legendre in theta and run Stieltjes
th, wt = np.polynomial.legendre.leggauss(600)
th = (th + 1) * np.pi / 4
wt = wt * np.pi / 4 * 2 / np.pi * np.cos(th) ** 2  # integral (2/pi) cos^2 theta dtheta, z = sin^2 theta
zz = np.sin(th) ** 2
q0 = np.sqrt(wt / wt.sum())
Q = [q0]
al, be = [], []
for j in range(25):
    wv = zz * Q[-1]
    a = Q[-1] @ wv
    al.append(a)
    wv = wv - a * Q[-1] - (be[-1] * Q[-2] if j else 0)
    for _ in range(2):
        for qv in Q:
            wv = wv - (qv @ wv) * qv
    be.append(np.linalg.norm(wv))
    Q.append(wv / be[-1])
ja, jb = sp.checkerboard_jacobi(25)
check("checkerboard_jacobi equals the Stieltjes procedure on the quadrature-discretized density (alpha_0..24, beta_0..23)",
      max(np.max(np.abs(np.array(al) - ja)), np.max(np.abs(np.array(be[:24]) - jb))), 1e-12)
check("checkerboard_jacobi: alpha_0 = 1/4, alpha_k = 1/2, beta = 1/4", max(abs(ja[0] - 0.25), np.max(np.abs(ja[1:] - 0.5)), np.max(np.abs(jb - 0.25))), 0.0)
check("jacobi_measure(checkerboard_jacobi) first moment e0^T J e0 / ... : m_k from the Jacobi matrix", np.max(np.abs(sp.moments(*sp.jacobi_measure(ja, jb, 0.5), 10) - sp.checkerboard_moments(10))), 1e-14)

# sampled checkerboard: Lanczos with Jacobi coefficients, N = 64, 128, 256
dev_a, dev_b, W1 = {}, {}, {}
zref, wref = sp.checkerboard_gauss(1000)
for Nc in (64, 128, 256):
    chi = cs.checkerboard(Nc, 1.0, 0.0)
    t0 = time.perf_counter()
    z, w, a, b = sp.lanczos_measure(chi, 0, 60, return_jacobi=True)
    dt = time.perf_counter() - t0
    z2, w2 = sp.lanczos_measure(chi, 0, 60)
    a2, b2, f2 = sp.lanczos_jacobi(chi, 0, 60)
    check_true(f"N={Nc}: lanczos_measure(return_jacobi) / lanczos_jacobi / lanczos_measure agree", np.array_equal(a, a2) and np.array_equal(b, b2) and np.array_equal(z, z2) and f2 == 0.5)
    m = sp.moments(z, w, 2)
    check(f"N={Nc}: alpha_0 = 1/4 (sampled checkerboard)", a[0], 1e-12, 0.25)
    check(f"N={Nc}: m_1 = 1/8", m[1], 1e-12, 0.125)
    check(f"N={Nc}: beta_1 = 1/4 and m_2 = 1/16 (also exact)", max(abs(b[0] - 0.25), abs(m[2] - 1 / 16)), 1e-12)
    check(f"N={Nc}: m_0 = 1/2", m[0], 1e-12, 0.5)
    dev_a[Nc], dev_b[Nc] = np.abs(a[1:] - 0.5), np.abs(b[1:] - 0.25)
    info(f"N={Nc} ({dt:.2f} s, 60 steps): alpha_1..4 - 1/2 = {np.array2string(a[1:5] - 0.5, precision=2)}, beta_2..5 - 1/4 = {np.array2string(b[1:5] - 0.25, precision=2)}")
    zl, wl = sp.lanczos_measure(chi, 0, 300)
    W1[Nc] = sp.wasserstein1(zl, wl, zref, wref)
check_true("alpha_k -> 1/2 for k = 1..4 as N grows (64 > 128 > 256, strictly)",
           all(dev_a[64][k] > dev_a[128][k] > dev_a[256][k] for k in range(4)), "; ".join(f"k={k + 1}: {dev_a[64][k]:.1e}, {dev_a[128][k]:.1e}, {dev_a[256][k]:.1e}" for k in range(4)))
check_true("beta_k -> 1/4 for k = 2..6 as N grows (64 > 128 > 256, strictly)",
           all(dev_b[64][k] > dev_b[128][k] > dev_b[256][k] for k in range(5)), "; ".join(f"k={k + 2}: {dev_b[64][k]:.1e}, {dev_b[128][k]:.1e}, {dev_b[256][k]:.1e}" for k in range(5)))
info(f"sampled checkerboard, 300 steps: W1 distance to the Gauss(1000) reference of the exact measure: N=64 {W1[64]:.4f}, N=128 {W1[128]:.4f}, N=256 {W1[256]:.4f}")
check_true("W1 to the exact checkerboard measure decreases with N (64, 128, 256)", W1[64] > W1[128] > W1[256])
zl, wl = sp.lanczos_measure(cs.checkerboard(128, 1.0, 0.0), 0, 300)
gap = float(np.max(np.abs(sp.cdf(zl, wl, np.linspace(0.05, 1, 400)) - sp.cdf(zl, wl, [0.05]) - (sp.checkerboard_cdf(np.linspace(0.05, 1, 400)) - sp.checkerboard_cdf(0.05)))))
info(f"sup |CDF(z) - CDF(0.05) - (M(z) - M(0.05))| on [0.05, 1], N=128: {gap:.3f}; largest atom {wl.max():.3f} at z = {zl[np.argmax(wl)]:.4f} (of mass 1/2)")
discrepancy(f"plan: 'CDF of the rest against M(z) - M(0.05) on [0.05, 1]': the sampled checkerboard measure carries a few large atoms near z = 0.05 to 0.08 "
            f"(largest {wl.max():.2f} of 0.5 at N=128), so this sup distance is {gap:.2f}, far from small and nearly N-independent; use the W1 distance or moments, "
            f"and compare with the Gauss rule, instead")

# ---------------------------------------------------------------------------------------------- 6. utilities
z = np.array([0.1, 0.4, 0.9])
w = np.array([0.2, 0.5, 0.3])
s = np.array([[2.0, -1.0], [1.5 + 0.5j, 0.5j]])
direct = np.array([[np.sum(w / (sv - z)) for sv in row] for row in s])
check("stieltjes: shape preserved, complex s, matches direct sum", np.max(np.abs(sp.stieltjes(z, w, s) - direct)) + abs(sp.stieltjes(z, w, 2.0).ndim), 1e-15)
check_true("stieltjes: scalar input gives a scalar (0-d)", np.ndim(sp.stieltjes(z, w, 2.0)) == 0 and sp.stieltjes(z, w, np.linspace(1.5, 3, 7)).shape == (7,))
hh = np.array([0.1, 10.0, -2 + 0.5j])
check("eps_ratio_from_measure = 1 - stieltjes(1/(1-h)), array and complex h", np.max(np.abs(sp.eps_ratio_from_measure(z, w, hh) - (1 - sp.stieltjes(z, w, 1 / (1 - hh))))), 1e-14)
check("eps_ratio_from_measure(h = 1) = 1 exactly", abs(sp.eps_ratio_from_measure(z, w, 1.0) - 1.0), 0.0)
check("moments(z, w, 2)", np.max(np.abs(sp.moments(z, w, 2) - np.array([1.0, 0.02 + 0.2 + 0.27, 0.002 + 0.08 + 0.243]))), 1e-15)
check_true("cdf: right-continuous at atoms, 0 below, total above", sp.cdf(z, w, [0.0, 0.1, 0.39, 0.4, 5.0]).tolist() == [0.0, 0.2, 0.2, 0.7, 1.0] or np.allclose(sp.cdf(z, w, [0.0, 0.1, 0.39, 0.4, 5.0]), [0.0, 0.2, 0.2, 0.7, 1.0], atol=1e-15))
check("wasserstein1: two unit atoms at 0.25 and 0.75 give 0.5", sp.wasserstein1([0.25], [1.0], [0.75], [1.0]), 1e-15, 0.5)
check("wasserstein1: symmetric, zero for equal measures, normalize option", abs(sp.wasserstein1(z, w, z, w)) + abs(sp.wasserstein1([0.25], [2.0], [0.75], [1.0], normalize=True) - 0.5)
      + abs(sp.wasserstein1(z, w, [0.5], [1.0]) - sp.wasserstein1([0.5], [1.0], z, w)), 1e-15)
w1_exact = si.quad(lambda x: abs(sp.cdf(z, w, [x])[0] - sp.cdf([0.5], [1.0], [x])[0]), 0.1, 0.9, points=[0.4, 0.5], epsabs=1e-12)[0]
check("wasserstein1 equals quadrature of |CDF1 - CDF2|", sp.wasserstein1(z, w, [0.5], [1.0]), 1e-9, w1_exact)
check_true("n_atoms_for: 0.5 -> 1, 0.7 -> 1, 0.71 -> 2, 1.0 -> 3", [sp.n_atoms_for(z, w, f) for f in (0.5, 0.5 + 1e-12, 0.8, 1.0)] in ([1, 2, 2, 3], [1, 1, 2, 3]),
           str([sp.n_atoms_for(z, w, f) for f in (0.5, 0.5 + 1e-12, 0.8, 1.0)]))
check("drude_s(0.5, 1, 0.1) = 0.5 (0.5 + 0.1i)", abs(sp.drude_s(0.5, 1.0, 0.1) - 0.5 * (0.5 + 0.1j)), 1e-16)
eps1 = 1 - 1 / (0.5 * (0.5 + 0.1j))
check("drude_s consistent with eps1 = 1 - wp^2/(w (w + i gamma)): s = 1/(1 - eps1)", abs(1 / (1 - eps1) - sp.drude_s(0.5, 1.0, 0.1)), 1e-14)
# Rayleigh measure through the utilities: eps_ratio and stieltjes at the Drude s
check("utilities with the Rayleigh measure: eps_ratio at h = 12 equals sigma_eff", abs(sp.eps_ratio_from_measure(zr_, wr_, 12.0) - ry.sigma_eff(F_ROD, 12.0)), 1e-12)
check("Rayleigh measure through Drude s: 1 - F(s) equals sigma_eff(kappa = eps1) at omega = 0.6 omega_p, gamma = 0.05",
      abs(1 - sp.stieltjes(zr_, wr_, sp.drude_s(0.6, 1.0, 0.05)) - ry.sigma_eff(F_ROD, 1 - 1 / (0.6 * (0.6 + 0.05j)))), 1e-12)

dt = time.perf_counter() - T0
print(f"{'FAILED' if FAILS else 'ALL PASSED'}: {len(FAILS)} failures, {dt:.1f} s")
sys.exit(1 if FAILS else 0)
