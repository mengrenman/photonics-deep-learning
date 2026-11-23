"""Standalone test of rayleigh.py: prints one PASS/FAIL/DISCREPANCY line per item, exits nonzero on any FAIL."""
import os
import sys
import time
from math import gamma, pi
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bounds
import rayleigh as ry

T0 = time.perf_counter()
FAILS = []


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


F_ROD = pi * 0.3 ** 2  # f of the eps = 12 rods of spatial_dispersion

# 1. lattice sums against a brute-force sum (origin excluded), Richardson in the square cutoff
def brute(P, n):
    p = np.arange(-P, P + 1)
    z = p[:, None] + 1j * p[None, :]
    z[P, P] = 1e30  # excluded origin (contributes 1e-120)
    return float(np.sum(z ** (-float(n))).real)

S = ry.square_lattice_sums(60)
for n in (4, 8):
    s1, s2 = brute(200, n), brute(400, n)
    rich = (4.0 * s2 - s1) / 3.0
    check(f"S_{n} closed form vs brute force |p|,|q| <= 400 (Richardson)", S[n], 1e-6, rich)
check("S_4 = Gamma(1/4)^8/(960 pi^2)", S[4], 1e-13, gamma(0.25) ** 8 / (960 * pi ** 2))
check("S_8 = (3/7) S_4^2", S[8], 1e-13, 3.0 / 7.0 * S[4] ** 2)
check("S_6 = 0 and S_n = 0 unless n = 0 mod 4 (max |S_n|, n = 2 mod 4)", max(abs(S[n]) for n in S if n % 4 == 2), 0.0)
check("S_60 = 4 - 4 2^-30 (axis neighbors plus the four diagonal neighbors (1+i)^-60 = -2^-30)", S[60], 1e-12, 4.0 - 4.0 * 2.0 ** -30)

# 2. L = 1 is Maxwell Garnett exactly
for f, k in ((0.3, 10.0), (0.5, 0.1), (0.2, -2 + 0.5j)):
    mg = bounds.maxwell_garnett2d(k, 1.0, f)
    check(f"L = 1 equals Maxwell Garnett (f = {f}, kappa = {k})", abs(ry.sigma_eff(f, k, 1) - mg), 1e-14)

# 3. the spatial_dispersion series
b12 = 11.0 / 13.0
series = 1 + 2 * F_ROD / (1 / b12 - F_ROD - 0.305827 * b12 * F_ROD ** 4 - 0.013362 * b12 * F_ROD ** 8)
check("f = pi 0.3^2, kappa = 12 vs spatial_dispersion series (recomputed from its coefficients)", ry.sigma_eff(F_ROD, 12.0), 1e-7, 1.6301245119)
check("  the recomputed series equals 1.6301245119", series, 1e-9, 1.6301245119)

# 4. f = 0.5, kappa = 10, and convergence in L
check("f = 0.5, kappa = 10 gives 2.4154673802", ry.sigma_eff(0.5, 10.0), 1e-9, 2.4154673802)
for f, k in ((0.5, 10.0), (0.5, 100.0), (F_ROD, 12.0)):
    r = ry.sigma_eff(f, k, 20) / ry.sigma_eff(f, k, 10) - 1
    check(f"converged: |sigma(L=20)/sigma(L=10) - 1| at (f, kappa) = ({f:.4f}, {k:g})", abs(r), 1e-11)

# 5. dilute limit, including the first lattice correction 3 S_4^2/pi^4 beta f^4 (coefficient 0.305827)
f, k = 0.01, 10.0
beta = (k - 1) / (k + 1)
mg = bounds.maxwell_garnett2d(k, 1.0, f)
check("dilute f = 0.01, kappa = 10: |sigma/sigma_MG - 1|", abs(ry.sigma_eff(f, k) / mg - 1), 1e-9)
c4 = 3 * S[4] ** 2 / pi ** 4
corr = 2 * f * c4 * beta * f ** 4 / (1 / beta - f) ** 2
check("  3 S_4^2/pi^4 = 0.305827", c4, 1e-6, 0.305827)
check("  (sigma - sigma_MG) equals the first lattice correction (relative)", (ry.sigma_eff(f, k) - mg) / corr, 1e-4, 1.0)

# 6. symmetry and spectral measure
for f in (0.01, 0.1, F_ROD, 0.5, 0.7):
    W = ry.rayleigh_matrix(f, 8)
    assert np.allclose(W, W.T, rtol=1e-13, atol=0)
check("W real symmetric (max asymmetry, f = 0.7, L = 8)", np.abs(ry.rayleigh_matrix(0.7, 8) - ry.rayleigh_matrix(0.7, 8).T).max(), 1e-14)
check("W_11 = f exactly", ry.rayleigh_matrix(0.37, 5)[0, 0], 1e-15, 0.37)
worst0 = worst1 = 0.0
for f in (0.01, 0.1, F_ROD, 0.5, 0.7):
    for L in (1, 2, 5, 20):
        z, w = ry.rayleigh_measure(f, L)
        worst0 = max(worst0, abs(w.sum() - f))
        worst1 = max(worst1, abs(np.sum(w * z) - f * (1 - f) / 2))
check("sum rule m0 = f, L in {1, 2, 5, 20}, five f (worst)", worst0, 1e-14)
check("sum rule m1 = f(1-f)/2, L in {1, 2, 5, 20}, five f (worst)", worst1, 1e-14)
mu_max = max(np.abs(1 - 2 * ry.rayleigh_measure(f, 20)[0]).max() for f in (0.01, 0.1, F_ROD, 0.5, 0.7))
check_true("atoms strictly inside (0, 1): max |mu_k| over five f", mu_max < 1, f"max |mu| = {mu_max:.4f}")
check_true("weights positive", all((ry.rayleigh_measure(f, 20)[1] > 0).all() for f in (0.1, 0.5, 0.7)))
check_true("atoms ascending", all(np.all(np.diff(ry.rayleigh_measure(f, 20)[0]) >= 0) for f in (0.1, 0.5)))

# measure reproduces sigma_eff at real and complex contrasts
worst = 0.0
for f in (0.1, F_ROD, 0.5, 0.7):
    z, w = ry.rayleigh_measure(f, 20)
    for kap in (0.1, 10.0, 100.0, -2 + 0.5j):
        s = 1.0 / (1.0 - kap)
        sig = 1.0 - np.sum(w / (s - z))
        worst = max(worst, abs(sig / ry.sigma_eff(f, kap, 20) - 1))
check("measure reproduces sigma_eff (relative, four contrasts incl. -2+0.5i, four f)", worst, 1e-12)
vals = ry.sigma_eff(0.5, np.array([0.1, 10.0, 100.0, -2 + 0.5j]))
check("sigma_eff vectorized equals scalar calls", max(abs(vals[i] - ry.sigma_eff(0.5, k)) for i, k in enumerate([0.1, 10.0, 100.0, -2 + 0.5j])), 1e-14)
check("sigma_eff(kappa = 1) = 1", abs(ry.sigma_eff(0.5, 1.0) - 1.0), 0.0)
# truncation convergence of the measure
for f in (0.1, F_ROD, 0.5):
    z20, w20 = ry.rayleigh_measure(f, 20)
    z40, w40 = ry.rayleigh_measure(f, 40)
    big = w20 > 1e-8
    # match each large atom of L = 20 with the nearest atom of L = 40
    dev = max(np.abs(z40 - zk).min() for zk in z20[big])
    wdev = max(abs(w40[np.argmin(np.abs(z40 - zk))] - wk) for zk, wk in zip(z20[big], w20[big]))
    check(f"truncation converged: atoms with weight > 1e-8 at L = 20 vs 40 (f = {f:.4f}): position", dev, 1e-10)
    check(f"  weight", wdev, 1e-10)

# dilute measure
z, w = ry.rayleigh_measure(0.01, 20)
j = np.argmax(w)
W3 = ry.rayleigh_matrix(0.01, 3)
pred_shift = W3[0, 1] ** 2 / (W3[0, 0] - W3[1, 1]) / 2  # second-order perturbation of mu_1 = W_11 (z = (1 - mu)/2)
pred_def = (W3[0, 1] / (W3[0, 0] - W3[1, 1])) ** 2
check("dilute f = 0.01: dominant atom shift from (1 - f)/2 equals second-order perturbation W_13^2/(2 (W_11 - W_33))",
      (0.495 - z[j]) / pred_shift, 1e-2, 1.0)
check("dilute f = 0.01: dominant weight deficit 1 - w/f equals (W_13/(W_11 - W_33))^2", (1 - w[j] / 0.01) / pred_def, 1e-2, 1.0)
check("dilute f = 0.01: dominant atom within 3e-7 of (1 - f)/2", z[j], 3e-7, 0.495)
check_true("dilute f = 0.01: dominant weight >= (1 - 1e-4) f", w[j] >= (1 - 1e-4) * 0.01, f"1 - w/f = {1 - w[j] / 0.01:.2e}")
print("DISCREPANCY  plan 02 section 2 dilute check (f = 0.01: atom within 1e-7 of (1-f)/2 and weight >= (1-1e-6) f): expected shift < 1e-7 and deficit < 1e-6, got "
      f"{0.495 - z[j]:.2e} and {1 - w[j] / 0.01:.2e}; diagnosis: the dipole-octupole coupling W_13 ~ f^2 shifts the atom by W_13^2/(2 f) ~ f^3 and takes weight ~ f^2 (both formulas asserted above); "
      "the plan's 1e-7 and 1e-6 would need f <~ 2e-3. Suggested: 3e-7 and 1e-4 at f = 0.01.")

# 7. dominant atom of the rods' f; the plan's scratch numbers belong to f = 0.2827, not to f = 0.282743
z, w = ry.rayleigh_measure(0.2827, 20)
j = np.argmax(w)
check("dominant atom at f = 0.2827 reproduces the scratch position 0.355273", z[j], 1e-5, 0.355273)
check("dominant atom at f = 0.2827 reproduces the scratch weight fraction 0.97717", w[j] / 0.2827, 1e-5, 0.97717)
z, w = ry.rayleigh_measure(F_ROD, 20)
j = np.argmax(w)
# independent location of the pole: root of det(I/beta - W) in beta near the dominant atom (plain root finding on det)
W = ry.rayleigh_matrix(F_ROD, 20)
zj = z[j]
mu_target = 1 - 2 * zj
lo, hi = mu_target - 1e-3, mu_target + 1e-3
# sign change of det across an eigenvalue (the determinant of a symmetric matrix changes sign at each simple eigenvalue)
d_lo = np.linalg.det(lo * np.eye(20) - W)
for _ in range(80):
    mid = 0.5 * (lo + hi)
    dm = np.linalg.det(mid * np.eye(20) - W)
    if np.sign(dm) == np.sign(d_lo):
        lo = mid
    else:
        hi = mid
check("f = pi 0.3^2: dominant atom position, eigenvalue root by determinant bisection", (1 - 0.5 * (lo + hi)) / 2, 1e-12, zj)
print("DISCREPANCY  rayleigh/spectral plan 02 section 2: 'dominant atom z = 0.355273, weight fraction 0.97717 at f = 0.282743' -- expected 0.355273 / 0.97717 "
      f"at f = 0.282743, got {zj:.6f} / {w[j] / F_ROD:.5f}; the scratch prototype evaluated f = 0.2827 (rounded). Correct values at f = pi 0.3^2: 0.355249 / 0.977158 (asserted above at 1e-5 only for f = 0.2827).")
check("f = pi 0.3^2: dominant atom (asserted correct value, independent determinant root above)", zj, 1e-5, 0.355249)
check("f = pi 0.3^2: dominant weight fraction", w[j] / F_ROD, 1e-5, 0.977158)
z, w = ry.rayleigh_measure(0.5, 20)
j = np.argmax(w)
check("f = 0.5: dominant atom z = 0.231774", z[j], 1e-5, 0.231774)
check("f = 0.5: weight fraction 93.4%", w[j] / 0.5, 1e-3, 0.934)
mu = np.linalg.eigvalsh(ry.rayleigh_matrix(0.5, 20))
check("f = 0.5: spectrum of W [-0.0954, 0.5365]", max(abs(mu[0] + 0.0954), abs(mu[-1] - 0.5365)), 1e-4)

# 8. cross-checks against other independent evaluations
# a pixel-free check with the exact Fourier-Galerkin bracket (see test_cellsolver for the full tests)
import cellsolver as cs
Lb, Ub, _ = cs.fg_bracket("disk", 24, R=np.sqrt(0.5 / pi), s_in=10.0, s_out=1.0)
check_true("FG bracket (K = 24) contains the Rayleigh value at (0.5, 10)", Lb <= ry.sigma_eff(0.5, 10.0) <= Ub, f"[{Lb:.6f}, {Ub:.6f}]")
# touching limit: strong divergence as f -> pi/4 for a perfect conductor, finite for moderate contrast; check monotone in f
vals = [ry.sigma_eff(f, 100.0) for f in (0.1, 0.3, 0.5, 0.7)]
check_true("sigma_eff increasing in f for kappa = 100", all(np.diff(vals) > 0), str(np.round(vals, 4)))
hs_lo, hs_up = bounds.hs2d(1.0, 10.0, 0.5)
s = ry.sigma_eff(0.5, 10.0)
check_true("Rayleigh (disks of phase 1 = 10 in 1, f = 0.5) lies inside HS", hs_lo <= s <= hs_up, f"{hs_lo:.4f} <= {s:.4f} <= {hs_up:.4f}")
check("MG (host 1, inclusion 10) equals the HS lower bound", bounds.maxwell_garnett2d(10.0, 1.0, 0.5), 1e-14, hs_lo)

dt = time.perf_counter() - T0
print(f"{'FAILED' if FAILS else 'ALL PASSED'}: {len(FAILS)} failures, {dt:.1f} s")
sys.exit(1 if FAILS else 0)
