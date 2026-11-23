"""Standalone test of bounds.py: prints one PASS/FAIL/DISCREPANCY line per item, exits nonzero on any FAIL."""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import bounds as bd
import cellsolver as cs
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


# ---------------------------------------------------------------- closed forms
lo, hi = bd.hs2d(1.0, 10.0, 0.5)
check("HS(1, 10, 0.5) lower = 2.384615...", lo, 1e-6, 2.3846153846)
check("HS(1, 10, 0.5) upper = 4.193548...", hi, 1e-6, 4.1935483871)
lo, hi = bd.hs2d(1.0, 10.0, 0.7)
check("HS(1, 10, 0.7 weak) lower", lo, 1e-5, 1.65060)
check("HS(1, 10, 0.7 weak) upper", hi, 1e-5, 2.71676)
l2, h2 = bd.hs2d(10.0, 1.0, 0.3)
check("HS is symmetric in the phase labels (10 with fraction 0.3 = 1 with 0.7)", abs(l2 - lo) + abs(h2 - hi), 1e-14)
vl, vh = bd.hs2d(np.array([1.0, 2.0]), 10.0, np.array([0.5, 0.2]))
check("HS vectorized", abs(vl[0] - 2.3846153846153846) + abs(vh[1] - bd.hs2d(2.0, 10.0, 0.2)[1]), 1e-14)
h_, a_ = bd.wiener(1.0, 10.0, 0.7)
check("Wiener harmonic", h_, 1e-14, 1 / (0.7 + 0.03))
check("Wiener arithmetic", a_, 1e-14, 0.7 + 3.0)
margins = [min(bd.hs2d(1, 10, f)[0] - bd.wiener(1, 10, f)[0], bd.wiener(1, 10, f)[1] - bd.hs2d(1, 10, f)[1]) for f in np.linspace(0.05, 0.95, 21)]
check_true("HS inside the Wiener box for 21 fractions", min(margins) > 0, f"min margin {min(margins):.3e}")
check("Bruggeman at f = 1/2 is sqrt(s1 s2) (Dykhne)", bd.bruggeman2d(1.0, 10.0, 0.5), 1e-14, np.sqrt(10.0))
for f in (0.2, 0.7):
    sb = bd.bruggeman2d(1.0, 10.0, f)
    res = f * (1 - sb) / (1 + sb) + (1 - f) * (10 - sb) / (10 + sb)
    check(f"Bruggeman solves its defining equation (f = {f})", res, 1e-14)
check("Bruggeman symmetric in the phase labels", abs(bd.bruggeman2d(1.0, 10.0, 0.3) - bd.bruggeman2d(10.0, 1.0, 0.7)), 1e-14)
check_true("Bruggeman between the HS bounds (f = 0.3, 0.7)", all(bd.hs2d(1, 10, f)[0] <= bd.bruggeman2d(1, 10, f) <= bd.hs2d(1, 10, f)[1] for f in (0.3, 0.7)))
check("MG = HS lower when the matrix (host) is the weaker phase (inclusion 10 in 1, f = 0.5)",
      bd.maxwell_garnett2d(10.0, 1.0, 0.5), 1e-14, bd.hs2d(1.0, 10.0, 0.5)[0])
check("MG = HS lower (weak matrix, f_inc = 0.3)", bd.maxwell_garnett2d(10.0, 1.0, 0.3), 1e-14, bd.hs2d(1.0, 10.0, 0.7)[0])
check("MG = HS upper when the matrix is the stronger phase (inclusion 1 in 10, f = 0.7)",
      bd.maxwell_garnett2d(1.0, 10.0, 0.7), 1e-14, bd.hs2d(1.0, 10.0, 0.7)[1])
check("MG vs Rayleigh at the dilute limit (f = 0.01, kappa = 10)", bd.maxwell_garnett2d(10.0, 1.0, 0.01), 1e-9, ry.sigma_eff(0.01, 10.0))

# ---------------------------------------------------------------- sequential laminates
s1, s2, f1 = 1.0, 10.0, 0.7
lh, la = bd.wiener(s1, s2, f1)
worst = 0.0
for matrix in ("weak", "strong"):
    A = bd.laminate_seq(s1, s2, 1 - f1, [0.0], [1.0]) if matrix == "weak" else bd.laminate_seq(s2, s1, f1, [0.0], [1.0])
    worst = max(worst, abs(A[0, 0] - lh), abs(A[1, 1] - la), abs(A[0, 1]))
check("rank-1 laminate (normal x) = Wiener means: harmonic along x, arithmetic along y (both matrix choices)", worst, 1e-14)
A = bd.laminate_seq(s1, s2, 1 - f1, [np.pi / 2], [1.0])
check("rank-1 laminate normal y: swapped", abs(A[0, 0] - la) + abs(A[1, 1] - lh), 1e-14)
vec = bd.laminate_seq(s1, s2, 1 - f1, np.array([[1.0, 0.0], [0.0, 2.0]]), [0.5, 0.5])
ang = bd.laminate_seq(s1, s2, 1 - f1, [0.0, np.pi / 2], [0.5, 0.5])
check("dirs as vectors equals dirs as angles", np.abs(vec - ang).max(), 1e-14)
hsl, hsu = bd.hs2d(s1, s2, f1)
Aw = bd.laminate_seq(s1, s2, 1 - f1, [0.0, np.pi / 2], [0.5, 0.5])
As = bd.laminate_seq(s2, s1, f1, [0.0, np.pi / 2], [0.5, 0.5])
check("isotropic rank-2, weak matrix = HS lower (theta1 = 0.7)", np.abs(Aw - hsl * np.eye(2)).max(), 1e-14)
check("isotropic rank-2, strong matrix = HS upper (theta1 = 0.7)", np.abs(As - hsu * np.eye(2)).max(), 1e-14)
Aw5 = bd.laminate_seq(1.0, 10.0, 0.5, [0.0, np.pi / 2], [0.5, 0.5])
check("isotropic rank-2, theta1 = 0.5: HS lower 2.384615", Aw5[0, 0], 1e-14, 2.3846153846153846)
# rotation covariance: all normals rotated by 30 degrees gives the rotated tensor
th = np.pi / 6
Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Ar = bd.laminate_seq(s1, s2, 1 - f1, [th, th + np.pi / 2], [0.3, 0.7])
Ax = bd.laminate_seq(s1, s2, 1 - f1, [0.0, np.pi / 2], [0.3, 0.7])
check("laminate_seq rotation covariance", np.abs(Ar - Rm @ Ax @ Rm.T).max(), 1e-13)

m = np.linspace(0, 1, 101)
cw = bd.rank2_curve(s1, s2, f1, m, "weak")
cst = bd.rank2_curve(s1, s2, f1, m, "strong")
low_w, up_w = bd.trace_slacks(cw[:, 0], cw[:, 1], s1, s2, f1)
low_s, up_s = bd.trace_slacks(cst[:, 0], cst[:, 1], s1, s2, f1)
check("weak-matrix rank-2 family on the LOWER trace curve (101 values of m): max |slack|", np.abs(low_w).max(), 1e-12)
check("strong-matrix rank-2 family on the UPPER trace curve (101 values of m): max |slack|", np.abs(up_s).max(), 1e-12)
check_true("each family satisfies the other trace inequality (slack >= -1e-12)", up_w.min() >= -1e-12 and low_s.min() >= -1e-12,
           f"min {up_w.min():.2e}, {low_s.min():.2e}")
check_true("rank-2 eigenvalues inside the Wiener box", cw.min() >= lh - 1e-12 and cst.max() <= la + 1e-12)
# theta1 = 0.5 also, and another contrast
for (a_, b_, f_) in ((1.0, 10.0, 0.5), (1.0, 3.0, 0.3), (2.0, 50.0, 0.8)):
    cw_ = bd.rank2_curve(a_, b_, f_, m, "weak")
    cs_ = bd.rank2_curve(a_, b_, f_, m, "strong")
    sl = np.abs(bd.trace_slacks(cw_[:, 0], cw_[:, 1], a_, b_, f_)[0]).max()
    su = np.abs(bd.trace_slacks(cs_[:, 0], cs_[:, 1], a_, b_, f_)[1]).max()
    check(f"trace curves for (s1, s2, f1) = ({a_}, {b_}, {f_}): max |slack|", max(sl, su), 1e-12)
# the anisotropic boundary point of notebook 03 section 5
p = bd.rank2_curve(s1, s2, f1, [0.25], "strong")[0]
check("strong-matrix rank-2 at m = 0.25 (weight 0.25 on normal x): lambda_x = 3.244 (plan scratch)", p[0], 1e-3, 3.244)
check("  lambda_y = 2.100 (plan scratch)", p[1], 1e-3, 2.100)
eps = 1e-6
pp, pm = bd.rank2_curve(s1, s2, f1, [0.25 + eps], "strong")[0], bd.rank2_curve(s1, s2, f1, [0.25 - eps], "strong")[0]
slope = (pp[1] - pm[1]) / (pp[0] - pm[0])
w_star = -1.0 / slope
print(f"INFO  anisotropic boundary point: d lambda_y/d lambda_x = {slope:.5f}, weight w = -1/slope = {w_star:.5f} (plan: about 0.68)")
l1, l2 = p
w_analytic = ((s2 - l1) / (s2 - l2)) ** 2  # the upper trace curve: 1/(s2-l1) + 1/(s2-l2) = const, so d l2/d l1 = -((s2-l2)/(s2-l1))^2
check("anisotropic objective weight: finite-difference w = -1/(d lambda_y/d lambda_x) equals the analytic ((s2 - l1)/(s2 - l2))^2", w_star, 1e-6, w_analytic)
print("DISCREPANCY  plan 03 section 5: 'w = -1/(d lambda2/d lambda1) at m* (about 0.68)' -- expected about 0.68, got "
      f"{w_star:.4f} (lambda = ({l1:.4f}, {l2:.4f}), slope {slope:.4f}); diagnosis: the closed form w = ((s2 - l1)/(s2 - l2))^2 gives {w_analytic:.4f} and the "
      "maximizer test below shows that w = 0.7314 is stationary at m = 0.25 while w = 0.68 puts the maximum elsewhere.")
m_grid = np.linspace(0, 1, 2001)
cu = bd.rank2_curve(s1, s2, f1, m_grid, "strong")
print(f"INFO  argmax_m of lam_x + w lam_y on the upper family: w = {w_star:.4f} -> m = {m_grid[np.argmax(cu[:, 0] + w_star * cu[:, 1])]:.4f}; "
      f"w = 0.68 -> m = {m_grid[np.argmax(cu[:, 0] + 0.68 * cu[:, 1])]:.4f}; w = 0.5 -> m = {m_grid[np.argmax(cu[:, 0] + 0.5 * cu[:, 1])]:.4f}")
check("maximizer of lam_x + w lam_y at w = -1/slope is m = 0.25", m_grid[np.argmax(cu[:, 0] + w_star * cu[:, 1])], 1e-3, 0.25)
# with this w the maximum of lam_x + w lam_y over the upper curve is at m*: stationary and a maximum
g = lambda mm: (lambda q: q[0] + w_star * q[1])(bd.rank2_curve(s1, s2, f1, [mm], "strong")[0])
check_true("lam_x + w lam_y over the strong-matrix family is maximal at m = 0.25 (grid of 201 m)",
           abs(np.argmax([g(x) for x in np.linspace(0, 1, 201)]) - 50) <= 1, f"argmax m = {np.linspace(0,1,201)[np.argmax([g(x) for x in np.linspace(0,1,201)])]:.3f}")
check_true("  with w = 0.5 the maximum sits at a laminate corner (m = 0 or 1)",
           np.argmax([(lambda q: q[0] + 0.5 * q[1])(bd.rank2_curve(s1, s2, f1, [x], "strong")[0]) for x in np.linspace(0, 1, 201)]) in (0, 200))

# ---------------------------------------------------------------- certified violation
check_true("no violation: HS isotropic points", not bd.certified_violation(hsl * np.eye(2), hsl * np.eye(2), s1, s2, f1)
           and not bd.certified_violation(hsu * np.eye(2), hsu * np.eye(2), s1, s2, f1))
check_true("no violation: every rank-2 curve point (202 points)",
           not any(bd.certified_violation(np.diag(c), np.diag(c), s1, s2, f1) for c in np.vstack([cw, cst])))
check_true("violation: isotropic 1.5 < HS lower 1.65 (inside the Wiener box)", bd.certified_violation(1.5 * np.eye(2), 1.5 * np.eye(2), s1, s2, f1))
check_true("violation: isotropic 2.9 > HS upper 2.717 (inside the Wiener box)", bd.certified_violation(2.9 * np.eye(2), 2.9 * np.eye(2), s1, s2, f1))
check_true("violation: outside the Wiener box (below the harmonic mean)", bd.certified_violation(np.diag([1.2, 3.0]), np.diag([1.2, 3.0]), s1, s2, f1))
check_true("violation: outside the Wiener box (above the arithmetic mean)", bd.certified_violation(np.diag([2.0, 3.9]), np.diag([2.0, 3.9]), s1, s2, f1))
check_true("not certified: wide bracket around HS lower ([1.5, 1.9] isotropic)", not bd.certified_violation(1.5 * np.eye(2), 1.9 * np.eye(2), s1, s2, f1))
check_true("not certified: bracket touching the region from outside ([2.5, 2.9] isotropic)", not bd.certified_violation(2.5 * np.eye(2), 2.9 * np.eye(2), s1, s2, f1))
check_true("certified: bracket [1.50, 1.60] entirely below HS lower", bd.certified_violation(1.5 * np.eye(2), 1.6 * np.eye(2), s1, s2, f1))
# consistency with the slacks for exact tensors: violation iff some slack is negative (or outside the box)
rng = np.random.default_rng(3)
agree = True
for _ in range(400):
    q = rng.uniform(lh * 0.9, la * 1.05, 2)
    th = rng.uniform(0, np.pi)
    Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    T = Rr @ np.diag(q) @ Rr.T
    qs = np.sort(np.linalg.eigvalsh(T))
    box = qs[0] < lh - 1e-12 or qs[1] > la + 1e-12
    sl = bd.trace_slacks(qs[0], qs[1], s1, s2, f1) if not box else (0.0, 0.0)
    expect = box or sl[0] < -1e-12 or sl[1] < -1e-12
    agree &= (bd.certified_violation(T, T, s1, s2, f1) == expect)
check_true("certified_violation(T, T) agrees with the slacks on 400 random exact tensors", agree)
# soundness against actual composites: Q1 brackets of random pixel fields contain no certified violation
viol = 0
for seed in range(12):
    a = cs.gaussian_threshold(24, 2 + seed % 3, 0.7, seed, (1.0, 10.0))
    L, U = cs.q1_bracket(a)
    viol += bd.certified_violation(L, U, 1.0, 10.0, 0.7)
check("no certified violation for 12 thresholded Gaussian cells (N = 24)", viol, 0)

# ---------------------------------------------------------------- Bergman-Milton support functions
phis = np.linspace(0, 2 * np.pi, 720, endpoint=False)
contrasts = (0.2 + 0.5j, -2 + 1j, -0.5 + 0.2j)
f = 0.5
zz = np.linspace(0, 1, 400001)
for hh in contrasts[:2]:
    s = 1.0 / (1.0 - hh)
    hf = bd.bm_support_f(s, f, phis)
    sub = phis[::12]
    brute = f * np.max(np.real(np.exp(-1j * sub)[:, None] / (s - zz[None, ::2])), axis=1)  # 200001-point scan
    dist = np.min(np.abs(s - zz[::100]))
    print(f"INFO  contrast {hh}: s = {s:.4f}, dist(s, [0,1]) = {dist:.4f}")
    check_true(f"bm_support_f >= a 200001-point scan ({hh})", np.all(hf[::12] >= brute - 1e-14), f"min diff {np.min(hf[::12] - brute):.2e}")
    check(f"bm_support_f vs the 200001-point scan (max |diff|, {hh})", np.abs(hf[::12] - brute).max(), 1e-8)

# exact values inside the regions
Dz = 1.0 / 4000
for hh in contrasts:
    s = 1.0 / (1.0 - hh)
    dist = np.min(np.abs(s - np.linspace(0, 1, 2001)))
    hf = bd.bm_support_f(s, f, phis)
    hi_, w = bd.bm_support_iso(s, f, phis, method="dual")
    exact = {"checkerboard sqrt(h)": 1.0 - np.sqrt(hh), "disk lattice f = 0.5 (Rayleigh)": 1.0 - ry.sigma_eff(0.5, hh), "MG": f / (s - 0.25)}
    grid_tol = f * Dz / dist ** 2
    print(f"INFO  contrast {hh}: dist(s,[0,1]) = {dist:.4f}, iso grid tolerance f Dz/dist^2 = {grid_tol:.2e}")
    for nm, Fx in exact.items():
        proj = np.real(np.exp(-1j * phis) * Fx)
        mf = np.max(proj - hf)
        mi = np.max(proj - hi_)
        check_true(f"{nm} inside the f-only region (h = {hh}): max(Re(e^-i phi F) - support) <= 1e-9", mf <= 1e-9, f"{mf:.3e}")
        check_true(f"{nm} inside the isotropic region (h = {hh}): <= grid tolerance", mi <= grid_tol + 1e-9, f"{mi:.3e} (tol {grid_tol:.2e})")
    check_true(f"isotropic support <= f-only support (h = {hh})", np.max(hi_ - hf) <= 1e-12, f"{np.max(hi_ - hf):.2e}")

# LP versus dual, extremal weights
s = 1.0 / (1.0 - (-2 + 1j))
hd, wd = bd.bm_support_iso(s, f, phis[::12], method="dual")
hl, wl = bd.bm_support_iso(s, f, phis[::12], method="lp")
check("LP (highs-ds) vs dual method: max |h_lp - h_dual| over 60 directions (HiGHS optimality tolerance 1e-7)", np.abs(hl - hd).max(), 1e-7)
check_true("dual method is at least as good as the LP (the dual value is a feasible primal)", np.all(hd >= hl - 1e-9), f"{(hd - hl).min():.2e}")
nz_lp = (wl > 1e-10).sum(axis=1)
check_true("LP extremal solutions have at most 2 nonzero weights (60 directions)", nz_lp.max() <= 2, f"max {nz_lp.max()}")
zgrid = np.linspace(0, 1, 4001)
okloc = True
for i in range(len(wl)):
    idx = np.nonzero(wl[i] > 1e-10)[0]
    zs = zgrid[idx]
    okloc &= bool(len(zs) == 1 or (zs.min() <= 0.25 + 1e-12 and zs.max() >= 0.25 - 1e-12))
    okloc &= abs(wl[i].sum() - f) < 1e-9 and abs((wl[i] * zgrid).sum() - f * (1 - f) / 2) < 1e-9
check_true("LP solutions straddle the mean z = (1-f)/2 and satisfy m0 = f, m1 = f(1-f)/2", okloc)
atoms_on_edge = 0
for i in range(len(wd)):
    idx = np.nonzero(wd[i] > 1e-10)[0]
    atoms_on_edge += int(any(abs(zgrid[j]) < 1e-12 or abs(zgrid[j] - 1) < 1e-12 for j in idx) or len(idx) == 1)
check_true("extremal atoms located at {z, 1}, {0, z} or the single MG atom (dual method; 60 directions)", atoms_on_edge == len(wd), f"{atoms_on_edge}/60")
# MG is the corner of the isotropic lens
for hh in contrasts:
    s = 1.0 / (1.0 - hh)
    hi_, _ = bd.bm_support_iso(s, f, phis, method="dual")
    d = np.max(np.real(np.exp(-1j * phis) * (f / (s - 0.25))) - hi_)
    check_true(f"MG touches the isotropic region at its corner (h = {hh}): max(Re(e^-i phi F_MG) - support_iso) >= -1e-8", d >= -1e-8 and d <= 1e-9, f"{d:.2e}")

dt = time.perf_counter() - T0
print(f"{'FAILED' if FAILS else 'ALL PASSED'}: {len(FAILS)} failures, {dt:.1f} s")
sys.exit(1 if FAILS else 0)
