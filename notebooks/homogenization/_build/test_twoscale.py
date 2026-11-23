"""Standalone test of twoscale.py: prints one PASS/FAIL/DISCREPANCY line per item, exits nonzero on any FAIL."""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp
from scipy.integrate import simpson

import cellsolver as cs
import twoscale as ts

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


def check_range(name, value, lo, hi):
    check_true(name, lo <= value <= hi, f"value {value:.6g} in [{lo}, {hi}]")


def slope(xs, ys):
    return -np.polyfit(np.log(xs), np.log(ys), 1)[0]


M_PIX = 16
CELL = cs.disk_pixels(M_PIX, 0.3, 10.0, 1.0)
CHI, SIG = ts.cell_correctors(CELL)
SS = SIG[0, 0]
GRAM = ts.corrector_gram(CHI)
BOX = ts.interior_box()

# ---------------------------------------------------------------------------------------------- 1. cell problem
U_cs, chi_cs = cs.q1_effective(CELL, return_fields=True)
check("cell_correctors sigma equals cellsolver.q1_effective (max abs)", np.abs(SIG - U_cs).max(), 1e-12)
check("cell_correctors chi equals cellsolver chi (max abs)", np.abs(CHI - chi_cs).max(), 1e-12)
check("isotropy |s_xx - s_yy| (C4 cell)", abs(SIG[0, 0] - SIG[1, 1]), 1e-12)
check("isotropy |s_xy|", abs(SIG[0, 1]), 1e-12)
check("sigma*_h reproduces the design scratch 1.688269", SS, 5e-7, 1.688269)
check("chi has zero mean", np.abs(CHI.mean(axis=(1, 2))).max(), 1e-14)
check("corrector_gram M_11 reproduces scratch 1.50455", GRAM[0, 0], 5e-6, 1.50455)
check("corrector_gram isotropic and symmetric (C4)", max(abs(GRAM[0, 0] - GRAM[1, 1]), abs(GRAM[0, 1]), abs(GRAM[1, 0])), 1e-12)
Ed = cs.q1_energy_density(np.ones((M_PIX, M_PIX)), CHI)  # exact element energies with unit coefficient
check("corrector_gram diagonal equals the exact unit-coefficient energy of cellsolver", max(abs(Ed[0].sum() - GRAM[0, 0]), abs(Ed[1].sum() - GRAM[1, 1])), 1e-13)
# M >= identity-like: <|e + grad chi|^2> >= |<e + grad chi>|^2 = 1 (zero-mean correctors), and M positive definite

# ---------------------------------------------------------------------------------------------- tiling and shifting


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


a_t = ts.periodic_medium(CELL, 3, 0)
check_true("periodic_medium tiles exactly", np.array_equal(a_t[:M_PIX, :M_PIX], CELL) and np.array_equal(a_t[M_PIX:2 * M_PIX, :M_PIX], CELL), "exact copies")
a_s = ts.periodic_medium(CELL, 3, 4)
check_true("periodic_medium shift s: pixel i carries cell pixel (i - s) mod m", np.array_equal(a_s[4:4 + M_PIX, 4:4 + M_PIX], CELL), "exact")
a_sp = ts.periodic_medium(CELL, 3, (4, 0))
check_true("periodic_medium accepts a shift pair", np.array_equal(a_sp[4:4 + M_PIX, :M_PIX], CELL), "exact")
nod = ts.periodic_nodal(CHI[0], 3, 4)
check_true("periodic_nodal: shape (n+1, n+1), shifted copy of the periodic field",
           nod.shape == (49, 49) and np.array_equal(nod[4:4 + M_PIX, 4:4 + M_PIX], CHI[0]) and np.array_equal(nod[0], nod[M_PIX]) and np.array_equal(nod[:, 0], nod[:, M_PIX]),
           "exact")

# ---------------------------------------------------------------------------------------------- torsion
xs = np.linspace(0, 1, 129)
u301 = ts.torsion(xs, xs, 301)[0]
u601 = ts.torsion(xs, xs, 601)[0]
u2401 = ts.torsion(xs, xs, 2401)[0]
print(f"DISCREPANCY  plan section 1: max nodal |u(301 terms) - u(601 terms)| = {np.abs(u301 - u601).max():.2e}, not 1e-10 "
      f"(the series error is about 3/nterms^3: {np.abs(u301 - u2401).max():.2e} at 301, {np.abs(u601 - u2401).max():.2e} at 601); asserted below with the measured behavior")
check("torsion series: max nodal |301 - 601| below 2e-7 (measured 1.3e-7)", np.abs(u301 - u601).max(), 2e-7)
check("torsion series: max nodal |601 - 2401| below 2e-8 (measured 1e-8)", np.abs(u601 - u2401).max(), 2e-8)
check_true("torsion series error falls at least like nterms^-2.5 from 301 to 601", np.abs(u301 - u2401).max() / np.abs(u601 - u2401).max() > 2 ** 2.5, f"ratio {np.abs(u301 - u2401).max() / np.abs(u601 - u2401).max():.1f}")
check("torsion center value (published 0.0736713533, 601 terms)", u601[64, 64], 2e-9, 0.07367135328)
p_ = np.arange(1, 4001, 2)
P_, Q_ = np.meshgrid(p_, p_, indexing="ij")
int_u = np.sum(64.0 / (np.pi ** 6 * P_ ** 2 * Q_ ** 2 * (P_ ** 2 + Q_ ** 2)))
x1 = np.linspace(0, 1, 1001)
u_fine = ts.torsion(x1, x1, 601)[0]
check("torsion integral equals its coefficient sum 64/(pi^6 p^2 q^2 (p^2+q^2)) (Simpson, 1001 pts)", simpson(simpson(u_fine, x=x1, axis=1), x=x1), 1e-8, int_u)
xx = (0.31, 0.62)
d_ = 1e-6
fd_x = (ts.torsion([xx[0] + d_], [xx[1]], 301)[0] - ts.torsion([xx[0] - d_], [xx[1]], 301)[0])[0, 0] / (2 * d_)
fd_y = (ts.torsion([xx[0]], [xx[1] + d_], 301)[0] - ts.torsion([xx[0]], [xx[1] - d_], 301)[0])[0, 0] / (2 * d_)
_, tx, ty = ts.torsion([xx[0]], [xx[1]], 301)
check("torsion gradient equals the central difference of u (x and y)", max(abs(fd_x - tx[0, 0]), abs(fd_y - ty[0, 0])), 1e-8)
check("torsion zero on the boundary", max(abs(u301[0]).max(), abs(u301[-1]).max(), abs(u301[:, 0]).max(), abs(u301[:, -1]).max()), 1e-15)
check("torsion symmetric under x <-> y", np.abs(u301 - u301.T).max(), 1e-15)

# ---------------------------------------------------------------------------------------------- assembled operators
rng = np.random.default_rng(3)
a_r = rng.random((7, 5)) + 0.5
Ko, nyo = ts.open_stiffness(a_r)
uu = np.repeat(np.arange(8.0), 6)  # u = i
check("open_stiffness: energy of u = i equals the sum of the pixel values", uu @ (Ko @ uu), 1e-12, a_r.sum())
check("open_stiffness: rows sum to zero and symmetric", max(abs(Ko @ np.ones(Ko.shape[0])).max(), abs(Ko - Ko.T).max()), 1e-13)
Kp, nyp = ts.open_stiffness(a_r, periodic_y=True)
check_true("open_stiffness periodic_y node count", Kp.shape[0] == 8 * 5 and nyp == 5, f"shape {Kp.shape}")
check("open_stiffness periodic_y: rows sum to zero", abs(Kp @ np.ones(Kp.shape[0])).max(), 1e-13)
mm = 12
rc = np.where(rng.random((mm, mm)) < 0.4, 7.0, 1.0)
Kcs, _ = cs.q1_stiffness(rc)
Kst, _ = ts.open_stiffness(rc, periodic_y=True)
nn = mm * mm
Pfold = sp.hstack([sp.identity(nn), sp.csr_matrix((np.ones(mm), (np.arange(mm), np.arange(mm))), shape=(nn, mm))]).tocsr()
check("open_stiffness(periodic_y) folded in x equals cellsolver.q1_stiffness", abs(Pfold @ Kst @ Pfold.T - Kcs).max(), 1e-13)

# corrector_gram with a non-symmetric cell: polarization identity written out independently
chi_r, _ = ts.cell_correctors(rc)
Gr = ts.corrector_gram(chi_r)
check("corrector_gram symmetric for a random cell", abs(Gr[0, 1] - Gr[1, 0]), 1e-13)
hh = 1.0 / mm
cc = cs._conn(mm, True)
cx_, cy_ = chi_r[0].reshape(-1)[cc], chi_r[1].reshape(-1)[cc]
# <|e_x + grad(chi_x + chi_y)|^2> from the cellsolver energy density of the SUM corrector (direction 0)
lhs = cs.q1_energy_density(np.ones((mm, mm)), np.stack([chi_r[0] + chi_r[1], chi_r[0] + chi_r[1]]))[0].sum()
cross = hh * (cy_ @ cs.G[0]).sum() + np.einsum("ek,kl,el->", cx_, cs.KE, cy_)  # <e_x . grad chi_y> + <grad chi_x . grad chi_y>
rhs = Gr[0, 0] + 2 * cross + (np.einsum("ek,kl,el->", cy_, cs.KE, cy_) + 0.0)
check("corrector_gram: energy of e_x + grad(chi_x + chi_y) expands as M_xx + 2 cross + |grad chi_y|^2", abs(lhs - rhs), 1e-12)
check("corrector_gram M_xy equals e_x.grad chi_y + e_y.grad chi_x + grad chi_x.grad chi_y (independent expansion)",
      Gr[0, 1], 1e-13, hh * (cy_ @ cs.G[0]).sum() + hh * (cx_ @ cs.G[1]).sum() + np.einsum("ek,kl,el->", cx_, cs.KE, cy_))

# ---------------------------------------------------------------------------------------------- fine_dirichlet and harmonic_dirichlet
errs_h = {}
for n in (128, 256):
    u_h = ts.fine_dirichlet(np.ones((n, n)))
    xg = np.arange(n + 1) / n
    errs_h[n] = np.abs(u_h - ts.torsion(xg, xg)[0]).max()
check_range("homogeneous a = 1: max|u_h - u0| ratio n = 128 over 256 (plan [3, 5])", errs_h[128] / errs_h[256], 3.0, 5.0)
check("fine_dirichlet scales as 1/a for constant a", np.abs(ts.fine_dirichlet(np.full((32, 32), 4.0)) - ts.fine_dirichlet(np.ones((32, 32))) / 4).max(), 1e-14)
a32 = ts.periodic_medium(CELL, 2, 0)
u32 = ts.fine_dirichlet(a32)
check("fine_dirichlet: constant rhs array equals scalar", np.abs(ts.fine_dirichlet(a32, np.ones((32, 32))) - u32).max(), 1e-14)
check("fine_dirichlet linear in rhs", np.abs(ts.fine_dirichlet(a32, 2.5) - 2.5 * u32).max(), 1e-14)
check("fine_dirichlet zero on the boundary", max(abs(u32[0]).max(), abs(u32[-1]).max(), abs(u32[:, 0]).max(), abs(u32[:, -1]).max()), 0.0)
check("fine_dirichlet D4 symmetry of the symmetric medium (transpose and flips)", max(np.abs(u32 - u32.T).max(), np.abs(u32 - u32[::-1]).max(), np.abs(u32 - u32[:, ::-1]).max()), 1e-13)
K32, _ = ts.open_stiffness(a32)
check("fine_dirichlet energy identity u^T K u = h^2 sum(u)", u32.ravel() @ (K32 @ u32.ravel()), 1e-13, (1.0 / 32) ** 2 * u32.sum())
r1 = np.zeros((32, 32))
r1[10, 11] = 3.0
u_r1 = ts.fine_dirichlet(a32, r1)
load = 3.0 * (1.0 / 32) ** 2 / 4 * (u_r1[10, 11] + u_r1[11, 11] + u_r1[11, 12] + u_r1[10, 12])
check("fine_dirichlet pixel load: energy equals load times the four corner values", u_r1.ravel() @ (K32 @ u_r1.ravel()), 1e-15, load)
check_true("fine_dirichlet rejects non-square or non-positive input", raises(lambda: ts.fine_dirichlet(np.ones((4, 5)))) and raises(lambda: ts.fine_dirichlet(np.zeros((4, 4)))), "ValueError")

n_h = 40
Xh, Yh = np.meshgrid(np.arange(n_h + 1) / n_h, np.arange(n_h + 1) / n_h, indexing="ij")
g_h = 0.3 + Xh * Yh + 2 * Xh - Yh
check("harmonic_dirichlet reproduces a bilinear function exactly", np.abs(ts.harmonic_dirichlet(np.where((Xh % 1 == 0) | (Yh % 1 == 0), g_h, 123.0)) - g_h).max(), 1e-13)
g_rand = np.random.default_rng(1).standard_normal((n_h + 1, n_h + 1))
u_h = ts.harmonic_dirichlet(g_rand)
Kh, _ = ts.open_stiffness(np.ones((n_h, n_h)))
bnd = np.zeros((n_h + 1, n_h + 1), bool)
bnd[0], bnd[-1], bnd[:, 0], bnd[:, -1] = True, True, True, True
check("harmonic_dirichlet: assembled Q1 Laplacian residual at interior nodes", np.abs((Kh @ u_h.ravel()).reshape(n_h + 1, n_h + 1)[~bnd]).max(), 1e-13)
check("harmonic_dirichlet: boundary values kept", np.abs(u_h[bnd] - g_rand[bnd]).max(), 0.0)
check_true("harmonic_dirichlet: discrete maximum principle", u_h.max() <= g_rand[bnd].max() + 1e-12 and u_h.min() >= g_rand[bnd].min() - 1e-12, "within the data range")

# ---------------------------------------------------------------------------------------------- strip problem and QEP
ms = 16
Ahom = np.ones((6 * ms, ms))
Uc, tc = ts.strip_tail(Ahom, np.full(ms, 2.5))
check("strip_tail: constant data gives the constant (tail and field)", max(abs(tc - 2.5), np.abs(Uc - 2.5).max()), 1e-12)
dat2 = np.stack([np.cos(2 * np.pi * np.arange(ms) / ms), np.sin(2 * np.pi * np.arange(ms) / ms) + 0.3])
U2, t2 = ts.strip_tail(Ahom, dat2)
U2a, t2a = ts.strip_tail(Ahom, dat2[1])
check("strip_tail: batched right-hand sides equal single solves", np.abs(U2[1] - U2a).max() + abs(t2[1] - t2a), 1e-14)
check("strip_tail: tail of data with mean c is c on a homogeneous strip", abs(t2[1] - 0.3), 1e-13)


def kappa_hom(m, k):
    """Exact decay exponent of the discrete homogeneous Q1 strip for the y-mode cos(2 pi k j/m): per pixel
    (2 - s)(2 + c) + (4 + s)(1 - c)/... = 0, i.e. s = r + 1/r = (8 - 2c)/(1 + 2c), c = cos(2 pi k/m)."""
    c = np.cos(2 * np.pi * k / m)
    s = (8 - 2 * c) / (1 + 2 * c)
    return m * np.arccosh(s / 2)


kap_h16, par_h16, det_h16 = ts.evanescent_exponents(np.ones((16, 16)), 8, return_details=True)
check("homogeneous QEP m = 16: kappa_1 equals scratch 6.36553 (plan: 1e-4)", kap_h16[0].real, 1e-4, 6.36553)
check("homogeneous QEP m = 16: kappa_1 equals the closed form of the Q1 strip", kap_h16[0].real, 1e-10, kappa_hom(16, 1))
check("homogeneous QEP m = 16: second pair (k = 2) equals the closed form", kap_h16[2].real, 1e-9, kappa_hom(16, 2))
check("homogeneous QEP: first exponent is a degenerate pair", abs(kap_h16[0] - kap_h16[1]), 1e-9)
check("homogeneous QEP: k = 3 pair (kappa = 21.5, near the accuracy floor of the eigensolver) matches the closed form", abs(kap_h16[4].real - kappa_hom(16, 3)), 1e-6)
kap_h8 = ts.evanescent_exponents(np.ones((8, 8)), 4)[0]
check("homogeneous QEP m = 8 equals scratch 6.63377", kap_h8[0].real, 1e-5, 6.63377)
ratio_q = abs(kap_h16[0].real - 2 * np.pi) / abs(kap_h8[0].real - 2 * np.pi)
check_range("homogeneous QEP converges at second order: |k1(16) - 2 pi| / |k1(8) - 2 pi| (plan [0.2, 0.3])", ratio_q, 0.2, 0.3)
check_true("homogeneous QEP: parities of the pair are one even and one odd", sorted(par_h16[:2].tolist()) == [-1, 1], f"parities {par_h16[:4].tolist()}")
check_true("homogeneous QEP: all returned exponents are below the cap 30", bool(np.all(kap_h16.real < 30.0)), f"max {kap_h16.real.max():.3f}")
check("homogeneous QEP: parity expectation is +-1 for kappa < 25 (after rotating degenerate pairs)", np.abs(np.abs(det_h16["parity_expectation"].real[kap_h16.real < 25]) - 1).max(), 1e-8)

kap_d, par_d, det_d = ts.evanescent_exponents(CELL, 6, return_details=True)
check("QEP residual (disk cell, a_in = 10) below 1e-8 (plan)", det_d["residual"].max(), 1e-8)
check("QEP parity expectation is +-1 within 1e-8 (plan)", np.abs(np.abs(det_d["parity_expectation"].real) - 1).max(), 1e-8)
check("QEP smallest exponent a_in = 10 reproduces scratch 6.8335", kap_d[0].real, 1e-4, 6.8335)
check("QEP second exponent a_in = 10 reproduces scratch 7.3387", kap_d[1].real, 1e-4, 7.3387)
check_true("QEP parities a_in = 10: even then odd", par_d[0] == 1 and par_d[1] == -1, f"{par_d.tolist()}")
check("QEP exponents are real for the symmetric disk cell (imag part)", np.abs(kap_d.imag).max(), 1e-8)
for meth_cell, name in ((np.ones((8, 8)), "homogeneous m = 8"), (CELL, "disk m = 16"), (cs.disk_pixels(8, 0.3, 100.0, 1.0), "disk a_in = 100 m = 8")):
    kr = ts.evanescent_exponents(meth_cell, 6, "reduced")[0]
    kf = ts.evanescent_exponents(meth_cell, 6, "full")[0]
    check(f"QEP reduced equals the full linearization for kappa < 15 ({name})", np.abs((kr - kf)[kr.real < 15]).max(), 1e-9)
rand_cell = np.where(np.random.default_rng(5).random((8, 8)) < 0.5, 5.0, 1.0)
kr = ts.evanescent_exponents(rand_cell, 5, "reduced")[0]
kf = ts.evanescent_exponents(rand_cell, 5, "full")[0]
check("QEP reduced equals the full linearization for Re kappa < 15 (random non-symmetric cell, complex exponents allowed)", np.abs((kr - kf)[kr.real < 15]).max(), 1e-9)
check_true("QEP: exponents ascending in Re, decaying (Re > 1e-6)", np.all(np.diff(kap_d.real) >= 0) and kap_d.real.min() > 1e-6, "ok")
scratch_k1 = {10.0: 6.8335, 100.0: 7.1569, 0.1: 6.7474}
scratch_k2 = {10.0: 7.3387, 100.0: 9.378, 0.1: 7.4113}
for ain in (0.1, 10.0, 100.0):
    kk = ts.evanescent_exponents(cs.disk_pixels(16, 0.3, ain, 1.0), 4)[0].real
    check_true(f"heterogeneity thins the layer, a_in = {ain:g}: kappa_1 {kk[0]:.4f} > homogeneous {kap_h16[0].real:.4f}", kk[0] > kap_h16[0].real, "strict")
    check(f"QEP scratch kappa_1, a_in = {ain:g}", kk[0], 6e-4, scratch_k1[ain])
    check(f"QEP scratch kappa_2, a_in = {ain:g}", kk[1], 6e-4, scratch_k2[ain])

# strip decay equals the QEP exponent of the matching parity (same discrete operator)
Ns = 10
dec = {}
for lab, shift, dcorr, par in (("odd (chi_2, s = 0)", (0, 0), 1, -1), ("even (chi_1, s = (1/4, 0))", (4, 0), 0, 1)):
    a_ = ts.periodic_medium(CELL, Ns, shift)
    cn_ = np.stack([ts.periodic_nodal(CHI[k], Ns, shift) for k in range(2)])
    tl, fl = ts.edge_tails(a_, cn_, M_PIX, 6, 0, True)
    rates = ts.strip_decay_rates(fl["left"][dcorr], M_PIX, tl[dcorr, 0])
    kq = kap_d[par_d == par][0].real
    check(f"strip decay between depths 1 and 2 equals the QEP exponent of its parity, a_in = 10, {lab} (plan 1e-2)", abs(rates[1] / kq - 1), 1e-2)
    dec[lab] = (rates[1], kq)
# homogeneous strip with a pure mode: decay is the closed-form exponent
Uh, th = ts.strip_tail(Ahom, np.cos(2 * np.pi * np.arange(ms) / ms))
rh = ts.strip_decay_rates(Uh, ms, th)
check("homogeneous strip, mode k = 1: decay rate equals the closed form", rh[1], 1e-6, kappa_hom(16, 1))

# ---------------------------------------------------------------------------------------------- norms and weak integrals
nb = 64
Xb, Yb = np.meshgrid(np.arange(nb + 1) / nb, np.arange(nb + 1) / nb, indexing="ij")
bil = Xb * Yb
li, l2, h1 = ts.norms(bil)
check("norms: L2 of the bilinear xy is 1/3 exactly (exact mass matrix)", l2, 1e-14, 1 / 3)
check("norms: H1 seminorm of xy is sqrt(2/3) exactly", h1, 1e-13, np.sqrt(2 / 3))
cr_ = rng.standard_normal((4, 5, 5))
check("element energy (differences form) equals the KE quadratic form", np.abs(ts._element_energy(cr_) - np.einsum("kij,kl,lij->ij", cr_, cs.KE, cr_)).max(), 1e-13)
check("norms: Linf of xy is 1", li, 1e-15, 1.0)
li, l2, h1 = ts.norms(bil, BOX)
x2 = (0.75 ** 3 - 0.25 ** 3) / 3
check("norms with the box: L2 over [1/4, 3/4]^2", l2, 1e-14, x2)
check("norms with the box: H1 over [1/4, 3/4]^2", h1, 1e-13, np.sqrt(2 * x2 * 0.5))
check("norms with the box: Linf over the box nodes", li, 1e-15, 0.5625)
ux_, uy_ = ts.element_gradients(bil)
xcb = (np.arange(nb) + 0.5) / nb
check("element_gradients of xy: mean gradient (y, x) at the pixel centers", max(np.abs(ux_ - xcb[None, :]).max(), np.abs(uy_ - xcb[:, None]).max()), 1e-13)
phib = np.sin(np.pi * xcb)[:, None] * np.sin(np.pi * xcb)[None, :]
one = np.ones((nb, nb))
Xl = np.broadcast_to((np.arange(nb + 1) / nb)[:, None], (nb + 1, nb + 1)).copy()
If_, Ig_ = ts.weak_integrals(Xl, 2.0 * one, phib)
check("weak_integrals of u = x: I_grad = int phi (midpoint)", Ig_, 1e-12, np.sum(phib) / nb ** 2)
check("weak_integrals of u = x: I_flux = 2 I_grad", If_, 1e-12, 2 * Ig_)
If_e, Ig_e = ts.weak_integrals(Xl, 2.0 * one, phib, exact_energy=True)
check("weak_integrals(exact_energy) agrees with the midpoint rule for a linear field", abs(If_e - If_) + abs(Ig_e - Ig_), 1e-12)
lf, lg, nv = ts.weak_limits(phib, np.ones((nb, nb)), np.zeros((nb, nb)), 3.0 * np.eye(2), 1.5 * np.eye(2))
check("weak_limits for grad u0 = e_x, sigma = 3, M = 1.5", max(abs(lf - 3 * Ig_), abs(lg - 1.5 * Ig_), abs(nv - Ig_)), 1e-12)
Sm, Mm = ts.midpoint_tensors(CELL, CHI)
check_true("midpoint_tensors close to (sigma, M) but not equal (twist energy)", 1e-3 < abs(Sm[0, 0] - SS) / SS < 0.05 and 1e-3 < abs(Mm[0, 0] - GRAM[0, 0]) / GRAM[0, 0] < 0.05,
           f"S {Sm[0, 0]:.5f} vs {SS:.5f}; M {Mm[0, 0]:.5f} vs {GRAM[0, 0]:.5f}")

# ---------------------------------------------------------------------------------------------- the notebook-04 computation
Nlist = [4, 8, 16, 32]
shifts = {0: 0, 0.25: 4, 0.5: 8}
R = {}
t_runs = time.perf_counter()
for sname, spx in shifts.items():
    for N in Nlist:
        n = N * M_PIX
        a = ts.periodic_medium(CELL, N, spx)
        u = ts.fine_dirichlet(a)
        xg = np.arange(n + 1) / n
        u0, ux, uy = (q / SS for q in ts.torsion(xg, xg))
        cn = np.stack([ts.periodic_nodal(CHI[d], N, spx) for d in range(2)])
        w = ts.first_order_expansion(u0, ux, uy, cn, N)
        tails = ts.edge_tails(a, cn, M_PIX, 6, 0)
        u1 = ts.harmonic_dirichlet(ts.edge_data(tails, ux, uy))
        e0, e1, e1i = ts.norms(u - u0), ts.norms(u - w), ts.norms(u - w, BOX)
        resid = ts.norms((u - w) * N - u1, BOX)[0]
        u1max = ts.norms(u1, BOX)[0]
        xc = (np.arange(n) + 0.5) / n
        phi = np.sin(np.pi * xc)[:, None] * np.sin(np.pi * xc)[None, :]
        _, cx, cy = ts.torsion(xc, xc)
        cx, cy = cx / SS, cy / SS
        out = dict(Linf0=e0[0], L20=e0[1], H1=e1[2], H1i=e1i[2], Linfi=e1i[0], resid=resid, u1max=u1max, tails=tails)
        for tag, ex in (("mid", False), ("ex", True)):
            If, Ig = ts.weak_integrals(u, a, phi, exact_energy=ex)
            lf, lg, nv = ts.weak_limits(phi, cx, cy, SIG if ex else Sm, GRAM if ex else Mm)
            out[f"flux_{tag}"], out[f"grad_{tag}"] = If - lf, Ig - lg
            if ex:
                lf0, lg0, nv0 = ts.weak_limits(phi, cx, cy, SIG, GRAM)
                out["naive_gap"], out["pred_gap"] = Ig - nv0, lg0 - nv0
        # plan version: midpoint integrals against the exact-tensor limits
        If, Ig = ts.weak_integrals(u, a, phi)
        lf, lg, nv = ts.weak_limits(phi, cx, cy, SIG, GRAM)
        out["flux_plan"], out["grad_plan"], out["naive_mid"], out["pred_mid"] = If - lf, Ig - lg, Ig - nv, lg - nv
        R[(sname, N)] = out
print(f"      (notebook-04 style runs, 12 fine solves up to n = 512: {time.perf_counter() - t_runs:.1f} s)")


def col(sname, key, Ns_=Nlist):
    return np.array([R[(sname, N)][key] for N in Ns_])


Ng = np.array(Nlist, float)
for sname in shifts:
    check(f"L2 slope of u_eps - u0 over N = 4-32, s = {sname} (plan: within 0.15 of 1)", slope(Ng, col(sname, "L20")), 0.15, 1.0)
check("L_inf, L2 of u_eps - u0 at s = 0, N = 4 are positive and decrease", float(np.all(np.diff(col(0, "Linf0")) < 0) and np.all(np.diff(col(0, "L20")) < 0)), 0.0, 1.0)
loc = np.log2(col(0.25, "H1")[-2] / col(0.25, "H1")[-1])
check_range("H1 with a generic cut: last-pair slope at s = 1/4 (plan [0.45, 0.70]; scratch 0.586)", loc, 0.45, 0.70)
check("H1 global slope at s = 1/4 reproduces scratch 0.65 (overall)", slope(Ng, col(0.25, "H1")), 0.02, 0.646)
check("H1 global slope, s = 0 (plan: within 0.1 of 1; scratch 0.96)", slope(Ng, col(0, "H1")), 0.1, 1.0)
check("H1 global slope, s = 1/2 (plan: within 0.1 of 1; scratch 0.99)", slope(Ng, col(0.5, "H1")), 0.1, 1.0)
for sname in shifts:
    check(f"interior H1 slope, s = {sname} (plan: within 0.1 of 1; scratch 1.007)", slope(Ng, col(sname, "H1i")), 0.1, 1.0)
# section 3
for sname in (0, 0.5):
    check(f"interior L_inf of u_eps - w_eps: slope over N = 8-32 at s = {sname} (plan: within 0.2 of 2)", slope(Ng[1:], col(sname, "Linfi")[1:]), 0.2, 2.0)
check("interior L_inf slope over N = 8-32 at s = 1/4 (plan: within 0.1 of 1)", slope(Ng[1:], col(0.25, "Linfi")[1:]), 0.1, 1.0)
for sname, scr in ((0, [8.59e-4, 2.39e-4, 6.21e-5]), (0.5, [1.18e-3, 3.27e-4, 8.57e-5]), (0.25, [5.68e-3, 2.79e-3, 1.37e-3])):
    got = col(sname, "Linfi")[:3]
    check(f"interior max error N = 4, 8, 16 at s = {sname} reproduces the design scratch (relative)", float(np.abs(got / np.array(scr) - 1).max()), 6e-3)
check("symmetry zeros: max|chi_1| on node columns y1 = 0 and 1/2", max(np.abs(CHI[0][0, :]).max(), np.abs(CHI[0][M_PIX // 2, :]).max()), 1e-12)
check("symmetry zeros: max|chi_2| on node rows y2 = 0 and 1/2", max(np.abs(CHI[1][:, 0]).max(), np.abs(CHI[1][:, M_PIX // 2]).max()), 1e-12)
# section 4
tail_q = R[(0.25, 32)]["tails"]
check("tail on x1 = 0 edge (chi_1, s = 1/4) reproduces scratch 0.13554", tail_q[0, 0], 5e-6, 0.13554)
check("tail on x1 = 1 edge (chi_1, s = 1/4) reproduces scratch 0.16978", tail_q[0, 1], 5e-6, 0.16978)
check("tails on the x2 edges for chi_2 mirror the x1 edges (diagonal symmetry of the cell)", abs(tail_q[1, 2] - tail_q[0, 0]) + abs(tail_q[1, 3] - tail_q[0, 1]), 1e-12)
check("tangential-corrector tails vanish: chi_2 on x1 edges, chi_1 on x2 edges (all N, s = 1/4)", max(np.abs(R[(0.25, N)]["tails"][[1, 1, 0, 0], [0, 1, 2, 3]]).max() for N in Nlist), 1e-12)
check("mirror cuts have zero tails (s = 0 and 1/2, all N)", max(np.abs(R[(s_, N)]["tails"]).max() for s_ in (0, 0.5) for N in Nlist), 1e-12)
a_d = ts.periodic_medium(CELL, 10, 4)
cn_d = np.stack([ts.periodic_nodal(CHI[k], 10, 4) for k in range(2)])
check("tail independent of strip depth: |tail(6 cells) - tail(8 cells)| (plan 1e-12)", np.abs(ts.edge_tails(a_d, cn_d, M_PIX, 6, 0) - ts.edge_tails(a_d, cn_d, M_PIX, 8, 0)).max(), 1e-12)
check("tails do not depend on the offset j0 along the edge (period m)", np.abs(ts.edge_tails(a_d, cn_d, M_PIX, 6, 0) - ts.edge_tails(a_d, cn_d, M_PIX, 6, 5)).max(), 1e-12)
tails_0 = ts.edge_tails(a_d, cn_d, M_PIX, 6, 0)
# wiring: transposed medium swaps the roles of the edges, reversed medium swaps left/right
tt = ts.edge_tails(a_d.T.copy(), cn_d[::-1].transpose(0, 2, 1).copy(), M_PIX, 6, 0)
check("edge_tails wiring: transposed medium maps (chi_j, left/right) to (chi_(1-j), bottom/top)", np.abs(tt[:, :2] - tails_0[::-1, 2:]).max() + np.abs(tt[:, 2:] - tails_0[::-1, :2]).max(), 1e-12)
tf = ts.edge_tails(a_d[::-1].copy(), cn_d[:, ::-1].copy(), M_PIX, 6, 0)
check("edge_tails wiring: x-reversed medium swaps left and right", np.abs(tf[:, 0] - tails_0[:, 1]).max() + np.abs(tf[:, 1] - tails_0[:, 0]).max(), 1e-12)
ge = ts.edge_data(np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]), np.full((9, 9), 0.5), np.full((9, 9), 0.25))
check("edge_data: left, right, bottom, top values (tail_x d_x u + tail_y d_y u)", max(abs(ge[0, 3] - 3.0), abs(ge[8, 3] - 6.0), abs(ge[3, 0] - (1.5 + 7.5)), abs(ge[3, 8] - (2.0 + 10.0))), 1e-14)
check("edge_data: interior is zero", np.abs(ge[1:-1, 1:-1]).max(), 0.0)
res_s = col(0.25, "resid")[1:]
check("u1bar prediction: max_int |(u_eps - w_eps)/eps - u1bar| slope over N = 8-32 (plan >= 0.8)", slope(Ng[1:], res_s) - 0.8, 1e9) if False else None
sl_res = slope(Ng[1:], res_s)
check_true("u1bar prediction: residual slope over N = 8-32 is >= 0.8 (plan)", sl_res >= 0.8, f"slope {sl_res:.3f}")
check("u1bar prediction: residual / max|u1bar| at N = 32 (plan <= 0.1; scratch 7%)", col(0.25, "resid")[-1] / col(0.25, "u1max")[-1], 0.1 - 0.0, 0.0) if False else None
ratio_u1 = col(0.25, "resid")[-1] / col(0.25, "u1max")[-1]
check_true("u1bar prediction: residual / max|u1bar| at N = 32 is <= 0.1 (plan; scratch 7%)", ratio_u1 <= 0.1, f"{ratio_u1:.4f}")
check("max_int|u1bar| at N = 32 reproduces scratch 0.02126", col(0.25, "u1max")[-1], 5e-5, 0.02126)
for N, scr in zip((8, 16, 32), (5.37e-3, 2.90e-3, 1.51e-3)):
    check(f"max_int |err/eps - u1bar| at N = {N} reproduces scratch (relative)", R[(0.25, N)]["resid"] / scr - 1, 7e-3)
check_true("the correction improves the interior error: residual < |err/eps| at every N, s = 1/4", all(R[(0.25, N)]["resid"] < R[(0.25, N)]["Linfi"] / (1.0 / N) for N in Nlist), "u1bar explains half or more")
# section 6
check("weak limits: M_11 for the cell reproduces scratch 1.50455", GRAM[0, 0], 5e-6, 1.50455)
fl_slope = slope(Ng, np.abs(col(0, "flux_plan")))
gr_slope = slope(Ng, np.abs(col(0, "grad_plan")))
check_true("div-curl, plan version (midpoint integral vs sigma*): slope of |I_flux - lim| over N = 4-32 >= 0.8", fl_slope >= 0.8, f"slope {fl_slope:.3f} (scratch 1.12)")
check_true("two-scale, plan version (midpoint integral vs M): slope of |I_grad - lim_M| over N = 4-32 >= 0.8", gr_slope >= 0.8, f"slope {gr_slope:.3f} (scratch 1.22)")
loc_f = np.log2(abs(col(0, "flux_plan")[-2] / col(0, "flux_plan")[-1]))
print(f"DISCREPANCY  plan section 6: the overall slopes pass but are pre-asymptotic: the last-pair slopes are {loc_f:.2f} (flux) and "
      f"{np.log2(abs(col(0, 'grad_plan')[-2] / col(0, 'grad_plan')[-1])):.2f} (grad) and the errors stall near 5e-5; the midpoint rule converges to "
      f"midpoint_tensors (S, M), not to (sigma*, M). With exact_energy=True the slopes are asserted below.")
fl_ex = np.abs(col(0, "flux_ex"))
gr_ex = np.abs(col(0, "grad_ex"))
check_true("div-curl with exact element energy: slope of |I_flux - lim| >= 0.8 and last-pair slope >= 1.5", slope(Ng, fl_ex) >= 0.8 and np.log2(fl_ex[-2] / fl_ex[-1]) >= 1.5,
           f"overall {slope(Ng, fl_ex):.3f}, last pair {np.log2(fl_ex[-2] / fl_ex[-1]):.3f}")
check_true("two-scale with exact element energy: slope of |I_grad - lim_M| >= 0.8 and last-pair slope >= 1.5", slope(Ng, gr_ex) >= 0.8 and np.log2(gr_ex[-2] / gr_ex[-1]) >= 1.5,
           f"overall {slope(Ng, gr_ex):.3f}, last pair {np.log2(gr_ex[-2] / gr_ex[-1]):.3f}")
fl_mid = np.abs(col(0, "flux_mid"))
check_true("midpoint integrals converge at rate 2 to the midpoint_tensors limits (flux)", np.log2(fl_mid[-2] / fl_mid[-1]) >= 1.5, f"last pair {np.log2(fl_mid[-2] / fl_mid[-1]):.3f}")
ratio_gap = R[(0, 32)]["naive_mid"] / R[(0, 32)]["pred_mid"]
check_range("misses the naive limit by the corrector term: |I_grad - naive| / predicted gap at N = 32 (plan [0.9, 1.1]; scratch 0.97)", ratio_gap, 0.9, 1.1)
check("predicted gap int phi grad u0.(M - I) grad u0 at N = 32 reproduces scratch 1.614e-3", R[(0, 32)]["pred_mid"], 5e-6, 1.614e-3)
check("measured gap at N = 32 reproduces scratch 1.565e-3", R[(0, 32)]["naive_mid"], 5e-6, 1.565e-3)
ratio_ex = R[(0, 32)]["naive_gap"] / R[(0, 32)]["pred_gap"]
check_range("same ratio with the exact element energy (consistent partner of M)", ratio_ex, 0.99, 1.01)
check_true("the gradient energy converges to the M-weighted limit, not the naive one (N = 32)", abs(R[(0, 32)]["grad_ex"]) < 0.05 * abs(R[(0, 32)]["pred_gap"]), f"|I - lim_M| {abs(R[(0, 32)]['grad_ex']):.2e} vs gap {R[(0, 32)]['pred_gap']:.2e}")

dt = time.perf_counter() - T0
print(f"\ntotal run time {dt:.1f} s; {len(FAILS)} failure(s)")
if FAILS:
    print("FAILED:", *FAILS, sep="\n  ")
    sys.exit(1)
