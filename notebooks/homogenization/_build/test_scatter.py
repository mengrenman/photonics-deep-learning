"""Standalone test of scatter.py: prints one PASS/FAIL line per item and exits nonzero on any FAIL.

    cd notebooks/homogenization && python _build/test_scatter.py

A line starting with DISCREPANCY marks a statement of PLAN.md that does not reproduce; it is not counted as a failure.
The testbed forward model is called only at n <= 64. Typical run time: about 15 s with 4 threads.
"""
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("NPROC", "4")
FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FOLDER))

import numpy as np

import bloch as bl
import scatter as sc
from bayes_scatter import detectability as det
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup, ring
from bayes_scatter.mie import mie_scattered
from bayes_scatter.phantoms import cylinder

T0 = time.perf_counter()
FAILS = []


def item(name, ok, value, tol):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {value} (tolerance {tol})", flush=True)
    if not ok:
        FAILS.append(name)


def le(name, value, tol):
    item(name, bool(value <= tol), f"{value:.3e}", f"<= {tol:g}")


def within(name, value, lo, hi):
    item(name, bool(lo <= value <= hi), f"{value:.4g}", f"in [{lo:g}, {hi:g}]")


def info(msg):
    print("INFO " + msg, flush=True)


def rel(a, b):
    return np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b)


tx, rx = ring(40, 4.0), ring(60, 4.5)           # the ch15 acquisition
rng = np.random.default_rng(0)

# ------------------------------------------------------------------------------------------------ multilayer Mie
for eps_r, R in ((2.25, 1.0), (12.0, 1.0), (4.0, 0.3)):
    ref = mie_scattered(tx, rx, R, eps_r)
    le(f"exact_data (copied multilayer) vs bayes_scatter.mie.mie_scattered, homogeneous disk eps={eps_r:g}, R={R:g} (max rel)",
       np.abs(sc.exact_data(tx, rx, [R], [eps_r]) - ref).max() / np.abs(ref).max(), 1e-12)
    le(f"mie_disk (centered) vs mie_scattered, eps={eps_r:g}, R={R:g} (max rel)",
       np.abs(sc.mie_disk(eps_r, R, (0, 0), tx, rx) - ref).max() / np.abs(ref).max(), 1e-12)
ref = mie_scattered(tx, rx, 0.7, 3.0, center=(0.1, -0.2))
le("mie_disk off center (0.1, -0.2) vs mie_scattered (max rel)",
   np.abs(sc.mie_disk(3.0, 0.7, (0.1, -0.2), tx, rx) - ref).max() / np.abs(ref).max(), 1e-12)
le("two layers with equal eps equal the homogeneous disk (multilayer consistency, rel)",
   rel(sc.exact_data(tx, rx, [0.4, 1.0], [2.25, 2.25]), sc.exact_data(tx, rx, [1.0], [2.25])), 1e-12)
radii, eps = sc.rings(1.0, 20, 0.5, 3.0, 1.5, "period")
le("M = 60 vs 90 orders, ring structure n_p = 20 (rel)", rel(sc.exact_data(tx, rx, radii, eps, M=60), sc.exact_data(tx, rx, radii, eps, M=90)), 1e-12)
le("M = 60 vs 90 orders, homogeneous disk (rel)", rel(sc.exact_data(tx, rx, [1.0], [2.25], M=60), sc.exact_data(tx, rx, [1.0], [2.25], M=90)), 1e-12)
le("M = 60 vs 90 orders, mie_disk eps=12 R=1 (rel)", rel(sc.mie_disk(12.0, 1.0, (0, 0), tx, rx, M=60), sc.mie_disk(12.0, 1.0, (0, 0), tx, rx, M=90)), 1e-12)

# ------------------------------------------------------------------------------------------------ tangents and fit
p4 = np.array([2.25, 1.0, 0.03, -0.02])
Ta = sc.disk_tangents(p4, tx, rx)
Tfd = []
for i in range(4):
    out = []
    for h in (2e-4, 1e-4):
        pp, pm = p4.copy(), p4.copy()
        pp[i] += h
        pm[i] -= h
        out.append((sc.mie_disk(*pp[:2], pp[2:], tx, rx) - sc.mie_disk(*pm[:2], pm[2:], tx, rx)) / (2 * h))
    Tfd.append((4 * out[1] - out[0]) / 3)                   # Richardson: fourth order in h
le("analytic tangents (eps, R, cx, cy) vs Richardson central differences (max rel)", max(rel(Ta[i], Tfd[i]) for i in range(4)), 1e-8)
Ta0 = sc.disk_tangents([2.25, 1.0, 0.0, 0.0], tx, rx)
Q0, Q1 = sc.c20_projector(0), sc.c20_projector(1)
le("model tangents: ||(I - Q_0) d_eps||, ||(I - Q_0) d_R|| relative (centered disk)",
   max(rel(Q0(Ta0[i]), Ta0[i]) for i in (0, 1)), 1e-12)
le("model tangents: ||(I - Q_1) d_cx||, d_cy relative (centered disk)",
   max(rel(Q1(Ta0[i]), Ta0[i]) for i in (2, 3)), 1e-12)
dh = sc.mie_disk(2.25, 1.0, (0, 0), tx, rx)
le("homogeneous centered-disk data lie in Q_0 (rel)", rel(Q0(dh), dh), 1e-12)
s0 = sc.noise_level(dh, 30.0)
ph, Dh, rh = sc.fit_disk(dh, tx, rx, [2.25, 1.0], s0)
le("fit_disk at the true parameters, noise-free: residual norm / ||whitened data||", np.linalg.norm(rh) / np.linalg.norm(sc.whiten(dh, s0)), 1e-12)
ph, Dh, rh = sc.fit_disk(sc.mie_disk(2.6, 0.93, (0, 0), tx, rx), tx, rx, [2.25, 1.0], s0, iters=15)
le("fit_disk recovers a homogeneous disk's own (eps, R) from noise-free data, started at (2.25, 1.0) (max abs)", np.abs(ph - [2.6, 0.93]).max(), 1e-9)
ph4, _, _ = sc.fit_disk(sc.mie_disk(2.6, 0.93, (0.04, -0.03), tx, rx), tx, rx, [2.25, 1.0, 0.0, 0.0], s0, free=("eps", "R", "cx", "cy"), iters=20)
le("fit_disk with the center free recovers (eps, R, cx, cy) (max abs)", np.abs(ph4 - [2.6, 0.93, 0.04, -0.03]).max(), 1e-9)
ph, _, _ = sc.fit_disk(sc.mie_disk(2.6, 0.93, (0, 0), tx, rx), tx, rx, [2.25, 1.0, 0.0, 0.0], s0, free=("R",), iters=10)
item("fit_disk free=('R',) leaves the other parameters at their start values", bool(ph[0] == 2.25 and ph[2] == 0 and ph[3] == 0), f"p = {np.round(ph, 4)}", "exact")
yw = sc.whiten(dh, s0)
le("whiten convention: Phi = (1/2)|r_w|^2 = sum |r|^2/s^2 (rel)", abs(0.5 * yw @ yw / (np.sum(np.abs(dh) ** 2) / s0**2) - 1), 1e-14)
le("noise_level: 30 dB (s^2 = mean|d|^2 1e-3) (rel)", abs(s0**2 / (np.mean(np.abs(dh) ** 2) * 1e-3) - 1), 1e-14)

# ------------------------------------------------------------------------------------------------ C20 projectors
dr = rng.standard_normal((40, 60)) + 1j * rng.standard_normal((40, 60))
Qs = [sc.c20_projector(k) for k in range(11)]
le("projectors: sum_k Q_k d = d on random data (rel)", rel(sum(Q(dr) for Q in Qs), dr), 1e-12)
err = 0.0
for k in range(11):
    for l in range(11):
        err = max(err, np.abs(Qs[k](Qs[l](dr)) - (Qs[k](dr) if k == l else 0 * dr)).max())
le("projectors: Q_k Q_l = delta_kl Q_k on random data, all 121 pairs (max abs, data O(1))", err, 1e-12)
le("projectors: real data stay real, Q_k real-linear (|Im Q_k Re d| max)", max(np.abs(np.imag(Q(dr.real))).max() for Q in Qs), 1e-15)
le("c20_fractions sum to 1", abs(sc.c20_fractions(dr).sum() - 1), 1e-14)
# Q_k on a pure rotation harmonic: a data array built to carry the character k
tt, rr_ = np.meshgrid(np.arange(40), np.arange(60), indexing="ij")
harm = np.cos(2 * np.pi * 3 * (tt / 2.0) / 20 + 0.3) * 1.0                    # depends on t only: roll by 2j shifts the phase by 2 pi 3 j/20
le("Q_3 keeps a pure k = 3 pattern, Q_2 kills it (rel)", max(rel(Qs[3](harm), harm), np.abs(Qs[2](harm)).max()), 1e-12)

# ------------------------------------------------------------------------------------------------ rings
def rings_scratch(RD, n_p, f, e1, e2, cut):
    p = RD / n_p
    radii, eps = [], []
    if cut == "period":
        for k in range(n_p):
            radii += [k * p + f * p, (k + 1) * p]
            eps += [e1, e2]
    else:
        r = (1 - f) * p / 2
        radii.append(r)
        eps.append(e2)
        for k in range(n_p):
            radii += [r + f * p]
            eps += [e1]
            r += f * p
            t = (1 - f) * p / 2 if k == n_p - 1 else (1 - f) * p
            radii += [r + t]
            eps += [e2]
            r += t
    return np.array(radii), np.array(eps)


err = 0.0
for cut in ("period", "mid"):
    for n_p in (5, 10, 40):
        a, b = sc.rings(1.0, n_p, 0.5, 3.0, 1.5, cut)
        a0, b0 = rings_scratch(1.0, n_p, 0.5, 3.0, 1.5, cut)
        err = max(err, np.abs(a - a0).max() if a.shape == a0.shape else 1.0, np.abs(b - b0).max() if b.shape == b0.shape else 1.0)
le("rings(period, mid) vs the explicit prototype construction (radii and eps, max abs; shapes must match)", err, 1e-14)
ok, worst = True, 0.0
for trial in range(40):
    f_, n_p, c = rng.uniform(0.2, 0.8), rng.choice([7, 12.4, 25]), rng.uniform(0, 1)
    a, b = sc.rings(1.3, n_p, f_, 3.0, 1.5, c)
    pp = 1.3 / n_p
    rs = rng.uniform(1e-3, 1.3 - 1e-3, 400)
    want = np.where(((rs / pp - n_p + c) % 1.0) < f_, 3.0, 1.5)
    got = b[np.searchsorted(a, rs)]
    mism = np.mean(got != want)
    worst = max(worst, mism)
    ok &= bool(a[-1] == 1.3 and np.all(np.diff(a) > 1e-9) and np.all(np.diff(b) != 0))
item("rings with random phase and non-integer n_p: layer eps equals the direct profile at 400 random radii; last radius R_D; no empty or merged layers",
     ok and worst == 0, f"mismatching fraction {worst:g}", "0")
a, b = sc.rings(1.0, 10, 0.5, 3.0, 1.5, "period")
item("period cut: core eps1, outermost eps2 ending at R_D; n_p periods", bool(b[0] == 3.0 and b[-1] == 1.5 and a[-1] == 1.0 and len(a) == 20), f"{len(a)} layers", "20 layers")
a, b = sc.rings(1.0, 10, 0.5, 3.0, 1.5, "mid")
item("mid cut: eps2 core and outermost layers of half thickness (1-f)p/2", bool(b[0] == 1.5 and b[-1] == 1.5 and abs(a[0] - 0.025) < 1e-14 and abs(a[-1] - a[-2] - 0.025) < 1e-14), f"{a[0]:.4f}, {a[-1] - a[-2]:.4f}", "0.025")
for cut in ("period", "mid"):
    a, b = sc.rings(1.0, 40, 0.5, 3.0, 1.5, cut)
    area = np.sum(np.diff(np.concatenate([[0.0], a ** 2])) * b) / 1.0
    le(f"ring area average of eps over the disk ({cut}) vs f eps1 + (1-f) eps2 at p = 0.025", abs(area - 2.25), 0.05)

# ------------------------------------------------------------------------------------------------ TM second-order formulas
for f_, e1, e2, p in ((0.5, 3.0, 1.5, 0.1), (0.3, 4.0, 1.0, 0.2), (0.7, 12.0, 1.0, 0.05)):
    le(f"1D second-order sum vs Rytov closed form, f={f_}, eps=({e1:g},{e2:g}) (rel)",
       abs(sc.tm_second_order_1d(f_, e1, e2, p) / sc.rytov_tm(f_, e1, e2, p) - 1), 1e-10)
sq = bl.Lattice.square(1.0)
G6, _ = bl.gvectors(sq, bl.gmax_for_count(sq, 600), 0)
for eps_r, plan in ((4.0, 1.2874), (12.0, 17.309)):
    rods = [bl.disk((0, 0), 0.3, eps_r)]
    gm = bl.gmax_for_count(sq, 600)
    S = sc.tm_second_order_sum(sq, rods, gm)
    le(f"second-order sum over 600 plane waves, eps={eps_r:g}, in (a/lambda)^2 vs PLAN {plan} (rel)", abs(S / plan - 1), 1e-3)
    Sk = sc.tm_second_order_sum(sq, rods, gm, units="k0")
    le(f"units: (2 pi)^2 S_k0 = S_(a/lambda)^2, eps={eps_r:g} (rel)", abs(S / ((2 * np.pi) ** 2 * Sk) - 1), 1e-14)
    Sfull = sc.tm_second_order_sum(sq, rods, 40 * np.pi)
    info(f"eps = {eps_r:g}: 600-wave sum {S:.5f}, converged sum (|G| < 40 pi) {Sfull:.5f}; PLAN's values are the 600-wave sums")
    B6 = bl.eps_matrix(sq, rods, G6)
    vals = {}
    for Ka in (0.1, 0.2):
        vals[Ka] = sc.tm_residual_k04(sq, rods, np.array([Ka, 0.0]), G6, Sk, B=B6)
    within(f"(eps_eff - eps_2nd)/k0^4 at Ka = 0.1 over 0.2, eps={eps_r:g}: within 5% of 1", vals[0.1] / vals[0.2], 0.95, 1.05)
    ee, k0_ = sc.eps_eff_pwe(sq, rods, np.array([0.05, 0.0]), G6, B=B6)
    avg = float(np.real(bl.eps_hat(sq, rods, np.zeros((1, 2)))[0]))
    le(f"PWE eps_eff - <eps> vs the second-order term at Ka = 0.05, eps={eps_r:g} (rel to the correction)",
       abs((ee - avg) / (k0_**2 * Sk) - 1), 5e-2 if eps_r == 12.0 else 1e-2)

# ------------------------------------------------------------------------------------------------ rings: what a homogeneous fit absorbs (section 3 of nb06)
e1, e2, f, RD = 3.0, 1.5, 0.5, 1.0
ebar, chibar, dchi = f * e1 + (1 - f) * e2, f * e1 + (1 - f) * e2 - 1, e1 - e2
d0 = sc.exact_data(tx, rx, [RD], [ebar])
s = sc.noise_level(d0, 30.0)
D0 = sc.disk_jacobian([ebar, RD], tx, rx, s)
nps = [10, 20, 40, 80]
rows = {"period": {}, "mid": {}}
for cut in rows:
    for n_p in nps:
        p = RD / n_p
        radii, eps = sc.rings(RD, n_p, f, e1, e2, cut)
        d = sc.exact_data(tx, rx, radii, eps)
        pR, DR, rR = sc.fit_disk(d, tx, rx, [ebar, RD], s, free=("R",))
        p2, D2, r2 = sc.fit_disk(d, tx, rx, [ebar, RD], s)
        delta = sc.whitened_delta(d, d0, s)
        base = det.analyze_error(D0, delta)
        if cut == "period":
            pf, Df, rf = sc.foot_point_R(d, tx, rx, ebar, RD, s)
            first = det.analyze_error(Df, rf)
            zNL = (p2[0] - ebar) / first["sigma"][0]
        else:
            first = base
            zNL = (p2[0] - ebar) / base["sigma"][0]
        rows[cut][n_p] = dict(p=p, raw=rel(d, d0), dRfit=pR[1] - RD, deps=p2[0] - ebar, dPhiNL=0.5 * r2 @ r2, first=first, base=base, zNL=zNL,
                              rytov=sc.rytov_tm(f, e1, e2, p))
pv = np.array([RD / n for n in nps])
for cut, target, tol in (("period", 1.0, 0.1), ("mid", 2.0, 0.15)):
    slope = np.polyfit(np.log(pv), np.log([rows[cut][n]["raw"] for n in nps]), 1)[0]
    item(f"raw error slope in p, n_p = 10-80, case {cut}: {target:g} +- {tol:g}", abs(slope - target) <= tol, f"{slope:.3f}", f"{target:g} +- {tol:g}")
ratios = [rows["period"][n]["dRfit"] / (-dchi * f * (1 - f) * rows["period"][n]["p"] / 2 / chibar) for n in nps]
le("termination = radius shift, case period: |dR_fit/dR_pred - 1| at p = 0.0125 (n_p = 80) and p = 0.025", max(abs(ratios[3] - 1), abs(ratios[2] - 1)), 0.05)
item("termination ratio dR_fit/dR_pred approaches 1 monotonically as p decreases (n_p = 10..80)", bool(np.all(np.diff(np.abs(1 - np.array(ratios))) < 0)), ", ".join(f"{r:.3f}" for r in ratios), "decreasing |1 - ratio|")
item("mid cut: radius shift is zero at O(p): dR_fit(p)/p -> 0 and dR_fit = O(p^2)", abs(rows["mid"][80]["dRfit"]) / RD * 80 < 0.01 and rows["mid"][80]["dRfit"] / rows["mid"][40]["dRfit"] < 0.3,
     f"dR_fit/p = {rows['mid'][80]['dRfit'] * 80:.4f}, ratio dR(80)/dR(40) = {rows['mid'][80]['dRfit'] / rows['mid'][40]['dRfit']:.3f}", "< 0.01; < 0.3")
for cut in rows:
    r = rows[cut][80]["deps"] / rows[cut][80]["rytov"]
    within(f"homogeneous fit absorbs only part of the Rytov correction, {cut}, p = 0.0125: d_eps/Rytov (PLAN: about 0.78)", r, 0.70, 0.85)
cg = max(abs(rows[c][n]["base"]["c_geom"] - rows[c][n]["base"]["c_direct"]) / rows[c][n]["base"]["c_direct"] for c in rows for n in nps)
le("law identity |c_geom - c_direct|/c_direct (analyze_error at the base point; all cases)", cg, 1e-10)
cg = max(abs(rows["period"][n]["first"]["c_geom"] - rows["period"][n]["first"]["c_direct"]) / rows["period"][n]["first"]["c_direct"] for n in nps)
le("law identity at the foot point, case period", cg, 1e-10)
zr = max(abs(rows[c][n]["first"]["z"] / rows[c][n]["zNL"] - 1) for c in rows for n in nps if rows[c][n]["p"] <= 0.05 + 1e-12)
dr_ = max(abs(rows[c][n]["first"]["dphi"] / rows[c][n]["dPhiNL"] - 1) for c in rows for n in nps if rows[c][n]["p"] <= 0.05 + 1e-12)
le("first order vs noise-free nonlinear fit, p <= 0.05 (base for mid, foot point for period): z (rel; PLAN 5%)", zr, 0.05)
le("first order vs noise-free nonlinear fit, p <= 0.05: DeltaPhi (rel; PLAN 10%)", dr_, 0.10)
info(f"  (mid) z first/NL at p = 0.05: {rows['mid'][20]['first']['z']:.3f}/{rows['mid'][20]['zNL']:.3f}; DeltaPhi {rows['mid'][20]['first']['dphi']:.3f}/{rows['mid'][20]['dPhiNL']:.3f} (PLAN 3.51/3.50, 18.40/18.56)")
j = det.analyze_error(D0, sc.whitened_delta(sc.exact_data(tx, rx, *sc.rings(RD, 10, f, e1, e2, "period")), d0, s))
info(f"  linearizing at the base point, case period, p = 0.1: DeltaPhi_1st = {j['dphi']:.0f} vs nonlinear {rows['period'][10]['dPhiNL']:.0f} (ratio {j['dphi'] / rows['period'][10]['dPhiNL']:.0f}; PLAN: about 100 at the base point)")
within("base-point linearization fails for the period cut: DeltaPhi_1st/DeltaPhi_NL at p = 0.025", rows["period"][40]["base"]["dphi"] / rows["period"][40]["dPhiNL"], 30.0, 300.0)
zs = np.array([rows["mid"][n]["base"]["z"] for n in nps])
dps = np.array([rows["mid"][n]["base"]["dphi"] for n in nps])
sz, sd = np.polyfit(np.log(pv), np.log(np.abs(zs)), 1)[0], np.polyfit(np.log(pv), np.log(dps), 1)[0]
item("scaling, case mid: slope of z_eps 2 +- 0.15 and of DeltaPhi 4 +- 0.3 (n_p = 10-80)", abs(sz - 2) <= 0.15 and abs(sd - 4) <= 0.3, f"{sz:.3f}, {sd:.3f}", "2 +- 0.15, 4 +- 0.3")
pz, _ = sc.loglog_crossing(pv[1:], np.abs(zs[1:]), 3.0)
pf_, _ = sc.loglog_crossing(pv[1:], dps[1:], sc.flag_nats(d0.size))
p3, _ = sc.loglog_crossing(pv[1:], dps[1:], 3.0)
le("flag level: flag_nats(2400) = 3 sqrt(2400) (PLAN 147)", abs(sc.flag_nats(d0.size) - 146.97), 0.01)
item("detectability ordering, case mid: p_3nat < p_z < p_flag", p3 < pz < pf_, f"{p3:.4f} < {pz:.4f} < {pf_:.4f}", "strict")
le("detectability windows vs scratch (0.032, 0.046, 0.084), max rel", max(abs(p3 / 0.032 - 1), abs(pz / 0.046 - 1), abs(pf_ / 0.084 - 1)), 0.05)
within("angle alpha of the normal residual at p = 0.0125, case mid: tends to 18.0 degrees", np.degrees(rows["mid"][80]["base"]["alpha"]), 17.9, 18.1)

# ------------------------------------------------------------------------------------------------ lattice in the testbed, n <= 64
st = ch15_setup(n=64, side=2.5)
fw = Forward(st, tol=1e-10, restart=100, maxiter=30)
txg, rxg = st.antennas.tx, st.antennas.rx
rods_ok, c4, selr, floor_ok = True, 0.0, 0.0, True
rows_l = []
for p in (0.5, 0.33):
    chi = sc.lattice_disk_chi(st.grid, 1.0, p)
    c4chi = np.abs(chi - np.rot90(chi)).max()
    d, u = fw(chi)
    d = np.asarray(d)
    res = float(np.max(fw.residual(chi, u)))
    eb = sc.lattice_disk_eps_bar(1.0, p)
    dm = sc.mie_disk(eb, 1.0, (0, 0), txg, rxg)
    dcyl = np.asarray(fw(cylinder(st.grid, 1.0, eb))[0])
    floor = np.linalg.norm(dcyl - dm)
    sq_ = sc.noise_level(dm, 30.0)
    D = sc.disk_jacobian([eb, 1.0, 0.0, 0.0], txg, rxg, sq_)
    dl = sc.whitened_delta(d, dm, sq_)
    a = det.analyze_error(D, dl)
    z0 = float(a["w_hat"] @ sc.whiten(Q0(d - dm), sq_))
    sector = np.linalg.norm(d - sum(sc.c20_projector(k)(d) for k in (0, 4, 8))) / np.linalg.norm(d)
    rows_l.append((p, c4chi, res, sector, abs(a["z"] - z0) / abs(a["z"]), np.linalg.norm(d - dm) / floor, a["dphi"], sc.c20_fractions(d - dm)[[0, 4, 8]]))
    n_rod = len(sc.lattice_disk_rods(1.0, p))
    area = chi.sum() * st.grid.h ** 2 / (n_rod * np.pi * (0.3 * p) ** 2 * 3.0)
    within(f"painted rod area vs exact N pi (0.3 p)^2 (eps_rod - 1), p = {p:g}, n = 64", area, 0.97, 1.03)
    item(f"solver residual of the lattice forward solve, p = {p:g}", res < 1e-8, f"{res:.1e}", "< 1e-8")
for p, c4chi, res, sector, zrel, ratio, dphi, fr in rows_l:
    item(f"lattice chi is exactly C4-invariant (max |chi - rot90 chi|), p = {p:g}", c4chi == 0.0, f"{c4chi:.1e}", "0")
    le(f"lattice data are C4-invariant: ||(I - Q_0 - Q_4 - Q_8) d||/||d||, p = {p:g}, n = 64", sector, 1e-8)
    le(f"selection rule |z(delta) - z(Q_0 delta)|/|z(delta)|, p = {p:g}", zrel, 1e-10)
    item(f"signal above the grid floor (||delta||/floor > 3), p = {p:g}, n = 64", ratio > 3, f"{ratio:.1f}", "> 3")
    info(f"  p = {p:g}: DeltaPhi = {dphi:.3g}; fractions of the error in Q_0, Q_4, Q_8 = {np.round(fr, 3)}")

# ------------------------------------------------------------------------------------------------ Born grating gain
ks = np.array([1.0, 1.5, 2.0, 3.0])
st35 = ch15_setup(n=64, side=3.5)
g35 = sc.born_grating_gain(Forward(st35, tol=1e-10), ks)
within("Born gain knee: gain(3)/gain(1.5), Gaussian window 3.5 wavelengths wide (n = 64)", g35[3] / g35[1], 0.0, 1e-2)
g25 = sc.born_grating_gain(fw, ks)
r25 = g25[3] / g25[1]
if r25 > 1e-2:
    print(f"DISCREPANCY Born gain gain(3)/gain(1.5) on the plan's grid (side 2.5, n = 64): expected <= 1e-2, got {r25:.4f} "
          f"(n = 96: 0.0177, n = 128: 0.0176; side 3.5: {g35[3] / g35[1]:.4f}). The side-2.5 window cuts the Gaussian at 2.5 sigma (edge value 4%), "
          f"and the hard truncation leaks to |kappa| > 2; a wider window removes it.", flush=True)
item("Born gain: gain(1) > 0.3, falls by 10x between kappa = 1.5 and 3 on both windows", bool(g25[0] > 0.3 and g35[0] > 0.3 and g25[3] / g25[1] < 0.1), f"gain(1) = {g25[0]:.3f}, {g35[0]:.3f}", "> 0.3")
gb = sc.born_grating_gain(Forward(ch15_setup(n=48, side=3.5), tol=1e-10), ks)
le("Born gain is converged in n (n = 48 vs 64, max rel)", np.max(np.abs(gb / g35 - 1)), 0.05)
item("Born gain: scalar kappa returns a float, array an array", isinstance(sc.born_grating_gain(fw, 1.0), float) and sc.born_grating_gain(fw, ks).shape == (4,), "ok", "ok")

dt = time.perf_counter() - T0
item("run time under 60 s", dt < 60, f"{dt:.1f} s", "< 60 s")
if FAILS:
    print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    sys.exit(1)
print("all items passed")
