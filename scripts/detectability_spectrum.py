"""Milestone 8 (DESIGN 15.3 item 3): the detectability spectrum of boundary deformations of the two-disk object.

    python scripts/detectability_spectrum.py run --n 192 [--eps 1.1 2 4] [--gaps 0.05 0.1] [--refit]    # one resumable file per base object in results/detectability/spectrum/n<N>/
    python scripts/detectability_spectrum.py collect [--main-n 192]                                      # spectrum.json, spectrum.md and the figures

Base objects: T1 (two equal disks, r = 0.2, eps_r in {1.1, 2, 4}, gap g in {0.05, 0.1}) on a grid of ``--n`` pixels (side 1.5, book rings; the milestone-7 inference grid is
n = 64), F2 base point p0 = (g, 0.2, eps_r, 0, 0), noise level s of the 30 dB data of that object's own noise-free data on that grid.

Deformations. The normal displacement of the boundary of each disk is v_n(phi) = eps cos(m phi) or eps sin(m phi) with phi the polar angle about the disk's own center
(exact boundary integrals, ``shapes.circle_mode_integrals``). The four functions (right cos, right sin, left cos, left sin) of harmonic m combine into the irreps of D2
(``detectability.irrep_mode_coefficients``; the transformation law is ``detectability.mode_action``, checked here against the permutation of the data):

    A1 (+,+): cos(m phi) on the right disk and (-1)^m cos(m phi) on the left;   B1 (-,+): cos right, -(-1)^m cos left;
    B2 (+,-): sin right, -(-1)^m sin left;                                      A2 (-,-): sin right, +(-1)^m sin left.

Two gap conventions (review of the first version). A deformation changes the deformed object's own edge-to-edge gap by ``gap_change`` = -2 (-1)^m eps (A1). The quantity
``fit`` (the first version's) is the response of the F2 estimate: dz_fit/d eps = w_hat^T tau, how far the estimate moves from the BASE gap. The error of the estimate is
``true``: z_true = z_fit - gap_change / sigma_g per unit amplitude, the first-order analysis of the gap-preserving tangent tau_m - (-1)^m tau_0 (tau_0 the m = 0 tangent, which
lies in the tangent space of F2). All detectability statements (eps_3, c_m, flags) use ``true``; ``fit`` is reported as the response of the estimate. For m = 0, 1 (in-model)
the checks are alpha = 0 and z_true = 0.

Per A1 mode, m = 2..8: dz/d eps in both conventions, alpha, beta, c = (1/2) tan^2 alpha / cos^2 beta = dPhi / z^2, the amplitude eps_3 = 3 / |dz_true/d eps| at which the gap
error is 3 sigma_g, the first-order misfit excess there dPhi(eps_3) = 9 c (unflagged when below the 3 sigma flag 3 sqrt(N) = 147 nats, N = 2400, i.e. when c < 16.3), the
amplitude eps_flag = sqrt(2 * 147) / ||P_perp tau|| at which the flag is reached and the z there, the small-deformation measures eps_3 m / r (boundary slope) and
x = eps_3 (eps_r - 1) / h (contrast displacement per boundary pixel), and ||tau_m|| / ||tau_2|| against the band-limit prediction J_m(2 k0 r) / J_2(2 k0 r) (k0 = 2 pi per
wavelength, so 2 k0 r = 2.51 for r = 0.2). Non-A1 modes: |z| / ||tau|| (first order, round-off expected) and the distance of the tangent from its own irrep.

``--refit``: the nonlinear check at eps_3 for the A1 modes, gap-preserving combination (the deformed object's gap is then exactly g). The data are F(chi0 + eps_3 dchi_gp), i.e.
the contrast displaced to FIRST order in eps (a polar rendering of a deformed boundary is not available to the accuracy the misfit needs: pixel quantization of 1e-3 is
already ~100 nats), and the F2 posterior mode is found from p0 with the real forward model; the shift of g in sigma and the misfit are compared with the linear values 3 and
9 c. It tests the nonlinearity of the forward map in chi (and of the fit), not the second-order rendering of the shape.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure

import parametric_run as prun
from bayes_scatter import detectability as dt
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "detectability"
SPEC = OUT / "spectrum"
SIDE, N_TX, N_RX, TOL = 1.5, 40, 60, 1e-12
N_DATA = N_TX * N_RX
FLAG = 3.0 * np.sqrt(N_DATA)  # 3 sigma misfit flag of milestone 7 (147 nats)
R = 0.2
MODES = tuple(range(2, 9))
K0 = 2.0 * np.pi  # wavenumber in 1 / wavelength
print = functools.partial(print, flush=True)


def label_of(eps_r, gap):
    return f"eps{eps_r:g}_g{gap:g}"


def spec_dir(n):
    return SPEC / f"n{n}"


def bessel_ratio(m):
    """J_m(2 k0 r) / J_2(2 k0 r): the far-field band-limit prediction of ||tau_m|| / ||tau_2||."""
    from scipy.special import jv

    return float(jv(m, 2 * K0 * R) / jv(2, 2 * K0 * R))


def analyze_modes(D, tau, tau0, m, act, irreps=dt.IRREPS):
    """Per-irrep record of the harmonic-m deformations: the tangents ``tau`` (4, 2N) in the order Rc, Rs, Lc, Ls; ``tau0`` is the m = 0 A1 tangent."""
    entry = {}
    for irrep in irreps:
        coef = dt.irrep_mode_coefficients(m, irrep)
        t = coef @ tau
        nt = float(np.linalg.norm(t))
        if nt == 0.0:
            entry[irrep] = {"zero_field": True}
            continue
        fit = dt.analyze_error(D, t, k=0)
        comp = act.project(irrep, t, stacked=True)
        e = {"coefficients_Rc_Rs_Lc_Ls": coef.tolist(), "norm_tau": nt, "z_over_norm": fit["z"] / nt, "irrep_leakage": float(np.linalg.norm(t - comp) / nt),
             "dz_deps": fit["z"], "sin_alpha": float(np.sin(fit["alpha"])), "alpha_deg": float(np.degrees(fit["alpha"])), "dphi_per_eps2": fit["dphi"],
             "dp_over_sigma_per_eps": (fit["dp"] / fit["sigma"]).tolist(), "gap_change_per_eps": dt.gap_change_per_amplitude(m, coef)}
        gp = dt.analyze_error(D, dt.true_gap_tangent(t, tau0, m, coef), k=0)
        e["true"] = {"dz_deps": gp["z"], "sin_alpha": float(np.sin(gp["alpha"])), "alpha_deg": float(np.degrees(gp["alpha"])), "dphi_per_eps2": gp["dphi"],
                     "norm_tangent": gp["norm_tangent"], "norm_perp": gp["norm_perp"]}
        if irrep == "A1":
            e["beta_deg"] = float(np.degrees(fit["beta"]))
            e["c_m"] = fit["c_geom"]  # response of the F2 estimate (base-gap convention)
            e["c_m_direct"] = fit["c_direct"]
            e["true"].update({"beta_deg": float(np.degrees(gp["beta"])), "c": gp["c_geom"], "c_direct": gp["c_direct"]})
        entry[irrep] = e
    return entry


def add_thresholds(a1, m, eps_r, h):
    """eps_3, flags and small-deformation measures of an A1 record, in both conventions (``a1`` itself: the response of the estimate; ``a1["true"]``: the gap error)."""
    for rr, dz, c in ((a1, a1["dz_deps"], a1["c_m"]), (a1["true"], a1["true"]["dz_deps"], a1["true"]["c"])):
        zr = abs(dz)
        eps3 = 3.0 / zr
        rr["eps_z3"] = eps3
        rr["eps_z3_over_r"] = eps3 / R
        rr["dphi_at_z3"] = 9.0 * c
        rr["unflagged_at_z3"] = bool(9.0 * c < FLAG)
        nperp = np.sqrt(2.0 * a1["dphi_per_eps2"])
        rr["eps_flag"] = float(np.sqrt(2.0 * FLAG) / nperp) if nperp > 0 else None
        rr["z_at_flag"] = float(rr["eps_flag"] * zr) if rr["eps_flag"] else None
        rr["small"] = {"eps_slope": float(eps3 * m / R), "x_pixel": float(eps3 * (eps_r - 1.0) / h)}


def run_base(ctx, n, eps_r, gap, refit, force=False):
    path = spec_dir(n) / f"{label_of(eps_r, gap)}.json"
    rec = json.loads(path.read_text()) if path.exists() and not force else None
    if rec is not None and (not refit or all("refit" in rec["modes"][str(m)]["A1"] for m in MODES)):
        print(f"n={n} {path.stem}: done")
        return rec
    setup, fwd, rend, model, act = ctx
    grid = setup.grid
    t0 = time.perf_counter()
    obj = sh.TwoDisks(g=gap, r=R, eps_r=eps_r)
    p0 = np.array([gap, R, eps_r, 0.0, 0.0])
    chi0 = obj.chi(grid)
    d0 = np.asarray(fwd(jnp.asarray(chi0))[0])
    s = noise_std_from_snr(d0, 30.0)
    resid = float(np.linalg.norm(np.asarray(fwd.residual(jnp.asarray(chi0), fwd(jnp.asarray(chi0))[1])).ravel()) / np.linalg.norm(np.asarray(fwd.u_inc).ravel()))
    s_m7 = float(np.load(ROOT / "results" / "parametric" / "data" / f"T1g{gap:g}.npz")["s"]) if abs(eps_r - 2.0) < 1e-12 else None
    D, chi_p = dt.whitened_jacobian(model, fwd, rend, p0, s)
    w_hat, sigma_g = dt.gap_direction(D, 0)
    rec = {"n": n, "h": grid.h, "eps_r": eps_r, "gap": gap, "noise_std": s, "s_m7": s_m7, "s_ratio_to_m7": None if s_m7 is None else s / s_m7, "relative_residual_of_base_solve": resid,
           "sigma_g": sigma_g, "sigma_all": np.sqrt(np.diag(np.linalg.inv(D.T @ D))).tolist(), "tol": TOL, "flag_nats": FLAG,
           "w_hat_a1_fraction": act.fractions(w_hat)["A1"]["frac_norm2"], "modes": {}}
    dchi0 = dt.mode_chi_derivatives(grid, obj, 0)
    tau0_all = dt.stack_real(dt.data_tangents(fwd, chi0, dchi0), s)
    coef0 = dt.irrep_mode_coefficients(0, "A1")
    tau0 = coef0 @ tau0_all  # m = 0, A1: d_r - 2 d_g (gap change -2 per unit amplitude)
    for m in (0, 1) + MODES:
        dchi = dt.mode_chi_derivatives(grid, obj, m)
        tau = dt.stack_real(dt.data_tangents(fwd, chi0, dchi), s)  # (4, 2N)
        eq = 0.0  # equivariance: P_g tau(c) = tau(M_g c) for the four basis fields
        for g in ("sx", "sy"):
            M = dt.mode_action(m, g)
            for k in range(4):
                lhs = act.apply_stacked(g, tau[k])
                rhs = M[:, k] @ tau
                eq = max(eq, float(np.linalg.norm(lhs - rhs) / max(np.linalg.norm(tau[k]), 1e-300)))
        entry = analyze_modes(D, tau, tau0, m, act)
        entry["equivariance_rel_error"] = eq
        rec["modes"][str(m)] = entry
        a1 = entry["A1"]
        if m >= 2:
            add_thresholds(a1, m, eps_r, grid.h)
        else:  # in-model checks: z_true = 0 against the base-gap response -+2 / sigma_g
            a1["z_true_over_z_fit"] = a1["true"]["dz_deps"] / a1["dz_deps"]
        if refit and m >= 2:
            eps3 = a1["true"]["eps_z3"] * np.sign(a1["true"]["dz_deps"])  # signed so that z_true = +3
            coef = dt.irrep_mode_coefficients(m, "A1")
            dchi_gp = np.einsum("i,ijk->jk", coef, dchi) + 0.5 * dt.gap_change_per_amplitude(m, coef) * np.einsum("i,ijk->jk", coef0, dchi0)
            d_def = np.asarray(fwd(jnp.asarray(chi0 + eps3 * dchi_gp))[0])
            post = sp.ShapePosterior(model, fwd, d_def, s, rend)
            p_star, phi, ok = dt.noise_free_fit(post, p0)
            shift = (p_star[0] - gap) / sigma_g
            tr = a1["true"]
            a1["refit"] = {"eps": float(eps3), "g_star": float(p_star[0]), "shift_sd": float(shift), "z_linear": float(eps3 * tr["dz_deps"]), "phi": phi,
                           "dphi_linear": float(eps3**2 * tr["dphi_per_eps2"]), "c_NL": float(phi / shift**2) if shift != 0 else None, "converged": ok, "p_star": p_star.tolist()}
            print(f"n={n} {label_of(eps_r, gap)} m={m}: refit shift {shift:+.3f} sd (linear {eps3 * tr['dz_deps']:+.3f}), phi {phi:.4g} (linear {eps3**2 * tr['dphi_per_eps2']:.4g})")
    n2 = rec["modes"]["2"]["A1"]["norm_tau"]
    for m in MODES:
        a1 = rec["modes"][str(m)]["A1"]
        a1["norm_ratio_to_m2"] = a1["norm_tau"] / n2
        a1["bessel_ratio"] = abs(bessel_ratio(m))
    rec["time_s"] = time.perf_counter() - t0
    rec["thread_env"] = prun.thread_env()
    prun.write_json(path, rec)
    print(f"n={n} {path.stem}: sigma_g {sigma_g:.3e}, s {s:.3e}, {rec['time_s']:.0f} s")
    return rec


# ----------------------------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------------------------

COLORS = {1.1: "#1f77b4", 2.0: "#d95f02", 4.0: "#4d4d4d"}
MARKS = {1.1: "o", 2.0: "s", 4.0: "^"}


def load_all():
    out = {}
    for d in sorted(SPEC.glob("n*")):
        if d.is_dir():
            out[int(d.name[1:])] = {(r["eps_r"], r["gap"]): r for r in (json.loads(p.read_text()) for p in sorted(d.glob("eps*_g*.json")))}
    return out


def collect(main_n=None):
    allr = load_all()
    main_n = max(allr) if main_n is None else main_n
    recs = sorted(allr[main_n].values(), key=lambda r: (r["gap"], r["eps_r"]))
    L = [f"# Detectability spectrum of A1 boundary deformations (30 dB), n = {main_n}\n",
         "Gap-error convention (z_true: the error of the F2 gap estimate against the deformed object's own gap; the gap-preserving tangent). eps_3: amplitude (wavelengths) at which |z_true| = 3; "
         f"dPhi(eps_3) = 9 c against the {FLAG:.0f}-nat flag; eps_flag: amplitude at which dPhi = {FLAG:.0f} and z_true there; eps m / r: boundary slope at eps_3; x = eps_3 (eps_r - 1) / h: contrast "
         "displacement per boundary pixel (the nonlinear refit of the first version agreed only for x <= 0.3).\n",
         "| eps_r | g | m | dz_true/deps | c | alpha (deg) | beta (deg) | eps_3 | dPhi(eps_3) | unflagged | eps_flag | z_true at flag | eps_3 m / r | x |", "|" + "---|" * 14]
    for r in recs:
        for m in MODES:
            a = r["modes"][str(m)]["A1"]
            t = a["true"]
            L.append(f"| {r['eps_r']:g} | {r['gap']:g} | {m} | {t['dz_deps']:.5g} | {t['c']:.4g} | {t['alpha_deg']:.2f} | {t['beta_deg']:.1f} | {t['eps_z3']:.4g} | {t['dphi_at_z3']:.4g} | {t['unflagged_at_z3']} | "
                     f"{t['eps_flag']:.4g} | {t['z_at_flag']:.4g} | {t['small']['eps_slope']:.3g} | {t['small']['x_pixel']:.3g} |")
    refit_ns = [n for n in sorted(allr) if any("refit" in r["modes"]["2"]["A1"] for r in allr[n].values())]
    for n in refit_ns:
        L.append(f"\n## Nonlinear refit at eps_3 (n = {n}; data F(chi0 + eps_3 dchi_gp), z_true = +3 by construction)\n")
        L.append("The refit shift should be 3 sd and the misfit 9 c; it tests the nonlinearity of the forward map in chi (and of the fit), not the second-order rendering of the shape.\n")
        L.append("| eps_r | g | m | eps_3 | refit shift (sd) | refit phi | linear dPhi = 9 c | phi / linear | c_NL = phi / shift^2 | converged |\n|" + "---|" * 10)
        for r in sorted(allr[n].values(), key=lambda r: (r["gap"], r["eps_r"])):
            for m in MODES:
                a = r["modes"][str(m)]["A1"]
                rf = a["refit"]
                L.append(f"| {r['eps_r']:g} | {r['gap']:g} | {m} | {rf['eps']:.4g} | {rf['shift_sd']:.4f} | {rf['phi']:.5g} | {rf['dphi_linear']:.5g} | {rf['phi'] / rf['dphi_linear']:.3f} | {rf['c_NL']:.4g} | {rf['converged']} |")
    L.append("\n## Response of the F2 gap estimate to the base gap (first-version quantity, NOT the error of the estimate)\n")
    L.append("dz_fit/deps = w_hat^T tau: how far the estimate moves from the BASE gap per unit amplitude; the deformation itself changes the gap by gap_change = -2 (-1)^m per unit amplitude, and "
             "dz_true/deps = dz_fit/deps - gap_change / sigma_g.\n")
    L.append("| eps_r | g | m | sigma_g | dz_fit/deps | gap_change / sigma_g | dz_true/deps | c_fit | c_true | ||tau_m|| / ||tau_2|| | J_m / J_2 (2 k0 r = 2.51) |\n|" + "---|" * 11)
    for r in recs:
        for m in MODES:
            a = r["modes"][str(m)]["A1"]
            L.append(f"| {r['eps_r']:g} | {r['gap']:g} | {m} | {r['sigma_g']:.4e} | {a['dz_deps']:.5g} | {a['gap_change_per_eps'] / r['sigma_g']:.5g} | {a['true']['dz_deps']:.5g} | {a['c_m']:.4g} | "
                     f"{a['true']['c']:.4g} | {a['norm_ratio_to_m2']:.4g} | {a['bessel_ratio']:.4g} |")
    ns = sorted(allr)
    if len(ns) > 1:
        L.append("\n## Grid convergence (same code, side 1.5; entries n = " + " / ".join(map(str, ns)) + ")\n")
        L.append("| eps_r | g | m | dz_fit/deps | dz_true/deps | c_true | c_fit | ||tau_m|| / ||tau_2|| | J_m / J_2 |\n|" + "---|" * 9)
        keys = sorted(set.intersection(*[set(allr[n]) for n in ns]), key=lambda k: (k[1], k[0]))
        for key in keys:
            for m in MODES:
                a = [allr[n][key]["modes"][str(m)]["A1"] for n in ns]
                L.append(f"| {key[0]:g} | {key[1]:g} | {m} | " + " / ".join(f"{x['dz_deps']:.4g}" for x in a) + " | " + " / ".join(f"{x['true']['dz_deps']:.5g}" for x in a) + " | "
                         + " / ".join(f"{x['true']['c']:.3g}" for x in a) + " | " + " / ".join(f"{x['c_m']:.3g}" for x in a) + " | " + " / ".join(f"{x['norm_ratio_to_m2']:.3g}" for x in a)
                         + f" | {a[0]['bessel_ratio']:.3g} |")
    L.append("\n## In-model checks (m = 0, 1) and non-A1 modes\n")
    L.append("| eps_r | g | m=0 sin(alpha) | m=0 z_true / z_fit | m=1 A1 sin(alpha) | m=1 A1 z_true / z_fit | m=0 dz_fit/deps x sigma_g / -2 | m=1 A1 dz_fit/deps x sigma_g / 2 | m=1 B1 sin(alpha) | m=1 B2 sin(alpha) | max non-A1 abs z / norm tau (m=2..8) | max irrep leakage | max equivariance error |\n|" + "---|" * 13)
    for r in recs:
        mo = r["modes"]
        na = max(abs(mo[str(m)][i]["z_over_norm"]) for m in MODES for i in ("B1", "B2", "A2"))
        leak = max(mo[str(m)][i]["irrep_leakage"] for m in MODES for i in dt.IRREPS)
        eq = max(mo[str(m)]["equivariance_rel_error"] for m in (0, 1) + MODES)
        sg = r["sigma_g"]
        L.append(f"| {r['eps_r']:g} | {r['gap']:g} | {mo['0']['A1']['sin_alpha']:.2e} | {mo['0']['A1']['z_true_over_z_fit']:.2e} | {mo['1']['A1']['sin_alpha']:.2e} | {mo['1']['A1']['z_true_over_z_fit']:.2e} | "
                 f"{mo['0']['A1']['dz_deps'] * sg / -2:.8f} | {mo['1']['A1']['dz_deps'] * sg / 2:.8f} | {mo['1']['B1']['sin_alpha']:.2e} | {mo['1']['B2']['sin_alpha']:.2e} | {na:.2e} | {leak:.2e} | {eq:.2e} |")
    (OUT / "spectrum.md").write_text("\n".join(L) + "\n")
    prun.write_json(OUT / "spectrum.json", {"main_n": main_n, "grids": {str(n): list(v.values()) for n, v in allr.items()}, "flag_nats": FLAG, "modes": list(MODES)})
    fig = Figure(figsize=(10, 7.5), constrained_layout=True)
    axes = fig.subplots(2, 2, sharex=True)
    for col, gap in enumerate(sorted({r["gap"] for r in recs})):
        for r in (x for x in recs if x["gap"] == gap):
            e = r["eps_r"]
            ms = list(MODES)
            axes[0, col].semilogy(ms, [r["modes"][str(m)]["A1"]["true"]["eps_z3"] for m in ms], marker=MARKS[e], color=COLORS[e], lw=1.8, ms=7, label=f"eps_r = {e:g}")
            axes[1, col].semilogy(ms, [r["modes"][str(m)]["A1"]["true"]["c"] for m in ms], marker=MARKS[e], color=COLORS[e], lw=1.8, ms=7, label=f"eps_r = {e:g}")
        axes[1, col].axhline(FLAG / 9.0, color="#888888", ls="--", lw=1)
        axes[1, col].text(5.0, FLAG / 9.0 * 1.3, "c = 16.3 (flag at z = 3)", fontsize=8, color="#444444", ha="center")
        axes[0, col].set_title(f"gap g = {gap:g} wavelengths, n = {main_n}")
        axes[1, col].set_xlabel("harmonic m of the A1 boundary deformation")
    axes[0, 0].set_ylabel("eps_3: gap error = 3 sd (wavelengths)")
    axes[1, 0].set_ylabel("c = dPhi / z_true^2")
    axes[0, 0].legend(frameon=False)
    for ax in axes.ravel():
        ax.grid(True, which="major", color="#e0e0e0", lw=0.6)
        ax.set_xticks(list(MODES))
    fig.savefig(OUT / "spectrum.png", dpi=130)
    if len(ns) > 1 and (2.0, 0.1) in allr[ns[0]]:
        fig2 = Figure(figsize=(11, 4), constrained_layout=True)
        ax = fig2.subplots(1, 3)
        marks = {n: m for n, m in zip(ns, ("o", "s", "^", "v"))}
        ms = list(MODES)
        for n in ns:
            a = [allr[n][(2.0, 0.1)]["modes"][str(m)]["A1"] for m in ms]
            ax[0].semilogy(ms, [abs(x["dz_deps"]) for x in a], marker=marks[n], label=f"n = {n}")
            ax[1].semilogy(ms, [x["norm_ratio_to_m2"] for x in a], marker=marks[n], label=f"n = {n}")
            ax[2].semilogy(ms, [abs(x["true"]["dz_deps"]) for x in a], marker=marks[n], label=f"n = {n}")
        ax[1].semilogy(ms, [abs(bessel_ratio(m)) for m in ms], "k--", label="J_m / J_2 at 2 k0 r")
        ax[0].set_title("|dz_fit/deps| (response of the estimate)")
        ax[1].set_title("||tau_m|| / ||tau_2||")
        ax[2].set_title("|dz_true/deps| (gap error)")
        for a_ in ax:
            a_.set_xlabel("m")
            a_.grid(True, color="#e0e0e0", lw=0.6)
            a_.legend(frameon=False, fontsize=8)
        fig2.suptitle("eps_r = 2, g = 0.1: grid convergence (side 1.5)")
        fig2.savefig(OUT / "spectrum_convergence.png", dpi=130)
    print(f"wrote spectrum.json, spectrum.md, spectrum.png ({len(recs)} base objects, main n = {main_n}; grids {ns})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--eps", type=float, nargs="+", default=[1.1, 2.0, 4.0])
    r.add_argument("--gaps", type=float, nargs="+", default=[0.05, 0.1])
    r.add_argument("--refit", action="store_true")
    r.add_argument("--force", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--main-n", type=int, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args.main_n)
        return
    setup = ch15_setup(n=args.n, side=SIDE, n_tx=N_TX, n_rx=N_RX)
    from bayes_scatter import ellipses as el

    ctx = (setup, Forward(setup, tol=TOL), el.EllipseRenderer(setup.grid), sm.F2(), dt.D2Action(setup.antennas))
    for g in args.gaps:
        for e in args.eps:
            run_base(ctx, args.n, e, g, args.refit, args.force)


if __name__ == "__main__":
    main()
