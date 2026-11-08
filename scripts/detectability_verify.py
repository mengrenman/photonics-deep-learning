"""Milestone 8 (DESIGN 15.3 items 1 and 2): does the first-order law c = (1/2) tan^2(alpha) / cos^2(beta) reproduce the c measured in milestone 7?

    python scripts/detectability_verify.py run --cases all            # every case (resumable: a finished case is skipped); or --cases E0.01g0.1,T3g0.05,grid256g0.05
    python scripts/detectability_verify.py collect                    # results/detectability/verify.json and verify.md from the case files

Cases (labels): the T2 ellipticity ladder ``E<level>g<gap>`` (level 0.005, 0.01, 0.02, 0.05, 0.15; level 0.15 is T2 itself) at gaps 0.05 and 0.1, ``T3g<gap>``, ``T5``, and the
grid error of the data of T1 on the n = 128 and n = 256 grids against the n = 64 model (``grid128g<gap>``, ``grid256g<gap>``).

Setting (the milestone-7 settings): book rings (40 transmitters, 60 receivers), side 1.5, inference grid n = 64, F2 (two equal circles, p = (g, r, eps_r, cx, cy)), base point
p0 = (g, 0.2, 2, 0, 0), noise level s of the 30 dB data of the truth (the cached ``s`` of ``results/parametric/{data,ellipticity/data}``), noise-free data. The noise-free
data of a truth are the cached ``d_clean`` of milestone 7 (n = 128 grid) and are never regenerated, except for the n = 256 grid error (computed here, cached in
``results/detectability/data``).

Variants of the model error delta (whitened, ``detectability.stack_real``) of one case:

* ``total``  : delta = y(d_clean on the n = 128 grid) - y(F(p0) on n = 64). This is the data of milestone 7, so the grid error is in it.
* ``pure``   : delta = y(truth rendered on the n = 64 grid) - y(F(p0) on n = 64) (T2 ladder, T3, T5): the model error without the grid error ("inverse crime" data).
* ``ref0``   : delta = y(d_clean(level) on n = 128) - y(d_clean(level 0) on n = 128) (ladder only): the shape error as seen from the level-0 data of the same grid, which is what
               the "c vs level 0" column of milestone 7 measures (the shift of the fit from the level-0 fit of the same noise).
* ``total_p0mean`` (T3 only): the ``total`` error against the base point eps_r = 2.2 (the mean of the two contrasts) instead of 2.

Per variant: the first-order quantities at p0 (``detectability.analyze_error``: z = w_hat^T delta in the convention (fitted - base) / sigma, dPhi, alpha, beta, c_geom, the
bias vector dp / sigma), the D2 decomposition of delta (fraction of ||delta||^2 and of z per irrep), the same quantities with D evaluated at the noise-free nonlinear MAP
(``at_map``: tangent space at the fit, delta still measured from p0), and the nonlinear noise-free refit (MAP of the F2 posterior, prior included as in milestone 7; starts: p0 and,
when it exists, the noisy milestone-7 MAP) with shift_sd = (g* - g) / sigma_g(p0), phi (= dPhi, the data carry no noise) and c_NL = phi / shift_sd^2. The measured side
(``measured``) is read from the milestone-7 files: dPhi = Phi_MAP - misfit of the noise realization itself, z_true = (g_true - g_hat) / sd (opposite sign to z here), c = dPhi / z^2,
and for the ladder c vs level 0 = dPhi / dz^2 with dz the shift from the level-0 fit of the same noise; for the grid error c = Phi_MAP / shift_sd^2 (noise-free fits).
Review additions (milestone 8 review). The reference for the law is the noise-free nonlinear fit (c_NL); ``measured`` is ONE noise draw (key 0) and carries the noise
interaction terms, which ``noise_decomposition`` models: z_fit = z_NL + w_hat*^T e and dPhi = Phi_NL + r*^T e - (1/2) ||P_T* e||^2 with e the whitened noise of the draw, r* the
whitened noise-free residual at the noise-free MAP p* and T* the tangent space there (DESIGN's dPhi is the expected excess over a correct model; the milestone-7 dPhi is
Phi_MAP - Phi(noise), lower by (1/2) ||P_T e||^2, mean k / 2). ``sigma_full``: the posterior sd of g from the finite-difference (observed-information) Hessian of the Laplace fit,
against the Gauss-Newton one that milestone 7 used; it differs when the residual curvature is large (T3). ``foot``: the base point is the profile foot point (g = g_true, the
nuisances at their least-squares values). ``second_order``: the second-order bias (``detectability.second_order_bias``: parameter-effects and intrinsic parts) against the
nonlinear shift, and the eigenvalues of the residual-curvature operator at the nonlinear MAP with the predicted log Z(FD) - log Z(GN).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import parametric_run as prun
import parametric_sweeps as psw
from bayes_scatter import detectability as dt
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup

ROOT = Path(__file__).resolve().parents[1]
M7 = ROOT / "results" / "parametric"
OUT = ROOT / "results" / "detectability"
N, SIDE, N_TX, N_RX, TOL = 64, 1.5, 40, 60, 1e-11
LEVELS = (0.005, 0.01, 0.02, 0.05, 0.15)
GAPS = (0.05, 0.1)
IRREP_ORDER = dt.IRREPS
print = functools.partial(print, flush=True)


def all_labels():
    labs = [f"E{lev:g}g{g:g}" for g in GAPS for lev in LEVELS]
    labs += [f"T3g{g:g}" for g in GAPS] + ["T5"]
    labs += [f"grid{n}g{g:g}" for g in GAPS for n in (128, 256)]
    return labs


def parse_label(label):
    if label == "T5":
        return {"kind": "T5", "gap": 0.1}
    if label.startswith("T3g"):
        return {"kind": "T3", "gap": float(label[3:])}
    if label.startswith("grid"):
        n, gap = label[4:].split("g")
        return {"kind": "grid", "gap": float(gap), "data_n": int(n)}
    if label.startswith("E"):
        lev, gap = label[1:].split("g")
        return {"kind": "ladder", "gap": float(gap), "level": float(lev)}
    raise ValueError(f"unknown case {label!r}")


class Context:
    """The n = 64 setting shared by all cases of one process."""

    def __init__(self):
        self.setup = ch15_setup(n=N, side=SIDE, n_tx=N_TX, n_rx=N_RX)
        self.grid = self.setup.grid
        self.forward = Forward(self.setup, tol=TOL)
        self.renderer = el.EllipseRenderer(self.grid)
        self.model = sm.F2()
        self.act = dt.D2Action(self.setup.antennas)

    def data64(self, chi):
        return np.asarray(self.forward(jnp.asarray(chi))[0])

    def chi_p(self, p):
        return np.asarray(self.renderer.chi(*self.model.scene(jnp.asarray(p, dtype=jnp.float64))))


def p0_of(gap, eps_r=2.0):
    return np.array([gap, 0.2, eps_r, 0.0, 0.0])


# ----------------------------------------------------------------------------------------------
# Data of a case
# ----------------------------------------------------------------------------------------------


def case_data(ctx, label, info):
    """``(s, variants, measured, d_obs)``: variants a dict name -> (noise-free data of the truth (Nt, Nr), base p0, None); the measured record; the noisy data of the
    milestone-7 draw (None for the noise-free grid cases). Reads the milestone-7 caches."""
    kind, gap = info["kind"], info["gap"]
    base = p0_of(gap)
    d_base = ctx.data64(ctx.chi_p(base))
    variants = {}
    d_obs = None
    if kind == "ladder":
        lev = info["level"]
        z = np.load(M7 / "ellipticity" / "data" / f"E{lev:g}g{gap:g}.npz")
        z0 = np.load(M7 / "ellipticity" / "data" / f"E0g{gap:g}.npz")
        s = float(z["s"])
        d128, d128_0 = np.asarray(z["d_clean"]), np.asarray(z0["d_clean"])
        scene = psw.make_scene(gap, lev)
        variants["total"] = (d128, base, None)
        variants["pure"] = (ctx.data64(scene.chi(ctx.grid)), base, None)
        variants["ref0"] = (d128 - d128_0 + d_base, base, None)  # d_base + (d128(level) - d128(level 0)): the shape error with the grid error removed
        measured = measured_ladder(gap, lev, z)
        d_obs = np.asarray(z["d_obs"])
    elif kind == "T3":
        z = np.load(M7 / "data" / f"T3g{gap:g}.npz")
        s = float(z["s"])
        d128 = np.asarray(z["d_clean"])
        truth = el.make_truth("T3", g=gap)
        variants["total"] = (d128, base, None)
        variants["pure"] = (ctx.data64(truth.chi(ctx.grid)), base, None)
        variants["total_p0mean"] = (d128, p0_of(gap, 2.2), None)
        measured = measured_snr(f"T3g{gap:g}")
        d_obs = np.asarray(z["d_obs"])
    elif kind == "T5":
        z = np.load(M7 / "data" / "T5.npz")
        s = float(z["s"])
        d128 = np.asarray(z["d_clean"])
        truth = el.make_truth("T5", g=0.1)
        variants["total"] = (d128, base, None)
        variants["pure"] = (ctx.data64(truth.chi(ctx.grid)), base, None)
        measured = measured_snr("T5")
        d_obs = np.asarray(z["d_obs"])
    elif kind == "grid":
        n_data = info["data_n"]
        z = np.load(M7 / "data" / f"T1g{gap:g}.npz")
        s = float(z["s"])
        if n_data == 128:
            d_n = np.asarray(z["d_clean"])
        else:
            cache = OUT / "data" / f"grid{n_data}g{gap:g}.npz"
            if cache.exists():
                d_n = np.load(cache)["d_clean"]
            else:
                big = Setup(Grid(n_data, SIDE), ctx.setup.antennas)
                t0 = time.perf_counter()
                d_n = np.asarray(Forward(big, tol=1e-10)(jnp.asarray(sh.TwoDisks(g=gap, r=0.2, eps_r=2.0).chi(big.grid)))[0])
                prun.save_npz(cache, d_clean=d_n, s=s)
                print(f"{label}: noise-free data on n = {n_data} computed in {time.perf_counter() - t0:.0f} s")
        variants["total"] = (d_n, base, None)
        measured = measured_grid(gap, n_data)
    else:
        raise ValueError(kind)
    return s, variants, measured, d_obs


def noise_only_phi(z):
    return float(np.sum(np.abs(np.asarray(z["d_obs"]) - np.asarray(z["d_clean"])) ** 2) / float(z["s"]) ** 2)


def measured_ladder(gap, lev, z):
    r = json.loads((M7 / "ellipticity" / f"E{lev:g}g{gap:g}.json").read_text())
    r0 = json.loads((M7 / "ellipticity" / f"E0g{gap:g}.json").read_text())
    dphi = r["phi_map"] - noise_only_phi(z)
    z_fit = -r["z_true"]  # (fitted - true) / sd, the sign of this module
    dz0 = (r["g_hat"] - r0["g_hat"]) / r["g_sd"]
    return {"source": f"results/parametric/ellipticity/E{lev:g}g{gap:g}.json (and E0g{gap:g}.json), data/E{lev:g}g{gap:g}.npz", "noise_seed": 0, "snr_db": 30,
            "phi_map": r["phi_map"], "dphi": dphi, "z_fit": z_fit, "g_hat": r["g_hat"], "g_sd": r["g_sd"], "c": dphi / z_fit**2, "z_vs0": dz0, "c_vs0": dphi / dz0**2,
            "p_hat": r["p_hat"]}


def measured_snr(label):
    rows = json.loads((M7 / "snr" / "snr_summary.json").read_text())["units"]
    r = next(u for u in rows if u["label"] == label and u["model"] == "F2" and u["snr"] == 30.0 and u["seed"] == 0)
    z_fit = -r["z_true"]
    return {"source": "results/parametric/snr/snr_summary.json (units, snr 30, seed 0, F2)", "noise_seed": 0, "snr_db": 30, "phi_map": r["phi_map"], "dphi": r["delta_phi"], "z_fit": z_fit,
            "g_hat": r["g_hat"], "g_sd": r["g_sd"], "c": r["delta_phi"] / z_fit**2, "p_hat": r["p_hat"]}


def measured_grid(gap, n_data):
    path = M7 / "discretization" / f"disc_g{gap:g}_n{n_data}.json"
    r = json.loads(path.read_text())
    return {"source": f"results/parametric/discretization/disc_g{gap:g}_n{n_data}.json (noise-free fit)", "noise_seed": None, "snr_db": 30, "phi_map": r["phi_map"], "dphi": r["phi_map"],
            "z_fit": r["shift_sd"], "g_hat": r["g_star"], "g_sd": r["laplace_sd"], "c": r["phi_map"] / r["shift_sd"] ** 2, "p_hat": None}


# ----------------------------------------------------------------------------------------------
# One variant
# ----------------------------------------------------------------------------------------------


def m7_map_start(label, info):
    """The physical F2 MAP of the milestone-7 fit of the same data, when it exists (T2 at 15 %, T3, T5), else None."""
    name = {"T3": f"T3g{info['gap']:g}_F2", "T5": "T5_F2"}.get(info["kind"])
    if info["kind"] == "ladder" and abs(info["level"] - 0.15) < 1e-12:
        name = f"T2g{info['gap']:g}_F2"
    if name is None or not (M7 / name / "map.npz").exists():
        return None
    u = np.load(M7 / name / "map.npz")["u_hat"]
    return np.asarray(sm.F2().p_from_u(jnp.asarray(u)))


def stack_y(d, s):
    """Whitened real-stacked vector of complex data (Nt, Nr) (``detectability.stack_real``)."""
    return dt.stack_real(d, s)


def laplace_sd_g(post, p_hat):
    """``(sd_gn, sd_full)``: the posterior sd of the gap (physical units, delta method through g = G_MAX sigmoid(u_0) as in milestone 7) from the Laplace fit of ``post`` at the
    physical point ``p_hat`` with the Gauss-Newton Hessian and with the finite-difference (observed information) Hessian; ``sd_full`` is None if that Hessian is not positive definite."""
    u = np.asarray(post.model.u_from_p(jnp.asarray(p_hat, dtype=jnp.float64)))
    g = float(p_hat[0])
    jac = g * (1.0 - g / sm.G_MAX)
    sd_gn = float(sp.laplace_fit(post, u, hessian="gn").sd()[0]) * jac
    try:
        sd_fd = float(sp.laplace_fit(post, u, hessian="fd").sd()[0]) * jac
    except ValueError:
        sd_fd = None
    return sd_gn, sd_fd


def analyze_variant(ctx, info, label, s, d_truth, p0, nonlinear, start_extra, second=False, curv_map=False):
    gap = info["gap"]
    D, chi0 = dt.whitened_jacobian(ctx.model, ctx.forward, ctx.renderer, p0, s)
    d_model = ctx.data64(chi0)
    delta = dt.stack_real(d_truth - d_model, s)
    res = dt.analyze_error(D, delta, k=0)
    sig = res["sigma"]
    out = {"p0": p0.tolist(), "sigma_g": res["sigma_k"], "sigma_all": sig.tolist(), "model_error_energy_nats": res["norm2_half"],
           "linear": {"z": res["z"], "dphi": res["dphi"], "alpha_deg": float(np.degrees(res["alpha"])), "beta_deg": float(np.degrees(res["beta"])), "c_geom": res["c_geom"],
                      "c_direct": res["c_direct"], "z_check_dp_over_sigma": float(res["dp"][0] / sig[0]), "dp": res["dp"].tolist(), "dp_over_sigma": (res["dp"] / sig).tolist(),
                      "param_names": list(ctx.model.param_names), "norm_tangent": res["norm_tangent"], "norm_perp": res["norm_perp"]}}
    # D2 decomposition of the error: fractions of ||delta||^2 and of z
    fr = ctx.act.fractions(delta, res["w_hat"])
    out["d2"] = {k: v for k, v in fr.items()}
    out["d2"]["z_sum_check"] = float(sum(v["z"] for v in fr.values()))
    out["d2"]["w_hat_a1_fraction"] = ctx.act.fractions(res["w_hat"])["A1"]["frac_norm2"]
    if second:
        H, asym = dt.curvature_tensor(ctx.model, ctx.forward, ctx.renderer, p0, s)
        so = dt.second_order_bias(D, H, delta)
        out["second_order"] = {"dg_linear": float(so["p1"][0]), "dg_p2_tangential": float(so["p2_tangential"][0]), "dg_p2_normal": float(so["p2_normal"][0]), "dg_second_order": float(so["p"][0]),
                               "z_second_order": float(so["p"][0] / sig[0]), "p1": so["p1"].tolist(), "p2": so["p2"].tolist(), "curvature_asymmetry": asym, "steps": dt.default_steps(p0).tolist()}
        lam0, _ = dt.residual_curvature_operator(D, H, delta - D @ so["p1"])  # curvature operator of the first-order residual at p0 (diagnostic of the series)
        out["second_order"]["lambda_p0_first_order_residual"] = lam0.tolist()
    if nonlinear:
        post = sp.ShapePosterior(ctx.model, ctx.forward, d_truth, s, ctx.renderer)
        starts = [("p0", p0)]
        if start_extra is not None:
            starts.append(("milestone-7 MAP", start_extra))
        best = None
        fits = []
        for name, p_start in starts:
            t0 = time.perf_counter()
            p_hat, phi, ok = dt.noise_free_fit(post, p_start)
            fits.append({"start": name, "phi": phi, "g_star": float(p_hat[0]), "converged": ok, "time_s": time.perf_counter() - t0})
            if best is None or phi < best[1]:
                best = (p_hat, phi, ok, name)
        p_star, phi, ok, name = best
        shift_sd = (p_star[0] - gap) / sig[0]
        D_star, _ = dt.whitened_jacobian(ctx.model, ctx.forward, ctx.renderer, p_star, s)
        res_star = dt.analyze_error(D_star, delta, k=0)
        sd_gn, sd_fd = laplace_sd_g(post, p_star)
        out["nonlinear"] = {"sd_g_gn": sd_gn, "sd_g_full": sd_fd, "shift_sd_full": float((p_star[0] - gap) / sd_fd) if sd_fd else None,
                            "c_NL_full": float(phi / ((p_star[0] - gap) / sd_fd) ** 2) if sd_fd and p_star[0] != gap else None, "dg_NL": float(p_star[0] - gap),
                            "p_star": p_star.tolist(), "start_used": name, "fits": fits, "converged": ok, "phi": phi, "shift_sd": float(shift_sd),
                            "shift_sd_at_map_sigma": float((p_star[0] - gap) / res_star["sigma_k"]), "c_NL": float(phi / shift_sd**2) if shift_sd != 0 else float("inf"),
                            "dp_over_sigma": ((p_star - p0) / sig).tolist(),
                            "z_linear_over_NL": float(res["z"] / shift_sd) if shift_sd != 0 else None, "dphi_linear_over_NL": float(res["dphi"] / phi) if phi > 0 else None}
        out["at_map"] = {"sigma_g": res_star["sigma_k"], "z": res_star["z"], "dphi": res_star["dphi"], "alpha_deg": float(np.degrees(res_star["alpha"])),
                         "beta_deg": float(np.degrees(res_star["beta"])), "c_geom": res_star["c_geom"],
                         "note": "tangent space (D, w_hat, sigma) at the nonlinear MAP; delta still the error measured from p0"}
        if curv_map:
            H_star, asym_star = dt.curvature_tensor(ctx.model, ctx.forward, ctx.renderer, p_star, s)
            r_star = stack_y(d_truth, s) - dt.stack_real(ctx.data64(ctx.chi_p(p_star)), s)
            lam, _ = dt.residual_curvature_operator(D_star, H_star, r_star)
            out["residual_curvature_at_map"] = {"eigenvalues": lam.tolist(), "lambda_max": float(lam[0]), "lambda_min": float(lam[-1]), "predicted_dlogZ_fd_minus_gn": dt.log_evidence_gap(lam),
                                                "residual_norm2_half": 0.5 * float(r_star @ r_star), "curvature_asymmetry": asym_star, "coordinates": "physical parameters p = (g, r, eps_r, cx, cy)"}
    return out


def noise_decomposition(ctx, s, d_clean, d_obs, p_star, gap, measured):
    """Model of the one-draw ``measured`` numbers from the noise-free nonlinear fit (module docstring): with e the whitened noise of the draw, z_fit = z_NL + w_hat*^T e and
    dPhi = Phi_NL + r*^T e - (1/2) ||P_T* e||^2, all at the noise-free MAP p* (D*, w_hat*, sigma*, r*)."""
    e = dt.stack_real(d_obs - d_clean, s)
    D_star, chi_star = dt.whitened_jacobian(ctx.model, ctx.forward, ctx.renderer, p_star, s)
    r_star = dt.stack_real(d_clean, s) - dt.stack_real(ctx.data64(chi_star), s)
    w_star, sig_star = dt.gap_direction(D_star, 0)
    Q, _ = np.linalg.qr(D_star)
    z_nl = (p_star[0] - gap) / sig_star
    we = float(w_star @ e)
    pte = 0.5 * float(np.sum((Q.T @ e) ** 2))
    phi_nl = 0.5 * float(r_star @ r_star)
    rse = float(r_star @ e)
    z_pred, dphi_pred = z_nl + we, phi_nl + rse - pte
    return {"z_NL_sigma_at_map": float(z_nl), "w_hat_star_dot_e": we, "z_predicted": float(z_pred), "z_measured": None,
            "dphi_NL": phi_nl, "r_star_dot_e": rse, "half_norm2_PT_e": pte, "dphi_predicted": float(dphi_pred), "dphi_measured": measured["dphi"],
            "expected_half_norm2_PT_e": 0.5 * D_star.shape[1],
            "note": "z_measured is (fitted - base) / sd_gn(m7) in the sign of this module; the prediction uses sigma* from the Fisher matrix without prior"}


def measured_full_sd(ctx, s, measured, d_obs, d_truth, p_star):
    """Posterior sd of the gap at the milestone-7 MAP (noisy data, milestone-7 p_hat) with the Gauss-Newton and the finite-difference Hessian, and the measured z and c
    recomputed with the latter. For the noise-free grid cases the noise-free MAP of this module and the noise-free data."""
    if measured["p_hat"] is not None and d_obs is not None:
        p_hat, data, src = np.asarray(measured["p_hat"], dtype=np.float64), d_obs, "milestone-7 MAP (p_hat of the case file) with the noisy data of the draw"
    else:
        p_hat, data, src = np.asarray(p_star, dtype=np.float64), d_truth, "noise-free MAP of this module with the noise-free data"
    post = sp.ShapePosterior(ctx.model, ctx.forward, data, s, ctx.renderer)
    sd_gn, sd_fd = laplace_sd_g(post, p_hat)
    out = {"source": src, "p_hat": p_hat.tolist(), "sd_g_gn": sd_gn, "sd_g_m7_recorded": measured["g_sd"], "sd_g_full": sd_fd}
    if sd_fd:
        zf = abs(measured["z_fit"]) * measured["g_sd"] / sd_fd
        out.update({"sd_ratio_full_over_gn": sd_fd / sd_gn, "z_gn": abs(measured["z_fit"]), "z_full": zf, "c_gn": measured["c"], "c_full": measured["dphi"] / zf**2})
    return out


def run_case(ctx, label, force=False, nonlinear=True):
    path = OUT / "verify" / f"{label}.json"
    if path.exists() and not force:
        print(f"{label}: done")
        return json.loads(path.read_text())
    t0 = time.perf_counter()
    info = parse_label(label)
    s, variants, measured, d_obs = case_data(ctx, label, info)
    # the profile foot point of the total error: g = g_true, nuisances at their least-squares values
    d_total, p_nom, _ = variants["total"]
    p_foot, phi_foot, ok_foot = dt.foot_point(ctx.model, ctx.forward, ctx.renderer, dt.stack_real(d_total, s), s, p_nom, k=0)
    variants["foot"] = (d_total, p_foot, None)
    extra = m7_map_start(label, info)
    rec = {"label": label, "info": info, "noise_std": s, "measured": measured, "foot_point": {"p": p_foot.tolist(), "phi": phi_foot, "converged": ok_foot}, "variants": {}}
    for name, (d_truth, p0, _) in variants.items():
        nl = nonlinear and name in ("total", "pure")
        start = extra if (name == "total" and extra is not None) else None
        rec["variants"][name] = analyze_variant(ctx, info, label, s, d_truth, p0, nl, start, second=name in ("total", "total_p0mean", "foot"), curv_map=(nl and name == "total"))
        v = rec["variants"][name]
        lin = v["linear"]
        nltxt = "" if "nonlinear" not in v else f"; NL shift {v['nonlinear']['shift_sd']:+.3f} sd, phi {v['nonlinear']['phi']:.4g}, c_NL {v['nonlinear']['c_NL']:.4g}, c_NL(sd full) {fmt(v['nonlinear']['c_NL_full'])}"
        sotxt = "" if "second_order" not in v else f"; dg lin {v['second_order']['dg_linear']:+.5f}, 2nd {v['second_order']['dg_second_order']:+.5f}"
        print(f"{label} [{name}]: z {lin['z']:+.3f}, dPhi {lin['dphi']:.4g}, alpha {lin['alpha_deg']:.2f} deg, beta {lin['beta_deg']:.2f} deg, c_geom {lin['c_geom']:.4g}{nltxt}{sotxt}")
    # the NL shift of the total error is the reference for the other bases of the same data
    tot = rec["variants"]["total"]
    if "nonlinear" in tot:
        for name in ("total_p0mean", "foot"):
            if name in rec["variants"]:
                rec["variants"][name]["second_order"]["dg_NL"] = tot["nonlinear"]["dg_NL"]
        rec["noise_decomposition"] = None if d_obs is None else noise_decomposition(ctx, s, d_total, d_obs, np.asarray(tot["nonlinear"]["p_star"]), info["gap"], measured)
        if rec["noise_decomposition"] is not None:
            rec["noise_decomposition"]["z_measured"] = measured["z_fit"]  # (fitted - true) / sd_gn(m7), the sign of this module
        rec["sigma_full"] = measured_full_sd(ctx, s, measured, d_obs, d_total, np.asarray(tot["nonlinear"]["p_star"]))
    rec["time_s"] = time.perf_counter() - t0
    rec["thread_env"] = prun.thread_env()
    prun.write_json(path, rec)
    return rec


# ----------------------------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------------------------


def fmt(x, spec="{:.4g}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else spec.format(x)


def collect():
    recs = {lab: json.loads((OUT / "verify" / f"{lab}.json").read_text()) for lab in all_labels() if (OUT / "verify" / f"{lab}.json").exists()}
    L = ["# DESIGN 15.3 items 1 and 2: the first-order law against the milestone-7 measurements\n",
         "z is (fitted - base) / sigma here; the measured z_fit is minus the milestone-7 z_true. c_geom = (1/2) tan^2 alpha / cos^2 beta from the noise-free error at the base point p0 "
         "(the nominal p0 = (g, 0.2, 2, 0, 0)). The reference for the law is the noise-free nonlinear fit (c_NL). `measured` columns are ONE noise draw (key 0) of milestone 7 and carry the noise "
         "interaction terms (table 3). sd full: observed-information (finite-difference Hessian) Laplace sd of g; sd GN: Gauss-Newton (Fisher), as in milestone 7.\n",
         "## 1. The law\n",
         "| case | variant | |z| linear | |z| NL (sigma at p0) | |z| NL (sigma at MAP) | |z| NL (sd full) | |z| one draw | dPhi linear | dPhi NL | dPhi one draw | alpha (deg) | beta (deg) | c_geom | c_NL (sigma at p0) | c_NL (sigma at MAP) | c_NL (sd full) | c one draw (sd GN) | c one draw (sd full) | c vs level 0, one draw |",
         "|" + "---|" * 19]
    for lab, r in recs.items():
        m = r["measured"]
        sf = r.get("sigma_full") or {}
        for vn, v in r["variants"].items():
            if vn == "foot":
                continue
            lin, nl = v["linear"], v.get("nonlinear")
            on_total = vn == "total"
            L.append(f"| {lab} | {vn} | {abs(lin['z']):.3f} | {fmt(abs(nl['shift_sd']) if nl else None, '{:.3f}')} | {fmt(abs(nl['shift_sd_at_map_sigma']) if nl else None, '{:.3f}')} | {fmt(abs(nl['shift_sd_full']) if nl and nl.get('shift_sd_full') else None, '{:.3f}')} | {abs(m['z_fit']):.3f} | "
                     f"{lin['dphi']:.4g} | {fmt(nl['phi'] if nl else None)} | {m['dphi']:.4g} | {lin['alpha_deg']:.2f} | {lin['beta_deg']:.2f} | {fmt(lin['c_geom'])} | {fmt(nl['c_NL'] if nl else None)} | "
                     f"{fmt(nl['phi'] / nl['shift_sd_at_map_sigma'] ** 2 if nl else None)} | {fmt(nl['c_NL_full'] if nl else None)} | {m['c']:.4g} | {fmt(sf.get('c_full') if on_total else None)} | {fmt(m.get('c_vs0'))} |")
    L.append("\n## 2. D2 decomposition of the model error (fraction of ||delta||^2; z carried; w_hat A1 fraction)\n")
    L.append("| case | variant | A1 frac | B1 frac | B2 frac | A2 frac | z(A1) | z(B1) | z(B2) | z(A2) | sum of z / z | w_hat in A1 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, r in recs.items():
        for vn, v in r["variants"].items():
            d2 = v["d2"]
            zt = v["linear"]["z"]
            L.append(f"| {lab} | {vn} | " + " | ".join(f"{d2[k]['frac_norm2']:.6f}" for k in IRREP_ORDER) + " | " + " | ".join(f"{d2[k]['z']:.3e}" for k in IRREP_ORDER)
                     + f" | {d2['z_sum_check'] / zt:.12f} | {d2['w_hat_a1_fraction']:.12f} |")
    L.append("\n## 3. One draw against the noise-free fit: z_fit = z_NL + w_hat*^T e, dPhi = Phi_NL + r*^T e - (1/2)||P_T* e||^2 (total error, key 0)\n")
    L.append("| case | z NL (sigma*) | w_hat*^T e | z predicted | z one draw | dPhi NL | r*^T e | (1/2)||P_T* e||^2 | dPhi predicted | dPhi one draw |\n|" + "---|" * 10)
    for lab, r in recs.items():
        nd = r.get("noise_decomposition")
        if nd:
            L.append(f"| {lab} | {nd['z_NL_sigma_at_map']:.5f} | {nd['w_hat_star_dot_e']:.4f} | {nd['z_predicted']:.5f} | {nd['z_measured']:.5f} | {nd['dphi_NL']:.5g} | {nd['r_star_dot_e']:.4f} | "
                     f"{nd['half_norm2_PT_e']:.4f} | {nd['dphi_predicted']:.5g} | {nd['dphi_measured']:.5g} |")
    L.append("\nThe mean of (1/2)||P_T e||^2 is k / 2 = 2.5 (k = 5 parameters); DESIGN's dPhi is the expected excess over a correct model, milestone 7's is lower by this term.\n")
    L.append("## 4. Posterior sd of g: Gauss-Newton against observed information (milestone-7 MAP, noisy data; grid cases: noise-free MAP)\n")
    L.append("| case | sd GN (this module) | sd GN (milestone 7 file) | sd full (FD Hessian) | full / GN | |z| GN | |z| full | c one draw GN | c one draw full |\n|" + "---|" * 9)
    for lab, r in recs.items():
        sf = r.get("sigma_full")
        if sf and sf.get("sd_g_full"):
            L.append(f"| {lab} | {sf['sd_g_gn']:.4e} | {sf['sd_g_m7_recorded']:.4e} | {sf['sd_g_full']:.4e} | {sf['sd_ratio_full_over_gn']:.4f} | {sf['z_gn']:.3f} | {sf['z_full']:.3f} | {sf['c_gn']:.4g} | {sf['c_full']:.4g} |")
    L.append("\n## 5. Second order and the curvature of the residual\n")
    L.append("Delta g (physical units) of the least-squares fit from the base point: first order p1, second order p1 + p2 (p2 = parameter-effects (tangential) + intrinsic (normal) part), against the noise-free nonlinear fit (prior included). "
             "lambda(r1) is the largest |eigenvalue| of the curvature operator of the first-order residual at the base point (the series converges only if it is well below 1).\n")
    L.append("| case | base | Delta g linear | p2 tangential | p2 normal | Delta g second order | Delta g NL | lambda_max(r1) | beta (deg) |\n|" + "---|" * 9)
    for lab, r in recs.items():
        for vn in ("total", "total_p0mean", "foot"):
            v = r["variants"].get(vn)
            if v and "second_order" in v:
                so = v["second_order"]
                lam = max(abs(x) for x in so["lambda_p0_first_order_residual"])
                L.append(f"| {lab} | {vn} (p0 = {', '.join(f'{x:.4g}' for x in v['p0'])}) | {so['dg_linear']:.5f} | {so['dg_p2_tangential']:.5f} | {so['dg_p2_normal']:.5f} | {so['dg_second_order']:.5f} | {fmt(so.get('dg_NL', v.get('nonlinear', {}).get('dg_NL')), '{:.5f}')} | {lam:.3f} | {v['linear']['beta_deg']:.1f} |")
    L.append("\nResidual-curvature operator (D^T D)^-1 <r, d2 mu> at the noise-free nonlinear MAP (physical parameters), and the predicted log Z(FD Hessian) - log Z(GN Hessian) = -(1/2) sum log(1 - lambda):\n")
    L.append("| case | lambda_max | lambda_min | predicted dlogZ | Phi_NL | curvature asymmetry |\n|---|---|---|---|---|---|")
    for lab, r in recs.items():
        rc = r["variants"]["total"].get("residual_curvature_at_map")
        if rc:
            L.append(f"| {lab} | {rc['lambda_max']:.4f} | {rc['lambda_min']:.4f} | {rc['predicted_dlogZ_fd_minus_gn']:.4f} | {rc['residual_norm2_half']:.5g} | {rc['curvature_asymmetry']:.1e} |")
    L.append("\n## 6. Profile foot point (g = g_true, nuisances at their least-squares values) as the base point\n")
    L.append("| case | foot point p | z | dPhi | alpha (deg) | beta (deg) | c_geom = (1/2) tan^2 alpha at beta = 0 |\n|---|---|---|---|---|---|---|")
    for lab, r in recs.items():
        v = r["variants"].get("foot")
        if v:
            lin = v["linear"]
            L.append(f"| {lab} | {', '.join(f'{x:.5g}' for x in r['foot_point']['p'])} | {lin['z']:.3f} | {lin['dphi']:.5g} | {lin['alpha_deg']:.2f} | {lin['beta_deg']:.2f} | {lin['c_geom']:.4g} |")
    (OUT / "verify.md").write_text("\n".join(L) + "\n")
    summary = {"cases": recs, "labels": list(recs), "settings": {"n": N, "side": SIDE, "n_tx": N_TX, "n_rx": N_RX, "tol": TOL, "snr_db": 30, "model": "F2",
                                                                  "irrep_signs_(sigma_x,sigma_y)": dt.SIGNS}}
    prun.write_json(OUT / "verify.json", summary)
    print(f"wrote {OUT / 'verify.json'} and verify.md ({len(recs)} of {len(all_labels())} cases)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cases", default="all", help="'all' or a comma-separated list of labels")
    r.add_argument("--force", action="store_true")
    r.add_argument("--no-nonlinear", action="store_true", help="skip the nonlinear refits (the linear analysis only)")
    sub.add_parser("collect")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect()
        return
    labels = all_labels() if args.cases == "all" else args.cases.split(",")
    ctx = Context()
    for lab in labels:
        run_case(ctx, lab, args.force, not args.no_nonlinear)


if __name__ == "__main__":
    main()
