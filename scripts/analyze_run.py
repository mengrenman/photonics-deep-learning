"""Post-process runs of scripts/run_posterior.py into the per-run metrics of the lab report.

For every run directory results/<name>/ (summary.json + chains.npz) it computes, from the stored chains,

  * err_model (eq. 15.21 on eps_r) of the MAP and of the posterior mean, err_chi, err_data (from summary.json);
  * the mean posterior standard deviation of log eps_r and of eps_r inside and outside the scatterer
    (inside = true contrast > 0 on the inference grid, so partially covered boundary pixels count as inside);
  * the fraction of pixels with |z| > 3, z = (posterior mean - truth) / posterior std in log eps_r (all pixels,
    inside, outside), with the truth log(eps_true) taken from the coverage-averaged phantom on the inference grid;
  * the posterior std of log eps_r relative to the prior std sigma (mean over pixels, inside, outside, and the
    fraction of pixels below 0.5 and above 0.9: data-determined versus prior-dominated);
  * the posterior probability that eps_r < 1, averaged over the true-background pixels (0.5 is the prior value);
  * ESS (summed over chains), tau and split R-hat of the data misfit Phi, of the five fixed pixels, and the minimum /
    median over ALL pixels of log eps_r (from the stored, thinned draws), ESS per Phi evaluation (with and without
    the setup cost: MAP evaluations, plus the Gauss-Newton products of a Laplace preconditioner) and per second.
    "Phi" is always the physical misfit. For Laplace-whitened runs the sampler's trace is the reparametrized
    Phi' = Phi + |xi|^2/2 - |zeta|^2/2, which is NOT the misfit; the misfit of the stored draws is
    chains.npz['phi_physical'] (one value per stored draw) and Phi' is reported separately as phi_prime_*;
  * Geyer's (more conservative) ESS of Phi and of the worst pixel, as a cross-check of the Sokal values;
  * for Laplace runs, a reference: the Laplace (Gauss-Newton) pixel standard deviation of log eps_r from the cached
    eigenpairs, and the ratio MCMC std / Laplace std (a std ratio well below 1 flags an unconverged sampler);
  * where the |z| > 3 pixels are: their share lying within one pixel of a material edge (edge pixels are about a
    third of the DOI of the austria phantom), and the maximum |z|;
  * the fraction of the prior variance of log eps_r at wavenumbers beyond 2 k0 (outside the Born band), which is
    what the data can never see under this prior.

Welford statistics (mean, variance over every post-warmup step) are used for means and standard deviations;
the thinned draws are used for P(eps_r < 1), R-hat and ESS over all pixels.

With --pool LABEL the chains of all runs given are pooled (e.g. the MAP-initialized and the prior-initialized
runs of one sampler) to compute the split R-hat across initializations, written to results/<LABEL>_pooled.json.

Usage:  python scripts/analyze_run.py RUN [RUN ...] [--pool LABEL]   (RUN is a name under results/)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import bayes_scatter  # noqa: F401
from bayes_scatter import diagnostics as dg
from bayes_scatter.geometry import Grid
from bayes_scatter.inference import laplace_theta_variance
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    d = ROOT / "results" / name
    summary = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    return d, summary, z


def pooled_mean_std(mean_c, var_c):
    pooled = np.asarray(var_c).mean(axis=0) + np.asarray(mean_c).var(axis=0)
    return np.asarray(mean_c).mean(axis=0), np.sqrt(pooled)


def ess_all_pixels(x):
    """x (chains, N, n, n): ESS summed over chains for every pixel, shape (n, n)."""
    return np.stack([dg.ess(c.reshape(c.shape[0], -1)) for c in x]).sum(axis=0).reshape(x.shape[2:])


def misfit_series(s, z):
    """(Phi series (chains, N), every, kind): the data misfit Phi, the thinning of that series (1 = every step)
    and the kind of the sampler's own trace ("physical" or a reparametrized potential)."""
    kind = s["phi"].get("kind", "reparametrized (stored before the fix)" if "laplace" in s else "physical")
    if "phi_physical" in z.files:
        return z["phi_physical"], s.get("phi_physical", {}).get("every", 1), kind
    if kind != "physical":
        raise ValueError("this Laplace run stored no physical misfit; re-run it with the current script")
    return z["phi"], 1, kind


def laplace_reference(s, a):
    """Pixel std of log eps_r under the Laplace posterior at the MAP, from the cached Gauss-Newton eigenpairs
    (None if the run has no cache). Directions beyond the computed K are prior-like."""
    lap = s.get("laplace")
    if not lap or not lap.get("cache"):
        return None
    path = Path(lap["cache"])
    path = path if path.is_absolute() else ROOT / path
    if not path.exists():
        return None
    c = np.load(path)
    field = GaussianField(Grid(a["n"], a["side"]), a["prior"], sigma=a["sigma"], ell=a["ell"], nu=a["nu"],
                          mean=a["mean"])
    return np.sqrt(laplace_theta_variance(field, c["lam"], c["V"]))


def prior_variance_fraction_beyond(a, k_cut):
    """Fraction of the prior variance of log eps_r at wavenumbers |k| > k_cut (rad per wavelength), torus spectrum."""
    field = GaussianField(Grid(a["n"], a["side"]), a["prior"], sigma=a["sigma"], ell=a["ell"], nu=a["nu"],
                          mean=a["mean"])
    k = np.asarray(field.wavenumbers)
    kk = np.hypot(*np.meshgrid(k, k, indexing="ij"))
    spec = np.asarray(field.spectrum)
    return float(spec[kk > k_cut].sum() / spec.sum())


def edge_mask(chi_true):
    """Pixels within one pixel of a material edge: dilation of the object minus the erosion of its full-contrast core."""
    inside = chi_true > 1e-9
    full = chi_true > (chi_true.max() - 1e-9)
    return binary_dilation(inside, iterations=1) & ~binary_erosion(full, iterations=1)


def analyze(name):
    d, s, z = load(name)
    a = s["args"]
    sigma = a["sigma"]
    eps_true = z["eps_true"]
    chi_true = z["chi_true"]
    inside = chi_true > 1e-9
    outside = ~inside
    theta_true = np.log(eps_true)

    mean_c, var_c = z["welford_mean"], z["welford_var"]  # (chains, 2, n, n): theta, eps_r
    (theta_mean, eps_mean), (theta_std, eps_std) = pooled_mean_std(mean_c, var_c)
    zmap = (theta_mean - theta_true) / theta_std
    ratio = theta_std / sigma

    samples = z["samples"].astype(np.float64)  # (chains, Ns, 2, n, n)
    theta_s, eps_s = samples[:, :, 0], samples[:, :, 1]
    p_below = (eps_s < 1.0).mean(axis=(0, 1))  # (n, n)
    chains, ns = theta_s.shape[:2]
    thin = a["thin"]

    phi_trace = z["phi"]  # what the sampler ran on (Phi', not Phi, for Laplace-whitened runs)
    phi, phi_every, phi_kind = misfit_series(s, z)  # the data misfit
    n_evals = s["forward_solves"]["sampling_forward_solves"]
    wall = s["wall_time_s"]["sampling"]
    lap = s.get("laplace")
    setup_evals = int(s["map"]["nfev"]) + (int(lap["matvecs_jvp_plus_vjp"]) if lap else 0)
    # ESS of the thinned draws; it equals the ESS of the full series when tau >> thin (the stiff regime here)
    ess_pix = ess_all_pixels(theta_s) if ns >= 8 else np.full(theta_mean.shape, np.nan)
    rhat_pix = dg.split_rhat(theta_s) if (chains >= 2 and ns >= 8) else np.full(theta_mean.shape, np.nan)
    rhat_phi = dg.split_rhat(phi) if (chains >= 2 and phi.shape[1] >= 8) else float("nan")
    ess_phi = float(sum(dg.ess(c) for c in phi)) if phi.shape[1] >= 8 else None
    ess_phi_geyer = float(sum(dg.ess_geyer(c) for c in phi)) if phi.shape[1] >= 8 else None
    ess_pix_geyer = (np.stack([dg.ess_geyer(c.reshape(c.shape[0], -1)) for c in theta_s]).sum(axis=0)
                     if ns >= 8 else np.full(theta_mean.size, np.nan))
    q = max(phi.shape[1] // 4, 1)

    # Laplace reference and the |z| > 3 pixels
    std_lap = laplace_reference(s, a)
    edge = edge_mask(chi_true)
    big = np.abs(zmap) > 3
    ref = {}
    if std_lap is not None:
        r = theta_std / std_lap
        ref = {"laplace_std_theta_inside": float(std_lap[inside].mean()) if inside.any() else None,
               "laplace_std_theta_outside": float(std_lap[outside].mean()) if outside.any() else None,
               "laplace_std_ratio_mean": float(std_lap.mean() / sigma),
               "mcmc_over_laplace_std_mean": float(r.mean()), "mcmc_over_laplace_std_min": float(r.min()),
               "mcmc_over_laplace_std_max": float(r.max()),
               "frac_abs_z_gt_3_laplace_std": float(np.mean(np.abs((theta_mean - theta_true) / std_lap) > 3))}

    m = s["metrics"]
    out = {
        "run": name,
        "sampler": a["sampler"], "init": "over" if (lap or {}).get("init") == "overdispersed" else a["init"], "phantom": a["phantom"], "n": a["n"], "side": a["side"],
        "data_n": s.get("data_grid_n", a["n"] * a.get("data_factor", 2)),
        "snr_db": a["snr"], "prior": a["prior"], "sigma": sigma, "ell": a["ell"], "nu": a["nu"],
        "chains": chains, "warmup": a["warmup"], "samples": a["samples"], "thin": thin,
        "acceptance": s["acceptance"]["mean"], "final_step": float(np.mean(s["final_step"])),
        "phi_kind": phi_kind, "phi_every": phi_every,
        "phi_mean": float(phi.mean()), "phi_std": float(phi.std()), "phi_map": s["map"]["phi"],
        "phi_first_quarter_mean": float(phi[:, :q].mean()), "phi_last_quarter_mean": float(phi[:, -q:].mean()),
        "phi_trace_mean": float(phi_trace.mean()),  # equals phi_mean unless the sampler ran on Phi'
        "n_data": s["phi"]["n_data"], "operating_point": s.get("operating_point"),
        "err_model": m["err_model"], "err_chi": m["err_chi"], "err_data": m["err_data"],
        "std_theta_inside": float(theta_std[inside].mean()) if inside.any() else None,
        "std_theta_outside": float(theta_std[outside].mean()) if outside.any() else None,
        "std_eps_inside": float(eps_std[inside].mean()) if inside.any() else None,
        "std_eps_outside": float(eps_std[outside].mean()) if outside.any() else None,
        "frac_abs_z_gt_3": float(np.mean(np.abs(zmap) > 3)),
        "frac_abs_z_gt_3_inside": float(np.mean(np.abs(zmap[inside]) > 3)) if inside.any() else None,
        "frac_abs_z_gt_3_outside": float(np.mean(np.abs(zmap[outside]) > 3)) if outside.any() else None,
        "std_ratio_mean": float(ratio.mean()),
        "std_ratio_inside": float(ratio[inside].mean()) if inside.any() else None,
        "std_ratio_outside": float(ratio[outside].mean()) if outside.any() else None,
        "frac_std_ratio_lt_0.5": float(np.mean(ratio < 0.5)),
        "frac_std_ratio_gt_0.9": float(np.mean(ratio > 0.9)),
        "p_eps_lt_1_background": float(p_below[outside].mean()) if outside.any() else None,
        "ess_phi": ess_phi,
        "tau_phi": None if ess_phi is None else float(np.mean([dg.tau_int(c) for c in phi])) * phi_every,
        "rhat_phi": None if not np.isfinite(rhat_phi) else float(rhat_phi),
        "ess_phi_geyer": ess_phi_geyer, "ess_pixels_min_geyer": float(np.nanmin(ess_pix_geyer)),
        "ess_phi_prime": s["phi"]["ess"] if phi_kind != "physical" else None,
        "rhat_phi_prime": s["phi"]["rhat"] if phi_kind != "physical" else None,
        "phi_prime_mean": float(phi_trace.mean()) if phi_kind != "physical" else None,
        "ess_pixels_min": float(np.nanmin(ess_pix)), "ess_pixels_median": float(np.nanmedian(ess_pix)),
        "rhat_pixels_max": float(np.nanmax(rhat_pix)), "rhat_pixels_median": float(np.nanmedian(rhat_pix)),
        "sampling_phi_evals": n_evals, "sampling_wall_s": wall, "setup_evals": setup_evals,
        "load_avg_at_start": s.get("machine", {}).get("load_avg_1min_at_start"),
        "ess_phi_per_eval": None if ess_phi is None else ess_phi / n_evals,
        "ess_phi_per_eval_incl_setup": None if ess_phi is None else ess_phi / (n_evals + setup_evals),
        "ess_phi_per_s": None if ess_phi is None else ess_phi / wall,
        "ess_pixmin_per_eval": float(np.nanmin(ess_pix)) / n_evals,
        "ess_pixmin_per_eval_incl_setup": float(np.nanmin(ess_pix)) / (n_evals + setup_evals),
        "ess_pixmin_per_s": float(np.nanmin(ess_pix)) / wall,
        "max_abs_z": float(np.abs(zmap).max()), "edge_pixel_fraction": float(edge.mean()),
        "frac_abs_z_gt_3_edge_share": float(big[edge].sum() / big.sum()) if big.any() else None,
        "prior_var_frac_beyond_2k0": prior_variance_fraction_beyond(a, 4.0 * np.pi),
        **ref,
        "pixel_ess_theta": {k: v["log_eps_r"]["ess"] for k, v in s["pixels"].items()},
        "pixel_rhat_theta": {k: v["log_eps_r"]["rhat"] for k, v in s["pixels"].items()},
        "max_solver_residual_final": s["max_solver_residual_final_states"],
        "map": s["map"], "wall_total_s": s["wall_time_s"]["total"],
        "laplace": None if not lap else {k: lap[k] for k in (
            "k", "lam_max", "cache_reused", "recenter", "zeta_v0_final_per_chain", "theta_mean_post_minus_map",
            "implied_max_stable_mala_step_in_xi", "implied_max_stable_pcn_langevin_delta_in_xi",
            "n_lam_gt_1.5", "setup_grad_evals") if k in lap} | {"recenter_info": lap.get("recenter_info")},
        "theta_mean_range": [float(theta_mean.min()), float(theta_mean.max())],
        "theta_std_range": [float(theta_std.min()), float(theta_std.max())],
    }
    (d / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def pooled(label, names):
    """Split R-hat across all chains of all runs (same thinning and length required)."""
    phis, thetas = [], []
    for nm in names:
        _, summ, z = load(nm)
        phis.append(misfit_series(summ, z)[0])
        thetas.append(z["samples"].astype(np.float64)[:, :, 0])
    ln = min(p.shape[1] for p in phis)
    lt = min(t.shape[1] for t in thetas)
    phi = np.concatenate([p[:, :ln] for p in phis])
    theta = np.concatenate([t[:, :lt] for t in thetas])
    rh_pix = dg.split_rhat(theta)
    res = {"label": label, "runs": names, "chains": int(phi.shape[0]),
           "rhat_phi": float(dg.split_rhat(phi)), "rhat_pixels_max": float(rh_pix.max()),
           "rhat_pixels_median": float(np.median(rh_pix)),
           "frac_pixels_rhat_lt_1.1": float(np.mean(rh_pix < 1.1))}
    (ROOT / "results" / f"{label}_pooled.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pool", default=None, help="label; pool the chains of all runs for the R-hat across inits")
    args = ap.parse_args(argv)
    for nm in args.runs:
        o = analyze(nm)
        std_ref = ("" if "mcmc_over_laplace_std_mean" not in o else
                   f"; MCMC/Laplace std {o['mcmc_over_laplace_std_mean']:.3f} (min {o['mcmc_over_laplace_std_min']:.3f})")
        print(f"{nm}: {o['sampler']}/{o['init']} acc {o['acceptance']:.3f} Phi {o['phi_mean']:.1f} (MAP {o['phi_map']:.1f}, N {o['n_data']}; "
              f"first/last quarter {o['phi_first_quarter_mean']:.1f}/{o['phi_last_quarter_mean']:.1f}"
              + ("" if o["phi_prime_mean"] is None else f"; sampler potential Phi' {o['phi_prime_mean']:.1f}") + ") "
              f"err_model MAP {o['err_model']['map']:.4f} mean {o['err_model']['posterior_mean']:.4f} "
              f"(vac {o['err_model']['vacuum']:.4f}); err_data MAP {o['err_data']['map']:.4f} mean {o['err_data']['posterior_mean']:.4f}; "
              f"std(theta) in/out {o['std_theta_inside']:.4f}/{o['std_theta_outside']:.4f}{std_ref}; frac|z|>3 {o['frac_abs_z_gt_3']:.3f}; "
              f"std ratio {o['std_ratio_mean']:.3f}; P(eps<1) bg {o['p_eps_lt_1_background']:.3f}; "
              f"ESS Phi {o['ess_phi']}; ESS pix min {o['ess_pixels_min']:.1f}; Rhat Phi {o['rhat_phi']}; Rhat pix max {o['rhat_pixels_max']:.3f}")
    if args.pool:
        print(json.dumps(pooled(args.pool, args.runs), indent=1))


if __name__ == "__main__":
    main()
