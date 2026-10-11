"""Print the Markdown tables of the prior-comparison report from the stored summary.json / comparison.json files.

    python scripts/prior_comparison_tables.py [RUN ...] [--dir results/prior_comparison]

Every number printed is read from <dir>/<run>/summary.json or <dir>/comparison.json (no recomputation), so the report can
cite those files. RUN defaults to the four DESIGN 11.3 runs pc_matern, pc_gmrf, pc_qggmrf_sg0.05, pc_qggmrf_sg0.1.
When they exist, the audits are printed as well: prior_energy.json (scripts/prior_energy_audit.py), convergence_audit.json
(scripts/convergence_audit.py), edge_analysis.json (scripts/edge_analysis.py) and, for the robustness controls
(pcs1_qggmrf_sg0.05, pcg1_qggmrf_sg0.05, pcg1_matern), their summary.json.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads)

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["pc_matern", "pc_gmrf", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1"]
CONTROL_RUNS = ["pcs1_qggmrf_sg0.05", "pcg1_qggmrf_sg0.05", "pcg1_matern"]


def f(x, nd=3):
    """Format a number (None / NaN as a dash)."""
    if x is None or x != x:
        return "-"
    return f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    args = ap.parse_args(argv)
    d = Path(args.dir)
    S = {r: json.loads((d / r / "summary.json").read_text()) for r in args.runs}
    first = next(iter(S.values()))
    op = first["operating_point"]
    print("### setup (any summary.json; operating point from results/prior_comparison/%s/summary.json)\n" % args.runs[0])
    print(f"n={first['args']['n']}, side={first['args']['side']}, eps_r={first['args']['eps_r']}, SNR={first['args']['snr']} dB, "
          f"seed={first['args']['seed']}, N={first['data']['n_data']}, noise std s={first['data']['noise_std']:.4e}, "
          f"d_obs sha256 {first['data']['d_obs_sha256'][:16]}; grid error {100 * op['rel_model_error']:.2f}% of ||d||, noise "
          f"{100 * op['rel_noise']:.2f}%, ratio {op['model_to_noise']:.3f}; Phi(truth) {op['phi_truth']:.1f}, noise-free-data Phi "
          f"{op['phi_clean']:.1f}\n")

    print("### MAP and Laplace\n")
    rows = []
    for r, s in S.items():
        m, lp = s["map"], s["laplace"]
        rows.append([r, m["nit"], m["nfev"], f(m["U"], 1), f(m["phi"], 1), f(m["R"], 1), f"{m['grad_inf']:.1e}",
                     f(lp["theta_std"]["inside"], 4), f(lp["theta_std"]["outside"], 4),
                     f"{lp['jacobian_check_rel_err']:.1e}"])
    print(table(["run", "MAP its", "evals", "U", "Phi", "R", "|grad U|inf", "Laplace sd in", "Laplace sd out", "Jacobian check"], rows))

    print("\n### sampler and convergence (DESIGN 11.2: R-hat Phi <= 1.01, max pixel R-hat <= 1.05, worst-pixel ESS >= 200)\n")
    rows = []
    for r, s in S.items():
        sa, c, ph = s["sampler"], s["convergence"], s["phi_physical"]
        rows.append([r, f"{sa['chains']} x ({sa['warmup']} + {sa['samples']})", f(sa["acceptance"]["mean"], 3),
                     f(ph["mean"], 1), f(ph["std"], 1), f(ph["first_quarter_mean"], 1), f(ph["last_quarter_mean"], 1),
                     f(c["rhat_phi"], 3), f(c["rhat_pixel_max"], 3), f(c["ess_pixel_min"], 0), f(c["ess_pixel_min_geyer"], 0),
                     "yes" if c["converged"] else "NO"])
    print(table(["run", "chains x (warmup + draws)", "accept", "Phi mean", "Phi sd", "Phi 1st quarter", "Phi last quarter",
                 "R-hat Phi", "max pixel R-hat", "worst ESS (Sokal)", "worst ESS (Geyer)", "converged"], rows))
    for r, s in S.items():
        if s["convergence"]["unconverged_reasons"]:
            print(f"\n{r}: " + "; ".join(s["convergence"]["unconverged_reasons"]))

    print("\n### posterior standard deviation of theta = log eps_r\n")
    rows = []
    for r, s in S.items():
        p = s["posterior"]
        pr = s["prior"]["prior_summary"].get("theta_std", {})
        ratio = p["theta_std_over_prior_std"]
        lm = p["mcmc_over_laplace_std"]
        rows.append([r, f(pr.get("inside"), 4), f(pr.get("outside"), 4), f(p["theta_std"]["inside"], 4),
                     f(p["theta_std"]["outside"], 4), f(ratio["inside"], 3) if ratio else "-", f(ratio["outside"], 3) if ratio else "-",
                     f(ratio["mean"], 3) if ratio else "-", f(lm["mean"], 3), f(lm["min"], 3), f(lm["max"], 3),
                     f(p["theta_mean_minus_map"], 4)])
    print(table(["run", "prior sd in", "prior sd out", "post sd in", "post sd out", "post/prior in", "post/prior out",
                 "post/prior mean", "MCMC/Laplace mean", "min", "max", "mean(theta) - MAP"], rows))

    print("\n### misfit to the truth: |z| = |mean - truth| / std in log eps_r; reconstruction errors\n")
    rows = []
    for r, s in S.items():
        z, m = s["z"], s["metrics"]
        rows.append([r, f(100 * z["frac_abs_z_gt_3"], 1), f(100 * z["frac_inside"], 1), f(100 * z["frac_outside"], 1),
                     f(z["edge_share"], 3), f(z["edge_fraction_of_doi"], 3), f(z["max_abs_z"], 2), f(z["rms"], 2),
                     f(m["err_model"]["map"], 4), f(m["err_model"]["posterior_mean"], 4), f(m["err_chi"]["map"], 4),
                     f(m["err_chi"]["posterior_mean"], 4), f(m["err_data"]["posterior_mean"], 4)])
    print(table(["run", "%|z|>3", "% inside", "% outside", "edge share", "edge share by chance", "max |z|", "rms z",
                 "err_model MAP", "err_model mean", "err_chi MAP", "err_chi mean", "err_data mean"], rows))

    print("\n### feature probabilities (delta = log2/2) from the stored draws; mcse = spread of the per-chain estimates\n")
    names = list(first["features"]["probabilities"])
    rows = []
    for r, s in S.items():
        rows.append([r, "yes" if s["convergence"]["converged"] else "no"] +
                    [f"{f(s['features']['probabilities'][k]['p'], 3)} (+-{f(s['features']['probabilities'][k]['mcse'], 3)})"
                     for k in names])
    print(table(["run", "converged"] + names, rows))
    print("\ntruth satisfies: " + ", ".join(f"{k}: {first['features']['probabilities'][k]['truth_satisfies']}" for k in names))

    cmp_path = d / "comparison.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text())
        print(f"\n### verdicts and disagreement (comparison.json; {cmp['n_converged']} converged runs of {len(cmp['labels'])})\n")
        print(table(["feature", "verdict"], [[k, v] for k, v in cmp["features"]["verdict"].items()]))
        rows = []
        for pair, info in cmp["pairs"].items():
            t = info["theta"]
            rows.append([pair, "yes" if info["both_converged"] else "no", f(100 * t["frac_gt_2"], 1), f(100 * t["frac_gt_2_inside"], 1),
                         f(100 * t["frac_gt_2_outside"], 1), f(100 * t["frac_gt_2_near_edge"], 1),
                         f(100 * t["frac_gt_2_away_from_edge"], 1), f(t["max"], 2), f(t["mean"], 2)])
        print()
        print(table(["pair", "both converged", "% pixels >2", "inside", "outside", "near edge", "away from edge", "max", "mean"], rows))

    extra_tables(d, args.runs)


def extra_tables(d, runs):
    """The audit tables, from the JSON files that exist in ``d`` (silently skipped otherwise)."""
    pe, ca, ea = (d / name for name in ("prior_energy.json", "convergence_audit.json", "edge_analysis.json"))
    controls = [r for r in CONTROL_RUNS if (d / r / "summary.json").exists()]
    if controls:
        print("\n### robustness controls (summary.json of each)\n")
        rows = []
        for r in list(runs) + controls:
            if not (d / r / "summary.json").exists():
                continue
            s = json.loads((d / r / "summary.json").read_text())
            sa, c, z = s["sampler"], s["convergence"], s["z"]
            rows.append([r, f"{s['args']['seed']} / {s['args']['data_factor']}", f"{sa['chains']} x ({sa['warmup']} + {sa['samples']})",
                         f(s["operating_point"]["model_to_noise"], 3), f(s["phi_physical"]["mean"], 1), f(c["rhat_phi"], 3),
                         f(100 * z["frac_abs_z_gt_3"], 1), f(z["edge_share"], 3), "yes" if c["converged"] else "NO"])
        print(table(["run", "seed / data factor", "chains x (warmup + draws)", "grid error / noise", "Phi mean", "R-hat Phi",
                     "%|z|>3", "edge share", "converged"], rows))
    if pe.exists():
        e = json.loads(pe.read_text())
        print("\n### prior energy R and data misfit Phi at the truth, the MAP and the posterior draws (prior_energy.json)\n")
        rows = []
        for r, x in e["runs"].items():
            R, P = x["R"], x["Phi"]
            rows.append([r, f(R["truth"], 1), f(R["map"], 1), f"{f(R['draws_mean'], 1)} +- {f(R['draws_sd'], 1)}",
                         f(R["truth_above_draws_in_sd"], 1), f(P["truth"], 1), f(P["noise_free_data"], 1), f(P["map"], 1),
                         f"{f(P['draws_mean'], 1)} +- {f(P['draws_sd'], 1)}", f(x["U"]["truth_minus_map"], 1)])
        print(table(["run", "R truth", "R MAP", "R draws", "(R truth - R draws) / sd", "Phi truth", "Phi noise-free data",
                     "Phi MAP", "Phi draws", "U truth - U MAP"], rows))
        print("\n### R and Phi of the truth blurred by a Gaussian of sigma pixels (prior_energy.json; Phi uses each run's data)\n")
        runs_e = e["runs"]
        sig = next(iter(runs_e.values()))["blur"]["sigma_px"]
        rows = []
        for r, x in runs_e.items():
            rows.append([r + " R"] + [f(v, 1) for v in x["blur"]["R"]])
            rows.append([r + " Phi"] + [f(v, 0) for v in x["blur"]["Phi"]])
        print(table(["quantity"] + [f"sigma {v:g}" for v in sig], rows))
        print("\n### posterior-mean map (prior_energy.json)\n")
        print(table(["run", "R mean map", "R truth - R mean map", "Phi mean map", "Phi mean map - Phi truth"],
                    [[r, f(x["posterior_mean"]["R"], 1), f(x["posterior_mean"]["R_truth_minus_mean"], 1),
                      f(x["posterior_mean"]["Phi"], 1), f(x["posterior_mean"]["Phi_mean_minus_truth"], 1)] for r, x in runs_e.items()]))
        print("\n### cost of a step of height log 2: one jump against a ramp (per pair line, prior_energy.json)\n")
        print(table(["potential", "sharp", "ramp 2 px", "ramp 4 px", "ramp 8 px", "sharp / ramp 8"],
                    [[x["label"], f(x["sharp"], 2), f(x["ramp_2"], 2), f(x["ramp_4"], 2), f(x["ramp_8"], 2), f(x["ratio_8"], 2)]
                     for x in e["step_cost"]]))
    if ca.exists():
        c = json.loads(ca.read_text())
        print("\n### convergence audit: rank-normalized R-hat, ESS and the AR(1) null (convergence_audit.json)\n")
        rows = []
        for r, x in c.items():
            rk, es, nl = x["rank_normalized"], x["ess"], x["null_ar1"]
            rows.append([r, f"{x['chains']} x {x['draws_per_chain']}", f(x["classic"]["rhat_phi"], 4), f(rk["phi"]["worst"], 4),
                         f(nl["p_classic_rhat_phi_above_1.01"], 2), f(x["classic"]["rhat_pixel_max"], 4), f(rk["pixel_max_worst"], 4),
                         f(nl["pixel_max_rank_median_q95"][1], 4), f"{rk['pixels_worst_above_1.05']} ({rk['pixels_worst_above_1.05_within_1px_of_edge']})",
                         f(es["worst_pixel_sokal_summed"], 0), f(es["worst_pixel_bulk_multichain"], 0), f(x["tau"]["phi"], 1)])
        print(table(["run", "chains x draws", "R-hat Phi", "rank R-hat Phi", "null P(R-hat Phi > 1.01)", "max pixel R-hat",
                     "max pixel rank R-hat", "null q95 of it", "pixels > 1.05 (near edge)", "worst ESS Sokal",
                     "worst ESS bulk", "tau Phi"], rows))
    if ea.exists():
        a = json.loads(ea.read_text())
        print("\n### edge miscoverage per run and per chain (edge_analysis.json)\n")
        rows = []
        for r, x in a.items():
            pc = x["z_per_chain_frac_abs_z_gt_3"]
            prof = {"-1": {}, **x["profile_by_signed_distance_to_edge_px"]}
            rows.append([r, "yes" if x["converged"] else "NO", f"{x['chains']} x {x['draws_per_chain']}",
                         f(100 * x["z_from_stored_draws"]["frac_abs_z_gt_3"], 1),
                         f"{f(100 * pc['min'], 1)} to {f(100 * pc['max'], 1)}", f(100 * pc["mc_error_of_pooled"], 2),
                         f(prof["-1"].get("bias"), 3), f(prof["-1"].get("std"), 3),
                         f(100 * prof["-1"]["frac_abs_z_gt_3"], 1) if prof["-1"] else "-",
                         f(100 * x["edge_band"]["frac_abs_z_gt_3"], 1)])
        print(table(["run", "converged", "chains x draws", "%|z|>3 pooled", "per chain min to max", "MC error (pooled, %)",
                     "bias d=-1", "sd d=-1", "% |z|>3 at d=-1", "% |z|>3 in |d|=1"], rows))


if __name__ == "__main__":
    main()
