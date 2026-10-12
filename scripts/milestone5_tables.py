"""Print the Markdown tables that are specific to milestone 5 (docs/DESIGN.md section 12) from the stored JSON files.

    python scripts/milestone5_tables.py [--dir results/prior_comparison] [--audit-dir results/prior_comparison/m5]
                                        [--runs pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt pc_gmrf_adapt]
                                        [--compare pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf]

Every number is read from a file (no recomputation) and each table names its source:

  * the lambda sweep of the TV prior (``tv_lambda_sweep.json`` and ``tv_lambda_sweep_delta0.003.json``);
  * the adaptation of each adapted run (``<run>/adaptation.json``: phases, step size, trajectory length, map updates);
  * the sampler of each run (``<run>/summary.json``: step, trajectory length, acceptance, gradient evaluations, wall time);
  * the edge statistics by signed distance to the object edge for the milestone-5 runs and the milestone-4 runs, from
    ``<audit-dir>/edge_analysis.json`` (scripts/edge_analysis.py): bias, posterior std and % |z| > 3 at d = -1 (the pixel row just
    outside the object), at d = +1 and +2 (just inside), in the band |d| = 1 and far from the edge (|d| >= 4), plus the pooled fraction
    with its per-chain range and Monte Carlo error.

The other tables (convergence, features, verdicts, prior energies, rank-normalized audit) come from
``scripts/prior_comparison_tables.py RUN ... --audit-dir <audit-dir>``.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads)

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M5_RUNS = ["pc_tv_lam3.75_adapt", "pc_tv_lam11.25_adapt", "pc_gmrf_adapt"]
M4_RUNS = ["pc_matern", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1", "pc_gmrf"]


def f(x, nd=3):
    """Format a number (None / NaN as a dash)."""
    if x is None or x != x:
        return "-"
    return f"{x:.{nd}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def sweep_table(path):
    """Rows of the lambda sweep ``path`` (tv_lambda_sweep*.json): lambda, Phi_MAP, R_MAP, err_model, err_chi, MAP iterations."""
    d = json.loads(Path(path).read_text())
    star = d["lambda_star"]["lam"]
    rows = []
    for k, r in d["runs"].items():
        mark = " **(lambda*)**" if abs(r["lam"] - star) < 1e-9 else ""
        rows.append([f"{r['lam']:g}{mark}", f(r["phi"], 1), f(r["R"], 1), f(r["err_model"], 4), f(r["err_chi"], 4),
                     f(r.get("edge_bias", {}).get("-1"), 3), r["nit"], f(r["pairs_below_delta"], 3)])
    return d, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--audit-dir", default=None, help="default <dir>/m5")
    ap.add_argument("--runs", nargs="*", default=M5_RUNS)
    ap.add_argument("--compare", nargs="*", default=M4_RUNS, help="milestone-4 runs shown next to the new ones in the edge table")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    ad = Path(args.audit_dir) if args.audit_dir else d / "m5"

    for name in ("tv_lambda_sweep.json", "tv_lambda_sweep_delta0.003.json"):
        if (d / name).exists():
            sw, rows = sweep_table(d / name)
            print(f"### TV lambda sweep, delta {sw['prior']['delta']} ({name}; N = {sw['problem']['n_data']}; lambda* = {sw['lambda_star']['lam']}, "
                  f"interpolated crossing {sw['lambda_star']['crossing_lam_interpolated']:.2f})\n")
            print(table(["lambda", "Phi_MAP", "R_MAP", "err_model", "err_chi", "MAP bias d=-1", "L-BFGS-B its", "pairs below delta"], rows))
            print()

    print("### adaptation (adaptation.json of each run)\n")
    rows = []
    for r in args.runs:
        p = d / r / "adaptation.json"
        if not p.exists():
            continue
        a = json.loads(p.read_text())
        for h in a["history"]:
            up = h.get("update", {})
            if h["phase"] == "readapt":  # the second-stage map of scripts/readapt_map.py: one window of recorded draws, no trajectories of its own
                rows.append([r, f"readapt from {h['source_run']}", f"{h['n_draws']} draws", "-", "-", "-", f"{up['time_s']:.0f}", f(up.get("weight"), 3),
                             f(up.get("trace_sample_over_laplace"), 3), f(up.get("var_ratio_median"), 3), f(up.get("center_shift"), 1), "yes"])
                continue
            rows.append([r, h["phase"], h["trajectories"], f(h["accept_mean"], 3), f(h["eps_end"], 4), f(h["mean_leapfrog"], 1),
                         f"{h['time_s']:.0f}", f(up.get("weight"), 3) if up else "-", f(up.get("trace_sample_over_laplace"), 3) if up else "-",
                         f(up.get("var_ratio_median"), 3) if up else "-", f(up.get("center_shift"), 1) if up else "-",
                         "skipped" if "skipped" in h else ("yes" if h["updated"] else "-")])
    print(table(["run", "phase", "trajectories", "accept", "eps (end)", "mean leapfrog", "s", "weight on Laplace", "tr S / tr Laplace",
                 "variance ratio (median)", "center shift", "map updated"], rows))

    print("\n### sampler of each run (summary.json)\n")
    rows = []
    for r in [*args.runs, *args.compare]:
        p = d / r / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        sa, ph, c = s["sampler"], s["phi_physical"], s["convergence"]
        segs = "+".join(str(x) for x in sa["samples_per_segment"])
        rows.append([r, f"{sa['chains']} x ({sa['warmup']} + {segs})", f(sa["acceptance"]["mean"], 3),
                     "/".join(f(x, 4) for x in sa["final_step"]), sa["n_leapfrog"], sa["n_phi_evals_per_chain"],
                     f"{max(sa['wall_s_per_chain']) / 3600:.2f}", f(ph["mean"], 1), f(ph["std"], 1), "yes" if c["converged"] else "NO"])
    print(table(["run", "chains x (warmup + draws)", "accept", "final step per chain", "L0", "gradient evals / chain",
                 "wall h (slowest chain)", "Phi mean", "Phi sd", "converged"], rows))

    ep = ad / "edge_analysis.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        print(f"\n### edge statistics by signed distance d to the object edge (+ inside; {ep.relative_to(d.parent) if d.parent in ep.parents else ep})\n")
        rows = []
        for r in [*args.compare, *args.runs]:
            if r not in e:
                continue
            x = e[r]
            pr = x["profile_by_signed_distance_to_edge_px"]
            g = lambda k, key: pr.get(str(k), {}).get(key)
            pc = x["z_per_chain_frac_abs_z_gt_3"]
            bc = pr.get("-1", {}).get("bias_per_chain")
            rng = f"{f(min(bc))} to {f(max(bc))}" if bc else "-"
            rows.append([r, "yes" if x["converged"] else "NO", f(g(-1, "bias")), f(x.get("log_mean_eps_bias", {}).get("bias_d-1")), rng, f(g(-1, "std")),
                         f(100 * g(-1, "frac_abs_z_gt_3"), 1),
                         f(g(1, "bias")), f(g(2, "bias")), f(100 * x["edge_band"]["frac_abs_z_gt_3"], 1),
                         f(100 * x["far_from_edge"]["frac_abs_z_gt_3"], 1), f(100 * x["z_from_stored_draws"]["frac_abs_z_gt_3"], 1),
                         f"{f(100 * pc['min'], 1)} to {f(100 * pc['max'], 1)}", f(100 * pc["mc_error_of_pooled"], 2)])
        print(table(["run", "converged", "bias d=-1 (E theta)", "bias d=-1 (log E eps_r)", "bias d=-1 per chain", "sd d=-1", "% |z|>3 d=-1", "bias d=+1", "bias d=+2", "% |z|>3 |d|=1",
                     "% |z|>3 |d|>=4", "% |z|>3 pooled", "per chain", "MC error (pp)"], rows))


if __name__ == "__main__":
    main()
