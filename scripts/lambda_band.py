"""Edge statistics of the TV posterior as a function of lambda across the discrepancy band (docs/DESIGN.md 12.6).

    python scripts/lambda_band.py [--dir results/prior_comparison] [--runs RUN ...] [--out FILE.json]

The discrepancy principle hardly constrains lambda here (milestone-5 review): Phi_MAP rises only from 2372 to 2458 over the whole sweep
(lambda 0.3 to 11.25, a factor 37), and every value but the last lies inside N +- sqrt(N) = 2351 to 2449. So "the |z| > 3 ring disappears at
lambda*" is a statement about a choice. This script puts the pieces side by side:

  * the MAP sweep (``tv_lambda_sweep.json``): Phi_MAP, whether it lies in the band N +- sqrt(N), its distance from the three candidate targets
    (N = 2400, the Phi of noise-free data 2438.2, the Phi of the truth 2578.1, which contains the grid error), R_MAP, R(truth) for the same lambda
    (computed here with the pseudo-Huber TV energy of ``make_overdispersed_starts.tv_energy``), the TV seminorm of the MAP and of the truth
    (their ratio is the contrast the MAP loses), the part lambda (TV_truth - TV_MAP) of R(truth) - R(MAP) that this accounts for, and the MAP bias
    at d = -1;
  * the log-log interpolated lambda for each target (``crossing_lambda``);
  * for every posterior run in ``--runs`` (default: the TV runs of milestone 5 and this follow-up that exist with a summary.json): lambda, draws,
    DESIGN convergence flag, and ``edge_analysis.analyze`` numbers: posterior-mean bias of theta and of log E[eps_r] at d = -1 and +2, the
    posterior sd at d = -1, and the |z| > 3 fractions at d = -1, in the band |d| = 1 and over all pixels, the first two against the truth
    and against the low-passed truth (3 cycles per wavelength, ``resolution.lowpass``).

Writes FILE.json (default <dir>/lambda_band.json) and, with --figure, a three-panel PNG: Phi_MAP against lambda with the band and the targets, the MAP and
posterior bias at d = -1, and the posterior sd and |z| > 3 fraction at d = -1. NumPy and matplotlib only.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

import edge_analysis as ea
from bayes_scatter import resolution as rs
from make_overdispersed_starts import tv_energy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["pc_tv_lam1.25_adapt", "pc_tv_lam3.75_adapt", "pc_tv_lam3.75_indep", "pc_tv_lam8.5_adapt", "pc_tv_lam11.25_adapt"]


def crossing_lambda(lams, phis, target):
    """Log-log interpolated lambda at which Phi_MAP = ``target`` (``lams`` ascending, ``phis`` increasing); None if the target is outside the sweep."""
    lams, phis = np.asarray(lams, dtype=float), np.asarray(phis, dtype=float)
    if not (phis[0] <= target <= phis[-1]) or np.any(np.diff(phis) <= 0):
        return None
    return float(np.exp(np.interp(np.log(target), np.log(phis), np.log(lams))))


def sweep_rows(sweep, theta_true):
    """One dict per swept lambda (ascending) with the quantities of the module docstring."""
    prob = sweep["problem"]
    n_data, phi_clean, phi_truth = prob["n_data"], prob["phi_clean"], prob["phi_truth"]
    delta, sigma_0, mu = sweep["prior"]["delta"], sweep["prior"]["sigma_0"], sweep["prior"]["mu"]
    rows = []
    for r in sorted(sweep["runs"].values(), key=lambda r: r["lam"]):
        r_truth, _ = tv_energy(theta_true, r["lam"], delta, sigma_0, mu)
        rows.append({"lam": r["lam"], "phi_map": r["phi"], "in_band_N_pm_sqrtN": bool(abs(r["phi"] - n_data) <= np.sqrt(n_data)),
                     "phi_map_minus_N": r["phi"] - n_data, "phi_map_minus_phi_clean": r["phi"] - phi_clean,
                     "phi_map_minus_phi_truth": r["phi"] - phi_truth, "R_map": r["R"], "R_truth": r_truth,
                     "R_truth_minus_R_map": r_truth - r["R"], "tv_map": r["tv"], "tv_truth": r["tv_truth"],
                     "tv_contrast_kept": r["tv"] / r["tv_truth"], "R_gap_from_tv_seminorm": r["lam"] * (r["tv_truth"] - r["tv"]),
                     "map_bias_d-1": r["edge_bias"]["-1"], "err_model": r["err_model"]})
    return rows, {"n_data": n_data, "phi_clean": phi_clean, "phi_truth": phi_truth, "band": [n_data - np.sqrt(n_data), n_data + np.sqrt(n_data)]}


def posterior_row(run, base, h, chains=None, label=None, burn=0):
    """The posterior numbers of one run (module docstring), from the stored draws of the chains ``chains`` (default: all). Everything is
    recomputed from ``chains.npz``: the posterior-mean bias of theta and of log E[eps_r] (log of the mean of exp(theta) over the draws) at d = -1 and +2,
    the sd at d = -1, the |z| > 3 fractions (d = -1, the band |d| = 1, all pixels; against the truth, and against the low-passed truth at
    3 cycles per wavelength), the far-field fraction and the bias of each chain's own mean at d = -1. The first ``burn`` recorded draws of every chain
    are dropped (the chains of a short run started away from the posterior bulk still carry their transient)."""
    d = Path(base) / run
    z = np.load(d / "chains.npz")
    truth, chi = z["theta_true"], z["chi_true"]
    dist = ea.signed_distance(chi)
    theta_all = z["theta"].astype(np.float64)[:, burn:]
    idx = list(range(theta_all.shape[0])) if chains is None else list(chains)
    theta = theta_all[idx]
    mean = theta.mean(axis=(0, 1))
    std = np.sqrt(theta.var(axis=1, ddof=1).mean(axis=0) + theta.mean(axis=1).var(axis=0)) if len(idx) > 1 else theta[0].std(axis=0, ddof=1)
    log_eps = np.log(np.exp(theta).mean(axis=(0, 1)))
    lp = rs.lowpass(truth, h, 3.0)
    zt, zlp = (mean - truth) / std, (mean - lp) / std
    band, far = np.abs(dist) == 1, np.abs(dist) >= 4
    summ = json.loads((d / "summary.json").read_text())
    per_chain = [float((theta[k].mean(axis=0) - truth)[dist == -1].mean()) for k in range(len(idx))]
    return {"run": run, "label": label or run, "chains_used": idx, "lam": summ["args"].get("lam"), "draws_per_chain": int(theta.shape[1]),
            "burn": burn, "design_converged": bool(summ["convergence"]["converged"]), "bias_theta_d-1": float((mean - truth)[dist == -1].mean()),
            "bias_theta_d+2": float((mean - truth)[dist == 2].mean()), "bias_log_mean_eps_d-1": float((log_eps - truth)[dist == -1].mean()),
            "sd_d-1": float(std[dist == -1].mean()), "bias_per_chain_d-1": per_chain,
            "frac_abs_z_gt_3_d-1": float((np.abs(zt[dist == -1]) > 3).mean()), "frac_abs_z_gt_3_band": float((np.abs(zt[band]) > 3).mean()),
            "frac_abs_z_gt_3_pooled": float((np.abs(zt) > 3).mean()), "frac_abs_z_gt_3_d-1_vs_lowpass3": float((np.abs(zlp[dist == -1]) > 3).mean()),
            "frac_abs_z_gt_3_pooled_vs_lowpass3": float((np.abs(zlp) > 3).mean()), "far_from_edge_frac_abs_z_gt_3": float((np.abs(zt[far]) > 3).mean())}


def posterior_rows(run, base, h, burn=50):
    """``posterior_row`` for all chains of ``run`` and, if ``starts.json`` says that some chains started at a draw of the rough GMRF posterior
    (kind ``gmrf``) and others did not, a second row without the rough-start chains and without the first ``burn`` recorded draws of the others
    (the rough start is the most overdispersed one and its transient is still in the draws of a short run; a chain started at the truth needs
    a few tens of trajectories to reach the bulk)."""
    rows = [posterior_row(run, base, h)]
    sj = Path(base) / run / "starts.json"
    if sj.exists():
        kinds = [s["kind"] for s in json.loads(sj.read_text())["starts"]]
        keep = [c for c, k in enumerate(kinds) if k != "gmrf"]
        if 0 < len(keep) < len(kinds):
            dropped = [c for c in range(len(kinds)) if c not in keep]
            rows.append(posterior_row(run, base, h, keep, burn=burn,
                                      label=f"{run} without the rough-start chains {dropped}, first {burn} draws dropped"))
    return rows


def plot(out, path):
    """Three-panel figure of ``main``'s result dict ``out`` written to ``path``: (a) Phi_MAP against lambda with the band N +- sqrt(N), N and the Phi of
    noise-free data; (b) the MAP bias and the posterior bias (E theta and log E eps_r) at d = -1; (c) the posterior sd at d = -1 and the |z| > 3
    fraction at d = -1 against the truth and against the low-passed truth. Posterior points are labeled by run."""
    rows, post, ref = out["sweep"], out["posteriors"], out["reference"]
    lam = [r["lam"] for r in rows]
    fig = Figure(figsize=(12, 3.6), layout="constrained")
    a, b, c = fig.subplots(1, 3)
    a.plot(lam, [r["phi_map"] for r in rows], "o-", ms=3)
    a.axhspan(*ref["band"], color="0.9", label="N +- sqrt(N)")
    a.axhline(ref["n_data"], color="0.5", lw=0.8, label="N")
    a.axhline(ref["phi_clean"], color="C3", lw=0.8, ls="--", label="Phi of noise-free data")
    a.set_ylabel("Phi_MAP")
    b.plot(lam, [r["map_bias_d-1"] for r in rows], "o-", ms=3, label="MAP")
    c2 = c.twinx()
    if post:
        pl = [q["lam"] for q in post]
        b.plot(pl, [q["bias_theta_d-1"] for q in post], "s", label="posterior mean of theta")
        b.plot(pl, [q["bias_log_mean_eps_d-1"] for q in post], "^", label="log E[eps_r]")
        c.plot(pl, [q["sd_d-1"] for q in post], "s", color="C0", label="posterior sd (left)")
        c2.plot(pl, [100 * q["frac_abs_z_gt_3_d-1"] for q in post], "o", color="C1", label="% |z|>3 vs truth (right)")
        c2.plot(pl, [100 * q["frac_abs_z_gt_3_d-1_vs_lowpass3"] for q in post], "x", color="C2", label="% |z|>3 vs low-pass (right)")
    b.set_ylabel("bias at d = -1 (log eps_r)")
    c.set_ylabel("posterior sd at d = -1")
    c2.set_ylabel("% |z| > 3 at d = -1")
    for ax in (a, b, c):
        ax.set_xscale("log")
        ax.set_xlabel("TV weight lambda")
        ax.tick_params(labelsize=8)
    for ax in (a, b):
        ax.legend(fontsize=7)
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.savefig(path, dpi=130)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--burn", type=int, default=50, help="recorded draws dropped per chain in the 'without rough-start chains' rows")
    ap.add_argument("--out", default=None, help="default <dir>/lambda_band.json")
    ap.add_argument("--figure", default=None, help="PNG of the three panels")
    args = ap.parse_args(argv)
    base = Path(args.dir)
    sweep = json.loads((base / "tv_lambda_sweep.json").read_text())
    truth = np.load(base / "pc_matern" / "chains.npz")["theta_true"]
    rows, ref = sweep_rows(sweep, truth)
    lams, phis = [r["lam"] for r in rows], [r["phi_map"] for r in rows]
    out = {"reference": ref, "sweep": rows,
           "crossing_lambda": {"phi_map=N": crossing_lambda(lams, phis, ref["n_data"]),
                               "phi_map=phi_clean": crossing_lambda(lams, phis, ref["phi_clean"]),
                               "phi_map=N-sqrtN": crossing_lambda(lams, phis, ref["band"][0]),
                               "phi_map=N+sqrtN": crossing_lambda(lams, phis, ref["band"][1]),
                               "phi_map=phi_truth": crossing_lambda(lams, phis, ref["phi_truth"]),
                               "note": "None: the target lies outside the swept Phi_MAP range 2372 to 2458"}}
    h = sweep["problem"]["side"] / sweep["problem"]["n"]
    out["posteriors"] = [row for r in args.runs if (base / r / "summary.json").exists() for row in posterior_rows(r, base, h)]
    p = Path(args.out) if args.out else base / "lambda_band.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=float))
    print(f"band N +- sqrt(N) = {ref['band'][0]:.0f} to {ref['band'][1]:.0f}; crossings: " + ", ".join(f"{k} {v}" for k, v in out["crossing_lambda"].items() if k != "note"))
    for r in rows:
        print(f"lambda {r['lam']:6.2f}: Phi_MAP {r['phi_map']:.1f} ({'in' if r['in_band_N_pm_sqrtN'] else 'OUT of'} band), TV kept {100 * r['tv_contrast_kept']:.1f}%, "
              f"R(truth)-R(MAP) {r['R_truth_minus_R_map']:.1f} (TV part {r['R_gap_from_tv_seminorm']:.1f}), MAP bias d=-1 {r['map_bias_d-1']:.3f}")
    for q in out["posteriors"]:
        print(f"posterior {q['label'][:60]:60s} lambda {q['lam']}: bias d=-1 {q['bias_theta_d-1']:.3f} (log E eps {q['bias_log_mean_eps_d-1']:.3f}), sd {q['sd_d-1']:.3f}, "
              f"|z|>3 d=-1 {100 * q['frac_abs_z_gt_3_d-1']:.1f}% pooled {100 * q['frac_abs_z_gt_3_pooled']:.1f}% (vs low-pass {100 * q['frac_abs_z_gt_3_pooled_vs_lowpass3']:.1f}%)")
    if args.figure:
        plot(out, args.figure)
    print(f"wrote {p}")
    return out


if __name__ == "__main__":
    main()
