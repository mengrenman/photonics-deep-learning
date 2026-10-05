"""Markdown tables and Phi-trace figures for a group of runs analyzed with scripts/analyze_run.py.

Usage:  python scripts/make_tables.py GROUP RUN [RUN ...] [--pool LABEL ...]

Reads results/<RUN>/metrics.json, writes results/tables/<GROUP>.md (efficiency table, accuracy / uncertainty table)
and results/figures/<GROUP>_phi.png (the data misfit Phi on a log axis and zoomed, one line per chain, colored by
sampler, dashed for chains started at the prior or overdispersed).

"Phi" is always the data misfit sum |F(chi) - d_obs|^2 / s^2. Laplace-whitened runs sample the reparametrized potential
Phi' = Phi + |xi|^2/2 - |zeta|^2/2, which is not the misfit (it sits thousands below it); it is shown in its own
column ("Phi' mean") and, if the group has such runs, in results/figures/<GROUP>_phiprime.png. The Phi columns and
the Phi figure use the misfit of the stored draws (every `thin`-th step; ESS and R-hat of Phi are computed on that
thinned series, which understates the ESS of a series whose autocorrelation time is shorter than the thinning).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"pcn": "tab:blue", "pcn_langevin": "tab:orange", "mala": "tab:green", "hmc": "tab:red"}


def f(x, p=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    ax = abs(x)
    if ax != 0 and (ax < 10 ** (-p) or ax >= 1e5):
        return f"{x:.2e}"
    return f"{x:.{p}f}"


def per_chain_evals(m):
    return m["sampling_phi_evals"] / m["chains"]


def unconverged(m):
    """True if the chains have visibly not converged (split R-hat of Phi above 1.1, or of some pixel above 2, or not
    finite), so that ESS values estimated from them carry no information about sampler efficiency."""
    rp, rx = m["rhat_phi"], m["rhat_pixels_max"]
    return rp is None or not np.isfinite(rp) or rp > 1.1 or not np.isfinite(rx) or rx > 2.0


def ess_cell(value, m, p=1):
    """ESS-type number, in parentheses when the run is unconverged."""
    txt = f(value, p)
    return f"({txt})" if unconverged(m) and value is not None else txt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pool", nargs="*", default=[], help="labels of results/<label>_pooled.json to append")
    args = ap.parse_args(argv)
    ms = [json.loads((ROOT / "results" / r / "metrics.json").read_text()) for r in args.runs]
    any_prime = any(m["phi_prime_mean"] is not None for m in ms)
    any_lap = any("mcmc_over_laplace_std_mean" in m for m in ms)

    lines = []
    lines.append(f"**Efficiency and convergence ({args.group}).** Eval = one Phi evaluation (all transmitters); a gradient "
                 "evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of "
                 "complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws "
                 "(drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of "
                 "log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products "
                 "(one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged "
                 "(split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine "
                 "(load average at start in parentheses), so rates per second are indicative only.\n")
    head = ("| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | "
            "Phi first / last quarter |" + (" Phi' mean (sampler potential) |" if any_prime else "") +
            " ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | "
            "R-hat Phi | R-hat pix max | wall s (load) |")
    lines.append(head)
    lines.append("|" + "---|" * (head.count("|") - 1))
    for r, m in zip(args.runs, ms):
        load = m.get("load_avg_at_start")
        prime = f" {f(m['phi_prime_mean'], 1)} |" if any_prime else ""
        lines.append(
            f"| {r} | {m['chains']} x ({m['warmup']}+{m['samples']}), {m['thin']} | {f(m['acceptance'], 3)} | {f(m['final_step'], 3)} | "
            f"{per_chain_evals(m):.0f} | {f(m['phi_mean'], 1)} ({f(m['phi_map'], 1)}, {m['n_data']}) | "
            f"{f(m['phi_first_quarter_mean'], 1)} / {f(m['phi_last_quarter_mean'], 1)} |{prime} {ess_cell(m['ess_phi'], m)} | "
            f"{ess_cell(None if m['ess_phi_per_eval'] is None else 1e3 * m['ess_phi_per_eval'], m, 3)} | "
            f"{ess_cell(m['ess_pixels_min'], m)} | {ess_cell(1e3 * m['ess_pixmin_per_eval'], m, 3)} | "
            f"{ess_cell(1e3 * m['ess_pixmin_per_eval_incl_setup'], m, 3)} | {ess_cell(m['ess_pixels_min_geyer'], m)} | "
            f"{f(m['rhat_phi'], 2)} | {f(m['rhat_pixels_max'], 2)} | {f(m['sampling_wall_s'], 0)} ({f(load, 0)}) |")
    lines.append("")
    if any(m.get("operating_point") for m in ms):
        lines.append("Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| "
                     "against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of "
                     "the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).\n")
        lines.append("| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r, m in zip(args.runs, ms):
            op = m.get("operating_point")
            if op is None:
                continue
            lines.append(f"| {r} | {m['n']}, {m['side']:g}, {m['data_n']} | {100 * op['rel_model_error']:.2f} | {100 * op['rel_noise']:.2f} | "
                         f"{op['model_to_noise']:.2f} | {f(op['phi_truth'], 0)} | {f(op['phi_clean'], 0)} | {f(op['phi_excess_sd'], 1)} |")
        lines.append("")
    lines.append(f"**Reconstruction and uncertainty ({args.group}).** err_model is eq. 15.21 on eps_r (vacuum = "
                 f"{f(ms[0]['err_model']['vacuum'], 4)}); err_chi is contrast-normalized; err_data is eq. 15.22. "
                 "z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std "
                 f"sigma = {ms[0]['sigma']:g}. P(eps<1) is averaged over true-background pixels (prior value 0.5)."
                 + (" 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; "
                    "'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well "
                    "below 1 means the sampler has not finished spreading out." if any_lap else "") + "\n")
    head = ("| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | "
            "std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | "
            "frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg |"
            + (" Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |" if any_lap else ""))
    lines.append(head)
    lines.append("|" + "---|" * (head.count("|") - 1))
    for r, m in zip(args.runs, ms):
        lap = ""
        if any_lap:
            if "mcmc_over_laplace_std_mean" in m:
                lap = (f" {f(m['laplace_std_theta_inside'], 4)} / {f(m['laplace_std_theta_outside'], 4)} | "
                       f"{f(m['mcmc_over_laplace_std_mean'], 3)} ({f(m['mcmc_over_laplace_std_min'], 3)}) |")
            else:
                lap = " n/a | n/a |"
        share = m.get("frac_abs_z_gt_3_edge_share")
        lines.append(
            f"| {r} | {f(m['err_model']['map'], 4)} | {f(m['err_model']['posterior_mean'], 4)} | {f(m['err_chi']['map'], 3)} | "
            f"{f(m['err_chi']['posterior_mean'], 3)} | {f(m['err_data']['map'], 4)} | {f(m['err_data']['posterior_mean'], 4)} | "
            f"{f(m['std_theta_inside'], 4)} / {f(m['std_theta_outside'], 4)} | {f(m['std_eps_inside'], 4)} / {f(m['std_eps_outside'], 4)} | "
            f"{f(m['std_ratio_mean'], 3)} ({f(m['std_ratio_inside'], 3)} / {f(m['std_ratio_outside'], 3)}) | {f(m['frac_std_ratio_lt_0.5'], 2)} | "
            f"{f(m['frac_abs_z_gt_3'], 2)} / {f(m['frac_abs_z_gt_3_inside'], 2)} / {f(m['frac_abs_z_gt_3_outside'], 2)} | "
            f"{f(share, 2)}, {f(m.get('max_abs_z'), 1)} | {f(m['p_eps_lt_1_background'], 3)} |{lap}")
    lines.append("")
    for lab in args.pool:
        p = ROOT / "results" / f"{lab}_pooled.json"
        if p.exists():
            d = json.loads(p.read_text())
            lines.append(f"Pooled across initializations `{lab}` ({d['chains']} chains, runs {', '.join(d['runs'])}): split R-hat of Phi "
                         f"{f(d['rhat_phi'], 2)}; of log eps_r over pixels: median {f(d['rhat_pixels_median'], 2)}, max {f(d['rhat_pixels_max'], 2)}, "
                         f"fraction of pixels with R-hat < 1.1: {f(d['frac_pixels_rhat_lt_1.1'], 2)}.\n")
    (ROOT / "results" / "tables").mkdir(exist_ok=True)
    (ROOT / "results" / "tables" / f"{args.group}.md").write_text("\n".join(lines))
    print("\n".join(lines))

    # Phi traces (the data misfit), then Phi' traces for Laplace-whitened runs
    (ROOT / "results" / "figures").mkdir(exist_ok=True)
    chains = {r: np.load(ROOT / "results" / r / "chains.npz") for r in args.runs}

    def misfit(r, m):
        z = chains[r]
        return z["phi_physical"] if "phi_physical" in z.files else z["phi"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for r, m in zip(args.runs, ms):
        phi = misfit(r, m)
        x = (np.arange(phi.shape[1]) + 1) * m["phi_every"]
        ls = "-" if m["init"] == "map" else "--"  # dashed: started at the prior or overdispersed
        for c in range(phi.shape[0]):
            for ax in axes:
                ax.plot(x, phi[c], ls=ls, color=COLORS.get(m["sampler"], "k"), lw=0.9,
                        label=f"{m['sampler']} {m['init']}" if c == 0 else None, alpha=0.85)
    for ax in axes:
        ax.set_xlabel("post-warmup step")
        ax.set_ylabel("Phi (data misfit of stored draws)")
        ax.axhline(ms[0]["n_data"], color="gray", lw=0.6, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_title("Phi (log axis)")
    lo = min(np.min(misfit(r, m)) for r, m in zip(args.runs, ms))
    hi = ms[0]["phi_map"] + 400 if not any_prime else max(np.max(misfit(r, m)) for r, m in zip(args.runs, ms)) + 20
    axes[1].set_ylim(lo - 5, hi)
    axes[1].set_title("Phi (zoom)")
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{args.group}: data misfit Phi (dotted line: N = {ms[0]['n_data']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / f"{args.group}_phi.png", dpi=130)
    plt.close(fig)

    if any_prime:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for r, m in zip(args.runs, ms):
            if m["phi_prime_mean"] is None:
                continue
            tr = chains[r]["phi"]
            ls = "-" if m["init"] == "map" else "--"  # dashed: started at the prior or overdispersed
            for c in range(tr.shape[0]):
                for ax in axes:
                    ax.plot(np.arange(tr.shape[1]) + 1, tr[c], ls=ls, color=COLORS.get(m["sampler"], "k"), lw=0.9,
                            label=f"{m['sampler']} {m['init']}" if c == 0 else None, alpha=0.85)
        for ax in axes:
            ax.set_xlabel("post-warmup step")
            ax.set_ylabel("Phi' = Phi + |xi|^2/2 - |zeta|^2/2")
        axes[0].set_title("sampler potential Phi' (NOT the data misfit)")
        prime_min = min(np.min(chains[r]["phi"]) for r, m in zip(args.runs, ms) if m["phi_prime_mean"] is not None)
        axes[1].set_yscale("log" if prime_min > 0 else "symlog")
        axes[1].set_title("Phi' (log axis)")
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle(f"{args.group}: reparametrized potential of the Laplace-whitened sampler", fontsize=9)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "figures" / f"{args.group}_phiprime.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
