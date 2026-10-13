"""Job file of the milestone-6 review follow-up (one ``batch_run.py`` batch, one XLA thread per job).

    python scripts/make_m6_review_jobs.py [--out scripts/jobs/m6_review.json] [--no-links]

Jobs, in the order they are started (longest first, so that nine slots finish together):

* ``truth``: the truth-start chain of the Matern spot checks (eps_r 2, g = 0.025 and 0.1) again, in the run directory ``..._truthfrozen`` with the step
  of the four overdispersed chains FROZEN (no adaptation during the transient: the first truth chains adapted to 0.108 and 0.098 against 0.16 to 0.21 and
  accepted 0.88 to 0.93) and 400 draws, so that the last 200 can be compared with the overdispersed chains after the relaxation (``--tail-segments 2``). The
  four overdispersed chains of the original run are linked into that directory (relative symbolic links, created here unless ``--no-links``).
  **This did not work** (kept for the record): from the truth the frozen step 0.17 is unstable, a 3-trajectory probe accepted none and spent 41 s per
  trajectory (3 to 8 s normally), and the chain made no progress in 110 CPU minutes; the step that the transient forces (0.108 and 0.098 in the first runs) is not an
  artifact that a bulk step can replace. A warmup that outlasts the transient (several hundred adapting trajectories) is the other option of the review and
  was not affordable.
* ``tv_od2``: TV at g = 0.1 with overdispersed starts (init scale 2; the first runs used 0.5, below 1) and 200 draws per chain, run directory ``..._od2``.
* ``tv_g015`` / ``matern_g015``: a configuration that BOTH priors resolve under the Laplace approximation (g = 0.15, eps_r 2), four chains each.
* ``lam``: the TV weight sensitivity (lambda* / 3 and 3 lambda*) at eps_r 1.1 and 4 for g = 0.05, 0.1, 0.15, each followed by the region-definition refeature
  of its own configurations.
* ``snr``: the image-domain threshold at 20 and 40 dB (eps_r 2, g = 0.05, 0.1, 0.15, both priors; TV at the 30 dB weight 0.5) in results/resolution/snr<dB>.
* ``mle``: the Monte Carlo maximum-likelihood check of the Cramer-Rao required SNR (``resolution_mle.py``): chunks of realizations at the required SNR of
  eps_r 2 for g = 0.05 and 0.1, and 10 dB above it for g = 0.05.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCMC_DIR = ROOT / "results" / "resolution" / "mcmc"

# (point name, realizations, realizations per job) of resolution_mle.py (POINTS there)
MLE_POINTS = (("eps2_g0.05_req", 48, 8), ("eps2_g0.1_req", 48, 8), ("eps2_g0.05_plus10", 24, 8))


def median_step(name):
    """Median of the final (frozen) step sizes of the four overdispersed chains of the run <name> (its summary.json)."""
    steps = sorted(json.loads((MCMC_DIR / name / "summary.json").read_text())["sampler"]["final_step"])
    return 0.5 * (steps[len(steps) // 2 - 1] + steps[len(steps) // 2]) if len(steps) % 2 == 0 else steps[len(steps) // 2]


def link_chains(name, tag, n_chains=4):
    """Relative symbolic links from <name><tag>/ to the segment files of chains 0 .. n_chains - 1 of <name>/ (the overdispersed chains)."""
    dst = MCMC_DIR / (name + tag)
    dst.mkdir(parents=True, exist_ok=True)
    for src in sorted((MCMC_DIR / name).glob("chain_[0-9]_seg[0-9].npz")):
        c = int(src.name.split("_")[1])
        if c >= n_chains:
            continue
        link = dst / src.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(src, dst))


def job(name, cmd, marker, threads=1):
    return {"name": name, "cmd": cmd, "device": "cpu", "threads": threads, "done_marker": marker}


def truth_jobs(links=True):
    jobs = []
    for g, name in ((0.025, "g0.025_eps2_matern"), (0.1, "g0.1_eps2_matern")):
        step = median_step(name)
        if links:
            link_chains(name, "_truthfrozen")
        jobs.append(job(f"m6r_mcmc_{name}_truthfrozen",
                        ["scripts/resolution_mcmc.py", "--stage", "chain", "--g", f"{g:g}", "--eps-r", "2", "--prior", "matern", "--chain-id", "4", "--chains", "5",
                         "--start", "truth", "--frozen-step", f"{step:.6f}", "--run-tag", "_truthfrozen", "--first", "0", "--last", "3", "--samples", "100"],
                        f"results/resolution/mcmc/{name}_truthfrozen/chain_4_seg3.npz"))
    return jobs


def chain_jobs(prefix, g, prior, run_tag, last, samples, warmup, init_scale, chains=4):
    name = f"g{g:g}_eps2_{prior}"
    return [job(f"m6r_{prefix}_c{c}",
                ["scripts/resolution_mcmc.py", "--stage", "chain", "--g", f"{g:g}", "--eps-r", "2", "--prior", prior, "--chain-id", str(c), "--chains", str(chains),
                 "--init-scale", f"{init_scale:g}"] + (["--run-tag", run_tag] if run_tag else [])
                + ["--first", "0", "--last", str(last), "--samples", str(samples), "--warmup", str(warmup)],
                f"results/resolution/mcmc/{name}{run_tag}/chain_{c}_seg{last}.npz") for c in range(chains)]


def lambda_jobs():
    jobs = []
    for eps in (1.1, 4.0):
        for scale in ("0.3333333333", "3"):
            tag = "x0.333333" if scale != "3" else "x3"
            jobs.append(job(f"m6r_lam_eps{eps:g}_{tag}",
                            ["scripts/resolution_laplace.py", "--stage", "descend", "--eps-r", f"{eps:g}", "--prior", "tv", "--lam-scale", scale,
                             "--gaps", "0.15", "0.1", "0.05", "--then-refeature"],
                            f"results/resolution/laplace_sens/g0.05_eps{eps:g}_tv_{tag}.json"))
    return jobs


def snr_jobs():
    jobs = []
    for snr in (40, 20):
        out = f"results/resolution/snr{snr}"
        for prior in ("matern", "tv"):
            cmd = ["scripts/resolution_laplace.py", "--stage", "descend", "--eps-r", "2", "--prior", prior, "--gaps", "0.15", "0.1", "0.05", "--snr", str(snr),
                   "--out-dir", out, "--then-refeature"]
            if prior == "tv":
                cmd += ["--lam", "0.5"]
            jobs.append(job(f"m6r_snr{snr}_{prior}", cmd, f"{out}/laplace_sens/g0.05_eps2_{prior}.json"))
    return jobs


def mle_jobs():
    jobs = []
    for point, n_reals, per_job in MLE_POINTS:
        for a in range(0, n_reals, per_job):
            b = min(a + per_job, n_reals)
            jobs.append(job(f"m6r_mle_{point}_{a}",
                            ["scripts/resolution_mle.py", "--stage", "chunk", "--point", point, "--first", str(a), "--count", str(b - a)],
                            f"results/resolution/mle/{point}/chunk_{a}_{b}.json"))
    return jobs


def finalize_jobs():
    """Analyses of the new chains, one job each (``resolution_mcmc.py --stage finalize`` physical Phi of every draw: a few minutes), then the Phi' criteria for the old summaries
    (``--stage energy``, no forward solves) and the feature probabilities for other strips from the stored draws (``--stage refeature``)."""
    def fin(name, g, prior, chains, extra, marker):
        return job(f"m6r_fin_{name}", ["scripts/resolution_mcmc.py", "--stage", "finalize", "--g", f"{g:g}", "--eps-r", "2", "--prior", prior, "--chains", str(chains)] + extra,
                   marker)

    base = "results/resolution/mcmc"
    return [fin("g0.025_matern_truthfrozen", 0.025, "matern", 5, ["--run-tag", "_truthfrozen", "--tail-segments", "2"], f"{base}/g0.025_eps2_matern_truthfrozen/summary.json"),
            fin("g0.1_matern_truthfrozen", 0.1, "matern", 5, ["--run-tag", "_truthfrozen", "--tail-segments", "2"], f"{base}/g0.1_eps2_matern_truthfrozen/summary.json"),
            fin("g0.1_tv_od2", 0.1, "tv", 4, ["--run-tag", "_od2"], f"{base}/g0.1_eps2_tv_od2/summary.json"),
            fin("g0.15_tv", 0.15, "tv", 4, [], f"{base}/g0.15_eps2_tv/summary.json"),
            fin("g0.15_matern", 0.15, "matern", 4, [], f"{base}/g0.15_eps2_matern/summary.json")]


def all_jobs(links=True):
    return (truth_jobs(links)
            + chain_jobs("tv_g0.1_od2", 0.1, "tv", "_od2", 3, 50, 50, 2.0)
            + chain_jobs("tv_g0.15", 0.15, "tv", "", 2, 50, 50, 2.0)
            + chain_jobs("matern_g0.15", 0.15, "matern", "", 1, 60, 60, 2.0)
            + lambda_jobs() + snr_jobs() + mle_jobs())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "scripts" / "jobs" / "m6_review.json"))
    ap.add_argument("--finalize", action="store_true", help="write the job file of the finalize batch (scripts/jobs/m6_review_finalize.json) instead")
    ap.add_argument("--no-links", action="store_true", help="do not create the symbolic links of the overdispersed chains in the truth-start run directories")
    args = ap.parse_args(argv)
    if args.finalize:
        jobs = finalize_jobs()
        if args.out == str(ROOT / "scripts" / "jobs" / "m6_review.json"):
            args.out = str(ROOT / "scripts" / "jobs" / "m6_review_finalize.json")
        Path(args.out).write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"wrote {args.out}: {len(jobs)} jobs")
        return jobs
    jobs = all_jobs(links=not args.no_links)
    Path(args.out).write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"wrote {args.out}: {len(jobs)} jobs")
    return jobs


if __name__ == "__main__":
    main()
