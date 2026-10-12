"""Write a scripts/batch_run.py job file for one stage of the prior-comparison experiment (DESIGN 11.3).

    python scripts/make_prior_jobs.py prepare --out scripts/jobs/prior_comparison_prepare.json
    python scripts/make_prior_jobs.py chain --segment 0 --warmup 300 --samples 400 --out scripts/jobs/pc_seg0.json
    python scripts/make_prior_jobs.py chain --segment 1 --samples 400 --priors gmrf qggmrf_sg0.05 --out ...
    python scripts/make_prior_jobs.py adapt --priors tv_lam4.5_adapt gmrf_adapt --out ...   (milestone 5, see below)

The problem is the DESIGN 11.3 one (austria_fixed eps_r 2, n = 48, side 2, 30 dB, data seed 0, 4 chains) and the four
priors are ``matern``, ``gmrf``, ``qggmrf_sg0.05`` and ``qggmrf_sg0.1`` (run directories results/prior_comparison/pc_<name>).
``prepare`` makes one job per prior (data, MAP, Laplace, prior std; stage prepare of run_prior_comparison.py);
``adapt`` one job per adapted prior (stage adapt, after prepare and before the chains);
``chain`` makes one job per (prior, chain) for the given ``--segment`` (stage chain); ``finalize`` one job per prior.

Milestone 5 priors: ``tv_lam<value>`` is the pseudo-Huber TV prior with that lambda (delta 0.01, sigma_0 1, --map-maxiter 4000; run directory
results/prior_comparison/pc_tv_lam<value>), and a name ending in ``_adapt`` (``gmrf_adapt``, ``tv_lam4.5_adapt``) is the same prior
sampled in the coordinates of an adapted affine map (``--adapt``, DESIGN 12.2); its run directory is pc_<name>, so the milestone-4
directories are never touched. Only the ``_adapt`` priors have an ``adapt`` job.
Every job is single-threaded by default (``--threads``), which is the most efficient use of the cores for this
problem (the cost of one value-and-gradient evaluation falls only 2.2-fold from 1 to 4 threads, results/prior_comparison/
cost.json). Run a file with

    python scripts/batch_run.py FILE --cpu-slots K --threads 1 --log-dir results/prior_comparison/logs
"""

import argparse
import json
import re
from pathlib import Path

PROBLEM = ["--n", "48", "--side", "2", "--eps-r", "2", "--snr", "30", "--seed", "0", "--chains", "4"]
PRIORS = {
    "matern": ["--prior", "matern", "--sigma", "0.5", "--ell", "0.4", "--nu", "1.5"],
    "gmrf": ["--prior", "gmrf", "--sigma", "0.5", "--ell", "0.4"],
    "qggmrf_sg0.05": ["--prior", "qggmrf", "--q", "1.2", "--T", "1", "--sigma-0", "1", "--sigma-g", "0.05"],
    "qggmrf_sg0.1": ["--prior", "qggmrf", "--q", "1.2", "--T", "1", "--sigma-0", "1", "--sigma-g", "0.1"],
}
RESULTS = "results/prior_comparison"
TV_NAME = re.compile(r"tv_lam([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)")


def prior_args(name):
    """The prior arguments of run_prior_comparison.py for a prior name: one of ``PRIORS``, ``tv_lam<value>`` (pseudo-Huber TV,
    delta 0.01, sigma_0 1, up to 4000 MAP iterations), each optionally followed by ``_adapt`` (adds --adapt). Raises ValueError for
    an unknown name."""
    adapt = name.endswith("_adapt")
    base = name[:-len("_adapt")] if adapt else name
    if base in PRIORS:
        args = list(PRIORS[base])
    elif TV_NAME.fullmatch(base):
        # the TV MAP needs 1000 to 1500 L-BFGS-B iterations (results/prior_comparison/tv_lambda_sweep.json), not the default 400
        args = ["--prior", "tv", "--lam", TV_NAME.fullmatch(base).group(1), "--delta", "0.01", "--sigma-0", "1",
                "--map-maxiter", "4000"]
    else:
        raise ValueError(f"unknown prior {name!r}: one of {sorted(PRIORS)}, tv_lam<value>, each optionally with _adapt")
    return args + (["--adapt"] if adapt else [])


def jobs(stage, priors, segment, warmup, samples, threads, extra, init_scale, prefix="pc_", chain_ids=(0, 1, 2, 3)):
    """The list of job dicts for ``stage`` over ``priors`` (names of ``PRIORS``)."""
    out = []
    for name in priors:
        run = prefix + name
        base = ["scripts/run_prior_comparison.py", *PROBLEM, *prior_args(name), "--out", run]
        if name in init_scale:
            base += ["--init-scale", str(init_scale[name])]
        base += extra
        if stage == "prepare":
            out.append({"name": f"{run}_prepare", "cmd": base + ["--stage", "prepare"], "device": "cpu",
                        "threads": threads, "done_marker": f"{RESULTS}/{run}/prepared.npz"})
        elif stage == "adapt":
            if name.endswith("_adapt"):
                out.append({"name": f"{run}_adaptation", "cmd": base + ["--stage", "adapt"], "device": "cpu", "threads": threads,
                            "done_marker": f"{RESULTS}/{run}/adaptation.json"})
        elif stage == "chain":
            for c in chain_ids:
                cmd = base + ["--stage", "chain", "--chain-id", str(c), "--segment", str(segment), "--samples",
                              str(samples)] + (["--warmup", str(warmup)] if segment == 0 else [])
                out.append({"name": f"{run}_c{c}_s{segment}", "cmd": cmd, "device": "cpu", "threads": threads,
                            "done_marker": f"{RESULTS}/{run}/chain_{c}_seg{segment}.npz"})
        else:
            out.append({"name": f"{run}_finalize", "cmd": base + ["--stage", "finalize"], "device": "cpu",
                        "threads": threads, "done_marker": f"{RESULTS}/{run}/summary.json"})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["prepare", "adapt", "chain", "finalize"])
    ap.add_argument("--priors", nargs="+", default=list(PRIORS), help="prior names (see the module docstring)")
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=400, help="trajectories per chain in this segment")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--prefix", default="pc_", help="run directory prefix (pilots use another one)")
    ap.add_argument("--chain-ids", type=int, nargs="*", default=[0, 1, 2, 3], help="chain stage: which chains")
    ap.add_argument("--init-scale", nargs="*", default=[], metavar="PRIOR=SCALE",
                    help="chain start scale per prior, e.g. gmrf=0.5 (default 2, the DESIGN 11.2 value)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further arguments passed to every job")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    scales = dict(kv.split("=") for kv in args.init_scale)
    try:
        for name in [*args.priors, *scales]:
            prior_args(name)
    except ValueError as err:
        ap.error(str(err))
    unknown = set(scales) - set(args.priors)
    if unknown:
        ap.error(f"priors in --init-scale that are not in --priors: {sorted(unknown)}")
    js = jobs(args.stage, args.priors, args.segment, args.warmup, args.samples, args.threads, args.extra, scales,
               args.prefix, args.chain_ids)
    Path(args.out).write_text(json.dumps(js, indent=2) + "\n")
    print(f"wrote {len(js)} jobs to {args.out}")
    return js


if __name__ == "__main__":
    main()
