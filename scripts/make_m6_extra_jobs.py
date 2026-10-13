"""Job file of the milestone-6 follow-up runs (one ``batch_run.py`` batch, one XLA thread per job).

    python scripts/make_m6_extra_jobs.py [--out scripts/jobs/m6_extra.json]

* ``null``: ``resolution_laplace.py --stage null`` for every contrast and both priors (the posterior of a CLOSED gap scored with the open-gap strips of the sweep;
  every start, with the stored g = 0.025 MAP as the continuation start), 6 jobs.
* ``truth``: the fifth chain of the DESIGN 13.3 spot checks, started at the pixelized truth (``resolution_mcmc.py --stage chain --chains 5 --chain-id 4
  --start truth``), with the sampler settings of the four overdispersed chains of the same configuration (``make_m6_jobs.py``), 4 jobs.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPS = (1.1, 2.0, 4.0)
PRIORS = ("matern", "tv")
# (g, eps_r, prior, segments, samples, warmup): the settings of the four overdispersed chains in results/resolution/mcmc/<name>/summary.json
TRUTH_CHAINS = ((0.025, 2.0, "matern", 3, 100, 60), (0.1, 2.0, "matern", 2, 100, 60), (0.025, 2.0, "tv", 4, 50, 50), (0.1, 2.0, "tv", 1, 50, 50))


def null_jobs():
    jobs = []
    for eps in EPS:
        for prior in PRIORS:
            jobs.append({"name": f"m6_null_eps{eps:g}_{prior}",
                         "cmd": ["scripts/resolution_laplace.py", "--stage", "null", "--eps-r", f"{eps:g}", "--prior", prior, "--neighbor-gap", "0.025", "--all-starts"],
                         "device": "cpu", "threads": 1, "done_marker": f"results/resolution/null/g0_eps{eps:g}_{prior}.json"})
    return jobs


def truth_jobs():
    jobs = []
    for g, eps, prior, segments, samples, warmup in TRUTH_CHAINS:
        name = f"g{g:g}_eps{eps:g}_{prior}"
        jobs.append({"name": f"m6_mcmc_{name}_truth",
                     "cmd": ["scripts/resolution_mcmc.py", "--stage", "chain", "--g", f"{g:g}", "--eps-r", f"{eps:g}", "--prior", prior, "--chain-id", "4", "--chains", "5",
                             "--start", "truth", "--first", "0", "--last", str(segments - 1), "--samples", str(samples), "--warmup", str(warmup)],
                     "device": "cpu", "threads": 1, "done_marker": f"results/resolution/mcmc/{name}/chain_4_seg{segments - 1}.npz"})
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "scripts" / "jobs" / "m6_extra.json"))
    args = ap.parse_args(argv)
    jobs = null_jobs() + truth_jobs()
    Path(args.out).write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"wrote {args.out}: {len(jobs)} jobs")
    return jobs


if __name__ == "__main__":
    main()
