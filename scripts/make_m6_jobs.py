"""Write the ``scripts/batch_run.py`` job files of milestone 6 (DESIGN 13): the Laplace pixel posteriors and the MCMC spot checks.

    python scripts/make_m6_jobs.py [--batch laplace1|laplace2|sens|mcmc] [--out-dir scripts/jobs] [--results results/resolution]

Batches (run ONE at a time, T = 1 thread per job, at most --slots jobs at once so that the XLA threads in total stay at 10 or fewer):

  laplace1   the lambda sweeps (one job per contrast, 9 weights each at the reference gap) and, per contrast, the Matern chain over the gaps
  laplace2   per contrast the TV chain over the gaps at lambda*(eps_r) (needs laplace_lambda.json: ``resolution_laplace.py --stage lambda-collect``
             and the Matern configurations)
  sens       the TV chains at lambda*/3 and 3 lambda* for eps_r = 2 (every gap)
  mcmc       the spot-check chains: one job per chain, a loop of checkpointed segments (``resolution_mcmc.py``)

A job whose done marker exists is skipped by ``batch_run.py``; every job is also idempotent on its own (the scripts skip finished work).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAPS = (0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.32, 0.5)
EPS = (1.1, 2.0, 4.0)
LAM_MAX = 50.0  # the last weight of the sweep: its JSON marks the sweep job of one contrast as done


def laplace_jobs(batch, results="results/resolution"):
    """The jobs of one batch. A Laplace job is one (eps_r, prior) chain over the gaps from the widest to the narrowest (``--stage descend``), so that every
    configuration can start from the MAP of the next wider gap; the done marker is the JSON of the narrowest gap."""
    jobs = []
    if batch == "laplace1":
        for eps in EPS:
            jobs.append({"name": f"m6_lambda_eps{eps:g}", "cmd": ["scripts/resolution_laplace.py", "--stage", "lambda", "--eps", f"{eps:g}"],
                         "device": "cpu", "threads": 1, "done_marker": f"{results}/lambda/eps{eps:g}_g0.25_lam{LAM_MAX:g}.json"})
        jobs += [_descend_job(eps, "matern", 1.0, results) for eps in EPS]
    elif batch == "laplace2":
        jobs += [_descend_job(eps, "tv", 1.0, results) for eps in EPS]
    elif batch == "sens":
        jobs += [_descend_job(2.0, "tv", scale, results) for scale in (3.0, 1.0 / 3.0)]
    else:
        raise ValueError(batch)
    return jobs


def mcmc_jobs(configs, chains=4, segments=4, samples=100, warmup=100, results="results/resolution"):
    """One job per chain of every spot-check configuration: ``resolution_mcmc.py --stage chain`` over ``segments`` segments of ``samples`` recorded
    trajectories (segment 0 adds ``warmup`` adaptation trajectories); the done marker is the last segment of the chain. A configuration is
    ``(g, eps_r, prior)`` or ``(g, eps_r, prior, segments, samples, warmup)`` to override the common sizes for that configuration."""
    jobs = []
    for cfg in configs:
        g, eps, prior = cfg[:3]
        segments_c, samples_c, warmup_c = cfg[3:] if len(cfg) == 6 else (segments, samples, warmup)
        name = stem(g, eps, prior)
        for c in range(chains):
            cmd = ["scripts/resolution_mcmc.py", "--stage", "chain", "--g", f"{g:g}", "--eps-r", f"{eps:g}", "--prior", prior, "--chain-id", str(c),
                   "--chains", str(chains), "--first", "0", "--last", str(segments_c - 1), "--samples", str(samples_c), "--warmup", str(warmup_c)]
            jobs.append({"name": f"m6_mcmc_{name}_c{c}", "cmd": cmd, "device": "cpu", "threads": 1,
                         "done_marker": f"{results}/mcmc/{name}/chain_{c}_seg{segments_c - 1}.npz"})
    return jobs


def _scale_arg(scale):
    return f"{scale:.10g}"


def stem(g, eps, prior, scale=1.0):
    """The file stem of a configuration (``resolution_laplace.config_name``)."""
    from math import isclose

    return f"g{g:g}_eps{eps:g}_{prior}" + ("" if prior != "tv" or isclose(scale, 1.0) else f"_x{scale:g}")


def _descend_job(eps, prior, scale, results):
    from math import isclose

    cmd = ["scripts/resolution_laplace.py", "--stage", "descend", "--eps-r", f"{eps:g}", "--prior", prior]
    if prior == "tv" and not isclose(scale, 1.0):
        cmd += ["--lam-scale", _scale_arg(scale)]
    name = f"m6_descend_eps{eps:g}_{prior}" + ("" if prior != "tv" or isclose(scale, 1.0) else f"_x{scale:g}")
    return {"name": name, "cmd": cmd, "device": "cpu", "threads": 1, "done_marker": f"{results}/laplace/{stem(min(GAPS), eps, prior, scale)}.json"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", choices=["laplace1", "laplace2", "sens", "mcmc"], nargs="+", default=["laplace1", "laplace2", "sens"])
    ap.add_argument("--out-dir", default=str(ROOT / "scripts" / "jobs"))
    ap.add_argument("--results", default="results/resolution")
    ap.add_argument("--mcmc-configs", nargs="+", default=[], metavar="G:EPS:PRIOR[:SEGMENTS:SAMPLES:WARMUP]",
                    help="batch mcmc: the spot-check configurations, e.g. 0.05:2:matern or 0.05:2:tv:2:50:50 (own sizes)")
    ap.add_argument("--mcmc-name", default="mcmc", help="batch mcmc: the job file is m6_<name>.json")
    ap.add_argument("--mcmc-chains", type=int, default=4)
    ap.add_argument("--mcmc-segments", type=int, default=4)
    ap.add_argument("--mcmc-samples", type=int, default=100)
    ap.add_argument("--mcmc-warmup", type=int, default=100)
    args = ap.parse_args(argv)
    for batch in args.batch:
        if batch == "mcmc":
            cfgs = []
            for spec in args.mcmc_configs:
                f = spec.split(":")
                cfgs.append((float(f[0]), float(f[1]), f[2]) + (tuple(int(x) for x in f[3:]) if len(f) == 6 else ()))
            if not cfgs:
                ap.error("--batch mcmc needs --mcmc-configs")
            jobs = mcmc_jobs(cfgs, args.mcmc_chains, args.mcmc_segments, args.mcmc_samples, args.mcmc_warmup, args.results)
            path = Path(args.out_dir) / f"m6_{args.mcmc_name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(jobs, indent=2) + "\n")
            print(f"wrote {path} ({len(jobs)} jobs)")
            continue
        jobs = laplace_jobs(batch, args.results)
        path = Path(args.out_dir) / f"m6_{batch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"wrote {path} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
