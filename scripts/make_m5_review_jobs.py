"""Write the ONE batch_run job file of the milestone-5 review follow-up (docs/DESIGN.md 12.6).

    python scripts/make_m5_review_jobs.py --out scripts/jobs/m5_review_batch.json
    python scripts/batch_run.py scripts/jobs/m5_review_batch.json --cpu-slots 17 --threads 1 --log-dir results/prior_comparison/logs

The review of the milestone-5 report found (i) that the discrepancy principle hardly constrains the TV weight (Phi_MAP 2372 to 2458 over the
whole sweep, every value but the last inside N +- sqrt(N)), so the edge statistics must be shown as a function of lambda across the band, and
(ii) that no TV chain started from an overdispersed state. This batch adds

* two TV posteriors, lambda = 1.25 (lambda* / 3) and lambda = 8.5 (Phi_MAP = 2438, the Phi of noise-free data, by log-log interpolation of
  ``tv_lambda_sweep.json``): ``pc_tv_lam1.25_adapt`` and ``pc_tv_lam8.5_adapt``, each with its own prepare / adapt stage (the schedule of the
  first-attempt TV runs: windows of 50 and 100 trajectories, kappa = 0.25 N, trajectory length 1.0) and four chains that start at four
  DIFFERENT maps (truth, MAP, Matern mean, GMRF draw; ``scripts/make_overdispersed_starts.py``), 50 step-size warmup + ``--segments`` x 100
  recorded trajectories each;
* an independent-start replication of lambda* in a NEW directory ``pc_tv_lam3.75_indep``: two chains in the frozen map of
  ``pc_tv_lam3.75_adapt`` (the same target), started at the truth and at a GMRF draw, 50 warmup + 200 recorded trajectories.

Dependencies by waiting jobs (``scripts/wait_then.py``): a waiting job sleeps and uses no CPU, so the number of XLA threads that compute is
set by the structure: prepare (2 x 4 threads) with the replication chains (2 x 1) = 10; adapt (2 x 4) with the replication chains = 10; the
eight new chains with the replication chains = 10. ``scripts/jobs/stop/m5v`` ends all chain loops before their next segment (a running
segment finishes): the budget switch.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_prior_jobs as mp  # noqa: E402

RESULTS = mp.RESULTS
LAMBDAS = (1.25, 8.5)
TV_ADAPT = ["--adapt-init", "50", "--adapt-first-window", "50", "--adapt-windows", "2", "--adapt-growth", "2", "--adapt-final", "50",
            "--adapt-kappa-per-dim", "0.25", "--traj-length", "1.0"]
NEW_KINDS = ("truth", "map", "matern", "gmrf")
INDEP_KINDS = ("truth", "gmrf")
STOP = "scripts/jobs/stop/m5v"


def problem(chains=4):
    """``mp.PROBLEM`` with ``--chains`` set to ``chains``."""
    p = list(mp.PROBLEM)
    p[p.index("--chains") + 1] = str(chains)
    return p


def run_name(lam):
    """The run directory name of the new TV posterior with weight ``lam``."""
    return f"pc_tv_lam{lam:g}_adapt"


def waiting(cmd, files=()):
    """``cmd`` (argument list starting with a script path) behind ``scripts/wait_then.py`` waiting for ``files``."""
    out = ["scripts/wait_then.py"]
    for f in files:
        out += ["--file", str(f)]
    return out + ["--", "python", *cmd]


def loop_cmd(out_dir, chain, last, samples, warmup, common):
    """The ``segment_loop.py`` command of one chain (segments 0..last)."""
    return ["scripts/segment_loop.py", "--out-dir", str(out_dir), "--chain-id", str(chain), "--first", "0", "--last", str(last),
            "--samples", str(samples), "--warmup", str(warmup), "--stop", STOP, "--", *common]


def build(segments=3, indep_segments=2):
    """All jobs of the batch (module docstring) as batch_run job dicts."""
    jobs = []
    for lam in LAMBDAS:
        run = run_name(lam)
        d = f"{RESULTS}/{run}"
        name = f"tv_lam{lam:g}_adapt"
        base = ["scripts/run_prior_comparison.py", *problem(), *mp.prior_args(name), "--out", run]
        jobs.append({"name": f"m5v_prep_lam{lam:g}", "cmd": [*base, "--stage", "prepare"], "device": "cpu", "threads": 4,
                     "done_marker": f"{d}/prepared.npz"})
        jobs.append({"name": f"m5v_starts_lam{lam:g}",
                     "cmd": waiting(["scripts/make_overdispersed_starts.py", "--run", run, "--kinds", *NEW_KINDS], [f"{d}/prepared.npz"]),
                     "device": "cpu", "threads": 1, "done_marker": f"{d}/starts.npy"})
        jobs.append({"name": f"m5v_adapt_lam{lam:g}", "cmd": waiting([*base, "--stage", "adapt", *TV_ADAPT], [f"{d}/prepared.npz"]),
                     "device": "cpu", "threads": 4, "done_marker": f"{d}/adaptation.json"})
        common = [*problem(), *mp.prior_args(name), "--out", run, "--start-theta", f"{d}/starts.npy"]
        for c in range(4):
            jobs.append({"name": f"m5v_lam{lam:g}_c{c}",
                         "cmd": waiting(loop_cmd(d, c, segments - 1, 100, 50, common), [f"{d}/adaptation.json", f"{d}/starts.npy"]),
                         "device": "cpu", "threads": 1, "done_marker": f"{d}/chain_{c}_seg{segments - 1}.npz"})
    # independent-start replication of lambda* in the frozen map of pc_tv_lam3.75_adapt
    d = f"{RESULTS}/pc_tv_lam3.75_indep"
    jobs.append({"name": "m5v_indep_starts",
                 "cmd": ["scripts/make_overdispersed_starts.py", "--run", "pc_tv_lam3.75_indep", "--copy-from", "pc_tv_lam3.75_adapt",
                         "--kinds", *INDEP_KINDS], "device": "cpu", "threads": 1, "done_marker": f"{d}/starts.npy"})
    common = [*problem(len(INDEP_KINDS)), *mp.prior_args("tv_lam3.75_adapt"), "--out", "pc_tv_lam3.75_indep", "--start-theta", f"{d}/starts.npy"]
    for c in range(len(INDEP_KINDS)):
        jobs.append({"name": f"m5v_indep_c{c}",
                     "cmd": waiting(loop_cmd(d, c, indep_segments - 1, 100, 50, common), [f"{d}/starts.npy"]),
                     "device": "cpu", "threads": 1, "done_marker": f"{d}/chain_{c}_seg{indep_segments - 1}.npz"})
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--segments", type=int, default=3, help="segments of 100 recorded trajectories per chain of the new lambda runs")
    ap.add_argument("--indep-segments", type=int, default=2, help="segments of 100 recorded trajectories per replication chain")
    args = ap.parse_args(argv)
    jobs = build(args.segments, args.indep_segments)
    Path(args.out).write_text(json.dumps(jobs, indent=2) + "\n")
    Path("scripts/jobs/stop").mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(jobs)} jobs to {args.out}")
    return jobs


if __name__ == "__main__":
    main()
