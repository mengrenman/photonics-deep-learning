"""Write the one batch_run job file of milestone 5 (docs/DESIGN.md section 12): preparation, adaptation, chains and extensions of the
posteriors TV at lambda* (tv_lam3.75_adapt), TV at the neighbor 3 lambda* (tv_lam11.25_adapt) and the GMRF with an adapted map (gmrf_adapt).

    python scripts/make_m5_batch.py --out scripts/jobs/m5_batch1.json
    python scripts/batch_run.py scripts/jobs/m5_batch1.json --cpu-slots 40 --threads 1 --log-dir results/prior_comparison/logs

All jobs are listed up front and gated by ``scripts/wait_then.py``: batch_run has no dependencies, so each later job sleeps until its inputs
exist (a prepare job's meta.json says ok, an adaptation.json exists, a segment file exists) and, for chains and extensions, until a go file
``scripts/jobs/go/<tag>`` exists. The go files are created by hand after the previous stage has been looked at; they are also how the
10-XLA-thread cap is kept (this file does not enforce it): the sum of the ``threads`` of the jobs that are computing at any time must stay
at most 10. ``scripts/jobs/skip/<tag>`` cancels a gated job without killing the batch.

Settings (the choices of 12.2 that the DESIGN leaves open are stated in docs/DESIGN.md 12.4 and 12.5):

* TV: a short adaptation chain (50 step-size trajectories, windows of 50 and 100, 50 step-size trajectories; the map is the Laplace map of the
  MAP, re-centered and re-estimated twice with shrinkage kappa = 0.25 N), trajectory length 1.0 in whitened units, then 4 chains x (50
  step-size warmup + 400 recorded trajectories), starts N(0, I) around the final adaptation state (--init-scale 1.0), extension segments of 300.
* GMRF: the adaptation chain starts AT the end state of chain 0 of the milestone-4 run pc_gmrf and the initial map is the Laplace map there (a
  state far from the MAP that the milestone-4 chain had reached in 600 trajectories, instead of a transient of several hundred trajectories from the
  MAP); 3 windows (50, 100, 200), kappa = 0.25 N, trajectory length 1.5; the four chains start AT the four milestone-4 end states
  (--start-theta), 50 step-size warmup + 500 recorded trajectories, extension segments of 500.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_prior_jobs as mp  # noqa: E402

RESULTS = mp.RESULTS
GMRF_DIR = f"{RESULTS}/pc_gmrf_adapt"
PYTHON = "python"
TV_ADAPT = ["--adapt-init", "50", "--adapt-first-window", "50", "--adapt-windows", "2", "--adapt-growth", "2", "--adapt-final", "50",
            "--adapt-kappa-per-dim", "0.25", "--traj-length", "1.0"]
GMRF_ADAPT = ["--adapt-init", "50", "--adapt-first-window", "50", "--adapt-windows", "3", "--adapt-growth", "2", "--adapt-final", "100",
              "--adapt-kappa-per-dim", "0.25", "--traj-length", "1.5", "--mass-theta", f"{GMRF_DIR}/m4_end0.npy",
              "--adapt-start-scale", "0"]
SETUPS = {
    # name: (adapt-stage extras, chain-stage extras, warmup, samples seg 0, samples per extension, chain groups (tag, chain ids, threads))
    "tv_lam3.75_adapt": (TV_ADAPT, ["--init-scale", "1.0"], 50, 400, 300, [("tvA", (0, 1, 2, 3))]),
    "tv_lam11.25_adapt": (TV_ADAPT, ["--init-scale", "1.0"], 50, 400, 300, [("tvB_01", (0, 1)), ("tvB_23", (2, 3))]),
    "gmrf_adapt": (GMRF_ADAPT, ["--start-theta", f"{GMRF_DIR}/m4_ends.npy"], 50, 500, 500, [("gmrf", (0, 1, 2, 3))]),
}


def gated(job, files=(), metas=(), go=None, skip=None):
    """``job`` (a batch_run job dict) with its command wrapped by wait_then.py."""
    pre = ["scripts/wait_then.py"]
    for f in files:
        pre += ["--file", f]
    for m in metas:
        pre += ["--meta-ok", m]
    if go:
        pre += ["--go", go]
    if skip:
        pre += ["--skip", skip]
    return {**job, "cmd": pre + ["--", PYTHON, *job["cmd"]]}


def build():
    out = []
    # 1. prepare: data, MAP, Laplace, prior std (TV: prior-only HMC); all at one thread
    for name in SETUPS:
        out += mp.jobs("prepare", [name], 0, 0, 0, 1, [], {})
    # 2. adaptation (GMRF at four threads: it is the longest sequential piece)
    for name, (adapt_extra, *_rest) in SETUPS.items():
        run = f"pc_{name}"
        j = mp.jobs("adapt", [name], 0, 0, 0, 4 if name == "gmrf_adapt" else 1, adapt_extra, {})[0]
        out.append(gated(j, files=[f"{RESULTS}/{run}/prepared.npz"], metas=[f"{run}_prepare"]))
    # 3. chains (segment 0) and the extension (segment 1), each behind a go file
    for name, (_a, chain_extra, warmup, samples, ext, groups) in SETUPS.items():
        run = f"pc_{name}"
        for tag, ids in groups:
            for seg, n_samp in ((0, samples), (1, ext)):
                for j in mp.jobs("chain", [name], seg, warmup, n_samp, 1, chain_extra, {}, chain_ids=ids):
                    c = int(j["name"].split("_c")[-1].split("_s")[0])
                    files = [f"{RESULTS}/{run}/adaptation.json"] + ([f"{RESULTS}/{run}/chain_{c}_seg0.npz"] if seg == 1 else [])
                    out.append(gated(j, files=files, go=f"scripts/jobs/go/{tag}_seg{seg}", skip=f"scripts/jobs/skip/{tag}_seg{seg}"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    js = build()
    Path(args.out).write_text(json.dumps(js, indent=2) + "\n")
    Path("scripts/jobs/go").mkdir(parents=True, exist_ok=True)
    Path("scripts/jobs/skip").mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(js)} jobs to {args.out}")
    return js


if __name__ == "__main__":
    main()
