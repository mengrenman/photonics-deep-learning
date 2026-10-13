"""Write the ``scripts/batch_run.py`` job files of milestone 7 (DESIGN 14): the posteriors of F2, F2u and Fe on the eight truths of the plan, and the pixel posteriors.

    python scripts/make_m7_jobs.py [--batch map-f2 map-rest chain finalize ti pixel] [--out-dir scripts/jobs] [--results results/parametric]

Run the batches ONE at a time, in this order, with ``--threads 1`` and at most ``--cpu-slots 8`` jobs at once (the XLA threads in total stay at or below 10):

  map-f2     the F2 MAP and Laplace fit of every truth, one job per truth (each truth has its own data cache, which the job writes; the later batches only read it)
  map-rest   the F2u and Fe MAPs (they start from the F2 MAP of the same truth as one of their starts)
  chain      one job per (case, chain): ``--stage chain --chain-id c`` over all segments; the done marker is the last segment of the chain
  finalize   per case: the predictive checks and the summary (``--stage all`` finds the chains complete and runs ppc and finalize)
  ti         thermodynamic integration of the cases ``--ti-cases`` (truth:gap:model, default T1 at 0.05 with F2 and with Fe): ``--ti-runs`` independent one-chain runs per case
             (tags r0, r1, ..., seeds 2, 3, ...), one process each, so a case takes the wall time of one chain; ``finalize`` combines them with the between-run spread
  pixel      the Laplace pixel posteriors of every truth (``--model pixel``)

A job whose done marker exists is skipped by ``batch_run.py``; every job is also idempotent on its own (``parametric_run.py`` skips finished work).
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = (("T1", 0.05), ("T1", 0.1), ("T2", 0.05), ("T2", 0.1), ("T3", 0.05), ("T3", 0.1), ("T4", 0.0), ("T5", 0.1))
MODELS = ("F2", "F2u", "Fe")


def label(name, gap):
    """The truth label of ``parametric_run.truth_label`` (T4 and T5 at 0.1 have no gap suffix)."""
    if name == "T4" or (name == "T5" and abs(gap - 0.1) < 1e-12):
        return name
    return f"{name}g{gap:g}"


def case_args(name, gap, model):
    return ["--truth", name, "--gap", f"{gap:g}", "--model", model]


def jobs_for(batch, results="results/parametric", chains=4, segments=5, ti_cases=None, extra=(), ti_runs=4, plan=None, models=None):
    """The jobs of one batch (a list of ``batch_run.py`` job dicts). ``extra`` arguments are appended to every command (e.g. ``--segment-draws 200``).

    ``plan`` is the list of ``(truth, gap)`` pairs (default ``PLAN``, the eight truths of DESIGN 14.3) and ``models`` the shape models whose chains and summaries are queued
    (default ``MODELS``); the F2 MAP jobs are always written for every truth of the plan (the nested models start from them). A restricted plan is how a time-limited run
    queues the priority cases first."""
    jobs = []
    plan = PLAN if plan is None else tuple(plan)
    models = MODELS if models is None else tuple(models)
    ti_cases = ti_cases if ti_cases is not None else [("T1", 0.05, "F2"), ("T1", 0.05, "Fe")]
    for name, gap in plan:
        lab = label(name, gap)
        if batch == "map-f2":
            jobs.append({"name": f"m7_map_{lab}_F2", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, "F2"), "--stage", "map", *extra], "device": "cpu", "threads": 1,
                         "done_marker": f"{results}/{lab}_F2/map.json"})
        elif batch == "map-rest":
            for model in [m for m in models if m != "F2"]:
                jobs.append({"name": f"m7_map_{lab}_{model}", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, model), "--stage", "map", *extra], "device": "cpu",
                             "threads": 1, "done_marker": f"{results}/{lab}_{model}/map.json"})
        elif batch == "chain":
            for model in models:
                for c in range(chains):
                    jobs.append({"name": f"m7_chain_{lab}_{model}_c{c}", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, model), "--stage", "chain", "--chain-id", str(c),
                                                                                  "--chains", str(chains), "--segments", str(segments), *extra], "device": "cpu", "threads": 1,
                                 "done_marker": f"{results}/{lab}_{model}/chains/c{c}_s{segments - 1}.npz"})
        elif batch == "finalize":
            for model in models:
                jobs.append({"name": f"m7_finalize_{lab}_{model}", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, model), "--chains", str(chains), "--segments", str(segments), *extra],
                             "device": "cpu", "threads": 1, "done_marker": f"{results}/{lab}_{model}/summary.json"})
        elif batch == "pixel":
            jobs.append({"name": f"m7_pixel_{lab}", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, "pixel"), *extra], "device": "cpu", "threads": 1,
                         "done_marker": f"{results}/{lab}_pixel/summary.json"})
    if batch == "ti":
        for name, gap, model in ti_cases:
            lab = label(name, gap)
            for j in range(ti_runs):
                jobs.append({"name": f"m7_ti_{lab}_{model}_r{j}", "cmd": ["scripts/parametric_run.py", *case_args(name, gap, model), "--stage", "ti", "--ti-tag", f"r{j}", "--ti-seed", str(2 + j),
                                                                          "--ti-chains", "1", *extra], "device": "cpu", "threads": 1, "done_marker": f"{results}/{lab}_{model}/ti_r{j}.json"})
    if batch not in ("map-f2", "map-rest", "chain", "finalize", "ti", "pixel"):
        raise ValueError(batch)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", nargs="+", default=["map-f2", "map-rest", "chain", "finalize", "ti", "pixel"], choices=["map-f2", "map-rest", "chain", "finalize", "ti", "pixel"])
    ap.add_argument("--out-dir", default=str(ROOT / "scripts" / "jobs"))
    ap.add_argument("--results", default="results/parametric")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--segments", type=int, default=5)
    ap.add_argument("--plan", nargs="+", default=None, metavar="TRUTH:GAP", help="restrict the truths (default: the eight of DESIGN 14.3), e.g. T1:0.05 T1:0.1 T4:0")
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS), help="restrict the shape models of the chain and finalize batches (default: all three)")
    ap.add_argument("--ti-runs", type=int, default=4, help="independent one-chain TI runs per case")
    ap.add_argument("--ti-cases", nargs="+", default=["T1:0.05:F2", "T1:0.05:Fe"], metavar="TRUTH:GAP:MODEL")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="arguments appended to every command (give last), e.g. --extra --segment-draws 200")
    args = ap.parse_args(argv)
    ti_cases = [(f.split(":")[0], float(f.split(":")[1]), f.split(":")[2]) for f in args.ti_cases]
    plan = None if args.plan is None else [(f.split(":")[0], float(f.split(":")[1])) for f in args.plan]
    for batch in args.batch:
        jobs = jobs_for(batch, args.results, args.chains, args.segments, ti_cases, tuple(args.extra), args.ti_runs, plan, args.models)
        path = Path(args.out_dir) / f"m7_{batch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"wrote {path} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
