"""Write the batch_run job file that RESUMES milestone 5 (docs/DESIGN.md section 12) after the machine reboot of 2026-10-11 20:55.

    python scripts/make_m5_resume.py --out scripts/jobs/m5_batch2.json
    python scripts/batch_run.py scripts/jobs/m5_batch2.json --cpu-slots 12 --threads 1 --log-dir results/prior_comparison/logs

The first attempt (scripts/make_m5_batch.py, one process per chain and per 300 to 500 trajectories) lost 2.8 hours of GMRF chains to the reboot
because nothing was written before the end of a segment. Here every chain is one ``scripts/segment_loop.py`` job that runs SHORT recorded
segments (``--seg-samples``, default 100 trajectories, 10 to 25 minutes on one thread) one after the other, each in its own process, so a
reboot loses at most one segment, and a restarted batch skips the segments that exist.

Three groups of four chains (one XLA thread per job; the caller keeps the sum of the threads of the jobs that compute at most 10):

* ``gmrf``: the adapted-map GMRF run pc_gmrf_adapt (map digest 555cc867a4a2ffb4, chains started at the four milestone-4 end states,
  ``--start-theta``): segment 0 = 50 step-size warmup + one segment of recorded trajectories, segments 1 to ``--gmrf-last``. Starts at once.
* ``tvA``: pc_tv_lam3.75_adapt (lambda*), whose segments 0 and 1 (400 recorded trajectories each) exist: segments 2 to ``--tva-last``.
  Waits for the file ``scripts/jobs/go/tvA``.
* ``tvB``: pc_tv_lam11.25_adapt (3 lambda*), whose segment 0 (400 trajectories) exists: segments 1 to ``--tvb-last``. Waits for
  ``scripts/jobs/go/tvB``.

``scripts/jobs/stop/<group>`` ends a group's chains cleanly before their next segment (a running segment finishes): the budget switch.

Batch 3 (``--batch 3``): the same three groups on RE-ADAPTED maps. The recorded draws of the first batch showed that the adaptation chains
were too short for the slowest directions (DESIGN 12.5, ``scripts/readapt_map.py``); each run was re-estimated from its own draws into a new
run directory (``pc_gmrf_adapt2``, ``pc_tv_lam3.75_adapt2``, ``pc_tv_lam11.25_adapt2``) whose chains start at the final states of the old
ones (``starts.npy``), tune the step size (50 trajectories) with the new map frozen, and then record. Groups ``gmrf2`` (starts at once),
``tvA2`` and ``tvB2`` (wait for their go files).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_prior_jobs as mp  # noqa: E402

RESULTS = mp.RESULTS
GMRF_START = ["--start-theta", f"{RESULTS}/pc_gmrf_adapt/m4_ends.npy"]


def loop_job(name, run, group, chain, first, last, samples, warmup, extra=(), go=False, out=None):
    """The batch_run job dict of chain ``chain`` of the run ``run`` (a prior name of make_prior_jobs, ``_adapt`` included): segments
    ``first`` to ``last`` of ``samples`` recorded trajectories (``warmup`` step-size trajectories before segment 0). ``out`` is the run
    directory name (default ``pc_<run>``; a re-adapted run has another one than its prior name)."""
    out = out or f"pc_{run}"
    cmd = ["scripts/segment_loop.py", "--out-dir", f"{RESULTS}/{out}", "--chain-id", str(chain), "--first", str(first), "--last", str(last),
           "--samples", str(samples), "--warmup", str(warmup), "--stop", f"scripts/jobs/stop/{group}"]
    if go:
        cmd += ["--go", f"scripts/jobs/go/{group}"]
    cmd += ["--", *mp.PROBLEM, *mp.prior_args(run), "--out", out, *extra]
    return {"name": name, "cmd": cmd, "device": "cpu", "threads": 1, "done_marker": f"{RESULTS}/{out}/chain_{chain}_seg{last}.npz"}


def build(seg_samples=100, gmrf_last=9, tva_last=7, tvb_last=4, warmup=50):
    """All jobs: four chains each of gmrf_adapt (segments 0..gmrf_last), tv_lam3.75_adapt (2..tva_last) and tv_lam11.25_adapt (1..tvb_last)."""
    out = []
    for c in range(4):
        out.append(loop_job(f"m5r_gmrf_c{c}", "gmrf_adapt", "gmrf", c, 0, gmrf_last, seg_samples, warmup, GMRF_START))
    for c in range(4):
        out.append(loop_job(f"m5r_tvA_c{c}", "tv_lam3.75_adapt", "tvA", c, 2, tva_last, seg_samples, warmup, go=True))
    for c in range(4):
        out.append(loop_job(f"m5r_tvB_c{c}", "tv_lam11.25_adapt", "tvB", c, 1, tvb_last, seg_samples, warmup, go=True))
    return out


def build3(seg_samples=100, gmrf_last=19, tv_last=5, warmup=50):
    """Batch 3: four chains each of the re-adapted runs pc_gmrf_adapt2 (segments 0..gmrf_last), pc_tv_lam3.75_adapt2 and
    pc_tv_lam11.25_adapt2 (segments 0..tv_last), each starting at its ``starts.npy``."""
    out = []
    for group, run, dest, last, go in (("gmrf2", "gmrf_adapt", "pc_gmrf_adapt2", gmrf_last, False),
                                       ("tvA2", "tv_lam3.75_adapt", "pc_tv_lam3.75_adapt2", tv_last, True),
                                       ("tvB2", "tv_lam11.25_adapt", "pc_tv_lam11.25_adapt2", tv_last, True)):
        for c in range(4):
            out.append(loop_job(f"m5r3_{group}_c{c}", run, group, c, 0, last, seg_samples, warmup,
                                ["--start-theta", f"{RESULTS}/{dest}/starts.npy"], go=go, out=dest))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-samples", type=int, default=100, help="recorded trajectories per segment")
    ap.add_argument("--gmrf-last", type=int, default=9)
    ap.add_argument("--tva-last", type=int, default=7)
    ap.add_argument("--tvb-last", type=int, default=4)
    ap.add_argument("--batch", type=int, choices=(2, 3), default=2, help="2: the resume batch; 3: the same runs on re-adapted maps")
    args = ap.parse_args(argv)
    js = build(args.seg_samples, args.gmrf_last, args.tva_last, args.tvb_last) if args.batch == 2 else build3(args.seg_samples)
    Path(args.out).write_text(json.dumps(js, indent=2) + "\n")
    for d in ("go", "stop"):
        Path("scripts/jobs", d).mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(js)} jobs to {args.out}")
    return js


if __name__ == "__main__":
    main()
