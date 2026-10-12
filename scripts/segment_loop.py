"""Run the recorded segments of ONE chain of scripts/run_prior_comparison.py one after the other, each in its own process.

    python scripts/segment_loop.py --out-dir results/prior_comparison/pc_gmrf_adapt --chain-id 0 --first 0 --last 9 \
        --samples 100 --warmup 50 [--go FILE] [--stop FILE] [--poll S] -- <arguments of run_prior_comparison.py>

The arguments after ``--`` are the common arguments of the run (the problem, the prior, ``--adapt``, ``--out``, ``--start-theta``,
...); ``--stage chain --chain-id C --segment J --samples S`` (and ``--warmup W`` for segment 0) are appended here. Why a loop:
a segment is the unit of checkpointing (a killed process loses at most the segment it was computing, DESIGN 12.4), but
``scripts/batch_run.py`` has no job dependencies, and one job per segment would have to sleep in a slot until the previous
segment exists. One loop per chain is one batch_run job (one XLA thread) that keeps its slot until the chain is done:

* a segment whose file ``chain_<C>_seg<J>.npz`` exists is skipped (so a restarted loop continues where the last one stopped);
* a segment J > 0 needs segment J - 1 on disk (``run_prior_comparison.py`` reads it); a missing predecessor is an error;
* ``--go FILE`` (optional) waits, before the first segment, until FILE exists (checked every ``--poll`` seconds);
* ``--stop FILE`` (optional): the loop exits with status 0 before starting a segment if FILE exists (the segment that is running
  is allowed to finish), which is how a budget is enforced without killing a computation;
* a segment that exits nonzero or does not write its file ends the loop with status 1.

The XLA thread count is whatever the environment says (``scripts/batch_run.py`` sets it); every segment inherits it.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_prior_comparison.py"


def segment_file(out_dir, chain_id, segment):
    """Path of the segment file ``chain_<chain_id>_seg<segment>.npz`` in ``out_dir``."""
    return Path(out_dir) / f"chain_{chain_id}_seg{segment}.npz"


def segment_command(common, chain_id, segment, samples, warmup, python=None):
    """The command line (list of str) that computes segment ``segment`` of chain ``chain_id``: the interpreter, the runner, the
    common arguments, then ``--stage chain --chain-id --segment --samples`` and, for segment 0 only, ``--warmup``."""
    cmd = [python or sys.executable, str(RUNNER), *common, "--stage", "chain", "--chain-id", str(chain_id), "--segment", str(segment),
           "--samples", str(samples)]
    if segment == 0:
        cmd += ["--warmup", str(warmup)]
    return cmd


def pending_segments(out_dir, chain_id, first, last):
    """The segments in ``[first, last]`` whose file is absent, in order. Raises ``ValueError`` if a pending segment J > 0 has no file for
    J - 1 on disk and J - 1 is not itself pending in this range (a gap that the loop could not fill)."""
    todo = [j for j in range(first, last + 1) if not segment_file(out_dir, chain_id, j).exists()]
    for j in todo:
        if j > 0 and not segment_file(out_dir, chain_id, j - 1).exists() and (j - 1) not in todo:
            raise ValueError(f"segment {j} of chain {chain_id} needs segment {j - 1}, which does not exist in {out_dir}")
    return todo


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        raise SystemExit("usage: segment_loop.py [options] -- ARGUMENTS OF run_prior_comparison.py")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True, help="the run directory that holds chain_<C>_seg<J>.npz")
    ap.add_argument("--chain-id", type=int, required=True)
    ap.add_argument("--first", type=int, default=0, help="first segment")
    ap.add_argument("--last", type=int, required=True, help="last segment (inclusive)")
    ap.add_argument("--samples", type=int, required=True, help="recorded trajectories per segment")
    ap.add_argument("--warmup", type=int, default=50, help="step-size warmup trajectories of segment 0")
    ap.add_argument("--go", default=None, help="wait until this file exists before the first segment")
    ap.add_argument("--stop", default=None, help="exit 0 before starting a segment if this file exists")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between polls of --go")
    args = ap.parse_args(argv[:split])
    common = argv[split + 1:]
    if not common:
        raise SystemExit("no run_prior_comparison.py arguments after --")
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    stop = None if args.stop is None else (Path(args.stop) if Path(args.stop).is_absolute() else ROOT / args.stop)
    if args.go is not None:
        go = Path(args.go) if Path(args.go).is_absolute() else ROOT / args.go
        print(f"segment_loop: waiting for {go}", flush=True)
        while not go.exists():
            if stop is not None and stop.exists():
                print("segment_loop: stop file exists, not starting", flush=True)
                return 0
            time.sleep(args.poll)
    todo = pending_segments(out_dir, args.chain_id, args.first, args.last)
    print(f"segment_loop: chain {args.chain_id}, segments {args.first}..{args.last}, pending {todo}", flush=True)
    for j in todo:
        if stop is not None and stop.exists():
            print(f"segment_loop: stop file exists, ending before segment {j}", flush=True)
            return 0
        if segment_file(out_dir, args.chain_id, j).exists():  # written by someone else since the plan
            continue
        t0 = time.perf_counter()
        print(f"segment_loop: {time.strftime('%Y-%m-%dT%H:%M:%S%z')} starting chain {args.chain_id} segment {j}", flush=True)
        rc = subprocess.run(segment_command(common, args.chain_id, j, args.samples, args.warmup), cwd=ROOT).returncode
        wall = time.perf_counter() - t0
        if rc != 0 or not segment_file(out_dir, args.chain_id, j).exists():
            print(f"segment_loop: segment {j} failed (exit {rc}, file {'present' if segment_file(out_dir, args.chain_id, j).exists() else 'absent'}) "
                  f"after {wall:.0f} s", flush=True)
            return 1
        print(f"segment_loop: segment {j} done in {wall:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
