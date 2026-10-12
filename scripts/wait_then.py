"""Wait for dependencies, then replace this process by a command: dependency and go/skip gating for scripts/batch_run.py jobs.

    python scripts/wait_then.py [--file PATH]... [--meta-ok NAME]... [--go PATH] [--skip PATH] [--poll S] -- CMD ARG...

batch_run.py has no job dependencies, so a job file can list a whole pipeline (prepare, adapt, chains, extensions) and let
each later job sleep until its inputs exist. ``--file PATH`` (repeatable) waits for PATH to exist (relative to the repository
root); ``--meta-ok NAME`` (repeatable) waits until ``results/prior_comparison/logs/NAME.meta.json`` (``--log-dir``) has status
"ok" and exits 1 if it ends in a failed state ("failed", "timeout", "interrupted"); ``--go PATH`` waits for PATH, a file that a
person creates after looking at the previous stage; ``--skip PATH`` makes the job exit 0 without running as soon as PATH exists
(checked at every poll, so a gated extension that turned out unnecessary can be canceled without killing the batch). When every
dependency holds the process is replaced (``os.execv``) by CMD; a first word ``python`` means this interpreter. A waiting job
uses no CPU (one sleep per ``--poll`` seconds), so waiting jobs may occupy batch_run slots, but THE CALLER decides when jobs start
computing (go files), and with them how many XLA threads are busy: the 10-thread cap of the project is not enforced here.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED_STATES = {"failed", "timeout", "interrupted"}


def meta_status(log_dir, name):
    """The "status" string of ``<log_dir>/<name>.meta.json``, or None if the file is absent or unreadable."""
    path = Path(log_dir) / f"{name}.meta.json"
    try:
        return json.loads(path.read_text()).get("status")
    except (OSError, ValueError):
        return None


def check(files, metas, go, skip, log_dir, root=ROOT):
    """One poll: ``"skip"`` if the skip file exists, ``"fail"`` if a watched job ended in a failed state, ``"ready"`` if every
    dependency holds, else ``"wait"``. Relative paths are relative to ``root``."""
    rel = lambda p: Path(p) if Path(p).is_absolute() else Path(root) / p
    if skip and rel(skip).exists():
        return "skip"
    states = [meta_status(log_dir, m) for m in metas]
    if any(s in FAILED_STATES for s in states):
        return "fail"
    if all(rel(f).exists() for f in [*files, *([go] if go else [])]) and all(s == "ok" for s in states):
        return "ready"
    return "wait"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("usage: wait_then.py [options] -- CMD ARG...")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--file", action="append", default=[], help="wait until this file exists (repeatable)")
    ap.add_argument("--meta-ok", action="append", default=[], metavar="NAME", help="wait until job NAME's meta.json says ok")
    ap.add_argument("--go", default=None, help="wait until this go file exists (created by hand)")
    ap.add_argument("--skip", default=None, help="exit 0 without running once this file exists")
    ap.add_argument("--log-dir", default="results/prior_comparison/logs")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between polls")
    args = ap.parse_args(argv[:split])
    cmd = argv[split + 1:]
    if not cmd:
        raise SystemExit("no command after --")
    log_dir = Path(args.log_dir) if Path(args.log_dir).is_absolute() else ROOT / args.log_dir
    print(f"wait_then: waiting for files {args.file} meta-ok {args.meta_ok} go {args.go} (skip {args.skip})", flush=True)
    while True:
        state = check(args.file, args.meta_ok, args.go, args.skip, log_dir)
        if state == "skip":
            print("wait_then: skip file exists, not running", flush=True)
            return 0
        if state == "fail":
            print("wait_then: a watched job failed, not running", flush=True)
            return 1
        if state == "ready":
            break
        time.sleep(args.poll)
    if cmd[0] in ("python", "python3"):
        cmd = [sys.executable, *cmd[1:]]
    print(f"wait_then: dependencies satisfied, running {' '.join(cmd)}", flush=True)
    os.chdir(ROOT)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
