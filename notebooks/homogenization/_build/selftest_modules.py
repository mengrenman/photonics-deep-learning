"""Run every module test in this folder (_build/test_*.py) and exit nonzero if any fails.

    cd notebooks/homogenization && python _build/selftest_modules.py [name ...]

Each test file is a standalone script that prints PASS/FAIL lines and exits nonzero on failure.
Run this before building a notebook and after any change to a module.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
folder = here.parent
env = dict(os.environ)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    env.setdefault(var, "4")
env.setdefault("NPROC", "4")
env["PYTHONPATH"] = str(folder) + os.pathsep + env.get("PYTHONPATH", "")

wanted = set(sys.argv[1:])
tests = sorted(p for p in here.glob("test_*.py") if not wanted or p.stem[5:] in wanted)
failed = []
for path in tests:
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], cwd=folder, env=env)
    dt = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"=== {path.name}: {status} in {dt:.1f} s", flush=True)
    if proc.returncode != 0:
        failed.append(path.name)
print(f"{len(tests) - len(failed)} of {len(tests)} test files passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
