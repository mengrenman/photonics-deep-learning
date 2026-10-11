"""Do the synthetic data depend on the XLA thread count? Build d_obs at several thread counts and compare.

    python scripts/check_data_threads.py [--threads 1 4] [--out results/prior_comparison/data_thread_check.json] [problem args]

The problem arguments are those of scripts/run_prior_comparison.py (--n, --side, --eps-r, --snr, --seed, --data-factor,
--n-tx, --n-rx; defaults: the DESIGN 11.3 problem). For every thread count T the data (``run_prior_comparison.build_data``:
the fine-grid forward model, then the noise) are built in a fresh subprocess whose environment is the one
``scripts/batch_run.py`` gives a CPU job of T threads (``NPROC``, ``XLA_FLAGS``, BLAS variables). The SHA-256 of d_obs
and the difference of the arrays in units of the noise std, max |d_T - d_1| / s, are written to the JSON. The Krylov solves
sum in a different order at a different thread count, so the bits (and the hash) differ while the numbers agree to
round-off (about 1e-14 of s): the reason the data comparison of the prior runs is numerical (``run_prior_comparison.D_OBS_TOL``)
and not by hash.

Run only when the machine has the threads to spare: the T-thread worker uses T XLA threads.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def compare_arrays(arrays, s):
    """``{"max_abs_diff_over_noise_std": ...}`` of the largest |d_a - d_ref| / s over the list ``arrays`` against its first
    element, and the same relative to max |d_ref|."""
    ref = np.asarray(arrays[0])
    worst = max((float(np.abs(np.asarray(a) - ref).max()) for a in arrays[1:]), default=0.0)
    return {"max_abs_diff_over_noise_std": worst / s, "max_abs_diff_over_max_abs_d": worst / float(np.abs(ref).max())}


def worker(argv):
    """Build the data in this process and save d_obs / noise std / SHA-256 (the subprocess side)."""
    import run_prior_comparison as rpc

    ap = argparse.ArgumentParser()
    ap.add_argument("--save", required=True)
    ns, rest = ap.parse_known_args(argv)
    args = rpc.parse_args(rest + ["--prior", "matern", "--out", "unused"])
    prob = rpc.build_data(args)
    np.savez(ns.save, d_obs=np.asarray(prob.d_obs), noise_std=prob.s, sha=prob.sha, thread_env=json.dumps(rpc.thread_env()))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "--worker":
        return worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", default=str(ROOT / "results" / "prior_comparison" / "data_thread_check.json"))
    args, problem = ap.parse_known_args(argv)
    import batch_run

    results, arrays, s = {}, [], None
    with tempfile.TemporaryDirectory() as tmp:
        for t in args.threads:
            f = Path(tmp) / f"d_{t}.npz"
            env = {**__import__("os").environ, **batch_run.cpu_job_env(t)}
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", "--save", str(f), *problem], check=True, env=env)
            z = np.load(f)
            arrays.append(z["d_obs"])
            s = float(z["noise_std"])
            results[str(t)] = {"d_obs_sha256": str(z["sha"]), "thread_env_seen_by_the_worker": json.loads(str(z["thread_env"]))}
            print(f"threads {t}: d_obs sha256 {results[str(t)]['d_obs_sha256'][:16]}")
    out = {"threads": args.threads, "problem_args": problem, "noise_std": s, "per_thread_count": results,
           "same_hash": len({v["d_obs_sha256"] for v in results.values()}) == 1, **compare_arrays(arrays, s)}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"same hash across thread counts: {out['same_hash']}; max |d_T - d_{args.threads[0]}| / s = "
          f"{out['max_abs_diff_over_noise_std']:.2e}; wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
