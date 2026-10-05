"""Operating-point table: grid model error at the truth against the noise, for problems with or without a run.

For each problem and inference grid size n, synthesize noise-free data on the 2n grid (the inverse-crime-free data
of run_posterior.py), evaluate the n-grid forward model on the coverage-averaged truth, and report
``bayes_scatter.inference.operating_point``: relative model error, relative noise, their ratio, Phi at the truth,
and its excess over the noise-free-data Phi in standard deviations (sd = sqrt(N)).

Usage:  python scripts/operating_point_table.py PROBLEM N [N ...] [--snr 30] [--seed 0]
        PROBLEM is weak (cylinder, eps_r 1.5, side 2), moderate (austria_fixed, eps_r 2, side 2) or book
        (austria_fixed, eps_r 2, side 5). Rows are appended to results/operating_points/<PROBLEM>.json.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup
from bayes_scatter.inference import operating_point
from bayes_scatter.likelihood import add_noise, noise_std_from_snr
from bayes_scatter.phantoms import austria_fixed, cylinder

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = {"weak": ("cylinder", 1.5, 2.0), "moderate": ("austria_fixed", 2.0, 2.0), "book": ("austria_fixed", 2.0, 5.0)}


def phantom(kind, grid, eps_r, side):
    return cylinder(grid, 0.25 * side, eps_r) if kind == "cylinder" else austria_fixed(grid, eps_r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("problem", choices=sorted(PROBLEMS))
    ap.add_argument("ns", type=int, nargs="+")
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    kind, eps_r, side = PROBLEMS[args.problem]
    out = ROOT / "results" / "operating_points" / f"{args.problem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else []
    for n in args.ns:
        t0 = time.time()
        setup = ch15_setup(n=n, side=side)
        data_setup = Setup(Grid(2 * n, side), setup.antennas)
        d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(phantom(kind, data_setup.grid, eps_r, side)))
        s = noise_std_from_snr(d_clean, args.snr)
        d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
        op = operating_point(Forward(setup, tol=1e-10), phantom(kind, setup.grid, eps_r, side), d_clean, d_obs, s)
        row = {"problem": args.problem, "n": n, "n_data_grid": 2 * n, "snr_db": args.snr, "seed": args.seed,
               **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in op.items()},
               "seconds": round(time.time() - t0, 1)}
        rows = [r for r in rows if not (r["n"] == n and r["snr_db"] == args.snr and r["seed"] == args.seed)] + [row]
        print(json.dumps(row), flush=True)
    out.write_text(json.dumps(sorted(rows, key=lambda r: (r["snr_db"], r["n"])), indent=1))


if __name__ == "__main__":
    main()
