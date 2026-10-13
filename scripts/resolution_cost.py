"""Cost of the milestone-6 computations (docs/DESIGN.md 13): the CRB grid (forward solve and the jvp tangents) and the n = 64 Laplace pipeline.

    NPROC=4 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true python scripts/resolution_cost.py [--crb-grids 128 256] [--out cost.json]

(or through ``scripts/batch_run.py --threads 4``; the thread environment decides what is measured and is stored). Run it on an idle machine:
every record carries the 1-minute load average before and after, and the minimum over repetitions is the cost of an idle machine.

Measured, at the CRB grids (side 1.5, 40 x 60 book rings, tol 1e-10, restart 100) for the two-disk object with g = 0.1 and eps_r = 2 and 4:

  * ``forward_s``: one scattering solve for all 40 transmitters (``Forward.__call__``);
  * ``linearize_s``: the primal solve of ``jax.linearize`` (``fisher.data_derivatives``); ``tangent_s``: one tangent (one more stacked Krylov
    solve with the same operator); ``tangents_total_s``: ``fisher.data_derivatives`` for the six shape parameters; ``born_s``: the Born derivatives;
    ``shape_derivatives_s``: the exact boundary derivatives of the six parameters (NumPy);
  * ``peak_rss_gb`` of the process after the grid;

and at the Laplace grid n = 64 (the n = 128 data grid for the data): ``data_s`` (the fine-grid forward model), ``u_value_and_grad_s`` for the Matern
and the TV prior (what an HMC leapfrog step pays), ``laplace_mass_s`` and its parts, ``map_newton`` iteration cost, ``theta_std_s``.
If <out-dir>/laplace.json exists, the per-configuration wall times recorded by ``resolution_laplace.py`` are summarized per prior and contrast.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import os
import resource
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import fisher as fi
from bayes_scatter import shapes as sh
from bayes_scatter import thetaspace as ts
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resolution_laplace as rl  # noqa: E402

RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)


def timed(fn, reps):
    """(min, median) wall seconds of ``fn()`` over ``reps`` calls after one warm-up call; the result is blocked on."""
    jax.block_until_ready(fn())
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - t0)
    return {"min_s": float(np.min(t)), "median_s": float(np.median(t)), "reps": reps}


def crb_grid_cost(n, eps, g, reps, side=1.5):
    fwd = Forward(ch15_setup(n=n, side=side), tol=1e-10, restart=100)
    obj = sh.TwoDisks(g=g, r=0.2, eps_r=eps)
    chi = jnp.asarray(obj.chi(fwd.grid))
    load0 = os.getloadavg()[0]
    out = {"n": n, "h": fwd.grid.h, "eps_r": eps, "g": g, "forward": timed(lambda: fwd(chi)[0], reps)}
    t0 = time.perf_counter()
    dchi = sh.shape_derivatives(fwd.grid, obj)
    out["shape_derivatives_s"] = time.perf_counter() - t0
    _, tangent = jax.linearize(lambda c: fwd(c)[0], chi)
    tangent = jax.jit(tangent)
    out["linearize"] = timed(lambda: jax.linearize(lambda c: fwd(c)[0], chi)[0], max(2, reps // 2))
    out["tangent_per_parameter"] = {name: timed(lambda k=k: tangent(jnp.asarray(dchi[k])), reps) for k, name in enumerate(sh.PARAM_NAMES)}
    out["tangents_total"] = timed(lambda: fi.data_derivatives(fwd, chi, dchi), max(2, reps // 2))
    out["born_derivatives"] = timed(lambda: fi.born_data_derivatives(fwd, dchi), reps)
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**3 if sys.platform == "darwin" else 1024.0**2)
    out["load_avg_before"], out["load_avg_after"] = load0, os.getloadavg()[0]
    return out


def laplace_grid_cost(reps, eps=2.0, g=0.1):
    args = rl.parse_args(["--stage", "config", "--g", str(g), "--eps-r", str(eps), "--prior", "matern"])
    t0 = time.perf_counter()
    problem = rl.build_problem(args, g, eps)
    out = {"n": args.n, "data_grid_n": args.n * args.data_factor, "eps_r": eps, "g": g, "build_problem_incl_data_s": time.perf_counter() - t0, "load_avg_before": os.getloadavg()[0]}
    theta = jnp.asarray(problem.theta_true)
    for name, lam in (("matern", None), ("tv", 1.0)):
        target = ts.ThetaTarget(problem.forward, rl.build_prior(args, name, problem.grid, lam), problem.d_obs, problem.s)
        _, _, aux = target.potential_and_grad(theta, target.initial_aux())
        out[f"u_value_and_grad_{name}"] = timed(lambda t=target: t.potential_and_grad(theta, aux), reps)
        target.laplace_mass(theta)
        runs = [target.laplace_mass(theta).info for _ in range(max(2, reps // 2))]
        best = min(runs, key=lambda i: i["time_total_s"])
        out[f"laplace_mass_{name}"] = {"total_min_s": best["time_total_s"], "parts_of_fastest": {k: best[k] for k in ("time_fields_s", "time_gn_s", "time_prior_hessian_s", "time_cholesky_s")}}
        mass = target.laplace_mass(theta)
        t0 = time.perf_counter()
        mass.theta_std()
        out[f"theta_std_{name}_s"] = time.perf_counter() - t0
        th, ninfo = ts.map_newton(target, theta + 0.02 * jnp.asarray(np.random.default_rng(0).standard_normal(theta.shape)), max_iter=3, tol=0.0)
        out[f"map_newton_{name}_per_iteration_s"] = float(np.mean([h["time_s"] for h in ninfo["history"]]))
    out["load_avg_after"] = os.getloadavg()[0]
    return out


def record_costs(out_dir):
    """Wall times of the recorded configurations by prior and contrast (median, min, max of the total and of the MAP part)."""
    p = Path(out_dir) / "laplace.json"
    if not p.exists():
        return None
    recs = json.loads(p.read_text())["records"].values()
    out = {}
    for prior in ("matern", "tv"):
        for eps in sorted({r["eps_r"] for r in recs}):
            rr = [r for r in recs if r["prior"] == prior and r["eps_r"] == eps and (r.get("lam_scale") in (None, 1.0))]
            if not rr:
                continue
            tot = np.array([r["timings_s"]["total"] for r in rr])
            mp = np.array([r["map"]["time_s_all_starts"] for r in rr])
            out[f"{prior}_eps{eps:g}"] = {"n_configurations": len(rr), "total_s": {"median": float(np.median(tot)), "min": float(tot.min()), "max": float(tot.max())},
                                           "map_all_starts_s": {"median": float(np.median(mp)), "min": float(mp.min()), "max": float(mp.max())},
                                           "mean_load_avg": float(np.mean([r["load_avg_1min"] for r in rr])), "thread_env": rr[0]["thread_env"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--crb-grids", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--eps", type=float, nargs="+", default=[2.0, 4.0], help="contrasts of the CRB-grid costs")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-laplace-grid", action="store_true")
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--out", default="cost.json")
    args = ap.parse_args(argv)
    out = {"thread_env": rl.thread_env(), "cpu_count": os.cpu_count(), "reps": args.reps, "crb_grid": {}}
    for n in args.crb_grids:
        for eps in args.eps:
            rec = crb_grid_cost(n, eps, 0.1, args.reps)
            out["crb_grid"][f"n{n}_eps{eps:g}"] = rec
            print(f"n = {n}, eps_r = {eps:g}: forward {rec['forward']['min_s']:.2f} s, tangents total {rec['tangents_total']['min_s']:.2f} s "
                  f"({rec['tangents_total']['min_s'] / 6:.2f} s per parameter), Born {rec['born_derivatives']['min_s']:.3f} s, peak RSS {rec['peak_rss_gb']:.1f} GB, load {rec['load_avg_before']:.1f}")
        jax.clear_caches()
    if not args.skip_laplace_grid:
        out["laplace_grid"] = laplace_grid_cost(args.reps)
        lg = out["laplace_grid"]
        print(f"n = 64: U and gradient {lg['u_value_and_grad_matern']['min_s']:.3f} s (Matern), {lg['u_value_and_grad_tv']['min_s']:.3f} s (TV); Laplace mass "
              f"{lg['laplace_mass_matern']['total_min_s']:.2f} s; Newton iteration {lg['map_newton_matern_per_iteration_s']:.2f} s; theta_std {lg['theta_std_matern_s']:.1f} s")
    out["recorded_configurations"] = record_costs(args.out_dir)
    path = Path(args.out_dir) / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    main()
