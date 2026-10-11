"""Cost of the theta-space machinery (docs/DESIGN.md 11.2): per-evaluation cost, Jacobian assembly, Laplace mass.

Driver (default): runs one worker process per (n, threads) pair, one process at a time, each with the thread
environment of scripts/batch_run.py (``NPROC`` sizing the XLA CPU pool, the BLAS variables), and writes
results/prior_comparison/cost.json:

    python scripts/measure_theta_cost.py [--ns 32 48] [--threads 1 4] [--reps 7] [--map-n 48] [--out cost.json]

Worker (``--worker``, used by the driver; run it by hand under your own thread settings if you wish):

    NPROC=4 python scripts/measure_theta_cost.py --worker --n 48

The worker builds the DESIGN 11.3 problem (``austria_fixed`` eps_r 2, side 2, data on 2n, 30 dB) and, at theta of the
truth with a converged warm start, times (median and minimum of ``--reps`` repetitions after a warm-up call):

  * ``phi``: the forward model and misfit only; ``u_value_and_grad``: U = Phi + R and its gradient (one forward and one
    adjoint solve for all transmitters; this is what every HMC leapfrog step pays) for the cropped Matern prior
    (dense n^2 x n^2 precision matvec) and for the q-GGMRF; ``prior_value_and_grad``: R alone;
    ``zeta_phi_fn``: the reparametrized potential and gradient of the samplers (adds two dense triangular solves);
  * the Jacobian: ``fields`` (Nt + Nr solves, no Jacobian matrix), ``assemble_dense_J`` (the (Nt Nr) x n^2 complex
    matrix from the fields), ``gauss_newton_gram`` (H_GN by the Hadamard product of two Gram matrices), ``dense_JhJ``
    (H_GN from the dense J, for comparison), ``laplace_mass_total`` and its parts (Cholesky included);
  * with ``--map`` the MAP of each prior (smooth start, then L-BFGS-B in theta): iterations, evaluations, seconds, and
    the Laplace mass build time at the MAP.

Timings are wall-clock on a shared machine: every record carries the 1-minute load average before and after. Use the
minimum, which is robust against other processes, as the cost of an idle machine, and the median as the typical one.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
RESULTS = ROOT / "results" / "prior_comparison"


def timed(fn, reps):
    """Median and minimum wall time of ``fn()`` over ``reps`` calls after one warm-up (compile) call."""
    import jax
    import numpy as np

    jax.block_until_ready(fn())
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - t0)
    return {"median_s": float(np.median(t)), "min_s": float(np.min(t)), "reps": reps}


def worker(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    import bayes_scatter  # noqa: F401
    from bayes_scatter import jacobian as jb
    from bayes_scatter import mrf
    from bayes_scatter import thetaspace as ts
    import run_prior_comparison as rpc

    load0 = os.getloadavg()[0]
    cfg = SimpleNamespace(n=args.n, side=2.0, eps_r=2.0, snr=30.0, seed=0, data_factor=2, n_tx=40, n_rx=60, tol=1e-10,
                          prior="matern", sigma=0.5, ell=0.4, nu=1.5, mean=0.0)
    prob = rpc.build_data(cfg)
    grid, fwd, s = prob.setup.grid, prob.forward, prob.s
    priors = {"matern": rpc.build_prior(cfg, grid), "qggmrf": mrf.QGGMRF(args.n, q=1.2, T=1.0, sigma_g=0.1, sigma_0=1.0)}
    targets = {k: ts.ThetaTarget(fwd, p, prob.d_obs, s) for k, p in priors.items()}
    theta = jnp.asarray(prob.theta_true)
    _, _, aux = targets["matern"].potential_and_grad(theta, targets["matern"].initial_aux())  # converged warm start
    out = {"n": args.n, "nproc": os.environ.get("NPROC"), "omp": os.environ.get("OMP_NUM_THREADS"),
           "cpu_count": os.cpu_count(), "load_avg_before": load0, "reps": args.reps,
           "n_tx": 40, "n_rx": 60, "side": 2.0, "point": "theta of the truth (austria_fixed eps_r 2), warm start"}
    r = args.reps
    cost = {"phi": timed(lambda: targets["matern"].misfit(theta, aux), r)}
    for k, t in targets.items():
        cost[f"u_value_and_grad_{k}"] = timed(lambda t=t: t.potential_and_grad(theta, aux), r)
        cost[f"prior_value_and_grad_{k}"] = timed(lambda t=t: t.prior.value_and_grad(theta), r)
    out["per_evaluation"] = cost

    chi_t = targets["matern"].chi(theta)
    scale = jb.link_derivative("log_eps", theta)
    jac = {"fields": timed(lambda: jb.jacobian_factors(fwd, chi_t), r)}
    u, w = jb.jacobian_factors(fwd, chi_t)
    jac["assemble_dense_J"] = timed(lambda: jb.assemble_jacobian(u, w, scale), r)
    J = jb.assemble_jacobian(u, w, scale)
    jac["gauss_newton_gram"] = timed(lambda: jb.gauss_newton_hessian(u, w, s, scale), r)
    jac["dense_JhJ"] = timed(lambda: jb.gauss_newton_hessian_dense(J, s), max(2, r // 3))
    jac["dense_J_bytes"] = int(J.size * 16)
    jac["n_solves"] = fwd.n_tx + fwd.n_rx
    out["jacobian"] = jac

    mass_cost = {}
    for k, t in targets.items():
        t.laplace_mass(theta)  # warm-up
        runs = [t.laplace_mass(theta).info for _ in range(max(2, r // 3))]
        best = min(runs, key=lambda i: i["time_total_s"])
        mass_cost[k] = {"total_min_s": best["time_total_s"], "parts_of_fastest": {x: best[x] for x in (
            "time_fields_s", "time_gn_s", "time_prior_hessian_s", "time_cholesky_s")},
            "total_all_s": [i["time_total_s"] for i in runs]}
    out["laplace_mass"] = mass_cost
    mass = targets["matern"].laplace_mass(theta)
    zt = targets["matern"].zeta_target(mass)
    z = zt.to_zeta(theta)
    out["per_evaluation"]["zeta_phi_fn_matern"] = timed(lambda: zt.phi_fn(z, aux), r)

    if args.map:
        runs = {}
        t0 = time.perf_counter()
        theta0, start_info = ts.smooth_start(fwd, prob.d_obs, s, grid, maxiter=100)
        start_s = time.perf_counter() - t0
        specs = {"matern": ("matern", {}), "gmrf": ("gmrf", {}), "qggmrf_sg0.05": ("qggmrf", {"sigma_g": 0.05}),
                 "qggmrf_sg0.1": ("qggmrf", {"sigma_g": 0.1})}
        for name, (kind, kw) in specs.items():
            c = SimpleNamespace(**{**vars(cfg), "prior": kind, **kw, "q": 1.2, "T": 1.0, "sigma_0": 1.0, "mu": 0.0,
                                   "sigma_g": kw.get("sigma_g", 0.1)})
            p = rpc.build_prior(c, grid)
            t = ts.ThetaTarget(fwd, p, prob.d_obs, s)
            th, info = ts.map_theta(t, theta0, maxiter=400, gtol=1e-6)
            m = t.laplace_mass(th)
            runs[name] = {"nit": info["nit"], "nfev": info["nfev"], "map_time_s": info["time_s"], "U": info["U"],
                          "phi": info["phi"], "R": info["R"], "grad_inf": info["grad_inf"],
                          "message": info["message"], "s_per_eval": info["time_s"] / info["nfev"],
                          "laplace_mass_s": m.info["time_total_s"]}
        out["map"] = {"smooth_start": {"nit": start_info["nit"], "nfev": start_info["nfev"], "time_s": start_s,
                                       "maxiter": 100}, "priors": runs}
    out["load_avg_after"] = os.getloadavg()[0]
    print("RESULT " + json.dumps(out, default=float))


def driver(args):
    sys.path.insert(0, str(ROOT / "scripts"))
    import batch_run as br

    results = []
    for n in args.ns:
        for threads in args.threads:
            env = {**os.environ, **br.cpu_job_env(threads, os.environ.get("XLA_FLAGS"))}
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--n", str(n), "--reps", str(args.reps)]
            if args.map_n == n and threads == max(args.threads):
                cmd.append("--map")
            print(f"n={n} threads={threads}: {' '.join(cmd[2:])}", flush=True)
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
                raise SystemExit(f"worker failed for n={n}, threads={threads}")
            line = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")][-1]
            rec = json.loads(line[len("RESULT "):])
            rec.update({"threads": threads, "env": {k: env[k] for k in ("NPROC", "OMP_NUM_THREADS", "XLA_FLAGS")},
                        "worker_wall_s": time.perf_counter() - t0})
            results.append(rec)
            pe = rec["per_evaluation"]
            print(f"  value+grad of U (min): matern {pe['u_value_and_grad_matern']['min_s']:.4f} s, qggmrf "
                  f"{pe['u_value_and_grad_qggmrf']['min_s']:.4f} s; load {rec['load_avg_before']:.1f} -> "
                  f"{rec['load_avg_after']:.1f}", flush=True)
    out = Path(args.out) if Path(args.out).is_absolute() else RESULTS / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", action="store_true", help="measure one grid in this process (used by the driver)")
    ap.add_argument("--n", type=int, default=48, help="worker: grid size")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--map", action="store_true", help="worker: also run the MAP of four priors")
    ap.add_argument("--ns", type=int, nargs="+", default=[32, 48])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4], help="XLA threads per worker (1 or >= 4)")
    ap.add_argument("--map-n", type=int, default=48, help="driver: grid size at which the MAP is also measured")
    ap.add_argument("--out", default="cost.json")
    args = ap.parse_args(argv)
    if args.worker:
        worker(args)
    else:
        if any(t in (2, 3) for t in args.threads):
            ap.error("NPROC = 2 or 3 can deadlock XLA (see scripts/batch_run.py); use 1 or >= 4 threads")
        driver(args)


if __name__ == "__main__":
    main()
