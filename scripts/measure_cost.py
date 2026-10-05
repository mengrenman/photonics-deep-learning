"""Measure the cost of one Phi evaluation (all transmitters) and one Phi + gradient evaluation for the
configurations of the nonlinear ladder, before sizing the chains (docs/DESIGN.md section 8, step 6).

For each configuration: build the log_eps target exactly as scripts/run_posterior.py does (data on a 2x finer
grid, 30 dB), run a short MAP (--map-maxiter iterations, from theta = 0), then time, at the MAP point with the
MAP's warm start:  Phi (forward only, what pCN pays),  Phi + gradient (forward + adjoint, what MALA /
pCN-Langevin pay and what each HMC leapfrog step pays), and Phi + gradient from a cold start (the incident
field).  Also reports the true relative residual of the solve at the MAP.

Run:  python scripts/measure_cost.py [--configs weak32 moderate32 book48 book64] [--map-maxiter 40]
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
from bayes_scatter.inference import map_estimate
from bayes_scatter.likelihood import Target, add_noise, noise_std_from_snr
from bayes_scatter.phantoms import austria_fixed, cylinder
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "weak32": dict(n=32, side=2.0, phantom="cylinder", eps=1.5, sigma=0.5, ell=0.4),
    "moderate32": dict(n=32, side=2.0, phantom="austria", eps=2.0, sigma=0.5, ell=0.4),
    "book48": dict(n=48, side=5.0, phantom="austria", eps=2.0, sigma=0.5, ell=0.4),
    "book64": dict(n=64, side=5.0, phantom="austria", eps=2.0, sigma=0.5, ell=0.4),
}


def phantom(grid, cfg):
    if cfg["phantom"] == "cylinder":
        return cylinder(grid, 0.5, cfg["eps"])
    return austria_fixed(grid, cfg["eps"])


def timed(fn, reps=5):
    jax.block_until_ready(fn())  # compile / first call
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t.append(time.perf_counter() - t0)
    return float(np.median(t)), float(np.min(t))


def measure(name, cfg, snr, map_maxiter):
    n, side = cfg["n"], cfg["side"]
    setup = ch15_setup(n=n, side=side)
    data_setup = Setup(Grid(2 * n, side), setup.antennas)
    chi_data = phantom(data_setup.grid, cfg)
    t0 = time.perf_counter()
    d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(chi_data))
    jax.block_until_ready(d_clean)
    t_data = time.perf_counter() - t0
    s = noise_std_from_snr(d_clean, snr)
    d_obs = add_noise(jax.random.key(0), d_clean, s)
    fwd = Forward(setup, tol=1e-10)
    field = GaussianField(setup.grid, "matern", sigma=cfg["sigma"], ell=cfg["ell"], nu=1.5, mean=0.0)
    target = Target(fwd, field, "log_eps", d_obs, s)
    xi_map, chi_map, info = map_estimate(target, jnp.zeros(field.xi_shape), maxiter=map_maxiter)
    aux = target.initial_aux()
    # a converged-warm aux at the MAP: one extra solve
    _, _, aux_map = target.potential_and_grad(xi_map, aux)
    t_phi = timed(lambda: target.potential(xi_map, aux_map))
    t_pg = timed(lambda: target.potential_and_grad(xi_map, aux_map))
    t_pg_cold = timed(lambda: target.potential_and_grad(xi_map, aux))
    resid = target.solver_residual(xi_map, aux_map)
    out = {
        "config": name, **cfg, "snr_db": snr, "xi_shape": list(field.xi_shape), "noise_std": s,
        "data_forward_s_incl_compile": t_data,
        "map_maxiter": map_maxiter, "map_nfev": info["nfev"], "map_time_s": info["time_s"],
        "map_s_per_eval": info["time_s"] / info["nfev"], "map_phi": info["phi"],
        "max_eps_at_map": float(1 + chi_map.max()),
        "phi_s_median_min": t_phi, "phi_grad_s_median_min": t_pg, "phi_grad_cold_s_median_min": t_pg_cold,
        "solver_residual_at_map": resid, "n_data": d_obs.size,
    }
    print(f"{name:11s} n={n:3d} side={side:g} P={field.P}: Phi {t_phi[0] * 1e3:8.1f} ms, Phi+grad {t_pg[0] * 1e3:8.1f} ms "
          f"(cold {t_pg_cold[0] * 1e3:8.1f} ms); MAP {info['nfev']} evals {info['time_s']:.1f} s "
          f"({info['time_s'] / info['nfev'] * 1e3:.0f} ms/eval), Phi {info['phi']:.0f}, resid {resid:.1e}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--map-maxiter", type=int, default=40)
    ap.add_argument("--out", default="cost/cost.json")
    args = ap.parse_args(argv)
    res = [measure(c, CONFIGS[c], args.snr, args.map_maxiter) for c in args.configs]
    p = ROOT / "results" / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print("wrote", p)


if __name__ == "__main__":
    main()
