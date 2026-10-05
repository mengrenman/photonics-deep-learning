"""Sample the posterior of the 2-D inverse scattering problem (docs/DESIGN.md sections 3-8).

Pipeline: phantom -> data on a finer grid (no inverse crime, see --data-factor) -> noise at --snr dB ->
Gaussian-field prior on theta = log eps_r (whitened coordinates) -> MAP (L-BFGS-B) -> MCMC with the chosen
sampler from prior draws or from the MAP -> summaries and figures.

Writes results/<out>/summary.json (all arguments, acceptance, final step, ESS and integrated autocorrelation
time of Phi and of 5 fixed pixels, split R-hat, wall time, forward-solve counts, ESS per second, the book's
error metrics of the MAP and of the posterior mean), chains.npz, maps.png (eps_r), maps_log_eps.png (theta)
and traces.png.

Two entries guard against misreading a run. "operating_point" is the discretization error of the inference
grid at the truth in units of the noise (||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude, and
Phi at the truth against N); when it is not small, the fit can reach Phi = N only by moving chi away from the
truth. "phi" is the potential that the sampler ran on (kind "physical" here), and "phi_physical" is the data
misfit Phi of the stored draws; the two differ for samplers that run in reparametrized coordinates (see
run_posterior_laplace.py), so quote "phi_physical" for the level of the misfit.

Examples
  python scripts/run_posterior.py --n 16 --side 2 --out demo
  python scripts/run_posterior.py --n 24 --side 2 --sampler hmc --warmup 300 --samples 500 --chains 2 --out hmc24
  python scripts/run_posterior.py --link identity --born --sampler pcn --samples 20000 --out born_pcn

Cost warning: the forward solve is unpreconditioned GMRES; high contrast is expensive (eps_r = 4 on the full
book setting costs seconds per solve, eps_r = 8 does not converge, see docs/DESIGN.md and forward.py), and
--phantom austria_variant draws eps_r up to 8.
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import diagnostics as dg
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ch15_setup
from bayes_scatter.inference import (err_chi, err_data, err_model, map_estimate, operating_point,
                                     sample)
from bayes_scatter.likelihood import Target, add_noise, noise_std_from_snr
from bayes_scatter.phantoms import austria_fixed, ch15_austria_variant, cylinder
from bayes_scatter.plotting import plot_maps, plot_traces
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]
print = functools.partial(print, flush=True)  # progress is visible while a long run is going
# Initial step sizes are deliberately small. A too large first proposal in an informative posterior lands at
# huge contrast, where the Krylov solve runs to its iteration cap (seconds per step); the Robbins-Monro
# adaptation grows the step by at most a factor e^0.43 per early step, so it recovers from a small start.
STEP0 = {"pcn": 0.01, "pcn_langevin": 1e-4, "mala": 1e-5, "hmc": 1e-3}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("geometry and truth")
    g.add_argument("--n", type=int, default=16, help="pixels per side of the inference grid")
    g.add_argument("--side", type=float, default=2.0, help="DOI side in wavelengths (the book uses 5)")
    g.add_argument("--n-tx", type=int, default=40, help="transmitters on the radius-4 ring")
    g.add_argument("--n-rx", type=int, default=60, help="receivers on the radius-4.5 ring")
    g.add_argument("--phantom", choices=["cylinder", "austria_fixed", "austria_variant"], default="cylinder")
    g.add_argument("--eps-r", type=float, default=1.5, help="relative permittivity of cylinder / austria_fixed "
                   "(austria_variant draws its own, up to 8)")
    g.add_argument("--radius", type=float, default=None, help="cylinder radius (default 0.25 * side)")
    g.add_argument("--snr", type=float, default=20.0, help="power SNR of the data in dB (book definition)")
    g.add_argument("--data-factor", type=int, default=2, help="data are synthesized on a grid this many times "
                   "finer than the inference grid; 1 is the same-grid control (inverse crime)")
    g = ap.add_argument_group("prior")
    g.add_argument("--prior", choices=["matern", "gmrf"], default="matern")
    g.add_argument("--sigma", type=float, default=0.5, help="prior standard deviation of theta (log eps_r)")
    g.add_argument("--ell", type=float, default=0.4, help="prior correlation length in wavelengths")
    g.add_argument("--nu", type=float, default=1.5, help="Matern smoothness (ignored by gmrf)")
    g.add_argument("--mean", type=float, default=0.0, help="prior mean of log eps_r (0 = vacuum background)")
    g.add_argument("--link", choices=["log_eps", "identity"], default="log_eps",
                   help="theta -> chi; identity is for validation (prior then on chi itself)")
    g.add_argument("--born", action="store_true", help="use the Born forward model for inference "
                   "(the data are still generated by the full model)")
    g = ap.add_argument_group("sampler")
    g.add_argument("--sampler", choices=["pcn", "pcn_langevin", "mala", "hmc"], default="mala")
    g.add_argument("--chains", type=int, default=4)
    g.add_argument("--warmup", type=int, default=500, help="step-size adaptation steps per chain")
    g.add_argument("--samples", type=int, default=1000, help="post-warmup steps per chain")
    g.add_argument("--thin", type=int, default=1, help="store every thin-th draw")
    g.add_argument("--step0", type=float, default=None, help="initial step size (default depends on the sampler)")
    g.add_argument("--n-leapfrog", type=int, default=10, help="HMC leapfrog steps")
    g.add_argument("--jitter", type=float, default=0.2, help="HMC step-size jitter (resonance protection)")
    g.add_argument("--init", choices=["prior", "map"], default="map", help="chain starts: the MAP plus --init-noise "
                   "times a standard normal vector, or draws of the prior (far in the tails of an informative "
                   "posterior: Phi is 10^3 times larger there)")
    g.add_argument("--init-noise", type=float, default=0.01, help="std of the perturbation of the MAP start")
    g = ap.add_argument_group("numerics and output")
    g.add_argument("--tol", type=float, default=1e-10, help="forward solver relative tolerance (<= 1e-10 for MCMC)")
    g.add_argument("--map-maxiter", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="posterior", help="output directory: results/<out>/ (or an absolute path)")
    return ap.parse_args(argv)


def make_phantom(grid, args):
    if args.phantom == "cylinder":
        return cylinder(grid, args.radius if args.radius is not None else 0.25 * args.side, args.eps_r)
    if args.phantom == "austria_fixed":
        return austria_fixed(grid, args.eps_r)
    return ch15_austria_variant(np.random.default_rng(args.seed), grid)  # same shapes on every grid


def pixel_indices(n):
    """Five fixed pixels (i, j): the center and the four quarter points."""
    return [(n // 2, n // 2), (n // 4, n // 4), (3 * n // 4, n // 4), (n // 4, 3 * n // 4), (3 * n // 4, 3 * n // 4)]


def series_stats(x, thin=1):
    """ESS (summed over chains), tau (mean over chains, in sampler steps) and split R-hat of x (chains, N)."""
    x = np.asarray(x, dtype=float)
    if x.shape[1] < 8:
        return {"ess": None, "tau": None, "rhat": None}
    ess = float(sum(dg.ess(c) for c in x))
    tau = float(np.mean([dg.tau_int(c) for c in x])) * thin
    rhat = float(dg.split_rhat(x)) if x.shape[0] >= 2 else None
    return {"ess": ess, "tau": tau, "rhat": rhat}


def pooled_mean_std(mean_c, var_c):
    """Mean and standard deviation of the pooled chains from per-chain Welford mean and variance."""
    mean_c, var_c = np.asarray(mean_c), np.asarray(var_c)
    pooled = var_c.mean(axis=0) + mean_c.var(axis=0)
    return mean_c.mean(axis=0), np.sqrt(pooled)


def build_problem(args):
    """Phantom, data (on a finer grid unless --data-factor 1), noise, prior and ``Target`` for ``args``.

    Returns a namespace with setup, data_setup, chi_data, chi_true (coverage-averaged truth on the inference
    grid), eps_true, d_clean, d_obs, s (noise level), forward, field and target.
    """
    n = args.n
    setup = ch15_setup(n=n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    data_setup = Setup(Grid(n * args.data_factor, args.side), setup.antennas)
    chi_data = make_phantom(data_setup.grid, args)
    chi_true = make_phantom(setup.grid, args)  # coverage-averaged truth on the inference grid
    eps_true = 1.0 + chi_true
    print(f"grid n={n} side={args.side} (h={setup.grid.h:.3f}); data grid n={n * args.data_factor}; "
          f"{args.n_tx} tx x {args.n_rx} rx; phantom {args.phantom}, max eps_r {1 + chi_data.max():.2f}; "
          f"pixels per wavelength in the densest medium on the data grid: "
          f"{1.0 / (data_setup.grid.h * np.sqrt(1.0 + chi_data.max())):.1f}")
    data_forward = Forward(data_setup, tol=1e-10)
    d_clean, _ = data_forward(jnp.asarray(chi_data))
    s = noise_std_from_snr(d_clean, args.snr)
    d_obs = add_noise(jax.random.key(args.seed), d_clean, s)

    forward = Forward(setup, tol=args.tol)
    field = GaussianField(setup.grid, args.prior, sigma=args.sigma, ell=args.ell, nu=args.nu, mean=args.mean)
    target = Target(forward, field, args.link, d_obs, s, use_born=args.born)
    return SimpleNamespace(setup=setup, data_setup=data_setup, chi_data=chi_data, chi_true=chi_true,
                           eps_true=eps_true, d_clean=d_clean, d_obs=d_obs, s=s, forward=forward, field=field,
                           target=target)


def window_means(x):
    """Mean over chains of the per-chain means of the first and the last quarter of x (chains, N)."""
    x = np.asarray(x, dtype=float)
    q = max(x.shape[1] // 4, 1)
    return float(x[:, :q].mean()), float(x[:, -q:].mean())


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    load_start = os.getloadavg()[0]
    n = args.n

    # ---- setup, truth and data (on a finer grid unless --data-factor 1) ----
    prob = build_problem(args)
    setup, forward, field, target = prob.setup, prob.forward, prob.field, prob.target
    chi_true, eps_true, d_clean, d_obs, s = prob.chi_true, prob.eps_true, prob.d_clean, prob.d_obs, prob.s

    # ---- how well can this grid represent the truth, in units of the noise? ----
    model = (lambda chi: (forward.born(chi), None)) if args.born else forward  # the model the chains use
    op = operating_point(model, chi_true, d_clean, d_obs, s)
    print(f"operating point: grid model error {100 * op['rel_model_error']:.2f}% of ||d|| against noise "
          f"{100 * op['rel_noise']:.2f}% (ratio {op['model_to_noise']:.2f}); Phi(truth on the inference grid) = "
          f"{op['phi_truth']:.0f} = N {op['phi_truth'] - op['n_data']:+.0f} (N = {op['n_data']}), "
          f"{op['phi_excess_sd']:.1f} sd above the Phi of the noise-free fine-grid data ({op['phi_clean']:.0f})")
    if op["model_to_noise"] > 0.35:
        print("WARNING: the inference grid is too coarse for this noise level (model error above 0.35 of the "
              "noise amplitude): the posterior is conditional on a visibly wrong forward model. Refine --n, "
              "lower --snr, or treat the run as a labeled control (--data-factor 1).")

    # ---- MAP ----
    t0 = time.perf_counter()
    xi_zero = jnp.zeros(field.xi_shape)
    xi_map, chi_map, map_info = map_estimate(target, xi_zero, maxiter=args.map_maxiter)
    eps_map = 1.0 + chi_map
    print(f"MAP: {map_info['nit']} iterations, {map_info['nfev']} evaluations, {map_info['time_s']:.1f} s, "
          f"Phi {map_info['phi']:.1f} (N data = {args.n_tx * args.n_rx}), |grad U|_inf {map_info['grad_inf']:.2e}; "
          f"{map_info['message']}")

    # ---- chains ----
    n_chains = args.chains
    k_init, k_run = jax.random.split(jax.random.key(args.seed + 1))
    if args.init == "map":
        xi0s = xi_map + args.init_noise * jax.random.normal(k_init, (n_chains,) + field.xi_shape)
    else:
        xi0s = field.sample_xi(k_init, n_chains)
    step0 = STEP0[args.sampler] if args.step0 is None else args.step0
    print(f"sampling: {args.sampler}, {n_chains} chains x ({args.warmup} warmup + {args.samples} samples), "
          f"thin {args.thin}, step0 {step0}, init {args.init}")
    res, info = sample(target, args.sampler, k_run, xi0s, n_warmup=args.warmup, n_samples=args.samples,
                       step0=step0, thin=args.thin, n_leapfrog=args.n_leapfrog, jitter=args.jitter)
    wall = info["wall_s"]

    # ---- summaries ----
    samples = np.asarray(res.samples)  # (chains, N/thin, 2, n, n): theta = log eps_r, eps_r
    phi = np.asarray(res.phi)  # (chains, n_samples): the potential the sampler ran on
    # Data misfit Phi: the sampler's own trace unless the sampler reports it separately (info["phi_physical"],
    # one value per stored draw, e.g. Laplace-whitened runs whose trace is the reparametrized Phi').
    phi_kind = info.get("phi_trace_kind", "physical")
    phi_phys = np.asarray(info["phi_physical"]) if "phi_physical" in info else phi
    phys_every = args.thin if "phi_physical" in info else 1
    accept_c = np.asarray(res.accept).mean(axis=1)
    step_c = np.asarray(res.step)
    mean_c, var_c = np.asarray(res.mean), np.asarray(res.var)  # (chains, 2, n, n)
    (theta_mean, eps_mean), (theta_std, eps_std) = pooled_mean_std(mean_c, var_c)
    theta_name = "log_eps_r" if args.link == "log_eps" else "chi"

    phi_stats = series_stats(phi)
    phys_stats = series_stats(phi_phys, phys_every)
    q1, q4 = window_means(phi_phys)
    pix = pixel_indices(n)
    pix_stats = {}
    for (i, j) in pix:
        pix_stats[f"{i},{j}"] = {
            theta_name: series_stats(samples[:, :, 0, i, j], args.thin),
            "eps_r": series_stats(samples[:, :, 1, i, j], args.thin),
        }
    ess_list = [phys_stats["ess"]] + [v[theta_name]["ess"] for v in pix_stats.values()]
    ess_min = min(e for e in ess_list if e is not None) if any(e is not None for e in ess_list) else None

    # reconstructions and the book's metrics (eps_r based eq. 15.21, contrast-normalized, data eq. 15.22)
    d_map, u_map = forward(jnp.asarray(chi_map))
    chi_mean = eps_mean - 1.0
    d_mean, _ = forward(jnp.asarray(chi_mean))
    metrics = {
        "err_model": {"vacuum": err_model(np.ones_like(eps_true), eps_true), "map": err_model(eps_map, eps_true),
                      "posterior_mean": err_model(eps_mean, eps_true)},
        "err_chi": {"vacuum": 1.0, "map": err_chi(chi_map, chi_true), "posterior_mean": err_chi(chi_mean, chi_true)},
        "err_data": {"vacuum": 1.0, "map": err_data(d_map, d_clean), "posterior_mean": err_data(d_mean, d_clean)},
    }
    z = (eps_mean - eps_true) / eps_std
    max_residual = max(target.solver_residual(res.xi[c], res.aux[c]) for c in range(n_chains))

    n_evals = info["n_phi_evals"] * n_chains
    summary = {
        "args": vars(args),
        "noise_std": s,
        "data_grid_n": n * args.data_factor,
        "inverse_crime_control": args.data_factor == 1,
        "acceptance": {"mean": float(accept_c.mean()), "per_chain": accept_c.tolist()},
        "final_step": step_c.tolist(),
        "phi": {**phi_stats, "mean": float(phi.mean()), "std": float(phi.std()), "n_data": args.n_tx * args.n_rx,
                "kind": phi_kind},
        "phi_physical": {**phys_stats, "mean": float(phi_phys.mean()), "std": float(phi_phys.std()),
                         "first_quarter_mean": q1, "last_quarter_mean": q4, "every": phys_every,
                         "n_per_chain": int(phi_phys.shape[1])},
        "operating_point": op,
        "pixels": pix_stats,
        "rhat_max_over_pixels": max([v[theta_name]["rhat"] for v in pix_stats.values() if v[theta_name]["rhat"] is not None],
                                    default=None),
        "wall_time_s": {"total": time.perf_counter() - t_start, "sampling": wall, "compile": info["compile_s"],
                        "map": map_info["time_s"]},
        "forward_solves": {"sampling_phi_evals": n_evals, "sampling_forward_solves": info["n_forward_solves"],
                           "sampling_adjoint_solves": info["n_adjoint_solves"],
                           "map_evaluations": map_info["nfev"], "note": "one solve = all transmitters"},
        "ess_per_s": {"min_over_phi_and_pixels": None if ess_min is None else ess_min / wall,
                      "phi": None if phys_stats["ess"] is None else phys_stats["ess"] / wall},
        "machine": {"cpu_count": os.cpu_count(), "load_avg_1min_at_start": load_start,
                    "load_avg_1min_at_end": os.getloadavg()[0]},
        "metrics": metrics,
        "map": {k: map_info[k] for k in ("success", "message", "nit", "nfev", "U", "phi", "grad_inf", "time_s")},
        "z_eps_r": {"rms": float(np.sqrt(np.mean(z**2))), "frac_abs_gt_2": float(np.mean(np.abs(z) > 2.0))},
        "max_solver_residual_final_states": max_residual,
        "posterior_mean_eps_r_range": [float(eps_mean.min()), float(eps_mean.max())],
        "posterior_std_eps_r_range": [float(eps_std.min()), float(eps_std.max())],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(
        out_dir / "chains.npz", samples=samples.astype(np.float32), phi=phi, phi_physical=phi_phys,
        accept=np.asarray(res.accept),
        step=step_c, welford_mean=mean_c, welford_var=var_c, xi_final=np.asarray(res.xi), xi_map=np.asarray(xi_map),
        chi_map=chi_map, chi_true=chi_true, eps_true=eps_true, d_obs=np.asarray(d_obs), d_clean=np.asarray(d_clean),
        pixels=np.array(pix), noise_std=s)

    # ---- figures ----
    title = (f"{args.phantom}, {args.sampler}, SNR {args.snr:g} dB, {args.prior} sigma={args.sigma} "
             f"ell={args.ell}, link {args.link}")
    plot_maps(eps_true, eps_map, eps_mean, eps_std, setup.grid, out_dir / "maps.png", quantity="eps_r",
              title=title)
    if args.link == "log_eps":
        plot_maps(np.log(eps_true), np.log(eps_map), theta_mean, theta_std, setup.grid, out_dir / "maps_log_eps.png",
                  quantity="log eps_r", title=title)
    extra = {f"{theta_name} pixel ({i},{j})": samples[:, :, 0, i, j] for (i, j) in pix}
    if "phi_physical" in info:
        extra = {"Phi (data misfit, stored draws)": phi_phys, **extra}
    plot_traces(phi, out_dir / "traces.png", extra=extra, title=title,
                phi_label="Phi" if phi_kind == "physical" else "Phi' (sampler potential, not the misfit)")

    # ---- report ----
    m = metrics
    print(f"\nacceptance {accept_c.mean():.3f}, final step {np.mean(step_c):.4g}, Phi (physical misfit) mean "
          f"{phi_phys.mean():.1f} (MAP {map_info['phi']:.1f}, N data {args.n_tx * args.n_rx})"
          + ("" if phi_kind == "physical" else f"; sampler trace mean {phi.mean():.1f} is {phi_kind}"))
    print(f"ESS of Phi {phys_stats['ess']}, tau {phys_stats['tau']}, min ESS over Phi and 5 pixels {ess_min}; "
          f"ESS/s {summary['ess_per_s']['min_over_phi_and_pixels']}; max pixel R-hat {summary['rhat_max_over_pixels']}")
    print(f"err_model (eps_r, eq. 15.21): vacuum {m['err_model']['vacuum']:.4f}, MAP {m['err_model']['map']:.4f}, "
          f"posterior mean {m['err_model']['posterior_mean']:.4f}")
    print(f"err_chi (contrast-normalized): MAP {m['err_chi']['map']:.4f}, posterior mean {m['err_chi']['posterior_mean']:.4f}")
    print(f"err_data (eq. 15.22): MAP {m['err_data']['map']:.4f}, posterior mean {m['err_data']['posterior_mean']:.4f}")
    print(f"z-map of eps_r (mean - truth)/std: rms {summary['z_eps_r']['rms']:.2f}, "
          f"|z|>2 for {100 * summary['z_eps_r']['frac_abs_gt_2']:.0f}% of pixels")
    print(f"forward solves: {info['n_forward_solves']} sampling + {map_info['nfev']} MAP; "
          f"max solver residual at the final states {max_residual:.2e} (tol {args.tol:g}); "
          f"wall {summary['wall_time_s']['total']:.1f} s (sampling {wall:.1f} s, compile {info['compile_s']:.1f} s)")
    print(f"wrote {out_dir}")
    return summary


if __name__ == "__main__":
    main()
