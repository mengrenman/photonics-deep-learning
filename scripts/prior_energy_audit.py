"""What each prior charges for the truth, and what the data charge: the energies behind the edge-ring question.

    python scripts/prior_energy_audit.py [RUN ...] [--dir results/prior_comparison] [--out FILE.json]

The posterior is exp(-U), U = Phi + R (docs/DESIGN.md 11.2). For every run this script evaluates, with the run's own prior
and data (rebuilt from the arguments in summary.json), at three places - the coverage-averaged truth, the MAP, and the stored
posterior draws -

  * the prior energy R: ``R_truth``, ``R_map`` (checked against summary.json), mean and sd of R over the draws, and
    (R_truth - mean R_draws) / sd R_draws, how many posterior sd the truth sits above a typical draw;
  * the data misfit Phi: of the truth (``operating_point.phi_truth``; with the grid error of the n = 48 inference grid),
    of the noise-free data (``phi_clean``, the realized noise energy |d_clean - d_obs|^2 / s^2 = Phi at the exact forward
    model), of the MAP and of the draws (summary.json);
  * U = Phi + R at the truth and at the MAP, and R and Phi of Gaussian-blurred copies of the truth (``blur``, sigma = 0,
    0.5, 1, 1.5, 2 pixels, edge-replicated): the prior energy of a blurred edge against its data cost, the direct
    comparison of what the prior prefers with what the data prefer.

and a one-dimensional table (``step_cost``): the cost of a step of height log 2 (the true edge jump) as one jump
(rho(log 2)) against a ramp of k pixels (k rho(log 2 / k)) for the q-GGMRF settings of the experiment and for the other
members of the family, per pair line (``mrf.rho``, b = 1). For a convex rho with rho(0) = 0, k rho(D / k) <= rho(D) for
every k (Jensen): every convex member of the family prefers a ramp to a sharp step; the ratio sharp / ramp is the size of
the preference. The Gaussian priors are quadratic forms and prefer blur too (``blur``).

Milestone 5 adds the pseudo-Huber TV prior (``mrf.PseudoHuberTV``) to the step-versus-ramp table (``TV_FAMILY``, ``step_cost_tv``:
lam (sqrt(D^2 + delta^2) - delta) per pair line for delta = 0.01, 0.003, 0.001). A ramp of k equal increments saves about
(k - 1) delta of the cost of the step (relative to D = log 2), so the step / 8-pixel-ramp ratio is 1.106 at delta = 0.01 (the DESIGN
12.1 value, ``it should be within a few percent`` holds only for ramps of 2 pixels, 1.014), 1.031 at 0.003 and 1.010 at 0.001, against
2.2 to 2.7 for the q-GGMRF settings of milestone 4. Runs with --prior tv are audited like the others (``audit_run`` rebuilds the
prior from summary.json).

Writes <dir>/prior_energy.json (or --out). One process; the forward solves are few (a handful per distinct data set).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

import bayes_scatter  # noqa: F401
from bayes_scatter import mrf
from bayes_scatter.geometry import Grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_prior_comparison as rpc  # noqa: E402

DEFAULT_RUNS = ["pc_matern", "pc_gmrf", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1", "pcs1_qggmrf_sg0.05", "pcg1_qggmrf_sg0.05",
                "pcg1_matern"]
BLUR_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0)
JUMP = float(np.log(2.0))  # the true edge jump in log eps_r
# (label, q, T, sigma_g): the two settings of the experiment, then members of the family for the question "which member
# does not prefer a ramp"; q = 1 with T sigma_g >= log 2 is quadratic over the whole jump, q = 1 with small T is near total variation
FAMILY = (("q-GGMRF q 1.2, T 1, sigma_g 0.05 (experiment)", 1.2, 1.0, 0.05),
          ("q-GGMRF q 1.2, T 1, sigma_g 0.1 (experiment)", 1.2, 1.0, 0.1),
          ("q 1.0, T 1, sigma_g 0.05", 1.0, 1.0, 0.05),
          ("q 1.0, T sigma_g = log 2 (sigma_g 0.05)", 1.0, JUMP / 0.05, 0.05),
          ("q 1.0, T 0.1, sigma_g 0.05 (near total variation)", 1.0, 0.1, 0.05),
          ("q 1.0, T 0.01, sigma_g 0.05 (total variation)", 1.0, 0.01, 0.05),
          ("q 1.5, T 1, sigma_g 0.05", 1.5, 1.0, 0.05))
RAMPS = (2, 4, 8)
TV_FAMILY = (("TV pseudo-Huber lam 1, delta 0.01 (DESIGN 12.1)", 1.0, 0.01), ("TV pseudo-Huber lam 1, delta 0.003", 1.0, 0.003),
             ("TV pseudo-Huber lam 1, delta 0.001", 1.0, 0.001))


def step_cost(q, T, sigma_g, jump=JUMP, ramps=RAMPS):
    """Cost per pair line of a step of height ``jump`` as one jump and as ramps of k pixels (equal steps), for the pair
    potential ``mrf.rho(delta, q, T, sigma_g)`` (b = 1): ``{"sharp": rho(jump), "ramp_k": k rho(jump / k), "ratio_k":
    sharp / ramp_k}``. A ratio above 1 means the prior prefers the ramp."""
    f = lambda d: float(mrf.rho(jnp.asarray(d, dtype=jnp.float64), q, T, sigma_g))
    out = {"q": q, "T": T, "sigma_g": sigma_g, "jump": jump, "sharp": f(jump)}
    for k in ramps:
        out[f"ramp_{k}"] = k * f(jump / k)
        out[f"ratio_{k}"] = out["sharp"] / out[f"ramp_{k}"]
    return out


def step_cost_tv(lam, delta, jump=JUMP, ramps=RAMPS):
    """``step_cost`` for the pseudo-Huber potential lam (sqrt(D^2 + delta^2) - delta) (``mrf.pseudo_huber``, b = 1); the ratios do not
    depend on lam."""
    f = lambda d: lam * float(mrf.pseudo_huber(jnp.asarray(d, dtype=jnp.float64), delta))
    out = {"lam": lam, "delta": delta, "jump": jump, "sharp": f(jump)}
    for k in ramps:
        out[f"ramp_{k}"] = k * f(jump / k)
        out[f"ratio_{k}"] = out["sharp"] / out[f"ramp_{k}"]
    return out


def energy_over_draws(prior, theta, chunk=200):
    """R over stored draws: theta (chains, N, n, n) -> (chains, N), jitted and vmapped in chunks."""
    f = jax.jit(jax.vmap(prior._energy))
    flat = theta.reshape((-1,) + theta.shape[2:])
    out = np.concatenate([np.asarray(f(jnp.asarray(flat[i:i + chunk], dtype=jnp.float64)))
                          for i in range(0, flat.shape[0], chunk)])
    return out.reshape(theta.shape[:2])


def audit_run(name, base, ctx_cache):
    """Energies of one run directory; ``ctx_cache`` maps (data_factor, seed, eps_r, n, snr) -> (``run_prior_comparison``
    context of that data set, Phi of the blurred truths), built once per distinct data set (the Phi of any theta map
    does not depend on the prior)."""
    d = base / name
    s = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    args = SimpleNamespace(**s["args"])
    grid = Grid(args.n, args.side)
    prior = rpc.build_prior(args, grid)
    truth, theta_hat = z["theta_true"], z["theta_hat"]
    theta = z["theta"].astype(np.float64)
    R = lambda th: float(prior.energy(jnp.asarray(th, dtype=jnp.float64)))
    r_truth, r_map = R(truth), R(theta_hat)
    r_draws = energy_over_draws(prior, theta)
    phi_draws = z["phi_physical"]
    op = s["operating_point"]
    out = {"run": name, "prior": s["prior"]["kind"], "converged": bool(s["convergence"]["converged"]),
           "data": {"data_factor": args.data_factor, "seed": args.seed, "grid_error_over_noise": op["model_to_noise"]},
           "draws": int(r_draws.size),
           "R": {"truth": r_truth, "map": r_map, "map_in_summary": s["map"]["R"], "draws_mean": float(r_draws.mean()),
                 "draws_sd": float(r_draws.std(ddof=1)), "truth_minus_map": r_truth - r_map,
                 "truth_minus_draws_mean": r_truth - float(r_draws.mean()),
                 "truth_above_draws_in_sd": (r_truth - float(r_draws.mean())) / float(r_draws.std(ddof=1))},
           "Phi": {"truth": op["phi_truth"], "noise_free_data": op["phi_clean"], "map": s["map"]["phi"],
                   "draws_mean": float(phi_draws.mean()), "draws_sd": float(phi_draws.std(ddof=1)),
                   "truth_minus_map": op["phi_truth"] - s["map"]["phi"], "n_data": s["data"]["n_data"]},
           "U": {"truth": op["phi_truth"] + r_truth, "map": s["map"]["phi"] + r_map,
                 "truth_minus_map": op["phi_truth"] + r_truth - s["map"]["phi"] - r_map}}
    out["blur"] = {"sigma_px": list(BLUR_SIGMAS),
                   "R": [R(gaussian_filter(truth, sg, mode="nearest")) if sg > 0 else r_truth for sg in BLUR_SIGMAS]}
    key = (args.data_factor, args.seed, args.eps_r, args.n, args.snr)
    if key not in ctx_cache:
        ctx = rpc.build_context(args)  # the data of this run (n, data_factor, seed): a few forward solves
        phi_of = lambda th: float(ctx.target.misfit(th)[0])
        ctx_cache[key] = (ctx, [phi_of(gaussian_filter(truth, sg, mode="nearest") if sg > 0 else truth) for sg in BLUR_SIGMAS])
    ctx, phi_blur = ctx_cache[key]
    out["blur"]["Phi"] = phi_blur
    out["blur"]["Phi_note"] = ("Phi of the isotropic Gaussian blur of the truth with this run's data (shared by the runs "
                               "with the same data). The posterior does not blur isotropically: it moves only in directions "
                               "the data barely constrain, see posterior_mean")
    theta_mean = z["welford_mean"].mean(axis=0)[0]  # pooled posterior mean of theta (equal-length chains)
    phi_mean = float(ctx.target.misfit(theta_mean)[0])
    r_mean = R(theta_mean)
    out["posterior_mean"] = {"R": r_mean, "Phi": phi_mean, "R_truth_minus_mean": r_truth - r_mean,
                             "Phi_mean_minus_truth": phi_mean - op["phi_truth"],
                             "note": "the posterior-mean map: the data cost of moving from the truth to it against what the prior saves"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--out", default=None, help="default <dir>/prior_energy.json")
    args = ap.parse_args(argv)
    base = Path(args.dir)
    cache, runs = {}, {}
    for r in args.runs:
        t0 = time.time()
        runs[r] = audit_run(r, base, cache)
        e = runs[r]
        print(f"{r:20s} R truth {e['R']['truth']:9.1f}  MAP {e['R']['map']:9.1f}  draws {e['R']['draws_mean']:9.1f} +- {e['R']['draws_sd']:.1f}  "
              f"Phi truth {e['Phi']['truth']:7.1f} MAP {e['Phi']['map']:7.1f} draws {e['Phi']['draws_mean']:7.1f}  "
              f"R(blur 2px) {e['blur']['R'][-1]:9.1f}  ({time.time() - t0:.0f} s)")
    steps = [{"label": lab, **step_cost(q, T, sg)} for lab, q, T, sg in FAMILY]
    steps += [{"label": lab, **step_cost_tv(lam, delta)} for lab, lam, delta in TV_FAMILY]
    out = {"runs": runs, "step_cost": steps, "jump": JUMP,
           "note": ("R = prior energy, Phi = data misfit, U = Phi + R; the posterior is exp(-U). The truth is the "
                    "coverage-averaged phantom on the inference grid. step_cost is per pair line (b = 1); ratio_k = sharp / ramp of k pixels.")}
    out_path = Path(args.out) if args.out else base / "prior_energy.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    for st in steps:
        print(f"  {st['label']:56s} sharp {st['sharp']:7.2f}  ramp8 {st['ramp_8']:7.2f}  ratio {st['ratio_8']:.2f}")
    print(f"wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
