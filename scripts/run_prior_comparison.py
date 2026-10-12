"""Run ONE prior of the prior-sensitivity experiment (docs/DESIGN.md sections 11.2 and 11.3) on the shared problem.

Problem (11.3): ``austria_fixed`` with eps_r = --eps-r (2) on a --side (2) wavelength DOI, book acquisition rings
(40 transmitters, 60 receivers), data synthesized on a grid --data-factor (2) times finer than the inference grid
--n (48) at --snr (30) dB. The data realization depends on --seed, --n, --side, --eps-r, --snr, the antenna counts and
the data factor, and on nothing else (in particular not on the prior or the chain settings), so every prior run with the
same data arguments sees the same ``d_obs`` up to floating-point round-off. The bits are NOT reproducible across XLA thread
counts (the Krylov solves sum in a different order): the SHA-256 in summary.json (``data.d_obs_sha256``) is the same for
runs at one thread count and differs between thread counts although the arrays agree to about 1e-14 of the noise std
(results/prior_comparison/data_thread_check.json). The equality test is therefore numerical: max |d_a - d_b| / s <= 1e-6
(``D_OBS_TOL``), applied by ``load_prepared`` (stored d_obs) and by scripts/compare_priors.py, which refuses to compare
runs whose data differ by more; ``data.thread_env`` stores the thread environment next to the hash.

Pipeline: phantom and data -> operating point -> smooth-prior MAP start (``thetaspace.smooth_start``, Matern 0.5/0.4,
--start-maxiter) -> MAP of U = Phi + R in theta = log eps_r by L-BFGS-B (``thetaspace.map_theta``) -> dense Laplace
mass H = H_GN + Hess R (exact Jacobian by reciprocity) and its Cholesky factor -> HMC in zeta (theta = theta_hat +
L_H^-T zeta, chains started from scale --init-scale overdispersed Laplace draws) -> summaries.

Priors (--prior): ``matern`` and ``gmrf`` are the DOI marginals of the torus fields of DESIGN 3 (--sigma, --ell, --nu,
--mean; Cholesky of the dense covariance, so n <= 59), ``qggmrf`` is the edge-preserving q-GGMRF (--q, --T, --sigma-g,
--sigma-0, --mu) and ``tv`` is the pseudo-Huber total-variation prior of DESIGN 12.1 (--lam, --delta, --sigma-0, --mu; the
same neighborhood and weights as the q-GGMRF). The prior std of ``qggmrf`` and ``tv`` comes from HMC on R alone.

Writes results/prior_comparison/<out>/ (or an absolute --out):
  summary.json : arguments, d_obs hash, operating point, smooth start and MAP info, Laplace info, sampler info, the
      physical misfit Phi of the stored draws (never the sampler's Phi'), convergence numbers and verdict per DESIGN
      11.2 (split R-hat of Phi <= 1.01, max pixel R-hat <= 1.05, worst-pixel Sokal ESS >= 200), prior and posterior
      standard deviations, MCMC / Laplace standard deviation ratios, |z| > 3 statistics, err_model / err_chi /
      err_data, the three feature probabilities of 11.3, timings.
  chains.npz   : thinned theta draws (float32, chains x draws x n x n), Welford mean and variance of (theta, eps_r) over
      all post-warmup steps, the physical Phi of the draws, the sampler trace Phi', acceptance, MAP, Laplace std, prior
      std, truth, d_obs.
  maps.png (theta maps and z-map), traces.png.

Starts and cost. The default chain starts are theta_hat + L_H^-T (2 N(0, I)) as in DESIGN 11.2. For priors whose
Laplace approximation is poor these are very far from the posterior: for the n = 32 GMRF the physical Phi of such
starts is 4.7e6 (3.8e5 at scale 1, 2.5e4 at scale 0.5, against N = 2400 at the posterior), the first trajectories are
rejected at solver-cap cost (about 1.4 s per evaluation instead of 0.06 s), and the chain then drifts outward in the
prior-dominated directions (|zeta|^2 grows from 1e3 to over 9e3 in 120 trajectories) instead of settling: read the
convergence verdict, and lower --init-scale to shorten the transient. The Matern and q-GGMRF runs of the same size
settle within the warmup.

Staged execution (--stage). The default ``all`` runs everything in one process, the chains one after the other. For
a long run on a machine where one thread per process is the most efficient use of the cores (measured in cost.json:
four threads speed one evaluation up only 2.0 to 2.2 times, so four single-thread processes do about 1.8 times the
work of one four-thread process), the work is split into

  prepare   data, MAP, Laplace mass check, prior std -> <out>/prepared.json, prepared.npz (theta_hat, prior std);
  chain     ONE chain (--chain-id) for --samples trajectories -> <out>/chain_<id>_seg<segment>.npz. Segment 0 starts
            the chain (--warmup adaptation trajectories, then --samples); segment j > 0 continues segment j - 1 from its
            final state and its frozen step size for --samples more trajectories, so a chain that has not converged
            can be extended without losing anything. The start and the PRNG stream of chain c are the same functions
            of (--chain-seed, c, segment) in every stage, so ``all`` and the staged runs agree;
  finalize  all chain files -> summary.json, chains.npz and the figures, exactly as ``all`` writes them.

Mass and starts (--mass-theta, --start-theta). By default the chains run in the Laplace-whitened coordinates of the MAP
(DESIGN 11.2) and start from overdispersed Laplace draws. For a prior whose posterior is far from its Laplace
approximation at the MAP (the GMRF: the typical sample is rough and the MAP is smooth, so the local Gauss-Newton
Hessian along the chain is far stiffer than H and the step size collapses), the mass can be rebuilt at a typical state:
--mass-theta FILE.npy ((n, n) array) sets the center and the mass H = H_GN(theta_m) + Hess R of the chains. This is
exact (an affine change of variables, DESIGN 11.2) and only changes the efficiency. --start-theta FILE.npy ((chains, n, n)
or (n, n)) replaces the overdispersed starts, for instance by the end states of a pilot. Both are stored in the
segment files, so continuations and ``finalize`` follow the chains' own mass. The Laplace standard deviation in the
summary is always that of the MAP.

Adapted affine map (--adapt, DESIGN 12.2). For a posterior far from its Laplace approximation (the GMRF) the fixed Laplace map
of the default path forces a tiny step and the chains drift. With ``--adapt`` the stages are

  prepare   unchanged (data, MAP, Laplace mass);
  adapt     ONE adaptation chain (``bayes_scatter.adapt.adapt_map``) from the Laplace map of the MAP (or of --mass-theta) started at
            --adapt-start-scale N(0, I) in its coordinates: an initial step-size buffer, --adapt-windows expanding windows
            (--adapt-first-window trajectories, multiplied by --adapt-growth) each ending in a map update (re-center at the window
            mean, shrink the sample covariance toward the Laplace covariance at the new center, weight kappa / (n + kappa), kappa =
            --adapt-kappa-per-dim * n^2), and a final step-size buffer; HMC with L uniform in [L0 / 2, 3 L0 / 2], L0 = round(--traj-length
            / eps), eps by dual averaging toward --target-accept. Writes <out>/adapted_map.npz (center, Cholesky factor of the map
            covariance, eps, trajectory length) and <out>/adaptation.json (the arguments, the history of every phase, timings);
  chain     ONE chain, as before, but in the coordinates of the FROZEN adapted map, started at --init-scale N(0, I) in them around the
            final state of the adaptation chain (--adapt-start final, the default: an actual visited state) or around the map center
            (--adapt-start center: the mean of the last window, which for a non-Gaussian posterior need not be a probable state; in
            the GMRF pilot every proposal from there was rejected), --warmup trajectories of dual averaging of eps only (map frozen;
            rounded up to a multiple of --adapt-block), then --samples recorded ones with eps and the trajectory length frozen. Every recorded segment is therefore an
            exact HMC chain for a fixed affine change of variables; segment files carry the digest of the map they used and finalize
            refuses a mixture. --n-leapfrog and --jitter do not apply (the trajectory length is --traj-length, jittered in L);
  finalize  unchanged (the summary gains an ``adaptation`` block; the adapted run is recognized from its segment files).

Every stage takes the same problem, prior and chain arguments; ``chain`` and ``finalize`` refuse a prepared.json
made with different data or prior arguments. The Laplace mass is rebuilt from theta_hat in each stage (about 0.5 s).

Example (the DESIGN 11.3 problem, one prior):
  python scripts/run_prior_comparison.py --prior qggmrf --sigma-g 0.05 --out qggmrf_sg005
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import dataclasses
import functools
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import adapt as ad
from bayes_scatter import diagnostics as dg
from bayes_scatter import features as ft
from bayes_scatter import inference as inf
from bayes_scatter import mrf
from bayes_scatter import samplers as sm
from bayes_scatter import thetaspace as ts
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr
from bayes_scatter.phantoms import austria_fixed
from bayes_scatter.plotting import plot_maps, plot_traces
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "prior_comparison"
print = functools.partial(print, flush=True)  # progress is visible while a long run is going

D_OBS_TOL = 1e-6  # two data sets are the same realization if max |d_a - d_b| / s is below this (noise std units)

# Convergence criteria of DESIGN 11.2.
RHAT_PHI_MAX, RHAT_PIXEL_MAX, ESS_PIXEL_MIN = 1.01, 1.05, 200.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("problem and data (identical for every prior being compared)")
    g.add_argument("--n", type=int, default=48, help="pixels per side of the inference grid")
    g.add_argument("--side", type=float, default=2.0, help="DOI side in wavelengths")
    g.add_argument("--eps-r", type=float, default=2.0, help="relative permittivity of austria_fixed")
    g.add_argument("--snr", type=float, default=30.0, help="power SNR of the data in dB")
    g.add_argument("--seed", type=int, default=0, help="seed of the data realization (noise)")
    g.add_argument("--data-factor", type=int, default=2, help="data grid = this many times finer than --n")
    g.add_argument("--n-tx", type=int, default=40)
    g.add_argument("--n-rx", type=int, default=60)
    g = ap.add_argument_group("prior")
    g.add_argument("--prior", choices=["matern", "gmrf", "qggmrf", "tv"], required=True)
    g.add_argument("--sigma", type=float, default=0.5, help="matern/gmrf: prior std of theta = log eps_r")
    g.add_argument("--ell", type=float, default=0.4, help="matern/gmrf: correlation length in wavelengths")
    g.add_argument("--nu", type=float, default=1.5, help="matern smoothness")
    g.add_argument("--mean", type=float, default=0.0, help="matern/gmrf: prior mean of theta")
    g.add_argument("--q", type=float, default=1.2, help="qggmrf: tail exponent in [1, 2)")
    g.add_argument("--T", type=float, default=1.0, help="qggmrf: transition at |Delta| = T sigma_g")
    g.add_argument("--sigma-g", type=float, default=0.1, help="qggmrf: neighbor difference scale (log eps_r units)")
    g.add_argument("--sigma-0", type=float, default=1.0, help="qggmrf/tv: weak anchor to the mean")
    g.add_argument("--mu", type=float, default=0.0, help="qggmrf/tv: prior mean of theta")
    g.add_argument("--lam", type=float, default=1.0, help="tv: total-variation weight per unit change of log eps_r")
    g.add_argument("--delta", type=float, default=0.01, help="tv: pseudo-Huber width (log eps_r units)")
    g = ap.add_argument_group("MAP and sampler")
    g.add_argument("--start-maxiter", type=int, default=100, help="iterations of the smooth-prior MAP start")
    g.add_argument("--map-maxiter", type=int, default=400, help="L-BFGS-B iterations of the theta-space MAP")
    g.add_argument("--map-gtol", type=float, default=1e-6)
    g.add_argument("--chains", type=int, default=4)
    g.add_argument("--warmup", type=int, default=300, help="step-size adaptation trajectories per chain")
    g.add_argument("--samples", type=int, default=1000, help="post-warmup trajectories per chain")
    g.add_argument("--thin", type=int, default=1, help="store every thin-th draw")
    g.add_argument("--n-leapfrog", type=int, default=10)
    g.add_argument("--jitter", type=float, default=0.2, help="HMC step-size jitter (resonance protection)")
    g.add_argument("--step0", type=float, default=0.3, help="initial leapfrog step in zeta")
    g.add_argument("--init-scale", type=float, default=2.0, help="chain starts: theta_hat + L_H^-T (scale N(0, I))")
    g.add_argument("--chain-seed", type=int, default=1, help="seed of the chain starts and the sampler")
    g.add_argument("--verify-columns", type=int, default=3, help="Jacobian columns checked against jax.jvp")
    g = ap.add_argument_group("staged execution (module docstring)")
    g.add_argument("--stage", choices=["all", "prepare", "adapt", "chain", "finalize"], default="all")
    g.add_argument("--chain-id", type=int, default=None, help="chain stage: which chain, 0 .. --chains - 1")
    g.add_argument("--segment", type=int, default=0,
                   help="chain stage: 0 starts the chain (--warmup + --samples); j > 0 continues segment j - 1 "
                        "(--samples more, step size frozen)")
    g.add_argument("--mass-theta", default=None, metavar="FILE.npy",
                   help="chain/all stage, segment 0: build the Laplace mass and center the chains at this theta "
                        "(n, n) instead of the MAP (later segments follow their predecessor)")
    g.add_argument("--start-theta", default=None, metavar="FILE.npy",
                   help="chain/all stage, segment 0: start chain c at theta[c] ((chains, n, n) or (n, n)) instead of "
                        "an overdispersed Laplace draw")
    g.add_argument("--max-segments", type=int, default=None,
                   help="finalize stage: use only the first K segments of every chain (default: all, which must "
                        "be the same number for every chain)")
    g.add_argument("--skip-segments", type=int, default=0,
                   help="finalize stage: discard the first K segments of every chain as burn-in (more warmup: they are counted in "
                        "'warmup' and in the cost, not in the draws, the moments or the convergence numbers; default 0)")
    g = ap.add_argument_group("adapted affine map (DESIGN 12.2; module docstring)")
    g.add_argument("--adapt", action="store_true",
                   help="sample in the coordinates of an affine map adapted in warmup and frozen before recording (stages adapt, "
                        "chain, all; finalize recognizes the adapted segments)")
    g.add_argument("--adapt-init", type=int, default=50, help="adapt: step-size-only trajectories before the first window")
    g.add_argument("--adapt-first-window", type=int, default=50, help="adapt: trajectories in the first map-update window")
    g.add_argument("--adapt-windows", type=int, default=4, help="adapt: number of map-update windows")
    g.add_argument("--adapt-growth", type=float, default=2.0, help="adapt: window length multiplier (expanding windows)")
    g.add_argument("--adapt-final", type=int, default=100, help="adapt: step-size-only trajectories after the last window")
    g.add_argument("--adapt-block", type=int, default=25, help="adapt: trajectories per compiled block (phases are rounded up)")
    g.add_argument("--adapt-kappa-per-dim", type=float, default=1.0,
                   help="adapt: pseudo-count of the Laplace covariance in the map update, in units of n^2 draws")
    g.add_argument("--adapt-start-scale", type=float, default=0.5,
                   help="adapt: the adaptation chain starts at scale * N(0, I) in the coordinates of the Laplace map")
    g.add_argument("--adapt-step0", type=float, default=0.3, help="adapt: first leapfrog step in whitened coordinates")
    g.add_argument("--adapt-start", choices=["final", "center"], default="final",
                   help="chain stage with --adapt: the --init-scale starts are around the final state of the adaptation chain "
                        "(default) or around the center of the adapted map")
    g.add_argument("--target-accept", type=float, default=0.8, help="dual-averaging target acceptance probability")
    g.add_argument("--traj-length", type=float, default=1.75, help="L0 * eps in whitened units (DESIGN 12.2: 1.5 to 2)")
    g.add_argument("--max-leapfrog", type=int, default=100, help="cap on L0 and on the longest trajectory")
    g = ap.add_argument_group("prior-only sampling (qggmrf, tv)")
    g.add_argument("--prior-chains", type=int, default=4)
    g.add_argument("--prior-warmup", type=int, default=200)
    g.add_argument("--prior-samples", type=int, default=600, help="0 skips the prior-only run (std left empty)")
    g = ap.add_argument_group("numerics and output")
    g.add_argument("--tol", type=float, default=1e-10, help="forward solver relative tolerance (<= 1e-10 for MCMC)")
    g.add_argument("--out", required=True, help="results/prior_comparison/<out>/ (or an absolute path)")
    args = ap.parse_args(argv)
    if args.prior == "qggmrf" and not 1.0 <= args.q < 2.0:
        ap.error("--q must be in [1, 2)")
    if args.stage == "chain" and (args.chain_id is None or not 0 <= args.chain_id < args.chains):
        ap.error("--stage chain needs --chain-id in 0 .. --chains - 1")
    if args.stage == "adapt" and not args.adapt:
        ap.error("--stage adapt needs --adapt")
    if args.adapt and args.stage == "chain" and args.mass_theta is not None:
        ap.error("--adapt chains take their coordinates from adapted_map.npz; --mass-theta belongs to --stage adapt")
    if args.adapt and args.start_theta is not None and args.stage == "adapt":
        ap.error("--start-theta applies to the chains, not to the adaptation chain (use --adapt-start-scale)")
    if args.segment < 0:
        ap.error("--segment must be >= 0")
    return args


def d_obs_hash(d_obs):
    """SHA-256 of the complex128 bytes of d_obs (C order)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(d_obs, dtype=np.complex128)).tobytes()).hexdigest()


def d_obs_max_diff(d_a, d_b, s):
    """max |d_a - d_b| / s over all data: the difference of two data sets in units of the noise std ``s``."""
    return float(np.max(np.abs(np.asarray(d_a) - np.asarray(d_b))) / s)


def thread_env():
    """The thread-related environment of this process (strings or None), stored next to the d_obs hash."""
    keys = ("NPROC", "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "JAX_PLATFORMS")
    return {k: os.environ.get(k) for k in keys}


def build_data(args):
    """Phantom, noisy data on the fine grid and the inference-grid forward model.

    Deterministic in (seed, n, side, eps_r, snr, data_factor, antenna counts): returns a namespace with setup,
    forward, chi_true / eps_true / theta_true (coverage-averaged truth on the inference grid), d_clean, d_obs, s and the
    SHA-256 of d_obs.
    """
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    data_setup = Setup(Grid(args.n * args.data_factor, args.side), setup.antennas)
    chi_data = austria_fixed(data_setup.grid, args.eps_r)
    chi_true = austria_fixed(setup.grid, args.eps_r)
    d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(chi_data))
    s = noise_std_from_snr(d_clean, args.snr)
    d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
    return SimpleNamespace(setup=setup, forward=Forward(setup, tol=args.tol), chi_true=chi_true,
                           eps_true=1.0 + chi_true, theta_true=np.log1p(chi_true), d_clean=d_clean, d_obs=d_obs, s=s,
                           sha=d_obs_hash(d_obs))


def build_prior(args, grid):
    """The ``mrf`` prior of ``--prior`` on ``grid``."""
    if args.prior == "qggmrf":
        return mrf.QGGMRF(grid.n, q=args.q, T=args.T, sigma_g=args.sigma_g, sigma_0=args.sigma_0, mu=args.mu)
    if args.prior == "tv":
        return mrf.PseudoHuberTV(grid.n, lam=args.lam, delta=args.delta, sigma_0=args.sigma_0, mu=args.mu)
    field = GaussianField(grid, args.prior, sigma=args.sigma, ell=args.ell, nu=args.nu, mean=args.mean)
    return mrf.GaussianEnergy(field)


def map_stats(a, inside):
    """Mean of a map over all pixels, inside the object and outside, plus min and max."""
    return {"mean": float(a.mean()), "inside": float(a[inside].mean()), "outside": float(a[~inside].mean()),
            "min": float(a.min()), "max": float(a.max())}


def prior_summary(args, prior, grid, inside):
    """Prior std of theta and eps_r: exact for the Gaussian fields, by HMC on R alone for the q-GGMRF and the TV prior."""
    if args.prior not in ("qggmrf", "tv"):
        sd_map = prior.prior_std()
        sd = float(prior.field.sigma)
        mu = float(prior.field.mean.mean())
        eps_sd = float(np.sqrt((np.exp(sd**2) - 1.0) * np.exp(2.0 * mu + sd**2)))  # log-normal marginal
        return {"method": "exact (stationary Gaussian field)", "theta_std": map_stats(sd_map, inside),
                "eps_std": {"mean": eps_sd}}, sd_map
    if args.prior_samples <= 0:
        return {"method": "skipped (--prior-samples 0)"}, None
    t0 = time.perf_counter()
    out = mrf.sample_prior(prior, jax.random.key(args.chain_seed + 100), n_chains=args.prior_chains,
                           n_warmup=args.prior_warmup, n_samples=args.prior_samples)
    info = {"method": "HMC on R alone in coordinates of the Hessian of R at the mean",
            "theta_std": map_stats(out["theta_std"], inside), "eps_std": map_stats(out["eps_std"], inside),
            "theta_mean": map_stats(out["theta_mean"], inside), "accept": out["accept"], "step": out["step"],
            "rhat_max": out["rhat_max"], "ess_min": out["ess_min"], "chains": args.prior_chains,
            "warmup": args.prior_warmup, "samples": args.prior_samples, "time_s": time.perf_counter() - t0}
    return info, out["theta_std"]


def convergence(phi_phys, theta):
    """DESIGN 11.2 convergence numbers: split R-hat of the physical Phi, max pixel R-hat of theta, worst-pixel Sokal ESS
    (summed over chains; Geyer as a cross-check) of theta. ``phi_phys`` (chains, N), ``theta`` (chains, N, n, n)."""
    chains, n_draws = theta.shape[:2]
    flat = theta.reshape(chains, n_draws, -1).astype(np.float64)
    ok = chains >= 2 and n_draws >= 8
    rhat_phi = float(dg.split_rhat(phi_phys)) if ok else float("nan")
    rhat_pix = dg.split_rhat(flat) if ok else np.full(flat.shape[2], np.nan)
    ess_sokal = np.stack([dg.ess(c) for c in flat]).sum(axis=0)
    ess_geyer = np.stack([dg.ess_geyer(c) for c in flat]).sum(axis=0)
    out = {"rhat_phi": rhat_phi, "rhat_pixel_max": float(np.nanmax(rhat_pix)),
           "rhat_pixel_median": float(np.nanmedian(rhat_pix)),
           "ess_pixel_min": float(ess_sokal.min()), "ess_pixel_median": float(np.median(ess_sokal)),
           "ess_pixel_min_geyer": float(np.nanmin(ess_geyer)),
           "ess_phi": float(sum(dg.ess(c) for c in phi_phys)),
           "thresholds": {"rhat_phi": RHAT_PHI_MAX, "rhat_pixel": RHAT_PIXEL_MAX, "ess_pixel_min": ESS_PIXEL_MIN}}
    reasons = []
    if not rhat_phi <= RHAT_PHI_MAX:
        reasons.append(f"split R-hat of Phi {rhat_phi:.3f} > {RHAT_PHI_MAX}")
    if not out["rhat_pixel_max"] <= RHAT_PIXEL_MAX:
        reasons.append(f"max pixel R-hat {out['rhat_pixel_max']:.3f} > {RHAT_PIXEL_MAX}")
    if not out["ess_pixel_min"] >= ESS_PIXEL_MIN:
        reasons.append(f"worst-pixel ESS {out['ess_pixel_min']:.0f} < {ESS_PIXEL_MIN:.0f}")
    out["converged"] = not reasons
    out["unconverged_reasons"] = reasons
    return out


# ----------------------------------------------------------------------------------------------
# Problem, MAP and Laplace mass (stage prepare)
# ----------------------------------------------------------------------------------------------

# Arguments that define the posterior; chain and finalize stages refuse a prepared.json that differs in any of them.
POSTERIOR_ARGS = ("n", "side", "eps_r", "snr", "seed", "data_factor", "n_tx", "n_rx", "prior", "sigma", "ell", "nu",
                  "mean", "q", "T", "sigma_g", "sigma_0", "mu", "tol", "lam", "delta")


def build_context(args):
    """Phantom, data, prior and ``ThetaTarget`` for ``args`` as a namespace (no MAP yet)."""
    prob = build_data(args)
    grid = prob.setup.grid
    prior = build_prior(args, grid)
    target = ts.ThetaTarget(prob.forward, prior, prob.d_obs, prob.s)
    return SimpleNamespace(args=args, prob=prob, grid=grid, prior=prior, target=target,
                           inside=prob.chi_true > 1e-9, n_data=args.n_tx * args.n_rx,
                           potential=target.potential_flat())  # one flat-potential object: adapt compiles its block once


def prepare(ctx):
    """Operating point, prior std, smooth start, MAP and Laplace mass of ``ctx``.

    Returns a namespace: ``theta_hat`` (n, n) NumPy, ``mass`` (``thetaspace.LaplaceMass``), ``prior_std_map`` ((n, n)
    or None), ``U_history`` (U at every MAP function evaluation) and ``info``, the JSON-serializable dict
    (``data``, ``operating_point``, ``prior_info``, ``start``, ``map``, ``laplace``, ``timings``) that prepared.json
    stores and the summary reuses.
    """
    args, prob, grid, target, prior = ctx.args, ctx.prob, ctx.grid, ctx.target, ctx.prior
    n, s = args.n, prob.s
    timings = {}
    print(f"problem: austria_fixed eps_r {args.eps_r}, n={n} side={args.side} (h={grid.h:.4f}), data grid "
          f"{n * args.data_factor}, SNR {args.snr} dB, seed {args.seed}; d_obs sha256 {prob.sha[:16]}...")
    op = inf.operating_point(prob.forward, jnp.asarray(prob.chi_true), prob.d_clean, prob.d_obs, s)
    print(f"operating point: grid model error {100 * op['rel_model_error']:.2f}% of ||d|| against noise "
          f"{100 * op['rel_noise']:.2f}% (ratio {op['model_to_noise']:.2f}); Phi(truth) {op['phi_truth']:.0f} "
          f"(N = {ctx.n_data})")
    if op["model_to_noise"] > 0.35:
        print("WARNING: the inference grid is too coarse for this noise level (model error above 0.35 of the noise).")
    t0 = time.perf_counter()
    prior_info, prior_std_map = prior_summary(args, prior, grid, ctx.inside)
    timings["prior_std_s"] = time.perf_counter() - t0
    print(f"prior {prior.describe()['kind']}: std of theta (mean over pixels) {prior_info.get('theta_std', {}).get('mean')}")

    t0 = time.perf_counter()
    theta0, start_info = ts.smooth_start(prob.forward, prob.d_obs, s, grid, maxiter=args.start_maxiter)
    timings["start_s"] = time.perf_counter() - t0
    parts0 = target.energies(theta0)
    print(f"smooth start: {start_info['nit']} iterations, {start_info['nfev']} evaluations, {timings['start_s']:.1f} s; "
          f"U {parts0['U']:.1f} (Phi {parts0['phi']:.1f}, R {parts0['R']:.1f})")
    theta_hat, map_info = ts.map_theta(target, theta0, maxiter=args.map_maxiter, gtol=args.map_gtol)
    print(f"MAP: {map_info['nit']} iterations, {map_info['nfev']} evaluations, {map_info['time_s']:.1f} s; U "
          f"{map_info['U']:.2f} = Phi {map_info['phi']:.2f} + R {map_info['R']:.2f} (N = {ctx.n_data}); |grad U|_inf "
          f"{map_info['grad_inf']:.2e}; {map_info['message']}")

    mass = target.laplace_mass(theta_hat, verify_columns=args.verify_columns)
    lap_std = mass.theta_std()
    print(f"Laplace mass: {mass.info['time_total_s']:.2f} s (fields {mass.info['time_fields_s']:.2f}, Gram "
          f"{mass.info['time_gn_s']:.2f}, Cholesky {mass.info['time_cholesky_s']:.2f}); Jacobian check "
          f"{mass.info.get('jacobian_check_rel_err')}; Laplace std of theta {lap_std.min():.4f} to {lap_std.max():.4f}")
    info = {
        "args": vars(args),
        "data": {"d_obs_sha256": prob.sha, "noise_std": s, "n_data": ctx.n_data, "data_grid_n": n * args.data_factor,
                 "snr_db": args.snr, "seed": args.seed, "thread_env": thread_env()},
        "operating_point": op,
        "prior_info": prior_info,
        "start": {"maxiter": args.start_maxiter, "nit": start_info["nit"], "nfev": start_info["nfev"],
                  "time_s": timings["start_s"], "phi": start_info["phi"],
                  "note": "whitened-coordinate Matern(0.5, 0.4, 1.5) MAP, theta = log eps_r of the result"},
        "map": {k: map_info[k] for k in ("success", "message", "nit", "nfev", "U", "phi", "R", "grad_inf", "time_s")}
        | {"phi_over_n_data": map_info["phi"] / ctx.n_data, "maxiter": args.map_maxiter},
        "laplace": {**mass.info, "theta_std": map_stats(lap_std, ctx.inside)},
        "timings": timings,
    }
    return SimpleNamespace(theta_hat=np.asarray(theta_hat), mass=mass, prior_std_map=prior_std_map,
                           U_history=map_info["U_history"], info=info, d_obs=np.asarray(prob.d_obs))


def save_prepared(prep, out_dir):
    """Write prepared.json and prepared.npz (theta_hat, prior std, MAP U history)."""
    (out_dir / "prepared.json").write_text(json.dumps(prep.info, indent=2, default=float))
    np.savez(out_dir / "prepared.npz", theta_hat=prep.theta_hat, U_history=prep.U_history,
             prior_theta_std=np.full(prep.theta_hat.shape, np.nan) if prep.prior_std_map is None else prep.prior_std_map,
             d_obs=np.asarray(prep.d_obs))


def load_prepared(ctx, out_dir):
    """The namespace of ``prepare`` from prepared.json / prepared.npz, with the Laplace mass rebuilt at theta_hat.

    Exits if the stored arguments differ from ``ctx.args`` in any data or prior argument (``POSTERIOR_ARGS``).
    """
    info = json.loads((out_dir / "prepared.json").read_text())
    stored = info["args"]
    # a prepared.json from before milestone 5 has no "lam" / "delta" (they only matter for --prior tv): skip absent keys
    diff = {k: (stored[k], getattr(ctx.args, k)) for k in POSTERIOR_ARGS if k in stored and stored[k] != getattr(ctx.args, k)}
    if diff:
        raise SystemExit(f"{out_dir}/prepared.json was made with different data or prior arguments: {diff}")
    z = np.load(out_dir / "prepared.npz")
    if info["data"]["d_obs_sha256"] != ctx.prob.sha:
        if "d_obs" in z.files:  # the hash is not reproducible across thread counts: compare the numbers
            diff = d_obs_max_diff(z["d_obs"], ctx.prob.d_obs, ctx.prob.s)
            if diff > D_OBS_TOL:
                raise SystemExit(f"{out_dir}: the data differ from the prepared run's (max |delta d| / s = {diff:.2e} > "
                                 f"{D_OBS_TOL:g})")
            print(f"d_obs hash differs from the prepared run's but the arrays agree to {diff:.1e} of the noise std "
                  f"(round-off across thread counts); continuing")
        else:
            print("WARNING: d_obs differs from the prepared run's in its SHA-256 and prepared.npz has no d_obs to compare "
                  "(made before d_obs was stored); the data may differ by round-off or more")
    prior_std = z["prior_theta_std"]
    mass = ctx.target.laplace_mass(z["theta_hat"])
    return SimpleNamespace(theta_hat=z["theta_hat"], mass=mass, U_history=z["U_history"], info=info,
                           prior_std_map=None if np.all(np.isnan(prior_std)) else prior_std)


# ----------------------------------------------------------------------------------------------
# Chains (stage chain)
# ----------------------------------------------------------------------------------------------


def make_zeta_target(ctx, prep, mass_theta=None):
    """The sampling coordinates: ``ZetaTarget`` of the MAP Laplace mass (``mass_theta`` None) or of the mass H(theta_m)
    centered at ``mass_theta`` (an (n, n) array). The Laplace mass at a point other than the MAP is H_GN(theta_m) +
    Hess R(theta_m), SPD for the same reasons, and the change of variables is exact for any center and any SPD H."""
    if mass_theta is None:
        return ctx.target.zeta_target(prep.mass)
    return ctx.target.zeta_target(ctx.target.laplace_mass(np.asarray(mass_theta, dtype=np.float64)))


def chain_streams(args):
    """``(k_init, keys)``: the key of the overdispersed starts and one sampler key per chain (``--chain-seed``)."""
    k_init, k_run = jax.random.split(jax.random.key(args.chain_seed))
    return k_init, jax.random.split(k_run, args.chains)


class ChainRunner:
    """Compiles and runs one HMC chain (or one continuation segment) on a ``thetaspace.ZetaTarget``.

    The executable is compiled once per (n_warmup, n_samples) and reused; the key, start, aux and initial step are
    run-time arguments.
    """

    def __init__(self, zt, args):
        self.zt, self.args = zt, args
        self._compiled = {}

    def run(self, key, xi0, aux0, *, n_warmup, n_samples, step0):
        """One chain from ``xi0`` (zeta, (n, n)) with warm start ``aux0``: ``n_warmup`` step-adapting trajectories (none
        for a continuation, which keeps ``step0``), then ``n_samples`` recorded ones. Returns a dict of NumPy arrays:
        ``theta`` (n_samples // thin, n, n), ``mean`` and ``var`` ((2, n, n): theta, eps_r; Welford over all
        n_samples steps), ``count``, ``phi_prime`` and ``accept`` ((n_samples,)), ``step`` (frozen), ``xi``
        (final zeta), ``aux`` (final fields), ``wall_s``, ``compile_s`` and the arguments of the call."""
        zt, a = self.zt, self.args
        sig = (int(n_warmup), int(n_samples))
        step0_arr = jnp.asarray(step0, dtype=jnp.float64)
        t0 = time.perf_counter()
        if sig not in self._compiled:
            def one(k, x, aux, s0):
                return sm.hmc(k, phi_fn=zt.phi_fn, xi0=x, aux0=aux, n_warmup=sig[0], n_samples=sig[1], step0=s0,
                              thin=a.thin, record_fn=zt.record, n_leapfrog=a.n_leapfrog, jitter=a.jitter)

            self._compiled[sig] = jax.jit(one).lower(key, xi0, aux0, step0_arr).compile()
            compile_s = time.perf_counter() - t0
        else:
            compile_s = 0.0
        t0 = time.perf_counter()
        res = jax.block_until_ready(self._compiled[sig](key, xi0, aux0, step0_arr))
        wall_s = time.perf_counter() - t0
        return {"theta": np.asarray(res.samples[:, 0]), "mean": np.asarray(res.mean), "var": np.asarray(res.var),
                "theta_final": np.asarray(zt.to_theta(res.xi)), "mass_theta": np.asarray(zt.center),
                "count": int(n_samples), "phi_prime": np.asarray(res.phi), "accept": np.asarray(res.accept),
                "step": float(res.step), "xi": np.asarray(res.xi), "aux": np.asarray(res.aux), "wall_s": wall_s,
                "compile_s": compile_s, "n_warmup": int(n_warmup), "n_samples": int(n_samples),
                "step0": float(step0), "n_leapfrog": int(a.n_leapfrog), "thin": int(a.thin),
                "load_avg_1min_at_start": os.getloadavg()[0]}


def run_segment(runner, args, chain_id, segment, prev=None, start_theta=None):
    """Segment ``segment`` of chain ``chain_id``; ``prev`` is the dict of segment - 1 (None for segment 0).

    Segment 0 starts at theta_hat + L_H^-T (init_scale N(0, I)) with the chain's key and ``--warmup`` adaptation
    trajectories; segment j > 0 starts at the final state and aux of segment j - 1 with its frozen step and the key
    ``fold_in(chain key, j)``. ``start_theta`` ((chains, n, n) or (n, n)) replaces the overdispersed start of segment 0.
    """
    k_init, keys = chain_streams(args)
    if segment == 0:
        if start_theta is None:
            xi0 = runner.zt.overdispersed_starts(k_init, args.chains, args.init_scale)[chain_id] + getattr(runner, "xi_ref", 0.0)
        else:
            st = np.asarray(start_theta, dtype=np.float64)
            xi0 = runner.zt.to_zeta(st if st.ndim == 2 else st[chain_id])
        return runner.run(keys[chain_id], xi0, runner.zt.initial_aux(), n_warmup=args.warmup, n_samples=args.samples,
                          step0=getattr(runner, "eps0", args.step0))
    if prev is None:
        raise ValueError("a continuation segment needs the previous segment")
    return runner.run(jax.random.fold_in(keys[chain_id], segment), jnp.asarray(prev["xi"]), jnp.asarray(prev["aux"]),
                      n_warmup=0, n_samples=args.samples, step0=float(prev["step"]))


def segment_path(out_dir, chain_id, segment):
    return out_dir / f"chain_{chain_id}_seg{segment}.npz"


def save_segment(path, seg):
    np.savez(path, **{k: np.asarray(v) for k, v in seg.items()})


def load_segment(path):
    """A segment file as a dict (0-d arrays back to Python scalars)."""
    z = np.load(path)
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def load_chain_segments(out_dir, args):
    """``[[segment dicts of chain 0], [chain 1], ...]`` from chain_<c>_seg<j>.npz (j = 0, 1, ... up to
    ``--max-segments``), without the first ``--skip-segments`` of each (burn-in, ``discard_burn_in``). Exits unless every chain has
    the same, nonzero number of segments."""
    per_chain = []
    for c in range(args.chains):
        segs, j = [], 0
        while segment_path(out_dir, c, j).exists() and (args.max_segments is None or j < args.max_segments):
            segs.append(load_segment(segment_path(out_dir, c, j)))
            j += 1
        per_chain.append(segs)
    counts = [len(s) for s in per_chain]
    if len(set(counts)) != 1 or counts[0] == 0:
        raise SystemExit(f"need the same nonzero number of segments for every chain, found {counts} "
                         f"(use --max-segments to drop the extras)")
    if args.skip_segments:
        if counts[0] <= args.skip_segments:
            raise SystemExit(f"--skip-segments {args.skip_segments} leaves no segment (every chain has {counts[0]})")
        per_chain = [discard_burn_in(segs, args.skip_segments) for segs in per_chain]
    return per_chain


def discard_burn_in(segs, k):
    """The segments of ONE chain without its first ``k``, which become burn-in of the first kept one: its ``n_warmup`` is the number of
    trajectories of the dropped segments (warmup and recorded) plus its own, and their wall time, compile time and gradient evaluations
    are added to its own, so that the cost of the run is still the cost of everything computed. The kept segments are exact continuations of
    the dropped ones (same map, same frozen step), so they remain a valid chain; only the draws, the Welford moments and the convergence
    numbers lose the transient. ``starts`` is carried over from segment 0. ``k`` = 0 returns ``segs`` unchanged."""
    if k <= 0:
        return segs
    dropped, kept = segs[:k], [dict(s) for s in segs[k:]]
    first = kept[0]
    first["n_warmup"] = int(first["n_warmup"] + sum(s["n_warmup"] + s["n_samples"] for s in dropped))
    for key in ("wall_s", "compile_s"):
        first[key] = float(first[key] + sum(s[key] for s in dropped))
    first["n_phi_evals"] = int(sum(segment_n_phi_evals(s) for s in segs[:k + 1]))
    first["starts"] = f"{dropped[0].get('starts', '')} (first {k} segment{'s' if k > 1 else ''} discarded as burn-in)".strip()
    return kept


def segment_n_phi_evals(seg):
    """Potential evaluations of one segment dict: the stored count, or (older files) that of fixed-length HMC."""
    return seg["n_phi_evals"] if "n_phi_evals" in seg else inf.n_phi_evals("hmc", seg["n_warmup"], seg["n_samples"], seg["n_leapfrog"])


def assemble_chains(per_chain, zt_legacy=None):
    """Stack the segments of every chain into arrays with a leading chain axis.

    Returns a dict: ``theta`` (chains, N / thin, n, n) float64 draws of all segments in order; ``mean`` / ``var``
    ((chains, 2, n, n): theta, eps_r) Welford moments over all post-warmup steps (``features.merge_moments``);
    ``phi_prime`` and ``accept`` (chains, steps); ``step`` (chains,) the step size adapted in segment 0; ``xi`` and
    ``aux`` the final states, ``theta_final`` the final theta (chains, n, n) and ``mass_theta`` (n, n) the center of the
    chains' coordinates (segment files from before these keys existed are completed with ``zt_legacy``, the MAP
    coordinates, which is what they used); ``segments`` the per-segment (n_warmup, n_samples) of chain 0; and timings.
    """
    def theta_final(segs):
        s = segs[-1]
        return s["theta_final"] if "theta_final" in s else np.asarray(zt_legacy.to_theta(jnp.asarray(s["xi"])))

    masses = [s["mass_theta"] for segs in per_chain for s in segs if "mass_theta" in s]
    if any(not np.array_equal(m, masses[0]) for m in masses):
        raise SystemExit("the segments were run with different mass centers")
    digests = {s.get("adapt_digest") for segs in per_chain for s in segs}
    if len(digests) != 1:
        raise SystemExit(f"the segments were run with different adapted maps (or some without one): {sorted(map(str, digests))}")
    mom = [ft.merge_moments([s["mean"] for s in segs], [s["var"] for s in segs], [s["count"] for s in segs])
           for segs in per_chain]
    return {
        "theta": np.stack([np.concatenate([s["theta"] for s in segs]) for segs in per_chain]),
        "mean": np.stack([m for m, _ in mom]), "var": np.stack([v for _, v in mom]),
        "phi_prime": np.stack([np.concatenate([s["phi_prime"] for s in segs]) for segs in per_chain]),
        "accept": np.stack([np.concatenate([s["accept"] for s in segs]) for segs in per_chain]),
        "step": np.array([segs[0]["step"] for segs in per_chain]),
        "xi": np.stack([segs[-1]["xi"] for segs in per_chain]), "aux": np.stack([segs[-1]["aux"] for segs in per_chain]),
        "theta_final": np.stack([theta_final(segs) for segs in per_chain]),
        "mass_theta": masses[0] if masses else None,
        "segments": [(s["n_warmup"], s["n_samples"]) for s in per_chain[0]],
        "wall_s": np.array([sum(s["wall_s"] for s in segs) for segs in per_chain]),
        "compile_s": np.array([sum(s["compile_s"] for s in segs) for segs in per_chain]),
        "load_avg_1min": np.array([s["load_avg_1min_at_start"] for segs in per_chain for s in segs]),
        "adapt_digest": digests.pop(),
        "n_phi_evals": int(sum(segment_n_phi_evals(s) for s in per_chain[0])),
        "n_leapfrog0": per_chain[0][0]["n_leapfrog"]}


# ----------------------------------------------------------------------------------------------
# Adapted affine map (stage adapt, and the chains that follow it)
# ----------------------------------------------------------------------------------------------


def adaptation_paths(out_dir):
    """``(adapted_map.npz, adaptation.json)`` of a run directory."""
    return out_dir / "adapted_map.npz", out_dir / "adaptation.json"


def adapt_config(args):
    """The ``adapt.AdaptConfig`` of the adaptation stage from the command line."""
    return ad.AdaptConfig(init_buffer=args.adapt_init, first_window=args.adapt_first_window, n_windows=args.adapt_windows,
                          growth=args.adapt_growth, final_buffer=args.adapt_final, block=args.adapt_block,
                          target_accept=args.target_accept, traj_length=args.traj_length, max_leapfrog=args.max_leapfrog,
                          kappa_per_dim=args.adapt_kappa_per_dim, step0=args.adapt_step0)


def laplace_covariance_fn(target):
    """``center (N,) -> H^-1 (N, N)``: the Laplace covariance at a center, H = H_GN + Hess R there (the shrinkage target of
    ``adapt.update_map``; one Nt + Nr solve Jacobian, a Gram product and two dense N^3 steps)."""
    n = target.n
    return lambda c: ad.laplace_covariance(target.laplace_mass(np.asarray(c).reshape(n, n)).chol)


def run_adaptation(ctx, prep, out_dir, mass_theta=None):
    """The adaptation stage: adapts the affine map of one chain (module docstring), writes adapted_map.npz and adaptation.json in
    ``out_dir`` and returns the ``adapt.AdaptResult``. The initial map is the Laplace map of the MAP (``prep.mass``), or of
    ``mass_theta`` ((n, n); the Laplace mass is rebuilt there) when given."""
    args, target, n = ctx.args, ctx.target, ctx.args.n
    mass = prep.mass if mass_theta is None else target.laplace_mass(np.asarray(mass_theta, dtype=np.float64))
    map0 = ad.AffineMap.from_precision_chol(mass.theta_hat, mass.chol)
    key = jax.random.fold_in(jax.random.key(args.chain_seed), 10007)
    k_start, k_run = jax.random.split(key)
    theta0 = np.asarray(map0.to_theta(args.adapt_start_scale * jax.random.normal(k_start, (n, n), dtype=jnp.float64)))
    cfg = adapt_config(args)
    print(f"adaptation: {cfg.total_trajectories()} trajectories ({[(nm, nb * cfg.block) for nm, nb, _ in cfg.phases()]}), "
          f"kappa = {cfg.kappa_for(n * n):g}, trajectory length {cfg.traj_length}, target acceptance {cfg.target_accept}, start scale "
          f"{args.adapt_start_scale}, initial map: Laplace at "
          f"{'the MAP' if mass_theta is None else 'a custom theta'}")

    def progress(rec):
        up = rec.get("update", {})
        print(f"  {rec['phase']:8s} {rec['trajectories']:4d} traj: accept {rec['accept_mean']:.3f}, eps {rec['eps_end']:.4f}, "
              f"mean L {rec['mean_leapfrog']:.1f}, {rec['time_s']:.0f} s" + (
                  f"; map updated (weight {up['weight']:.3f}, variance ratio median {up['var_ratio_median']:.3g}, update "
                  f"{up['time_s']:.1f} s)" if rec["updated"] else ("; " + rec["skipped"] if "skipped" in rec else "")))

    res = ad.adapt_map(ctx.potential, k_run, theta0, target.initial_aux(), map0, laplace_covariance_fn(target), cfg,
                       progress=progress)
    path_map, path_json = adaptation_paths(out_dir)
    res.map.save(path_map, eps=res.eps, traj_length=res.traj_length, max_leapfrog=res.max_leapfrog, theta_final=res.theta)
    update_s = [h["update"]["time_s"] for h in res.history if h.get("updated")]
    info = {"args": {k: v for k, v in vars(args).items() if k.startswith("adapt") or k in ("target_accept", "traj_length", "max_leapfrog",
                                                                                   "chain_seed")},
            "config": dataclasses.asdict(cfg), "kappa": cfg.kappa_for(n * n), "initial_map": "Laplace map of the MAP" if mass_theta is None
            else "Laplace map at --mass-theta", "eps": res.eps, "traj_length": res.traj_length, "max_leapfrog": res.max_leapfrog,
            "n_leapfrog0": res.n_leapfrog0, "map_digest": res.map.digest(), "n_grad_evals": res.n_grad_evals, "wall_s": res.wall_s,
            "update_time_s": update_s, "history": res.history,
            "note": "the map and eps are frozen before any recorded draw: a recorded segment is an exact HMC chain for the fixed map"}
    path_json.write_text(json.dumps(info, indent=2, default=float))
    print(f"adapted map: eps {res.eps:.4f}, L0 {res.n_leapfrog0}, {res.n_grad_evals} gradient evaluations, {res.wall_s:.0f} s; "
          f"digest {info['map_digest'][:16]}; wrote {path_map.name} and {path_json.name}")
    return res


def load_adaptation(out_dir, n):
    """``(AffineMap, info)`` of a run directory: the frozen map of adapted_map.npz with its center reshaped to (n, n), and the dict of
    adaptation.json updated with the stored ``eps``, ``traj_length`` and ``max_leapfrog``. Exits if the stage adapt has not been run
    or the map does not match the digest in adaptation.json."""
    path_map, path_json = adaptation_paths(out_dir)
    if not path_map.exists() or not path_json.exists():
        raise SystemExit(f"{out_dir}: no adapted_map.npz / adaptation.json (run --stage adapt first)")
    z = np.load(path_map)
    amap = ad.AffineMap(z["center"].reshape(n, n), z["chol"])
    info = json.loads(path_json.read_text())
    if amap.digest() != info["map_digest"]:
        raise SystemExit(f"{path_map} does not match the digest in {path_json.name}")
    info.update({"eps": float(z["eps"]), "traj_length": float(z["traj_length"]), "max_leapfrog": int(z["max_leapfrog"]),
                 "theta_final": z["theta_final"].reshape(n, n)})
    return amap, info


class AdaptedChainRunner:
    """The chain runner of ``--adapt``: HMC in the coordinates of a frozen ``adapt.AffineMap`` with variable-length trajectories.

    Same interface as ``ChainRunner`` (``run_segment`` drives either): ``zt`` is the ``ZetaTarget`` of the adapted map and ``run``
    returns the same dict. Segment 0 first runs ``n_warmup`` trajectories (rounded up to a multiple of the block size) of dual
    averaging of the step size ALONE (map frozen) from the adapted step ``eps0``, then ``n_samples`` recorded trajectories
    (``adapt.sample_chain``, step, trajectory length and map frozen); a continuation has ``n_warmup = 0`` and keeps its step.
    """

    def __init__(self, ctx, amap, info, args):
        self.ctx, self.amap, self.info, self.args = ctx, amap, info, args
        self.zt = ctx.target.zeta_target_with_map(amap)
        self.potential = ctx.potential
        self.eps0, self.traj_length, self.max_leapfrog = info["eps"], info["traj_length"], info["max_leapfrog"]
        # the starts of segment 0 are xi_ref + init_scale N(0, I): zeta of the adaptation chain's final state, or 0 (the map center)
        self.xi_ref = amap.to_zeta(jnp.asarray(info["theta_final"])) if args.adapt_start == "final" else jnp.zeros(amap.shape)
        self.digest = amap.digest()
        self._compiled = {}

    def _warmup(self, key, xi0, aux0, n_warmup):
        """Dual averaging of eps with the map frozen: ``(eps, xi, aux, n_grad_evals, n_trajectories)`` (``n_warmup`` rounded up to a
        whole number of blocks)."""
        a = self.args
        cfg = ad.AdaptConfig(init_buffer=n_warmup, n_windows=0, final_buffer=0, block=a.adapt_block, target_accept=a.target_accept,
                             traj_length=self.traj_length, max_leapfrog=self.max_leapfrog, step0=self.eps0)
        theta0 = np.asarray(self.amap.to_theta(jnp.asarray(xi0))).ravel()
        res = ad.adapt_map(self.potential, key, theta0, aux0, self.amap, None, cfg)
        return (res.eps, self.amap.to_zeta(jnp.asarray(res.theta.reshape(self.amap.shape))), res.aux, res.n_grad_evals,
                cfg.total_trajectories())

    def run(self, key, xi0, aux0, *, n_warmup, n_samples, step0):
        zt, a = self.zt, self.args
        k_warm, k_rec = jax.random.split(key)
        n_warm_grad, n_warm = 0, 0
        if n_warmup > 0:
            eps, xi0, aux0, n_warm_grad, n_warm = self._warmup(k_warm, xi0, aux0, n_warmup)
        else:
            eps = float(step0)
        sig = int(n_samples)
        t0 = time.perf_counter()
        if sig not in self._compiled:
            def one(k, x, aux, e):
                return ad.sample_chain(k, zt.phi_fn, x, aux, eps=e, traj_length=self.traj_length, max_leapfrog=self.max_leapfrog,
                                       n_samples=sig, thin=a.thin, record_fn=zt.record)

            self._compiled[sig] = jax.jit(one).lower(k_rec, jnp.asarray(xi0), aux0, jnp.asarray(eps, dtype=jnp.float64)).compile()
            compile_s = time.perf_counter() - t0
        else:
            compile_s = 0.0
        t0 = time.perf_counter()
        res = jax.block_until_ready(self._compiled[sig](k_rec, jnp.asarray(xi0), aux0, jnp.asarray(eps, dtype=jnp.float64)))
        wall_s = time.perf_counter() - t0
        n_rec_grad = int(np.asarray(res["n_leapfrog"]).sum())
        l0 = int(np.clip(round(self.traj_length / eps), 1, self.max_leapfrog))
        return {"theta": np.asarray(res["samples"][:, 0]), "mean": np.asarray(res["mean"]), "var": np.asarray(res["var"]),
                "theta_final": np.asarray(zt.to_theta(res["xi"])), "mass_theta": np.asarray(zt.center),
                "count": int(n_samples), "phi_prime": np.asarray(res["phi"]), "accept": np.asarray(res["accept"]),
                "step": float(eps), "xi": np.asarray(res["xi"]), "aux": np.asarray(res["aux"]), "wall_s": wall_s,
                "compile_s": compile_s, "n_warmup": int(n_warm), "n_samples": sig, "step0": float(self.eps0 if n_warmup > 0 else step0),
                "n_leapfrog": l0, "thin": int(a.thin), "load_avg_1min_at_start": os.getloadavg()[0],
                "n_phi_evals": 1 + n_warm_grad + n_rec_grad, "adapt_digest": self.digest, "traj_length": float(self.traj_length),
                "max_leapfrog": int(self.max_leapfrog), "mean_leapfrog": float(np.asarray(res["n_leapfrog"]).mean())}


# ----------------------------------------------------------------------------------------------
# Summary (stage finalize)
# ----------------------------------------------------------------------------------------------


def summarize(ctx, prep, ch, out_dir, t_start):
    """Everything computed from the stored draws ``ch`` (``assemble_chains``): physical Phi, convergence, posterior
    moments, z statistics, errors, features; writes summary.json, chains.npz, maps.png and traces.png in ``out_dir``
    and returns the summary dict."""
    args, prob, grid, target, prior = ctx.args, ctx.prob, ctx.grid, ctx.target, ctx.prior
    n, n_data, info = args.n, ctx.n_data, prep.info
    theta_hat, lap_std, map_info = prep.theta_hat, prep.mass.theta_std(), info["map"]
    timings = dict(info["timings"])
    accept_c = ch["accept"].mean(axis=1)

    # ---- physical misfit of the stored draws (never the sampler's Phi') ----
    theta_s = ch["theta"]  # (chains, N / thin, n, n)
    t0 = time.perf_counter()
    phi_phys = inf.physical_phi_of_draws(target, target.eps_r(theta_s))
    timings["physical_phi_s"] = time.perf_counter() - t0
    (theta_mean, eps_mean), (theta_std, eps_std) = ft.pooled_moments(ch["mean"], ch["var"])
    conv = convergence(phi_phys, theta_s)
    q1 = max(phi_phys.shape[1] // 4, 1)
    print(f"physical Phi of the draws: mean {phi_phys.mean():.1f} (MAP {map_info['phi']:.1f}, N {n_data}); split R-hat "
          f"{conv['rhat_phi']:.3f}; max pixel R-hat {conv['rhat_pixel_max']:.3f}; worst-pixel ESS "
          f"{conv['ess_pixel_min']:.0f} -> {'CONVERGED' if conv['converged'] else 'UNCONVERGED: ' + '; '.join(conv['unconverged_reasons'])}")

    # ---- summaries against the truth ----
    zstats = ft.z_statistics(theta_mean, theta_std, prob.theta_true, prob.chi_true)
    chi_map = np.asarray(target.chi(theta_hat))
    eps_map = 1.0 + chi_map
    chi_mean = eps_mean - 1.0
    d_map, _ = prob.forward(jnp.asarray(chi_map))
    d_mean, _ = prob.forward(jnp.asarray(chi_mean))
    metrics = {
        "err_model": {"vacuum": inf.err_model(np.ones_like(prob.eps_true), prob.eps_true),
                      "map": inf.err_model(eps_map, prob.eps_true),
                      "posterior_mean": inf.err_model(eps_mean, prob.eps_true)},
        "err_chi": {"vacuum": 1.0, "map": inf.err_chi(chi_map, prob.chi_true),
                    "posterior_mean": inf.err_chi(chi_mean, prob.chi_true)},
        "err_data": {"vacuum": 1.0, "map": inf.err_data(d_map, prob.d_clean),
                     "posterior_mean": inf.err_data(d_mean, prob.d_clean)}}
    feats = ft.feature_probabilities(theta_s, grid)
    feats_truth = {k: bool(v) for k, v in ft.feature_indicators(prob.theta_true[None, None], grid).items()}
    for k in feats:
        feats[k]["truth_satisfies"] = feats_truth[k]
    ratio = theta_std / lap_std
    prior_std_map = prep.prior_std_map
    max_resid = max(float(jnp.max(prob.forward.residual(target.chi(jnp.asarray(ch["theta_final"][c])),
                                                        jnp.asarray(ch["aux"][c])))) for c in range(args.chains))
    mass_shift = 0.0 if ch["mass_theta"] is None else float(np.sqrt(np.mean((ch["mass_theta"] - theta_hat) ** 2)))
    print(f"err_model (eps_r): MAP {metrics['err_model']['map']:.4f}, posterior mean {metrics['err_model']['posterior_mean']:.4f}; "
          f"err_chi: MAP {metrics['err_chi']['map']:.4f}, mean {metrics['err_chi']['posterior_mean']:.4f}; |z|>3 in "
          f"{100 * zstats['frac_abs_z_gt_3']:.1f}% of pixels (edge share {zstats['edge_share']:.2f})")
    print("features: " + ", ".join(f"{k} P={v['p']:.3f}" for k, v in feats.items()))

    n_seg = len(ch["segments"])
    adapted = ch["adapt_digest"] is not None
    summary = {
        "args": vars(args),
        "data": info["data"],
        "operating_point": info["operating_point"],
        "prior": {**prior.describe(), "prior_summary": info["prior_info"]},
        "start": info["start"],
        "map": map_info,
        "laplace": info["laplace"],
        "sampler": {"name": ("hmc (zeta coordinates of the adapted affine map theta = c + A zeta, A A^T = Sigma_hat, frozen before "
                             "recording; L uniform in [L0/2, 3 L0/2])" if adapted
                             else "hmc (zeta coordinates, theta = theta_hat + L_H^-T zeta)"), "chains": args.chains,
                    "warmup": ch["segments"][0][0], "samples": int(sum(s for _, s in ch["segments"])),
                    "segments": n_seg, "samples_per_segment": [int(s) for _, s in ch["segments"]], "thin": args.thin,
                    "n_leapfrog": ch["n_leapfrog0"] if adapted else args.n_leapfrog,  # adapted: L0 of chain 0
                    "jitter": "L uniform in [L0/2, 3 L0/2]" if adapted else args.jitter, "init_scale": args.init_scale,
                    "step0": ch["adaptation"]["eps"] if adapted else args.step0, "chain_seed": args.chain_seed,
                    "mass": (f"adapted affine map {ch['adapt_digest'][:16]} (center rms distance to the MAP {mass_shift:.4f})" if adapted
                             else "Laplace mass at the MAP" if mass_shift == 0.0 else
                             f"Laplace mass at a custom theta (rms distance to the MAP {mass_shift:.4f})"),
                    "mass_theta_rms_from_map": mass_shift, "starts": ch["starts"],
                    "acceptance": {"mean": float(accept_c.mean()), "per_chain": accept_c.tolist()},
                    "final_step": ch["step"].tolist(), "n_phi_evals_per_chain": ch["n_phi_evals"],
                    "wall_s": float(ch["wall_s"].sum()), "wall_s_per_chain": ch["wall_s"].tolist(),
                    "compile_s": float(ch["compile_s"].sum()),
                    "phi_prime_mean": float(ch["phi_prime"].mean()),
                    "phi_prime_note": "the sampler's own potential Phi' = U - |zeta|^2/2 is NOT the misfit",
                    "max_solver_residual_final_states": max_resid, "tol": args.tol},
        "phi_physical": {"mean": float(phi_phys.mean()), "std": float(phi_phys.std()),
                         "per_chain_mean": phi_phys.mean(axis=1).tolist(),
                         "first_quarter_mean": float(phi_phys[:, :q1].mean()),
                         "last_quarter_mean": float(phi_phys[:, -q1:].mean()), "every": args.thin,
                         "n_per_chain": int(phi_phys.shape[1]), "map_phi": map_info["phi"], "n_data": n_data},
        "convergence": conv,
        **({"adaptation": {**{k: ch["adaptation"][k] for k in ("map_digest", "eps", "traj_length", "n_leapfrog0", "max_leapfrog", "kappa",
                                                              "n_grad_evals", "wall_s", "update_time_s", "initial_map", "config",
                                                              "history")},
                           "files": "adapted_map.npz (center, Cholesky factor), adaptation.json (full history)"}} if adapted else {}),
        "posterior": {"theta_mean": map_stats(theta_mean, ctx.inside), "theta_std": map_stats(theta_std, ctx.inside),
                      "eps_mean": map_stats(eps_mean, ctx.inside), "eps_std": map_stats(eps_std, ctx.inside),
                      "mcmc_over_laplace_std": map_stats(ratio, ctx.inside),
                      "theta_std_over_prior_std": None if prior_std_map is None
                      else map_stats(theta_std / prior_std_map, ctx.inside),
                      "theta_mean_minus_map": float((theta_mean - np.asarray(theta_hat)).mean())},
        "z": zstats,
        "metrics": metrics,
        "features": {"delta": ft.DELTA, "probabilities": feats},
        "timings": {**timings, "mass_s": prep.mass.info["time_total_s"], "map_s": map_info["time_s"],
                    "sampling_s": float(ch["wall_s"].sum()), "sampling_s_max_chain": float(ch["wall_s"].max()),
                    "compile_s": float(ch["compile_s"].sum()), "finalize_s": time.perf_counter() - t_start},
        "machine": {"cpu_count": os.cpu_count(), "load_avg_1min_at_start": float(ch["load_avg_1min"].min()),
                    "load_avg_1min_at_chain_starts_max": float(ch["load_avg_1min"].max()),
                    "load_avg_1min_at_chain_starts_mean": float(ch["load_avg_1min"].mean()),
                    "load_avg_1min_at_end": os.getloadavg()[0], "nproc_env": os.environ.get("NPROC")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(
        out_dir / "chains.npz", theta=theta_s.astype(np.float32), welford_mean=ch["mean"], welford_var=ch["var"],
        phi_physical=phi_phys, phi_prime=ch["phi_prime"], accept=ch["accept"], step=ch["step"],
        theta_hat=np.asarray(theta_hat), laplace_theta_std=lap_std,
        prior_theta_std=np.full((n, n), np.nan) if prior_std_map is None else prior_std_map,
        theta_true=prob.theta_true, chi_true=prob.chi_true, eps_true=prob.eps_true, d_obs=np.asarray(prob.d_obs),
        d_clean=np.asarray(prob.d_clean), noise_std=prob.s)

    # ---- figures ----
    title = f"{args.prior} " + ", ".join(f"{k}={v}" for k, v in prior.describe().items()
                                         if k in ("sigma", "ell", "q", "T", "sigma_g", "sigma_0", "lam", "delta")) \
        + f", austria eps_r {args.eps_r:g}, n={n}, SNR {args.snr:g} dB"
    plot_maps(prob.theta_true, np.asarray(theta_hat), theta_mean, theta_std, grid, out_dir / "maps.png",
              quantity="log eps_r", title=title)
    extra = {f"theta pixel ({i},{j})": theta_s[:, :, i, j] for (i, j) in [(n // 2, n // 2), (n // 4, n // 4),
                                                                             (3 * n // 4, n // 4)]}
    plot_traces(ch["phi_prime"], out_dir / "traces.png", extra={"Phi (data misfit, stored draws)": phi_phys, **extra},
                title=title, phi_label="Phi' (sampler potential, not the misfit)")
    return summary


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out) if Path(args.out).is_absolute() else RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    ctx = build_context(args)
    setup_s = time.perf_counter() - t0
    if args.stage in ("all", "prepare"):
        prep = prepare(ctx)
        prep.info["timings"]["setup_s"] = setup_s
        if args.stage == "prepare":
            save_prepared(prep, out_dir)
            print(f"wrote {out_dir}/prepared.json and prepared.npz (total {time.perf_counter() - t_start:.0f} s)")
            return prep.info
    else:
        prep = load_prepared(ctx, out_dir)
    mass_theta = None if args.mass_theta is None else np.load(args.mass_theta)
    start_theta = None if args.start_theta is None else np.load(args.start_theta)
    start_text = "start-theta file" if start_theta is not None else (
        f"overdispersed draws (scale {args.init_scale}) around the {'final state of the adaptation chain' if args.adapt_start == 'final' else 'center'}"
        f" of the adapted map" if args.adapt else f"overdispersed Laplace draws, scale {args.init_scale}")

    if args.stage == "adapt":
        return run_adaptation(ctx, prep, out_dir, mass_theta)

    if args.stage == "chain":
        prev = None if args.segment == 0 else load_segment(segment_path(out_dir, args.chain_id, args.segment - 1))
        if args.adapt:
            amap, ainfo = load_adaptation(out_dir, args.n)
            if prev is not None and prev.get("adapt_digest") != amap.digest():
                raise SystemExit("the previous segment was run with a different map than adapted_map.npz")
            runner = AdaptedChainRunner(ctx, amap, ainfo, args)
            print(f"chain {args.chain_id} segment {args.segment}: HMC in zeta of the adapted map {amap.digest()[:16]}, "
                  f"{args.warmup if args.segment == 0 else 0} step-size warmup + {args.samples} samples, thin {args.thin}, trajectory "
                  f"length {runner.traj_length}, step {runner.eps0 if prev is None else prev['step']:.4f}, init scale {args.init_scale}")
        else:
            if prev is not None:  # a continuation follows the coordinates of its predecessor
                prev_mass = prev.get("mass_theta")
                if mass_theta is not None and (prev_mass is None or not np.array_equal(mass_theta, prev_mass)):
                    raise SystemExit("--mass-theta differs from the center the chain was started with")
                mass_theta = None if prev_mass is None or np.array_equal(prev_mass, prep.theta_hat) else prev_mass
            runner = ChainRunner(make_zeta_target(ctx, prep, mass_theta), args)
            print(f"chain {args.chain_id} segment {args.segment}: HMC in zeta, "
                  f"{args.warmup if args.segment == 0 else 0} warmup + {args.samples} samples, thin {args.thin}, "
                  f"{args.n_leapfrog} leapfrog steps, step0 {args.step0 if prev is None else prev['step']:.4f}, init scale "
                  f"{args.init_scale}")
        seg = run_segment(runner, args, args.chain_id, args.segment, prev, start_theta)
        seg["starts"] = start_text if args.segment == 0 else ""
        save_segment(segment_path(out_dir, args.chain_id, args.segment), seg)
        print(f"done in {seg['wall_s']:.0f} s (compile {seg['compile_s']:.0f} s): acceptance {seg['accept'].mean():.3f}, "
              f"step {seg['step']:.4f}, |zeta|^2/2 at the end {0.5 * float(np.sum(seg['xi'] ** 2)):.0f}, "
              f"mean Phi' {seg['phi_prime'].mean():.1f}")
        return seg

    if args.stage == "all":
        if args.adapt:
            run_adaptation(ctx, prep, out_dir, mass_theta)
            amap, ainfo = load_adaptation(out_dir, args.n)
            runner = AdaptedChainRunner(ctx, amap, ainfo, args)
            print(f"sampling: HMC in zeta of the adapted map, {args.chains} chains x ({args.warmup} step-size warmup + "
                  f"{args.samples} samples), thin {args.thin}, trajectory length {runner.traj_length}, init scale {args.init_scale}")
        else:
            runner = ChainRunner(make_zeta_target(ctx, prep, mass_theta), args)
            print(f"sampling: HMC in zeta, {args.chains} chains x ({args.warmup} warmup + {args.samples} samples), thin "
                  f"{args.thin}, {args.n_leapfrog} leapfrog steps, step0 {args.step0}, init scale {args.init_scale}")
        per_chain = [[run_segment(runner, args, c, 0, start_theta=start_theta)] for c in range(args.chains)]
        for segs in per_chain:
            segs[0]["starts"] = start_text
        print(f"sampling done in {sum(s[0]['wall_s'] for s in per_chain):.0f} s: acceptance "
              f"{[round(float(s[0]['accept'].mean()), 3) for s in per_chain]}, final step "
              f"{[round(s[0]['step'], 3) for s in per_chain]}")
    else:  # finalize
        per_chain = load_chain_segments(out_dir, args)
    ch = assemble_chains(per_chain, zt_legacy=None if all("theta_final" in s for segs in per_chain for s in segs)
                         else ctx.target.zeta_target(prep.mass))
    ch["starts"] = per_chain[0][0].get("starts", "overdispersed Laplace draws, scale %g" % args.init_scale)
    if ch["adapt_digest"] is not None:  # an adapted run: the map the segments used must be the one on disk
        amap, ainfo = load_adaptation(out_dir, args.n)
        if amap.digest() != ch["adapt_digest"]:
            raise SystemExit("the segments were run with a map other than adapted_map.npz")
        ch["adaptation"] = ainfo
    summary = summarize(ctx, prep, ch, out_dir, t_start)
    summary["timings"]["total_s"] = time.perf_counter() - t_start
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out_dir} (total {summary['timings']['total_s']:.0f} s)")
    return summary


if __name__ == "__main__":
    main()
