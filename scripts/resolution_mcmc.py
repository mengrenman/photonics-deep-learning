"""MCMC spot checks of the Laplace gap-valley probabilities (DESIGN 13.3): Laplace-mass HMC, 4 chains, at selected configurations.

The Laplace posterior N(MAP, (H_GN + Hess R)^-1) of ``resolution_laplace.py`` is exact only for a Gaussian posterior. For a spot check the same
problem (same data, same MAP) is sampled by HMC in the Laplace-whitened coordinates of milestone 4 (theta = theta_hat + L_H^-T zeta, exact for
the target exp(-U(theta)); ``thetaspace.ZetaTarget``, ``run_prior_comparison.ChainRunner``: fixed-length leapfrog with a jittered step size, Robbins-Monro
step adaptation toward acceptance 0.7 in the warmup, then the step is frozen). The gap-valley probability, the means and standard deviations
of the strip and core averages, and the pixel standard deviations are then compared with the Laplace values.

Stages (--stage):

  chain     ONE chain (--chain-id) of the configuration (--g, --eps-r, --prior): segment 0 is --warmup adaptation trajectories plus --samples recorded ones
            from an overdispersed Laplace draw (--init-scale; default 2 for the Matern prior as in DESIGN 11.2, 0.5 for TV as in the milestone-5 runs),
            segment j > 0 --samples more from the final state and the frozen step of segment j - 1.
            ``--start truth`` replaces the overdispersed start of segment 0 by the pixelized truth (``problem.theta_true``): the fifth chain of the
            DESIGN 13.3 spot checks (``--chains 5 --chain-id 4``; chains 0 to 3 are the overdispersed ones; ``finalize --chains 5 --tag _truth5``
            analyzes the five together and writes summary_truth5.json next to the four-chain summary.json). Segments --first .. --last are computed in ONE
            process, each written to results/resolution/mcmc/<name>/chain_<c>_seg<j>.npz as soon as it is done (a rerun skips the segments that
            exist and continues from the last one), so a killed job loses at most one segment. The MAP is read from results/resolution/laplace/<name>.npz
            (``resolution_laplace.py --stage config`` must have run).
  finalize  all chains -> <dir>/summary<tag>.json (physical Phi of the draws, the convergence numbers of DESIGN 11.2, the gap-valley probability by MCMC next to the
            Laplace one with its Monte Carlo standard error, the strip / core averages, the MCMC / Laplace standard deviation ratios) and series.npz.
  refeature the gap-valley probability from the stored draws of every configuration with a summary, for the other strip heights and core radii of
            ``resolution_laplace.py --stage refeature`` (--band-fracs, --core-fracs), next to the Laplace value of laplace_sensitivity.json:
            results/resolution/mcmc_refeature<tag>.json. The variants with a strip of half height r/2 or r/4 are the ones that tell a gap from no gap
            (null.json), and their probabilities are intermediate, where a Gaussian approximation is least reliable.
  energy    add the Phi' criteria (``energy_convergence``) to every summary*.json that lacks them, from the stored segments (no forward solves): the
            chains, burn-in and tail of each summary are read from it, and the file is rewritten with ``convergence["converged"]`` now including them
            (``converged_classic`` keeps the old flag). A summary written by ``finalize`` has them already.
  collect   every summary -> results/resolution/mcmc.json and mcmc_spotcheck.png.

The convergence flag is the classic one of DESIGN 11.2 (split R-hat of the physical Phi <= 1.01, max pixel R-hat <= 1.05, worst-pixel Sokal ESS >= 200) PLUS the
two criteria on the sampler's own potential Phi' = U - |zeta|^2 / 2 that the milestone-6 review added (``energy_convergence``: split R-hat of Phi' <= 1.05 and
the Sokal ESS of Phi' summed over the chains >= 200): the physical Phi is a smooth, weakly informative functional of the pixels, whereas Phi' is the direction
along which HMC mixes slowest. ``convergence["converged_classic"]`` is the old flag. Chains that miss the criteria are reported as unconverged, and the
functionals of the feature (strip and core averages, the indicator) carry their own R-hat and ESS.

Run options added by the review follow-up: ``--run-tag`` (a different run directory for the same configuration, e.g. another overdispersion or length),
``--frozen-step`` (the chain of segment 0 runs with this step and no adaptation: the truth-start chain reuses the step of the overdispersed chains instead of
adapting it during its own transient, when it accepted 0.88 to 0.93 against the 0.7 target) and ``--tail-segments`` (finalize/refeature on the last K segments
of every chain, so that a chain with a long relaxation is compared with the others after it).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import diagnostics as dg
from bayes_scatter import gapvalley as gv
from bayes_scatter import inference as inf
from bayes_scatter import thetaspace as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import resolution_laplace as rl  # noqa: E402
import run_prior_comparison as rpc  # noqa: E402

RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)
INIT_SCALE = {"matern": 2.0, "tv": 0.5}
N_LEAPFROG = {"matern": 10, "tv": 20}
RHAT_PHI_PRIME_MAX, ESS_PHI_PRIME_MIN = 1.05, 200.0  # criteria on the sampler's potential Phi' (the energy-like direction), see ``energy_convergence``


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=["chain", "finalize", "collect", "refeature", "energy"], required=True)
    ap.add_argument("--g", type=float)
    ap.add_argument("--eps-r", type=float)
    ap.add_argument("--prior", choices=rl.PRIORS)
    ap.add_argument("--lam", type=float, default=None, help="TV weight (default lambda*(eps_r) of laplace_lambda.json)")
    ap.add_argument("--lam-scale", type=float, default=1.0)
    ap.add_argument("--chain-id", type=int, default=0)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="first segment")
    ap.add_argument("--last", type=int, default=3, help="last segment (inclusive)")
    ap.add_argument("--samples", type=int, default=100, help="recorded trajectories per segment")
    ap.add_argument("--warmup", type=int, default=100, help="step-size adaptation trajectories of segment 0")
    ap.add_argument("--n-leapfrog", type=int, default=None, help="leapfrog steps per trajectory (default 10 Matern, 20 TV)")
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--step0", type=float, default=0.3)
    ap.add_argument("--init-scale", type=float, default=None)
    ap.add_argument("--chain-seed", type=int, default=1)
    ap.add_argument("--start", choices=["overdispersed", "truth"], default="overdispersed",
                    help="chain: start of segment 0, an overdispersed Laplace draw or the pixelized truth")
    ap.add_argument("--run-tag", default="", help="suffix of the run directory (mcmc/<config><run-tag>/): another run of the same configuration; the Laplace MAP is that of <config>")
    ap.add_argument("--frozen-step", type=float, default=None,
                    help="chain: run segment 0 with this step size and no adaptation (--warmup is ignored): the step of the overdispersed chains for the truth-start chain")
    ap.add_argument("--runs", nargs="+", default=None, help="refeature/energy: only these run directories of <out-dir>/mcmc/ (default all with a summary.json)")
    ap.add_argument("--tail-segments", type=int, default=None,
                    help="finalize/refeature: use the last K segments of every chain (chains may have different numbers of segments), the earlier ones are burn-in")
    ap.add_argument("--band-fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25], help="refeature: strip half-height as a fraction of the disk radius")
    ap.add_argument("--core-fracs", type=float, nargs="+", default=[0.5, 0.25], help="refeature: core radius as a fraction of the disk radius")
    ap.add_argument("--tag", default="", help="finalize: suffix of summary<tag>.json and series<tag>.npz (the four-chain analysis has none)")
    ap.add_argument("--max-segments", type=int, default=None, help="finalize: use only the first K segments of every chain")
    ap.add_argument("--skip-segments", type=int, default=0, help="finalize: discard the first K segments of every chain as burn-in (the warmup count of the first kept one grows by them)")
    ap.add_argument("--stop", default=None, help="chain: exit 0 before starting a segment if this file exists")
    ap.add_argument("--out-dir", default=str(RESULTS))
    for name, default in (("n", 64), ("data-factor", 2), ("side", 1.5), ("radius", 0.2), ("snr", 30.0), ("seed", 0), ("n-tx", 40), ("n-rx", 60), ("tol", 1e-10),
                          ("sigma", 0.5), ("ell", 0.4), ("nu", 1.5), ("delta-tv", 0.01), ("sigma-0", 1.0), ("core-frac", gv.CORE_FRAC)):
        ap.add_argument(f"--{name}", type=type(default), default=default)
    args = ap.parse_args(argv)
    if args.stage in ("chain", "finalize") and (args.g is None or args.eps_r is None or args.prior is None):
        ap.error("--stage chain and finalize need --g, --eps-r and --prior")
    return args


def run_dir(args):
    return Path(args.out_dir) / "mcmc" / (rl.config_name(args.g, args.eps_r, args.prior, args.lam_scale) + args.run_tag)


def build_context(args):
    """Problem, prior, target and the Laplace mass at the stored MAP of the configuration; ``args`` as for ``resolution_laplace``."""
    problem = rl.build_problem(args, args.g, args.eps_r)
    lam = None
    if args.prior == "tv":
        lam = args.lam if args.lam is not None else rl.lambda_star_of(args.out_dir, args.eps_r) * args.lam_scale
    prior = rl.build_prior(args, args.prior, problem.grid, lam)
    target = ts.ThetaTarget(problem.forward, prior, problem.d_obs, problem.s)
    name = rl.config_name(args.g, args.eps_r, args.prior, args.lam_scale)
    npz = rl.config_paths(args.out_dir, name)[1]
    if not npz.exists():
        raise SystemExit(f"{npz} does not exist: run resolution_laplace.py --stage config for this configuration first")
    theta_hat = np.load(npz)["theta_hat"]
    return SimpleNamespace(problem=problem, prior=prior, target=target, lam=lam, theta_hat=theta_hat, name=name)


def sampler_args(args):
    """The namespace ``run_prior_comparison.ChainRunner`` / ``run_segment`` read."""
    frozen = args.frozen_step is not None
    return SimpleNamespace(chains=args.chains, thin=1, n_leapfrog=args.n_leapfrog or N_LEAPFROG[args.prior], jitter=args.jitter,
                           step0=args.frozen_step if frozen else args.step0,
                           init_scale=args.init_scale if args.init_scale is not None else INIT_SCALE[args.prior], chain_seed=args.chain_seed,
                           warmup=0 if frozen else args.warmup, samples=args.samples, max_segments=args.max_segments, skip_segments=args.skip_segments,
                           frozen_step=args.frozen_step)


def load_segments(out, args):
    """``[[segment dicts of chain 0], ...]`` of the run directory ``out`` for ``args.chains`` chains.

    By default as ``run_prior_comparison.load_chain_segments`` (every chain must have the same number of segments; ``--max-segments`` and ``--skip-segments``).
    With ``--tail-segments K`` each chain keeps its LAST K segments, the earlier ones becoming the burn-in of the first kept one
    (``run_prior_comparison.discard_burn_in``); the chains may have different numbers of segments, at least K each.
    """
    if args.tail_segments is None:
        return rpc.load_chain_segments(out, SimpleNamespace(chains=args.chains, max_segments=args.max_segments, skip_segments=args.skip_segments))
    per_chain = []
    for c in range(args.chains):
        segs, j = [], 0
        while rpc.segment_path(out, c, j).exists() and (args.max_segments is None or j < args.max_segments):
            segs.append(rpc.load_segment(rpc.segment_path(out, c, j)))
            j += 1
        per_chain.append(segs)
    counts = [len(x) for x in per_chain]
    if min(counts) < args.tail_segments or args.tail_segments < 1:
        raise SystemExit(f"--tail-segments {args.tail_segments} needs at least that many segments in every chain, found {counts}")
    return [rpc.discard_burn_in(segs, len(segs) - args.tail_segments) for segs in per_chain]


def energy_convergence(conv, phi_prime):
    """The convergence dict ``conv`` (``run_prior_comparison.convergence``) extended by the criteria on the sampler's potential Phi' = U - |zeta|^2 / 2.

    ``phi_prime`` (chains, draws) is the recorded Phi' of the draws. Added: ``rhat_phi_prime`` (split R-hat), ``ess_phi_prime`` (Sokal ESS summed over the
    chains) with ``ess_phi_prime_per_chain`` and ``ess_phi_prime_geyer_per_chain`` (the conservative Geyer estimate), and the thresholds. The milestone-6 review
    found a run that met the classic criteria (physical Phi, pixels) with split R-hat 1.073 and per-chain ESS 7 to 26 of Phi': the energy-like direction mixes
    slowest under HMC, and a chain that has not equilibrated in it has not sampled the posterior. ``converged`` now requires R-hat(Phi') <= ``RHAT_PHI_PRIME_MAX``
    and ESS(Phi') >= ``ESS_PHI_PRIME_MIN`` as well; ``converged_classic`` keeps the old flag, ``unconverged_reasons`` lists every miss.
    """
    pp = np.asarray(phi_prime, dtype=np.float64)
    chains, n_draws = pp.shape
    ok = chains >= 2 and n_draws >= 8
    rhat = float(dg.split_rhat(pp)) if ok else float("nan")
    ess_c = [float(dg.ess(c)) for c in pp]
    ess_g = [float(dg.ess_geyer(c)) for c in pp]
    out = dict(conv)
    # a stuck chain (zero variance, ESS NaN) contributes nothing to the sum
    out.update({"rhat_phi_prime": rhat, "ess_phi_prime": float(np.nansum(ess_c)), "ess_phi_prime_per_chain": ess_c, "ess_phi_prime_geyer_per_chain": ess_g,
                "thresholds_phi_prime": {"rhat_phi_prime": RHAT_PHI_PRIME_MAX, "ess_phi_prime_min": ESS_PHI_PRIME_MIN},
                "converged_classic": bool(conv["converged"])})
    reasons = list(conv["unconverged_reasons"])
    if not rhat <= RHAT_PHI_PRIME_MAX:
        reasons.append(f"split R-hat of Phi' {rhat:.3f} > {RHAT_PHI_PRIME_MAX}")
    if not out["ess_phi_prime"] >= ESS_PHI_PRIME_MIN:
        reasons.append(f"ESS of Phi' {out['ess_phi_prime']:.0f} < {ESS_PHI_PRIME_MIN:.0f}")
    out["unconverged_reasons"] = reasons
    out["converged"] = not reasons
    return out


# ----------------------------------------------------------------------------------------------
# Chains
# ----------------------------------------------------------------------------------------------


def stage_chain(args):
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    sa = sampler_args(args)
    todo = [j for j in range(args.first, args.last + 1) if not rpc.segment_path(out, args.chain_id, j).exists()]
    if not todo:
        print(f"{out.name} chain {args.chain_id}: segments {args.first}..{args.last} exist")
        return
    stop = None if args.stop is None else Path(args.stop)
    ctx = build_context(args)
    mass = ctx.target.laplace_mass(ctx.theta_hat)
    zt = ctx.target.zeta_target(mass)
    runner = rpc.ChainRunner(zt, sa)
    how = ("the pixelized truth" if args.start == "truth" else f"init scale {sa.init_scale}") + (f", frozen step {sa.frozen_step:.4f} (no warmup)" if sa.frozen_step is not None else f", warmup {sa.warmup}")
    print(f"{out.name} chain {args.chain_id}: Laplace mass {mass.info['time_total_s']:.1f} s; segments {todo} of {args.samples} trajectories "
          f"({sa.n_leapfrog} leapfrog steps, start: {how})")
    for j in todo:
        if stop is not None and stop.exists():
            print("stop file exists, ending before segment", j)
            return
        prev = None
        if j > 0:
            p = rpc.segment_path(out, args.chain_id, j - 1)
            if not p.exists():
                raise SystemExit(f"segment {j} needs segment {j - 1} ({p})")
            prev = rpc.load_segment(p)
        t0 = time.perf_counter()
        seg = rpc.run_segment(runner, sa, args.chain_id, j, prev=prev, start_theta=ctx.problem.theta_true if args.start == "truth" else None)
        seg["starts"] = (("the pixelized truth" if args.start == "truth" else f"theta_hat + L_H^-T ({sa.init_scale} N(0, I))") if j == 0
                         else f"final state of segment {j - 1}")
        if j == 0 and sa.frozen_step is not None:
            seg["starts"] += f", frozen step {sa.frozen_step:.6g}"
        rpc.save_segment(rpc.segment_path(out, args.chain_id, j), seg)
        print(f"  segment {j}: {time.perf_counter() - t0:.0f} s (compile {seg['compile_s']:.0f} s), step {seg['step']:.4f}, acceptance {float(np.mean(seg['accept'])):.3f}, "
              f"load {seg['load_avg_1min_at_start']:.1f}")


# ----------------------------------------------------------------------------------------------
# Finalize
# ----------------------------------------------------------------------------------------------


def functional_series(theta, w):
    """w . theta for every draw: theta (chains, draws, n, n), w (N,) -> (chains, draws)."""
    return theta.reshape(theta.shape[:2] + (-1,)) @ w


def series_stats(x):
    """Pooled mean and sd of a (chains, draws) series with its split R-hat, summed Sokal ESS and the Monte Carlo standard error of the mean."""
    ess = float(sum(dg.ess(c) for c in x))
    sd = float(x.std(ddof=1))
    return {"mean": float(x.mean()), "sd": sd, "rhat": float(dg.split_rhat(x)) if x.shape[0] >= 2 and x.shape[1] >= 8 else float("nan"),
            "ess": ess, "mcse_mean": sd / np.sqrt(max(ess, 1.0)), "per_chain_mean": [float(m) for m in x.mean(axis=1)]}


def stage_finalize(args):
    out = run_dir(args)
    t_start = time.perf_counter()
    ctx = build_context(args)
    sa = sampler_args(args)
    per_chain = load_segments(out, args)
    ch = rpc.assemble_chains(per_chain)
    theta = ch["theta"]
    chains, n_draws = theta.shape[:2]
    problem, target = ctx.problem, ctx.target
    phi = inf.physical_phi_of_draws(target, target.eps_r(theta))
    conv = energy_convergence(rpc.convergence(phi, theta), ch["phi_prime"])
    mass = target.laplace_mass(ctx.theta_hat)
    lap_std = mass.theta_std()
    fg = gv.feature_geometry(problem.grid, problem.obj, args.core_frac)
    solve = rl.chol_solver(mass.chol)
    lap = gv.gaussian_valley(fg, ctx.theta_hat, solve)
    w = fg.weights
    ser = {k: functional_series(theta, w[k]) for k in ("strip", "core1", "core2", "mid", "c1", "c2")}
    depth_region = 0.5 * (ser["core1"] + ser["core2"]) - ser["strip"]
    ind_region = (ser["strip"] < np.minimum(ser["core1"], ser["core2"]) - fg.delta)
    ind_point = (ser["mid"] < np.minimum(ser["c1"], ser["c2"]) - fg.delta)

    def prob_block(ind, lap_block):
        p = float(ind.mean())
        ess = float(sum(dg.ess(c.astype(float)) if c.std() > 0 else len(c) for c in ind))
        return {"p_valley_mcmc": p, "per_chain": [float(x) for x in ind.mean(axis=1)], "n_draws": int(ind.size), "ess_indicator": ess,
                "mcse": float(np.sqrt(max(p * (1 - p), 0.0) / max(ess, 1.0))), "p_valley_laplace": None if lap_block is None else lap_block["p_valley"]}

    starts = [str(np.load(rpc.segment_path(out, c, 0))["starts"]) for c in range(chains)]
    res = {"name": out.name, "config": ctx.name, "run_tag": args.run_tag, "tail_segments": args.tail_segments, "g": args.g, "eps_r": args.eps_r, "prior": args.prior, "lam": ctx.lam, "chains": chains, "draws_per_chain": int(n_draws),
           "tag": args.tag, "skip_segments": args.skip_segments, "chain_starts": starts, "segments": ch["segments"], "geometry": fg.describe(), "delta": fg.delta,
           "sampler": {"name": "hmc in Laplace-whitened zeta coordinates", "n_leapfrog": sa.n_leapfrog, "jitter": sa.jitter, "step0": sa.step0, "init_scale": sa.init_scale,
                       "chain_seed": sa.chain_seed, "final_step": ch["step"].tolist(), "acceptance": float(ch["accept"].mean()),
                       "acceptance_per_chain": ch["accept"].mean(axis=1).tolist(), "n_phi_evals_per_chain": ch["n_phi_evals"],
                       "wall_s_per_chain": ch["wall_s"].tolist(), "compile_s": float(ch["compile_s"].sum()), "mass_theta_is_map": bool(
                           ch["mass_theta"] is None or float(np.abs(ch["mass_theta"] - ctx.theta_hat).max()) < 1e-12)},
           "phi_physical": {"mean": float(phi.mean()), "std": float(phi.std()), "per_chain_mean": phi.mean(axis=1).tolist(), "map_phi": float(target.energies(ctx.theta_hat)["phi"]),
                            "n_data": problem.n_data, "note": "physical misfit of the stored draws, not the sampler's Phi'"},
           "convergence": conv,
           "valley_region": None if not fg.representable else prob_block(ind_region, lap["region"]),
           "valley_point": prob_block(ind_point, lap["point"]),
           "region_status": "ok" if fg.representable else "unrepresentable on this grid",
           "functionals": {k: series_stats(v) for k, v in ser.items()} | {"depth_region": series_stats(depth_region)},
           "laplace": {"strip": None if lap["region"] is None else {k: lap["region"][k] for k in ("strip_mean", "strip_sd", "core1_mean", "core1_sd", "core2_mean", "core2_sd",
                                                                                                "depth_mean", "depth_sd")},
                       "point": {k: lap["point"][k] for k in ("strip_mean", "strip_sd", "core1_mean", "core1_sd", "core2_mean", "core2_sd", "depth_mean", "depth_sd")}},
           "pixel_std": {"mcmc_over_laplace_median": float(np.median(theta.reshape(-1, *theta.shape[2:]).std(axis=0, ddof=1) / lap_std)),
                         "mcmc_over_laplace_strip_mean": float((theta.reshape(-1, *theta.shape[2:]).std(axis=0, ddof=1) / lap_std)[fg.strip].mean()) if fg.n_strip else None,
                         "mcmc_mean_minus_map_rms": float(np.sqrt(np.mean((theta.reshape(-1, *theta.shape[2:]).mean(axis=0) - ctx.theta_hat) ** 2))),
                         "laplace_std_mean": float(lap_std.mean())},
           "timings_s": {"finalize": time.perf_counter() - t_start, "sampling_total": float(ch["wall_s"].sum())},
           "thread_env": rl.thread_env()}
    rl.write_json(out / f"summary{args.tag}.json", res)
    np.savez_compressed(out / f"series{args.tag}.npz", phi=phi, phi_prime=ch["phi_prime"], depth_region=depth_region, strip=ser["strip"], core1=ser["core1"], core2=ser["core2"], mid=ser["mid"], c1=ser["c1"], c2=ser["c2"],
                        theta_mean=theta.reshape(-1, *theta.shape[2:]).mean(axis=0), theta_std=theta.reshape(-1, *theta.shape[2:]).std(axis=0, ddof=1),
                        theta_hat=ctx.theta_hat, laplace_std=lap_std)
    vr, vp = res["valley_region"], res["valley_point"]
    print(f"{ctx.name}: {chains} x {n_draws} draws; physical Phi {phi.mean():.1f} (MAP {res['phi_physical']['map_phi']:.1f}); split R-hat Phi {conv['rhat_phi']:.3f}, max pixel R-hat "
          f"{conv['rhat_pixel_max']:.3f}, worst ESS {conv['ess_pixel_min']:.0f}; Phi': R-hat {conv['rhat_phi_prime']:.3f}, ESS {conv['ess_phi_prime']:.0f} "
          f"-> {'CONVERGED' if conv['converged'] else 'UNCONVERGED'} (classic criteria alone: {'met' if conv['converged_classic'] else 'missed'})")
    if vr:
        print(f"  P(valley) region: MCMC {vr['p_valley_mcmc']:.4f} +- {vr['mcse']:.4f} (per chain {np.round(vr['per_chain'], 3).tolist()}), Laplace {vr['p_valley_laplace']:.4f}")
    print(f"  P(valley) point: MCMC {vp['p_valley_mcmc']:.4f} +- {vp['mcse']:.4f}, Laplace {vp['p_valley_laplace']:.4f}")
    return res


# ----------------------------------------------------------------------------------------------
# Collect
# ----------------------------------------------------------------------------------------------


def stage_collect(args):
    out_dir = Path(args.out_dir)
    sums = {}
    for p in sorted((out_dir / "mcmc").glob("*/summary.json")):
        r = json.loads(p.read_text())
        sums[r["name"]] = r
    if not sums:
        raise SystemExit(f"no summaries in {out_dir / 'mcmc'}")
    rows = []
    for r in sums.values():
        vr, vp = r["valley_region"], r["valley_point"]
        rows.append({"name": r["name"], "g": r["g"], "eps_r": r["eps_r"], "prior": r["prior"], "lam": r["lam"], "chains": r["chains"], "draws_per_chain": r["draws_per_chain"],
                     "converged": r["convergence"]["converged"], "converged_classic": r["convergence"].get("converged_classic", r["convergence"]["converged"]),
                     "unconverged_reasons": r["convergence"]["unconverged_reasons"],
                     "rhat_phi": r["convergence"]["rhat_phi"], "rhat_pixel_max": r["convergence"]["rhat_pixel_max"], "ess_pixel_min": r["convergence"]["ess_pixel_min"],
                     "rhat_phi_prime": r["convergence"].get("rhat_phi_prime"), "ess_phi_prime": r["convergence"].get("ess_phi_prime"),
                     "ess_phi_prime_per_chain": r["convergence"].get("ess_phi_prime_per_chain"),
                     "rhat_depth": r["functionals"]["depth_region"]["rhat"], "ess_depth": r["functionals"]["depth_region"]["ess"],
                     "config": r.get("config", r["name"]), "run_tag": r.get("run_tag", ""), "init_scale": r["sampler"]["init_scale"], "final_step": r["sampler"]["final_step"],
                     "region_status": r["region_status"],
                     "p_region_mcmc": None if vr is None else vr["p_valley_mcmc"], "p_region_mcse": None if vr is None else vr["mcse"],
                     "p_region_laplace": None if vr is None else vr["p_valley_laplace"], "p_region_per_chain": None if vr is None else vr["per_chain"],
                     "p_point_mcmc": vp["p_valley_mcmc"], "p_point_mcse": vp["mcse"], "p_point_laplace": vp["p_valley_laplace"],
                     "depth_mean_mcmc": r["functionals"]["depth_region"]["mean"], "depth_sd_mcmc": r["functionals"]["depth_region"]["sd"],
                     "depth_mean_laplace": None if r["laplace"]["strip"] is None else r["laplace"]["strip"]["depth_mean"],
                     "depth_sd_laplace": None if r["laplace"]["strip"] is None else r["laplace"]["strip"]["depth_sd"],
                     "phi_mean": r["phi_physical"]["mean"], "phi_map": r["phi_physical"]["map_phi"], "acceptance": r["sampler"]["acceptance"],
                     "std_ratio_median": r["pixel_std"]["mcmc_over_laplace_median"], "std_ratio_strip": r["pixel_std"]["mcmc_over_laplace_strip_mean"]})
    # the five-chain analyses (four overdispersed chains and one started at the truth, ``finalize --chains 5 --tag _truth5``), next to the four-chain ones
    truth5 = {}
    for p in sorted((out_dir / "mcmc").glob("*/summary_truth5*.json")):
        r = json.loads(p.read_text())
        vr = r["valley_region"]
        truth5[r["name"] + r["tag"]] = {"name": r["name"], "tag": r["tag"], "skip_segments": r.get("skip_segments", 0), "tail_segments": r.get("tail_segments"),
                             "chains": r["chains"], "draws_per_chain": r["draws_per_chain"], "chain_starts": r["chain_starts"],
                             "converged": r["convergence"]["converged"], "converged_classic": r["convergence"].get("converged_classic", r["convergence"]["converged"]),
                             "unconverged_reasons": r["convergence"]["unconverged_reasons"],
                             "rhat_phi": r["convergence"]["rhat_phi"], "rhat_pixel_max": r["convergence"]["rhat_pixel_max"], "ess_pixel_min": r["convergence"]["ess_pixel_min"],
                             "rhat_phi_prime": r["convergence"].get("rhat_phi_prime"), "ess_phi_prime": r["convergence"].get("ess_phi_prime"),
                             "ess_phi_prime_per_chain": r["convergence"].get("ess_phi_prime_per_chain"), "final_step": r["sampler"]["final_step"],
                             "rhat_depth": r["functionals"]["depth_region"]["rhat"], "ess_depth": r["functionals"]["depth_region"]["ess"],
                             "p_region_mcmc": None if vr is None else vr["p_valley_mcmc"], "p_region_mcse": None if vr is None else vr["mcse"],
                             "p_region_per_chain": None if vr is None else vr["per_chain"], "p_region_laplace": None if vr is None else vr["p_valley_laplace"],
                             "p_point_mcmc": r["valley_point"]["p_valley_mcmc"], "p_point_per_chain": r["valley_point"]["per_chain"],
                             "depth_mean_mcmc": r["functionals"]["depth_region"]["mean"], "depth_per_chain_mean": r["functionals"]["depth_region"]["per_chain_mean"],
                             "phi_per_chain_mean": r["phi_physical"]["per_chain_mean"], "acceptance_per_chain": r["sampler"]["acceptance_per_chain"]}
    for key, rec in truth5.items():  # the transient of the chain started at the truth: block means of the physical Phi and the valley depth, per chain
        series = out_dir / "mcmc" / rec["name"] / f"series{rec['tag']}.npz"
        if series.exists():
            z = np.load(series)
            blocks = np.array_split(np.arange(z["phi"].shape[1]), 6)
            rec["block_means"] = {"blocks": [[int(b[0]), int(b[-1]) + 1] for b in blocks],
                                  "phi": [[float(z["phi"][c, b].mean()) for b in blocks] for c in range(z["phi"].shape[0])],
                                  **({"phi_prime": [[float(z["phi_prime"][c, b].mean()) for b in blocks] for c in range(z["phi_prime"].shape[0])]} if "phi_prime" in z.files else {}),
                                  "depth_region": [[float(z["depth_region"][c, b].mean()) for b in blocks] for c in range(z["depth_region"].shape[0])]}
    rl.write_json(out_dir / "mcmc.json", {"table": rows, "summaries": sums, "with_truth_chain": truth5})
    # figure: depth histograms (MCMC) against the Laplace Gaussian for every configuration
    names = sorted(sums)
    cols = min(4, len(names))
    nrows = int(np.ceil(len(names) / cols))
    fig = Figure(figsize=(3.4 * cols, 2.8 * nrows), dpi=120)
    axes = np.atleast_2d(fig.subplots(nrows, cols, squeeze=False))
    for ax, name in zip(axes.ravel(), names):
        r = sums[name]
        z = np.load(out_dir / "mcmc" / name / "series.npz")
        d = z["depth_region"].ravel() if r["region_status"] == "ok" else (z["c1"] + z["c2"]).ravel() * 0.5 - z["mid"].ravel()
        label = "region depth" if r["region_status"] == "ok" else "point depth"
        ax.hist(d, bins=40, density=True, color="#2a78d6", alpha=0.55, lw=0)
        lap = r["laplace"]["strip"] if r["region_status"] == "ok" else r["laplace"]["point"]
        xs = np.linspace(d.min() - 0.5 * d.std(), d.max() + 0.5 * d.std(), 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - lap["depth_mean"]) / lap["depth_sd"]) ** 2) / (lap["depth_sd"] * np.sqrt(2 * np.pi)), color="#eb6834", lw=1.4)
        ax.axvline(r["delta"], color="#0b0b0b", lw=0.9, ls=":")
        key = "valley_region" if r["region_status"] == "ok" else "valley_point"
        ax.set_title(f"{name}\nP(valley): MCMC {r[key]['p_valley_mcmc']:.3f}, Laplace {r[key]['p_valley_laplace']:.3f}"
                     f"{'' if r['convergence']['converged'] else ' (unconverged)'}", fontsize=7, color=rl.INK, loc="left")
        rl.style(ax, f"{label} = mean(cores) - strip", "density")
    fig.tight_layout()
    fig.savefig(out_dir / "mcmc_spotcheck.png")
    print(f"wrote {out_dir / 'mcmc.json'} ({len(sums)} configurations)")


def stage_refeature(args):
    """P(gap valley) from the stored draws for other region definitions, per configuration with a summary.json, next to the Laplace value (laplace_sensitivity.json).

    Draws: all ``--chains`` chains of the run directory (``--chains 5`` includes the chain started at the truth; ``--tag _truth5`` then names the output). Per
    configuration and variant: ``p_mcmc`` with its Monte Carlo standard error (Sokal ESS of the indicator, summed over the chains), the per-chain values, ``p_laplace``.
    Writes <out-dir>/mcmc_refeature<tag>.json.
    """
    out_dir = Path(args.out_dir)
    sens = json.loads((out_dir / "laplace_sensitivity.json").read_text())["configurations"]
    res = {}
    for summary in sorted((out_dir / "mcmc").glob("*/summary.json")):
        if args.runs is not None and summary.parent.name not in args.runs:
            continue
        meta = json.loads(summary.read_text())
        run, name = meta["name"], meta.get("config", meta["name"])  # name: the configuration of laplace_sensitivity.json, run: the run directory
        per_chain = load_segments(summary.parent, args)
        theta = rpc.assemble_chains(per_chain)["theta"]
        problem = rl.build_problem(args, meta["g"], meta["eps_r"])
        rec = {"g": meta["g"], "eps_r": meta["eps_r"], "prior": meta["prior"], "chains": int(theta.shape[0]), "draws_per_chain": int(theta.shape[1]),
               "converged_in_summary_json": meta["convergence"]["converged"], "rhat_pixel_max_in_summary_json": meta["convergence"]["rhat_pixel_max"],
               "note_convergence": "the two convergence entries are those of the four-chain summary.json, not of the chains used here when --chains differs",
               "variants": {}}
        for b in args.band_fracs:
            for c in args.core_fracs:
                fg = gv.feature_geometry(problem.grid, problem.obj, c, b)
                key = rl.variant_key(b, c)
                lap = sens.get(name, {}).get(key, {}).get("p_valley_region")
                if not fg.representable:
                    rec["variants"][key] = {"band_frac": b, "core_frac": c, "region_status": "unrepresentable on this grid", "p_mcmc": None, "p_laplace": lap}
                    continue
                w = fg.weights
                ind = gv.valley_indicator(theta, w["strip"], w["core1"], w["core2"], fg.delta)
                p = float(ind.mean())
                ess = float(sum(dg.ess(x.astype(float)) if x.std() > 0 else len(x) for x in ind))
                rec["variants"][key] = {"band_frac": b, "core_frac": c, "region_status": "ok", "n_strip": fg.n_strip, "n_core": fg.n_core1, "p_mcmc": p,
                                        "mcse": float(np.sqrt(max(p * (1 - p), 0.0) / max(ess, 1.0))), "per_chain": [float(x) for x in ind.mean(axis=1)],
                                        "ess_indicator": ess, "p_laplace": lap, "difference_mcmc_minus_laplace": None if lap is None else p - lap}
        res[run] = rec
        print(f"{run} ({rec['chains']} x {rec['draws_per_chain']} draws): " + ", ".join(
            f"{k} {v['p_mcmc']:.3f} (Laplace {v['p_laplace']:.3f})" for k, v in rec["variants"].items() if v["p_mcmc"] is not None and v["p_laplace"] is not None))
    if not res:
        raise SystemExit(f"no summaries in {out_dir / 'mcmc'}")
    rl.write_json(out_dir / f"mcmc_refeature{args.tag}.json", {"configurations": res, "chains": args.chains,
                                                               "note": "p_mcmc: fraction of the stored draws with strip < min(cores) - delta; p_laplace: laplace_sensitivity.json"})
    return res


def stage_energy(args):
    """Add the Phi' criteria to every summary*.json under <out-dir>/mcmc/ that lacks them (module docstring); returns {file stem: the extended convergence dict}."""
    out, done = Path(args.out_dir) / "mcmc", {}
    for path in sorted(out.glob("*/summary*.json")):
        if args.runs is not None and path.parent.name not in args.runs:
            continue
        r = json.loads(path.read_text())
        if "rhat_phi_prime" in r["convergence"]:
            done[f"{path.parent.name}/{path.stem}"] = r["convergence"]
            continue
        sub = SimpleNamespace(chains=r["chains"], max_segments=None, skip_segments=r.get("skip_segments", 0), tail_segments=r.get("tail_segments"))
        ch = rpc.assemble_chains(load_segments(path.parent, sub))
        if ch["phi_prime"].shape[1] != r["draws_per_chain"]:
            raise SystemExit(f"{path}: {ch['phi_prime'].shape[1]} draws per chain on disk against {r['draws_per_chain']} in the summary (more segments since it was written?)")
        r["convergence"] = energy_convergence(r["convergence"], ch["phi_prime"])
        r["convergence_note"] = "Phi' criteria added afterwards by --stage energy from the stored segments (energy_convergence)"
        rl.write_json(path, r)
        done[f"{path.parent.name}/{path.stem}"] = r["convergence"]
        c = r["convergence"]
        print(f"{path.parent.name}/{path.stem}: Phi' R-hat {c['rhat_phi_prime']:.3f}, ESS {c['ess_phi_prime']:.0f} (per chain {np.round(c['ess_phi_prime_per_chain'], 1).tolist()}) -> "
              f"{'CONVERGED' if c['converged'] else 'UNCONVERGED'} (classic: {'met' if c['converged_classic'] else 'missed'})")
    return done


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "energy":
        return stage_energy(args)
    if args.stage == "chain":
        return stage_chain(args)
    if args.stage == "finalize":
        return stage_finalize(args)
    if args.stage == "refeature":
        return stage_refeature(args)
    return stage_collect(args)


if __name__ == "__main__":
    main()
