"""The DESIGN 13.3 pixel posteriors of the two-disk object: MAP, dense Laplace covariance and the gap-valley probability.

One configuration is (gap g, contrast eps_r, prior). Problem: two disks of radius 0.2 and permittivity eps_r, edge-to-edge gap g, in a DOI
of side 1.5 wavelengths; the book's rings (40 transmitters at radius 4, 60 receivers at radius 4.5); data from the n = 128 grid
(h = 0.0117, exact-coverage rendering, tolerance 1e-10) at 30 dB with a fixed noise seed (--seed 0, the same key for every configuration),
inference on the n = 64 grid (h = 0.0234; the truth on it is the exact-coverage rendering, so the data are not an inverse crime), link
theta = log eps_r. Priors: Matern (sigma 0.5, ell 0.4, nu 1.5; the cropped Gaussian of DESIGN 3, ``mrf.GaussianEnergy``) and the pseudo-Huber
TV prior of DESIGN 12.1 (delta 0.01, sigma_0 1) at the weight lambda* that the discrepancy rule picks for this geometry (stage ``lambda``).

Stages (--stage):

  lambda          MAP sweep over the TV weight at the reference gap (--lambda-gap, 0.25) for every contrast; lambda*(eps_r) is the swept
                  value whose Phi_MAP is closest to N = 2400 (Morozov's discrepancy rule, a MAP heuristic; the log-log interpolated crossing is
                  also stored). One JSON per (eps_r, lambda) in <out-dir>/lambda/, merged into <out-dir>/laplace_lambda.json by
                  ``lambda-collect``. lambda* is therefore a function of the contrast, not of the gap.
  config          one (g, eps_r, prior): the MAP by damped Gauss-Newton with the Laplace mass (``thetaspace.map_newton``; it reaches the same point as L-BFGS in a
                  fifth of the time and, for the Matern prior, converges where 1000 L-BFGS iterations do not) from several starts, the lowest U among the runs that
                  converged winning (``start_and_map``: vacuum and the smooth-prior start of milestones 4 and 5 for Matern; the smooth start, the Matern MAP and the
                  next wider gap's MAP for TV), the dense Laplace mass H = H_GN + Hess R at the MAP (exact Jacobian by reciprocity, Cholesky), and the gap-valley
                  probability of the Gaussian N(MAP, H^-1) (``gapvalley``): resolution-scale (strip against the disk cores, delta = log(eps_r) / 2;
                  "unrepresentable" if the strip is narrower than a pixel) and the labeled single-point values. Writes <out-dir>/laplace/<name>.json and .npz; a
                  configuration whose JSON exists is skipped (``--force`` redoes it), so an interrupted batch loses at most the configurations in progress.
                  ``--lam-scale`` multiplies lambda* (a sensitivity run; the name gets the suffix _x<scale>).
  descend         every gap of ONE (eps_r, prior) from the widest to the narrowest, each with the previous successful MAP as an extra start (the unit of the
                  batch jobs: the continuation in g finds the data-consistent MAP of the narrow gaps at eps_r = 4, where the vacuum and smooth starts end in
                  local minima with Phi ~ 3e5). A configuration whose MAP failed (not converged or Phi > 1.1 N) is stored but carries no probability.
  refeature       the gap-valley probability of every stored configuration (or of --names) for other strip heights (--band-fracs) and core radii (--core-fracs), from the
                  stored MAPs: laplace_sensitivity.json (does the verdict depend on the region definition?). ``descend --then-refeature`` does it for the configurations
                  of that descent.
  null            the negative control of the feature: the posterior of the data of a CLOSED gap (touching disks, --g 0 by default) for ONE (eps_r, prior), scored with
                  the strips of the gaps --strip-gaps and the variants --band-fracs x --core-fracs; <out-dir>/null/<name>.json and .npz. A feature that reads "present"
                  at a closed gap does not detect the gap.
  null-collect    <out-dir>/null/*.json and laplace_sensitivity.json -> <out-dir>/null.json (closed gap against open gap, per prior, strip gap and variant) and null_control.png.
  collect         every JSON on disk -> <out-dir>/laplace.json (table, per-configuration records) and the figures laplace_pvalley.png (P(gap valley)
                  against g for both priors with the Smith / CRB overlay), laplace_maps.png, laplace_profiles.png, laplace_lambda.png.

A MAP counts as failed (``map_ok`` false: no probability is reported, the record is kept) if Newton did not converge or Phi_MAP exceeds 1.1 N.

Threads: one XLA thread per process (``scripts/batch_run.py --threads 1``, jobs from ``make_m6_jobs.py``); the 4096 x 4096 dense algebra runs on the BLAS
threads of ``_threadcap``.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
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
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.linalg import solve_triangular

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import gapvalley as gv
from bayes_scatter import inference as inf
from bayes_scatter import mrf
from bayes_scatter import shapes as sh
from bayes_scatter import thetaspace as ts
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)

GAPS = (0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.32, 0.5)
EPS = (1.1, 2.0, 4.0)
LAMS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
PRIORS = ("matern", "tv")
NULL_STRIP_GAPS = (0.025, 0.05, 0.1, 0.15)
MAP_OK_PHI_OVER_N = 1.1  # a MAP whose Phi exceeds 1.1 N, or that did not converge, is a failed optimization and carries no probability
COLORS = {"matern": "#2a78d6", "tv": "#eb6834", "truth": "#0b0b0b"}
EPS_COLORS = {1.1: "#2a78d6", 2.0: "#eb6834", 4.0: "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------------------------
# Arguments and names
# ----------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=["config", "descend", "lambda", "lambda-collect", "collect", "refeature", "null", "null-collect"], required=True)
    g = ap.add_argument_group("configuration (stage config)")
    g.add_argument("--g", type=float, help="edge-to-edge gap in wavelengths")
    g.add_argument("--eps-r", type=float, help="relative permittivity of both disks")
    g.add_argument("--prior", choices=PRIORS)
    g.add_argument("--lam", type=float, default=None, help="TV weight; default lambda*(eps_r) of <out-dir>/laplace_lambda.json times --lam-scale")
    g.add_argument("--lam-scale", type=float, default=1.0, help="multiplier of lambda* (sensitivity runs)")
    g.add_argument("--neighbor-gap", type=float, default=None, help="config: also start from the stored MAP of this (wider) gap")
    g.add_argument("--gaps", type=float, nargs="+", default=list(GAPS), help="descend: the gaps, run from the widest to the narrowest")
    g = ap.add_argument_group("lambda sweep (stage lambda)")
    g.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    g.add_argument("--eps", type=float, nargs="+", default=list(EPS), help="contrasts of the sweep and of ``collect``")
    g.add_argument("--lambda-gap", type=float, default=0.25, help="gap of the object the discrepancy rule is applied to")
    g = ap.add_argument_group("problem")
    g.add_argument("--n", type=int, default=64, help="inference grid")
    g.add_argument("--data-factor", type=int, default=2, help="data grid = this many times finer than --n (128)")
    g.add_argument("--side", type=float, default=1.5)
    g.add_argument("--radius", type=float, default=0.2)
    g.add_argument("--snr", type=float, default=30.0)
    g.add_argument("--seed", type=int, default=0, help="noise key, the same for every configuration")
    g.add_argument("--n-tx", type=int, default=40)
    g.add_argument("--n-rx", type=int, default=60)
    g.add_argument("--tol", type=float, default=1e-10)
    g = ap.add_argument_group("priors and MAP")
    g.add_argument("--sigma", type=float, default=0.5)
    g.add_argument("--ell", type=float, default=0.4)
    g.add_argument("--nu", type=float, default=1.5)
    g.add_argument("--delta-tv", type=float, default=0.01, help="pseudo-Huber width")
    g.add_argument("--sigma-0", type=float, default=1.0)
    g.add_argument("--start-maxiter", type=int, default=100, help="L-BFGS iterations of the smooth-prior start")
    g.add_argument("--newton-iter", type=int, default=60, help="Newton iterations from the vacuum start")
    g.add_argument("--newton-iter-smooth", type=int, default=40, help="Newton iterations from the smooth-prior start")
    g.add_argument("--newton-tol", type=float, default=1e-3, help="stop at a Newton decrement below this (nats)")
    g.add_argument("--core-frac", type=float, default=gv.CORE_FRAC)
    g.add_argument("--band-fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25], help="refeature: strip half-height as a fraction of the disk radius")
    g.add_argument("--core-fracs", type=float, nargs="+", default=[0.5, 0.25], help="refeature: core radius as a fraction of the disk radius")
    g.add_argument("--strip-gaps", type=float, nargs="+", default=list(NULL_STRIP_GAPS), help="null: the gaps whose strips score the closed-gap posterior")
    g.add_argument("--all-starts", action="store_true", help="run every start of ``start_and_map`` instead of stopping at the first successful one")
    g.add_argument("--names", nargs="+", default=None, help="refeature: only these configurations (file stems of <out-dir>/laplace/, e.g. g0.05_eps2_tv_x3); default all")
    g.add_argument("--then-refeature", action="store_true", help="descend: afterwards run the refeature stage on the configurations just computed (one job per sensitivity run)")
    g.add_argument("--out-dir", default=str(RESULTS))
    g.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.stage == "config" and (args.g is None or args.eps_r is None or args.prior is None):
        ap.error("--stage config needs --g, --eps-r and --prior")
    if args.stage in ("descend", "null") and (args.eps_r is None or args.prior is None):
        ap.error(f"--stage {args.stage} needs --eps-r and --prior")
    return args


def config_name(g, eps, prior, lam_scale=1.0):
    """File stem of a configuration: g<g>_eps<eps>_<prior>, with _x<scale> for a TV sensitivity run."""
    stem = f"g{g:g}_eps{eps:g}_{prior}"
    return stem if prior != "tv" or lam_scale == 1.0 else f"{stem}_x{lam_scale:g}"


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)  # an interrupted write never leaves a half-written JSON that would count as done


def thread_env():
    return {k: os.environ.get(k) for k in ("NPROC", "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_PLATFORMS")}


# ----------------------------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------------------------


def build_problem(args, g, eps):
    """Object, data on the fine grid, the inference-grid forward model and the pixelized truth.

    Returns a namespace: ``obj`` (``shapes.TwoDisks``), ``setup``, ``grid`` (inference), ``forward``, ``chi_true`` / ``theta_true`` ((n, n), the
    exact-coverage rendering on the inference grid), ``d_clean`` / ``d_obs`` ((Nt, Nr), from the data grid ``n * data_factor``), ``s`` (noise std at
    ``--snr``), ``sha`` (hash of d_obs) and ``n_data`` (the number of complex data N).
    """
    obj = sh.TwoDisks(g=g, r=args.radius, eps_r=eps)
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    data_setup = Setup(Grid(args.n * args.data_factor, args.side), setup.antennas)
    d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(obj.chi(data_setup.grid)))
    s = noise_std_from_snr(d_clean, args.snr)
    d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
    chi_true = obj.chi(setup.grid)
    sha = hashlib.sha256(np.ascontiguousarray(np.asarray(d_obs, dtype=np.complex128)).tobytes()).hexdigest()
    return SimpleNamespace(obj=obj, setup=setup, grid=setup.grid, forward=Forward(setup, tol=args.tol), chi_true=chi_true,
                           theta_true=np.log1p(chi_true), d_clean=d_clean, d_obs=d_obs, s=s, sha=sha, n_data=args.n_tx * args.n_rx)


def build_prior(args, name, grid, lam=None):
    """The ``mrf`` prior: the cropped Matern field or the pseudo-Huber TV prior (weight ``lam``)."""
    if name == "matern":
        return mrf.GaussianEnergy(GaussianField(grid, "matern", sigma=args.sigma, ell=args.ell, nu=args.nu, mean=0.0))
    return mrf.PseudoHuberTV(grid.n, lam=lam, delta=args.delta_tv, sigma_0=args.sigma_0, mu=0.0)


def chol_solver(chol):
    """``solve(B)`` = H^-1 B for H = L L^T with the lower Cholesky factor ``chol`` (two triangular solves), B (N, K) or (N,)."""
    return lambda B: solve_triangular(chol, solve_triangular(chol, B, lower=True), lower=True, trans=1)


def lambda_star_of(out_dir, eps):
    """lambda*(eps_r) from <out-dir>/laplace_lambda.json (exits if the sweep has not been collected)."""
    p = Path(out_dir) / "laplace_lambda.json"
    if not p.exists():
        raise SystemExit(f"{p} does not exist: run --stage lambda and --stage lambda-collect first, or pass --lam")
    data = json.loads(p.read_text())
    key = f"{eps:g}"
    if key not in data["lambda_star"]:
        raise SystemExit(f"{p} has no lambda* for eps_r = {eps:g} (it has {sorted(data['lambda_star'])})")
    return float(data["lambda_star"][key]["lam"])


def _history(ninfo):
    return [{k: h[k] for k in ("it", "U", "decrement", "alpha")} for h in ninfo["history"]]


def start_and_map(problem, target, args, prior_name, theta0=None, theta_matern=None, theta_neighbor=None):
    """The MAP by damped Gauss-Newton with the Laplace mass (``thetaspace.map_newton``) from a sequence of starts. Returns ``(theta_smooth_start, theta_hat, info)``.

    Starts, in order, for the Matern prior: ``neighbor`` (``theta_neighbor``: the MAP of the next wider gap, a continuation in g that ``stage_descend``
    supplies), ``smooth`` (the smooth-prior start of milestones 4 and 5, 100 L-BFGS iterations of the whitened Matern MAP, ``thetaspace.smooth_start``; computed
    here unless ``theta0`` is given) and ``vacuum`` (theta = 0); for the TV prior ``matern`` (``theta_matern``: the Matern MAP of the same configuration),
    ``neighbor`` and ``smooth``. A run is successful if Newton converged and Phi < 1.1 N. By default the sequence stops at the first successful run
    (``--all-starts`` runs every start): U is not convex in theta for the full forward model, and at eps_r = 4 a failed start ends at Phi ~ 3e5 after
    all its iterations (about 7 s each), so the expensive runs are the failing ones; the vacuum start is the last resort (it succeeded where the smooth
    start failed at eps_r = 4, g = 0.1, and failed at eps_r = 4, g = 0.25 where the smooth start succeeded). ``info["candidates"]`` holds every run made and
    ``info["map"]`` the winner (lowest U among the successful runs, else among the converged ones, else the lowest U), with ``u_gap`` the largest difference in U
    (nats) between runs that converged (None if fewer than two): with ``--all-starts`` it is the indicator of a second local minimum.
    """
    info = {"candidates": {}}
    n = problem.grid.n
    smooth = {"theta": None if theta0 is None else jnp.asarray(theta0)}

    def smooth_start():
        if smooth["theta"] is None:
            t0 = time.perf_counter()
            th, start_info = ts.smooth_start(problem.forward, problem.d_obs, problem.s, problem.grid, maxiter=args.start_maxiter)
            smooth["theta"] = jnp.asarray(th)
            info["start"] = {"maxiter": args.start_maxiter, "nit": start_info["nit"], "nfev": start_info["nfev"], "phi": start_info["phi"],
                             "time_s": time.perf_counter() - t0,
                             "note": "whitened-coordinate Matern(0.5, 0.4, 1.5) MAP, theta = log eps_r of the result (the milestone 4 and 5 recipe)"}
        return smooth["theta"]

    if prior_name == "matern":
        order = [("neighbor", theta_neighbor, args.newton_iter), ("smooth", None, args.newton_iter_smooth), ("vacuum", jnp.zeros((n, n)), args.newton_iter)]
    else:
        order = [("matern", theta_matern, args.newton_iter_smooth), ("neighbor", theta_neighbor, args.newton_iter), ("smooth", None, args.newton_iter_smooth)]
    runs = []
    for label, start, max_iter in order:
        if start is None and label != "smooth":
            continue
        th, ninfo = ts.map_newton(target, smooth_start() if label == "smooth" else jnp.asarray(start), max_iter=max_iter, tol=args.newton_tol)
        runs.append((label, np.asarray(th), ninfo))
        info["candidates"][label] = {k: v for k, v in ninfo.items() if k != "history"} | {"max_iter": max_iter, "history": _history(ninfo)}
        if not args.all_starts and ninfo["converged"] and ninfo["phi"] < MAP_OK_PHI_OVER_N * problem.n_data:
            break
    best = min(runs, key=lambda r: (not (r[2]["converged"] and r[2]["phi"] < MAP_OK_PHI_OVER_N * problem.n_data), not r[2]["converged"], r[2]["U"]))
    ninfo = best[2]
    done = [r[2]["U"] for r in runs if r[2]["converged"]]
    info["map"] = {k: v for k, v in ninfo.items() if k != "history"} | {
        "method": "damped Gauss-Newton with the Laplace mass (thetaspace.map_newton), starts tried: " + "/".join(r[0] for r in runs),
        "chosen_start": best[0], "tol": args.newton_tol, "u_gap": float(max(done) - min(done)) if len(done) > 1 else None, "all_starts": bool(args.all_starts),
        "history": _history(ninfo), "time_s_all_starts": sum(r[2]["time_s"] for r in runs)}
    info.setdefault("start", None)
    return None if smooth["theta"] is None else np.asarray(smooth["theta"]), best[1], info


def matern_map(problem, args, theta0=None):
    """The Matern-prior MAP of ``problem`` (``start_and_map`` with the smooth start, then the vacuum): ``(theta_hat, info)``; the start of the TV runs."""
    target = ts.ThetaTarget(problem.forward, build_prior(args, "matern", problem.grid), problem.d_obs, problem.s)
    _, th, info = start_and_map(problem, target, args, "matern", theta0=theta0)
    return th, info


# ----------------------------------------------------------------------------------------------
# One configuration
# ----------------------------------------------------------------------------------------------


def run_config(args, g, eps, prior_name, lam_scale=1.0, theta_neighbor=None):
    """The record of one configuration (module docstring): operating point, MAP, Laplace mass, features; also returns the arrays to store.
    ``theta_neighbor`` (n, n) is the MAP of the next wider gap, an extra start (``start_and_map``)."""
    t_all = time.perf_counter()
    problem = build_problem(args, g, eps)
    grid, n = problem.grid, args.n
    lam = None
    if prior_name == "tv":
        lam = args.lam if args.lam is not None else lambda_star_of(args.out_dir, eps) * lam_scale
    t0 = time.perf_counter()
    prior = build_prior(args, prior_name, grid, lam)
    t_prior = time.perf_counter() - t0
    target = ts.ThetaTarget(problem.forward, prior, problem.d_obs, problem.s)
    op = inf.operating_point(problem.forward, jnp.asarray(problem.chi_true), problem.d_clean, problem.d_obs, problem.s)
    print(f"{config_name(g, eps, prior_name, lam_scale)}: operating point model/noise {op['model_to_noise']:.3f}, Phi(truth) {op['phi_truth']:.0f} (N = {problem.n_data}); prior {t_prior:.1f} s")
    theta_matern = None
    if prior_name == "tv":
        theta_matern = matern_start_for(args, problem, g, eps)
    theta0, theta_hat, minfo = start_and_map(problem, target, args, prior_name, theta_matern=theta_matern, theta_neighbor=theta_neighbor)
    m = minfo["map"]
    cand = minfo["candidates"]
    print(f"  MAP ({m['chosen_start']} start): {m['nit']} Newton iterations, converged {m['converged']}, decrement {m['decrement']:.1e}; Phi {m['phi']:.1f} R {m['R']:.1f} U {m['U']:.2f}; "
          + ", ".join(f"U {k} {v['U']:.2f} ({v['nit']} its, {'converged' if v['converged'] else 'NOT converged'})" for k, v in cand.items()) + f"; {m['time_s_all_starts']:.0f} s")
    t0 = time.perf_counter()
    mass = target.laplace_mass(theta_hat, verify_columns=2)
    solve = chol_solver(mass.chol)
    fg = gv.feature_geometry(grid, problem.obj, args.core_frac)
    feats = gv.gaussian_valley(fg, theta_hat, solve)
    t_features = time.perf_counter() - t0
    t0 = time.perf_counter()
    lap_std = mass.theta_std()
    t_std = time.perf_counter() - t0
    chi_hat = np.asarray(target.chi(theta_hat))
    d_hat, _ = problem.forward(jnp.asarray(chi_hat))
    w = fg.weights
    th = theta_hat.ravel()
    map_strip, map_c1, map_c2 = float(w["strip"] @ th), float(w["core1"] @ th), float(w["core2"] @ th)
    inside = problem.chi_true > 1e-9
    rec = {"name": config_name(g, eps, prior_name, lam_scale), "g": g, "eps_r": eps, "prior": prior_name, "prior_params": prior.describe(),
           "lam": lam, "lam_scale": lam_scale if prior_name == "tv" else None,
           "problem": {"n": n, "side": args.side, "h": grid.h, "data_grid_n": args.n * args.data_factor, "radius": args.radius, "snr_db": args.snr,
                       "seed": args.seed, "n_tx": args.n_tx, "n_rx": args.n_rx, "n_data": problem.n_data, "noise_std": problem.s,
                       "d_obs_sha256": problem.sha, "tol": args.tol},
           "operating_point": op, "start": minfo["start"], "map": minfo["map"], "map_candidates": minfo["candidates"], "neighbor_start_given": theta_neighbor is not None,
           "map_summary": {"phi": m["phi"], "R": m["R"], "U": m["U"], "phi_over_N": m["phi"] / problem.n_data, "grad_inf": m["grad_inf"],
                           "converged": m["converged"], "chosen_start": m["chosen_start"], "u_gap": m["u_gap"],
                           "map_ok": bool(m["converged"] and m["phi"] < MAP_OK_PHI_OVER_N * problem.n_data), "err_model": inf.err_model(1.0 + chi_hat, 1.0 + problem.chi_true),
                           "err_chi": inf.err_chi(chi_hat, problem.chi_true), "err_data": inf.err_data(d_hat, problem.d_clean),
                           "strip_mean": map_strip, "core1_mean": map_c1, "core2_mean": map_c2,
                           "map_has_valley_region": bool(map_strip < min(map_c1, map_c2) - fg.delta) if fg.representable else None,
                           "map_has_valley_point": bool(float(w["mid"] @ th) < min(float(w["c1"] @ th), float(w["c2"] @ th)) - fg.delta)},
           "laplace": {**{k: v for k, v in mass.info.items()},
                       "theta_std_inside": float(lap_std[inside].mean()), "theta_std_outside": float(lap_std[~inside].mean()),
                       "theta_std_strip": float(lap_std[fg.strip].mean()) if fg.n_strip else None, "theta_std_max": float(lap_std.max())},
           "features": feats, "truth": gv.truth_valley(fg, problem.theta_true),
           "timings_s": {"prior": t_prior, "laplace_and_features": t_features, "laplace_std": t_std, "total": time.perf_counter() - t_all},
           "thread_env": thread_env(), "load_avg_1min": os.getloadavg()[0]}
    region = feats["region"]
    print(f"  Laplace mass {mass.info['time_total_s']:.1f} s (Jacobian check {mass.info.get('jacobian_check_rel_err'):.1e}); "
          f"P(valley) region {'unrepresentable' if region is None else format(region['p_valley'], '.4f')}, point {feats['point']['p_valley']:.4f}; total {rec['timings_s']['total']:.0f} s")
    arrays = {"theta_hat": theta_hat, "laplace_std": lap_std.astype(np.float32), "theta_true": problem.theta_true,
              "chi_true": problem.chi_true, "d_obs": np.asarray(problem.d_obs)}
    if theta0 is not None:
        arrays["theta_start"] = theta0
    return rec, arrays


def matern_start_for(args, problem, g, eps):
    """The Matern MAP of the same (g, eps_r) from its configuration file if it exists and is a successful MAP, else computed (``matern_map``)."""
    jpath, npath = config_paths(args.out_dir, config_name(g, eps, "matern"))
    if jpath.exists() and npath.exists() and json.loads(jpath.read_text())["map_summary"]["map_ok"]:
        return np.load(npath)["theta_hat"]
    print("  (no successful Matern configuration on disk: computing the Matern MAP as a start)")
    th, minfo = matern_map(problem, args)
    return th if minfo["map"]["converged"] and minfo["map"]["phi"] < MAP_OK_PHI_OVER_N * problem.n_data else None


def config_paths(out_dir, name):
    d = Path(out_dir) / "laplace"
    return d / f"{name}.json", d / f"{name}.npz"


def load_good_map(out_dir, name):
    """theta_hat of a stored configuration if its MAP is a successful one (``map_ok``), else None."""
    jpath, npath = config_paths(out_dir, name)
    if jpath.exists() and npath.exists() and json.loads(jpath.read_text())["map_summary"]["map_ok"]:
        return np.load(npath)["theta_hat"]
    return None


def write_config(args, name, rec, arrays):
    jpath, npath = config_paths(args.out_dir, name)
    npath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npath, **arrays)
    write_json(jpath, rec)  # last: the JSON is the done marker


def stage_config(args):
    name = config_name(args.g, args.eps_r, args.prior, args.lam_scale)
    jpath, _ = config_paths(args.out_dir, name)
    if jpath.exists() and not args.force:
        print(f"{name}: done ({jpath})")
        return json.loads(jpath.read_text())
    neighbor = None
    if args.neighbor_gap is not None:
        neighbor = load_good_map(args.out_dir, config_name(args.neighbor_gap, args.eps_r, args.prior, args.lam_scale))
    rec, arrays = run_config(args, args.g, args.eps_r, args.prior, args.lam_scale, theta_neighbor=neighbor)
    write_config(args, name, rec, arrays)
    return rec


def stage_descend(args):
    """The configurations of ONE (eps_r, prior) for the gaps ``--gaps`` from the widest to the narrowest, each with the MAP of the previous successful one
    as an extra start (a continuation in g: a narrow gap at high contrast has local minima that the vacuum and smooth starts fall into). A configuration whose
    JSON exists is skipped when its MAP succeeded (its theta_hat is still the next start) and recomputed when it failed."""
    prev, out = None, []
    for g in sorted(args.gaps, reverse=True):
        name = config_name(g, args.eps_r, args.prior, args.lam_scale)
        jpath, npath = config_paths(args.out_dir, name)
        if jpath.exists() and not args.force:
            good = load_good_map(args.out_dir, name)
            if good is not None:
                print(f"{name}: done")
                prev = good
                out.append(json.loads(jpath.read_text()))
                continue
            print(f"{name}: the stored MAP failed; computing it again with the continuation start")
        rec, arrays = run_config(args, g, args.eps_r, args.prior, args.lam_scale, theta_neighbor=prev)
        write_config(args, name, rec, arrays)
        out.append(rec)
        if rec["map_summary"]["map_ok"]:
            prev = arrays["theta_hat"]
        else:
            print(f"  WARNING: the MAP of {name} failed (Phi {rec['map_summary']['phi']:.0f}, converged {rec['map_summary']['converged']}); no probability is reported for it")
    if args.then_refeature:  # the sensitivity files of THIS descent only; laplace_sensitivity.json is merged afterwards (``--stage refeature``), not here, so that concurrent jobs never write it
        args.names = [r["name"] for r in out]
        stage_refeature(args, merge=False)
    return out


# ----------------------------------------------------------------------------------------------
# The TV weight (discrepancy rule)
# ----------------------------------------------------------------------------------------------


def lambda_path(out_dir, eps, g, lam):
    """The JSON of the MAP at TV weight ``lam`` for contrast ``eps`` and gap ``g`` of the sweep."""
    return Path(out_dir) / "lambda" / f"eps{eps:g}_g{g:g}_lam{lam:g}.json"


def stage_lambda(args):
    """MAP of the TV prior at every swept lambda for every contrast at the reference gap; one JSON each; the smooth start is shared per contrast."""
    out_dir = Path(args.out_dir)
    for eps in args.eps:
        todo = [lam for lam in args.lams if args.force or not lambda_path(out_dir, eps, args.lambda_gap, lam).exists()]
        if not todo:
            print(f"eps_r = {eps:g}: every lambda done")
            continue
        problem = build_problem(args, args.lambda_gap, eps)
        t0 = time.perf_counter()
        theta0, start_info = ts.smooth_start(problem.forward, problem.d_obs, problem.s, problem.grid, maxiter=args.start_maxiter)
        print(f"eps_r = {eps:g}, g = {args.lambda_gap:g}: smooth start {start_info['nit']} iterations, {time.perf_counter() - t0:.0f} s")
        theta_m, minfo_m = matern_map(problem, args, theta0=np.asarray(theta0))
        mm = minfo_m["map"]
        print(f"  Matern MAP ({mm['chosen_start']} start): {mm['nit']} iterations, converged {mm['converged']}, Phi {mm['phi']:.1f}")
        for lam in todo:
            prior = build_prior(args, "tv", problem.grid, lam)
            target = ts.ThetaTarget(problem.forward, prior, problem.d_obs, problem.s)
            _, theta_hat, minfo = start_and_map(problem, target, args, "tv", theta0=np.asarray(theta0), theta_matern=theta_m)
            m = minfo["map"]
            chi_hat = np.asarray(target.chi(theta_hat))
            rec = {"eps_r": eps, "g": args.lambda_gap, "lam": lam, "n_data": problem.n_data, "phi": m["phi"], "R": m["R"], "U": m["U"],
                   "phi_over_N": m["phi"] / problem.n_data, "converged": m["converged"], "nit": m["nit"], "time_s": m["time_s_all_starts"],
                   "chosen_start": m["chosen_start"], "u_gap": m["u_gap"], "U_by_start": {k: v["U"] for k, v in minfo["candidates"].items()},
                   "decrement": m["decrement"], "grad_inf": m["grad_inf"], "err_chi": inf.err_chi(chi_hat, problem.chi_true),
                   "err_model": inf.err_model(1.0 + chi_hat, 1.0 + problem.chi_true), "delta": args.delta_tv, "sigma_0": args.sigma_0,
                   "noise_std": problem.s, "d_obs_sha256": problem.sha}
            write_json(lambda_path(out_dir, eps, args.lambda_gap, lam), rec)
            print(f"  lambda {lam:g}: Phi_MAP {m['phi']:.1f} (N {problem.n_data}), R {m['R']:.1f}, err_chi {rec['err_chi']:.4f}, "
                  f"{m['nit']} iterations ({m['chosen_start']} start; starts tried {m['method'].split(': ')[-1]}), {m['time_s_all_starts']:.0f} s, converged {m['converged']}")


def crossing(recs, n_data):
    """Log-log interpolated lambda at Phi_MAP = N between the two neighbors that bracket it (None if Phi_MAP does not cross N)."""
    recs = sorted(recs, key=lambda r: r["lam"])
    for a, b in zip(recs[:-1], recs[1:]):
        if (a["phi"] - n_data) * (b["phi"] - n_data) < 0:
            t = (n_data - a["phi"]) / (b["phi"] - a["phi"])
            return float(np.exp(np.log(a["lam"]) + t * (np.log(b["lam"]) - np.log(a["lam"]))))
    return None


def stage_lambda_collect(args):
    out_dir = Path(args.out_dir)
    runs = {}
    for p in sorted((out_dir / "lambda").glob("eps*_g*_lam*.json")):
        r = json.loads(p.read_text())
        if abs(r["g"] - args.lambda_gap) < 1e-12:
            runs.setdefault(f"{r['eps_r']:g}", []).append(r)
    if not runs:
        raise SystemExit(f"no lambda records in {out_dir / 'lambda'}")
    star, cross = {}, {}
    for eps, recs in runs.items():
        n_data = recs[0]["n_data"]
        usable = [r for r in recs if r["converged"]] or recs  # a MAP that did not converge does not enter the discrepancy rule
        best = min(usable, key=lambda r: (abs(r["phi"] - n_data), r["lam"]))
        star[eps] = {"lam": best["lam"], "phi_map": best["phi"], "n_data": n_data, "swept": sorted(r["lam"] for r in recs),
                     "crossing_lam_interpolated": crossing(recs, n_data),
                     "criterion": "swept lambda with Phi_MAP closest to N (Morozov discrepancy rule, a MAP heuristic)"}
    out = {"reference_gap": args.lambda_gap, "prior": {"kind": "tv", "delta": args.delta_tv, "sigma_0": args.sigma_0},
           "lambda_star": star, "runs": {eps: sorted(recs, key=lambda r: r["lam"]) for eps, recs in runs.items()}}
    write_json(out_dir / "laplace_lambda.json", out)
    for eps, s_ in star.items():
        print(f"eps_r = {eps}: lambda* = {s_['lam']:g} (Phi_MAP {s_['phi_map']:.1f} against N = {s_['n_data']}); interpolated crossing {s_['crossing_lam_interpolated']}")
    fig = Figure(figsize=(5.4, 3.6), dpi=130)
    ax = fig.subplots()
    for eps, recs in sorted(runs.items(), key=lambda kv: float(kv[0])):
        recs = sorted(recs, key=lambda r: r["lam"])
        ax.semilogx([r["lam"] for r in recs], [r["phi"] for r in recs], color=EPS_COLORS.get(float(eps), "#2a78d6"), lw=1.6, marker="o", ms=4, label=f"eps_r = {eps}")
        ax.plot([star[eps]["lam"]], [star[eps]["phi_map"]], marker="*", ms=11, color=EPS_COLORS.get(float(eps), "#2a78d6"), mec=INK, mew=0.6)
    n_data = next(iter(star.values()))["n_data"]
    ax.axhline(n_data, color=MUTED, lw=0.9)
    ax.axhspan(n_data - np.sqrt(n_data), n_data + np.sqrt(n_data), color="#e4e3dd", alpha=0.6, lw=0)
    style(ax, "TV weight lambda", "Phi_MAP")
    ax.set_title(f"discrepancy rule at g = {args.lambda_gap:g}; N = {n_data}, band N +- sqrt(N)", fontsize=9, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "laplace_lambda.png")
    return out


# ----------------------------------------------------------------------------------------------
# Collection and figures
# ----------------------------------------------------------------------------------------------


def style(ax, xlabel, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def gap_axis(ax, gaps=(0.0125, 0.025, 0.05, 0.1, 0.2, 0.5)):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(gaps)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def truth_fails(holds):
    """True if the pixelized truth was evaluated and does NOT show the valley (``holds`` False; None = not evaluated)."""
    return holds is False


def table_row(rec):
    """The headline numbers of one configuration. A region probability whose pixelized truth fails the feature is not reported (``gapvalley.UNREPRESENTABLE_TRUTH``,
    milestone-6 review); records written before that rule keep the probability in the file and it is withheld here, with the stored value as ``p_valley_region_raw``."""
    f, mp = rec["features"], rec["map_summary"]
    reg, pt = f["region"], f["point"]
    raw = None if reg is None else reg["p_valley"]
    status = f["region_status"]
    if reg is not None and truth_fails(rec["truth"]["region"]["holds"]):
        reg, status = None, gv.UNREPRESENTABLE_TRUTH
    return {"name": rec["name"], "g": rec["g"], "eps_r": rec["eps_r"], "prior": rec["prior"], "lam": rec["lam"], "lam_scale": rec["lam_scale"],
            "region_status": status, "p_valley_region_raw": raw, "p_valley_region": None if reg is None else reg["p_valley"],
            "z_depth_region": None if reg is None else reg["z_depth"], "depth_mean_region": None if reg is None else reg["depth_mean"],
            "depth_sd_region": None if reg is None else reg["depth_sd"], "strip_mean": None if reg is None else reg["strip_mean"],
            "core_mean_avg": None if reg is None else 0.5 * (reg["core1_mean"] + reg["core2_mean"]),
            "p_valley_point": pt["p_valley"], "delta": f["delta"], "strip_width_pixels": f["geometry"]["strip_width_pixels"],
            "n_strip": f["geometry"]["n_strip"], "truth_holds_region": rec["truth"]["region"]["holds"], "truth_holds_point": rec["truth"]["point"]["holds"],
            "phi_map": mp["phi"], "phi_over_N": mp["phi_over_N"], "map_converged": mp["converged"], "map_ok": mp["map_ok"], "chosen_start": mp["chosen_start"],
            "u_gap": mp["u_gap"], "err_chi": mp["err_chi"],
            "map_has_valley_region": mp["map_has_valley_region"], "map_has_valley_point": mp["map_has_valley_point"],
            "model_to_noise": rec["operating_point"]["model_to_noise"], "phi_truth": rec["operating_point"]["phi_truth"],
            "theta_std_strip": rec["laplace"]["theta_std_strip"], "time_s": rec["timings_s"]["total"]}


def variant_key(band_frac, core_frac):
    return f"band{band_frac:g}_core{core_frac:g}"


def stage_refeature(args, merge=True):
    """The gap-valley probability of every stored configuration for other region definitions, from the stored MAP and a rebuilt Laplace mass.

    ``--band-fracs`` and ``--core-fracs`` give the variants (every combination); the stored configuration files are not modified. One file per configuration
    in <out-dir>/laplace_sens/ (a rerun skips the ones that exist), merged into <out-dir>/laplace_sensitivity.json with, per variant, the verdict table and the
    narrowest gap that is "present" for every wider gap (``verdict_table``). The question: do the verdicts depend on the definition of the strip and the cores
    (the full band includes the lens-shaped void above and below the narrowest point of the gap, which a narrower band leaves out)?
    """
    out_dir = Path(args.out_dir)
    variants = [(b, c) for b in args.band_fracs for c in args.core_fracs]
    sens_dir = out_dir / "laplace_sens"
    sens_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (out_dir / "laplace").glob("g*_eps*_*.json"))
    if args.names is not None:
        missing = sorted(set(args.names) - set(names))
        if missing:
            raise SystemExit(f"--names: no configuration {missing} in {out_dir / 'laplace'}")
        names = sorted(args.names)
    for name in names:
        path = sens_dir / f"{name}.json"
        if path.exists() and not args.force:
            continue
        rec = json.loads(config_paths(out_dir, name)[0].read_text())
        t0 = time.perf_counter()
        problem = build_problem(args, rec["g"], rec["eps_r"])
        target = ts.ThetaTarget(problem.forward, build_prior(args, rec["prior"], problem.grid, rec["lam"]), problem.d_obs, problem.s)
        theta_hat = np.load(config_paths(out_dir, name)[1])["theta_hat"]
        solve = chol_solver(target.laplace_mass(theta_hat).chol)
        res = {}
        for b, c in variants:
            fg = gv.feature_geometry(problem.grid, problem.obj, c, b)
            f = gv.gaussian_valley(fg, theta_hat, solve)
            reg = f["region"]
            res[variant_key(b, c)] = {"band_frac": b, "core_frac": c, "region_status": f["region_status"], "n_strip": fg.n_strip, "n_core": fg.n_core1,
                                      "p_valley_region": None if reg is None else reg["p_valley"], "z_depth_region": None if reg is None else reg["z_depth"],
                                      "strip_mean": None if reg is None else reg["strip_mean"], "depth_mean": None if reg is None else reg["depth_mean"],
                                      "depth_sd": None if reg is None else reg["depth_sd"], "truth_holds_region": gv.truth_valley(fg, problem.theta_true)["region"]["holds"]}
        write_json(path, {"name": name, "g": rec["g"], "eps_r": rec["eps_r"], "prior": rec["prior"], "lam": rec["lam"], "lam_scale": rec["lam_scale"],
                          "map_ok": rec["map_summary"]["map_ok"], "variants": res, "time_s": time.perf_counter() - t0})
        print(f"{name}: " + ", ".join(f"{k} {'-' if v['p_valley_region'] is None else format(v['p_valley_region'], '.3f')}" for k, v in res.items()))
    if merge:
        merge_sensitivity(out_dir, variants)


def merge_sensitivity(out_dir, variants):
    """laplace_sensitivity.json from the files of <out-dir>/laplace_sens/: per variant the probabilities and the verdict table (``verdict_table``)."""
    recs = [json.loads(p.read_text()) for p in sorted((Path(out_dir) / "laplace_sens").glob("*.json"))]
    out = {"variants": [{"band_frac": b, "core_frac": c, "key": variant_key(b, c)} for b, c in variants], "per_variant": {}}
    for b, c in variants:
        key = variant_key(b, c)
        rows = [{"eps_r": r["eps_r"], "g": r["g"], "prior": r["prior"], "lam_scale": r["lam_scale"], "map_ok": r["map_ok"],
                 "p_valley_region": None if truth_fails(r["variants"][key]["truth_holds_region"]) else r["variants"][key]["p_valley_region"],  # withheld: the truth fails
                 "p_valley_point": None} for r in recs]
        table, star = verdict_table(rows)
        out["per_variant"][key] = {"band_frac": b, "core_frac": c, "verdicts": table, "narrowest_gap_present_for_all_wider": star,
                                   "n_strip_by_gap": {f"{g:g}": next(r["variants"][key]["n_strip"] for r in recs if r["g"] == g) for g in sorted({r["g"] for r in recs})}}
    out["configurations"] = {r["name"]: {k: dict(v, region_status=gv.UNREPRESENTABLE_TRUTH, p_valley_region_raw=v["p_valley_region"], p_valley_region=None)
                                         if truth_fails(v["truth_holds_region"]) and v["p_valley_region"] is not None else v for k, v in r["variants"].items()} for r in recs}
    write_json(Path(out_dir) / "laplace_sensitivity.json", out)
    for key, v in out["per_variant"].items():
        print(f"{key}: narrowest gap 'present' for every wider gap (Matern and TV agree): {v['narrowest_gap_present_for_all_wider']}")
    return out


def null_paths(out_dir, name):
    d = Path(out_dir) / "null"
    return d / f"{name}.json", d / f"{name}.npz"


def stage_null(args):
    """The negative control of the gap-valley feature: the posterior of a CLOSED gap scored with the strips of open gaps.

    Data of two touching disks (``--g`` 0 unless given) at the same SNR, seed, grids and priors as the sweep; MAP and Laplace mass as in ``run_config``
    (``start_and_map``). For every gap ``gs`` of ``--strip-gaps`` and every variant of ``--band-fracs`` x ``--core-fracs`` the strip is that of the open object
    ``TwoDisks(g=gs)`` (centered, same radius and contrast), so the closed-gap posterior is scored on the region where an open gap of width ``gs`` would be:
    ``P_null`` = P(strip < cores - delta) for data that contain no gap. The pixelized truth is scored the same way (``truth_holds``). A feature with
    P_null ~ 1 for a variant does not detect the gap at all for that strip. Writes <out-dir>/null/<name>.json (done marker) and .npz.
    """
    g_true = 0.0 if args.g is None else args.g
    name = config_name(g_true, args.eps_r, args.prior, args.lam_scale)
    jpath, npath = null_paths(args.out_dir, name)
    if jpath.exists() and not args.force:
        print(f"{name}: done ({jpath})")
        return json.loads(jpath.read_text())
    t_all = time.perf_counter()
    problem = build_problem(args, g_true, args.eps_r)
    lam = None
    if args.prior == "tv":
        lam = args.lam if args.lam is not None else lambda_star_of(args.out_dir, args.eps_r) * args.lam_scale
    target = ts.ThetaTarget(problem.forward, build_prior(args, args.prior, problem.grid, lam), problem.d_obs, problem.s)
    theta_matern = None
    if args.prior == "tv":
        th, minfo = matern_map(problem, args)
        theta_matern = th if minfo["map"]["converged"] and minfo["map"]["phi"] < MAP_OK_PHI_OVER_N * problem.n_data else None
    neighbor = None
    if args.neighbor_gap is not None:  # a continuation in g from the stored MAP of the open gap (eps_r = 4: the vacuum and smooth starts end in local minima at small gaps)
        neighbor = load_good_map(args.out_dir, config_name(args.neighbor_gap, args.eps_r, args.prior, args.lam_scale))
    _, theta_hat, minfo = start_and_map(problem, target, args, args.prior, theta_matern=theta_matern, theta_neighbor=neighbor)
    m = minfo["map"]
    map_ok = bool(m["converged"] and m["phi"] < MAP_OK_PHI_OVER_N * problem.n_data)
    print(f"{name}: MAP ({m['chosen_start']} start) {m['nit']} Newton iterations, converged {m['converged']}, Phi {m['phi']:.1f} (N = {problem.n_data}), U {m['U']:.2f}")
    solve = chol_solver(target.laplace_mass(theta_hat).chol)
    variants = {}
    for gs in args.strip_gaps:
        obj = sh.TwoDisks(g=gs, r=args.radius, eps_r=args.eps_r)
        for b in args.band_fracs:
            for c in args.core_fracs:
                fg = gv.feature_geometry(problem.grid, obj, c, b)
                f = gv.gaussian_valley(fg, theta_hat, solve)
                reg = f["region"]
                variants[f"g{gs:g}_{variant_key(b, c)}"] = {
                    "strip_gap": gs, "band_frac": b, "core_frac": c, "region_status": f["region_status"], "n_strip": fg.n_strip, "n_core": fg.n_core1,
                    "p_valley_region": None if reg is None else reg["p_valley"], "strip_mean": None if reg is None else reg["strip_mean"],
                    "strip_sd": None if reg is None else reg["strip_sd"], "core_mean_avg": None if reg is None else 0.5 * (reg["core1_mean"] + reg["core2_mean"]),
                    "depth_mean": None if reg is None else reg["depth_mean"], "depth_sd": None if reg is None else reg["depth_sd"],
                    "delta": fg.delta, "truth_holds_region": gv.truth_valley(fg, problem.theta_true)["region"]["holds"]}
    rec = {"name": name, "g_true": g_true, "eps_r": args.eps_r, "prior": args.prior, "lam": lam, "lam_scale": args.lam_scale if args.prior == "tv" else None,
           "map": {k: v for k, v in m.items() if k != "history"}, "map_ok": map_ok, "phi_over_N": m["phi"] / problem.n_data,
           "map_candidates": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in minfo["candidates"].items()}, "neighbor_start_given": neighbor is not None,
           "problem": {"n": args.n, "data_grid_n": args.n * args.data_factor, "snr_db": args.snr, "seed": args.seed, "noise_std": problem.s, "d_obs_sha256": problem.sha},
           "variants": variants, "time_s": time.perf_counter() - t_all, "thread_env": thread_env(), "load_avg_1min": os.getloadavg()[0]}
    npath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npath, theta_hat=theta_hat, theta_true=problem.theta_true)
    write_json(jpath, rec)  # last: the JSON is the done marker
    print("  P_null(region) " + ", ".join(f"{k} {'-' if v['p_valley_region'] is None else format(v['p_valley_region'], '.3f')}" for k, v in variants.items()
                                          if v["core_frac"] == 0.5) + f"; {rec['time_s']:.0f} s")
    return rec


def stage_null_collect(args):
    """<out-dir>/null.json: per contrast, prior, variant and strip gap, P(valley) for the closed gap (``stage_null``) next to P(valley) for the open gap of the same width
    (the sweep, ``laplace_sensitivity.json``), their difference, and the pixelized-truth indicators; and ``discriminates`` = P_open > 0.95 and P_null < 0.05
    (the feature separates the open gap from no gap for that prior). ``narrowest_discriminating_gap``: per (eps_r, variant), the narrowest strip gap from which
    every wider swept strip gap discriminates under BOTH priors (None if the widest does not)."""
    out_dir = Path(args.out_dir)
    sens = json.loads((out_dir / "laplace_sensitivity.json").read_text())["configurations"]
    nulls = [json.loads(p.read_text()) for p in sorted((out_dir / "null").glob("g*_eps*_*.json"))]
    if not nulls:
        raise SystemExit(f"no null records in {out_dir / 'null'}")
    rows = []
    for r in nulls:
        for key, v in r["variants"].items():
            gs = v["strip_gap"]
            open_name = config_name(gs, r["eps_r"], r["prior"], r["lam_scale"] or 1.0)
            vk = variant_key(v["band_frac"], v["core_frac"])
            o = sens.get(open_name, {}).get(vk)
            p_open = None if o is None else o["p_valley_region"]
            p_null = v["p_valley_region"]
            p_open_raw = p_open if o is None or "p_valley_region_raw" not in o else o["p_valley_region_raw"]
            p_null_raw = p_null
            if o is not None and truth_fails(o["truth_holds_region"]):  # the strip of an open gap that the pixelized truth itself does not show: no probability of that gap
                p_open, p_null = None, None
            rows.append({"eps_r": r["eps_r"], "prior": r["prior"], "lam": r["lam"], "strip_gap": gs, "band_frac": v["band_frac"], "core_frac": v["core_frac"],
                         "variant": vk, "p_null": p_null, "p_open": p_open, "p_null_raw": p_null_raw, "p_open_raw": p_open_raw,
                         "p_difference": None if p_null is None or p_open is None else p_open - p_null,
                         "truth_holds_null": v["truth_holds_region"], "truth_holds_open": None if o is None else o["truth_holds_region"],
                         "strip_mean_null": v["strip_mean"], "strip_mean_open": None if o is None else o["strip_mean"], "map_ok_null": r["map_ok"],
                         "discriminates": None if p_null is None or p_open is None else bool(p_open > 0.95 and p_null < 0.05),
                         "withheld_truth_fails": bool(o is not None and truth_fails(o["truth_holds_region"]))})
    star = {}
    for eps in sorted({x["eps_r"] for x in rows}):
        for vk in sorted({x["variant"] for x in rows}):
            gaps = sorted({x["strip_gap"] for x in rows if x["eps_r"] == eps and x["variant"] == vk})
            g_star = None
            for gs in sorted(gaps, reverse=True):
                cell = [x for x in rows if x["eps_r"] == eps and x["variant"] == vk and x["strip_gap"] == gs]
                if len(cell) < len(PRIORS) or not all(x["discriminates"] for x in cell):
                    break
                g_star = gs
            star[f"eps{eps:g}_{vk}"] = g_star
    out = {"definition": "P_null: Laplace probability of strip < min(cores) - delta for the data of touching disks (g = 0), strip and cores of the open object of the stated "
                         "strip gap; P_open: the same for the data of that open gap (laplace_sensitivity.json); discriminates: P_open > 0.95 and P_null < 0.05",
           "rows": rows, "narrowest_discriminating_gap": star, "records": {r["name"]: {k: r[k] for k in ("g_true", "eps_r", "prior", "lam", "map_ok", "phi_over_N")} for r in nulls}}
    smith = smith_gaps(out_dir)
    out["smith_gap_30db"] = smith
    write_json(out_dir / "null.json", out)
    make_null_figure(rows, out_dir)
    make_overlay_figure(rows, smith, out_dir, sens)
    print(f"wrote {out_dir / 'null.json'} ({len(nulls)} closed-gap posteriors); narrowest discriminating strip gap: {star}")
    return out


def smith_gaps(out_dir, grid=128):
    """The gap at which Smith's criterion sigma_CRB(g) = g holds at the reference SNR, per contrast: log-log interpolation between the two bracketing gaps of the CRB
    records of the extended sweep <out-dir>/smith_limit/crb.json (grid ``grid``; gaps from 1e-4 to 3e-3 wavelengths). None per contrast if the file is missing or
    the criterion is not bracketed."""
    path = Path(out_dir) / "smith_limit" / "crb.json"
    if not path.exists():
        return {}
    recs = json.loads(path.read_text())["records"]
    out = {}
    for eps in EPS:
        pts = sorted((r["params"]["g"], r["crb_g_unknown"]) for r in recs.values() if r["grid"]["n"] == grid and r["params"]["eps_r"] == eps)
        out[f"{eps:g}"] = None
        for (g1, s1), (g2, s2) in zip(pts, pts[1:]):
            f1, f2 = np.log(s1 / g1), np.log(s2 / g2)  # linear in log g where sigma_CRB is a power law of g (flat here), so the interpolation is exact for it
            if f1 > 0.0 >= f2:
                t = f1 / (f1 - f2)
                out[f"{eps:g}"] = float(np.exp(np.log(g1) + t * (np.log(g2) - np.log(g1))))
                break
    return out


def make_overlay_figure(rows, smith, out_dir, sens=None):
    """P(gap valley) against the gap with Smith's criterion overlaid (DESIGN 13.4): per contrast, the strip of half height r/2 (the variant that tells a gap from no gap) for
    the data with the gap (solid; every swept gap when ``sens`` = the "configurations" of laplace_sensitivity.json is given, else the strip gaps of the closed-gap runs) and
    with a closed gap (dashed), both priors, and the full-height strip of the design with a closed gap (dotted, a lenient feature); the shaded band is the range of gaps
    with sigma_CRB(g) <= g at the reference SNR (``smith_gaps``)."""
    eps_values = sorted({r["eps_r"] for r in rows})
    fig = Figure(figsize=(3.7 * len(eps_values) + 0.3, 4.0), dpi=130)
    axes = fig.subplots(1, len(eps_values), squeeze=False, sharey=True)[0]
    for ax, eps in zip(axes, eps_values):
        g_s = smith.get(f"{eps:g}")
        if g_s is not None:
            ax.axvspan(g_s, 0.7, color=EPS_COLORS.get(eps, "#2a78d6"), alpha=0.10, lw=0)
            ax.axvline(g_s, color=EPS_COLORS.get(eps, "#2a78d6"), lw=1.0)
            ax.text(g_s * 1.25, 0.5, f"Smith: sigma_CRB(g) <= g\nfor g >= {g_s:.1e}", fontsize=6.5, color=MUTED, va="center", rotation=90)
        for prior in PRIORS:
            lab = "Matern" if prior == "matern" else "TV"
            sel = lambda b: sorted((r for r in rows if r["eps_r"] == eps and r["prior"] == prior and r["band_frac"] == b and r["core_frac"] == 0.5), key=lambda r: r["strip_gap"])
            half, full = sel(0.5), sel(1.0)
            if sens is not None:
                gs = [g for g in GAPS if sens.get(config_name(g, eps, prior), {}).get(variant_key(0.5, 0.5), {}).get("p_valley_region") is not None]
                po = [sens[config_name(g, eps, prior)][variant_key(0.5, 0.5)]["p_valley_region"] for g in gs]
            else:
                gs, po = [r["strip_gap"] for r in half], [np.nan if r["p_open"] is None else r["p_open"] for r in half]
            ax.plot(gs, po, color=COLORS[prior], lw=1.8, marker="o", ms=4, label=f"{lab}, strip r/2, gap present")
            ax.plot([r["strip_gap"] for r in half], [np.nan if r["p_null"] is None else r["p_null"] for r in half], color=COLORS[prior], lw=1.4, ls="--", marker="s", ms=3.5, label=f"{lab}, strip r/2, closed gap")
            ax.plot([r["strip_gap"] for r in full], [np.nan if r["p_null"] is None else r["p_null"] for r in full], color=COLORS[prior], lw=1.0, ls=":", label=f"{lab}, full strip, closed gap")
        ax.axhline(0.95, color=MUTED, lw=0.7, ls=":")
        ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(8e-5, 0.7)
        ax.set_ylim(-0.04, 1.04)
        style(ax, "gap g (wavelengths)", "P(gap valley)" if eps == eps_values[0] else "")
        ax.set_title(f"eps_r = {eps:g}", color=INK, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(Path(out_dir) / "resolution_overlay.png")


def make_null_figure(rows, out_dir):
    """Rows: strip heights (band_frac); columns: contrasts. P(valley) against the strip gap for data with the gap (solid) and with a closed gap (dashed), both priors."""
    eps_values = sorted({r["eps_r"] for r in rows})
    bands = sorted({r["band_frac"] for r in rows}, reverse=True)
    fig = Figure(figsize=(3.4 * len(eps_values) + 0.3, 2.3 * len(bands) + 0.7), dpi=130)
    axes = fig.subplots(len(bands), len(eps_values), squeeze=False, sharex=True, sharey=True)
    for i, b in enumerate(bands):
        for j, eps in enumerate(eps_values):
            ax = axes[i, j]
            for prior in PRIORS:
                rr = sorted((r for r in rows if r["eps_r"] == eps and r["band_frac"] == b and r["core_frac"] == 0.5 and r["prior"] == prior), key=lambda r: r["strip_gap"])
                g = [r["strip_gap"] for r in rr]
                lab = "Matern" if prior == "matern" else "TV"
                ax.plot(g, [np.nan if r["p_open"] is None else r["p_open"] for r in rr], color=COLORS[prior], lw=1.8, marker="o", ms=4, label=f"{lab}, gap present in the data")
                ax.plot(g, [np.nan if r["p_null"] is None else r["p_null"] for r in rr], color=COLORS[prior], lw=1.4, ls="--", marker="s", ms=3.5, label=f"{lab}, closed gap in the data")
            ax.axhline(0.95, color=MUTED, lw=0.7, ls=":")
            ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
            ax.set_ylim(-0.04, 1.04)
            style(ax, "strip gap (wavelengths)" if i == len(bands) - 1 else "", f"P(valley), strip height {b:g} r" if j == 0 else "")
            gap_axis(ax, gaps=(0.025, 0.05, 0.1, 0.15))
            if i == 0:
                ax.set_title(f"eps_r = {eps:g}", color=INK, fontsize=10, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(Path(out_dir) / "null_control.png")


def classify(p_a, p_b, hi=0.95, lo=0.05):
    """The DESIGN 11.3 rule for two priors: "present" if both probabilities exceed ``hi``, "absent" if both are below ``lo``, else "prior-dependent"
    (the data do not determine the feature: the two priors give different answers or an intermediate one). None in either argument (an unrepresentable gap) gives
    "unrepresentable"."""
    if p_a is None or p_b is None:
        return "unrepresentable"
    if p_a > hi and p_b > hi:
        return "present"
    if p_a < lo and p_b < lo:
        return "absent"
    return "prior-dependent"


def verdict_table(rows):
    """Per (eps_r, g): the Matern and TV(lambda*) region probabilities and ``classify`` of them, plus per contrast the narrowest gap g* from which every wider
    gap of the sweep is "present" (None if the widest gap is not)."""
    base = {(r["eps_r"], r["g"], r["prior"]): r for r in rows if r["lam_scale"] in (None, 1.0) and r["map_ok"]}
    table, star = [], {}
    for eps in sorted({r["eps_r"] for r in rows}):
        gaps = sorted({r["g"] for r in rows if r["eps_r"] == eps})
        for g in gaps:
            m, t = base.get((eps, g, "matern")), base.get((eps, g, "tv"))
            if m is None or t is None:
                continue
            pm, pt = m["p_valley_region"], t["p_valley_region"]
            table.append({"eps_r": eps, "g": g, "p_matern": pm, "p_tv": pt, "p_point_matern": m["p_valley_point"], "p_point_tv": t["p_valley_point"],
                          "verdict": classify(pm, pt), "verdict_point": classify(m["p_valley_point"], t["p_valley_point"])})
        ok = [r for r in table if r["eps_r"] == eps]
        g_star = None
        for r in sorted(ok, key=lambda r: -r["g"]):
            if r["verdict"] != "present":
                break
            g_star = r["g"]
        star[f"{eps:g}"] = g_star
    return table, star


def stage_collect(args):
    out_dir = Path(args.out_dir)
    recs = {}
    for p in sorted((out_dir / "laplace").glob("g*_eps*_*.json")):
        r = json.loads(p.read_text())
        recs[r["name"]] = r
    if not recs:
        raise SystemExit(f"no configuration records in {out_dir / 'laplace'}")
    rows = [table_row(r) for r in sorted(recs.values(), key=lambda r: (r["prior"], r.get("lam_scale") or 1.0, r["eps_r"], r["g"]))]
    lam_info = json.loads((out_dir / "laplace_lambda.json").read_text()) if (out_dir / "laplace_lambda.json").exists() else None
    crb = json.loads((out_dir / "crb.json").read_text()) if (out_dir / "crb.json").exists() else None
    out = {"setup": {"n": args.n, "side": args.side, "data_grid_n": args.n * args.data_factor, "radius": args.radius, "snr_db": args.snr, "seed": args.seed,
                     "priors": {"matern": {"sigma": args.sigma, "ell": args.ell, "nu": args.nu, "form": "cropped Gaussian field (DESIGN 3)"},
                                "tv": {"delta": args.delta_tv, "sigma_0": args.sigma_0, "lam": "lambda*(eps_r) of laplace_lambda.json (x lam_scale for sensitivity runs)"}},
                     "feature": "P(mean theta over the gap strip < min(mean theta over the two disk cores) - delta), delta = log(eps_r) / 2; strip = pixel centers "
                                "between the facing disk edges within the disk-height band, core = pixel centers within core_frac x radius of a disk center; "
                                "'unrepresentable' if g < h", "core_frac": args.core_frac,
                     "posterior": "Laplace: N(MAP, (H_GN + Hess R)^-1) in theta = log eps_r, exact bivariate-normal orthant probability"},
           "lambda": lam_info, "table": rows, "records": recs}
    verdicts, g_star = verdict_table(rows)
    out["verdicts"] = {"rule": "present: P > 0.95 under both Matern and TV(lambda*); absent: P < 0.05 under both; else prior-dependent (DESIGN 11.3, 8.7)",
                       "table": verdicts, "narrowest_gap_present_for_all_wider": g_star}
    write_json(out_dir / "laplace.json", out)
    make_figures(rows, recs, crb, args, out_dir)
    make_verdict_figure(verdicts, args, out_dir)
    print(f"wrote {out_dir / 'laplace.json'} ({len(recs)} configurations); narrowest gap with verdict 'present' for all wider gaps: {g_star}")
    return out


def make_figures(rows, recs, crb, args, out_dir):
    eps_values = sorted({r["eps_r"] for r in rows})
    base = [r for r in rows if r["prior"] in PRIORS and (r["lam_scale"] in (None, 1.0)) and r["map_ok"]]  # a failed MAP carries no probability
    # 1. P(gap valley) against g, both priors, with the CRB / Smith overlay in the row below
    fig = Figure(figsize=(3.7 * len(eps_values) + 0.3, 6.2), dpi=130)
    axes = fig.subplots(2, len(eps_values), squeeze=False, sharex="col", gridspec_kw={"height_ratios": [2.2, 1.2]})
    for j, eps in enumerate(eps_values):
        ax = axes[0, j]
        for prior in PRIORS:
            rr = sorted((r for r in base if r["eps_r"] == eps and r["prior"] == prior), key=lambda r: r["g"])
            if not rr:
                continue
            g = np.array([r["g"] for r in rr])
            reg = np.array([np.nan if r["p_valley_region"] is None else r["p_valley_region"] for r in rr])
            pt = np.array([r["p_valley_point"] for r in rr])
            ax.plot(g, reg, color=COLORS[prior], lw=1.8, marker="o", ms=4, label=f"{'Matern' if prior == 'matern' else 'TV'}, strip vs cores")
            ax.plot(g, pt, color=COLORS[prior], lw=1.0, ls="--", marker="s", ms=3, alpha=0.8, label=f"{'Matern' if prior == 'matern' else 'TV'}, single points")
            bad = [r["g"] for r in rr if r["p_valley_region"] is None]
            ax.plot(bad, [0.0] * len(bad), ls="none", marker="x", ms=7, color=COLORS[prior], mew=1.6)
        ax.axhline(0.95, color=MUTED, lw=0.7, ls=":")
        ax.axhline(0.05, color=MUTED, lw=0.7, ls=":")
        ax.set_ylim(-0.04, 1.04)
        style(ax, "", "P(gap valley)" if j == 0 else "")
        ax.set_title(f"eps_r = {eps:g}", color=INK, fontsize=10, loc="left")
        if j == 0:
            ax.legend(frameon=False, fontsize=7, loc="center right")
            ax.text(0.0135, 0.02, "x: unrepresentable (g < h or truth fails)", fontsize=7, color=MUTED, va="bottom")
        ax2 = axes[1, j]
        if crb is not None:
            rr = sorted((r for r in crb["table"] if r["eps_r"] == eps), key=lambda r: r["g"])
            ax2.plot([r["g"] for r in rr], [r["required_snr_db"]["unknown"] for r in rr], color=EPS_COLORS.get(eps, "#2a78d6"), lw=1.8, marker="o", ms=3)
            ax2.fill_between([r["g"] for r in rr], [r["required_snr_db"]["unknown"] for r in rr], args.snr, color=EPS_COLORS.get(eps, "#2a78d6"), alpha=0.12, lw=0)
            ax2.axhline(args.snr, color=MUTED, lw=0.9)
            ax2.text(0.0135, args.snr - 2.0, f"data SNR {args.snr:g} dB; shaded: margin of Smith's criterion", color=MUTED, fontsize=7, va="top")
            ax2.set_ylim(-42, 36)
        style(ax2, "gap g (wavelengths)", "SNR with sigma_CRB(g) = g (dB)" if j == 0 else "")
        gap_axis(ax2)
    fig.tight_layout()
    fig.savefig(out_dir / "laplace_pvalley.png")
    # 2. maps and 3. profiles at eps_r = 2 (or the first contrast) for selected gaps
    eps_show = 2.0 if 2.0 in eps_values else eps_values[0]
    gaps_show = [g for g in (0.025, 0.05, 0.1, 0.25) if all(f"g{g:g}_eps{eps_show:g}_{p}" in recs for p in PRIORS)]
    if gaps_show:
        make_maps(recs, eps_show, gaps_show, args, out_dir)
        make_profiles(recs, eps_show, gaps_show, args, out_dir)


def make_verdict_figure(verdicts, args, out_dir):
    """Grid of eps_r (rows) by gap (columns): the verdict of ``classify`` for the strip-against-cores probability, with both probabilities printed."""
    if not verdicts:
        return
    eps_values = sorted({v["eps_r"] for v in verdicts}, reverse=True)
    gaps = sorted({v["g"] for v in verdicts})
    colors = {"present": "#2a78d6", "absent": "#eb6834", "prior-dependent": "#c9c8c0", "unrepresentable": "#ffffff"}
    fig = Figure(figsize=(1.0 * len(gaps) + 1.6, 0.9 * len(eps_values) + 1.7), dpi=130)
    ax = fig.subplots()
    for i, eps in enumerate(eps_values):
        for j, g in enumerate(gaps):
            v = next((x for x in verdicts if x["eps_r"] == eps and x["g"] == g), None)
            if v is None:
                continue
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=colors[v["verdict"]], edgecolor="#fcfcfb", lw=2, alpha=0.9 if v["verdict"] != "prior-dependent" else 1.0,
                                   hatch="///" if v["verdict"] == "unrepresentable" else None))
            fm = lambda p: "-" if p is None else f"{p:.2f}"
            ax.text(j + 0.5, i + 0.5, f"M {fm(v['p_matern'])}\nTV {fm(v['p_tv'])}", ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if v["verdict"] in ("present", "absent") else INK)
    ax.set_xlim(0, len(gaps))
    ax.set_ylim(0, len(eps_values))
    ax.set_xticks([j + 0.5 for j in range(len(gaps))])
    ax.set_xticklabels([f"{g:g}" for g in gaps], fontsize=8, color=MUTED)
    ax.set_yticks([i + 0.5 for i in range(len(eps_values))])
    ax.set_yticklabels([f"{e:g}" for e in eps_values], fontsize=9, color=MUTED)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("gap g (wavelengths)", color=INK, fontsize=10)
    ax.set_ylabel("eps_r", color=INK, fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[k], edgecolor=MUTED, lw=0.5, hatch="///" if k == "unrepresentable" else None) for k in colors]
    ax.legend(handles, ["present: P > 0.95 under both priors", "absent: P < 0.05 under both", "prior-dependent", "unrepresentable (g < pixel, or the truth fails)"], frameon=False, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "laplace_verdict.png")


def _load_npz(out_dir, name):
    return np.load(config_paths(out_dir, name)[1])


def make_maps(recs, eps, gaps, args, out_dir):
    cols = ["truth", "Matern MAP", "TV MAP", "Matern Laplace sd", "TV Laplace sd"]
    fig = Figure(figsize=(2.9 * len(cols), 1.9 * len(gaps) + 0.5), dpi=120)
    axes = fig.subplots(len(gaps), len(cols), squeeze=False)
    half = 0.5 * args.side
    xlim, ylim = (-0.72, 0.72), (-0.45, 0.45)
    vmax = float(np.log(eps)) * 1.25
    for i, g in enumerate(gaps):
        zm, zt = _load_npz(out_dir, f"g{g:g}_eps{eps:g}_matern"), _load_npz(out_dir, f"g{g:g}_eps{eps:g}_tv")
        ims = [zm["theta_true"], zm["theta_hat"], zt["theta_hat"], zm["laplace_std"], zt["laplace_std"]]
        for j, (ax, im) in enumerate(zip(axes[i], ims)):
            sd = j >= 3
            art = ax.imshow(np.asarray(im, dtype=float).T, origin="lower", extent=(-half, half, -half, half), cmap="magma" if sd else "viridis",
                            vmin=0.0 if sd else -0.1 * vmax, vmax=0.2 if sd else vmax, interpolation="nearest")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"g = {g:g}", fontsize=9, color=INK)
    fig.suptitle(f"theta = log eps_r, eps_r = {eps:g}: truth on the n = {args.n} grid, MAPs and Laplace standard deviations (sd color scale 0 to 0.2)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "laplace_maps.png")


def make_profiles(recs, eps, gaps, args, out_dir):
    fig = Figure(figsize=(3.3 * len(gaps), 2.9), dpi=130)
    axes = fig.subplots(1, len(gaps), squeeze=False, sharey=True)[0]
    x = (np.arange(args.n) + 0.5) * (args.side / args.n) - 0.5 * args.side
    j0 = args.n // 2  # the two rows nearest y = 0
    for ax, g in zip(axes, gaps):
        zm, zt = _load_npz(out_dir, f"g{g:g}_eps{eps:g}_matern"), _load_npz(out_dir, f"g{g:g}_eps{eps:g}_tv")
        prof = lambda a: 0.5 * (np.asarray(a)[:, j0 - 1] + np.asarray(a)[:, j0])
        ax.plot(x, prof(zm["theta_true"]), color=COLORS["truth"], lw=1.2, drawstyle="steps-mid", label="truth (n = 64 rendering)")
        for z, key in ((zm, "matern"), (zt, "tv")):
            ax.plot(x, prof(z["theta_hat"]), color=COLORS[key], lw=1.6, label="Matern MAP" if key == "matern" else "TV MAP")
        ax.axvspan(-0.5 * g, 0.5 * g, color="#e4e3dd", alpha=0.7, lw=0)
        ax.set_xlim(-0.55, 0.55)
        style(ax, "x (wavelengths)", "theta along y = 0" if ax is axes[0] else "")
        ax.set_title(f"g = {g:g} ({g / (args.side / args.n):.1f} pixels)", color=INK, fontsize=9, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_dir / "laplace_profiles.png")


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "config":
        return stage_config(args)
    if args.stage == "descend":
        return stage_descend(args)
    if args.stage == "lambda":
        return stage_lambda(args)
    if args.stage == "lambda-collect":
        return stage_lambda_collect(args)
    if args.stage == "refeature":
        return stage_refeature(args)
    if args.stage == "null":
        return stage_null(args)
    if args.stage == "null-collect":
        return stage_null_collect(args)
    return stage_collect(args)


if __name__ == "__main__":
    main()
