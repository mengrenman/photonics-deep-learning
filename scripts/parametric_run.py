"""Milestone 7 (DESIGN 14): one truth x one model per call. Shape-space posteriors of F2, F2u and Fe (or the pixel posteriors) on the data of one truth.

    python scripts/parametric_run.py --truth T1 --gap 0.05 --model F2                    # prepare, map, chains, ppc, finalize
    python scripts/parametric_run.py --truth T1 --gap 0.05 --model F2 --stage ti         # thermodynamic integration (long)
    python scripts/parametric_run.py --stage cost                                        # per-evaluation cost at n = 64 and the budget estimate

Case. ``--truth`` T1..T5 (``ellipses.make_truth``: T1 two equal disks, T2 two ellipses, T3 unequal permittivities, T4 two touching disks and a bridge, T5 T1 plus a
third disk), ``--gap`` the edge-to-edge gap (T4 ignores it), ``--model`` F2 / F2u / Fe (``shapemodels``) or ``pixel`` (the milestone-6 Laplace pixel posteriors of the
same data under the Matern and the TV prior at lambda*, scored with the half-height gap-valley strip). A case writes to
<out-dir>/<truth label>_<model>/ with the truth label T1g0.05, T2g0.1, T3g0.05, T4, T5 (T5 at its default gap 0.1; another gap adds a g suffix). The data of a truth
are generated ONCE (the data grid is ``--data-factor`` times finer than the inference grid, so no inverse crime; 30 dB; one noise key, ``--seed``) and cached in
<out-dir>/data/<truth label>.npz, so the three models and the pixel posteriors fit literally the same numbers.

Stages (--stage; ``all`` = prepare, map, chain, ppc, finalize; every stage skips work whose output exists, ``--force`` redoes it):

  prepare   the data of the truth and the problem record (<dir>/problem.json): noise level, N = Nt Nr, the operating point (the grid error of the inference grid at the
            truth against the noise, Phi at the truth).
  map       the MAP in the unconstrained space from the prior mode, ``--n-prior-starts`` prior draws and, for F2u and Fe, the F2 MAP of the same data (nested starts, if
            <truth>_F2/map.npz exists); then from the symmetry images of the best mode (``ShapeModel.symmetry_images``: mirror, exchange of the disks' properties). L-BFGS-B then
            damped Gauss-Newton from each start (``shapepost.map_estimate``); the Laplace fit (Fisher + prior Hessian, and the finite-difference Hessian), the Laplace log
            evidence; <dir>/map.json and map.npz.
  chain     ``--chains`` chains of ``--segments`` segments of ``--segment-draws`` recorded draws (segment 0 adds ``--warmup`` steps of step-size adaptation from an
            overdispersed start, ``--init-scale`` x N(0, I) in the Laplace coordinates; each later segment continues from the final state with the frozen step), one npz per
            (chain, segment) in <dir>/chains/, written as soon as it is done; a rerun continues from the last one. ``--chain-id`` selects one chain (one process per chain under
            ``scripts/batch_run.py``), ``--first-segment`` / ``--last-segment`` a range of segments.
  ppc       the posterior-predictive checks (``shapepost.predictive_check``, refit and draw modes) from the pooled draws: <dir>/ppc.json.
  ti        thermodynamic integration (``shapepost.ti_temperature``), beta_k = (k / K)^``--ti-power``, ``--ti-temps`` temperatures, ``--ti-chains`` chains each; one file per
            temperature in <dir>/ti/ (a rerun continues at the first missing one), then <dir>/ti.json (``ti_combine``). ``--ti-tag a --ti-seed 3 --ti-chains 1`` is an
            independent run in <dir>/ti_a/ and ti_a.json (one chain per process: four such runs take the wall time of one chain); ``finalize`` combines all the runs of a case
            (``combine_ti_runs``: the mean, and the between-run standard deviation as the direct error bar).
  finalize  <dir>/summary.json: the gap (mean, 90 % interval, coverage of the true gap, P(g < g_det)), the MAP misfit against N and against Phi at the truth, the
            convergence report, the multimodality report, the evidence (Laplace, and TI if run), the predictive check, the cost.
  pixel     (``--model pixel``) the Laplace pixel posteriors, <dir>/summary.json.
  cost      per-evaluation cost (value + gradient) of the three models at n = 64 with the book rings, and the budget estimate for the DESIGN 14.3 plan:
            <out-dir>/cost.json.

Threads: one XLA thread per process under ``scripts/batch_run.py --threads 1`` (at most 10 XLA threads in total on the Mac).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import ellipses as el
from bayes_scatter import gapvalley as gv
from bayes_scatter import inference as inf
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter.adapt import AffineMap
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parametric"
print = functools.partial(print, flush=True)

MODELS = ("F2", "F2u", "Fe")
STAGES = ("all", "prepare", "map", "chain", "ppc", "ti", "finalize", "pixel", "cost")
# the DESIGN 14.3 plan: (truth, gap) pairs
PLAN_TRUTHS = (("T1", 0.05), ("T1", 0.1), ("T2", 0.05), ("T2", 0.1), ("T3", 0.05), ("T3", 0.1), ("T4", 0.0), ("T5", 0.1))
DETECTION_JSON = ROOT / "results" / "resolution" / "detection.json"


# ----------------------------------------------------------------------------------------------
# Arguments, names, files
# ----------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, default="all")
    g = ap.add_argument_group("case")
    g.add_argument("--truth", choices=el.TRUTH_NAMES)
    g.add_argument("--gap", type=float, default=0.1, help="edge-to-edge gap of the truth (T4 ignores it)")
    g.add_argument("--model", choices=MODELS + ("pixel",))
    g.add_argument("--eps-r", type=float, default=2.0)
    g.add_argument("--radius", type=float, default=0.2)
    g = ap.add_argument_group("problem")
    g.add_argument("--n", type=int, default=64, help="inference grid")
    g.add_argument("--data-factor", type=int, default=2, help="data grid = this many times finer than --n (128)")
    g.add_argument("--side", type=float, default=1.5)
    g.add_argument("--snr", type=float, default=30.0)
    g.add_argument("--seed", type=int, default=0, help="noise key of the data")
    g.add_argument("--n-tx", type=int, default=40)
    g.add_argument("--n-rx", type=int, default=60)
    g.add_argument("--tol", type=float, default=1e-10, help="Krylov tolerance of the inference forward model (DESIGN 2.4: <= 1e-10 for sampling)")
    g = ap.add_argument_group("map")
    g.add_argument("--n-prior-starts", type=int, default=3)
    g.add_argument("--map-seed", type=int, default=0)
    g.add_argument("--lbfgs-iter", type=int, default=300)
    g.add_argument("--polish-iter", type=int, default=25)
    g = ap.add_argument_group("chains")
    g.add_argument("--chains", type=int, default=4)
    g.add_argument("--chain-id", type=int, default=None, help="run only this chain (default all)")
    g.add_argument("--warmup", type=int, default=150, help="step-size adaptation steps of segment 0")
    g.add_argument("--segment-draws", type=int, default=100)
    g.add_argument("--segments", type=int, default=5)
    g.add_argument("--first-segment", type=int, default=0)
    g.add_argument("--last-segment", type=int, default=None)
    g.add_argument("--n-leapfrog", type=int, default=8)
    g.add_argument("--jitter", type=float, default=0.25)
    g.add_argument("--init-scale", type=float, default=2.0, help="overdispersion of the starts (DESIGN 11.2: 2)")
    g.add_argument("--chain-seed", type=int, default=1)
    g = ap.add_argument_group("thermodynamic integration")
    g.add_argument("--ti-temps", type=int, default=16, help="number of temperatures (the design asks for at least 12)")
    g.add_argument("--ti-power", type=float, default=5.0)
    g.add_argument("--ti-chains", type=int, default=4)
    g.add_argument("--ti-pilot", type=int, default=40)
    g.add_argument("--ti-warmup", type=int, default=60)
    g.add_argument("--ti-draws", type=int, default=150)
    g.add_argument("--ti-leapfrog", type=int, default=8)
    g.add_argument("--ti-seed", type=int, default=2)
    g.add_argument("--ti-tag", default="", help="name of this TI run: files ti_<tag>/ and ti_<tag>.json instead of ti/ and ti.json, so that independent runs (other --ti-seed, e.g. "
                   "one chain each, one process each) can go side by side; the summary combines all the runs it finds")
    g = ap.add_argument_group("predictive check")
    g.add_argument("--ppc-reps", type=int, default=100)
    g.add_argument("--ppc-seed", type=int, default=3)
    g = ap.add_argument_group("pixel posteriors and cost")
    g.add_argument("--sigma", type=float, default=0.5)
    g.add_argument("--ell", type=float, default=0.4)
    g.add_argument("--nu", type=float, default=1.5)
    g.add_argument("--delta-tv", type=float, default=0.01)
    g.add_argument("--sigma-0", type=float, default=1.0)
    g.add_argument("--start-maxiter", type=int, default=100)
    g.add_argument("--newton-iter", type=int, default=60)
    g.add_argument("--newton-iter-smooth", type=int, default=40)
    g.add_argument("--newton-tol", type=float, default=1e-3)
    g.add_argument("--all-starts", action="store_true")
    g.add_argument("--core-frac", type=float, default=gv.CORE_FRAC)
    g.add_argument("--band-frac", type=float, default=gv.BAND_FRAC, help="strip half-height as a fraction of the radius (decision of 2026-10-13: 0.5)")
    g.add_argument("--lambda-dir", default=str(ROOT / "results" / "resolution"), help="where laplace_lambda.json (lambda* of milestone 6) is read")
    g.add_argument("--cost-evals", type=int, default=8, help="cost: timed evaluations per kind")
    g.add_argument("--budget-ti-models", nargs="+", default=["F2", "Fe"], help="cost: the models of the cases of the plan that get a thermodynamic integration (one case each)")
    g.add_argument("--budget-slots", type=int, nargs="+", default=[5, 10], help="cost: single-thread processes for the wall-clock estimate")
    g = ap.add_argument_group("output")
    g.add_argument("--out-dir", default=str(RESULTS))
    g.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.stage != "cost" and (args.truth is None or args.model is None):
        ap.error("--truth and --model are required (except for --stage cost)")
    if args.model == "pixel" and args.stage not in ("all", "pixel", "prepare"):
        ap.error("--model pixel runs the stages prepare / pixel (or all)")
    if args.ti_temps < 12 and args.stage == "ti":
        ap.error("the design asks for at least 12 temperatures")
    if args.last_segment is None:
        args.last_segment = args.segments - 1
    return args


def truth_label(name, gap):
    """T1g0.05, T2g0.1, T3g0.05; T4 (the gap is ignored) and T5 at its default gap 0.1 have no suffix."""
    if name == "T4" or (name == "T5" and abs(gap - 0.1) < 1e-12):
        return name
    return f"{name}g{gap:g}"


def case_name(args):
    return f"{truth_label(args.truth, args.gap)}_{args.model}"


def case_dir(args):
    return Path(args.out_dir) / case_name(args)


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    tmp.replace(path)  # an interrupted write never leaves a half-written file that would count as done


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def thread_env():
    return {k: os.environ.get(k) for k in ("NPROC", "XLA_FLAGS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_PLATFORMS")}


def g_detection(eps_r):
    """The gap detectable at 30 dB against g = 0 (level 0.05, power 0.9; milestone 6, ``detection.json``) for the contrast nearest to ``eps_r``; None if the file is absent."""
    if not DETECTION_JSON.exists():
        return None
    rows = json.loads(DETECTION_JSON.read_text())["at_zero"]
    row = min(rows, key=lambda r: abs(r["eps_r"] - eps_r))
    return float(row["detection_limit_g"])


# ----------------------------------------------------------------------------------------------
# The problem: truth, data, inference forward model
# ----------------------------------------------------------------------------------------------


def data_path(args):
    return Path(args.out_dir) / "data" / f"{truth_label(args.truth, args.gap)}.npz"


def build_problem(args):
    """Truth, data and the inference-grid forward model; returns a namespace.

    Fields: ``truth`` (a renderer with ``chi(grid)``), ``g_true`` (the true edge-to-edge gap), ``setup``, ``grid``, ``forward`` (tolerance ``--tol``), ``renderer``
    (``ellipses.EllipseRenderer`` of the inference grid), ``chi_true`` / ``theta_true`` ((n, n), the exact-coverage rendering on the inference grid), ``d_clean`` / ``d_obs``
    ((Nt, Nr)), ``s`` (noise level at ``--snr``), ``sha`` (hash of d_obs), ``n_data`` (N = Nt Nr), ``operating_point`` (``inference.operating_point``), ``g_det``.
    The data come from the cache of the truth if it exists (all models fit the same numbers), else from the data grid ``n * data_factor`` (tolerance 1e-10) with the noise key
    ``--seed``, and are cached.
    """
    truth = el.make_truth(args.truth, g=args.gap, eps_r=args.eps_r, r=args.radius)
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    grid = setup.grid
    path = data_path(args)
    if path.exists() and not args.force:
        z = np.load(path)
        d_clean, d_obs, s = jnp.asarray(z["d_clean"]), jnp.asarray(z["d_obs"]), float(z["s"])
        if d_obs.shape != (args.n_tx, args.n_rx):
            raise SystemExit(f"{path} holds data of shape {d_obs.shape}, not {(args.n_tx, args.n_rx)}: use --force or another --out-dir")
    else:
        data_setup = Setup(Grid(args.n * args.data_factor, args.side), setup.antennas)
        d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(truth.chi(data_setup.grid)))
        s = noise_std_from_snr(d_clean, args.snr)
        d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
        save_npz(path, d_clean=np.asarray(d_clean), d_obs=np.asarray(d_obs), s=s, seed=args.seed, snr=args.snr)
    forward = Forward(setup, tol=args.tol)
    chi_true = truth.chi(grid)
    sha = hashlib.sha256(np.ascontiguousarray(np.asarray(d_obs, dtype=np.complex128)).tobytes()).hexdigest()
    g_true = getattr(truth, "true_gap", None)
    if g_true is None:
        g_true = getattr(truth, "g", None)
    return SimpleNamespace(truth=truth, g_true=g_true, setup=setup, grid=grid, forward=forward, renderer=el.EllipseRenderer(grid), chi_true=chi_true,
                           theta_true=np.log1p(chi_true), d_clean=d_clean, d_obs=d_obs, s=float(s), sha=sha, n_data=args.n_tx * args.n_rx,
                           operating_point=inf.operating_point(forward, jnp.asarray(chi_true), d_clean, d_obs, float(s)), g_det=g_detection(args.eps_r))


def make_posterior(args, prob):
    return sp.ShapePosterior(sm.make_model(args.model), prob.forward, prob.d_obs, prob.s, prob.renderer)


def problem_record(args, prob):
    return {"truth": prob.truth.describe() if hasattr(prob.truth, "describe") else {"name": args.truth, "g": prob.truth.g, "r": prob.truth.r, "eps_r": prob.truth.eps_r},
            "truth_label": truth_label(args.truth, args.gap), "g_true": prob.g_true, "g_det": prob.g_det, "n": args.n, "side": args.side, "h": prob.grid.h,
            "data_grid_n": args.n * args.data_factor, "snr_db": args.snr, "seed": args.seed, "n_tx": args.n_tx, "n_rx": args.n_rx, "n_data": prob.n_data,
            "noise_std": prob.s, "d_obs_sha256": prob.sha, "tol": args.tol, "operating_point": prob.operating_point, "thread_env": thread_env()}


def stage_prepare(args):
    path = case_dir(args) / "problem.json"
    if path.exists() and not args.force:
        return json.loads(path.read_text())
    prob = build_problem(args)
    rec = problem_record(args, prob)
    write_json(path, rec)
    op = rec["operating_point"]
    print(f"{case_name(args)}: N = {prob.n_data}, s = {prob.s:.3e}; grid error of the model at the truth {op['model_to_noise']:.3f} of the noise, Phi(truth) = {op['phi_truth']:.0f}")
    return rec


# ----------------------------------------------------------------------------------------------
# MAP and Laplace
# ----------------------------------------------------------------------------------------------


def sibling_start(args, model):
    """Extra starts from the F2 MAP of the same truth (the nested models F2u and Fe), as ``(label, u)`` pairs; empty if the F2 map does not exist yet."""
    if args.model == "F2":
        return []
    path = Path(args.out_dir) / f"{truth_label(args.truth, args.gap)}_F2" / "map.npz"
    if not path.exists():
        return []
    f2 = sm.F2()
    p = np.asarray(f2.p_from_u(jnp.asarray(np.load(path)["u_hat"])))
    q = f2.to_f2u(p) if args.model == "F2u" else f2.to_fe(p)
    return [("from F2 MAP", np.asarray(model.u_from_p(jnp.asarray(q))))]


def stage_map(args):
    d = case_dir(args)
    if (d / "map.json").exists() and not args.force:
        print(f"{case_name(args)}: map done")
        return json.loads((d / "map.json").read_text())
    t_all = time.perf_counter()
    prob = build_problem(args)
    model = sm.make_model(args.model)
    post = make_posterior(args, prob)
    rng = np.random.default_rng(args.map_seed)
    starts = [("prior mode", np.zeros(model.dim))]
    starts += [(f"prior draw {k}", u) for k, u in enumerate(model.start_points_u(rng, args.n_prior_starts))]
    starts += sibling_start(args, model)
    print(f"{case_name(args)}: MAP from {len(starts)} starts")
    search = sp.map_estimate(post, np.stack([u for _, u in starts]), [lab for lab, _ in starts], lbfgs_iter=args.lbfgs_iter, polish_iter=args.polish_iter, verbose=True)
    p_best = post.physical(search.u_hat)
    images = model.symmetry_images(p_best)
    if images:
        print(f"  symmetry images of the best mode: {', '.join(lab for lab, _ in images)}")
        s2 = sp.map_estimate(post, np.stack([np.asarray(model.u_from_p(jnp.asarray(q))) for _, q in images]), [f"image: {lab}" for lab, _ in images],
                             lbfgs_iter=args.lbfgs_iter, polish_iter=args.polish_iter, verbose=True)
        cands = search.candidates + s2.candidates
        ok = [i for i, c in enumerate(cands) if c["converged"]] or list(range(len(cands)))
        search = sp.MapSearch(cands, min(ok, key=lambda i: cands[i]["U"]))
        search.u_hat = cands[search.best]["u"]
    best = search.best_candidate
    fit = sp.laplace_fit(post, search.u_hat)
    ev = {"laplace_gn": sp.laplace_log_evidence(post, fit)}
    fit_fd = None
    try:
        fit_fd = sp.laplace_fit(post, search.u_hat, hessian="fd")
        ev["laplace_fd"] = sp.laplace_log_evidence(post, fit_fd)
    except ValueError as err:
        ev["laplace_fd"] = None
        ev["laplace_fd_error"] = str(err)
    p_hat = post.physical(search.u_hat)
    rec = {"case": case_name(args), "model": model.describe(), "param_names": list(model.param_names), "u_hat": search.u_hat, "p_hat": p_hat,
           "candidates": search.summary(), "best": search.best, "phi_map": best["phi"], "n_data": prob.n_data, "phi_over_N": best["phi"] / prob.n_data,
           "phi_truth": prob.operating_point["phi_truth"], "U_map": best["U"], "map_converged": best["converged"], "grad_inf": best["grad_inf"],
           "u_gap": _u_gap(search.candidates), "laplace": {"sd_u": fit.sd(), "logdet_gn": fit.logdet, "logdet_fd": None if fit_fd is None else fit_fd.logdet,
                                                           "fisher_cond": float(np.linalg.cond(fit.fisher)), "log_prior_at_map": fit.log_prior},
           "evidence": ev, "time_s": time.perf_counter() - t_all, "thread_env": thread_env()}
    save_npz(d / "map.npz", u_hat=search.u_hat, H=fit.H, cov=fit.cov, H_fd=np.full_like(fit.H, np.nan) if fit_fd is None else fit_fd.H)
    write_json(d / "map.json", rec)  # last: the JSON is the done marker
    print(f"  MAP: Phi {best['phi']:.1f} (N = {prob.n_data}, Phi(truth) {prob.operating_point['phi_truth']:.1f}), p_hat = "
          + ", ".join(f"{k} {v:.4g}" for k, v in zip(model.param_names, p_hat)) + f"; Laplace log Z {ev['laplace_gn']:.2f} (fd {ev['laplace_fd']}); {rec['time_s']:.0f} s")
    return rec


def _u_gap(cands):
    done = [c["U"] for c in cands if c["converged"]]
    return float(max(done) - min(done)) if len(done) > 1 else None


def load_map(args):
    d = case_dir(args)
    if not (d / "map.json").exists():
        raise SystemExit(f"{d / 'map.json'} does not exist: run --stage map first")
    z = np.load(d / "map.npz")
    return json.loads((d / "map.json").read_text()), z


def laplace_map(z):
    """The ``AffineMap`` of the stored Laplace fit: center u_hat, covariance H^-1 (Gauss-Newton Hessian)."""
    return AffineMap.from_covariance(z["u_hat"], z["cov"])


# ----------------------------------------------------------------------------------------------
# Chains
# ----------------------------------------------------------------------------------------------


def segment_path(d, c, j):
    return d / "chains" / f"c{c}_s{j}.npz"


def stage_chain(args):
    d = case_dir(args)
    rec, z = load_map(args)
    amap = laplace_map(z)
    digest = amap.digest()
    prob = build_problem(args)
    post = make_posterior(args, prob)
    chains = range(args.chains) if args.chain_id is None else [args.chain_id]
    for c in chains:
        ckey = jax.random.fold_in(jax.random.key(args.chain_seed), c)
        for j in range(args.first_segment, args.last_segment + 1):
            path = segment_path(d, c, j)
            if path.exists() and not args.force:
                continue
            if j == 0:
                zeta0 = sp.overdispersed_starts(amap, jax.random.fold_in(ckey, 10_000), 1, args.init_scale)
                aux0, step0, n_warm = None, 1.0, args.warmup
            else:
                prev = segment_path(d, c, j - 1)
                if not prev.exists():
                    raise SystemExit(f"{prev} is missing: segment {j} of chain {c} continues from it")
                pz = np.load(prev)
                if str(pz["map_digest"]) != digest:
                    raise SystemExit(f"{prev} was made with another Laplace map than map.npz now holds")
                zeta0, aux0, step0, n_warm = pz["final_zeta"][None], None, float(pz["step"]), 0
            t0 = time.perf_counter()
            batch = sp.run_hmc(post, amap, zeta0, jax.random.fold_in(ckey, j), n_warmup=n_warm, n_samples=args.segment_draws, step0=step0, n_leapfrog=args.n_leapfrog,
                               jitter=args.jitter, aux0=aux0)
            batch.add_phi(post)
            save_npz(path, u=batch.u[0], zeta=batch.zeta[0], phi=batch.phi[0], phi_prime=batch.phi_prime[0], accept=batch.accept[0], step=batch.step[0],
                     final_zeta=batch.final_zeta[0], map_digest=digest, n_warmup=n_warm, n_leapfrog=args.n_leapfrog, jitter=args.jitter, n_grad=batch.n_grad,
                     time_s=time.perf_counter() - t0)
            print(f"{case_name(args)} chain {c} segment {j}: {args.segment_draws} draws, step {batch.step[0]:.3f}, acceptance {batch.acceptance:.2f}, "
                  f"Phi {batch.phi.mean():.1f} +- {batch.phi.std():.1f}, {time.perf_counter() - t0:.0f} s")


def assemble_chains(args, skip_segments=0):
    """The recorded draws of every chain: ``u`` (C, n, d), ``phi`` (C, n), ``accept``, ``steps`` (C,), ``n_grad`` and ``time_s`` totals; None if the chains are incomplete."""
    d = case_dir(args)
    _, z = load_map(args)
    digest = laplace_map(z).digest()
    per = []
    for c in range(args.chains):
        segs = []
        for j in range(args.segments):
            p = segment_path(d, c, j)
            if not p.exists():
                break
            segs.append(np.load(p))
        if len(segs) < args.segments:
            return None
        for s in segs:
            if str(s["map_digest"]) != digest:
                raise SystemExit(f"a segment of chain {c} was made with another Laplace map")
        per.append(segs[skip_segments:])
    return {"u": np.stack([np.concatenate([s["u"] for s in segs]) for segs in per]), "phi": np.stack([np.concatenate([s["phi"] for s in segs]) for segs in per]),
            "accept": np.stack([np.concatenate([s["accept"] for s in segs]) for segs in per]), "phi_prime": np.stack([np.concatenate([s["phi_prime"] for s in segs]) for segs in per]),
            "steps": np.array([float(per[c][-1]["step"]) for c in range(args.chains)]),
            "n_grad": int(sum(int(s["n_grad"]) for segs in per for s in segs)), "time_s": float(sum(float(s["time_s"]) for segs in per for s in segs))}


# ----------------------------------------------------------------------------------------------
# Predictive checks
# ----------------------------------------------------------------------------------------------


def stage_ppc(args):
    d = case_dir(args)
    if (d / "ppc.json").exists() and not args.force:
        return json.loads((d / "ppc.json").read_text())
    chains = assemble_chains(args)
    if chains is None:
        raise SystemExit("the chains are incomplete: run --stage chain first")
    rec, z = load_map(args)
    prob = build_problem(args)
    post = make_posterior(args, prob)
    draws = chains["u"].reshape(-1, chains["u"].shape[-1])
    key = jax.random.key(args.ppc_seed)
    t0 = time.perf_counter()
    out = {"refit": sp.predictive_check(post, draws, key, n_rep=args.ppc_reps, mode="refit", u_hat=z["u_hat"]),
           "draw": sp.predictive_check(post, draws, key, n_rep=args.ppc_reps, mode="draw"), "n_data": prob.n_data, "time_s": time.perf_counter() - t0}
    write_json(d / "ppc.json", out)
    print(f"{case_name(args)}: predictive p-value (refit) {out['refit']['p_value']:.3f} [T_obs {out['refit']['t_obs']:.1f} vs replicates {out['refit']['t_rep_mean']:.1f} +- {out['refit']['t_rep_sd']:.1f}], "
          f"(draw) {out['draw']['p_value']:.3f}")
    return out


# ----------------------------------------------------------------------------------------------
# Thermodynamic integration
# ----------------------------------------------------------------------------------------------


def ti_tag_suffix(tag):
    return f"_{tag}" if tag else ""


def ti_path(d, k, tag=""):
    return d / f"ti{ti_tag_suffix(tag)}" / f"t{k:02d}.npz"


def ti_json_path(d, tag=""):
    return d / f"ti{ti_tag_suffix(tag)}.json"


def stage_ti(args):
    d = case_dir(args)
    tag = args.ti_tag
    if ti_json_path(d, tag).exists() and not args.force:
        print(f"{case_name(args)}: TI {tag or 'run'} done")
        return json.loads(ti_json_path(d, tag).read_text())
    prob = build_problem(args)
    model = sm.make_model(args.model)
    post = make_posterior(args, prob)
    betas = sp.ti_schedule(args.ti_temps, args.ti_power)
    keys = jax.random.split(jax.random.key(args.ti_seed), args.ti_temps)
    rng = np.random.default_rng(args.ti_seed)
    starts = np.asarray(jax.vmap(model.u_from_p)(jnp.asarray(model.sample_prior_p(rng, args.ti_chains))))
    amap = sp.prior_map(post)
    all_stats = []
    for k, beta in enumerate(betas):
        path = ti_path(d, k, tag)
        if path.exists() and not args.force:
            z = np.load(path)
            starts, amap = z["final_u"], AffineMap(z["center"], z["chol"])
            all_stats.append(json.loads(str(z["stats"])))
            continue
        t0 = time.perf_counter()
        res = sp.ti_temperature(post, beta, starts, amap, keys[k], n_pilot=args.ti_pilot if k > 0 else 0, n_warmup=args.ti_warmup, n_samples=args.ti_draws,
                                n_leapfrog=args.ti_leapfrog, adapt=k > 0)
        st = dict(res.stats)
        st["wall_s"] = time.perf_counter() - t0
        starts, amap = res.u[:, -1, :], res.batch.amap
        save_npz(path, phi=res.phi, u=res.u, final_u=starts, center=amap.center_np, chol=amap.chol_np, step=res.batch.step, stats=json.dumps(st, default=_default))
        all_stats.append(st)
        print(f"{case_name(args)} TI{ti_tag_suffix(tag)} {k + 1}/{len(betas)}: beta {beta:.3e}, E[phi] {st['mean_phi']:.1f}, var {st['var_phi']:.1f}, ESS {st['ess_phi']:.0f}, "
              f"R-hat {st['rhat_phi']:.3f}, acceptance {st['acceptance']:.2f}, {st['wall_s']:.0f} s")
    out = sp.ti_combine(post, all_stats)
    out["settings"] = {k: getattr(args, k) for k in ("ti_temps", "ti_power", "ti_chains", "ti_pilot", "ti_warmup", "ti_draws", "ti_leapfrog", "ti_seed", "ti_tag")}
    out["wall_s"] = float(sum(s.get("wall_s", 0.0) for s in all_stats))
    out["n_grad_per_chain"] = int(sum(s["n_grad_per_chain"] for s in all_stats))
    write_json(ti_json_path(d, tag), out)
    print(f"{case_name(args)}: TI{ti_tag_suffix(tag)} log Z = {out['log_z']:.2f} +- {out['mc_se']:.2f} (trapezoid {out['log_z_trapezoid']:.2f}, Richardson error estimate "
          f"{out['discretization']['richardson_error_estimate']:.2f}); converged {out['converged']}")
    return out


def combine_ti_runs(d):
    """The thermodynamic-integration runs of a case (<dir>/ti.json and <dir>/ti_<tag>.json) as one estimate; None if there is none.

    One run is reported as it is (``runs`` = 1). Several independent runs (different seeds, e.g. one chain each) are averaged: ``log_z`` the mean, ``mc_se`` the quadrature sum of
    the runs' errors divided by the number of runs, and ``between_run_sd`` the standard deviation of the run estimates (the direct, assumption-free error bar; compare it with
    ``mc_se``). ``converged`` needs every run converged.
    """
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("ti*.json")) if p.name == "ti.json" or p.name.startswith("ti_")]
    if not runs:
        return None
    z = np.array([r["log_z"] for r in runs])
    out = {"runs": len(runs), "log_z": float(z.mean()), "log_z_trapezoid": float(np.mean([r["log_z_trapezoid"] for r in runs])),
           "mc_se": float(np.sqrt(np.sum([r["mc_se"] ** 2 for r in runs])) / len(runs)), "between_run_sd": float(z.std(ddof=1)) if len(runs) > 1 else None,
           "per_run": [{"log_z": r["log_z"], "mc_se": r["mc_se"], "converged": r["converged"], "tag": r["settings"].get("ti_tag", "")} for r in runs],
           "converged": bool(all(r["converged"] for r in runs)), "worst_rhat": float(max(r["worst_rhat"] for r in runs)), "min_ess": float(min(r["min_ess"] for r in runs)),
           "n_temps": runs[0]["n_temps"], "wall_s": float(sum(r["wall_s"] for r in runs)), "settings": runs[0]["settings"],
           "discretization": {"richardson_error_estimate": float(np.mean([r["discretization"]["richardson_error_estimate"] for r in runs])),
                              "corrected_minus_trapezoid": float(np.mean([r["discretization"]["corrected_minus_trapezoid"] for r in runs]))}}
    return out


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def stage_finalize(args):
    d = case_dir(args)
    if (d / "summary.json").exists() and not args.force:
        print(f"{case_name(args)}: summary done")
        return json.loads((d / "summary.json").read_text())
    chains = assemble_chains(args)
    if chains is None:
        raise SystemExit("the chains are incomplete: run --stage chain first")
    rec, z = load_map(args)
    problem = json.loads((d / "problem.json").read_text()) if (d / "problem.json").exists() else stage_prepare(args)
    prob = build_problem(args)
    model = sm.make_model(args.model)
    post = make_posterior(args, prob)
    u = chains["u"]
    p = sp.physical_draws(post, u)
    conv = sp.convergence_report(chains["phi"], u, p, names=model.param_names)
    # multimodality: rebuild the MAP search from the stored candidates
    cands = []
    for c in rec["candidates"]:
        cands.append({"label": c["label"], "u": np.asarray(c["u"]), "U": c["U"], "converged": c["converged"]})
    search = sp.MapSearch(cands, rec["best"], np.asarray(rec["u_hat"]))
    fit = sp.laplace_fit(post, z["u_hat"])
    batch_like = SimpleNamespace(u=u)
    multi = sp.detect_multimodality(post, search, fit, batch_like, names=list(model.param_names))
    names = list(model.param_names)
    pooled = p.reshape(-1, p.shape[-1])
    param_summary = {k: {"mean": float(pooled[:, i].mean()), "sd": float(pooled[:, i].std(ddof=1)), "ci90": list(sp.credible_interval(pooled[:, i], 0.9)),
                         "laplace_map": float(rec["p_hat"][i])} for i, k in enumerate(names)}
    gap = None
    if "g" in names:
        gap = sp.gap_summary(p[..., names.index("g")], g_true=prob.g_true, g_det=prob.g_det)
        gap["per_chain_mean"] = p[..., names.index("g")].mean(axis=1).tolist()
    ti = combine_ti_runs(d)
    ppc = json.loads((d / "ppc.json").read_text()) if (d / "ppc.json").exists() else None
    ev = dict(rec["evidence"])
    ev["ti"] = ti
    out = {"case": case_name(args), "truth_label": truth_label(args.truth, args.gap), "model": args.model, "problem": problem, "g_true": prob.g_true, "g_det": prob.g_det,
           "map": {"phi_map": rec["phi_map"], "n_data": rec["n_data"], "phi_over_N": rec["phi_over_N"], "phi_truth": rec["phi_truth"],
                   "phi_map_minus_phi_truth": rec["phi_map"] - rec["phi_truth"], "p_hat": dict(zip(names, rec["p_hat"])), "converged": rec["map_converged"],
                   "u_gap": rec["u_gap"], "n_candidates": len(rec["candidates"])},
           "evidence": ev, "posterior": {"params": param_summary, "gap": gap, "n_chains": u.shape[0], "n_draws_per_chain": u.shape[1], "acceptance": float(chains["accept"].mean()),
                                         "steps": chains["steps"], "phi_mean": float(chains["phi"].mean()), "phi_sd": float(chains["phi"].std(ddof=1)),
                                         "phi_over_N": float(chains["phi"].mean() / prob.n_data), "n_grad_total": chains["n_grad"], "chain_time_s": chains["time_s"]},
           "convergence": conv, "multimodality": multi, "ppc": ppc,
           "cost": {"map_time_s": rec["time_s"], "chain_time_s": chains["time_s"], "ppc_time_s": None if ppc is None else ppc["time_s"], "ti_time_s": None if ti is None else ti["wall_s"]},
           "thread_env": thread_env()}
    write_json(d / "summary.json", out)
    gtxt = "" if gap is None else f"; g = {gap['mean']:.5f} +- {gap['sd']:.5f}, 90% CI [{gap['ci'][0]:.5f}, {gap['ci'][1]:.5f}], covers {gap['covers_true']}, P(g < g_det) = {gap.get('p_below_g_det')}"
    print(f"{case_name(args)}: Phi_MAP/N = {rec['phi_over_N']:.3f}, converged {conv['converged']} (R-hat {conv['max_rhat']:.3f}, ESS {conv['worst_ess_bulk']:.0f}), multimodal {multi['multimodal']}{gtxt}")
    return out


# ----------------------------------------------------------------------------------------------
# Pixel posteriors of the same data (milestone 6)
# ----------------------------------------------------------------------------------------------


def stage_pixel(args):
    """The Laplace pixel posteriors of the truth's data under the Matern prior and the TV prior at lambda*(eps_r) of milestone 6: MAP, dense Laplace covariance and the
    gap-valley probability of the half-height strip (``--band-frac``; region means, never single pixels; a strip the grid or the pixelized truth cannot show is reported as
    unrepresentable), with the single-point values labeled."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import resolution_laplace as rl
    from bayes_scatter import thetaspace as ts

    d = case_dir(args)
    if (d / "summary.json").exists() and not args.force:
        print(f"{case_name(args)}: pixel summary done")
        return json.loads((d / "summary.json").read_text())
    prob = build_problem(args)
    prob.obj = el.feature_reference(prob.truth, prob.g_true if prob.g_true is not None else 0.0, r=args.radius, eps_r=args.eps_r)
    fo = el.FeatureObject(prob.truth, prob.obj)
    fg = gv.feature_geometry(prob.grid, fo, args.core_frac, args.band_frac)
    out = {"case": case_name(args), "truth_label": truth_label(args.truth, args.gap), "g_true": prob.g_true, "feature": fg.describe(), "band_frac": args.band_frac,
           "core_frac": args.core_frac, "priors": {}, "operating_point": prob.operating_point}
    ns = SimpleNamespace(**{k: getattr(args, k) for k in ("start_maxiter", "newton_iter", "newton_iter_smooth", "newton_tol", "all_starts", "sigma", "ell", "nu", "delta_tv", "sigma_0")})
    theta_matern = None
    for prior_name in ("matern", "tv"):
        t0 = time.perf_counter()
        lam = rl.lambda_star_of(args.lambda_dir, args.eps_r) if prior_name == "tv" else None
        prior = rl.build_prior(ns, prior_name, prob.grid, lam)
        target = ts.ThetaTarget(prob.forward, prior, prob.d_obs, prob.s)
        _, theta_hat, minfo = rl.start_and_map(prob, target, ns, prior_name, theta_matern=theta_matern)
        if prior_name == "matern":
            theta_matern = np.asarray(theta_hat)
        m = minfo["map"]
        mass = target.laplace_mass(theta_hat)
        feats = gv.gaussian_valley(fg, theta_hat, rl.chol_solver(mass.chol))
        out["priors"][prior_name] = {"lam": lam, "phi_map": m["phi"], "phi_over_N": m["phi"] / prob.n_data, "R": m["R"], "converged": m["converged"],
                                     "map_ok": bool(m["converged"] and m["phi"] < rl.MAP_OK_PHI_OVER_N * prob.n_data), "features": feats, "truth": gv.truth_valley(fg, prob.theta_true),
                                     "time_s": time.perf_counter() - t0}
        reg = feats["region"]
        print(f"{case_name(args)} {prior_name}: Phi_MAP {m['phi']:.1f} (N = {prob.n_data}); P(valley) region "
              f"{'unrepresentable (' + feats['region_status'] + ')' if reg is None else format(reg['p_valley'], '.4f')}, point {feats['point']['p_valley']:.4f}")
    write_json(d / "summary.json", out)
    return out


# ----------------------------------------------------------------------------------------------
# Cost and budget
# ----------------------------------------------------------------------------------------------


def time_calls(fn, n):
    """Seconds per call of ``fn()`` (which must block) after one untimed call: ``(median, mean, min)`` over n calls."""
    fn()
    ts_ = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts_.append(time.perf_counter() - t0)
    return float(np.median(ts_)), float(np.mean(ts_)), float(np.min(ts_))


def measure_model_cost(args, model_name, prob, n_evals):
    """Per-evaluation cost of one model on the inference grid of ``prob``: value + gradient at the truth-like point (warm solver start, as inside a chain), at prior draws
    (cold start from the previous draw), the misfit alone, the Fisher matrix; the compile time of the first call."""
    model = sm.make_model(model_name)
    post = sp.ShapePosterior(model, prob.forward, prob.d_obs, prob.s, prob.renderer)
    rng = np.random.default_rng(0)
    f2 = sm.F2()
    p_f2 = np.array([args.gap if args.truth != "T4" else 0.0, args.radius, args.eps_r, 0.0, 0.0])
    p_f2[0] = max(p_f2[0], 1e-3)
    u_near = np.asarray(f2.u_from_p(jnp.asarray(p_f2))) if model_name == "F2" else np.asarray(model.u_from_p(jnp.asarray(f2.to_f2u(p_f2) if model_name == "F2u" else f2.to_fe(p_f2))))
    aux = post.initial_aux()
    t0 = time.perf_counter()
    jax.block_until_ready(post.phi_and_grad(jnp.asarray(u_near), aux, post.data))
    compile_s = time.perf_counter() - t0
    state = {"aux": aux}

    def near():
        _, g, state["aux"] = post.phi_and_grad(jnp.asarray(u_near + 1e-3 * rng.standard_normal(model.dim)), state["aux"], post.data)
        jax.block_until_ready(g)

    near_med, near_mean, near_min = time_calls(near, n_evals)
    draws = model.start_points_u(rng, n_evals + 1)
    it = iter(draws)

    def prior_draw():
        _, g, state["aux"] = post.phi_and_grad(jnp.asarray(next(it)), state["aux"], post.data)
        jax.block_until_ready(g)

    pd_med, pd_mean, pd_min = time_calls(prior_draw, n_evals)
    phi_med, phi_mean, _ = time_calls(lambda: jax.block_until_ready(post.phi(jnp.asarray(u_near), state["aux"], post.data)[0]), n_evals)
    t0 = time.perf_counter()
    post.fisher(u_near)
    fisher_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    post.fisher(u_near)
    fisher_s = min(fisher_s, time.perf_counter() - t0)
    return {"model": model_name, "dim": model.dim, "compile_s": compile_s, "eval_near_truth_s": {"median": near_med, "mean": near_mean, "min": near_min},
            "eval_prior_draws_s": {"median": pd_med, "mean": pd_mean, "min": pd_min}, "misfit_only_s": phi_med, "fisher_s": fisher_s, "fisher_over_eval": fisher_s / near_mean}


def estimate_budget(costs, args, plan=PLAN_TRUTHS, models=MODELS):
    """Gradient-evaluation counts and time for the DESIGN 14.3 plan with ``args`` settings, from the measured per-evaluation costs.

    Counts per case and model (value+gradient evaluations unless stated; E is the near-truth warm cost, the cost inside a chain):

    * MAP: (1 + ``n_prior_starts`` + 1 sibling start + the symmetry images) starts, each ~ ``lbfgs_iter`` L-BFGS evaluations (an upper bound: runs stop earlier) plus
      ``polish_iter`` x (1 + Fisher) with the Fisher matrix costing ``fisher_over_eval`` evaluations; plus the finite-difference Hessian (2 dim evaluations).
    * chains: ``chains`` x (``warmup`` + ``segments`` x ``segment_draws``) x ``n_leapfrog`` evaluations, plus ``segments`` x ``segment_draws`` x ``chains`` misfit-only
      evaluations (``misfit_only_s``).
    * predictive check: ``ppc_reps`` refits (each ~4 Newton iterations of (1 + Fisher) + line search 2 evaluations) in "refit" mode plus 2 misfit evaluations per replicate in "draw" mode.
    * TI (one case per model of ``budget_ti_models``, default F2 and Fe): ``ti_temps`` x ``ti_chains`` x (``ti_pilot`` x 2 + ``ti_warmup`` + ``ti_draws``) x ``ti_leapfrog``
      evaluations at the mean of the near-truth and the prior-draw cost (the early temperatures sit in the prior), plus the misfit passes.
    The wall-clock estimate divides the single-thread seconds by the effective parallelism of ``slots`` single-thread processes on the Mac: ``slots`` up to 5, and 0.24 of a
      process for each one beyond (milestone 5 measured 19 % more gradients per second with 9 processes than with 5: DESIGN 12.5).
    """
    n_cases = len(plan)
    per_model = {}
    total_s = 0.0
    total_evals = 0.0
    for name in models:
        c = costs[name]
        e = c["eval_near_truth_s"]["mean"]
        e_ti = 0.5 * (e + c["eval_prior_draws_s"]["mean"])
        fisher_evals = c["fisher_over_eval"]
        d = c["dim"]
        n_starts = 1 + args.n_prior_starts + (1 if name != "F2" else 0) + 2
        map_evals = n_starts * (args.lbfgs_iter + args.polish_iter * (1.0 + fisher_evals)) + 2 * d
        draws = args.segments * args.segment_draws
        chain_evals = args.chains * (args.warmup + draws) * args.n_leapfrog
        chain_misfit = args.chains * draws * (c["misfit_only_s"] / e)
        ppc_evals = args.ppc_reps * (4.0 * (1.0 + fisher_evals) + 2.0) + 2.0 * args.ppc_reps * (c["misfit_only_s"] / e)
        case_evals = map_evals + chain_evals + chain_misfit + ppc_evals
        ti_evals = args.ti_temps * args.ti_chains * (2 * args.ti_pilot + args.ti_warmup + args.ti_draws) * args.ti_leapfrog
        ti_misfit = args.ti_temps * args.ti_chains * args.ti_draws * (c["misfit_only_s"] / e)
        ti_s = (ti_evals + ti_misfit) * e_ti  # seconds: the evaluations at the TI cost
        case_s = case_evals * e
        per_model[name] = {"dim": d, "eval_s": e, "eval_ti_s": e_ti, "evals_per_case": case_evals, "seconds_per_case_1thread": case_s,
                           "evals_map": map_evals, "evals_chains": chain_evals, "evals_ppc": ppc_evals, "ti_evals_per_case": ti_evals, "ti_seconds_per_case_1thread": ti_s}
        total_s += case_s * n_cases
        total_evals += case_evals * n_cases
    ti_models = list(args.budget_ti_models)
    ti_total = sum(per_model[m]["ti_seconds_per_case_1thread"] for m in ti_models)
    out = {"n_cases": n_cases * len(models), "plan": [list(x) for x in plan], "models": list(models), "per_model": per_model, "posterior_total_1thread_s": total_s,
           "posterior_total_evals": total_evals, "ti_cases": ti_models, "ti_total_1thread_s": ti_total, "total_1thread_s": total_s + ti_total, "wall_hours": {}}
    for slots in args.budget_slots:
        eff = min(slots, 5) + 0.24 * max(slots - 5, 0)
        out["wall_hours"][str(slots)] = {"effective_parallelism": eff, "posterior_h": total_s / eff / 3600.0, "ti_h": ti_total / eff / 3600.0, "total_h": (total_s + ti_total) / eff / 3600.0}
    out["note"] = ("seconds are single-thread seconds from the measured near-truth cost; MAP counts are upper bounds (L-BFGS and Newton stop early); a model that fits badly (Fe) may "
                   "need more MAP iterations and has costlier evaluations at its extreme states; TI evaluations use the mean of the near-truth and prior-draw costs.")
    return out


def stage_cost(args):
    path = Path(args.out_dir) / "cost.json"
    if path.exists() and not args.force:
        print(f"cost done ({path})")
        return json.loads(path.read_text())
    args.truth = args.truth or "T1"
    prob = build_problem(args)  # T1 at --gap: the data only set the noise level and a realistic warm start
    costs = {}
    for name in MODELS:
        costs[name] = measure_model_cost(args, name, prob, args.cost_evals)
        c = costs[name]
        print(f"{name} (dim {c['dim']}): value+grad near the truth {c['eval_near_truth_s']['mean'] * 1e3:.0f} ms (median {c['eval_near_truth_s']['median'] * 1e3:.0f}), at prior draws "
              f"{c['eval_prior_draws_s']['mean'] * 1e3:.0f} ms, misfit only {c['misfit_only_s'] * 1e3:.0f} ms, Fisher {c['fisher_s']:.2f} s, first-call compile {c['compile_s']:.1f} s")
    budget = estimate_budget(costs, args)
    out = {"n": args.n, "n_tx": args.n_tx, "n_rx": args.n_rx, "tol": args.tol, "thread_env": thread_env(), "load_avg_1min": os.getloadavg()[0], "costs": costs, "budget": budget,
           "args": {k: getattr(args, k) for k in ("chains", "warmup", "segments", "segment_draws", "n_leapfrog", "n_prior_starts", "lbfgs_iter", "polish_iter", "ppc_reps",
                                                    "ti_temps", "ti_chains", "ti_pilot", "ti_warmup", "ti_draws", "ti_leapfrog", "budget_ti_models", "budget_slots")}}
    write_json(path, out)
    for slots, w in budget["wall_hours"].items():
        print(f"budget: {budget['n_cases']} cases + {len(budget['ti_cases'])} TI runs = {budget['total_1thread_s'] / 3600:.1f} single-thread hours; "
              f"{slots} processes: {w['total_h']:.1f} h wall (posteriors {w['posterior_h']:.1f} h, TI {w['ti_h']:.1f} h)")
    return out


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "cost":
        return stage_cost(args)
    if args.model == "pixel":
        stage_prepare(args)
        return stage_pixel(args)
    if args.stage in ("all", "prepare"):
        stage_prepare(args)
    if args.stage in ("all", "map"):
        stage_map(args)
    if args.stage in ("all", "chain"):
        stage_chain(args)
    if args.stage in ("all", "ppc"):
        if args.stage == "ppc" or assemble_chains(args) is not None:
            stage_ppc(args)
    if args.stage == "ti":
        stage_ti(args)
    if args.stage in ("all", "finalize"):
        if args.stage == "finalize" or assemble_chains(args) is not None:
            stage_finalize(args)
    if args.stage == "pixel":
        raise SystemExit("--stage pixel needs --model pixel")


if __name__ == "__main__":
    main()
