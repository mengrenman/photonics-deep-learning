"""MAP sweep over the weight lambda of the pseudo-Huber TV prior; lambda* by the discrepancy principle (DESIGN 12.1).

    python scripts/tv_lambda_sweep.py [--lams 0.3 1 3 10] [--delta 0.01] [--update] [--out tv_lambda_sweep.json]

The problem is the milestone-4 one (austria_fixed eps_r 2, side 2, inference grid n = 48, data on n = 96, 30 dB, data seed 0,
40 x 60 antennas; ``run_prior_comparison.build_data``), so the data are the realization of results/prior_comparison/pc_*
(checked numerically against ``--reference-run``'s d_obs: max |delta d| / s <= ``D_OBS_TOL``). For every lambda the MAP of
U(theta) = Phi + R in theta = log eps_r is computed by the milestone-4 path: the short smooth-prior start
(``thetaspace.smooth_start``, computed once and shared) followed by L-BFGS-B (``thetaspace.map_theta``, --map-maxiter, --map-gtol).
Every lambda starts from the same point, so the MAPs are comparable and independent of the order of the sweep; U is not convex in
theta (the forward model is nonlinear), so another start could reach another local minimum.

Reported per lambda: ``phi`` (Phi_MAP), ``R``, ``U``, ``phi_over_n_data``, ``grad_inf`` (max-norm of grad U at the MAP), ``nit``,
``nfev``, ``success`` and ``message`` of L-BFGS-B, ``time_s``, ``u_convergence`` (best U so far at fractions of the run and the
decrease over its last 10%); the errors ``err_model`` (book eq. 15.21 on eps_r), ``err_chi``
(contrast-normalized) and ``err_data`` (of the MAP's data against the noise-free data); ``tv`` (sum_pairs b |D_ij| of the MAP and of the truth);
``pairs_below_delta`` (fraction of the neighbor pairs with |D| < delta, the part of the MAP that sits in the quadratic core of the
prior); ``edge_bias`` (mean of MAP - truth in log eps_r at signed distance d = -1 outside, +1 and +2 inside the object, as in
scripts/edge_analysis.py); and the Laplace standard deviation of theta at the MAP (mean inside / outside the object).

lambda* is the swept value whose Phi_MAP is closest to N (the number of complex data, 2400): Morozov's discrepancy principle,
which is a MAP heuristic. It is not a statement about the posterior: draws sit above the MAP in Phi (milestone 4: 2434.7
against 2378.4 for the Matern prior), and Phi of the truth on the n = 48 grid is itself 2578, not N. ``crossing_lam_interpolated`` also gives the log-log-linear interpolation of lambda at Phi_MAP = N between the two neighbors, for information
and for choosing the next lambda to add; it is not used in the choice.

Writes <dir>/<out> (default results/prior_comparison/tv_lambda_sweep.json), <dir>/tv_lambda_sweep_maps.npz (theta_hat per
lambda) and <dir>/tv_lambda_sweep.png. ``--update`` keeps the lambdas already in the JSON (a refinement run adds lambdas to
the sweep; a lambda that is run again is replaced); lambda* is then chosen over all stored lambdas.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure

import bayes_scatter  # noqa: F401
from bayes_scatter import inference as inf
from bayes_scatter import mrf
from bayes_scatter import thetaspace as ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import edge_analysis as ea  # noqa: E402
import run_prior_comparison as rpc  # noqa: E402

RESULTS = ROOT / "results" / "prior_comparison"
print = functools.partial(print, flush=True)
EDGE_DISTANCES = (-1, 1, 2)


def key_of(lam):
    """The JSON key of a lambda: its shortest round-trip representation."""
    return repr(float(lam))


def pair_stats(theta, truth, n, delta):
    """``(tv, tv_truth, below)``: sum_pairs b |D| of ``theta`` and of ``truth`` ((n, n) arrays), and the fraction of the
    neighbor pairs of ``theta`` with |D| < delta."""
    i, j, b = mrf.neighbor_pairs(n)
    d = np.asarray(theta).ravel()[i] - np.asarray(theta).ravel()[j]
    dt = np.asarray(truth).ravel()[i] - np.asarray(truth).ravel()[j]
    return float(np.sum(b * np.abs(d))), float(np.sum(b * np.abs(dt))), float(np.mean(np.abs(d) < delta))


def edge_bias(theta_hat, theta_true, chi_true, distances=EDGE_DISTANCES):
    """Mean of (theta_hat - theta_true) over the pixels at each signed distance to the object edge (positive inside)."""
    d = ea.signed_distance(chi_true)
    return {str(k): float(np.mean((np.asarray(theta_hat) - theta_true)[d == k])) for k in distances if np.any(d == k)}


def u_convergence(history, marks=(0.25, 0.5, 0.75, 0.9)):
    """How far the MAP iteration had settled: the best-so-far U at fractions ``marks`` of the function evaluations and at the end,
    and the decrease over the last 10% of the evaluations (``u_drop_last_10pct``; 0 means no change at the printed precision)."""
    best = np.minimum.accumulate(np.asarray(history, dtype=np.float64))
    k = len(best)
    out = {f"u_at_{int(100 * m)}pct": float(best[min(int(m * k), k - 1)]) for m in marks}
    out["u_end"] = float(best[-1])
    out["u_drop_last_10pct"] = float(best[min(int(0.9 * k), k - 1)] - best[-1])
    out["nfev"] = int(k)
    return out


def run_lambda(lam, ctx, theta0, args):
    """MAP, errors and summaries at one lambda (see the module docstring); returns a JSON-serializable dict and theta_hat."""
    prob, grid, n = ctx.prob, ctx.grid, ctx.args.n
    prior = mrf.PseudoHuberTV(n, lam=lam, delta=args.delta, sigma_0=args.sigma_0, mu=args.mu)
    target = ts.ThetaTarget(prob.forward, prior, prob.d_obs, prob.s)
    theta_hat, info = ts.map_theta(target, theta0, maxiter=args.map_maxiter, gtol=args.map_gtol)
    chi_map = np.asarray(target.chi(theta_hat))
    d_map, _ = prob.forward(jnp.asarray(chi_map))
    tv, tv_truth, below = pair_stats(np.asarray(theta_hat), prob.theta_true, n, args.delta)
    mass = target.laplace_mass(theta_hat)
    lap = mass.theta_std()
    out = {"lam": float(lam), "phi": info["phi"], "R": info["R"], "U": info["U"], "n_data": ctx.n_data,
           "phi_over_n_data": info["phi"] / ctx.n_data, "grad_inf": info["grad_inf"], "nit": info["nit"],
           "nfev": info["nfev"], "success": info["success"], "message": info["message"], "time_s": info["time_s"],
           "err_model": inf.err_model(1.0 + chi_map, prob.eps_true), "err_chi": inf.err_chi(chi_map, prob.chi_true),
           "err_data": inf.err_data(d_map, prob.d_clean), "tv": tv, "tv_truth": tv_truth, "pairs_below_delta": below,
           "u_convergence": u_convergence(info["U_history"]),
           "edge_bias": edge_bias(theta_hat, prob.theta_true, prob.chi_true),
           "laplace_theta_std": rpc.map_stats(lap, ctx.inside), "laplace_pivot_ratio_sq": mass.info["pivot_ratio_sq"]}
    return out, np.asarray(theta_hat)


def choose_lambda_star(runs, n_data):
    """``(lam_star, crossing)`` over the stored runs (dict lam-key -> record): the lambda with Phi_MAP closest to ``n_data``,
    and the interpolated lambda at Phi_MAP = n_data between its two neighbors in lambda (None if Phi_MAP does not cross)."""
    recs = sorted(runs.values(), key=lambda r: r["lam"])
    star = min(recs, key=lambda r: (abs(r["phi"] - n_data), r["lam"]))
    crossing = None
    for a, b in zip(recs[:-1], recs[1:]):
        if (a["phi"] - n_data) * (b["phi"] - n_data) < 0:
            t = (n_data - a["phi"]) / (b["phi"] - a["phi"])
            crossing = float(np.exp(np.log(a["lam"]) + t * (np.log(b["lam"]) - np.log(a["lam"]))))
            break
    return star, crossing


def plot_sweep(runs, n_data, path):
    recs = sorted(runs.values(), key=lambda r: r["lam"])
    lam = [r["lam"] for r in recs]
    fig = Figure(figsize=(10, 3.2))
    ax = fig.subplots(1, 3)
    ax[0].semilogx(lam, [r["phi"] for r in recs], "o-", label="Phi_MAP")
    ax[0].axhline(n_data, color="k", ls="--", lw=0.8, label="N")
    ax[0].set(xlabel="lambda", ylabel="Phi_MAP")
    ax[0].legend()
    ax[1].loglog(lam, [r["R"] for r in recs], "o-")
    ax[1].set(xlabel="lambda", ylabel="R_MAP")
    ax[2].semilogx(lam, [r["err_model"] for r in recs], "o-", label="err_model")
    ax[2].semilogx(lam, [r["err_chi"] for r in recs], "s-", label="err_chi")
    ax[2].set(xlabel="lambda", ylabel="MAP error")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0, 2.0, 3.0, 4.5, 10.0])
    ap.add_argument("--delta", type=float, default=0.01, help="pseudo-Huber width (log eps_r units)")
    ap.add_argument("--sigma-0", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--side", type=float, default=2.0)
    ap.add_argument("--eps-r", type=float, default=2.0)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--start-maxiter", type=int, default=100)
    ap.add_argument("--map-maxiter", type=int, default=4000, help="L-BFGS-B iterations (the TV MAP needs more than the 400 of the Gaussian and q-GGMRF priors)")
    ap.add_argument("--map-gtol", type=float, default=1e-6)
    ap.add_argument("--reference-run", default="pc_matern", help="a milestone-4 run directory whose d_obs is the data to match")
    ap.add_argument("--dir", default=str(RESULTS))
    ap.add_argument("--out", default="tv_lambda_sweep.json")
    ap.add_argument("--update", action="store_true", help="keep the lambdas already stored in the JSON")
    args = ap.parse_args(argv)
    if min(args.lams) <= 0:
        ap.error("every lambda must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    base = Path(args.dir)
    out_path = Path(args.out) if Path(args.out).is_absolute() else base / args.out
    rargs = rpc.parse_args(["--prior", "tv", "--lam", str(args.lams[0]), "--delta", str(args.delta), "--sigma-0", str(args.sigma_0),
                            "--mu", str(args.mu), "--n", str(args.n), "--side", str(args.side), "--eps-r", str(args.eps_r),
                            "--snr", str(args.snr), "--seed", str(args.seed), "--data-factor", str(args.data_factor),
                            "--n-tx", str(args.n_tx), "--n-rx", str(args.n_rx), "--tol", str(args.tol), "--out", "unused"])
    t0 = time.perf_counter()
    ctx = rpc.build_context(rargs)
    prob = ctx.prob
    data_check = {"reference_run": args.reference_run, "d_obs_sha256": prob.sha}
    ref = base / args.reference_run / "chains.npz"
    if ref.exists():
        diff = rpc.d_obs_max_diff(np.load(ref)["d_obs"], prob.d_obs, prob.s)
        data_check["max_abs_diff_over_noise_std"] = diff
        data_check["tolerance"] = rpc.D_OBS_TOL
        if diff > rpc.D_OBS_TOL:
            raise SystemExit(f"the data differ from {ref}'s d_obs (max |delta d| / s = {diff:.2e} > {rpc.D_OBS_TOL:g})")
    else:
        data_check["note"] = f"{ref} not found: the data were not compared"
    op = inf.operating_point(prob.forward, jnp.asarray(prob.chi_true), prob.d_clean, prob.d_obs, prob.s)
    print(f"problem: n={args.n} side={args.side} eps_r={args.eps_r} snr={args.snr} seed={args.seed}; N = {ctx.n_data}; "
          f"d_obs sha256 {prob.sha[:16]}...; data check {data_check.get('max_abs_diff_over_noise_std')}")
    t1 = time.perf_counter()
    theta0, start_info = ts.smooth_start(prob.forward, prob.d_obs, prob.s, ctx.grid, maxiter=args.start_maxiter)
    start_s = time.perf_counter() - t1
    print(f"smooth start: {start_info['nit']} iterations, {start_s:.1f} s")

    runs, maps = {}, {}
    if args.update and out_path.exists():
        old = json.loads(out_path.read_text())
        runs = dict(old["runs"])
        npz = out_path.with_name(out_path.stem + "_maps.npz")
        if npz.exists():
            maps = dict(np.load(npz))
    for lam in args.lams:
        rec, theta_hat = run_lambda(lam, ctx, theta0, args)
        runs[key_of(lam)] = rec
        maps[f"theta_hat_lam{key_of(lam)}"] = theta_hat
        print(f"lambda {lam:g}: Phi_MAP {rec['phi']:.1f} (N {ctx.n_data}), R {rec['R']:.1f}, err_model {rec['err_model']:.4f}, "
              f"err_chi {rec['err_chi']:.4f}, |grad|inf {rec['grad_inf']:.1e}, {rec['nit']} its / {rec['nfev']} evals, "
              f"{rec['time_s']:.0f} s; {rec['message']}")
    star, crossing = choose_lambda_star(runs, ctx.n_data)
    out = {"problem": {"n": args.n, "side": args.side, "eps_r": args.eps_r, "snr_db": args.snr, "seed": args.seed,
                       "data_factor": args.data_factor, "n_tx": args.n_tx, "n_rx": args.n_rx, "n_data": ctx.n_data,
                       "noise_std": prob.s, "tol": args.tol, "phi_truth": op["phi_truth"], "phi_clean": op["phi_clean"],
                       "model_to_noise": op["model_to_noise"], "data_check": data_check},
           "prior": {"kind": "tv", "delta": args.delta, "sigma_0": args.sigma_0, "mu": args.mu,
                     "form": "lam sum_pairs b (sqrt(D^2 + delta^2) - delta) + |theta - mu|^2 / (2 sigma_0^2), 8 neighbors"},
           "map": {"start": f"thetaspace.smooth_start, maxiter {args.start_maxiter}, shared by every lambda",
                   "start_nit": start_info["nit"], "start_time_s": start_s, "optimizer": "thetaspace.map_theta (L-BFGS-B)",
                   "maxiter": args.map_maxiter, "gtol": args.map_gtol},
           "runs": dict(sorted(runs.items(), key=lambda kv: float(kv[0]))),
           "lambda_star": {"lam": star["lam"], "phi_map": star["phi"], "n_data": ctx.n_data,
                           "criterion": "swept lambda with Phi_MAP closest to N (Morozov discrepancy principle; a MAP heuristic)",
                           "crossing_lam_interpolated": crossing,
                           "swept": sorted(float(k) for k in runs)},
           "wall_s": time.perf_counter() - t0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    np.savez(out_path.with_name(out_path.stem + "_maps.npz"), **maps)
    plot_sweep(runs, ctx.n_data, out_path.with_suffix(".png"))
    print(f"lambda* = {star['lam']:g} (Phi_MAP {star['phi']:.1f}); interpolated crossing {crossing}; wrote {out_path}")
    return out


if __name__ == "__main__":
    main()
