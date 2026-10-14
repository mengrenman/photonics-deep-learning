"""Milestone 7: the marginal posterior of the gap g by profiling the other parameters, for cases where HMC in the unconstrained gap coordinate is degenerate (T4).

    python scripts/parametric_profile.py --truth T4 --gap 0 --model F2                 # boundary case: a wide grid in u_g = logit(g / 0.4)
    python scripts/parametric_profile.py --truth T1 --gap 0.05 --model F2 --around-map 6 --grid-n 25      # interior case, to check the method against the chains

Why. If the data want g -> 0 (a closed gap, T4), the MAP of F2 sits on the wall of the truncation box |u_g| <= 10 (g = 1.8e-5), the Laplace sd of u_g is 14, and HMC with the Laplace
map is degenerate: chains started outside the box never move, the one inside crawls (DESIGN 14.5, results/parametric/T4_F2/chains_degenerate/). The marginal posterior of g
is still a one-dimensional object, and it is computed here without sampling and without the truncation of u_g: for each value u_g of a grid, the other coordinates v are
profiled (damped Gauss-Newton, warm started from the neighboring grid point) and integrated out by the Laplace approximation in v, which is exact for a Gaussian conditional posterior,

    m(u_g) = exp(lik_const - Phi(u_g, v*) + log pi_u(u_g, v*)) (2 pi)^((d - 1) / 2) det(H_v)^(-1/2),      H_v = Fisher_v + (-Hessian_v log pi_u),

then converted to the density of g, m_g(g) = m(u_g) / (g (1 - g / 0.4)) (g = 0.4 sigmoid(u_g)). Reported: the mean, the 90% interval, P(g < g_det) and P(g < g_true) by quadrature
over the grid (trapezoid in u_g, which must reach the tails: the tail mass beyond the grid is reported), and the profile misfit Phi(u_g, v*), which for T4 falls monotonically to g -> 0.
The marginal is interpolated in u_g (cubic spline of log m on a 32 times finer grid, ``marginal_statistics``), not the CDF linearly in g: with the grid spacing of 0.5 in u_g (a
factor 1.65 in g) the linear interpolation in g put the 95 % point of the T4 marginal 15 % too high. ``truncation`` in the output is the posterior mass of u_g inside the box |u_g| <= 10 of the HMC target
(``shapepost.U_BOUND``): for T4 it is the factor by which the truncated target differs from the posterior (log Z lower by minus its log).
Output <out-dir>/<truth label>_<model>/profile.json. Resumable: every finished grid point is written to profile_partial.json (removed when profile.json is complete), so an
interrupted run (a reboot, a sleep) continues at the first missing point.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from math import log, pi
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

import parametric_run as prun
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp

ROOT = Path(__file__).resolve().parents[1]
print = functools.partial(print, flush=True)


class FixedCoordinate:
    """The problem ``base`` with coordinate ``index`` held at ``value``: dimension d - 1, the same interface (``shapepost`` module docstring). Its ``log_prior`` is the full
    log prior at the completed vector, so the fixed coordinate's prior term is a constant of the sub-problem; the truncation box of ``shapepost`` applies to the free coordinates only."""

    def __init__(self, base, index, value):
        self.base, self.index, self.value = base, int(index), float(value)
        self.dim = base.dim - 1
        self.lik_const = base.lik_const
        self.data = base.data
        self._free = np.delete(np.arange(base.dim), self.index)
        k, val = self.index, self.value

        def full(v):
            return jnp.concatenate([v[:k], jnp.full((1,), val, dtype=v.dtype), v[k:]])

        free = jnp.asarray(self._free)
        self._full = full
        self.log_prior = jax.jit(lambda v: base.log_prior(full(v)))

        def phi_and_grad(v, aux, data):
            phi, g, aux_new = base.phi_and_grad(full(v), aux, data)
            return phi, g[free], aux_new

        def phi(v, aux, data):
            return base.phi(full(v), aux, data)

        self.phi_and_grad = jax.jit(phi_and_grad)
        self.phi = jax.jit(phi)
        self._U_and_grad = None
        self._prior_hessian = None

    def initial_aux(self):
        return self.base.initial_aux()

    def full(self, v):
        """The completed vector (NumPy)."""
        return np.asarray(self._full(jnp.asarray(v, dtype=jnp.float64)))

    def fisher(self, v):
        F = self.base.fisher(self.full(v))
        return F[np.ix_(self._free, self._free)]


def profile_marginal(problem, index, u_hat, grid, max_iter=30, tol=1e-3, verbose=False, checkpoint=None):
    """Profile and Laplace-marginalize every coordinate except ``index`` of ``problem`` on ``grid`` (values of that coordinate): returns a dict with ``grid``, ``log_m`` (the log marginal
    density of the coordinate, unnormalized: the common constant lik_const is kept so the values are comparable between cases), ``phi`` (profile misfit), ``log_prior`` (full log prior at
    the profile point), ``logdet`` (log det H_v), ``converged``, ``v`` (the profiled free coordinates, (n, d - 1)), ``nit``.

    The grid is processed outward from the grid point nearest to ``u_hat[index]`` and each solve starts from the nearest solved neighbor (the first from the MAP's other coordinates).
    ``checkpoint`` (a path): after every grid point the finished points are written there as JSON; a rerun with the same grid reads them and solves only the missing ones.
    """
    grid = np.asarray(grid, dtype=np.float64)
    order = np.argsort(np.abs(grid - u_hat[index]))
    keys = ("log_m", "phi", "log_prior", "logdet", "converged", "v", "nit")
    res = {k: [None] * len(grid) for k in keys}
    solved = []  # (grid value, v)
    v_map = np.delete(np.asarray(u_hat, dtype=np.float64), index)
    if checkpoint is not None and Path(checkpoint).exists():
        saved = json.loads(Path(checkpoint).read_text())
        if len(saved["grid"]) == len(grid) and np.allclose(saved["grid"], grid, rtol=0, atol=1e-12):
            for pos in order:
                if saved["log_m"][pos] is not None:
                    for k in keys:
                        res[k][pos] = saved[k][pos]
                    solved.append((float(grid[pos]), np.asarray(saved["v"][pos], dtype=np.float64)))
            if verbose:
                print(f"  checkpoint: {len(solved)} of {len(grid)} grid points already solved")
    for pos in order:
        if res["log_m"][pos] is not None:
            continue
        t0 = time.perf_counter()
        u_g = float(grid[pos])
        sub = FixedCoordinate(problem, index, u_g)
        v0 = v_map if not solved else min(solved, key=lambda s: abs(s[0] - u_g))[1]
        v, info = sp.newton_map(sub, v0, max_iter=max_iter, tol=tol)
        H = sub.fisher(v) + sp.prior_hessian(sub, v)
        sign, logdet = np.linalg.slogdet(H)
        if sign <= 0:
            raise RuntimeError(f"H_v is not positive definite at u_g = {u_g}")
        lp = float(sub.log_prior(jnp.asarray(v)))
        res["log_m"][pos] = float(problem.lik_const - info["phi"] + lp + 0.5 * (problem.dim - 1) * log(2.0 * pi) - 0.5 * logdet)
        res["phi"][pos], res["log_prior"][pos], res["logdet"][pos] = info["phi"], lp, float(logdet)
        res["converged"][pos], res["v"][pos], res["nit"][pos] = bool(info["converged"]), v.tolist(), info["nit"]
        solved.append((u_g, v))
        if checkpoint is not None:
            prun.write_json(checkpoint, {"grid": grid.tolist(), **res})
        if verbose:
            print(f"  u_g {u_g:8.3f}: Phi {info['phi']:.2f}, log m {res['log_m'][pos]:.2f}, converged {info['converged']} ({info['nit']} its, {time.perf_counter() - t0:.0f} s)")
    out = {k: res[k] for k in res}
    out["grid"] = grid.tolist()
    return out


REFINE = 32  # points per grid cell of the fine grid on which the interpolated marginal is integrated


def marginal_statistics(grid, log_m, transform=None, edges=(0.05, 0.95)):
    """Quadrature statistics of the marginal density m(u) given by ``log_m`` on ``grid``: the normalization, the mass in the first and last grid cell (a check that the grid reaches the
    tails), the mean and sd of x = transform(u) (default u), and its ``edges`` quantiles.

    log m is interpolated by a cubic spline (smooth and positive, exact for the log of a Gaussian), the density is integrated on a grid ``REFINE`` times finer by the trapezoid rule
    and the CDF of u is read from that. Two cruder choices were wrong on the coarse profile grid of T4 (spacing 0.5 in u_g, a factor 1.65 in g): interpolating the CDF linearly in
    g = g_max sigmoid(u) put the 95 % point 15 % too high, and the trapezoid CDF on the grid points itself still 5 % (the marginal of u_g is a double exponential, e^u exp(-k e^u),
    with a width of 1). ``transform`` must be increasing (then the quantile of x is the transform of the quantile of u). Returns a dict that also holds ``u`` and ``x`` (the grid
    points), ``u_fine``, ``cdf_fine`` and the callables ``cdf_u`` (CDF of u inside the grid) and ``quantile_u``."""
    u = np.asarray(grid, dtype=np.float64)
    lm = np.asarray(log_m, dtype=np.float64)
    order = np.argsort(u)
    u, lm = u[order], lm[order]
    ident = (lambda t: t) if transform is None else transform
    n_fine = (len(u) - 1) * REFINE + 1
    u_fine = np.interp(np.arange(n_fine) / REFINE, np.arange(len(u)), u)  # REFINE points per cell (cells need not be equal)
    shift = lm.max()
    w = np.exp(CubicSpline(u, lm - shift)(u_fine))
    du = np.diff(u_fine)
    seg = 0.5 * (w[1:] + w[:-1]) * du
    total = seg.sum()
    cdf = np.concatenate([[0.0], np.cumsum(seg)]) / total
    x = np.asarray(ident(u_fine), dtype=np.float64)
    mean_x = float(np.sum(0.5 * (w[1:] * x[1:] + w[:-1] * x[:-1]) * du) / total)
    var_x = float(np.sum(0.5 * (w[1:] * x[1:] ** 2 + w[:-1] * x[:-1] ** 2) * du) / total - mean_x**2)

    def cdf_u(t):
        return np.interp(t, u_fine, cdf)

    def quantile_u(q):
        return float(np.interp(q, cdf, u_fine))

    q = {str(p): float(ident(np.asarray(quantile_u(p)))) for p in edges}
    return {"log_norm": float(shift + np.log(total)), "tail_mass_low": float(cdf[REFINE]), "tail_mass_high": float(1.0 - cdf[-1 - REFINE]), "mean": mean_x,
            "sd": float(np.sqrt(max(var_x, 0.0))), "quantiles": q, "u": u, "x": np.asarray(ident(u), dtype=np.float64), "u_fine": u_fine, "cdf_fine": cdf, "cdf_u": cdf_u, "quantile_u": quantile_u}


def cdf_at(stats, u):
    """CDF of the marginal at u (0 below the grid, 1 above it) from ``marginal_statistics``."""
    if u <= stats["u"][0]:
        return 0.0
    if u >= stats["u"][-1]:
        return 1.0
    return float(stats["cdf_u"](u))


def logit_gap(g, g_max):
    """u_g with g = g_max sigmoid(u_g); -inf for g <= 0."""
    return -np.inf if g <= 0.0 else float(np.log(g / (g_max - g)))


def gap_marginal(grid, log_m, g_max, g_true=None, g_det=None):
    """The marginal of the gap g = g_max sigmoid(u_g) from the marginal of u_g: the statistics of g (mean, sd, 90% interval), P(g < g_det) and P(g < g_true) from the CDF of u_g
    (``marginal_statistics``: interpolated in u_g, evaluated at logit(g / g_max)).

    The density of g is m(u_g) / (g (1 - g / g_max)); the mean of g uses the u_g marginal directly (``marginal_statistics`` with the sigmoid transform)."""
    stats = marginal_statistics(grid, log_m, transform=lambda u: g_max / (1.0 + np.exp(-u)))
    out = {k: v for k, v in stats.items() if k not in ("u", "x", "u_fine", "cdf_fine", "cdf_u", "quantile_u")}
    out["ci90"] = [stats["quantiles"]["0.05"], stats["quantiles"]["0.95"]]
    out["median"] = float(g_max / (1.0 + np.exp(-stats["quantile_u"](0.5))))
    if g_det is not None:
        out["g_det"] = float(g_det)
        out["p_below_g_det"] = cdf_at(stats, logit_gap(g_det, g_max))
    if g_true is not None:
        out["g_true"] = float(g_true)
        out["covers_true"] = bool(out["ci90"][0] <= g_true <= out["ci90"][1])
        out["p_below_g_true"] = cdf_at(stats, logit_gap(g_true, g_max))
    return out


def truncation_mass(grid, log_m, u_bound=sp.U_BOUND):
    """Posterior mass of u_g inside the box |u_g| <= ``u_bound`` of the HMC target (``shapepost.U_BOUND``), from the profile marginal on ``grid``: ``mass_inside`` and
    ``delta_log_z`` = log(mass_inside), the change of log Z of the truncated target against the posterior of the same coordinate. Needs a grid that covers the box or the support."""
    st = marginal_statistics(grid, log_m)
    lo, hi = max(-u_bound, st["u"][0]), min(u_bound, st["u"][-1])
    inside = cdf_at(st, hi) - cdf_at(st, lo)
    return {"u_bound": float(u_bound), "mass_inside": float(inside), "delta_log_z": float(np.log(inside)) if inside > 0 else float("-inf"),
            "grid_covers_box": bool(st["u"][0] <= -u_bound and st["u"][-1] >= u_bound)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--gap", type=float, default=0.1)
    ap.add_argument("--model", default="F2", choices=("F2", "F2u"))
    ap.add_argument("--eps-r", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--grid-lo", type=float, default=-24.0, help="lowest u_g of the grid (g = 0.4 sigmoid(u_g); -24 is g = 1.5e-11)")
    ap.add_argument("--grid-hi", type=float, default=-3.0, help="highest u_g (-3 is g = 0.019)")
    ap.add_argument("--grid-n", type=int, default=43)
    ap.add_argument("--around-map", type=float, default=None, help="instead of --grid-lo/--grid-hi: the MAP's u_g plus or minus this many Laplace standard deviations")
    ap.add_argument("--newton-iter", type=int, default=30)
    ap.add_argument("--out-dir", default=str(prun.RESULTS))
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pargs = prun.parse_args(["--truth", args.truth, "--gap", str(args.gap), "--model", args.model, "--out-dir", args.out_dir, "--n", str(args.n), "--data-factor", str(args.data_factor),
                             "--side", str(args.side), "--snr", str(args.snr), "--seed", str(args.seed), "--n-tx", str(args.n_tx), "--n-rx", str(args.n_rx), "--tol", str(args.tol),
                             "--eps-r", str(args.eps_r), "--radius", str(args.radius)])
    d = prun.case_dir(pargs)
    path = d / "profile.json"
    if path.exists() and not args.force:
        print(f"{prun.case_name(pargs)}: profile done")
        return json.loads(path.read_text())
    rec, z = prun.load_map(pargs)
    prob = prun.build_problem(pargs)
    post = prun.make_posterior(pargs, prob)
    u_hat = np.asarray(z["u_hat"])
    if args.around_map is not None:
        sd = float(rec["laplace"]["sd_u"][0])
        grid = np.linspace(u_hat[0] - args.around_map * sd, u_hat[0] + args.around_map * sd, args.grid_n)
    else:
        grid = np.linspace(args.grid_lo, args.grid_hi, args.grid_n)
    t0 = time.perf_counter()
    partial = d / "profile_partial.json"
    prof = profile_marginal(post, 0, u_hat, grid, max_iter=args.newton_iter, verbose=True, checkpoint=partial)
    gap = gap_marginal(prof["grid"], prof["log_m"], sm.G_MAX, g_true=prob.g_true, g_det=prob.g_det)
    trunc = truncation_mass(prof["grid"], prof["log_m"])
    # the profile minimum over the grid in terms of g, for the table
    i_best = int(np.argmin(prof["phi"]))
    out = {"case": prun.case_name(pargs), "g_true": prob.g_true, "g_det": prob.g_det, "u_hat_map": u_hat.tolist(), "g_map": float(post.physical(u_hat)[0]), "phi_map": rec["phi_map"],
           "grid_u_g": prof["grid"], "grid_g": (sm.G_MAX / (1.0 + np.exp(-np.asarray(prof["grid"])))).tolist(), "profile": prof, "gap": gap, "truncation": trunc, "phi_profile_min": float(prof["phi"][i_best]),
           "u_g_profile_min": float(prof["grid"][i_best]), "all_converged": bool(all(prof["converged"])), "time_s": time.perf_counter() - t0, "thread_env": prun.thread_env()}
    prun.write_json(path, out)
    partial.unlink(missing_ok=True)
    print(f"{out['case']}: g mean {gap['mean']:.4g}, sd {gap['sd']:.3g}, 90% interval [{gap['ci90'][0]:.4g}, {gap['ci90'][1]:.4g}], P(g < g_det) {gap.get('p_below_g_det')}, "
          f"tail mass low/high {gap['tail_mass_low']:.2e}/{gap['tail_mass_high']:.2e}, mass inside |u_g| <= {trunc['u_bound']:g}: {trunc['mass_inside']:.3g}, all converged {out['all_converged']}; "
          f"{out['time_s']:.0f} s")
    return out


if __name__ == "__main__":
    main()
