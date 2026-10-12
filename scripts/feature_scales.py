"""The three DESIGN 11.3 feature probabilities at several spatial scales, per chain and per block of draws, and a Laplace check without MCMC.

    python scripts/feature_scales.py RUN [RUN ...] [--dir results/prior_comparison] [--radii 0 0.08 0.12] [--block 100]
        [--laplace RUN [--laplace-draws 4000]] [--out FILE.json]

DESIGN 11.3 defines the features through the value of theta at single points (bilinear interpolation). A posterior with pixel-scale roughness (the
GMRF, pixel sd 0.34) puts the variance of a single pixel into a single-point feature, so its probability can be low although the data fix the
feature at the resolution of the experiment (milestone-5 review). For every run (summary.json and chains.npz must exist) and every radius in
``--radii`` this reports the probabilities of ``features.feature_probabilities`` with ``radius`` = 0 (the DESIGN point values) or the mean of
theta over the disk of that radius about each point (0.08 wavelength: half the ring thickness, 2 pixels at n = 48, about 12 pixels in the disk;
0.12: about 26 pixels; both well below the half wavelength of the resolution limit):

  * the pooled probability, the per-chain probabilities (``p_chain``; the chain-to-chain RANGE is quoted, not a standard error: the GMRF chains
    are not converged, see convergence_audit.json), and the probabilities in consecutive blocks of ``--block`` draws per chain (pooled over
    chains), whose range over blocks is the honest scatter of an unconverged run;
  * with ``--laplace RUN`` the same probabilities for ``--laplace-draws`` exact draws of the Laplace approximation at the MAP of RUN,
    N(theta_hat, H^-1) (the prepared.npz of RUN; H = H_GN + Hess R; theta = theta_hat + L^-T z, z ~ N(0, I)): a posterior approximation that
    needs no MCMC, so it is free of the convergence question (its non-Gaussianity is what it ignores).

Writes FILE.json (default <dir>/feature_scales.json). The Laplace part imports JAX (one XLA thread of the caller's NPROC); the rest is NumPy.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

from bayes_scatter import features as ft
from bayes_scatter.geometry import Grid

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("gap_valley", "ring_hole", "ring_continuity")


def probabilities(theta, grid, radius):
    """``{feature: {"p", "p_chain"}}`` for draws ``theta`` (chains, N, n, n) at ``radius``."""
    f = ft.feature_probabilities(theta, grid, radius=radius)
    return {k: {"p": f[k]["p"], "p_chain": f[k]["p_chain"]} for k in FEATURES}


def block_probabilities(theta, grid, radius, block):
    """``{feature: [P in block 0, P in block 1, ...]}``: the probabilities pooled over chains within consecutive blocks of ``block`` draws
    per chain (a trailing block shorter than ``block`` is dropped)."""
    nb = theta.shape[1] // block
    ind = ft.feature_indicators(theta[:, :nb * block], grid, radius=radius)
    return {k: [float(v[:, i * block:(i + 1) * block].mean()) for i in range(nb)] for k, v in ind.items()}


def spread(values):
    """``{"min", "max"}`` of a list of floats."""
    return {"min": float(np.min(values)), "max": float(np.max(values))}


def analyze(run, base, radii, block):
    """The dict of the module docstring for one run directory."""
    d = Path(base) / run
    summ = json.loads((d / "summary.json").read_text())
    theta = np.load(d / "chains.npz")["theta"].astype(np.float64)
    grid = Grid(summ["args"]["n"], summ["args"]["side"])
    out = {"run": run, "converged": summ["convergence"]["converged"], "chains": int(theta.shape[0]), "draws_per_chain": int(theta.shape[1]),
           "radii": {}}
    for r in radii:
        pr = probabilities(theta, grid, r)
        bl = block_probabilities(theta, grid, r, block) if theta.shape[1] >= block else {}
        out["radii"][f"{r:g}"] = {
            "pixels_in_disk": int((np.hypot(*(a - c for a, c in zip(grid.centers(), ft.austria_points(grid).gap_mid))) <= r).sum()) if r > 0 else 1,
            "probabilities": {k: {"p": v["p"], "p_chain": v["p_chain"], "chain_range": spread(v["p_chain"]),
                                  "blocks": bl.get(k), "block_range": spread(bl[k]) if bl.get(k) else None} for k, v in pr.items()}}
    return out


def laplace_analysis(run, base, radii, n_draws, seed=0):
    """Feature probabilities of ``n_draws`` exact draws of the Laplace approximation N(theta_hat, H^-1) at the MAP of ``run``."""
    from scipy.linalg import solve_triangular

    import run_prior_comparison as rpc  # imports JAX

    d = Path(base) / run
    prep_args = json.loads((d / "prepared.json").read_text())["args"]
    ns = argparse.Namespace(**prep_args)
    ctx = rpc.build_context(ns)
    prep = rpc.load_prepared(ctx, d)
    L = prep.mass.chol
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((L.shape[0], n_draws))
    theta = prep.theta_hat.reshape(-1, 1) + solve_triangular(L, z, lower=True, trans="T")  # theta = theta_hat + L^-T z, covariance H^-1
    theta = theta.T.reshape(1, n_draws, *prep.theta_hat.shape)
    grid = ctx.grid
    out = {"run": run, "draws": int(n_draws), "seed": seed, "laplace_pixel_sd_mean": float(theta[0].std(axis=0).mean()), "radii": {}}
    for r in radii:
        pr = probabilities(theta, grid, r)
        out["radii"][f"{r:g}"] = {k: v["p"] for k, v in pr.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--radii", type=float, nargs="+", default=[0.0, 0.08, 0.12], help="disk radii in wavelengths; 0 = the DESIGN point values")
    ap.add_argument("--block", type=int, default=100, help="draws per chain in a block")
    ap.add_argument("--laplace", nargs="*", default=[], metavar="RUN", help="runs whose MAP Laplace approximation is also evaluated")
    ap.add_argument("--laplace-draws", type=int, default=4000)
    ap.add_argument("--out", default=None, help="default <dir>/feature_scales.json")
    args = ap.parse_args(argv)
    res = {r: analyze(r, args.dir, args.radii, args.block) for r in args.runs}
    if args.laplace:
        res["laplace"] = {r: laplace_analysis(r, args.dir, args.radii, args.laplace_draws) for r in args.laplace}
    p = Path(args.out) if args.out else Path(args.dir) / "feature_scales.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    for r in args.runs:
        for rad, e in res[r]["radii"].items():
            pr = e["probabilities"]
            print(f"{r:24s} radius {rad:>5s}: " + "  ".join(f"{k} {pr[k]['p']:.3f} (chains {pr[k]['chain_range']['min']:.2f}-{pr[k]['chain_range']['max']:.2f})" for k in FEATURES))
    for r, e in res.get("laplace", {}).items():
        for rad, pr in e["radii"].items():
            print(f"Laplace at the MAP of {r}: radius {rad:>5s}: " + "  ".join(f"{k} {pr[k]:.3f}" for k in FEATURES))
    print(f"wrote {p}")
    return res


if __name__ == "__main__":
    main()
