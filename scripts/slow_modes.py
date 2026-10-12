"""The slowest directions of an adapted run, in the coordinates of its own map: the diagnosis behind scripts/readapt_map.py (docs/DESIGN.md 12.5).

    python scripts/slow_modes.py RUN [RUN ...] [--dir results/prior_comparison] [--skip-segments K] [--top 8] [--out FILE.json]

An adapted run records theta in the coordinates zeta = A^-1 (theta - c) of its frozen map. If the map whitened the posterior, zeta would be
N(0, I) over the recorded draws: |zeta - mean|^2 / 2 would be about N / 2 per draw and every eigenvalue of the covariance of zeta would be 1. For
every run this reads the segment files, maps the draws to zeta (a triangular solve with the stored Cholesky factor) and reports

  * ``mean_half_sq_dev`` : the mean over draws of |zeta - mean|^2 / 2 (against N / 2) and its trace per coordinate;
  * ``eigenvalues_top`` and ``eigenvalue_quantiles``: the largest eigenvalues of the pooled covariance of zeta and its quantiles (50, 90, 99 %);
  * for the top ``--top`` principal directions v_k of the pooled draws: the mean of the projection zeta . v_k of each chain, the within-chain
    standard deviation, the means of the first and of the second half of every chain (a drift shows as a difference), and the classic split
    R-hat of the projection (one chain count and draw count for all, ``diagnostics.split_rhat``).

An eigenvalue far above 1 is a direction in which the chain moves in a trajectory of length 1.5 (in these units) a distance far below its
standard deviation: about (sigma / 1.5)^2 trajectories per independent draw. The first GMRF map had eigenvalues of 910, 740, 600, 330 (standard
deviations of 30 in map units) and chains that were still drifting after 300 draws. ``--skip-segments`` leaves out the first K segments of
every chain (burn-in). ``--figure FILE.png`` also draws, for every run, the four leading principal directions of the pooled draws in
theta space (the spatial pattern of the directions that dominate the variance) and the posterior mean of theta. Writes FILE.json (default
<dir>/slow_modes.json). One process, numpy and matplotlib only.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.linalg import solve_triangular

import bayes_scatter  # noqa: F401
from bayes_scatter import diagnostics as dg

ROOT = Path(__file__).resolve().parents[1]


def zeta_draws(run_dir, skip_segments=0, chains=None):
    """``(Z (chains, S, N) float64, N)``: the recorded draws of the run in ``run_dir`` in the coordinates of its map ``adapted_map.npz``
    (chains with the same number of segments, the first ``skip_segments`` left out)."""
    run_dir = Path(run_dir)
    z = np.load(run_dir / "adapted_map.npz")
    center, chol = z["center"].ravel(), z["chol"]
    per_chain = []
    c = 0
    while (run_dir / f"chain_{c}_seg0.npz").exists() and (chains is None or c < chains):
        segs, j = [], 0
        while (run_dir / f"chain_{c}_seg{j}.npz").exists():
            segs.append(np.load(run_dir / f"chain_{c}_seg{j}.npz")["theta"].astype(np.float64).reshape(-1, center.size))
            j += 1
        per_chain.append(segs)
        c += 1
    k = min(len(s) for s in per_chain)
    if skip_segments >= k:
        raise SystemExit(f"--skip-segments {skip_segments} leaves no segment (the chains have {k})")
    Z = [solve_triangular(chol, (np.concatenate(s[skip_segments:k]) - center).T, lower=True).T for s in per_chain]
    return np.stack(Z), center.size


def slow_mode_report(Z, top=8):
    """The dict of the module docstring for ``Z`` (chains, S, N)."""
    C, S, N = Z.shape
    mu = Z.mean(axis=(0, 1))
    X = (Z - mu).reshape(C * S, N)
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    ev = sv**2 / (C * S - 1)
    h = S // 2
    out = {"chains": int(C), "draws_per_chain": int(S), "dim": int(N),
           "mean_half_sq_dev": float(0.5 * (X**2).sum(axis=1).mean()), "half_dim": N / 2.0,
           "variance_per_coordinate": float(ev.sum() / N),
           "eigenvalues_top": [float(e) for e in ev[:top]],
           "eigenvalue_quantiles": {"50%": float(np.quantile(ev, 0.5)), "90%": float(np.quantile(ev, 0.9)), "99%": float(np.quantile(ev, 0.99))},
           "directions": []}
    for k in range(min(top, vt.shape[0])):
        p = Z @ vt[k]  # (C, S)
        out["directions"].append({
            "eigenvalue": float(ev[k]), "chain_means": [float(m) for m in p.mean(axis=1)],
            "within_chain_sd": [float(s) for s in p.std(axis=1, ddof=1)],
            "first_half_means": [float(m) for m in p[:, :h].mean(axis=1)], "second_half_means": [float(m) for m in p[:, h:].mean(axis=1)],
            "split_rhat": float(dg.split_rhat(p[..., None])[0])})
    return out


def theta_pca_figure(run_dirs, path, skip_segments=0, n_modes=4):
    """Figure ``path``: one row per run with the ``n_modes`` leading principal directions of the pooled theta draws (n x n images, symmetric
    color scale) and the posterior mean of theta; the title of a direction gives its variance (theta units^2 per unit-norm direction)."""
    runs = [Path(r) for r in run_dirs]
    fig = Figure(figsize=(2.8 * (n_modes + 1), 2.8 * len(runs)), layout="constrained")
    axs = np.atleast_2d(fig.subplots(len(runs), n_modes + 1))
    for row, rd in enumerate(runs):
        Z, N = zeta_draws(rd, skip_segments)
        m = np.load(rd / "adapted_map.npz")
        theta = (Z @ m["chol"].T + m["center"].ravel()).reshape(-1, N)
        X = theta - theta.mean(axis=0)
        _, sv, vt = np.linalg.svd(X, full_matrices=False)
        ev = sv**2 / (len(X) - 1)
        side = int(round(np.sqrt(N)))
        for k in range(n_modes):
            v = vt[k].reshape(side, side)
            axs[row, k].imshow(v, cmap="RdBu_r", vmin=-np.abs(v).max(), vmax=np.abs(v).max())
            axs[row, k].set_title(f"{rd.name} PC{k + 1}: var {ev[k]:.2f}", fontsize=8)
        axs[row, n_modes].imshow(theta.mean(axis=0).reshape(side, side), cmap="viridis")
        axs[row, n_modes].set_title(f"{rd.name}: mean theta", fontsize=8)
        for a in axs[row]:
            a.axis("off")
    fig.savefig(path, dpi=70)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--skip-segments", type=int, default=0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figure", default=None, help="write the theta-space principal-direction figure to this PNG")
    args = ap.parse_args(argv)
    res = {}
    for r in args.runs:
        Z, _ = zeta_draws(Path(args.dir) / r, args.skip_segments)
        res[r] = slow_mode_report(Z, args.top)
        e = res[r]
        print(f"{r}: {e['chains']} x {e['draws_per_chain']} draws, mean |zeta - mean|^2/2 = {e['mean_half_sq_dev']:.0f} (N/2 = {e['half_dim']:.0f}); "
              f"top eigenvalues {', '.join(f'{x:.1f}' for x in e['eigenvalues_top'][:6])}; median {e['eigenvalue_quantiles']['50%']:.2f}, "
              f"99% {e['eigenvalue_quantiles']['99%']:.1f}; split R-hat of the top projections "
              f"{', '.join(f'{d['split_rhat']:.2f}' for d in e['directions'][:5])}")
    out = Path(args.out) if args.out else Path(args.dir) / "slow_modes.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    if args.figure:
        theta_pca_figure([Path(args.dir) / r for r in args.runs], args.figure, args.skip_segments)
        print(f"wrote {args.figure}")
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
