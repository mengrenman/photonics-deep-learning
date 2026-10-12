"""Where the chains started and whether they were still drifting: block means of |zeta|^2 / 2 and of the prior energy (docs/DESIGN.md 12.6).

    python scripts/start_drift_audit.py RUN [RUN ...] [--dir results/prior_comparison] [--block 100] [--out FILE.json]

The milestone-5 report quoted per-chain agreement and R-hat of Phi as evidence of convergence, but none of the runs started from an overdispersed state (DESIGN 12.2
asks for that): the GMRF chains started at the end states of the stuck milestone-4 chains, the TV chains at 0.5 N(0, I) in the coordinates of the adapted map around
ONE state of the adaptation chain (a manual launch, because --init-scale 1.0 gave a physical Phi of 4e4 to 8e4), and the re-adapted runs at the final states of the
first runs' chains. This script reads the run directories and reports, per run:

  * the ``starts`` description stored in the first segment file of every chain, and the physical Phi and prior energy R of the first recorded draw of every chain;
  * the ring bias of every chain (the mean over the pixels at d = -1 of theta - truth, ``edge_analysis.signed_distance``): of the START of the chain (``starts.npy``, if
    the run has one), of its first recorded draw, of the means of consecutive blocks of ``--block`` draws, and of the draws after the first ``--burn`` (default 50);
    with the split R-hat (classic, ``diagnostics.split_rhat``) of the ring-bias series and of the physical Phi over the chains after the burn-in. Chains that
    start at different places (truth: bias 0; MAP; Matern mean; a rough GMRF draw) and end at the same ring bias are the independent-start evidence;
  * for blocks of ``--block`` recorded draws, the mean over each chain of |zeta|^2 / 2 (zeta = L^-1 (theta - c) in the coordinates of the adapted map, c the map
    center, L the Cholesky factor of its covariance; N / 2 = 1152 for a posterior that the map whitens), the physical Phi, and the prior energy R = Phi' + |zeta|^2 / 2 - Phi
    (U = Phi + R and the sampler's Phi' = U - |zeta|^2 / 2; ``chains.npz`` stores Phi and Phi'), and the first-block-to-last-block change. A chain that has not
    reached its stationary distribution shows a trend in these block means.

Needs ``adapted_map.npz`` (adapted runs only). NumPy and SciPy only. Writes FILE.json (default <dir>/m5/review/start_drift.json).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from bayes_scatter import diagnostics as dg
from edge_analysis import signed_distance

ROOT = Path(__file__).resolve().parents[1]


def half_sq_norm(theta, center, chol):
    """|zeta|^2 / 2 of draws ``theta`` (S, n, n) in the coordinates of the map theta = center + chol zeta: (S,)."""
    r = theta.reshape(theta.shape[0], -1).astype(np.float64) - center[None, :]
    z = solve_triangular(chol, r.T, lower=True)
    return 0.5 * (z * z).sum(axis=0)


def block_means(x, block):
    """Means of ``x`` (S,) over consecutive blocks of ``block`` draws (a short trailing block is dropped): list of floats."""
    nb = len(x) // block
    return [float(x[i * block:(i + 1) * block].mean()) for i in range(nb)]


def ring_bias(theta, truth, dist):
    """Mean over the pixels at signed distance -1 of ``theta - truth`` for draws ``theta`` (..., n, n): shape (...)."""
    return (np.asarray(theta, dtype=np.float64)[..., dist == -1] - truth[dist == -1]).mean(axis=-1)


def ring_report(run_dir, theta, phi, truth, dist, block, burn):
    """The ring-bias part of the audit (module docstring) for chains ``theta`` (C, S, n, n) and ``phi`` (C, S)."""
    rb = ring_bias(theta, truth, dist)  # (C, S)
    starts = np.load(run_dir / "starts.npy") if (run_dir / "starts.npy").exists() else None
    post = rb[:, burn:]
    out = {"burn": burn, "chains": [], "ring_bias_after_burn_mean_per_chain": [float(x) for x in post.mean(axis=1)]}
    for c in range(theta.shape[0]):
        out["chains"].append({"chain": c, "start_ring_bias": None if starts is None or c >= len(starts) else float(ring_bias(starts[c], truth, dist)),
                              "first_draw_ring_bias": float(rb[c, 0]), "ring_bias_blocks": block_means(rb[c], block),
                              "ring_bias_after_burn": float(post[c].mean())})
    if theta.shape[0] >= 2 and post.shape[1] >= 8:
        out["split_rhat_ring_bias_after_burn"] = float(dg.split_rhat(post[:, :, None])[0])
        out["split_rhat_phi_after_burn"] = float(dg.split_rhat(phi[:, burn:][:, :, None])[0])
    out["between_chain_sd_of_ring_bias_after_burn"] = float(post.mean(axis=1).std(ddof=1)) if theta.shape[0] >= 2 else None
    return out


def audit_run(run, base, block=100, burn=50):
    """The dict of the module docstring for one run directory ``base / run``."""
    d = Path(base) / run
    z = np.load(d / "chains.npz")
    amap = np.load(d / "adapted_map.npz")
    center, chol = amap["center"].astype(np.float64), amap["chol"]
    theta, phi, phi_prime = z["theta"], z["phi_physical"].astype(np.float64), z["phi_prime"].astype(np.float64)
    ring = ring_report(d, theta, phi, z["theta_true"], signed_distance(z["chi_true"]), block, burn)
    chains = []
    for c in range(theta.shape[0]):
        hs = half_sq_norm(theta[c], center, chol)
        R = phi_prime[c] + hs - phi[c]
        seg0 = np.load(d / f"chain_{c}_seg0.npz")
        hb, rb, pb = block_means(hs, block), block_means(R, block), block_means(phi[c], block)
        chains.append({"chain": c, "starts": str(seg0["starts"]), "first_draw_phi": float(phi[c][0]), "first_draw_R": float(R[0]),
                       "half_sq_zeta_blocks": hb, "R_blocks": rb, "phi_blocks": pb,
                       "half_sq_zeta_first_to_last": [hb[0], hb[-1]] if hb else None, "R_first_to_last": [rb[0], rb[-1]] if rb else None})
    return {"run": run, "draws_per_chain": int(theta.shape[1]), "block": block, "half_N": 0.5 * center.size, "chains": chains, "ring": ring,
            "R_blocks_mean_over_chains": [float(np.mean(v)) for v in zip(*[c["R_blocks"] for c in chains])] if chains[0]["R_blocks"] else [],
            "half_sq_zeta_blocks_mean_over_chains": [float(np.mean(v)) for v in zip(*[c["half_sq_zeta_blocks"] for c in chains])]
            if chains[0]["half_sq_zeta_blocks"] else []}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--block", type=int, default=100)
    ap.add_argument("--burn", type=int, default=50, help="draws discarded per chain before the 'after burn-in' ring-bias statistics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = {r: audit_run(r, args.dir, args.block, args.burn) for r in args.runs}
    p = Path(args.out) if args.out else Path(args.dir) / "m5" / "review" / "start_drift.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    for r, e in res.items():
        print(f"{r}: N/2 = {e['half_N']:.0f}; R by block (mean over chains) {np.round(e['R_blocks_mean_over_chains'], 0).tolist()}")
        rg = e["ring"]
        print(f"  ring bias (d = -1) after {rg['burn']} draws: per chain {np.round(rg['ring_bias_after_burn_mean_per_chain'], 3).tolist()}, "
              f"split R-hat of the ring bias {rg.get('split_rhat_ring_bias_after_burn')}, of Phi {rg.get('split_rhat_phi_after_burn')}")
        for rc in rg["chains"]:
            print(f"    chain {rc['chain']}: start {rc['start_ring_bias']}, first draw {rc['first_draw_ring_bias']:.3f}, blocks {np.round(rc['ring_bias_blocks'], 3).tolist()}")
        for c in e["chains"]:
            print(f"  chain {c['chain']}: |zeta|^2/2 first -> last block {c['half_sq_zeta_first_to_last']}; R {c['R_first_to_last']}; start: {c['starts'][:90]}")
    print(f"wrote {p}")
    return res


if __name__ == "__main__":
    main()
