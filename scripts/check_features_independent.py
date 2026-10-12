"""Independent recomputation of the three DESIGN 11.3 feature probabilities from the stored draws of runs (docs/DESIGN.md 12.5).

    python scripts/check_features_independent.py RUN [RUN ...] [--dir results/prior_comparison] [--out FILE.json]

``features.feature_probabilities`` gives the probabilities in every ``summary.json``. This script recomputes them from ``chains.npz`` WITHOUT
``bayes_scatter.features``: the interpolation is ``scipy.ndimage.map_coordinates`` (order 1), and the geometry of the austria_fixed phantom is
written out from the DESIGN text (s = side / 5, disks of radius 0.5 s at (+-0.9 s, +1.0 s), ring radii 0.9 s and 1.3 s about (0, -0.8 s), pixel
centers at -side / 2 + (i + 1/2) side / n, theta[..., i, j] with i along x and j along y, delta = log(2) / 2). The two must agree to rounding; a
difference would mean that the feature code, not the sampler, is wrong. Also reports what ring continuity would be if the four compass values were
independent: the product of the four marginal probabilities P(theta_k > delta), once with Gaussian marginals and once with the empirical marginals,
and the largest absolute correlation between two compass values. The Gaussian marginal of the interpolated value at a compass point uses the mean
and the standard deviation OF THE INTERPOLATED DRAWS; the bilinear interpolation of the pixel standard deviations, which an earlier version of this
script used, is NOT the standard deviation of the interpolated value (the interpolation averages four partly independent pixel values) and for a
rough posterior it overstates it, which made the Gaussian product 0.35 where the sampled probability is 0.50 (review of milestone 5). Writes FILE.json
(default <dir>/independent_features.json).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]


def recompute(theta, n, side):
    """``dict`` of the three probabilities and the Gaussian ring-continuity value for draws ``theta`` (S, n, n) on the n x n grid of side ``side``."""
    h = side / n
    xc = -side / 2 + h * (np.arange(n) + 0.5)

    def interp(t, x, y):  # t (..., n, n) -> (...,)
        fx, fy = (x - xc[0]) / h, (y - xc[0]) / h
        flat = t.reshape(-1, n, n)
        return np.array([map_coordinates(a, [[fx], [fy]], order=1, mode="nearest")[0] for a in flat]).reshape(t.shape[:-2])

    s = side / 5.0
    disks = [(-0.9 * s, 1.0 * s), (0.9 * s, 1.0 * s)]
    ring = (0.0, -0.8 * s)
    r_in, r_out = 0.9 * s, 1.3 * s
    r_mid = 0.5 * (r_in + r_out)
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    rho = np.hypot(X - ring[0], Y - ring[1])
    annulus = (rho > r_in) & (rho <= r_out)
    delta = np.log(2.0) / 2.0
    t = np.asarray(theta, dtype=np.float64)
    gap = interp(t, 0.5 * (disks[0][0] + disks[1][0]), 0.5 * (disks[0][1] + disks[1][1]))
    disk_min = np.minimum(interp(t, *disks[0]), interp(t, *disks[1]))
    pts = [(ring[0] + r_mid, ring[1]), (ring[0], ring[1] + r_mid), (ring[0] - r_mid, ring[1]), (ring[0], ring[1] - r_mid)]
    comp = np.stack([interp(t, *q) for q in pts])
    marginals = [1.0 - norm.cdf((delta - c.mean()) / c.std(ddof=1)) for c in comp]  # Gaussian marginals of the INTERPOLATED draws
    empirical = [float(np.mean(c > delta)) for c in comp]
    corr = np.corrcoef(comp)
    offdiag = np.abs(corr[~np.eye(len(pts), dtype=bool)])
    return {"gap_valley": float(np.mean(gap < disk_min - delta)),
            "ring_hole": float(np.mean(interp(t, *ring) < t[:, annulus].mean(axis=1) - delta)),
            "ring_continuity": float(np.mean(np.all(comp > delta, axis=0))), "annulus_pixels": int(annulus.sum()),
            "ring_continuity_gaussian_independent_points": float(np.prod(marginals)),
            "compass_marginals_gaussian": [float(m) for m in marginals],
            "ring_continuity_empirical_marginals_product": float(np.prod(empirical)), "compass_marginals_empirical": empirical,
            "compass_max_abs_correlation": float(offdiag.max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = {}
    for r in args.runs:
        d = Path(args.dir) / r
        summ = json.loads((d / "summary.json").read_text())
        th = np.load(d / "chains.npz")["theta"].astype(np.float64)
        a = summ["args"]
        out = recompute(th.reshape(-1, *th.shape[2:]), a["n"], a["side"])
        ref = summ["features"]["probabilities"]
        out["summary"] = {k: ref[k]["p"] for k in ("gap_valley", "ring_hole", "ring_continuity")}
        out["max_abs_difference"] = max(abs(out[k] - out["summary"][k]) for k in out["summary"])
        res[r] = out
        print(f"{r:24s} independent {out['gap_valley']:.4f} {out['ring_hole']:.4f} {out['ring_continuity']:.4f}   summary "
              f"{out['summary']['gap_valley']:.4f} {out['summary']['ring_hole']:.4f} {out['summary']['ring_continuity']:.4f}   "
              f"max diff {out['max_abs_difference']:.1e}; independent-points continuity: Gaussian {out['ring_continuity_gaussian_independent_points']:.3f}, "
              f"empirical marginals {out['ring_continuity_empirical_marginals_product']:.3f}; max |corr| of compass values {out['compass_max_abs_correlation']:.2f}")
    p = Path(args.out) if args.out else Path(args.dir) / "independent_features.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"wrote {p}")
    return res


if __name__ == "__main__":
    main()
