"""The resolution-scale gap-valley feature of the two-disk object (docs/DESIGN.md 13.3, decision of 2026-10-12).

A posterior over theta = log eps_r on the n x n inference grid ([i, j] = [x, y], C-order flattening, N = n^2) is scored on whether the
gap between the two disks of ``shapes.TwoDisks`` is a valley, with averages over regions at the resolution scale and never single
pixels:

    gap valley:  mean theta over the gap strip  <  min(mean theta over core 1, mean theta over core 2) - delta,   delta = log(eps_r) / 2,

half of the true jump log(eps_r) of theta between a disk and the vacuum. Regions (pixel centers; ``FeatureGeometry``):

* strip: the pixels whose centers lie strictly between the facing edges of the disks (|x - x_mid| < g / 2, x_mid = cx + dr / 2) and
  within the disk-height band (|y - cy| <= band_frac min(r1, r2); ``band_frac`` = 1/2 is the headline (decision of 2026-10-13), 1 the lenient full band. Away from y = cy the circles bend apart, so
  for a gap of a few pixels the strip over the full band also contains the lens-shaped void above and below the narrowest point: a smaller
  ``band_frac`` tests the gap proper, at the price of fewer pixels);
* core i: the pixels whose centers lie within ``core_frac`` (default 1/2) of the radius of disk i from its center.

If the gap is narrower than one pixel (g < h) or no pixel center lies in the strip, the feature is "unrepresentable on this grid" and no
probability is reported (``FeatureGeometry.fits_grid``). It is also unrepresentable when the PIXELIZED TRUTH (the exact-coverage rendering of the object itself
on the inference grid) fails the feature (``FeatureGeometry.truth_holds``): a posterior probability of a valley that the grid cannot show even for the true object
is not a probability of the gap (milestone-6 review; e.g. eps_r 4, g = 0.025, a strip of a quarter of the disk height: its pixels are about half disk). ``representable`` is both
conditions; ``status`` says which one failed. The single-point version (``point`` weights: the bilinear interpolation of theta at the
gap midpoint against the disk centers) is computed alongside and must be labeled as such: it is exposed to pixel-scale roughness
and is not the feature.

Every functional above is LINEAR in theta, a weight vector w (N,) with w . theta, so under a Gaussian (Laplace) posterior
N(m, H^-1) the three means (strip, core 1, core 2) are jointly Gaussian with mean w_j . m and covariance w_j^T H^-1 w_k (one solve per
weight vector, no dense inverse: ``gaussian_functionals``), and the probability of the gap valley is the probability that
X = (strip - core 1 + delta, strip - core 2 + delta) is negative in both components: a bivariate normal orthant probability
(``orthant_probability``, scipy's Genz routine at a tolerance of 1e-9; checked in the tests against Monte Carlo). For MCMC draws
the same weights give the indicator of every draw (``valley_indicator``).
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import erf
from scipy.stats import multivariate_normal

CORE_FRAC = 0.5  # core radius as a fraction of the disk radius
BAND_FRAC = 0.5  # strip half-height as a fraction of min(r1, r2): the headline feature (decision of 2026-10-13); 1.0 is the lenient bound
UNREPRESENTABLE = "unrepresentable on this grid"
UNREPRESENTABLE_TRUTH = "unrepresentable on this grid (the pixelized truth fails the feature)"


def is_unrepresentable(status):
    """True for both ``region_status`` strings of an unrepresentable feature (``UNREPRESENTABLE``, ``UNREPRESENTABLE_TRUTH``)."""
    return str(status).startswith(UNREPRESENTABLE)


def delta_of(eps_r):
    """The feature threshold delta = log(eps_r) / 2 (half the jump of theta = log eps_r between a disk and the vacuum)."""
    return 0.5 * float(np.log(eps_r))


def bilinear_weights(grid, x, y):
    """Weight vector (N,) of the bilinear interpolation of a pixel field at the point (x, y): theta(x, y) = w . theta.ravel().

    The four surrounding pixel centers carry the usual bilinear weights; points outside the lattice of pixel centers are clamped to
    it, as in ``features.bilinear`` (which this reproduces).
    """
    n, h = grid.n, grid.h
    cx, cy = grid.center
    fx = float(np.clip((x - (cx - 0.5 * grid.side + 0.5 * h)) / h, 0.0, n - 1.0))
    fy = float(np.clip((y - (cy - 0.5 * grid.side + 0.5 * h)) / h, 0.0, n - 1.0))
    i0, j0 = min(int(np.floor(fx)), n - 2), min(int(np.floor(fy)), n - 2)
    ax, ay = fx - i0, fy - j0
    w = np.zeros((n, n))
    w[i0, j0] += (1 - ax) * (1 - ay)
    w[i0 + 1, j0] += ax * (1 - ay)
    w[i0, j0 + 1] += (1 - ax) * ay
    w[i0 + 1, j0 + 1] += ax * ay
    return w.ravel()


def _mean_weights(mask):
    """Uniform weights (N,) over the True entries of ``mask`` (n, n); all zero if the mask is empty."""
    w = mask.astype(np.float64).ravel()
    s = w.sum()
    return w / s if s > 0 else w


@dataclass(frozen=True)
class FeatureGeometry:
    """Regions and weight vectors of the gap-valley feature of one object on one grid (module docstring).

    Attributes: ``strip``, ``core1``, ``core2`` ((n, n) bool masks); ``n_strip``, ``n_core1``, ``n_core2`` pixel counts;
    ``strip_width_pixels`` = g / h; ``representable`` (bool); ``weights`` (dict of (N,) arrays: ``strip``, ``core1``, ``core2`` uniform
    means, and ``mid``, ``c1``, ``c2`` bilinear point weights at the gap midpoint and the two disk centers); ``delta``.
    """

    grid: object
    obj: object
    core_frac: float
    band_frac: float
    strip: np.ndarray
    core1: np.ndarray
    core2: np.ndarray
    weights: dict
    delta: float

    @property
    def n_strip(self):
        return int(self.strip.sum())

    @property
    def n_core1(self):
        return int(self.core1.sum())

    @property
    def n_core2(self):
        return int(self.core2.sum())

    @property
    def strip_width_pixels(self):
        return self.obj.g / self.grid.h

    @property
    def fits_grid(self):
        """False if the gap is narrower than one pixel or the regions hold no pixel center: no resolution-scale probability exists."""
        return bool(self.obj.g >= self.grid.h and self.n_strip > 0 and self.n_core1 > 0 and self.n_core2 > 0)

    @property
    def truth_holds(self):
        """Whether the pixelized truth of ``obj`` (``TwoDisks.theta(grid)``, exact coverage) satisfies the feature of THIS geometry (strip and cores, ``delta``);
        None if the geometry does not fit the grid. When it does not hold, no posterior can be expected to show the valley at this strip and the probability is
        not reported."""
        if not self.fits_grid:
            return None
        th = np.asarray(self.obj.theta(self.grid), dtype=np.float64).ravel()
        w = self.weights
        return bool(w["strip"] @ th < min(w["core1"] @ th, w["core2"] @ th) - self.delta)

    @property
    def representable(self):
        """True if a region probability is reported: the geometry fits the grid (``fits_grid``) and the pixelized truth shows the valley (``truth_holds``)."""
        return bool(self.fits_grid and self.truth_holds)

    @property
    def status(self):
        """"ok", "unrepresentable on this grid" (the geometry does not fit) or "unrepresentable on this grid (the pixelized truth fails the feature)"."""
        if not self.fits_grid:
            return UNREPRESENTABLE
        return "ok" if self.truth_holds else UNREPRESENTABLE_TRUTH
    def describe(self):
        """Plain-dict summary (counts, widths, delta, representability) for JSON files."""
        return {"n_strip": self.n_strip, "n_core1": self.n_core1, "n_core2": self.n_core2, "strip_width_pixels": self.strip_width_pixels,
                "fits_grid": self.fits_grid, "truth_holds": self.truth_holds, "representable": self.representable, "status": self.status, "delta": self.delta, "core_frac": self.core_frac, "band_frac": self.band_frac,
                "gap_mid": [self.obj.cx + 0.5 * self.obj.dr, self.obj.cy], "pixel": self.grid.h}


def feature_geometry(grid, obj, core_frac=CORE_FRAC, band_frac=BAND_FRAC):
    """``FeatureGeometry`` of the object ``obj`` (``shapes.TwoDisks``) on ``grid`` (``geometry.Grid``); ``core_frac`` and ``band_frac`` as in the module docstring."""
    X, Y = grid.centers()
    (x1, y1, r1), (x2, y2, r2) = obj.left, obj.right
    x_mid = 0.5 * ((x1 + r1) + (x2 - r2))
    y_mid = 0.5 * (y1 + y2)
    band = band_frac * min(r1, r2)
    strip = (np.abs(X - x_mid) < 0.5 * obj.g) & (np.abs(Y - y_mid) <= band)
    core1 = np.hypot(X - x1, Y - y1) <= core_frac * r1
    core2 = np.hypot(X - x2, Y - y2) <= core_frac * r2
    weights = {"strip": _mean_weights(strip), "core1": _mean_weights(core1), "core2": _mean_weights(core2),
               "mid": bilinear_weights(grid, x_mid, y_mid), "c1": bilinear_weights(grid, x1, y1), "c2": bilinear_weights(grid, x2, y2)}
    return FeatureGeometry(grid, obj, float(core_frac), float(band_frac), strip, core1, core2, weights, delta_of(obj.eps_r))


# ----------------------------------------------------------------------------------------------
# Gaussian (Laplace) posterior
# ----------------------------------------------------------------------------------------------


def gaussian_functionals(W, mean, solve):
    """Mean vector and covariance matrix of the linear functionals W theta under theta ~ N(mean, H^-1).

    ``W`` (K, N), ``mean`` (N,) (or (n, n)), ``solve(B)`` returns H^-1 B for B (N, K) (e.g. two triangular solves with the Cholesky
    factor of H). Returns ``(m (K,), S (K, K))`` with m = W mean and S = W H^-1 W^T, symmetrized.
    """
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    m = W @ np.asarray(mean, dtype=np.float64).ravel()
    S = W @ np.asarray(solve(W.T))
    return m, 0.5 * (S + S.T)


def orthant_probability(mean, cov):
    """P(X_1 < 0, ..., X_K < 0) for X ~ N(mean, cov), K = 1 or 2 (a bivariate normal orthant probability by Genz's routine in scipy at
    an absolute tolerance of 1e-9). A degenerate component (variance 0) is treated as a point mass."""
    mean = np.atleast_1d(np.asarray(mean, dtype=np.float64))
    cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
    if mean.shape[0] == 1:
        sd = float(np.sqrt(cov[0, 0]))
        return float(0.5 * (1.0 + erf(-mean[0] / (sd * np.sqrt(2.0))))) if sd > 0 else float(mean[0] < 0)
    p = multivariate_normal.cdf(np.zeros_like(mean), mean=mean, cov=cov, allow_singular=True, abseps=1e-9, releps=1e-9,
                                maxpts=200000 * mean.shape[0])
    return float(np.clip(p, 0.0, 1.0))


def valley_statistics(m, S, delta):
    """The gap-valley probability from the Gaussian functionals of (strip, core 1, core 2): ``m`` (3,), ``S`` (3, 3).

    P(strip < core 1 - delta and strip < core 2 - delta). Returns a dict: ``p_valley``, the two marginal probabilities ``p_vs_core1`` and
    ``p_vs_core2``, the means ``strip``, ``core1``, ``core2`` with their standard deviations, ``depth_mean`` / ``depth_sd`` of
    (core1 + core2) / 2 - strip (the average valley depth; the feature needs it to exceed delta plus the core imbalance) and
    ``z_depth`` = (depth_mean - delta) / depth_sd.
    """
    A = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
    mx, Sx = A @ m + delta, A @ S @ A.T
    depth = np.array([-1.0, 0.5, 0.5])
    d_mean, d_var = float(depth @ m), float(depth @ S @ depth)
    return {"p_valley": orthant_probability(mx, Sx),
            "p_vs_core1": orthant_probability(mx[:1], Sx[:1, :1]), "p_vs_core2": orthant_probability(mx[1:], Sx[1:, 1:]),
            "strip_mean": float(m[0]), "strip_sd": float(np.sqrt(S[0, 0])), "core1_mean": float(m[1]), "core1_sd": float(np.sqrt(S[1, 1])),
            "core2_mean": float(m[2]), "core2_sd": float(np.sqrt(S[2, 2])), "depth_mean": d_mean, "depth_sd": float(np.sqrt(d_var)),
            "z_depth": float((d_mean - delta) / np.sqrt(d_var))}


def gaussian_valley(fg, mean, solve):
    """Gap-valley results of a Gaussian posterior N(mean, H^-1) for ``fg`` (a ``FeatureGeometry``): a JSON-serializable dict.

    ``region``: ``valley_statistics`` of the region means, or ``None`` with ``region_status`` "unrepresentable on this grid" when the
    strip is narrower than a pixel; ``point``: the same for the bilinear point values, labeled (single-point values carry pixel-scale
    roughness; they are not the resolution-scale feature). ``solve`` is as in ``gaussian_functionals``.
    """
    w = fg.weights
    out = {"geometry": fg.describe(), "delta": fg.delta}
    if fg.representable:
        m, S = gaussian_functionals(np.stack([w["strip"], w["core1"], w["core2"]]), mean, solve)
        out["region"], out["region_status"] = valley_statistics(m, S, fg.delta), "ok"
    else:
        out["region"], out["region_status"] = None, fg.status
    m, S = gaussian_functionals(np.stack([w["mid"], w["c1"], w["c2"]]), mean, solve)
    out["point"] = valley_statistics(m, S, fg.delta)
    out["point_label"] = "single-point values (bilinear interpolation at the gap midpoint and the disk centers), not the resolution-scale feature"
    return out


# ----------------------------------------------------------------------------------------------
# Draws and the truth
# ----------------------------------------------------------------------------------------------


def valley_indicator(theta, weights_strip, weights_core1, weights_core2, delta):
    """Indicator (bool, ``theta.shape[:-2]``) of strip < min(core 1, core 2) - delta for theta (..., n, n) and weight vectors (N,)."""
    th = np.asarray(theta, dtype=np.float64)
    flat = th.reshape(th.shape[:-2] + (-1,))
    s, c1, c2 = flat @ weights_strip, flat @ weights_core1, flat @ weights_core2
    return s < np.minimum(c1, c2) - delta


def draws_valley(fg, theta):
    """Gap-valley probability from posterior draws ``theta`` (chains, draws, n, n) or (draws, n, n): region and point versions.

    Returns ``{"region": {"p_valley", "per_chain"}, "point": {...}, "region_status"}`` (``region`` None if unrepresentable); ``per_chain`` is
    the list of per-chain probabilities (a one-element list for a single sequence of draws).
    """
    th = np.asarray(theta, dtype=np.float64)
    if th.ndim == 3:
        th = th[None]
    w = fg.weights

    def summarize(ws, wc1, wc2):
        ind = valley_indicator(th, ws, wc1, wc2, fg.delta)
        return {"p_valley": float(ind.mean()), "per_chain": [float(x) for x in ind.mean(axis=1)], "n_draws": int(ind.size)}

    out = {"point": summarize(w["mid"], w["c1"], w["c2"])}
    if fg.representable:
        out["region"], out["region_status"] = summarize(w["strip"], w["core1"], w["core2"]), "ok"
    else:
        out["region"], out["region_status"] = None, fg.status
    return out


def truth_valley(fg, theta_true):
    """The feature evaluated on the truth theta_true (n, n) as rendered on the inference grid: region and point means and whether
    strip < min(core means) - delta holds. If it does not, the grid cannot show the valley at all (the pixelized truth does not have it)."""
    th = np.asarray(theta_true, dtype=np.float64).ravel()
    w = fg.weights
    out = {}
    for label, keys in (("point", ("mid", "c1", "c2")), ("region", ("strip", "core1", "core2"))):
        s, c1, c2 = (float(w[k] @ th) for k in keys)
        out[label] = {"strip": s, "core1": c1, "core2": c2, "holds": bool(s < min(c1, c2) - fg.delta)}
    if not fg.fits_grid:
        out["region"]["holds"] = None
    return out
