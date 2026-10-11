"""Posterior summaries for the prior-sensitivity experiment (docs/DESIGN.md section 11.3).

Everything works on theta = log eps_r maps of shape (n, n) ("ij" indexing, [i, j] = [x, y]) on a ``geometry.Grid``.

* ``pooled_moments``: mean and standard deviation of pooled chains from per-chain Welford mean and variance.
* ``edge_mask`` and ``z_statistics``: pixels within one pixel of a material edge of a phantom, and the |z| > 3
  fraction (all pixels, inside, outside) and the share of those pixels lying near an edge, with
  z = (posterior mean - truth) / posterior std in log eps_r.
* ``disagreement``: the pixelwise two-prior statistic |m_A - m_B| / sqrt(s_A^2 + s_B^2).
* ``austria_points`` and ``feature_probabilities``: the three feature probabilities of DESIGN 11.3 for the
  ``phantoms.austria_fixed`` object, computed on theta draws. The value of theta at a point is the bilinear
  interpolation of the four surrounding pixel-center values (``bilinear``), so the points need not sit on pixel
  centers; it is linear in the draws, so a draw of the interpolated value is the interpolation of the draw.
"""

from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

DELTA = float(np.log(2.0) / 2.0)  # the feature threshold of DESIGN 11.3: half the true edge jump log 2


def pooled_moments(mean_c, var_c):
    """Mean and standard deviation of the pooled chains from per-chain moments.

    ``mean_c``, ``var_c``: (chains, ...) per-chain mean and (unbiased) variance over all post-warmup steps (the
    Welford output of ``samplers``). The pooled variance is the mean of the within-chain variances plus the variance
    of the chain means (equal-length chains). Returns ``(mean, std)`` with the trailing shape.
    """
    mean_c, var_c = np.asarray(mean_c, dtype=np.float64), np.asarray(var_c, dtype=np.float64)
    return mean_c.mean(axis=0), np.sqrt(var_c.mean(axis=0) + mean_c.var(axis=0))


def merge_moments(means, variances, counts):
    """Mean and unbiased variance of the concatenation of consecutive segments of ONE chain.

    ``means``, ``variances``: sequences of per-segment arrays of one common shape (running mean and unbiased variance
    over all steps of the segment, as the Welford output of ``samplers``); ``counts``: the number of steps of each
    segment. Returns ``(mean, var)`` of all the steps together: the pooled mean sum(c_k m_k) / C and
    ((c_k - 1) v_k summed + c_k (m_k - mean)^2 summed) / (C - 1) (the standard parallel-variance combination, exact).
    """
    c = np.asarray(counts, dtype=np.float64)
    m = np.stack([np.asarray(x, dtype=np.float64) for x in means])
    v = np.stack([np.asarray(x, dtype=np.float64) for x in variances])
    w = c.reshape((-1,) + (1,) * (m.ndim - 1))
    total = c.sum()
    mean = (w * m).sum(axis=0) / total
    m2 = ((w - 1.0) * v).sum(axis=0) + (w * (m - mean) ** 2).sum(axis=0)
    return mean, m2 / (total - 1.0)


def edge_mask(chi_true):
    """Pixels within one pixel of a material edge of a phantom: the dilation of the object (chi > 0) minus the erosion
    of its full-contrast core, so partially covered boundary pixels and their 8-neighbors count. (n, n) bool."""
    inside = chi_true > 1e-9
    full = chi_true > (chi_true.max() - 1e-9)
    return binary_dilation(inside, iterations=1) & ~binary_erosion(full, iterations=1)


def z_statistics(mean, std, truth, chi_true):
    """The miscoverage statistics of the report (REPORT section 7.2) for log eps_r maps (n, n) each.

    z = (mean - truth) / std. Returns a dict of plain floats: ``frac_abs_z_gt_3`` (all pixels), ``frac_inside``,
    ``frac_outside`` (inside = chi_true > 0, boundary pixels included), ``edge_share`` (the share of the |z| > 3
    pixels that lie within one pixel of an edge, NaN if there are none), ``edge_fraction_of_doi`` (the share of the
    DOI that is near an edge, what a chance edge share would be), ``max_abs_z`` and ``rms``.
    """
    z = (np.asarray(mean) - np.asarray(truth)) / np.asarray(std)
    big = np.abs(z) > 3.0
    inside = chi_true > 1e-9
    edge = edge_mask(chi_true)
    return {"frac_abs_z_gt_3": float(big.mean()),
            "frac_inside": float(big[inside].mean()) if inside.any() else float("nan"),
            "frac_outside": float(big[~inside].mean()) if (~inside).any() else float("nan"),
            "edge_share": float((big & edge).sum() / big.sum()) if big.any() else float("nan"),
            "edge_fraction_of_doi": float(edge.mean()), "max_abs_z": float(np.abs(z).max()),
            "rms": float(np.sqrt(np.mean(z**2)))}


def disagreement(mean_a, std_a, mean_b, std_b):
    """|m_A - m_B| / sqrt(s_A^2 + s_B^2) per pixel; values above 2 mark pixels where two posteriors disagree."""
    return np.abs(np.asarray(mean_a) - np.asarray(mean_b)) / np.sqrt(np.asarray(std_a) ** 2 + np.asarray(std_b) ** 2)


# ----------------------------------------------------------------------------------------------
# Features of the austria_fixed phantom
# ----------------------------------------------------------------------------------------------


def bilinear(theta, grid, x, y):
    """theta interpolated bilinearly at the point (x, y): theta (..., n, n) -> (...). Points outside the pixel-center
    lattice are clamped to it."""
    cx, cy = grid.center
    n, h = grid.n, grid.h
    fx = np.clip((x - (cx - 0.5 * grid.side + 0.5 * h)) / h, 0.0, n - 1.0)
    fy = np.clip((y - (cy - 0.5 * grid.side + 0.5 * h)) / h, 0.0, n - 1.0)
    i0, j0 = min(int(np.floor(fx)), n - 2), min(int(np.floor(fy)), n - 2)
    ax, ay = fx - i0, fy - j0
    t = np.asarray(theta)
    return ((1 - ax) * (1 - ay) * t[..., i0, j0] + ax * (1 - ay) * t[..., i0 + 1, j0]
            + (1 - ax) * ay * t[..., i0, j0 + 1] + ax * ay * t[..., i0 + 1, j0 + 1])


def austria_points(grid):
    """Geometry of ``phantoms.austria_fixed`` on ``grid`` and the points of the three features (DESIGN 11.3).

    Returns a namespace: ``disk_centers`` (two points), ``disk_radius``, ``ring_center``, ``ring_radii`` (inner,
    outer), ``gap_mid`` (midpoint between the two disk centers, i.e. the middle of the gap), ``compass`` (the four
    points of the annulus mid-radius to the east, north, west, south of the ring center) and ``annulus`` (an (n, n)
    boolean mask of the pixels whose centers lie in the annulus, inner radius < rho <= outer radius). Lengths are
    those of ``austria_fixed``: disks of radius 0.5 s at (+-0.9 s, +1.0 s), ring radii 0.9 s and 1.3 s about
    (0, -0.8 s), s = side / 5, offset by the grid center.
    """
    s = grid.side / 5.0
    cx, cy = grid.center
    disks = [(cx - 0.9 * s, cy + 1.0 * s), (cx + 0.9 * s, cy + 1.0 * s)]
    ring_c = (cx, cy - 0.8 * s)
    r_in, r_out = 0.9 * s, 1.3 * s
    r_mid = 0.5 * (r_in + r_out)
    X, Y = grid.centers()
    rho = np.hypot(X - ring_c[0], Y - ring_c[1])
    return SimpleNamespace(
        disk_centers=disks, disk_radius=0.5 * s, ring_center=ring_c, ring_radii=(r_in, r_out),
        gap_mid=(0.5 * (disks[0][0] + disks[1][0]), 0.5 * (disks[0][1] + disks[1][1])),
        compass=[(ring_c[0] + r_mid, ring_c[1]), (ring_c[0], ring_c[1] + r_mid),
                 (ring_c[0] - r_mid, ring_c[1]), (ring_c[0], ring_c[1] - r_mid)],
        annulus=(rho > r_in) & (rho <= r_out))


def feature_indicators(theta, grid, delta=DELTA):
    """The three feature indicators on theta maps (..., n, n); returns a dict of boolean arrays of shape (...).

    * ``gap_valley``: theta at the gap midpoint < min(theta at the two disk centers) - delta;
    * ``ring_hole``: theta at the ring center < mean of theta over the annulus pixels - delta;
    * ``ring_continuity``: theta at all four compass points of the annulus mid-radius > delta.
    """
    pts = austria_points(grid)
    t = np.asarray(theta, dtype=np.float64)
    gap = bilinear(t, grid, *pts.gap_mid)
    disk_min = np.minimum(bilinear(t, grid, *pts.disk_centers[0]), bilinear(t, grid, *pts.disk_centers[1]))
    ring_mean = t[..., pts.annulus].mean(axis=-1)
    compass = np.stack([bilinear(t, grid, *p) for p in pts.compass], axis=-1)
    return {"gap_valley": gap < disk_min - delta, "ring_hole": bilinear(t, grid, *pts.ring_center) < ring_mean - delta,
            "ring_continuity": np.all(compass > delta, axis=-1)}


def feature_probabilities(theta_draws, grid, delta=DELTA):
    """Posterior probability of each feature from draws ``theta_draws`` (chains, N, n, n).

    Returns ``{feature: {"p": pooled probability, "p_chain": per-chain probabilities, "mcse": std of the chain
    estimates / sqrt(chains)}}`` (the chain-to-chain spread is the honest Monte Carlo error of a probability from
    autocorrelated draws; with one chain ``mcse`` is NaN).
    """
    ind = feature_indicators(theta_draws, grid, delta)
    out = {}
    for name, v in ind.items():
        per_chain = v.mean(axis=1)
        out[name] = {"p": float(v.mean()), "p_chain": [float(x) for x in per_chain],
                     "mcse": float(per_chain.std(ddof=1) / np.sqrt(len(per_chain))) if len(per_chain) > 1
                     else float("nan")}
    return out


# The DESIGN 8.7 minimum for a "data-determined" verdict: structurally different priors, at least a Matern field and a GMRF.
REQUIRED_FAMILIES = ("matern", "gmrf")


def verdict(probabilities, converged, family=None, required_families=REQUIRED_FAMILIES, threshold=0.95):
    """Classify a feature from its probability under several posteriors (DESIGN 11.3, 8.7).

    ``probabilities``: dict run name -> P; ``converged``: dict run name -> bool; ``family``: dict run name -> prior
    family (``"matern"``, ``"gmrf"``, ``"qggmrf"``; default: every run is its own family). Different parameter settings
    of one family (two q-GGMRF values of sigma_g) count as ONE prior: they are not structurally different priors.
    Only converged runs vote. The result is one of

    * ``"no verdict (fewer than 2 converged runs)"``;
    * ``"not determined"``: the converged runs do not all give P > ``threshold`` (present) or all P < 1 - ``threshold``
      (absent);
    * ``"not established: ..."``: the converged runs agree, but an UNCONVERGED run gives the opposite side of the
      threshold. An unconverged chain cannot confirm a feature, but it can block a verdict; a sound reading needs it to
      converge;
    * ``"provisional (present)"`` / ``"provisional (absent)"``: the converged runs agree and no unconverged run
      disagrees, but the DESIGN minimum comparison is missing (some family of ``required_families`` has no converged
      run); the text names the families that agree;
    * ``"data-determined (present)"`` / ``"data-determined (absent)"``: as above and every required family has a
      converged run.
    """
    fam = (lambda name: family.get(name, name)) if family else (lambda name: name)
    conv = [name for name in probabilities if converged.get(name, False)]
    if len(conv) < 2:
        return "no verdict (fewer than 2 converged runs)"
    ps = [probabilities[name] for name in conv]
    lo = 1.0 - threshold
    if all(p > threshold for p in ps):
        side, bad = "present", [n for n in probabilities if not converged.get(n, False) and probabilities[n] <= threshold]
    elif all(p < lo for p in ps):
        side, bad = "absent", [n for n in probabilities if not converged.get(n, False) and probabilities[n] >= lo]
    else:
        return "not determined"
    if bad:
        return f"not established: unconverged {', '.join(sorted(fam(n) for n in bad))} disagrees with the converged runs ({side})"
    families = sorted({fam(n) for n in conv})
    missing = [f for f in required_families if f not in families]
    if missing:
        return f"provisional ({side}): agreed by {', '.join(families)} only; the DESIGN minimum needs a converged {', '.join(missing)}"
    return f"data-determined ({side})"
