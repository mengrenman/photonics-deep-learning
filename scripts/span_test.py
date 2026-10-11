"""Span test of the book's NMM (docs/DESIGN.md section 11.4; Ch. 15, eqs. 15.1-15.6, Table 15.3).

Why the test exists. The NMM output is o = alpha_hat g_r(.) (eq. 15.1) with alpha_hat = T G_r^+ (eq. 15.6), where
T = [t_1, ..., t_P] (N x P) holds the P training permittivity images, so o = T (G_r^+ g_r(.)) lies in span(T). Nothing
is added after alpha_hat: the thresholds b (first layer) and p (second layer) enter inside the activations g_c and
g_r, and eq. 15.1 has no output bias. The best output the NMM could produce for a test scene m is therefore the
least-squares projection Pi_T m onto span(T), and its error

    bound(m) = ||Pi_T m - m|| / ||m||        (eq. 15.21 with m = eps_r, background included)

lower-bounds the NMM column of Table 15.3 (3.26-3.31 %). It does NOT bound the IEM column (1.03-1.09 %): the IEM is a
CNN applied after the NMM and is not confined to span(T). The identity is verified on a toy ELM in
tests/test_span_test.py.

What is computed (all errors in percent of ||m||, m = eps_r = 1 + chi on Grid(96, 5.0), N = 9216, [i, j] = [x, y]
order flattened in C order):

  (a) 1000 held-out draws of phantoms.ch15_austria_variant (Table 15.2) against R independent training sets of
      P = 100 (the book), under the painter's rule (later shapes overwrite) and the max-overlap rule (larger eps_r wins
      where shapes overlap), and against P = 25 ... 3200; plus a tail scan of 20000 fresh draws, because the minimum
      over 1000 draws is set by the single easiest draw;
  (b) Fig. 15.9-like scenes (hand-digitized from the PDF raster, see SCENES): Test #1 = disk + rectangle with
      eps_r <= 2, Test #4 = three overlapping disks; 8 named instantiations of each (plus one high-contrast Test #4
      outside the figure's colorbar) and 200 jittered variants of the digitized geometry per type;
  extras for every case: nearest-training-image error min_j ||t_j - m|| / ||m||, the error of the vacuum guess
  eps_r = 1 (||chi|| / ||m||), of the training-mean image, of span(T u {1}) (what an output bias could buy; NOT the
  book's model), and of a contrast-trained NMM (project chi on span of the training chi, output 1 + projection); and
  the bound of the empty scene m = 1, the part of the error that a mostly-vacuum scene cannot avoid.

Numerics. The projection uses Householder QR with a rank check (SVD fallback for rank-deficient T) and one
reorthogonalization pass; the normal equations are never formed. Independent implementation (section "Implementation
B", numpy only, no bayes_scatter import): its own sampler of Table 15.2 (Philox stream, rejection sampling for
r4 > r3), sub-pixel rendering on a 6x finer grid composed on eps_r and block-averaged, its own scene renderer, and SVD
projection. cross_check compares the two: identical inputs through QR, gelsd and SVD; the same shapes through both
rasterizers; and independent draws by z-scores that include the test-sample variance. The verdict is stored in the JSON
(`agree`); use --no-cross-check to skip it. The binary-mask variant (pixel-center masks, no antialiasing) is run with
implementation B.

  (c) other readings of the NMM (``run_variants``, "variants" in the JSON; added after the review): the training
      permittivity law U[1, 8] (Table 15.2), U[1, 3] and U[1, 2] (the colorbar of Fig. 15.9), each with the plain span, the
      span plus a constant, the in-span output with an oracle shift clipped to the colorbar [1, 2], and the span plus a
      constant clipped; the chi-normalized error; a check that a squared misfit contradicts the book's BA and VBIM numbers
      (``metric_reading_checks``); and ``heldout_min_diagnosis``: the pooled minimum of the held-out errors is the easiest
      value of ONE draw, so the report quotes the median and the low percentiles, not the minimum.

Writes results/span_test/span_test.json, span_test.npz (raw error arrays) and span_test.png.
Thread cap: at most 2 BLAS threads (set before numpy is imported).

Usage:  python scripts/span_test.py [--out DIR] [--quick] [--replicates 40] [--n-test 1000] [--seed 0]
        python scripts/span_test.py --figure-only [--out DIR]   (redraw the figure from the stored JSON / NPZ)
        python scripts/span_test.py --variants-only [--out DIR]  (add the review variants to the stored JSON / NPZ)
"""

import os

MAX_THREADS = 2
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    try:
        _cur = int(os.environ.get(_var, MAX_THREADS))
    except ValueError:
        _cur = MAX_THREADS
    os.environ[_var] = str(max(1, min(MAX_THREADS, _cur)))

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg

import bayes_scatter  # noqa: F401
from bayes_scatter.geometry import Grid
from bayes_scatter.phantoms import ch15_austria_variant, coverage, paint, paint_max

ROOT = Path(__file__).resolve().parents[1]

# Table 15.3 of the book, model misfit in percent (eq. 15.21 on eps_r), Tests #1-#4
BOOK_NMM_PCT = (3.2615, 3.3103, 3.2916, 3.2812)
BOOK_IEM_PCT = (1.0346, 1.0951, 1.0415, 1.0864)
BOOK_BA_PCT = (4.8230, 8.2708, 12.7513, 13.0708)  # Table 15.3, the same tests, the conventional methods
BOOK_VBIM_PCT = (2.0181, 3.6295, 5.5959, 4.0328)
THRESH_PCT = 3.3  # DESIGN 11.4: report the fraction of cases with bound <= 3.3 %

RULES = ("painter", "max")

# --------------------------------------------------------------------------------------------------------------
# Fig. 15.9-like scenes. Lengths in meters (= wavelengths) on the book's 5 m DOI centered at the origin; shapes are
# ("disk", cx, cy, r, eps_r) or ("rect", cx, cy, lx, ly, eps_r), painted in list order.
#
# The geometry of the "digitized" scenes was read from the ground-truth column of Fig. 15.9 (PDF page 482, 400 dpi
# raster): the panel spans -2.5..2.5 m on 264 px; shapes were located by fitting circles / boxes to the color
# regions and eps_r by matching the parula colors to the colorbar (1..2). Accuracy about +-0.05 m and +-0.1 in
# eps_r. Test #1: a disk near (0.05, 1.0), r 0.64, eps_r 1.3, and a flat rectangle near (0.05, -1.55), 1.56 x 0.44,
# eps_r 1.2. Test #4: three overlapping disks of r about 0.42-0.45 near the origin with eps_r about 1.4 (blue, top),
# 1.6 (green, left), 1.8 (orange, right); where they overlap the figure shows orange over green over blue, which is
# consistent with both the painter's rule (painted in that order) and the max-overlap rule. All colors of all four
# rows of the figure lie within the colorbar 1..2, so eps_r <= 2 holds for Test #4 as well. The other instantiations
# move, resize and re-contrast the digitized ones to cover plausible alternatives.
# --------------------------------------------------------------------------------------------------------------
SCENES = {
    "test1": [
        {"name": "T1-a digitized", "shapes": [("disk", 0.05, 1.00, 0.64, 1.3), ("rect", 0.05, -1.55, 1.56, 0.44, 1.2)]},
        {"name": "T1-b small, stronger", "shapes": [("disk", 0.0, 1.0, 0.50, 1.5), ("rect", 0.0, -1.5, 1.00, 0.30, 1.3)]},
        {"name": "T1-c large disk", "shapes": [("disk", 0.0, 0.9, 0.75, 1.4), ("rect", 0.0, -1.5, 1.60, 0.50, 1.2)]},
        {"name": "T1-d off-center", "shapes": [("disk", -0.5, 0.5, 0.60, 1.3), ("rect", 0.3, -1.2, 1.20, 0.40, 1.2)]},
        {"name": "T1-e eps_r up to 2", "shapes": [("disk", 0.5, 1.2, 0.60, 2.0), ("rect", -0.5, -1.4, 1.50, 0.40, 1.5)]},
        {"name": "T1-f large, central", "shapes": [("disk", 0.0, 0.5, 0.80, 1.6), ("rect", 0.0, -1.6, 2.00, 0.50, 1.4)]},
        {"name": "T1-g side by side", "shapes": [("disk", -0.8, 0.0, 0.50, 1.2), ("rect", 0.8, 0.0, 0.40, 1.50, 1.3)]},
        {"name": "T1-h very small", "shapes": [("disk", 0.0, 1.0, 0.35, 1.5), ("rect", 0.0, -1.5, 0.80, 0.20, 1.3)]},
    ],
    "test4": [
        {"name": "T4-a digitized", "shapes": [("disk", 0.00, 0.38, 0.42, 1.4), ("disk", -0.35, 0.00, 0.42, 1.6),
                                              ("disk", 0.33, -0.02, 0.45, 1.8)]},
        {"name": "T4-b reversed order", "shapes": [("disk", 0.33, -0.02, 0.45, 1.8), ("disk", -0.35, 0.00, 0.42, 1.6),
                                                   ("disk", 0.00, 0.38, 0.42, 1.4)]},
        {"name": "T4-c large triangle", "shapes": [("disk", 0.0, 0.5, 0.60, 1.5), ("disk", -0.5, -0.3, 0.60, 1.7),
                                                   ("disk", 0.5, -0.3, 0.60, 1.9)]},
        {"name": "T4-d small tight", "shapes": [("disk", 0.0, 0.2, 0.30, 1.3), ("disk", -0.2, -0.1, 0.30, 1.5),
                                                ("disk", 0.2, -0.1, 0.30, 1.7)]},
        {"name": "T4-e displaced", "shapes": [("disk", 0.6, 0.9, 0.45, 1.4), ("disk", 0.2, 0.5, 0.45, 1.6),
                                              ("disk", 0.9, 0.5, 0.45, 1.8)]},
        {"name": "T4-f eps_r = 2", "shapes": [("disk", -0.3, 0.3, 0.50, 2.0), ("disk", 0.3, 0.3, 0.50, 2.0),
                                              ("disk", 0.0, -0.2, 0.50, 2.0)]},
        {"name": "T4-g chain", "shapes": [("disk", -0.6, 0.0, 0.50, 1.3), ("disk", 0.0, 0.0, 0.50, 1.6),
                                          ("disk", 0.6, 0.0, 0.50, 1.9)]},
        {"name": "T4-h large", "shapes": [("disk", 0.0, 0.4, 0.80, 1.2), ("disk", -0.5, -0.3, 0.80, 1.5),
                                          ("disk", 0.5, -0.3, 0.80, 1.8)]},
    ],
    # outside the colorbar of Fig. 15.9; the text says the contrast increases from Test #1 to #4
    "test4_high": [
        {"name": "T4-i high contrast (2, 4, 6)", "shapes": [("disk", 0.00, 0.38, 0.42, 2.0), ("disk", -0.35, 0.00, 0.42, 4.0),
                                                              ("disk", 0.33, -0.02, 0.45, 6.0)]},
    ],
}
JITTER_BASE = {"test1": SCENES["test1"][0]["shapes"], "test4": SCENES["test4"][0]["shapes"]}


# --------------------------------------------------------------------------------------------------------------
# Implementation A: package generator, QR projection
# --------------------------------------------------------------------------------------------------------------


def _inside(shape, scale):
    """Indicator function inside(X, Y) of a ("disk", cx, cy, r, eps), ("rect", cx, cy, lx, ly, eps) or
    ("ring", cx, cy, r_in, r_out, eps) shape, lengths multiplied by ``scale``."""
    if shape[0] == "disk":
        _, cx, cy, r, _ = shape
        cx, cy, r = scale * cx, scale * cy, scale * r
        return lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
    if shape[0] == "rect":
        _, cx, cy, lx, ly, _ = shape
        cx, cy, lx, ly = scale * cx, scale * cy, scale * lx, scale * ly
        return lambda X, Y: (np.abs(X - cx) <= 0.5 * lx) & (np.abs(Y - cy) <= 0.5 * ly)
    if shape[0] == "ring":
        _, cx, cy, r_in, r_out, _ = shape
        cx, cy, r_in, r_out = scale * cx, scale * cy, scale * r_in, scale * r_out

        def inside(X, Y):
            rho2 = (X - cx) ** 2 + (Y - cy) ** 2
            return (rho2 >= r_in**2) & (rho2 <= r_out**2)

        return inside
    raise ValueError(f"unknown shape {shape[0]!r}")


def paint_scene(grid, shapes, overlap="painter"):
    """eps_r image (n, n) of a scene, "ij" indexing, built like the training phantoms: coverage-averaged shapes
    (``phantoms.coverage``, 4 x 4 sub-sampling) blended with ``paint`` ("painter": later shapes overwrite) or
    ``paint_max`` ("max": larger contrast wins). Lengths scale with grid.side / 5."""
    put = {"painter": paint, "max": paint_max}[overlap]
    scale = grid.side / 5.0
    chi = np.zeros((grid.n, grid.n))
    for s in shapes:
        chi = put(chi, coverage(grid, _inside(s, scale)), s[-1] - 1.0)
    return 1.0 + chi


def random_images(rng, count, grid, overlap, eps_range=(1.0, 8.0)):
    """``count`` independent Table 15.2 draws as eps_r = 1 + chi, shape (count, N) with N = n^2, C-order [i, j]
    flattening. Consecutive draws come from one stream, so sets drawn with the same seed are nested.
    ``eps_range`` is the law of the permittivity of each shape (the book's table: [1, 8])."""
    out = np.empty((count, grid.n * grid.n))
    for k in range(count):
        out[k] = (1.0 + ch15_austria_variant(rng, grid, overlap, eps_range)).ravel()
    return out


def basis_of_span(T):
    """Orthonormal basis of the column space of T (N x P). Returns (Q, info) with Q of shape (N, r).

    Householder QR; if the diagonal of R shows (near) rank deficiency, or P > N, the basis comes from the SVD, keeping
    singular values above max(N, P) eps s_1. ``info`` holds the rank, the path taken and min|R_ii| / max|R_ii|."""
    T = np.asarray(T, dtype=np.float64)
    N, P = T.shape
    tol = max(N, P) * np.finfo(np.float64).eps
    if P <= N:
        Q, R = np.linalg.qr(T)
        d = np.abs(np.diag(R))
        ratio = float(d.min() / d.max()) if d.max() > 0 else 0.0
        if ratio > 1e3 * tol:
            return Q, {"rank": P, "path": "qr", "r_diag_ratio": ratio}
    else:
        ratio = float("nan")
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    r = int(np.sum(s > tol * s[0])) if s[0] > 0 else 0
    return U[:, :r], {"rank": r, "path": "svd", "r_diag_ratio": ratio}


def residual_norms(Q, M):
    """Column norms of (I - Q Q^T) M for an orthonormal Q (N, r) and M (N, K); one reorthogonalization pass."""
    R = M - Q @ (Q.T @ M)
    R -= Q @ (Q.T @ R)
    return np.linalg.norm(R, axis=0)


def span_error(Q, M):
    """Bound ||Pi m - m|| / ||m|| for each column m of M (N, K), as a fraction. Q orthonormal basis of span(T)."""
    return residual_norms(Q, M) / np.linalg.norm(M, axis=0)


def lstsq_error(T, M):
    """Same bound by SciPy's SVD-based least squares (LAPACK gelsd), the second route of the projection."""
    X = scipy.linalg.lstsq(T, M, lapack_driver="gelsd")[0]
    return np.linalg.norm(M - T @ X, axis=0) / np.linalg.norm(M, axis=0)


def nearest_training_error(T, M):
    """min_j ||t_j - m|| / ||m|| for each column of M (N, K) over the columns of T (N, P) (Gram expansion in float64)."""
    d2 = (M * M).sum(0)[:, None] + (T * T).sum(0)[None, :] - 2.0 * (M.T @ T)
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0)) / np.linalg.norm(M, axis=0)


def vacuum_error(M):
    """||chi|| / ||m|| = ||m - 1|| / ||m||: error of guessing eps_r = 1 everywhere (not itself in span(T))."""
    return np.linalg.norm(M - 1.0, axis=0) / np.linalg.norm(M, axis=0)


def mean_image_error(T, M):
    """Error of predicting the mean training image: ||m - mean_j t_j|| / ||m||."""
    return np.linalg.norm(M - T.mean(axis=1, keepdims=True), axis=0) / np.linalg.norm(M, axis=0)


def span_plus_constant_error(Q, M):
    """Bound for span(T u {1}): the constant image added to the span (what an additive output bias could buy;
    the book's eq. 15.1 has none)."""
    one = np.ones((Q.shape[0], 1))
    q = one - Q @ (Q.T @ one)
    q -= Q @ (Q.T @ q)
    nq = np.linalg.norm(q)
    if nq > 1e-10 * np.sqrt(Q.shape[0]):
        Q = np.hstack([Q, q / nq])
    return span_error(Q, M)


def contrast_trained_error(T, M):
    """Bound for a hypothetical NMM trained on the contrast: project chi_m = m - 1 on span of the training
    chi_j = t_j - 1 and output 1 + projection; error ||Pi_chi chi - chi|| / ||m|| (denominator still ||m||)."""
    Qc, _ = basis_of_span(T - 1.0)
    return span_error_abs(Qc, M - 1.0, np.linalg.norm(M, axis=0))


def span_error_abs(Q, X, denom):
    """||(I - Q Q^T) x|| / denom for the columns of X (N, K) with the given denominators (K,)."""
    return residual_norms(Q, X) / denom


def summarize(err_pct):
    """Distribution summary of an array of errors in percent: n, mean, median, 5/25/75/95 percentiles, min, max and
    the fraction <= THRESH_PCT (3.3 %)."""
    x = np.asarray(err_pct, dtype=np.float64).ravel()
    p = np.percentile(x, [5, 25, 75, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "p05": float(p[0]),
            "p25": float(p[1]), "p75": float(p[2]), "p95": float(p[3]), "min": float(x.min()),
            "max": float(x.max()), "frac_le_3p3": float(np.mean(x <= THRESH_PCT))}


def welch_z(a, b):
    """Welch z-score (mean(a) - mean(b)) / sqrt(var(a)/n_a + var(b)/n_b) for two samples with n >= 2 each."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else 0.0


def jittered_scenes(base_shapes, rng, count, pos=0.25, size=0.2, eps_range=(1.1, 2.0)):
    """``count`` random variants of a digitized scene: centers +- ``pos`` (uniform), every length multiplied by
    U[1 - size, 1 + size] (independently per length), eps_r ~ U[eps_range]; painting order kept."""
    out = []
    for _ in range(count):
        shapes = []
        for s in base_shapes:
            eps = float(rng.uniform(*eps_range))
            cx = s[1] + rng.uniform(-pos, pos)
            cy = s[2] + rng.uniform(-pos, pos)
            lens = [float(v * rng.uniform(1 - size, 1 + size)) for v in s[3:-1]]
            shapes.append((s[0], float(cx), float(cy), *lens, eps))
        out.append(shapes)
    return out


def scene_sets(grid, rule, n_jitter, seed):
    """Dictionary key -> (matrix (N, k) of scene images composed with ``rule``, list of scene names).

    Keys: test1, test4, test4_high (the named instantiations) and test1_jitter, test4_jitter (``n_jitter`` random
    variants of the digitized scenes, seed ``seed``)."""
    sets = {}
    for key in ("test1", "test4", "test4_high"):
        imgs = [paint_scene(grid, sc["shapes"], rule).ravel() for sc in SCENES[key]]
        sets[key] = (np.array(imgs).T, [sc["name"] for sc in SCENES[key]])
    rng = np.random.default_rng([seed, 3])
    for key, eps_range in (("test1", (1.1, 2.0)), ("test4", (1.2, 2.0))):
        variants = jittered_scenes(JITTER_BASE[key], rng, n_jitter, eps_range=eps_range)
        imgs = [paint_scene(grid, sh, rule).ravel() for sh in variants]
        sets[key + "_jitter"] = (np.array(imgs).T, [f"{key}-jitter-{i}" for i in range(n_jitter)])
    return sets


def replicates_for(P, R):
    """Number of independent training sets used at training-set size P (fewer for the expensive large P)."""
    if P <= 100:
        return R
    if P <= 400:
        return min(R, max(2, R // 2))
    if P <= 800:
        return min(R, max(2, R // 4))
    return min(R, 3)


def run_main(grid, p_sweep, p_train, n_replicates, n_test, n_jitter, seed, log=print):
    """Compute (a) and (b) with implementation A. Returns (results dict for the JSON, raw dict of arrays).

    For every rule in RULES and every P in ``p_sweep`` (``p_train`` is always included) it draws
    ``replicates_for(P, n_replicates)`` independent training sets (seed [seed, 1, r]; the sets for different P with
    the same r are nested prefixes, and the painter and max rules share the shapes), the held-out set
    (seed [seed, 2], shared by all training sets) and the scene sets of ``scene_sets``, and records the errors of
    span(T), span(T u {1}), nearest training image, training-mean image and (P <= 800) the contrast-trained span, in
    percent."""
    sweep = sorted(set(p_sweep) | {p_train})
    reps = {P: replicates_for(P, n_replicates) for P in sweep}
    r_max = max(reps.values())
    results = {"rules": {}}
    raw = {}
    for rule in RULES:
        t0 = time.time()
        test_M = random_images(np.random.default_rng([seed, 2]), n_test, grid, rule).T  # (N, n_test)
        ones = np.ones((grid.n * grid.n, 1))
        sets = {}
        for comp in RULES:
            for key, (mat, names) in scene_sets(grid, comp, n_jitter, seed).items():
                sets[f"{key}@{comp}"] = (mat, names)
        keys = list(sets)
        S_all = np.hstack([sets[k][0] for k in keys])
        cuts = np.cumsum([0] + [sets[k][0].shape[1] for k in keys])
        scene_sl = {k: slice(cuts[i], cuts[i + 1]) for i, k in enumerate(keys)}
        acc = {P: {"test": [], "test_const": [], "test_nearest": [], "test_mean": [], "test_contrast": [],
                   "scene": [], "scene_const": [], "scene_nearest": [], "scene_mean": [], "scene_contrast": [],
                   "empty": [], "rank": [], "cond": [], "path": []} for P in sweep}
        for r in range(r_max):
            p_need = max(P for P in sweep if r < reps[P])
            imgs = random_images(np.random.default_rng([seed, 1, r]), p_need, grid, rule)  # (p_need, N)
            for P in sweep:
                if r >= reps[P]:
                    continue
                T = np.ascontiguousarray(imgs[:P].T)
                Q, info = basis_of_span(T)
                a = acc[P]
                a["empty"].append(100 * float(span_error(Q, ones)[0]))
                a["test"].append(100 * span_error(Q, test_M))
                a["test_const"].append(100 * span_plus_constant_error(Q, test_M))
                a["test_nearest"].append(100 * nearest_training_error(T, test_M))
                a["test_mean"].append(100 * mean_image_error(T, test_M))
                a["scene"].append(100 * span_error(Q, S_all))
                a["scene_const"].append(100 * span_plus_constant_error(Q, S_all))
                a["scene_nearest"].append(100 * nearest_training_error(T, S_all))
                a["scene_mean"].append(100 * mean_image_error(T, S_all))
                if P <= 800:
                    a["test_contrast"].append(100 * contrast_trained_error(T, test_M))
                    a["scene_contrast"].append(100 * contrast_trained_error(T, S_all))
                a["rank"].append(info["rank"])
                a["path"].append(info["path"])
                if P <= 400:
                    s = np.linalg.svd(T, compute_uv=False)
                    a["cond"].append(float(s[0] / s[-1]))
                else:
                    a["cond"].append(None)
        res_rule = {"test_vacuum_pct": summarize(100 * vacuum_error(test_M)), "sweep": {}}
        for P in sweep:
            a = acc[P]
            entry = {"n_train_sets": len(a["test"]), "rank": a["rank"], "path": a["path"], "cond": a["cond"],
                     "empty_scene_pct": {"per_set": a["empty"], "mean": float(np.mean(a["empty"])),
                                         "min": float(np.min(a["empty"])), "max": float(np.max(a["empty"]))}}
            for name in ("test", "test_const", "test_nearest", "test_mean", "test_contrast"):
                if a[name]:
                    arr = np.array(a[name])  # (replicates, n_test)
                    entry[name] = {"pooled": summarize(arr), "per_set_median": np.median(arr, axis=1).tolist(),
                                   "per_set_p05": np.percentile(arr, 5, axis=1).tolist(),
                                   "per_set_min": arr.min(axis=1).tolist()}
                    raw[f"{rule}/P{P}/{name}"] = arr
            for name in ("scene", "scene_const", "scene_nearest", "scene_mean", "scene_contrast"):
                if a[name]:
                    raw[f"{rule}/P{P}/{name}"] = np.array(a[name])  # (replicates, n_scene_columns)
            scenes = {}
            if a["scene"]:
                arrs = {nm: np.array(a[nm]) for nm in ("scene", "scene_const", "scene_nearest", "scene_mean",
                                                       "scene_contrast") if a[nm]}
                for k in keys:
                    sl = scene_sl[k]
                    scenes[k] = {"names": sets[k][1]}
                    for nm, arr in arrs.items():
                        sub = arr[:, sl]  # (replicates, k)
                        scenes[k][nm] = {"pooled": summarize(sub),
                                         "per_scene_median": np.median(sub, axis=0).tolist(),
                                         "per_scene_min": sub.min(axis=0).tolist(),
                                         "per_scene_max": sub.max(axis=0).tolist(),
                                         "per_scene_p05": np.percentile(sub, 5, axis=0).tolist()}
                    scenes[k]["vacuum_pct"] = (100 * vacuum_error(sets[k][0])).tolist()
            entry["scenes"] = scenes
            res_rule["sweep"][str(P)] = entry
        results["rules"][rule] = res_rule
        raw[f"{rule}/test_vacuum"] = 100 * vacuum_error(test_M)
        raw[f"{rule}/scene_vacuum"] = 100 * vacuum_error(S_all)
        raw[f"{rule}/scene_columns"] = np.array(sum(([f"{k}:{n}" for n in sets[k][1]] for k in keys), []))
        log(f"  rule {rule}: sweep {sweep} done in {time.time() - t0:.0f} s")
    results["sweep"] = sweep
    results["replicates"] = {str(P): reps[P] for P in sweep}
    return results, raw


def tail_scan(grid, rule, p_train, n_tail, n_sets, seed, chunk=1000):
    """Bound (percent) of ``n_tail`` fresh held-out draws (seed [seed, 4], distinct from the test set of ``run_main``)
    against ``n_sets`` training sets of ``p_train`` images (the first ``n_sets`` of ``run_main``); array
    (n_sets, n_tail). The minimum over a few thousand draws is set by the single easiest draw; this scan looks at the
    tail on a much larger pool."""
    Qs = [basis_of_span(random_images(np.random.default_rng([seed, 1, r]), p_train, grid, rule).T.copy())[0]
          for r in range(n_sets)]
    rng = np.random.default_rng([seed, 4])
    errs = np.empty((n_sets, n_tail))
    for start in range(0, n_tail, chunk):
        k = min(chunk, n_tail - start)
        M = random_images(rng, k, grid, rule).T
        for r, Q in enumerate(Qs):
            errs[r, start:start + k] = 100 * span_error(Q, M)
    return errs


def tail_summary(errs):
    """Summary of a tail-scan array: ``summarize`` plus the 0.1 and 1 percentiles and the fractions <= 5 and 10 %."""
    out = summarize(errs)
    out.update({"p00_1": float(np.percentile(errs, 0.1)), "p01": float(np.percentile(errs, 1)),
                "frac_le_5": float(np.mean(errs <= 5.0)), "frac_le_10": float(np.mean(errs <= 10.0)),
                "n_draws": int(errs.shape[1]), "n_train_sets": int(errs.shape[0])})
    return out


# --------------------------------------------------------------------------------------------------------------
# Other readings of the NMM (review): training eps_r range, output clipping, a constant in the span, the metric
# --------------------------------------------------------------------------------------------------------------

VARIANT_EPS_RANGES = ((1.0, 8.0), (1.0, 3.0), (1.0, 2.0))  # Table 15.2; an intermediate range; the colorbar of Fig. 15.9
CLIP_RANGE = (1.0, 2.0)  # the colorbar of Fig. 15.9: eps_r in [1, 2]
VARIANT_NAMES = ("span", "span_const", "shift_clip", "const_clip")


def clip_shift_error(Q, M, lo=CLIP_RANGE[0], hi=CLIP_RANGE[1], c_grid=None):
    """Error of the best in-span output followed by a clip to [lo, hi], after the best shift along Pi_T 1.

    For each column m of M (N, K): o(c) = clip(Pi_T m + c Pi_T 1, lo, hi) with Pi_T = Q Q^T; the shift c is chosen
    per column to minimize ||o(c) - m|| (an ORACLE choice, using the truth: the favorable end of what this kind of
    post-processing can do; c = 0 is the plain clip). Clipping is not a linear map, so the output leaves span(T); the
    shift along Pi_T 1 stays in span(T) and lets the clip remove the in-span background residual. The grid for c is
    ``c_grid`` (default 81 points on [-1, 1]) refined on 51 points around the best one. Returns ``(error (K,) as a
    fraction of ||m||, best c (K,))``.
    """
    p = Q @ (Q.T @ M)
    g = Q @ (Q.T @ np.ones((Q.shape[0], 1)))
    norm = np.linalg.norm(M, axis=0)

    def err(c):  # c: (K,) shifts -> (K,) errors
        return np.linalg.norm(np.clip(p + g * c[None, :], lo, hi) - M, axis=0) / norm

    grid = np.linspace(-1.0, 1.0, 81) if c_grid is None else np.asarray(c_grid, dtype=np.float64)
    best_e = np.full(M.shape[1], np.inf)
    best_c = np.zeros(M.shape[1])
    for c in grid:
        e = err(np.full(M.shape[1], c))
        better = e < best_e
        best_e, best_c = np.where(better, e, best_e), np.where(better, c, best_c)
    step = grid[1] - grid[0] if grid.size > 1 else 0.0
    for off in np.linspace(-step, step, 51):
        c = best_c + off
        e = err(c)
        better = e < best_e
        best_e, best_c = np.where(better, e, best_e), np.where(better, c, best_c)
    return best_e, best_c


def variant_errors(T, M, lo=CLIP_RANGE[0], hi=CLIP_RANGE[1]):
    """The four readings of one training matrix T (N, P) on the scene columns M (N, K), as fractions of ||m||:

    ``span`` (the book's eq. 15.1-15.6 output, no post-processing), ``span_const`` (span(T u {1}): an output bias that
    eq. 15.1 does not have), ``shift_clip`` (``clip_shift_error``: the in-span output, shifted along Pi_T 1, clipped to
    the colorbar [lo, hi]) and ``const_clip`` (the span(T u {1}) projection clipped to [lo, hi]). Also ``chi_norm``: the
    span error relative to ||chi|| = ||m - 1|| instead of ||m|| (eq. 15.21 applied to the contrast), which only makes the
    number larger. Returns a dict of (K,) arrays."""
    Q, _ = basis_of_span(T)
    one = np.ones((Q.shape[0], 1))
    q = one - Q @ (Q.T @ one)
    q -= Q @ (Q.T @ q)
    nq = np.linalg.norm(q)
    Qc = np.hstack([Q, q / nq]) if nq > 1e-10 * np.sqrt(Q.shape[0]) else Q
    norm = np.linalg.norm(M, axis=0)
    const_clip = np.linalg.norm(np.clip(Qc @ (Qc.T @ M), lo, hi) - M, axis=0) / norm
    shift_clip, _ = clip_shift_error(Q, M, lo, hi)
    chi = np.linalg.norm(M - 1.0, axis=0)
    return {"span": span_error(Q, M), "span_const": span_error(Qc, M), "shift_clip": shift_clip, "const_clip": const_clip,
            "chi_norm": residual_norms(Q, M) / np.where(chi > 0, chi, np.nan)}


def metric_reading_checks(scene_vacuum_pct):
    """Is the book's Table 15.3 consistent with eq. 15.21 on eps_r (first power), or with a squared misfit?

    ``scene_vacuum_pct``: {"test1": percent error of the vacuum guess eps_r = 1 on the digitized Test #1-like scene,
    "test4": same}. A reconstruction of Test #k that is worse than the vacuum guess is not a reconstruction. For the
    book's BA and VBIM numbers b (percent) of Test #1 and #4: as printed (first power) b is compared with the vacuum
    guess; as a squared relative misfit the relative error would be sqrt(b / 100) * 100. Returns the numbers and
    ``squared_reading_worse_than_vacuum`` (all four), the evidence against a squared misfit."""
    out = {"vacuum_guess_pct": dict(scene_vacuum_pct), "tests": {}}
    for key, idx in (("test1", 0), ("test4", 3)):
        for method, vals in (("BA", BOOK_BA_PCT), ("VBIM", BOOK_VBIM_PCT)):
            b = vals[idx]
            sq = float(np.sqrt(b / 100.0) * 100.0)
            out["tests"][f"{key}_{method}"] = {"book_pct": b, "as_printed_vs_vacuum": b / scene_vacuum_pct[key],
                                                "if_squared_relative_error_pct": sq, "squared_vs_vacuum": sq / scene_vacuum_pct[key]}
    out["squared_reading_worse_than_vacuum"] = bool(all(v["squared_vs_vacuum"] > 1.0 for v in out["tests"].values()))
    out["printed_reading_better_than_vacuum"] = {k: bool(v["as_printed_vs_vacuum"] < 1.0) for k, v in out["tests"].items()}
    return out


def heldout_min_diagnosis(err_pct):
    """Where does the minimum of a held-out error array come from? ``err_pct``: (training sets, draws) percent.

    The error of each draw averaged over the training sets, sorted: the easiest draw (index, mean over sets, and its
    smallest value over sets), the next ones, and the pooled low-tail statistics (median, 5th, 1st, 0.1th percentile of
    all values). The pooled minimum of 40 x 1000 values is the easiest value of the single easiest draw: it moves by
    several percentage points when one draw is replaced, whereas the median and the low percentiles do not."""
    e = np.asarray(err_pct, dtype=np.float64)
    per_draw = e.mean(axis=0)
    order = np.argsort(per_draw)
    pooled = e.ravel()
    return {"pooled_min": float(pooled.min()), "easiest_draw_index": int(order[0]), "easiest_draw_mean_over_sets": float(per_draw[order[0]]),
            "easiest_draw_min_over_sets": float(e[:, order[0]].min()),
            "next_easiest_draws_mean_over_sets": [float(per_draw[i]) for i in order[1:6]],
            "min_without_easiest_draw": float(np.delete(e, order[0], axis=1).min()),
            "median": float(np.median(pooled)), "p05": float(np.percentile(pooled, 5)), "p01": float(np.percentile(pooled, 1)),
            "p00_1": float(np.percentile(pooled, 0.1)), "n_sets": int(e.shape[0]), "n_draws": int(e.shape[1])}


def run_variants(grid, p_train, n_sets, n_test, seed, log=print):
    """The review's alternative readings at P = ``p_train``, painter's rule, ``n_sets`` independent training sets.

    For each training eps_r law in ``VARIANT_EPS_RANGES`` (the same seeds, hence the same shapes; only the
    permittivities differ): training sets (seed [seed, 1, r], the sets of ``run_main``), the empty-scene bound, the
    held-out draws (seed [seed, 2], same range) and the Test #1-like and Test #4-like scenes of ``SCENES`` (the eight named
    instantiations each; the digitized one is the first) under the four readings of ``variant_errors`` plus the
    chi-normalized span error. Returns ``(results dict for the JSON, raw dict of arrays)`` with errors in percent."""
    results = {"p_train": p_train, "n_sets": n_sets, "n_test": n_test, "rule": "painter", "clip_range": list(CLIP_RANGE),
               "readings": {"span": "book eqs. 15.1-15.6 as printed: o in span(T)",
                            "span_const": "span(T u {1}): an additive output bias (not in eq. 15.1)",
                            "shift_clip": "in-span output + oracle shift along Pi_T 1, clipped to the colorbar [1, 2]",
                            "const_clip": "span(T u {1}) output clipped to [1, 2]"},
               "eps_ranges": {}}
    raw = {}
    scene_mats = {key: np.array([paint_scene(grid, sc["shapes"], "painter").ravel() for sc in SCENES[key]]).T
                  for key in ("test1", "test4")}
    for eps_range in VARIANT_EPS_RANGES:
        tag = f"{eps_range[0]:g}-{eps_range[1]:g}"
        t0 = time.time()
        test_M = random_images(np.random.default_rng([seed, 2]), n_test, grid, "painter", eps_range).T
        acc = {k: {n: [] for n in VARIANT_NAMES + ("chi_norm",)} for k in scene_mats}
        held, empty = [], []
        ones = np.ones((grid.n * grid.n, 1))
        for r in range(n_sets):
            T = np.ascontiguousarray(random_images(np.random.default_rng([seed, 1, r]), p_train, grid, "painter", eps_range).T)
            Q, _ = basis_of_span(T)
            held.append(100 * span_error(Q, test_M))
            empty.append(100 * float(span_error(Q, ones)[0]))
            for key, M in scene_mats.items():
                ve = variant_errors(T, M)
                for name, v in ve.items():
                    acc[key][name].append(100 * v)
        entry = {"empty_scene_pct": {"mean": float(np.mean(empty)), "min": float(np.min(empty)), "max": float(np.max(empty))},
                 "heldout_span": summarize(np.array(held)), "heldout_min_diagnosis": heldout_min_diagnosis(np.array(held)),
                 "scenes": {}}
        raw[f"variants/{tag}/heldout_span"] = np.array(held)
        for key in scene_mats:
            entry["scenes"][key] = {"scene_names": [sc["name"] for sc in SCENES[key]]}
            for name, lst in acc[key].items():
                arr = np.array(lst)  # (n_sets, 8 scenes)
                raw[f"variants/{tag}/{key}/{name}"] = arr
                entry["scenes"][key][name] = {"pooled_over_8_scenes": summarize(arr), "digitized": summarize(arr[:, 0]),
                                               "per_scene_median": np.median(arr, axis=0).tolist(),
                                               "per_scene_min": arr.min(axis=0).tolist()}
        results["eps_ranges"][tag] = entry
        log(f"  eps_r ~ U{list(eps_range)}: {n_sets} training sets of {p_train}, done in {time.time() - t0:.0f} s")
    # the best reading per test (smallest median over the digitized scene, and smallest value anywhere)
    best = {}
    for key in scene_mats:
        cands = [(tag, name, e["scenes"][key][name]) for tag, e in results["eps_ranges"].items() for name in VARIANT_NAMES]
        by_med = min(cands, key=lambda x: x[2]["digitized"]["median"])
        by_min = min(cands, key=lambda x: x[2]["pooled_over_8_scenes"]["min"])
        names = results["eps_ranges"][by_min[0]]["scenes"][key]["scene_names"]
        scene_of_min = names[int(np.argmin(by_min[2]["per_scene_min"]))]
        best[key] = {"digitized_median": {"eps_range": by_med[0], "reading": by_med[1], "median": by_med[2]["digitized"]["median"],
                                          "min": by_med[2]["digitized"]["min"]},
                     "smallest_value_over_all_scenes_and_sets": {"eps_range": by_min[0], "reading": by_min[1], "scene": scene_of_min,
                                                                  "min": by_min[2]["pooled_over_8_scenes"]["min"]}}
    results["best_reading"] = best
    # the book's NMM numbers are flat across the tests (3.2615 ... 3.3103); the bound is not, under any reading
    results["flatness"] = {"book_nmm_test4_minus_test1_pp": BOOK_NMM_PCT[3] - BOOK_NMM_PCT[0],
                           "book_nmm_max_minus_min_pp": max(BOOK_NMM_PCT) - min(BOOK_NMM_PCT),
                           "digitized_median_test4_minus_test1_pp": {
                               tag: {name: e["scenes"]["test4"][name]["digitized"]["median"] - e["scenes"]["test1"][name]["digitized"]["median"]
                                     for name in VARIANT_NAMES} for tag, e in results["eps_ranges"].items()}}
    vac = {k: float(100 * vacuum_error(scene_mats[k])[0]) for k in scene_mats}
    results["metric_reading"] = metric_reading_checks(vac)
    return results, raw


# --------------------------------------------------------------------------------------------------------------
# Independent implementation B (numpy only; shares no code with the package or with implementation A)
# --------------------------------------------------------------------------------------------------------------


def indep_shapes(rng):
    """One draw of the Table 15.2 scene family as a list of shapes in painting order (disk, disk, ring, rectangle,
    square), written out from the table: r1, r2 in [0.1, 1], r3 in [0.1, 1], r4 in [0.2, 2] with r4 > r3 by rejection,
    l1, l2, l3 in [0.1, 2], all centers in [-2, 2]^2, eps_1..5 in [1, 8]."""
    shapes = []
    for _ in range(2):
        r = rng.uniform(0.1, 1.0)
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        shapes.append(("disk", x, y, r, rng.uniform(1, 8)))
    r3 = rng.uniform(0.1, 1.0)
    while True:
        r4 = rng.uniform(0.2, 2.0)
        if r4 > r3:
            break
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    shapes.append(("ring", x, y, r3, r4, rng.uniform(1, 8)))
    l1, l2 = rng.uniform(0.1, 2.0), rng.uniform(0.1, 2.0)
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    shapes.append(("rect", x, y, l1, l2, rng.uniform(1, 8)))
    l3 = rng.uniform(0.1, 2.0)
    x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
    shapes.append(("rect", x, y, l3, l3, rng.uniform(1, 8)))
    return shapes


def indep_render(shapes, n, side, ss):
    """Render ``shapes`` (lengths in meters on a DOI of ``side`` meters centered at the origin) as eps_r on an
    n x n grid, "ij" indexing. Shapes are composed on a (ss n) x (ss n) sub-pixel grid under both rules (painter's
    rule: assignment; max rule: maximum with the background eps_r = 1), then averaged over ss x ss blocks.
    Returns (painter, max), each (n, n). Shapes given for the 5 m DOI are scaled by side / 5."""
    m = n * ss
    ax = -side / 2 + (np.arange(m) + 0.5) * (side / m)
    X, Y = ax[:, None], ax[None, :]
    sc = side / 5.0
    pa = np.ones((m, m))
    mx = np.ones((m, m))
    for s in shapes:
        eps = s[-1]
        if s[0] == "disk":
            _, cx, cy, r, _ = s
            mask = (X - sc * cx) ** 2 + (Y - sc * cy) ** 2 <= (sc * r) ** 2
        elif s[0] == "ring":
            _, cx, cy, r3, r4, _ = s
            d2 = (X - sc * cx) ** 2 + (Y - sc * cy) ** 2
            mask = (d2 >= (sc * r3) ** 2) & (d2 <= (sc * r4) ** 2)
        else:
            _, cx, cy, lx, ly, _ = s
            mask = (np.abs(X - sc * cx) <= 0.5 * sc * lx) & (np.abs(Y - sc * cy) <= 0.5 * sc * ly)
        pa[mask] = eps
        mx[mask] = np.maximum(mx[mask], eps)
    block = lambda a: a.reshape(n, ss, n, ss).mean(axis=(1, 3))
    return block(pa), block(mx)


def indep_images(seed_seq, count, n, side, ss):
    """``count`` independent draws from a Philox stream; returns (painter (count, N), max (count, N))."""
    rng = np.random.Generator(np.random.Philox(seed_seq))
    pa = np.empty((count, n * n))
    mx = np.empty((count, n * n))
    for k in range(count):
        a, b = indep_render(indep_shapes(rng), n, side, ss)
        pa[k], mx[k] = a.ravel(), b.ravel()
    return pa, mx


def indep_svd_error(T, M):
    """Bound by the SVD of T (N x P): project on the left singular vectors with s > max(N, P) eps s_1; as a fraction."""
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    keep = s > max(T.shape) * np.finfo(float).eps * s[0]
    U = U[:, keep]
    R = M - U @ (U.T @ M)
    return np.linalg.norm(R, axis=0) / np.linalg.norm(M, axis=0)


def run_independent(n, side, p_train, n_replicates, n_test, ss, seed, scene_ss, log=print):
    """Implementation B of the baseline: (a) painter and max rules at P = p_train, ``n_replicates`` training sets and
    one held-out set of ``n_test`` draws; (b) the named scenes of SCENES (test1, test4, rendered by B).

    Returns (out, scenes, test_images): out[rule] = {"test": (R, n_test) percent, "scenes": {key: (R, 8) percent}},
    scenes[rule][key] = B-rendered scene images (N, 8), test_images[rule] = held-out images (N, n_test)."""
    ss_test = np.random.SeedSequence([seed + 1000, 2])
    test_p, test_m = indep_images(ss_test, n_test, n, side, ss)
    test_images = {"painter": test_p.T, "max": test_m.T}
    scenes = {rule: {} for rule in RULES}
    for key in ("test1", "test4"):
        imgs = [indep_render(sc["shapes"], n, side, scene_ss) for sc in SCENES[key]]
        scenes["painter"][key] = np.array([a.ravel() for a, _ in imgs]).T
        scenes["max"][key] = np.array([b.ravel() for _, b in imgs]).T
    out = {rule: {"test": [], "scenes": {k: [] for k in scenes[rule]}} for rule in RULES}
    for r in range(n_replicates):
        tp, tm = indep_images(np.random.SeedSequence([seed + 1000, 1, r]), p_train, n, side, ss)
        for rule, T in (("painter", tp.T), ("max", tm.T)):
            out[rule]["test"].append(100 * indep_svd_error(T, test_images[rule]))
            for key, S in scenes[rule].items():
                out[rule]["scenes"][key].append(100 * indep_svd_error(T, S))
    for rule in RULES:
        out[rule]["test"] = np.array(out[rule]["test"])
        out[rule]["scenes"] = {k: np.array(v) for k, v in out[rule]["scenes"].items()}
    log(f"  independent implementation: {n_replicates} training sets of {p_train}, {n_test} held-out draws")
    return out, scenes, test_images


def rasterizer_bridge(grid, args, n_test=300):
    """Same shapes, two rasterizers. Shapes come from the independent sampler (``args.p_train`` training shapes and
    ``n_test`` test shapes); each is rendered by implementation B (sub-pixel composition, ``args.indep_ss``) and by
    implementation A (``paint_scene``: coverage-averaged shapes, 4 x 4 sub-samples, blended per pixel). The bound of the
    test images against the training images is computed with A (QR) on A's images and with B (SVD) on B's images; the
    same is done for the 16 named scenes (rendered with ``args.indep_scene_ss`` by B). Only the rasterization differs
    and the training-set noise cancels, so the differences measure the implementation difference directly.
    Returns per rule the differences A - B in percentage points (max |diff|, mean, rms), the median relative image
    difference and the mean errors of A and B, for the random test images and for the named scenes."""
    rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([args.seed + 2000, 5])))
    train = [indep_shapes(rng) for _ in range(args.p_train)]
    test = [indep_shapes(rng) for _ in range(n_test)]
    named = [sc["shapes"] for key in ("test1", "test4") for sc in SCENES[key]]
    render_b = lambda shapes, ss: [indep_render(sh, grid.n, grid.side, ss) for sh in shapes]
    tb, mb, sb = render_b(train, args.indep_ss), render_b(test, args.indep_ss), render_b(named, args.indep_scene_ss)
    out = {}
    for k, rule in enumerate(RULES):
        TB = np.array([x[k].ravel() for x in tb]).T
        MB = np.array([x[k].ravel() for x in mb]).T
        SB = np.array([x[k].ravel() for x in sb]).T
        TA = np.array([paint_scene(grid, sh, rule).ravel() for sh in train]).T
        MA = np.array([paint_scene(grid, sh, rule).ravel() for sh in test]).T
        SA = np.array([paint_scene(grid, sh, rule).ravel() for sh in named]).T
        QA = basis_of_span(TA)[0]
        entry = {}
        for tag, XA, XB in (("images", MA, MB), ("scenes", SA, SB)):
            eA = 100 * span_error(QA, XA)
            eB = 100 * indep_svd_error(TB, XB)
            d = eA - eB
            entry[tag] = {"n": int(d.size), "max_abs_diff_pp": float(np.abs(d).max()), "mean_diff_pp": float(d.mean()),
                          "rms_diff_pp": float(np.sqrt((d**2).mean())),
                          "image_rel_diff_median": float(np.median(np.linalg.norm(XA - XB, axis=0) / np.linalg.norm(XA, axis=0))),
                          "mean_error_A_pct": float(eA.mean()), "mean_error_B_pct": float(eB.mean())}
        out[rule] = entry
    return out


def run_binary_variant(grid, args):
    """Sensitivity to the rasterization: independent implementation B with binary masks (ss = 1: a pixel belongs to a
    shape iff its center does, no coverage averaging), which is what a plain MATLAB generator would plausibly do.
    P = ``args.p_train``, 10 training sets, ``args.n_test`` held-out draws, the named scenes rendered the same way.
    Returns {rule: {"test": summarize(pooled), "test1": summarize, "test4": summarize}} (percent)."""
    B, _, _ = run_independent(grid.n, grid.side, args.p_train, min(10, args.indep_replicates), args.n_test, 1,
                              args.seed + 500, 1, log=lambda *a: None)
    return {rule: {"test": summarize(B[rule]["test"]), "test1": summarize(B[rule]["scenes"]["test1"]),
                   "test4": summarize(B[rule]["scenes"]["test4"])} for rule in RULES}


def bootstrap_median_z(a, b, n_boot=2000, seed=123):
    """z = (median(a) - median(b)) / sqrt(var_boot(median a) + var_boot(median b)) for two independent samples,
    each resampled with replacement ``n_boot`` times."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ma = np.median(a[rng.integers(0, a.size, (n_boot, a.size))], axis=1)
    mb = np.median(b[rng.integers(0, b.size, (n_boot, b.size))], axis=1)
    return float((np.median(a) - np.median(b)) / np.sqrt(ma.var(ddof=1) + mb.var(ddof=1)))


def cross_check(grid, results, raw, args, log=print):
    """Compare implementation A (``results``, ``raw``) with the independent implementation B.

    Deterministic bridges (identical inputs, no sampling noise):
      1. projection: the bound of 200 held-out images against A's first training set by QR (A), SciPy gelsd (A') and
         SVD (B): maximum absolute difference of the errors (fractions);
      2. scenes: the named scenes rendered by A and by B, relative image difference and the difference (percentage
         points) of their bound against the same training set;
      4. rasterizers: the same random shapes rendered by A and by B (``rasterizer_bridge``): per-image differences of
         the bound in percentage points.
    Statistical comparisons of independent draws (A: ``args.replicates`` training sets, B: ``args.indep_replicates``;
    z-scores that include every source of sampling variance):
      3a. generator laws: mean eps_r, occupied fraction (eps_r > 1) and vacuum error ||chi|| / ||m|| of the held-out
          images of A and B (n_test iid images each), Welch z;
      3b. held-out bound: for each held-out image the bound averaged over the training sets, A vs B: Welch z of the mean
          and bootstrap z of the median over images (this includes the test-sample variance, which a z-score over
          training-set medians alone would miss);
      3c. scenes: for each training set the mean bound over the 8 named scenes of a type, A vs B, Welch z across
          training sets (one number per set, so the correlation between scenes is accounted for);
      the per-scene z-scores are listed for information.
    The two rasterizers legitimately differ at the 1-2 % level in the images (coverage quantization of the 4 x 4
    sub-sampling), which moves the bound by a fraction of a percentage point; the criteria below allow for that and are
    far tighter than the gap to the book's 3.3 %.
    ``agree`` is True when bridge 1 < 1e-9, bridge 2 < 0.1 pp, bridge 4 max |diff| < 1 pp and relative bias of the mean
    error < 2 % (images and scenes), and every z of 3a-3c satisfies |z| < 3.5 (14 comparisons, partly correlated)."""
    P = args.p_train
    c = {"projection_abs_diff_max": 1e-9, "scene_error_pp_max": 0.1, "rasterizer_pp_max": 1.0,
         "rasterizer_rel_bias_max": 0.02, "z_max": 3.5}
    out = {"p_train": P, "criteria": c}
    t0 = time.time()
    # bridge 1
    T = random_images(np.random.default_rng([args.seed, 1, 0]), P, grid, "painter").T.copy()
    M = random_images(np.random.default_rng([args.seed, 2]), min(args.n_test, 200), grid, "painter").T
    Q, _ = basis_of_span(T)
    e_qr, e_ls, e_svd = span_error(Q, M), lstsq_error(T, M), indep_svd_error(T, M)
    out["bridge1_projection"] = {"max_abs_diff_qr_vs_gelsd": float(np.abs(e_qr - e_ls).max()),
                                 "max_abs_diff_qr_vs_svd": float(np.abs(e_qr - e_svd).max()),
                                 "n_images": int(M.shape[1])}
    B, B_scenes, B_test = run_independent(grid.n, grid.side, P, args.indep_replicates, args.n_test, args.indep_ss,
                                          args.seed, args.indep_scene_ss, log=log)
    for rule in RULES:
        raw[f"indep/{rule}/test"] = B[rule]["test"]
        for key, arr in B[rule]["scenes"].items():
            raw[f"indep/{rule}/scene_{key}"] = arr
    # bridge 2
    rel, pp = [], []
    for rule in RULES:
        for key in ("test1", "test4"):
            A_mat = scene_sets(grid, rule, 1, args.seed)[key][0]
            B_mat = B_scenes[rule][key]
            rel += (np.linalg.norm(A_mat - B_mat, axis=0) / np.linalg.norm(A_mat, axis=0)).tolist()
            pp += (100 * np.abs(span_error(Q, A_mat) - span_error(Q, B_mat))).tolist()
    out["bridge2_scenes"] = {"image_rel_diff_max": float(np.max(rel)), "error_diff_pp_max": float(np.max(pp)),
                             "n_scenes": len(rel)}
    # bridge 4
    out["bridge4_rasterizers"] = rasterizer_bridge(grid, args)
    # statistical comparisons
    zs = {}
    laws = {}
    for rule in RULES:
        A_img = random_images(np.random.default_rng([args.seed, 2]), args.n_test, grid, rule)  # (n_test, N)
        B_img = B_test[rule].T
        feats = {"mean_eps": lambda X: X.mean(axis=1), "occupied_fraction": lambda X: (X > 1.0 + 1e-9).mean(axis=1),
                 "vacuum_error": lambda X: np.linalg.norm(X - 1.0, axis=1) / np.linalg.norm(X, axis=1)}
        laws[rule] = {}
        for name, f in feats.items():
            fa, fb = f(A_img), f(B_img)
            laws[rule][name] = {"mean_A": float(fa.mean()), "mean_B": float(fb.mean()), "z": welch_z(fa, fb)}
            zs[f"law/{rule}/{name}"] = laws[rule][name]["z"]
    out["law_check"] = laws
    held = {}
    for rule in RULES:
        a_img = raw[f"{rule}/P{P}/test"].mean(axis=0)  # (n_test,) bound averaged over A's training sets
        b_img = B[rule]["test"].mean(axis=0)
        held[rule] = {"A": summarize(a_img), "B": summarize(b_img), "z_mean": welch_z(a_img, b_img),
                      "z_median": bootstrap_median_z(a_img, b_img)}
        zs[f"heldout/{rule}/mean"], zs[f"heldout/{rule}/median"] = held[rule]["z_mean"], held[rule]["z_median"]
    out["heldout_bound"] = held
    scene_cmp = {}
    for rule in RULES:
        cols = list(raw[f"{rule}/scene_columns"])
        scene_cmp[rule] = {}
        for key in ("test1", "test4"):
            idx = [cols.index(f"{key}@{rule}:{sc['name']}") for sc in SCENES[key]]
            A_sc = raw[f"{rule}/P{P}/scene"][:, idx]  # (R_A, 8)
            B_sc = B[rule]["scenes"][key]  # (R_B, 8)
            z_set = welch_z(A_sc.mean(axis=1), B_sc.mean(axis=1))
            zs[f"scenes/{rule}/{key}"] = z_set
            scene_cmp[rule][key] = {"z_set_mean": z_set, "mean_A": float(A_sc.mean()), "mean_B": float(B_sc.mean()),
                                    "z_per_scene": [welch_z(A_sc[:, i], B_sc[:, i]) for i in range(A_sc.shape[1])],
                                    "median_A": np.median(A_sc, axis=0).tolist(),
                                    "median_B": np.median(B_sc, axis=0).tolist()}
    out["scene_bound"] = scene_cmp
    out["z_all"] = zs
    out["z_max_abs"] = float(np.max(np.abs(list(zs.values()))))
    out["indep_replicates"], out["indep_ss"] = args.indep_replicates, args.indep_ss
    out["wall_time_s"] = time.time() - t0
    r4 = out["bridge4_rasterizers"]
    out["agree"] = bool(
        max(out["bridge1_projection"]["max_abs_diff_qr_vs_gelsd"],
            out["bridge1_projection"]["max_abs_diff_qr_vs_svd"]) < c["projection_abs_diff_max"]
        and out["bridge2_scenes"]["error_diff_pp_max"] < c["scene_error_pp_max"]
        and all(r4[r][t]["max_abs_diff_pp"] < c["rasterizer_pp_max"]
                and abs(r4[r][t]["mean_diff_pp"]) < c["rasterizer_rel_bias_max"] * r4[r][t]["mean_error_B_pct"]
                for r in RULES for t in ("images", "scenes"))
        and out["z_max_abs"] < c["z_max"])
    return out


# --------------------------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------------------------


def make_figure(grid, results, raw, args, path):
    """Three summary panels (ECDF of the bound for (a); bound versus P; (b) per scene) and the best-in-span
    reconstructions of the digitized Test #1 and Test #4 scenes (truth, projection onto span of the first training set,
    residual)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.gridspec import GridSpec

    blue, orange, violet, gray, ink = "#2a78d6", "#eb6834", "#4a3aa7", "#8a8984", "#2b2b29"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": gray, "axes.labelcolor": ink, "xtick.color": ink,
                         "ytick.color": ink, "axes.spines.top": False, "axes.spines.right": False})
    P0 = args.p_train
    sweep = results["sweep"]
    fig = plt.figure(figsize=(12.5, 7.6), constrained_layout=True)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.15, 1.0])

    ax = fig.add_subplot(gs[0, 0:2])
    for arr, color, label in ((raw[f"painter/P{P0}/test"], blue, f"painter, P = {P0}"),
                              (raw[f"max/P{P0}/test"], orange, f"max-overlap, P = {P0}"),
                              (raw[f"painter/P{sweep[-1]}/test"], violet, f"painter, P = {sweep[-1]}")):
        x = np.sort(arr.ravel())
        ax.plot(x, np.arange(1, x.size + 1) / x.size, color=color, lw=1.4, label=label)
    ax.axvline(THRESH_PCT, color=gray, lw=1.6)
    ax.annotate("book NMM\n3.26-3.31 %", (THRESH_PCT, 0.5), xytext=(6, 0), textcoords="offset points",
                color=ink, fontsize=7.5, va="center")
    ax.set_xlim(left=0)
    ax.set_xlabel("best-in-span error ||Pi m - m|| / ||m||  (%, eq. 15.21, eps_r)")
    ax.set_ylabel("fraction of held-out draws")
    ax.set_title("(a) held-out Table 15.2 draws", loc="left", fontsize=9)
    ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax.grid(color="#e4e3df", lw=0.5)

    ax = fig.add_subplot(gs[0, 2:4])
    rows = results["rules"]["painter"]["sweep"]
    med = [rows[str(P)]["test"]["pooled"]["median"] for P in sweep]
    p05 = [rows[str(P)]["test"]["pooled"]["p05"] for P in sweep]
    mn = [rows[str(P)]["test"]["pooled"]["min"] for P in sweep]
    nn = [rows[str(P)]["test_nearest"]["pooled"]["median"] for P in sweep]
    empty = [rows[str(P)]["empty_scene_pct"]["mean"] for P in sweep]
    for y, color, ls, label in ((med, blue, "-", "median"), (p05, orange, "-", "5th pct"),
                                (mn, violet, "-", "minimum"), (nn, gray, "--", "nearest image\n(median)"),
                                (empty, ink, ":", "empty scene\nm = 1")):
        ax.plot(sweep, y, color=color, ls=ls, marker="o", ms=3.5, lw=1.3)
        ax.annotate(label, (sweep[-1], y[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=ink,
                    va="center", annotation_clip=False)
    ax.axhline(THRESH_PCT, color=gray, lw=1.6)
    ax.annotate("book NMM 3.26-3.31 %", (sweep[0], THRESH_PCT), xytext=(0, 4), textcoords="offset points",
                color=ink, fontsize=7.5, ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(sweep[0] * 0.9, sweep[-1] * 3.6)
    ax.set_xticks(sweep)
    ax.set_xticklabels([str(P) for P in sweep], fontsize=7)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.5, 1, 2, 3.3, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0.5", "1", "2", "3.3", "5", "10", "20", "50", "100"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("number of training images P")
    ax.set_ylabel("error (%)")
    ax.set_title("(a) bound versus P (painter's rule)", loc="left", fontsize=9)
    ax.grid(color="#e4e3df", lw=0.5, which="both")

    ax = fig.add_subplot(gs[0, 4:6])
    cols = list(raw[f"painter/scene_columns"])
    pos = 0
    ticks = []
    for key, label in (("test1", "Test #1-like"), ("test4", "Test #4-like")):
        names = [sc["name"] for sc in SCENES[key]]
        idx = [cols.index(f"{key}@painter:{nm}") for nm in names]
        arr = raw[f"painter/P{P0}/scene"][:, idx]  # (R, 8)
        arr_c = raw[f"painter/P{P0}/scene_const"][:, idx]
        vac = raw["painter/scene_vacuum"][idx]
        for i in range(arr.shape[1]):
            x = pos + i
            ax.plot([x, x], [arr[:, i].min(), arr[:, i].max()], color=blue, lw=1.0)
            ax.plot(x, np.median(arr[:, i]), "o", color=blue, ms=4.5, zorder=3)
            ax.plot(x, np.median(arr_c[:, i]), "D", mfc="none", mec=orange, ms=4.5, mew=1.2, zorder=3)
            ax.plot(x, vac[i], "_", color=gray, ms=9, mew=1.6)
        ticks.append((pos + 3.5, label))
        pos += arr.shape[1] + 1
    ax.axhline(THRESH_PCT, color=gray, lw=1.6)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks])
    ax.set_ylabel("error (%)")
    ax.set_ylim(0, 33)
    ax.set_title(f"(b) scenes, P = {P0}: median and range over training sets", loc="left", fontsize=9)
    ax.plot([], [], "o", color=blue, label="best in span")
    ax.plot([], [], "D", mfc="none", mec=orange, ms=4.5, mew=1.2, label="span + constant (not the book's model)")
    ax.plot([], [], "_", color=gray, ms=9, mew=1.6, label="vacuum guess")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(color="#e4e3df", lw=0.5, axis="y")

    # best-in-span reconstructions of the digitized scenes against the first training set of the baseline
    T = random_images(np.random.default_rng([args.seed, 1, 0]), P0, grid, "painter").T.copy()
    Q, _ = basis_of_span(T)
    panels = []
    for key in ("test1", "test4"):
        m = paint_scene(grid, SCENES[key][0]["shapes"], "painter").ravel()
        proj = Q @ (Q.T @ m)
        panels.append((key, m, proj, 100 * np.linalg.norm(proj - m) / np.linalg.norm(m)))
    ext = (-grid.side / 2, grid.side / 2, -grid.side / 2, grid.side / 2)
    vmax = max(1.0 + 1e-9, max(p[1].max() for p in panels))
    for r, (key, m, proj, err) in enumerate(panels):
        specs = ((m, "viridis", (1.0, 2.0), f"{'Test #1' if key == 'test1' else 'Test #4'}-like: truth eps_r"),
                 (proj, "viridis", (1.0, 2.0), f"best in span ({err:.1f} %)"),
                 (proj - m, "RdBu_r", (-0.5, 0.5), "residual, eps_r"))
        for c, (img, cmap, (lo, hi), title) in enumerate(specs):
            sub = fig.add_subplot(gs[1, 3 * r + c])
            im = sub.imshow(img.reshape(grid.n, grid.n).T, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            sub.set_title(title, fontsize=8)
            sub.set_xticks([-2, 0, 2])
            sub.set_yticks([-2, 0, 2])
            if c == 2:
                fig.colorbar(im, ax=sub, shrink=0.8, pad=0.02)
    fig.suptitle(f"Span test of the book's NMM: Grid({grid.n}, {grid.side:g}), N = {grid.n ** 2}; "
                 "gray line = Table 15.3 NMM column (3.26-3.31 %)", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "span_test"))
    ap.add_argument("--n", type=int, default=96, help="grid size n (n x n pixels)")
    ap.add_argument("--side", type=float, default=5.0, help="DOI side in wavelengths")
    ap.add_argument("--p-train", type=int, default=100, help="baseline number of training images (the book's P)")
    ap.add_argument("--p-sweep", default="25,50,100,200,400,800,1600,3200", help="training-set sizes to sweep")
    ap.add_argument("--replicates", type=int, default=40, help="independent training sets at P <= 100")
    ap.add_argument("--n-test", type=int, default=1000, help="held-out draws for (a)")
    ap.add_argument("--n-jitter", type=int, default=200, help="jittered scenes per type for (b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tail", type=int, default=20000, help="fresh held-out draws of the tail scan")
    ap.add_argument("--tail-sets", type=int, default=3, help="training sets of the tail scan")
    ap.add_argument("--indep-replicates", type=int, default=40, help="training sets of the independent implementation")
    ap.add_argument("--indep-ss", type=int, default=6, help="sub-pixel factor of the independent training rasterizer")
    ap.add_argument("--indep-scene-ss", type=int, default=8, help="sub-pixel factor of the independent scene rasterizer")
    ap.add_argument("--no-cross-check", action="store_true", help="skip the independent implementation")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--figure-only", action="store_true", help="redraw span_test.png from span_test.json / .npz in --out")
    ap.add_argument("--variants-only", action="store_true",
                    help="compute only the review variants (eps_r range, clipping, constant, metric) and the minimum "
                         "diagnosis, and add them to the span_test.json / .npz in --out (the stored numbers are not changed)")
    ap.add_argument("--variant-sets", type=int, default=40, help="training sets per eps_r law in the variants")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration (n = 24)")
    args = ap.parse_args(argv)
    if args.quick:
        args.n, args.p_train, args.p_sweep = 24, 20, "10,20,40"
        args.replicates, args.n_test, args.n_jitter = 3, 40, 6
        args.n_tail, args.tail_sets = 100, 2
        args.indep_replicates, args.indep_ss, args.indep_scene_ss = 3, 3, 4
        args.variant_sets = 3
    return args


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.figure_only:
        results = json.loads((out / "span_test.json").read_text())
        stored = argparse.Namespace(**results["meta"]["args"])
        with np.load(out / "span_test.npz") as z:
            raw = {k: z[k] for k in z.files}
        make_figure(Grid(stored.n, stored.side), results, raw, stored, out / "span_test.png")
        print(f"wrote {out / 'span_test.png'}")
        return results
    if args.variants_only:
        results = json.loads((out / "span_test.json").read_text())
        stored = argparse.Namespace(**results["meta"]["args"])
        stored.variant_sets = args.variant_sets
        with np.load(out / "span_test.npz") as z:
            raw = {k: z[k] for k in z.files}
        add_variants(Grid(stored.n, stored.side), stored, results, raw)
        (out / "span_test.json").write_text(json.dumps(results, indent=1))
        np.savez_compressed(out / "span_test.npz", **raw)
        print_variants(results)
        print(f"wrote the variants into {out / 'span_test.json'} and span_test.npz ({time.time() - t_start:.0f} s)")
        return results
    grid = Grid(args.n, args.side)
    sweep = [int(v) for v in args.p_sweep.split(",") if v.strip()]
    print(f"span test: Grid({grid.n}, {grid.side:g}), N = {grid.n ** 2}, P sweep {sorted(set(sweep) | {args.p_train})}, "
          f"baseline P = {args.p_train}, {args.replicates} replicates, {args.n_test} held-out draws, threads <= {MAX_THREADS}")
    results, raw = run_main(grid, sweep, args.p_train, args.replicates, args.n_test, args.n_jitter, args.seed)
    print("tail scan ...")
    results["tail_scan"] = {}
    for rule in RULES:
        errs = tail_scan(grid, rule, args.p_train, args.n_tail, args.tail_sets, args.seed)
        raw[f"tail/{rule}"] = errs
        results["tail_scan"][rule] = tail_summary(errs)
    results["meta"] = {"args": vars(args), "grid": {"n": grid.n, "side": grid.side, "N": grid.n ** 2},
                       "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
                       "machine": platform.platform(), "thread_env": {k: os.environ.get(k) for k in (
                           "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                       "command": "python scripts/span_test.py " + " ".join(sys.argv[1:] if argv is None else argv)}
    results["book"] = book_block()
    if not args.no_cross_check:
        print("cross-check against the independent implementation ...")
        results["cross_check"] = cross_check(grid, results, raw, args)
        cc = results["cross_check"]
        r4 = cc["bridge4_rasterizers"]
        print(f"  bridge 1 (QR vs gelsd / SVD): {cc['bridge1_projection']['max_abs_diff_qr_vs_gelsd']:.2e} / "
              f"{cc['bridge1_projection']['max_abs_diff_qr_vs_svd']:.2e};  bridge 2 (scenes): image rel "
              f"{cc['bridge2_scenes']['image_rel_diff_max']:.2e}, error {cc['bridge2_scenes']['error_diff_pp_max']:.4f} pp")
        print("  bridge 4 (same shapes, two rasterizers, A - B): " + ";  ".join(
            f"{r} {t}: max {r4[r][t]['max_abs_diff_pp']:.3f} pp, mean {r4[r][t]['mean_diff_pp']:+.3f} pp "
            f"({100 * r4[r][t]['mean_diff_pp'] / r4[r][t]['mean_error_B_pct']:+.2f} %)" for r in RULES
            for t in ("images", "scenes")))
        print(f"  independent draws: max |z| {cc['z_max_abs']:.2f} over {len(cc['z_all'])} comparisons;  agree = {cc['agree']}")
        if not cc["agree"]:
            print("  WARNING: the two implementations disagree; do not quote these numbers", file=sys.stderr)
    results["binary_masks"] = run_binary_variant(grid, args)
    add_variants(grid, args, results, raw)
    results["wall_time_s"] = time.time() - t_start
    (out / "span_test.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out / "span_test.npz", **raw)
    if not args.no_figure:
        make_figure(grid, results, raw, args, out / "span_test.png")
    print_summary(results, args)
    print_variants(results)
    print(f"wrote {out / 'span_test.json'} ({time.time() - t_start:.0f} s)")
    return results


def baseline_min_diagnosis(raw, p_train):
    """``heldout_min_diagnosis`` of the baseline held-out errors (implementation A, both rules; implementation B when
    present) from the raw arrays: {"painter": {...}, "max": {...}, "independent_painter": {...}, ...}."""
    out = {}
    for rule in RULES:
        out[rule] = heldout_min_diagnosis(raw[f"{rule}/P{p_train}/test"])
        if f"indep/{rule}/test" in raw:
            out[f"independent_{rule}"] = heldout_min_diagnosis(raw[f"indep/{rule}/test"])
    return out


def book_block():
    """The "book" entry of the JSON: Table 15.3 and the identity that makes the NMM bound valid."""
    return {
        "table_15_3_nmm_pct": list(BOOK_NMM_PCT), "table_15_3_iem_pct": list(BOOK_IEM_PCT),
        "table_15_3_ba_pct": list(BOOK_BA_PCT), "table_15_3_vbim_pct": list(BOOK_VBIM_PCT), "threshold_pct": THRESH_PCT,
        "identity": ("o = alpha_hat g_r(w_r |beta_hat g_c(w_c x + b)| + p) with alpha_hat = T pinv(G_r); every output is "
                     "T (pinv(G_r) g_r(.)), an element of span(T); the biases b and p sit inside g_c and g_r and eq. 15.1 "
                     "has no output bias. The IEM (CNN) acts after the NMM and is not confined to span(T), so the bound "
                     "applies to the NMM column only."),
        "scenes": {k: [{"name": sc["name"], "shapes": [list(s) for s in sc["shapes"]]} for sc in v]
                   for k, v in SCENES.items()}}


def add_variants(grid, args, results, raw):
    """Compute the review variants and the minimum diagnosis, store them in ``results`` / ``raw`` (in place)."""
    t0 = time.time()
    print("review variants (eps_r law, clipping, constant, metric) ...")
    results["variants"], raw_v = run_variants(grid, args.p_train, args.variant_sets, args.n_test, args.seed)
    raw.update(raw_v)
    results["heldout_min_diagnosis"] = baseline_min_diagnosis(raw, args.p_train)
    results["book"] = book_block()
    results["variants"]["wall_time_s"] = time.time() - t0
    results["variants"]["command"] = "python scripts/span_test.py " + " ".join(sys.argv[1:])


def print_variants(results):
    """Print the variants tables (percent of ||m||)."""
    v = results["variants"]
    print(f"variants at P = {v['p_train']} ({v['n_sets']} training sets, painter's rule); median [min] of the digitized scene / "
          f"median [min] over the 8 named scenes")
    for tag, e in v["eps_ranges"].items():
        print(f"  training eps_r U[{tag.replace('-', ', ')}]: empty scene {e['empty_scene_pct']['mean']:.2f}, held-out median "
              f"{e['heldout_span']['median']:.2f}")
        for key in ("test1", "test4"):
            row = []
            for name in VARIANT_NAMES:
                s = e["scenes"][key][name]
                row.append(f"{name} {s['digitized']['median']:.2f} [{s['digitized']['min']:.2f}] / {s['pooled_over_8_scenes']['median']:.2f} "
                           f"[{s['pooled_over_8_scenes']['min']:.2f}]")
            print(f"    {key}: " + ";  ".join(row))
    for key, b in v["best_reading"].items():
        s = b["smallest_value_over_all_scenes_and_sets"]
        print(f"  best reading {key}: digitized median {b['digitized_median']['median']:.2f} ({b['digitized_median']['reading']}, "
              f"eps {b['digitized_median']['eps_range']}); smallest value anywhere {s['min']:.2f} ({s['scene']}, {s['reading']}, eps {s['eps_range']})")
    fl = v["flatness"]
    gaps = [g for d in fl["digitized_median_test4_minus_test1_pp"].values() for g in d.values()]
    print(f"  Test #4 minus Test #1 (digitized medians) over all readings: {min(gaps):.2f} to {max(gaps):.2f} pp; book NMM: "
          f"{fl['book_nmm_test4_minus_test1_pp']:+.3f} pp (spread {fl['book_nmm_max_minus_min_pp']:.3f} pp)")
    mr = v["metric_reading"]
    print(f"  squared-misfit reading worse than the vacuum guess for all BA/VBIM entries: {mr['squared_reading_worse_than_vacuum']}")
    for rule, d in results["heldout_min_diagnosis"].items():
        print(f"  held-out minimum [{rule}]: pooled min {d['pooled_min']:.2f}, easiest draw #{d['easiest_draw_index']} (mean over sets "
              f"{d['easiest_draw_mean_over_sets']:.2f}), min without it {d['min_without_easiest_draw']:.2f}; median {d['median']:.2f}, "
              f"p05 {d['p05']:.2f}, p0.1 {d['p00_1']:.2f}")


def print_summary(results, args):
    """Print the headline tables (percent of ||m||)."""
    P0 = str(args.p_train)
    fmt = lambda s: (f"median {s['median']:6.2f}  p05 {s['p05']:6.2f}  min {s['min']:6.2f}  "
                     f"frac<=3.3% {s['frac_le_3p3']:.4f}  (n = {s['n']})")
    for rule in RULES:
        rr = results["rules"][rule]
        print(f"[{rule}] (a) held-out draws; vacuum guess: {fmt(rr['test_vacuum_pct'])}")
        for P in results["sweep"]:
            e = rr["sweep"][str(P)]
            print(f"  P = {P:5d} span: {fmt(e['test']['pooled'])}   nearest median {e['test_nearest']['pooled']['median']:.2f}"
                  f"   span+1 median {e['test_const']['pooled']['median']:.2f}")
        e = rr["sweep"][P0]
        print(f"  empty scene (m = 1) bound at P = {P0}: mean {e['empty_scene_pct']['mean']:.2f} %  "
              f"(min {e['empty_scene_pct']['min']:.2f}, max {e['empty_scene_pct']['max']:.2f})")
        t = results["tail_scan"][rule]
        print(f"  tail scan ({t['n_draws']} draws x {t['n_train_sets']} sets): min {t['min']:.2f}  p0.1 {t['p00_1']:.2f}  "
              f"p1 {t['p01']:.2f}  frac<=3.3% {t['frac_le_3p3']:.5f}  <=5% {t['frac_le_5']:.5f}  <=10% {t['frac_le_10']:.5f}")
        for key in ("test1", "test4", "test4_high", "test1_jitter", "test4_jitter"):
            s = e["scenes"][f"{key}@{rule}"]["scene"]["pooled"]
            print(f"  (b) {key:13s} P = {P0}: {fmt(s)}")
        b = results["binary_masks"][rule]
        print(f"  binary-mask rasterization (implementation B, P = {P0}): held-out {fmt(b['test'])}")
        print(f"    test1 {fmt(b['test1'])}")
        print(f"    test4 {fmt(b['test4'])}")


if __name__ == "__main__":
    main()
