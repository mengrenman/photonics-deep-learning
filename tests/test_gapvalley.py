"""Tests for bayes_scatter.gapvalley: regions, weights, Gaussian orthant probabilities and the draw-based indicator."""

import json

import numpy as np
import pytest
from scipy.stats import norm

from bayes_scatter import features as ft
from bayes_scatter import gapvalley as gv
from bayes_scatter import shapes as sh
from bayes_scatter.geometry import Grid


def feature_geometry_full(*args, **kwargs):
    """The full-height strip (band_frac 1), the lenient bound: the pixel counts asserted below are for this band."""
    return gv.feature_geometry(*args, band_frac=1.0, **kwargs)

GRID64 = Grid(64, 1.5)  # the inference grid of DESIGN 13.3: h = 0.0234375


def test_strip_and_core_counts_match_a_direct_count_and_regions_do_not_overlap():
    obj = sh.TwoDisks(g=0.1, r=0.2, eps_r=2.0)
    fg = feature_geometry_full(GRID64, obj)
    h = GRID64.h
    xs = (np.arange(64) + 0.5) * h - 0.75
    n_cols = sum(abs(x) < 0.05 for x in xs)
    n_rows = sum(abs(y) <= 0.2 for y in xs)
    assert (n_cols, n_rows) == (4, 18) and fg.n_strip == 72
    assert fg.strip_width_pixels == pytest.approx(0.1 / h)
    for core, center in ((fg.core1, obj.left), (fg.core2, obj.right)):
        X, Y = GRID64.centers()
        assert np.array_equal(core, np.hypot(X - center[0], Y - center[1]) <= 0.1)
    assert not (fg.strip & fg.core1).any() and not (fg.strip & fg.core2).any() and not (fg.core1 & fg.core2).any()
    # mirror symmetry of the symmetric object
    assert np.array_equal(fg.strip, fg.strip[::-1]) and np.array_equal(fg.core1, fg.core2[::-1])
    for k in ("strip", "core1", "core2", "mid", "c1", "c2"):
        w = fg.weights[k]
        assert w.shape == (4096,) and np.all(w >= 0) and w.sum() == pytest.approx(1.0, abs=1e-14)
    json.dumps(fg.describe())


def test_a_narrower_band_keeps_a_subset_of_the_strip_rows_and_describes_itself():
    obj = sh.TwoDisks(g=0.1, r=0.2, eps_r=2.0)
    full, half, quarter = (gv.feature_geometry(GRID64, obj, band_frac=b) for b in (1.0, 0.5, 0.25))
    assert full.n_strip == 72 and half.n_strip == 4 * 8 and quarter.n_strip == 4 * 4  # |y| <= 0.1 -> 8 rows, |y| <= 0.05 -> 4 rows
    assert (half.strip & ~full.strip).sum() == 0 and (quarter.strip & ~half.strip).sum() == 0
    assert half.describe()["band_frac"] == 0.5 and full.describe()["band_frac"] == 1.0
    # the truth at one pixel of gap: the strip pixels near y = 0 are partly covered by the disks, those far from y = 0 (where the circles bend apart) are
    # empty, so the strip mean of theta RISES as the band narrows (0.083, 0.186, 0.327 for bands 1, 1/2, 1/4): the full band is the lenient definition
    narrow = sh.TwoDisks(g=0.025, eps_r=2.0)
    th = narrow.theta(GRID64)
    means = [float(gv.feature_geometry(GRID64, narrow, band_frac=b).weights["strip"] @ th.ravel()) for b in (1.0, 0.5, 0.25)]
    assert 0.0 < means[0] < means[1] < means[2] and means[0] == pytest.approx(0.0827, abs=1e-3) and means[1] == pytest.approx(0.186, abs=1e-3) and means[2] == pytest.approx(0.327, abs=1e-3)


def test_a_gap_narrower_than_a_pixel_is_unrepresentable_and_has_no_region_probability():
    h = GRID64.h
    assert not feature_geometry_full(GRID64, sh.TwoDisks(g=0.0125)).representable  # 0.53 pixel
    assert feature_geometry_full(GRID64, sh.TwoDisks(g=0.025)).representable  # 1.07 pixels, two pixel centers in the gap
    assert feature_geometry_full(GRID64, sh.TwoDisks(g=0.025)).n_strip == 2 * 18
    assert feature_geometry_full(GRID64, sh.TwoDisks(g=h * 0.999)).representable is False
    fg = feature_geometry_full(GRID64, sh.TwoDisks(g=0.0125, eps_r=2.0))
    rng = np.random.default_rng(0)
    out = gv.draws_valley(fg, rng.standard_normal((3, 10, 64, 64)))
    assert out["region"] is None and out["region_status"] == "unrepresentable on this grid" and 0.0 <= out["point"]["p_valley"] <= 1.0
    res = gv.gaussian_valley(fg, np.zeros(4096), lambda B: B)
    assert res["region"] is None and res["region_status"] == "unrepresentable on this grid"
    assert res["point"] is not None and "single-point" in res["point_label"]
    assert gv.truth_valley(fg, np.zeros((64, 64)))["region"]["holds"] is None


def test_bilinear_weights_reproduce_the_interpolation_of_features():
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((64, 64))
    for x, y in ((0.0, 0.0), (-0.275, 0.013), (0.7, -0.74), (-0.9, 0.9), (0.0117, 0.0117)):
        w = gv.bilinear_weights(GRID64, x, y)
        assert w.sum() == pytest.approx(1.0) and (w > 0).sum() <= 4
        assert w @ theta.ravel() == pytest.approx(ft.bilinear(theta, GRID64, x, y), abs=1e-13)
    # a point at a pixel center returns that pixel
    x0 = (10 + 0.5) * GRID64.h - 0.75
    y0 = (20 + 0.5) * GRID64.h - 0.75
    w = gv.bilinear_weights(GRID64, x0, y0).reshape(64, 64)
    assert w[10, 20] == pytest.approx(1.0)


def test_truth_values_of_the_pixelized_object():
    """The rendering of the truth on the inference grid has the valley at every representable gap of the sweep, but for g = 0.025 (1.07
    pixels) the strip pixels are partly covered by the disks (the strip mean of theta is 0.083 against 0 for g >= 0.05), so the grid
    already blurs the valley before any inference, and the single-point version (the bilinear value at the midpoint, 0.370 against the
    threshold 0.346) already fails on the truth itself while the region version holds."""
    strips = {}
    for g in (0.25, 0.1, 0.05, 0.025):
        obj = sh.TwoDisks(g=g, eps_r=2.0)
        fg = feature_geometry_full(GRID64, obj)
        res = gv.truth_valley(fg, obj.theta(GRID64))
        assert res["region"]["holds"] is True, g
        assert res["point"]["holds"] is (g > 0.025), g  # the pixelized truth fails the single-point version at one pixel of gap (0.370 > 0.346)
        assert res["region"]["core1"] == pytest.approx(np.log(2.0))
        strips[g] = res["region"]["strip"]
    assert strips[0.25] == pytest.approx(0.0, abs=1e-12) and strips[0.1] == pytest.approx(0.0, abs=1e-12) and strips[0.05] == pytest.approx(0.0, abs=1e-12)
    assert 0.01 < strips[0.025] < gv.delta_of(2.0)
    # a made-up truth whose strip is half filled does not have the valley
    fg = feature_geometry_full(GRID64, sh.TwoDisks(g=0.1, eps_r=2.0))
    th = sh.TwoDisks(g=0.1, eps_r=2.0).theta(GRID64)
    th[fg.strip] = 0.6 * np.log(2.0)
    assert gv.truth_valley(fg, th)["region"]["holds"] is False


def _small_gaussian(seed=3, tau=60.0):
    """A 20 x 20 grid, a representable object, a mean with a valley of depth about delta and an SPD precision with correlations."""
    grid = Grid(20, 1.5)
    obj = sh.TwoDisks(g=0.2, r=0.2, eps_r=2.0)
    fg = feature_geometry_full(grid, obj)
    assert fg.representable and fg.n_strip == 12
    mean = np.full((20, 20), 0.3)
    mean[fg.core1] = mean[fg.core2] = 0.55
    mean[fg.strip] = 0.15  # depth 0.4 against delta = 0.347
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((400, 60))
    H = tau * np.eye(400) + A @ A.T / 6.0
    return fg, mean, H


def test_gaussian_functionals_match_the_dense_computation():
    fg, mean, H = _small_gaussian()
    Hinv = np.linalg.inv(H)
    W = np.stack([fg.weights["strip"], fg.weights["core1"], fg.weights["core2"]])
    m, S = gv.gaussian_functionals(W, mean, lambda B: np.linalg.solve(H, B))
    np.testing.assert_allclose(m, W @ mean.ravel(), rtol=1e-14)
    np.testing.assert_allclose(S, W @ Hinv @ W.T, rtol=1e-10, atol=1e-14)
    np.testing.assert_array_equal(S, S.T)
    assert np.all(np.linalg.eigvalsh(S) > 0)


def test_orthant_probability_against_closed_forms_and_monte_carlo():
    assert gv.orthant_probability([0.3], [[0.25]]) == pytest.approx(norm.cdf(-0.3 / 0.5), abs=1e-12)
    assert gv.orthant_probability([0.0, 0.0], np.eye(2)) == pytest.approx(0.25, abs=1e-8)
    m, S = np.array([-0.2, 0.1]), np.array([[0.04, 0.0], [0.0, 0.09]])  # independent: the product of the marginals
    assert gv.orthant_probability(m, S) == pytest.approx(norm.cdf(0.2 / 0.2) * norm.cdf(-0.1 / 0.3), abs=1e-8)
    assert gv.orthant_probability([0.0, 0.0], [[1.0, 1.0], [1.0, 1.0]]) == pytest.approx(0.5, abs=1e-6)  # fully correlated, singular
    m, S = np.array([-0.15, 0.05]), np.array([[0.04, 0.03], [0.03, 0.05]])  # correlation 0.67
    rng = np.random.default_rng(5)
    x = rng.multivariate_normal(m, S, size=2_000_000)
    p_mc = float(np.mean((x[:, 0] < 0) & (x[:, 1] < 0)))
    se = np.sqrt(p_mc * (1 - p_mc) / 2_000_000)
    assert abs(gv.orthant_probability(m, S) - p_mc) < 4 * se
    # the orthant of positive correlation exceeds the product of the marginals
    assert gv.orthant_probability(m, S) > norm.cdf(0.15 / 0.2) * norm.cdf(-0.05 / np.sqrt(0.05))


def test_gaussian_valley_probability_matches_draws_and_the_indicator():
    fg, mean, H = _small_gaussian()
    L = np.linalg.cholesky(H)
    res = gv.gaussian_valley(fg, mean, lambda B: np.linalg.solve(H, B))
    assert res["region_status"] == "ok" and res["region"] is not None
    n_draws = 200_000
    rng = np.random.default_rng(7)
    z = rng.standard_normal((n_draws, 400))
    theta = (mean.ravel()[None, :] + np.linalg.solve(L.T, z.T).T).reshape(n_draws, 20, 20)  # draws of N(mean, H^-1)
    emp = gv.draws_valley(fg, theta)
    for key in ("region", "point"):
        p, p_mc = res[key]["p_valley"], emp[key]["p_valley"]
        se = np.sqrt(max(p_mc * (1 - p_mc), 1e-4) / n_draws)
        assert abs(p - p_mc) < 4 * se, (key, p, p_mc)
    assert 0.1 < res["region"]["p_valley"] < 0.95  # an intermediate probability: the test is not vacuous
    ind = gv.valley_indicator(theta[:1000], fg.weights["strip"], fg.weights["core1"], fg.weights["core2"], fg.delta)
    assert ind.shape == (1000,) and ind.mean() == pytest.approx(emp["region"]["p_valley"], abs=0.05)
    assert res["region"]["p_valley"] <= min(res["region"]["p_vs_core1"], res["region"]["p_vs_core2"]) + 1e-9
    assert emp["region"]["per_chain"][0] == pytest.approx(emp["region"]["p_valley"]) and emp["region"]["n_draws"] == n_draws
    # a narrower gap in the mean lowers the probability; reaching the full depth raises it
    deep = mean.copy()
    deep[fg.strip] = -0.3
    assert gv.gaussian_valley(fg, deep, lambda B: np.linalg.solve(H, B))["region"]["p_valley"] > 0.95
    flat = np.full((20, 20), 0.3)
    assert gv.gaussian_valley(fg, flat, lambda B: np.linalg.solve(H, B))["region"]["p_valley"] < 0.05
    json.dumps(res)


def test_draws_valley_accepts_chains_and_reports_per_chain_values():
    fg, mean, H = _small_gaussian()
    rng = np.random.default_rng(2)
    th = mean[None, None] + 0.02 * rng.standard_normal((3, 50, 20, 20))
    out = gv.draws_valley(fg, th)
    assert len(out["region"]["per_chain"]) == 3 and out["region"]["n_draws"] == 150
    single = gv.draws_valley(fg, th[0])
    assert single["region"]["per_chain"] == [out["region"]["per_chain"][0]]


def test_a_variant_whose_pixelized_truth_fails_the_valley_is_unrepresentable():
    """Milestone-6 review: eps_r 4, g = 0.025 (1.07 pixels), strip of a quarter of the disk height: the strip pixels are about half disk in the truth itself (mean
    theta 0.77 against the cores' log 4 = 1.39: depth 0.62 < delta = 0.69), so the grid cannot show the valley and a probability of it is not reported. The full-height
    and half-height strips of the same object hold (the first contains the lens-shaped void above and below the neck). The geometric condition (g >= h) alone, the old
    test, called all three representable."""
    obj = sh.TwoDisks(g=0.025, eps_r=4.0)
    full = feature_geometry_full(GRID64, obj)
    half = gv.feature_geometry(GRID64, obj, band_frac=0.5)
    quarter = gv.feature_geometry(GRID64, obj, band_frac=0.25)
    for fg in (full, half):
        assert fg.fits_grid and fg.truth_holds is True and fg.representable and fg.status == "ok"
    assert quarter.fits_grid and quarter.truth_holds is False and not quarter.representable and quarter.status == gv.UNREPRESENTABLE_TRUTH
    assert gv.is_unrepresentable(quarter.status) and gv.is_unrepresentable(gv.UNREPRESENTABLE) and not gv.is_unrepresentable("ok")
    d = quarter.describe()
    assert d["truth_holds"] is False and d["fits_grid"] is True and d["representable"] is False and d["status"] == gv.UNREPRESENTABLE_TRUTH
    res = gv.gaussian_valley(quarter, np.zeros(4096), lambda B: B)
    assert res["region"] is None and res["region_status"] == gv.UNREPRESENTABLE_TRUTH and res["point"] is not None
    out = gv.draws_valley(quarter, np.random.default_rng(0).standard_normal((2, 5, 64, 64)))
    assert out["region"] is None and out["region_status"] == gv.UNREPRESENTABLE_TRUTH
    # the truth values themselves are still reported (holds False, not None): the variant fits the grid, the truth just does not show it
    tv = gv.truth_valley(quarter, obj.theta(GRID64))
    assert tv["region"]["holds"] is False and tv["region"]["strip"] == pytest.approx(0.768, abs=1e-3) and tv["region"]["core1"] == pytest.approx(np.log(4.0))
    # a geometry that does not fit the grid keeps the old status and holds = None
    thin = feature_geometry_full(GRID64, sh.TwoDisks(g=0.0125, eps_r=4.0))
    assert thin.truth_holds is None and thin.status == gv.UNREPRESENTABLE and gv.truth_valley(thin, np.zeros((64, 64)))["region"]["holds"] is None
    # a gap that is wide enough keeps every variant
    wide = sh.TwoDisks(g=0.1, eps_r=4.0)
    assert all(gv.feature_geometry(GRID64, wide, band_frac=b).representable for b in (1.0, 0.5, 0.25))


def test_default_feature_is_the_half_height_strip():
    """Decision of 2026-10-13: the headline gap valley uses |y - cy| <= r/2; the full band is reported only as a lenient bound."""
    obj = sh.TwoDisks(g=0.1, eps_r=2.0)
    default, half, full = gv.feature_geometry(GRID64, obj), gv.feature_geometry(GRID64, obj, band_frac=0.5), feature_geometry_full(GRID64, obj)
    assert gv.BAND_FRAC == 0.5 and default.band_frac == 0.5
    assert np.array_equal(default.weights["strip"], half.weights["strip"]) and 0 < default.n_strip < full.n_strip
