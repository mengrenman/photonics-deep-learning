"""Tests for bayes_scatter.features: pooled moments, edge/z statistics, bilinear sampling and the DESIGN 11.3 features."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from bayes_scatter import features as ft
from bayes_scatter.geometry import Grid
from bayes_scatter.phantoms import austria_fixed


def test_pooled_moments_are_the_mixture_moments_of_equal_length_chains():
    mean_c = np.array([[1.0, -2.0], [3.0, -2.0]])  # (chains=2, pixels=2)
    var_c = np.array([[0.5, 2.0], [1.5, 2.0]])
    mean, std = ft.pooled_moments(mean_c, var_c)
    np.testing.assert_allclose(mean, [2.0, -2.0])
    np.testing.assert_allclose(std**2, [1.0 + 1.0, 2.0])  # mean variance + variance of the chain means (biased)
    rng = np.random.default_rng(0)  # against the moments of the concatenated draws
    x = rng.standard_normal((4, 20000, 3)) * np.array([1.0, 2.0, 0.5]) + np.array([0.0, 1.0, 3.0])
    m, s = ft.pooled_moments(x.mean(axis=1), x.var(axis=1, ddof=1))
    np.testing.assert_allclose(m, x.reshape(-1, 3).mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(s, x.reshape(-1, 3).std(axis=0), rtol=1e-4)


def test_edge_mask_marks_the_boundary_band_of_the_object():
    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    edge = ft.edge_mask(chi)
    inside = chi > 1e-9
    assert edge[inside].any() and edge[~inside].any()  # the band reaches one pixel out of the object
    assert not edge[24, 47] and not edge[0, 0] and not edge[47, 47]  # far outside
    deep = (chi > 1.0 - 1e-9)
    from scipy.ndimage import binary_erosion

    assert not np.any(edge & binary_erosion(deep, iterations=1))  # not in the interior of the full-contrast core
    assert 0.1 < edge.mean() < 0.5


def test_z_statistics_counts_and_edge_share():
    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    truth = np.log1p(chi)
    std = np.full(truth.shape, 0.05)
    edge = ft.edge_mask(chi)
    mean = truth.copy()
    mean[edge] += 0.5 * std[edge]  # |z| = 0.5 everywhere: nothing flagged
    st = ft.z_statistics(mean, std, truth, chi)
    assert st["frac_abs_z_gt_3"] == 0.0 and np.isnan(st["edge_share"]) and st["max_abs_z"] == pytest.approx(0.5)
    k = np.argwhere(edge)[:7]
    for i, j in k:
        mean[i, j] = truth[i, j] + 4.0 * std[i, j]
    mean[0, 0] = truth[0, 0] - 5.0 * std[0, 0]  # one flagged pixel far from any edge
    st = ft.z_statistics(mean, std, truth, chi)
    np.testing.assert_allclose(st["frac_abs_z_gt_3"], 8 / truth.size)
    np.testing.assert_allclose(st["edge_share"], 7 / 8)
    np.testing.assert_allclose(st["edge_fraction_of_doi"], edge.mean())
    assert st["max_abs_z"] == pytest.approx(5.0)
    flagged = np.abs((mean - truth) / std) > 3.0
    inside = chi > 1e-9
    assert st["frac_inside"] == pytest.approx(flagged[inside].sum() / inside.sum())
    assert st["frac_outside"] == pytest.approx(flagged[~inside].sum() / (~inside).sum())


def test_disagreement_formula():
    d = ft.disagreement(np.array([0.0, 1.0]), np.array([0.3, 0.1]), np.array([0.5, 1.0]), np.array([0.4, 0.2]))
    np.testing.assert_allclose(d, [0.5 / 0.5, 0.0])


def test_bilinear_reproduces_affine_fields_at_arbitrary_points_and_clamps():
    grid = Grid(10, 2.0, center=(0.3, -0.2))
    X, Y = grid.centers()
    theta = 0.7 * X - 1.3 * Y + 0.2
    for x, y in [(0.3, -0.2), (0.11, 0.57), (-0.5, -0.9)]:
        np.testing.assert_allclose(ft.bilinear(theta, grid, x, y), 0.7 * x - 1.3 * y + 0.2, atol=1e-12)
    np.testing.assert_allclose(ft.bilinear(theta, grid, X[3, 4], Y[3, 4]), theta[3, 4], atol=1e-12)
    np.testing.assert_allclose(ft.bilinear(theta, grid, 50.0, 50.0), theta[-1, -1])  # clamped to the lattice
    batch = np.stack([theta, 2 * theta])
    np.testing.assert_allclose(ft.bilinear(batch, grid, 0.11, 0.57), [theta_at(theta, grid, 0.11, 0.57) * c
                                                                       for c in (1, 2)], atol=1e-12)


def theta_at(theta, grid, x, y):
    return float(ft.bilinear(theta, grid, x, y))


def test_austria_points_follow_the_phantom():
    grid = Grid(48, 2.0)
    pts = ft.austria_points(grid)
    np.testing.assert_allclose(pts.disk_centers, [(-0.36, 0.4), (0.36, 0.4)])
    np.testing.assert_allclose(pts.gap_mid, (0.0, 0.4))
    np.testing.assert_allclose(pts.ring_center, (0.0, -0.32))
    np.testing.assert_allclose(pts.ring_radii, (0.36, 0.52))
    np.testing.assert_allclose(pts.compass, [(0.44, -0.32), (0.0, 0.12), (-0.44, -0.32), (0.0, -0.76)], atol=1e-12)
    area = np.pi * (0.52**2 - 0.36**2)
    assert abs(pts.annulus.sum() * grid.h**2 / area - 1.0) < 0.05
    X, Y = grid.centers()
    k = np.argmin(np.hypot(X - 0.0, Y + 0.32))
    assert not pts.annulus.ravel()[k]  # the pixel nearest the ring center is not in the annulus
    # the geometry matches austria_fixed: the phantom is full contrast at the disk centers and compass points, empty at
    # the gap midpoint and the ring center
    chi = austria_fixed(Grid(96, 2.0), 2.0)
    g96 = Grid(96, 2.0)
    for p in pts.disk_centers + pts.compass:
        assert ft.bilinear(chi, g96, *p) == pytest.approx(1.0)
    for p in (pts.gap_mid, pts.ring_center):
        assert ft.bilinear(chi, g96, *p) == pytest.approx(0.0, abs=1e-12)


def test_features_hold_for_the_truth_fail_for_vacuum_and_for_a_blurred_object():
    grid = Grid(48, 2.0)
    theta = np.log1p(austria_fixed(grid, 2.0))
    ind = ft.feature_indicators(theta, grid)
    assert all(bool(v) for v in ind.values())
    vac = ft.feature_indicators(np.zeros_like(theta), grid)
    assert not any(bool(v) for v in vac.values())  # nothing is present in an empty scene
    blurred = ft.feature_indicators(gaussian_filter(theta, 4.0), grid)  # sigma = 4 pixels = 0.17 wavelengths
    assert not blurred["gap_valley"] and not blurred["ring_continuity"]  # the gap fills in, the thin ring fades


def test_feature_probabilities_pool_and_split_chains():
    grid = Grid(48, 2.0)
    theta = np.log1p(austria_fixed(grid, 2.0))
    rng = np.random.default_rng(1)
    draws = theta + 0.02 * rng.standard_normal((3, 50, 48, 48))
    draws[2, :25] = 0.0  # half of chain 2 is empty
    pr = ft.feature_probabilities(draws, grid)
    assert set(pr) == {"gap_valley", "ring_hole", "ring_continuity"}
    for name, v in pr.items():
        assert v["p_chain"][0] == 1.0 and v["p_chain"][1] == 1.0 and v["p_chain"][2] == pytest.approx(0.5)
        assert v["p"] == pytest.approx((1 + 1 + 0.5) / 3)
        assert v["mcse"] == pytest.approx(np.std(v["p_chain"], ddof=1) / np.sqrt(3))
    assert np.isnan(ft.feature_probabilities(draws[:1], grid)["gap_valley"]["mcse"])


def test_verdict_rule_on_the_converged_runs():
    conv = {"a": True, "b": True, "c": False}
    nomin = {"required_families": ()}  # the rule on the votes alone, without the DESIGN minimum
    assert ft.verdict({"a": 0.99, "b": 0.97, "c": 0.99}, conv, **nomin) == "data-determined (present)"
    assert ft.verdict({"a": 0.01, "b": 0.04, "c": 0.01}, conv, **nomin) == "data-determined (absent)"
    assert ft.verdict({"a": 0.99, "b": 0.5, "c": 0.99}, conv, **nomin) == "not determined"
    assert ft.verdict({"a": 0.99, "b": 0.01, "c": 0.5}, conv, **nomin) == "not determined"
    assert ft.verdict({"a": 0.95, "b": 0.99, "c": 1.0}, conv, **nomin) == "not determined"  # P must exceed 0.95 strictly
    assert ft.verdict({"a": 0.99, "b": 0.99}, {"a": True, "b": False}).startswith("no verdict")


FAMILY = {"pc_matern": "matern", "pc_gmrf": "gmrf", "pc_qg05": "qggmrf", "pc_qg10": "qggmrf"}


def test_verdict_is_provisional_without_the_design_minimum_pair():
    """The review finding: Matern + two q-GGMRF settings is not the DESIGN 8.7 minimum (Matern against GMRF)."""
    conv = {"pc_matern": True, "pc_gmrf": False, "pc_qg05": True, "pc_qg10": True}
    p = {"pc_matern": 0.99, "pc_gmrf": 0.996, "pc_qg05": 1.0, "pc_qg10": 1.0}
    v = ft.verdict(p, conv, family=FAMILY)
    assert v.startswith("provisional (present)") and "matern, qggmrf" in v and "gmrf" in v.split("needs")[1]
    assert not v.startswith("data-determined")
    # the same votes with the GMRF converged and agreeing are data-determined
    assert ft.verdict(p, {**conv, "pc_gmrf": True}, family=FAMILY) == "data-determined (present)"
    # absent features are provisional in the same way
    q = {k: 0.01 for k in p}
    assert ft.verdict(q, conv, family=FAMILY).startswith("provisional (absent)")
    assert ft.verdict(q, {**conv, "pc_gmrf": True}, family=FAMILY) == "data-determined (absent)"


def test_two_settings_of_one_prior_family_are_not_two_priors():
    conv = {"pc_qg05": True, "pc_qg10": True, "pc_matern": False, "pc_gmrf": False}
    p = {"pc_qg05": 1.0, "pc_qg10": 1.0, "pc_matern": 1.0, "pc_gmrf": 1.0}
    v = ft.verdict(p, conv, family=FAMILY)
    assert v.startswith("provisional (present)") and "agreed by qggmrf only" in v
    # with a converged Matern but no GMRF the minimum is still missing
    assert ft.verdict(p, {**conv, "pc_matern": True}, family=FAMILY).startswith("provisional")


def test_tv_is_a_family_of_its_own_and_two_weights_are_one_prior():
    """Milestone 5: Matern + GMRF (the DESIGN minimum) + q-GGMRF + TV; two TV weights count once."""
    fam = {**FAMILY, "pc_tv4": "tv", "pc_tv12": "tv", "pc_gmrf_adapt": "gmrf"}
    conv = {"pc_matern": True, "pc_gmrf": False, "pc_gmrf_adapt": True, "pc_qg05": True, "pc_qg10": True, "pc_tv4": True,
            "pc_tv12": True}
    p = {k: 1.0 for k in conv}
    assert ft.verdict(p, conv, family=fam) == "data-determined (present)"  # converged Matern and (adapted) GMRF are both there
    v = ft.verdict(p, {**conv, "pc_gmrf_adapt": False}, family=fam)
    assert v.startswith("provisional (present)") and "matern, qggmrf, tv" in v
    only_tv = {"pc_tv4": True, "pc_tv12": True, "pc_matern": False, "pc_gmrf": False, "pc_gmrf_adapt": False, "pc_qg05": False,
               "pc_qg10": False}
    assert "agreed by tv only" in ft.verdict(p, only_tv, family=fam)
    # a TV posterior that gives the other answer: the converged runs disagree
    assert ft.verdict({**p, "pc_tv4": 0.0, "pc_tv12": 0.0}, conv, family=fam) == "not determined"


def test_an_unconverged_run_that_disagrees_blocks_the_verdict():
    """Ring continuity of the milestone: converged runs say P = 1, the rough GMRF (unconverged, falling) says 0.93."""
    conv = {"pc_matern": True, "pc_gmrf": False, "pc_qg05": True, "pc_qg10": True}
    p = {"pc_matern": 1.0, "pc_gmrf": 0.929, "pc_qg05": 1.0, "pc_qg10": 1.0}
    v = ft.verdict(p, conv, family=FAMILY)
    assert v.startswith("not established") and "gmrf" in v and "present" in v
    # an unconverged run that agrees does not block
    assert ft.verdict({**p, "pc_gmrf": 0.99}, conv, family=FAMILY).startswith("provisional (present)")
    # absent side: an unconverged run at P = 0.2 blocks, one at 0.01 does not
    q = {"pc_matern": 0.0, "pc_gmrf": 0.2, "pc_qg05": 0.0, "pc_qg10": 0.0}
    assert ft.verdict(q, conv, family=FAMILY).startswith("not established")
    assert ft.verdict({**q, "pc_gmrf": 0.01}, conv, family=FAMILY).startswith("provisional (absent)")
    # disagreement among the converged runs is "not determined" whatever the unconverged ones say
    assert ft.verdict({**p, "pc_qg10": 0.5}, conv, family=FAMILY) == "not determined"


def test_merge_moments_equals_the_moments_of_the_concatenated_segments():
    rng = np.random.default_rng(3)
    segs = [rng.standard_normal((n, 2, 3)) * (k + 1.0) + k for k, n in enumerate((7, 50, 13))]
    mean, var = ft.merge_moments([s.mean(axis=0) for s in segs], [s.var(axis=0, ddof=1) for s in segs],
                                 [len(s) for s in segs])
    allx = np.concatenate(segs)
    np.testing.assert_allclose(mean, allx.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(var, allx.var(axis=0, ddof=1), rtol=1e-12)
    m1, v1 = ft.merge_moments([segs[1].mean(axis=0)], [segs[1].var(axis=0, ddof=1)], [len(segs[1])])
    np.testing.assert_allclose(v1, segs[1].var(axis=0, ddof=1), rtol=1e-12)  # one segment: identity
