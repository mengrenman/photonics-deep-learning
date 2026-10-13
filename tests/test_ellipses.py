"""Tests for bayes_scatter.ellipses: exact ellipse coverage, exact boundary derivatives in JAX, rectangles, and the DESIGN 14.3 truths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapes as sh
from bayes_scatter.geometry import Grid

GRID = Grid(24, 1.5)
RENDER = el.EllipseRenderer(GRID)
E_GENERIC = np.array([0.05, -0.03, 0.4, 0.15, 0.6])  # rotated ellipse inside the DOI, no tangency to a pixel edge


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_a_circle_has_the_coverage_of_the_closed_form_disk_area():
    for xc, yc, r in ((-0.23, 0.04, 0.2), (0.0, 0.0, 0.31), (0.31, -0.2, 0.05)):
        cov = np.asarray(RENDER.coverage(jnp.array([xc, yc, r, r, 0.3])))
        ref = sh.disk_pixel_areas(GRID, xc, yc, r) / GRID.h**2
        assert np.abs(cov - ref).max() < 1e-13
        assert cov.sum() * GRID.h**2 == pytest.approx(np.pi * r * r, rel=1e-13)


def test_a_circle_tangent_to_pixel_edges_is_accurate_to_the_square_root_of_round_off():
    """A circle whose extreme points sit on pixel edges (0.3 - 0.05 = 0.25 = 4 h): the crossing angle arccos(1 - delta) is only known to sqrt(delta) ~ 1e-8, which
    bounds the error of the coverage (and of the derivative) of the two pixels next to the tangent point by ~1e-8 of a pixel; away from tangency it is 1e-14."""
    cov = np.asarray(RENDER.coverage(jnp.array([0.3, -0.2, 0.05, 0.05, 0.0])))
    ref = sh.disk_pixel_areas(GRID, 0.3, -0.2, 0.05) / GRID.h**2
    assert np.abs(cov - ref).max() < 1e-6
    assert cov.sum() * GRID.h**2 == pytest.approx(np.pi * 0.05**2, rel=1e-6)


def test_a_rotated_ellipse_matches_dense_subsampling_and_conserves_area():
    cov = np.asarray(RENDER.coverage(jnp.asarray(E_GENERIC)))
    ref = el.EllipseScene(E_GENERIC[None], 2.0).chi(GRID, ss=400)  # chi = coverage for eps_r = 2
    assert np.abs(cov - ref).max() < 4e-4  # the 400 x 400 sub-sampling is itself accurate to ~1e-4 per pixel
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert cov.sum() * GRID.h**2 == pytest.approx(np.pi * 0.4 * 0.15, rel=1e-13)
    # swapping the semi-axes and adding a quarter turn is the same ellipse
    swapped = np.asarray(RENDER.coverage(jnp.array([0.05, -0.03, 0.15, 0.4, 0.6 + 0.5 * np.pi])))
    assert np.abs(swapped - cov).max() < 1e-12
    # and so is a half turn
    assert np.abs(np.asarray(RENDER.coverage(jnp.array([0.05, -0.03, 0.4, 0.15, 0.6 + np.pi]))) - cov).max() < 1e-12


def test_pixels_inside_have_coverage_one_and_far_pixels_zero():
    cov = np.asarray(RENDER.coverage(jnp.asarray(E_GENERIC)))
    X, Y = GRID.centers()
    c, s = np.cos(0.6), np.sin(0.6)
    q = (((X - 0.05) * c + (Y + 0.03) * s) / 0.4) ** 2 + ((-(X - 0.05) * s + (Y + 0.03) * c) / 0.15) ** 2
    assert np.all(np.abs(cov[q < 0.5] - 1.0) < 1e-13) and np.all(np.abs(cov[q > 1.8])) < 1e-13 and (q < 0.5).any() and (q > 1.8).any()


def test_exact_derivatives_match_central_differences_of_the_exact_coverage():
    _, d = RENDER.coverage_and_derivatives(jnp.asarray(E_GENERIC))
    d = np.asarray(d)
    for m, name in enumerate(el.PARAM_NAMES):
        step = 1e-6
        ep, em = E_GENERIC.copy(), E_GENERIC.copy()
        ep[m] += step
        em[m] -= step
        fd = (np.asarray(RENDER.coverage(jnp.asarray(ep))) - np.asarray(RENDER.coverage(jnp.asarray(em)))) / (2 * step)
        assert rel(d[m], fd) < 1e-6, name


def test_total_area_derivatives_are_pi_b_pi_a_and_zero():
    _, d = RENDER.coverage_and_derivatives(jnp.asarray(E_GENERIC))
    total = np.asarray(d).sum(axis=(1, 2)) * GRID.h**2
    np.testing.assert_allclose(total, [0.0, 0.0, np.pi * 0.15, np.pi * 0.4, 0.0], atol=1e-12)


def test_a_circle_is_stationary_under_rotation_and_a_b_derivatives_add_up_to_the_radius_derivative():
    e = jnp.array([-0.11, 0.043, 0.2, 0.2, 0.7])  # (not tangent to a pixel edge: there a central difference only converges like sqrt(step))
    _, d = RENDER.coverage_and_derivatives(e)
    d = np.asarray(d)
    assert np.abs(d[4]).max() < 1e-11  # a circle does not change under rotation
    eps = 1e-6
    fd_r = (np.asarray(RENDER.coverage(e.at[2].add(eps).at[3].add(eps))) - np.asarray(RENDER.coverage(e.at[2].add(-eps).at[3].add(-eps)))) / (2 * eps)
    assert rel(d[2] + d[3], fd_r) < 1e-6


def test_a_shape_crossing_the_doi_is_truncated_consistently_with_its_derivative():
    e = np.array([0.6, 0.1, 0.35, 0.2, -0.4])  # sticks out of the right edge (x = 0.75)
    cov = np.asarray(RENDER.coverage(jnp.asarray(e)))
    ref = el.EllipseScene(e[None], 2.0).chi(GRID, ss=300)
    assert np.abs(cov - ref).max() < 5e-4 and cov.sum() * GRID.h**2 < np.pi * 0.35 * 0.2
    _, d = RENDER.coverage_and_derivatives(jnp.asarray(e))
    for m in range(5):
        ep, em = e.copy(), e.copy()
        ep[m] += 1e-6
        em[m] -= 1e-6
        fd = (np.asarray(RENDER.coverage(jnp.asarray(ep))) - np.asarray(RENDER.coverage(jnp.asarray(em)))) / 2e-6
        assert rel(np.asarray(d)[m], fd) < 1e-6 if np.linalg.norm(fd) > 0 else True


def test_jax_differentiation_through_the_coverage_uses_the_exact_boundary_derivative():
    w = np.random.default_rng(0).standard_normal((24, 24))
    f = lambda e: jnp.sum(w * RENDER.coverage(e))
    g = np.asarray(jax.grad(f)(jnp.asarray(E_GENERIC)))
    _, d = RENDER.coverage_and_derivatives(jnp.asarray(E_GENERIC))
    np.testing.assert_allclose(g, np.einsum("ij,mij->m", w, np.asarray(d)), rtol=1e-12, atol=1e-12)
    dirn = np.array([0.3, -0.2, 0.5, 0.1, 0.4])
    _, tangent = jax.jvp(RENDER.coverage, (jnp.asarray(E_GENERIC),), (jnp.asarray(dirn),))
    np.testing.assert_allclose(np.asarray(tangent), np.einsum("m,mij->ij", dirn, np.asarray(d)), rtol=1e-12, atol=1e-12)
    many = np.asarray(RENDER.coverage_many(jnp.stack([jnp.asarray(E_GENERIC), jnp.array([-0.2, 0.1, 0.1, 0.1, 0.0])])))
    assert many.shape == (2, 24, 24) and np.abs(many[0] - np.asarray(RENDER.coverage(jnp.asarray(E_GENERIC)))).max() < 1e-12


def test_two_equal_disks_of_f2_reproduce_twodisks_rendering_and_all_five_shape_derivatives():
    """The independent numerical arcs code of ``shapes`` and the JAX ellipse code agree on chi and on dchi/dp for p = (g, r, eps_r, cx, cy)."""
    model = sm.F2()
    obj = sh.TwoDisks(g=0.13, r=0.19, eps_r=2.3, cx=0.03, cy=-0.04)
    p = np.array([obj.g, obj.r, obj.eps_r, obj.cx, obj.cy])
    u = np.asarray(model.u_from_p(p))
    chi = np.asarray(model.chi(RENDER, jnp.asarray(u)))
    assert np.abs(chi - obj.chi(GRID)).max() < 1e-12
    names = ("g", "r", "eps_r", "cx", "cy")
    ref = sh.shape_derivatives(GRID, obj, names)  # d chi / d p, (5, n, n)
    jac_p = np.asarray(jax.jacfwd(lambda q: model.chi(RENDER, model.u_from_p(q)))(jnp.asarray(p)))  # (n, n, 5)
    for k, name in enumerate(names):
        assert rel(np.moveaxis(jac_p, -1, 0)[k], ref[k]) < 1e-9, name


def test_rect_and_disk_rect_areas_are_exact():
    x0, x1, y0, y1 = -0.13, 0.21, -0.07, 0.31
    cov = el.rect_pixel_areas(GRID, x0, x1, y0, y1)
    assert cov.sum() == pytest.approx((x1 - x0) * (y1 - y0), rel=1e-13)
    # a disk wholly inside the rectangle gives the disk area, a disk wholly outside gives 0
    inside = el.disk_rect_pixel_areas(GRID, 0.04, 0.12, 0.05, x0, x1, y0, y1)
    assert inside.sum() == pytest.approx(np.pi * 0.05**2, rel=1e-12)
    assert el.disk_rect_pixel_areas(GRID, 0.5, 0.5, 0.1, x0, x1, y0, y1).sum() == 0.0
    # a disk cut by the rectangle: the half disk when the rectangle edge passes through the center
    half = el.disk_rect_pixel_areas(GRID, 0.0, 0.0, 0.2, 0.0, 0.5, -0.5, 0.5)
    assert half.sum() == pytest.approx(0.5 * np.pi * 0.04, rel=1e-12)


def test_truths_t1_to_t5_render_with_the_documented_geometry_and_conserve_area():
    grid = Grid(48, 1.5)
    t1, t2, t3, t4, t5 = (el.make_truth(k, g=0.1) for k in el.TRUTH_NAMES)
    assert isinstance(t1, sh.TwoDisks) and t1.g == 0.1
    # T2: ellipses 0.2 x 0.23 (long axis y), gap 0.1 along x
    assert t2.ellipses[0, 2] == 0.2 and t2.ellipses[0, 3] == pytest.approx(0.23) and t2.ellipses[1, 0] - t2.ellipses[0, 0] == pytest.approx(0.2 * 2 + 0.1)
    # T3: eps 2.0 and 2.4 (left, right)
    assert t3.eps.tolist() == [2.0, 2.4]
    chi3 = t3.chi(grid)
    assert chi3[:24].max() == pytest.approx(1.0) and chi3[24:].max() == pytest.approx(1.4)
    # T4: union area, true gap 0
    assert t4.true_gap == 0.0 and el.pixel_area_sum_check(grid, t4) == pytest.approx(0.0, abs=1e-8)
    # T5: the third disk (radius 0.06 at (0, 0.1)) does not touch either main disk
    e = t5.ellipses
    for k in (0, 1):
        assert np.hypot(e[k, 0] - e[2, 0], e[k, 1] - e[2, 1]) > e[k, 2] + e[2, 2]
    for t in (t1, t2, t3, t5):
        assert abs(el.pixel_area_sum_check(grid, t)) < 1e-12
    assert t5.total_contrast_exact() == pytest.approx((np.pi * (2 * 0.2**2 + 0.06**2)) * 1.0)
    for t, name in ((t2, "T2"), (t3, "T3"), (t4, "T4"), (t5, "T5")):
        ref = t.chi(grid, ss=80)
        assert np.abs(t.chi(grid) - ref).max() < 1.2e-3 * (0.4 if name == "T3" else 1.0) + 2e-3, name
    with pytest.raises(ValueError):
        el.make_truth("T9")


def test_the_bridge_is_counted_once_where_it_overlaps_the_disks():
    b = el.BridgedDisks(r=0.2, width=0.05, eps_r=3.0)
    grid = Grid(96, 1.5)
    cov = b.coverage(grid)
    assert cov.max() <= 1.0 and cov.sum() * grid.h**2 == pytest.approx(b.union_area_exact(), abs=1e-8)
    assert b.union_area_exact() < 2 * np.pi * 0.04 + 0.05 * 0.4  # strictly less than the sum of the parts
    assert b.chi(grid).max() == pytest.approx(2.0)  # chi = eps_r - 1 where the coverage is one
    # the top of the bridge, outside both disks, is painted
    X, Y = grid.centers()
    sel = (np.abs(X - 0.5 * grid.h) < 1e-9) & (np.abs(Y - 0.18) < grid.h / 2)  # the pixel [0, h] x [0.172, 0.1875] lies in the bridge
    assert sel.any() and cov[sel].min() > 0.99


def test_feature_object_takes_the_regions_of_a_reference_and_the_theta_of_the_truth():
    grid = Grid(64, 1.5)
    truth = el.make_truth("T3", g=0.1)
    ref = el.feature_reference(truth, g=0.1, eps_r=2.2)
    fo = el.FeatureObject(truth, ref)
    assert fo.g == 0.1 and fo.left == ref.left and fo.eps_r == 2.2
    np.testing.assert_array_equal(fo.theta(grid), truth.theta(grid))
    from bayes_scatter import gapvalley as gv

    fg = gv.feature_geometry(grid, fo)
    assert fg.fits_grid and fg.truth_holds is True
    assert gv.feature_geometry(grid, el.FeatureObject(el.make_truth("T4"), el.feature_reference(None, 0.0))).fits_grid is False  # a closed gap is unrepresentable
