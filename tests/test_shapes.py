"""Tests for bayes_scatter.shapes: exact coverage, exact boundary-integral shape derivatives, the two-disk family."""

import numpy as np
import pytest

from bayes_scatter import shapes as sh
from bayes_scatter.geometry import Grid

GRID = Grid(24, 1.5)
OBJ = sh.TwoDisks(g=0.15, r=0.2, eps_r=2.0)


def rel_per_param(a, b):
    """Frobenius relative difference of two (K, n, n) derivative stacks, per parameter."""
    return np.linalg.norm(a - b, axis=(1, 2)) / np.linalg.norm(b, axis=(1, 2))


def test_gap_is_edge_to_edge_for_any_radius_difference():
    for obj in (OBJ, OBJ.replace(dr=0.06), OBJ.replace(cx=0.1, dr=-0.08)):
        (x1, _, r1), (x2, _, r2) = obj.left, obj.right
        assert (x2 - r2) - (x1 + r1) == pytest.approx(obj.g, abs=1e-15)
        assert 0.5 * (r1 + r2) == pytest.approx(obj.r)
        assert 0.5 * (x1 + x2) == pytest.approx(obj.cx)


def test_invalid_objects_are_rejected():
    with pytest.raises(ValueError):
        sh.TwoDisks(g=-0.01)
    with pytest.raises(ValueError):
        sh.TwoDisks(g=0.1, r=0.1, dr=0.3)
    with pytest.raises(ValueError):  # the right disk leaves the DOI
        sh.TwoDisks(g=0.1, cx=0.5).check_inside(GRID)
    with pytest.raises(ValueError):
        sh.shape_derivatives(GRID, sh.TwoDisks(g=0.1, cx=0.5))


def test_exact_coverage_matches_dense_subsampling_and_conserves_area():
    cov = OBJ.coverage(GRID)
    ref = OBJ.coverage(GRID, ss=300)
    assert np.abs(cov - ref).max() < 5e-4  # the 300 x 300 sub-sampling is itself accurate to ~1e-4 per pixel
    assert cov.min() >= 0.0 and cov.max() <= 1.0 + 1e-14
    assert cov.sum() * GRID.h**2 == pytest.approx(2.0 * np.pi * OBJ.r**2, rel=1e-13)
    asym = OBJ.replace(dr=0.06, cx=0.05, cy=-0.1)
    assert asym.coverage(GRID).sum() * GRID.h**2 == pytest.approx(np.pi * (0.23**2 + 0.17**2), rel=1e-13)
    # chi is the volume-averaged contrast: (eps_r - 1) * coverage
    np.testing.assert_allclose(OBJ.chi(GRID), (OBJ.eps_r - 1.0) * cov, rtol=0, atol=0)
    np.testing.assert_allclose(OBJ.theta(GRID), np.log1p(OBJ.chi(GRID)))


def test_a_pixel_inside_a_disk_has_coverage_one_and_far_pixels_zero():
    cov = OBJ.coverage(GRID)
    X, Y = GRID.centers()
    (x1, y1, r1) = OBJ.left
    deep = np.hypot(X - x1, Y - y1) < r1 - GRID.h
    far = (np.hypot(X - x1, Y - y1) > r1 + GRID.h) & (np.hypot(X - OBJ.right[0], Y - OBJ.right[1]) > OBJ.right[2] + GRID.h)
    assert deep.any() and np.all(np.abs(cov[deep] - 1.0) < 1e-13)
    assert far.any() and np.all(np.abs(cov[far]) < 1e-13)


def test_symmetric_object_is_exactly_mirror_symmetric_and_the_derivatives_have_the_irrep_parity():
    cov = OBJ.coverage(GRID)
    np.testing.assert_allclose(cov, cov[::-1, :], atol=1e-13)
    d = sh.shape_derivatives(GRID, OBJ)
    for k, name in enumerate(sh.PARAM_NAMES):
        sign = 1.0 if name in sh.EVEN_PARAMS else -1.0
        np.testing.assert_allclose(d[k], sign * d[k][::-1, :], atol=1e-11 * np.abs(d[k]).max(), err_msg=name)


def test_exact_derivatives_match_central_differences_of_the_exact_rendering():
    d = sh.shape_derivatives(GRID, OBJ)
    fd = sh.shape_derivatives_fd(GRID, OBJ, step=1e-5)
    assert rel_per_param(d, fd).max() < 1e-6  # round-off of the difference quotient, ~1e-8


def test_agreement_with_finite_differences_of_a_supersampled_rendering_is_limited_by_the_rendering():
    """Measured on this grid (n = 24, h = 0.0625), 128 x 128 sub-samples, step 0.004 (h / 16): relative Frobenius differences
    6.0e-3 (g), 7.4e-3 (r), 1.0e-4 (eps_r), 3.0e-3 (cx), 6.5e-3 (cy), 6.5e-3 (dr). Two things limit them, neither is an error of the
    exact derivative: (i) the sub-sampled rendering is a STEP function of every shape parameter (a sub-sample that changes side changes
    the coverage by 1 / ss^2), so its difference quotient has noise ~ 1 / (ss^2 step) per boundary pixel: 8 x 8 sub-samples at the same
    step give 6e-2 to 0.35, 32 x 32 give 1.3e-2 to 2.7e-2; eps_r, which moves no boundary, is accurate to 1e-4; (ii) the central formula
    has truncation error O((displacement / h)^2) and O(sqrt(step)) at tangent configurations, which is why a larger step does not help
    (step 0.016 = h / 4 raises r, cx and cy to 0.05 to 0.23). The exact-rendering difference above agrees to 1e-8."""
    d = sh.shape_derivatives(GRID, OBJ)
    fd = sh.shape_derivatives_fd(GRID, OBJ, step=0.004, ss=128)
    err = rel_per_param(d, fd)
    assert err.max() < 1.2e-2
    assert err.max() > 1e-3  # the sub-sampled rendering does limit the agreement (the exact-rendering difference gives 1e-8)
    assert err[sh.PARAM_NAMES.index("eps_r")] < 1e-3
    coarse = sh.shape_derivatives_fd(GRID, OBJ, step=0.004, ss=8)  # 8 x 8 sub-samples: noise 1 / (64 * 0.004) = 4 per pixel
    assert rel_per_param(d, coarse).max() > 3 * err.max()


def test_sampled_boundary_integral_agrees_with_the_exact_arcs():
    exact = sh.shape_derivatives(GRID, OBJ)
    sampled = sh.shape_derivatives(GRID, OBJ, method="sampled", n_samples=400_000)
    assert rel_per_param(sampled, exact).max() < 2e-4
    with pytest.raises(ValueError):
        sh.shape_derivatives(GRID, OBJ, method="nope")


def test_total_contrast_derivative_is_the_analytic_area_derivative():
    """The integral of chi is (eps_r - 1) pi (r1^2 + r2^2): d/dr = 2 pi R (eps_r - 1) per disk, 0 for g, cx, cy."""
    for obj in (OBJ, OBJ.replace(dr=0.06, cx=0.03, cy=-0.05, eps_r=3.5)):
        for method in ("arcs", "sampled"):
            d = sh.shape_derivatives(GRID, obj, method=method)
            got = d.sum(axis=(1, 2)) * GRID.h**2
            want = sh.total_contrast_gradient(obj)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-11, err_msg=method)
    r1, r2 = OBJ.left[2], OBJ.right[2]
    assert sh.total_contrast_gradient(OBJ)[1] == pytest.approx(2.0 * np.pi * r1 * (OBJ.eps_r - 1) + 2.0 * np.pi * r2 * (OBJ.eps_r - 1))
    # and the finite difference of the integral of the exact rendering
    h = 1e-6
    f = lambda o: sh.total_contrast(GRID, o.chi(GRID))
    fd_r = (f(OBJ.replace(r=OBJ.r + h)) - f(OBJ.replace(r=OBJ.r - h))) / (2 * h)
    assert fd_r == pytest.approx(sh.total_contrast_gradient(OBJ)[1], rel=1e-6)


def test_derivative_of_eps_r_is_the_coverage_and_shape_derivatives_scale_with_the_contrast():
    d1 = sh.shape_derivatives(GRID, OBJ)
    d3 = sh.shape_derivatives(GRID, OBJ.replace(eps_r=4.0))
    k = sh.PARAM_NAMES.index("eps_r")
    np.testing.assert_allclose(d1[k], OBJ.coverage(GRID), atol=0)
    for j, name in enumerate(sh.PARAM_NAMES):
        if name != "eps_r":
            np.testing.assert_allclose(d3[j], 3.0 * d1[j], rtol=1e-12, atol=1e-12)


def test_pixel_edge_crossings_are_handled_when_a_circle_is_tangent_to_a_pixel_edge():
    """A circle whose extreme point sits exactly on a pixel edge (xc - R = edge) must still integrate to the exact total, and the
    exact derivative there (the cap beyond the edge has area ~ delta^1.5, so its derivative is 0 at delta = 0) is what a central
    difference approaches only like sqrt(step): the difference quotient (A(+s) - A(-s)) / 2s = c s^0.5 is not small. The agreement
    with the exact rendering is therefore ~4 sqrt(step) at this configuration, against ~step^2 at a generic one."""
    grid = Grid(20, 1.0)  # h = 0.05, pixel edges at multiples of 0.05
    names = ("r", "cx", "cy", "g")
    obj = sh.TwoDisks(g=0.1, r=0.15)  # left disk spans x in [-0.4, -0.1]: both extremes on edges; y in [-0.15, 0.15]
    d = sh.shape_derivatives(grid, obj, names)
    np.testing.assert_allclose(d.sum(axis=(1, 2)) * grid.h**2, sh.total_contrast_gradient(obj, names), atol=1e-12)
    e4 = rel_per_param(d, sh.shape_derivatives_fd(grid, obj, names, step=1e-4))
    e6 = rel_per_param(d, sh.shape_derivatives_fd(grid, obj, names, step=1e-6))
    assert np.all(e6 < 6.0 * np.sqrt(1e-6)) and np.all(e4 < 6.0 * np.sqrt(1e-4))
    assert np.all(e4 / e6 > 8.0) and np.all(e4 / e6 < 12.0)  # 100 times the step, 10 times the error
    generic = obj.replace(cx=0.0123, cy=0.0071)  # no circle tangent to a pixel edge
    dg = sh.shape_derivatives(grid, generic, names)
    assert rel_per_param(dg, sh.shape_derivatives_fd(grid, generic, names, step=1e-6)).max() < 1e-8


def test_stretch_derivatives_are_the_exact_ellipticity_nuisances_with_the_gap_held_fixed():
    """d chi / d (s_x, s_y): the common radius is the sum of the two stretches (r = s_x = s_y changes both semi-axes and moves the centers apart by one, as
    in the 'r' column), the area changes by pi r per disk and unit stretch, and the finite differences of the sub-sampled ellipses agree to the quantization."""
    grid = Grid(32, 1.5)
    obj = sh.TwoDisks(g=0.1, r=0.2, eps_r=2.0)
    d = sh.stretch_derivatives(grid, obj)
    assert d.shape == (2, 32, 32)
    dr = sh.shape_derivatives(grid, obj, ("r",))[0]
    np.testing.assert_allclose(d[0] + d[1], dr, atol=1e-10 * np.abs(dr).max())
    for k in (0, 1):
        assert d[k].sum() * grid.h**2 == pytest.approx((obj.eps_r - 1.0) * 2.0 * np.pi * obj.r, rel=1e-12)
    for k, name in enumerate(("sx", "sy")):
        step = 2e-3
        fd = (sh.elliptic_chi(grid, 0.1, 0.2, 2.0, ss=96, **{name: step}) - sh.elliptic_chi(grid, 0.1, 0.2, 2.0, ss=96, **{name: -step})) / (2.0 * step)
        assert np.linalg.norm(fd - d[k]) / np.linalg.norm(d[k]) < 0.03, name  # measured 0.4 to 1 %: the 1 / ss^2 quantization of the reference rendering
    # the inner edges stay at +- g / 2: the x-derivative integrated along the line y = 0 across the gap is zero (the strip between the disks is untouched)
    X, Y = grid.centers()
    inside_gap = (np.abs(X) < 0.04) & (np.abs(Y) < 0.1)  # pixels (0.047 wide) wholly between the facing edges at +- 0.05
    assert inside_gap.sum() >= 8 and np.abs(d[0][inside_gap]).max() < 1e-12 and np.abs(d[1][inside_gap]).max() < 1e-12
    with pytest.raises(ValueError):  # a disk leaving the domain is refused, as for the other derivatives
        sh.stretch_derivatives(grid, sh.TwoDisks(g=0.1, r=0.2, eps_r=2.0, cx=0.4))
