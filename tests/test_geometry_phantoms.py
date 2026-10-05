"""Tests for geometry.py and phantoms.py (docs/DESIGN.md 1, 2.5, 2.6)."""

import numpy as np
import pytest

import bayes_scatter.phantoms as phantoms
from bayes_scatter.geometry import Antennas, Grid, Setup, ch15_setup, ring
from bayes_scatter.phantoms import (austria_fixed, ch15_austria_variant, coverage, cylinder,
                                    paint)


# --------------------------------------------------------------------------- geometry


def test_grid_geometry_and_ij_indexing():
    g = Grid(n=4, side=2.0, center=(1.0, -3.0))
    assert g.h == 0.5
    assert np.isclose(g.a, 0.5 / np.sqrt(np.pi))
    assert np.isclose(np.pi * g.a**2, g.h**2)  # equal-area disk
    X, Y = g.centers()
    assert X.shape == Y.shape == (4, 4)
    # x_i = cx - side/2 + (i + 1/2) h varies along axis 0, y_j along axis 1
    np.testing.assert_allclose(X[:, 0], [0.25, 0.75, 1.25, 1.75])
    np.testing.assert_allclose(Y[0, :], [-3.75, -3.25, -2.75, -2.25])
    assert np.all(X == X[:, :1]) and np.all(Y == Y[:1, :])
    # flattening is C order: k = i n + j
    assert X.ravel()[1 * 4 + 2] == X[1, 2] and Y.ravel()[1 * 4 + 2] == Y[1, 2]


def test_grid_validation_and_frozen():
    with pytest.raises(ValueError):
        Grid(0, 1.0)
    with pytest.raises(ValueError):
        Grid(4, -1.0)
    g = Grid(4, 1.0)
    with pytest.raises(Exception):
        g.n = 5  # frozen


def test_ring_angles_and_offset():
    p = ring(8, 2.0)
    assert p.shape == (8, 2)
    np.testing.assert_allclose(p[0], [2.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 2.0)
    np.testing.assert_allclose(p[2], [0.0, 2.0], atol=1e-15)  # 90 degrees
    q = ring(8, 2.0, offset=np.pi / 2)
    np.testing.assert_allclose(q[0], [0.0, 2.0], atol=1e-15)
    c = ring(3, 1.0, center=(5.0, 6.0))
    np.testing.assert_allclose(c.mean(axis=0), [5.0, 6.0], atol=1e-15)


def test_antennas_validation():
    with pytest.raises(ValueError):
        Antennas(np.zeros((3, 3)), np.zeros((2, 2)))
    ant = Antennas([[1, 0], [0, 1]], np.zeros((3, 2)))
    assert ant.tx.dtype == np.float64 and ant.n_tx == 2 and ant.n_rx == 3


def test_ch15_setup_matches_design():
    s = ch15_setup()
    assert s.grid.n == 96 and s.grid.side == 5.0 and s.grid.center == (0.0, 0.0)
    assert np.isclose(s.k0, 2 * np.pi) and s.wavelength == 1.0
    tx, rx = s.antennas.tx, s.antennas.rx
    assert tx.shape == (40, 2) and rx.shape == (60, 2)
    np.testing.assert_allclose(np.linalg.norm(tx, axis=1), 4.0)
    np.testing.assert_allclose(np.linalg.norm(rx, axis=1), 4.5)
    np.testing.assert_allclose(tx[0], [4.0, 0.0], atol=1e-14)  # both start at angle 0
    np.testing.assert_allclose(rx[0], [4.5, 0.0], atol=1e-14)
    ang_tx = np.degrees(np.arctan2(tx[1, 1], tx[1, 0]))
    ang_rx = np.degrees(np.arctan2(rx[1, 1], rx[1, 0]))
    assert np.isclose(ang_tx, 9.0) and np.isclose(ang_rx, 6.0)  # every 9 and 6 degrees
    r = ch15_setup(n=32, side=2.0, n_tx=7, n_rx=9)
    assert r.grid.n == 32 and r.antennas.n_tx == 7 and r.antennas.n_rx == 9


# --------------------------------------------------------------------------- phantoms


def test_coverage_exact_for_pixel_aligned_shape_and_bounds():
    g = Grid(8, 8.0)  # h = 1, pixel edges at integers
    cov = coverage(g, lambda X, Y: (X > -2) & (X < 2) & (Y > 0) & (Y < 3), ss=4)
    expected = np.zeros((8, 8))
    expected[2:6, 4:7] = 1.0  # x in (-2, 2) -> i = 2..5; y in (0, 3) -> j = 4..6
    np.testing.assert_array_equal(cov, expected)
    # half-covered pixels get fractional coverage
    cov2 = coverage(g, lambda X, Y: X < -3.5, ss=4)
    assert np.allclose(cov2[0, :], 0.5) and np.all(cov2[1:, :] == 0)


def test_coverage_area_of_disk():
    g = Grid(48, 4.0)
    cov = coverage(g, lambda X, Y: X**2 + Y**2 <= 1.0)
    assert cov.min() >= 0 and cov.max() <= 1
    assert abs(cov.sum() * g.h**2 - np.pi) < 3e-3


def test_paint_blends_by_coverage():
    chi = np.full((2, 2), 3.0)
    cov = np.array([[0.0, 0.25], [0.5, 1.0]])
    out = paint(chi, cov, 7.0)
    np.testing.assert_allclose(out, [[3.0, 4.0], [5.0, 7.0]])


def test_cylinder_values_area_centroid_and_dtype():
    g = Grid(96, 2.0)
    chi = cylinder(g, 0.6, 3.0, center=(0.4, -0.2))
    assert chi.shape == (96, 96) and chi.dtype == np.float64
    assert chi.max() == 2.0 and chi.min() == 0.0  # contrast = eps_r - 1
    area = (chi / 2.0).sum() * g.h**2
    assert abs(area - np.pi * 0.6**2) / (np.pi * 0.6**2) < 3e-3
    X, Y = g.centers()
    w = chi / chi.sum()
    assert abs((w * X).sum() - 0.4) < 2e-3 and abs((w * Y).sum() + 0.2) < 2e-3  # axis 0 is x
    # boundary pixels carry fractional (volume-averaged) contrast
    assert np.any((chi > 0) & (chi < 2.0))


def test_cylinder_complex_permittivity_gives_complex_contrast():
    chi = cylinder(Grid(32, 2.0), 0.6, 2.0 + 0.5j)
    assert np.iscomplexobj(chi) and np.isclose(chi.max().real, 1.0) and np.isclose(chi.max().imag, 0.5)


def test_austria_fixed_structure():
    g = Grid(96, 5.0)
    for eps in (2.0, 4.0):
        chi = austria_fixed(g, eps)
        assert chi.shape == (96, 96) and chi.dtype == np.float64
        assert chi.min() == 0.0 and np.isclose(chi.max(), eps - 1.0)
        # area: two disks of radius 0.5 plus a ring with radii 0.9, 1.3
        area_exact = 2 * np.pi * 0.5**2 + np.pi * (1.3**2 - 0.9**2)
        area = (chi / (eps - 1)).sum() * g.h**2
        assert abs(area - area_exact) / area_exact < 5e-3
    # symmetric under x -> -x (disks at +-0.9, ring centered on x = 0)
    np.testing.assert_allclose(chi, chi[::-1, :], atol=1e-12)
    # the object lies where DESIGN says: disk centers at (+-0.9, 1.0), ring center (0, -0.8) is empty
    X, Y = g.centers()
    i = np.argmin(np.abs(X[:, 0] - 0.9)); j = np.argmin(np.abs(Y[0, :] - 1.0))
    assert chi[i, j] > 0
    i0 = np.argmin(np.abs(X[:, 0])); j0 = np.argmin(np.abs(Y[0, :] + 0.8))
    assert chi[i0, j0] == 0 and chi[i0, np.argmin(np.abs(Y[0, :] + 0.8 - 1.1))] > 0


def test_austria_fixed_scales_with_side():
    a = austria_fixed(Grid(64, 5.0), 2.0)
    b = austria_fixed(Grid(64, 2.0), 2.0)  # same shapes scaled by 2/5
    assert np.mean(a != b) < 2e-3
    # positions are measured from grid.center, so shifting the grid carries the object with it
    c = austria_fixed(Grid(64, 5.0, center=(0.5, -0.5)), 2.0)
    assert np.mean(a != c) < 2e-3 and c.max() == 1.0


def test_ch15_variant_deterministic_and_in_range():
    g = Grid(48, 5.0)
    a = ch15_austria_variant(np.random.default_rng(3), g)
    b = ch15_austria_variant(np.random.default_rng(3), g)
    c = ch15_austria_variant(np.random.default_rng(4), g)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.dtype == np.float64 and a.shape == (48, 48)
    for seed in range(20):
        chi = ch15_austria_variant(np.random.default_rng(seed), g)
        assert chi.min() >= 0.0 and chi.max() <= 7.0 + 1e-12  # eps_r in [1, 8]
        assert chi.max() > 0


def test_ch15_variant_parameter_ranges(monkeypatch):
    """Record every shape the generator builds and check Table 15.2 ranges, scaled by side/5."""
    calls = {"disk": [], "annulus": [], "rect": []}
    orig = (phantoms._disk, phantoms._annulus, phantoms._rectangle)

    def rec(name, fn):
        def wrapper(*args):
            calls[name].append(args)
            return fn(*args)
        return wrapper

    monkeypatch.setattr(phantoms, "_disk", rec("disk", orig[0]))
    monkeypatch.setattr(phantoms, "_annulus", rec("annulus", orig[1]))
    monkeypatch.setattr(phantoms, "_rectangle", rec("rect", orig[2]))
    side = 3.0
    s = side / 5.0
    g = Grid(24, side, center=(0.2, -0.1))
    n_draws = 300
    rng = np.random.default_rng(0)
    for _ in range(n_draws):
        ch15_austria_variant(rng, g)
    assert len(calls["disk"]) == 2 * n_draws and len(calls["annulus"]) == n_draws
    assert len(calls["rect"]) == 2 * n_draws  # the rectangle and the square
    cx0, cy0 = g.center

    def check_center(cx, cy):
        assert abs(cx - cx0) <= 2 * s + 1e-12 and abs(cy - cy0) <= 2 * s + 1e-12

    radii = []
    for cx, cy, r in calls["disk"]:
        check_center(cx, cy)
        assert 0.1 * s - 1e-12 <= r <= 1.0 * s + 1e-12
        radii.append(r / s)
    for cx, cy, r3, r4 in calls["annulus"]:
        check_center(cx, cy)
        assert 0.1 * s - 1e-12 <= r3 <= 1.0 * s + 1e-12
        assert 0.2 * s - 1e-12 <= r4 <= 2.0 * s + 1e-12
        assert r4 > r3
    for cx, cy, lx, ly in calls["rect"]:
        check_center(cx, cy)
        assert 0.1 * s - 1e-12 <= lx <= 2.0 * s + 1e-12 and 0.1 * s - 1e-12 <= ly <= 2.0 * s + 1e-12
    squares = calls["rect"][1::2]
    assert all(lx == ly for _, _, lx, ly in squares)
    rects = calls["rect"][0::2]
    assert np.mean([lx != ly for _, _, lx, ly in rects]) == 1.0
    # uniform draws: the mean disk radius should be near 0.55 (in units of side/5)
    assert abs(np.mean(radii) - 0.55) < 0.03


def test_ch15_variant_later_shapes_overwrite():
    """With every shape painted at full coverage, the last shape painted wins where shapes overlap."""
    g = Grid(32, 5.0)
    seen = []
    orig_paint = phantoms.paint

    def spy(chi, cov, val):
        seen.append((cov.copy(), val))
        return orig_paint(chi, cov, val)

    phantoms.paint = spy
    try:
        chi = ch15_austria_variant(np.random.default_rng(11), g)
    finally:
        phantoms.paint = orig_paint
    assert len(seen) == 5
    cov_last, val_last = seen[-1]
    inner = cov_last == 1.0
    assert inner.any()
    np.testing.assert_allclose(chi[inner], val_last)
