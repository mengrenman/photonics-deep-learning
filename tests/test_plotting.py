"""Tests for bayes_scatter.plotting: orientation of maps, the z-map, and that every figure is written."""

import numpy as np
from matplotlib.figure import Figure

from bayes_scatter.geometry import Grid
from bayes_scatter.plotting import _show, plot_born_check, plot_maps, plot_traces, z_map


def test_z_map_values_and_nan_where_std_is_zero():
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean = truth + np.array([[0.5, -1.0], [0.0, 2.0]])
    std = np.array([[0.5, 0.5], [0.0, 4.0]])
    z = z_map(mean, truth, std)
    np.testing.assert_allclose(z[0, 0], 1.0)
    np.testing.assert_allclose(z[0, 1], -2.0)
    assert np.isnan(z[1, 0])
    np.testing.assert_allclose(z[1, 1], 0.5)


def test_maps_are_drawn_with_x_to_the_right_and_y_up():
    """Arrays are [i, j] = [x, y]; imshow shows a.T with origin lower, so the image row is y, the column x."""
    grid = Grid(4, 2.0, center=(1.0, -1.0))
    a = np.arange(16.0).reshape(4, 4)  # a[i, j] = 4 i + j
    fig = Figure()
    ax = fig.subplots()
    im = _show(fig, ax, grid, a, "t")
    shown = np.asarray(im.get_array())
    assert im.origin == "lower"
    assert shown[3, 0] == a[0, 3]  # row j = 3 (largest y), column i = 0 (smallest x)
    assert shown[0, 3] == a[3, 0]
    np.testing.assert_allclose(im.get_extent(), [0.0, 2.0, -2.0, 0.0])  # center +- side/2


def test_all_figures_are_written(tmp_path):
    grid = Grid(8, 2.0)
    rng = np.random.default_rng(0)
    truth = 1.0 + rng.random((8, 8))
    mean = truth + 0.1 * rng.standard_normal((8, 8))
    std = 0.1 + 0.05 * rng.random((8, 8))
    p1 = plot_maps(truth, truth * 1.02, mean, std, grid, tmp_path / "sub" / "maps.png", title="t")
    p2 = plot_maps(truth, None, mean, std, grid, tmp_path / "maps_nomap.png")
    p3 = plot_traces(rng.standard_normal((3, 200)), tmp_path / "traces.png",
                     extra={"pixel": rng.standard_normal((3, 50))})
    p4 = plot_traces(rng.standard_normal(100), tmp_path / "traces1.png")
    res = {k: {"mean": mean, "std": std, "z": rng.standard_normal(64)} for k in ("a", "b")}
    p5 = plot_born_check(truth, std, res, tmp_path / "born.png", title="t")
    for p in (p1, p2, p3, p4, p5):
        assert p.exists() and p.stat().st_size > 5000
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
