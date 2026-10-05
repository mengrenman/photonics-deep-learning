"""Contrast phantoms (docs/DESIGN.md section 2.6).

Every phantom is a real NumPy array of the contrast chi = eps_r - 1, shape (n, n), "ij" indexing.
Shapes are painted as chi <- (1 - c) chi + c chi_shape, where c is the fractional pixel area inside the
shape (``coverage``), so boundary pixels carry the volume-averaged contrast. Lengths and positions that
the book specifies for its 5 x 5 DOI are multiplied by ``grid.side / 5`` and measured from
``grid.center``; permittivities are not scaled.
"""

import numpy as np


def coverage(grid, inside_fn, ss=4):
    """Fractional area of each pixel inside a shape, by ss x ss sub-sampling.

    ``inside_fn(X, Y)`` returns a boolean array for coordinate arrays X, Y of any (equal) shape.
    Returns a float array (n, n) with values in {0, 1/ss^2, ..., 1}.
    """
    X, Y = grid.centers()
    offs = ((np.arange(ss) + 0.5) / ss - 0.5) * grid.h
    acc = np.zeros(X.shape)
    for dx in offs:
        for dy in offs:
            acc += inside_fn(X + dx, Y + dy)
    return acc / ss**2


def paint(chi, cov, chi_shape):
    """Return chi <- (1 - cov) * chi + cov * chi_shape (new array)."""
    return (1.0 - cov) * chi + cov * chi_shape


def _disk(cx, cy, radius):
    return lambda X, Y: (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2


def _annulus(cx, cy, r_in, r_out):
    def inside(X, Y):
        rho2 = (X - cx) ** 2 + (Y - cy) ** 2
        return (rho2 >= r_in**2) & (rho2 <= r_out**2)

    return inside


def _rectangle(cx, cy, lx, ly):
    return lambda X, Y: (np.abs(X - cx) <= 0.5 * lx) & (np.abs(Y - cy) <= 0.5 * ly)


def cylinder(grid, radius, eps_r, center=(0.0, 0.0)):
    """Homogeneous circular cylinder of relative permittivity ``eps_r`` (contrast eps_r - 1)."""
    cov = coverage(grid, _disk(center[0], center[1], radius))
    return paint(np.zeros((grid.n, grid.n)), cov, eps_r - 1.0)


def austria_fixed(grid, eps_r):
    """A fixed test object of our own choosing (not from the book), all with permittivity ``eps_r``:
    two disks of radius 0.5 at (+-0.9, 1.0) and a ring with radii 0.9 and 1.3 centered at (0, -0.8),
    scaled by grid.side / 5 and offset by grid.center."""
    s = grid.side / 5.0
    cx, cy = grid.center
    chi = np.zeros((grid.n, grid.n))
    shapes = [
        _disk(cx - 0.9 * s, cy + 1.0 * s, 0.5 * s),
        _disk(cx + 0.9 * s, cy + 1.0 * s, 0.5 * s),
        _annulus(cx, cy - 0.8 * s, 0.9 * s, 1.3 * s),
    ]
    for inside in shapes:
        chi = paint(chi, coverage(grid, inside), eps_r - 1.0)
    return chi


def ch15_austria_variant(rng, grid):
    """The book's random training model (Table 15.2, Fig. 15.8), as a contrast map.

    Two disks (radius in [0.1, 1], center in [-2, 2]^2), a ring (inner radius r3 in [0.1, 1], outer radius
    r4 in [0.2, 2] conditioned on r4 > r3, center in [-2, 2]^2), a rectangle (sides l1, l2 in [0.1, 2]) and
    a square (side l3 in [0.1, 2]) with centers in [-2, 2]^2, each with eps_r ~ U[1, 8]. Painted in that
    order, later shapes overwriting earlier ones. The rectangle and square are axis-aligned. Lengths and
    positions are scaled by grid.side / 5.

    ``rng`` is a ``numpy.random.Generator``. Draw order per shape: size parameters, then center (x, y),
    then eps_r; so a given seed always gives the same phantom. r4 is drawn from U[max(0.2, r3), 2], which
    is the uniform law on [0.2, 2] conditioned on r4 > r3.
    """
    s = grid.side / 5.0
    cx0, cy0 = grid.center
    chi = np.zeros((grid.n, grid.n))

    def center_and_eps():
        x, y = rng.uniform(-2.0, 2.0, size=2)
        return cx0 + s * x, cy0 + s * y, rng.uniform(1.0, 8.0)

    for _ in range(2):
        radius = rng.uniform(0.1, 1.0)
        cx, cy, eps = center_and_eps()
        chi = paint(chi, coverage(grid, _disk(cx, cy, s * radius)), eps - 1.0)

    r3 = rng.uniform(0.1, 1.0)
    r4 = rng.uniform(max(0.2, r3), 2.0)
    cx, cy, eps = center_and_eps()
    chi = paint(chi, coverage(grid, _annulus(cx, cy, s * r3, s * r4)), eps - 1.0)

    l1, l2 = rng.uniform(0.1, 2.0, size=2)
    cx, cy, eps = center_and_eps()
    chi = paint(chi, coverage(grid, _rectangle(cx, cy, s * l1, s * l2)), eps - 1.0)

    l3 = rng.uniform(0.1, 2.0)
    cx, cy, eps = center_and_eps()
    chi = paint(chi, coverage(grid, _rectangle(cx, cy, s * l3, s * l3)), eps - 1.0)
    return chi
