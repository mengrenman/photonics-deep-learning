"""Exact pixel coverage of ellipses and exact boundary shape derivatives, in JAX (docs/DESIGN.md 14.2).

Ellipse. An ellipse is e = (xc, yc, a, b, psi): center (xc, yc), semi-axes a (along the direction psi) and b (perpendicular to it) and the rotation
angle psi (radians, counterclockwise from the x axis). Its boundary is, for t in [0, 2 pi) (a counterclockwise parametrization),

    x(t) = xc + A cos t - B sin t,    y(t) = yc + C cos t + D sin t,    A = a cos psi, B = b sin psi, C = a sin psi, D = b cos psi.

A circle of radius R is a = b = R (psi is then irrelevant). Arrays are indexed [i, j] = [x, y] with pixel (i, j) = [xe_i, xe_(i+1)] x [ye_j, ye_(j+1)] (the
conventions of ``geometry.Grid``); the DOI truncates a shape that leaves it (the coverage is the area inside the DOI, so every function below is defined
for any ellipse, and the derivative of the truncated shape is the one of the part inside).

Exact area (Green's theorem). The region Omega = pixel intersected with the ellipse is convex, and for the counterclockwise boundary of Omega

    area(Omega) = (contour integral of (x - xc) dy) = integral over the ellipse arcs inside the pixel of (x - xc) dy
                  + (x_right - xc) L(x_right) - (x_left - xc) L(x_left),

with L(x_e) the length of the part of the vertical pixel edge at x = x_e (between the pixel's y limits) that lies inside the ellipse, closed form from the chord of
the ellipse at abscissa x_e. The arcs are cut at their crossings with the pixel-edge lines (closed-form angles), each arc lies in a single pixel (found from its
midpoint), and the integral of (x - xc) dy over an arc is a trigonometric polynomial integral in closed form. Subtracting xc keeps every term of the size of the
shape, so the round-off of a pixel area is about 1e-14 of a pixel area (tested against ``shapes.disk_pixel_areas`` and against dense sub-sampling).

Exact shape derivatives (Reynolds transport). The derivative of the area of the ellipse inside a pixel with respect to a parameter p of the boundary is the
integral over the boundary arc in the pixel of the normal speed, v . n ds. For the counterclockwise parametrization n ds = (y'(t), -x'(t)) dt, so

    xc:  y' dt            ->  [C cos t + D sin t]                 (integral over the arc [t_lo, t_hi]; [f] = f(t_hi) - f(t_lo))
    yc:  -x' dt           ->  -[A cos t - B sin t]
    a:   b cos^2 t dt     ->  b [t / 2 + sin(2 t) / 4]
    b:   a sin^2 t dt     ->  a [t / 2 - sin(2 t) / 4]
    psi: -(1/2) d(rho^2)  ->  -(1/2) [a^2 cos^2 t + b^2 sin^2 t]

(rho^2 = (x - xc)^2 + (y - yc)^2; a rotation moves a boundary point by the 90-degree rotation of its radius vector). These are exact up to round-off; the
derivative of the total area is pi b (a), pi a (b), 0 (xc, yc, psi), which the tests check. ``EllipseRenderer.coverage`` is a ``jax.custom_jvp`` function
whose tangent is this exact derivative, so ``jax.grad``, ``jax.jvp`` and ``jax.linearize`` through a coverage use the boundary integrals (and the
reverse mode transposes the linear tangent rule); nothing is differentiated through the sort and the arc bookkeeping. Fixed sizes: an ellipse crosses each of
the n + 1 vertical and n + 1 horizontal pixel-edge lines at most twice, so 4 (n + 1) candidate angles are sorted (the missing crossings are placed at angle 0, which
yields arcs of length zero), and everything jits, vmaps and runs inside ``lax.scan``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import shapes as sh

N_PARAMS = 5  # xc, yc, a, b, psi
PARAM_NAMES = ("xc", "yc", "a", "b", "psi")


# ----------------------------------------------------------------------------------------------
# The renderer
# ----------------------------------------------------------------------------------------------


def _arc_integrals(A, B, C, D, a, b, lo, hi):
    """Integrals over the arcs [lo, hi] (arrays of angles) of (x - xc) dy and of the normal speeds of xc, yc, a, b, psi: array (M, 6)."""

    def anti_area(t):
        s2 = jnp.sin(2.0 * t)
        return -0.5 * (A * C + B * D) * jnp.sin(t) ** 2 + 0.5 * (A * D + B * C) * t + 0.25 * (A * D - B * C) * s2

    def d(f):
        return f(hi) - f(lo)

    i_area = d(anti_area)
    i_xc = d(lambda t: C * jnp.cos(t) + D * jnp.sin(t))
    i_yc = -d(lambda t: A * jnp.cos(t) - B * jnp.sin(t))
    i_a = b * d(lambda t: 0.5 * t + 0.25 * jnp.sin(2.0 * t))
    i_b = a * d(lambda t: 0.5 * t - 0.25 * jnp.sin(2.0 * t))
    i_psi = -0.5 * d(lambda t: a * a * jnp.cos(t) ** 2 + b * b * jnp.sin(t) ** 2)
    return jnp.stack([i_area, i_xc, i_yc, i_a, i_b, i_psi], axis=-1)


def _crossing_angles(xc, yc, A, B, C, D, xe, ye):
    """Sorted angles in [0, 2 pi] of the crossings of the ellipse with every pixel-edge line, with 0 and 2 pi included; fixed size 4 (n + 1) + 2.

    A crossing with x = xe solves A cos t - B sin t = xe - xc, i.e. r_x cos(t + phi_x) = dx with r_x = hypot(A, B), phi_x = atan2(B, A); with y = ye:
    C cos t + D sin t = dy, r_y cos(t - phi_y) = dy. A line that misses the ellipse gets angle 0 (a zero-length arc).
    """
    rx, phx = jnp.hypot(A, B), jnp.arctan2(B, A)
    ry, phy = jnp.hypot(C, D), jnp.arctan2(D, C)
    ux, uy = (xe - xc) / rx, (ye - yc) / ry
    ax, ay = jnp.arccos(jnp.clip(ux, -1.0, 1.0)), jnp.arccos(jnp.clip(uy, -1.0, 1.0))
    okx, oky = jnp.abs(ux) <= 1.0, jnp.abs(uy) <= 1.0
    cand = jnp.concatenate([jnp.where(okx, -phx + ax, 0.0), jnp.where(okx, -phx - ax, 0.0),
                            jnp.where(oky, phy + ay, 0.0), jnp.where(oky, phy - ay, 0.0)])
    cand = jnp.mod(cand, 2.0 * jnp.pi)
    return jnp.sort(jnp.concatenate([jnp.zeros(1), cand, jnp.full(1, 2.0 * jnp.pi)]))


def _edge_chord_lengths(xc, yc, a, b, psi, rx, xe, ye):
    """L[i, j] = length of the part of the vertical line x = xe[i] between ye[j] and ye[j + 1] that lies inside the ellipse: (n + 1, n).

    The chord at abscissa x has half-width sqrt(disc) / m22 with disc = m22 - det(M) dx^2 = det(M) rx^2 (1 - ux^2), ux = dx / rx (rx = hypot(A, B) is the half-width of the
    ellipse in x); the same ux decides the crossings of ``_crossing_angles``, so a line that is nearly tangent is treated alike by the arcs and by the chord (their small
    errors then cancel).
    """
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    m22 = sp**2 / a**2 + cp**2 / b**2
    m12 = sp * cp * (1.0 / a**2 - 1.0 / b**2)
    dx = xe - xc
    ux = dx / rx
    sq = rx * jnp.sqrt(jnp.maximum(1.0 - ux**2, 0.0)) / (a * b)  # sqrt(disc)
    ylo = yc + (-m12 * dx - sq) / m22
    yhi = yc + (-m12 * dx + sq) / m22
    lo = jnp.maximum(ye[None, :-1], ylo[:, None])
    hi = jnp.minimum(ye[None, 1:], yhi[:, None])
    return jnp.where((jnp.abs(ux) < 1.0)[:, None], jnp.maximum(hi - lo, 0.0), 0.0)


def _render(e, xe, ye, n, h):
    """(area (n, n), d area (n, n, 5)) of one ellipse ``e`` = (xc, yc, a, b, psi) on the grid with edges ``xe``, ``ye`` (the module docstring)."""
    xc, yc, a, b, psi = e[0], e[1], e[2], e[3], e[4]
    cp, sp = jnp.cos(psi), jnp.sin(psi)
    A, B, C, D = a * cp, b * sp, a * sp, b * cp
    t = _crossing_angles(xc, yc, A, B, C, D, xe, ye)
    lo, hi = t[:-1], t[1:]
    mid = 0.5 * (lo + hi)
    xm = xc + A * jnp.cos(mid) - B * jnp.sin(mid)
    ym = yc + C * jnp.cos(mid) + D * jnp.sin(mid)
    fi = jnp.floor((xm - xe[0]) / h).astype(jnp.int32)
    fj = jnp.floor((ym - ye[0]) / h).astype(jnp.int32)
    inside = (fi >= 0) & (fi < n) & (fj >= 0) & (fj < n)
    vals = _arc_integrals(A, B, C, D, a, b, lo, hi) * inside[:, None]
    arcs = jnp.zeros((n, n, 6)).at[jnp.clip(fi, 0, n - 1), jnp.clip(fj, 0, n - 1)].add(vals)
    chord = _edge_chord_lengths(xc, yc, a, b, psi, jnp.hypot(A, B), xe, ye)
    area = (xe[1:, None] - xc) * chord[1:] - (xe[:-1, None] - xc) * chord[:-1] + arcs[..., 0]
    return jnp.clip(area, 0.0, h * h), arcs[..., 1:]


class EllipseRenderer:
    """Exact coverage of ellipses on one ``geometry.Grid`` and its exact derivatives (module docstring).

    Methods (all jitted, float64, ``e`` = (xc, yc, a, b, psi) or a stack (E, 5)):

    * ``coverage(e)`` : fraction of every pixel inside the ellipse, (n, n); differentiable (``jax.custom_jvp`` with the exact boundary derivative);
    * ``coverage_many(E)`` : (E, n, n) for a stack of ellipses (``jax.vmap`` of ``coverage``);
    * ``coverage_and_derivatives(e)`` : ``(coverage (n, n), d coverage / d (xc, yc, a, b, psi) (5, n, n))``, the derivative per unit of the parameter
      (d area_pix / d p / h^2).
    """

    def __init__(self, grid):
        self.grid = grid
        n, h = grid.n, grid.h
        xe, ye = sh.pixel_edges(grid)
        xe, ye = jnp.asarray(xe), jnp.asarray(ye)

        @jax.custom_jvp
        def coverage(e):
            return _render(e, xe, ye, n, h)[0] / (h * h)

        @coverage.defjvp
        def coverage_jvp(primals, tangents):
            (e,), (de,) = primals, tangents
            area, darea = _render(e, xe, ye, n, h)
            return area / (h * h), jnp.einsum("ijm,m->ij", darea, de) / (h * h)

        def with_derivatives(e):
            area, darea = _render(e, xe, ye, n, h)
            return area / (h * h), jnp.moveaxis(darea, -1, 0) / (h * h)

        self._coverage = coverage
        self.coverage = jax.jit(coverage)
        self.coverage_many = jax.jit(jax.vmap(coverage))
        self.coverage_and_derivatives = jax.jit(with_derivatives)

    def chi(self, ellipses, eps):
        """Contrast of disjoint ellipses with relative permittivities ``eps``: sum_k (eps_k - 1) coverage_k, (n, n). JAX arrays or NumPy: (E, 5), (E,)."""
        return jnp.einsum("k,kij->ij", jnp.asarray(eps) - 1.0, self.coverage_many(jnp.asarray(ellipses)))


def ellipse_coverage(grid, e):
    """NumPy (n, n) coverage of one ellipse ``e`` = (xc, yc, a, b, psi) on ``grid`` (``EllipseRenderer``; builds the renderer, so reuse one for many calls)."""
    return np.asarray(EllipseRenderer(grid).coverage(jnp.asarray(e, dtype=jnp.float64)))


# ----------------------------------------------------------------------------------------------
# Rectangles and the overlap of a disk with a rectangle (truths with overlapping parts)
# ----------------------------------------------------------------------------------------------


def rect_pixel_areas(grid, x0, x1, y0, y1):
    """Exact area of the axis-aligned rectangle [x0, x1] x [y0, y1] inside every pixel, (n, n) NumPy ("ij" indexing)."""
    xe, ye = sh.pixel_edges(grid)
    wx = np.clip(np.minimum(xe[1:], x1) - np.maximum(xe[:-1], x0), 0.0, None)
    wy = np.clip(np.minimum(ye[1:], y1) - np.maximum(ye[:-1], y0), 0.0, None)
    return wx[:, None] * wy[None, :]


def disk_rect_pixel_areas(grid, xc, yc, radius, x0, x1, y0, y1):
    """Exact area of (disk intersected with the rectangle [x0, x1] x [y0, y1]) inside every pixel, (n, n) NumPy.

    Every pixel is clipped to the rectangle and the area of the disk in the clipped box is the second difference of the quadrant area at its corners
    (``shapes._quadrant_area``, as in ``shapes.disk_pixel_areas``).
    """
    xe, ye = sh.pixel_edges(grid)
    xa, xb = np.clip(xe[:-1], x0, x1), np.clip(xe[1:], x0, x1)
    ya, yb = np.clip(ye[:-1], y0, y1), np.clip(ye[1:], y0, y1)

    def quad(u, v):
        return sh._quadrant_area((u - xc)[:, None], (v - yc)[None, :], radius)

    area = quad(xb, yb) - quad(xa, yb) - quad(xb, ya) + quad(xa, ya)
    return np.clip(area, 0.0, grid.h**2)


# ----------------------------------------------------------------------------------------------
# Scenes (truths)
# ----------------------------------------------------------------------------------------------


def _eps_array(eps, count):
    eps = np.atleast_1d(np.asarray(eps, dtype=np.float64))
    return np.full(count, eps[0]) if eps.size == 1 else eps


class EllipseScene:
    """Disjoint ellipses with their own relative permittivities, rendered exactly: ``chi = sum_k (eps_k - 1) coverage_k``.

    ``ellipses``: (E, 5) rows (xc, yc, a, b, psi); ``eps``: scalar or (E,). The ellipses must not overlap (checked only by the callers' geometry; overlapping
    shapes would be summed). ``chi(grid)`` / ``coverage(grid)`` / ``theta(grid)`` are NumPy (n, n); ``renderer`` reuse: pass the grid each time (renderers are cached per grid).
    """

    _renderers = {}

    def __init__(self, ellipses, eps, name="scene", true_gap=None, description=""):
        self.ellipses = np.atleast_2d(np.asarray(ellipses, dtype=np.float64))
        self.eps = _eps_array(eps, self.ellipses.shape[0])
        self.name, self.true_gap, self.description = name, true_gap, description

    @staticmethod
    def renderer(grid):
        key = (grid.n, grid.side, grid.center)
        if key not in EllipseScene._renderers:
            EllipseScene._renderers[key] = EllipseRenderer(grid)
        return EllipseScene._renderers[key]

    def coverage_each(self, grid):
        """(E, n, n) coverage of every ellipse."""
        return np.asarray(self.renderer(grid).coverage_many(jnp.asarray(self.ellipses)))

    def coverage(self, grid):
        return self.coverage_each(grid).sum(axis=0)

    def chi(self, grid, ss=None):
        """Contrast (n, n); ``ss`` (an integer) uses the ss x ss sub-sampled coverage of ``phantoms.coverage`` instead (the finite-difference style reference)."""
        if ss is None:
            return np.einsum("k,kij->ij", self.eps - 1.0, self.coverage_each(grid))
        from . import phantoms

        out = np.zeros((grid.n, grid.n))
        for (xc, yc, a, b, psi), eps in zip(self.ellipses, self.eps):
            c, s = np.cos(psi), np.sin(psi)
            out += (eps - 1.0) * phantoms.coverage(grid, lambda X, Y, xc=xc, yc=yc, a=a, b=b, c=c, s=s:
                                                   (((X - xc) * c + (Y - yc) * s) / a) ** 2 + ((-(X - xc) * s + (Y - yc) * c) / b) ** 2 <= 1.0, ss=ss)
        return out

    def theta(self, grid, ss=None):
        return np.log1p(self.chi(grid, ss))

    def total_contrast_exact(self):
        """The integral of chi over the plane (when every ellipse is inside the DOI): sum_k (eps_k - 1) pi a_k b_k."""
        return float(np.sum((self.eps - 1.0) * np.pi * self.ellipses[:, 2] * self.ellipses[:, 3]))

    def describe(self):
        return {"name": self.name, "kind": type(self).__name__, "ellipses": self.ellipses.tolist(), "eps": self.eps.tolist(), "true_gap": self.true_gap,
                "description": self.description}


class BridgedDisks:
    """Two touching disks joined by a bridge: the DESIGN 14.3 truth T4 (the true gap is 0).

    Disks of radius ``r`` and one relative permittivity ``eps_r`` centered at (-+ r, 0) (they touch at the origin) and a rectangle of width ``width`` (along x)
    and height 2 r (the full disk height) centered at the origin, of the same permittivity. The contrast is that of the UNION (the overlap of the bridge with the disks is
    counted once): area(union) = area(D1) + area(D2) + area(rectangle) - area(D1 and rectangle) - area(D2 and rectangle) (the disks meet in one point), every term
    exact per pixel (``disk_rect_pixel_areas``).
    """

    def __init__(self, r=0.2, width=0.05, eps_r=2.0, cx=0.0, cy=0.0):
        self.r, self.width, self.eps_r, self.cx, self.cy = float(r), float(width), float(eps_r), float(cx), float(cy)
        self.name, self.true_gap = "bridged_disks", 0.0
        self.description = f"two touching disks of radius {r} joined by a rectangle {width} wide and {2 * r} high"

    @property
    def left(self):
        return (self.cx - self.r, self.cy, self.r)

    @property
    def right(self):
        return (self.cx + self.r, self.cy, self.r)

    def rect(self):
        return (self.cx - 0.5 * self.width, self.cx + 0.5 * self.width, self.cy - self.r, self.cy + self.r)

    def coverage(self, grid):
        x0, x1, y0, y1 = self.rect()
        cov = rect_pixel_areas(grid, x0, x1, y0, y1)
        for xc, yc, rad in (self.left, self.right):
            cov = cov + sh.disk_pixel_areas(grid, xc, yc, rad) - disk_rect_pixel_areas(grid, xc, yc, rad, x0, x1, y0, y1)
        return np.clip(cov / grid.h**2, 0.0, 1.0)

    def chi(self, grid, ss=None):
        if ss is None:
            return (self.eps_r - 1.0) * self.coverage(grid)
        from . import phantoms

        x0, x1, y0, y1 = self.rect()
        (xl, yl, r), (xr, yr, _) = self.left, self.right
        cov = phantoms.coverage(grid, lambda X, Y: ((X - xl) ** 2 + (Y - yl) ** 2 <= r**2) | ((X - xr) ** 2 + (Y - yr) ** 2 <= r**2)
                                | ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)), ss=ss)
        return (self.eps_r - 1.0) * cov

    def theta(self, grid, ss=None):
        return np.log1p(self.chi(grid, ss))

    def union_area_exact(self):
        """Area of the union by direct integration (a check of the pixel sums): disks 2 pi r^2 plus the rectangle minus the two overlaps, where the overlap of the
        right disk with the strip 0 <= x <= w/2 (and by symmetry the left one) is the integral of 2 sqrt(r^2 - (x - r)^2) clipped to the rectangle height."""
        w, r = self.width, self.r
        xs = np.linspace(0.0, 0.5 * w, 200001)
        chord = np.minimum(2.0 * np.sqrt(np.maximum(r * r - (xs - r) ** 2, 0.0)), 2.0 * r)
        overlap_half = float(np.sum(0.5 * (chord[1:] + chord[:-1]) * np.diff(xs)))  # disk part with 0 <= x <= w/2, per disk (two disks: both halves)
        return 2.0 * np.pi * r * r + w * 2.0 * r - 2.0 * overlap_half

    def describe(self):
        return {"name": self.name, "kind": "BridgedDisks", "r": self.r, "width": self.width, "eps_r": self.eps_r, "cx": self.cx, "cy": self.cy,
                "true_gap": 0.0, "description": self.description}


# ----------------------------------------------------------------------------------------------
# The DESIGN 14.3 truths
# ----------------------------------------------------------------------------------------------

TRUTH_NAMES = ("T1", "T2", "T3", "T4", "T5")


def two_disks_ellipses(g, r1, r2, cx=0.0, cy=0.0):
    """Ellipse rows (2, 5) of two disks of radii r1 (left) and r2 (right) with edge-to-edge gap g and pair center (cx, cy): the geometry of
    ``shapes.TwoDisks`` (centers cx -+ (rbar + g / 2), rbar = (r1 + r2) / 2) for unequal radii."""
    rbar = 0.5 * (r1 + r2)
    return np.array([[cx - (rbar + 0.5 * g), cy, r1, r1, 0.0], [cx + (rbar + 0.5 * g), cy, r2, r2, 0.0]])


def make_truth(name, g=0.1, eps_r=2.0, r=0.2):
    """The truths of DESIGN 14.3 as renderers with ``chi(grid)`` / ``theta(grid)``:

    * ``T1`` : ``shapes.TwoDisks(g, r, eps_r)``, two equal disks (the correct family);
    * ``T2`` : two ellipses with semi-axes (r, 1.15 r) = 0.2 x 0.23 for r = 0.2 (long axis along y), edge-to-edge gap g along x;
    * ``T3`` : two disks with eps_r = 2.0 (left) and 2.4 (right) (``eps_r`` is the left one, the right one is 1.2 times it), gap g;
    * ``T4`` : ``BridgedDisks``: two touching disks and a bridge rectangle 0.05 wide, full disk height (g is ignored: the true gap is 0);
    * ``T5`` : T1 at g (default 0.1) plus a third disk of radius 0.06 centered 0.1 above the gap midpoint (0, 0.1), disjoint from both disks.
    """
    if name == "T1":
        return sh.TwoDisks(g=g, r=r, eps_r=eps_r)
    if name == "T2":
        ay = 1.15 * r
        ell = np.array([[-(r + 0.5 * g), 0.0, r, ay, 0.0], [r + 0.5 * g, 0.0, r, ay, 0.0]])
        return EllipseScene(ell, eps_r, name="T2", true_gap=g, description=f"two ellipses {r} x {ay:.3f} (long axis along y), gap {g}")
    if name == "T3":
        return EllipseScene(two_disks_ellipses(g, r, r), [eps_r, 1.2 * eps_r], name="T3", true_gap=g,
                            description=f"two disks, eps_r {eps_r:g} (left) and {1.2 * eps_r:g} (right), gap {g}")
    if name == "T4":
        return BridgedDisks(r=r, width=0.05, eps_r=eps_r)
    if name == "T5":
        ell = np.vstack([two_disks_ellipses(g, r, r), [[0.0, 0.1, 0.06, 0.06, 0.0]]])
        return EllipseScene(ell, eps_r, name="T5", true_gap=g, description=f"two disks, gap {g}, plus a disk of radius 0.06 at (0, 0.1)")
    raise ValueError(f"unknown truth {name!r}; choose from {TRUTH_NAMES}")


def feature_reference(truth, g, r=0.2, eps_r=2.0):
    """A ``shapes.TwoDisks`` that defines the REGIONS of the gap-valley feature (``gapvalley.feature_geometry``) for a truth: the two-disk reference with gap
    ``g`` (the truth's true gap), radius ``r`` and ``eps_r`` (which sets delta = log(eps_r) / 2). The feature itself is evaluated on the data of ``truth``;
    ``FeatureObject`` bundles the two."""
    return sh.TwoDisks(g=max(float(g), 0.0), r=r, eps_r=eps_r)


class FeatureObject:
    """The object ``gapvalley.feature_geometry`` expects (``left``, ``right``, ``g``, ``eps_r``, ``cx``, ``cy``, ``dr`` of a reference ``TwoDisks``) whose
    ``theta(grid)`` (used for the pixelized-truth check) is that of the actual ``truth``."""

    def __init__(self, truth, reference):
        self.truth, self.reference = truth, reference
        for k in ("left", "right", "g", "eps_r", "cx", "cy", "dr", "r"):
            setattr(self, k, getattr(reference, k))

    def theta(self, grid, ss=None):
        return self.truth.theta(grid, ss) if ss is not None else self.truth.theta(grid)


def pixel_area_sum_check(grid, truth):
    """The integral of the coverage (sum of the pixel areas) of ``truth`` minus its exact area, a check that the rendering conserves area (0 up to round-off for a shape inside the DOI)."""
    if isinstance(truth, EllipseScene):
        return float(truth.coverage(grid).sum() * grid.h**2 - np.sum(np.pi * truth.ellipses[:, 2] * truth.ellipses[:, 3]))
    if isinstance(truth, BridgedDisks):
        return float(truth.coverage(grid).sum() * grid.h**2 - truth.union_area_exact())
    return float(truth.coverage(grid).sum() * grid.h**2 - 2.0 * math.pi * truth.r**2)
