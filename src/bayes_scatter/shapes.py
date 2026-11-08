"""The two-disk object family of docs/DESIGN.md 13.2: rendering and exact shape derivatives.

Object. Two disjoint disks of one relative permittivity ``eps_r`` in vacuum, side by side along x. The parameters are

    p = (g, r, eps_r, cx, cy, dr)

* ``g``    edge-to-edge gap (wavelengths), g >= 0;
* ``r``    mean radius, r1 = r + dr / 2 (left disk) and r2 = r - dr / 2 (right disk);
* ``eps_r`` common relative permittivity (contrast chi = eps_r - 1 inside, 0 outside);
* ``cx, cy`` the center of the pair: the disk centers are (cx -+ (r + g / 2), cy), so the center distance is 2 r + g
  whatever ``dr`` is and the edge-to-edge gap is g;
* ``dr``   radius difference r1 - r2 (0 for the symmetric family; the 6th parameter exists for the asymmetric checks).

Reflection x -> -x (about x = 0, the center of the DOI and of the antenna rings) maps (g, r, eps_r, cx, cy, dr) to
(g, r, eps_r, -cx, cy, -dr): ``EVEN_PARAMS`` = (g, r, eps_r, cy) are unchanged and ``ODD_PARAMS`` = (cx, dr) change sign, so the
derivative of chi with respect to an even (odd) parameter at a symmetric object (cx = dr = 0) is even (odd) under the reflection.

Rendering (arrays are (n, n), "ij" indexing, [i, j] = [x, y], as everywhere in this package). The contrast of a pixel is the
volume average ``(eps_r - 1) * coverage``, where the coverage is the fraction of the pixel area inside the disks, exactly as
``phantoms.paint`` does. ``TwoDisks.chi(grid)`` uses the EXACT area of disk and pixel (``disk_pixel_areas``, a closed form from
the area of a disk inside a quadrant, no sampling); ``TwoDisks.chi(grid, ss=k)`` uses the k x k sub-sampling of
``phantoms.coverage`` (the book-style rendering, with its 1 / k^2 quantization), which is what the finite-difference checks
use.

Shape derivatives. By the Reynolds transport theorem, the derivative of the area of a region inside a pixel with respect to
a parameter of its boundary is the integral of the normal speed over the part of the boundary inside the pixel,

    d area_pix / d p = integral over (boundary and pixel) of (v . n) ds,

with v the velocity of the boundary point and n the outward normal. For a circle of radius R about (xc, yc), the point at polar
angle phi has n = (cos phi, sin phi), and each parameter above moves the circle by a combination of a radius change (v . n = a), an
x translation (v . n = b cos phi) and a y translation (v . n = c sin phi), so v . n = a + b cos phi + c sin phi with the
coefficients of ``_NORMAL_SPEED``. Then d chi_pix / d p = (eps_r - 1) (d area_pix / d p) / h^2, and d chi / d eps_r = coverage.
Two evaluations of the boundary integral are provided:

* ``method="arcs"`` (default): cut each circle at its crossings with the pixel-edge lines; between two consecutive crossings
  the arc lies in a single pixel and the integral of a + b cos phi + c sin phi is closed form. Exact up to round-off.
* ``method="sampled"``: M equally spaced boundary points, each carrying the weight R dphi (the design's "dense sampling with the points
  binned into pixels"). The error is that of binning a point: a point that falls into the wrong side of a pixel edge moves its
  weight 2 pi R / M to the neighbor pixel, so the relative error in a pixel is of the order of the arc length per point over the
  arc length in the pixel, 1e-4 to 1e-3 at the default M.

``shape_derivatives_fd`` is the independent check (central differences of the rendering). Its agreement with the exact derivative
is limited by the rendering it differences: the exact rendering is smooth to round-off, so the difference is truncation error
O(step^2) of the central formula and round-off O(eps / step); a k x k sub-sampled rendering is a step function of the
parameter (each sub-sample that crosses the boundary changes the coverage by 1 / k^2), so its difference quotient has
noise of order 1 / (k^2 step) per pixel, which a larger step or finer sub-sampling reduces and the O(step^2) truncation error of a
larger step raises.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import phantoms

PARAM_NAMES = ("g", "r", "eps_r", "cx", "cy", "dr")
EVEN_PARAMS = ("g", "r", "eps_r", "cy")
ODD_PARAMS = ("cx", "dr")

# Normal speed v . n = a + b cos(phi) + c sin(phi) of each parameter for (left disk, right disk): ((a, b, c), (a, b, c)).
# Left center x1 = cx - (r + g/2), right center x2 = cx + (r + g/2), radii r1 = r + dr/2, r2 = r - dr/2.
_NORMAL_SPEED = {
    "g": ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)),    # centers move apart by 1/2 each; radii fixed
    "r": ((1.0, -1.0, 0.0), (1.0, 1.0, 0.0)),    # both radii +1 and the centers move apart by 1 each (gap fixed)
    "cx": ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),    # rigid translation in x
    "cy": ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),    # rigid translation in y
    "dr": ((0.5, 0.0, 0.0), (-0.5, 0.0, 0.0)),   # r1 +1/2, r2 -1/2
}


@dataclass(frozen=True)
class TwoDisks:
    """The two-disk object (module docstring). Lengths in wavelengths; ``eps_r`` > 1 is the common relative permittivity.

    Defaults: radius 0.2, eps_r 2, centered at the origin, equal radii. ``left`` / ``right`` are ``(xc, yc, radius)``.
    """

    g: float
    r: float = 0.2
    eps_r: float = 2.0
    cx: float = 0.0
    cy: float = 0.0
    dr: float = 0.0

    def __post_init__(self):
        for name in PARAM_NAMES:
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.g < 0.0:
            raise ValueError("the gap g must be >= 0 (the disks must not overlap)")
        if not self.r - 0.5 * abs(self.dr) > 0.0:
            raise ValueError("both radii must be positive: r > |dr| / 2")

    @property
    def left(self):
        """(xc, yc, radius) of the left disk."""
        return (self.cx - (self.r + 0.5 * self.g), self.cy, self.r + 0.5 * self.dr)

    @property
    def right(self):
        """(xc, yc, radius) of the right disk."""
        return (self.cx + (self.r + 0.5 * self.g), self.cy, self.r - 0.5 * self.dr)

    def replace(self, **kw):
        """A copy with some parameters changed."""
        return replace(self, **kw)

    def vector(self, names=PARAM_NAMES):
        """The parameter values for ``names`` as a float array."""
        return np.array([getattr(self, k) for k in names], dtype=np.float64)

    def check_inside(self, grid):
        """Raise ``ValueError`` unless both disks lie inside the DOI of ``grid`` (the derivatives integrate over the whole boundary)."""
        cxg, cyg = grid.center
        half = 0.5 * grid.side
        for xc, yc, rad in (self.left, self.right):
            if not (cxg - half <= xc - rad and xc + rad <= cxg + half and cyg - half <= yc - rad and yc + rad <= cyg + half):
                raise ValueError(f"the disk at ({xc:.4g}, {yc:.4g}) with radius {rad:.4g} does not lie inside the DOI "
                                 f"(side {grid.side}, center {grid.center})")

    def coverage(self, grid, ss=None):
        """Fraction of each pixel inside the disks, (n, n): exact area (``ss`` None) or ``phantoms.coverage`` with ss x ss sub-samples."""
        if ss is None:
            return sum(disk_pixel_areas(grid, *disk) for disk in (self.left, self.right)) / grid.h**2
        (x1, y1, r1), (x2, y2, r2) = self.left, self.right
        return phantoms.coverage(grid, lambda X, Y: ((X - x1) ** 2 + (Y - y1) ** 2 <= r1**2)
                                 | ((X - x2) ** 2 + (Y - y2) ** 2 <= r2**2), ss=ss)

    def chi(self, grid, ss=None):
        """Contrast chi = (eps_r - 1) coverage, (n, n) float64 (volume-averaged boundary pixels, like ``phantoms.paint``)."""
        return (self.eps_r - 1.0) * self.coverage(grid, ss)

    def theta(self, grid, ss=None):
        """theta = log eps_r = log(1 + chi) of the rendered object, (n, n)."""
        return np.log1p(self.chi(grid, ss))


# ----------------------------------------------------------------------------------------------
# Exact area of a disk inside a pixel
# ----------------------------------------------------------------------------------------------


def _half_chord_integral(u, radius):
    """S(u) = integral from -R to u of sqrt(R^2 - t^2) dt, for u clipped to [-R, R]; S(R) = pi R^2 / 2."""
    u = np.clip(u, -radius, radius)
    return 0.5 * (u * np.sqrt(np.maximum(radius * radius - u * u, 0.0)) + radius * radius * np.arcsin(u / radius)) \
        + 0.25 * np.pi * radius * radius


def _quadrant_area(u, v, radius):
    """Area of {(t, w): t <= u, w <= v} inside the disk of radius R about the origin; u and v broadcast.

    At abscissa t the disk covers w in [-h(t), h(t)], h = sqrt(R^2 - t^2), so the area below w = v is
    clamp(v, -h, h) + h. With v_c = clip(v, -R, R) and u* = sqrt(R^2 - v_c^2), h(t) > |v_c| exactly for |t| < u*: there the
    integrand is v_c + h; for |t| >= u* it is 2 h when v_c > 0 and 0 when v_c <= 0.
    """
    a = np.clip(u, -radius, radius)
    vc = np.clip(v, -radius, radius)
    ustar = np.sqrt(np.maximum(radius * radius - vc * vc, 0.0))
    a2 = np.clip(a, -ustar, ustar)
    middle = vc * (a2 + ustar) + _half_chord_integral(a2, radius) - _half_chord_integral(-ustar, radius)
    outer = 2.0 * _half_chord_integral(np.minimum(a, -ustar), radius) \
        + 2.0 * (_half_chord_integral(np.maximum(a, ustar), radius) - _half_chord_integral(ustar, radius))
    return middle + np.where(vc > 0.0, outer, 0.0)


def pixel_edges(grid):
    """Pixel edge coordinates ``(x_edges, y_edges)``, each of shape (n + 1,)."""
    cx, cy = grid.center
    off = np.arange(grid.n + 1) * grid.h - 0.5 * grid.side
    return cx + off, cy + off


def disk_pixel_areas(grid, xc, yc, radius):
    """Exact area of the intersection of the disk (center (xc, yc), radius) with every pixel, (n, n), "ij" indexing.

    From the area F(x, y) of the disk inside the quadrant {X <= x, Y <= y} at the (n + 1)^2 pixel corners (``_quadrant_area``): the
    area in a pixel is the second difference of F. The sum over pixels is pi R^2 for a disk inside the DOI. Round-off is about 1e-13
    of a pixel area (cancellation of numbers of the order of the disk area).
    """
    xe, ye = pixel_edges(grid)
    F = _quadrant_area((xe - xc)[:, None], (ye - yc)[None, :], radius)
    area = F[1:, 1:] - F[:-1, 1:] - F[1:, :-1] + F[:-1, :-1]
    return np.clip(area, 0.0, grid.h**2)  # the second difference can leave [0, h^2] by round-off (a few 1e-14 h^2)


# ----------------------------------------------------------------------------------------------
# Boundary integrals of the normal speed
# ----------------------------------------------------------------------------------------------


def _crossing_angles(grid, xc, yc, radius):
    """Sorted polar angles in [0, 2 pi] at which the circle crosses a pixel-edge line, with 0 and 2 pi included."""
    xe, ye = pixel_edges(grid)
    ang = [np.array([0.0, 2.0 * np.pi])]
    cx = (xe - xc) / radius
    cx = cx[np.abs(cx) <= 1.0]
    ang += [np.arccos(cx), -np.arccos(cx)]
    sy = (ye - yc) / radius
    sy = sy[np.abs(sy) <= 1.0]
    ang += [np.arcsin(sy), np.pi - np.arcsin(sy)]
    a = np.concatenate(ang) % (2.0 * np.pi)
    a = np.concatenate([np.sort(a), [2.0 * np.pi]])
    return a


def _pixel_index(grid, x, y):
    """Pixel indices (i, j) of points (x, y) and a mask of the points inside the DOI."""
    cx, cy = grid.center
    fi = np.floor((x - (cx - 0.5 * grid.side)) / grid.h).astype(np.int64)
    fj = np.floor((y - (cy - 0.5 * grid.side)) / grid.h).astype(np.int64)
    ok = (fi >= 0) & (fi < grid.n) & (fj >= 0) & (fj < grid.n)
    return fi, fj, ok


def circle_basis_integrals(grid, xc, yc, radius, method="arcs", n_samples=200_000):
    """Boundary integrals over (circle and pixel) of the three basis normal speeds, as an array (3, n, n):

    [0] integral of 1 ds, [1] of cos(phi) ds, [2] of sin(phi) ds (lengths, ds = R dphi), for the circle about (xc, yc).
    A normal speed a + b cos phi + c sin phi integrates to a [0] + b [1] + c [2]. ``method`` is "arcs" (exact, closed form per
    arc between pixel-edge crossings) or "sampled" (``n_samples`` equally spaced boundary points binned into pixels).
    """
    n = grid.n
    out = np.zeros((3, n, n))
    if method == "arcs":
        a = _crossing_angles(grid, xc, yc, radius)
        lo, hi = a[:-1], a[1:]
        keep = hi - lo > 1e-15
        lo, hi = lo[keep], hi[keep]
        mid = 0.5 * (lo + hi)
        i, j, ok = _pixel_index(grid, xc + radius * np.cos(mid), yc + radius * np.sin(mid))
        lo, hi, i, j = lo[ok], hi[ok], i[ok], j[ok]
        vals = (radius * (hi - lo), radius * (np.sin(hi) - np.sin(lo)), -radius * (np.cos(hi) - np.cos(lo)))
        for k in range(3):
            np.add.at(out[k], (i, j), vals[k])
    elif method == "sampled":
        phi = 2.0 * np.pi * (np.arange(n_samples) + 0.5) / n_samples
        i, j, ok = _pixel_index(grid, xc + radius * np.cos(phi), yc + radius * np.sin(phi))
        w = radius * 2.0 * np.pi / n_samples
        basis = (np.ones_like(phi), np.cos(phi), np.sin(phi))
        flat = i[ok] * n + j[ok]
        for k in range(3):
            out[k] = np.bincount(flat, weights=w * basis[k][ok], minlength=n * n).reshape(n, n)
    else:
        raise ValueError("method must be 'arcs' or 'sampled'")
    return out


def circle_mode_integrals(grid, xc, yc, radius, m, method="arcs"):
    """Exact boundary integrals over (circle and pixel) of the harmonic normal speeds cos(m phi) and sin(m phi), an array (2, n, n) (lengths, ds = R dphi).

    phi is the polar angle about the circle's own center (xc, yc). A normal displacement field v_n(phi) = a cos(m phi) + b sin(m phi) of the boundary changes the area of the
    region inside a pixel by a [0] + b [1] (Reynolds transport, module docstring). Between two consecutive crossings of the pixel-edge lines the arc lies in one pixel and
    the integrals are closed form, R [sin(m phi)]/m and -R [cos(m phi)]/m (R [phi] and 0 for m = 0); this is the same arc decomposition as ``circle_basis_integrals``, of which
    m = 1 is the (cos, sin) part: ``circle_mode_integrals(..., 1)`` equals ``circle_basis_integrals(...)[1:]`` to round-off. ``method`` "arcs" is the only one (exact).
    The circle must lie inside the DOI for the integral over the whole boundary to be the full derivative of the area.
    """
    if method != "arcs":
        raise ValueError("method must be 'arcs'")
    m = int(m)
    if m < 0:
        raise ValueError("m must be >= 0")
    out = np.zeros((2, grid.n, grid.n))
    a = _crossing_angles(grid, xc, yc, radius)
    lo, hi = a[:-1], a[1:]
    keep = hi - lo > 1e-15
    lo, hi = lo[keep], hi[keep]
    mid = 0.5 * (lo + hi)
    i, j, ok = _pixel_index(grid, xc + radius * np.cos(mid), yc + radius * np.sin(mid))
    lo, hi, i, j = lo[ok], hi[ok], i[ok], j[ok]
    if m == 0:
        np.add.at(out[0], (i, j), radius * (hi - lo))
    else:
        np.add.at(out[0], (i, j), radius * (np.sin(m * hi) - np.sin(m * lo)) / m)
        np.add.at(out[1], (i, j), -radius * (np.cos(m * hi) - np.cos(m * lo)) / m)
    return out


def coverage_derivatives(grid, obj, names=PARAM_NAMES, method="arcs", n_samples=200_000):
    """d coverage / d p for the shape parameters in ``names`` (not ``eps_r``): array (len(names), n, n).

    d coverage_pix / d p = (1 / h^2) * sum over the two disks of the boundary integral of the normal speed (module docstring).
    """
    obj.check_inside(grid)
    basis = [circle_basis_integrals(grid, xc, yc, rad, method, n_samples) for xc, yc, rad in (obj.left, obj.right)]
    out = np.zeros((len(names), grid.n, grid.n))
    for k, name in enumerate(names):
        if name not in _NORMAL_SPEED:
            raise ValueError(f"{name!r} is not a shape parameter ({sorted(_NORMAL_SPEED)})")
        for (a, b, c), bas in zip(_NORMAL_SPEED[name], basis):
            out[k] += a * bas[0] + b * bas[1] + c * bas[2]
    return out / grid.h**2


def shape_derivatives(grid, obj, names=PARAM_NAMES, method="arcs", n_samples=200_000):
    """d chi / d p_k for ``names`` (any of ``PARAM_NAMES``, ``eps_r`` included): array (len(names), n, n), "ij" indexing.

    d chi / d p = (eps_r - 1) d coverage / d p for a shape parameter (exact boundary integral, ``method`` "arcs" or "sampled") and
    d chi / d eps_r = coverage (exact area). The disks must lie inside the DOI (``TwoDisks.check_inside``).
    """
    obj.check_inside(grid)
    out = np.zeros((len(names), grid.n, grid.n))
    shape_names = [k for k in names if k != "eps_r"]
    if shape_names:
        cov_d = coverage_derivatives(grid, obj, shape_names, method, n_samples)
        for row, name in zip(cov_d, shape_names):
            out[list(names).index(name)] = (obj.eps_r - 1.0) * row
    if "eps_r" in names:
        out[list(names).index("eps_r")] = obj.coverage(grid)
    return out


def circle_stretch_integrals(grid, xc, yc, radius):
    """Exact boundary integrals over (circle and pixel) of cos^2(phi) ds and sin^2(phi) ds, an array (2, n, n) (lengths, ds = R dphi): the normal speeds of an
    x-stretch and a y-stretch of the circle about its own center (the semi-axis a changes by 1: v . n = cos^2 phi, resp. sin^2 phi). Same arc decomposition as
    ``circle_basis_integrals`` (closed form between pixel-edge crossings)."""
    n = grid.n
    out = np.zeros((2, n, n))
    a = _crossing_angles(grid, xc, yc, radius)
    lo, hi = a[:-1], a[1:]
    keep = hi - lo > 1e-15
    lo, hi = lo[keep], hi[keep]
    mid = 0.5 * (lo + hi)
    i, j, ok = _pixel_index(grid, xc + radius * np.cos(mid), yc + radius * np.sin(mid))
    lo, hi, i, j = lo[ok], hi[ok], i[ok], j[ok]
    half, osc = 0.5 * (hi - lo), 0.25 * (np.sin(2.0 * hi) - np.sin(2.0 * lo))
    np.add.at(out[0], (i, j), radius * (half + osc))
    np.add.at(out[1], (i, j), radius * (half - osc))
    return out


def stretch_derivatives(grid, obj):
    """d chi / d (s_x, s_y), array (2, n, n): the derivatives along the two EVEN shape nuisances of the circularity assumption.

    Both disks become ellipses with semi-axes (r + s_x, r + s_y) (s_x = s_y = 0 at the object); the centers move so that the edge-to-edge gap g stays FIXED (the inner
    edges at +- g / 2 about the pair center do not move). Normal speeds at the circles: s_y: sin^2 phi for both disks; s_x: cos^2 phi - cos phi (left disk) and
    cos^2 phi + cos phi (right disk), the cos phi being the translation that keeps the inner edge in place. The total area changes by pi r per disk and per unit
    of s_x or s_y. Used to ask how much of sigma_CRB(g) is due to the exact circularity of the model (the review of milestone 6): the span of {r, s_x, s_y} has
    dimension 2, since a common radius change is r = s_x = s_y. d chi / d s = (eps_r - 1) (integral of the normal speed over the boundary in the pixel) / h^2."""
    obj.check_inside(grid)
    (x1, y1, r1), (x2, y2, r2) = obj.left, obj.right
    L, Rr = circle_stretch_integrals(grid, x1, y1, r1), circle_stretch_integrals(grid, x2, y2, r2)
    Lb, Rb = circle_basis_integrals(grid, x1, y1, r1), circle_basis_integrals(grid, x2, y2, r2)
    dsx = (L[0] - Lb[1]) + (Rr[0] + Rb[1])
    dsy = L[1] + Rr[1]
    return (obj.eps_r - 1.0) * np.stack([dsx, dsy]) / grid.h**2


def elliptic_chi(grid, g, r, eps_r, sx=0.0, sy=0.0, ss=32):
    """Contrast of the two ELLIPSES of ``stretch_derivatives`` (semi-axes r + sx and r + sy, inner edges at +- g / 2 about the origin), sub-sampled ss x ss
    (``phantoms.coverage``); the finite-difference reference of the derivatives along sx and sy."""
    ax, ay = r + sx, r + sy
    x1, x2 = -(0.5 * g + ax), 0.5 * g + ax
    cov = phantoms.coverage(grid, lambda X, Y: (((X - x1) / ax) ** 2 + (Y / ay) ** 2 <= 1.0) | (((X - x2) / ax) ** 2 + (Y / ay) ** 2 <= 1.0), ss=ss)
    return (eps_r - 1.0) * cov


def shape_derivatives_fd(grid, obj, names=PARAM_NAMES, step=1e-5, ss=None):
    """Central finite differences of the rendering, (chi(p + step e_k) - chi(p - step e_k)) / (2 step), array (len(names), n, n).

    ``ss`` None differences the exact rendering, an integer the ss x ss sub-sampled one (module docstring on the limits of the
    agreement). ``step`` is in the units of the parameter (wavelengths; eps_r is dimensionless).
    """
    out = np.zeros((len(names), grid.n, grid.n))
    for k, name in enumerate(names):
        hi = obj.replace(**{name: getattr(obj, name) + step})
        lo = obj.replace(**{name: getattr(obj, name) - step})
        out[k] = (hi.chi(grid, ss) - lo.chi(grid, ss)) / (2.0 * step)
    return out


def total_contrast(grid, chi):
    """The integral of chi over the DOI, sum(chi) h^2 (a float)."""
    return float(np.sum(chi) * grid.h**2)


def total_contrast_gradient(obj, names=PARAM_NAMES):
    """Analytic gradient of the integral of chi = (eps_r - 1) (pi r1^2 + pi r2^2) with respect to ``names``, a float array.

    d/dg = d/dcx = d/dcy = 0 (areas do not change), d/dr = (eps_r - 1) 2 pi (r1 + r2) (= 2 pi R (eps_r - 1) per disk), d/deps_r =
    pi (r1^2 + r2^2), d/ddr = (eps_r - 1) pi (r1 - r2). Valid while the disks lie inside the DOI.
    """
    r1, r2 = obj.left[2], obj.right[2]
    table = {"g": 0.0, "cx": 0.0, "cy": 0.0,
             "r": (obj.eps_r - 1.0) * 2.0 * np.pi * (r1 + r2),
             "eps_r": np.pi * (r1**2 + r2**2),
             "dr": (obj.eps_r - 1.0) * np.pi * (r1 - r2)}
    return np.array([table[k] for k in names], dtype=np.float64)
