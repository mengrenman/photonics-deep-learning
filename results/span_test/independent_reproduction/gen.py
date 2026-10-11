"""Independent implementation of the Table 15.2 training generator and the span-projection error.

Conventions: DOI [-2.5, 2.5]^2 m, n = 96 pixels per side, pixel (i, j) <-> (x, y) ("ij" indexing),
pixel centers x_i = -2.5 + (i + 1/2) h with h = 5/96.  Images are flattened in C order to N = 9216-vectors.
All arithmetic float64.

Assumptions (the book does not state them) are options:
  ring_rule   how r4 > r3 is enforced: 'resample' (redraw the pair), 'swap', 'resample_r4'
              (redraw r4 ~ U[max(0.2, r3), 2]), 'void' (ring absent when r4 <= r3)
  containment 'clip' (centers in [-2, 2]^2, shapes cut off by the DOI edge) or 'inside'
              (each shape redrawn until it lies entirely inside the DOI)
  rotate      False: rectangle/square axis aligned; True: random angle ~ U[0, pi)
  overlap     'painter' (order disk1, disk2, ring, rect, square; later overwrites earlier),
              'painter_rev', 'painter_rand' (random order per scene), 'max' (larger eps_r wins),
              'add' (contrasts add)
  ss          coverage supersampling: 1 = pixel-center membership (binary), 4 = 4x4 area fraction
              (boundary pixels get volume-averaged permittivity)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import numpy as np

try:
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=2)
except Exception:  # pragma: no cover
    pass

SIDE = 5.0
NPIX = 96
H = SIDE / NPIX
N = NPIX * NPIX
SHAPE_ORDER = ("disk1", "disk2", "ring", "rect", "square")


def draw_params(rng, ring_rule="resample", containment="clip", rotate=False):
    """One draw of the 22 Table-15.2 parameters (+ optional rotation angles).

    Returns a list of 5 shape dicts in the order disk1, disk2, ring, rect, square.
    """
    U = rng.uniform

    def inside_ok(kind, p):
        if kind == "disk":
            return (abs(p["cx"]) + p["r"] <= SIDE / 2) and (abs(p["cy"]) + p["r"] <= SIDE / 2)
        if kind == "ring":
            return (abs(p["cx"]) + p["r4"] <= SIDE / 2) and (abs(p["cy"]) + p["r4"] <= SIDE / 2)
        # box (rect / square): circumscribed box of a possibly rotated rectangle
        w, h, th = p["w"], p["h"], p["th"]
        ex = abs(w / 2 * np.cos(th)) + abs(h / 2 * np.sin(th))
        ey = abs(w / 2 * np.sin(th)) + abs(h / 2 * np.cos(th))
        return (abs(p["cx"]) + ex <= SIDE / 2) and (abs(p["cy"]) + ey <= SIDE / 2)

    def disk():
        while True:
            p = dict(r=U(0.1, 1.0), cx=U(-2, 2), cy=U(-2, 2), eps=U(1, 8))
            if containment == "clip" or inside_ok("disk", p):
                return dict(kind="disk", **p)

    def ring():
        while True:
            r3, r4 = U(0.1, 1.0), U(0.2, 2.0)
            if r4 <= r3:
                if ring_rule == "resample":
                    continue
                if ring_rule == "swap":
                    r3, r4 = r4, r3
                elif ring_rule == "resample_r4":
                    r4 = U(max(0.2, r3), 2.0)
                elif ring_rule == "void":
                    pass
                else:
                    raise ValueError(ring_rule)
            p = dict(r3=r3, r4=r4, cx=U(-2, 2), cy=U(-2, 2), eps=U(1, 8))
            if containment == "clip" or inside_ok("ring", p):
                return dict(kind="ring", **p)

    def box(square):
        while True:
            w = U(0.1, 2.0)
            h = w if square else U(0.1, 2.0)
            th = U(0, np.pi) if rotate else 0.0
            p = dict(w=w, h=h, th=th, cx=U(-2, 2), cy=U(-2, 2), eps=U(1, 8))
            if containment == "clip" or inside_ok("box", p):
                return dict(kind="box", **p)

    return [disk(), disk(), ring(), box(False), box(True)]


# ---------------------------------------------------------------------------------------------
def _indicator(shape, X, Y):
    k = shape["kind"]
    if k == "disk":
        return (X - shape["cx"]) ** 2 + (Y - shape["cy"]) ** 2 <= shape["r"] ** 2
    if k == "ring":
        d2 = (X - shape["cx"]) ** 2 + (Y - shape["cy"]) ** 2
        return (d2 >= shape["r3"] ** 2) & (d2 <= shape["r4"] ** 2) & (shape["r4"] > shape["r3"])
    if k == "box":
        c, s = np.cos(shape["th"]), np.sin(shape["th"])
        u = (X - shape["cx"]) * c + (Y - shape["cy"]) * s
        v = -(X - shape["cx"]) * s + (Y - shape["cy"]) * c
        return (np.abs(u) <= shape["w"] / 2) & (np.abs(v) <= shape["h"] / 2)
    raise ValueError(k)


def _bbox_radius(shape):
    k = shape["kind"]
    if k == "disk":
        return shape["r"], shape["r"]
    if k == "ring":
        return shape["r4"], shape["r4"]
    th, w, h = shape["th"], shape["w"], shape["h"]
    return (abs(w / 2 * np.cos(th)) + abs(h / 2 * np.sin(th)),
            abs(w / 2 * np.sin(th)) + abs(h / 2 * np.cos(th)))


def coverage(shape, ss=4):
    """Fractional area of each pixel inside `shape`, (96, 96) array indexed [i, j] = [x, y]."""
    cov = np.zeros((NPIX, NPIX))
    ex, ey = _bbox_radius(shape)
    i0 = max(int(np.floor((shape["cx"] - ex + SIDE / 2) / H)) - 1, 0)
    i1 = min(int(np.ceil((shape["cx"] + ex + SIDE / 2) / H)) + 1, NPIX)
    j0 = max(int(np.floor((shape["cy"] - ey + SIDE / 2) / H)) - 1, 0)
    j1 = min(int(np.ceil((shape["cy"] + ey + SIDE / 2) / H)) + 1, NPIX)
    if i1 <= i0 or j1 <= j0:
        return cov
    off = (np.arange(ss) + 0.5) / ss
    xs = (-SIDE / 2 + (np.arange(i0, i1)[:, None] + off[None, :]) * H).ravel()
    ys = (-SIDE / 2 + (np.arange(j0, j1)[:, None] + off[None, :]) * H).ravel()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ind = _indicator(shape, X, Y).astype(float)
    ind = ind.reshape(i1 - i0, ss, j1 - j0, ss).mean(axis=(1, 3))
    cov[i0:i1, j0:j1] = ind
    return cov


def render(shapes, overlap="painter", ss=4, rng=None, background=1.0):
    """eps_r image (96, 96), [i, j] = [x, y], for a list of shape dicts (order = painter order)."""
    order = list(range(len(shapes)))
    if overlap == "painter_rev":
        order = order[::-1]
    elif overlap == "painter_rand":
        order = list(rng.permutation(len(shapes)))
    eps = np.full((NPIX, NPIX), background)
    for idx in order:
        s = shapes[idx]
        c = coverage(s, ss)
        if overlap in ("painter", "painter_rev", "painter_rand"):
            eps = (1 - c) * eps + c * s["eps"]
        elif overlap == "max":
            eps = np.maximum(eps, background + c * (s["eps"] - background))
        elif overlap == "add":
            eps = eps + c * (s["eps"] - background)
        else:
            raise ValueError(overlap)
    return eps


def make_set(P, seed, ring_rule="resample", containment="clip", rotate=False, overlap="painter", ss=4):
    """P independent scenes, returned as (N, P) matrix of flattened eps_r images (columns)."""
    rng = np.random.default_rng(seed)
    out = np.empty((N, P))
    for p in range(P):
        sh = draw_params(rng, ring_rule, containment, rotate)
        out[:, p] = render(sh, overlap, ss, rng).ravel()
    return out


# ---------------------------------------------------------------------------------------------
def orth_basis(T, tol=None):
    """Orthonormal basis of col span(T) by SVD, discarding directions below tol * s_max."""
    U, s, _ = np.linalg.svd(T, full_matrices=False)
    if tol is None:
        tol = max(T.shape) * np.finfo(float).eps
    r = int((s > tol * s[0]).sum())
    return U[:, :r], s


def proj_err(Q, M):
    """Relative least-squares projection error ||M - Q Q^T M|| / ||M|| column by column."""
    R = M - Q @ (Q.T @ M)
    return np.linalg.norm(R, axis=0) / np.linalg.norm(M, axis=0)


def nearest_err(T, M):
    """Relative distance from each column of M to the nearest column of T."""
    d2 = (np.sum(M ** 2, 0)[None, :] + np.sum(T ** 2, 0)[:, None] - 2 * T.T @ M)
    return np.sqrt(np.maximum(d2.min(axis=0), 0)) / np.linalg.norm(M, axis=0)
