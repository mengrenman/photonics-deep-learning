"""Our own Test-#1-like, Test-#2-like (extra) and Test-#4-like scenes, read off Fig. 15.9 by eye.

Fig. 15.9 (DOI [-2.5, 2.5]^2, color scale 1..2):
  Test #1: a disk (diameter ~1.1 m) centered near (0, +1.0), eps_r ~ 1.5, and a thin horizontal rectangle
           (~1.6 x 0.37 m) near (0, -1.5), eps_r ~ 1.2.
  Test #4: three overlapping disks (radius ~0.55): top (0, +0.43), left (-0.43, -0.06), right (+0.24, -0.06);
           eps_r ~ 1.4, 1.6, 1.95; the right (orange) disk is drawn on top.
  Test #2 (extra): a cross, arms ~0.5 m wide, ~2.4 m long, eps_r ~ 1.5.
Each *_like(rng) draws one instantiation with jitter around these values (all eps_r <= 2); the nominal scene is
the jitter-free one (rng=None).
"""
import numpy as np


def _disk(cx, cy, r, eps):
    return dict(kind="disk", cx=cx, cy=cy, r=r, eps=eps)


def _box(cx, cy, w, h, eps):
    return dict(kind="box", cx=cx, cy=cy, w=w, h=h, th=0.0, eps=eps)


def test1_like(rng=None):
    if rng is None:
        return [_disk(0.0, 1.0, 0.55, 1.5), _box(0.0, -1.5, 1.6, 0.37, 1.2)]
    U = rng.uniform
    return [_disk(U(-0.2, 0.2), U(0.8, 1.2), U(0.4, 0.7), U(1.3, 2.0)),
            _box(U(-0.3, 0.3), U(-1.7, -1.3), U(1.0, 2.0), U(0.25, 0.5), U(1.1, 2.0))]


def test4_like(rng=None):
    if rng is None:
        return [_disk(0.0, 0.43, 0.55, 1.4), _disk(-0.43, -0.06, 0.55, 1.6), _disk(0.24, -0.06, 0.55, 1.95)]
    U = rng.uniform
    j = lambda c: c + U(-0.15, 0.15)
    return [_disk(j(0.0), j(0.43), U(0.45, 0.7), U(1.3, 1.6)),
            _disk(j(-0.43), j(-0.06), U(0.45, 0.7), U(1.5, 1.8)),
            _disk(j(0.24), j(-0.06), U(0.45, 0.7), U(1.8, 2.0))]


def test2_like(rng=None):
    if rng is None:
        return [_box(0.0, 0.0, 0.5, 2.1, 1.5), _box(0.0, 0.0, 2.4, 0.5, 1.5)]
    U = rng.uniform
    e = U(1.3, 1.8)
    return [_box(0, 0, U(0.4, 0.6), U(1.8, 2.4), e), _box(0, 0, U(2.0, 2.6), U(0.4, 0.6), e)]
