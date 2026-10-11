"""Sanity checks of the generator and of the projection numerics."""
import time

import gen  # sets the thread-count environment variables; must precede the numpy import
import numpy as np

rng = np.random.default_rng(0)

# 1. area of single shapes fully inside the DOI: sum(coverage) * h^2 vs analytic area
h2 = gen.H ** 2
cases = [
    dict(kind="disk", r=0.8, cx=0.3, cy=-0.4, eps=3.0),
    dict(kind="ring", r3=0.4, r4=1.1, cx=-0.5, cy=0.7, eps=3.0),
    dict(kind="box", w=1.3, h=0.7, th=0.0, cx=0.1, cy=0.2, eps=3.0),
    dict(kind="box", w=1.0, h=1.0, th=0.6, cx=-0.2, cy=0.2, eps=3.0),
]
areas = [np.pi * 0.8 ** 2, np.pi * (1.1 ** 2 - 0.4 ** 2), 1.3 * 0.7, 1.0]
for s, a in zip(cases, areas):
    for ss in (1, 4, 8):
        c = gen.coverage(s, ss)
        print(f"{s['kind']:5s} ss={ss}: area from coverage {c.sum() * h2:.5f} analytic {a:.5f}  rel {c.sum() * h2 / a - 1:+.2e}")

# 2. orientation: [i, j] = [x, y]; a disk at (x=+1.5, y=0) must light up large i, middle j
c = gen.coverage(dict(kind="disk", r=0.3, cx=1.5, cy=0.0, eps=2.0), 4)
ii, jj = np.nonzero(c)
print("orientation: mean i", ii.mean(), "(expect", (1.5 + 2.5) / gen.H - 0.5, ") mean j", jj.mean(), "(expect", 2.5 / gen.H - 0.5, ")")

# 3. prior-draw statistics: P(r4 <= r3) should be 0.1975 under a naive draw; check area stats
cnt = 0
tot = 20000
for _ in range(tot):
    r3, r4 = rng.uniform(0.1, 1), rng.uniform(0.2, 2)
    cnt += r4 <= r3
print("P(r4<=r3) empirical", cnt / tot, "analytic", (0.8 ** 2 / 2 / 1.8) / 0.9)

# 4. timing and eps range
t0 = time.time()
T = gen.make_set(100, 1)
print("make_set(100) seconds", time.time() - t0, "min/max eps", T.min(), T.max(), "mean eps", T.mean())

# 5. projection numerics: QR/SVD vs lstsq vs normal equations; training columns have zero error
Q, s = gen.orth_basis(T)
print("rank", Q.shape[1], "of", T.shape[1], " s_max", s[0], " s_min", s[-1], " cond", s[0] / s[-1])
Mtest = gen.make_set(50, 2)
e_svd = gen.proj_err(Q, Mtest)
coef = np.linalg.lstsq(T, Mtest, rcond=None)[0]
e_lstsq = np.linalg.norm(Mtest - T @ coef, axis=0) / np.linalg.norm(Mtest, axis=0)
print("max |svd - lstsq| over 50 test columns:", np.abs(e_svd - e_lstsq).max())
print("training columns projection error max:", gen.proj_err(Q, T).max())
print("random held-out error: mean", e_svd.mean(), "min", e_svd.min(), "max", e_svd.max())
