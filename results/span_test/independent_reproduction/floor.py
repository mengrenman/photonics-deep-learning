"""Distance of the empty scene (eps_r = 1 everywhere) and of a single tiny weak object to span(T)."""
import gen
import numpy as np
import scenes

one = np.ones((gen.N, 1))
tiny = gen.render([dict(kind="disk", cx=0.0, cy=0.0, r=0.2, eps=1.2)], "painter", 4).reshape(-1, 1)
for P, R in ((100, 20), (200, 10), (500, 5)):
    e0, e1 = [], []
    for r in range(R):
        T = gen.make_set(P, 70000 + P + r)
        Q, _ = gen.orth_basis(T)
        e0.append(gen.proj_err(Q, one)[0]); e1.append(gen.proj_err(Q, tiny)[0])
    print(f"P={P}: empty scene (eps=1) distance to span(T): mean {100*np.mean(e0):.2f}% min {100*np.min(e0):.2f}% max {100*np.max(e0):.2f}%  (n={R} training sets)"
          f" | tiny weak disk (r=0.2, eps=1.2): mean {100*np.mean(e1):.2f}% min {100*np.min(e1):.2f}%")
