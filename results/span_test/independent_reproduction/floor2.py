"""Empty-scene floor and nominal Test-#1/#4-like bounds for P between 200 and 500 (baseline generator)."""
import gen
import numpy as np
import scenes

one = np.ones((gen.N, 1))
t1 = gen.render(scenes.test1_like(None), "painter", 4).reshape(-1, 1)
t4 = gen.render(scenes.test4_like(None), "painter", 4).reshape(-1, 1)
for P in (250, 300, 350, 400, 450):
    R = 5
    e = np.array([[gen.proj_err(gen.orth_basis(T)[0], X)[0] for X in (one, t1, t4)]
                  for T in (gen.make_set(P, 80000 + P + r) for r in range(R))])
    print(f"P={P}: empty {100*e[:,0].mean():.2f}% (min {100*e[:,0].min():.2f})  Test1 nominal {100*e[:,1].mean():.2f}% (min {100*e[:,1].min():.2f})  Test4 nominal {100*e[:,2].mean():.2f}% (min {100*e[:,2].min():.2f})  [{R} training sets]")
