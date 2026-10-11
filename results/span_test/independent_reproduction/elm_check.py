"""Implement the NMM of eqs. 15.1-15.6 literally and check that every output lies in span(T).

o = alpha g_r( w_r | beta g_c(w_c x + b) | + p ),  beta = T Gc^+,  alpha = T Gr^+,  g_c = arcsinh, g_r = sigmoid.
The data x_j are synthetic (a random complex linear map of the contrast image plus noise): the span property
does not depend on what x is, only on alpha = T * (something).
"""
import gen  # sets the thread-count environment variables; must precede the numpy import
import numpy as np
import scenes

rng = np.random.default_rng(5)
P = 100
L = S = 2 * P + 1  # Hecht-Nelson: 2n + 1 nodes
M = 400

T = gen.make_set(P, 42)
Q, _ = gen.orth_basis(T)
A = (rng.normal(size=(M, gen.N)) + 1j * rng.normal(size=(M, gen.N))) / np.sqrt(gen.N)


def data(imgs):  # (N, K) eps images -> (M, K) complex "scattered fields"
    x = A @ (imgs - 1.0)
    return x + 1e-3 * np.linalg.norm(x, axis=0) / np.sqrt(M) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # overflow-safe


X = data(T)
wc = rng.uniform(-1, 1, (L, M)) + 1j * rng.uniform(-1, 1, (L, M))
b = rng.uniform(-1, 1, L) + 1j * rng.uniform(-1, 1, L)
Gc = np.arcsinh(wc @ X + b[:, None])  # L x P complex
beta = T @ np.linalg.pinv(Gc)  # N x L complex, eq. 15.5
print("beta shape", beta.shape, " training fit of first layer |beta Gc - T|/|T| =",
      np.linalg.norm(beta @ Gc - T) / np.linalg.norm(T))

held = gen.make_set(300, 43)
t1 = np.stack([gen.render(scenes.test1_like(None), "painter", 4).ravel(),
               gen.render(scenes.test4_like(None), "painter", 4).ravel(),
               gen.render(scenes.test2_like(None), "painter", 4).ravel()], axis=1)
tests = {"held-out in-family (300)": held, "Test1/Test4/cross-like (3)": t1}

for wr_name, scale in (("w_r ~ U[-1,1] (saturating sigmoid)", 1.0), ("w_r ~ U[-1,1]/sqrt(N)", 1 / np.sqrt(gen.N))):
    wr = rng.uniform(-1, 1, (S, gen.N)) * scale
    p = rng.uniform(-1, 1, S)
    Z = np.abs(beta @ Gc)  # N x P real
    Gr = sigmoid(wr @ Z + p[:, None])  # S x P
    alpha = T @ np.linalg.pinv(Gr)  # N x S, eq. 15.6
    sv = np.linalg.svd(Gr, compute_uv=False)
    print(f"\n[{wr_name}] Gr rank (rcond default) {np.linalg.matrix_rank(Gr)}, cond {sv[0] / sv[-1]:.2e}, "
          f"training fit |alpha Gr - T|/|T| = {np.linalg.norm(alpha @ Gr - T) / np.linalg.norm(T):.2e}")
    for name, imgs in tests.items():
        Xn = data(imgs)
        z = np.abs(beta @ np.arcsinh(wc @ Xn + b[:, None]))
        o = alpha @ sigmoid(wr @ z + p[:, None])  # N x K outputs of the NMM (eq. 15.1)
        dist_span = gen.proj_err(Q, o)  # distance of the OUTPUT to span(T), relative to |o|
        err_model = np.linalg.norm(o - imgs, axis=0) / np.linalg.norm(imgs, axis=0)
        bound = gen.proj_err(Q, imgs)
        print(f"  {name}: output-to-span(T) relative distance max {dist_span.max():.2e}; "
              f"Err_model (this toy NMM) min {err_model.min():.4f}; projection bound min {bound.min():.4f}; "
              f"Err_model >= bound for all: {bool(np.all(err_model >= bound - 1e-12))}")

# the optimum of the training problem itself: alpha Gr = T is solved exactly, so training scenes are reproduced
print("\ntraining outputs reproduce T (rel err):", np.linalg.norm(alpha @ Gr - T) / np.linalg.norm(T))
