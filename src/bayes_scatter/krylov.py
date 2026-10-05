"""Restarted GMRES with per-lane tolerances, for stacks of independent systems.

Why not ``jax.scipy.sparse.linalg.gmres``? Measured with JAX 0.11.2 on the complex Lippmann-Schwinger
systems of this package (eps_r up to 6 on a 24 x 24 grid, tol 1e-6 ... 1e-12):

* ``solve_method="incremental"`` stops inside a restart cycle as soon as its Givens residual estimate
  passes the tolerance, but then builds the update from an identity-padded triangular factor and the
  *unmasked* rotated right-hand side, so the residual estimate itself is added to the solution along the
  next Krylov vector. The true residual ends up about ||A|| times the tolerance (7 to 8 times tol here) and
  does not improve with further restarts.
* ``solve_method="batched"`` respects the tolerance but always performs ``restart`` Arnoldi steps per cycle,
  so a warm start (or an easy right-hand side) saves nothing.

This module implements the textbook algorithm (Saad, Iterative Methods, 6.5): Arnoldi with modified
Gram-Schmidt (backward stable for GMRES, and the sweep touches only the Krylov vectors built so far, which at
the problem sizes of interest costs more than the operator itself), Givens rotations for the least-squares
problem, early exit inside a cycle when the rotated residual estimate drops below the target, a
back-substitution restricted to the steps actually taken, and an explicit recomputation of the true residual
at the end of every cycle. Complex dtypes are supported.

All arrays carry a leading "lane" axis B. Lanes are independent linear systems that share one ``matvec``
call (which must act on every lane separately); each lane stops at its own
``||r_b|| <= max(tol ||b_b||, atol)``. Lanes that converge early keep taking (ignored) Arnoldi steps while
other lanes in the stack are still iterating, which wastes some work but needs no per-lane control flow.
"""

import numpy as np
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import solve_triangular


def _norm(v):
    """2-norm over the last axis, shape (B,)."""
    return jnp.sqrt(jnp.sum(jnp.abs(v) ** 2, axis=-1))


def _safe(x):
    return jnp.where(x > 0, x, 1.0)


def gmres_stacked(matvec, b, x0=None, *, tol=1e-8, atol=0.0, restart=40, maxiter=25):
    """Solve A_b x_b = b_b for every lane b of ``b`` (shape (B, ...)) by restarted GMRES(restart).

    Parameters
    ----------
    matvec : callable, maps an array of the shape of ``b`` to the same shape, lane by lane.
    b : right-hand sides (B, ...), real or complex.
    x0 : optional initial guess of the shape of ``b``; zero if omitted.
    tol, atol : lane b stops when ||b_b - A x_b||_2 <= max(tol ||b_b||_2, atol).
    restart : Krylov dimension per cycle (memory is (restart + 1) * size of ``b``).
    maxiter : maximum number of restart cycles. If reached before the tolerance, the best iterate so far
        is returned without error; check the residual.

    Returns x with the shape of ``b``. Not differentiable (it is meant to run inside
    ``lax.custom_linear_solve``, which supplies the derivatives).
    """
    B = b.shape[0]
    shape = b.shape
    N = int(np.prod(shape[1:]))
    m = int(min(restart, N))
    dtype = b.dtype
    rdtype = jnp.finfo(dtype).dtype

    def mv(v):  # flat (B, N) -> flat (B, N)
        return matvec(v.reshape(shape)).reshape(B, N)

    bf = b.reshape(B, N)
    target = jnp.maximum(tol * _norm(bf), atol)
    xf = jnp.zeros_like(bf) if x0 is None else x0.reshape(B, N).astype(dtype)
    rf = bf - mv(xf)
    cols = jnp.arange(m)

    def one_cycle(xf, rf, rnorm):
        V = jnp.zeros((m + 1, B, N), dtype).at[0].set(rf / _safe(rnorm)[:, None])
        Rt = jnp.zeros((B, m, m + 1), dtype)  # Rt[b, k, :] is column k of the triangular factor
        cs = jnp.zeros((B, m), rdtype)
        sn = jnp.zeros((B, m), dtype)
        g = jnp.zeros((B, m + 1), dtype).at[:, 0].set(rnorm.astype(dtype))
        active = rnorm > target
        nused = jnp.zeros(B, jnp.int32)

        def cond(c):
            k, active = c[0], c[-1]
            return (k < m) & jnp.any(active)

        def body(c):
            k, V, Rt, cs, sn, g, nused, active = c

            def mgs(j, carry):
                w, h = carry
                vj = V[j]
                hj = jnp.sum(vj.conj() * w, axis=-1)
                return w - hj[:, None] * vj, h.at[:, j].set(hj)

            w, h = lax.fori_loop(0, k + 1, mgs, (mv(V[k]), jnp.zeros((B, m + 1), dtype)))
            hn = _norm(w)
            V = V.at[k + 1].set(w / _safe(hn)[:, None])
            col = h.at[:, k + 1].set(hn.astype(dtype))

            def rotate(i, col):
                c_i, s_i = cs[:, i], sn[:, i]
                x1, x2 = col[:, i], col[:, i + 1]
                col = col.at[:, i].set(c_i * x1 + s_i * x2)
                return col.at[:, i + 1].set(-s_i.conj() * x1 + c_i * x2)

            col = lax.fori_loop(0, k, rotate, col)
            a, bb = col[:, k], col[:, k + 1]
            abs_a, abs_b = jnp.abs(a), jnp.abs(bb)
            rho = jnp.sqrt(abs_a**2 + abs_b**2)
            c_k = jnp.where(rho > 0, abs_a / _safe(rho), 1.0)
            phase = jnp.where(abs_a > 0, a / _safe(abs_a), 1.0)
            s_k = jnp.where(rho > 0, phase * bb.conj() / _safe(rho), 0.0)
            # unitary [[c, s], [-conj(s), c]] with c real maps (a, bb) to (phase * rho, 0)
            col = col.at[:, k].set(c_k * a + s_k * bb).at[:, k + 1].set(0.0)
            cs = cs.at[:, k].set(c_k)
            sn = sn.at[:, k].set(s_k)
            Rt = Rt.at[:, k, :].set(col)
            gk = g[:, k]
            g = g.at[:, k].set(c_k * gk).at[:, k + 1].set(-s_k.conj() * gk)
            nused = nused + active.astype(jnp.int32)
            active = active & (jnp.abs(g[:, k + 1]) > target)
            return k + 1, V, Rt, cs, sn, g, nused, active

        _, V, Rt, _, _, g, nused, _ = lax.while_loop(
            cond, body, (0, V, Rt, cs, sn, g, nused, active)
        )
        # Back-substitute using only the first nused steps of each lane; the rest is masked to the identity.
        used = cols[None, :] < nused[:, None]
        R = jnp.swapaxes(Rt[:, :, :m], 1, 2)  # upper triangular, R[b, j, k] = Rt[b, k, j]
        both = used[:, :, None] & used[:, None, :]
        R = jnp.where(both, R, 0.0) + jnp.eye(m, dtype=dtype)[None] * (~used)[:, :, None]
        y = solve_triangular(R, jnp.where(used, g[:, :m], 0.0)[..., None], lower=False)[..., 0]
        xf = xf + jnp.einsum("bk,kbn->bn", y, V[:m])
        rf = bf - mv(xf)
        return xf, rf, _norm(rf)

    def outer_cond(c):
        cyc, _, _, rnorm = c
        return (cyc < maxiter) & jnp.any(rnorm > target)

    def outer_body(c):
        cyc, xf, rf, rnorm = c
        xf, rf, rnorm = one_cycle(xf, rf, rnorm)
        return cyc + 1, xf, rf, rnorm

    _, xf, _, _ = lax.while_loop(outer_cond, outer_body, (0, xf, rf, _norm(rf)))
    return xf.reshape(shape)
