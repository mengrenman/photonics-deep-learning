"""Exact Jacobian of the scattering data with respect to the contrast, by reciprocity (docs/DESIGN.md 11.2).

Setting (DESIGN 2.2-2.3). For the contrast chi (n x n, flattened C order, N = n^2 pixels) and X = diag(chi),

    A u_t = u_inc,t,   A = I - G X,        d[t, r] = sum_m R[r, m] chi_m u_t[m] = (R X u_t)_r,

with G the (N x N) pixel Green's matrix, R the (Nr x N) receiver matrix and u_t the total field of transmitter t.

Derivation. The solution map is holomorphic in chi (A is affine in chi), so for a perturbation dX = diag(dchi)
(real or complex): A du_t = G dX u_t, hence du_t = A^-1 G dX u_t and

    dd[t, r] = (R dX u_t)_r + (R X du_t)_r = [R (I + X A^-1 G) dX u_t]_r.

The push-through identity X (I - G X)^-1 = (I - X G)^-1 X gives X A^-1 G = (I - X G)^-1 X G, so
I + X A^-1 G = (I - X G)^-1 and

    dd[t, r] = sum_p [R (I - X G)^-1]_{r p} u_t[p] dchi_p,   i.e.   J[(t, r), p] = w_r[p] u_t[p],

    w_r^T = R_r (I - X G)^-1  <=>  w_r = (I - X G)^-T R_r^T = (I - G^T X)^-1 R_r^T = A^-1 R_r^T,

where R_r is row r of R. The last step uses that G is symmetric, G^T = G (the kernel g(m - m') depends only on the
distance between pixel centers), and that X is diagonal. Only the plain transpose is used, never the conjugate
transpose, so the formula holds for complex chi as well. (A itself is not symmetric, even for real chi: A^T =
I - X G. What is symmetric is G, and that is all that is needed.) So the receiver "adjoint" fields w_r solve the same
system as the forward fields, with the receiver kernel row as the right-hand side:

    Nt solves (the total fields u_t, needed for the forward model anyway) + Nr solves (the w_r)

give every entry of the dense Jacobian J (Nt Nr x N), row index t Nr + r (C order of d). Because
R[r, m] = 2 pi k0 a J1(k0 a) u_inc,r[m] (u_inc,r the incident field of a unit line source at the receiver r; compare
docs/DESIGN.md 2.3), w_r = 2 pi k0 a J1(k0 a) u^(r), with u^(r) = A^-1 u_inc,r the total field of a source at the
receiver: this is the reciprocity theorem, the receiver at r_r sees the pixel p through the same Green's function
that the transmitter at r_r would use. For chi = 0, u_t = u_inc,t and w_r = R_r, so J is the Born matrix A_B of
``Forward.born_matrix``.

With theta = log eps_r (chi = e^theta - 1) the Jacobian with respect to theta is J_theta = J diag(e^theta); for
the identity link it is J itself. Gauss-Newton Hessian of Phi = sum |d - d_obs|^2 / s^2 with respect to theta:

    H_GN = (2 / s^2) Re(J_theta^H J_theta),   (J^H J)_{pq} = e^{theta_p + theta_q} (U^H U)_{pq} (W^H W)_{pq},

where U (Nt x N) has rows u_t and W (Nr x N) rows w_r, because the sum over (t, r) factorizes. This Hadamard product
of two Gram matrices costs (Nt + Nr) N^2 complex multiply-adds instead of Nt Nr N^2 and never forms J
(``gauss_newton_hessian``); ``gauss_newton_hessian_dense`` forms it from J for tests.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .priors import get_link


def jacobian_factors(forward, chi, born=False):
    """The fields behind the Jacobian: ``(u, w)`` with u (Nt, n, n) the total fields and w (Nr, n, n) the
    reciprocal fields w_r = A^-1 R_r^T, solved with the tolerance of ``forward``.

    ``born=True`` returns the Born pair (u_inc, R rows) instead, which is the Jacobian at chi = 0 whatever chi
    is (the Born model is linear in chi).
    """
    n = forward.n
    rhs = forward.R.reshape(forward.n_rx, n, n)
    if born:
        return forward.u_inc, rhs
    chi = jnp.asarray(chi)
    u = forward.solve_fields(chi)
    w = forward.solve_rhs(chi, rhs)
    return u, w


def assemble_jacobian(u, w, scale=None):
    """Dense J (Nt Nr, N) complex128 from the fields: J[t Nr + r, p] = u[t, p] w[r, p] scale[p].

    ``scale`` (n, n) or None is the derivative d chi / d (variable) per pixel (e^theta for theta).
    """
    nt, nr = u.shape[0], w.shape[0]
    uf, wf = u.reshape(nt, -1), w.reshape(nr, -1)
    j = (uf[:, None, :] * wf[None, :, :]).reshape(nt * nr, -1)
    return j if scale is None else j * jnp.asarray(scale).reshape(1, -1)


def link_derivative(link, theta):
    """d chi / d theta per pixel for a link object: e^theta for log_eps, 1 for the identity link."""
    link = get_link(link)
    return jax.vmap(jax.grad(lambda t: jnp.real(link(t))))(jnp.asarray(theta).ravel()).reshape(jnp.shape(theta))


def jacobian_chi(forward, chi, born=False):
    """Dense Jacobian dd/dchi, complex (Nt Nr, n^2), by reciprocity (module docstring). Needs Nt + Nr solves.

    d has shape (Nt, Nr) and is flattened in C order, so row t Nr + r is d[t, r]; columns are the pixels in C order.
    """
    u, w = jacobian_factors(forward, chi, born=born)
    return assemble_jacobian(u, w)


def jacobian_theta(forward, theta, link="log_eps", born=False):
    """Dense Jacobian dd/dtheta = dd/dchi diag(dchi/dtheta), complex (Nt Nr, n^2), chi = link(theta).

    ``link`` is "log_eps" (dchi/dtheta = e^theta), "identity" or a link object of ``priors``.
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)
    chi = get_link(link)(theta)
    u, w = jacobian_factors(forward, chi, born=born)
    return assemble_jacobian(u, w, link_derivative(link, theta))


@jax.jit
def _gram_hadamard(uf, wf, scale, factor):
    gu = jnp.conj(uf).T @ uf  # (U^H U)_{pq} = sum_t conj(u_tp) u_tq
    gw = jnp.conj(wf).T @ wf
    return factor * scale[:, None] * jnp.real(gu * gw) * scale[None, :]


def gauss_newton_hessian(u, w, s, scale=None):
    """H_GN = (2 / s^2) Re(J^H J), shape (N, N) float64, from the fields without forming J (module docstring).

    ``u`` (Nt, n, n), ``w`` (Nr, n, n) from ``jacobian_factors``; ``scale`` (n, n) the per-pixel d chi / d variable
    (None means 1, the Hessian with respect to chi). Equal to ``gauss_newton_hessian_dense`` up to round-off.
    """
    nt, nr = u.shape[0], w.shape[0]
    n_pix = int(np.prod(u.shape[1:]))
    sc = jnp.ones(n_pix) if scale is None else jnp.asarray(scale, dtype=jnp.float64).ravel()
    return _gram_hadamard(u.reshape(nt, -1), w.reshape(nr, -1), sc, 2.0 / float(s) ** 2)


def gauss_newton_hessian_dense(J, s):
    """(2 / s^2) Re(J^H J) from a dense Jacobian J (rows = complex data, columns = parameters), float64."""
    J = jnp.asarray(J)
    return (2.0 / float(s) ** 2) * (J.real.T @ J.real + J.imag.T @ J.imag)


def jvp_columns(forward, chi, cols, scale=None):
    """Columns ``cols`` (pixel flat indices) of dd/dchi by forward-mode autodiff of the forward model: complex array
    (Nt Nr, len(cols)). ``scale`` (n, n) multiplies the tangent (dchi/dtheta for theta columns).

    This is the independent check of ``jacobian_chi`` (``jax.jvp`` through the Krylov solve), one solve pair per
    column, so use it for a few columns.
    """
    n = forward.n
    chi = jnp.asarray(chi)
    dd_fn = jax.jit(lambda c, e: jax.jvp(lambda c_: forward(c_)[0], (c,), (e,))[1])
    out = []
    for p in np.atleast_1d(cols):
        e = jnp.zeros(n * n).at[int(p)].set(1.0).reshape(n, n)
        if scale is not None:
            e = e * jnp.asarray(scale)
        out.append(dd_fn(chi, e.astype(chi.dtype)).ravel())
    return jnp.stack(out, axis=1)


def verify_jacobian(forward, chi, J, cols, scale=None):
    """Largest relative difference ||J[:, p] - jvp_p|| / ||jvp_p|| over the pixel columns ``cols``.

    ``J`` is the dense Jacobian (with respect to chi, or to theta if ``scale`` is given and applied to both).
    """
    ref = jvp_columns(forward, chi, cols, scale)
    got = jnp.asarray(J)[:, np.atleast_1d(cols)]
    num = jnp.linalg.norm(got - ref, axis=0)
    return float(jnp.max(num / jnp.linalg.norm(ref, axis=0)))
