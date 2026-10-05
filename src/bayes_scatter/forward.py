"""JAX forward model: FFT Lippmann-Schwinger solve, scattered data and Born approximation.

Conventions (docs/DESIGN.md 1-2): contrast chi = eps_r - 1 of shape (n, n), "ij" indexing; fields
u, u_inc of shape (Nt, n, n); data d[t, r] of shape (Nt, Nr), complex128. The discrete system for each
transmitter t is (I - G diag(chi)) u_t = u_inc,t, with G applied by FFT on a (2n x 2n) circular embedding.

Linear solves go through ``jax.lax.custom_linear_solve``, so reverse-mode derivatives with respect to chi
are the implicit-function (adjoint) gradient. Two deliberate departures from calling
``jax.scipy.sparse.linalg.gmres`` as a black box (DESIGN 2.4 names it):

* The Krylov method is our own restarted GMRES (``krylov.gmres_stacked``): JAX 0.11.2's
  ``solve_method="incremental"`` ends with a true residual about ||A|| x tol for these complex systems, and
  ``"batched"`` cannot exit early, so warm starts would save nothing (details in ``krylov.py``).
* We wrap ``custom_linear_solve`` ourselves. JAX's ``gmres`` reuses the *forward* solve for the transposed
  (adjoint) solve, including the warm start ``x0`` and the absolute tolerance ``tol * ||b_forward||``. Here
  the forward solve is warm-started from ``u0`` and the adjoint solve starts from zero with a tolerance
  relative to its own right-hand side. With identical forward solves, tol = 1e-8 and a 1e-2 relative
  misfit, this lowered the gradient error against a tol = 1e-13 reference from 2.8e-5 to 9.8e-7 (cold
  start) and from 2.6e-5 to 3.8e-9 (warm start); tests/test_gradients.py reproduces the comparison.

Implementer's choices:

* ``batch="stack"`` (default): one stacked GMRES over all transmitters. Every transmitter has its own Krylov
  space and stops at ``tol`` relative to its own ``||u_inc,t||``; the stack iterates until the slowest
  transmitter is done. ``batch="vmap"`` runs the same solver per transmitter under ``jax.vmap``; it is
  mathematically identical but slower (see scripts/validate_forward.py for timings). BiCGSTAB
  (``solver="bicgstab"``, JAX's implementation) is always vmapped per transmitter.
* JAX returns no iteration counts, so ``matvec_counts`` counts operator applications with a host callback.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.sparse.linalg import bicgstab

from .greens import build_operators
from .krylov import gmres_stacked


class _Config(NamedTuple):
    """Hashable solver configuration (static under jit)."""

    solver: str
    tol: float
    atol: float
    restart: int
    maxiter: int
    batch: str


def _apply_G(g_hat, w):
    """G w for w of shape (..., n, n): zero-pad to (2n, 2n), multiply in Fourier space, crop."""
    n = w.shape[-1]
    pad = [(0, 0)] * (w.ndim - 2) + [(0, n), (0, n)]
    out = jnp.fft.ifft2(g_hat * jnp.fft.fft2(jnp.pad(w, pad)))
    return out[..., :n, :n]


def _krylov(matvec, rhs, x0, cfg):
    """Solve matvec(x) = rhs, stopping at ||residual|| <= max(tol ||rhs||, atol) for each transmitter.

    ``rhs`` has shape (B, n, n) (stacked transmitters, GMRES only) or (n, n) (one transmitter, under
    vmap); ``matvec`` acts on the same shape.
    """
    if cfg.solver == "bicgstab":  # always one transmitter at a time (see Forward)
        x, _ = bicgstab(matvec, rhs, x0=x0, tol=cfg.tol, atol=cfg.atol, maxiter=cfg.maxiter * cfg.restart)
        return x
    if rhs.ndim == 2:  # one transmitter: a stack of one lane
        x = gmres_stacked(lambda v: matvec(v[0])[None], rhs[None], None if x0 is None else x0[None],
                          tol=cfg.tol, atol=cfg.atol, restart=cfg.restart, maxiter=cfg.maxiter)
        return x[0]
    return gmres_stacked(matvec, rhs, x0, tol=cfg.tol, atol=cfg.atol, restart=cfg.restart,
                         maxiter=cfg.maxiter)


def _solve_linear(g_hat, chi, b, x0, cfg, matvec_hook=None):
    """Solve (I - G diag chi) x = b with a differentiable adjoint; see the module docstring.

    The operator is complex-linear in x and real-analytic in chi. Its transpose (not conjugate
    transpose), which is what ``custom_linear_solve`` solves in the backward pass, is I - diag(chi) G
    because the kernel is symmetric.
    """

    def matvec(x):
        if matvec_hook is not None:
            matvec_hook()
        return x - _apply_G(g_hat, chi * x)

    def forward_solve(mv, rhs):
        return _krylov(mv, rhs, x0, cfg)

    def adjoint_solve(mv, rhs):
        return _krylov(mv, rhs, None, cfg)

    return lax.custom_linear_solve(matvec, b, solve=forward_solve, transpose_solve=adjoint_solve)


def _solve_fields(g_hat, u_inc, chi, u0, cfg):
    chi = jnp.asarray(chi)
    if u0 is not None:
        u0 = lax.stop_gradient(jnp.asarray(u0, dtype=u_inc.dtype))
    if cfg.batch == "stack":
        return _solve_linear(g_hat, chi, u_inc, u0, cfg)
    if u0 is None:
        return jax.vmap(lambda b: _solve_linear(g_hat, chi, b, None, cfg))(u_inc)
    return jax.vmap(lambda b, x0: _solve_linear(g_hat, chi, b, x0, cfg))(u_inc, u0)


def _scattered(R, chi, u):
    """d[t, r] = sum_m R[r, m] chi_m u_t,m."""
    return jnp.einsum("rm,tm->tr", R, (chi * u).reshape(u.shape[0], -1))


_apply_G_jit = jax.jit(_apply_G)
_solve_fields_jit = jax.jit(_solve_fields, static_argnames=("cfg",))
_scattered_jit = jax.jit(_scattered)


@partial(jax.jit, static_argnames=("cfg",))
def _forward(g_hat, R, u_inc, chi, u0, cfg):
    u = _solve_fields(g_hat, u_inc, chi, u0, cfg)
    return _scattered(R, chi, u), u


@jax.jit
def _born_matrix(R, u_inc):
    return jnp.einsum("rm,tm->trm", R, u_inc.reshape(u_inc.shape[0], -1)).reshape(
        u_inc.shape[0] * R.shape[0], -1
    )


@jax.jit
def _residual(g_hat, u_inc, chi, u):
    r = u - _apply_G(g_hat, chi * u) - u_inc
    ax = (-2, -1)
    return jnp.sqrt(jnp.sum(jnp.abs(r) ** 2, axis=ax) / jnp.sum(jnp.abs(u_inc) ** 2, axis=ax))


class Forward:
    """Differentiable scattering forward model for one ``Setup``.

    Parameters
    ----------
    setup : ``Setup`` (grid, antennas, wavelength). Transmitters and receivers must be farther than the
        equal-area radius a from every pixel center (``ValueError`` otherwise).
    solver : ``"gmres"`` (default, ``krylov.gmres_stacked``) or ``"bicgstab"`` (JAX).
    tol, atol : each transmitter's solve stops when ||r|| <= max(tol ||u_inc,t||, atol), with r the true
        residual of (I - G diag chi) u = u_inc.
    restart : GMRES restart length (Krylov dimension per cycle). Memory is 16 bytes * (restart + 1) * Nt * n^2,
        0.6 GB for the book setting. Long restarts matter at high contrast: with 40 the eps_r = 4 benchmark
        phantom stagnates (about 3000 operator applications, unconverged), with 100 it needs 96 to 295.
    maxiter : maximum number of GMRES restart cycles (BiCGSTAB: maxiter * restart iterations). If this cap
        is hit before ``tol`` the solve silently returns the best iterate; use ``residual`` to check.
    batch : ``"stack"`` (default) or ``"vmap"``, see the module docstring; forced to ``"vmap"`` for BiCGSTAB.

    Attributes
    ----------
    g_hat (2n, 2n), R (Nr, n^2), u_inc (Nt, n, n): complex128 jnp arrays (geometry only).

    Methods are jitted. ``solve_fields``, ``__call__`` and ``born`` are differentiable with respect to
    real chi (reverse mode, adjoint gradient); gradients do not flow into ``u0``.
    """

    def __init__(self, setup, solver="gmres", tol=1e-8, atol=0.0, restart=100, maxiter=30, *,
                 batch="stack"):
        if solver not in ("gmres", "bicgstab"):
            raise ValueError("solver must be 'gmres' or 'bicgstab'")
        if batch not in ("stack", "vmap"):
            raise ValueError("batch must be 'stack' or 'vmap'")
        if solver == "bicgstab":
            batch = "vmap"  # JAX's BiCGSTAB solves one system at a time
        self.setup = setup
        self.grid = setup.grid
        self.n = setup.grid.n
        self.n_tx = setup.antennas.n_tx
        self.n_rx = setup.antennas.n_rx
        ops = build_operators(setup)
        self.g_hat = jnp.asarray(ops.g_hat)
        self.R = jnp.asarray(ops.R)
        self.u_inc = jnp.asarray(ops.u_inc)
        self._cfg = _Config(solver, float(tol), float(atol), int(restart), int(maxiter), batch)
        self._n_matvec = 0
        self._counted_solve = jax.jit(self._make_counted_solve())

    @property
    def tol(self):
        return self._cfg.tol

    def _make_counted_solve(self):
        cfg = self._cfg

        def bump():
            self._n_matvec += 1

        def counted(g_hat, chi, b, x0):  # b, x0: one transmitter, shape (n, n)
            hook = lambda: jax.debug.callback(bump)
            if cfg.solver == "gmres":
                x0 = None if x0 is None else x0[None]
                return _solve_linear(g_hat, chi, b[None], x0, cfg, matvec_hook=hook)[0]
            return _solve_linear(g_hat, chi, b, x0, cfg, matvec_hook=hook)

        return counted

    def apply_G(self, w):
        """G w for w of shape (..., n, n) (complex or real); returns the same shape, complex."""
        return _apply_G_jit(self.g_hat, jnp.asarray(w))

    def solve_fields(self, chi, u0=None):
        """Total fields u (Nt, n, n) solving (I - G diag chi) u_t = u_inc,t for every transmitter.

        ``u0``: optional warm start of shape (Nt, n, n). The result does not depend on it beyond ``tol``.
        """
        return _solve_fields_jit(self.g_hat, self.u_inc, chi, u0, self._cfg)

    def scattered(self, chi, u):
        """Data d[t, r] = sum_m R[r, m] chi_m u_t,m, shape (Nt, Nr), from total fields u (Nt, n, n)."""
        return _scattered_jit(self.R, jnp.asarray(chi), u)

    def __call__(self, chi, u0=None):
        """Full forward map: returns (d, u) with d (Nt, Nr) and the total fields u (Nt, n, n)."""
        return _forward(self.g_hat, self.R, self.u_inc, jnp.asarray(chi), u0, self._cfg)

    def born(self, chi):
        """Born data: the scattered field with u replaced by u_inc, d = A chi, shape (Nt, Nr)."""
        return _scattered_jit(self.R, jnp.asarray(chi), self.u_inc)

    def born_matrix(self):
        """Dense Born matrix A (Nt*Nr, n^2), A[t*Nr + r, m] = R[r, m] u_inc,t[m]; small grids only."""
        return _born_matrix(self.R, self.u_inc)

    def residual(self, chi, u):
        """Relative residuals ||u_t - G(chi u_t) - u_inc,t|| / ||u_inc,t||, shape (Nt,)."""
        return _residual(self.g_hat, self.u_inc, jnp.asarray(chi), u)

    def matvec_counts(self, chi, u0=None, transmitters=None):
        """Operator applications (G products) used by the solve of each transmitter, int array.

        ``transmitters``: indices to count (default all); the result has one entry per index.
        Diagnostic only: solves the transmitters one at a time (as if alone, whereas the stacked solve
        iterates until its slowest transmitter is done) in a Python loop with a host callback in the
        operator, so it is slow. Counts include the residual evaluation at each GMRES restart.
        """
        chi = jnp.asarray(chi)
        idx = np.arange(self.n_tx) if transmitters is None else np.asarray(transmitters)
        counts = np.zeros(len(idx), dtype=int)
        for i, t in enumerate(idx):
            self._n_matvec = 0
            x0 = None if u0 is None else jnp.asarray(u0)[t]
            self._counted_solve(self.g_hat, chi, self.u_inc[t], x0).block_until_ready()
            jax.effects_barrier()
            counts[i] = self._n_matvec
        return counts
