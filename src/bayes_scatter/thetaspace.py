"""Posterior in theta-space with a dense Laplace mass matrix (docs/DESIGN.md section 11.2).

The unknown is theta = log eps_r on the n x n DOI grid ([i, j] = [x, y], C-order flattening, N = n^2) and the
posterior is exp(-U(theta)) with

    U(theta) = Phi(chi(theta)) + R(theta),   Phi = sum |F(chi) - d_obs|^2 / s^2,   chi = e^theta - 1,

R a prior energy from ``bayes_scatter.mrf`` (q-GGMRF, or a Matern / GMRF field cropped to the DOI). Nothing
here depends on the prior beyond the interface ``energy / value_and_grad / hessian``.

* ``ThetaTarget`` : U, its gradient (adjoint solves), the physical data misfit Phi of a theta or chi map.
* ``map_theta`` : the MAP by SciPy L-BFGS-B on a jitted value-and-grad, and ``smooth_start``, the short
  smooth-prior MAP in whitened coordinates (``inference.map_estimate``, Matern) that supplies its start. A
  start matters because R (a Matern precision has condition number ~10^7) and the data term (Gauss-Newton top
  eigenvalue ~10^7) make U badly scaled in theta, and L-BFGS needs many iterations from theta = mu; the
  whitened Matern MAP is well conditioned and lands near the MAP of any of the priors.
* ``ThetaTarget.laplace_mass(theta_hat)`` : H = H_GN + nabla^2 R(theta_hat), H_GN = (2/s^2) Re(J^H J) with
  J = dd/dtheta from ``jacobian`` (exact, by reciprocity), dense (N, N), checked symmetric positive definite
  by Cholesky H = L_H L_H^T (``LaplaceMass``).
* ``ThetaTarget.zeta_target(mass)`` : the target for the samplers of ``samplers`` in the coordinates
  theta = center + L_H^-T zeta. Those samplers target exp(-Phi'(zeta)) N(zeta; 0, I), so the potential is

      Phi'(zeta) = U(theta(zeta)) - |zeta|^2 / 2,   grad Phi' = L_H^-1 grad U(theta) - zeta,

  which makes the target exactly exp(-U(theta(zeta))): the Jacobian of an affine map is constant, so this holds for
  any fixed center and any SPD H (a poor H only costs efficiency). Phi' is NOT the data misfit: report ``Phi`` of
  the stored draws (``ThetaTarget.misfit``, ``inference.physical_phi_of_draws``), never the sampler's trace.

``link="identity"`` (chi = theta) and ``use_born=True`` exist for linear-Gaussian validation only, like
``likelihood.Target``: with both the posterior is exactly Gaussian in theta.
"""

import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

from . import jacobian as jb
from .inference import map_estimate
from .likelihood import Target, misfit
from .mrf import WhitenedMap
from .priors import GaussianField, get_link


# ----------------------------------------------------------------------------------------------
# Laplace mass
# ----------------------------------------------------------------------------------------------


@dataclass
class LaplaceMass:
    """Dense Laplace mass matrix at ``theta_hat``: H = H_GN + nabla^2 R, its lower Cholesky factor, and diagnostics.

    ``H`` and ``chol`` are NumPy (N, N) float64 with H = chol chol^T; ``theta_hat`` is (n, n). ``info`` holds the
    timings of each stage in seconds (``time_fields_s``: the Nt + Nr solves, ``time_gn_s``: the Gauss-Newton
    Gram products, ``time_prior_hessian_s``, ``time_cholesky_s``, ``time_total_s``), ``asymmetry`` (max |H - H^T|
    relative to max |H|), ``pivot_ratio_sq`` = (max / min diag(chol))^2 (a lower bound for cond(H)), and
    ``jacobian_check_rel_err`` when columns were verified against ``jax.jvp``.
    """

    theta_hat: np.ndarray
    H: np.ndarray
    chol: np.ndarray
    info: dict = field(default_factory=dict)

    def theta_variance(self):
        """diag(H^-1) as an (n, n) array: the Laplace posterior variance of theta, from the Cholesky factor
        (diag(H^-1)_p = sum_k (L^-1)_{kp}^2)."""
        Linv = solve_triangular(self.chol, np.eye(self.chol.shape[0]), lower=True)
        return np.sum(Linv**2, axis=0).reshape(self.theta_hat.shape)

    def theta_std(self):
        """Laplace posterior standard deviation of theta, (n, n)."""
        return np.sqrt(self.theta_variance())


# ----------------------------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------------------------


class ThetaTarget:
    """U(theta) = Phi(chi(theta)) + R(theta) for one data set; see the module docstring.

    Parameters
    ----------
    forward : ``forward.Forward``; its grid must match ``prior.n``.
    prior : a ``mrf.ThetaPrior`` (``QGGMRF`` or ``GaussianEnergy``).
    d_obs : observed data (Nt, Nr), complex; ``s`` the noise level (E|e|^2 = s^2).
    link : "log_eps" (default, chi = e^theta - 1) or "identity" (validation only).
    use_born : forward map d = A chi (no solve) instead of the full model (validation only).

    Methods are jitted and take/return JAX arrays; ``aux`` is the (Nt, n, n) field array of the last solve, a pure
    warm start (use ``tol <= 1e-10`` in the forward model when sampling, DESIGN 2.4).
    """

    def __init__(self, forward, prior, d_obs, s, link="log_eps", use_born=False):
        if prior.n != forward.n:
            raise ValueError(f"prior grid n={prior.n} does not match forward grid n={forward.n}")
        d_obs = jnp.asarray(d_obs, dtype=jnp.complex128)
        if d_obs.shape != (forward.n_tx, forward.n_rx):
            raise ValueError(f"d_obs must have shape {(forward.n_tx, forward.n_rx)}, got {d_obs.shape}")
        if not s > 0:
            raise ValueError("noise level s must be positive")
        self.forward, self.prior, self.d_obs = forward, prior, d_obs
        self.s = float(s)
        self.n = forward.n
        self.link = get_link(link)
        self.use_born = bool(use_born)
        link_fn, energy, s_ = self.link, prior.energy, self.s

        def phi_of_chi(chi, aux):
            if self.use_born:
                return misfit(forward.born(chi), d_obs, s_), aux
            d, u = forward(chi, aux)
            return misfit(d, d_obs, s_), u

        def u_fn(theta, aux):
            phi, u = phi_of_chi(link_fn(theta), aux)
            return phi + energy(theta), u

        vg = jax.value_and_grad(u_fn, has_aux=True)

        def potential(theta, aux):
            u_val, u = u_fn(theta, aux)
            return u_val, jax.lax.stop_gradient(u)

        def potential_and_grad(theta, aux):
            (u_val, u), g = vg(theta, aux)
            return u_val, g, jax.lax.stop_gradient(u)

        def misfit_of_chi(chi, aux):
            phi, u = phi_of_chi(chi, aux)
            return phi, jax.lax.stop_gradient(u)

        self._potential = jax.jit(potential)
        self._potential_and_grad = jax.jit(potential_and_grad)
        self._misfit_of_chi = jax.jit(misfit_of_chi)
        self._misfit = jax.jit(lambda theta, aux: misfit_of_chi(link_fn(theta), aux))
        self._chi = jax.jit(link_fn)

    # ---- maps ----

    def chi(self, theta):
        """Contrast chi = link(theta), same shape as theta."""
        return self._chi(jnp.asarray(theta, dtype=jnp.float64))

    def eps_r(self, theta):
        """Relative permittivity 1 + chi(theta)."""
        return 1.0 + self.chi(theta)

    def initial_aux(self):
        """Warm start for the first solve: the incident field (the exact total field at chi = 0)."""
        return self.forward.u_inc

    # ---- potential ----

    def potential(self, theta, aux):
        """``(U, aux_new)``: U(theta) = Phi + R and the total fields of this solve."""
        return self._potential(jnp.asarray(theta, dtype=jnp.float64), aux)

    def potential_and_grad(self, theta, aux):
        """``(U, grad U (n, n), aux_new)``; the data part of the gradient is the adjoint solve."""
        return self._potential_and_grad(jnp.asarray(theta, dtype=jnp.float64), aux)

    def misfit(self, theta, aux=None):
        """``(Phi, aux_new)``: the physical data misfit of a theta map (no prior term)."""
        return self._misfit(jnp.asarray(theta, dtype=jnp.float64), self.initial_aux() if aux is None else aux)

    def misfit_of_chi(self, chi, aux):
        """``(Phi, aux_new)`` for a contrast map chi; the interface ``inference.physical_phi_of_draws`` uses."""
        return self._misfit_of_chi(jnp.asarray(chi), aux)

    def energies(self, theta, aux=None):
        """Plain-float breakdown ``{"U", "phi", "R"}`` at theta."""
        phi, _ = self.misfit(theta, aux)
        r = float(self.prior.energy(jnp.asarray(theta, dtype=jnp.float64)))
        return {"U": float(phi) + r, "phi": float(phi), "R": r}

    # ---- Laplace mass ----

    def laplace_mass(self, theta_hat, aux=None, verify_columns=0, seed=0):
        """H = H_GN + nabla^2 R(theta_hat), dense, SPD-checked, with its Cholesky factor (``LaplaceMass``).

        H_GN = (2/s^2) Re(J^H J), J = dd/dtheta = dd/dchi diag(dchi/dtheta) from ``jacobian`` (Nt + Nr solves at
        chi(theta_hat), exact; ``jacobian.gauss_newton_hessian``). Raises ``ValueError`` if H is not positive
        definite (it cannot be for a convex R, so this signals a bug or a non-convex prior).

        ``verify_columns`` = k > 0 also compares k random columns of J with ``jax.jvp`` columns of the forward
        model and stores the largest relative difference in ``info["jacobian_check_rel_err"]``.
        """
        t_all = time.perf_counter()
        theta_hat = jnp.asarray(theta_hat, dtype=jnp.float64)
        chi = self.chi(theta_hat)
        t0 = time.perf_counter()
        u, w = jb.jacobian_factors(self.forward, chi, born=self.use_born)
        jax.block_until_ready((u, w))
        t_fields = time.perf_counter() - t0
        scale = jb.link_derivative(self.link, theta_hat)
        t0 = time.perf_counter()
        h_gn = np.asarray(jax.block_until_ready(jb.gauss_newton_hessian(u, w, self.s, scale)))
        t_gn = time.perf_counter() - t0
        t0 = time.perf_counter()
        h_prior = self.prior.hessian(theta_hat)
        t_prior = time.perf_counter() - t0
        H = h_gn + h_prior
        asym = float(np.abs(H - H.T).max() / np.abs(H).max())
        H = 0.5 * (H + H.T)
        t0 = time.perf_counter()
        try:
            L = cholesky(H, lower=True)
        except np.linalg.LinAlgError as err:
            raise ValueError(f"the Laplace mass matrix is not positive definite (Cholesky failed: {err}); "
                             f"min diag H = {H.diagonal().min():.3g}") from err
        t_chol = time.perf_counter() - t0
        info = {"time_fields_s": t_fields, "time_gn_s": t_gn, "time_prior_hessian_s": t_prior,
                "time_cholesky_s": t_chol, "time_total_s": time.perf_counter() - t_all, "asymmetry": asym,
                "pivot_ratio_sq": float((L.diagonal().max() / L.diagonal().min()) ** 2),
                "h_diag_min": float(H.diagonal().min()), "h_diag_max": float(H.diagonal().max()),
                "n_solves": int(0 if self.use_born else self.forward.n_tx + self.forward.n_rx)}
        if verify_columns:
            rng = np.random.default_rng(seed)
            cols = rng.choice(self.n * self.n, size=int(verify_columns), replace=False)
            J = jb.assemble_jacobian(u, w, scale)
            if self.use_born:
                ref = (self.forward.born_matrix() * np.asarray(scale).ravel()[None, :])[:, cols]
                info["jacobian_check_rel_err"] = float(jnp.max(jnp.linalg.norm(J[:, cols] - ref, axis=0)
                                                               / jnp.linalg.norm(ref, axis=0)))
            else:
                info["jacobian_check_rel_err"] = jb.verify_jacobian(self.forward, chi, J, cols, scale)
        return LaplaceMass(np.asarray(theta_hat), H, L, info)

    def zeta_target(self, mass, center=None):
        """``ZetaTarget`` in the coordinates theta = center + L_H^-T zeta (default center ``mass.theta_hat``)."""
        return ZetaTarget(self, mass, center)

    def zeta_target_with_map(self, wmap):
        """``ZetaTarget`` in the coordinates of any affine map ``wmap`` (``adapt.AffineMap``, or ``mrf.WhitenedMap``): theta =
        wmap.to_theta(zeta). The map is a constant of the target, so the chain is an exact HMC chain for U (module docstring)."""
        return ZetaTarget(self, None, wmap=wmap)

    def potential_flat(self):
        """The flat potential ``(theta (N,), aux) -> (U, grad U (N,), aux)`` that ``adapt`` works with."""
        pg, shape = self._potential_and_grad, (self.n, self.n)

        def potential(theta, aux):
            u_val, g, aux_new = pg(theta.reshape(shape), aux)
            return u_val, g.ravel(), aux_new

        return potential

    def reparametrized_phi_fn(self, mass, center=None):
        """The callable ``(zeta, aux) -> (Phi', grad Phi', aux_new)`` of ``zeta_target(mass, center)``, for the
        samplers of ``samplers`` (Phi'(zeta) = U(theta(zeta)) - |zeta|^2/2; module docstring)."""
        return self.zeta_target(mass, center).phi_fn


class ZetaTarget:
    """The target of ``ThetaTarget`` in Laplace-whitened coordinates, in the form ``inference.sample`` expects.

    theta = center + L_H^-T zeta, with H = L_H L_H^T the ``LaplaceMass``; ``phi_fn(zeta, aux)`` returns
    ``(Phi', grad Phi', aux_new)`` with Phi'(zeta) = U(theta(zeta)) - |zeta|^2 / 2 (module docstring). The sampler's
    own trace is Phi', not the data misfit. ``record(zeta)`` stacks theta and eps_r = 1 + chi(theta), shape
    (2, n, n), so the Welford moments of a run give both. ``use_born`` is copied from the target.

    ``wmap`` (an ``adapt.AffineMap`` or any object with ``center``, ``to_theta``, ``to_zeta`` and ``grad_to_zeta`` acting on
    (n, n) arrays) replaces the Laplace map: the coordinates are then theta = wmap.to_theta(zeta) and ``mass`` is not needed (None).
    """

    def __init__(self, target, mass, center=None, wmap=None):
        self.target = target
        self.mass = mass
        self.use_born = target.use_born
        if wmap is None:
            center = mass.theta_hat if center is None else center
            wmap = WhitenedMap(center, mass.chol)
        self.wmap = wmap
        self.center = self.wmap.center
        self.shape = (target.n, target.n)
        wmap, tgt = self.wmap, target

        def phi_fn(zeta, aux):
            u_val, g, aux_new = tgt._potential_and_grad(wmap.to_theta(zeta), aux)
            return u_val - 0.5 * jnp.sum(zeta * zeta), wmap.grad_to_zeta(g) - zeta, aux_new

        def phi_only(zeta, aux):
            u_val, aux_new = tgt._potential(wmap.to_theta(zeta), aux)
            return u_val - 0.5 * jnp.sum(zeta * zeta), None, aux_new

        def record(zeta):
            th = wmap.to_theta(zeta)
            return jnp.stack([th, 1.0 + tgt._chi(th)])

        self._phi_fn = jax.jit(phi_fn)
        self._phi_fn_no_grad = jax.jit(phi_only)
        self._record = record

    @property
    def phi_fn(self):
        return self._phi_fn

    @property
    def phi_fn_no_grad(self):
        return self._phi_fn_no_grad

    @property
    def record(self):
        return self._record

    def initial_aux(self):
        return self.target.initial_aux()

    def to_theta(self, zeta):
        return self.wmap.to_theta(zeta)

    def to_zeta(self, theta):
        return self.wmap.to_zeta(jnp.asarray(theta, dtype=jnp.float64))

    def overdispersed_starts(self, key, n_chains, scale=2.0):
        """``n_chains`` starts zeta = scale * N(0, I): overdispersed Laplace draws around the center, (chains, n, n)."""
        return scale * jax.random.normal(key, (n_chains,) + self.shape, dtype=jnp.float64)

    def misfit_of_chi(self, chi, aux):
        return self.target.misfit_of_chi(chi, aux)


# ----------------------------------------------------------------------------------------------
# MAP
# ----------------------------------------------------------------------------------------------


def map_theta(target, theta0, maxiter=400, gtol=1e-6, ftol=1e-13, maxcor=30):
    """MAP in theta-space: minimize U(theta) = Phi + R by SciPy L-BFGS-B on a jitted value-and-grad.

    The optimizer loop runs on the host; forward solves are warm-started from the fields of the previous
    evaluation. L-BFGS-B is not preconditioned, so give it a good start (``smooth_start``).

    Returns ``(theta_hat (n, n) jnp, info)`` with ``info`` = ``success``, ``message``, ``nit``, ``nfev``, ``U``,
    ``phi``, ``R``, ``grad_inf`` (max-norm of grad U at theta_hat), ``time_s`` and ``U_history`` (U at every
    function evaluation).
    """
    theta0 = jnp.asarray(theta0, dtype=jnp.float64)
    shape = theta0.shape
    pg = target._potential_and_grad

    @jax.jit
    def value_and_grad(x, aux):
        u_val, g, aux_new = pg(x.reshape(shape), aux)
        return u_val, g.ravel(), aux_new

    state = {"aux": target.initial_aux()}
    history = []

    def fun(x):
        u_val, g, aux_new = value_and_grad(jnp.asarray(x), state["aux"])
        u_val = float(u_val)
        if np.isfinite(u_val):  # keep the last good warm start if a line-search trial diverged
            state["aux"] = aux_new
        history.append(u_val)
        return u_val, np.asarray(g, dtype=np.float64)

    t0 = time.perf_counter()
    res = minimize(fun, np.asarray(theta0, dtype=np.float64).ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": float(ftol),
                            "maxcor": int(maxcor), "maxfun": 10 * int(maxiter) + 100})
    time_s = time.perf_counter() - t0
    theta_hat = jnp.asarray(res.x).reshape(shape)
    u_val, g, _ = value_and_grad(jnp.asarray(res.x), state["aux"])
    parts = target.energies(theta_hat, state["aux"])
    info = {"success": bool(res.success), "message": str(res.message), "nit": int(res.nit),
            "nfev": int(res.nfev), "U": float(u_val), "phi": parts["phi"], "R": parts["R"],
            "grad_inf": float(jnp.max(jnp.abs(g))), "time_s": time_s, "U_history": np.asarray(history)}
    return theta_hat, info


def smooth_start(forward, d_obs, s, grid, *, sigma=0.5, ell=0.4, nu=1.5, maxiter=100):
    """A short smooth-prior MAP in whitened coordinates, as the start of ``map_theta``: returns ``(theta0 (n, n), info)``.

    Minimizes Phi(xi) + |xi|^2 / 2 for the Matern prior (sigma, ell, nu; mean 0, the vacuum background) with
    ``inference.map_estimate`` (L-BFGS-B, at most ``maxiter`` iterations) and returns theta = log eps_r of the result.
    In whitened coordinates the prior term is the identity and the iteration is well conditioned, unlike the same
    problem in theta (module docstring). The MAP of the cropped Matern energy in theta-space is the same point
    (minimizing |xi|^2 over xi with crop(L xi) = theta gives theta^T C_D^-1 theta), so for the Matern prior this
    start is already the answer up to ``maxiter``. ``info`` is that of ``map_estimate`` plus ``time_s``.
    """
    field_ = GaussianField(grid, "matern", sigma=sigma, ell=ell, nu=nu, mean=0.0)
    target = Target(forward, field_, "log_eps", d_obs, s)
    xi_map, _, info = map_estimate(target, jnp.zeros(field_.xi_shape), maxiter=maxiter)
    return jnp.asarray(field_.theta(xi_map)), info
