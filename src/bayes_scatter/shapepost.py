"""Posteriors in shape space: MAP, Laplace evidence, HMC, multimodality, thermodynamic integration, predictive checks (docs/DESIGN.md 14.2, 14.4).

A *problem* is anything with the interface below; ``ShapePosterior`` is the scattering problem of a ``shapemodels.ShapeModel`` and ``GaussianToy`` is a linear-Gaussian
validation problem with a known evidence. All algorithms here take a problem, so they are tested on the toy where the answers are exact.

Problem interface (u is an unconstrained vector in R^dim, arrays float64):

    dim                     number of parameters
    lik_const               constant with  log L(u) = lik_const - phi(u)  (log of the normalized likelihood of the data)
    data                    the observed data (any pytree of arrays); ``phi_and_grad``/``phi`` take it as an ARGUMENT so that compiled functions are shared by data sets
    initial_aux()           the solver warm start for the first evaluation (a pure accelerator: phi must not depend on it beyond the solver tolerance)
    log_prior(u)            JAX scalar: log density of u under the prior, **Jacobian included** (``shapemodels``)
    phi_and_grad(u, aux, data) -> (phi, dphi/du (dim,), aux_new)      JAX, jitted; phi = -log L + lik_const (the data misfit)
    phi(u, aux, data) -> (phi, aux_new)
    fisher(u) -> (dim, dim)  NumPy Gauss-Newton information of phi at u: Re(J^H J) (2 / s^2), the expected Hessian of phi
    simulate(u, key) -> data of the same kind generated at u with fresh noise
    with_data(data) -> the same problem bound to other data

Posterior at inverse temperature beta (beta = 1 is the posterior, beta = 0 the prior):  pi_beta(u) proportional to exp(-beta phi(u)) pi_u(u),  U_beta = beta phi - log pi_u.

Contents.

* ``map_estimate``: L-BFGS-B from several starts, then a damped Gauss-Newton ("Newton with the Fisher matrix plus the prior Hessian") polish; ``laplace_fit``
  (H = Fisher + prior Hessian, or the finite-difference Hessian of the exact gradient), ``laplace_log_evidence``.
* ``run_hmc`` / ``sample_posterior``: ``samplers.hmc`` in the coordinates u = center + C zeta of an affine map (C a lower Cholesky factor of the covariance; the Laplace map
  at the MAP: exact for any fixed map, the Jacobian is constant), potential Phi'(zeta) = U_beta(u(zeta)) - |zeta|^2 / 2. The map, beta, the data and the solver warm start travel
  in ``aux`` (traced), so one compilation serves every temperature, map and data set.
* ``detect_multimodality``: distinct MAP modes (several starts and the symmetry images of the model), chains whose means are separated in within-chain standard deviations, and
  a bimodality coefficient per coordinate. Multimodality is **reported**, never averaged over silently.
* ``ti_temperature`` / ``ti_combine`` / ``thermodynamic_integration``: log evidence by thermodynamic integration over power posteriors, beta_k = (k / K)^5, with its own
  convergence diagnostics (split R-hat and bulk ESS of phi per temperature, Monte Carlo error, the Euler-Maclaurin correction of the trapezoid, the coarse-fine difference,
  monotonicity of E_beta[phi]).
* ``predictive_check``: replicate the data at posterior draws with fresh noise and compare the observed misfit with the replicated ones; ``mode="refit"`` compares the
  best-fit misfits (calibrated: its p-value is uniform for a correct model), ``mode="draw"`` the classical misfit at the draw (conservative).
"""

import copy
import math
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy import stats
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize

from . import diagnostics as dg
from . import samplers as sm
from .adapt import AffineMap
from .ellipses import EllipseRenderer
from .likelihood import add_noise, misfit

LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------------------------------------------------------------------------
# Problems
# ----------------------------------------------------------------------------------------------


U_BOUND = 10.0  # |u_k| <= U_BOUND: the prior is truncated there (see ``_U_and_grad_fn``)


def _U_and_grad_fn(problem):
    """Jitted ``(u, aux, data, beta) -> (U_beta, grad U_beta, aux_new)`` with U_beta = beta phi - log pi_u. For beta = 0 the (expensive) data term is skipped by ``lax.cond``.

    **Truncation.** Outside the box |u_k| <= ``U_BOUND`` (10) the density is zero: U = +inf, the gradient is 0 and no solve is done, so a proposal that flies there is rejected
    at once. Why: a trajectory that leaves the region of moderate contrast ends where the Krylov solve stagnates (eps_r ~ e^5 or more) and costs seconds per evaluation. The mass removed
    is at most 2 sigmoid(-10) = 9e-5 per logistic coordinate and 1.5e-23 for a standard normal one (a change of log Z below 1e-3 for the models here).
    """
    if getattr(problem, "_U_and_grad", None) is not None:
        return problem._U_and_grad
    log_prior_vg = jax.value_and_grad(problem.log_prior)

    def fn(u, aux, data, beta):
        inside = jnp.all(jnp.abs(u) <= U_BOUND)

        def with_data(_):
            phi, g, aux_new = problem.phi_and_grad(u, aux, data)
            return beta * phi, beta * g, aux_new

        def without_data(_):
            return jnp.zeros(()), jnp.zeros_like(u), aux

        bphi, bg, aux_new = lax.cond((beta > 0.0) & inside, with_data, without_data, None)
        lp, glp = log_prior_vg(u)
        val = jnp.where(inside, bphi - lp, jnp.inf)
        grad = jnp.where(inside, bg - glp, jnp.zeros_like(u))
        return val, grad, aux_new

    problem._U_and_grad = jax.jit(fn)
    return problem._U_and_grad


def _prior_hessian_fn(problem):
    """Jitted ``u -> -d^2 log pi_u / du^2`` (dim, dim)."""
    if getattr(problem, "_prior_hessian", None) is None:
        problem._prior_hessian = jax.jit(lambda u: -jax.hessian(problem.log_prior)(u))
    return problem._prior_hessian


def _prepare(problem):
    """Create the cached compiled helpers on ``problem`` (potential, HMC potential, misfit scan) so that copies made by ``with_data`` share them (and their compilations)."""
    _U_and_grad_fn(problem)
    _prior_hessian_fn(problem)
    _zeta_phi_fn(problem)
    _phi_scan_fn(problem)
    return problem


def prior_hessian(problem, u):
    """-Hessian of log pi_u at u as a NumPy (dim, dim) matrix (positive semi-definite for the log-concave priors of ``shapemodels``)."""
    return np.asarray(_prior_hessian_fn(problem)(jnp.asarray(u, dtype=jnp.float64)))


class ShapePosterior:
    """The scattering posterior of a ``shapemodels.ShapeModel`` for one data set (the problem interface of the module docstring).

    Parameters
    ----------
    model : ``ShapeModel`` (F2, F2u, Fe), on the grid of ``forward``.
    forward : ``forward.Forward`` (use tol <= 1e-10 for sampling, DESIGN 2.4).
    d_obs : observed data (Nt, Nr) complex; ``s`` : noise level, E|e|^2 = s^2 (``likelihood``).
    renderer : ``ellipses.EllipseRenderer`` for the grid (built if omitted; share one between posteriors on one grid).

    phi(u) = sum |F(chi(u)) - d_obs|^2 / s^2 with chi the exact-coverage rendering of the model at u; its gradient is ONE adjoint solve (the gradient of phi with respect to chi,
    JAX's ``custom_linear_solve`` transpose) contracted with the exact boundary derivative d chi / d u (``ellipses``) and the analytic d p / d u. ``lik_const`` = -N log(pi s^2).
    """

    def __init__(self, model, forward, d_obs, s, renderer=None):
        if not s > 0:
            raise ValueError("noise level s must be positive")
        d_obs = jnp.asarray(d_obs, dtype=jnp.complex128)
        if d_obs.shape != (forward.n_tx, forward.n_rx):
            raise ValueError(f"d_obs must have shape {(forward.n_tx, forward.n_rx)}, got {d_obs.shape}")
        self.model, self.forward, self.s = model, forward, float(s)
        self.renderer = renderer if renderer is not None else EllipseRenderer(forward.grid)
        self.dim = model.dim
        self.n_data = forward.n_tx * forward.n_rx
        self.lik_const = -self.n_data * math.log(math.pi * self.s**2)
        self.data = d_obs
        self.names = model.param_names
        rend, s_ = self.renderer, self.s

        def chi_of_u(u):
            return model.chi(rend, u)

        def data_term(u, aux, data):
            d, fields = forward(chi_of_u(u), aux)
            return misfit(d, data, s_), fields

        vg = jax.value_and_grad(data_term, has_aux=True)

        def phi_and_grad(u, aux, data):
            (phi, fields), g = vg(u, aux, data)
            return phi, g, lax.stop_gradient(fields)

        def phi_only(u, aux, data):
            phi, fields = data_term(u, aux, data)
            return phi, lax.stop_gradient(fields)

        self._chi_of_u = jax.jit(chi_of_u)
        self._p_many = jax.jit(jax.vmap(model.p_from_u))
        self.phi_and_grad = jax.jit(phi_and_grad)
        self.phi = jax.jit(phi_only)
        self.log_prior = jax.jit(model.log_prior_u)
        def jac_cols(u, aux):
            _, tangent = jax.linearize(lambda v: forward(chi_of_u(v), aux)[0], u)
            eye = jnp.eye(model.dim)
            return jnp.stack([tangent(eye[k]).reshape(-1) for k in range(model.dim)], axis=1)

        self._jac_cols = jax.jit(jac_cols)
        self._simulate_clean = jax.jit(lambda u, aux: forward(chi_of_u(u), aux)[0])
        self._U_and_grad = None
        self._prior_hessian = None

    def initial_aux(self):
        return self.forward.u_inc

    def with_data(self, data):
        other = copy.copy(_prepare(self))
        other.data = jnp.asarray(data, dtype=jnp.complex128)
        return other

    def chi(self, u):
        """Contrast (n, n) at the unconstrained point u."""
        return self._chi_of_u(jnp.asarray(u, dtype=jnp.float64))

    def physical(self, u):
        """Physical parameters p (dim,) at u (NumPy)."""
        return np.asarray(self.model.p_from_u(jnp.asarray(u, dtype=jnp.float64)))

    def physical_many(self, u):
        """Physical parameters (m, k) at the rows of u (m, dim) (NumPy)."""
        return np.asarray(self._p_many(jnp.asarray(u, dtype=jnp.float64)))

    def data_jacobian(self, u):
        """J = d d / d u, complex (Nt Nr, dim): one ``jax.linearize`` of the forward map (one primal solve) and one tangent solve per parameter; row t Nr + r is d[t, r].
        The function is jitted once per problem (a fresh ``linearize`` per call would recompile the tangent every time)."""
        return self._jac_cols(jnp.asarray(u, dtype=jnp.float64), self.initial_aux())

    def fisher(self, u):
        """Gauss-Newton information of phi in u: (2 / s^2) Re(J^H J), NumPy (dim, dim), symmetrized. Its inverse is the Cramer-Rao covariance of the unconstrained parameters."""
        J = self.data_jacobian(u)
        F = (2.0 / self.s**2) * np.asarray(jnp.real(J.conj().T @ J))
        return 0.5 * (F + F.T)

    def simulate(self, u, key):
        """Data generated at u: F(chi(u)) + e with e ~ CN(0, s^2) from ``key`` (the model's own generative process, on the inference grid)."""
        d = self._simulate_clean(jnp.asarray(u, dtype=jnp.float64), self.initial_aux())
        return add_noise(key, d, self.s)

    def clean_data(self, u):
        return self._simulate_clean(jnp.asarray(u, dtype=jnp.float64), self.initial_aux())


class GaussianToy:
    """Linear-Gaussian validation problem: y = A u + e, e ~ N(0, noise_sd^2 I), prior u ~ N(0, prior_sd^2 I).

    ``A`` (m, dim). phi(u) = |y - A u|^2 / (2 noise_sd^2), ``lik_const`` = -(m / 2) log(2 pi noise_sd^2). The posterior is Gaussian, the evidence is
    ``exact_log_evidence`` = log N(y; 0, prior_sd^2 A A^T + noise_sd^2 I), and the Laplace evidence is exact. ``aux`` is a dummy scalar.
    """

    def __init__(self, A, y, noise_sd=1.0, prior_sd=1.0):
        self.A = np.asarray(A, dtype=np.float64)
        self.m, self.dim = self.A.shape
        self.noise_sd, self.prior_sd = float(noise_sd), float(prior_sd)
        self.data = jnp.asarray(y, dtype=jnp.float64)
        self.lik_const = -0.5 * self.m * math.log(2.0 * math.pi * self.noise_sd**2)
        self.names = tuple(f"u{k}" for k in range(self.dim))
        A_j, sd, ps = jnp.asarray(self.A), self.noise_sd, self.prior_sd

        def phi_fn(u, aux, data):
            r = data - A_j @ u
            return 0.5 * jnp.sum(r * r) / sd**2, aux

        vg = jax.value_and_grad(lambda u, data: 0.5 * jnp.sum((data - A_j @ u) ** 2) / sd**2)

        def phi_and_grad(u, aux, data):
            phi, g = vg(u, data)
            return phi, g, aux

        self.phi = jax.jit(phi_fn)
        self.phi_and_grad = jax.jit(phi_and_grad)
        self.log_prior = jax.jit(lambda u: -0.5 * jnp.sum(u * u) / ps**2 - 0.5 * self.dim * math.log(2.0 * math.pi * ps**2))
        self._U_and_grad = None
        self._prior_hessian = None

    def initial_aux(self):
        return jnp.zeros(())

    def with_data(self, y):
        other = copy.copy(_prepare(self))
        other.data = jnp.asarray(y, dtype=jnp.float64)
        return other

    def physical(self, u):
        return np.asarray(u)

    def fisher(self, u):
        return self.A.T @ self.A / self.noise_sd**2

    def simulate(self, u, key):
        return jnp.asarray(self.A) @ jnp.asarray(u) + self.noise_sd * jax.random.normal(key, (self.m,), dtype=jnp.float64)

    def exact_log_evidence(self, y=None):
        """log p(y) = log N(y; 0, prior_sd^2 A A^T + noise_sd^2 I)."""
        y = np.asarray(self.data if y is None else y, dtype=np.float64)
        S = self.prior_sd**2 * self.A @ self.A.T + self.noise_sd**2 * np.eye(self.m)
        return float(stats.multivariate_normal(mean=np.zeros(self.m), cov=S).logpdf(y))

    def exact_posterior(self, y=None):
        """(mean, covariance) of the Gaussian posterior of u."""
        y = np.asarray(self.data if y is None else y, dtype=np.float64)
        prec = self.A.T @ self.A / self.noise_sd**2 + np.eye(self.dim) / self.prior_sd**2
        cov = np.linalg.inv(prec)
        return cov @ (self.A.T @ y) / self.noise_sd**2, 0.5 * (cov + cov.T)


# ----------------------------------------------------------------------------------------------
# Evaluations on the host
# ----------------------------------------------------------------------------------------------


def potential(problem, u, beta=1.0, aux=None, data=None):
    """``(U_beta (float), grad U_beta (dim,) NumPy, aux_new)`` at u."""
    data = problem.data if data is None else data
    aux = problem.initial_aux() if aux is None else aux
    val, g, aux_new = _U_and_grad_fn(problem)(jnp.asarray(u, dtype=jnp.float64), aux, data, jnp.asarray(beta, dtype=jnp.float64))
    return float(val), np.asarray(g, dtype=np.float64), aux_new


def log_posterior(problem, u, beta=1.0, aux=None, data=None):
    """``(log pi_beta(u) + log Z-free constant, gradient)``: the unnormalized log posterior in the unconstrained coordinates, lik_const - beta phi(u) + log pi_u(u), with its
    gradient in u (one adjoint solve for the data term, the exact boundary derivatives and the analytic Jacobian of the parametrization for the rest). It is minus
    ``potential`` plus ``lik_const`` (beta = 1: log likelihood + log prior, the integrand of the evidence). Returns ``(float, (dim,) NumPy)``."""
    val, g, _ = potential(problem, u, beta, aux, data)
    return float(problem.lik_const - val), -g


def _phi_scan_fn(problem):
    """Jitted ``(rows (m, dim), aux, data) -> phi (m,)``: scan of ``problem.phi`` over the rows with the solver warm start carried along (cached on the problem)."""
    if getattr(problem, "_phi_scan", None) is None:
        phi_fn = problem.phi

        @jax.jit
        def scan_all(rows, aux, data):
            def body(a, row):
                phi, a = phi_fn(row, a, data)
                return a, phi

            _, phis = lax.scan(body, aux, rows)
            return phis

        problem._phi_scan = scan_all
    return problem._phi_scan


def phi_at_draws(problem, u_draws, data=None, aux=None):
    """The data misfit phi at every row of ``u_draws`` (..., dim): NumPy array of shape ``u_draws.shape[:-1]`` (one solve per draw, warm-started from the previous row)."""
    data = problem.data if data is None else data
    u = jnp.asarray(u_draws, dtype=jnp.float64)
    lead = u.shape[:-1]
    aux = problem.initial_aux() if aux is None else aux
    return np.asarray(_phi_scan_fn(problem)(u.reshape(-1, u.shape[-1]), aux, data)).reshape(lead)


# ----------------------------------------------------------------------------------------------
# MAP
# ----------------------------------------------------------------------------------------------


def _solve_spd(H, rhs, max_tries=8):
    """Solve H x = rhs for a symmetric positive (semi)definite H, adding a growing multiple of the mean diagonal on failure. Returns (x, jitter_used)."""
    H = 0.5 * (H + H.T)
    scale = float(np.mean(np.diag(H)))
    add = 0.0
    for k in range(max_tries):
        try:
            L = cholesky(H + add * np.eye(H.shape[0]), lower=True)
            return cho_solve((L, True), rhs), add
        except np.linalg.LinAlgError:
            add = max(add * 10.0, 1e-12 * scale * 10.0**k)
    raise np.linalg.LinAlgError("matrix is not positive definite even after jitter")


def newton_map(problem, u0, data=None, beta=1.0, max_iter=25, tol=1e-3, max_backtrack=14, c1=1e-4):
    """Minimize U_beta = beta phi - log pi_u from u0 by damped Gauss-Newton: step -(beta F(u) + P(u))^-1 grad U (F the Fisher matrix, P the prior Hessian: positive definite,
    so the step is a descent direction), halved until the Armijo condition holds. Stops when the Newton decrement g^T H^-1 g / 2 is below ``tol`` nats.

    Returns ``(u_hat (dim,), info)`` with ``info`` = ``converged``, ``nit``, ``U``, ``phi``, ``log_prior``, ``decrement``, ``grad_inf``, ``nfev`` (value-and-gradient calls), ``fisher_calls``.
    """
    data = problem.data if data is None else data
    u = np.asarray(u0, dtype=np.float64).copy()
    aux = problem.initial_aux()
    U, g, aux = potential(problem, u, beta, aux, data)
    nfev, nfish, converged, decrement, it = 1, 0, False, float("inf"), 0
    for it in range(int(max_iter)):
        H = beta * problem.fisher(u) + prior_hessian(problem, u)
        nfish += 1
        step, _ = _solve_spd(H, -g)
        slope = float(g @ step)
        decrement = -0.5 * slope
        if decrement < tol:
            converged = True
            break
        alpha, accepted = 1.0, False
        for _ in range(int(max_backtrack)):
            cand = u + alpha * step
            Uc, gc, auxc = potential(problem, cand, beta, aux, data)
            nfev += 1
            if np.isfinite(Uc) and Uc <= U + c1 * alpha * slope:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break
        u, U, g, aux = cand, Uc, gc, auxc
    phi = float(problem.phi(jnp.asarray(u), aux, data)[0])
    lp = float(problem.log_prior(jnp.asarray(u)))
    info = {"converged": bool(converged), "nit": it + (1 if converged else 0), "U": U, "phi": phi, "log_prior": lp, "decrement": float(decrement),
            "grad_inf": float(np.abs(g).max()), "nfev": nfev, "fisher_calls": nfish}
    return u, info


def lbfgs_map(problem, u0, data=None, beta=1.0, maxiter=300, gtol=1e-6, ftol=1e-13, maxcor=30):
    """Minimize U_beta by SciPy L-BFGS-B (host loop, jitted value-and-gradient, warm-started solves) from u0. Returns ``(u (dim,), info)``."""
    data = problem.data if data is None else data
    state = {"aux": problem.initial_aux()}
    history = []

    def fun(x):
        val, g, aux_new = potential(problem, x, beta, state["aux"], data)
        if np.isfinite(val):
            state["aux"] = aux_new
        history.append(val)
        return val, g

    res = minimize(fun, np.asarray(u0, dtype=np.float64), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": float(ftol), "maxcor": int(maxcor), "maxfun": 10 * int(maxiter) + 100})
    return np.asarray(res.x), {"success": bool(res.success), "message": str(res.message), "nit": int(res.nit), "nfev": int(res.nfev), "U": float(res.fun),
                               "U_history": np.asarray(history)}


@dataclass
class MapSearch:
    """Result of ``map_estimate``: every candidate (dict with ``label``, ``u``, ``U``, ``phi``, ``converged``, ``grad_inf``, ``lbfgs``, ``newton``, ``time_s``), the index of the
    best (lowest U among the converged, else among all) and ``u_hat``."""

    candidates: list
    best: int
    u_hat: np.ndarray = field(default=None)

    @property
    def best_candidate(self):
        return self.candidates[self.best]

    def summary(self):
        return [{k: v for k, v in c.items() if k not in ("u", "lbfgs", "newton")} | {"u": c["u"].tolist(), "lbfgs_nit": c["lbfgs"]["nit"], "newton_nit": c["newton"]["nit"]}
                for c in self.candidates]


def map_estimate(problem, starts, labels=None, data=None, beta=1.0, lbfgs_iter=300, polish_iter=25, tol=1e-3, verbose=False):
    """The MAP (mode of pi_beta in u) from several starts: L-BFGS-B (``lbfgs_iter`` iterations) then ``newton_map`` (``polish_iter`` iterations) from each start.

    ``starts`` (K, dim). U is not convex in u for the full forward model, so the start matters; the candidates are returned so that distinct local modes can be inspected
    (``detect_multimodality``). A candidate is "converged" if its Newton decrement is below ``tol`` nats. Returns a ``MapSearch``.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=np.float64))
    labels = [f"start{k}" for k in range(len(starts))] if labels is None else list(labels)
    cands = []
    for label, u0 in zip(labels, starts):
        t0 = time.perf_counter()
        u1, li = lbfgs_map(problem, u0, data, beta, maxiter=lbfgs_iter)
        u2, ni = newton_map(problem, u1, data, beta, max_iter=polish_iter, tol=tol)
        cands.append({"label": label, "u": u2, "U": ni["U"], "phi": ni["phi"], "converged": ni["converged"], "grad_inf": ni["grad_inf"], "lbfgs": li, "newton": ni,
                      "start_u": u0, "time_s": time.perf_counter() - t0})
        if verbose:
            print(f"  MAP start {label}: U {ni['U']:.3f} phi {ni['phi']:.2f} converged {ni['converged']} (L-BFGS {li['nit']} its, Newton {ni['nit']} its, {cands[-1]['time_s']:.0f} s)")
    ok = [i for i, c in enumerate(cands) if c["converged"]]
    pool = ok if ok else list(range(len(cands)))
    best = min(pool, key=lambda i: cands[i]["U"])
    return MapSearch(cands, best, cands[best]["u"])


# ----------------------------------------------------------------------------------------------
# Laplace approximation and evidence
# ----------------------------------------------------------------------------------------------


@dataclass
class LaplaceFit:
    """Gaussian approximation N(u_hat, H^-1) of pi_beta at its mode.

    ``H`` (dim, dim) = beta F(u_hat) + P(u_hat) (``hessian`` = "gn") or the finite-difference Hessian of the exact gradient of U_beta (``"fd"``); ``chol`` its lower Cholesky factor;
    ``cov`` = H^-1; ``logdet`` = log det H; ``U`` = U_beta(u_hat); ``phi``; ``log_prior``; ``fisher``, ``prior_hessian`` the two parts at u_hat. ``beta``.
    """

    u_hat: np.ndarray
    H: np.ndarray
    chol: np.ndarray
    cov: np.ndarray
    logdet: float
    U: float
    phi: float
    log_prior: float
    fisher: np.ndarray
    prior_hessian: np.ndarray
    hessian: str
    beta: float

    def sd(self):
        return np.sqrt(np.diag(self.cov))

    def affine_map(self):
        """``adapt.AffineMap`` with center u_hat and covariance H^-1: u = u_hat + C zeta, whitening the Laplace posterior."""
        return AffineMap.from_covariance(self.u_hat, self.cov)


def fd_hessian(problem, u, data=None, beta=1.0, scale=None, rel_step=0.05):
    """Central-difference Hessian of U_beta at u from the exact gradient: column k = (g(u + h_k e_k) - g(u - h_k e_k)) / (2 h_k), symmetrized. ``scale`` (dim,) gives the
    step h_k = rel_step * scale_k (default 1)."""
    data = problem.data if data is None else data
    u = np.asarray(u, dtype=np.float64)
    d = u.size
    scale = np.ones(d) if scale is None else np.asarray(scale, dtype=np.float64)
    H = np.zeros((d, d))
    for k in range(d):
        h = rel_step * scale[k]
        e = np.zeros(d)
        e[k] = h
        _, gp, _ = potential(problem, u + e, beta, None, data)
        _, gm, _ = potential(problem, u - e, beta, None, data)
        H[:, k] = (gp - gm) / (2.0 * h)
    return 0.5 * (H + H.T)


def laplace_fit(problem, u_hat, data=None, beta=1.0, hessian="gn"):
    """``LaplaceFit`` at the mode ``u_hat``. ``hessian`` "gn" (Fisher + prior Hessian, positive definite) or "fd" (finite differences of the exact gradient, which includes the
    residual-curvature term that Gauss-Newton drops: the right Hessian for the evidence of a misfitting model; raises ``ValueError`` if it is not positive definite)."""
    data = problem.data if data is None else data
    u_hat = np.asarray(u_hat, dtype=np.float64)
    F = problem.fisher(u_hat)
    P = prior_hessian(problem, u_hat)
    H = beta * F + P
    if hessian == "fd":
        sd = np.sqrt(np.diag(np.linalg.inv(H)))
        H = fd_hessian(problem, u_hat, data, beta, scale=sd)
    elif hessian != "gn":
        raise ValueError("hessian must be 'gn' or 'fd'")
    H = 0.5 * (H + H.T)
    try:
        L = cholesky(H, lower=True)
    except np.linalg.LinAlgError as err:
        raise ValueError(f"the {hessian} Hessian at the mode is not positive definite ({err}): not a minimum of U") from err
    cov = cho_solve((L, True), np.eye(H.shape[0]))
    phi = float(problem.phi(jnp.asarray(u_hat), problem.initial_aux(), data)[0])
    lp = float(problem.log_prior(jnp.asarray(u_hat)))
    return LaplaceFit(u_hat, H, L, 0.5 * (cov + cov.T), float(2.0 * np.sum(np.log(np.diag(L)))), beta * phi - lp, phi, lp, F, P, hessian, float(beta))


def laplace_log_evidence(problem, fit):
    """Laplace log evidence log Z = lik_const - U + (dim / 2) log(2 pi) - (1/2) log det H, with U = phi - log pi_u at the mode and H the Hessian of U (``fit``; beta = 1).

    The approximation is for the density of u, so it depends on the parametrization (the sigmoid / logit maps of ``shapemodels``); it is exact for a Gaussian posterior
    in u. Thermodynamic integration (``thermodynamic_integration``) checks it.
    """
    if fit.beta != 1.0:
        raise ValueError("the evidence is the integral of the posterior at beta = 1")
    return float(problem.lik_const - fit.U + 0.5 * fit.u_hat.size * LOG_2PI - 0.5 * fit.logdet)


# ----------------------------------------------------------------------------------------------
# HMC in the coordinates of an affine map
# ----------------------------------------------------------------------------------------------


def _zeta_phi_fn(problem):
    """The ``samplers.hmc`` potential ``(zeta, aux) -> (Phi', grad Phi', aux)`` for the coordinates u = center + C zeta; ``aux`` = (solver warm start, beta, center, C, data) is
    traced, so ONE compilation of ``samplers.hmc`` serves every temperature, map and data set of a problem. Phi'(zeta) = U_beta(u(zeta)) - |zeta|^2 / 2, so the sampler's target
    exp(-Phi' - |zeta|^2 / 2) is exp(-U_beta(u(zeta))) for any fixed C (constant Jacobian)."""
    if getattr(problem, "_zeta_phi_fn", None) is not None:
        return problem._zeta_phi_fn
    U_and_grad = _U_and_grad_fn(problem)

    def phi_fn(zeta, aux):
        inner, beta, center, chol, data = aux
        u = center + chol @ zeta
        val, g, inner = U_and_grad(u, inner, data, beta)
        return val - 0.5 * jnp.sum(zeta * zeta), chol.T @ g - zeta, (inner, beta, center, chol, data)

    problem._zeta_phi_fn = phi_fn
    return phi_fn


@dataclass
class ChainBatch:
    """Chains of ``run_hmc`` (C chains, n recorded draws, dim parameters).

    ``zeta`` (C, n, dim) whitened coordinates, ``u`` (C, n, dim) = center + C zeta, ``phi_prime`` (C, n) the sampler's potential after each step, ``accept`` (C, n) bool,
    ``step`` (C,) frozen leapfrog step sizes, ``final_zeta`` (C, dim), ``final_aux`` list of C solver warm starts, ``amap`` (the ``AffineMap``), ``beta``, ``n_leapfrog``, ``jitter``,
    ``n_grad`` the number of gradient evaluations per chain (warmup + recorded), ``time_s``. ``phi`` (C, n) is the data misfit at the draws (``add_phi``), None until then.
    """

    zeta: np.ndarray
    u: np.ndarray
    phi_prime: np.ndarray
    accept: np.ndarray
    step: np.ndarray
    final_zeta: np.ndarray
    final_aux: list
    amap: AffineMap
    beta: float
    n_leapfrog: int
    jitter: float
    n_grad: int
    time_s: float
    phi: np.ndarray = None

    def add_phi(self, problem, data=None):
        """Compute the data misfit at every draw (one forward solve each); sets and returns ``phi`` (C, n)."""
        self.phi = np.stack([phi_at_draws(problem, self.u[c], data) for c in range(self.u.shape[0])])
        return self.phi

    @property
    def acceptance(self):
        return float(self.accept.mean())


def run_hmc(problem, amap, zeta0, key, *, beta=1.0, n_warmup=100, n_samples=200, step0=1.0, n_leapfrog=8, jitter=0.25, target_accept=0.7, data=None, aux0=None):
    """HMC chains for pi_beta in the coordinates of ``amap`` (an ``AffineMap``: u = center + chol zeta), one chain per row of ``zeta0`` (C, dim).

    Each chain runs ``samplers.hmc`` (fixed ``n_leapfrog`` steps with the step jittered by +-``jitter``, Robbins-Monro step-size adaptation toward ``target_accept`` during
    the ``n_warmup`` steps, then the step is frozen) and records every step. ``step0`` is the initial step (1 is natural for a whitened coordinate); to continue a chain
    from a previous segment pass its ``final_zeta`` as ``zeta0``, its ``final_aux`` as ``aux0`` (list), ``n_warmup=0`` and its step as ``step0``. ``key`` is split over the chains.
    The compiled sampler is shared as long as ``n_warmup``, ``n_samples``, ``n_leapfrog`` and ``jitter`` are unchanged.
    """
    data = problem.data if data is None else data
    zeta0 = np.atleast_2d(np.asarray(zeta0, dtype=np.float64))
    n_chains = zeta0.shape[0]
    keys = jax.random.split(key, n_chains)
    phi_fn = _zeta_phi_fn(problem)
    center, chol = jnp.asarray(amap.center_np.reshape(-1)), jnp.asarray(amap.chol_np)
    beta_j = jnp.asarray(beta, dtype=jnp.float64)
    zeta, phip, acc, steps, finals, auxs = [], [], [], [], [], []
    t0 = time.perf_counter()
    for c in range(n_chains):
        inner = problem.initial_aux() if aux0 is None else aux0[c]
        step_c = float(step0[c]) if np.ndim(step0) else float(step0)
        res = sm.hmc(keys[c], phi_fn, jnp.asarray(zeta0[c]), (inner, beta_j, center, chol, data), n_warmup=int(n_warmup), n_samples=int(n_samples), step0=step_c,
                     thin=1, target_accept=float(target_accept), n_leapfrog=int(n_leapfrog), jitter=float(jitter), record_fn=sm.identity)
        zeta.append(np.asarray(res.samples))
        phip.append(np.asarray(res.phi))
        acc.append(np.asarray(res.accept))
        steps.append(float(res.step))
        finals.append(np.asarray(res.xi))
        auxs.append(res.aux[0])
    zeta = np.stack(zeta)
    u = np.asarray(amap.center_np.reshape(-1)) + zeta @ np.asarray(amap.chol_np).T
    return ChainBatch(zeta, u, np.stack(phip), np.stack(acc), np.asarray(steps), np.stack(finals), auxs, amap, float(beta), int(n_leapfrog), float(jitter),
                      (int(n_warmup) + int(n_samples)) * int(n_leapfrog), time.perf_counter() - t0)


def overdispersed_starts(amap, key, n_chains, scale=2.0):
    """``n_chains`` starts zeta = scale * N(0, I): overdispersed draws around the center of the map (DESIGN 11.2: scale 2), (n_chains, dim)."""
    return scale * np.asarray(jax.random.normal(key, (n_chains, amap.size), dtype=jnp.float64))


def convergence_report(phi, u, p=None, names=None, rhat_phi_max=1.01, rhat_param_max=1.05, ess_min=200.0):
    """Convergence numbers of DESIGN 12.2 for C chains of n draws: ``phi`` (C, n), ``u`` (C, n, dim), optionally physical parameters ``p`` (C, n, k).

    Reported: the classic split R-hat of phi (``rhat_phi``) and its rank-normalized bulk and folded versions (``rhat_phi_rank``, the worst of the two), the maximum classic split R-hat
    over the u (and p) coordinates, the worst rank-normalized R-hat, the worst multi-chain bulk ESS (rank-normalized) and the worst summed Sokal ESS, and the flag
    ``converged`` = classic and rank R-hat of phi <= ``rhat_phi_max``, max R-hat of the coordinates <= ``rhat_param_max`` and worst bulk ESS >= ``ess_min``.
    """
    phi, u = np.asarray(phi, dtype=np.float64), np.asarray(u, dtype=np.float64)
    out = {}
    x = phi[..., None]
    out["rhat_phi"] = float(dg.split_rhat(x)[0])
    out["rhat_phi_rank"] = float(dg.split_rhat_rank(x)[2][0])
    out["ess_bulk_phi"] = float(dg.ess_bulk(x)[0])
    sets = {"u": (u, [f"u{k}" for k in range(u.shape[2])])}
    if p is not None:
        p = np.asarray(p, dtype=np.float64)
        sets["p"] = (p, list(names) if names is not None else [f"p{k}" for k in range(p.shape[2])])
    worst_ess, worst_rhat = np.inf, 0.0
    for tag, (arr, nm) in sets.items():
        rh = dg.split_rhat(arr)
        rk = dg.split_rhat_rank(arr)[2]
        eb = dg.ess_bulk(arr)
        sokal = np.array([sum(float(dg.ess(arr[c, :, k])) for c in range(arr.shape[0])) for k in range(arr.shape[2])])
        out[tag] = {"names": nm, "rhat": rh.tolist(), "rhat_rank": rk.tolist(), "ess_bulk": eb.tolist(), "ess_sokal_sum": sokal.tolist()}
        worst_rhat = max(worst_rhat, float(np.nanmax(rh)))
        worst_ess = min(worst_ess, float(np.nanmin(eb)))
    out["max_rhat"] = worst_rhat
    out["max_rhat_rank"] = float(max(np.nanmax(out[t]["rhat_rank"]) for t in sets))
    out["worst_ess_bulk"] = worst_ess
    out["worst_ess_sokal_sum"] = float(min(np.nanmin(out[t]["ess_sokal_sum"]) for t in sets))
    out["thresholds"] = {"rhat_phi": rhat_phi_max, "rhat_param": rhat_param_max, "ess": ess_min}
    out["converged"] = bool(out["rhat_phi"] <= rhat_phi_max and out["rhat_phi_rank"] <= rhat_phi_max and worst_rhat <= rhat_param_max and worst_ess >= ess_min)
    return out


@dataclass
class PosteriorSamples:
    """Output of ``sample_posterior``: the ``ChainBatch`` (with ``phi`` set), the physical draws ``p`` (C, n, k) and the convergence report (``convergence_report``)."""

    batch: ChainBatch
    p: np.ndarray
    convergence: dict


def physical_draws(problem, u):
    """Physical parameters at every u (..., dim) -> (..., k) NumPy (``problem.physical_many``; u itself for a problem without a model)."""
    u = np.asarray(u, dtype=np.float64)
    many = getattr(problem, "physical_many", None)
    if many is None:
        return u
    return np.asarray(many(u.reshape(-1, u.shape[-1]))).reshape(u.shape[:-1] + (-1,))


def sample_posterior(problem, fit, key, *, n_chains=4, n_warmup=100, n_samples=300, init_scale=2.0, n_leapfrog=8, jitter=0.25, starts_u=None, data=None, amap=None, step0=1.0):
    """HMC for the posterior (beta = 1) in the Laplace coordinates of ``fit`` (or of ``amap``): ``n_chains`` chains from overdispersed starts (``init_scale`` x N(0, I) around the
    mode, or the rows of ``starts_u``), the data misfit of the draws, the physical draws and the convergence report. Returns ``PosteriorSamples``."""
    k_start, k_run = jax.random.split(key)
    amap = fit.affine_map() if amap is None else amap
    if starts_u is None:
        zeta0 = overdispersed_starts(amap, k_start, n_chains, init_scale)
    else:
        starts_u = np.atleast_2d(np.asarray(starts_u, dtype=np.float64))
        zeta0 = np.stack([np.linalg.solve(amap.chol_np, s - amap.center_np.reshape(-1)) for s in starts_u])
    batch = run_hmc(problem, amap, zeta0, k_run, beta=1.0, n_warmup=n_warmup, n_samples=n_samples, step0=step0, n_leapfrog=n_leapfrog, jitter=jitter, data=data)
    batch.add_phi(problem, data)
    p = physical_draws(problem, batch.u)
    conv = convergence_report(batch.phi, batch.u, p, names=getattr(problem, "names", None))
    return PosteriorSamples(batch, p, conv)


def credible_interval(x, level=0.9):
    """Equal-tailed credible interval of the pooled draws ``x`` (any shape): ``(lo, hi)``."""
    a = 0.5 * (1.0 - level)
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


def gap_summary(g_draws, g_true=None, g_det=None, level=0.9):
    """DESIGN 14.4 summary of the gap draws (pooled): mean, sd, ``level`` credible interval, whether it covers ``g_true``, P(g < g_det)."""
    g = np.asarray(g_draws, dtype=np.float64).ravel()
    lo, hi = credible_interval(g, level)
    out = {"mean": float(g.mean()), "sd": float(g.std(ddof=1)), "median": float(np.median(g)), "ci_level": level, "ci": [lo, hi], "n": int(g.size)}
    if g_true is not None:
        out["g_true"] = float(g_true)
        out["covers_true"] = bool(lo <= g_true <= hi)
        out["z_true"] = float((g_true - g.mean()) / g.std(ddof=1))
    if g_det is not None:
        out["g_det"] = float(g_det)
        out["p_below_g_det"] = float(np.mean(g < g_det))
    return out


# ----------------------------------------------------------------------------------------------
# Multimodality
# ----------------------------------------------------------------------------------------------


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient (skew^2 + 1) / (excess kurtosis + 3 (n - 1)^2 / ((n - 2)(n - 3))) of a 1-D sample: 1/3 for a Gaussian, above 5/9 for a
    bimodal (or uniform, or strongly skewed) sample. A heuristic; ``detect_multimodality`` reports it and does not decide on it alone."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 8 or np.ptp(x) == 0.0:
        return float("nan")
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, bias=False)
    return float((skew**2 + 1.0) / (kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def chain_separation(u):
    """Largest Mahalanobis distance between two chain means in units of the pooled within-chain covariance W: max over pairs of sqrt((m_c - m_c')^T W^-1 (m_c - m_c')).
    Chains from one unimodal posterior give much less than 1 (the means differ by sd / sqrt(ESS)); chains in different modes give much more. ``u`` (C, n, dim)."""
    u = np.asarray(u, dtype=np.float64)
    C, n, d = u.shape
    means = u.mean(axis=1)
    W = np.mean([np.cov(u[c].T).reshape(d, d) for c in range(C)], axis=0)
    W = 0.5 * (W + W.T) + 1e-14 * np.trace(W) / d * np.eye(d)
    Winv = np.linalg.inv(W)
    worst = 0.0
    for a in range(C):
        for b in range(a + 1, C):
            diff = means[a] - means[b]
            worst = max(worst, float(np.sqrt(max(diff @ Winv @ diff, 0.0))))
    return worst


def detect_multimodality(problem, search, fit, batch=None, names=None, mode_distance=8.0, weight_floor=1e-3, separation_max=1.0, bimodality_flag=0.6):
    """Look for a second mode and say so (DESIGN 14.2: multimodal targets are detected and reported, not averaged over).

    Three independent signals:

    1. **Distinct MAP modes.** The converged candidates of ``map_estimate`` (several starts, including the symmetry images of the model) are clustered: two candidates are the
       same mode if their Mahalanobis distance under the Laplace covariance of ``fit`` is below ``mode_distance``. Each distinct mode gets its own Laplace weight
       log w = -U - (1/2) log det H (GN Hessian at that mode) and ``relative_weight`` = w / w_best; a distinct mode with relative weight above ``weight_floor`` makes the target
       multimodal.
    2. **Chain separation** (``batch`` = a ``ChainBatch``): ``chain_separation`` of the u draws above ``separation_max`` within-chain standard deviations.
    3. **Bimodality coefficient** (``bimodality_coefficient``) of every coordinate of the pooled u draws, reported and flagged above ``bimodality_flag`` (informational: skewed
       unimodal posteriors, such as a gap pressed against g = 0, also exceed 5/9).

    Returns a dict: ``multimodal`` (signal 1 or 2), ``reasons``, ``modes`` (list with ``u``, ``U``, ``relative_weight``, ``distance``, ``label``), ``chain_separation``,
    ``bimodality`` (list, ``flagged`` names).
    """
    u_best = fit.u_hat
    modes = [{"label": search.best_candidate["label"], "u": u_best.tolist(), "U": fit.U, "distance": 0.0, "logw": float(-fit.U - 0.5 * fit.logdet), "relative_weight": 1.0}]
    reasons = []
    for i, c in enumerate(search.candidates):
        if i == search.best or not c["converged"]:
            continue
        dist = float(np.linalg.norm(fit.chol.T @ (c["u"] - u_best)))  # sqrt((u - u_hat)^T H (u - u_hat)), H = L L^T
        known = [m for m in modes if np.linalg.norm(fit.chol.T @ (c["u"] - np.asarray(m["u"]))) < mode_distance]
        if dist < mode_distance or known:
            continue
        try:
            fit_m = laplace_fit(problem, c["u"])
            logw = float(-fit_m.U - 0.5 * fit_m.logdet)
        except ValueError:
            logw = float(-c["U"])
        modes.append({"label": c["label"], "u": c["u"].tolist(), "U": c["U"], "distance": dist, "logw": logw, "relative_weight": float(math.exp(min(logw - modes[0]["logw"], 700.0)))})
    for m in modes[1:]:
        if m["relative_weight"] > weight_floor:
            reasons.append(f"distinct mode '{m['label']}' at {m['distance']:.1f} posterior sd with relative Laplace weight {m['relative_weight']:.3g}")
    sep = None
    if batch is not None and batch.u.shape[0] > 1:
        sep = chain_separation(batch.u)
        if sep > separation_max:
            reasons.append(f"chain means separated by {sep:.2f} within-chain standard deviations (> {separation_max})")
    bim = None
    if batch is not None:
        pooled = batch.u.reshape(-1, batch.u.shape[-1])
        names_u = list(names) if names is not None else [f"u{k}" for k in range(pooled.shape[1])]
        coeffs = [bimodality_coefficient(pooled[:, k]) for k in range(pooled.shape[1])]
        bim = {"names": names_u, "coefficient": coeffs, "flagged": [n for n, b in zip(names_u, coeffs) if np.isfinite(b) and b > bimodality_flag], "threshold": bimodality_flag}
    return {"multimodal": bool(reasons), "reasons": reasons, "modes": modes, "chain_separation": sep, "bimodality": bim, "thresholds":
            {"mode_distance": mode_distance, "weight_floor": weight_floor, "separation_max": separation_max}}


# ----------------------------------------------------------------------------------------------
# Thermodynamic integration
# ----------------------------------------------------------------------------------------------


def ti_schedule(n_temps, power=5.0):
    """The temperatures beta_k = (k / K)^power, k = 0..K with K = n_temps - 1 (so ``n_temps`` values; the design uses power 5 and at least 12 temperatures)."""
    if n_temps < 3:
        raise ValueError("need at least 3 temperatures")
    k = np.arange(n_temps, dtype=np.float64)
    return (k / (n_temps - 1)) ** power


def prior_map(problem):
    """Affine map of the prior-scale preconditioner at u = 0: center 0, covariance P(0)^-1 (the prior Hessian)."""
    d = problem.dim
    P = prior_hessian(problem, np.zeros(d))
    cov, _ = _solve_spd(P, np.eye(d))
    return AffineMap.from_covariance(np.zeros(d), 0.5 * (cov + cov.T))


def adapt_map_from_draws(draws, prev_map, kappa=None):
    """A new map from pilot ``draws`` (m, dim): center their mean, covariance (1 - w) S + w Sigma_prev with w = kappa / (m + kappa), kappa = 2 dim by default (a small pseudo-count: the
    pilot of a low-dimensional problem determines S well). Used in warmup only; the recorded chain runs with the map frozen (an exact HMC for that map)."""
    draws = np.asarray(draws, dtype=np.float64)
    m, d = draws.shape
    kappa = 2.0 * d if kappa is None else kappa
    w = kappa / (m + kappa)
    S = np.cov(draws.T).reshape(d, d) if m > 1 else prev_map.covariance
    cov = (1.0 - w) * S + w * prev_map.covariance
    return AffineMap.from_covariance(draws.mean(axis=0), cov)


@dataclass
class TIResult:
    """One temperature of thermodynamic integration: ``beta``, ``phi`` (C, n) misfit draws, ``u`` (C, n, dim), the ``ChainBatch`` (frozen map and step, final states), the
    chain statistics ``mean_phi``, ``var_phi``, ``ess_phi`` (bulk, raw), ``rhat_phi``, ``rhat_phi_rank``, ``acceptance``, ``step_mean``, and the cost."""

    beta: float
    phi: np.ndarray
    u: np.ndarray
    batch: ChainBatch
    stats: dict


def ti_temperature(problem, beta, u_starts, amap_prev, key, *, n_pilot=40, n_warmup=60, n_samples=150, n_leapfrog=8, jitter=0.25, data=None, aux0=None, adapt=True, step0=1.0):
    """One power posterior pi_beta: pilot (map adaptation), step-size warmup, recorded draws, and the statistics TI needs.

    Starting from ``u_starts`` (C, dim) (the final states at the previous temperature, in u coordinates) the chains first run ``n_pilot`` steps (after ``n_pilot`` of step-size
    warmup) in the previous map ``amap_prev``; the pooled pilot draws define the new map (``adapt_map_from_draws``, skipped if ``adapt`` is False); then ``n_warmup`` steps adapt
    the step size with the map frozen, and ``n_samples`` steps are recorded (HMC for the fixed map: exact). The data misfit of the recorded draws is computed afterwards.
    """
    k1, k2 = jax.random.split(key)
    u_starts = np.atleast_2d(np.asarray(u_starts, dtype=np.float64))
    amap = amap_prev
    zeta0 = np.stack([np.linalg.solve(amap.chol_np, s - amap.center_np.reshape(-1)) for s in u_starts])
    pilot_cost = 0
    if adapt and n_pilot > 0:
        pilot = run_hmc(problem, amap, zeta0, k1, beta=beta, n_warmup=n_pilot, n_samples=n_pilot, step0=step0, n_leapfrog=n_leapfrog, jitter=jitter, data=data, aux0=aux0)
        amap = adapt_map_from_draws(pilot.u.reshape(-1, pilot.u.shape[-1]), amap_prev)
        zeta0 = np.stack([np.linalg.solve(amap.chol_np, pilot.u[c, -1] - amap.center_np.reshape(-1)) for c in range(pilot.u.shape[0])])
        aux0 = pilot.final_aux
        step0 = float(np.mean(pilot.step))
        pilot_cost = pilot.n_grad
    batch = run_hmc(problem, amap, zeta0, k2, beta=beta, n_warmup=n_warmup, n_samples=n_samples, step0=step0, n_leapfrog=n_leapfrog, jitter=jitter, data=data, aux0=aux0)
    batch.add_phi(problem, data)
    phi = batch.phi
    x = phi[..., None]
    n_total = phi.size
    ess_raw = float(dg.ess_bulk(x, rank=False)[0]) if phi.shape[1] >= 8 else float("nan")
    stats_ = {"beta": float(beta), "mean_phi": float(phi.mean()), "var_phi": float(phi.var(ddof=1)), "ess_phi": ess_raw, "n_draws": int(n_total),
              "rhat_phi": float(dg.split_rhat(x)[0]) if phi.shape[1] >= 4 else float("nan"),
              "rhat_phi_rank": float(dg.split_rhat_rank(x)[2][0]) if phi.shape[1] >= 4 else float("nan"),
              "acceptance": batch.acceptance, "step_mean": float(batch.step.mean()), "n_grad_per_chain": batch.n_grad + pilot_cost, "time_s": batch.time_s,
              "u_mean": batch.u.reshape(-1, batch.u.shape[-1]).mean(axis=0).tolist(), "adapted": bool(adapt and n_pilot > 0)}
    return TIResult(float(beta), phi, batch.u, batch, stats_)


def ti_combine(problem, temps, betas=None):
    """Combine the temperatures into the TI estimate of log Z and its diagnostics.

    ``temps``: list of ``TIResult`` or of their ``stats`` dicts (keys ``beta``, ``mean_phi``, ``var_phi``, ``ess_phi``, ``rhat_phi``, ``rhat_phi_rank``, ``acceptance``), ordered
    by beta from 0 to 1. With f_k = lik_const - E_k[phi] (the expected log-likelihood at beta_k) log Z = integral of f over [0, 1], and f'(beta) = Var_beta[phi] >= 0.

    * ``log_z_trapezoid``: the trapezoid rule on the schedule; ``log_z``: with the end correction -(Delta^2 / 12)(f'_(k+1) - f'_k) of each interval (Euler-Maclaurin; Friel, Hurn and
      Wyse 2014), which uses the sample variances; the reported estimate is the corrected one.
    * ``mc_se``: Monte Carlo standard error, f_k with variance Var_k / ESS_k (bulk ESS of phi, raw) and the variance estimates with variance 2 Var_k^2 / ESS_k (Gaussian
      approximation), propagated through the quadrature weights, independent temperatures.
    * ``discretization``: |trapezoid - corrected| and the trapezoid on every second temperature (the coarse rule), whose difference to the fine trapezoid over 3 estimates the
      remaining error of the fine rule (Richardson, interval error O(Delta^2)).
    * ``monotone``: E_k[phi] must not increase with beta (f' >= 0); violations larger than 2 standard errors are listed.
    * ``worst_rhat``, ``min_ess``, ``acceptance_range`` and ``converged`` (every temperature: split R-hat of phi <= 1.05 and rank R-hat <= 1.05, ESS >= 20, acceptance in [0.3, 0.99];
      deliberately looser than the posterior criteria since a bias of E_beta[phi] is what matters here and the Monte Carlo error is propagated).
    """
    st = [t.stats if isinstance(t, TIResult) else t for t in temps]
    beta = np.array([s["beta"] for s in st]) if betas is None else np.asarray(betas, dtype=np.float64)
    if beta[0] != 0.0 or beta[-1] != 1.0 or np.any(np.diff(beta) <= 0):
        raise ValueError("temperatures must increase from 0 to 1")
    phi_mean = np.array([s["mean_phi"] for s in st])
    var = np.array([s["var_phi"] for s in st])
    ess = np.array([s["ess_phi"] for s in st])
    f = problem.lik_const - phi_mean
    df = np.diff(beta)
    w = np.zeros_like(beta)
    w[:-1] += 0.5 * df
    w[1:] += 0.5 * df
    trap = float(np.sum(w * f))
    corr_terms = (df**2 / 12.0) * (var[1:] - var[:-1])
    corrected = float(trap - np.sum(corr_terms))
    var_f = var / ess
    cvec = np.zeros_like(beta)
    cvec[1:] += df**2 / 12.0
    cvec[:-1] -= df**2 / 12.0
    var_var = 2.0 * var**2 / ess
    mc_se = float(np.sqrt(np.sum(w**2 * var_f) + np.sum(cvec**2 * var_var)))
    idx = np.arange(0, len(beta), 2)
    if idx[-1] != len(beta) - 1:
        idx = np.append(idx, len(beta) - 1)
    bc, fc = beta[idx], f[idx]
    coarse = float(np.sum(0.5 * np.diff(bc) * (fc[1:] + fc[:-1])))
    se_k = np.sqrt(var_f)
    viol = [int(k) for k in range(len(beta) - 1) if phi_mean[k + 1] - phi_mean[k] > 2.0 * np.hypot(se_k[k], se_k[k + 1])]
    rh = np.array([s["rhat_phi"] for s in st])
    rr = np.array([s.get("rhat_phi_rank", np.nan) for s in st])
    acc = np.array([s["acceptance"] for s in st])
    ok = bool(np.all(rh[1:] <= 1.05) and np.all(rr[1:] <= 1.05) and np.all(ess >= 20.0) and np.all((acc > 0.3) & (acc < 0.99)))
    return {"betas": beta.tolist(), "n_temps": int(len(beta)), "mean_phi": phi_mean.tolist(), "var_phi": var.tolist(), "ess_phi": ess.tolist(), "f": f.tolist(),
            "log_z": corrected, "log_z_trapezoid": trap, "mc_se": mc_se,
            "discretization": {"corrected_minus_trapezoid": corrected - trap, "coarse_trapezoid": coarse, "fine_minus_coarse": trap - coarse, "richardson_error_estimate": (trap - coarse) / 3.0},
            "monotone": {"violations_at": viol, "ok": not viol}, "worst_rhat": float(np.nanmax(rh[1:])), "worst_rhat_rank": float(np.nanmax(rr[1:])), "min_ess": float(np.nanmin(ess)),
            "acceptance_range": [float(acc.min()), float(acc.max())], "converged": ok, "lik_const": float(problem.lik_const)}


def thermodynamic_integration(problem, key, *, n_temps=12, power=5.0, n_chains=4, n_pilot=40, n_warmup=60, n_samples=150, n_leapfrog=8, jitter=0.25, data=None, prior_starts=None,
                              callback=None, rng_seed=0):
    """Log evidence by thermodynamic integration over beta_k = (k / K)^power (``n_temps`` values), sequentially from the prior (beta = 0) to the posterior (beta = 1).

    At beta = 0 the chains start from i.i.d. prior draws (``prior_starts`` (C, dim), or drawn from ``problem.model.sample_prior_p``; for a problem without a model they must be given)
    in the prior-scale map; at every later temperature from the final states of the previous one, with the previous map as the starting point of the adaptation
    (``ti_temperature``). The schedule needs no MAP and no Laplace fit, which is the point of an evidence estimate that checks the Laplace one.
    ``callback(k, TIResult)`` is called after each temperature (checkpointing hook).
    Returns ``(ti_combine(...) dict, list of TIResult)``.
    """
    betas = ti_schedule(n_temps, power)
    keys = jax.random.split(key, n_temps)
    if prior_starts is None:
        model = getattr(problem, "model", None)
        if model is None:
            raise ValueError("a problem without a model needs explicit prior_starts")
        rng = np.random.default_rng(rng_seed)
        prior_starts = np.asarray(jax.vmap(model.u_from_p)(jnp.asarray(model.sample_prior_p(rng, n_chains))))
    amap = prior_map(problem)
    starts = np.asarray(prior_starts, dtype=np.float64)
    results = []
    for k, beta in enumerate(betas):
        res = ti_temperature(problem, beta, starts, amap, keys[k], n_pilot=n_pilot if k > 0 else 0, n_warmup=n_warmup, n_samples=n_samples, n_leapfrog=n_leapfrog,
                             jitter=jitter, data=data, adapt=k > 0)
        results.append(res)
        if callback is not None:
            callback(k, res)
        starts = res.u[:, -1, :]
        amap = res.batch.amap
    return ti_combine(problem, results), results


# ----------------------------------------------------------------------------------------------
# Posterior-predictive check
# ----------------------------------------------------------------------------------------------


def predictive_check(problem, u_draws, key, *, n_rep=100, mode="refit", u_hat=None, data=None, max_iter=12):
    """Posterior-predictive check of the data misfit (DESIGN 14.4): replicate the data at posterior draws with fresh noise and compare with the observed misfit.

    ``u_draws`` (n, dim) posterior draws (pooled over chains); ``n_rep`` of them (evenly spaced) each generate one replicate ``problem.simulate(u_j, key_j)``.

    * ``mode="refit"`` (default): the statistic is the best-fit misfit, T = phi(u_hat(y); y) with u_hat(y) the MAP for data y (``newton_map`` from the generating draw for a
      replicate and from ``u_hat`` for the observed data). If the model is right, replicate and observed statistics have the same law, so the p-value p = P(T_rep >= T_obs) is
      uniform over data sets (tested). A model that cannot fit the data has T_obs far above T_rep and p near 0.
    * ``mode="draw"``: the classical Gelman-Meng-Stern statistic, the misfit AT the draw: T_obs,j = phi(u_j; y_obs), T_rep,j = phi(u_j; y_rep,j), p = mean_j [T_rep,j >= T_obs,j].
      The data are used twice (to form the posterior and to evaluate it), so for a right model p concentrates near 0.5; it is still near 0 for a model that cannot fit.

    Returns a dict: ``p_value``, ``mode``, ``t_obs`` (mean over draws for "draw"), ``t_rep_mean``, ``t_rep_sd``, ``t_rep`` (list), ``n_rep``, ``z`` = (t_obs - mean) / sd of the replicates.
    """
    data = problem.data if data is None else data
    u_draws = np.asarray(u_draws, dtype=np.float64)
    idx = np.unique(np.linspace(0, len(u_draws) - 1, int(n_rep)).astype(int))
    keys = jax.random.split(key, len(idx))
    t_rep = []
    aux0 = problem.initial_aux()
    if mode == "refit":
        if u_hat is None:
            raise ValueError("mode 'refit' needs u_hat (the MAP for the observed data)")
        u_obs, _ = newton_map(problem, u_hat, data, 1.0, max_iter=max_iter)
        t_obs = float(problem.phi(jnp.asarray(u_obs), aux0, data)[0])
        for j, kj in zip(idx, keys):
            y_rep = problem.simulate(u_draws[j], kj)
            u_rep, _ = newton_map(problem, u_draws[j], y_rep, 1.0, max_iter=max_iter)
            t_rep.append(float(problem.phi(jnp.asarray(u_rep), aux0, y_rep)[0]))
        t_rep = np.array(t_rep)
        p_value = float(np.mean(t_rep >= t_obs))
    elif mode == "draw":
        t_obs_each = []
        for j, kj in zip(idx, keys):
            y_rep = problem.simulate(u_draws[j], kj)
            t_rep.append(float(problem.phi(jnp.asarray(u_draws[j]), aux0, y_rep)[0]))
            t_obs_each.append(float(problem.phi(jnp.asarray(u_draws[j]), aux0, data)[0]))
        t_rep, t_obs_each = np.array(t_rep), np.array(t_obs_each)
        t_obs = float(t_obs_each.mean())
        p_value = float(np.mean(t_rep >= t_obs_each))
    else:
        raise ValueError("mode must be 'refit' or 'draw'")
    sd = float(t_rep.std(ddof=1)) if len(t_rep) > 1 else float("nan")
    return {"mode": mode, "p_value": p_value, "t_obs": t_obs, "t_rep_mean": float(t_rep.mean()), "t_rep_sd": sd, "z": float((t_obs - t_rep.mean()) / sd) if sd > 0 else float("nan"),
            "n_rep": int(len(t_rep)), "t_rep": t_rep.tolist()}
