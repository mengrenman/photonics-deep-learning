"""Warmup adaptation of the affine map of the theta-space HMC (docs/DESIGN.md section 12.2).

The chains of ``thetaspace`` run in coordinates theta = center + A zeta whose map is fixed; the milestone-4 map is the
Laplace map of the MAP (center = MAP, A A^T = H^-1). When the posterior is far from its Laplace approximation (the GMRF:
the local Hessian along the chains reached 270 times the Laplace scale) a fixed map forces a tiny step size and the chains
drift. This module re-estimates the map from warmup draws and then freezes it:

* ``AffineMap`` : theta = center + chol @ zeta, ``chol`` the lower Cholesky factor of the map covariance Sigma = chol chol^T
  (flat vectors of length N, or any array shape with N entries); ``to_theta``, ``to_zeta``, ``grad_to_zeta`` (= chol^T g), ``save`` /
  ``load`` (npz) and a SHA-256 ``digest``. The potential in zeta is Phi'(zeta) = U(theta(zeta)) - |zeta|^2 / 2 with gradient
  chol^T grad U - zeta (the protocol of ``samplers``: they add the |zeta|^2 / 2 back).
* ``adapt_map`` : the warmup. HMC with a *variable number of leapfrog steps* (``trajectory``: L uniform on
  {ceil(L0 / 2), ..., floor(3 L0 / 2)}, L0 = round(traj_length / eps), capped at ``max_leapfrog``), the step size eps by Nesterov dual
  averaging toward ``target_accept`` (0.8), restarted after every map update. The schedule has an initial step-size buffer, then
  windows whose length multiplies by ``growth`` (2: expanding windows), each of which ends with a map update from that window's
  draws only, then a final step-size buffer. ``update_map`` is the update: re-center at the window mean, sample covariance S of the
  window, shrink toward the Laplace covariance T at the new center, Sigma_hat = (1 - w) S + w T with w = kappa / (n + kappa)
  (the posterior mean under an inverse-Wishart prior centered at T and worth ``kappa`` draws; ``kappa`` = N by default, see
  ``AdaptConfig``), then the Cholesky factor. After the last window the map is frozen.
* ``sample_chain`` : the recorded segment: HMC for a fixed map, step size and trajectory length, Welford moments and thinned
  draws, as ``samplers`` but with the variable-length trajectories and a count of the gradient evaluations.

Why freezing matters. For any *fixed* affine map the chain is an exact HMC chain for the target (the Jacobian of an affine map
is constant and cancels), so the map, the step size and the trajectory length may be chosen however one likes before the first
recorded draw; a *state-dependent* update during recording is not a Markov chain with the target as invariant law and biases the
result (``tests/test_adapt.py`` shows it on a banana). Hence the map and the step size are frozen before recording,
and ``AdaptResult`` carries everything needed to reproduce a recorded segment: the map (center and Cholesky factor), eps, the
trajectory length and the adaptation history.

Everything here works on flat vectors of length N and a *potential function* ``potential_fn(theta, aux) -> (U, grad U, aux)`` of
the flat theta; ``aux`` is an arbitrary pytree (the forward-solver warm start).
"""

import functools
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular as jsolve_triangular
from scipy.linalg import cholesky, solve_triangular

from . import samplers as sm

LOG_EPS_MIN = float(np.log(1e-8))
LOG_EPS_MAX = float(np.log(4.0))  # a step above ~2 sigma_min is unstable; in whitened coordinates sigma_min ~ 1


# ----------------------------------------------------------------------------------------------
# The affine map
# ----------------------------------------------------------------------------------------------


class AffineMap:
    """theta = center + chol @ zeta with ``chol`` lower triangular (N, N), Sigma = chol chol^T the covariance of theta that the
    map whitens: if theta ~ N(center, Sigma) then zeta ~ N(0, I).

    ``center`` has the shape of theta (``shape``; N entries); zeta has the same shape. ``to_theta(zeta)`` = center + chol zeta,
    ``to_zeta(theta)`` = chol^-1 (theta - center) by a triangular solve, ``grad_to_zeta(g)`` = chol^T g, the chain rule of the
    affine map for the gradient of a function of theta. All three take/return JAX arrays of shape ``shape``; they call
    module-level jitted functions with the map as arguments, so a new map of the same size does not recompile anything.
    (``mrf.WhitenedMap`` is the map of a precision Cholesky factor, theta = center + L^-T zeta; this one stores a covariance
    factor so that a step costs two dense matrix-vector products instead of two triangular solves.)
    """

    def __init__(self, center, chol):
        center = np.asarray(center, dtype=np.float64)
        chol = np.asarray(chol, dtype=np.float64)
        self.shape = center.shape
        self.size = int(center.size)
        if chol.shape != (self.size, self.size):
            raise ValueError(f"chol must have shape {(self.size, self.size)}, got {chol.shape}")
        self.center_np, self.chol_np = center, chol
        self.center = jnp.asarray(center)
        self.chol = jnp.asarray(chol)

    def to_theta(self, zeta):
        """theta = center + chol zeta, shape ``shape``."""
        return _to_theta(self.center.ravel(), self.chol, jnp.asarray(zeta).ravel()).reshape(self.shape)

    def to_zeta(self, theta):
        """zeta = chol^-1 (theta - center) by a triangular solve, shape ``shape``."""
        return _to_zeta(self.center.ravel(), self.chol, jnp.asarray(theta).ravel()).reshape(self.shape)

    def grad_to_zeta(self, g):
        """chol^T g: the gradient with respect to zeta of a function of theta whose gradient is ``g``."""
        return _grad_to_zeta(self.chol, jnp.asarray(g).ravel()).reshape(self.shape)

    @classmethod
    def from_covariance(cls, center, cov, jitter=0.0):
        """The map whose covariance is ``cov`` (N, N) symmetric positive definite; ``jitter`` is added to its diagonal first."""
        cov = np.asarray(cov, dtype=np.float64)
        return cls(center, _cholesky(cov, jitter))

    @classmethod
    def from_precision_chol(cls, center, L):
        """The Laplace map of ``mrf.WhitenedMap``: precision H = L L^T (L lower), covariance H^-1 = L^-T L^-1."""
        return cls.from_covariance(center, laplace_covariance(L))

    @property
    def covariance(self):
        """chol chol^T, the covariance (N, N) that the map whitens."""
        return self.chol_np @ self.chol_np.T

    def digest(self):
        """SHA-256 (hex) of the center and the Cholesky factor bytes: the identity of the map in a segment file."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.center_np).tobytes())
        h.update(np.ascontiguousarray(self.chol_np).tobytes())
        return h.hexdigest()

    def save(self, path, **extra):
        """Write ``center``, ``chol`` and any ``extra`` arrays to an npz file."""
        np.savez(path, center=self.center_np, chol=self.chol_np, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["center"], z["chol"])


@jax.jit
def _to_theta(center, chol, zeta):
    return center + chol @ zeta


@jax.jit
def _to_zeta(center, chol, theta):
    return jsolve_triangular(chol, theta - center, lower=True)


@jax.jit
def _grad_to_zeta(chol, g):
    return chol.T @ g


def _cholesky(cov, jitter=0.0, max_tries=8):
    """Lower Cholesky factor of a symmetric matrix; on failure add a growing multiple of the mean diagonal to it."""
    cov = 0.5 * (cov + cov.T)
    scale = float(np.mean(np.diag(cov)))
    add = float(jitter)
    for k in range(max_tries):
        try:
            return cholesky(cov + add * np.eye(cov.shape[0]), lower=True)
        except np.linalg.LinAlgError:
            add = max(add * 10.0, 1e-12 * scale * 10.0**k)
    raise np.linalg.LinAlgError("covariance is not positive definite even after jitter")


def laplace_covariance(L):
    """H^-1 = L^-T L^-1 (N, N) from the lower Cholesky factor L of the precision H = L L^T."""
    Linv = solve_triangular(np.asarray(L, dtype=np.float64), np.eye(L.shape[0]), lower=True)
    cov = Linv.T @ Linv
    return 0.5 * (cov + cov.T)


def zeta_phi_fn(potential_fn, center, chol):
    """``phi_fn(zeta, aux) -> (Phi', grad Phi', aux)`` with Phi'(zeta) = U(center + chol zeta) - |zeta|^2 / 2 and gradient
    chol^T grad U - zeta, for flat ``center`` (N,) and ``chol`` (N, N) (JAX arrays, traced or constant) and a flat
    ``potential_fn(theta, aux) -> (U, grad U, aux)``."""
    def phi_fn(zeta, aux):
        u, g, aux = potential_fn(center + chol @ zeta, aux)
        return u - 0.5 * jnp.sum(zeta * zeta), chol.T @ g - zeta, aux

    return phi_fn


# ----------------------------------------------------------------------------------------------
# HMC with a variable number of leapfrog steps, and dual averaging
# ----------------------------------------------------------------------------------------------


def leapfrog_dynamic(phi_fn, q, p, grad_phi, aux, step, n_steps):
    """``n_steps`` (a traced integer) leapfrog steps of size ``step`` for H = Phi'(q) + |q|^2/2 + |p|^2/2; the same integrator as
    ``samplers.leapfrog`` (the last momentum update is a half step) but with a run-time trip count. Returns ``(q, p, phi', grad, aux)``."""
    p = p - 0.5 * step * (grad_phi + q)

    def body(i, carry):
        q, p, _, _, aux = carry
        q = q + step * p
        phi, g, aux = phi_fn(q, aux)
        p = p - jnp.where(i == n_steps - 1, 0.5, 1.0) * step * (g + q)
        return q, p, jnp.asarray(phi, dtype=q.dtype), g, aux

    init = (q, p, jnp.zeros((), q.dtype), grad_phi, aux)
    return lax.fori_loop(0, n_steps, body, init)


def n_leapfrog_range(eps, traj_length, max_leapfrog):
    """``(L0, lo, hi)``: L0 = round(traj_length / eps) clipped to [1, max_leapfrog], and the inclusive range
    lo = ceil(L0 / 2) .. hi = floor(3 L0 / 2) (at most ``max_leapfrog``) from which each trajectory draws its length uniformly."""
    l0 = jnp.clip(jnp.round(traj_length / eps), 1.0, float(max_leapfrog))
    lo = jnp.maximum(jnp.ceil(0.5 * l0), 1.0)
    hi = jnp.minimum(jnp.floor(1.5 * l0), float(max_leapfrog))
    return l0, lo, hi


def trajectory(key, state, eps, phi_fn, traj_length, max_leapfrog):
    """One HMC transition with a jittered number of leapfrog steps: L uniform on ``n_leapfrog_range`` (independent of the state, so
    the transition is reversible and volume preserving for every L), unit mass in zeta. Returns ``(new_state, alpha, accepted, L)``
    with ``alpha`` = min(1, exp(-Delta H)) (NaN counts as rejection) and ``L`` the number of gradient evaluations."""
    k_p, k_l, k_u = jax.random.split(key, 3)
    p0 = jax.random.normal(k_p, state.xi.shape, state.xi.dtype)
    _, lo, hi = n_leapfrog_range(eps, traj_length, max_leapfrog)
    n_steps = jnp.minimum(lo + jnp.floor(jax.random.uniform(k_l, dtype=state.xi.dtype) * (hi - lo + 1.0)), hi).astype(jnp.int32)
    q, p, phi, g, aux = leapfrog_dynamic(phi_fn, state.xi, p0, state.grad, state.aux, eps, n_steps)
    h_cur = state.phi + 0.5 * jnp.sum(state.xi**2) + 0.5 * jnp.sum(p0**2)
    h_prop = phi + 0.5 * jnp.sum(q**2) + 0.5 * jnp.sum(p**2)
    new, alpha, accepted = sm._mh_step(k_u, state, sm.ChainState(q, phi, g, aux), h_cur - h_prop)
    return new, alpha, accepted, n_steps


class DualAveraging(NamedTuple):
    """Nesterov dual averaging state of the step size (Hoffman and Gelman 2014): ``log_eps`` the current iterate, ``log_eps_bar``
    the averaged iterate (the step size to freeze), ``h_bar`` the running mean of target_accept - alpha, ``t`` the iteration
    count and ``mu`` the shrinkage point log(10 eps0)."""

    log_eps: Any
    log_eps_bar: Any
    h_bar: Any
    t: Any
    mu: Any


def da_init(eps0):
    """Dual averaging restarted at ``eps0``: iterate eps0, mu = log(10 eps0), zero averages (Stan's recipe)."""
    e = jnp.log(jnp.asarray(eps0, dtype=jnp.float64))
    return DualAveraging(e, jnp.zeros((), jnp.float64), jnp.zeros((), jnp.float64), jnp.zeros((), jnp.float64), e + jnp.log(10.0))


def da_update(da, alpha, target_accept, gamma=0.05, t0=10.0, kappa=0.75):
    """One dual-averaging step with the acceptance probability ``alpha``: h_bar <- (1 - 1/(t + t0)) h_bar + (target - alpha)/(t + t0),
    log eps <- mu - sqrt(t) h_bar / gamma, log eps_bar <- t^-kappa log eps + (1 - t^-kappa) log eps_bar (t counted from 1)."""
    t = da.t + 1.0
    w = 1.0 / (t + t0)
    h_bar = (1.0 - w) * da.h_bar + w * (target_accept - alpha)
    log_eps = jnp.clip(da.mu - jnp.sqrt(t) / gamma * h_bar, LOG_EPS_MIN, LOG_EPS_MAX)
    eta = t ** (-kappa)
    return DualAveraging(log_eps, eta * log_eps + (1.0 - eta) * da.log_eps_bar, h_bar, t, da.mu)


# ----------------------------------------------------------------------------------------------
# The map update
# ----------------------------------------------------------------------------------------------


def shrinkage_weight(n_draws, kappa):
    """w = kappa / (n_draws + kappa): the weight of the Laplace covariance in the map update of a window of ``n_draws`` draws."""
    return float(kappa) / (float(n_draws) + float(kappa))


def update_map(draws, laplace_cov_fn, kappa, prev=None):
    """The map from one window of draws: ``(AffineMap, info)``.

    ``draws`` (n, N) are the theta states of the window's trajectories. The new center is their mean m; S is their sample
    covariance (denominator n - 1), T = ``laplace_cov_fn(m)`` the Laplace covariance at m (N, N), and the new covariance is
    Sigma_hat = (1 - w) S + w T with w = ``shrinkage_weight(n, kappa)``. S has rank at most n - 1, so for n < N the T term is what
    makes Sigma_hat positive definite and what it contributes in the directions the window did not explore; w decays like kappa / n
    so a long window is dominated by its own draws. ``prev`` (an ``AffineMap``) is only used for the diagnostics in ``info``:
    ``weight``, ``n_draws``, ``trace_sample_over_laplace`` (tr S / tr T: how much variance the window saw relative to the Laplace
    covariance it is shrunk toward; far below 1 means the chain barely moved), ``center_shift`` (|m - previous center| in the previous map's whitened units), ``var_ratio_{min,median,max}``
    (new over previous marginal variances), ``trace_ratio``, ``pivot_ratio_sq`` (max/min squared diagonal of the Cholesky factor,
    a lower bound on the condition number of Sigma_hat) and the timings in seconds: ``time_sample_cov_s`` (mean and S), ``time_laplace_s``
    (``laplace_cov_fn``), ``time_mix_cholesky_s`` (the mixture and its Cholesky factor), ``time_s`` (the whole update).
    """
    t_all = time.perf_counter()
    draws = np.asarray(draws, dtype=np.float64)
    n = draws.shape[0]
    m = draws.mean(axis=0)
    X = draws - m
    S = X.T @ X / max(n - 1, 1)
    time_cov = time.perf_counter() - t_all
    t0 = time.perf_counter()
    T = np.asarray(laplace_cov_fn(m), dtype=np.float64)
    time_laplace = time.perf_counter() - t0
    w = shrinkage_weight(n, kappa)
    t0 = time.perf_counter()
    cov = (1.0 - w) * S + w * T
    new = AffineMap.from_covariance(m.reshape(prev.shape) if prev is not None else m, cov)
    time_chol = time.perf_counter() - t0
    info = {"n_draws": int(n), "weight": w, "trace_sample_over_laplace": float(np.trace(S) / np.trace(T)),
            "time_sample_cov_s": time_cov, "time_laplace_s": time_laplace, "time_mix_cholesky_s": time_chol,
            "pivot_ratio_sq": float((new.chol_np.diagonal().max() / new.chol_np.diagonal().min()) ** 2)}
    if prev is not None:
        var_new, var_prev = np.sum(new.chol_np**2, axis=1), np.sum(prev.chol_np**2, axis=1)  # diag(chol chol^T) without forming it
        d = var_new / var_prev
        info.update({"var_ratio_min": float(d.min()), "var_ratio_median": float(np.median(d)), "var_ratio_max": float(d.max()),
                     "trace_ratio": float(var_new.sum() / var_prev.sum()),
                     "center_shift": float(np.linalg.norm(solve_triangular(prev.chol_np, m - prev.center_np.ravel(), lower=True)))})
    info["time_s"] = time.perf_counter() - t_all
    return new, info


# ----------------------------------------------------------------------------------------------
# Adaptation driver
# ----------------------------------------------------------------------------------------------


@dataclass
class AdaptConfig:
    """Schedule and tuning of ``adapt_map`` (all lengths in trajectories).

    ``init_buffer`` : step-size-only phase with the initial map. ``first_window`` and ``n_windows``, ``growth`` : the map-update
    windows have lengths first_window * growth^k, k = 0 .. n_windows - 1 (expanding windows; growth 1 gives equal windows, used by
    the negative control). ``final_buffer`` : step-size-only phase with the final map (its dual-averaging average is the step size
    that is frozen). ``block`` : trajectories per compiled block (every phase is rounded up to a multiple; one compilation serves
    all phases). ``target_accept`` : dual-averaging target, 0.8. ``traj_length`` : L0 * eps in whitened units (DESIGN: 1.5 to 2).
    ``max_leapfrog`` : cap on L0 and on the longest trajectory (a tiny step would otherwise make a trajectory last minutes; the
    trajectory is then shorter than ``traj_length``, still a valid HMC transition). ``kappa`` : pseudo-count of the Laplace
    covariance in ``update_map``; None means kappa = ``kappa_per_dim`` * N, the number of draws a full-rank sample
    covariance needs (a window shorter than N is mostly shrinkage; a window of several N is mostly data). ``min_accept_update`` :
    a window whose mean acceptance is below this (a stuck chain: its sample covariance is meaningless) does not update the map.
    ``step0`` : the first step size. ``dual_averaging`` : (gamma, t0, kappa) of ``da_update``.
    """

    init_buffer: int = 50
    first_window: int = 50
    n_windows: int = 4
    growth: float = 2.0
    final_buffer: int = 100
    block: int = 25
    target_accept: float = 0.8
    traj_length: float = 1.75
    max_leapfrog: int = 100
    kappa: float = None
    kappa_per_dim: float = 1.0
    min_accept_update: float = 0.1
    step0: float = 0.3
    dual_averaging: tuple = (0.05, 10.0, 0.75)

    def phases(self):
        """``[(name, n_blocks, update_map)]`` for the schedule, every length rounded up to whole blocks."""
        nb = lambda n: int(-(-int(n) // self.block))
        out = []
        if self.init_buffer > 0:
            out.append(("init", nb(self.init_buffer), False))
        for k in range(self.n_windows):
            out.append((f"window{k}", nb(round(self.first_window * self.growth**k)), True))
        if self.final_buffer > 0:
            out.append(("final", nb(self.final_buffer), False))
        return out

    def total_trajectories(self):
        return sum(nb for _, nb, _ in self.phases()) * self.block

    def kappa_for(self, n_dim):
        return float(self.kappa) if self.kappa is not None else self.kappa_per_dim * n_dim


@dataclass
class AdaptResult:
    """Output of ``adapt_map``: the frozen ``map`` (``AffineMap``), the step size ``eps`` and ``traj_length`` (L0 eps = traj_length,
    L0 = ``n_leapfrog0``), the final state ``theta`` (N,) and ``aux``, ``history`` (one dict per phase), the gradient-evaluation
    count ``n_grad_evals``, wall time ``wall_s`` and the per-block draws ``draws`` (only with ``keep_draws``: list of (n, N))."""

    map: AffineMap
    eps: float
    traj_length: float
    max_leapfrog: int
    theta: np.ndarray
    aux: Any
    history: list
    n_grad_evals: int
    wall_s: float
    draws: list = field(default_factory=list)

    @property
    def n_leapfrog0(self):
        return int(np.clip(round(self.traj_length / self.eps), 1, self.max_leapfrog))


@functools.lru_cache(maxsize=16)
def make_block_runner(potential_fn, n_block, traj_length, max_leapfrog, target_accept, dual_averaging=(0.05, 10.0, 0.75)):
    """The compiled block of ``n_block`` trajectories shared by every phase of ``adapt_map`` (cached on its arguments, so calls of
    ``adapt_map`` with the same ``potential_fn`` object and configuration compile it once).

    Returns ``run(key, state, center, chol, da) -> (state, da, out)``: ``center`` (N,) and ``chol`` (N, N) are the (traced) map,
    ``state`` a ``samplers.ChainState`` in zeta and ``da`` a ``DualAveraging``. The step size of every trajectory is exp(da.log_eps)
    and the dual averaging is updated with the trajectory's acceptance probability. ``out`` is a dict of arrays with a leading axis
    ``n_block``: ``theta`` (n_block, N) the state after each trajectory, ``alpha``, ``accepted``, ``n_leapfrog`` and ``u`` (the
    potential U at the new state).
    """
    gamma, t0, kappa = dual_averaging

    @jax.jit
    def run(key, state, center, chol, da):
        phi_fn = zeta_phi_fn(potential_fn, center, chol)

        def body(carry, k):
            state, da = carry
            state, alpha, accepted, n_steps = trajectory(k, state, jnp.exp(da.log_eps), phi_fn, traj_length, max_leapfrog)
            da = da_update(da, alpha, target_accept, gamma, t0, kappa)
            out = {"theta": center + chol @ state.xi, "alpha": alpha, "accepted": accepted, "n_leapfrog": n_steps,
                   "u": state.phi + 0.5 * jnp.sum(state.xi**2)}
            return (state, da), out

        (state, da), out = lax.scan(body, (state, da), jax.random.split(key, n_block))
        return state, da, out

    return run


def adapt_map(potential_fn, key, theta0, aux0, map0, laplace_cov_fn, cfg=None, *, keep_draws=False, progress=None):
    """Adapt the affine map (and the step size) of an HMC chain from warmup draws; returns an ``AdaptResult``.

    ``potential_fn(theta, aux) -> (U, grad U, aux)`` is the flat potential, ``theta0`` (N,) the start, ``aux0`` its warm start,
    ``map0`` the initial ``AffineMap`` (the Laplace map of the MAP), ``laplace_cov_fn(center) -> (N, N)`` the Laplace covariance at a
    center (the shrinkage target of ``update_map``), ``cfg`` an ``AdaptConfig``. The chain runs the phases of ``cfg.phases()`` in
    order. Every phase restarts the dual averaging at the previous phase's averaged step. A window phase ends with ``update_map``
    from the window's draws (if its acceptance is at least ``cfg.min_accept_update``), after which the state is re-expressed in
    the new coordinates (theta is unchanged; zeta = chol^-1 (theta - center)) and re-evaluated. The map the function returns is
    the one used by the last phase, i.e. frozen before any draw recorded afterwards; with ``keep_draws`` every phase's draws are
    returned too (``AdaptResult.draws``), which is how the negative control records with adaptation left on.
    ``progress`` (optional callable) is called with each history entry as it is completed.
    """
    cfg = cfg or AdaptConfig()
    t_start = time.perf_counter()
    theta0 = np.asarray(theta0, dtype=np.float64).ravel()
    n_dim = theta0.size
    kappa = cfg.kappa_for(n_dim)
    run = make_block_runner(potential_fn, cfg.block, cfg.traj_length, cfg.max_leapfrog, cfg.target_accept, cfg.dual_averaging)

    @jax.jit
    def evaluate(center, chol, zeta, aux):
        phi, g, aux = zeta_phi_fn(potential_fn, center, chol)(zeta, aux)
        return sm.ChainState(zeta, jnp.asarray(phi, dtype=zeta.dtype), g, aux)

    amap = AffineMap(np.asarray(map0.center_np).ravel(), map0.chol_np)
    state = evaluate(amap.center, amap.chol, amap.to_zeta(jnp.asarray(theta0)), aux0)
    da = da_init(cfg.step0)
    history, kept, n_grad, counter, eps_end = [], [], 1, 0, cfg.step0
    for name, n_blocks, do_update in cfg.phases():
        t_phase = time.perf_counter()
        draws, alpha, acc, nl = [], [], [], []
        for _ in range(n_blocks):
            key_b = jax.random.fold_in(key, counter)
            counter += 1
            state, da, out = run(key_b, state, amap.center, amap.chol, da)
            draws.append(np.asarray(out["theta"]))
            alpha.append(np.asarray(out["alpha"]))
            acc.append(np.asarray(out["accepted"]))
            nl.append(np.asarray(out["n_leapfrog"]))
        draws, alpha, acc, nl = (np.concatenate(x) for x in (draws, alpha, acc, nl))
        n_grad += int(nl.sum())
        eps_end = float(np.exp(float(da.log_eps_bar)))
        rec = {"phase": name, "trajectories": int(draws.shape[0]), "accept_mean": float(acc.mean()),
               "alpha_mean": float(alpha.mean()), "eps_end": eps_end, "eps_last_iterate": float(np.exp(float(da.log_eps))),
               "n_leapfrog0_end": int(np.clip(round(cfg.traj_length / eps_end), 1, cfg.max_leapfrog)),
               "n_grad_evals": int(nl.sum()), "mean_leapfrog": float(nl.mean()), "map_digest_before": amap.digest()[:16],
               "updated": False}
        if keep_draws:
            kept.append(draws)
        if do_update:
            if rec["accept_mean"] < cfg.min_accept_update:
                rec["skipped"] = f"acceptance {rec['accept_mean']:.3f} < {cfg.min_accept_update}: the chain was stuck"
            else:
                theta_cur = np.asarray(out["theta"][-1])
                amap, info = update_map(draws, laplace_cov_fn, kappa, prev=amap)
                state = evaluate(amap.center, amap.chol, amap.to_zeta(jnp.asarray(theta_cur)), state.aux)
                n_grad += 1
                rec.update({"updated": True, "update": info, "map_digest_after": amap.digest()[:16]})
        da = da_init(eps_end)  # restart dual averaging at the averaged step (also after a phase without update)
        rec["time_s"] = time.perf_counter() - t_phase
        history.append(rec)
        if progress is not None:
            progress(rec)
    return AdaptResult(amap, eps_end, cfg.traj_length, cfg.max_leapfrog, np.asarray(amap.center_np.ravel() + amap.chol_np @ np.asarray(state.xi)),
                       state.aux, history, n_grad, time.perf_counter() - t_start, kept)


# ----------------------------------------------------------------------------------------------
# Recording with everything frozen
# ----------------------------------------------------------------------------------------------


def sample_chain(key, phi_fn, xi0, aux0, *, eps, traj_length, max_leapfrog, n_samples, thin=1, record_fn=None):
    """HMC for a fixed map (inside ``phi_fn``), step size and trajectory length: ``n_samples`` trajectories after the start ``xi0``.

    ``phi_fn(zeta, aux) -> (Phi', grad Phi', aux)`` is the potential in zeta (``zeta_phi_fn`` or ``thetaspace.ZetaTarget.phi_fn``),
    ``record_fn(zeta) -> array`` what to store (default zeta itself). Same outputs and conventions as ``samplers.hmc``'s
    ``ChainResult`` (every ``thin``-th recorded value is stored; ``mean`` and ``var`` are Welford moments over ALL trajectories, ``var``
    unbiased; ``phi`` is the sampler potential Phi' after every trajectory) plus ``n_leapfrog`` (the gradient evaluations of every
    trajectory). Returns a dict: ``samples`` (n_samples // thin, ...), ``phi``, ``accept`` (bool), ``n_leapfrog`` (int), ``mean``, ``var``,
    ``xi`` (final zeta), ``aux`` (its warm start). Jit it (``phi_fn`` and the integers static) for speed.
    """
    if n_samples < 1 or thin < 1:
        raise ValueError("need n_samples >= 1 and thin >= 1")
    record_fn = record_fn or sm.identity
    xi0 = jnp.asarray(xi0, dtype=jnp.float64)
    phi0, g0, aux = phi_fn(xi0, aux0)
    state = sm.ChainState(xi0, jnp.asarray(phi0, dtype=xi0.dtype), g0, aux)
    eps = jnp.asarray(eps, dtype=xi0.dtype)
    r0 = jnp.asarray(record_fn(state.xi))
    mean0 = jnp.zeros(r0.shape, dtype=jnp.result_type(r0.dtype, float))
    m2_0 = jnp.zeros(r0.shape, dtype=mean0.real.dtype)

    def one_step(carry, k):
        state, r, count, mean, m2 = carry
        state, _, accepted, n_steps = trajectory(k, state, eps, phi_fn, traj_length, max_leapfrog)
        r = jnp.asarray(record_fn(state.xi))
        count = count + 1.0
        delta = r - mean
        mean = mean + delta / count
        m2 = m2 + jnp.real(delta * jnp.conj(r - mean))
        return (state, r, count, mean, m2), (state.phi, accepted, n_steps)

    def block(carry, ks):
        carry, out = lax.scan(one_step, carry, ks)
        return carry, (carry[1], *out)

    keys = jax.random.split(key, n_samples)
    n_blocks, rest = divmod(n_samples, thin)
    carry = (state, r0, jnp.zeros((), xi0.dtype), mean0, m2_0)
    carry, (samples, phis, accs, nls) = lax.scan(block, carry, keys[:n_blocks * thin].reshape((n_blocks, thin) + keys.shape[1:]))
    phis, accs, nls = phis.reshape(-1), accs.reshape(-1), nls.reshape(-1)
    if rest:
        carry, (phi_r, acc_r, nl_r) = lax.scan(one_step, carry, keys[n_blocks * thin:])
        phis, accs, nls = (jnp.concatenate([a, b]) for a, b in ((phis, phi_r), (accs, acc_r), (nls, nl_r)))
    state, _, count, mean, m2 = carry
    var = m2 / jnp.maximum(count - 1.0, 1.0)
    return {"samples": samples, "phi": phis, "accept": accs, "n_leapfrog": nls, "mean": mean, "var": var, "xi": state.xi,
            "aux": state.aux}
