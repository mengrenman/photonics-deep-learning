"""MCMC samplers in JAX: pCN, pCN-Langevin, MALA and HMC (docs/DESIGN.md, section 5).

All samplers target

    pi(xi) ~ exp(-Phi(xi)) N(xi; 0, I),

are physics-agnostic, and take the same callable ``phi_fn(xi, aux) -> (phi, grad_phi, aux_new)``:
``phi`` is the scalar potential Phi (negative log-likelihood), ``grad_phi`` its gradient with the
same shape as ``xi`` (``None`` is allowed for pCN, which never reads it), and ``aux`` an arbitrary
pytree (a solver warm start in the scattering problem) that must not change Phi beyond solver
tolerance. ``xi`` may have any shape; the prior N(0, I) is applied to every entry. In terms of
U(xi) = Phi(xi) + |xi|^2 / 2 the target is exp(-U).

Common behavior of every sampler
--------------------------------
* The chain runs in ``jax.lax.scan`` and every sampler is ``jax.jit``-compiled (``phi_fn``,
  ``n_warmup``, ``n_samples``, ``thin``, ``record_fn`` and ``n_leapfrog`` are static arguments), so
  all of them also work under ``jax.vmap`` (see ``run_chains``).
* ``aux`` is threaded through the chain with the state it belongs to: a proposal is evaluated with the
  aux of the current state, and on rejection the aux of the current state is kept (for HMC, the aux
  at the start of the trajectory), so the returned ``ChainResult.aux`` always matches
  ``ChainResult.xi``.
* The first ``n_warmup`` steps adapt the step size by a Robbins-Monro recursion on log(step),
  log s_{t+1} = log s_t + gamma_t (alpha_t - target_accept), gamma_t = (t + 10)^(-0.6), where alpha_t
  = min(1, exp(log acceptance ratio)) is the acceptance probability of step t (smoother than the 0/1
  flag). The step is then frozen. ``samples``, ``phi``, ``accept``, ``mean`` and ``var`` use only the
  ``n_samples`` post-warmup steps.
* ``record_fn(xi) -> array`` chooses what is stored (default: ``xi`` itself). Every ``thin``-th
  recorded value is kept in ``samples`` (so ``n_samples // thin`` values; the leftover
  ``n_samples % thin`` steps are still run and counted in the other outputs), while ``mean`` and
  ``var`` are Welford running statistics of ``record_fn`` over ALL post-warmup steps, ``var`` being
  the unbiased sample variance (denominator count - 1). ``phi`` and ``accept`` have one entry per
  post-warmup step.
* Rejected steps repeat the current state, so ``phi``, ``mean`` and ``var`` include the repeats, as
  required for a correct Markov chain average.

Step-size meaning
-----------------
``pcn``: beta in (0, 1] (clipped at 1); ``pcn_langevin``: delta in (0, 2] (clipped at 2); ``mala``: h > 0;
``hmc``: leapfrog step epsilon > 0 with a fixed ``n_leapfrog``. A NaN or +inf Phi at a proposal
is treated as a rejection.
"""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

# Robbins-Monro gain: gamma_t = (t + _T0) ** -_KAPPA. KAPPA in (1/2, 1] makes sum(gamma) diverge and
# sum(gamma^2) converge; _T0 keeps the first corrections moderate (at most ~0.25 in log step).
_T0 = 10.0
_KAPPA = 0.6

# pCN-Langevin: the AR coefficient (2 - delta)/(2 + delta) is 0 at delta = 2 (independent proposals) and
# tends to -1 as delta -> infinity (the chain would just flip sign). When Phi is flat the acceptance
# is exactly 1 for every delta, so an uncapped adaptation would run away to that useless regime.
_LOG_DELTA_MAX = 0.6931471805599453  # log 2

_STATIC = ("phi_fn", "n_warmup", "n_samples", "thin", "record_fn")


class ChainState(NamedTuple):
    """State of one chain: position, its potential and gradient, and the matching aux."""

    xi: Any
    phi: Any
    grad: Any
    aux: Any


class ChainResult(NamedTuple):
    """Output of a sampler.

    samples : (n_samples // thin, *record_shape) thinned ``record_fn`` values.
    phi     : (n_samples,) potential Phi of the state after every post-warmup step.
    accept  : (n_samples,) bool, whether each post-warmup proposal was accepted.
    step    : final (frozen) step size.
    mean    : running mean of ``record_fn`` over all post-warmup steps (``record_shape``).
    var     : running unbiased variance of ``record_fn`` over all post-warmup steps.
    xi      : final state.
    aux     : aux belonging to the final state.
    With ``run_chains`` every field gets a leading chain axis.
    """

    samples: Any
    phi: Any
    accept: Any
    step: Any
    mean: Any
    var: Any
    xi: Any
    aux: Any


def identity(xi):
    """Default ``record_fn``: record the state itself."""
    return xi


def _evaluate(phi_fn, xi, aux):
    """Evaluate ``phi_fn`` at ``xi`` and return the full ChainState."""
    phi, grad, aux_new = phi_fn(xi, aux)
    return ChainState(xi, jnp.asarray(phi, dtype=xi.dtype), grad, aux_new)


def _mh_step(key, state, proposal, log_ratio):
    """Metropolis-Hastings accept/reject. Returns (new_state, alpha, accepted).

    A NaN log ratio (diverged proposal) counts as -inf. The whole state, aux included, is taken from
    the proposal if accepted and from ``state`` otherwise.
    """
    log_ratio = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
    alpha = jnp.exp(jnp.minimum(log_ratio, 0.0))
    accepted = jnp.log(jax.random.uniform(key, dtype=log_ratio.dtype)) < log_ratio
    new = jax.tree.map(lambda p, s: jnp.where(accepted, p, s), proposal, state)
    return new, alpha, accepted


def _sq(x):
    return jnp.sum(x * x)


# ---------------------------------------------------------------------------------------------
# Kernels: kernel(key, state, step, phi_fn) -> (new_state, alpha, accepted)
# ---------------------------------------------------------------------------------------------


def _pcn_kernel(key, state, beta, phi_fn):
    k_w, k_u = jax.random.split(key)
    w = jax.random.normal(k_w, state.xi.shape, state.xi.dtype)
    v = jnp.sqrt(jnp.maximum(1.0 - beta**2, 0.0)) * state.xi + beta * w
    prop = _evaluate(phi_fn, v, state.aux)
    return _mh_step(k_u, state, prop, state.phi - prop.phi)


def _pcn_langevin_kernel(key, state, delta, phi_fn):
    """Crank-Nicolson Langevin (infinity-MALA): proposal q(.|u) = N(a u - b grad Phi(u), c^2 I)."""
    a = (2.0 - delta) / (2.0 + delta)
    b = 2.0 * delta / (2.0 + delta)
    c2 = 8.0 * delta / (2.0 + delta) ** 2

    def mean(s):
        return a * s.xi - b * s.grad

    k_w, k_u = jax.random.split(key)
    w = jax.random.normal(k_w, state.xi.shape, state.xi.dtype)
    v = mean(state) + jnp.sqrt(c2) * w
    prop = _evaluate(phi_fn, v, state.aux)
    log_target = (-prop.phi - 0.5 * _sq(v)) - (-state.phi - 0.5 * _sq(state.xi))
    log_q_fwd = -_sq(v - mean(state)) / (2.0 * c2)  # log q(v | u), constants cancel
    log_q_bwd = -_sq(state.xi - mean(prop)) / (2.0 * c2)  # log q(u | v)
    return _mh_step(k_u, state, prop, log_target + log_q_bwd - log_q_fwd)


def _mala_kernel(key, state, h, phi_fn):
    """MALA on U = Phi + |xi|^2 / 2: proposal q(.|u) = N(u - (h/2) grad U(u), h I)."""

    def mean(s):
        return s.xi - 0.5 * h * (s.grad + s.xi)

    k_w, k_u = jax.random.split(key)
    w = jax.random.normal(k_w, state.xi.shape, state.xi.dtype)
    v = mean(state) + jnp.sqrt(h) * w
    prop = _evaluate(phi_fn, v, state.aux)
    u_cur = state.phi + 0.5 * _sq(state.xi)
    u_prop = prop.phi + 0.5 * _sq(v)
    log_q_fwd = -_sq(v - mean(state)) / (2.0 * h)
    log_q_bwd = -_sq(state.xi - mean(prop)) / (2.0 * h)
    return _mh_step(k_u, state, prop, u_cur - u_prop + log_q_bwd - log_q_fwd)


def leapfrog(phi_fn, q, p, grad_phi, aux, step, n_steps):
    """``n_steps`` leapfrog steps of size ``step`` for H(q, p) = Phi(q) + |q|^2/2 + |p|^2/2.

    ``grad_phi`` is the gradient of Phi at ``q`` (reused, so a trajectory costs ``n_steps`` calls of
    ``phi_fn``). ``aux`` is passed along the trajectory. Returns ``(q, p, phi, grad_phi, aux)`` at the
    end of the trajectory. The map (q, p) -> (q', p') is reversible (flip p) and volume preserving,
    and the energy error is O(step^2) at fixed trajectory length ``step * n_steps``.
    """
    p = p - 0.5 * step * (grad_phi + q)

    def body(carry, i):
        q, p, _, _, aux = carry
        q = q + step * p
        phi, g, aux = phi_fn(q, aux)
        # Full momentum step, except the last one which is a half step (momentum is then synchronous).
        p = p - jnp.where(i == n_steps - 1, 0.5, 1.0) * step * (g + q)
        return (q, p, jnp.asarray(phi, dtype=q.dtype), g, aux), None

    init = (q, p, jnp.zeros((), q.dtype), grad_phi, aux)
    (q, p, phi, g, aux), _ = lax.scan(body, init, jnp.arange(n_steps))
    return q, p, phi, g, aux


def _hmc_kernel(key, state, eps, phi_fn, *, n_leapfrog, jitter):
    k_p, k_j, k_u = jax.random.split(key, 3)
    p0 = jax.random.normal(k_p, state.xi.shape, state.xi.dtype)
    # Optional step jitter eps * (1 + jitter * U(-1, 1)), drawn independently of the state, so each
    # trajectory is still reversible and volume preserving.
    eps = eps * (1.0 + jitter * (2.0 * jax.random.uniform(k_j, dtype=state.xi.dtype) - 1.0))
    q, p, phi, g, aux = leapfrog(phi_fn, state.xi, p0, state.grad, state.aux, eps, n_leapfrog)
    h_cur = state.phi + 0.5 * _sq(state.xi) + 0.5 * _sq(p0)
    h_prop = phi + 0.5 * _sq(q) + 0.5 * _sq(p)
    return _mh_step(k_u, state, ChainState(q, phi, g, aux), h_cur - h_prop)


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------


def _run_chain(kernel, key, phi_fn, xi0, aux0, *, n_warmup, n_samples, step0, thin, record_fn,
               target_accept, log_step_max):
    """Warmup with Robbins-Monro step adaptation, then sampling, for any kernel.

    ``kernel(key, state, step, phi_fn) -> (new_state, alpha, accepted)``. ``log_step_max`` caps log(step)
    (0 for pCN, whose step beta must not exceed 1; ``inf`` otherwise). Not jitted itself: the public
    samplers wrap it in ``jax.jit``.
    """
    if n_warmup < 0 or n_samples < 1 or thin < 1:
        raise ValueError("need n_warmup >= 0, n_samples >= 1 and thin >= 1")
    dtype = jnp.result_type(float)
    state = _evaluate(phi_fn, jnp.asarray(xi0, dtype=dtype), aux0)
    log_step = jnp.minimum(jnp.log(jnp.asarray(step0, dtype=dtype)), log_step_max)

    def warmup_body(carry, t):
        key, state, log_step = carry
        key, sub = jax.random.split(key)
        state, alpha, _ = kernel(sub, state, jnp.exp(log_step), phi_fn)
        gamma = (t + _T0) ** (-_KAPPA)
        log_step = jnp.minimum(log_step + gamma * (alpha - target_accept), log_step_max)
        return (key, state, log_step), None

    (key, state, log_step), _ = lax.scan(
        warmup_body, (key, state, log_step), jnp.arange(n_warmup, dtype=dtype))
    step = jnp.exp(log_step)

    r0 = jnp.asarray(record_fn(state.xi))
    mean0 = jnp.zeros(r0.shape, dtype=jnp.result_type(r0.dtype, float))
    m2_0 = jnp.zeros(r0.shape, dtype=mean0.real.dtype)

    def one_step(carry, _):
        key, state, r, count, mean, m2 = carry
        key, sub = jax.random.split(key)
        state, _, accepted = kernel(sub, state, step, phi_fn)
        r = jnp.asarray(record_fn(state.xi))
        count = count + 1.0
        delta = r - mean
        mean = mean + delta / count
        m2 = m2 + jnp.real(delta * jnp.conj(r - mean))  # Welford update
        return (key, state, r, count, mean, m2), (state.phi, accepted)

    def block(carry, _):
        carry, out = lax.scan(one_step, carry, None, length=thin)
        return carry, (carry[2], *out)

    carry = (key, state, r0, jnp.zeros((), dtype), mean0, m2_0)
    n_blocks, rest = divmod(n_samples, thin)
    carry, (samples, phis, accs) = lax.scan(block, carry, None, length=n_blocks)
    phis, accs = phis.reshape(-1), accs.reshape(-1)
    if rest:
        carry, (phi_rest, acc_rest) = lax.scan(one_step, carry, None, length=rest)
        phis = jnp.concatenate([phis, phi_rest])
        accs = jnp.concatenate([accs, acc_rest])
    _, state, _, count, mean, m2 = carry
    var = m2 / jnp.maximum(count - 1.0, 1.0)
    return ChainResult(samples, phis, accs, step, mean, var, state.xi, state.aux)


# ---------------------------------------------------------------------------------------------
# Public samplers
# ---------------------------------------------------------------------------------------------


@partial(jax.jit, static_argnames=_STATIC)
def pcn(key, phi_fn, xi0, aux0, *, n_warmup, n_samples, step0, thin=1, record_fn=identity,
        target_accept=0.25):
    """Preconditioned Crank-Nicolson Metropolis (gradient free).

    Proposal v = sqrt(1 - beta^2) u + beta w, w ~ N(0, I), which is reversible with respect to the
    prior, so the log acceptance ratio is Phi(u) - Phi(v). ``step0`` is beta (clipped to <= 1);
    default adaptation target 0.25. ``phi_fn`` may return ``grad_phi = None``.
    """
    return _run_chain(_pcn_kernel, key, phi_fn, xi0, aux0, n_warmup=n_warmup, n_samples=n_samples,
                      step0=step0, thin=thin, record_fn=record_fn, target_accept=target_accept,
                      log_step_max=0.0)


@partial(jax.jit, static_argnames=_STATIC)
def pcn_langevin(key, phi_fn, xi0, aux0, *, n_warmup, n_samples, step0, thin=1, record_fn=identity,
                 target_accept=0.5):
    """Crank-Nicolson Langevin (infinity-MALA, Cotter et al. 2013, theta = 1/2).

    Proposal v = ((2 - delta)/(2 + delta)) u - (2 delta/(2 + delta)) grad Phi(u)
    + (sqrt(8 delta)/(2 + delta)) w, accepted with the generic Metropolis-Hastings ratio
    [-Phi(v) - |v|^2/2] - [-Phi(u) - |u|^2/2] + log q(u|v) - log q(v|u), where
    q(.|u) = N(mu(u), 8 delta/(2 + delta)^2 I) is the explicit Gaussian proposal density.
    ``step0`` is delta in (0, 2] (capped at 2, where the autoregressive coefficient vanishes and the
    proposals are independent; adaptation never exceeds it); default adaptation target 0.5.
    """
    return _run_chain(_pcn_langevin_kernel, key, phi_fn, xi0, aux0, n_warmup=n_warmup,
                      n_samples=n_samples, step0=step0, thin=thin, record_fn=record_fn,
                      target_accept=target_accept, log_step_max=_LOG_DELTA_MAX)


@partial(jax.jit, static_argnames=_STATIC)
def mala(key, phi_fn, xi0, aux0, *, n_warmup, n_samples, step0, thin=1, record_fn=identity,
         target_accept=0.574):
    """Metropolis-adjusted Langevin on U = Phi + |xi|^2/2.

    Proposal v = u - (h/2) grad U(u) + sqrt(h) w, accepted with probability
    exp(-U(v) + U(u) + log q(u|v) - log q(v|u)), q(.|u) = N(u - (h/2) grad U(u), h I).
    ``step0`` is h > 0; default adaptation target 0.574.
    """
    return _run_chain(_mala_kernel, key, phi_fn, xi0, aux0, n_warmup=n_warmup, n_samples=n_samples,
                      step0=step0, thin=thin, record_fn=record_fn, target_accept=target_accept,
                      log_step_max=jnp.inf)


@partial(jax.jit, static_argnames=_STATIC + ("n_leapfrog",))
def hmc(key, phi_fn, xi0, aux0, *, n_warmup, n_samples, step0, thin=1, record_fn=identity,
        target_accept=0.7, n_leapfrog=10, jitter=0.0):
    """Hamiltonian Monte Carlo with identity mass matrix and a fixed number of leapfrog steps.

    p ~ N(0, I), ``n_leapfrog`` leapfrog steps of size epsilon on H = Phi + |xi|^2/2 + |p|^2/2,
    accepted with probability exp(H(u, p) - H(v, p')). ``step0`` is epsilon > 0; default adaptation
    target 0.7. Each step costs ``n_leapfrog`` calls of ``phi_fn``; on rejection the aux at the start
    of the trajectory is restored.

    ``jitter`` (extension to DESIGN, default 0 = the plain algorithm): each trajectory uses the step
    epsilon * (1 + jitter * U(-1, 1)). With a fixed trajectory length epsilon * n_leapfrog, directions
    of a near-Gaussian target whose period divides the trajectory time barely move (resonance); this
    shows up as a few directions with tiny ESS and R-hat above 1. ``jitter`` of 0.1 to 0.3 removes it.
    The adapted ``step`` returned is the nominal epsilon.
    """
    kernel = partial(_hmc_kernel, n_leapfrog=n_leapfrog, jitter=jitter)
    return _run_chain(kernel, key, phi_fn, xi0, aux0, n_warmup=n_warmup, n_samples=n_samples,
                      step0=step0, thin=thin, record_fn=record_fn, target_accept=target_accept,
                      log_step_max=jnp.inf)


def run_chains(sampler, keys, xi0s, aux0s, **kw):
    """Run independent chains with ``jax.vmap``.

    sampler : ``pcn``, ``pcn_langevin``, ``mala`` or ``hmc``.
    keys    : one PRNG key per chain, leading axis n_chains (e.g. ``jax.random.split(key, n)``).
    xi0s    : (n_chains, *xi_shape) initial states.
    aux0s   : pytree whose leaves have leading axis n_chains (``None`` if the aux is empty).
    **kw    : everything else, passed unbatched to the sampler: ``phi_fn``, ``n_warmup``,
              ``n_samples``, ``step0``, ``thin``, ``record_fn``, ``target_accept``, ``n_leapfrog``.
    Returns a ``ChainResult`` whose fields have a leading chain axis.
    """
    return jax.vmap(lambda k, x, a: sampler(k, xi0=x, aux0=a, **kw))(keys, xi0s, aux0s)
