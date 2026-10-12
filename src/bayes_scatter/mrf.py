"""Prior energies R(theta) on the n x n DOI grid of theta = log eps_r (docs/DESIGN.md section 11.2).

Unlike ``priors.GaussianField`` (a field on a padded torus, sampled in whitened coordinates xi), the priors
here are *energies* on the DOI grid itself, so that the posterior in theta-space is exp(-U(theta)),
U = Phi(chi(theta)) + R(theta). Conventions: theta has shape (n, n) with "ij" indexing ([i, j] = [x, y]);
vectors and matrices over pixels use the C-order flattening k = i n + j.

Common interface (``ThetaPrior``): ``energy(theta) -> scalar``, ``grad(theta) -> (n, n)``,
``value_and_grad``, ``hvp(theta, v) -> (n, n)`` (Hessian-vector product by forward-over-reverse autodiff),
``hessian(theta) -> (n^2, n^2)`` NumPy array, and the prior mean ``mean`` (n, n). ``describe()`` returns the
hyperparameters as a plain dict. All methods are jitted and take/return JAX arrays except ``hessian``.

* ``QGGMRF`` : the edge-preserving q-generalized Gaussian MRF of Thibault, Sauer, Bouman and Hsieh (2007) with
  p = 2, eight neighbors, free boundary.
* ``PseudoHuberTV`` : the pseudo-Huber total-variation prior (docs/DESIGN.md section 12.1), same neighborhood and
  weights as the q-GGMRF, convex, with a (nearly) linear pair potential so that a sharp step costs about the same as a ramp.
* ``GaussianEnergy`` : R(theta) = 1/2 (theta - mu)^T C_D^-1 (theta - mu) with C_D =
  ``GaussianField.covariance_matrix()``, the exact DOI marginal of the torus field, so a Matern or GMRF
  prior of docs/DESIGN.md section 3 cropped to the DOI.

``WhitenedMap`` is the affine map theta = center + L^-T zeta (L the lower Cholesky factor of a symmetric positive
definite "mass" matrix H = L L^T) that the Laplace-whitened samplers run in, and ``sample_prior`` runs HMC on R
alone in the coordinates given by the Hessian of R at the prior mean, to estimate the prior standard deviation of
a prior that has no closed form (the q-GGMRF).
"""

import functools
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from scipy.linalg import cho_solve, cholesky

from . import diagnostics as dg
from . import samplers as sm

# Smoothing of |Delta| inside |Delta|^(p-q): (Delta^2 + EPS^2)^((p-q)/2). The derivative of |Delta|^s with
# s = p - q < 1 is infinite at 0, and autodiff then returns 0 * inf = NaN; with EPS the derivative is
# finite everywhere and 0 at Delta = 0. At |Delta| >= 1e-6 the relative change of the energy is below 1e-12.
SMOOTH_EPS = 1e-12
P_EXPONENT = 2.0  # the exponent p of the q-GGMRF; only p = 2 is implemented (convexity holds for 1 <= q <= p)
DIAGONAL_WEIGHT = 1.0 / np.sqrt(2.0)


# ----------------------------------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------------------------------


class ThetaPrior(ABC):
    """A prior energy R(theta) on the n x n grid; see the module docstring for the interface.

    Subclasses implement ``_energy(theta)`` (a pure jnp function of an (n, n) array) and ``describe``;
    ``hessian`` has a generic (autodiff) fallback that subclasses override when a closed form is cheaper.
    """

    n: int
    mean: jnp.ndarray

    def _setup_jit(self):
        self.energy = jax.jit(self._energy)
        self.grad = jax.jit(jax.grad(self._energy))
        self.value_and_grad = jax.jit(jax.value_and_grad(self._energy))
        self._hvp = jax.jit(lambda theta, v: jax.jvp(jax.grad(self._energy), (theta,), (v,))[1])

    @abstractmethod
    def _energy(self, theta):
        """R(theta) for theta of shape (n, n): scalar."""

    @abstractmethod
    def describe(self):
        """Hyperparameters as a plain dict (JSON-serializable)."""

    def hvp(self, theta, v):
        """Hessian-vector product (nabla^2 R)(theta) v, shapes (n, n) -> (n, n)."""
        return self._hvp(jnp.asarray(theta, dtype=jnp.float64), jnp.asarray(v, dtype=jnp.float64))

    def hessian(self, theta):
        """Dense Hessian nabla^2 R(theta), shape (n^2, n^2), NumPy float64 (autodiff fallback)."""
        theta = jnp.asarray(theta, dtype=jnp.float64)
        flat = lambda x: self._energy(x.reshape(self.n, self.n))
        return np.asarray(jax.hessian(flat)(theta.ravel()))

    def _check(self, theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        if theta.shape != (self.n, self.n):
            raise ValueError(f"theta must have shape {(self.n, self.n)}, got {theta.shape}")
        return theta


def _as_mean(mu, n):
    return jnp.broadcast_to(jnp.asarray(mu, dtype=jnp.float64), (n, n))


# ----------------------------------------------------------------------------------------------
# q-GGMRF
# ----------------------------------------------------------------------------------------------


def rho(delta, q, T, sigma_g):
    """The q-GGMRF pair potential for p = 2:

        rho(D) = D^2 / (2 sigma_g^2) / (1 + |D / (T sigma_g)|^(2 - q)),

    quadratic for |D| << T sigma_g and growing like |D|^q / (2 sigma_g^2) (T sigma_g)^(2-q) beyond. |D|^(2-q) is
    evaluated as (D^2 + SMOOTH_EPS^2)^((2-q)/2), which makes value, gradient and Hessian finite at D = 0.
    Elementwise on arrays of any shape.
    """
    s = (delta * delta + SMOOTH_EPS**2) ** (0.5 * (P_EXPONENT - q))
    return delta * delta / (2.0 * sigma_g**2) / (1.0 + s / (T * sigma_g) ** (P_EXPONENT - q))


def neighbor_pairs(n):
    """Unordered 8-neighbor pairs of an n x n grid (free boundary): ``(i, j, b)``.

    ``i``, ``j`` are int arrays of flat C-order pixel indices (each unordered pair appears once, as {i, j}) and
    ``b`` the weights: 1 for horizontal and vertical pairs, 1/sqrt(2) for diagonal pairs. Number of pairs
    2 n (n - 1) + 2 (n - 1)^2; a corner pixel has 3 neighbors, an edge pixel 5, an interior pixel 8.
    """
    a, c = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    flat = a * n + c
    ii, jj, bb = [], [], []
    for da, dc, w in ((1, 0, 1.0), (0, 1, 1.0), (1, 1, DIAGONAL_WEIGHT), (1, -1, DIAGONAL_WEIGHT)):
        ok = (a + da < n) & (c + dc >= 0) & (c + dc < n)
        ii.append(flat[ok])
        jj.append(flat[np.clip(a + da, 0, n - 1), np.clip(c + dc, 0, n - 1)][ok])
        bb.append(np.full(ok.sum(), w))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(bb)


def pairwise_hessian(pairs, w, n, sigma_0):
    """Dense Hessian (n^2, n^2) of sum_{{i,j}} f_ij(theta_i - theta_j) + |theta|^2 / (2 sigma_0^2).

    ``pairs`` = ``neighbor_pairs(n)`` and ``w`` (number of pairs,) the second derivatives f_ij''(theta_i - theta_j) (weight
    b_ij included): H = sum_pairs w (e_i - e_j)(e_i - e_j)^T + I / sigma_0^2, the weighted graph Laplacian plus the anchor.
    Shared by ``QGGMRF`` and ``PseudoHuberTV``.
    """
    i, j, _ = pairs
    N = n * n
    H = np.zeros((N, N))
    np.add.at(H, (i, i), w)
    np.add.at(H, (j, j), w)
    np.add.at(H, (i, j), -w)
    np.add.at(H, (j, i), -w)
    H[np.diag_indices(N)] += 1.0 / sigma_0**2
    return H


class QGGMRF(ThetaPrior):
    """q-generalized Gaussian MRF (Thibault, Sauer, Bouman and Hsieh 2007; Bouman, Ch. 8), p = 2.

        R(theta) = sum_{{i,j}} b_ij rho(theta_i - theta_j) + |theta - mu|^2 / (2 sigma_0^2),

    the sum running over the unordered pairs {i, j} of 8-neighbors inside the DOI (free boundary, so no wrap
    and no pairs across the edge; ``neighbor_pairs``), b = 1 for horizontal and vertical pairs and 1/sqrt(2)
    for diagonal pairs, and ``rho`` the pair potential (quadratic below T sigma_g, ~ |D|^q above). Each unordered
    pair is counted once; counting ordered pairs would double R, equivalent to halving sigma_g^2.

    Parameters (all in log eps_r units): ``q`` in [1, 2] (the tail exponent; q = 2 is the Gaussian MRF
    with rho = D^2 / (4 sigma_g^2)), ``T`` > 0 (the transition is at |D| = T sigma_g), ``sigma_g`` > 0 (neighbor
    difference scale), ``sigma_0`` > 0 (weak anchor to the mean, makes the prior proper), ``mu`` (scalar or
    (n, n)). R is convex for 1 <= q <= 2 and p = 2; the Hessian is >= 1/sigma_0^2 times the identity.
    ``hessian`` is assembled in closed form, sum_pairs b rho''(D_ij) (e_i - e_j)(e_i - e_j)^T + I / sigma_0^2,
    with rho'' by autodiff of the scalar ``rho``.
    """

    def __init__(self, n, q=1.2, T=1.0, sigma_g=0.1, sigma_0=1.0, mu=0.0):
        if not 1.0 <= q <= 2.0:
            raise ValueError("q must be in [1, 2]")
        if not (T > 0 and sigma_g > 0 and sigma_0 > 0):
            raise ValueError("T, sigma_g and sigma_0 must be positive")
        self.n = int(n)
        self.q, self.T, self.sigma_g, self.sigma_0 = float(q), float(T), float(sigma_g), float(sigma_0)
        self.mean = _as_mean(mu, self.n)
        i, j, b = neighbor_pairs(self.n)
        self._i, self._j, self._b = jnp.asarray(i), jnp.asarray(j), jnp.asarray(b)
        self._pairs = (i, j, b)
        q_, T_, sg = self.q, self.T, self.sigma_g
        self._rho = lambda d: rho(d, q_, T_, sg)
        rho_dd = jax.vmap(jax.grad(jax.grad(lambda d: rho(d, q_, T_, sg))))
        self._weights = jax.jit(lambda theta: self._b * rho_dd(theta.ravel()[self._i] - theta.ravel()[self._j]))
        self._setup_jit()

    def _energy(self, theta):
        t = theta.ravel()
        d = t[self._i] - t[self._j]
        anchor = jnp.sum((theta - self.mean) ** 2) / (2.0 * self.sigma_0**2)
        return jnp.sum(self._b * self._rho(d)) + anchor

    def describe(self):
        return {"kind": "qggmrf", "n": self.n, "q": self.q, "T": self.T, "sigma_g": self.sigma_g,
                "sigma_0": self.sigma_0, "mu": float(np.mean(np.asarray(self.mean))), "p": P_EXPONENT,
                "neighborhood": 8, "boundary": "free", "pairs": "unordered, each pair once"}

    def hessian(self, theta):
        """Dense Hessian (n^2, n^2): closed form, exact up to the SMOOTH_EPS regularization of rho."""
        theta = self._check(theta)
        return pairwise_hessian(self._pairs, np.asarray(self._weights(theta)), self.n, self.sigma_0)


# ----------------------------------------------------------------------------------------------
# Pseudo-Huber total variation
# ----------------------------------------------------------------------------------------------


def pseudo_huber(delta, eps):
    """The pseudo-Huber function sqrt(D^2 + eps^2) - eps, elementwise, evaluated as D^2 / (sqrt(D^2 + eps^2) + eps).

    The two forms are equal; the second has no cancellation for |D| << eps (the first loses all digits below
    |D| ~ 1e-8 eps) and is smooth at D = 0 for eps > 0 (value 0, derivative 0, second derivative 1/eps).
    It is quadratic, D^2 / (2 eps), for |D| << eps and |D| - eps + eps^2 / (2 |D|) + ... for |D| >> eps.
    """
    return delta * delta / (jnp.sqrt(delta * delta + eps * eps) + eps)


def pseudo_huber_second_derivative(delta, eps):
    """d^2/dD^2 of ``pseudo_huber`` = eps^2 / (D^2 + eps^2)^(3/2), in (0, 1/eps], elementwise."""
    return eps**2 / (delta * delta + eps * eps) ** 1.5


class PseudoHuberTV(ThetaPrior):
    """Pseudo-Huber total-variation prior (docs/DESIGN.md section 12.1), same neighborhood and weights as ``QGGMRF``.

        R(theta) = lam sum_{{i,j}} b_ij (sqrt(D_ij^2 + delta^2) - delta) + |theta - mu|^2 / (2 sigma_0^2),
        D_ij = theta_i - theta_j,

    the sum over the unordered 8-neighbor pairs of the DOI with free boundary (``neighbor_pairs``: b = 1 horizontal and
    vertical, 1/sqrt(2) diagonal; each pair once). Convex (a sum of convex functions of linear forms), smooth for
    delta > 0, with pair Hessian lam b delta^2 / (D^2 + delta^2)^(3/2) in (0, lam b / delta], so ``hessian`` is
    >= I / sigma_0^2 and the Laplace mass of ``thetaspace`` stays symmetric positive definite.

    For |D| >> delta the pair cost is lam (|D| - delta): total variation with weight lam, for which a step costs the same
    as any monotone ramp of the same height. At finite delta a ramp of k equal increments D / k is cheaper than the step by
    about (k - 1) delta lam per pair line (first order in delta / D), so the step-over-ramp cost ratio is
    1 + (k - 1) delta / D + O(delta^2): 1.10 for k = 8, D = log 2, delta = 0.01 (see ``scripts/prior_energy_audit.py``;
    1.03 needs delta ~ 0.003 and 1.01 delta ~ 0.001). Units: lam is the TV weight per unit change of theta = log eps_r, delta
    the width of the quadratic core in the same units, sigma_0 the weak anchor to the mean ``mu`` (scalar or (n, n)).
    The pair value uses ``pseudo_huber`` (no cancellation at small D). ``hvp`` is by autodiff, ``hessian`` in closed form.
    """

    def __init__(self, n, lam=1.0, delta=0.01, sigma_0=1.0, mu=0.0):
        if not (lam > 0 and delta > 0 and sigma_0 > 0):
            raise ValueError("lam, delta and sigma_0 must be positive")
        self.n = int(n)
        self.lam, self.delta, self.sigma_0 = float(lam), float(delta), float(sigma_0)
        self.mean = _as_mean(mu, self.n)
        i, j, b = neighbor_pairs(self.n)
        self._i, self._j, self._b = jnp.asarray(i), jnp.asarray(j), jnp.asarray(b)
        self._pairs = (i, j, b)
        lam_, d_ = self.lam, self.delta
        self._weights = jax.jit(lambda theta: lam_ * self._b * pseudo_huber_second_derivative(
            theta.ravel()[self._i] - theta.ravel()[self._j], d_))
        self._setup_jit()

    def _energy(self, theta):
        t = theta.ravel()
        d = t[self._i] - t[self._j]
        anchor = jnp.sum((theta - self.mean) ** 2) / (2.0 * self.sigma_0**2)
        return self.lam * jnp.sum(self._b * pseudo_huber(d, self.delta)) + anchor

    def describe(self):
        return {"kind": "tv", "n": self.n, "lam": self.lam, "delta": self.delta, "sigma_0": self.sigma_0,
                "mu": float(np.mean(np.asarray(self.mean))), "neighborhood": 8, "boundary": "free",
                "pairs": "unordered, each pair once", "potential": "lam b (sqrt(D^2 + delta^2) - delta)"}

    def hessian(self, theta):
        """Dense Hessian (n^2, n^2) in closed form: sum_pairs lam b delta^2 / (D^2 + delta^2)^(3/2) (e_i - e_j)(e_i - e_j)^T
        + I / sigma_0^2."""
        theta = self._check(theta)
        return pairwise_hessian(self._pairs, np.asarray(self._weights(theta)), self.n, self.sigma_0)


# ----------------------------------------------------------------------------------------------
# Cropped Gaussian fields
# ----------------------------------------------------------------------------------------------


class GaussianEnergy(ThetaPrior):
    """R(theta) = 1/2 (theta - mu)^T C_D^-1 (theta - mu) for the DOI marginal C_D of a ``GaussianField``.

    C_D = ``field.covariance_matrix()`` is the exact covariance of the cropped torus field (n^2 x n^2 dense,
    DESIGN 11.2), Cholesky-factored once (``chol``, lower, C_D = chol chol^T). The energy is exactly minus the log
    of the N(mu, C_D) density up to the constant 1/2 log det(2 pi C_D) (``log_norm_const``), with the quadratic
    form evaluated through the precision matrix P = C_D^-1 (``precision``, from the Cholesky factor), so
    ``grad`` = P (theta - mu) and ``hessian`` = P for every theta. The marginal std of theta is ``field.sigma`` at
    every pixel. Needs ``field.covariance_matrix()``, which builds an (n^2, P^2) factor: n up to 48 with
    pad = 2 (about 3 s at n = 48).

    Raises ``numpy.linalg.LinAlgError`` if C_D is not numerically positive definite.
    """

    def __init__(self, field):
        self.field = field
        self.n = int(field.n)
        self.mean = _as_mean(field.mean, self.n)
        C = np.asarray(field.covariance_matrix(), dtype=np.float64)
        C = 0.5 * (C + C.T)
        self.chol = cholesky(C, lower=True)
        P = cho_solve((self.chol, True), np.eye(C.shape[0]))
        self.precision = 0.5 * (P + P.T)
        self.log_norm_const = float(np.sum(np.log(np.diag(self.chol))) + 0.5 * C.shape[0] * np.log(2.0 * np.pi))
        self._P = jnp.asarray(self.precision)
        self._setup_jit()

    def _energy(self, theta):
        d = (theta - self.mean).ravel()
        return 0.5 * d @ (self._P @ d)

    def describe(self):
        f = self.field
        return {"kind": f.kind, "n": self.n, "sigma": f.sigma, "ell": float(f.ell), "nu": f.nu,
                "kappa": f.kappa, "mean": float(np.mean(np.asarray(self.mean))), "pad": f.pad,
                "form": "DOI marginal of the torus field (dense covariance, Cholesky)"}

    def hessian(self, theta):
        return self.precision.copy()

    def prior_std(self):
        """Exact marginal prior std of theta, (n, n): sqrt(diag C_D) = field.sigma (stationary field)."""
        return np.sqrt(np.sum(self.chol**2, axis=1)).reshape(self.n, self.n)

    def sample(self, key, num=None):
        """Exact prior draws mu + chol z: shape (n, n), or (num, n, n)."""
        N = self.n * self.n
        z = jax.random.normal(key, (N,) if num is None else (num, N), dtype=jnp.float64)
        th = z @ jnp.asarray(self.chol).T  # rows: chol z
        return th.reshape((self.n, self.n) if num is None else (num, self.n, self.n)) + self.mean


# ----------------------------------------------------------------------------------------------
# Affine whitening and the prior-only sampler
# ----------------------------------------------------------------------------------------------


class WhitenedMap:
    """The affine map theta = center + L^-T zeta, with H = L L^T (``chol`` = L, lower triangular, (n^2, n^2)).

    If theta ~ N(center, H^-1) then zeta ~ N(0, I). ``to_theta(zeta)`` and ``to_zeta(theta)`` act on (n, n)
    arrays (triangular solves, so no explicit inverse is formed) and ``grad_to_zeta(g)`` = L^-1 g maps a gradient
    with respect to theta to the gradient with respect to zeta (chain rule of the affine map).
    """

    def __init__(self, center, chol):
        center = jnp.asarray(center, dtype=jnp.float64)
        self.n = int(center.shape[0])
        self.center = center
        self.chol = chol if isinstance(chol, jax.Array) else jnp.asarray(chol, dtype=jnp.float64)
        n, c, L = self.n, center.ravel(), self.chol
        self.to_theta = jax.jit(lambda z: (c + solve_triangular(L, z.ravel(), lower=True, trans=1)).reshape(n, n))
        self.to_zeta = jax.jit(lambda th: (L.T @ (th.ravel() - c)).reshape(n, n))
        self.grad_to_zeta = jax.jit(lambda g: solve_triangular(L, g.ravel(), lower=True).reshape(n, n))


def prior_potential_fn(prior, wmap):
    """``phi_fn(zeta, aux) -> (phi', grad, aux)`` of the prior alone in the coordinates of ``wmap``:
    phi'(zeta) = R(theta(zeta)) - |zeta|^2 / 2, so that the samplers of ``samplers`` (which add |zeta|^2 / 2)
    target exp(-R(theta)). ``aux`` is passed through unchanged."""
    vg = prior.value_and_grad

    @jax.jit
    def phi_fn(zeta, aux):
        e, g = vg(wmap.to_theta(zeta))
        return e - 0.5 * jnp.sum(zeta * zeta), wmap.grad_to_zeta(g) - zeta, aux

    return phi_fn


def sample_prior(prior, key, *, n_chains=4, n_warmup=300, n_samples=1000, thin=1, n_leapfrog=10, jitter=0.2,
                 step0=0.5, init_scale=1.0, center=None):
    """HMC on the prior energy alone, to estimate the prior mean and pixel standard deviation of theta and eps_r.

    The chain runs in zeta with theta = center + L^-T zeta, H = L L^T the Hessian of R at ``center`` (default the
    prior mean), which is exact whitening for a Gaussian prior (then phi' = 0 and zeta ~ N(0, I)) and a good
    preconditioner for the q-GGMRF, whose Hessian at the mean is the stiffest it gets (rho'' is largest at
    D = 0 for every q in [1, 2] and T, sigma_g tried, so the tails are softer than the Gaussian map assumes). Starts: center + init_scale * L^-T N(0, I). Step size adapted by the sampler's
    Robbins-Monro warmup toward acceptance 0.7.

    Returns a dict: ``theta_mean``, ``theta_std``, ``eps_mean``, ``eps_std`` ((n, n), pooled Welford moments over
    all post-warmup steps and chains), ``theta_samples`` ((chains, N // thin, n, n) float32), ``accept`` (mean),
    ``step`` (per chain), ``rhat_max`` (split R-hat over pixels of the thinned theta), ``ess_min`` (worst-pixel
    Sokal ESS summed over chains) and ``n_grad_evals`` (per chain). eps_r = exp(theta).
    """
    n = prior.n
    center = prior.mean if center is None else jnp.asarray(center, dtype=jnp.float64)
    H = prior.hessian(center)
    wmap = WhitenedMap(center, cholesky(H, lower=True))
    phi_fn = prior_potential_fn(prior, wmap)

    def record(zeta):
        th = wmap.to_theta(zeta)
        return jnp.stack([th, jnp.exp(th)])

    k_init, k_run = jax.random.split(key)
    z0 = init_scale * jax.random.normal(k_init, (n_chains, n, n), dtype=jnp.float64)
    keys = jax.random.split(k_run, n_chains)
    aux0 = jnp.zeros(())
    run = functools.partial(sm.hmc, phi_fn=phi_fn, n_warmup=int(n_warmup), n_samples=int(n_samples),
                            step0=step0, thin=int(thin), record_fn=record, n_leapfrog=int(n_leapfrog),
                            jitter=jitter)
    outs = [run(keys[c], xi0=z0[c], aux0=aux0) for c in range(n_chains)]
    mean = np.stack([np.asarray(o.mean) for o in outs])  # (chains, 2, n, n)
    var = np.stack([np.asarray(o.var) for o in outs])
    samples = np.stack([np.asarray(o.samples[:, 0]) for o in outs])  # (chains, N / thin, n, n) theta
    pooled_var = var.mean(axis=0) + mean.var(axis=0)
    flat = samples.reshape(n_chains, samples.shape[1], -1)
    rhat = dg.split_rhat(flat) if n_chains >= 1 and flat.shape[1] >= 8 else np.full(flat.shape[2], np.nan)
    ess = np.stack([dg.ess(c) for c in flat]).sum(axis=0)
    return {"theta_mean": mean.mean(axis=0)[0], "theta_std": np.sqrt(pooled_var[0]),
            "eps_mean": mean.mean(axis=0)[1], "eps_std": np.sqrt(pooled_var[1]),
            "theta_samples": samples.astype(np.float32),
            "accept": float(np.mean([np.asarray(o.accept).mean() for o in outs])),
            "step": [float(o.step) for o in outs], "rhat_max": float(np.nanmax(rhat)),
            "ess_min": float(np.nanmin(ess)), "n_grad_evals": 1 + (n_warmup + n_samples) * n_leapfrog}
