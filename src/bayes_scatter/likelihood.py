"""Noise model, data misfit and the posterior ``Target`` (docs/DESIGN.md section 4).

Data are complex arrays d[t, r] of shape (Nt, Nr). Noise is circular complex Gaussian, d_obs = d + e with
e ~ CN(0, s^2) i.i.d.: real and imaginary parts are independent N(0, s^2/2). The noise level comes from a
power SNR in dB, s^2 = mean|d_clean|^2 * 10^(-SNR/10).

The potential (negative log-likelihood up to an additive constant) is

    Phi(xi) = sum_{t,r} |F(chi(xi))[t, r] - d_obs[t, r]|^2 / s^2,

with no factor 1/2, because each complex entry carries two real Gaussians of variance s^2/2. The posterior
in whitened coordinates is pi(xi) ~ exp(-Phi(xi)) N(xi; 0, I), the target of every sampler in
``bayes_scatter.samplers``.

``Target.phi_fn(xi, aux) -> (phi, grad_phi, aux_new)`` is the callable the samplers expect. ``aux`` is the
(Nt, n, n) array of total fields from the last solve; it warm-starts the next Krylov solve and is a pure
accelerator, so Phi depends on it only through the solver tolerance (use ``tol <= 1e-10`` in the forward
model when sampling, DESIGN 2.4). With the Born forward model no solve takes place and ``aux`` is passed
through unchanged.
"""

import jax
import jax.numpy as jnp

from .priors import compose, get_link


def noise_std_from_snr(d_clean, snr_db):
    """Noise level s (a Python float) for a power SNR in dB: s^2 = mean|d_clean|^2 * 10^(-snr_db/10).

    ``d_clean`` is the noise-free data of any shape; the mean runs over all entries. Then
    E|e|^2 = s^2 for e ~ CN(0, s^2), so 10 log10(mean|d|^2 / E|e|^2) = snr_db.
    """
    power = float(jnp.mean(jnp.abs(jnp.asarray(d_clean)) ** 2))
    return float((power * 10.0 ** (-float(snr_db) / 10.0)) ** 0.5)


def add_noise(key, d_clean, s):
    """d_clean + e with e ~ CN(0, s^2) i.i.d.: Re e and Im e independent N(0, s^2/2). Shape of d_clean."""
    d_clean = jnp.asarray(d_clean, dtype=jnp.complex128)
    k_re, k_im = jax.random.split(key)
    shape = d_clean.shape
    e = jax.random.normal(k_re, shape) + 1j * jax.random.normal(k_im, shape)
    return d_clean + (s / jnp.sqrt(2.0)) * e


def misfit(d_model, d_obs, s):
    """Phi = sum |d_model - d_obs|^2 / s^2, the negative log-likelihood up to the constant N log(pi s^2).

    N is the number of complex entries. Written with real and imaginary parts so that the gradient is
    finite at zero residual.
    """
    r = jnp.asarray(d_model) - jnp.asarray(d_obs)
    return jnp.sum(jnp.real(r) ** 2 + jnp.imag(r) ** 2) / s**2


class Target:
    """Posterior in whitened coordinates for one data set.

    Parameters
    ----------
    forward : ``bayes_scatter.forward.Forward``.
    field : ``bayes_scatter.priors.GaussianField`` on the same grid; xi has shape ``field.xi_shape``.
    link : ``"log_eps"`` (default for inference, chi = e^theta - 1 with theta = log eps_r), ``"identity"``
        (chi = theta; linear-Gaussian validation only), or a link object from ``bayes_scatter.priors``.
    d_obs : observed data (Nt, Nr), complex.
    s : noise level, E|e|^2 = s^2 (see ``noise_std_from_snr``).
    use_born : if True the forward map is the Born approximation d = A chi (no linear solve); the data may
        still have been generated with the full model.

    Methods are jitted and take/return JAX arrays. ``potential``, ``potential_and_grad`` and ``phi_fn`` are
    deterministic in xi up to the forward solver tolerance, whatever ``aux`` is.
    """

    def __init__(self, forward, field, link, d_obs, s, use_born=False):
        if field.n != forward.n:
            raise ValueError(f"field grid n={field.n} does not match forward grid n={forward.n}")
        d_obs = jnp.asarray(d_obs, dtype=jnp.complex128)
        if d_obs.shape != (forward.n_tx, forward.n_rx):
            raise ValueError(f"d_obs must have shape {(forward.n_tx, forward.n_rx)}, got {d_obs.shape}")
        if not s > 0:
            raise ValueError("noise level s must be positive")
        self.forward = forward
        self.field = field
        self.link = get_link(link)
        self.d_obs = d_obs
        self.s = float(s)
        self.use_born = bool(use_born)
        self._chi = compose(field, self.link)

        def phi_of_chi(chi, aux):
            if self.use_born:
                return misfit(forward.born(chi), d_obs, self.s), aux
            d, u = forward(chi, aux)
            return misfit(d, d_obs, self.s), u

        def phi_and_field(xi, aux):
            return phi_of_chi(self._chi(xi), aux)

        def potential(xi, aux):
            phi, u = phi_and_field(xi, aux)
            return phi, jax.lax.stop_gradient(u)

        vg = jax.value_and_grad(phi_and_field, has_aux=True)

        def potential_and_grad(xi, aux):
            (phi, u), grad = vg(xi, aux)
            return phi, grad, jax.lax.stop_gradient(u)

        def misfit_of_chi(chi, aux):
            phi, u = phi_of_chi(chi, aux)
            return phi, jax.lax.stop_gradient(u)

        def potential_only(xi, aux):  # gradient-free variant for pCN: (phi, None, aux_new)
            phi, u = potential(xi, aux)
            return phi, None, u

        self._chi_jit = jax.jit(self._chi)
        self._misfit_of_chi = jax.jit(misfit_of_chi)
        self._potential = jax.jit(potential)
        self._potential_and_grad = jax.jit(potential_and_grad)
        self._potential_only = jax.jit(potential_only)
        self._residual = jax.jit(lambda xi, u: forward.residual(self._chi(xi), u))

    # ---- maps from xi to the physical fields ----

    def chi(self, xi):
        """Contrast chi = eps_r - 1 of shape (n, n) (batched xi allowed: (..., P, P) -> (..., n, n))."""
        return self._chi_jit(xi)

    def theta(self, xi):
        """Field theta = log eps_r (``log_eps`` link) of shape (n, n)."""
        return self.field.theta(xi)

    def eps_r(self, xi):
        """Relative permittivity eps_r = 1 + chi of shape (n, n)."""
        return 1.0 + self._chi_jit(xi)

    def record(self, xi):
        """Default ``record_fn`` for the samplers: array (2, n, n) stacking theta = log eps_r and eps_r."""
        return jnp.stack([self.field.theta(xi), 1.0 + self._chi(xi)])

    # ---- potential ----

    def initial_aux(self):
        """Warm start for the first solve: the incident field u_inc (Nt, n, n), i.e. the Born total field.

        It is the exact total field for chi = 0 and a better start than zero for small contrast, and it
        has the same shape and dtype as every later aux (the total field of the last solve).
        """
        return self.forward.u_inc

    def potential(self, xi, aux):
        """Returns ``(phi, aux_new)``: the potential Phi(xi) and the total fields of this solve."""
        return self._potential(xi, aux)

    def potential_and_grad(self, xi, aux):
        """Returns ``(phi, grad_phi, aux_new)``; grad_phi has the shape of xi (adjoint gradient)."""
        return self._potential_and_grad(xi, aux)

    @property
    def phi_fn(self):
        """The callable ``(xi, aux) -> (phi, grad_phi, aux_new)`` that the samplers take."""
        return self._potential_and_grad

    @property
    def phi_fn_no_grad(self):
        """``(xi, aux) -> (phi, None, aux_new)``: for pCN, which never reads the gradient."""
        return self._potential_only

    def misfit_of_chi(self, chi, aux):
        """Returns ``(phi, aux_new)``: the data misfit Phi of a contrast map chi (n, n), whatever coordinates
        or reparametrization produced it. Same forward model, data and noise level as ``potential``, so
        ``misfit_of_chi(target.chi(xi), aux)`` equals ``potential(xi, aux)`` (this is how Laplace-whitened runs
        recover the physical Phi from stored draws, see scripts/run_posterior_laplace.py)."""
        return self._misfit_of_chi(jnp.asarray(chi), aux)

    def solver_residual(self, xi, aux):
        """Largest relative true residual of the field solve at xi, given the fields ``aux`` it returned.

        Diagnostic: a value well above ``forward.tol`` means a solve hit its iteration cap. With the Born
        model (no solve) it is not meaningful and 0 is returned.
        """
        if self.use_born:
            return 0.0
        return float(jnp.max(self._residual(xi, aux)))
