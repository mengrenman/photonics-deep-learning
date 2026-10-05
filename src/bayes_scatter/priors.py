"""Gaussian-field priors in whitened coordinates, and links from field to contrast.

Conventions (docs/DESIGN.md, sections 1 and 3):

* The inference parameter is theta = log eps_r, so eps_r = exp(theta) is log-normal at each pixel. The
  prior on theta is a stationary Gaussian field with mean mu (mu = 0 is the vacuum background).
* The DOI grid has n x n pixels of side h = side / n, indexed [i, j] = [x, y], flattened in C order.
* Inference runs in whitened coordinates xi ~ N(0, I) with ``xi.shape == (P, P)``, P = pad * n.
* theta = mu + crop(Re ifft2(sqrt(S) * fft2(xi))), the crop being the first n x n entries. S is a real,
  even spectral density on the P x P torus, so the filter is a real symmetric circulant and the torus
  covariance is the circulant whose eigenvalues are S. Its marginal variance is mean(S), and S is
  rescaled so that mean(S) = sigma^2. The cropped covariance is exactly stationary (a sub-block of a
  circulant), and its marginal variance is sigma^2 at every pixel.
* chi = link(theta) is the contrast eps_r - 1 that the forward model consumes. Exactly two links exist:
  ``log_eps`` (default; chi = exp(theta) - 1, inverse theta = log(1 + chi)) and ``identity``
  (chi = theta; for linear-Gaussian validation only, where it makes xi -> chi affine).

Everything that depends on xi is plain jax.numpy, so it runs under jit, vmap and grad, and accepts a
batch of xi of shape (..., P, P). The small-grid helpers ``sqrt_matrix`` and ``covariance_matrix``
return NumPy arrays and are meant for tests and the exact Born posterior only.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

_KINDS = ("matern", "gmrf")


def _grid_n_and_h(grid):
    """Return (n, h) from a grid with attributes ``n`` and ``h`` (or ``side``, h = side / n)."""
    n = int(grid.n)
    h = getattr(grid, "h", None)
    if h is None:
        h = float(grid.side) / n
    return n, float(h)


class GaussianField:
    """Stationary Gaussian prior on theta = log eps_r over the DOI grid, sampled by FFT on a padded torus.

    Parameters
    ----------
    grid : object with ``n`` and ``h`` (or ``side``); the DOI grid (n x n pixels, spacing h).
    kind : ``"matern"`` or ``"gmrf"``.
        ``"matern"``: S(k) ~ (kappa^2 + |k|^2)^(-(nu + 1)), the 2-D Matern spectrum.
        ``"gmrf"``: S(k) ~ 1 / (kappa^2 + lam_k), lam_k = (4 - 2cos(kx h) - 2cos(ky h)) / h^2, the
        discrete-Laplacian eigenvalues. This is the Gaussian Markov random field with precision
        proportional to kappa^2 I - Laplacian on the torus (nearest-neighbor, 5-point stencil).
    sigma : marginal standard deviation of theta - mu at every pixel, in natural-log units of eps_r
        (sigma = 0.5 is a typical factor exp(0.5) = 1.65 in eps_r).
    ell : correlation length in wavelengths (units of ``h``). For ``"matern"`` it defines
        kappa = sqrt(2 nu) / ell, so the correlation at distance ell is a fixed function of nu alone
        (about 0.37, 0.48, 0.52 for nu = 0.5, 1.5, 2.5). For ``"gmrf"`` it defines kappa = 1 / ell
        (the GMRF has no free smoothness parameter). Ignored if ``kappa`` is given.
    nu : Matern smoothness (``"matern"`` only; ignored for ``"gmrf"``).
    kappa : inverse length scale; overrides the value derived from ``ell``.
    mean : scalar or array broadcastable to (n, n); the prior mean mu of theta = log eps_r (0 = vacuum).
    pad : torus size is P = pad * n; ``pad = 2`` keeps wrap-around correlation negligible when
        ell is small compared with the DOI side.

    Attributes
    ----------
    n, h, P, kappa, ell, nu, sigma, mean (n, n), wavenumbers (P,), spectrum (P, P).
    ``spectrum`` is the normalized density S in FFT order (index 0 is k = 0), with mean(S) = sigma^2.
    """

    def __init__(self, grid, kind="matern", sigma=1.0, ell=0.5, nu=1.5, kappa=None, mean=0.0, pad=2):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        if not sigma > 0:
            raise ValueError("sigma must be positive")
        if int(pad) != pad or pad < 1:
            raise ValueError("pad must be an integer >= 1")
        self.grid = grid
        self.kind = kind
        self.sigma = float(sigma)
        self.pad = int(pad)
        self.n, self.h = _grid_n_and_h(grid)
        self.P = self.pad * self.n

        if kind == "matern":
            if not nu > 0:
                raise ValueError("nu must be positive")
            self.nu = float(nu)
            if kappa is None:
                if not ell > 0:
                    raise ValueError("ell must be positive")
                kappa = np.sqrt(2.0 * self.nu) / ell
            self.kappa = float(kappa)
            self.ell = np.sqrt(2.0 * self.nu) / self.kappa
        else:
            self.nu = None
            if kappa is None:
                if not ell > 0:
                    raise ValueError("ell must be positive")
                kappa = 1.0 / ell
            self.kappa = float(kappa)
            self.ell = 1.0 / self.kappa
        if not self.kappa > 0:
            raise ValueError("kappa must be positive")

        # Continuous wavenumbers of the P x P torus with spacing h (FFT order).
        k = 2.0 * np.pi * np.fft.fftfreq(self.P, d=self.h)
        kx, ky = np.meshgrid(k, k, indexing="ij")
        if kind == "matern":
            s = (self.kappa**2 + kx**2 + ky**2) ** (-(self.nu + 1.0))
        else:
            lam = (4.0 - 2.0 * np.cos(kx * self.h) - 2.0 * np.cos(ky * self.h)) / self.h**2
            s = 1.0 / (self.kappa**2 + lam)
        s = s * (self.sigma**2 / s.mean())

        self.wavenumbers = jnp.asarray(k)
        self.spectrum = jnp.asarray(s)
        # Half spectrum for rfft2/irfft2. The input xi is real and S is real and even, so
        # Re ifft2(sqrt(S) fft2(xi)) equals irfft2(sqrt(S) rfft2(xi)) exactly (up to round-off).
        self._sqrt_s_half = jnp.asarray(np.sqrt(s)[:, : self.P // 2 + 1])
        self.mean = jnp.broadcast_to(jnp.asarray(mean, dtype=jnp.float64), (self.n, self.n))

    def __repr__(self):
        extra = f", nu={self.nu}" if self.kind == "matern" else ""
        return (f"GaussianField(kind={self.kind!r}, n={self.n}, P={self.P}, sigma={self.sigma}, "
                f"ell={self.ell:.4g}, kappa={self.kappa:.4g}{extra})")

    @property
    def xi_shape(self):
        """Shape (P, P) of one whitened-coordinate vector xi."""
        return (self.P, self.P)

    def torus_field(self, xi):
        """Uncropped C^(1/2) xi on the P x P torus (no mean). Shape (..., P, P) -> (..., P, P), real."""
        xi = jnp.asarray(xi, dtype=jnp.float64)
        if xi.shape[-2:] != self.xi_shape:
            raise ValueError(f"xi must have trailing shape {self.xi_shape}, got {xi.shape}")
        spec = jnp.fft.rfft2(xi, axes=(-2, -1))
        return jnp.fft.irfft2(self._sqrt_s_half * spec, s=self.xi_shape, axes=(-2, -1))

    def theta(self, xi):
        """Field theta = mu + crop(C^(1/2) xi). Shape (..., P, P) -> (..., n, n), real."""
        n = self.n
        return self.mean + self.torus_field(xi)[..., :n, :n]

    def sample_xi(self, key, num=None):
        """Draw xi ~ N(0, I): shape (P, P), or (num, P, P) if ``num`` is given."""
        shape = self.xi_shape if num is None else (num,) + self.xi_shape
        return jax.random.normal(key, shape, dtype=jnp.float64)

    def sample(self, key, num=None):
        """Draw prior samples of theta: shape (n, n), or (num, n, n) if ``num`` is given."""
        return self.theta(self.sample_xi(key, num))

    def sqrt_matrix(self, max_elements=int(5e7)):
        """Matrix L of shape (n*n, P*P) with theta.ravel() = mean.ravel() + L @ xi.ravel(). NumPy.

        Built from the circulant kernel m = Re ifft2(sqrt(S)): the response of the torus filter to a unit
        impulse at the origin, so L[(i, j), (c, d)] = m[(i - c) mod P, (j - d) mod P]. For small grids only.
        """
        n, P = self.n, self.P
        if n * n * P * P > max_elements:
            raise ValueError(f"L would have {n * n * P * P} entries (> {max_elements}); use a smaller grid")
        m = np.fft.ifft2(np.sqrt(np.asarray(self.spectrum))).real
        idx = (np.arange(n)[:, None] - np.arange(P)[None, :]) % P  # (n, P)
        L = m[idx[:, None, :, None], idx[None, :, None, :]]  # (n, n, P, P)
        return L.reshape(n * n, P * P)

    def covariance_matrix(self):
        """Covariance L L^T of theta, shape (n*n, n*n), C-order pixel index. NumPy; small grids only."""
        L = self.sqrt_matrix()
        return L @ L.T


# ----------------------------------------------------------------------------------------------
# Links chi = link(theta). Elementwise, jittable, differentiable; ``inverse`` maps chi back to theta.
# Exactly two exist (docs/DESIGN.md section 3); chi is the contrast eps_r - 1 used by the forward model.
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogEps:
    """chi = exp(theta) - 1 with theta = log eps_r, so chi > -1 and dchi/dtheta = exp(theta) = eps_r.

    ``inverse(chi) = log(1 + chi)`` needs chi > -1 (it returns -inf at -1 and NaN below). No clipping is
    applied: exp overflows only for theta > 709, far outside any sensible prior.
    """

    def __call__(self, theta):
        return jnp.exp(theta) - 1.0

    def inverse(self, chi):
        return jnp.log1p(chi)


@dataclass(frozen=True)
class Identity:
    """chi = theta. Validation only: the exact Born posterior is Gaussian when xi -> chi is affine."""

    def __call__(self, theta):
        return jnp.asarray(theta)

    def inverse(self, chi):
        return jnp.asarray(chi)


log_eps = LogEps()
identity = Identity()
LINKS = {"log_eps": log_eps, "identity": identity}


def get_link(link):
    """Resolve a link given by name ("log_eps" or "identity") or as one of the two link objects."""
    if isinstance(link, str):
        if link not in LINKS:
            raise ValueError(f"unknown link {link!r}; the links are {sorted(LINKS)}")
        return LINKS[link]
    if not (callable(link) and callable(getattr(link, "inverse", None))):
        raise ValueError("a link needs a forward map (__call__) and an ``inverse`` method")
    return link


def compose(field, link=log_eps):
    """Return the map chi(xi) = link(field.theta(xi)); shape (..., P, P) -> (..., n, n), jittable.

    ``link`` defaults to ``log_eps`` (chi = exp(theta) - 1); pass ``identity`` for linear-Gaussian tests.
    """
    link = get_link(link)

    def chi(xi):
        return link(field.theta(xi))

    return chi
