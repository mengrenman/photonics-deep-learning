"""Tests for bayes_scatter.priors: whitened Gaussian-field priors and links (docs/DESIGN.md section 3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma, kv

from bayes_scatter import priors
from bayes_scatter.priors import (
    LINKS,
    GaussianField,
    compose,
    get_link,
    identity,
    log_eps,
)


@dataclass(frozen=True)
class _Grid:
    """Minimal stand-in for geometry.Grid: n x n pixels covering a square of the given side."""

    n: int
    side: float
    center: tuple = (0.0, 0.0)

    @property
    def h(self):
        return self.side / self.n


@dataclass(frozen=True)
class _GridNoH:
    n: int
    side: float


SMALL = _Grid(n=12, side=3.0)  # h = 0.25, P = 24 with pad = 2
FINE = _Grid(n=24, side=6.0)  # h = 0.25, P = 48: correlation-shape tests
KINDS = ["matern", "gmrf"]


def make(kind, grid=SMALL, **kw):
    kw.setdefault("sigma", 1.7)
    kw.setdefault("ell", 0.75)
    return GaussianField(grid, kind=kind, **kw)


def matern_corr(r, ell, nu):
    """Analytic Matern correlation with kappa = sqrt(2 nu) / ell."""
    x = np.sqrt(2.0 * nu) * r / ell
    return 2.0 ** (1.0 - nu) / gamma(nu) * x**nu * kv(nu, x)


# ---------------------------------------------------------------------------------------------
# (1) theta = mean + L xi, real
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_theta_real_and_equals_mean_plus_L_xi(kind):
    rng = np.random.default_rng(0)
    # Non-constant, non-symmetric mean: catches [x, y] vs [y, x] mix-ups and a missing mean.
    mean = rng.normal(size=(SMALL.n, SMALL.n))
    f = make(kind, mean=mean)
    assert f.xi_shape == (24, 24)
    L = f.sqrt_matrix()
    assert L.shape == (SMALL.n**2, f.P**2)
    for _ in range(3):
        xi = rng.normal(size=f.xi_shape)
        th = f.theta(xi)
        assert th.shape == (SMALL.n, SMALL.n)
        assert th.dtype == jnp.float64  # real, not complex
        ref = (mean.ravel() + L @ xi.ravel()).reshape(SMALL.n, SMALL.n)
        assert np.max(np.abs(np.asarray(th) - ref)) < 1e-12
        # theta(xi) must be non-trivial: the check above would pass vacuously for xi = 0.
        assert np.std(ref - mean) > 0.5


@pytest.mark.parametrize("kind", KINDS)
def test_sqrt_matrix_matches_literal_design_formula(kind):
    """theta - mean = crop(Re ifft2(sqrt(S) fft2(xi))) evaluated literally from the spectrum."""
    f = make(kind)
    xi = np.random.default_rng(1).normal(size=f.xi_shape)
    S = np.asarray(f.spectrum)
    full = np.fft.ifft2(np.sqrt(S) * np.fft.fft2(xi))
    assert np.max(np.abs(full.imag)) < 1e-14  # real, even filter: exactly real up to round-off
    ref = full.real[: SMALL.n, : SMALL.n]
    assert np.max(np.abs(np.asarray(f.theta(xi)) - ref)) < 1e-12


def test_xi_zero_gives_mean_and_scalar_mean_broadcasts():
    f = make("matern", mean=0.3)
    assert f.mean.shape == (SMALL.n, SMALL.n)
    assert np.all(np.asarray(f.theta(np.zeros(f.xi_shape))) == 0.3)


def test_theta_batched_jit_vmap_and_gradient():
    rng = np.random.default_rng(2)
    f = make("matern", mean=0.1)
    xib = rng.normal(size=(3, 2) + f.xi_shape)
    thb = f.theta(xib)
    assert thb.shape == (3, 2, SMALL.n, SMALL.n)
    for a in range(3):
        for b in range(2):
            assert np.max(np.abs(np.asarray(thb[a, b]) - np.asarray(f.theta(xib[a, b])))) < 1e-13
    assert np.max(np.abs(np.asarray(jax.jit(f.theta)(xib) - thb))) < 1e-13
    assert np.max(np.abs(np.asarray(jax.vmap(f.theta)(xib[:, 0]) - thb[:, 0]))) < 1e-13

    # d/dxi sum(w * theta) = L^T w (the field operator is symmetric, so this tests the FFT adjoint).
    w = rng.normal(size=(SMALL.n, SMALL.n))
    xi = rng.normal(size=f.xi_shape)
    g = jax.grad(lambda x: jnp.sum(w * f.theta(x)))(xi)
    ref = (f.sqrt_matrix().T @ w.ravel()).reshape(f.xi_shape)
    assert np.max(np.abs(np.asarray(g) - ref)) < 1e-11
    assert np.max(np.abs(ref)) > 0.1


# ---------------------------------------------------------------------------------------------
# (2) marginal variance = sigma^2
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_spectrum_is_real_even_positive_and_normalized(kind):
    f = make(kind)
    S = np.asarray(f.spectrum)
    assert S.shape == (f.P, f.P) and S.dtype == np.float64
    assert np.all(S > 0)
    assert abs(S.mean() - f.sigma**2) < 1e-13  # mean(S) = sigma^2, so sigma != 1 is exercised
    idx = (-np.arange(f.P)) % f.P
    assert np.max(np.abs(S - S[np.ix_(idx, idx)])) < 1e-13 * S.max()  # S(-k) = S(k)
    assert np.max(np.abs(S - S.T)) < 1e-13 * S.max()  # square grid: kx <-> ky symmetric


def test_matern_spectrum_exponent_and_kappa():
    ell, nu = 0.75, 2.0
    f = make("matern", ell=ell, nu=nu)
    assert abs(f.kappa - np.sqrt(2 * nu) / ell) < 1e-14
    S = np.asarray(f.spectrum)
    k = np.asarray(f.wavenumbers)
    assert abs(k[1] - 2 * np.pi / (f.P * f.h)) < 1e-13
    a, b = (3, 0), (0, 7)
    k2a, k2b = k[a[0]] ** 2 + k[a[1]] ** 2, k[b[0]] ** 2 + k[b[1]] ** 2
    ratio = S[a] / S[b]
    expected = ((f.kappa**2 + k2b) / (f.kappa**2 + k2a)) ** (nu + 1.0)
    assert abs(ratio / expected - 1) < 1e-12
    # An exponent of nu instead of nu + 1 would change this ratio by a large factor.
    wrong = ((f.kappa**2 + k2b) / (f.kappa**2 + k2a)) ** nu
    assert abs(expected / wrong - 1) > 0.5


def test_gmrf_spectrum_is_inverse_discrete_laplacian_symbol():
    f = make("gmrf", ell=0.8)
    assert abs(f.kappa - 1 / 0.8) < 1e-14
    S = np.asarray(f.spectrum)
    k = np.asarray(f.wavenumbers)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    lam = (4 - 2 * np.cos(kx * f.h) - 2 * np.cos(ky * f.h)) / f.h**2
    c = f.sigma**2 / np.mean(1.0 / (f.kappa**2 + lam))
    assert np.max(np.abs(S * (f.kappa**2 + lam) / c - 1)) < 1e-12


@pytest.mark.parametrize("kind", KINDS)
def test_marginal_variance_exact(kind):
    f = make(kind)  # sigma = 1.7: var = 2.89, so sigma vs sigma^2 confusion fails
    C = f.covariance_matrix()
    assert C.shape == (SMALL.n**2, SMALL.n**2)
    assert np.max(np.abs(np.diag(C) - 1.7**2)) < 1e-12
    assert np.max(np.abs(C - C.T)) < 1e-12
    assert np.linalg.eigvalsh(C).min() > -1e-12  # positive semidefinite


@pytest.mark.parametrize("kind", KINDS)
def test_circulant_covariance_equals_the_factor_product_and_serves_large_grids(kind):
    """The DOI covariance from the torus kernel c = Re ifft2(S) is L L^T of ``sqrt_matrix`` without forming the (n^2, P^2) factor, and
    ``covariance_matrix`` falls back to it when the factor would exceed ``max_elements`` (n = 64 needs 6.7e7 entries)."""
    f = make(kind)
    C = f.covariance_matrix()
    Cc = f.covariance_matrix_circulant()
    assert Cc.shape == C.shape
    assert np.max(np.abs(Cc - C)) < 1e-13 * f.sigma**2
    np.testing.assert_allclose(f.covariance_matrix(max_elements=1), Cc, rtol=0, atol=0)  # the fallback is the circulant form
    with pytest.raises(ValueError):
        f.sqrt_matrix(max_elements=1)  # while the factor itself still refuses
    big = GaussianField(_Grid(64, 1.5), kind, sigma=0.5, ell=0.4)  # n^2 P^2 = 6.7e7 > 5e7
    Cb = big.covariance_matrix()
    assert Cb.shape == (4096, 4096)
    assert np.max(np.abs(np.diag(Cb) - 0.25)) < 1e-12 and np.max(np.abs(Cb - Cb.T)) < 1e-14
    # stationary: the covariance of two pixels depends on their offset only
    C4 = Cb.reshape(64, 64, 64, 64)
    assert abs(C4[3, 5, 8, 9] - C4[20, 30, 25, 34]) < 1e-14 and abs(C4[0, 0, 63, 63] - C4[63, 63, 0, 0]) < 1e-14


@pytest.mark.parametrize("kind", KINDS)
def test_marginal_variance_empirical_4000_samples(kind):
    N = 4000
    mu = 0.4
    f = make(kind, mean=mu)
    th = np.asarray(f.sample(jax.random.PRNGKey(7), num=N))
    assert th.shape == (N, SMALL.n, SMALL.n)
    var = th.var(axis=0, ddof=1)
    # Per-pixel relative sd of a sample variance is sqrt(2 / (N - 1)) = 2.2%: allow 5 sd for every pixel.
    mc = np.sqrt(2.0 / (N - 1))
    assert np.max(np.abs(var / f.sigma**2 - 1)) < 5 * mc
    # Pixel-averaged variance has a smaller error; it must still be well inside the 5-sd band.
    assert abs(var.mean() / f.sigma**2 - 1) < 2.5 * mc
    # The sample mean of theta is mu with standard error sigma / sqrt(N).
    assert np.max(np.abs(th.mean(axis=0) - mu)) < 5 * f.sigma / np.sqrt(N)
    # Lag-1 covariance in x, averaged over pairs, matches the exact covariance.
    C = f.covariance_matrix().reshape((SMALL.n,) * 4)
    exact = np.mean([C[i, j, i + 1, j] for i in range(SMALL.n - 1) for j in range(SMALL.n)])
    d = th - th.mean(axis=0)
    emp = np.mean(d[:, :-1, :] * d[:, 1:, :]) * N / (N - 1)
    assert abs(emp - exact) / f.sigma**2 < 0.04
    assert exact / f.sigma**2 > 0.3  # correlated neighbors (0.47 gmrf, 0.8 matern): the lag test is not trivial


def test_sample_shapes_and_key_determinism():
    f = make("matern")
    k = jax.random.PRNGKey(0)
    a, b = f.sample(k), f.sample(k)
    assert a.shape == (SMALL.n, SMALL.n) and bool(jnp.all(a == b))
    assert f.sample(k, num=5).shape == (5, SMALL.n, SMALL.n)
    assert f.sample_xi(k).shape == f.xi_shape
    assert not bool(jnp.all(f.sample(jax.random.PRNGKey(1)) == a))


# ---------------------------------------------------------------------------------------------
# (3) stationarity
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_stationarity_covariance_depends_only_on_offset(kind):
    f = make(kind)
    n = SMALL.n
    C = f.covariance_matrix().reshape(n, n, n, n)  # [i, j, i', j']
    # Same offset (da, db) at different base points gives the same covariance.
    for da, db in [(1, 0), (0, 2), (3, 1), (-2, 4), (5, -3)]:
        vals = []
        for (i, j) in [(3, 5), (6, 4), (5, 7), (4, 6)]:
            vals.append(C[i, j, i + da, j + db])
        assert np.ptp(vals) < 1e-12, (da, db, vals)
    # Cross-check against the torus covariance function c = Re ifft2(S), an independent computation.
    c = np.fft.ifft2(np.asarray(f.spectrum)).real
    for (i, j, a, b) in [(0, 0, 4, 7), (3, 5, 9, 2), (11, 11, 0, 3)]:
        assert abs(C[i, j, a, b] - c[(a - i) % f.P, (b - j) % f.P]) < 1e-12
    # The field is isotropic enough on the square grid to be even and x <-> y symmetric.
    assert abs(C[4, 4, 6, 7] - C[4, 4, 2, 7]) < 1e-12  # even in dx
    assert abs(C[4, 4, 6, 7] - C[4, 4, 7, 6]) < 1e-12  # x <-> y
    # Offsets matter: covariance really changes with the offset (the check above is not vacuous).
    assert C[4, 4, 4, 4] - C[4, 4, 5, 4] > 0.05 * f.sigma**2


# ---------------------------------------------------------------------------------------------
# (4) correlation shape; GMRF precision
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nu,tol", [(1.5, 0.005), (2.5, 0.005), (0.5, 0.05)])
def test_matern_correlation_at_ell_and_monotone(nu, tol):
    ell = 1.0  # = 4 pixels on the FINE grid, so the offset (4, 0) is at distance exactly ell
    f = GaussianField(FINE, kind="matern", sigma=1.3, ell=ell, nu=nu)
    n = FINE.n
    C = f.covariance_matrix().reshape(n, n, n, n)
    i0 = 8
    rho = lambda a, b: C[i0, i0, i0 + a, i0 + b] / C[i0, i0, i0, i0]
    expected = matern_corr(ell, ell, nu)
    assert 0.3 < rho(4, 0) < 0.6  # sensible range: Matern at distance ell is 0.37 ... 0.52 for nu 0.5 ... 2.5
    assert abs(rho(4, 0) - expected) < tol
    # Decreases with distance, along an axis and along the diagonal.
    axis = [rho(m, 0) for m in range(0, 12)]
    diag = [rho(m, m) for m in range(0, 8)]
    assert all(np.diff(axis) < 0) and all(np.diff(diag) < 0)
    assert rho(0, 0) == pytest.approx(1.0, abs=1e-14)
    # Correlation at a longer distance matches the analytic curve too (decay rate, not just one point).
    assert abs(rho(8, 0) - matern_corr(2 * ell, ell, nu)) < tol
    # Isotropy: the diagonal offset (3, 3), at distance 3 sqrt(2) h, lies on the same analytic curve.
    assert abs(rho(3, 3) - matern_corr(3 * np.sqrt(2) * FINE.h, ell, nu)) < tol


def test_matern_longer_ell_means_larger_correlation():
    n = FINE.n
    out = []
    for ell in (0.5, 1.0, 2.0):
        f = GaussianField(FINE, kind="matern", ell=ell, nu=1.5)
        C = f.covariance_matrix().reshape(n, n, n, n)
        out.append(C[8, 8, 12, 8] / C[8, 8, 8, 8])
    assert out[0] < out[1] < out[2]


def _torus_operator(f):
    """Dense (P^2, P^2) matrix of the uncropped torus filter, built by applying it to unit vectors."""
    P = f.P
    eye = jnp.eye(P * P).reshape(P * P, P, P)
    cols = jax.vmap(f.torus_field)(eye).reshape(P * P, P * P)  # row j = image of e_j
    return np.asarray(cols).T


def test_gmrf_uncropped_precision_is_nearest_neighbor_laplacian():
    grid = _Grid(n=5, side=2.5)  # h = 0.5, P = 10
    f = GaussianField(grid, kind="gmrf", sigma=1.0, ell=0.8)  # kappa = 1.25: kappa^2 != kappa
    P, h, kappa = f.P, f.h, f.kappa
    M = _torus_operator(f)
    Sigma = M @ M.T  # torus covariance, no cropping
    assert np.linalg.cond(Sigma) < 1e3  # well conditioned, so the inverse below is accurate to ~1e-13
    Q = np.linalg.inv(Sigma)

    # Independent expected precision: (kappa^2 I - Laplacian_torus) / c with c = sigma^2 / mean(1 / (kappa^2 + lam)).
    k = 2 * np.pi * np.fft.fftfreq(P, d=h)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    lam = (4 - 2 * np.cos(kx * h) - 2 * np.cos(ky * h)) / h**2
    c = 1.0 / np.mean(1.0 / (kappa**2 + lam))
    Qe = np.zeros_like(Q)
    for a in range(P):
        for b in range(P):
            r = a * P + b
            Qe[r, r] = (kappa**2 + 4 / h**2) / c
            for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                Qe[r, ((a + da) % P) * P + (b + db) % P] = -1 / (h**2 * c)
    assert np.max(np.abs(Q - Qe)) < 1e-10 * Qe[0, 0]

    # Banded: exactly five entries per row above round-off (diagonal + 4 neighbors), the rest ~ 1e-15.
    big = np.abs(Q) > 1e-10 * np.abs(np.diag(Q)).max()
    assert np.all(big.sum(axis=1) == 5)
    assert np.all(np.diag(Q) > 0) and np.all(Q[big & ~np.eye(P * P, dtype=bool)] < 0)


def test_gmrf_cropped_precision_is_not_banded_so_banded_test_is_discriminating():
    grid = _Grid(n=5, side=2.5)
    f = GaussianField(grid, kind="gmrf", sigma=1.0, ell=1.0)
    n = grid.n
    Q = np.linalg.inv(f.covariance_matrix())
    mask = np.ones((n * n, n * n), bool)
    for a in range(n):
        for b in range(n):
            r = a * n + b
            mask[r, r] = False
            for da, db in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if 0 <= a + da < n and 0 <= b + db < n:
                    mask[r, (a + da) * n + b + db] = False
    # Cropping a Markov field destroys the exact sparsity; the violation is orders above round-off.
    assert np.abs(Q[mask]).max() / np.diag(Q).max() > 1e-4


def test_gmrf_correlation_decays_and_matches_screened_green_function_scale():
    f = GaussianField(FINE, kind="gmrf", sigma=1.0, ell=1.0)
    n = FINE.n
    C = f.covariance_matrix().reshape(n, n, n, n)
    rho = [C[8, 8, 8 + m, 8] / C[8, 8, 8, 8] for m in range(10)]
    assert all(np.diff(rho) < 0)
    assert 0.1 < rho[4] < 0.7  # one correlation length (4 pixels) away


# ---------------------------------------------------------------------------------------------
# Constructor options
# ---------------------------------------------------------------------------------------------


def test_kappa_override_and_grid_without_h_attribute():
    f = GaussianField(_GridNoH(n=12, side=3.0), kind="matern", kappa=3.0, nu=1.5)
    assert f.h == 0.25 and f.kappa == 3.0 and abs(f.ell - np.sqrt(3.0) / 3.0) < 1e-14
    g = GaussianField(SMALL, kind="matern", kappa=3.0, nu=1.5)
    assert np.max(np.abs(np.asarray(f.spectrum - g.spectrum))) == 0
    h = GaussianField(SMALL, kind="gmrf", kappa=2.0)
    assert h.kappa == 2.0


def test_pad_changes_torus_size_and_pad_one_still_works():
    f1 = GaussianField(SMALL, pad=1)
    f3 = GaussianField(SMALL, pad=3)
    assert f1.xi_shape == (12, 12) and f3.xi_shape == (36, 36)
    for f in (f1, f3):
        assert np.max(np.abs(np.diag(f.covariance_matrix()) - 1.0)) < 1e-12
        assert f.theta(np.ones(f.xi_shape)).shape == (12, 12)


def test_input_validation():
    with pytest.raises(ValueError):
        GaussianField(SMALL, kind="exponential")
    with pytest.raises(ValueError):
        GaussianField(SMALL, sigma=0.0)
    with pytest.raises(ValueError):
        GaussianField(SMALL, ell=-1.0)
    with pytest.raises(ValueError):
        GaussianField(SMALL, nu=0.0)
    with pytest.raises(ValueError):
        GaussianField(SMALL, pad=0)
    with pytest.raises(ValueError):
        GaussianField(SMALL, mean=np.zeros((5, 5)))
    f = GaussianField(SMALL)
    with pytest.raises(ValueError):
        f.theta(np.zeros((SMALL.n, SMALL.n)))  # xi must live on the padded torus
    with pytest.raises(ValueError):
        GaussianField(_Grid(n=96, side=5.0)).sqrt_matrix()  # too large for the dense helper


# ---------------------------------------------------------------------------------------------
# (5) links
# ---------------------------------------------------------------------------------------------


def test_exactly_two_links_and_no_bounded_links():
    assert set(LINKS) == {"log_eps", "identity"}
    assert LINKS["log_eps"] is log_eps and LINKS["identity"] is identity
    assert get_link("log_eps") is log_eps and get_link(identity) is identity
    for name in ("sigmoid_box", "softplus", "SigmoidBox", "Softplus"):
        assert not hasattr(priors, name)
    with pytest.raises(ValueError):
        get_link("softplus")
    with pytest.raises(ValueError):
        get_link(lambda t: t)  # a forward map without an inverse is not a link


def test_identity_link_exact():
    th = np.random.default_rng(3).normal(size=(7, 5)) * 10
    out = identity(th)
    assert np.array_equal(np.asarray(out), th)
    assert np.array_equal(np.asarray(jax.jit(identity)(th)), th)
    assert np.array_equal(np.asarray(identity.inverse(th)), th)
    assert float(jax.grad(lambda t: identity(t).sum())(jnp.asarray(2.0))) == 1.0
    # The two links really differ: log_eps is not the identity, so a swapped default would be caught.
    assert abs(float(log_eps(jnp.log(8.0))) - float(identity(jnp.log(8.0)))) > 4.0


def test_log_eps_anchor_values():
    assert float(log_eps(0.0)) == 0.0  # theta = 0 is the vacuum background, chi = 0 exactly
    assert abs(float(log_eps(jnp.log(8.0))) - 7.0) < 1e-14  # eps_r = 8 is chi = 7 (book maximum)
    assert abs(float(log_eps(1.0)) - (np.e - 1.0)) < 1e-15
    assert abs(float(log_eps(-1.0)) - (1.0 / np.e - 1.0)) < 1e-15
    assert float(log_eps.inverse(0.0)) == 0.0
    assert abs(float(log_eps.inverse(7.0)) - np.log(8.0)) < 1e-15
    assert abs(float(log_eps.inverse(-0.5)) - np.log(0.5)) < 1e-15  # eps_r = 1/2 <-> chi = -1/2
    # Array in, array out, same shape, under jit as well.
    th = np.linspace(-3.0, 3.0, 13).reshape(1, 13)
    assert log_eps(th).shape == (1, 13)
    assert np.array_equal(np.asarray(jax.jit(log_eps)(th)), np.asarray(log_eps(th)))
    chi = np.linspace(-0.95, 7.0, 13).reshape(1, 13)  # the inverse is defined for chi > -1 only
    assert log_eps.inverse(chi).shape == (1, 13)
    assert np.array_equal(np.asarray(jax.jit(log_eps.inverse)(chi)), np.asarray(log_eps.inverse(chi)))


def test_log_eps_inverse_round_trip():
    th = np.linspace(-12.0, 6.0, 721)
    err = np.abs(np.asarray(log_eps.inverse(log_eps(th))) - th)
    # theta -> chi -> theta. chi = eps_r - 1 is stored with absolute resolution ~1e-16 near -1, so the
    # recovered theta carries an error ~1e-16 / eps_r = 1e-16 exp(-theta) (an inherent limit of any link
    # that outputs chi, not of the formula). It is below 1e-12 for theta >= -8 (eps_r >= 3e-4, i.e.
    # 3.3 sigma below vacuum even for sigma = 2.4) ...
    ok = th >= -8.0
    assert err[ok].max() < 1e-12
    # ... and follows that scaling over the whole range, so the bound is the only loss.
    assert np.all(err <= 1.1e-16 * np.exp(-th) + 4e-15)
    # chi -> theta -> chi over the book's contrast range eps_r in [1, 8] and below vacuum.
    chi0 = np.linspace(-0.99, 7.0, 400)
    assert np.max(np.abs(np.asarray(log_eps(log_eps.inverse(chi0))) - chi0)) < 1e-12
    # eps_r = 1 + chi = exp(theta): the inverse of chi is log eps_r, not log chi.
    assert np.max(np.abs(np.asarray(log_eps.inverse(chi0)) - np.log(1.0 + chi0))) < 1e-15
    assert abs(float(log_eps.inverse(chi0[0])) - np.log(0.01)) < 1e-12


def test_log_eps_chi_greater_than_minus_one_and_monotone():
    th = np.linspace(-30.0, 30.0, 6001)  # exp(-30) - 1 = -1 + 9e-14 is still distinguishable from -1 in float64
    chi = np.asarray(log_eps(th))
    assert np.all(chi > -1.0)
    assert np.all(np.diff(chi) > 0)
    # chi < 0 exactly where theta < 0 (eps_r < 1 is allowed; it is the log-normal statement).
    assert np.all((chi < 0) == (th < 0)) and np.all((chi > 0) == (th > 0))
    # Even a very wide prior (sigma = 2, mean -1) keeps chi > -1 at every pixel of every draw.
    f = make("matern", sigma=2.0, mean=-1.0)
    chis = np.asarray(compose(f)(f.sample_xi(jax.random.PRNGKey(11), num=200)))
    assert chis.min() > -1.0 and chis.min() < -0.9 and chis.max() > 1.0  # the draws do explore both sides


def test_log_eps_gradient_is_exp_theta():
    th = jnp.asarray(np.linspace(-6.0, 6.0, 49))
    g = jax.vmap(jax.grad(log_eps))(th)
    assert np.max(np.abs(np.asarray(g) / np.exp(np.asarray(th)) - 1.0)) < 1e-14
    assert abs(float(jax.grad(log_eps)(jnp.log(8.0))) - 8.0) < 1e-13  # dchi/dtheta = eps_r
    assert float(jax.grad(log_eps)(0.0)) == 1.0
    # Gradient of the inverse is 1 / (1 + chi) = 1 / eps_r.
    chi = jnp.asarray([-0.9, -0.5, 0.0, 3.0, 7.0])
    gi = jax.vmap(jax.grad(log_eps.inverse))(chi)
    assert np.max(np.abs(np.asarray(gi) * (1.0 + np.asarray(chi)) - 1.0)) < 1e-14
    # No clipping and no NaN at moderately extreme theta: the gradient is just exp(theta).
    big = jnp.asarray([-30.0, 30.0])
    gb = jax.vmap(jax.grad(log_eps))(big)
    assert np.all(np.isfinite(np.asarray(gb)))
    assert abs(float(gb[1]) / np.exp(30.0) - 1.0) < 1e-14


def test_prior_is_lognormal_in_eps_r():
    """With the log_eps link, eps_r = 1 + chi = exp(theta) is log-normal: median exp(mu), mean exp(mu + sigma^2/2)."""
    N, mu, sig = 4000, 0.3, 0.5
    f = GaussianField(SMALL, kind="matern", sigma=sig, ell=0.75, mean=mu)
    xi = f.sample_xi(jax.random.PRNGKey(21), num=N)
    eps_r = 1.0 + np.asarray(compose(f)(xi))
    assert np.all(eps_r > 0)
    log_e = np.log(eps_r)
    assert np.max(np.abs(log_e.mean(axis=0) - mu)) < 5 * sig / np.sqrt(N)
    assert np.max(np.abs(log_e.std(axis=0, ddof=1) / sig - 1)) < 5 * np.sqrt(0.5 / (N - 1))
    # Mean of a log-normal: exp(mu + sigma^2 / 2) = 1.53 here, not the median exp(mu) = 1.35.
    mean_exact = np.exp(mu + sig**2 / 2)
    sd_exact = mean_exact * np.sqrt(np.expm1(sig**2))
    assert abs(mean_exact - np.exp(mu)) > 0.15  # the two candidate answers are well separated
    assert np.max(np.abs(eps_r.mean(axis=0) - mean_exact)) < 5 * sd_exact / np.sqrt(N)
    # Median is exp(mu): half the draws lie below, per pixel, within 5 binomial sd.
    frac_below = (eps_r < np.exp(mu)).mean(axis=0)
    assert np.max(np.abs(frac_below - 0.5)) < 5 * 0.5 / np.sqrt(N)
    # Skewness of eps_r is positive (log-normal), of theta is ~0 (Gaussian).
    z = (log_e - log_e.mean(axis=0)) / log_e.std(axis=0)
    assert np.max(np.abs((z**3).mean(axis=0))) < 5 * np.sqrt(6.0 / N)
    w = (eps_r - eps_r.mean(axis=0)) / eps_r.std(axis=0)
    assert (w**3).mean() > 0.5


def test_compose_field_and_link():
    f = make("matern", mean=-0.2)
    xi = np.random.default_rng(4).normal(size=f.xi_shape)
    th = np.asarray(f.theta(xi))
    # Default link is log_eps: chi = exp(theta) - 1.
    chi_fn = compose(f)
    chi = chi_fn(xi)
    assert chi.shape == (SMALL.n, SMALL.n)
    np.testing.assert_allclose(np.asarray(chi), np.exp(th) - 1.0, rtol=1e-14, atol=1e-15)  # XLA vs NumPy exp: ~1 ulp
    assert np.all(np.asarray(chi) > -1.0) and np.any(np.asarray(chi) < 0) and np.any(np.asarray(chi) > 0)
    assert np.max(np.abs(np.asarray(log_eps.inverse(chi)) - th)) < 1e-13  # theta recovered from chi
    # Explicit link arguments: object, name, identity.
    assert np.array_equal(np.asarray(compose(f, log_eps)(xi)), np.asarray(chi))
    assert np.array_equal(np.asarray(compose(f, "log_eps")(xi)), np.asarray(chi))
    assert np.array_equal(np.asarray(compose(f, identity)(xi)), th)
    assert np.array_equal(np.asarray(compose(f, "identity")(xi)), th)
    with pytest.raises(ValueError):
        compose(f, "sigmoid_box")
    # Jitted and batched.
    assert np.max(np.abs(np.asarray(jax.jit(chi_fn)(xi) - chi))) < 1e-13
    xib = np.stack([xi, -xi])
    chib = chi_fn(xib)
    assert chib.shape == (2, SMALL.n, SMALL.n)
    np.testing.assert_allclose(np.asarray(chib[0]), np.asarray(chi), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(chib[1]), np.exp(np.asarray(f.theta(-xi))) - 1.0, rtol=1e-14, atol=1e-15)


def test_compose_gradient_chain_rule_and_finite_differences():
    f = make("matern", mean=-0.2)
    chi_fn = compose(f)
    rng = np.random.default_rng(5)
    xi = rng.normal(size=f.xi_shape)
    w = rng.normal(size=(SMALL.n, SMALL.n))
    # d/dxi sum(w * chi) = L^T (w * exp(theta)): the log_eps chain rule is dchi/dtheta = eps_r.
    g = jax.grad(lambda x: jnp.sum(w * chi_fn(x)))(xi)
    th = np.asarray(f.theta(xi))
    ref = (f.sqrt_matrix().T @ (w * np.exp(th)).ravel()).reshape(f.xi_shape)
    assert np.max(np.abs(np.asarray(g) - ref)) < 1e-11 * max(1.0, np.abs(ref).max())
    assert np.abs(ref).max() > 0.5
    # A missing factor exp(theta) (i.e. the identity chain rule) would give a visibly different gradient.
    wrong = (f.sqrt_matrix().T @ w.ravel()).reshape(f.xi_shape)
    assert np.abs(ref - wrong).max() > 0.1
    # Central finite differences along a random direction, for a nonlinear function of chi.
    fn = lambda x: float(jnp.sum(chi_fn(x) ** 2))
    g2 = jax.grad(lambda x: jnp.sum(chi_fn(x) ** 2))(xi)
    assert g2.shape == f.xi_shape and bool(jnp.all(jnp.isfinite(g2))) and float(jnp.abs(g2).max()) > 0
    d = rng.normal(size=f.xi_shape)
    eps = 1e-5
    fd = (fn(xi + eps * d) - fn(xi - eps * d)) / (2 * eps)
    an = float(jnp.vdot(g2, d))
    assert abs(fd - an) / abs(an) < 1e-7
