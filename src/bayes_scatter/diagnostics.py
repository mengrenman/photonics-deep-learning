"""MCMC diagnostics: autocorrelation, integrated autocorrelation time, ESS, MCSE, split R-hat.

Conventions (docs/DESIGN.md, section 6). Everything here is post-processing in NumPy float64.

* A *series* is an array whose time axis is ``axis`` (default 0); all other axes are independent
  components (pixels, parameters, chains), and every function returns one value per component.
* ``autocorr`` uses the standard FFT estimator c(t) = (1/N) sum_s (x_s - mean)(x_{s+t} - mean), i.e.
  the sum is divided by N, not N - t. This biased estimator is the one for which Sokal's windowed
  estimate of tau is well behaved.
* ``tau_int`` is Sokal's automatic windowing: tau(W) = 1 + 2 sum_{t=1}^{W} rho(t), with W the smallest
  window such that W >= c * tau(W). Inside this test tau(W) is floored at 1, which changes nothing for
  positively correlated series but keeps the window at least c lags long for antithetic ones (without
  it, rho(1) = -0.5 gives tau(1) = 0 and the window stops at W = 1, grossly biasing tau). The estimate
  has relative standard deviation of about sqrt(2 (2W + 1) / N), so N should be several hundred times tau.
* A series with exactly zero variance (a stuck chain) has no autocorrelation; the functions return
  NaN for it rather than inventing a value.
* ``ess_geyer`` is the same quantity by Geyer's initial monotone positive sequence estimator (sums of
  adjacent autocorrelation pairs, truncated at the first non-positive pair and made non-increasing), which
  needs no window constant and is the more conservative estimate on short chains; it is used as a cross-check
  of the Sokal ESS.
* ``ess`` and ``mcse_mean`` take ONE chain. For several chains, add the ESS values (independent
  chains) and combine the per-chain MCSEs in quadrature.
"""

import numpy as np
from scipy import fft as _sfft


def _time_first(x, axis):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        raise ValueError("a series needs at least one axis")
    return np.moveaxis(x, axis, 0)


def autocorr(x, axis=0):
    """Normalized autocorrelation rho(t), t = 0..N-1, by FFT; rho(0) = 1.

    x : array with N samples along ``axis``. Returns an array of the same shape.
    Components with zero variance give NaN.
    """
    x = _time_first(x, axis)
    n = x.shape[0]
    xc = x - x.mean(axis=0)
    nfft = _sfft.next_fast_len(2 * n, real=True)  # zero-pad to avoid circular wrap-around
    f = np.fft.rfft(xc, n=nfft, axis=0)
    acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n]
    constant = np.ptp(x, axis=0) == 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = acov / acov[0]
    rho = np.where(constant, np.nan, rho)
    return np.moveaxis(rho, 0, axis)


def tau_int(x, c=5.0, axis=0):
    """Integrated autocorrelation time with Sokal's automatic window.

    tau = 1 + 2 sum_{t=1}^{W} rho(t), W = smallest window with W >= c * max(tau(W), 1) (``c`` = 5 is
    the usual choice; the floor at 1 only matters for antithetic series, see the module docstring).
    If no window satisfies the rule, the full series is used (and the estimate is unreliable).
    Returns one tau per component (the time axis is removed). tau < 1 is legitimate for an
    antithetic series; a value near or below 0 means the series is too short to estimate it.
    """
    rho = autocorr(_time_first(x, axis), axis=0)
    n = rho.shape[0]
    taus = 2.0 * np.cumsum(rho, axis=0) - 1.0  # taus[W] = tau(W)
    t = np.arange(n).reshape((n,) + (1,) * (rho.ndim - 1))
    ok = t >= c * np.maximum(taus, 1.0)
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), n - 1)
    return np.take_along_axis(taus, first[None], axis=0)[0]


def ess(x, c=5.0, axis=0):
    """Effective sample size N / tau_int of one chain, per component.

    As in Stan, tau is floored at 1 / log10(N) (and log10(N) at 1), i.e. ESS <= N log10(N), so that
    estimation noise on a strongly antithetic series cannot give a negative or enormous ESS.
    """
    x = _time_first(x, axis)
    n = x.shape[0]
    return n / np.maximum(tau_int(x, c=c, axis=0), 1.0 / max(np.log10(n), 1.0))


def ess_geyer(x, axis=0):
    """Effective sample size of one chain by Geyer's initial monotone positive sequence, per component.

    With pair sums G_k = rho(2k) + rho(2k+1), the sequence is cut at the first G_k <= 0 and replaced by its
    running minimum; tau = -1 + 2 sum_k G_k and ESS = N / tau, floored like ``ess`` (ESS <= N log10 N).
    Constant components give NaN.
    """
    rho = autocorr(_time_first(x, axis), axis=0)
    n = rho.shape[0]
    m = n // 2
    if m < 1:
        raise ValueError("need at least 2 draws")
    gam = rho[0:2 * m:2] + rho[1:2 * m:2]
    positive = np.cumprod(gam > 0.0, axis=0).astype(bool)  # initial positive sequence
    mono = np.where(positive, np.minimum.accumulate(gam, axis=0), 0.0)
    tau = np.where(np.isnan(rho[0]), np.nan, -1.0 + 2.0 * mono.sum(axis=0))
    return n / np.maximum(tau, 1.0 / max(np.log10(n), 1.0))


def mcse_mean(x, c=5.0, axis=0):
    """Monte Carlo standard error of the mean of one chain: std(x) / sqrt(ess(x)), per component."""
    x = _time_first(x, axis)
    return np.std(x, axis=0, ddof=1) / np.sqrt(ess(x, c=c, axis=0))


def split_rhat(chains):
    """Split potential scale reduction factor (Gelman et al., BDA3; Stan).

    chains : array (M, N, ...) of M chains with N draws each. Every chain is cut into its first and
    last N // 2 draws (the middle draw is dropped for odd N), giving 2M half-chains of n = N // 2 draws.
    Then W is the mean within-half-chain variance, B = n * var(half-chain means), and
    rhat = sqrt(((n - 1) / n * W + B / n) / W). Values near 1 (below about 1.01) indicate that all
    halves agree; the split also flags a trend inside a single chain. Returns one value per component.
    """
    x = np.asarray(chains, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("chains must have shape (n_chains, n_draws, ...)")
    n = x.shape[1] // 2
    if n < 2:
        raise ValueError("need at least 4 draws per chain")
    halves = np.concatenate([x[:, :n], x[:, x.shape[1] - n:]], axis=0)
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = n * halves.mean(axis=1).var(axis=0, ddof=1)
    var_plus = (n - 1) / n * within + between / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(var_plus / within)
