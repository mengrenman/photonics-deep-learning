"""Tests for bayes_scatter.diagnostics against analytic AR(1) and iid results.

For x_t = phi x_{t-1} + e_t the autocorrelation is rho(t) = phi^|t| and the integrated autocorrelation
time is tau = (1 + phi) / (1 - phi): 19 for phi = 0.9, 1/3 for phi = -0.5.
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from bayes_scatter import diagnostics as dg


def ar1(phi, n, n_series=1, seed=0):
    """Stationary AR(1) series, shape (n, n_series), unit innovation variance."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, n_series))
    e[0] *= 1.0 / np.sqrt(1.0 - phi**2)  # start in the stationary distribution
    return lfilter([1.0], [1.0, -phi], e, axis=0)


def test_autocorr_matches_direct_sum_and_ar1():
    x = ar1(0.9, 4000, seed=1)[:, 0]
    rho = dg.autocorr(x)
    xc = x - x.mean()
    direct = np.array([np.dot(xc[: len(x) - t], xc[t:]) for t in range(60)]) / np.dot(xc, xc)
    np.testing.assert_allclose(rho[:60], direct, atol=1e-12)
    assert rho[0] == pytest.approx(1.0, abs=1e-14)
    # AR(1): rho(t) = 0.9^t; sampling error at lag 10 with N = 4000 is a few percent
    assert rho[10] == pytest.approx(0.9**10, abs=0.08)


def test_autocorr_axis_and_constant_series():
    x = ar1(0.5, 500, n_series=3, seed=2)
    r0 = dg.autocorr(x, axis=0)
    r1 = dg.autocorr(x.T, axis=1)
    assert r0.shape == x.shape
    np.testing.assert_allclose(r0, r1.T, atol=1e-14)
    const = np.full(100, 3.0)
    assert np.all(np.isnan(dg.autocorr(const)))
    assert np.isnan(dg.tau_int(const)) and np.isnan(dg.ess(const))


def test_tau_int_ar1_within_10_percent():
    phi, n = 0.9, 1_000_000
    tau_exact = (1 + phi) / (1 - phi)
    assert tau_exact == pytest.approx(19.0)
    tau = dg.tau_int(ar1(phi, n, n_series=4, seed=3))  # one value per column
    assert tau.shape == (4,)
    # relative sd of the windowed estimate is about sqrt(2 (2W + 1) / N) = 2% for W ~ 5 tau
    np.testing.assert_allclose(tau, tau_exact, rtol=0.10)
    assert abs(tau.mean() / tau_exact - 1) < 0.03


def test_tau_int_detects_negative_and_zero_correlation():
    # Anti-correlated AR(1): tau = (1 - 0.5) / (1 + 0.5) = 1/3, so ESS exceeds N.
    x_neg = ar1(-0.5, 200_000, seed=4)[:, 0]
    assert dg.tau_int(x_neg) == pytest.approx(1.0 / 3.0, rel=0.10)
    assert dg.ess(x_neg) == pytest.approx(3.0 * len(x_neg), rel=0.10)
    # extremely antithetic (tau = 0.0256): ESS is capped at N log10(N) rather than blowing up
    x_ext = ar1(-0.95, 100_000, seed=15)[:, 0]
    assert dg.ess(x_ext) <= len(x_ext) * np.log10(len(x_ext)) * (1 + 1e-12)
    assert dg.ess(x_ext) > 2.0 * len(x_ext)
    tau_iid = dg.tau_int(np.random.default_rng(5).standard_normal(200_000))
    assert tau_iid == pytest.approx(1.0, abs=0.05)


def test_tau_int_window_rule_is_sokal():
    # tau(W) is the full cumulative sum; the chosen W is the first with W >= c tau(W).
    x = ar1(0.8, 20_000, seed=6)[:, 0]
    rho = dg.autocorr(x)
    taus = 1 + 2 * np.cumsum(rho[1:])  # taus[W-1] = tau(W)
    w = next(w for w in range(1, len(x)) if w >= 5 * taus[w - 1])
    assert dg.tau_int(x, c=5) == pytest.approx(taus[w - 1], rel=1e-12)
    # a larger c can only widen the window
    assert dg.tau_int(x, c=10) != dg.tau_int(x, c=2)


def test_ess_iid_normal_is_about_n():
    n = 100_000
    x = np.random.default_rng(7).standard_normal((n, 3))
    e = dg.ess(x)
    assert e.shape == (3,)
    np.testing.assert_allclose(e, n, rtol=0.05)


def test_ess_ar1_is_n_over_tau():
    n = 400_000
    e = dg.ess(ar1(0.9, n, seed=8)[:, 0])
    assert e == pytest.approx(n / 19.0, rel=0.10)


def test_mcse_mean_matches_empirical_spread_of_means():
    # 200 independent AR(1) series; the std of their means is the true MCSE.
    phi, n, m = 0.9, 4000, 200
    x = ar1(phi, n, n_series=m, seed=9)
    sd_of_means = x.mean(axis=0).std(ddof=1)
    sigma = 1 / np.sqrt(1 - phi**2)
    theory = sigma * np.sqrt(19.0 / n)
    assert sd_of_means == pytest.approx(theory, rel=0.15)  # spread of the 200 means vs theory
    est = dg.mcse_mean(x)
    assert est.shape == (m,)
    assert np.median(est) == pytest.approx(theory, rel=0.15)


def test_axis_argument_for_ess_and_mcse():
    x = ar1(0.7, 3000, n_series=2, seed=10)
    np.testing.assert_allclose(dg.ess(x.T, axis=1), dg.ess(x, axis=0))
    np.testing.assert_allclose(dg.mcse_mean(x.T, axis=1), dg.mcse_mean(x, axis=0))


def test_split_rhat_identical_chains_near_one():
    chains = np.random.default_rng(11).standard_normal((4, 2000, 5))
    r = dg.split_rhat(chains)
    assert r.shape == (5,)
    assert np.all(np.abs(r - 1) < 0.01)


def test_split_rhat_shifted_means_above_threshold():
    rng = np.random.default_rng(12)
    chains = rng.standard_normal((4, 2000)) + np.array([0.0, 0.5, 1.0, 1.5])[:, None]
    # sqrt(1 + B / (n W)) with B / n = var of chain means = 0.4167: expected 1.19
    assert dg.split_rhat(chains) == pytest.approx(np.sqrt(1 + 0.41667), rel=0.02)
    assert dg.split_rhat(chains) > 1.1


def test_split_rhat_flags_trend_inside_identical_chains():
    # Every chain has the same distribution, but each drifts: halves disagree, plain R-hat would not.
    rng = np.random.default_rng(13)
    n = 2000
    trend = np.linspace(0.0, 2.0, n)
    chains = rng.standard_normal((4, n)) + trend
    assert dg.split_rhat(chains) > 1.1


def test_split_rhat_odd_length_and_validation():
    chains = np.random.default_rng(14).standard_normal((3, 1001))
    assert abs(dg.split_rhat(chains) - 1) < 0.02
    with pytest.raises(ValueError):
        dg.split_rhat(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        dg.split_rhat(np.zeros(10))


def test_ess_geyer_matches_ar1_and_iid_and_is_conservative_on_a_trend():
    phi, n = 0.9, 400_000
    x = ar1(phi, n, n_series=3, seed=11)
    ess = dg.ess_geyer(x)
    assert ess.shape == (3,)
    np.testing.assert_allclose(ess, n / 19.0, rtol=0.12)  # tau = (1 + phi)/(1 - phi) = 19
    iid = np.random.default_rng(12).standard_normal(200_000)
    assert dg.ess_geyer(iid) == pytest.approx(len(iid), rel=0.05)
    anti = ar1(-0.5, 200_000, seed=13)[:, 0]
    assert dg.ess_geyer(anti) == pytest.approx(3.0 * len(anti), rel=0.12)  # tau = 1/3
    assert np.isnan(dg.ess_geyer(np.full(100, 2.0)))
    # axis handling
    np.testing.assert_allclose(dg.ess_geyer(x.T, axis=1), ess, rtol=1e-12)
    # a pure trend plus noise has a huge autocorrelation time: both estimators give a small ESS, Geyer not larger
    t = np.arange(600)
    trend = 0.03 * t + np.random.default_rng(14).standard_normal(600)
    assert dg.ess_geyer(trend) < 30.0 and dg.ess_geyer(trend) <= dg.ess(trend) * 1.5
