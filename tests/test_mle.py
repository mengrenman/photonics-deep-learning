"""Tests for bayes_scatter.mle: the Levenberg-Marquardt maximum likelihood fit of the two-disk parameters and its agreement with the Cramer-Rao bound."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import fisher as fi
from bayes_scatter import mle
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.likelihood import add_noise, noise_std_from_snr

TRUTH = sh.TwoDisks(g=0.15, r=0.2, eps_r=2.0, cx=0.01, cy=-0.02)


@pytest.fixture(scope="module")
def small():
    """24 x 24 grid on a 1.5 x 1.5 DOI with 8 transmitters and 12 receivers (the fixture of test_fisher.py), tight solver tolerance."""
    setup = Setup(Grid(24, 1.5), Antennas(ring(8, 4.0), ring(12, 4.5)))
    fwd = Forward(setup, tol=1e-12)
    d_clean, _ = fwd(jnp.asarray(TRUTH.chi(setup.grid)))
    return {"fwd": fwd, "d": d_clean}


def test_project_keeps_the_box_and_the_disks_inside_the_domain_and_is_idempotent():
    b = mle.Bounds.default()
    side = 1.5
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = np.array([rng.uniform(-0.3, 1.0), rng.uniform(0.0, 0.6), rng.uniform(0.5, 9.0), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
        q = mle.project(p, b, side)
        for i, k in enumerate(mle.PARAMS):
            assert b.lo[k] - 1e-12 <= q[i] <= b.hi[k] + 1e-12
        assert 2 * q[1] + 0.5 * q[0] + abs(q[3]) <= 0.5 * side - b.margin + 1e-12 and q[1] + abs(q[4]) <= 0.5 * side - b.margin + 1e-12
        np.testing.assert_allclose(mle.project(q, b, side), q, atol=1e-13)
        mle.as_object(q).check_inside(Grid(24, side))  # no ValueError: both disks inside the DOI
    inside = np.array([0.05, 0.2, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(mle.project(inside, b, side), inside)  # a feasible point is unchanged


def test_noise_free_data_are_fitted_exactly_from_a_perturbed_start(small):
    fwd, d = small["fwd"], small["d"]
    p_true = TRUTH.vector(mle.PARAMS)
    p0 = p_true * np.array([1.6, 0.9, 1.3, 1.0, 1.0]) + np.array([0.0, 0.0, 0.0, 0.015, 0.02])
    s = noise_std_from_snr(d, 60.0)  # the noise level only sets the units of Phi here: the data have none
    r = mle.fit(fwd, d, s, p0, scale=np.full(5, 1e-4), tol=1e-3, phi_tol=1e-8, max_iter=60)
    assert r["converged"] and r["phi"] < 1e-6 * float(mle.misfit(fwd(jnp.asarray(mle.as_object(p0).chi(fwd.grid)))[0], d, s))
    np.testing.assert_allclose(r["p"], p_true, atol=2e-5)
    assert r["n_eval"] >= r["n_iter"] and r["at_boundary"] == []


def test_a_gap_that_is_zero_stays_on_the_boundary(small):
    fwd = small["fwd"]
    obj = sh.TwoDisks(g=0.0, r=0.2, eps_r=2.0)
    d, _ = fwd(jnp.asarray(obj.chi(fwd.grid)))
    s = noise_std_from_snr(d, 50.0)
    r = mle.fit(fwd, d, s, np.array([0.1, 0.2, 2.0, 0.0, 0.0]), scale=np.full(5, 1e-3), tol=1e-3, phi_tol=1e-6, max_iter=60)
    assert r["p"][0] >= 0.0 and r["p"][0] < 2e-3  # the noise-free gap 0 is found from g = 0.1 (the constraint g >= 0 may be active)
    assert r["phi"] < 1e-2 * float(mle._evaluate(fwd, d, s, np.array([0.1, 0.2, 2.0, 0.0, 0.0]))[3])


def test_multistart_returns_the_lowest_misfit_run(small):
    fwd, d = small["fwd"], small["d"]
    s = noise_std_from_snr(d, 40.0)
    p_true = TRUTH.vector(mle.PARAMS)
    bad = np.array([0.5, 0.1, 5.0, 0.12, 0.12])  # a poor start with a tiny iteration budget: it cannot reach the truth
    best, runs = mle.fit_multistart(fwd, d, s, [("bad", bad), ("truth", p_true)], max_iter=1, scale=np.full(5, 1e-3))
    assert len(runs) == 2 and best["label"] == min(runs, key=lambda r: r["phi"])["label"] == "truth" and {r["label"] for r in runs} == {"bad", "truth"}
    assert runs[0]["converged"] is False and runs[0]["stop"] == "max_iter"


def test_monte_carlo_error_of_the_estimator_matches_the_cramer_rao_bound_at_high_snr(small):
    """At 25 dB the five-parameter MLE of the 24 x 24 problem is unbiased to a fraction of its sd and its sd is the Cramer-Rao one within the sampling error of 40
    realizations (a relative error of about 1 / sqrt(2 * 40) = 11 % in each sd)."""
    fwd, d = small["fwd"], small["d"]
    snr = 25.0
    s = noise_std_from_snr(d, snr)
    D = fi.data_derivatives(fwd, TRUTH.chi(fwd.grid), sh.shape_derivatives(fwd.grid, TRUTH, mle.PARAMS))
    F = fi.fisher_matrix(D, s)
    sd = np.sqrt(np.diag(np.linalg.inv(F)))
    p_true = TRUTH.vector(mle.PARAMS)
    key = jax.random.key(7)
    est = []
    for k in range(40):
        d_obs = add_noise(jax.random.fold_in(key, k), d, s)
        r = mle.fit(fwd, d_obs, s, p_true, scale=sd, tol=0.01, phi_tol=1e-3, max_iter=30)
        assert r["converged"]
        est.append(r["p"])
    summ = mle.estimator_summary(np.array(est), p_true, sd)
    ratio = np.array(summ["sd"]) / sd
    assert np.all((ratio > 0.7) & (ratio < 1.35)), ratio
    assert np.all(np.abs(np.array(summ["bias"])) < 0.5 * sd)  # 3.2 standard errors of a mean of 40 is 0.5 sd
    assert 0.85 <= np.mean(summ["coverage_1p96"]) <= 1.0


def test_estimator_summary_on_known_samples():
    est = np.array([[1.0, 10.0], [3.0, 10.0], [2.0, 10.0], [2.0, 10.0]])
    out = mle.estimator_summary(est, truth=[2.0, 9.0], crb_sd=[0.5, 2.0])
    assert out["n"] == 4 and out["bias"] == pytest.approx([0.0, 1.0]) and out["sd"][1] == 0.0 and out["rmse"] == pytest.approx([np.sqrt(0.5), 1.0])
    assert out["rmse_over_crb"] == pytest.approx([np.sqrt(0.5) / 0.5, 0.5]) and out["coverage_1p96"] == [0.5, 1.0]
