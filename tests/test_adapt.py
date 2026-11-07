"""Tests for bayes_scatter.adapt: the affine map, dual averaging, variable-length HMC, the map update, the adaptation driver, and
the validation DESIGN 12.2 asks for: (a) adaptation followed by frozen recording recovers the mean and covariance diagonal of an
ill-conditioned Gaussian of dimension >= 500, (b) the adapted map whitens it, (c) negative control: adaptation left on during
recording biases a banana."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import adapt as ad
from bayes_scatter import diagnostics as dg
from bayes_scatter import mrf
from bayes_scatter import samplers as sm

# ---------------------------------------------------------------------------------------------
# The affine map
# ---------------------------------------------------------------------------------------------


def _spd(rng, n, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T + n * np.eye(n))


def test_affine_map_round_trip_chain_rule_and_covariance():
    rng = np.random.default_rng(0)
    n = 4
    cov = _spd(rng, n * n, 0.1)
    center = rng.standard_normal((n, n))
    amap = ad.AffineMap.from_covariance(center, cov)
    np.testing.assert_allclose(amap.covariance, cov, rtol=1e-12)
    assert np.allclose(amap.chol_np, np.tril(amap.chol_np))
    z = jnp.asarray(rng.standard_normal((n, n)))
    th = amap.to_theta(z)
    assert th.shape == (n, n)
    np.testing.assert_allclose(np.asarray(amap.to_zeta(th)), np.asarray(z), atol=1e-12)
    np.testing.assert_allclose(np.asarray(th).ravel(), center.ravel() + amap.chol_np @ np.asarray(z).ravel(), rtol=1e-12)
    # theta ~ N(center, cov) <=> zeta ~ N(0, I): the map of the unit vectors has Gram matrix cov
    M = np.stack([np.asarray(amap.to_theta(jnp.asarray(e.reshape(n, n)))).ravel() - center.ravel() for e in np.eye(n * n)], axis=1)
    np.testing.assert_allclose(M @ M.T, cov, rtol=1e-10)
    # chain rule: d/dzeta f(theta(zeta)) = chol^T grad f
    f = lambda t: jnp.sum(jnp.sin(t) * jnp.arange(n * n, dtype=jnp.float64).reshape(n, n))
    g_direct = jax.grad(lambda zz: f(amap.to_theta(zz)))(z)
    np.testing.assert_allclose(np.asarray(amap.grad_to_zeta(jax.grad(f)(th))), np.asarray(g_direct), rtol=1e-10, atol=1e-12)


def test_affine_map_from_precision_chol_is_the_whitened_map_of_milestone_4(tmp_path):
    rng = np.random.default_rng(1)
    n = 3
    H = _spd(rng, n * n)
    L = np.linalg.cholesky(H)
    center = rng.standard_normal((n, n))
    w = mrf.WhitenedMap(center, L)
    amap = ad.AffineMap.from_precision_chol(center, L)
    np.testing.assert_allclose(amap.covariance, np.linalg.inv(H), rtol=1e-10)
    # the two maps differ by an orthogonal change of zeta (A = L^-T Q), so they give the same theta distribution; the
    # covariances of theta - center agree, and so does the potential of a Gaussian target in either coordinates
    M = np.stack([np.asarray(w.to_theta(jnp.asarray(e.reshape(n, n)))).ravel() - center.ravel() for e in np.eye(n * n)], axis=1)
    np.testing.assert_allclose(M @ M.T, amap.covariance, rtol=1e-9)
    # save / load / digest
    path = tmp_path / "map.npz"
    amap.save(path, eps=0.3)
    back = ad.AffineMap.load(path)
    assert back.digest() == amap.digest() and np.load(path)["eps"] == 0.3
    other = ad.AffineMap.from_covariance(center + 1e-9, amap.covariance)
    assert other.digest() != amap.digest()
    with pytest.raises(ValueError):
        ad.AffineMap(center, np.eye(3))


def test_zeta_phi_fn_is_the_sampler_potential_of_the_affine_change_of_variables():
    rng = np.random.default_rng(2)
    N = 6
    P = _spd(rng, N)
    mu = rng.standard_normal(N)
    potential = lambda th, aux: (0.5 * (th - mu) @ P @ (th - mu), P @ (th - mu), aux)
    amap = ad.AffineMap.from_covariance(rng.standard_normal(N), _spd(rng, N, 0.05))
    phi_fn = ad.zeta_phi_fn(potential, amap.center, amap.chol)
    z = jnp.asarray(rng.standard_normal(N))
    phi, grad, aux = phi_fn(z, "aux")
    assert aux == "aux"
    u, _, _ = potential(amap.to_theta(z), None)
    np.testing.assert_allclose(float(phi) + 0.5 * float(z @ z), float(u), rtol=1e-12)  # Phi' = U - |zeta|^2/2
    auto = jax.grad(lambda zz: phi_fn(zz, None)[0])(z)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(auto), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------------------------
# Trajectories and dual averaging
# ---------------------------------------------------------------------------------------------


def test_number_of_leapfrog_steps_is_uniform_on_half_to_three_halves_of_L0():
    for eps, tl, cap, want in ((0.35, 1.75, 100, (5, 3, 7)), (0.5, 1.75, 100, (4, 2, 6)), (1.0, 1.75, 100, (2, 1, 3)),
                               (10.0, 1.75, 100, (1, 1, 1)), (1e-4, 1.75, 40, (40, 20, 40))):
        l0, lo, hi = (int(v) for v in ad.n_leapfrog_range(jnp.asarray(eps), tl, cap))
        assert (l0, lo, hi) == want, (eps, l0, lo, hi)
    # the drawn lengths cover {3, ..., 7} uniformly for eps = 0.35 (L0 = 5), whatever the state, and the count is returned
    phi_fn = lambda q, aux: (0.0, jnp.zeros_like(q), aux)
    state = sm.ChainState(jnp.zeros(3), jnp.zeros(()), jnp.zeros(3), jnp.zeros(()))
    keys = jax.random.split(jax.random.key(0), 6000)
    n_steps = np.asarray(jax.vmap(lambda k: ad.trajectory(k, state, 0.35, phi_fn, 1.75, 100)[3])(keys))
    assert set(n_steps.tolist()) == {3, 4, 5, 6, 7}
    freq = np.bincount(n_steps)[3:] / n_steps.size
    np.testing.assert_allclose(freq, 0.2, atol=0.025)


def test_dynamic_leapfrog_equals_the_fixed_length_integrator():
    P = jnp.asarray(np.diag([1.0, 4.0, 0.25]))
    phi_fn = lambda q, aux: (0.5 * q @ (P - jnp.eye(3)) @ q, (P - jnp.eye(3)) @ q, aux)  # Phi' so that H = 1/2 q^T P q
    q, p = jnp.array([0.3, -0.2, 1.1]), jnp.array([0.5, 0.1, -0.7])
    g0 = phi_fn(q, None)[1]
    for n_steps in (1, 4, 9):
        a = sm.leapfrog(phi_fn, q, p, g0, 0.0, 0.17, n_steps)
        b = ad.leapfrog_dynamic(phi_fn, q, p, g0, 0.0, 0.17, jnp.asarray(n_steps))
        for x, y in zip(a, b):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-13, atol=1e-14)


def test_dual_averaging_algebra_and_convergence_to_the_target_acceptance():
    da = ad.da_init(0.3)
    np.testing.assert_allclose(float(jnp.exp(da.log_eps)), 0.3)
    np.testing.assert_allclose(float(da.mu), np.log(3.0))
    # one update by hand: t = 1, w = 1/11, h_bar = (0.8 - alpha)/11, log eps = mu - sqrt(1)/gamma h_bar
    new = ad.da_update(da, jnp.asarray(0.5), 0.8)
    h_bar = 0.3 / 11.0
    np.testing.assert_allclose(float(new.h_bar), h_bar)
    np.testing.assert_allclose(float(new.log_eps), np.log(3.0) - h_bar / 0.05)
    np.testing.assert_allclose(float(new.log_eps_bar), float(new.log_eps))  # eta = 1 at t = 1
    # a chain on a 30-d standard normal adapts its step so that the acceptance probability is ~0.8
    rng = np.random.default_rng(0)
    N = 30
    potential = lambda th, aux: (0.5 * th @ th, th, aux)
    cfg = ad.AdaptConfig(init_buffer=400, n_windows=0, final_buffer=0, block=50, step0=0.05, target_accept=0.8)
    res = ad.adapt_map(potential, jax.random.key(0), 3 * rng.standard_normal(N), jnp.zeros(()), ad.AffineMap(np.zeros(N), np.eye(N)),
                       None, cfg)
    h = res.history[0]
    assert h["trajectories"] == 400 and 0.7 < h["alpha_mean"] < 0.9 and 0.7 < h["accept_mean"] < 0.9
    assert 0.4 < res.eps < 1.5  # the step of a unit-scale Gaussian
    # the dual-averaging bound on the step size
    assert ad.LOG_EPS_MIN < float(ad.da_update(da, jnp.asarray(0.0), 0.8).log_eps) < ad.LOG_EPS_MAX


def test_hmc_with_jittered_trajectory_length_is_exact_for_a_correlated_gaussian():
    """Moments and gradient-evaluation counts of ``sample_chain`` in the coordinates of a (deliberately wrong) affine map."""
    rng = np.random.default_rng(3)
    N = 8
    P = _spd(rng, N, 0.3)
    Sigma = np.linalg.inv(P)
    mu = rng.standard_normal(N)
    potential = lambda th, aux: (0.5 * (th - mu) @ P @ (th - mu), P @ (th - mu), aux)
    amap = ad.AffineMap.from_covariance(np.zeros(N), 3.0 * np.diag(np.diag(Sigma)))  # wrong scale and no correlations
    phi_fn = ad.zeta_phi_fn(potential, amap.center, amap.chol)
    run = jax.jit(lambda k, x: ad.sample_chain(k, phi_fn, x, jnp.zeros(()), eps=0.25, traj_length=1.75, max_leapfrog=50, n_samples=6000,
                                                thin=2, record_fn=amap.to_theta))
    out = run(jax.random.key(4), amap.to_zeta(jnp.zeros(N)))
    th = np.asarray(out["samples"])
    assert th.shape == (3000, N) and np.asarray(out["n_leapfrog"]).shape == (6000,)
    assert set(np.unique(np.asarray(out["n_leapfrog"]))) == {4, 5, 6, 7, 8, 9, 10}  # L0 = 7: 4 .. 10
    ess = dg.ess(th)
    np.testing.assert_allclose(th.mean(axis=0), mu, atol=float(np.max(4 * np.sqrt(np.diag(Sigma) / ess))))
    np.testing.assert_allclose(th.var(axis=0, ddof=1), np.diag(Sigma), rtol=0.15)
    np.testing.assert_allclose(np.asarray(out["mean"]), mu, atol=float(np.max(4 * np.sqrt(np.diag(Sigma) / ess))))  # Welford, all draws
    assert 0.5 < float(np.asarray(out["accept"]).mean()) < 1.0


def test_sample_chain_is_reproducible_and_its_thinning_and_welford_moments_are_consistent():
    N = 5
    potential = lambda th, aux: (0.5 * th @ th, th, aux)
    phi_fn = ad.zeta_phi_fn(potential, jnp.zeros(N), jnp.eye(N))
    kw = dict(eps=0.4, traj_length=1.75, max_leapfrog=30, n_samples=101)
    x0 = jnp.ones(N)
    a = ad.sample_chain(jax.random.key(0), phi_fn, x0, jnp.zeros(()), thin=1, **kw)
    b = ad.sample_chain(jax.random.key(0), phi_fn, x0, jnp.zeros(()), thin=1, **kw)
    c = ad.sample_chain(jax.random.key(0), phi_fn, x0, jnp.zeros(()), thin=3, **kw)
    np.testing.assert_array_equal(np.asarray(a["samples"]), np.asarray(b["samples"]))
    np.testing.assert_array_equal(np.asarray(c["samples"]), np.asarray(a["samples"])[2:99:3])  # every 3rd; 101 // 3 = 33 stored
    assert c["samples"].shape[0] == 33
    np.testing.assert_allclose(np.asarray(c["mean"]), np.asarray(a["mean"]), atol=1e-12)  # moments use all 101 trajectories
    np.testing.assert_allclose(np.asarray(a["mean"]), np.asarray(a["samples"]).mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(a["var"]), np.asarray(a["samples"]).var(axis=0, ddof=1), atol=1e-12)
    assert np.asarray(a["phi"]).shape == (101,) and np.asarray(a["accept"]).dtype == bool
    np.testing.assert_array_equal(np.asarray(a["xi"]), np.asarray(a["samples"])[-1])
    with pytest.raises(ValueError):
        ad.sample_chain(jax.random.key(0), phi_fn, x0, jnp.zeros(()), eps=0.4, traj_length=1.0, max_leapfrog=5, n_samples=0)


# ---------------------------------------------------------------------------------------------
# The map update
# ---------------------------------------------------------------------------------------------


def test_update_map_is_the_shrinkage_of_the_window_covariance_toward_the_laplace_covariance():
    rng = np.random.default_rng(5)
    N, n = 12, 7  # fewer draws than dimensions: the sample covariance is singular, the Laplace term makes the map valid
    T = _spd(rng, N, 0.05)
    draws = rng.standard_normal((n, N)) + 3.0
    prev = ad.AffineMap.from_covariance(np.zeros(N), np.eye(N))
    calls = []
    new, info = ad.update_map(draws, lambda c: calls.append(c) or T, kappa=N, prev=prev)
    m = draws.mean(axis=0)
    S = np.cov(draws.T)
    w = N / (n + N)
    np.testing.assert_allclose(calls[0], m)  # the Laplace covariance is asked for at the NEW center
    np.testing.assert_allclose(new.center_np, m)
    np.testing.assert_allclose(new.covariance, (1 - w) * S + w * T, rtol=1e-10, atol=1e-12)
    assert np.linalg.matrix_rank(S) < N and np.linalg.eigvalsh(new.covariance).min() > 0
    assert info["weight"] == pytest.approx(w) and info["n_draws"] == n
    np.testing.assert_allclose(info["trace_ratio"], np.trace(new.covariance) / N)
    np.testing.assert_allclose(info["trace_sample_over_laplace"], np.trace(S) / np.trace(T))
    assert info["var_ratio_min"] <= info["var_ratio_median"] <= info["var_ratio_max"]
    np.testing.assert_allclose(info["center_shift"], np.linalg.norm(m))  # prev is the identity map at 0
    assert {"time_s", "time_laplace_s", "time_sample_cov_s", "time_mix_cholesky_s", "pivot_ratio_sq"} <= set(info)
    # long windows are dominated by their own draws: the weight kappa / (n + kappa) decays
    big = rng.standard_normal((4000, N)) @ np.linalg.cholesky(T).T * 2.0
    new2, info2 = ad.update_map(big, lambda c: T, kappa=N)
    assert info2["weight"] < 0.003
    np.testing.assert_allclose(new2.covariance, 4.0 * T, rtol=0.0, atol=0.2 * 4 * np.abs(T).max())
    assert ad.shrinkage_weight(100, 25) == pytest.approx(0.2)


def test_laplace_covariance_inverts_the_precision_from_its_cholesky_factor():
    rng = np.random.default_rng(6)
    H = _spd(rng, 9)
    np.testing.assert_allclose(ad.laplace_covariance(np.linalg.cholesky(H)), np.linalg.inv(H), rtol=1e-10)


# ---------------------------------------------------------------------------------------------
# The adaptation driver: schedule, bookkeeping, freezing
# ---------------------------------------------------------------------------------------------


def test_schedule_rounds_phases_up_to_whole_blocks_and_expands_windows():
    cfg = ad.AdaptConfig(init_buffer=30, first_window=40, n_windows=3, growth=2.0, final_buffer=1, block=25)
    assert cfg.phases() == [("init", 2, False), ("window0", 2, True), ("window1", 4, True), ("window2", 7, True), ("final", 1, False)]
    assert cfg.total_trajectories() == (2 + 2 + 4 + 7 + 1) * 25
    assert ad.AdaptConfig(kappa=7.0).kappa_for(100) == 7.0 and ad.AdaptConfig().kappa_for(100) == 100.0
    equal = ad.AdaptConfig(init_buffer=0, first_window=10, n_windows=3, growth=1.0, final_buffer=0, block=10).phases()
    assert equal == [("window0", 1, True), ("window1", 1, True), ("window2", 1, True)]


def test_adapt_map_updates_the_map_only_in_window_phases_and_accounts_for_every_gradient():
    rng = np.random.default_rng(7)
    N = 10
    P = _spd(rng, N, 0.2)
    calls = []

    def potential(th, aux):
        jax.debug.callback(lambda: calls.append(1))  # counts every evaluation that is actually run
        return 0.5 * th @ P @ th, P @ th, aux

    map0 = ad.AffineMap.from_covariance(np.zeros(N), np.eye(N))
    cfg = ad.AdaptConfig(init_buffer=25, first_window=25, n_windows=2, growth=2.0, final_buffer=25, block=25, kappa=N)
    res = ad.adapt_map(potential, jax.random.key(1), np.zeros(N), jnp.zeros(()), map0, lambda c: np.eye(N), cfg, keep_draws=True)
    assert [h["phase"] for h in res.history] == ["init", "window0", "window1", "final"]
    assert [h["updated"] for h in res.history] == [False, True, True, False]
    # the map is frozen in the last phase and is the one the result carries
    assert res.history[-1]["map_digest_before"] == res.map.digest()[:16] == res.history[-2]["map_digest_after"]
    assert res.history[0]["map_digest_before"] == map0.digest()[:16]
    # every gradient evaluation is counted: 1 at the start, sum of the trajectory lengths, 1 per re-evaluation after an update
    assert res.n_grad_evals == 1 + sum(h["n_grad_evals"] for h in res.history) + 2
    jax.effects_barrier()
    assert len(calls) == res.n_grad_evals  # the potential itself counted them
    assert [d.shape for d in res.draws] == [(25, N), (25, N), (50, N), (25, N)]
    assert np.all(np.isfinite(res.theta)) and res.eps > 0 and res.n_leapfrog0 >= 1


def test_a_window_with_no_acceptance_does_not_update_the_map():
    N = 4
    potential = lambda th, aux: (0.5 * th @ th, th, aux)
    map0 = ad.AffineMap(np.zeros(N), np.eye(N))
    cfg = ad.AdaptConfig(init_buffer=0, first_window=25, n_windows=1, final_buffer=0, block=25, min_accept_update=1.01)
    res = ad.adapt_map(potential, jax.random.key(0), np.ones(N), jnp.zeros(()), map0, lambda c: pytest.fail("called"), cfg)
    assert res.history[0]["updated"] is False and "stuck" in res.history[0]["skipped"]
    assert res.map.digest() == map0.digest()


# ---------------------------------------------------------------------------------------------
# DESIGN 12.2 validation: an ill-conditioned Gaussian of dimension >= 500
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ill_conditioned_gaussian():
    """A 23 x 23 GMRF (4-neighbor graph Laplacian + 6e-4 I) as a Gaussian target in N = 529 dimensions, condition number of the
    precision 1.3e4, mean mu (random). The Laplace-like start and shrinkage target is a GMRF with 100 times the anchor (a correlation
    length ~10 times shorter: the variance of its longest-wave modes is ~100 times too small, the order of the mismatch of the
    milestone-4 GMRF, whose local Hessian along the chains was 270 times the Laplace one).
    Adaptation (5 expanding windows from 75 trajectories), then 4 overdispersed chains of 1500 recorded trajectories (thin 2)
    with the map and step frozen."""
    m, eps_g, wrong = 23, 6e-4, 0.06
    N = m * m
    idx = np.arange(N).reshape(m, m)
    Lap = np.zeros((N, N))
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        np.add.at(Lap, (a, a), 1.0)
        np.add.at(Lap, (b, b), 1.0)
        np.add.at(Lap, (a, b), -1.0)
        np.add.at(Lap, (b, a), -1.0)
    P = Lap + eps_g * np.eye(N)
    Sigma = np.linalg.inv(P)
    ev = np.linalg.eigvalsh(P)
    mu = np.random.default_rng(0).standard_normal(N)
    Pj, muj = jnp.asarray(P), jnp.asarray(mu)

    def potential(theta, aux):
        d = theta - muj
        g = Pj @ d
        return 0.5 * d @ g, g, aux

    Sigma_wrong = np.linalg.inv(Lap + wrong * np.eye(N))
    map0 = ad.AffineMap.from_covariance(np.zeros(N), Sigma_wrong)
    cfg = ad.AdaptConfig(init_buffer=50, first_window=75, n_windows=5, growth=2.0, final_buffer=100, block=25)
    res = ad.adapt_map(potential, jax.random.key(1), np.zeros(N), jnp.zeros(()), map0, lambda c: Sigma_wrong, cfg)
    amap = res.map
    phi_fn = ad.zeta_phi_fn(potential, amap.center, amap.chol)
    run = jax.jit(jax.vmap(lambda k, x0: ad.sample_chain(k, phi_fn, x0, jnp.zeros(()), eps=res.eps, traj_length=res.traj_length,
                                                         max_leapfrog=res.max_leapfrog, n_samples=1500, thin=2,
                                                         record_fn=amap.to_theta)))
    keys = jax.random.split(jax.random.key(5), 4)
    starts = jnp.stack([2.0 * jax.random.normal(jax.random.key(100 + c), (N,)) for c in range(4)])
    out = jax.block_until_ready(run(keys, starts))
    return {"N": N, "P": P, "Sigma": Sigma, "mu": mu, "cond_P": ev.max() / ev.min(), "map0": map0, "res": res,
            "draws": np.asarray(out["samples"]), "n_leapfrog": np.asarray(out["n_leapfrog"]), "accept": np.asarray(out["accept"]),
            "Sigma_wrong": Sigma_wrong}


def _whitened_spectrum(amap, Sigma):
    """Eigenvalues of chol^-1 Sigma chol^-T: the covariance of zeta = chol^-1 (theta - center) if theta ~ N(., Sigma)."""
    Ai = np.linalg.solve(amap.chol_np, np.eye(Sigma.shape[0]))
    W = Ai @ Sigma @ Ai.T
    return np.linalg.eigvalsh(0.5 * (W + W.T))


def test_gaussian_target_has_dimension_over_500_and_condition_number_over_1e4(ill_conditioned_gaussian):
    g = ill_conditioned_gaussian
    assert g["N"] >= 500 and g["cond_P"] >= 1e4


def test_a_adaptation_then_frozen_recording_recovers_the_mean_and_covariance_diagonal_within_mc_error(ill_conditioned_gaussian):
    g = ill_conditioned_gaussian
    N, mu, Sigma = g["N"], g["mu"], g["Sigma"]
    th = g["draws"][:, 50:]  # (chains, draws, N), the first 100 trajectories of every chain discarded as burn-in
    ess_m = np.stack([dg.ess(c) for c in th]).sum(axis=0)
    z_mean = (th.reshape(-1, N).mean(axis=0) - mu) / (np.sqrt(np.diag(Sigma)) / np.sqrt(ess_m))
    sq = (th - mu) ** 2
    ess_v = np.stack([dg.ess(c) for c in sq]).sum(axis=0)
    v_hat = sq.reshape(-1, N).mean(axis=0)
    z_var = (v_hat - np.diag(Sigma)) / (sq.reshape(-1, N).std(axis=0, ddof=1) / np.sqrt(ess_v))
    # under the null the z-scores are N(0, 1) up to the error of the ESS estimates: rms ~ 1, no |z| above ~4
    # (the Sokal ESS of the slowest, longest-wave coordinates is optimistic for 4 x 700 draws: rms z of the means is 1.8, not 1)
    assert np.sqrt(np.mean(z_mean**2)) < 2.5 and np.abs(z_mean).max() < 6.0
    assert np.sqrt(np.mean(z_var**2)) < 2.0 and np.abs(z_var).max() < 5.5
    assert abs(np.mean(v_hat / np.diag(Sigma)) - 1.0) < 0.05  # the average variance is right to 5%
    # the chains began 2 sd from the map center and the recording is stationary: first and second half of the draws agree
    half = th.shape[1] // 2
    d = th[:, :half].mean(axis=1) - th[:, half:].mean(axis=1)
    assert np.sqrt(np.mean((d / np.sqrt(np.diag(Sigma))) ** 2)) < 0.5
    # the chains are healthy: acceptance near the 0.8 target, jittered lengths around L0
    assert 0.6 < g["accept"].mean() < 0.95
    l0 = g["res"].n_leapfrog0
    assert set(np.unique(g["n_leapfrog"])) <= set(range(max(l0 // 2, 1), int(1.5 * l0) + 1))


def test_b_the_adapted_map_whitens_the_gaussian(ill_conditioned_gaussian):
    g = ill_conditioned_gaussian
    N, Sigma = g["N"], g["Sigma"]
    res = g["res"]
    w_before = _whitened_spectrum(g["map0"], Sigma)
    w_after = _whitened_spectrum(res.map, Sigma)
    cond_before, cond_after = w_before.max() / w_before.min(), w_after.max() / w_after.min()
    assert cond_before > 50  # the shrinkage target alone is wrong by ~100 in the spectrum of the whitened covariance
    assert cond_after < 25 and cond_after < cond_before / 4
    assert 0.2 < w_after.min() and w_after.max() < 6.0
    # the sample covariance of zeta = chol^-1 (theta - center) over the recorded draws is near I: diagonal near 1 (the map is a
    # little narrow by the weight of the Laplace term, see AdaptConfig), eigenvalues within the same band as the exact ones widened by
    # the sampling error of 5000 correlated draws in 529 dimensions
    th = g["draws"][:, 50:].reshape(-1, N)
    Z = np.linalg.solve(res.map.chol_np, (th - res.map.center_np[None, :]).T).T
    C = np.cov(Z.T)
    assert 0.8 < np.mean(np.diag(C)) < 2.2 and np.diag(C).min() > 0.5 and np.diag(C).max() < 3.5
    ev = np.linalg.eigvalsh(C)
    assert ev.min() > 0.08 and ev.max() < 10.0
    # the first window learned the largest scales first: the variance ratios of the updates are of order one by the last window
    ups = [h["update"] for h in res.history if h["updated"]]
    assert len(ups) == 5 and ups[0]["weight"] > 0.8 > ups[-1]["weight"] > 0.1
    assert all(0.2 < u["var_ratio_median"] < 5.0 for u in ups)
    # the map is frozen after the last window: the final phase did not update it
    assert res.history[-1]["updated"] is False and res.history[-1]["map_digest_before"] == res.map.digest()[:16]


def test_adapted_map_samples_far_more_efficiently_than_the_unadapted_one(ill_conditioned_gaussian):
    """Per gradient evaluation, one chain's worst-coordinate ESS under the adapted map exceeds the Laplace-like start's by >5x."""
    g = ill_conditioned_gaussian
    N = g["N"]
    P, muj = jnp.asarray(g["P"]), jnp.asarray(g["mu"])

    def potential(theta, aux):
        d = theta - muj
        gr = P @ d
        return 0.5 * d @ gr, gr, aux

    map0 = g["map0"]
    res0 = ad.adapt_map(potential, jax.random.key(2), np.zeros(N), jnp.zeros(()), map0, None,
                        ad.AdaptConfig(init_buffer=100, n_windows=0, final_buffer=0, block=25))  # step size only, Laplace-like map
    phi0 = ad.zeta_phi_fn(potential, map0.center, map0.chol)
    out0 = jax.jit(lambda k, x: ad.sample_chain(k, phi0, x, jnp.zeros(()), eps=res0.eps, traj_length=res0.traj_length,
                                                max_leapfrog=res0.max_leapfrog, n_samples=1500, thin=2,
                                                record_fn=map0.to_theta))(jax.random.key(3), map0.to_zeta(jnp.asarray(res0.theta)))
    ess0 = dg.ess(np.asarray(out0["samples"])[50:])
    used0 = int(np.asarray(out0["n_leapfrog"]).sum())
    ess1 = dg.ess(g["draws"][0][50:])
    used1 = int(g["n_leapfrog"][0].sum())  # the adapted chains also ran 1500 trajectories
    assert 0.5 < used1 / used0 < 2.0
    assert (ess1.min() / used1) > 5.0 * (ess0.min() / used0), (ess1.min() / used1, ess0.min() / used0)


# ---------------------------------------------------------------------------------------------
# DESIGN 12.2 negative control: adaptation left on during recording biases a non-Gaussian target
# ---------------------------------------------------------------------------------------------

BANANA_B = 0.01  # x1 ~ N(0, 10^2), x2 | x1 ~ N(-b (x1^2 - 100), 1): E x1^2 = 100, E x2 = 0, E x2^2 = 1 + 2e4 b^2 = 3


def _banana_potential(x, aux):
    def U(x):
        return 0.5 * ((x[0] / 10.0) ** 2 + (x[1] + BANANA_B * (x[0] ** 2 - 100.0)) ** 2)  # unit-Jacobian shear of a Gaussian

    u, g = jax.value_and_grad(U)(x)
    return u, g, aux


def _banana_features(x):
    return np.stack([x[:, 0] ** 2, x[:, 1], x[:, 1] ** 2], axis=1)


BANANA_TRUTH = np.array([100.0, 0.0, 1.0 + 2e4 * BANANA_B**2])
FROZEN_FACTOR = 16  # frozen-arm chains are this many times longer than the adaptation-on arm's (resolution of the control)
# Independent chains per arm. With 8, the standard errors (from the spread of 8 chain means) were themselves so noisy that the
# z scores of the adaptation-on arm had sd ~3 over seeds, and the assertions below failed for 5 of 20 seeds on the Mac and 6 of 20
# on the workstation (seed sweep, 2026-11-08); the test as written passed only because seed 0 was a good one. 32 chains give the
# same thresholds the power they were meant to have.
BANANA_CHAINS = 32


def _banana_chains(adapt_on, n_chains=BANANA_CHAINS, window=25, n_windows=160, burn=100, frozen_factor=1):
    """Feature means of ``n_chains`` independent chains: each adapts (3 expanding windows), then records ``window * n_windows``
    trajectories either with the map and step FROZEN (``sample_chain``; ``frozen_factor`` times as many trajectories) or with
    adaptation LEFT ON (``adapt_map`` windows of ``window`` trajectories, each re-centering at its own mean and re-estimating the
    covariance from its own draws). Returns ``(z, estimate, se)``: the z scores of the three feature means against ``BANANA_TRUTH``,
    the means and their standard errors (std of the chain means / sqrt(n_chains))."""
    map0 = ad.AffineMap.from_covariance(np.zeros(2), 4.0 * np.eye(2))
    lap = lambda c: 4.0 * np.eye(2)
    cfg = ad.AdaptConfig(init_buffer=50, first_window=25, n_windows=3, growth=2.0, final_buffer=50, block=25, max_leapfrog=60)
    cfg_on = ad.AdaptConfig(init_buffer=0, first_window=window, n_windows=n_windows, growth=1.0, final_buffer=0, block=window,
                            max_leapfrog=60)

    @jax.jit
    def frozen(k, x0, center, chol, eps):
        phi_fn = ad.zeta_phi_fn(_banana_potential, center, chol)
        return ad.sample_chain(k, phi_fn, x0, jnp.zeros(()), eps=eps, traj_length=1.75, max_leapfrog=60,
                               n_samples=window * n_windows * frozen_factor, record_fn=lambda z: center + chol @ z)["samples"]

    feats = []
    for seed in range(n_chains):
        k0, k1, k2 = jax.random.split(jax.random.key(seed), 3)
        res = ad.adapt_map(_banana_potential, k1, 2.0 * np.asarray(jax.random.normal(k0, (2,))), jnp.zeros(()), map0, lap, cfg)
        if adapt_on:
            on = ad.adapt_map(_banana_potential, k2, res.theta, res.aux, res.map, lap,
                              dataclasses.replace(cfg_on, step0=res.eps), keep_draws=True)
            draws = np.concatenate(on.draws)
        else:
            draws = np.asarray(frozen(k2, res.map.to_zeta(jnp.asarray(res.theta)), res.map.center, res.map.chol, res.eps))
        feats.append(_banana_features(draws[burn:]).mean(axis=0))
    feats = np.stack(feats)
    se = feats.std(axis=0, ddof=1) / np.sqrt(n_chains)
    return (feats.mean(axis=0) - BANANA_TRUTH) / se, feats.mean(axis=0), se


def test_c_negative_control_adaptation_left_on_during_recording_biases_a_banana_and_freezing_does_not():
    # The frozen arm records FROZEN_FACTOR times as many trajectories as the adaptation-on arm: with the same length its standard error of
    # E x1^2 (13, i.e. 13 %) was as large as the bias of the on arm (-14 %), so the test showed that leaving adaptation on biases the
    # estimate but could not show that freezing does not (review of milestone 5). With the longer chains below the resolution of the check is
    # asserted: a bias of the size the on arm shows would stand out of the frozen arm by at least 3 standard errors in every moment.
    z_frozen, est_frozen, se_frozen = _banana_chains(adapt_on=False, frozen_factor=FROZEN_FACTOR)
    z_on, est_on, _ = _banana_chains(adapt_on=True)
    # frozen map and step: an exact HMC chain for a fixed affine change of variables, unbiased within Monte Carlo error
    # (the z scores are not standard normal: these moments are heavy-tailed, so a seed whose chains reach far into the tail inflates the
    # mean and its estimated standard error together and z is skewed. Over 60 seeds of 32 chains: z sd 1.3 to 1.5, mean shifts -0.8 to
    # +0.5, worst |z| 5.08; at 4x the chain length the shift shrank instead of doubling, and the pooled means were 100.010, -0.0001 and
    # 3.0001 against 100, 0 and 3, so the frozen arm is unbiased and the bound is on the skewed statistic, 2026-11-08)
    assert np.abs(z_frozen).max() < 6.0, (z_frozen, est_frozen)
    # adaptation on (state-dependent kernel): E x1^2 and E x2^2 are low and E x2 is high by many standard errors
    assert z_on[0] < -6.0 and z_on[1] > 4.0 and z_on[2] < -4.0, (z_on, est_on)
    assert est_on[0] < 0.92 * BANANA_TRUTH[0]  # a 8%+ bias in a second moment, 4 times the frozen run's
    # resolution of the frozen arm: the bias of the on arm is at least 3 standard errors of the frozen arm in each moment
    bias_on = np.abs(est_on - BANANA_TRUTH)
    assert np.all(bias_on > 3.0 * se_frozen), (bias_on, se_frozen)
    # and the frozen arm's own deviation from the truth is a small fraction of the on arm's bias
    assert np.all(np.abs(est_frozen - BANANA_TRUTH) < 0.7 * bias_on), (est_frozen, est_on)
