"""Tests for bayes_scatter.samplers: pCN, pCN-Langevin, MALA, HMC.

Main target (linear-Gaussian, whitened coordinates, dimension 50)::

    Phi(xi) = ||B xi - y||^2 / (2 s^2),   B: 80 x 50,   prior xi ~ N(0, I),

so the exact posterior is N(m, Lambda^-1) with Lambda = I + B^T B / s^2 and m = Lambda^-1 B^T y / s^2.
B has i.i.d. N(0, 1/80) entries and s = 0.6; the posterior variance along the eigenvectors of Lambda
is 1 / (1 + ev(B^T B) / s^2), i.e. between 0.11 and 0.88 times the prior variance (asserted below),
and the diagonal of the posterior covariance is between 0.29 and 0.47. So the data are informative
(variance cut by up to 9x in the best-determined direction, 12% in the worst) while the prior still
matters, and pCN at its tuned step (about 0.19) still accepts about 25%.

Every correct sampler must pass three checks, evaluated on 4 chains (see ``Report``):

* mean:     z = (mean_hat - m) / MCSE over the 50 coordinates, with MCSE from ``mcse_mean`` on each
            chain and combined across chains. For a correct sampler z ~ N(0, 1) per coordinate (the
            coordinates are correlated, so only loose bounds are used): |mean z| < 1, rms z < 2,
            max |z| < 5.
* variance: ratio of the Welford variance to the exact marginal variance. Each ratio has Monte Carlo sd
            of about 4% (up to 8% in the slowest direction), the average over 50 coordinates and 4
            chains about 1%: the average must lie in [0.95, 1.05] and every coordinate in [0.8, 1.2].
* accept:   mean post-warmup acceptance within 0.05 of the sampler's adaptation target (the last
            Robbins-Monro iterate fluctuates by a few percent).

NEGATIVE CONTROLS: the real kernels are run with a single fault injected into the Metropolis-Hastings
step (``samplers._mh_step`` is replaced during tracing), e.g. a flipped sign or a missing proposal-density
correction. Each broken variant must fail the specific checks listed in ``BROKEN``, while the same
harness without the fault passes all of them (``test_harness_without_fault_passes``).
"""

import contextlib
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import diagnostics as dg
from bayes_scatter import samplers as sm

# ---------------------------------------------------------------------------------------------
# Linear-Gaussian problem
# ---------------------------------------------------------------------------------------------

D, M, S_NOISE, C = 50, 80, 0.6, 4
_rng = np.random.default_rng(0)
B = _rng.standard_normal((M, D)) / np.sqrt(M)
Y = B @ _rng.standard_normal(D) + S_NOISE * _rng.standard_normal(M)
PRECISION = np.eye(D) + B.T @ B / S_NOISE**2
COV = np.linalg.inv(PRECISION)
MEAN = COV @ B.T @ Y / S_NOISE**2
_B, _Y = jnp.asarray(B), jnp.asarray(Y)


def phi_fn(xi, aux):
    """Potential and gradient of the linear-Gaussian likelihood; aux is unused."""
    r = _B @ xi - _Y
    return 0.5 * jnp.sum(r * r) / S_NOISE**2, _B.T @ r / S_NOISE**2, aux


KEYS = jax.random.split(jax.random.key(1), C)
XI0 = jax.random.normal(jax.random.key(2), (C, D))

# name -> (sampler, step0, n_warmup, n_samples, adaptation target, extra kwargs)
SAMPLERS = {
    "pcn": (sm.pcn, 0.2, 1000, 40000, 0.25, {}),
    "pcn_langevin": (sm.pcn_langevin, 0.05, 1000, 10000, 0.5, {}),
    "mala": (sm.mala, 0.1, 1000, 10000, 0.574, {}),
    # jitter: see the HMC docstring and test_hmc_jitter_removes_resonance
    "hmc": (sm.hmc, 0.2, 1000, 3000, 0.7, {"n_leapfrog": 10, "jitter": 0.2}),
}

Z_MEAN_MAX, Z_RMS_MAX, Z_ABS_MAX = 1.0, 2.0, 5.0
VAR_AVG_BAND, VAR_DIM_BAND, ACC_TOL = (0.95, 1.05), (0.8, 1.2), 0.05


class Report:
    """Accuracy of C chains against the exact posterior (see module docstring)."""

    def __init__(self, res, target):
        x = np.asarray(res.samples)  # (C, N, D)
        mcse = np.stack([dg.mcse_mean(xc) for xc in x])
        self.z = (x.mean(axis=1).mean(axis=0) - MEAN) / (np.sqrt((mcse**2).sum(axis=0)) / x.shape[0])
        self.var_ratio = np.asarray(res.var).mean(axis=0) / np.diag(COV)
        self.accept = float(np.asarray(res.accept).mean())
        self.target = target
        self.ess_min = np.stack([dg.ess(xc) for xc in x]).sum(axis=0).min()
        self.rhat_max = dg.split_rhat(x).max()

    def checks(self):
        z, v = self.z, self.var_ratio
        return {
            "mean": abs(z.mean()) < Z_MEAN_MAX and np.sqrt((z**2).mean()) < Z_RMS_MAX
            and np.abs(z).max() < Z_ABS_MAX,
            "var": VAR_AVG_BAND[0] < v.mean() < VAR_AVG_BAND[1]
            and v.min() > VAR_DIM_BAND[0] and v.max() < VAR_DIM_BAND[1],
            "accept": abs(self.accept - self.target) < ACC_TOL,
        }

    def __str__(self):
        return (f"z mean {self.z.mean():.2f} rms {np.sqrt((self.z**2).mean()):.2f} "
                f"max {np.abs(self.z).max():.2f}; var ratio mean {self.var_ratio.mean():.3f} "
                f"[{self.var_ratio.min():.3f}, {self.var_ratio.max():.3f}]; accept "
                f"{self.accept:.3f} (target {self.target}); min ESS {self.ess_min:.0f}; "
                f"max rhat {self.rhat_max:.3f}")


_cache = {}


def run_sampler(name):
    """Run (once per session) one of the four public samplers on the linear-Gaussian target."""
    if name not in _cache:
        fn, step0, n_warmup, n_samples, target, kw = SAMPLERS[name]
        res = sm.run_chains(fn, KEYS, XI0, None, phi_fn=phi_fn, n_warmup=n_warmup,
                            n_samples=n_samples, step0=step0, **kw)
        _cache[name] = (res, Report(res, target))
    return _cache[name]


def test_problem_is_informative_but_not_degenerate():
    ev = np.linalg.eigvalsh(COV)  # posterior / prior variance along principal axes (prior = I)
    assert 0.09 < ev.min() < 0.13 and 0.85 < ev.max() < 0.90
    assert 0.28 < np.diag(COV).min() and np.diag(COV).max() < 0.48
    # the MAP differs substantially from the prior mean, so the mean test has power
    assert np.linalg.norm(MEAN) > 3.0


# ---------------------------------------------------------------------------------------------
# (a), (b), (c): correct samplers recover the exact posterior
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_posterior_mean(name):
    res, rep = run_sampler(name)
    assert rep.checks()["mean"], str(rep)
    # the stored samples (thin=1) and the Welford mean agree
    np.testing.assert_allclose(np.asarray(res.mean), np.asarray(res.samples).mean(axis=1), atol=1e-10)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_posterior_variance(name):
    res, rep = run_sampler(name)
    assert rep.checks()["var"], str(rep)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_acceptance_near_target_and_step_positive(name):
    res, rep = run_sampler(name)
    assert rep.checks()["accept"], str(rep)
    assert np.all(np.asarray(res.step) > 0)
    if name == "pcn":
        assert np.all(np.asarray(res.step) <= 1.0)


@pytest.mark.parametrize("name", list(SAMPLERS))
def test_chains_converge_to_the_same_distribution(name):
    # started from 4 different random points; split R-hat near 1 for every coordinate
    _, rep = run_sampler(name)
    assert rep.rhat_max < 1.03, str(rep)


def test_hmc_jitter_removes_resonance():
    # With a fixed trajectory length some eigen-directions (period dividing the trajectory time) barely
    # move; the jitter option fixes it. Same settings as SAMPLERS["hmc"], jitter off.
    fn, step0, n_warmup, n_samples, target, kw = SAMPLERS["hmc"]
    res0 = sm.run_chains(fn, KEYS, XI0, None, phi_fn=phi_fn, n_warmup=n_warmup, n_samples=n_samples,
                         step0=step0, n_leapfrog=10, jitter=0.0)
    rep0, rep1 = Report(res0, target), run_sampler("hmc")[1]
    assert rep1.ess_min > 2.0 * rep0.ess_min, f"no jitter: {rep0}\njitter:    {rep1}"


# ---------------------------------------------------------------------------------------------
# Negative controls: the real kernels with one fault injected into the MH step
# ---------------------------------------------------------------------------------------------


def _u_diff(state, prop):
    """U(u) - U(v), U = Phi + |xi|^2 / 2: the log ratio of the target alone, without proposal terms."""
    return (state.phi + 0.5 * jnp.sum(state.xi**2)) - (prop.phi + 0.5 * jnp.sum(prop.xi**2))


def _faulty_mh(fault):
    """Replacement for ``samplers._mh_step`` with log_ratio -> fault(state, proposal, log_ratio)."""
    orig = sm._mh_step
    return lambda key, state, prop, lr: orig(key, state, prop, fault(state, prop, lr))


@contextlib.contextmanager
def _patched_mh(replacement):
    orig = sm._mh_step
    sm._mh_step = replacement
    try:
        yield
    finally:
        sm._mh_step = orig


def run_harness(kernel, replacement=None, *, step0, n_warmup, n_samples, target, log_step_max):
    """Run C chains of ``_run_chain(kernel)``, optionally with a replaced MH step (traced inside)."""
    with _patched_mh(replacement or sm._mh_step):
        run = jax.jit(jax.vmap(lambda k, x: sm._run_chain(
            kernel, k, phi_fn, x, None, n_warmup=n_warmup, n_samples=n_samples, step0=step0,
            thin=1, record_fn=sm.identity, target_accept=target, log_step_max=log_step_max)))
        res = run(KEYS, XI0)
        jax.block_until_ready(res.samples)
    return Report(res, target)


HMC_K = partial(sm._hmc_kernel, n_leapfrog=10, jitter=0.2)
ALWAYS = lambda st, p, lr: jnp.inf  # accept every proposal  # noqa: E731
FLIP = lambda st, p, lr: -lr  # noqa: E731
INF = jnp.inf

# name -> (kernel, fault, run settings, checks that must FAIL). "adapt" settings let the step adapt to
# the usual target; "fixed" settings freeze the step at the value the correct sampler adapts to, so the
# bias of the broken sampler cannot be hidden by a tiny adapted step.
_PCN = dict(step0=0.2, n_warmup=1000, n_samples=40000, target=0.25, log_step_max=0.0)
_PCNL = dict(step0=0.05, n_warmup=1000, n_samples=10000, target=0.5, log_step_max=INF)
_MALA = dict(step0=0.1, n_warmup=1000, n_samples=10000, target=0.574, log_step_max=INF)
_HMC = dict(step0=0.2, n_warmup=1000, n_samples=3000, target=0.7, log_step_max=INF)
_fixed = lambda base, h: {**base, "step0": h, "n_warmup": 0}  # noqa: E731

BROKEN = {
    "pcn: acceptance sign flipped": (
        sm._pcn_kernel, FLIP, _PCN, {"mean", "var", "accept"}),
    "pcn: prior term wrongly included": (
        sm._pcn_kernel,
        lambda st, p, lr: lr - 0.5 * (jnp.sum(p.xi**2) - jnp.sum(st.xi**2)), _PCN, {"mean", "var"}),
    "pcn_langevin: sign flipped": (sm._pcn_langevin_kernel, FLIP, _PCNL, {"var"}),
    "pcn_langevin: no proposal-density correction": (
        sm._pcn_langevin_kernel, lambda st, p, lr: _u_diff(st, p), _fixed(_PCNL, 0.10), {"var", "accept"}),
    "mala: sign flipped": (sm._mala_kernel, FLIP, _MALA, {"var"}),
    "mala: no proposal-density correction": (
        sm._mala_kernel, lambda st, p, lr: _u_diff(st, p), _fixed(_MALA, 0.15), {"var", "accept"}),
    "mala: unadjusted (ULA) at the tuned step": (
        sm._mala_kernel, ALWAYS, _fixed(_MALA, 0.15), {"var", "accept"}),
    "mala: unadjusted (ULA) at a large step": (
        sm._mala_kernel, ALWAYS, _fixed(_MALA, 0.30), {"var", "accept"}),
    "hmc: sign flipped": (HMC_K, FLIP, _HMC, {"var"}),
    "hmc: potential energy only (no kinetic term)": (
        HMC_K, lambda st, p, lr: _u_diff(st, p), _fixed(_HMC, 0.35), {"var", "accept"}),
}

# Same harness and settings without the fault: must pass everything.
CLEAN = {
    "pcn": (sm._pcn_kernel, _PCN),
    "pcn_langevin": (sm._pcn_langevin_kernel, _PCNL),
    "mala": (sm._mala_kernel, _MALA),
    "hmc": (HMC_K, _HMC),
    "pcn_langevin fixed": (sm._pcn_langevin_kernel, _fixed(_PCNL, 0.10)),
    "mala fixed": (sm._mala_kernel, _fixed(_MALA, 0.15)),
    "hmc fixed": (HMC_K, _fixed(_HMC, 0.35)),
}


@pytest.mark.parametrize("name", list(CLEAN))
def test_harness_without_fault_passes(name):
    kernel, settings = CLEAN[name]
    rep = run_harness(kernel, None, **settings)
    assert all(rep.checks().values()), f"{rep.checks()} {rep}"


@pytest.mark.parametrize("name", list(BROKEN))
def test_broken_sampler_fails_the_same_checks(name):
    kernel, fault, settings, must_fail = BROKEN[name]
    rep = run_harness(kernel, _faulty_mh(fault), **settings)
    checks = rep.checks()
    failed = {k for k, ok in checks.items() if not ok}
    assert must_fail <= failed, f"expected {must_fail} to fail, got {failed}: {rep}"


def test_checks_detect_a_small_bias_and_a_small_rescaling():
    # Sensitivity of the thresholds themselves, on correct pCN output: shifting every coordinate by
    # 0.15 (about a quarter of a posterior sd) must trip the mean check; inflating the spread by 8%
    # (17% in variance) must trip the variance check. The unmodified output passes both.
    res, rep = run_sampler("pcn")
    assert rep.checks()["mean"] and rep.checks()["var"]
    x = np.asarray(res.samples)
    shifted = Report(res._replace(samples=x + 0.15), 0.25)
    assert not shifted.checks()["mean"], str(shifted)
    scaled = res._replace(samples=MEAN + 1.08 * (x - MEAN), var=np.asarray(res.var) * 1.08**2)
    assert not Report(scaled, 0.25).checks()["var"]


# ---------------------------------------------------------------------------------------------
# aux threading
# ---------------------------------------------------------------------------------------------

TOY_C = jnp.array([1.0, -0.5, 0.25])


def toy_phi(xi, aux):
    """Phi = 2 |xi - c|^2; aux records the last evaluated point and the number of evaluations."""
    r = xi - TOY_C
    return 2.0 * jnp.sum(r * r), 4.0 * r, {"last": xi, "n": aux["n"] + 1}


def toy_aux0():
    return {"last": jnp.zeros(3), "n": jnp.zeros((), jnp.int32)}


# sampler, step0, kwargs, evaluations of phi_fn per proposal
TOY_STEPS = {
    "pcn": (sm.pcn, 0.6, {}, 1),
    "pcn_langevin": (sm.pcn_langevin, 0.3, {}, 1),
    "mala": (sm.mala, 0.4, {}, 1),
    "hmc": (sm.hmc, 0.7, {"n_leapfrog": 6}, 6),
}


def aux_consistent(res, per_prop):
    """aux belongs to the final state, and counts only evaluations along the accepted path."""
    acc = np.asarray(res.accept)
    same_point = np.array_equal(np.asarray(res.aux["last"]), np.asarray(res.xi))
    # (n_warmup = 0: 1 initial evaluation + per_prop evaluations for every accepted proposal)
    return same_point, int(res.aux["n"]) == 1 + per_prop * int(acc.sum())


@pytest.mark.parametrize("name", list(TOY_STEPS))
def test_aux_is_restored_on_rejection(name):
    fn, step0, kw, per_prop = TOY_STEPS[name]
    key = jax.random.key(3)
    res = fn(key, toy_phi, jnp.zeros(3), toy_aux0(), n_warmup=0, n_samples=400, step0=step0, **kw)
    acc = np.asarray(res.accept)
    assert 0.25 < acc.mean() < 0.95, "need both accepted and rejected proposals for this test"
    assert aux_consistent(res, per_prop) == (True, True)

    # same with warmup (adaptation active): the last evaluation of the chain was not necessarily
    # accepted, so a missing restoration would show up here as aux["last"] != xi
    res = fn(key, toy_phi, jnp.zeros(3), toy_aux0(), n_warmup=50, n_samples=7, step0=step0, **kw)
    np.testing.assert_array_equal(np.asarray(res.aux["last"]), np.asarray(res.xi))


def _no_aux_restore(orig):
    def step(key, state, prop, lr):
        new, alpha, accepted = orig(key, state, prop, lr)
        return new._replace(aux=prop.aux), alpha, accepted  # BUG: aux of the proposal even if rejected
    return step


@pytest.mark.parametrize("kernel,per_prop,step0", [
    (sm._pcn_kernel, 1, 0.6), (sm._mala_kernel, 1, 0.4),
    (partial(sm._hmc_kernel, n_leapfrog=6, jitter=0.0), 6, 0.7)])
def test_aux_threading_check_catches_missing_restore(kernel, per_prop, step0):
    with _patched_mh(_no_aux_restore(sm._mh_step)):
        res = jax.jit(lambda k: sm._run_chain(
            kernel, k, toy_phi, jnp.zeros(3), toy_aux0(), n_warmup=0, n_samples=400, step0=step0,
            thin=1, record_fn=sm.identity, target_accept=0.5, log_step_max=jnp.inf))(jax.random.key(3))
    acc = np.asarray(res.accept)
    assert 0.25 < acc.mean() < 0.95
    # the count is always wrong (every proposal's aux was kept); whether the final aux["last"] differs
    # from xi depends on whether the last proposal happened to be accepted
    assert aux_consistent(res, per_prop)[1] is False


# ---------------------------------------------------------------------------------------------
# bookkeeping: traces, thinning, Welford, step adaptation, run_chains, input handling
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(TOY_STEPS))
def test_phi_trace_matches_phi_of_the_stored_states(name):
    fn, step0, kw, _ = TOY_STEPS[name]
    res = fn(jax.random.key(4), toy_phi, jnp.zeros(3), toy_aux0(), n_warmup=20, n_samples=300,
             step0=step0, **kw)
    recomputed = jax.vmap(lambda x: toy_phi(x, toy_aux0())[0])(res.samples)
    np.testing.assert_allclose(np.asarray(res.phi), np.asarray(recomputed), rtol=1e-12)
    # rejected steps repeat the previous state exactly
    x, acc = np.asarray(res.samples), np.asarray(res.accept)
    same = np.all(x[1:] == x[:-1], axis=1)
    np.testing.assert_array_equal(same, ~acc[1:])


@pytest.mark.parametrize("name", list(TOY_STEPS))
def test_thinning_remainder_and_welford(name):
    fn, step0, kw, _ = TOY_STEPS[name]
    key = jax.random.key(5)
    args = (key, toy_phi, jnp.zeros(3), toy_aux0())
    full = fn(*args, n_warmup=10, n_samples=10, step0=step0, thin=1, **kw)
    thin = fn(*args, n_warmup=10, n_samples=10, step0=step0, thin=3, **kw)
    assert full.samples.shape == (10, 3) and thin.samples.shape == (3, 3)
    assert thin.phi.shape == thin.accept.shape == (10,)
    np.testing.assert_allclose(np.asarray(thin.samples), np.asarray(full.samples)[2::3][:3], atol=1e-12)
    np.testing.assert_allclose(np.asarray(thin.phi), np.asarray(full.phi), atol=1e-12)
    # Welford statistics cover all 10 post-warmup steps, whatever the thinning
    x = np.asarray(full.samples)
    for r in (full, thin):
        np.testing.assert_allclose(np.asarray(r.mean), x.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(np.asarray(r.var), x.var(axis=0, ddof=1), atol=1e-12)
    np.testing.assert_array_equal(np.asarray(thin.xi), np.asarray(full.xi))


def test_record_fn_scalar_shape_generality_and_gradient_free_pcn():
    # Prior-only target on a (3, 4) array, record the scalar |xi|^2 ~ chi^2_12: mean 12, variance 24.
    def prior_only(xi, aux):
        return jnp.zeros(()), jnp.zeros_like(xi), aux

    def prior_only_no_grad(xi, aux):  # pCN never reads the gradient
        return jnp.zeros(()), None, aux

    record = lambda xi: jnp.sum(xi**2)  # noqa: E731
    cases = [(sm.pcn, prior_only_no_grad, 0.5, {}), (sm.pcn, prior_only, 0.5, {}),
             (sm.pcn_langevin, prior_only, 0.5, {}), (sm.mala, prior_only, 0.5, {}),
             (sm.hmc, prior_only, 0.5, {"n_leapfrog": 5, "jitter": 0.2})]
    for fn, phi, step0, kw in cases:
        res = fn(jax.random.key(6), phi, jnp.ones((3, 4)), None, n_warmup=500, n_samples=40000,
                 step0=step0, record_fn=record, **kw)
        assert res.samples.shape == (40000,) and res.xi.shape == (3, 4)
        x = np.asarray(res.samples)
        assert abs(float(res.mean) - 12.0) < 5 * dg.mcse_mean(x), (fn.__name__, float(res.mean))
        assert float(res.var) == pytest.approx(24.0, rel=0.12), (fn.__name__, float(res.var))


def test_step_is_adapted_in_warmup_only_and_frozen_after():
    key, args = jax.random.key(7), (jax.random.key(7), phi_fn, XI0[0], None)
    for fn, step0, kw in [(sm.mala, 0.1, {}), (sm.hmc, 0.2, {"n_leapfrog": 10}),
                          (sm.pcn_langevin, 0.05, {}), (sm.pcn, 0.2, {})]:
        short = fn(*args, n_warmup=300, n_samples=50, step0=step0, **kw)
        long = fn(*args, n_warmup=300, n_samples=500, step0=step0, **kw)
        assert float(short.step) == float(long.step)  # more sampling steps do not move the step
        assert float(short.step) != step0  # but warmup did
        none = fn(*args, n_warmup=0, n_samples=50, step0=step0, **kw)
        assert float(none.step) == pytest.approx(step0, rel=1e-14)  # no warmup, no adaptation
    big = sm.pcn(*args, n_warmup=0, n_samples=10, step0=3.0)
    assert float(big.step) == 1.0  # beta is capped at 1
    big = sm.pcn_langevin(*args, n_warmup=0, n_samples=10, step0=9.0)
    assert float(big.step) == pytest.approx(2.0, rel=1e-14)  # delta is capped at 2


@pytest.mark.parametrize("name", ["pcn", "mala", "hmc"])
def test_adaptation_reaches_the_target_from_far_away_starts(name):
    fn, step0, n_warmup, _, target, kw = SAMPLERS[name]
    out = []
    for factor in (0.01, 30.0):
        res = sm.run_chains(fn, KEYS, XI0, None, phi_fn=phi_fn, n_warmup=2000, n_samples=2000,
                            step0=step0 * factor, **kw)
        out.append((np.asarray(res.step), float(np.asarray(res.accept).mean())))
        assert abs(out[-1][1] - target) < 0.07, (factor, out[-1])
    # both starts end at the same step size, up to the stochastic-approximation noise
    ratio = out[0][0].mean() / out[1][0].mean()
    assert 0.8 < ratio < 1.25, ratio


def test_run_chains_matches_individual_runs_with_pytree_aux():
    n = 3
    keys = jax.random.split(jax.random.key(8), n)
    x0 = jax.random.normal(jax.random.key(9), (n, 3))
    aux0 = {"last": jnp.zeros((n, 3)), "n": jnp.zeros((n,), jnp.int32)}
    kw = dict(phi_fn=toy_phi, n_warmup=30, n_samples=200, step0=0.4)
    batched = sm.run_chains(sm.mala, keys, x0, aux0, **kw)
    assert batched.samples.shape == (n, 200, 3) and batched.step.shape == (n,)
    assert batched.aux["n"].shape == (n,)
    for i in range(n):
        single = sm.mala(keys[i], xi0=x0[i], aux0=jax.tree.map(lambda a: a[i], aux0), **kw)
        np.testing.assert_allclose(np.asarray(batched.samples[i]), np.asarray(single.samples), atol=1e-9)
        np.testing.assert_allclose(float(batched.step[i]), float(single.step), rtol=1e-9)
        np.testing.assert_array_equal(np.asarray(batched.accept[i]), np.asarray(single.accept))
    # chains with different keys are different chains
    assert not np.allclose(np.asarray(batched.samples[0]), np.asarray(batched.samples[1]))


def test_input_validation():
    for bad in (dict(thin=0), dict(n_samples=0), dict(n_warmup=-1)):
        kw = dict(n_warmup=1, n_samples=5, step0=0.5) | bad
        with pytest.raises(ValueError):
            sm.pcn(jax.random.key(0), toy_phi, jnp.zeros(3), toy_aux0(), **kw)


# ---------------------------------------------------------------------------------------------
# Leapfrog integrator: order, reversibility, volume preservation
# ---------------------------------------------------------------------------------------------

_Q, _ = np.linalg.qr(np.random.default_rng(10).standard_normal((4, 4)))
A_GAUSS = jnp.asarray(_Q @ np.diag([0.5, 1.0, 2.0, 4.0]) @ _Q.T)  # SPD, eigenvalues 0.5 .. 4


def gauss_phi(q, aux):
    """Phi = q^T A q / 2 with a fixed SPD A, so U = q^T (A + I) q / 2 (a Gaussian target)."""
    return 0.5 * q @ A_GAUSS @ q, A_GAUSS @ q, aux


def hamiltonian(q, p):
    return 0.5 * q @ A_GAUSS @ q + 0.5 * q @ q + 0.5 * p @ p


def lf_integrator(eps, n):
    def run(q, p):
        _, g, _ = gauss_phi(q, None)
        q1, p1, *_ = sm.leapfrog(gauss_phi, q, p, g, None, eps, n)
        return q1, p1
    return run


def euler_integrator(eps, n):
    """Symplectic (semi-implicit) Euler: first order, the standard wrong-order control."""
    def run(q, p):
        for _ in range(n):
            p = p - eps * (A_GAUSS @ q + q)
            q = q + eps * p
        return q, p
    return run


def energy_error_rms(make, eps, t_total=1.0, n_starts=300):
    rng = np.random.default_rng(11)
    q0 = jnp.asarray(rng.standard_normal((n_starts, 4)))
    p0 = jnp.asarray(rng.standard_normal((n_starts, 4)))
    q1, p1 = jax.vmap(make(eps, int(round(t_total / eps))))(q0, p0)
    dh = jax.vmap(hamiltonian)(q1, p1) - jax.vmap(hamiltonian)(q0, p0)
    return float(jnp.sqrt(jnp.mean(dh**2)))


def loglog_slope(make, eps_list=(0.2, 0.1, 0.05)):
    e = [energy_error_rms(make, eps) for eps in eps_list]
    return np.diff(np.log(e)) / np.diff(np.log(eps_list))


def test_leapfrog_energy_error_scales_as_step_squared():
    slopes = loglog_slope(lf_integrator)
    assert np.all(np.abs(slopes - 2.0) < 0.15), slopes
    # control: the same measurement on a first-order integrator reads about 1, far outside the band
    slopes_euler = loglog_slope(euler_integrator)  # tends to 1 from above as the step shrinks
    assert np.all((slopes_euler > 0.9) & (slopes_euler < 1.3)), slopes_euler
    assert not np.all(np.abs(slopes_euler - 2.0) < 0.15)


def test_leapfrog_is_reversible_and_volume_preserving():
    rng = np.random.default_rng(12)
    q0, p0 = jnp.asarray(rng.standard_normal(4)), jnp.asarray(rng.standard_normal(4))
    step = lf_integrator(0.15, 7)
    q1, p1 = step(q0, p0)
    q2, p2 = step(q1, -p1)  # integrate back with flipped momentum
    np.testing.assert_allclose(np.asarray(q2), np.asarray(q0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(-p2), np.asarray(p0), atol=1e-12)
    jac = jax.jacfwd(lambda z: jnp.concatenate(step(z[:4], z[4:])))(jnp.concatenate([q0, p0]))
    assert abs(float(jnp.linalg.det(jac)) - 1.0) < 1e-10
    # a single step against the textbook formula
    q, p = lf_integrator(0.3, 1)(q0, p0)
    p_half = p0 - 0.15 * (A_GAUSS @ q0 + q0)
    q_exp = q0 + 0.3 * p_half
    np.testing.assert_allclose(np.asarray(q), np.asarray(q_exp), atol=1e-13)
    np.testing.assert_allclose(np.asarray(p), np.asarray(p_half - 0.15 * (A_GAUSS @ q_exp + q_exp)), atol=1e-13)
    # momentum update counts exactly n_steps evaluations of phi_fn
    calls = sm.leapfrog(toy_phi, jnp.zeros(3), jnp.ones(3), 4.0 * (jnp.zeros(3) - TOY_C), toy_aux0(), 0.1, 9)[4]
    assert int(calls["n"]) == 9
