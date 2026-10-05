"""THE key validation (docs/DESIGN.md 8, step 5): samplers on the exact Born posterior.

Problem (``bayes_scatter.validation.make_born_problem``): the physics forward model with the Born
linearization on a 12 x 12 grid of side 1.5 wavelengths, the book's antenna rings (40 transmitters, 60
receivers, 2400 complex data), a Matern prior on chi with the identity link, and noise at -20 dB per datum
(see the validation module for why that is the right regime). Then xi -> chi is affine and the posterior is
exactly Gaussian, so ``born_gaussian_posterior`` is the reference. The same ``Target`` and the same samplers
as in a nonlinear run are driven on it with chains started from the prior (draws of N(0, I)).

For each sampler, with 4 chains (``inference.compare_to_gaussian`` on the 144 pixels of chi):

* z_mean = (mean_hat - mean) / MCSE, expected N(0, 1) per pixel, with MCSE from Sokal effective sample sizes;
* z_var, the same for the variance (through q = (x - mean)^2, whose mean is exactly the variance), and the
  variance ratio;
* the posterior mean of Phi against its exact value, E[Phi] = Phi(m) + (dim xi - tr cov_xi) / 2;
* split R-hat, acceptance rate.

Pixel z-values are strongly correlated (about 30 data-dominated directions), so only RMS and max are
bounded, with margins read off 7 runs of every sampler (z RMS 0.9 to 1.2, max |z| up to 4.5; variance
ratios 0.83 to 1.23 per pixel, 0.99 to 1.02 on average).

NEGATIVE CONTROLS: the same chains are compared with the exact posterior of a wrong model, noise level
s * sqrt(2) and s / sqrt(2) (the effect of a factor 2 in Phi, the typical normalization slip), and with the
exact posterior of complex-conjugated data (an e^{+i omega t} convention slip). Each must fail.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import diagnostics as dg
from bayes_scatter.inference import born_gaussian_posterior, compare_to_gaussian
from bayes_scatter.validation import DEFAULT_SETTINGS, make_born_problem, run_born_check

TARGET_ACCEPT = {"pcn": 0.25, "pcn_langevin": 0.5, "mala": 0.574, "hmc": 0.7}

# thresholds (see the module docstring)
Z_RMS_MAX, Z_MAX_MAX = 1.7, 5.0
VAR_MEAN_BAND, VAR_PIXEL_BAND = (0.94, 1.06), (0.70, 1.40)
PHI_Z_MAX, RHAT_MAX, ACC_TOL = 4.0, 1.10, 0.10


def checks(st, accept=None, phi_z=None, name=None):
    """Named boolean checks of a ``compare_to_gaussian`` result (plus acceptance and Phi if given)."""
    out = {
        "mean": st["z_mean_rms"] < Z_RMS_MAX and st["z_mean_max"] < Z_MAX_MAX,
        "variance": st["z_var_rms"] < Z_RMS_MAX and st["z_var_max"] < Z_MAX_MAX
        and VAR_MEAN_BAND[0] < st["var_ratio_mean"] < VAR_MEAN_BAND[1]
        and VAR_PIXEL_BAND[0] < st["var_ratio_min"] and st["var_ratio_max"] < VAR_PIXEL_BAND[1],
        "rhat": st["rhat_max"] < RHAT_MAX,
    }
    if accept is not None:
        out["accept"] = abs(accept - TARGET_ACCEPT[name]) < ACC_TOL
    if phi_z is not None:
        out["phi"] = abs(phi_z) < PHI_Z_MAX
    return out


def describe(name, st, **extra):
    return (f"{name}: z_mean rms {st['z_mean_rms']:.2f} max {st['z_mean_max']:.2f}; z_var rms "
            f"{st['z_var_rms']:.2f} max {st['z_var_max']:.2f}; var ratio mean {st['var_ratio_mean']:.3f} "
            f"[{st['var_ratio_min']:.3f}, {st['var_ratio_max']:.3f}]; min ESS {st['ess_min']:.0f}; "
            f"R-hat {st['rhat_max']:.3f}; " + ", ".join(f"{k} {v:.3f}" for k, v in extra.items()))


@pytest.fixture(scope="module")
def problem():
    return make_born_problem()


@pytest.fixture(scope="module")
def runs(problem):
    """All four samplers run once per session on the Born problem (about 40 s)."""
    return run_born_check(problem, DEFAULT_SETTINGS, n_chains=4, seed=1)


# ---------------------------------------------------------------------------------------------
# The problem itself is informative but not degenerate
# ---------------------------------------------------------------------------------------------


def test_problem_is_informative_and_the_reference_is_exact(problem):
    p = problem
    ev = np.linalg.eigvalsh(p.exact.precision_xi)  # whitened posterior precision; prior = identity
    assert 0.99 < ev.min() < 1.01  # most directions are prior-dominated
    # the sharpest direction has posterior PRECISION ~300 (prior: 1), i.e. a standard deviation ~17x smaller
    assert 250.0 < ev.max() < 350.0
    assert 15.0 < np.sqrt(ev.max()) < 20.0  # prior sd / sd of the sharpest direction = sqrt(precision)
    assert 33 <= int(np.sum(ev > 1.5)) <= 37  # 35 data-dominated directions (precision above 1.5)
    std = np.sqrt(np.diag(p.exact.cov_chi))
    assert std.max() < 0.6 * p.field.sigma and std.min() > 0.2 * p.field.sigma  # data help, prior still matters
    # the exact posterior mean is far from the prior mean, so the mean test has power
    assert np.linalg.norm(p.exact.mean_chi) > 3.0 * np.mean(std)
    assert p.phi_mean_exact > 0.5 * 2400  # E[Phi] is of the order of the number of complex data


def test_target_matches_the_born_model_and_differs_from_the_full_one(problem):
    """use_born=True evaluates d = A chi; the full nonlinear model differs from it at this contrast scale."""
    from bayes_scatter.likelihood import Target

    p = problem
    xi = jnp.asarray(p.exact.mean_xi).reshape(p.field.xi_shape)
    full = Target(p.forward, p.field, "identity", p.d_obs, p.s, use_born=False)
    phi_born = float(p.target.potential(xi, p.target.initial_aux())[0])
    phi_full = float(full.potential(xi, full.initial_aux())[0])
    assert phi_born != phi_full
    # at 1% of the contrast the two models agree to the second order in chi
    phi_b1 = float(p.target.potential(0.01 * xi, p.target.initial_aux())[0])
    phi_f1 = float(full.potential(0.01 * xi, full.initial_aux())[0])
    assert abs(phi_b1 - phi_f1) < 1e-3 * phi_b1


# ---------------------------------------------------------------------------------------------
# Correct samplers reproduce the exact posterior
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(DEFAULT_SETTINGS))
def test_posterior_mean_matches_exact(name, runs):
    r = runs[name]
    ok = checks(r["stats"], r["accept"], r["phi_z"], name)
    assert ok["mean"], describe(name, r["stats"], phi_z=r["phi_z"])


@pytest.mark.parametrize("name", list(DEFAULT_SETTINGS))
def test_posterior_variance_matches_exact(name, runs):
    r = runs[name]
    ok = checks(r["stats"], r["accept"], r["phi_z"], name)
    assert ok["variance"], describe(name, r["stats"])


@pytest.mark.parametrize("name", list(DEFAULT_SETTINGS))
def test_phi_mean_acceptance_and_convergence(name, runs):
    r = runs[name]
    ok = checks(r["stats"], r["accept"], r["phi_z"], name)
    assert ok["phi"], f"{name}: mean Phi {r['phi_mean']:.2f} z {r['phi_z']:.2f}"
    assert ok["accept"], f"{name}: acceptance {r['accept']:.3f}"
    assert ok["rhat"], describe(name, r["stats"])
    assert r["stats"]["ess_min"] > 100  # enough independent draws for the z-statistics to mean something


def test_welford_statistics_agree_with_stored_samples(problem, runs):
    """The running mean of the chi map over all post-warmup steps agrees with the thinned samples."""
    for name, r in runs.items():
        samples = np.asarray(r["result"].samples)  # (chains, N/thin, n, n)
        welford = np.asarray(r["result"].mean)
        assert welford.shape == (4, problem.n, problem.n)
        sd = np.sqrt(np.asarray(r["result"].var)).mean()
        # same chain, different thinning: the two means differ by a fraction of the posterior sd
        assert np.max(np.abs(welford - samples.mean(axis=1))) < 0.2 * sd, name


def test_chains_started_from_the_prior_found_the_posterior(problem, runs):
    """Phi starts near its prior-draw value (thousands above the posterior level) and ends at the exact
    posterior mean of Phi: the warmup did its work."""
    for name, r in runs.items():
        phi = np.asarray(r["result"].phi)
        assert abs(phi.mean() / problem.phi_mean_exact - 1.0) < 0.01, name


# ---------------------------------------------------------------------------------------------
# Negative controls: the same chains must be rejected against a wrong exact posterior
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [np.sqrt(2.0), 1.0 / np.sqrt(2.0)], ids=["s_x_sqrt2", "s_div_sqrt2"])
@pytest.mark.parametrize("name", list(DEFAULT_SETTINGS))
def test_wrong_noise_level_is_rejected(name, factor, problem, runs):
    p = problem
    wrong = born_gaussian_posterior(p.forward, p.field, p.d_obs, factor * p.s)
    st = compare_to_gaussian(np.asarray(runs[name]["result"].samples), wrong.mean_chi, np.diag(wrong.cov_chi))
    ok = checks(st)
    assert not ok["variance"], describe(name, st)
    assert st["z_var_rms"] > 3.0, describe(name, st)  # not a marginal failure (correct runs: 1.0 to 1.2)
    # and the mean is detected as well for the stiffest sampler, whose MCSE is smallest
    if name == "hmc":
        assert not ok["mean"], describe(name, st)


@pytest.mark.parametrize("name", list(DEFAULT_SETTINGS))
def test_conjugated_data_convention_is_rejected(name, problem, runs):
    """The exact posterior for conj(d_obs) is the one an e^{+i omega t} convention error would give."""
    p = problem
    wrong = born_gaussian_posterior(p.forward, p.field, np.conj(np.asarray(p.d_obs)), p.s)
    st = compare_to_gaussian(np.asarray(runs[name]["result"].samples), wrong.mean_chi, np.diag(wrong.cov_chi))
    assert not checks(st)["mean"], describe(name, st)


def test_mcse_is_not_inflated(runs):
    """Sanity check of the statistic itself: the MCSE of a pixel mean equals the spread of the chain means
    to within a factor of 2 (4 chains, so a loose bound), i.e. z is not small because MCSE is huge."""
    r = runs["hmc"]
    samples = np.asarray(r["result"].samples)[..., 5, 5]  # one pixel, (chains, N)
    mcse = np.sqrt(np.sum([dg.mcse_mean(c) ** 2 for c in samples])) / samples.shape[0]
    spread = np.std(samples.mean(axis=1), ddof=1) / np.sqrt(samples.shape[0])
    assert 0.2 * spread < mcse < 5.0 * spread


# ---------------------------------------------------------------------------------------------
# Longer chains, tighter agreement
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_longer_chains_agree_more_tightly():
    """Three times longer chains, another random seed and another noise realization: smaller Monte Carlo
    error, so the variance ratios tighten around 1."""
    problem = make_born_problem(seed=5)
    settings = {k: dict(v, n_samples=3 * v["n_samples"]) for k, v in DEFAULT_SETTINGS.items()}
    runs = run_born_check(problem, settings, n_chains=4, seed=7)
    for name, r in runs.items():
        st = r["stats"]
        assert st["z_mean_rms"] < 1.5 and st["z_var_rms"] < 1.5, describe(name, st)
        assert st["z_mean_max"] < 4.5 and st["z_var_max"] < 5.0, describe(name, st)
        assert 0.96 < st["var_ratio_mean"] < 1.04, describe(name, st)
        assert 0.8 < st["var_ratio_min"] and st["var_ratio_max"] < 1.25, describe(name, st)
        assert abs(r["phi_z"]) < 3.5 and st["rhat_max"] < 1.05, (name, r["phi_z"])
