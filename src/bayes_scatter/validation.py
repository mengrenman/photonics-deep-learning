"""Born end-to-end validation harness (docs/DESIGN.md section 8, step 5), shared by the test and the script.

The physics forward model with the Born linearization and the identity link makes the posterior Gaussian,
so ``inference.born_gaussian_posterior`` gives the exact answer that the samplers, driven through the same
``Target`` as in a nonlinear run, must reproduce.

``make_born_problem`` builds the problem; ``run_born_check`` runs the samplers and compares each with the
exact posterior (``inference.compare_to_gaussian`` on the chi map, plus the posterior mean of Phi).

Why the default SNR is so low. Born data of N = Nt * Nr = 2400 complex samples carry about N times the
information of one sample about a mode that all transmitters and receivers see. In whitened coordinates the
prior has precision 1 in every direction. At +20 dB per datum the sharpest posterior direction has precision
about 3e6 (posterior standard deviation about 1700 times smaller than the prior's, condition number of the
precision matrix 3e6; 69 directions have precision above 1.5), which no sampler with an identity mass matrix
can mix. At -20 dB per datum the posterior is still far from the prior but is sampled well by all four
samplers in a few seconds: the sharpest direction has precision about 300 (standard deviation about 17 times
smaller than the prior's) and 35 of the 576 directions have precision above 1.5. (Measured on the default
problem; tests/test_born_end_to_end.py pins these numbers.)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import diagnostics as dg
from .forward import Forward
from .geometry import ch15_setup
from .inference import BornPosterior, born_gaussian_posterior, compare_to_gaussian, sample
from .likelihood import Target, add_noise, noise_std_from_snr
from .priors import GaussianField


@dataclass
class BornProblem:
    """A Born inverse problem with its exact Gaussian posterior (identity link)."""

    setup: object
    forward: Forward
    field: GaussianField
    target: Target
    chi_true: np.ndarray  # (n, n)
    d_clean: jnp.ndarray  # (Nt, Nr)
    d_obs: jnp.ndarray
    s: float
    snr_db: float
    exact: BornPosterior
    phi_mean_exact: float  # E_post[Phi]

    @property
    def n(self):
        return self.forward.n

    @property
    def var_chi(self):
        """Exact marginal posterior variance of every pixel, shape (n, n)."""
        return np.diag(self.exact.cov_chi).reshape(self.n, self.n)

    @property
    def mean_chi(self):
        return self.exact.mean_chi.reshape(self.n, self.n)


def bump_phantom(grid, amplitude=0.3, width=0.2, center=(0.12, -0.08)):
    """Smooth Gaussian contrast bump chi = amplitude * exp(-|r - center|^2 / (2 width^2)) on the grid, in
    units of grid.side / 1.5 (so a side-1.5 DOI has width 0.2 and the given center in wavelengths)."""
    X, Y = grid.centers()
    sc = grid.side / 1.5
    return amplitude * np.exp(-((X - sc * center[0]) ** 2 + (Y - sc * center[1]) ** 2) / (2.0 * (sc * width) ** 2))


def make_born_problem(n=12, side=1.5, snr_db=-20.0, sigma=0.3, ell=0.3, nu=1.5, kind="matern", mean=0.0,
                      n_tx=40, n_rx=60, seed=0, tol=1e-10):
    """Born problem on the book's antenna rings: truth = smooth bump, Born data + noise, identity link.

    The prior is a Matern (or GMRF) field on chi directly (identity link) with standard deviation
    ``sigma`` and correlation length ``ell`` wavelengths. The data are the Born data of the bump plus
    circular complex Gaussian noise at ``snr_db`` (power SNR per datum, see the module docstring for why
    it is negative). The Born posterior (mean, covariance of xi and chi) and the posterior mean of Phi are
    computed exactly.
    """
    setup = ch15_setup(n=n, side=side, n_tx=n_tx, n_rx=n_rx)
    forward = Forward(setup, tol=tol)
    field = GaussianField(setup.grid, kind, sigma=sigma, ell=ell, nu=nu, mean=mean)
    chi_true = bump_phantom(setup.grid)
    d_clean = forward.born(jnp.asarray(chi_true))
    s = noise_std_from_snr(d_clean, snr_db)
    d_obs = add_noise(jax.random.key(seed), d_clean, s)
    target = Target(forward, field, "identity", d_obs, s, use_born=True)
    exact = born_gaussian_posterior(forward, field, d_obs, s)
    # E[Phi] under N(m, S): Phi(m) + tr(Br S Br^T)/s^2 = Phi(m) + (D - tr S)/2, with D = dim(xi), because
    # Br^T Br / s^2 = (Lambda - I)/2 and S = Lambda^-1.
    phi_m = float(target.potential(jnp.asarray(exact.mean_xi).reshape(field.xi_shape),
                                   target.initial_aux())[0])
    phi_mean = phi_m + 0.5 * (exact.mean_xi.size - np.trace(exact.cov_xi))
    return BornProblem(setup, forward, field, target, chi_true, d_clean, d_obs, s, float(snr_db), exact,
                       float(phi_mean))


# (n_warmup, n_samples, thin, step0, extra kwargs) per sampler
DEFAULT_SETTINGS = {
    "pcn": dict(n_warmup=4000, n_samples=60000, thin=20, step0=0.05),
    "pcn_langevin": dict(n_warmup=2000, n_samples=20000, thin=10, step0=0.05),
    "mala": dict(n_warmup=2000, n_samples=20000, thin=10, step0=0.01),
    "hmc": dict(n_warmup=500, n_samples=3000, thin=2, step0=0.1, n_leapfrog=10, jitter=0.2),
}


def run_born_check(problem, settings=None, n_chains=4, seed=1):
    """Run the samplers named in ``settings`` on the problem and compare them with the exact posterior.

    Chains start from independent draws of the prior, xi ~ N(0, I). Returns a dict name -> dict with
    ``result`` (samplers.ChainResult), ``info`` (inference.sample timing and counts), ``stats``
    (inference.compare_to_gaussian on the chi map), ``phi_z`` (standardized error of the mean of Phi
    against its exact posterior mean), ``accept`` (mean acceptance) and ``ess_per_s`` (minimum pixel ESS
    divided by the run time).
    """
    settings = DEFAULT_SETTINGS if settings is None else settings
    target = problem.target
    key = jax.random.key(seed)
    k_init, key = jax.random.split(key)
    xi0s = problem.field.sample_xi(k_init, n_chains)
    out = {}
    for name, kw in settings.items():
        key, sub = jax.random.split(key)
        res, info = sample(target, name, sub, xi0s, record_fn=target.chi, chain_method="vmap", **kw)
        samples = np.asarray(res.samples)  # (chains, N, n, n)
        stats = compare_to_gaussian(samples, problem.mean_chi, problem.var_chi)
        phi = np.asarray(res.phi)  # (chains, n_samples)
        phi_hat = phi.mean()
        mcse = np.sqrt(np.sum([dg.mcse_mean(p) ** 2 for p in phi])) / len(phi)
        out[name] = {
            "result": res, "info": info, "stats": stats,
            "phi_z": float((phi_hat - problem.phi_mean_exact) / mcse), "phi_mean": float(phi_hat),
            "accept": float(np.asarray(res.accept).mean()),
            "ess_per_s": float(stats["ess_min"] / info["wall_s"]),
        }
    return out
