# photonics-dl: a Bayesian inverse scattering testbed

The book chapter this project follows (Campbell and Werner, eds., *Advances in Electromagnetics Empowered
by Artificial Intelligence and Deep Learning*, Ch. 15, section 15.4.1.1) reconstructs the permittivity of a
2-D scatterer from multi-static scattering data and reports point estimates only. This package puts a
posterior on the same problem and samples it with MCMC, to see which features of a reconstruction the data
determine and which come from the prior.

Everything is written from scratch in JAX (float64): a Lippmann-Schwinger forward model with FFT-accelerated
pixel Green's functions and an adjoint gradient, Gaussian-field priors on log eps_r in whitened coordinates,
four MCMC samplers (pCN, pCN-Langevin, MALA, HMC), convergence diagnostics, a MAP baseline, and an exact
Gaussian reference posterior (Born model) against which the samplers are validated. The contract for every
module is [docs/DESIGN.md](docs/DESIGN.md).

## Setup

```bash
conda create -n photonics-dl -c conda-forge python=3.12 numpy scipy matplotlib jupyterlab pytest
conda activate photonics-dl
pip install -U jax                   # CPU; for the Linux GPU workstation see docs/WORKSTATION.md
pip install -e ".[dev]"              # from the repository root
```

## Tests

```bash
pytest -q -m "not slow"              # default suite, about 2.5 minutes
pytest -q -m slow                    # one longer Monte Carlo run (tighter agreement with the exact posterior), about 2 minutes
pytest tests/test_born_end_to_end.py -m "not slow" -v   # the key validation (about 45 s)
```

Every test file is named after its module (`test_geometry_phantoms.py`, `test_greens.py`, `test_forward.py`,
`test_gradients.py`, `test_priors.py`, `test_samplers.py`, `test_diagnostics.py`, `test_likelihood.py`,
`test_inference.py`, `test_plotting.py`, `test_scripts.py`) plus `test_born_end_to_end.py`.

## Scripts

| script | what it does |
|---|---|
| `scripts/validate_forward.py` | Mie convergence table, timing and GMRES effort of the forward model, stacked versus vmapped solves, warm starts |
| `scripts/born_exact_check.py` | Born end-to-end check at a larger size: four samplers against the exact Gaussian posterior, z-score table, ESS per second, figure in `results/born_exact/` |
| `scripts/run_posterior.py` | The full pipeline: phantom, data on a finer grid, noise, prior, MAP, MCMC, summaries and figures in `results/<out>/`; also reports the operating point of the inference grid (model error against the noise, Phi at the truth against N) |
| `scripts/run_posterior_laplace.py` | Prototype outside the ladder: the same pipeline with the samplers run in Laplace-whitened coordinates (Gauss-Newton eigenpairs at the MAP, log-normal re-centering). Its sampler trace is the reparametrized Phi', not the misfit; the misfit of the stored draws is reported separately |
| `scripts/map_multistart.py` | MAP from several prior-draw starts: one basin or several |
| `scripts/analyze_run.py`, `scripts/make_tables.py` | Per-run metrics (misfit, ESS, R-hat, z-maps, Laplace reference std) and the Markdown tables / misfit figures in `results/tables/` and `results/figures/` |
| `scripts/measure_cost.py`, `scripts/plot_gn_spectrum.py` | Cost per evaluation on the benchmark configurations; Gauss-Newton spectrum figure |

```bash
python scripts/born_exact_check.py                       # about 2.5 min
python scripts/run_posterior.py --n 16 --side 2 --out demo
python scripts/run_posterior.py --n 24 --side 2 --phantom austria_fixed --eps-r 1.5 \
    --sampler hmc --chains 2 --warmup 300 --samples 500 --init map --out austria24
python scripts/run_posterior.py --help                   # all options
```

`run_posterior.py` writes `summary.json` (arguments, acceptance, final step size, ESS and integrated
autocorrelation time of Phi and of five fixed pixels, split R-hat, wall time, forward-solve counts, ESS per
second, the book's error metrics for the MAP and the posterior mean), `chains.npz`, `maps.png` (truth, MAP,
posterior mean, posterior standard deviation, and the z-map (mean - truth)/std, all in eps_r),
`maps_log_eps.png` (the same in log eps_r) and `traces.png`.

## Package layout

```
src/bayes_scatter/
  geometry.py     Grid, Antennas, Setup, ring(), ch15_setup()
  phantoms.py     coverage(), cylinder(), austria_fixed(), ch15_austria_variant()
  greens.py       pixel kernel (Richmond equal-area cells), FFT embedding, receiver matrix, incident fields
  krylov.py       restarted GMRES with per-transmitter tolerances (see forward.py for why)
  forward.py      Forward: apply_G, solve_fields, scattered, __call__ (differentiable), born, born_matrix
  mie.py          analytic cylinder scattering for validation
  priors.py       GaussianField (Matern, GMRF), links (log_eps, identity), compose
  likelihood.py   noise_std_from_snr, add_noise, misfit, Target
  samplers.py     pcn, pcn_langevin, mala, hmc, run_chains, ChainResult
  diagnostics.py  autocorr, tau_int, ess, ess_geyer, mcse_mean, split_rhat
  inference.py    map_estimate, born_gaussian_posterior, err_model/err_chi/err_data, operating_point, sample, compare_to_gaussian,
                  physical_phi_of_draws, laplace_theta_variance, lognormal_center_shift
  validation.py   Born problem builder and sampler comparison shared by the test and the script
  plotting.py     plot_maps, plot_traces, plot_born_check
```

## Summary of the design

**Conventions.** Lengths in free-space wavelengths (k0 = 2 pi), time dependence e^{-i omega t}, 2-D TM
polarization with u = E_z. Pixel arrays are indexed `[i, j] = [x, y]` and flattened in C order. The contrast
is chi = eps_r - 1.

**Forward model.** The total field of each transmitter solves (I - G diag(chi)) u = u_inc, with G the
block-Toeplitz pixel Green's operator applied by FFT on a 2n x 2n embedding. Each square pixel is replaced
by an equal-area disk, which gives closed forms for the self and off-cell kernels. Data are
d[t, r] = sum_m R[r, m] chi_m u_t[m] for the book's rings: 40 transmitters on a circle of radius 4, 60
receivers on a circle of radius 4.5, a 5 x 5 wavelength domain at n = 96. The solve is differentiable in
chi by the adjoint (implicit-function) gradient, checked against finite differences. The Born
approximation (u replaced by u_inc) makes the data linear in chi.

**Prior.** The parameter is theta = log eps_r, so eps_r = e^theta is log-normal and positive by
construction. theta = mu + C^{1/2} xi with xi ~ N(0, I) (whitened coordinates), C a stationary Matern or
GMRF covariance applied by FFT on a padded torus. mu = 0 is the vacuum background, sigma is in natural-log
units. The identity link chi = theta exists only for linear-Gaussian validation.

**Likelihood.** Circular complex Gaussian noise at a power SNR in dB. The potential is
Phi(xi) = sum |F(chi(xi)) - d_obs|^2 / s^2 and the posterior is pi(xi) proportional to
exp(-Phi(xi)) N(xi; 0, I).

**Samplers.** All four share one callable `phi_fn(xi, aux) -> (phi, grad, aux_new)`: pCN, pCN-Langevin
(infinity-MALA), MALA and HMC, each in `lax.scan`, with Robbins-Monro step-size adaptation during warmup,
Welford statistics, and a warm-start field `aux` that is restored on rejection. `inference.sample` drives
them on a `Target`.

**Inference utilities.** `map_estimate` (Tikhonov MAP by L-BFGS-B), `born_gaussian_posterior` (exact
posterior for the Born model with the identity link), and the book's metrics `err_model` (eq. 15.21, on
eps_r) and `err_data` (eq. 15.22), plus the contrast-normalized `err_chi`, because eq. 15.21 includes the
background eps_r = 1 and compresses errors in fine structure.

## Validation ladder (status)

Numbers are in `results/tables/` and `results/figures/` (commands in `scripts/reproduce_report.sh`); the status says which rungs are built and tested.

| step | check | status | where |
|---|---|---|---|
| 1 | kernel against quadrature (self and off-cell terms) | done | `tests/test_greens.py` |
| 2 | forward against the Mie series, error decreasing under refinement | done | `tests/test_forward.py`, `scripts/validate_forward.py` |
| 3 | gradient against central finite differences | done | `tests/test_gradients.py`; Target level in `tests/test_likelihood.py` |
| 4 | samplers on a known Gaussian, with negative controls | done | `tests/test_samplers.py` |
| 5 | samplers on the exact Born posterior through the physics `Target` | done | `tests/test_born_end_to_end.py`, `scripts/born_exact_check.py` |
| 6 | nonlinear posterior (log_eps link), inverse crime avoided | run at n = 32 (weak and moderate contrast) with the Laplace-whitened samplers; identity-mass samplers did not converge; book scale (n = 48, and n = 96 at 20 dB) is out of reach: the MAP does not converge, see `results/tables/c_book.md` | `scripts/run_posterior.py`, `scripts/run_posterior_laplace.py` |
| 7 | prior sensitivity (Matern versus GMRF, further priors later) | one GMRF control at moderate contrast; edge-preserving priors not done, so no resolution claim yet | `scripts/run_posterior_laplace.py --prior gmrf` |

Step 5 uses a deliberately low SNR (-20 dB per datum) because 2400 complex data multiply the information
about the best-determined directions; see the docstring of `bayes_scatter/validation.py`.

## Known limits

- High contrast is expensive: the forward solver is unpreconditioned GMRES. At n = 96 an eps_r = 2 phantom
  costs about 0.6 s per forward solve, eps_r = 4 about 5 s, and eps_r = 8 does not converge at restart 100.
  Solves that hit the iteration cap are returned silently; `Target.solver_residual` reports the true residual.
- Chains of the full forward model run one after another (`chain_method="sequential"`); vmapping the Krylov
  loops over chains is much slower to compile and run. Batching is used for the Born model.
- Posterior directions that the data determine are far narrower than the prior (condition numbers of
  10^3 to 10^6 in whitened coordinates at realistic SNR), so samplers with an identity mass matrix mix slowly
  in the prior-dominated directions. This is the quantity the ESS per second columns measure.
- The inference grid is itself a source of error. At 30 dB the data noise is 3.2 percent of ||d||; the forward
  model on the coarse inference grid misses the finer-grid data by 0.7 percent (weak cylinder, n = 32), 2.1 percent
  (moderate austria phantom, n = 32; 0.85 percent at n = 48) and 8.0 percent (book scale, side 5, n = 48; 2.6
  percent at n = 96). `run_posterior.py` prints and stores this as `operating_point` (Phi at the truth against N).
- Samplers that run in transformed coordinates (the Laplace-whitened prototype) have a potential that is not the data
  misfit; quote `summary.json["phi_physical"]`, not the sampler trace (`summary.json["phi"]["kind"]` says which).
- Later milestones (Laplace pass, edge-preserving and learned priors, delayed acceptance, replica exchange,
  symmetry reduction, experimental data) are listed in DESIGN.md section 10.
