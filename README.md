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
`test_inference.py`, `test_plotting.py`, `test_scripts.py`) plus `test_born_end_to_end.py`, and, for milestone 4,
`test_mrf.py`, `test_jacobian.py`, `test_thetaspace.py`, `test_features.py`, `test_prior_comparison_scripts.py`,
`test_measure_theta_cost.py` and `test_span_test.py`; for milestone 5, `test_adapt.py` (and the TV prior in `test_mrf.py`, the adapted
runner, the lambda sweep, the re-adapted map, burn-in segments, the slow-mode and cross-run diagnostics in `test_prior_comparison_scripts.py`),
`test_wait_then.py` and `test_segment_loop.py` (job gating and the per-chain segment loop); for its review follow-up, `test_m5_review.py` (low-pass reference, disk-averaged features, corrected independence checks, overdispersed starts, review batch).

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
| `scripts/run_prior_comparison.py` | Milestone 4 (DESIGN 11): ONE prior (`matern`, `gmrf`, `qggmrf`, and `tv`, the pseudo-Huber TV prior of milestone 5) on the shared moderate problem in theta = log eps_r: smooth start, L-BFGS-B MAP, dense Laplace mass from the exact reciprocity Jacobian, HMC in Laplace-whitened coordinates, convergence verdict, feature probabilities; writes `results/prior_comparison/<out>/` (`summary.json` with the d_obs hash and thread environment, `chains.npz`, `maps.png`); `--stage prepare / chain / finalize` splits it into one process per chain with continuation segments; `--adapt` (milestone 5, DESIGN 12.2) adds `--stage adapt` and samples in the coordinates of an affine map adapted in expanding warmup windows and frozen before recording (`adapted_map.npz`, `adaptation.json`). Two runs have the same data if max abs(d_a - d_b) / s <= 1e-6; the hash alone is not the criterion (it depends on the XLA thread count) |
| `scripts/make_prior_jobs.py` | Writes `batch_run.py` job files for the staged form of `run_prior_comparison.py` (`--stage prepare`, `adapt`, `chain` with `--segment` for extension, `finalize`): one single-thread process per chain, about 1.8x the throughput per thread of four threads per process (cost.json). Milestone 5 prior names: `tv_lam<value>` and a `_adapt` suffix (`gmrf_adapt`, `tv_lam3.75_adapt`) |
| `scripts/tv_lambda_sweep.py` | Milestone 5 (DESIGN 12.1): the milestone-4 MAP path for the pseudo-Huber TV prior at several lambda; Phi_MAP, R_MAP, the errors, convergence of the MAP and the MAP edge bias per lambda; lambda* by the discrepancy principle (`tv_lambda_sweep.json`, `.png`, `_maps.npz`; `--update` adds lambdas) |
| `scripts/segment_loop.py` | Milestone 5: ONE chain's recorded segments of `run_prior_comparison.py --stage chain`, one process per segment (a segment is the checkpoint: a killed run loses at most the segment in flight; existing segment files are skipped; `--go` / `--stop` files start and end the loop between segments); one `batch_run.py` job per chain |
| `scripts/make_m5_resume.py`, `scripts/make_m5_batch.py`, `scripts/wait_then.py` | Milestone 5 job files: the resume batch of 12 segment loops (adapted GMRF, TV at lambda* and 3 lambda*), the first batch (one process per chain and segment, gated by `wait_then.py` dependency / go / skip files), and the gate itself |
| `scripts/check_chain_starts.py`, `scripts/milestone5_tables.py` | Physical Phi, R and U at the chain starts of an adapted run before the chains are launched; the Markdown tables specific to milestone 5 (lambda sweep, adaptation, sampler, edge statistics with per-chain bias) from the stored JSON files |
| `scripts/make_m5_review_jobs.py`, `scripts/make_overdispersed_starts.py` | Milestone 5 review follow-up (DESIGN 12.6): ONE batch with two more TV posteriors (lambda = 1.25 and 8.5, chains from four different starts: truth, MAP, Matern mean, a rough GMRF draw) and an independent-start replication of lambda* in a new directory; the start maps and their TV prior energy (`starts.npy`, `starts.json`) |
| `scripts/resolution_analysis.py`, `scripts/feature_scales.py`, `scripts/lambda_band.py` | Review follow-up analyses: bias and z of the posterior mean against the low-passed or cell-averaged truth (`bayes_scatter.resolution`); the three features at point and disk-averaged scales, per chain and per block, and for the Laplace approximation without MCMC; the edge statistics of the TV posterior as a function of lambda across the discrepancy band |
| `scripts/equal_length_audit.py`, `scripts/start_drift_audit.py` | Convergence numbers at equal numbers of draws (and per gradient evaluation); block means of \|zeta\|^2 / 2 and the prior energy, and the recorded starts of every chain |
| `scripts/m5_commands_as_run.md`, `scripts/m5_review_analysis.sh` | The commands that were actually run for milestone 5 (including the manual launches), and the analysis commands of the review follow-up |
| `scripts/edge_analysis.py` | Bias, posterior std and z by signed distance to the object edge, an independent recomputation of the |z| statistics from the stored draws, the |z| > 3 fraction of each chain on its own, the bias of log E[eps_r] next to that of E[theta], and the per-draw blur at d = -1 (`edge_analysis.json`, `edge_profile.png`) |
| `scripts/convergence_audit.py` | Rank-normalized bulk / folded split R-hat, multi-chain bulk ESS and the AR(1) null spread of the R-hat statistics, per run, from the stored draws (`convergence_audit.json`) |
| `scripts/prior_energy_audit.py` | Prior energy R and data misfit Phi at the truth, the MAP, the posterior mean and the draws, the same for Gaussian blurs of the truth, and the cost of a log 2 step as a jump or a ramp for the q-GGMRF family (`prior_energy.json`) |
| `scripts/check_data_threads.py` | Builds d_obs at two XLA thread counts in subprocesses and compares hash and numbers (`data_thread_check.json`) |
| `scripts/span_test.py` | Milestone 4 (DESIGN 11.4): the best-in-span lower bound for the book's NMM on held-out Table 15.2 draws and Fig. 15.9-like scenes, an independent second implementation, the P sweep, and the review variants (training eps_r range, clipping, a constant in the span, the metric); writes `results/span_test/`; `--variants-only` adds the variants to stored results |
| `scripts/span_test_tables.py` | Prints the span-test report tables from `results/span_test/span_test.json` (no recomputation) |
| `scripts/_threadcap.py` | Imported first by the standalone scripts: caps the BLAS / OpenMP thread variables at 2 before numpy is imported |
| `scripts/prior_feature_baseline.py` | Prior probabilities of the three DESIGN 11.3 features, the baseline for the posterior probabilities (`prior_features.json`) |
| `scripts/prior_comparison_tables.py` | Prints the report tables from the stored `summary.json` / `comparison.json` files (no recomputation) |
| `scripts/compare_priors.py` | Reads several `run_prior_comparison.py` directories, refuses runs with different data, writes `comparison.json` and the mean / std, disagreement and feature-probability figures |
| `scripts/measure_theta_cost.py` | Cost of U and its gradient, the Jacobian assembly, the Laplace mass and the MAP per prior (milestone 5: the TV prior, the adapted-map potential and one adaptation window update, `cost_tv.json`), at n = 32 and 48 and 1 or 4 XLA threads, one process at a time |

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
  mrf.py          theta-space prior energies: QGGMRF (edge-preserving), PseudoHuberTV (milestone 5), GaussianEnergy (Matern / GMRF
                  cropped to the DOI), WhitenedMap, sample_prior (HMC on R alone)
  adapt.py        milestone 5 (DESIGN 12.2): AffineMap, adapt_map (expanding-window map adaptation, dual-averaged step size,
                  variable-length trajectories), update_map (shrinkage toward the Laplace covariance), sample_chain (frozen recording)
  jacobian.py     exact dd/dchi and dd/dtheta by reciprocity (Nt + Nr solves), Gauss-Newton Hessian, jvp check
  thetaspace.py   ThetaTarget (U = Phi + R), LaplaceMass (dense H, Cholesky), ZetaTarget (reparametrized potential), map_theta,
                  smooth_start
  features.py     pooled moments, edge mask, |z| statistics, disagreement, the DESIGN 11.3 feature probabilities (point values, or
                  disk means of a given radius)
  resolution.py   ideal circular low-pass and disk average of a map (the band-limited reference for the posterior mean), Pearson r
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
