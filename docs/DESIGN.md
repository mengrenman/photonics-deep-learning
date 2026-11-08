# Bayesian inverse scattering testbed: design

Goal: put a posterior on the 2-D high-contrast inverse scattering problem of Campbell & Werner (eds.),
*Advances in Electromagnetics Empowered by AI and Deep Learning*, Ch. 15, §15.4.1.1 (printed pp. 450–451).
The book reconstructs point estimates with no uncertainty. Here we sample the posterior with MCMC to see
which features the data actually determines and which come from the prior.

This document is the contract every module is built against. Where it fixes a formula or a convention,
code must match it exactly; where it says "implementer's choice", document the choice in the docstring.

## 1. Units, conventions, array layout

- Lengths in free-space wavelengths: λ₀ = 1, k₀ = 2π. Time convention e^{−iωt}, so outgoing waves are
  H₀⁽¹⁾. Nonmagnetic media.
- 2-D TM polarization: the unknown field is u = E_z (scalar).
- Relative permittivity ε_r(r) = 1 + χ(r). χ is the **contrast**, the quantity the forward model consumes.
  The **inference parameter is θ = log ε_r** (§3), so χ = e^θ − 1 > −1. Real ε_r in this milestone; the
  forward model accepts complex χ (lossy media) but gradients are taken with respect to real χ.
- Grid: n × n square pixels of side h = side/n covering the domain of interest (DOI) centered at `center`.
  Pixel (i, j) has center x_i = cx − side/2 + (i + ½)h, y_j = cy − side/2 + (j + ½)h.
  **Arrays are indexed [i, j] = [x, y]** ("ij" indexing). Flattening is C-order: k = i·n + j.
  Plot with `imshow(a.T, origin="lower", extent=...)`.
- Float64 / complex128 everywhere (`jax_enable_x64` is set in `bayes_scatter/__init__.py`).
- Randomness: JAX PRNG keys in samplers and priors; `numpy.random.default_rng(seed)` for phantoms.

## 2. Forward model

### 2.1 Lippmann–Schwinger equation

u(r) = u_inc(r) + k₀² ∫_D G(r, r′) χ(r′) u(r′) dr′,  G(r, r′) = (i/4) H₀⁽¹⁾(k₀|r − r′|).

### 2.2 Discretization (Richmond 1965: equal-area circular cells)

Each square pixel is replaced by a disk of equal area, radius a = h/√π, with χ and u constant on it. Define
the pixel kernel K(r, cell m′) = k₀² ∫_{cell m′} G(r, r′) dr′. Then, with ρ = |r − center of m′|:

- off-cell (ρ > a): K = (iπ/2) k₀a J₁(k₀a) H₀⁽¹⁾(k₀ρ)
- self term (r at its own cell center): K = (iπ/2) k₀a H₁⁽¹⁾(k₀a) − 1

(Derivation: ∫₀^a H₀(kρ)ρ dρ = [ka H₁(ka) + 2i/π]/k², using x H₁⁽¹⁾(x) → −2i/π as x → 0; for the off-cell
term use Graf's addition theorem, ∫_disk H₀(k|r − r′|) dA′ = (2πa/k) J₁(ka) H₀(kρ).)

Discrete system for the total field in the DOI, one per transmitter t:

u_t = u_inc,t + G (χ ∘ u_t),  i.e.  (I − G diag χ) u_t = u_inc,t,

where (G w)_m = Σ_{m′} g(m − m′) w_{m′} and g(Δi, Δj) is the kernel above at ρ = h·√(Δi² + Δj²).
G is block-Toeplitz with Toeplitz blocks, so apply it by FFT: embed g in a (2n × 2n) array
K[Δi mod 2n, Δj mod 2n] for |Δi|, |Δj| ≤ n − 1, zero-pad w to (2n × 2n), multiply in Fourier space,
and crop to the first n × n entries.

### 2.3 Sources, receivers, data

- Incident field of a unit line source at r_t: u_inc,t(r) = (i/4) H₀⁽¹⁾(k₀|r − r_t|).
- Scattered field at receiver r_r: u_sc(r_r) = Σ_{m′} R[r, m′] χ_{m′} u_t,{m′} with
  R[r, m′] = (iπ/2) k₀a J₁(k₀a) H₀⁽¹⁾(k₀|r_r − c_{m′}|) (the off-cell kernel). R is a dense (N_r × n²) matrix.
- Data: d[t, r] = u_sc,t(r_r), complex array of shape (N_t, N_r).
- Born approximation: replace u_t by u_inc,t. Then d = A χ is linear, with
  A[(t, r), m′] = R[r, m′] u_inc,t,{m′} (row index t·N_r + r).

Green's kernels, R and u_inc depend only on geometry. Precompute them in NumPy/SciPy
(`scipy.special.hankel1`, `jv`) and store as `jnp` arrays. Only χ-dependent work runs in JAX.

### 2.4 Linear solver

Iterative Krylov solve (`jax.scipy.sparse.linalg.gmres`, optionally `bicgstab`) with relative tolerance
`tol` (default 1e-8) and optional warm start `u0` (an (N_t, n, n) array). It must be `jit`-able and
differentiable with respect to χ: JAX's Krylov solvers go through `lax.custom_linear_solve`, which gives
the adjoint (implicit-function) gradient. **This must be verified against finite differences**, not assumed.

MCMC needs Φ to be a deterministic function of ξ: HMC reversibility and MH correctness both assume it.
Warm starts and loose tolerances make Φ depend slightly on the chain's history. Use tol ≤ 1e-10 in
samplers, and check that |Φ(warm) − Φ(cold)| is far below 1 at the operating point. (A direct dense
solve for n ≤ 48, and an FDFD + sparse-LU backend for high contrast, are planned follow-ups.)

### 2.5 Benchmark geometry (book, §15.4.1.1)

`ch15_setup()`: 300 MHz (λ₀ = 1), 40 transmitters uniformly on a circle of radius 4, 60 receivers
uniformly on a circle of radius 4.5, both starting at angle 0; DOI 5 × 5 centered at the origin, n = 96.
`ch15_setup(n=..., side=...)` gives reduced variants with the same antenna rings.

Sampling note: the scattered field of a domain of radius ρ_D carries about 2k₀ρ_D + 1 angular harmonics.
For the full DOI (half-diagonal 3.54λ, k₀ρ_D ≈ 22) that is about 45, so the 40-transmitter ring (spacing
2π·4/40 ≈ 0.63λ) slightly undersamples, while the 60 receivers (0.47λ) do not. For the reduced side-2 DOI
(k₀ρ_D ≈ 8.9, about 19 harmonics) both rings are well sampled. Interpret full-DOI posteriors with this in mind.

Symmetry note (for later milestones): transmitters every 9°, receivers every 6°, so the acquisition is
invariant under rotations by multiples of 18° (C₂₀) and reflection in the x-axis; together with the square
grid the common symmetry group is D₄.

### 2.6 Phantoms

- `coverage(grid, inside_fn, ss=4)`: fractional area of each pixel inside a shape, by ss × ss
  sub-sampling. Shapes are painted as χ ← (1 − c)·χ + c·χ_shape, so boundary pixels get volume-averaged
  contrast.
- `ch15_austria_variant(rng, grid)`: the book's random training model (Table 15.2, Fig. 15.8): two disks
  (radius ∈ [0.1, 1], center ∈ [−2, 2]²), a ring (inner radius r₃ ∈ [0.1, 1], outer r₄ ∈ [0.2, 2],
  enforce r₄ > r₃, center ∈ [−2, 2]²), a rectangle (sides l₁, l₂ ∈ [0.1, 2]) and a square (side
  l₃ ∈ [0.1, 2]), centers ∈ [−2, 2]², each with ε_r ~ U[1, 8]. Painted in that order; later shapes
  overwrite earlier ones. (The book's table lists "y2" twice; the first is y₁.) Parameters scale with
  `grid.side/5` so the variant also works on smaller DOIs. `ch15_austria_variant(rng, grid, overlap="painter",
  eps_range=(1, 8))`: `overlap="max"` (`paint_max`: the larger contrast wins where shapes overlap) is the alternative the
  book leaves open, `eps_range` the law of each shape's eps_r (the same seed gives the same shapes; the colorbar of
  Fig. 15.9 only reaches 2).
- `austria_fixed(grid, eps_r)`: a fixed test object of our own choosing (not from the book): two disks of
  radius 0.5 at (±0.9, 1.0) and a ring with radii 0.9 and 1.3 centered at (0, −0.8), all with relative
  permittivity eps_r, scaled by `grid.side/5`.
- `cylinder(grid, radius, eps_r, center=(0, 0))`.

## 3. Priors (whitened coordinates)

The prior is a Gaussian field on **θ = log ε_r**, so ε_r is log-normal at each pixel. All inference runs in
**whitened coordinates** ξ ~ N(0, I) under the prior:

θ = log ε_r = μ + C^{1/2} ξ,  ε_r = e^θ,  χ = e^θ − 1.

Consequences of this choice:
- Positivity ε_r > 0 holds by construction, and the prior is scale-symmetric (it penalizes relative, not
  absolute, permittivity changes). The book's range ε_r ∈ [1, 8] is θ ∈ [0, log 8 ≈ 2.08].
- μ = 0 is the vacuum background. A centered prior puts half its per-pixel mass on ε_r < 1; that is the
  honest log-normal statement and the data decide. A slightly positive μ is a modeling option, not a fix.
- σ is in natural-log units: σ = 0.5 means a typical factor e^{0.5} ≈ 1.65 in ε_r.
- dχ/dθ = e^θ = ε_r, so the gradient chain rule is cheap and needs no clipping.

C is a stationary covariance applied by FFT on a padded torus of size P × P, P = pad·n (default pad = 2),
then cropped to the n × n DOI: θ = μ + crop(Re ifft2(√S ⊙ fft2(ξ))), with ξ of shape (P, P). With a
real, even spectral filter S, the result is exactly real and the covariance on the torus is the circulant
with eigenvalues S. Its marginal variance is mean(S), so normalize S ← S·σ²/mean(S).

- `kind="matern"`: S(k) ∝ (κ² + |k|²)^{−(ν + 1)}, κ = √(2ν)/ℓ, k the continuous wavenumbers of the
  P × P torus with spacing h.
- `kind="gmrf"`: S(k) ∝ 1/(κ² + λ_k), λ_k = (4 − 2cos(k_x h) − 2cos(k_y h))/h², the discrete-Laplacian
  eigenvalues (the Gaussian Markov random field of Bouman, Ch. 6).

Links (the map θ → χ):
- `log_eps` (**default**): χ = e^θ − 1, inverse θ = log(1 + χ). Used for all nonlinear inference.
- `identity`: χ = θ. **Validation only**: the exact Born posterior (§7) is Gaussian only when ξ → χ is
  affine, so the linear-Gaussian checks use this link. No other links.

Small-grid helpers for tests: `sqrt_matrix()` returns L (n² × P²) with θ = μ + L ξ, and
`covariance_matrix()` returns L Lᵀ.

## 4. Likelihood and target

- Noise: circular complex Gaussian, d_obs = d + ε, ε ~ CN(0, s²) i.i.d. (real and imaginary parts each
  N(0, s²/2)). From a power SNR in dB (the book's definition): s² = mean|d_clean|² · 10^(−SNR/10).
- Potential (negative log-likelihood up to a constant): Φ(ξ) = Σ_{t,r} |F(χ(ξ))[t, r] − d_obs[t, r]|² / s².
- Posterior in whitened coordinates: π(ξ) ∝ exp(−Φ(ξ)) · N(ξ; 0, I).
- `Target` object: `potential(xi, aux) -> (phi, aux_new)` and `potential_and_grad(xi, aux) ->
  (phi, grad_phi, aux_new)`. `aux` is the (N_t, n, n) total-field array used to warm-start the next solve.
  It is a pure solver accelerator: Φ must not depend on aux beyond solver tolerance.

## 5. Samplers (from scratch, JAX)

Every sampler targets π(ξ) ∝ exp(−Φ(ξ)) N(ξ; 0, I) and takes the same callable
`phi_fn(xi, aux) -> (phi, grad_phi, aux_new)` (grad may be ignored by gradient-free samplers). Each runs
inside `jax.lax.scan`, carries `aux`, and **on rejection restores the aux of the current state**.

| name | proposal | log acceptance ratio |
|---|---|---|
| `pcn` | v = √(1 − β²) u + β w | Φ(u) − Φ(v) |
| `pcn_langevin` (∞-MALA, Cotter et al. 2013) | v = ((2 − δ)/(2 + δ)) u − (2δ/(2 + δ)) ∇Φ(u) + (√(8δ)/(2 + δ)) w | generic MH: [−Φ(v) − ½‖v‖²] − [−Φ(u) − ½‖u‖²] + log q(u∣v) − log q(v∣u), with q(·∣u) = N(μ(u), (8δ/(2 + δ)²) I) |
| `mala` | U = Φ + ½‖ξ‖², v = u − (h/2)∇U(u) + √h w | −U(v) + U(u) + log q(u∣v) − log q(v∣u), q(·∣u) = N(u − (h/2)∇U(u), hI) |
| `hmc` | p ~ N(0, I), L leapfrog steps of size ε on H = U + ½‖p‖² | H(u, p) − H(v, p′) |

w ~ N(0, I). Common features:

- Warmup with step-size adaptation (Robbins–Monro on log step size toward a target acceptance:
  pCN 0.25, pCN-Langevin 0.5, MALA 0.574, HMC 0.7), then the step is frozen. Only post-warmup draws are
  used for estimates.
- `record_fn(xi) -> array` chooses what is stored (default: the χ map). Store every `thin`-th recorded
  value, plus running mean and variance (Welford) of `record_fn` over **all** post-warmup steps, plus Φ
  and the acceptance flag at every step.
- Return a `ChainResult` (samples, phi trace, acceptance trace, final step size, running mean, running
  variance, final state, final aux).
- Several independent chains run with `jax.vmap` over keys and initial states.

## 6. Diagnostics

- `autocorr(x)`: normalized autocorrelation by FFT.
- `tau_int(x, c=5)`: τ = 1 + 2 Σ_{t=1}^{W} ρ(t), with Sokal's automatic window (smallest W with W ≥ c·τ(W)).
- `ess(x) = N / τ`; `mcse_mean(x) = std(x)/√ess(x)`; `split_rhat(chains)`. Cross-check: `ess_geyer(x)`, Geyer's
  initial monotone positive sequence estimator (more conservative on short chains than Sokal's window).

## 7. Inference utilities

- `map_estimate(target, xi0)`: minimize U(ξ) = Φ(ξ) + ½‖ξ‖² with SciPy L-BFGS-B on a jitted
  value-and-grad (host loop). This is the regularized nonlinear least-squares baseline (Tikhonov MAP).
- `born_gaussian_posterior(setup, prior, d_obs, s)`: exact posterior for the Born model with identity
  link. B = A L (complex), stacked real B_r = [Re B; Im B], y_r = [Re(d_obs − Aμ); Im(d_obs − Aμ)],
  σ_r² = s²/2. Posterior on ξ: precision Λ = I + B_rᵀB_r/σ_r², mean m_ξ = Λ⁻¹ B_rᵀ y_r/σ_r².
  On χ: mean μ + L m_ξ, covariance L Λ⁻¹ Lᵀ. Small grids only.
- Reconstruction metrics from the book: Err_model = ‖m_R − m_T‖/‖m_T‖ (eq. 15.21) with m = ε_r, and
  Err_data = ‖d_R − d_T‖/‖d_T‖ (eq. 15.22). Because m includes the background ε_r = 1, eq. 15.21 is
  dominated by background pixels and compresses errors in fine structure. Always also report the
  **contrast-normalized** error Err_χ = ‖χ_R − χ_T‖/‖χ_T‖, and state which metric a number is.

## 8. Validation ladder

1. **Kernel**: self term matches direct numerical quadrature of k₀² ∫_disk G; off-cell term matches the
   quadrature for ρ > a.
2. **Forward vs. Mie series**: dielectric cylinder, line-source illumination, analytic coefficients
   a_n = [k₁J_n′(k₁R)J_n(k₀R) − k₀J_n(k₁R)J_n′(k₀R)] / [k₀J_n(k₁R)H_n′(k₀R) − k₁J_n′(k₁R)H_n(k₀R)],
   u_sc = (i/4) Σ_n a_n H_n(k₀ρ_t) H_n(k₀ρ) e^{in(φ − φ_t)}. Relative error at the receivers must decrease
   under grid refinement.
3. **Gradient**: `jax.grad` of Φ vs. central finite differences along random directions (relative error
   < 1e-5 at tol = 1e-12).
4. **Samplers on a known Gaussian**: posterior mean and variance recovered within Monte Carlo error. A
   deliberately broken sampler must fail the same test (negative control), so the test is not vacuous.
5. **Samplers on the exact Born posterior**: the physics forward model with the Born linearization, on a
   small grid, against `born_gaussian_posterior`.
6. **Nonlinear posterior** (log_eps link): the book's acquisition rings, reduced DOI and contrast first,
   then toward the book's 5λ × 5λ, ε_r ≤ 8 setting. Report posterior mean and standard deviation of both
   log ε_r and ε_r, z-maps, ESS per second, and the book's error metrics (computed on ε_r).
   **Avoid the inverse crime**: synthesize observed data on a 2× finer grid (n_data = 2n, same DOI,
   coverage-averaged phantom) and infer on the n grid. Same-grid runs are allowed only as a labeled
   control. Keep ≥ 10 pixels per interior wavelength λ₀/√ε_r,max on the data grid.
   **Report the operating point** of every problem (`inference.operating_point`): the grid error of the
   forward model at the truth, ‖F_n(χ_true) − d_clean‖/‖d_clean‖, against the noise amplitude
   s√N/‖d_clean‖, and Φ(χ_true) against N. Because E[Φ(χ_true)] − N = (model error / noise)² N, a model
   error of 0.25 of the noise amplitude already puts Φ at the truth 0.06 N (3 sd for N = 2400) above N;
   larger ratios mean the posterior is conditional on a visibly wrong forward model and its edge
   behavior partly measures discretization error. Choose n so the ratio is ≲ 0.25, or lower the SNR.
   **Report the physical misfit.** A sampler that runs in reparametrized coordinates (for example
   Laplace-whitened ζ) has a potential that is not Φ; quote Φ of the stored draws, not the sampler's trace.
7. **Prior sensitivity**: a feature counts as data-determined only if structurally different priors agree
   on it (at minimum Matérn vs. GMRF here; edge-preserving and shape-family priors later). Features
   outside the Born band |K| ≤ 2k₀ are prior-determined under any single Gaussian prior by construction.

## 9. Layout

```
src/bayes_scatter/
  geometry.py     Grid, Antennas, Setup, ring(), ch15_setup()
  phantoms.py     coverage(), cylinder(), austria_fixed(), ch15_austria_variant()
  greens.py       pixel kernel, kernel FFT embedding, receiver matrix, incident fields
  forward.py      Forward: apply_G, solve_fields, scattered, __call__, born, born_matrix
  mie.py          analytic cylinder scattering for validation
  priors.py       GaussianField, links
  likelihood.py   noise model, misfit, Target
  samplers.py     pcn, pcn_langevin, mala, hmc, ChainResult
  diagnostics.py  autocorr, tau_int, ess, mcse_mean, split_rhat
  inference.py    map_estimate, born_gaussian_posterior, error metrics
  plotting.py     maps and traces
  mrf.py          theta-space prior energies (QGGMRF, GaussianEnergy), WhitenedMap, sample_prior          (milestone 4)
  jacobian.py     exact Jacobian by reciprocity, Gauss-Newton Hessian                                    (milestone 4)
  thetaspace.py   ThetaTarget, LaplaceMass, ZetaTarget, map_theta, smooth_start                          (milestone 4)
  features.py     pooled / merged moments, edge mask, |z| statistics, the 11.3 features                   (milestone 4)
  adapt.py        AffineMap, adapt_map (expanding-window map adaptation), update_map, sample_chain          (milestone 5)
                  mrf.py also holds PseudoHuberTV (12.1); thetaspace.ZetaTarget accepts any affine map
  shapes.py       TwoDisks, exact coverage and boundary-integral shape derivatives, stretch_derivatives     (milestone 6, 13.2)
  fisher.py       data derivatives, Fisher matrix, CRB, required SNR, Born ratios (same-s and matched-power),
                  irrep/symmetry split, nuisance_report (local regime), detection_limit                      (milestone 6)
  gapvalley.py    the resolution-scale gap-valley feature, Gaussian functionals, orthant probability          (milestone 6, 13.3)
  mle.py          Levenberg-Marquardt maximum likelihood of the two-disk parameters (CRB threshold check)     (milestone 6 review)
  ellipses.py     EllipseRenderer (exact coverage and boundary derivatives of ellipses, JAX), the truths T1 to T5,
                  FeatureObject                                                                              (milestone 7, 14.2, 14.3)
  shapemodels.py  F2, F2u, Fe: unconstrained parametrizations, priors with Jacobians, symmetry images           (milestone 7, 14.2)
  shapepost.py    ShapePosterior, GaussianToy, MAP, Laplace evidence, HMC, multimodality, thermodynamic integration,
                  predictive checks                                                                          (milestone 7, 14.2, 14.4)
scripts/          validate_forward.py, born_exact_check.py, run_posterior.py (+ analyze_run.py, make_tables.py,
                  measure_cost.py, plot_gn_spectrum.py); prototypes outside the ladder: run_posterior_laplace.py
                  (Laplace-whitened sampling), map_multistart.py (MAP from several starts);
                  milestone 4: run_prior_comparison.py (+ make_prior_jobs, compare_priors, edge_analysis,
                  convergence_audit, prior_energy_audit, prior_feature_baseline, prior_comparison_tables,
                  gmrf_diagnostic, check_data_threads, measure_theta_cost) and span_test.py (+ span_test_tables); _threadcap.py is
                  imported first by the standalone scripts (BLAS threads <= 2);
                  milestone 5: tv_lambda_sweep.py; run_prior_comparison.py gains --prior tv and --adapt (stage adapt),
                  make_prior_jobs.py the names tv_lam<value> and <prior>_adapt; segment_loop.py (one chain's segments, one process
                  each), make_m5_resume.py / make_m5_batch.py / wait_then.py (job files and gating), check_chain_starts.py,
                  milestone5_tables.py;
                  milestone 6: resolution_crb.py (13.2; --book-gaps, --born-power), resolution_laplace.py (13.3: lambda, config, descend, refeature,
                  null, collect), resolution_mcmc.py (chain, finalize, energy, refeature, collect; --run-tag, --frozen-step, --tail-segments),
                  resolution_cost.py, make_m6_jobs.py / make_m6_extra_jobs.py / make_m6_review_jobs.py (job files for batch_run.py);
                  review follow-up: resolution_mle.py (Monte Carlo ML), resolution_detect.py (detection limit, ellipticity),
                  resolution_review_analysis.py (low-pass reading, SNR check, TV-weight sensitivity, p_eff, book geometry);
                  milestone 7: parametric_run.py (stages prepare, map, chain, ppc, ti, finalize, pixel, cost), parametric_collect.py,
                  make_m7_jobs.py (batch_run job files, restrictable to a plan and to models), parametric_profile.py (profile marginal of the gap),
                  parametric_sweeps.py (F2 Laplace fits over noise seeds and over the T2 ellipticity ladder), parametric_ti_offline.py;
                  review follow-up (14.7): parametric_snr.py (F2/F2u fits at 10, 20, 30 dB over noise seeds, analytic misfit flag),
                  parametric_evidence_is.py (importance-sampling log evidence), parametric_discretization.py (noise-free fits to data of
                  other grids, offset-corrected calibration); make_m7_jobs.py gains the batches is, snr, disc
tests/            one test file per module, plus end-to-end Born test
```

## 10. Later milestones (not in this build)

- **Laplace pass** (next): exact Jacobian by reciprocity (40 + 60 solves; J[(t,r), p] ∝ u_t(p) u_r(p)),
  the Gram eigendecomposition at background, truth and MAP, and the count of data-constrained directions
  versus SNR. (The MCMC/BO judge reproduced 380/468/628/797 of 9,216 Born directions above
  1e-1/1e-2/1e-4/1e-6 σ₁ for the full book geometry; the Ewald-disk count is π(2/λ)²(5λ)² ≈ 314.)
- **Span test** of the book's NMM (done, §11.4): by eqs. 15.5–15.6 its output lies in the span of the 100 training
  targets. Independent implementations of the Table 15.2 generator give a best-in-span error with median 34% and
  0.1th percentile 15.5% on held-out draws at P = 100 (the minimum, 9.9–16%, is set by the single easiest draw and
  is not quoted), and median 9.1% (Test #1-like) and 10.7% (Test #4-like) for Fig. 15.9-like scenes, against the
  reported 3.26–3.31%. The cause is unknown (results/span_test/, results/SPAN_TEST.md).
- Resolution audit: is sub-wavelength detail in the posterior, or only in the prior? Verdict by agreement
  across three priors (GMRF, smoothed TV via HMC, the book's 22-parameter scene family via tempered SMC).
- Edge-preserving (Huber/TV-MRF) and learned score-based priors (plug-and-play Langevin).
- Delayed-acceptance MCMC with a neural surrogate as the cheap first stage, for the full book setting.
- Replica exchange for multimodal (cycle-skipping) posteriors at high contrast.
- D₄ symmetry: block-diagonalize by irreducible representation; equivariant priors.
- Institut Fresnel experimental data (the book's "FoamMetExt" example; 2-D at 5 GHz, 241 receivers at 1°
  over a 240° arc, 8 or 18 transmitters, per Ch. 15 pp. 454–459).
- Resolution on an object manifold: pull back the Fisher metric DF*Σ⁻¹DF to parametrized object families
  and compare Cramér–Rao bounds for gap/size parameters with posterior widths (Ch. 15 Tests #4–#8).
- Stable degrees of freedom of the receiver arc: spectrum of the aperture-limited concentration operator.

## 11. Milestone 4: prior sensitivity (the edge question) and the span test

### 11.1 Question

Under the smooth Matérn prior, 10–13% of pixels of the moderate object miss the truth by more than 3σ, almost
all within one pixel of an edge, and removing the grid error does not remove the ring (results/REPORT.md §7.2).
Is the ring a property of the prior? And which features (the 0.32λ gap between the disks, the 0.16λ-thick ring,
the ring's hole) are determined by the data, i.e. agreed on by structurally different priors (§8 step 7)?

### 11.2 Sampling in θ-space with a dense Laplace mass matrix

All priors in this milestone are energies on the n × n grid of θ = log ε_r (no padded torus):

U(θ) = Φ(χ(θ)) + R(θ),  χ = e^θ − 1.

- **Matérn / GMRF (cropped)**: R(θ) = ½ (θ − μ)ᵀ C_D⁻¹ (θ − μ), with C_D = `GaussianField.covariance_matrix()`,
  the exact DOI marginal of the torus field of §3 (dense n² × n², Cholesky-factored once). Same prior as §3.
- **q-GGMRF** (Thibault, Sauer, Bouman & Hsieh 2007; Bouman Ch. 8): edge-preserving,
  R(θ) = Σ_{i~j} b_ij ρ(θ_i − θ_j) + ‖θ − μ‖²/(2σ₀²), over the 8-neighborhood inside the DOI (free boundary),
  b = 1 for horizontal/vertical pairs and 1/√2 for diagonal pairs, with
  ρ(Δ) = (|Δ|^p / (p σ_g^p)) · 1 / (1 + |Δ/(T σ_g)|^{p−q}),  p = 2, 1 ≤ q < 2.
  Quadratic for |Δ| ≪ Tσ_g, growing like |Δ|^q beyond. Convex for 1 ≤ q ≤ 2. The weak anchor σ₀ makes the prior
  proper. **Implementation must be autodiff-safe at Δ = 0**: write |Δ|^{p−q} as (Δ² + ε²)^{(p−q)/2} with
  ε ~ 1e-12, or use a `jnp.where` guard. Test that gradient and Hessian-vector products are finite at Δ = 0.
- **Laplace mass matrix**: H = H_GN + ∇²R at the MAP θ̂, with H_GN = (2/s²) Re(Jᴴ J), J = ∂d/∂θ (complex,
  N_t N_r × n²) = (∂d/∂χ) diag(e^θ). Assemble J exactly, preferably by reciprocity (one extra solve per
  receiver, using the receiver kernel row as the incident field). Verify columns against `jax.jvp`. H must be
  SPD (Gauss–Newton is PSD, and the priors are convex); check with Cholesky.
- **Reparametrization**: θ = θ̂ + L_H⁻ᵀ ζ with H = L_H L_Hᵀ. The samplers of §5 target exp(−Φ(ξ)) N(ξ; 0, I),
  so feed them Φ'(ζ) = U(θ(ζ)) − ½‖ζ‖². This is exact for any fixed θ̂ and H (an affine change of variables).
  Report the physical misfit Φ of the stored draws, never Φ' (§8 step 6).
- **Sampler**: HMC in ζ (dual-averaging step size during warmup, jittered trajectory length), 4 chains started
  from θ̂ + overdispersed Laplace draws (scale 2). Convergence: split R̂ ≤ 1.01 for Φ, max pixel R̂ ≤ 1.05,
  worst-pixel ESS ≥ 200 (Sokal) pooled over chains. Chains that miss this are reported as unconverged and no
  conclusion rests on them. Compare the MCMC pixel std with the Laplace std as a non-Gaussianity diagnostic.
  These thresholds are within the noise at 300 to 600 draws per chain (an AR(1) null with τ ≈ 3 gives a classic R̂ of Φ
  above 1.01 in 16–28% of runs), so also report the rank-normalized bulk and folded split R̂ with their AR(1) null
  spread, the multi-chain bulk ESS and the per-chain |z| > 3 fractions (`scripts/convergence_audit.py`,
  `scripts/edge_analysis.py`); the DESIGN flag "converged" stays the classic one.

### 11.3 Experiment

Moderate problem: austria_fixed, ε_r = 2, side 2 (disks of radius 0.2λ at (±0.36, 0.4), edge-to-edge gap
0.32λ; ring radii 0.36λ and 0.52λ at (0, −0.32), thickness 0.16λ), inference grid n = 48 (grid error 0.27 of the
noise, results/operating_points/moderate.json), data on n = 96, SNR 30 dB, **one data realization shared by all
priors**. Priors: Matérn (σ 0.5, ℓ 0.4), GMRF (the §3 parameters used in results/), q-GGMRF with q = 1.2,
T = 1, σ₀ = 1 and two values of σ_g (0.05 and 0.1, log-ε_r units per neighbor difference; the true edge jump
is log 2 ≈ 0.69).

Report per prior: physical Φ of the draws; posterior mean and std of θ and ε_r; prior std (by sampling the
prior alone for q-GGMRF); |z| > 3 fraction and its share within one pixel of an edge; err_model, err_χ; the
convergence numbers above. Report across priors: a pixelwise disagreement map
|m_A − m_B| / √(s_A² + s_B²) > 2, and three feature probabilities with δ = log(2)/2:
- **gap valley**: P(θ at the gap midpoint (0, 0.4) < min(θ at the two disk centers) − δ),
- **ring hole**: P(θ at the ring center < mean θ over the annulus − δ),
- **ring continuity**: P(θ at the four compass points of the annulus mid-radius all > δ).
A feature is **data-determined** only if every converged posterior gives P > 0.95, or every one gives P < 0.05, AND the
minimum of §8 step 7 is met: a converged Matérn and a converged GMRF are among them. Two settings of one family (the two
q-GGMRF values of σ_g) are one prior, not two. Without the pair the verdict is **provisional**; if an unconverged run
gives the other side of the threshold it is **not established** (`features.verdict`). A prior that cannot reach the
feature amplitude (the q-GGMRF with marginal sd 0.057 against δ = 0.347 has prior P = 0) says nothing about structure.

Concurrency on the M3 Max (12 performance cores): use scripts/batch_run.py. Never exceed 12 XLA threads in total.
NPROC = 2 can deadlock XLA, so use T = 1 or T ≥ 4 per job. Measure the per-evaluation cost first.

### 11.4 Span test of the book's NMM (Ch. 15, eqs. 15.1–15.6, Table 15.3)

By eqs. 15.5–15.6, α̂ = T G_r⁺, so every NMM output o = α̂ g_r(·) lies in span(T), the span of the P = 100
training permittivity images (96 × 96, 5λ DOI). The best output the NMM could possibly produce on a test scene m is
therefore the least-squares projection of m onto span(T). Its error ‖Π_T m − m‖/‖m‖ (eq. 15.21, m = ε_r including
background) lower-bounds the NMM's reported misfit. The book reports 3.26–3.31% for Tests #1–#4 (Table 15.3).
Compute the bound for:
(a) 1000 held-out draws from the Table 15.2 generator;
(b) Fig. 15.9-like scenes: Test #1 is "the simple disk and the rectangle with the low contrasts" (ε_r ≤ 2); Test #4
    is "three overlapped disks". The exact geometries are not given, so use several plausible instantiations and
    report the range.
Also report variants: the painter's rule vs. max-overlap, P = 100 vs. larger P, and nearest-training-image error.
Two independent implementations must agree before any number is reported (§8 anomaly rule).
Quote the median and low percentiles (5th, 0.1th from the tail scan), not the minimum: the minimum of 40 × 1000 values is
the easiest value of ONE draw (9.9% in one implementation, 14–16% in the others; `heldout_min_diagnosis`).
Anomaly triage: report which deviations from the literal reading of eqs. 15.1–15.6 were tested and which are ruled
out or open: training eps_r law (U[1, 8], U[1, 3], U[1, 2]), a constant in the span, clipping to the colorbar after an
oracle shift, the chi-normalized error, a squared misfit (contradicted by the book's BA and VBIM numbers). Anchor the
conclusion on the scene where every reading fails (Test #4), not on the one where a reading comes close (Test #1).

### 11.5 Execution notes (added after the run)

- One value-and-gradient evaluation at n = 48 speeds up only 2.0 to 2.2 times from one to four XLA threads
  (results/prior_comparison/cost.json), so four single-thread processes do about 1.8 times the work of one four-thread
  process. The experiment therefore ran one chain per process (`scripts/run_prior_comparison.py --stage prepare / chain /
  finalize`, job files from `scripts/make_prior_jobs.py`, `scripts/batch_run.py --threads 1`), with at most 10 threads
  in total. A chain can be continued in segments (`--segment j`) from its final state with the frozen step size; the
  moments of the segments are merged exactly (`features.merge_moments`).
- Laplace mass at another point: `--mass-theta FILE.npy` rebuilds H = H_GN(theta_m) + Hess R at a chosen theta_m and
  centers the chains there (exact, an affine change of variables); `--start-theta` replaces the overdispersed starts.
  It was tried only for the GMRF prior, whose typical posterior state is far from its MAP: it cut the largest eigenvalue
  of the local Hessian in the chain coordinates from about 270 to 8 at the pilot states (results/prior_comparison/
  gmrf_diagnostic.json) but the chains kept drifting outward and the GMRF run stayed unconverged.

### 11.6 Review follow-up (added after the review)

- The edge ring is **not** shown to be independent of the prior: every prior tested prefers blurred edges (prior energy of
  the truth against the MAP and the draws, results/prior_comparison/prior_energy.json); the q-GGMRF family is convex, so every
  member prefers a ramp to a sharp step (Jensen). Whether a prior that does not penalize sharp edges (near total variation,
  a non-convex potential, the shape family) removes the ring is untested.
- Matérn was extended to 600 draws per chain and q-GGMRF σ_g = 0.05 to 900 (`--segment 1`, `2`; the 600-draw state missed
  R̂ of Φ ≤ 1.01 by 0.0003 and is kept in `pc_qggmrf_sg0.05_600draws/`); a Matérn same-grid control (`pcg1_matern`, data on
  the inference grid) pairs the q-GGMRF one, both at 3 × (300 + 600).
- The d_obs hash is not reproducible across XLA thread counts; data equality is numerical (max |Δd| / s ≤ 1e-6).


## 12. Milestone 5: an edge-indifferent prior and a converged GMRF

Milestone 4 (results/PRIORS_REPORT.md) showed that the edge bias is common to every prior tried. For convex ρ with
ρ(0) = 0, k·ρ(Δ/k) ≤ ρ(Δ), so every convex pair potential weakly prefers a k-pixel ramp to a step; total variation
is the equality case. Two gaps remain: no prior that is (nearly) indifferent between a step and a ramp was tried,
and the GMRF, required by the §8 step 7 minimum comparison, did not converge.

### 12.1 Pseudo-Huber total-variation prior

R(θ) = λ Σ_{i~j} b_ij (√(Δ_ij² + δ²) − δ) + ‖θ − μ‖²/(2σ₀²),  Δ_ij = θ_i − θ_j,

with the 8-neighborhood and weights of §11.2, δ = 0.01 (log-ε_r units, ≪ the true jump log 2), σ₀ = 1, μ = 0. It is
smooth and convex, its Hessian is PSD (so the Laplace mass of §11.2 stays SPD), and its asymptotic slope λ is the TV
weight. Report the step-vs-8-pixel-ramp cost ratio (it should be within a few percent of 1).

Choosing λ: a MAP sweep over λ spanning at least a factor of 30 (e.g. 0.3, 1, 3, 10), reporting Φ_MAP, R_MAP and
err_model at each value. λ* is the value whose Φ_MAP is closest to N (Morozov's discrepancy principle; a MAP
heuristic, stated as such). Run full posteriors at λ* and at one neighbor (λ*/3 or 3λ*) to show sensitivity.

### 12.2 GMRF to convergence: adapted affine map

The milestone-4 GMRF chains drifted: the local Hessian along the chains reached about 270 times the MAP Laplace
scale, which forced the HMC step down to about 0.06 while |ζ| kept growing. Remedy:
- **Adapt the affine map in warmup only.** θ = θ_c + L ζ, with θ_c and L Lᵀ = Σ̂ re-estimated from warmup draws in
  expanding windows. Σ̂ shrinks the sample covariance toward the current Laplace covariance H⁻¹; the weight and
  windows are the implementer's choice, documented. Freeze the map before any recorded draw, so every recorded
  segment is an exact HMC chain for a fixed affine change of variables.
- **Validate on a Gaussian target of the same dimension** with an ill-conditioned covariance that adaptation does
  not bias, with a negative control (adaptation left on during recording on a non-Gaussian target).
- **Longer, jittered trajectories:** L uniform in [L₀/2, 3L₀/2], with L₀ ε ≈ 1.5 to 2 in whitened units; ε by dual
  averaging toward 0.8.
- **Run 4 chains, overdispersed from the adapted map, with at least 1000 recorded draws each.** Same convergence
  criteria as §11.2, plus rank-normalized split R̂ (bulk and folded) ≤ 1.01 for Φ and multi-chain bulk ESS ≥ 200
  for the worst pixel. Unconverged chains are reported as such.

### 12.3 Verdicts

Same data realization as milestone 4 (moderate object, n = 48, data on n = 96, 30 dB, seed 0). Re-run the three
feature probabilities and the §11.3 verdicts with the families Matérn, GMRF, q-GGMRF, TV. If the GMRF converges, the
§8 step 7 minimum (a converged Matérn and a converged GMRF) can be met. For each converged posterior report the edge
statistics of PRIORS_REPORT §4.5 (bias and std at d = −1, |z| > 3 by distance to the edge), and the prior-energy
audit (R and Φ at the truth, MAP and draws). The question to answer: under the TV prior, does the edge bias shrink,
and does the |z| > 3 ring disappear?

### 12.4 Implementation notes (added after the milestone-5 build)

- **TV prior** (`mrf.PseudoHuberTV`; `run_prior_comparison.py --prior tv --lam --delta --sigma-0 --mu`; the prior std is by HMC on R
  alone, as for the q-GGMRF). The pair value is evaluated as D²/(√(D² + δ²) + δ), which equals √(D² + δ²) − δ without its cancellation.
  **The step-over-ramp ratio is not within a few percent for the 8-pixel ramp at δ = 0.01**: a ramp of k equal increments saves about
  (k − 1) δ per pair line, so the ratio is 1 + (k − 1) δ / log 2 to first order: at δ = 0.01 it is 1.014 (k = 2), 1.044 (k = 4) and
  1.106 (k = 8); at δ = 0.003 1.004 / 1.013 / 1.031; at δ = 0.001 1.001 / 1.004 / 1.010 (`prior_energy_audit.py` rows
  "TV pseudo-Huber"; the q-GGMRF settings of milestone 4 give 2.2 and 2.7). δ = 0.01 stays the default.
- **MAP path** for the sweep is the milestone-4 one (smooth Matérn start, `thetaspace.map_theta`), but the TV MAP needs 680 to
  1550 L-BFGS-B iterations, not 400 (`--map-maxiter 4000`; `make_prior_jobs.py` passes it for `tv_lam<value>`). With 1000 iterations
  Φ_MAP was already within 0.01 of its value at convergence (`tv_lambda_sweep_pilot1000.json`).
- **lambda sweep** (`scripts/tv_lambda_sweep.py`, `results/prior_comparison/tv_lambda_sweep.json`; delta = 0.01, sigma_0 = 1, the
  data of `pc_*` (max |delta d| / s = 6.5e-14), N = 2400; each MAP from the same smooth start, converged by L-BFGS-B's relative
  reduction criterion in 630 to 1920 iterations, the best-so-far U changing by at most 3e-4 over the last 10% of evaluations):

  | lambda | Phi_MAP | R_MAP | err_model | err_chi | MAP bias at d = -1 |
  |---|---|---|---|---|---|
  | 0.3 | 2372.0 | 176.2 | 0.1156 | 0.3548 | 0.109 |
  | 1 | 2379.4 | 372.6 | 0.1156 | 0.3549 | 0.094 |
  | 1.25 | 2381.4 | 441.7 | 0.1129 | 0.3467 | 0.092 |
  | 2 | 2387.1 | 647.3 | 0.1159 | 0.3559 | 0.094 |
  | 3 | 2394.2 | 918.8 | 0.1188 | 0.3646 | 0.097 |
  | **3.75** | **2399.5** | 1120.8 | 0.1204 | 0.3697 | 0.100 |
  | 4.5 | 2404.8 | 1321.7 | 0.1219 | 0.3742 | 0.102 |
  | 10 | 2447.1 | 2770.2 | 0.1306 | 0.4010 | 0.120 |
  | 11.25 | 2457.9 | 3094.3 | 0.1324 | 0.4065 | 0.124 |

  **lambda* = 3.75** (Phi_MAP 2399.5 against N = 2400; the log-log interpolated crossing is 3.82); lambda*/3 = 1.25 and
  3 lambda* = 11.25 are in the table. Phi_MAP rises monotonically, 2372 to 2458 over a factor 37 in lambda (the data fix most of the MAP);
  err_model of the MAP is 0.113 to 0.132, below the Matern (0.1260), GMRF (0.1246) and q-GGMRF (0.1296, 0.1322) MAPs of milestone 4; the
  differences of about 0.003 between neighboring lambda at the small end are at the level of the MAP's convergence history (Phi is
  not convex in theta), not a trend. The MAP's bias just outside the object (mean of MAP - truth at d = -1) is +0.09 to +0.12 in
  log eps_r, about half the +0.2 of the milestone-4 posterior means (a MAP-to-posterior-mean comparison; see PRIORS_REPORT 4.5).
  At delta = 0.003 (`tv_lambda_sweep_delta0.003.json`): Phi_MAP 2382.6, 2402.7, 2458.8 and err_model 0.1048, 0.1136, 0.1267 for
  lambda = 1.25, 3.75, 11.25: Phi_MAP stays within 4 of its delta = 0.01 value and err_model is 0.006 to 0.008 smaller, i.e. the discrepancy
  choice of lambda does not depend on delta in this range.
- **Adaptation** (`adapt.py`; `run_prior_comparison.py --adapt`, stage `adapt`). Choices left open by 12.2:
  windows of 50, 100, 200, 400 trajectories after an initial 50 (step size only) and before a final 100 (step size only), each
  rounded up to blocks of 25 (`--adapt-init`, `--adapt-first-window`, `--adapt-windows`, `--adapt-growth`, `--adapt-final`,
  `--adapt-block`); the update uses the draws of its own window only (the trajectory states, repeats included), re-centers at their
  mean, and shrinks with weight w = κ/(n + κ), κ = N = n² (`--adapt-kappa-per-dim 1`), toward the Laplace covariance **rebuilt at the
  new center** (H = H_GN + ∇²R there, 0.5 s at n = 48), i.e. Σ̂ = (1 − w) S + w H⁻¹(θ_c). κ ∝ N because a window is shorter than N: a
  smaller κ leaves too much of the rank-deficient sample covariance in the map and whitens worse (see Validation: condition
  number of the whitened covariance 159 / 32 / 16 for κ = 0.05 N / 0.2 N / 0.5 N against 11 for κ = N and 7 for κ = 2N). The weight
  decays like κ/n, so windows of several N are dominated by the draws. A window with mean acceptance below 0.1 does not update.
  HMC: L uniform on {⌈L₀/2⌉, …, ⌊3L₀/2⌋}, L₀ = round(1.75/ε) capped at 100 (`--traj-length`, `--max-leapfrog`), no step-size jitter;
  ε by Nesterov dual averaging toward 0.8 (γ = 0.05, t₀ = 10, κ = 0.75), restarted at the averaged ε after every phase.
- **Run structure.** One adaptation chain (from the Laplace map of the MAP, started at 0.5 N(0, I) in its coordinates) writes
  `adapted_map.npz` (center, lower Cholesky factor of Σ̂, ε, trajectory length) and `adaptation.json` (arguments, history of every
  phase with its acceptance, ε, L₀, gradient evaluations, and for every update the weight, variance ratios, center shift and timings).
  The recording chains start `--init-scale` × N(0, I) in the coordinates of that map around the final state of the adaptation chain
  (`--adapt-start final`, the default; `center` starts around the map center, the mean of the last window, which in the GMRF pilot
  was not a probable state: Φ = 2.2e5 and every proposal was rejected), run `--warmup` trajectories (rounded up to
  blocks) of dual averaging of ε alone with the map frozen, then `--samples` recorded trajectories with ε, L₀ and the map frozen. Each
  segment file carries the digest of the map; `finalize` and `chain` refuse a mixture or a map file that no longer matches
  `adaptation.json`. Continuations (`--segment j`) keep the chain's frozen ε. `thetaspace.ZetaTarget` takes any affine map
  (`ThetaTarget.zeta_target_with_map`); the default (non-adapted) path is unchanged.
- **Validation** (`tests/test_adapt.py`): a 23 x 23 GMRF Gaussian, N = 529, condition number of the precision 1.3e4, started from a
  Laplace-like map that is wrong by a factor 100 in the variance of its long-wave modes (the order of the milestone-4 GMRF mismatch).
  After 5 windows (75 to 1200 trajectories; 17,000 gradient evaluations, 9 s on one thread) the exact whitened covariance A^-1 Sigma
  A^-T has eigenvalues in [0.30, 3.7] (condition number 12, against 99 for the starting map); four overdispersed chains of 1500
  recorded trajectories with the map and step frozen recover the mean and variance diagonal (rms z 1.8 and 1.0, max |z| 3.0 and 2.9,
  mean variance ratio 0.98), and the worst-coordinate ESS per gradient evaluation of one chain is 14 to 18 times that of the
  unadapted map with its own dual-averaged step. Negative control, a banana (x1 ~ N(0, 10^2), x2 = -0.01 (x1^2 - 100) + N(0, 1), so
  E x1^2 = 100, E x2 = 0, E x2^2 = 3): 8 chains of 4000 trajectories with the map and step frozen give z = +0.5, -0.4, +0.6 for these
  three moments; with adaptation left on during recording (windows of 25 trajectories, each re-centering and re-estimating from its
  own draws) z = -10.1, +8.0, -8.6 (E x1^2 = 85.7, E x2 = 0.17, E x2^2 = 2.5). The shrinkage weight matters: on a 576-dimensional
  prototype with the same construction, kappa = 0.05 N, 0.2 N, 0.5 N, N, 2N left the whitened covariance at condition number 159,
  32, 16, 11 and 7.
- **Cost at n = 48** (`results/prior_comparison/cost_tv.json`, shared machine, load 6 to 8): one value-and-gradient of U for the TV prior
  0.114 s on one XLA thread and 0.057 s on four (Matern 0.113 / 0.058, q-GGMRF 0.114 / 0.057: the prior is free); the potential in the
  coordinates of an adapted map (two dense matrix-vector products) costs the same as in the Laplace map (two triangular solves).
  One window update (sample covariance of 400 draws, Laplace mass and covariance at the new center, mixture and Cholesky, N = 2304):
  1.10 s on one thread, 0.44 s on four.
- **Finding for the TV posterior** (a pilot that was stopped after three of its six adaptation phases, run directory
  `pilot_tv_lam3.75_adapt`, logs in results/prior_comparison/logs/pilot_tv_lam3.75_adapt_adaptation.log): at delta = 0.01 the pair
  curvature lam b / delta at a pair with D = 0 is about 130 times that at |D| = 0.05 (lam b delta^2 / (D^2 + delta^2)^1.5), so the local Hessian along a chain varies by
  two orders of magnitude and a single linear map cannot whiten it; the dual-averaged step in the Laplace map of the MAP was 0.045
  (L0 = 39, mean 42 leapfrog steps per trajectory, about 4.4 s per trajectory on four threads), after the first update 0.041 and after the
  second 0.020 (mean 65 steps). Expect about 8 s per trajectory on one thread (65 steps at 0.114 s) for the TV posterior at delta = 0.01; a larger delta is the lever.
- **Finding for the GMRF** (pilot `pilot_gmrf_adapt`: the MAP of `pc_gmrf` copied, a short schedule of 25 / 25, 50, 100 / 50
  trajectories, 15,561 gradient evaluations in 48 minutes on a loaded machine; then two chains with `--warmup 25 --samples 20`): the
  dual-averaged step fell with every phase, 0.039 (Laplace map of the MAP), 0.033, 0.020, 0.018, 0.017 (L0 at the cap of 100), because
  the chain leaves the MAP (the window means were at |ζ| = 229, 81 and 62 from the previous centers in the previous map's whitened units; a Laplace draw has |ζ| ≈ 48) while the map
  hardly changes: the sample covariance had weight 1 − w = 1 to 4 % (κ = N = 2304 against windows of 25 to 100 trajectories) and the
  Laplace covariance rebuilt at the window mean has the same marginal variances as at the MAP (median ratio 0.99 to 1.00). The
  recording chains show why the start matters: around the map center (the last window's mean) the physical Φ is 2.2e5 and every
  proposal was rejected (`pilot_gmrf_adapt/start_center/`); around the final state of the adaptation chain (now the default) acceptance
  was 0.85 and 1.0, the step 0.011 and 0.016 and the physical Φ of the draws 2429.8 ± 6.6, the level of the milestone-4 GMRF (2427.7);
  20 draws are far from convergence (R-hat and ESS are reported as failing). So the machinery runs at n = 48, but a schedule of a few
  hundred trajectories does not fix the GMRF step size. The levers are longer windows (w = 0.5 at n = N), a smaller
  `--adapt-kappa-per-dim`, and `--mass-theta` at a typical state (milestone 4: `pc_gmrf/mass_theta_pilot.npy`) for the initial map;
  `update_map` now reports `trace_sample_over_laplace`, the variance the window saw relative to the Laplace covariance, for that tuning.

### 12.5 Execution and results (added after the milestone-5 runs; corrected after the review, see 12.6)

Every number below is read from a file under `results/prior_comparison` (named in each item); the tables are printed by
`scripts/prior_comparison_tables.py` and `scripts/milestone5_tables.py` into `m5/tables_main.md` and `m5/tables_m5.md`, and the commands
are in `scripts/m5_analysis.sh` and `scripts/m5_review_analysis.sh`. Run notes (what was stopped when, and why): `logs/m5_resume_notes.txt`;
the commands that were actually run, including the manual launches: `scripts/m5_commands_as_run.md`.

- **History and checkpointing.** A first attempt (one process per chain and per segment of 300 to 500 trajectories, `make_m5_batch.py`) ran
  8.5 hours and was killed by a reboot (2026-10-11 about 20:55); the GMRF chains, 2.8 hours in, had written nothing. The resumption runs each
  chain as a loop of segments of 100 recorded trajectories, one process per segment (`scripts/segment_loop.py`; job files by
  `scripts/make_m5_resume.py`; a restarted loop skips the segments that exist), so a reboot loses at most one segment. Settings that did not
  change: data seed 0, n = 48, data on n = 96, 30 dB, TV delta 0.01, sigma_0 1, lambda* = 3.75 and its neighbor 3 lambda* = 11.25, the adapted
  maps of the first attempt (`pc_gmrf_adapt`, `pc_tv_lam3.75_adapt`, `pc_tv_lam11.25_adapt`; digests in each `adaptation.json`).
- **Starts: none of the milestone-5 runs started from an overdispersed state** (DESIGN 12.2 asks for that). The GMRF chains started at
  `pc_gmrf_adapt/m4_ends.npy`, the end states of the four stuck milestone-4 GMRF chains (pixel sd 0.187), and the adaptation chain at
  `m4_end0.npy` (`--mass-theta`, `--adapt-start-scale 0`). The TV chains started at 0.5 N(0, I) in the coordinates of the adapted map around ONE state, the
  final state of the adaptation chain. The job file `scripts/jobs/m5_batch1.json` says `--init-scale 1.0`, whose starts have a physical Phi of 4e4 to 8e4
  (`logs/pc_tv_lam3.75_adapt_c0_s0_manual.log`, first line), so segments 0 and 1 of the TV chains were launched by hand with `--init-scale 0.5`
  (the `starts` field of each segment file says "scale 0.5"). The re-adapted runs start at the final states of the chains they continue. So the
  per-chain agreement (bias within 0.002) and the R-hat of Phi of about 1.000 are weak evidence, and the "AR(1) null" (chains from ONE distribution) does
  not describe GMRF chains that are still drifting: |zeta|^2 / 2 of the first against the last 100 recorded draws is 3677 to 5817, 3407 to 7469,
  3991 to 9274 and 3555 to 6793 for the four chains (N / 2 = 1152), and the mean prior energy by block of 100 draws rises from 1035 to 1114
  (`m5/review/start_drift.json`; the TV chains at lambda* are flat, prior energy 2844 to 2880 by block). Overdispersed starts for TV are
  in 12.6.
- **The machine saturates at about five single-thread processes.** One value-and-gradient of the GMRF potential took 0.14 to 0.15 s with four
  chains plus the benchmark running and 0.22 to 0.28 s with nine processes (also with the extra four at `nice` 20): nine processes gave 19 %
  more gradients per second than five and slowed each by half (numbers in `logs/m5_resume_notes.txt`, scratch benchmark). The GMRF, the item
  that decides the section 8 step 7 minimum, therefore got the machine (the TV chains of the first batch were stopped with SIGSTOP from
  21:34 until that batch was ended at 22:58); more processes would not have produced more draws.
- **Re-adapting the map from the recorded draws** (`scripts/readapt_map.py`, new run directories `pc_gmrf_adapt2`, `pc_tv_lam3.75_adapt2`,
  `pc_tv_lam11.25_adapt2`; the source runs are never modified). Why: in the coordinates of the first GMRF map the recorded draws kept drifting
  (|zeta|^2/2 of chain 0 rose from 3,082 to 4,357 over 300 draws against N / 2 = 1,152) and the pooled covariance of zeta had eigenvalues of
  910, 740, 600, 330 (`scripts/slow_modes.py`, `m5/slow_modes.json`) where a whitening map gives 1. The recorded draws are a far longer window than the
  adaptation had (450 trajectories), so `adapt.update_map` was applied once more to the pooled draws of all chains (weights on the Laplace term
  0.66 GMRF, 0.42 TV lambda*, 0.59 TV 3 lambda*; `adaptation.json` of each new run, entry `readapt`), the chains restarted from the final states of
  the old ones (`starts.npy`) with a fresh 50-trajectory step-size warmup, and the old draws are warmup. **A re-adapted run is a continuation of the
  same four chains, not new chains**: `starts.npy` equals `theta_final` of the source's last segment (max |difference| 0;
  `m5/cross_run_tv_lam3.75.json`, key `relations`). **Result at equal numbers of draws per chain** (`m5/review/equal_length_audit.json`): the re-adapted map
  **helped TV lambda\*** (500 draws: max pixel R-hat 1.20 against 1.745 for the first 500 draws of the first map and 1.483 for its draws 301 to 800;
  worst bulk ESS 19.9 against 9.9 and 11.6; 25.4 against 37.0 leapfrog steps per trajectory, i.e. about three times the bulk ESS per gradient evaluation) and **hurt the
  GMRF** (200 draws: max R-hat 1.62 against 1.135, 1.118 and 1.175 in the three 200-draw windows of the first map; bulk ESS 7 against 19 to 29) and
  **TV 3 lambda\*** (200 draws: 1.635 against 1.349 and 1.359; bulk ESS per gradient 0.28 against 0.30 and 0.34). The first version of this item said "it helped neither" and
  compared runs of unequal length (the decision to stop `tvA2` compared 400 draws with 800). The largest zeta-Hessian
  eigenvalue at the chain end states (power iteration) rose for the GMRF (3,937 and 4,475 to 5,295 and 5,019) and fell for TV (285 and 355 to 151 and 154 at
  lambda*; 343 and 380 to 242 and 228 at 3 lambda*), and the TV step grew (0.025 to 0.037-0.047 at lambda*, 0.022 to 0.030 at 3 lambda*). The GMRF chains of
  the new map (`pc_gmrf_adapt2`, 200 draws) have a physical Phi of 2447.7 against 2429.6, with the same leading eigenvalues in the NEW coordinates (996, 806, 528): in these
  few directions the posterior is much wider than any covariance estimated from a chain that has not yet crossed them, and each re-estimate lets the chain move
  proportionally further. Along the leading principal directions of the pooled GMRF draws the chain means differ by several units of theta
  (`m5/slow_modes_theta_pca.png`: localized, fine-scale blobs spread over the domain, not long-wave modes). The re-adapted GMRF run was therefore stopped after
  200 draws and the chains of the first map were continued (to 600 draws).
- **Adaptation settings of the production runs differ from the DESIGN and the validation** (exactness is not affected, efficiency is). DESIGN 12.4
  states kappa = N and its prototype found 0.2 N worse (condition number 32 against 11); the production runs used **kappa = 0.25 N = 576** (`--adapt-kappa-per-dim 0.25`,
  `adaptation.json`), windows of 50 + 100 trajectories
  (TV) or 50 + 100 + 200 (GMRF), against five windows and 2,325 trajectories in the validation, a TV trajectory length of 1.0 where 12.2 asks for 1.5 to 2
  (the GMRF used 1.5), and 800 / 400 / 600 recorded draws per chain (TV lambda\* / 3 lambda\* / GMRF) where 12.2 asks for at least 1000. The freeze is verified: the segment digests
  555cc867 (GMRF), a71befbb (TV lambda\*) and 59c6b7a1 (TV 3 lambda\*) equal the `map_digest` of each `adaptation.json`, every final phase has `updated` false,
  the warmup of a chain uses `n_windows` 0, and `assemble_chains` refuses mixed maps. "The posterior is wider than any covariance a chain can estimate" is therefore a
  statement about this SHORT schedule, not about adaptive HMC.
- **TV posteriors** (`pc_tv_lam3.75_adapt`: 4 x (50 + 800); `pc_tv_lam11.25_adapt`: 4 x (50 + 400); step 0.025 to 0.031 and 0.022, 39 and 46 leapfrog
  steps, acceptance 0.85 and 0.90): **unconverged by the DESIGN criteria** (max pixel R-hat 1.32 and 1.14, worst Sokal ESS 98 and 75; R-hat of Phi
  1.004 and 1.001; `summary.json`, `m5/convergence_audit.json`). Where the slow pixels are (`m5/convergence_audit.json`, `flagged_by_region`): of the 72 pixels
  with a rank-normalized R-hat above 1.05 at lambda*, 10 lie inside the object (d >= 2), 4 within one pixel of the edge (|d| = 1) and 58 two or more pixels outside it
  (d = -16 to -2; 51 at d <= -4), i.e. mostly far-field background, not "the interior of the two small disks" as the first version said; at 3 lambda* 111
  pixels, 96 of them at d <= -4. The pixels that matter for the edge question are therefore better sampled than the maximum suggests: the pooled
  last 500 draws of the first-map and re-adapted chains give max R-hat 1.052 over the 498 pixels with |d| <= 1 and worst bulk ESS 144
  (`m5/cross_run_tv_lam3.75.json`), **but that pool is the same four chains in two consecutive windows with two maps, a stationarity check, not eight chains** and not an
  independent replication; independent-start replications are in 12.6. The two windows agree: at lambda* the bias at d = -1 is 0.190 and 0.189, the sd
  0.106 and 0.106, the |z| > 3 fraction 2.2 and 2.4 %; at 3 lambda* 0.181 and 0.179, 0.063 and 0.062, 12.5 and 12.9 % (`m5/edge_analysis.json`).
- **GMRF** (`pc_gmrf_adapt`, first map, 4 x (50 + 600), step 0.016 to 0.017, 89 leapfrog steps, acceptance 0.86): R-hat of Phi 1.000, **max pixel R-hat 1.064 (7
  pixels above 1.05 by the rank-normalized statistic, 3 of them within one pixel of an edge), worst Sokal ESS 231, worst multi-chain bulk ESS 58, rank-normalized max R-hat 1.062: unconverged** (it misses the max pixel R-hat of 1.05 and
  the bulk ESS of 200; it passes the other two classic criteria). The audit's AR(1) null (probability 0.1 of a max pixel R-hat at least this large for chains from ONE distribution,
  `m5/convergence_audit.json`) is NOT evidence of convergence here: the chains started from one family of states and were still drifting (see "Starts" above). Extending from 300 to 500 and 600 draws moved the max pixel R-hat 1.079, 1.069, 1.064 and the worst
  Sokal ESS 122, 230, 231 (`summary_300draws.json`, `summary_500draws.json`, `summary.json`), i.e. slowly. Discarding the first 200 draws as burn-in
  (`--skip-segments 2`, `pc_gmrf_adapt_burn2`) does not help (1.090, ESS 154, bulk 38). The split R-hat of the leading principal directions of the pooled draws in map
  coordinates is 3.1, 3.2, 2.5, 2.2, 1.8 (eigenvalues 1,024, 817, 685, 508, 379; `m5/slow_modes.json`): about six directions have not mixed, which the pixel criteria see only in the pixels that load on them.
  The posterior is far wider than the milestone-4 GMRF run said: the pixel sd is 0.33 to 0.34 (0.18 for `pc_gmrf`, which never left the neighborhood of the MAP; prior sd 0.5) and the
  prior energy of the draws 1,085 +- 42 against 525 (`m5/prior_energy.json`). The Laplace approximation at the GMRF MAP (no MCMC) has pixel sd 0.349, close to the MCMC value
  (`m5/review/feature_scales_laplace.json`). The eight chains of the two maps do not sample one distribution yet (R-hat of Phi 1.41 over the last 200 draws of each, `m5/cross_run_gmrf.json`; these are
  four chains in two windows).
- **Features** (`m5/comparison.json`; the milestone-4 GMRF `pc_gmrf` is superseded and does not vote): gap valley / ring hole / ring continuity: Matern 1.000 / 0.984 / 1.000, q-GGMRF 1.000 /
  1.000 / 1.000 (both settings), TV lambda* and 3 lambda* 1.000 / 1.000 / 1.000 (0.9997 for continuity at lambda*), **adapted GMRF 0.920 / 0.713 / 0.501** at single points (DESIGN 11.3).
  The range over the four chains is 0.90-0.94 / 0.67-0.75 / 0.49-0.52, over 100-draw blocks pooled over chains 0.870-0.968 for the gap valley, and the re-adapted map (a continuation of the same chains,
  200 draws) gives 0.912 / 0.711 / 0.304 with chains from 0.15 to 0.41 for continuity: the chain-to-chain "+-" of an earlier version (0.007 for continuity) understated the uncertainty of an
  unconverged run by an order of magnitude and is dropped (`m5/review/feature_scales.json`). The milestone-4 report had 1.000 / 0.996 / 0.929 for the stuck GMRF. An independent recomputation from the stored draws (`scripts/check_features_independent.py`: scipy interpolation, geometry written out from the DESIGN text) reproduces every probability of every run
  exactly (`m5/independent_features.json`, max difference 0). **The single-point GMRF numbers do not show that the gap and the hole are undetermined by the data**: a single-point value of a posterior with pixel-scale
  roughness (pixel sd 0.34) carries all of that roughness, while the experiment resolves about a quarter wavelength. Averaged over a disk of radius 0.08 wavelength (10 to 14 pixels, half the ring thickness) the GMRF gives
  1.000 / 0.954 / 0.840 and over 0.12 wavelength (24 to 28 pixels) 1.000 / 1.000 / 0.962, with the gap valley 1.000 in every chain from 0.08 wavelength on (0.994 pooled at 0.06) and the ring hole 0.99 at 0.10; the Laplace approximation at the GMRF MAP, with no MCMC (4000 exact draws),
  gives 0.933 / 0.669 / 0.596 at points, 0.999 / 0.904 / 0.911 at 0.08 and 1.000 / 0.998 / 0.998 at 0.12 (`m5/review/feature_scales.json`, `feature_scales_laplace.json`). Under the GMRF the gap valley and the ring hole are therefore determined at the resolution scale;
  ring continuity is not at 0.08 (0.84 MCMC, 0.91 Laplace), approaches 0.96 to 1.00 at 0.12, and falls again at 0.16 (0.83), where the averaging disk is wider than the ring: the answer for continuity depends on the averaging scale. (The review quoted 1.000 for the ring hole at 0.08 from its own
  disk definition; with the definition of `features.disk_mean`, all pixel centers within the radius, it is 0.954 by MCMC and 0.904 by the Laplace approximation, and 1.000 from 0.12.) Whether the DESIGN 11.3 features
  should be resolution-scale averages is a design decision; the DESIGN definition (points) is kept for the verdicts. The verdict function gives "not established: unconverged gmrf disagrees with the converged runs (present)" for all three features, and the **section 8 step 7 minimum (a converged Matern and a converged GMRF) is not met**.
  Ring continuity needs all four compass points above delta; with independent compass values its probability would be the product of the four marginals. For the adapted GMRF that product is 0.516 with Gaussian
  marginals of the interpolated draws and 0.502 with the empirical marginals, against the sampled 0.501, and the largest correlation between two compass values is 0.08: the compass points are effectively
  independent. An earlier version of this item said the sampled 0.50 exceeded the product of Gaussian marginals (0.35) "because the compass points are positively correlated"; the 0.35 came from using the
  bilinear interpolation of the pixel sds as the sd of the interpolated value, which overstates it for a rough posterior (`m5/independent_features.json`, fixed in `scripts/check_features_independent.py`).
- **The edge question: the posterior-mean bias is the resolution limit of the data, common to all priors; the priors differ in the pixel-scale variance they admit, and that sets the |z| > 3 ring**
  (`m5/edge_analysis.json`, `m5/review/resolution_analysis.json`, `m5/prior_energy.json`, `m5/edge_profile.png`). Bias of the posterior mean at d = -1 (the pixel row just outside the object): Matern 0.215, q-GGMRF 0.208 and 0.204, TV
  lambda* 0.189, TV 3 lambda* 0.180, adapted GMRF 0.134 (unconverged); the MAP of the TV prior has 0.100 and 0.124 (`tv_lambda_sweep.json`), the posterior mean does not. The data fix the spatial-frequency
  content of the object only inside a disk of radius of order 2 k0 = 2 cycles per wavelength (more with multiple scattering). **The ideal circular low-pass of the true theta at 3 cycles per wavelength** (1.5 times the Born support) has bias
  +0.182 at d = -1, -0.131 at d = +2 and +0.178 at d = +4 (the overshoot at the disk centers; at 2.5 cycles per wavelength +0.234 and -0.201), rms 0.107, and **the posterior-mean bias map of every prior is close to this one**: correlation 0.93 (Matern), 0.89 (q-GGMRF
  0.05), 0.89 (TV lambda*), 0.86 (TV 3 lambda*), 0.87 (GMRF), 0.91 to 0.97 within |d| <= 3, rms 0.111 to 0.126. The TV MAP correlates less (0.69): the mode super-resolves, the mean does not. Against the low-passed truth the |z| > 3
  fractions fall from 12.5, 15.5, 5.8, 2.2, 12.5 and 0.0 % (Matern, q-GGMRF 0.05 and 0.1, TV lambda*, TV 3 lambda*, GMRF) to 0.2, 3.4, 0.5, 0.3, 4.6 and 0.0 %, and at d = -1 from 61.3, 64.5, 27.4, 7.7, 32.7 and 0.0 % to 0.0, 3.2, 0.0, 0.0, 3.6
  and 0.0 %, i.e. the "ring" is the difference between the truth and its low-passed version in units of the claimed posterior sd. **It is not "chosen by the prior" (milestone 4) and the blur is not made by averaging**: each TV draw is itself blurred. The mean over the 248 pixels at d = -1 of one
  draw's bias is 0.189 +- 0.008 from draw to draw (`m5/edge_analysis.json`, `per_draw_edge_blur`), the pixel values at d = -1 are unimodal (median 0.177, 99th percentile 0.556, 0.4 % above 0.6; the true jump is 0.69), so these are not sharp edges at random positions. A possible
  mechanism, not tested here: with a pair-difference noise of 0.1 to 0.15, E|a + noise| is quadratic in a small mean increment a, which makes TV act like a strictly convex potential on the mean profile and prefer a ramp. The priors differ in how
  much pixel-scale variance they allow: the sd at d = -1 is 0.061 (Matern), 0.052 and 0.083 (q-GGMRF), 0.106 (TV lambda*), 0.063 (TV 3 lambda*), 0.34 (GMRF), and the |z| > 3 fraction at d = -1 follows it: 61.3 %
  (Matern), 64.5 and 27.4 % (q-GGMRF), 7.7 % at lambda* and 32.7 % at 3 lambda*, 0 % for the GMRF; pooled over all pixels 12.5, 15.5, 5.8, 2.2, 12.5 and 0.0 %. So the ring "disappears" at lambda* only because the posterior gets wider; it is back at 3 lambda* with Matern's sd (0.063 against 0.061).
  **The bias of the adapted GMRF (0.134) is not comparable with the others.** Far from the object the posterior mean of theta lies below the MAP by -0.061, close to -sd^2/2 = -0.058: the mean of log eps_r of a wide
  posterior is lower than the log of the mean permittivity (Jensen), and the same offset makes the posterior-mean map data-inconsistent (physical Phi 3.6e5). The bias of log E[eps_r] at d = -1 is 0.191 for the GMRF, 0.217 (Matern),
  0.195 (TV lambda*) and 0.183 (TV 3 lambda*) (`m5/edge_analysis.json`, `log_mean_eps_bias`): in this metric TV lambda* reduces the bias against Matern by 10 %, not 12 %, and the GMRF is no better than the others.
  Prior energies (`m5/prior_energy.json`): R(truth) - R(MAP) = 214 at lambda* against 14,598 for Matern, 1,687 and 712 for the q-GGMRF settings and 56 for the GMRF. For TV lambda* 122 of the 214 nats are the loss of
  contrast of the MAP (the TV seminorm of the MAP is 310.0 against 342.6 for the truth, 90.5 %; `m5/review/lambda_band.json`), so the prior is not "nearly indifferent" to the edge at the MAP, only much less
  opposed to it than the smooth priors; at delta = 0.003 the MAP bias at d = -1 is 0.080 against 0.100 at 0.01 (`tv_lambda_sweep_delta0.003.json`). The excess energy of typical draws over the truth (R(draws) 2,860 +- 43 against R(truth) 1,335 at lambda*) is
  the generic typical-set effect, not specific to edges: the Matern prior has R(draws) - R(MAP) = 1,093 too, so R(truth) against R(draws) is not a like-for-like comparison and says nothing about "rough configurations".
- **Limits.** No run of milestone 5 passes the DESIGN criteria. A single data realization (a second one was not run for TV); lambda* by a MAP heuristic that hardly constrains lambda (12.6); delta = 0.01 is not exactly TV (step over 8-pixel ramp 1.106);
  the bias is that of the posterior mean of log eps_r (E[theta], and log E[eps_r] for cross-prior comparisons); the machine was shared with other work (load averages 6 to 45) so wall times are indicative; the re-adapted maps did not converge faster
  for the GMRF and TV 3 lambda*.

### 12.6 Review follow-up (added after the review of the milestone-5 report)

The review reproduced the numbers of the report from the raw files and kept its top-level conclusion (nothing converged, the section 8 step 7 minimum is not met, TV does not remove
the posterior-mean edge bias). It found explanations and supporting checks that did not hold; the corrections are in 12.5 (marked there) and the new evidence is here.
Everything new is under `results/prior_comparison/m5/review/` unless a path says otherwise; the commands are `scripts/m5_review_analysis.sh` (analysis) and
`scripts/make_m5_review_jobs.py` + `scripts/batch_run.py` (one batch: two more TV posteriors and an independent-start replication; 4 + 4 + 2 single-thread chain loops after
the adaptations, at most 10 XLA threads at any time).

- **The posterior-mean edge bias is the resolution limit of the data** (`resolution_analysis.json`, `src/bayes_scatter/resolution.py`, `scripts/resolution_analysis.py`): see the edge bullet of 12.5.
  Against the low-passed truth (ideal circular cutoff at 3 cycles per wavelength, zero padded) the pooled |z| > 3 fraction is 0.0 to 4.6 % for every run, against 0.0 to 15.5 % against the truth; the bias maps of all priors correlate 0.86 to 0.93 with the low-pass bias map (12.5).
- **The discrepancy principle hardly constrains lambda** (`lambda_band.json`, `lambda_band.png`, `scripts/lambda_band.py`). Phi_MAP rises only from 2372.0 to 2457.9 over lambda = 0.3 to 11.25; the band N +- sqrt(N) is 2351 to 2449 and
  every swept value but 11.25 is inside it. The log-log interpolated lambda for Phi_MAP = N = 2400 is 3.82, for Phi_MAP = the Phi of noise-free data (2438.2) 8.46, for the upper edge of the band 10.21; Phi of the truth (2578.1, which
  contains the grid error) lies above the swept range. "The ring disappears at lambda*" is therefore a statement about one choice inside a wide band; the edge statistics are given as a function of lambda
  (table below). The runs at lambda*/3 = 1.25 and 8.5 are `pc_tv_lam1.25_adapt` and `pc_tv_lam8.5_adapt`: the schedule of the first-attempt TV runs (windows 50 + 100, kappa 0.25 N, trajectory length 1.0), 50 warmup + 100 recorded draws per chain,
  chains started at four DIFFERENT maps (truth, the MAP of that lambda, the Matern posterior mean, a draw of the rough adapted GMRF; `starts.json`: TV prior energy at lambda = 8.5 2910, 2379, 3253 and 23,090),
  not at one state of the adaptation chain.

  | lambda | Phi_MAP (band 2351 to 2449) | MAP bias d=-1 | posterior bias d=-1, E theta / log E eps_r | sd d=-1 | % abs z>3 at d=-1 | % abs z>3 pooled | pooled, against the low-passed truth | chains x draws used |
  |---|---|---|---|---|---|---|---|---|
  | 1.25 (lambda*/3) | 2381.4 | 0.092 | 0.162 / 0.183 | 0.207 | 0.0 | 0.0 | 0.0 | 3 x 50 (chains 0-2, first 50 dropped; prior energy still rising) |
  | 3.75 (lambda*) | 2399.5 | 0.100 | 0.189 / 0.195 | 0.106 | 7.7 | 2.2 | 0.3 | 4 x 800 |
  | 3.75, truth start (replication) | | | 0.192 / 0.197 | 0.106 | 10.1 | 3.5 | 0.3 | 1 x 150 |
  | 8.5 (Phi_MAP = Phi of noise-free data) | 2434.7 | 0.115 | 0.184 / 0.187 | 0.073 | 26.6 | 8.4 | 2.0 | 3 x 50 (chains 0-2, first 50 dropped) |
  | 11.25 (3 lambda*) | 2457.9 (above the band) | 0.124 | 0.180 / 0.183 | 0.063 | 32.7 | 12.5 | 4.6 | 4 x 400 |

  (`lambda_band.json`, `lambda_band.png`; all-chain rows are in the JSON: lambda = 1.25 gives 0.160 / 0.185, sd 0.226; lambda = 8.5 gives 0.178 / 0.187, sd 0.120, 6.0 % at d = -1, because the chain started at the rough GMRF draw is far from stationary.)
  **The posterior-mean bias at d = -1 is 0.16 to 0.19 for every lambda in the band, the sd falls from 0.21 to 0.06 as lambda rises from 1.25 to 11.25, and the |z| > 3 fraction at d = -1 rises from 0 to 33 % with it.** The ring is a statement about the
  claimed width, set by a lambda that the discrepancy principle does not fix; against the low-passed truth the pooled fraction is 0.0 to 4.6 % at every lambda. The runs at 1.25 and 8.5 have 100 recorded draws per chain (one segment of the chain loop; the stop file `scripts/jobs/stop/m5v` was created
  58 minutes into the 90-minute budget, the machine ran at load 8 to 11) and none of them is converged (`convergence_audit.json`: max pixel R-hat 2.9 and 16, bulk ESS 5, because of the transients below).
- **Independent-start replication** (`pc_tv_lam3.75_indep`: two chains in the frozen map of `pc_tv_lam3.75_adapt`, the same target, started at the truth and at a GMRF draw (TV prior energy 1335 and 10,302 against 2860 for typical draws); 50 warmup + 200 recorded draws;
  `start_drift.json`, `cross_run_tv_lam3.75_indep.json`). The chain from the truth (ring bias 0) reaches the posterior bulk: its prior energy falls from 2980 to 2846 (2844 to 2880 by block in the first run) with physical Phi 2442 to 2447, and its ring bias rises from 0.165 to 0.19 to 0.198 in blocks of 25 draws (0.192 after the first 50 draws; sd 0.106; |z| > 3
  10.1 % at d = -1 and 3.5 % pooled; `lambda_band.json`): **the edge numbers of the first run (0.189, 0.106, 7.7 %) are reproduced by an independent start near the bulk.** The chain from the rough start is **not** at the bulk after 250 trajectories: prior energy 5795 to 4081, Phi 2479 to 2461
  (against 2442), |zeta|^2 / 2 from 3.6e5 to 2.7e5 (typical 3.9e3), ring bias 0.156 to 0.178; split R-hat over the two chains 2.08 for Phi and 1.81 for the ring bias (after 50 draws; `start_drift_new_runs.json`), ring continuity per chain 0.99 and 0.12. The same happens in the new lambda runs: the chains started at the truth, the MAP and
  the Matern mean (ring bias 0, 0.09 to 0.12 and 0.215) are at ring bias 0.16 (lambda = 1.25) and 0.18 to 0.19 (lambda = 8.5) after their first tens of draws, while the rough-start chain has prior energy 7090 (against 4030) at lambda = 8.5 after 150 trajectories. So (i) independent starts support the TV edge numbers
  for starts in the neighborhood of the bulk, with a spread of the ring bias between chains of about 0.004 (new runs) to 0.014 (replication: 0.192 and 0.171) instead of the 0.002 quoted before; (ii) from a really overdispersed start (the rough GMRF draw) the TV chains relax slowly in the adapted map, so the agreement of the first runs' chains
  (their recorded draws begin at |zeta|^2 / 2 of 3.65e3 to 4.0e3: one family of states near the bulk) is not a convergence result. The pooled check against the first run is in `cross_run_tv_lam3.75_indep.json` (`chains_independent` true; pooled R-hat of Phi 1.70 over the last 200 draws, max pixel R-hat 8.7).
- **Overdispersed starts and the run length required by 12.2 are still not met for the GMRF.** The GMRF was not re-run (its chains need hours for 1000 draws at the observed step of 0.016); the GMRF numbers stay "unconverged, started from the
  stuck milestone-4 end states, still drifting". A GMRF run with starts that differ (the truth, the MAP, the Matern mean, a TV draw), a longer adaptation (windows up to 400 trajectories, kappa = N as in the validation) and at least 1000 draws per chain is the open item
  for the section 8 step 7 minimum.
- **Feature definitions** (`feature_scales.json`, `features.feature_indicators(radius=...)`): at the resolution scale the adapted GMRF gives 1.000 / 0.954 / 0.840 with disks of radius 0.08 and 1.000 / 1.000 / 0.962 with 0.12 (points: 0.920 / 0.713 / 0.501); the Laplace approximation at its MAP, without MCMC, 0.999 / 0.904 / 0.911 and 1.000 / 0.998 / 0.998 (points 0.933 / 0.669 / 0.596)
  (`feature_scales.json`, `feature_scales_laplace.json`; 12.5). The weak TV run at lambda = 1.25 (sd 0.21) has point values 1.000 / 0.895 / 0.728 (chain continuity 0.82, 0.98, 0.96 and 0.15 for the rough-start chain) and disk values at 0.08 of 1.000 / 1.000 / 0.820 (1.00, 1.00, 1.00 and 0.28), i.e. the same
  mechanism: pixel-scale variance lowers the point probabilities, not a lack of determination at the resolution scale. DESIGN 11.3 keeps the point definition for the verdicts; whether the features should be
  resolution-scale averages (disk radius 0.08 to 0.12 wavelength) is a design decision for the user.
- **Smaller corrections.** The ring-continuity check uses the sd of the interpolated draws (`check_features_independent.py`, test `test_independent_check_uses_the_sd_of_the_interpolated_draws_not_the_interpolated_pixel_sd`); re-adaptation is compared at equal draws (`scripts/equal_length_audit.py`,
  `equal_length_audit.json`); the cross-run pools carry a `relations` / `chains_independent` flag (`cross_run_check.py`); the convergence audit reports where the slow pixels are (`flagged_by_region`) and can audit a window of the draws (`--first`, `--draws`);
  the edge analysis reports the bias of log E[eps_r] and the per-draw blur; the table of feature probabilities prints the chain range, not a standard error, for unconverged runs; the frozen arm of the adaptation negative control (`tests/test_adapt.py`)
  records 16 times as many trajectories (standard error of E x1^2 1.5 instead of 13) and asserts that a bias of the size the adaptation-on arm shows would stand out of it by at least 3 standard errors in every moment.
- **Not done.** A second data realization for TV; the GMRF re-run above; a decision on the feature definition; the pair-noise mechanism for the ramp preference of the TV mean (stated as a possible mechanism only).

## 13. Milestone 6: the resolution sweep

### 13.1 Question

Ch. 15 (Sec. 15.4.2.1, pp. 464–467) calls a reconstruction "super-resolved" when the image shows features below 0.25λ (p. 464).
It supports this with two sets of single reconstructions (corrected after the milestone-6 review; the first version of this
section quoted "gaps from λ/4 down to λ/20 at 10–20 dB", which is not what the book shows):
**Test #4**, six objects of unequal permittivity (3.2 down to 2.2 in steps of 0.2) with adjacent spacings 5/20 to 1/20 λ, compared
with VBIM, for which **no noise level is stated**; and **Tests #5–8**, nine connected objects (permittivity 2.2 to 3.8 in steps of
0.2) with spacings 4/10, 3/10, 2/10 and 1/10 λ in noise-free, 20 dB and 10 dB environments (Table 15.9). The noisy operating
points are therefore λ/10 to 2λ/5 at 10 and 20 dB, and λ/20 appears only without a stated noise level. The measurements are 3-D
(128 × 162 = 20,736 antenna pairs against 40 × 60 = 2400 here: +9.4 dB of aggregate SNR at equal per-datum SNR) at 500 MHz
(λ = 0.6 m) with antenna planes 0.35 to 0.45 m (0.58 to 0.75 λ) from the objects. Over that standoff an evanescent component of
5 cycles/λ (the half period of a λ/10 gap) decays by about e⁻¹⁸ per path and one of 10 cycles/λ (λ/20) by about e⁻³⁶, so near-field
information does not supply the missing spatial frequencies in the book's geometry either (`results/resolution/book_geometry.json`):
the gap in its λ/20 images cannot come from evanescent data. What differs from this benchmark is the 3-D geometry, 9 times more data,
objects of unequal permittivity or connected, and a trained network with a learned prior.
Milestone 5 found that every pixel posterior
mean equals the truth low-passed at about 3 cycles/λ. The question here is which gap widths the **data** determine,
as a function of gap, contrast and SNR, and where pixel posteriors stop reflecting the data and start reflecting
the prior. The two halves are kept apart:
(a) **information**: Fisher information and Cramér–Rao bounds for a parametrized object family, with no pixel prior;
(b) **pixel posteriors**: Laplace posteriors under the Matérn and TV(λ*) priors on the same data, scored with
resolution-scale features, with MCMC spot checks.

### 13.2 Object family and exact shape derivatives

Two disks of common radius r (default 0.2λ) and common ε_r, centers (c_x ± (r + g/2), c_y), with edge-to-edge gap g.
Parameters p = (g, r, ε_r, c_x, c_y) plus, for the asymmetric checks, the radius difference r₁ − r₂. χ(p) is rendered
by fractional coverage. **∂χ/∂p is computed exactly from the boundary**, not by differencing the rendering:
∂coverage_pix/∂(boundary displacement) = (∫_{boundary ∩ pixel} (displacement · n) ds)/h². Evaluate it by dense
sampling of each circle, with points binned into pixels. Check it against finite differences of high-supersampling
renderings. Then ∂d/∂p_k = J (∂χ/∂p_k), evaluated as a `jax.jvp` (no dense J at fine grids), and the Fisher matrix
in the shape parameters is F = (2/s²) Re(D_pᴴ D_p), D_p = [∂d/∂p_k].

- **CRB**: σ_CRB(g) = √(F⁻¹)_gg with nuisance parameters unknown; also report √(1/F_gg) with nuisances known.
  Statistical resolution criterion (Smith 2005): g is resolvable when σ_CRB(g) ≤ g. F ∝ 1/s², so report the SNR
  at which σ_CRB(g) = g, as a "required SNR vs gap" curve (power SNR as in §4).
- **Born comparison**: the same F with the Born forward (u = u_inc). The ratio F_full/F_Born for g, as a function of
  contrast, is reported in **two normalizations** (review): at one noise level (that of the full model), where it mostly
  measures how much more power the Born model scatters (an effective contrast scaling), and with each model at its own power
  SNR (§4), which is the comparison per unit of scattered power and the one that answers whether multiple scattering adds
  information about the gap; with P_full/P_Born for every object.
- **Local regime** (review): σ_CRB(g) = g at SNR_req is a statement about an estimator only where the nuisance parameters are
  determined there. Report sd(r)/r and sd(ε_r)/(ε_r − 1) at SNR_req; if either exceeds 0.1 the row is a lower bound of unknown
  tightness, not a required SNR. For at least two such points run a threshold-aware check (Monte Carlo maximum likelihood of the
  five parameters at SNR_req: `scripts/resolution_mle.py`).
- **Detection limit** (review): F stays well conditioned as g → 0 (the center distance 2r + g never shrinks to zero, unlike two
  point sources), so the bound is meaningful at small g apart from the boundary g ≥ 0. Report the gap detectable against g = 0 by a
  one-sided test (level 0.05, power 0.9), (z₀.₉₅ + z₀.₉)·σ_CRB(0) = 2.93 σ_CRB(0), next to Smith's g*; and the bound with the
  ellipticity of both disks (s_x, s_y, gap fixed) added to the unknowns.
- **Symmetry check**: the antenna rings and the symmetric two-disk object are invariant under x → −x. Split the
  parameters into the even irrep (g, r, ε_r, c_y) and the odd irrep (c_x, r₁ − r₂). F must be block-diagonal across
  the two irreps to solver precision. Test it.
- **Grid**: forward model on a fine grid (side 1.5λ, n ≥ 128, h ≤ 0.012λ, so the smallest gap spans ≥ 2 pixels).
  Check CRB convergence between two grid sizes.

Sweep: g ∈ {0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.32, 0.5}λ, ε_r ∈ {1.1, 2, 4}, book antenna rings
(40 transmitters at r = 4, 60 receivers at r = 4.5), SNR reference 30 dB.

### 13.3 Pixel posteriors on the same objects

Inference grid side 1.5λ, n = 64 (h ≈ 0.023λ), data from the n ≥ 128 grid at 30 dB (no inverse crime), log ε_r
link, Matérn (σ 0.5, ℓ 0.4) and TV at λ* (re-choose λ* by the discrepancy rule for this geometry and state it). For
each (g, ε_r): MAP, the dense Laplace covariance of §11.2, and the gap-valley probability. **Resolution-scale
features (decision of 2026-10-12):** score a feature by averages over regions at the resolution scale, never single
pixels. Gap valley: P(mean θ over the gap strip < min(mean θ over the two disk cores) − δ), δ = log(ε_r)/2. The gap
strip is the pixels between the disks within half the disk height, |y − c_y| ≤ r/2 (decision of 2026-10-13: the full-height
strip also counts the lens-shaped void beside the neck and can fire for a closed gap, so it is reported only as a lenient bound); if the strip is narrower than one pixel, **or if the
pixelized truth itself (the exact-coverage rendering on the inference grid) fails the feature**, report "unrepresentable on
this grid" instead of a probability (added after the review: the first version tested only the width, and reported
probabilities for strips that the grid cannot show even for the true object). Report single-point values alongside, labeled. MCMC
(Laplace-mass HMC, 4 chains) spot checks at two gaps (one resolvable, one not) and one contrast, to check the Laplace
probabilities.

### 13.4 Deliverables

results/resolution/: crb.json (F, CRB, required SNR, Born ratios, symmetry-block residuals, grid-convergence check),
laplace.json (per configuration: MAP, Φ_MAP, feature probabilities), MCMC spot-check summaries, and figures:
σ_CRB/g vs g per contrast; required SNR vs g with the book's operating points marked (Tests #5–8: gaps λ/10 to 2λ/5 at 10 and
20 dB; Test #4: λ/20 to λ/4, no noise level stated) and the points where the bound is not local shown open; the pixel-posterior P(gap valley) vs g for both priors, overlaid on the Smith criterion.

### 13.5 Execution and results (added after the runs; the report is `results/RESOLUTION_REPORT.md`)

CRB sweep (27 objects, grids 128 and 256, `scripts/resolution_crb.py`, `results/resolution/crb.json`), the extended Smith sweep (g = 1e-4 to 3e-3, `smith_limit/crb.json`),
72 Laplace configurations (`laplace.json`: 54 primary plus 18 TV runs at lambda*/3 and 3 lambda* at eps_r 2), a closed-gap negative control of the feature (`null.json`),
and MCMC spot checks at eps_r 2, g = 0.025 and 0.1 for both priors (`mcmc.json`). Headline: the information limit g* of Smith's criterion is 2.6e-4 to 3.2e-4 lambda at 30 dB
and the bound is local there; the gap that a pixel posterior shows at eps_r 2 and 4 (strip of half height, both priors) is 0.1 to 0.15 lambda (not below 0.5 at eps_r 1.1 for the
Matern prior) and is set by the bandwidth of the data (2 to 3 cycles per lambda) and by delta, not by the noise. The design's full-height strip is satisfied by a closed gap up to
g = 0.1 (`null.json`) and is not a detector; a strip of half height is recommended as the headline feature (a decision for the user).

### 13.6 Review follow-up (added after the review of the milestone-6 report)

The review reproduced every number it checked and corrected four interpretations; the corrections are in 13.1 (the book's operating points and the evanescent argument),
13.2 (two Born normalizations, the local regime, the detection limit), 13.3 (representability) and in the report. New code and files:

- `fisher.matched_power_ratios`, `nuisance_report` (sd of r, eps_r, cx, cy at SNR_req and the 0.1 local-regime rule), `detection_limit`; `analyze` stores `power_born` and the
  matched-power ratios; `resolution_crb.py --born-power --book-gaps` (`born_power.json`, the gaps 0.3 and 0.4 of the book's Tests #5 to #8), `BOOK` (the corrected operating points).
- `bayes_scatter/mle.py` and `scripts/resolution_mle.py`: Levenberg-Marquardt maximum likelihood of (g, r, eps_r, cx, cy) from noisy data of the same model at the required SNR
  (`mle.json`, `mle_check.png`, chunks in `mle/`): the threshold-aware check of the bound.
- `shapes.stretch_derivatives` (ellipticity nuisances s_x, s_y with the gap fixed) and `scripts/resolution_detect.py` (`detection.json`: sigma_CRB at g = 0, conditioning, the
  detectable gap, the ellipticity bound).
- `gapvalley.FeatureGeometry.truth_holds` / `representable` / `status`: a strip whose pixelized truth fails the feature is unrepresentable (the tables withhold the stored
  probabilities of old records, keeping them as `*_raw`).
- `resolution_mcmc.py`: criteria on the sampler's potential Phi' (`energy_convergence`, `--stage energy`), `--run-tag`, `--frozen-step`, `--tail-segments`, `--runs`.
  The frozen-step truth chain is NOT a fix: from the truth the bulk step is unstable (acceptance 0 in a probe) and the runs made no progress.
- `scripts/resolution_review_analysis.py` (`lowpass_check.json`, `snr_check.json`, `lambda_sensitivity.json`, `peff.json`, `book_geometry.json`, `verdict_range.json`,
  `mcmc_review.json`); `make_m6_review_jobs.py` and `scripts/jobs/m6_review.json` (the follow-up batch; the Matern g = 0.15 chains, a fourth TV chain and the 20 dB runs
  were dropped from the queue to stay in the time budget).


## 14. Milestone 7: parametric priors, right and wrong

### 14.1 Question

Milestone 6 (results/RESOLUTION_REPORT.md) found two limits. Inside the known two-disk model the data determine the gap down
to about 3×10⁻⁴λ at 30 dB. Model-free pixel posteriors resolve it only from about 0.1–0.15λ. Between the two, the prior does
the work. A network trained on one scene family (Ch. 15) acts like a parametric prior. This milestone asks: when the family is
**right**, does a shape-family posterior recover sub-λ/10 gaps with calibrated uncertainty? When it is **wrong**, does it still
report a confident gap, and do the data (misfit, posterior-predictive checks, evidence) flag the error?

### 14.2 Models (posteriors in shape space)

- **F2**: two equal circles (g, r, ε_r, c_x, c_y), the milestone-6 family. Priors: g = 0.4·sigmoid(u_g) (so 0 < g < 0.4λ),
  r ~ U[0.1, 0.3], ε_r − 1 log-normal (median 1, σ 0.5), centers ~ U[−0.2, 0.2]². Write every prior in a reparametrized
  unconstrained space and include the Jacobians.
- **F2u**: two circles with separate radii and permittivities (7 parameters).
- **Fe**: one ellipse (center, semi-axes, rotation, ε_r), the "one object" alternative.
- Shape derivatives: exact boundary derivatives (§13.2), extended to ellipses (normal speed of each parameter along
  the boundary). Gradient of the log posterior = (∂χ/∂p)ᵀ ∇_χ(−Φ) from one adjoint solve, plus the prior terms. Check against
  finite differences.
- Sampler: HMC in the unconstrained parameters with a dense mass from the Fisher plus prior Hessian at the MAP (as in §11.2),
  4 chains with overdispersed starts, convergence criteria as in §12.2. Multimodal targets (e.g. disks swapping roles) must be
  detected and reported, not averaged over silently.
- Evidence: Laplace log-evidence for each (truth, model), plus thermodynamic integration (≥ 12 temperatures) on at least two
  cases, to check the Laplace values.

### 14.3 Truths (ε_r = 2 unless stated, data from the n = 128 grid at 30 dB, inference on n = 64, side 1.5λ, book rings)

- T1 correct family: two equal disks, r = 0.2, g ∈ {0.05, 0.1} (λ/20 and λ/10).
- T2 ellipticity: both disks slightly elliptical (semi-axes 0.2 × 0.23, long axis along y), g ∈ {0.05, 0.1}.
- T3 unequal permittivities: ε_r = 2.0 and 2.4 (the book's Test #4 also used unequal contrasts), g ∈ {0.05, 0.1}.
- T4 closed gap with a neck: two touching disks joined by a bridge (rectangle 0.05 wide, full disk height), so the true gap is 0.
- T5 three objects: T1 at g = 0.1 plus a third disk of radius 0.06 centered 0.1λ above the gap.

For every truth fit F2, F2u and Fe. Also report the milestone-6 pixel posteriors (Matérn, TV at λ*; Laplace, half-height
strip) on the same data.

### 14.4 Report, per (truth, model)

- Posterior of g: mean, 90% credible interval, coverage of the true g, P(g < g_det(30 dB)).
- MAP misfit Φ_MAP against N and against Φ(truth). Posterior-predictive check: distribution of Φ for replicated data versus
  the observed Φ, as a p-value.
- Log evidence (Laplace; TI where run) and Bayes factors F2 vs Fe and F2 vs F2u.
- The pixel posterior P(gap valley) on the same data.

The questions to answer:
(i) Under the right family, are sub-λ/10 gaps recovered with calibrated intervals?
(ii) Under a wrong family, is the reported gap confidently wrong (interval excludes the truth), and by how much?
(iii) Do Φ, the predictive check or the evidence flag the misspecification at 30 dB? If not, that is the strongest statement
     this project can make about learned-prior "super-resolution": the data cannot tell whether it is real.


### 14.5 Implementation notes (added after the milestone-7 build)

**Modules.** `ellipses.py`: `EllipseRenderer` (exact pixel coverage of an ellipse (xc, yc, a, b, psi) and its exact boundary derivatives in JAX; `coverage` is a `jax.custom_jvp`, so grad, jvp and vmap use the boundary integrals),
the truths (`make_truth("T1".."T5")`: `EllipseScene`, `BridgedDisks`, `shapes.TwoDisks` for T1), `FeatureObject`. `shapemodels.py`: `F2`, `F2u`, `Fe` with their unconstrained parametrizations (g = 0.4 sigmoid(u_g), r and the
radii uniform on [0.1, 0.3], eps_r - 1 log-normal, centers uniform on [-0.2, 0.2]^2, Fe with ordered semi-axes b <= a on the triangle [0.05, 0.55]^2), the priors with their Jacobians and the symmetry images. `shapepost.py`: `ShapePosterior`
(gradient = one adjoint solve x the exact boundary derivative x the analytic dp/du), `GaussianToy` (validation with exact evidence) and the algorithms (MAP, Laplace, HMC in the Laplace map, multimodality, TI, predictive checks). The
posterior is truncated to |u_k| <= 10 (U = +inf outside, no solve): a trajectory that flies to eps_r ~ e^5 ends in a stagnating Krylov solve; the removed PRIOR mass is below 9e-5 per logistic coordinate, which bounds the change of
log Z (below 1e-3) only for a posterior that lies inside the box. T4 with F2 and F2u does not: its data want g -> 0, the profile marginal of u_g peaks near -12, and only 4.4e-4 of the posterior mass is inside the box, so the truncated target
is a different distribution (log Z lower by 7.7, `T4_F2/profile.json`, `truncation`) and the MAP, Laplace evidence and chains of those two cases describe it, not the posterior (14.7). Scripts: `parametric_run.py`,
`parametric_collect.py`, `make_m7_jobs.py` (job files for `batch_run.py`, one process and one XLA thread per chain), the three scripts added during the run (14.6): `parametric_profile.py`, `parametric_sweeps.py`, `parametric_ti_offline.py`, and the three of the
review follow-up (14.7): `parametric_snr.py`, `parametric_evidence_is.py`, `parametric_discretization.py`.

**Sampling and diagnostics.** `run_hmc`: eight leapfrog steps, step jitter +-25 %, step adaptation to acceptance 0.7 in 150 warmup steps from an overdispersed start (2 x N(0, I) in the Laplace coordinates), then frozen; 4 chains x 5 segments of 100
recorded draws, one npz per segment (resumable). `convergence_report`: classic and rank-normalized split R-hat of Phi, coordinate R-hat <= 1.05, bulk ESS >= 200 (DESIGN 12.2); the limit for Phi is ESS-aware (14.7): 1.01 above the split R-hat to expect from independent sampling,
`shapepost.expected_split_rhat` = 1 + (K / ESS - 1 / n) / 2 (K = 8 split sequences of n = 250 draws), and `converged_strict` keeps the fixed 1.01. `detect_multimodality`: distinct MAP modes from several
starts and the symmetry images, chain separation, bimodality coefficient. Evidence: Laplace in u (Gauss-Newton and finite-difference Hessians; lik_const = -N log(pi s^2), so values are comparable between models of one data set) and TI,
beta_k = (k/15)^5 with 16 temperatures (or the geometric schedule `ti_schedule(beta_min=...)`), pilot 40 + warmup 60 + 150 draws per temperature, trapezoid with the Euler-Maclaurin end correction applied only to intervals where Var_beta[Phi] changes by at most
a factor of 10 (14.7: elsewhere it is invalid, and the run is flagged `schedule_ok` False), Monte Carlo error from the ESS, Richardson estimate of the discretization error; importance sampling with a Student-t proposal at the Laplace fit
(`importance_log_evidence`) as the independent check of the Laplace value. Predictive check:
`refit` (best-fit misfit of replicated data, p-value uniform for a correct model: toy test, KS p = 0.10; with R replicates it cannot resolve p below 1/(R + 1)) and `draw` (classical, conservative); `misfit_p_value`: the analytic flag
P(Gamma(N - dim/2) >= Phi_MAP), no resolution limit. Validation: `tests/test_ellipses.py`, `test_shapemodels.py`, `test_shapepost.py`,
`test_parametric_scripts.py` (the Laplace evidence and the profile marginal are exact on a linear-Gaussian toy to 1e-6, TI agrees with the exact evidence within its error on the toy, a prior without its Jacobian is rejected by the prior-only HMC test).

**Cost (Mac, one XLA thread, `results/parametric/cost.json`).** Value and gradient of Phi at n = 64: 244 ms (F2), 247 ms (F2u), 255 ms (Fe) on an idle machine; 0.4 to 0.5 s with ten processes running (a chain of 650 steps took 43 to 50 minutes,
a TI temperature 17 to 24 minutes).


### 14.6 Execution and results (added after the runs and corrected after the review, see 14.7; the report is `results/PARAMETRIC_REPORT.md`, every number below is in a file under `results/parametric/`)

**Run.** Data: n = 128, 30 dB, noise key 0, cached in `data/<truth>.npz`; inference n = 64, 40 transmitters, 60 receivers, tolerance 1e-10; N = 2400; the grid error of the n = 64 model at the truth is 0.061 of the noise (T1, g = 0.05,
`T1g0.05_F2/problem.json`). Fitted: all 24 cases (8 truths x F2, F2u, Fe) to the MAP with the Laplace evidence (`<case>/map.json`; `tables.md` section 5); HMC (4 chains x 500) for T1 (both gaps) x {F2, Fe}, T2 g = 0.1 x {F2, Fe}, T4 x Fe;
T4 x F2 by the profile marginal (`T4_F2/profile.json`; the chains are degenerate, below); T3 g = 0.05 x F2u with 4 of 5 segments (400 draws per chain, `T3g0.05_F2u/chains_partial.json`); pixel posteriors (Matern, TV at lambda* = 0.5, Laplace, half-height strip) for all eight
truths (`<truth>_pixel/summary.json`); TI (two independent one-chain runs each) for T1 g = 0.05 x F2 and T2 g = 0.1 x F2, with the beta = 1 temperature taken from the production chains (`offline_ti.json`). Not run: HMC for T2 g = 0.05, T3 (F2, Fe),
T5 and the F2u/Fe chains of the other truths (their MAP, Laplace interval and Laplace evidence are in `tables.md` section 5); TI of a third case; the last TI temperature as a TI temperature (the machine slept and was shared; the runs were stopped at 15 of 16
temperatures, 15 completed).

**(i) Right family.** T1: g = 0.05022 +- 0.00028 (90 % interval [0.04977, 0.05068], z of the truth -0.8) and 0.10029 +- 0.00028 ([0.09984, 0.10076], z -1.0), converged (R-hat 1.002 and 1.003, bulk ESS 1456 and 1612, acceptance 0.72), unimodal,
refit predictive p = 0.21; the posterior sd is 2.8e-4, the milestone-6 bound is 2.6e-4 to 2.8e-4; the Laplace and profile marginals agree with the chains (0.050224 +- 0.000271 against 0.050224 +- 0.000275). **These two cases are one data realization**
(both use noise key 0; the correlation of z between the two gaps over the 41 keys is 0.985, `discretization/discretization.md`), so one interval that covers is weak evidence by itself; the statement rests on the calibration over noise keys below and on the agreement of
the Laplace Gaussian with the chains and with importance sampling (weight ESS 788 of 1000, `T1g0.05_F2/is_evidence.json`). T3, where the right family is F2u: g = 0.0501 (Laplace; 0.050071 +- 0.000276 from 4 x 400 draws), Phi/N 1.014.
**Calibration over 41 noise realizations** of T1 (F2 Laplace, `ellipticity/calibration_g*.md`): the width is calibrated (sd of z 1.00 +- 0.11 at g = 0.05, 0.98 +- 0.11 at 0.1) and the location is offset (mean z -1.15 +- 0.16 and -1.19 +- 0.16, so the 90 % interval covers
the truth in 66 % (se 5 %) and 68 % of the data sets; bias +3.1e-4 and +3.3e-4). **The offset is a deterministic forward-model (grid) bias, not noise** (`discretization/discretization.md`): (1) with the same noise keys the data generated on the inference grid
(`ellipticity_crime/`, 8 keys) differ from the n = 128 data by a constant shift of z, -1.019 +- 0.002 (the noise cancels in the pairing); (2) F2 fitted to NOISE-FREE T1 data at g = 0.05 gives g* - g = +0.005 sd for n = 64 data, +1.024 sd (+2.78e-4) for
n = 128 data and +1.237 sd for n = 256 data (successive shifts shrink by 0.21; extrapolated to the continuum 1.29 sd); +0.002 and +1.057 sd at g = 0.1; (3) with the offset removed the 41 noise-driven z have mean -0.13 +- 0.16 and sd 1.00 (0.98 at g = 0.1) and the shifted
interval covers 88 % (nominal 90 %). So the interval WIDTH is calibrated and the 1.0 sd location offset at 30 dB is a model error: the family absorbs it almost entirely (the noise-free misfit is 0.112 nats against a model-error energy of 8.85 nats at the true parameters), so it is
not flagged, and because it is fixed in wavelengths it is 0.10 sd at 10 dB, 0.32 sd at 20 dB, 1.02 at 30 dB and 3.2 sd at 40 dB. This is the concrete case of a model error that a family absorbs without a flag, and it is not benign for g. At the book's 10 to 20 dB the right-family fit
is unbiased at the level of the noise (T1 over 40 keys: mean z -0.3 at 10 dB and -0.5 at 20 dB, coverage of the truth by the 3 sd interval 100 %, `snr/snr_summary.md`).

**(ii) Wrong family.** F2 on T2 (15 % ellipticity): g = 0.07597 +- 0.00028 for a true 0.1 (z = 87; 0.02826 for 0.05, z 78 by Laplace). F2 on T3 (unequal contrasts): 0.04175 and 0.09339 for 0.05 and 0.1 (z 30 and 20 with the observed-information sd; 31 and 24 with the Gauss-Newton sd first reported; corrected 2026-11-08, see 15.5). F2 on T5 (a third small disk): 0.0691 for
0.1 (z 110). The intervals are tight (sd 2.8e-4 to 3.2e-4) and wrong by 20 to 110 sd (6 to 33 widths of the 90 % interval), by 0.0066 to 0.031 wavelengths. T4 (closed gap with a neck): F2 reports g ~ 0 (mean 2.3e-6, 90 % interval [1.2e-7, 7.0e-6], P(g < g_det) = 1), which is the true value
for the wrong reason (the profile misfit falls steeply to g -> 0, slope 4.3e5 per unit g, because the bridge is better fit by overlap). The ellipticity ladder shows how the bias grows (`ellipticity/ellipticity_g0.1.md`): g error -0.00056 (z 2.0) at 0.5 % ellipticity, -0.0014 (z 5.0) at 1 %, -0.0031 (z 11) at 2 %.

**(iii) Flags.** *Fit at 30 dB, noise key 0.* Phi_MAP/N = 1.015 for the right family and 8.1 to 8.4 (T2), 15.1 and 16.9 (T3), 4.0 (T4), 5.1 (T5) for F2 (F2u does not help T2, T4, T5: 8.1, 4.0, 5.1); the refit predictive p-value (100 replicates, resolution 1/101) is 0.000 for
every wrong case (z 355 for T2); the pixel posteriors fit every truth (Phi/N 1.000) and so cannot flag anything. *The scaling.* The misfit excess grows as the SQUARE of the error of the gap, dPhi = c z^2 (z in posterior sd, dPhi = Phi_MAP minus the misfit of the noise realization
itself), and c depends on the kind of model error and not on the SNR (the error is fixed in wavelengths; z scales as 1/s and dPhi as 1/s^2): c = 2.6 to 2.9 for the 15 % ellipticity of T2, 2.0 to 2.3 for the ladder up to 5 % (against the shift from the level-0 fit of the same noise; 2.5 to 2.7 at 15 %), 0.80 for the third disk (T5), 32 to 92 for unequal contrasts (T3; corrected 2026-11-08, 15.5; a second-order quantity specific to this contrast imbalance), 0.09 to 0.11
for the grid error (`snr/snr_summary.md`, `ellipticity/ellipticity_g*.md`, `discretization/discretization.md`; c changes by less than 15 % between 30, 20 and 10 dB). A misfit flag of 3 sd (dPhi = 3 sqrt(N - dim/2) = 147 nats) therefore needs z = sqrt(147 / c) at ANY SNR:
about 7.5 for ellipticity, 13.6 for a third small disk, 1.3 to 2.1 for unequal contrasts, 37 to 41 for the grid error (about 61 dB). *The ladder at 30 dB* (the analytic flag P(Gamma(N - 5/2) >= Phi_MAP), no resolution limit; the 30 replicates of the first version could not resolve p
below 0.03, and the realization of key 0 is +0.78 sd high): 0.5 % ellipticity (z 2.0): p = 0.12, not a flag (the refit p of 0.033 was one replicate in thirty); 1 % (z 5): p = 0.011; 2 % (z 11): p = 5e-11. Over 40 noise keys (`snr/snr_summary.md`) a 1 % ellipticity moves the gap interval
off the truth by more than 3 sd in 97.5 % of the draws while the misfit flags it at the 5 % level in 42.5 % and at 3 sd in 2.5 %; at 2 % the flag is certain. So the earlier sentence "the data flag a model error once it biases the gap by about 2 sigma, never later" was wrong: the misfit lags the
gap error by a factor up to the square of the ratio of thresholds. *At the book's SNRs* (40 noise keys per cell, Laplace fits, analytic null; `snr/snr_summary.md`): at 20 dB every wrong case is flagged at 3 sd in every draw (T5: z of the truth 34.6, dPhi 977 = 20 sd; T2 g = 0.1: z 26, dPhi 1748). **At 10 dB the F2 interval on T5 excludes the true gap by 10.8 sd (in every draw) while
the misfit excess is 97 nats = 2.0 sd: the 5 % flag fires in 70 % of the draws (28 of 40) and the 3 sd flag in 17.5 % (7 of 40).** T2 (z 7.6 to 8.2) is flagged at 5 % in every draw and at 3 sd in 75 %; T3 (z 2.1 to 2.8, so the interval misses the truth by 3 sd in only 15 to 45 % of the draws; observed-information sd, corrected 2026-11-08) is flagged in every draw; T4
(dPhi 70 nats, 12 keys): 50 % at 5 %, none at 3 sd. The refit predictive check agrees for the two cells run (T5 10 dB: p = 0.000 of 20 replicates, z 3.2; T2 g = 0.1 10 dB: z 5.4). The evidence separates right from wrong only when a better family exists: F2 against F2u log BF = +9.8 to +10.4
on every truth where F2u does not fit better (Occam factor, IS and Laplace agree), -33672 and -38050 on T3 where it does; between two wrong families the Bayes factor says nothing (F2 against Fe +10971 on T4, 1e5 on the others, each family misfitting). **So the strongest statement of (iii)
is reached at the book's 10 dB for a deformation like the third disk (and for ellipticity below about 1.5 % at 30 dB): the interval is confidently wrong by 11 sd and the data raise no flag in 30 % (5 % level) to 82.5 % (3 sd) of the draws. It is not reached at 20 dB, nor at 30 dB for 2 % ellipticity and above or for the third disk (z 110 against its 3 sd threshold of 13.6).** The caveat remains that the noise level is exactly known and that the forward operator is exact up to the grid; the grid error itself is such an unflagged model error (it needs about 60 dB to be seen).

**Evidence check.** Importance sampling with a Student-t proposal (5 degrees of freedom, shape 1.2 x the Laplace covariance, 1000 draws; `parametric_evidence_is.py`, `<case>/is_evidence.json`, `tables.md` section 7) reproduces the Laplace log evidence of all 15 cases with an interior posterior (F2 and F2u on seven truths, Fe on T4): IS minus Laplace (finite-difference Hessian) is within 0.03 nats everywhere and IS minus Laplace (Gauss-Newton Hessian) within 0.09 except T3 g = 0.1 F2 (0.28; Phi/N 16.9, where the Gauss-Newton Hessian is not the exact one); the IS standard error is at most 0.022 and the weight ESS 667 to 789 of 1000 (e.g. T1 g = 0.05 F2: 35397.847 +- 0.016 against 35397.841). The Occam factor of F2 against F2u is therefore verified: by IS +9.82 to +10.41 (T1 g = 0.05 10.38, g = 0.1 10.41; T2 9.82, 9.92; T5 10.16), by Laplace +9.81 to +10.39, and on T3 -33672.30 and -38049.35 (IS) against -33672.39 and -38049.64. **Thermodynamic integration** (`offline_ti.json`): the first version applied the Euler-Maclaurin end correction to every interval and gave 35424.4 +- 12.8 and 17367.8 +- 7.3, i.e. 26.6 and 15.4 nats ABOVE Laplace (`offline_ti_v1.json`). That was an artifact: in the first four intervals of the power schedule Var_beta[Phi] changes by 15 to 230 times between neighbors (1.9e11 to 8.5e8 in one interval), where a one-term expansion is invalid, and the correction added 30 to 64 nats there. With the guard (`ti_combine`: correct only where the variance changes by at most 10-fold) the TI values are 35375.0 +- 3.1 (between runs 4.5) and 17334.5 +- 2.1 (2.1), 22.8 and 17.8 nats BELOW the IS values (35397.85 and 17352.29), `schedule_ok` False in all four runs (intervals 0 to 3, or 1 to 3, without correction): trapezoid lower bounds, which is what a coarse schedule gives for a concave integrand (trapezoid 35373.5 and 17332.9). The TI check is therefore not a measurement of the evidence at this schedule and budget; the Laplace values were right, and IS is the check. A TI run on a geometric schedule (`parametric_run.py --stage ti --ti-beta-min 1e-7`) was not made (about 5 hours of single-thread time).

**Failures and corrections.** (1) T4, F2 and F2u: the MAP lies on the |u_g| = 10 wall (g = 1.8e-5, MAP not converged), the Laplace sd of u_g is 14, and HMC is degenerate: two of four chains started outside the box and never moved, a third had a step of 5e-4
(kept in `T4_F2/chains_degenerate/`). The gap marginal was computed instead by profiling the other parameters (`parametric_profile.py`, validated against the T1 chains and exact on a toy). The posterior mass inside the box is 4.4e-4, so the truncated target (not only its Laplace value)
differs from the posterior by 7.7 nats in log Z (`T4_F2/profile.json`, `truncation`); the Laplace evidence of T4 F2 (28060.7) is unreliable; the profile value is 28062.8. (2) Three cases were flagged unconverged by the R-hat of Phi alone (classic or rank-normalized 1.0107 to 1.0133 against 1.01; all classic coordinate R-hats below 1.005): T1 g = 0.1 Fe,
T2 g = 0.1 F2 and Fe. With 4 x 500 draws and a bulk ESS of Phi of 229 to 414 the R-hat to expect from perfect mixing is 1.0077 to 1.0154 (`shapepost.expected_split_rhat`), so the limit is now ESS-aware (14.7) and all seven chain sets converge; the strict limit 1.01 still flags those three (`tables.md`
section 1 shows both, with the classic and rank-normalized maxima). (3) The jobs were restructured four times (placeholder jobs that hold slots until running processes exit, so that the XLA threads stayed at ten); no finished work was lost.

### 14.7 Review follow-up (added after the review of the milestone-7 report)

The review reproduced the code and the numbers and corrected three interpretations; this section records what changed (the finding-by-finding log is in `results/PARAMETRIC_REPORT.md`).

- **(iii) is settled by a scaling law, not by a threshold of 2 sigma** (14.6). New: `shapepost.misfit_p_value` (the analytic null Gamma(N - dim/2); the replicate-based p-value has a resolution of 1/(R + 1) and the key-0 realization is +0.78 sd high), `scripts/parametric_snr.py` (F2 and F2u fits
  at 10, 20 and 30 dB over 40 noise keys, `snr/`; the noise generator checked against Gamma(N, 1) over 1000 keys, `snr/noise_null_check.json`: mean 2400.5, sd 47.8, KS p 0.89; keys 0 to 39 have sd 41.3, which is why the 41-seed empirical null was narrow), the ladder
  with p, dPhi and c (`ellipticity/ellipticity_g*.md`).
- **(i) the offset is a grid bias** (`scripts/parametric_discretization.py`, `discretization/`).
- **Evidence.** `shapepost.importance_log_evidence` and `scripts/parametric_evidence_is.py` (`<case>/is_evidence.json`); `ti_combine` applies the Euler-Maclaurin end correction only where Var_beta[Phi] changes by at most 10-fold between neighbors and reports `schedule_ok`; `ti_schedule(beta_min=...)`
  and `parametric_run.py --ti-beta-min` give a geometric schedule (ratio 3.2 for 16 temperatures from 1e-7); the offline TI is recomputed (`offline_ti.json`, the first version `offline_ti_v1.json`).
- **Convergence.** `shapepost.expected_split_rhat`; `convergence_report` has `rhat_phi_expected`, `rhat_phi_limit` (the expected value at max(ESS, 200) plus 0.01: an allowance that grew as the ESS of unmixed chains collapses would excuse them), `converged` (ESS-aware) and `converged_strict`; `tables.md` quotes the classic and rank maxima and the expected R-hat; the seven summaries were regenerated (`parametric_run.py --stage finalize`; every
  other number in them is unchanged) and the partial chains of T3 g = 0.05 F2u have their own report (`chains_partial.json`).
- **Truncation.** The statement that the box removes less than 9e-5 of the mass is about the prior and holds for interior posteriors only (`shapepost._U_and_grad_fn`, 14.5); `profile.json` reports the mass inside the box (T4: 4.4e-4).
- **Profile marginal.** `parametric_profile.py` interpolates the marginal in u_g (cubic spline of log m on a 32 times finer grid) instead of the CDF linearly in g, which put the 95 % point of the T4 gap 12 % too high (7.89e-6 against 7.03e-6; the same coarse grid with the new rule gives 7.03e-6 and the 145-point grid 7.03e-6);
  the run is resumable per grid point.
- **Fe orientation** (`shapemodels.Fe` docstring): the coordinate of psi splits a posterior near +-pi/2; every Fe fit has |psi| <= 0.0013 (`tables.md` section 9).
- **Tests.** New tests for each code defect: the TI end correction on a sharp posterior (10.8 nats too high without the guard, 4.2 too low and flagged with it), the geometric schedule, importance sampling on the Gaussian toy and a heavy-tailed target, the analytic null by simulation, the noise generator, the expected R-hat against simulated AR(1) chains and the
  ESS-aware criterion, the quantiles of the profile marginal against an exponential law, the profile checkpoint, the SNR units (cached noise reproduced, scaled copies, resumable), the importance-sampling script, the discretization fit and analysis, the ladder flags, the offline TI guard, the new `tables.md` sections and the job files.

## 15. Milestone 8: the geometry of detectability

### 15.1 The law and its first-order derivation

Milestone 7 (results/PARAMETRIC_REPORT.md §0 (iii)) measured ΔΦ = c z² for a misspecified shape family. Here ΔΦ is the
excess of the MAP misfit over that of a correct model, z is the error of the gap estimate in posterior standard
deviations, and c depends only on the kind of model error, not on the SNR. Derivation at first order:

- **Whitened coordinates.** With CN(0, s²) noise, real and imaginary parts each have variance s²/2, so in whitened
  real-stacked data y_w = y_r/(s/√2) the misfit is Φ = ½‖r_w‖². The likelihood is exp(−Φ), and the Fisher matrix of a
  model with whitened Jacobian D (2N × k, real) is DᵀD.
- **Model error.** Let δ (whitened) be the difference between the true data and the model's data at the base point p₀
  (noise-free). To first order in δ, the least-squares bias is Δp = (DᵀD)⁻¹Dᵀδ, and the misfit excess is
  ΔΦ = ½‖P_⊥δ‖², where P_⊥ = I − D(DᵀD)⁻¹Dᵀ projects off the tangent space T = range(D) of the model manifold.
- **The gap direction.** With e_g the gap's unit vector in parameter space, Δg = vᵀδ for v = D(DᵀD)⁻¹e_g, and
  ‖v‖² = [(DᵀD)⁻¹]_gg = σ_g² (the Cramér–Rao variance). So z = Δg/σ_g = ŵᵀδ, where ŵ = v/‖v‖ is the unit vector in T
  orthogonal to the tangent directions of every other parameter. It is the "pure gap" direction of the model manifold in
  data space.
- **Which gap.** Δg = (fitted − base). It is the error of the estimate only if the base gap is the true gap. A model error
  that also changes the object's own gap (every A1 boundary deformation with m ≥ 2 changes the edge-to-edge gap by
  −2(−1)ᵐε at first order) has z_true = z_fit − (gap change)/σ_g; equivalently δ is replaced by its gap-preserving
  version, which differs by a multiple of the in-model tangent d_r − 2 d_g and so has the same ΔΦ. Any statement about
  "bias" or "detectability" uses z_true; z_fit is the response of the estimate.
- **Result.** If δ makes an angle α with T, and its tangential part makes an angle β with ŵ, then
  ‖P_⊥δ‖ = ‖δ‖ sin α and ŵᵀδ = ‖δ‖ cos α cos β, so

  **c = ΔΦ/z² = ½ tan²α / cos²β,**

  independent of ‖δ‖ for a fixed direction of δ (all A1 errors of a family that scale together) and of the SNR, because
  both scale with the noise level and cancel. An error whose data signature lies almost in the model's tangent space
  (α → 0) and along the pure gap direction (β → 0) biases g without raising the misfit. Those are the confounded
  directions. If z = O(‖δ‖²) (no A1 component, see 15.2), ΔΦ = O(‖δ‖²) too and c scales as 1/‖δ‖²: it is then a
  second-order quantity that this formula cannot predict.
- **Conditions.**
  - *Exactness.* δ is the exact difference between the true data and the model at p₀, so the nonlinearity of the truth is
    irrelevant. The formula is exact iff the model map μ is affine between p₀ and the fit p*. The corrections are the
    intrinsic curvature times ‖P_⊥δ‖ and the parameter-effects curvature times ‖p₁‖; both are independent of the SNR, so c is
    SNR independent also nonlinearly for noise-free fits with a negligible prior. Second order: p = p₁ + p₂ with
    p₂ = (DᵀD)⁻¹(−½DᵀH[p₁,p₁] + H[p₁,·]ᵀP_⊥δ), H = d²μ (`detectability.second_order_bias`); the series fails when the
    residual-curvature operator below has an eigenvalue near 1 (milestone 8: T3).
  - *Base point.* First-order z and ΔΦ do not change if p₀ moves along nuisance directions (ŵ and P_⊥ annihilate those
    columns), so the base-point dependence is itself a second-order curvature diagnostic. The base point of the analysis is
    the profile foot point (g = g_true, the nuisances at their least-squares values; `detectability.foot_point`): there
    Dᵀδ has only a gap component, the tangential part of δ is exactly along ŵ (β = 0 or π) and c = ½ tan²α. "D evaluated at
    the noise-free MAP with δ measured from p₀" has no first-order meaning (at the exact MAP the tangential residual is
    zero) and is not used; the correction for a distant base point is the second-order one.
  - *Curvature of the residual.* At a fit with residual r = y − μ(p*), the Hessian of Φ is DᵀD − ⟨r, d²μ⟩ = DᵀD(I − A),
    A = (DᵀD)⁻¹⟨r, d²μ⟩. The Gauss–Newton sd (Fisher) is then wrong by the factors 1/√(1 − λ) in the directions of
    the eigenvectors of A; z and c must use the observed-information sd when Φ/N ≫ 1 (milestone 7's T3: the sd of g is 16 %
    larger at g = 0.1). The Laplace log evidence changes by −½ Σ log(1 − λ).
  - *Noise.* DESIGN's ΔΦ is the expected excess over a correct model. Milestone 7 measured Φ_MAP − Φ(noise), which is lower by
    ½‖P_T e‖² (mean k/2 = 2.5). One noise draw also has z-noise of sd 1 and ΔΦ-noise of sd √(2ΔΦ); the interaction terms are
    z = z_NL + ŵ*ᵀe and ΔΦ = Φ_NL + r*ᵀe − ½‖P_T* e‖² (e the whitened noise). The reference for the law is therefore the
    noise-free nonlinear fit, not a single draw.
  - Nuisance parameters free (projected out through (DᵀD)⁻¹).

### 15.2 Symmetry selection rule

The book's rings are invariant under σ_x: x → −x and σ_y: y → −y (milestone 6 checked the angle sets), so the
forward map and its linearization at a D₂-symmetric object intertwine the D₂ actions on shape perturbations and on data
(antenna permutations). The data space splits into four isotypic components (A₁, B₁, B₂, A₂), and so does the space of
model errors. The gap g, the radius r and ε_r are A₁; c_x and r₁ − r₂ change sign under σ_x; c_y changes sign under σ_y
(the pair rotation, A₂ at m = 1, is outside F2). The isotypic components of the real D₂ irreps are one dimensional and
mutually orthogonal, and an intertwiner preserves them, so ŵ (built from A₁ columns, and for the nuisances c_x, c_y from
columns orthogonal to A₁) lies in A₁ and **at first order only the A₁ part of a model error can bias g**. Errors in the
other irreps give z = O(‖δ‖²) and therefore a large c ∝ 1/‖δ‖². This predicts the large c of T3 (unequal contrasts: the
A₁ part, the mean contrast, is absorbed by ε_r, and what is left is σ_x-odd) and the small c of T5 (the third disk lies on
the σ_x axis, so its A₁ part is nonzero).

Conditions: a D₂-invariant base point (a symmetric object p₀ with c_x = c_y = 0 and equal disks), antenna sets invariant
under both reflections (exact permutations, `D2Action` raises otherwise), iid noise (the whitened metric is the identity, so
the permutations are orthogonal), and exact grid flips for the contrast. The rule holds at every base point that
satisfies them (milestone 8 checked the nominal ε_r = 2 base and the mean-contrast base of T3): "z is second order" for T3
holds at both, and c_T3 is a second-order, curvature-driven number (15.5).

### 15.3 Computations

1. **Verify the formula.** From noise-free data, compute α, β, c_geom for T2 (ellipticity 0.5%, 1%, 2%, 5%, 15%), T3,
   T5 and the grid error (n = 128 and n = 256 data against the n = 64 model). The reference is the noise-free nonlinear fit
   c_NL; the milestone-7 values (results/parametric/snr, ellipticity, discretization) are ONE noise draw each and are
   compared through the noise decomposition of 15.1. Report where the first-order formula fails and why (second-order
   `second_order_bias`, residual-curvature eigenvalues, observed-information sd).
2. **Selection rule check.** For every truth, decompose δ into the four D₂ isotypic components (average over the group
   action on data) and report the fraction of ‖δ‖² and of z carried by each. Non-A₁ components must give |z| at
   round-off in the linear computation.
3. **Detectability spectrum.** Boundary deformations of the two disks, normal displacement v_n(φ) = ε·cos(mφ) or
   ε·sin(mφ) in each disk's polar angle, combined into D₂ irreps. A₁ modes are cos(mφ) on the right disk plus its σ_x
   image cos(m(π − φ)) = (−1)^m cos(mφ) on the left. For m = 2…8 at ε_r = 1.1, 2, 4 and g = 0.05, 0.1, on a grid whose
   harmonics are resolved (n = 192; the spectrum is shown against n = 64 and 128), report in the gap-error convention
   (z_true, the gap-preserving tangent τ_m − (−1)^m τ₀, 15.1):
   - dz_true/dε (gap error per unit deformation, in posterior sd at 30 dB; the response of the F2 estimate dz_fit/dε is
     reported separately and labeled as such),
   - c_m,
   - the deformation amplitude at which z_true = 3 while ΔΦ stays below the 3σ flag (147 nats), in wavelengths,
   - the small-deformation measures ε m/r (boundary slope) and x = ε(ε_r − 1)/h (contrast displacement per boundary pixel),
   - ‖τ_m‖/‖τ₂‖ against the far-field band-limit prediction J_m(2k₀r)/J₂(2k₀r) (2k₀r = 2.51).

   m = 0 and m = 1 lie inside the model (r and g), so they serve as checks: α = 0 and z_true = 0. Use the existing exact
   boundary derivatives (shapes.py, ellipses.py) with v_n as the displacement field, and jvp for the data tangents.
4. Report as results/DETECTABILITY_REPORT.md.

### 15.4 Implementation notes (milestone 8)

- `src/bayes_scatter/detectability.py`: whitened Jacobian, `analyze_error` (z, ΔΦ, α, β, c), `D2Action`, the boundary modes, and (after the review)
  `gap_change_per_amplitude`/`true_gap_tangent` (gap-error convention), `curvature_tensor`/`second_order_bias`/`residual_curvature_operator`/`log_evidence_gap`
  (second order, steps 1e-3 max(|p|, 0.1), central differences of the exact jvp Jacobian) and `foot_point` (profile foot point). `shapes.circle_mode_integrals`
  gives the exact arc integrals of cos(mφ) and sin(mφ).
- `scripts/detectability_verify.py` (15.3 items 1 and 2) and `scripts/detectability_spectrum.py` (item 3, `--n` selects the grid; n = 64, 128, 192 are in
  `results/detectability/spectrum/n<N>/`). Everything is resumable and the numbers are in `results/detectability/`; the report is
  `results/DETECTABILITY_REPORT.md`.
- Cost: one n = 192 base object takes 5 to 10 min on one core (the 6 bases of one grid ran in parallel on the workstation, one process per core); one verify case takes 1.5 to
  4 min on the Mac.

### 15.5 Review follow-up (added after the review of the milestone-8 draft)

1. **The spectrum measured the wrong gap (critical).** z = ŵᵀτ is the response of the F2 estimate to the BASE gap, but every A1 mode with m ≥ 2 changes the deformed object's
   own gap by −2(−1)ᵐε. The error of the estimate is z_true = z_fit + 2(−1)ᵐ/σ_g per unit amplitude (15.1 "Which gap"); m = 0 then has z_true = 0 as it must. In this
   convention all 42 A1 cases are unflagged at z_true = 3 (c ≤ 4.5, ε₃ = 2.3e-4 to 9.6e-4 wavelengths at n = 192), the confounded modes are the high-m gap-preserving ones that lie
   beyond the far-field band limit (m > 2k₀r = 2.5; ‖τ_m‖/‖τ₂‖ follows J_m/J₂), and T2 (a gap-preserving m = 2 deformation to first order, the m = 0 mode at +er/2 plus the
   m = 2 mode at −er/2) is reproduced: c_true(m = 2) = 1.892 at ε_r 2, g 0.1, n = 192 against 1.871 for the independent ellipse renderer (`test_ellipticity_is_the_gap_preserving_m2_mode...`).
   The first version's conclusions ("low harmonics are the most damaging", "many are flagged first") were artifacts of the base-gap convention.
2. **n = 64 is not converged for m ≥ 5.** The pixelization floor of ‖τ_m‖ is ≈ 1 % of ‖τ₂‖; the base-gap response dz_fit changes by factors 5 to 300 between n = 64, 128 and 192
   for m ≥ 5. The true-gap quantities converge (|dz_true/dε| within 0.5 % between n = 128 and 192, ≤ 5 % between 64 and 192) because they are dominated by 2/σ_g. The spectrum is
   reported at n = 192 with the convergence table; the m = 6 and 7 "extremes" of the first version are removed.
3. **The T3 sd (major).** The Gauss-Newton sd understates the posterior sd of g when Φ/N ≫ 1: at the milestone-7 T3 MAPs the observed-information (finite-difference Hessian) sd is 1.159
   times larger at g = 0.1 (3.243e-4 against 2.799e-4) and 1.037 times larger at g = 0.05; z = 23.64 → 20.40 and 30.67 → 29.58, c = 68.1 → 91.5 and 35.8 → 38.5. The largest eigenvalue of
   (DᵀD)⁻¹⟨r, d²μ⟩ at the T3 g = 0.1 MAP is 0.443, which predicts log Z(FD) − log Z(GN) = 0.270 against 0.252 in `results/parametric/T3g0.1_F2/is_evidence.json` (importance sampling agrees with
   the FD Laplace value to 0.03). The milestone-7 report (results/PARAMETRIC_REPORT.md sections 0(iii), 3, 4.2, 4.4) and 14.6 above were corrected on 2026-11-08 at the user's request, for every T3 unit of the SNR experiment (`scripts/parametric_t3_fd_sd.py`, `results/parametric/snr_fd/t3_fd_summary.md`; the stored MAPs, nothing refit): at 30 / 20 / 10 dB, key 0, z = 29.6 / 9.5 / 3.2 and c = 38.5 / 36.7 / 32.4 (g = 0.05), z = 20.4 / 6.6 / 2.2 and c = 91.5 / 86.5 / 76.6 (g = 0.1); at 10 dB over 40 keys the interval excludes the truth at 3 sd in 18 and 6 of 40 (were 21 and 8).
4. **T3 is second order at both base points (major).** At the nominal base 95.7 % of the A1 energy lies along the ε_r tangent (cos 0.978) and the first-order z comes from the non-tangent
   remainder; c_T3 = ΔΦ/z² with z = O(Δ²) is a curvature quantity that the first-order c_geom cannot predict, so the agreement of c_geom (40, 54) with the measured c (35.8, 68) is
   a cancellation of a +34 % overshoot in Δg and a +15 % overshoot in σ_g(p₀). The selection rule holds at both bases; c_T3 needs the second-order theory (15.1) and scales as 1/Δ². At the symmetric bases the non-A₁ part of z is below 6e-15 of z; at the profile foot points of T3 (g = 0.05: c_x = 0.0042, where the B₁ error is absorbed by c_x) and T5 (c_y = 0.0050) the base is not D₂-symmetric and 23 % and 2 % of z are carried by non-A₁ components, which is what the condition of 15.2 says.
5. **Attributions.** (a) The nonlinear-refit failures of the first version are explained by x = ε₃(ε_r − 1)/h (all 20 agreeing cases have x ≤ 0.31, all 15 failures x ≥ 0.41), not by a small
   projection; with the true-gap amplitudes every case has x ≤ 0.12 and the refit agrees (shift 2.89 to 3.04, misfit within 3 % of 9c wherever 9c > 0.03). (b) The 15 % ellipticity
   discrepancy is parameter-effects curvature (p₂ tangential +0.00359, normal −0.00128 on Δg = −0.02733 against NL −0.02399), not a 10 to 15 % change of σ_g (it is 6.6 to 6.8 % for T2).
   (c) The measured c of milestone 7 is one noise draw; the grid-error rows are noise-free and the agreement of the g = 0.05 rows is not the same as that of the g = 0.1 rows (15.3 item 1).

## 16. Warm-started adjoint solve

### 16.1 What was done

The forward solve of every gradient is warm-started from the previous total fields `u`; the adjoint solve (`A^T w = ∂Φ/∂u`, one system per transmitter, same operator, tolerance relative to its own right-hand side) used to start from zero. Now `aux` may be a pair `(u, w)` (`forward.py`: `Forward.initial_aux(warm_adjoint=True)`, `value_and_grad_aux`, `split_aux`, `aux_fields`, `with_fields`), and the adjoint solve starts from `w` of the previous gradient.

- **Backward compatible.** An array `aux` (or None) takes the old `custom_linear_solve` path, returns an array, and keeps forward mode (`jvp`, `linearize`). Only a pair takes the new path. `Target`, `ThetaTarget` and `ShapePosterior` take `warm_adjoint=False` (default off); with `True`, `initial_aux()` returns `(u_inc, 0)` and everything that carries `aux` (`samplers`, `adapt`, `shapepost.run_hmc`, `lax.scan`, `lax.cond`, accept/reject) carries the pair unchanged, because structure, shapes and dtypes of the pair are stable. `inference.run_chains` broadcasts the pair leaf by leaf. Value-only calls (`potential`, `misfit_of_chi`) refresh `u` and pass `w` through.
- **How `w` leaves `jax.grad`.** A warm solve is a `custom_vjp` (`forward._solve_linear_warm`): the backward rule solves `A^T w = cotangent` from `w0`, returns `-(∂(A(χ)u)/∂χ)^T w` for `χ`, `w` for the right-hand side, and `w` again as the "cotangent" of the start `w0`. `value_and_grad_aux` differentiates with respect to `(x, w0)` and returns the second gradient as the new adjoint field. A non-finite `w0` is replaced by zeros. The pair path supports reverse mode only.
- **Exactness.** Same system, same relative tolerance, so the gradient changes only within solver tolerance: largest relative difference warm versus cold over all timed gradients 3.5e-11 at tol 1e-10 (`results/adjoint_warm/*.json`, `grad_rel_diff`); the misfit was bit-identical at every timed point (the forward solves are the same).
- **Tests** (`tests/test_warm_adjoint.py`): gradient equality along a 5-point trajectory for GMRES stack, GMRES vmap and BiCGSTAB (against the tol 1e-13 reference too); the exported `w` solves `A^T w = ct`; adjoint operator applications counted with `Forward.adjoint_matvec_counts` (warm < cold; from the exact solution 1 to 2); non-finite start; pair stable in `scan` and `cond`; HMC with the pair reproduces the cold chain; `ThetaTarget` and `ShapePosterior` (including the `beta = 0` branch).

### 16.2 Measurement (`scripts/measure_adjoint_warm.py`, `scripts/adjoint_warm_table.py`; JSON in `results/adjoint_warm/`)

Problem: T1 truth (g = 0.05, r = 0.2, ε_r = 2), data from the 2n grid at 30 dB, F2 posterior, book rings (40 × 60), tol 1e-10, whitened coordinates of the Laplace map at the truth, step 1.3 (jitter 0.25), 8 leapfrog steps, real accept/reject (acceptance 0.75 to 0.86). Cold and warm chains are evaluated at the same points, back to back in alternating order. Mac: one process, NPROC 4 (the machine was loaded by other work; indicative). tr CPU: 16 single-thread jobs, one per L3 group (cores 0, 4, ..., 60), seeds 0 to 15, 512 gradients per grid; tr GPU: one process, 48 gradients per grid.

| platform | grid | restart | cold s/grad | warm s/grad | warm/cold | adjoint operator applications per transmitter, cold / warm |
|---|---|---|---|---|---|---|
| Mac CPU, NPROC 4 | 48:2 | 100 | 0.0629 | 0.0612 | 0.973 | 14.0 / 13.0 |
| Mac CPU, NPROC 4 | 64:1.5 | 100 | 0.1199 | 0.1165 | 0.972 | 14.0 / 13.0 |
| tr, one core (16 concurrent) | 48:2 | 100 | 0.8675 | 0.8634 | 0.995 | 14.0 / 13.2 |
| tr, one core (16 concurrent) | 64:1.5 | 100 | 2.0578 | 2.0418 | 0.992 | 14.0 / 13.0 |
| tr, one core (16 concurrent) | 48:2 | 20 | 0.3862 | 0.3809 | 0.986 | 14.0 / 13.2 |
| tr, one core (16 concurrent) | 64:1.5 | 20 | 1.1763 | 1.1582 | 0.985 | 14.0 / 13.0 |
| tr GPU, one process | 48:2 | 100 | 0.0181 | 0.0173 | 0.953 | 14.0 / 13.0 |
| tr GPU, one process | 64:1.5 | 100 | 0.0266 | 0.0255 | 0.959 | 14.0 / 13.0 |
| tr GPU, one process | 48:2 | 20 | 0.0162 | 0.0154 | 0.951 | 14.0 / 13.0 |
| tr GPU, one process | 64:1.5 | 20 | 0.0237 | 0.0226 | 0.954 | 14.0 / 13.0 |

The count (residual evaluations included) is 14 cold and 13 warm for nearly every transmitter: the warm start saves one Arnoldi step of about twelve. Reason: at the posterior the adjoint right-hand side is the data residual, which is noise-dominated (Φ ≈ N), so a leapfrog step of 1.3 posterior sd changes it by a few percent, which is one decimal digit of a solve that has to reduce the residual by 1e-10 at a rate of about one digit per step. The saving cannot exceed that without a larger step or a looser tolerance. The wall-time gain is 1 to 2 % on one CPU core (about 1 step of the 28 of forward plus adjoint, diluted by costs that do not depend on the iteration count), 3 % on the Mac and 4.5 to 5 % on the GPU. A warm start is not always better: on the tiny 12:1.5 grid of the test, where consecutive points are far apart in data space, the warm count averaged 13.2 against 13.0.

### 16.3 Recommendation

Keep `warm_adjoint` as an opt-in flag (default False). It is exact and free to switch on for GPU chains (about 5 %), but it does not change the cost picture: the gradient costs the same to within 1 to 5 %. The cost per solve depends far more on the restart length than on the iteration count: with the same 14 operator applications, one CPU core takes 0.868 s (restart 100) against 0.386 s (restart 20) per gradient at 48:2, and 2.06 s against 1.18 s at 64:1.5, so the larger lever is the size of the Krylov basis (restart, and the concurrency choices of the performance study), not the iteration count. (Inference from these timings: the basis of (restart + 1) x 40 x n^2 complex numbers is allocated and read at every solve; I did not profile the allocation separately.)
