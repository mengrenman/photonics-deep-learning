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
                  milestone5_tables.py
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

Ch. 15 calls a reconstruction "super-resolved" when the image shows features below 0.25λ (p. 464). It supports this
with single reconstructions of gaps from λ/4 down to λ/20 at 10–20 dB. Milestone 5 found that every pixel posterior
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
  contrast, measures whether multiple scattering adds information about the gap.
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
strip is the pixels between the disks within the disk-height band; if the strip is narrower than one pixel, report
"unrepresentable on this grid" instead of a probability. Report single-point values alongside, labeled. MCMC
(Laplace-mass HMC, 4 chains) spot checks at two gaps (one resolvable, one not) and one contrast, to check the Laplace
probabilities.

### 13.4 Deliverables

results/resolution/: crb.json (F, CRB, required SNR, Born ratios, symmetry-block residuals, grid-convergence check),
laplace.json (per configuration: MAP, Φ_MAP, feature probabilities), MCMC spot-check summaries, and figures:
σ_CRB/g vs g per contrast; required SNR vs g with the book's claimed operating points (gaps λ/20 to λ/4 at
10–20 dB) marked; the pixel-posterior P(gap valley) vs g for both priors, overlaid on the Smith criterion.
