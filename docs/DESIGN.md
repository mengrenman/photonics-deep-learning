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
scripts/          validate_forward.py, born_exact_check.py, run_posterior.py (+ analyze_run.py, make_tables.py,
                  measure_cost.py, plot_gn_spectrum.py); prototypes outside the ladder: run_posterior_laplace.py
                  (Laplace-whitened sampling), map_multistart.py (MAP from several starts);
                  milestone 4: run_prior_comparison.py (+ make_prior_jobs, compare_priors, edge_analysis,
                  convergence_audit, prior_energy_audit, prior_feature_baseline, prior_comparison_tables,
                  gmrf_diagnostic, check_data_threads, measure_theta_cost) and span_test.py (+ span_test_tables); _threadcap.py is
                  imported first by the standalone scripts (BLAS threads <= 2)
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

