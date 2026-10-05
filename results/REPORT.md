# Bayesian inverse scattering testbed: experiment report

Corrected after an independent review and a numbers-vs-outputs audit (see the logs at the end).

All numbers come from results/ (tables in results/tables, figures in results/figures, per-run summary.json / metrics.json / chains.npz, commands in scripts/reproduce_report.sh). Problem: 2-D TM, book rings (40 tx, 60 rx, N = 2400 complex data), power SNR 30 dB unless stated. The prior is a Gaussian field on theta = log eps_r (link log_eps; Matern nu 1.5, sigma 0.5, correlation length 0.4, mean 0). Only the exact Born validation uses the identity link.

## 0. Summary
1. The machinery is validated (kernel, Mie, adjoint gradient, noise model, four samplers against the exact Born posterior; 259 tests in the project's own files, 8 review mutants killed).
2. Every nonlinear posterior is conditional on a measured discretization error: grid error at the truth is 0.74% of ||d|| (weak cylinder, n=32), 2.09% (moderate, n=32; 0.85% at n=48), 8.0% (book scale n=48; 2.6% at n=96); noise is 3.16% at 30 dB.
3. Identity-mass samplers do not converge at n=32, 30 dB (Gauss-Newton top eigenvalue 1.6e7). No sampler ranking is made from these runs.
4. Laplace-whitened samplers converge within the MAP basin. Physical misfit Phi of the draws is stationary at 2432-2436 (N 2400; noise-free data 2438). Posterior std of log eps_r is about 0.058 (weak), 0.054 inside / 0.060 outside the moderate object, ratio 0.117 to the prior; HMC and the Gauss-Newton (Laplace) calculation agree to 1-2%; MALA and pCN-Langevin at equal budget are 3-7% low.
5. At equal evaluation budget HMC has 1.5-2.8x (3001 evals/chain) and 3.2-5.4x (12,001) the worst-pixel ESS of MALA, with the preconditioner setup (MAP + 302 Gauss-Newton products = 737 gradient evaluations) counted. One problem, one seed, untuned trajectory length.
6. Convergence is shown within the MAP basin only: six MAPs from prior draws find one minimum, but a preconditioned HMC chain started at a prior draw does not reach the posterior.
7. Edge miscoverage of the moderate object (10-13% of pixels with |z|>3, 87-95% within one pixel of an edge, max |z| about 9) is not caused by the n=32 forward-model error (same-grid control 10.6-12.0%, n=48 11.4-13.2%). A rough GMRF prior gives 0.2-0.6% but that chain is unconverged. The smooth prior is the live explanation, not established.
8. Book scale is not reachable with the present optimizer: n=48 grid error alone is 2.5x the noise (Phi at the truth 17,990); at n=96, 20 dB (grid error 0.26 of noise) the MAP still ends at Phi 38,069 after 400 iterations.

## 1. What Phi means
Phi = sum |F(chi) - d_obs|^2 / s^2 is the data misfit (mean N = 2400, sd 49; realized noise energy of this data set 2438). Laplace-whitened runs sample Phi' = Phi + |xi|^2/2 - |zeta|^2/2, which is not the misfit. summary.json marks the trace (phi.kind) and stores the misfit of every stored draw (phi_physical, recomputed from the eps_r maps); tables and figures use it, Phi' has its own column and *_phiprime.png. "MAP" rows are the 400-iteration (200 for groups a, b) L-BFGS iterate, not converged: U is 9.1 (weak) and 40.7 (moderate) above the 800-iteration minimum (Phi 2389.0 vs 2376.3; 2432.9 vs 2378.3), and 208 above for the 200-iteration moderate MAP of group b (Phi 2610). Efficiency is quoted per Phi evaluation; seconds are indicative only (the batch ran 8-13 JAX processes on 16 cores, recorded 1-minute load averages up to 150 (summary.json machine entries; reproduce_report.sh quotes 200)). Idle cost at n=32: 0.020 s (weak) / 0.029 s (moderate) per value-and-gradient evaluation, 0.0009 s per potential-only evaluation (results/cost/cost.json).

## 2. Operating points (grid error vs noise; ratio = model error / noise amplitude; E[Phi(truth)] - N = ratio^2 N)
Source: results/operating_points/{weak,moderate,book}.json, made by scripts/operating_point_table.py (data on the 2n grid, seed 0).
| problem | n (data grid) | SNR | model error % of ||d|| | noise % | ratio | Phi(truth) | excess over noise-free-data Phi (2438) |
|---|---|---|---|---|---|---|---|
| weak cylinder r 0.5, eps_r 1.5, side 2 | 32 (64) | 30 | 0.74 | 3.16 | 0.23 | 2596 | 3.2 sd |
| moderate austria_fixed eps_r 2, side 2 | 32 (64) | 30 | 2.09 | 3.16 | 0.66 | 3401 | 19.7 sd |
| moderate | 48 (96) | 30 | 0.85 | 3.16 | 0.27 | 2578 | 2.9 sd |
| moderate (not run) | 64 (128) | 30 | 0.44 | 3.16 | 0.14 | 2473 | 0.7 sd |
| moderate same-grid control | 32 (32) | 30 | 0 | 3.16 | 0 | 2438 | 0 |
| book austria_fixed eps_r 2, side 5 | 48 (96) | 30 | 8.01 | 3.16 | 2.53 | 17,990 | 317 sd |
| book | 64 (128) | 30 | 4.69 | 3.16 | 1.48 | 7799 | 109 sd |
| book | 96 (192) | 30 | 2.58 | 3.16 | 0.81 | 4058 | 33 sd |
| book, lower SNR | 96 (192) | 20 | 2.58 | 10.0 | 0.26 | 2606 | 3.4 sd |
Replaces "mildly misspecified, 1-2%". The weak problem is mild; the moderate problem at n=32 is not (ratio 0.66). The review's "n >= 96 at 30 dB is enough" is disputed: ratio 0.81 there.

## 3. Validation (unchanged, re-confirmed)
Kernel vs quadrature < 1e-8. Mie (R 0.6, eps_r 2) error 3.09e-2, 8.03e-3, 2.43e-3, 7.41e-4 at n = 24, 48, 96, 192 (second order; same for eps_r 4 and lossy); conjugated data gives 0.36 error. Adjoint gradient vs central differences 1e-10 to 1e-7 (threshold 1e-5). Born exact check (16x16, -20 dB per datum, 4 chains; sharpest posterior precision 194): z-mean rms 0.95-1.05, variance ratios 0.993-1.000, Phi within 1.6 sd, R-hat <= 1.022; negative controls fail. pytest -m "not slow": 323 passed in 2 min 36 s (259 in the project's own files; 64 are another agent's tests/test_batch_run.py); slow test passes. The log link is covered by gradient and Target tests, not by the Born check (identity link only).

## 4. Identity-mass samplers (a: weak, b: moderate; 2 chains x 3000 steps from the MAP, 2 x 2000 from prior draws)
* From the MAP: split R-hat of Phi 1.10-1.51 (weak), 1.15-1.90 (moderate); worst pixel R-hat 4.5-7.7, 3.6-6.5; Phi 2440-2455, 2631-2645; std of log eps_r 0.0012-0.0021 = 2-4% of the Laplace 0.058 (|z|>3 for 86% and 94-95% of pixels, max |z| 180-420): chains never left their start.
* From the prior: Phi averaged over the 1000 stored steps 1.8e5-1.6e6 (weak), 4.7e5-6.2e5 (moderate); pooled R-hat of Phi 7.2-31, pixel R-hat up to 156.
* Why: Gauss-Newton top eigenvalue 1.62e7 (weak), 9.7e6 (moderate); 110 directions above 1.5, 85 above 100. pCN-Langevin needs delta < 2/lambda_max = 1.23e-7 / 2.06e-7 (adapted 1.2-1.3e-7, 2.0e-7). MALA needs h < 4/(1+lambda_max) = 2.46e-7 / 4.12e-7 (adapted 2.1e-7 = 87%, 3.5e-7); the first draft's 2/(1+lambda) was half of that. Moving a soft direction by one prior sd then takes 10^6-10^7 steps.
* No ranking: the earlier "ESS of Phi per 1000 evals pCN 2.6, pCN-Langevin 6.3, MALA 3.9" and "pCN wins per second" came from chains with Phi R-hat up to 1.5 and pixel R-hat up to 7.7 (ESS saturates near 14, the value of a pure trend) and from loaded runs; such ESS values are parenthesized in the tables.

## 5. Book scale (c)
| run | n | SNR | grid error / noise | MAP after 300 / 400 iterations | err_model vacuum, MAP |
|---|---|---|---|---|---|
| c_mala_map | 48 | 30 | 2.53 | Phi 370,902 | 0.326, 0.705 |
| c_book96_mala_map | 96 | 20 | 0.26 | Phi 38,069 (|grad U|_inf 13.7) | 0.332, 0.645 |
At n=48 even the exact coarse-grid truth has Phi 17,990, so no optimizer fix could give a meaningful posterior (the first draft blamed only the optimizer). At n=96, 20 dB the grid is adequate and the MAP still fails (435 evaluations, 91 minutes under load); slow convergence vs local minimum is not separated. The 80-step MCMC that followed is stuck at Phi 38,115 and carries no information. Needs a global optimizer (continuation or coarse-to-fine) before sampling.

## 6. Laplace-whitened samplers (L: 3001 evals/chain for every sampler; Ll: 12,001)
Construction: xi = xi_c + M zeta, M = Sigma^(1/2), Sigma = (I + V diag(lam) V^T)^-1 from the 150 largest Gauss-Newton eigenpairs at the MAP (302 jvp+vjp products, 19-205 s); exact target in zeta, so any fixed center is valid. 2 chains; MALA/pCN-L 300 + 2700 steps, HMC 60 + 240 trajectories of 10 leapfrog steps (jitter 0.2); Ll: MALA 300 + 11,700, HMC 150 + 1050.

6.2 Misfit stationary at the noise level (corrects the first draft): Phi of the stored draws 2432.3-2433.4 (weak), 2433.2-2435.4 (moderate) for all runs; first vs last quarter within 4.6; about 56 above the converged MAP, as expected (half of ~110 data-determined directions). The "drift to 1400-1900" was the trace of Phi' (mean 2248-2354 for the L and Ll runs; 2225-2227 in two control runs), whose slow movement is the 27-35 sd excursion of section 6.5. Recommendation "resolve the Phi level" is withdrawn.

6.3 Posterior std of log eps_r. Laplace std at the MAP (in / out): weak 0.0581 / 0.0585 (ratio 0.117), moderate 0.0540 / 0.0600 (0.117). Sampled std (in / out) and mean ratio to Laplace (minimum over pixels):
| run (evals per chain) | weak std | weak ratio | moderate std | moderate ratio |
|---|---|---|---|---|
| HMC (3001) | 0.0586 / 0.0584 | 1.000 (0.84) | 0.0532 / 0.0600 | 0.996 (0.76) |
| HMC overdispersed (3001) | 0.0567 / 0.0578 | 0.986 (0.80) | 0.0529 / 0.0591 | 0.984 (0.81) |
| MALA (3001) | 0.0541 / 0.0555 | 0.945 (0.69) | 0.0519 / 0.0572 | 0.955 (0.73) |
| MALA overdispersed (3001) | 0.0562 / 0.0562 | 0.963 (0.73) | 0.0527 / 0.0577 | 0.966 (0.74) |
| pCN-Langevin (3001) | 0.0532 / 0.0544 | 0.928 (0.68) | 0.0517 / 0.0577 | 0.962 (0.75) |
| MALA (12,001) | 0.0549 / 0.0574 | 0.974 (0.84) | 0.0528 / 0.0593 | 0.986 (0.84) |
| HMC (12,001) | 0.0581 / 0.0587 | 1.003 (0.92) | 0.0543 / 0.0602 | 1.003 (0.92) |
Headline: std of log eps_r about 0.058 (weak), 0.054 / 0.060 (moderate), ratio 0.117. MALA / pCN-Langevin are 3-7% low on average, up to 32% low per pixel, and rise with budget: unconverged (the first draft's 0.0509 / 0.0528 and "0.105 -> 0.108 with longer chains" were a value still growing). P(eps_r < 1) in true-background pixels 0.48-0.50 (prior 0.5).

6.4 HMC vs MALA vs pCN-Langevin at equal budget. Worst-pixel ESS of log eps_r summed over 2 chains (Sokal; Geyer in parentheses); setup = 737 (weak) / 729 (moderate) gradient evaluations = 12% of a 3001-eval two-chain run, 3% of a 12,001 run.
| run | evals/chain | worst-pixel ESS | per 1000 evals | per 1000 incl. setup | pooled pixel R-hat (max; fraction < 1.1) |
|---|---|---|---|---|---|
| weak MALA / pCN-L / HMC | 3001 | 19 (14) / 19 (15) / 53 (62) | 3.2 / 3.2 / 8.9 | 2.9 / 2.8 / 7.9 | MALA 1.29; 0.83 / HMC 1.04; 1.00 |
| moderate MALA / pCN-L / HMC | 3001 | 22 (17) / 25 (22) / 33 (29) | 3.7 / 4.2 / 5.4 | 3.3 / 3.7 / 4.8 | MALA 1.28; 0.90 / HMC 1.09; 1.00 |
| weak MALA / HMC | 12,001 | 52 (57) / 282 (291) | 2.2 / 11.7 | 2.1 / 11.4 | per run 1.08 / 1.02 |
| moderate MALA / HMC | 12,001 | 43 (53) / 138 (138) | 1.8 / 5.8 | 1.7 / 5.6 | per run 1.07 / 1.02 |
HMC/MALA worst-pixel ESS: 1.5-2.8x at 3001 (Sokal; 1.7-4.4 Geyer), 3.2-5.4x at 12,001 (2.6-5.1 Geyer). Caveats: one seed per cell, untuned trajectory length and steps, noisy 240-draw ESS (two HMC runs with identical settings on the moderate problem gave worst-pixel ESS 32.5 and 54.7, so the 1.5x lower end of the 3001 range is within run-to-run spread); Geyer/Sokal for Phi 0.71-1.74 at 3001 (Sokal/Geyer 0.58-1.41), within 22% for the worst pixel at 12,001. The earlier claim compared 240 HMC trajectories (3001 evals) with 1500 MALA steps (1801 evals) without setup cost.

6.5 Log-link shift and re-centering. Along the stiffest direction v0 (DOI average of theta) the MAP-centered chains sat 31-49 Laplace sd from the center (first draft). That is the log-normal mean-vs-mode shift: the data fix the DOI average of exp(theta), the MAP has zero unresolved fluctuation, fluctuations of variance Var add Var/2 to E exp(theta), so the posterior mean of theta lies about Var/2 below the MAP (Var 0.0034, shift -0.0017). Not an unconverged MAP. The center is now shifted by that amount (35 sd weak, 27-28 sd moderate). Observed posterior-mean minus MAP: -0.0022..-0.0025 (weak), -0.0026..-0.0031 (moderate, n=48, same-grid): the rule gets the sign and 56-79% of the size; final states are 1-21 sd (median ~8) from the new center. The shift is capped at 100 sd: for the GMRF prior it would be 831 sd (Var 0.105) and a chain started there stays at Phi 3e5 with acceptance 0.008; GMRF runs use --laplace-recenter none.

6.6 Scope of convergence. Starts: center + 0.01 N(0,I) in zeta or 2 N(0,I); pooled split R-hat of Phi 1.00-1.03, pixels at most 1.29 (MALA), 1.04-1.09 (HMC): convergence within the MAP basin. Six MAPs from prior draws (two-stage L-BFGS) all reach U = 2385.80 (weak), 2465.54 (moderate) to 1e-3 with identical fields (at most 4.1e-6 rms): one minimum at these contrasts. A preconditioned HMC chain from a prior draw (400 warmup + 240 trajectories) ends at Phi 4734 and 460,653 (last-quarter means 4863 and 472,485) and falls slowly, stiff coordinate 1900-4300 sd away: the Laplace map is local, global exploration (tempering, replica exchange) is needed for any claim beyond the basin.

## 7. What the data determine
7.1 Conditional on this smooth prior every mode it admits is data-determined; sub-wavelength structure is excluded by the prior, not measured. The Matern prior carries only 3.5% of its variance beyond 2 k0 (reproduced: 0.0351); per the review its DOI covariance needs 27 / 132 / 471 modes for 90 / 99 / 99.9% of the variance versus ~110 Gauss-Newton directions above 1.5. The GMRF prior has 54% beyond 2 k0 and Laplace pixel std 0.32 (0.65 of the prior's). A GMRF / edge-preserving comparison (DESIGN step 7) is a prerequisite for any resolution claim; none is made.

7.2 Edge miscoverage of the moderate object (z = (mean - truth)/std in log eps_r; M_edges.md):
| run | grid error / noise | prior | fraction |z|>3 (inside object) | share within 1 px of an edge | max |z| | err_model of mean |
|---|---|---|---|---|---|---|
| n=32 MALA / HMC (3001) | 0.66 | Matern | 0.115 (0.26) / 0.109 (0.25) | 0.94 / 0.91 | 9.0 / 9.4 | 0.111 / 0.110 |
| n=32 MALA / HMC (12,001) | 0.66 | Matern | 0.107 / 0.100 | 0.95 / 0.94 | 9.0 / 8.5 | 0.111 / 0.111 |
| same-grid control MALA / HMC | 0 | Matern | 0.120 (0.27) / 0.106 (0.24) | 0.94 / 0.94 | 8.5 / 9.1 | 0.111 / 0.111 |
| n=48 MALA / HMC | 0.27 | Matern | 0.132 (0.28) / 0.114 (0.24) | 0.87 / 0.90 | 9.8 / 8.3 | 0.128 / 0.126 |
| n=32 HMC (3001), GMRF | 0.66 | GMRF | 0.006 (0.012) | 0.50 | 4.2 | 0.109 |
| n=32 HMC (12,001), GMRF | 0.66 | GMRF | 0.002 (0) | 1.0 (2 pixels) | 3.7 | 0.109 |
A third of the DOI is within one pixel of an edge; the weak cylinder (also a step edge) has no |z|>3 pixel in the HMC runs or the 12,001-evaluation runs (max 2.4-2.8) and 1-2 background edge pixels in the 3001-evaluation MALA / pCN-Langevin runs (max 3.0-3.5). Using the Laplace std instead changes the fractions by < 0.01 for HMC and by up to 0.016 for MALA (0.120 -> 0.105 in the same-grid control). Removing the model error (same grid) or cutting it to 0.85% (n=48) leaves the ring, so the first draft's alternative reading (grid error absorbed near edges) is not supported. The smooth Matern prior is the remaining candidate; with the GMRF prior the same data give 0.2-0.6%, but those chains are not converged: HMC posterior std 0.109 / 0.141 (ratio to Laplace 0.41) at 3001 evals and 0.140 / 0.177 (0.52) at 12,001, still rising, mean shift -0.0098 -> -0.0157 (predicted -Var/2 = -0.053), stiff coordinate 150 -> 230 sd from the MAP; misfit stationary at 2427-2428. Longer GMRF runs and an edge-preserving prior are needed before attributing the edge failures to the prior.

## 8. Recommendations
1. Use Laplace-whitened HMC at this scale; report Phi of the draws, not the sampler trace; report the operating point with every run.
2. Choose grid and SNR together so the grid error is <= ~0.25 of the noise amplitude (n=48 for the moderate object at 30 dB; n of ~170-210 for the book object at 30 dB (extrapolated from the table in Section 2, not run)), or label the run misspecified.
3. Book scale: first a global optimizer (continuation / coarse-to-fine) that reaches Phi of order N at n=96, 20 dB; then sampling.
4. Before any resolution claim: converged GMRF and edge-preserving posteriors on the same data, compared pixel by pixel with Matern.
5. Global exploration (tempering, replica exchange) to extend section 6.6 beyond the MAP basin.
6. A pilot-chain-mean center instead of the -Var/2 rule to remove the remaining offset along v0 (final states 1-21 sd, median ~8).

## 9. Review log
| # | finding | resolution |
|---|---|---|
| 1 | Laplace "Phi" is Phi' (major) | Fixed: physical Phi recomputed per stored draw (Target.misfit_of_chi, inference.physical_phi_of_draws), stored as phi_physical, trace labeled, tables/figures use it, Phi' separate. Misfit 2432-2436 stationary. Tests: test_laplace_target_potential_is_phi_prime_not_the_misfit, test_script_labels_the_sampler_trace_as_phi_prime_and_reports_the_physical_misfit, test_misfit_of_chi_is_the_potential_of_the_contrast_map, test_physical_phi_of_draws_matches_the_potential_of_each_draw. |
| 2 | MALA / pCN-Langevin std 5-10% low (major) | Fixed: Laplace reference std (inference.laplace_theta_variance, tested vs dense) and MCMC/Laplace columns; headline 0.058 / ratio 0.117 from HMC and Laplace; MALA / pCN-Langevin labeled unconverged (0.945 -> 0.974 with budget). |
| 3 | Book-scale grid grossly misspecified (major) | Fixed: operating_point in run_posterior.py (warning above ratio 0.35), scripts/operating_point.py for old runs, per-problem values quoted. Disputed: n >= 96 at 30 dB is not enough (ratio 0.81); re-run at n=96, 20 dB still fails to converge (optimizer). |
| 4 | Edge ring: prior or grid error? (minor) | Controls run; grid error not supported as the cause; GMRF suggestive but unconverged; presented as open. |
| 5 | Center 36 sd from the mass (minor) | Fixed: log-normal -Var/2 re-centering (capped at 100 sd), explained as a log-link effect; explains 56-79% of the shift; pilot-mean center deferred. |
| 6 | MALA limit is 4/(1+lambda_max) (minor) | Fixed: fields and text corrected; test shows adapted step = 0.83 x 4/a. |
| 7 | Rankings from non-stationary / unequal chains (minor) | Fixed: ranking dropped for identity-mass; equal-evaluation HMC / MALA / pCN-L with setup cost and Geyer cross-check; ESS of unconverged chains parenthesized. |
| 8 | Cross-init probes only the MAP basin (minor) | Fixed: wording; six-start MAP (single minimum); prior-started HMC does not converge. |
| 9 | "Nothing prior-dominated" misleads (minor) | Fixed: rephrased, 3.5% beyond 2 k0; GMRF / edge-preserving comparison stated as prerequisite (GMRF unconverged, edge-preserving not done). |
| 10 | validation.py docstring (minor) | Fixed: precision ~300 (sd 17x), 35 directions above 1.5; at +20 dB ~3e6 (1700x), 69; test pins them. |
Report corrections 1-8: (1) Phi drift -> 6.2; (2) headline std -> 6.3; (3) "mildly misspecified" -> 2; (4) book scale -> 5; (5) MALA limit and sampler ranking -> 4; (6) "HMC most efficient", "two starts agree" -> 6.4, 6.6; (7) "nothing prior-dominated" -> 7.1; (8) edge miscoverage -> 7.2.
Additional findings: the 400-iteration MAP is not converged (U 9 and 41 above the minimum; moderate Phi 2433 vs 2378); the log-normal shift rule fails for the rough GMRF prior and is capped; ESS of frozen, unconverged chains looks large (e.g. 128 per 1000 evals at n=96 book scale) and is now parenthesized.
User request "put the prior on log eps_r": already the case for all nonlinear inference (and kept in every new piece); the Born validation must keep the identity link because its exact Gaussian posterior requires an affine map.

## Files
scripts/{run_posterior,run_posterior_laplace,map_multistart,operating_point,analyze_run,make_tables,plot_gn_spectrum}.py, scripts/reproduce_report.sh, src/bayes_scatter/{inference,likelihood,diagnostics,plotting,validation}.py (edited), tests/test_laplace.py (new) plus additions to tests/test_{likelihood,inference,diagnostics,scripts,born_end_to_end}.py, docs/DESIGN.md, README.md; results/tables/{a_weak,b_moderate,c_book,L_weak,L_mod,Ll_weak,Ll_mod,M_edges,P_weak}.md, results/figures/*.png, results/{MS_weak,MS_mod}/multistart.json, results/laplace_cache/*.npz. Operating points: results/operating_points/*.json.
## 10. Numbers audit log
An independent audit traced 89 quantitative claims to the raw outputs (summary.json, metrics.json, chains.npz, caches, logs) and recomputed the key ones from the stored chains: 65 matched, 11 differed only in rounding, 8 were wrong and 5 had no stored source. All counts and every equal-budget comparison reproduced, as did the headline posterior std, worst-pixel ESS, |z|>3 fractions and the misfit range of the stored draws (five draws re-evaluated through the forward model). Corrections applied: the weak cylinder does have 1-2 edge pixels with |z|>3 in three 3001-evaluation MALA / pCN-Langevin runs; the Phi ESS ratio range is Geyer/Sokal, not Sokal/Geyer; the prior-started HMC end values; the weak prior-start Phi lower bound; the GMRF pixel count (2, not 1); the Laplace-std sensitivity of the |z| fraction (up to 0.016 for MALA); the moderate idle cost (0.029 s); the gradient-check range (1e-10 to 1e-7); the recorded load average (150, not 250); and the book-scale grid recommendation (n ~170-210 at 30 dB, extrapolated, not 128-190). The operating points that existed only in a scratch log are now regenerated and stored in results/operating_points/. No conclusion of the report changed.
