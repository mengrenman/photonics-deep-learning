# Milestone 4: prior comparison on the moderate problem (DESIGN 11.3)

> **Note added after milestone 5** (results/M5_REPORT.md §0.2, §4): the reading below that the edge bias is "chosen by the prior" is superseded. The posterior-mean bias of every prior correlates 0.86 to 0.93 with the bias of the truth low-passed at about 3 cycles per wavelength, so it is the resolution limit of the data, common to all priors here; against the low-passed truth the |z| > 3 fractions fall to 0 to 5 %. The prior-energy numbers in §4 remain correct as statements about what each prior prefers.

Revised after the independent review. Every number below is read from a file under results/prior_comparison (the file is named in each section). The tables are printed by `scripts/prior_comparison_tables.py` straight from `summary.json`, `comparison.json`, `prior_energy.json`, `convergence_audit.json` and `edge_analysis.json`. Per-run files are `pc_<prior>/summary.json` and `chains.npz`.

## 0. Result

1. **Convergence (DESIGN 11.2 classic criteria; section 2).** Three runs converged: Matern 4 x (300 + 600) draws, q-GGMRF sigma_g 0.05 4 x (300 + 900), q-GGMRF sigma_g 0.1 4 x (300 + 600). The GMRF did not converge (4 x (100 + 500)). The two q-GGMRF settings are one prior family, so only two families have a converged posterior. No conclusion rests on the GMRF.
2. **The edge ring is not specific to the smooth prior, and it is not shown to be independent of the prior.** Every prior tested prefers blurred edges. The prior energy R of the truth lies above that of the MAP for all four priors (R 14,685.5 vs 87.5 Matern; 4,406.2 vs 2,718.8 q-GGMRF 0.05; 1,772.7 vs 1,060.6 q-GGMRF 0.1; 89.1 vs 33.1 GMRF), and for both q-GGMRF settings a step of height log 2 costs 2.2 to 2.7 times more as one jump than as an 8-pixel ramp. In the same-grid controls the data do not ask for the blur: for q-GGMRF 0.05 Phi(truth) 2438.2 < Phi(MAP) 2470.9 < Phi(draws) 2507.6 (the data favor the sharp truth by 33 and 69 nats) while R(truth) exceeds R(MAP) by 1,733 nats and R(draws) by 453; for Matern the draws fit the data as well as the truth (Phi 2434.4 against 2438.2) while R(truth) exceeds R(draws) by 13,508. So the edge bias is chosen by the prior, in directions the data barely constrain, and the |z| > 3 fraction measures the width the prior claims. Whether a prior that does not penalize sharp edges removes the ring is untested.
3. **The bias at the edge is nearly the same under the three converged priors** (+0.204 to +0.215 in log eps_r just outside the object, -0.106 to -0.125 two pixels inside); what differs is the posterior std there (0.052 q-GGMRF 0.05, 0.061 Matern, 0.083 q-GGMRF 0.1; 0.179 for the unconverged GMRF). The |z| > 3 fraction is 15.5, 12.5, 5.8% (GMRF 0.4%), and the per-chain fractions lie within 0.1 to 0.6 percentage points of the pooled value (0.1 to 0.5 between the chains of one run) (section 4).
4. **Features (section 5).** Gap valley and ring hole: provisional (present), agreed by Matern and q-GGMRF only. Ring continuity: not established (the unconverged GMRF gives 0.929, below 0.95, and it fell from 0.974 as its chains were extended). The DESIGN 8.7 minimum comparison (a converged Matern against a converged GMRF) was not met, so no feature is called data-determined.
5. **Controls (section 4).** Grid error is not the cause of the ring under either prior: same-grid Matern 12.3% against 12.5% at n = 48 with grid error; same-grid q-GGMRF 0.05 16.3% against 15.5%; second data realization 16.1%.
6. **Data identity.** The d_obs hash depends on the XLA thread count (T = 1: efd5e84f..., T = 4: d1000a64...) while the arrays agree to 6.5e-14 of the noise std (data_thread_check.json). Data equality is now tested numerically (max |d_a - d_b| / s <= 1e-6).

## 1. Setup and operating point (pc_matern/summary.json)

austria_fixed eps_r 2, side 2, inference grid n = 48, data on n = 96, 40 tx x 60 rx, N = 2400 complex data, SNR 30 dB, data seed 0, noise std s = 3.9228e-4.

- Operating point: grid model error 0.85% of ||d||, noise 3.16%, ratio 0.269. Phi(truth) = 2578.1; the noise-free-data Phi is 2438.2 (the realized noise energy, 0.78 sd above N = 2400 with sd sqrt(N)); Phi(truth) is 2.86 sd above it.
- One data realization is shared by all pc_ runs: d_obs SHA-256 efd5e84fa5f9a77f in all four (comparison.json `data_check.identical` true, max |delta d| / s = 0). Runs made with data on the inference grid (pcg1_*) share the hash 5745046460dbf188.
- Priors: Matern (sigma 0.5, ell 0.4, nu 1.5), GMRF (sigma 0.5, ell 0.4), q-GGMRF (q 1.2, T 1, sigma_0 1, sigma_g 0.05 and 0.1).
- MAP (L-BFGS-B in theta) and dense Laplace mass; the reciprocity Jacobian agrees with jax.jvp columns to 1.4e-10 to 1.6e-10 in the four main runs (1.4e-10 to 1.7e-10 including the three control runs).

| run | MAP its | U | Phi | R | grad U inf | Laplace sd in / out |
|---|---|---|---|---|---|---|
| matern | 350 | 2465.9 | 2378.4 | 87.5 | 5.9e-3 | 0.0543 / 0.0608 |
| gmrf | 702 | 2400.1 | 2367.0 | 33.1 | 9.1e-3 | 0.3449 / 0.3503 |
| qggmrf 0.05 | 134 | 5193.2 | 2474.3 | 2718.8 | 1.0e-2 | 0.0463 / 0.0331 |
| qggmrf 0.1 | 191 | 3467.1 | 2406.5 | 1060.6 | 5.8e-3 | 0.0752 / 0.0577 |

Agreement with the previous milestone (results/REPORT.md sections 6 and 7): the Matern run reproduces it with independent code and now 600 draws: physical Phi 2434.7 (REPORT 2433 to 2435), |z| > 3 12.5% (11.4 to 13.2%), edge share 0.88 (0.87 to 0.90), MCMC/Laplace std 0.999 (0.984 to 1.003), err_model of the mean 0.1263 (0.126 to 0.128).

## 2. Per prior (summary.json of each run; convergence_audit.json)

HMC in zeta = Laplace-whitened coordinates, 10 leapfrog steps, jitter 0.2, step adapted in warmup then frozen. One chain per process, one XLA thread. Phi is the physical misfit of the stored draws, not the sampler's Phi'.

| run | chains x (warmup + draws) | accept | Phi mean (sd) | R-hat Phi | max pixel R-hat | worst ESS Sokal / Geyer / multi-chain bulk | converged |
|---|---|---|---|---|---|---|---|
| matern | 4 x (300 + 600) | 0.660 | 2434.7 (7.6) | 1.002 | 1.009 | 883 / 906 / 774 | yes |
| gmrf | 4 x (100 + 500) | 0.524 | 2427.7 (8.4) | 1.033 | 1.091 | 127 / 131 / 33 | NO |
| qggmrf 0.05 | 4 x (300 + 900) | 0.707 | 2511.4 (13.0) | 1.003 | 1.005 | 2142 / 2070 / 1968 | yes |
| qggmrf 0.1 | 4 x (300 + 600) | 0.705 | 2452.4 (9.0) | 1.001 | 1.006 | 1242 / 1233 / 1211 | yes |

How the extensions went (R-hat Phi / max pixel R-hat / worst Sokal ESS; the 300-draw numbers are in `summary_seg0_only.json` of each run):

| run | 300 draws | 600 draws | 900 draws |
|---|---|---|---|
| matern | 1.0069 / 1.0173 / 482 (pass) | 1.0018 / 1.0092 / 883 (pass) | not run |
| qggmrf 0.05 | 1.0083 / 1.0121 / 606 (pass) | 1.0103 / 1.0061 / 1273 (**R-hat Phi misses 1.01 by 0.0003**; pc_qggmrf_sg0.05_600draws/summary.json) | 1.0028 / 1.0046 / 2142 (pass) |
| qggmrf 0.1 | 1.0120 / 1.0134 / 498 (miss; summary_seg0_only.json) | 1.0012 / 1.0062 / 1242 (pass) | not run |

The q-GGMRF 0.05 run was extended a second time because its 600-draw R-hat of Phi missed the DESIGN threshold, as the q-GGMRF 0.1 run had been extended after 300; the DESIGN flag is the classic one and the extension rule is "extend once more if a classic criterion is missed". The |z| > 3 fraction hardly moved (12.15 to 12.5% Matern, 15.45 to 15.5% q-GGMRF 0.05).

**Convergence audit (convergence_audit.json; section 3 of the review).** The classic thresholds are within the noise at 300 draws per chain: under an AR(1) null with the observed autocorrelation time, P(R-hat Phi > 1.01) is 0.17 and 0.27 for the 3 x 300 same-grid controls before their extension (logs/convergence_audit_part1.log, part2.log) and 0.23 for the 2 x 600 second-realization run, but 0.02 to 0.03 at 4 x 600 (so the 600-draw miss of q-GGMRF 0.05, null probability 0 of 60, was not pure noise). Rank-normalized split R-hat (bulk and folded, Vehtari et al. 2021) and the AR(1) null of its maximum over pixels (independent pixels with the observed per-statistic autocorrelation times; this null is permissive where non-equilibration inflates the estimated time):

| run | chains x draws | R-hat Phi / rank-normalized worst | max pixel R-hat / rank-normalized worst | null q95 of that | pixels rank R-hat > 1.05 (within 1 px of an edge) |
|---|---|---|---|---|---|
| matern | 4 x 600 | 1.0018 / 1.0022 | 1.0092 / 1.0221 | 1.0370 | 0 (0) |
| qggmrf 0.05 | 4 x 900 | 1.0028 / 1.0106 | 1.0046 / 1.0249 | 1.0389 | 0 (0) |
| qggmrf 0.05 at 600 draws | 4 x 600 | 1.0103 / 1.0113 | 1.0061 / 1.0522 | 1.0743 | 1 (0) |
| qggmrf 0.1 | 4 x 600 | 1.0012 / 1.0011 | 1.0062 / 1.0405 | 1.0495 | 0 (0) |
| gmrf | 4 x 500 | 1.0327 / 1.0330 | 1.0906 / 1.0898 | 1.1206 | 46 (3) |
| qggmrf 0.05 second data seed | 2 x 600 | 1.0097 / 1.0097 | 1.0104 / 1.0604 | 1.1143 | 4 (2) |
| qggmrf 0.05 same-grid control | 3 x 600 | 1.0009 / 1.0066 | 1.0071 / 1.0521 | 1.0716 | 1 (1) |
| matern same-grid control | 3 x 600 | 1.0043 / 1.0043 | 1.0094 / 1.0215 | 1.0349 | 0 (0) |

For q-GGMRF 0.05 the folded rank-normalized R-hat of Phi is 1.0106 at 900 draws (its null median 1.0044, q95 1.0099), a marginal residual sign that the classic flag does not show. Before the extension the 3 x 300 same-grid q-GGMRF control had 15 pixels above 1.05, 12 of them within one pixel of an edge (max 1.093, inside its permissive null q95 1.131); after extension to 3 x 600 one pixel is above 1.05. The multi-chain bulk ESS is lower than the summed Sokal ESS in every run except the second-realization run (626 against 609 for the worst pixel) and 3.8 times lower for the GMRF (33 against 127), which is the chains-have-not-mixed signature; every converged run keeps it above 200.

Posterior std of theta = log eps_r (summary.json `posterior`; prior std from `prior.prior_summary`):

| run | prior sd (in / out) | posterior sd (in / out) | posterior / prior (in / out / mean) | MCMC / Laplace sd, mean (min to max) |
|---|---|---|---|---|
| matern | 0.5 / 0.5 | 0.0542 / 0.0607 | 0.108 / 0.121 / 0.119 | 0.999 (0.890 to 1.105) |
| gmrf (unconv.) | 0.5 / 0.5 | 0.1553 / 0.1966 | 0.311 / 0.393 / 0.375 | 0.538 (0.350 to 0.780) |
| qggmrf 0.05 | 0.0519 / 0.0578 | 0.0488 / 0.0403 | 0.942 / 0.709 / 0.760 | 1.230 (0.786 to 1.646) |
| qggmrf 0.1 | 0.0920 / 0.1007 | 0.0817 / 0.0742 | 0.891 / 0.745 / 0.777 | 1.278 (0.822 to 1.709) |

- The q-GGMRF prior std comes from HMC on R alone (R-hat 1.004 and 1.003, worst-pixel ESS 1752 and 2850 in the summaries; prior_features.json has the separate 4 x (200 + 600) run).
- Matern is Gaussian to about 1%; the q-GGMRF posteriors are 23 to 28% wider than Laplace on average and up to 71% per pixel.
- The q-GGMRF posterior std is mostly the prior std (posterior / prior 0.76 to 0.78); the data narrow these priors only slightly, unlike Matern (0.12).

Miscoverage and reconstruction errors (z = (mean - truth) / std in log eps_r):

| run | % abs z > 3 (inside / outside) | edge share (chance 0.233) | max abs z | err_model MAP / mean | err_chi MAP / mean | err_data mean |
|---|---|---|---|---|---|---|
| matern | 12.5 (27.0 / 8.4) | 0.882 | 8.41 | 0.1260 / 0.1263 | 0.3868 / 0.3879 | 0.0062 |
| gmrf (unconv.) | 0.4 (0.8 / 0.3) | 0.900 | 3.76 | 0.1246 / 0.1585 | 0.3825 / 0.4868 | 0.0068 |
| qggmrf 0.05 | 15.5 (38.4 / 9.1) | 0.810 | 12.57 | 0.1322 / 0.1345 | 0.4059 / 0.4129 | 0.0074 |
| qggmrf 0.1 | 5.8 (13.2 / 3.8) | 0.918 | 6.71 | 0.1296 / 0.1316 | 0.3980 / 0.4040 | 0.0062 |

The GMRF row is descriptive only: its posterior mean is worse than its MAP (err_model 0.1585 against 0.1246), what a chain still drifting through prior-dominated directions looks like.

Figures (relative to results/): prior_comparison/comparison_mean_std.png, prior_comparison/pc_matern/maps.png, prior_comparison/pc_qggmrf_sg0.05/maps.png, prior_comparison/pc_qggmrf_sg0.1/maps.png, prior_comparison/pc_gmrf/maps.png (unconverged).

## 3. Disagreement map (comparison.json `pairs`)

Statistic |m_A - m_B| / sqrt(s_A^2 + s_B^2) on theta; fraction of pixels above 2.

| pair | both converged | % pixels > 2 | max | mean |
|---|---|---|---|---|
| matern vs qggmrf 0.05 | yes | 0.2 (none near an edge) | 2.33 | 0.37 |
| matern vs qggmrf 0.1 | yes | 0.0 | 1.37 | 0.20 |
| qggmrf 0.05 vs qggmrf 0.1 | yes | 0.0 | 0.51 | 0.12 |
| matern vs gmrf | no | 0.0 | 1.97 | 0.42 |
| gmrf vs qggmrf 0.05 / 0.1 | no | 0.1 / 0.0 | 2.23 / 1.94 | 0.45 / 0.41 |

The converged posteriors agree: no pixel within one pixel of an edge exceeds 2 in any converged pair; the few pixels above 2 sit away from the object. The GMRF pairs are lenient because its std is large and still growing, and are descriptive only. Agreement of posteriors that all share the same blurred edge is not evidence that the edge position is determined (section 4). Figure: prior_comparison/comparison_disagreement.png.

## 4. Is the edge ring a property of the prior? (prior_energy.json, edge_analysis.json)

The posterior is exp(-U), U = Phi + R, with Phi = -log-likelihood in nats. `scripts/prior_energy_audit.py` evaluates R and Phi at the coverage-averaged truth, at the MAP and over the stored draws, with each run's own prior and data.

| run | R truth | R MAP | R draws (mean +- sd) | (R truth - R draws) / sd | Phi truth | Phi noise-free data | Phi MAP | Phi draws | U truth - U MAP |
|---|---|---|---|---|---|---|---|---|---|
| matern | 14685.5 | 87.5 | 1180.7 +- 32.9 | 409.9 | 2578.1 | 2438.2 | 2378.4 | 2434.7 +- 7.6 | 14797.8 |
| gmrf (unconv.) | 89.1 | 33.1 | 524.7 +- 45.7 | -9.5 | 2578.1 | 2438.2 | 2367.0 | 2427.7 +- 8.4 | 267.1 |
| qggmrf 0.05 | 4406.2 | 2718.8 | 3998.7 +- 38.7 | 10.5 | 2578.1 | 2438.2 | 2474.3 | 2511.4 +- 13.0 | 1791.2 |
| qggmrf 0.1 | 1772.7 | 1060.6 | 2357.0 +- 38.3 | -15.3 | 2578.1 | 2438.2 | 2406.5 | 2452.4 +- 9.0 | 883.7 |
| qggmrf 0.05, seed 1 | 4406.2 | 2698.4 | 3978.4 +- 38.7 | 11.0 | 2572.2 | 2420.0 | 2461.0 | 2500.7 +- 11.2 | 1819.0 |
| qggmrf 0.05, same grid | 4406.2 | 2673.7 | 3953.3 +- 40.2 | 11.3 | 2438.2 | 2438.2 | 2470.9 | 2507.6 +- 11.4 | 1699.9 |
| matern, same grid | 14685.5 | 84.7 | 1177.1 +- 34.1 | 396.1 | 2438.2 | 2438.2 | 2378.0 | 2434.4 +- 7.9 | 14661.1 |

Reading, in order of strength of the evidence:

1. **Every prior prefers the blurred reconstruction to the truth.** R(truth) > R(MAP) in all seven rows. R decreases monotonically when the truth is blurred by a Gaussian of 0, 0.5, 1, 1.5, 2 pixels (blur table in prior_energy.json): Matern 14,685.5 / 4,311.2 / 274.1 / 79.6 / 28.5; q-GGMRF 0.05 4,406.2 / 3,769.6 / 2,801.2 / 2,020.1 / 1,435.1; q-GGMRF 0.1 1,772.7 / 1,485.4 / 1,061.1 / 745.3 / 518.8; GMRF 89.1 / 60.4 / 30.8 / 18.2 / 11.2.
2. **The cost of a sharp step.** Per pair line, a step of height log 2 (the true jump) costs 10.45 as one jump and 4.71 as an 8-pixel ramp under q-GGMRF 0.05 (ratio 2.22), and 4.21 against 1.59 under q-GGMRF 0.1 (2.65). The potential is convex with rho(0) = 0, so k rho(D / k) <= rho(D) for every k: every convex member of the family prefers a ramp (tested). Other members: q 1.0, T 1, sigma_g 0.05 ratio 1.47; q 1.5 3.92; q 1.0 with T sigma_g = log 2, quadratic over the whole jump, 4.50 (the most blur-preferring); q 1.0 with T 0.1 or 0.01 (near total variation) 1.05 and 1.01. The q-GGMRF "preserves edges" only relative to a quadratic.
3. **Against typical posterior draws the truth is atypical only for Matern and q-GGMRF 0.05**: +410 sd (14,685.5 against 1,180.7 +- 32.9) and +10.5 sd (4,406.2 against 3,998.7 +- 38.7). For q-GGMRF 0.1 and the GMRF the truth is cheaper than a typical draw (-15.3 and -9.5 sd) because their draws are rough; there the preference for blur is relative to the MAP only, and the ring fraction is smaller because the claimed width is larger.
4. **The data do not ask for the blur** (same-grid controls, no grid error, Phi(truth) = 2438.2 is exactly the realized noise energy). With q-GGMRF 0.05: Phi(truth) 2438.2 < Phi(MAP) 2470.9 < Phi(draws) 2507.6, so the data favor the sharp truth by 32.7 nats against the MAP and 69.4 against the draws, while the prior charges the truth 1,732.5 more than the MAP and 452.9 more than a typical draw. With Matern the MAP fits part of the noise (Phi 2378.0 < 2438.2) and the draws fit as well as the truth (2434.4 against 2438.2): the data are indifferent, and the prior charges the truth 13,508 more than a typical draw. An isotropic blur of the truth is not free for the data (same-grid Phi 2438 / 5065 / 31,353 / 83,081 / 152,845 for blur 0 / 0.5 / 1 / 1.5 / 2 px), so the posterior blur is not isotropic: it lives in a few directions the data barely constrain.
5. **The bias is common, the claimed width is not.** At d = -1 (the pixel row just outside the object; positive distances are inside):

| run | bias at d=-1 | posterior sd at d=-1 | % abs z > 3 at d=-1 | % abs z > 3 in the band abs d = 1 | % abs z > 3 at abs d >= 4 | % abs z > 3 pooled | per chain min to max | Monte Carlo error of the pooled fraction (pp) |
|---|---|---|---|---|---|---|---|---|
| matern | +0.215 | 0.061 | 61.3 | 49.4 | 1.2 | 12.5 | 12.2 to 12.6 | 0.08 |
| qggmrf 0.05 | +0.208 | 0.052 | 64.5 | 54.0 | 1.8 | 15.5 | 15.6 to 15.7 | 0.02 |
| qggmrf 0.1 | +0.204 | 0.083 | 27.4 | 23.1 | 0.6 | 5.8 | 5.9 to 6.4 | 0.11 |
| gmrf (unconv.) | +0.177 | 0.179 | 2.4 | 1.8 | 0.1 | 0.4 | 0.3 to 0.4 | 0.02 |

(Per-chain fractions use each chain's own mean and within-chain std, so they tend to lie above the pooled value (true of the q-GGMRF runs; for Matern the per-chain mean is 12.41% against a pooled 12.46%, and for the GMRF 0.40% against 0.43%); edge_analysis.json `z_per_chain_frac_abs_z_gt_3`. At d = +2 the bias is -0.106 Matern, -0.125 q-GGMRF 0.05, -0.118 q-GGMRF 0.1, -0.112 GMRF.) With the bias nearly fixed, the fraction beyond 3 sigma falls as the claimed sd rises; the GMRF's 0.4% comes from an sd that is still growing (0.128 and 0.131 in the two pilots to 0.188) and unconverged, so it is weak evidence for prior dependence, not evidence against it.
6. **Grid error is ruled out, now like for like** (n = 48 for both priors):

| run | what differs | converged | % abs z > 3 | edge share | Phi mean |
|---|---|---|---|---|---|
| pc_matern | baseline, 4 x (300 + 600) | yes | 12.5 | 0.882 | 2434.7 |
| pcg1_matern | same-grid control (data on n = 48, grid error 0), 3 x (300 + 600) | yes | 12.3 | 0.883 | 2434.4 |
| pc_qggmrf_sg0.05 | baseline, 4 x (300 + 900) | yes | 15.5 | 0.810 | 2511.4 |
| pcg1_qggmrf_sg0.05 | same-grid control, 3 x (300 + 600) | yes | 16.3 | 0.798 | 2507.6 |
| pcs1_qggmrf_sg0.05 | second data realization (seed 1), 2 x (300 + 600) | yes (R-hat Phi 1.010) | 16.1 | 0.798 | 2500.7 |

The earlier comparison of a q-GGMRF n = 48 control with the previous milestone's Matern n = 32 control (10.6 to 12.0%) is replaced by this table. The ring is not a feature of one noise draw.

**Conclusion.** The ring is not specific to the Matern prior and it is not shown to be independent of the prior: all priors tested favor blurred edges by prior energy, the data are indifferent to or favor the sharp truth in the controls, the bias is common and the |z| > 3 fraction tracks the width each prior claims. What is not tested: a prior that does not penalize sharp edges (the near-total-variation end of the q-GGMRF family, q = 1 with T sigma_g far below the jump; a non-convex potential, which breaks the SPD Laplace mass of DESIGN 11.2; the shape family), a grid or SNR change, and the GMRF to convergence.

## 5. Features and verdicts (comparison.json, summary.json `features`; priors alone: prior_features.json)

delta = log(2) / 2 = 0.347. The truth satisfies all three features (geometry: disks r 0.2 at (+-0.36, 0.4), gap 0.32, ring at (0, -0.32) with radii 0.36 / 0.52, 250 annulus pixels). The +- values are the standard error across chains (the sd of the per-chain estimates divided by the square root of the number of chains, `mcse` in features.feature_probabilities); the sd itself is twice as large.

| run | converged | gap valley | ring hole | ring continuity |
|---|---|---|---|---|
| prior alone, Matern | - | 0.061 | 0.165 | 0.017 |
| prior alone, GMRF | - | 0.117 | 0.188 | 0.004 |
| prior alone, q-GGMRF 0.05 / 0.1 | - | 0 / 0 | 0 / 0 | 0 / 0 |
| posterior, matern | yes | 1.000 | 0.984 (+-0.003) | 1.000 |
| posterior, qggmrf 0.05 | yes | 1.000 | 1.000 | 1.000 |
| posterior, qggmrf 0.1 | yes | 1.000 | 1.000 | 1.000 |
| posterior, gmrf | NO | 1.000 | 0.996 (+-0.004) | 0.929 (+-0.012) |

Verdicts (`features.verdict`; the prior families are matern, gmrf, qggmrf and the two q-GGMRF settings count once; converged families: matern, qggmrf; `design_minimum_met` false):

| feature | verdict |
|---|---|
| gap valley | provisional (present): agreed by matern, qggmrf only; the DESIGN minimum needs a converged gmrf |
| ring hole | provisional (present): agreed by matern, qggmrf only; the DESIGN minimum needs a converged gmrf |
| ring continuity | not established: unconverged gmrf disagrees with the converged runs (present) |

- The converged posteriors agree on all three features (P from 0.984 to 1.000), but the features are coarse (a 0.32 lambda gap, a 0.16 lambda ring) and every blurred reconstruction shows them.
- Ring continuity is the weakest: the GMRF, the one structurally rough prior, gives 0.929 and its value fell from 0.974 to 0.929 as its chains were extended; the compass-point margin under the GMRF is smaller and its std still grows, so convergence is unlikely to raise it.
- The q-GGMRF prior-alone probability of 0 says nothing about structure: its marginal std (0.057 and 0.099) is below delta = 0.347, so the prior cannot reach the feature amplitude at all.
- Gap valley and ring hole are agreed even by the unconverged GMRF, but that run cannot count. The verdict becomes data-determined only with a converged GMRF (or Matern against another structurally different prior).

Figure: prior_comparison/comparison_features.png.

## 6. The GMRF (gmrf_diagnostic.json; pilots pg025_gmrf, pg05_gmrf)

- Starts at the DESIGN scale 2 are hopeless: the n = 32 builder check gave physical Phi 4.7e6 and solves at the iteration cap.
- Two single-chain pilots (start scale 0.25 and 0.5, 40 + 60 trajectories) took 782 s and 1927 s for 1000 evaluations and settled at step 0.058 with acceptance 0.82 and 0.78. The local Hessian of the pilot end states has largest eigenvalue 274.6 and 261.7 in the MAP coordinates (step limit about 2 / sqrt(lambda_max)); rebuilding the mass at the mean of the two pilot end states gives 8.30 and 8.31 (smallest 0.121 and 0.120). The final run used that mass (exact affine change of variables), starts at scale 0.5 around it; the step stayed 0.061 to 0.076 and acceptance 0.39 to 0.68.
- The chains kept drifting: Phi' at the end of segment 1 and 2 was -30,034 and -49,847 (chain 0; -30,000 to -55,000 across chains), |zeta|^2 / 2 grew from 33,000 to 52,800 (chain 0), the mean of the first half of the draws differs from that of the second by 0.047 to 0.056 rms against a pixel std of 0.184 to 0.186, and the posterior sd grew from 0.128 and 0.131 (the two pilots) to 0.188.
- Result: R-hat Phi 1.033, max pixel R-hat 1.091 (46 pixels above 1.05, 3 near edges), worst-pixel ESS 127 (multi-chain bulk 33) after 4 x (100 + 500): unconverged, no conclusion drawn. Remedies not tried: a local (Riemannian) mass, much longer trajectories, tempering, a prior-informed reparametrization.

## 7. Honest limits

- The minimum DESIGN comparison, Matern against GMRF, was not achieved; the feature agreement rests on Matern and two settings of one q-GGMRF family.
- The DESIGN q-GGMRF settings give a tight prior (marginal sd 0.057 and 0.099 against Matern's 0.5; the MAP pays R = 2,719 and 1,061), so those posteriors are mostly prior width and say little about what the data alone determine.
- Convergence is by the DESIGN classic flag; the rank-normalized folded R-hat of Phi for q-GGMRF 0.05 (1.0106) is at the 95th percentile of its null, and the AR(1) null is permissive for pixels. q-GGMRF 0.1 was extended after its 300-draw state missed the classic R-hat of Phi, and q-GGMRF 0.05 from 600 to 900 draws after its R-hat of Phi missed (its extension from 300 to 600 followed a pass and was made in the review batch with the Matern) (a repeated-stopping choice, stated); Matern was extended on request of the review.
- Single data realization per DESIGN. The seed-1 replicate exists only for q-GGMRF 0.05 and has 2 chains; the same-grid controls have 3 chains of 600 draws.
- The |z| statistics and the disagreement statistic use the posterior's own std. They test calibration against one truth, not the posterior's validity under its own prior.
- The truth is the coverage-averaged phantom on the inference grid; the operating-point ratio is 0.27, below the 0.35 warning level.
- The machine was shared (1-minute load 12 to 15 at chain starts, `machine` in each summary.json), so wall times are indicative only: 0.25 to 0.9 s per evaluation (wall_s_per_chain over n_phi_evals_per_chain in each summary.json: 0.45 to 0.50 Matern, 0.37 q-GGMRF 0.05, 0.42 to 0.90 GMRF) against 0.115 s measured at a 1-minute load of 8.7 (cost.json, n = 48, one thread). At most 10 XLA threads of mine ran at any time; the q-GGMRF 0.05 third segment (4 processes) was started by hand while the 3-chain control batch ran (7 threads), not as a second batch.
- The prior energies are evidence about what each prior prefers; they are not a proof that the posterior blur comes only from the prior (the data fix the blur in the strongly constrained directions, which is why an isotropic blur is expensive for the data).
- Tests: 442 passed in the fast suite at NPROC = 8 (logs/pytest_fast_review.log). The pre-existing test test_map_equals_exact_posterior_mean_in_the_born_linear_case fails at NPROC = 4 and passes at NPROC = 1 and 8; the finite-difference tolerance of test_laplace_target_potential_is_phi_prime_not_the_misfit was relaxed from 1e-6 to 1e-5 (below its own stated noise floor).

## 8. Code added or changed in the review

- `features.verdict` (families, DESIGN minimum, provisional, not established); `compare_priors.py` (numerical data check, families, thread environments).
- `run_prior_comparison.py`: numerical d_obs comparison (`D_OBS_TOL` 1e-6), d_obs in prepared.npz, `data.thread_env`, `_threadcap`.
- `diagnostics.py`: `rank_normalize`, `split_rhat_rank`, `ess_bulk`, `ar1_rhat_null`; `scripts/convergence_audit.py`, `scripts/prior_energy_audit.py`, `scripts/check_data_threads.py`, `scripts/_threadcap.py`; `edge_analysis.py` (per-chain fractions); `prior_comparison_tables.py` (audit tables).
- Job files scripts/jobs/review_batch1.json (Matern and q-GGMRF 0.05 segment 1, Matern same-grid prepare), review_pcg1_matern_chains.json, review_pcg1_ext_seg1.json; q-GGMRF 0.05 segment 2 by hand: `python scripts/run_prior_comparison.py --n 48 --side 2 --eps-r 2 --snr 30 --seed 0 --chains 4 --prior qggmrf --q 1.2 --T 1 --sigma-0 1 --sigma-g 0.05 --out pc_qggmrf_sg0.05 --stage chain --chain-id C --segment 2 --samples 300` for C = 0..3, then `--stage finalize` (single-thread environment of batch_run.py).
- Files: results/prior_comparison/{comparison.json, prior_energy.json, convergence_audit.json, edge_analysis.json, data_thread_check.json, pcg1_matern/, pc_qggmrf_sg0.05_600draws/}; logs in results/prior_comparison/logs.

## 9. Review log

| finding | resolution |
|---|---|
| A1 (major): "edge ring is not a property of the smooth prior; blur data-limited and prior-independent" | Fixed. Headline and section 4 rewritten: every prior tested prefers blurred edges; bias prior-chosen in data-weak directions; |z| fraction tracks the claimed width; R and Phi table added (prior_energy.json). Nuances found: truth atypical against draws only for Matern and q-GGMRF 0.05; in the Matern control the data are indifferent rather than in favor of the truth. |
| A1 candidate "q = 1 with T sigma_g >= log 2" | Disputed with evidence: it is the most blur-preferring member (ratio 4.50, prior_energy.json step_cost); the near-TV end (ratios 1.01 to 1.05) is the edge-indifferent candidate. |
| A1 optional q = 1 experiment | Deferred, no claim made (needs its own weight choice, pilot runs and a sampler check; outside the time budget). |
| A2 (major): features "data-determined" without the DESIGN minimum; GMRF contradicts ring continuity | Fixed. Verdict function and tests changed; gap valley and ring hole provisional, ring continuity not established; q-GGMRF settings counted as one family. |
| A3: DESIGN thresholds within noise; q-GGMRF 0.05 edge pixels not equilibrated | Fixed. Rank-normalized R-hat, bulk ESS and AR(1) null added; Matern extended to 600, q-GGMRF 0.05 to 900 (600-draw state missed R-hat Phi by 0.0003); flagged edge pixels gone; per-chain |z| fractions reported as Monte Carlo error. |
| A4: d_obs hash depends on the thread count | Fixed. Numerical criterion (max |delta d| / s <= 1e-6), thread environment stored in summary.json for new runs (pcg1_matern only; the four main runs rely on logs/*_prepare.meta.json, NPROC = 1), thread check run (6.5e-14), docstrings and tests amended. |
| A5: grid-error control not like for like | Fixed. Matern n = 48 same-grid control run; both controls at 3 x (300 + 600): 12.3% and 16.3%. |
| Process: thread cap in standalone scripts | Fixed in the repo (_threadcap, static test); the reproduction's one-off breach disclosed. |

## Numbers audit log

An independent numbers-vs-outputs audit traced 67 quantitative claims in this report to the raw outputs and recomputed the headline statistics from the stored arrays: 52 matched, 4 differed only in rounding, 7 were wrong or overstated and 4 had no stored source. All 13 text corrections it proposed are applied above; none changes a conclusion.
