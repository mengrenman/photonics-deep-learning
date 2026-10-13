# Milestone 6 (DESIGN 13): the resolution sweep, revised after review

Every number below is read from a file under `/Users/mengren/Documents/new_projects/photonics-dl/results/resolution/` (the file is named per item; `crb.json` means `results/resolution/crb.json`). Repository state: HEAD is bf5e76a (the DESIGN 13 commit), not d723799 as the task text said, and the working tree is not clean (the builder's changes to `priors.py`, `thetaspace.py` and their tests, the milestone-6 modules and results, and this follow-up; `docs/DESIGN.md` is edited). The review reproduced every number it checked and found no arithmetic error; four headline interpretations did not survive. This version changes them, answers the seven minor findings, and adds the evidence each needs. What changed against the first report is in the review log at the end. Two parts of the follow-up did not finish within the budget (the MCMC reruns, sections 5 and 9) and are marked.

## 0. Result

**Feature definition (decision of 2026-10-13):** the headline gap-valley feature is the half-height strip (|y − c_y| ≤ r/2, `band_frac` 0.5); the full-height strip is a lenient bound only, because it can fire for a closed gap (item 7). `gapvalley.feature_geometry` now defaults to the half-height strip.

1. **Information (CRB, no pixel prior): the gap is resolvable in Smith's sense far below anything an image shows, and at that limit the bound is local.** sigma_CRB(g) at 30 dB is almost independent of g and of the contrast: 2.61e-4 to 4.04e-4 wavelengths over the nine swept gaps and three contrasts (`crb.json`, n = 128), and flat to better than 1 % (change of -0.63 %, +0.34 %, +0.30 % for eps_r 1.1, 2, 4) from g = 1e-4 to 3e-3 (3.20e-4, 2.58e-4, 2.65e-4; `smith_limit/crb.json`). The gap at which sigma_CRB(g) = g holds at 30 dB is **g\* = 3.20e-4 (eps_r 1.1), 2.58e-4 (eps_r 2), 2.65e-4 (eps_r 4) wavelengths**, i.e. lambda/3100, lambda/3900, lambda/3800 (`null.json`, `smith_gap_30db`; n = 128 and 256 agree to 0.1 %, `smith_limit/crb.json`, `convergence`). At those SNRs the nuisance parameters are determined to better than 1.3 % (sd(r)/r <= 7.4e-3, sd(eps_r)/(eps_r - 1) <= 1.3e-2; all 18 rows of the extended sweep are inside the local-regime rule, `smith_limit/crb.json`, `local_regime`), so g\* is a statement about a regime in which the bound can be read as one. The gap that a one-sided test of g = 0 against g > 0 detects (level 0.05, power 0.9) is 2.93 sigma_CRB(0) = **9.4e-4 (eps_r 1.1), 7.5e-4 (eps_r 2), 7.7e-4 (eps_r 4) wavelengths** (lambda/1068, lambda/1326, lambda/1291), 9.33 dB above the Smith SNR (`detection.json`, `at_zero`). F stays well conditioned as g -> 0 (condition number of F scaled to unit diagonal 37.9, 12.5, 102.7): no degeneracy of g with r, which would need the pair to merge, while the center distance 2 r + g never falls below 2 r.
2. **Required SNRs for the book's gaps are lower bounds, not required SNRs; the estimator attains them for g in the known model.** For sigma_CRB(g) = g the SNR is -14.5 / -15.3 / -14.9 dB at lambda/20, -20.6 / -21.1 / -19.6 dB at lambda/10 and -27.2 / -28.9 / -25.9 dB at lambda/4 for eps_r 1.1 / 2 / 4 (Table 2). There the model's own nuisance parameters are essentially undetermined: at eps_r 2 sd(r)/r = 0.07, 0.14, 0.39 and sd(eps_r)/(eps_r - 1) = 0.16, 0.31, 0.82 at lambda/20, lambda/10, lambda/4 (aggregate SNR over the 2400 data 18.5, 12.7, 4.9 dB; `crb.json`, `book_claims`). Only 4 of the 27 swept rows satisfy the local-regime rule (both relative sds <= 0.1: g = 0.0125 at all three contrasts and g = 0.025 at eps_r 2; `crb.json`, `local_regime`); none of the book's gaps does. The threshold-aware check (section 2, `mle.json`) finds that the five-parameter maximum likelihood estimator of the exactly known two-circle model nevertheless attains the bound for g: RMSE(g_hat)/sigma_CRB = **0.85 at lambda/20 and 1.09 at lambda/10** (48 noise realizations each; 1.01 at lambda/20 and 10 dB higher), with g_hat = 0 in 12.5 % and 8.3 % of the realizations; for r and eps_r the bound is attained at lambda/20 (1.10, 1.04) and exceeded at lambda/10 (1.53, 1.92). So the review's "cannot apply" is too strong for g and right for the nuisances. sigma_CRB = g means a 100 % error of the gap (median |g_hat - g|/g = 0.69 and 0.81), and the bound with the ellipticity of both disks as extra unknowns is 1.11 to 1.31 times larger (+0.9 to +2.3 dB, `detection.json`, `ellipticity`). The margin of 24.5 to 48.9 dB quoted in the first report is withdrawn. At the book's own SNRs (10 and 20 dB) the nuisance bounds ARE local (largest sd/value 1.5 % at 10 dB, `crb.json`, `book_claims`), and sigma_CRB(g)/g there is a statement about an estimator of the exactly known model: 0.054 / 0.017 at lambda/20, 0.028 / 0.009 at lambda/10, 0.0070 / 0.0022 at 2 lambda/5 (eps_r 2; 10 / 20 dB).
3. **Multiple scattering: per unit of scattered power it neither consistently adds nor removes gap information; at a fixed noise level the full model carries less because it scatters less.** At one noise level F_full/F_Born is 0.91 to 1.02 (eps_r 1.1), 0.42 to 0.68 (eps_r 2) and 0.21 to 0.50 (eps_r 4) for known nuisances, but this follows P_full/P_Born = 0.986 to 1.007, 0.650 to 0.688 and 0.241 to 0.317 (the Born model overestimates the scattered power by 1.5 times at eps_r 2 and 3 to 4 times at eps_r 4). With each model at its own 30 dB the ratio is 0.90 to 1.03 / 0.62 to 1.05 / 0.67 to 1.96 for known nuisances and 1.01 to 1.13 / 1.16 to 1.74 / 0.65 to 1.54 for unknown ones (`crb.json`, `table`; the Born table in section 1). "Multiple scattering does not add gap information; it only lowers it" is withdrawn.
4. **The pixel-posterior threshold is an image-domain limit set by the effective bandwidth and by delta, not a noise-limited information limit.** The valley depth of the MAP of either prior follows that of an ideal circular low-pass of the truth at about 2.5 cycles per wavelength (eps_r 2, strip of half height: depth minus delta, low-pass +0.038 / +0.196 / +0.312 at g = 0.05 / 0.1 / 0.15, Matern MAP +0.032 / +0.149 / +0.239, TV MAP -0.042 / +0.076 / +0.172; closed gap -0.156 / -0.186 / -0.234 for the low-pass; `lowpass_check.json`), consistent with milestone 5 and not an anomaly. A 10 dB higher SNR changes the probabilities at the threshold gaps by at most 0.07, and in the direction of less resolution for TV (Table 8; `snr_check.json`). The image-domain threshold (0.1 to 0.15 wavelengths at eps_r 2 and 4) and g\* (2.6e-4 to 3.2e-4) are two differently defined limits, one for a posterior probability of a region mean exceeding a fixed delta, the other for the standard deviation of a parameter estimate; their ratio (380 to 580) is not a loss factor.
5. **Matern and TV disagree in the unresolved band** (0.025 to 0.1): the prior sets the answer there (Table 4). The Laplace probabilities agree with HMC for Matern (0.120 against 0.135 for the half strip at g = 0.025, 1.000 against 1.000 at g = 0.1) and are not reliable for TV near the threshold (half strip g = 0.1: 0.990 by MCMC against 0.868 by Laplace; the TV chains are not stationary, every TV chain's Phi' falls by about 5e4 per 50 draws, `mcmc_review.json`). No MCMC run meets the extended convergence criteria (section 5).
6. **Narrowest gap that discriminates (both priors, P_open > 0.95 and P_null < 0.05): eps_r 2: 0.1 to 0.15; eps_r 4: 0.15 (0.1 for the full strip); eps_r 1.1: none up to 0.15 (TV alone from 0.1, robust to lambda*/3 and 3 lambda*; Matern only at 0.5 for the half strip).** eps_r 2 is 0.15 by the Laplace rule and 0.1 if the original TV MCMC value 0.990 replaces the Laplace 0.868 at g = 0.1; the TV weight 3 lambda* also gives 0.967 there (`verdict_range.json`, `lambda_sensitivity.json`).
7. **Decision for the user: which strip is the feature.** The design's full-height strip (kept as the default per the 2026-10-12 decision) includes the lens-shaped void above and below the neck and is satisfied by a closed gap (Matern, eps_r 2: P_null = 0.88 / 0.81 / 0.28 for strips of 0.025 / 0.05 / 0.1); a strip of half height r/2 is not (P_null <= 0.26 under Matern, <= 0.03 under TV at eps_r 1.1; <= 0.001 at eps_r 2 and 4; `null.json`). I recommend r/2 as the headline and the full strip as the lenient bound.

## 1. Fisher information and CRB (`crb.json`, `crb/`, `smith_limit/crb.json`, `born_power.json`, `detection.json`; figures `crb_sigma_over_g.png`, `crb_required_snr.png`, `crb_born_ratio.png`, `crb_grid_convergence.png`)

Setup (`crb.json`, `setup`): two disks, r = 0.2, common eps_r, side 1.5, book rings (40 transmitters at r = 4, 60 receivers at r = 4.5), 30 dB power SNR of the scattered data (DESIGN 4), exact boundary derivatives of chi, D = jvp of the forward map, F = (2/s^2) Re(D^H D), s from the object's own data. Grids n = 128 (primary; the smallest gap is 1.07 pixels) and n = 256 (2.1 pixels). The numbers of the first report are unchanged (the aggregate re-run reproduces every old field to 0).

**Table 1.** sigma_CRB(g)/g at 30 dB, the SNR at which sigma_CRB(g) = g (nuisances unknown; * = not in the local regime: sd(r)/r or sd(eps_r)/(eps_r - 1) at that SNR above 0.1), and the nuisance bounds there for eps_r 2 (`crb.json`, `table`).

| g | sigma/g 1.1 | sigma/g 2 | sigma/g 4 | SNR_req 1.1 (dB) | SNR_req 2 | SNR_req 4 | eps_r 2: sd(r)/r, sd(eps_r)/(eps_r-1) at SNR_req |
|---|---|---|---|---|---|---|---|
| 0.0125 | 2.50e-02 | 2.09e-02 | 2.14e-02 | -2.0 | -3.6 | -3.4 | 0.02, 0.04 |
| 0.025 | 1.23e-02 | 1.06e-02 | 1.09e-02 | -8.2* | -9.5 | -9.2* | 0.04, 0.08 |
| 0.05 | 5.95e-03 | 5.41e-03 | 5.72e-03 | -14.5* | -15.3* | -14.9* | 0.07, 0.16 |
| 0.1 | 2.94e-03 | 2.79e-03 | 3.30e-03 | -20.6* | -21.1* | -19.6* | 0.14, 0.31 |
| 0.15 | 2.04e-03 | 1.89e-03 | 2.53e-03 | -23.8* | -24.5* | -21.9* | 0.21, 0.48 |
| 0.2 | 1.62e-03 | 1.42e-03 | 2.02e-03 | -25.8* | -26.9* | -23.9* | 0.30, 0.65 |
| 0.25 | 1.37e-03 | 1.13e-03 | 1.60e-03 | -27.2* | -28.9* | -25.9* | 0.39, 0.82 |
| 0.32 | 1.12e-03 | 8.79e-04 | 1.19e-03 | -29.0* | -31.1* | -28.5* | 0.51, 1.03 |
| 0.5 | 6.58e-04 | 5.69e-04 | 6.91e-04 | -33.6* | -34.9* | -33.2* | 0.70, 1.47 |

(eps_r 1.1 and 4 have the same pattern, e.g. 0.12, 0.20 and 0.12, 0.20 at g = 0.05; 1.07, 1.72 and 1.07, 1.88 at g = 0.5.) SNR_req = 30 + 20 log10(sigma_CRB(30 dB)/g) in closed form (F is proportional to 1/s^2); recomputing the bounds at SNR_req reproduces sigma/g = 1 to 5.3e-15 for the 33 checked records (`required_snr_check`), which checks the scaling algebra, not the physics. n = 256 gives the same table to within 0.025 dB. The unknown-to-known factor of sigma_CRB(g) is 1.17 (eps_r 2, g = 0.0125) to 2.54 (eps_r 4, g = 0.15); the condition number of the scaled F is 12.5 to 375.

**Table 3. The Smith limit measured directly** (`smith_limit/crb.json`, gaps 1e-4 to 3e-3, n = 128; all rows local at their required SNR).

| g | sigma/g 1.1 | sigma/g 2 | sigma/g 4 | SNR_req 1.1 | SNR_req 2 | SNR_req 4 |
|---|---|---|---|---|---|---|
| 1e-4 | 3.198 | 2.577 | 2.647 | 40.1 | 38.2 | 38.5 |
| 2e-4 | 1.599 | 1.289 | 1.324 | 34.1 | 32.2 | 32.4 |
| 3e-4 | 1.066 | 0.859 | 0.883 | 30.6 | 28.7 | 28.9 |
| 5e-4 | 0.639 | 0.516 | 0.530 | 26.1 | 24.2 | 24.5 |
| 1e-3 | 0.319 | 0.258 | 0.265 | 20.1 | 18.2 | 18.5 |
| 3e-3 | 0.106 | 0.086 | 0.089 | 10.5 | 8.7 | 8.9 |

sigma_CRB(g) is 3.198e-4 / 2.577e-4 / 2.647e-4 at g = 1e-4 and changes by -0.63 % / +0.34 % / +0.30 % at 3e-3: the bound is the localization precision of the two boundaries, which does not depend on how far apart they are, so "resolution" in Smith's sense for a known-shape model measures position precision, not whether an image can show a gap. Grid convergence of this sweep: max relative difference of sigma 1.0e-3, of the required SNR 0.009 dB. The CRB is local in the interior of the parameter space; at g close to sigma the constraint g >= 0 truncates the estimator (g_hat = 0 in 8 to 12.5 % of the realizations at sigma_CRB = g, section 2).

**Born comparison, two normalizations** (`crb.json`, `table`, with `born_power.json`: P_full recomputed agrees with the records to 4.4e-16). Ranges over the nine gaps:

| eps_r | P_full/P_Born | same noise level: known / unknown | each model at its own 30 dB: known / unknown |
|---|---|---|---|
| 1.1 | 0.986 to 1.007 | 0.91 to 1.02 / 1.01 to 1.13 | 0.90 to 1.03 / 1.01 to 1.13 |
| 2 | 0.650 to 0.688 | 0.42 to 0.68 / 0.77 to 1.16 | 0.62 to 1.05 / 1.16 to 1.74 |
| 4 | 0.241 to 0.317 | 0.21 to 0.50 / 0.17 to 0.37 | 0.67 to 1.96 / 0.65 to 1.54 |

Example, eps_r 4, g = 0.0125: same-s known ratio 0.24 against P_full/P_Born = 0.241, matched 1.01. The matched ratio is the same-s ratio times P_Born/P_full (independent computation with the Born data at their own noise level in `tests/test_fisher.py`). Per unit of scattered power: no change at eps_r 1.1; at eps_r 2 less information about g when the nuisances are known (0.62 to 1.05) and more when they are unknown (1.16 to 1.74); at eps_r 4 mixed in both. Symmetry (all 27 objects; `crb.json`, `symmetry`): cross-block residual of F under x -> -x 2.2e-13, under y -> -y 3.9e-13, column parity 4.2e-11 (solver tolerance 1e-10, max primal residual 9.6e-11); the asymmetric negative control is a unit test. Grid (n = 128 against 256, `convergence.max`): sigma unknown 2.9e-3, known 1.2e-2, Born ratio 2.3e-2, required SNR 0.025 dB.

**Detection limit and identifiability at g = 0** (`detection.json`, `at_zero`; n = 128, touching disks, exact one-sided derivative along g; agrees with the g = 1e-4 sweep to 2.4e-4, 1.2e-4, 1.1e-4):

| eps_r | sigma_CRB(0) (nuisances unknown) | with nuisances known | condition number of scaled F | information correlation g-r / g-eps_r | correlation of the estimates g_hat-r_hat / g_hat-eps_hat | detectable gap 2.93 sigma |
|---|---|---|---|---|---|---|
| 1.1 | 3.199e-4 | 1.793e-4 | 37.9 | 0.404 / -0.074 | -0.827 / 0.790 | 9.36e-4 (lambda/1068) |
| 2 | 2.577e-4 | 2.212e-4 | 12.5 | 0.157 / -0.173 | -0.490 / 0.494 | 7.54e-4 (lambda/1326) |
| 4 | 2.647e-4 | 1.731e-4 | 102.7 | 0.424 / 0.220 | -0.742 / 0.692 | 7.75e-4 (lambda/1291) |

F is not singular as g -> 0 (center distance 2 r + g >= 0.4), so the bound is meaningful at small g apart from the boundary g >= 0; the test is the one-sided Wald/likelihood-ratio approximation on a boundary. **Ellipticity** (`detection.json`, `ellipticity`; both disks become ellipses with semi-axes r + s_x, r + s_y, gap fixed, `shapes.stretch_derivatives`): sigma_CRB(g) with s_x and s_y unknown against the circular model: 1.110 (eps_r 2, g 0.05), 1.306 (1.1, 0.05), 1.147 (4, 0.15), 1.256 (2, 0.25), 1.115 (2, 0), i.e. +0.90, +2.32, +1.19, +1.98, +0.95 dB; either stretch alone gives the same bound (the span of {r, s_x, s_y} has dimension 2). General shape freedom is not covered.

## 2. Monte Carlo maximum-likelihood check of the bound (`mle.json`, `mle_check.png`, chunks in `mle/`)

Data and fit on the n = 64 grid of the Laplace sweep with the exact-coverage rendering (the model is the true one; the bound is recomputed on that grid: sigma_CRB(g) 0.0500 at the lambda/20 point, 0.1000 at lambda/10), the book rings, SNR from sigma_CRB(g) = g on that grid. Five-parameter Levenberg-Marquardt maximum likelihood (`bayes_scatter/mle.py`), once from the truth and from 2 (1 for the third point) starts drawn from a box that does not depend on the truth; the best Phi is the "global" estimate.

| point | SNR (aggregate) | M | sigma_CRB(g) | sd(g_hat) | bias(g_hat) | RMSE/CRB g | RMSE/CRB r / eps_r | g_hat = 0 | g_hat > g/2 | coverage of +-1.96 sigma_CRB (g / r / eps_r) |
|---|---|---|---|---|---|---|---|---|---|---|
| lambda/20, eps_r 2 | -15.32 dB (18.5 dB) | 48 | 0.0500 | 0.0424 | +0.0069 | 0.85 | 1.10 / 1.04 | 12.5 % | 71 % | 1.00 / 0.96 / 0.96 |
| lambda/10, eps_r 2 | -21.06 dB (12.7 dB) | 48 | 0.1000 | 0.0994 | +0.0473 | 1.09 | 1.53 / 1.92 | 8.3 % | 77 % | 0.94 / 0.81 / 0.77 |
| lambda/20, +10 dB | -5.32 dB (28.5 dB) | 24 | 0.0158 | 0.0162 | +0.0025 | 1.01 | 1.04 / 0.99 | 0 % | 100 % | 0.92 / 0.92 / 0.92 |

The sd of 48 realizations has a sampling error of about 10 %. Findings: (i) every start of every realization reaches the same optimum (no start has a Phi lower than the truth start's by more than 1; the 17 % of lambda/10 realizations with a start that stopped at the iteration cap end at the same optimum), so there is no second basin at these points in this box; (ii) the estimator attains the bound for g at both required SNRs and at +10 dB; the lower sd at lambda/20 comes from the truncation g >= 0; (iii) at lambda/10 the nuisance estimates are poor (eps_r biased by +0.32, one sigma_CRB; r and eps_r RMSE 1.5 to 1.9 times the bound, coverage 0.77 to 0.81), the regime where the review's concern holds. What this does not cover: a model with unknown shapes (the ellipticity bound above), 3-D data, or a reconstruction without a parametric model.

## 3. The book's operating points (corrected) against the bound

Source: Ch. 15, Sec. 15.4.2.1, pp. 464 to 467. **Test #4** (comparison with VBIM): six objects, adjacent spacings 5/20, 4/20, 3/20, 2/20, 1/20 wavelength, permittivity 3.2 down to 2.2 in steps of 0.2; **no noise level is stated**. **Tests #5 to #8**: nine connected objects with permittivity 2.2 to 3.8, spacings 4/10, 3/10, 2/10, 1/10 wavelength, in noise-free, 20 dB and 10 dB environments (Table 15.9). So "lambda/20 at 10 to 20 dB" is not a book operating point; the noisy points are lambda/10 to 2 lambda/5. Geometry (`book_geometry.json`): 500 MHz (0.6 m); 128 transmitters in planes at z = -0.1 and 0.7 m, 162 receivers at -0.2 and 0.8 m, objects 0.1 m high about z = 0.3 m, so the antenna planes are 0.35 m (0.583 wavelength) and 0.45 m (0.75 wavelength) from the objects; 20,736 data against 2400 here (8.64 times, +9.37 dB of aggregate SNR at equal per-datum SNR). Over the nearest standoff an evanescent component decays per path by e^-6.3 at 2 cycles/lambda (1.7e-3), e^-18.0 at 5 cycles/lambda (the half period of a lambda/10 gap) and e^-36.5 at 10 cycles/lambda (lambda/20); in this benchmark the far rings (3.575 wavelengths from the pair at the nearest) give e^-38.9 already at 2 cycles/lambda. So the near-field standoff of the book does not supply the missing spatial frequencies either, and the earlier "near-field / evanescent" explanation of the 2-D versus book difference is dropped. What differs is the 3-D geometry, 8.64 times more data, objects of unequal permittivity or connected, and a trained network with a learned prior.

**Table 2.** Required SNR (dB) for the book's gaps, nuisances unknown (known in brackets), and the nuisance bounds there for eps_r 2 (`crb.json`, `book_claims`). No gap is local at its required SNR.

| gap (book test) | eps_r 1.1 | eps_r 2 | eps_r 4 | eps_r 2: sd(r)/r, sd(eps_r)/(eps_r-1), aggregate SNR |
|---|---|---|---|---|
| 0.05 (#4) | -14.5 (-18.4) | -15.3 (-17.0) | -14.9 (-20.0) | 0.07, 0.16, 18.5 dB |
| 0.1 (#4, #5-8) | -20.6 (-23.7) | -21.1 (-23.2) | -19.6 (-27.1) | 0.14, 0.31, 12.7 dB |
| 0.15 (#4) | -23.8 (-26.7) | -24.5 (-26.8) | -21.9 (-30.0) | 0.21, 0.48, 9.3 dB |
| 0.2 (#4, #5-8) | -25.8 (-29.0) | -26.9 (-29.4) | -23.9 (-30.7) | 0.30, 0.65, 6.9 dB |
| 0.25 (#4) | -27.2 (-31.1) | -28.9 (-31.2) | -25.9 (-31.1) | 0.39, 0.82, 4.9 dB |
| 0.3 (#5-8) | -28.5 (-33.1) | -30.6 (-32.6) | -27.8 (-32.1) | 0.48, 0.98, 3.3 dB |
| 0.4 (#5-8) | -31.1 (-36.4) | -33.1 (-34.6) | -31.0 (-34.5) | 0.62, 1.23, 0.7 dB |

At the book's own SNRs (Tests #5 to #8; sigma_CRB(g)/g, nuisances unknown; the bound is local there: largest nuisance sd/value 4.6e-3 at 20 dB and 1.5e-2 at 10 dB):

| gap | eps_r 1.1 (20 / 10 dB) | eps_r 2 | eps_r 4 |
|---|---|---|---|
| 0.1 | 0.0093 / 0.0294 | 0.0088 / 0.0279 | 0.0104 / 0.0330 |
| 0.2 | 0.0051 / 0.0162 | 0.0045 / 0.0142 | 0.0064 / 0.0202 |
| 0.3 | 0.0038 / 0.0119 | 0.0030 / 0.0094 | 0.0041 / 0.0128 |
| 0.4 | 0.0028 / 0.0089 | 0.0022 / 0.0070 | 0.0028 / 0.0089 |

For Test #4 (no noise level) sigma_CRB/g at 30 dB is 5.95e-3, 2.94e-3, 2.04e-3, 1.62e-3, 1.37e-3 (eps_r 1.1) and 5.41e-3, 2.79e-3, 1.89e-3, 1.42e-3, 1.13e-3 (eps_r 2) at gaps 0.05 to 0.25, scaled by 10^(-(SNR - 30)/20) for another SNR (0.054 and 0.017 at 0.05 for eps_r 2 at 10 and 20 dB). These are statements about a perfectly known two-disk model in 2-D with far rings, not a test of the book's claims.

## 4. Pixel posteriors (`laplace.json`, `laplace/`, `laplace_lambda.json`, `laplace_sensitivity.json`, `null.json`; figures `laplace_pvalley.png`, `laplace_verdict.png`, `laplace_maps.png`, `laplace_profiles.png`, `laplace_lambda.png`, `null_control.png`, `resolution_overlay.png`)

Inference grid n = 64 (h = 0.0234), data from n = 128 at 30 dB, seed 0 (one noise realization), Matern (0.5, 0.4, 1.5) and TV (delta 0.01, sigma_0 1) at lambda*, MAP by damped Gauss-Newton, dense Laplace covariance, exact bivariate-normal orthant probability. 84 configurations: 54 primary (9 gaps x 3 contrasts x 2 priors), 18 TV runs at lambda*/3 and 3 lambda* (eps_r 2) and the 12 new ones at eps_r 1.1 and 4 (`laplace.json`); all MAPs succeeded (Phi_MAP 2388.7 to 2410.6 at the primary configurations against N = 2400). lambda* = 10 / 0.5 / 2 for eps_r 1.1 / 2 / 4 (interpolated crossings 13.9 / 0.35 / 3.1; `laplace_lambda.json`); the rule hardly fixes lambda (every swept value but one is inside N +- sqrt(N)). g = 0.0125 is narrower than a pixel (0.53 h): unrepresentable. A region probability is also withheld when the pixelized truth fails the feature: this affects 4 of 144 null rows (eps_r 4, g = 0.025, quarter-height strip; stored value 0.000 in each, kept as `p_open_raw` / `p_null_raw` in `null.json`) and the 2 matching sensitivity configurations; no verdict changes.

**Table 4. Gap-valley probability, open gap / closed gap (null)** (Laplace; P(strip mean < min(core means) - delta), delta = log(eps_r)/2, cores of 0.5 r; `null.json`, `rows`; unchanged from the first report).

| eps_r | g | full strip, Matern | full strip, TV | r/2 strip, Matern | r/2 strip, TV |
|---|---|---|---|---|---|
| 1.1 | 0.025 | 0.62 / 0.49 | 0.99 / 0.86 | 0.36 / 0.26 | 0.30 / 0.03 |
| 1.1 | 0.05 | 0.71 / 0.47 | 1.00 / 0.84 | 0.45 / 0.24 | 0.78 / 0.02 |
| 1.1 | 0.1 | 0.88 / 0.37 | 1.00 / 0.69 | 0.62 / 0.17 | 1.00 / 0.00 |
| 1.1 | 0.15 | 0.96 / 0.19 | 1.00 / 0.17 | 0.75 / 0.08 | 1.00 / 0.00 |
| 2 | 0.025 | 1.00 / 0.88 | 0.60 / 0.10 | 0.14 / 0.00 | 0.02 / 0.00 |
| 2 | 0.05 | 1.00 / 0.81 | 0.95 / 0.10 | 0.72 / 0.00 | 0.16 / 0.00 |
| 2 | 0.1 | 1.00 / 0.28 | 1.00 / 0.05 | 1.00 / 0.00 | 0.87 / 0.00 |
| 2 | 0.15 | 1.00 / 0.00 | 1.00 / 0.01 | 1.00 / 0.00 | 1.00 / 0.00 |
| 4 | 0.025 | 0.00 / 0.00 | 0.04 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 | 0.05 | 0.17 / 0.00 | 0.99 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 | 0.1 | 1.00 / 0.00 | 1.00 / 0.00 | 0.06 / 0.00 | 0.32 / 0.00 |
| 4 | 0.15 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |

Open-gap probabilities for wider gaps (half strip, `laplace_sensitivity.json`): eps_r 1.1 Matern 0.85 / 0.90 / 0.92 / 0.98 at g = 0.2 / 0.25 / 0.32 / 0.5, TV 1.00 throughout; eps_r 2 and 4: 1.00 for both priors from 0.2 on. The pixelized truth of a closed gap satisfies the full-strip definition for strips of 0.025, 0.05, 0.1 at all contrasts and fails it at 0.15, and fails every r/2 and r/4 strip (`null.json`, `truth_holds_null`); at eps_r 4 only the continuation start of the MAP converges, so unimodality there is not shown (`null/*.json`).

**Verdicts** (`null.json`, `narrowest_discriminating_gap`; both priors, P_open > 0.95 and P_null < 0.05 at that strip gap and every wider swept gap up to 0.15): full strip: eps_r 1.1 none, eps_r 2 0.15, eps_r 4 0.1; r/2 strip: none, 0.15, 0.15. eps_r 2 half strip with the MCMC cross-check (`verdict_range.json`): Laplace 0.15; 0.1 if the original TV MCMC probability at g = 0.1 (0.990 +- 0.007, 4 chains x 50 draws, unconverged) replaces the Laplace 0.868. Report 0.1 to 0.15, set by the TV weight and by the Laplace-versus-MCMC difference. Matern against TV: at eps_r 1.1 TV is decisive from 0.1 (half strip 1.00, null 0.00) while Matern reaches 0.75 only at 0.15 and 0.975 at 0.5; at eps_r 2 Matern is more resolving at 0.05 (0.72 against 0.16) and 0.1 (1.00 against 0.87); at eps_r 4 both are 0 up to 0.05 and 1.00 at 0.15.

**Table 7. TV weight sensitivity at the other contrasts** (`lambda_sensitivity.json`; half-height strip, Laplace; lambda*/3, lambda*, 3 lambda*; lambda* = 10, 0.5, 2; all MAPs within 0.997 to 1.006 N).

| eps_r | g = 0.05 | g = 0.1 | g = 0.15 |
|---|---|---|---|
| 1.1 | 0.625 / 0.781 / 0.910 | 0.974 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| 2 | 0.344 / 0.162 / 0.032 | 0.827 / 0.868 / 0.967 | 0.992 / 1.000 / 1.000 |
| 4 | 0.000 / 0.000 / 0.000 | 0.801 / 0.320 / 0.215 | 1.000 / 1.000 / 1.000 |

The eps_r 1.1 TV verdict ("decisive from 0.1") holds for lambda within a factor 3 of lambda* = 10 (0.974 at lambda*/3), and the eps_r 4 verdict (0.15) too (0.80 is the largest value at g = 0.1). The 3-fold change in lambda moves the unresolved band by about a factor 1.5 in g at eps_r 2. **The discrepancy target** (`peff.json`): the Matern MCMC runs give p_eff = 2 (E[Phi] - Phi_MAP) = 74.1 (g = 0.025, 4 x 300 draws) and 74.8 (g = 0.1, 4 x 200), so E[Phi_MAP] is about N - p_eff/2 = 2362.8, not 2400 (sqrt(N) = 49: systematic but within the noise). Phi_MAP at every swept lambda is at least 2387.4 (eps_r 1.1, lambda = 0.1), so no swept value reaches the corrected target and the corrected rule would pick the lower end of the sweep (0.1, 0.1, 0.5 for eps_r 1.1, 2, 4), outside the tested lambda*/3 to 3 lambda* band: the TV verdicts are conditional on lambda within that band.

**Table 8. The image-domain threshold: low-pass and SNR** (eps_r 2, half-height strip, depth minus delta of theta; positive = the valley holds; delta = 0.347; `lowpass_check.json`; open gap / closed gap).

| g | low-pass 2 | low-pass 2.5 | low-pass 3 | Matern MAP | TV MAP | closed gap: low-pass 2.5 / Matern MAP / TV MAP |
|---|---|---|---|---|---|---|
| 0.025 | -0.308 | -0.053 | +0.049 | -0.035 | -0.108 | -0.148 / -0.111 / -0.181 |
| 0.05 | -0.244 | +0.038 | +0.138 | +0.032 | -0.042 | -0.156 / -0.120 / -0.182 |
| 0.1 | -0.113 | +0.196 | +0.279 | +0.149 | +0.076 | -0.186 / -0.153 / -0.188 |
| 0.15 | +0.013 | +0.312 | +0.371 | +0.239 | +0.172 | -0.234 / -0.207 / -0.217 |
| 0.25 | +0.203 | +0.415 | +0.448 | +0.352 | +0.302 | -0.389 / -0.377 / -0.362 |

(Low-pass of the pixelized truth at 2, 2.5, 3 cycles per wavelength, `resolution.lowpass`, pad 4; the open and closed objects also differ in disk position, which the blurred image keeps.) The TV MAP depth sits below the 2.5-cycle value at every gap (0.05 to 0.14 below) and the Matern MAP depth from g = 0.05 on (0.01 to 0.07 below; at g = 0.025 it is 0.02 above); the Matern MAP crosses delta where the low-pass does (between 0.025 and 0.05) and the TV MAP later (between 0.05 and 0.1). A 10 dB higher SNR (`snr_check.json`, Laplace at 40 dB, eps_r 2, both priors, TV weight 0.5):

| g | strip | Matern 30 dB -> 40 dB | TV 30 dB -> 40 dB | MAP depth minus delta, Matern / TV, 30 -> 40 dB |
|---|---|---|---|---|
| 0.05 | half | 0.721 -> 0.661 | 0.162 -> 0.097 | +0.032 -> +0.023 / -0.042 -> -0.063 |
| 0.05 | full | 1.000 -> 1.000 | 0.948 -> 0.958 | |
| 0.1 | half | 1.000 -> 1.000 | 0.868 -> 0.844 | +0.149 -> +0.145 / +0.076 -> +0.070 |
| 0.15 | half | 1.000 -> 1.000 | 1.000 -> 1.000 | +0.239 -> +0.239 / +0.172 -> +0.178 |

The threshold gaps do not move with the noise (a factor 10 in noise power changes P by at most 0.07 and the TV half-strip probabilities go down; the TV full-strip value at g = 0.05 rises by 0.010). The 20 dB runs were cut from the queue (time) and are not reported. This is the image-domain statement of result 4.

## 5. MCMC spot checks (`mcmc.json`, `mcmc_review.json`, `mcmc_refeature.json`, `mcmc_refeature_truth5.json`, `mcmc_spotcheck.png`)

Laplace-mass HMC at eps_r 2, g = 0.025 and 0.1, Matern and TV(lambda*); four overdispersed chains (scale 2 Matern, 0.5 TV) and a fifth chain started at the truth. The convergence flag now has the classic DESIGN 11.2 criteria (split R-hat of the physical Phi <= 1.01, max pixel R-hat <= 1.05, worst pixel ESS >= 200) AND criteria on the sampler's own potential Phi' = U - |zeta|^2/2 (split R-hat <= 1.05, Sokal ESS summed over the chains >= 200): the review found the one "converged" run with Phi' R-hat 1.073 and per-chain ESS 15 to 23.

**Table 5.** Rows from `mcmc.json` (`table`, `with_truth_chain`); "skip 1" drops the first segment of every chain; P = full-strip probability.

| config | chains x draws | R-hat Phi | max pixel R-hat | worst pixel ESS | R-hat Phi' | ESS Phi' | converged: classic / with Phi' | P(valley) MCMC (+- MCSE) | Laplace |
|---|---|---|---|---|---|---|---|---|---|
| Matern g = 0.025 | 4 x 300 | 1.005 | 1.034 | 242 | 1.073 | 73 | yes / NO | 0.999 +- 0.001 | 1.000 |
| + truth chain | 5 x 300 | 1.022 | 1.383 | 260 | 1.806 | 80 | no / no | 0.999 | |
| + truth chain, skip 1 | 5 x 200 | 1.008 | 1.122 | 202 | 1.976 | 94 | no / no | 0.998 | |
| Matern g = 0.1 | 4 x 200 | 1.012 | 1.041 | 163 | 1.062 | 73 | no / no | 1.000 | 1.000 |
| + truth chain | 5 x 200 | 1.154 | 1.860 | 172 | 7.174 | 80 | no / no | 1.000 | |
| TV g = 0.025 | 4 x 200 | 1.046 | 2.040 | 34 | 2.072 | 28 | no / no | 0.780 +- 0.031 | 0.600 |
| + truth chain | 5 x 200 | 1.036 | 1.915 | 43 | 2.064 | 35 | no / no | 0.772 | |
| TV g = 0.1 | 4 x 50 | 1.027 | 3.322 | 33 | 2.475 | 30 | no / no | 1.000 | 1.000 |

With the criteria on Phi' **no run is converged**, including the four-chain Matern run at g = 0.025 that the first report called converged. Laplace against MCMC for the discriminating features (`mcmc_refeature.json`; four chains; MCMC / Laplace): Matern g = 0.025 half strip 0.120 / 0.135 (MCSE 0.013), quarter strip 0.050 / 0.061, half strip with cores of 0.25 r 0.595 / 0.621; Matern g = 0.1 1.000 / 1.000 and 0.998 / 0.994; TV g = 0.025 half strip 0.045 / 0.019, quarter 0.072 / 0.022; TV g = 0.1 half strip 0.990 / 0.868, quarter 0.905 / 0.783 (unconverged). With the truth chain: Matern 0.113 (g = 0.025) and 0.978 (g = 0.1, the chain had not relaxed). Pixel sd MCMC/Laplace, median: 0.99 (Matern, both gaps), 3.34 and 2.71 (TV).

**The truth chain's transient** (`mcmc_review.json`, `phi_prime_by_segment`; `mcmc.json`, `with_truth_chain.block_means`). The truth chain adapted a step of 0.108 (g = 0.025) and 0.098 (g = 0.1) against 0.16 to 0.21 and 0.17 to 0.19 for the overdispersed chains; the mean of Phi' by segment is -619, 1391, 2049 at g = 0.025 and -3942, -1554 at g = 0.1, against 1.9e3 to 2.1e3 in every segment of every other chain. Its physical Phi (2443 to 2435 against about 2436) and valley depth (0.554, 0.485, 0.452, 0.407, 0.424, 0.443 in blocks of 50 against 0.44 to 0.45) reach the other chains' values at g = 0.025; at g = 0.1 it had not relaxed in 200 draws (depth 0.723 falling to 0.516 against 0.59 to 0.60). So "unconverged with the truth chain" is a long transient plus slow mixing of the energy-like direction, not a second mode (at g = 0.025 the truth chain reaches the same physical Phi and depth). **Deferred:** the review's two remedies. Reusing the step of the overdispersed chains FAILS: with the step frozen at 0.1708 (g = 0.025) and 0.1847 (g = 0.1) the chain started at the truth made no progress (a probe accepted none of its trajectories (the probe output was not saved; DESIGN 13.6 records only 'acceptance 0'); the batch jobs ran 6640 s without finishing segment 0, `results/logs/m6r_mcmc_g0.025_eps2_matern_truthfrozen.meta.json`, killed), so the step the transient forces is not an artifact that a bulk step can replace. A warmup of several hundred adapting trajectories was not affordable.

**TV chains are not stationary** (`mcmc_review.json`). The mean of Phi' per segment of every TV chain of the first runs falls steadily, e.g. -9.1e4, -1.40e5, -1.91e5, -2.32e5 (g = 0.025, chain 0) while the Matern chains stay at 2.0e3; so 0.78 and 0.990 are indicative and the Laplace-versus-MCMC difference for TV is not settled.

**Follow-up runs (partial, deferred).** Launched: TV at g = 0.1 with overdispersed starts of scale 2 (four chains, 200 draws), TV at g = 0.15 (scale 2), Matern at g = 0.15. The TV chains adapted to a step of 0.013 to 0.024 (acceptance 0.94 to 1.00) during their transient and needed 80 to 110 CPU minutes for the first segment (the machine ran at load 15 to 31 from other users); within the budget only single chains finished (`mcmc_review.json`): TV g = 0.1: one chain of 200 draws and one of 50 (half-strip P = 1.000, valley depth 0.430 +- 0.021 against Laplace 0.423 +- 0.051; Phi' means -8.5e3, -1.78e4, -2.60e4, -3.46e4), TV g = 0.15: one chain of 150 draws (P = 1.000, depth 0.521 +- 0.015 against 0.519 +- 0.044). They have no R-hat, are not stationary (Phi' drifting by about -7e3 to -9e3 per 50 draws) and prove nothing about the TV probability at g = 0.1; the Matern g = 0.15 chains, a fourth TV chain and the 20 dB Laplace runs were dropped from the queue. The verdict range of result 6 does not rest on them.

## 6. Answers to the DESIGN 13.1 questions

- *Which gap widths do the data determine?* In the parametric sense (known two circles) all gaps down to 2.6e-4 to 3.2e-4 wavelengths at 30 dB (detectable against g = 0: 7.5e-4 to 9.4e-4), and by Smith's criterion gaps of lambda/20 to 2 lambda/5 already at -15 to -33 dB for a known model, where the maximum likelihood estimator attains the bound for g (section 2). In the image (pixel posterior) sense, gaps of 0.1 to 0.15 wavelengths at eps_r 2 and 4 for the half-height strip, not at eps_r 1.1 below 0.5 for Matern (TV from 0.1).
- *Where do pixel posteriors stop reflecting the data?* Below about 0.1 to 0.15 wavelengths; between 0.025 and 0.1 the answer is chosen by the prior and by the feature definition. The threshold is the effective bandwidth (about 2.5 cycles per wavelength) plus delta, and does not move with the noise (Table 8).
- *Does multiple scattering help?* Per unit of scattered power neither consistently adds nor removes gap information (result 3).
- *Does the book's lambda/20 survive?* The book gives no noise level for its lambda/20 (Test #4); its noisy points are lambda/10 to 2 lambda/5. As information about a model with known shapes, the 2-D bound is met by many dB at 10 to 20 dB (sigma_CRB/g 0.054 at lambda/20 and 10 dB); as a feature of a pixel reconstruction of far-ring 2-D data, lambda/20 is prior-dependent at 30 dB (half strip, eps_r 2: Matern 0.72, TV 0.16; eps_r 4: 0.00 / 0.00; eps_r 1.1: 0.45 / 0.78).

## 7. Honest limits

1. The CRB is local, model-based, for a known two-circle family in 2-D far rings. The Monte Carlo check covers one radius, eps_r 2 and two SNRs on the n = 64 grid (data and fit on the same grid); it does not cover unknown shapes beyond the two ellipticity parameters, 3-D, or a model-free reconstruction.
2. One noise realization (seed 0) for the Laplace sweep, one inference grid (n = 64), one radius. At eps_r 4 the n = 64 grid error is 0.22 to 0.35 of the noise (`laplace.json`, `operating_point.model_to_noise`), above the DESIGN 8.6 guideline of 0.25; the eps_r 4 posteriors are conditional on a visibly imperfect model, and only continuation starts reach a data-consistent MAP there.
3. The closed-gap null covers strip gaps 0.025 to 0.15 and one closed-gap realization per prior and contrast.
4. No MCMC run meets the extended convergence criteria; the TV chains are not stationary; the truth-chain remedies (frozen step, long warmup) were not completed; the g = 0.15 and the overdispersed TV checks exist only as single non-stationary chains.
5. The discrepancy rule does not fix lambda, and with the p_eff-corrected target it would choose a lambda below the tested band; the TV verdicts are conditional on lambda within lambda*/3 to 3 lambda* (eps_r 1.1, 2, 4 all tested).
6. The ideal low-pass reading is a consistency check against milestone 5, not a proof that the bandwidth is the only cause.
7. The compute budget was exceeded: the follow-up batch (`scripts/jobs/m6_review.json`, 37 jobs, 9 slots) ran from 08:56 to 10:55 (about 2 hours against the 90 minutes), because the machine ran at load 15 to 31 from other users (jobs took 2 to 3 times the first-pass times; at most 10 XLA threads of mine at any time) and the first MCMC segments needed 80 to 110 CPU minutes. Queued low-value jobs were killed on start to limit the overrun.

## 8. Code, tests, files

New or changed code: `src/bayes_scatter/fisher.py` (`matched_power_ratios`, `nuisance_report`, `detection_limit`, `power_born` in `analyze`), `shapes.py` (`stretch_derivatives`, `elliptic_chi`), `gapvalley.py` (`fits_grid`, `truth_holds`, `representable`, `status`), `mle.py` (new); `scripts/resolution_crb.py` (`BOOK`, `--book-gaps`, `--born-power`, nuisance and local-regime fields, corrected figures), `resolution_laplace.py` (`--names`, `--then-refeature`, withheld probabilities), `resolution_mcmc.py` (`energy_convergence`, `--stage energy`, `--run-tag`, `--frozen-step`, `--tail-segments`, `--runs`, log line names the start actually used), `resolution_mle.py`, `resolution_detect.py`, `resolution_review_analysis.py` (new), `make_m6_review_jobs.py` and `scripts/jobs/m6_review.json` (new); `docs/DESIGN.md` (13.1 to 13.4 corrected, 13.5 and 13.6 added, section 9 layout). Tests added: `tests/test_mle.py` (6), `tests/test_resolution_review.py` (8), and in `test_fisher.py` (matched-power Born ratio computed independently, nuisance report, detection limit), `test_shapes.py` (stretch derivatives against finite differences of ellipses), `test_gapvalley.py` (truth-fails representability), `test_resolution_scripts.py` (book constants and new CRB fields, Phi' criteria on synthetic chains, frozen-step truth chain with tail segments and the energy stage, withheld probabilities in the tables, `--then-refeature`). Tests that would have caught the code defects: the matched-power ratio test (the Born reading), the nuisance-report test (the local-regime rule), the truth-fails test (representability), the Phi' criteria tests (the "converged" run, which now fails them), the log-line assertion of the truth chain. Fast suite (`NPROC=4 XLA_FLAGS=--xla_cpu_multi_thread_eigen=true pytest -m "not slow"`): 611 passed, 5 deselected, 354 s; the three slow milestone-6 tests (`test_resolution_scripts.py -m slow`, NPROC = 1) pass (102 s); the two other deselected slow tests are older and were not run.

Commands as run (conda env photonics-dl): `python scripts/resolution_crb.py --grids 128 256 --book-gaps --born-power` (NPROC = 1, 33 + 27 forward-only objects and 6 new records), the same with `--aggregate-only` after the figure fixes and for `--out-dir results/resolution/smith_limit --gaps 0.0001 ... 0.003`; `python scripts/resolution_detect.py --grid 128`; `python scripts/make_m6_review_jobs.py && python scripts/batch_run.py scripts/jobs/m6_review.json --cpu-slots 9 --threads 1` plus manual `scripts/resolution_mle.py --stage chunk` runs in the tenth slot (chunk lock files prevent double computation); `python scripts/resolution_mle.py --stage collect`; `python scripts/resolution_mcmc.py --stage energy` and `--stage collect`; `python scripts/resolution_laplace.py --stage refeature`, `--stage null-collect`, `--stage collect`; `python scripts/resolution_review_analysis.py --stage all`. New data and figures: `crb.json` (extended), `born_power.json`, `detection.json`, `mle.json` and `mle/`, `mle_check.png`, `book_geometry.json`, `lowpass_check.json`, `snr_check.json`, `snr40/`, `lambda_sensitivity.json`, `peff.json`, `verdict_range.json`, `mcmc_review.json`, `mcmc/g0.1_eps2_tv_od2/` and `mcmc/g0.15_eps2_tv/` (partial chains), 12 new Laplace configurations, updated `crb_required_snr.png`, `crb_born_ratio.png`, `laplace.json`, `laplace_sensitivity.json`, `null.json`, `mcmc.json`. Logs: `results/logs/m6r_*.log` and `.meta.json` (the killed jobs are recorded as failed).

## 9. Review log (finding -> resolution)

1. **Major, required SNRs read from a bound where it cannot apply** -> fixed with evidence that refines the claim. Nuisance sds at SNR_req stored and flagged (`crb.json`): 4 of 27 rows local, none of the book's gaps; those rows are reported as lower bounds. Monte Carlo MLE (`mle.json`, section 2): the bound is attained for g (RMSE/CRB 0.85 and 1.09) and for r, eps_r at lambda/20, exceeded for r, eps_r at lambda/10 (1.53, 1.92); the 24.5 to 48.9 dB margin is withdrawn and replaced by sigma_CRB/g at the book's own SNRs, where the bound is local. The review's statement that the bound "cannot apply" is partly not borne out (it applies to g of the known model) and partly confirmed (nuisances).
2. **Major, Born ratio is a power scaling** -> fixed and the conclusion reworded (result 3; `crb.json`, `born_power.json`, `crb_born_ratio.png`); reproduced the review's ranges exactly.
3. **Major, book operating points misquoted, evanescent explanation unsupported** -> fixed after checking the book text: DESIGN 13.1 to 13.4, `BOOK`, figure, `book_geometry.json`; records at g = 0.3 and 0.4 added.
4. **Minor, image threshold is blur plus delta** -> fixed: `lowpass_check.json` reproduces the review's low-pass numbers; the 40 dB check shows no SNR dependence (`snr_check.json`); "factor 300 to 600" rephrased as two differently defined limits. 20 dB runs dropped for time.
5. **Minor, convergence story** -> (a) fixed: Phi' criteria, all summaries rewritten, no run converged, truth-chain transient documented by segment (`mcmc_review.json`); (b) deferred: the frozen-step rerun fails (acceptance 0 from the truth), the long-warmup rerun was not affordable.
6. **Minor, eps_r 2 verdict hinges on one TV probability** -> verdict reported as 0.1 to 0.15 (`verdict_range.json`); the extra MCMC checks finished only as single non-stationary chains (deferred).
7. **Minor, identifiability and detection limit** -> fixed: `detection.json` (g_det 7.5e-4 to 9.4e-4, well-conditioned F, ellipticity bound +0.9 to +2.3 dB); the estimates of g and r are correlated (-0.49 to -0.83), which the review's information correlation (0.16) does not show.
8. **Minor, lambda* weakly determined, sensitivity only at eps_r 2** -> fixed: 12 TV runs at eps_r 1.1 and 4 (`lambda_sensitivity.json`), p_eff = 74 and the corrected target (`peff.json`).
9. **Minor, representability ignores the truth** -> fixed in code and tables (4 of 144 rows withheld, no verdict changes), with tests.
10. **Minor, "flat to four digits"** -> fixed: flat to better than 1 % (0.30 to 0.63 %).
Also corrected from the first report's problem list: the truth-chain log line no longer prints "init scale 2.0"; DESIGN.md sections 9 and 13 are updated (they were not).
## Numbers audit log

An independent numbers-vs-outputs audit traced 44 quantitative claims to the raw outputs and recomputed the headline quantities (CRBs and required SNRs from the stored Fisher matrices, a Born ratio, a symmetry residual, Laplace gap-valley probabilities): 29 matched, 4 differed only in rounding, 8 were wrong or overstated and 3 had no stored source. 14 of its 14 text corrections are applied above.
