# Milestone 7 (DESIGN 14): parametric priors, right and wrong, revised after review

Every number below is read from a file under `/Users/mengren/Documents/new_projects/photonics-dl/results/parametric/` (named per item; `snr/x` means `results/parametric/snr/x`). `tables.md` holds the case tables, `snr/snr_summary.md` the SNR experiment, `ellipticity/*.md` the ladder and the calibration, `discretization/discretization.md` the grid-bias analysis. Repository state: HEAD is 07857dc ("Milestone 7 (in progress): parametric-prior runs and review follow-up", committed at your request on 2026-10-14; its parent 761fd7e is the milestone-7 build commit). The working tree was clean when this report was audited, and this report file is added after the commit. The commit contains: DESIGN 14.5 to 14.7, the changes to `shapepost.py`, `shapemodels.py` and the scripts, the new scripts, tests and job files, and `results/parametric/**`.

## 0. Result

**(i) Right family: the interval width is calibrated; the location carries a deterministic grid bias of 1.0 sd at 30 dB, which is itself an unflagged model error.** T1 gives g = 0.05022 +- 0.00028 and 0.10029 +- 0.00028 (HMC, converged), but these are one data realization (noise key 0; the correlation of z between the two gaps over 41 keys is 0.985). Over 41 noise keys the sd of z is 1.00 +- 0.11 and 0.98 +- 0.11, the mean z is -1.15 +- 0.16 and -1.19 +- 0.16 and the 90 % interval covers 66 % and 68 % (`ellipticity/calibration_g0.05.md`, `calibration_g0.1.md`). The offset is not noise: F2 fitted to noise-free data of the n = 128 grid is off by +1.024 sd (+1.237 sd for n = 256 data, 1.29 extrapolated; +0.005 for n = 64 data), the paired shift between data grids is -1.019 +- 0.002 over 8 keys, and with the offset removed z has mean -0.13 +- 0.16, sd 1.00 and 88 % coverage (`discretization/discretization.md`). Fixed in wavelengths, it is 0.10 sd at 10 dB, 0.32 at 20 dB, 1.02 at 30 dB, 3.2 at 40 dB.

**(ii) Wrong family: confidently wrong.** F2 on T2, T3 and T5 gives intervals with sd 2.8e-4 that miss the true gap by 24 to 110 sd (7 to 33 widths of the 90 % interval), 0.0066 to 0.031 wavelengths (`tables.md` section 5; T2 g = 0.1 chains: z = 87). T4 (closed gap with a neck): g = 2.35e-6, 90 % interval [1.20e-7, 7.03e-6], P(g < g_det) = 1, the true value for the wrong reason (`T4_F2/profile.json`).

**(iii) What flags it: the misfit excess grows as the square of the gap error, so the strongest statement is reached at the book's 10 dB.** dPhi = c z^2, with c fixed by the kind of error and independent of the SNR: 2.6 to 2.9 (15 % ellipticity), 2.0 to 2.3 (ellipticity up to 5 %), 0.80 (third small disk, T5), 30 to 68 (unequal contrasts, T3), 0.09 to 0.11 (the grid error). A 3 sd misfit flag (147 nats) needs z of about 7.2 to 7.6, 8.0 to 8.6, 13.6, 1.5 to 2.2 and 37 to 41 respectively, at any SNR. At 10 dB the F2 interval on T5 excludes the true gap by 10.8 sd on average (8.3 to 13.1 sd) and by more than 3 sd in all 40 noise draws while the misfit excess is 97 nats = 2.0 sd; the 5 % flag fires in 28 of 40 draws (70 %) and the 3 sd flag in 7 of 40 (17.5 %) (`snr/snr_summary.md`). At 20 dB every wrong case is flagged at 3 sd in every draw. At 30 dB a 1 % ellipticity moves the gap by 4.9 sd (interval excludes the truth by more than 3 sd in 39 of 40 draws) and the misfit flags it at 5 % in 17 of 40 and at 3 sd in 1 of 40; at 2 % the flag is certain. The first report's sentence "the data flag a model error once it biases the gap by about 2 sigma, never later", and its conclusion that the strongest statement is not reached, were wrong.

**What changed from the first report** (details in the review log): (a) section 4 is rewritten with the scaling law, the analytic null and fits at 10, 20 and 30 dB over many noise keys; (b) the 1.2 sd offset of (i) is explained as a grid bias and "exact forward models" is withdrawn; (c) the TI check, which came out 15 to 27 nats above Laplace, was a quadrature artifact; importance sampling confirms the Laplace evidence in all 15 interior cases it was run on (F2 and F2u on seven truths, Fe on T4) to 0.03 nats (finite-difference Hessian) or 0.09 nats (Gauss-Newton, one case 0.28). Smaller: the wrong gaps are 24 to 110 sd, not 20 to 300; the Occam factor is +9.8 to +10.4; two HMC cases are one realization; three chain sets flagged unconverged sit at the noise floor of perfect mixing.

## 1. Scope

- **r/2:** all pixel posteriors use the half-height strip (|y - c_y| <= r/2, `band_frac` 0.5; decision of 2026-10-13 in DESIGN 13.3); the strip, the representability test and delta are in `<truth>_pixel/summary.json`.
- **Ran:** all 24 cases (8 truths x F2, F2u, Fe) to the MAP with Laplace evidence (`<case>/map.json`, `tables.md` section 5); HMC (4 chains x 500 draws, 150 warmup) for T1 (both gaps) x {F2, Fe}, T2 g = 0.1 x {F2, Fe}, T4 x Fe; T4 x F2 by the profile marginal; T3 g = 0.05 x F2u with 4 of 5 segments (4 x 400 draws, `T3g0.05_F2u/chains_partial.json`); pixel posteriors (Matern, TV at lambda* = 0.5, Laplace) for all 8 truths; TI (two one-chain runs) on T1 g = 0.05 and T2 g = 0.1 with the beta = 1 temperature from the production chains. **Review follow-up:** importance-sampling evidence for 15 cases, F2/F2u fits at 10, 20, 30 dB, 30 dB ladder rungs over 40 keys, noise-free grid-bias fits, a refined T4 profile, regenerated summaries.
- **Dropped for time in the first run (unchanged):** HMC for T2 g = 0.05, T3 (F2, Fe), T5 and the F2u/Fe chains of the other truths (MAP, Laplace and now IS evidence stand in); the last TI temperature as a TI temperature; a third TI case.

## 2. (i) Right family

**Single data set (noise key 0, HMC 4 x 500; `T1g0.05_F2/summary.json`, `T1g0.1_F2/summary.json`):**

| truth | g mean +- sd | 90 % interval | z of truth | R-hat max classic / rank, worst bulk ESS, acceptance | refit predictive p (100 replicates) |
|---|---|---|---|---|---|
| T1 g = 0.05 | 0.050224 +- 0.000275 | [0.049768, 0.05068] | -0.8 | 1.002 / 1.010, 1456, 0.72 | 0.21 |
| T1 g = 0.10 | 0.10029 +- 0.000277 | [0.099836, 0.10076] | -1.0 | 1.003 / 1.009, 1612, 0.72 | 0.21 |

Both are unimodal, P(g < g_det) = 0 (g_det = 7.5e-4), Phi_MAP/N = 1.015; the sd 2.8e-4 matches the milestone-6 bound for the known two-circle model at eps_r 2 (2.70e-4 at g = 0.05 and 2.79e-4 at g = 0.1; 2.6e-4 to 2.8e-4 over g = 0.0125 to 0.15; `results/resolution/crb.json`, outside `results/parametric/`); Laplace, profile and chain marginals agree (0.050224 +- 0.000271 against +- 0.000275, `T1g0.05_F2/profile.json`). **They are one data realization** (both use noise key 0 and the same chain keys; correlation of z between the gaps over 41 keys 0.985): a single covering interval is weak evidence, and the statement rests on the 41-key calibration plus the agreement of the Laplace Gaussian with the chains and with importance sampling (weight ESS 788 of 1000, `T1g0.05_F2/is_evidence.json`). T3, where F2u is the right family: g = 0.0500709 +- 0.000276 from 4 x 400 draws (`T3g0.05_F2u/chains_partial.json`), Laplace 0.050068, Phi/N 1.014.

**Calibration over 41 noise keys (F2 Laplace):**

| gap | sd of z (width) | mean z (location) | bias of g | 90 % coverage |
|---|---|---|---|---|
| 0.05 | 1.00 +- 0.11 | -1.15 +- 0.16 | +3.1e-4 | 0.66 (se 0.07; 0.05 if the true coverage were 0.90) |
| 0.10 | 0.98 +- 0.11 | -1.19 +- 0.16 | +3.3e-4 | 0.68 (se 0.07; 0.05 if the true coverage were 0.90) |

**The offset is a deterministic forward-model bias** (`discretization/discretization.md`, `disc_g*_n*.json`): data on the n = 128 grid, inversion on n = 64. F2 fitted to noise-free T1 data, 30 dB noise level:

| data grid | g* - g (g = 0.05) | shift (sd) | Phi_MAP | model-error energy at the truth |
|---|---|---|---|---|
| n = 64 | +1.26e-06 | +0.005 | 0.000 | 0.00 |
| n = 128 | +2.78e-04 | +1.024 | 0.112 | 8.85 |
| n = 256 | +3.35e-04 | +1.237 (1.29 extrapolated) | 0.134 | 13.30 |

At g = 0.1 the shifts are +0.002, +1.057, +1.298 sd. Paired by noise key, z(n = 128 data) - z(n = 64 data) = -1.019 +- 0.002 over 8 keys (a constant: the first report called this attribution "weak (8 seeds)"). With the offset removed, the 41 noise-driven z have mean -0.13 +- 0.16 and sd 1.00 (0.98 at g = 0.1) and the shifted interval covers 88 % at both gaps. F2 absorbs the error almost completely (Phi_MAP 0.112 against 8.85), so it raises no flag yet biases g by 1.0 sd: a model error the family absorbs "benignly" for the fit and not for the gap. In sd units it scales as 10^((SNR - 30)/20): 0.10 (10 dB), 0.32 (20), 1.02 (30), 3.2 (40), 10 (50). At the book's SNRs the right-family fit is unbiased at the level of the noise: over 40 keys the mean z is -0.3 (10 dB) and -0.5 (20 dB) and the 3 sd interval covers the truth in 40 of 40 draws (`snr/snr_summary.md`, T1g0.05). The "exact forward models" caveat of the first report is withdrawn.

## 3. (ii) Wrong family

**Chains (T2 g = 0.1, `T2g0.1_F2/summary.json`):** g = 0.075972 +- 0.000276, 90 % interval [0.075535, 0.076438], truth 0.1, z = 87.

**Laplace intervals** (`tables.md` section 5, sd about 2.8e-4):

| truth | F2 g | true g | z | error (wavelengths) |
|---|---|---|---|---|
| T2 | 0.02826 | 0.05 | 78 | 0.0217 |
| T2 | 0.07598 | 0.1 | 83 | 0.0240 |
| T3 | 0.04175 | 0.05 | 31 | 0.0083 |
| T3 | 0.09339 | 0.1 | 24 | 0.0066 |
| T5 | 0.06912 | 0.1 | 110 | 0.0309 |

The F2 gaps are wrong by 24 to 110 sd, i.e. 7 to 33 widths of the 90 % interval (the first report said 20 to 300; no case reaches 300).

**T4 (closed gap with a neck):** profile marginal (`T4_F2/profile.json`; HMC is degenerate, section 6): mean 2.35e-6, sd 2.35e-6, 90 % interval [1.20e-7, 7.03e-6], P(g < g_det) = 1. This is correct for the wrong reason: the profile misfit falls with slope 4.3e5 per unit g toward g -> 0 because the bridge is better fit by overlap. The Laplace "covers: yes" of T4 F2 in `tables.md` section 5 is spurious (MAP on the wall, not converged).

## 4. (iii) What flags the misspecification

### 4.1 Fit and predictive check at 30 dB, key 0 (unchanged)

Phi_MAP/N for F2: 1.015 (right family); 8.1 and 8.4 (T2), 15.0 and 16.9 (T3), 4.0 (T4), 5.1 (T5). F2u does not help T2, T4, T5 (8.1, 4.0, 5.1) and fits T3 (1.014); Fe gives 43 to 81 on the two-disk truths and 8.5 on T4 (`tables.md` sections 2 and 5). The refit predictive p-value (100 replicates, resolution 1/101) is 0.000 for every wrong case that has chains (T1 g = 0.05 and 0.1 under Fe, T2 g = 0.1 under F2 and Fe, T4 under Fe; z 355 for T2 F2) and 0.21 for the right family. The pixel posteriors fit every truth (Phi/N 0.999 to 1.002 with 4096 free parameters) and cannot flag anything. Gap-valley probabilities (half strip, Laplace; `tables.md` section 4), Matern / TV: T1 g = 0.05 0.72 / 0.16; T1 g = 0.1 1.000 / 0.87; T2 g = 0.1 0.9996 / 0.69; T3 g = 0.1 0.989 / 0.63; T5 0.80 / 0.21; T4 unrepresentable on this grid.

### 4.2 The scaling law dPhi = c z^2

Let z be the distance of the F2 gap from the true gap in Laplace sd, and dPhi = Phi_MAP minus the misfit of the noise realization itself. A model error fixed in wavelengths moves z as 1/s and dPhi as 1/s^2, so c = dPhi / z^2 does not depend on the SNR. Measured, one realization (noise key 0 scaled to the SNR; `snr/snr_summary.md`; ladder from `ellipticity/ellipticity_g*.md`):

| model error | z at 30 / 20 / 10 dB | c at 30 / 20 / 10 dB | z for a 3 sd flag (147 nats) |
|---|---|---|---|
| T2 15 % ellipticity, g = 0.05 | 78 / 25 / 7.9 | 2.80 / 2.82 / 2.91 | 7.2 |
| T2 15 % ellipticity, g = 0.1 | 83 / 26 / 8.4 | 2.55 / 2.60 / 2.72 | 7.6 |
| ellipticity 0.5 % to 5 % (ladder; c against the level-0 fit of the same noise) | 2 to 29 at 30 dB | 2.0 to 2.3 | 8.0 to 8.6 |
| T5, third disk r = 0.06 | 110 / 35 / 11 | 0.80 / 0.80 / 0.79 | 13.6 |
| T3 unequal contrasts, g = 0.05 | 31 / 9.9 / 3.3 | 35.8 / 34.2 / 30.4 | 2.0 to 2.2 |
| T3 unequal contrasts, g = 0.1 | 24 / 7.5 / 2.4 | 68.1 / 66.4 / 62.1 | 1.5 |
| grid error n = 128 -> 64, absorbed by F2 (`discretization/`) | 1.02 / 0.32 / 0.10 | 0.11 (g = 0.05), 0.09 (g = 0.1) | 37 to 41 (about 61 dB) |

A third small disk can move the gap by 13.6 sd before the misfit rises by 3 sd, ellipticity by 7.5 sd, a grid error by 37 sd. The data are not blind; they see the error quadratically.

### 4.3 The ellipticity ladder at 30 dB, key 0 (`ellipticity/ellipticity_g0.1.md`, `ellipticity_g0.05.md`)

The analytic flag is p = P(Gamma(N - 5/2) >= Phi_MAP) (`shapepost.misfit_p_value`: the exact null of the best-fit misfit of a linear model with complex Gaussian noise, checked by simulation and Kolmogorov-Smirnov in `tests/test_shapepost.py`). The refit p-value of the first version used 30 replicates, so it cannot resolve p below 1/31, and key 0 is +0.78 sd high (noise-only misfit 2438.2 against 2400; `snr/noise_null_check.json`).

| ellipticity | z of truth | Phi_MAP/N | dPhi | p (analytic) | refit p (30 replicates, floor 1/31) |
|---|---|---|---|---|---|
| 0.5 % | 2.0 | 1.023 | 17.8 | 0.12 (not a flag) | 0.033 |
| 1 % | 5.0 | 1.046 | 72.3 | 0.011 | below 0.032 |
| 2 % | 11.1 | 1.136 | 288.9 | 5.5e-11 | below 0.032 |
| 5 % | 28.8 | 1.783 | 1839.8 | 2e-216 | below 0.032 |
| 15 % (T2) | 83.2 | 8.372 | 17655.4 | 0 | below 0.032 |

The g = 0.05 ladder agrees to within a few percent (1 %: z 4.8, p = 0.014). **Typical draws** (40 keys per rung, `snr/snr_summary.md`): 0.5 % is flagged at 5 % in 4 of 40 draws (the null rate is 1 to 2 of 40), 1 % in 17 of 40 (3 sd: 1 of 40, while the interval excludes the truth by more than 3 sd in 39 of 40), 2 % in 40 of 40 (3 sd: 40 of 40). At 2 sigma p is 0.12; at 5 sigma the 5 % flag fires in less than half the draws; the 3 sd flag needs about 8 sigma.

### 4.4 At the book's SNRs: 40 noise keys per cell (`snr/snr_summary.md`; F2 Laplace fit from the 30 dB MAP; p against the analytic null)

| truth | SNR | mean z of truth | interval excludes truth at 3 sd | mean dPhi (sd) | flagged p < 0.05 | flagged at 3 sd (p < 0.00135) |
|---|---|---|---|---|---|---|
| T5 | 10 dB | +10.8 | 40/40 | +97.4 (12.9) = 2.0 sd | **28/40** | **7/40** |
| T5 | 20 dB | +34.6 | 40/40 | +976.9 (40.5) | 40/40 | 40/40 |
| T2 g = 0.1 | 10 dB | +8.2 | 40/40 | +172.9 (18.9) | 40/40 | 30/40 |
| T2 g = 0.1 | 20 dB | +26.2 | 40/40 | +1747.6 (59.2) | 40/40 | 40/40 |
| T2 g = 0.05 | 10 dB | +7.6 | 40/40 | +166.6 (17.8) | 40/40 | 30/40 |
| T3 g = 0.1 | 10 dB | +2.1 | 8/40 | +378.9 (25.1) | 40/40 | 40/40 |
| T3 g = 0.05 | 10 dB | +2.9 | 21/40 | +336.6 (23.8) | 40/40 | 40/40 |
| T4 (12 keys) | 10 dB | gap on the boundary: not meaningful | - | +69.5 (13.0) | 6/12 | 0/12 |
| T4 (12 keys) | 20 dB | - | - | +701.4 (41.5) | 12/12 | 12/12 |
| T1 g = 0.05, right family | 10 / 20 dB | -0.3 / -0.5 | 0/40 | -2.1 (1.6) | 1/40 | 0/40 |
| T3 with F2u, right family | 10 / 20 dB | -0.2 / -0.4 | 0/40 | -3.1 (1.7) | 1/40 | 0/40 |

The right-family rows show that the null behaves: dPhi has the mean -dim/2 (-2.5 for F2, -3.5 for F2u) and the sd 1.6 of a Gamma(dim/2), and the 5 % flag fires in 1 of 40 draws. One realization (key 0): T5 at 10 dB gives g = 0.0690 +- 0.0028 (z 11.0), Phi_MAP/N 1.0555, dPhi +95.0 = 1.9 sd, p = 0.0032, refit p 0.000 (z 3.2, 20 replicates); T2 g = 0.1 at 10 dB: z 8.4, dPhi +189.9, refit z 5.4; at 20 dB the refit z are 23.3 (T5) and 42.5 (T2). **At 10 dB the F2 gap on T5 is wrong by 8.3 to 13.1 sd (mean 10.8) in every draw, and the data raise no 3 sd flag in 33 of 40 draws (no 5 % flag in 12 of 40).** The statement of DESIGN 14.4 (iii) is reached there, and for ellipticity below about 1.5 % at 30 dB; it is not reached at 20 dB, nor at 30 dB for 2 % ellipticity and above or for T5 (z 110 against a threshold of 13.6). Caveat: each fit starts from the 30 dB MAP (one start), so a lower local minimum would only reduce the flagged fractions further; the standard errors of the fractions at 40 draws are up to 8 %, and the same 40 noise keys (0 to 39) are used in every cell, so the cells are paired realizations and not independent replications.

### 4.5 Evidence and caveats

The evidence separates right from wrong only when a better family exists: log BF of F2 against F2u is +9.8 to +10.4 on every truth where F2u does not fit better (the Occam factor; IS and Laplace agree, section 5) and -33672 and -38050 on T3 where it does; between two wrong families the Bayes factor says nothing (F2 against Fe +10971 on T4, 0.86e5 to 1.8e5 elsewhere; each family misfits; `tables.md` section 5). Caveats: the noise level is known exactly and the forward operator is exact up to the grid (the grid error is itself an unflagged model error: about 61 dB would be needed to see it); the deformations tried are ellipticity, unequal contrast, a small third disk and a neck. On the null of the refit check: the 41-key right-family misfit has sd 40.7 against the theoretical 49.0 because the first 40 noise keys have sd 41.3 (sd 47.8 over 1000 keys, KS p = 0.89 against Gamma(2400, 1); `snr/noise_null_check.json`): a property of those keys, not of the generator.

## 5. Evidence: Laplace verified by importance sampling; the TI check was an artifact

**Importance sampling** (Student-t proposal, 5 degrees of freedom, shape 1.2 x the Laplace covariance, 1000 draws; `<case>/is_evidence.json`, `tables.md` section 7). IS against Laplace (Gauss-Newton / finite-difference Hessian), all 15 interior cases:

| case | log Z (IS) +- se | weight ESS | Laplace GN | IS - GN | IS - FD |
|---|---|---|---|---|---|
| T1 g = 0.05 F2 | 35397.847 +- 0.016 | 788 | 35397.841 | +0.006 | +0.007 |
| T1 g = 0.05 F2u | 35387.470 +- 0.019 | 733 | 35387.483 | -0.013 | -0.014 |
| T1 g = 0.1 F2 | 35460.523 +- 0.016 | 788 | 35460.517 | +0.006 | +0.007 |
| T1 g = 0.1 F2u | 35450.110 +- 0.019 | 733 | 35450.123 | -0.013 | -0.013 |
| T2 g = 0.05 F2 | 17927.700 +- 0.017 | 777 | 17927.755 | -0.055 | +0.006 |
| T2 g = 0.05 F2u | 17917.876 +- 0.020 | 721 | 17917.948 | -0.072 | -0.014 |
| T2 g = 0.1 F2 | 17352.289 +- 0.017 | 777 | 17352.339 | -0.050 | +0.007 |
| T2 g = 0.1 F2u | 17342.374 +- 0.020 | 718 | 17342.449 | -0.075 | -0.010 |
| T3 g = 0.05 F2 | 1080.712 +- 0.017 | 785 | 1080.631 | +0.081 | +0.014 |
| T3 g = 0.05 F2u | 34753.010 +- 0.019 | 733 | 34753.023 | -0.013 | -0.014 |
| T3 g = 0.1 F2 | -3259.376 +- 0.022 | 667 | -3259.657 | +0.281 | +0.029 |
| T3 g = 0.1 F2u | 34789.970 +- 0.019 | 733 | 34789.983 | -0.013 | -0.013 |
| T4 Fe | 17089.590 +- 0.018 | 757 | 17089.565 | +0.025 | +0.007 |
| T5 F2 | 25574.227 +- 0.016 | 789 | 25574.198 | +0.029 | +0.006 |
| T5 F2u | 25564.065 +- 0.019 | 735 | 25564.054 | +0.011 | -0.013 |

The Laplace evidence is right: within 0.03 nats with the finite-difference Hessian, within 0.09 with the Gauss-Newton one except T3 g = 0.1 F2 (0.28; Phi/N 16.9, where Gauss-Newton is not the exact Hessian). **Occam factor of F2 against F2u** (log BF): IS +9.82 to +10.41 (T1 g = 0.05 10.38, g = 0.1 10.41; T2 9.82 and 9.92; T5 10.16; se 0.025), Laplace +9.81 to +10.39; on T3 -33672.30 and -38049.35 (IS), -33672.39 and -38049.64 (Laplace). So the Occam factor is verified to 0.025 nats (the largest IS minus Laplace difference, T2 g = 0.1; the IS standard error is 0.025).

**Thermodynamic integration** (`offline_ti.json`, first version `offline_ti_v1.json`). First version: 35424.4 +- 12.8 and 17367.8 +- 7.2, i.e. 26.6 and 15.4 nats above Laplace. That was a quadrature artifact: in the first four intervals of the power schedule (intervals 1 to 3 only, in one of the four runs) Var_beta[Phi] changes by 14 to 230 times between neighbors (1.9e11 to 8.5e8 in one), where a one-term Euler-Maclaurin expansion is invalid, and the correction added 30 to 64 nats there. With the guard (correct only where the variance changes by at most 10-fold; `ti_combine`) TI gives 35375.0 +- 3.1 (between runs 4.5) and 17334.5 +- 2.1 (2.1), 22.8 and 17.8 nats below IS (35397.85 and 17352.29), `schedule_ok` False in all four runs: trapezoid lower bounds (35373.5 and 17332.9). The TI at this schedule and budget does not measure the evidence; it does not contradict the Laplace values either. A TI run on the geometric schedule (`parametric_run.py --stage ti --ti-beta-min 1e-7`) was not made (about 5 single-thread hours).

## 6. Failures and limits

- **T4 x F2 and F2u: HMC degenerate; the truncation box differs from the posterior.** The MAP sits on the |u_g| = 10 wall (g = 1.8e-5, not converged) and the Laplace sd of u_g is 14; two of four chains started outside the box and never moved (`T4_F2/chains_degenerate/`). The gap marginal is the profile marginal (`T4_F2/profile.json`: 145 points in u_g from -24 to -6, all solves converged; the first run's 43 points are in `profile_coarse.json`). New: the posterior mass of u_g inside the box is 4.4e-4 (`truncation`), so the truncated target differs from the posterior by 7.7 nats in log Z; the statement "the box removes less than 9e-5, a change of log Z below 1e-3" holds for the prior and for interior posteriors only. The profile evidence is 28062.8; the Laplace value 28060.7 belongs to the truncated target. The first run interpolated the CDF linearly in g (upper end 7.89e-6, 12 % too high); now 7.03e-6 on both grids.
- **Convergence flags.** With a bulk ESS of Phi of 229 to 414 in 4 x 500 draws the split R-hat to expect from perfect mixing is 1.0077 to 1.0154 (`tables.md` section 1). The fixed limit 1.01 flagged T1 g = 0.1 Fe, T2 g = 0.1 F2 and Fe (classic or rank 1.0107 to 1.0133), within 0.005 of that floor. The criterion is now ESS-aware (excess over the expected value at most 0.01, evaluated at ESS >= 200): all seven chain sets converge; the strict flag still fails those three. T3 g = 0.05 F2u (4 x 400 draws): R-hat of Phi 1.004 / 1.0099 (expected 1.011), worst coordinate rank R-hat 1.016, worst bulk ESS 841, converged (`tables.md` section 10; the first report's "rank R-hat of Phi 1.016, which flags it" was a coordinate).
- **Fe orientation.** psi = pi (sigmoid(u) - 1/2) puts the two ends of [-pi/2, pi/2) at u = -inf and +inf, so a posterior for an ellipse along y would split (documented in `shapemodels.Fe`). It does not matter here: every Fe MAP has psi <= 0.0013 and the chains give 0.0007 to 0.0013 +- 0.0007 (`tables.md` section 9).
- **Operations.** The machine was shared during the follow-up (the load average was not recorded in the job logs). The review batch ran 42 jobs on 7 single-thread slots from 09:31 to 10:06 with no failure (`results/logs/m7r_batch.out`), plus at most 3 more single-thread processes of mine (at most 10 XLA threads at any time). Deferred, unchanged from before: HMC for the dropped cases, the README update.

## 7. Code, tests, files

**Code.** `src/bayes_scatter/shapepost.py`: `misfit_null`, `misfit_p_value`, `importance_log_evidence`, `expected_split_rhat`, ESS-aware `convergence_report` (`rhat_phi_expected`, `rhat_phi_limit`, `converged`, `converged_strict`), `ti_combine` with the validity guard and `end_correction` / `schedule_ok`, `ti_schedule(beta_min)`, `thermodynamic_integration(beta_min)`, corrected truncation docstring; `shapemodels.py`: Fe note. Scripts: new `parametric_snr.py` (also `--noise-check`), `parametric_evidence_is.py`, `parametric_discretization.py`; changed `parametric_profile.py` (spline marginal, resumable, truncation mass), `parametric_sweeps.py` (analytic p, dPhi, c), `parametric_collect.py` (sections 7 to 10, convergence columns), `parametric_ti_offline.py`, `parametric_run.py` (`--ti-beta-min`), `make_m7_jobs.py` (batches `is`, `snr`, `disc`). Job files `scripts/jobs/m7r_all.json` (42 jobs) and `m7r_is.json`, `m7r_snr.json`, `m7r_disc.json`. DESIGN 14.5 to 14.7 and section 9. Note that `parametric_profile.py`, `parametric_sweeps.py`, `parametric_ti_offline.py` and `scripts/jobs/m7_*.json` from the first run were untracked until commit 07857dc added them.

**Tests.** Fast suite, all 35 test files, `-m "not slow"`, four parallel single-thread groups: 119 + 88 + 196 + 295 = 698 passed, 0 failed. Slow tests run: the edited TI-on-scattering test (now on the geometric schedule; 167 s), the HMC-on-scattering test, and the two slow script tests: 4 passed; 5 other slow tests (unchanged areas) not run. An intermediate run had two failures that the new tests exposed in my own code and I fixed: a fake Fe fixture without parameter names, and an ESS-aware allowance that grew without bound when unmixed chains collapse the ESS (now capped at ESS = 200). New tests catch each code defect: the TI end correction on a sharp posterior (+10.8 nats without the guard, -4.2 and flagged with it), the geometric schedule, importance sampling on the Gaussian toy and a heavy-tailed target, the analytic null by simulation, the noise generator, expected R-hat against simulated AR(1) chains and the ESS-aware criterion (including the unmixed case), profile quantiles against an exponential law (old rule more than 8 % off) and the truncation mass, the profile checkpoint, SNR units (cached noise reproduced, scaled copies, resumable), the IS script, the discretization fit and analysis, ladder flags, the offline TI guard, the new `tables.md` sections and the job files.

**Commands as run (single XLA thread, env as in `batch_run.py --threads 1`).** `python scripts/parametric_profile.py --truth T4 --gap 0 --model F2 --grid-lo -24 --grid-hi -6 --grid-n 145 --force`; `python scripts/make_m7_jobs.py --batch is snr disc`; `nice -n 5 python scripts/batch_run.py scripts/jobs/m7r_all.json --cpu-slots 7 --threads 1`; `python scripts/parametric_run.py --truth ... --model ... --stage finalize --chains 4 --segments 5` for the 7 cases with chains (summaries backed up and regenerated; only the convergence block changed); `python scripts/parametric_ti_offline.py --truth T1 --gap 0.05 --model F2` and `--truth T2 --gap 0.1`; `python scripts/parametric_snr.py --noise-check` and `--collect`; `python scripts/parametric_discretization.py analyze`; `python scripts/parametric_sweeps.py --gap 0.05 --collect` and `--gap 0.1 --collect`; `python scripts/parametric_collect.py`.

## 8. Review log

| finding | resolution |
|---|---|
| **Critical.** (iii): the 2-sigma sentence is wrong, "not reached" is wrong, the 10-20 dB question is unanswered, p = 0.033 is 1 of 30 replicates | **Fixed.** Reproduced p = 0.116 (0.123 at g = 0.05) and 0.011. Replaced by dPhi = c z^2 with c per case (2.6-2.9 / 2.0-2.3 / 0.80 / 30-68 / 0.09-0.11) and thresholds z = 7.5 / 13.6 / 1.5-2.2 / 37-41. 10/20/30 dB fits for all 8 truths over 40 keys (`snr/`). At 10 dB T5: z 10.8 in 40/40, dPhi 2.0 sd, flags 28/40 (5 %) and 7/40 (3 sd): strongest statement reached; not reached at 20 dB. Analytic p-values; refit resolution stated. The low replicate sd is a property of keys 0-39 (sd 41.3), the generator is Gamma(2400) (KS p 0.89). |
| **Major.** (i) attribution is established and deterministic; "exact forward models" is wrong | **Fixed.** Noise-free fits +0.005 / +1.024 / +1.237 sd (n = 64 / 128 / 256), paired shift -1.019 +- 0.002, corrected z mean -0.13 +- 0.16, coverage 0.88, Phi_MAP 0.112 vs 8.85; offset 0.10 / 0.32 / 1.02 / 3.2 sd at 10 / 20 / 30 / 40 dB. "Weak" and "exact" removed; grid error documented as an unflagged model error. |
| **Major.** Laplace is right, TI is biased high by the end correction | **Fixed.** IS on 15 cases: within 0.03 nats (FD) / 0.09 nats (GN; one 0.28). `ti_combine` guard + geometric schedule option; TI recomputed: 22.8 and 17.8 nats below IS, `schedule_ok` False; Occam factor verified (+9.82 to +10.41). |
| **Minor.** "20-300 sd" | **Fixed.** 24 to 110 sd (7 to 33 widths). |
| **Minor.** DESIGN "+10.0 to +10.4" | **Fixed.** +9.8 to +10.4. |
| **Minor.** truncation bound false for g -> 0 | **Fixed.** Docstring and DESIGN restricted to interior posteriors; T4 mass inside the box 4.4e-4, truncated target 7.7 nats off. |
| **Minor.** two HMC cases are one realization | **Fixed.** Stated; correlation 0.985; the 41-key calibration carries (i). |
| **Minor.** R-hat flags at the noise floor; rank maximum not quoted | **Fixed.** Expected R-hat and ESS-aware criterion (capped at ESS 200), strict flag kept, rank maxima in `tables.md`, summaries regenerated. |
| **Minor.** Fe orientation redundancy | **Fixed** by documentation and evidence (psi <= 0.0013); periodic coordinate deferred. |
| **Minor.** profile quantile interpolation | **Fixed.** Spline marginal, 145-point grid, resumable; 7.03e-6 upper end. The reviewer's exponential-law numbers (slope 4.36e5, 6.88e-6) are **disputed in detail**: the measured slope is 4.26e5 and gives 7.03e-6; the qualitative finding stands. |
| Deferred | Geometric-schedule TI (about 5 h; IS covers it), HMC for the dropped cases, periodic Fe coordinate, README. |

## Numbers audit log

An independent numbers-vs-outputs audit traced 49 quantitative claims to the raw outputs and recomputed headline quantities from the stored chains, MAPs and Hessians: 33 matched, 6 differed only in rounding, 8 were wrong or overstated and 2 had no stored source. 18 of its 18 text corrections are applied above.
