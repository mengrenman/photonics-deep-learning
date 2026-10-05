**Efficiency and convergence (L_mod).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 2 x (300+2700), 3 | 0.386 | 0.093 | 3001 | 2435.4 (2432.9, 2400) | 2433.3 / 2437.9 | 2321.6 | 99.3 | 16.550 | 22.1 | 3.674 | 3.276 | 17.1 | 1.02 | 1.47 | 936 (8) |
| L_mod_mala_over | 2 x (300+2700), 3 | 0.506 | 0.077 | 3001 | 2433.2 (2432.9, 2400) | 2433.7 / 2432.0 | 2334.3 | 107.2 | 17.868 | 23.2 | 3.865 | 3.447 | 22.9 | 1.03 | 1.31 | 1396 (18) |
| L_mod_pcnl_map | 2 x (300+2700), 3 | 0.373 | 0.059 | 3001 | 2435.4 (2432.9, 2400) | 2435.7 / 2435.7 | 2354.1 | 177.6 | 29.592 | 25.2 | 4.205 | 3.749 | 22.3 | 1.00 | 1.30 | 1393 (18) |
| L_mod_hmc_map | 2 x (60+240), 1 | 0.467 | 0.220 | 3001 | 2434.9 (2432.9, 2400) | 2435.7 / 2438.2 | 2247.8 | 117.2 | 19.534 | 32.5 | 5.422 | 4.835 | 29.5 | 1.01 | 1.16 | 2367 (18) |
| L_mod_hmc_over | 2 x (60+240), 1 | 0.596 | 0.209 | 3001 | 2434.9 (2432.9, 2400) | 2435.7 / 2435.8 | 2296.5 | 173.9 | 28.979 | 54.7 | 9.117 | 8.130 | 43.7 | 1.01 | 1.06 | 1748 (18) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| L_mod_mala_over | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| L_mod_pcnl_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| L_mod_hmc_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| L_mod_hmc_over | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |

**Reconstruction and uncertainty (L_mod).** err_model is eq. 15.21 on eps_r (vacuum = 0.3182); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 0.1170 | 0.1112 | 0.368 | 0.349 | 0.0068 | 0.0061 | 0.0519 / 0.0572 | 0.0870 / 0.0579 | 0.112 (0.104 / 0.114) | 1.00 | 0.12 / 0.26 / 0.07 | 0.94, 9.0 | 0.495 | 0.0540 / 0.0600 | 0.955 (0.729) |
| L_mod_mala_over | 0.1170 | 0.1111 | 0.368 | 0.349 | 0.0068 | 0.0062 | 0.0527 / 0.0577 | 0.0884 / 0.0586 | 0.113 (0.105 / 0.115) | 1.00 | 0.11 / 0.23 / 0.07 | 0.93, 8.7 | 0.493 | 0.0540 / 0.0600 | 0.966 (0.744) |
| L_mod_pcnl_map | 0.1170 | 0.1121 | 0.368 | 0.352 | 0.0068 | 0.0061 | 0.0517 / 0.0577 | 0.0867 / 0.0585 | 0.113 (0.103 / 0.115) | 1.00 | 0.12 / 0.27 / 0.07 | 0.93, 10.5 | 0.493 | 0.0540 / 0.0600 | 0.962 (0.755) |
| L_mod_hmc_map | 0.1170 | 0.1103 | 0.368 | 0.347 | 0.0068 | 0.0063 | 0.0532 / 0.0600 | 0.0893 / 0.0608 | 0.117 (0.106 / 0.120) | 1.00 | 0.11 / 0.25 / 0.07 | 0.91, 9.4 | 0.492 | 0.0540 / 0.0600 | 0.996 (0.763) |
| L_mod_hmc_over | 0.1170 | 0.1110 | 0.368 | 0.349 | 0.0068 | 0.0061 | 0.0529 / 0.0591 | 0.0887 / 0.0599 | 0.115 (0.106 / 0.118) | 1.00 | 0.11 / 0.25 / 0.07 | 0.94, 8.6 | 0.493 | 0.0540 / 0.0600 | 0.984 (0.814) |

Pooled across initializations `L_mod_mala_pool` (4 chains, runs L_mod_mala_map, L_mod_mala_over): split R-hat of Phi 1.03; of log eps_r over pixels: median 1.04, max 1.28, fraction of pixels with R-hat < 1.1: 0.90.

Pooled across initializations `L_mod_hmc_pool` (4 chains, runs L_mod_hmc_map, L_mod_hmc_over): split R-hat of Phi 1.01; of log eps_r over pixels: median 1.01, max 1.09, fraction of pixels with R-hat < 1.1: 1.00.
