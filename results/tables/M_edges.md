**Efficiency and convergence (M_edges).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 2 x (300+2700), 3 | 0.386 | 0.093 | 3001 | 2435.4 (2432.9, 2400) | 2433.3 / 2437.9 | 2321.6 | 99.3 | 16.550 | 22.1 | 3.674 | 3.276 | 17.1 | 1.02 | 1.47 | 936 (8) |
| L_mod_hmc_map | 2 x (60+240), 1 | 0.467 | 0.220 | 3001 | 2434.9 (2432.9, 2400) | 2435.7 / 2438.2 | 2247.8 | 117.2 | 19.534 | 32.5 | 5.422 | 4.835 | 29.5 | 1.01 | 1.16 | 2367 (18) |
| Mdf1_mod_mala_map | 2 x (300+2700), 3 | 0.411 | 0.089 | 3001 | 2434.5 (2444.2, 2400) | 2432.7 / 2434.7 | 2322.1 | 122.7 | 20.436 | 24.6 | 4.104 | 3.657 | 20.2 | 1.04 | 1.38 | 934 (8) |
| Mdf1_mod_hmc_map | 2 x (60+240), 1 | 0.523 | 0.208 | 3001 | 2432.3 (2444.2, 2400) | 2434.7 / 2432.1 | 2225.0 | 136.1 | 22.673 | 29.6 | 4.928 | 4.391 | 23.3 | 1.06 | 1.22 | 2691 (12) |
| Mn48_mod_mala_map | 2 x (300+2700), 3 | 0.496 | 0.078 | 3001 | 2435.8 (2446.8, 2400) | 2438.1 / 2434.0 | 2321.8 | 133.1 | 22.173 | 23.5 | 3.913 | 3.482 | 19.9 | 1.03 | 1.32 | 3574 (18) |
| Mn48_mod_hmc_map | 2 x (60+240), 1 | 0.679 | 0.171 | 3001 | 2434.3 (2446.8, 2400) | 2435.8 / 2435.4 | 2226.7 | 109.2 | 18.195 | 48.2 | 8.032 | 7.146 | 35.5 | 1.01 | 1.21 | 2189 (24) |
| Mgmrf_mod_mala_map | 2 x (300+2700), 3 | 0.008 | 1.47e-05 | 3001 | 3.08e+05 (2371.3, 2400) | 3.09e+05 / 3.07e+05 | 3.08e+05 | (14.5) | (2.411) | (13.9) | (2.312) | (2.059) | (6.7) | 1.46 | 32.82 | 879 (8) |
| Mgmrf_mod_mala_none | 2 x (300+2700), 3 | 0.679 | 0.006 | 3001 | 2467.3 (2371.3, 2400) | 2509.5 / 2442.9 | 1619.7 | (14.8) | (2.467) | (18.1) | (3.023) | (2.692) | (16.6) | 1.57 | 1.42 | 1346 (138) |
| Mgmrf_mod_hmc_none | 2 x (60+240), 1 | 0.604 | 0.066 | 3001 | 2428.0 (2371.3, 2400) | 2428.5 / 2425.8 | -5882.1 | 178.2 | 29.697 | 33.7 | 5.607 | 4.993 | 31.4 | 1.04 | 1.16 | 1814 (115) |
| Mgmrf_mod_hmc_long | 2 x (150+1050), 1 | 0.405 | 0.060 | 12001 | 2427.2 (2371.3, 2400) | 2427.6 / 2429.5 | -18831.3 | 185.8 | 7.741 | 35.6 | 1.482 | 1.438 | 53.6 | 1.01 | 1.14 | 8385 (12) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| L_mod_hmc_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| Mdf1_mod_mala_map | 32, 2, 32 | 0.00 | 3.16 | 0.00 | 2438 | 2438 | 0.0 |
| Mdf1_mod_hmc_map | 32, 2, 32 | 0.00 | 3.16 | 0.00 | 2438 | 2438 | 0.0 |
| Mn48_mod_mala_map | 48, 2, 96 | 0.85 | 3.16 | 0.27 | 2578 | 2438 | 2.9 |
| Mn48_mod_hmc_map | 48, 2, 96 | 0.85 | 3.16 | 0.27 | 2578 | 2438 | 2.9 |
| Mgmrf_mod_mala_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| Mgmrf_mod_mala_none | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| Mgmrf_mod_hmc_none | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| Mgmrf_mod_hmc_long | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |

**Reconstruction and uncertainty (M_edges).** err_model is eq. 15.21 on eps_r (vacuum = 0.3182); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_mod_mala_map | 0.1170 | 0.1112 | 0.368 | 0.349 | 0.0068 | 0.0061 | 0.0519 / 0.0572 | 0.0870 / 0.0579 | 0.112 (0.104 / 0.114) | 1.00 | 0.12 / 0.26 / 0.07 | 0.94, 9.0 | 0.495 | 0.0540 / 0.0600 | 0.955 (0.729) |
| L_mod_hmc_map | 0.1170 | 0.1103 | 0.368 | 0.347 | 0.0068 | 0.0063 | 0.0532 / 0.0600 | 0.0893 / 0.0608 | 0.117 (0.106 / 0.120) | 1.00 | 0.11 / 0.25 / 0.07 | 0.91, 9.4 | 0.492 | 0.0540 / 0.0600 | 0.996 (0.763) |
| Mdf1_mod_mala_map | 0.1184 | 0.1113 | 0.372 | 0.350 | 0.0073 | 0.0061 | 0.0515 / 0.0569 | 0.0857 / 0.0577 | 0.111 (0.103 / 0.114) | 1.00 | 0.12 / 0.27 / 0.07 | 0.94, 8.5 | 0.488 | 0.0541 / 0.0600 | 0.950 (0.720) |
| Mdf1_mod_hmc_map | 0.1184 | 0.1110 | 0.372 | 0.349 | 0.0073 | 0.0062 | 0.0535 / 0.0595 | 0.0891 / 0.0604 | 0.116 (0.107 / 0.119) | 1.00 | 0.11 / 0.24 / 0.07 | 0.94, 9.1 | 0.490 | 0.0541 / 0.0600 | 0.992 (0.773) |
| Mn48_mod_mala_map | 0.1325 | 0.1276 | 0.407 | 0.392 | 0.0074 | 0.0062 | 0.0515 / 0.0591 | 0.0879 / 0.0605 | 0.115 (0.103 / 0.118) | 1.00 | 0.13 / 0.28 / 0.09 | 0.87, 9.8 | 0.483 | 0.0544 / 0.0608 | 0.968 (0.690) |
| Mn48_mod_hmc_map | 0.1325 | 0.1261 | 0.407 | 0.387 | 0.0074 | 0.0064 | 0.0561 / 0.0618 | 0.0958 / 0.0632 | 0.121 (0.112 / 0.124) | 1.00 | 0.11 / 0.24 / 0.08 | 0.90, 8.3 | 0.482 | 0.0544 / 0.0608 | 1.020 (0.844) |
| Mgmrf_mod_mala_map | 0.1087 | 0.1206 | 0.341 | 0.379 | 0.0054 | 0.3571 | 0.0198 / 0.0121 | 0.0314 / 0.0118 | 0.028 (0.040 / 0.024) | 1.00 | 0.71 / 0.79 / 0.68 | 0.37, 139.2 | 0.721 | 0.3203 / 0.3250 | 0.043 (0.003) |
| Mgmrf_mod_mala_none | 0.1087 | 0.1093 | 0.341 | 0.343 | 0.0054 | 0.0063 | 0.0641 / 0.0864 | 0.1068 / 0.0873 | 0.162 (0.128 / 0.173) | 1.00 | 0.05 / 0.12 / 0.03 | 0.88, 9.6 | 0.488 | 0.3203 / 0.3250 | 0.251 (0.154) |
| Mgmrf_mod_hmc_none | 0.1087 | 0.1093 | 0.341 | 0.343 | 0.0054 | 0.0065 | 0.1094 / 0.1415 | 0.1813 / 0.1422 | 0.268 (0.219 / 0.283) | 1.00 | 5.86e-03 / 0.01 / 3.85e-03 | 0.50, 4.2 | 0.496 | 0.3203 / 0.3250 | 0.414 (0.251) |
| Mgmrf_mod_hmc_long | 0.1087 | 0.1094 | 0.341 | 0.344 | 0.0054 | 0.0088 | 0.1395 / 0.1774 | 0.2313 / 0.1780 | 0.337 (0.279 / 0.355) | 1.00 | 1.95e-03 / 0.00 / 2.56e-03 | 1.00, 3.7 | 0.503 | 0.3203 / 0.3250 | 0.521 (0.336) |
