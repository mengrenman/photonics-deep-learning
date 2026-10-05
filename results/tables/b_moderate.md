**Efficiency and convergence (b_moderate).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b_pcn_map | 2 x (600+2400), 4 | 0.240 | 6.26e-04 | 3001 | 2645.2 (2610.2, 2400) | 2660.3 / 2636.9 | (15.2) | (2.531) | (14.0) | (2.330) | (2.250) | (6.7) | 1.90 | 6.54 | 1043 (n/a) |
| b_pcn_langevin_map | 2 x (600+2400), 4 | 0.491 | 2.03e-07 | 3001 | 2631.3 (2610.2, 2400) | 2625.6 / 2631.2 | (23.0) | (3.832) | (14.1) | (2.351) | (2.271) | (6.5) | 1.27 | 4.03 | 2033 (n/a) |
| b_mala_map | 2 x (600+2400), 4 | 0.558 | 3.50e-07 | 3001 | 2631.2 (2610.2, 2400) | 2628.6 / 2630.7 | (47.6) | (7.923) | (14.4) | (2.406) | (2.325) | (6.2) | 1.15 | 3.64 | 2025 (n/a) |
| b_pcn_prior | 2 x (1000+1000), 2 | 0.124 | 0.019 | 2001 | 6.20e+05 (2610.2, 2400) | 6.75e+05 / 5.73e+05 | (13.8) | (3.459) | (13.8) | (3.447) | (3.275) | (5.7) | 23.37 | 75.08 | 983 (n/a) |
| b_pcn_langevin_prior | 2 x (1000+1000), 2 | 0.663 | 1.70e-07 | 2001 | 5.04e+05 (2610.2, 2400) | 6.71e+05 / 3.92e+05 | (14.1) | (3.511) | (13.7) | (3.434) | (3.262) | (5.5) | 6.03 | 59.36 | 1719 (n/a) |
| b_mala_prior | 2 x (1000+1000), 2 | 0.709 | 2.98e-07 | 2001 | 4.73e+05 (2610.2, 2400) | 6.35e+05 / 3.65e+05 | (14.1) | (3.535) | (13.6) | (3.409) | (3.238) | (5.5) | 5.97 | 53.78 | 1711 (n/a) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| b_pcn_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| b_pcn_langevin_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| b_mala_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| b_pcn_prior | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| b_pcn_langevin_prior | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| b_mala_prior | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |

**Reconstruction and uncertainty (b_moderate).** err_model is eq. 15.21 on eps_r (vacuum = 0.3182); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5).

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b_pcn_map | 0.1229 | 0.1228 | 0.386 | 0.386 | 0.0111 | 0.0111 | 0.0016 / 0.0020 | 0.0027 / 0.0020 | 0.004 (0.003 / 0.004) | 1.00 | 0.95 / 0.99 / 0.94 | 0.35, 419.9 | 0.496 |
| b_pcn_langevin_map | 0.1229 | 0.1229 | 0.386 | 0.386 | 0.0111 | 0.0109 | 0.0017 / 0.0021 | 0.0029 / 0.0021 | 0.004 (0.003 / 0.004) | 1.00 | 0.94 / 0.99 / 0.93 | 0.35, 406.2 | 0.494 |
| b_mala_map | 0.1229 | 0.1228 | 0.386 | 0.386 | 0.0111 | 0.0109 | 0.0017 / 0.0020 | 0.0028 / 0.0020 | 0.004 (0.003 / 0.004) | 1.00 | 0.95 / 0.99 / 0.93 | 0.35, 422.8 | 0.494 |
| b_pcn_prior | 0.1229 | 0.3996 | 0.386 | 1.256 | 0.0111 | 0.7849 | 0.1881 / 0.1899 | 0.2461 / 0.2662 | 0.379 (0.376 / 0.380) | 0.75 | 0.14 / 0.20 / 0.13 | 0.45, 19.1 | 0.412 |
| b_pcn_langevin_prior | 0.1229 | 0.2127 | 0.386 | 0.668 | 0.0111 | 0.3577 | 0.1358 / 0.1118 | 0.1900 / 0.1223 | 0.235 (0.272 / 0.224) | 0.91 | 0.18 / 0.25 / 0.16 | 0.59, 29.0 | 0.388 |
| b_mala_prior | 0.1229 | 0.2106 | 0.386 | 0.662 | 0.0111 | 0.3471 | 0.1309 / 0.1098 | 0.1847 / 0.1192 | 0.230 (0.262 / 0.220) | 0.92 | 0.18 / 0.25 / 0.15 | 0.60, 34.9 | 0.392 |

Pooled across initializations `b_pcn_pool` (4 chains, runs b_pcn_map, b_pcn_prior): split R-hat of Phi 31.19; of log eps_r over pixels: median 10.71, max 81.08, fraction of pixels with R-hat < 1.1: 0.00.

Pooled across initializations `b_pcn_langevin_pool` (4 chains, runs b_pcn_langevin_map, b_pcn_langevin_prior): split R-hat of Phi 7.18; of log eps_r over pixels: median 8.97, max 89.14, fraction of pixels with R-hat < 1.1: 0.00.

Pooled across initializations `b_mala_pool` (4 chains, runs b_mala_map, b_mala_prior): split R-hat of Phi 7.20; of log eps_r over pixels: median 8.38, max 83.57, fraction of pixels with R-hat < 1.1: 0.00.
