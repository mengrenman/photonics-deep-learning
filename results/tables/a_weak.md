**Efficiency and convergence (a_weak).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_pcn_map | 2 x (600+2400), 4 | 0.196 | 5.45e-04 | 3001 | 2454.8 (2402.5, 2400) | 2471.8 / 2447.5 | (15.6) | (2.592) | (14.5) | (2.408) | (2.324) | (6.0) | 1.51 | 7.67 | 737 (n/a) |
| a_pcn_langevin_map | 2 x (600+2400), 4 | 0.481 | 1.25e-07 | 3001 | 2440.3 (2402.5, 2400) | 2436.5 / 2441.7 | (37.6) | (6.269) | (14.5) | (2.414) | (2.331) | (6.4) | 1.10 | 5.27 | 1739 (n/a) |
| a_mala_map | 2 x (600+2400), 4 | 0.557 | 2.14e-07 | 3001 | 2440.9 (2402.5, 2400) | 2436.0 / 2441.8 | (23.7) | (3.942) | (14.3) | (2.384) | (2.301) | (6.7) | 1.17 | 4.54 | 1716 (n/a) |
| a_pcn_prior | 2 x (1000+1000), 2 | 0.104 | 0.017 | 2001 | 1.77e+05 (2402.5, 2400) | 2.27e+05 / 1.38e+05 | (13.9) | (3.478) | (14.2) | (3.544) | (3.363) | (5.9) | 2.14 | 22.05 | 662 (n/a) |
| a_pcn_langevin_prior | 2 x (1000+1000), 2 | 0.659 | 6.41e-08 | 2001 | 1.62e+06 (2402.5, 2400) | 2.09e+06 / 1.13e+06 | (13.9) | (3.478) | (13.8) | (3.441) | (3.265) | (5.5) | 8.07 | 114.69 | 1569 (n/a) |
| a_mala_prior | 2 x (1000+1000), 2 | 0.718 | 1.09e-07 | 2001 | 1.57e+06 (2402.5, 2400) | 2.03e+06 / 1.11e+06 | (13.9) | (3.473) | (13.7) | (3.427) | (3.252) | (5.5) | 8.09 | 114.61 | 1552 (n/a) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| a_pcn_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| a_pcn_langevin_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| a_mala_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| a_pcn_prior | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| a_pcn_langevin_prior | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| a_mala_prior | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |

**Reconstruction and uncertainty (a_weak).** err_model is eq. 15.21 on eps_r (vacuum = 0.1954); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5).

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_pcn_map | 0.0432 | 0.0433 | 0.221 | 0.222 | 0.0049 | 0.0054 | 0.0013 / 0.0016 | 0.0019 / 0.0016 | 0.003 (0.003 / 0.003) | 1.00 | 0.86 / 0.96 / 0.83 | 0.15, 194.5 | 0.497 |
| a_pcn_langevin_map | 0.0432 | 0.0432 | 0.221 | 0.221 | 0.0049 | 0.0049 | 0.0012 / 0.0015 | 0.0017 / 0.0015 | 0.003 (0.002 / 0.003) | 1.00 | 0.86 / 0.96 / 0.83 | 0.15, 179.4 | 0.495 |
| a_mala_map | 0.0432 | 0.0432 | 0.221 | 0.221 | 0.0049 | 0.0049 | 0.0013 / 0.0015 | 0.0018 / 0.0015 | 0.003 (0.003 / 0.003) | 1.00 | 0.86 / 0.98 / 0.83 | 0.15, 183.5 | 0.496 |
| a_pcn_prior | 0.0432 | 0.1029 | 0.221 | 0.527 | 0.0049 | 0.2041 | 0.0563 / 0.0778 | 0.0793 / 0.0778 | 0.146 (0.113 / 0.156) | 1.00 | 0.17 / 0.25 / 0.15 | 0.23, 19.1 | 0.508 |
| a_pcn_langevin_prior | 0.0432 | 0.1894 | 0.221 | 0.969 | 0.0049 | 0.6632 | 0.1613 / 0.1419 | 0.1952 / 0.1601 | 0.292 (0.323 / 0.284) | 0.85 | 0.08 / 0.08 / 0.08 | 0.30, 14.9 | 0.428 |
| a_mala_prior | 0.0432 | 0.1863 | 0.221 | 0.953 | 0.0049 | 0.6466 | 0.1595 / 0.1396 | 0.1933 / 0.1567 | 0.288 (0.319 / 0.279) | 0.85 | 0.09 / 0.09 / 0.09 | 0.31, 15.4 | 0.429 |

Pooled across initializations `a_pcn_pool` (4 chains, runs a_pcn_map, a_pcn_prior): split R-hat of Phi 7.45; of log eps_r over pixels: median 6.51, max 26.85, fraction of pixels with R-hat < 1.1: 0.00.

Pooled across initializations `a_pcn_langevin_pool` (4 chains, runs a_pcn_langevin_map, a_pcn_langevin_prior): split R-hat of Phi 9.07; of log eps_r over pixels: median 12.89, max 155.10, fraction of pixels with R-hat < 1.1: 0.00.

Pooled across initializations `a_mala_pool` (4 chains, runs a_mala_map, a_mala_prior): split R-hat of Phi 9.09; of log eps_r over pixels: median 13.20, max 156.30, fraction of pixels with R-hat < 1.1: 0.00.
