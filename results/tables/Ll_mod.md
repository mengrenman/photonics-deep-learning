**Efficiency and convergence (Ll_mod).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ll_mod_mala_map | 2 x (300+11700), 12 | 0.391 | 0.093 | 12001 | 2435.2 (2432.9, 2400) | 2435.5 / 2434.6 | 2316.8 | 348.6 | 14.522 | 43.2 | 1.801 | 1.748 | 52.7 | 1.00 | 1.07 | 4124 (19) |
| Ll_mod_hmc_map | 2 x (150+1050), 1 | 0.708 | 0.183 | 12001 | 2434.1 (2432.9, 2400) | 2434.2 / 2435.8 | 2277.2 | 613.0 | 25.539 | 138.5 | 5.770 | 5.599 | 137.6 | 1.00 | 1.02 | 4257 (19) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| Ll_mod_mala_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |
| Ll_mod_hmc_map | 32, 2, 64 | 2.09 | 3.16 | 0.66 | 3401 | 2438 | 19.7 |

**Reconstruction and uncertainty (Ll_mod).** err_model is eq. 15.21 on eps_r (vacuum = 0.3182); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ll_mod_mala_map | 0.1170 | 0.1111 | 0.368 | 0.349 | 0.0068 | 0.0061 | 0.0528 / 0.0593 | 0.0884 / 0.0601 | 0.115 (0.106 / 0.119) | 1.00 | 0.11 / 0.24 / 0.07 | 0.95, 9.0 | 0.494 | 0.0540 / 0.0600 | 0.986 (0.838) |
| Ll_mod_hmc_map | 0.1170 | 0.1107 | 0.368 | 0.348 | 0.0068 | 0.0062 | 0.0543 / 0.0602 | 0.0911 / 0.0610 | 0.118 (0.109 / 0.120) | 1.00 | 0.10 / 0.21 / 0.07 | 0.94, 8.5 | 0.493 | 0.0540 / 0.0600 | 1.003 (0.916) |
