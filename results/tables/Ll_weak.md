**Efficiency and convergence (Ll_weak).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ll_weak_mala_map | 2 x (300+11700), 12 | 0.438 | 0.071 | 12001 | 2432.8 (2389.0, 2400) | 2433.1 / 2432.4 | 2293.8 | 310.0 | 12.915 | 51.9 | 2.161 | 2.096 | 56.9 | 1.00 | 1.08 | 4553 (14) |
| Ll_weak_hmc_map | 2 x (150+1050), 1 | 0.688 | 0.181 | 12001 | 2432.3 (2389.0, 2400) | 2432.5 / 2433.1 | 2265.4 | 768.0 | 31.997 | 281.8 | 11.740 | 11.390 | 290.6 | 1.00 | 1.02 | 4817 (14) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| Ll_weak_mala_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| Ll_weak_hmc_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |

**Reconstruction and uncertainty (Ll_weak).** err_model is eq. 15.21 on eps_r (vacuum = 0.1954); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ll_weak_mala_map | 0.0426 | 0.0429 | 0.218 | 0.219 | 0.0047 | 0.0055 | 0.0549 / 0.0574 | 0.0788 / 0.0576 | 0.114 (0.110 / 0.115) | 1.00 | 0.00 / 0.00 / 0.00 | n/a, 2.8 | 0.501 | 0.0581 / 0.0585 | 0.974 (0.845) |
| Ll_weak_hmc_map | 0.0426 | 0.0422 | 0.218 | 0.216 | 0.0047 | 0.0056 | 0.0581 / 0.0587 | 0.0833 / 0.0589 | 0.117 (0.116 / 0.117) | 1.00 | 0.00 / 0.00 / 0.00 | n/a, 2.4 | 0.501 | 0.0581 / 0.0585 | 1.003 (0.918) |
