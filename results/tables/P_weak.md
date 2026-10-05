**Efficiency and convergence (P_weak).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_weak_hmc_map | 2 x (60+240), 1 | 0.715 | 0.176 | 3001 | 2433.2 (2389.0, 2400) | 2433.6 / 2432.5 | 2257.3 | 187.8 | 31.293 | 53.2 | 8.859 | 7.890 | 62.3 | 1.00 | 1.08 | 988 (5) |
| L_weak_hmc_prior | 2 x (400+240), 1 | 0.688 | 0.011 | 6401 | 2.63e+05 (2389.0, 2400) | 2.92e+05 / 2.39e+05 | -1.21e+07 | (14.0) | (1.093) | (13.7) | (1.071) | (1.013) | (5.6) | 20.12 | 152.40 | 3635 (14) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| L_weak_hmc_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| L_weak_hmc_prior | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |

**Reconstruction and uncertainty (P_weak).** err_model is eq. 15.21 on eps_r (vacuum = 0.1954); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_weak_hmc_map | 0.0426 | 0.0426 | 0.218 | 0.218 | 0.0047 | 0.0057 | 0.0586 / 0.0584 | 0.0842 / 0.0585 | 0.117 (0.117 / 0.117) | 1.00 | 0.00 / 0.00 / 0.00 | n/a, 2.6 | 0.502 | 0.0581 / 0.0585 | 1.000 (0.839) |
| L_weak_hmc_prior | 0.0426 | 0.8901 | 0.218 | 4.554 | 0.0047 | 1.2808 | 0.3975 / 0.3501 | 0.4174 / 0.4105 | 0.721 (0.795 / 0.700) | 0.54 | 0.09 / 0.14 / 0.08 | 0.20, 43.8 | 0.447 | 0.0581 / 0.0585 | 6.204 (0.207) |
