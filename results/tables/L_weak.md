**Efficiency and convergence (L_weak).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | Phi' mean (sampler potential) | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_weak_mala_map | 2 x (300+2700), 3 | 0.446 | 0.071 | 3001 | 2432.6 (2389.0, 2400) | 2432.0 / 2434.1 | 2303.7 | 54.6 | 9.099 | 19.4 | 3.237 | 2.883 | 14.3 | 1.04 | 1.45 | 655 (2) |
| L_weak_mala_over | 2 x (300+2700), 3 | 0.530 | 0.058 | 3001 | 2433.3 (2389.0, 2400) | 2432.2 / 2432.1 | 2295.9 | 93.2 | 15.520 | 20.0 | 3.324 | 2.961 | 16.5 | 1.03 | 1.36 | 881 (5) |
| L_weak_pcnl_map | 2 x (300+2700), 3 | 0.284 | 0.053 | 3001 | 2432.5 (2389.0, 2400) | 2432.6 / 2433.0 | 2309.8 | 168.6 | 28.095 | 19.2 | 3.195 | 2.845 | 14.7 | 1.01 | 1.50 | 801 (5) |
| L_weak_hmc_map | 2 x (60+240), 1 | 0.715 | 0.176 | 3001 | 2433.2 (2389.0, 2400) | 2433.6 / 2432.5 | 2257.3 | 187.8 | 31.293 | 53.2 | 8.859 | 7.890 | 62.3 | 1.00 | 1.08 | 988 (5) |
| L_weak_hmc_over | 2 x (60+240), 1 | 0.635 | 0.192 | 3001 | 2433.4 (2389.0, 2400) | 2433.0 / 2434.4 | 2286.0 | 171.3 | 28.537 | 55.6 | 9.269 | 8.256 | 60.2 | 1.01 | 1.06 | 1322 (5) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| L_weak_mala_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| L_weak_mala_over | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| L_weak_pcnl_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| L_weak_hmc_map | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |
| L_weak_hmc_over | 32, 2, 64 | 0.74 | 3.16 | 0.23 | 2596 | 2438 | 3.2 |

**Reconstruction and uncertainty (L_weak).** err_model is eq. 15.21 on eps_r (vacuum = 0.1954); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5). 'Laplace' = Gauss-Newton (Laplace) std of log eps_r at the MAP from the cached eigenpairs; 'MCMC / Laplace' = mean (min) over pixels of the ratio of the sampled std to it; a ratio well below 1 means the sampler has not finished spreading out.

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg | Laplace std log eps_r in / out | MCMC / Laplace std mean (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L_weak_mala_map | 0.0426 | 0.0433 | 0.218 | 0.222 | 0.0047 | 0.0054 | 0.0541 / 0.0555 | 0.0778 / 0.0556 | 0.110 (0.108 / 0.111) | 1.00 | 9.77e-04 / 0.00 / 1.25e-03 | 1.00, 3.0 | 0.500 | 0.0581 / 0.0585 | 0.945 (0.687) |
| L_weak_mala_over | 0.0426 | 0.0431 | 0.218 | 0.221 | 0.0047 | 0.0055 | 0.0562 / 0.0562 | 0.0808 / 0.0563 | 0.112 (0.112 / 0.112) | 1.00 | 9.77e-04 / 0.00 / 1.25e-03 | 1.00, 3.1 | 0.499 | 0.0581 / 0.0585 | 0.963 (0.727) |
| L_weak_pcnl_map | 0.0426 | 0.0431 | 0.218 | 0.220 | 0.0047 | 0.0053 | 0.0532 / 0.0544 | 0.0766 / 0.0546 | 0.108 (0.106 / 0.109) | 1.00 | 1.95e-03 / 0.00 / 2.50e-03 | 1.00, 3.5 | 0.502 | 0.0581 / 0.0585 | 0.928 (0.678) |
| L_weak_hmc_map | 0.0426 | 0.0426 | 0.218 | 0.218 | 0.0047 | 0.0057 | 0.0586 / 0.0584 | 0.0842 / 0.0585 | 0.117 (0.117 / 0.117) | 1.00 | 0.00 / 0.00 / 0.00 | n/a, 2.6 | 0.502 | 0.0581 / 0.0585 | 1.000 (0.839) |
| L_weak_hmc_over | 0.0426 | 0.0423 | 0.218 | 0.217 | 0.0047 | 0.0056 | 0.0567 / 0.0578 | 0.0816 / 0.0580 | 0.115 (0.113 / 0.116) | 1.00 | 0.00 / 0.00 / 0.00 | n/a, 2.5 | 0.500 | 0.0581 / 0.0585 | 0.986 (0.801) |

Pooled across initializations `L_weak_mala_pool` (4 chains, runs L_weak_mala_map, L_weak_mala_over): split R-hat of Phi 1.03; of log eps_r over pixels: median 1.05, max 1.29, fraction of pixels with R-hat < 1.1: 0.83.

Pooled across initializations `L_weak_hmc_pool` (4 chains, runs L_weak_hmc_map, L_weak_hmc_over): split R-hat of Phi 1.00; of log eps_r over pixels: median 1.01, max 1.04, fraction of pixels with R-hat < 1.1: 1.00.
