**Efficiency and convergence (c_book).** Eval = one Phi evaluation (all transmitters); a gradient evaluation adds one adjoint solve (pCN has none). Phi = data misfit of the stored draws, N = number of complex data; 'Phi first/last quarter' are the means over the first and last quarter of the draws (drift check). ESS from Sokal windows (with Geyer's estimator of the worst pixel as a cross-check), summed over chains; 'pix min' = smallest ESS over all pixels of log eps_r. 'incl. setup' adds the MAP evaluations and, for Laplace runs, the Gauss-Newton products (one gradient evaluation each) to the evaluations. ESS values in parentheses come from chains that have not converged (split R-hat of Phi above 1.1 or of a pixel above 2): they carry no information about efficiency. Seconds are wall time of runs that shared the machine (load average at start in parentheses), so rates per second are indicative only.

| run | chains x (warmup+samples), thin | accept | final step | evals / chain | Phi mean (MAP, N) | Phi first / last quarter | ESS Phi | ESS Phi / 1e3 eval | ESS pix min | ESS pix min / 1e3 eval | ESS pix min / 1e3 eval incl. setup | ESS pix min (Geyer) | R-hat Phi | R-hat pix max | wall s (load) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c_mala_map | 2 x (40+160), 2 | 0.000 | 3.68e-07 | 201 | 3.74e+05 (3.71e+05, 2400) | 3.74e+05 / 3.74e+05 | (nan) | (nan) | (nan) | (nan) | (nan) | (nan) | 2.43e+13 | nan | 749 (n/a) |
| c_book96_mala_map | 2 x (20+60), 2 | 0.358 | 3.20e-06 | 81 | 38115.5 (38068.8, 2400) | 38120.5 / 38113.9 | (20.8) | (128.200) | (14.7) | (90.726) | (24.619) | (5.7) | 1.39 | 15.32 | 1240 (42) |

Operating point of each run's inference grid at the truth: grid model error ||F_n(chi_true) - d_clean|| / ||d_clean|| against the noise amplitude s sqrt(N) / ||d_clean||, and Phi(chi_true) on the inference grid against N and against the Phi of the noise-free fine-grid data (the realized noise energy), in sd of Phi (= sqrt(N)).

| run | n, side, data grid | model error % | noise % | ratio | Phi(truth) | Phi(noise-free data) | excess (sd) |
|---|---|---|---|---|---|---|---|
| c_mala_map | 48, 5, 96 | 8.01 | 3.16 | 2.53 | 17990 | 2438 | 317.4 |
| c_book96_mala_map | 96, 5, 192 | 2.58 | 10.00 | 0.26 | 2606 | 2438 | 3.4 |

**Reconstruction and uncertainty (c_book).** err_model is eq. 15.21 on eps_r (vacuum = 0.3257); err_chi is contrast-normalized; err_data is eq. 15.22. z = (posterior mean - truth) / posterior std in log eps_r. 'std ratio' = posterior std of log eps_r / prior std sigma = 0.5. P(eps<1) is averaged over true-background pixels (prior value 0.5).

| run | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data MAP | err_data mean | std log eps_r in / out | std eps_r in / out | std ratio all (in / out) | frac std ratio < 0.5 | frac abs z > 3 (all / in / out) | edge share of abs z > 3, max abs z | P(eps<1) bg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c_mala_map | 0.7052 | 0.7054 | 2.165 | 2.166 | 0.3920 | 0.3929 | 0.0034 / 0.0031 | 0.0036 / 0.0031 | 0.006 (0.007 / 0.006) | 1.00 | 0.98 / 1.00 / 0.98 | 0.24, 1.40e+05 | 0.623 |
| c_book96_mala_map | 0.6448 | 0.6448 | 1.943 | 1.943 | 0.3880 | 0.3880 | 0.0025 / 0.0028 | 0.0023 / 0.0028 | 0.006 (0.005 / 0.006) | 1.00 | 0.98 / 1.00 / 0.97 | 0.12, 2224.0 | 0.633 |
