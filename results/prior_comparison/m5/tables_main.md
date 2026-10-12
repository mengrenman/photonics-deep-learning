### setup (any summary.json; operating point from results/prior_comparison/pc_matern/summary.json)

n=48, side=2.0, eps_r=2.0, SNR=30.0 dB, seed=0, N=2400, noise std s=3.9228e-04, d_obs sha256 efd5e84fa5f9a77f; grid error 0.85% of ||d||, noise 3.16%, ratio 0.269; Phi(truth) 2578.1, noise-free-data Phi 2438.2

### MAP and Laplace

| run | MAP its | evals | U | Phi | R | |grad U|inf | Laplace sd in | Laplace sd out | Jacobian check |
|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 350 | 366 | 2465.9 | 2378.4 | 87.5 | 5.9e-03 | 0.0543 | 0.0608 | 1.4e-10 |
| pc_qggmrf_sg0.05 | 134 | 145 | 5193.2 | 2474.3 | 2718.8 | 1.0e-02 | 0.0463 | 0.0331 | 1.6e-10 |
| pc_qggmrf_sg0.1 | 191 | 208 | 3467.1 | 2406.5 | 1060.6 | 5.8e-03 | 0.0752 | 0.0577 | 1.5e-10 |
| pc_gmrf_adapt | 702 | 734 | 2400.1 | 2367.0 | 33.1 | 9.1e-03 | 0.3449 | 0.3503 | 1.4e-10 |
| pc_tv_lam3.75_adapt | 961 | 1006 | 3520.3 | 2399.5 | 1120.8 | 8.6e-03 | 0.0725 | 0.0387 | 1.2e-10 |
| pc_tv_lam11.25_adapt | 691 | 735 | 5552.2 | 2457.9 | 3094.3 | 3.9e-03 | 0.0392 | 0.0238 | 1.2e-10 |
| pc_tv_lam1.25_adapt | 1922 | 2005 | 2823.1 | 2381.4 | 441.7 | 5.2e-03 | 0.1370 | 0.0671 | 9.7e-11 |
| pc_tv_lam8.5_adapt | 743 | 776 | 4813.9 | 2434.7 | 2379.1 | 1.6e-02 | 0.0464 | 0.0270 | 1.0e-10 |
| pc_tv_lam3.75_indep | 961 | 1006 | 3520.3 | 2399.5 | 1120.8 | 8.6e-03 | 0.0725 | 0.0387 | 1.2e-10 |

### sampler and convergence (DESIGN 11.2: R-hat Phi <= 1.01, max pixel R-hat <= 1.05, worst-pixel ESS >= 200)

| run | chains x (warmup + draws) | accept | Phi mean | Phi sd | Phi 1st quarter | Phi last quarter | R-hat Phi | max pixel R-hat | worst ESS (Sokal) | worst ESS (Geyer) | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 4 x (300 + 600) | 0.660 | 2434.7 | 7.6 | 2434.5 | 2436.1 | 1.002 | 1.009 | 883 | 906 | yes |
| pc_qggmrf_sg0.05 | 4 x (300 + 900) | 0.707 | 2511.4 | 13.0 | 2512.5 | 2511.0 | 1.003 | 1.005 | 2142 | 2070 | yes |
| pc_qggmrf_sg0.1 | 4 x (300 + 600) | 0.705 | 2452.4 | 9.0 | 2451.6 | 2451.7 | 1.001 | 1.006 | 1242 | 1233 | yes |
| pc_gmrf_adapt | 4 x (50 + 600) | 0.861 | 2429.6 | 7.6 | 2429.0 | 2430.1 | 1.000 | 1.064 | 231 | 216 | NO |
| pc_tv_lam3.75_adapt | 4 x (50 + 800) | 0.847 | 2442.3 | 8.5 | 2442.3 | 2442.4 | 1.004 | 1.319 | 98 | 87 | NO |
| pc_tv_lam11.25_adapt | 4 x (50 + 400) | 0.904 | 2494.1 | 11.5 | 2495.7 | 2493.6 | 1.001 | 1.142 | 75 | 72 | NO |
| pc_tv_lam1.25_adapt | 4 x (50 + 100) | 0.923 | 2430.6 | 7.9 | 2430.8 | 2431.1 | 0.999 | 2.909 | 36 | 21 | NO |
| pc_tv_lam8.5_adapt | 4 x (50 + 100) | 0.885 | 2486.9 | 23.8 | 2486.3 | 2488.8 | 2.341 | 16.063 | 32 | 18 | NO |
| pc_tv_lam3.75_indep | 2 x (50 + 200) | 0.850 | 2457.7 | 17.7 | 2460.6 | 2453.8 | 2.013 | 14.642 | 16 | 8 | NO |

pc_gmrf_adapt: max pixel R-hat 1.064 > 1.05

pc_tv_lam3.75_adapt: max pixel R-hat 1.319 > 1.05; worst-pixel ESS 98 < 200

pc_tv_lam11.25_adapt: max pixel R-hat 1.142 > 1.05; worst-pixel ESS 75 < 200

pc_tv_lam1.25_adapt: max pixel R-hat 2.909 > 1.05; worst-pixel ESS 36 < 200

pc_tv_lam8.5_adapt: split R-hat of Phi 2.341 > 1.01; max pixel R-hat 16.063 > 1.05; worst-pixel ESS 32 < 200

pc_tv_lam3.75_indep: split R-hat of Phi 2.013 > 1.01; max pixel R-hat 14.642 > 1.05; worst-pixel ESS 16 < 200

### posterior standard deviation of theta = log eps_r

| run | prior sd in | prior sd out | post sd in | post sd out | post/prior in | post/prior out | post/prior mean | MCMC/Laplace mean | min | max | mean(theta) - MAP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 0.5000 | 0.5000 | 0.0542 | 0.0607 | 0.108 | 0.121 | 0.119 | 0.999 | 0.890 | 1.105 | -0.0022 |
| pc_qggmrf_sg0.05 | 0.0519 | 0.0578 | 0.0488 | 0.0403 | 0.942 | 0.709 | 0.760 | 1.230 | 0.786 | 1.646 | -0.0008 |
| pc_qggmrf_sg0.1 | 0.0920 | 0.1007 | 0.0817 | 0.0742 | 0.891 | 0.745 | 0.777 | 1.278 | 0.822 | 1.709 | -0.0031 |
| pc_gmrf_adapt | 0.5000 | 0.5000 | 0.3312 | 0.3415 | 0.662 | 0.683 | 0.678 | 0.972 | 0.814 | 1.402 | -0.0620 |
| pc_tv_lam3.75_adapt | 0.1008 | 0.1102 | 0.1039 | 0.0856 | 1.031 | 0.786 | 0.839 | 2.585 | 0.172 | 6.223 | -0.0013 |
| pc_tv_lam11.25_adapt | 0.0432 | 0.0475 | 0.0543 | 0.0361 | 1.255 | 0.776 | 0.880 | 1.818 | 0.203 | 3.755 | 0.0015 |
| pc_tv_lam1.25_adapt | 0.2478 | 0.2613 | 0.2237 | 0.2212 | 0.903 | 0.851 | 0.863 | 3.742 | 0.245 | 12.363 | -0.0263 |
| pc_tv_lam8.5_adapt | 0.0526 | 0.0583 | 0.1461 | 0.0919 | 2.781 | 1.613 | 1.866 | 4.415 | 0.215 | 36.173 | -0.0094 |
| pc_tv_lam3.75_indep | 0.1008 | 0.1102 | 0.1777 | 0.1386 | 1.763 | 1.273 | 1.379 | 4.359 | 0.229 | 32.814 | -0.0223 |

### misfit to the truth: |z| = |mean - truth| / std in log eps_r; reconstruction errors

| run | %|z|>3 | % inside | % outside | edge share | edge share by chance | max |z| | rms z | err_model MAP | err_model mean | err_chi MAP | err_chi mean | err_data mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 12.5 | 27.0 | 8.4 | 0.882 | 0.233 | 8.41 | 1.97 | 0.1260 | 0.1263 | 0.3868 | 0.3879 | 0.0062 |
| pc_qggmrf_sg0.05 | 15.5 | 38.4 | 9.1 | 0.810 | 0.233 | 12.57 | 2.35 | 0.1322 | 0.1345 | 0.4059 | 0.4129 | 0.0074 |
| pc_qggmrf_sg0.1 | 5.8 | 13.2 | 3.8 | 0.918 | 0.233 | 6.71 | 1.39 | 0.1296 | 0.1316 | 0.3980 | 0.4040 | 0.0062 |
| pc_gmrf_adapt | 0.0 | 0.0 | 0.0 | - | 0.233 | 1.41 | 0.38 | 0.1246 | 0.1271 | 0.3825 | 0.3902 | 0.0346 |
| pc_tv_lam3.75_adapt | 2.2 | 6.4 | 1.1 | 0.980 | 0.233 | 5.56 | 1.09 | 0.1204 | 0.1321 | 0.3697 | 0.4056 | 0.0063 |
| pc_tv_lam11.25_adapt | 12.5 | 40.8 | 4.7 | 0.774 | 0.233 | 13.03 | 2.14 | 0.1324 | 0.1356 | 0.4065 | 0.4163 | 0.0069 |
| pc_tv_lam1.25_adapt | 0.0 | 0.0 | 0.0 | - | 0.233 | 2.56 | 0.54 | 0.1129 | 0.1348 | 0.3467 | 0.4140 | 0.0182 |
| pc_tv_lam8.5_adapt | 1.7 | 4.8 | 0.8 | 0.974 | 0.233 | 6.38 | 1.08 | 0.1284 | 0.1519 | 0.3941 | 0.4663 | 0.0227 |
| pc_tv_lam3.75_indep | 0.7 | 2.4 | 0.2 | 0.938 | 0.233 | 4.42 | 1.01 | 0.1204 | 0.2195 | 0.3697 | 0.6740 | 0.0293 |

### feature probabilities (delta = log2/2) from the stored draws; converged runs: (+-mcse) = spread of the per-chain estimates over sqrt(chains); unconverged runs: [min-max] over the chains (a standard error from unconverged chains understates the uncertainty)

| run | converged | gap_valley | ring_hole | ring_continuity |
|---|---|---|---|---|
| pc_matern | yes | 1.000 (+-0.000) | 0.984 (+-0.003) | 1.000 (+-0.000) |
| pc_qggmrf_sg0.05 | yes | 1.000 (+-0.000) | 1.000 (+-0.000) | 1.000 (+-0.000) |
| pc_qggmrf_sg0.1 | yes | 1.000 (+-0.000) | 1.000 (+-0.000) | 1.000 (+-0.000) |
| pc_gmrf_adapt | no | 0.920 [0.898-0.937] | 0.713 [0.668-0.752] | 0.501 [0.487-0.520] |
| pc_tv_lam3.75_adapt | no | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 1.000 [0.999-1.000] |
| pc_tv_lam11.25_adapt | no | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] |
| pc_tv_lam1.25_adapt | no | 1.000 [1.000-1.000] | 0.895 [0.740-1.000] | 0.728 [0.150-0.980] |
| pc_tv_lam8.5_adapt | no | 1.000 [1.000-1.000] | 1.000 [1.000-1.000] | 0.750 [0.000-1.000] |
| pc_tv_lam3.75_indep | no | 0.978 [0.955-1.000] | 1.000 [1.000-1.000] | 0.557 [0.120-0.995] |

truth satisfies: gap_valley: True, ring_hole: True, ring_continuity: True

### verdicts and disagreement (comparison.json; 3 converged runs of 6)

| feature | verdict |
|---|---|
| gap_valley | not established: unconverged gmrf disagrees with the converged runs (present) |
| ring_hole | not established: unconverged gmrf disagrees with the converged runs (present) |
| ring_continuity | not established: unconverged gmrf disagrees with the converged runs (present) |

| pair | both converged | % pixels >2 | inside | outside | near edge | away from edge | max | mean |
|---|---|---|---|---|---|---|---|---|
| pc_matern vs pc_qggmrf_sg0.05 | yes | 0.2 | 0.0 | 0.2 | 0.0 | 0.2 | 2.33 | 0.37 |
| pc_matern vs pc_qggmrf_sg0.1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.37 | 0.20 |
| pc_matern vs pc_gmrf_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.57 | 0.18 |
| pc_matern vs pc_tv_lam3.75_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.22 | 0.22 |
| pc_matern vs pc_tv_lam11.25_adapt | no | 0.3 | 0.6 | 0.2 | 0.6 | 0.2 | 2.22 | 0.50 |
| pc_qggmrf_sg0.05 vs pc_qggmrf_sg0.1 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.51 | 0.12 |
| pc_qggmrf_sg0.05 vs pc_gmrf_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.80 | 0.20 |
| pc_qggmrf_sg0.05 vs pc_tv_lam3.75_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.60 | 0.13 |
| pc_qggmrf_sg0.05 vs pc_tv_lam11.25_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.07 | 0.25 |
| pc_qggmrf_sg0.1 vs pc_gmrf_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.68 | 0.19 |
| pc_qggmrf_sg0.1 vs pc_tv_lam3.75_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.37 | 0.08 |
| pc_qggmrf_sg0.1 vs pc_tv_lam11.25_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.08 | 0.25 |
| pc_gmrf_adapt vs pc_tv_lam3.75_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.63 | 0.18 |
| pc_gmrf_adapt vs pc_tv_lam11.25_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.78 | 0.21 |
| pc_tv_lam3.75_adapt vs pc_tv_lam11.25_adapt | no | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.88 | 0.18 |

### robustness controls (summary.json of each)

| run | seed / data factor | chains x (warmup + draws) | grid error / noise | Phi mean | R-hat Phi | %|z|>3 | edge share | converged |
|---|---|---|---|---|---|---|---|---|
| pc_matern | 0 / 2 | 4 x (300 + 600) | 0.269 | 2434.7 | 1.002 | 12.5 | 0.882 | yes |
| pc_qggmrf_sg0.05 | 0 / 2 | 4 x (300 + 900) | 0.269 | 2511.4 | 1.003 | 15.5 | 0.810 | yes |
| pc_qggmrf_sg0.1 | 0 / 2 | 4 x (300 + 600) | 0.269 | 2452.4 | 1.001 | 5.8 | 0.918 | yes |
| pc_gmrf_adapt | 0 / 2 | 4 x (50 + 600) | 0.269 | 2429.6 | 1.000 | 0.0 | - | NO |
| pc_tv_lam3.75_adapt | 0 / 2 | 4 x (50 + 800) | 0.269 | 2442.3 | 1.004 | 2.2 | 0.980 | NO |
| pc_tv_lam11.25_adapt | 0 / 2 | 4 x (50 + 400) | 0.269 | 2494.1 | 1.001 | 12.5 | 0.774 | NO |
| pc_tv_lam1.25_adapt | 0 / 2 | 4 x (50 + 100) | 0.269 | 2430.6 | 0.999 | 0.0 | - | NO |
| pc_tv_lam8.5_adapt | 0 / 2 | 4 x (50 + 100) | 0.269 | 2486.9 | 2.341 | 1.7 | 0.974 | NO |
| pc_tv_lam3.75_indep | 0 / 2 | 2 x (50 + 200) | 0.269 | 2457.7 | 2.013 | 0.7 | 0.938 | NO |
| pcs1_qggmrf_sg0.05 | 1 / 2 | 2 x (300 + 600) | 0.269 | 2500.7 | 1.010 | 16.1 | 0.798 | yes |
| pcg1_qggmrf_sg0.05 | 0 / 1 | 3 x (300 + 600) | 0.000 | 2507.6 | 1.001 | 16.3 | 0.798 | yes |
| pcg1_matern | 0 / 1 | 3 x (300 + 600) | 0.000 | 2434.4 | 1.004 | 12.3 | 0.883 | yes |

### prior energy R and data misfit Phi at the truth, the MAP and the posterior draws (prior_energy.json)

| run | R truth | R MAP | R draws | (R truth - R draws) / sd | Phi truth | Phi noise-free data | Phi MAP | Phi draws | U truth - U MAP |
|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 14685.5 | 87.5 | 1180.7 +- 32.9 | 409.9 | 2578.1 | 2438.2 | 2378.4 | 2434.7 +- 7.6 | 14797.8 |
| pc_qggmrf_sg0.05 | 4406.2 | 2718.8 | 3998.7 +- 38.7 | 10.5 | 2578.1 | 2438.2 | 2474.3 | 2511.4 +- 13.0 | 1791.2 |
| pc_qggmrf_sg0.1 | 1772.7 | 1060.6 | 2357.0 +- 38.3 | -15.3 | 2578.1 | 2438.2 | 2406.5 | 2452.4 +- 9.0 | 883.7 |
| pc_gmrf | 89.1 | 33.1 | 524.7 +- 45.7 | -9.5 | 2578.1 | 2438.2 | 2367.0 | 2427.7 +- 8.4 | 267.1 |
| pc_gmrf_adapt | 89.1 | 33.1 | 1085.0 +- 42.0 | -23.7 | 2578.1 | 2438.2 | 2367.0 | 2429.6 +- 7.6 | 267.1 |
| pc_gmrf_adapt_burn2 | 89.1 | 33.1 | 1101.7 +- 32.7 | -31.0 | 2578.1 | 2438.2 | 2367.0 | 2429.7 +- 7.7 | 267.1 |
| pc_gmrf_adapt2 | 89.1 | 33.1 | 1046.8 +- 57.3 | -16.7 | 2578.1 | 2438.2 | 2367.0 | 2447.7 +- 11.0 | 267.1 |
| pc_tv_lam3.75_adapt | 1334.6 | 1120.8 | 2859.6 +- 42.8 | -35.6 | 2578.1 | 2438.2 | 2399.5 | 2442.3 +- 8.5 | 392.5 |
| pc_tv_lam3.75_adapt2 | 1334.6 | 1120.8 | 2811.0 +- 61.7 | -23.9 | 2578.1 | 2438.2 | 2399.5 | 2441.4 +- 8.3 | 392.5 |
| pc_tv_lam11.25_adapt | 3821.8 | 3094.3 | 4713.7 +- 41.5 | -21.5 | 2578.1 | 2438.2 | 2457.9 | 2494.1 +- 11.5 | 847.8 |
| pc_tv_lam11.25_adapt2 | 3821.8 | 3094.3 | 4648.5 +- 75.7 | -10.9 | 2578.1 | 2438.2 | 2457.9 | 2493.2 +- 11.2 | 847.8 |
| pc_tv_lam1.25_adapt | 505.5 | 441.7 | 2284.9 +- 307.6 | -5.8 | 2578.1 | 2438.2 | 2381.4 | 2430.6 +- 7.9 | 260.5 |
| pc_tv_lam8.5_adapt | 2909.8 | 2379.1 | 5309.9 +- 2258.4 | -1.1 | 2578.1 | 2438.2 | 2434.7 | 2486.9 +- 23.9 | 674.1 |
| pc_tv_lam3.75_indep | 1334.6 | 1120.8 | 3841.0 +- 994.9 | -2.5 | 2578.1 | 2438.2 | 2399.5 | 2457.7 +- 17.8 | 392.5 |

### R and Phi of the truth blurred by a Gaussian of sigma pixels (prior_energy.json; Phi uses each run's data)

| quantity | sigma 0 | sigma 0.5 | sigma 1 | sigma 1.5 | sigma 2 |
|---|---|---|---|---|---|
| pc_matern R | 14685.5 | 4311.2 | 274.1 | 79.6 | 28.5 |
| pc_matern Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_qggmrf_sg0.05 R | 4406.2 | 3769.6 | 2801.2 | 2020.1 | 1435.1 |
| pc_qggmrf_sg0.05 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_qggmrf_sg0.1 R | 1772.7 | 1485.4 | 1061.1 | 745.3 | 518.8 |
| pc_qggmrf_sg0.1 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_gmrf R | 89.1 | 60.4 | 30.8 | 18.2 | 11.2 |
| pc_gmrf Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_gmrf_adapt R | 89.1 | 60.4 | 30.8 | 18.2 | 11.2 |
| pc_gmrf_adapt Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_gmrf_adapt_burn2 R | 89.1 | 60.4 | 30.8 | 18.2 | 11.2 |
| pc_gmrf_adapt_burn2 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_gmrf_adapt2 R | 89.1 | 60.4 | 30.8 | 18.2 | 11.2 |
| pc_gmrf_adapt2 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam3.75_adapt R | 1334.6 | 1277.5 | 1132.0 | 932.7 | 748.9 |
| pc_tv_lam3.75_adapt Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam3.75_adapt2 R | 1334.6 | 1277.5 | 1132.0 | 932.7 | 748.9 |
| pc_tv_lam3.75_adapt2 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam11.25_adapt R | 3821.8 | 3663.3 | 3252.6 | 2676.5 | 2143.1 |
| pc_tv_lam11.25_adapt Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam11.25_adapt2 R | 3821.8 | 3663.3 | 3252.6 | 2676.5 | 2143.1 |
| pc_tv_lam11.25_adapt2 Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam1.25_adapt R | 505.5 | 482.3 | 425.1 | 351.4 | 284.2 |
| pc_tv_lam1.25_adapt Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam8.5_adapt R | 2909.8 | 2788.5 | 2475.1 | 2037.1 | 1631.9 |
| pc_tv_lam8.5_adapt Phi | 2578 | 6328 | 35197 | 89229 | 160749 |
| pc_tv_lam3.75_indep R | 1334.6 | 1277.5 | 1132.0 | 932.7 | 748.9 |
| pc_tv_lam3.75_indep Phi | 2578 | 6328 | 35197 | 89229 | 160749 |

### posterior-mean map (prior_energy.json)

| run | R mean map | R truth - R mean map | Phi mean map | Phi mean map - Phi truth |
|---|---|---|---|---|
| pc_matern | 87.9 | 14597.6 | 2860.6 | 282.4 |
| pc_qggmrf_sg0.05 | 2755.2 | 1651.0 | 2587.8 | 9.7 |
| pc_qggmrf_sg0.1 | 1079.9 | 692.8 | 3499.9 | 921.7 |
| pc_gmrf | 122.1 | -33.0 | 29887.6 | 27309.5 |
| pc_gmrf_adapt | 34.8 | 54.2 | 355550.8 | 352972.6 |
| pc_gmrf_adapt_burn2 | 35.7 | 53.3 | 376715.9 | 374137.7 |
| pc_gmrf_adapt2 | 41.4 | 47.7 | 428812.2 | 426234.1 |
| pc_tv_lam3.75_adapt | 1321.6 | 13.0 | 4813.5 | 2235.4 |
| pc_tv_lam3.75_adapt2 | 1330.7 | 3.9 | 4648.3 | 2070.1 |
| pc_tv_lam11.25_adapt | 3349.8 | 472.0 | 2611.5 | 33.4 |
| pc_tv_lam11.25_adapt2 | 3358.7 | 463.1 | 2597.1 | 19.0 |
| pc_tv_lam1.25_adapt | 734.8 | -229.2 | 77809.2 | 75231.0 |
| pc_tv_lam8.5_adapt | 3543.5 | -633.7 | 20213.8 | 17635.7 |
| pc_tv_lam3.75_indep | 2139.7 | -805.1 | 47295.9 | 44717.8 |

### cost of a step of height log 2: one jump against a ramp (per pair line, prior_energy.json)

| potential | sharp | ramp 2 px | ramp 4 px | ramp 8 px | sharp / ramp 8 |
|---|---|---|---|---|---|
| q-GGMRF q 1.2, T 1, sigma_g 0.05 (experiment) | 10.45 | 8.42 | 6.49 | 4.71 | 2.22 |
| q-GGMRF q 1.2, T 1, sigma_g 0.1 (experiment) | 4.21 | 3.24 | 2.35 | 1.59 | 2.65 |
| q 1.0, T 1, sigma_g 0.05 | 6.47 | 6.06 | 5.38 | 4.40 | 1.47 |
| q 1.0, T sigma_g = log 2 (sigma_g 0.05) | 48.05 | 32.03 | 19.22 | 10.68 | 4.50 |
| q 1.0, T 0.1, sigma_g 0.05 (near total variation) | 0.69 | 0.68 | 0.67 | 0.66 | 1.05 |
| q 1.0, T 0.01, sigma_g 0.05 (total variation) | 0.07 | 0.07 | 0.07 | 0.07 | 1.01 |
| q 1.5, T 1, sigma_g 0.05 | 20.34 | 13.23 | 8.39 | 5.19 | 3.92 |
| TV pseudo-Huber lam 1, delta 0.01 (DESIGN 12.1) | 0.68 | 0.67 | 0.65 | 0.62 | 1.11 |
| TV pseudo-Huber lam 1, delta 0.003 | 0.69 | 0.69 | 0.68 | 0.67 | 1.03 |
| TV pseudo-Huber lam 1, delta 0.001 | 0.69 | 0.69 | 0.69 | 0.69 | 1.01 |

### convergence audit: rank-normalized R-hat, ESS and the AR(1) null (convergence_audit.json)

| run | chains x draws | R-hat Phi | rank R-hat Phi | null P(R-hat Phi > 1.01) | max pixel R-hat | max pixel rank R-hat | null q95 of it | pixels > 1.05 (near edge) | worst ESS Sokal | worst ESS bulk | tau Phi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pc_matern | 4 x 600 | 1.0018 | 1.0022 | 0.03 | 1.0092 | 1.0221 | 1.0370 | 0 (0) | 883 | 774 | 3.4 |
| pc_qggmrf_sg0.05 | 4 x 900 | 1.0028 | 1.0106 | 0.00 | 1.0046 | 1.0249 | 1.0389 | 0 (0) | 2142 | 1968 | 2.8 |
| pc_qggmrf_sg0.1 | 4 x 600 | 1.0012 | 1.0011 | 0.02 | 1.0062 | 1.0405 | 1.0495 | 0 (0) | 1242 | 1211 | 3.2 |
| pc_gmrf | 4 x 500 | 1.0327 | 1.0330 | 0.90 | 1.0906 | 1.0898 | 1.1206 | 46 (3) | 127 | 33 | 13.2 |
| pc_gmrf_adapt | 4 x 600 | 1.0002 | 1.0003 | 0.00 | 1.0643 | 1.0624 | 1.0913 | 7 (3) | 231 | 58 | 2.1 |
| pc_gmrf_adapt_burn2 | 4 x 400 | 1.0005 | 1.0031 | 0.00 | 1.0900 | 1.0875 | 1.0742 | 24 (10) | 154 | 38 | 2.0 |
| pc_gmrf_adapt2 | 4 x 200 | 1.0318 | 1.0325 | 0.62 | 1.6231 | 1.5401 | 1.2224 | 766 (192) | 52 | 7 | 3.5 |
| pc_tv_lam3.75_adapt | 4 x 800 | 1.0037 | 1.0039 | 0.02 | 1.3188 | 1.1863 | 1.1279 | 72 (6) | 98 | 15 | 4.0 |
| pc_tv_lam3.75_adapt2 | 4 x 500 | 1.0107 | 1.0102 | 0.23 | 1.2019 | 1.1579 | 1.1813 | 246 (4) | 68 | 20 | 4.7 |
| pc_tv_lam11.25_adapt | 4 x 400 | 1.0008 | 1.0044 | 0.28 | 1.1421 | 1.1308 | 1.1919 | 111 (3) | 75 | 22 | 3.5 |
| pc_tv_lam11.25_adapt2 | 4 x 200 | 1.0605 | 1.0601 | 0.80 | 1.6354 | 1.5009 | 1.2399 | 597 (22) | 48 | 7 | 5.5 |
| pc_tv_lam1.25_adapt | 4 x 100 | 0.9987 | 1.0101 | 0.28 | 2.9088 | 2.1707 | 1.3466 | 1822 (359) | 36 | 5 | 1.5 |
| pc_tv_lam8.5_adapt | 4 x 100 | 2.3405 | 1.5760 | 0.87 | 16.0630 | 2.8551 | 1.3097 | 2267 (522) | 32 | 5 | 3.6 |
| pc_tv_lam3.75_indep | 2 x 200 | 2.0132 | 1.8332 | 0.90 | 14.6416 | 2.5122 | 1.6194 | 2177 (505) | 16 | 2 | 11.3 |

### edge miscoverage per run and per chain (edge_analysis.json)

| run | converged | chains x draws | %|z|>3 pooled | per chain min to max | MC error (pooled, %) | bias d=-1 | sd d=-1 | % |z|>3 at d=-1 | % |z|>3 in |d|=1 |
|---|---|---|---|---|---|---|---|---|---|
| pc_matern | yes | 4 x 600 | 12.5 | 12.2 to 12.6 | 0.08 | 0.215 | 0.061 | 61.3 | 49.4 |
| pc_qggmrf_sg0.05 | yes | 4 x 900 | 15.5 | 15.6 to 15.7 | 0.02 | 0.208 | 0.052 | 64.5 | 54.0 |
| pc_qggmrf_sg0.1 | yes | 4 x 600 | 5.8 | 5.9 to 6.4 | 0.11 | 0.204 | 0.083 | 27.4 | 23.1 |
| pc_gmrf | NO | 4 x 500 | 0.4 | 0.3 to 0.4 | 0.02 | 0.177 | 0.179 | 2.4 | 1.8 |
| pc_gmrf_adapt | NO | 4 x 600 | 0.0 | 0.0 to 0.0 | 0.00 | 0.134 | 0.338 | 0.0 | 0.0 |
| pc_gmrf_adapt_burn2 | NO | 4 x 400 | 0.0 | 0.0 to 0.0 | 0.00 | 0.131 | 0.343 | 0.0 | 0.0 |
| pc_gmrf_adapt2 | NO | 4 x 200 | 0.0 | 0.0 to 0.0 | 0.00 | 0.130 | 0.347 | 0.0 | 0.0 |
| pc_tv_lam3.75_adapt | NO | 4 x 800 | 2.2 | 2.3 to 2.5 | 0.05 | 0.189 | 0.106 | 7.7 | 8.8 |
| pc_tv_lam3.75_adapt2 | NO | 4 x 500 | 2.4 | 2.3 to 2.7 | 0.08 | 0.189 | 0.106 | 7.7 | 9.2 |
| pc_tv_lam11.25_adapt | NO | 4 x 400 | 12.5 | 12.5 to 12.9 | 0.12 | 0.180 | 0.063 | 32.7 | 37.3 |
| pc_tv_lam11.25_adapt2 | NO | 4 x 200 | 12.9 | 13.2 to 13.6 | 0.12 | 0.179 | 0.062 | 32.7 | 37.8 |
| pc_tv_lam1.25_adapt | NO | 4 x 100 | 0.0 | 0.0 to 0.3 | 0.06 | 0.160 | 0.226 | 0.0 | 0.0 |
| pc_tv_lam8.5_adapt | NO | 4 x 100 | 1.7 | 3.1 to 29.8 | 5.72 | 0.178 | 0.120 | 6.0 | 6.8 |
| pc_tv_lam3.75_indep | NO | 2 x 200 | 0.7 | 1.8 to 10.2 | 4.19 | 0.178 | 0.144 | 1.6 | 2.8 |
