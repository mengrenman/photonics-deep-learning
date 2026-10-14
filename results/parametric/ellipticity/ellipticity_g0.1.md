# T2 ellipticity ladder (F2 fit, Laplace interval; the noise realization is the same at every level)

p (Gamma null) = P(Gamma(N - 5/2) >= Phi_MAP) is the flag; the refit p-value is from replicates and 0.000 means below its resolution. dPhi = Phi_MAP minus the misfit of this noise realization itself (+0.78 sd high for seed 0); c = dPhi / z^2 (z of truth) and c vs level 0 = dPhi / dz^2 with dz the shift of the gap from the level-0 fit of the same noise.

| ellipticity | semi-axes | Phi_MAP/N | Phi_MAP - N | p (Gamma null) | z of Phi | refit p (30 reps, resolution 1/31) | refit z | g MAP | g sd | g error | z of truth | covers | dPhi (excess over noise-only) | c = dPhi / z^2 | c vs level 0 | log Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2 x 0.2000 | 1.015 | 37 | 0.21 | 0.8 | 0.167 | 1.0 | 0.10029 | 2.80e-04 | +0.00029 | -1.0 | yes | -1.6 | - | - | 35460.5 |
| 0.005 | 0.2 x 0.2010 | 1.023 | 56 | 0.12 | 1.2 | 0.033 | 1.5 | 0.09944 | 2.80e-04 | -0.00056 | 2.0 | no | 17.8 | - | 1.95 | 35426.6 |
| 0.01 | 0.2 x 0.2020 | 1.046 | 111 | 0.011 | 2.3 | 0.000 | 2.9 | 0.09859 | 2.80e-04 | -0.00141 | 5.0 | no | 72.3 | 2.87 | 1.97 | 35357.7 |
| 0.02 | 0.2 x 0.2040 | 1.136 | 327 | 5.5e-11 | 6.7 | 0.000 | 8.6 | 0.09689 | 2.81e-04 | -0.00311 | 11.1 | no | 288.9 | 2.36 | 1.97 | 35112.0 |
| 0.05 | 0.2 x 0.2100 | 1.783 | 1878 | 1.7e-216 | 38.4 | 0.000 | 49.2 | 0.09187 | 2.82e-04 | -0.00813 | 28.8 | no | 1839.8 | 2.22 | 2.07 | 33472.7 |
| 0.15 | 0.2 x 0.2300 | 8.372 | 17694 | 0 | 361.4 | 0.000 | 463.3 | 0.07598 | 2.89e-04 | -0.02402 | 83.2 | no | 17655.4 | 2.55 | 2.49 | 17352.3 |
