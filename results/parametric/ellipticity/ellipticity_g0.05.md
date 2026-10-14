# T2 ellipticity ladder (F2 fit, Laplace interval; the noise realization is the same at every level)

p (Gamma null) = P(Gamma(N - 5/2) >= Phi_MAP) is the flag; the refit p-value is from replicates and 0.000 means below its resolution. dPhi = Phi_MAP minus the misfit of this noise realization itself (+0.78 sd high for seed 0); c = dPhi / z^2 (z of truth) and c vs level 0 = dPhi / dz^2 with dz the shift of the gap from the level-0 fit of the same noise.

| ellipticity | semi-axes | Phi_MAP/N | Phi_MAP - N | p (Gamma null) | z of Phi | refit p (30 reps, resolution 1/31) | refit z | g MAP | g sd | g error | z of truth | covers | dPhi (excess over noise-only) | c = dPhi / z^2 | c vs level 0 | log Z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2 x 0.2000 | 1.015 | 37 | 0.21 | 0.8 | 0.167 | 1.0 | 0.05023 | 2.71e-04 | +0.00023 | -0.8 | yes | -1.6 | - | - | 35397.8 |
| 0.005 | 0.2 x 0.2010 | 1.023 | 55 | 0.12 | 1.2 | 0.033 | 1.4 | 0.04947 | 2.71e-04 | -0.00053 | 2.0 | no | 16.3 | - | 2.09 | 35365.9 |
| 0.01 | 0.2 x 0.2020 | 1.044 | 106 | 0.014 | 2.2 | 0.000 | 2.8 | 0.04871 | 2.72e-04 | -0.00129 | 4.8 | no | 68.1 | 3.01 | 2.19 | 35299.9 |
| 0.02 | 0.2 x 0.2040 | 1.131 | 314 | 2.7e-10 | 6.5 | 0.000 | 8.2 | 0.04719 | 2.72e-04 | -0.00281 | 10.3 | no | 275.9 | 2.58 | 2.22 | 35063.8 |
| 0.05 | 0.2 x 0.2100 | 1.755 | 1811 | 7.3e-204 | 37.0 | 0.000 | 47.5 | 0.04270 | 2.73e-04 | -0.00730 | 26.7 | no | 1773.1 | 2.48 | 2.33 | 33480.8 |
| 0.15 | 0.2 x 0.2300 | 8.113 | 17070 | 0 | 348.7 | 0.000 | 447.4 | 0.02826 | 2.78e-04 | -0.02174 | 78.1 | no | 17031.8 | 2.80 | 2.74 | 17927.8 |
