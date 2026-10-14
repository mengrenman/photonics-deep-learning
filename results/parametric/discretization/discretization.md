# The offset of the right-family gap interval is a forward-model (grid) bias

## Noise-free F2 fits to T1 data of different grids (inference on n = 64, noise level of the 30 dB data)

| gap | data grid n | g* - g | shift in Laplace sd | Phi_MAP | model-error energy at the truth | extrapolated shift (sd) |
|---|---|---|---|---|---|---|
| 0.05 | 64 | +1.263e-06 | +0.005 | 0.000 | 0.00 |  |
| 0.05 | 128 | +2.775e-04 | +1.024 | 0.112 | 8.85 |  |
| 0.05 | 256 | +3.349e-04 | +1.237 | 0.134 | 13.30 | 1.29 |
| 0.1 | 64 | +5.232e-07 | +0.002 | 0.000 | 0.00 |  |
| 0.1 | 128 | +2.955e-04 | +1.057 | 0.098 | 8.89 |  |
| 0.1 | 256 | +3.627e-04 | +1.298 | 0.115 | 13.47 | 1.37 |

## Paired seeds (same noise key, data n = 128 against data n = 64, g = 0.05, 8 seeds)

z(n = 128) - z(n = 64) = -1.019 +- 0.002 (sd over seeds; the values are -1.019, -1.020, -1.019, -1.018, -1.022, -1.018, -1.021, -1.018); the gap estimate shifts by +2.763e-04 +- 4.3e-07. The noise cancels in the pairing: a constant shift is the signature of a deterministic bias.

## Calibration of the 90 % interval over noise seeds, as run and with the deterministic offset removed (data n = 128)

| gap | seeds | mean z | sd z | coverage | offset removed (sd units, seed-wise) | mean z corrected (se) | sd z corrected | coverage corrected |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 41 | -1.15 | 1.00 | 0.66 | +1.02 | -0.13 (0.16) | 1.00 | 0.88 |
| 0.1 | 41 | -1.19 | 0.98 | 0.68 | +1.06 | -0.13 (0.15) | 0.98 | 0.88 |

Correlation of z between g = 0.05 and g = 0.1 over the 41 common noise keys: 0.985. The two HMC cases of the main experiment (both noise key 0) are therefore ONE data realization; the calibration over seeds carries the statement.


## The offset at other SNRs (g = 0.05; fixed in wavelengths, so in sd it scales as 10^((SNR - 30) / 20))

| SNR (dB) | sd of g (Laplace) | offset (sd) |
|---|---|---|
| 10 | 2.71e-03 | 0.10 |
| 20 | 8.57e-04 | 0.32 |
| 30 | 2.71e-04 | 1.02 |
| 40 | 8.57e-05 | 3.24 |
| 50 | 2.71e-05 | 10.24 |
