# T3 under F2: Gauss-Newton against observed-information (FD) sd of the gap

From `scripts/parametric_t3_fd_sd.py` (one file per unit in this directory). z = (g_true - g_hat) / sd; c = dPhi / z^2 with dPhi as
stored by `parametric_snr.py`; the MAPs are the stored ones (nothing is refit). Key 0 is the milestone-7 realization.

| truth | SNR | seeds | sd FD / GN (mean, range) | key 0: z GN -> FD | key 0: c GN -> FD | mean z GN -> FD | excludes at 3 sd GN -> FD |
|---|---|---|---|---|---|---|---|
| T3g0.05 | 30 dB | 1 | 1.0370 (1.0370 to 1.0370) | +30.67 -> +29.58 | 35.8 -> 38.5 | +30.67 -> +29.58 | 1/1 -> 1/1 |
| T3g0.05 | 20 dB | 40 | 1.0380 (1.0336 to 1.0452) | +9.88 -> +9.53 | 34.2 -> 36.7 | +9.47 -> +9.12 | 40/40 -> 40/40 |
| T3g0.05 | 10 dB | 40 | 1.0359 (0.9074 to 1.0865) | +3.26 -> +3.16 | 30.4 -> 32.4 | +2.85 -> +2.77 | 21/40 -> 18/40 |
| T3g0.1 | 30 dB | 1 | 1.1589 (1.1589 to 1.1589) | +23.64 -> +20.40 | 68.1 -> 91.5 | +23.64 -> +20.40 | 1/1 -> 1/1 |
| T3g0.1 | 20 dB | 40 | 1.0593 (0.8247 to 1.5040) | +7.54 -> +6.61 | 66.4 -> 86.5 | +7.06 -> +6.76 | 40/40 -> 40/40 |
| T3g0.1 | 10 dB | 40 | 1.0182 (0.9017 to 1.2644) | +2.43 -> +2.19 | 62.1 -> 76.6 | +2.09 -> +2.05 | 8/40 -> 6/40 |
