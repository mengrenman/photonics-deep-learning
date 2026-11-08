# Milestone 8: the geometry of detectability (DESIGN 15)

Every number cites a file under `results/detectability/` (abbreviated `D/`) unless another path is given. The contract is DESIGN section 15 (15.1 to 15.5). This report went through an independent premise review (section 6) and a numbers audit against the stored files; the audit's 20 corrections are applied.

Conventions. Whitened real-stacked data: y_w = sqrt(2)/s [Re d; Im d], Phi = (1/2)||r_w||^2, Fisher = D^T D. z here is (fitted - base)/sigma; the milestone-7 files use z_true = (g_true - g_hat)/sd, the opposite sign, so only |z| is compared. "One draw" is the milestone-7 noise draw (30 dB, key 0).

## 0. Summary

1. The first-order law c = Delta Phi / z^2 = (1/2) tan^2 alpha / cos^2 beta holds. At the symmetric nominal base it reproduces the noise-free nonlinear c_NL to 1 to 3.3 % for the ellipticity ladder at 2 % to 5 %, to 1 % for the grid errors, and the one-draw measurements once their noise terms are modeled (agreement 1e-4 relative for the ladder, 3e-4 for T5, 4e-3 for T3; `D/verify.md` section 3).
2. It fails, in a controlled way, where the model manifold is curved: T2 at 15 % (parameter-effects curvature) and T3 (second-order phenomenon: z is O(Delta^2) at both base points, c_T3 about 91 and 38, curvature-driven; the first-order formula cannot predict it).
3. The D2 selection rule is exact at first order: non-A1 parts of z are below 6e-15 of z at every symmetric base. It needs a D2-symmetric base point.
4. Detectability spectrum (corrected): measured against the deformed object's own gap, all 42 A1 boundary deformations (m = 2 to 8, eps_r 1.1, 2, 4, g 0.05, 0.1) are UNFLAGGED when the gap error reaches 3 sd, at amplitudes 2.3e-4 to 9.6e-4 wavelengths (c_true 2.2e-8 to 4.5, `D/spectrum.md`, n = 192). The confounded deformations are the high-m gap-preserving ones, beyond the far-field band limit 2 k0 r = 2.51. The first draft's conclusion ("low harmonics are the most damaging, many flagged first") was an artifact of measuring against the base gap and is withdrawn.
5. T3's posterior sd is the observed-information one: z 23.64 -> 20.40 and 30.67 -> 29.58, c 68.1 -> 91.5 and 35.8 -> 38.5. `results/PARAMETRIC_REPORT.md` and DESIGN 14.6 are corrected (2026-11-08) for every T3 unit of the SNR experiment (`results/parametric/snr_fd/t3_fd_summary.md`).

## 1. Derivation and its conditions (DESIGN 15.1)

Let delta (whitened, 2N) be the exact difference between the true noise-free data and the model data at the base point p0, D (2N x k) the whitened Jacobian at p0, T = range(D), P_perp = I - D(D^T D)^-1 D^T.

- Least squares to first order: Delta p = (D^T D)^-1 D^T delta; Delta Phi = (1/2)||P_perp delta||^2.
- Gap direction: Delta g = v^T delta with v = D(D^T D)^-1 e_g, ||v||^2 = [(D^T D)^-1]_gg = sigma_g^2. So z = Delta g / sigma_g = w_hat^T delta, w_hat = v/||v|| the unit vector of T orthogonal to the other parameters' tangents ("pure gap" direction).
- With alpha the angle of delta to T and beta the angle of its tangential part to w_hat: ||P_perp delta|| = ||delta|| sin alpha, w_hat^T delta = ||delta|| cos alpha cos beta, hence **c = (1/2) tan^2 alpha / cos^2 beta**, independent of ||delta|| for a fixed direction of delta and of the SNR. Confounded directions: alpha -> 0, beta -> 0. If z = O(||delta||^2) (no A1 component) then Delta Phi = O(||delta||^2) too and c scales as 1/||delta||^2: a second-order quantity.
- Checks of the algebra: c_direct = Delta Phi/z^2 equals c_geom to 1e-10 in a linear-Gaussian toy; D^T D equals ShapePosterior.fisher transformed by dp/du to 1e-8 (`tests/test_detectability.py`).

Conditions (each one was tested):

- **Which gap.** Delta g is (fit - base). It is the error of the estimate only if the base gap is the true gap. A deformation that changes the object's own edge-to-edge gap by gap_change gives z_true = z_fit - gap_change/sigma_g; equivalently use the gap-preserving tangent tau + (gap_change/2) tau_0 (tau_0 = the m = 0 tangent, d_r - 2 d_g, which lies in T, so Delta Phi is unchanged). For A1 boundary modes gap_change = -2(-1)^m eps (`detectability.gap_change_per_amplitude`).
- **Exactness.** delta is exact, so the truth's nonlinearity is irrelevant. The formula is exact iff the model map mu is affine between p0 and the fit p*. The corrections are intrinsic curvature times ||P_perp delta|| and parameter-effects curvature times ||p1||; both are SNR independent, so c is SNR independent also nonlinearly for noise-free fits with a negligible prior. Second order: p = p1 + p2, p2 = (D^T D)^-1(-(1/2) D^T H[p1,p1] + H[p1,.]^T P_perp delta), H = d2 mu (`second_order_bias`, `curvature_tensor`: central differences of the exact jvp Jacobian, asymmetry indicator <= 4.6e-3 in `D/verify.md` section 5).
- **Base point.** First-order z and Delta Phi are invariant under moves of p0 along nuisance directions (w_hat and P_perp annihilate them), so base-point dependence is a second-order diagnostic. The analysis base is the profile foot point (g = g_true, nuisances at their least-squares values, `detectability.foot_point`; `D/verify.md` section 6): there D^T delta has only a gap component, the tangential part is exactly along w_hat (beta = 0 or 180 deg, all 17 cases) and c = (1/2) tan^2 alpha. "D at the noise-free MAP with delta from p0" has no first-order meaning (at the exact MAP the tangential residual is zero); it is not used (the values remain in the case JSONs under `at_map`, with a note).
- **Curvature of the residual.** At a fit with residual r, Hess(Phi) = D^T D (I - A), A = (D^T D)^-1 <r, d2 mu>. The Gauss-Newton (Fisher) sd is wrong by 1/sqrt(1 - lambda) along the eigenvectors; use the observed-information sd when Phi/N >> 1. The Laplace log evidence changes by -(1/2) sum log(1 - lambda).
- **Noise.** DESIGN's Delta Phi is the expected excess over a correct model; milestone 7 measured Phi_MAP - Phi(noise), lower by (1/2)||P_T e||^2 (mean k/2 = 2.5, observed 0.98 to 1.17). One draw has z-noise sd 1 and Delta Phi noise sd sqrt(2 Delta Phi). Interaction terms: z = z_NL + w*^T e, Delta Phi = Phi_NL + r*^T e - (1/2)||P_T* e||^2. The reference for the law is c_NL (noise-free nonlinear fit), not one draw.

## 2. Selection rule (DESIGN 15.2)

The rings are invariant under sigma_x and sigma_y. `D2Action` builds the exact antenna permutations by matching coordinates (worst match error 6.0e-15 for the book rings; it raises for symmetry-breaking rings); the forward map is equivariant on a random non-symmetric contrast to 1e-11 relative (tests). D2 irreps are one dimensional and real, an intertwiner preserves isotypic components, and the components are orthogonal, so w_hat lies in A1 (fraction >= 1 - 1e-12 at every symmetric base, `D/verify.json` d2.w_hat_a1_fraction) and **only the A1 part of delta carries first-order z**. Conditions: a D2-invariant base (equal disks, c_x = c_y = 0), invariant antenna sets, iid noise, exact grid flips. A2 at m = 1 (pair rotation) is outside F2.

Decomposition of delta at the nominal base (fraction of ||delta||^2; `D/verify.md` section 2):

| case | base | A1 | B1 | B2 | A2 | sum of irrep z / z |
|---|---|---|---|---|---|---|
| all E (ladder, 10 cases), grid (4) | nominal | 1.0000 | 0 | 0 | 0 | 1.000000000000 |
| T3 g 0.1 | nominal (eps_r 2) | 0.6851 | 0.3149 | 0 | 0 | 1.000000000000 |
| T3 g 0.1 | mean (eps_r 2.2) | 0.0168 | 0.9832 | 0 | 0 | 1.000000000000 |
| T3 g 0.05 | nominal / mean | 0.7034 / 0.0167 | 0.2966 / 0.9833 | 0 | 0 | 1.000000000000 |
| T5 | nominal | 0.6562 | 0 | 0.3438 | 0 | 1.000000000000 |

- Non-A1 z / total z: at most 5.5e-15 (absolute 1.2e-13) over all 17 cases and variants at symmetric bases. T5 has A1 + B2 only, as predicted by the third disk lying on the sigma_x axis.
- T3 is second order at BOTH bases (the review's correction). At the nominal base 95.7 % of the A1 energy lies along the eps_r tangent (cos 0.978, coefficient 0.2034 against the first-order 0.2; g 0.05: 0.978, 0.2018): the A1 part is mostly in-model and z comes entirely from the 84.4-norm remainder (w^T remainder = -27.360 = total z). At the mean base A1 holds 1.7 % of ||delta||^2 and z is still 21.6.
- Where the base is NOT symmetric the rule fails, as it must: at the T3 g 0.05 foot point (c_x = 0.0042) 23 % of z is non-A1 (w_hat A1 fraction 0.99855), at the T5 foot point (c_y = 0.0050) 2 % (0.99923); all other foot points except T3 g 0.1 (c_x = -2.6e-5; non-A1 z 0.0155, 7e-4 of z, w_hat A1 fraction 0.99999998) keep < 1e-17 relative (`D/verify.json`).

## 3. Verification of the law (DESIGN 15.3 items 1 and 2)

Setting: book rings (40 tx, 60 rx), side 1.5, inference grid n = 64, F2, p0 = (g, 0.2, 2, 0, 0), tol 1e-11; s and d_clean are the cached milestone-7 values (results/parametric/{data,ellipticity/data}); the n = 256 grid-error data are in `D/data/`. Case files: `D/verify/<case>.json`; tables: `D/verify.md`.

| case | abs z lin | abs z NL (sd full) | abs z one draw (sd full) | Delta Phi lin | Delta Phi NL | Delta Phi one draw | alpha | beta | c_geom (nominal) | c_geom (foot) | c_NL (sd full) | c one draw (GN) | c one draw (full) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0.5% g0.05 | 1.779 | 1.768 | 1.961 | 16.07 | 16.09 | 16.27 | 22.7 | 97.5 | 5.079 | 5.119 | 5.15 | 4.232 | 4.229 |
| E1% g0.05 | 4.587 | 4.563 | 4.757 | 65.64 | 65.8 | 68.1 | 25.4 | 101.0 | 3.12 | 3.148 | 3.16 | 3.013 | 3.009 |
| E2% g0.05 | 10.22 | 10.15 | 10.35 | 267.7 | 268.8 | 275.9 | 27.0 | 103.0 | 2.564 | 2.598 | 2.608 | 2.584 | 2.578 |
| E5% g0.05 | 27.07 | 26.74 | 26.93 | 1733 | 1747 | 1773 | 28.4 | 104.4 | 2.364 | 2.411 | 2.444 | 2.48 | 2.445 |
| E15% g0.05 | 80.96 | 78.99 | 79.20 | 1.693e4 | 1.689e4 | 1.703e4 | 30.9 | 105.3 | 2.584 | 2.572 | 2.706 | 2.795 | 2.715 |
| E0.5% g0.1 | 1.980 | 1.973 | 1.988 | 17.03 | 17.03 | 17.85 | 23.1 | 98.3 | 4.343 | 4.361 | 4.377 | 4.519 | 4.514 |
| E1% g0.1 | 5.026 | 5.006 | 5.022 | 68.83 | 68.83 | 72.33 | 25.7 | 101.9 | 2.725 | 2.735 | 2.746 | 2.873 | 2.868 |
| E2% g0.1 | 11.14 | 11.07 | 11.09 | 279.6 | 279.5 | 288.9 | 27.2 | 104.0 | 2.255 | 2.262 | 2.28 | 2.359 | 2.349 |
| E5% g0.1 | 29.46 | 29.27 | 29.29 | 1814 | 1808 | 1840 | 28.6 | 105.5 | 2.09 | 2.089 | 2.111 | 2.22 | 2.145 |
| E15% g0.1 (T2) | 88.79 | 85.48 | 85.53 | 1.803e4 | 1.75e4 | 1.766e4 | 31.3 | 106.5 | 2.287 | 2.165 | 2.395 | 2.553 | 2.414 |
| T3 g 0.05 | 29.99 | 29.29 | 29.58 | 3.597e4 | 3.382e4 | 3.368e4 | 34.0 | 94.3 | 40.0 | 44.41 | 39.42 | 35.8 | 38.5 |
| T3 g 0.1 | 27.36 | 20.05 | 20.40 | 4.015e4 | 3.82e4 | 3.806e4 | 35.2 | 93.9 | 53.64 | 82.08 | 95.05 | 68.12 | 91.48 |
| T5 | 104.4 | 108.5 | 108.6 | 1.014e4 | 9714 | 9701 | 40.8 | 129.3 | 0.9312 | 0.8905 | 0.8254 | 0.8018 | 0.8229 |
| grid 128, g 0.05 | 1.019 | 1.024 | 1.024 | 0.1121 | 0.1121 | 0.1121 | 6.5 | 75.9 | 0.108 | 0.1078 | 0.1069 | 0.1069 | 0.1069 |
| grid 256, g 0.05 | 1.231 | 1.237 | 1.237 | 0.1343 | 0.1343 | 0.1343 | 5.8 | 76.1 | 0.0886 | 0.0884 | 0.0878 | 0.0878 | 0.0878 |
| grid 128, g 0.1 | 1.055 | 1.057 | 1.057 | 0.0978 | 0.0979 | 0.0979 | 6.0 | 75.4 | 0.0879 | 0.0879 | 0.0876 | 0.0876 | 0.0876 |
| grid 256, g 0.1 | 1.295 | 1.298 | 1.298 | 0.1148 | 0.115 | 0.115 | 5.3 | 75.5 | 0.0685 | 0.0684 | 0.0683 | 0.0683 | 0.0683 |

(Columns: alpha, beta in degrees at the nominal base; "one draw" is the milestone-7 measurement, `results/parametric/{snr,ellipticity,discretization}`, with the sd either Gauss-Newton (GN, as published) or observed information (full). The grid-error rows are noise-free fits: "one draw" is the same computation as c_NL.)

Reading the table:

- **The law is confirmed where the manifold is flat.** Grid errors: c_geom 0.108 / 0.0886 / 0.0879 / 0.0685 against c_NL 0.1069 / 0.0878 / 0.0876 / 0.0683 (0.3 to 1 %). Ladder 0.5 % to 5 %: c_geom within 0.8 to 3.3 % of c_NL; the foot-point c_geom is closer still for the small levels.
- **One draw is not a measurement of c.** With e the whitened noise of key 0 (the same unit vector in every case), `D/verify.md` section 3 predicts the ladder one-draw numbers to 1e-4 (relative; T5 z to 3e-4, T3 z to 4e-3): E1% g0.1 z -5.01712 against -5.01722 and Delta Phi 72.3313 against 72.3323; E0.5% g0.05 z -1.96102 against -1.96090 and 16.271 against 16.272; T5 -110.027 against -109.998 and 9701.29 against 9701.30. The terms: w*^T e = -0.194 (g 0.05), -0.0165 to -0.060 (g 0.1, E1% to E15%), (1/2)||P_T e||^2 = 0.98 to 1.17, r*^T e = +4.62 (E1% g0.1), +1.35 (E0.5% g0.05), -11.4 (T5). E0.5% g0.05: c_NL is 5.15 and one draw 4.23 (-18 %, entirely w*^T e); a single draw has about 100 % uncertainty in c at |z| about 2. The earlier statement that the g = 0.05 rows "agree in the same way" as g = 0.1 is withdrawn.
- **Where the formula fails.**
  - T2 at 15 % (E15% g0.1): c_geom 2.287 against c_NL(sd full) 2.395 and (sd GN at MAP) 2.534. The cause is the model's parameter-effects curvature, not a change of sigma_g: sigma_g(p0) = 3.078e-4 against 2.888e-4 at the MAP (6.6 %, not 10 to 15 %). Delta g linear -0.02733, second order -0.02502 (tangential +0.00359, normal -0.00128), NL -0.02399; the sigma and Delta g errors partly cancel (`D/verify.md` section 5). c_geom/c_NL = (Delta Phi ratio 1.030)(Delta g ratio squared 0.770)(sigma ratio squared 1.137) = 0.902.
  - T3 (below).
  - T5: z linear 104.4 against NL 108.5 (4 %), c_geom 0.931 against c_NL 0.825 (the second-order Delta g -0.03094 against NL -0.03086 repairs it).

### T3 in detail (unequal contrasts)

- **sd.** Gauss-Newton against observed information at the milestone-7 MAP with the noisy data (`D/verify.md` section 4): sd_g 2.799e-4 -> 3.243e-4 (g 0.1, ratio 1.159), 2.690e-4 -> 2.789e-4 (g 0.05, 1.037); corr(g, c_x) -0.008 against -0.305 at g 0.1. z 23.64 -> 20.40 and 30.67 -> 29.58; c one draw 68.12 -> 91.48 and 35.8 -> 38.5. Other cases: T5 +1.3 % in sd, T2 g 0.1 -2.8 %, ladder 0.0 to -2.8 %.
- **Why.** The B1 residual is huge (Phi about 3.8e4 for N = 2400). The curvature operator A = (D^T D)^-1 <r, d2 mu> at the noise-free MAP has top eigenvalue 0.443 (g 0.1) and 0.194 (g 0.05), in the c_x-g block (the second fundamental form contracted with the B1 residual maps B1 x B1 to A1). Predicted log Z(FD) - log Z(GN) = 0.270 (g 0.1), against 0.252 from `results/parametric/T3g0.1_F2/is_evidence.json` (laplace_fd - laplace_gn); importance sampling agrees with the FD Laplace to 0.029. Eigenvalue maxima elsewhere: E1% 0.005, E15% 0.038 to 0.059 (minimum -0.12), T5 0.064 (all `D/verify.md` section 5).
- **c_T3 is a second-order quantity.** Delta g (g 0.1) linear / second order / NL: nominal base -0.00880 / -0.00547 / -0.00657; mean base -0.00603 / -0.00608 / -0.00657; foot point -0.00602 / -0.00607 / -0.00657. The first-order series from the nominal base overshoots by +34 % (the curvature operator at p0 has an eigenvalue -1.04 in magnitude, so the series does not converge; lambda_max of the first-order residual 0.111); c_geom at the nominal base (40, 54) agrees with the measured (35.8, 68) only because that +34 % overshoot in Delta g combines with a +15 % overshoot in sigma_g(p0) (3.215e-4 against 2.798e-4). At g 0.05 the foot point has c_x = 0.0042: the B1 error is absorbed by c_x and its curvature coupling to g is the second-order mechanism (foot: second order -0.00806 against NL -0.00817). Since z = O(Delta^2) and Delta Phi = O(Delta^2), c_T3 scales as 1/Delta^2 in the contrast imbalance; it is not fixed by the kind of error. The selection rule (z second order for non-A1 errors) holds at both bases.

## 4. Detectability spectrum (DESIGN 15.3 item 3)

Six base objects: eps_r in {1.1, 2, 4} x g in {0.05, 0.1}, r = 0.2, F2, tol 1e-12, noise s of each object's own 30 dB data on that grid. A1 modes cos(m phi) on the right disk and (-1)^m cos(m phi) on the left, m = 2 to 8. Files: `D/spectrum/n{64,128,192}/eps<e>_g<g>.json` (n = 64 with refits), `D/spectrum.json`, `D/spectrum.md`, `D/spectrum.png`, `D/spectrum_convergence.png`. Main grid n = 192 (one base takes 326 to 618 s on one workstation core; six in parallel). For eps_r = 2 the n = 192 s is 1.0002 times the cached milestone-7 s (0.9988 at n = 64) (s_ratio_to_m7 in the n = 192 JSONs).

Checks (`D/spectrum.md`, last table, all six bases, n = 192): m = 0 and A1 m = 1: sin alpha <= 3.1e-13; z_true/z_fit = 0 to 1e-13 (m = 0: exactly 0); dz_fit/deps x sigma_g/(-+2) = 1.00000000; B1 and B2 m = 1 sin alpha <= 8.5e-13; max non-A1 |z|/||tau|| over m = 2 to 8 <= 1.7e-13; max irrep leakage <= 2.7e-12; max equivariance error (P_g tau(c) = tau(M_g c)) <= 6.7e-12; relative residual of the base solves <= 5.3e-14.

**The two gap conventions.** dz_fit/deps = w_hat^T tau is how far the F2 estimate moves from the BASE gap. The error of the estimate is dz_true/deps = dz_fit/deps - gap_change/sigma_g with gap_change = -2(-1)^m. Reductio for the first draft: m = 0 (a pure radius change, fitted exactly by F2) has z_fit = -2 eps/sigma_g and c = 0, i.e. it would have counted as the most dangerous deformation. In the true-gap convention it has z_true = 0.

**Spectrum at n = 192** (gap error in sd at 30 dB; eps_3 = amplitude in 1e-4 wavelengths at which |z_true| = 3; unflagged means Delta Phi(eps_3) = 9 c < 147 nats, c < 16.3):

| eps_r | g | sigma_g (1e-4) | c m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 | eps_3 m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | 0.05 | 2.973 | 0.706 | 0.0635 | 0.00479 | 4.04e-4 | 2.14e-5 | 6.72e-7 | 3.21e-7 | 5.46 | 4.15 | 4.26 | 4.40 | 4.46 | 4.45 | 4.46 |
| 2 | 0.05 | 2.701 | 2.16 | 0.269 | 0.018 | 1.22e-3 | 1.05e-4 | 1.18e-5 | 2.89e-6 | 5.40 | 4.62 | 4.09 | 4.04 | 4.05 | 4.05 | 4.06 |
| 4 | 0.05 | 2.851 | 0.883 | 4.47 | 0.207 | 0.0148 | 1.92e-3 | 3.93e-4 | 9.79e-5 | 2.34 | 9.63 | 5.31 | 4.35 | 4.26 | 4.26 | 4.27 |
| 1.1 | 0.1 | 2.936 | 1.06 | 0.0751 | 0.00661 | 3.12e-4 | 2.26e-5 | 4.93e-7 | 2.23e-8 | 6.25 | 4.53 | 4.29 | 4.36 | 4.40 | 4.40 | 4.41 |
| 2 | 0.1 | 2.785 | 1.89 | 0.279 | 0.0219 | 9.33e-4 | 7.06e-5 | 4.87e-6 | 5.73e-7 | 4.97 | 4.94 | 4.27 | 4.17 | 4.17 | 4.18 | 4.18 |
| 4 | 0.1 | 3.292 | 1.03 | 2.57 | 0.298 | 0.0126 | 1.09e-3 | 1.20e-4 | 2.05e-5 | 2.44 | 7.75 | 6.50 | 5.13 | 4.95 | 4.94 | 4.94 |

- All 42 cases are unflagged at z_true = 3: c_true <= 4.47 (maximum eps_r 4, g 0.05, m 3) against the flag-equivalent 16.3; eps_3,true = 2.34e-4 to 9.63e-4 wavelengths (shortest: eps_r 4, g 0.05, m 2). For m >= 4 eps_3 -> 1.5 sigma_g (about 4.1e-4 to 4.9e-4): the data do not see the high-m part of the deformation, the F2 estimate stays at the base gap while the object's own gap changes by 2 eps, and the error is 2 eps/sigma_g with almost no misfit. The flag amplitude eps_flag (where Delta Phi = 147) is larger than eps_3 in every case (by a factor 1.9 at eps_r 4, g 0.05, m 3, up to 2.7e4 at high m).
- **Why high m is confounded (band limit).** ||tau_m||/||tau_2|| follows J_m(2 k0 r)/J_2(2 k0 r) with 2 k0 r = 2.51: at n = 192, eps_r 2, g 0.1, m = 3 to 8: 0.383, 0.124, 0.0274, 0.00751, 0.00199, 6.7e-4 against J ratios 0.489, 0.168, 0.0446, 0.00971, 0.0018, 2.9e-4. A gap-preserving deformation of harmonic m >= 3 is seen by the data only through its area (m = 0) part, which F2 assigns to r; the high-m part that restores the gap is beyond the band limit. m = 2 is the mode most visible to the data (c_true 0.71 to 2.16). The most confounded deformations are the high-m gap-preserving ones, with c_true falling by factors of typically 8 to 25 per harmonic from m = 3 on (extremes 2 and 46). (The reviewer's suggestion that eps_r = 4 at r = 0.2 sits near the lowest TM resonance, sqrt(eps_r) k0 r = 2.51 against j_01 = 2.405, as the cause of the large dz_true at m = 2 and of the maximum at m = 3 for eps_r 4 was not tested.)
- **T2 is a gap-preserving m = 2 deformation.** To first order the T2 ellipse (r, (1 + e) r) is the m = 0 mode at +e r/2 plus the m = 2 mode at -e r/2. c_true(m = 2) at eps_r 2, g 0.1 is 1.861 (n = 64), 1.892 (n = 192), against c_geom 1.871 for the independent ellipse renderer (E0.5% g0.1 pure, `D/verify.md`); the same at g 0.05: 2.122 (n = 64) against 2.132. A test pins this (`test_ellipticity_is_the_gap_preserving_m2_mode_and_c_true_reproduces_the_t2_linear_c`).
- **Small-deformation criterion.** The slope eps_3 m/r is <= 0.020 for all 42 cases (so the deformations at eps_3 are small); x = eps_3 (eps_r - 1)/h (contrast displacement per boundary pixel) is <= 0.12 at n = 64 and <= 0.37 at n = 192 (larger because h is smaller; the nonlinear refit was run at n = 64).

**Grid convergence** (`D/spectrum.md` "Grid convergence", n = 64 / 128 / 192, eps_r 2, g 0.1):

| m | dz_fit/deps (response of the estimate) | dz_true/deps | c_true | ||tau_m||/||tau_2|| | J_m/J_2 |
|---|---|---|---|---|---|
| 2 | -1090 / -1147 / -1141 | 6058 / 6026 / 6040 | 1.86 / 1.90 / 1.89 | 1 | 1 |
| 3 | 1056 / 1096 / 1111 | -6092 / -6077 / -6070 | 0.275 / 0.278 / 0.279 | 0.383 / 0.383 / 0.383 | 0.489 |
| 4 | -181 / -158 / -151 | 6966 / 7015 / 7031 | 0.0234 / 0.0223 / 0.0219 | 0.129 / 0.125 / 0.124 | 0.168 |
| 5 | -87 / -28 / -18 | -7235 / -7201 / -7200 | 9.1e-4 / 9.4e-4 / 9.3e-4 | 0.0298 / 0.0278 / 0.0274 | 0.0446 |
| 6 | -6.5 / -1.2 / +4.8 | 7141 / 7172 / 7186 | 9.2e-5 / 6.3e-5 / 7.1e-5 | 0.0124 / 0.00762 / 0.00751 | 0.00971 |
| 7 | -50.8 / -16.2 / +1.6 | -7199 / -7189 / -7180 | 1.9e-5 / 5.4e-6 / 4.9e-6 | 0.00862 / 0.00275 / 0.00199 | 0.0018 |
| 8 | -27 / -12 / -2.6 | 7121 / 7161 / 7179 | 7.3e-5 / 4.1e-6 / 5.7e-7 | 0.00926 / 0.00205 / 0.00067 | 0.00029 |

- dz_true is converged: over all 42 cases |dz_true(128)/dz_true(192) - 1| <= 0.46 % and |dz_true(64)/dz_true(192) - 1| <= 4.6 % (m = 2). The response of the estimate dz_fit (and c_fit) is NOT converged for m >= 5, even at n = 192 (sign flips, factors 5 to 300; c_fit(m = 7) = 0.375 / 1.06 / 93): it is a pixelization floor of about 1 % of ||tau_2|| and a near-zero crossing of z_fit, so the m = 6 "near-orthogonal" mode and the oscillation of c_m and eps_3 with m in the first draft were grid artifacts. c_true of m >= 7 changes by up to an order of magnitude with n (eps_r 2, g 0.1: 4.9e-6 at n = 192) and is 1.2e-3 at most over all bases and grids (3.9e-4 at n = 192), far from the flag. The tangent norms at m = 8 are not yet converged at n = 192 (6.7e-4 against the continuum 2.9e-4).

**Nonlinear refit (n = 64, `D/spectrum.md`, refit table).** The data F(chi0 + eps_3 dchi_gp) at the gap-preserving amplitude with z_true = +3 are refit with the real forward model. All 42 refits converged: shift 2.889 to 3.035 sd (the largest deviation, eps_r 4, g 0.05, m 3, has x = 0.12), misfit within 3 % of the linear 9c wherever 9c > 0.03; the remaining deviations (ratios 0.88 to 1.71) occur only at 9c < 0.03 nats. This tests the nonlinearity of the forward map in chi (not the second-order rendering of the shape). On the first-version amplitudes the 15 refit failures all had x >= 0.412 and the 20 passes x <= 0.314, so x, not the size of a "small projection", decides it (first-draft refits, not stored; the split was found independently by the reviewer and confirmed by the fix pass).

**What this says about detectability.** For boundary deformations of the two disks the gap error reaches 3 sd at amplitudes of 1e-4 to 1e-3 wavelengths (about 1 to 4 % of the pixel size at n = 64) while the misfit stays at c z^2 <= 40.2 nats, far below the 147-nat flag: the misfit never warns first. The confounded deformations are the high harmonics of a gap-preserving boundary change, which lie beyond the far-field band limit; low harmonics (m = 2, close to the ellipticity of T2) are the most visible to the data.

## 5. Settings and provenance

- Everything uses float64. Seeds: noise key 0 for every one-draw number (cached in results/parametric); the noise-free fits are deterministic. Fit conventions: MAP with the milestone-7 prior (starts: p0 and, when it exists, the milestone-7 MAP), keeping the lower Phi.
- Compute: verify (17 cases, 80 to 150 s each) and the n = 64 spectrum (with refits) ran on the Mac with 4 single-thread processes; the n = 128 and n = 192 spectra (6 single-core processes each, JAX on CPU, separate batches) ran on tr from the scratch clone `~/work/sweeps/m8_review` (rsync of the Mac working tree; logs in `D/logs/`, thread settings in each JSON `thread_env`). No GPU, no MPS daemon was started; no processes of mine remain on either machine.
- Code: `src/bayes_scatter/detectability.py`, `scripts/detectability_verify.py`, `scripts/detectability_spectrum.py` (`--n`), `shapes.circle_mode_integrals`. Fast suite after the review: `pytest -m "not slow"` 731 passed, 9 deselected, 0 failed in 445 s on the Mac working tree that includes the adjoint warm-start edits (forward.py, inference.py, likelihood.py, shapepost.py, thetaspace.py, tests/test_warm_adjoint.py). Interaction with the warm-start changes: none found (the detectability code uses the default forward path and reproduces the first draft's fit-convention numbers bit for bit; `laplace_fit(hessian="fd")` from the modified shapepost.py was used unchanged).
- 17 new tests since the first draft (21 in `tests/test_detectability.py`): gap change per irrep, true-gap tangent of in-model modes, T2 as the gap-preserving m = 2 mode with c_true agreement to 2 %, second-order bias against the exact nonlinear fit of a quadratic model (error of order delta^3), the residual-curvature operator and the evidence gap against exact determinants, the curvature tensor against second differences, and the foot point.

## 6. Review log

1. Critical (spectrum measured against the base gap): confirmed by reductio (m = 0) and by the T2 identity; fixed (section 4, DESIGN 15.1/15.3/15.5); first-draft conclusions withdrawn. The resonance explanation for eps_r = 4 is untested.
2. Major (n = 64 not converged for m >= 5): confirmed; spectrum recomputed at n = 128 and 192, convergence table and figure added, m = 6/7 "extremes" removed.
3. Major (T3 uses the Gauss-Newton sd): confirmed (c 68.1 -> 91.5, 35.8 -> 38.5). Reported here and in DESIGN 15.5. `results/PARAMETRIC_REPORT.md` (sections 0, 3, 4.2, 4.4) and DESIGN 14.6 were corrected on 2026-11-08, at the stored MAPs of all 162 T3 units (`scripts/parametric_t3_fd_sd.py`).
4. Major (T3 second order at both bases): confirmed (cos 0.978, coefficient 0.2034, z = -27.360 entirely from the remainder; second-order Delta g -0.00547 / foot -0.00607 against NL -0.00657). Open issue 4 rewritten; foot-point base implemented.
5. Minor (one draw, dPhi definition, "g = 0.05 rows agree"): confirmed; noise decomposition added; statement withdrawn.
6. Minor (refit failures): confirmed on the first-draft data (x split at 0.31 / 0.41); criterion replaced.
7. Minor (15 % case): confirmed (sigma +6.6 %, parameter-effects curvature dominates).
8. Minor (DESIGN conditions): all four points implemented in DESIGN 15.1/15.2.
9. Disputed: nothing. One refinement of the review's wording: "harmonics m >= 3 lie beyond the far-field band limit" is quantitative only as ||tau_m||/||tau_2|| = J_m/J_2 within about 25 % (m = 3: 0.383 against 0.489); the confounding strengthens gradually with m rather than switching on at m = 3.

## 7. Open items

- Not done: a nonlinear refit at n = 192 (it would test only the forward map's nonlinearity, already covered at n = 64); a polar rendering of the deformed boundary at second order in eps; the profile foot point at the posterior level (it is a noise-free, first-order construct); the resonance explanation for eps_r = 4 (section 4).
- Scratch: `~/work/sweeps/m8_review` on tr holds a copy of the working tree, the logs and the n = 128 / 192 spectrum outputs (the outputs are also in `D/spectrum/`).
