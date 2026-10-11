# Span test of the book's NMM (DESIGN 11.4)

Revised after the independent review. Every number except those of the third reproduction (read from independent_reproduction/), the 90% background fractions and the reviewer's figures in section 8 is read from results/span_test/span_test.json (raw arrays in span_test.npz, figure span_test.png, logs run.log and variants_run.log). The tables are printed by `scripts/span_test_tables.py` from that JSON. The independent third reproduction is stored in results/span_test/independent_reproduction/ (scripts, logs, results_A/B/C.json). All errors are percent of ||m||, i.e. eq. 15.21 with m = eps_r = 1 + chi (background included) on Grid(96, 5.0), N = 9216. "Bound" is the best-in-span error ||Pi_T m - m|| / ||m||. The main run (19 minutes) is deterministic and was not repeated; the review variants were added to its JSON with `python scripts/span_test.py --variants-only` (117 s) without changing any stored number.

## 0. Result

1. **Algebra.** By eq. 15.1 the last operation of the NMM is multiplication by alpha_hat, and eq. 15.6 gives alpha_hat = T G_r^+ with T (N x P) the P training images, so every output o = T (G_r^+ g_r(.)) lies in span(T) for any weights, activation or threshold placed inside g_c and g_r. The best-in-span error therefore lower-bounds the NMM column of Table 15.3 (3.26 to 3.31%), provided the output is not post-processed and T is the 100 training images. The IEM column is not bounded by this test.
2. **Numbers at P = 100.** On held-out Table 15.2 draws (painter's rule, 40 training sets x 1000 draws) the bound has median 33.63%, 5th percentile 25.18%, 0.1th percentile 15.55% (a tail scan of 3 x 20,000 fresh draws gives median 33.59%, 5th percentile 24.92%, 0.1th percentile 15.54%, 2 of 60,000 draw-training set pairs at or below 10% (both from the same draw), none at or below 3.3%). For Fig. 15.9-like scenes the median is 9.10% (Test #1-like; digitized scene 8.57%) and 10.71% (Test #4-like; digitized scene 10.45%).
3. **The minimum is not quoted.** The pooled minimum (9.86%) is the easiest value of one held-out draw (#13: mean over the 40 training sets 12.11%, next easiest 16.41, 19.28, 19.68%); without it the minimum is 15.55%. The independent implementation B has minimum 14.17% on its own draws. The robust statements are the median and the 5th percentile. The 0.1th percentile of the held-out pool (15.55%) is itself set by draw #13: its 40 values are the 40 smallest of the 40,000, so it equals the minimum without that draw by construction (without draw #13 it is 17.85%; a bootstrap over draws gives 11.9 to 19.5%). The 0.1th percentile is quoted from the tail scan (15.54%; its 60 smallest values come from 26 distinct draws).
4. **Test #4 is the anchor.** No tested reading of eqs. 15.1 to 15.6 reaches Test #4's 3.28% at P = 100. For the digitized Test #4 scene the best reading (training eps_r in [1, 2], an oracle shift along Pi_T 1, clipping to the colorbar) gives median 6.09% (minimum 5.45%); the smallest value over the eight Test #4-like scenes and 40 training sets is 3.71%, for a scene smaller than the figure's (T4-d, disks of radius 0.3). Test #1's 3.26% is reachable to within 0.15 to 0.67 pp (digitized medians 3.42 to 3.93%) under a constant term, the colorbar eps_r law or clipping, so Test #1 alone would be a fragile basis. For the digitized scenes, under every reading the bound for Test #4 exceeds that of Test #1 by 1.89 to 3.39 pp (pooled over the eight named scenes 1.61 to 3.18 pp; for the 200 jittered scenes of each type the order reverses under the span reading, Test #4-like 10.48% against Test #1-like 11.14%), while the book's NMM numbers for Tests #1 and #4 differ by 0.020 pp (spread over the four tests 0.049 pp).
5. **Defensible statement.** Table 15.3's NMM column cannot be reproduced from eqs. 15.1 to 15.6 as printed with P = 100 images from Table 15.2; the cause is unknown. This is not a statement that the book's numbers are wrong.

## 1. Why the NMM output lies in span(T) (eqs. 15.1 to 15.6; printed pp. 438 to 440, read from the PDF raster)

- Eq. 15.1: o_j = alpha g_r(w_r | beta g_c(w_c x_j + b) | + p). The last operation is multiplication by alpha (N x S); no additive term follows.
- Eq. 15.6: alpha_hat = T G_r^+, T = [t_1 ... t_P] (N x P). So alpha_hat v = T (G_r^+ v) for every v, and every output lies in range(alpha_hat), a subset of span(T). This uses nothing about the rank of G_r, the activation or the inputs. The biases b and p enter inside g_c and g_r and cannot add a component outside range(alpha_hat); the complex first layer enters only through the modulus.
- Eq. 15.4 as printed reads "beta = T"; it omits G_c (dimensionally it must be beta G_c = T, matching eq. 15.5). With S >= P and G_r of rank P, alpha_hat G_r = T exactly, so the training images are reproduced and the reported 3.3% is a test error. P = 100 gives 2P + 1 = 201 hidden nodes, matching "beta and alpha are 9216 x 201".
- Verified numerically in tests/test_span_test.py (toy ELM built from the equations, 5 seeds plus rank-limited S < P, L < P and large thresholds): outputs lie in span(T) to 1e-10 relative, also 3 times outside the training scale, and the training images are interpolated to 1e-7; adding an output bias of 10% of ||o|| breaks the identity (negative control).

## 2. Results at P = 100

### 2.1 (a) held-out Table 15.2 draws (rules.<rule>.sweep[100].test.pooled, heldout_min_diagnosis, tail_scan)

| case | n | median | 5th pct | 0.1th pct | min | min without the easiest draw | fraction <= 3.3% |
|---|---|---|---|---|---|---|---|
| held-out, painter | 40,000 | 33.63 | 25.18 | 15.55 | 9.86 | 15.55 | 0 |
| tail scan, painter | 60,000 | 33.59 | 24.92 | 15.54 | 9.23 | - | 0 |
| held-out, max-overlap | 40,000 | 32.71 | 24.48 | 19.67 | 14.69 | 14.69 | 0 |
| tail scan, max-overlap | 60,000 | 32.64 | 24.21 | 16.37 | 10.60 | - | 0 |

Where the minimum comes from (heldout_min_diagnosis; "its mean over sets" is the draw's bound averaged over the 40 training sets):

| implementation / rule | pooled min | easiest draw | its mean over sets | next three | min without it | median | 5th pct | 0.1th pct |
|---|---|---|---|---|---|---|---|---|
| A, painter | 9.86 | #13 | 12.11 | 16.41, 19.28, 19.68 | 15.55 | 33.63 | 25.18 | 15.55 |
| B (independent), painter | 14.17 | #802 | 17.09 | 17.66, 17.97, 18.84 | 15.20 | 33.80 | 24.84 | 16.96 |
| A, max | 14.69 | #180 | 19.90 | 20.42, 21.42, 21.72 | 14.69 | 32.71 | 24.48 | 19.67 |
| B, max | 14.99 | #665 | 17.62 | 17.77, 18.63, 19.61 | 15.49 | 32.79 | 24.04 | 17.39 |

The DESIGN section 10 sentence now reads "median 34%, 0.1th percentile 15.5% at P = 100". Other references at P = 100 (rules.painter.sweep[100]): the nearest training image has median 62.87%; the empty scene m = 1 has bound 8.35% (6.62 to 10.31% over the training sets).

### 2.2 Why the bound is large: the empty scene alone costs 8%

A span of 100 images with eps_r in [1, 8] cannot reproduce a flat eps_r = 1 background, and the digitized Test #1 and Test #4 scenes are 91.5% and 94.0% background, and 12 of the 16 named scenes exceed 90% (range 81.6 to 97.5%). Empty-scene bound against P (painter, mean over training sets): P = 25: 18.30; 50: 12.66; 100: 8.35; 200: 5.21; 400: 3.03; 800: 1.73; 1600: 0.92; 3200: 0.43%.

### 2.3 Bound versus P (rules.painter.sweep)

| P | training sets | held-out median | 5th pct | min | nearest image median | Test #1-like median | Test #4-like median |
|---|---|---|---|---|---|---|---|
| 25 | 40 | 48.38 | 36.74 | 16.95 | 68.46 | 18.57 | 20.00 |
| 50 | 40 | 40.90 | 30.83 | 12.95 | 65.45 | 13.15 | 14.88 |
| 100 | 40 | 33.63 | 25.18 | 9.86 | 62.87 | 9.10 | 10.71 |
| 200 | 20 | 27.10 | 20.19 | 8.22 | 60.49 | 6.16 | 7.74 |
| 400 | 20 | 21.32 | 15.80 | 6.43 | 58.31 | 4.03 | 5.44 |
| 800 | 10 | 16.28 | 11.99 | 5.45 | 56.36 | 2.76 | 3.88 |
| 1600 | 3 | 11.80 | 8.60 | 4.19 | 54.52 | 1.82 | 2.71 |
| 3200 | 3 | 7.72 | 5.46 | 2.92 | 52.72 | 1.14 | 1.80 |

Log-interpolating the scene medians, they cross 3.3% near P = 600 (Test #1-like) and P = 1100 (Test #4-like); the claim is specific to P = 100. The held-out median is still 7.72% at P = 3200 (35% of N).

### 2.4 (b) Fig. 15.9-like scenes (rules.<rule>.sweep[100].scenes)

| case | rule | n | median | 5th pct | min | fraction <= 3.3% |
|---|---|---|---|---|---|---|
| Test #1-like (8 named x 40 sets) | painter | 320 | 9.10 | 8.09 | 6.94 | 0 |
| Test #1-like | max | 320 | 9.36 | 8.19 | 6.92 | 0 |
| Test #4-like (8 named x 40 sets) | painter | 320 | 10.71 | 9.28 | 8.25 | 0 |
| Test #4-like | max | 320 | 10.81 | 9.33 | 8.34 | 0 |
| Test #1-like jittered (200 x 40) | painter | 8000 | 11.14 | 8.48 | 6.77 | 0 |
| Test #4-like jittered (200 x 40) | painter | 8000 | 10.48 | 8.74 | 6.97 | 0 |
| Test #4 eps_r 2, 4, 6 (outside the colorbar) | painter | 40 | 30.36 | 27.06 | 26.29 | 0 |

The geometries of Fig. 15.9 are not given; the "digitized" scenes were read from the ground-truth column (PDF page 482 at 400 dpi: circle and box fits to the colored regions, eps_r by matching parula colors to the colorbar 1 to 2; about +-0.05 m and +-0.1 in eps_r). Test #1 digitized: disk (0.05, 1.00) r 0.64 eps_r 1.3, rectangle (0.05, -1.55) 1.56 x 0.44 eps_r 1.2. Test #4 digitized: three disks near the origin, r 0.42 to 0.45, eps_r 1.4, 1.6, 1.8. Tests #2 and #3 (cross, bracket) are not in the Table 15.2 family and were not evaluated. Per named scene (painter): bound median [min, max] over the training sets, span + constant, nearest image, vacuum guess:

| scene | bound | span + constant | nearest image | vacuum guess |
|---|---|---|---|---|
| T1-a digitized | 8.57 [7.36, 10.68] | 3.71 | 54.75 | 7.30 |
| T1-b small, stronger | 9.32 [8.18, 11.32] | 5.09 | 55.18 | 9.05 |
| T1-c large disk | 8.83 [7.57, 10.60] | 4.77 | 53.68 | 10.65 |
| T1-d off-center | 8.68 [6.94, 10.69] | 3.55 | 56.41 | 6.73 |
| T1-e eps_r up to 2 | 12.74 [11.28, 14.15] | 10.73 | 54.92 | 20.69 |
| T1-f large, central | 10.52 [9.73, 11.85] | 7.94 | 51.24 | 17.22 |
| T1-g side by side | 8.74 [7.29, 10.42] | 3.43 | 56.23 | 5.61 |
| T1-h very small | 9.03 [7.48, 10.85] | 4.28 | 56.41 | 6.39 |
| T4-a digitized | 10.45 [9.14, 12.29] | 7.10 | 52.80 | 14.91 |
| T4-b reversed order | 10.50 [9.09, 12.35] | 6.98 | 53.16 | 13.54 |
| T4-c large triangle | 11.56 [10.03, 13.43] | 9.27 | 51.71 | 22.83 |
| T4-d small tight | 9.45 [8.25, 12.29] | 5.07 | 55.81 | 8.75 |
| T4-e displaced | 10.41 [9.20, 12.05] | 7.13 | 52.71 | 15.26 |
| T4-f eps_r = 2 | 12.87 [11.13, 14.36] | 10.53 | 51.92 | 23.85 |
| T4-g chain | 10.93 [9.67, 13.06] | 7.95 | 52.52 | 17.50 |
| T4-h large | 10.56 [9.23, 12.65] | 8.30 | 50.48 | 22.95 |

Some scenes have a vacuum-guess error below the bound (T1-a: 7.30% against 8.57%): the guess eps_r = 1 is not in span(T), which is the background effect of section 2.2.

## 3. Anomaly triage: other readings of the NMM (span_test.json variants)

Four readings, each at P = 100 with 40 training sets (the same seeds for every eps_r law, so the shapes are identical and only the permittivities differ), painter's rule: **span** (eqs. 15.1 to 15.6 as printed); **span + constant** (span(T u {1}): an output bias that eq. 15.1 does not have; it is the same for every eps_r law because span{1 + s chi_j, 1} = span{chi_j, 1}); **shift + clip** (the in-span output shifted along Pi_T 1 by an oracle coefficient chosen against the truth, then clipped to the colorbar [1, 2]: a post-processing step that leaves the span, at its most favorable); **constant + clip** (span(T u {1}) clipped to [1, 2]). Training eps_r laws: U[1, 8] (Table 15.2), U[1, 3], U[1, 2] (the colorbar of Fig. 15.9). Digitized scene, median [min] over the 40 training sets:

| Test #1 digitized | U[1, 8] | U[1, 3] | U[1, 2] |
|---|---|---|---|
| span | 8.57 [7.36] | 4.58 [4.23] | 3.93 [3.59] |
| span + constant | 3.71 [3.36] | 3.71 [3.36] | 3.71 [3.36] |
| shift + clip | 4.82 [4.31] | 3.66 [3.32] | 3.42 [3.09] |
| constant + clip | 3.43 [3.12] | 3.43 [3.12] | 3.43 [3.12] |

| Test #4 digitized | U[1, 8] | U[1, 3] | U[1, 2] |
|---|---|---|---|
| span | 10.45 [9.14] | 7.62 [6.68] | 7.21 [6.35] |
| span + constant | 7.10 [6.30] | 7.10 [6.30] | 7.10 [6.30] |
| shift + clip | 7.06 [6.09] | 6.22 [5.50] | 6.09 [5.45] |
| constant + clip | 6.47 [5.80] | 6.47 [5.80] | 6.47 [5.80] |

Over the eight named scenes (median [min]): Test #1-like span + constant 4.48 [3.19], best (U[1, 2], shift + clip) 4.02 [2.93]; Test #4-like span + constant 7.66 [4.40], best (U[1, 2], shift + clip) 6.73 [3.71]. The smallest value anywhere is 2.93% for Test #1 (T1-g, side by side) and 3.71% for Test #4 (T4-d, small tight). The eps_r law also moves the empty-scene floor (8.35, 3.17, 1.68% for U[1, 8], U[1, 3], U[1, 2]). It also moves the in-family held-out bound (span reading, painter, P = 100, 40 sets x 1000 draws; span_test.json variants): under U[1, 3] median 18.36%, 0.1th percentile 5.19%, minimum 3.75%, none at or below 3.3%; under U[1, 2] median 10.91%, 0.1th percentile 2.67%, minimum 2.00%, and 80 of 40,000 values (0.20%) at or below 3.3%. So "no held-out draw reaches 3.3%" holds for the book's stated U[1, 8] law, not for every training law; the Test #4 statement below does not depend on it. It also moves the in-family held-out bound (median 33.63, 18.36, 10.91%; 0.1th percentile 15.55, 5.19, 2.67%): for U[1, 2], 80 of 40,000 held-out values (0.2%, minimum 2.00%) are at or below 3.3%, so the 'none at or below 3.3%' of sections 0 and 2.1 holds for the Table 15.2 law only.

**Ruled out by the evidence**

- A squared misfit: with the book's printed BA and VBIM numbers (Table 15.3) a squared relative error would put Test #1 at 21.96% (BA) and 14.21% (VBIM) and Test #4 at 36.15% and 20.08%, all worse than the vacuum guess (7.30% and 14.91%), contradicting "can reconstruct the scatterers well"; as printed (first power) the numbers sit at 0.66, 0.28, 0.88, 0.27 of the vacuum guess (variants.metric_reading).
- Chi normalization: the span error relative to ||chi|| is 117% (Test #1) and 70% (Test #4) at U[1, 8], 54% and 48% at U[1, 2]; it only makes the error larger.
- P much larger than 100: contradicts "beta and alpha are 9216 x 201"; the scene medians would need P near 600 to 1100 (section 2.3).
- Overlap rule, rasterization, ring rule, rotation, containment, painter order: both implementations and the independent third reproduction (independent_reproduction/log_B.txt, log_C.txt) move the P = 100 held-out mean only between 32.76 and 36.49% (36.49% is the main script's binary-mask run, painter) and the digitized-scene bounds by up to 30% relative in the third reproduction (Test #1-like 11.91% against 9.17% for the containment-inside plus rotation variant, log_C.txt; Test #4-like up to 10%: 12.08% against 10.97%), while the rasterizer variants alone move them by at most about 8% (binary masks: +7.6% and +7.4%); the binary-mask rasterization gives held-out median 36.07%, Test #1-like 9.79%, Test #4-like 11.50% (binary_masks); the smallest single held-out value over the third reproduction's generator variants (log_B.txt) is 13.57%, against 9.86% (tail scan 9.23%) for the main run's baseline and 14.17% for implementation B.
- An output bias or contrast-trained NMM, a training eps_r law of [1, 2] and clipping to the colorbar, each alone or combined, **for Test #4**: the digitized scene stays at 6.09% or more.

**Not ruled out (open)**

- Test-like scenes inside the book's training set (the Test #1-like and Test #4-like scenes are 50 to 56% from their nearest training image (named scenes, painter: 50.48 to 56.41%; random held-out draws at P = 100: median 62.87%), so a training set containing the test scenes would be a different T).
- A different T (a different generator, a larger P or a different training set) or a non-literal pipeline: the identity alpha_hat G_r = T means the IEM, trained on "the 100 sets of outputs from the NMM", would see exactly the training images, which suggests the NMM column is not the literal output of eqs. 15.1 to 15.6 on held-out scenes.
- A stronger, unreported post-processing. The NMM column is constant to 0.049 pp across tests whose bounds differ by 1.89 to 3.39 pp (digitized Test #4 minus Test #1, every reading), which points to a test-independent process.

## 4. Independent implementations (cross_check, independent_reproduction/)

- Implementation B (in span_test.py, shares no code with the package or with A): its own Philox-stream sampler of Table 15.2 (rejection sampling for r4 > r3), sub-pixel painting on a 6x finer grid (8x for the named scenes) composed on eps_r then block-averaged, its own scene renderer, SVD projection; 40 training sets and 1000 held-out draws. Held-out median of the per-draw bound averaged over the 40 training sets A 33.72 / B 33.84% (painter; the pooled 40,000-value medians of section 2.1 are 33.63 / 33.80%), mean 34.02 / 33.97%; Welch z of the mean 0.19 (painter), 0.43 (max); bootstrap z of the median -0.40, -0.78; max |z| over 14 comparisons 1.93; agree = true.
- Bridges: the same inputs through QR (A), gelsd and SVD (B) agree to 3.6e-15 and 1.9e-15; 32 scene images rendered by A and B differ in the bound by at most 0.033 pp; the same shapes through both rasterizers differ (A - B mean) by +0.076 pp (+0.22%) on held-out images and +0.032 pp on scenes (painter), +0.115 and +0.029 pp (max).
- A third independent implementation written by the reviewer's reproduction track (own generator, own digitization of the scenes; its files are copied to results/span_test/independent_reproduction/): P = 100 in-family held-out mean 34.18%, median 33.75%, minimum 16.23% over 20,000 draw-training set pairs, none below 3.31% (log_A.txt); Test #1-like nominal 9.17%, Test #4-like nominal 10.97% (log_C.txt); span + constant Test #1-like nominal 5.12% (instantiation minimum 3.11%); at P = 500 the Test #1-like nominal bound is 3.83% (6.8% of pairs below 3.31%).
- Disclosure: the first run of that reproduction imported numpy before setting the thread variables and ran at about 1110% CPU (about 11 BLAS threads) for about 2 minutes (log_A.txt shows it); its numbers are unaffected, and its later logged runs stayed at 134% and 135% CPU (log_B.txt, log_C.txt).

## 5. Method and checks

- Training targets: `phantoms.ch15_austria_variant(rng, grid, overlap, eps_range)` on Grid(96, 5.0), eps_r = 1 + chi; training set r uses seed [0, 1, r], sets for different P are nested prefixes, painter and max rules share the shapes; held-out set seed [0, 2], tail scan [0, 4].
- Projection: Householder QR with rank check (SVD fallback) and a reorthogonalization pass; the normal equations are never formed. cond(T) 54 to 109 at P = 100, full rank everywhere.
- Headline numbers recomputed from the raw arrays with plain numpy match the JSON; a from-scratch np.linalg.lstsq on training set 0 and 60 held-out images matches the stored QR errors to 2.6e-13 pp; count columns recomputed from raw: (40, 1000), (3, 20000), 320 = 8 x 40, 8000 = 200 x 40. tests/test_span_test.py recomputes n, medians, minima and fractions from the npz.
- Variants: `clip_shift_error` (oracle shift on 81 + 51 grid points), `variant_errors`, `metric_reading_checks`, `heldout_min_diagnosis`, all tested (including that the clipped error is never above the span error for scenes inside [1, 2], that the returned shift reproduces the returned error, that the variants at U[1, 8] equal the main run's arrays exactly, and that `--variants-only` leaves the stored results untouched).

## 6. Limits

- The geometries of Tests #1 and #4 are digitized from the PDF raster (about +-0.05 m, +-0.1 in eps_r) and varied (8 named, 200 jittered per type); Tests #2 and #3 were not evaluated.
- The overlap rule of the book's generator is unknown; both rules are reported and differ by less than 1 pp at the median.
- The shift + clip reading uses an oracle shift chosen against the truth, a favorable bound for that class of post-processing, not a model of a real one. The variants use the painter's rule only and 40 training sets.
- The pooled minimum depends on the number of draws; P = 1600 and 3200 use 3 training sets and P = 800 uses 10.
- Eq. 15.4 is read as beta G_c = T (the printed form omits G_c). Eqs. 15.1 to 15.6 were read from the PDF raster because the text extraction garbles bars and daggers.
- The IEM column (1.03 to 1.09%) is not bounded by this test.

## 7. Deliverables

`scripts/span_test.py` (CLI; json, npz, png; `--quick`, `--figure-only`, `--variants-only`, `--no-cross-check`; at most 2 BLAS threads, set before numpy is imported), `scripts/span_test_tables.py`, `tests/test_span_test.py` (35 tests, about 5 s with test_geometry_phantoms.py (51 tests), none slow), `src/bayes_scatter/phantoms.py` (`paint_max`, `overlap`, `eps_range`, defaults unchanged), DESIGN sections 2.6, 9, 10, 11.4 and README updated.

## 8. Review log

| finding | resolution |
|---|---|
| B1: the minimum 9.86 vs 16.23 comes from one outlier draw; DESIGN section 10 and the report rest on it | Fixed. Diagnosed from the stored arrays (draw #13, set-mean 12.11%, next easiest 16.41%; min without it 15.55%); the report quotes the median 33.63%, 5th percentile 25.18% and 0.1th percentile 15.55% (tail scan 15.54%); DESIGN section 10 rewritten to "median 34%, 0.1th percentile 15.5% at P = 100". |
| B2: anomaly triage misses the eps_r range and clipping; conclusion should rest on Test #4 | Fixed. eps_r laws U[1, 8], U[1, 3], U[1, 2] and the shift + clip, constant + clip, chi-normalized and squared-misfit checks added; conclusion stated on Test #4 (digitized best 6.09% against 3.28%; Test #1 within 0.15 to 0.67 pp); ruled-out and open explanations listed. My Test #1 numbers differ slightly from the reviewer's (3.93 vs 3.87% at U[1, 2]; 40 vs 8 training sets; the reviewer's figures are quoted from the review and are not stored in results/); Test #4 span 7.21% agrees. |
| B3 (process): the reproduction exceeded the thread cap | Disclosed in section 4; the repo's span_test.py already capped threads before numpy, and the shared scripts now import `_threadcap` first (static test). |

## Numbers audit log

An independent numbers-vs-outputs audit traced 61 quantitative claims in this report to the raw outputs and recomputed the headline statistics from the stored arrays: 43 matched, 4 differed only in rounding, 11 were wrong or overstated and 3 had no stored source. All 16 text corrections it proposed are applied above; none changes a conclusion.
