# Homogenization: plan and status

This file is the build contract for the notebooks in this folder. It records the design (made on 2026-11-23 by a design workflow: three independent plans, a judge who prototyped the key computations, and the orchestrator's amendments below), what each notebook must compute and check, and the claims the notebooks may and may not make. `README.md` will hold the findings once the notebooks exist.

## State

- 2026-11-23: built. The seven modules pass their tests (`_build/selftest_modules.py`), and the six notebooks execute top to bottom with every check passing; `README.md` holds the findings, the run times and the check counts. Branch `claude/homogenization-exploration-9feabb`; nothing committed.
- This file is the design record: the plan as designed, the corrections made before the build (next sections), and the expectations written before each computation. Where a notebook's result differs from what is written below, the notebook is right.
- After the build, each notebook was audited by a reviewer who recomputed its headline numbers with separate code, and a senior reviewer tested whether the conclusions survive other readings. No certified number was wrong. The fixes made after that review, now in the notebooks: notebook 02 gained the exact quarter-square solution (the Mortola-Steffe formula sqrt((1+3h)/(3+h)), conjectured by Mortola and Steffe in 1985 and proved by Craster and Obnosov and by Milton in 2001), which charges the whole corner band [1/4, 3/4]; the Q1 pair satisfies L_h U_h = (1+3h)/(3+h) on that cell to rounding (observed, not derived); notebook 06 states both readings of the detectability window (against the volume average and against eps_eff(k0) = <eps> + Rytov) and shows that the odd boundary term of the ring fits is not universal; in notebook 05, item 05-a's end correction of 0.20a is the value for the uncorrected length L = Na, 0.076a the value for L = Na + s, and the end correction proper is 0.0595a; notebook 04 locates the weak-contrast window by root finding; notebook 03 adds a smooth superellipse control that matches the optimized design; notebook 01 corrects a mislabeled octave and restates the staircase and kappa = 100 readings.

## Orchestrator's decisions and amendments

1. Notebook 05 (band edges) is kept: it is the representation-theory notebook, and the reader works in representation theory. It is built in parallel with the others.
2. Notebook 02, sections 2 and 3, gain one exact check. The checkerboard measure dmu = (1/pi) sqrt((1-z)/z) dz has moments m_k = C_k / (2 * 4^k), with C_k the Catalan numbers (1/2, 1/8, 1/16, 5/128, ...), because 1 - sqrt(1 - 1/s) is the Catalan generating function. Its orthogonal polynomials are the Chebyshev polynomials of the fourth kind mapped to [0,1], so the Jacobi (Lanczos) coefficients are alpha_0 = 1/4, alpha_k = 1/2 for k >= 1, and off-diagonal beta_k = 1/4 for k >= 1. The Lanczos run on the sampled checkerboard operator (N = 64, 128, 256) prints its first 20 coefficients and moments against these, and asserts alpha_0 = 1/4 and m_1 = 1/8 to 1e-12 (exact for N divisible by 4) and the convergence trend of the next ones with N. The Gauss quadrature of the exact weight is the reference for the discrete measure's CDF.
3. Every notebook ends with a caveats section and its own reference list, and the prose follows the printed outcome of every check marked "to test", including outcomes that contradict this plan.
4. The design prototypes live in the session scratch directory, `<design scratch>` below; they are not part of the repository. Re-validate rather than trust them.
5. Module tests: each module `X.py` has its own standalone test script `_build/test_X.py` (prints PASS/FAIL lines, exits nonzero on failure, under 60 s with 4 threads). `_build/selftest_modules.py` runs them all (or the named ones: `python _build/selftest_modules.py cellsolver bounds`). This replaces the single file with `test_<module>()` functions described under Modules.

## Corrections from the premise review and the module builds (2026-11-23). These override everything below.

An independent premise review (Opus) of this plan and the builders of the seven modules checked every claim the notebooks rest on. The core of the plan stands: the Rayleigh matrix and its spectral measure, the Q1 primal/dual bracket and the window ordering, the discrete Keller and Dykhne identities, the four-tile corner exponents (lambda_1 = (4/pi) arctan(kappa^-1/2) for kappa = b/a > 1; the second root is 2 - lambda_1), the Q1 rate 2 lambda_1, the Q1 partial fractions and the percolation atom, amendment 2 (Catalan moments, Chebyshev fourth-kind Jacobi coefficients), the order of limits at negative contrast, the trace bounds and laminates, the Hom dimensions (3, 0, 1), the TM second-order law, and the termination as a radius shift. The items below replace what the sections further down say. Where a section below conflicts with an item here, this list wins.

### General

- G1. Before any per-atom statistic (counting atoms, naming a dominant atom, cluster weights), merge eigenvalues of a discrete measure that agree within 1e-9: C4 cells give exactly degenerate E pairs, and eigh splits their weight arbitrarily.
- G2. Discrete Q1 and FG measures are Rayleigh-Ritz compressions: their atoms do not converge individually even when sigma* is excellent (the FG disk at K = 16 smears the 97.7% Rayleigh atom over about 36 atoms). Describe them by W1 distance, moments, cluster weights in windows, or normalized variance, not by single-atom positions.
- G3. Literature: use the verified list in the section "Literature (verified)" at the end of this file. Cite nothing as checked that is not marked checked there.
- G4. Module APIs differ from the function list below in places (all additions or deliberate): read the module docstrings. Notable: cellsolver.fg_bracket returns scalar bounds for one diagonal entry (assert info['converged']; the bounds stay rigorous if CG stops early, because each bound is the exact energy of the actual iterate); bloch.point_group_ops(lat, k0=None, rods=None) -> (ops, name) and rep_matrices(lat, C, G, k0, ops, B); scatter.fit_disk uses analytic tangents (jac='analytic'); bounds.bm_support_iso has method='dual' (0.3 s for 720 directions, more accurate than the LP), keep the LP on a subset as the cross-check.

### Notebook 01

- 01-a (section 3). Corner check: evaluate the smallest singular value of D at the closed-form lambda_1 (expect <= 1e-14) and report the numerically minimized root separately with tolerance 1e-10. The second root is 2 - lambda_1 = (4/pi) arctan(kappa^1/2).
- 01-b (section 3). kappa = 100 is pre-asymptotic: besides the local rates, print the compensated error e_N N^(2 lambda_1) (it must level off if 2 lambda_1 is the rate) and a three-parameter fit rate(N) = r_inf + c N^-g of the five local rates with its spread. Keep the check rate(256->512) - 2 lambda_1 >= 0.03 but do not let the prose claim more than the fit supports.
- 01-c (section 3). Sampled (Moulinec-Suquet) rates: add N = 1024 and 2048 (FFT PCG, seconds) and say whether the local rate settles; claim no law.
- 01-d (section 4). Staircase disks: report the certified gap (Rayleigh(f_pix) - U_h)/Rayleigh (or the bracket's distance from Rayleigh(f_pix)) at N = 64, 128, 256. The other reading is an O(h) perimeter effect that vanishes with N; the prose must say which reading the numbers support.
- 01-e (section 5). The antithetic identity det U[sigma] det L[sigma_swap] = (ab)^2 is algebraically the interchange identity and holds for every two-valued field: label it a unit test of the code path, not a finding.
- 01-f (section 5). CLT scaling: report the bootstrap sd of the fitted slope and test |slope + 1| <= 3 sd (instead of the window [-1.3, -0.7]).
- 01-g (section 5). Ordering of apparent tensors: per realization, U_aff >= U_per by inclusion of primal trial spaces (periodic fluctuations contain those vanishing on the boundary), and L_per >= L_flux by inclusion of dual trial spaces (periodic divergence-free fields contain those with J.n = e.n on the boundary). Give this two-line proof. Huet (1990) proves the ensemble ordering of the kinematic and static apparent tensors; the periodic middle is not in Huet (Kanit et al. 2003 use it).

### Notebook 02

- 02-a. Use f = pi 0.3^2 = 0.282743 everywhere, with these references: dominant atom z = 0.3552495, weight fraction 0.977158, MG atom 0.3586285; next atoms 0.505807 (weight 4.389e-3) and 0.497544 (2.013e-3); Drude resonance 0.596028 omega_p (MG 0.598856). The values 0.355273, 0.97717, 0.358650, 0.59605, 0.59888 in the sections below are for f = 0.2827 and must not be used.
- 02-b (section 2). Dilute check at f = 0.01: z_dom - (1-f)/2 = -(3 S4^2/(2 pi^4)) f^3 = -1.5291e-7 to 1e-10, and 1 - w_dom/f = (3 S4^2/pi^4) f^2 = 3.058e-5 to 1% relative (octupole mixing; the O(f^4) statement holds for sigma_eff, not for the measure).
- 02-c (section 2). Truncation check: f <= 0.5: L = 20 vs 40, atoms with weight > 1e-8, tolerance 1e-9 (f = 0.5 is at 5.8e-11). f = 0.7: L = 60 vs 100, atoms with weight > 1e-6, tolerance 1e-7.
- 02-d (sections 1, 3). The FG atom at 0 of a smooth inclusion is not O(1/K): it decays super-algebraically (5.9e-5, 3.3e-7, 1.7e-9, 8.8e-12 at K = 4, 6, 8, 10; round-off from K = 12). Check it at K in {4, 6, 8, 10} with a log-linear fit and print the decay rate; report the weight in (0, 0.05] separately. Delete "spurious atom of order 1/K" wherever it appears (findings, thesis).
- 02-e (sections 3, 4). Delete the check "FG disk 99%-atom count <= 5" and the monotone check "dominant-atom error and W1 decreasing over K = 8, 12, 16, 20" (measured W1 1.84e-2, 1.14e-2, 8.7e-3, 1.10e-2). Use W1(K = 20) < W1(K = 8), or print only.
- 02-f (section 4). Square versus disk: run the identical statistic on the equal-area disk as a control. Atom counts and the weight outside [0.23, 0.77] behave the same for both shapes, so they cannot test Perfekt-Putinar. Use a distributional statistic on merged atoms: the normalized variance of the measure (the premise review found square 0.0229 -> 0.0177, disk 0.0109 -> 0.0035 over K = 6 to 24) and the quantiles q10/q90 of the CDF (square q10 -> 0.25-0.26). If the statistics do not separate the shapes, state that FG at K <= 24 cannot test the band. The formula itself is confirmed in the literature (Perfekt-Putinar Theorem 7; Bonnetier-Zhang: sigma_ess = [-(1/2)|1 - alpha/pi|, (1/2)|1 - alpha/pi|] in the standard normalization), so the band for 90 degrees is z in [1/4, 3/4].
- 02-g (section 5). Drude sum rule: integrate from omega = 0 (substitute omega = u^2, or adaptive quad on [0, 3]) and add the exact tail on [3, inf) by quad; the specified grid fails 1e-3 for the checkerboard (Im eps* ~ omega^-1/2 at small omega). The lossless absorption check must go through the density's Stieltjes transform (quad with the algebraic endpoint weight) against 1 - sqrt(1 - 1/s), not through the closed form sqrt(eps1) (that would be vacuous).
- 02-h (section 8). The testbed's 30 dB is a power SNR, so the amplitude threshold is 10^-1.5 sigma_1: counts 3, 12, 38 (also print 1e-3 sigma_1, which is 60 dB amplitude: 6, 22, 75). State that h is geometrically spaced. Make the comparison like-for-like in data count: give design (a) 120 real contrasts, or designs (b) and (c) 30 frequencies (then each has 120 real numbers).
- 02-i (section 7). Extremal measures of the isotropic LP are a single atom at (1-f)/2 (MG), {0, 1}, {0, z}, or {z, 1}; accept all four. The lens has a second corner at the {0, 1} measure, which is not a microstructure; the phase-1-host HS point lies on the {0, z} arc at z = (1 + f2)/2; the m0/m1 region is valid but not the sharp 2D isotropic region (Milton 1981 also uses the measure of the phase-swapped medium). The MG-corner check must also require that MG attains the support in more than one direction (count them), with HiGHS tolerances 1e-10, or use method='dual'.
- 02-j (section 1). Sampled-scheme first moment: m1 = f(1-f)/2 - (1/2) sum |chi1_hat(k)|^2 over the k != 0 whose Nyquist-zeroed wavevector vanishes ((N/2, 0), (0, N/2), (N/2, N/2)), for componentwise zeroing and C4 cells. For the checkerboard with 4 | N, m0, m1 and m2 are exact (m2 = 1/16, b_1 = 1/4); m3 has relative errors 4.9e-5, 4.8e-6, 4.5e-7 at N = 64, 128, 256.
- 02-k (sections 2, 3; amendment 2). The sampled checkerboard measure carries a few large atoms near z = 0.05 to 0.08 (largest 0.13 of 0.5 at N = 128), so the sup distance of the CDF on [0.05, 1] is about 0.11 and nearly N-independent. Describe it by the W1 distance to the Gauss rule of the exact measure (0.0069, 0.0064, 0.0060 at N = 64, 128, 256), the moments, and the Jacobi coefficients (alpha_1..4 - 1/2 and beta_2..5 - 1/4 shrink with N; see the module test).
- 02-l (run time). Use bm_support_iso(method='dual') for the sweep and the LP on a 60-direction subset; one reorthogonalization pass in Lanczos is enough; keep Q1 at N <= 40.

### Notebook 03

- 03-a (section 1). G-closure sources: the trace bounds are Lurie-Cherkaev (1984, 1986) and Tartar ("Estimations fines des coefficients homogeneises", 1985); Grabovsky (1993) states them for any dimension as Tr(A(C* - A)^-1) <= (1/theta_B) Tr(A(B - A)^-1) + theta_A/theta_B and Tr(B(B - C*)^-1) <= (1/theta_A) Tr(B(B - A)^-1) - theta_B/theta_A. Show that the form used in bounds.trace_slacks is equivalent (one line), and keep the laminate-equality and HS checks.
- 03-b (section 5). The weight of the anisotropic objective is w = ((s2 - l1)/(s2 - l2))^2 = 0.7314 at m* = 0.25, lambda = (3.24397, 2.10031), not 0.68 (with 0.68 the maximum moves to m = 0.192).
- 03-c (sections 4, 5). Replace the criterion "certified gap strictly decreasing in N means P1": under P2 (a true gap g_inf > 0) the certified gap also falls, because the Q1 bracket narrows (weak-disk bracket widths are 3.8%, 1.9%, 1.0% of the HS range at N = 32, 64, 128). Report the certified interval of the gap [(HS_up - U_h)/range, (HS_up - L_h)/range] at N = 32, 64, 128 next to the weak-disk control at the same N, and conclude only "single-scale designs come within x of HS_up", with x the certified upper end at N = 128.
- 03-d (section 5). Design convergence: warm-start each N from the upsampled N/2 design and report the Hamming change caused by refinement alone, with the certified objective interval.
- 03-e. Literature for attainment: L. P. Liu, Proc. R. Soc. A 466 (2010) 3693-3713 (a periodic microstructure attains an HS bound if and only if it is a periodic E-inclusion; includes conductive composites); Liu, James and Leo (2007) construct periodic E-inclusions (in 2D they coincide with Vigdergauz's shapes) but do not state HS attainment; Vigdergauz (1994) and Grabovsky-Kohn (1995) are elasticity.

### Notebook 04

- 04-a (section 1). Torsion series: use p, q <= 2001 (or the single-series form) and set the convergence tolerance from the tail (about 1e-8 for u; the difference between 301 and 601 terms is 3.3e-7, and grad u0 is off by 6e-4 at edge nodes with 301 terms).
- 04-b (section 6). Compute I_flux, I_grad and their limits with the same quadrature: 2x2 Gauss points, exact for Q1 (twoscale.weak_integrals(..., exact_energy=True) with the exact sigma*_h = 1.688269 and M = 1.504546). Mixing midpoint gradients with exact limits leaves an eps-independent 0.8% floor (M_mid 1.492440, Sigma_mid 1.674409) that makes the slope checks meaningless.
- 04-c (sections 2-4). State the conditions of the cut dependence: the discrete cell is mirror symmetric about the cut node lines (m even, centered staircase), the cell is centrosymmetric (no O(eps) dispersive term), the same cell phase sits at opposite edges (integer N); torsion corners add at most log factors. If the budget allows, add N = 64 with m = 8 as a second sequence to show the rates are not pre-asymptotic.

### Notebook 05

- 05-a (section 7). Rewrite the theorem. For any wall phase x_w, with N whole cells between two walls at the same cell phase, u = psi_k(x) psi_-k(x_w) - psi_-k(x) psi_k(x_w) gives u(x_w + Na) = 2i sin(kNa) psi_k(x_w) psi_-k(x_w), and psi_k(x_w) != 0 in band (the Wronskian of psi_k and psi_-k is a nonzero constant). So every in-band mode sits exactly at k = n pi/(Na) for ANY wall phase, and the envelope error is O(N^-4) with no end correction. Mirror planes decide only which edge standing wave is a j = 0 mode. Replace the a/4 experiment with three cases: both walls at 0; both at a/4 (same exact quantization, no j = 0 mode); walls at different cell phases (0 and Na + a/4): the Dirichlet-like edge has error slope -3 without a correction, and a fitted delta = 0.20a cuts the error 15x (N = 10) to 24x (N = 40); the other edge then has mixed ends and needs half-integer j. Restrict the fit to the upper gap edge and skip in-gap (surface) roots when counting modes from the edge.
- 05-b (section 6). Massive-Dirac loop phase: the leading-order formula holds to 0.005 pi at r = m/v and 0.0195 pi at 2m/v, but 0.034 pi at 3m/v (trigonal warping and O(r^2) terms). Use r in {m/(2v), m/v, 2m/v} with 0.03 pi, show the deviation divided by r^2 is roughly constant over m/(2v) to 3m/v, and show that the second-order k.p loop phase (kp_berry_phase, order=2) follows the PWE more closely. Average +q and -q when comparing the Dirac velocity with a PWE slope (trigonal warping is odd in q).
- 05-c (section 6). The quantized Berry phase pi of the equal-rod lattice is exact by construction (inversion center at the origin makes eps_hat real and the pencil real symmetric): label it a test of the code, and make the unequal-rod deviation-versus-r^2 test the physics check.
- 05-d (section 3). k.p tensor versus finite differences: use |q| a = 0.01 and 0.02 with Richardson (agreement 4.3e-8); at 1e-3 and 2e-3 round-off (machine epsilon times lambda_max / h^2) limits it to about 8e-6.
- 05-e (section 4). The [3, 5] ratio is for the relative error of omega^2 - omega0^2 (the absolute error scales like q^4).
- 05-f (section 6). Fefferman-Weinstein (2012) is a theorem for -Delta + V; the TM problem is the generalized problem -Delta u = (omega/c)^2 eps u. Say the symmetry argument (dim Hom_C3v(V, End E) = 1) is what transfers, and that the theorem is an analogy here.

### Notebook 06

- 06-a (section 1). The plan's 1.2874 and 17.309 are sums truncated at about 600 plane waves (1.28744, 17.30895); converged values are about 1.2876 and 17.310. Quote each sum with its truncation.
- 06-b (section 2). Born knee: the side-2.5 domain cuts the Gaussian window at 2.5 sigma (4% of its peak), and the leakage makes gain(3)/gain(1.5) = 0.0177. Use a domain of side >= 3.5 (gain(3)/gain(1.5) = 0.0035; n = 64 is converged to 0.5%), print the gains at all kappa_g, and compare with the leakage predicted from the window's spectrum. The knee at 2 cycles per wavelength is real (gains 0.71, 0.68, 0.63, 0.12, 0.012 at kappa = 1, 1.5, 2, 2.5, 3).
- 06-c (section 3). Before interpreting the fitted d_eps/Rytov of about 0.78, run the controls that separate the readings: swap the layer order (eps1 <-> eps2: Rytov unchanged, the Born-linear boundary terms flip sign), halve Delta chi at fixed mean (Rytov scales as Delta chi^2), and fit a homogeneous disk whose truth is eps = 2.25 + Rytov (the fit must return it). Remove "the fit does not isolate the bulk correction" as a finding unless the controls support it; state the number either way.
- 06-d (section 3). Detectability window: evaluate z and DeltaPhi directly at p = 0.046, 0.065 and 0.084, not only by log-log extrapolation; report the flag probability P(chi2_4800/2 + DeltaPhi > 2400 + 147) instead of DeltaPhi < 147. The ordering p_3nat < p_z < p_flag follows from c = DeltaPhi/z^2 = 1.49 lying in (3/9, 147/9); say so.
- 06-e (section 4). Tangents are analytic (scatter.disk_tangents) or central differences with step 1e-5; set the sector tolerances and the selection-rule tolerance to 1e-8. Assert exact C4 invariance of the painted chi (1e-15) before solving. State the conditions: tangents of the exact rotation-invariant Mie model (testbed tangents are only C4-covariant), grid, lattice and rings sharing the center with the 90-degree rotation mapping tx j -> j + 10 and rx l -> l + 15, scalar whitening. The base eps is lattice_disk_eps_bar (whole rods with centers in the disk: 1.8775, 1.8527, 1.8269).
- 06-f (section 5). Before computing coverage, compare the linearized z_i with nonlinear fits for 4 held-out cuts spanning X0 (including the period cut); if they differ by more than 0.1 sd, compute the approximation-error model in the foot-point frame.

## Thesis

For a two-phase periodic medium, homogenization is a pair of dual convex programs. Their conforming discretizations certify the effective tensor from both sides and satisfy Keller-Dykhne duality exactly on the pixel grid: L_h U_h = ab for the checkerboard at every resolution, which the design scratch confirmed to 1e-12. The four-tile corner exponent lambda_1 = (4/pi) arctan(kappa^-1/2) sets how fast that bracket closes. As a function of contrast, the same operator is a Stieltjes function whose positive spectral measure organizes four things: the sum rules, the corner bands, the plasmonic resonances, and the Bergman-Milton bounds. The measure for a disk array is given exactly by the eigenpairs of a real symmetric Rayleigh matrix. How much a spectrum says about a microstructure depends on how close the probing contrasts come to the support of the measure. The set of attainable tensors (the G-closure) is bounded by trace inequalities. Whether a single periodic inclusion reaches those bounds, which the Vigdergauz and E-inclusion literature predicts and three proposals denied, is tested with certified optimization. On a finite domain, the first-order two-scale expansion is wrong at O(eps) in the interior only through boundary-layer tails, and these vanish when the boundary cuts the cell along a mirror line. With such a cut the H1 error is O(eps), not the textbook eps^(1/2). Both effects were verified in scratch runs. At band edges the homogenized object is an envelope equation. Its form is fixed by the little group (three constants at the C4v doublet, one Dirac velocity at C3v), and its boundary condition again depends on where the walls cut the cell. In the TM imaging testbed the leading homogenized medium is the plain average, and a second-order term, exact in contrast, is the only bulk trace of the microstructure. Near the object's boundary, the termination of a layered microstructure appears exactly as a radius shift, which a homogeneous fit absorbs. The detectability law of DESIGN section 15 then decides whether the remainder biases the composition estimate or flags the model. For the termination-free ring structure the scratch run gives a window of periods where the estimate is biased by more than 3 sd while the misfit passes the 3-sigma goodness-of-fit test.

## Proposal scores (design judge)

- **A: mathematical structure first**: 7.5/10. Strengths: Broadest coverage of the mathematical structure for this reader: Q1 primal/dual bounds with a proof sketch, G-closure, the Bergman-Milton measure tied to the Neumann-Poincare operator, two-scale rates with the evanescent-Bloch reading of boundary layers, little-group k.p (C4v quadratic touching, C3v Dirac, Berry phase), and the testbed link through the detectability law. Concrete solvers and honest confidence flags. Weaknesses: Several concrete checks fail or are infeasible (verified in scratch). The Q1 spectral measure of the checkerboard has a spurious atom at z = 0 of weight 0.17-0.20 at N = 16-32, and m1 converges slowly (0.1220, 0.1235, 0.1241), so 'mass and first moment to 1e-6 at N = 48' and 'Kolmogorov distance decreasing' are wrong. The percolation crossover cannot be measured with pixel Q1, because shared nodes connect diagonal pixels. 'HS is not attainable by a single-scale C4v cell' contradicts the Vigdergauz and E-inclusion literature. Second-order correctors with an eps^3 interior rate ignore boundary-layer tails; the interior error is O(eps) or O(eps^2) depending on the cut. 'High-index inclusions slow the boundary layer' is reversed at R = 0.3 (kappa_1 > 2 pi). Fitting the boundary-layer decay in the fine solve is infeasible, since the layer falls by e^-2pi per cell. A homogeneous QEP cannot match 2 pi to 1e-10 (the error is O(h^2)). Dilute-pole expectations do not hold at f = 0.28 (Rayleigh: 97.7% of the weight, shifted 3.4e-3). Six heavy notebooks.
- **B: verification-first computation**: 7/10. Strengths: The exact-coefficient Fourier-Galerkin bracket for the true disk is a rigorous certificate with no geometric error; scratch confirms width about 1.4/K and that it contains Rayleigh's 2.4154673802. Its scratch numbers reproduce (MS checkerboard rates 1.28 and about 0.5; Rayleigh 2.41547). Each notebook is built around a prediction that could fail. The concentric-ring scatterer with exact multilayer Mie is the best idea for a clean, exact imaging experiment. Weaknesses: 'Bracket midpoint agrees to 1e-4 at K = 64' is wrong: the midpoint is 3.7e-3 high because the upper bound converges about 10 times more slowly; only per-bound Richardson extrapolation reaches 1e-4. The solver-zoo notebook is engineering, not structure. The corner notebook mixes the sampled scheme's empirical rates with conforming corner theory; conforming Q1 follows 2 lambda_1 exactly (scratch). 'Single scale cannot attain HS' shares the same likely-wrong premise. It drops two-scale analysis, where the sharpest correct result (cut-dependent tails) lives. The 2D supercell envelope test is too expensive, and the near-touching Keller test needs about 40 multipoles (scratch, f = 0.78).
- **C: optics and the project**: 6.5/10. Strengths: The organizing TM observation is right and central: the leading order is the arithmetic mean, and the second-order law eps_eff = <eps> + k0^2 sum |eps_G|^2/|G|^2 is exact in contrast. Scratch PWE confirms the residual is O(k0^4) for eps 4 and 12. Also useful: the Born-cutoff argument, using the detectability law on a physical model error, the angular selection rule, and the approximation-error model. Weaknesses: The EMA zoo and RSA ensembles are applied and tutorial-adjacent for this reader. It depends on the untested rodlattice.cluster for its central experiment. The hand-estimate check fails (the first G-shell gives 1.094, the full sum 1.287 (a/lambda)^2). 'P_perp delta scales like a from boundary-layer correctors' uses a divergence-form mechanism that TM does not have. In scratch the O(p) termination effect is a pure radius shift lying in the tangent space. The relaxation thesis (designs do not converge) is doubtful in 2D conductivity, where periodic E-inclusions attain the bounds. 'Confidently wrong before detectable' holds only against the 147-nat goodness-of-fit flag, not a 3-nat evidence test.
- **S: orchestrator draft**: 8/10. Strengths: Its premises are the most nearly correct: the discrete interchange question (settled exactly by the primal/dual pair), the checkerboard density (1/pi) sqrt((1-z)/z), imaginary corner exponents and non-commuting limits at negative contrast, Vigdergauz/E-inclusion attainment, Huet ordering, the Stieltjes inverse problem, the div-curl lemma, and the TM law with Rytov's limit. Coverage is balanced across the uncovered topics. Weaknesses: Thin specification: no grids, tolerances, or failure predictions. The Gamma-point Conca-Vanninathan TE Hessian duplicates the static TE permittivity already checked against Rayleigh in spatial_dispersion, and the Burnett-sign claim is unverified. The arccos corner formula is the larger branch; the smallest exponent is (4/pi) arctan(kappa^-1/2). The two-scale part misses the cut dependence, and its 'eps^1/2 boundary layer in H1' is false for mirror-line cuts (scratch slope 0.99).

## Notebooks

### 01 `01_cell_problem_duality`: The cell problem as a pair of convex programs: certified brackets, Keller-Dykhne on the grid, corner-limited convergence

**Question.** For a two-phase periodic cell, what do conforming discretizations of the Dirichlet (potential) and Thomson (flux) principles certify about the effective tensor? Which duality identities do they satisfy exactly on the grid, and what limits how fast the certified bracket closes?

**Why.** Every later notebook measures effective tensors with these solvers. This one turns their numerical error into a rigorous interval and proves that the discrete pair respects Keller-Dykhne duality exactly. It also explains, from the four-tile corner exponent, why pixel solvers converge so slowly at high contrast, so the reader gets certificates instead of convergence plots.

Run-time budget: 150 s with 4 threads. Modules: cellsolver.py, bounds.py, rayleigh.py.

#### 1. Two minimum principles, one solver

State the two principles. Upper: sigma*_ee = min over periodic phi of <sigma |e + grad phi|^2>. Lower: (sigma*^-1)_ee = min over divergence-free periodic J with <J> = e of <sigma^-1 |J|^2>. In 2D, J = e' + R grad psi with R = [[0,-1],[1,0]], so the flux principle is the potential principle for sigma^-1 rotated, and the lower bound is L_h[sigma] := inv(R^T U_h[1/sigma] R), where U_h is the Q1 value. Two-line proof: U_h >= sigma* >= L_h in the Loewner order for any pixel field, because Q1 spaces are conforming subspaces of H^1_per. Both bounds are monotone under 2x2 pixel refinement of a fixed pixel geometry, because the spaces are nested. Element matrix (1/6)[[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]] times the pixel value, independent of h. Load integrals: int dN/dx = h(-1/2, 1/2, 1/2, -1/2), int dN/dy = h(-1/2, -1/2, 1/2, 1/2). Nodes are ordered (i,j), (i+1,j), (i+1,j+1), (i,j+1), with i the x index.

*Computations.* cellsolver.q1_bracket on: (a) laminates N = 64 with 32-pixel stripes, sigma in {1, 10}, stripes normal to x and to y; (b) 10 random iid two-phase pixel fields, N = 32, p = 1/2, values {1, 10}, seeds 0-9, and their phase-swapped fields; (c) the 2x2-tile checkerboard (a = 1, b = 10) at N = 16, 32, 64, each obtained from the previous by 2x2 pixel replication.

*Checks.*

- laminate: both bounds equal the Wiener means. Reference: exact harmonic mean 1/(0.5/1 + 0.5/10) = 1.818181... across the stripes, arithmetic mean 5.5 along them. Target: L_h = U_h = exact value, both orientations, xx and yy. Tolerance: 1e-12 absolute (the exact potential is piecewise linear, so it lies in the Q1 space) (a priori).
- Loewner order on random fields. Reference: variational theorem. Target: min eigenvalue of U_h - L_h >= -1e-12 for all 10 fields. Tolerance: 1e-12 (a priori).
- discrete interchange identity (unit test of the rotation convention). Reference: definition of L_h with 1/sigma = sigma_swap/(ab). Target: ||U_h[sigma_swap] - ab R inv(L_h[sigma]) R^T||_F / ||U_h[sigma_swap]||_F for all 10 fields. Tolerance: 1e-10 (a priori).
- monotone under refinement. Reference: nested Q1 spaces. Target: U_xx(16) >= U_xx(32) >= U_xx(64) and L_xx non-decreasing (checkerboard, kappa = 10). Tolerance: slack 1e-12 (a priori).

#### 2. Keller-Dykhne on the grid

Continuum results: Keller's interchange sigma*_xx(a,b) sigma*_yy(b,a) = ab, and Dykhne's sqrt(ab) for the checkerboard. Discrete result: the swapped checkerboard is a one-tile translate and a 90-degree rotation of itself, both exact grid symmetries for even N. Hence L_h U_h = ab exactly at every N, the bracket is symmetric in log scale, and its geometric mean is exact. The primal-primal product U_h[sigma] U_h[sigma_swap] is not exact. This answers proposal S's question: the interchange theorem holds exactly for the primal-dual pair and only to discretization error otherwise.

*Computations.* Checkerboard with a = 1, b = kappa in {4, 10, 100}, N in {16, 32, 64, 128, 256, 512}; store U_h and L_h for section 3. In the design scratch (q1.py) |L_h U_h/ab - 1| <= 1.4e-12 for all cases, the largest at kappa = 100, N = 512.

*Checks.*

- discrete Dykhne identity. Reference: exact identity from grid symmetries. Target: |L_h U_h/(ab) - 1| for all kappa and N. Tolerance: 1e-10 (a priori).
- primal-primal product is not exact. Reference: U_h^2/(ab) - 1 is about 2x the relative primal error. Target: |U_h[sigma]_xx U_h[sigma_swap]_yy/(ab) - 1| >= 1e-3 at kappa = 10, N = 64 (scratch relative primal error 2.9e-2). Tolerance: lower bound 1e-3 (a priori).

#### 3. The four-tile corner sets the rate

Separable solutions r^lambda Phi(theta) at a cross point with values a, b, a, b in the four quadrants. Continuity of Phi and of sigma Phi' at the four interfaces gives an 8x8 system D(lambda) for (A_j, B_j) in Phi_j = A_j cos(lambda theta) + B_j sin(lambda theta). The mirror classes give tan^2(lambda pi/4) = 1/kappa or kappa, so the smallest exponent is lambda_1 = (4/pi) arctan(kappa^-1/2) (proposal S's arccos form is the other branch). The energy error of a conforming method is about h^(2 lambda_1). As kappa grows, lambda_1 -> 0 and r^lambda_1 is about 1 + lambda_1 ln r on resolved scales, so the power law shows only for N >> N* = exp(1/lambda_1): 5 for kappa = 4, 13 for 10, about 2.6e3 for 100. For comparison, the sampled-coefficient spectral (Moulinec-Suquet) discretization on the same grids, solved by PCG with the inverse-Laplacian preconditioner (Nyquist derivative modes set to zero; mean-coefficient scaling). It has no sign guarantee and no corner theory here.

*Computations.* (a) lambda from D(lambda). Scan lambda in (0.001, 1.999) on 4000 points for local minima of the smallest singular value of the row-normalized D(lambda), then refine with scipy.optimize.minimize_scalar. Do not use sign changes: roots can be double by symmetry. Do this for kappa in {4, 10, 100, 1000}. (b) Relative errors e_N = U_h/sqrt(kappa) - 1 from section 2 and local rates log2(e_N/e_2N). (c) cellsolver.spectral_sampled for N in {32, 64, 128, 256, 512}, kappa in {10, 100}, tol 1e-11: error, local rate, iterations. Scratch, same algorithms. Q1 local rates: kappa = 4: 1.1871, 1.1832, 1.1818, 1.1811, 1.1809 (2 lambda_1 = 1.1807); kappa = 10: 0.809, 0.796, 0.789, 0.785, 0.783 (0.780); kappa = 100: 0.383, 0.353, 0.332, 0.316, 0.304 (0.254). MS errors: kappa = 10: +6.1e-3, +2.5e-3, +1.0e-3, +4.3e-4 for N = 32-256 (rate 1.28); kappa = 100: +0.34, +0.24, +0.17, +0.12 (rates 0.53, 0.51, 0.49); 30-92 PCG iterations. Report the MS rate next to the candidates 2 lambda_1, 4 lambda_1 and 2 lambda_1 + 1/2 without claiming any of them.

*Checks.*

- corner exponent: numerical root vs closed form. Reference: (4/pi) arctan(kappa^-1/2). Target: |lambda_root - formula| for kappa in {4, 10, 100, 1000}; smallest singular value at the root <= 1e-12. Tolerance: 1e-9 (a priori).
- second branch. Reference: (4/pi) arctan(kappa^1/2) > 1. Target: a root of D within 1e-9 of it for each kappa. Tolerance: 1e-9 (a priori).
- Q1 local rate -> 2 lambda_1 for kappa = 4 and 10. Reference: FE corner-singularity theory. Target: |rate(256->512) - 2 lambda_1|. Tolerance: 0.01. Set from the design scratch run of this algorithm; the a priori statement is only convergence of the local rate to 2 lambda_1 from above. (set after a convergence study or from the design scratch).
- kappa = 100 is pre-asymptotic. Reference: N* = exp(1/lambda_1), about 2.6e3, >> 512. Target: local rates strictly decreasing in N, all > 2 lambda_1, and rate(256->512) - 2 lambda_1 >= 0.03 (scratch 0.05). Tolerance: as stated (set after a convergence study or from the design scratch).
- sampled scheme is far more accurate here but uncertified. Reference: Q1 relative error at the same N. Target: MS relative error at kappa = 10, N = 256 < 0.1 x Q1 relative error (scratch 4.3e-4 vs 9.8e-3); MS error positive and decreasing in N. Tolerance: factor 0.1 (set after a convergence study or from the design scratch).
- preconditioned iterations nearly N-independent. Reference: spectral equivalence of the preconditioner. Target: iterations(N=512)/iterations(N=64) <= 2 for both kappa (scratch 35/33 and 92/72). Tolerance: factor 2 (set after a convergence study or from the design scratch).

#### 4. Disks: a certified bracket on the true geometry

Exact disk coefficients: sigma_hat(k) = s_out delta_k0 + (s_in - s_out) f 2 J1(|k|R)/(|k|R) e^(-i k.c), with k = 2 pi (n_x, n_y). They make the Galerkin quadratic form exact on trigonometric fields of degree K in each coordinate, since products need sigma_hat only up to degree 2K. Apply it by zero-padded FFT on an M x M grid with M = 4K + 2, which prevents aliasing into |k| <= K. The primal gives a rigorous upper bound for the TRUE disk, the stream-function dual (coefficients of 1/sigma, same shape, direction swapped) a rigorous lower bound. Both are monotone in K and close like 1/K because of the gradient jump across a smooth curve. Exact reference: rayleigh.sigma_eff (the symmetric Rayleigh matrix W of this plan). Also HS, Maxwell Garnett (equal to the HS bound in 2D), Bruggeman, Wiener. A pixel disk is a different object (a staircase); proposal B conjectured its error is mostly the area fraction, which is tested here.

*Computations.* (i) Square arrays of disks at (f, kappa) in {(0.5, 10), (0.5, 100), (0.282743, 12)}. The last is the TE static permittivity of the eps = 12 rods of spatial_dispersion, with eps playing the role of sigma. fg_bracket at K in {8, 16, 32, 64, 128}. (ii) rayleigh.sigma_eff with L in {5, 10, 20}. (iii) q1_bracket on center-sampled pixel disks, N in {64, 128, 256}: strong disks (0.5, 10), and weak disks (phase 1 = 1 inclusions in phase 2 = 10) at f = 0.5 and 0.7, with f_pix printed. Scratch, FG at (0.5, 10): K = 8 [2.400875, 2.581203], K = 16 [2.408174, 2.494377], K = 32 [2.411771, 2.454969], K = 64 [2.413617, 2.435077]. Rayleigh 2.4154673802 (converged by L = 5). Linear-in-1/K extrapolation of each bound gives 2.4154 and 2.4152; the K = 64 midpoint is 2.4243. Q1 weak disks (f = 0.5): N = 64 [4.11272, 4.14078] and N = 128 [4.11233, 4.12509], while Rayleigh(f_pix) = 4.15703 and 4.13147.

*Checks.*

- Rayleigh converged. Reference: exponential convergence in multipole order. Target: |sigma(L=20)/sigma(L=10) - 1| for the three cases. Tolerance: 1e-11 (a priori).
- Rayleigh vs spatial_dispersion's series at eps = 12. Reference: 1 + 2f/(1/b - f - 0.305827 b f^4 - 0.013362 b f^8) = 1.6301245119 (spatial_dispersion nb03 agreed with it to 1e-7). Target: |sigma - 1.6301245119| (scratch 5e-8). Tolerance: 1e-7 (a priori).
- Rayleigh value at (0.5, 10). Reference: design scratch 2.4154673802; proposal B 2.41547. Target: |sigma - 2.4154673802|. Tolerance: 1e-9 (set after a convergence study or from the design scratch).
- dilute limit. Reference: Maxwell Garnett; first lattice correction 3 S4^2/pi^4 beta f^4. Target: |sigma/sigma_MG - 1| at f = 0.01, kappa = 10. Tolerance: 1e-9 (a priori).
- FG bracket contains Rayleigh. Reference: theorem (conforming trig spaces, exact form). Target: L_K <= sigma_Rayleigh <= U_K for all K and cases. Tolerance: slack 1e-12; any violation is a bug (a priori).
- FG monotone in K. Reference: nested spaces. Target: U_K non-increasing, L_K non-decreasing. Tolerance: slack 1e-12 (a priori).
- bracket closes like 1/K. Reference: best approximation of a gradient jump across a smooth curve. Target: slope of log(U_K - L_K) vs log K over K = 16-128 in [-1.15, -0.85]. Tolerance: 0.15 (a priori).
- extrapolated bounds. Reference: Rayleigh. Target: each bound extrapolated linearly in 1/K from K = 64 and 128 within 5e-4 relative of Rayleigh at (0.5, 10); printed, not rigorous. Tolerance: 5e-4 (set after a convergence study or from the design scratch).
- midpoint is biased (correction of proposal B). Reference: Rayleigh. Target: relative midpoint error at K = 64, (0.5, 10), >= 1e-3 (scratch 3.7e-3). Tolerance: lower bound 1e-3 (set after a convergence study or from the design scratch).
- HS contains every exact value; MG = HS. Reference: HS theorem; algebra. Target: HS_lo <= sigma <= HS_up for every Rayleigh value; |MG - HS bound| for the matching phase order. Tolerance: 1e-14 (a priori).
- staircase is not only an area-fraction error (proposal B's conjecture). Reference: Rayleigh at f and at f_pix. Target: the certified bracket of the weak staircase disk (f = 0.5, N = 128) excludes Rayleigh(f_pix). Scratch: yes. If a rerun differs, the prose must follow the printed outcome.. Tolerance: exact interval test (set after a convergence study or from the design scratch).

#### 5. Windows of a random checkerboard: ordering and Dykhne in law

Apparent tensors of a finite window W come in three kinds: affine (fluctuation phi = 0 on the boundary), periodic, and uniform flux (stream function psi = 0 on the boundary, which is the affine problem for 1/sigma rotated, inverted). Trial-space inclusion gives L_flux <= L_per <= sigma*_per(W) <= U_per <= U_aff for every realization, both discrete and continuum; the ensemble version is Huet's partition theorem. For p = 1/2 the phase-swapped field has the same law. Section 2's identity then gives det U_per[sigma] det L_per[sigma_swap] = (ab)^2 exactly, so antithetic pairs make the discrete bracket respect Dykhne's sqrt(ab) in law. The scatter of the apparent tensor is about 1/L (CLT, d = 2).

*Computations.* Random checkerboard of L x L tiles, L in {8, 16, 32}, m = 4 pixels per tile (N = 32, 64, 128), p = 1/2, a = 1, b = 10. 16 antithetic pairs per L, seeds 0-15. Full 2x2 tensors from q1_bracket (periodic) and q1_window (affine and flux).

*Checks.*

- ordering per realization. Reference: trial-space inclusion. Target: eigenvalues of U_aff - U_per, U_per - L_per, L_per - L_flux >= -1e-10 for all 96 windows. Tolerance: 1e-10 (a priori).
- antithetic Dykhne identity. Reference: section 2 identity. Target: |log det U_per[sigma] + log det L_per[sigma_swap] - 2 log(ab)| per pair. Tolerance: 1e-10 (a priori).
- CLT scaling of the scatter. Reference: variance proportional to 1/area. Target: fitted slope of log std((U_per)_xx) vs log L in [-1.3, -0.7]. Tolerance: 0.3 (sampling error of a std from 32 windows over two octaves) (a priori).

*Findings to test.*

- The Q1 local rate on the checkerboard approaches 2 lambda_1 from above; kappa = 100 stays pre-asymptotic up to N = 512, as N* = exp(1/lambda_1) predicts.
- The sampled spectral scheme converges faster than Q1 on the checkerboard (rate about 1.28 at kappa = 10) with no certificate; no corner law is claimed for it.
- The exact-coefficient FG bracket for the true disk closes like 1/K, and its midpoint is biased because the upper bound converges slowly (correction of proposal B).
- Staircase disks: the certified bracket excludes Rayleigh at the pixel area fraction, so the staircase shape matters beyond the area (refutes proposal B's conjecture for weak inclusions).
- Antithetic pairs satisfy the discrete Dykhne identity exactly; the scatter of apparent tensors scales like 1/L.

*Risks and mitigations.*

- The determinant D(lambda) has double roots by symmetry: use the smallest singular value, not sign changes.
- splu at N = 512 with kappa = 100 takes a few seconds (scratch fine). If the notebook exceeds 150 s, drop N = 512 for kappa = 4 and 100 in section 3 (keep kappa = 10), then K = 128 in section 4.
- FG PCG at kappa = 100 needs about 100-150 iterations: cap at 2000 and assert convergence (residual below tol).
- Rayleigh S_2 convention: use the derived W, whose (1,1) entry carries pi R^2, and validate that L = 1 reproduces MG exactly before any other use.
- Window flux bounds are inverses of 2x2 tensors, never componentwise reciprocals.

### 02 `02_spectral_measure`: The effective permittivity as a Stieltjes function: spectral measures, corners, plasmons, and what a spectrum determines

**Question.** The effective permittivity of a two-phase cell is F(s) = 1 - eps*/eps2 = integral dmu(z)/(s - z), with s = 1/(1 - eps1/eps2) and a positive measure mu on [0,1]. What does mu look like for smooth inclusions, corners and touching corners, and how does each discretization distort it? What does mu imply for plasmonic absorption, for bounds at complex contrast, and for how much a spectrum can say about a microstructure?

**Why.** It is the operator-theoretic core of homogenization: Bergman-Milton spectral theory, Neumann-Poincare spectra at corners, and Herglotz bounds. In one object it explains bounds, plasmon resonances and an ill-posed inverse problem. Exact references exist (the checkerboard density and the Rayleigh matrix), so every distortion can be measured.

Run-time budget: 150 s with 4 threads. Modules: cellsolver.py, spectral.py, rayleigh.py, bounds.py.

#### 1. One Stieltjes function, three discretizations

Sum rules: m0 = f1, and m1 = f1 f2/2 for C4-symmetric cells. Q1 discretization: the pencil K1 v = t K v on the pinned periodic Q1 space, with K the unit-coefficient stiffness and K1 that of the phase-1 pixels. It gives F_h(s) = f/s + (1/s) sum w_k/(s - t_k), w_k = (v_k^T b)^2 with V^T K V = I and b the phase-1 load. Partial fractions put weight w_k/t_k at t_k and weight f - sum w_k/t_k at z = 0. The atom at 0 equals lim U_h/sigma1 as sigma1 -> infinity, so it is nonzero exactly when phase 1 percolates in the discrete space. Fourier-Galerkin with exact coefficients: the same structure with K0 = diag(|k|^2) and K1 = [chi1_hat(k - l) k.l], a Hermitian problem after scaling by K0^-1/2. Sampled spectral (MS): T = chi1 Gamma chi1 on phase-1 vector fields, F(s) = <u, (s - T)^-1 u> with u = chi1 e. Lanczos with full reorthogonalization from u/||u|| (mean inner product, ||u||^2 = f) gives Gauss nodes and weights f times the squared first components. Prediction: an FG measure carries a spurious atom at 0 even for an isolated disk (no trigonometric polynomial is affine on the disk), with weight about O(1/K). A Q1 measure of an isolated pixel inclusion carries none, because Q1 can be affine on any non-percolating pixel set.

*Computations.* Disk R = 0.3 (f = 0.282743). Q1 at N = 32 (center-sampled), FG at K = 16, Lanczos on the sampled disk at N = 128 with 150 and 300 steps. Direct solves (q1_effective, fg_bracket upper, spectral_sampled) at h = sigma1/sigma2 in {0.1, 10, 100}.

*Checks.*

- Q1 measure reproduces the Q1 solve. Reference: algebraic identity. Target: |sigma_from_measure/sigma_direct - 1| at h in {0.1, 10, 100}. Tolerance: 1e-10 (a priori).
- FG measure reproduces the FG upper bound. Reference: algebraic identity. Target: same three contrasts. Tolerance: 1e-10 (a priori).
- Lanczos measure reproduces spectral_sampled. Reference: Gauss quadrature convergence for s outside [0,1]. Target: |relative difference| at 300 steps; also |value(300) - value(150)| printed. h = 100 (s = -0.0101) converges slowest.. Tolerance: 1e-8, fixed after the 150/300-step comparison: if 300 steps do not reach it, raise to 600 and record the count (set after a convergence study or from the design scratch).
- zeroth moment exact. Reference: m0 = f for pixel-constant phases (all three). Target: |sum of weights - f|. Tolerance: 1e-12 (a priori).
- first moment. Reference: Q1 and FG: m1 <= f(1-f)/2 (variational); sampled: m1 = f(1-f)/2 minus half the Nyquist-mode energy of chi1. Target: inequalities for Q1 and FG; identity for Lanczos. Tolerance: 1e-12 for the identity (a priori).
- atom at 0. Reference: Q1 is affine on a non-percolating set; trigonometric polynomials are not. Target: Q1 isolated disk: atom weight <= 1e-12. FG disk: atom weight positive and decreasing for K = 8, 12, 16, 20 (to test; predicted about 1/K).. Tolerance: 1e-12 (Q1) (a priori).

#### 2. Exact measures

(a) Maxwell Garnett: a single atom of weight f at z = (1 - f)/2. Isolated disk: z = 1/2. Dilute ellipse along its long axis: the depolarization factor b/(a+b), which is 1/3 for 2:1. (b) Checkerboard: dmu = (1/pi) sqrt((1 - z)/z) dz on (0,1), with no atoms, from eps* = sqrt(eps1 eps2); CDF M(z) = (arcsin sqrt(z) + sqrt(z(1-z)))/pi. (c) Square array of disks, exactly: W_lm = sqrt(l/m) R^(l+m) [pi delta_l1 delta_m1 + C(l+m-1, l) S_(l+m) 1(l+m >= 4)] over odd l, m = 1, 3, ..., 2L - 1. Here S_n are the square-lattice Eisenstein sums, and the pi term is the periodic-field (S_2) term. Then eps*/eps_m = 1 + 2f [(I/beta - W)^-1]_11 with beta = (kappa-1)/(kappa+1). The atoms are z_k = (1 - mu_k)/2 with weights f (v_k)_1^2, where W = V diag(mu) V^T. W_11 = f makes m0 = f and m1 = f(1-f)/2 exact at every truncation. Derived in this design: it reproduces MG at L = 1 and the 0.3058 f^4 coefficient at L = 2.

*Computations.* rayleigh.rayleigh_measure at f in {0.01, 0.1, 0.282743, 0.5, 0.7}, L = 20 and 40. Checkerboard density moments by scipy.integrate.quad with the algebraic endpoint weight. Scratch: at f = 0.282743 the dominant atom is z = 0.355273 (MG 0.358650) with weight fraction 0.97717, and the next atoms are 0.505804 (4.39e-3) and 0.497545 (2.01e-3). At f = 0.5 the dominant atom is z = 0.231774 (93.4%), and W has spectrum [-0.0954, 0.5365].

*Checks.*

- Rayleigh sum rules at every truncation. Reference: W_11 = f. Target: |m0 - f|, |m1 - f(1-f)/2| for L in {1, 2, 5, 20}. Tolerance: 1e-14 (a priori).
- atoms inside (0,1). Reference: positivity of mu. Target: eig(W) strictly inside (-1, 1) for all f listed. Tolerance: strict (a priori).
- measure reproduces sigma_eff. Reference: rayleigh.sigma_eff. Target: kappa in {0.1, 10, 100, -2 + 0.5i}. Tolerance: 1e-12 relative (a priori).
- truncation converged. Reference: exponential convergence. Target: atoms with weight > 1e-8 identical at L = 20 and 40. Tolerance: 1e-10 (a priori).
- dilute limit. Reference: first lattice correction O(f^4). Target: f = 0.01: dominant atom within 1e-7 of (1-f)/2 and carrying >= (1 - 1e-6) f. Tolerance: as stated (a priori).
- dominant atom at f = 0.2827 (corrects proposal A's '> 99% within 1e-3'). Reference: scratch 0.355273, weight fraction 0.97717. Target: position and weight fraction. Tolerance: 1e-5 (set after a convergence study or from the design scratch).
- checkerboard density. Reference: exact. Target: integral = 1/2, first moment = 1/8, M(1) = 1/2. Tolerance: 1e-10 (a priori).

#### 3. What discretization does to a measure

(a) Disk: compare Q1 (staircase, against Rayleigh at f_pix), FG (true disk) and Lanczos with the exact Rayleigh atoms, using the L1 distance of CDFs (Wasserstein-1) and the error of the dominant atom. (b) Checkerboard, corner contacts. In Q1, diagonal tiles share a node, so phase 1 percolates in the discrete space and the measure carries an atom at z = 0. Scratch weights: 0.1978, 0.1794, 0.1683 at N = 16, 24, 32, decaying roughly like 1/ln N (point-contact spreading resistance). m1 = 0.12199, 0.12351, 0.12410 approaches 1/8 from below. The sampled scheme has m1 = 1/8 exactly. This corrects proposal A's check (mass and m1 to 1e-6 at N = 48, Kolmogorov distance decreasing): the Kolmogorov distance is dominated by the spurious atom.

*Computations.* Disk f = 0.282743: Q1 at N in {16, 24, 32, 40}, FG at K in {8, 12, 16, 20}, Lanczos at N in {64, 128, 256} with 150 steps. Checkerboard: Q1 at N in {16, 24, 32}, with the atom at 0 compared to q1_effective at sigma1 = 1e8, sigma2 = 1, divided by 1e8. Lanczos at N in {64, 128, 256} with 300 steps: weight below z = 0.01, and the CDF of the rest against M(z) - M(0.05) on [0.05, 1].

*Checks.*

- atom at 0 = percolation limit. Reference: F ~ mu0/s as s -> 0. Target: |mu0 - U_h(1e8)/1e8| / mu0, checkerboard, N = 16, 24, 32. Tolerance: 1e-6 (next term O(sigma2/sigma1)) (a priori).
- Q1 checkerboard m1 increases toward 1/8. Reference: m1 is a maximum over nested spaces. Target: m1(16) < m1(24)? Nested only under 2x replication, so assert m1(16) <= m1(32) < 1/8, and print all three. Tolerance: strict (a priori).
- sampled checkerboard m1 exact. Reference: no Nyquist content when N is divisible by 4. Target: |m1 - 1/8| at N = 64, 128, 256. Tolerance: 1e-12 (a priori).
- FG disk converges to Rayleigh. Reference: Rayleigh atoms. Target: dominant-atom error and W1 distance both decreasing over K = 8, 12, 16, 20 (to test). Tolerance: monotone decrease (set after a convergence study or from the design scratch).

#### 4. Corners: a band of spectrum

From memory (Perfekt-Putinar 2017): the essential Neumann-Poincare spectrum of a curvilinear polygon with interior angle alpha is [-|1 - alpha/pi|/2, |1 - alpha/pi|/2]. For 90 degrees, with z = 1/2 - lambda, this is the band z in [1/4, 3/4]. A smooth inclusion's atoms accumulate only at 1/2. Test this with a square inclusion (pixel aligned, side 1/2, f = 1/4) against a disk of equal area (R = 0.282095), using FG as the primary discretization. Q1 and Lanczos see staircases, whose pixel corners are themselves 90-degree corners, so they are shown but not asserted. At negative contrast the four-tile exponents become imaginary (proposal S): tan^2(lambda pi/4) = kappa^(+-1) with kappa < 0 has roots lambda = +-(4i/pi) artanh(|kappa|^(-1/2)) for |kappa| > 1 (use |kappa|^(1/2) for |kappa| < 1). These are oscillating singular fields, consistent with a purely imaginary sqrt(eps1 eps2).

*Computations.* FG at K in {8, 12, 16, 20} for the square (sinc coefficients) and the equal-area disk. Q1 at N in {16, 32, 48} and Lanczos at N in {64, 128, 256}, shown only. Statistics: the number of atoms carrying 99% of f, and the weight outside [0.23, 0.77]. Imaginary exponents for kappa in {-2, -0.5, -10}, checked as zeros of nb01's D(lambda) at complex lambda.

*Checks.*

- square develops a band (to test against Perfekt-Putinar). Reference: essential spectrum [1/4, 3/4], formula from memory. Target: FG square: the 99%-atom count grows with K, and the weight outside [0.23, 0.77] decreases with K. If either fails, the notebook states the formula is not confirmed by this test.. Tolerance: monotone trends (set after a convergence study or from the design scratch).
- disk stays atomic. Reference: Rayleigh: 3 atoms carry > 99.9% at f = 0.28. Target: FG disk 99%-atom count <= 5 at every K (to test; it may be inflated by the spurious atom at 0). Tolerance: <= 5 (set after a convergence study or from the design scratch).
- imaginary corner exponents. Reference: closed form. Target: smallest singular value of D at the predicted complex lambda. Tolerance: 1e-10 (a priori).

#### 5. Plasmonic composites

Drude inclusions eps1(omega) = 1 - omega_p^2/(omega(omega + i gamma)) in vacuum give s = omega(omega + i gamma)/omega_p^2. Each atom is a Lorentz oscillator at omega_k = omega_p sqrt(z_k) with strength w_k omega_p^2. Hence, for every microstructure, the sum rule integral_0^infinity omega Im eps*(omega) d omega = (pi/2) f omega_p^2, from m0 = f. The checkerboard absorbs over the whole band omega < omega_p even as gamma -> 0.

*Computations.* Measures: Rayleigh (disk lattice, f = 0.282743), checkerboard (exact sqrt), square (FG K = 20), MG. gamma/omega_p in {0.02, 0.1}; omega/omega_p on 20000 points in [0.005, 3]; the analytic tail beyond 3 omega_p is added. Scratch dominant disk atom 0.355273 puts the resonance at 0.59605 omega_p (MG: 0.59888 omega_p).

*Checks.*

- absorption sum rule. Reference: (pi/2) f omega_p^2. Target: relative error of the quadrature for each measure and gamma. Tolerance: 1e-3 (a priori).
- MG peak position. Reference: omega_p sqrt((1-f)/2), shift O(gamma^2). Target: |peak - prediction| at gamma = 0.02. Tolerance: 1e-3 omega_p (a priori).
- lossless checkerboard absorbs. Reference: Im eps* -> sqrt(|eps1(omega)|) for omega < omega_p as gamma -> 0. Target: gamma = 1e-6 at omega/omega_p in {0.3, 0.6, 0.9}. Tolerance: 1e-5 (a priori).

#### 6. Lossless but absorbing: the order of limits

Checkerboard with eps1 = -2 + i delta and eps2 = 1, so s = 1/3 at delta = 0. In the continuum, eps* -> i sqrt(2) as delta -> 0+. A finite discretization is a finite sum of atoms, so at fixed resolution Im eps*_h(delta) is proportional to delta as delta -> 0, unless an atom lies within about delta of 1/3. The limits h -> 0 and delta -> 0 therefore do not commute.

*Computations.* Q1 measures at N in {16, 24, 32} and FG at K in {8, 12, 16} (exact discrete measures; not Lanczos). delta in {1e-1, 1e-2, 1e-3, 1e-4}. Print d_min = min |z_k - 1/3|.

*Checks.*

- at fixed delta = 0.1 the discretizations approach the continuum. Reference: sqrt(-2 + 0.1i), principal branch. Target: error at the finest N (and K) below the error at the coarsest (to test). Tolerance: strict inequality (set after a convergence study or from the design scratch).
- at fixed resolution the lossless limit is real. Reference: Taylor expansion of a finite sum. Target: for every case with d_min > 1e-2: |(Im eps*_h(1e-4)/1e-4)/(Im eps*_h(1e-3)/1e-3) - 1|. Tolerance: 1e-2 (a priori).

#### 7. Bounds at complex contrast

Fix s outside [0,1]. The set {F(s): mu >= 0 on [0,1], m0 = f} is convex, and so is the smaller set that also fixes m1 = f(1-f)/2 (isotropy). These are the Bergman-Milton bounds. The support function of the f-only set is f max_z Re(e^(-i phi)/(s - z)), an exact 1-D maximization. The isotropic set's support function is a linear program over a z-grid; basic solutions have at most two atoms, and the extremal families have atoms at {z, 1} or {0, z}. MG, a single atom at (1-f)/2, is the corner where the two families of the isotropic lens meet.

*Computations.* f = 0.5. Contrasts h = eps1/eps2 in {0.2 + 0.5i, -2 + 1i, -0.5 + 0.2i}, all at distance >= 0.087 from [0,1] in s. 720 directions. Isotropic LP with scipy.optimize.linprog(method='highs-ds') on z_j = j/4000. Points tested: checkerboard sqrt(h), disk lattice at f = 0.5 (Rayleigh), MG.

*Checks.*

- exact values inside the f-only region. Reference: theorem. Target: Re(e^(-i phi) F_exact) <= support_f(phi) for all phi. Tolerance: 1e-9 (exact 1-D maximization with bounded refinement) (a priori).
- exact values inside the isotropic region. Reference: theorem. Target: Re(e^(-i phi) F_exact) <= support_iso(phi) + tol for all phi. Tolerance: grid tolerance f Dz / dist(s,[0,1])^2 with Dz = 2.5e-4, printed per contrast (a priori).
- extremal measures. Reference: basic LP solutions. Target: nonzero weights per isotropic solution <= 2, located at {z,1} or {0,z}. Tolerance: weights > 1e-10 counted (a priori).
- MG is a corner of the isotropic lens. Reference: MG's atom lies on the grid (1000/4000). Target: max over phi of [Re(e^(-i phi) F_MG) - support_iso(phi)] >= -1e-8. Tolerance: 1e-8 (a priori).

#### 8. How many numbers does a spectrum carry?

The map from a measure density, sampled on 400 z-points, to the data F(s_j) is linear, so its singular values give an information count. Three designs: (a) 60 lossless positive contrasts h in [1.5, 1000], i.e. s < 0, away from the support; (b) Drude spectra at 60 frequencies with omega/omega_p in [0.2, 1.2] and gamma = 0.2; (c) the same with gamma = 0.05. Count singular values above 1e-3 sigma_1, the 30 dB amplitude level. This is the imaging-theory reading: static or real-contrast data meet the exponential ill-posedness of the Stieltjes moment problem, while lossy plasmonic spectra probe s near the support and resolve the measure locally.

*Computations.* SVD of the real-stacked matrices. Scratch counts: (a) 6, (b) 22, (c) 75.

*Checks.*

- real-contrast data carry few numbers. Reference: exponential ill-posedness of the moment problem; scratch 6. Target: count(a) <= 10. Tolerance: <= 10 (set after a convergence study or from the design scratch).
- information grows as the probe nears the support. Reference: geometry of the Cauchy kernel; scratch 6 < 22 < 75. Target: count(a) < count(b) < count(c). Tolerance: strict (set after a convergence study or from the design scratch).

*Findings to test.*

- Conforming primal measures have an atom at 0 exactly when phase 1 percolates in the discrete space; for the Q1 checkerboard it decays only logarithmically (corrects proposal A).
- FG measures of smooth inclusions carry a spurious atom at 0 of order 1/K.
- At f = 0.28 the disk-array dipole atom carries 97.7% of the weight, not over 99% (corrects proposal A).
- Square inclusions develop a band of spectrum consistent with [1/4, 3/4] (Perfekt-Putinar, from memory).
- The order of limits h -> 0 and delta -> 0 matters for negative contrast.
- MG is the corner of the isotropic Bergman-Milton lens.
- Real-contrast data determine about six numbers of a microstructure at 30 dB; plasmonic spectra determine many more.

*Risks and mitigations.*

- Dense generalized eigh at N = 40 or 48 costs 10-20 s each: keep Q1 at N <= 40 except the checkerboard (N <= 32). If over budget, cut Q1 N = 40 to 32, FG K = 20 to 16, and directions 720 to 180.
- Lanczos loses orthogonality: use full reorthogonalization (300 steps on vectors of length 2 N^2 is cheap).
- LP near the support is inaccurate: contrasts were chosen with dist(s,[0,1]) >= 0.087, and the grid tolerance is computed, not guessed.
- The Perfekt-Putinar statement is from memory: assertions are trend tests, and the prose must follow the outcome.
- Q1 pencil: pin node 0 in both K and K1, and use scipy.linalg.eigh(K1, K).

### 03 `03_g_closure_and_optimal_cells`: Which tensors two phases can make: the G-closure, certified optimization, and whether one periodic inclusion reaches the bounds

**Question.** Which effective conductivity tensors can two isotropic phases (sigma1 = 1, sigma2 = 10, weak fraction theta1 = 0.7) produce in 2D? Which boundary points does a single periodic cell attain, and what does gradient-based design find once every design is certified with the Q1 bracket?

**Why.** Convexity and relaxation are the reader's home ground, and autodiff design is the ML link. Three proposals assumed that single-scale periodic cells cannot reach the Hashin-Shtrikman bound. The Vigdergauz and periodic-E-inclusion literature says otherwise. This notebook decides between the two with certified numbers instead of gray-scale objective values.

Run-time budget: 170 s with 4 threads. Modules: cellsolver.py, bounds.py, rayleigh.py.

#### 1. The G-closure in 2D

Wiener box: lambda_h <= lambda_1, lambda_2 <= lambda_a. Two trace bounds (Lurie-Cherkaev; Murat-Tartar; quoted from memory): 1/(l1 - s1) + 1/(l2 - s1) <= 1/(lh - s1) + 1/(la - s1), and 1/(s2 - l1) + 1/(s2 - l2) <= 1/(s2 - lh) + 1/(s2 - la). Sequential laminate formula: thetaB inv(A - sA I) = (sB - sA)^-1 I + (thetaA/sA) sum_k m_k e_k e_k^T. Rank-1 laminates give the corners; rank-2 laminates with weak matrix sigma1 trace the lower curve exactly and with strong matrix sigma2 the upper curve; the isotropic rank-2 laminates are HS. C4 symmetry forces isotropy (Schur). At theta1 = 0.7: HS = [1.65060, 2.71676], Wiener [1.36986, 3.7], weak-disk array (Rayleigh, R = 0.4720) 2.46202, which is 9.4% below HS_up. At theta1 = 0.5 that gap is only 1.28%, below what N = 64 certification can resolve, hence theta1 = 0.7.

*Computations.* bounds.laminate_seq over m in linspace(0, 1, 101) for both matrices; HS, Wiener, MG. A C4-symmetrized random cell at N = 32 for the isotropy check.

*Checks.*

- rank-2 laminates lie on the trace curves. Reference: algebra (formula and bounds checked jointly). Target: |trace-bound slack| for all 101 m, both curves. Tolerance: 1e-12 (a priori).
- rank-1 = Wiener, isotropic rank-2 = HS. Reference: closed forms. Target: absolute differences. Tolerance: 1e-14 (a priori).
- C4 forces isotropy. Reference: Schur's lemma; exact grid rotation. Target: |U_xx - U_yy|, |U_xy| of a C4-symmetrized random cell. Tolerance: 1e-12 (a priori).

#### 2. Random cells, certified

Thresholded Gaussian random fields fill the interior of the region. A tensor is only known to lie in [L_h, U_h], so a violation is certified only if U_h violates the lower trace bound (or falls below the Wiener box) or L_h violates the upper one (or exceeds the box). These are theorems, so any certified violation is a bug.

*Computations.* 300 cells at N = 32: FFT-filtered white noise with Gaussian correlation length ell in {1, 2, 4, 8} pixels (75 each, seeds fixed), thresholded by rank so the weak fraction is exactly 0.7. Full 2x2 q1_bracket of each; eigenvalue pairs plotted in the region.

*Checks.*

- no certified violation. Reference: G-closure theorem. Target: count of certified violations == 0. Tolerance: exact count (a priori).

#### 3. The gradient: envelope theorem and an autodiff cross-check

dU_ee/dsigma_p = E_p(chi_e), the element energy of e + grad chi_e in pixel p (envelope theorem; exact for Q1, no adjoint solve). E_p = h^2 + 2h g.chi_p + chi_p^T K_E chi_p for the element's nodal values, so sum_p sigma_p E_p = U_ee. Cross-check with JAX: a dense Q1 cell solve at N = 12, differentiated by jax.grad.

*Computations.* Random two-phase field at N = 12. Compare envelope, jax.grad, and central differences (step 1e-6) on 5 pixels.

*Checks.*

- envelope vs autodiff. Reference: both exact. Target: max relative difference over all pixels. Tolerance: 1e-10 (a priori).
- envelope vs finite differences. Reference: FD error O(step^2). Target: relative difference on 5 pixels. Tolerance: 1e-6 (a priori).
- energy identity. Reference: sum_p sigma_p E_p = U_ee. Target: relative difference. Tolerance: 1e-12 (a priori).

#### 4. The isotropic maximum: does one periodic inclusion reach Hashin-Shtrikman?

Two predictions. P1 (Vigdergauz; Liu, James and Leo 2007, from memory): a single periodic inclusion with uniform interior field (a periodic E-inclusion, a rounded square here) attains HS_up, so the certified gap shrinks toward 0 with N and the interior field becomes uniform. P2 (proposals A, B, C): single scale cannot attain HS, so the gap saturates. Method: C4v-symmetric density (gradients averaged over the 8 group images); rho = 1 strong. Interpolation sigma = s1 + (s2 - s1) rho^3. Periodic cone filter of radius 2 px. Tanh projection with threshold eta set by bisection so that mean(projected) = 0.3; beta continuation 1, 2, 4, 8, 16, 32 (40 iterations each); Adam, step 0.05, clipped to [0,1]. Objective: U_xx of the projected design. Final binarization by rank on the fundamental domain, keeping symmetry; the weak fraction must be within 8/N^2 of 0.7, and HS is recomputed at the actual fraction. Certify with q1_bracket at N and at 2N (2x2 replication, nested).

*Computations.* Starts: (i) weak disk R = 0.4720; (ii) weak square of side 0.8367; (iii, iv) 0.3 + 0.05 noise, seeds 0 and 1. N = 32 and 64: 240 iterations, 4 starts. N = 128: 160 iterations, 2 starts warm-started from the best upsampled N = 64 designs. Reported: certified gap g = (HS_up - L_h)/(HS_up - HS_lo); CV of |e + grad chi| over weak pixels (element-centered gradients); the same for the weak-disk and weak-square cells. Scratch reference, weak disk at theta = 0.7: Q1 brackets N = 32 [2.41217, 2.45218], N = 64 [2.42159, 2.44214], N = 128 [2.45619, 2.46674]; Rayleigh 2.46202.

*Checks.*

- rigor. Reference: HS theorem. Target: L_h(binary design) <= HS_up(theta_pix) for every design. Tolerance: slack 1e-12; violation is a bug (a priori).
- optimization beats the disk array. Reference: Rayleigh weak-disk array at theta_pix. Target: best certified L_h at N = 128 > Rayleigh value (to test). Tolerance: strict (set after a convergence study or from the design scratch).
- P1 vs P2: gap trend. Reference: Vigdergauz / E-inclusion literature vs proposals A, B, C. Target: best certified gap at N = 128 < at N = 64 < at N = 32 (P1 predicts yes). The verdict is printed either way; the assertion records whichever outcome the run gives, after a second run with another seed confirms it.. Tolerance: strict monotonicity (set after a convergence study or from the design scratch).
- field uniformity. Reference: E-inclusions have uniform interior field. Target: CV(optimized) < CV(weak disk) < CV(weak square), all at N = 128 (to test). Tolerance: strict (set after a convergence study or from the design scratch).

#### 5. An anisotropic boundary point

Target: the point of the upper trace curve at m* = 0.25 of the strong-matrix rank-2 family; scratch: lambda = (3.244, 2.100). Maximize U_xx + w U_yy with w = -1/(d lambda2/d lambda1) at m*, computed exactly in section 1 (about 0.68). With this w the maximum over G_theta is attained only at m*; with w = 0.5 the maximum sits at the laminate corner, which is why 0.5 is not used. A rank-2 laminate (two scales) attains it. P1 says a single anisotropic periodic inclusion also does; P2 (relaxation, proposal C) says designs develop ever finer stripes. Design convergence is measured by the fraction of differing pixels between the binary design at N and the 2x2-upsampled design at N/2.

*Computations.* D2-symmetric density (mirrors in x and y); same pipeline as section 4. Starts: seeds 0 and 1, and a stripe laminate normal to y. N in {32, 64, 128}.

*Checks.*

- rigor. Reference: G-closure theorem. Target: certified objective L_xx + w L_yy <= exact maximum. Tolerance: slack 1e-12 (a priori).
- gap trend and design convergence (P1 vs P2). Reference: competing predictions. Target: certified gap decreasing in N; Hamming distance between successive resolutions reported. P1 predicts it decreasing, P2 non-decreasing. The prose states the verdict from the printed numbers.. Tolerance: monotone trend (set after a convergence study or from the design scratch).

*Findings to test.*

- Whether a single C4v periodic inclusion approaches HS_up under refinement with a nearly uniform interior field (Vigdergauz / E-inclusions), or saturates (proposals A, B, C).
- Whether anisotropic boundary points are reached by single-scale designs that converge under refinement, against the relaxation prediction of ever finer stripes.
- No random cell violates the trace bounds, with certificates.
- The envelope-theorem gradient equals the autodiff gradient to rounding.

*Risks and mitigations.*

- Gray designs and the projection loss: only binary, certified values are reported.
- Local optima: 4 starts at N <= 64; report the best and the spread.
- The trace bounds are from memory: section 1 checks them against laminates (equality) and section 2 against random cells (no violation) before they are used as targets.
- Runtime: N = 128 LU is about 50-100 ms per iteration. If over budget, run one start at N = 128 and 120 iterations.
- Symmetric binarization can miss the exact fraction: HS is recomputed at the actual fraction.

### 04 `04_two_scale_and_boundary_layers`: Two scales on a finite domain: rates, the cut of the cell, boundary-layer tails, and the div-curl lemma

**Question.** For -div(a(x/eps) grad u_eps) = 1 on the unit square with u_eps = 0, how fast does u_eps approach the homogenized solution? Where is the first-order two-scale expansion w_eps = u0 + eps chi(x/eps).grad u0 wrong, and what determines its O(eps) interior error?

**Why.** This is homogenization as analysis: H-convergence, correctors, two-scale convergence and the div-curl lemma, measured rather than quoted. Scratch runs found that the textbook rates depend on where the domain boundary cuts the cell. Boundary-layer tails predict the interior error quantitatively, and the layer decay is an evanescent static Bloch wave. These are research-level points with exact checks, and they foreshadow the termination effects of notebooks 05 and 06.

Run-time budget: 120 s with 4 threads. Modules: twoscale.py, cellsolver.py.

#### 1. Correctors and the homogenized problem

Cell: a disk of radius 0.3 (cell units) with a_in = 10 in a_out = 1, center-sampled on m = 16 pixels per cell (a staircase disk; everything below uses the same discrete cell, so the comparisons test homogenization, not geometry). Q1 nodal correctors chi_1, chi_2 and sigma*_h; scratch sigma*_h = 1.688269 (the smooth disk's Rayleigh value is 1.60294). u0 = (torsion function)/sigma*_h, with torsion = sum over odd p, q of 16/(pi^4 p q (p^2 + q^2)) sin(p pi x) sin(q pi y), summed to p, q <= 301. Fine problems: N in {4, 8, 16, 32} cells per side (grids n = 64 to 512); the cell pattern is shifted by s cells, s in {0, 1/4, 1/2} (0, 4, 8 pixels), so the boundary cuts cells at y = 0 (cell edge), 3/4 (generic), or 1/2 (through inclusion centers).

*Computations.* twoscale.cell_correctors, fine_dirichlet, torsion. Homogeneous test with a = 1 at n = 128 and 256.

*Checks.*

- isotropy of sigma*_h. Reference: C4 symmetry of the cell. Target: |s_xx - s_yy|, |s_xy|. Tolerance: 1e-12 (a priori).
- same value as cellsolver. Reference: cellsolver.q1_effective on the same cell. Target: relative difference. Tolerance: 1e-12 (a priori).
- homogeneous medium: only the Q1 error remains. Reference: O(h^2) nodal error. Target: max|u_h - u0| ratio between n = 128 and 256 in [3, 5]. Tolerance: as stated (a priori).
- torsion series converged. Reference: tail of 1/(pq(p^2+q^2)). Target: max nodal difference between p, q <= 301 and <= 601. Tolerance: 1e-10 (a priori).

#### 2. Global rates, and why the textbook H1 rate needs a generic cut

The standard results are ||u_eps - u0||_L2 = O(eps), and ||grad(u_eps - w_eps)|| = O(eps^1/2) without a boundary-layer corrector. The second holds only for generic cuts. The mismatch of w_eps on an edge is eps chi_j grad_j u0. The tangential derivative of u0 vanishes there, and the normal corrector vanishes on mirror lines of the D4 cell. So for s = 0 and 1/2 there is no first-order boundary layer, and the H1 error is O(eps). Scratch: global H1 slopes 0.96 (s = 0), 0.99 (s = 1/2), and 0.65 overall with 0.586 for the last pair (s = 1/4); L2 slope 1.006; interior H1 slope 1.007.

*Computations.* For all s and N: L_inf and L2 of u_eps - u0; global H1 seminorm of the nodal interpolant of u_eps - w_eps (e^T K_1 e with the unit stiffness); interior H1 over elements with centers in [1/4, 3/4]^2.

*Checks.*

- L2 rate. Reference: BLP / JKO. Target: slope of ||u_eps - u0||_L2 over N = 4-32 within 0.15 of 1 (all s). Tolerance: 0.15 (a priori).
- H1 with a generic cut. Reference: boundary layer of width eps. Target: s = 1/4: local slope of the last pair in [0.45, 0.70]. Tolerance: as stated (pre-asymptotic; scratch 0.586) (set after a convergence study or from the design scratch).
- H1 with mirror-line cuts (corrects the general eps^1/2 claim). Reference: chi_n = 0 on mirror lines, tangential derivative of u0 = 0. Target: s = 0 and 1/2: overall slope within 0.1 of 1. Tolerance: 0.1 (a priori).
- interior H1. Reference: interior estimate. Target: slope within 0.1 of 1 (all s). Tolerance: 0.1 (a priori).

#### 3. The interior error depends on where the boundary cuts the cell

chi_1 is odd about y1 = 1/2, so it vanishes on y1 in {0, 1/2}, and likewise chi_2 about y2. For mirror-line cuts the boundary-layer tails vanish and the interior error of w_eps is O(eps^2). For the generic cut it is O(eps). Scratch interior max errors: s = 0: 8.59e-4, 2.39e-4, 6.21e-5 (rates 1.85, 1.94); s = 1/2: 1.18e-3, 3.27e-4, 8.57e-5 (rates 1.86, 1.93); s = 1/4: 5.68e-3, 2.79e-3, 1.37e-3 (rate 1.03) for N = 4, 8, 16.

*Computations.* Interior max over nodes in [1/4, 3/4]^2 of |u_eps - w_eps| for all s, N = 4-32.

*Checks.*

- mirror-line cuts: O(eps^2). Reference: vanishing tails. Target: slope over N = 8-32 within 0.2 of 2 for s = 0 and 1/2. Tolerance: 0.2 (a priori).
- generic cut: O(eps). Reference: nonzero tails. Target: slope over N = 8-32 within 0.1 of 1 for s = 1/4. Tolerance: 0.1 (a priori).
- symmetry zeros. Reference: oddness of the correctors. Target: max|chi_1| on node columns y1 = 0 and 1/2. Tolerance: 1e-12 (a priori).

#### 4. Boundary-layer tails predict the O(eps) interior error

Per edge, solve a strip problem: a semi-infinite strip truncated to 6 cells deep, one cell wide and periodic along the edge, with Dirichlet data -chi_j at the cut and natural BC at the far end. Its tail beta_j_inf is the mean of the far column. The interior correction eps u1bar has u1bar harmonic (sigma* isotropic), with boundary value sum_j beta_j_inf(edge) d_j u0 on each edge; only normal-derivative terms survive. Strips are cut from the fine medium next to each edge: the left edge uses a[0:6m, j0:j0+m]; the right edge is reversed in x1; the bottom and top edges are transposed. Scratch for s = 1/4: tails 0.13554 on the x1 = 0 edge (chi_1) and 0.16978 on x1 = 1; chi_2 tails on those edges 2e-17; max|u1bar| in the interior 0.02126; max_int |(u_eps - w_eps)/eps - u1bar| = 5.37e-3, 2.90e-3, 1.51e-3 for N = 8, 16, 32. This is the Allaire-Amar / Moskow-Vogelius tail mechanism. It replaces proposal A's second-order correctors, which cannot improve the interior error without second-order tails.

*Computations.* twoscale.strip_tail for the four edges and s in {0, 1/4, 1/2}; harmonic_dirichlet on the fine grid; comparison for N = 8, 16, 32.

*Checks.*

- tangential-corrector tails vanish. Reference: oddness along the edge. Target: |chi_2 tail| on x1-edges and |chi_1 tail| on x2-edges. Tolerance: 1e-12 (a priori).
- mirror cuts have zero tails. Reference: symmetry. Target: all tails for s = 0 and 1/2. Tolerance: 1e-12 (a priori).
- tail independent of strip depth. Reference: decay e^(-6 kappa_1), about 1e-18. Target: |tail(6 cells) - tail(8 cells)|. Tolerance: 1e-12 (a priori).
- tails predict the interior error. Reference: two-scale theory with tails. Target: max_int |(u_eps - w_eps)/eps - u1bar|: slope >= 0.8 over N = 8-32, and <= 0.1 max|u1bar| at N = 32 (scratch 7%). Tolerance: as stated (set after a convergence study or from the design scratch).

#### 5. How thick is the boundary layer? Static evanescent Bloch waves

Solutions e^(-kappa y1) p(y), with p periodic, of div(a grad u) = 0. On one Q1 cell, open in y1 and periodic in y2, with Bloch factor lambda = e^(-kappa) across the cell, this gives the QEP lambda^2 A1 + lambda A0 + A_-1 = 0. Here A0 = K_aa + S^T K_bb S, A1 = K_ab S, A_-1 = S^T K_ba, with S picking column 0. Linearize to a 2n x 2n generalized eigenproblem; keep |lambda| < 1 and |kappa| > 1e-6. Each mode has a parity under y2 -> 1 - y2; edge data excite one parity. Homogeneous cell: kappa_1 -> 2 pi with O(h^2) error. Scratch, m = 8 gives 6.63377, m = 16 gives 6.36553, error ratio 0.235. Heterogeneous cells at m = 16: a_in = 10: 6.8335, 7.3387; a_in = 100: 7.1569, 9.378; a_in = 0.1: 6.7474, 7.4113. All exceed the discrete homogeneous value, so the layers are thinner, which reverses proposal A's 'high-index inclusions slow it'. The layer falls by about e^-6.8, roughly 1e-3, per cell, which is below the O(eps^2) interior error after one cell. Proposal A's fit in the fine solve is therefore infeasible; the decay is measured on the strips.

*Computations.* twoscale.evanescent_exponents at m in {8, 16} (dense scipy.linalg.eig; m = 24 optional if time allows), contrasts a_in in {0.1, 10, 100} and homogeneous, with parity labels. Strip decay rate between depths 1 and 2 cells: L2 over the column of beta - beta_inf. s = 0 with chi_2 data excites the odd sector; s = 1/4 with chi_1 data excites the even sector.

*Checks.*

- homogeneous QEP converges at second order. Reference: Q1 dispersion. Target: |kappa_1(16) - 2 pi| / |kappa_1(8) - 2 pi| in [0.2, 0.3]. Tolerance: as stated (a priori).
- strip decay = QEP exponent of matching parity. Reference: same discrete operator. Target: relative difference, a_in = 10, both parities. Tolerance: 1e-2 (contamination from the next same-parity mode < 1e-3) (a priori).
- heterogeneity thins the layer (corrects proposal A). Reference: scratch. Target: kappa_1 > discrete homogeneous kappa_1 for a_in in {0.1, 10, 100}. Tolerance: strict (set after a convergence study or from the design scratch).

#### 6. Weak limits and the div-curl lemma

With phi_k = sin(k pi x) sin(k pi y), I_flux(eps) = int phi a_eps |grad u_eps|^2 -> int phi sigma* |grad u0|^2; this is the div-curl lemma (a divergence-controlled factor times a curl-free one). By contrast I_grad(eps) = int phi |grad u_eps|^2 -> int phi grad u0 . M grad u0 with M = <(I + grad chi)^T (I + grad chi)>, not int phi |grad u0|^2; both factors are curl-free and nothing controls the oscillation. Scratch: M_11 = M_22 = 1.50455; |I_grad - naive| = 1.565e-3 at N = 32, against the predicted 1.614e-3; |I_flux - lim| slope 1.12; |I_grad - lim_M| slope 1.22.

*Computations.* Element-centered gradients of the Q1 fine solution, phi at element centers, k in {1, 2}, s = 0, N = 4-32. M from exact Q1 element integrals of the correctors. Limits by quadrature of the sine-series gradients on the same centers.

*Checks.*

- div-curl: flux energy converges. Reference: div-curl lemma. Target: slope of |I_flux - lim| over N = 4-32 >= 0.8. Tolerance: 0.8 (a priori).
- gradient energy converges to the M-weighted limit. Reference: two-scale convergence. Target: slope of |I_grad - lim_M| >= 0.8. Tolerance: 0.8 (a priori).
- and misses the naive limit by the corrector term. Reference: int phi grad u0.(M - I) grad u0. Target: |I_grad - naive| / |predicted gap| in [0.9, 1.1] at N = 32 (scratch 0.97). Tolerance: 0.1 (a priori).

*Findings to test.*

- With a generic cut, the interior error of the first-order expansion is eps u1bar + O(eps^2), with u1bar built from boundary-layer tails; with mirror-line cuts the tails vanish and the error is O(eps^2).
- The textbook O(eps^1/2) H1 rate of w_eps needs a generic cut; mirror-line cuts give O(eps).
- The boundary-layer decay equals the smallest static evanescent Bloch exponent of the excited parity, and heterogeneity at R = 0.3 makes the layer thinner than in a homogeneous medium.
- The div-curl lemma: flux-weighted energies converge to the homogenized energy, plain gradient energies converge to the corrector-weighted limit.

*Risks and mitigations.*

- Fine solves at n = 512 take about 5 s each and there are 3 of them: fine. If over budget, drop N = 32 for s = 1/2.
- The QEP has spurious eigenvalues at lambda near 0 and infinity: filter by |lambda| < 1 - 1e-9 and |kappa| > 1e-6, and check the residual ||Q(lambda) v|| / ||v|| < 1e-8.
- Parity labels: compute <v, P v>/<v, v> on the eigenvector's first node column; it must be +-1 within 1e-8.
- All comparisons use the discrete cell; the staircase disk is stated, not hidden.

### 05 `05_band_edges_and_symmetry`: Band edges: high-frequency homogenization, little groups, Dirac cones and the envelope's boundary condition

**Question.** Near a band edge of a TM rod lattice, high-frequency homogenization replaces the crystal by an envelope equation whose effective tensor is fixed by the little group. Do the symmetry-allowed forms come out of exact band computations: the number of constants, the quadratic touching at M, the Dirac cone at K with its Berry phase? And does finite-size quantization come out too, including the envelope's boundary condition?

**Why.** Representation theory and topology meet homogenization at nonzero frequency. The number of k.p constants is a Hom-space dimension, Dirac points come from a one-dimensional Hom space at C3v, and the quantized Berry phase follows from PT symmetry. The 1D finite-stack test shows that the envelope's boundary condition depends on where the walls cut the cell, the high-frequency counterpart of notebook 04's tails.

Run-time budget: 150 s with 4 threads. Modules: bloch.py, ../spatial_dispersion/rodlattice.py.

#### 1. A plane-wave solver with exact symmetry

TM: -(grad + ik)^2 u = (omega/c)^2 eps u, a generalized Hermitian problem with B = [eps_hat(G - G')], positive definite. Plane waves are kept inside a circular cutoff |k0 + G| <= Gmax around a high-symmetry point k0, so the truncated problem keeps the little group and degeneracies are exact. The design scratch found that a parallelogram cutoff splits the honeycomb Dirac point by about 1e-6. Disk Fourier coefficients come from the J1 formula; ellipses use rho = sqrt((G_x r_x)^2 + (G_y r_y)^2). Lattices: square, R = 0.3a, eps = 12. Honeycomb: a1 = (1, 0), a2 = (1/2, sqrt(3)/2), rods at (a1 + a2)/3 and 2(a1 + a2)/3 (C6v about the origin), R = 0.2a, eps = 12. Frequencies are in a/lambda. Scratch, square lattice: X: 0.19199, 0.30085, 0.40771, 0.55371; M: 0.23052, 0.34941 (double), 0.51816; Gamma: 0, 0.40157 (double). Honeycomb: K: 0.26964 (double), 0.43737.

*Computations.* Gmax chosen for about 300, 600 and 1000 plane waves. Cross-check with ../spatial_dispersion/rodlattice.bloch_frequencies (TM, M = 5 multipoles) at X and M for the 4 lowest bands.

*Checks.*

- PWE vs multipole. Reference: rodlattice.bloch_frequencies (spatial_dispersion found PWE vs multipole within 3e-5). Target: relative difference at the finest Gmax. Tolerance: fixed after the convergence study: 3x the change between the two finest Gmax, capped at 2e-4 (set after a convergence study or from the design scratch).
- exact degeneracies. Reference: little group preserved by the circular cutoff. Target: |omega_2 - omega_3| at M (square) and |omega_1 - omega_2| at K (honeycomb). Tolerance: 1e-10 (a priori).

#### 2. Little groups and irreps

Symmetry operators act on coefficient vectors by permuting G, using g k0 = k0 + G0. Character projection labels each degenerate set: C4v at Gamma and M, C2v at X, C3v at K. Bands 2-3 at M form the E irrep.

*Computations.* Characters of bands 1-6 at Gamma, X, M (square) and 1-4 at K (honeycomb).

*Checks.*

- characters are irreducible characters. Reference: character tables of C4v, C2v, C3v. Target: traces on each degenerate set match one irrep's characters. Tolerance: 1e-8 (a priori).

#### 3. A non-degenerate edge: effective mass by perturbation theory

Band 1 at X is a saddle (maximum along Gamma-X, minimum along X-M). Its inverse-mass tensor in lambda = (omega/c)^2 units comes from H(k0 + q) = H0 + q.H1 + |q|^2 (U^H U-weighted) plus the second-order sum over all other bands, with B-orthonormal eigenvectors.

*Computations.* bloch.kp_second_order at X, band 1, compared with central differences of the PWE band at |q| a = 1e-3 and 2e-3, Richardson-combined.

*Checks.*

- perturbation theory vs finite differences. Reference: same truncated basis; FD error O(q^2). Target: relative difference of the tensor entries. Tolerance: 1e-6 (a priori).

#### 4. The degenerate edge at M: three constants and a quadratic touching

dim Hom_C4v(Sym^2 V, End E) = 3 and dim Hom_C4v(V, End E) = 0, computed from characters. So the Loewdin 2x2 Hamiltonian is lambda0 + A q^2 + B(q_x^2 - q_y^2) sigma_z + C q_x q_y sigma_x, in the symmetry-adapted basis, with no linear term. Fit-free prediction: omega^2_+- = omega0^2 + A q^2 +- q^2 sqrt(B^2 cos^2 2phi + (C^2/4) sin^2 2phi), using whichever normalization the code defines consistently. The pseudospin, the Bloch vector of the projected 2-vector, winds twice around M.

*Computations.* bloch.kp_second_order on bands 2-3 at M. PWE at |q| a in {0.02, 0.04, 0.08}, 24 angles. Winding of the upper band's pseudospin on a circle with |q| a = 0.04, 64 points.

*Checks.*

- Hom dimensions. Reference: character inner products. Target: 3 and 0 as computed integers. Tolerance: 1e-12 (a priori).
- no linear term, three-parameter form. Reference: symmetry. Target: ||H1 restricted to the doublet|| and the residual after projecting H2 onto the allowed basis. Tolerance: 1e-10 (a priori).
- k.p predicts the bands. Reference: PWE. Target: relative error of omega^2 - omega0^2 at |q| a = 0.04 <= 1e-2, and error ratio between |q| a = 0.08 and 0.04 in [3, 5]. Tolerance: as stated (a priori).
- winding number. Reference: quadratic form. Target: |winding| == 2. Tolerance: 0.05 (a priori).

#### 5. Breaking C4: elliptical rods

Area-preserving ellipses with semi-axes R(1+e)^(1/2) and R(1+e)^(-1/2) along x and y reduce the symmetry to C2v and split the doublet linearly in e. The slope comes from first-order degenerate perturbation theory with dB/de, in the generalized-problem form.

*Computations.* e in {0.005, 0.01, 0.02, 0.04}.

*Checks.*

- first-order slope. Reference: degenerate perturbation theory. Target: relative difference between PT slope and PWE splitting/e at e = 0.005. Tolerance: 1e-2 (a priori).
- linearity. Reference: first order. Target: splitting(0.01)/splitting(0.005) in [1.95, 2.05]. Tolerance: as stated (a priori).

#### 6. Dirac cone at K and its Berry phase

dim Hom_C3v(V, End E) = 1, which gives one Dirac velocity v. Equal rods are PT-symmetric, so the Berry phase on a loop around K is pi. Unequal radii (0.21, 0.19) break inversion: m is half the omega^2 splitting at K, and the loop phase of the lower band is pi(1 - m/sqrt(m^2 + v^2 r^2)) in magnitude, with the opposite sign at K'. Scratch: Dirac frequency 0.26964; splitting/q about 0.055 in (a/lambda) per (1/a); unequal-rod gap 0.0124.

*Computations.* Wilson-loop products of B-overlaps of cell-periodic coefficient vectors on a fixed G set (circular cutoff around K), 64 points. Loop radii r a in {0.01, 0.02, 0.04} for equal rods, and r in {m/v, 2m/v, 3m/v} for unequal rods, at K and K'. v from degenerate first-order k.p, against the PWE slope at |q| a = 0.002.

*Checks.*

- Hom dimension at K. Reference: characters. Target: 1. Tolerance: 1e-12 (a priori).
- Dirac velocity. Reference: PWE slope. Target: relative difference. Tolerance: 1e-3 (O(q) correction) (a priori).
- quantized Berry phase. Reference: PT symmetry. Target: ||phase| - pi| for equal rods. Tolerance: 1e-4 (a priori).
- massive Dirac phase. Reference: pi(1 - m/sqrt(m^2 + v^2 r^2)). Target: | |phase| - formula | for the three radii (to test; trigonal warping and O(r^2) terms neglected). Tolerance: 0.03 pi (set after a convergence study or from the design scratch).
- valley contrast. Reference: time reversal maps K to K' and flips the curvature. Target: |phase(K) + phase(K')| mod 2 pi. Tolerance: 1e-6 (a priori).

#### 7. Finite stacks: the envelope equation and its boundary condition (1D, exact)

Symmetric cell [B 0.35a | A 0.3a | B 0.35a] with eps_A = 12, eps_B = 1, so the cell boundary is a mirror plane. With walls (u = 0) at cell boundaries, u = psi_k(x) - psi_k(-x) vanishes at 0, and u(Na) = 2i sin(kNa) psi_k(0). All in-band modes therefore sit exactly at k = n pi/(Na). The HFH envelope prediction omega_j^2 = omega_edge^2 + T (j pi/(Na))^2, with T = (1/2) d^2(omega^2)/dk^2 at the edge, is then exact up to the band's quartic term, O(N^-4). The edge mode that is odd about the wall fits the walls itself (j = 0, Neumann-like envelope); the even one does not (Dirichlet-like). With walls shifted by a/4, which is not a mirror plane, an end correction delta = O(a) enters: omega^2 = omega_edge^2 + T (j pi/(Na + 2 delta))^2, and setting delta = 0 leaves an O(N^-3) error.

*Computations.* 2x2 transfer matrices for (u, u'). Exact dispersion cos(ka) = Tr M(omega)/2. Finite-stack roots of u(x_w + Na) from a 2000-point omega scan plus brentq. N in {5, 10, 20, 40}, j = 1, 2, 3, both edges of the first gap. delta fitted at N = 20 for the a/4 walls.

*Checks.*

- exact quantization. Reference: theorem above. Target: Bloch k of each computed mode equals (N - j) pi/(Na). Tolerance: 1e-10 (a priori).
- envelope error O(N^-4) for mirror walls. Reference: quartic band term. Target: slope of |omega_j^2 - prediction| over N = 10-40 within 0.3 of -4. Tolerance: 0.3 (a priori).
- exactly one edge has a j = 0 mode. Reference: parity of the edge standing waves. Target: one edge frequency is a finite-stack root. Tolerance: 1e-10 (a priori).
- generic walls need an end correction. Reference: derivation. Target: delta = 0 error slope within 0.3 of -3; the fitted delta cuts the error at N = 10 and 40 by >= 10x (to test). Tolerance: as stated (set after a convergence study or from the design scratch).

*Findings to test.*

- Bands 2-3 at M of the eps = 12, R = 0.3a TM lattice are an E doublet with a 3-constant quadratic touching and pseudospin winding 2.
- The honeycomb (R = 0.2a, eps = 12) has a Dirac point at K between bands 1 and 2 with Berry phase pi; unequal rods give the massive-Dirac loop phase and valley contrast.
- Circular plane-wave cutoffs preserve degeneracies; parallelogram cutoffs split them at about 1e-6.
- Finite stacks: walls on mirror planes give exact Bloch quantization (envelope error N^-4), generic walls need an O(a) end correction (error N^-3 without it).

*Risks and mitigations.*

- Eigenvector gauge in degenerate subspaces: always work with projectors or with symmetry-adapted bases from the character projection.
- Wilson loops need a k-independent G set around the loop center: build it once per loop.
- Runtime: about 400 eigensolves at 400-600 plane waves. If over budget, cut loops to 32 points and the finest Gmax to about 800 plane waves.
- rodlattice.bloch_frequencies scans a mode condition: pass fmax and nscan large enough to find 4 bands, and match bands by nearest frequency.

### 06 `06_what_scattering_sees`: What TM scattering sees of a microstructure: the average, the exact second-order term, the termination, and the detectability law

**Question.** In the TM testbed (lambda0 = 1, ch15 rings, 30 dB), what part of a subwavelength microstructure survives in the scattering data? How does the detectability law of DESIGN section 15 sort it into what biases a homogeneous-object estimate of the composition and what flags the model as wrong?

**Why.** This connects homogenization to the project's Bayesian imaging: which microstructure information is identifiable, which is absorbed into biased parameters, and when the misfit reveals model error. It uses exact references (multilayer Mie, PWE) and the testbed's own detectability machinery, and generalizes the D2 selection rule of DESIGN 15.2 to rotations.

Run-time budget: 170 s with 4 threads. Modules: scatter.py, bloch.py, bayes_scatter (geometry, forward, likelihood, detectability, mie, phantoms).

#### 1. TM homogenization: an average plus an exact second-order term

E_z is tangential and continuous, so the leading order of any z-invariant composite is <eps>. Expanding the Bloch problem -(grad + iK)^2 p = k0^2 eps p gives eps_eff(k0) = <eps> + k0^2 sum_{G != 0} |eps_G|^2/|G|^2 + O(k0^4). This is exact in contrast, isotropic and positive (derived in the notebook). In 1D it is Rytov's ordinary-wave (pi^2/3)(a/lambda)^2 f^2 (1-f)^2 Delta eps^2. For R = 0.3a the full sum is 1.2874 (eps = 4) and 17.309 (eps = 12) in (a/lambda)^2; the first G-shell alone gives 1.094 for eps = 4, so proposal C's 'within 5%' hand check fails.

*Computations.* Rytov closed form vs the 1D sum (1e5 terms plus the n^-4 tail). bloch.py PWE (square lattice, about 600 plane waves) with K along Gamma-X, K a in {0.05, 0.1, 0.2, 0.4}: eps_eff = K^2/k0^2 against the truncated sum, and (eps_eff - eps_2nd)/k0^4. Scratch: 3.587e-3 and 3.593e-3 (eps = 4, K a = 0.1, 0.2); 0.1222 and 0.1226 (eps = 12).

*Checks.*

- 1D sum equals Rytov. Reference: closed form. Target: relative difference. Tolerance: 1e-10 (a priori).
- residual is O(k0^4). Reference: next order of the expansion. Target: ratio of (eps_eff - eps_2nd)/k0^4 at K a = 0.1 and 0.2, both eps, within 5% of 1. Tolerance: 5% (a priori).

#### 2. The Born cutoff

Ring-aperture Born data sample the object spectrum only for |q| <= 2 k0 (Ewald disk). A lattice with p < lambda/2 therefore has no first-order signature in the bulk; at first order a finite object's microstructure shows only through its boundary.

*Computations.* Testbed n = 96, side 2.5, Forward.born on chi = 0.05 w(x) cos(2 pi kappa_g x), w = exp(-r^2/(2 * 0.5^2)), kappa_g (cycles per lambda) in {0.25, 0.5, ..., 4}. Gain = ||born data||/||chi||. Testbed timing in the design scratch: about 0.4 s per n = 96 forward.

*Checks.*

- knee at 2 cycles per wavelength. Reference: Ewald disk with Gaussian-window leakage. Target: gain(3)/gain(1.5). Tolerance: <= 1e-2 (a priori).

#### 3. Concentric rings: what a homogeneous fit absorbs (exact)

Truth: a disk R_D = 1 of concentric rings of period p with eps1 = 3 and eps2 = 1.5 (thickness fraction 1/2). Two terminations: (i) period cut, with core eps1 and outermost eps2 ending at R_D; (ii) mid-layer cut, with the outermost eps2 layer of half thickness. Exact data come from multilayer Mie (cloaking nb04's validated code, copied into scatter.py). Model: homogeneous disk (eps, R) by exact Mie. Noise at 30 dB of the base-disk data; whitening as in detectability.stack_real. Born-order integration by parts plus TM continuity give three predictions. The termination acts at O(p) exactly as a radius shift dR = X0(R_D)/chibar, where X0 is the zero-mean antiderivative of chi - chibar; that is -Delta chi f(1-f) p/(2 chibar) = -0.15 p for (i) and 0 for (ii). The bulk dispersion is O(p^2) and absorbed by eps. The normal residual is O(p^2), so DeltaPhi ~ p^4, z_eps ~ p^2 and c = DeltaPhi/z^2 is constant. Scratch: dR_fit/dR_pred = 0.86, 0.91, 0.95, 0.973 for p = 0.2-0.025; fitted d_eps/Rytov = 0.93, 0.81, 0.77 (i) and 0.86, 0.80, 0.79 (ii). So the fit does not isolate the bulk correction. For (ii), the first-order analysis at the base equals the nonlinear fit: z = 3.51 vs 3.50 and DeltaPhi = 18.40 vs 18.56 at p = 0.05, with alpha -> 18.0 degrees. For (i), linearizing at the base point fails (DeltaPhi_1st/DeltaPhi_NL about 100), so the profile foot point of DESIGN 15.1 is used: eps fixed at epsbar = 2.25, R at its least-squares value.

*Computations.* n_p in {5, 10, 20, 40, 80} (p = 0.2-0.0125), both cuts. For each: raw error vs the (epsbar, R_D) disk; R-only fit; (eps, R) Gauss-Newton fit (FD Jacobian, step 1e-6). detectability.analyze_error at the base point (ii) or the foot point (i), giving z_eps, DeltaPhi, alpha, beta, c_geom, c_direct. Windows at 30 dB from log-log fits: p_z (|z_eps| = 3), p_flag (DeltaPhi = 147 nats, the 3-sigma chi^2 flag for 2400 complex data) and p_3nat (DeltaPhi = 3). Scratch for (ii): 0.046, 0.084 and 0.032.

*Checks.*

- multilayer code vs testbed Mie. Reference: bayes_scatter.mie.mie_scattered. Target: max relative difference for a homogeneous disk (scratch 2e-15); M = 60 vs 90 orders. Tolerance: 1e-12 (a priori).
- raw error slopes. Reference: Born-order derivation. Target: slope 1 +- 0.1 for (i) and 2 +- 0.15 for (ii), over n_p = 10-80. Tolerance: as stated (a priori).
- termination = radius shift. Reference: dR = X0(R_D)/chibar. Target: |dR_fit/dR_pred - 1| <= 0.05 at p = 0.025 and decreasing with p, case (i). Tolerance: 0.05 (a priori).
- law identity. Reference: analyze_error. Target: |c_geom - c_direct|/c_direct. Tolerance: 1e-10 (a priori).
- first order vs nonlinear fit (base or foot point). Reference: DESIGN 15.1 conditions. Target: for p <= 0.05: DeltaPhi within 10% and z within 5% of the noise-free nonlinear fit (to test; scratch 1% for (ii)). Tolerance: as stated (set after a convergence study or from the design scratch).
- scaling of bias and misfit. Reference: derivation. Target: slope of z_eps 2 +- 0.15 and of DeltaPhi 4 +- 0.3, case (ii). Tolerance: as stated (a priori).
- detectability ordering (refines proposal C). Reference: scratch (0.032, 0.046, 0.084). Target: p_3nat < p_z < p_flag for case (ii). Tolerance: strict (set after a convergence study or from the design scratch).

#### 4. A lattice in the testbed: the rotational selection rule

The ch15 acquisition is invariant under rotation by 18 degrees (C20: transmitter index +2, receiver index +3). The real isotypic projectors are Q_k = (1/20) sum_j c_k cos(2 pi j k/20) R^j, with c_0 = c_10 = 1 and c_k = 2 otherwise, for k = 0-10. The exact Mie model of a centered disk has its eps and R tangents in Q_0 and its center tangents in Q_1. A lattice-filled disk centered on a rod, painted on an even grid centered at the origin, is C4-invariant, so its data lie in Q_0 + Q_4 + Q_8. The Q_4 + Q_8 part of a model error is orthogonal to every model tangent: it raises DeltaPhi and never biases the estimate. This generalizes DESIGN 15.2 from reflections to rotations.

*Computations.* Square lattice of rods (eps_rod = 4, radius 0.3p, centers within R_D = 1, one rod at the origin), p in {0.5, 0.33, 0.25}. Testbed n = 128, side 2.5 (h = 0.0195), coverage ss = 8, tol 1e-10; scratch timing about 0.8 s per n = 128 forward. Model: exact Mie disk (eps, R, cx, cy) with base epsbar = area average of the rods over the disk and R = R_D. delta at the base, its Q_k fractions, z, and DeltaPhi split into the Q_0 part and the Q_4 + Q_8 part. Grid-error floor: testbed vs Mie for the homogeneous base disk.

*Checks.*

- model tangents in the predicted sectors. Reference: rotational symmetry of the Mie model and of the acquisition. Target: ||(I - Q_0) d_eps||, ||(I - Q_0) d_R|| relative and ||(I - Q_1) d_cx|| relative. Tolerance: 1e-12 (a priori).
- lattice data are C4-invariant. Reference: exact grid and lattice symmetry. Target: ||(I - Q_0 - Q_4 - Q_8) d_lattice||/||d_lattice||. Tolerance: 1e-8 (solver tolerance 1e-10) (a priori).
- selection rule. Reference: w_hat lies in Q_0. Target: |z(delta) - z(Q_0 delta)|/|z(delta)|. Tolerance: 1e-10 (a priori).
- signal above the grid floor. Reference: testbed vs Mie for the base disk. Target: ||delta|| > 3 x floor for p in {0.5, 0.33}; p = 0.25 reported. Tolerance: factor 3 (set after a convergence study or from the design scratch).

#### 5. Approximation-error model for the composition (rings, exact)

Kaipio-Somersalo: treat the termination as random. Estimate the mean and covariance of delta from an ensemble, and use them as a shifted mean and extra noise covariance. Coverage of the 90% interval for epsbar = 2.25 (the composition) is computed from the linearized estimator, without sampling.

*Computations.* p = 0.07; 64 ring structures with the cut uniformly random within the outermost period (core adjusted to keep R_D). Split 32 for estimation and 32 held out. Woodbury for (I + C_delta)^-1 in whitened data. Naive coverage = mean over held-out cases of P(|N(z_i, 1)| <= 1.645); AEM coverage likewise with the AEM mean and sd.

*Checks.*

- naive model undercovers. Reference: bias of several sd. Target: naive coverage < 0.8 (to test). Tolerance: as stated (set after a convergence study or from the design scratch).
- AEM restores coverage. Reference: Kaipio-Somersalo. Target: held-out AEM coverage in [0.80, 0.97] (to test). Tolerance: as stated (set after a convergence study or from the design scratch).

*Findings to test.*

- TM sees <eps> at leading order, and the bulk microstructure only through an exact-in-contrast k0^2 term (PWE residual O(k0^4)).
- Sub-lambda/2 periodicity is invisible at Born order except through the boundary.
- The termination of a ring microstructure is exactly a radius shift at O(p) and is absorbed by the fit; the residual's normal part is O(p^2) at a constant angle.
- A homogeneous fit does not recover the bulk effective eps even as p -> 0 (ratio about 0.78 in scratch).
- For the termination-free rings, the composition estimate is biased by more than 3 sd while passing the 147-nat goodness-of-fit flag for p between about 0.046 and 0.084; against a 3-nat evidence test the error is caught first.
- Linearizing at the base point fails for the period cut; the profile foot point is required (DESIGN 15.1).
- C4-invariant lattice errors in the Q_4 and Q_8 sectors only raise the misfit and never bias.
- An approximation-error model restores coverage for the composition.

*Risks and mitigations.*

- At p = 0.25 testbed rods are about 4 pixels in radius: the lattice truth is the testbed's lattice, and the floor is printed.
- GMRES at eps_rod = 4: restart 100, maxiter 30, tol 1e-10; assert the residuals.
- Sign conventions: z = (fitted - base)/sigma as in detectability.analyze_error; state this.
- The 147-nat flag is for 2400 complex data (sd of Phi = sqrt(2400), about 49); recompute it from the data size, do not hard-code it.
- If over budget, drop section 5 first, then p = 0.25 in section 4.

## Modules

### `cellsolver.py`

Periodic and window cell problems on pixel grids: Q1 primal/dual (certified Loewner brackets), the sampled-coefficient spectral (Moulinec-Suquet) scheme by PCG, and Fourier-Galerkin with exact Fourier coefficients for analytic shapes (rigorous bounds on the true geometry). Plus geometry generators. Start from the design prototypes q1.py, ms.py and fg.py in <design scratch>/; these may be deleted by cleanup, so copy before relying on them.

Functions:

- q1_stiffness(a) -> (K csr (N^2,N^2), conn int (N^2,4)): periodic unit cell, a[i,j] with i the x index, node (i,j) at (i/N, j/N) with id i*N + j; element nodes (i,j), (i+1,j), (i+1,j+1), (i,j+1); element matrix (1/6)[[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]] * a_e
- q1_effective(a, return_fields=False) -> U (2,2) [, chi (2,N,N)]: loads f_i = sum_e a_e h g_i (g_x = (-1/2, 1/2, 1/2, -1/2), g_y = (-1/2, -1/2, 1/2, 1/2)); pin node 0; one splu for both loads; U_ij = <a> delta_ij + f_j . chi_i with K chi_i = -f_i. Contract: symmetric to 1e-12, U >= sigma* (Loewner), exact for grid-aligned laminates.
- q1_dual(a) -> L = inv(R.T @ q1_effective(1/a) @ R), R = [[0,-1],[1,0]]; q1_bracket(a) -> (L, U)
- q1_energy_density(a, chi) -> E (2,N,N): E_e = h^2 + 2h g.chi_e + chi_e^T K_E chi_e per element (dU_ii/da_e). Contract: sum_e a_e E_e = U_ii to 1e-12.
- q1_window(a, bc) with bc in {'affine', 'flux'} -> (2,2): non-periodic window, (N+1)^2 nodes; affine = fluctuation 0 on boundary nodes; flux = inv(R^T U_affine(1/a) R). Contract: U_aff >= U_per >= L_per >= L_flux.
- spectral_sampled(a, tol=1e-11, maxit=5000) -> (S (2,2), iterations): potential PCG with spectral derivatives (Nyquist components of k set to 0; zero-symbol modes excluded from the preconditioner) and preconditioner (-mean(a) Laplacian)^-1
- fg_coefficients(shape, K2, **p) -> (4K+1, 4K+1) complex: 'disk' (R, s_in, s_out, center=(.5,.5)): s_out delta + (s_in - s_out) f 2J1(|k|R)/(|k|R) e^(-ik.c); 'square' (side w, s_in, s_out): (s_in - s_out) w^2 sinc(k_x w/2) sinc(k_y w/2) e^(-ik.c); 'checkerboard' (s_a, s_b), tiles of side 1/2 with s_a on [0,1/2)^2 and [1/2,1)^2; 'laminate' (f, s_in, s_out, axis)
- fg_bracket(shape, K, tol=1e-12, maxit=3000, **p) -> (L, U, info): trig trial fields of degree K per coordinate; matvec on an M = 4K + 2 zero-padded grid with sigma truncated at degree 2K; primal with sigma coefficients, dual with 1/sigma coefficients (same shape, reciprocal phase values) in the swapped direction; diagonal preconditioner 1/(mean * |k|^2). Contract: L <= sigma* <= U, monotone in K, exact in the direction parallel to laminate layers.
- generators: checkerboard(N, a, b), disk_pixels(N, R, a_in, a_out, center=(.5,.5)), square_pixels(N, side_px, a_in, a_out), laminate_pixels(N, n_in, a_in, a_out, axis), random_checkerboard(L, m, p, a, b, rng), gaussian_threshold(N, ell, theta, rng, values)

Validation:

- Laminate N = 64, values {1, 10}: Q1 L = U = harmonic and arithmetic means to 1e-12.
- Discrete Dykhne: checkerboard kappa = 10, N = 32 and 64: |L U/ab - 1| <= 1e-12.
- Random field N = 16: q1_energy_density vs central FD (step 1e-6) on 3 pixels, 1e-6 relative.
- FG disk (0.5, 10) at K = 16: bracket contains 2.4154673802 and equals the scratch values [2.408174, 2.494377] to 1e-5.
- spectral_sampled checkerboard kappa = 10, N = 64: relative error +2.5e-3 +- 2e-4 (scratch), at most 60 iterations.
- q1_window ordering on one random window.

### `bounds.py`

Closed forms and bound tests: Wiener, 2D Hashin-Shtrikman, Maxwell Garnett, Bruggeman, sequential laminates, the 2D G-closure trace bounds with certified violation tests, and Bergman-Milton support functions (f-only by exact 1-D maximization, isotropic by LP).

Functions:

- wiener(s1, s2, f1) -> (harmonic, arithmetic)
- hs2d(s1, s2, f1) -> (lo, hi) = (s_min + f_max/(1/(s_max - s_min) + f_min/(2 s_min)), the mirror)
- maxwell_garnett2d(s_inc, s_host, f_inc); bruggeman2d(s1, s2, f1) (explicit positive root of the quadratic)
- laminate_seq(sA, sB, thetaB, dirs, weights) -> (2,2): thetaB inv(A - sA I) = (sB - sA)^-1 I + (thetaA/sA) sum_k m_k e_k e_k^T
- trace_slacks(l1, l2, s1, s2, f1) -> (lower_slack, upper_slack): RHS minus LHS of the two trace inequalities, >= 0 inside
- certified_violation(L, U, s1, s2, f1) -> bool: uses eigenvalues of U for the lower trace bound and the Wiener lower bound, and of L for the upper ones
- bm_support_f(s, f, phis) -> support values (bounded scalar maximization over z in [0,1] after a 4001-point scan)
- bm_support_iso(s, f, phis, nz=4001) -> (support values, extremal weights): linprog(method='highs-ds') with constraints sum mu = f, sum z mu = f(1-f)/2

Validation:

- HS(1, 10, 0.5) = (2.384615..., 4.193548...); HS(1, 10, 0.7 weak) = (1.65060, 2.71676) to 1e-5.
- Rank-1 laminate = Wiener means; isotropic rank-2 = HS (1e-14).
- Rank-2 families lie on the trace curves (slack 1e-12) for 101 values of m.
- MG = HS lower when the matrix is the weaker phase (1e-14).
- LP extremal solutions have at most 2 nonzero weights.

### `rayleigh.py`

Static Rayleigh multipole method for the square array of disks: lattice sums, the real symmetric Rayleigh matrix W, the effective conductivity at real or complex contrast, and the exact spectral measure. The formulas were derived and checked in this design (prototype rayleigh.py in the design scratch).

Functions:

- square_lattice_sums(nmax) -> dict n -> S_n: S_4 = Gamma(1/4)^8/(960 pi^2) = 3.1512120021539; S_6 = 0; for k >= 4, (2k+1)(k-3)(2k-1) S_2k = 3 sum_{j=2}^{k-2} (2j-1)(2k-2j-1) S_2j S_(2k-2j); S_n = 0 unless n = 0 mod 4
- rayleigh_matrix(f, L) -> W (L,L): odd l, m in 1..2L-1, W_lm = sqrt(l/m) R^(l+m) [pi delta_l1 delta_m1 + C(l+m-1, l) S_(l+m) 1(l+m >= 4)], R = sqrt(f/pi), unit period
- sigma_eff(f, kappa, L=20) -> 1 + 2f [(I/beta - W)^-1]_11, beta = (kappa-1)/(kappa+1); kappa may be complex
- rayleigh_measure(f, L=20) -> (z, w): z = (1 - mu)/2, w = f V[0,:]^2 from eigh(W)

Validation:

- S_4 closed form vs a brute-force sum over |p|, |q| <= 400 (origin excluded; Richardson in the cutoff) to 1e-6; S_8 = (3/7) S_4^2.
- L = 1 gives Maxwell Garnett exactly.
- f = 0.282743, kappa = 12 vs spatial_dispersion's series 1.6301245119 to 1e-7.
- f = 0.5, kappa = 10 gives 2.4154673802 to 1e-9.
- m0 = f and m1 = f(1-f)/2 to 1e-14 for L in {1, 2, 20}.

### `spectral.py`

Discrete spectral (Bergman-Milton) measures of the three discretizations, with measure utilities and closed forms.

Functions:

- q1_measure(chi1, direction=0) -> (z, w): dense scipy.linalg.eigh(K1, K) on the pinned periodic Q1 space; weights w_k/t_k at t_k plus an atom at 0 of weight f - sum
- fg_measure(shape, K, direction=0, **p) -> (z, w): Hermitian eigenproblem of K0^-1/2 K1 K0^-1/2 on trig fields of degree K, same partial fractions
- lanczos_measure(chi1, direction=0, steps=150) -> (z, w): T = chi1 Gamma chi1 with the spectral Gamma of cellsolver.spectral_sampled, full reorthogonalization, start u = chi1 e with mean inner product, Gauss weights f * (first eigenvector components)^2
- stieltjes(z, w, s); eps_ratio_from_measure(z, w, h) = 1 - F(1/(1-h)); moments(z, w, kmax); cdf(z, w, grid); wasserstein1(z1, w1, z2, w2); n_atoms_for(z, w, frac)
- drude_s(omega, omega_p, gamma) = omega (omega + i gamma)/omega_p^2; checkerboard_density(z), checkerboard_cdf(z)

Validation:

- Each measure reproduces its direct solver (q1_effective, fg_bracket upper, spectral_sampled) at h in {0.1, 10} to 1e-10 (Lanczos 1e-8).
- m0 = f exactly for all three.
- Checkerboard: Q1 atom at 0 at N = 16 equals 0.1978 (scratch) to 1e-3, and the Lanczos m1 = 1/8 to 1e-12 at N = 64.

### `twoscale.py`

Finite-domain two-scale computations: Q1 cell correctors with nodal values, fine Dirichlet solves on the unit square, the torsion series, boundary-layer strip problems and tails, harmonic extensions, static evanescent Bloch exponents with parity, error norms and weak-limit integrals. Prototypes: twoscale.py, tail.py, qep.py, rates.py and rates2.py in the design scratch.

Functions:

- cell_correctors(cell) -> (chi (2,m,m) nodal, zero mean, sigma (2,2)) using the cellsolver Q1 conventions
- fine_dirichlet(a, rhs=1.0) -> u ((n+1),(n+1)) nodal: Q1 on [0,1]^2, u = 0 on the boundary, consistent load h^2 per interior node
- torsion(x, y, nterms=301) -> (u, u_x, u_y): odd p, q, coefficients 16/(pi^4 p q (p^2+q^2))
- strip_tail(strip_a, data) -> (U, tail): pixels (L m, m), periodic in y, Dirichlet data on node column 0, natural BC at the far end; tail = mean of the last column
- harmonic_dirichlet(g) -> u: Laplace on the fine grid with boundary values from g
- evanescent_exponents(cell, nkeep=8) -> (kappa, parity): QEP lambda^2 A1 + lambda A0 + A_-1 = 0, A0 = K_aa + S^T K_bb S, A1 = K_ab S, A_-1 = S^T K_ba; keep |lambda| < 1 - 1e-9 and |kappa| > 1e-6; parity from the y2 mirror on the first node column
- norms(e, a=None) -> L_inf, L2 and H1 seminorms (unit stiffness, or the masked interior stiffness); weak_integrals(u, a, phi)

Validation:

- a = 1: max|u_h - u0| ratio between n = 128 and 256 in [3, 5].
- sigma from cell_correctors equals cellsolver.q1_effective on the same cell to 1e-12; scratch 1.688269 for R = 0.3, a_in = 10, m = 16.
- Homogeneous QEP: kappa_1 = 6.36553 at m = 16 (scratch) to 1e-4.
- Tail symmetry zeros at s = 0 and 1/2 to 1e-12.

### `bloch.py`

TM plane-wave Bloch solver for 2D lattices of circular or elliptical rods with symmetry-preserving circular cutoffs; little-group tools; degenerate and non-degenerate k.p by Loewdin perturbation in the generalized problem; Wilson-loop Berry phases; 1D transfer matrices for finite stacks. Prototypes: pwe.py and hc.py in the design scratch.

Functions:

- Lattice.square(a=1), Lattice.honeycomb(a=1): a1, a2, b1, b2, cell area; high-symmetry points
- gvectors(lat, gmax, center) -> (G (n,2), (i,j) indices) with |center + G| <= gmax
- eps_hat(lat, rods, dG): rods = list of dicts {center, rx, ry, eps}, background 1; ellipse form via rho = sqrt((G_x rx)^2 + (G_y ry)^2)
- tm_solve(lat, rods, k, G, nbands) -> (f = k0 a/2pi, C B-orthonormal (n, nbands), B)
- point_group_ops(lat, name) and rep_matrices(C, G, k0, ops) -> characters on degenerate sets
- kp_second_order(lat, rods, k0, G, bands) -> (H1 (2,d,d), H2 (2,2,d,d)) in lambda = (omega/c)^2 units, sum over all other computed states (full eigh of the truncated problem)
- berry_phase(lat, rods, center, radius, band, nk=64, gmax) -> phase in (-pi, pi] (fixed G set); pseudospin_winding(...)
- transfer_1d(layers, omega), bloch_k_1d(layers, omega), finite_stack_modes(layers, N, wall_offset, omega_window)

Validation:

- Square lattice R = 0.3, eps = 12 at M: bands 2 and 3 degenerate to 1e-10, value 0.34941 +- 1e-4 (scratch).
- Against rodlattice.bloch_frequencies at X: 0.19199 +- 2e-4.
- Honeycomb R = 0.2, eps = 12 at K: bands 1 and 2 degenerate to 1e-10 with the circular cutoff, value 0.26964 +- 1e-4.
- 1D dispersion cos(ka) = cos(kA dA) cos(kB dB) - (1/2)(nA/nB + nB/nA) sin sin against transfer_1d to 1e-12.

### `scatter.py`

Bridge to the testbed: exact multilayer Mie for concentric rings (copied with attribution from notebooks/cloaking/04, cells 4 and 32, where it was validated against bayes_scatter.mie), ring geometries, Rytov and TM second-order formulas, homogeneous-disk fits, foot points, C20 isotypic projectors of the ch15 acquisition, the lattice-disk painter, and the Born grating gain. Prototypes: rings.py and rings2.py in the design scratch.

Functions:

- log_ratios, multilayer(radii, eps, k0=2pi, M=None), exact_data(tx, rx, radii, eps, M=60) (copied)
- rings(R_D, n_p, f, eps1, eps2, cut in {'period', 'mid', float phase}) -> (radii, eps)
- rytov_tm(f, eps1, eps2, p, k0); tm_second_order_sum(lat, rods, gmax) (uses bloch.eps_hat)
- whiten(d, s) -> sqrt(2)/s [Re; Im] (the detectability.stack_real convention); disk_data(eps, R, center, tx, rx)
- fit_disk(d, tx, rx, p0, s, free=('eps','R'), iters=10) -> (p, D whitened at p, residual); foot_point_R(d, tx, rx, eps_fixed, R0, s)
- c20_projector(k) -> callable on (40,60) data: (1/20) sum_j c_k cos(2 pi j k/20) roll(d, (2j, 3j)), c_0 = c_10 = 1, else 2
- lattice_disk_chi(grid, R_D, p, rod_frac=0.3, eps_rod=4.0, ss=8) via phantoms.coverage and paint
- born_grating_gain(forward, kappa_g, width=0.5, amp=0.05)

Validation:

- exact_data vs bayes_scatter.mie.mie_scattered for a homogeneous disk: 1e-12 (scratch 2e-15).
- M = 60 vs 90 orders: 1e-12.
- Projectors: sum_k Q_k = I and Q_k Q_l = delta_kl Q_k on random data, 1e-12; homogeneous centered-disk data in Q_0, 1e-12.
- fit_disk recovers a homogeneous disk's own (eps, R) from noise-free data to 1e-9.

### `_build/selftest_modules.py`

Runs every module's validation list (under 60 s with 4 threads) and exits nonzero on any failure. Run it before building any notebook and after any module change. The existing _build/setup_cell.py (thread caps, figure style, check, check_in) is the shared first cell and must not be edited. check() uses absolute differences, so relative tests pass the relative error as the value with target 0.

Functions:

- main(): imports each module, runs test_<module>() functions, prints PASS/FAIL lines, sys.exit(1) on failure

Validation:

- Must pass before each notebook build; the notebooks repeat the essential validations with check().

## Build order

- 0. Copy the design prototypes from <design scratch>/ (q1.py, ms.py, fg.py, spec.py, rayleigh.py, pwe.py, hc.py, twoscale.py, tail.py, qep.py, rates.py, rates2.py, rings.py, rings2.py, stj.py, tb.py, t4.py) to a private working copy before they are cleaned up. They encode every number quoted in this plan. Re-validate rather than trust them.
- 1. cellsolver.py, bounds.py, rayleigh.py, and _build/selftest_modules.py with their tests; run the selftest.
- 2. nb01 (_build/nb01.py; percent format; the first cell is the line SETUP). Build with notebooks/homogenization/_build/build.sh nb01 01_cell_problem_duality and inspect with _build/show_nb.py. Record the run time and cut per the notebook's risk list if over 150 s.
- 3. spectral.py (+ tests), then nb02.
- 4. nb03 (uses cellsolver, bounds, rayleigh; JAX only for the autodiff check).
- 5. twoscale.py (+ tests), then nb04. It can be built in parallel with steps 3-4 once step 1 is done.
- 6. bloch.py (+ tests), then nb05. Can run in parallel with step 5.
- 7. scatter.py (+ tests; multilayer copied from notebooks/cloaking/04 with attribution), then nb06 (needs bloch.py from step 6).
- 8. README.md: table (number, notebook, question, method, run time), main findings written from the printed outputs only, conventions (exp(-i omega t); TM = E_z, TE = H_z; lengths in units of the period for cell and band problems, lambda0 = 1 in nb06; American English), how the notebooks were checked (independent methods inside each notebook, the selftest, which tolerances were fixed a priori and which after a convergence study), known limits, literature with each entry marked 'checked' only after verification against a publisher or arXiv record, otherwise 'from memory'.
- 9. Re-execute all six notebooks top to bottom with 4 threads and record run times. Every notebook ends with a short caveats section and its own reference list. Commit only when the user asks.

## Claims register

- [theorem] The Q1 primal value U_h and the dual L_h = inv(R^T U_h[1/sigma] R) bracket sigma* in the Loewner order for every pixel microstructure, and both are monotone under 2x2 refinement. *Conditions:* Conforming periodic Q1 spaces, exact integration (pixel-constant coefficients), R the 90-degree rotation, 2D only (stream function), nested meshes for monotonicity. *Source:* Dirichlet and Thomson principles (Milton 2002); design scratch q1.py
- [theorem (new two-line proof in nb01); verified in scratch to 1.4e-12] Discrete Keller interchange: U_h[sigma_swap] = ab R inv(L_h[sigma]) R^T for any two-phase pixel field; for the checkerboard L_h U_h = ab at every N, so the geometric mean of the bracket is exact. *Conditions:* Two phases a, b. For the checkerboard, N even, so one-tile translation and 90-degree rotation are grid symmetries. *Source:* this design; continuum: Keller 1964, Dykhne 1971, Mendelson 1975
- [settled (theorem plus scratch: primal-primal error about twice the primal error)] The interchange theorem holds exactly for the primal-dual pair but not for the primal-primal product (proposal S's open question). *Conditions:* Same as above. *Source:* this design
- [standard result; confirmed indirectly by Q1 rates in scratch (local rate 1.1809 vs 2 lambda_1 = 1.1807 at kappa = 4); proposal S's arccos form is the other branch] The leading corner exponent of the four-tile checkerboard is lambda_1 = (4/pi) arctan(kappa^-1/2); the other branch (4/pi) arctan(kappa^1/2) > 1. *Conditions:* Quadrants a, b, a, b; kappa = b/a > 0; potential ~ r^lambda. *Source:* Kellogg 1971 (from memory); nb01 root-finding
- [standard FE theory plus measurement] The conforming Q1 energy error on the checkerboard decays like N^(-2 lambda_1); local rates approach from above, and the asymptotic regime needs N >> exp(1/lambda_1). *Conditions:* Uniform pixel grid with cross points at nodes. Scratch local rates: kappa = 10: 0.783 vs 0.780; kappa = 100: 0.304 vs 0.254 at N = 512 (pre-asymptotic). *Source:* design scratch q1.py
- [measured (proposal B's numbers reproduced); no theory claimed] The sampled-coefficient (Moulinec-Suquet) scheme converges faster than Q1 on the checkerboard, with no certificate. *Conditions:* Spectral derivatives with Nyquist components zeroed, PCG with inverse-Laplacian preconditioner (30-92 iterations, nearly N-independent). Proposal B's '6N iterations' applies only to unpreconditioned CG. *Source:* proposal B; design scratch ms.py (rates 1.28 at kappa = 10; 0.53 to 0.49 at kappa = 100)
- [theorem (bounds) plus standard approximation argument (rate); verified in scratch] Fourier-Galerkin with exact disk coefficients gives rigorous upper and lower bounds for the true disk, with width about 1/K. *Conditions:* Trial degree K per coordinate, coefficients to 2K, zero-padded grid M >= 4K + 2. Scratch widths 0.180, 0.086, 0.043, 0.0215 for K = 8-64 at f = 0.5, kappa = 10. *Source:* Vondrejc-Zeman-Marek 2014 (variational FFT-Galerkin, from memory); design scratch fg.py
- [corrected] Proposal B: 'the bracket midpoint agrees with Rayleigh to about 1e-4 at K = 64'. *Conditions:* The midpoint is 3.7e-3 high at K = 64 because the upper bound converges about 10 times more slowly in absolute terms. Extrapolating each bound linearly in 1/K gives 2.4154 and 2.4152 (non-rigorous). *Source:* design scratch fg.py
- [derived in this design; verified (MG at L = 1, the 0.3058 f^4 coefficient at L = 2, sum rules to 1e-15, spatial_dispersion's series to 5e-8)] Rayleigh matrix: W_lm = sqrt(l/m) R^(l+m) [pi delta_l1 delta_m1 + C(l+m-1, l) S_(l+m)] is real symmetric; eps*/eps_m = 1 + 2f [(I/beta - W)^-1]_11; its eigenpairs give the exact spectral measure (z_k = (1 - mu_k)/2, w_k = f v_k1^2), and m0 = f and m1 = f(1-f)/2 hold at every truncation. *Conditions:* Square array of non-touching disks, odd multipoles; the S_2 slot carries the periodic-field term pi (Weierstrass convention plus the -pi x correction). *Source:* Rayleigh 1892; Perrins, McKenzie and McPhedran 1979 (structure); design scratch rayleigh.py
- [standard (Eisenstein series of the square lattice, g3 = 0)] S_4 = Gamma(1/4)^8/(960 pi^2) = 3.15121200; S_8 = (3/7) S_4^2; S_n = 0 unless n = 0 mod 4. *Conditions:* Unit square lattice, origin excluded. *Source:* from memory; brute-force check in rayleigh selftest
- [computed (agrees with proposal B's 2.41547 and lies inside the FG bracket)] sigma*(f = 0.5, kappa = 10) = 2.4154673802 for the square array of disks. *Conditions:* L >= 5 multipoles. *Source:* design scratch rayleigh.py
- [conjecture to test; scratch counterexample] Proposal B: the error of a staircase disk is dominated by its area fraction. *Conditions:* Weak disks (phase 1 = 1 in 10), f = 0.5: the certified staircase brackets at N = 64 and 128 exclude Rayleigh at the pixel area fraction. *Source:* design scratch t4.py
- [theorem (trial-space inclusion)] Apparent tensors of a window are ordered affine >= periodic >= flux for every realization, discretely and in the continuum. *Conditions:* Same window, same discretization family. *Source:* Huet 1990 (ensemble version)
- [theorem (corollary of the discrete interchange identity)] For p = 1/2 random checkerboards, antithetic pairs satisfy det U_per[sigma] det L_per[sigma_swap] = (ab)^2 exactly, so the discrete bracket respects Dykhne's sqrt(ab) in law. *Conditions:* Two phases; the swapped field has the same law. *Source:* this design
- [corrected (excluded)] Proposal A: the percolation crossover sigma*(p_c, h) ~ h^1/2 can be measured with pixel Q1 brackets. *Conditions:* Q1 connects diagonal pixels through shared nodes (the dual does so for the other phase), so the bracket straddles two connectivity classes. Scratch: the Q1 checkerboard measure carries an atom at z = 0 of weight 0.17-0.20. *Source:* design scratch spec.py
- [theorem] F(s) = 1 - eps*/eps2 = integral dmu/(s - z) with mu >= 0 on [0,1], m0 = f1, and m1 = f1 f2/2 for C4-symmetric cells. *Conditions:* Two isotropic phases, s = 1/(1 - eps1/eps2). *Source:* Bergman 1978; Milton 1980/1981; Golden and Papanicolaou 1983
- [exact (derived from eps* = sqrt(eps1 eps2) by Stieltjes inversion)] The checkerboard measure is (1/pi) sqrt((1 - z)/z) dz on (0,1), with CDF (arcsin sqrt(z) + sqrt(z(1-z)))/pi. *Conditions:* Infinite periodic checkerboard. *Source:* this design; proposals A and S
- [exact algebra] Maxwell Garnett is one atom at (1-f)/2 of weight f; an isolated disk has 1/2; a dilute ellipse along its long axis has b/(a+b). *Conditions:* 2D. *Source:* standard
- [derived here; Q1 checkerboard weights measured in scratch (0.1978, 0.1794, 0.1683 at N = 16, 24, 32); FG part to test] Conforming primal measures carry an atom at z = 0 exactly when phase 1 percolates in the discrete space. Q1 measures of isolated pixel inclusions have none; FG measures of smooth inclusions have one of order 1/K. *Conditions:* Primal (upper-bound) discretizations. *Source:* this design; scratch spec.py
- [corrected] Proposal A: the checkerboard Q1 measure has mass 0.5 and first moment 0.125 to 1e-6 at N = 48, with Kolmogorov distance decreasing. *Conditions:* Mass is exact, but m1 = 0.1220, 0.1235, 0.1241 at N = 16, 24, 32 and the Kolmogorov distance is the spurious atom at 0. In the sampled scheme m1 = 1/8 exactly (algebraic identity, N divisible by 4). *Source:* design scratch spec.py
- [computed; corrects proposal A's 'more than 99% within 1e-3 of (1-f)/2', which holds only for dilute f] At f = 0.2827 the disk-array dipole atom is at z = 0.355273 (MG 0.358650) and carries 97.7% of the weight. *Conditions:* Rayleigh L = 20. *Source:* design scratch rayleigh.py
- [conjecture to test (formula from memory)] A 90-degree corner contributes essential Neumann-Poincare spectrum mapped to z in [1/4, 3/4]. *Conditions:* Curvilinear polygons; z = 1/2 - lambda. *Source:* Perfekt and Putinar 2017; Bonnetier and Zhang 2019
- [exact algebra] At negative contrast the four-tile corner exponents are purely imaginary for kappa < 0, kappa != -1. *Conditions:* tan^2(lambda pi/4) = kappa^(+-1). *Source:* proposal S; nb02 check
- [exact (from m0 = f; each atom is a Lorentz oscillator)] Drude absorption sum rule: integral_0^infinity omega Im eps*(omega) d omega = (pi/2) f omega_p^2 for every microstructure. *Conditions:* Drude inclusions in a vacuum host. *Source:* this design
- [exact (continuum) plus elementary (discrete)] A lossless checkerboard with eps1 < 0 < eps2 has eps* = i sqrt(|eps1| eps2); any finite discretization gives a real limit as the loss vanishes, so the limits do not commute. *Conditions:* Loss -> 0+ principal branch; finite atom sums. *Source:* proposal S; this design
- [theorem plus LP basic-solution argument] Bergman-Milton bounds: F(s) lies in the convex region of measures with fixed m0 (and m1); extremal measures have at most 1 (2) atoms; MG is the corner of the isotropic lens. *Conditions:* s outside [0,1]. *Source:* Bergman 1980; Milton 1980/1981
- [measured in scratch] At 30 dB, 60 real-contrast measurements determine about 6 numbers of the spectral measure, Drude spectra with gamma = 0.2 about 22, and with gamma = 0.05 about 75. *Conditions:* 400-point z grid, 60 data, threshold 1e-3 sigma_1. *Source:* design scratch stj.py
- [theorem (from memory); self-consistency verified (reduces to HS at isotropy, equality for rank-1 and rank-2 laminates)] The 2D G-closure of two isotropic phases is the Wiener box intersected with two trace inequalities; rank-2 laminates attain the curved boundaries, rank-1 laminates the corners, HS the isotropic points. *Conditions:* Two isotropic phases, fixed volume fraction, 2D conductivity. *Source:* Lurie and Cherkaev 1984; Murat and Tartar 1985
- [theorem (Schur)] A symmetric 2x2 tensor invariant under C4 is isotropic. *Conditions:* Any group containing a rotation of order >= 3. *Source:* standard
- [corrected premise, to test] Proposals A, B, C: a single-scale periodic C4v cell cannot reach the HS bound. *Conditions:* The literature (Vigdergauz for 2D elasticity; Liu, James and Leo 2007 for conductivity; from memory) says periodic inclusions with uniform interior field attain HS. nb03 tests gap -> 0 and field uniformity against saturation. *Source:* Vigdergauz 1994; Grabovsky and Kohn 1995; Liu, James and Leo 2007
- [conjecture to test] Proposal C: optimal designs do not converge under mesh refinement (relaxation). *Conditions:* Contradicted if periodic E-inclusions attain the anisotropic boundary in 2D. *Source:* proposal C; Allaire 2002
- [computed (Rayleigh plus Q1 bracket widths 0.68% at N = 64, 0.31-0.43% at N = 128)] The weak-disk array is 1.28% below HS_up at theta = 0.5 but 9.4% below at theta = 0.7, so certification can resolve optimized designs only at theta = 0.7. *Conditions:* sigma = 1 and 10. *Source:* design scratch t4.py
- [theorem] dU_ee/dsigma(x) = |e + grad chi_e(x)|^2 (envelope theorem); the element version is exact for Q1. *Conditions:* Self-adjoint cell problem. *Source:* standard
- [standard result; scratch slopes 1.006 and 1.007] ||u_eps - u0||_L2 = O(eps); the interior H1 error of w_eps is O(eps). *Conditions:* Smooth f, Dirichlet data 0, square domain. *Source:* Bensoussan, Lions and Papanicolaou 1978; Jikov, Kozlov and Oleinik 1994
- [corrected: true only for cuts that are not mirror lines] The global H1 error of the first-order expansion is O(eps^1/2) (boundary layer). *Conditions:* For mirror-line cuts of D4 cells the mismatch vanishes on the edges (normal corrector zero, tangential derivative of u0 zero), so the rate is O(eps). Scratch slopes: 0.586 (s = 1/4, last pair), 0.99 (s = 1/2), 0.96 (s = 0). *Source:* design scratch rates.py, rates2.py
- [standard theory plus scratch verification] The interior error of w_eps is eps u1bar + O(eps^2), with u1bar harmonic and boundary data from the boundary-layer tails; tails vanish for mirror-line cuts, giving O(eps^2). *Conditions:* Periodic D4 cell, edges aligned with the lattice. Scratch: tails 0.13554 and 0.16978; |err/eps - u1bar| = 5.4e-3, 2.9e-3, 1.5e-3 for N = 8, 16, 32; interior rates 1.85-1.94 (mirror cuts) and 1.03 (generic). *Source:* Allaire and Amar 1999; Moskow and Vogelius 1997; design scratch tail.py
- [corrected] Proposal A: second-order correctors give an eps^3 interior error. *Conditions:* Boundary-layer tails (and second-order tails) limit the interior error to O(eps) or O(eps^2) depending on the cut. *Source:* this design
- [standard; QEP verified in scratch (homogeneous O(h^2): 6.634 and 6.366 at m = 8 and 16)] The boundary layer decays like exp(-kappa_1 d/eps), with kappa_1 the smallest static evanescent Bloch exponent of the excited parity; kappa_1 = 2 pi for a homogeneous medium. *Conditions:* Edges aligned with the lattice. *Source:* design scratch qep.py
- [corrected] Proposal A: high-index inclusions slow the boundary-layer decay; the decay can be fitted in the fine solve within 5%. *Conditions:* At R = 0.3, kappa_1 exceeds the homogeneous value for a_in = 0.1, 10 and 100 (6.75, 6.83, 7.16 vs 6.37 at m = 16). The layer drops by about 1e-3 per cell, below the interior error, so it is measured on strips. *Source:* design scratch qep.py
- [theorem; scratch: M_11 = 1.5045, gap 1.565e-3 vs predicted 1.614e-3 at N = 32] Div-curl lemma: a_eps |grad u_eps|^2 converges weakly to sigma* |grad u0|^2, while |grad u_eps|^2 converges weakly to grad u0 . M grad u0 with M = <(I + grad chi)^T (I + grad chi)> != I. *Conditions:* Test functions vanishing on the boundary. *Source:* Murat and Tartar; Allaire 1992
- [computed in scratch] Bands 2-3 at M of the TM square lattice (R = 0.3a, eps = 12) form an E doublet (0.34941 a/lambda). *Conditions:* C4v-symmetric plane-wave cutoff. *Source:* design scratch pwe.py
- [theorem (character inner products: dim Hom = 3, 0, 1)] The second-order k.p at a C4v E doublet has exactly 3 constants and no linear term; at a C3v E doublet there is exactly 1 linear constant (Dirac velocity). *Conditions:* Time-reversal-symmetric scalar TM problem. *Source:* standard representation theory; this design
- [computed (scratch: 0.26964, linear splitting, gap 0.0124 for radii 0.21/0.19) plus theorem (PT quantization) plus standard model] The honeycomb TM rod lattice (R = 0.2a, eps = 12) has a Dirac point at K between bands 1 and 2; its Berry phase is pi; unequal rods give the massive-Dirac loop phase with opposite sign at K'. *Conditions:* Origin at the hexagon center; circular cutoff. *Source:* Fefferman and Weinstein 2012; design scratch hc.py
- [measured in scratch] A parallelogram plane-wave cutoff splits the honeycomb Dirac point at about 1e-6; a circular cutoff restores exact degeneracy. *Conditions:* About 300-500 plane waves. *Source:* design scratch hc.py
- [theorem (derived here) plus derivation to test] In a finite 1D stack of a symmetric cell with walls on mirror planes, every in-band mode sits exactly at k = n pi/(Na), so the envelope prediction is exact up to the quartic band term (error N^-4); generic walls need an O(a) end correction (error N^-3 without it). *Conditions:* u = 0 walls, symmetric cell. *Source:* this design; Craster, Kaplunov and Pichugin 2010 (envelope theory)
- [derived; scratch PWE: residual/k0^4 constant (3.59e-3 for eps = 4, 0.122 for eps = 12); Rytov limit checked analytically] TM: eps_eff = <eps> at leading order, and eps_eff(k0) = <eps> + k0^2 sum_{G != 0} |eps_G|^2/|G|^2 + O(k0^4), exact in contrast and isotropic; in 1D it equals Rytov's ordinary-wave formula. *Conditions:* Centrosymmetric or reciprocal medium, E along the invariant axis. *Source:* proposals C and S; Rytov 1956; design scratch pwe.py
- [corrected: the full sum is 1.287 (eps = 4, R = 0.3a), 18% larger] Proposal C's hand check 'the first four G give 1.09 (a/lambda)^2 within 5% of the full sum'. *Conditions:* Truncation at |n| <= 12. *Source:* design scratch pwe.py
- [standard (Fourier diffraction theorem)] Ring-aperture Born data sample the object spectrum only for |q| <= 2 k0, so periodic structure with p < lambda/2 is invisible at first order except through the boundary. *Conditions:* Born approximation, far-field rings. *Source:* Wolf 1969; Devaney
- [derived (Born integration by parts plus TM continuity); scratch dR_fit/dR_pred = 0.973 at p = 0.025, alpha -> 18.0 degrees] For concentric rings the termination acts at O(p) exactly as a radius shift dR = X0(R_D)/chibar, absorbed by a radius fit. The bulk dispersion is O(p^2) and shifts eps; the normal residual is O(p^2), so DeltaPhi ~ p^4 at a constant angle. *Conditions:* eps1 = 3, eps2 = 1.5, f = 1/2, R_D = 1 lambda. *Source:* this design; design scratch rings.py
- [measured in scratch; to test at smaller p] A homogeneous-disk fit does not recover the bulk (Rytov) effective eps even as p -> 0 (fitted shift about 0.78 of Rytov). *Conditions:* Both terminations. *Source:* design scratch rings.py
- [measured in scratch; refines proposal C's 'confidently wrong before detectable' (true against the 147-nat flag, false against a 3-nat evidence test)] For the mid-layer-cut rings at 30 dB, p_3nat < p_z < p_flag (about 0.032, 0.046, 0.084 lambda): the composition estimate is biased by more than 3 sd while passing the 3-sigma chi^2 flag. *Conditions:* ch15 rings, noise from the base-disk data. *Source:* design scratch rings2.py
- [measured in scratch; consistent with DESIGN 15.1 conditions] For the period cut, the first-order law linearized at the base point overestimates DeltaPhi about 100-fold; the profile foot point is needed. *Conditions:* Large in-model (radius) component of the model error. *Source:* design scratch rings2.py; docs/DESIGN.md section 15.1
- [corrected for TM] Proposal C: the unabsorbed residual scales like a because boundary-layer correctors are O(a). *Conditions:* The TM principal part does not oscillate (no O(1) gradient corrector). The O(p) effect is geometric and lies in the tangent space. *Source:* this design
- [theorem (symmetry), generalizing DESIGN 15.2] C20 selection rule: for an exactly rotation-invariant model and the ch15 acquisition, components of a C4-invariant model error in the Q_4 and Q_8 sectors are orthogonal to all model tangents and only raise DeltaPhi. *Conditions:* Grid and lattice centered with exact C4 symmetry; 40 and 60 equally spaced antennas. *Source:* this design; docs/DESIGN.md section 15.2
- [conjecture to test] An approximation-error model estimated from random terminations restores coverage for the composition. *Conditions:* Linearized estimator, held-out realizations. *Source:* Kaipio and Somersalo 2005, 2007

## Literature (verified)

Checked on 2026-11-23 against publisher, DOI, arXiv or library records by a bibliography agent; "checked" means the bibliographic details were confirmed, not that the content was read beyond what the notes say. Entries marked "corrected" are given in their corrected form. The statement checks that follow say what each source actually states.

- (checked) A. Bensoussan, J.-L. Lions, G. Papanicolaou, Asymptotic Analysis for Periodic Structures, North-Holland, Amsterdam, 1978 (reprinted by AMS Chelsea Publishing, 2011). Record: Open Library catalog search (first published 1978, North-Holland); AMS Chelsea reprint record https://doi.org/10.1090/chel/374 (Crossref: 2011). Grabovsky 1993 also cites it as North-Holland 1978.. No pages needed. Series volume number not checked.
- (checked) V. V. Jikov, S. M. Kozlov, O. A. Oleinik, Homogenization of Differential Operators and Integral Functionals, Springer-Verlag, Berlin, Heidelberg, 1994. Record: https://doi.org/10.1007/978-3-642-84659-5 (Crossref record: authors Jikov, Kozlov, Oleinik; 1994).. Translator and ISBN not checked.
- (checked) G. W. Milton, The Theory of Composites, Cambridge University Press, 2002 (published online 6 May 2002; ISBN 9780521781251 / 9780511613357). A SIAM Classics edition appeared in 2022. Record: https://doi.org/10.1017/CBO9780511613357 (Crossref). SIAM reprint https://doi.org/10.1137/1.9781611977486.. Chapter and section numbers not checked. Lim and Milton (arXiv 1809.10373) cite section 23.9 of this book for two-dimensional periodic Vigdergauz-type structures, which supports using that section for nb03 once its content is read.
- (checked) G. Allaire, Homogenization and two-scale convergence, SIAM J. Math. Anal. 23 (1992), 1482-1518. Record: https://doi.org/10.1137/0523084 (Crossref).. Volume, pages and year all match.
- (checked) G. Allaire, Shape Optimization by the Homogenization Method, Applied Mathematical Sciences, Springer, New York, 2002. Record: https://doi.org/10.1007/978-1-4684-9286-6 (Crossref; chapter 1 'Homogenization', pp. 1-89, is 10.1007/978-1-4684-9286-6_1).. Series volume number not checked.
- (corrected) L. Tartar, The General Theory of Homogenization: A Personalized Introduction, Lecture Notes of the Unione Matematica Italiana 7, Springer, Berlin, Heidelberg (copyright 2009; Springer/Crossref record dated 2010). Record: https://doi.org/10.1007/978-3-642-05195-1 (Crossref: published 2010, ISBNs 9783642051944 / 9783642051951; sibling chapters 10.1007/978-3-642-05195-1_2 and _3).. The plan omits the subtitle. 2009 vs 2010 depends on whether one uses the copyright year or the record date; pick one and state it.
- (checked) F. Murat, L. Tartar, H-convergence, in: Topics in the Mathematical Modelling of Composite Materials, Birkhauser, Boston, 1997, pp. 21-43 (reprinted in Modern Birkhauser Classics, 2018). Record: https://doi.org/10.1007/978-1-4612-2032-9_3 (Crossref: pp. 21-43, 1997); reprint https://doi.org/10.1007/978-3-319-97184-1_3.. Crossref did not list editors. This is the English translation of a 1977-78 Paris seminar; the original French has a different date.
- (checked) G. Allaire, M. Amar, Boundary layer tails in periodic homogenization, ESAIM: Control, Optimisation and Calculus of Variations 4 (1999), 209-243. Record: https://doi.org/10.1051/cocv:1999110 ; Numdam record https://numdam.org/item/COCV_1999__4__209_0 (Tome 4, 1999, pp. 209-243).. Matches the plan exactly.
- (checked) S. Moskow, M. Vogelius, First-order corrections to the homogenised eigenvalues of a periodic composite medium. A convergence proof, Proc. Roy. Soc. Edinburgh Sect. A 127 (1997), 1263-1299. Record: https://doi.org/10.1017/S0308210500027050 (Crossref).. Companion paper is Santosa and Vogelius, SIAM J. Appl. Math. 53 (1993) 1636-1668, with an erratum in 55 (1995) 864. Whether this paper covers polygonal domains was not read.
- (checked) D. Gerard-Varet, N. Masmoudi, Homogenization and boundary layers, Acta Mathematica 209 (2012), 133-178. Record: https://doi.org/10.1007/s11511-012-0083-5 (Crossref).. A related paper, Homogenization in polygonal domains, J. Eur. Math. Soc. 13 (2011) 1477-1503 (10.4171/JEMS/286), may be closer to the plan's 'polygonal domains' use.
- (checked) Z. Hashin, S. Shtrikman, A variational approach to the theory of the effective magnetic permeability of multiphase materials, J. Appl. Phys. 33 (1962), 3125-3131. Record: https://doi.org/10.1063/1.1728579 (Crossref, with abstract).. Abstract says the results also hold for dielectric constant and electric conductivity, and that for two phases the bounds are the most restrictive possible in terms of phase properties and volume fractions.
- (checked) J. B. Keller, A theorem on the conductivity of a composite medium, J. Math. Phys. 5 (1964), 548-549. Record: https://doi.org/10.1063/1.1704146 (Crossref).. Abstract: rectangular lattice of identical parallel cylinders of arbitrary cross section.
- (corrected) A. M. Dykhne, Conductivity of a two-dimensional two-phase system, Sov. Phys. JETP 32 (1971), no. 1, 63-65 (January 1971); Russian original Zh. Eksp. Teor. Fiz. 59 (1970), 110-115 (July 1970; received 5 Nov 1969). Record: Scanned English translation (3 pages, header 'Soviet Physics JETP, Volume 32, Number 1, January 1971', 'Zh. Eksp. Teor. Fiz. 59, 110-115 (July, 1970)') at https://courses.physics.ucsd.edu/2019/Spring/physics235/Dykhne.pdf ; JETP archive https://ftp.jetp.ras.ru/cgi-bin/e/index/r/59/1/p110?a=list. The plan omits the title. The end page 65 follows from the 3-page scan. The Russian original is 1970 (the JETP archive page lists January 1971, but the scan says July 1970). Keep the translation year 1971 if you cite JETP.
- (checked) K. S. Mendelson, A theorem on the effective conductivity of a two-dimensional heterogeneous medium, J. Appl. Phys. 46 (1975), 4740-4741. Record: https://doi.org/10.1063/1.321549 (Crossref, with abstract).. The plan omits the title. Abstract proves eta*_R(eta_R^-1) eta*(eta) = 1.
- (checked) Lord Rayleigh, On the influence of obstacles arranged in rectangular order upon the properties of a medium, Philosophical Magazine (5th series) 34 (1892), 481-502 (item LVI). Record: https://doi.org/10.1080/14786449208620364 (Crossref).. Reprinted in Scientific Papers vol. 4, pp. 19-38 of the 2009 CUP reprint (https://doi.org/10.1017/CBO9780511703997.005).
- (checked) W. T. Perrins, D. R. McKenzie, R. C. McPhedran, Transport properties of regular arrays of cylinders, Proc. R. Soc. Lond. A 369 (1979), 207-225. Record: https://doi.org/10.1098/rspa.1979.0160 (Crossref).. Bibliographic data only; no table value for sigma* of the square array was reachable (see statements).
- (checked) D. J. Bergman, The dielectric constant of a composite material - A problem in classical physics, Physics Reports 43 (1978), 377-407. Record: https://doi.org/10.1016/0370-1573(78)90009-1 (Crossref); Tel Aviv University CRIS record.. Title in the plan is correct.
- (corrected) D. J. Bergman, Exactly solvable microscopic geometries and rigorous bounds for the complex dielectric constant of a two-component composite material, Phys. Rev. Lett. 44 (1980), 1285-1287. Record: https://doi.org/10.1103/PhysRevLett.44.1285 (Crossref).. Add title and end page 1287. Crossref also lists a record with the same title at PRL 45, 148 (10.1103/PhysRevLett.45.148.2), probably an erratum; I did not open it.
- (checked) G. W. Milton, Bounds on the complex dielectric constant of a composite material, Appl. Phys. Lett. 37 (1980), 300-302. Record: https://doi.org/10.1063/1.91895 (Crossref).. Matches.
- (corrected) G. W. Milton, Bounds on the complex permittivity of a two-component composite material, J. Appl. Phys. 52 (1981), 5286-5293. (Companion: Bounds on the transport and optical properties of a two-component composite material, J. Appl. Phys. 52, 5294-5304.) Record: https://doi.org/10.1063/1.329385 (and 10.1063/1.329386 for the companion).. The plan omits the title. Pages are right. Do not confuse with the companion paper that follows it.
- (checked) K. Golden, G. Papanicolaou, Bounds for effective parameters of heterogeneous media by analytic continuation, Commun. Math. Phys. 90 (1983), 473-491. Record: https://doi.org/10.1007/BF01216179 (Crossref); first page of the paper read at https://www.math.lsu.edu/~shipman/courses/08A-7390/GoldenPapanicol1983 (header 'Commun. Math. Phys. 90, 473-491 (1983)').. Matches.
- (checked) K.-M. Perfekt, M. Putinar, The essential spectrum of the Neumann-Poincare operator on a domain with corners, Arch. Ration. Mech. Anal. 223 (2017), no. 2, 1019-1033 (published online 31 Oct 2016). Record: https://doi.org/10.1007/s00205-016-1051-6 (Crossref: published-print 2017-02, online 2016-10-31, vol 223, issue 2); arXiv:1601.03183.. Some indexes list 2016 (online date). Year 2017 is the print year.
- (checked) E. Bonnetier, H. Zhang, Characterization of the essential spectrum of the Neumann-Poincare operator in 2D domains with corner via Weyl sequences, Rev. Mat. Iberoam. 35 (2019), no. 3, 925-948. Record: https://doi.org/10.4171/rmi/1075 ; https://ems.press/journals/rmi/articles/16182 ; arXiv:1702.08127.. The plan omits the title.
- (checked) R. B. Kellogg, Singularities in interface problems, in: B. Hubbard (ed.), Numerical Solution of Partial Differential Equations-II (SYNSPADE 1970), Academic Press, New York, 1971, pp. 351-400. Record: https://doi.org/10.1016/B978-0-12-358502-8.50015-3 (Crossref: pp. 351-400, 1971). Editor name taken from a secondary reference list (Applicable Analysis, DOI 10.1080/00036817408839086), not from the publisher record.. The formula the plan attributes to Kellogg is stated and proved in the later paper R. B. Kellogg, On the Poisson equation with intersecting interfaces, Applicable Analysis 4 (1974/75), 101-129, which a recent arXiv paper (2608.22991) cites for the matching relations. Consider citing that paper instead, or in addition. I could not open the 1971 chapter, so I cannot say it contains the checkerboard exponent.
- (checked) K. A. Lurie, A. V. Cherkaev, Exact estimates of conductivity of composites formed by two isotropically conducting media taken in prescribed proportion, Proc. Roy. Soc. Edinburgh Sect. A 99 (1984), 71-87. See also the sequel: Exact estimates of the conductivity of a binary mixture of isotropic materials, Proc. Roy. Soc. Edinburgh Sect. A 104 (1986), 21-38. Record: https://doi.org/10.1017/S030821050002597X ; sequel https://doi.org/10.1017/S0308210500019041 ; Cambridge abstract page.. Bibliographic data match. The sequel (1986) is cited alongside the 1984 paper by Grabovsky 1993 as the pair that solves the two-isotropic-phase G-closure, so the plan should cite both.
- (corrected) F. Murat, L. Tartar, Calcul des variations et homogeneisation, in: Les methodes de l'homogeneisation: theorie et applications en physique (Breau-sans-Nappe, 1983), Collection de la Direction des Etudes et Recherches d'Electricite de France 57, Eyrolles, Paris, 1985, pp. 319-369 (one source gives 319-370). English translation: Calculus of variations and homogenization, in Topics in the Mathematical Modelling of Composite Materials, Birkhauser 1997, pp. 139-173. The G-closure/optimal-bounds statement is usually credited to a different paper: L. Tartar, Estimations fines des coefficients homogeneises, in: P. Kree (ed.), Ennio De Giorgi Colloquium, Pitman Res. Notes in Math. 125, London, 1985, pp. 168-187. Record: Search results quoting reference lists (Springer/Birkhauser lists, ESAIM M2AN, arXiv 1108.5764 for the Tartar paper); translation https://doi.org/10.1007/978-1-4612-2032-9_6 (Crossref, pp. 139-173); Grabovsky 1993 (https://faculty.cst.temple.edu/~yury/Gcl.pdf), ref. [9] cites Tartar's Pitman paper as the Murat-Tartar result.. The plan's entry has no venue. The Eyrolles pages are from reference lists only, not a publisher record. Decide which paper nb03 means: the Pitman paper (Grabovsky's choice) or the Eyrolles chapter.
- (checked) S. B. Vigdergauz, Two-dimensional grained composites of extreme rigidity, J. Appl. Mech. 61 (1994), 390-394. Record: https://doi.org/10.1115/1.2901456 (Crossref with abstract). Related: Grabovsky's discussion and Vigdergauz's closure, J. Appl. Mech. 62 (1995) 550.. Abstract concerns a thin elastic plate with periodic inclusions. Not a conductivity paper.
- (checked) Y. Grabovsky, R. V. Kohn, Microstructures minimizing the energy of a two phase elastic composite in two space dimensions. II: The Vigdergauz microstructure, J. Mech. Phys. Solids 43 (1995), no. 6, 949-972. Record: https://doi.org/10.1016/0022-5096(95)00017-D (Crossref); author copy https://faculty.cst.temple.edu/~yury/vigdergauz.pdf (header 'J. Mech. Phys. Solids, 43, No. 6, pp. 949-972, 1995').. The plan omits the title. Part I is 43, 933-947 (10.1016/0022-5096(95)00016-C).
- (checked) L. P. Liu, R. D. James, P. H. Leo, Periodic inclusion-matrix microstructures with constant field inclusions, Metall. Mater. Trans. A 38 (2007), 781-787 (online 21 Feb 2007). Record: https://doi.org/10.1007/s11661-006-9019-z (Crossref); full text read at https://ow3.math.rutgers.edu/~ll502/papers/LiuJamesLeo2007.pdf. Bibliographic data match. The paper does not itself state HS attainment for conductivity (see statements). The attainment statement is in the companion draft 'New extremal inclusions and their applications to two-phase composites' (Liu, James, Leo; draft of 13 Sep 2007, https://ow3.math.rutgers.edu/~ll502/papers/Einclusions.pdf; cited by Lim and Milton as 'preprint, 2008', no journal version found) and in L. P. Liu, Hashin-Shtrikman bounds and their attainability for multi-phase composites, Proc. R. Soc. A 466 (2010), 3693-3713 (https://doi.org/10.1098/rspa.2009.0554; author preprint read). I recommend adding the 2010 paper to the literature list.
- (checked) H. Moulinec, P. Suquet, A numerical method for computing the overall response of nonlinear composites with complex microstructure, Comput. Methods Appl. Mech. Engrg. 157 (1998), 69-94. Record: https://doi.org/10.1016/S0045-7825(97)00218-1 (Crossref).. The plan omits the title. (The 1994 C. R. Acad. Sci. Paris paper is the original FFT scheme; Vondrejc et al. cite it as Moulinec-Suquet 1994.)
- (checked) J. Zeman, J. Vondrejc, J. Novak, I. Marek, Accelerating a FFT-based solver for numerical homogenization of periodic media by conjugate gradients, J. Comput. Phys. 229 (2010), 8065-8071. Record: https://doi.org/10.1016/j.jcp.2010.07.010 (Crossref).. The plan omits the title.
- (corrected) J. Vondrejc, J. Zeman, I. Marek, An FFT-based Galerkin method for homogenization of periodic media, Comput. Math. Appl. 68 (2014), 156-173 (arXiv:1311.0089). For guaranteed bounds the right paper is: J. Vondrejc, J. Zeman, I. Marek, Guaranteed upper-lower bounds on homogenized properties by FFT-based Galerkin method, Comput. Methods Appl. Mech. Engrg. 297 (2015), 258-291 (arXiv:1404.3614). Record: https://doi.org/10.1016/j.camwa.2014.05.014 ; https://doi.org/10.1016/j.cma.2015.09.003 ; arXiv abs 1311.0089 and 1404.3614 ; also J. Vondrejc, Improved guaranteed computable bounds on homogenized properties of periodic media by the Fourier-Galerkin method with exact integration, Int. J. Numer. Methods Eng. 107 (2016), 1106-1135 (https://doi.org/10.1002/nme.5199, arXiv:1412.2033).. The 2014 CAMWA paper is correct as cited for 'variational FFT-Galerkin' but contains no guaranteed bounds. The 2016 single-author IJNME paper (exact integration of coefficients) is probably the closest to the plan's 'exact disk coefficients'; I did not read it. There is also a PAMM 14 (2014) 563-564 abstract.
- (checked) C. Huet, Application of variational concepts to size effects in elastic heterogeneous bodies, J. Mech. Phys. Solids 38 (1990), no. 6, 813-841. Record: https://doi.org/10.1016/0022-5096(90)90041-2 (Crossref); HAL open copy https://hal.archives-ouvertes.fr/hal-03686624/file/Huet1990.pdf (read).. Matches. Content caveat in statements: this paper treats static (uniform traction) and kinematic (uniform strain) conditions, not periodic ones.
- (checked) R. V. Craster, J. Kaplunov, A. V. Pichugin, High-frequency homogenization for periodic media, Proc. R. Soc. A 466 (2010), 2341-2362. Record: https://doi.org/10.1098/rspa.2009.0612 (Crossref, with abstract).. Matches.
- (checked) G. Allaire, C. Conca, Bloch wave homogenization and spectral asymptotic analysis, J. Math. Pures Appl. (9) 77 (1998), 153-208. Record: https://doi.org/10.1016/S0021-7824(98)80068-8 (Crossref).. Matches.
- (checked) C. Conca, M. Vanninathan, Homogenization of periodic structures via Bloch decomposition, SIAM J. Appl. Math. 57 (1997), 1639-1659. Record: https://doi.org/10.1137/S0036139995294743 (Crossref).. The plan omits the title.
- (checked) C. L. Fefferman, M. I. Weinstein, Honeycomb lattice potentials and Dirac points, J. Amer. Math. Soc. 25 (2012), no. 4, 1169-1220. Record: https://doi.org/10.1090/S0894-0347-2012-00745-0 (Crossref, with abstract).. Matches.
- (corrected) S. M. Rytov, Electromagnetic properties of a finely stratified medium, Sov. Phys. JETP 2 (1956), no. 3, 466-475; Russian original Zh. Eksp. Teor. Fiz. 29 (1955), 605-616 (November 1955; submitted 9 June 1954). Record: Scanned English translation from the journal archive, http://jetp.ras.ru/cgi-bin/dn/e_002_03_0466.pdf (10 pages; the scan's header reads 'Volume 2, Number 3, May, 1956', and gives 'J. Exper. Theoret. Phys. USSR 29, 605-616 (November, 1955)'). Archive listing https://jetp.ras.ru/cgi-bin/r/index/e/2/3/p466?a=list gives March 1956 for the issue.. The plan's pages '466-475' are right (10 pages). The plan omits the Russian original (1955, pp. 605-616). The issue month is inconsistent between the scan header (May) and the archive page (March); the year 1956 is safe.
- (checked) J. Kaipio, E. Somersalo, Statistical and Computational Inverse Problems, Applied Mathematical Sciences, Springer, New York, 2005. Record: https://doi.org/10.1007/b138659 (Crossref: published 2005, series Applied Mathematical Sciences).. Series volume number and chapter on the approximation-error model not checked.
- (checked) J. Kaipio, E. Somersalo, Statistical inverse problems: Discretization, model reduction and inverse crimes, J. Comput. Appl. Math. 198 (2007), no. 2, 493-504. Record: https://doi.org/10.1016/j.cam.2005.09.027 (Crossref); abstract on the Aalto research portal.. Matches.
- (checked) E. Wolf, Three-dimensional structure determination of semi-transparent objects from holographic data, Optics Communications 1 (1969), 153-156. Record: https://doi.org/10.1016/0030-4018(69)90052-2 (Crossref).. Matches. 'Devaney' in the plan has no citation to check.

- In-repository sources: `docs/DESIGN.md` sections 2 and 15; `notebooks/cloaking/04_partial_waves_and_scattering_cancellation.ipynb` (multilayer Mie); `notebooks/spatial_dispersion/rodlattice.py` and its notebook 03 (Rayleigh series check). Cite by path.

### What the sources state (statement checks)

- **Perfekt and Putinar (2017); Bonnetier and Zhang (2019)** (confirmed): Perfekt-Putinar (arXiv 1601.03183, read): their K is normalized as (2/pi) times the normal derivative of log, with spectrum in [-1,1]. Theorem 3: for a wedge W_alpha, sigma(K) = {|x| <= |1 - alpha/pi|} union {1}, and the rest is essential with multiplicity 2. Theorem 7: for a C^2 curvilinear polygon with angles alpha_1..alpha_N, sigma_ess(K) = {x real: |x| <= max_j |1 - alpha_j/pi|}, in the energy norm. In the standard normalization (spectrum in (-1/2, 1/2]) this is half: Bonnetier-Zhang (arXiv 1702.08127, read) write sigma_ess(K*_D) = [lambda-, lambda+], lambda+ = -lambda- = (1/2)(1 - alpha/pi). So the plan's interval is right in the standard normalization; for alpha = pi/2 it is [-1/4, 1/4]. The mapping z = 1/2 - lambda gives [1/4, 3/4]; I checked it is consistent with s = 1/(1 - eps1/eps2) at the plasmonic resonance eps1/eps2 = (lambda+1/2)/(lambda-1/2). Caveats: with several corners the band is set by the largest |1 - alpha_j/pi| (a reflex corner counts too), not only the 'most acute' angle as Bonnetier-Zhang phrase it; the result needs C^2 arcs between corners; Perfekt-Putinar do not control a possible singular continuous part. Bonnetier-Zhang add: a second proof by singular Weyl sequences built from elliptic corner singularity functions r^eta phi(theta). They show (Theorem 2) that the critical interval [lambda-, lambda+], the contrast range where eta is complex ('plasmonic black-hole waves' not in H^1), is contained in sigma_ess, and (Theorem 3, via a localization and a system of PDEs near the vertex) that sigma_ess equals it. The plan can drop 'formula from memory'.
- **Lurie and Cherkaev (1984); Murat and Tartar (1985)** (partially): Grabovsky 1993 (Proc. R. Soc. Edinburgh 123A, 423-432; open copy read) says the two-isotropic-component G-closure was completely solved by Murat-Tartar (his ref. = Tartar's Pitman 1985 paper, pp. 168-187) and independently by Lurie-Cherkaev (1984 and 1986), by compensated compactness. His Theorem 1 and Theorem 3 (for two well-ordered, possibly anisotropic conductors A < B, any dimension n, theta_A + theta_B = 1) say the G-closure is exactly the set of C* with (i) the Wiener bounds H <= C* <= M in the sense of quadratic forms (H the harmonic, M the arithmetic mean) and (ii) the trace bounds Tr(A(C*-A)^-1) <= (1/theta_B) Tr(A(B-A)^-1) + theta_A/theta_B and Tr(B(B-C*)^-1) <= (1/theta_A) Tr(B(B-A)^-1) - theta_B/theta_A. For isotropic A = aI, B = bI in 2D this is sum_i a/(lambda_i - a) <= 2a/(theta_B (b-a)) + theta_A/theta_B and the analogous inequality at b. I checked by hand that equality in the first one reproduces the 2D Hashin-Shtrikman lower bound at lambda1 = lambda2, and that a simple laminate gives equality. Lurie-Cherkaev 1984 (Cambridge abstract) solves the problem for two isotropic media u+E and u-E with E = ii + jj, which indicates 2D; it contrasts itself with an earlier 2D G-closure of an arbitrary set (Lurie-Cherkaev, J. Optim. Theory Appl. 42, 283-304, 1984). Differences and unverified parts: (a) the trace inequalities hold in any dimension; the 2D case is the Lurie-Cherkaev 1984 paper, with the sequel in 1986; (b) the plan's Murat-Tartar 'Calcul des variations et homogeneisation (1985)' is a different paper from the one Grabovsky cites (see the corrected entry); (c) I did not verify which laminates realize which parts of the boundary (one search result says second-rank laminates realize the bound for two materials; the 'rank-1 at corners' claim is unchecked).
- **Liu, James and Leo (2007) and related papers** (partially): The MMTA 2007 paper (full text read) constructs periodic 'E-inclusions' for magnetostatics/eigenstrain: constant magnetization of the inclusion gives constant field in it, for any volume fraction in (0,1), any Bravais lattice, by a free-boundary obstacle problem; its abstract says in 2D the results coincide with Vigdergauz's microstructures. It does not mention HS bounds. It touches conductivity only in one closing sentence, saying E-inclusions can be shown to characterize the G-closure of conductivity tensors (reference to other work). The Laplace-type equation makes the transfer to conductivity natural, but the paper does not state it. The attainment statements are in the companion draft 'New extremal inclusions and their applications to two-phase composites' (13 Sep 2007; read): Theorem 7 and Remarks 6, 10 give equality in the structure-independent HS-type bounds for periodic E-inclusions (under conditions: Tr F nonzero, I in the range of L2-L1, Q in a class Q), say periodic structures attaining the optimal bounds must be periodic E-inclusions, and Remark 10 treats two conductors (trace bounds, attainable per Grabovsky). That draft was never found in a journal (Lim-Milton cite it as 'preprint, 2008'). The published statement is L. P. Liu, Proc. R. Soc. A 466 (2010) 3693-3713 (author preprint read): a periodic microstructure attains an HS bound iff it is a corresponding periodic E-inclusion (Theorem 2.1); the paper includes a section on conductive composites. Lim and Milton (arXiv 1809.10373) summarize: the Liu-James-Leo variational approach is not limited to 2D and found 3D periodic arrays that saturate HS. My inference, not a source statement: a single C4-symmetric Vigdergauz-type cell has isotropic effective conductivity by Schur, so it would sit on the isotropic HS bound if the attainment conditions hold. So the plan's corrected premise is supported, but the correct citation is Liu 2010 plus the 2007 draft, not the MMTA paper alone.
- **Vigdergauz (1994); Grabovsky and Kohn (1995)** (partially): Both are about elasticity. Vigdergauz 1994 abstract (Crossref): a thin elastic plate with periodic inclusions, shapes minimizing stress energy given mean stresses and volume fraction, via elliptic functions; also effective moduli and a finite-step multiphase construction with macroisotropy. Grabovsky-Kohn II (text read): Vigdergauz's extremal elastic microstructure for two isotropic phases in 2D, a periodic array of inclusions; explicit inclusion shape, limits, relevance to minimizing maximum stress and to Michell trusses. A keyword search of the Grabovsky-Kohn text for conductivity, dielectric, thermal and antiplane found nothing. For conductivity the attainment comes from the E-inclusion papers: Lim and Milton (arXiv 1809.10373) describe Vigdergauz microstructures as 2D E-inclusions with periodic boundary conditions and note that finding them is important for conductivity extremes attaining HS; Liu-James-Leo's draft (Remark 10) treats conductivity trace bounds; Milton's book section 23.9 is cited for 2D periodic multi-inclusion structures. The plan's 'check that the conductivity analogue is stated there' therefore fails for these two papers.
- **Allaire and Amar (1999)** (partially): I could not open the full text (the publisher PDF returned an error). From the abstract as relayed by the IP Paris research portal and Cambridge/EUDML summaries: they study boundary layers in periodic homogenization in rectangular domains, with either a fixed or an oscillating boundary. The layers oscillate strongly near the boundary and decay exponentially into the interior, leaving a nonzero limit, the 'boundary layer tail'. At first order in the period the tails add a dispersive term to the homogenized equation and yield an effective Fourier boundary condition; the paper also derives interior estimates using these tails. So 'rectangular domains, tails matter at first order' is supported. Not verified: the boundary condition type (Dirichlet), the exact interior rate, and the plan's claims that tails vanish for mirror-line cuts of D4 cells and then give O(eps^2); those are the plan's own scratch results. Non-rectangular and irrational-slope boundaries are in Moskow-Vogelius and Gerard-Varet-Masmoudi, not here.
- **Kellogg (1971) or later sources** (confirmed): I could not read Kellogg 1971. Independent checks: (1) I solved the transmission problem numerically (potential r^lambda f(theta), f = A cos(lambda theta) + B sin(lambda theta) in each quadrant, continuity of f and of sigma f' at the four rays; 8x8 determinant, pure Python script in the scratch folder, kel2.py). Smallest roots: 0.590334 (kappa = 4), 0.389964 (kappa = 10), 0.126902 (kappa = 100), equal to (4/pi) arctan(kappa^-1/2) to 6 digits; the next root is 2 - lambda_1 (1.4097, 1.6100, 1.8731), which is the plan's 'other branch'. (2) Zhang, arXiv 2608.22991, cites Kellogg's matching relations (R. B. Kellogg, Applicable Analysis 4 (1974/75) 101-129) and gives R = cot^2(pi*gamma/4), i.e. gamma = (4/pi) arctan(R^(-1/2)) in (0,1) for R > 1, with u = r^gamma mu(theta), a bijection between gamma in (0,1) and R in (1, infinity). The plan's 'tan^2(lambda pi/4) = kappa^(+-1)' is the same relation. Note the primary source for this formula appears to be Kellogg 1974/75, not necessarily the 1971 chapter.
- **Plan (Eisenstein series of the square lattice); Perrins, McKenzie and McPhedran (1979)** (partially): Confirmed by computation: Gamma(1/4)^8/(960 pi^2) = 3.1512120022; the q-series 2 zeta(4)(1 + 240 sum sigma_3(n) e^(-2 pi n)) gives 3.1512120022; brute-force sum over |m|,|n| <= 1000 of 1/(m+in)^4 gives 3.15121167 (tail still converging). Godin 2012 (arXiv 1201.1419, section V.A) lists S = 3.15121 and the eighth-power sum 4.25577 for the unit square lattice. Recursion: DLMF 23.9 gives c_n = (2n-1) sum over nonzero lattice points of w^(-2n), c2 = g2/20, c3 = g3/28, and for n >= 4 c_n = 3/((2n+1)(n-3)) sum_{m=2}^{n-2} c_m c_{n-m}. With g3 = 0 (c3 = 0) this gives G8 = (3/7) G4^2 = 4.2557730, matching a brute-force sum of 1/(m+in)^8 (4.2557730); also G12 = 3.9388490 and G16 = 4.0156950 match brute force to 12 digits. All odd-index c_n vanish, so S_n = 0 unless n is a multiple of 4 (for n > 2). DLMF does not state the Gamma(1/4) closed form; I checked it numerically only. Not found: a table value of sigma* in Perrins-McKenzie-McPhedran 1979; the paper is not open (searches returned only the bibliographic record), so sigma*(f = 0.5, kappa = 10) = 2.4154673802 is unverified against it. Godin's series (his eq. 76) is said to agree with Rayleigh and Perrins et al. but he gives no table at that point.
- **Craster, Kaplunov and Pichugin (2010)** (confirmed): Publisher abstract (Crossref): two-scale asymptotics about standing waves, periodic or double-period on the cell scale, in a doubly periodic medium whose microstructure is much smaller than the macrostructure; these frequencies lie outside classical homogenization, hence the name 'high-frequency homogenization'. The resulting long-wave equations depend explicitly only on the macroscale, with the microscale entering through integral quantities, and reproduce the Bloch spectrum near Brillouin-zone edges. Only the abstract was read: the plan's use for wall boundary conditions of a finite 1D stack is not covered by the abstract.
- **Fefferman and Weinstein (2012)** (confirmed): Abstract (Crossref): the 2D Schrodinger operator with a potential having honeycomb symmetry has dispersion surfaces with conical singularities (Dirac points) at the vertices of the Brillouin zone, with no assumption on the size of the potential; the conical points are robust to a restrictive class of perturbations that break the honeycomb symmetry; general small perturbations have smooth dispersion surfaces. Caveats: the theorem is for the scalar Schrodinger operator -Delta + V, not the TM Maxwell operator with epsilon in divergence form, so the plan's TM rod lattice is an analogy; a non-degeneracy condition on a Fourier coefficient of V is, from memory, required and is not in the abstract; the Berry phase pi is not in the abstract.
- **Kaipio and Somersalo (2007)** (partially): Abstract (Aalto portal and aggregators): when an inverse problem is posed in function spaces and then discretized, a discretization error arises; because the problem is ill-posed, ignoring it can degrade the reconstruction badly; the Bayesian framework can estimate the statistics of the discretization error and fold it into the measurement and modeling errors, and also gives a controlled way to reduce dimension; a computed example. This is the content the plan uses. The phrase 'approximation error approach' does not occur in the abstract; a Bayesian approximation-error page of the UEF inverse-problems group attributes the approach to Kaipio and Somersalo. I did not check the 2005 book for the corresponding section.
- **Huet (1990)** (partially): Huet's paper (open HAL copy read) abstract: general procedure for finite specimens when no representative volume applies; three kinds of results: relations between a large specimen and sets of smaller ones, hierarchies between families of specimens of different sizes, and absolute bounds as the size goes to zero or to the representative volume. Text: statistical (ensemble-averaged) static apparent modulus is a lower bound and kinematic one an upper bound (static/kinematic uniform conditions), with hierarchies for nested partitions. The word 'periodic' does not occur in the paper. So the two ends of the plan's chain are Huet's; the periodic middle is not. Kanit et al. 2003 (Int. J. Solids Struct. 40, 3647-3679, read) cite Huet 1990 for S_app^-1 <= C_eff <= C_app (static below, kinematic above) and then only state that the periodic estimate 'also lies between', without proof. Hazanov and Huet 1994 (J. Mech. Phys. Solids 42, 1995-2011) extended this to mixed conditions. Periodic <= kinematic is an elementary trial-space inclusion (periodic fluctuations contain the zero-boundary ones); periodic >= static is not an inclusion of trial spaces, so the plan's label 'theorem (trial-space inclusion)' for the full chain overstates it; it needs the Hill-Mandel/mixed-boundary result or its own proof. The plan says 'ensemble version' for Huet, which fits, but the per-realization ordering static <= kinematic is the standard one.
- **Vondrejc, Zeman and Marek** (partially): The 2015 paper is the right one for bounds: Vondrejc, Zeman, Marek, 'Guaranteed upper-lower bounds on homogenized properties by FFT-based Galerkin method', CMAME 297 (2015) 258-291 (arXiv 1404.3614; posted April 2014, v3 April 2015, first titled 'Accurate upper-lower bounds ...'). Abstract: guaranteed upper-lower bounds on homogenized coefficients from the periodic cell problem in the scalar elliptic setting, built on the variational reformulation of the Moulinec-Suquet scheme (conforming Galerkin with trigonometric polynomials) by adapting the primal-dual finite-element framework of Dvorak and Wieckowski; the discretization differs for odd and non-odd numbers of grid points. The 2014 CAMWA paper (arXiv 1311.0089) shows Moulinec-Suquet is a Galerkin discretization with trigonometric polynomials and a numerical integration scheme, and improves solver performance; it has no guaranteed bounds. The related single-author paper (Vondrejc, IJNME 107 (2016) 1106-1135, 'exact integration') is the likely home for exact-coefficient bounds like the plan's disk case; I did not read it. The plan's 'width about 1/K' is not claimed in any abstract I read.
- **Rytov (1956)** (confirmed): Scanned translation read (JETP 2, 466-475, page 470, Eq. 24). For layers a and b (period d = a + b), E parallel to the layers: eps^e = eps_bar [1 + (k^2 a^2 b^2 / 12 d^2) (mu_bar mu_tilde / (mu1 mu2)) (n1^2 - n2^2) (eps1 - eps2)/eps_bar], with eps_bar the arithmetic mean, and an analogous mu^e. Rytov says a more precise evaluation shows the correction terms are of order k^2 d^2. For mu = 1 (n_i^2 = eps_i) this reduces to eps^e = <eps> + k^2 a^2 b^2 (eps1 - eps2)^2 / (12 d^2). I checked it equals the plan's sum: sum_{G != 0} |eps_G|^2/|G|^2 is the variance of the mean-zero antiderivative of eps - <eps>, a triangle wave of peak-to-peak height a b (eps1 - eps2)/d, whose variance is height^2/12. For propagation across the layers (Eq. 31-34 region) he states the correction to the index n is of second order in kd. He also writes that the validity of the correction (24) is limited by condition of small differences between layer parameters. The leading-order tensors (arithmetic mean along, harmonic mean across) are Eqs. 20-23.
- **Bergman (1978, 1980), Milton (1980, 1981), Golden and Papanicolaou (1983)** (partially): Golden-Papanicolaou (CMP 90, read): abstract says they give a mathematical formulation of the method of Bergman and Milton, which does not use a variational principle but the analytic dependence of the effective parameter on the component parameters, restricted to two components. Their Representation Theorem (4.11-4.12): with s = 1 - 1/h there exist finite Borel measures mu_ik on [0,1], the diagonal ones positive, with F_ik(s) = integral mu_ik(dz)/(s - z) for s outside [0,1]; moments of the positive measures are determined by derivatives at h = 1 (5.6-5.7). Bergman 1978 (abstract only): defines characteristic geometric functions of the microstructure, derives their general analytic properties, and notes that the same applies to magnetic permeability, conductivity and diffusivity. Bergman 1980 PRL 44, 1285: only the title (exactly solvable geometries and rigorous bounds on the complex dielectric constant) was verified. Milton 1980 APL 37, 300 (abstract): generalizes the Wiener and Hashin-Shtrikman bounds to complex dielectric constants; Milton 1981 JAP 52, 5286 (abstract): the composite permittivity lies in a simply constructed region of the complex plane that depends on what is known of the structure, and in many cases is the most restrictive region possible from the information. Not verified: the specific m0 = f1 and m1 = f1 f2/2 for C4-symmetric cells (m0 is the volume fraction by construction; the value of m1 was not located in what I read); that extremal measures have at most 1 (2) atoms and that Maxwell Garnett is the isotropic-lens corner.

- Added by the premise review: L. P. Liu, "Hashin-Shtrikman bounds and their attainability for multi-phase composites", Proc. R. Soc. A 466 (2010) 3693-3713, https://doi.org/10.1098/rspa.2009.0554 (attainment of HS bounds by periodic E-inclusions, including conductivity); R. B. Kellogg, "On the Poisson equation with intersecting interfaces", Applicable Analysis 4 (1974/75) 101-129 (the four-quadrant exponents); J. Vondrejc, J. Zeman, I. Marek, "Guaranteed upper-lower bounds on homogenized properties by FFT-based Galerkin method", Comput. Methods Appl. Mech. Engrg. 297 (2015) 258-291 (the guaranteed bounds); T. Kanit et al., Int. J. Solids Struct. 40 (2003) 3647-3679 (periodic apparent properties between the kinematic and static ones); Y. Grabovsky, Proc. R. Soc. Edinburgh A 123 (1993) 423-432 (the G-closure trace bounds in any dimension). Their details come from the bibliography agent's notes above.


## Excluded, and why

- **Percolation crossover at p_c with pixel FE (proposal A)**: Q1 shares nodes between diagonal pixels, so primal and dual see different connectivities (8-neighbor for the conducting phase). The bracket straddles two percolation thresholds, and corner-limited convergence at high contrast is only logarithmic.
- **Keller near-touching asymptotics (proposals B and C)**: Needs about 40 multipoles near touching (scratch: f = 0.78, kappa = 1e6 still changes between L = 20 and 40), and pixel solvers cannot resolve the gap. It tests the multipole solver more than homogenization.
- **FFT solver zoo as a notebook: Eyre-Milton, basic scheme, Willot (proposal B)**: Solver engineering rather than structure. One sampled-scheme vs Q1 comparison is kept in nb01 section 3.
- **EMA zoo and RSA disk ensembles (proposal C)**: Applied and tutorial-adjacent for this reader, and budget-heavy. MG, Bruggeman and HS appear as formulas and exact references.
- **Multipole cluster solver for lattice-filled disks (proposal C)**: rodlattice.cluster is untested. The exact ring scatterer (validated multilayer Mie) covers the quantitative part, and testbed lattices cover the symmetry part.
- **Conca-Vanninathan Gamma-point Hessian and the Burnett tensor (proposal S)**: The TE static permittivity of the rod lattice is already checked against Rayleigh in spatial_dispersion, and fourth-order dispersion overlaps its k^2 coefficients. The Burnett sign claim is unverified.
- **Second-order correctors with an eps^3 interior rate (proposal A)**: Wrong premise: boundary-layer tails limit the interior error (scratch-verified).
- **2D finite-crystal supercell envelope test (proposal B)**: Supercell eigenproblems exceed the budget. The exact 1D stack tests the same envelope theory, including its boundary condition.
- **Accidental Dirac cones at Gamma (proposal B) and 1D Zak-phase domain walls (proposal S)**: The K-point Dirac cone and Berry phases already carry the topology, and the budget is spent there.
- **Jackiw-Rebbi edge states on a honeycomb supercell (proposal A)**: Supercell cost.
- **Neural surrogates of the cell problem (all proposals)**: No independent reference beyond the solvers, low mathematical content, and the training budget is above 3 minutes.
- **Bayesian sampling of the spectral measure (proposal A)**: Replaced by the linear information count of nb02 section 8, which answers the identifiability question at no sampler cost.
- **Zhikov high-contrast homogenization, artificial magnetism, dynamic nonlocal eps(omega, k), retrieval**: Covered by spatial_dispersion 01-09.
- **3D, elasticity, nonlinear composites**: Outside the electromagnetic-imaging brief and the run-time budget.
