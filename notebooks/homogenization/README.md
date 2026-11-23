# Homogenization

Six notebooks on homogenization proper: what the effective tensor of a periodic two-phase medium is
as a mathematical object, how to compute it with certificates instead of convergence plots, what its
analytic structure says, which tensors two phases can make, how fast the two-scale approximation
converges on a finite domain and why, what replaces it at a band edge, and what a TM scattering
experiment of this project sees of a sub-wavelength microstructure. The dynamic side of the subject,
the nonlocal permittivity eps(omega, k), artificial magnetism and parameter retrieval, is in
[`../spatial_dispersion`](../spatial_dispersion/README.md) and is not repeated here. Read the
notebooks in order. Each one runs on its own.

| # | Notebook | Question | Method | Run time |
|---|---|---|---|---|
| 01 | `01_cell_problem_duality` | What do conforming discretizations of the two minimum principles certify, which duality identities hold exactly on the grid, and what limits the convergence? | Q1 primal/dual pair, Fourier-Galerkin with exact coefficients, Rayleigh multipoles, four-tile corner exponents, apparent tensors of windows | 94 s |
| 02 | `02_spectral_measure` | What does the spectral measure of the effective permittivity look like for smooth inclusions, corners and plasmonic constituents, and how many numbers does a spectrum determine? | Bergman-Milton measures of three discretizations, the Rayleigh matrix, the exact checkerboard and quarter-square measures, Lanczos, linear programs for the bounds | 36 s |
| 03 | `03_g_closure_and_optimal_cells` | Which effective tensors can two phases make, and how close does a single periodic inclusion come to the bounds? | Trace bounds and sequential laminates, envelope-theorem gradients checked against `jax.grad`, density-based design certified by Q1 brackets | 115 s |
| 04 | `04_two_scale_and_boundary_layers` | On a finite domain, how fast does the two-scale expansion converge, and where is it wrong? | Fine Q1 solves against correctors, boundary-layer strips and tails, static evanescent Bloch waves, weak limits | 50 s |
| 05 | `05_band_edges_and_symmetry` | At a band edge, what fixes the form of the effective (envelope) equation, and what is its boundary condition? | Plane waves with symmetry-preserving cutoffs, characters and Hom dimensions, degenerate k.p, Wilson loops, 1D transfer matrices | 40 s |
| 06 | `06_what_scattering_sees` | What does a TM scattering experiment see of a sub-wavelength microstructure, and when does the microstructure bias an estimate rather than flag the model? | The TM second-order law, the testbed's Born model, exact multilayer Mie for ring structures, the detectability law of `docs/DESIGN.md` section 15, rotation sectors | 107 s |

Run times are for an M3 Max with four threads. The solvers are modules in this folder:
`cellsolver.py` (Q1 primal/dual brackets, periodic and window problems; the sampled Moulinec-Suquet
scheme; Fourier-Galerkin with exact coefficients), `bounds.py` (Wiener, Hashin-Shtrikman, Maxwell
Garnett, Bruggeman, sequential laminates, trace bounds, Bergman-Milton support functions),
`rayleigh.py` (lattice sums, the Rayleigh matrix and its exact spectral measure), `spectral.py`
(discrete spectral measures, Lanczos, exact checkerboard moments and Jacobi coefficients),
`twoscale.py` (fine solves, correctors, boundary-layer tails, evanescent exponents, weak limits),
`bloch.py` (TM plane waves, little groups, k.p, Berry phases, 1D stacks) and `scatter.py`
(multilayer Mie for rings, homogeneous-disk fits, the rotation-sector projectors of the testbed's
acquisition, the bridge to `bayes_scatter`).

## Main findings

**01. The cell problem as a pair of convex programs.** The Q1 primal value U_h and the dual value
L_h = (R^T U_h[1/sigma] R)^-1 bracket the effective tensor of every pixel field in the Loewner order:
the flux principle in 2D is the potential principle for 1/sigma, rotated by 90 degrees, so one solver
gives both bounds. On the checkerboard the discrete Keller identity L_h U_h = ab holds to 1.4e-12 at
every size and contrast tried, because the phase-swapped checkerboard is a translate of itself on an
even grid; the geometric mean of the bracket is therefore exact, while the product of two primal
values is not. A second identity of the same kind, L_h U_h = (1 + 3h)/(3 + h) on the quarter-square
array, holds to 1.2e-13 and is observed, not derived. The four-tile corner exponent
lambda_1 = (4/pi) arctan(kappa^-1/2) sets the Q1 rate 2 lambda_1 for kappa = 4 and 10 (last local rates
1.1809 and 0.7829 against 1.1807 and 0.7799). For kappa = 100 the data alone put the limit at 0.272,
above 2 lambda_1 = 0.2538; only the theory and the pre-asymptotic scale N* = exp(1/lambda_1), about
2.6e3, argue for 2 lambda_1. Fourier-Galerkin with exact disk coefficients brackets the Rayleigh value
rigorously and closes like 1/K, but lopsidedly: the upper bound is about kappa times farther from the
truth than the lower one, so the midpoint is biased (3.7e-3 at K = 64). Staircase disks are not disks
of the same area: in eight of nine cases the certified bracket excludes the Rayleigh value at the pixel
area fraction, consistent with an O(h) perimeter effect but without a certified rate. Apparent tensors
of random windows are ordered (affine >= periodic >= periodic dual >= flux) in all 96 realizations.

**02. The effective permittivity as a Stieltjes function.** Three discretizations give finite measures
that reproduce their own solvers to rounding. The disk array at f = pi 0.3^2 has one atom at
z = 0.3552494 carrying 0.977158 of f (Maxwell Garnett: 0.3586283), shifted by the dipole-octupole
coupling. The square of side 1/2 has the exact Mortola-Steffe value sqrt((1 + 3h)/(3 + h)), whose
measure is the checkerboard density mapped onto exactly [1/4, 3/4]: the whole Perfekt-Putinar band of a
90-degree corner is charged (for this cell), with variance 1/64, which the Fourier-Galerkin measures
approach (0.01766 at K = 24) while the equal-area disk stays near its single atom. Discrete measures are
Rayleigh-Ritz compressions: their atoms do not converge individually, and the large Q1 atom of the
square at z = 1/4 is the checkerboard's spurious percolation atom relabeled. That atom comes from the
node shared by diagonal tiles; its inverse grows like (4/pi) ln N, a slope derived by a
spreading-resistance argument (a free-slope fit up to N = 256 returns 1.273231 against
4/pi = 1.273240). The checkerboard's exact moments
are Catalan numbers over 2 * 4^k and its Jacobi coefficients those of the fourth-kind Chebyshev
polynomials; the sampled scheme reproduces the first ones exactly and the next ones as N grows. For
Drude inclusions every microstructure obeys integral omega Im eps* d omega = (pi/2) f omega_p^2. The
lossless checkerboard and the lossless quarter square absorb (the square in a continuous band
0.5 < omega/omega_p < 0.866), which no finite discretization reproduces: at fixed resolution the
absorption vanishes with the loss, so the limits do not commute. The exact values lie inside the
Bergman-Milton regions, whose isotropic lens has corners at Maxwell Garnett and at the {0, 1} measure.
Real contrasts of one sign determine 3 to 6 numbers of the measure (30 dB power to 1e-3 of the largest
singular value), both signs 6 to 11, lossy Drude spectra 12 to 75: moving the probe along the support
helps, adding contrasts in the same range does not.

**03. Which tensors two phases can make.** The trace bounds of the 2D G-closure, in the form used here,
equal Grabovsky's form; rank-2 laminates lie on them, and 300 random cells are certified inside. The
envelope-theorem gradient of the Q1 tensor equals `jax.grad` to 2.7e-15. Gradient-based design of a
single C4v periodic inclusion at weak fraction 0.7 gives a rounded square whose certified gap to the
Hashin-Shtrikman upper bound is [0.55%, 0.67%] of the HS range at N = 128, against 2.9% for a square and
23.85% for the weak-disk array, with a nearly uniform interior field (vector coefficient of variation 0.037,
against 0.093 for the square and 0.246 for the disk), as for a periodic E-inclusion. A smooth
superellipse of exponent 8 does as well ([0.54%, 0.66%], 0.036), so the optimizer finds nothing a
one-parameter shape does not, and the computation excludes only single-scale saturation above 0.67% of
the HS range; it does not decide attainment, which Liu (2010) proves for periodic E-inclusions. For an
anisotropic boundary point the optimizer, started from a rectangle, converges to a rounded rectangle
certified within 0.75% of the objective range (the rank-1 laminate: 3.87%), on the trace curve but not
at the targeted point; generic starts end in laminate-like local optima.

**04. Two scales on a finite domain.** For the torsion problem with an oscillating coefficient,
||u_eps - u_0|| in L^2 is O(eps) for every way the boundary cuts the cell, but the error of the
first-order expansion depends on the cut. When the boundary lies on mirror lines of the cell the
corrector's normal component vanishes there and the expansion meets the boundary condition: the H^1
error is O(eps) and the interior error O(eps^2), not the textbook eps^(1/2) and eps. For a generic cut
the interior error is O(eps) and is predicted, to 7% at N = 32, by the harmonic extension of the
boundary-layer tails (0.13554 and 0.16978); the H^1 rate keeps falling (0.548 at N = 64), consistent
with the textbook 1/2 but not yet observing it. The boundary layer is a static evanescent Bloch wave
whose decay is an eigenvalue of a quadratic eigenproblem on one cell. Heterogeneity thins it at strong
contrast, but for 0.77493 < a_in < 1.23907 the smallest exponent lies below the homogeneous one (first-order
splitting of a degenerate pair), and for the parity a generic edge excites the layer is thicker for
1 < a_in < 1.23907. The div-curl lemma is visible: the flux energy converges to sigma*|grad u_0|^2 and
the plain gradient energy to grad u_0 . M grad u_0 with M = 1.504546, not sigma*, both at rate 2 when
the same quadrature is used on both sides.

**05. Band edges.** A plane-wave solver with circular cutoffs keeps the little group exactly (a
parallelogram cutoff splits the honeycomb Dirac point by 1.1e-6). At M of the square TM rod lattice the
E doublet has a 2x2 k.p matrix with exactly the three constants that dim Hom_C4v(Sym^2 V, End E) = 3
allows and no linear term (dim Hom_C4v(V, End E) = 0); it predicts the bands with a relative error that
scales as q^2, and the pseudospin winds twice. At K of the honeycomb lattice dim Hom_C3v(V, End E) = 1
gives one Dirac velocity, the equal-rod loop phase is pi, and for unequal rods the massive-Dirac loop
formula misses by up to 0.0345 pi at three times the gap radius, about half of it from the part of the
Bloch vector outside the doublet. In a finite 1D stack with both walls at the same cell phase every
in-band mode sits exactly at k = (N - j) pi/(Na), whatever the wall phase: the envelope equation needs
no end correction (error N^-4), and the wall phase decides only the edge and gap modes. Walls at
different cell phases need an end correction, 0.0595a in the geometric length, from the exact condition
k N a + Theta(omega) = n pi.

**06. What TM scattering sees.** For TM the leading homogenized permittivity of any microstructure is
the volume average, and the next term is exact in the contrast: eps_eff(k0) = <eps> +
k0^2 sum_{G != 0} |eps_G|^2/|G|^2 + O(k0^4), Rytov's formula in 1D; with a C4 axis, direction dependence
first appears at O(k0^6). At Born order the testbed's ring data see spatial frequencies only up to
2 k0, so a sub-lambda/2 grating under a smooth window is nearly invisible (gain ratio 0.0035) while a
hard-edged patch of it is not (0.137 to 0.148): at first order the boundary is what sees the microstructure. For
concentric rings the termination acts at O(p) exactly as a radius shift, which a homogeneous fit
absorbs. The part of the fitted permittivity shift that is even in the contrast equals the Rytov term
(1.0008 to 1.0029 over 11 structures); the odd part, a boundary term linear in the contrast, ranges from
-0.66 to +0.71 times Rytov with the core phase, the mean and the radius. So a homogeneous fit reports
eps_eff(k0) plus a structure-dependent boundary term. Against the volume average it is biased by more
than 3 sd from p = 0.046, before the misfit flag fires; against eps_eff(k0) its error stays within 2 sd
(3.5 sd for the layer-swapped structure) up to the flag in the integer-period family, but reaches about
4 sd at p = 0.07 with another core phase. For a lattice-filled disk the part of the model error in the
Q4 and Q8 rotation sectors of the acquisition is orthogonal to every tangent of the disk model, so it
raises the misfit and never biases the estimate (the rotation analogue of `docs/DESIGN.md` 15.2). An
approximation-error model trained on random terminations transfers across that family but not across
period or contrast.

In one sentence: the effective tensor is the value of a pair of dual convex programs whose
discretizations certify it from both sides and keep 2D duality exactly, its contrast dependence is a
Stieltjes function whose measure carries the corners and the plasmons, single periodic cells come within
a certified 0.7% of the Hashin-Shtrikman bound, the two-scale expansion is limited on a finite domain by
how the boundary cuts the cell, the little group fixes the envelope equation at a band edge, and a TM
imaging experiment measures the second-order effective permittivity plus a termination term rather than
the volume average.

## Running

Use the `photonics-dl` environment. The notebooks need numpy, scipy, matplotlib and JAX (CPU, float64);
notebook 06 also imports the project package `bayes_scatter` from `../../src` (the checkout it lives
in, not an installed copy).

```bash
cd notebooks/homogenization && python _build/selftest_modules.py
```

```bash
cd notebooks/homogenization && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- `_build/selftest_modules.py` runs the seven module test scripts `_build/test_*.py` (about a
  minute); run it after any change to a module.
- The notebooks are generated from the percent-format sources in `_build/`:
  `_build/build.sh nb02 02_spectral_measure` rebuilds and executes one. Editing an `.ipynb` directly
  puts its source out of date.
- The first code cell is the same in all six: thread caps (four, set only if the environment does not
  set them; `NPROC` sizes the XLA pool), the figure style of the other folders, `check(name, value,
  tol, target)` and `check_in(name, value, lo, hi)`, both of which print PASS or FAIL and assert.
  The six notebooks execute 764 such checks; a rerun that changes a result stops with an error.
- A few printed values depend on the thread count through summation order (noted where it matters,
  for example the round-off floor of the Fourier-Galerkin atom at z = 0 in notebook 02).

## Conventions

- Notebooks 01 to 04 are static: a scalar conductivity sigma(x) on the torus R^2/Z^2, which is the
  in-plane permittivity of the TE (H along z) static problem. Lengths are in units of the period.
  Pixel fields are arrays a[i, j] with i the x index. The 90-degree rotation is R = [[0, -1], [1, 0]].
- Notebooks 05 and 06 use time dependence exp(-i omega t) and TM for E along the invariant axis z, as
  in `docs/DESIGN.md`. Notebook 05 measures lengths in units of the period and frequency as a/lambda;
  notebook 06 uses lambda0 = 1 and the testbed geometry of `docs/DESIGN.md` section 2.5.
- Bergman's variable is s = 1/(1 - eps1/eps2) with phase 1 the inclusion, and F(s) = 1 - eps*/eps2 =
  integral dmu(z)/(s - z) with mu >= 0 on [0, 1].
- American English.

## How the notebooks were checked

- **Inside the notebooks.** Every headline number has a reference that does not share its code path:
  certified two-sided brackets (Q1 primal and dual; Fourier-Galerkin with exact coefficients), exact
  solutions (laminates, the checkerboard, the quarter-square array, the Rayleigh multipole series,
  Rytov's formula, the four-tile corner exponents), second solvers (plane waves against the multipole
  solver of `../spatial_dispersion/rodlattice.py`; multilayer Mie against the testbed's forward model;
  `jax.grad` against the envelope-theorem gradient), and theorems whose conditions are stated where
  they are used. Tolerances fixed after seeing a result are listed in each notebook's caveats.
- **Modules.** `_build/selftest_modules.py` runs the seven test scripts, which reproduce every
  validation item of the design with the stated tolerances.
- **Process.** `PLAN.md` records the design: three independent plans, a judge who prototyped the
  key computations, an independent premise review of the plan and a bibliography check, and the
  corrections they caused. After the build, each notebook was audited by a reviewer who recomputed
  its headline numbers with separate code, and a senior reviewer tested whether the conclusions
  survive other readings of the same numbers. No certified number was found wrong; the corrections
  were to interpretations, weak checks and missed opportunities, and they are in the notebooks.
- **Expectations that the computations contradicted** (recorded before the computation in `PLAN.md`,
  kept in the notebooks): the spurious atom of a Fourier-Galerkin measure at z = 0 decays like
  exp(-2.6 K), not like 1/K; the "untestable" corner band of the square is fully charged, by an exact
  solution the design had missed; the reference values for the rod lattice had been computed at a
  rounded volume fraction; a weakly contrasting cell can make the boundary layer thicker, not thinner;
  in a 1D stack the in-band modes are quantized exactly for any wall position, not only at mirror
  planes; the massive-Dirac loop formula misses by 0.0345 pi at three times the gap radius; a
  homogeneous fit does recover the bulk Rytov term (in the part even in the contrast), and the
  "biased before detectable" window of the design is mostly the second-order effective permittivity
  being measured correctly.

## Known limits

- Two dimensions, two phases, scalar problems (the static conductivity problem and TM). No
  three-dimensional cell problems, no anisotropic phases, no vector Maxwell homogenization in 3D.
- **01.** The kappa = 100 rate is not pinned. The staircase shape effect has no certified rate. The
  quarter-square identity of the discrete pair is observed, not derived. Random windows go up to 32 by
  32 tiles with at most 8 pixels per tile.
- **02.** The corner band is tested on one cell, through its exact solution, not for polygons in
  general. The constant of the percolation atom (1/mu_0 = c + (4/pi) ln N) is fitted, not derived. The
  Lanczos coefficients alpha_k for k >= 5 do not converge over N <= 256. The Bergman-Milton lens is not
  the sharp isotropic region. The information counts come from one linearization without a prior.
- **03.** Attainment of the Hashin-Shtrikman bound by a single inclusion is not decided (only
  saturation above 0.67% of the HS range is excluded, at N <= 128); one fraction (0.7) and one contrast
  (10); the anisotropic target point is not reached, and generic starts end in local optima.
- **04.** One family of D4 cells (a disk, and at the coarser resolution a square) and one right-hand
  side. The eps^(1/2) asymptote of the generic cut is not observed. The exponent margins against the
  homogeneous cell are not converged in the pixel resolution.
- **05.** No two-dimensional finite-crystal test of the envelope equation. The Dirac-point theorem of
  Fefferman and Weinstein is for -Delta + V; for the TM problem the symmetry count is what transfers.
  The attribution of the doublet-only deviation to trigonal warping is a conjecture.
- **06.** The odd boundary term of the ring fits is not derived; only its symmetry, scaling and
  first-order character are shown. The lattice section asserts the selection rule, not the size of the
  bias. The approximation-error model is tested on one family.
- Each notebook ends with its own caveats, including the tolerances set after seeing a result.

## Literature

Entries marked (checked) had their bibliographic details confirmed against publisher, DOI, arXiv or
library records on 2026-11-23; that does not mean their content was read beyond the notes in
`PLAN.md` ("Literature (verified)"), which also says what each source actually states.

- Books. A. Bensoussan, J.-L. Lions, G. Papanicolaou, *Asymptotic Analysis for Periodic Structures*
  (North-Holland, 1978) (checked); V. V. Jikov, S. M. Kozlov, O. A. Oleinik, *Homogenization of
  Differential Operators and Integral Functionals* (Springer, 1994) (checked); G. W. Milton, *The Theory
  of Composites* (Cambridge University Press, 2002) (checked); G. Allaire, *Shape Optimization by the
  Homogenization Method* (Springer, 2002) (checked); L. Tartar, *The General Theory of Homogenization: A
  Personalized Introduction* (Springer, 2009) (checked).
- Variational bounds and duality. Z. Hashin, S. Shtrikman, J. Appl. Phys. 33, 3125 (1962); J. B.
  Keller, J. Math. Phys. 5, 548 (1964); A. M. Dykhne, Sov. Phys. JETP 32, 63 (1971); K. S. Mendelson,
  J. Appl. Phys. 46, 4740 (1975); C. Huet, J. Mech. Phys. Solids 38, 813 (1990) (all checked).
- Exact solutions and corners. Lord Rayleigh, Phil. Mag. 34, 481 (1892); W. T. Perrins, D. R. McKenzie,
  R. C. McPhedran, Proc. R. Soc. Lond. A 369, 207 (1979); R. B. Kellogg, in *Numerical Solution of
  Partial Differential Equations II* (Academic Press, 1971), 351 (all checked); R. V. Craster,
  Yu. V. Obnosov, SIAM J. Appl. Math. 61, 1839 (2001) and G. W. Milton, J. Math. Phys. 42, 4873 (2001)
  (checked, bibliographic records only), the proofs of the conjecture of S. Mortola and S. Steffe,
  Atti Accad. Naz. Lincei (1985) (from memory).
- Spectral representation and complex bounds. D. J. Bergman, Phys. Rep. 43, 377 (1978) and Phys. Rev.
  Lett. 44, 1285 (1980); G. W. Milton, Appl. Phys. Lett. 37, 300 (1980) and J. Appl. Phys. 52, 5286
  (1981); K. Golden, G. Papanicolaou, Commun. Math. Phys. 90, 473 (1983); K.-M. Perfekt, M. Putinar,
  Arch. Ration. Mech. Anal. 223, 1019 (2017); E. Bonnetier, H. Zhang, Rev. Mat. Iberoam. 35, 925
  (2019) (all checked).
- G-closure and optimal microstructures. K. A. Lurie, A. V. Cherkaev, Proc. Roy. Soc. Edinburgh A 99,
  71 (1984) and 104, 21 (1986); S. B. Vigdergauz, J. Appl. Mech. 61, 390 (1994); Y. Grabovsky, R. V.
  Kohn, J. Mech. Phys. Solids 43, 949 (1995); L. P. Liu, R. D. James, P. H. Leo, Metall. Mater. Trans.
  A 38, 781 (2007) (all checked); L. P. Liu, Proc. R. Soc. A 466, 3693 (2010) (checked; attainment of the
  HS bounds by periodic E-inclusions); Y. Grabovsky, Proc. Roy. Soc. Edinburgh A 123, 423 (1993)
  (checked; the trace bounds in any dimension); L. Tartar, "Estimations fines des coefficients
  homogeneises", Pitman Res. Notes Math. 125, 168 (1985) (details from reference lists).
- FFT and Galerkin cell solvers. H. Moulinec, P. Suquet, Comput. Methods Appl. Mech. Engrg. 157, 69
  (1998); J. Zeman, J. Vondrejc, J. Novak, I. Marek, J. Comput. Phys. 229, 8065 (2010) (both checked);
  J. Vondrejc, J. Zeman, I. Marek, Comput. Methods Appl. Mech. Engrg. 297, 258 (2015) (checked;
  guaranteed bounds).
- Two-scale analysis and boundary layers. G. Allaire, SIAM J. Math. Anal. 23, 1482 (1992); F. Murat,
  L. Tartar, "H-convergence", in *Topics in the Mathematical Modelling of Composite Materials*
  (Birkhauser, 1997), 21; G. Allaire, M. Amar, ESAIM: COCV 4, 209 (1999); S. Moskow, M. Vogelius, Proc.
  Roy. Soc. Edinburgh A 127, 1263 (1997); D. Gerard-Varet, N. Masmoudi, Acta Math. 209, 133 (2012) (all
  checked).
- Band edges. R. V. Craster, J. Kaplunov, A. V. Pichugin, Proc. R. Soc. A 466, 2341 (2010); C. Conca,
  M. Vanninathan, SIAM J. Appl. Math. 57, 1639 (1997); G. Allaire, C. Conca, J. Math. Pures Appl. 77,
  153 (1998); C. L. Fefferman, M. I. Weinstein, J. Amer. Math. Soc. 25, 1169 (2012) (all checked).
- Imaging. S. M. Rytov, Sov. Phys. JETP 2, 466 (1956); E. Wolf, Opt. Commun. 1, 153 (1969); J. Kaipio,
  E. Somersalo, *Statistical and Computational Inverse Problems* (Springer, 2005) and J. Comput. Appl.
  Math. 198, 493 (2007) (all checked).
- In this repository: `docs/DESIGN.md` sections 2 and 15 (the testbed and the detectability law);
  `notebooks/spatial_dispersion/rodlattice.py` (the multipole reference for the bands);
  `notebooks/cloaking/04_partial_waves_and_scattering_cancellation.ipynb` (the multilayer Mie code that
  `scatter.py` adapts).
