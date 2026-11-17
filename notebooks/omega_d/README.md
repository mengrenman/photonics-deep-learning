# The Ωd problem: numerical stability of matrix methods for multilayers

Four notebooks on one numerical problem. The transfer matrix of a layer in which a wave is
evanescent holds exp(+kappa d) and exp(-kappa d) together, and in finite precision the small one is
lost when kappa d is large. Because kappa grows with frequency this is known as the "Ωd problem" (the
name is from Pérez-Álvarez and García-Moliner) or the "large fd problem" of ultrasonics, and the
usual advice is to replace the transfer matrix by a scattering, stiffness or hybrid matrix. The
notebooks ask what exactly is lost, in which problems it matters, why the replacements work, and
what else they are good for. Every double-precision result is compared with the same model
evaluated in multiprecision arithmetic. Read them in order. Each one runs on its own.

| # | Notebook | Question | Method | Run time |
|---|---|---|---|---|
| 01 | `01_one_channel` | For isotropic layers and one polarization, which results of the transfer matrix are wrong, and from what thickness? | 2x2 transfer matrix in several arrangements against multiprecision; frustrated total reflection, a double barrier, random stacks, Bragg mirrors, metal stacks | 28 s |
| 02 | `02_many_channels` | Why is the problem serious with two or more channels, and what do the cures have in common? | A uniaxial slab (4x4) and a lamellar grating (Fourier modal method); transfer matrix against scattering matrix, enhanced transmittance, orthonormal frames, Plücker coordinates, global matrix | 6 s |
| 03 | `03_charts` | How are the transfer, stiffness, hybrid and scattering matrices related, and where does each fail? | The four matrices of a layer and their composition rules; symmetry checks; sweeps over thickness with one and two channels | 5 s |
| 04 | `04_modes` | What happens to the secular function of guided modes, and can modes be counted instead of searched for? | Secular functions from the transfer matrix and from Plücker coordinates; mode counting with stiffness matrices; coupled guides | 37 s |

Run times are for an M3 Max with four threads. `multilayer.py` holds all the solvers and is shared
by the four notebooks.

## Main findings

**01. One channel.** A transfer matrix through an evanescent layer is, past kappa d = (1/2) ln(1/eps)
= 18.02, a rank-one matrix in double precision: its determinant has an error eps exp(2 kappa d)
(fitted exponent 2.000), and its entries overflow at kappa d = 709.8. Reflection and transmission
do not need the lost part. From either convention of the product they are accurate to 1e-13 up to
the overflow, provided the determinant is used as a fact and not computed; the one arrangement
that fails is the transmission by back-substitution, at kappa d = 18. Carrying one vector from the
exit side, rescaled by powers of two, removes the overflow as well, and returns the transmission
as a mantissa and an exponent where a scattering matrix underflows. The reason is a contraction:
from the exit side the physical solution is the one a barrier attracts (errors shrink by
exp(-2 kappa d)), from the entrance side the one it repels, so fields by forward marching are
wrong beyond a depth 18.02/kappa. There is a second, milder loss: with the layer applied as
cosines and sines, a resonance hidden behind a barrier costs the reflection coefficient a factor
1/detuning in accuracy (353 times the error of the alternative over the range measured), while
applying the layer in its own two waves does not. With these arrangements the recursion on a
vector was as accurate as the scattering matrix in every test, including stacks of 3000 random
layers and a cavity between Bragg mirrors of 1200 pairs each, where its error stayed within a
factor 14 of the larger of the intrinsic sensitivity and 1e-14.

**02. Many channels.** With two or more channels the transfer matrix fails for results of order
one. For a uniaxial slab between prisms in which one wave propagates and the other is evanescent,
the transmitted power is between 0.47 and 0.92 at every thickness, and both conventions return it
with an error eps exp(kappa d) (fitted exponent 0.97) that reaches 0.1 at kappa d = 36.9, a slab of
15 wavelengths. What is lost is the plane spanned by the n outgoing solutions: carried as n
vectors they align with the fastest-growing wave, the condition number of the frame grows as
exp(kappa d), and the plane it determines is off by eps times that. The plane itself converges.
Every stable method carries the plane without the frame: in a chart (the reflection matrix: the
scattering-matrix and enhanced-transmittance recursions), as an orthonormal frame, or in Plücker
coordinates (the compound or delta matrix), and a global linear system does the same job by
elimination. All of these stay below 2.2e-13 to kappa d = 712. For a lamellar grating in the
Fourier modal method the number of channels is the truncation order, and the transfer matrix
loses the zeroth-order efficiency when kappa_max d is between 41 and 49 for the three depths
tried: order 23, 10 and 5 for depths of 0.5, 1 and 2 wavelengths.

**03. Charts.** The transfer, stiffness, hybrid and scattering matrices describe one linear
relation between the fields on two faces as a graph over four different choices of independent
variables, with the matrix product, a Schur complement, and (for both hybrid and scattering) the
Redheffer star product as composition rules. For a lossless stack the relation is a Lagrangian
subspace; the transfer matrices are the part of the Lagrangian Grassmannian that leaves out the
opaque limit, the stiffness and hybrid matrices are affine charts, and the flux-normalized
scattering matrix (symmetric and unitary, verified to 4e-15) is a coordinate on all of it. Each
variant fails where its chart is singular: the stiffness matrix for thin layers, with an error
eps/phi, and at layers a whole number of half-waves thick; the hybrid matrix at odd quarter-waves;
both lose digits as eps/delta near those thicknesses and return an error of order 0.1 at them. Over
fifteen decades of a common thickness scale the hybrid and scattering matrices stay below 1.2e-13
with one and with two channels, which is the conclusion of Pérez-Álvarez, Pernas-Salomón and
Velasco. The scattering-matrix recursion in layer amplitudes divides by zero exactly at the pole
of a single interface (a surface plasmon of a lossless metal) and is accurate one rounding unit
away from it.

**04. Modes.** With one channel the secular function 1/t from the rescaled vector is accurate, and
what fails is the search: two supermodes of coupled guides closer than the search grid give no
sign change. With two channels the secular determinant from the transfer matrix has Dunkin's
cancellation, an error eps exp((kappa_1 - kappa_2) D) (fitted exponent 1.03) that reaches 0.1 at
(kappa_1 - kappa_2) D = 36.8, and at a buffer of 34 wavelengths it shows 429 sign changes on a grid
where there are 4 modes; from Plücker coordinates it is accurate to 2.4e-13 up to 60 wavelengths.
Modes of dielectric stacks can be counted: the number above a given effective index is the number
of negative pivots of the assembled stiffness matrix, with layers cut thin enough to have no modes
of their own (the Wittrick-Williams algorithm, a matrix form of Sturm's oscillation theorem).
Bisection on the count reproduced every mode found by other means to 7e-16 or better and gives
multiplicities. For two coupled guides it follows the splitting from 3e-2 to 4e-16 along
exp(-kappa g) (slope 1.000 in units of kappa) and then reports one mode of multiplicity two. For
two guides coupled through a birefringent layer the splitting of the first pair of hybrid
supermodes fits A exp(-kappa_e D) - B exp(-kappa_o D) with both coefficients positive (residual
5e-5) and vanishes at D = 2.512 wavelengths, where the two supermodes cross.

In one sentence: the Ωd problem is the loss of a plane of solutions through an ill-conditioned
frame, which for one channel can be removed by arranging the algebra and for more needs a method
that carries the plane itself, and the stiffness matrix that such methods bring along also counts
the modes.

## Running

Use the `photonics-dl` environment with one addition, `mpmath`, which provides the multiprecision
reference (`pip install mpmath`; pure Python). The notebooks use numpy, scipy, matplotlib and
mpmath, and do not use JAX or `bayes_scatter`.

```bash
cd notebooks/omega_d && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- The first code cell is the same in all four: thread caps, the figure style of the other notebook
  folders, and the `check` helper.
- `check(name, value, tol, target)` prints PASS or FAIL and asserts. The four notebooks hold 97 such
  checks plus plain assertions; a rerun that changes a result stops with an error.
- The notebooks are generated from the percent-format sources in `_build/`:
  `_build/build.sh nb03 03_charts` rebuilds and executes one. Editing an `.ipynb` directly puts
  its source out of date.

## Conventions

- Time dependence exp(-i omega t). Lengths in free-space wavelengths, lambda_0 = 1 and k_0 = 2 pi, as
  in `docs/DESIGN.md`, so a thickness d is Omega d / (2 pi c). Layers are stacked along z, the plane
  of incidence is xz, and beta = k_x / k_0.
- Polarizations are called s (E along y) and p (H along y). The s polarization is what
  `docs/DESIGN.md` calls TM for structures invariant along the field; s and p are used here to avoid
  the clash with the usual naming for layers.
- One channel: field u (E_y or H_y) and v = u'/w with w = 1 for s and the permittivity for p; a
  forward wave has v = i eta u. Many channels: the 2n continuous field components and the modes of
  each layer, with "forward" meaning decaying or carrying power toward +z. Anisotropic layers use
  (E_x, E_y, H_x, H_y) with the vacuum impedance set to one.
- The reference takes k_0 as the double nearest to 2 pi, and all inputs as the doubles the other
  routines receive, so that it solves the same problem.

## How the notebooks were checked

- Against multiprecision arithmetic: every error quoted is a distance from the transfer-matrix
  formulas evaluated with mpmath at enough digits to make exp(2 kappa d) harmless (matrix
  exponentials for the many-channel problems). For the grating this covers 7 and 11 Fourier orders.
- Independent of the reference: wavenumbers of the uniaxial medium against closed forms; isotropic
  layers through the 4x4 code against the one-channel code; flux balance and reciprocity; the
  two-channel mode count against the sum of the one-channel counts for an isotropic layer; the
  count against the number of zeros of the decaying solution (Sturm); modes from the count against
  zeros of the secular function, and a sign change of the multiprecision secular function across
  each.
- Several statements made before the numbers were seen were wrong. Four were written into a
  notebook and caught by their own checks failing: that marching forward expands errors for any
  starting direction (it expands them only next to the decaying wave, which is where the physical
  field is); that the logarithm of the transmission is an adequate way to avoid underflow (it costs
  eps |ln t|, so the code keeps a mantissa and a binary exponent); that the sign in the reciprocity
  relation between cross-polarized transmissions is negative (it is positive in these amplitudes);
  and that the splitting of hybrid supermodes decays at the slower of the two rates of the layer
  (it is a difference of two exponentials with a zero). Two were corrected on seeing the output:
  that the error near a singular thickness of the stiffness matrix goes as the inverse square of
  the distance (it is the inverse), and that the grating results for different depths fall on one
  curve (they share an onset to within the spread printed).
- Not done: no independent implementation and no review by anyone but the author of the code. The
  tolerances of several checks (the factor allowed between an error and the intrinsic sensitivity,
  the onset of failure for the grating) were set after the results were seen and are labeled so in
  the notebooks.

## Known limits

- **01.** The "intrinsic sensitivity" is the response of the exact answer to one kind of
  perturbation, a relative rounding of each thickness. Gain media and exactly degenerate cases such
  as a vanishing normal wavenumber were not tried. The optical constants in the table of
  thresholds are round numbers.
- **02.** One anisotropic configuration and one grating, in one polarization. The factor by which
  weak coupling delays the failure for the grating was observed and not derived. Beyond 11 orders
  the grating results rest on energy balance and on the agreement of four stable methods that
  share one eigendecomposition. No cost comparison.
- **03.** The geometric picture is supported by the numerical checks and by standard facts (the
  theorem of Arnold, the Lagrangian graph of a symplectic map) and was not written out as a proof;
  I did not find it stated in this form in the references checked. The scattering matrix was
  always computed in layer amplitudes, not relative to a fixed propagating reference medium.
  Singular sets with several channels were not mapped.
- **04.** Lossless dielectrics only: with metals the count loses its meaning in p polarization, and
  complex (lossy or leaky) modes were not treated. The two-channel count assumes eps_xz = eps_yz = 0.
  The rule for cutting layers is exact for uncoupled channels and was checked by refinement, not
  proved, for the coupled case. The crossing of the hybrid supermodes rests on a fit with the sign
  of the splitting assigned by hand; the parities of the two modes were not computed.
- Each notebook ends with its own list of caveats and open ends.

## Literature

Entries marked (checked) had their bibliographic details confirmed against publisher, arXiv or
index records on 2026-11-17. Of the contents, only the arXiv version of the first entry was read;
the others are cited for what they are generally known for.

- R. Pérez-Álvarez, R. Pernas-Salomón and V. R. Velasco, "Relations between transfer matrices and
  numerical stability analysis to avoid the Ωd problem", SIAM J. Appl. Math. 75, 1403 (2015),
  arXiv:1503.09038 (checked). The relations between the associated transfer, stiffness, compliance,
  hybrid and scattering matrices for matrix Sturm-Liouville problems, and which avoid the problem.
  According to it the name was coined in R. Pérez-Álvarez and F. García-Moliner, *Transfer Matrix,
  Green Function and Related Techniques* (Universitat Jaume I, 2004), which was not consulted.
- J. W. Dunkin, "Computation of modal solutions in layered, elastic media at high frequencies",
  Bull. Seismol. Soc. Am. 55, 335 (1965) (checked); E. N. Thrower, "The computation of the
  dispersion of elastic waves in layered media", J. Sound Vib. 2, 210 (1965) (checked). The loss of
  significance in the secular function and the delta (compound) matrix.
- M. J. S. Lowe, "Matrix techniques for modeling ultrasonic waves in multilayered media", IEEE
  Trans. Ultrason. Ferroelectr. Freq. Control 42, 525 (1995) (checked). Review, including the
  global matrix.
- S. I. Rokhlin and L. Wang, "Stable recursive algorithm for elastic wave propagation in layered
  anisotropic media: Stiffness matrix method", J. Acoust. Soc. Am. 112, 822 (2002) (checked).
- E. L. Tan, "Hybrid compliance-stiffness matrix method for stable analysis of elastic wave
  propagation in multilayered anisotropic media", J. Acoust. Soc. Am. 119, 45 (2006) (checked).
- D. Y. K. Ko and J. C. Inkson, "Matrix method for tunneling in heterostructures: Resonant
  tunneling in multilayer systems", Phys. Rev. B 38, 9945 (1988) (checked); D. Y. K. Ko and J. R.
  Sambles, "Scattering matrix method for propagation of radiation in stratified media: attenuated
  total reflection studies of liquid crystals", J. Opt. Soc. Am. A 5, 1863 (1988) (checked). The
  scattering-matrix recursion; the second is the two-channel anisotropic case of notebook 02.
- M. G. Moharam, D. A. Pommet, E. B. Grann and T. K. Gaylord, "Stable implementation of the
  rigorous coupled-wave analysis for surface-relief gratings: enhanced transmittance matrix
  approach", J. Opt. Soc. Am. A 12, 1077 (1995) (checked).
- L. Li, "Formulation and comparison of two recursive matrix algorithms for modeling layered
  diffraction gratings", J. Opt. Soc. Am. A 13, 1024 (1996) (checked); "Note on the S-matrix
  propagation algorithm", J. Opt. Soc. Am. A 20, 655 (2003) (checked).
- R. Redheffer, "On the relation of transmission-line theory to scattering and transfer", J. Math.
  and Phys. 41, 1 (1962) (checked). The star product.
- W. H. Wittrick and F. W. Williams, "A general algorithm for computing natural frequencies of
  elastic structures", Q. J. Mech. Appl. Math. 24, 263 (1971) (checked). Counting eigenvalues with
  the dynamic stiffness matrix.
- V. I. Arnold, "Characteristic class entering in quantization conditions", Funct. Anal. Appl. 1, 1
  (1967) (checked). The Lagrangian Grassmannian as U(n)/O(n) and the Maslov index.
- From memory, not checked: L. G. Parratt, Phys. Rev. 95, 359 (1954), the recursion for the
  reflection of a stratified medium; M. Born and E. Wolf, *Principles of Optics*, section 1.6, the
  characteristic matrix; D. W. Berreman, J. Opt. Soc. Am. 62, 502 (1972), the 4x4 formulation for
  anisotropic layers; F. V. Atkinson, *Discrete and Continuous Boundary Problems* (Academic Press,
  1964), oscillation theory for matrix systems.
