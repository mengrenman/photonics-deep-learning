# Spatial dispersion and optical magnetism

Four notebooks on one question from metamaterials research: when a structure made of nonmagnetic
parts is said to have a magnetic permeability at optical frequencies, what is that permeability, and
is it more than a wavevector dependence of the permittivity? The starting point is the argument of
Landau and Lifshitz that the permeability loses its meaning at high frequency and that everything
can be carried by one tensor, eps(omega, k). Notebook 01 is the theory. Notebooks 02 to 04 compute
it all exactly for a square lattice of high-index dielectric rods in two dimensions. Read them in
order. Each one runs on its own.

| # | Notebook | Question | Method | Run time |
|---|---|---|---|---|
| 01 | `01_one_tensor_or_two` | What part of second-order spatial dispersion is a permeability, and what does the bulk tensor not contain? | Exact algebra checked on random media, group averaging, transfer matrices | 6 s |
| 02 | `02_rod_lattice_multipoles_and_bands` | Where do the magnetic resonances of the rods fall in the band structure? | Mie coefficients, lattice sums, a Hermitian multiple-scattering mode condition | 45 s |
| 03 | `03_nonlocal_permittivity_of_the_lattice` | What is eps(omega, k) of the lattice, and how much of its k^2 term is a permeability? | Source-driven homogenization by multiple scattering, Taylor coefficients by Cauchy integrals | 16 s |
| 04 | `04_interfaces_and_retrieval` | Does the bulk permeability give the reflection of a slab? What does a retrieval return? | Exact slabs of line arrays, parameter retrieval | 8 s |

Run times are for an M3 Max with four threads. `rodlattice.py` is the solver shared by notebooks
02 to 04 (and used for one Mie check in 01).

## Main findings

**01. One tensor or two.** Local permittivity, permeability and magnetoelectric couplings map
exactly onto one tensor eps(omega, k): order k^0 is the permittivity, order k^1 is gyrotropy, and
the permeability is the term K^T (1 - 1/mu) K / k0^2 at order k^2 (K is the cross-product matrix of
k). Free waves in the bulk fix only the index; every split of n^2 into eps and mu gives the same
tensor on the dispersion curve. By symmetry, a square lattice with E in the plane has three
second-order coefficients, of which one (tau) is a permeability, one (lambda) is longitudinal, and
one (delta) is an anisotropy that transverse waves see as a permeability varying like cos(4 phi);
with E along the axis every second-order term has the form of a permeability. The Landau-Lifshitz
criterion for a meaningful magnetization becomes an inequality between these coefficients. A
dielectric stack with no magnetism at all has a retrieved permeability below or above one depending
on where it is cut, (mu - 1)/(k0 a)^2 = -0.25 or +0.33 for the example, while its index is one curve.

**02. The lattice.** For nonmagnetic rods the two magnetic-dipole coefficients, m = 0 for H along the rod
(TE) and m = 1 for E along the rod (TM), are the same function of frequency. For rods of radius 0.3 a
the resonance is at a/lambda = 0.36 for a rod permittivity of 12 and at 0.126 for 100. With 12 the
first TE band reaches the zone boundary before the resonance; with 100 the bands have the textbook
shape on the path sampled, a flattening band, a gap from 0.127 to 0.142, and a band starting at
Gamma, and the O'Brien-Pendry permeability with a static permittivity places the first band at X
to 0.2% (0.6% from the top of the band, which is at M).

**03. The nonlocal permittivity.** For TE only tau has the resonance pole, and only tau grows with
the rod permittivity, with exactly the slope of a current loop, eta R^2 / 8, where eta = pi R^2 /
a^2 = 0.283 is the filling fraction (fitted 0.0031826 against 0.0031809). The rest is a background
set mostly by the geometry: delta of about -0.01 a^2, lambda of about 0.004 a^2, and a part of tau
that is not the loop, about -0.015 a^2. The ratio of the loop term to the largest of these, the two
sides of the Landau-Lifshitz inequality, is 2.9 for a rod permittivity of 12 and 19.7 for 100.
This puts numbers on Merlin's "large permittivity begets high-frequency magnetism", with two
qualifications: the ratio is for the static limit, and the loop term is the formula, identified
with a part of tau by its scaling and not separated from the rest by a computation. Far below
resonance the apparent mu - 1 along Gamma-X and Gamma-M differ by 57% of mu - 1 for 12 and by 6.5%
for 100; close to the resonance the two have poles at slightly different frequencies and differ by
any factor. The index anisotropy that delta implies is found in the exact Bloch wavenumbers to 0.4%
at a/lambda = 0.035 and to 10% at 0.165; near the zone boundary the prediction is several times too
large. For TM the k^2 coefficient is isotropic by symmetry and vanishes like omega^2 (0.1% of the
loop estimate at a/lambda = 0.01). At zero frequency this is exact for rods of any cross-section,
because the driven field is then the plane wave of the source; in the bulk response the magnetic
dipole of the rod is not a permeability. The lattice also has Bloch modes that are odd under the
mirror containing k, carry no macroscopic E or B, and do not appear in eps(omega, k); the one near
a/lambda = 0.40 is read as the longitudinal magnetic mode.

**04. Interfaces.** The index retrieved from a slab converges to the Bloch index with thickness
(1.5e-4 at 16 layers, within 0.3% for one layer in the cases printed). The admittance is another
matter. All slabs here have their faces half a period outside the outer rows of rods, and the
numbers that follow are for that termination. For TM at low frequency three procedures give three
permeabilities: positive from the rod's magnetic dipole, zero at order omega^2 from the bulk
expansion, and negative from the slab, (mu - 1)/(a/lambda)^2 = -7.5, independent of the number of
layers. The slab's admittance exceeds the bulk one by 0.046 (qa)^2. This is read as the
Bloch-impedance effect of a lattice cut between its polarizable elements; the lumped-circuit limit
gives 1/8 for the coefficient, 2.7 times more, and the difference was not derived. For TE through
the magnetic resonance with lossy rods, 100 (1 + 0.02 i), the retrieved permittivity has an
antiresonance with an imaginary part down to -1.3, while the bulk permittivity stays between 1.77
and 1.79 with a positive imaginary part below 0.001; just below the resonance the bulk parameters
give 54 to 65% of the exact reflection of eight layers. The ratio of the single-layer impedance to
the bulk impedance falls with qa where the rods act as an electric polarizability and rises, to
4.6 at qa = 2.9, where they act as magnetic resonators. In TE thin slabs also carry a static
surface term proportional to 1/N. Extrapolated to a thick slab at a/lambda = 0.01, the retrieved
permittivity equals the bulk one to 3e-4. The permeability agrees to 1.5e-4, which is the size of
mu - 1 itself for a rod permittivity of 12 and 12% of it for 100, so that test says little about mu.

In one sentence: for H along the rods and a large rod permittivity, a permeability describes the
resonant part of the bulk response to second order in ka, on a direction-dependent background that
is small compared with it away from the poles; the number a slab returns under the same name mixes
that response with the periodicity and the termination, and the two were found to agree only in
the index and, for the permittivity, in the limit of a thick slab at long wavelength.

## Running

Use the `photonics-dl` environment. Nothing beyond its packages is needed (numpy, scipy,
matplotlib; the notebooks do not use JAX or `bayes_scatter`).

```bash
cd notebooks/spatial_dispersion && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- The first code cell is the same in all four: thread caps (four, set only if the environment does
  not set them), the figure style and palette of the cloaking notebooks, and the `check` helper.
- `check(name, value, tol, target)` prints PASS or FAIL and asserts. The four notebooks hold 88
  such checks plus plain assertions; a rerun that changes a result stops with an error.
- The Markdown avoids typed-in results and refers to the printed output instead.
- The notebooks are generated from the percent-format sources in `_build/`; `PLAN.md` says how to
  rebuild one, and records the state in which the work was paused and what is left to do.

## Conventions

- Time dependence exp(-i omega t); outgoing cylindrical waves are Hankel functions of the first
  kind. eps0 = mu0 = c = 1.
- Lengths are in units of the lattice period a, and frequency is a/lambda_0 = k0 a / 2 pi. This
  differs from the rest of the project (lambda_0 = 1) because the folder sweeps frequency at a
  fixed lattice.
- The polarization with E along the rods is called TM and the one with H along the rods TE, as in
  `docs/DESIGN.md`.
- Rods have radius 0.3 a and relative permittivity 12 or 100, lossless unless stated.

## How the notebooks were checked

- Independent methods inside the notebooks: lattice sums against brute-force sums at complex
  frequency (2.5e-11); Bloch frequencies against a plane-wave expansion for TM (3e-5); the driven
  permittivity against a plane-wave solution for TM (2e-4) and against Rayleigh's static series
  for TE (1e-7 for a rod permittivity of 12, 4e-7 for 100); the Bloch modes from the mode condition satisfy k^2 = k0^2 eps(omega, k) from the
  driven calculation (lossless rods, real k, and only the modes that are even under the mirror
  containing k; two odd modes out of nine were excluded after the fact, and both paths share the
  Mie coefficients and lattice sums, so this is a check of consistency and not of the split of n^2
  into eps and mu); flux balance and reciprocity of slabs to
  rounding; the retrieved index of slabs equals the bulk Bloch index, which ties the slab solver to
  the bulk solver.
- Checked in a scratch calculation and not kept in a notebook: the TE driven tensor at finite k
  against an in-plane plane-wave solution, which agrees to 0.2% and converges slowly toward the
  multiple-scattering value.
- Several statements written before the numbers were seen turned out wrong and were caught by
  their own assertions, then corrected in the text: the sign of one term in the bianisotropic
  identity; "the single-rod resonance lies in the gap" (it lies just below the top of the first
  band); "the background coefficients do not change with frequency" (true for a rod permittivity
  of 100, not for 12); "the anisotropy of mu is small near resonance" (small relative to tau, not
  in mu, whose two poles differ).
- On 2026-11-16, after the work was paused, the record was audited without new computation: four
  agents (Sonnet) traced every number and comparative statement in this README and in the notebook
  prose to the stored outputs, a second agent rechecked each reported problem, and one reviewer
  (Opus) read `PLAN.md` and this file as a record. Twelve problems were confirmed, all in wording
  or rounding and none in a computed number, and the reviewer added conditions and reference
  choices that the first version left out. They are corrected here and in the notebooks.
- Still not done: nobody has recomputed a headline number with separate code or tested the
  conclusions with a solver of their own, as was done for the cloaking folder. `PLAN.md` lists what
  most needs it.

## Known limits

- Two dimensions and one lattice (square). The results are for one rod radius (0.3 a) and two rod
  permittivities (12 and 100); a second radius and other permittivities appear only in validation
  checks and in the static scan of notebook 03.
- `rodlattice.py`: the coupling between rows of a slab is tested only through the slab checks, and
  rods with a permeability are not tested at all.
- **03.** The split of tau into a loop part and a background is identified by its dependence on the
  rod permittivity, not derived term by term. Singularities of eps(omega, k) in the complex k plane
  were avoided, not located. Fourth-order coefficients are printed and not analyzed.
- **04.** Normal incidence along a lattice axis and one termination only, so the dependence of
  the slab results on termination is inferred from the one-dimensional stack of notebook 01 and
  not computed. The semi-infinite lattice
  was not solved, so the surface term and the Bloch-impedance effect are separated only at low
  frequency. No additional boundary condition for the nonlocal description was derived. Finite
  bodies cut from the lattice were not computed; `rodlattice.cluster` is written for that and is
  untested.
- Band structures in 02 are sampled on one path and are for orientation; the gap quoted for a
  rod permittivity of 100 was not confirmed on a grid of the whole zone.
- Each notebook ends with its own list of caveats and open ends.

## Literature

The Landau-Lifshitz argument and the replies to it. Entries marked (checked) had their
bibliographic details confirmed against publisher, ADS or arXiv records on 2026-11-16; the text of
the book sections was not available online and section numbers for the book are from memory, with
section 79 corroborated by papers that cite it.

- L. D. Landau, E. M. Lifshitz and L. P. Pitaevskii, *Electrodynamics of Continuous Media*, 2nd ed.
  (Pergamon, 1984). Section 79, on the dispersion of the permeability: the magnetization is
  meaningful only if the magnetization current dominates the polarization current in the magnetic
  moment, which fails from optical frequencies on, so mu = 1 there (page 251 of the 1960 edition,
  as cited by Merlin). Sections 103 to 106: spatial dispersion, with all of the response in
  eps(omega, k), natural optical activity at first order in k, and the second-order terms.
- R. Merlin, "Metamaterials and the Landau-Lifshitz permeability argument: Large permittivity
  begets high-frequency magnetism", Proc. Natl. Acad. Sci. USA 106, 1693 (2009) (checked). The
  argument restated and its loophole: inclusions of large permittivity.
- V. M. Agranovich and Yu. N. Gartstein, "Electrodynamics of metamaterials and the Landau-Lifshitz
  approach to the magnetic permeability", Metamaterials 3, 1 (2009) (checked); "Spatial dispersion
  and negative refraction of light", Phys. Usp. 49, 1029 (2006) (checked).
- V. M. Agranovich and V. L. Ginzburg, *Crystal Optics with Spatial Dispersion, and Excitons*
  (Springer, 1984).
- M. G. Silveirinha, Phys. Rev. B 75, 115104 (2007) (checked). Source-driven homogenization, the
  method of notebook 03.
- J. Skaar, H. O. Hågenvik and C. A. Dirdal, "Four definitions of magnetic permeability for
  periodic metamaterials", Phys. Rev. B 99, 064407 (2019) (checked). C. A. Dirdal and J. Skaar,
  Eur. Phys. J. B 91, 131 (2018) (checked), on diamagnetism and the high-frequency limit of mu.
  C. A. Dirdal, H. O. Hågenvik, H. A. Haave and J. Skaar, IEEE Trans. Antennas Propag. 66, 6403
  (2018) (checked), on multipoles beyond the quadrupole at second order.
- V. A. Markel and J. C. Schotland, Phys. Rev. E 85, 066603 (2012) (checked). Homogenization of
  periodic composites with the reflection of a half-space computed explicitly.
- A. Alù, Phys. Rev. B 84, 075153 (2011); C. R. Simovski, Metamaterials 1, 62 (2007); C. R.
  Simovski and S. A. Tretyakov, Phys. Rev. B 75, 195111 (2007); D. R. Smith et al., Phys. Rev. B
  65, 195104 (2002); T. Koschny et al., Phys. Rev. E 68, 065602 (2003) and Phys. Rev. B 71, 245105
  (2005); S. O'Brien and J. B. Pendry, J. Phys.: Condens. Matter 14, 4035 (2002); L. Peng et al.,
  Phys. Rev. Lett. 98, 157403 (2007); C. M. Linton, SIAM Review 52, 630 (2010). From memory, not
  checked.
- The reference lists at the end of the notebooks contain further entries that were not checked
  either: Raab and de Lange (2005), Lorentz (1878), Pastrnak and Vedam (1971), Burnett, Levine and
  Shirley (2001), Yaghjian, Alù and Silveirinha (2013), Rayleigh (1892), Perrins, McKenzie and
  McPhedran (1979), and the textbook of Joannopoulos and coauthors.
