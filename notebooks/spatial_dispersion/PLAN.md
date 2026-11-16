# Plan and status, frozen 2026-11-16

The exploration is paused. This file records what exists, what the plan was, and what is left, so
that work can resume without the conversation that produced it. `README.md` has the findings.

## The question

Spatial dispersion and its relation to optical magnetism: when a metamaterial built from
nonmagnetic parts is given a permeability at optical frequencies, what is that number, and is it
more than the k^2 term of a wavevector-dependent permittivity (Landau and Lifshitz, volume 8,
sections 79 and 103)?

## Scope that was fixed

- One test structure: a square lattice of circular dielectric rods in two dimensions, radius
  0.3 a, relative permittivity 12 and 100, both polarizations.
- Exact semi-analytical methods only (Mie coefficients, lattice sums, multiple scattering), with
  plane-wave expansions as independent checks. No JAX, no use of `bayes_scatter`.
- Three views of the same lattice: its Bloch modes, its bulk response eps(omega, k) to a source of
  arbitrary wavevector, and the reflection and transmission of slabs.
- Notebooks in the format of `notebooks/cloaking`: stored with outputs, results asserted with
  `check`, prose that refers to printed output.

## Done

| Item | State |
|---|---|
| `rodlattice.py`: Mie coefficients, lattice sums (square lattice and line array), Hermitian Bloch-mode finder, source-driven eps(omega, k), slab solver, finite-cluster solver | Written. Everything except `cluster` and `cluster_far_field` is validated in the notebooks |
| `01_one_tensor_or_two` | Runs, 16 checks |
| `02_rod_lattice_multipoles_and_bands` | Runs, 13 checks |
| `03_nonlocal_permittivity_of_the_lattice` | Runs, 35 checks |
| `04_interfaces_and_retrieval` | Runs, 24 checks |
| `README.md`: findings, conventions, checks, limits, literature | Written |
| Literature around the Landau-Lifshitz argument (requested during the session) | Searched on 2026-11-16; details of Merlin 2009, Agranovich and Gartstein 2006 and 2009, Silveirinha 2007, Skaar et al. 2019, Dirdal et al. 2018 (two papers) and Markel and Schotland 2012 confirmed against publisher, ADS or arXiv records. The text of the book itself was not available online |

## Not done, in the order it would be taken up

### 1. Independent review (the first thing to do on resuming)

Nobody other than the author of the notebooks has recomputed a headline number or challenged a
conclusion. For the cloaking folder this step found 24 issues, 7 of them serious, all in
conclusions. The review is written as a workflow script and has not been run:
`_build/review_workflow.js`. It has one recompute auditor per notebook (own code, no reuse of
`rodlattice.py` for the number being checked), two premise skeptics on a small context, and a fix
pass. The results that most need an independent reproduction, because they are surprising and rest
on one code path each:

- The TM second-order coefficient vanishing like omega^2 (notebook 03, section 6), which says the
  magnetic dipole of a rod is canceled in the bulk response.
- The static slope of tau against the rod permittivity being exactly f R^2 / 8 (notebook 03,
  section 3). The TE driven tensor at finite k was compared with a plane-wave solution only in a
  scratch calculation, to 0.2%.
- The retrieved admittance of slabs: the negative mu - 1 in TM and the antiresonance in TE
  (notebook 04). Both depend on the phase reference of the reflection coefficient in
  `rodlattice.slab`, which is tested only through the long-wavelength limit and the 1/N behavior.
- The longitudinal magnetic mode (notebook 03, section 6) and the reading of it.

### 2. Open physics, from the caveats of the notebooks

- Oblique incidence and slabs cut along Gamma-M. The anisotropy delta predicts that no single
  permeability serves all angles; not computed.
- The semi-infinite lattice. Needed to separate the static surface term from the (qa)^2 departure
  of the Bloch impedance at all frequencies; so far they are separated only at low frequency.
- An additional boundary condition that makes the full eps(omega, k) reproduce the slab.
- A term-by-term multipole account of tau (loop, quadrupole, dipole from field curvature, second
  moments) for the lattice. The loop part is identified only by its dependence on the rod
  permittivity.
- Location of the singularities of eps(omega, k) in the complex k plane (the true radius of
  convergence of the expansion), and the fourth-order coefficients, which are printed and unused.
- Loss: the coefficients for lossy rods.
- Other lattices, a second rod radius, three dimensions.

### 3. The link to the rest of the project

A finite body cut from the lattice, compared with a homogeneous magnetodielectric cylinder with
the bulk parameters and with the retrieved ones, and then given to the inverse-scattering testbed:
what does an inversion that assumes mu = 1 return for an object whose response is magnetic?
`rodlattice.cluster` and `cluster_far_field` were written for this and are untested.

### 4. Housekeeping

- Gallery entries and figures for the top-level `README.md` and `docs/figures`, as was done for
  the cloaking folder.
- References still cited from memory are listed as such at the end of `README.md`. The section
  numbers of Landau and Lifshitz (79, 103 to 106) should be checked against the book.
- No pull request has been opened.

## How to resume

- Environment `photonics-dl`. All four notebooks execute in about 75 s:
  `cd notebooks/spatial_dispersion && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb`
- The notebooks are generated from percent-format sources in `_build/` (`nb01.py` to `nb04.py`,
  with `setup_cell.py` as the shared first cell). Edit the source and rebuild one notebook with
  `_build/build.sh nb03 03_nonlocal_permittivity_of_the_lattice`. The sources and the stored
  notebooks were identical cell for cell when this file was written; if a notebook is edited
  directly, the source falls out of date.
- `_build/show_nb.py <notebook> <dir>` prints the text outputs of an executed notebook and saves
  its figures as PNG files, which is how the outputs were read during the build.
- Exploratory scripts that preceded the notebooks (parameter surveys, a plane-wave check of the TE
  driven tensor, a coated-cylinder effective-medium comparison that was dropped from the notebooks)
  were scratch files and were not kept. Their conclusions are in the notebooks or in section 1
  above.

## Decisions taken along the way

- The solver lives next to the notebooks as `rodlattice.py`, not in `src/bayes_scatter`, because it
  has nothing to do with the testbed.
- Lengths are in units of the lattice period here, not the vacuum wavelength as elsewhere in the
  project, because every notebook sweeps frequency at a fixed lattice.
- A coated-cylinder (CPA) effective-medium model was tried as a third local description and left
  out. In its small-argument form it was inaccurate already at a/lambda = 0.1, and in its full form
  it reproduced the retrieved (slab) parameters and not the bulk ones, so it added a method without
  adding a distinction.
