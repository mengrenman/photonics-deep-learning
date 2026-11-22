# Plan and status, updated 2026-11-23

This file records what exists, what the plan was, and what is left, so that work can resume
without the conversation that produced it. `README.md` has the findings.

## State

- Notebooks 01 to 04 were merged into `main` as pull requests 2 and 3 (merge commits e9a1ef4 and
  1e9b549, 2026-11-16).
- Notebooks 05 to 09 and the modules `basislattice.py`, `layered.py` and `wires.py` were built on
  2026-11-22 and 23 on the branch `claude/spatial-dispersion-more-cases` (from `main` at 41d6264),
  in response to "consider more cases, not just lattices of dielectric rods". They were reviewed
  (see `README.md`, "How the notebooks were checked") and corrected. The same branch changes
  notebook 04 (the lumped-circuit limit, 1/12 and not 1/8, with two new checks, and 17 statements of
  its prose narrowed to what it prints), this file, the README of this folder, and the folder's entry
  in `notebooks/README.md`. When this was written nothing on the branch was committed, and
  `git status --short` in the repository root showed:
  - modified: `notebooks/README.md`, `notebooks/spatial_dispersion/04_interfaces_and_retrieval.ipynb`,
    `notebooks/spatial_dispersion/PLAN.md`, `notebooks/spatial_dispersion/README.md`,
    `notebooks/spatial_dispersion/_build/nb04.py`;
  - untracked: the notebooks `05_layered_media.ipynb`,
    `06_metal_rods_and_the_landau_lifshitz_limit.ipynb`, `07_wire_medium.ipynb`,
    `08_rings_of_plasmonic_rods.ipynb` and `09_dimers_quadrupoles_and_bianisotropy.ipynb`, their
    sources `_build/nb05.py` to `_build/nb09.py`, `_build/validate_basis.py`, and the modules
    `basislattice.py`, `layered.py` and `wires.py`.

  Whether this is committed and merged must be checked with `git log main --
  notebooks/spatial_dispersion`.
- Notebooks 01 to 04 have still had no independent review (section 1 below).

## The question

Spatial dispersion and its relation to optical magnetism: when a metamaterial built from
nonmagnetic parts is given a permeability at optical frequencies, what is that number, and is it
more than the k^2 term of a wavevector-dependent permittivity (Landau and Lifshitz, volume 8,
sections 79 and 103)?

## Scope that was fixed

- For 01 to 04: one test structure, a square lattice of circular dielectric rods in two
  dimensions, radius 0.3 a, relative permittivity 12 and 100, both polarizations. Extended for 05 to
  09 (2026-11-22 and 23) to periodic stacks (one-dimensional, 05), Drude-metal rods, the wire medium
  (a wave-vector component beta along the wires, 07), several rods per cell (`basislattice.py`) and
  rods with a permeability; the rod lattices are still square.
- Exact semi-analytical methods only (Mie coefficients, lattice sums, multiple scattering), with
  plane-wave expansions as independent checks. No JAX, no use of `bayes_scatter`.
- Three views of the same lattice: its Bloch modes, its bulk response eps(omega, k) to a source of
  arbitrary wavevector, and the reflection and transmission of slabs.
- Notebooks in the format of `notebooks/cloaking`: stored with outputs, results asserted with
  `check`, prose that refers to printed output.

## Done

| Item | State |
|---|---|
| `rodlattice.py`: Mie coefficients, lattice sums (square lattice and line array), Hermitian Bloch-mode finder, source-driven eps(omega, k), slab solver, finite-cluster solver | Written. Validated in the notebooks for nonmagnetic rods: Mie coefficients, lattice sums (square lattice and line array), Bloch modes, driven eps(omega, k), slabs. Tested only through the slab checks: `column_translation`. Rods with a permeability: tested only through `basislattice` (TE static limit and plane waves; TM with mu != 1 not implemented there). Metal rods: used in 06 to 09; `mie_pq` fails for eps mu < 0 (section 0). Never exercised: `cluster`, `cluster_far_field` |
| `01_one_tensor_or_two` | Runs, 16 checks |
| `02_rod_lattice_multipoles_and_bands` | Runs, 13 checks |
| `03_nonlocal_permittivity_of_the_lattice` | Runs, 35 checks |
| `04_interfaces_and_retrieval` | Runs, 26 checks (24 until 2026-11-23, when the lumped-circuit comparison was corrected and 17 statements in its prose were narrowed to what it prints) |
| `05_layered_media`, `layered.py` | Runs, 79 checks. Reviewed: selected numbers recomputed independently (not A to E, the ENZ reach, B(0) or 1/12 and -1/6; see section 0), premise review, fixes |
| `06_metal_rods_and_the_landau_lifshitz_limit` | Runs, 80 checks. Reviewed as 05 (selected numbers recomputed; see section 0) |
| `07_wire_medium`, `wires.py` | Runs, 50 checks. Reviewed as 05 (selected numbers recomputed; see section 0) |
| `08_rings_of_plasmonic_rods` | Runs, 54 checks. Reviewed as 05 (selected numbers recomputed; see section 0); its resonant numbers have no second method |
| `09_dimers_quadrupoles_and_bianisotropy` | Runs, 68 checks. Reviewed as 08 |
| `basislattice.py`: several rods per cell, rods with a permeability, molecule T-matrix | 63 checks in `_build/validate_basis.py` (about 45 s), including the comparison with `rodlattice` on the same lattice described two ways (metal rods, 4 or 9 per cell) added after the review |
| `README.md`: findings, conventions, checks, limits, literature | Written for all nine |
| Literature around the Landau-Lifshitz argument (requested during the session) | Searched on 2026-11-16; details of Merlin 2009, Agranovich and Gartstein 2006 and 2009, Silveirinha 2007, Skaar et al. 2019, Dirdal et al. 2018 (two papers) and Markel and Schotland 2012 confirmed against publisher, ADS or arXiv records. The text of the book itself was not available online |

## Not done, in the order it would be taken up

The sections run 0 to 4 in that order. Within section 0 the items are not ranked; the cheap one is
the `mie_pq` fix (with both workarounds). Check `git log main -- notebooks/spatial_dispersion` first
to see whether the branch was merged.

### 0. Open items from notebooks 05 to 09 (2026-11-23)

- Not recomputed independently by the review (the records were in the session scratchpad and are
  not kept):
  - 05: the coefficients A to E, the ENZ reach, B(0), and the lumped limits 1/12 and -1/6;
  - 06: all Landau-Lifshitz ratios, the half-saturation size, the shrink path, the true-mu control,
    all TM results, and the perfect-conductor extrapolation;
  - 07: the beta = 0 multipole comparison;
  - 08: every Drude-lattice number;
  - 09: every lattice number and the radiated shares;
  - `basislattice.molecule_t_matrix`.
- Known to be fragile:
  - 08: keep the Cauchy radius small near the pole (0.15 gives 0.5% in tau at a detuning of 1e-5);
    its extremes of mu with loss are grid values good to a few percent; f_p = 0.5 lies beyond the
    Bragg estimate.
  - 09: noise floor 1e-6 to 1e-5 in the coefficients (M = 10, radius 0.01); the cluster at M = 12
    is ill-conditioned (unitarity 1e-6); the pole moves 1e-5 between M = 8 and 10; size scans can
    jump to another mode (hybrid at r = 0.20), so follow modes by continuity.
  - 06: rod functions overflow near a/lambda_p = 1000 (300 is the largest used); the TE plane-wave
    check converges slowly and noisily (1 to 3%).
  - 05: the reach measure is ill-conditioned above 1.2 f_ENZ.
  - 07: the m = 0 model errs by 8% at R = 0.05 and 22% at R = 0.1 near the zone edge.
- `rodlattice.mie_pq` returns NaN for eps mu < 0, so Hermitian Bloch modes of metal rods fail in
  `rodlattice` and in `basislattice.kkr_hermitian` and `bloch_frequencies`. Notebook 08 patches
  `mie_pq` in its kernel (imaginary-wavenumber form), and notebook 09 rebuilds the Hermitian form
  with a complex square root (`kkr_hermitian_any`, section 3.1). Fix it in the module, remove both
  workarounds, rebuild 08 and 09, and add a metal-rod Bloch-mode check to `_build/validate_basis.py`.
- The resonant results of 08 and 09 rest on one method (multiple scattering with lattice sums),
  checked by internal consistency and by the two-descriptions test of `basislattice`. A second
  method that converges for metal rods (finite elements or finite differences with subpixel
  averaging) would close this.
- No slab of rings or dimers: the retrieval problem of notebook 04 for these structures, and the
  two-sided reflection that the bianisotropy of 09 implies.
- 06: the TM control without net current is a plane-wave static solution converged to a few
  percent; the separation of the net current from the rest of m = 0 was not computed.
- 08 and 09: the loss values. Bulk-silver loss 1/300 lies within published Drude fits (1/440 to
  1/190, checked 2026-11-23), but at the physical sizes implied (rods of 2 to 5 nm, 09's gap of
  2 nm) surface damping would raise it several times, and the nonlocal response and tunneling
  across the gap are not modeled. A loss model with size-dependent damping would make the loss
  conclusions specific.
- 09: whether the size of the bianisotropic correction at the loss-limited peak is set by the
  split omega_L - omega_T over the loss width (stated as a hypothesis); the hybrid mode at r = 0.20.
- 05: the claim that the stack's lumped limit 1/12 applies to the rod lattice of 04 is a
  hypothesis for the rods; the remaining factor between 0.046 and 1/12 was not derived.
- References of 05 to 09: bibliographic details were checked on 2026-11-23 (two corrected; README,
  Literature); what each is cited for was not checked against its text, except where an abstract
  was read. (Four stale "unchecked" or "not rerun" tags in 05, 06 and 08 were fixed the same day.)

### 1. Independent review of notebooks 01 to 04 (the first research task on resuming them, after the cheap fixes of section 0)

Nobody other than the author of the notebooks has recomputed a headline number or challenged a
conclusion. For the cloaking folder this step found 24 issues, 7 of them serious, all in
conclusions. The review is written as a workflow script and has not been run:
`_build/review_workflow.js`. It has one recompute auditor per notebook (own code, no reuse of
`rodlattice.py` for the number being checked), two premise skeptics on a small context, and a fix
pass. The results that most need an independent reproduction, because they are surprising and rest
on one code path each:

- The TM second-order coefficient vanishing like omega^2 (notebook 03, section 6), which says the
  magnetic dipole of a rod is canceled in the bulk response.
- The static slope of tau against the rod permittivity being exactly eta R^2 / 8, with eta the
  filling fraction (notebook 03, section 3). The TE driven tensor at finite k was compared with a plane-wave solution only in a
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
- References still cited from memory are listed as such at the end of `README.md`, which also
  names the further unchecked entries in the notebooks' own reference lists. The section
  numbers of Landau and Lifshitz (79, 103 to 106) should be checked against the book.
- The pull request and merge are recorded under State.

## How to resume

- Environment `photonics-dl`. All nine notebooks execute in about seven and a half minutes with four threads (08 takes
  200 s, 09 78 s, 06 63 s, the others under 45 s each):
  `cd notebooks/spatial_dispersion && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb`
- The notebooks are generated from percent-format sources in `_build/` (`nb01.py` to `nb09.py`,
  with `setup_cell.py` as the shared first cell). Edit the source and rebuild one notebook with
  `_build/build.sh nb03 03_nonlocal_permittivity_of_the_lattice`. The sources and the stored
  notebooks were identical cell for cell when this file was written; if a notebook is edited
  directly, the source falls out of date.
- `python _build/show_nb.py <notebook> <dir> 100000` prints the full text outputs of an executed
  notebook (without the last argument each output is cut at 6000 characters) and saves its figures
  as PNG files, which is how the outputs were read during the build.
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

## Record audit of 2026-11-16: what it found and what was not acted on

A workflow of nine agents audited the record without new computation: four auditors traced the
numbers and comparative statements of `README.md` and the notebook prose to stored outputs, four
verifiers rechecked what they reported, and one reviewer read this file and `README.md` as a
record. The twelve confirmed problems of wording and rounding were corrected. The rest of this
section is the reviewer's list, kept here because the work was paused before it could be used.

### Further open items (to be added to "Not done")

- High: only one termination was computed. The slab numbers, in particular (mu - 1)/(a/lambda)^2 =
  -7.5 for TM, hold for faces half a period outside the outer rows; the dependence on termination
  is inferred from the stack of notebook 01.
- Medium: rods with a permeability (`mie_t`, `slab`, `bloch_frequencies` with mu != 1) and the lossy
  TE paths of `driven` and `slab` have no check of their own. Section 3 above needs the first.
- Medium: the gap of the TE bands for a rod permittivity of 100 (0.1268 to 0.1417) is from one path;
  confirm it on a grid over the irreducible zone (about 20 points).
- Medium: for TE and a rod permittivity of 12 the slab extrapolation does not resolve the bulk
  mu - 1 (1.2e-4), and the extrapolated value has the opposite sign (-1.1e-5).
- Low: the parity of the excluded Gamma-M mode (f = 0.36344 at kB = (2, 2)) is asserted in prose
  only; magnetic point groups and nonreciprocal media in the counts of notebook 01; the general
  multipole decomposition and its dependence on the origin; which operator ordering a structure
  realizes and Hermitian surface terms; the frequency range above the zero of mu and above the
  first diffraction threshold; rods that are not circular.

### Cheap independent checks proposed for section 1

- Slab phase reference: one weak layer (rod permittivity 1.01) at a/lambda = 0.05 against the
  first-order Born result referred to x = -a/2. This tests `column_translation` and the reference
  planes directly.
- TM coefficient beta: it vanishes exactly at zero frequency by a one-line argument (now in notebook
  03, section 6), so the number is not in doubt and the open question is the reading "the magnetic
  dipole is canceled". For the omega^2 term, expand the driven problem in k0^2 with a plane-wave
  series and fit the k^2 term, without `rodlattice.py`.
- TE driven tensor at finite k: put the plane-wave solution into notebook 03 as a check at one
  (f, k) with a rod permittivity of 4, where plane waves converge fastest.
- Landau-Lifshitz ratio: repeat the scan in rod permittivity at a second radius (0.2 a). If the
  non-loop part of tau saturates at the scaled value, the identification of the loop term holds.
  The ratio is for the static limit, while the argument of Landau and Lifshitz concerns high
  frequency; this gap is not closed.
- Longitudinal magnetic mode: the TM plane-wave eigenvector at kB = (1, 0), f = 0.4029; confirm it
  is odd in y and that its G = 0 component vanishes.

### Conditions that the documents now state, and where they come from

- t_0(TE) = t_1(TM) holds for nonmagnetic rods in vacuum; the reviewer found it fails for a rod
  permeability of 2.
- The redundancy c_T + c_L (equal on Gamma-X and Gamma-M) is a consequence of C4v at second order.
- k^2 = k0^2 eps(omega, k) at Bloch modes holds for lossless rods, real k, and modes that are even
  under the mirror containing k.

### Problems in `_build/review_workflow.js`, to be fixed before it is run

The script has not been changed since it was written. Before running it:

1. The counting formula in the target for notebook 01 is wrong for the rows with E along z: there
   the count is (1/|G|) sum of the character of Sym^2, not of its square.
2. The prompts use f both for the frequency and for the filling fraction. Use eta = pi R^2 / a^2 =
   0.283 for the latter, as the notebooks now do, or an auditor will divide by the wrong number.
3. Two targets are not computable as specified in the time allowed: the TE coefficients by central
   differences of a plane-wave solution, and the TM beta of about 4e-5 at a/lambda = 0.01. Give the
   method (the expansion in k0^2 above) or drop them.
4. Disagreements from the recompute auditors go to the fix pass without verification, and `agrees`
   is a bare boolean. Add a verification stage, as for the skeptics' findings.
5. Minor findings of the skeptics are filtered out before verification and never returned.
6. Fix routing matches a free-text notebook field. Findings about `README.md` or `rodlattice.py` are
   dropped, a finding about two notebooks goes to two fixers, and the recount of checks is assigned
   to the auditor of notebook 04. Use a fixed set of tags.
7. Nothing uses the output of the fix pass: `README.md` and this file are never updated, and a
   rebuilt notebook that fails an assertion leaves source and notebook out of step unreported. The
   fixers may not touch `rodlattice.py` or `README.md`, so findings about them have no path.
8. The four fixers execute notebooks concurrently in one directory. Run them one after another, or
   in separate worktrees.
9. The phase table records the skeptics as Opus, but one of them runs on Fable.
10. The script assumed the folder was on an unpushed branch; it is on `main` now, so this is moot.

