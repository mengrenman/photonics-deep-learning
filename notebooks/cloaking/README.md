# Cloaking notebooks

Five notebooks on invisibility and cloaking. The starting point is Leonhardt and Philbin,
*Geometry and Light: The Science of Invisibility* (Dover, 2010). Notebooks 01 and 02 work through
the book's ray optics and transformation optics; notebooks 03 to 05 go beyond it, toward what
cloaking means for this project's inverse-scattering testbed. Read them in order. Each one runs on
its own.

| # | Notebook | Question | Method | Run time |
|---|---|---|---|---|
| 01 | `01_rays_and_invisible_lenses` | How much of invisibility can be read off from rays alone? | Ray tracing with `jax.grad`, Luneburg's inversion, the conformal-map cloak | 10 s |
| 02 | `02_transformation_optics_cylindrical_cloak` | What medium does a coordinate map prescribe, and how fragile is the cylindrical cloak? | `jax.jacfwd`, a radial ODE for the partial waves | 25 s |
| 03 | `03_full_wave_cloaks_and_gauge_freedom` | Does a cloak built from a map work in a wave solver, and what does the same freedom do to an inverse problem? | Q1 finite elements with a perfectly matched layer | 1 min |
| 04 | `04_partial_waves_and_scattering_cancellation` | Are ray-invisible lenses invisible to waves? What can one homogeneous shell do? | Multilayer Mie series, direct radial integration | 35 s |
| 05 | `05_cloaking_meets_inverse_scattering` | Can the testbed's forward model design a dielectric cloak, and can the testbed detect it? | `bayes_scatter` forward model with JAX gradients, L-BFGS-B, Fisher information | 3 to 5 min |

Run times are for an M3 Max with four threads.

## Main findings

**01. Rays.** One ray tracer reproduces the Luneburg lens, Maxwell's fish-eye, the Eaton lens and
the invisible sphere. Luneburg's inversion turns a prescribed deflection function into an index
profile. With deflection counted positive toward the center, a lens of finite radius cannot
deflect its rim rays outward, so a constant negative deflection is impossible, while a weakly
diverging lens can be built. The conformal cloak (Zhukovski map with a Kepler profile on the lower
sheet) hides a region with only 1.8% of the area of the device's inner disk, and every ray that
crosses the branch cut is delayed by the same optical path.

**02. Transformation optics.** The medium of any coordinate map follows from one Jacobian. A
cylindrical cloak truncated just outside its inner radius scatters like a thin wire in virtual
space. With a conducting lining the residual falls only logarithmically: 1.5% of the bare
cross-section at a truncation of 10^-3 radii and still 0.05% at 10^-12. With a magnetic-wall lining
it falls with the fourth power of the truncation. The reduced parameters of the microwave
experiments leave 28% of the bare cross-section; about 70% of that comes from the impedance
mismatch at the outer boundary and the rest from the monopole wave, which no reduced set cloaks.

**03. Full waves and gauge freedom.** A cloak generated from a map and automatic differentiation,
for a circle and for a squircle, cuts the cross-section of the hidden object in a finite-element
solver. At 30 points per wavelength the circular cloak leaves 23% of the bare cross-section with a
conducting lining and 3.4% with a magnetic-wall lining. The same change of variables is a gauge
freedom of the inverse problem for anisotropic media: a smoothly deformed, pulled-back medium
reproduces the scattering data of the original object to 2.8%, falling under refinement, while the
naively deformed object differs by 77%. Isotropic nonmagnetic media, the testbed's class, do not
have this freedom, because conformal maps that are the identity outside a disk are trivial.

**04. Partial waves.** Ray-invisible is not wave-invisible. The invisible lens gives every partial
wave the same phase, so the cylinder is nearly invisible only at integer k0 R (Q = 0.03 at
k0 R = 10) and scatters with Q close to 4 at half-integers. For the sphere the condition is the
opposite one, half-integer k0 R, because of the caustic phase (scalar waves: Q = 3.95 at k0 R = 10
and 0.06 at 10.5). Capping the singular center at an index of 8 to 20 destroys the effect: Q at
integer k0 R is then 0.5 to 1.6. One homogeneous shell cancels the scattering of a small cylinder
(in TM the shell permittivity must be below one, here -0.6; in TE a positive value works, here
0.52), deeply only for objects much smaller than the wavelength. An all-dielectric shell cannot do
this in the quasi-static range and reaches at most 8 dB, in bands of larger sizes, over the range
scanned.

**05. Cloaking in the testbed.** Gradient descent on a shell of pixels hides a dielectric core from
one transmitter to 46 dB on the grid it was tuned on (50 dB in another run; see the limits below).
The design does not transfer: evaluated on a grid twice as fine it keeps 28 dB, which is the
mismatch between the two discretizations, while the same pixels re-tuned on the fine grid are deep
again there. Hiding the core from all 40
transmitters is far harder: the designs found reduce the scattered energy by 5 to 7 dB. In the
testbed's noise units (30 dB) the single-view design has a detection index between 1 and 2 when
data and design share one operator and 13 to 16 when the data come from a finer grid; every design is
plainly visible from all 40 views. One view at one frequency is exactly non-identifiable in the
model, with or without noise. With all views every design found is detected, and the uniqueness
theorems for complete data exclude an exact cloak in this class of media.

## Running

Use the `photonics-dl` environment. Nothing beyond its packages is needed.

```bash
cd notebooks/cloaking && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- Notebooks 02 to 05 import `bayes_scatter` (02 and 03 only its Mie series). Each adds `../../src`
  to `sys.path` when that folder exists, so it uses the checkout it lives in.
- The first code cell is the same in all five: thread caps (four, set only if the environment does
  not set them), the figure style and palette, and the `check` helper.
- `check(name, value, tol, target)` prints PASS or FAIL and asserts. The five notebooks hold 224
  such checks; a rerun that changes a result stops with an error.
- The Markdown avoids typed-in results and refers to the printed output instead, to keep the prose
  from drifting away from the numbers.

## Conventions

- Time dependence exp(-i omega t); lengths in free-space wavelengths, lambda_0 = 1, k_0 = 2 pi
  (notebook 01 uses the lens radius, since no wavelength enters).
- The polarization with E along z is called TM, as in `docs/DESIGN.md`. The book calls it TE.
- Book references are by section and equation number. `CLAIMS.md` and `ERRATA.md` are the reading
  notes on the book in `references/geometry-and-light-notes/`, which is git-ignored and exists only
  in the main checkout.

## How the notebooks were checked

- Each notebook was built from a written specification by one agent (Sonnet) and audited by a
  second, which recomputed the headline numbers with its own code. No wrong number was found.
- Two senior reviewers (Opus on 01 to 03, Fable on 04 and 05) then tested the conclusions against
  the other plausible reading, with their own solvers. They raised 24 findings, 7 of them serious,
  all about conclusions and none about numbers. Every finding was addressed in the notebook, by a
  correction, an added computation or a stated limit, and each change was verified by a rerun.
- Recomputed a second time by the main session: the semiclassical phases of the invisible and
  Eaton profiles; the lining laws of notebook 02 (the three conducting-lining values and the
  fourth-power slope); the exact uncapped invisible profile for the cylinder and the scalar sphere.
- All five were also executed in a plain environment, with no path or thread variables set.

## Known limits

- **03.** The cloak ratios are not converged. They fall under refinement, but three grids support a
  trend and not an exponent, the regime right at the lining is not separately validated, and at 30
  points per wavelength the residual of the magnetic-wall cloak is partly discretization error.
- **04.** The sphere is solved for scalar waves only, at four sizes.
- **05.** The depths are stopping points of the optimizer and not properties of the design problem.
  The depth of the single-view design changes with the thread environment: 46.5 dB in the stored
  outputs, 50.0 dB in a run with no thread variables set, with detection indices of 1.6 and 1.1.
  Every check passes in both.
  The all-view designs are local optima from one or two starts; a reviewer found a ring design at
  8 dB with a different solver, which the notebook does not reproduce.
- The wave calculations are two-dimensional, apart from the scalar sphere in 04, and
  single-frequency, apart from the dispersion and bandwidth sections.
- Each notebook ends with its own list of caveats and open ends.
