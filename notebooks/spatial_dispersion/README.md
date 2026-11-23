# Spatial dispersion and optical magnetism

Nine notebooks on one question from metamaterials research: when a structure made of nonmagnetic
parts is said to have a magnetic permeability at optical frequencies, what is that permeability, and
is it more than a wavevector dependence of the permittivity? The starting point is the argument of
Landau and Lifshitz that the permeability loses its meaning at high frequency and that everything
can be carried by one tensor, eps(omega, k). Notebook 01 is the theory. Notebooks 02 to 04 compute
it all exactly for a square lattice of high-index dielectric rods in two dimensions. Notebooks 05 to
09 apply the same tests to other structures: a layered medium, where eps(omega, k) is exact and
several coefficients have closed forms; metal rods; the wire medium; and two plasmonic
metamolecules, a ring of four rods and a pair of rods. Read 01 first. The others run on their own
and can be read in any order after 01; 05 to 09 refer back to 03 for the method.

| # | Notebook | Question | Method | Run time |
|---|---|---|---|---|
| 01 | `01_one_tensor_or_two` | What part of second-order spatial dispersion is a permeability, and what does the bulk tensor not contain? | Exact algebra checked on random media, group averaging, transfer matrices | 6 s |
| 02 | `02_rod_lattice_multipoles_and_bands` | Where do the magnetic resonances of the rods fall in the band structure? | Mie coefficients, lattice sums, a Hermitian multiple-scattering mode condition | 41 s |
| 03 | `03_nonlocal_permittivity_of_the_lattice` | What is eps(omega, k) of the lattice, and how much of its k^2 term is a permeability? | Source-driven homogenization by multiple scattering, Taylor coefficients by Cauchy integrals | 15 s |
| 04 | `04_interfaces_and_retrieval` | Does the bulk permeability give the reflection of a slab? What does a retrieval return? | Exact slabs of line arrays, parameter retrieval | 7 s |
| 05 | `05_layered_media` | What are eps(omega, k) and its permeability for a periodic stack, where all of it is exact? What happens near epsilon-near-zero? | Exact 1D driven problem, transfer matrices, closed forms (obtained from sympy series run outside the notebook) checked against Taylor coefficients, the Bloch relation and the Bloch admittance | 20 s |
| 06 | `06_metal_rods_and_the_landau_lifshitz_limit` | How large is the magnetism of metal rods, how does it scale with size, and is it a permeability in the Landau-Lifshitz sense? | `rodlattice` with Drude rods, a true-permeability control through `basislattice`, plane-wave checks | 63 s |
| 07 | `07_wire_medium` | What does spatial dispersion look like when it is strong and not magnetic (k along the wires)? | Thin-wire model on the lattice sums, Bloch modes, checks against the multipole solver | 9 s |
| 08 | `08_rings_of_plasmonic_rods` | Does a ring of plasmonic rods give an optical permeability, and how strong is it with loss and size? | Cluster T-matrix, several rods per cell (`basislattice`), Bloch modes against the driven response | 200 s |
| 09 | `09_dimers_quadrupoles_and_bianisotropy` | Is the magnetism of a dimer separable from its quadrupole? What first-order (bianisotropic) term does an unequal dimer have? | Group averaging, molecule T-matrix, `basislattice`, Cauchy integrals in several directions | 78 s |

Run times are wall-clock times of `_build/build.sh` (kernel start included) on an M3 Max with four
threads, measured on 2026-11-23; all nine take about seven and a half minutes. `rodlattice.py` is
the solver for one rod per cell, used by notebooks 02 to 04 and 06, by the thin-wire model of 07
(its lattice sums, and `driven` as the comparison), and by 08 and 09 directly and through
`basislattice.py` (and for one Mie check in 01). `basislattice.py` extends it to several rods per
cell and to rods with a permeability (notebooks 06, 08 and 09). `layered.py` is the exact solver for
stacks (05), and `wires.py` the thin-wire model (07).

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
on where it is cut, (mu - 1)/(k0 a)^2 = -0.25 or +0.33 for the example, while its index is one
curve.

**02. The lattice.** For nonmagnetic rods the two magnetic-dipole coefficients, m = 0 for H along
the rod (TE) and m = 1 for E along the rod (TM), are the same function of frequency. For rods of
radius 0.3 a the resonance is at a/lambda = 0.36 for a rod permittivity of 12 and at 0.126 for 100.
With 12 the first TE band reaches the zone boundary before the resonance; with 100 the bands have
the textbook shape on the path sampled, a flattening band, a gap from 0.127 to 0.142, and a band
starting at Gamma, and the O'Brien-Pendry permeability with a static permittivity places the first
band at X to 0.2% (0.6% from the top of the band, which is at M).

**03. The nonlocal permittivity.** For TE only tau has the resonance pole, and only tau grows with
the rod permittivity, with the slope of a current loop, eta R^2 / 8, where eta = pi R^2 / a^2 =
0.283 is the filling fraction (fitted between rod permittivities 300 and 1000 at a/lambda = 0.001:
0.0031826 against 0.0031809). The rest is a background set mostly by the geometry; for rod
permittivity 100 and above it is delta of about -0.01 a^2, lambda of about 0.004 a^2, and a part of
tau that is not the loop, about -0.015 a^2 (for 12: -0.0065, 0.0025 and -0.012). The ratio of the
loop term to the largest of these, the two sides of the Landau-Lifshitz inequality, is 2.9 for a rod
permittivity of 12 and 19.7 for 100. This puts numbers on Merlin's "large permittivity begets
high-frequency magnetism", with two qualifications: the ratio is for the static limit, and the loop
term is the formula, identified with a part of tau by its scaling and not separated from the rest by
a computation. Far below resonance the apparent mu - 1 along Gamma-X and Gamma-M differ by 57% of
mu - 1 for 12 and by 6.5% for 100; close to the resonance the two have poles at slightly different
frequencies and differ by any factor. For rod permittivity 12 the index anisotropy (k_GM -
k_GX)/k_GX that delta implies is found in the exact Bloch wavenumbers to 0.4% at low frequency and
to 10% at a/lambda = 0.165; near the zone boundary the prediction is several times too large (about
6 times at 0.32 for rod permittivity 12, 9 times at 0.1244 for 100). For TM the k^2 coefficient is
isotropic by symmetry and vanishes like omega^2 (computed for rod permittivity 12: 0.1% of the loop
estimate at a/lambda = 0.01). At zero frequency this is exact for rods of any cross-section, by an
analytic argument: the driven field is then the plane wave of the source. In the bulk response the
magnetic dipole of the rod is not a permeability. For TM with rod permittivity 12 the lattice also
has Bloch modes (parity checked along Gamma-X only) that are odd under the mirror containing k,
carry no macroscopic E or B, and do not appear in eps(omega, k); the one near a/lambda = 0.40 is
read as the longitudinal magnetic mode.

**04. Interfaces.** The index retrieved from a slab of lossless rods tends to the bulk Bloch index
with thickness (1.5e-4 at 16 layers, within 0.3% for one layer in the four TM and TE cases printed).
The admittance is another matter. All slabs here have their faces half a period outside the outer
rows of rods, and the numbers that follow are for that termination. For TM with rod permittivity 12
at low frequency three procedures give three permeabilities: positive from the rod's magnetic
dipole, zero at order omega^2 from the bulk expansion, and negative from the slab, (mu -
1)/(a/lambda)^2 = -7.5 at a/lambda = 0.01 to 0.02 (growing in magnitude with frequency), independent
of the number of layers (the admittances from 1 and 16 layers agree to 3e-4 up to a/lambda = 0.10).
The slab's admittance exceeds the bulk one by the fraction c (qa)^2, with c = 0.046 at low frequency
(a/lambda = 0.01 to 0.02; c grows to 0.135 at 0.17). This is read as the Bloch-impedance effect of a
lattice cut between its polarizable elements. The lumped-circuit limit of the coefficient, against
the same Bloch-index reference, is 1/12, 1.8 times more; the remaining factor was not derived for
the rods. (Until 2026-11-23 this compared with 1/8, which is the limit against the static line
admittance; notebook 05 found the difference.) For TE through the magnetic resonance with lossy
rods, 100 (1 + 0.02 i), the permittivity retrieved from one layer has an antiresonance with an
imaginary part down to -1.3, while the bulk permittivity stays between 1.77 and 1.79 with a positive
imaginary part below 0.001; just below the resonance the bulk parameters give 0.54 to 0.65 of the
exact reflection amplitude |r| of eight layers (a/lambda = 0.118 to 0.127). The ratio of the
single-layer impedance to the bulk impedance falls with qa for TM and for TE with rod permittivity
12 (0.335 at qa = 2.7 for TM, 0.896 at qa = 2.9 for TE), and for TE with rod permittivity 100 it
first falls and then rises, to 4.6 at qa = 2.9 (the one resonant case computed: lossless TE, f =
0.1262); the notebook reads this as a lumped electric or magnetic response of the cell but does not
establish it. In TE thin slabs also carry a static surface term proportional to 1/N. In TE (rod
permittivities 12 and 100), extrapolated to a thick slab (1/N -> 0) at a/lambda = 0.01, the
retrieved permittivity equals the bulk one to 3e-4; in TM it differs by 3e-3, which the notebook
attributes to the (qa)^2 term of the admittance. In TE the extrapolated permeability agrees with the
bulk one to 1.5e-4 (in TM it is 0.99925 against 1, 7.5e-4 off), which is the size of mu - 1 itself
for a TE rod permittivity of 12 and 12% of it for 100, so that test says little about mu.

In one sentence: for H along the rods and a large rod permittivity, a permeability describes the
resonant part of the bulk response to second order in ka, on a direction-dependent background that
is small compared with it away from the poles; the number a slab returns under the same name mixes
that response with the periodicity and the termination, and the two were found to agree only in
the index and, in TE, for the permittivity in the limit of a thick slab at long wavelength (in TM
a difference of order (qa)^2 remains).

### Other structures (05 to 09)

**05. Layered media.** For a periodic stack the source-driven problem is a one-dimensional equation,
so eps(omega, k) is exact (`layered.py`, checked against a second solver built from plane waves to
1.4e-15; the two share the equations, which are tested by the Bloch condition of the transfer
matrix, to 6.7e-16 for TM and 4.6e-9 for TE, and by the static limit). For the dielectric stack of
notebook 01 (a layer of permittivity 12 filling eta = 0.3 of the period), with E along the layers
and k normal to them, the k^2 coefficient vanishes like f^2 and the bulk mu - 1 is of order f^4:
mu - 1 = c22 (k0 a)^4 with c22 = eta^2 (1 - eta)^2 (eps1 - eps2)^2 [1 + 2 eta (1 - eta)] / 240 =
0.0316, a closed form found here. Read with k along the layers, the same tensor component is of
order f^2 ((mu - 1)/(k0 a)^2 = +0.025), so the stack has no bulk mu independent of direction at
order f^2. The two permeabilities retrieved from the Bloch admittance in notebook 01 (-0.25 and
+0.33 in units of (k0 a)^2) have closed forms, Y_B/n_B - 1 = c (qa)^2 with c = +0.058 and -0.076 for
the two cuts, and lumped-circuit limits of 1/12 and -1/6 against the Bloch index, reached only by a
thin layer of very high permittivity (this stack's +0.058 is 0.70 of 1/12); this is the difference
that led to notebook 04's comparison with 1/8 being corrected. The second-order index correction n^2
= <eps> + C (k0 a)^2, with C = eta^2 (1 - eta)^2 (eps1 - eps2)^2 / 12 (Rytov's form as the notebook
recalls it, not checked against the paper), comes out the same three ways (Taylor coefficients, a
fit of the exact Bloch relation, the closed form) for E along the layers and normal incidence; for E
normal to the layers the correction is a different one (0.0348 in the units printed). Of the five
second-order TE coefficients that the symmetry of a stack allows, one combination is mu_zz; the
stack's coefficients do not have its pattern (A = 0 holds exactly, but D is not zero and B, C and -E
are not equal), and the size of a "magnetic part" projected out of them depends on the inner product
chosen by factors of up to about 11. Near epsilon-near-zero in a lossless metal-dielectric stack
(metal filling 0.25, dielectric permittivity 4, a/lambda_p = 0.25) the period is a fifteenth of the
wavelength, and still: the exact ENZ frequency is 2.1% below the local one (the shift equals the
Rytov term to 2%); the static local medium has the wrong k_x at k_y = 0 (36% at 0.975 f_ENZ, 65% at
1.025 f_ENZ, and a floored relative error of 3.8 at the crossing itself, where the exact k_x is
zero); the local medium with the exact eps(omega, 0) holds, to 5% in k_x, to k_y of order k0 away
from the crossing (and fails almost at once on it, where eps_yy vanishes); and the second-order
tensor reaches at least 2.6 times further than that local model (sweep 0.85 to 1.1 f_ENZ; no other
local model was tried, and the notebook does not take the factor as robust). Up to 1.1 f_ENZ the
largest second-order coefficient is D, the longitudinal one along the layers, which no permeability
can produce.

**06. Metal rods.** For TE and a collisionless Drude metal, k0^2 tau tends at low frequency to a
nonzero constant, 1 - 1/mu_OBP, where mu_OBP is the O'Brien-Pendry permeability in London's limit
(checked over four decades of a/lambda_p); it is not mu_OBP - 1, and for a dielectric rod the same
coefficient falls like f^2. This static diamagnetism is proportional to (R/delta_s)^2 for rods
thinner than the skin depth delta_s = lambda_p / 2 pi and saturates at flux exclusion, -eta/(1 -
eta); |mu - 1| reaches half its saturated value at R/delta_s = 3.3 (a/lambda_p = 1.75 for R = 0.3).
As f -> 0 a plasma-metal rod and a rod with a true permeability keep a static mu; a dielectric rod
and an ohmic metal rod do not (ohmic: real part like f^2, imaginary part like f). With f_gamma/f_p =
1/100 (the only loss for which the crossover was computed), a rod several skin depths thick keeps
its eddy-current screening until the ohmic skin depth reaches about R/2 (R/delta_ohm = 1.9 and 2.5
at the half-value frequency for a/lambda_p = 3 and 30), far below f_gamma (f_gamma/252 at a/lambda_p
= 30); a thin rod (a/lambda_p = 0.3) loses it at about f_gamma. The skin-depth criterion is the
reading that fits the three printed sizes, not a separate computation. The Landau-Lifshitz ratio,
with the mean-field part as the magnetic one (a model split), is about 0.13 to 0.16 |eps_m| for thin
rods (1.9 at |eps_m| = 15), independent of their size, and within a factor of two of a dielectric
rod of the same |eps| in five of the six printed cases (the sixth, |eps| = 1000 at a/lambda_0 =
0.01, compares with a dielectric rod near its resonance and is not like for like): it grows with
|eps| as Merlin's argument says, but is of order one only for |eps_m| <~ 15, where noble metals are
in the visible. For thick rods it is about 20 to 370 at a/lambda_0 = 0.05 to 0.1 and grows like
1/f^2 as the frequency falls (up to about 8400 at 0.01), a consequence of the definition that a rod
with a true permeability shares. For TM the rods are a two-dimensional wire medium: the lattice
reproduces the thin-wire plasma wavenumber for perfectly conducting rods extrapolated in 1/q (5e-5
in k_p^2 at R = 0.01, 4.5e-3 at R = 0.05) and, at R = 0.01, the constant 0.52734 of notebook 07 (the
lattice implies 0.52766; the check resolves 2e-3, and the rest is read as an O(R^2) correction). At
R = 0.3 the k^2 coefficient of perfectly conducting rods (extrapolated in 1/q) is 0.18 of the
Maxwell Garnett value. A control, a plane-wave solution at sigma = 100 converged to a few percent
(sigma_eff 1.8275 against Rayleigh's 1.7688 at that contrast), gives k0^2 beta = -0.83 against the
infinite-contrast Maxwell Garnett -0.79 and -0.145 for the wires, so the difference is attributed to
the net wire current; the control does not separate that current from the rest of m = 0 (at R = 0.3
the wire current alone, M = 0, gives +11.8 against -0.145 for the full calculation, so the
wire-current and flux-exclusion parts do not add). That coefficient is formal anyway: at R = 0.3,
k_p a = 6.3 > pi, and the TM field is evanescent at every subwavelength frequency.

**07. The wire medium.** Thin metal wires along z, with a wave vector that has a component beta
along them. A thin-wire model built on the lattice sums of `rodlattice` reproduces Belov's plasma
wavenumber (k_p a = 1.3809 at r0 = 0.01 a, 5e-5 in k_p^2; the constant of the formula is ln[4
pi^(3/2) / Gamma(1/4)^2] = 0.52734), Silveirinha's formula for metal wires, and the full multipole
solver at beta = 0 (1.9e-3 at R = 0.02). For perfect wires eps_zz has a pole at beta = k0 at every
a/lambda, so the local plasma model is accurate only to (beta/k0)^2, misses the TEM branch k0 = beta
and, at beta = 2 and k_perp = 0.5 (the case printed), has a branch near the plasma frequency with no
counterpart in the exact modes. At the two points counted (k0 = 2 and 3, k_perp = 0.5; perfect
wires, m = 0 thin-wire model) there are three real roots in beta^2 (ordinary, TEM and TM), against
at most two for any local uniaxial (eps, mu), so no local uniaxial (eps, mu) reproduces the modes at
the printed points. The beta^2 term is the plasma term -k_p^2/k0^4 and cannot be a permeability for
any mu tensor (the zz entry of the magnetic term contains only k_perp); the k_perp^2 term is smaller
by a factor that falls with frequency (1356 at a/lambda_0 = 0.05, 143 at 0.15, for r0 = 0.01 a) and,
in this model, whose only currents are axial (eps = diag(1, 1, eps_zz)), has the form of a
permeability only at beta = 0. With Drude wires (collisionless, r0 = 0.01 a, f = 0.05, exact
impedance) the beta dependence survives once the wire radius exceeds a crossover r0/delta_s, where
it is half its perfect-wire value, that grows with beta/k0: 0.85 at beta = 0.3 k0, 2.19 at 0.9 k0
(1.79 with the thin impedance; it also depends on r0) and 15.7 at 0.99 k0, where thick wires are
needed; the thin-impedance closed form diverges at the TEM pole. The spatial dispersion does not
shrink with a/lambda (the beta series has radius of convergence k0, printed at f = 0.05, and the
relative beta dependence matches x^2/(1 - x^2), x = beta/k0, at f = 0.02 to 0.1), and none of the
second-order coefficients computed is a permeability once beta is not 0 (in the m = 0 model; the m =
+-1 term, 6% of the k_perp^2 coefficient at R = 0.01, is probably eddy-current diamagnetism and was
not tested).

**08. A ring of four plasmonic rods.** The isolated ring (a/lambda_p = 0.2, rods of radius 0.08 a at
0.22 a from the center) has a magnetic-dipole resonance at a/lambda_0 = 0.1277, below its electric
ones, carried by four equal tangential dipoles; a coupled-dipole model places it 0.73% high. In the
lattice the pole of tau is at 0.12773 (the Bloch mode and the driven response agree to 4e-9), and
the pole is in tau alone, as the symmetry of the mode requires. Without loss mu is negative in a
band of relative width 4.4e-4, reaching -1 in its middle. With bulk-silver loss, f_gamma/f_p =
1/300, Re mu stays between 0.915 and 1.08; the negative band survives only below f_gamma/f_p = 2e-4
to 3e-4, with a figure of merit -Re mu / Im mu of 14 at 1e-5 and 1.2 at 1e-4. The Landau-Lifshitz
ratio of the pole term (the pole of a Lorentz fit taken as the magnetic part, a model split) exceeds
10 below the pole only within 0.12% of it without loss (0.41% if the longitudinal coefficient is
left out of the comparison); above the pole it is 10.4 at 0.1% and 0.55 at 1%. With bulk-silver loss
the ratio reaches at most 4.4 (15 without the longitudinal coefficient), and 1.6 with the loss ten
times larger. Scaling the whole structure (a/lambda_p from 0.1 to 0.5) moves the resonance with
lambda_p; its dimensionless strength grows like (a/lambda_p)^1.5 on average from 0.1 to 0.4 (the
local exponent falls from 1.8 to below zero, and the strength is lower at 0.5 than at 0.4), and with
f_gamma/f_p = 1/300 the width of the band stays below the loss width f_gamma/f_m at every size (at
most 0.17 of it), so no size gives Re mu < 0 with that loss. In physical units the rings are tiny
(with silver's plasma wavelength, a period of about 28 nm and rods of about 2 nm), where
size-dependent damping is expected to raise the loss toward the 1/30 case, in which the resonance is
barely visible.

**09. Dimers.** Two Drude rods per cell (a/lambda_p = 0.25, radius 0.15 a, gap 0.06 a). Group
averaging gives five second-order TE coefficients for the symmetric dimer, one of them magnetic, and
no first-order term; unequal rods allow exactly one first-order term, eps_xy = -eps_yx = g k_y. The
antisymmetric mode of the isolated dimer (a/lambda_0 = 0.1482, radiative Q 2697) radiates, within
the block of even m, 86% in m = 0 (an upper bound for its magnetic dipole) and 14% in m = +-2. In
the lattice its pole in the transverse coefficient is fifty times weaker for waves across the dimer
than along it (R_y/R_x = 0.020), so the apparent permeabilities mu_X (waves along the dimer axis)
and mu_Y (across it) differ, with bulk-silver loss at the peak of Im mu_X (a/lambda_0 = 0.14755), by
2.2 times |mu_tau - 1| (a modulus; 0.34 at a/lambda_0 = 0.06). Since mu_zz is one number, that
difference is not magnetic, and about a fifth of the resonant part of tau itself is the square of
the quadrupole coupling (0.75 of the magnetic one in amplitude; from the single-mode relation).
Without loss mu_X < 0 in a band 0.12% wide; with bulk-silver loss the smallest Re mu_X is 0.79. The
first-order term of the unequal dimer is the magnetoelectric component xi_yz, linear in the
difference of the radii for differences up to 0.01 a (|g| = 0.054 a at the resonance with
bulk-silver loss, for radii 0.16 a and 0.14 a, 2.7 times the difference); by reciprocity and the
mirror it cannot change the bulk dispersion at first order. At second order it makes the dark mode
visible in eps_yy(omega, 0), with a strength growing as the square of the radius difference, and for
waves along y it cancels the pole of the dark mode in the apparent magnetic coefficient and moves
the resonance those waves see from the pole omega_T to the zero omega_L of eps_yy, higher by 3.7e-5
(collisionless, same pair of radii; the split grows as the square of the difference, and with
bulk-silver loss the g^2/eps_yy term is about 3% of the genuine coefficient at the peak). At fixed
g/r = 0.4, following the lowest antisymmetric mode by continuity gives residues 4.6e-5, 1.79e-4 and
7.9e-5 for radii 0.10, 0.15 and 0.20; at 0.20 the gap to the next cell is comparable to the gap
inside the dimer and a stronger hybrid mode appears, so these are three geometries and not a scaling
law.

Across notebooks 05 to 09 the same test separates three things that the word permeability is used
for. For a collisionless (London) metal, the static diamagnetism of closed currents passes the tests
of notebook 06 that need no model split: k0^2 tau tends to a frequency-independent constant, and a
rod with a true permeability matching the metal rod's gives the same coefficient (the limit mu_r ->
0, flux exclusion, is approached by metal rods only for R >> delta_s). An ohmic rod has no static
permeability, and the Landau-Lifshitz ratio, which does rest on a model split, is of order one for
thin rods with |eps_m| <~ 15. Resonant terms that are magnetic by symmetry exist in the plasmonic
ring and dimer, but they are weak and not dominant: the band where they make mu negative is 1.3e-4
to 8.7e-4 of the frequency wide for the ring (five sizes) and 5.3e-4 for the dimer's mu_tau (7.8e-5
in a/lambda_0; its apparent mu_X, not magnetic by symmetry, is negative over 1.2e-3), none of the
five ring sizes (base ring geometry) or the one dimer geometry computed with bulk-silver loss has Re
mu < 0, and non-magnetic second-order terms are comparable near the resonance (08 and 09 give the
numbers). In the wire medium and in metal stacks near epsilon-near-zero the local description fails
at long wavelength without a resonance of the inclusions, and the second-order terms are not
magnetic in form; no common measure of strength was computed, so the notebooks do not rank these
cases against the rings and dimers.

## Running

Use the `photonics-dl` environment. Nothing beyond its packages is needed (numpy, scipy,
matplotlib; the notebooks do not use JAX or `bayes_scatter`).

```bash
cd notebooks/spatial_dispersion && jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

- The first code cell is the same in all nine: thread caps (four, set only if the environment does
  not set them), the figure style and palette of the cloaking notebooks, and the `check` helper.
- `check(name, value, tol, target)` prints PASS or FAIL and asserts. The nine notebooks hold 421
  such checks (90 in 01 to 04) plus plain assertions; a rerun that moves a checked result outside
  its tolerance stops with an error.
- The Markdown mostly refers to the printed output, but some numbers are typed into the prose (for
  example c = 0.046 and 0.54 to 0.65 in 04, 0.004 to 0.016 a^2 in 03) without an assertion on the
  typed value.
- The notebooks are generated from the percent-format sources in `_build/`; `PLAN.md` says how to
  rebuild one, and records the state in which the work was paused and what is left to do.

## Conventions

- Time dependence exp(-i omega t); outgoing cylindrical waves are Hankel functions of the first
  kind. eps0 = mu0 = c = 1.
- Lengths are in units of the lattice period a, and frequency is a/lambda_0 = k0 a / 2 pi. This
  differs from the rest of the project (lambda_0 = 1) because the folder sweeps frequency at a fixed
  lattice.
- The polarization with E along the rods is called TM and the one with H along the rods TE (in 05
  the same axis z lies in the plane of the layers, parallel to every interface), as in
  `docs/DESIGN.md`.
- In 01 to 04, rods have radius 0.3 a and relative permittivity 12 or 100, lossless unless stated.
- eta is a filling fraction: pi R^2 / a^2 for rods, the thickness fraction of the high-index (or
  metal) layer in 05. The letter f is only ever the frequency a/lambda_0.
- Metals are Drude, eps_m = 1 - f_p^2 / (f (f + i f_gamma)), with f_p = a/lambda_p the plasma
  frequency in the same units and delta_s = lambda_p / 2 pi the skin depth of a plasma. "Bulk-silver
  loss" means f_gamma/f_p = 1/300. Published Drude fits for silver give gamma/omega_p between about
  1/440 and 1/190 and lambda_p between 135 and 147 nm (checked on 2026-11-23 against Drachev et al.,
  Opt. Express 16, 1186 (2008), Olmon et al., Phys. Rev. B 86, 235147 (2012), and fits quoted from
  Ordal and from Johnson and Christy). At the sizes that 08 and 09 imply in physical units (rods of
  radius 2 to 5 nm), surface damping, gamma_bulk + A v_F / r with A of order 1, adds several times
  the bulk value, so their loss-free and bulk-loss numbers are expected to overstate the response
  (the larger damping was not computed; the case f_gamma/f_p = 1/30 of notebook 08 is the closest
  computed proxy, and the 2 nm gap of 09 also brings nonlocal and tunneling effects that are not
  modeled).

## How the notebooks were checked

### Notebooks 01 to 04

- Independent methods inside the notebooks: lattice sums against brute-force sums at complex
  frequency (2.5e-11); Bloch frequencies against a plane-wave expansion for TM (3e-5); the driven
  permittivity against a plane-wave solution for TM (2e-4) and against Rayleigh's static series for
  TE (1e-7 for a rod permittivity of 12, 4e-7 for 100); the Bloch modes from the mode condition
  satisfy k^2 = k0^2 eps(omega, k) from the driven calculation (lossless rods, real k, and only the
  modes that are even under the mirror containing k; two odd modes out of nine were excluded after
  the fact, and both paths share the Mie coefficients and lattice sums, so this is a check of
  consistency and not of the split of n^2 into eps and mu); flux balance and reciprocity of slabs to
  rounding; the retrieved index of slabs of lossless rods tends to the bulk Bloch index (1.5e-4 at
  16 layers, 0.3% at most for one layer in the four TM and TE cases printed), which ties the slab
  solver to the bulk solver.
- Checked in a scratch calculation and not kept in a notebook: the TE driven tensor at finite k
  against an in-plane plane-wave solution, which agrees to 0.2% and converges slowly toward the
  multiple-scattering value.
- Several statements written before the numbers were seen turned out wrong and were caught by their
  own assertions, then corrected in the text: the sign of one term in the bianisotropic identity;
  "the single-rod resonance lies in the gap" (it lies just below the top of the first band); "the
  background coefficients do not change with frequency" (true for a rod permittivity of 100, not for
  12); "the anisotropy of mu is small near resonance" (small relative to tau, not in mu, whose two
  poles differ).
- On 2026-11-16, after the work was paused, the record was audited without new computation: four
  agents (Sonnet) traced every number and comparative statement in this README and in the notebook
  prose to the stored outputs, a second agent rechecked each reported problem, and one reviewer
  (Opus) read `PLAN.md` and this file as a record. Twelve problems were confirmed, all in wording or
  rounding and none in a computed number, and the reviewer added conditions and reference choices
  that the first version left out. They are corrected here and in the notebooks.
- Still not done for 01 to 04: nobody has recomputed a headline number with separate code or tested
  the conclusions with a solver of their own, as was done for the cloaking folder and for 05 to 09.
  `PLAN.md` lists what most needs it. On 2026-11-23 all four were rerun: 01 to 03 printed the same
  output as stored (apart from one timing line) and were left as they were; 04 was changed in its
  lumped-circuit comparison (two new checks) and, after a consistency audit, in 17 statements of its
  prose that said more than its printed output.

### Notebooks 05 to 09 (2026-11-22 and 23)

- Each notebook asserts its own checks: for 05 a second solver and closed forms (obtained from sympy
  series run outside the notebook, not included in it; the notebook checks each formula numerically
  against Taylor coefficients, Bloch fits and exact admittances); for 07 brute-force lattice sums at
  imaginary wavenumber, Belov's and Silveirinha's formulas and the multipole solver; for 06
  plane-wave solutions for metal rods in TE (one to three percent with the Laurent rule, 4% with the
  inverse rule in the two cases run; slow and noisy convergence) and TM, closed-form static limits
  and a control with a true permeability; for 08 and 09 internal consistency checks (the Bloch modes
  against the poles of the driven response, which share the Mie coefficients and lattice sums, as
  for 01 to 04, and the unitarity and symmetry of cluster T-matrices). The checks against outside
  results are static limits against plane waves for dielectric rods (eps = 4) in the same geometries
  and, in 09, one metal rod against Rayleigh's series. The plane-wave expansion did not converge for
  the metal rods even off resonance (08 at a/lambda_0 = 0.08, 09 at 0.1), so no metal-rod lattice
  number of 08 or 09 has a plane-wave check in the notebooks.
- `basislattice.py` passes the 63 checks of `_build/validate_basis.py` (sums between sublattices
  against brute force to 5e-12, one rod per cell against `rodlattice`, translation of the basis,
  plane waves for dielectric bases, the static limit of a permeable rod, guards). Two of them were
  added after the review, which compared it with `rodlattice` on the same lattice described two
  ways: four or nine metal rods per cell against one rod in a cell of a half or a third of the
  period. The reviewer found agreement to 4e-14 in the plasmonic regime, 6e-11 near the plasmon
  resonances and 1e-7 with gaps of 0.06 a, the gap used in 09; the kept checks reach 1e-12 and 1e-9.
- Review: for every notebook (and for `basislattice.py`) one agent (Sonnet) recomputed a selection
  of headline numbers with code of its own. Every number attempted agreed within the auditor's
  convergence; for 09's metal rods off resonance that convergence was about 4% for g (5.9e-3 against
  6.14e-3) and (mu_X - mu_Y)/(mu_tau - 1) came out as -0.40 with a scatter of 0.06 against -0.337
  (plane waves with a normal-vector factorization, per the audit record, not a notebook output). Not
  recomputed independently: - 05: the five TE coefficients A to E, the ENZ reach comparison and its
  lossy case, B(0), and the lumped limits 1/12 and -1/6; - 06: the half-saturation size, all
  Landau-Lifshitz ratios, the shrink path, the true-permeability control, all of TM, and the
  perfect-conductor extrapolation; - 07: the comparison with the multipole solver at beta = 0; - 08:
  every number of the Drude-rod lattice (only the isolated ring and the lattice with eps = 4 rods
  were recomputed); - 09: every lattice number and the radiated multipole shares;
  - `basislattice.py`: the molecule T-matrix.

Three premise reviewers (Opus; one for 05 and 07, one for 06, one for 08 and 09) tested the
conclusions with computations of their own; each serious or moderate finding was rechecked by an
agent told to refute it, and none was refuted. Confirmed problems (thirteen serious or moderate
findings, none refuted, plus one minor one), all corrected in the notebooks: a "magnetic part" in 05
that depended on the choice of inner product by factors up to 11; an order f^4 in 05 that belonged
to one direction of k; in 07 a question about the k_perp^2 term that the model could settle (it is
not a permeability), and a crossover that depended on an arbitrary choice of beta; in 06 a
Landau-Lifshitz ratio "of order one for thin rods" that is set by |eps_m| instead, an ohmic
crossover that is set by the skin depth and not by f_gamma, and an explanation of the TM Maxwell
Garnett disagreement that a control computation then confirmed in part; in 07 a
radius-of-convergence argument that was taken to exclude any local (eps, mu) (only mode counting
excludes a local uniaxial one); in 06 a k_perp^2 coefficient of TM that was called an in-plane
permeability (it is mostly axial-current dispersion), and a 1/f^2 fall of the LL ratio that was read
as Landau and Lifshitz's argument (it follows from the definition and holds for a true permeability
too); in 08 a Landau-Lifshitz statement that holds only without loss and a size exponent that
included two powers of units; in 09 points "below the pole" that straddled it (the pole is 3.7e-5
below the Bloch frequency), and a size scan that had jumped to another mode. A last pass checked
that the intros, summaries and caveats of 04 to 09 say what the corrected sections print.
- The references of 05 to 09 and the silver Drude parameters were checked against primary records on
  2026-11-23 (see Conventions and Literature); what each reference is cited for was checked only
  where its abstract could be read.

## Known limits

- Two dimensions and one lattice (square) throughout. In 01 to 04 the results are for one rod radius
  (0.3 a) and two rod permittivities (12 and 100); a second radius and other permittivities appear
  only in validation checks, in the single-rod resonance table of 02 (rod permittivities 30 and
  1000), in the static scan of notebook 03, and in the lossy rods of 04 (100 (1 + 0.02 i)).
- `rodlattice.py`: the coupling between rows of a slab is tested only through the slab checks. Rods
  with a permeability are tested only through `basislattice` (TE static limit and plane waves).
  `mie_pq`, which the Hermitian mode condition uses, returns NaN for eps mu < 0, so
  `rodlattice.bloch_frequencies`, `basislattice.kkr_hermitian` and `basislattice.bloch_frequencies`
  fail for metal rods; notebook 08 replaces `mie_pq` in its own kernel, notebook 09 rebuilds the
  Hermitian form with a complex square root, and both modules are unchanged.
- **03.** The split of tau into a loop part and a background is identified by its dependence on the
  rod permittivity, not derived term by term. Singularities of eps(omega, k) in the complex k plane
  were avoided, not located. Fourth-order coefficients are printed and not analyzed.
- **04.** Normal incidence along a lattice axis and one termination only, so the dependence of the
  slab results on termination is inferred from the one-dimensional stack of notebook 01 and not
  computed. The semi-infinite lattice was not solved, so the surface term and the Bloch-impedance
  effect are separated only at low frequency. No additional boundary condition for the nonlocal
  description was derived. Finite bodies cut from the lattice were not computed;
  `rodlattice.cluster` is written for that and is untested.
- Band structures in 02 are sampled on one path and are for orientation; the gap quoted for a rod
  permittivity of 100 was not confirmed on a grid of the whole zone.
- **05.** Normal incidence and k_y = 0 for the closed forms. The Drude metal has no nonlocal
  response of its own, so only the spatial dispersion of the geometry is computed. The "magnetic
  part" of the stack's coefficients depends on a chosen inner product.
- **06.** One radius (0.3 a) for most of TE. The lossy low-frequency laws are for one size
  (a/lambda_p = 3) and two losses (f_gamma/f_p = 1/300 and 1/30); the ohmic crossover is for three
  sizes at one loss (1/100). The Landau-Lifshitz ratio rests on a model split (the mean-field part
  as the magnetic one), exact only in the static limit. The TM no-net-current control is a
  plane-wave solution converged to a few percent. No resonators: closed rods have diamagnetism and
  no magnetic resonance.
- **07.** Thin-wire model with the axial current only (m = 0); the missing m = +-1 response changes
  the k_perp^2 coefficient by about 6% at R = 0.01 and beta = 0 (the correction grows with R: 0.026
  against 0.058 at R = 0.05, from notebook 06). No slab, so no additional boundary condition.
  Collisionless Drude wires; loss only spot-checked.
- **08 and 09.** One base geometry each (08 also with four other pairs of ring and rod radii and
  five sizes; 09 with three sizes and unequal rods), TE only, Drude metal without interband or
  surface damping. The resonances and poles of the lattices have no check by a second method (the
  isolated ring and dimer resonances were recomputed independently, and the ring also against a
  coupled-dipole model). No slab was computed for either structure, so the different reflection from
  the two sides that the bianisotropy of 09 implies is not tested.
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
- C. R. Simovski, Metamaterials 1, 62 (2007); C. R. Simovski and S. A. Tretyakov, Phys. Rev. B 75,
  195111 (2007); D. R. Smith et al., Phys. Rev. B 65, 195104 (2002); T. Koschny et al., Phys. Rev. E
  68, 065602 (2003) and Phys. Rev. B 71, 245105 (2005); L. Peng et al., Phys. Rev. Lett. 98, 157403
  (2007); C. M. Linton, SIAM Review 52, 630 (2010). From memory, not checked. (S. O'Brien and J. B.
  Pendry, J. Phys.: Condens. Matter 14, 4035 (2002), and A. Alù, Phys. Rev. B 84, 075153 (2011),
  were checked on 2026-11-23 with the references of 05 to 09.)
- The reference lists at the end of the notebooks contain further entries that were not checked
  either: Raab and de Lange (2005), Lorentz (1878), Pastrnak and Vedam (1971), Burnett, Levine and
  Shirley (2001), Yaghjian, Alù and Silveirinha (2013), and the textbook of Joannopoulos and
  coauthors. (Rayleigh (1892) and Perrins, McKenzie and McPhedran (1979) were checked on 2026-11-23.)

For notebooks 05 to 09 (cited in the notebooks with full titles). Bibliographic details of all of
these were checked on 2026-11-23 against publisher, DOI, arXiv, ADS or library records, and two were
corrected (the page range of Maslovski et al.; Rayleigh's journal is Phil. Mag., Series 5). For most
the abstract could not be read, so what each is cited for is not checked against its text:

- Layered media: S. M. Rytov, Sov. Phys. JETP 2, 466 (1956); J. Elser et al., Appl. Phys. Lett. 90,
  191109 (2007); A. A. Orlov et al., Phys. Rev. B 84, 045424 (2011); A. V. Chebykin et al., Phys.
  Rev. B 84, 115438 (2011).
- Metal rods and wires: S. O'Brien and J. B. Pendry, J. Phys.: Condens. Matter 14, 4035 and 6383
  (2002); J. B. Pendry et al., IEEE Trans. Microwave Theory Tech. 47, 2075 (1999) and Phys. Rev.
  Lett. 76, 4773 (1996); J. Zhou et al., Phys. Rev. Lett. 95, 223902 (2005); F. London and H.
  London, Proc. R. Soc. Lond. A 149, 71 (1935); P. A. Belov et al., Phys. Rev. B 67, 113103 (2003);
  M. G. Silveirinha, Phys. Rev. E 73, 046612 (2006); S. I. Maslovski, S. A. Tretyakov and P. A.
  Belov, Microwave Opt. Technol. Lett. 35, 47-51 (2002); C. R. Simovski et al., Adv. Mater. 24, 4229
  (2012).
- Rayleigh, Phil. Mag. (Series 5) 34, 481 (1892); W. T. Perrins, D. R. McKenzie and R. C. McPhedran,
  Proc. R. Soc. Lond. A 369, 207 (1979).
- Plasmonic rings and dimers: A. Alù, A. Salandrino and N. Engheta, Opt. Express 14, 1557 (2006); A.
  Alù and N. Engheta, Phys. Rev. B 78, 085112 (2008); J. Petschulat et al., Phys. Rev. A 78, 043811
  (2008), "Multipole approach to metamaterials"; A. Alù, Phys. Rev. B 84, 075153 (2011); A. N.
  Serdyukov, I. V. Semchenko, S. A. Tretyakov and A. Sihvola, *Electromagnetics of Bi-anisotropic
  Materials: Theory and Applications* (2001; the title and year were confirmed, the publisher was
  not).