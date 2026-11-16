export const meta = {
  name: 'spatial-dispersion-review',
  description: 'Independent review of notebooks/spatial_dispersion: recompute auditors, premise skeptics, verification, fix pass',
  whenToUse: 'On resuming notebooks/spatial_dispersion, before extending or merging it. Written 2026-11-16 and not yet run.',
  phases: [
    { title: 'Recompute', detail: 'one auditor per notebook recomputes its headline numbers with code of its own', model: 'sonnet' },
    { title: 'Skeptics', detail: 'two premise skeptics test the conclusions against the other plausible reading', model: 'opus' },
    { title: 'Verify', detail: 'every serious finding is rechecked by an agent told to refute it', model: 'sonnet' },
    { title: 'Fix', detail: 'one fix pass per notebook on confirmed findings, in its source under _build', model: 'sonnet' },
  ],
}

// Paths are relative to the repository root (the agents start there). Override with args if needed.
const DIR = (args && args.dir) || 'notebooks/spatial_dispersion'
const PY = (args && args.python) || 'python'   // pass args.python to name the interpreter of the photonics-dl environment

const COMMON = `
You are reviewing the notebook folder ${DIR} of the photonics-dl repository. Read ${DIR}/README.md and
${DIR}/PLAN.md first. Notebooks are stored with outputs; print the text outputs of one with
  ${PY} ${DIR}/_build/show_nb.py ${DIR}/<notebook>.ipynb <a scratch directory> 100000
and its Markdown and code with a few lines of Python over the JSON. The shared solver is ${DIR}/rodlattice.py.
Conventions: time dependence exp(-i w t), lattice period a = 1, frequency f = a/lambda0 = k0/(2 pi), rod radius 0.3,
TM means E along the rods, TE means H along the rods. Use the Python above (numpy, scipy, matplotlib available).
Keep each thread pool at 4 threads (OMP_NUM_THREADS=4). Write scratch files only outside the repository.
Do not commit, do not push, do not edit any file in the repository unless your task says so explicitly.`

const TARGETS = {
  '01_one_tensor_or_two': `
- The table of counts in section 3 (36/6, 3/1, 2/1 in three dimensions; 9/1, 3/1, 2/1 and 3/3, 1/1 in two): recompute the
  number of invariants from characters, (1/|G|) sum_g chi(g)^2 with chi the character of Sym^2, and for the continuous
  groups by integrating over the group; recompute the dimension of the magnetic subspace separately.
- Section 4: t_0 and t_1 of a weak rod at k0 R = 0.2 from Mie formulas you write yourself; the corrections -x^2/4 and +x^2/4.
- Section 5.1: peak reflectance of the two orderings (the notebook prints both).
- Section 5.2: the band edge of the stack and (mu - 1)/(k0 a)^2 for the two cuts, from a transfer matrix you write yourself.`,
  '02_rod_lattice_multipoles_and_bands': `
- The resonance table (x = k0 R sqrt(eps_r) for eps_r = 12, 30, 100, 1000) and the identity t_0(TE) = t_1(TM).
- One set of lattice sums G_n by a method that is not the one in rodlattice.py (for instance Ewald summation, or the
  accelerated reciprocal-space formula of McPhedran and coworkers), at a real k0.
- TM band frequencies at X and M for eps_r = 12, R = 0.3, from a plane-wave expansion you write yourself.
- TE, eps_r = 100: the top of the first band and the bottom of the next (the notebook prints 0.1268 and 0.1417), by a
  method independent of rodlattice.py. A plane-wave expansion converges slowly here; state the convergence you reached.`,
  '03_nonlocal_permittivity_of_the_lattice': `
- eps_zz(w, k) for TM at three (f, k) points from your own plane-wave solution with a source.
- The TE coefficients tau, delta, lambda at f = 0.01 for eps_r = 12 (the notebook prints them) from your own solution of the
  driven in-plane problem, with central differences in k instead of the Cauchy integral. State your convergence.
- The static slope of tau against eps_r and the formula f R^2 / 8: derive the formula yourself and recompute the slope.
- TM: beta at f = 0.005 and 0.01 from your own code, the exponent, and beta at f = 0.01 over f (eps_r - 1) R^2 / 8.
- TM: the frequency of the pole of eps_zz(w, 0), and the Bloch mode near f = 0.403 at kB = (1, 0): find it with your own
  plane-wave band calculation and determine its parity under y -> -y from its eigenvector.`,
  '04_interfaces_and_retrieval': `
- Reflection and transmission of ONE layer of rods for TM, eps_r = 12, at f = 0.05 and 0.10, normal incidence, from a
  solver that shares nothing with rodlattice.py (two-dimensional finite differences with periodic boundaries along y and
  absorbing layers or plane-wave ports along x, with a convergence study). Reference planes half a period from the row.
  Then the retrieved mu - 1, to be compared with the notebook's table.
- The same for TE, eps_r = 100 (1 + 0.02 i), at f = 0.120 and 0.127: the retrieved permittivity and the sign of its
  imaginary part.
- Recount the checks of all four notebooks (the README says 88) and the figures (17).`,
}

const RECOMPUTE_SCHEMA = {
  type: 'object',
  properties: {
    notebook: { type: 'string' },
    items: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          quantity: { type: 'string' },
          notebook_value: { type: 'string' },
          recomputed_value: { type: 'string' },
          method: { type: 'string' },
          agrees: { type: 'boolean' },
          tolerance_reached: { type: 'string' },
          note: { type: 'string' },
        },
        required: ['quantity', 'notebook_value', 'recomputed_value', 'method', 'agrees'],
      },
    },
    not_attempted: { type: 'array', items: { type: 'string' } },
  },
  required: ['notebook', 'items', 'not_attempted'],
}

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          notebook: { type: 'string' },
          location: { type: 'string' },
          claim: { type: 'string' },
          concern: { type: 'string' },
          evidence: { type: 'string' },
          severity: { type: 'string', enum: ['serious', 'moderate', 'minor'] },
          proposed_change: { type: 'string' },
        },
        required: ['notebook', 'location', 'claim', 'concern', 'evidence', 'severity', 'proposed_change'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['refuted', 'reason'],
}

const TRIAGE = `
Anomaly triage, to be applied to every claim you examine:
- A result that contradicts the cited literature, or a simple estimate, by more than its stated accuracy must be reproduced
  independently, from the equations and with code of your own, before you accept it.
- Every count (checks, figures, coefficients allowed by a symmetry, modes) must be recomputed.
- Every identity the notebooks state must be checked for the conditions under which it holds, in every code path that uses it.
- Every comparison must be like for like: same frequency, same direction of propagation, same loss, same number of layers,
  same reference planes, same definition of the compared quantity.`

const SKEPTICS = [
  {
    key: 'bulk',
    model: 'opus',
    prompt: `${COMMON}
You are a premise skeptic for notebooks 01 and 03 (the bulk response). Do not check arithmetic that the notebooks already
assert; test whether the conclusions survive the other plausible reading. Questions to press, among your own:
- "Only tau is resonant and only tau grows with the rod permittivity": is the split of tau into a loop part and a background
  an identification by scaling only? Would another radius or filling fraction change the picture? Compute one other case.
- "The TM second-order coefficient vanishes at low frequency": is this a property of the definition of the macroscopic
  polarization (the weight exp(-i k.r)) and so true of any structure, or a property of rods? What does it say about the
  claim that the magnetic dipole "is canceled"?
- "The longitudinal magnetic mode cannot be represented by eps(w, k)": is that a statement about the mode, or about the
  choice of exciting it with a z-directed current at k along a mirror axis? What happens slightly off the axis?
- The Cauchy-integral Taylor coefficients: independence of the radius is asserted at one frequency. Check near the
  resonance, where singularities approach k = 0.
- The Landau-Lifshitz ratio: is "loop coefficient over the largest other coefficient" the inequality of section 79, or a
  different one? If you have access to the book, compare with its wording; if not, say so.
${TRIAGE}
You may edit nothing. Report findings only.`,
  },
  {
    key: 'interface',
    model: 'fable',
    prompt: `${COMMON}
You are a premise skeptic for notebooks 02 and 04 (bands and interfaces). Do not check arithmetic that the notebooks already
assert; test whether the conclusions survive the other plausible reading. Questions to press, among your own:
- The retrieved admittance depends on where the faces of the slab are put. The notebooks put them half a period outside the
  outer rows. How do the negative mu - 1 (TM) and the antiresonance (TE) change if the faces move? Is "the Bloch impedance of
  a lattice cut between its elements" the right reading, or is part of the effect the choice of thickness d = N a?
- "Bulk parameters with Fresnel boundary conditions miss the reflection": is the comparison like for like (same loss, same
  direction, mu along Gamma-X)? Would the O'Brien-Pendry permeability do better or worse, and why was it not shown?
- The lumped-circuit picture (T and Pi sections) is offered as the mechanism with limits "not fits". Is there a quantitative
  test that could fail? Run one.
- The 1/N surface term in TE: is it a property of the lattice or of referring the retrieval to d = N a?
- Notebook 02: the statement about which frequency intervals are gaps is made from one path. Check one of them on a grid of
  the whole zone.
${TRIAGE}
You may edit nothing. Report findings only.`,
  },
]

// Recompute auditors and skeptics run together; verification of each serious or moderate finding starts as soon as its
// source finishes (pipeline, no barrier).
const notebooks = Object.keys(TARGETS)

const recompute = pipeline(
  notebooks,
  (nb) => agent(`${COMMON}
You are a recompute auditor for ${DIR}/${nb}.ipynb. Recompute the quantities below with code that you write yourself.
You may read rodlattice.py to learn the conventions, and you may import it for parts that are NOT the quantity under test,
but the number you report must not come from it. For each quantity give the notebook's value (from the stored output), your
value, your method, and the convergence you reached. A disagreement beyond your own convergence is a finding; do not adjust
your method to make it agree. If a quantity is beyond what you can compute in about 20 minutes, list it under not_attempted
with the reason instead of guessing.
${TARGETS[nb]}`, { label: `recompute:${nb.slice(0, 2)}`, phase: 'Recompute', schema: RECOMPUTE_SCHEMA, model: 'sonnet' }),
)

const skeptics = pipeline(
  SKEPTICS,
  (s) => agent(s.prompt, { label: `skeptic:${s.key}`, phase: 'Skeptics', schema: FINDINGS_SCHEMA, model: s.model }),
  (res) => parallel(((res && res.findings) || []).filter((f) => f.severity !== 'minor').map((f, i) => () =>
    agent(`${COMMON}
A reviewer raised the finding below about ${DIR}. Try to refute it: find the output, the code or a short computation of
your own that shows the notebook's statement stands as written. If you cannot refute it, say so; if you are unsure, answer
refuted = false. You may edit nothing.
Notebook: ${f.notebook}
Location: ${f.location}
Claim in the notebook: ${f.claim}
Concern: ${f.concern}
Evidence offered: ${f.evidence}`, { label: `verify:${f.notebook.slice(0, 2)}-${i}`, phase: 'Verify', schema: VERDICT_SCHEMA, model: 'sonnet' })
      .then((v) => ({ ...f, verdict: v })))),
)

const [recomputed, judged] = await Promise.all([recompute, skeptics])
const disagreements = recomputed.filter(Boolean).flatMap((r) => r.items.filter((it) => !it.agrees).map((it) => ({ notebook: r.notebook, ...it })))
const confirmed = judged.flat().filter(Boolean).filter((f) => f.verdict && !f.verdict.refuted)
log(`recompute: ${disagreements.length} disagreements; skeptics: ${confirmed.length} findings survived verification`)

// Fix pass: one agent per notebook that has confirmed findings or disagreements. Each edits only its own source file.
phase('Fix')
const byNotebook = notebooks.map((nb) => ({
  nb,
  findings: confirmed.filter((f) => f.notebook.includes(nb.slice(0, 2))),
  disagreements: disagreements.filter((d) => d.notebook.includes(nb.slice(0, 2))),
})).filter((x) => x.findings.length + x.disagreements.length > 0)

const fixes = await parallel(byNotebook.map((x) => () => agent(`${COMMON}
Fix pass for ${x.nb}. Edit ONLY ${DIR}/_build/nb${x.nb.slice(0, 2)}.py (the percent-format source of the notebook), then rebuild with
  ${DIR}/_build/build.sh nb${x.nb.slice(0, 2)} ${x.nb}
(run it with the photonics-dl environment first on PATH, so that "python" and "jupyter" are that environment's).
Address each item by a correction, an added computation, or a stated limit, and keep the conventions of the folder: results
asserted with check(), prose that refers to printed output, American spelling. Do not touch rodlattice.py; if an item needs a
change there, describe the change in your report and leave it. Do not commit.
Confirmed findings: ${JSON.stringify(x.findings)}
Disagreements from the recompute audit: ${JSON.stringify(x.disagreements)}
Return a list of what you changed, item by item, and what you left and why.`, { label: `fix:${x.nb.slice(0, 2)}`, phase: 'Fix', model: 'sonnet' })))

return { recomputed, disagreements, findings: judged.flat().filter(Boolean), confirmed, fixes }
