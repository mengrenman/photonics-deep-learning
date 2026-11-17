# Notebooks

Exploratory notebooks, one subfolder per topic. Each folder has its own README with the reading
order, the main findings and the known limits.

| Folder | Topic |
|---|---|
| [`cloaking/`](cloaking/README.md) | Invisibility and cloaking: rays and invisible lenses, transformation optics, full-wave cloaks and gauge freedom, scattering cancellation, designed dielectric cloaks in the testbed |
| [`spatial_dispersion/`](spatial_dispersion/README.md) | Spatial dispersion and optical magnetism: the Landau-Lifshitz single-tensor description, the nonlocal permittivity of a lattice of high-index rods, how much of it is a permeability, and what a slab retrieval returns instead |
| [`omega_d/`](omega_d/README.md) | The Ωd problem: which results of the transfer matrix of a multilayer are lost for thick layers and why, with one channel and with many; the scattering, stiffness and hybrid matrices as charts of one object; secular functions and counting of guided modes |

## Conventions for a topic folder

- Notebooks are numbered in reading order and each one runs on its own, top to bottom, in the
  `photonics-dl` environment. They are stored with their outputs.
- Run them from their own folder (Jupyter does this by default). A notebook that uses the project
  package adds `../../src` to `sys.path` when that folder exists, so it imports the
  `bayes_scatter` of the checkout it lives in and not an installed copy.
- Physics conventions follow `docs/DESIGN.md`: time dependence exp(-i omega t), lengths in
  free-space wavelengths (lambda_0 = 1, k_0 = 2 pi), and "TM" for the polarization with E along z.
- Results that the text relies on are asserted in code, so a rerun that changes one fails instead of
  silently disagreeing with the prose.
