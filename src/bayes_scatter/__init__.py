"""Bayesian inverse scattering testbed.

2-D TM Lippmann-Schwinger forward model with FFT-accelerated pixel Green's functions,
Gaussian-field priors in whitened coordinates, and from-scratch MCMC samplers (pCN,
pCN-Langevin, MALA, HMC) in JAX. See docs/DESIGN.md for conventions.
"""

import jax

# Float64 throughout: the Green's-function sums and the adjoint gradient checks need it.
jax.config.update("jax_enable_x64", True)
