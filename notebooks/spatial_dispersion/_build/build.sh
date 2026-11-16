#!/bin/bash
# Rebuild and execute one notebook from its percent-format source.
#   notebooks/spatial_dispersion/_build/build.sh nb03 03_nonlocal_permittivity_of_the_lattice
# Run with the photonics-dl environment active (needs nbformat, nbconvert, ipykernel).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/build_nb.py" "$here/$1.py" "$here/../$2.ipynb"
cd "$here/.." && jupyter nbconvert --to notebook --execute --inplace "$2.ipynb" --ExecutePreprocessor.timeout=1800
