#!/bin/bash
# Rebuild and execute one notebook from its percent-format source.
#   notebooks/homogenization/_build/build.sh nb01 01_the_cell_problem
# Run with the photonics-dl environment active (needs nbformat, nbconvert, ipykernel).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/build_nb.py" "$here/$1.py" "$here/../$2.ipynb"
cd "$here/.." && jupyter nbconvert --to notebook --execute --inplace "$2.ipynb" --ExecutePreprocessor.timeout=1800
