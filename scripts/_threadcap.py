"""Cap the BLAS / OpenMP thread count of a standalone script at ``MAX_THREADS`` (2).

    import _threadcap  # noqa: F401  (must be the first import: numpy reads these variables when it is imported)

The machine is also a working machine, and a stray numpy/scipy call (a dense Cholesky, a QR) otherwise uses every
core. Importing this module lowers ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS`` and
``VECLIB_MAXIMUM_THREADS`` to at most ``MAX_THREADS``; a value the caller has already set below that stays (so
``scripts/batch_run.py`` jobs with one thread keep one thread). It does not touch the XLA thread pool, which is sized by
``NPROC`` / ``XLA_FLAGS`` before the JAX backend starts (``scripts/batch_run.py`` sets both).
``tests/test_prior_comparison_scripts.py`` checks by parsing the scripts that this import precedes numpy.
"""

import os

MAX_THREADS = 2
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cap(max_threads=MAX_THREADS):
    """Lower every variable of ``THREAD_VARS`` to at most ``max_threads`` (set it if absent or unparsable)."""
    for var in THREAD_VARS:
        try:
            cur = int(os.environ.get(var, max_threads))
        except ValueError:
            cur = max_threads
        os.environ[var] = str(max(1, min(max_threads, cur)))


cap()
