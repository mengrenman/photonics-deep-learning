"""Validation and timing of the forward model (docs/DESIGN.md section 8, steps 1-3).

Prints
  1. a Mie convergence table (relative error of the forward data at the receivers under grid refinement);
  2. timing and GMRES effort for one full forward solve (all transmitters) and one value-and-grad of a
     misfit, for the book's acquisition rings (ch15_setup) at n = 32, 48, 64, 96 with an eps_r = 2
     cylinder, and at n = 96 for the fixed test object at eps_r = 2 and 4 (and 8 with --hard);
  3. stacked versus vmapped solves, and the effect of a warm start.

Run:  python scripts/validate_forward.py [--quick] [--hard]
"""

import argparse
import time

import numpy as np

import jax
import jax.numpy as jnp

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ch15_setup, ring
from bayes_scatter.mie import mie_scattered
from bayes_scatter.phantoms import austria_fixed, coverage, cylinder

TX_MIE = ring(6, 3.0)
RX_MIE = ring(8, 3.5, offset=np.pi / 8)


def timed(fn, reps=3):
    """Run fn once to compile, then return (compile_plus_first_s, median_s, last_result)."""
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return first, float(np.median(times)), out


def mie_table(sizes):
    print("== 1. Forward vs Mie series: relative error ||d - d_mie|| / ||d_mie|| over 6 tx x 8 rx ==")
    print("   DOI side 2, tx ring r=3, rx ring r=3.5, tol 1e-10, lossless cylinder centered at the origin")
    cases = [("R=0.6 eps_r=2", 0.6, 2.0), ("R=0.8 eps_r=4", 0.8, 4.0), ("R=0.6 eps_r=2+0.5i", 0.6, 2.0 + 0.5j)]
    for label, radius, eps in cases:
        d_mie, n_modes = mie_scattered(TX_MIE, RX_MIE, radius, eps, return_nmax=True)
        print(f"   cylinder {label}  (Mie series truncated at n={n_modes})")
        print("      n     h       err(ss=4)   ratio    err(ss=64)  ratio")
        prev = {}
        for n in sizes:
            grid = Grid(n, 2.0)
            fwd = Forward(Setup(grid, Antennas(TX_MIE, RX_MIE)), tol=1e-10)
            row = []
            for ss in (4, 64):
                cov = coverage(grid, lambda X, Y: X**2 + Y**2 <= radius**2, ss=ss)
                d, _ = fwd((eps - 1.0) * cov)
                row.append(np.linalg.norm(np.asarray(d) - d_mie) / np.linalg.norm(d_mie))
            cells = []
            for ss, e in zip((4, 64), row):
                r = prev[ss] / e if ss in prev else float("nan")
                cells.append(f"{e:10.3e}  {r:5.2f}")
                prev[ss] = e
            print(f"   {n:4d}  {grid.h:.4f}  " + "   ".join(cells))


def solve_and_grad_row(label, setup, chi, d_obs, **fwd_kw):
    fwd = Forward(setup, **fwd_kw)
    chi = jnp.asarray(chi)

    def misfit(c):
        d, _ = fwd(c)
        return jnp.sum(jnp.abs(d - d_obs) ** 2)

    vg = jax.jit(jax.value_and_grad(misfit))
    c0, t_fwd, u = timed(lambda: fwd.solve_fields(chi))
    resid = float(np.asarray(fwd.residual(chi, u)).max())
    lanes = [0, fwd.n_tx // 4, fwd.n_tx // 2, 3 * fwd.n_tx // 4]
    counts = fwd.matvec_counts(chi, transmitters=lanes)
    c1, t_vg, (val, g) = timed(lambda: vg(chi))
    ok = bool(np.all(np.isfinite(np.asarray(g))))
    print(f"   {label:28s} fwd {t_fwd:7.2f}s (compile {c0:5.1f}s)  value+grad {t_vg:7.2f}s (compile {c1:5.1f}s)"
          f"  G-products/tx {int(counts.min())}-{int(counts.max())}  max rel resid {resid:.1e}"
          f"{'' if ok else '  GRAD NOT FINITE'}")
    return fwd


def timing_table(hard, quick):
    print("\n== 2. Timing: ch15 rings (40 tx, 60 rx), tol 1e-8, GMRES restart 100 ==")
    print("   'G-products/tx' = FFT operator applications per transmitter solved alone (min-max over 4 tx),")
    print("   including one residual evaluation per restart cycle. JAX returns no iteration counts.")
    sizes = (32, 48) if quick else (32, 48, 64, 96)
    for n in sizes:
        setup = ch15_setup(n=n)
        grid = setup.grid
        chi = cylinder(grid, 1.0, 2.0)
        d_obs = Forward(setup)(austria_fixed(grid, 2.0))[0]
        solve_and_grad_row(f"n={n} cylinder r=1 eps_r=2", setup, chi, d_obs)
    if quick:
        return
    setup = ch15_setup(n=96)
    grid = setup.grid
    for eps in (2.0, 4.0) + ((8.0,) if hard else ()):
        chi = austria_fixed(grid, eps)
        d_obs = Forward(setup)(austria_fixed(grid, eps * 0.9))[0]
        solve_and_grad_row(f"n=96 austria_fixed eps_r={eps:g}", setup, chi, d_obs)
    if hard:
        print("   (eps_r=8: GMRES(100) does not reach tol within maxiter=30 cycles; 'max rel resid' is the true")
        print("    residual left. With restart 200 the hard transmitters need ~1400 G-products.)")
    else:
        print("   (--hard adds the eps_r=8 phantom, which is slow and does not converge at restart 100)")


def batching_and_warm_start():
    print("\n== 3. stack vs vmap, and warm start (n=48, ch15 rings, eps_r=2 cylinder) ==")
    setup = ch15_setup(n=48)
    grid = setup.grid
    chi = jnp.asarray(cylinder(grid, 1.0, 2.0))
    for batch in ("stack", "vmap"):
        fwd = Forward(setup, batch=batch)
        c0, t, u = timed(lambda: fwd.solve_fields(chi))
        print(f"   batch={batch:5s}: solve {t:6.3f}s (compile {c0:5.1f}s)  max rel resid "
              f"{float(np.asarray(fwd.residual(chi, u)).max()):.1e}")
    fwd = Forward(setup)
    u_prev = fwd.solve_fields(chi)
    rng = np.random.default_rng(0)
    for eps in (1e-3, 1e-2):
        chi_new = chi + eps * jnp.asarray(rng.standard_normal(chi.shape))
        cold = fwd.matvec_counts(chi_new, transmitters=[0, 10, 20, 30])
        warm = fwd.matvec_counts(chi_new, u0=u_prev, transmitters=[0, 10, 20, 30])
        _, t_cold, _ = timed(lambda: fwd.solve_fields(chi_new))
        _, t_warm, _ = timed(lambda: fwd.solve_fields(chi_new, u_prev))
        print(f"   chi perturbed by {eps:g}: G-products/tx cold {cold.tolist()} warm {warm.tolist()};"
              f" solve time cold {t_cold:.3f}s warm {t_warm:.3f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip n=64, 96 and the largest Mie grid")
    ap.add_argument("--hard", action="store_true", help="also time the eps_r=8 phantom (slow)")
    args = ap.parse_args()
    print(f"JAX {jax.__version__}, devices {jax.devices()}")
    mie_table((24, 48, 96) if args.quick else (24, 48, 96, 192))
    timing_table(args.hard, args.quick)
    batching_and_warm_start()
