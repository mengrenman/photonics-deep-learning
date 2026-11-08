"""Benchmark the cost of the forward model, its gradient and batched gradients on the backend JAX sees.

Problem. For each grid "n:side" (n pixels per side, side in wavelengths) it builds the benchmark of the
nonlinear ladder: the Ch. 15 antenna rings (40 transmitters, 60 receivers), the fixed Austria phantom at
--eps-r, the log_eps link and a Matern prior (sigma 0.5, ell 0.4); the synthetic data come from the same grid,
which is fine for timing. All timings are taken at one point xi0 near the truth (the Wiener-regularized
whitened coordinates of log eps_r of the phantom, smoothed to the data resolution), because the Krylov cost
depends on the contrast: prior draws would reach eps_r > 5 and time a different problem. --xi prior uses a
prior draw instead.

Modes (--modes)
  forward  Forward model at xi0, all transmitters, cold start (aux = incident field) and warm start
           (aux = total fields of a solve at a nearby xi).
  grad     Target.potential_and_grad (value and adjoint gradient), cold and warm.
  batched  jax.vmap of potential_and_grad over B different xi (xi0 plus --spread noise) with a per-lane
           warm start, for each B in --batch; reports evaluations per second = B / time. The GMRES loop of a
           batch runs until its slowest lane has converged. Memory is about 16 bytes * (restart + 1) * 40 * n^2 * B
           for the Krylov basis, so large B at large n runs out of memory; that is recorded and the run goes on.
  fft      Micro-benchmark of the matvec core ifft2(g * fft2(x)) for x of shape (40, 2n, 2n) in complex64 and
           complex128 (the FP32/FP64 gap of this FFT-dominated operator, whatever the backend).

Method. Arguments are parsed first and the environment (JAX_PLATFORMS, thread variables, XLA_FLAGS) is set
before jax or bayes_scatter is imported, because XLA reads it at import. Each timed call is run --warmup times
first (compilation, excluded), then --repeats times with jax.block_until_ready; the table gives the median
and the interquartile range. "cpu util" is process CPU time over wall time during the repeats (about the number
of busy threads). "resid" is the largest true relative residual of the field solves; a value above 10 x tol
means the solve hit its iteration cap and the timing is not that of a converged solve. GMRES operator
applications per transmitter are counted for --matvec-counts transmitters with Forward.matvec_counts (a
diagnostic: one transmitter solved at a time, including the residual evaluation at each restart; the
stacked solve iterates until its slowest transmitter is done, so take the maximum).

Output. results/bench/<hostname>_<backend>[_<tag>]_<timestamp>.json (or --out) with provenance (host, OS, CPU
model, jax/jaxlib versions, devices, x64 flag, XLA_FLAGS and thread variables, load average, GPU memory
statistics), and a markdown table on stdout. A run that is interrupted leaves <name>.partial.json instead.
On a loaded machine the numbers are meaningless: the load average at start and end is recorded and flagged.
Exit status: 0 on success (out-of-memory cases are an expected outcome and are only recorded); 1 if some case
failed for another reason (the file is still written); 2 if the requested backend cannot be initialized; 130 if
interrupted.

Warm starts. "warm" replaces only the starting guess of the forward field solve. The adjoint solve inside the
gradient starts from zero here (an array aux, see forward.py), so a warm gradient is cheaper than a cold one by less than
a warm forward solve is cheaper than a cold one. Carrying the adjoint fields too (aux as a pair) is measured in
scripts/measure_adjoint_warm.py (DESIGN section 16).

Examples
  python scripts/bench_backend.py --platform cpu --threads 1 --grids 32:2,64:5 --modes forward,grad
  taskset -c 0-3 python scripts/bench_backend.py --platform cpu --threads 4 --modes grad,batched
  python scripts/bench_backend.py --platform gpu --modes grad,batched,fft --batch 1,8,32
"""

import argparse
import datetime
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Neither numpy nor jax is imported here: numpy's BLAS reads OPENBLAS_NUM_THREADS / OMP_NUM_THREADS when numpy is
# imported, and XLA reads JAX_PLATFORMS / XLA_FLAGS / NPROC when jax is. Both are imported by ``load_jax`` after
# ``apply_environment`` has set the environment.
np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_run  # noqa: E402  (standard library only, so it is safe to import before jax)

MODES = ("forward", "grad", "batched", "fft")
ENV_KEYS = ("JAX_PLATFORMS", "XLA_FLAGS", "NPROC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_ENABLE_X64")
N_TX = 40  # transmitters of the Ch. 15 setup (the FFT micro-benchmark uses the same stack height)


# ----------------------------------------------------------------------------------------------
# Arguments and environment (no jax here)
# ----------------------------------------------------------------------------------------------


def parse_grids(text):
    """``"32:2,64:5"`` -> ``[(32, 2.0), (64, 5.0)]``."""
    grids = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.fullmatch(r"(\d+):([0-9]*\.?[0-9]+)", item)
        if not m or int(m.group(1)) < 2 or float(m.group(2)) <= 0:
            raise argparse.ArgumentTypeError(f"bad grid {item!r}; expected n:side such as 64:5")
        grids.append((int(m.group(1)), float(m.group(2))))
    if not grids:
        raise argparse.ArgumentTypeError("no grids given")
    return grids


def parse_int_list(text):
    try:
        vals = [int(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad integer list {text!r}") from exc
    if not vals or any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return vals


def parse_modes(text):
    modes = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad or not modes:
        raise argparse.ArgumentTypeError(f"modes must be a comma list of {MODES}, got {text!r}")
    return modes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grids", type=parse_grids, default=parse_grids("32:2,64:2,64:5,96:5,128:5,160:5"),
                    help="comma list of n:side (pixels per side : DOI side in wavelengths)")
    ap.add_argument("--modes", type=parse_modes, default=list(MODES), help=f"comma list of {MODES}")
    ap.add_argument("--batch", type=parse_int_list, default=[1, 8, 32, 128], help="batch sizes of the batched mode")
    ap.add_argument("--repeats", type=int, default=5, help="timed repetitions (median and IQR over them)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed calls before the repetitions (compilation)")
    ap.add_argument("--eps-r", type=float, default=2.0, help="relative permittivity of the Austria phantom")
    ap.add_argument("--tol", type=float, default=1e-10, help="forward solver relative tolerance")
    ap.add_argument("--restart", type=int, default=100, help="GMRES restart length (Forward default)")
    ap.add_argument("--maxiter", type=int, default=30, help="GMRES restart cycles cap (Forward default)")
    ap.add_argument("--snr", type=float, default=30.0, help="dB, sets the noise level of the synthetic data")
    ap.add_argument("--xi", choices=["truth", "prior"], default="truth", help="timing point: near the truth or a "
                    "prior draw")
    ap.add_argument("--spread", type=float, default=0.05, help="std of the per-lane perturbation of xi (batched)")
    ap.add_argument("--warm-delta", type=float, default=0.02, help="std of the perturbation of xi that produces "
                    "the warm-start fields")
    ap.add_argument("--matvec-counts", type=int, default=5, help="transmitters whose GMRES operator applications "
                    "are counted in the grad mode (0 disables)")
    ap.add_argument("--fft-batch", type=int, default=N_TX, help="leading dimension of the FFT micro-benchmark")
    ap.add_argument("--no-mem-guard", action="store_true", help="on the CPU backend, attempt batched cases whose "
                    "estimated Krylov basis exceeds the available memory (may kill the process)")
    ap.add_argument("--platform", choices=["auto", "cpu", "gpu"], default="auto",
                    help="cpu sets JAX_PLATFORMS=cpu; gpu sets JAX_PLATFORMS=cuda,cpu (fails if no GPU is visible)")
    ap.add_argument("--threads", type=int, default=None, help="limit the CPU threads of XLA / BLAS / OpenMP to this "
                    "number (NPROC and friends; default: leave the environment alone). 1 is exact; for 2 or 3 the XLA "
                    "pool is 4, because a pool of 2 hangs. Combine with taskset to pin to that many cores")
    ap.add_argument("--xla-flags", default=None, help="extra XLA_FLAGS, merged into the environment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="inserted in the default output file name")
    ap.add_argument("--out", default=None, help="output JSON path (default: <out-dir>/<host>_<backend>_<time>.json)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "bench"))
    args = ap.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or (args.threads is not None and args.threads < 1):
        ap.error("--repeats must be >= 1, --warmup >= 0, --threads >= 1")
    return args


def apply_environment(args):
    """Set JAX_PLATFORMS, thread variables and XLA_FLAGS from ``args``; return what was changed.

    Must run before jax is imported: XLA reads the environment when the backend is created.
    """
    if "jax" in sys.modules:
        raise RuntimeError("jax is already imported; the environment can no longer be set. Run this as a script.")
    env, changed = os.environ, {}

    def put(key, value):
        if env.get(key) != value:
            changed[key] = value
            env[key] = value

    if args.platform == "cpu":
        put("JAX_PLATFORMS", "cpu")
    elif args.platform == "gpu":
        put("JAX_PLATFORMS", "cuda,cpu")  # cpu too: jax.debug.callback needs a CPU device
    if args.platform != "cpu" and "XLA_PYTHON_CLIENT_PREALLOCATE" not in env:
        put("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # memory statistics then show what the run needs
    if args.threads:
        for key, value in batch_run.thread_env(args.threads, env.get("XLA_FLAGS"),
                                               cpu_flags=args.platform != "gpu").items():
            put(key, value)
    if args.xla_flags:
        put("XLA_FLAGS", batch_run.merge_xla_flags(env.get("XLA_FLAGS"), args.xla_flags.split()))
    return changed


# ----------------------------------------------------------------------------------------------
# Timing and summaries
# ----------------------------------------------------------------------------------------------


def summarize(times):
    """Median, quartiles, IQR, min, max (seconds) of a list of timings."""
    t = np.asarray(times, dtype=float)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    return {"median_s": float(med), "q1_s": float(q1), "q3_s": float(q3), "iqr_s": float(q3 - q1),
            "min_s": float(t.min()), "max_s": float(t.max()), "mean_s": float(t.mean()), "repeats": int(t.size)}


def load_average():
    """1, 5 and 15 minute load averages, rounded; None where the OS has none."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def os_thread_count():
    """Number of OS threads of this process (Linux /proc, macOS ps); None if unknown."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except OSError:
        pass
    text = batch_run._run(["ps", "-M", "-p", str(os.getpid())])
    return max(len(text.splitlines()) - 1, 0) if text else None


def is_oom(exc):
    msg = f"{type(exc).__name__} {exc}".lower()
    return isinstance(exc, MemoryError) or any(s in msg for s in ("resource_exhausted", "out of memory", "oom"))


def error_row(grid, mode, exc, **extra):
    msg = f"{type(exc).__name__}: {str(exc).strip()[:300]}"
    return {"grid": grid, "mode": mode, "variant": None, "error": msg, "oom": is_oom(exc), **extra}


def failed_cases(rows):
    """Error rows that are not an expected out-of-memory outcome (nor a case skipped because a smaller B failed)."""
    return [r for r in rows if r.get("error") and not r.get("oom") and not r["error"].startswith("skipped")]


# ----------------------------------------------------------------------------------------------
# Everything below needs jax (imported by ``load_jax`` after the environment is set)
# ----------------------------------------------------------------------------------------------

jax = jnp = None


def load_jax():
    """Import numpy, jax and bayes_scatter (float64). Call only after ``apply_environment``."""
    global jax, jnp, np
    import numpy as _np
    import jax as _jax
    import jax.numpy as _jnp

    import bayes_scatter  # noqa: F401  (enables float64)

    np, jax, jnp = _np, _jax, _jnp
    return _jax


def time_calls(fn, repeats, warmup):
    """Time ``fn()`` (which returns jax arrays): ``warmup`` untimed calls, then ``repeats`` timed ones.

    Returns (summary dict, warm-up seconds, cpu utilization during the repeats).
    """
    t0 = time.perf_counter()
    for _ in range(max(warmup, 0)):
        jax.block_until_ready(fn())
    warm_s = time.perf_counter() - t0
    times = []
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t)
    wall = time.perf_counter() - wall0
    util = (time.process_time() - cpu0) / wall if wall > 0 else None
    return summarize(times), warm_s, util


def timing_row(grid, mode, variant, fn, args, *, batch=None, **extra):
    stats, warm_s, util = time_calls(fn, args.repeats, args.warmup)
    lanes = batch or 1
    return {"grid": grid, "mode": mode, "variant": variant, "batch": batch, **stats,
            "evals_per_s": lanes / stats["median_s"], "warmup_s": warm_s, "cpu_util": util, "error": None, **extra}


def near_truth_xi(field, chi_true, rel_reg=1e-2):
    """Whitened coordinates xi with theta(xi) close to log eps_r of the phantom (smoothed to data resolution).

    theta = crop(Re ifft2(M fft2(xi))) with M = sqrt(S), so a Wiener-regularized inverse of the torus filter
    applied to log eps_r padded with zeros solves M fft2(xi) ~ fft2(theta_pad). The regularization
    rel_reg * max(M)^2 damps the wavenumbers where M is below 10% of its maximum, k beyond roughly 2.3 kappa
    (about 10 rad per wavelength, near the 2 k0 band the data can see), so the sharp phantom edges are not
    reproduced.
    """
    n, P = field.n, field.P
    pad = np.zeros((P, P))
    pad[:n, :n] = np.log1p(np.asarray(chi_true))
    m = np.sqrt(np.asarray(field.spectrum))
    xi = np.fft.ifft2(m / (m**2 + rel_reg * m.max() ** 2) * np.fft.fft2(pad)).real
    return jnp.asarray(xi)


def build_case(n, side, args):
    """Forward model, target and the timing point(s) for one grid. Returns a dict."""
    from bayes_scatter.forward import Forward
    from bayes_scatter.geometry import ch15_setup
    from bayes_scatter.likelihood import Target, add_noise, noise_std_from_snr
    from bayes_scatter.phantoms import austria_fixed
    from bayes_scatter.priors import GaussianField

    t0 = time.perf_counter()
    setup = ch15_setup(n=n, side=side)
    fwd = Forward(setup, tol=args.tol, restart=args.restart, maxiter=args.maxiter)
    chi_true = jnp.asarray(austria_fixed(setup.grid, args.eps_r))
    d_clean, _ = fwd(chi_true)
    s = noise_std_from_snr(d_clean, args.snr)
    d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
    field = GaussianField(setup.grid, "matern", sigma=0.5, ell=0.4, nu=1.5, mean=0.0)
    target = Target(fwd, field, "log_eps", d_obs, s)
    key = jax.random.key(args.seed + 1)
    xi0 = near_truth_xi(field, chi_true) if args.xi == "truth" else field.sample_xi(key)
    k_near, k_lanes = jax.random.split(jax.random.fold_in(key, 7))
    xi_near = xi0 + args.warm_delta * jax.random.normal(k_near, xi0.shape)
    aux_cold = target.initial_aux()
    _, _, aux_warm = target.potential_and_grad(xi_near, aux_cold)
    jax.block_until_ready(aux_warm)
    chi0 = target.chi(xi0)
    info = {"n": n, "side": side, "xi_shape": list(field.xi_shape), "eps_r_max_at_xi": float(1.0 + chi0.max()),
            "eps_r_min_at_xi": float(1.0 + chi0.min()), "setup_s": time.perf_counter() - t0,
            "xi_source": args.xi, "restart": args.restart, "maxiter": args.maxiter, "tol": args.tol}
    return {"fwd": fwd, "target": target, "field": field, "xi0": xi0, "xi_near": xi_near, "chi0": chi0,
            "aux_cold": aux_cold, "aux_warm": aux_warm, "lane_key": k_lanes, "info": info}


def max_residual(fwd, chi, u):
    return float(jnp.max(fwd.residual(chi, u)))


def converged(resid, tol):
    return bool(resid <= 10.0 * tol)


def run_forward(case, grid, args):
    fwd, chi0, rows = case["fwd"], case["chi0"], []
    for variant, aux in (("cold", case["aux_cold"]), ("warm", case["aux_warm"])):
        resid = max_residual(fwd, chi0, fwd(chi0, aux)[1])
        rows.append(timing_row(grid, "forward", variant, lambda a=aux: fwd(chi0, a), args,
                               residual=resid, converged=converged(resid, args.tol)))
    return rows


def run_grad(case, grid, args):
    fwd, target, xi0, rows = case["fwd"], case["target"], case["xi0"], []
    counts = {}
    if args.matvec_counts > 0:
        idx = np.unique(np.linspace(0, fwd.n_tx - 1, min(args.matvec_counts, fwd.n_tx)).round().astype(int))
        for variant, aux in (("cold", case["aux_cold"]), ("warm", case["aux_warm"])):
            counts[variant] = fwd.matvec_counts(case["chi0"], u0=aux, transmitters=idx).tolist()
        counts["transmitters"] = idx.tolist()
    for variant, aux in (("cold", case["aux_cold"]), ("warm", case["aux_warm"])):
        _, _, u = target.potential_and_grad(xi0, aux)
        resid = max_residual(fwd, case["chi0"], u)
        extra = {"residual": resid, "converged": converged(resid, args.tol)}
        if counts:
            c = np.asarray(counts[variant])
            extra["matvecs_per_transmitter"] = {"transmitters": counts["transmitters"], "counts": counts[variant],
                                                "min": int(c.min()), "median": float(np.median(c)), "max": int(c.max())}
        rows.append(timing_row(grid, "grad", variant, lambda a=aux: target.potential_and_grad(xi0, a), args, **extra))
    return rows


def available_memory_bytes():
    """Memory a batched case may use: free device memory on a GPU, MemAvailable (or 80% of RAM) on the CPU."""
    dev = jax.devices()[0]
    if dev.platform != "cpu":
        stats = dev.memory_stats() or {}
        if "bytes_limit" in stats:
            return stats["bytes_limit"] - stats.get("bytes_in_use", 0)
        return None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    total = batch_run.mem_total_bytes()
    return int(0.8 * total) if total else None


def krylov_basis_bytes(n, n_tx, restart, batch):
    """Dominant memory of a stacked complex128 GMRES cycle: (restart + 1) basis vectors per lane and system."""
    return 16 * (restart + 1) * batch * n_tx * n * n


def cleanup():
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:  # noqa: BLE001  (older jax)
        pass


def run_batched(case, grid, args, emit):
    """Batched value-and-grad for each B in --batch; ``emit(row)`` is called as rows complete."""
    fwd, target, xi0 = case["fwd"], case["target"], case["xi0"]
    n = fwd.n
    vg = jax.jit(jax.vmap(target.phi_fn))
    on_cpu = jax.devices()[0].platform == "cpu"
    failed_at = None
    for B in sorted(args.batch):
        label = {"grid": grid, "mode": "batched", "variant": "warm", "batch": B}
        if failed_at is not None:
            emit({**label, "error": f"skipped: B={failed_at} already failed", "oom": False})
            continue
        est = krylov_basis_bytes(n, fwd.n_tx, args.restart, B)
        avail = available_memory_bytes()
        if on_cpu and not args.no_mem_guard and avail is not None and est > 0.9 * avail:
            failed_at = B
            emit({**label, "error": f"skipped: estimated Krylov basis {est / 2**30:.1f} GiB exceeds 90% of the "
                  f"{avail / 2**30:.1f} GiB available (--no-mem-guard to try anyway)", "oom": True})
            continue
        try:
            k1, k2 = jax.random.split(jax.random.fold_in(case["lane_key"], B))
            xis = xi0 + args.spread * jax.random.normal(k1, (B,) + xi0.shape)
            xis_near = xis + args.warm_delta * jax.random.normal(k2, xis.shape)
            aux0 = jnp.broadcast_to(case["aux_cold"], (B,) + case["aux_cold"].shape)
            _, _, auxs = vg(xis_near, aux0)  # per-lane warm start from a nearby point
            jax.block_until_ready(auxs)
            del aux0
            _, _, u = vg(xis, auxs)
            resid = float(jnp.max(jax.vmap(fwd.residual)(target.chi(xis), u)))
            del u
            row = timing_row(grid, "batched", "warm", lambda: vg(xis, auxs), args, batch=B, residual=resid,
                             converged=converged(resid, args.tol), est_basis_gib=est / 2**30)
            emit(row)
        except Exception as exc:  # noqa: BLE001  (OOM and anything else: record, free memory, go on)
            failed_at = B
            emit(error_row(grid, "batched", exc, variant="warm", batch=B, est_basis_gib=est / 2**30))
            cleanup()
        finally:
            xis = xis_near = auxs = None  # noqa: F841
    return


def run_fft(ns, args, emit):
    """ifft2(g * fft2(x)) for x of shape (fft_batch, 2n, 2n), complex64 and complex128."""
    rng = np.random.default_rng(args.seed)
    for n in sorted(set(ns)):
        grid = f"fft n={n}"
        times = {}
        for name, dtype in (("c64", np.complex64), ("c128", np.complex128)):
            try:
                shape = (args.fft_batch, 2 * n, 2 * n)
                x = jnp.asarray((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype))
                g = jnp.asarray((rng.standard_normal(shape[1:]) + 1j * rng.standard_normal(shape[1:])).astype(dtype))
                f = jax.jit(lambda x, g: jnp.fft.ifft2(g * jnp.fft.fft2(x)))
                out_dtype = str(f(x, g).dtype)
                if out_dtype != str(np.dtype(dtype)):
                    raise RuntimeError(f"result dtype {out_dtype} differs from the input dtype {np.dtype(dtype)}")
                row = timing_row(grid, "fft", name, lambda: f(x, g), args, shape=list(shape), out_dtype=out_dtype)
                size = shape[1] * shape[2]
                row["gflops"] = 2 * 5.0 * shape[0] * size * math.log2(size) / row["median_s"] / 1e9
                row["nominal_gb_s"] = 4.0 * x.nbytes / row["median_s"] / 1e9
                times[name] = row["median_s"]
                emit(row)
            except Exception as exc:  # noqa: BLE001
                emit(error_row(grid, "fft", exc, variant=name))
                cleanup()
        if len(times) == 2:
            emit({"grid": grid, "mode": "fft", "variant": "c128/c64", "ratio": times["c128"] / times["c64"],
                  "error": None})


# ----------------------------------------------------------------------------------------------
# Provenance and reporting
# ----------------------------------------------------------------------------------------------


def package_versions(names):
    from importlib import metadata

    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def memory_stats():
    out = {}
    for d in jax.devices():
        try:
            stats = d.memory_stats()
        except Exception:  # noqa: BLE001
            stats = None
        if stats:
            out[str(d)] = {k: stats[k] for k in ("bytes_limit", "bytes_in_use", "peak_bytes_in_use") if k in stats}
    return out


def x64_flag():
    try:
        return bool(jax.config.jax_enable_x64)
    except AttributeError:
        return bool(jnp.asarray(1.0).dtype == jnp.float64)


def provenance(args, env_changed, started):
    import jaxlib

    info = batch_run.host_info()
    devices = jax.devices()
    load = load_average()
    prov = {
        **info,
        "started": started,
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": getattr(jaxlib, "__version__", None),
        "packages": package_versions(["numpy", "scipy", "jax-cuda12-plugin", "jax-cuda12-pjrt", "jax-cuda13-plugin",
                                      "jax-cuda13-pjrt", "nvidia-cudnn-cu12", "nvidia-cufft-cu12",
                                      "nvidia-cublas-cu12"]),
        "devices": [str(d) for d in devices],
        "device_kinds": sorted({d.device_kind for d in devices}),
        "default_backend": jax.default_backend(),
        "x64_enabled": x64_flag(),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
        "env_set_by_script": env_changed,
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "loadavg_start": load,
        "nvidia_smi": batch_run._run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                                      "--format=csv,noheader"]),
        "memory_stats_start": memory_stats(),
    }
    ncpu = info["logical_cpus"] or 1
    prov["machine_busy_at_start"] = bool(load and load[0] > 0.25 * ncpu)
    return prov


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def note_of(row):
    if row.get("error"):
        return row["error"]
    parts = []
    if row["mode"] == "fft":
        if "gflops" in row:
            parts.append(f"{row['gflops']:.1f} GFLOP/s, nominal {row['nominal_gb_s']:.0f} GB/s")
        if "ratio" in row:
            parts.append(f"c128 takes {row['ratio']:.2f}x the time of c64")
    mv = row.get("matvecs_per_transmitter")
    if mv:
        parts.append(f"matvecs/tx min {mv['min']} med {mv['median']:g} max {mv['max']}")
    if row.get("converged") is False:
        parts.append("NOT CONVERGED (iteration cap): timing is of an unconverged solve")
    return "; ".join(parts)


def markdown_table(rows):
    head = ["grid", "mode", "variant", "B", "median ms", "IQR ms", "evals/s", "cpu util", "resid", "note"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        cells = [r["grid"], r["mode"], r.get("variant") or "", "" if r.get("batch") is None else str(r["batch"]),
                 fmt(None if "median_s" not in r else r["median_s"] * 1e3, ".2f"),
                 fmt(None if "iqr_s" not in r else r["iqr_s"] * 1e3, ".2f"),
                 fmt(r.get("evals_per_s"), ".3g"), fmt(r.get("cpu_util"), ".2f"),
                 fmt(r.get("residual"), ".1e"), note_of(r)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text) or "x"


def main(argv=None):
    args = parse_args(argv)
    env_changed = apply_environment(args)  # before jax is imported
    started = datetime.datetime.now().astimezone()
    try:
        load_jax()
        devices = jax.devices()
    except Exception as exc:  # noqa: BLE001  (RuntimeError, or AssertionError when no plugin is installed)
        hint = " Is jax[cuda12] installed and does nvidia-smi work?" if args.platform == "gpu" else ""
        print(f"bench_backend: cannot initialize the requested JAX backend ({type(exc).__name__}: {exc}).{hint}",
              file=sys.stderr)
        return 2
    backend = jax.default_backend()
    if args.platform == "gpu" and backend == "cpu":
        print("bench_backend: --platform gpu but JAX sees only the CPU", file=sys.stderr)
        return 2
    if args.platform == "cpu" and backend != "cpu":
        print(f"bench_backend: --platform cpu but the default backend is {backend}", file=sys.stderr)
        return 2

    prov = provenance(args, env_changed, started.isoformat(timespec="seconds"))
    host = sanitize(prov["hostname"].split(".")[0])
    stamp = started.strftime("%Y%m%dT%H%M%S")
    name = f"{host}_{sanitize(backend)}{'_' + sanitize(args.tag) if args.tag else ''}_{stamp}.json"
    out = Path(args.out) if args.out else Path(args.out_dir) / name
    partial = out.with_name(out.stem + ".partial.json")
    print(f"bench_backend: {prov['cpu_model']} / {', '.join(prov['devices'])}, backend {backend}, "
          f"jax {prov['jax']}, x64 {prov['x64_enabled']}, threads {os.environ.get('NPROC', 'default')}, "
          f"load {prov['loadavg_start']}", file=sys.stderr, flush=True)
    if prov["machine_busy_at_start"]:
        print("bench_backend: WARNING: the machine is busy (1-minute load average above a quarter of the logical "
              "CPUs); timings will be unreliable", file=sys.stderr, flush=True)

    rows, cases = [], {}
    payload = {"schema": 1, "provenance": prov, "args": vars(args), "cases": cases, "results": rows,
               "interrupted": False, "finished": None}

    def emit(row):
        rows.append(row)
        what = f"{row['grid']} {row['mode']} {row.get('variant') or ''}" + (f" B={row['batch']}" if row.get("batch") else "")
        if row.get("error"):
            print(f"  {what}: {row['error']}", file=sys.stderr, flush=True)
        elif "median_s" in row:
            print(f"  {what}: {row['median_s'] * 1e3:.2f} ms (IQR {row['iqr_s'] * 1e3:.2f})", file=sys.stderr, flush=True)
        write_json(partial, payload)

    status = 0
    try:
        solver_modes = [m for m in args.modes if m != "fft"]
        if solver_modes:
            for n, side in args.grids:
                grid = f"{n}:{side:g}"
                print(f"grid {grid}", file=sys.stderr, flush=True)
                try:
                    case = build_case(n, side, args)
                except Exception as exc:  # noqa: BLE001
                    emit(error_row(grid, "setup", exc))
                    cleanup()
                    continue
                cases[grid] = case["info"]
                for mode in solver_modes:
                    try:
                        if mode == "forward":
                            for r in run_forward(case, grid, args):
                                emit(r)
                        elif mode == "grad":
                            for r in run_grad(case, grid, args):
                                emit(r)
                        else:
                            run_batched(case, grid, args, emit)
                    except Exception as exc:  # noqa: BLE001
                        emit(error_row(grid, mode, exc))
                        cleanup()
                cases[grid]["memory_stats_after"] = memory_stats()
                del case
                cleanup()
        if "fft" in args.modes:
            run_fft([n for n, _ in args.grids], args, emit)
    except KeyboardInterrupt:
        payload["interrupted"] = True
        status = 130
        print("bench_backend: interrupted; partial results kept", file=sys.stderr)

    payload["provenance"]["loadavg_end"] = load_average()
    payload["provenance"]["memory_stats_end"] = memory_stats()
    payload["provenance"]["os_threads_end"] = os_thread_count()
    payload["finished"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    failed = failed_cases(rows)
    payload["failed_cases"] = len(failed)
    if status == 0:
        write_json(out, payload)
        partial.unlink(missing_ok=True)
        where = out
        if failed:  # results are kept, but a job runner must not take this for a clean run
            status = 1
            print(f"bench_backend: {len(failed)} case(s) failed for a reason other than running out of memory "
                  f"(see the error rows); exit status 1", file=sys.stderr)
    else:
        write_json(partial, payload)
        where = partial
    print(markdown_table(rows))
    print(f"\nbackend {backend}; devices {', '.join(prov['devices'])}; jax {prov['jax']}; load average "
          f"{prov['loadavg_start']} -> {payload['provenance']['loadavg_end']}; wrote {where}")
    return status


if __name__ == "__main__":
    sys.exit(main())
