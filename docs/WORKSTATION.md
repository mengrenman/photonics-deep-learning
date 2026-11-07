# Running on the Linux workstation (Threadripper 3990X + RTX 3090)

Target machine: AMD Threadripper 3990X (64 cores / 128 threads, Zen 2: 16 core complexes of 4 cores, each
complex sharing a 16 MB L3; 4-channel DDR4), 256 GB RAM, NVIDIA RTX 3090 (24 GB; FP64 runs at 1/64 of the
FP32 rate). The library runs in float64 (`jax_enable_x64` is set in `bayes_scatter/__init__.py`).

Two new scripts support this: `scripts/bench_backend.py` (what does one evaluation cost on this backend?) and
`scripts/batch_run.py` (run many independent jobs with CPU/GPU slots and core pinning). Order of work on the
first day: (a) SSH, (b) the repository, (c) environment, (d) tests, then `batch_run.py --show-topology`, a
one-job GPU smoke through `batch_run.py` (in (f)), (e) a quick benchmark, and only then (f) sweeps.

**Verification status.** Everything here was written and smoke-tested on the Mac (macOS, CPU backend, JAX
0.11.2) with tiny settings, and `tests/test_batch_run.py` exercises the pinning logic against a fake sysfs tree
and through a fake `taskset`. Nothing has run on Linux or on the GPU yet; the commands in (c) are what establish
that, and the items marked "to verify" below are assumptions to check on the machine.

**No number measured on the Mac is a performance result.** Those smoke runs used n = 16 to 48, one or a few
threads, and a machine whose load average was several times its core count, so every timing they printed
is meaningless for the workstation and none is quoted here as a result. The only measurements quoted in this
document are ones that do not depend on load (thread counts, which thread states were seen, whether a process
hangs) and the CPU-time ratios of the thread-limiting section, which are marked as indicative.

## (a) SSH from the Mac

Every command below assumes the host alias `tr` works from the Mac. SSH is not set up yet, so do this first.

1. On the workstation (at its keyboard, or however you can reach it today), make sure an SSH server runs and
   note its address and your login name: `sudo systemctl enable --now ssh` (the unit is called `sshd` on Fedora
   and Arch), `hostname -I`, `whoami`.
2. On the Mac, create a key if you have none (`ls ~/.ssh/id_ed25519` fails): `ssh-keygen -t ed25519`, and give
   it a passphrase.
3. Add this to `~/.ssh/config` on the Mac, replacing the two bracketed values (the only things this document
   cannot know):

   ```text
   Host tr
       HostName <address or hostname of the workstation>
       User <your login on the workstation>
       ServerAliveInterval 30
   ```
4. Copy the key once: `ssh-copy-id tr` (asks for the workstation password, and for confirmation of the host
   key fingerprint, which you can compare with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the
   workstation). Afterwards `ssh tr` logs in with the key.

Check it with

```bash
ssh tr 'hostname; nvidia-smi -L; lscpu | grep -E "Model name|^CPU\(s\)|L3"'
```

which should show the 3990X (64 cores, 128 CPUs), an RTX 3090 and 16 L3 instances of 16 MiB. If `nvidia-smi`
fails, install the NVIDIA driver first (see the driver note in (c)).

## (b) Moving the repository

The code goes through git; `references/` is git-ignored (copyrighted book material) and must never be copied
to the workstation by any other means (no `scp -r`, no `rsync` of the project root).

```bash
# once, on the Mac, from the repository root (commit your work first: there is nothing to push before the
# first commit, and the branch must be called main)
git branch --show-current          # main; if it prints anything else: git branch -M main
ssh tr 'git init --bare ~/repos/photonics-dl.git && git -C ~/repos/photonics-dl.git symbolic-ref HEAD refs/heads/main'
git remote add tr tr:repos/photonics-dl.git
git push -u tr main

# once, on the workstation
ssh tr 'mkdir -p ~/work && git clone ~/repos/photonics-dl.git ~/work/photonics-dl'

# later updates
git push tr main && ssh tr 'cd ~/work/photonics-dl && git pull --ff-only'
```

The `symbolic-ref` step matters: a new bare repository points its `HEAD` at the default branch name of the
workstation's git, usually `master`. Without the step, the clone prints "remote HEAD refers to nonexistent ref,
unable to checkout" and leaves an empty working tree (reproduced in a sandbox with `init.defaultBranch=master`;
with the step the clone has the files on `main`).

`results/**/*.npz` is git-ignored too, so chains stay on the machine that produced them until you rsync them (g).

## (c) Environment

On the workstation (`ssh tr`):

```bash
# Miniforge
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash   # then log in again

conda create -n photonics-dl python=3.12 -y && conda activate photonics-dl
cd ~/work/photonics-dl
pip install -e ".[dev]"
pip install -U "jax[cuda12]"        # CUDA 12 wheels: CUDA and cuDNN come from pip, no system toolkit needed
```

The Mac runs JAX 0.11.2. For comparable results use the same version on the workstation
(`pip install -U "jax[cuda12]==0.11.2"`; if pip cannot resolve it, drop the pin and note the version, which
`bench_backend.py` records in its output).

**Driver.** The CUDA 12 wheels need an NVIDIA driver of at least 525 on Linux (the JAX installation page; the
CUDA 13 wheels, `jax[cuda13]`, need 580). The RTX 3090 (Ampere, compute capability 8.6) is supported by both.
`nvidia-smi` must work before JAX can see the GPU; the driver is installed from the distribution or NVIDIA,
not from pip. The JAX installation page (read on 2026-10-05) lists `jax[cuda13]` as the recommended extra and
`jax[cuda12]` as the legacy one; the CUDA 12 wheels support SM 5.2 and newer, the CUDA 13 wheels SM 7.5 and
newer. Check the current minimum on that page when you upgrade JAX. Use `JAX_PLATFORMS=cuda,cpu` for "the GPU or
fail" (as `batch_run.py` and `bench_backend.py --platform gpu` do): the plain default falls back to the CPU
with only a log warning if the CUDA plugin cannot load, while a listed platform that cannot start is an error
(checked on the workstation with the GPU hidden: `Unable to initialize backend 'cuda'`). Do not use `cuda`
alone: `jax.debug.callback`, which the samplers' evaluation counters use, needs a CPU device, and without one
those programs fail at run time (4 GPU test failures on 2026-11-08 had this cause).

**Verification.**

```bash
python - <<'EOF'
import jax, jax.numpy as jnp
import bayes_scatter
print(jax.__version__, jax.devices(), jax.default_backend())   # [CudaDevice(id=0)] gpu
print(jnp.ones(1).dtype)                                       # float64: x64 is on
EOF

# a tiny forward solve on the default backend, then on the CPU: the norms must agree to about 1e-9 relative
cat > /tmp/tiny_solve.py <<'EOF'
import jax, jax.numpy as jnp
import bayes_scatter
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.phantoms import cylinder

setup = ch15_setup(n=32, side=2.0)
fwd = Forward(setup, tol=1e-10)
chi = jnp.asarray(cylinder(setup.grid, 0.5, 1.5))
d, u = fwd(chi)
print(jax.default_backend(), d.dtype, d.shape, f"residual {float(jnp.max(fwd.residual(chi, u))):.1e}",
      f"norm {float(jnp.linalg.norm(d)):.12e}")
EOF
python /tmp/tiny_solve.py                        # gpu complex128 (40, 60) residual ~1e-11 norm 4.217235372566e-01
JAX_PLATFORMS=cpu NPROC=4 python /tmp/tiny_solve.py
```

The norm printed on the Mac (CPU, JAX 0.11.2) is `4.217235372566e-01`; the residual there was `9.3e-12`
(rerun on 2026-10-05 with the script above). On the GPU expect agreement to about 1e-9 relative, not to all digits.

## (d) Tests

```bash
cd ~/work/photonics-dl && conda activate photonics-dl
JAX_PLATFORMS=cpu NPROC=8 python -m pytest -q -m "not slow"     # CPU; NPROC keeps XLA from starting 128 threads
JAX_PLATFORMS=cpu NPROC=8 python -m pytest -q                   # including the slow tests
JAX_PLATFORMS=cuda,cpu python -m pytest -q -m "not slow"        # GPU (fails loudly if the GPU is not visible)
python -m pytest -q tests/test_batch_run.py                     # the job runner and bench_backend's import order
```

`tests/test_batch_run.py` runs no JAX computation of its own (a few subprocess tests start JAX on the CPU
briefly; `conftest.py` imports `jax`). The rest of the suite is small-grid float64 code, so the GPU run mostly checks that the CUDA path gives the same answers
(finite-difference gradient checks, Born comparisons) rather than that it is fast. A tolerance failure that
appears only on the GPU is worth reading before running anything long.

## (e) Benchmarking one evaluation

`bench_backend.py` times the forward model (40 transmitters), the value-and-gradient of the posterior target, a
`vmap`-batched gradient with per-lane warm starts, and a complex64/complex128 FFT micro-benchmark. It
records provenance (host, CPU, JAX versions, devices, `XLA_FLAGS`, thread variables, load average, GPU memory
statistics) in `results/bench/<host>_<backend>[_<tag>]_<time>.json` and prints a markdown table.
Check `uptime` first and do not benchmark while a sweep is running: the load average is recorded, and a busy
machine is flagged in the output.

```bash
# CPU, one thread on one core
taskset -c 0 python scripts/bench_backend.py --platform cpu --threads 1 --tag t1 \
    --grids 32:2,64:2,64:5,96:5 --modes forward,grad,fft

# CPU, four threads on one L3 group (cores 0-3 if --show-topology below says so)
taskset -c 0-3 python scripts/bench_backend.py --platform cpu --threads 4 --tag t4 \
    --grids 64:5,96:5,128:5 --modes forward,grad,batched --batch 1,8,32

# GPU, batched
python scripts/bench_backend.py --platform gpu --tag gpu --modes grad,batched,fft --batch 1,8,32,128
```

Reading the table:

* `cold` starts the Krylov solve from the incident field; `warm` from the fields of a solve at a nearby xi, the
  situation inside a chain. Only the forward field solve is warm-started: the adjoint solve inside the gradient
  always starts from zero (`forward.py`), so a warm gradient gains less than a warm forward solve. The batched
  mode is warm only (each lane starts from the fields of its own nearby xi). `evals/s` is `B / time` for the
  batched mode and `1 / time` for the others.
* In the batched mode the B lanes are B different parameter vectors (the timing point plus independent
  `--spread` noise per lane), each with its own warm-start fields; the vmapped result equals the single
  evaluation of each lane to round-off. The `median ms` column is the time of the whole batch.
* All timings are at one point near the truth, because the cost depends on the contrast. The point is the
  phantom smoothed to the prior's resolution, so its largest eps_r is below `--eps-r` (the JSON has
  `eps_r_max_at_xi` per grid).
* `resid` above ten times the tolerance, flagged `NOT CONVERGED`, means the solve hit its iteration cap, so
  the time is that of an unconverged solve.
* `matvecs/tx` counts operator applications per transmitter (`Forward.matvec_counts`, a diagnostic that solves one
  transmitter at a time; the stacked solve runs until its slowest transmitter is done, so read the maximum).
* `cpu util` is CPU time over wall time during the repeats, about the number of busy threads.
* The batched mode needs about `16 B * 101 * 40 * n^2` bytes for the GMRES basis at the default restart of 100:
  63 MiB per lane at n = 32, 253 MiB at 64, 568 MiB at 96, 1.0 GiB at 128, 1.6 GiB at 160 (plus working copies).
  On the 24 GB card, expect out-of-memory at large `B` and `n` (it is recorded per case and the run continues;
  larger `B` at that grid are then skipped). On the CPU backend a guard skips cases whose basis would exceed
  the available RAM. GPU memory is allocated on demand (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), so the peak
  bytes in the output are meaningful; the peak is a process-lifetime maximum, so to attribute it to one case run
  one grid per invocation.
* The batched mode is where the library may be inefficient, and this is a hypothesis to test, not a result:
  `jax.vmap` of the GMRES `while_loop` makes every lane iterate until the slowest has converged and may cost
  extra memory traffic on the whole Krylov basis at each iteration. Compare `evals/s` at `B = 1` and `B = 8`:
  a ratio well above 1 means batching pays, a ratio at or below 1 means it does not.
* Exit status: 0 on success (out-of-memory cases are recorded, not failures), 1 if some case failed for another
  reason (the JSON is still written, with `failed_cases`), 2 if the requested backend cannot start (for example
  `--platform gpu` without a working CUDA install), 130 if interrupted (only `<name>.partial.json` is kept, so a
  `batch_run.py` done marker never mistakes it for a finished run).
* The `fft` rows give the complex128/complex64 time ratio of `ifft2(g * fft2(x))`, the core of the operator, which
  is what the 1/64 FP64 rate of the 3090 does or does not hurt in practice. Run it before drawing conclusions
  about FP64 on the GPU.

**Throughput scaling of independent single-thread jobs** (decides how many CPU slots a sweep should use). Run
the same benchmark as `N` simultaneous copies, one per core, and compare the median time with the `N = 1`
value; copies that overlap in time are required, so use enough repeats for all of them to be timing at once and
check `provenance.started` and `finished` in the JSON files.

```bash
for N in 1 8 16 32 64; do
python - <<EOF
import json
N = $N
jobs = [{"name": f"scale_c{N}_{i:02d}",
         "cmd": ["scripts/bench_backend.py", "--platform", "cpu", "--threads", "1", "--grids", "64:5",
                 "--modes", "grad", "--repeats", "15", "--matvec-counts", "0",
                 "--out", f"results/bench/scale_c{N}_{i:02d}.json"],
         "done_marker": f"results/bench/scale_c{N}_{i:02d}.json"} for i in range(N)]
json.dump(jobs, open(f"/tmp/scale_{N}.json", "w"), indent=1)
EOF
python scripts/batch_run.py /tmp/scale_$N.json --cpu-slots $N --threads 1 --pin
done
```

## (f) Sweeps with `batch_run.py`

A jobs file is a JSON list; see `scripts/jobs/example_cpu_sweep.json` and `example_mixed_cpu_gpu.json`. Each job
has a unique `name`, a `cmd` (run from the repository root; `.py` scripts are run with the Python that runs
`batch_run.py`), a `device` (`cpu` or `gpu`), an optional `threads`, and a `done_marker` (default
`results/<name>/summary.json`, which is what `run_posterior.py --out <name>` writes, so keep `--out` equal to
`name`). Jobs whose marker exists are skipped unless `--force`, so a sweep can be interrupted and restarted.
`run_posterior.py` writes `summary.json` before `chains.npz` and the figures, so a marker alone can be left behind
by a job that was killed or crashed afterwards; therefore a job is also run again when its last recorded run
(`results/logs/<name>.meta.json`) says `failed`, `interrupted`, `timeout` or `running` (a runner that was killed).
A job run by hand, with no record, is trusted by its marker.

```bash
python scripts/batch_run.py --show-topology                                  # check the L3 groups first
python scripts/batch_run.py scripts/jobs/example_cpu_sweep.json --cpu-slots 3 --pin --dry-run
python scripts/batch_run.py scripts/jobs/example_cpu_sweep.json --cpu-slots 3 --pin
```

`--show-topology` should print 16 L3 groups of 4 physical cores, each core listed with its SMT sibling (for
example `0+64`). If it prints something else (one big group, or no L3 information), the BIOS or the kernel is
not exposing the core complexes and `--pin` cannot be trusted; run without `--pin`.

Per job it writes `results/logs/<name>.log` and `<name>.meta.json` (host, final command, environment overrides,
cores, start and end, wall seconds, exit code, status). Ctrl-C terminates the jobs' process groups (SIGTERM,
then SIGKILL after `--grace` seconds, or at once on a second Ctrl-C) and exits 130; the exit status is 1 if any
job failed. `--job-timeout S` terminates a job still running after S seconds of wall time and records it as
`timeout`: use it as the guard against a hung job, which otherwise holds its slot at 0% CPU forever (set it
to a few times the expected run time). Run long sweeps inside `tmux` (a dropped SSH connection otherwise kills
the runner and leaves its jobs running unsupervised).

**Slot counts for the 3990X.**

| goal | command line |
|---|---|
| 63 single-thread jobs + 1 GPU job | `--cpu-slots 63 --threads 1 --gpu-slots 1 --pin` |
| 15 four-thread jobs (one per L3 group) + 1 GPU job | `--cpu-slots 15 --threads 4 --gpu-slots 1 --pin` |
| CPU only, all physical cores, one thread each | `--cpu-slots 64 --threads 1 --pin` |
| CPU only, one 4-thread job per L3 group | `--cpu-slots 16 --threads 4 --pin` |

In the 15 + 1 row the last L3 group keeps 3 spare cores, which a 4-thread job cannot use (a job with `"threads": 3` can).
The single-thread rows are the well-tested configuration; the multi-thread rows depend on the XLA pool behavior
described below and are untested on Linux.

With `--pin`, a job gets `T` physical cores (the first hardware thread of each) inside one L3 group, started
under `taskset -c`; no two running jobs share a CPU, and `T` cannot exceed 4. Free groups are used round-robin
(`--pack` fills one group first). Each GPU slot reserves a core for its host thread so CPU jobs cannot take
it, which is why the totals above leave one core out. `--smt` lets jobs use sibling hardware threads once every
core's first thread is taken (two jobs then share a physical core; only for experiments). A pinned request that
does not fit (for example 64 slots plus a GPU job) is refused with the number that fit.

GPU jobs run with `JAX_PLATFORMS=cuda,cpu` (an unavailable GPU is an error, not a silent fall back to the CPU),
`CUDA_VISIBLE_DEVICES` from `--gpu-devices` (default `0`), `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9 / (jobs sharing that device)`. One GPU slot is the sensible default; two
only help if a single job does not keep the card busy. A GPU job does not get `NPROC` or the XLA CPU flags (its
CPU client only runs callbacks); it does get the BLAS/OpenMP variables set to `--gpu-threads` (default 1), and with
`--pin` it runs under `taskset` on that many reserved cores.

**Smoke the GPU path through `batch_run.py` before a sweep** (this also checks a GPU job pinned to one core,
which is how a sweep will run it), with a CPU job beside it:

```bash
cat > /tmp/smoke_jobs.json <<'EOF'
[{"name": "smoke_gpu", "device": "gpu", "done_marker": "results/bench/smoke_gpu.json",
  "cmd": ["scripts/bench_backend.py", "--platform", "gpu", "--grids", "16:2", "--modes", "grad,fft", "--repeats", "3",
          "--out", "results/bench/smoke_gpu.json"]},
 {"name": "smoke_cpu", "device": "cpu", "done_marker": "results/bench/smoke_cpu.json",
  "cmd": ["scripts/bench_backend.py", "--platform", "cpu", "--threads", "1", "--grids", "16:2", "--modes", "grad",
          "--repeats", "3", "--out", "results/bench/smoke_cpu.json"]}]
EOF
python scripts/batch_run.py /tmp/smoke_jobs.json --cpu-slots 1 --gpu-slots 1 --pin --job-timeout 600
```

Both jobs should end `ok`; if `smoke_gpu` hangs or fails, read `results/logs/smoke_gpu.log` before anything else.

**How CPU jobs are limited to `T` threads.** Measured on JAX 0.11.2 (macOS, CPU backend, 16 logical CPUs), one
process at a time, with a forward-and-gradient loop at n = 24 and `ps -M` sampled every 0.3 s for 12 s:

| environment | OS threads | most threads running at once | CPU time / wall (loaded machine: indicative only) |
|---|---|---|---|
| nothing set | 78 | 16 | 1.8 |
| `--xla_cpu_multi_thread_eigen=false` alone | 66 | 11 | 1.4 |
| `NPROC=1` and that flag | 36 | 1 | 0.5 |
| `NPROC=4`, `--xla_cpu_multi_thread_eigen=true` | 42 | 4 | 0.9 |

* The XLA CPU thread pools are sized by the environment variable `NPROC` (read by jaxlib when the CPU client is
  created). The thread count grows by 2 per unit of `NPROC`: two pools of `NPROC` threads plus a remainder that
  depends on the workload. Without it, each of 64 concurrent jobs would start pools as large as the machine.
* `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` alone does not make a job single-threaded: it only keeps
  Eigen's matrix products (and, judging by the hang described below, the parallel FFT) out of the pool. Both
  settings are needed for one thread, and then at most one thread was ever seen running.
* `intra_op_parallelism_threads` is not an XLA flag in this jaxlib (it is a field of a protobuf). Written
  `--intra_op_parallelism_threads=1` it makes XLA abort with "Unknown flag"; without the dashes the parser
  silently ignores it. The often-quoted `"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
  therefore reduces to the first flag.

**What is and is not controlled.** One thread (`T = 1`) is controlled: Eigen off and `NPROC=1`. For `T > 1`
`NPROC` sets the size of two pools and the BLAS/OpenMP variables are set to `T`, so the number of threads that
can run at once is *about* `T`, not exactly `T`: the measurement above saw at most 4 running with `NPROC=4`,
but nothing in XLA promises it. Only `--pin` (`taskset`) bounds the cores a job can use. Without it, "`T`
threads" means "pools sized for `T`".

**A hang to know about.** With Eigen multi-threading on (every `T > 1`), XLA's FFT thunk blocks its pool worker
while helper tasks it queued on the same pool run. If all workers are blocked inside an FFT, the process stops at
0% CPU (a stack sample showed two pool workers of an `NPROC=2` process both inside `ducc0::c2c`). With a bare
forward-and-gradient loop with respect to chi (the script below), `NPROC=2` hung in 7 of 7 runs, `NPROC` = 3, 4
and 8 did not hang in 11 runs, and no run with Eigen off hung. The Target-based modes of `bench_backend.py` (n up
to 48) and a reduced `run_posterior.py` at n = 24 did not hang at `NPROC=2` or 3, so whether a graph can hang depends
on how many FFTs it runs at once. Consequences: `T = 1` is safe by construction (nothing waits on the pool);
`batch_run.py` and `bench_backend.py --threads` never use a pool of two or three (a job with `T = 2` or `3` gets
`NPROC=4`, so it may run up to four threads unless pinned); `--job-timeout` bounds the damage of any hang that is
not predicted here. This was seen on macOS; whether Linux builds of the same jaxlib behave the same is to verify.

So a CPU job gets `JAX_PLATFORMS=cpu`, `NPROC` (1 for `T = 1`, otherwise the larger of `T` and 4),
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` equal to `T`, and
`XLA_FLAGS` with `--xla_cpu_multi_thread_eigen=false` for `T = 1` and `=true` otherwise (an existing `XLA_FLAGS`
is kept, with this flag replaced). `NPROC` is not a documented JAX setting, so recheck after a JAX upgrade, on
Linux, with the thread counts and the hang probe:

```bash
for t in 1 4 16; do NPROC=$t JAX_PLATFORMS=cpu python -c \
  "import jax, os; jax.numpy.ones(3).block_until_ready(); print($t, len(os.listdir('/proc/self/task')))"; done
taskset -c 0-3 python -c "import jax, os; jax.numpy.ones(3).block_until_ready(); print('taskset 0-3:', len(os.listdir('/proc/self/task')))"

cat > /tmp/hang_probe.py <<'EOF'
import jax, jax.numpy as jnp
import bayes_scatter
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.phantoms import cylinder

setup = ch15_setup(n=24, side=2.0)
fwd = Forward(setup, tol=1e-10)
chi = jnp.asarray(cylinder(setup.grid, 0.5, 1.5))
f = jax.jit(jax.value_and_grad(lambda c: jnp.sum(jnp.abs(fwd(c)[0]) ** 2)))
for _ in range(50):
    jax.block_until_ready(f(chi))
print("finished")
EOF
for t in 2 3 4; do echo -n "NPROC=$t: "; NPROC=$t JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=true \
  timeout 120 python /tmp/hang_probe.py || echo "no result in 120 s (exit $?)"; done
```

The thread count should grow with `NPROC` (by 2 per unit on macOS; the constant part may differ on Linux). The
`taskset` line is to verify: I expect XLA's default pool size to follow the CPU affinity mask, in which case
`taskset` alone limits the pool, but this is not confirmed. The jobs set both, so either way they stay within
their cores. On the Mac the probe finished for `NPROC` = 3 and 4 in about 5 s and hung for `NPROC=2`; if Linux
hangs for other values, raise `MIN_XLA_POOL` in `scripts/batch_run.py` (it is the only place the floor is set).

## (g) Syncing results back to the Mac

From the Mac, in the repository root. `-n` first to see what would change; `-u` never overwrites a newer local
file. The large `.npz` chain files are excluded by default.

```bash
rsync -avzun --exclude='*.npz' tr:work/photonics-dl/results/ results/        # preview
rsync -avzu  --exclude='*.npz' tr:work/photonics-dl/results/ results/

# the chains of one run, when wanted
rsync -avz tr:work/photonics-dl/results/mix_n48_book_mala_gpu/chains.npz results/mix_n48_book_mala_gpu/
```

Benchmark JSONs arrive in `results/bench/`, job logs and metadata in `results/logs/`. Do not rsync in the other
direction (Mac to workstation) for `results/`; the workstation's copy is the record of what ran there.

## (h) Caveats

* **FP64 on the 3090.** The card's FP64 rate is 1/64 of FP32 (roughly 0.56 against 35.6 TFLOP/s by the
  published specifications), whereas the 3990X has several TFLOP/s of FP64 peak in total. The operator is
  FFT- and vector-bound, so the practical gap should be set by memory traffic (a factor near 2 between complex64
  and complex128) rather than by the 64, but that is an expectation, not a result: the `fft` mode and the
  `grad`/`batched` modes measure it. If the GPU turns out compute-bound in FP64, the remedy would be a mixed
  precision matvec, which this code does not have.
* **Memory bandwidth per core on the 3990X.** Four DDR4 channels (about 100 GB/s peak at DDR4-3200) shared by 64
  cores is about 1.6 GB/s per core when all are busy. One stacked Krylov vector (40 transmitters) is 2.6 MB at
  n = 64 and fits an L3, but the 101-vector basis that every Gram-Schmidt sweep touches is 265 MB, so jobs at
  n >= 64 stream from DRAM and will scale sub-linearly with the number of concurrent jobs. Use the scaling
  experiment in (e) before choosing between 64 single-thread slots and 16 four-thread slots, and expect the
  best slot count to be below 64 for the larger grids.
* **Oversubscription.** More running threads than cores (jobs times threads above 64, or jobs started without
  `NPROC`) slows every job and corrupts timings. `batch_run.py` warns when unpinned slots times threads exceed the
  logical CPUs, refuses pinned plans that do not fit, and sets the thread variables for you; a hand-started job
  does not get them. Keep other work off the machine while benchmarking, and read the load average in the
  benchmark output.
* **Multi-thread jobs.** Prefer many single-thread jobs: they are the controlled, tested configuration, and the
  jobs here are memory-bound rather than compute-bound. A job with `T > 1` runs about `T` threads, not exactly
  `T`, unless pinned, and a pool of exactly two XLA workers can hang (see "A hang to know about"). If a sweep
  stalls with jobs at 0% CPU in `top`, that is the symptom: stop the runner (Ctrl-C), and rerun with
  `--job-timeout` and `--threads 1`.
* **Reproducibility.** CPU and GPU results agree only to solver tolerance and floating-point round-off, so a
  chain started from the same seed on both will diverge after a few steps; compare posteriors statistically
  (means, standard deviations, effective sample sizes), not trajectories.
* **Unverified here:** the Linux sysfs reading on the real machine (checked only against a fake tree; see
  `--show-topology`), `taskset` and the pool size under affinity, whether the `NPROC=2` hang and the `NPROC`
  rules above hold on the Linux build of jaxlib, everything on the GPU (including a GPU job pinned to one
  core, which the smoke job in (f) tests), the exact `ssh-copy-id`/`sshd` steps for your distribution in (a),
  and the git steps in (b) against the real workstation (they were run only against a local directory standing
  in for it).
