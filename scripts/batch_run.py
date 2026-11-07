"""Run many independent jobs from a JSON file on a CPU+GPU workstation: slot-limited, optionally pinned.

The jobs file is a JSON list of objects:

    {"name": "a_mala", "cmd": ["scripts/run_posterior.py", "--n", "16", "--out", "a_mala"],
     "device": "cpu", "threads": 1, "done_marker": "results/a_mala/summary.json"}

* ``name`` (required): letters, digits, ``_ . -``; names the log files, so it must be unique.
* ``cmd`` (required): argument list, every element a string, run with the repository root as the working
  directory. If ``cmd[0]`` ends in ``.py`` it is run with the Python interpreter that runs this script
  (override with ``--python``); ``python`` / ``python3`` as ``cmd[0]`` means the same interpreter.
  Anything else is executed as given.
* ``device``: ``"cpu"`` (default) or ``"gpu"``.
* ``threads`` (CPU jobs): threads for this job, default ``--threads``.
* ``done_marker``: path (relative to the repository root) of a file whose existence means the job is
  finished. Default ``results/<name>/summary.json``, which is what ``scripts/run_posterior.py --out <name>``
  writes, so keep ``--out`` equal to ``name``. A job whose marker exists is skipped unless ``--force`` or unless
  its last recorded run (``<log-dir>/<name>.meta.json``) ended as failed, interrupted or still "running"
  (run_posterior writes summary.json before chains.npz and the figures, so the marker alone can be premature).

Resources. ``--cpu-slots K`` jobs run at once on the CPU, each with ``--threads T`` threads (a job may
override T); ``--gpu-slots G`` GPU jobs run at once. CPU jobs get ``JAX_PLATFORMS=cpu`` and the settings
that make XLA's CPU backend use T threads (``cpu_job_env``: the XLA thread pool is sized by the ``NPROC``
environment variable in JAX 0.11, ``XLA_FLAGS`` alone cannot do it; a pool of two hangs, so 2 or 3 threads get a
pool of four; see docs/WORKSTATION.md). GPU jobs get
``JAX_PLATFORMS=cuda,cpu``, ``CUDA_VISIBLE_DEVICES`` (``--gpu-devices``), ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
and ``XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9 / (jobs sharing that device)``.

Pinning (``--pin``, Linux only). The logical CPUs are grouped by the L3 cache they share (sysfs
``cache/index*/shared_cpu_list`` of the index whose ``level`` is 3; one group per core complex on Zen 2),
and cores by ``thread_siblings_list``. A CPU job gets T logical CPUs from one L3 group, one logical CPU per
physical core, and is started under ``taskset -c``. Concurrent jobs never share a CPU (nor, unless ``--smt``,
a physical core), and a job never spans two L3 groups, so T cannot exceed the cores of the largest group
(4 on a Threadripper 3990X). Free groups are used round-robin ("spread", more L3 per job when the machine is
not full; ``--pack`` fills one group first). Each GPU slot reserves ``--gpu-threads`` cores for its host
thread so CPU jobs cannot take them. Without ``--pin`` (or off Linux) jobs run unpinned.

Per job it writes ``<log-dir>/<name>.log`` (stdout and stderr) and ``<log-dir>/<name>.meta.json`` (host,
command, environment overrides, cores, start and end time, wall seconds, exit code). Ctrl-C (or SIGTERM)
terminates the children's process groups and exits 130. The exit status is 1 if any job failed. With
``--job-timeout S`` a job that is still running after S seconds of wall time is terminated (SIGTERM, then
SIGKILL after ``--grace``) and recorded as "timeout"; this bounds a hung job, which would otherwise hold its
slot forever.

Examples
  python scripts/batch_run.py scripts/jobs/example_cpu_sweep.json --cpu-slots 3 --dry-run
  python scripts/batch_run.py jobs.json --cpu-slots 63 --gpu-slots 1 --pin
  python scripts/batch_run.py jobs.json --cpu-slots 15 --threads 4 --gpu-slots 1 --pin
  python scripts/batch_run.py --show-topology
"""

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSFS = Path("/sys/devices/system/cpu")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
JOB_KEYS = {"name", "cmd", "device", "threads", "done_marker"}
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


# ----------------------------------------------------------------------------------------------
# Environment recipes (shared with scripts/bench_backend.py)
# ----------------------------------------------------------------------------------------------


MIN_XLA_POOL = 4  # smallest XLA thread pool of a multi-threaded CPU job, see ``xla_pool_size``


def xla_cpu_flags(threads):
    """``XLA_FLAGS`` entries for a CPU job of ``threads`` threads.

    ``--xla_cpu_multi_thread_eigen`` only decides whether Eigen's matrix products and the FFT thunk may use
    XLA's thread pool; the other XLA CPU kernels (fusions) run on the pool whatever it says, so the flag alone
    does not give a single-threaded process (measured on JAX 0.11.2: CPU time over wall time 1.41 against 2.24
    by default and 0.75 with ``NPROC=1``). The pool size comes from ``NPROC`` (``thread_env``). Note that
    ``intra_op_parallelism_threads`` is not an XLA flag in this jaxlib: spelled ``--intra_op_...`` it makes
    XLA abort with "Unknown flag", and without the dashes the parser silently ignores it.
    """
    return [f"--xla_cpu_multi_thread_eigen={'false' if threads == 1 else 'true'}"]


def xla_pool_size(threads):
    """``NPROC`` for a CPU job of ``threads`` threads: exactly 1 for one thread, else at least ``MIN_XLA_POOL``.

    With Eigen multi-threading on, XLA's FFT thunk blocks its pool worker until helper tasks it queued on the
    same pool have run; if every worker is blocked in an FFT, nothing runs and the process hangs at 0% CPU
    (stack: all workers in ``ducc0::c2c``). Measured on JAX 0.11.2 with a bare forward-and-gradient loop with
    respect to chi (n = 24): ``NPROC=2`` hung in 7 of 7 runs; ``NPROC`` = 3, 4, 8 in none of 11; no run with
    Eigen off (``NPROC`` 1, 2, 4) hung. The Target-based modes of bench_backend (n up to 48) and a reduced
    run_posterior at n = 24 did not hang at ``NPROC`` = 2 or 3, so whether a graph can hang depends on how many
    FFTs it runs at once. A pool of four leaves a free worker for such graphs, which costs a pinned job nothing:
    ``--pin`` (taskset) is what keeps it on its ``threads`` cores. Unpinned, a job of 2 or 3 threads may use up
    to ``MIN_XLA_POOL`` of them. One thread means Eigen off, where nothing ever waits on the pool.
    """
    return 1 if threads == 1 else max(threads, MIN_XLA_POOL)


def merge_xla_flags(existing, extra):
    """``existing`` (an ``XLA_FLAGS`` string or None) with the flags in ``extra`` set, replacing duplicates."""
    keys = {flag.split("=", 1)[0] for flag in extra}
    kept = [tok for tok in (existing or "").split() if tok.split("=", 1)[0] not in keys]
    return " ".join(kept + list(extra))


def thread_env(threads, base_xla_flags=None, cpu_flags=True):
    """Thread-count variables for ``threads`` threads: the BLAS/OpenMP variables (read by numpy and scipy),
    and for a CPU-backend job (``cpu_flags``) also ``NPROC`` (``xla_pool_size``: sizes the XLA CPU client's
    thread pools, read by jaxlib when the client is created) and ``XLA_FLAGS``. A GPU job gets neither: its
    process has no XLA CPU client, and a one-worker pool for the CUDA client is an assumption not worth making."""
    env = {var: str(threads) for var in THREAD_ENV_VARS}
    if cpu_flags:
        env = {"NPROC": str(xla_pool_size(threads)), **env,
               "XLA_FLAGS": merge_xla_flags(base_xla_flags, xla_cpu_flags(threads))}
    return env


def cpu_job_env(threads, base_xla_flags=None):
    """Environment overrides of a CPU job: JAX on the CPU backend with ``threads`` threads."""
    return {"JAX_PLATFORMS": "cpu", **thread_env(threads, base_xla_flags), "PYTHONUNBUFFERED": "1"}


def gpu_job_env(device, n_sharing=1, host_threads=1, platform_name="cuda,cpu"):
    """Environment overrides of a GPU job on CUDA device ``device`` shared by ``n_sharing`` concurrent jobs.

    ``JAX_PLATFORMS=cuda,cpu`` makes JAX fail loudly when it cannot see the GPU instead of silently falling
    back to the CPU; the CPU device is listed too because ``jax.debug.callback`` (the samplers' evaluation
    counters) needs one, and with ``cuda`` alone such a program fails at run time. Memory is allocated on demand and capped at 0.9 / n_sharing of the device, rounded down.
    """
    fraction = math.floor(0.9 / max(1, n_sharing) * 1000 + 1e-9) / 1000
    return {
        "JAX_PLATFORMS": platform_name,
        "CUDA_VISIBLE_DEVICES": str(device),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{fraction:g}",
        **thread_env(host_threads, cpu_flags=False),
        "PYTHONUNBUFFERED": "1",
    }


# ----------------------------------------------------------------------------------------------
# Host information (shared with scripts/bench_backend.py)
# ----------------------------------------------------------------------------------------------


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def cpu_model():
    """CPU model string: /proc/cpuinfo or lscpu on Linux, sysctl on macOS, else ``platform.processor()``."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        text = _run(["lscpu"])
        for line in (text or "").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    elif system == "Darwin":
        text = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if text:
            return text
    return platform.processor() or None


def physical_cpu_count():
    """Physical cores (lscpu on Linux, sysctl on macOS); None if unknown."""
    system = platform.system()
    if system == "Darwin":
        text = _run(["sysctl", "-n", "hw.physicalcpu"])
        return int(text) if text and text.isdigit() else None
    if system == "Linux":
        text = _run(["lscpu", "-p=CORE,SOCKET"])
        if text:
            pairs = {tuple(line.split(",")[:2]) for line in text.splitlines() if line and not line.startswith("#")}
            return len(pairs) or None
    return None


def mem_total_bytes():
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    elif system == "Darwin":
        text = _run(["sysctl", "-n", "hw.memsize"])
        return int(text) if text and text.isdigit() else None
    return None


def host_info():
    """Hostname, OS, CPU model, core counts and memory of this machine (JSON-serializable)."""
    return {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": physical_cpu_count(),
        "mem_total_bytes": mem_total_bytes(),
    }


# ----------------------------------------------------------------------------------------------
# CPU topology and the core pool
# ----------------------------------------------------------------------------------------------


def parse_cpu_list(text):
    """Sorted logical CPU numbers of a kernel list such as ``"0-3,64-67"``."""
    cpus = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def format_cpu_list(cpus):
    """Inverse of ``parse_cpu_list``: ``[0, 1, 2, 3, 64]`` -> ``"0-3,64"`` (the form ``taskset -c`` takes)."""
    cpus = sorted(set(cpus))
    parts, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        parts.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(parts)


@dataclass(frozen=True)
class L3Group:
    """Physical cores that share one L3 cache. ``cores`` holds, per physical core, its logical CPUs
    (hardware threads) in ascending order."""

    key: str
    cores: tuple

    @property
    def n_cores(self):
        return len(self.cores)

    def cpus(self):
        return sorted(c for core in self.cores for c in core)

    def cpu_order(self, smt=False):
        """CPUs in the order a pool hands them out: the first hardware thread of every core, then (``smt``)
        the remaining threads."""
        order = [core[0] for core in self.cores]
        if smt:
            order += [c for core in self.cores for c in core[1:]]
        return order


@dataclass(frozen=True)
class Topology:
    groups: tuple
    notes: tuple = ()

    @property
    def n_cores(self):
        return sum(g.n_cores for g in self.groups)

    @property
    def n_cpus(self):
        return sum(len(g.cpus()) for g in self.groups)

    def group_index(self, cpu):
        for i, g in enumerate(self.groups):
            if cpu in g.cpus():
                return i
        raise KeyError(cpu)

    def describe(self):
        lines = [f"{len(self.groups)} L3 group(s), {self.n_cores} physical core(s), {self.n_cpus} logical CPU(s)"]
        for i, g in enumerate(self.groups):
            cores = " ".join("+".join(map(str, core)) for core in g.cores)
            lines.append(f"  group {i:2d} (L3 {g.key}): {cores}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _l3_shared_cpus(cpu_dir):
    """CPUs sharing this CPU's L3, from the cache index whose ``level`` file says 3 (None if absent)."""
    cache = cpu_dir / "cache"
    if not cache.is_dir():
        return None
    for idx in sorted(cache.glob("index*")):
        if _read(idx / "level") != "3":
            continue
        kind = _read(idx / "type")
        if kind is not None and kind.lower() == "instruction":
            continue
        shared = _read(idx / "shared_cpu_list")
        if shared:
            return parse_cpu_list(shared)
    return None


def read_topology(sysfs_root=DEFAULT_SYSFS, allowed=None):
    """Read the L3 groups and physical cores from a sysfs CPU tree (``/sys/devices/system/cpu`` layout).

    ``allowed``: optional set of logical CPUs to keep (for example ``os.sched_getaffinity(0)``). Offline CPUs
    (``cpuN/online`` containing 0) are dropped. CPUs without L3 information are put in one extra group.
    Groups are ordered by their lowest CPU; within a group, cores likewise.
    """
    root = Path(sysfs_root)
    cpu_dirs = {}
    if root.is_dir():
        for p in root.iterdir():
            m = re.fullmatch(r"cpu(\d+)", p.name)
            if m and p.is_dir():
                cpu_dirs[int(m.group(1))] = p
    online = {c for c, d in cpu_dirs.items() if _read(d / "online") != "0"}
    if allowed is not None:
        online &= set(allowed)

    l3_of_core = {}  # core (tuple of its logical CPUs) -> tuple of the CPUs sharing its L3, or None
    notes = []
    for cpu in sorted(online):
        d = cpu_dirs[cpu]
        sib = _read(d / "topology" / "thread_siblings_list")
        siblings = [c for c in parse_cpu_list(sib) if c in online] if sib else [cpu]
        core = tuple(sorted(set(siblings) | {cpu}))
        if core in l3_of_core:
            continue
        shared = _l3_shared_cpus(d)
        l3_of_core[core] = tuple(c for c in shared if c in online) if shared else None

    by_l3 = {}
    for core, key in l3_of_core.items():
        by_l3.setdefault(key, []).append(core)
    if None in by_l3:
        notes.append("no L3 information in sysfs for some CPUs: they form one group")
    groups = []
    for key, members in by_l3.items():
        label = "unknown" if key is None else format_cpu_list(key)
        groups.append(L3Group(label, tuple(sorted(members))))
    groups.sort(key=lambda g: g.cores[0][0])
    return Topology(tuple(groups), tuple(notes))


class CorePool:
    """Hands out sets of logical CPUs, each inside one L3 group, never the same CPU to two holders.

    ``acquire(n)`` returns n CPUs from one group (a sorted tuple) or None if no group has n free. It prefers
    a group whose next n free CPUs are all first hardware threads (one CPU per physical core), then the group
    with the most free CPUs ("spread"; ``pack=True`` takes the fewest that still fit), then the lowest group
    index (the highest with ``reverse=True``, used for the GPU hosts' reservation). With ``smt`` the sibling
    hardware threads become available after the first threads of a group are used up.
    """

    def __init__(self, topology, smt=False):
        self.topology = topology
        self._order = [g.cpu_order(smt) for g in topology.groups]
        self._rank = [{c: i for i, c in enumerate(order)} for order in self._order]
        self._primary = {core[0] for g in topology.groups for core in g.cores}
        self._free = [list(order) for order in self._order]

    def capacity(self):
        """Largest number of CPUs one holder can ever get (the biggest group)."""
        return max((len(o) for o in self._order), default=0)

    def free_count(self):
        return sum(len(f) for f in self._free)

    def acquire(self, n, *, pack=False, reverse=False):
        best, best_key = None, None
        for gi, free in enumerate(self._free):
            if len(free) < n or n < 1:
                continue
            clean = all(c in self._primary for c in free[:n])
            fill = -len(free) if pack else len(free)
            key = (clean, fill, gi if reverse else -gi)
            if best_key is None or key > best_key:
                best, best_key = gi, key
        if best is None:
            return None
        free = self._free[best]
        taken, self._free[best] = free[:n], free[n:]
        return tuple(sorted(taken))

    def release(self, cpus):
        for gi, rank in enumerate(self._rank):
            mine = [c for c in cpus if c in rank]
            if mine:
                self._free[gi] = sorted(self._free[gi] + mine, key=rank.__getitem__)


def plan_slots(topology, n_slots, threads, *, reserved=(), smt=False, pack=False):
    """CPU sets for ``n_slots`` simultaneous slots of ``threads`` CPUs (after reserving ``reserved``, a list
    of CPU counts, for GPU hosts). Raises ``ValueError`` if they do not fit."""
    pool = CorePool(topology, smt=smt)
    for r in reserved:
        if pool.acquire(r, reverse=True) is None:
            raise ValueError(f"cannot reserve {r} CPU(s) for a GPU host thread")
    if threads > pool.capacity():
        raise ValueError(f"threads={threads} exceeds the largest L3 group ({pool.capacity()} CPUs); a pinned "
                         "slot never spans two L3 groups")
    slots = []
    for i in range(n_slots):
        cpus = pool.acquire(threads, pack=pack)
        if cpus is None:
            raise ValueError(f"only {i} of {n_slots} slots of {threads} thread(s) fit on this host "
                             f"({topology.n_cores} physical cores in {len(topology.groups)} L3 groups"
                             f"{', ' + str(len(reserved)) + ' GPU host reservation(s)' if reserved else ''})")
        slots.append(cpus)
    return slots


# ----------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------


class JobFileError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    name: str
    cmd: tuple
    device: str = "cpu"
    threads: int = None
    done_marker: str = None

    @property
    def marker(self):
        return self.done_marker or f"results/{self.name}/summary.json"


def parse_jobs(data, source="jobs"):
    """Validate a decoded jobs file (a list of objects) and return a list of ``Job``."""
    if not isinstance(data, list):
        raise JobFileError(f"{source}: expected a JSON list of jobs")
    jobs, seen = [], set()
    for i, item in enumerate(data):
        where = f"{source}[{i}]"
        if not isinstance(item, dict):
            raise JobFileError(f"{where}: each job must be an object")
        unknown = set(item) - JOB_KEYS
        if unknown:
            raise JobFileError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(JOB_KEYS)}")
        name = item.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise JobFileError(f"{where}: 'name' must be a string of letters, digits, '_', '.', '-'")
        if name in seen:
            raise JobFileError(f"{where}: duplicate job name {name!r}")
        seen.add(name)
        cmd = item.get("cmd")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(a, str) for a in cmd):
            raise JobFileError(f"{where} ({name}): 'cmd' must be a non-empty list of strings (quote numbers)")
        device = item.get("device", "cpu")
        if device not in ("cpu", "gpu"):
            raise JobFileError(f"{where} ({name}): 'device' must be \"cpu\" or \"gpu\"")
        threads = item.get("threads")
        if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
            raise JobFileError(f"{where} ({name}): 'threads' must be a positive integer")
        marker = item.get("done_marker")
        if marker is not None and (not isinstance(marker, str) or not marker):
            raise JobFileError(f"{where} ({name}): 'done_marker' must be a non-empty string")
        jobs.append(Job(name, tuple(cmd), device, threads, marker))
    return jobs


def load_jobs(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise JobFileError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise JobFileError(f"{path}: invalid JSON: {exc}") from exc
    return parse_jobs(data, source=path.name)


def marker_path(job, root):
    p = Path(job.marker)
    return p if p.is_absolute() else Path(root) / p


def recorded_status(log_dir, name):
    """``status`` in ``<log_dir>/<name>.meta.json`` (written by an earlier batch_run), or None if there is none."""
    try:
        return json.loads((Path(log_dir) / f"{name}.meta.json").read_text()).get("status")
    except (OSError, ValueError, AttributeError):
        return None


def stale_marker_status(job, root, log_dir):
    """If the done marker exists but the last batch_run record of the job says it did not end well
    ("failed", "interrupted", or "running" for a runner that was killed), return that status, else None.

    scripts/run_posterior.py writes summary.json before chains.npz and the figures, so a job killed or crashed
    after that leaves the marker behind with incomplete outputs; such a job must run again."""
    if not marker_path(job, root).exists():
        return None
    status = recorded_status(log_dir, job.name)
    return status if status not in (None, "ok") else None


def job_is_done(job, root, log_dir=None):
    """True if the marker exists and (when ``log_dir`` is given) the last recorded run did not end badly."""
    if not marker_path(job, root).exists():
        return False
    return log_dir is None or stale_marker_status(job, root, log_dir) is None


@dataclass
class Options:
    root: Path
    log_dir: Path
    python: str
    cpu_slots: int = 1
    threads: int = 1
    gpu_slots: int = 1
    gpu_devices: tuple = ("0",)
    gpu_threads: int = 1
    gpu_platform: str = "cuda,cpu"
    pinned: bool = False
    taskset: str = None
    smt: bool = False
    pack: bool = False
    poll: float = 0.5
    grace: float = 10.0
    job_timeout: float = 0.0  # seconds of wall time per job; 0 = no limit

    def threads_of(self, job):
        return job.threads or self.threads


def gpu_slot_env(opts, slot):
    """Environment overrides of GPU slot number ``slot``: slots are spread over ``gpu_devices`` in turn."""
    devices = opts.gpu_devices
    device = devices[slot % len(devices)]
    sharing = sum(1 for s in range(opts.gpu_slots) if devices[s % len(devices)] == device)
    return gpu_job_env(device, sharing, opts.gpu_threads, opts.gpu_platform)


def job_env(opts, job, slot):
    """Environment overrides for ``job`` running in slot number ``slot`` (a dict of strings)."""
    if job.device == "gpu":
        return gpu_slot_env(opts, slot)
    return cpu_job_env(opts.threads_of(job), os.environ.get("XLA_FLAGS"))


def build_command(opts, job, cpus=None):
    """The argument vector to launch: the interpreter is filled in for ``.py`` scripts, and ``taskset -c``
    prefixed when the job is pinned to ``cpus``."""
    argv = list(job.cmd)
    if argv[0] in ("python", "python3"):
        argv[0] = opts.python
    elif argv[0].endswith(".py"):
        argv = [opts.python] + argv
    if cpus and opts.pinned:
        argv = [opts.taskset or "taskset", "-c", format_cpu_list(cpus)] + argv
    return argv


# ----------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _say(msg):
    print(f"[batch_run {datetime.now():%H:%M:%S}] {msg}", flush=True)


def _write_meta(path, meta):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, path)


@dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    log: object
    meta: dict
    meta_path: Path
    cpus: tuple
    slot: int
    t0: float
    timed_out: bool = False
    kill_at: float = 0.0


class Runner:
    """Starts jobs as slots free up, FIFO within the CPU queue and within the GPU queue."""

    def __init__(self, opts, jobs, pool=None, gpu_reserved=None):
        self.opts = opts
        self.queues = {"gpu": [j for j in jobs if j.device == "gpu"], "cpu": [j for j in jobs if j.device == "cpu"]}
        self.total = len(jobs)
        self.pool = pool
        self.gpu_reserved = gpu_reserved or {}
        self.free_slots = {"cpu": list(range(opts.cpu_slots)), "gpu": list(range(opts.gpu_slots))}
        self.running = []
        self.results = []  # (job, status, wall, exit_code)
        self.started = 0
        self.host = host_info()

    # -- starting ------------------------------------------------------------------------------

    def _try_start(self, kind):
        queue = self.queues[kind]
        if not queue or not self.free_slots[kind]:
            return False
        job = queue[0]
        cpus = None
        if self.opts.pinned:
            if kind == "cpu":
                cpus = self.pool.acquire(self.opts.threads_of(job), pack=self.opts.pack)
                if cpus is None:
                    return False
            else:
                cpus = self.gpu_reserved.get(self.free_slots[kind][0])
        queue.pop(0)
        slot = self.free_slots[kind].pop(0)
        self._launch(job, slot, cpus)
        return True

    def _launch(self, job, slot, cpus):
        opts = self.opts
        opts.log_dir.mkdir(parents=True, exist_ok=True)
        overrides = job_env(opts, job, slot)
        env = {**os.environ, **overrides}
        argv = build_command(opts, job, cpus)
        log_path = opts.log_dir / f"{job.name}.log"
        meta_path = opts.log_dir / f"{job.name}.meta.json"
        log = open(log_path, "wb")
        log.write(f"# batch_run {_now()} slot {job.device}{slot}\n# env {json.dumps(overrides)}\n"
                  f"# cmd {' '.join(argv)}\n".encode())
        log.flush()
        meta = {
            "name": job.name, "status": "running", "device": job.device, "slot": slot,
            "threads": opts.threads_of(job) if job.device == "cpu" else opts.gpu_threads,
            "cmd": argv, "cmd_in_jobs_file": list(job.cmd), "cwd": str(opts.root), "env": overrides,
            "pinned": bool(cpus and opts.pinned), "cores": format_cpu_list(cpus) if cpus else None,
            "host": self.host, "log": str(log_path), "done_marker": str(marker_path(job, opts.root)),
            "start": _now(), "end": None, "wall_s": None, "exit_code": None,
        }
        _write_meta(meta_path, meta)
        try:
            proc = subprocess.Popen(argv, cwd=opts.root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            log.write(f"# could not start: {exc}\n".encode())
            log.close()
            meta.update(status="failed", end=_now(), wall_s=0.0, exit_code=127, error=str(exc))
            _write_meta(meta_path, meta)
            self._release(job, slot, cpus)
            self.results.append((job, "failed", 0.0, 127))
            _say(f"FAIL  {job.name}: could not start {argv[0]!r}: {exc}")
            return
        self.started += 1
        self.running.append(Running(job, proc, log, meta, meta_path, cpus, slot, time.monotonic()))
        where = f"cores {format_cpu_list(cpus)}" if cpus else "unpinned"
        _say(f"START {job.name} ({job.device}{slot}, {where}) [{self.started}/{self.total}]")

    def _release(self, job, slot, cpus):
        self.free_slots[job.device].append(slot)
        self.free_slots[job.device].sort()
        if self.opts.pinned and job.device == "cpu" and cpus:
            self.pool.release(cpus)

    # -- finishing -----------------------------------------------------------------------------

    def _finish(self, r, status=None):
        code = r.proc.returncode
        wall = time.monotonic() - r.t0
        r.log.close()
        marker = marker_path(r.job, self.opts.root).exists()
        if status is None:
            status = "ok" if code == 0 else "failed"
        r.meta.update(status=status, end=_now(), wall_s=round(wall, 3), exit_code=code, marker_present=marker)
        if code is not None and code < 0:
            try:
                r.meta["signal"] = signal.Signals(-code).name
            except ValueError:
                r.meta["signal"] = str(-code)
        _write_meta(r.meta_path, r.meta)
        self._release(r.job, r.slot, r.cpus)
        self.results.append((r.job, status, wall, code))
        extra = ""
        if status == "ok" and not marker:
            extra = f" (WARNING: exited 0 but {r.meta['done_marker']} does not exist; it will run again next time)"
        _say(f"{'DONE ' if status == 'ok' else status.upper():5s} {r.job.name} exit {code} in {wall:.1f} s{extra}")

    def _reap(self):
        still = []
        now = time.monotonic()
        for r in self.running:
            if r.proc.poll() is not None:
                self._finish(r, status="timeout" if r.timed_out else None)
                continue
            limit = self.opts.job_timeout
            if limit and not r.timed_out and now - r.t0 > limit:
                r.timed_out, r.kill_at = True, now + self.opts.grace
                _say(f"TIMEOUT {r.job.name} after {now - r.t0:.0f} s (limit {limit:g} s): terminating")
                self._signal(r, signal.SIGTERM)
            elif r.timed_out and now > r.kill_at:
                self._signal(r, signal.SIGKILL)
            still.append(r)
        self.running = still

    def terminate_all(self):
        """SIGTERM every child's process group, wait ``grace`` seconds, then SIGKILL what is left."""
        for r in self.running:
            self._signal(r, signal.SIGTERM)
        deadline = time.monotonic() + self.opts.grace
        try:
            while time.monotonic() < deadline and any(r.proc.poll() is None for r in self.running):
                time.sleep(0.05)
        except KeyboardInterrupt:  # a second Ctrl-C: stop waiting politely
            _say("second interrupt: killing the remaining job(s) now")
        for r in self.running:
            if r.proc.poll() is None:
                self._signal(r, signal.SIGKILL)
            r.proc.wait()
            self._finish(r, status="interrupted")
        self.running = []

    @staticmethod
    def _signal(r, sig):
        try:
            os.killpg(r.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    # -- main loop -----------------------------------------------------------------------------

    def run(self):
        try:
            while self.queues["cpu"] or self.queues["gpu"] or self.running:
                self._reap()
                progressed = True
                while progressed:
                    progressed = self._try_start("gpu") | self._try_start("cpu")
                if not self.running and (self.queues["cpu"] or self.queues["gpu"]):
                    raise RuntimeError("no job can start and none is running (resource plan is inconsistent)")
                time.sleep(self.opts.poll)
        except KeyboardInterrupt:
            _say(f"interrupted: terminating {len(self.running)} running job(s)")
            self.terminate_all()
            return 130
        except BaseException:  # for example a full disk while writing a log: do not leave children running
            self.terminate_all()
            raise
        return 1 if any(status != "ok" for _, status, _, _ in self.results) else 0


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("jobs_file", nargs="?", help="JSON list of jobs (see the module docstring)")
    ap.add_argument("--cpu-slots", type=int, default=1, help="CPU jobs running at once")
    ap.add_argument("--threads", type=int, default=1, help="threads per CPU job (a job may override it)")
    ap.add_argument("--gpu-slots", type=int, default=1, help="GPU jobs running at once")
    ap.add_argument("--gpu-devices", default="0", help="comma-separated CUDA device ids; slot i uses entry i mod len")
    ap.add_argument("--gpu-threads", type=int, default=1, help="CPU threads (reserved when pinned) per GPU job")
    ap.add_argument("--gpu-platform", default="cuda,cpu", help="JAX_PLATFORMS value of GPU jobs")
    ap.add_argument("--pin", action="store_true", help="Linux only: pin each job to cores of one L3 group (taskset)")
    ap.add_argument("--smt", action="store_true", help="with --pin: allow sibling hardware threads once every "
                    "core's first thread is in use (two jobs then share a physical core)")
    ap.add_argument("--pack", action="store_true", help="with --pin: fill one L3 group before the next "
                    "(default: spread jobs over the groups)")
    ap.add_argument("--sysfs-root", default=str(DEFAULT_SYSFS), help="CPU sysfs tree used for --pin")
    ap.add_argument("--dry-run", action="store_true", help="print what would run and exit")
    ap.add_argument("--force", action="store_true", help="run jobs even if their done marker exists")
    ap.add_argument("--log-dir", default="results/logs", help="per-job .log and .meta.json (relative to --root)")
    ap.add_argument("--root", default=str(ROOT), help="repository root: working directory of the jobs")
    ap.add_argument("--python", default=sys.executable, help="interpreter for .py commands")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between checks of the running jobs")
    ap.add_argument("--grace", type=float, default=10.0, help="seconds between SIGTERM and SIGKILL on Ctrl-C "
                    "or timeout")
    ap.add_argument("--job-timeout", type=float, default=0.0, help="terminate a job that runs longer than this many "
                    "seconds of wall time (status timeout, exit status 1); 0 = no limit. A hung job otherwise "
                    "holds its slot forever")
    ap.add_argument("--show-topology", action="store_true", help="print the L3 groups and cores found and exit")
    return ap.parse_args(argv)


def _fail(msg):
    print(f"batch_run: error: {msg}", file=sys.stderr)
    return 2


def _dry_run_report(opts, runnable, skipped, pool, gpu_reserved):
    """Print what a real run would start first: one wave of jobs, one slot each; the rest are queued."""
    print(f"[dry-run] {len(runnable)} job(s) to run, {len(skipped)} skipped (done marker exists)")
    for job in skipped:
        print(f"SKIP   {job.name}  (done: {marker_path(job, opts.root)})")
    busy = {"cpu": 0, "gpu": 0}
    for job in runnable:
        kind = job.device
        n_threads = opts.threads_of(job) if kind == "cpu" else opts.gpu_threads
        slot, cpus = None, None
        if busy[kind] < (opts.cpu_slots if kind == "cpu" else opts.gpu_slots):
            if kind == "gpu":
                slot, cpus = busy[kind], gpu_reserved.get(busy[kind])
            elif opts.pinned:
                cpus = pool.acquire(n_threads, pack=opts.pack)
                slot = busy[kind] if cpus else None
            else:
                slot = busy[kind]
        if slot is None:
            print(f"QUEUED {job.name}  device={kind} threads={n_threads}  "
                  f"(starts when a {kind} slot frees; cores are assigned then)")
            continue
        busy[kind] += 1
        where = f"cores={format_cpu_list(cpus)}" if cpus else "unpinned"
        print(f"RUN    {job.name}  device={kind} slot={slot} threads={n_threads} {where}")
        print(f"       env: {' '.join(f'{k}={v}' for k, v in job_env(opts, job, slot).items())}")
        print(f"       cmd: {' '.join(build_command(opts, job, cpus))}")
        print(f"       log: {opts.log_dir / (job.name + '.log')}")


def main(argv=None, *, system=None, which=None):
    """Entry point; returns the exit status. ``system`` and ``which`` replace ``platform.system()`` and
    ``shutil.which`` (tests use them to exercise the Linux pinning path elsewhere)."""
    args = parse_args(argv)
    system = system or platform.system()
    which = which or shutil.which
    root = Path(args.root)
    sysfs = Path(args.sysfs_root)

    topology = None
    if args.show_topology or args.pin:
        if system != "Linux" and not args.show_topology:
            print("batch_run: warning: --pin works only on Linux; running unpinned", file=sys.stderr)
            args.pin = False
        else:
            allowed = None
            if sysfs == DEFAULT_SYSFS and hasattr(os, "sched_getaffinity"):
                allowed = set(os.sched_getaffinity(0))
            topology = read_topology(sysfs, allowed)
            if not topology.groups:
                if args.show_topology:
                    return _fail(f"no CPUs found under {sysfs}")
                print(f"batch_run: warning: no CPU topology under {sysfs}; running unpinned", file=sys.stderr)
                args.pin = False
    if args.show_topology:
        print(topology.describe())
        return 0
    if not args.jobs_file:
        return _fail("a jobs file is required")
    if args.cpu_slots < 0 or args.gpu_slots < 0 or args.threads < 1 or args.gpu_threads < 1 or args.job_timeout < 0:
        return _fail("--cpu-slots/--gpu-slots/--job-timeout must be >= 0 and --threads/--gpu-threads >= 1")
    devices = tuple(d.strip() for d in args.gpu_devices.split(",") if d.strip())
    if not devices:
        return _fail("--gpu-devices is empty")

    try:
        jobs = load_jobs(args.jobs_file)
    except JobFileError as exc:
        return _fail(str(exc))

    log_dir = Path(args.log_dir)
    opts = Options(root=root, log_dir=log_dir if log_dir.is_absolute() else root / log_dir, python=args.python,
                   cpu_slots=args.cpu_slots, threads=args.threads, gpu_slots=args.gpu_slots,
                   gpu_devices=devices, gpu_threads=args.gpu_threads, gpu_platform=args.gpu_platform,
                   pinned=bool(args.pin), smt=args.smt, pack=args.pack, poll=args.poll, grace=args.grace,
                   job_timeout=args.job_timeout)
    if opts.pinned:
        opts.taskset = which("taskset")
        if opts.taskset is None:
            if args.dry_run:
                opts.taskset = "taskset"
            else:
                print("batch_run: warning: taskset not found; running unpinned", file=sys.stderr)
                opts.pinned = False

    runnable = [j for j in jobs if args.force or not job_is_done(j, root, opts.log_dir)]
    skipped = [j for j in jobs if j not in runnable]
    for j in runnable:
        stale = None if args.force else stale_marker_status(j, root, opts.log_dir)
        if stale:
            print(f"batch_run: {j.name}: the done marker exists but the last run was recorded as {stale!r}; "
                  f"running it again", file=sys.stderr)
    for j in runnable:
        if j.device == "cpu" and opts.cpu_slots < 1:
            return _fail(f"job {j.name!r} runs on the CPU but --cpu-slots is 0")
        if j.device == "gpu" and opts.gpu_slots < 1:
            return _fail(f"job {j.name!r} needs a GPU but --gpu-slots is 0")

    pool, gpu_reserved = None, {}
    if opts.pinned:
        cpu_jobs = [j for j in runnable if j.device == "cpu"]
        n_gpu = min(opts.gpu_slots, sum(1 for j in runnable if j.device == "gpu"))
        try:
            reserved = [opts.gpu_threads] * n_gpu
            n_default = min(opts.cpu_slots, sum(1 for j in cpu_jobs if j.threads is None))
            if n_default:
                plan_slots(topology, n_default, opts.threads, reserved=reserved, smt=opts.smt, pack=opts.pack)
            for t in sorted({j.threads for j in cpu_jobs if j.threads}):
                plan_slots(topology, 1, t, reserved=reserved, smt=opts.smt, pack=opts.pack)
        except ValueError as exc:
            return _fail(str(exc))
        pool = CorePool(topology, smt=opts.smt)
        gpu_reserved = {slot: pool.acquire(opts.gpu_threads, reverse=True) for slot in range(n_gpu)}
    elif opts.cpu_slots * opts.threads > (os.cpu_count() or 1) and any(j.device == "cpu" for j in runnable):
        print(f"batch_run: warning: {opts.cpu_slots} slots x {opts.threads} threads exceed the "
              f"{os.cpu_count()} logical CPUs of this machine: jobs will be oversubscribed", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] cpu slots {opts.cpu_slots} x {opts.threads} thread(s), gpu slots {opts.gpu_slots}, "
              f"{'pinned (taskset)' if opts.pinned else 'unpinned'}; jobs {args.jobs_file}")
        _dry_run_report(opts, runnable, skipped, pool, gpu_reserved)
        return 0

    _say(f"{len(runnable)} job(s) to run, {len(skipped)} skipped; cpu slots {opts.cpu_slots} x {opts.threads} "
         f"thread(s), gpu slots {opts.gpu_slots}, {'pinned' if opts.pinned else 'unpinned'}; logs in {opts.log_dir}")
    for j in skipped:
        _say(f"SKIP  {j.name} (done marker exists)")
    if not runnable:
        return 0

    runner = Runner(opts, runnable, pool=pool, gpu_reserved=gpu_reserved)
    previous = signal.getsignal(signal.SIGTERM)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        status = runner.run()
    finally:
        signal.signal(signal.SIGTERM, previous)

    print("\n" + f"{'job':32s} {'status':12s} {'wall s':>9s} {'exit':>5s}")
    for job, st, wall, code in runner.results:
        print(f"{job.name:32s} {st:12s} {wall:9.1f} {str(code):>5s}")
    n_bad = sum(1 for _, st, _, _ in runner.results if st != "ok")
    _say(f"{len(runner.results) - n_bad} ok, {n_bad} not ok" + ("; interrupted" if status == 130 else ""))
    return status


if __name__ == "__main__":
    sys.exit(main())
