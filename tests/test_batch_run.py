"""Tests of scripts/batch_run.py: CPU topology and core slots from a fake sysfs, job files and skip logic,
environments of CPU and GPU jobs, dry-run output, and a few real (trivial) subprocess runs, including a pinned
run through a fake ``taskset``. A few tests at the end cover the import order of scripts/bench_backend.py. No JAX
computation is done (tests/conftest.py imports jax, which is all)."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "batch_run.py"


def _load():
    spec = importlib.util.spec_from_file_location("batch_run", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["batch_run"] = module
    spec.loader.exec_module(module)
    return module


br = _load()
LINUX_WITH_TASKSET = dict(system="Linux", which=lambda name: "/usr/bin/taskset")


# ----------------------------------------------------------------------------------------------
# A fake sysfs tree
# ----------------------------------------------------------------------------------------------


def make_sysfs(root, n_groups=16, cores_per_group=4, smt=2, l3_index=3, offline=(), with_cache=True,
               interleave=False):
    """A /sys/devices/system/cpu look-alike. Default: 3990X-like, 16 L3 groups of 4 cores with 2 threads each.
    Core c has logical CPUs c and c + n_cores (the usual AMD numbering), and group g holds cores
    g*cores_per_group ... (g+1)*cores_per_group - 1. ``l3_index``: the cache index directory that holds the L3
    (the L2 goes to the other of index2 / index3, so a reader must look at the level files). ``interleave``:
    number the hardware threads of a core next to each other (core c has CPUs c*smt ... c*smt + smt - 1, the
    usual Intel numbering) instead of c and c + n_cores."""
    n_cores = n_groups * cores_per_group

    def cpus_of(core):
        return [core * smt + k for k in range(smt)] if interleave else [core + k * n_cores for k in range(smt)]

    for core in range(n_cores):
        group = core // cores_per_group
        threads = cpus_of(core)
        group_cpus = sorted(c for core2 in range(group * cores_per_group, (group + 1) * cores_per_group)
                            for c in cpus_of(core2))
        for cpu in threads:
            d = root / f"cpu{cpu}"
            (d / "topology").mkdir(parents=True)
            (d / "topology" / "thread_siblings_list").write_text(br.format_cpu_list(threads) + "\n")
            if cpu != 0:
                (d / "online").write_text("0\n" if cpu in offline else "1\n")
            if with_cache:
                l2_index = 2 if l3_index == 3 else 3
                caches = {0: (1, "Data", threads), 1: (1, "Instruction", threads),
                          l2_index: (2, "Unified", threads), l3_index: (3, "Unified", group_cpus)}
                for idx, (level, kind, shared) in caches.items():
                    c = d / "cache" / f"index{idx}"
                    c.mkdir(parents=True)
                    (c / "level").write_text(f"{level}\n")
                    (c / "type").write_text(f"{kind}\n")
                    (c / "shared_cpu_list").write_text(br.format_cpu_list(shared) + "\n")
    for noise in ("cpufreq", "cpuidle", "hotplug"):
        (root / noise).mkdir(exist_ok=True)
    (root / "possible").write_text(f"0-{n_cores * smt - 1}\n")
    return root


@pytest.fixture(scope="module")
def sysfs(tmp_path_factory):
    """The default fake tree, built once; tests only read it."""
    return make_sysfs(tmp_path_factory.mktemp("sys") / "cpu")


@pytest.fixture(scope="module")
def topo(sysfs):
    return br.read_topology(sysfs)


def group_of(topology, cpu):
    return topology.group_index(cpu)


# ----------------------------------------------------------------------------------------------
# CPU lists and topology
# ----------------------------------------------------------------------------------------------


def test_cpu_list_parse_and_format_roundtrip():
    assert br.parse_cpu_list("0-3,64-67\n") == [0, 1, 2, 3, 64, 65, 66, 67]
    assert br.parse_cpu_list("5") == [5]
    assert br.format_cpu_list([0, 1, 2, 3, 64, 66, 67]) == "0-3,64,66-67"
    assert br.format_cpu_list([7, 3, 3, 4]) == "3-4,7"
    for text in ("0-3,64-67", "0,64", "12"):
        assert br.format_cpu_list(br.parse_cpu_list(text)) == text


def test_topology_of_a_3990x_like_tree(topo):
    assert len(topo.groups) == 16 and topo.n_cores == 64 and topo.n_cpus == 128
    g0 = topo.groups[0]
    assert g0.key == "0-3,64-67"
    assert g0.cores == ((0, 64), (1, 65), (2, 66), (3, 67))
    assert topo.groups[15].cores[-1] == (63, 127)
    assert all(g.n_cores == 4 for g in topo.groups)
    assert [g.cores[0][0] for g in topo.groups] == [4 * g for g in range(16)]
    assert "16 L3 group(s), 64 physical core(s), 128 logical CPU(s)" in topo.describe()


def test_adjacent_sibling_numbering_gives_the_same_groups(tmp_path):
    """Nothing may assume that the second hardware thread of core c is CPU c + 64."""
    topo = br.read_topology(make_sysfs(tmp_path / "s", interleave=True))
    assert len(topo.groups) == 16 and topo.n_cores == 64 and topo.n_cpus == 128
    assert topo.groups[0].key == "0-7" and topo.groups[0].cores == ((0, 1), (2, 3), (4, 5), (6, 7))
    slots = br.plan_slots(topo, 64, 1)
    check_slots(topo, slots)
    assert sorted(c for s in slots for c in s) == list(range(0, 128, 2))  # first hardware threads only
    four = br.plan_slots(topo, 16, 4)
    check_slots(topo, four)
    assert sorted(four)[1] == (8, 10, 12, 14)


@pytest.mark.parametrize("l3_index", [3, 2])
def test_l3_is_found_by_its_level_file_not_its_index_number(tmp_path, l3_index):
    topo = br.read_topology(make_sysfs(tmp_path / "s", n_groups=4, l3_index=l3_index))
    assert len(topo.groups) == 4 and all(g.n_cores == 4 for g in topo.groups)  # L2 (shared by 2 threads) not used


def test_topology_drops_offline_cpus_and_honors_allowed(tmp_path):
    root = make_sysfs(tmp_path / "s", n_groups=2, cores_per_group=2, smt=2, offline=(5,))  # cpus 0-3, siblings 4-7
    topo = br.read_topology(root)
    assert topo.groups[1].cores == ((2, 6), (3, 7)) and topo.groups[0].cores == ((0, 4), (1,))
    only = br.read_topology(root, allowed={0, 1, 2, 3})
    assert [g.cores for g in only.groups] == [((0,), (1,)), ((2,), (3,))] and only.n_cpus == 4


def test_topology_without_cache_information_is_one_group(tmp_path):
    topo = br.read_topology(make_sysfs(tmp_path / "s", n_groups=2, cores_per_group=2, with_cache=False))
    assert len(topo.groups) == 1 and topo.groups[0].key == "unknown" and topo.n_cores == 4
    assert any("no L3" in note for note in topo.notes)


def test_missing_sysfs_gives_an_empty_topology(tmp_path):
    assert br.read_topology(tmp_path / "nope").groups == ()


# ----------------------------------------------------------------------------------------------
# Core slots
# ----------------------------------------------------------------------------------------------


def check_slots(topology, slots):
    """No CPU twice; each slot inside one L3 group."""
    seen = set()
    for slot in slots:
        assert len({group_of(topology, c) for c in slot}) == 1, slot
        assert not seen & set(slot), slot
        seen |= set(slot)


def test_sixty_four_single_thread_slots_use_one_cpu_per_physical_core(topo):
    slots = br.plan_slots(topo, 64, 1)
    check_slots(topo, slots)
    assert sorted(c for s in slots for c in s) == list(range(64))  # first hardware threads only, no 64+
    assert all(sum(group_of(topo, s[0]) == g for s in slots) == 4 for g in range(16))
    assert len({group_of(topo, s[0]) for s in slots[:16]}) == 16  # spread: the first 16 land in 16 groups
    with pytest.raises(ValueError, match="only 64 of 65"):
        br.plan_slots(topo, 65, 1)


def test_sixteen_four_thread_slots_fill_one_l3_group_each(topo):
    slots = br.plan_slots(topo, 16, 4)
    check_slots(topo, slots)
    assert sorted(slots) == [tuple(range(4 * g, 4 * g + 4)) for g in range(16)]
    with pytest.raises(ValueError, match="only 16 of 17"):
        br.plan_slots(topo, 17, 4)


@pytest.mark.parametrize("threads", [1, 2, 3, 4])
def test_slots_never_overlap_or_span_groups(topo, threads):
    pool = br.CorePool(topo)
    slots = []
    while (s := pool.acquire(threads)) is not None:
        slots.append(s)
    check_slots(topo, slots)
    assert len(slots) == 16 * (4 // threads)  # a 3-thread slot leaves one core idle rather than straddling
    assert all(c < 64 for s in slots for c in s)


def test_threads_beyond_the_largest_l3_group_are_rejected(topo):
    with pytest.raises(ValueError, match="exceeds the largest L3 group"):
        br.plan_slots(topo, 1, 5)


def test_smt_is_opt_in_and_used_only_after_every_first_thread(topo):
    assert br.CorePool(topo).capacity() == 4 and br.CorePool(topo, smt=True).capacity() == 8
    pool = br.CorePool(topo, smt=True)
    slots = [pool.acquire(1) for _ in range(128)]
    assert pool.acquire(1) is None and sorted(c for s in slots for c in s) == list(range(128))
    assert all(s[0] < 64 for s in slots[:64]) and all(s[0] >= 64 for s in slots[64:])
    pairs = br.plan_slots(topo, 64, 2, smt=True)
    check_slots(topo, pairs)
    assert sorted(c for s in pairs for c in s) == list(range(128))
    assert all(c < 64 for s in pairs[:32] for c in s)  # 32 slots of 2 fit on first threads alone


def test_release_makes_cpus_available_again_in_their_group(topo):
    pool = br.CorePool(topo)
    held = [pool.acquire(4) for _ in range(16)]
    assert pool.acquire(1) is None and pool.free_count() == 0
    pool.release(held[5])
    again = pool.acquire(4)
    assert again == held[5] and pool.free_count() == 0
    pool.release(held[0])
    assert pool.acquire(2) == (0, 1) and pool.acquire(2) == (2, 3)


def test_gpu_host_reservation_comes_out_of_the_last_group(topo):
    pool = br.CorePool(topo)
    assert pool.acquire(1, reverse=True) == (60,)
    cpu_slots = []
    while (s := pool.acquire(4)) is not None:
        cpu_slots.append(s)
    assert len(cpu_slots) == 15 and all(60 not in s for s in cpu_slots)
    assert len(br.plan_slots(topo, 63, 1, reserved=[1])) == 63
    with pytest.raises(ValueError, match="only 63 of 64"):
        br.plan_slots(topo, 64, 1, reserved=[1])


def test_pack_fills_one_group_before_the_next(topo):
    pool = br.CorePool(topo)
    first = [pool.acquire(1, pack=True)[0] for _ in range(5)]
    assert first == [0, 1, 2, 3, 4] and [group_of(topo, c) for c in first] == [0, 0, 0, 0, 1]


# ----------------------------------------------------------------------------------------------
# Job files and skip logic
# ----------------------------------------------------------------------------------------------


def write_jobs(path, jobs):
    path.write_text(json.dumps(jobs))
    return path


def test_parse_jobs_accepts_the_documented_fields():
    jobs = br.parse_jobs([{"name": "a", "cmd": ["scripts/run_posterior.py", "--n", "16"]},
                          {"name": "b.c-1", "cmd": ["x"], "device": "gpu", "threads": 2, "done_marker": "m.json"}])
    assert jobs[0].device == "cpu" and jobs[0].threads is None
    assert jobs[0].marker == "results/a/summary.json" and jobs[1].marker == "m.json"
    assert jobs[1].device == "gpu" and jobs[1].threads == 2


@pytest.mark.parametrize("bad, match", [
    ({"name": "a"}, "cmd"),
    ({"name": "a", "cmd": []}, "cmd"),
    ({"name": "a", "cmd": ["x", 16]}, "cmd"),
    ({"name": "a/b", "cmd": ["x"]}, "name"),
    ({"name": "a", "cmd": ["x"], "device": "tpu"}, "device"),
    ({"name": "a", "cmd": ["x"], "threads": 0}, "threads"),
    ({"name": "a", "cmd": ["x"], "done_marke": "m"}, "unknown key"),
])
def test_parse_jobs_rejects_bad_entries(bad, match):
    with pytest.raises(br.JobFileError, match=match):
        br.parse_jobs([bad])


def test_parse_jobs_rejects_duplicates_and_non_lists():
    with pytest.raises(br.JobFileError, match="duplicate"):
        br.parse_jobs([{"name": "a", "cmd": ["x"]}, {"name": "a", "cmd": ["y"]}])
    with pytest.raises(br.JobFileError, match="list"):
        br.parse_jobs({"name": "a"})


def test_job_is_done_follows_the_marker(tmp_path):
    job = br.Job("j", ("x",))
    assert not br.job_is_done(job, tmp_path)
    (tmp_path / "results" / "j").mkdir(parents=True)
    (tmp_path / "results" / "j" / "summary.json").write_text("{}")
    assert br.job_is_done(job, tmp_path)
    custom = br.Job("k", ("x",), done_marker="out/k.json")
    assert not br.job_is_done(custom, tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "k.json").write_text("{}")
    assert br.job_is_done(custom, tmp_path)
    assert br.job_is_done(br.Job("a", ("x",), done_marker=str(tmp_path / "out" / "k.json")), Path("/elsewhere"))


def test_a_marker_left_by_a_failed_or_interrupted_run_does_not_count_as_done(tmp_path):
    """run_posterior writes summary.json before chains.npz and the figures; a job killed or crashed after that
    leaves the marker behind. The meta.json of the last batch_run says what really happened."""
    job = br.Job("j", ("x",))
    logs = tmp_path / "logs"
    logs.mkdir()
    (tmp_path / "results" / "j").mkdir(parents=True)
    (tmp_path / "results" / "j" / "summary.json").write_text("{}")
    assert br.job_is_done(job, tmp_path, logs)  # no record: the marker is trusted (job run by hand)
    for status, done in (("ok", True), ("failed", False), ("interrupted", False), ("running", False)):
        (logs / "j.meta.json").write_text(json.dumps({"name": "j", "status": status}))
        assert br.job_is_done(job, tmp_path, logs) is done, status
        assert br.stale_marker_status(job, tmp_path, logs) == (None if done else status)
    (logs / "j.meta.json").write_text("not json {")
    assert br.job_is_done(job, tmp_path, logs)  # an unreadable record does not block the marker
    assert br.job_is_done(job, tmp_path)  # without a log directory only the marker counts
    (tmp_path / "results" / "j" / "summary.json").unlink()
    (logs / "j.meta.json").write_text(json.dumps({"status": "ok"}))
    assert not br.job_is_done(job, tmp_path, logs)  # no marker: never done, whatever the record says


def three_jobs(tmp_path):
    return write_jobs(tmp_path / "jobs.json", [
        {"name": "alpha", "cmd": ["scripts/run_posterior.py", "--out", "alpha"]},
        {"name": "beta", "cmd": ["scripts/run_posterior.py", "--out", "beta"]},
        {"name": "gamma", "cmd": ["scripts/run_posterior.py", "--out", "gamma"], "done_marker": "marks/gamma.done"}])


def test_dry_run_skips_done_jobs_unless_forced(tmp_path, capsys):
    jobs = three_jobs(tmp_path)
    (tmp_path / "results" / "beta").mkdir(parents=True)
    (tmp_path / "results" / "beta" / "summary.json").write_text("{}")
    assert br.main([str(jobs), "--root", str(tmp_path), "--cpu-slots", "1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 job(s) to run, 1 skipped" in out
    assert "SKIP   beta" in out and "RUN    alpha" in out and "QUEUED gamma" in out
    assert not (tmp_path / "results" / "logs").exists()  # a dry run writes nothing
    assert br.main([str(jobs), "--root", str(tmp_path), "--cpu-slots", "3", "--dry-run", "--force"]) == 0
    forced = capsys.readouterr().out
    assert "SKIP" not in forced and all(f"RUN    {n}" in forced for n in ("alpha", "beta", "gamma"))


def test_dry_run_shows_command_environment_and_log(tmp_path, capsys):
    jobs = three_jobs(tmp_path)
    assert br.main([str(jobs), "--root", str(tmp_path), "--threads", "2", "--cpu-slots", "1", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"cmd: {sys.executable} scripts/run_posterior.py --out alpha" in out
    assert "JAX_PLATFORMS=cpu" in out and "NPROC=4" in out and "OMP_NUM_THREADS=2" in out  # pool floor, see below
    assert f"log: {tmp_path / 'results' / 'logs' / 'alpha.log'}" in out and "unpinned" in out


def test_dry_run_pinned_assigns_spread_cores_and_reserves_the_gpu_host(tmp_path, capsys, sysfs):
    jobs = write_jobs(tmp_path / "jobs.json", [
        {"name": "c1", "cmd": ["a.py"]}, {"name": "c2", "cmd": ["a.py"]}, {"name": "c3", "cmd": ["a.py"]},
        {"name": "g1", "cmd": ["a.py"], "device": "gpu"}])
    argv = [str(jobs), "--root", str(tmp_path), "--pin", "--sysfs-root", str(sysfs), "--cpu-slots", "3",
            "--dry-run"]
    assert br.main(argv, **LINUX_WITH_TASKSET) == 0
    out = capsys.readouterr().out
    assert "pinned (taskset)" in out
    assert f"cmd: /usr/bin/taskset -c 60 {sys.executable} a.py" in out  # GPU host: first core of the last group
    # CPU jobs: one core in each of groups 0, 1, 2 (spread), the reserved core 60 is not used
    for cpu in (0, 4, 8):
        assert f"cmd: /usr/bin/taskset -c {cpu} {sys.executable} a.py" in out
    assert "device=gpu slot=0 threads=1 cores=60" in out and "CUDA_VISIBLE_DEVICES=0" in out
    assert br.main(argv[:-1] + ["--threads", "4", "--dry-run"], **LINUX_WITH_TASKSET) == 0
    four = capsys.readouterr().out
    assert "taskset -c 0-3 " in four and "taskset -c 4-7 " in four and "taskset -c 8-11 " in four


def test_pin_requests_that_do_not_fit_are_rejected(tmp_path, capsys, sysfs):
    jobs = write_jobs(tmp_path / "jobs.json", [{"name": f"c{i}", "cmd": ["a.py"]} for i in range(20)])
    base = [str(jobs), "--root", str(tmp_path), "--pin", "--sysfs-root", str(sysfs), "--dry-run"]
    assert br.main(base + ["--cpu-slots", "17", "--threads", "4"], **LINUX_WITH_TASKSET) == 2
    assert "only 16 of 17" in capsys.readouterr().err
    assert br.main(base + ["--cpu-slots", "2", "--threads", "5"], **LINUX_WITH_TASKSET) == 2
    assert "exceeds the largest L3 group" in capsys.readouterr().err
    assert br.main(base + ["--cpu-slots", "16", "--threads", "4"], **LINUX_WITH_TASKSET) == 0


def test_pin_with_a_gpu_job_leaves_one_core_less(tmp_path, capsys, sysfs):
    many = [{"name": f"c{i}", "cmd": ["a.py"]} for i in range(70)] + [{"name": "g", "cmd": ["a.py"], "device": "gpu"}]
    jobs = write_jobs(tmp_path / "jobs.json", many)
    base = [str(jobs), "--root", str(tmp_path), "--pin", "--sysfs-root", str(sysfs), "--dry-run", "--gpu-slots", "1"]
    assert br.main(base + ["--cpu-slots", "63"], **LINUX_WITH_TASKSET) == 0
    assert br.main(base + ["--cpu-slots", "64"], **LINUX_WITH_TASKSET) == 2
    assert "only 63 of 64" in capsys.readouterr().err


def test_pin_off_linux_warns_and_runs_unpinned(tmp_path, capsys, sysfs):
    jobs = three_jobs(tmp_path)
    argv = [str(jobs), "--root", str(tmp_path), "--pin", "--sysfs-root", str(sysfs), "--dry-run"]
    assert br.main(argv, system="Darwin") == 0
    cap = capsys.readouterr()
    assert "--pin works only on Linux" in cap.err and "unpinned" in cap.out and "taskset" not in cap.out
    assert br.main(argv, system="Linux", which=lambda name: None) == 0  # a real run would warn; dry run shows plan
    capsys.readouterr()


def test_show_topology_prints_groups_and_cores(sysfs, capsys):
    assert br.main(["--show-topology", "--sysfs-root", str(sysfs)], system="Linux") == 0
    out = capsys.readouterr().out
    assert "16 L3 group(s), 64 physical core(s), 128 logical CPU(s)" in out and "group  0 (L3 0-3,64-67): 0+64 1+65" in out


def test_job_file_errors_and_slot_checks_exit_with_status_2(tmp_path, capsys):
    bad = write_jobs(tmp_path / "bad.json", [{"name": "a", "cmd": ["x"], "device": "tpu"}])
    assert br.main([str(bad), "--root", str(tmp_path), "--dry-run"]) == 2
    assert "device" in capsys.readouterr().err
    gpu = write_jobs(tmp_path / "gpu.json", [{"name": "g", "cmd": ["x"], "device": "gpu"}])
    assert br.main([str(gpu), "--root", str(tmp_path), "--gpu-slots", "0", "--dry-run"]) == 2
    assert "--gpu-slots is 0" in capsys.readouterr().err
    assert br.main([str(tmp_path / "missing.json"), "--root", str(tmp_path)]) == 2


# ----------------------------------------------------------------------------------------------
# Environments and commands
# ----------------------------------------------------------------------------------------------


def test_cpu_job_environment_for_one_thread():
    env = br.cpu_job_env(1)
    assert env["JAX_PLATFORMS"] == "cpu" and env["NPROC"] == "1"
    assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert env[var] == "1"
    assert "intra_op_parallelism_threads" not in env["XLA_FLAGS"]  # not an XLA flag in this jaxlib


def test_cpu_job_environment_for_several_threads():
    env = br.cpu_job_env(4)
    assert env["NPROC"] == "4" and env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=true"
    assert all(env[v] == "4" for v in br.THREAD_ENV_VARS) and env["JAX_PLATFORMS"] == "cpu"


def test_xla_pool_never_has_exactly_two_workers():
    """NPROC=2 with Eigen multi-threading on hangs the FFT thunk (both workers wait for helpers queued on the
    same pool); measured, see batch_run.xla_pool_size. One thread means a pool of one with Eigen off."""
    assert [br.xla_pool_size(t) for t in (1, 2, 3, 4, 8, 16)] == [1, 4, 4, 4, 8, 16]
    assert br.xla_pool_size(1) == 1 and br.cpu_job_env(1)["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=false"
    for t in (2, 3, 4, 8):
        env = br.cpu_job_env(t)
        assert int(env["NPROC"]) >= max(t, br.MIN_XLA_POOL) and env["OMP_NUM_THREADS"] == str(t)
        assert env["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=true"


def test_xla_flags_merge_keeps_foreign_flags_and_replaces_duplicates():
    merged = br.merge_xla_flags("--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=true",
                                ["--xla_cpu_multi_thread_eigen=false"])
    assert merged == "--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false"
    assert br.cpu_job_env(1, "--foo=1")["XLA_FLAGS"] == "--foo=1 --xla_cpu_multi_thread_eigen=false"
    assert br.merge_xla_flags(None, ["--a=1"]) == "--a=1"


def test_gpu_job_environment():
    env = br.gpu_job_env("1", n_sharing=1)
    assert env["JAX_PLATFORMS"] == "cuda,cpu" and env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.9"
    assert "XLA_FLAGS" not in env and "NPROC" not in env and env["OMP_NUM_THREADS"] == "1"  # no XLA CPU client
    assert br.gpu_job_env(0, n_sharing=2)["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.45"
    assert br.gpu_job_env(0, n_sharing=3)["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.3"
    assert br.gpu_job_env(0, n_sharing=7)["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.128"  # rounded down


def test_gpu_slots_spread_over_devices_and_share_memory_per_device(tmp_path):
    opts = br.Options(root=tmp_path, log_dir=tmp_path, python="py", gpu_slots=3, gpu_devices=("0", "1"))
    job = br.Job("g", ("x",), device="gpu")
    envs = [br.job_env(opts, job, s) for s in range(3)]
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "0"]
    assert [e["XLA_PYTHON_CLIENT_MEM_FRACTION"] for e in envs] == ["0.45", "0.9", "0.45"]
    cpu = br.job_env(opts, br.Job("c", ("x",), threads=3), 0)
    assert cpu["NPROC"] == "4" and cpu["OMP_NUM_THREADS"] == "3" and cpu["JAX_PLATFORMS"] == "cpu"


def test_build_command(tmp_path):
    opts = br.Options(root=tmp_path, log_dir=tmp_path, python="/env/bin/python", pinned=True, taskset="/usr/bin/taskset")
    assert br.build_command(opts, br.Job("a", ("scripts/x.py", "--n", "8"))) == [
        "/env/bin/python", "scripts/x.py", "--n", "8"]
    assert br.build_command(opts, br.Job("a", ("python", "-c", "pass")), (4, 5, 6, 7, 68))[:3] == [
        "/usr/bin/taskset", "-c", "4-7,68"]
    assert br.build_command(opts, br.Job("a", ("python3", "-m", "pytest"))) == ["/env/bin/python", "-m", "pytest"]
    assert br.build_command(opts, br.Job("a", ("echo", "hi"))) == ["echo", "hi"]
    opts.pinned = False
    assert br.build_command(opts, br.Job("a", ("echo",)), (1,)) == ["echo"]


# ----------------------------------------------------------------------------------------------
# Real runs of trivial commands
# ----------------------------------------------------------------------------------------------

WRITE_ENV = (
    "import json, os, sys, time\n"
    "name = sys.argv[1]\n"
    "keys = ['JAX_PLATFORMS', 'XLA_FLAGS', 'NPROC', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',\n"
    "        'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION']\n"
    "json.dump({k: os.environ.get(k) for k in keys}, open('env_' + name + '.json', 'w'))\n"
    "os.makedirs('results/' + name, exist_ok=True)\n"
    "open('results/' + name + '/summary.json', 'w').write('{}')\n"
    "print('hello from', name)\n"
)


def job(name, device="cpu", code=WRITE_ENV, **kw):
    return {"name": name, "cmd": [sys.executable, "-c", code, name], "device": device, **kw}


def run_main(tmp_path, jobs, *extra):
    f = write_jobs(tmp_path / "jobs.json", jobs)
    return br.main([str(f), "--root", str(tmp_path), "--poll", "0.02", *extra])


def test_run_writes_logs_meta_and_passes_the_environment(tmp_path, capsys):
    jobs = [job("c1", threads=2), job("g1", device="gpu")]
    assert run_main(tmp_path, jobs, "--cpu-slots", "1", "--gpu-slots", "1", "--gpu-devices", "1") == 0
    logs = tmp_path / "results" / "logs"
    assert "hello from c1" in (logs / "c1.log").read_text() and "hello from g1" in (logs / "g1.log").read_text()
    cpu_env = json.loads((tmp_path / "env_c1.json").read_text())
    assert cpu_env["JAX_PLATFORMS"] == "cpu" and cpu_env["NPROC"] == "4" and cpu_env["OMP_NUM_THREADS"] == "2"
    assert "--xla_cpu_multi_thread_eigen=true" in cpu_env["XLA_FLAGS"]
    gpu_env = json.loads((tmp_path / "env_g1.json").read_text())
    assert gpu_env["JAX_PLATFORMS"] == "cuda,cpu" and gpu_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert gpu_env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false" and gpu_env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.9"
    meta = json.loads((logs / "c1.meta.json").read_text())
    assert meta["status"] == "ok" and meta["exit_code"] == 0 and meta["wall_s"] >= 0
    assert meta["device"] == "cpu" and meta["threads"] == 2 and meta["pinned"] is False and meta["cores"] is None
    assert meta["env"]["NPROC"] == "4" and meta["cmd"][0] == sys.executable
    assert meta["host"]["hostname"] and meta["start"] and meta["end"] and meta["marker_present"] is True
    assert json.loads((logs / "g1.meta.json").read_text())["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "# env" in (logs / "c1.log").read_text()


def test_second_run_skips_finished_jobs_and_force_reruns_them(tmp_path, capsys):
    jobs = [job("c1"), job("c2")]
    assert run_main(tmp_path, jobs) == 0
    log = tmp_path / "results" / "logs" / "c1.log"
    first = log.stat().st_mtime_ns
    capsys.readouterr()
    assert run_main(tmp_path, jobs) == 0
    out = capsys.readouterr().out
    assert "0 job(s) to run, 2 skipped" in out and log.stat().st_mtime_ns == first
    assert run_main(tmp_path, jobs, "--force") == 0
    assert log.stat().st_mtime_ns > first


def test_failed_job_gives_nonzero_status_but_the_others_still_run(tmp_path, capsys):
    jobs = [job("bad", code="import sys; sys.exit(3)"), job("good")]
    assert run_main(tmp_path, jobs, "--cpu-slots", "1") == 1
    logs = tmp_path / "results" / "logs"
    bad = json.loads((logs / "bad.meta.json").read_text())
    assert bad["status"] == "failed" and bad["exit_code"] == 3
    assert json.loads((logs / "good.meta.json").read_text())["status"] == "ok"
    out = capsys.readouterr().out
    assert "FAILED" in out and "1 ok, 1 not ok" in out


def test_missing_executable_is_a_failed_job(tmp_path):
    f = {"name": "nope", "cmd": ["/definitely/not/a/program"]}
    assert run_main(tmp_path, [f, job("fine")]) == 1
    meta = json.loads((tmp_path / "results" / "logs" / "nope.meta.json").read_text())
    assert meta["status"] == "failed" and meta["exit_code"] == 127


def test_exit_zero_without_the_marker_is_reported(tmp_path, capsys):
    assert run_main(tmp_path, [job("quiet", code="print('no marker written')")]) == 0
    assert "WARNING: exited 0 but" in capsys.readouterr().out
    meta = json.loads((tmp_path / "results" / "logs" / "quiet.meta.json").read_text())
    assert meta["marker_present"] is False


STAMP = (
    "import sys, time\n"
    "name = sys.argv[1]\n"
    "t0 = time.time()\n"
    "time.sleep(0.3)\n"
    "open('stamp_' + name, 'w').write(f'{t0} {time.time()}')\n"
)

# A rendezvous instead of a sleep: the job announces itself, then waits (up to 60 s) until ``need`` jobs have
# announced themselves. It succeeds only if that many jobs were running at the same time, however slow the
# machine is, so concurrency tests do not depend on how long a Python start-up takes.
RENDEZVOUS = (
    "import glob, sys, time\n"
    "name, need = sys.argv[1], int(sys.argv[2])\n"
    "open('here_' + name, 'w').write('1')\n"
    "t0 = time.time()\n"
    "while len(glob.glob('here_*')) < need and time.time() - t0 < 60:\n"
    "    time.sleep(0.01)\n"
    "open('met_' + name, 'w').write(str(len(glob.glob('here_*')) >= need))\n"
)


def rendezvous_job(name, need, device="cpu"):
    return {"name": name, "cmd": [sys.executable, "-c", RENDEZVOUS, name, str(need)], "device": device,
            "done_marker": f"met_{name}"}


def stamps(tmp_path, names):
    out = []
    for n in names:
        a, b = (tmp_path / f"stamp_{n}").read_text().split()
        out.append((float(a), float(b)))
    return out


def test_one_cpu_slot_runs_jobs_one_after_the_other(tmp_path):
    jobs = [job(n, code=STAMP, done_marker=f"m_{n}") for n in ("a", "b")]
    assert run_main(tmp_path, jobs, "--cpu-slots", "1") == 0  # no marker is only a warning
    (s1, e1), (s2, e2) = stamps(tmp_path, ["a", "b"])
    assert s2 >= e1 - 0.05  # b starts after a ends


def test_two_cpu_slots_run_two_jobs_at_once(tmp_path):
    jobs = [rendezvous_job(n, 2) for n in ("a", "b")]
    assert run_main(tmp_path, jobs, "--cpu-slots", "2") == 0
    assert all((tmp_path / f"met_{n}").read_text() == "True" for n in ("a", "b"))  # each saw the other running


def test_cpu_and_gpu_queues_run_side_by_side(tmp_path):
    jobs = [rendezvous_job("c", 2), rendezvous_job("g", 2, device="gpu")]
    assert run_main(tmp_path, jobs, "--cpu-slots", "1", "--gpu-slots", "1") == 0
    assert all((tmp_path / f"met_{n}").read_text() == "True" for n in ("c", "g"))


def test_a_failed_run_with_a_marker_is_run_again_and_a_clean_one_is_skipped(tmp_path, capsys):
    """The job writes its marker and then fails the first time (as run_posterior would if plotting crashed)."""
    code = (
        "import os, sys\n"
        "os.makedirs('results/flaky', exist_ok=True)\n"
        "open('results/flaky/summary.json', 'w').write('{}')\n"
        "first = not os.path.exists('seen')\n"
        "open('seen', 'w').write('1')\n"
        "sys.exit(3 if first else 0)\n"
    )
    jobs = [{"name": "flaky", "cmd": [sys.executable, "-c", code]}]
    assert run_main(tmp_path, jobs) == 1  # fails, marker left behind
    assert (tmp_path / "results" / "flaky" / "summary.json").exists()
    capsys.readouterr()
    assert run_main(tmp_path, jobs) == 0  # the marker is not trusted: it runs again and now succeeds
    cap = capsys.readouterr()
    assert "1 job(s) to run, 0 skipped" in cap.out and "recorded as 'failed'" in cap.err
    assert run_main(tmp_path, jobs) == 0
    assert "0 job(s) to run, 1 skipped" in capsys.readouterr().out  # clean record: skipped now


def test_children_are_terminated_when_the_runner_itself_fails(tmp_path, monkeypatch):
    pid_code = "import os, sys, time\nopen('pid_' + sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    opts = br.Options(root=tmp_path, log_dir=tmp_path / "logs", python=sys.executable, cpu_slots=2, poll=0.02,
                      grace=5.0)
    runner = br.Runner(opts, br.parse_jobs([job("s1", code=pid_code), job("s2", code=pid_code)]))
    calls = []
    real_reap = runner._reap

    def reap_then_fail():
        calls.append(1)
        if all((tmp_path / f"pid_{n}").exists() and (tmp_path / f"pid_{n}").read_text() for n in ("s1", "s2")):
            raise OSError("disk full")
        real_reap()

    monkeypatch.setattr(runner, "_reap", reap_then_fail)
    with pytest.raises(OSError, match="disk full"):
        runner.run()
    for n in ("s1", "s2"):
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / f"pid_{n}").read_text()), 0)
    assert json.loads((tmp_path / "logs" / "s1.meta.json").read_text())["status"] == "interrupted"


def test_job_timeout_kills_a_hung_job_and_the_rest_still_run(tmp_path, capsys):
    hang = "import time\ntime.sleep(30)\n"
    jobs = [job("hung", code=hang), job("fine")]
    status = run_main(tmp_path, jobs, "--cpu-slots", "1", "--job-timeout", "1", "--grace", "2")
    assert status == 1  # a timeout is a failure
    logs = tmp_path / "results" / "logs"
    hung = json.loads((logs / "hung.meta.json").read_text())
    assert hung["status"] == "timeout" and hung["signal"] == "SIGTERM" and 1 <= hung["wall_s"] < 60
    assert json.loads((logs / "fine.meta.json").read_text())["status"] == "ok"  # the slot was freed for the next job
    out = capsys.readouterr().out
    assert "TIMEOUT hung after" in out and "1 ok, 1 not ok" in out
    assert br.stale_marker_status(br.Job("hung", ("x",), done_marker="results/hung/summary.json"), tmp_path,
                                  logs) is None  # no marker was written, so nothing is stale


def test_no_job_timeout_by_default(tmp_path):
    assert br.Options(root=tmp_path, log_dir=tmp_path, python="py").job_timeout == 0.0
    assert run_main(tmp_path, [job("quick")]) == 0
    assert br.main([str(tmp_path / "jobs.json"), "--root", str(tmp_path), "--job-timeout", "-1"]) == 2


FAKE_TASKSET = "#!/bin/sh\n# taskset -c LIST command...: record LIST, then run the command unpinned\necho \"$2\" >> \"$FAKE_TASKSET_LOG\"\nshift 2\nexec \"$@\"\n"

PINNED_JOB = (
    "import glob, sys, time\n"
    "name, barrier = sys.argv[1], int(sys.argv[2])\n"
    "t0 = time.time()\n"
    "open('here_' + name, 'w').write('1')\n"
    "while barrier and len(glob.glob('here_*')) < barrier and time.time() - t0 < 60:\n"
    "    time.sleep(0.01)\n"
    "time.sleep(0.05)\n"
    "open('stamp_' + name, 'w').write(f'{t0} {time.time()}')\n"
    "open('done_' + name, 'w').write('1')\n"
)


def test_a_pinned_run_never_gives_one_cpu_to_two_running_jobs(tmp_path, monkeypatch):
    """Live runner, fake sysfs (2 L3 groups of 2 cores), fake taskset: 8 one-thread jobs through 3 slots, so
    cores are acquired and released repeatedly. The first three jobs meet at a barrier, so they certainly
    overlap; any two jobs whose lifetimes overlap must hold disjoint CPUs, all inside one L3 group."""
    sysfs = make_sysfs(tmp_path / "sys", n_groups=2, cores_per_group=2, smt=1)
    fake = tmp_path / "taskset"
    fake.write_text(FAKE_TASKSET)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_TASKSET_LOG", str(tmp_path / "taskset.log"))
    names = [f"p{i}" for i in range(8)]
    jobs = [{"name": n, "cmd": [sys.executable, "-c", PINNED_JOB, n, "3" if i < 3 else "0"],
             "done_marker": f"done_{n}"} for i, n in enumerate(names)]
    f = write_jobs(tmp_path / "jobs.json", jobs)
    argv = [str(f), "--root", str(tmp_path), "--poll", "0.02", "--pin", "--sysfs-root", str(sysfs), "--cpu-slots", "3"]
    assert br.main(argv, system="Linux", which=lambda name: str(fake)) == 0
    logs = tmp_path / "results" / "logs"
    metas = {n: json.loads((logs / f"{n}.meta.json").read_text()) for n in names}
    assert all(m["status"] == "ok" and m["pinned"] for m in metas.values())
    cpus = {n: set(br.parse_cpu_list(m["cores"])) for n, m in metas.items()}
    assert all(len(c) == 1 and c <= {0, 1, 2, 3} for c in cpus.values())
    life = dict(zip(names, stamps(tmp_path, names)))
    overlapping = 0
    for i, a in enumerate(names):
        for b in names[:i]:
            if life[a][0] < life[b][1] and life[b][0] < life[a][1]:  # a and b were alive at the same time
                overlapping += 1
                assert not cpus[a] & cpus[b], (a, b, cpus[a], cpus[b])
    assert overlapping >= 3  # at least the barrier trio
    taskset_args = sorted((tmp_path / "taskset.log").read_text().split())
    assert taskset_args == sorted(metas[n]["cores"] for n in names)  # the command line really carried the cores


def test_a_pinned_run_with_threads_keeps_each_job_inside_one_l3_group(tmp_path, monkeypatch):
    sysfs = make_sysfs(tmp_path / "sys", n_groups=4, cores_per_group=4, smt=2)
    fake = tmp_path / "taskset"
    fake.write_text(FAKE_TASKSET)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_TASKSET_LOG", str(tmp_path / "taskset.log"))
    names = [f"q{i}" for i in range(6)]
    jobs = [{"name": n, "cmd": [sys.executable, "-c", PINNED_JOB, n, "4" if i < 4 else "0"],
             "done_marker": f"done_{n}", "threads": 3 if i % 2 else 2} for i, n in enumerate(names)]
    f = write_jobs(tmp_path / "jobs.json", jobs)
    argv = [str(f), "--root", str(tmp_path), "--poll", "0.02", "--pin", "--sysfs-root", str(sysfs), "--cpu-slots", "4"]
    assert br.main(argv, system="Linux", which=lambda name: str(fake)) == 0
    topology = br.read_topology(sysfs)
    held = []
    for i, n in enumerate(names):
        meta = json.loads((tmp_path / "results" / "logs" / f"{n}.meta.json").read_text())
        cpus = br.parse_cpu_list(meta["cores"])
        assert len(cpus) == (3 if i % 2 else 2) and all(c < 16 for c in cpus)  # first hardware threads only
        assert len({topology.group_index(c) for c in cpus}) == 1
        held.append(set(cpus))
    life = stamps(tmp_path, names)
    for i in range(len(names)):
        for j in range(i):
            if life[i][0] < life[j][1] and life[j][0] < life[i][1]:
                assert not held[i] & held[j]


def test_ctrl_c_terminates_children_and_exits_130(tmp_path):
    pid_code = (
        "import os, sys, time\n"
        "open('pid_' + sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    f = write_jobs(tmp_path / "jobs.json", [job("sleeper", code=pid_code), job("sleeper2", code=pid_code)])
    proc = subprocess.Popen([sys.executable, str(SCRIPT), str(f), "--root", str(tmp_path), "--cpu-slots", "2",
                             "--poll", "0.02", "--grace", "5"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not all((tmp_path / f"pid_{n}").exists() and (tmp_path / f"pid_{n}").read_text()
                                                  for n in ("sleeper", "sleeper2")):
            time.sleep(0.05)
        pids = [int((tmp_path / f"pid_{n}").read_text()) for n in ("sleeper", "sleeper2")]
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 130, out
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    meta = json.loads((tmp_path / "results" / "logs" / "sleeper.meta.json").read_text())
    assert meta["status"] == "interrupted" and meta["signal"] == "SIGTERM"


# ----------------------------------------------------------------------------------------------
# scripts/bench_backend.py: import order and small helpers (no JAX computation)
# ----------------------------------------------------------------------------------------------

BENCH_DIR = SCRIPT.parent


def run_python(code, env_extra=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("JAX_PLATFORMS", "XLA_FLAGS", "NPROC", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)


def test_bench_backend_sets_the_environment_before_jax_or_numpy_are_imported():
    code = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(BENCH_DIR)!r})\n"
        "import bench_backend as b\n"
        "heavy = {'jax', 'jaxlib', 'numpy', 'bayes_scatter', 'scipy'}\n"
        "assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))\n"
        "a = b.parse_args(['--platform', 'cpu', '--threads', '2', '--xla-flags=--foo=1'])\n"
        "changed = b.apply_environment(a)\n"
        "assert not heavy & set(sys.modules)\n"
        "print(json.dumps({k: os.environ.get(k) for k in ('JAX_PLATFORMS', 'NPROC', 'OMP_NUM_THREADS', 'XLA_FLAGS')}))\n"
    )
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    env = json.loads(out.stdout.strip().splitlines()[-1])
    assert env["JAX_PLATFORMS"] == "cpu" and env["NPROC"] == "4" and env["OMP_NUM_THREADS"] == "2"  # pool of 2 hangs
    assert "--xla_cpu_multi_thread_eigen=true" in env["XLA_FLAGS"] and "--foo=1" in env["XLA_FLAGS"]


def test_bench_backend_refuses_to_set_the_environment_after_jax_is_imported():
    code = (
        f"import sys; sys.path.insert(0, {str(BENCH_DIR)!r})\n"
        "import bench_backend as b\n"
        "import jax\n"
        "try:\n"
        "    b.apply_environment(b.parse_args(['--platform', 'cpu']))\n"
        "except RuntimeError as exc:\n"
        "    print('refused', exc)\n"
    )
    out = run_python(code)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("refused jax is already imported")


def test_bench_backend_helpers(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH_DIR))
    import argparse

    import bench_backend as b

    assert b.parse_grids("32:2, 64:5") == [(32, 2.0), (64, 5.0)]
    for bad in ("64", "1:2", "64:0", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            b.parse_grids(bad)
    assert b.parse_int_list("1,8,32") == [1, 8, 32]
    with pytest.raises(argparse.ArgumentTypeError):
        b.parse_int_list("0")
    rows = [{"error": None}, {"error": "RESOURCE_EXHAUSTED", "oom": True},
            {"error": "skipped: B=8 already failed", "oom": False}, {"error": "ValueError: boom", "oom": False}]
    assert b.failed_cases(rows) == [rows[3]]  # out-of-memory and skipped cases are expected outcomes
    assert b.is_oom(MemoryError()) and b.is_oom(RuntimeError("RESOURCE_EXHAUSTED: Out of memory")) and not b.is_oom(ValueError("x"))
    assert b.krylov_basis_bytes(96, 40, 100, 1) == 16 * 101 * 40 * 96**2  # about 568 MiB per lane, as WORKSTATION.md says
    assert b.krylov_basis_bytes(96, 40, 100, 4) == 4 * b.krylov_basis_bytes(96, 40, 100, 1)


def test_bench_backend_exit_status_and_result_file(tmp_path):
    """Runs the FFT micro-benchmark (tiny, CPU, one thread) in a subprocess: exit 0 and a final file when all is
    well; with an injected failure in the timing routine, exit 1 but the file is still written, and no
    .partial.json is left behind."""
    def run(inject):
        out = tmp_path / ("bad.json" if inject else "good.json")
        code = (
            f"import sys; sys.path.insert(0, {str(BENCH_DIR)!r})\n"
            "import bench_backend as b\n"
            + ("def boom(*a, **k):\n    raise ValueError('injected failure')\nb.time_calls = boom\n" if inject else "")
            + f"sys.exit(b.main(['--platform', 'cpu', '--threads', '1', '--grids', '8:2', '--modes', 'fft', "
              f"'--repeats', '2', '--out', {str(out)!r}]))\n"
        )
        return run_python(code, {"PYTHONPATH": ""}), out

    good, good_file = run(False)
    assert good.returncode == 0, good.stderr
    data = json.loads(good_file.read_text())
    assert data["failed_cases"] == 0 and data["interrupted"] is False and data["provenance"]["x64_enabled"] is True
    rows = {r["variant"]: r for r in data["results"]}
    assert set(rows) == {"c64", "c128", "c128/c64"} and rows["c64"]["median_s"] > 0 and rows["c64"]["repeats"] == 2
    assert rows["c128"]["out_dtype"] == "complex128" and rows["c64"]["out_dtype"] == "complex64"
    assert data["provenance"]["env"]["NPROC"] == "1" and data["provenance"]["env"]["JAX_PLATFORMS"] == "cpu"
    assert "| fft n=8 | fft | c64 |" in good.stdout
    bad, bad_file = run(True)
    assert bad.returncode == 1, bad.stderr
    data = json.loads(bad_file.read_text())
    assert data["failed_cases"] == 2 and all("injected failure" in r["error"] for r in data["results"] if r.get("error"))
    assert not list(tmp_path.glob("*.partial.json"))
