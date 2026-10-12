"""scripts/wait_then.py (dependency and go/skip gating of batch_run jobs) and scripts/make_m5_batch.py (the milestone-5 job file)."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_m5_batch as mb  # noqa: E402
import wait_then as wt  # noqa: E402

SCRIPT = str(ROOT / "scripts" / "wait_then.py")


def meta(log_dir, name, status):
    (log_dir / f"{name}.meta.json").write_text(json.dumps({"name": name, "status": status}))


def run(args, **kw):
    return subprocess.run([sys.executable, SCRIPT, "--poll", "0.05", *args], capture_output=True, text=True, timeout=60, **kw)


def test_check_waits_until_every_dependency_holds(tmp_path):
    f, go = tmp_path / "input.npz", tmp_path / "go"
    logs = tmp_path / "logs"
    logs.mkdir()
    kw = dict(files=[str(f)], metas=["prep"], go=str(go), skip=str(tmp_path / "skip"), log_dir=logs)
    assert wt.check(**kw) == "wait"
    f.write_text("x")
    meta(logs, "prep", "running")
    assert wt.check(**kw) == "wait"  # the watched job has not finished
    meta(logs, "prep", "ok")
    assert wt.check(**kw) == "wait"  # no go file yet
    go.write_text("")
    assert wt.check(**kw) == "ready"
    (tmp_path / "skip").write_text("")
    assert wt.check(**kw) == "skip"  # skip wins over everything


@pytest.mark.parametrize("status", ["failed", "timeout", "interrupted"])
def test_check_reports_a_failed_dependency(tmp_path, status):
    meta(tmp_path, "prep", status)
    assert wt.check([], ["prep"], None, None, tmp_path) == "fail"


def test_relative_paths_are_relative_to_the_given_root(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    assert wt.check(["a.txt"], [], None, None, tmp_path, root=tmp_path) == "ready"
    assert wt.check(["b.txt"], [], None, None, tmp_path, root=tmp_path) == "wait"


def test_runs_the_command_once_the_go_file_appears(tmp_path):
    go = tmp_path / "go"
    proc = subprocess.Popen([sys.executable, SCRIPT, "--poll", "0.05", "--go", str(go), "--", "python", "-c", "print('ran', 6 * 7)"],
                            stdout=subprocess.PIPE, text=True)
    time.sleep(0.5)
    assert proc.poll() is None  # still waiting: nothing has been run
    go.write_text("")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0 and "ran 42" in out


def test_skip_file_exits_zero_without_running(tmp_path):
    (tmp_path / "skip").write_text("")
    r = run(["--go", str(tmp_path / "never"), "--skip", str(tmp_path / "skip"), "--", "python", "-c", "print('ran')"])
    assert r.returncode == 0 and "ran\n" not in r.stdout and "skip file exists" in r.stdout


def test_failed_dependency_exits_nonzero_without_running(tmp_path):
    meta(tmp_path, "prep", "failed")
    r = run(["--log-dir", str(tmp_path), "--meta-ok", "prep", "--", "python", "-c", "print('ran')"])
    assert r.returncode == 1 and "ran\n" not in r.stdout


def test_missing_command_is_an_error():
    r = run(["--go", "x"])
    assert r.returncode != 0 and "usage" in (r.stderr + r.stdout)


def test_the_milestone_5_job_file_is_gated_and_within_the_thread_budget():
    jobs = mb.build()
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)) == 30
    by_name = {j["name"]: j for j in jobs}
    # prepares run immediately; every later job goes through wait_then.py and a chain job also through a go file
    for n in ("tv_lam3.75_adapt", "tv_lam11.25_adapt", "gmrf_adapt"):
        assert by_name[f"pc_{n}_prepare"]["cmd"][0] == "scripts/run_prior_comparison.py"
        assert by_name[f"pc_{n}_adaptation"]["cmd"][0] == "scripts/wait_then.py"
    chains = [j for j in jobs if "_c" in j["name"] and j["name"].endswith(("_s0", "_s1"))]
    assert len(chains) == 24 and all("--go" in j["cmd"] and "--skip" in j["cmd"] for j in chains)
    # only the GMRF adaptation chain (the longest sequential piece) uses more than one thread
    assert {j["name"] for j in jobs if j.get("threads", 1) > 1} == {"pc_gmrf_adapt_adaptation"}
    assert by_name["pc_gmrf_adapt_adaptation"]["threads"] == 4
    # the GMRF adaptation starts at a milestone-4 end state and the chains at the four end states
    gm = " ".join(by_name["pc_gmrf_adapt_adaptation"]["cmd"])
    assert "--mass-theta" in gm and "m4_end0.npy" in gm and "--adapt-start-scale 0" in gm
    assert "m4_ends.npy" in " ".join(by_name["pc_gmrf_adapt_c2_s0"]["cmd"])
    # a segment-1 job needs its own segment 0
    assert "chain_2_seg0.npz" in " ".join(by_name["pc_tv_lam11.25_adapt_c2_s1"]["cmd"])
