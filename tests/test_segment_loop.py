"""scripts/segment_loop.py (one chain's recorded segments, one process each) and scripts/make_m5_resume.py (the resume job file)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_m5_resume as mr  # noqa: E402
import segment_loop as sl  # noqa: E402

# a stand-in for run_prior_comparison.py: parses --out/--chain-id/--segment, appends a line to calls.txt, writes the segment file
FAKE_RUNNER = """
import sys
from pathlib import Path
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
out, chain, seg = Path(get("--out")), get("--chain-id"), get("--segment")
with open(out.parent / "calls.txt", "a") as fh:
    fh.write(" ".join([chain, seg, get("--samples"), get("--warmup") if "--warmup" in a else "-"]) + "\\n")
if "--fail-at" in a and a[a.index("--fail-at") + 1] == seg:
    sys.exit(3)
out.mkdir(exist_ok=True)
(out / f"chain_{chain}_seg{seg}.npz").write_bytes(b"x")
"""


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    monkeypatch.setattr(sl, "RUNNER", script)
    return tmp_path / "run"


def loop(run_dir, *extra, last=3, chain=1, common=()):
    return sl.main(["--out-dir", str(run_dir), "--chain-id", str(chain), "--first", "0", "--last", str(last), "--samples", "100",
                    "--warmup", "50", *extra, "--", "--out", str(run_dir), *common])


def calls(run_dir):
    return [line.split() for line in (run_dir.parent / "calls.txt").read_text().splitlines()]


def test_segment_command_warms_up_only_segment_zero():
    c0 = sl.segment_command(["--prior", "gmrf", "--adapt"], 2, 0, 100, 50, python="py")
    assert c0[0] == "py" and c0[1].endswith("run_prior_comparison.py")
    assert c0[2:5] == ["--prior", "gmrf", "--adapt"]  # the common arguments come first
    assert c0[5:] == ["--stage", "chain", "--chain-id", "2", "--segment", "0", "--samples", "100", "--warmup", "50"]
    c3 = sl.segment_command(["--prior", "gmrf"], 2, 3, 100, 50, python="py")
    assert "--warmup" not in c3 and c3[-4:] == ["--segment", "3", "--samples", "100"]


def test_pending_segments_skips_files_that_exist_and_rejects_a_gap(tmp_path):
    for j in (0, 1, 3):
        sl.segment_file(tmp_path, 0, j).write_bytes(b"x")
    assert sl.pending_segments(tmp_path, 0, 0, 5) == [2, 4, 5]
    assert sl.pending_segments(tmp_path, 1, 0, 2) == [0, 1, 2]  # another chain: nothing exists
    with pytest.raises(ValueError, match="segment 5 .* needs segment 4"):
        sl.pending_segments(tmp_path, 0, 5, 6)


def test_the_loop_runs_the_segments_in_order_and_resumes(fake):
    assert loop(fake, last=2) == 0
    assert calls(fake) == [["1", "0", "100", "50"], ["1", "1", "100", "-"], ["1", "2", "100", "-"]]
    assert all(sl.segment_file(fake, 1, j).exists() for j in range(3))
    assert loop(fake, last=4) == 0  # a second loop does only the new segments
    assert [c[1] for c in calls(fake)] == ["0", "1", "2", "3", "4"]


def test_the_stop_file_ends_the_loop_between_segments(fake, tmp_path):
    stop = tmp_path / "stop"
    stop.write_text("")
    assert loop(fake, "--stop", str(stop)) == 0
    assert not (tmp_path / "calls.txt").exists()  # stopped before the first segment
    stop.unlink()
    assert loop(fake, "--stop", str(stop), last=0) == 0 and len(calls(fake)) == 1


def test_a_failing_segment_ends_the_loop_with_status_one(fake):
    assert loop(fake, common=["--fail-at", "1"]) == 1
    assert [c[1] for c in calls(fake)] == ["0", "1"]  # segment 2 was not tried
    assert sl.segment_file(fake, 1, 0).exists() and not sl.segment_file(fake, 1, 1).exists()


def test_the_go_file_is_waited_for(fake, tmp_path):
    go = tmp_path / "go"
    go.write_text("")
    assert loop(fake, "--go", str(go), "--poll", "0.05", last=0) == 0 and len(calls(fake)) == 1


def test_resume_batch_has_twelve_single_thread_loops_with_the_right_segments():
    jobs = mr.build(seg_samples=100, gmrf_last=9, tva_last=7, tvb_last=4)
    assert len(jobs) == 12 and len({j["name"] for j in jobs}) == 12 and all(j["threads"] == 1 for j in jobs)
    json.dumps(jobs)  # serializable
    g0, a0, b0 = jobs[0], jobs[4], jobs[8]
    assert g0["cmd"][0] == "scripts/segment_loop.py" and "--go" not in g0["cmd"]  # the GMRF starts at once
    assert "--start-theta" in g0["cmd"] and "--adapt" in g0["cmd"] and g0["cmd"][g0["cmd"].index("--prior") + 1] == "gmrf"
    assert g0["done_marker"].endswith("pc_gmrf_adapt/chain_0_seg9.npz")
    for j, run, first, last in ((a0, "pc_tv_lam3.75_adapt", "2", "7"), (b0, "pc_tv_lam11.25_adapt", "1", "4")):
        assert j["cmd"][j["cmd"].index("--first") + 1] == first and j["cmd"][j["cmd"].index("--last") + 1] == last
        assert "--go" in j["cmd"] and j["cmd"][j["cmd"].index("--out") + 1] == run  # TV waits for a go file
    assert [j["done_marker"].split("/")[-1] for j in jobs[:4]] == [f"chain_{c}_seg9.npz" for c in range(4)]


def test_batch3_runs_the_readapted_directories_from_their_stored_starts():
    jobs = mr.build3(seg_samples=100, gmrf_last=19, tv_last=5)
    assert len(jobs) == 12 and len({j["name"] for j in jobs}) == 12
    g, a, b = jobs[0], jobs[4], jobs[8]
    for j, dest, last, go in ((g, "pc_gmrf_adapt2", 19, False), (a, "pc_tv_lam3.75_adapt2", 5, True), (b, "pc_tv_lam11.25_adapt2", 5, True)):
        cmd = j["cmd"]
        assert cmd[cmd.index("--out") + 1] == dest and cmd[cmd.index("--out-dir") + 1].endswith(dest)
        assert cmd[cmd.index("--start-theta") + 1].endswith(f"{dest}/starts.npy") and cmd[cmd.index("--first") + 1] == "0"
        assert cmd[cmd.index("--last") + 1] == str(last) and ("--go" in cmd) == go
        assert j["done_marker"].endswith(f"{dest}/chain_0_seg{last}.npz")
    assert g["cmd"][g["cmd"].index("--prior") + 1] == "gmrf" and a["cmd"][a["cmd"].index("--lam") + 1] == "3.75"
