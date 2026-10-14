"""Tests for scripts/parametric_run.py and scripts/parametric_collect.py (milestone 7): names, data caching, the staged resumable pipeline on a tiny problem, the cost
budget and the tables."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_m7_jobs as mjobs  # noqa: E402
import parametric_collect as pcol  # noqa: E402
import parametric_sweeps as pell  # noqa: E402
import parametric_profile as pprof  # noqa: E402
import parametric_run as prun  # noqa: E402

TINY = ["--n", "16", "--n-tx", "6", "--n-rx", "8", "--snr", "40", "--tol", "1e-11", "--side", "1.5"]
SMALL_RUN = ["--segment-draws", "30", "--segments", "2", "--warmup", "30", "--chains", "2", "--ppc-reps", "6", "--n-prior-starts", "1"]


def tiny_args(out, model="F2", truth="T1", gap="0.2", extra=()):
    return prun.parse_args(["--truth", truth, "--gap", gap, "--model", model, "--out-dir", str(out), *TINY, *extra])


# ----------------------------------------------------------------------------------------------
# Names, arguments, data
# ----------------------------------------------------------------------------------------------


def test_case_names_follow_the_truth_label_and_model():
    assert prun.truth_label("T1", 0.05) == "T1g0.05" and prun.truth_label("T2", 0.1) == "T2g0.1" and prun.truth_label("T3", 0.05) == "T3g0.05"
    assert prun.truth_label("T4", 0.0) == "T4" and prun.truth_label("T4", 0.3) == "T4"
    assert prun.truth_label("T5", 0.1) == "T5" and prun.truth_label("T5", 0.15) == "T5g0.15"
    a = prun.parse_args(["--truth", "T1", "--gap", "0.05", "--model", "F2u", "--out-dir", "somewhere"])
    assert prun.case_name(a) == "T1g0.05_F2u" and prun.case_dir(a) == Path("somewhere") / "T1g0.05_F2u"
    assert a.last_segment == a.segments - 1 and a.ti_temps >= 12
    with pytest.raises(SystemExit):
        prun.parse_args(["--stage", "map"])  # no truth / model
    with pytest.raises(SystemExit):
        prun.parse_args(["--truth", "T1", "--model", "F2", "--stage", "ti", "--ti-temps", "8"])  # fewer than 12 temperatures
    with pytest.raises(SystemExit):
        prun.parse_args(["--truth", "T1", "--model", "pixel", "--stage", "chain"])


def test_detection_limit_is_read_from_the_milestone_6_file():
    g_det = prun.g_detection(2.0)
    assert g_det == pytest.approx(7.54e-4, rel=1e-3) and prun.g_detection(4.0) == pytest.approx(7.75e-4, rel=1e-2)


def test_the_data_of_a_truth_are_generated_once_and_cached(tmp_path):
    a = tiny_args(tmp_path)
    p1 = prun.build_problem(a)
    assert prun.data_path(a).exists()
    p2 = prun.build_problem(tiny_args(tmp_path, model="Fe"))  # another model, same truth: the cached numbers
    assert p1.sha == p2.sha and np.array_equal(np.asarray(p1.d_obs), np.asarray(p2.d_obs))
    p3 = prun.build_problem(tiny_args(tmp_path, extra=["--force"]))  # regenerated: the same noise key gives the same data to solver round-off
    assert float(np.abs(np.asarray(p3.d_obs) - np.asarray(p1.d_obs)).max() / p1.s) < 1e-6
    assert p1.n_data == 48 and p1.g_true == 0.2 and p1.operating_point["phi_truth"] > 0 and p1.g_det == pytest.approx(7.54e-4, rel=1e-3)
    for name, gap in (("T2", "0.1"), ("T3", "0.1"), ("T4", "0.0"), ("T5", "0.1")):
        p = prun.build_problem(tiny_args(tmp_path, truth=name, gap=gap))
        assert p.chi_true.shape == (16, 16) and p.chi_true.max() > 0
        assert p.g_true == (0.0 if name == "T4" else 0.1)
    assert prun.stage_prepare(a)["n_data"] == 48 and (tmp_path / "T1g0.2_F2" / "problem.json").exists()


# ----------------------------------------------------------------------------------------------
# The pipeline on a tiny problem
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("parametric")
    prun.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(out), *TINY, *SMALL_RUN])
    return out


def test_all_stages_write_the_documented_files(run_dir):
    d = run_dir / "T1g0.2_F2"
    for name in ("problem.json", "map.json", "map.npz", "ppc.json", "summary.json"):
        assert (d / name).exists(), name
    assert sorted(p.name for p in (d / "chains").glob("*.npz")) == ["c0_s0.npz", "c0_s1.npz", "c1_s0.npz", "c1_s1.npz"]
    assert (run_dir / "data" / "T1g0.2.npz").exists()
    seg = np.load(d / "chains" / "c0_s0.npz")
    assert seg["u"].shape == (30, 5) and seg["phi"].shape == (30,) and seg["accept"].shape == (30,) and int(seg["n_warmup"]) == 30
    assert int(np.load(d / "chains" / "c0_s1.npz")["n_warmup"]) == 0 and float(np.load(d / "chains" / "c0_s1.npz")["step"]) == float(seg["step"])  # frozen step
    assert not list(d.rglob("*.tmp*"))


def test_summary_contents(run_dir):
    s = json.loads((run_dir / "T1g0.2_F2" / "summary.json").read_text())
    assert s["case"] == "T1g0.2_F2" and s["model"] == "F2" and s["g_true"] == 0.2
    gap = s["posterior"]["gap"]
    assert gap["ci"][0] < gap["mean"] < gap["ci"][1] and gap["n"] == 120 and "covers_true" in gap and "p_below_g_det" in gap
    assert s["posterior"]["n_chains"] == 2 and s["posterior"]["n_draws_per_chain"] == 60
    m = s["map"]
    assert m["n_data"] == 48 and 0 < m["phi_over_N"] < 3 and m["phi_map_minus_phi_truth"] == pytest.approx(m["phi_map"] - m["phi_truth"])
    assert m["converged"] and set(m["p_hat"]) == {"g", "r", "eps_r", "cx", "cy"}
    assert np.isfinite(s["evidence"]["laplace_gn"]) and abs(s["evidence"]["laplace_gn"] - s["evidence"]["laplace_fd"]) < 1.0 and s["evidence"]["ti"] is None
    assert s["multimodality"]["multimodal"] is False and "converged" in s["convergence"] and set(s["ppc"]) >= {"refit", "draw"}
    assert 0.0 <= s["ppc"]["refit"]["p_value"] <= 1.0


def test_a_rerun_skips_finished_work_and_a_foreign_map_is_refused(run_dir, tmp_path):
    shutil.copytree(run_dir, tmp_path / "copy")
    run_dir = tmp_path / "copy"
    d = run_dir / "T1g0.2_F2"
    before = {p: p.stat().st_mtime_ns for p in list(d.rglob("*.npz")) + list(d.glob("*.json"))}
    prun.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(run_dir), *TINY, *SMALL_RUN])
    assert {p: p.stat().st_mtime_ns for p in before} == before  # nothing was recomputed
    # resume after an interruption: remove the last segment of one chain and the summary; only that segment is recomputed
    seg = d / "chains" / "c1_s1.npz"
    old = np.load(seg)["u"].copy()
    seg.unlink()
    (d / "summary.json").unlink()
    (d / "ppc.json").unlink()
    prun.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(run_dir), *TINY, *SMALL_RUN])
    assert seg.exists() and (d / "summary.json").exists()
    assert (d / "chains" / "c0_s0.npz").stat().st_mtime_ns == before[d / "chains" / "c0_s0.npz"]
    assert np.allclose(np.load(seg)["u"], old, atol=1e-8)  # the segment is a deterministic function of the key and the previous segment (to solver round-off)
    # a map file that no longer matches the segments is refused
    z = dict(np.load(d / "map.npz"))
    z["cov"] = 2.0 * z["cov"]
    np.savez(d / "map.npz", **z)
    with pytest.raises(SystemExit):
        prun.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--stage", "finalize", "--force", "--out-dir", str(run_dir), *TINY, *SMALL_RUN])


@pytest.mark.slow
def test_ti_stage_is_resumable_and_feeds_the_summary(tmp_path):
    base = ["--truth", "T1", "--gap", "0.2", "--out-dir", str(tmp_path), *TINY, *SMALL_RUN]
    prun.main([*base, "--model", "F2"])
    ti = ["--model", "F2", "--stage", "ti", "--ti-temps", "12", "--ti-chains", "2", "--ti-pilot", "15", "--ti-warmup", "20", "--ti-draws", "40"]
    d = tmp_path / "T1g0.2_F2"
    prun.main([*base, *ti])
    assert len(list((d / "ti").glob("t*.npz"))) == 12 and (d / "ti.json").exists()
    res = json.loads((d / "ti.json").read_text())
    assert res["n_temps"] == 12 and res["betas"][1] == pytest.approx((1 / 11) ** 5) and np.isfinite(res["log_z"]) and res["settings"]["ti_temps"] == 12
    # an interrupted run: drop the last two temperatures and the result; the rerun recomputes only those
    mtimes = {p: p.stat().st_mtime_ns for p in (d / "ti").glob("t*.npz")}
    for k in (10, 11):
        prun.ti_path(d, k).unlink()
    prun.ti_json_path(d).unlink()
    prun.main([*base, *ti])
    assert all(p.stat().st_mtime_ns == mtimes[p] for p in mtimes if p.name not in ("t10.npz", "t11.npz"))
    assert json.loads((d / "ti.json").read_text())["n_temps"] == 12
    prun.main([*base, "--model", "F2", "--stage", "finalize", "--force"])
    ev = json.loads((d / "summary.json").read_text())["evidence"]["ti"]
    assert ev["runs"] == 1 and ev["log_z"] == pytest.approx(res["log_z"])
    # a second, independent one-chain run side by side: the summary combines the two and reports their spread
    one = ["--model", "F2", "--stage", "ti", "--ti-temps", "12", "--ti-chains", "1", "--ti-pilot", "15", "--ti-warmup", "20", "--ti-draws", "40", "--ti-tag", "a", "--ti-seed", "7"]
    prun.main([*base, *one])
    assert (d / "ti_a.json").exists() and len(list((d / "ti_a").glob("t*.npz"))) == 12 and len(list((d / "ti").glob("t*.npz"))) == 12
    combo = prun.combine_ti_runs(d)
    other = json.loads((d / "ti_a.json").read_text())
    assert combo["runs"] == 2 and combo["log_z"] == pytest.approx(0.5 * (res["log_z"] + other["log_z"])) and combo["between_run_sd"] == pytest.approx(abs(res["log_z"] - other["log_z"]) / np.sqrt(2))
    assert combo["mc_se"] == pytest.approx(np.hypot(res["mc_se"], other["mc_se"]) / 2) and [r["tag"] for r in combo["per_run"]] == ["", "a"]


@pytest.mark.slow
def test_nested_models_start_from_the_f2_map_and_the_pixel_stage_runs(tmp_path):
    base = ["--truth", "T1", "--gap", "0.2", "--out-dir", str(tmp_path), *TINY, *SMALL_RUN]
    prun.main([*base, "--model", "F2"])
    prun.main([*base, "--model", "F2u", "--stage", "map"])
    cands = json.loads((tmp_path / "T1g0.2_F2u" / "map.json").read_text())["candidates"]
    assert any(c["label"] == "from F2 MAP" for c in cands) and any(c["label"].startswith("image:") for c in cands)
    prun.main([*base, "--model", "pixel"])
    pix = json.loads((tmp_path / "T1g0.2_pixel" / "summary.json").read_text())
    assert set(pix["priors"]) == {"matern", "tv"} and pix["feature"]["band_frac"] == 0.5 and pix["band_frac"] == 0.5
    for rec in pix["priors"].values():
        assert rec["features"]["region"] is not None and 0.0 <= rec["features"]["region"]["p_valley"] <= 1.0
    tables = pcol.collect(tmp_path)
    assert {r["prior"] for r in tables["pixel"]} == {"matern", "tv"}


# ----------------------------------------------------------------------------------------------
# Cost and budget
# ----------------------------------------------------------------------------------------------


def fake_costs(scale=1.0):
    mk = lambda dim, e: {"model": "x", "dim": dim, "compile_s": 5.0, "eval_near_truth_s": {"median": e, "mean": e, "min": e}, "eval_prior_draws_s": {"median": 2 * e, "mean": 2 * e, "min": e},
                         "misfit_only_s": 0.4 * e, "fisher_s": 6.0 * e * scale, "fisher_over_eval": 6.0 * scale}
    return {"F2": mk(5, 0.2), "F2u": mk(7, 0.24), "Fe": mk(6, 0.22)}


def test_budget_counts_follow_the_plan_and_scale_with_the_settings():
    args = prun.parse_args(["--stage", "cost"])
    b = prun.estimate_budget(fake_costs(), args)
    assert b["n_cases"] == 24 and len(b["plan"]) == 8 and b["models"] == ["F2", "F2u", "Fe"]
    pm = b["per_model"]["F2"]
    assert pm["evals_chains"] == 4 * (150 + 500) * 8
    assert pm["seconds_per_case_1thread"] == pytest.approx(pm["evals_per_case"] * 0.2)
    assert pm["ti_evals_per_case"] == 16 * 4 * (2 * 40 + 60 + 150) * 8
    assert b["total_1thread_s"] == pytest.approx(b["posterior_total_1thread_s"] + b["ti_total_1thread_s"])
    # the wall clock: 5 processes count as 5, 10 as 5 + 0.24 * 5
    assert b["wall_hours"]["5"]["effective_parallelism"] == 5.0 and b["wall_hours"]["10"]["effective_parallelism"] == pytest.approx(6.2)
    assert b["wall_hours"]["5"]["total_h"] == pytest.approx(b["total_1thread_s"] / 5 / 3600)
    assert b["ti_cases"] == ["F2", "Fe"] and b["ti_total_1thread_s"] == pytest.approx(pm["ti_seconds_per_case_1thread"] + b["per_model"]["Fe"]["ti_seconds_per_case_1thread"])
    # doubling the draws and the chains doubles the chain evaluations
    args2 = prun.parse_args(["--stage", "cost", "--chains", "8", "--segments", "10", "--warmup", "300"])
    assert prun.estimate_budget(fake_costs(), args2)["per_model"]["F2"]["evals_chains"] == 8 * (300 + 1000) * 8
    # a costlier Fisher matrix only changes the MAP and the predictive-check parts
    b2 = prun.estimate_budget(fake_costs(scale=2.0), args)
    assert b2["per_model"]["F2"]["evals_chains"] == pm["evals_chains"] and b2["per_model"]["F2"]["evals_map"] > pm["evals_map"]


def test_cost_of_the_three_models_is_measured_on_a_tiny_grid(tmp_path):
    args = prun.parse_args(["--stage", "cost", "--out-dir", str(tmp_path), "--cost-evals", "3", *TINY])
    out = prun.stage_cost(args)
    assert set(out["costs"]) == {"F2", "F2u", "Fe"} and [out["costs"][m]["dim"] for m in ("F2", "F2u", "Fe")] == [5, 7, 6]
    for c in out["costs"].values():
        assert c["eval_near_truth_s"]["mean"] > 0 and c["misfit_only_s"] > 0 and c["fisher_s"] > 0 and c["compile_s"] > 0
    assert (tmp_path / "cost.json").exists() and out["budget"]["total_1thread_s"] > 0
    assert prun.stage_cost(args)["budget"]["total_1thread_s"] == out["budget"]["total_1thread_s"]  # cached


# ----------------------------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------------------------


def fake_summary(model, g_mean, ci, covers, phi_over_n, laplace, ti=None, ppc_p=0.5, g_true=0.05, converged=True, multimodal=False):
    gap = None if model == "Fe" else {"mean": g_mean, "sd": 1e-3, "ci": ci, "covers_true": covers, "z_true": -0.5, "p_below_g_det": 0.0}
    return {"case": "x", "model": model, "g_true": g_true, "g_det": 7.5e-4,
            "map": {"phi_map": 2400 * phi_over_n, "n_data": 2400, "phi_over_N": phi_over_n, "phi_truth": 2500.0, "phi_map_minus_phi_truth": 2400 * phi_over_n - 2500.0, "converged": True},
            "evidence": {"laplace_gn": laplace, "laplace_fd": laplace + 0.1, "ti": ti}, "posterior": {"gap": gap, "phi_over_N": phi_over_n},
            "convergence": {"converged": converged, "max_rhat": 1.004, "worst_ess_bulk": 400.0},
            "multimodality": {"multimodal": multimodal, "reasons": ["distinct mode"] if multimodal else []},
            "ppc": {"refit": {"p_value": ppc_p, "z": 0.1}, "draw": {"p_value": 0.5}}}


@pytest.fixture()
def fake_results(tmp_path):
    ti = lambda z: {"log_z": z, "mc_se": 0.3, "converged": True, "discretization": {"richardson_error_estimate": 0.05}}
    cases = {"T1g0.05": {"F2": fake_summary("F2", 0.0501, [0.0496, 0.0506], True, 1.02, 1000.0, ti(1000.5)),
                         "F2u": fake_summary("F2u", 0.0502, [0.0495, 0.0508], True, 1.01, 996.0, ti(996.2)),
                         "Fe": fake_summary("Fe", None, None, None, 9.0, 600.0, ppc_p=0.0)},
             "T4": {"F2": fake_summary("F2", 0.004, [0.003, 0.005], False, 1.3, 800.0, g_true=0.0, ppc_p=0.01, multimodal=True)}}
    for label, models in cases.items():
        for model, s in models.items():
            d = tmp_path / f"{label}_{model}"
            d.mkdir()
            (d / "summary.json").write_text(json.dumps(s))
    pix = {"case": "x", "g_true": 0.05, "priors": {"matern": {"lam": None, "phi_over_N": 1.0, "map_ok": True, "truth": {"region": {"holds": True}}, "features": {
        "region": {"p_valley": 0.12}, "region_status": "ok", "point": {"p_valley": 0.3}, "geometry": {"strip_width_pixels": 2.1}}},
        "tv": {"lam": 0.5, "phi_over_N": 1.0, "map_ok": True, "truth": {"region": {"holds": False}}, "features": {
            "region": {"p_valley": 0.9}, "region_status": "ok", "point": {"p_valley": 0.8}, "geometry": {"strip_width_pixels": 2.1}}}}}
    (tmp_path / "T1g0.05_pixel").mkdir()
    (tmp_path / "T1g0.05_pixel" / "summary.json").write_text(json.dumps(pix))
    return tmp_path


def test_collect_builds_the_tables_bayes_factors_and_figures(fake_results):
    tables = pcol.collect(fake_results)
    for name in ("tables.md", "tables.json", "parametric_gap.png", "parametric_fit.png", "parametric_evidence.png"):
        assert (fake_results / name).exists(), name
    gap = {(r["truth"], r["model"]): r for r in tables["gap"]}
    assert gap[("T1g0.05", "F2")]["covers_true"] is True and gap[("T4", "F2")]["covers_true"] is False and gap[("T4", "F2")]["multimodal"] is True
    assert gap[("T1g0.05", "Fe")]["g_mean"] is None
    bf = {(r["truth"], r["other"]): r for r in tables["bayes_factors"]}
    assert bf[("T1g0.05", "F2u")]["laplace_gn"] == pytest.approx(4.0) and bf[("T1g0.05", "F2u")]["ti"] == pytest.approx(4.3)
    assert bf[("T1g0.05", "Fe")]["laplace_gn"] == pytest.approx(400.0) and bf[("T1g0.05", "Fe")]["ti"] is None
    assert ("T4", "F2u") not in bf
    # the pixel posterior whose pixelized truth fails the feature reports no region probability
    px = {r["prior"]: r for r in tables["pixel"]}
    assert px["matern"]["p_valley_region"] == 0.12 and px["tv"]["p_valley_region"] is None
    md = (fake_results / "tables.md").read_text()
    assert "## 1. Posterior of the gap" in md and "## 3. Log evidence and Bayes factors" in md and "F2 vs Fe" in md and "T1g0.05" in md
    assert [c for c in pcol.load_cases(fake_results)] == ["T1g0.05", "T4"]
    with pytest.raises(SystemExit):
        pcol.collect(fake_results / "nowhere")


# ----------------------------------------------------------------------------------------------
# Job files
# ----------------------------------------------------------------------------------------------


def test_job_files_cover_the_plan_with_unique_names_and_the_markers_the_scripts_write():
    import batch_run

    plan = prun.PLAN_TRUTHS
    assert tuple(mjobs.PLAN) == tuple(plan)
    counts = {"map-f2": 8, "map-rest": 16, "chain": 8 * 3 * 4, "finalize": 24, "pixel": 8, "ti": 8}
    names = set()
    for batch, n in counts.items():
        jobs = mjobs.jobs_for(batch)
        assert len(jobs) == n, batch
        for j in jobs:
            assert batch_run.NAME_RE.fullmatch(j["name"]) and j["name"] not in names and j["threads"] == 1 and j["cmd"][0] == "scripts/parametric_run.py"
            assert set(j) <= batch_run.JOB_KEYS
            names.add(j["name"])
    chain = mjobs.jobs_for("chain", chains=3, segments=2, extra=("--segment-draws", "50"))
    first = chain[0]
    assert first["done_marker"] == "results/parametric/T1g0.05_F2/chains/c0_s1.npz" and first["cmd"][-2:] == ["--segment-draws", "50"] and "--chain-id" in first["cmd"]
    assert mjobs.jobs_for("pixel")[6]["done_marker"] == "results/parametric/T4_pixel/summary.json"
    # the command lines parse with the run script's own parser and name the case whose marker they promise
    for batch in counts:
        for j in mjobs.jobs_for(batch):
            a = prun.parse_args(j["cmd"][1:])
            marker = j["done_marker"].split("/")[2]
            assert marker == prun.case_name(a)
    ti = mjobs.jobs_for("ti", ti_runs=2)
    assert [j["name"] for j in ti] == ["m7_ti_T1g0.05_F2_r0", "m7_ti_T1g0.05_F2_r1", "m7_ti_T1g0.05_Fe_r0", "m7_ti_T1g0.05_Fe_r1"]
    assert ti[1]["done_marker"] == "results/parametric/T1g0.05_F2/ti_r1.json"
    assert ti[1]["cmd"][ti[1]["cmd"].index("--ti-tag"):][:6] == ["--ti-tag", "r1", "--ti-seed", "3", "--ti-chains", "1"]
    with pytest.raises(ValueError):
        mjobs.jobs_for("nope")


def test_job_files_can_be_restricted_to_a_plan_and_to_models():
    plan = [("T1", 0.05), ("T4", 0.0)]
    assert [j["name"] for j in mjobs.jobs_for("map-f2", plan=plan)] == ["m7_map_T1g0.05_F2", "m7_map_T4_F2"]
    assert [j["name"] for j in mjobs.jobs_for("map-rest", plan=plan, models=("F2", "Fe"))] == ["m7_map_T1g0.05_Fe", "m7_map_T4_Fe"]
    assert len(mjobs.jobs_for("chain", plan=plan, models=("F2", "Fe"))) == 2 * 2 * 4
    assert [j["name"] for j in mjobs.jobs_for("finalize", plan=plan, models=("Fe",))] == ["m7_finalize_T1g0.05_Fe", "m7_finalize_T4_Fe"]
    assert len(mjobs.jobs_for("pixel", plan=plan)) == 2


FE_NAMES = ["cx", "cy", "a", "b", "psi", "eps_r"]


def fake_map(model, p_hat, sd_u0, phi_over_n, log_z, fd=None, converged=True):
    return {"case": "x", "model": {"name": model}, "param_names": FE_NAMES if model == "Fe" else None, "p_hat": p_hat, "phi_map": 2400 * phi_over_n, "phi_over_N": phi_over_n, "phi_truth": 2500.0, "map_converged": converged,
            "laplace": {"sd_u": [sd_u0, 0.1, 0.1]}, "evidence": {"laplace_gn": log_z, "laplace_fd": fd}}


def test_laplace_table_needs_only_the_map_stage_and_propagates_the_gap_interval(tmp_path):
    g, sd_u = 0.05, 0.006
    for model, rec in (("F2", fake_map("F2", [g, 0.2], sd_u, 1.02, 1000.0, 1000.2)), ("F2u", fake_map("F2u", [0.051, 0.2], sd_u, 1.01, 990.0, 990.1)),
                       ("Fe", fake_map("Fe", [0.0, 0.0, 0.4, 0.2, 0.001, 1.9], 0.1, 9.0, 100.0, 100.5))):
        d = tmp_path / f"T1g0.05_{model}"
        d.mkdir()
        (d / "map.json").write_text(json.dumps(rec))
    d = tmp_path / "T4_F2"
    d.mkdir()
    (d / "map.json").write_text(json.dumps(fake_map("F2", [2e-5, 0.2], 0.5, 4.0, 800.0, None, converged=False)))
    assert [c for c in pcol.load_maps(tmp_path)] == ["T1g0.05", "T4"] and pcol.true_gap_of_label("T4") == 0.0 and pcol.true_gap_of_label("T5") == 0.1
    assert pcol.true_gap_of_label("T2g0.1") == 0.1
    tables = pcol.collect(tmp_path)  # no summary.json at all: only the Laplace section
    rows = {(r["truth"], r["model"]): r for r in tables["laplace"]}
    sd = sd_u * g * (1 - g / 0.4)
    assert rows[("T1g0.05", "F2")]["g_sd"] == pytest.approx(sd) and rows[("T1g0.05", "F2")]["covers_true"] is True and rows[("T1g0.05", "F2")]["z_true"] == pytest.approx(0.0)
    assert rows[("T1g0.05", "F2u")]["z_true"] == pytest.approx((0.05 - 0.051) / (sd_u * 0.051 * (1 - 0.051 / 0.4)))
    assert rows[("T1g0.05", "Fe")]["g_hat"] is None and rows[("T4", "F2")]["map_converged"] is False
    assert rows[("T1g0.05", "F2")]["p_below_g_det"] is not None and rows[("T1g0.05", "F2")]["p_below_g_det"] < 1e-6
    bf = {(r["truth"], r["other"]): r for r in tables["laplace_bayes_factors"]}
    assert bf[("T1g0.05", "F2u")]["laplace_gn"] == pytest.approx(10.0) and bf[("T1g0.05", "Fe")]["laplace_fd"] == pytest.approx(899.7) and ("T4", "F2u") not in bf
    assert "## 5. Laplace fits of every case with a MAP" in (tmp_path / "tables.md").read_text()


def test_ellipticity_ladder_level_zero_is_two_circles_and_a_level_is_resumable(tmp_path):
    scene = pell.make_scene(0.2, 0.15)
    ref = prun.el.make_truth("T2", g=0.2)
    assert np.allclose(scene.ellipses, ref.ellipses) and pell.level_label(0.1, 0.02) == "E0.02g0.1"
    assert np.allclose(pell.make_scene(0.2, 0.0).ellipses[:, 2], pell.make_scene(0.2, 0.0).ellipses[:, 3])
    args = ["--gap", "0.2", "--out-dir", str(tmp_path), *TINY, "--ppc-reps", "4", "--n-prior-starts", "1"]
    pell.main([*args, "--level", "0.0", "0.15"])
    r0, r1 = (json.loads((tmp_path / f"E{lv}g0.2.json").read_text()) for lv in ("0", "0.15"))
    assert r0["level"] == 0.0 and r1["semi_axes"][1] == pytest.approx(0.23) and r0["n_data"] == 48
    for r in (r0, r1):
        assert 0.0 <= r["ppc_refit"]["p_value"] <= 1.0 and r["g_sd"] > 0 and r["g_ci"][0] <= r["g_hat"] <= r["g_ci"][1]
    assert r1["phi_over_N"] > r0["phi_over_N"]  # the wrong family fits worse than the right one (same noise realization)
    mtime = (tmp_path / "E0g0.2.json").stat().st_mtime
    pell.main([*args, "--level", "0.0"])
    assert (tmp_path / "E0g0.2.json").stat().st_mtime == mtime
    out = pell.main([*args, "--collect"])
    assert [r["level"] for r in out["ladder"]] == [0.0, 0.15] and (tmp_path / "ellipticity_g0.2.md").exists() and (tmp_path / "ellipticity_g0.2.png").exists()
    assert out["calibration"] == []  # one seed per level: no calibration table


def test_sweeps_seeds_give_other_noise_and_the_calibration_summary(tmp_path):
    args = ["--gap", "0.2", "--out-dir", str(tmp_path), *TINY, "--ppc-reps", "0", "--n-prior-starts", "1"]
    pell.main([*args, "--level", "0.0", "--seeds", "0", "1", "2", "3"])
    assert pell.level_label(0.2, 0.0, 3) == "E0g0.2s3" and pell.level_label(0.2, 0.0, 0) == "E0g0.2"
    recs = [json.loads((tmp_path / f"E0g0.2{'' if s == 0 else f's{s}'}.json").read_text()) for s in range(4)]
    assert all(r["ppc_refit"] is None and r["seed"] == s for s, r in enumerate(recs))
    assert len({round(r["g_hat"], 7) for r in recs}) == 4  # the noise differs
    cal = pell.main([*args, "--collect"])["calibration"]
    assert len(cal) == 1 and cal[0]["n_seeds"] == 4 and cal[0]["z_sd"] > 0 and 0.0 <= cal[0]["coverage_90"] <= 1.0
    z = np.array([r["z_true"] for r in recs])
    assert cal[0]["z_mean"] == pytest.approx(z.mean()) and cal[0]["z_sd"] == pytest.approx(z.std(ddof=1)) and (tmp_path / "calibration_g0.2.md").exists()


# ----------------------------------------------------------------------------------------------
# Profile marginal of the gap (boundary cases)
# ----------------------------------------------------------------------------------------------


def test_profile_marginal_of_a_gaussian_toy_is_the_exact_marginal_and_its_integral_the_evidence():
    from bayes_scatter import shapepost as sp

    rng = np.random.default_rng(3)
    toy = sp.GaussianToy(rng.standard_normal((9, 3)), rng.standard_normal(9), noise_sd=0.7, prior_sd=2.0)
    mean, cov = toy.exact_posterior()
    grid = np.linspace(mean[0] - 7 * np.sqrt(cov[0, 0]), mean[0] + 7 * np.sqrt(cov[0, 0]), 61)
    prof = pprof.profile_marginal(toy, 0, mean, grid)
    st = pprof.marginal_statistics(prof["grid"], prof["log_m"])
    assert st["mean"] == pytest.approx(mean[0], abs=1e-6) and st["sd"] == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-6)
    assert st["log_norm"] == pytest.approx(toy.exact_log_evidence(), abs=1e-6)  # the marginal integrates to the evidence
    assert st["tail_mass_low"] < 1e-6 and st["tail_mass_high"] < 1e-6 and all(prof["converged"])
    # the gap transform: a logistic variable with a Gaussian u_g law; P(g < g0) is the Gaussian CDF of logit(g0 / 0.4)
    from scipy.stats import norm

    ug = np.linspace(-12, -4, 400)
    gm = pprof.gap_marginal(ug, norm.logpdf(ug, loc=-8.0, scale=0.9), 0.4, g_true=0.4 / (1 + np.exp(8.0)), g_det=7.5e-4)
    z = (np.log(7.5e-4 / (0.4 - 7.5e-4)) + 8.0) / 0.9
    assert gm["p_below_g_det"] == pytest.approx(norm.cdf(z), abs=2e-4) and gm["p_below_g_true"] == pytest.approx(0.5, abs=2e-4) and gm["covers_true"] is True


def test_fixed_coordinate_matches_the_full_problem_and_the_profile_of_a_tiny_f2_case(tmp_path):
    import jax.numpy as jnp

    base = ["--truth", "T1", "--gap", "0.2", "--out-dir", str(tmp_path), *TINY, *SMALL_RUN, "--model", "F2"]
    prun.main([*base, "--stage", "map"])
    args = prun.parse_args(base)
    rec, z = prun.load_map(args)
    prob = prun.build_problem(args)
    post = prun.make_posterior(args, prob)
    u_hat = np.asarray(z["u_hat"])
    sub = pprof.FixedCoordinate(post, 0, float(u_hat[0]))
    v = u_hat[1:] + 0.01
    phi, g, _ = sub.phi_and_grad(jnp.asarray(v), sub.initial_aux(), sub.data)
    phi_full, g_full, _ = post.phi_and_grad(jnp.asarray(np.concatenate([u_hat[:1], v])), post.initial_aux(), post.data)
    assert float(phi) == pytest.approx(float(phi_full), rel=1e-12) and np.allclose(np.asarray(g), np.asarray(g_full)[1:], rtol=1e-9, atol=1e-9)
    assert sub.fisher(v).shape == (4, 4) and np.allclose(sub.full(v)[1:], v) and sub.full(v)[0] == u_hat[0]
    out = pprof.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(tmp_path), *TINY, "--around-map", "5", "--grid-n", "11"])
    gap = out["gap"]
    sd_laplace = rec["laplace"]["sd_u"][0] * rec["p_hat"][0] * (1 - rec["p_hat"][0] / 0.4)
    assert gap["mean"] == pytest.approx(rec["p_hat"][0], abs=0.5 * sd_laplace) and gap["sd"] == pytest.approx(sd_laplace, rel=0.15)
    assert out["all_converged"] and gap["tail_mass_low"] < 1e-3 and (tmp_path / "T1g0.2_F2" / "profile.json").exists()
    assert pprof.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(tmp_path), *TINY])["case"] == "T1g0.2_F2"  # cached


def test_collect_lists_profile_marginals_next_to_the_chains(fake_results):
    d = fake_results / "T4_F2"
    d.mkdir(exist_ok=True)
    (fake_results / "T4_F2" / "profile.json").write_text(json.dumps({"g_true": 0.0, "phi_profile_min": 9487.8, "all_converged": True, "gap": {
        "mean": 2.3e-6, "sd": 2.3e-6, "ci90": [1.2e-7, 7.9e-6], "covers_true": False, "p_below_g_det": 1.0, "log_norm": 28062.8, "tail_mass_low": 4e-6, "tail_mass_high": 0.0}}))
    d = fake_results / "T1g0.05_F2"
    (d / "profile.json").write_text(json.dumps({"g_true": 0.05, "phi_profile_min": 2436.6, "all_converged": True, "gap": {
        "mean": 0.0501, "sd": 1e-3, "ci90": [0.0496, 0.0506], "covers_true": True, "p_below_g_det": 0.0, "log_norm": 1000.1, "tail_mass_low": 1e-8, "tail_mass_high": 1e-8}}))
    tables = pcol.collect(fake_results)
    rows = {(r["truth"], r["model"]): r for r in tables["profile"]}
    assert rows[("T4", "F2")]["mean"] == 2.3e-6 and rows[("T4", "F2")]["chain_mean"] == pytest.approx(0.004)  # the fixture's T4 case also has a (fake) summary: paired
    assert rows[("T1g0.05", "F2")]["chain_mean"] == pytest.approx(0.0501) and rows[("T1g0.05", "F2")]["chain_sd"] == pytest.approx(1e-3)
    assert "## 6. Profile marginal of the gap" in (fake_results / "tables.md").read_text()


# ----------------------------------------------------------------------------------------------
# Review follow-up (DESIGN 14.7)
# ----------------------------------------------------------------------------------------------

import parametric_discretization as pdisc  # noqa: E402
import parametric_evidence_is as pis  # noqa: E402
import parametric_snr as psnr  # noqa: E402
import parametric_ti_offline as pti  # noqa: E402

TINY_GEOM = ["--n", "16", "--n-tx", "6", "--n-rx", "8", "--tol", "1e-11", "--side", "1.5"]


@pytest.fixture(scope="module")
def tiny_map_dir(tmp_path_factory):
    """Data (cached) and the F2 MAP of a tiny T1 case (gap 0.2, 40 dB, n = 16), the inputs of the review follow-up scripts."""
    d = tmp_path_factory.mktemp("tiny_map")
    prun.main(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--out-dir", str(d), *TINY, *SMALL_RUN, "--stage", "map"])
    return d


def exponential_gap_log_marginal(u, lam, g_max=0.4):
    """log m(u_g) for a gap that is exponential with rate ``lam``: m_g(g) = lam exp(-lam g), m(u) = m_g(g(u)) dg/du with g = g_max sigmoid(u)."""
    g = g_max / (1.0 + np.exp(-u))
    return np.log(lam) - lam * g + np.log(g * (1.0 - g / g_max))


def test_gap_marginal_interpolates_the_cdf_in_u_so_a_coarse_grid_still_gives_the_quantiles():
    """The T4 gap is nearly exponential (profile misfit linear in g, slope 4.4e5). On the grid of the first profile run (spacing 0.5 in u_g, a factor 1.65 in g) interpolating the CDF
    linearly in g put the 95 % point 15 % too high; interpolating in u_g (monotone cubic) is within 2 %. Also the mean, P(g < g_det) and the mass inside the truncation box."""
    lam = 4.36e5
    u = np.arange(-24.0, -3.0 + 1e-9, 0.5)
    lm = exponential_gap_log_marginal(u, lam)
    g_det = 7.54e-4
    out = pprof.gap_marginal(u, lm, 0.4, g_true=0.0, g_det=g_det)
    exact = lambda q: -np.log(1.0 - q) / lam  # noqa: E731
    assert out["ci90"][0] == pytest.approx(exact(0.05), rel=0.005) and out["ci90"][1] == pytest.approx(exact(0.95), rel=0.005)
    assert out["median"] == pytest.approx(exact(0.5), rel=0.005) and out["mean"] == pytest.approx(1.0 / lam, rel=0.005) and out["sd"] == pytest.approx(1.0 / lam, rel=0.005)
    assert out["p_below_g_det"] == pytest.approx(1.0 - np.exp(-lam * g_det), abs=1e-6) and out["p_below_g_true"] == 0.0 and out["covers_true"] is False
    # what the first version computed on this grid (trapezoid CDF at the grid points, interpolated linearly in g): the 95 % point is more than 8 % too high
    w = np.exp(lm - lm.max())
    cdf_old = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(u))])
    cdf_old /= cdf_old[-1]
    old_upper = float(np.interp(0.95, cdf_old, 0.4 / (1.0 + np.exp(-u))))
    assert abs(old_upper / exact(0.95) - 1.0) > 0.08 > abs(out["ci90"][1] / exact(0.95) - 1.0)
    box = pprof.truncation_mass(u, lm)
    g_box = 0.4 / (1.0 + np.exp(10.0))
    assert box["mass_inside"] == pytest.approx(np.exp(-lam * g_box), rel=0.01) and box["delta_log_z"] == pytest.approx(-lam * g_box, abs=0.01) and box["grid_covers_box"] is False
    # an interior posterior loses nothing to the box
    inner = pprof.truncation_mass(np.linspace(-14, 6, 81), norm_logpdf(np.linspace(-14, 6, 81), -4.0, 1.0))
    assert inner["mass_inside"] == pytest.approx(1.0, abs=1e-6) and abs(inner["delta_log_z"]) < 1e-6 and inner["grid_covers_box"] is False


def norm_logpdf(x, loc, scale):
    return -0.5 * ((x - loc) / scale) ** 2 - np.log(scale * np.sqrt(2.0 * np.pi))


def test_profile_marginal_resumes_from_its_checkpoint_and_solves_only_the_missing_points(tmp_path, monkeypatch):
    from bayes_scatter import shapepost as sp

    rng = np.random.default_rng(3)
    toy = sp.GaussianToy(rng.standard_normal((9, 3)), rng.standard_normal(9), noise_sd=0.7, prior_sd=2.0)
    mean, cov = toy.exact_posterior()
    grid = np.linspace(mean[0] - 4 * np.sqrt(cov[0, 0]), mean[0] + 4 * np.sqrt(cov[0, 0]), 9)
    ck = tmp_path / "partial.json"
    full = pprof.profile_marginal(toy, 0, mean, grid, checkpoint=ck)
    saved = json.loads(ck.read_text())
    for k in (2, 5, 6):  # an interrupted run: three points missing
        for key in ("log_m", "phi", "log_prior", "logdet", "converged", "v", "nit"):
            saved[key][k] = None
    ck.write_text(json.dumps(saved))
    calls = []
    real = sp.newton_map
    monkeypatch.setattr(sp, "newton_map", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    again = pprof.profile_marginal(toy, 0, mean, grid, checkpoint=ck)
    assert len(calls) == 3 and np.allclose(again["log_m"], full["log_m"], rtol=1e-9, atol=1e-6)
    calls.clear()
    pprof.profile_marginal(toy, 0, mean, grid, checkpoint=ck)
    assert calls == []  # complete: nothing to solve
    # a different grid ignores the checkpoint
    pprof.profile_marginal(toy, 0, mean, grid[:5], checkpoint=ck)
    assert len(calls) == 5


def test_snr_units_reproduce_the_cached_noise_scale_it_and_are_resumable(tiny_map_dir, tmp_path):
    out = tmp_path / "snr"
    base = ["--truth", "T1", "--gap", "0.2", "--results", str(tiny_map_dir), "--out-dir", str(out), *TINY_GEOM]
    psnr.main([*base, "--snr", "40", "30", "--seeds", "0", "1"])
    z = np.load(tiny_map_dir / "data" / "T1g0.2.npz")
    cached_noise_only = float(np.sum(np.abs(z["d_obs"] - z["d_clean"]) ** 2) / float(z["s"]) ** 2)
    r40, r30 = (json.loads((out / f"T1g0.2_F2_snr{snr}_s0.json").read_text()) for snr in (40, 30))
    assert r40["noise_std"] == pytest.approx(float(z["s"]), rel=1e-12) and r40["noise_only_phi"] == pytest.approx(cached_noise_only, rel=1e-10)  # seed 0 at the cached SNR is the cached data
    assert r30["noise_std"] == pytest.approx(r40["noise_std"] * 10 ** (10 / 20), rel=1e-9) and r30["noise_only_phi"] == pytest.approx(r40["noise_only_phi"], rel=1e-9)  # a scaled copy
    r40b = json.loads((out / "T1g0.2_F2_snr40_s1.json").read_text())
    assert r40b["noise_only_phi"] != r40["noise_only_phi"]  # another key, another realization
    from bayes_scatter import shapepost as sp

    for r in (r40, r30, r40b):
        flag = sp.misfit_p_value(r["phi_map"], r["n_data"], 5)
        assert r["p_null"] == pytest.approx(flag["p_value"]) and r["z_phi"] == pytest.approx(flag["z"]) and r["delta_phi"] == pytest.approx(r["phi_map"] - r["noise_only_phi"])
        assert r["map_converged"] and r["g_sd"] > 0 and r["g_ci"][0] <= r["g_hat"] <= r["g_ci"][1] and r["z_true"] == pytest.approx((0.2 - r["g_hat"]) / r["g_sd"])
        assert r["model"] == "F2" and r["n_data"] == 48
    assert r30["g_sd"] > 3.0 * r40["g_sd"] > 0  # 10 dB less SNR: about 3.16 times the sd
    mtime = (out / "T1g0.2_F2_snr40_s0.json").stat().st_mtime
    psnr.main([*base, "--snr", "40", "--seeds", "0"])
    assert (out / "T1g0.2_F2_snr40_s0.json").stat().st_mtime == mtime
    res = psnr.main(["--collect", "--out-dir", str(out)])
    groups = {(g["snr"], g["n_seeds"]) for g in res["summary"]}
    assert groups == {(40.0, 2), (30.0, 2)} and len(res["units"]) == 4
    md = (out / "snr_summary.md").read_text()
    assert "One realization" in md and "Typical draws" in md and (out / "snr_summary.json").exists()
    g40 = next(g for g in res["summary"] if g["snr"] == 40.0)
    chk = psnr.noise_null_check(n_data=48, n_keys=400, first=20, out_dir=out)  # the generator has the Gamma(N, 1) law of the noise-only misfit
    assert (out / "noise_null_check.json").exists() and chk["mean"] == pytest.approx(48.0, abs=4 * np.sqrt(48 / 400)) and chk["sd"] == pytest.approx(np.sqrt(48), rel=0.15) and chk["ks_p_value"] > 0.01
    assert g40["frac_flagged_p05"] in (0.0, 0.5, 1.0) and (g40["c"] is None) == (g40["z_abs_mean"] <= 3.0)  # c only where the mean |z| is above 3


def test_snr_summary_gives_c_only_for_a_wrong_family_and_counts_flags():
    mk = lambda seed, z, dphi, p: {"label": "X", "model": "F2", "snr": 10.0, "seed": seed, "g_true": 0.1, "g_hat": 0.07, "g_sd": 0.003, "z_true": z, "delta_phi": dphi, "p_null": p,  # noqa: E731
                                   "phi_over_N": 1.05, "z_phi": 1.0, "map_converged": True, "ppc_refit": None}
    units = [mk(0, 10.0, 90.0, 0.003), mk(1, 12.0, 130.0, 0.0001), mk(2, 8.0, 60.0, 0.2)]
    row = psnr.summarize(units)[0]
    assert row["n_seeds"] == 3 and row["frac_flagged_p05"] == pytest.approx(2 / 3) and row["frac_flagged_3sd"] == pytest.approx(1 / 3)
    assert row["c"] == pytest.approx(np.mean([90.0, 130.0, 60.0]) / np.mean([100.0, 144.0, 64.0])) and row["frac_interval_excludes_3sd"] == 1.0
    assert row["p_null_median"] == 0.003 and row["z_abs_mean"] == pytest.approx(10.0)


def test_importance_sampling_evidence_script_reproduces_the_laplace_evidence_of_a_tiny_case(tiny_map_dir):
    args = ["--truth", "T1", "--gap", "0.2", "--model", "F2", "--draws", "60", "--out-dir", str(tiny_map_dir), *TINY_GEOM]
    out = pis.main(args)
    assert (tiny_map_dir / "T1g0.2_F2" / "is_evidence.json").exists()
    for key in ("log_z", "se", "weight_ess", "max_weight", "n_draws", "laplace_gn", "is_minus_laplace_gn", "df", "scale"):
        assert np.isfinite(out[key]), key
    assert out["n_draws"] == 60 and out["weight_ess"] > 10 and out["se"] > 0
    assert abs(out["is_minus_laplace_gn"]) < 6.0 * out["se"] + 0.3  # the posterior is close to Gaussian here (40 dB)
    mtime = (tiny_map_dir / "T1g0.2_F2" / "is_evidence.json").stat().st_mtime
    pis.main(args)
    assert (tiny_map_dir / "T1g0.2_F2" / "is_evidence.json").stat().st_mtime == mtime
    with pytest.raises(SystemExit):
        pis.main(["--truth", "T2", "--gap", "0.2", "--model", "F2", "--out-dir", str(tiny_map_dir), *TINY_GEOM])


def test_discretization_fit_vanishes_on_the_inference_grid_and_analysis_removes_the_offset(tiny_map_dir, tmp_path):
    out = tmp_path / "disc"
    pdisc.main(["fit", "--gap", "0.2", "--data-n", "16", "32", "--n", "16", "--n-tx", "6", "--n-rx", "8", "--tol", "1e-11", "--results", str(tiny_map_dir), "--out-dir", str(out)])
    a, b = (json.loads(pdisc.unit_path(out, 0.2, n).read_text()) for n in (16, 32))
    assert abs(a["shift_sd"]) < 0.02 and a["phi_map"] < 1e-3 and a["model_error_energy_at_truth"] < 1e-6  # the inverse crime: no offset, no misfit
    assert b["cached_clean_data_max_rel_diff"] < 1e-9  # the n = 32 data are the cached noise-free data of the experiment
    assert abs(b["shift_sd"]) > abs(a["shift_sd"]) and b["model_error_energy_at_truth"] > 1e-3 and b["converged"]
    mtime = pdisc.unit_path(out, 0.2, 16).stat().st_mtime
    pdisc.main(["fit", "--gap", "0.2", "--data-n", "16", "--n", "16", "--n-tx", "6", "--n-rx", "8", "--results", str(tiny_map_dir), "--out-dir", str(out)])
    assert pdisc.unit_path(out, 0.2, 16).stat().st_mtime == mtime
    assert pdisc.extrapolate_shift([0.0, 1.0, 1.25]) == pytest.approx((1.3333333, 0.25), rel=1e-5)  # geometric: 1.25 + 0.25 * 0.25 / 0.75 = 1.3333
    # analysis on fabricated unit and seed files: a constant shift between the data grids (the same noise) and a mean z that is the offset plus noise
    res_dir = tmp_path / "res"
    (res_dir / "ellipticity").mkdir(parents=True)
    (res_dir / "ellipticity_crime").mkdir()
    rng = np.random.default_rng(1)
    offs = {0.05: 2.8e-4, 0.1: 3.0e-4}
    an_out = tmp_path / "an"
    an_out.mkdir()
    for gap, off in offs.items():
        sd = 2.7e-4
        for n, mult in ((64, 0.0), (128, 1.0), (256, 1.2)):
            (an_out / f"disc_g{gap:g}_n{n}.json").write_text(json.dumps({"gap": gap, "data_n": n, "g_star_minus_true": off * mult, "shift_sd": off * mult / sd, "laplace_sd": sd, "phi_map": 0.1,
                                                                        "model_error_energy_at_truth": 8.0}))
        noise = rng.standard_normal(12)
        for k in range(12):
            g_hat = gap + off + 0.9 * sd * noise[k]
            rec = {"level": 0.0, "seed": k, "g_true": gap, "g_hat": g_hat, "g_sd": sd, "z_true": (gap - g_hat) / sd}
            (res_dir / "ellipticity" / f"E0g{gap:g}{'' if k == 0 else f's{k}'}.json").write_text(json.dumps(rec))
            if gap == 0.05 and 1 <= k <= 8:
                crime = dict(rec, g_hat=g_hat - off, z_true=(gap - g_hat + off) / sd)
                (res_dir / "ellipticity_crime" / f"E0g0.05s{k}.json").write_text(json.dumps(crime))
    res = pdisc.main(["analyze", "--results", str(res_dir), "--out-dir", str(an_out)])
    assert res["paired_crime"]["dz_mean"] == pytest.approx(off_in_sd(2.8e-4, 2.7e-4), abs=1e-9) and res["paired_crime"]["dz_sd"] < 1e-9
    cal = res["gaps"]["0.05"]["calibration"]
    assert cal["z_mean"] < -0.5 and abs(cal["z_corrected_mean"]) < cal["z_mean"] * -1.0 and cal["coverage_corrected"] >= cal["coverage"] and cal["z_corrected_sd"] == pytest.approx(cal["z_sd"], rel=1e-6)
    assert res["gaps"]["0.05"]["extrapolated_sd"] > 1.2 * 2.8e-4 / 2.7e-4 and res["z_correlation_between_gaps"]["n_seeds"] == 12
    rows = {r["snr"]: r for r in res["offset_vs_snr_g0.05"]}
    assert rows[10]["offset_sd"] == pytest.approx(rows[30]["offset_sd"] * 10 ** (-20 / 20)) and rows[40]["offset_sd"] == pytest.approx(rows[30]["offset_sd"] * 10 ** (10 / 20))
    assert "constant shift is the signature" in (an_out / "discretization.md").read_text()


def off_in_sd(off, sd):
    return -off / sd  # z = (g_true - g_hat) / sd falls by off / sd when the offset is present: z(n = 128) - z(n = 64) = -off / sd


def test_ladder_flags_use_the_analytic_null_and_the_misfit_excess_scaling(tmp_path):
    from bayes_scatter import shapepost as sp

    (tmp_path / "data").mkdir()
    rng = np.random.default_rng(0)
    s = 0.5
    recs = []
    for level, phi_map, g_hat, z in ((0.0, 2395.0, 0.1002, -0.5), (0.01, 2470.0, 0.0986, 5.0), (0.02, 2690.0, 0.0968, 11.0)):
        lab = f"E{level:g}g0.1"
        noise = (rng.standard_normal((6, 8)) + 1j * rng.standard_normal((6, 8))) * s / np.sqrt(2.0)
        np.savez(tmp_path / "data" / f"{lab}.npz", d_clean=np.zeros((6, 8), dtype=complex), d_obs=noise, s=s)
        recs.append({"label": lab, "level": level, "phi_map": phi_map, "n_data": 2400, "g_hat": g_hat, "g_sd": 2.8e-4, "z_true": z})
    # the data above carry 48 numbers, not 2400: only the bookkeeping of noise_only_misfit is tested with them; flags use n_data of the record
    out = pell.ladder_derived(recs, tmp_path)
    for r in out:
        assert r["p_null"] == pytest.approx(sp.misfit_p_value(r["phi_map"], 2400, 5)["p_value"])
        assert r["noise_only_phi"] == pytest.approx(np.sum(np.abs(np.load(tmp_path / "data" / f"{r['label']}.npz")["d_obs"]) ** 2) / s**2)
        assert r["delta_phi"] == pytest.approx(r["phi_map"] - r["noise_only_phi"])
    assert out[0]["c_true"] is None and out[0]["c_shift"] is None and out[0]["dz_shift"] == 0.0  # level 0: no model error, nothing to scale
    assert out[2]["c_true"] == pytest.approx(out[2]["delta_phi"] / 11.0**2) and out[1]["dz_shift"] == pytest.approx((0.0986 - 0.1002) / 2.8e-4)
    assert out[2]["c_shift"] == pytest.approx(out[2]["delta_phi"] / out[2]["dz_shift"] ** 2)
    assert pell.noise_only_misfit(tmp_path, "nonexistent") is None


def test_ti_offline_runs_apply_the_validity_guard_and_report_an_unfit_schedule(tmp_path):
    from bayes_scatter import shapepost as sp

    betas = sp.ti_schedule(16)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_shapepost as tsp

    st, exact = tsp.sharp_gaussian_ti_stats(betas)
    case = tmp_path / "T9_F2"
    (case / "ti_r0").mkdir(parents=True)
    for k in range(15):
        np.savez(case / "ti_r0" / f"t{k:02d}.npz", stats=json.dumps(st[k]))
    runs = pti.offline_runs(case, st[15], 0.0, 16)
    assert len(runs) == 1 and runs[0]["schedule_ok"] is False and runs[0]["end_correction"]["skipped_intervals"] == [1, 2, 3]
    assert runs[0]["log_z"] - exact < 0.5 and runs[0]["log_z"] <= runs[0]["log_z_trapezoid"] + 10.0
    (case / "ti_r0.json").write_text("{}")  # a run that finished by itself is not used
    assert pti.offline_runs(case, st[15], 0.0, 16) == []


def test_collect_lists_the_review_sections(fake_results):
    for model, z in (("F2", 1000.2), ("F2u", 990.0)):
        d = fake_results / f"T1g0.05_{model}"
        (d / "is_evidence.json").write_text(json.dumps({"log_z": z, "se": 0.03, "weight_ess": 233.0, "n_draws": 300, "laplace_gn": z + 0.02, "laplace_fd": z + 0.03,
                                                         "is_minus_laplace_gn": -0.02, "is_minus_laplace_fd": -0.03, "map_converged": True}))
    d = fake_results / "T1g0.05_F2"
    (d / "offline_ti.json").write_text(json.dumps({"n_runs": 2, "log_z": 975.0, "mc_se": 3.0, "between_run_sd": 4.5, "log_z_trapezoid": 974.0, "richardson_error_estimate": 18.0, "laplace_gn": 1000.22,
                                                    "ti_minus_laplace_gn": -25.2, "importance_sampling_log_z": 1000.2, "ti_minus_importance_sampling": -25.2, "schedule_ok": False,
                                                    "skipped_intervals_per_run": {"ti_r0": [0, 1, 2, 3]}, "converged": False, "worst_rhat": 1.22, "min_ess": 4.7}))
    (fake_results / "T1g0.05_Fe" / "map.json").write_text(json.dumps({"case": "x", "model": {"name": "Fe"}, "param_names": ["cx", "cy", "a", "b", "psi", "eps_r"], "p_hat": [0.0, 0.0, 0.43, 0.19, 0.0008, 1.9],
                                                                        "phi_map": 100000.0, "phi_over_N": 43.0, "phi_truth": 2500.0, "map_converged": True, "laplace": {"sd_u": [0.1] * 6},
                                                                        "evidence": {"laplace_gn": -1000.0, "laplace_fd": -1000.0}}))
    chain = np.random.default_rng(0)
    pc = fake_results / "T9g0.1_F2"
    (pc / "chains").mkdir(parents=True)
    for c in range(4):
        for j in range(3):
            np.savez(pc / "chains" / f"c{c}_s{j}.npz", u=chain.standard_normal((100, 5)) * 0.05 - 3.0, phi=2400.0 + chain.standard_normal(100) * 49.0)
    tables = pcol.collect(fake_results)
    occam = {r["truth"]: r for r in tables["is_occam"]}
    assert occam["T1g0.05"]["is"] == pytest.approx(10.2) and occam["T1g0.05"]["laplace_gn"] == pytest.approx(10.2)
    assert {(r["truth"], r["model"]) for r in tables["is_evidence"]} == {("T1g0.05", "F2"), ("T1g0.05", "F2u")}
    ti = tables["ti_offline"][0]
    assert ti["schedule_ok"] is False and ti["ti_minus_is"] == pytest.approx(-25.2) and ti["skipped"] == {"ti_r0": [0, 1, 2, 3]}
    fe = tables["fe_orientation"][0]
    assert fe["truth"] == "T1g0.05" and fe["psi_map"] == pytest.approx(0.0008) and fe["near_wall"] is False
    part = tables["partial_chains"][0]
    assert part["truth"] == "T9g0.1" and part["n_segments"] == 3 and part["n_draws_per_chain"] == 300 and part["gap_mean"] is not None and (pc / "chains_partial.json").exists()
    md = (fake_results / "tables.md").read_text()
    for head in ("## 7. Importance-sampling log evidence", "## 8. Thermodynamic integration", "## 9. Orientation of the Fe fits", "## 10. Cases with chains but no summary"):
        assert head in md, head
    assert "R-hat of Phi (expected under independence)" in md


def test_review_job_files_have_unique_names_and_markers_and_leave_t4_to_the_profile():
    is_jobs = mjobs.review_jobs_for("is")
    assert len(is_jobs) == 15 and all("T4_F2" not in j["name"] for j in is_jobs) and "m7r_is_T4_Fe" in [j["name"] for j in is_jobs]
    assert all(j["done_marker"].endswith("is_evidence.json") and "--draws" in j["cmd"] for j in is_jobs)
    snr = mjobs.review_jobs_for("snr", snr_seeds=5, ppc_reps=3)
    disc = mjobs.review_jobs_for("disc")
    allj = snr + disc + is_jobs
    assert len({j["name"] for j in allj}) == len(allj) and len({j["done_marker"] for j in allj}) == len(allj)
    ppc = [j for j in snr if "_ppc_" in j["name"]]
    assert {j["name"] for j in ppc} == {"m7r_snr_ppc_T5_F2_10", "m7r_snr_ppc_T5_F2_20", "m7r_snr_ppc_T2g0.1_F2_10", "m7r_snr_ppc_T2g0.1_F2_20"} and all(j["cmd"][-1] == "3" for j in ppc)
    t5 = next(j for j in snr if j["name"] == "m7r_snr_T5_F2")
    assert "0" not in t5["cmd"][t5["cmd"].index("--seeds") + 1:]  # seed 0 belongs to the predictive-check unit
    t4 = [j for j in snr if "T4" in j["name"] and "30dB" not in j["name"]]
    assert len(t4) == 2 and all(j["cmd"].count("--seeds") == 1 and len(j["cmd"][j["cmd"].index("--seeds") + 1:]) == min(5, mjobs.T4_SEEDS) for j in t4)
    assert any("--label" in j["cmd"] and "E0.01g0.1" in j["cmd"] for j in snr) and any(j["name"] == "m7r_snr_T1g0.1_F2_30dB" for j in snr)
    with pytest.raises(ValueError):
        mjobs.review_jobs_for("map-f2")


def test_ti_stage_accepts_the_geometric_schedule_option():
    args = prun.parse_args(["--truth", "T1", "--gap", "0.2", "--model", "F2", "--stage", "ti", "--ti-beta-min", "1e-7"])
    assert args.ti_beta_min == 1e-7 and prun.parse_args(["--truth", "T1", "--model", "F2"]).ti_beta_min is None
    from bayes_scatter import shapepost as sp

    betas = sp.ti_schedule(args.ti_temps, args.ti_power, args.ti_beta_min)
    assert betas[0] == 0.0 and betas[1] == pytest.approx(1e-7) and len(betas) == args.ti_temps == 16
