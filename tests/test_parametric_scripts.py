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
