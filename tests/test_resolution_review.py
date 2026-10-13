"""Milestone-6 review follow-up scripts: resolution_review_analysis.py (low-pass reading, SNR check, TV weight sensitivity, p_eff, book geometry),
resolution_detect.py (the detection limit and the ellipticity nuisance) and the job file of the follow-up batch."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import batch_run as br  # noqa: E402
import make_m6_review_jobs as mr  # noqa: E402
import resolution_detect as rd  # noqa: E402
import resolution_review_analysis as ra  # noqa: E402
from bayes_scatter import shapes as sh  # noqa: E402


def test_depth_minus_delta_separates_an_open_gap_from_a_closed_one_and_falls_with_the_cutoff():
    """The pixelized truth of a gap of 0.1 holds the r/2-strip valley (positive), the touching disks do not (negative); the ideal low-pass at 2.5 cycles per wavelength
    keeps the open gap above delta only for wide gaps, and the depth is smaller at a cutoff of 2 than at 6 cycles per wavelength."""
    from bayes_scatter.resolution import lowpass

    h = ra.GRID.h
    for eps in (1.1, 2.0, 4.0):
        open_th = sh.TwoDisks(g=0.1, r=0.2, eps_r=eps).theta(ra.GRID)
        closed_th = sh.TwoDisks(g=0.0, r=0.2, eps_r=eps).theta(ra.GRID)
        assert ra.depth_minus_delta(open_th, 0.1, eps) == pytest.approx(0.5 * np.log(eps), abs=1e-9)  # empty strip: depth = log(eps), minus delta = log(eps) / 2
        assert ra.depth_minus_delta(closed_th, 0.1, eps) < -0.25 * 0.5 * np.log(eps)  # the lens-shaped voids above and below the tangency fill part of the strip
        depths = [ra.depth_minus_delta(lowpass(open_th, h, kc, pad=4), 0.1, eps) for kc in (2.0, 2.5, 3.0, 6.0)]
        assert depths[-1] > depths[0]  # much more bandwidth, deeper valley (not monotone in between: Gibbs ringing of the ideal filter)
    # the stored values of the review (eps_r 2, r/2 strip, 2.5 cycles/lambda, open gap): positive from g = 0.05 on, increasing with the gap
    d = [ra.depth_minus_delta(lowpass(sh.TwoDisks(g=g, r=0.2, eps_r=2.0).theta(ra.GRID), h, 2.5, pad=4), g, 2.0) for g in (0.05, 0.1, 0.15)]
    assert d[0] < d[1] < d[2] and d[1] > 0.1


def test_book_geometry_reproduces_the_quoted_standoff_data_count_and_decay(tmp_path):
    res = ra.stage_book(tmp_path)
    assert res["standoff_wavelengths"]["transmitters_m"] == pytest.approx(0.35 / 0.6)
    assert res["standoff_wavelengths"]["receivers_m"] == pytest.approx(0.45 / 0.6)
    assert res["n_data"]["book"] == 20736 and res["n_data"]["ratio"] == pytest.approx(8.64) and res["n_data"]["aggregate_snr_gain_db_at_equal_per_datum_snr"] == pytest.approx(9.365, abs=1e-3)
    log = res["evanescent_decay_per_path"]["log_factors"]
    assert log["gap lambda/10 (f = 5)"] == pytest.approx(-2 * np.pi * np.sqrt(24.0) * 0.35 / 0.6) and log["gap lambda/10 (f = 5)"] == pytest.approx(-17.95, abs=0.01)
    assert log["gap lambda/20 (f = 10)"] == pytest.approx(-36.47, abs=0.02)
    assert res["evanescent_decay_per_path"]["factors"]["gap lambda/4 (period 2g = lambda/2, f = 2)"] == pytest.approx(1.8e-3, rel=0.05)
    assert res["evanescent_decay_this_benchmark"]["log_factors"]["gap lambda/4 (period 2g = lambda/2, f = 2)"] < -35.0  # the 2-D far rings: no evanescent information at all
    assert json.loads((tmp_path / "book_geometry.json").read_text())["quoted"]["wavelength_m"] == 0.6


def test_peff_and_the_discrepancy_target_from_synthetic_mcmc_and_lambda_files(tmp_path):
    mc = {"summaries": {"a_matern": {"prior": "matern", "chains": 4, "draws_per_chain": 300, "convergence": {"converged": False, "converged_classic": True},
                                     "phi_physical": {"mean": 2436.4, "map_phi": 2399.3}},
                        "b_matern": {"prior": "matern", "chains": 4, "draws_per_chain": 200, "convergence": {"converged": False},
                                     "phi_physical": {"mean": 2440.0, "map_phi": 2399.3}},
                        "c_tv": {"prior": "tv", "chains": 4, "draws_per_chain": 200, "convergence": {"converged": False}, "phi_physical": {"mean": 3000.0, "map_phi": 2400.0}}}}
    lam = {"lambda_star": {"2": {"lam": 1.0}}, "runs": {"2": [{"lam": l, "phi": p, "converged": True} for l, p in ((0.1, 2340.0), (1.0, 2380.0), (10.0, 2420.0))]}}
    (tmp_path / "mcmc.json").write_text(json.dumps(mc))
    (tmp_path / "laplace_lambda.json").write_text(json.dumps(lam))
    res = ra.stage_peff(tmp_path)
    assert set(res["matern_runs"]) == {"a_matern", "b_matern"}  # TV is not used: its chains are unconverged
    assert res["matern_runs"]["a_matern"]["p_eff"] == pytest.approx(2 * 37.1) and res["p_eff_median"] == pytest.approx(np.median([74.2, 81.4]))
    assert res["target"] == pytest.approx(2400 - 0.5 * res["p_eff_median"])
    rule = res["lambda_rule"]["2"]
    assert rule["lambda_closest_to_target"] == 1.0 and rule["lambda_star_N"] == 1.0
    assert rule["crossing_target"] is not None and rule["crossing_target"] < rule["crossing_N"]  # a lower target moves the crossing to a smaller weight


def test_detect_script_at_a_tiny_grid_gives_the_wald_detection_limit_and_a_not_smaller_bound_with_more_nuisances(tmp_path):
    res = rd.main(["--grid", "16", "--out-dir", str(tmp_path)])
    assert (tmp_path / "detection.json").exists()
    for z in res["at_zero"]:
        assert z["detection_limit_over_sigma"] == pytest.approx(2.9265, abs=1e-3) and z["snr_above_smith_db"] == pytest.approx(9.33, abs=0.01)
        assert z["detection_limit_g"] == pytest.approx(z["detection_limit_over_sigma"] * z["sigma_crb_g_at_0"])
        assert np.isfinite(z["cond_F_scaled"]) and z["cond_F_scaled"] < 1e4 and abs(z["corr_g_r"]) < 1.0 and z["min_eigenvalue_over_max_F_scaled"] > 0.0
    for e in res["ellipticity"]:
        s = e["sigma_crb_g"]
        assert s["plus_sx"] >= s["circles"] * (1 - 1e-6) and s["plus_sx_sy"] >= s["plus_sx"] * (1 - 1e-6)  # a bound with more unknowns is never smaller
        assert s["plus_sx"] == pytest.approx(s["plus_sy"], rel=1e-6)  # the span of {r, sx, sy} is two-dimensional: either stretch frees the same direction
        assert e["snr_shift_db"] == pytest.approx(20 * np.log10(e["ratio_plus_sx_sy_over_circles"]))


def test_review_job_file_names_the_files_the_scripts_write(tmp_path):
    jobs = mr.all_jobs(links=False)
    out = tmp_path / "jobs.json"
    out.write_text(json.dumps(jobs))
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names) and all(br.NAME_RE.fullmatch(n) for n in names)
    for j in jobs:
        assert j["threads"] == 1 and j["device"] == "cpu" and j["done_marker"].startswith("results/resolution/") and all(isinstance(c, str) for c in j["cmd"])
    by = {j["name"]: j for j in jobs}
    # the truth chains reuse the median step of the four overdispersed chains and a tagged directory; 4 segments of 100 draws
    t = by["m6r_mcmc_g0.025_eps2_matern_truthfrozen"]
    assert "--frozen-step" in t["cmd"] and "--run-tag" in t["cmd"] and t["cmd"][t["cmd"].index("--run-tag") + 1] == "_truthfrozen" and t["done_marker"].endswith("_truthfrozen/chain_4_seg3.npz")
    steps = sorted(json.loads((mr.MCMC_DIR / "g0.025_eps2_matern" / "summary.json").read_text())["sampler"]["final_step"]) if (mr.MCMC_DIR / "g0.025_eps2_matern" / "summary.json").exists() else None
    if steps:
        assert float(t["cmd"][t["cmd"].index("--frozen-step") + 1]) == pytest.approx(0.5 * (steps[1] + steps[2]), abs=1e-6)
    assert "--init-scale" in by["m6r_tv_g0.1_od2_c0"]["cmd"] and by["m6r_tv_g0.1_od2_c0"]["cmd"][by["m6r_tv_g0.1_od2_c0"]["cmd"].index("--init-scale") + 1] == "2"
    assert by["m6r_tv_g0.1_od2_c3"]["done_marker"].endswith("g0.1_eps2_tv_od2/chain_3_seg3.npz") and by["m6r_tv_g0.15_c0"]["done_marker"].endswith("g0.15_eps2_tv/chain_0_seg2.npz")
    assert "--run-tag" not in by["m6r_tv_g0.15_c0"]["cmd"]  # a new configuration: the default run directory
    mle_chunks = [j for j in jobs if j["name"].startswith("m6r_mle_")]
    assert sum(1 for j in mle_chunks if "eps2_g0.05_req" in j["name"]) == 6 and sum(1 for j in mle_chunks if "plus10" in j["name"]) == 3
    assert all(j["cmd"][0] == "scripts/resolution_mle.py" for j in mle_chunks)
    for j in jobs:
        if j["name"].startswith("m6r_lam_") or j["name"].startswith("m6r_snr"):
            assert "--then-refeature" in j["cmd"]
    assert {j["name"] for j in jobs if j["name"].startswith("m6r_snr")} == {f"m6r_snr{s}_{p}" for s in (20, 40) for p in ("matern", "tv")}


def test_finalize_job_file_names_the_summaries_the_finalize_stage_writes():
    jobs = mr.finalize_jobs()
    names = [j["name"] for j in jobs]
    assert len(set(names)) == len(names) == 5 and all(br.NAME_RE.fullmatch(n) for n in names)
    by = {j["name"]: j for j in jobs}
    t = by["m6r_fin_g0.025_matern_truthfrozen"]["cmd"]
    assert t[t.index("--chains") + 1] == "5" and t[t.index("--tail-segments") + 1] == "2" and t[t.index("--run-tag") + 1] == "_truthfrozen"
    assert by["m6r_fin_g0.1_tv_od2"]["done_marker"] == "results/resolution/mcmc/g0.1_eps2_tv_od2/summary.json"
    assert by["m6r_fin_g0.15_tv"]["done_marker"] == "results/resolution/mcmc/g0.15_eps2_tv/summary.json" and "--run-tag" not in by["m6r_fin_g0.15_tv"]["cmd"]


def test_verdict_range_replaces_the_laplace_probability_by_the_mcmc_one_where_a_run_exists(tmp_path):
    """eps_r 2, half-height strip: Laplace says TV P_open(0.1) = 0.87 (the narrowest discriminating gap is 0.15), an MCMC run says 0.99 (then 0.1 discriminates);
    a run that does not cover a gap leaves that gap on the Laplace value."""
    var = "band0.5_core0.5"

    def row(prior, g, p_open, p_null):
        return {"eps_r": 2.0, "prior": prior, "strip_gap": g, "variant": var, "p_open": p_open, "p_null": p_null}

    rows = [row("matern", 0.15, 1.0, 0.0), row("tv", 0.15, 1.0, 0.0), row("matern", 0.1, 1.0, 0.0), row("tv", 0.1, 0.87, 0.0),
            row("matern", 0.05, 0.72, 0.0), row("tv", 0.05, 0.16, 0.0)]
    (tmp_path / "null.json").write_text(json.dumps({"rows": rows}))
    ref = {"configurations": {"g0.1_eps2_tv": {"variants": {var: {"p_mcmc": 0.99, "mcse": 0.01}}}, "g0.15_eps2_tv": {"variants": {var: {"p_mcmc": 0.5, "mcse": 0.05}}}}}
    (tmp_path / "mcmc_refeature.json").write_text(json.dumps({"configurations": {"g0.1_eps2_tv": ref["configurations"]["g0.1_eps2_tv"]}}))
    review = {"configurations": {"g0.1_eps2_tv_od2": {"variants": {var: {"p_mcmc": 0.80, "mcse": 0.05}}}, "g0.15_eps2_tv": ref["configurations"]["g0.15_eps2_tv"]}}
    (tmp_path / "mcmc_refeature_review.json").write_text(json.dumps(review))
    (tmp_path / "mcmc.json").write_text(json.dumps({"table": [{"name": "g0.1_eps2_tv", "converged": False, "chains": 4, "draws_per_chain": 50},
                                                               {"name": "g0.1_eps2_tv_od2", "converged": False, "chains": 4, "draws_per_chain": 200}]}))
    res = ra.stage_verdict(tmp_path)
    assert res["narrowest_gap"]["laplace"] == 0.15
    assert res["narrowest_gap"]["mcmc_original_runs"] == 0.1  # TV 0.99 at 0.1 from the original run; 0.05 stays out (Matern 0.72 and TV 0.16 by Laplace)
    cell = next(c for c in res["table"] if c["strip_gap"] == 0.1)
    assert set(cell["tv"]["mcmc"]) == {"g0.1_eps2_tv", "g0.1_eps2_tv_od2"} and cell["tv"]["mcmc"]["g0.1_eps2_tv"]["p_open_mcmc"] == 0.99 and cell["matern"]["mcmc"] == {}
    assert cell["tv"]["mcmc"]["g0.1_eps2_tv_od2"]["draws_per_chain"] == 200
    # the 0.15 TV run says 0.5: with it the 0.15 gap no longer discriminates, and neither does anything below
    assert res["narrowest_gap"]["mcmc_all_runs_where_present"] is None
    assert (tmp_path / "verdict_range.json").exists()


def test_mle_chunk_lock_is_exclusive_and_stale_locks_are_taken_over(tmp_path):
    """Two jobs with the same chunk (a batch job and a manual run) must not both compute it: the second finds the lock; a lock left by a killed job (older than the
    stale limit) is taken over; a finished chunk (its JSON) is returned without computing."""
    import os
    import time

    import resolution_mle as rmle

    lock = tmp_path / "mle" / "pt" / "chunk_0_8.lock"
    assert rmle.acquire_lock(lock) is True and lock.exists()
    assert rmle.acquire_lock(lock) is False  # held
    old = time.time() - 3 * 3600
    os.utime(lock, (old, old))
    assert rmle.acquire_lock(lock) is True  # stale: taken over, fresh again
    assert rmle.acquire_lock(lock) is False
    lock.unlink()
    # a finished chunk is returned as is and no lock is created
    path = rmle.chunk_path(tmp_path, "eps2_g0.05_req", 0, 8)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"point": "eps2_g0.05_req", "records": []}))
    res = rmle.main(["--stage", "chunk", "--point", "eps2_g0.05_req", "--first", "0", "--count", "8", "--out-dir", str(tmp_path)])
    assert res["records"] == [] and not path.with_suffix(".lock").exists()
    # a chunk whose lock is held elsewhere returns None without computing (no forward model is built)
    held = rmle.chunk_path(tmp_path, "eps2_g0.1_req", 8, 16).with_suffix(".lock")
    assert rmle.acquire_lock(held)
    assert rmle.main(["--stage", "chunk", "--point", "eps2_g0.1_req", "--first", "8", "--count", "8", "--out-dir", str(tmp_path)]) is None
    assert held.exists()  # the holder's lock is untouched
