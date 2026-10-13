"""Milestone 6 scripts: resolution_crb.py, resolution_laplace.py, resolution_mcmc.py and make_m6_jobs.py on tiny problems.

The numerical validation lives in test_shapes.py, test_fisher.py, test_gapvalley.py and test_thetaspace.py; here the scripts must run end to end,
write the files they promise, resume, and agree with each other (names, markers, job files).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import batch_run as br  # noqa: E402
import make_m6_extra_jobs as mx  # noqa: E402
import make_m6_jobs as mj  # noqa: E402
import resolution_cost as rcost  # noqa: E402
import resolution_crb as rc  # noqa: E402
import resolution_laplace as rl  # noqa: E402
import resolution_mcmc as rm  # noqa: E402

TINY = ["--n", "12", "--side", "1.5"]  # h = 0.125, data grid 24: g = 0.25 is two pixels wide


# ---------------------------------------------------------------------------------------------
# resolution_crb.py
# ---------------------------------------------------------------------------------------------


def test_crb_script_runs_resumes_and_aggregates(tmp_path, capsys):
    out = tmp_path / "res"
    argv = ["--grids", "16", "20", "--gaps", "0.1", "0.25", "--eps", "2", "--out-dir", str(out)]
    res = rc.main(argv)
    for f in ("crb.json", "crb_sigma_over_g.png", "crb_required_snr.png", "crb_born_ratio.png", "crb_grid_convergence.png"):
        assert (out / f).exists(), f
    assert sorted(p.name for p in (out / "crb").glob("*.json")) == sorted(f"n{n}_eps2_g{g}.json" for n in (16, 20) for g in ("0.1", "0.25"))
    on_disk = json.loads((out / "crb.json").read_text())
    assert [r["g"] for r in on_disk["table"]] == [0.1, 0.25] and on_disk["setup"]["grids"] == [16, 20]
    assert on_disk["setup"]["nuisances_unknown"] == ["r", "eps_r", "cx", "cy"]
    # closed-form SNR: recomputing the bounds at the required SNR returns sigma_CRB(g) = g
    assert on_disk["required_snr_check"]["n_checked"] == 2 and on_disk["required_snr_check"]["max_abs_sigma_over_g_minus_1"] < 1e-9
    # the symmetric object: cross blocks and column parities are solver round-off
    sym = on_disk["symmetry"]
    assert sym["max_cross_residual_x"] < 1e-8 and sym["max_cross_residual_y"] < 1e-8 and sym["max_parity_x"] < 1e-8 and sym["max_solver_residual"] < 1e-8
    row = on_disk["table"][0]
    assert row["sigma_over_g_known"] <= row["sigma_over_g_unknown"] * (1 + 1e-12) and row["sigma_over_g_unknown_all"] == pytest.approx(row["sigma_over_g_unknown"], rel=1e-6)
    # the book's operating points (Tests #5 to #8: gaps 0.1 to 0.4 at 20 and 10 dB) scale as 10^(-(SNR - 30) / 20); Test #4 has no noise level
    bc = on_disk["book_claims"]["groups"]
    e = bc["tests5to8"]["by_eps"]["2"]["0.1"]
    assert e["snr"]["10"]["sigma_over_g"]["unknown"] == pytest.approx(row["sigma_over_g_unknown"] * 10.0 ** 1.0, rel=1e-12)
    assert e["snr"]["20"]["sigma_over_g"]["unknown"] == pytest.approx(row["sigma_over_g_unknown"] * 10.0 ** 0.5, rel=1e-12)
    assert e["snr"]["10"]["margin_over_required_db"] == pytest.approx(10.0 - row["required_snr_db"]["unknown"])
    assert e["snr"]["10"]["nuisance_rel_sd"]["r"] == pytest.approx(row["nuisance_at_required"]["rel_sd"]["r"] * 10.0 ** (-(10.0 - row["required_snr_db"]["unknown"]) / 20.0))  # sd ~ 10^(-SNR / 20)
    assert e["snr"]["10"]["bound_is_local_there"] is True and set(row["nuisance_at_book_snr"]) == {"10", "20"}  # at the book's SNR the bound is local, at the required SNR not
    t4 = bc["test4"]["by_eps"]["2"]
    assert bc["test4"]["snr_db"] == [] and t4["0.1"]["snr"] == {} and set(t4) == {"0.1", "0.25"} and set(bc["tests5to8"]["by_eps"]["2"]) == {"0.1"}  # the gaps of the sweep only
    # the nuisance bounds at the required SNR and the local-regime flag: sd scales as 10^(-(SNR - 30) / 20) and the flag is the 0.1 rule on r and eps_r - 1
    nu = row["nuisance_at_required"]
    assert set(nu["sd"]) == {"r", "eps_r", "cx", "cy", "g"} and nu["sd"]["g"] == pytest.approx(0.1, rel=1e-9)  # sigma_CRB(g) = g at the required SNR
    assert nu["rel_sd"]["r"] == pytest.approx(nu["sd"]["r"] / 0.2) and nu["rel_sd"]["eps_r"] == pytest.approx(nu["sd"]["eps_r"])
    assert row["local_regime_at_required"] == (max(nu["rel_sd"].values()) <= 0.1) and nu["aggregate_snr_db"] == pytest.approx(nu["snr_db"] + 10 * np.log10(2400))
    conv = on_disk["convergence"]
    assert conv["grids"] == [16, 20] and len(conv["rows"]) == 2 and conv["max"]["rel_diff_sigma_unknown"] < 0.5
    assert set(on_disk["records"]) == {"n16_eps2_g0.1", "n16_eps2_g0.25", "n20_eps2_g0.1", "n20_eps2_g0.25"}
    # resume: nothing is recomputed (the record files keep their modification time), --force recomputes
    mtimes = {p: p.stat().st_mtime_ns for p in (out / "crb").glob("*.json")}
    capsys.readouterr()
    rc.main(argv)
    assert "records exist" in capsys.readouterr().out
    assert {p: p.stat().st_mtime_ns for p in (out / "crb").glob("*.json")} == mtimes
    rc.main(argv[:-2] + ["--out-dir", str(out), "--aggregate-only"])
    assert res["table"] == json.loads((out / "crb.json").read_text())["table"]
    # --book-gaps (primary grid only) and --born-power: the extra book gaps enter book_claims but not the table; the matched-power Born ratios use born_power.json
    capsys.readouterr()
    rc.main(argv + ["--book-gaps", "--born-power"])
    ext = json.loads((out / "crb.json").read_text())
    assert sorted(p.name for p in (out / "crb").glob("n*_eps2_g0.[34].json")) == ["n16_eps2_g0.3.json", "n16_eps2_g0.4.json"]
    assert set(ext["book_claims"]["groups"]["tests5to8"]["by_eps"]["2"]) == {"0.1", "0.3", "0.4"} and [r["g"] for r in ext["table"]] == [0.1, 0.25]
    bp = json.loads((out / "born_power.json").read_text())
    assert {"n16_eps2_g0.1", "n20_eps2_g0.25", "n16_eps2_g0.4"} <= set(bp) and ext["born_power"]["max_rel_diff_power_full_vs_record"] < 1e-8
    r0 = ext["table"][0]
    assert r0["power_born"] == bp["n16_eps2_g0.1"]["power_born"] and r0["power_ratio_full_over_born"] == pytest.approx(bp["n16_eps2_g0.1"]["power_ratio_full_over_born"])
    assert r0["info_ratio_known_matched"] == pytest.approx(r0["info_ratio_known"] * r0["power_born"] / r0["power"], rel=1e-12)
    assert r0["info_ratio_unknown_matched"] == pytest.approx(r0["info_ratio_unknown"] * r0["power_born"] / r0["power"], rel=1e-12)
    assert ext["local_regime"]["n_rows"] == 2 and ext["local_regime"]["n_local_at_required"] + len(ext["local_regime"]["rows_outside"]) == 2


def test_crb_convergence_and_book_claims_on_synthetic_records():
    F = np.diag([4.0, 1.0, 1.0, 1.0, 1.0, 1.0]).tolist()

    def rec(n, sig, g=0.1):
        return {"n": n, "eps_r": 2.0, "g": g, "F": F, "crb_g_unknown": sig, "crb_g_known": 0.5 * sig, "info_ratio_known": 0.6,
                "required_snr_db": {"unknown": 30 + 20 * np.log10(sig / g)}}

    recs = {(16, 2.0, 0.1): rec(16, 1e-3), (32, 2.0, 0.1): rec(32, 1.01e-3)}
    out = rc.convergence(recs, [16, 32])
    assert out["rows"][0]["rel_diff_sigma_unknown"] == pytest.approx(0.01 / 1.01, rel=1e-9) and out["max"]["max_abs_diff_F_scaled"] == 0.0
    assert rc.convergence(recs, [16])["note"].startswith("a single grid")
    nuis = {"sd": {}, "rel_sd": {"r": 0.05, "eps_r": 0.2}, "local_regime": False}
    rows = [{"eps_r": 2.0, "g": g, "sigma_over_g_unknown": 0.01, "sigma_over_g_known": 0.005, "required_snr_db": {"unknown": -10.0, "known": -16.0},
             "nuisance_at_required": nuis, "local_regime_at_required": False,
             "nuisance_at_book_snr": {"10": {"rel_sd": {"r": 0.001, "eps_r": 0.002}, "local_regime": True}, "20": {"rel_sd": {"r": 0.0003, "eps_r": 0.0006}, "local_regime": True}}}
            for g in (0.05, 0.1, 0.4)]
    bc = rc.book_claims(rows, [2.0])["groups"]
    assert bc["tests5to8"]["by_eps"]["2"]["0.1"]["snr"]["10"]["sigma_over_g"]["known"] == pytest.approx(0.005 * 10.0)
    assert bc["tests5to8"]["by_eps"]["2"]["0.4"]["snr"]["20"]["margin_over_required_db"] == pytest.approx(30.0)
    assert set(bc["test4"]["by_eps"]["2"]) == {"0.05", "0.1"} and bc["test4"]["snr_db"] == [] and bc["test4"]["by_eps"]["2"]["0.05"]["local_regime_at_required"] is False
    assert rc.tag(128, 1.1, 0.0125) == "n128_eps1.1_g0.0125"
    # the operating points of the book as the review corrected them: Test #4 noise level not stated, Tests #5 to #8 at 20 and 10 dB
    assert rc.BOOK["test4"]["snr_db"] == () and rc.BOOK["tests5to8"]["snr_db"] == (20.0, 10.0) and rc.BOOK["tests5to8"]["gaps"] == (0.4, 0.3, 0.2, 0.1)
    assert rc.BOOK["test4"]["gaps"] == (0.25, 0.2, 0.15, 0.1, 0.05) and rc.BOOK_EXTRA_GAPS == (0.3, 0.4)


# ---------------------------------------------------------------------------------------------
# resolution_laplace.py and resolution_mcmc.py
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_results(tmp_path_factory):
    """The whole Laplace pipeline at n = 12 (data on n = 24): lambda sweep, both priors at g = 0.25 and, for the Matern prior, g = 0.0125 (narrower than a
    pixel, started also from the g = 0.25 MAP), a TV sensitivity run, collection."""
    out = tmp_path_factory.mktemp("resolution")
    base = TINY + ["--out-dir", str(out)]
    rl.main(["--stage", "lambda", "--eps", "2", "--lams", "1", "10"] + base)
    rl.main(["--stage", "lambda-collect", "--eps", "2"] + base)
    rl.main(["--stage", "descend", "--eps-r", "2", "--prior", "matern", "--gaps", "0.0125", "0.25"] + base)  # run from the widest gap down
    rl.main(["--stage", "descend", "--eps-r", "2", "--prior", "tv", "--gaps", "0.25"] + base)
    rl.main(["--stage", "descend", "--eps-r", "2", "--prior", "tv", "--gaps", "0.25", "--lam-scale", "3"] + base)
    rl.main(["--stage", "collect", "--eps", "2"] + base)
    # every start (the multimodality check): both reach the same stationary point at this contrast
    rl.main(["--stage", "config", "--g", "0.25", "--eps-r", "2", "--prior", "matern", "--all-starts"] + TINY + ["--out-dir", str(out / "allstarts")])
    return out


def test_lambda_sweep_picks_the_value_closest_to_n_and_stores_the_crossing(tiny_results):
    lam = json.loads((tiny_results / "laplace_lambda.json").read_text())
    star = lam["lambda_star"]["2"]
    runs = lam["runs"]["2"]
    assert [r["lam"] for r in runs] == [1.0, 10.0] and star["swept"] == [1.0, 10.0]
    best = min(runs, key=lambda r: abs(r["phi"] - r["n_data"]))
    assert star["lam"] == best["lam"] and star["n_data"] == 2400
    assert runs[1]["phi"] > runs[0]["phi"] and runs[1]["R"] > runs[0]["R"]  # a stronger prior fits the data worse
    assert all(r["converged"] for r in runs)
    assert (tiny_results / "laplace_lambda.png").exists()
    assert rl.crossing([{"lam": 1.0, "phi": 2300.0}, {"lam": 10.0, "phi": 2500.0}], 2400) == pytest.approx(np.sqrt(10.0))
    assert rl.crossing([{"lam": 1.0, "phi": 2500.0}, {"lam": 10.0, "phi": 2600.0}], 2400) is None


def test_config_records_have_the_promised_content(tiny_results):
    lam_star = json.loads((tiny_results / "laplace_lambda.json").read_text())["lambda_star"]["2"]["lam"]
    jm, nm = rl.config_paths(tiny_results, "g0.25_eps2_matern")
    jt, _ = rl.config_paths(tiny_results, "g0.25_eps2_tv")
    mat, tv = json.loads(jm.read_text()), json.loads(jt.read_text())
    assert mat["lam"] is None and tv["lam"] == lam_star and tv["lam_scale"] == 1.0
    for rec in (mat, tv):
        assert rec["problem"]["n"] == 12 and rec["problem"]["data_grid_n"] == 24 and rec["problem"]["n_data"] == 2400
        assert rec["map"]["converged"] and rec["map"]["method"].startswith("damped Gauss-Newton") and rec["map_summary"]["map_ok"]
        assert rec["map_summary"]["phi_over_N"] == pytest.approx(rec["map_summary"]["phi"] / 2400)
        assert rec["laplace"]["jacobian_check_rel_err"] < 1e-7 and rec["laplace"]["asymmetry"] < 1e-12
        feats = rec["features"]
        assert feats["region_status"] == "ok" and 0.0 <= feats["region"]["p_valley"] <= 1.0 and 0.0 <= feats["point"]["p_valley"] <= 1.0
        assert feats["delta"] == pytest.approx(0.5 * np.log(2.0)) and "single-point" in feats["point_label"]
        assert rec["truth"]["region"]["holds"] is True and rec["operating_point"]["phi_truth"] > 0
        assert rec["thread_env"] is not None and rec["timings_s"]["total"] > 0
    # the search stops at the first successful start: the smooth start for the widest Matern gap, the Matern MAP for TV
    assert set(mat["map_candidates"]) == {"smooth"} and set(tv["map_candidates"]) == {"matern"} and not mat["map"]["all_starts"]
    assert mat["map"]["chosen_start"] == "smooth" and tv["map"]["chosen_start"] == "matern" and mat["map_summary"]["u_gap"] is None
    assert mat["neighbor_start_given"] is False and tv["neighbor_start_given"] is False
    z = np.load(nm)
    assert z["theta_hat"].shape == (12, 12) and z["laplace_std"].shape == (12, 12) and np.all(z["laplace_std"] > 0)
    np.testing.assert_allclose(z["theta_true"], np.log1p(z["chi_true"]))
    # the data are the same for both priors (one noise key)
    assert mat["problem"]["d_obs_sha256"] == tv["problem"]["d_obs_sha256"] and mat["problem"]["noise_std"] == tv["problem"]["noise_std"]


def test_all_starts_reach_the_same_stationary_point_for_a_moderate_contrast(tiny_results):
    rec = json.loads(rl.config_paths(tiny_results / "allstarts", "g0.25_eps2_matern")[0].read_text())
    assert set(rec["map_candidates"]) == {"smooth", "vacuum"} and rec["map"]["all_starts"] is True
    assert rec["map_summary"]["u_gap"] is not None and rec["map_summary"]["u_gap"] < 1e-3
    first = json.loads(rl.config_paths(tiny_results, "g0.25_eps2_matern")[0].read_text())
    assert rec["map_summary"]["U"] == pytest.approx(first["map_summary"]["U"], abs=1e-3)  # the winner is the same point as the sequential search found


def test_tv_sensitivity_run_and_the_unrepresentable_gap(tiny_results):
    lam_star = json.loads((tiny_results / "laplace_lambda.json").read_text())["lambda_star"]["2"]["lam"]
    sens = json.loads(rl.config_paths(tiny_results, "g0.25_eps2_tv_x3")[0].read_text())
    assert sens["lam"] == pytest.approx(3.0 * lam_star) and sens["name"] == "g0.25_eps2_tv_x3"
    thin = json.loads(rl.config_paths(tiny_results, "g0.0125_eps2_matern")[0].read_text())
    assert set(thin["map_candidates"]) == {"neighbor"} and thin["neighbor_start_given"] is True  # the descent started it from the g = 0.25 MAP, which succeeded
    assert thin["features"]["region"] is None and thin["features"]["region_status"] == "unrepresentable on this grid"
    assert 0.0 <= thin["features"]["point"]["p_valley"] <= 1.0 and thin["truth"]["region"]["holds"] is None
    row = [r for r in json.loads((tiny_results / "laplace.json").read_text())["table"] if r["g"] == 0.0125][0]
    assert row["p_valley_region"] is None and row["region_status"] == "unrepresentable on this grid"


def test_collect_writes_the_table_and_figures_and_a_finished_configuration_is_skipped(tiny_results, capsys, tmp_path):
    col = json.loads((tiny_results / "laplace.json").read_text())
    assert len(col["table"]) == 4 and set(col["records"]) == {"g0.25_eps2_matern", "g0.25_eps2_tv", "g0.25_eps2_tv_x3", "g0.0125_eps2_matern"}
    assert col["lambda"]["lambda_star"]["2"]["n_data"] == 2400 and "bivariate-normal" in col["setup"]["posterior"]
    for f in ("laplace_pvalley.png", "laplace_lambda.png", "laplace_verdict.png"):
        assert (tiny_results / f).exists(), f
    assert "table" in col["verdicts"] and col["verdicts"]["table"][0]["eps_r"] == 2.0 and col["verdicts"]["table"][0]["verdict"] in ("present", "absent", "prior-dependent")
    jpath = rl.config_paths(tiny_results, "g0.25_eps2_matern")[0]
    before = jpath.stat().st_mtime_ns
    capsys.readouterr()
    rl.main(["--stage", "config", "--g", "0.25", "--eps-r", "2", "--prior", "matern"] + TINY + ["--out-dir", str(tiny_results)])
    assert "done" in capsys.readouterr().out and jpath.stat().st_mtime_ns == before
    rl.main(["--stage", "descend", "--eps-r", "2", "--prior", "matern", "--gaps", "0.0125", "0.25"] + TINY + ["--out-dir", str(tiny_results)])
    assert capsys.readouterr().out.count(": done") == 2 and jpath.stat().st_mtime_ns == before  # a descent over finished configurations computes nothing
    # a stored MAP that failed is not used as a start, and a successful one is
    assert rl.load_good_map(tiny_results, "g0.25_eps2_matern") is not None and rl.load_good_map(tiny_results, "g0.5_eps2_matern") is None
    fake = json.loads(jpath.read_text())
    fake["map_summary"]["map_ok"] = False
    bad = rl.config_paths(tmp_path, "g0.2_eps2_matern")
    bad[0].parent.mkdir(parents=True)
    bad[0].write_text(json.dumps(fake))
    np.savez(bad[1], theta_hat=np.zeros((12, 12)))
    assert rl.load_good_map(tmp_path, "g0.2_eps2_matern") is None
    fake["map_summary"]["map_ok"] = True
    bad[0].write_text(json.dumps(fake))
    assert rl.load_good_map(tmp_path, "g0.2_eps2_matern") is not None


@pytest.mark.slow
def test_mcmc_spot_check_chains_checkpoint_per_segment_and_finalize(tiny_results, capsys):
    base = TINY + ["--out-dir", str(tiny_results), "--g", "0.25", "--eps-r", "2", "--prior", "matern", "--chains", "2", "--samples", "10", "--warmup", "10",
                   "--last", "1", "--n-leapfrog", "5"]
    for c in (0, 1):
        rm.main(["--stage", "chain", "--chain-id", str(c)] + base)
    d = tiny_results / "mcmc" / "g0.25_eps2_matern"
    assert sorted(p.name for p in d.glob("chain_*_seg*.npz")) == [f"chain_{c}_seg{j}.npz" for c in (0, 1) for j in (0, 1)]
    stamp = (d / "chain_0_seg0.npz").stat().st_mtime_ns
    capsys.readouterr()
    rm.main(["--stage", "chain", "--chain-id", "0"] + base)  # everything exists: nothing is recomputed
    assert "exist" in capsys.readouterr().out and (d / "chain_0_seg0.npz").stat().st_mtime_ns == stamp
    res = rm.main(["--stage", "finalize"] + base)
    assert res["chains"] == 2 and res["draws_per_chain"] == 20 and res["segments"] == [(10, 10), (0, 10)]
    assert res["valley_region"]["n_draws"] == 40 and 0.0 <= res["valley_region"]["p_valley_mcmc"] <= 1.0
    assert res["valley_region"]["p_valley_laplace"] == pytest.approx(
        json.loads(rl.config_paths(tiny_results, "g0.25_eps2_matern")[0].read_text())["features"]["region"]["p_valley"])
    assert res["phi_physical"]["mean"] > 0 and res["sampler"]["mass_theta_is_map"] is True
    assert (d / "summary.json").exists() and (d / "series.npz").exists()
    z = np.load(d / "series.npz")
    assert z["strip"].shape == (2, 20) and z["phi"].shape == (2, 20)
    rm.main(["--stage", "collect", "--out-dir", str(tiny_results)])
    assert (tiny_results / "mcmc.json").exists() and (tiny_results / "mcmc_spotcheck.png").exists()


@pytest.mark.slow
def test_mcmc_chain_started_at_the_truth_joins_the_overdispersed_chains(tiny_results):
    base = TINY + ["--out-dir", str(tiny_results), "--g", "0.25", "--eps-r", "2", "--prior", "matern", "--samples", "10", "--warmup", "10", "--last", "1",
                   "--n-leapfrog", "5"]
    for c in (0, 1):  # the overdispersed chains of the spot check (a no-op if they exist)
        rm.main(["--stage", "chain", "--chain-id", str(c), "--chains", "2"] + base)
    rm.main(["--stage", "chain", "--chain-id", "2", "--chains", "3", "--start", "truth"] + base)
    d = tiny_results / "mcmc" / "g0.25_eps2_matern"
    first = np.load(d / "chain_2_seg0.npz")
    assert str(first["starts"]) == "the pixelized truth" and str(np.load(d / "chain_2_seg1.npz")["starts"]).startswith("final state")
    assert str(np.load(d / "chain_0_seg0.npz")["starts"]).startswith("theta_hat")
    res = rm.main(["--stage", "finalize", "--chains", "3", "--tag", "_truth5"] + base)
    assert res["chains"] == 3 and res["tag"] == "_truth5" and res["chain_starts"][2] == "the pixelized truth" and res["chain_starts"][0].startswith("theta_hat")
    assert (d / "summary_truth5.json").exists() and (d / "series_truth5.npz").exists()
    assert len(res["valley_region"]["per_chain"]) == 3 and res["valley_region"]["n_draws"] == 60
    rm.main(["--stage", "collect", "--out-dir", str(tiny_results)])
    wt = json.loads((tiny_results / "mcmc.json").read_text())["with_truth_chain"]
    assert wt["g0.25_eps2_matern_truth5"]["chains"] == 3 and len(wt["g0.25_eps2_matern_truth5"]["p_region_per_chain"]) == 3
    bm = wt["g0.25_eps2_matern_truth5"]["block_means"]
    assert len(bm["phi"]) == 3 and len(bm["phi"][0]) == 6 and len(bm["depth_region"]) == 3 and bm["blocks"][0][0] == 0 and bm["blocks"][-1][1] == 20


def test_energy_convergence_flags_a_slowly_mixing_potential_that_the_classic_criteria_miss():
    """Phi' = U - |zeta|^2 / 2 is the direction HMC mixes slowest. Four iid chains pass; AR(1) chains with rho 0.98 (ESS about 1 % of the draws) and chains with
    offset means fail, the classic flag is kept as ``converged_classic``, and the reasons name the Phi' criteria."""
    rng = np.random.default_rng(0)
    classic = {"converged": True, "unconverged_reasons": [], "rhat_phi": 1.0}
    iid = rng.standard_normal((4, 400))
    ok = rm.energy_convergence(classic, iid)
    assert ok["converged"] and ok["converged_classic"] and ok["rhat_phi_prime"] < 1.02 and ok["ess_phi_prime"] > 800 and ok["unconverged_reasons"] == []
    ar = np.zeros((4, 400))
    for t in range(1, 400):
        ar[:, t] = 0.98 * ar[:, t - 1] + np.sqrt(1 - 0.98**2) * rng.standard_normal(4)
    slow = rm.energy_convergence(classic, ar)
    assert slow["converged_classic"] is True and slow["converged"] is False and slow["ess_phi_prime"] < 200
    assert any("ESS of Phi'" in r for r in slow["unconverged_reasons"]) and len(slow["ess_phi_prime_per_chain"]) == 4 and len(slow["ess_phi_prime_geyer_per_chain"]) == 4
    offset = iid + np.array([[0.0], [0.0], [0.0], [3.0]])
    off = rm.energy_convergence({"converged": False, "unconverged_reasons": ["pixel"], "rhat_phi": 1.2}, offset)
    assert off["rhat_phi_prime"] > rm.RHAT_PHI_PRIME_MAX and off["converged"] is False and off["converged_classic"] is False
    assert off["unconverged_reasons"][0] == "pixel" and any("R-hat of Phi'" in r for r in off["unconverged_reasons"])
    assert classic["converged"] is True and "rhat_phi_prime" not in classic  # the input is not modified


@pytest.mark.slow
def test_mcmc_frozen_step_truth_chain_tail_segments_and_the_energy_stage(tiny_results, capsys):
    base = TINY + ["--out-dir", str(tiny_results), "--g", "0.25", "--eps-r", "2", "--prior", "matern", "--samples", "10", "--warmup", "10", "--n-leapfrog", "5",
                   "--run-tag", "_tf"]
    for c in (0, 1):
        rm.main(["--stage", "chain", "--chain-id", str(c), "--chains", "3", "--last", "1"] + base)
    d = tiny_results / "mcmc" / "g0.25_eps2_matern_tf"
    step = round(float(np.load(d / "chain_0_seg0.npz")["step"]), 4)  # the adapted step of an overdispersed chain, reused by the truth chain
    capsys.readouterr()
    rm.main(["--stage", "chain", "--chain-id", "2", "--chains", "3", "--start", "truth", "--frozen-step", str(step), "--last", "2"] + base)  # one segment more than the others
    log = capsys.readouterr().out.split("chain 2:")[1].split("\n")[0]
    assert "start: the pixelized truth, frozen step" in log and "(no warmup)" in log and "init scale" not in log  # the log names the start that was used (it used to print init scale 2.0)
    first, second = np.load(d / "chain_2_seg0.npz"), np.load(d / "chain_2_seg1.npz")
    assert float(first["step"]) == pytest.approx(step, rel=1e-12) and float(second["step"]) == pytest.approx(step, rel=1e-12) and int(first["n_warmup"]) == 0  # no adaptation
    assert str(first["starts"]).startswith("the pixelized truth") and f"frozen step {step:g}" in str(first["starts"])
    assert int(np.load(d / "chain_0_seg0.npz")["n_warmup"]) == 10
    assert not (tiny_results / "mcmc" / "g0.25_eps2_matern" / "chain_2_seg2.npz").exists()  # the tagged run directory is separate
    with pytest.raises(SystemExit):  # chains with different numbers of segments cannot be pooled without --tail-segments
        rm.main(["--stage", "finalize", "--chains", "3", "--last", "2"] + base[:-2] + ["--run-tag", "_tf", "--tag", "_all"])
    res = rm.main(["--stage", "finalize", "--chains", "3", "--tail-segments", "2"] + base)
    assert res["name"] == "g0.25_eps2_matern_tf" and res["config"] == "g0.25_eps2_matern" and res["tail_segments"] == 2 and res["draws_per_chain"] == 20
    assert res["segments"] == [(10, 10), (0, 10)]  # chain 0: its own two segments
    assert res["valley_region"]["n_draws"] == 60 and len(res["valley_region"]["per_chain"]) == 3
    conv = res["convergence"]
    assert {"rhat_phi_prime", "ess_phi_prime", "ess_phi_prime_per_chain", "converged_classic", "thresholds_phi_prime"} <= set(conv) and len(conv["ess_phi_prime_per_chain"]) == 3
    assert conv["converged"] == (conv["converged_classic"] and not any("Phi'" in r for r in conv["unconverged_reasons"]))
    z = np.load(d / "series.npz")
    assert z["phi_prime"].shape == (3, 20) and z["phi"].shape == (3, 20)
    # the energy stage adds the same numbers to a summary that lacks them (the summaries written before the Phi' criteria existed)
    path = d / "summary.json"
    old = json.loads(path.read_text())
    for key in ("rhat_phi_prime", "ess_phi_prime", "ess_phi_prime_per_chain", "ess_phi_prime_geyer_per_chain", "thresholds_phi_prime", "converged_classic"):
        old["convergence"].pop(key)
    old["convergence"]["unconverged_reasons"] = [r for r in old["convergence"]["unconverged_reasons"] if "Phi'" not in r]
    old["convergence"]["converged"] = not old["convergence"]["unconverged_reasons"]
    path.write_text(json.dumps(old))
    rm.main(["--stage", "energy", "--out-dir", str(tiny_results)])
    again = json.loads(path.read_text())["convergence"]
    assert again["rhat_phi_prime"] == pytest.approx(conv["rhat_phi_prime"]) and again["ess_phi_prime"] == pytest.approx(conv["ess_phi_prime"])
    assert again["converged"] == conv["converged"] and again["unconverged_reasons"] == conv["unconverged_reasons"]
    rm.main(["--stage", "collect", "--out-dir", str(tiny_results)])
    mj = json.loads((tiny_results / "mcmc.json").read_text())
    row = next(r for r in mj["table"] if r["name"] == "g0.25_eps2_matern_tf")
    assert row["config"] == "g0.25_eps2_matern" and row["run_tag"] == "_tf" and "ess_phi_prime" in row


def test_verdict_rule_follows_the_two_prior_agreement_of_design_11_3():
    assert rl.classify(0.99, 0.96) == "present" and rl.classify(0.02, 0.0) == "absent"
    assert rl.classify(0.99, 0.94) == "prior-dependent" and rl.classify(0.06, 0.01) == "prior-dependent" and rl.classify(0.6, 0.6) == "prior-dependent"
    assert rl.classify(0.99, None) == "unrepresentable" and rl.classify(None, None) == "unrepresentable"
    rows = []
    for g, pm, pt in ((0.5, 0.99, 0.99), (0.2, 0.99, 0.97), (0.1, 0.97, 0.9), (0.05, 0.99, 0.99)):  # a gap that is "present" below the first prior-dependent one does not count
        for prior, p in (("matern", pm), ("tv", pt)):
            rows.append({"eps_r": 2.0, "g": g, "prior": prior, "lam_scale": None if prior == "matern" else 1.0, "map_ok": True, "p_valley_region": p, "p_valley_point": p / 2})
    table, star = rl.verdict_table(rows)
    assert [t["verdict"] for t in table] == ["present", "prior-dependent", "present", "present"] and star == {"2": 0.2}
    assert table[0]["g"] == 0.05 and table[1]["verdict_point"] == "prior-dependent"  # the table is ordered by gap
    rows[0]["map_ok"] = False  # a failed MAP leaves no verdict for its configuration
    assert [t["g"] for t in rl.verdict_table(rows)[0]] == [0.05, 0.1, 0.2]


def test_refeature_recomputes_the_probabilities_for_other_region_definitions(tiny_results):
    base = TINY + ["--out-dir", str(tiny_results)]
    rl.main(["--stage", "refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)
    sens = json.loads((tiny_results / "laplace_sensitivity.json").read_text())
    assert [v["key"] for v in sens["variants"]] == ["band1_core0.5", "band0.5_core0.5"]
    stored = json.loads(rl.config_paths(tiny_results, "g0.25_eps2_matern")[0].read_text())["features"]["region"]["p_valley"]
    full = sens["configurations"]["g0.25_eps2_matern"]["band1_core0.5"]
    assert full["p_valley_region"] == pytest.approx(stored, abs=1e-8) and full["n_strip"] == 2 * 4  # the baseline variant reproduces the stored probability
    half = sens["configurations"]["g0.25_eps2_matern"]["band0.5_core0.5"]
    assert half["n_strip"] == 2 * 2 and half["strip_mean"] != full["strip_mean"]
    assert sens["configurations"]["g0.0125_eps2_matern"]["band1_core0.5"]["region_status"] == "unrepresentable on this grid"
    assert set(sens["per_variant"]["band1_core0.5"]["narrowest_gap_present_for_all_wider"]) == {"2"}
    before = (tiny_results / "laplace_sens" / "g0.25_eps2_matern.json").stat().st_mtime_ns
    rl.main(["--stage", "refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)
    assert (tiny_results / "laplace_sens" / "g0.25_eps2_matern.json").stat().st_mtime_ns == before  # resumable per configuration


def test_descend_then_refeature_writes_the_sensitivity_files_of_its_own_configurations_only(tiny_results):
    """The batch jobs of the lambda and SNR checks end with the refeature of the configurations they computed (``--then-refeature``), without touching the files of
    other configurations and without merging laplace_sensitivity.json (concurrent jobs would write it at the same time)."""
    base = TINY + ["--out-dir", str(tiny_results)]
    rl.main(["--stage", "refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)
    before = {p.name: p.stat().st_mtime_ns for p in (tiny_results / "laplace_sens").glob("*.json")}
    merged = (tiny_results / "laplace_sensitivity.json").stat().st_mtime_ns
    rl.main(["--stage", "descend", "--eps-r", "2", "--prior", "tv", "--gaps", "0.25", "--lam-scale", "5", "--then-refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)
    new = tiny_results / "laplace_sens" / "g0.25_eps2_tv_x5.json"
    assert new.exists() and set(json.loads(new.read_text())["variants"]) == {"band1_core0.5", "band0.5_core0.5"}
    assert {p.name: p.stat().st_mtime_ns for p in (tiny_results / "laplace_sens").glob("*.json") if p.name in before} == before  # the others are untouched
    assert (tiny_results / "laplace_sensitivity.json").stat().st_mtime_ns == merged  # no merge from inside the job
    with pytest.raises(SystemExit):  # an unknown name in --names is refused
        rl.main(["--stage", "refeature", "--names", "g9_eps2_tv"] + base)
    rl.main(["--stage", "refeature", "--names", "g0.25_eps2_tv_x5", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)  # all exist: nothing recomputed, merged
    assert "g0.25_eps2_tv_x5" in json.loads((tiny_results / "laplace_sensitivity.json").read_text())["configurations"]


def test_null_control_scores_the_posterior_of_a_closed_gap_with_open_gap_strips(tiny_results):
    base = TINY + ["--out-dir", str(tiny_results)]
    variants = ["--band-fracs", "1", "0.5", "--core-fracs", "0.5", "--strip-gaps", "0.25"]
    rl.main(["--stage", "refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)  # the open-gap side of the comparison (resumable)
    rl.main(["--stage", "null", "--eps-r", "2", "--prior", "matern"] + variants + base)
    rl.main(["--stage", "null", "--eps-r", "2", "--prior", "tv"] + variants + base)
    rec = json.loads(rl.null_paths(tiny_results, "g0_eps2_matern")[0].read_text())
    assert rec["g_true"] == 0.0 and rec["map_ok"] and set(rec["variants"]) == {"g0.25_band1_core0.5", "g0.25_band0.5_core0.5"}
    v = rec["variants"]["g0.25_band1_core0.5"]
    assert v["strip_gap"] == 0.25 and v["n_strip"] == 2 * 4 and 0.0 <= v["p_valley_region"] <= 1.0 and v["delta"] == pytest.approx(0.5 * np.log(2.0))
    z = np.load(rl.null_paths(tiny_results, "g0_eps2_matern")[1])
    assert z["theta_hat"].shape == (12, 12) and z["theta_true"].max() <= np.log(2.0) + 1e-12
    # the data of a closed gap differ from those of an open one (the touching disks, not the g = 0.25 object)
    open_rec = json.loads(rl.config_paths(tiny_results, "g0.25_eps2_matern")[0].read_text())
    assert rec["problem"]["d_obs_sha256"] != open_rec["problem"]["d_obs_sha256"]
    out = rl.main(["--stage", "null-collect"] + base)
    assert (tiny_results / "null.json").exists() and (tiny_results / "null_control.png").exists() and (tiny_results / "resolution_overlay.png").exists()
    assert out["smith_gap_30db"] == {}  # no extended CRB sweep in this directory
    rows = [r for r in out["rows"] if r["prior"] == "matern" and r["variant"] == "band1_core0.5"]
    assert len(rows) == 1 and rows[0]["p_null"] == pytest.approx(v["p_valley_region"]) and rows[0]["p_open"] is not None
    assert rows[0]["p_difference"] == pytest.approx(rows[0]["p_open"] - rows[0]["p_null"])
    assert rows[0]["discriminates"] == bool(rows[0]["p_open"] > 0.95 and rows[0]["p_null"] < 0.05)
    assert set(out["narrowest_discriminating_gap"]) == {"eps2_band1_core0.5", "eps2_band0.5_core0.5"}
    before = rl.null_paths(tiny_results, "g0_eps2_matern")[0].stat().st_mtime_ns
    rl.main(["--stage", "null", "--eps-r", "2", "--prior", "matern"] + variants + base)
    assert rl.null_paths(tiny_results, "g0_eps2_matern")[0].stat().st_mtime_ns == before  # a finished null configuration is skipped


def test_probabilities_whose_pixelized_truth_fails_are_withheld_in_the_tables(tiny_results, tmp_path):
    """Records written before the rule 'a truth that fails the feature has no probability' keep the number in the file; the tables (table_row, the sensitivity merge and
    the null control) withhold it, status 'unrepresentable on this grid (the pixelized truth fails the feature)', and keep the stored value as *_raw."""
    import shutil

    from bayes_scatter import gapvalley as gv

    rec = json.loads(rl.config_paths(tiny_results, "g0.25_eps2_matern")[0].read_text())
    ok = rl.table_row(rec)
    assert ok["p_valley_region"] is not None and ok["region_status"] == "ok" and ok["p_valley_region_raw"] == ok["p_valley_region"]
    rec["truth"]["region"]["holds"] = False
    bad = rl.table_row(rec)
    assert bad["p_valley_region"] is None and bad["p_valley_region_raw"] == ok["p_valley_region"] and bad["region_status"] == gv.UNREPRESENTABLE_TRUTH
    assert rl.truth_fails(False) and not rl.truth_fails(True) and not rl.truth_fails(None)
    # sensitivity merge and null control on a copy of the directory with one stored variant flagged as failing
    copy = tmp_path / "copy"
    shutil.copytree(tiny_results, copy, ignore=shutil.ignore_patterns("mcmc", "allstarts"))
    base = TINY + ["--out-dir", str(copy)]
    variants = ["--band-fracs", "1", "0.5", "--core-fracs", "0.5", "--strip-gaps", "0.25"]
    rl.main(["--stage", "refeature", "--band-fracs", "1", "0.5", "--core-fracs", "0.5"] + base)
    rl.main(["--stage", "null", "--eps-r", "2", "--prior", "matern"] + variants + base)
    rl.main(["--stage", "null", "--eps-r", "2", "--prior", "tv"] + variants + base)
    path = copy / "laplace_sens" / "g0.25_eps2_matern.json"
    sens_rec = json.loads(path.read_text())
    stored = sens_rec["variants"]["band0.5_core0.5"]["p_valley_region"]
    assert stored is not None
    sens_rec["variants"]["band0.5_core0.5"]["truth_holds_region"] = False
    path.write_text(json.dumps(sens_rec))
    rl.merge_sensitivity(copy, [(1.0, 0.5), (0.5, 0.5)])
    merged = json.loads((copy / "laplace_sensitivity.json").read_text())
    cell = merged["configurations"]["g0.25_eps2_matern"]["band0.5_core0.5"]
    assert cell["p_valley_region"] is None and cell["p_valley_region_raw"] == stored and cell["region_status"] == gv.UNREPRESENTABLE_TRUTH
    assert merged["configurations"]["g0.25_eps2_matern"]["band1_core0.5"]["p_valley_region"] is not None  # the other variant is untouched
    row = next(t for t in merged["per_variant"]["band0.5_core0.5"]["verdicts"] if t["eps_r"] == 2.0 and t["g"] == 0.25)
    assert row["p_matern"] is None and row["verdict"] == "unrepresentable"  # the withheld Matern probability leaves the gap without a verdict
    out = rl.main(["--stage", "null-collect"] + base)
    rows = [r for r in out["rows"] if r["prior"] == "matern" and r["variant"] == "band0.5_core0.5"]
    assert len(rows) == 1 and rows[0]["p_open"] is None and rows[0]["p_null"] is None and rows[0]["discriminates"] is None and rows[0]["withheld_truth_fails"] is True
    assert rows[0]["p_open_raw"] == stored and rows[0]["p_null_raw"] is not None
    other = [r for r in out["rows"] if r["prior"] == "matern" and r["variant"] == "band1_core0.5"]
    assert other[0]["p_open"] is not None and other[0]["withheld_truth_fails"] is False


def test_smith_gap_is_the_log_log_crossing_of_sigma_crb_with_g(tmp_path):
    def rec(eps, g, sigma, n=128):
        return {"grid": {"n": n}, "params": {"g": g, "eps_r": eps}, "crb_g_unknown": sigma}

    # sigma_CRB(g) = 3e-4 (flat): the criterion holds at g = 3e-4; the n = 256 records and other contrasts must not be mixed in
    recs = {f"a{i}": rec(2.0, g, 3e-4) for i, g in enumerate((1e-4, 2e-4, 5e-4, 1e-3))}
    recs["b"] = rec(2.0, 4e-4, 9e-4, n=256)
    recs.update({f"c{i}": rec(4.0, g, 1e-5) for i, g in enumerate((1e-4, 2e-4))})  # sigma < g everywhere: not bracketed
    (tmp_path / "smith_limit").mkdir()
    (tmp_path / "smith_limit" / "crb.json").write_text(json.dumps({"records": recs}))
    out = rl.smith_gaps(tmp_path)
    assert out["2"] == pytest.approx(3e-4, rel=1e-9) and out["4"] is None and out["1.1"] is None
    assert rl.smith_gaps(tmp_path / "missing") == {}


def test_chol_solver_and_functional_helpers():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 30))
    H = A @ A.T + 30 * np.eye(30)
    L = np.linalg.cholesky(H)
    B = rng.standard_normal((30, 4))
    np.testing.assert_allclose(rl.chol_solver(L)(B), np.linalg.solve(H, B), rtol=1e-10)
    np.testing.assert_allclose(rl.chol_solver(L)(B[:, 0]), np.linalg.solve(H, B[:, 0]), rtol=1e-10)
    theta = rng.standard_normal((3, 50, 4, 4))
    w = rng.standard_normal(16)
    np.testing.assert_allclose(rm.functional_series(theta, w), np.einsum("cdij,ij->cd", theta, w.reshape(4, 4)), rtol=1e-12)
    st = rm.series_stats(theta[:, :, 0, 0])
    assert st["mean"] == pytest.approx(theta[:, :, 0, 0].mean()) and st["ess"] > 0 and st["rhat"] > 0.9 and len(st["per_chain_mean"]) == 3
    assert rl.config_name(0.1, 2.0, "matern") == "g0.1_eps2_matern" and rl.config_name(0.1, 2.0, "tv", 1.0) == "g0.1_eps2_tv"
    assert rl.config_name(0.1, 2.0, "tv", 3.0) == "g0.1_eps2_tv_x3" and rl.config_name(0.1, 2.0, "matern", 3.0) == "g0.1_eps2_matern"
    with pytest.raises(SystemExit):
        rl.lambda_star_of("/nonexistent/dir", 2.0)


# ---------------------------------------------------------------------------------------------
# make_m6_jobs.py
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("batch,count", [("laplace1", 6), ("laplace2", 3), ("sens", 2)])
def test_job_files_are_valid_for_batch_run_and_name_the_files_the_scripts_write(batch, count, tmp_path):
    jobs = mj.laplace_jobs(batch)
    assert len(jobs) == count and len({j["name"] for j in jobs}) == count
    path = tmp_path / f"{batch}.json"
    path.write_text(json.dumps(jobs))
    parsed = br.load_jobs(str(path))  # batch_run's own validation
    assert len(parsed) == count and all(j.threads == 1 for j in parsed)
    for job in jobs:
        args = rl.parse_args(job["cmd"][1:])
        if args.stage == "descend":  # one chain over the gaps; the marker is the narrowest gap
            name = rl.config_name(min(args.gaps), args.eps_r, args.prior, args.lam_scale)
            assert job["done_marker"] == f"results/resolution/laplace/{name}.json" and job["name"].startswith(f"m6_descend_eps{args.eps_r:g}_{args.prior}")
            assert sorted(args.gaps) == sorted(mj.GAPS)
        else:
            assert args.stage == "lambda" and max(args.lams) == mj.LAM_MAX
            assert job["done_marker"] == rl.lambda_path("results/resolution", args.eps[0], args.lambda_gap, max(args.lams)).as_posix()
    if batch == "laplace1":
        assert sum(1 for j in jobs if "matern" in j["name"]) == 3 and sum(1 for j in jobs if j["name"].startswith("m6_lambda")) == 3
    if batch == "sens":
        scales = sorted({rl.parse_args(j["cmd"][1:]).lam_scale for j in jobs})
        assert scales == pytest.approx([1.0 / 3.0, 3.0]) and all(rl.parse_args(j["cmd"][1:]).eps_r == 2.0 for j in jobs)


def test_mcmc_job_file_names_the_segments_the_script_writes(tmp_path):
    jobs = mj.mcmc_jobs([(0.05, 2.0, "matern"), (0.025, 2.0, "tv")], chains=3, segments=2, samples=50, warmup=20)
    assert len(jobs) == 6 and len({j["name"] for j in jobs}) == 6
    path = tmp_path / "m.json"
    path.write_text(json.dumps(jobs))
    assert len(br.load_jobs(str(path))) == 6
    for job in jobs:
        args = rm.parse_args(job["cmd"][1:])
        assert args.stage == "chain" and (args.first, args.last, args.samples, args.warmup, args.chains) == (0, 1, 50, 20, 3)
        run = rm.run_dir(args)
        assert job["done_marker"] == f"results/resolution/mcmc/{run.name}/chain_{args.chain_id}_seg1.npz"
        assert rm.rpc.segment_path(Path("results/resolution/mcmc") / run.name, args.chain_id, 1).as_posix() == job["done_marker"]
    assert [rm.parse_args(j["cmd"][1:]).prior for j in jobs] == ["matern"] * 3 + ["tv"] * 3
    own = mj.mcmc_jobs([(0.05, 2.0, "matern", 3, 100, 60), (0.05, 2.0, "tv", 2, 50, 50)], chains=2, segments=9, samples=9, warmup=9)
    a = [rm.parse_args(j["cmd"][1:]) for j in own]
    assert [(x.prior, x.last, x.samples, x.warmup) for x in a] == [("matern", 2, 100, 60)] * 2 + [("tv", 1, 50, 50)] * 2
    assert own[0]["done_marker"].endswith("chain_0_seg2.npz") and own[3]["done_marker"].endswith("g0.05_eps2_tv/chain_1_seg1.npz")


def test_extra_job_file_names_the_files_the_null_and_truth_chain_jobs_write(tmp_path):
    jobs = mx.null_jobs() + mx.truth_jobs()
    assert len(jobs) == 10 and len({j["name"] for j in jobs}) == 10 and all(j["threads"] == 1 for j in jobs)
    path = tmp_path / "x.json"
    path.write_text(json.dumps(jobs))
    assert len(br.load_jobs(str(path))) == 10
    for job in mx.null_jobs():
        a = rl.parse_args(job["cmd"][1:])
        assert a.stage == "null" and a.neighbor_gap == 0.025 and a.all_starts
        assert job["done_marker"] == rl.null_paths("results/resolution", rl.config_name(0.0, a.eps_r, a.prior))[0].as_posix()
    done = {}
    for job in mx.truth_jobs():
        a = rm.parse_args(job["cmd"][1:])
        assert a.stage == "chain" and (a.chain_id, a.chains, a.start) == (4, 5, "truth")
        run = rm.run_dir(a)
        assert job["done_marker"] == f"results/resolution/mcmc/{run.name}/chain_4_seg{a.last}.npz"
        done[run.name] = (a.last + 1, a.samples, a.warmup)
    # the same segments, samples and warmup as the four overdispersed chains of the spot check (m6_mcmc*.json), so that the five chains can be analyzed together
    # (the TV chain of g = 0.025 was continued by m6_mcmc3.json: the segments of the last job file that names a configuration are the ones on disk)
    final = {}
    for jobfile in ("m6_mcmc1", "m6_mcmc2", "m6_mcmc3"):
        for j in json.loads((SCRIPTS / "jobs" / f"{jobfile}.json").read_text()):
            a = rm.parse_args(j["cmd"][1:])
            if a.chain_id == 0:
                final[rm.run_dir(a).name] = (a.last + 1, a.samples, a.warmup)
    assert {k: final[k] for k in done} == done


def test_cost_script_measures_the_crb_grid_and_summarizes_recorded_wall_times(tiny_results):
    out = rcost.main(["--crb-grids", "16", "--eps", "2", "--reps", "1", "--skip-laplace-grid", "--out-dir", str(tiny_results), "--out", "cost_test.json"])
    rec = out["crb_grid"]["n16_eps2"]
    assert rec["n"] == 16 and rec["forward"]["min_s"] > 0 and set(rec["tangent_per_parameter"]) == set(rc.sh.PARAM_NAMES)
    assert rec["tangents_total"]["min_s"] > 0 and rec["born_derivatives"]["min_s"] > 0 and rec["peak_rss_gb"] > 0
    assert (tiny_results / "cost_test.json").exists()
    recorded = out["recorded_configurations"]
    assert set(recorded) == {"matern_eps2", "tv_eps2"}
    assert recorded["matern_eps2"]["n_configurations"] == 2 and recorded["tv_eps2"]["n_configurations"] == 1  # the lambda-scaled run is not counted
    assert recorded["matern_eps2"]["total_s"]["min"] <= recorded["matern_eps2"]["total_s"]["median"] <= recorded["matern_eps2"]["total_s"]["max"]
