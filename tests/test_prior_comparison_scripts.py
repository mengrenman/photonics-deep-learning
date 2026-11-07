"""End-to-end tests of scripts/run_prior_comparison.py and scripts/compare_priors.py on a tiny problem."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

import bayes_scatter  # noqa: F401
from bayes_scatter import thetaspace as ts

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import compare_priors as cp  # noqa: E402
import run_prior_comparison as rpc  # noqa: E402

TINY = ["--n", "8", "--side", "2", "--n-tx", "6", "--n-rx", "8", "--chains", "2", "--warmup", "10", "--samples", "20",
        "--start-maxiter", "10", "--map-maxiter", "30", "--prior-chains", "2", "--prior-warmup", "20",
        "--prior-samples", "40", "--verify-columns", "2"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("prior_comparison")
    out = {}
    for name, extra in {"qg": ["--prior", "qggmrf", "--sigma-g", "0.1"], "mat": ["--prior", "matern"]}.items():
        rpc.main(TINY + extra + ["--out", str(root / name)])
        out[name] = root / name
    return {"root": root, **out}


def test_run_writes_the_documented_outputs(runs):
    for name in ("qg", "mat"):
        d = runs[name]
        assert {p.name for p in d.iterdir()} == {"summary.json", "chains.npz", "maps.png", "traces.png"}
        s = json.loads((d / "summary.json").read_text())
        for key in ("args", "data", "operating_point", "prior", "start", "map", "laplace", "sampler", "phi_physical",
                    "convergence", "posterior", "z", "metrics", "features", "timings"):
            assert key in s
        assert len(s["data"]["d_obs_sha256"]) == 64
        assert s["laplace"]["jacobian_check_rel_err"] < 1e-8
        assert set(s["metrics"]) == {"err_model", "err_chi", "err_data"}
        assert set(s["features"]["probabilities"]) == {"gap_valley", "ring_hole", "ring_continuity"}
        assert set(s["convergence"]) >= {"rhat_phi", "rhat_pixel_max", "ess_pixel_min", "converged", "thresholds"}
        assert s["convergence"]["thresholds"] == {"rhat_phi": 1.01, "rhat_pixel": 1.05, "ess_pixel_min": 200.0}
        assert s["sampler"]["max_solver_residual_final_states"] < 1e-8
        z = np.load(d / "chains.npz")
        assert z["theta"].shape == (2, 20, 8, 8) and z["theta"].dtype == np.float32
        assert z["welford_mean"].shape == (2, 2, 8, 8) and z["phi_physical"].shape == (2, 20)
        assert np.all(np.isfinite(z["phi_physical"])) and np.all(np.isfinite(z["theta_hat"]))
    assert json.loads((runs["qg"] / "summary.json").read_text())["prior"]["prior_summary"]["method"].startswith("HMC")
    assert json.loads((runs["mat"] / "summary.json").read_text())["prior"]["prior_summary"]["theta_std"]["mean"] == \
        pytest.approx(0.5)


def test_every_prior_sees_the_same_data_and_the_hash_follows_the_seed(runs):
    h = [json.loads((runs[k] / "summary.json").read_text())["data"]["d_obs_sha256"] for k in ("qg", "mat")]
    assert h[0] == h[1]
    a0 = rpc.parse_args(TINY + ["--prior", "matern", "--out", "x"])
    assert rpc.build_data(a0).sha == h[0]  # deterministic in the seed
    a1 = rpc.parse_args(TINY + ["--prior", "matern", "--out", "x", "--seed", "1"])
    assert rpc.build_data(a1).sha != h[0]
    a2 = rpc.parse_args(TINY + ["--prior", "qggmrf", "--sigma-g", "0.3", "--chains", "3", "--out", "y"])
    assert rpc.build_data(a2).sha == h[0]  # prior and chain settings do not touch the data


def test_stored_physical_misfit_is_the_misfit_of_the_stored_draws(runs):
    d = runs["qg"]
    s = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    args = SimpleNamespace(**s["args"])
    prob = rpc.build_data(args)
    prior = rpc.build_prior(args, prob.setup.grid)
    target = ts.ThetaTarget(prob.forward, prior, prob.d_obs, prob.s)
    for c, k in ((0, 3), (1, 17)):
        phi, _ = target.misfit(z["theta"][c, k].astype(np.float64))
        np.testing.assert_allclose(float(phi), z["phi_physical"][c, k], atol=0.1)
    # the sampler's own trace is a different quantity (Phi' = U - |zeta|^2/2), stored under a separate name
    assert not np.allclose(z["phi_prime"], z["phi_physical"])
    assert "NOT the misfit" in s["sampler"]["phi_prime_note"]
    np.testing.assert_allclose(s["phi_physical"]["mean"], z["phi_physical"].mean())


def test_compare_priors_tabulates_and_checks_the_data(runs):
    out = runs["root"] / "cmp"
    res = cp.main([str(runs["qg"]), str(runs["mat"]), "--out", str(out), "--labels", "qggmrf", "matern"])
    assert (out / "comparison.json").exists() and (out / "comparison_features.png").exists()
    assert (out / "comparison_mean_std.png").exists() and (out / "comparison_disagreement.png").exists()
    assert res["data_check"]["identical"] and set(res["runs"]) == {"qggmrf", "matern"}
    pair = res["pairs"]["qggmrf vs matern"]
    assert 0.0 <= pair["theta"]["frac_gt_2"] <= 1.0 and pair["theta"]["max"] >= pair["theta"]["mean"]
    npz = np.load(out / "comparison.npz")
    s_a, s_b = (np.sqrt(np.load(runs[k] / "chains.npz")["welford_var"].mean(axis=0)[0]) for k in ("qg", "mat"))
    assert npz["disagreement_theta"].shape == (1, 8, 8) and np.all(npz["disagreement_theta"] >= 0.0)
    # with two runs, at most one of which converged in 20 draws, no feature gets a verdict
    assert res["n_converged"] < 2
    for verdict in res["features"]["verdict"].values():
        assert verdict.startswith("no verdict")
    assert s_a.shape == s_b.shape == (8, 8)


def test_compare_priors_refuses_runs_with_different_data(runs):
    bad = runs["root"] / "bad"
    shutil.copytree(runs["mat"], bad)
    s = json.loads((bad / "summary.json").read_text())
    s["data"]["d_obs_sha256"] = "0" * 64
    (bad / "summary.json").write_text(json.dumps(s))
    z = dict(np.load(bad / "chains.npz"))
    z["d_obs"] = z["d_obs"] * 1.01  # 1% different data
    np.savez_compressed(bad / "chains.npz", **z)
    with pytest.raises(SystemExit, match="data differ"):
        cp.main([str(runs["qg"]), str(bad), "--out", str(runs["root"] / "cmp_bad")])
    # a different hash with arrays equal to round-off is accepted with a warning
    z["d_obs"] = z["d_obs"] / 1.01 * (1 + 1e-12)
    np.savez_compressed(bad / "chains.npz", **z)
    res = cp.main([str(runs["qg"]), str(bad), "--out", str(runs["root"] / "cmp_ok")])
    assert not res["data_check"]["identical"] and res["data_check"]["max_rel_diff"] < 1e-9
    # different data arguments are refused outright
    s["args"]["seed"] = 5
    (bad / "summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="different data arguments"):
        cp.main([str(runs["qg"]), str(bad), "--out", str(runs["root"] / "cmp_bad2")])


# ---- staged execution: prepare / chain (one process per chain, with continuation) / finalize ----


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """The tiny Matern problem run in stages (2 chains, 2 segments of 20 samples) and the same in one process."""
    root = tmp_path_factory.mktemp("staged")
    common = TINY + ["--prior", "matern", "--samples", "20", "--thin", "1"]
    one = root / "one"
    rpc.main(common + ["--out", str(one)])
    st = root / "staged"
    rpc.main(common + ["--stage", "prepare", "--out", str(st)])
    for seg in (0, 1):
        for c in (0, 1):
            rpc.main(common + ["--stage", "chain", "--chain-id", str(c), "--segment", str(seg), "--out", str(st)])
    fin = root / "fin"
    shutil.copytree(st, fin)
    rpc.main(common + ["--stage", "finalize", "--max-segments", "1", "--out", str(fin)])  # segment 0 only
    rpc.main(common + ["--stage", "finalize", "--out", str(st)])  # both segments
    return {"root": root, "one": one, "staged": st, "first_segment": fin, "common": common}


def test_skip_segments_discards_the_first_segments_as_burn_in_and_keeps_the_cost(staged, tmp_path):
    full = json.loads((staged["staged"] / "summary.json").read_text())
    d = tmp_path / "skip1"
    shutil.copytree(staged["staged"], d)
    rpc.main(staged["common"] + ["--stage", "finalize", "--skip-segments", "1", "--out", str(d)])
    s = json.loads((d / "summary.json").read_text())
    z_full, z = np.load(staged["staged"] / "chains.npz"), np.load(d / "chains.npz")
    assert z["theta"].shape[1] == 20 and z_full["theta"].shape[1] == 40
    np.testing.assert_array_equal(z["theta"], z_full["theta"][:, 20:])  # the draws of segment 1 only
    assert s["sampler"]["segments"] == 1 and s["sampler"]["samples"] == 20 and s["sampler"]["samples_per_segment"] == [20]
    assert s["sampler"]["warmup"] == full["sampler"]["warmup"] + 20  # the dropped recorded draws count as warmup
    assert "discarded as burn-in" in s["sampler"]["starts"] and s["sampler"]["starts"].startswith(full["sampler"]["starts"])
    np.testing.assert_allclose(s["sampler"]["wall_s_per_chain"], full["sampler"]["wall_s_per_chain"])  # nothing computed is forgotten
    assert s["sampler"]["n_phi_evals_per_chain"] == full["sampler"]["n_phi_evals_per_chain"]
    # the Welford moments follow the kept draws
    np.testing.assert_allclose(z["welford_mean"][:, 0], z["theta"].mean(axis=1), atol=1e-6)  # the stored draws are float32
    with pytest.raises(SystemExit, match="leaves no segment"):
        rpc.main(staged["common"] + ["--stage", "finalize", "--skip-segments", "2", "--out", str(d)])
    assert rpc.discard_burn_in([{"a": 1}], 0) == [{"a": 1}]


def test_staged_run_with_one_segment_reproduces_the_single_process_run(staged):
    a, b = np.load(staged["one"] / "chains.npz"), np.load(staged["first_segment"] / "chains.npz")
    for key in ("theta", "phi_physical", "welford_mean", "welford_var", "accept", "step", "theta_hat"):
        np.testing.assert_allclose(b[key], a[key], rtol=1e-9, atol=1e-10, err_msg=key)
    sa = json.loads((staged["one"] / "summary.json").read_text())
    sb = json.loads((staged["first_segment"] / "summary.json").read_text())
    assert sa["data"]["d_obs_sha256"] == sb["data"]["d_obs_sha256"]
    assert sb["sampler"]["segments"] == 1 and sb["sampler"]["samples"] == 20
    assert sb["convergence"]["rhat_phi"] == pytest.approx(sa["convergence"]["rhat_phi"], rel=1e-9)


def test_continuation_extends_the_chains_and_merges_the_moments(staged):
    st = staged["staged"]
    assert {p.name for p in st.iterdir()} >= {"prepared.json", "prepared.npz", "chain_0_seg0.npz", "chain_1_seg1.npz",
                                              "summary.json", "chains.npz"}
    s0, s1 = (rpc.load_segment(rpc.segment_path(st, 0, j)) for j in (0, 1))
    assert s0["n_warmup"] == 10 and s1["n_warmup"] == 0 and s1["n_samples"] == 20
    assert s1["step"] == pytest.approx(s0["step"]) and s1["step0"] == pytest.approx(s0["step"])  # step frozen
    assert not np.allclose(s0["theta"], s1["theta"])  # a fresh stream, not a replay
    z = np.load(st / "chains.npz")
    assert z["theta"].shape == (2, 40, 8, 8) and z["phi_physical"].shape == (2, 40)
    # thin = 1 stores every post-warmup state, so the merged Welford moments are the moments of the stored draws
    theta = z["theta"].astype(np.float64)
    np.testing.assert_allclose(z["welford_mean"][:, 0], theta.mean(axis=1), atol=1e-6)
    np.testing.assert_allclose(z["welford_var"][:, 0], theta.var(axis=1, ddof=1), rtol=1e-4, atol=1e-8)
    s = json.loads((st / "summary.json").read_text())
    assert s["sampler"]["segments"] == 2 and s["sampler"]["samples"] == 40 and s["sampler"]["warmup"] == 10
    assert s["sampler"]["n_phi_evals_per_chain"] == (1 + 30 * 10) + (1 + 20 * 10)  # (10 + 20) and 20 trajectories


def test_staged_stages_refuse_a_different_posterior_and_unequal_chains(staged, tmp_path):
    common = staged["common"]
    with pytest.raises(SystemExit, match="different data or prior arguments"):
        rpc.main(common + ["--stage", "chain", "--chain-id", "0", "--sigma", "0.7", "--out", str(staged["staged"])])
    half = tmp_path / "half"
    shutil.copytree(staged["staged"], half)
    rpc.segment_path(half, 1, 1).unlink()  # chain 1 has one segment, chain 0 two
    with pytest.raises(SystemExit, match="same nonzero number of segments"):
        rpc.main(common + ["--stage", "finalize", "--out", str(half)])
    with pytest.raises(SystemExit):  # chain stage needs a chain id inside 0 .. chains - 1
        rpc.parse_args(common + ["--stage", "chain", "--chain-id", "2", "--out", "x"])


# ---- custom mass center, custom starts, and segment files from before those options existed ----


def test_custom_mass_center_and_starts_are_used_stored_and_inherited_by_continuations(tmp_path):
    common = TINY + ["--prior", "matern", "--samples", "20"]
    d = tmp_path / "run"
    rpc.main(common + ["--stage", "prepare", "--out", str(d)])
    theta_hat = np.load(d / "prepared.npz")["theta_hat"]
    mass_theta = theta_hat + 0.05 * np.sin(np.arange(64.0).reshape(8, 8))
    starts = np.stack([theta_hat + 0.02, theta_hat - 0.02])
    np.save(tmp_path / "mass.npy", mass_theta)
    np.save(tmp_path / "starts.npy", starts)
    extra = ["--mass-theta", str(tmp_path / "mass.npy"), "--start-theta", str(tmp_path / "starts.npy")]
    for c in (0, 1):
        # a vanishing step makes the first recorded state the start itself (one trajectory of length 1e-5)
        rpc.main(common + extra + ["--stage", "chain", "--chain-id", str(c), "--warmup", "0", "--step0", "1e-6", "--jitter",
                                   "0", "--samples", "20", "--out", str(d)])
    for c in (0, 1):
        seg = rpc.load_segment(rpc.segment_path(d, c, 0))
        np.testing.assert_allclose(seg["mass_theta"], mass_theta)
        assert np.abs(seg["theta"][0] - starts[c]).max() < 1e-3
        assert seg["starts"] == "start-theta file"
        np.testing.assert_allclose(seg["theta_final"], seg["theta"][-1], atol=1e-3)
    # the continuation takes the coordinates of its predecessor without being told
    for c in (0, 1):
        rpc.main(common + ["--stage", "chain", "--chain-id", str(c), "--segment", "1", "--out", str(d)])
        seg1 = rpc.load_segment(rpc.segment_path(d, c, 1))
        np.testing.assert_allclose(seg1["mass_theta"], mass_theta)
    other = tmp_path / "other.npy"
    np.save(other, mass_theta + 1e-3)
    with pytest.raises(SystemExit, match="differs from the center"):
        rpc.main(common + ["--stage", "chain", "--chain-id", "0", "--segment", "1", "--mass-theta", str(other),
                           "--out", str(d)])
    s = rpc.main(common + ["--stage", "finalize", "--out", str(d)])
    assert s["sampler"]["segments"] == 2 and "custom theta" in s["sampler"]["mass"]
    assert s["sampler"]["mass_theta_rms_from_map"] > 0.01 and s["sampler"]["starts"] == "start-theta file"
    assert s["sampler"]["max_solver_residual_final_states"] < 1e-8
    # the Laplace reference stays that of the MAP, whatever the chains' coordinates
    assert s["laplace"] == json.loads((d / "prepared.json").read_text())["laplace"]


def test_finalize_accepts_segment_files_written_before_mass_and_final_theta_were_stored(staged, tmp_path):
    old = tmp_path / "old"
    shutil.copytree(staged["staged"], old)
    for c in (0, 1):
        for j in (0, 1):
            p = rpc.segment_path(old, c, j)
            seg = rpc.load_segment(p)
            for key in ("theta_final", "mass_theta", "starts"):
                seg.pop(key, None)
            rpc.save_segment(p, seg)
    s_old = rpc.main(staged["common"] + ["--stage", "finalize", "--out", str(old)])
    s_new = json.loads((staged["staged"] / "summary.json").read_text())
    assert s_old["sampler"]["mass"] == "Laplace mass at the MAP"
    assert s_old["sampler"]["max_solver_residual_final_states"] == pytest.approx(
        s_new["sampler"]["max_solver_residual_final_states"], abs=1e-9)
    np.testing.assert_allclose(np.load(old / "chains.npz")["theta"], np.load(staged["staged"] / "chains.npz")["theta"])


# ---- report scripts: edge analysis and tables ----


def test_edge_analysis_recomputes_the_summary_statistics_and_the_tables_print(runs, tmp_path, capsys):
    import edge_analysis as ea
    import prior_comparison_tables as pt

    res = ea.main(["qg", "mat", "--dir", str(runs["root"]), "--out", str(tmp_path)])
    for name in ("qg", "mat"):
        e = res[name]
        assert e["stored_draw_mean_vs_welford_max_abs"] < 1e-5  # thin = 1: the stored draws are every post-warmup state
        assert e["z_from_stored_draws"]["frac_abs_z_gt_3"] == pytest.approx(e["z_from_summary"]["frac_abs_z_gt_3"], abs=0.02)
        assert set(e["profile_by_signed_distance_to_edge_px"]) <= {str(b) for b in ea.BINS}
    assert (tmp_path / "edge_analysis.json").exists() and (tmp_path / "edge_profile.png").exists()
    chi = np.zeros((8, 8))
    chi[2:5, 2:5] = 1.0
    d = ea.signed_distance(chi)
    assert d[3, 3] == 2 and d[2, 2] == 1 and d[0, 0] == -3 and d[1, 3] == -1 and d[5, 3] == -1
    pt.main(["qg", "mat", "--dir", str(runs["root"])])
    out = capsys.readouterr().out
    assert "### feature probabilities" in out and "| qg |" in out and "| mat |" in out


# ---- review fixes: families and provisional verdicts, numerical data check, audits, thread cap ----


def _converged_copy(src, dst, p):
    """Copy of a tiny run directory with summary.json claiming convergence and every feature probability ``p``."""
    shutil.copytree(src, dst)
    s = json.loads((dst / "summary.json").read_text())
    s["convergence"]["converged"] = True
    for v in s["features"]["probabilities"].values():
        v["p"] = p
    (dst / "summary.json").write_text(json.dumps(s))
    return dst


def test_compare_priors_calls_matern_plus_qggmrf_provisional_and_counts_families(runs, tmp_path):
    a = _converged_copy(runs["qg"], tmp_path / "a", 0.99)
    b = _converged_copy(runs["mat"], tmp_path / "b", 0.99)
    res = cp.main([str(a), str(b), "--out", str(tmp_path / "cmp"), "--labels", "qggmrf_run", "matern_run"])
    f = res["features"]
    assert f["family"] == {"qggmrf_run": "qggmrf", "matern_run": "matern"}
    assert f["converged_families"] == ["matern", "qggmrf"] and f["design_minimum_met"] is False
    for v in f["verdict"].values():  # two converged posteriors agree, but there is no converged GMRF
        assert v.startswith("provisional (present)") and "gmrf" in v
    # the same two runs as a family pair that includes a GMRF would be data-determined
    s = json.loads((b / "summary.json").read_text())
    s["prior"]["kind"] = "gmrf"
    (b / "summary.json").write_text(json.dumps(s))
    res = cp.main([str(a), str(b), "--out", str(tmp_path / "cmp2"), "--labels", "qggmrf_run", "gmrf_run"])
    assert res["features"]["design_minimum_met"] is False  # q-GGMRF + GMRF still lacks the Matern of the minimum pair
    # an unconverged run that disagrees blocks the verdict
    c = _converged_copy(runs["mat"], tmp_path / "c", 0.5)
    sc = json.loads((c / "summary.json").read_text())
    sc["convergence"]["converged"] = False
    sc["prior"]["kind"] = "gmrf"
    (c / "summary.json").write_text(json.dumps(sc))
    d = _converged_copy(runs["mat"], tmp_path / "d", 0.99)
    sd = json.loads((d / "summary.json").read_text())
    sd["prior"]["kind"] = "matern"
    (d / "summary.json").write_text(json.dumps(sd))
    res = cp.main([str(a), str(d), str(c), "--out", str(tmp_path / "cmp3"), "--labels", "q", "m", "g"])
    for v in res["features"]["verdict"].values():
        assert v.startswith("not established") and "gmrf" in v


def test_compare_priors_reports_the_numerical_data_difference_and_the_thread_environment(runs, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(runs["mat"], bad)
    s = json.loads((bad / "summary.json").read_text())
    s["data"]["d_obs_sha256"] = "1" * 64
    s["data"]["thread_env"] = {"NPROC": "4"}
    (bad / "summary.json").write_text(json.dumps(s))
    z = dict(np.load(bad / "chains.npz"))
    z["d_obs"] = z["d_obs"] + 1e-9 * float(z["noise_std"])  # 1e-9 of the noise std: below the 1e-6 criterion
    np.savez_compressed(bad / "chains.npz", **z)
    res = cp.main([str(runs["qg"]), str(bad), "--out", str(tmp_path / "ok")])
    dc = res["data_check"]
    assert not dc["identical"] and 0.0 < dc["max_abs_diff_over_noise_std"] < 1e-6 and dc["tolerance"] == 1e-6
    assert dc["thread_env"][bad.name] == {"NPROC": "4"}
    z["d_obs"] = np.load(runs["mat"] / "chains.npz")["d_obs"] + 1e-3 * float(z["noise_std"])  # 1e-3 of s: refused
    np.savez_compressed(bad / "chains.npz", **z)
    with pytest.raises(SystemExit, match="data differ"):
        cp.main([str(runs["qg"]), str(bad), "--out", str(tmp_path / "refused")])


def test_a_hash_that_differs_only_by_the_thread_count_does_not_refuse_the_data(staged, tmp_path, capsys):
    """The review finding: d_obs bits depend on the XLA thread count, so equal data can have unequal hashes."""
    d = tmp_path / "prep"
    shutil.copytree(staged["staged"], d)
    args = rpc.parse_args(staged["common"] + ["--stage", "chain", "--chain-id", "0", "--out", str(d)])
    ctx = rpc.build_context(args)
    prepared = json.loads((d / "prepared.json").read_text())
    assert "thread_env" in prepared["data"] and "d_obs" in np.load(d / "prepared.npz").files
    prepared["data"]["d_obs_sha256"] = "0" * 64  # the other thread count's hash
    (d / "prepared.json").write_text(json.dumps(prepared))
    rpc.load_prepared(ctx, d)  # equal numbers: accepted, with a message
    assert "round-off" in capsys.readouterr().out
    z = dict(np.load(d / "prepared.npz"))
    z["d_obs"] = z["d_obs"] + 1e-9 * ctx.prob.s  # still round-off
    np.savez(d / "prepared.npz", **z)
    rpc.load_prepared(ctx, d)
    z["d_obs"] = z["d_obs"] + 1e-3 * ctx.prob.s  # a different realization: refused
    np.savez(d / "prepared.npz", **z)
    with pytest.raises(SystemExit, match="data differ"):
        rpc.load_prepared(ctx, d)
    del z["d_obs"]  # prepared.npz from before d_obs was stored: warn, cannot compare
    np.savez(d / "prepared.npz", **z)
    rpc.load_prepared(ctx, d)
    assert "no d_obs to compare" in capsys.readouterr().out
    assert rpc.d_obs_max_diff(np.array([1.0, 2.0]), np.array([1.0, 2.5]), 0.25) == pytest.approx(2.0)


def test_thread_cap_is_imported_before_numpy_in_every_standalone_script():
    """A script that imports numpy first lets BLAS use every core; the cap module has to come first."""
    import ast

    heavy = {"numpy", "scipy", "jax", "matplotlib", "bayes_scatter"}
    checked = []
    for name in ("run_prior_comparison", "compare_priors", "edge_analysis", "convergence_audit", "prior_energy_audit",
                 "check_data_threads", "span_test", "tv_lambda_sweep", "resolution_analysis", "feature_scales", "lambda_band",
                 "equal_length_audit", "start_drift_audit", "make_overdispersed_starts", "cross_run_check", "check_features_independent"):
        tree = ast.parse((SCRIPTS / f"{name}.py").read_text())
        first_heavy = first_cap = None
        for i, node in enumerate(tree.body):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            if "_threadcap" in mods and first_cap is None:
                first_cap = i
            if heavy & set(mods) and first_heavy is None:
                first_heavy = i
        if name == "span_test":  # sets the variables itself, inline, before its first heavy import
            src = (SCRIPTS / "span_test.py").read_text()
            assert src.index("os.environ[_var]") < src.index("import numpy as np")
        else:
            assert first_cap is not None and first_heavy is not None and first_cap < first_heavy, name
        checked.append(name)
    assert len(checked) == 16
    import _threadcap

    import os

    for var in _threadcap.THREAD_VARS:
        assert int(os.environ[var]) <= _threadcap.MAX_THREADS
    old = {v: os.environ.get(v) for v in _threadcap.THREAD_VARS}
    try:
        os.environ["OMP_NUM_THREADS"] = "16"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ.pop("OPENBLAS_NUM_THREADS", None)
        _threadcap.cap()
        assert os.environ["OMP_NUM_THREADS"] == "2" and os.environ["MKL_NUM_THREADS"] == "1"  # lowered / kept
        assert os.environ["OPENBLAS_NUM_THREADS"] == "2"  # set when absent
    finally:
        for v, val in old.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def test_convergence_audit_recomputes_the_summary_numbers_and_adds_the_rank_statistics(runs, tmp_path):
    import convergence_audit as ca

    res = ca.main(["qg", "mat", "--dir", str(runs["root"]), "--null-reps", "6", "--out", str(tmp_path / "audit.json")])
    for name in ("qg", "mat"):
        e = res[name]
        s = json.loads((runs[name] / "summary.json").read_text())
        assert e["classic"]["rhat_phi"] == pytest.approx(s["convergence"]["rhat_phi"], rel=1e-9)
        # the stored draws are float32, the summary was computed from the float64 draws: agreement to float32 round-off
        assert e["classic"]["rhat_pixel_max"] == pytest.approx(s["convergence"]["rhat_pixel_max"], rel=1e-5)
        assert e["ess"]["worst_pixel_sokal_summed"] == pytest.approx(s["convergence"]["ess_pixel_min"], rel=1e-4)
        r = e["rank_normalized"]
        assert r["phi"]["worst"] == pytest.approx(max(r["phi"]["bulk"], r["phi"]["folded"]))
        assert r["pixel_max_worst"] >= max(r["pixel_max_bulk"], r["pixel_max_folded"]) - 1e-12
        assert 0 <= r["pixels_worst_above_1.05_within_1px_of_edge"] <= r["pixels_worst_above_1.05"] <= r["pixels"]
        assert e["ess"]["worst_pixel_bulk_multichain"] > 0 and e["tau"]["phi"] >= 0
        n = e["null_ar1"]
        assert 0.0 <= n["p_classic_rhat_phi_above_1.01"] <= 1.0 and len(n["pixel_max_rank_median_q95"]) == 2
    assert json.loads((tmp_path / "audit.json").read_text()).keys() == {"qg", "mat"}


def test_rank_stats_taus_see_the_autocorrelation_of_the_folded_series_that_the_draws_hide():
    """A pixel whose scale wanders has iid-looking draws (tau 1) but a strongly autocorrelated |x - median|."""
    import convergence_audit as ca

    rng = np.random.default_rng(0)
    log_scale = np.zeros((4, 600))
    for t in range(1, 600):
        log_scale[:, t] = 0.95 * log_scale[:, t - 1] + 0.3 * rng.standard_normal(4)
    slow = rng.standard_normal((4, 600)) * np.exp(log_scale)
    st = ca.rank_stats(slow[:, :, None])
    assert st["tau_bulk"][0] < 2.5 and st["tau_folded"][0] > 3.0 * st["tau_bulk"][0]
    assert st["worst"][0] == max(st["bulk"][0], st["folded"][0])


def test_prior_energy_audit_reproduces_the_run_energies_and_the_step_cost(runs, tmp_path):
    import jax.numpy as jnp
    import prior_energy_audit as pea
    from bayes_scatter import mrf

    res = pea.main(["qg", "mat", "--dir", str(runs["root"]), "--out", str(tmp_path / "energy.json")])
    for name in ("qg", "mat"):
        e = res["runs"][name]
        assert e["R"]["map"] == pytest.approx(e["R"]["map_in_summary"], rel=1e-10)  # same prior energy as the MAP stage
        assert e["R"]["truth_minus_map"] == pytest.approx(e["R"]["truth"] - e["R"]["map"])
        z = np.load(runs[name] / "chains.npz")
        assert e["Phi"]["draws_mean"] == pytest.approx(z["phi_physical"].mean())
        assert e["Phi"]["truth_minus_map"] == pytest.approx(e["Phi"]["truth"] - e["Phi"]["map"])
        assert e["U"]["truth_minus_map"] == pytest.approx(e["Phi"]["truth_minus_map"] + e["R"]["truth_minus_map"])
        assert e["blur"]["R"][0] == pytest.approx(e["R"]["truth"]) and len(e["blur"]["Phi"]) == len(pea.BLUR_SIGMAS)
        assert e["draws"] == 2 * 20
    # R of each stored draw equals the prior's own energy (vmapped chunks against a loop)
    prior = mrf.QGGMRF(8, q=1.2, T=1.0, sigma_g=0.1, sigma_0=1.0)
    theta = np.random.default_rng(0).standard_normal((2, 5, 8, 8)) * 0.2
    got = pea.energy_over_draws(prior, theta, chunk=3)
    want = np.array([[float(prior.energy(jnp.asarray(t))) for t in row] for row in theta])
    np.testing.assert_allclose(got, want, rtol=1e-12)
    # the sharp-versus-ramp table: Jensen for the convex members, the quadratic-over-the-jump member prefers blur the most
    steps = {st["label"]: st for st in res["step_cost"]}
    for st in steps.values():
        assert st["sharp"] >= st["ramp_2"] >= st["ramp_4"] >= st["ramp_8"] > 0  # k rho(D / k) decreases with k
        assert all(st[f"ratio_{k}"] >= 1.0 for k in pea.RAMPS)
    a = steps["q-GGMRF q 1.2, T 1, sigma_g 0.05 (experiment)"]
    assert a["sharp"] == pytest.approx(10.45, abs=0.01) and a["ramp_8"] == pytest.approx(4.71, abs=0.01)
    b = steps["q-GGMRF q 1.2, T 1, sigma_g 0.1 (experiment)"]
    assert b["sharp"] == pytest.approx(4.21, abs=0.01) and b["ramp_8"] == pytest.approx(1.59, abs=0.01)
    tv = steps["q 1.0, T 0.01, sigma_g 0.05 (total variation)"]
    quad = steps["q 1.0, T sigma_g = log 2 (sigma_g 0.05)"]
    assert tv["ratio_8"] < 1.05 < a["ratio_8"] < quad["ratio_8"]


def test_edge_analysis_reports_the_chain_to_chain_spread_of_the_miscoverage_fraction(runs, tmp_path):
    import edge_analysis as ea

    res = ea.main(["qg", "--dir", str(runs["root"]), "--out", str(tmp_path)])
    pc = res["qg"]["z_per_chain_frac_abs_z_gt_3"]
    assert len(pc["values"]) == 2 and pc["min"] <= pc["max"] and pc["sd"] >= 0.0
    assert pc["mc_error_of_pooled"] == pytest.approx(pc["sd"] / np.sqrt(2))
    # the bias of each bin is the mean of the chains' own biases (equal draws per chain), with their spread as its Monte Carlo error
    for b in res["qg"]["profile_by_signed_distance_to_edge_px"].values():
        assert len(b["bias_per_chain"]) == 2 and np.mean(b["bias_per_chain"]) == pytest.approx(b["bias"], abs=1e-9)
        assert b["bias_mc_error"] == pytest.approx(np.std(b["bias_per_chain"], ddof=1) / np.sqrt(2))


def test_check_data_threads_compares_arrays_in_noise_units():
    import check_data_threads as cdt

    d = np.array([[1.0 + 1j, 2.0], [3.0, 4.0 - 1j]])
    out = cdt.compare_arrays([d, d + 1e-9, d], 1e-3)
    assert out["max_abs_diff_over_noise_std"] == pytest.approx(1e-6) and out["max_abs_diff_over_max_abs_d"] > 0
    assert cdt.compare_arrays([d], 1.0)["max_abs_diff_over_noise_std"] == 0.0


@pytest.mark.slow
def test_check_data_threads_end_to_end_in_subprocesses(tmp_path):
    import check_data_threads as cdt

    out = cdt.main(["--threads", "1", "4", "--out", str(tmp_path / "check.json"), "--n", "8", "--side", "2", "--n-tx", "6",
                    "--n-rx", "8"])
    assert out["max_abs_diff_over_noise_std"] < 1e-6  # equal to round-off, whether or not the hash is
    assert set(out["per_thread_count"]) == {"1", "4"} and out["per_thread_count"]["4"]["thread_env_seen_by_the_worker"]["NPROC"] == "4"


def test_tables_print_the_audit_sections_when_their_json_files_exist(runs, tmp_path, capsys):
    import convergence_audit as ca
    import edge_analysis as ea
    import prior_comparison_tables as pt
    import prior_energy_audit as pea

    d = tmp_path / "dir"
    shutil.copytree(runs["root"], d, ignore=shutil.ignore_patterns("cmp*", "bad*"))
    pt.main(["qg", "mat", "--dir", str(d)])
    assert "### convergence audit" not in capsys.readouterr().out  # nothing to print without the audit files
    ca.main(["qg", "mat", "--dir", str(d), "--null-reps", "6"])
    ea.main(["qg", "mat", "--dir", str(d), "--out", str(d)])
    pea.main(["qg", "mat", "--dir", str(d)])
    pt.main(["qg", "mat", "--dir", str(d)])
    out = capsys.readouterr().out
    for head in ("### prior energy R and data misfit Phi", "### R and Phi of the truth blurred", "### posterior-mean map",
                 "### cost of a step of height log 2", "### convergence audit", "### edge miscoverage per run and per chain"):
        assert head in out
    assert "| qg |" in out and "q-GGMRF q 1.2, T 1, sigma_g 0.05 (experiment)" in out


def test_convergence_audit_update_keeps_the_entries_of_other_runs(runs, tmp_path):
    import convergence_audit as ca

    out = tmp_path / "audit.json"
    ca.main(["qg", "--dir", str(runs["root"]), "--null-reps", "3", "--out", str(out)])
    ca.main(["mat", "--dir", str(runs["root"]), "--null-reps", "3", "--out", str(out), "--update"])
    assert json.loads(out.read_text()).keys() == {"qg", "mat"}
    ca.main(["mat", "--dir", str(runs["root"]), "--null-reps", "3", "--out", str(out)])  # without --update: replaced
    assert json.loads(out.read_text()).keys() == {"mat"}


# ---------------------------------------------------------------------------------------------------------------
# Milestone 5: the pseudo-Huber TV prior in the runner, the lambda sweep, and the adapted affine map (DESIGN 12)
# ---------------------------------------------------------------------------------------------------------------

ADAPT_TINY = ["--adapt", "--adapt-init", "10", "--adapt-first-window", "10", "--adapt-windows", "2", "--adapt-final", "10",
              "--adapt-block", "5"]


@pytest.fixture(scope="module")
def tv_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("tv")
    rpc.main(TINY + ["--prior", "tv", "--lam", "3", "--delta", "0.01", "--out", str(root / "tv")])
    return root


def test_the_runner_registers_the_tv_prior_and_records_its_parameters(tv_run):
    d = tv_run / "tv"
    s = json.loads((d / "summary.json").read_text())
    assert s["prior"]["kind"] == "tv" and s["prior"]["lam"] == 3.0 and s["prior"]["delta"] == 0.01
    assert s["prior"]["prior_summary"]["method"].startswith("HMC")  # prior std by HMC on R alone, as for the q-GGMRF
    assert s["args"]["lam"] == 3.0 and s["laplace"]["jacobian_check_rel_err"] < 1e-8
    assert s["sampler"]["max_solver_residual_final_states"] < 1e-8 and np.isfinite(s["phi_physical"]["mean"])
    assert "adaptation" not in s and s["sampler"]["name"].startswith("hmc (zeta coordinates, theta = theta_hat")  # default path
    prob = rpc.build_data(rpc.parse_args(TINY + ["--prior", "tv", "--out", "x"]))
    assert prob.sha == s["data"]["d_obs_sha256"]  # the data do not depend on the prior
    args = rpc.parse_args(TINY + ["--prior", "tv", "--lam", "3", "--sigma-0", "0.5", "--mu", "0.1", "--out", "x"])
    prior = rpc.build_prior(args, rpc.build_context(args).grid)
    assert prior.describe()["sigma_0"] == 0.5 and float(prior.mean.mean()) == pytest.approx(0.1)
    with pytest.raises(SystemExit):
        rpc.parse_args(TINY + ["--prior", "tv", "--stage", "adapt", "--out", "x"])  # --stage adapt needs --adapt


def test_a_prepared_json_from_before_milestone_5_is_still_accepted(tv_run, tmp_path):
    """load_prepared compares the posterior-defining arguments; the milestone-4 files have no lam / delta."""
    d = tmp_path / "prep"
    rpc.main(TINY + ["--prior", "matern", "--stage", "prepare", "--out", str(d)])
    info = json.loads((d / "prepared.json").read_text())
    del info["args"]["lam"], info["args"]["delta"]
    (d / "prepared.json").write_text(json.dumps(info))
    ctx = rpc.build_context(rpc.parse_args(TINY + ["--prior", "matern", "--stage", "chain", "--chain-id", "0", "--out", str(d)]))
    rpc.load_prepared(ctx, d)
    info["args"]["sigma"] = 0.7
    (d / "prepared.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit, match="different data or prior arguments"):
        rpc.load_prepared(ctx, d)


def test_prior_energy_audit_covers_the_tv_run_and_the_tv_step_over_ramp_rows(tv_run, tmp_path):
    import prior_energy_audit as pea

    res = pea.main(["tv", "--dir", str(tv_run), "--out", str(tmp_path / "energy.json")])
    e = res["runs"]["tv"]
    assert e["prior"] == "tv" and e["R"]["map"] == pytest.approx(e["R"]["map_in_summary"], rel=1e-10)
    assert e["R"]["truth_minus_map"] == pytest.approx(e["R"]["truth"] - e["R"]["map"])
    steps = {st["label"]: st for st in res["step_cost"]}
    tv = [steps[lab] for lab, _, _ in pea.TV_FAMILY]
    for st in tv:
        assert st["sharp"] >= st["ramp_2"] >= st["ramp_4"] >= st["ramp_8"] > 0 and all(st[f"ratio_{k}"] >= 1.0 for k in pea.RAMPS)
    # the DESIGN 12.1 ratio at delta = 0.01 is 1.106 for the 8-pixel ramp (1.014 for 2 pixels), not 'a few percent'
    assert tv[0]["ratio_8"] == pytest.approx(1.106, abs=0.002) and tv[0]["ratio_2"] == pytest.approx(1.0145, abs=0.001)
    assert tv[1]["ratio_8"] == pytest.approx(1.031, abs=0.002) and tv[2]["ratio_8"] == pytest.approx(1.010, abs=0.001)
    q = steps["q-GGMRF q 1.2, T 1, sigma_g 0.05 (experiment)"]
    assert q["ratio_8"] > 2.0 > 1.2 > tv[0]["ratio_8"]
    # lam scales the cost, not the ratio
    assert pea.step_cost_tv(5.0, 0.01)["sharp"] == pytest.approx(5.0 * pea.step_cost_tv(1.0, 0.01)["sharp"])
    assert pea.step_cost_tv(5.0, 0.01)["ratio_8"] == pytest.approx(pea.step_cost_tv(1.0, 0.01)["ratio_8"])


def test_tv_lambda_sweep_runs_the_map_path_per_lambda_and_picks_the_discrepancy_value(tmp_path):
    import tv_lambda_sweep as sw

    common = ["--n", "8", "--n-tx", "6", "--n-rx", "8", "--start-maxiter", "10", "--map-maxiter", "40", "--reference-run", "none",
              "--dir", str(tmp_path)]
    out = sw.main(common + ["--lams", "0.3", "3"])
    assert set(out["runs"]) == {"0.3", "3.0"} and out["problem"]["n_data"] == 48
    for r in out["runs"].values():
        assert r["phi"] > 0 and r["R"] > 0 and r["U"] == pytest.approx(r["phi"] + r["R"])
        assert r["nit"] > 0 and r["nfev"] >= r["nit"] and np.isfinite(r["grad_inf"]) and r["time_s"] > 0
        assert r["err_model"] > 0 and r["err_chi"] > 0 and 0.0 <= r["pairs_below_delta"] <= 1.0 and r["tv"] > 0
        assert set(r["u_convergence"]) >= {"u_end", "u_drop_last_10pct", "nfev"} and r["u_convergence"]["u_drop_last_10pct"] >= 0
        assert r["laplace_theta_std"]["inside"] > 0
    # a larger lambda costs more prior energy and fits the data less
    assert out["runs"]["3.0"]["R"] > out["runs"]["0.3"]["R"] and out["runs"]["3.0"]["phi"] > out["runs"]["0.3"]["phi"]
    star = out["lambda_star"]
    assert star["lam"] == min(out["runs"].values(), key=lambda r: abs(r["phi"] - 48))["lam"]
    assert (tmp_path / "tv_lambda_sweep.json").exists() and (tmp_path / "tv_lambda_sweep_maps.npz").exists()
    assert (tmp_path / "tv_lambda_sweep.png").exists()
    assert np.load(tmp_path / "tv_lambda_sweep_maps.npz")["theta_hat_lam3.0"].shape == (8, 8)
    # --update keeps the lambdas that are there and re-chooses lambda* over all of them
    out2 = sw.main(common + ["--lams", "1", "--update"])
    assert set(out2["runs"]) == {"0.3", "1.0", "3.0"} and out2["runs"]["3.0"]["phi"] == out["runs"]["3.0"]["phi"]
    out3 = sw.main(common + ["--lams", "1"])  # without --update the file is replaced
    assert set(out3["runs"]) == {"1.0"}


def test_choose_lambda_star_is_the_closest_phi_to_n_and_interpolates_the_crossing():
    import tv_lambda_sweep as sw

    runs = {"1.0": {"lam": 1.0, "phi": 2379.0}, "3.0": {"lam": 3.0, "phi": 2394.0}, "10.0": {"lam": 10.0, "phi": 2447.0}}
    star, crossing = sw.choose_lambda_star(runs, 2400)
    assert star["lam"] == 3.0 and crossing == pytest.approx(np.exp(np.log(3.0) + (6.0 / 53.0) * np.log(10.0 / 3.0)))
    runs["4.5"] = {"lam": 4.5, "phi": 2405.0}
    star, _ = sw.choose_lambda_star(runs, 2400)
    assert star["lam"] == 4.5  # |5| < |6|
    assert sw.choose_lambda_star({"1.0": {"lam": 1.0, "phi": 2390.0}}, 2400)[1] is None  # no crossing in the sweep
    edge = np.zeros((8, 8))
    edge[2:5, 2:5] = 1.0
    th = np.zeros((8, 8))
    th[2:5, 2:5] = 0.5
    th[1, 3] = 0.2  # a pixel just outside
    b = sw.edge_bias(th, np.zeros((8, 8)), edge)
    assert b["-1"] == pytest.approx(0.2 / np.sum(sw.ea.signed_distance(edge) == -1)) and b["1"] == pytest.approx(0.5)


@pytest.fixture(scope="module")
def adapted(tmp_path_factory):
    """The tiny TV problem run with --adapt in stages: prepare, adapt, 2 chains x 2 segments, finalize; and in one process (all)."""
    root = tmp_path_factory.mktemp("adapted")
    common = TINY + ["--prior", "tv", "--lam", "3"] + ADAPT_TINY
    st = root / "staged"
    rpc.main(common + ["--stage", "prepare", "--out", str(st)])
    rpc.main(common + ["--stage", "adapt", "--out", str(st)])
    for seg in (0, 1):
        for c in (0, 1):
            rpc.main(common + ["--stage", "chain", "--chain-id", str(c), "--segment", str(seg), "--out", str(st)])
    fin = root / "first_segment"
    shutil.copytree(st, fin)
    rpc.main(common + ["--stage", "finalize", "--max-segments", "1", "--out", str(fin)])
    rpc.main(common + ["--stage", "finalize", "--out", str(st)])
    one = root / "one"
    rpc.main(common + ["--out", str(one)])
    return {"root": root, "staged": st, "first_segment": fin, "one": one, "common": common}


def test_readapt_builds_a_new_run_from_the_recorded_draws_and_it_can_be_sampled(adapted, tmp_path):
    import readapt_map as rm

    src = adapted["staged"]
    dst = adapted["root"] / "readapted"
    common = adapted["common"]
    up = rm.main(["--from", str(src), "--to", str(dst), "--kappa-per-dim", "1.0", "--"] + common)
    assert up["n_draws"] == 2 * 40 and up["weight"] == pytest.approx(64 / (80 + 64))  # 2 chains x (2 x 20) draws, kappa = N = 64
    amap_old, info_old = rpc.load_adaptation(src, 8)
    amap, info = rpc.load_adaptation(dst, 8)
    assert amap.digest() != amap_old.digest() and info["map_digest"] == amap.digest() and info["source_run"] == src.name
    assert [h["phase"] for h in info["history"]] == ["init", "window0", "window1", "final", "readapt"]
    assert info["history"][-1]["map_digest_before"] == amap_old.digest()[:16] and info["eps"] == info_old["eps"]
    starts = np.load(dst / "starts.npy")
    assert starts.shape == (2, 8, 8)
    for c in (0, 1):  # the starts are the chains' final states
        np.testing.assert_allclose(starts[c], rpc.load_segment(rpc.segment_path(src, c, 1))["theta_final"])
    # the new map is the source's window mean and the mixture: its center is the pooled mean of the draws
    draws = np.concatenate([np.concatenate([rpc.load_segment(rpc.segment_path(src, c, j))["theta"] for j in (0, 1)]).reshape(-1, 64)
                            for c in (0, 1)])
    np.testing.assert_allclose(np.asarray(amap.center_np).ravel(), draws.mean(axis=0), atol=1e-10)
    # an ordinary adapted run from here on: chains start at the stored states, the segment carries the NEW digest, the old run is intact
    for c in (0, 1):
        rpc.main(common + ["--stage", "chain", "--chain-id", str(c), "--segment", "0", "--start-theta", str(dst / "starts.npy"),
                           "--out", str(dst)])
    seg = rpc.load_segment(rpc.segment_path(dst, 0, 0))
    assert seg["adapt_digest"] == amap.digest() and rpc.load_segment(rpc.segment_path(src, 0, 0))["adapt_digest"] == amap_old.digest()
    s = rpc.main(common + ["--stage", "finalize", "--out", str(dst)])
    assert s["sampler"]["segments"] == 1 and s["sampler"]["starts"] == "start-theta file"
    with pytest.raises(SystemExit, match="already holds segment files"):
        rm.main(["--from", str(src), "--to", str(dst), "--"] + common)


def test_adapt_stage_writes_the_frozen_map_and_the_history(adapted):
    d = adapted["staged"]
    info = json.loads((d / "adaptation.json").read_text())
    z = np.load(d / "adapted_map.npz")
    assert set(z.files) == {"center", "chol", "eps", "traj_length", "max_leapfrog", "theta_final"}
    N = 64
    assert z["chol"].shape == (N, N) and np.allclose(z["chol"], np.tril(z["chol"])) and np.all(np.diag(z["chol"]) > 0)
    amap, ainfo = rpc.load_adaptation(d, 8)
    assert amap.digest() == info["map_digest"] and amap.shape == (8, 8) and ainfo["eps"] == pytest.approx(float(z["eps"]))
    assert [h["phase"] for h in info["history"]] == ["init", "window0", "window1", "final"]
    assert [h["updated"] for h in info["history"]] == [False, True, True, False]
    assert info["history"][-1]["map_digest_before"] == info["map_digest"][:16]  # frozen before anything is recorded
    assert info["kappa"] == N and info["config"]["traj_length"] == 1.75 and info["config"]["target_accept"] == 0.8
    assert all(u > 0 for u in info["update_time_s"]) and len(info["update_time_s"]) == 2
    up = info["history"][1]["update"]
    assert up["weight"] == pytest.approx(N / (10 + N)) and up["n_draws"] == 10
    assert info["n_grad_evals"] > 0 and info["eps"] > 0 and info["n_leapfrog0"] >= 1
    # the map is a valid Cholesky factor of a positive definite covariance, centered inside the DOI's range of theta
    assert np.linalg.eigvalsh(amap.covariance).min() > 0 and np.abs(z["center"]).max() < 3.0


def test_adapted_segments_carry_the_map_digest_and_continue_with_the_frozen_step(adapted):
    d = adapted["staged"]
    digest = json.loads((d / "adaptation.json").read_text())["map_digest"]
    segs = {(c, j): rpc.load_segment(rpc.segment_path(d, c, j)) for c in (0, 1) for j in (0, 1)}
    for (c, j), s in segs.items():
        assert s["adapt_digest"] == digest and s["n_phi_evals"] > 0 and s["traj_length"] == 1.75
        assert s["mean_leapfrog"] >= 1 and s["theta"].shape == (20, 8, 8) and s["mean"].shape == (2, 8, 8)
        np.testing.assert_allclose(s["mass_theta"], np.load(d / "adapted_map.npz")["center"].reshape(8, 8))
    assert segs[(0, 0)]["n_warmup"] == 10 and segs[(0, 1)]["n_warmup"] == 0
    assert segs[(0, 1)]["step"] == pytest.approx(segs[(0, 0)]["step"])  # the step is frozen in a continuation
    assert segs[(0, 0)]["step"] != segs[(1, 0)]["step"]  # each chain adapts its own step (the map is shared)
    assert not np.allclose(segs[(0, 0)]["theta"], segs[(0, 1)]["theta"])
    # a recorded segment is a deterministic function of (map, step, key, start): running the same segment again reproduces it
    again = adapted["root"] / "again"
    shutil.copytree(d, again)
    rpc.segment_path(again, 0, 0).unlink()
    rpc.main(adapted["common"] + ["--stage", "chain", "--chain-id", "0", "--segment", "0", "--out", str(again)])
    new = rpc.load_segment(rpc.segment_path(again, 0, 0))
    np.testing.assert_allclose(new["theta"], segs[(0, 0)]["theta"], rtol=1e-9, atol=1e-10)
    assert new["step"] == pytest.approx(segs[(0, 0)]["step"], rel=1e-9) and new["n_phi_evals"] == segs[(0, 0)]["n_phi_evals"]


def test_adapted_summary_has_the_adaptation_block_and_counts_the_gradient_evaluations(adapted):
    s = json.loads((adapted["staged"] / "summary.json").read_text())
    info = json.loads((adapted["staged"] / "adaptation.json").read_text())
    assert "adapted affine map" in s["sampler"]["name"] and s["sampler"]["segments"] == 2 and s["sampler"]["samples"] == 40
    assert s["sampler"]["jitter"].startswith("L uniform") and s["sampler"]["mass"].startswith("adapted affine map")
    assert s["adaptation"]["map_digest"] == info["map_digest"] and len(s["adaptation"]["history"]) == 4
    assert s["sampler"]["step0"] == pytest.approx(info["eps"])
    segs = [rpc.load_segment(rpc.segment_path(adapted["staged"], 0, j)) for j in (0, 1)]
    assert s["sampler"]["n_phi_evals_per_chain"] == sum(x["n_phi_evals"] for x in segs)
    assert s["sampler"]["max_solver_residual_final_states"] < 1e-8
    z = np.load(adapted["staged"] / "chains.npz")
    assert z["theta"].shape == (2, 40, 8, 8) and np.all(np.isfinite(z["phi_physical"]))
    assert 0.0 < s["sampler"]["acceptance"]["mean"] <= 1.0
    # the physical misfit is that of the stored draws, not the sampler's Phi'
    assert not np.allclose(z["phi_prime"], z["phi_physical"])


def test_adapted_chains_start_around_the_final_state_of_the_adaptation_chain_unless_told_to_use_the_center(adapted):
    """The GMRF pilot: every proposal from the map center (a window mean, not a visited state) was rejected."""
    d = adapted["staged"]
    z = np.load(d / "adapted_map.npz")
    args = rpc.parse_args(adapted["common"] + ["--stage", "chain", "--chain-id", "0", "--out", str(d)])
    ctx = rpc.build_context(args)
    amap, info = rpc.load_adaptation(d, 8)
    np.testing.assert_allclose(info["theta_final"].ravel(), z["theta_final"])
    seen = {}

    class Capture:  # records the start that run_segment hands to the runner
        def __init__(self, runner):
            self.zt, self.eps0, self.xi_ref = runner.zt, runner.eps0, runner.xi_ref

        def run(self, key, xi0, aux0, **kw):
            seen["xi0"] = np.asarray(xi0)
            return {}

    final = rpc.AdaptedChainRunner(ctx, amap, info, args)
    np.testing.assert_allclose(np.asarray(amap.to_theta(final.xi_ref)).ravel(), z["theta_final"], atol=1e-10)
    rpc.run_segment(Capture(final), args, 0, 0)
    k_init, _ = rpc.chain_streams(args)
    scatter = np.asarray(final.zt.overdispersed_starts(k_init, args.chains, args.init_scale)[0])
    np.testing.assert_allclose(seen["xi0"], np.asarray(final.xi_ref) + scatter, atol=1e-12)
    center = rpc.AdaptedChainRunner(ctx, amap, info, rpc.parse_args(adapted["common"] + ["--adapt-start", "center", "--stage", "chain",
                                                                                           "--chain-id", "0", "--out", str(d)]))
    assert np.all(np.asarray(center.xi_ref) == 0.0)
    rpc.run_segment(Capture(center), args, 0, 0)  # run_segment reads xi_ref from the runner it is given
    np.testing.assert_allclose(seen["xi0"], scatter, atol=1e-12)
    assert "final state of the adaptation chain" in json.loads((d / "summary.json").read_text())["sampler"]["starts"]


def test_one_process_run_with_adapt_equals_the_staged_run_and_refuses_a_mixed_or_foreign_map(adapted, tmp_path):
    a, b = np.load(adapted["one"] / "chains.npz"), np.load(adapted["first_segment"] / "chains.npz")
    one_digest = json.loads((adapted["one"] / "adaptation.json").read_text())["map_digest"]
    staged_digest = json.loads((adapted["staged"] / "adaptation.json").read_text())["map_digest"]
    if jax.default_backend() == "cpu":
        # the CPU runs are bit-reproducible: the same adapted map (to the byte) and the same draws
        for key in ("theta", "phi_physical", "welford_mean", "welford_var", "accept", "step", "theta_hat"):
            np.testing.assert_allclose(b[key], a[key], rtol=1e-8, atol=1e-9, err_msg=key)
        assert one_digest == staged_digest
    else:
        # On the GPU the adaptation's reductions are not bit-reproducible between the two runs, so the maps differ in the last digits
        # and their digests differ (measured on the RTX 3090, 2026-11-08: center 1.9e-5, chol 9.5e-6, eps 2.6e-5 apart; theta_hat 1e-14).
        # The draws then differ by a constant ~5e-4 (no growth over the 20 draws: not chaotic divergence). The bounds are about 10 times the
        # measured differences; a wrong map, seed or chain assignment would put the draws O(0.1) apart.
        m1, m2 = np.load(adapted["one"] / "adapted_map.npz"), np.load(adapted["staged"] / "adapted_map.npz")
        for key in ("center", "chol", "theta_final"):
            np.testing.assert_allclose(m2[key], m1[key], rtol=0, atol=2e-3 * np.abs(m1[key]).max(), err_msg=key)
        np.testing.assert_allclose(m2["eps"], m1["eps"], rtol=2e-3)
        assert m1["traj_length"] == m2["traj_length"] and np.array_equal(m1["max_leapfrog"], m2["max_leapfrog"])
        gpu_tol = {"theta": dict(rtol=0, atol=1e-2), "phi_physical": dict(rtol=5e-3), "welford_mean": dict(rtol=0, atol=2e-3),
                   "welford_var": dict(rtol=0, atol=2e-4), "accept": dict(rtol=0, atol=0), "step": dict(rtol=2e-2),
                   "theta_hat": dict(rtol=1e-10, atol=1e-12)}
        for key, tol in gpu_tol.items():
            np.testing.assert_allclose(b[key], a[key], err_msg=key, **tol)
    # finalize refuses segments run with different maps
    mixed = tmp_path / "mixed"
    shutil.copytree(adapted["staged"], mixed)
    seg = rpc.load_segment(rpc.segment_path(mixed, 1, 1))
    seg["adapt_digest"] = "0" * 64
    rpc.save_segment(rpc.segment_path(mixed, 1, 1), seg)
    with pytest.raises(SystemExit, match="different adapted maps"):
        rpc.main(adapted["common"] + ["--stage", "finalize", "--out", str(mixed)])
    # a map file that was edited after the segments were written no longer matches its digest
    edited = tmp_path / "edited"
    shutil.copytree(adapted["staged"], edited)
    z = dict(np.load(edited / "adapted_map.npz"))
    z["center"] = z["center"] + 1e-6
    np.savez(edited / "adapted_map.npz", **z)
    with pytest.raises(SystemExit, match="does not match the digest"):
        rpc.main(adapted["common"] + ["--stage", "chain", "--chain-id", "0", "--segment", "1", "--out", str(edited)])
    missing = tmp_path / "missing"
    rpc.main(adapted["common"] + ["--stage", "prepare", "--out", str(missing)])
    with pytest.raises(SystemExit, match="run --stage adapt first"):
        rpc.main(adapted["common"] + ["--stage", "chain", "--chain-id", "0", "--out", str(missing)])


def test_make_prior_jobs_builds_tv_and_adapted_jobs_in_new_run_directories(tmp_path):
    import make_prior_jobs as mj

    assert mj.prior_args("tv_lam4.5") == ["--prior", "tv", "--lam", "4.5", "--delta", "0.01", "--sigma-0", "1", "--map-maxiter", "4000"]
    assert mj.prior_args("gmrf_adapt")[-1] == "--adapt" and mj.prior_args("gmrf_adapt")[:-1] == mj.prior_args("gmrf")
    assert mj.prior_args("tv_lam1.5_adapt")[-1] == "--adapt"
    for bad in ("tv_lam", "tv_lamx", "foo", "matern_adapt2"):
        with pytest.raises(ValueError):
            mj.prior_args(bad)
    js = mj.jobs("adapt", ["matern", "gmrf_adapt", "tv_lam4.5_adapt"], 0, 300, 400, 1, [], {})
    assert [j["name"] for j in js] == ["pc_gmrf_adapt_adaptation", "pc_tv_lam4.5_adapt_adaptation"]  # only adapted priors adapt
    assert all("--stage" in j["cmd"] and "adapt" in j["cmd"][j["cmd"].index("--stage") + 1] for j in js)
    assert js[1]["done_marker"] == "results/prior_comparison/pc_tv_lam4.5_adapt/adaptation.json"
    chains = mj.jobs("chain", ["tv_lam4.5_adapt"], 0, 300, 400, 1, [], {})
    assert [j["name"] for j in chains] == [f"pc_tv_lam4.5_adapt_c{c}_s0" for c in range(4)] and "--adapt" in chains[0]["cmd"]
    out = tmp_path / "jobs.json"
    mj.main(["prepare", "--priors", "tv_lam3", "--out", str(out)])
    assert json.loads(out.read_text())[0]["name"] == "pc_tv_lam3_prepare"
    with pytest.raises(SystemExit):
        mj.main(["prepare", "--priors", "bogus", "--out", str(out)])


def test_cross_run_check_pools_the_chains_of_runs_of_one_target(adapted, tmp_path):
    import cross_run_check as crc

    # the staged run and the single-process run of the same tiny problem use the same data and the same adapted map
    res = crc.main([adapted["staged"].name, adapted["one"].name, "--dir", str(adapted["root"]), "--out", str(tmp_path / "x.json")])
    assert res["chains_per_run"] == [2, 2] and res["draws_used_per_chain"] == 20  # the smaller run has 20 draws per chain (staged: 2 x 20)
    p = res["pooled"]
    assert p["all"]["pixels"] == 64 and 0 < p["edge_band_abs_d_le_1"]["pixels"] <= p["abs_d_le_2"]["pixels"] <= 64
    assert np.isfinite(p["all"]["rhat_max"]) and np.isfinite(res["rhat_phi"]) and set(res["features_pooled"]) == set(res["features_per_run"][adapted["one"].name])
    assert (tmp_path / "x.json").exists()
    with pytest.raises(SystemExit, match="smallest number of draws"):
        crc.cross_check(adapted["root"], [adapted["staged"].name, adapted["one"].name], last=1000)


def test_slow_modes_reports_the_covariance_of_zeta_in_the_map_coordinates(adapted, tmp_path):
    import slow_modes as sm

    res = sm.main([adapted["staged"].name, "--dir", str(adapted["root"]), "--top", "3", "--out", str(tmp_path / "s.json"),
                   "--figure", str(tmp_path / "pca.png")])
    assert (tmp_path / "pca.png").stat().st_size > 1000
    e = res[adapted["staged"].name]
    assert e["chains"] == 2 and e["draws_per_chain"] == 40 and e["dim"] == 64
    ev = e["eigenvalues_top"]
    assert len(ev) == 3 and ev[0] >= ev[1] >= ev[2] > 0 and len(e["directions"]) == 3
    # the trace of the covariance of zeta is the sum of the eigenvalues: variance per coordinate x N, and |zeta - mean|^2 / 2 averages to half of it
    # (the covariance of the pooled draws, denominator C S - 1 against the mean over draws of the squared deviation: equal up to that factor)
    assert e["mean_half_sq_dev"] == pytest.approx(0.5 * e["variance_per_coordinate"] * 64 * (2 * 40 - 1) / (2 * 40), rel=1e-9)
    Z, _ = sm.zeta_draws(adapted["staged"], skip_segments=1)
    assert Z.shape == (2, 20, 64)
    with pytest.raises(SystemExit, match="leaves no segment"):
        sm.zeta_draws(adapted["staged"], skip_segments=2)
    # a direction's projection of an exactly-whitened Gaussian has variance one and eigenvalues near one
    rng = np.random.default_rng(0)
    g = sm.slow_mode_report(rng.standard_normal((4, 500, 10)), top=2)
    assert 0.8 < g["eigenvalues_top"][0] < 1.5 and g["directions"][0]["split_rhat"] < 1.05
    drift = rng.standard_normal((4, 500, 10))
    drift[:, :, 0] += np.linspace(0, 20, 500)  # a drifting direction: large eigenvalue, split R-hat well above one
    r = sm.slow_mode_report(drift, top=1)
    assert r["eigenvalues_top"][0] > 20 and r["directions"][0]["split_rhat"] > 1.2


def test_independent_feature_check_agrees_with_the_feature_code(runs, tmp_path):
    import check_features_independent as cfi

    res = cfi.main(["qg", "mat", "--dir", str(runs["root"]), "--out", str(tmp_path / "f.json")])
    for name in ("qg", "mat"):
        assert res[name]["max_abs_difference"] < 1e-9  # the same probabilities from another implementation of the interpolation
    assert res["qg"]["annulus_pixels"] > 0


# ---------------------------------------------------------------------------------------------------------------
# Review follow-up of milestone 5 (DESIGN 12.6): continuations are not independent chains; windows; Laplace features
# ---------------------------------------------------------------------------------------------------------------


def test_cross_run_check_knows_that_a_readapted_run_continues_the_same_chains(adapted, tmp_path):
    """The milestone-5 report pooled `pc_*_adapt` with `pc_*_adapt2` as 'eight chains'; the chains of the second run START at the final states of the
    first run's chains, so the pool is four chains in two windows (review). The script must say so, from the run directories."""
    import cross_run_check as crc
    import readapt_map as rm

    src = adapted["staged"]
    dst = adapted["root"] / "readapted_for_relations"
    rm.main(["--from", str(src), "--to", str(dst), "--"] + adapted["common"])
    for c in (0, 1):
        rpc.main(adapted["common"] + ["--stage", "chain", "--chain-id", str(c), "--segment", "0", "--start-theta", str(dst / "starts.npy"),
                                      "--samples", "20", "--out", str(dst)])
    rpc.main(adapted["common"] + ["--stage", "finalize", "--samples", "20", "--out", str(dst)])
    rel = crc.relations(adapted["root"], [src.name, dst.name])
    assert rel[src.name] == {"continues": None}
    assert rel[dst.name]["continues"] == src.name and rel[dst.name]["max_abs_start_minus_source_final"] == 0.0
    res = crc.cross_check(adapted["root"], [src.name, dst.name], last=20)
    assert res["chains_independent"] is False and "stationarity check" in res["interpretation"]
    assert res["relations"][dst.name]["continues"] == src.name
    # two runs that are not continuations of each other are independent samplers
    ind = crc.cross_check(adapted["root"], [adapted["staged"].name, adapted["one"].name], last=20)
    assert ind["chains_independent"] is True and ind["interpretation"] == "the pooled chains are independent chains"
    assert crc.final_states(adapted["root"], src.name).shape == (2, 8, 8)


def test_convergence_audit_windows_compare_runs_at_equal_length(runs):
    import convergence_audit as ca

    base = runs["root"]
    full = ca.audit_run("qg", base, 3)
    win = ca.audit_run("qg", base, 3, first=5, draws=12)
    assert full["draws_per_chain"] == 20 and full["window"] == [0, 20]
    assert win["draws_per_chain"] == 12 and win["window"] == [5, 17]
    z = np.load(base / "qg" / "chains.npz")
    from bayes_scatter import diagnostics as dg

    th = z["theta"][:, 5:17].astype(np.float64).reshape(2, 12, -1)
    assert win["classic"]["rhat_pixel_max"] == pytest.approx(float(np.nanmax(dg.split_rhat(th))))
    assert sum(v for k, v in win["rank_normalized"]["flagged_by_region"].items() if not k.startswith("outside_d_m")) == \
        win["rank_normalized"]["pixels_worst_above_1.05"]
    with pytest.raises(SystemExit, match="not within"):
        ca.audit_run("qg", base, 3, first=15, draws=12)
    with pytest.raises(SystemExit, match="at least 8"):
        ca.audit_run("qg", base, 3, first=0, draws=4)


def test_laplace_features_come_from_exact_gaussian_draws_without_mcmc(adapted):
    import feature_scales as fs

    out = fs.laplace_analysis(adapted["staged"].name, adapted["root"], [0.0, 0.5], 4000, seed=3)
    assert out["draws"] == 4000 and set(out["radii"]) == {"0", "0.5"}
    for pr in out["radii"].values():
        assert set(pr) == set(fs.FEATURES) and all(0.0 <= v <= 1.0 for v in pr.values())
    # the draws have the Laplace covariance: their mean pixel sd is the MAP's Laplace sd (H^-1 diagonal) within Monte Carlo error
    args = rpc.parse_args(adapted["common"] + ["--out", "x"])
    ctx = rpc.build_context(args)
    prep = rpc.load_prepared(ctx, adapted["staged"])
    assert out["laplace_pixel_sd_mean"] == pytest.approx(float(prep.mass.theta_std().mean()), rel=0.03)
