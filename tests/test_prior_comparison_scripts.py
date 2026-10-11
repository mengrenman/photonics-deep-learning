"""End-to-end tests of scripts/run_prior_comparison.py and scripts/compare_priors.py on a tiny problem."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

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
                 "check_data_threads", "span_test"):
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
    assert len(checked) == 7
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
