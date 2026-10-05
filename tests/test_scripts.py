"""Smoke tests of the two command-line scripts on tiny problems (they must run end to end and write their
outputs; the numerical validation lives in test_born_end_to_end.py and test_inference.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_posterior_end_to_end_on_a_tiny_problem(tmp_path, capsys):
    rp = _load("run_posterior")
    out = tmp_path / "run"
    summary = rp.main([
        "--n", "8", "--side", "1.0", "--n-tx", "6", "--n-rx", "8", "--phantom", "cylinder", "--eps-r", "1.6",
        "--radius", "0.3", "--snr", "35", "--prior", "matern", "--sigma", "0.5", "--ell", "0.3",
        "--sampler", "mala", "--chains", "2", "--warmup", "40", "--samples", "80", "--thin", "2",
        "--init", "map", "--seed", "3", "--out", str(out)])
    for f in ("summary.json", "chains.npz", "maps.png", "maps_log_eps.png", "traces.png"):
        assert (out / f).exists(), f
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["args"]["sampler"] == "mala" and on_disk["args"]["link"] == "log_eps"
    assert on_disk["args"]["mean"] == 0.0 and on_disk["args"]["data_factor"] == 2
    assert on_disk["data_grid_n"] == 16 and not on_disk["inverse_crime_control"]
    # evaluation counts: 2 chains x (1 + 40 + 80) potential evaluations, each with its adjoint
    assert on_disk["forward_solves"]["sampling_phi_evals"] == 2 * 121
    assert on_disk["forward_solves"]["sampling_adjoint_solves"] == 2 * 121
    assert 0.0 < on_disk["acceptance"]["mean"] <= 1.0 and len(on_disk["final_step"]) == 2
    assert on_disk["phi"]["ess"] > 0 and on_disk["ess_per_s"]["phi"] > 0
    assert len(on_disk["pixels"]) == 5 and on_disk["wall_time_s"]["sampling"] > 0
    m = on_disk["metrics"]
    assert m["err_model"]["map"] < m["err_model"]["vacuum"] and m["err_data"]["map"] < 0.5
    assert m["err_chi"]["map"] < 0.8
    assert on_disk["max_solver_residual_final_states"] < 1e-8
    # chains.npz
    z = np.load(out / "chains.npz")
    assert z["samples"].shape == (2, 40, 2, 8, 8)  # thin 2 of 80 steps; (theta, eps_r) per draw
    assert z["phi"].shape == (2, 80) and np.all(np.exp(z["samples"][:, :, 0]) == pytest.approx(z["samples"][:, :, 1], rel=1e-5))
    assert summary["metrics"] == on_disk["metrics"]
    # the sampler ran on the physical potential, so the misfit entry is the trace itself
    assert on_disk["phi"]["kind"] == "physical" and on_disk["phi_physical"]["every"] == 1
    assert on_disk["phi_physical"]["mean"] == pytest.approx(on_disk["phi"]["mean"])
    assert np.array_equal(z["phi_physical"], z["phi"])
    # operating point: this 8 x 8 grid is far too coarse for 35 dB data (model error 3x the noise), so the
    # script must say so; Phi at the truth is then far above the realized noise energy
    op = on_disk["operating_point"]
    assert op["n_data"] == 48 and op["model_to_noise"] > 1.0 and op["phi_excess_sd"] > 10.0
    assert op["phi_truth"] > op["phi_clean"] > 0.0 and on_disk["machine"]["cpu_count"] >= 1
    assert "WARNING: the inference grid is too coarse" in capsys.readouterr().out
    # the post-hoc script reproduces the operating point stored at run time from chains.npz alone
    (out / "summary.json").write_text(json.dumps({**on_disk, "operating_point": None}))
    _load("operating_point").main([str(out)])
    redone = json.loads((out / "summary.json").read_text())["operating_point"]
    for k, v in op.items():
        assert redone[k] == pytest.approx(v, rel=1e-7, abs=1e-9), k


def test_run_posterior_pcn_identity_born_variant(tmp_path):
    rp = _load("run_posterior")
    out = tmp_path / "born"
    s = rp.main([
        "--n", "8", "--side", "1.0", "--n-tx", "6", "--n-rx", "8", "--eps-r", "1.2", "--radius", "0.3",
        "--snr", "30", "--link", "identity", "--born", "--sigma", "0.3", "--ell", "0.3", "--sampler", "pcn",
        "--chains", "2", "--warmup", "30", "--samples", "60", "--data-factor", "1", "--out", str(out)])
    assert s["inverse_crime_control"] and s["forward_solves"]["sampling_adjoint_solves"] == 0
    # the operating point of a Born run measures the Born model (not the full one) against the full-model data
    assert s["operating_point"]["rel_model_error"] > 1e-4 and s["args"]["link"] == "identity"
    assert (out / "maps.png").exists() and not (out / "maps_log_eps.png").exists()


def test_born_exact_check_script_runs(tmp_path):
    be = _load("born_exact_check")
    out = tmp_path / "born"
    be.main(["--n", "8", "--side", "1.0", "--n-tx", "8", "--n-rx", "12", "--chains", "2", "--scale", "0.05",
             "--samplers", "pcn", "hmc", "--out", str(out)])
    data = json.loads((out / "summary.json").read_text())
    assert set(data["samplers"]) == {"pcn", "hmc"} and (out / "born_exact.png").exists()
    assert data["samplers"]["hmc"]["wall_s"] > 0 and np.isfinite(data["samplers"]["hmc"]["z_mean_rms"])
