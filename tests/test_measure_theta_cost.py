"""scripts/measure_theta_cost.py: the worker on a tiny grid (record structure, consistency) and the driver's guard."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import measure_theta_cost as mt  # noqa: E402


def test_worker_reports_every_cost_and_the_map_of_four_priors(capsys):
    mt.worker(SimpleNamespace(n=8, reps=1, map=True, tv_lam=4.0, window_draws=20))
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("RESULT ")][-1]
    rec = json.loads(line[len("RESULT "):])
    pe = rec["per_evaluation"]
    assert {"phi", "u_value_and_grad_matern", "u_value_and_grad_qggmrf", "zeta_phi_fn_matern",
            "prior_value_and_grad_matern", "prior_value_and_grad_qggmrf", "u_value_and_grad_tv", "prior_value_and_grad_tv",
            "zeta_phi_fn_tv", "zeta_phi_fn_adapted_tv"} <= set(pe)
    assert all(v["min_s"] > 0 and v["min_s"] <= v["median_s"] for v in pe.values())
    assert pe["phi"]["min_s"] < pe["u_value_and_grad_matern"]["min_s"]  # forward only is cheaper than forward + adjoint
    jac = rec["jacobian"]
    assert jac["n_solves"] == 100 and jac["dense_J_bytes"] == 40 * 60 * 64 * 16
    for k in ("fields", "assemble_dense_J", "gauss_newton_gram", "dense_JhJ"):
        assert jac[k]["min_s"] > 0
    assert set(rec["laplace_mass"]) == {"matern", "qggmrf", "tv"} and rec["laplace_mass"]["matern"]["total_min_s"] > 0
    w = rec["adaptation_window_update"]
    assert w["n_draws"] == 20 and w["dim"] == 64 and w["kappa"] == 64.0 and w["total_min_s"] > 0
    assert set(w["parts_of_fastest"]) == {"time_sample_cov_s", "time_laplace_s", "time_mix_cholesky_s"}
    assert sum(w["parts_of_fastest"].values()) <= w["total_min_s"] * 1.05
    assert set(rec["map"]["priors"]) == {"matern", "gmrf", "qggmrf_sg0.05", "qggmrf_sg0.1"}
    for v in rec["map"]["priors"].values():
        assert v["nit"] > 0 and v["nfev"] >= v["nit"] and v["map_time_s"] > 0 and v["laplace_mass_s"] > 0
    assert rec["load_avg_before"] >= 0 and rec["n"] == 8


def test_driver_refuses_two_or_three_xla_threads():
    with pytest.raises(SystemExit):
        mt.main(["--threads", "2", "--ns", "8"])
