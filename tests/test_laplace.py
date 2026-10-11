"""Tests for the Laplace-whitened sampling prototype (scripts/run_posterior_laplace.py) and the helpers it uses
in ``bayes_scatter.inference`` (Laplace pixel variance, log-normal re-centering).

Everything runs on a tiny problem (8 x 8 grid, side 1, 6 transmitters, 8 receivers, 256 whitened coordinates) so
that the dense Gauss-Newton Hessian can be built and compared with the matrix-free pieces.
"""

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import samplers as sm
from bayes_scatter.inference import laplace_theta_variance, lognormal_center_shift
from bayes_scatter.priors import LogEps

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # the script imports run_posterior by name

TINY = ["--n", "8", "--side", "1.0", "--n-tx", "6", "--n-rx", "8", "--phantom", "cylinder", "--eps-r", "1.6",
        "--radius", "0.3", "--snr", "35", "--prior", "matern", "--sigma", "0.5", "--ell", "0.3", "--seed", "3"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lp():
    return _load("run_posterior_laplace")


@pytest.fixture(scope="module")
def tiny(lp):
    """Target, MAP and the Gauss-Newton eigenpairs of the tiny problem (all 256 directions)."""
    rp = lp.rp
    prob = rp.build_problem(rp.parse_args(TINY + ["--out", "unused"]))
    target = prob.target
    xi_map, _, _ = lp._map_orig(target, jnp.zeros(prob.field.xi_shape), maxiter=200)
    lam, V, _ = lp.gn_eigs(target, xi_map, 40, 1e-8)
    return {"prob": prob, "target": target, "field": prob.field, "xi_map": xi_map, "lam": lam, "V": V}


def _dense_gn(t):
    """Dense whitened Gauss-Newton Hessian (2/s^2) J_r^T J_r at the MAP."""
    target, xi_map = t["target"], t["xi_map"]
    _, _, aux = target.potential_and_grad(xi_map, target.initial_aux())

    def f(x):
        d, _ = target.forward(target.link(target.field.theta(x.reshape(xi_map.shape))), aux)
        return jnp.concatenate([jnp.real(d).ravel(), jnp.imag(d).ravel()])

    J = np.asarray(jax.jacfwd(f)(xi_map.ravel()))
    return (2.0 / target.s**2) * J.T @ J


# ---------------------------------------------------------------------------------------------
# Matrix-free Laplace helpers against dense linear algebra
# ---------------------------------------------------------------------------------------------


def test_matrix_free_eigenpairs_are_the_top_of_the_dense_gauss_newton_spectrum(tiny):
    H = _dense_gn(tiny)
    w = np.sort(np.linalg.eigvalsh(H))[::-1]
    np.testing.assert_allclose(tiny["lam"], w[:40], rtol=1e-5)
    assert tiny["lam"][0] > 1e3 and tiny["lam"][-1] < 1.0 < tiny["lam"][5]  # a wide range of curvatures


def test_laplace_theta_variance_matches_the_dense_posterior_covariance(tiny):
    """var(theta_p) = (L Sigma L^T)_pp with Sigma = (I + H)^-1; here H is the rank-K truncation V lam V^T, and
    for K = all directions the full Gauss-Newton Hessian. Prior pixels have variance sigma^2 exactly."""
    field = tiny["field"]
    L = field.sqrt_matrix()
    np.testing.assert_allclose((L**2).sum(1), field.sigma**2, rtol=1e-12)
    lam, V = tiny["lam"], tiny["V"]
    H_k = (V * lam) @ V.T
    cov = np.linalg.inv(np.eye(H_k.shape[0]) + H_k)
    dense = np.einsum("pi,ij,pj->p", L, cov, L).reshape(field.n, field.n)
    np.testing.assert_allclose(laplace_theta_variance(field, lam, V), dense, rtol=1e-9)
    # all directions: eigendecomposition of the full H
    w, U = np.linalg.eigh(_dense_gn(tiny))
    full = np.einsum("pi,ij,pj->p", L, np.linalg.inv(np.eye(len(w)) + (U * w) @ U.T), L).reshape(field.n, field.n)
    np.testing.assert_allclose(laplace_theta_variance(field, w, U), full, rtol=1e-8)
    # the data shrink the variance in every pixel but not to zero
    v = laplace_theta_variance(field, lam, V)
    assert np.all(v < field.sigma**2) and np.all(v > 0.0)


def test_lognormal_center_shift_lowers_mean_theta_by_half_the_variance(tiny):
    field, lam, V = tiny["field"], tiny["lam"], tiny["V"]
    delta, info = lognormal_center_shift(field, lam, V)
    var_mean = float(np.mean(laplace_theta_variance(field, lam, V)))
    assert info["var_mean"] == pytest.approx(var_mean, rel=1e-12)
    shift = np.asarray(field.theta(tiny["xi_map"] + delta) - field.theta(tiny["xi_map"]))  # = L delta (affine)
    assert shift.mean() == pytest.approx(-0.5 * var_mean, rel=1e-9)  # -Var/2, not +Var/2 and not -Var
    assert info["theta_mean_shift"] == pytest.approx(-0.5 * var_mean, rel=1e-12)
    # along the stiffest eigenvector only
    d = np.asarray(delta).ravel()
    assert abs(d @ V[:, 0]) == pytest.approx(np.linalg.norm(d), rel=1e-12)
    assert info["n_sd"] == pytest.approx(abs(info["t"]) * np.sqrt(1.0 + lam[0]), rel=1e-12)
    assert info["n_sd"] > 5.0  # many Laplace standard deviations: the reason to re-center
    assert not info["capped"] and info["n_sd_full"] == pytest.approx(info["n_sd"], rel=1e-12)


def test_lognormal_center_shift_cap_scales_the_shift_down(tiny):
    field, lam, V = tiny["field"], tiny["lam"], tiny["V"]
    delta_full, full = lognormal_center_shift(field, lam, V)
    delta_cap, capped = lognormal_center_shift(field, lam, V, max_sd=0.25 * full["n_sd"])
    assert capped["capped"] and capped["n_sd"] == pytest.approx(0.25 * full["n_sd"], rel=1e-12)
    assert capped["n_sd_full"] == pytest.approx(full["n_sd"], rel=1e-12)
    np.testing.assert_allclose(np.asarray(delta_cap), 0.25 * np.asarray(delta_full), rtol=1e-10)
    assert capped["theta_mean_shift"] == pytest.approx(0.25 * full["theta_mean_shift"], rel=1e-10)
    assert capped["theta_mean_shift_full"] == pytest.approx(full["theta_mean_shift"], rel=1e-12)
    _, loose = lognormal_center_shift(field, lam, V, max_sd=10.0 * full["n_sd"])
    assert not loose["capped"] and loose["n_sd"] == pytest.approx(full["n_sd"], rel=1e-12)


# ---------------------------------------------------------------------------------------------
# The Laplace-whitened target
# ---------------------------------------------------------------------------------------------


def test_laplace_target_potential_is_phi_prime_not_the_misfit(lp, tiny):
    """Phi'(zeta) = Phi(xi) + |xi|^2/2 - |zeta|^2/2: the sampler's trace is not the data misfit, and the physical
    misfit is recovered as Phi' + (|zeta|^2 - |xi|^2)/2. The gradient is checked against central differences."""
    target, lam, V = tiny["target"], tiny["lam"], tiny["V"]
    xi_c = tiny["xi_map"] + 0.05 * jax.random.normal(jax.random.key(0), tiny["xi_map"].shape)
    lt = lp.LaplaceTarget(target, xi_c, lam, V)
    zeta = 3.0 * jax.random.normal(jax.random.key(1), xi_c.shape)  # a generic point, far from the center
    xi = lt.to_xi(zeta)
    np.testing.assert_allclose(np.asarray(lt.to_zeta(xi)), np.asarray(zeta), atol=1e-9)
    phi_xi, _ = target.potential(xi, target.initial_aux())
    phi_p, g, _ = lt.phi_fn(zeta, lt.initial_aux())
    offset = 0.5 * float(jnp.sum(xi**2) - jnp.sum(zeta**2))
    assert float(phi_p) == pytest.approx(float(phi_xi) + offset, rel=1e-10)
    assert abs(offset) > 100.0  # the offset is large, so Phi' and Phi are far apart
    assert float(phi_xi) == pytest.approx(float(phi_p) + 0.5 * float(jnp.sum(zeta**2) - jnp.sum(xi**2)), rel=1e-10)
    # same density: exp(-Phi'(zeta) - |zeta|^2/2) is exp(-Phi(xi) - |xi|^2/2) up to a constant (here exactly)
    assert float(phi_p) + 0.5 * float(jnp.sum(zeta**2)) == pytest.approx(
        float(phi_xi) + 0.5 * float(jnp.sum(xi**2)), rel=1e-10)
    # gradient of Phi' in zeta by central differences
    aux = lt.initial_aux()
    rng = np.random.default_rng(5)
    eps = 1e-5  # solver tolerance 1e-10 on Phi ~ 2000 limits central differences at 1e-6 to a few 1e-6 relative
    for _ in range(3):
        v = rng.standard_normal(zeta.shape)
        v = jnp.asarray(v / np.linalg.norm(v))
        fd = (lt.phi_fn(zeta + eps * v, aux)[0] - lt.phi_fn(zeta - eps * v, aux)[0]) / (2 * eps)
        assert abs(float(fd) - float(jnp.vdot(g, v))) < 1e-5 * max(abs(float(fd)), 1.0)  # measured 5e-8 to 1.2e-6 (one full-suite run)
    # gradient-free variant has the same value
    assert float(lt.phi_fn_no_grad(zeta, aux)[0]) == pytest.approx(float(phi_p), rel=1e-12)


def test_laplace_map_squares_to_the_laplace_covariance(lp, tiny):
    """M M^T = Sigma = (I + V lam V^T)^-1 with M symmetric, so Sigma^-1 (M M z) = z for every z."""
    lam, V = tiny["lam"], tiny["V"]
    lt = lp.LaplaceTarget(tiny["target"], tiny["xi_map"], lam, V)
    shape = tiny["xi_map"].shape
    origin = lt.to_xi(jnp.zeros(shape))

    def apply_m(x):  # the linear part of zeta -> xi
        return np.asarray(lt.to_xi(jnp.asarray(x).reshape(shape)) - origin).ravel()

    z = np.asarray(jax.random.normal(jax.random.key(2), shape)).ravel()
    m2 = apply_m(apply_m(z))
    np.testing.assert_allclose(m2 + V @ (lam * (V.T @ m2)), z, atol=1e-8 * np.linalg.norm(z))
    w = np.asarray(jax.random.normal(jax.random.key(3), shape)).ravel()
    assert apply_m(z) @ w == pytest.approx(z @ apply_m(w), rel=1e-9)  # M is symmetric


# ---------------------------------------------------------------------------------------------
# Step-size stability limits
# ---------------------------------------------------------------------------------------------


def test_mala_step_limit_is_four_over_curvature_not_two(lp):
    """For U = a x^2/2 the MALA proposal mean is (1 - h a/2) x, contracting only for h < 4/a. The step that
    Robbins-Monro adaptation settles on (acceptance 0.574) is 0.83 * 4/a for every a, i.e. 1.66 times the old,
    wrong limit 2/a; the step limit the script reports is 4/(1 + lam_max)."""
    lam_max = 1.0e4
    a = 1.0 + lam_max
    limit = lp.mala_step_limit(lam_max)
    assert limit == pytest.approx(4.0 / a, rel=1e-14)
    assert lp.pcn_langevin_step_limit(lam_max) == pytest.approx(2.0 / lam_max, rel=1e-14)

    def phi_fn(x, aux):  # Phi = lam x^2/2, so that U = Phi + x^2/2 = a x^2/2
        return 0.5 * lam_max * jnp.sum(x * x), lam_max * x, aux

    def acceptance(h):
        res = sm.mala(jax.random.key(0), phi_fn, jnp.full((1,), 1.0 / np.sqrt(a)), jnp.zeros(()), n_warmup=0,
                      n_samples=4000, step0=h)
        return float(jnp.mean(res.accept))

    # acceptance falls steadily through the limit: a step of 2/a is still conservative (acceptance far above
    # the 0.574 target), the limit itself has acceptance near one half, and beyond it the proposals expand
    assert acceptance(0.5 * limit) > 0.7
    assert 0.4 < acceptance(limit) < 0.6
    assert acceptance(1.5 * limit) < 0.4 < acceptance(0.5 * limit)
    res = sm.mala(jax.random.key(1), phi_fn, jnp.full((1,), 1.0 / np.sqrt(a)), jnp.zeros(()), n_warmup=3000,
                  n_samples=10, step0=1e-3 * limit)
    ratio = float(res.step) / limit
    assert 0.7 < ratio < 1.0  # adapted step: close to, and below, 4/a (it would be 1.66 * (2/a) -> ratio 0.83)
    assert float(res.step) > 1.4 * 2.0 / a  # incompatible with 2/a being the stability limit


# ---------------------------------------------------------------------------------------------
# The script end to end
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_run(lp, tmp_path_factory):
    out = tmp_path_factory.mktemp("lap") / "run"
    cache = out.parent / "cache.npz"
    summary = lp.main(TINY + ["--sampler", "mala", "--chains", "2", "--warmup", "60", "--samples", "120",
                              "--thin", "3", "--map-maxiter", "200", "--laplace-k", "20", "--laplace-tol", "1e-8",
                              "--laplace-cache", str(cache), "--out", str(out)])
    return {"out": out, "cache": cache, "summary": summary, "chains": np.load(out / "chains.npz")}


def test_script_labels_the_sampler_trace_as_phi_prime_and_reports_the_physical_misfit(lp, tiny_run, tiny):
    r = tiny_run
    s = json.loads((r["out"] / "summary.json").read_text())
    assert s["phi"]["kind"].startswith("reparametrized Phi'")
    z = r["chains"]
    assert z["phi_physical"].shape == (2, 40) and z["phi"].shape == (2, 120)  # one per stored draw / per step
    assert s["phi_physical"]["every"] == 3 and s["phi_physical"]["n_per_chain"] == 40
    # the final state is the last stored draw (120 steps, thin 3): trace value = Phi'(zeta_final), and the
    # stored physical misfit is Phi(xi_final); their difference is (|zeta|^2 - |xi|^2)/2
    cache = np.load(r["cache"])
    lt = lp.LaplaceTarget(tiny["target"], jnp.asarray(z["xi_center"]), cache["lam"], cache["V"])
    for c in range(2):
        xi_f = jnp.asarray(z["xi_final"][c])
        zeta_f = lt.to_zeta(xi_f)
        phi_f, _ = tiny["target"].potential(xi_f, tiny["target"].initial_aux())
        assert float(z["phi_physical"][c, -1]) == pytest.approx(float(phi_f), rel=1e-7)
        assert float(z["phi"][c, -1]) == pytest.approx(
            float(phi_f) + 0.5 * float(jnp.sum(xi_f**2) - jnp.sum(zeta_f**2)), rel=1e-7)
        # not the same quantity (the offset is only O(1) on this tiny problem; thousands at n = 32)
        assert abs(float(z["phi"][c, -1]) - float(z["phi_physical"][c, -1])) > 0.5
    # summary level: the 'mean' under 'phi' is the trace, under 'phi_physical' the misfit
    assert s["phi_physical"]["mean"] == pytest.approx(float(z["phi_physical"].mean()))
    assert s["phi"]["mean"] == pytest.approx(float(z["phi"].mean()))
    # the physical misfit of draws of a posterior fitted to 48 complex data is of the order of N
    assert 0.5 * 48 < s["phi_physical"]["mean"] < 6 * 48


def test_script_keeps_the_prior_on_log_eps_r(tiny_run, tiny):
    """The prior is on theta = log eps_r (link log_eps) in the Laplace pipeline too: stored eps_r = e^theta."""
    s, z = tiny_run["summary"], tiny_run["chains"]
    assert s["args"]["link"] == "log_eps"
    assert isinstance(tiny["target"].link, LogEps)
    np.testing.assert_allclose(np.exp(z["samples"][:, :, 0]), z["samples"][:, :, 1], rtol=1e-5)
    assert s["phi"]["n_data"] == 48


def test_recentering_puts_the_chain_near_the_center_and_matches_the_lognormal_shift(lp, tiny_run):
    lap = tiny_run["summary"]["laplace"]
    info = lap["recenter_info"]
    assert lap["recenter"] == "lognormal" and info["n_sd"] > 5.0
    # the log-normal prediction: posterior mean of log eps_r minus MAP is about -Var/2 (right sign and size)
    got, pred = lap["theta_mean_post_minus_map"], info["theta_mean_shift_full"]
    assert got < 0.0 and 0.5 < got / pred < 1.6
    # re-centered: the final states sit within a fraction of the shift along the stiffest direction
    assert np.max(np.abs(lap["zeta_v0_final_per_chain"])) < 0.5 * info["n_sd"]
    # setup cost bookkeeping: MAP evaluations plus eigsh products
    assert lap["setup_grad_evals"] == lap["map_nfev"] + lap["matvecs_jvp_plus_vjp"]
    assert lap["implied_max_stable_mala_step_in_xi"] == pytest.approx(4.0 / (1.0 + lap["lam_max"]))
    assert lap["implied_max_stable_pcn_langevin_delta_in_xi"] == pytest.approx(2.0 / lap["lam_max"])


def test_cache_from_a_different_map_is_refused(lp, tiny, tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, lam=tiny["lam"], V=tiny["V"], nmv=1, xi_map=np.asarray(tiny["xi_map"]) + 1.0)
    with pytest.raises(ValueError, match="different MAP"):
        lp.load_or_compute_eigs(tiny["target"], tiny["xi_map"], 20, 1e-6, str(path))
    path2 = tmp_path / "right.npz"
    np.savez(path2, lam=tiny["lam"], V=tiny["V"], nmv=7, xi_map=np.asarray(tiny["xi_map"]), time_s=1.5)
    lam, V, nmv, xi_c, t, reused = lp.load_or_compute_eigs(tiny["target"], tiny["xi_map"], 20, 1e-6, str(path2))
    assert reused and nmv == 7 and t == 1.5 and np.array_equal(lam, tiny["lam"])


# ---------------------------------------------------------------------------------------------
# Post-processing (scripts/analyze_run.py) of a Laplace-whitened run
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ar():
    return _load("analyze_run")


def test_edge_mask_is_the_one_pixel_band_around_the_object_boundary(ar):
    chi = np.zeros((9, 9))
    chi[3:6, 3:6] = 1.0  # a 3 x 3 block of full contrast
    edge = ar.edge_mask(chi)
    # 4-connected one-pixel dilation (21 pixels: the block plus 12 neighbors) minus the 1-pixel eroded core
    assert edge.sum() == 20
    assert not edge[4, 4] and edge[3, 3] and edge[2, 3] and not edge[2, 2] and not edge[1, 1] and not edge[0, 0]
    chi[3, 3] = 0.5  # a partially covered boundary pixel is inside and not part of the full-contrast core
    assert ar.edge_mask(chi)[3, 3]


def test_prior_variance_fraction_beyond_a_wavenumber(ar):
    a = {"n": 16, "side": 2.0, "prior": "matern", "sigma": 0.5, "ell": 0.4, "nu": 1.5, "mean": 0.0}
    f0 = ar.prior_variance_fraction_beyond(a, 0.0)
    f_mid = ar.prior_variance_fraction_beyond(a, 4.0 * np.pi)
    f_hi = ar.prior_variance_fraction_beyond(a, 1e6)
    assert 0.9 < f0 < 1.0 and 0.0 < f_mid < f0 and f_hi == 0.0  # f0 excludes only the k = 0 mode
    rough = ar.prior_variance_fraction_beyond({**a, "prior": "gmrf"}, 4.0 * np.pi)
    assert rough > 2.0 * f_mid  # the GMRF spectrum decays only like |k|^-2: far more variance beyond 2 k0


def test_analyze_run_reports_the_misfit_the_laplace_reference_and_the_setup_cost(ar, tiny_run):
    m = ar.analyze(str(tiny_run["out"]))
    z, s = tiny_run["chains"], tiny_run["summary"]
    assert m["phi_kind"].startswith("reparametrized") and m["phi_every"] == 3
    assert m["phi_mean"] == pytest.approx(float(z["phi_physical"].mean()))  # the misfit, not the trace
    assert m["phi_prime_mean"] == pytest.approx(float(z["phi"].mean()))
    assert m["phi_trace_mean"] == pytest.approx(m["phi_prime_mean"])
    assert m["setup_evals"] == s["map"]["nfev"] + s["laplace"]["matvecs_jvp_plus_vjp"]
    n_evals = s["forward_solves"]["sampling_forward_solves"]
    assert m["ess_phi_per_eval_incl_setup"] == pytest.approx(m["ess_phi"] / (n_evals + m["setup_evals"]))
    assert m["ess_phi_per_eval_incl_setup"] < m["ess_phi_per_eval"]
    # the Laplace reference is the matrix-free variance of the cached eigenpairs
    assert 0.0 < m["laplace_std_theta_inside"] < m["sigma"] and 0.0 < m["laplace_std_theta_outside"] < m["sigma"]
    assert m["mcmc_over_laplace_std_min"] <= m["mcmc_over_laplace_std_mean"] <= m["mcmc_over_laplace_std_max"]
    assert 0.0 < m["prior_var_frac_beyond_2k0"] < 0.2
    assert m["laplace"]["recenter"] == "lognormal" and m["operating_point"]["n_data"] == 48
    # the pooled R-hat uses the misfit series, which have equal length for the two runs here
    res = ar.pooled("unused_label_for_test", [str(tiny_run["out"]), str(tiny_run["out"])])
    assert res["chains"] == 4
    (ar.ROOT / "results" / "unused_label_for_test_pooled.json").unlink()


def test_map_multistart_from_prior_draws_finds_the_same_map_and_reports_the_physical_misfit(tiny_run, tmp_path):
    ms = _load("map_multistart")
    out = ms.main(TINY + ["--map-maxiter", "150", "--starts", "2", "--refine-maxiter", "300", "--precondition",
                          "--laplace-cache", str(tiny_run["cache"]), "--out", str(tmp_path / "ms")])
    on_disk = json.loads((tmp_path / "ms" / "multistart.json").read_text())
    assert on_disk["preconditioned"] and len(on_disk["starts"]) == 2
    ref_phi = out["reference"]["phi"]
    assert 0.0 < ref_phi < 100.0  # the physical misfit (not Phi', which can be negative), about N/2 .. N here
    for row in out["starts"]:
        assert abs(row["dU"]) < 1e-3 and row["rms_dtheta"] < 1e-3  # same basin, same field
        assert row["phi"] == pytest.approx(ref_phi, abs=0.05)
        assert row["rms_dtheta_over_laplace_std"] < 0.1
    # a different local minimum would show up as dU > 0 and a field difference; check that the measure can see one:
    # stopping after a handful of plain iterations leaves a visible difference
    short = ms.main(TINY + ["--map-maxiter", "5", "--starts", "1", "--laplace-cache", str(tiny_run["cache"]),
                            "--out", str(tmp_path / "ms_short")])
    assert short["starts"][0]["dU"] > 1.0
