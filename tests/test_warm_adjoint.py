"""Warm-started adjoint solve (docs/DESIGN.md section 16): the aux carries the adjoint fields w next to the total fields u.

What must hold: the gradient with a warm adjoint equals the cold gradient to solver tolerance (same system, same relative
tolerance); the exported w really solves the adjoint system; a warm start needs fewer operator applications; the pair keeps
its structure inside ``lax.scan`` / ``lax.cond`` / the samplers; an old-style array aux keeps working and stays an array.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from bayes_scatter import samplers as sm
from bayes_scatter import shapemodels as shm
from bayes_scatter import shapepost as sp
from bayes_scatter import ellipses as el
from bayes_scatter import mrf
from bayes_scatter import thetaspace as ts
from bayes_scatter.forward import Forward, split_aux, value_and_grad_aux
from bayes_scatter.geometry import Antennas, Grid, Setup, ch15_setup, ring
from bayes_scatter.likelihood import Target, add_noise, misfit, noise_std_from_snr
from bayes_scatter.phantoms import cylinder
from bayes_scatter.priors import GaussianField

N = 16


def _setup():
    return Setup(Grid(N, 2.0), Antennas(ring(8, 4.0), ring(12, 4.5)))


def _chi(scale=1.0, bump=0.0):
    chi = np.zeros((N, N))
    chi[4:12, 5:10] = 1.5 * scale
    chi[8:13, 9:13] += bump
    return jnp.asarray(chi)


def _loss(fwd, d_obs):
    def f(chi, aux):
        d, u = fwd(chi, aux)
        return jnp.sum(jnp.abs(d - d_obs) ** 2), u
    return f


def _rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


@pytest.mark.parametrize("kwargs", [dict(), dict(batch="vmap"), dict(solver="bicgstab", maxiter=300)],
                         ids=["gmres-stack", "gmres-vmap", "bicgstab"])
def test_warm_adjoint_gradient_equals_cold_gradient_along_a_trajectory(kwargs):
    """Five nearby points (a trajectory): the pair-aux gradient equals the array-aux gradient and the tol = 1e-13
    reference to the solver tolerance (measured about 1e-12 relative at tol 1e-10); the carried w is nonzero and moves."""
    fwd = Forward(_setup(), tol=1e-10, **kwargs)
    ref = Forward(_setup(), tol=1e-13)
    d_obs = fwd(_chi())[0]
    f, f_ref = _loss(fwd, d_obs), _loss(ref, d_obs)
    aux_w = (fwd.u_inc, jnp.zeros_like(fwd.u_inc))
    aux_c = fwd.u_inc
    w_prev = None
    for k in range(5):
        chi = _chi(0.6 + 0.01 * k, bump=0.02 * k)
        v_w, g_w, aux_w = value_and_grad_aux(f, chi, aux_w)
        v_c, g_c, aux_c = value_and_grad_aux(f, chi, aux_c)
        g_ref = jax.grad(lambda c: f_ref(c, None)[0])(chi)
        assert abs(float(v_w - v_c)) < 1e-10 * float(v_c)
        assert _rel(g_w, g_c) < 1e-8 and _rel(g_w, g_ref) < 1e-8, (k, _rel(g_w, g_c), _rel(g_w, g_ref))
        w = aux_w[1]
        assert float(jnp.max(jnp.abs(w))) > 0
        if w_prev is not None:
            assert float(jnp.max(jnp.abs(w - w_prev))) > 0
        w_prev = w
        np.testing.assert_allclose(np.asarray(aux_w[0]), np.asarray(aux_c), atol=1e-8 * float(jnp.max(jnp.abs(aux_c))))


def test_array_aux_is_unchanged_and_stays_an_array():
    """Old callers: aux an array (or None) takes the custom_linear_solve path and returns an array."""
    fwd = Forward(_setup(), tol=1e-11)
    d_obs = fwd(_chi())[0]
    f = _loss(fwd, d_obs)
    v, g, aux = value_and_grad_aux(f, _chi(0.7), fwd.u_inc)
    assert not isinstance(aux, tuple) and aux.shape == fwd.u_inc.shape
    g_plain = jax.grad(lambda c: f(c, None)[0])(_chi(0.7))
    assert _rel(g, g_plain) < 1e-12
    assert split_aux(aux)[1] is None and split_aux((aux, aux))[1] is aux
    # forward mode still works on the plain path (the pair path is reverse mode only)
    jax.jvp(lambda c: fwd(c)[0], (_chi(0.7),), (_chi(0.1),))


def test_exported_adjoint_solves_the_adjoint_system_and_warm_start_saves_operator_applications():
    """w in the aux solves A^T w = (cotangent of u) to tolerance; starting the next adjoint solve from it needs fewer
    operator applications (counted with ``adjoint_matvec_counts``, residual evaluations included) than from zero."""
    fwd = Forward(_setup(), tol=1e-10)
    d_obs = fwd(_chi())[0]
    f = _loss(fwd, d_obs)
    s_ = 1.0

    def cotangent(chi, u):  # d loss / d u, the right-hand side the gradient hands to the adjoint solve
        return jax.grad(lambda uu: misfit(fwd.scattered(chi, uu), d_obs, s_))(u)

    aux = (fwd.u_inc, jnp.zeros_like(fwd.u_inc))
    chi1, chi2 = _chi(0.6), _chi(0.605, bump=0.01)
    _, _, aux = value_and_grad_aux(f, chi1, aux)
    u1, w1 = aux
    ct1 = cotangent(chi1, u1)
    res = w1 - chi1 * fwd.apply_G(w1) - ct1  # A^T w = w - diag(chi) G w (G symmetric) must equal the cotangent
    assert float(jnp.linalg.norm(res) / jnp.linalg.norm(ct1)) < 1e-9
    _, _, aux2 = value_and_grad_aux(f, chi2, aux)
    u2 = aux2[0]
    ct2 = cotangent(chi2, u2)
    idx = [0, 3, 7]
    cold = fwd.adjoint_matvec_counts(chi2, ct2, None, transmitters=idx)
    warm = fwd.adjoint_matvec_counts(chi2, ct2, w1, transmitters=idx)
    assert np.all(warm < cold), (cold, warm)
    # starting from the exact solution costs one residual evaluation
    exact = fwd.adjoint_matvec_counts(chi2, ct2, aux2[1], transmitters=idx)
    assert np.all(exact <= 2) and np.all(exact < warm)


def test_nonfinite_adjoint_start_is_discarded():
    fwd = Forward(_setup(), tol=1e-10)
    d_obs = fwd(_chi())[0]
    f = _loss(fwd, d_obs)
    bad = jnp.full(fwd.u_inc.shape, jnp.nan + 0j)
    _, g_bad, aux = value_and_grad_aux(f, _chi(0.7), (fwd.u_inc, bad))
    _, g_cold, _ = value_and_grad_aux(f, _chi(0.7), fwd.u_inc)
    assert _rel(g_bad, g_cold) < 1e-8 and bool(jnp.all(jnp.isfinite(aux[1])))


def test_pair_aux_structure_is_stable_inside_scan_and_cond():
    """The pair is a valid ``lax.scan`` carry (JAX checks structure, shapes and dtypes), also through ``lax.cond``."""
    fwd = Forward(_setup(), tol=1e-10)
    d_obs = fwd(_chi())[0]
    f = _loss(fwd, d_obs)
    chis = jnp.stack([_chi(0.6 + 0.01 * k) for k in range(4)])
    aux0 = (fwd.u_inc, jnp.zeros_like(fwd.u_inc))

    def body(aux, chi):
        v, g, aux = value_and_grad_aux(f, chi, aux)
        aux = lax.cond(v > 0, lambda a: a, lambda a: a, aux)
        return aux, (v, g)

    aux_end, (vs, gs) = jax.jit(lambda a, c: lax.scan(body, a, c))(aux0, chis)
    assert jax.tree.structure(aux_end) == jax.tree.structure(aux0)
    assert [x.shape for x in aux_end] == [x.shape for x in aux0] and all(x.dtype == jnp.complex128 for x in aux_end)
    for k in range(4):  # scan result = the cold gradients
        g_k = jax.grad(lambda c: f(c, None)[0])(chis[k])
        assert _rel(gs[k], g_k) < 1e-8


# ---- Target, ThetaTarget, ShapePosterior and the samplers ----


@pytest.fixture(scope="module")
def tgt_pair():
    setup = ch15_setup(n=8, side=1.0, n_tx=6, n_rx=10)
    fwd = Forward(setup, tol=1e-12)
    field = GaussianField(setup.grid, "matern", sigma=0.4, ell=0.3, nu=1.5, mean=0.2)
    xi_true = field.sample_xi(jax.random.key(0))
    d_clean, _ = fwd(jnp.exp(field.theta(xi_true)) - 1.0)
    s = noise_std_from_snr(d_clean, 20.0)
    d_obs = add_noise(jax.random.key(1), d_clean, s)
    cold = Target(fwd, field, "log_eps", d_obs, s)
    warm = Target(fwd, field, "log_eps", d_obs, s, warm_adjoint=True)
    return {"cold": cold, "warm": warm, "xi": 0.7 * xi_true + 0.3 * field.sample_xi(jax.random.key(2)), "fwd": fwd}


def test_target_pair_aux_gradient_value_and_passthrough(tgt_pair):
    cold, warm, xi = tgt_pair["cold"], tgt_pair["warm"], tgt_pair["xi"]
    assert isinstance(warm.initial_aux(), tuple) and not isinstance(cold.initial_aux(), tuple)
    phi_c, g_c, _ = cold.potential_and_grad(xi, cold.initial_aux())
    phi_w, g_w, aux = warm.potential_and_grad(xi, warm.initial_aux())
    assert abs(float(phi_w - phi_c)) < 1e-9 * float(phi_c) and _rel(g_w, g_c) < 1e-9
    xi2 = xi + 1e-3
    _, g_w2, aux2 = warm.potential_and_grad(xi2, aux)
    _, g_c2, _ = cold.potential_and_grad(xi2, cold.initial_aux())
    assert _rel(g_w2, g_c2) < 1e-9
    # value-only calls refresh u and leave w alone
    _, aux3 = warm.potential(xi2, aux)
    np.testing.assert_array_equal(np.asarray(aux3[1]), np.asarray(aux[1]))
    _, aux4 = warm.misfit_of_chi(warm.chi(xi2), aux)
    np.testing.assert_array_equal(np.asarray(aux4[1]), np.asarray(aux[1]))
    assert warm.solver_residual(xi2, aux2[0]) < 1e-9
    _, _, aux5 = warm.phi_fn_no_grad(xi2, aux)
    assert isinstance(aux5, tuple)


def test_target_with_old_array_aux_still_works_when_warm_adjoint_is_on(tgt_pair):
    warm, cold, xi = tgt_pair["warm"], tgt_pair["cold"], tgt_pair["xi"]
    _, g, aux = warm.potential_and_grad(xi, cold.initial_aux())
    assert not isinstance(aux, tuple)
    assert _rel(g, cold.potential_and_grad(xi, cold.initial_aux())[1]) < 1e-12


def test_hmc_with_pair_aux_reproduces_the_cold_chain(tgt_pair):
    """Same key, same start: the chain carries the pair through scan, cond and accept/reject and visits the same
    states as the cold chain to solver tolerance."""
    cold, warm, xi = tgt_pair["cold"], tgt_pair["warm"], tgt_pair["xi"]
    kw = dict(n_warmup=0, n_samples=6, step0=0.05, n_leapfrog=4)
    r_c = sm.hmc(jax.random.key(5), cold.phi_fn, xi, cold.initial_aux(), **kw)
    r_w = sm.hmc(jax.random.key(5), warm.phi_fn, xi, warm.initial_aux(), **kw)
    np.testing.assert_allclose(np.asarray(r_w.samples), np.asarray(r_c.samples), atol=1e-7)
    np.testing.assert_allclose(np.asarray(r_w.phi), np.asarray(r_c.phi), rtol=1e-8)
    assert jax.tree.structure(r_w.aux) == jax.tree.structure(warm.initial_aux())
    assert float(jnp.max(jnp.abs(r_w.aux[1]))) > 0


def test_thetatarget_and_shapeposterior_pair_aux():
    grid = Grid(6, 1.2)
    setup = Setup(grid, Antennas(ring(5, 2.0), ring(7, 2.4, offset=0.2)))
    fwd = Forward(setup, tol=1e-12)
    d_clean, _ = fwd(jnp.asarray(cylinder(grid, 0.4, 1.8)))
    s = noise_std_from_snr(d_clean, 30.0)
    d_obs = add_noise(jax.random.key(0), d_clean, s)
    prior = mrf.GaussianEnergy(GaussianField(grid, "matern", sigma=0.5, ell=0.3, nu=1.5))
    t_c = ts.ThetaTarget(fwd, prior, d_obs, s)
    t_w = ts.ThetaTarget(fwd, prior, d_obs, s, warm_adjoint=True)
    theta = jnp.asarray(0.1 * np.random.default_rng(3).standard_normal((6, 6)))
    _, g_c, _ = t_c.potential_and_grad(theta, t_c.initial_aux())
    _, g_w, aux = t_w.potential_and_grad(theta, t_w.initial_aux())
    _, g_w2, _ = t_w.potential_and_grad(theta + 1e-3, aux)
    assert _rel(g_w, g_c) < 1e-9
    assert _rel(g_w2, t_c.potential_and_grad(theta + 1e-3, t_c.initial_aux())[1]) < 1e-9
    assert float(t_w.misfit(theta, aux)[0]) == pytest.approx(float(t_c.misfit(theta)[0]), rel=1e-9)

    setup2 = ch15_setup(n=16, side=1.5, n_tx=6, n_rx=8)
    fwd2 = Forward(setup2, tol=1e-11)
    rend = el.EllipseRenderer(setup2.grid)
    from bayes_scatter import shapes as sh
    d2, _ = fwd2(jnp.asarray(sh.TwoDisks(g=0.2, r=0.2, eps_r=2.0).chi(setup2.grid)))
    s2 = noise_std_from_snr(d2, 50.0)
    d_obs2 = add_noise(jax.random.key(0), d2, s2)
    model = shm.make_model("F2")
    p_c = sp.ShapePosterior(model, fwd2, d_obs2, s2, rend)
    p_w = sp.ShapePosterior(model, fwd2, d_obs2, s2, rend, warm_adjoint=True)
    u = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
    v_c, g_c, _ = sp.potential(p_c, u)
    v_w, g_w, aux = sp.potential(p_w, u)
    v_w2, g_w2, aux2 = sp.potential(p_w, u + 1e-3, aux=aux)
    v_c2, g_c2, _ = sp.potential(p_c, u + 1e-3)
    assert abs(v_w - v_c) < 1e-8 * abs(v_c) and np.linalg.norm(g_w - g_c) < 1e-8 * np.linalg.norm(g_c)
    assert abs(v_w2 - v_c2) < 1e-8 * abs(v_c2) and np.linalg.norm(g_w2 - g_c2) < 1e-8 * np.linalg.norm(g_c2)
    assert isinstance(aux2, tuple)
    # beta = 0 skips the data term through lax.cond and returns the aux unchanged
    _, _, aux0 = sp.potential(p_w, u, beta=0.0, aux=aux)
    np.testing.assert_array_equal(np.asarray(aux0[1]), np.asarray(aux[1]))
    # forward-mode helpers accept the pair and use its fields
    assert p_w.fisher(u).shape == (5, 5)


# ---- the measurement scripts ----


def test_measure_script_and_table_on_a_tiny_grid(tmp_path, capsys):
    """scripts/measure_adjoint_warm.py end to end on a tiny grid: both chains see the same points, the gradients agree to
    solver tolerance, operator applications are counted, and scripts/adjoint_warm_table.py pools the file."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import adjoint_warm_table as awt
    import measure_adjoint_warm as maw

    out = tmp_path / "rec.json"
    maw.main(["--grids", "12:1.5", "--n-tx", "6", "--n-rx", "8", "--n-traj", "2", "--burn", "1", "--n-leapfrog", "3",
              "--counts", "2", "--count-points", "2", "--restart", "30", "--tag", "tiny", "--out", str(out)])
    rec = json.loads(out.read_text())
    assert len(rec["t_cold"]) == len(rec["t_warm"]) == len(rec["pos"]) == 6 and rec["pos"][:3] == [0, 1, 2]
    assert max(rec["grad_rel_diff"]) < 1e-8 and max(rec["phi_rel_diff"]) < 1e-9
    assert len(rec["counts"]["cold"]) == len(rec["counts"]["warm"]) == 4
    # no ordering is asserted here: on this tiny problem consecutive points are far apart in data space and a warm start
    # can cost an operation more than a cold one (the counts are the script's measurement, not a guarantee)
    assert min(rec["counts"]["warm"]) >= 1 and min(rec["counts"]["cold"]) >= 1
    capsys.readouterr()
    awt.main([str(out)])
    table = capsys.readouterr().out
    assert "tiny" in table and "12:1.5" in table
