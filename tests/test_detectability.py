"""Tests for bayes_scatter.detectability (DESIGN 15): the first-order law, the D2 action on data and the boundary deformation modes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import detectability as dt
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, ch15_setup

SETUP = ch15_setup(n=16, side=1.5, n_tx=8, n_rx=12)
GRID = SETUP.grid
FWD = Forward(SETUP, tol=1e-12)
REND = el.EllipseRenderer(GRID)
MODEL = sm.F2()
P0 = np.array([0.1, 0.2, 2.0, 0.0, 0.0])
S = 2e-3
ACT = dt.D2Action(SETUP.antennas)
OBJ = sh.TwoDisks(g=0.1, r=0.2, eps_r=2.0)


@pytest.fixture(scope="module")
def jac():
    D, chi0 = dt.whitened_jacobian(MODEL, FWD, REND, P0, S)
    return D, chi0


# ----------------------------------------------------------------------------------------------
# The linear-Gaussian toy: the geometric formula against the exact least-squares fit
# ----------------------------------------------------------------------------------------------


def test_c_geom_equals_the_exact_excess_over_the_square_of_the_gap_shift_in_a_linear_gaussian_toy():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((60, 5))
    delta = rng.standard_normal(60) * 0.3 + D @ rng.standard_normal(5) * 0.2
    out = dt.analyze_error(D, delta, k=0)
    # exact fit of the model p -> D p to the "data" delta (model error at the base point): shift dp, residual, sigma from the normal equations
    dp, *_ = np.linalg.lstsq(D, delta, rcond=None)
    sigma = np.sqrt(np.linalg.inv(D.T @ D)[0, 0])
    z_exact = dp[0] / sigma
    dphi_exact = 0.5 * np.sum((delta - D @ dp) ** 2)
    assert out["z"] == pytest.approx(z_exact, rel=1e-12)
    assert out["dphi"] == pytest.approx(dphi_exact, rel=1e-12)
    assert out["c_geom"] == pytest.approx(dphi_exact / z_exact**2, rel=1e-10)
    assert out["c_direct"] == pytest.approx(out["c_geom"], rel=1e-10)
    np.testing.assert_allclose(out["dp"], dp, rtol=1e-10)
    assert out["sigma_k"] == pytest.approx(sigma, rel=1e-12)
    assert out["sigma"][0] == pytest.approx(sigma, rel=1e-12)
    # c does not depend on the size of the error
    scaled = dt.analyze_error(D, 7.0 * delta, k=0)
    assert scaled["c_geom"] == pytest.approx(out["c_geom"], rel=1e-10)
    assert scaled["z"] == pytest.approx(7.0 * out["z"], rel=1e-12)


def test_w_hat_is_orthogonal_to_the_other_tangent_directions_and_projectors_are_idempotent():
    rng = np.random.default_rng(5)
    D = rng.standard_normal((40, 4))
    w, sigma = dt.gap_direction(D, k=1)
    assert np.linalg.norm(w) == pytest.approx(1.0, rel=1e-14)
    assert np.abs(np.delete(D, 1, axis=1).T @ w).max() < 1e-13
    # w lies in range(D) and equals v / ||v|| with v = D (D^T D)^-1 e_k
    v = D @ np.linalg.solve(D.T @ D, np.eye(4)[1])
    np.testing.assert_allclose(w, v / np.linalg.norm(v), atol=1e-13)
    assert sigma == pytest.approx(np.linalg.norm(v), rel=1e-12)
    P = dt.projector_perp(D)
    np.testing.assert_allclose(P @ P, P, atol=1e-13)
    np.testing.assert_allclose(P, P.T, atol=1e-14)
    assert np.abs(P @ D).max() < 1e-13
    assert np.abs(P @ w).max() < 1e-13


def test_whitened_jacobian_gives_the_fisher_matrix_of_the_shape_posterior(jac):
    """D^T D in p equals the Fisher matrix of ``ShapePosterior`` in u transformed by dp/du (the whitening and the factor 1/2 of DESIGN 15.1)."""
    D, _ = jac
    d0 = np.asarray(FWD(jnp.asarray(REND.chi(*MODEL.scene(jnp.asarray(P0)))))[0])
    post = sp.ShapePosterior(MODEL, FWD, d0, S, REND)
    u0 = np.asarray(MODEL.u_from_p(jnp.asarray(P0)))
    dpdu = np.asarray(jax.jacfwd(MODEL.p_from_u)(jnp.asarray(u0)))
    F_u = post.fisher(u0)
    F_from_D = dpdu.T @ (D.T @ D) @ dpdu
    assert np.abs(F_u - F_from_D).max() / np.abs(F_u).max() < 1e-8


# ----------------------------------------------------------------------------------------------
# The D2 action
# ----------------------------------------------------------------------------------------------


def test_the_group_acts_on_data_by_exact_permutations_forming_d2():
    n = ACT.n_tx * ACT.n_rx
    mats = {g: ACT.permutation_matrix(g) for g in dt.GROUP}
    for g, M in mats.items():
        assert set(np.unique(M)) <= {0.0, 1.0}
        assert np.all(M.sum(axis=0) == 1) and np.all(M.sum(axis=1) == 1)
        np.testing.assert_array_equal(M @ M, np.eye(n))  # involution
    np.testing.assert_array_equal(mats["sx"] @ mats["sy"], mats["c2"])
    np.testing.assert_array_equal(mats["sx"] @ mats["sy"], mats["sy"] @ mats["sx"])
    np.testing.assert_array_equal(mats["e"], np.eye(n))
    d = np.random.default_rng(0).standard_normal((8, 12)) + 1j * np.random.default_rng(1).standard_normal((8, 12))
    np.testing.assert_array_equal(ACT.apply("sx", d).ravel(), mats["sx"] @ d.ravel())
    y = dt.stack_real(d)
    np.testing.assert_array_equal(ACT.apply_stacked("sy", y), dt.stack_real(ACT.apply("sy", d)))
    np.testing.assert_array_equal(dt.unstack_real(y, (8, 12)), d)


def test_the_book_rings_are_invariant_with_40_transmitters_and_60_receivers():
    act = dt.D2Action(ch15_setup(n=8, side=1.5).antennas)
    assert np.array_equal(act.perm_tx["sx"], (20 - np.arange(40)) % 40)
    assert np.array_equal(act.perm_tx["sy"], (-np.arange(40)) % 40)
    assert np.array_equal(act.perm_rx["sx"], (30 - np.arange(60)) % 60)
    assert np.array_equal(act.perm_rx["sy"], (-np.arange(60)) % 60)


def test_an_antenna_set_without_the_symmetry_is_rejected():
    from bayes_scatter.geometry import Antennas, ring

    with pytest.raises(ValueError, match="not invariant"):
        dt.D2Action(Antennas(tx=ring(8, 4.0, offset=0.1), rx=ring(12, 4.5)))


def test_isotypic_components_are_orthogonal_idempotent_and_sum_to_the_identity():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(2 * 96)
    comps = ACT.components(y, stacked=True)
    np.testing.assert_allclose(sum(comps.values()), y, atol=1e-13)
    for a in dt.IRREPS:
        np.testing.assert_allclose(ACT.project(a, comps[a], stacked=True), comps[a], atol=1e-13)
        for b in dt.IRREPS:
            if a != b:
                assert abs(comps[a] @ comps[b]) < 1e-12
                assert np.abs(ACT.project(a, comps[b], stacked=True)).max() < 1e-13
    assert sum(float(v @ v) for v in comps.values()) == pytest.approx(float(y @ y), rel=1e-13)
    fr = ACT.fractions(y)
    assert sum(v["frac_norm2"] for v in fr.values()) == pytest.approx(1.0, rel=1e-13)
    # the character of each component: P_g v = chi(g) v
    for name, v in comps.items():
        sx, sy = dt.SIGNS[name]
        np.testing.assert_allclose(ACT.apply_stacked("sx", v), sx * v, atol=1e-13)
        np.testing.assert_allclose(ACT.apply_stacked("sy", v), sy * v, atol=1e-13)


def test_the_forward_map_is_equivariant_and_a_symmetric_object_has_symmetric_data():
    rng = np.random.default_rng(4)
    chi = 0.8 * rng.random((16, 16)) * (rng.random((16, 16)) < 0.4)  # a generic object without symmetry
    d, _ = FWD(jnp.asarray(chi))
    for g in ("sx", "sy", "c2"):
        dg, _ = FWD(jnp.asarray(dt.flip_chi(chi, g).copy()))
        np.testing.assert_allclose(np.asarray(dg), ACT.apply(g, np.asarray(d)), atol=1e-11 * np.abs(np.asarray(d)).max())
    chi_sym = OBJ.chi(GRID)
    ds, _ = FWD(jnp.asarray(chi_sym))
    ds = np.asarray(ds)
    for g in ("sx", "sy", "c2"):
        np.testing.assert_allclose(dt.flip_chi(chi_sym, g), chi_sym, atol=1e-13)
        np.testing.assert_allclose(ACT.apply(g, ds), ds, atol=1e-11 * np.abs(ds).max())
    comps = ACT.components(dt.stack_real(ds), stacked=True)
    assert max(np.linalg.norm(comps[k]) for k in ("B1", "B2", "A2")) < 1e-10 * np.linalg.norm(comps["A1"])


# ----------------------------------------------------------------------------------------------
# Boundary deformation modes
# ----------------------------------------------------------------------------------------------


def test_circle_mode_integrals_reduce_to_the_basis_integrals_and_match_a_polar_rendering():
    grid = Grid(24, 1.5)
    xc, yc, rad = 0.21, -0.03, 0.2
    basis = sh.circle_basis_integrals(grid, xc, yc, rad)
    np.testing.assert_allclose(sh.circle_mode_integrals(grid, xc, yc, rad, 1), basis[1:], atol=1e-14)
    np.testing.assert_allclose(sh.circle_mode_integrals(grid, xc, yc, rad, 0)[0], basis[0], atol=1e-14)
    assert np.abs(sh.circle_mode_integrals(grid, xc, yc, rad, 0)[1]).max() == 0.0
    for m in (2, 3, 5):
        ints = sh.circle_mode_integrals(grid, xc, yc, rad, m)
        assert np.abs(ints.sum(axis=(1, 2))).max() < 1e-13  # the area does not change to first order
    # polar rendering of rho <= R + eps f(phi) by dense sub-sampling: the difference quotient reproduces the exact integrals
    ss, eps = 60, 2e-3
    sub = (np.arange(ss) + 0.5) / ss
    xs = (-0.75 + (np.arange(grid.n)[:, None] + sub[None, :]) * grid.h).ravel()
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    phi = np.arctan2(Y - yc, X - xc)
    rho = np.hypot(X - xc, Y - yc)

    def cover(f):
        inside = (rho <= rad + eps * f).astype(float)
        return inside.reshape(grid.n, ss, grid.n, ss).mean(axis=(1, 3))

    for m, kind in ((2, 0), (3, 1)):
        f = np.cos(m * phi) if kind == 0 else np.sin(m * phi)
        fd = (cover(f) - cover(-f)) / (2.0 * eps)
        exact = sh.circle_mode_integrals(grid, xc, yc, rad, m)[kind] / grid.h**2
        assert np.abs(fd - exact).max() < 0.1 * np.abs(exact).max()  # sub-sampling quantization 1 / ss^2 over eps h


def test_mode_action_is_the_d2_action_on_the_deformation_fields():
    """dchi(M_g c) = flip_g(dchi(c)); the irrep vectors are eigenvectors with the characters of their irrep and span the 4 dimensional space."""
    for m in (0, 1, 2, 3, 4):
        dchi = dt.mode_chi_derivatives(GRID, OBJ, m)
        rng = np.random.default_rng(m)
        c = rng.standard_normal(4)
        for g in dt.GROUP:
            M = dt.mode_action(m, g)
            lhs = np.einsum("i,ijk->jk", M @ c, dchi)
            rhs = dt.flip_chi(np.einsum("i,ijk->jk", c, dchi), g)
            assert np.abs(lhs - rhs).max() < 1e-12 * max(np.abs(rhs).max(), 1.0)
        np.testing.assert_allclose(dt.mode_action(m, "sx") @ dt.mode_action(m, "sx"), np.eye(4), atol=1e-15)
        np.testing.assert_allclose(dt.mode_action(m, "sx") @ dt.mode_action(m, "sy"), dt.mode_action(m, "sy") @ dt.mode_action(m, "sx"), atol=1e-15)
        basis = np.array([dt.irrep_mode_coefficients(m, a) for a in dt.IRREPS])
        if m > 0:
            assert np.linalg.matrix_rank(basis) == 4
        for a in dt.IRREPS:
            v = dt.irrep_mode_coefficients(m, a)
            sx, sy = dt.SIGNS[a]
            np.testing.assert_allclose(dt.mode_action(m, "sx") @ v, sx * v, atol=1e-15)
            np.testing.assert_allclose(dt.mode_action(m, "sy") @ v, sy * v, atol=1e-15)
            np.testing.assert_allclose(dt.mode_irrep_projector(m, a) @ v, v, atol=1e-15)
            for b in dt.IRREPS:
                if b != a:
                    assert np.abs(dt.mode_irrep_projector(m, b) @ v).max() < 1e-15


def test_non_a1_deformations_of_a_symmetric_object_have_z_at_round_off_and_a1_ones_do_not(jac):
    D, chi0 = jac
    w_hat, sigma_g = dt.gap_direction(D, 0)
    for m in (2, 3, 4):
        tau = dt.mode_tangents(FWD, OBJ, m, S, chi0)  # (4, 2N) in the order Rc, Rs, Lc, Ls
        z = {}
        for a in dt.IRREPS:
            t = dt.irrep_mode_coefficients(m, a) @ tau
            assert np.linalg.norm(t) > 0
            # the tangent lies in the irrep of the data space
            comp = ACT.project(a, t, stacked=True)
            assert np.linalg.norm(t - comp) < 1e-8 * np.linalg.norm(t)
            z[a] = float(w_hat @ t) / np.linalg.norm(t)
        assert abs(z["A1"]) > 1e-3  # the A1 deformation does move the gap direction
        for a in ("B1", "B2", "A2"):
            assert abs(z[a]) < 1e-9


def test_m0_and_m1_a1_deformations_are_inside_the_model(jac):
    """m = 0 (both radii up, centers fixed) is d_r - 2 d_g and A1 m = 1 (cos phi right, -cos phi left) is 2 d_g: alpha = 0, c = 0, dz/d eps = -+ 2 / sigma_g."""
    D, chi0 = jac
    w_hat, sigma_g = dt.gap_direction(D, 0)
    for m, expected in ((0, -2.0 / sigma_g), (1, 2.0 / sigma_g)):
        tau = dt.mode_tangents(FWD, OBJ, m, S, chi0)
        t = dt.irrep_mode_coefficients(m, "A1") @ tau
        res = dt.analyze_error(D, t, k=0)
        assert np.sin(res["alpha"]) < 1e-8
        assert res["z"] == pytest.approx(expected, rel=1e-8)
        assert res["c_geom"] < 1e-12  # c = 0 up to round-off
    # m = 1 B1 is the x translation and B2 the common y translation, which F2 also contains
    for irrep in ("B1", "B2"):
        t = dt.irrep_mode_coefficients(1, irrep) @ dt.mode_tangents(FWD, OBJ, 1, S, chi0)
        res = dt.analyze_error(D, t, k=0)
        assert np.sin(res["alpha"]) < 1e-8


def test_a_high_harmonic_a1_deformation_is_outside_the_model_with_a_finite_c(jac):
    D, chi0 = jac
    t = dt.irrep_mode_coefficients(3, "A1") @ dt.mode_tangents(FWD, OBJ, 3, S, chi0)
    res = dt.analyze_error(D, t, k=0)
    assert 1e-3 < np.sin(res["alpha"]) <= 1.0
    assert np.isfinite(res["c_geom"]) and res["c_geom"] > 0
    assert res["c_direct"] == pytest.approx(res["c_geom"], rel=1e-10)


def test_noise_free_fit_recovers_the_parameters_of_its_own_data():
    p_true = np.array([0.12, 0.21, 2.1, 0.01, -0.01])
    d = np.asarray(FWD(jnp.asarray(REND.chi(*MODEL.scene(jnp.asarray(p_true)))))[0])
    post = sp.ShapePosterior(MODEL, FWD, d, 2e-5, REND)  # informative data, so that the weak prior does not move the fit
    p_hat, phi, ok = dt.noise_free_fit(post, np.array([0.1, 0.2, 2.0, 0.0, 0.0]))
    assert ok and phi < 1e-2
    assert abs(p_hat[0] - p_true[0]) < 5e-4


# ----------------------------------------------------------------------------------------------
# Review of milestone 8: the gap-error convention, second order, the foot point
# ----------------------------------------------------------------------------------------------


def _ellipse_pair(level, gap=0.1, r=0.2, eps_r=2.0):
    ell = np.array([[-(r + 0.5 * gap), 0.0, r, (1.0 + level) * r, 0.0], [r + 0.5 * gap, 0.0, r, (1.0 + level) * r, 0.0]])
    return el.EllipseScene(ell, eps_r, name="T2e", true_gap=gap, description="test")


def test_gap_change_per_amplitude_of_each_irrep_and_harmonic():
    for m in range(0, 7):
        s = (-1.0) ** m
        assert dt.gap_change_per_amplitude(m, dt.irrep_mode_coefficients(m, "A1")) == pytest.approx(-2.0 * s)
        assert dt.gap_change_per_amplitude(m, dt.irrep_mode_coefficients(m, "B1")) == pytest.approx(0.0)
        assert dt.gap_change_per_amplitude(m, dt.irrep_mode_coefficients(m, "B2")) == pytest.approx(0.0)
        assert dt.gap_change_per_amplitude(m, dt.irrep_mode_coefficients(m, "A2")) == pytest.approx(0.0)


def test_true_gap_tangent_of_in_model_modes_has_zero_gap_error_and_the_gap_change_is_measured_geometrically(jac):
    """m = 0 and the A1 m = 1 mode are in the model: the error of the gap estimate is z_true = 0 (the base-gap response is -+2 / sigma_g)."""
    D, chi0 = jac
    w_hat, sigma_g = dt.gap_direction(D, 0)
    tau0 = dt.irrep_mode_coefficients(0, "A1") @ dt.mode_tangents(FWD, OBJ, 0, S, chi0)
    for m in (0, 1):
        coef = dt.irrep_mode_coefficients(m, "A1")
        tau = coef @ dt.mode_tangents(FWD, OBJ, m, S, chi0)
        gp = dt.true_gap_tangent(tau, tau0, m, coef)
        assert abs(w_hat @ gp) < 1e-6 * abs(w_hat @ tau)  # z_true = 0 against |z_fit| = 2 / sigma_g
        assert abs(w_hat @ tau) == pytest.approx(2.0 / sigma_g, rel=1e-8)
    # P_perp tau0 = 0: dPhi is the same in both conventions
    coef = dt.irrep_mode_coefficients(3, "A1")
    tau = coef @ dt.mode_tangents(FWD, OBJ, 3, S, chi0)
    a = dt.analyze_error(D, tau, 0)
    b = dt.analyze_error(D, dt.true_gap_tangent(tau, tau0, 3, coef), 0)
    assert b["dphi"] == pytest.approx(a["dphi"], rel=1e-8)
    assert b["z"] == pytest.approx(a["z"] + 2.0 * (-1.0) ** 3 / sigma_g, rel=1e-8)  # z_true = z_fit - gap_change / sigma_g, gap_change = -2 (-1)^m


def test_ellipticity_is_the_gap_preserving_m2_mode_and_c_true_reproduces_the_t2_linear_c(jac):
    """Radial displacement of the T2 ellipse (r, (1 + e) r) about each center is e r sin^2 phi = (e r / 2)(1 - cos 2 phi): the m = 0 mode at +e r/2 plus the A1 m = 2 mode
    at -e r/2, a gap-preserving deformation. Its data change is (e r / 2)(tau_0 - tau_2); c computed from the ellipse data equals c_true of the m = 2 mode (1 %)."""
    D, chi0 = jac
    e = 2e-3
    tau = dt.mode_tangents(FWD, OBJ, 2, S, chi0)
    coef2, coef0 = dt.irrep_mode_coefficients(2, "A1"), dt.irrep_mode_coefficients(0, "A1")
    t2 = coef2 @ tau
    t0 = coef0 @ dt.mode_tangents(FWD, OBJ, 0, S, chi0)
    d_ell = np.asarray(FWD(jnp.asarray(_ellipse_pair(e).chi(GRID)))[0])
    d_circ = np.asarray(FWD(jnp.asarray(chi0))[0])
    delta = dt.stack_real(d_ell - d_circ, S)
    pred = 0.5 * e * 0.2 * (t0 - t2)
    assert np.linalg.norm(delta - pred) < 0.02 * np.linalg.norm(pred)
    ell = dt.analyze_error(D, delta, 0)
    gp = dt.analyze_error(D, dt.true_gap_tangent(t2, t0, 2, coef2), 0)
    assert gp["c_geom"] == pytest.approx(ell["c_geom"], rel=0.02)
    assert abs(ell["z"]) == pytest.approx(0.5 * e * 0.2 * abs(gp["z"]), rel=0.02)
    # the base-gap response of the same mode differs strongly (the old convention): z_fit = z_true - 2 s / sigma_g per unit amplitude
    fit = dt.analyze_error(D, t2, 0)
    assert fit["z"] == pytest.approx(gp["z"] - 2.0 / dt.gap_direction(D, 0)[1], rel=1e-6)


def test_second_order_bias_matches_the_nonlinear_fit_of_a_quadratic_model_to_second_order():
    from scipy.optimize import least_squares

    rng = np.random.default_rng(11)
    n, k = 80, 4
    D = rng.standard_normal((n, k))
    H = rng.standard_normal((n, k, k)) * 0.5
    H = 0.5 * (H + H.transpose(0, 2, 1))
    d0 = rng.standard_normal(n)

    def p_exact(scale):
        delta = scale * d0
        sol = least_squares(lambda p: delta - D @ p - 0.5 * np.einsum("ijl,j,l->i", H, p, p), np.zeros(k), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        return sol.x

    errs1, errs2 = [], []
    for scale in (0.02, 0.04):
        so = dt.second_order_bias(D, H, scale * d0)
        px = p_exact(scale)
        errs1.append(np.linalg.norm(px - so["p1"]))
        errs2.append(np.linalg.norm(px - so["p"]))
        np.testing.assert_allclose(so["p2"], so["p2_tangential"] + so["p2_normal"], atol=1e-15)
    assert errs2[0] < 0.1 * errs1[0] and errs2[1] < 0.1 * errs1[1]  # the series is correct at order delta^2
    assert errs2[1] / errs2[0] == pytest.approx(8.0, rel=0.3)  # the remainder is O(delta^3)


def test_residual_curvature_operator_is_the_gap_between_the_true_and_the_gauss_newton_hessian():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((50, 3))
    H = rng.standard_normal((50, 3, 3))
    H = 0.5 * (H + H.transpose(0, 2, 1))
    r = rng.standard_normal(50) * 0.05
    lam, A = dt.residual_curvature_operator(D, H, r)
    full = D.T @ D - np.einsum("i,ijl->jl", r, H)  # Hessian of Phi = (1/2) ||y - mu||^2
    ev = np.sort(np.linalg.eigvals(np.linalg.solve(D.T @ D, np.einsum("i,ijl->jl", r, H))).real)[::-1]
    np.testing.assert_allclose(lam, ev, atol=1e-12)
    # det(full) = det(D^T D) prod (1 - lambda)
    assert np.linalg.det(full) == pytest.approx(np.linalg.det(D.T @ D) * np.prod(1.0 - lam), rel=1e-10)
    assert dt.log_evidence_gap(lam) == pytest.approx(-0.5 * (np.linalg.slogdet(full)[1] - np.linalg.slogdet(D.T @ D)[1]), rel=1e-10)


def test_curvature_tensor_agrees_with_second_differences_of_the_data_map(jac):
    D, chi0 = jac
    steps = dt.default_steps(P0)
    H, asym = dt.curvature_tensor(MODEL, FWD, REND, P0, S, steps)
    assert asym < 1e-4  # H is symmetric: the finite-difference error indicator

    def y(p):
        return dt.stack_real(np.asarray(FWD(jnp.asarray(REND.chi(*MODEL.scene(jnp.asarray(p, dtype=jnp.float64)))))[0]), S)

    for j in (0, 1, 2):
        h = 5e-3 * max(abs(P0[j]), 0.1)
        e = np.zeros(5)
        e[j] = h
        sd = (y(P0 + e) + y(P0 - e) - 2.0 * y(P0)) / h**2
        assert np.linalg.norm(sd - H[:, j, j]) < 2e-3 * np.linalg.norm(sd)


def test_foot_point_fixes_the_gap_and_zeroes_the_nuisance_gradient():
    p_true = np.array([0.1, 0.2, 2.0, 0.0, 0.0])
    scene = _ellipse_pair(0.03, gap=0.1)
    d = np.asarray(FWD(jnp.asarray(scene.chi(GRID)))[0])
    y = dt.stack_real(d, S)
    p, phi, ok = dt.foot_point(MODEL, FWD, REND, y, S, p_true, k=0)
    assert ok and p[0] == p_true[0]
    D, chi_p = dt.whitened_jacobian(MODEL, FWD, REND, p, S)
    delta = y - dt.stack_real(np.asarray(FWD(jnp.asarray(chi_p))[0]), S)
    assert np.abs(D[:, 1:].T @ delta).max() < 1e-6 * np.linalg.norm(D[:, 1:], axis=0).max() * np.linalg.norm(delta)
    assert phi == pytest.approx(0.5 * float(delta @ delta), rel=1e-12)
    # the starting point (all parameters at the nominal values) has a larger misfit
    r0 = y - dt.stack_real(np.asarray(FWD(jnp.asarray(REND.chi(*MODEL.scene(jnp.asarray(p_true)))))[0]), S)
    assert phi <= 0.5 * float(r0 @ r0)
