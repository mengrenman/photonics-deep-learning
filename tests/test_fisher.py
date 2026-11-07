"""Tests for bayes_scatter.fisher: data derivatives along shape parameters, Fisher matrix, CRB scaling, Born comparison, symmetry."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from bayes_scatter import fisher as fi
from bayes_scatter import jacobian as jb
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Antennas, Grid, Setup, ring
from bayes_scatter.likelihood import noise_std_from_snr

SYMMETRIC = sh.TwoDisks(g=0.15, r=0.2, eps_r=2.0)
ASYMMETRIC = sh.TwoDisks(g=0.15, r=0.2, eps_r=2.3, cx=0.0113, cy=-0.0071, dr=0.031)
GENERIC_NAMES = sh.PARAM_NAMES


@pytest.fixture(scope="module")
def small():
    """24 x 24 grid on a 1.5 x 1.5 DOI, 8 transmitters (r = 4) and 12 receivers (r = 4.5): both rings are invariant under
    x -> -x and y -> -y (angles 45 and 30 degrees apart, starting at 0), tight solver tolerance."""
    setup = Setup(Grid(24, 1.5), Antennas(ring(8, 4.0), ring(12, 4.5)))
    fwd = Forward(setup, tol=1e-12)
    out = {"fwd": fwd, "grid": setup.grid}
    for key, obj in (("sym", SYMMETRIC), ("asym", ASYMMETRIC)):
        chi = obj.chi(setup.grid)
        dchi = sh.shape_derivatives(setup.grid, obj, GENERIC_NAMES)
        d, _ = fwd(jnp.asarray(chi))
        out[key] = {"obj": obj, "chi": chi, "dchi": dchi, "d": d, "D": fi.data_derivatives(fwd, chi, dchi)}
    return out


def test_data_derivatives_match_finite_differences_of_the_forward_model(small):
    """D[:, k] = jvp of the forward map along the exact dchi/dp_k, against central differences of F(chi(p)) with the exact rendering
    (error: truncation ~step^2 of the rendering and the solver tolerance 1e-12 divided by the step, a few 1e-8)."""
    fwd, grid, rec = small["fwd"], small["grid"], small["asym"]
    D = np.asarray(rec["D"])
    assert D.shape == (fwd.n_tx * fwd.n_rx, len(GENERIC_NAMES)) and D.dtype == np.complex128
    step = 3e-5
    for k, name in enumerate(GENERIC_NAMES):
        hi = rec["obj"].replace(**{name: getattr(rec["obj"], name) + step}).chi(grid)
        lo = rec["obj"].replace(**{name: getattr(rec["obj"], name) - step}).chi(grid)
        fd = (np.asarray(fwd(jnp.asarray(hi))[0]) - np.asarray(fwd(jnp.asarray(lo))[0])).ravel() / (2.0 * step)
        assert np.linalg.norm(D[:, k] - fd) / np.linalg.norm(fd) < 2e-6, name  # measured 5e-10 to 2e-7


def test_data_derivatives_equal_the_reciprocity_form(small):
    """dd[t, r] = sum_p w_r[p] u_t[p] dchi[p] (jacobian.py): the tangent solves and the Nt + Nr reciprocity solves agree."""
    fwd, rec = small["fwd"], small["asym"]
    u, w = jb.jacobian_factors(fwd, jnp.asarray(rec["chi"]))
    uf, wf = np.asarray(u).reshape(fwd.n_tx, -1), np.asarray(w).reshape(fwd.n_rx, -1)
    for k in range(len(GENERIC_NAMES)):
        dk = rec["dchi"][k].ravel()
        ref = np.einsum("tp,rp->tr", uf * dk[None, :], wf).ravel()
        D = np.asarray(rec["D"])[:, k]
        assert np.linalg.norm(D - ref) / np.linalg.norm(ref) < 1e-9


def test_fisher_matrix_is_symmetric_positive_semidefinite_and_matches_the_definition(small):
    rec = small["asym"]
    s = noise_std_from_snr(rec["d"], 30.0)
    F = fi.fisher_matrix(rec["D"], s)
    assert F.shape == (6, 6)
    np.testing.assert_array_equal(F, F.T)
    ev = np.linalg.eigvalsh(F)
    assert ev.min() > -1e-12 * ev.max()
    D = np.asarray(rec["D"])
    ref = (2.0 / s**2) * (D.real.T @ D.real + D.imag.T @ D.imag)  # Re(D^H D) = Re^T Re + Im^T Im
    # off-diagonal entries are sums with cancellation (|F_14| ~ 160 next to entries ~ 4e7), so their round-off is relative to the
    # matrix scale, not to the entry: 3e-12 relative on F_14 on the workstation (Linux, x86) against 1e-12 allowed before
    np.testing.assert_allclose(F, ref, rtol=1e-12, atol=1e-14 * np.abs(ref).max())
    # the score has covariance F: Monte Carlo over complex noise with E|e|^2 = s^2 (checks the factor 2 / s^2)
    rng = np.random.default_rng(0)
    e = (rng.standard_normal((100_000, D.shape[0])) + 1j * rng.standard_normal((100_000, D.shape[0]))) * s / np.sqrt(2.0)
    score = (2.0 / s**2) * np.real(e @ np.conj(D))  # sum_j conj(D_jk) e_j
    emp = score.T @ score / score.shape[0]
    assert np.abs(emp - F).max() / np.abs(F).max() < 0.03


def test_crb_scales_with_the_noise_and_the_required_snr_is_the_closed_form(small):
    """F proportional to 1 / s^2, so sigma_CRB proportional to s = 10^(-SNR / 20) times a constant at a fixed object; the SNR at which
    sigma_CRB(g) = g follows in closed form from the 30 dB value, and recomputing F at that SNR confirms it."""
    rec = small["sym"]
    names = list(GENERIC_NAMES)
    sub = [k for k in fi.SYMMETRIC_PARAMS]
    s30 = noise_std_from_snr(rec["d"], 30.0)
    F30 = fi.fisher_matrix(rec["D"], s30)
    F10 = fi.fisher_matrix(rec["D"], s30 * 10.0)
    np.testing.assert_allclose(F10, F30 / 100.0, rtol=1e-12)
    c30, c10 = fi.crb_std(F30, names, "g", sub), fi.crb_std(F10, names, "g", sub)
    assert c10 / c30 == pytest.approx(10.0, rel=1e-10)
    for conv in (fi.crb_std_known,):
        assert conv(F10, names, "g") / conv(F30, names, "g") == pytest.approx(10.0, rel=1e-10)
    g = rec["obj"].g
    snr_req = float(fi.required_snr_db(c30, g, 30.0))
    s_req = noise_std_from_snr(rec["d"], snr_req)
    F_req = fi.fisher_matrix(rec["D"], s_req)
    assert fi.crb_std(F_req, names, "g", sub) == pytest.approx(g, rel=1e-10)
    np.testing.assert_allclose(fi.scale_to_snr(F30, snr_req), F_req, rtol=1e-10)
    # more noise, more required SNR: sigma_CRB(30 dB) larger than g needs more than 30 dB
    assert fi.required_snr_db(10.0 * g, g, 30.0) == pytest.approx(50.0)
    assert fi.required_snr_db(0.1 * g, g, 30.0) == pytest.approx(10.0)


def test_crb_with_nuisances_unknown_is_never_below_the_known_bound(small):
    rec = small["asym"]
    names = list(GENERIC_NAMES)
    F = fi.fisher_matrix(rec["D"], noise_std_from_snr(rec["d"], 30.0))
    known = fi.crb_std_known(F, names, "g")
    assert known == pytest.approx(1.0 / np.sqrt(F[0, 0]))
    prev = known
    for subset in (("g", "r"), ("g", "r", "eps_r"), ("g", "r", "eps_r", "cx", "cy"), tuple(names)):
        c = fi.crb_std(F, names, "g", subset)
        assert c >= prev * (1.0 - 1e-12)  # adding unknown parameters can only raise the bound
        prev = c
    inv = np.linalg.inv(F)
    assert fi.crb_std(F, names, "g") == pytest.approx(np.sqrt(inv[0, 0]), rel=1e-9)
    assert fi.crb_std(np.zeros((2, 2)), ["g", "r"], "g") == float("inf")  # singular


def test_born_data_derivatives_and_fisher_equal_the_explicit_linear_computation(small):
    """Born model d = A chi with the dense matrix A of Forward.born_matrix: dd/dp_k = A dchi_k and F = (2 / s^2) Re(D^H D)."""
    fwd, rec = small["fwd"], small["asym"]
    A = np.asarray(fwd.born_matrix())
    Db = np.asarray(fi.born_data_derivatives(fwd, rec["dchi"]))
    ref = np.stack([A @ rec["dchi"][k].ravel() for k in range(6)], axis=1)
    np.testing.assert_allclose(Db, ref, rtol=1e-12, atol=1e-18 * np.abs(ref).max())
    s = noise_std_from_snr(rec["d"], 30.0)  # the full model's noise level, as in analyze
    Fb = fi.fisher_matrix(Db, s)
    expl = (2.0 / s**2) * (ref.real.T @ ref.real + ref.imag.T @ ref.imag)
    np.testing.assert_allclose(Fb, expl, rtol=1e-12)
    # the full model differs from the Born model at this contrast (the ratio is not 1), and the Born F is the same function of the
    # shape derivatives for every contrast scale: scaling dchi by c scales F_Born by c^2
    Dfull = np.asarray(rec["D"])
    assert np.abs(np.linalg.norm(Dfull, axis=0) / np.linalg.norm(Db, axis=0) - 1.0).max() > 0.02
    Fb2 = fi.fisher_matrix(fi.born_data_derivatives(fwd, 2.0 * rec["dchi"]), s)
    np.testing.assert_allclose(Fb2, 4.0 * Fb, rtol=1e-12)


def test_reflection_permutations_and_mirrored_data(small):
    ant = small["fwd"].setup.antennas
    for axis in (0, 1):
        for pts in (ant.tx, ant.rx):
            p = fi.reflection_permutation(pts, axis=axis)
            flip = np.array([-1.0, 1.0]) if axis == 0 else np.array([1.0, -1.0])
            np.testing.assert_allclose(pts[p], pts * flip, atol=1e-12)
            np.testing.assert_array_equal(p[p], np.arange(len(pts)))  # an involution
    with pytest.raises(ValueError):
        fi.reflection_permutation(ring(8, 4.0, offset=0.1))
    with pytest.raises(ValueError):
        fi.reflection_permutation(np.array([[1.0, 0.0], [2.0, 1.0]]))
    px = (fi.reflection_permutation(ant.tx), fi.reflection_permutation(ant.rx))
    d = np.arange(ant.tx.shape[0] * ant.rx.shape[0], dtype=float).reshape(ant.tx.shape[0], ant.rx.shape[0])
    np.testing.assert_array_equal(fi.mirror_data(d, *px), fi.mirror_data(fi.mirror_data(fi.mirror_data(d, *px), *px), *px))
    cols = fi.mirror_columns(d.reshape(-1, 1), *px).reshape(d.shape)
    np.testing.assert_array_equal(cols, fi.mirror_data(d, *px))


def test_the_even_odd_cross_block_vanishes_for_the_symmetric_object_and_not_for_an_asymmetric_one(small):
    """Negative control included: the same code at an asymmetric object (cx, dr, cy nonzero) gives an O(1) cross block."""
    fwd = small["fwd"]
    ant = fwd.setup.antennas
    names = list(GENERIC_NAMES)
    px = (fi.reflection_permutation(ant.tx), fi.reflection_permutation(ant.rx))
    py = (fi.reflection_permutation(ant.tx, axis=1), fi.reflection_permutation(ant.rx, axis=1))
    sym, asym = small["sym"], small["asym"]
    split = {}
    for key, rec in (("sym", sym), ("asym", asym)):
        F = fi.fisher_matrix(rec["D"], noise_std_from_snr(rec["d"], 30.0))
        split[key] = (fi.irrep_split(F, names), fi.irrep_split(F, names, fi.EVEN_PARAMS_Y, fi.ODD_PARAMS_Y),
                      fi.parity_residuals(rec["D"], names, *px), fi.parity_residuals(rec["D"], names, *py, fi.EVEN_PARAMS_Y, fi.ODD_PARAMS_Y))
    x_sym, y_sym, px_sym, py_sym = split["sym"]
    assert x_sym["even"] == ["g", "r", "eps_r", "cy"] and x_sym["odd"] == ["cx", "dr"]
    assert x_sym["cross_residual"] < 1e-9 and x_sym["cross_max_scaled"] < 1e-9
    assert y_sym["cross_residual"] < 1e-9  # cy (odd under y) decouples from the rest at the centered object
    assert max(px_sym.values()) < 1e-9 and max(py_sym.values()) < 1e-9
    x_as, y_as, px_as, py_as = split["asym"]
    assert x_as["cross_residual"] > 1e-2 and x_as["cross_max_scaled"] > 1e-2
    assert y_as["cross_residual"] > 1e-3
    assert max(px_as.values()) > 1e-2  # the columns have lost their parity
    # the diagonal blocks themselves are not small: the vanishing cross block is a property of the symmetry, not of F being small
    assert np.linalg.norm(x_sym["F_even"]) > 1e3 * x_sym["cross_residual"] * np.linalg.norm(x_sym["F_odd"]) ** 0.5
    # g and the nuisances: at the symmetric object the odd parameters do not change the gap bound
    F = fi.fisher_matrix(sym["D"], noise_std_from_snr(sym["d"], 30.0))
    with_odd = fi.crb_std(F, names, "g")
    without_odd = fi.crb_std(F, names, "g", ("g", "r", "eps_r", "cy"))
    assert with_odd == pytest.approx(without_odd, rel=1e-8)
    assert fi.crb_std(F, names, "g", ("g", "r", "eps_r")) == pytest.approx(without_odd, rel=1e-8)  # cy decouples as well
    # ... but not at the asymmetric one
    Fa = fi.fisher_matrix(asym["D"], noise_std_from_snr(asym["d"], 30.0))
    assert abs(fi.crb_std(Fa, names, "g") / fi.crb_std(Fa, names, "g", ("g", "r", "eps_r", "cy")) - 1.0) > 1e-4


def test_analyze_returns_a_serializable_record_with_consistent_fields(small):
    rec = fi.analyze(small["fwd"], SYMMETRIC)
    json.dumps(rec)  # no NumPy scalars or arrays left
    names = rec["names"]
    F = np.array(rec["F"])
    assert names == list(GENERIC_NAMES) and F.shape == (6, 6)
    assert rec["crb_g_known"] == pytest.approx(1.0 / np.sqrt(F[0, 0]))
    assert rec["crb_g_known"] <= rec["crb_g_unknown"] * (1.0 + 1e-12)
    assert rec["crb_g_unknown_all"] == pytest.approx(rec["crb_g_unknown"], rel=1e-8)  # dr is odd: it does not couple to g
    for key in ("unknown", "unknown_all", "known"):
        sigma = rec[f"crb_g_{key}"]
        assert rec["required_snr_db"][key] == pytest.approx(30.0 + 20.0 * np.log10(sigma / SYMMETRIC.g))
    assert rec["info_ratio_known"] == pytest.approx(F[0, 0] / np.array(rec["F_born"])[0, 0])
    assert rec["info_ratio_unknown"] == pytest.approx((rec["crb_g_unknown_born"] / rec["crb_g_unknown"]) ** 2)
    assert rec["irreps"]["cross_residual"] < 1e-9
    assert set(rec["timing_s"]) == {"forward_s", "tangents_s", "born_s"}
    assert rec["grid"]["n"] == 24 and rec["s"] > 0
    assert 0.0 <= rec["solver_residual_max"] < 100 * rec["solver_tol"]
    nob = fi.analyze(small["fwd"], SYMMETRIC, born=False, names=("g", "r", "eps_r"))
    assert "F_born" not in nob and nob["names"] == ["g", "r", "eps_r"] and nob["irreps"]["cross_residual"] == 0.0


def test_matched_power_born_ratio_is_the_ratio_with_each_model_at_its_own_snr(small):
    """The review of milestone 6: at one noise level F_full / F_Born mostly measures how much more power the Born model scatters. The matched ratio is computed here
    independently, by giving the Born data their OWN noise level (30 dB of the Born power) and taking F_full(s_full) / F_Born(s_Born), for the known and the unknown
    nuisances; ``analyze`` must agree, and the same-s ratio must be that times P_full / P_Born."""
    fwd, rec = small["fwd"], small["sym"]
    out = fi.analyze(fwd, SYMMETRIC)
    d_born = np.asarray(fwd.born(jnp.asarray(rec["chi"])))
    p_born, p_full = float(np.mean(np.abs(d_born) ** 2)), float(np.mean(np.abs(np.asarray(rec["d"])) ** 2))
    assert out["power_born"] == pytest.approx(p_born, rel=1e-12) and out["power"] == pytest.approx(p_full, rel=1e-12)
    assert out["power_ratio_full_over_born"] == pytest.approx(p_full / p_born, rel=1e-12) and p_full / p_born != pytest.approx(1.0, abs=1e-3)
    s_full = noise_std_from_snr(rec["d"], 30.0)
    s_born = noise_std_from_snr(d_born, 30.0)
    F = fi.fisher_matrix(rec["D"], s_full)
    Fb = fi.fisher_matrix(fi.born_data_derivatives(fwd, rec["dchi"]), s_born)  # the Born model at its own 30 dB
    names = list(GENERIC_NAMES)
    sub = ("g", "r", "eps_r", "cx", "cy")
    assert out["info_ratio_known_matched"] == pytest.approx(F[0, 0] / Fb[0, 0], rel=1e-10)
    assert out["info_ratio_unknown_matched"] == pytest.approx((fi.crb_std(Fb, names, "g", sub) / fi.crb_std(F, names, "g", sub)) ** 2, rel=1e-10)
    assert out["info_ratio_known"] == pytest.approx(out["info_ratio_known_matched"] * out["power_ratio_full_over_born"], rel=1e-10)
    assert out["info_ratio_unknown"] == pytest.approx(out["info_ratio_unknown_matched"] * out["power_ratio_full_over_born"], rel=1e-10)
    km, um, pr = fi.matched_power_ratios(out["info_ratio_known"], out["info_ratio_unknown"], p_full, p_born)
    assert km == pytest.approx(out["info_ratio_known_matched"]) and um == pytest.approx(out["info_ratio_unknown_matched"]) and pr == pytest.approx(p_full / p_born)
    # the two readings really differ at this contrast (the full model scatters less than the Born model): the same-s ratio is below 1 by about the power ratio
    assert out["info_ratio_known"] < 1.0 and out["info_ratio_known_matched"] > out["info_ratio_known"]


def test_nuisance_report_scales_with_the_snr_and_flags_the_non_local_regime(small):
    rec = fi.analyze(small["fwd"], SYMMETRIC, born=False)
    F, names = np.array(rec["F"]), rec["names"]
    truth = rec["params"]
    at30 = fi.nuisance_report(F, names, 30.0, truth, n_data=96)
    at10 = fi.nuisance_report(F, names, 10.0, truth, n_data=96)
    for k in at30["sd"]:
        assert at10["sd"][k] == pytest.approx(at30["sd"][k] * 10.0, rel=1e-12)  # sd ~ 10^(-(SNR - 30) / 20)
    assert at30["sd"]["g"] == pytest.approx(rec["crb_g_unknown"], rel=1e-12)
    assert at30["rel_sd"]["r"] == pytest.approx(at30["sd"]["r"] / truth["r"]) and at30["rel_sd"]["eps_r"] == pytest.approx(at30["sd"]["eps_r"] / (truth["eps_r"] - 1.0))
    assert at30["aggregate_snr_db"] == pytest.approx(30.0 + 10.0 * np.log10(96))
    # the flag is the 0.1 rule on both relative sds; the SNR at which it flips is where the larger one equals 0.1
    worst = max(at30["rel_sd"].values())
    snr_flip = 30.0 + 20.0 * np.log10(worst / fi.LOCAL_REGIME_MAX_REL_SD)
    assert fi.nuisance_report(F, names, snr_flip + 0.5, truth)["local_regime"] and not fi.nuisance_report(F, names, snr_flip - 0.5, truth)["local_regime"]
    assert fi.nuisance_report(F, names, snr_flip - 0.5, truth, max_rel_sd=10.0)["local_regime"]  # a looser threshold


def test_detection_limit_is_the_one_sided_wald_quantile_sum():
    from scipy.stats import norm

    assert fi.detection_limit(1.0) == pytest.approx(norm.ppf(0.95) + norm.ppf(0.9), rel=1e-14) and fi.detection_limit(1.0) == pytest.approx(2.9265, abs=1e-3)
    assert fi.detection_limit(2.5e-4, alpha=0.01, power=0.5) == pytest.approx(2.5e-4 * norm.ppf(0.99), rel=1e-14)
    # +9.3 dB over the Smith criterion sigma_CRB = g, in SNR (sigma scales as 10^(-dB / 20))
    assert 20.0 * np.log10(fi.detection_limit(1.0)) == pytest.approx(9.33, abs=0.01)
