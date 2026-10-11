"""Tests for scripts/span_test.py (docs/DESIGN.md 11.4) and the max-overlap rule of
phantoms.ch15_austria_variant.

* the projection on span(T) against a case whose answer is known by construction (T = B A, m = B c + z with z in the
  orthogonal complement), for well- and ill-conditioned and rank-deficient T, with the normal equations as a negative
  control;
* the span identity of the book's NMM (eqs. 15.1-15.6) on a toy ELM built from the equations with random features:
  every output lies in span(T) to 1e-10, whatever the random weights and thresholds, and an output bias would break it;
* properties of the error measures, the scene builder, the independent implementation and a tiny end-to-end run.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bayes_scatter.geometry import Grid
from bayes_scatter.phantoms import ch15_austria_variant, paint, paint_max

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st = _load("span_test")


# --------------------------------------------------------------------------- projection with a known answer


def _split(rng, N, P, K):
    """B (N, P) orthonormal and Bp (N, K) orthonormal with B^T Bp = 0, from one QR factorization."""
    Qf, _ = np.linalg.qr(rng.standard_normal((N, P + K)))
    return Qf[:, :P], Qf[:, P:]


def _known_case(rng, N=200, P=12, K=30, n_cols=4):
    """T with span(T) = span(B) and M = B C + Bp D, so the exact bound of each column is ||d|| / sqrt(||c||^2 + ||d||^2)."""
    B, Bp = _split(rng, N, P, K)
    A = rng.standard_normal((P, P)) + 3.0 * np.eye(P)
    C, D = rng.standard_normal((P, n_cols)), rng.standard_normal((K, n_cols))
    exact = np.linalg.norm(D, axis=0) / np.sqrt((C**2).sum(0) + (D**2).sum(0))
    return B, A, B @ A, B @ C + Bp @ D, exact


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_projection_matches_the_known_answer_for_all_three_routes(seed):
    B, A, T, M, exact = _known_case(np.random.default_rng(seed))
    Q, info = st.basis_of_span(T)
    assert info["path"] == "qr" and info["rank"] == T.shape[1]
    np.testing.assert_allclose(st.span_error(Q, M), exact, atol=1e-13, rtol=0)
    np.testing.assert_allclose(st.lstsq_error(T, M), exact, atol=1e-13, rtol=0)
    np.testing.assert_allclose(st.indep_svd_error(T, M), exact, atol=1e-13, rtol=0)


def test_projection_is_stable_for_an_ill_conditioned_training_matrix_where_normal_equations_fail():
    rng = np.random.default_rng(4)
    N, P, K = 200, 10, 40
    B, Bp = _split(rng, N, P, K)
    V, _ = np.linalg.qr(rng.standard_normal((P, P)))
    s = np.logspace(0, -8, P)  # condition number 1e8 (the normal equations square it)
    T = B @ (np.diag(s) @ V)  # span(T) = span(B) exactly
    c, d = rng.standard_normal((P, 3)), rng.standard_normal((K, 3))
    M = B @ c + Bp @ d
    exact = np.linalg.norm(d, axis=0) / np.sqrt((c**2).sum(0) + (d**2).sum(0))
    Q, _ = st.basis_of_span(T)
    assert np.abs(st.span_error(Q, M) - exact).max() < 1e-6
    assert np.abs(st.lstsq_error(T, M) - exact).max() < 1e-6
    # negative control: the normal equations lose the information (cond(T^T T) = 1e16)
    coef = np.linalg.pinv(T.T @ T) @ (T.T @ M)
    naive = np.linalg.norm(M - T @ coef, axis=0) / np.linalg.norm(M, axis=0)
    assert np.abs(naive - exact).max() > 1e-3


def test_projection_handles_rank_deficient_and_wide_training_matrices():
    rng = np.random.default_rng(5)
    B, A, T, M, exact = _known_case(rng, N=100, P=8, K=20, n_cols=3)
    T_def = np.hstack([T, T[:, :1], np.zeros((100, 1)), T[:, 1:2] - 2.0 * T[:, 2:3]])  # three redundant columns
    Q, info = st.basis_of_span(T_def)
    assert info["path"] == "svd" and info["rank"] == 8 and Q.shape == (100, 8)
    np.testing.assert_allclose(st.span_error(Q, M), exact, atol=1e-12, rtol=0)
    # more columns than pixels: the span is the whole space, the bound is zero
    W = rng.standard_normal((10, 15))
    Qw, infow = st.basis_of_span(W)
    assert infow["rank"] == 10
    assert st.span_error(Qw, rng.standard_normal((10, 4))).max() < 1e-13


def test_error_measures_properties_on_random_data():
    rng = np.random.default_rng(6)
    N, P = 150, 9
    T = 1.0 + 7.0 * rng.random((N, P))
    M = 1.0 + 2.0 * rng.random((N, 6))
    Q, _ = st.basis_of_span(T)
    e = st.span_error(Q, M)
    assert np.all(e > 0) and np.all(e < 1)
    # the residual is orthogonal to every training column
    r = M - Q @ (Q.T @ M)
    assert np.abs(T.T @ r).max() < 1e-11 * np.linalg.norm(T) * np.linalg.norm(r)
    # members of the span (and of its closure under the mean) have zero bound; scale invariance
    assert st.span_error(Q, T @ rng.standard_normal((P, 3))).max() < 1e-13
    np.testing.assert_allclose(st.span_error(Q, 5.0 * M), e, rtol=1e-12)
    # anything inside the span cannot beat the projection
    assert np.all(st.nearest_training_error(T, M) >= e - 1e-13)
    assert np.all(st.mean_image_error(T, M) >= e - 1e-13)
    # a bigger span cannot be worse
    assert np.all(st.span_plus_constant_error(Q, M) <= e + 1e-13)
    # vacuum guess: ||m - 1|| / ||m||
    np.testing.assert_allclose(st.vacuum_error(M), np.linalg.norm(M - 1, axis=0) / np.linalg.norm(M, axis=0))
    # nearest image of a training column is itself
    assert st.nearest_training_error(T, T[:, :3]).max() < 1e-7
    # contrast-trained: zero bound for a contrast that lies in the span of the training contrasts
    X = T - 1.0
    Mc = 1.0 + X @ rng.standard_normal((P, 2))
    assert st.contrast_trained_error(T, Mc).max() < 1e-12


def test_summarize_and_welch_z():
    s = st.summarize(np.arange(1.0, 101.0))  # 1 ... 100 percent
    assert s["n"] == 100 and s["median"] == 50.5 and s["min"] == 1.0 and s["max"] == 100.0
    assert s["frac_le_3p3"] == 0.03 and np.isclose(s["p05"], 5.95)
    assert st.summarize([1, 2, 3, 4, 5])["frac_le_3p3"] == 0.6
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    assert abs(st.welch_z(a, b)) < 4 and st.welch_z(a + 1.0, b) > 8


def test_replicates_shrink_with_training_set_size():
    reps = [st.replicates_for(P, 20) for P in (25, 100, 200, 400, 800, 1600, 3200)]
    assert reps == [20, 20, 10, 10, 5, 3, 3] and st.replicates_for(3200, 2) == 2


# --------------------------------------------------------------------------- the span identity on a toy ELM


def _toy_elm(rng, N=40, M=24, P=6, L=None, S=None, bias_scale=1.0):
    """The NMM of eqs. 15.1-15.6 with random features: returns T (N, P), the training inputs X (M, P) and a function
    forward(x) -> (N, K) evaluating eq. 15.1 for complex inputs x (M, K), with beta_hat = T pinv(G_c) (eq. 15.5) and
    alpha_hat = T pinv(G_r) (eq. 15.6) trained on (X, T)."""
    L = 2 * P + 1 if L is None else L  # Hecht-Nelson node count 2n + 1 of the book
    S = 2 * P + 1 if S is None else S
    T = 1.0 + 7.0 * rng.random((N, P))  # real, positive permittivity images
    X = rng.standard_normal((M, P)) + 1j * rng.standard_normal((M, P))
    wc = (rng.standard_normal((L, M)) + 1j * rng.standard_normal((L, M))) / np.sqrt(M)
    b = bias_scale * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
    wr = rng.standard_normal((S, N)) / np.sqrt(N)
    p = bias_scale * rng.standard_normal(S)
    g_c = np.arcsinh  # eq. 15.2 (complex argument)
    g_r = lambda z: 1.0 / (1.0 + np.exp(-z))  # eq. 15.3

    Gc = g_c(wc @ X + b[:, None])  # L x P complex
    beta = T @ np.linalg.pinv(Gc)  # N x L, K = N
    Gr = g_r(wr @ np.abs(beta @ Gc) + p[:, None])  # S x P
    alpha = T @ np.linalg.pinv(Gr)  # N x S

    def forward(x):
        x = np.asarray(x).reshape(M, -1)
        h = np.abs(beta @ g_c(wc @ x + b[:, None]))  # |beta g_c(w_c x + b)|, real N x K
        return alpha @ g_r(wr @ h + p[:, None])  # eq. 15.1

    return T, X, forward, {"Gc": Gc, "Gr": Gr}


def _relative_residual(T, O):
    Q, _ = np.linalg.qr(T)
    return np.linalg.norm(O - Q @ (Q.T @ O), axis=0) / np.linalg.norm(O, axis=0)


@pytest.mark.parametrize("seed", range(5))
def test_toy_elm_output_lies_in_span_of_the_training_images_to_1e_10(seed):
    rng = np.random.default_rng(seed)
    T, X, forward, G = _toy_elm(rng)
    assert np.linalg.cond(G["Gc"]) < 1e6 and np.linalg.cond(G["Gr"]) < 1e6  # full column rank: trained exactly
    # on the training inputs the ELM interpolates T (eqs. 15.4-15.6) ...
    np.testing.assert_allclose(forward(X), T, rtol=1e-7, atol=1e-7)
    # ... and on arbitrary new inputs, far from the training distribution, it stays in span(T)
    Xt = 3.0 * (rng.standard_normal((X.shape[0], 30)) + 1j * rng.standard_normal((X.shape[0], 30)))
    O = forward(Xt)
    assert O.shape == (T.shape[0], 30) and np.isrealobj(O)
    assert _relative_residual(T, O).max() < 1e-10
    # the same statement through the script's projection code
    Q, _ = st.basis_of_span(T)
    assert st.span_error(Q, O).max() < 1e-10
    # a nonzero bound for a target outside the span: the check is not vacuous
    assert st.span_error(Q, 1.0 + 0.5 * rng.random((T.shape[0], 3))).min() > 1e-2


@pytest.mark.parametrize("kwargs", [{"S": 3}, {"L": 4}, {"L": 3, "S": 2}, {"bias_scale": 10.0}])
def test_toy_elm_span_holds_for_rank_limited_layers_and_large_thresholds(kwargs):
    rng = np.random.default_rng(11)
    T, X, forward, _ = _toy_elm(rng, **kwargs)
    O = forward(2.0 * (rng.standard_normal((X.shape[0], 20)) + 1j * rng.standard_normal((X.shape[0], 20))))
    assert _relative_residual(T, O).max() < 1e-10
    fit = np.linalg.norm(forward(X) - T) / np.linalg.norm(T)
    if kwargs.get("S", 99) < T.shape[1]:
        # fewer second-layer nodes than training images: G_r cannot have rank P, T is not interpolated, the span holds
        assert fit > 1e-3
    elif "L" in kwargs:
        # a narrow first layer does not matter for the fit: alpha_hat G_r = T whenever G_r has rank P (eq. 15.6)
        assert fit < 1e-7


def test_an_output_bias_would_escape_the_span_and_the_check_sees_it():
    rng = np.random.default_rng(3)
    T, X, forward, _ = _toy_elm(rng)
    O = forward(X + 0.1)
    q = rng.standard_normal(T.shape[0])
    q *= 0.1 * np.linalg.norm(O[:, 0]) / np.linalg.norm(q)
    assert _relative_residual(T, O).max() < 1e-10
    assert _relative_residual(T, O + q[:, None]).min() > 1e-2  # eq. 15.1 has no such term


# --------------------------------------------------------------------------- max-overlap rule of the phantom generator


def test_paint_max_blends_by_coverage_and_never_lowers_a_value():
    chi = np.array([[0.0, 3.0], [3.0, 3.0]])
    cov = np.array([[1.0, 1.0], [0.5, 0.0]])
    out = paint_max(chi, cov, 1.0)
    np.testing.assert_allclose(out, [[1.0, 3.0], [3.0, 3.0]])  # where the old value is larger, it stays
    np.testing.assert_allclose(paint_max(chi, cov, 5.0), [[5.0, 5.0], [4.0, 3.0]])
    np.testing.assert_allclose(paint_max(np.zeros((2, 2)), cov, 2.0), paint(np.zeros((2, 2)), cov, 2.0))  # on background
    # order independence for full-coverage overlapping shapes
    c1, c2 = np.array([[1.0, 1.0, 0.0]]), np.array([[0.0, 1.0, 1.0]])
    a = paint_max(paint_max(np.zeros((1, 3)), c1, 4.0), c2, 2.0)
    b = paint_max(paint_max(np.zeros((1, 3)), c2, 2.0), c1, 4.0)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, [[4.0, 4.0, 2.0]])
    np.testing.assert_array_equal(paint(paint(np.zeros((1, 3)), c1, 4.0), c2, 2.0), [[4.0, 2.0, 2.0]])


def test_austria_variant_overlap_rules():
    g = Grid(24, 5.0)
    differ = 0
    for seed in range(12):
        rng_p, rng_m = np.random.default_rng(seed), np.random.default_rng(seed)
        painter = ch15_austria_variant(rng_p, g)
        explicit = ch15_austria_variant(np.random.default_rng(seed), g, overlap="painter")
        mx = ch15_austria_variant(rng_m, g, overlap="max")
        np.testing.assert_array_equal(painter, explicit)  # the default is unchanged
        assert np.all(mx >= painter - 1e-12)  # max(old, new) >= new
        assert rng_p.random() == rng_m.random()  # identical random stream (same shapes, same eps_r)
        differ += int(np.abs(mx - painter).max() > 1e-9)
    assert differ >= 6  # five large shapes in a 5 x 5 DOI overlap in most draws
    with pytest.raises(ValueError):
        ch15_austria_variant(np.random.default_rng(0), g, overlap="average")


# --------------------------------------------------------------------------- scenes and the independent implementation


def test_paint_scene_area_scaling_and_scene_definitions():
    g = Grid(96, 5.0)
    shape = [("disk", 0.3, -0.4, 0.64, 1.3)]
    eps = st.paint_scene(g, shape).reshape(96, 96)
    assert eps.min() == 1.0 and np.isclose(eps.max(), 1.3)
    area = (eps - 1.0).sum() * g.h**2 / 0.3
    assert abs(area - np.pi * 0.64**2) / (np.pi * 0.64**2) < 5e-3
    X, Y = g.centers()
    w = (eps - 1.0) / (eps - 1.0).sum()
    assert abs((w * X).sum() - 0.3) < 2e-3 and abs((w * Y).sum() + 0.4) < 2e-3  # axis 0 is x
    # lengths scale with side / 5 (a half-size DOI gives a half-size disk, a quarter of the area)
    small = st.paint_scene(Grid(96, 2.5), shape).reshape(96, 96)
    assert abs((small - 1.0).sum() * (2.5 / 96) ** 2 / 0.3 - np.pi * 0.32**2) / (np.pi * 0.32**2) < 5e-3
    rect = st.paint_scene(g, [("rect", 0.0, 0.0, 12 * g.h, 6 * g.h, 2.0)]).reshape(96, 96)  # pixel-aligned edges
    assert (rect - 1.0).sum() == pytest.approx(72.0, abs=1e-9) and rect.max() == 2.0
    ring = st.paint_scene(g, [("ring", 0.1, 0.2, 0.5, 0.9, 3.0)]).reshape(96, 96)
    assert abs((ring - 1.0).sum() * g.h**2 / 2.0 - np.pi * (0.9**2 - 0.5**2)) / (np.pi * (0.9**2 - 0.5**2)) < 5e-3
    with pytest.raises(ValueError):
        st.paint_scene(g, [("triangle", 0, 0, 0.5, 0.9, 2.0)])
    # documented scenes: eps_r <= 2 inside the figure's colorbar, three disks in Test #4, a disk and a rectangle in #1
    for key in ("test1", "test4"):
        assert len(st.SCENES[key]) == 8
        for sc in st.SCENES[key]:
            assert max(s[-1] for s in sc["shapes"]) <= 2.0 and min(s[-1] for s in sc["shapes"]) > 1.0
    assert all([s[0] for s in sc["shapes"]] == ["disk", "rect"] for sc in st.SCENES["test1"])
    assert all([s[0] for s in sc["shapes"]] == ["disk"] * 3 for sc in st.SCENES["test4"] + st.SCENES["test4_high"])
    assert max(s[-1] for s in st.SCENES["test4_high"][0]["shapes"]) > 2.0


def test_scene_overlap_rules_are_consistent_with_the_figure():
    g = Grid(96, 5.0)
    for sc in st.SCENES["test1"]:  # disk and rectangle never touch
        np.testing.assert_array_equal(st.paint_scene(g, sc["shapes"], "painter"), st.paint_scene(g, sc["shapes"], "max"))
    a = st.SCENES["test4"][0]  # digitized: eps_r increases along the painting order, so both rules agree
    np.testing.assert_allclose(st.paint_scene(g, a["shapes"], "painter"), st.paint_scene(g, a["shapes"], "max"), atol=1e-12)
    b = st.SCENES["test4"][1]  # reversed order: the painter's rule now differs in the overlaps
    pa, mx = st.paint_scene(g, b["shapes"], "painter"), st.paint_scene(g, b["shapes"], "max")
    assert np.all(mx >= pa - 1e-12) and np.abs(mx - pa).max() > 0.1


def test_jittered_scenes_are_reproducible_and_stay_in_range():
    base = st.SCENES["test1"][0]["shapes"]
    j1 = st.jittered_scenes(base, np.random.default_rng(1), 5)
    j2 = st.jittered_scenes(base, np.random.default_rng(1), 5)
    assert j1 == j2 and len(j1) == 5
    for shapes in j1:
        assert [s[0] for s in shapes] == ["disk", "rect"] and all(1.1 <= s[-1] <= 2.0 for s in shapes)
        assert abs(shapes[0][1] - base[0][1]) <= 0.25 and 0.8 * 0.64 <= shapes[0][3] <= 1.2 * 0.64
        assert len(shapes[1]) == len(base[1])


def test_independent_generator_ranges_and_rendering():
    rng = np.random.default_rng(0)
    for _ in range(300):
        d1, d2, ring, rect, sq = st.indep_shapes(rng)
        assert 0.1 <= d1[3] <= 1 and 0.1 <= d2[3] <= 1 and 0.1 <= ring[3] < ring[4] <= 2 and ring[4] >= 0.2
        assert 0.1 <= rect[3] <= 2 and 0.1 <= rect[4] <= 2 and sq[3] == sq[4] and 0.1 <= sq[3] <= 2
        assert all(-2 <= s[1] <= 2 and -2 <= s[2] <= 2 and 1 <= s[-1] <= 8 for s in (d1, d2, ring, rect, sq))
    pa, mx = st.indep_render([("disk", 0.3, -0.4, 0.64, 1.3)], 96, 5.0, 4)
    g = Grid(96, 5.0)
    ref = st.paint_scene(g, [("disk", 0.3, -0.4, 0.64, 1.3)]).reshape(96, 96)
    np.testing.assert_array_equal(pa, mx)  # one shape: both rules agree
    assert np.linalg.norm(pa - ref) / np.linalg.norm(ref) < 2e-3  # two different rasterizers of the same disk
    # overlap: painter's rule and max rule differ exactly where a lower eps_r is painted over a higher one
    pa, mx = st.indep_render([("disk", 0.0, 0.0, 1.0, 3.0), ("disk", 0.0, 0.0, 0.5, 2.0)], 48, 5.0, 4)
    assert pa.min() == 1.0 and mx.max() == 3.0 and pa[24, 24] == 2.0 and mx[24, 24] == 3.0


def test_tail_scan_matches_a_direct_computation_and_does_not_depend_on_the_chunk_size():
    g = Grid(16, 5.0)
    a = st.tail_scan(g, "painter", 10, 25, 2, seed=3, chunk=10)
    b = st.tail_scan(g, "painter", 10, 25, 2, seed=3, chunk=25)
    assert a.shape == (2, 25)
    np.testing.assert_allclose(a, b, atol=1e-10)
    M = st.random_images(np.random.default_rng([3, 4]), 25, g, "painter").T
    T = st.random_images(np.random.default_rng([3, 1, 1]), 10, g, "painter").T
    np.testing.assert_allclose(a[1], 100 * st.lstsq_error(T, M), atol=1e-9)  # second training set, scipy route
    t = st.tail_summary(a)
    assert t["n_draws"] == 25 and t["n_train_sets"] == 2 and t["n"] == 50
    assert t["min"] <= t["p00_1"] <= t["p01"] <= t["median"] and t["frac_le_5"] <= t["frac_le_10"]
    assert t["frac_le_10"] == pytest.approx(np.mean(a <= 10.0))


def test_bootstrap_median_z_detects_shifts_and_not_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(30, 6, 1000), rng.normal(30, 6, 1000)
    assert abs(st.bootstrap_median_z(a, b)) < 3.5
    assert st.bootstrap_median_z(a + 3.0, b) > 4.0  # 3.0 is about 9 standard errors of a median of 1000 draws


def test_rasterizer_bridge_runs_the_same_shapes_through_both_renderers():
    import argparse

    g = Grid(24, 5.0)
    args = argparse.Namespace(p_train=12, seed=0, indep_ss=4, indep_scene_ss=4)
    out = st.rasterizer_bridge(g, args, n_test=15)
    for rule in ("painter", "max"):
        for tag, n in (("images", 15), ("scenes", 16)):
            e = out[rule][tag]
            assert e["n"] == n and np.isfinite(e["rms_diff_pp"]) and e["max_abs_diff_pp"] >= abs(e["mean_diff_pp"])
            assert 0 < e["image_rel_diff_median"] < 0.2  # same shapes, different rasterizers: images agree to a few percent
            assert abs(e["mean_error_A_pct"] - e["mean_error_B_pct"]) < 3.0


# --------------------------------------------------------------------------- end to end (tiny)


def test_quick_run_writes_consistent_outputs(tmp_path):
    out = tmp_path / "span"
    res = st.main(["--quick", "--out", str(out)])
    for f in ("span_test.json", "span_test.npz", "span_test.png"):
        assert (out / f).exists(), f
    on_disk = json.loads((out / "span_test.json").read_text())
    z = np.load(out / "span_test.npz")
    P0 = str(on_disk["meta"]["args"]["p_train"])
    n_test, reps = on_disk["meta"]["args"]["n_test"], on_disk["meta"]["args"]["replicates"]
    for rule in ("painter", "max"):
        arr = z[f"{rule}/P{P0}/test"]  # recomputed from the raw array, not from the JSON summary
        entry = on_disk["rules"][rule]["sweep"][P0]["test"]
        assert arr.shape == (reps, n_test) and entry["pooled"]["n"] == reps * n_test
        assert entry["pooled"]["median"] == pytest.approx(np.median(arr))
        assert entry["pooled"]["min"] == pytest.approx(arr.min())
        assert entry["pooled"]["frac_le_3p3"] == pytest.approx(np.mean(arr <= 3.3))
        np.testing.assert_allclose(entry["per_set_median"], np.median(arr, axis=1))
        assert np.all((arr > 0) & (arr < 100))
        # span(T u {1}) can only help, the nearest training image can only hurt
        e = on_disk["rules"][rule]["sweep"][P0]
        assert e["test_const"]["pooled"]["median"] <= e["test"]["pooled"]["median"] + 1e-9
        assert e["test_nearest"]["pooled"]["median"] >= e["test"]["pooled"]["median"] - 1e-9
        for key in ("test1", "test4", "test4_high", "test1_jitter", "test4_jitter"):
            s = e["scenes"][f"{key}@{rule}"]
            assert s["scene"]["pooled"]["n"] == reps * len(s["names"]) and len(s["vacuum_pct"]) == len(s["names"])
    cc = on_disk["cross_check"]
    assert cc["bridge1_projection"]["max_abs_diff_qr_vs_gelsd"] < 1e-12
    assert cc["bridge1_projection"]["max_abs_diff_qr_vs_svd"] < 1e-12
    assert cc["bridge2_scenes"]["image_rel_diff_max"] < 0.05 and cc["bridge2_scenes"]["error_diff_pp_max"] < 0.5
    assert cc["bridge2_scenes"]["n_scenes"] == 32 and len(cc["z_all"]) == 14
    assert cc["bridge4_rasterizers"]["painter"]["images"]["n"] == 300 and cc["bridge4_rasterizers"]["max"]["scenes"]["n"] == 16
    assert on_disk["tail_scan"]["painter"]["n_draws"] == on_disk["meta"]["args"]["n_tail"]
    assert on_disk["binary_masks"]["max"]["test4"]["n"] == 3 * 8  # 3 training sets x 8 scenes
    assert isinstance(cc["agree"], bool) and np.isfinite(cc["z_max_abs"])
    assert on_disk["sweep"] == sorted(on_disk["sweep"]) and res["book"]["threshold_pct"] == 3.3
    assert on_disk["book"]["scenes"]["test4"][0]["shapes"][0][0] == "disk"


# --------------------------------------------------------------------------- the review's alternative readings


def _training(n=24, P=20, eps_range=(1.0, 2.0), seed=0):
    grid = Grid(n, 5.0)
    T = np.ascontiguousarray(st.random_images(np.random.default_rng(seed), P, grid, "painter", eps_range).T)
    return grid, T


def test_clip_shift_error_is_between_the_plain_clip_and_the_span_and_matches_a_direct_evaluation():
    grid, T = _training()
    M = np.array([st.paint_scene(grid, sc["shapes"], "painter").ravel() for sc in st.SCENES["test1"][:4]]).T  # eps_r in [1, 2]
    Q, _ = st.basis_of_span(T)
    norm = np.linalg.norm(M, axis=0)
    plain = np.linalg.norm(np.clip(Q @ (Q.T @ M), 1.0, 2.0) - M, axis=0) / norm
    e0, c0 = st.clip_shift_error(Q, M, c_grid=[0.0])
    np.testing.assert_allclose(e0[c0 == 0.0], plain[c0 == 0.0], rtol=1e-12)  # no shift: the plain clip
    e, c = st.clip_shift_error(Q, M)
    assert np.all(e <= e0 + 1e-12) and np.all(e <= st.span_error(Q, M) + 1e-12)  # clip onto a convex set holding m never hurts
    g = Q @ (Q.T @ np.ones((Q.shape[0], 1)))
    direct = np.linalg.norm(np.clip(Q @ (Q.T @ M) + g * c[None, :], 1.0, 2.0) - M, axis=0) / norm
    np.testing.assert_allclose(e, direct, rtol=1e-12)  # the returned shift reproduces the returned error
    # the empty scene m = 1: its in-span image has a background residual; shifting it down and clipping at 1 removes part of it
    ones = np.ones((Q.shape[0], 1))
    e_empty, c_empty = st.clip_shift_error(Q, ones)
    plain_empty = np.linalg.norm(np.clip(Q @ (Q.T @ ones), 1.0, 2.0) - ones) / np.linalg.norm(ones)
    assert e_empty[0] < plain_empty and c_empty[0] < 0.0


def test_variant_errors_order_the_readings_and_the_chi_norm_is_larger():
    grid, T = _training(P=20)
    M = np.array([st.paint_scene(grid, sc["shapes"], "painter").ravel() for sc in st.SCENES["test4"][:5]]).T
    v = st.variant_errors(T, M)
    assert set(v) == {"span", "span_const", "shift_clip", "const_clip", "chi_norm"}
    eps = 1e-12
    assert np.all(v["span_const"] <= v["span"] + eps)  # span(T u 1) contains span(T)
    assert np.all(v["const_clip"] <= v["span_const"] + eps) and np.all(v["shift_clip"] <= v["span"] + eps)  # m is in [1, 2]
    assert np.all(v["chi_norm"] > v["span"]) and np.all(v["span"] > 0) and all(a.shape == (5,) for a in v.values())
    np.testing.assert_allclose(v["span"], st.span_error(st.basis_of_span(T)[0], M), rtol=1e-12)


def test_metric_reading_checks_say_a_squared_misfit_is_worse_than_doing_nothing():
    out = st.metric_reading_checks({"test1": 7.30, "test4": 14.91})
    assert out["squared_reading_worse_than_vacuum"] and all(out["printed_reading_better_than_vacuum"].values())
    assert out["tests"]["test1_BA"]["if_squared_relative_error_pct"] == pytest.approx(100 * np.sqrt(0.048230))
    assert out["tests"]["test4_VBIM"]["book_pct"] == 4.0328 and out["tests"]["test1_VBIM"]["book_pct"] == 2.0181
    # with a vacuum guess as bad as 40 % the squared reading would no longer contradict the table: the check is not vacuous
    assert not st.metric_reading_checks({"test1": 40.0, "test4": 40.0})["squared_reading_worse_than_vacuum"]


def test_heldout_min_diagnosis_finds_the_single_draw_that_sets_the_minimum():
    rng = np.random.default_rng(0)
    e = 33.0 + 4.0 * rng.standard_normal((6, 80))
    e[:, 7] = 12.0 + rng.standard_normal(6)  # one very easy draw, easy under every training set
    d = st.heldout_min_diagnosis(e)
    assert d["easiest_draw_index"] == 7 and d["pooled_min"] == pytest.approx(e[:, 7].min())
    assert d["min_without_easiest_draw"] > d["pooled_min"] + 4.0  # the minimum moves by several points without that one draw
    assert d["median"] == pytest.approx(np.median(e)) and d["p00_1"] == pytest.approx(np.percentile(e, 0.1))
    clean = np.delete(e, 7, axis=1)
    assert abs(np.median(clean) - d["median"]) < 0.5 and abs(np.percentile(clean, 5) - d["p05"]) < 1.0  # robust statistics


def test_run_variants_tiny_is_paired_with_the_baseline_and_ordered_by_the_eps_law():
    grid = Grid(24, 5.0)
    res, raw = st.run_variants(grid, p_train=10, n_sets=2, n_test=20, seed=0, log=lambda *a: None)
    base, raw_base = st.run_main(grid, [10], 10, 2, 20, 3, 0, log=lambda *a: None)
    np.testing.assert_allclose(raw["variants/1-8/heldout_span"], raw_base["painter/P10/test"], rtol=1e-12)  # same seeds
    np.testing.assert_allclose(raw["variants/1-8/test1/span"], raw_base["painter/P10/scene"][:, :8], rtol=1e-12)
    assert list(res["eps_ranges"]) == ["1-8", "1-3", "1-2"]
    empties = [res["eps_ranges"][t]["empty_scene_pct"]["mean"] for t in res["eps_ranges"]]
    assert empties[0] > empties[1] > empties[2]  # a smaller permittivity contrast puts the empty scene closer to the span
    for tag, e in res["eps_ranges"].items():
        for key in ("test1", "test4"):
            s = e["scenes"][key]
            assert s["scene_names"][0].endswith("digitized") and s["span"]["pooled_over_8_scenes"]["n"] == 2 * 8
            assert s["span"]["digitized"]["n"] == 2 and s["chi_norm"]["pooled_over_8_scenes"]["median"] > s["span"]["pooled_over_8_scenes"]["median"]
            assert s["span_const"]["pooled_over_8_scenes"]["median"] <= s["span"]["pooled_over_8_scenes"]["median"] + 1e-9
        assert e["heldout_min_diagnosis"]["n_sets"] == 2 and e["heldout_min_diagnosis"]["n_draws"] == 20
    assert set(res["best_reading"]) == {"test1", "test4"} and res["metric_reading"]["squared_reading_worse_than_vacuum"]
    gaps = res["flatness"]["digitized_median_test4_minus_test1_pp"]
    assert set(gaps) == set(res["eps_ranges"]) and set(gaps["1-8"]) == set(st.VARIANT_NAMES)
    assert res["flatness"]["book_nmm_test4_minus_test1_pp"] == pytest.approx(3.2812 - 3.2615)


def test_variants_only_adds_to_stored_results_without_changing_them(tmp_path):
    out = tmp_path / "span"
    full = st.main(["--quick", "--out", str(out)])
    assert "variants" in full and "heldout_min_diagnosis" in full and full["book"]["table_15_3_ba_pct"][0] == 4.8230
    stored = json.loads((out / "span_test.json").read_text())
    z = {k: v for k, v in np.load(out / "span_test.npz").items() if not k.startswith("variants/")}
    variants_json = stored.pop("variants")
    stored.pop("heldout_min_diagnosis")
    (out / "span_test.json").write_text(json.dumps(stored))
    np.savez_compressed(out / "span_test.npz", **z)
    again = st.main(["--variants-only", "--variant-sets", "3", "--out", str(out)])
    assert again["variants"]["eps_ranges"]["1-2"]["scenes"]["test4"]["span"]["digitized"]["median"] == pytest.approx(
        variants_json["eps_ranges"]["1-2"]["scenes"]["test4"]["span"]["digitized"]["median"])
    after = json.loads((out / "span_test.json").read_text())
    for key in ("rules", "tail_scan", "cross_check", "binary_masks", "sweep"):
        assert after[key] == stored[key]  # the stored numbers are untouched
    z2 = np.load(out / "span_test.npz")
    assert "variants/1-2/test4/shift_clip" in z2.files and np.array_equal(z2["painter/P20/test"], z["painter/P20/test"])


def test_report_tables_print_from_the_stored_json(tmp_path, capsys):
    tables = _load("span_test_tables")
    out = tmp_path / "span"
    st.main(["--quick", "--out", str(out)])
    capsys.readouterr()
    tables.main(["--dir", str(out)])
    text = capsys.readouterr().out
    for head in ("### (a) held-out Table 15.2 draws", "### bound versus P", "### (b) Fig. 15.9-like scenes",
                 "### other readings of the NMM", "### independent implementation B", "Squared-misfit reading", "T1-a digitized"):
        assert head in text
    assert "| 100 |" not in text and "| 20 |" in text  # the quick configuration: P = 20, not the book's 100
