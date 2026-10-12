"""Tests for the milestone-5 review follow-up (docs/DESIGN.md 12.6): the resolution reference, disk-averaged features, the corrected
independence checks, the continuation detection, the overdispersed starts and the review batch.

Each test pins a defect or a missing safeguard that the review of the milestone-5 report found:
* the Gaussian marginals of the ring-continuity check used the interpolation of the pixel sd, not the sd of the interpolated draws;
* ``cross_run_check`` presented the continuation of a run (same chains, new map) as independent chains;
* the chains of the TV runs all started near one state (``make_overdispersed_starts`` makes starts that differ);
* the frozen arm of the adaptation negative control had a standard error as large as the bias it was meant to exclude (tests/test_adapt.py);
* single-point features of a rough posterior understate what the data determine (``features`` with a ``radius``).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

import bayes_scatter  # noqa: F401
from bayes_scatter import features as ft
from bayes_scatter import mrf
from bayes_scatter import resolution as rs
from bayes_scatter.geometry import Grid
from bayes_scatter.phantoms import austria_fixed

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import edge_analysis as ea  # noqa: E402
import make_overdispersed_starts as mos  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------
# resolution: the low-passed reference
# ---------------------------------------------------------------------------------------------------------------


def _modes(n=32, side=2.0):
    h = side / n
    x = -side / 2 + h * (np.arange(n) + 0.5)
    return h, x[:, None] * np.ones((1, n)), np.ones((n, 1)) * x[None, :]


def test_lowpass_keeps_modes_below_the_cutoff_and_removes_those_above():
    h, X, Y = _modes()
    # the periodic lattice of a 2-wavelength DOI has spacing 0.5 cycles per wavelength: 1 and 4 cycles per wavelength are on it
    low, high = np.cos(2 * np.pi * 1.0 * X), np.cos(2 * np.pi * 4.0 * Y)
    out = rs.lowpass(low + high, h, 2.0, pad=1)
    np.testing.assert_allclose(out, low, atol=1e-10)
    np.testing.assert_allclose(rs.lowpass(low + high, h, 5.0, pad=1), low + high, atol=1e-10)
    assert np.abs(rs.lowpass(high, h, 2.0, pad=1)).max() < 1e-10
    # the cutoff is on the radius |k|, not on each axis: a diagonal mode with kx = ky = 1.5 has |k| = 2.12
    diag = np.cos(2 * np.pi * 1.5 * (X + Y))
    assert np.abs(rs.lowpass(diag, h, 2.0, pad=1)).max() < 1e-10
    np.testing.assert_allclose(rs.lowpass(diag, h, 2.2, pad=1), diag, atol=1e-10)


def test_lowpass_is_a_projection_for_pad_1_and_a_contraction_for_any_pad():
    h, _, _ = _modes()
    f = np.random.default_rng(0).standard_normal((32, 32))
    once = rs.lowpass(f, h, 3.0, pad=1)
    np.testing.assert_allclose(rs.lowpass(once, h, 3.0, pad=1), once, atol=1e-10)  # idempotent
    assert np.linalg.norm(once) <= np.linalg.norm(f)
    assert np.linalg.norm(rs.lowpass(f, h, 2.0, pad=1) - f) >= np.linalg.norm(rs.lowpass(f, h, 3.0, pad=1) - f)  # nested bands
    assert rs.lowpass(f, h, 3.0, pad=4).shape == (32, 32) and rs.lowpass(f, h, 3.0).dtype == np.float64
    np.testing.assert_allclose(rs.lowpass(np.zeros((32, 32)), h, 3.0), 0.0, atol=1e-14)
    np.testing.assert_allclose(rs.lowpass(f, h, 1e3, pad=1), f, atol=1e-12)  # everything kept
    with pytest.raises(ValueError):
        rs.lowpass(np.zeros((4, 5)), h, 3.0)
    with pytest.raises(ValueError):
        rs.lowpass(f, h, 3.0, pad=0)


def test_lowpass_of_the_benchmark_object_blurs_its_edges_outside_in_and_overshoots_the_disk_centers():
    """The reference of the resolution argument: bias +0.18 just outside the object, -0.13 two pixels inside, +0.18 at the disk centers
    (3 cycles per wavelength; the values the review found and the report quotes, results/prior_comparison/m5/review/resolution_analysis.json)."""
    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    truth = np.log1p(chi)
    dist = ea.signed_distance(chi)
    b3 = rs.lowpass(truth, grid.h, 3.0) - truth
    assert 0.17 < b3[dist == -1].mean() < 0.19 and -0.14 < b3[dist == 2].mean() < -0.12 and 0.16 < b3[dist == 4].mean() < 0.19
    b25 = rs.lowpass(truth, grid.h, 2.5) - truth
    assert 0.22 < b25[dist == -1].mean() < 0.245 and -0.21 < b25[dist == 2].mean() < -0.19
    # the pad matters at the 0.003 level once the frequency lattice is fine (the discrete set of |k| <= 3 changes a little with it)
    np.testing.assert_allclose(rs.lowpass(truth, grid.h, 3.0, pad=4), rs.lowpass(truth, grid.h, 3.0, pad=6), atol=5e-3)
    # a wider band blurs less: the rms error shrinks with the cutoff
    rms = [np.sqrt(((rs.lowpass(truth, grid.h, k) - truth) ** 2).mean()) for k in (2.0, 3.0, 4.0)]
    assert rms[0] > rms[1] > rms[2]


def test_disk_average_and_pearson():
    h = 0.05
    f = np.zeros((40, 40))
    f[18:22, 18:22] = 1.0
    out = rs.disk_average(f, h, 0.1)
    assert out.sum() == pytest.approx(f.sum()) and out.max() < 1.0 and out[20, 20] > 0.3  # mass kept away from the border, blurred
    np.testing.assert_allclose(rs.disk_average(f, h, 0.01), f)  # a radius below the pixel size leaves the field alone
    a = np.random.default_rng(1).standard_normal((6, 6))
    assert rs.pearson(a, 2 * a + 3) == pytest.approx(1.0) and rs.pearson(a, -a) == pytest.approx(-1.0)
    m = np.zeros((6, 6), bool)
    m[:3] = True
    b = a.copy()
    b[3:] = -a[3:]
    assert rs.pearson(a, b, m) == pytest.approx(1.0) and rs.pearson(a, b) < 0.9
    assert np.isnan(rs.pearson(a, np.ones((6, 6))))
    assert rs.bin_by_distance(a, np.arange(36).reshape(6, 6) % 3, [0, 2, 7]).keys() == {"0", "2"}


def test_resolution_analysis_compares_a_posterior_with_the_references(tmp_path):
    import resolution_analysis as ra

    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    truth = np.log1p(chi)
    dist = ea.signed_distance(chi)
    lp = rs.lowpass(truth, grid.h, 3.0)
    rng = np.random.default_rng(0)
    # a "posterior" centered on the low-passed truth with sd 0.05: miscovers the truth at the edge, not the low-pass
    theta = lp + 0.05 * rng.standard_normal((2, 60, 48, 48))
    d = tmp_path / "run"
    d.mkdir()
    np.savez(d / "chains.npz", theta=theta.astype(np.float32), theta_true=truth, chi_true=chi)
    (d / "summary.json").write_text(json.dumps({"args": {"n": 48, "side": 2.0}, "convergence": {"converged": True}}))
    np.savez(d / "prepared.npz", theta_hat=lp)
    res = ra.analyze("run", tmp_path, [3.0], [0.08])
    ref = res["references"]["lowpass_3"]
    assert ref["posterior_mean_vs_reference"]["corr_bias_maps_all"] > 0.99
    assert ref["posterior_mean_vs_reference"]["frac_abs_z_gt_3_vs_truth"] > 0.05 > 3 * ref["posterior_mean_vs_reference"]["frac_abs_z_gt_3_vs_reference"]
    assert ref["reference"]["d-1"] == pytest.approx((lp - truth)[dist == -1].mean())
    assert res["MAP"]["d-1"] == pytest.approx(ref["reference"]["d-1"]) and "disk_0.08" in res["references"]
    out = tmp_path / "r.json"
    ra.main(["run", "--dir", str(tmp_path), "--out", str(out), "--cutoffs", "3", "--radii", "0.1"])
    assert set(json.loads(out.read_text())["run"]["references"]) == {"lowpass_3", "disk_0.1"}


# ---------------------------------------------------------------------------------------------------------------
# features at the resolution scale
# ---------------------------------------------------------------------------------------------------------------


def test_disk_mean_is_the_mean_over_the_pixel_centers_in_the_disk_and_needs_a_pixel():
    grid = Grid(48, 2.0)
    th = np.random.default_rng(0).standard_normal((3, 48, 48))
    X, Y = grid.centers()
    m = np.hypot(X - 0.1, Y + 0.3) <= 0.12
    np.testing.assert_allclose(ft.disk_mean(th, grid, 0.1, -0.3, 0.12), th[:, m].mean(axis=-1))
    assert 20 <= m.sum() <= 30
    np.testing.assert_allclose(ft.point_value(th, grid, 0.1, -0.3, 0.0), ft.bilinear(th, grid, 0.1, -0.3))
    np.testing.assert_allclose(ft.point_value(th, grid, 0.1, -0.3, 0.12), ft.disk_mean(th, grid, 0.1, -0.3, 0.12))
    with pytest.raises(ValueError, match="no pixel center"):
        ft.disk_mean(th, grid, 0.1, -0.3, 0.001)


def test_pixel_scale_roughness_hides_the_features_at_points_but_not_at_the_resolution_scale():
    """A posterior centered on the truth with iid pixel noise of sd 0.35 (the GMRF's): single-point features are uncertain, the
    disk-averaged ones (about 12 pixels, sd 0.35 / sqrt 12 = 0.10) are determined. The same draws, the same truth."""
    grid = Grid(48, 2.0)
    truth = np.log1p(austria_fixed(grid, 2.0))
    rng = np.random.default_rng(0)
    th = truth + 0.35 * rng.standard_normal((2, 400, 48, 48))
    point = ft.feature_probabilities(th, grid)
    disk = ft.feature_probabilities(th, grid, radius=0.12)
    assert point["ring_continuity"]["p"] < 0.8 and point["gap_valley"]["p"] < 0.95
    assert disk["gap_valley"]["p"] > 0.99 and disk["ring_hole"]["p"] > 0.99 and disk["ring_continuity"]["p"] > 0.9
    assert ft.feature_probabilities(th, grid, radius=0.0)["gap_valley"]["p"] == point["gap_valley"]["p"]  # radius 0 is the DESIGN definition
    # the indicators are the explicit comparison of the disk means
    pts = ft.austria_points(grid)
    ind = ft.feature_indicators(th[:, :5], grid, radius=0.12)
    gap = ft.disk_mean(th[:, :5], grid, *pts.gap_mid, 0.12)
    dm = np.minimum(ft.disk_mean(th[:, :5], grid, *pts.disk_centers[0], 0.12), ft.disk_mean(th[:, :5], grid, *pts.disk_centers[1], 0.12))
    np.testing.assert_array_equal(ind["gap_valley"], gap < dm - ft.DELTA)


def test_feature_scales_blocks_chain_ranges_and_the_laplace_draws(tmp_path):
    import feature_scales as fs

    grid = Grid(48, 2.0)
    truth = np.log1p(austria_fixed(grid, 2.0))
    th = truth + 0.3 * np.random.default_rng(1).standard_normal((2, 40, 48, 48))
    blocks = fs.block_probabilities(th, grid, 0.0, 10)
    pooled = ft.feature_probabilities(th, grid)
    for k in fs.FEATURES:
        assert len(blocks[k]) == 4 and np.mean(blocks[k]) == pytest.approx(pooled[k]["p"])  # equal blocks: the mean of the blocks is the pooled value
    assert fs.block_probabilities(th[:, :35], grid, 0.0, 10)["gap_valley"].__len__() == 3  # the short trailing block is dropped
    assert fs.spread([0.2, 0.5, 0.4]) == {"min": 0.2, "max": 0.5}
    d = tmp_path / "r"
    d.mkdir()
    np.savez(d / "chains.npz", theta=th.astype(np.float32))
    (d / "summary.json").write_text(json.dumps({"args": {"n": 48, "side": 2.0}, "convergence": {"converged": False}}))
    out = fs.analyze("r", tmp_path, [0.0, 0.08], 20)
    e = out["radii"]["0.08"]
    assert 8 <= e["pixels_in_disk"] <= 14 and e["probabilities"]["gap_valley"]["block_range"]["max"] >= e["probabilities"]["gap_valley"]["p"] - 1e-12
    assert out["radii"]["0"]["pixels_in_disk"] == 1 and len(e["probabilities"]["ring_hole"]["p_chain"]) == 2


# ---------------------------------------------------------------------------------------------------------------
# the independent feature check: the Gaussian marginal of an interpolated value
# ---------------------------------------------------------------------------------------------------------------


def test_independent_check_uses_the_sd_of_the_interpolated_draws_not_the_interpolated_pixel_sd():
    """Draws with iid pixels (sd 0.4, mean 0.6, delta 0.347): the bilinear value at a point between pixels averages 2 or 4 pixels, so its sd is
    0.4 sqrt(sum w^2) < 0.4. The interpolation of the pixel sds (0.4) overstates it; the earlier script's Gaussian product was 0.35 where the sampled
    probability was 0.50 for the adapted GMRF (review of milestone 5)."""
    import check_features_independent as cfi

    n, side = 48, 2.0
    sd, mu = 0.4, 0.6
    t = mu + sd * np.random.default_rng(0).standard_normal((20000, n, n))
    out = cfi.recompute(t, n, side)
    delta = np.log(2.0) / 2.0
    # weights of the four compass points, from the same geometry as the check (pixel centers -side/2 + (i + 1/2) h)
    h = side / n
    s = side / 5.0
    r_mid = 0.5 * (0.9 + 1.3) * s
    pts = [(r_mid, -0.8 * s), (0.0, -0.8 * s + r_mid), (-r_mid, -0.8 * s), (0.0, -0.8 * s - r_mid)]
    for (x, y), got in zip(pts, out["compass_marginals_gaussian"]):
        fx, fy = (x + side / 2) / h - 0.5, (y + side / 2) / h - 0.5
        ax, ay = fx - np.floor(fx), fy - np.floor(fy)
        w2 = (((1 - ax) * (1 - ay)) ** 2 + (ax * (1 - ay)) ** 2 + ((1 - ax) * ay) ** 2 + (ax * ay) ** 2)
        expected = 1.0 - norm.cdf((delta - mu) / (sd * np.sqrt(w2)))
        wrong = 1.0 - norm.cdf((delta - mu) / sd)
        assert got == pytest.approx(expected, abs=0.02)
        assert abs(got - wrong) > 0.03  # the quantity the old script computed is measurably different (w2 < 1 at every compass point)
    # independent pixels: the compass values are uncorrelated, and the joint probability is the product of the marginals
    assert out["compass_max_abs_correlation"] < 0.05
    assert out["ring_continuity"] == pytest.approx(out["ring_continuity_empirical_marginals_product"], abs=0.02)
    assert out["ring_continuity"] == pytest.approx(out["ring_continuity_gaussian_independent_points"], abs=0.03)


# ---------------------------------------------------------------------------------------------------------------
# overdispersed starts and the review batch
# ---------------------------------------------------------------------------------------------------------------


def test_tv_energy_numpy_equals_the_jax_prior():
    th = 0.3 * np.random.default_rng(0).standard_normal((9, 9))
    prior = mrf.PseudoHuberTV(9, lam=2.5, delta=0.02, sigma_0=0.7, mu=0.1)
    R, tv = mos.tv_energy(th, 2.5, 0.02, 0.7, 0.1)
    assert R == pytest.approx(float(prior.energy(th)), rel=1e-12)
    assert tv == pytest.approx(R - np.sum((th - 0.1) ** 2) / (2 * 0.7**2), rel=1e-12)


def _fake_base(tmp_path, n=6, lam=3.75):
    base = tmp_path / "base"
    rng = np.random.default_rng(0)
    truth = np.zeros((n, n))
    truth[2:4, 2:4] = np.log(2.0)
    (base / "pc_matern").mkdir(parents=True)
    np.savez(base / "pc_matern" / "chains.npz", theta_true=truth, theta=(truth + 0.1 * rng.standard_normal((2, 5, n, n))).astype(np.float32))
    (base / "pc_gmrf_adapt").mkdir()
    np.savez(base / "pc_gmrf_adapt" / "chains.npz", theta=(0.4 * rng.standard_normal((2, 5, n, n))).astype(np.float32))
    for run in ("pc_src", "pc_new"):
        d = base / run
        d.mkdir()
        (d / "prepared.json").write_text(json.dumps({"args": {"lam": lam, "delta": 0.01, "sigma_0": 1.0, "mu": 0.0}}))
        np.savez(d / "prepared.npz", theta_hat=0.9 * truth)
    np.savez(base / "pc_src" / "adapted_map.npz", center=np.zeros(n * n))
    (base / "pc_src" / "adaptation.json").write_text("{}")
    return base, truth


def test_overdispersed_starts_differ_from_each_other_and_are_described(tmp_path):
    base, truth = _fake_base(tmp_path)
    arr, info = mos.build(base / "pc_new", mos.KINDS, base=base)
    assert arr.shape == (4, 6, 6) and arr.dtype == np.float64
    np.testing.assert_array_equal(arr[0], truth)
    np.testing.assert_allclose(arr[1], 0.9 * truth)
    assert [i["kind"] for i in info] == list(mos.KINDS)
    pair_dist = [np.abs(arr[i] - arr[j]).max() for i in range(4) for j in range(i)]
    assert min(pair_dist) > 0.05  # four different starting maps
    assert info[3]["tv_prior_energy"] > info[0]["tv_prior_energy"]  # the rough start costs more prior energy than the truth
    np.testing.assert_array_equal(np.load(base / "pc_new" / "starts.npy"), arr)
    meta = json.loads((base / "pc_new" / "starts.json").read_text())
    assert meta["lam"] == 3.75 and len(meta["starts"]) == 4 and meta["starts"][2]["source"].startswith("mean of the draws")
    with pytest.raises(ValueError, match="unknown start kind"):
        mos.load_start("bogus", base / "pc_new", base)


def test_overdispersed_starts_copy_the_frozen_map_of_the_source_run(tmp_path):
    base, truth = _fake_base(tmp_path)
    arr, _ = mos.build(base / "pc_rep", ("truth", "gmrf"), base=base, copy_from="pc_src")
    assert arr.shape == (2, 6, 6)
    assert {p.name for p in (base / "pc_rep").iterdir()} >= {"prepared.json", "prepared.npz", "adapted_map.npz", "adaptation.json", "starts.npy"}
    assert (base / "pc_rep" / "adapted_map.npz").read_bytes() == (base / "pc_src" / "adapted_map.npz").read_bytes()


def test_review_batch_keeps_the_thread_budget_in_every_phase_and_never_touches_older_runs():
    import make_m5_review_jobs as mj
    from batch_run import parse_jobs

    jobs = mj.build()
    parse_jobs(jobs)  # valid batch_run job file, unique names
    by = {j["name"]: j for j in jobs}
    assert len(jobs) == 17
    threads = lambda names: sum(by[n]["threads"] for n in names)
    prep = [f"m5v_prep_lam{lam:g}" for lam in mj.LAMBDAS]
    adapt = [f"m5v_adapt_lam{lam:g}" for lam in mj.LAMBDAS]
    chains = [f"m5v_lam{lam:g}_c{c}" for lam in mj.LAMBDAS for c in range(4)]
    indep = ["m5v_indep_c0", "m5v_indep_c1"]
    # waiting jobs use no CPU: the computing jobs of each phase are the ones listed; the project cap is 10 XLA threads
    assert threads(prep + indep) == 10 and threads(adapt + indep) == 10 and threads(chains + indep) == 10
    for n in adapt + chains + indep:
        assert "scripts/wait_then.py" in by[n]["cmd"][0]  # gated on its inputs
    for lam in mj.LAMBDAS:
        run = f"pc_tv_lam{lam:g}_adapt"
        cmd = " ".join(by[f"m5v_lam{lam:g}_c0"]["cmd"])
        assert f"{run}/adaptation.json" in cmd and f"{run}/starts.npy" in cmd and f"--out {run}" in cmd and "--start-theta" in cmd
        assert "--adapt-kappa-per-dim 0.25" in " ".join(by[f"m5v_adapt_lam{lam:g}"]["cmd"]) and "--stage adapt" in " ".join(by[f"m5v_adapt_lam{lam:g}"]["cmd"])
    protected = {"pc_matern", "pc_gmrf", "pc_gmrf_adapt", "pc_gmrf_adapt2", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1", "pc_tv_lam3.75_adapt",
                 "pc_tv_lam3.75_adapt2", "pc_tv_lam11.25_adapt", "pc_tv_lam11.25_adapt2"}
    for j in jobs:
        outs = {a for prev, a in zip(j["cmd"], j["cmd"][1:]) if prev == "--out"}
        assert not (outs & protected), j["name"]
        assert j["done_marker"].split("/")[-2] not in protected
    ind = " ".join(by["m5v_indep_c0"]["cmd"])
    assert "--chains 2" in ind and "pc_tv_lam3.75_indep" in ind and "pc_tv_lam3.75_adapt/" not in ind  # a NEW directory, 2 chains
    assert mj.problem(2)[mj.problem(2).index("--chains") + 1] == "2" and mj.problem()[mj.problem().index("--chains") + 1] == "4"


# ---------------------------------------------------------------------------------------------------------------
# convergence audit windows and regions, edge analysis additions, lambda band
# ---------------------------------------------------------------------------------------------------------------


def test_flagged_pixels_are_counted_by_region_and_add_up():
    import convergence_audit as ca

    dist = np.array([[3, 2, 1, -1], [-2, -3, -4, -9]])
    flagged = np.array([[True, False, True, True], [True, False, True, True]])
    r = ca.flagged_by_region(flagged, dist)
    assert r == {"inside_object_d_ge_2": 1, "within_1px_of_edge": 2, "outside_d_-3_to_-2": 1, "outside_d_le_-4": 2, "outside_d_min": -9,
                 "outside_d_max": -1}
    assert sum(v for k, v in r.items() if not k.startswith("outside_d_m")) == flagged.sum()
    assert ca.flagged_by_region(np.zeros((2, 4), bool), dist)["outside_d_min"] is None


def test_edge_analysis_reports_log_mean_eps_bias_and_per_draw_blur(tmp_path):
    """A synthetic posterior with sd 0.4 around the truth: log E[eps] - E[theta] = sd^2 / 2 (Jensen), the review's reason why the E[theta] bias of
    a wide posterior is not comparable with that of a narrow one; and the per-draw ring mean has its own spread."""
    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    truth = np.log1p(chi)
    rng = np.random.default_rng(0)
    theta = truth + 0.4 * rng.standard_normal((2, 300, 48, 48))
    d = tmp_path / "r"
    d.mkdir()
    eps = np.exp(theta)
    wm = np.stack([theta.mean(axis=1), eps.mean(axis=1)], axis=1)  # (chains, 2, n, n)
    wv = np.stack([theta.var(axis=1, ddof=1), eps.var(axis=1, ddof=1)], axis=1)
    np.savez(d / "chains.npz", theta=theta.astype(np.float32), welford_mean=wm, welford_var=wv, theta_true=truth, chi_true=chi,
             theta_hat=truth, laplace_theta_std=np.full(truth.shape, 0.4))
    z = ft.z_statistics(theta.mean(axis=(0, 1)), theta.std(axis=(0, 1)), truth, chi)
    (d / "summary.json").write_text(json.dumps({"convergence": {"converged": False}, "z": {k: z[k] for k in
                                                ("frac_abs_z_gt_3", "frac_inside", "frac_outside", "edge_share", "max_abs_z")}}))
    out, _ = ea.analyze("r", tmp_path)
    lg = out["log_mean_eps_bias"]
    assert lg["bias_d-1"] - lg["theta_bias_d-1"] == pytest.approx(0.5 * 0.4**2, abs=0.01)  # log E e^theta - E theta = sd^2 / 2 for a Gaussian
    assert lg["mean_log_eps_minus_theta_mean_far_background"] == pytest.approx(lg["half_variance_far_background"], abs=0.01)
    assert set(lg["by_distance"]) >= {"-1", "2"} and lg["bias_d-1"] >= lg["theta_bias_d-1"]
    b = out["per_draw_edge_blur"]
    assert b["pixels_at_d-1"] > 100 and b["mean_over_pixels_of_draw_bias_d-1"] == pytest.approx(lg["theta_bias_d-1"], abs=1e-9)
    assert b["sd_across_draws"] == pytest.approx(0.4 / np.sqrt(b["pixels_at_d-1"]), rel=0.3)  # independent pixels: the ring mean averages them
    assert b["pixel_value_quantiles_d-1"]["0.5"] == pytest.approx(truth[ea.signed_distance(chi) == -1].mean(), abs=0.02)


def test_lambda_band_crossings_and_sweep_rows():
    import lambda_band as lb

    lams, phis = [0.3, 1.0, 3.0, 10.0], [2372.0, 2379.0, 2394.0, 2447.0]
    assert lb.crossing_lambda(lams, phis, 2400.0) == pytest.approx(np.exp(np.interp(np.log(2400.0), np.log(phis), np.log(lams))))
    assert 3.0 < lb.crossing_lambda(lams, phis, 2400.0) < 10.0
    assert lb.crossing_lambda(lams, phis, 2300.0) is None and lb.crossing_lambda(lams, phis, 2500.0) is None
    assert lb.crossing_lambda(lams, [2372.0, 2379.0, 2379.0, 2447.0], 2400.0) is None  # not monotone
    theta = np.zeros((5, 5))
    theta[1:4, 1:4] = np.log(2.0)
    tv_truth = mos.tv_energy(theta, 1.0, 0.01, 1e9, 0.0)[1]
    sweep = {"problem": {"n_data": 2400, "phi_clean": 2438.2, "phi_truth": 2578.1, "side": 2.0, "n": 5},
             "prior": {"delta": 0.01, "sigma_0": 1.0, "mu": 0.0},
             "runs": {"2.0": {"lam": 2.0, "phi": 2400.0, "R": 50.0, "tv": 0.9 * tv_truth, "tv_truth": tv_truth, "edge_bias": {"-1": 0.1}, "err_model": 0.1},
                      "10.0": {"lam": 10.0, "phi": 2460.0, "R": 90.0, "tv": 0.8 * tv_truth, "tv_truth": tv_truth, "edge_bias": {"-1": 0.12}, "err_model": 0.12}}}
    rows, ref = lb.sweep_rows(sweep, theta)
    assert [r["lam"] for r in rows] == [2.0, 10.0] and rows[0]["in_band_N_pm_sqrtN"] and not rows[1]["in_band_N_pm_sqrtN"]
    assert ref["band"] == pytest.approx([2400 - 2400**0.5, 2400 + 2400**0.5])
    assert rows[0]["tv_contrast_kept"] == pytest.approx(0.9) and rows[0]["R_gap_from_tv_seminorm"] == pytest.approx(2.0 * 0.1 * tv_truth)
    assert rows[0]["phi_map_minus_phi_clean"] == pytest.approx(2400.0 - 2438.2)


# ---------------------------------------------------------------------------------------------------------------
# start drift and equal-length audits (numpy only)
# ---------------------------------------------------------------------------------------------------------------


def test_start_drift_audit_recovers_zeta_and_the_prior_energy_per_block(tmp_path):
    import start_drift_audit as sda

    n, chains, draws = 4, 2, 300
    N = n * n
    rng = np.random.default_rng(0)
    L = np.tril(rng.standard_normal((N, N))) * 0.1 + 2.0 * np.eye(N)
    center = rng.standard_normal(N)
    # chain 0 stationary (zeta ~ N(0, I)), chain 1 drifting outward (zeta scale grows from 1 to 3 over the draws)
    scale = np.stack([np.ones(draws), np.linspace(1.0, 3.0, draws)])
    zeta = rng.standard_normal((chains, draws, N)) * scale[:, :, None]
    theta = center + zeta @ L.T
    R_true = 50.0 + np.arange(draws)[None, :] * 0.1 + np.zeros((chains, 1))  # a prior energy that rises by 30 over the run
    phi = 2400.0 + rng.standard_normal((chains, draws))
    phi_prime = phi + R_true - 0.5 * (zeta**2).sum(axis=-1)  # U = Phi + R; Phi' = U - |zeta|^2 / 2
    d = tmp_path / "run"
    d.mkdir()
    chi = np.zeros((n, n))
    chi[1:3, 1:3] = 1.0  # a 2 x 2 object: the 12 pixels around it are at d = -1 or -2
    truth = np.log1p(chi)
    np.savez(d / "chains.npz", theta=theta.reshape(chains, draws, n, n).astype(np.float64), phi_physical=phi, phi_prime=phi_prime,
             theta_true=truth, chi_true=chi)
    start_a, start_b = truth.copy(), truth + 0.5  # chain 0 starts at the truth (ring bias 0), chain 1 0.5 above it
    np.save(d / "starts.npy", np.stack([start_a, start_b]))
    np.savez(d / "adapted_map.npz", center=center, chol=L)
    for c in range(chains):
        np.savez(d / f"chain_{c}_seg0.npz", starts=np.array("overdispersed draws (scale 0.5) around the final state of the adaptation chain"))
    out = sda.audit_run("run", tmp_path, block=100)
    assert out["half_N"] == N / 2 and out["draws_per_chain"] == draws
    c0, c1 = out["chains"]
    assert c0["half_sq_zeta_blocks"] == pytest.approx([N / 2] * 3, rel=0.15)  # stationary: N / 2 in every block
    assert c1["half_sq_zeta_first_to_last"][1] > 3.0 * c1["half_sq_zeta_first_to_last"][0]  # drifting: grows by (3 / 1.33)^2 or more
    assert c0["R_blocks"] == pytest.approx([R_true[0, :100].mean(), R_true[0, 100:200].mean(), R_true[0, 200:].mean()], abs=1e-9)
    assert out["R_blocks_mean_over_chains"][2] - out["R_blocks_mean_over_chains"][0] == pytest.approx(20.0, abs=1e-9)
    ring = out["ring"]
    dist = ea.signed_distance(chi)
    assert ring["chains"][0]["start_ring_bias"] == pytest.approx(0.0) and ring["chains"][1]["start_ring_bias"] == pytest.approx(0.5)
    expected = (theta.reshape(chains, draws, n, n)[:, :, dist == -1] - truth[dist == -1]).mean(axis=-1)  # (chains, draws)
    assert ring["chains"][0]["first_draw_ring_bias"] == pytest.approx(expected[0, 0])
    assert ring["ring_bias_after_burn_mean_per_chain"] == pytest.approx(list(expected[:, 50:].mean(axis=1)))
    assert ring["chains"][1]["ring_bias_blocks"] == pytest.approx([expected[1, :100].mean(), expected[1, 100:200].mean(), expected[1, 200:].mean()])
    assert np.isfinite(ring["split_rhat_ring_bias_after_burn"]) and np.isfinite(ring["split_rhat_phi_after_burn"])
    np.testing.assert_allclose(sda.half_sq_norm(theta[0, :5].reshape(5, n, n), center, L), 0.5 * (zeta[0, :5] ** 2).sum(axis=-1))
    assert sda.block_means(np.arange(10.0), 4) == [1.5, 5.5]


def test_leapfrog_cost_of_a_window_is_weighted_over_the_segments_it_spans(tmp_path):
    import equal_length_audit as ela

    d = tmp_path / "run"
    d.mkdir()
    for c in range(2):
        np.savez(d / f"chain_{c}_seg0.npz", n_samples=100, mean_leapfrog=40.0)
        np.savez(d / f"chain_{c}_seg1.npz", n_samples=100, mean_leapfrog=20.0 + 10.0 * c)
    assert ela.mean_leapfrog_in_window(d, 0, 100) == pytest.approx(40.0)
    # 50 draws of segment 0 (40) and 50 of segment 1 (20 and 30): chain means 30 and 35
    assert ela.mean_leapfrog_in_window(d, 50, 100) == pytest.approx(32.5)
    with pytest.raises(SystemExit, match="hold"):
        ela.mean_leapfrog_in_window(d, 150, 100)


def test_lambda_band_figure_is_written_with_and_without_posteriors(tmp_path):
    import lambda_band as lb

    sweep = [{"lam": l, "phi_map": 2370.0 + 8 * l, "map_bias_d-1": 0.1} for l in (0.3, 1.0, 3.0, 10.0)]
    ref = {"band": [2351.0, 2449.0], "n_data": 2400, "phi_clean": 2438.2, "phi_truth": 2578.1}
    post = [{"lam": 3.75, "sd_d-1": 0.1, "bias_theta_d-1": 0.19, "bias_log_mean_eps_d-1": 0.195, "frac_abs_z_gt_3_d-1": 0.08,
             "frac_abs_z_gt_3_d-1_vs_lowpass3": 0.0}]
    lb.plot({"sweep": sweep, "posteriors": post, "reference": ref}, tmp_path / "a.png")
    lb.plot({"sweep": sweep, "posteriors": [], "reference": ref}, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 5000 and (tmp_path / "b.png").stat().st_size > 5000


def test_lambda_band_posterior_rows_are_recomputed_from_draws_and_can_drop_the_rough_start_chains(tmp_path):
    import lambda_band as lb

    grid = Grid(48, 2.0)
    chi = austria_fixed(grid, 2.0)
    truth = np.log1p(chi)
    dist = ea.signed_distance(chi)
    rng = np.random.default_rng(0)
    theta = truth + 0.05 * rng.standard_normal((4, 60, 48, 48))
    theta[3] += 0.3 * rng.standard_normal((48, 48))  # chain 3 (started at a rough GMRF draw) sits somewhere else
    d = tmp_path / "run"
    d.mkdir()
    np.savez(d / "chains.npz", theta=theta.astype(np.float32), theta_true=truth, chi_true=chi)
    (d / "summary.json").write_text(json.dumps({"args": {"lam": 3.0, "n": 48, "side": 2.0}, "convergence": {"converged": False}}))
    rows = lb.posterior_rows("run", tmp_path, grid.h)
    assert len(rows) == 1 and rows[0]["chains_used"] == [0, 1, 2, 3] and rows[0]["lam"] == 3.0  # no starts.json: one row
    (d / "starts.json").write_text(json.dumps({"starts": [{"kind": k} for k in ("truth", "map", "matern", "gmrf")]}))
    rows = lb.posterior_rows("run", tmp_path, grid.h)
    assert len(rows) == 2 and rows[1]["chains_used"] == [0, 1, 2] and "without the rough-start chains [3], first 50 draws dropped" in rows[1]["label"] and rows[1]["burn"] == 50 and rows[1]["draws_per_chain"] == 10
    t32 = theta.astype(np.float32).astype(np.float64)
    allr, sub = rows
    assert allr["bias_theta_d-1"] == pytest.approx((t32.mean(axis=(0, 1)) - truth)[dist == -1].mean())
    assert sub["bias_theta_d-1"] == pytest.approx((t32[:3, 50:].mean(axis=(0, 1)) - truth)[dist == -1].mean())
    assert sub["bias_log_mean_eps_d-1"] >= sub["bias_theta_d-1"]  # Jensen: log E exp >= E
    assert sub["sd_d-1"] == pytest.approx(0.05, rel=0.1) and allr["sd_d-1"] > 2 * sub["sd_d-1"]  # the offset chain widens the pooled sd
    assert len(sub["bias_per_chain_d-1"]) == 3 and sub["frac_abs_z_gt_3_pooled_vs_lowpass3"] >= 0.0
