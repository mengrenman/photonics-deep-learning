"""Where the posterior misses the truth, by distance to the object edge, and an independent recomputation of the z statistics.

    python scripts/edge_analysis.py RUN [RUN ...] [--dir results/prior_comparison] [--out results/prior_comparison]

For each run of scripts/run_prior_comparison.py (summary.json and chains.npz must exist) this

  * recomputes the posterior mean and std of theta = log eps_r directly from the stored thinned draws (chains.npz
    ``theta``), not from the Welford moments, and checks them against the Welford values used by the summary;
  * recomputes the |z| > 3 fraction and the edge share from those, and from the two halves of the draws of each chain
    pooled separately (a Monte Carlo stability check) and with the Laplace std of the MAP in place of the posterior std;
  * reports the |z| > 3 fraction of each chain on its own (its own mean and within-chain std), the chain-to-chain spread of
    which is the Monte Carlo error of the headline fraction;
  * bins the bias (mean - truth), the posterior std and z by the signed distance to the object boundary in pixels
    (Euclidean distance rounded to pixels, positive inside the object, negative outside; +-1 are the pixels next to the
    boundary, the "ring");
  * reports the signed bias sums of the edge band (blur: negative inside, positive outside);
  * reports the bias of log E[eps_r] (the log of the posterior mean of the permittivity, Welford mean of eps_r over all steps) next to the
    bias of E[theta] = E[log eps_r], by distance. They differ by about sd^2 / 2 (Jensen) where the posterior is wide: the posterior mean of
    theta of a rough posterior is lower than the log of the mean permittivity, so a comparison of E[theta] biases across priors whose posterior
    sd differs (the adapted GMRF, sd 0.34, against Matern, sd 0.06) is not like-for-like; the log E[eps_r] bias is the one to compare;
  * reports, per posterior draw, the mean of theta over the pixels at d = -1 (its mean and sd across draws: the blur is in each draw, not only
    in the average) and quantiles of the pixel values at d = -1 pooled over draws.

Writes <out>/edge_analysis.json and <out>/edge_profile.png.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import distance_transform_edt

import bayes_scatter  # noqa: F401
from bayes_scatter import features as ft

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["pc_matern", "pc_gmrf", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1", "pcs1_qggmrf_sg0.05", "pcg1_qggmrf_sg0.05",
                "pcg1_matern"]
BINS = list(range(-6, 7))


def signed_distance(chi_true):
    """Signed Euclidean distance (pixel centers, rounded to an integer) to the object boundary: +d inside (d >= 1 is the
    distance to the nearest outside pixel), -d outside (d >= 1 the distance to the nearest inside pixel). |d| = 1 holds the
    4- and 8-neighbors of the boundary (distances 1 and sqrt 2), i.e. the "ring" of the report."""
    inside = chi_true > 1e-9
    return np.where(inside, distance_transform_edt(inside), -distance_transform_edt(~inside)).round().astype(int)


def z_stats(mean, std, truth, chi_true):
    z = ft.z_statistics(mean, std, truth, chi_true)
    return {k: z[k] for k in ("frac_abs_z_gt_3", "frac_inside", "frac_outside", "edge_share", "max_abs_z")}


def analyze(run, base):
    d = base / run
    s = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    theta = z["theta"].astype(np.float64)  # (chains, N, n, n)
    truth, chi_true = z["theta_true"], z["chi_true"]
    chains, n_draws = theta.shape[:2]
    mean = theta.mean(axis=(0, 1))
    std = np.sqrt(theta.var(axis=1, ddof=1).mean(axis=0) + theta.mean(axis=1).var(axis=0))
    w_mean, w_std = ft.pooled_moments(z["welford_mean"], z["welford_var"])
    out = {"run": run, "converged": s["convergence"]["converged"], "draws_per_chain": int(n_draws), "chains": int(chains),
           "stored_draw_mean_vs_welford_max_abs": float(np.abs(mean - w_mean[0]).max()),
           "stored_draw_std_vs_welford_max_rel": float((np.abs(std - w_std[0]) / w_std[0]).max()),
           "z_from_stored_draws": z_stats(mean, std, truth, chi_true),
           "z_from_summary": {k: s["z"][k] for k in ("frac_abs_z_gt_3", "frac_inside", "frac_outside", "edge_share", "max_abs_z")},
           "z_with_laplace_std": z_stats(mean, z["laplace_theta_std"], truth, chi_true),
           "z_of_the_MAP_with_posterior_std": z_stats(z["theta_hat"], std, truth, chi_true)}
    per_chain = [ft.z_statistics(theta[c].mean(axis=0), theta[c].std(axis=0, ddof=1), truth, chi_true)["frac_abs_z_gt_3"]
                 for c in range(chains)]
    out["z_per_chain_frac_abs_z_gt_3"] = {"values": per_chain, "min": float(min(per_chain)), "max": float(max(per_chain)),
                                          "sd": float(np.std(per_chain, ddof=1)) if chains > 1 else float("nan"),
                                          "mc_error_of_pooled": float(np.std(per_chain, ddof=1) / np.sqrt(chains)) if chains > 1
                                          else float("nan"),
                                          "note": "each chain's own mean and std against the truth; the pooled fraction uses all chains"}
    half = n_draws // 2
    for name, sl in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
        t = theta[:, sl]
        out[f"z_{name}_of_draws"] = z_stats(t.mean(axis=(0, 1)),
                                            np.sqrt(t.var(axis=1, ddof=1).mean(axis=0) + t.mean(axis=1).var(axis=0)),
                                            truth, chi_true)
    dist = signed_distance(chi_true)
    bias, zz = mean - truth, (mean - truth) / std
    chain_bias = theta.mean(axis=1) - truth  # (chains, n, n): each chain's own posterior-mean bias
    prof = {}
    for b in BINS:
        m = dist == b
        if m.any():
            per = [float(chain_bias[c][m].mean()) for c in range(chains)]
            prof[str(b)] = {"pixels": int(m.sum()), "bias": float(bias[m].mean()), "abs_bias": float(np.abs(bias[m]).mean()),
                            "std": float(std[m].mean()), "z": float(zz[m].mean()), "frac_abs_z_gt_3": float((np.abs(zz[m]) > 3).mean()),
                            # the bias of each chain's own mean in this bin: its spread is the Monte Carlo error of "bias"
                            "bias_per_chain": per,
                            "bias_mc_error": float(np.std(per, ddof=1) / np.sqrt(chains)) if chains > 1 else float("nan")}
    out["profile_by_signed_distance_to_edge_px"] = prof
    log_bias = np.log(w_mean[1]) - truth  # bias of log E[eps_r]
    far_bg = np.abs(dist) >= 8
    out["log_mean_eps_bias"] = {
        "by_distance": {str(b): float(log_bias[dist == b].mean()) for b in BINS if (dist == b).any()},
        "bias_d-1": float(log_bias[dist == -1].mean()), "bias_d+2": float(log_bias[dist == 2].mean()),
        "theta_bias_d-1": float(bias[dist == -1].mean()), "theta_bias_d+2": float(bias[dist == 2].mean()),
        "mean_log_eps_minus_theta_mean_far_background": float((log_bias - bias)[far_bg & (dist < 0)].mean()) if (far_bg & (dist < 0)).any() else None,
        "half_variance_far_background": float((0.5 * std**2)[far_bg & (dist < 0)].mean()) if (far_bg & (dist < 0)).any() else None,
        "note": "log E[eps_r] - theta_true; E[theta] - theta_true is 'profile_by_signed_distance_to_edge_px'. The two differ by ~ sd^2 / 2."}
    ring = dist == -1
    draw_ring = theta[:, :, ring].mean(axis=-1) - truth[ring].mean()  # (chains, N): bias of the ring-pixel mean of every draw
    vals = theta[:, :, ring].ravel()
    out["per_draw_edge_blur"] = {
        "pixels_at_d-1": int(ring.sum()), "mean_over_pixels_of_draw_bias_d-1": float(draw_ring.mean()),
        "sd_across_draws": float(draw_ring.std(ddof=1)), "min_over_draws": float(draw_ring.min()), "max_over_draws": float(draw_ring.max()),
        "pixel_value_quantiles_d-1": {q: float(np.quantile(vals, float(q))) for q in ("0.01", "0.25", "0.5", "0.75", "0.99")},
        "frac_pixel_values_above_0.6_d-1": float((vals > 0.6).mean()), "true_edge_jump_log2": float(np.log(2.0)),
        "note": "mean over the d = -1 pixels of theta - truth, per draw: if the average blurred sharp edges at random positions the draws "
                "would show a bimodal pixel distribution (values near 0 and near log 2) and a large draw-to-draw spread of this mean"}
    band = np.abs(dist) == 1
    out["edge_band"] = {"pixels": int(band.sum()), "frac_of_doi": float(band.mean()),
                        "frac_abs_z_gt_3": float((np.abs(zz[band]) > 3).mean()),
                        "inside_pixels_mean_bias": float(bias[dist == 1].mean()),
                        "outside_pixels_mean_bias": float(bias[dist == -1].mean())}
    far = np.abs(dist) >= 4
    out["far_from_edge"] = {"pixels": int(far.sum()),
                            "frac_abs_z_gt_3": float((np.abs(zz[far]) > 3).mean()) if far.any() else float("nan")}
    return out, prof


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    base = Path(args.dir)
    out_dir = Path(args.out) if args.out else base
    res, profs = {}, {}
    for r in args.runs:
        res[r], profs[r] = analyze(r, base)
        e = res[r]
        print(f"{r:20s} {'converged' if e['converged'] else 'UNCONVERGED':11s} |z|>3: summary {100 * e['z_from_summary']['frac_abs_z_gt_3']:.1f}% "
              f"(per chain {100 * e['z_per_chain_frac_abs_z_gt_3']['min']:.1f} to {100 * e['z_per_chain_frac_abs_z_gt_3']['max']:.1f}%), "
              f"stored draws {100 * e['z_from_stored_draws']['frac_abs_z_gt_3']:.1f}%, halves "
              f"{100 * e['z_first_half_of_draws']['frac_abs_z_gt_3']:.1f}% / {100 * e['z_second_half_of_draws']['frac_abs_z_gt_3']:.1f}%, "
              f"Laplace std {100 * e['z_with_laplace_std']['frac_abs_z_gt_3']:.1f}%; edge band {100 * e['edge_band']['frac_abs_z_gt_3']:.1f}% "
              f"vs far from edge {100 * e['far_from_edge']['frac_abs_z_gt_3']:.1f}%")
    (out_dir / "edge_analysis.json").write_text(json.dumps(res, indent=2, default=float))
    fig = Figure(figsize=(11, 3.4), layout="constrained")
    ax = fig.subplots(1, 3)
    xs = BINS
    main_runs = [r for r in profs if r not in DEFAULT_RUNS[4:]] or list(profs)  # the robustness controls are in the JSON only
    for r in main_runs:
        prof = profs[r]
        lab = r.replace("pc_", "") + ("" if res[r]["converged"] else " (unconverged)")
        ys = [[prof.get(str(b), {}).get(k, np.nan) for b in xs] for k in ("bias", "std", "z")]
        for a, y in zip(ax, ys):
            a.plot(xs, y, marker="o", ms=3, label=lab)
    for a, t in zip(ax, ("bias: posterior mean - truth (log eps_r)", "posterior std (log eps_r)", "mean z")):
        a.axvline(0.0, color="0.7", lw=0.8)
        a.set_xlabel("signed distance to the object edge [pixels] (+ inside)", fontsize=8)
        a.set_title(t, fontsize=9)
        a.tick_params(labelsize=7)
    ax[0].legend(fontsize=7)
    fig.savefig(out_dir / "edge_profile.png", dpi=130)
    print(f"wrote {out_dir}/edge_analysis.json and edge_profile.png")
    return res


if __name__ == "__main__":
    main()
