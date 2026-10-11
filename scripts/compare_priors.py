"""Compare the runs of scripts/run_prior_comparison.py across priors (docs/DESIGN.md section 11.3).

    python scripts/compare_priors.py RUN [RUN ...] [--out DIR] [--labels A B ...]

RUN is a directory name under results/prior_comparison/ or a path. The script

  1. refuses to compare runs whose data differ: the data arguments must agree and the stored d_obs arrays must agree
     to max |d_a - d_b| / s <= 1e-6 (s the noise std). The SHA-256 in each summary.json is reported, but it is not the
     criterion: the bits of d_obs depend on the XLA thread count, so equal data can have different hashes (it warns and
     continues in that case);
  2. tabulates per prior: convergence (DESIGN 11.2), physical Phi of the draws, posterior mean / std of theta and eps_r,
     |z| > 3 fraction and its edge share, err_model, err_chi;
  3. computes the pixelwise disagreement |m_A - m_B| / sqrt(s_A^2 + s_B^2) of every pair of posteriors, for theta =
     log eps_r and for eps_r, and the fraction of pixels above 2 (all, inside the object, outside, near an edge);
  4. gives the three feature probabilities of DESIGN 11.3 per prior and the verdict (``features.verdict``): the
     converged posteriors must all give P > 0.95 (present) or all P < 0.05 (absent); the verdict is "data-determined"
     only if, in addition, the DESIGN 8.7 minimum pair (a converged Matern AND a converged GMRF) is among them, with the
     q-GGMRF settings counting as one prior family; otherwise it is "provisional", and it is "not established" if an
     unconverged run gives the opposite side of the threshold.

Writes DIR/comparison.json (default DIR results/prior_comparison), DIR/comparison.npz (disagreement maps) and the
figures comparison_mean_std.png, comparison_disagreement.png, comparison_features.png.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import itertools
import json
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

import bayes_scatter  # noqa: F401
from bayes_scatter import features as ft
from bayes_scatter.geometry import Grid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "prior_comparison"
DATA_ARGS = ("n", "side", "eps_r", "snr", "seed", "data_factor", "n_tx", "n_rx")
D_OBS_TOL = 1e-6  # max |d_a - d_b| / s (noise std units) below which two data sets are the same realization


def load_run(spec):
    """(label, summary dict, chains.npz as a dict of arrays) for a run name or path."""
    path = Path(spec) if Path(spec).is_absolute() or Path(spec).exists() else RESULTS / spec
    summary = json.loads((path / "summary.json").read_text())
    z = np.load(path / "chains.npz")
    keep = ("welford_mean", "welford_var", "theta_true", "chi_true", "eps_true", "d_obs", "noise_std", "theta_hat",
            "laplace_theta_std", "prior_theta_std")
    return path.name, summary, {k: z[k] for k in keep}


def check_same_data(labels, summaries, arrays):
    """Exit unless all runs saw the same d_obs; returns a dict describing the check.

    The data arguments must agree, and every d_obs must match the first run's to ``D_OBS_TOL`` of the noise std. The
    hashes are reported (``identical`` is True when they are all equal, then the arrays are bit-identical) but a
    different hash with numerically equal arrays is accepted with a warning: the bits depend on the XLA thread count
    (scripts/run_prior_comparison.py), and the thread environments stored with the runs are returned.
    """
    ref = summaries[0]
    for lab, s in zip(labels, summaries):
        diff = {k: (ref["args"][k], s["args"][k]) for k in DATA_ARGS if ref["args"][k] != s["args"][k]}
        if diff:
            raise SystemExit(f"runs {labels[0]} and {lab} were made with different data arguments: {diff}")
    hashes = {lab: s["data"]["d_obs_sha256"] for lab, s in zip(labels, summaries)}
    threads = {lab: s["data"].get("thread_env") for lab, s in zip(labels, summaries)}
    s_ref = float(arrays[0]["noise_std"])
    worst = max(float(np.abs(a["d_obs"] - arrays[0]["d_obs"]).max()) / s_ref for a in arrays[1:]) if len(arrays) > 1 else 0.0
    if worst > D_OBS_TOL:
        raise SystemExit(f"the data differ between runs (d_obs hashes {hashes}; max |delta d| / s = {worst:.2e} > {D_OBS_TOL:g})")
    out = {"d_obs_sha256": hashes[labels[0]] if len(set(hashes.values())) == 1 else hashes,
           "identical": len(set(hashes.values())) == 1, "max_abs_diff_over_noise_std": worst, "tolerance": D_OBS_TOL,
           "max_rel_diff": worst * s_ref / float(np.abs(arrays[0]["d_obs"]).max()), "thread_env": threads}
    if not out["identical"]:
        print(f"WARNING: d_obs hashes differ but the arrays agree to {worst:.1e} of the noise std (round-off, e.g. a different "
              f"XLA thread count); continuing")
    return out


def pooled(arr):
    mean, std = ft.pooled_moments(arr["welford_mean"], arr["welford_var"])
    return mean, std  # (2, n, n) each: theta, eps_r


def pair_stats(d, inside, edge):
    big = d > 2.0
    return {"frac_gt_2": float(big.mean()), "frac_gt_2_inside": float(big[inside].mean()),
            "frac_gt_2_outside": float(big[~inside].mean()), "frac_gt_2_near_edge": float(big[edge].mean()),
            "frac_gt_2_away_from_edge": float(big[~edge].mean()), "max": float(d.max()), "mean": float(d.mean())}


def _extent(grid):
    cx, cy = grid.center
    h = 0.5 * grid.side
    return [cx - h, cx + h, cy - h, cy + h]


def _panel(fig, ax, grid, a, title, cmap, vmin=None, vmax=None, label=None):
    im = ax.imshow(np.asarray(a).T, origin="lower", extent=_extent(grid), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=6)
    if label:
        cb.set_label(label, fontsize=7)


def save(fig, path):
    fig.savefig(path, dpi=130)


def figure_mean_std(grid, labels, theta_true, means, stds, path):
    k = len(labels)
    fig = Figure(figsize=(2.9 * (k + 1), 5.6), layout="constrained")
    ax = fig.subplots(2, k + 1, squeeze=False)
    lo = min(theta_true.min(), *(m.min() for m in means))
    hi = max(theta_true.max(), *(m.max() for m in means))
    _panel(fig, ax[0, 0], grid, theta_true, "truth", "viridis", lo, hi, "log eps_r")
    ax[1, 0].axis("off")
    smax = max(s.max() for s in stds)
    for c, lab in enumerate(labels):
        _panel(fig, ax[0, c + 1], grid, means[c], f"{lab}\nposterior mean", "viridis", lo, hi, "log eps_r")
        _panel(fig, ax[1, c + 1], grid, stds[c], "posterior std", "magma", 0.0, smax, "log eps_r")
    save(fig, path)


def figure_disagreement(grid, pairs, maps, stats, path):
    if not pairs:
        return
    k = len(pairs)
    cols = min(k, 3)
    rows = int(np.ceil(k / cols))
    fig = Figure(figsize=(3.6 * cols, 3.4 * rows), layout="constrained")
    ax = np.atleast_2d(fig.subplots(rows, cols, squeeze=False))
    for a in ax.ravel():
        a.axis("off")
    for (i, (a, b)), d in zip(enumerate(pairs), maps):
        axi = ax[i // cols, i % cols]
        axi.axis("on")
        _panel(fig, axi, grid, d, f"{a} vs {b}\n{100 * stats[i]['frac_gt_2']:.1f}% of pixels > 2", "magma", 0.0, 4.0,
               "|dm| / sqrt(s_A^2 + s_B^2)")
    save(fig, path)


def figure_features(labels, converged, feats, truth, verdicts, path):
    names = list(verdicts)
    fig = Figure(figsize=(4.4 * len(names), 0.9 * len(labels) + 3.2), layout="constrained")
    ax = np.atleast_1d(fig.subplots(1, len(names), squeeze=False)[0])
    for a, name in zip(ax, names):
        p = np.array([feats[lab][name]["p"] for lab in labels])
        e = np.array([feats[lab][name]["mcse"] for lab in labels])
        e = np.where(np.isfinite(e), e, 0.0)
        y = np.arange(len(labels))[::-1]
        colors = ["tab:blue" if converged[lab] else "0.7" for lab in labels]
        a.barh(y, p, xerr=e, color=colors, height=0.6)
        a.axvline(0.05, color="k", lw=0.6, ls=":")
        a.axvline(0.95, color="k", lw=0.6, ls=":")
        a.set_yticks(y)
        a.set_yticklabels([lab + ("" if converged[lab] else " (unconverged)") for lab in labels], fontsize=7)
        a.set_xlim(0, 1)
        a.set_xlabel("posterior probability", fontsize=8)
        a.tick_params(labelsize=7)
        a.set_title(f"{name}\n{textwrap.fill(verdicts[name], 40)}" + ("\n(truth: yes)" if truth[name] else "\n(truth: no)"),
                    fontsize=7)
    save(fig, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories (names under results/prior_comparison/ or paths)")
    ap.add_argument("--out", default=str(RESULTS), help="output directory for comparison.json and the figures")
    ap.add_argument("--labels", nargs="*", default=None, help="display labels (default: the directory names)")
    args = ap.parse_args(argv)
    loaded = [load_run(r) for r in args.runs]
    labels = args.labels if args.labels else [lab for lab, _, _ in loaded]
    if len(labels) != len(loaded) or len(set(labels)) != len(labels):
        raise SystemExit("need one distinct label per run")
    summaries = [s for _, s, _ in loaded]
    arrays = [a for _, _, a in loaded]
    data_check = check_same_data(labels, summaries, arrays)

    a0 = summaries[0]["args"]
    grid = Grid(a0["n"], a0["side"])
    theta_true, chi_true = arrays[0]["theta_true"], arrays[0]["chi_true"]
    inside = chi_true > 1e-9
    edge = ft.edge_mask(chi_true)
    pm = [pooled(a) for a in arrays]  # per run: (mean (2, n, n), std (2, n, n))
    converged = {lab: bool(s["convergence"]["converged"]) for lab, s in zip(labels, summaries)}

    runs = {}
    for lab, s in zip(labels, summaries):
        runs[lab] = {
            "dir": args.runs[labels.index(lab)], "prior": {k: v for k, v in s["prior"].items() if k != "prior_summary"},
            "prior_theta_std_mean": s["prior"]["prior_summary"].get("theta_std", {}).get("mean"),
            "converged": converged[lab], "convergence": {k: s["convergence"][k] for k in (
                "rhat_phi", "rhat_pixel_max", "ess_pixel_min", "ess_pixel_min_geyer", "unconverged_reasons")},
            "phi_physical_mean": s["phi_physical"]["mean"], "map_phi": s["map"]["phi"], "map_nit": s["map"]["nit"],
            "acceptance": s["sampler"]["acceptance"]["mean"], "z": s["z"], "err_model": s["metrics"]["err_model"],
            "err_chi": s["metrics"]["err_chi"],
            "theta_std": s["posterior"]["theta_std"], "mcmc_over_laplace_std": s["posterior"]["mcmc_over_laplace_std"],
            "theta_std_over_prior_std": s["posterior"]["theta_std_over_prior_std"],
            "operating_point_model_to_noise": s["operating_point"]["model_to_noise"]}

    # pairwise disagreement
    pairs, dmaps, dmaps_eps, pair_info = [], [], [], {}
    for (ia, a), (ib, b) in itertools.combinations(enumerate(labels), 2):
        d_theta = ft.disagreement(pm[ia][0][0], pm[ia][1][0], pm[ib][0][0], pm[ib][1][0])
        d_eps = ft.disagreement(pm[ia][0][1], pm[ia][1][1], pm[ib][0][1], pm[ib][1][1])
        pairs.append((a, b))
        dmaps.append(d_theta)
        dmaps_eps.append(d_eps)
        pair_info[f"{a} vs {b}"] = {"theta": pair_stats(d_theta, inside, edge), "eps_r": pair_stats(d_eps, inside, edge),
                                    "both_converged": converged[a] and converged[b]}
    edge_info = {"edge_fraction_of_doi": float(edge.mean()), "inside_fraction_of_doi": float(inside.mean())}

    # features
    feats = {lab: s["features"]["probabilities"] for lab, s in zip(labels, summaries)}
    names = list(next(iter(feats.values())))
    truth_ok = {name: bool(feats[labels[0]][name]["truth_satisfies"]) for name in names}
    family = {lab: s["prior"]["kind"] for lab, s in zip(labels, summaries)}  # q-GGMRF settings share one family
    verdicts = {name: ft.verdict({lab: feats[lab][name]["p"] for lab in labels}, converged, family=family) for name in names}
    conv_families = sorted({family[lab] for lab in labels if converged[lab]})
    minimum_met = all(f in conv_families for f in ft.REQUIRED_FAMILIES)
    table = {name: {lab: {"p": feats[lab][name]["p"], "mcse": feats[lab][name]["mcse"],
                          "p_chain": feats[lab][name]["p_chain"], "converged": converged[lab]} for lab in labels}
             for name in names}
    features = {"delta": summaries[0]["features"]["delta"], "rule": "data-determined only if every converged "
                "posterior gives P > 0.95 (present) or every one gives P < 0.05 (absent) AND the DESIGN 8.7 minimum pair "
                f"{list(ft.REQUIRED_FAMILIES)} has a converged run each (q-GGMRF settings are one family); provisional "
                "without the pair; not established if an unconverged run disagrees",
                "family": family, "converged_families": conv_families, "design_minimum_met": minimum_met,
                "table": table, "verdict": verdicts, "truth_satisfies_feature": truth_ok}

    out = {"runs": runs, "labels": labels, "data_check": data_check, "pairs": pair_info, "edge": edge_info,
           "features": features, "n_converged": int(sum(converged.values())),
           "note": "pairs involving an unconverged run are descriptive only; no conclusion rests on them"}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=2, default=float))
    np.savez_compressed(out_dir / "comparison.npz", labels=np.array(labels),
                        pairs=np.array([f"{a} vs {b}" for a, b in pairs]), disagreement_theta=np.array(dmaps),
                        disagreement_eps=np.array(dmaps_eps), theta_mean=np.stack([m[0][0] for m in pm]),
                        theta_std=np.stack([m[1][0] for m in pm]), eps_mean=np.stack([m[0][1] for m in pm]),
                        eps_std=np.stack([m[1][1] for m in pm]))
    figure_mean_std(grid, [lab + ("" if converged[lab] else " (unconverged)") for lab in labels], theta_true,
                    [m[0][0] for m in pm], [m[1][0] for m in pm], out_dir / "comparison_mean_std.png")
    figure_disagreement(grid, pairs, dmaps, [pair_info[f"{a} vs {b}"]["theta"] for a, b in pairs],
                        out_dir / "comparison_disagreement.png")
    figure_features(labels, converged, feats, truth_ok, verdicts, out_dir / "comparison_features.png")

    print(f"data: {'identical d_obs' if data_check['identical'] else 'd_obs agree to round-off'}; "
          f"{out['n_converged']} of {len(labels)} runs converged; DESIGN minimum (matern + gmrf converged) "
          f"{'met' if minimum_met else 'NOT met'}")
    for lab in labels:
        r = runs[lab]
        print(f"  {lab:24s} {'converged  ' if r['converged'] else 'UNCONVERGED'} Phi {r['phi_physical_mean']:.1f}  |z|>3 "
              f"{100 * r['z']['frac_abs_z_gt_3']:.1f}%  err_model {r['err_model']['posterior_mean']:.4f}  err_chi "
              f"{r['err_chi']['posterior_mean']:.4f}")
    for name, info in pair_info.items():
        print(f"  {name}: {100 * info['theta']['frac_gt_2']:.1f}% of pixels with disagreement > 2 (theta)")
    for name in names:
        ps = ", ".join(f"{lab} {feats[lab][name]['p']:.3f}" for lab in labels)
        print(f"  {name}: {ps} -> {verdicts[name]}")
    print(f"wrote {out_dir}")
    return out


if __name__ == "__main__":
    main()
