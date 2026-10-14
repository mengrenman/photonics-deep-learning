"""Milestone 7 side experiments with F2 Laplace fits (DESIGN 14, questions i to iii), cheap enough to repeat over many data sets (one MAP and one Fisher matrix each, no chains).

    python scripts/parametric_sweeps.py --gap 0.1 --level 0.02                # (iii) one ellipticity level: data, F2 MAP, Laplace, refit predictive check
    python scripts/parametric_sweeps.py --gap 0.05 --level 0 --seeds 1 2 3 --ppc-reps 0   # (i) the right family (level 0 = T1) over noise realizations: calibration
    python scripts/parametric_sweeps.py --gap 0.1 --collect                   # tables (ellipticity_g<gap>.*, calibration_g<gap>.*) and figure from the files found

(iii) A dose-response in the ellipticity of T2: how small a model error does the data still flag, and how wrong is the gap by then? (i) Calibration: over many noise
realizations of the same truth, how often does the 90% Laplace interval of the gap cover the true gap, and are the z-scores standard normal? The Laplace Gaussian is
checked against the chains of the main experiment (``results/parametric``) at the T1 gaps.

Flags of the ladder (``ladder_derived``). Besides the replicate-based refit p-value (30 replicates: it cannot resolve p below 1/31, and 0.000 means "below 0.032"), every level gets
the analytic one, p = P(Gamma(N - 5/2) >= Phi_MAP) (``shapepost.misfit_p_value``, no resolution limit); the excess of the misfit over the noise-only misfit of the same noise
realization, dPhi = Phi_MAP - sum |d_obs - d_clean|^2 / s^2 (the realization's own luck removed: seed 0 is +0.78 sd high); and the coefficient c = dPhi / z^2 of the scaling dPhi = c z^2
between the misfit excess and the squared distance z (in posterior sd) of the gap from the truth. c does not depend on the SNR (both scale as 1 / s^2) and so tells at which
z a 3 sd misfit flag (dPhi = 3 sqrt(N) = 147) is reached: z = sqrt(147 / c).

The truth is the pair of ellipses of T2 with semi-axes (r, (1 + level) r) (long axis along y, level 0.15 is T2 itself, level 0 is T1 rendered by the ellipse code), the
edge-to-edge gap ``--gap`` along x, eps_r 2, data from the n = 128 grid at 30 dB of the object's own scattered power (``--seeds``: the noise key; the same realization at every
level for a given seed), inference on n = 64 with the book rings, the F2 (two equal circles) model of the main experiment. Per level: the MAP misfit against N, the Laplace interval of the
gap (as in ``parametric_collect.laplace_rows``) and whether it covers the true gap, and the refit posterior-predictive check (``shapepost.predictive_check``, draws from the
Laplace Gaussian, ``--ppc-reps`` replicates). The Laplace interval is a Gaussian approximation, not a sampled posterior; for the levels that matter (the misfit is
large) the main experiment's chains check it at level 0.15. Files: <out-dir>/E<level>g<gap>[s<seed>].json (seed 0 has no suffix; resumable: a finished (level, seed) is skipped), <out-dir>/data/E<level>g<gap>[s<seed>].npz.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import sys
import time
from math import erf, sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.figure import Figure

import parametric_run as prun
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "parametric" / "ellipticity"
LEVELS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.15)
print = functools.partial(print, flush=True)


def level_label(gap, level, seed=0):
    return f"E{level:g}g{gap:g}" + (f"s{seed}" if seed else "")


def make_scene(gap, level, r=0.2, eps_r=2.0):
    """Two ellipses of semi-axes (r, (1 + level) r), long axis along y, edge-to-edge gap ``gap`` along x: ``ellipses.EllipseScene``."""
    ay = (1.0 + level) * r
    ell = np.array([[-(r + 0.5 * gap), 0.0, r, ay, 0.0], [r + 0.5 * gap, 0.0, r, ay, 0.0]])
    return el.EllipseScene(ell, eps_r, name="T2e", true_gap=gap, description=f"two ellipses {r} x {ay:.4f} (long axis along y), gap {gap}")


def run_level(args, level, seed=0):
    """Data, F2 MAP, Laplace fit and refit predictive check (skipped for ``--ppc-reps 0``) of one ellipticity level and noise seed; writes and returns the JSON record (skipped if it exists)."""
    lab = level_label(args.gap, level, seed)
    path = Path(args.out_dir) / f"{lab}.json"
    if path.exists() and not args.force:
        print(f"{lab}: done")
        return json.loads(path.read_text())
    t0 = time.perf_counter()
    scene = make_scene(args.gap, level, args.radius, args.eps_r)
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    dpath = Path(args.out_dir) / "data" / f"{lab}.npz"
    if dpath.exists() and not args.force:
        z = np.load(dpath)
        d_obs, s = jnp.asarray(z["d_obs"]), float(z["s"])
    else:
        data_setup = Setup(Grid(args.n * args.data_factor, args.side), setup.antennas)
        d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(scene.chi(data_setup.grid)))
        s = float(noise_std_from_snr(d_clean, args.snr))
        d_obs = add_noise(jax.random.key(seed), d_clean, s)
        prun.save_npz(dpath, d_clean=np.asarray(d_clean), d_obs=np.asarray(d_obs), s=s, seed=seed, snr=args.snr)
    forward = Forward(setup, tol=args.tol)
    model = sm.F2()
    post = sp.ShapePosterior(model, forward, d_obs, s, el.EllipseRenderer(setup.grid))
    n_data = args.n_tx * args.n_rx
    rng = np.random.default_rng(args.map_seed)
    starts = np.vstack([np.zeros((1, model.dim)), model.start_points_u(rng, args.n_prior_starts)])
    search = sp.map_estimate(post, starts, ["prior mode"] + [f"prior draw {k}" for k in range(args.n_prior_starts)], verbose=False)
    fit = sp.laplace_fit(post, search.u_hat)
    p_hat = post.physical(search.u_hat)
    g_hat = float(p_hat[0])
    sd_g = float(fit.sd()[0]) * g_hat * (1.0 - g_hat / sm.G_MAX)
    lo, hi = max(g_hat - 1.645 * sd_g, 0.0), g_hat + 1.645 * sd_g
    g_det = prun.g_detection(args.eps_r)
    draws = np.asarray(search.u_hat) + (np.linalg.cholesky(fit.cov) @ np.random.default_rng(args.ppc_seed).standard_normal((model.dim, args.ppc_reps))).T
    ppc = None
    if args.ppc_reps > 0:
        ppc = sp.predictive_check(post, draws, jax.random.key(args.ppc_seed), n_rep=args.ppc_reps, mode="refit", u_hat=search.u_hat)
    best = search.best_candidate
    rec = {"label": lab, "level": level, "seed": seed, "semi_axes": [args.radius, (1.0 + level) * args.radius], "g_true": args.gap, "g_det": g_det, "n_data": n_data, "noise_std": s,
           "phi_map": best["phi"], "phi_over_N": best["phi"] / n_data, "map_converged": best["converged"], "p_hat": p_hat, "g_hat": g_hat, "g_sd": sd_g, "g_ci": [lo, hi],
           "g_error": g_hat - args.gap, "z_true": (args.gap - g_hat) / sd_g if sd_g > 0 else None, "covers_true": bool(lo <= args.gap <= hi),
           "p_below_g_det": 0.5 * (1.0 + erf((g_det - g_hat) / (sd_g * sqrt(2.0)))) if g_det is not None and sd_g > 0 else None,
           "ppc_refit": None if ppc is None else {k: ppc[k] for k in ("p_value", "t_obs", "t_rep_mean", "t_rep_sd", "z", "n_rep")}, "log_z_laplace": sp.laplace_log_evidence(post, fit),
           "time_s": time.perf_counter() - t0, "thread_env": prun.thread_env()}
    prun.write_json(path, rec)
    ptxt = "" if ppc is None else f"refit p {ppc['p_value']:.3f} (z {ppc['z']:.1f}); "
    print(f"{lab}: Phi_MAP/N {rec['phi_over_N']:.3f}, {ptxt}g = {g_hat:.5f} +- {sd_g:.5f} (true {args.gap}, z {rec['z_true']:.1f}), covers {rec['covers_true']}; {rec['time_s']:.0f} s")
    return rec


def load_records(args, seed_zero_only):
    recs = []
    for p in Path(args.out_dir).glob(f"E*g{args.gap:g}*.json"):
        r = json.loads(p.read_text())
        if "level" in r and abs(r["g_true"] - args.gap) < 1e-12 and (r["seed"] == 0 or not seed_zero_only):
            recs.append(r)
    return sorted(recs, key=lambda r: (r["level"], r["seed"]))


def calibration_summary(recs):
    """Per ellipticity level with at least 4 noise seeds: coverage of the 90% Laplace interval of the gap, mean and sd of z = (g_true - g_hat) / sd, the rms error over the mean
    sd, and the mean Phi_MAP / N. (With M seeds the coverage has a standard error of sqrt(0.09 / M) and the sd of z one of about 1 / sqrt(2 (M - 1)).)"""
    out = []
    for level in sorted({r["level"] for r in recs}):
        grp = [r for r in recs if r["level"] == level]
        if len(grp) < 4:
            continue
        z = np.array([r["z_true"] for r in grp])
        err = np.array([r["g_error"] for r in grp])
        sd = np.array([r["g_sd"] for r in grp])
        m = len(grp)
        out.append({"level": level, "g_true": grp[0]["g_true"], "n_seeds": m, "seeds": [r["seed"] for r in grp], "coverage_90": float(np.mean([r["covers_true"] for r in grp])),
                    "coverage_se": float(np.sqrt(0.09 / m)), "z_mean": float(z.mean()), "z_sd": float(z.std(ddof=1)), "z_sd_se": float(1.0 / np.sqrt(2.0 * (m - 1))),
                    "z_mean_se": float(1.0 / np.sqrt(m)), "bias": float(err.mean()), "rmse": float(np.sqrt(np.mean(err**2))), "mean_sd": float(sd.mean()),
                    "rmse_over_mean_sd": float(np.sqrt(np.mean(err**2)) / sd.mean()), "phi_over_N_mean": float(np.mean([r["phi_over_N"] for r in grp])),
                    "phi_over_N_sd": float(np.std([r["phi_over_N"] for r in grp], ddof=1))})
    return out


def noise_only_misfit(out_dir, label):
    """sum |d_obs - d_clean|^2 / s^2 of the data of one level file: the misfit of the noise realization itself (the value Phi_MAP would have if the model were exact and had no
    free parameters); None if the data file is missing."""
    path = Path(out_dir) / "data" / f"{label}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return float(np.sum(np.abs(z["d_obs"] - z["d_clean"]) ** 2) / float(z["s"]) ** 2)


def ladder_derived(recs, out_dir, dim=5):
    """The rows of the ladder (``recs``: level records of one gap, seed 0) with the analytic flag and the misfit-excess scaling added (module docstring): ``p_null`` and ``z_phi``
    (``shapepost.misfit_p_value``, ``dim`` parameters), ``noise_only_phi``, ``delta_phi`` = Phi_MAP - noise-only misfit, ``c_true`` = delta_phi / z_true^2 (None for |z| < 2: there the noise and grid components of z are not small against the model error), and
    against the right-family fit of the same noise (level 0): ``dz_shift`` = (g_hat - g_hat(level 0)) / sd and ``c_shift`` = delta_phi / dz_shift^2, which cancels the noise and
    grid components of the distance to the truth."""
    base = next((r for r in recs if r["level"] == 0.0), None)
    out = []
    for r in recs:
        d = dict(r)
        flag = sp.misfit_p_value(r["phi_map"], r["n_data"], dim)
        d["p_null"], d["z_phi"] = flag["p_value"], flag["z"]
        d["noise_only_phi"] = noise_only_misfit(out_dir, r["label"])
        d["delta_phi"] = None if d["noise_only_phi"] is None else r["phi_map"] - d["noise_only_phi"]
        z = r["z_true"]
        d["c_true"] = None if d["delta_phi"] is None or z is None or abs(z) < 2.0 else d["delta_phi"] / z**2
        d["dz_shift"] = None if base is None else (r["g_hat"] - base["g_hat"]) / r["g_sd"]
        d["c_shift"] = None if d["delta_phi"] is None or d["dz_shift"] is None or abs(d["dz_shift"]) < 2.0 else d["delta_phi"] / d["dz_shift"] ** 2
        out.append(d)
    return out


def collect(args):
    """Tables and figure from the files of this gap: <out-dir>/ellipticity_g<gap>.json / .md / .png (the seed-0 level ladder) and calibration_g<gap>.json / .md (levels with at least
    4 noise seeds)."""
    recs = load_records(args, seed_zero_only=True)
    allrecs = load_records(args, seed_zero_only=False)
    if not allrecs:
        raise SystemExit(f"no level files for gap {args.gap:g} in {args.out_dir}")
    stem = f"ellipticity_g{args.gap:g}"  # not Path.with_suffix: "g0.1" would lose its ".1"
    out = Path(args.out_dir)
    cal = calibration_summary(allrecs)
    if cal:
        header = ["ellipticity", "g true", "seeds", "coverage of the 90% interval", "mean z", "sd z", "bias of g", "rmse of g", "mean sd (Laplace)", "rmse / mean sd", "Phi_MAP/N mean (sd)"]
        rows = [[f"{c['level']:g}", f"{c['g_true']:g}", str(c["n_seeds"]), f"{c['coverage_90']:.2f} (se {c['coverage_se']:.2f})", f"{c['z_mean']:+.2f} (se {c['z_mean_se']:.2f})",
                 f"{c['z_sd']:.2f} (se {c['z_sd_se']:.2f})", f"{c['bias']:+.2e}", f"{c['rmse']:.2e}", f"{c['mean_sd']:.2e}", f"{c['rmse_over_mean_sd']:.2f}",
                 f"{c['phi_over_N_mean']:.3f} ({c['phi_over_N_sd']:.3f})"] for c in cal]
        md = ["# Calibration of the F2 Laplace interval of the gap over noise realizations\n", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        md += ["| " + " | ".join(r) + " |" for r in rows]
        (out / f"calibration_g{args.gap:g}.md").write_text("\n".join(md) + "\n")
        (out / f"calibration_g{args.gap:g}.json").write_text(json.dumps(cal, indent=1, default=prun._default))
        print(f"wrote {out / f'calibration_g{args.gap:g}.md'} ({len(cal)} levels with at least 4 seeds)")
    if len(recs) > 1 and all(r["ppc_refit"] is not None for r in recs):
        recs = ladder_derived(recs, out)
        n_rep = recs[0]["ppc_refit"]["n_rep"]
        fm = lambda x, spec: "-" if x is None else format(x, spec)  # noqa: E731
        header = ["ellipticity", "semi-axes", "Phi_MAP/N", "Phi_MAP - N", "p (Gamma null)", "z of Phi", f"refit p ({n_rep} reps, resolution 1/{n_rep + 1})", "refit z", "g MAP", "g sd", "g error",
                  "z of truth", "covers", "dPhi (excess over noise-only)", "c = dPhi / z^2", "c vs level 0", "log Z"]
        rows = [[f"{r['level']:g}", f"{r['semi_axes'][0]:g} x {r['semi_axes'][1]:.4f}", f"{r['phi_over_N']:.3f}", f"{r['phi_map'] - r['n_data']:.0f}", f"{r['p_null']:.2g}", f"{r['z_phi']:.1f}",
                 f"{r['ppc_refit']['p_value']:.3f}", f"{r['ppc_refit']['z']:.1f}", f"{r['g_hat']:.5f}", f"{r['g_sd']:.2e}", f"{r['g_error']:+.5f}", f"{r['z_true']:.1f}",
                 "yes" if r["covers_true"] else "no", fm(r["delta_phi"], ".1f"), fm(r["c_true"], ".2f"), fm(r["c_shift"], ".2f"), f"{r['log_z_laplace']:.1f}"] for r in recs]
        md = ["# T2 ellipticity ladder (F2 fit, Laplace interval; the noise realization is the same at every level)\n",
              "p (Gamma null) = P(Gamma(N - 5/2) >= Phi_MAP) is the flag; the refit p-value is from replicates and 0.000 means below its resolution. dPhi = Phi_MAP minus the misfit of this "
              "noise realization itself (+0.78 sd high for seed 0); c = dPhi / z^2 (z of truth) and c vs level 0 = dPhi / dz^2 with dz the shift of the gap from the level-0 fit of the same noise.\n",
              "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
        md += ["| " + " | ".join(r) + " |" for r in rows]
        (out / f"{stem}.md").write_text("\n".join(md) + "\n")
        (out / f"{stem}.json").write_text(json.dumps(recs, indent=1, default=prun._default))
        fig = Figure(figsize=(7.0, 5.2), dpi=130)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)
        x = np.array([max(r["level"], 1e-3) for r in recs])
        ax1.errorbar(x, [r["g_hat"] for r in recs], yerr=[[r["g_hat"] - r["g_ci"][0] for r in recs], [r["g_ci"][1] - r["g_hat"] for r in recs]], fmt="o", color="#2a78d6", capsize=3, ms=4)
        ax1.axhline(args.gap, color="#0b0b0b", lw=1.0)
        ax1.set_xscale("log")
        ax1.set_ylabel("F2 gap, 90% Laplace interval\n(true gap in black)", fontsize=9)
        ax2.plot(x, [r["phi_over_N"] for r in recs], "o-", color="#eb6834", ms=4)
        ax2.axhline(1.0, color="#0b0b0b", lw=0.9)
        ax2.set_yscale("log")
        ax2.set_xscale("log")
        ax2.set_ylabel("Phi_MAP / N", fontsize=9)
        ax2.set_xlabel("ellipticity (b/a - 1); level 0 plotted at 1e-3", fontsize=9)
        for ax in (ax1, ax2):
            ax.grid(color="#e4e3dd", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png")
        print(f"wrote {out / (stem + '.md')} ({len(recs)} levels)")
    return {"ladder": recs, "calibration": cal}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gap", type=float, default=0.1)
    ap.add_argument("--level", type=float, nargs="+", default=None, help=f"ellipticity levels (default: {LEVELS})")
    ap.add_argument("--collect", action="store_true", help="only build the table and figure from the level files that exist")
    ap.add_argument("--eps-r", type=float, default=2.0)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--data-factor", type=int, default=2)
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="noise keys of the data; one run per (level, seed). Seed 0 is the key of the main experiment")
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--n-prior-starts", type=int, default=2)
    ap.add_argument("--map-seed", type=int, default=0)
    ap.add_argument("--ppc-reps", type=int, default=30)
    ap.add_argument("--ppc-seed", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.collect:
        return collect(args)
    for level in (args.level if args.level is not None else LEVELS):
        for seed in args.seeds:
            run_level(args, level, seed)


if __name__ == "__main__":
    main()
