"""Milestone 7 (DESIGN 14, question iii): does the misfit flag a wrong family at the book's SNRs, and how often? F2 (or F2u) Laplace fits of the data of a truth at 10, 20 and 30 dB.

    python scripts/parametric_snr.py --truth T5 --gap 0.1 --snr 10 20 --seeds 0           # one data realization (the cached noise key 0, scaled)
    python scripts/parametric_snr.py --truth T5 --gap 0.1 --snr 10 20 --seeds 0 1 2 ... 19  # typical draws: 20 noise realizations per SNR
    python scripts/parametric_snr.py --label E0.01g0.1 --gap 0.1 --data-dir results/parametric/ellipticity/data --snr 30 --seeds 0 1 2 ...   # a rung of the ladder
    python scripts/parametric_snr.py --collect                                              # tables snr_summary.md / .json from the unit files that exist

Why. The 30 dB fits of the main experiment (``parametric_run.py``) separate the families by a large margin, but the dose-response of the ellipticity ladder shows that the misfit
excess grows as the SQUARE of the gap error, dPhi = c z^2, with c fixed by the kind of model error and independent of the SNR (the data of a fixed deformation have a bias fixed in
wavelengths; z scales as 1 / s and dPhi as 1 / s^2). A 3 sd misfit flag (dPhi = 3 sqrt(N) = 147 nats) therefore needs z = sqrt(147 / c) whatever the SNR, and z can be large
before dPhi is: at the book's 10 dB the F2 interval on T5 can exclude the true gap by 11 sd while the misfit is 2 sd above its null. This script measures it.

A unit is (truth, model, SNR, noise seed) and writes <out-dir>/<truth label>_<model>_snr<SNR>_s<seed>.json (resumable: a finished unit is skipped). The data are the noise-free
data of the truth (<results>/data/<label>.npz, n = 128 grid) plus noise of the SNR (``likelihood.noise_std_from_snr``, key ``jax.random.key(seed)``; seed 0 at 30 dB reproduces the
cached data exactly, and seed 0 at another SNR is that realization scaled by 10^((30 - SNR) / 20)). The fit is the MAP of the n = 64 model (damped Gauss-Newton from the 30 dB MAP
of the same case; L-BFGS first if that does not converge), the Laplace interval of the gap (as in ``parametric_collect.laplace_rows``), and the flags: Phi_MAP against the analytic null
Gamma(N - dim / 2) (``shapepost.misfit_p_value``), the excess over the noise-only misfit of the same realization (dPhi), and c = dPhi / z^2. ``--ppc-reps R`` adds the refit
posterior-predictive check from R replicates (Laplace draws; slow, for a few units).

``--noise-check``: the noise generator against its law. The noise-only misfit sum |e|^2 / s^2 of N complex entries is Gamma(N, 1); over the first 1000 keys the mean is 2400.5 and the
sd 47.8 (theory 49.0; Kolmogorov-Smirnov p = 0.89), while keys 0 to 39 happen to have sd 41.3 and key 0 sits +0.78 sd high (2438.2): the narrow empirical null of the 41-seed
calibration (sd 40.7) and of 30 predictive replicates (38.2) is a property of those keys, not of the null law (<out-dir>/noise_null_check.json).

The Laplace Gaussian is a MAP-and-curvature summary. It is checked against chains at 30 dB (``results/parametric/T1g0.05_F2/summary.json``); for T4 (g -> 0, on the boundary of the
gap coordinate) the interval is not a posterior and only the misfit is of interest.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import parametric_run as prun
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parametric"
OUT = RESULTS / "snr"
print = functools.partial(print, flush=True)

def unit_label(label, model, snr, seed):
    return f"{label}_{model}_snr{snr:g}_s{seed}"


def reference_p(model_name, g):
    """A generic start p for the model near the right family (equal disks of radius 0.2 and eps_r 2 at the gap ``g``; F2u: separate radii and permittivities)."""
    if model_name == "F2":
        return np.array([max(g, 0.005), 0.2, 2.0, 0.0, 0.0])
    if model_name == "F2u":
        return np.array([max(g, 0.005), 0.2, 0.2, 2.0, 2.0, 0.0, 0.0])
    raise ValueError("the SNR experiment fits F2 or F2u")


def fit_unit(post, u_starts, newton_iter=40):
    """The MAP of ``post`` from the first start that converges under damped Gauss-Newton; if none does, L-BFGS (100 iterations) then Newton from each start (``shapepost``). Returns
    ``(u_hat, info, label)``."""
    best = None
    for k, u0 in enumerate(u_starts):
        u, info = sp.newton_map(post, u0, max_iter=newton_iter)
        if info["converged"]:
            return u, info, f"newton from start {k}"
        if best is None or info["U"] < best[1]["U"]:
            best = (u, info, f"newton from start {k}")
    cands = []
    for k, u0 in enumerate(u_starts):
        u1, _ = sp.lbfgs_map(post, u0, maxiter=100)
        u2, info = sp.newton_map(post, u1, max_iter=25)
        cands.append((u2, info, f"L-BFGS then Newton from start {k}"))
    ok = [c for c in cands if c[1]["converged"]] or cands + [best]
    return min(ok, key=lambda c: c[1]["U"])


def run_unit(args, label, g_true, snr, seed, ctx):
    """One (truth, model, SNR, seed) unit: fit, flags, JSON record (skipped if it exists)."""
    model_name = args.model
    path = Path(args.out_dir) / f"{unit_label(label, model_name, snr, seed)}.json"
    if path.exists() and not args.force:
        print(f"{path.stem}: done")
        return json.loads(path.read_text())
    t0 = time.perf_counter()
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.results) / "data"
    z = np.load(data_dir / f"{label}.npz")
    d_clean = jnp.asarray(z["d_clean"])
    s = noise_std_from_snr(d_clean, snr)
    d_obs = add_noise(jax.random.key(seed), d_clean, s)
    n_data = int(np.asarray(d_obs).size)
    noise_only = float(np.sum(np.abs(np.asarray(d_obs) - np.asarray(d_clean)) ** 2) / s**2)
    model = sm.make_model(model_name)
    post = sp.ShapePosterior(model, ctx["forward"], d_obs, s, ctx["renderer"])
    starts = []
    map30 = Path(args.results) / f"{label}_{model_name}" / "map.npz"
    if map30.exists():
        u30 = np.load(map30)["u_hat"]
        if u30[0] > -8.0:  # a MAP on the g -> 0 wall (T4) is no start for Gauss-Newton: it crawls along the flat direction
            starts.append(u30)
    starts.append(np.asarray(model.u_from_p(jnp.asarray(reference_p(model_name, g_true)))))
    u_hat, info, how = fit_unit(post, starts)
    fit = sp.laplace_fit(post, u_hat)
    p_hat = post.physical(u_hat)
    g_hat = float(p_hat[0])
    sd_g = float(fit.sd()[0]) * g_hat * (1.0 - g_hat / sm.G_MAX)
    flag = sp.misfit_p_value(info["phi"], n_data, model.dim)
    rec = {"label": label, "model": model_name, "snr": float(snr), "seed": int(seed), "g_true": float(g_true), "n_data": n_data, "noise_std": float(s), "phi_map": info["phi"],
           "phi_over_N": info["phi"] / n_data, "map_converged": info["converged"], "fit": how, "noise_only_phi": noise_only, "delta_phi": info["phi"] - noise_only,
           "p_null": flag["p_value"], "z_phi": flag["z"], "p_hat": p_hat, "g_hat": g_hat, "g_sd": sd_g, "g_ci": [max(g_hat - 1.645 * sd_g, 0.0), g_hat + 1.645 * sd_g],
           "g_error": g_hat - g_true, "z_true": (g_true - g_hat) / sd_g if sd_g > 0 else None, "log_z_laplace": sp.laplace_log_evidence(post, fit), "ppc_refit": None}
    rec["covers_true"] = bool(rec["g_ci"][0] <= g_true <= rec["g_ci"][1])
    if args.ppc_reps > 0:
        draws = u_hat + (np.linalg.cholesky(fit.cov) @ np.random.default_rng(args.ppc_seed).standard_normal((model.dim, args.ppc_reps))).T
        ppc = sp.predictive_check(post, draws, jax.random.key(args.ppc_seed), n_rep=args.ppc_reps, mode="refit", u_hat=u_hat)
        rec["ppc_refit"] = {k: ppc[k] for k in ("p_value", "t_obs", "t_rep_mean", "t_rep_sd", "z", "n_rep")}
    rec["time_s"] = time.perf_counter() - t0
    rec["thread_env"] = prun.thread_env()
    prun.write_json(path, rec)
    print(f"{path.stem}: Phi/N {rec['phi_over_N']:.4f}, dPhi {rec['delta_phi']:+.1f}, p(null) {rec['p_null']:.3g}; g = {g_hat:.5f} +- {sd_g:.5f} (true {g_true}, z {rec['z_true']:+.1f}); "
          f"{how}, {rec['time_s']:.0f} s")
    return rec


def noise_null_check(n_data=2400, n_keys=1000, first=40, out_dir=OUT):
    """Misfit of the noise alone, sum |e|^2 / s^2 for ``add_noise`` keys 0 .. ``n_keys`` - 1 on ``n_data`` entries, against its Gamma(``n_data``, 1) law: mean, sd, Kolmogorov-Smirnov p, the
    sd of the first ``first`` keys (the seeds of the calibration) and the value of key 0. Writes <out-dir>/noise_null_check.json."""
    from scipy import stats

    d = jnp.zeros((n_data,), dtype=jnp.complex128)
    keys = jax.vmap(jax.random.key)(jnp.arange(n_keys))
    phis = np.asarray(jax.vmap(lambda k: jnp.sum(jnp.abs(add_noise(k, d, 1.0)) ** 2))(keys))
    ks = stats.kstest(phis, stats.gamma(a=float(n_data)).cdf)
    out = {"n_data": n_data, "n_keys": n_keys, "mean": float(phis.mean()), "sd": float(phis.std(ddof=1)), "theory_mean": float(n_data), "theory_sd": float(np.sqrt(n_data)),
           "sd_of_sd": float(np.sqrt(n_data) / np.sqrt(2.0 * (n_keys - 1))), "ks_p_value": float(ks.pvalue), f"sd_first_{first}_keys": float(phis[:first].std(ddof=1)),
           "sd_of_sd_first": float(np.sqrt(n_data) / np.sqrt(2.0 * (first - 1))), "key0": float(phis[0]), "key0_z": float((phis[0] - n_data) / np.sqrt(n_data))}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prun.write_json(Path(out_dir) / "noise_null_check.json", out)
    print(f"noise-only misfit over {n_keys} keys: mean {out['mean']:.1f}, sd {out['sd']:.1f} (theory {out['theory_sd']:.1f}), KS p {out['ks_p_value']:.2f}; first {first} keys sd "
          f"{out[f'sd_first_{first}_keys']:.1f}; key 0 {out['key0']:.1f} (z {out['key0_z']:+.2f})")
    return out


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------


def load_units(out_dir):
    return [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*_snr*_s*.json"))]


def summarize(units, flag_p=0.05):
    """Per (truth, model, SNR): over the seeds of the units, the number of seeds, the mean and sd of the gap estimate, the mean Laplace sd, the mean |z| of the truth and the fraction of
    seeds with |z| > 3 (the interval excludes the truth at 3 sd), the mean excess dPhi over the noise-only misfit and its sd, the median analytic p, the fraction of seeds flagged by
    the misfit (p < ``flag_p``, and p < 0.00135 = the one-sided 3 sd level) and c = mean dPhi / mean z^2 (None unless the mean |z| is above 3)."""
    out = []
    keys = sorted({(u["label"], u["model"], u["snr"]) for u in units}, key=lambda k: (k[0], k[1], -k[2]))
    for label, model, snr in keys:
        grp = sorted([u for u in units if (u["label"], u["model"], u["snr"]) == (label, model, snr)], key=lambda u: u["seed"])
        z = np.array([u["z_true"] for u in grp])
        dphi = np.array([u["delta_phi"] for u in grp])
        p = np.array([u["p_null"] for u in grp])
        g = np.array([u["g_hat"] for u in grp])
        m = len(grp)
        row = {"label": label, "model": model, "snr": snr, "g_true": grp[0]["g_true"], "n_seeds": m, "seeds": [u["seed"] for u in grp], "g_mean": float(g.mean()),
               "g_sd_across_seeds": float(g.std(ddof=1)) if m > 1 else None, "g_sd_laplace": float(np.mean([u["g_sd"] for u in grp])), "z_mean": float(z.mean()),
               "z_abs_mean": float(np.abs(z).mean()), "frac_interval_excludes_3sd": float(np.mean(np.abs(z) > 3.0)), "delta_phi_mean": float(dphi.mean()),
               "delta_phi_sd": float(dphi.std(ddof=1)) if m > 1 else None, "phi_over_N_mean": float(np.mean([u["phi_over_N"] for u in grp])), "p_null_median": float(np.median(p)),
               "frac_flagged_p05": float(np.mean(p < flag_p)), "frac_flagged_3sd": float(np.mean(p < 0.00135)), "z_phi_mean": float(np.mean([u["z_phi"] for u in grp])),
               "c": float(dphi.mean() / np.mean(z**2)) if np.abs(z).mean() > 3.0 else None, "all_converged": bool(all(u["map_converged"] for u in grp)),
               "ppc_refit": [{"seed": u["seed"], **u["ppc_refit"]} for u in grp if u.get("ppc_refit")]}
        out.append(row)
    return out


def render_markdown(units, summary):
    fm = lambda x, spec: "-" if x is None else format(x, spec)  # noqa: E731
    parts = ["# F2 / F2u fits of the data of each truth at 10, 20 and 30 dB (DESIGN 14, question iii)\n",
             "Laplace fits of the n = 64 model (``scripts/parametric_snr.py``). z = (g_true - g_hat) / sd_Laplace. dPhi = Phi_MAP minus the misfit of the noise realization itself. "
             "p is P(Gamma(N - dim/2) >= Phi_MAP). c = dPhi / z^2 where the mean |z| is above 3. Under a correct model dPhi is about -dim/2 (-2.5 for F2) and p is uniform.\n"]
    one = [u for u in units if u["seed"] == 0]
    parts.append("## One realization (noise key 0, the realization of the main experiment, scaled to the SNR)\n")
    header = ["truth", "model", "SNR (dB)", "g true", "g hat", "sd", "z of truth", "Phi_MAP/N", "dPhi", "z of Phi", "p (null)", "c = dPhi / z^2", "refit p (reps)", "converged"]
    rows = []
    for u in sorted(one, key=lambda u: (u["label"], u["model"], -u["snr"])):
        z = u["z_true"]
        c = u["delta_phi"] / z**2 if z is not None and abs(z) > 3.0 else None
        pp = u.get("ppc_refit")
        rows.append([u["label"], u["model"], f"{u['snr']:g}", f"{u['g_true']:g}", f"{u['g_hat']:.5f}", f"{u['g_sd']:.2e}", fm(z, "+.1f"), f"{u['phi_over_N']:.4f}", f"{u['delta_phi']:+.1f}",
                     f"{u['z_phi']:+.1f}", f"{u['p_null']:.3g}", fm(c, ".2f"), "-" if not pp else f"{pp['p_value']:.3f} ({pp['n_rep']}; z {pp['z']:.1f})", "yes" if u["map_converged"] else "no"])
    parts.append("| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    multi = [r for r in summary if r["n_seeds"] > 1]
    if multi:
        parts.append("\n## Typical draws: several noise realizations per SNR\n")
        header = ["truth", "model", "SNR (dB)", "seeds", "g hat mean (sd across seeds)", "sd Laplace", "mean z of truth", "interval excludes truth at 3 sd", "mean dPhi (sd)", "median p (null)",
                  "flagged p < 0.05", "flagged at 3 sd (p < 0.00135)", "c = mean dPhi / mean z^2"]
        rows = [[r["label"], r["model"], f"{r['snr']:g}", str(r["n_seeds"]), f"{r['g_mean']:.5f} ({r['g_sd_across_seeds']:.1e})", f"{r['g_sd_laplace']:.2e}", f"{r['z_mean']:+.1f}",
                 f"{r['frac_interval_excludes_3sd']:.2f}", f"{r['delta_phi_mean']:+.1f} ({r['delta_phi_sd']:.1f})", f"{r['p_null_median']:.3g}", f"{r['frac_flagged_p05']:.2f}",
                 f"{r['frac_flagged_3sd']:.2f}", fm(r["c"], ".2f")] for r in multi]
        parts.append("| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows))
    return "\n".join(parts) + "\n"


def collect(args):
    units = load_units(args.out_dir)
    if not units:
        raise SystemExit(f"no unit files in {args.out_dir}")
    summary = summarize(units)
    out = Path(args.out_dir)
    (out / "snr_summary.md").write_text(render_markdown(units, summary))
    (out / "snr_summary.json").write_text(json.dumps({"units": units, "summary": summary}, indent=1, default=prun._default))
    print(f"wrote {out / 'snr_summary.md'} ({len(units)} units, {len(summary)} groups)")
    return {"units": units, "summary": summary}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", help="T1..T5 (``parametric_run.truth_label`` names the data file)")
    ap.add_argument("--gap", type=float, default=0.1, help="the true edge-to-edge gap (T4: ignored, 0)")
    ap.add_argument("--label", help="instead of --truth: the data file <data-dir>/<label>.npz (e.g. E0.01g0.1, a rung of the ellipticity ladder); --gap is then its true gap")
    ap.add_argument("--data-dir", default=None, help="where the data files are (default <results>/data; the ladder's are in results/parametric/ellipticity/data)")
    ap.add_argument("--model", default="F2", choices=("F2", "F2u"))
    ap.add_argument("--snr", type=float, nargs="+", default=[10.0, 20.0], help="SNRs in dB (30 reproduces the cached data for seed 0)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ppc-reps", type=int, default=0)
    ap.add_argument("--ppc-seed", type=int, default=3)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--noise-check", action="store_true", help="only check the noise generator against Gamma(N, 1) over 1000 keys -> noise_null_check.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--results", default=str(RESULTS), help="where the data/ and <case>/map.npz files of the main experiment are")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if not (args.collect or args.noise_check) and args.truth is None and args.label is None:
        ap.error("--truth or --label is required (or --collect or --noise-check)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.collect:
        return collect(args)
    if args.noise_check:
        return noise_null_check(out_dir=args.out_dir)
    label = args.label or prun.truth_label(args.truth, args.gap)
    g_true = 0.0 if args.truth == "T4" else args.gap
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    ctx = {"forward": Forward(setup, tol=args.tol), "renderer": el.EllipseRenderer(setup.grid)}
    for snr in args.snr:
        for seed in args.seeds:
            run_unit(args, label, g_true, snr, seed, ctx)


if __name__ == "__main__":
    main()
