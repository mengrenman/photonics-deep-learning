"""Milestone 7 (DESIGN 14, question i): the offset of the right-family gap interval is a deterministic forward-model (grid) bias, not a noise effect.

    python scripts/parametric_discretization.py fit --gap 0.05 0.1 --data-n 64 128 256     # noise-free fits (units: <out-dir>/disc_g<gap>_n<n>.json)
    python scripts/parametric_discretization.py analyze                                    # offset-corrected calibration and paired seeds -> discretization.json / .md

Finding being tested. The F2 interval of the gap for T1 (two equal disks, the right family) is calibrated in width over 41 noise realizations (sd of z 1.00) but offset in
location (mean z -1.15, coverage 66 %; ``ellipticity/calibration_g*.md``). The data are generated on the n = 128 grid and inverted on the n = 64 grid with exact-coverage
rendering of the disks on both, so the forward operators differ (their difference is the "grid error": 0.061 of the noise at the truth).

``fit``: F2 (the n = 64 model) fitted to NOISE-FREE data of T1 generated on the n = ``--data-n`` grid, with the noise level of the 30 dB data of the main experiment (so the
shift is in the same posterior-sd units). The pseudo-true gap g* minus the true gap, in Laplace sd, is the deterministic location offset of the interval; ``phi_map`` is the misfit
the n = 64 family cannot absorb, and ``model_error_energy_at_truth`` the misfit of the true parameters (the grid error before the fit absorbs part of it into g, r, eps_r). With data
generated on the inference grid itself (n = 64) the offset must vanish (the inverse crime); n = 128 is the data of the experiment; n = 256 shows the trend (``extrapolated_sd``:
geometric extrapolation of the successive shifts to the continuum).

``analyze``: from the seed files of ``ellipticity/`` (41 seeds per gap, data n = 128) and ``ellipticity_crime/`` (8 seeds, data on n = 64, the same noise keys) -- no new fits. The
paired difference of z between the two data grids (same noise key, so the noise cancels), the calibration after removing the deterministic offset (z + offset / sd, coverage of
the shifted interval), the correlation of z between the two gaps (the same 41 noise realizations: the two HMC cases of the main experiment are one realization, noise key 0), and the
offset in sd at other SNRs (it is fixed in wavelengths, so it scales as 10^((SNR - 30) / 20)).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import parametric_run as prun
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import Grid, Setup, ch15_setup

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parametric"
OUT = RESULTS / "discretization"
print = functools.partial(print, flush=True)


def unit_path(out_dir, gap, n):
    return Path(out_dir) / f"disc_g{gap:g}_n{n}.json"


def fit_unit(args, gap, n_data, ctx):
    """Noise-free F2 fit to T1 data generated on the n = ``n_data`` grid; writes and returns the record (skipped if it exists)."""
    path = unit_path(args.out_dir, gap, n_data)
    if path.exists() and not args.force:
        print(f"{path.stem}: done")
        return json.loads(path.read_text())
    t0 = time.perf_counter()
    label = prun.truth_label("T1", gap)
    z = np.load(Path(args.results) / "data" / f"{label}.npz")
    s = float(z["s"])
    truth = sh.TwoDisks(g=gap, r=0.2, eps_r=2.0)
    setup = ctx["setup"]
    data_setup = setup if n_data == args.n else Setup(Grid(n_data, args.side), setup.antennas)
    d_clean = np.asarray(Forward(data_setup, tol=1e-10)(jnp.asarray(truth.chi(data_setup.grid)))[0])
    check = None
    if n_data == 2 * args.n:
        check = float(np.abs(d_clean - z["d_clean"]).max() / np.abs(z["d_clean"]).max())  # reproduces the cached noise-free data of the experiment
    post = sp.ShapePosterior(sm.F2(), ctx["forward"], d_clean, s, ctx["renderer"])
    u_start = np.load(Path(args.results) / f"{label}_F2" / "map.npz")["u_hat"]
    u1, _ = sp.lbfgs_map(post, u_start, maxiter=100)
    u, info = sp.newton_map(post, u1, max_iter=25, tol=1e-6)
    fit = sp.laplace_fit(post, u)
    p = post.physical(u)
    g_star = float(p[0])
    sd = float(fit.sd()[0]) * g_star * (1.0 - g_star / sm.G_MAX)
    p_true = np.array([gap, 0.2, 2.0, 0.0, 0.0])
    resid = d_clean - np.asarray(post.clean_data(post.model.u_from_p(jnp.asarray(p_true))))
    rec = {"gap": gap, "data_n": n_data, "inference_n": args.n, "noise_std": s, "g_star": g_star, "g_star_minus_true": g_star - gap, "laplace_sd": sd, "shift_sd": (g_star - gap) / sd,
           "phi_map": info["phi"], "model_error_energy_at_truth": float(np.sum(np.abs(resid) ** 2) / s**2), "p_star": p.tolist(), "converged": info["converged"],
           "cached_clean_data_max_rel_diff": check, "time_s": time.perf_counter() - t0, "thread_env": prun.thread_env()}
    prun.write_json(path, rec)
    print(f"{path.stem}: g* - g = {rec['g_star_minus_true']:+.3e} = {rec['shift_sd']:+.3f} sd (sd {sd:.3e}); Phi_MAP {info['phi']:.3f}, model-error energy at the truth "
          f"{rec['model_error_energy_at_truth']:.3f}; {rec['time_s']:.0f} s")
    return rec


def extrapolate_shift(shifts):
    """Geometric extrapolation of the shifts at successive grid doublings (n, 2n, 4n) to the continuum: s_inf = s_3 + (s_3 - s_2) r / (1 - r), r = (s_3 - s_2) / (s_2 - s_1)."""
    s1, s2, s3 = shifts
    r = (s3 - s2) / (s2 - s1)
    return float(s3 + (s3 - s2) * r / (1.0 - r)), float(r)


def load_seed_files(directory, gap):
    recs = {}
    for p in Path(directory).glob(f"E0g{gap:g}*.json"):
        r = json.loads(p.read_text())
        if "level" in r and r["level"] == 0.0 and abs(r["g_true"] - gap) < 1e-12:
            recs[r["seed"]] = r
    return recs


def analyze(args):
    out_dir = Path(args.out_dir)
    result = {"gaps": {}}
    lines = ["# The offset of the right-family gap interval is a forward-model (grid) bias\n"]
    ell, crime = Path(args.results) / "ellipticity", Path(args.results) / "ellipticity_crime"
    units = {g: {n: json.loads(unit_path(out_dir, g, n).read_text()) for n in (64, 128, 256) if unit_path(out_dir, g, n).exists()} for g in (0.05, 0.1)}
    lines.append("## Noise-free F2 fits to T1 data of different grids (inference on n = 64, noise level of the 30 dB data)\n")
    lines.append("| gap | data grid n | g* - g | shift in Laplace sd | Phi_MAP | model-error energy at the truth | extrapolated shift (sd) |\n|---|---|---|---|---|---|---|")
    for gap, per in units.items():
        entry = {"units": per}
        if len(per) == 3:
            ext, ratio = extrapolate_shift([per[n]["shift_sd"] for n in (64, 128, 256)])
            entry["extrapolated_sd"], entry["successive_shift_ratio"] = ext, ratio
        for n, u in sorted(per.items()):
            ext_txt = f"{entry['extrapolated_sd']:.2f}" if n == 256 and "extrapolated_sd" in entry else ""
            lines.append(f"| {gap:g} | {n} | {u['g_star_minus_true']:+.3e} | {u['shift_sd']:+.3f} | {u['phi_map']:.3f} | {u['model_error_energy_at_truth']:.2f} | {ext_txt} |")
        result["gaps"][f"{gap:g}"] = entry
    lines.append("")
    seeds_main = {g: load_seed_files(ell, g) for g in (0.05, 0.1)}
    crime_main = load_seed_files(crime, 0.05)
    # (1) paired difference: same noise key, data n = 128 against data n = 64
    common = sorted(set(crime_main) & set(seeds_main[0.05]) - {0})
    if common:
        dz = np.array([seeds_main[0.05][k]["z_true"] - crime_main[k]["z_true"] for k in common])
        dg = np.array([seeds_main[0.05][k]["g_hat"] - crime_main[k]["g_hat"] for k in common])
        result["paired_crime"] = {"seeds": common, "dz_mean": float(dz.mean()), "dz_sd": float(dz.std(ddof=1)), "dg_mean": float(dg.mean()), "dg_sd": float(dg.std(ddof=1))}
        lines.append(f"## Paired seeds (same noise key, data n = 128 against data n = 64, g = 0.05, {len(common)} seeds)\n")
        lines.append(f"z(n = 128) - z(n = 64) = {dz.mean():+.3f} +- {dz.std(ddof=1):.3f} (sd over seeds; the values are {', '.join(f'{x:+.3f}' for x in dz)}); the gap estimate shifts by "
                     f"{dg.mean():+.3e} +- {dg.std(ddof=1):.1e}. The noise cancels in the pairing: a constant shift is the signature of a deterministic bias.\n")
    # (2) offset-corrected calibration
    lines.append("## Calibration of the 90 % interval over noise seeds, as run and with the deterministic offset removed (data n = 128)\n")
    lines.append("| gap | seeds | mean z | sd z | coverage | offset removed (sd units, seed-wise) | mean z corrected (se) | sd z corrected | coverage corrected |\n|---|---|---|---|---|---|---|---|---|")
    for gap in (0.05, 0.1):
        recs = seeds_main[gap]
        if not recs or 128 not in units[gap]:
            continue
        off_g = units[gap][128]["g_star_minus_true"]
        zs = np.array([recs[k]["z_true"] for k in sorted(recs)])
        sds = np.array([recs[k]["g_sd"] for k in sorted(recs)])
        zc = zs + off_g / sds
        cov, covc = float(np.mean(np.abs(zs) <= 1.645)), float(np.mean(np.abs(zc) <= 1.645))
        n = len(zs)
        result["gaps"][f"{gap:g}"]["calibration"] = {"n_seeds": n, "offset_g": off_g, "z_mean": float(zs.mean()), "z_sd": float(zs.std(ddof=1)), "coverage": cov,
                                                     "z_corrected_mean": float(zc.mean()), "z_corrected_mean_se": float(zc.std(ddof=1) / np.sqrt(n)), "z_corrected_sd": float(zc.std(ddof=1)),
                                                     "coverage_corrected": covc}
        lines.append(f"| {gap:g} | {n} | {zs.mean():+.2f} | {zs.std(ddof=1):.2f} | {cov:.2f} | {off_g / sds.mean():+.2f} | {zc.mean():+.2f} ({zc.std(ddof=1) / np.sqrt(n):.2f}) | "
                     f"{zc.std(ddof=1):.2f} | {covc:.2f} |")
    # (3) the two gaps are one set of noise realizations
    both = sorted(set(seeds_main[0.05]) & set(seeds_main[0.1]))
    if len(both) > 3:
        za = np.array([seeds_main[0.05][k]["z_true"] for k in both])
        zb = np.array([seeds_main[0.1][k]["z_true"] for k in both])
        result["z_correlation_between_gaps"] = {"n_seeds": len(both), "correlation": float(np.corrcoef(za, zb)[0, 1])}
        lines.append(f"\nCorrelation of z between g = 0.05 and g = 0.1 over the {len(both)} common noise keys: {np.corrcoef(za, zb)[0, 1]:.3f}. The two HMC cases of the main experiment "
                     "(both noise key 0) are therefore ONE data realization; the calibration over seeds carries the statement.\n")
    # (4) the offset at other SNRs
    if 128 in units[0.05]:
        sd30 = units[0.05][128]["laplace_sd"]
        off = units[0.05][128]["g_star_minus_true"]
        rows = []
        for snr in (10, 20, 30, 40, 50):
            rows.append({"snr": snr, "sd": sd30 * 10 ** ((30 - snr) / 20), "offset_sd": off / (sd30 * 10 ** ((30 - snr) / 20))})
        result["offset_vs_snr_g0.05"] = rows
        lines.append("\n## The offset at other SNRs (g = 0.05; fixed in wavelengths, so in sd it scales as 10^((SNR - 30) / 20))\n")
        lines.append("| SNR (dB) | sd of g (Laplace) | offset (sd) |\n|---|---|---|")
        lines += [f"| {r['snr']} | {r['sd']:.2e} | {r['offset_sd']:.2f} |" for r in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "discretization.md").write_text("\n".join(lines) + "\n")
    (out_dir / "discretization.json").write_text(json.dumps(result, indent=1, default=prun._default))
    print(f"wrote {out_dir / 'discretization.md'}")
    return result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("fit", "analyze"))
    ap.add_argument("--gap", type=float, nargs="+", default=[0.05, 0.1])
    ap.add_argument("--data-n", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--n", type=int, default=64, help="inference grid")
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.action == "analyze":
        return analyze(args)
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    ctx = {"setup": setup, "forward": Forward(setup, tol=args.tol), "renderer": el.EllipseRenderer(setup.grid)}
    for gap in args.gap:
        for n in args.data_n:
            fit_unit(args, gap, n, ctx)


if __name__ == "__main__":
    main()
