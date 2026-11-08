"""T3 with the observed-information sd (milestone 8 correction of the milestone-7 SNR experiment, DESIGN 15.5).

    python scripts/parametric_t3_fd_sd.py --shard 0 --n-shards 4      # recompute one quarter of the units (resumable)
    python scripts/parametric_t3_fd_sd.py --collect                   # tables -> results/parametric/snr_fd/t3_fd_summary.{json,md}

Why. ``parametric_snr.py`` reports the Laplace sd of the gap from the Gauss-Newton (Fisher) Hessian. For T3 (unequal contrasts fitted by
F2) the misfit is large (Phi/N 15 to 17 at 30 dB) and the residual-curvature term that Gauss-Newton drops is not small: at the 30 dB
MAP the observed-information (finite-difference) sd of g is 16 % larger at g = 0.1 and 4 % larger at g = 0.05 (DESIGN 15.5,
``results/DETECTABILITY_REPORT.md`` section 3). The deterministic part of that term, (D^T D)^-1 <r, d2 mu> with the model-error residual
r, does not depend on the SNR (both factors scale as 1 / s^2), so every T3 unit of the SNR experiment is affected; the noise part of the
residual adds a key-dependent term that grows relative to it as s (at 20 and 10 dB the ratio of the two sd varies from about 0.8 to 1.5
over the 40 keys). This script refits
nothing: for each stored T3 F2 unit (``results/parametric/snr/T3g*_F2_snr*_s*.json``) it rebuilds the data of that unit exactly as
``parametric_snr.run_unit`` does (same clean data, ``noise_std_from_snr``, ``add_noise`` with ``jax.random.key(seed)``), takes the stored
MAP ``p_hat``, and evaluates ``shapepost.laplace_fit(hessian="fd")`` there. It writes <out-dir>/<unit>.json with the Gauss-Newton and
finite-difference sd of g, z = (g_true - g_hat) / sd for both, and c = dPhi / z^2 (dPhi as stored).
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
SNR_DIR = RESULTS / "snr"
OUT = RESULTS / "snr_fd"
print = functools.partial(print, flush=True)


def unit_files(snr_dir=SNR_DIR):
    return sorted(Path(snr_dir).glob("T3g*_F2_snr*_s*.json"))


def gap_sd(fit, g_hat):
    """sd of g from the sd of the logit coordinate u_0 (as in ``parametric_snr.run_unit``)."""
    return float(fit.sd()[0]) * g_hat * (1.0 - g_hat / sm.G_MAX)


def run_unit(path, ctx, out_dir, force=False):
    rec = json.loads(Path(path).read_text())
    out = Path(out_dir) / Path(path).name
    if out.exists() and not force:
        print(f"{out.stem}: done")
        return json.loads(out.read_text())
    t0 = time.perf_counter()
    z = np.load(RESULTS / "data" / f"{rec['label']}.npz")
    d_clean = jnp.asarray(z["d_clean"])
    s = noise_std_from_snr(d_clean, rec["snr"])
    if not np.isclose(float(s), rec["noise_std"], rtol=1e-12):
        raise RuntimeError(f"{path.name}: rebuilt noise sd {float(s)} differs from the stored {rec['noise_std']}")
    d_obs = add_noise(jax.random.key(rec["seed"]), d_clean, s)
    noise_only = float(np.sum(np.abs(np.asarray(d_obs) - np.asarray(d_clean)) ** 2) / s**2)
    if not np.isclose(noise_only, rec["noise_only_phi"], rtol=1e-10):
        raise RuntimeError(f"{path.name}: rebuilt noise-only misfit {noise_only} differs from the stored {rec['noise_only_phi']}")
    model = sm.make_model("F2")
    post = sp.ShapePosterior(model, ctx["forward"], d_obs, s, ctx["renderer"])
    u_hat = np.asarray(model.u_from_p(jnp.asarray(rec["p_hat"], dtype=jnp.float64)))
    gn = sp.laplace_fit(post, u_hat, hessian="gn")
    fd = sp.laplace_fit(post, u_hat, hessian="fd")
    g_hat, g_true = rec["g_hat"], rec["g_true"]
    sd_gn, sd_fd = gap_sd(gn, g_hat), gap_sd(fd, g_hat)
    z_gn, z_fd = (g_true - g_hat) / sd_gn, (g_true - g_hat) / sd_fd
    res = {"unit": out.stem, "label": rec["label"], "snr": rec["snr"], "seed": rec["seed"], "g_true": g_true, "g_hat": g_hat,
           "delta_phi": rec["delta_phi"], "phi_map_stored": rec["phi_map"], "phi_at_p_hat": gn.phi,
           "g_sd_stored": rec["g_sd"], "z_stored": rec["z_true"], "g_sd_gn": sd_gn, "g_sd_fd": sd_fd, "sd_ratio_fd_gn": sd_fd / sd_gn,
           "z_gn": z_gn, "z_fd": z_fd, "c_gn": rec["delta_phi"] / z_gn**2, "c_fd": rec["delta_phi"] / z_fd**2,
           "time_s": time.perf_counter() - t0, "thread_env": prun.thread_env()}
    if not np.isclose(sd_gn, rec["g_sd"], rtol=1e-6):
        raise RuntimeError(f"{path.name}: rebuilt Gauss-Newton sd {sd_gn} differs from the stored {rec['g_sd']} (the MAP or the data are not the stored ones)")
    prun.write_json(out, res)
    print(f"{out.stem}: sd GN {sd_gn:.4e} -> FD {sd_fd:.4e} (x{sd_fd / sd_gn:.4f}); z {z_gn:+.2f} -> {z_fd:+.2f}; c {res['c_gn']:.2f} -> {res['c_fd']:.2f}; {res['time_s']:.0f} s")
    return res


def summarize(rows):
    out = []
    for key in sorted({(r["label"], r["snr"]) for r in rows}, key=lambda k: (k[0], -k[1])):
        grp = sorted([r for r in rows if (r["label"], r["snr"]) == key], key=lambda r: r["seed"])
        zg, zf = np.array([r["z_gn"] for r in grp]), np.array([r["z_fd"] for r in grp])
        dphi = np.array([r["delta_phi"] for r in grp])
        ratio = np.array([r["sd_ratio_fd_gn"] for r in grp])
        k0 = next((r for r in grp if r["seed"] == 0), None)
        out.append({"label": key[0], "snr": key[1], "n_seeds": len(grp), "sd_ratio_mean": float(ratio.mean()), "sd_ratio_min": float(ratio.min()),
                    "sd_ratio_max": float(ratio.max()), "z_mean_gn": float(zg.mean()), "z_mean_fd": float(zf.mean()),
                    "frac_excludes_3sd_gn": float(np.mean(np.abs(zg) > 3.0)), "frac_excludes_3sd_fd": float(np.mean(np.abs(zf) > 3.0)),
                    "n_excludes_3sd_gn": int(np.sum(np.abs(zg) > 3.0)), "n_excludes_3sd_fd": int(np.sum(np.abs(zf) > 3.0)),
                    "c_mean_gn": float(dphi.mean() / np.mean(zg**2)) if np.abs(zg).mean() > 3.0 else None,
                    "c_mean_fd": float(dphi.mean() / np.mean(zf**2)) if np.abs(zf).mean() > 3.0 else None,
                    "key0": None if k0 is None else {k: k0[k] for k in ("g_sd_gn", "g_sd_fd", "sd_ratio_fd_gn", "z_gn", "z_fd", "c_gn", "c_fd", "delta_phi")}})
    return out


def render(summary):
    lines = ["# T3 under F2: Gauss-Newton against observed-information (FD) sd of the gap", "",
             "From `scripts/parametric_t3_fd_sd.py` (one file per unit in this directory). z = (g_true - g_hat) / sd; c = dPhi / z^2 with dPhi as",
             "stored by `parametric_snr.py`; the MAPs are the stored ones (nothing is refit). Key 0 is the milestone-7 realization.", "",
             "| truth | SNR | seeds | sd FD / GN (mean, range) | key 0: z GN -> FD | key 0: c GN -> FD | mean z GN -> FD | excludes at 3 sd GN -> FD |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        k = r["key0"] or {}
        z0 = f"{k['z_gn']:+.2f} -> {k['z_fd']:+.2f}" if k else "-"
        c0 = f"{k['c_gn']:.1f} -> {k['c_fd']:.1f}" if k else "-"
        lines.append(f"| {r['label']} | {r['snr']:g} dB | {r['n_seeds']} | {r['sd_ratio_mean']:.4f} ({r['sd_ratio_min']:.4f} to {r['sd_ratio_max']:.4f}) | {z0} | {c0} | "
                     f"{r['z_mean_gn']:+.2f} -> {r['z_mean_fd']:+.2f} | {r['n_excludes_3sd_gn']}/{r['n_seeds']} -> {r['n_excludes_3sd_fd']}/{r['n_seeds']} |")
    return "\n".join(lines) + "\n"


def collect(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("T3g*_F2_snr*_s*.json"))]
    summary = summarize(rows)
    prun.write_json(Path(out_dir) / "t3_fd_summary.json", summary)
    (Path(out_dir) / "t3_fd_summary.md").write_text(render(summary))
    print(render(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.collect:
        return collect(args.out_dir)
    setup = ch15_setup(n=64, side=1.5, n_tx=40, n_rx=60)  # the milestone-7 inference setting (parametric_snr defaults)
    ctx = {"forward": Forward(setup, tol=1e-10), "renderer": el.EllipseRenderer(setup.grid)}
    files = unit_files()
    for i, path in enumerate(files):
        if i % args.n_shards == args.shard:
            run_unit(path, ctx, args.out_dir, force=args.force)


if __name__ == "__main__":
    main()
