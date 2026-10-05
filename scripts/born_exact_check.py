"""Born end-to-end check at a larger size (docs/DESIGN.md section 8, step 5).

Builds the Born inverse problem of ``bayes_scatter.validation`` (book antenna rings, Matern prior on chi,
identity link, noise at a low per-datum SNR), computes the exact Gaussian posterior with
``born_gaussian_posterior``, runs pCN, pCN-Langevin, MALA and HMC from prior draws through the same
``Target`` that nonlinear runs use, and prints for each sampler

* standardized errors of the pixel means and variances against the exact posterior (RMS and max |z|;
  expected near 1 and a few for a correct sampler, pixel z-values being correlated),
* variance ratios, the posterior mean of Phi against its exact value, acceptance, split R-hat,
* the minimum effective sample size over pixels and ESS per second.

It saves a figure (estimated against exact mean and std per pixel, histogram of z) and summary.json to
``results/born_exact/``.

Run:  python scripts/born_exact_check.py [--n 16 --side 2.0 --snr -20 --scale 1.0]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter.plotting import plot_born_check
from bayes_scatter.validation import DEFAULT_SETTINGS, make_born_problem, run_born_check

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=16, help="pixels per side of the inference grid")
    ap.add_argument("--side", type=float, default=2.0, help="DOI side in wavelengths")
    ap.add_argument("--snr", type=float, default=-20.0, help="power SNR per datum in dB (negative on purpose, see "
                    "bayes_scatter.validation)")
    ap.add_argument("--sigma", type=float, default=0.3, help="prior standard deviation of chi")
    ap.add_argument("--ell", type=float, default=0.3, help="prior correlation length in wavelengths")
    ap.add_argument("--nu", type=float, default=1.5, help="Matern smoothness")
    ap.add_argument("--prior", choices=["matern", "gmrf"], default="matern")
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--scale", type=float, default=2.5, help="multiplier of the default chain lengths")
    ap.add_argument("--samplers", nargs="+", default=list(DEFAULT_SETTINGS), choices=list(DEFAULT_SETTINGS))
    ap.add_argument("--seed", type=int, default=0, help="seed of the data noise; chains use seed + 1")
    ap.add_argument("--out", default="results/born_exact", help="output directory (relative to the project)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    prob = make_born_problem(n=args.n, side=args.side, snr_db=args.snr, sigma=args.sigma, ell=args.ell,
                             nu=args.nu, kind=args.prior, n_tx=args.n_tx, n_rx=args.n_rx, seed=args.seed)
    ev = np.linalg.eigvalsh(prob.exact.precision_xi)
    std = np.sqrt(np.diag(prob.exact.cov_chi))
    print(f"Born problem: n={args.n} side={args.side} wavelengths, {args.n_tx} tx x {args.n_rx} rx, "
          f"SNR {args.snr:g} dB per datum (s={prob.s:.3g}), {args.prior} prior sigma={args.sigma} ell={args.ell}")
    print(f"  xi has {prob.exact.mean_xi.size} components; posterior precision (prior = 1): max {ev.max():.1f}, "
          f"{int(np.sum(ev > 1.5))} directions above 1.5, {int(np.sum(ev > 10))} above 10")
    print(f"  exact posterior std of chi per pixel: {std.min():.3f} to {std.max():.3f} (prior {args.sigma}); "
          f"exact E[Phi] = {prob.phi_mean_exact:.2f}; setup {time.perf_counter() - t0:.1f} s")

    settings = {k: dict(v, n_samples=int(args.scale * v["n_samples"]), n_warmup=int(args.scale * v["n_warmup"]))
                for k, v in DEFAULT_SETTINGS.items() if k in args.samplers}
    runs = run_born_check(prob, settings, n_chains=args.chains, seed=args.seed + 1)

    header = (f"{'sampler':13s} {'accept':>6s} {'step':>8s} | {'z_mean':>11s} | {'z_var':>11s} | "
              f"{'var ratio (mean [min,max])':>27s} | {'Phi z':>6s} | {'min ESS':>7s} {'ESS/s':>7s} {'wall s':>6s} "
              f"{'Rhat':>5s}")
    print("\n" + header)
    print(f"{'':13s} {'':6s} {'':8s} | {'rms / max':>11s} | {'rms / max':>11s} |")
    print("-" * len(header))
    summary = {"args": vars(args), "snr_db": args.snr, "s": prob.s, "phi_mean_exact": prob.phi_mean_exact,
               "precision_max": float(ev.max()), "samplers": {}}
    fig_results = {}
    for name, r in runs.items():
        st, info = r["stats"], r["info"]
        step = float(np.mean(np.asarray(r["result"].step)))
        print(f"{name:13s} {r['accept']:6.3f} {step:8.4f} | {st['z_mean_rms']:5.2f} /{st['z_mean_max']:5.2f} | "
              f"{st['z_var_rms']:5.2f} /{st['z_var_max']:5.2f} | "
              f"{st['var_ratio_mean']:6.3f} [{st['var_ratio_min']:5.3f}, {st['var_ratio_max']:5.3f}] | "
              f"{r['phi_z']:6.2f} | {st['ess_min']:7.0f} {r['ess_per_s']:7.1f} {info['wall_s']:6.1f} "
              f"{st['rhat_max']:5.3f}")
        samples = np.asarray(r["result"].samples)  # (chains, N, n, n)
        fig_results[name] = {
            "mean": samples.mean(axis=(0, 1)), "std": samples.reshape(-1, args.n, args.n).std(axis=0, ddof=1),
            "z": st["z_mean"],
        }
        summary["samplers"][name] = {
            "settings": settings[name], "accept": r["accept"], "step": step, "wall_s": info["wall_s"],
            "compile_s": info["compile_s"], "n_phi_evals_per_chain": info["n_phi_evals"],
            "phi_mean": r["phi_mean"], "phi_z": r["phi_z"], "ess_min": st["ess_min"],
            "ess_median": float(np.median(st["ess"])), "ess_per_s": r["ess_per_s"], "rhat_max": st["rhat_max"],
            **{k: st[k] for k in ("z_mean_rms", "z_mean_max", "z_var_rms", "z_var_max", "var_ratio_mean",
                                  "var_ratio_min", "var_ratio_max")},
        }
    print("\nExpected for a correct sampler: z rms near 1 (pixel z-values are correlated, so max |z| of 3 to 5 "
          "is normal), variance ratio near 1, Phi z within 3.")

    fig_path = plot_born_check(prob.exact.mean_chi, std, fig_results, out_dir / "born_exact.png",
                               title=f"Born exact check: n={args.n}, side {args.side}, SNR {args.snr:g} dB per datum")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nfigure: {fig_path}\nsummary: {out_dir / 'summary.json'}\ntotal {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
