"""Convergence audit of the prior-comparison runs from their stored draws: rank-normalized R-hat, multi-chain ESS, null spread.

    python scripts/convergence_audit.py [RUN ...] [--dir results/prior_comparison] [--null-reps 60] [--out FILE.json] [--update]

The DESIGN 11.2 criteria (classic split R-hat of Phi <= 1.01, max pixel R-hat <= 1.05, worst-pixel Sokal ESS >= 200) are
met by margins of the size of the noise at 300 to 600 draws per chain: with an integrated autocorrelation time of 3 to 4 a
classic R-hat above 1.01 happens by chance in 16 to 28 % of runs even when all chains sample one distribution. This script
reads chains.npz of every run (thinned theta draws and the physical Phi of the draws, chains x draws) and reports

  * the classic split R-hat of Phi and the max pixel R-hat (recomputed; they must equal the summary.json values),
  * the rank-normalized bulk and folded split R-hat (Vehtari et al. 2021, ``diagnostics.split_rhat_rank``) of Phi and of
    every pixel: the maximum over pixels, the number of pixels above 1.01 and 1.05, how many of those lie within one pixel
    of an edge (``features.edge_mask``) and their chain-to-chain std ratios (max over chains / min over chains of the
    within-chain std),
  * the worst-pixel effective sample size three ways: the sum of per-chain Sokal ESS (the DESIGN criterion), Geyer's, and
    the multi-chain rank-normalized bulk ESS (``diagnostics.ess_bulk``), which is small when chains have not mixed,
  * the null: the distribution of the same R-hat statistics for chains that sample ONE distribution, simulated as
    independent AR(1) series with the observed integrated autocorrelation times (``diagnostics.ar1_rhat_null``): for Phi
    (its own tau) and for the maximum over pixels (the observed per-pixel taus; pixels are treated as independent), with
    the probability that the null gives a statistic at least as large as the observed one.

Writes <dir>/convergence_audit.json (or --out). Single process, at most 2 BLAS threads.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
import time
from pathlib import Path

import numpy as np

import bayes_scatter  # noqa: F401
from bayes_scatter import diagnostics as dg
from bayes_scatter import features as ft

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ["pc_matern", "pc_gmrf", "pc_qggmrf_sg0.05", "pc_qggmrf_sg0.1", "pcs1_qggmrf_sg0.05", "pcg1_qggmrf_sg0.05",
                "pcg1_matern"]


def mean_tau(x):
    """Mean over chains of the Sokal integrated autocorrelation time of each component: x (chains, N, ...) -> (...)."""
    return np.mean([dg.tau_int(c, axis=0) for c in x], axis=0)


def rank_stats(x):
    """Rank-normalized split R-hat of x (chains, N, K) and the autocorrelation times of the statistics' series.

    Returns a dict of (K,) arrays: ``bulk`` and ``folded`` (``diagnostics.split_rhat_rank``), ``worst`` (their maximum)
    and ``tau_bulk``, ``tau_folded`` (mean over chains of the Sokal tau of the rank-normalized draws and of the
    rank-normalized absolute deviations from the pooled median, floored at 1): the null of each statistic needs the
    autocorrelation of the series it is computed from, and |x - median| is much more autocorrelated than x when the
    scale of a pixel wanders (non-Gaussian, volatility-like posteriors).
    """
    zb = dg.rank_normalize(x)
    zf = dg.rank_normalize(np.abs(x - np.median(x.reshape((-1,) + x.shape[2:]), axis=0)))
    bulk, folded = dg.split_rhat(zb), dg.split_rhat(zf)
    return {"bulk": bulk, "folded": folded, "worst": np.maximum(bulk, folded),
            "tau_bulk": np.maximum(mean_tau(zb), 1.0), "tau_folded": np.maximum(mean_tau(zf), 1.0)}


def null_worst(chains, n_draws, st, reps, seed):
    """Null distribution of max(bulk, folded) R-hat: AR(1) series with the tau of each statistic's series.

    ``st`` is ``rank_stats``. Returns ``(first, max_over_components)``, arrays over ``reps`` replicates: the worst of the
    two statistics for component 0 and its maximum over all components (the maximum of the pointwise maximum is the
    maximum of the two maxima)."""
    nb = dg.ar1_rhat_null(chains, n_draws, st["tau_bulk"], n_rep=reps, seed=seed, rank=False)
    nf = dg.ar1_rhat_null(chains, n_draws, st["tau_folded"], n_rep=reps, seed=seed + 7, rank=False)
    return np.maximum(nb["classic"], nf["classic"]), np.maximum(nb["classic_max"], nf["classic_max"])


def audit_run(name, base, null_reps, seed=0):
    """Audit one run directory ``base / name``; returns a JSON-serializable dict."""
    d = base / name
    summary = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    theta = z["theta"].astype(np.float64)  # (chains, N, n, n)
    chains, n_draws, n = theta.shape[0], theta.shape[1], theta.shape[2]
    flat = theta.reshape(chains, n_draws, -1)
    phi = z["phi_physical"].astype(np.float64)
    edge = ft.edge_mask(z["chi_true"]).ravel()
    t0 = time.time()

    classic_phi = float(dg.split_rhat(phi))
    classic_pix = dg.split_rhat(flat)
    st_phi = rank_stats(phi[:, :, None])
    st_pix = rank_stats(flat)
    rank_pix = st_pix["worst"]
    sd_chain = flat.std(axis=1, ddof=1)  # (chains, pixels)
    sd_ratio = sd_chain.max(axis=0) / sd_chain.min(axis=0)
    flagged = rank_pix > 1.05
    out = {"run": name, "chains": int(chains), "draws_per_chain": int(n_draws), "converged_design": bool(summary["convergence"]["converged"]),
           "summary_rhat_phi": summary["convergence"]["rhat_phi"], "summary_rhat_pixel_max": summary["convergence"]["rhat_pixel_max"],
           "classic": {"rhat_phi": classic_phi, "rhat_pixel_max": float(np.nanmax(classic_pix)),
                       "rhat_pixel_max_edge_pixels": float(np.nanmax(classic_pix[edge])),
                       "rhat_pixel_max_non_edge_pixels": float(np.nanmax(classic_pix[~edge]))},
           "rank_normalized": {"phi": {"bulk": float(st_phi["bulk"][0]), "folded": float(st_phi["folded"][0]), "worst": float(st_phi["worst"][0])},
                               "pixel_max_bulk": float(np.nanmax(st_pix["bulk"])), "pixel_max_folded": float(np.nanmax(st_pix["folded"])),
                               "pixel_max_worst": float(np.nanmax(rank_pix)),
                               "pixels_worst_above_1.01": int(np.sum(rank_pix > 1.01)),
                               "pixels_worst_above_1.05": int(flagged.sum()),
                               "pixels_worst_above_1.05_within_1px_of_edge": int(np.sum(flagged & edge)),
                               "edge_pixels": int(edge.sum()), "pixels": int(edge.size),
                               "chain_sd_ratio_max_over_pixels_above_1.05": float(sd_ratio[flagged].max()) if flagged.any() else None,
                               "chain_sd_ratio_median_over_pixels_above_1.05": float(np.median(sd_ratio[flagged])) if flagged.any() else None,
                               "chain_sd_ratio_median_all_pixels": float(np.median(sd_ratio)),
                               "tau_folded_median": float(np.median(st_pix["tau_folded"])), "tau_bulk_median": float(np.median(st_pix["tau_bulk"]))}}
    ess_sokal = np.sum([dg.ess(c, axis=0) for c in flat], axis=0)
    ess_geyer = np.sum([dg.ess_geyer(c, axis=0) for c in flat], axis=0)
    ess_b = dg.ess_bulk(flat)
    out["ess"] = {"worst_pixel_sokal_summed": float(np.nanmin(ess_sokal)), "worst_pixel_geyer_summed": float(np.nanmin(ess_geyer)),
                  "worst_pixel_bulk_multichain": float(np.nanmin(ess_b)), "median_pixel_bulk_multichain": float(np.nanmedian(ess_b)),
                  "phi_bulk_multichain": float(dg.ess_bulk(phi)), "phi_sokal_summed": float(sum(dg.ess(c) for c in phi)),
                  "total_draws": int(chains * n_draws), "design_threshold": 200.0}
    tau_phi = float(np.mean([dg.tau_int(c) for c in phi]))
    tau_pix = np.maximum(mean_tau(flat), 1.0)
    out["tau"] = {"phi": tau_phi, "pixel_median": float(np.median(tau_pix)), "pixel_max": float(tau_pix.max())}
    reps_pix = max(null_reps // 3, 10)
    nc_phi = dg.ar1_rhat_null(chains, n_draws, max(tau_phi, 1.0), n_rep=null_reps, seed=seed, rank=False)["classic"]
    nr_phi, _ = null_worst(chains, n_draws, {k: v[:1] for k, v in st_phi.items()}, null_reps, seed + 3)
    nc_pix = dg.ar1_rhat_null(chains, n_draws, tau_pix, n_rep=reps_pix, seed=seed + 1, rank=False)["classic_max"]
    _, nr_pix = null_worst(chains, n_draws, st_pix, reps_pix, seed + 5)
    q = lambda a: [float(v) for v in np.quantile(a, [0.5, 0.95])]
    obs_rank_pix = out["rank_normalized"]["pixel_max_worst"]
    out["null_ar1"] = {
        "note": ("chains drawn from ONE distribution: independent AR(1) series with the observed integrated autocorrelation "
                 "time of the series each statistic is computed from (classic: the draws; rank-normalized bulk and folded: the "
                 "rank-normalized draws and |draw - median|); pixels treated as independent"),
        "reps_phi": int(null_reps), "reps_pixels": int(reps_pix),
        "phi_classic_rhat_median_q95": q(nc_phi), "phi_rank_rhat_median_q95": q(nr_phi),
        "p_classic_rhat_phi_above_1.01": float(np.mean(nc_phi > 1.01)),
        "p_null_classic_rhat_phi_at_least_observed": float(np.mean(nc_phi >= classic_phi)),
        "p_null_rank_rhat_phi_at_least_observed": float(np.mean(nr_phi >= out["rank_normalized"]["phi"]["worst"])),
        "pixel_max_classic_median_q95": q(nc_pix), "pixel_max_rank_median_q95": q(nr_pix),
        "p_null_pixel_max_classic_at_least_observed": float(np.mean(nc_pix >= out["classic"]["rhat_pixel_max"])),
        "p_null_pixel_max_rank_at_least_observed": float(np.mean(nr_pix >= obs_rank_pix)),
        "observed_pixel_max_rank_exceeds_null_q95": bool(obs_rank_pix > np.quantile(nr_pix, 0.95))}
    out["time_s"] = time.time() - t0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--out", default=None, help="default <dir>/convergence_audit.json")
    ap.add_argument("--null-reps", type=int, default=60, help="AR(1) replicates for Phi (a third as many for the pixel maximum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--update", action="store_true", help="keep the entries of the existing output file for runs not listed")
    args = ap.parse_args(argv)
    base = Path(args.dir)
    res = {}
    for r in args.runs:
        res[r] = audit_run(r, base, args.null_reps, args.seed)
        e = res[r]
        print(f"{r:20s} {e['chains']}x{e['draws_per_chain']}  R-hat Phi classic {e['classic']['rhat_phi']:.4f} rank {e['rank_normalized']['phi']['worst']:.4f} "
              f"(null AR(1) P>1.01 {e['null_ar1']['p_classic_rhat_phi_above_1.01']:.2f});  pixel max classic {e['classic']['rhat_pixel_max']:.4f} "
              f"rank {e['rank_normalized']['pixel_max_worst']:.4f} (null q95 {e['null_ar1']['pixel_max_rank_median_q95'][1]:.4f}, "
              f"{e['rank_normalized']['pixels_worst_above_1.05']} px > 1.05);  worst ESS sokal {e['ess']['worst_pixel_sokal_summed']:.0f} "
              f"bulk {e['ess']['worst_pixel_bulk_multichain']:.0f}")
    out_path = Path(args.out) if args.out else base / "convergence_audit.json"
    if args.update and out_path.exists():
        res = {**json.loads(out_path.read_text()), **res}
    out_path.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out_path}")
    return res


if __name__ == "__main__":
    main()
