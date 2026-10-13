"""Analyses of the milestone-6 review follow-up that need no forward solves: the low-pass reading of the image-domain threshold, its SNR dependence, the
TV-weight sensitivity at the other contrasts, the effective number of parameters and the discrepancy target, and the geometry of the book's measurements.

    python scripts/resolution_review_analysis.py --stage lowpass|snr|lambda|peff|book|all [--out-dir results/resolution]

* ``lowpass``  -> lowpass_check.json. For the r/2 strip of an open gap (and the closed gap scored with the same strip) the valley depth minus delta,
  depth = (mean of the two cores) - (mean of the strip) of theta, (positive: the valley holds) of (a) the ideal circular low-pass of the pixelized truth at 2, 2.5 and 3
  cycles per wavelength (``resolution.lowpass``, pad 4), (b) the stored MAPs of both priors. The image-domain threshold of milestone 6 is where this depth crosses delta:
  it is set by the effective bandwidth (about 2.5 cycles/lambda) and by delta = log(eps_r) / 2, not by the noise.
* ``snr``      -> snr_check.json. The same quantities and the Laplace probabilities P(r/2 strip valley) and P(full strip) for the open gap at 20, 30 and 40 dB
  (eps_r 2, g = 0.05, 0.1, 0.15, both priors; the 20 and 40 dB runs of ``make_m6_review_jobs.py`` in results/resolution/snr20 and snr40).
* ``lambda``   -> lambda_sensitivity.json. P(r/2 strip valley), full strip and the truth indicator of the TV prior at lambda*/3, lambda*, 3 lambda* for the contrasts and gaps
  that were run (the Laplace sensitivity files).
* ``verdict``  -> verdict_range.json. The eps_r 2 narrowest discriminating gap (half-height strip) with the Laplace probabilities of null.json and, where an MCMC spot check
  exists, with its probability (mcmc_refeature.json, mcmc_refeature_review.json) in place of the Laplace one: the range of the verdict that the unconverged TV
  chains and the Laplace-vs-MCMC difference leave open. A strip gap discriminates if P_open > 0.95 under both priors and P_null < 0.05.
* ``mcmc``     -> mcmc_review.json. The chains of the follow-up runs that were not finalized (run directories under mcmc/ without a summary.json: the chains that finished
  before the compute budget ended, one or two per run): per chain the step, acceptance, block means of the sampler's potential Phi' (a stationary chain has none
  drifting), the valley depth and the indicator of the feature for the full, half and quarter strip, next to the Laplace probability of the same configuration. No R-hat
  exists for one chain, and none of these is a converged result; the file records what they show.
* ``peff``     -> peff.json. The effective number of parameters of the Matern posterior from the MCMC runs, p_eff = 2 (E[Phi] - Phi_MAP) (a Gaussian posterior has
  E[Phi] - Phi_MAP = p_eff / 2), and the TV weight that the discrepancy rule picks when its target is N - p_eff / 2 instead of N.
* ``book``     -> book_geometry.json. The measurement geometry of the book's Tests #4 to #8 (Sec. 15.4.2.1, pp. 464 to 467) against this benchmark: wavelength, antenna
  planes, standoff in wavelengths, number of data, aggregate SNR gain, and the decay of evanescent spatial frequencies over the standoff.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import gapvalley as gv
from bayes_scatter import shapes as sh
from bayes_scatter.geometry import Grid
from bayes_scatter.resolution import lowpass

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resolution"
GRID = Grid(64, 1.5)
KCUTS = (2.0, 2.5, 3.0)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=["lowpass", "snr", "lambda", "peff", "book", "verdict", "mcmc", "all"], required=True)
    ap.add_argument("--out-dir", default=str(RESULTS))
    return ap.parse_args(argv)


def depth_minus_delta(theta, gs, eps, band_frac=0.5, core_frac=0.5, grid=GRID):
    """(mean of the two core means) - (strip mean) - delta of the gap-valley feature of the open object ``TwoDisks(g=gs, eps_r=eps)`` for a field ``theta`` (n, n): positive
    when the valley holds (the strip is lower than the cores by more than delta = log(eps) / 2)."""
    fg = gv.feature_geometry(grid, sh.TwoDisks(g=gs, r=0.2, eps_r=eps), core_frac, band_frac)
    th = np.asarray(theta, dtype=np.float64).ravel()
    w = fg.weights
    return float(0.5 * (w["core1"] @ th + w["core2"] @ th) - w["strip"] @ th - fg.delta)


def _load(out_dir, rel):
    p = Path(out_dir) / rel
    return np.load(p)["theta_hat"] if p.exists() else None


def stage_lowpass(out_dir, tag="lowpass_check", gaps=(0.025, 0.05, 0.1, 0.15, 0.25), eps_values=(1.1, 2.0, 4.0)):
    h = GRID.h
    out = {"definition": "depth minus delta of the gap-valley feature (r/2 strip unless stated, cores of 0.5 r): positive = the valley holds; 'open' = the object with the "
                         "gap of the strip, 'closed' = touching disks scored with the same strip; lowpass_k: the ideal circular low-pass of the pixelized truth at k "
                         "cycles per wavelength; map_*: the stored Laplace MAP (open) or the closed-gap MAP of the null runs",
           "h": h, "rows": []}
    for eps in eps_values:
        for gs in gaps:
            row = {"eps_r": eps, "strip_gap": gs, "delta": gv.delta_of(eps)}
            for state, g_true in (("open", gs), ("closed", 0.0)):
                th = sh.TwoDisks(g=g_true, r=0.2, eps_r=eps).theta(GRID)
                cell = {"truth": depth_minus_delta(th, gs, eps)}
                for kc in KCUTS:
                    lp = lowpass(th, h, kc, pad=4)
                    cell[f"lowpass_{kc:g}"] = depth_minus_delta(lp, gs, eps)
                    cell[f"lowpass_{kc:g}_full_strip"] = depth_minus_delta(lp, gs, eps, band_frac=1.0)
                for prior in ("matern", "tv"):
                    rel = f"laplace/g{gs:g}_eps{eps:g}_{prior}.npz" if state == "open" else f"null/g0_eps{eps:g}_{prior}.npz"
                    mp = _load(out_dir, rel)
                    cell[f"map_{prior}"] = None if mp is None else depth_minus_delta(mp, gs, eps)
                row[state] = cell
            out["rows"].append(row)
    (Path(out_dir) / f"{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def _sens_p(out_dir, name, key):
    p = Path(out_dir) / "laplace_sens" / f"{name}.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text())["variants"][key]
    return {"p": v["p_valley_region"], "truth_holds": v["truth_holds_region"], "status": v["region_status"]}


def stage_snr(out_dir):
    out_dir = Path(out_dir)
    h = GRID.h
    out = {"definition": "eps_r 2; Laplace probabilities of the r/2-strip and full-strip valley for the open gap, and the depth minus delta of the MAP and of the ideal 2.5 "
                         "cycle/lambda low-pass, at three SNRs; the TV weight is that of 30 dB (0.5) at every SNR", "rows": []}
    for gs in (0.05, 0.1, 0.15):
        for snr in (20, 30, 40):
            base = out_dir if snr == 30 else out_dir / f"snr{snr}"
            for prior in ("matern", "tv"):
                name = f"g{gs:g}_eps2_{prior}"
                rec = {"snr_db": snr, "strip_gap": gs, "prior": prior, "present": (base / "laplace" / f"{name}.json").exists()}
                if rec["present"]:
                    j = json.loads((base / "laplace" / f"{name}.json").read_text())
                    rec["phi_over_N"] = j["map_summary"]["phi_over_N"]
                    rec["map_ok"] = j["map_summary"]["map_ok"]
                    rec["lam"] = j["lam"]
                    rec["p_full_strip"] = _sens_p(base, name, "band1_core0.5")
                    rec["p_half_strip"] = _sens_p(base, name, "band0.5_core0.5")
                    rec["p_quarter_strip"] = _sens_p(base, name, "band0.25_core0.5")
                    mp = _load(base, f"laplace/{name}.npz")
                    rec["map_depth_minus_delta_half_strip"] = None if mp is None else depth_minus_delta(mp, gs, 2.0)
                out["rows"].append(rec)
    lp = {f"{gs:g}": {f"{kc:g}": depth_minus_delta(lowpass(sh.TwoDisks(g=gs, r=0.2, eps_r=2.0).theta(GRID), h, kc, pad=4), gs, 2.0) for kc in KCUTS} for gs in (0.05, 0.1, 0.15)}
    out["lowpass_depth_minus_delta_half_strip_open"] = lp
    (out_dir / "snr_check.json").write_text(json.dumps(out, indent=1))
    return out


def stage_lambda(out_dir):
    out_dir = Path(out_dir)
    lam_star = json.loads((out_dir / "laplace_lambda.json").read_text())["lambda_star"]
    rows = []
    for eps in (1.1, 2.0, 4.0):
        for gs in (0.05, 0.1, 0.15):
            cell = {"eps_r": eps, "g": gs, "lambda_star": lam_star[f"{eps:g}"]["lam"], "by_scale": {}}
            for scale, suffix in ((1 / 3, "_x0.333333"), (1.0, ""), (3.0, "_x3")):
                name = f"g{gs:g}_eps{eps:g}_tv{suffix}"
                cell["by_scale"][f"{scale:.3g}"] = {"name": name, "half_strip": _sens_p(out_dir, name, "band0.5_core0.5"), "full_strip": _sens_p(out_dir, name, "band1_core0.5"),
                                                    "phi_over_N": json.loads((out_dir / "laplace" / f"{name}.json").read_text())["map_summary"]["phi_over_N"]
                                                    if (out_dir / "laplace" / f"{name}.json").exists() else None}
            rows.append(cell)
    res = {"definition": "TV weight sensitivity of the gap-valley probability (Laplace) at lambda*/3, lambda*, 3 lambda* of each contrast", "rows": rows}
    (out_dir / "lambda_sensitivity.json").write_text(json.dumps(res, indent=1))
    return res


def _crossing(lams, phis, target):
    for (l1, p1), (l2, p2) in zip(zip(lams, phis), zip(lams[1:], phis[1:])):
        if (p1 - target) * (p2 - target) < 0:
            t = (target - p1) / (p2 - p1)
            return float(np.exp(np.log(l1) + t * (np.log(l2) - np.log(l1))))
    return None


def _mcmc_probabilities(out_dir):
    """{run directory: {variant key: {p_mcmc, mcse, p_laplace, per_chain}}} from mcmc_refeature.json and mcmc_refeature_review.json (those that exist), and the convergence flags
    {run: {converged, converged_classic, rhat_pixel_max, ...}} from mcmc.json."""
    out_dir = Path(out_dir)
    probs = {}
    for f in ("mcmc_refeature.json", "mcmc_refeature_review.json"):
        if (out_dir / f).exists():
            for run, rec in json.loads((out_dir / f).read_text())["configurations"].items():
                probs.setdefault(run, {}).update({k: dict(v, source=f) for k, v in rec["variants"].items()})
    conv = {}
    if (out_dir / "mcmc.json").exists():
        for row in json.loads((out_dir / "mcmc.json").read_text())["table"]:
            conv[row["name"]] = {k: row.get(k) for k in ("converged", "converged_classic", "rhat_pixel_max", "rhat_phi_prime", "ess_pixel_min", "ess_phi_prime", "draws_per_chain", "chains")}
    return probs, conv


def stage_verdict(out_dir, eps=2.0, variant="band0.5_core0.5", hi=0.95, lo=0.05):
    out_dir = Path(out_dir)
    null = json.loads((out_dir / "null.json").read_text())
    probs, conv = _mcmc_probabilities(out_dir)
    mcmc_review = json.loads((out_dir / "mcmc_review.json").read_text()) if (out_dir / "mcmc_review.json").exists() else {}
    rows = [r for r in null["rows"] if r["eps_r"] == eps and r["variant"] == variant]
    gaps = sorted({r["strip_gap"] for r in rows})
    # the MCMC runs available per (prior, gap): the run directory names
    runs = {}
    for run in probs:
        stem = run.split("_od2")[0]
        for prior in ("matern", "tv"):
            for g in gaps:
                if stem == f"g{g:g}_eps{eps:g}_{prior}":
                    runs.setdefault((prior, g), []).append(run)
    table = []
    for g in gaps:
        cell = {"strip_gap": g}
        for prior in ("matern", "tv"):
            r = next((x for x in rows if x["strip_gap"] == g and x["prior"] == prior), None)
            cell[prior] = {"p_open_laplace": None if r is None else r["p_open"], "p_null_laplace": None if r is None else r["p_null"],
                           "mcmc": {run: {"p_open_mcmc": probs[run][variant].get("p_mcmc"), "mcse": probs[run][variant].get("mcse"), "chains": conv.get(run, {}).get("chains"),
                                          "draws_per_chain": conv.get(run, {}).get("draws_per_chain"), "converged": conv.get(run, {}).get("converged"),
                                          "rhat_pixel_max": conv.get(run, {}).get("rhat_pixel_max"), "source": probs[run][variant].get("source")}
                                    for run in runs.get((prior, g), []) if variant in probs[run]}}
        single = []
        for prior in ("matern", "tv"):
            for run, r in mcmc_review.get("runs", {}).items():
                if r["config"] == f"g{g:g}_eps{eps:g}_{prior}":
                    for c, rec in r["chains"].items():
                        v = rec["variants"].get(variant)
                        if v is not None and v["p_valley_mcmc_chain"] is not None:
                            single.append({"prior": prior, "run": run, "chain": int(c), "draws": rec["draws"], "p_open_mcmc_single_chain": v["p_valley_mcmc_chain"],
                                           "phi_prime_drift_per_block": rec["phi_prime_drift_per_block"], "step": rec["step"], "acceptance": rec["acceptance"]})
        cell["single_chains_not_stationary"] = single
        table.append(cell)

    def narrowest(choice):
        """Narrowest strip gap from which every wider swept gap discriminates; ``choice`` maps (prior, gap) -> the open-gap probability to use (None: Laplace)."""
        star = None
        for cell in sorted(table, key=lambda c: -c["strip_gap"]):
            ok = True
            for prior in ("matern", "tv"):
                c = cell[prior]
                p_open = choice(prior, cell["strip_gap"], c)
                ok &= bool(p_open is not None and c["p_null_laplace"] is not None and p_open > hi and c["p_null_laplace"] < lo)
            if not ok:
                break
            star = cell["strip_gap"]
        return star

    def laplace_only(prior, g, c):
        return c["p_open_laplace"]

    def with_mcmc(sources):
        def choice(prior, g, c):
            ms = [v["p_open_mcmc"] for run, v in c["mcmc"].items() if v["source"] in sources and v["p_open_mcmc"] is not None]
            return ms[-1] if ms else c["p_open_laplace"]  # the last run in the order of the files: the review runs come after the original ones
        return choice

    res = {"definition": f"eps_r {eps:g}, variant {variant}: a strip gap discriminates if P_open > {hi} under both priors and P_null < {lo} (null.json); the MCMC variants "
                         "replace the Laplace P_open by the MCMC one where a run exists (P_null stays Laplace: no MCMC of the closed-gap posterior exists)",
           "table": table, "narrowest_gap": {"laplace": narrowest(laplace_only),
                                             "mcmc_original_runs": narrowest(with_mcmc({"mcmc_refeature.json"})),
                                             "mcmc_all_runs_where_present": narrowest(with_mcmc({"mcmc_refeature.json", "mcmc_refeature_review.json"}))}}
    (out_dir / "verdict_range.json").write_text(json.dumps(res, indent=1))
    return res


def stage_mcmc_review(out_dir, block=50):
    import re

    out_dir = Path(out_dir)
    sens = json.loads((out_dir / "laplace_sensitivity.json").read_text())["configurations"] if (out_dir / "laplace_sensitivity.json").exists() else {}
    res = {"phi_prime_by_segment": None, "definition": "single chains of runs without a summary.json (unfinalized): Phi' block means (drift = not stationary), valley depth of the half-height strip, indicator "
                         "of the feature for three strips; laplace: the Laplace probability of the same configuration (laplace_sensitivity.json)", "runs": {}}
    by_seg = {}
    for d in sorted((out_dir / "mcmc").glob("g*_eps*_*")):
        m = re.fullmatch(r"g([0-9.]+)_eps([0-9.]+)_(matern|tv)(_.*)?", d.name)
        if m is None:
            continue
        for c in range(8):  # every chain of every run: the step, and acceptance and mean of Phi' per segment (a chain that is still relaxing has a Phi' that moves between segments)
            segs = sorted((p for p in d.glob(f"chain_{c}_seg[0-9].npz") if p.is_file()), key=lambda p: int(p.stem.split("seg")[1]))
            if segs:
                z = [np.load(p) for p in segs]
                by_seg.setdefault(d.name, {})[str(c)] = {"step": float(z[0]["step"]), "start": str(z[0]["starts"]) if "starts" in z[0].files else None,
                                                         "acceptance_by_segment": [float(x["accept"].mean()) for x in z], "phi_prime_mean_by_segment": [float(x["phi_prime"].mean()) for x in z]}
        if (d / "summary.json").exists():
            continue
        g, eps, prior, tag = float(m.group(1)), float(m.group(2)), m.group(3), m.group(4) or ""
        config = f"g{g:g}_eps{eps:g}_{prior}"
        obj = sh.TwoDisks(g=g, r=0.2, eps_r=eps)
        chains = {}
        for c in range(8):
            segs = sorted(d.glob(f"chain_{c}_seg[0-9].npz"), key=lambda p: int(p.stem.split("seg")[1]))
            segs = [p for p in segs if p.is_file()]  # a link to a missing file is not a segment
            if not segs:
                continue
            z = [np.load(p) for p in segs]
            theta = np.concatenate([x["theta"] for x in z])
            pp = np.concatenate([x["phi_prime"] for x in z])
            acc = np.concatenate([x["accept"] for x in z])
            f = theta.reshape(len(theta), -1)
            rec = {"segments": len(segs), "draws": int(len(theta)), "step": float(z[0]["step"]), "acceptance": float(acc.mean()), "n_warmup_seg0": int(z[0]["n_warmup"]),
                   "phi_prime_block_means": [float(pp[i:i + block].mean()) for i in range(0, len(pp), block)],
                   "phi_prime_drift_per_block": float((pp[-block:].mean() - pp[:block].mean()) / max(1, (len(pp) - block) // block)) if len(pp) > block else None,
                   "variants": {}}
            for b in (1.0, 0.5, 0.25):
                fg = gv.feature_geometry(GRID, obj, 0.5, b)
                w = fg.weights
                depth = 0.5 * (f @ w["core1"] + f @ w["core2"]) - f @ w["strip"]
                ind = gv.valley_indicator(theta[None], w["strip"], w["core1"], w["core2"], fg.delta)[0]
                lap = sens.get(config, {}).get(f"band{b:g}_core0.5", {})
                rec["variants"][f"band{b:g}_core0.5"] = {"representable": fg.representable, "p_valley_mcmc_chain": float(ind.mean()), "depth_mean": float(depth.mean()),
                                                         "depth_sd": float(depth.std(ddof=1)), "depth_block_means": [float(depth[i:i + block].mean()) for i in range(0, len(depth), block)],
                                                         "delta": fg.delta, "p_valley_laplace": lap.get("p_valley_region"), "depth_mean_laplace": lap.get("depth_mean"),
                                                         "depth_sd_laplace": lap.get("depth_sd")}
            chains[str(c)] = rec
        if chains:
            res["runs"][d.name] = {"config": config, "g": g, "eps_r": eps, "prior": prior, "tag": tag, "chains": chains}
    res["phi_prime_by_segment"] = by_seg
    (out_dir / "mcmc_review.json").write_text(json.dumps(res, indent=1))
    return res


def stage_peff(out_dir):
    out_dir = Path(out_dir)
    mc = json.loads((out_dir / "mcmc.json").read_text())
    n_data = 2400
    pe = {}
    for name, s in mc["summaries"].items():
        if s["prior"] != "matern":
            continue
        ph = s["phi_physical"]
        pe[name] = {"phi_mean": ph["mean"], "phi_map": ph["map_phi"], "p_eff": 2.0 * (ph["mean"] - ph["map_phi"]), "chains": s["chains"],
                    "draws_per_chain": s["draws_per_chain"], "converged": s["convergence"]["converged"], "converged_classic": s["convergence"].get("converged_classic")}
    p_eff = float(np.median([v["p_eff"] for v in pe.values()])) if pe else None
    target = n_data - 0.5 * p_eff if p_eff is not None else None
    lam = json.loads((out_dir / "laplace_lambda.json").read_text())
    rules = {}
    for eps, runs in lam["runs"].items():
        runs = [r for r in runs if r["converged"]]
        lams, phis = [r["lam"] for r in runs], [r["phi"] for r in runs]
        closest = lambda t: float(lams[int(np.argmin([abs(p - t) for p in phis]))])  # noqa: E731
        rules[eps] = {"lambda_star_N": lam["lambda_star"][eps]["lam"], "lambda_closest_to_target": None if target is None else closest(target),
                      "crossing_N": _crossing(lams, phis, n_data), "crossing_target": None if target is None else _crossing(lams, phis, target),
                      "phi_map_by_lambda": dict(zip([f"{x:g}" for x in lams], phis))}
    res = {"definition": "p_eff = 2 (E[Phi] - Phi_MAP) of the Matern MCMC runs (Gaussian posterior: E[Phi] - Phi_MAP = p_eff / 2); discrepancy target N - p_eff / 2 against N",
           "n_data": n_data, "matern_runs": pe, "p_eff_median": p_eff, "target": target, "sqrt_n": float(np.sqrt(n_data)), "lambda_rule": rules}
    (out_dir / "peff.json").write_text(json.dumps(res, indent=1))
    return res


def stage_book(out_dir):
    """Geometry of the book's 3-D measurements (Ch. 15 Sec. 15.4.2.1, quoted values) and what it implies for evanescent information."""
    lam = 0.6  # m, 500 MHz
    tx_planes, rx_planes = (-0.1, 0.7), (-0.2, 0.8)  # z (m) of the 128 transmitters and the 162 receivers
    doi_center_z, object_height = 0.3, 0.1  # objects 0.1 m (0.167 lambda) high about z = 0.3 m (xy slices at z = 0.3 m)
    z_lo, z_hi = doi_center_z - 0.5 * object_height, doi_center_z + 0.5 * object_height
    standoff = {"transmitters_m": min(min(abs(z - z_lo), abs(z - z_hi)) for z in tx_planes), "receivers_m": min(min(abs(z - z_lo), abs(z - z_hi)) for z in rx_planes)}
    standoff_wl = {k: v / lam for k, v in standoff.items()}
    d_wl = {"nearest_antenna_plane": min(standoff_wl.values()), "farthest_of_the_nearest": max(standoff_wl.values())}
    book_n, our_n = 128 * 162, 40 * 60
    # an evanescent plane-wave component of spatial frequency f (cycles per wavelength > 1) decays over a distance d (wavelengths) by exp(-2 pi sqrt(f^2 - 1) d)
    def decay(f, d):
        return float(np.exp(-2.0 * np.pi * np.sqrt(f * f - 1.0) * d))

    freqs = {"gap lambda/4 (period 2g = lambda/2, f = 2)": 2.0, "gap lambda/10 (f = 5)": 5.0, "gap lambda/20 (f = 10)": 10.0}
    ours_rings = {"tx_ring_radius": 4.0, "rx_ring_radius": 4.5, "object_half_extent": 2 * 0.2 + 0.025}  # the pair of r = 0.2 disks reaches x = +- (2 r + g / 2)
    standoff_ours = ours_rings["tx_ring_radius"] - ours_rings["object_half_extent"]
    res = {"source": "Campbell and Werner (eds.), Advances in Electromagnetics Empowered by AI and DL, Ch. 15, Sec. 15.4.2.1, pp. 464-467 (Xiao et al.)",
           "quoted": {"frequency_MHz": 500, "wavelength_m": lam, "transmitters": 128, "receivers": 162, "tx_planes_z_m": list(tx_planes), "rx_planes_z_m": list(rx_planes),
                      "doi_m": "1.4 x 1.4 x 0.4 about z = 0.3 m", "object_height_m": object_height,
                      "test4": "six objects, adjacent spacings 5/20, 4/20, 3/20, 2/20, 1/20 lambda, relative permittivity 3.2 down to 2.2 in steps of 0.2; compared with VBIM; no noise level stated",
                      "tests5to8": "spacings 4/10, 3/10, 2/10, 1/10 lambda, nine connected objects with permittivity 2.2 to 3.8 in steps of 0.2; noise-free, 20 dB and 10 dB (power SNR), Table 15.9"},
           "standoff_m": standoff, "standoff_wavelengths": standoff_wl, "standoff_range_wavelengths": d_wl,
           "n_data": {"book": book_n, "this_benchmark": our_n, "ratio": book_n / our_n, "aggregate_snr_gain_db_at_equal_per_datum_snr": float(10 * np.log10(book_n / our_n))},
           "evanescent_decay_per_path": {"book_standoff_nearest_plane_wavelengths": d_wl["nearest_antenna_plane"],
                                         "factors": {k: decay(f, d_wl["nearest_antenna_plane"]) for k, f in freqs.items()},
                                         "log_factors": {k: -2.0 * np.pi * np.sqrt(f * f - 1.0) * d_wl["nearest_antenna_plane"] for k, f in freqs.items()}},
           "evanescent_decay_this_benchmark": {"standoff_wavelengths_nearest": float(standoff_ours), **ours_rings,
                                               "factors": {k: decay(f, standoff_ours) for k, f in freqs.items()},
                                               "log_factors": {k: -2.0 * np.pi * np.sqrt(f * f - 1.0) * standoff_ours for k, f in freqs.items()}},
           "note": "frequency needed to resolve a gap g: half a period equals g, f = 1 / (2 g) cycles per wavelength; the near-field standoff of the book (>= 0.58 lambda) does not supply "
                   "the missing spatial frequencies: the decay of f = 5 and 10 over one path is far below any noise level"}
    (Path(out_dir) / "book_geometry.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None):
    args = parse_args(argv)
    stages = {"lowpass": stage_lowpass, "snr": stage_snr, "lambda": stage_lambda, "peff": stage_peff, "book": stage_book, "verdict": stage_verdict, "mcmc": stage_mcmc_review}
    todo = list(stages) if args.stage == "all" else [args.stage]
    out = {}
    for st in todo:
        out[st] = stages[st](args.out_dir)
        print(f"{st}: written")
    return out


if __name__ == "__main__":
    main()
