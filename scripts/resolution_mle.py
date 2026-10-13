"""Monte Carlo maximum-likelihood check of the Cramer-Rao required SNR (milestone-6 review, DESIGN 13.2).

    python scripts/resolution_mle.py --stage chunk --point eps2_g0.05_req --first 0 --count 8
    python scripts/resolution_mle.py --stage collect

The Cramer-Rao bound is local. At the SNR where sigma_CRB(g) = g the nuisance parameters (r, eps_r, c_x, c_y) can be essentially undetermined, so the bound
may not describe the error of any estimator there. This script draws noisy data of the two-disk model at a given SNR and fits the five parameters
p = (g, r, eps_r, c_x, c_y) by Levenberg-Marquardt maximum likelihood (``bayes_scatter.mle``), once from the truth (the local, oracle estimate) and from
``n_random_starts`` starts that do not depend on the truth (a global search by best Phi), then compares the empirical error with the bound.

Points (``POINTS``; eps_r 2, r = 0.2, the book's rings, side 1.5 wavelengths, data AND fit on the n = 64 grid of the Laplace sweep, exact-coverage rendering, so that
the model is the true one and the question is only the statistics of the estimator; the bound is recomputed on the same grid):

  eps2_g0.05_req     SNR = the SNR_req of g = 0.05 (nuisances unknown, n = 64): lambda/20, the book's finest gap, where the bound says sigma_CRB(g) = g
  eps2_g0.1_req      the same for g = 0.1 (lambda/10)
  eps2_g0.05_plus10  g = 0.05 at SNR_req + 10 dB: where the bound is expected to be tight, a check that the estimator and the bound agree there

Stages. ``chunk``: realizations --first .. --first + --count - 1 of one point, noise key ``fold_in(key(point seed), k)``, written to
results/resolution/mle/<point>/chunk_<first>_<last + 1>.json (the done marker; a rerun skips it). ``collect``: every chunk -> results/resolution/mle.json
(per point: the bound at the SNR, the empirical bias / sd / RMSE of the truth-start and best-of-starts estimators, the fraction on the boundary g = 0, the fraction
where the best start is another basin than the truth start, coverage of +-1.96 sigma_CRB) and mle_check.png.

Threads: one XLA thread per job (``scripts/batch_run.py --threads 1``, jobs from ``make_m6_review_jobs.py``).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
from matplotlib.figure import Figure

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import fisher as fi
from bayes_scatter import mle
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import add_noise, noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)

RADIUS, SIDE, N_GRID, TOL = 0.2, 1.5, 64, 1e-10
POINTS = {
    "eps2_g0.05_req": {"eps_r": 2.0, "g": 0.05, "snr_offset_db": 0.0, "n_random_starts": 2, "seed": 101},
    "eps2_g0.1_req": {"eps_r": 2.0, "g": 0.1, "snr_offset_db": 0.0, "n_random_starts": 2, "seed": 102},
    "eps2_g0.05_plus10": {"eps_r": 2.0, "g": 0.05, "snr_offset_db": 10.0, "n_random_starts": 1, "seed": 103},
}
INK, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", choices=["chunk", "collect"], required=True)
    ap.add_argument("--point", choices=sorted(POINTS))
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--n", type=int, default=N_GRID, help="grid of both the data and the fit")
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args(argv)
    if args.stage == "chunk" and args.point is None:
        ap.error("--stage chunk needs --point")
    return args


def chunk_path(out_dir, point, first, last):
    return Path(out_dir) / "mle" / point / f"chunk_{first}_{last}.json"


def point_setup(forward, point):
    """The bound at the point's SNR: ``(obj, d_clean, snr_db, s, crb)``.

    ``crb``: dict with ``names``, ``snr_req_db`` (30 dB reference + 20 log10(sigma_CRB(g) / g), nuisances unknown, on this grid), ``snr_db`` (that plus the point's offset),
    ``sd`` (the Cramer-Rao standard deviations of (g, r, eps_r, cx, cy) at ``snr_db``, all five unknown), ``F`` (at ``snr_db``)."""
    cfg = POINTS[point]
    obj = sh.TwoDisks(g=cfg["g"], r=RADIUS, eps_r=cfg["eps_r"])
    rec = fi.analyze(forward, obj, snr_ref_db=30.0, names=mle.PARAMS, born=False)
    snr_req = float(rec["required_snr_db"]["unknown"])
    snr = snr_req + cfg["snr_offset_db"]
    chi = obj.chi(forward.grid)
    d_clean, _ = forward(jax.numpy.asarray(chi))
    s = noise_std_from_snr(d_clean, snr)
    F = fi.scale_to_snr(np.array(rec["F"]), snr, 30.0)
    sd = np.sqrt(np.diag(np.linalg.inv(F)))
    return obj, d_clean, snr, s, {"names": list(mle.PARAMS), "snr_req_db": snr_req, "snr_db": snr, "sd": sd.tolist(), "F": F.tolist(),
                                  "aggregate_snr_db": float(snr + 10.0 * np.log10(forward.n_tx * forward.n_rx)), "sigma_crb_g_at_30db": float(rec["crb_g_unknown"]),
                                  "cond_F_scaled": float(rec["cond_F_scaled"])}


def stage_chunk(args):
    cfg = POINTS[args.point]
    last = args.first + args.count
    path = chunk_path(args.out_dir, args.point, args.first, last)
    if path.exists():
        print(f"{args.point} [{args.first}, {last}): done ({path})")
        return json.loads(path.read_text())
    lock = path.with_suffix(".lock")
    if not acquire_lock(lock):
        print(f"{args.point} [{args.first}, {last}): being computed by another process ({lock}); nothing to do")
        return None
    try:
        return compute_chunk(args, cfg, last, path)
    finally:
        lock.unlink(missing_ok=True)


def acquire_lock(lock, stale_s=7200.0):
    """Create ``lock`` exclusively (True) unless another process holds it (False); a lock older than ``stale_s`` seconds is a leftover of a killed job and is taken over.
    Two jobs with the same chunk (a batch job and a manual run) then never compute it twice."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > stale_s:
                lock.unlink(missing_ok=True)
                continue
            return False
    return False


def compute_chunk(args, cfg, last, path):
    t_all = time.perf_counter()
    forward = Forward(ch15_setup(n=args.n, side=SIDE), tol=TOL, restart=100)
    obj, d_clean, snr, s, crb = point_setup(forward, args.point)
    p_true = obj.vector(mle.PARAMS)
    sd = np.array(crb["sd"])
    print(f"{args.point}: n = {args.n}, SNR {snr:.2f} dB (required {crb['snr_req_db']:.2f}), aggregate {crb['aggregate_snr_db']:.1f} dB, sigma_CRB " +
          ", ".join(f"{k} {v:.3g}" for k, v in zip(mle.PARAMS, sd)))
    key = jax.random.key(cfg["seed"])
    records = []
    for k in range(args.first, last):
        t0 = time.perf_counter()
        d_obs = add_noise(jax.random.fold_in(key, k), d_clean, s)
        rng = np.random.default_rng(cfg["seed"] * 100003 + k)
        starts = [("truth", p_true)] + [(f"random{i}", mle.random_start(rng)) for i in range(cfg["n_random_starts"])]
        best, runs = mle.fit_multistart(forward, d_obs, s, starts, scale=sd)
        phi_truth = float(mle._evaluate(forward, d_obs, s, p_true)[3])
        records.append({"k": k, "phi_at_truth": phi_truth, "best": best["label"], "p_best": best["p"].tolist(), "phi_best": best["phi"],
                        "runs": [{"label": r["label"], "p0": dict(starts)[r["label"]].tolist(), "p": r["p"].tolist(), "phi": r["phi"], "n_iter": r["n_iter"], "n_eval": r["n_eval"],
                                  "converged": r["converged"], "stop": r["stop"], "at_boundary": r["at_boundary"], "time_s": r["time_s"]} for r in runs]})
        tr = next(r for r in runs if r["label"] == "truth")
        print(f"  k = {k}: truth start g_hat {tr['p'][0]:.4f} (Phi {tr['phi']:.1f}, {tr['n_iter']} its, {tr['stop']}), best {best['label']} g_hat {best['p'][0]:.4f} "
              f"(Phi {best['phi']:.1f}), Phi(truth) {phi_truth:.1f}; {time.perf_counter() - t0:.0f} s")
    rec = {"point": args.point, "config": cfg, "truth": dict(zip(mle.PARAMS, p_true.tolist())), "grid_n": args.n, "side": SIDE, "radius": RADIUS, "noise_std": s,
           "crb": crb, "n_data": forward.n_tx * forward.n_rx, "first": args.first, "last": last, "records": records, "time_s": time.perf_counter() - t_all,
           "thread_env": {k: os.environ.get(k) for k in ("NPROC", "XLA_FLAGS", "OMP_NUM_THREADS")}, "load_avg_1min": os.getloadavg()[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(path)
    return rec


def summarize_point(chunks):
    """Statistics of one point from its chunk records (a list of chunk dicts of ``stage_chunk``)."""
    recs = sorted((r for c in chunks for r in c["records"]), key=lambda r: r["k"])
    head = chunks[0]
    truth = np.array([head["truth"][k] for k in mle.PARAMS])
    sd = np.array(head["crb"]["sd"])
    p_local = np.array([next(x for x in r["runs"] if x["label"] == "truth")["p"] for r in recs])
    p_best = np.array([r["p_best"] for r in recs])
    g = truth[0]
    out = {"config": head["config"], "truth": head["truth"], "snr_db": head["crb"]["snr_db"], "snr_req_db": head["crb"]["snr_req_db"],
           "aggregate_snr_db": head["crb"]["aggregate_snr_db"], "crb_sd": dict(zip(mle.PARAMS, sd.tolist())), "crb_sd_over_truth": dict(zip(mle.PARAMS, (sd / truth).tolist())),
           "grid_n": head["grid_n"], "n_realizations": len(recs), "ks": [r["k"] for r in recs],
           "truth_start": mle.estimator_summary(p_local, truth, sd), "best_of_starts": mle.estimator_summary(p_best, truth, sd),
           "fraction_g_hat_at_zero_truth_start": float(np.mean(p_local[:, 0] < 1e-9)), "fraction_g_hat_at_zero_best": float(np.mean(p_best[:, 0] < 1e-9)),
           "fraction_best_is_not_truth_start": float(np.mean([r["best"] != "truth" for r in recs])),
           "fraction_best_phi_lower_than_truth_start_by_more_than_1": float(np.mean([
               next(x for x in r["runs"] if x["label"] == "truth")["phi"] - r["phi_best"] > 1.0 for r in recs])),
           "fraction_all_starts_converged": float(np.mean([all(x["converged"] for x in r["runs"]) for r in recs])),
           "fraction_g_hat_above_half_g_best": float(np.mean(p_best[:, 0] > 0.5 * g)), "fraction_g_hat_above_half_g_truth_start": float(np.mean(p_local[:, 0] > 0.5 * g)),
           "median_abs_error_over_g_best": float(np.median(np.abs(p_best[:, 0] - g)) / g), "median_abs_error_over_g_truth_start": float(np.median(np.abs(p_local[:, 0] - g)) / g),
           "mean_phi_best_minus_phi_at_truth": float(np.mean([r["phi_best"] - r["phi_at_truth"] for r in recs])),
           "g_hat_best": p_best[:, 0].tolist(), "g_hat_truth_start": p_local[:, 0].tolist(), "eps_hat_best": p_best[:, 2].tolist(), "eps_hat_truth_start": p_local[:, 2].tolist(),
           "wall_s_total": float(sum(c["time_s"] for c in chunks))}
    return out


def stage_collect(args):
    out_dir = Path(args.out_dir)
    result = {"note": "empirical error of the Levenberg-Marquardt maximum likelihood estimator of (g, r, eps_r, cx, cy) from noisy data of the same model at the stated SNR, against the "
                      "Cramer-Rao standard deviations at that SNR (all five parameters unknown, same grid); truth_start: the fit started at the true parameters (local, oracle); "
                      "best_of_starts: lowest Phi among the truth start and starts drawn from a box that does not depend on the truth", "points": {}}
    for point in POINTS:
        files = sorted((out_dir / "mle" / point).glob("chunk_*.json"))
        if not files:
            continue
        chunks = [json.loads(f.read_text()) for f in files]
        ks = [r["k"] for c in chunks for r in c["records"]]
        if len(set(ks)) != len(ks):
            raise SystemExit(f"{point}: duplicate realizations in the chunks")
        result["points"][point] = summarize_point(chunks)
        p = result["points"][point]
        print(f"{point}: {p['n_realizations']} realizations, SNR {p['snr_db']:.2f} dB (aggregate {p['aggregate_snr_db']:.1f} dB); g = {p['truth']['g']}: CRB sd {p['crb_sd']['g']:.4g}; "
              f"truth-start sd {p['truth_start']['sd'][0]:.4g} rmse {p['truth_start']['rmse'][0]:.4g}, best-of-starts sd {p['best_of_starts']['sd'][0]:.4g} rmse {p['best_of_starts']['rmse'][0]:.4g}")
    if not result["points"]:
        raise SystemExit(f"no chunks in {out_dir / 'mle'}")
    (out_dir / "mle.json").write_text(json.dumps(result, indent=1))
    make_figure(result, out_dir)
    print(f"wrote {out_dir / 'mle.json'}")
    return result


def make_figure(result, out_dir):
    pts = list(result["points"].items())
    fig = Figure(figsize=(3.7 * len(pts) + 0.3, 3.6), dpi=130)
    axes = fig.subplots(1, len(pts), squeeze=False)[0]
    for ax, (name, p) in zip(axes, pts):
        g, sd = p["truth"]["g"], p["crb_sd"]["g"]
        hi = max(max(p["g_hat_best"]), g + 3 * sd, max(p["g_hat_truth_start"]))
        bins = np.linspace(0.0, hi * 1.02, 24)
        ax.hist(p["g_hat_best"], bins=bins, color=BLUE, alpha=0.55, lw=0, label="best of starts")
        ax.hist(p["g_hat_truth_start"], bins=bins, histtype="step", color=ORANGE, lw=1.4, label="from the truth")
        xs = np.linspace(0.0, hi, 300)
        dens = np.exp(-0.5 * ((xs - g) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) * p["n_realizations"] * (bins[1] - bins[0])
        ax.plot(xs, dens, color=INK, lw=1.0, label="Cramer-Rao Gaussian")
        ax.axvline(g, color=MUTED, lw=0.9, ls=":")
        ax.set_title(f"{name}: {p['snr_db']:.1f} dB, M = {p['n_realizations']}", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("estimated gap g_hat (wavelengths)", color=INK, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "mle_check.png")


def main(argv=None):
    args = parse_args(argv)
    return stage_chunk(args) if args.stage == "chunk" else stage_collect(args)


if __name__ == "__main__":
    main()
