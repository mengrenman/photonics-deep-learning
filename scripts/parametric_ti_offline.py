"""Milestone 7: complete a thermodynamic-integration run whose last temperature (beta = 1) did not finish, using the production chains for that temperature.

    python scripts/parametric_ti_offline.py --truth T1 --gap 0.05 --model F2          # <out-dir>/<case>/offline_ti.json

The power posterior at beta = 1 IS the posterior, so the moments of Phi at the last temperature (mean, variance, bulk ESS, R-hat, acceptance) are those of the production chains
(``scripts/parametric_run.py --stage chain``: 4 chains of 500 draws), which are larger and better converged than one more TI temperature (one chain of 150 draws). Every earlier
temperature is read from the TI run's own files (<case>/ti_<tag>/t00.npz ... ), and ``shapepost.ti_combine`` does the quadrature unchanged. The output marks the estimate as
``offline`` (the last temperature substituted) and lists the runs found; nothing is written to the ``ti*.json`` files of the run itself, so a run that later finishes by itself is not
counted twice. A run is used only if it has all temperatures but the last, and the combination over runs is the same as ``parametric_run.combine_ti_runs`` (mean, quadrature-sum
error, between-run standard deviation).

The Euler-Maclaurin end correction of ``shapepost.ti_combine`` is applied only to intervals where Var_beta[phi] changes by at most a factor of 10. In the first version of this
analysis it was applied everywhere and added 26.7 of 36.0 nats in one interval of the first run (beta = 1.3e-6 to 4.2e-5, where the variance falls from 1.9e11 to 8.5e8, outside the
validity of a one-term expansion), which put TI 15 to 27 nats ABOVE the Laplace evidence; the first output is kept as offline_ti_v1.json. Now ``schedule_ok`` is False (the first
intervals of the power schedule are too coarse for these posteriors; ``shapepost.ti_schedule(beta_min=...)`` is the geometric alternative) and the TI value is a trapezoid lower bound
there. The importance-sampling evidence (``parametric_evidence_is.py``, ``is_evidence.json``) is the check of the Laplace value; if it exists its difference to TI is listed.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

import parametric_run as prun
from bayes_scatter import diagnostics as dg
from bayes_scatter import shapepost as sp


def chain_stats_at_beta_one(case_dir, n_chains=4, n_segments=5):
    """Stats dict (the keys ``ti_combine`` reads) of the production chains' Phi, with beta = 1."""
    phi = np.stack([np.concatenate([np.load(case_dir / "chains" / f"c{c}_s{j}.npz")["phi"] for j in range(n_segments)]) for c in range(n_chains)])
    acc = np.stack([np.concatenate([np.load(case_dir / "chains" / f"c{c}_s{j}.npz")["accept"] for j in range(n_segments)]) for c in range(n_chains)])
    x = phi[..., None]
    return {"beta": 1.0, "mean_phi": float(phi.mean()), "var_phi": float(phi.var(ddof=1)), "ess_phi": float(dg.ess_bulk(x, rank=False)[0]), "n_draws": int(phi.size),
            "rhat_phi": float(dg.split_rhat(x)[0]), "rhat_phi_rank": float(dg.split_rhat_rank(x)[2][0]), "acceptance": float(acc.mean()), "source": "production chains"}


def offline_runs(case_dir, last, lik_const, n_temps):
    """``ti_combine`` of every run directory ``ti_*`` (and ``ti``) of the case that holds temperatures 0 .. n_temps - 2, plus the substituted last one."""
    out = []
    for d in sorted([p for p in case_dir.iterdir() if p.is_dir() and (p.name == "ti" or p.name.startswith("ti_"))]):
        files = [d / f"t{k:02d}.npz" for k in range(n_temps - 1)]
        if not all(f.exists() for f in files):
            continue
        if (case_dir / f"{d.name}.json").exists():
            continue  # the run finished by itself: use its own file
        stats = [json.loads(str(np.load(f)["stats"])) for f in files] + [last]
        toy = type("P", (), {"lik_const": lik_const})()
        res = sp.ti_combine(toy, stats)
        res["run"] = d.name
        res["wall_s_previous_temps"] = float(sum(s.get("wall_s", 0.0) for s in stats[:-1]))
        out.append(res)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--gap", type=float, required=True)
    ap.add_argument("--model", default="F2")
    ap.add_argument("--n-temps", type=int, default=16)
    ap.add_argument("--out-dir", default=str(prun.RESULTS))
    args = ap.parse_args(argv)
    case_dir = Path(args.out_dir) / f"{prun.truth_label(args.truth, args.gap)}_{args.model}"
    z = np.load(Path(args.out_dir) / "data" / f"{prun.truth_label(args.truth, args.gap)}.npz")
    s, n_data = float(z["s"]), int(np.asarray(z["d_obs"]).size)
    lik_const = -n_data * float(np.log(np.pi * s**2))
    last = chain_stats_at_beta_one(case_dir)
    runs = offline_runs(case_dir, last, lik_const, args.n_temps)
    if not runs:
        raise SystemExit(f"no unfinished TI run with {args.n_temps - 1} temperatures in {case_dir}")
    zs = np.array([r["log_z"] for r in runs])
    out = {"case": case_dir.name, "offline": True, "note": "beta = 1 temperature taken from the production chains (4 x 500 draws); temperatures 0 .. n-2 from the TI runs", "last_temperature": last,
           "n_runs": len(runs), "log_z": float(zs.mean()), "mc_se": float(np.sqrt(np.sum([r["mc_se"] ** 2 for r in runs])) / len(runs)),
           "between_run_sd": float(zs.std(ddof=1)) if len(runs) > 1 else None, "log_z_trapezoid": float(np.mean([r["log_z_trapezoid"] for r in runs])),
           "richardson_error_estimate": float(np.mean([r["discretization"]["richardson_error_estimate"] for r in runs])),
           "converged": bool(all(r["converged"] for r in runs)), "worst_rhat": float(max(r["worst_rhat"] for r in runs)), "min_ess": float(min(r["min_ess"] for r in runs)),
           "per_run": [{"run": r["run"], "log_z": r["log_z"], "mc_se": r["mc_se"], "log_z_trapezoid": r["log_z_trapezoid"], "richardson": r["discretization"]["richardson_error_estimate"],
                        "converged": r["converged"], "worst_rhat": r["worst_rhat"], "min_ess": r["min_ess"], "monotone": r["monotone"], "acceptance_range": r["acceptance_range"]} for r in runs],
           "runs_detail": runs}
    out["schedule_ok"] = bool(all(r["schedule_ok"] for r in runs))
    out["skipped_intervals_per_run"] = {r["run"]: r["end_correction"]["skipped_intervals"] for r in runs}
    out["end_correction_skipped_would_add_per_run"] = {r["run"]: r["end_correction"]["skipped_would_add"] for r in runs}
    laplace = json.loads((case_dir / "map.json").read_text())["evidence"]
    out["laplace_gn"], out["laplace_fd"] = laplace["laplace_gn"], laplace.get("laplace_fd")
    out["ti_minus_laplace_gn"] = out["log_z"] - out["laplace_gn"]
    is_path = case_dir / "is_evidence.json"
    if is_path.exists():
        out["importance_sampling_log_z"] = json.loads(is_path.read_text())["log_z"]
        out["ti_minus_importance_sampling"] = out["log_z"] - out["importance_sampling_log_z"]
    prun.write_json(case_dir / "offline_ti.json", out)
    print(f"{out['case']}: offline TI log Z = {out['log_z']:.2f} +- {out['mc_se']:.2f} (between runs {out['between_run_sd']}), trapezoid {out['log_z_trapezoid']:.2f}, Richardson "
          f"{out['richardson_error_estimate']:.2f}; Laplace {out['laplace_gn']:.2f}; difference {out['ti_minus_laplace_gn']:+.2f}; converged {out['converged']}, schedule_ok {out['schedule_ok']}")
    return out


if __name__ == "__main__":
    main()
