"""Convergence numbers of the milestone-5 runs at EQUAL length, with the gradient cost of the window (docs/DESIGN.md 12.6).

    python scripts/equal_length_audit.py [--dir results/prior_comparison] [--null-reps 3] [--out FILE.json]

The milestone-5 report said that re-adapting the affine map from the recorded draws "helped neither" run, comparing a re-adapted run of 500 draws with the first
map's 800, and the first map's running maximum at different lengths. R-hat and ESS depend on the number of draws and on the age of the chains (the max pixel
R-hat of the TV chains at lambda* was 1.745 at 500 draws and 1.319 at 800), so this script compares the first-map run and the re-adapted run on the SAME number of
draws per chain, and gives the first-map run in several windows so that the window-to-window scatter is visible. For each (run, first, draws) entry of ``SPEC`` it
runs ``convergence_audit.audit_run`` on the draws ``first`` .. ``first + draws - 1`` of every chain and reports the classic and rank-normalized max pixel R-hat,
the number of pixels above 1.05, the worst-pixel Sokal and multi-chain bulk ESS, and the mean number of leapfrog steps per trajectory of the window (from
the segment files; a trajectory costs about that many gradient evaluations) with the bulk ESS per 1000 gradient evaluations.

Writes FILE.json (default <dir>/m5/review/equal_length_audit.json).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

import convergence_audit as ca

ROOT = Path(__file__).resolve().parents[1]
# (label, run, first draw, draws per chain): first map against the re-adapted map at equal length, and the first map in later windows
SPEC = [("TV lambda* first map", "pc_tv_lam3.75_adapt", 0, 500), ("TV lambda* first map, later window", "pc_tv_lam3.75_adapt", 300, 500),
        ("TV lambda* re-adapted", "pc_tv_lam3.75_adapt2", 0, 500),
        ("GMRF first map", "pc_gmrf_adapt", 0, 200), ("GMRF first map, window 2", "pc_gmrf_adapt", 200, 200),
        ("GMRF first map, window 3", "pc_gmrf_adapt", 400, 200), ("GMRF re-adapted", "pc_gmrf_adapt2", 0, 200),
        ("TV 3 lambda* first map", "pc_tv_lam11.25_adapt", 0, 200), ("TV 3 lambda* first map, window 2", "pc_tv_lam11.25_adapt", 200, 200),
        ("TV 3 lambda* re-adapted", "pc_tv_lam11.25_adapt2", 0, 200)]


def mean_leapfrog_in_window(run_dir, first, draws):
    """Mean over chains of the leapfrog steps per trajectory in the draws ``first`` .. ``first + draws - 1``: the segment files' ``mean_leapfrog``
    weighted by the number of window trajectories each segment holds. Raises if the segments do not cover the window."""
    out = []
    c = 0
    while (run_dir / f"chain_{c}_seg0.npz").exists():
        tot, steps, seg_start, j = 0, 0.0, 0, 0
        while (run_dir / f"chain_{c}_seg{j}.npz").exists():
            z = np.load(run_dir / f"chain_{c}_seg{j}.npz")
            n = int(z["n_samples"])
            lo, hi = max(first, seg_start), min(first + draws, seg_start + n)
            if hi > lo:
                steps += (hi - lo) * float(z["mean_leapfrog"])
                tot += hi - lo
            seg_start += n
            j += 1
        if tot != draws:
            raise SystemExit(f"{run_dir.name} chain {c}: the segments hold {tot} of the {draws} window draws")
        out.append(steps / tot)
        c += 1
    return float(np.mean(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--null-reps", type=int, default=3, help="AR(1) replicates of the audit (the null is not used here)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    base = Path(args.dir)
    rows = []
    for label, run, first, draws in SPEC:
        a = ca.audit_run(run, base, args.null_reps, first=first, draws=draws)
        L = mean_leapfrog_in_window(base / run, first, draws)
        grads = L * draws
        rows.append({"label": label, "run": run, "window": a["window"], "chains": a["chains"], "draws_per_chain": a["draws_per_chain"],
                     "rhat_pixel_max_classic": a["classic"]["rhat_pixel_max"], "rhat_pixel_max_rank": a["rank_normalized"]["pixel_max_worst"],
                     "pixels_above_1.05_rank": a["rank_normalized"]["pixels_worst_above_1.05"],
                     "ess_worst_sokal": a["ess"]["worst_pixel_sokal_summed"], "ess_worst_bulk": a["ess"]["worst_pixel_bulk_multichain"],
                     "rhat_phi_classic": a["classic"]["rhat_phi"], "mean_leapfrog_per_trajectory": L, "gradients_per_chain": grads,
                     "bulk_ess_per_1000_gradients_all_chains": 1000.0 * a["ess"]["worst_pixel_bulk_multichain"] / (grads * a["chains"])})
        r = rows[-1]
        print(f"{label:34s} draws {first:4d}..{first + draws - 1:4d}: max R-hat {r['rhat_pixel_max_classic']:.3f} (rank {r['rhat_pixel_max_rank']:.3f}, "
              f"{r['pixels_above_1.05_rank']} px > 1.05), ESS Sokal {r['ess_worst_sokal']:.0f} bulk {r['ess_worst_bulk']:.1f}, {L:.1f} leapfrog/traj, "
              f"bulk ESS per 1000 gradients {r['bulk_ess_per_1000_gradients_all_chains']:.2f}")
    p = Path(args.out) if args.out else base / "m5" / "review" / "equal_length_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"spec": rows, "note": "windows of equal length; bulk ESS per 1000 gradient evaluations pools all chains"}, indent=2, default=float))
    print(f"wrote {p}")
    return rows


if __name__ == "__main__":
    main()
