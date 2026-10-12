"""Convergence and posterior statistics of SEVERAL runs of the SAME target pooled as one set of chains (docs/DESIGN.md 12.5, 12.6).

    python scripts/cross_run_check.py RUN RUN [RUN ...] [--dir results/prior_comparison] [--last K] [--out FILE.json]

READ THIS BEFORE USING THE NUMBERS. Two runs of one prior that differ in the affine map of their chains are NOT independent samplers when one
is the re-adaptation of the other (``scripts/readapt_map.py``): the chains of the second run START at the final states of the chains of the
first (``starts.npy`` of the second run equals ``theta_final`` of one of the first's segments, max |difference| 0: the last segment for
pc_tv_lam3.75_adapt2 and pc_tv_lam11.25_adapt2, but segment 2 (draw 300 of 600) for pc_gmrf_adapt2; reported below as ``relations``).
Pooling them gives the SAME FOUR chains seen in two windows (consecutive for the TV runs, overlapping forks for the GMRF) with two
different step sizes and maps, not eight chains: R-hat over the pool is a stationarity check across the two windows, and the
effective number of independent chains is four. Runs that are not continuations of each other (different starting states, e.g. the
independent-start replication ``pc_tv_lam3.75_indep``, ``scripts/make_overdispersed_starts.py``) are independent chains, and only those support
a statement that chains from different starts agree. This script detects continuations from the run directories (a ``readapt`` entry in
``adaptation.json`` and the equality of the starts with the source's final states) and reports ``chains_independent`` accordingly.

It reads ``chains.npz`` of every run (the thinned draws and the physical Phi), keeps the LAST ``--last`` draws of every chain (default: the
smallest number of draws any run has, so the chains have equal length), and reports for the pooled chains

  * the split R-hat of Phi, the maximum over pixels of the classic and of the rank-normalized (bulk / folded) split R-hat, the number of pixels above
    1.05, and the worst-pixel multi-chain bulk ESS, for all pixels and for the pixels within one and two pixels of the object edge (signed distance
    |d| <= 1, <= 2, ``edge_analysis.signed_distance``), which are the pixels the edge question is about;
  * the posterior-mean bias (mean - truth) at d = -1 and d = +2 of each run and of the pooled chains, the pooled posterior std at d = -1 and the
    pooled |z| > 3 fraction (all pixels and the edge band): if the runs disagree on these, the numbers of the single runs are not Monte Carlo
    converged;
  * the three feature probabilities of each run and of the pooled chains.

The pooled numbers are DIAGNOSTICS: the DESIGN convergence flag stays the one of ``summary.json`` of the single run (``finalize`` refuses to
mix maps in one run directory). Writes FILE.json (default <dir>/cross_run_check.json).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

import bayes_scatter  # noqa: F401
from bayes_scatter import diagnostics as dg
from bayes_scatter import features as ft
from bayes_scatter.geometry import Grid
from edge_analysis import signed_distance

ROOT = Path(__file__).resolve().parents[1]


def load_run(base, run):
    """``(theta (C, S, n, n) float64, phi (C, S), truth theta (n, n), chi_true (n, n), summary dict)`` of a run directory."""
    d = Path(base) / run
    z = np.load(d / "chains.npz")
    return (z["theta"].astype(np.float64), z["phi_physical"].astype(np.float64), z["theta_true"], z["chi_true"],
            json.loads((d / "summary.json").read_text()))


def subset_diagnostics(theta, mask):
    """R-hat and ESS statistics of the pixels in ``mask`` ((n, n) bool) for chains ``theta`` (C, S, n, n)."""
    x = theta[:, :, mask]  # (C, S, m)
    rh = dg.split_rhat(x)
    bulk, folded, worst = dg.split_rhat_rank(x)
    ess = dg.ess_bulk(x)
    return {"pixels": int(mask.sum()), "rhat_max": float(np.nanmax(rh)), "rhat_rank_max": float(np.nanmax(worst)),
            "n_above_1.05": int((rh > 1.05).sum()), "ess_bulk_min": float(np.nanmin(ess)), "ess_bulk_median": float(np.nanmedian(ess))}


def profile_stats(theta, truth, dist):
    """Bias (mean - truth) at d = -1 and +2, the posterior std at d = -1 and the pooled |z| > 3 fraction (all pixels, d = -1) of the chains."""
    mean = theta.mean(axis=(0, 1))
    std = np.sqrt(theta.var(axis=1, ddof=1).mean(axis=0) + theta.mean(axis=1).var(axis=0))
    z = (mean - truth) / std
    out = {"bias_d-1": float((mean - truth)[dist == -1].mean()), "bias_d+2": float((mean - truth)[dist == 2].mean()),
           "std_d-1": float(std[dist == -1].mean()), "frac_abs_z_gt_3": float((np.abs(z) > 3).mean()),
           "frac_abs_z_gt_3_d-1": float((np.abs(z[dist == -1]) > 3).mean()),
           "frac_abs_z_gt_3_band": float((np.abs(z[np.abs(dist) == 1]) > 3).mean())}
    return out


def final_states(base, run):
    """``(chains, n, n)`` array of ``theta_final`` of the LAST segment file of every chain in the run directory ``base / run``."""
    d = Path(base) / run
    out = []
    c = 0
    while (d / f"chain_{c}_seg0.npz").exists():
        k = 0
        while (d / f"chain_{c}_seg{k + 1}.npz").exists():
            k += 1
        out.append(np.load(d / f"chain_{c}_seg{k}.npz")["theta_final"])
        c += 1
    return np.stack(out)


def relations(base, runs):
    """How the runs relate: ``{run: {"continues": source run, "max_abs_start_minus_source_final": float}}`` for every run that is a
    re-adaptation of another run in ``runs`` (a ``readapt`` entry of its ``adaptation.json`` naming the source, and ``starts.npy`` compared
    with the final states of the source's chains), ``{run: {"continues": None}}`` otherwise."""
    out = {}
    for r in runs:
        entry = {"continues": None}
        path = Path(base) / r / "adaptation.json"
        if path.exists():
            hist = [h for h in json.loads(path.read_text()).get("history", []) if h.get("phase") == "readapt"]
            src = hist[-1]["source_run"] if hist else None
            if src in runs and (Path(base) / r / "starts.npy").exists():
                st, fin = np.load(Path(base) / r / "starts.npy"), final_states(base, src)
                entry = {"continues": src, "max_abs_start_minus_source_final": float(np.abs(st - fin).max()) if st.shape == fin.shape else None}
        out[r] = entry
    return out


def cross_check(base, runs, last=None):
    """The dict of the module docstring for ``runs`` (names under ``base``)."""
    loaded = {r: load_run(base, r) for r in runs}
    truth, chi = loaded[runs[0]][2], loaded[runs[0]][3]
    for r in runs[1:]:
        if not (np.allclose(loaded[r][2], truth) and np.allclose(loaded[r][3], chi)):
            raise SystemExit(f"{r} has another truth than {runs[0]}: not the same problem")
    k = last or min(v[0].shape[1] for v in loaded.values())
    if k > min(v[0].shape[1] for v in loaded.values()):
        raise SystemExit("--last is larger than the smallest number of draws")
    theta = np.concatenate([v[0][:, -k:] for v in loaded.values()])  # (sum of chains, k, n, n)
    phi = np.concatenate([v[1][:, -k:] for v in loaded.values()])
    dist = signed_distance(chi)
    a0 = loaded[runs[0]][4]["args"]
    grid = Grid(a0["n"], a0["side"])
    masks = {"all": np.ones_like(chi, dtype=bool), "edge_band_abs_d_le_1": np.abs(dist) <= 1, "abs_d_le_2": np.abs(dist) <= 2}
    rel = relations(base, runs)
    independent = not any(v["continues"] for v in rel.values())
    out = {"runs": list(runs), "chains_per_run": [int(loaded[r][0].shape[0]) for r in runs], "draws_used_per_chain": int(k),
           "relations": rel, "chains_independent": independent,
           "interpretation": ("the pooled chains are independent chains" if independent else
                              "at least one run continues the chains of another (its chains start at the source's final states): the pool "
                              "is the same chains in consecutive windows, a stationarity check, not an independent replication"),
           "rhat_phi": float(dg.split_rhat(phi[..., None])[0]),
           "pooled": {name: subset_diagnostics(theta, m) for name, m in masks.items()},
           "profile_pooled": profile_stats(theta, truth, dist),
           "profile_per_run": {r: profile_stats(loaded[r][0][:, -k:], truth, dist) for r in runs},
           "features_pooled": {name: v["p"] for name, v in ft.feature_probabilities(theta, grid).items()},
           "features_per_run": {r: {name: v["p"] for name, v in ft.feature_probabilities(loaded[r][0][:, -k:], grid).items()} for r in runs},
           "note": "diagnostic only: the chains of different runs use different affine maps (and may be continuations of each other, see "
                   "relations); the DESIGN flag is the single run's summary.json"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = cross_check(args.dir, args.runs, args.last)
    p = Path(args.out) if args.out else Path(args.dir) / "cross_run_check.json"
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"{len(res['runs'])} runs, chains {res['chains_per_run']}, last {res['draws_used_per_chain']} draws: R-hat Phi {res['rhat_phi']:.4f}; "
          f"{res['interpretation']}")
    for name, s in res["pooled"].items():
        print(f"  {name:22s} {s['pixels']:5d} px: R-hat max {s['rhat_max']:.3f} (rank {s['rhat_rank_max']:.3f}), {s['n_above_1.05']} above 1.05, "
              f"bulk ESS min {s['ess_bulk_min']:.0f} median {s['ess_bulk_median']:.0f}")
    for r, s in {**res["profile_per_run"], "pooled": res["profile_pooled"]}.items():
        print(f"  {r:24s} bias d=-1 {s['bias_d-1']:.3f}  d=+2 {s['bias_d+2']:.3f}  sd d=-1 {s['std_d-1']:.3f}  |z|>3 {100 * s['frac_abs_z_gt_3']:.1f}%")
    print("  features (gap valley, ring hole, ring continuity): " + "; ".join(
        f"{r} " + "/".join(f"{v:.3f}" for v in f.values()) for r, f in {**res["features_per_run"], "pooled": res["features_pooled"]}.items()))
    print(f"wrote {p}")
    return res


if __name__ == "__main__":
    main()
