"""Chain starts that differ from each other and from the posterior bulk, for the TV posteriors of the milestone-5 review follow-up.

    python scripts/make_overdispersed_starts.py --run pc_tv_lam1.25_adapt --kinds truth map matern gmrf
    python scripts/make_overdispersed_starts.py --run pc_tv_lam3.75_indep --copy-from pc_tv_lam3.75_adapt --kinds truth gmrf

Why. The chains of the first milestone-5 TV runs all started at 0.5 N(0, I) in the coordinates of the adapted map around ONE state of
the adaptation chain (``--init-scale 0.5``; 1.0 gave a physical Phi of 4e4 to 8e4), so their agreement (per-chain bias within 0.002,
R-hat of Phi about 1.000) says little about convergence (DESIGN 12.5, review log). Here chain c starts at one of four theta maps that are
as different from each other as the available maps are, each a plausible reconstruction of the same data:

* ``truth``: the coverage-averaged true log eps_r (sharp edges; zero bias at the edge; physical Phi 2578);
* ``map``: the MAP of THIS run's prior (``prepared.npz``; smooth-edged, bias +0.10 at d = -1 at the discrepancy lambda);
* ``matern``: the posterior mean of the converged Matern run ``pc_matern`` (blurred: bias +0.215 at d = -1);
* ``gmrf``: the last draw of chain 0 of the adapted-map GMRF run ``pc_gmrf_adapt`` (pixel-scale rough, posterior sd 0.34).

The starts are written to ``<run>/starts.npy`` ((k, n, n) float64, row c = start of chain c) for ``run_prior_comparison.py
--start-theta`` and described in ``<run>/starts.json`` (kind, source file, and the pseudo-Huber TV prior energy R and the prior misfit of
each, computed here with plain NumPy so that the script needs no XLA threads). ``--copy-from RUN`` first copies prepared.json,
prepared.npz, adapted_map.npz and adaptation.json of RUN into the run directory (a replication of RUN's posterior with other starts in a
NEW directory: the frozen map of RUN is reused, so the chains are exact for the same target and the same affine change of variables).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "prior_comparison"
KINDS = ("truth", "map", "matern", "gmrf")
COPY_FILES = ("prepared.json", "prepared.npz", "adapted_map.npz", "adaptation.json")


def tv_energy(theta, lam, delta, sigma_0=1.0, mu=0.0):
    """Pseudo-Huber TV prior energy R(theta) of ``mrf.PseudoHuberTV`` (DESIGN 12.1) in plain NumPy.

    theta (n, n) log eps_r; R = lam sum_{i~j} b_ij (sqrt(D^2 + delta^2) - delta) + sum (theta - mu)^2 / (2 sigma_0^2) over the
    8-neighborhood inside the grid (each pair once), b = 1 for horizontal and vertical pairs and 1/sqrt 2 for diagonal pairs. Returns
    ``(R, tv_part)`` with ``tv_part`` the first term.
    """
    t = np.asarray(theta, dtype=np.float64)

    def pairs(d):
        return np.sqrt(d * d + delta * delta) - delta

    tv = (pairs(t[1:, :] - t[:-1, :]).sum() + pairs(t[:, 1:] - t[:, :-1]).sum()
          + (pairs(t[1:, 1:] - t[:-1, :-1]).sum() + pairs(t[1:, :-1] - t[:-1, 1:]).sum()) / np.sqrt(2.0))
    return float(lam * tv + np.sum((t - mu) ** 2) / (2.0 * sigma_0**2)), float(lam * tv)


def load_start(kind, run_dir, base=BASE):
    """``(theta (n, n) float64, source description)`` of the start ``kind`` (module docstring)."""
    if kind == "truth":
        return np.load(base / "pc_matern" / "chains.npz")["theta_true"].astype(np.float64), "theta_true of pc_matern/chains.npz"
    if kind == "map":
        return np.load(run_dir / "prepared.npz")["theta_hat"].astype(np.float64), f"{run_dir.name}/prepared.npz theta_hat"
    if kind == "matern":
        th = np.load(base / "pc_matern" / "chains.npz")["theta"]
        return th.reshape(-1, *th.shape[-2:]).astype(np.float64).mean(axis=0), "mean of the draws of pc_matern/chains.npz"
    if kind == "gmrf":
        th = np.load(base / "pc_gmrf_adapt" / "chains.npz")["theta"]
        return th[0, -1].astype(np.float64), "last draw of chain 0 of pc_gmrf_adapt/chains.npz"
    raise ValueError(f"unknown start kind {kind!r}: one of {KINDS}")


def build(run_dir, kinds, base=BASE, copy_from=None):
    """Write ``starts.npy`` and ``starts.json`` of ``run_dir`` for the start ``kinds`` (one chain each, in order) and return
    ``(starts (k, n, n), info list)``. ``copy_from`` (a run directory name under ``base``) first copies its prepared and adapted-map files."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if copy_from is not None:
        for f in COPY_FILES:
            shutil.copyfile(base / copy_from / f, run_dir / f)
    args = json.loads((run_dir / "prepared.json").read_text())["args"]
    lam, delta, sigma_0, mu = args.get("lam", 1.0), args.get("delta", 0.01), args.get("sigma_0", 1.0), args.get("mu", 0.0)
    starts, info = [], []
    for c, kind in enumerate(kinds):
        theta, source = load_start(kind, run_dir, base)
        R, tv = tv_energy(theta, lam, delta, sigma_0, mu)
        starts.append(theta)
        info.append({"chain": c, "kind": kind, "source": source, "tv_prior_energy": R, "tv_term": tv, "theta_min": float(theta.min()),
                     "theta_max": float(theta.max()), "theta_pixel_std": float(theta.std())})
    arr = np.stack(starts)
    np.save(run_dir / "starts.npy", arr)
    (run_dir / "starts.json").write_text(json.dumps({"lam": lam, "delta": delta, "sigma_0": sigma_0, "mu": mu, "starts": info}, indent=2))
    return arr, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run directory name under results/prior_comparison (or an absolute path)")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS, help="the start of chain 0, 1, ...")
    ap.add_argument("--copy-from", default=None, help="copy prepared and adapted-map files from this run first")
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).is_absolute() else BASE / args.run
    _, info = build(run_dir, args.kinds, copy_from=args.copy_from)
    for i in info:
        print(f"chain {i['chain']}: {i['kind']:7s} R_TV {i['tv_prior_energy']:9.1f}  pixel sd {i['theta_pixel_std']:.3f}  ({i['source']})")
    print(f"wrote {run_dir}/starts.npy and starts.json")


if __name__ == "__main__":
    main()
