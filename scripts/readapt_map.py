"""Second-stage adaptation of the affine map from the recorded draws of a run (docs/DESIGN.md 12.2, 12.5).

    python scripts/readapt_map.py --from pc_gmrf_adapt --to pc_gmrf_adapt2 [--skip-segments K] [--max-segments K]
        [--kappa-per-dim 1.0] -- <arguments of run_prior_comparison.py for the run: problem, prior, --adapt>

Why. The adaptation chain of milestone 5 is a few hundred trajectories, shorter than the time the GMRF chain needs to spread over its slowest
directions: in the coordinates of the map that chain produced the recorded draws still drifted (|zeta|^2 / 2 rose from 3,100 to 4,500 over 200
trajectories, and the leading eigenvalues of the pooled covariance of zeta were 900, 740, 600, ..., i.e. the long-wave modes have a standard
deviation of about 30 in coordinates where the map assumed 1; HMC with trajectory length 1.5 then needs hundreds of trajectories per
independent draw along them). The recorded draws of ``--from`` are a much longer window than the adaptation had, so they are used as one more
adaptation window: ``adapt.update_map`` on the pooled draws of all chains (re-center at their mean, sample covariance S, shrinkage toward the
Laplace covariance T at the new center, Sigma_hat = (1 - w) S + w T, w = kappa / (n + kappa), kappa = ``--kappa-per-dim`` N).

What it writes (in ``results/prior_comparison/<to>/``, never in ``<from>``): copies of prepared.json / prepared.npz, ``adapted_map.npz``
(the new map, with the step size, trajectory length and cap of the source as starting values), ``adaptation.json`` (the source's history plus
a ``readapt`` entry with the update diagnostics; ``map_digest`` is the new map's) and ``starts.npy`` ((chains, n, n): the final theta of
every chain of the source, the starting states of the new chains, ``--start-theta``). The new run is then an ordinary adapted run: its chain
stage dual-averages the step size alone (``--warmup``) with the new map frozen, then records; the source's draws are warmup of the new
run and are not used by it (segments of different maps are never mixed; ``finalize`` refuses a mixture). Every recorded segment of the new run is
an exact HMC chain for the target in the coordinates of the new, fixed map.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import run_prior_comparison as rpc
from bayes_scatter import adapt as ad


def pooled_draws(per_chain):
    """``(draws, finals)`` from ``rpc.load_chain_segments`` output: ``draws`` (chains * draws, N) float64 theta states of every chain and
    segment (chain-major), ``finals`` (chains, n, n) the last state of every chain."""
    draws = np.concatenate([np.concatenate([s["theta"] for s in segs]).reshape(-1, segs[0]["theta"][0].size) for segs in per_chain])
    finals = np.stack([np.asarray(segs[-1]["theta_final"]) for segs in per_chain])
    return draws.astype(np.float64), finals


def readapt(ctx, src_dir, dst_dir, per_chain, kappa_per_dim=1.0):
    """Build and save the re-estimated map in ``dst_dir`` from the segments ``per_chain`` of the run in ``src_dir``. Returns the info dict
    of ``adapt.update_map`` (also stored in adaptation.json)."""
    n = ctx.args.n
    src_map, src_info = rpc.load_adaptation(src_dir, n)
    draws, finals = pooled_draws(per_chain)
    new_map, up = ad.update_map(draws, rpc.laplace_covariance_fn(ctx.target), kappa_per_dim * n * n, prev=src_map)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("prepared.json", "prepared.npz"):
        shutil.copy2(src_dir / name, dst_dir / name)
    np.save(dst_dir / "starts.npy", finals)
    new_map.save(dst_dir / "adapted_map.npz", eps=src_info["eps"], traj_length=src_info["traj_length"],
                 max_leapfrog=src_info["max_leapfrog"], theta_final=finals.mean(axis=0))
    info = {k: v for k, v in src_info.items() if k not in ("theta_final",)}
    info["history"] = list(src_info["history"]) + [{
        "phase": "readapt", "source_run": src_dir.name, "chains": len(per_chain), "draws_per_chain": int(draws.shape[0] // len(per_chain)),
        "segments_per_chain": [len(s) for s in per_chain], "n_draws": int(draws.shape[0]), "kappa": kappa_per_dim * n * n,
        "map_digest_before": src_info["map_digest"][:16], "updated": True, "update": up}]
    info.update({"source_run": src_dir.name, "map_digest": new_map.digest(), "kappa_readapt": kappa_per_dim * n * n,
                 "initial_map": f"the map of {src_dir.name} re-estimated from its recorded draws (scripts/readapt_map.py)",
                 "update_time_s": list(src_info.get("update_time_s", [])) + [up["time_s"]],
                 "note": "the map is frozen before any recorded draw of this run; eps is the source's, the chain stage re-tunes it per chain"})
    (dst_dir / "adaptation.json").write_text(json.dumps(info, indent=2, default=float))
    return up, new_map


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        raise SystemExit("usage: readapt_map.py --from RUN --to RUN [options] -- <run_prior_comparison.py arguments>")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="src", required=True, help="run directory (name under results/prior_comparison) with the recorded segments")
    ap.add_argument("--to", dest="dst", required=True, help="new run directory (must not exist or must hold no segment files)")
    ap.add_argument("--skip-segments", type=int, default=0, help="leave out the first K segments of every chain (burn-in)")
    ap.add_argument("--max-segments", type=int, default=None, help="use only the first K segments of every chain")
    ap.add_argument("--kappa-per-dim", type=float, default=1.0, help="pseudo-count of the Laplace covariance, in units of N")
    own = ap.parse_args(argv[:split])
    args = rpc.parse_args([*argv[split + 1:], "--stage", "adapt", "--out", own.dst])  # the new run's name wins over any --out
    if not args.adapt:
        raise SystemExit("the run must be an --adapt run")
    src = rpc.RESULTS / own.src if not Path(own.src).is_absolute() else Path(own.src)
    dst = rpc.RESULTS / own.dst if not Path(own.dst).is_absolute() else Path(own.dst)
    if dst.exists() and any(dst.glob("chain_*_seg*.npz")):
        raise SystemExit(f"{dst} already holds segment files; choose a new name")
    args.max_segments, args.skip_segments = own.max_segments, own.skip_segments
    per_chain = rpc.load_chain_segments(src, args)
    ctx = rpc.build_context(args)
    up, new_map = readapt(ctx, src, dst, per_chain, own.kappa_per_dim)
    print(f"re-estimated map from {up['n_draws']} draws of {len(per_chain)} chains (weight on the Laplace covariance {up['weight']:.3f}, "
          f"tr S / tr T {up['trace_sample_over_laplace']:.3f}); variance ratio new/old min {up['var_ratio_min']:.3g}, median "
          f"{up['var_ratio_median']:.3g}, max {up['var_ratio_max']:.3g}; center shift {up['center_shift']:.1f}; pivot ratio^2 "
          f"{up['pivot_ratio_sq']:.3g}; {up['time_s']:.1f} s")
    print(f"wrote {dst}/adapted_map.npz (digest {new_map.digest()[:16]}), adaptation.json, starts.npy, prepared.*")
    return up


if __name__ == "__main__":
    main()
