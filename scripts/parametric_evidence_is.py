"""Milestone 7 (DESIGN 14.2, evidence): an independent check of the Laplace log evidence of a (truth, model) case by importance sampling.

    python scripts/parametric_evidence_is.py --truth T1 --gap 0.05 --model F2            # <out-dir>/<case>/is_evidence.json
    python scripts/parametric_evidence_is.py --truth T1 --gap 0.05 --model F2 --draws 1000 --seed 8 --force

Why. Thermodynamic integration (``parametric_run.py --stage ti``, ``parametric_ti_offline.py``) was meant to check the Laplace evidence, but the power schedule beta_k = (k/15)^5
is too coarse at the prior end for these posteriors and its Euler-Maclaurin correction was invalid there (``shapepost.ti_combine``), so it came out 15 to 27 nats above the Laplace
value without being a measurement of it. Importance sampling needs no schedule: with a Student-t proposal (5 degrees of freedom, shape 1.2 x the Laplace covariance) at the stored
Laplace fit (<case>/map.npz) the estimate log Z = log mean_j [exp(lik_const - phi(u_j) + log pi_u(u_j)) / q(u_j)] is unbiased for Z and its error follows from the weights
(``shapepost.importance_log_evidence``). One forward solve per draw (about 0.3 s at n = 64). The output holds the IS value and error, the weight ESS (near the number of draws
when the proposal is close to the posterior), the Laplace values of the case and their difference.

Not for a case whose MAP lies on the truncation wall (T4 with F2 or F2u: the Laplace proposal is far wider than the box, ``shapepost._U_and_grad_fn``); their evidence is the profile
integral (``parametric_profile.py``). Resumable: a finished case is skipped.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import parametric_run as prun
from bayes_scatter import ellipses as el
from bayes_scatter import shapemodels as sm
from bayes_scatter import shapepost as sp
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup

print = functools.partial(print, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", required=True)
    ap.add_argument("--gap", type=float, default=0.1)
    ap.add_argument("--model", required=True, choices=prun.MODELS)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--df", type=float, default=5.0, help="degrees of freedom of the Student-t proposal")
    ap.add_argument("--scale", type=float, default=1.2, help="the proposal shape is this multiple of the Laplace covariance")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--out-dir", default=str(prun.RESULTS))
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    label = prun.truth_label(args.truth, args.gap)
    case_dir = Path(args.out_dir) / f"{label}_{args.model}"
    path = case_dir / "is_evidence.json"
    if path.exists() and not args.force:
        print(f"{case_dir.name}: importance-sampling evidence done")
        return json.loads(path.read_text())
    if not (case_dir / "map.json").exists():
        raise SystemExit(f"{case_dir / 'map.json'} does not exist: run parametric_run.py --stage map first")
    t0 = time.perf_counter()
    z = np.load(Path(args.out_dir) / "data" / f"{label}.npz")
    mz = np.load(case_dir / "map.npz")
    mj = json.loads((case_dir / "map.json").read_text())
    setup = ch15_setup(n=args.n, side=args.side, n_tx=args.n_tx, n_rx=args.n_rx)
    post = sp.ShapePosterior(sm.make_model(args.model), Forward(setup, tol=args.tol), jnp.asarray(z["d_obs"]), float(z["s"]), el.EllipseRenderer(setup.grid))
    res = sp.importance_log_evidence(post, mz["u_hat"], mz["cov"], n_draws=args.draws, df=args.df, scale=args.scale, seed=args.seed)
    ev = mj["evidence"]
    out = {"case": case_dir.name, **res, "laplace_gn": ev["laplace_gn"], "laplace_fd": ev.get("laplace_fd"), "is_minus_laplace_gn": res["log_z"] - ev["laplace_gn"],
           "is_minus_laplace_fd": None if ev.get("laplace_fd") is None else res["log_z"] - ev["laplace_fd"], "map_converged": mj["map_converged"], "time_s": time.perf_counter() - t0,
           "thread_env": prun.thread_env()}
    prun.write_json(path, out)
    print(f"{out['case']}: IS log Z {res['log_z']:.3f} +- {res['se']:.3f} (weight ESS {res['weight_ess']:.0f} of {res['n_draws']}); Laplace GN {ev['laplace_gn']:.3f}; "
          f"IS - Laplace {out['is_minus_laplace_gn']:+.3f}; {out['time_s']:.0f} s")
    return out


if __name__ == "__main__":
    main()
