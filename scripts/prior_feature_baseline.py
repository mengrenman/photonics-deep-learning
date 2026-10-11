"""Prior probabilities of the three DESIGN 11.3 features: the baseline against which a posterior probability is read.

    python scripts/prior_feature_baseline.py [--out results/prior_comparison/prior_features.json]

A feature that a prior already makes likely is not evidence of anything when the posterior agrees. The probabilities are
those of ``features.feature_probabilities`` (delta = log(2)/2, the austria_fixed geometry on the n = 48, side 2 grid)
under the prior alone: exact draws mu + chol z for the Matern and GMRF fields (``mrf.GaussianEnergy.sample``), and HMC on
the energy R alone (``mrf.sample_prior``) for the two q-GGMRF priors of the experiment. The prior chains use the
settings of ``run_prior_comparison.prior_summary`` (4 chains, 200 warmup, 600 draws).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import bayes_scatter  # noqa: F401
from bayes_scatter import features as ft
from bayes_scatter import mrf
from bayes_scatter.geometry import Grid
from bayes_scatter.priors import GaussianField

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "results" / "prior_comparison" / "prior_features.json"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--side", type=float, default=2.0)
    ap.add_argument("--draws", type=int, default=4000, help="exact prior draws per Gaussian prior")
    args = ap.parse_args(argv)
    grid = Grid(args.n, args.side)
    out = {"n": args.n, "side": args.side, "delta": ft.DELTA}
    key = jax.random.key(0)
    for kind, kw in (("matern", {"nu": 1.5}), ("gmrf", {})):
        t0 = time.perf_counter()
        prior = mrf.GaussianEnergy(GaussianField(grid, kind, sigma=0.5, ell=0.4, **kw))
        th = np.asarray(prior.sample(key, args.draws)).reshape(4, args.draws // 4, args.n, args.n)
        feats = ft.feature_probabilities(th, grid)
        out[kind] = {"method": f"exact draws ({args.draws})", **{k: v["p"] for k, v in feats.items()},
                     "mcse": {k: v["mcse"] for k, v in feats.items()}, "time_s": time.perf_counter() - t0}
        print(kind, {k: round(v["p"], 4) for k, v in feats.items()})
    for sg in (0.05, 0.1):
        t0 = time.perf_counter()
        prior = mrf.QGGMRF(args.n, q=1.2, T=1.0, sigma_g=sg, sigma_0=1.0)
        res = mrf.sample_prior(prior, jax.random.key(1), n_chains=4, n_warmup=200, n_samples=600)
        feats = ft.feature_probabilities(res["theta_samples"], grid)
        out[f"qggmrf_sg{sg}"] = {"method": "HMC on R alone, 4 x (200 + 600)", **{k: v["p"] for k, v in feats.items()},
                                 "mcse": {k: v["mcse"] for k, v in feats.items()}, "rhat_max": res["rhat_max"],
                                 "ess_min": res["ess_min"], "time_s": time.perf_counter() - t0}
        print(f"qggmrf sigma_g {sg}", {k: round(v["p"], 4) for k, v in feats.items()})
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
