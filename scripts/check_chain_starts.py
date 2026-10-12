"""Physical misfit and prior energy at the chain starts of an adapted run, for several overdispersion scales (docs/DESIGN.md 12.2).

    python scripts/check_chain_starts.py --scales 0.5 1 1.5 2 -- <the arguments of run_prior_comparison.py for the run>

The arguments after ``--`` are those of the run (problem, prior, ``--adapt``, ``--out``, optionally ``--start-theta``). After the stage
adapt, the chain stage starts chain c at ``xi_ref + scale * z_c`` in the coordinates of the adapted map (``xi_ref`` = the final state of the
adaptation chain with ``--adapt-start final``, zero with ``center``), with ``z_c`` the overdispersed N(0, I) draws of
``thetaspace.ZetaTarget.overdispersed_starts``; with ``--start-theta`` the starts are those states instead. This prints, for every scale (or the
given states), Phi, R and U of the four starts against N, so that ``--init-scale`` can be chosen before the chains are launched: a start
whose Phi is orders of magnitude above the posterior's (the milestone-4 GMRF at scale 2: 4.7e6 against N = 2400) is rejected at solver-cap
cost and wastes the warmup. One value-and-gradient-free energy evaluation per start (a forward solve), one process.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import run_prior_comparison as rpc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" not in argv:
        raise SystemExit("usage: check_chain_starts.py --scales S [S ...] -- <run_prior_comparison.py arguments>")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    own = ap.parse_args(argv[:split])
    args = rpc.parse_args(argv[split + 1:])
    if not args.adapt:
        raise SystemExit("the run must be an --adapt run")
    out_dir = Path(args.out) if Path(args.out).is_absolute() else rpc.RESULTS / args.out
    ctx = rpc.build_context(args)
    amap, ainfo = rpc.load_adaptation(out_dir, args.n)
    runner = rpc.AdaptedChainRunner(ctx, amap, ainfo, args)
    k_init, _ = rpc.chain_streams(args)

    def report(label, thetas):
        for c, th in enumerate(thetas):
            e = ctx.target.energies(jnp.asarray(th))
            print(f"{label:22s} chain {c}: Phi {e['phi']:12.1f}  R {e['R']:10.1f}  U {e['U']:12.1f}  (N = {ctx.n_data})", flush=True)

    if args.start_theta is not None:
        st = np.load(args.start_theta)
        report("start-theta file", [st if st.ndim == 2 else st[c] for c in range(args.chains)])
        return
    ea = ctx.target.energies(jnp.asarray(ainfo["theta_final"]))
    print(f"final state of the adaptation chain: Phi {ea['phi']:.1f}, R {ea['R']:.1f}, U {ea['U']:.1f}; eps {ainfo['eps']:.4f}, "
          f"L0 {ainfo['n_leapfrog0']}")
    for s in own.scales:
        xi0 = runner.zt.overdispersed_starts(k_init, args.chains, s) + runner.xi_ref
        report(f"scale {s:g}", [np.asarray(runner.zt.to_theta(xi0[c])) for c in range(args.chains)])


if __name__ == "__main__":
    main()
