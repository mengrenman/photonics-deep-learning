"""MAP estimates from several starting points: is there one basin or several?

Same problem, flags and data as scripts/run_posterior.py (it builds the problem with that script's
``build_problem``). The reference MAP starts at the prior mean (xi = 0, as in run_posterior.py); the other starts are
draws of the prior, xi ~ N(0, I), scaled by --start-scale. For every start it reports the objective U = Phi + |xi|^2/2,
the misfit Phi, and the distance of the resulting log eps_r field from the reference MAP (rms and max over pixels,
and in units of the Laplace posterior std if --laplace-cache points at the cached eigenpairs of the reference).

With --precondition (needs --laplace-cache) each optimization gets a second stage after the --map-maxiter plain
L-BFGS iterations: it continues from there in the Laplace-whitened coordinates zeta of scripts/run_posterior_laplace.py around
the cached reference MAP, for --refine-maxiter more iterations (U(xi) = Phi' + |zeta|^2/2 exactly, so the objective is
unchanged and only the conditioning improves). Plain L-BFGS in xi stalls in the stiff directions (curvature up to 10^7)
and leaves U hundreds above its minimum after 800 iterations; started from a far prior draw, the whitened coordinates
alone do not help (the Laplace map is a local object), which is why this is a refinement stage. A different local
minimum would still be a different local minimum.

One basin is what the Laplace-preconditioned runs of run_posterior_laplace.py assume: they start every chain at, or
near, the reference MAP, so their agreement with each other probes the neighborhood of that MAP only. If starts
from far away reach the same U and the same field to a small fraction of the posterior std, cycle skipping into a
different mode is not seen at this contrast (it does not prove there is none).

Usage:  python scripts/map_multistart.py [run_posterior.py flags] --starts 6 [--start-scale 1.0]
            [--laplace-cache PATH [--precondition]] --out NAME
Writes results/<out>/multistart.json and prints one line per start.
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_posterior as rp  # noqa: E402
from bayes_scatter.inference import laplace_theta_variance, map_estimate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own = {"--starts": 6, "--start-scale": 1.0, "--laplace-cache": None, "--refine-maxiter": 400}
    precondition = "--precondition" in argv
    if precondition:
        argv.remove("--precondition")
    for flag, default in own.items():
        if flag in argv:
            i = argv.index(flag)
            own[flag] = type(default)(argv[i + 1]) if default is not None else argv[i + 1]
            del argv[i:i + 2]
    args = rp.parse_args(argv)
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / "results" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    prob = rp.build_problem(args)
    target, field = prob.target, prob.field
    t0 = time.perf_counter()
    if precondition:
        if not own["--laplace-cache"]:
            raise SystemExit("--precondition needs --laplace-cache")
        import run_posterior_laplace as lp  # noqa: E402

        c = np.load(own["--laplace-cache"])
        lt = lp.LaplaceTarget(target, jnp.asarray(c["xi_map"]), c["lam"], c["V"])

        class _ZetaProblem:  # duck-typed Target in zeta for map_estimate; U(zeta) = U(xi(zeta))
            phi_fn = lt.phi_fn
            initial_aux = staticmethod(lt.initial_aux)
            chi = staticmethod(lambda z: target.chi(lt.to_xi(z)))

    def solve(xi0):
        """Plain L-BFGS in xi, then (with --precondition) a refinement in zeta. Returns xi, info with the physical Phi."""
        xi, _, info = map_estimate(target, xi0, maxiter=args.map_maxiter)
        if precondition:
            z, _, info2 = map_estimate(_ZetaProblem, lt.to_zeta(xi), maxiter=int(own["--refine-maxiter"]))
            xi = lt.to_xi(z)
            info = {**info2, "nit": info["nit"] + info2["nit"], "nfev": info["nfev"] + info2["nfev"],
                    "time_s": info["time_s"] + info2["time_s"], "stage1_U": info["U"]}
        info["phi"] = float(target.potential(xi, target.initial_aux())[0])  # physical misfit (not Phi')
        return xi, info

    xi_ref, info_ref = solve(jnp.zeros(field.xi_shape))
    theta_ref = np.asarray(field.theta(xi_ref))
    print(f"reference MAP (from the prior mean): U {info_ref['U']:.3f}, Phi {info_ref['phi']:.1f}, "
          f"{info_ref['nit']} iterations, {info_ref['nfev']} evaluations, {info_ref['time_s']:.1f} s")

    std_lap = None
    if own["--laplace-cache"]:
        c = np.load(own["--laplace-cache"])
        std_lap = np.sqrt(laplace_theta_variance(field, c["lam"], c["V"]))
    sigma_post = float(np.mean(std_lap)) if std_lap is not None else None

    rows = []
    keys = jax.random.split(jax.random.key(args.seed + 100), int(own["--starts"]))
    for k, key in enumerate(keys):
        xi0 = own["--start-scale"] * field.sample_xi(key)
        xi_k, info = solve(xi0)
        d = np.asarray(field.theta(xi_k)) - theta_ref
        row = {"start": k, "U": info["U"], "phi": info["phi"], "dU": info["U"] - info_ref["U"],
               "rms_dtheta": float(np.sqrt(np.mean(d**2))), "max_abs_dtheta": float(np.abs(d).max()),
               "rms_dtheta_over_laplace_std": None if std_lap is None else float(np.sqrt(np.mean((d / std_lap) ** 2))),
               "nit": info["nit"], "nfev": info["nfev"], "grad_inf": info["grad_inf"], "time_s": info["time_s"],
               "message": info["message"]}
        rows.append(row)
        print(f"start {k}: U {row['U']:.3f} (dU {row['dU']:+.3f}), Phi {row['phi']:.1f}, rms dtheta {row['rms_dtheta']:.2e} "
              f"(max {row['max_abs_dtheta']:.2e})"
              + ("" if std_lap is None else f", {row['rms_dtheta_over_laplace_std']:.2f} Laplace sd")
              + f"; {row['nit']} it, {row['nfev']} evals, |grad U|_inf {row['grad_inf']:.2g}")
    summary = {"args": vars(args), "preconditioned": precondition, "start_scale": own["--start-scale"], "reference": {k: info_ref[k] for k in (
        "U", "phi", "nit", "nfev", "grad_inf", "time_s", "message")}, "mean_laplace_std_theta": sigma_post,
        "starts": rows, "wall_s": time.perf_counter() - t0}
    (out_dir / "multistart.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"wrote {out_dir / 'multistart.json'}")
    return summary


if __name__ == "__main__":
    main()
