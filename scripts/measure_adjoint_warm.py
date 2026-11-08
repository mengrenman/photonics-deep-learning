"""Cost of the gradient along an HMC trajectory with a cold versus a warm-started adjoint solve (docs/DESIGN.md section 16).

Problem. For each grid ``n:side`` it builds the milestone-7 inference problem: the book's rings (40 transmitters, 60
receivers), the T1 truth (two equal disks, edge-to-edge gap ``--gap``, radius ``--radius``, eps_r ``--eps-r``), data from the
grid ``2n`` at ``--snr`` dB, and the F2 shape posterior (``shapepost.ShapePosterior``) with a Gauss-Newton Laplace map at the
truth point. The forward model has tolerance ``--tol`` (1e-10, as sampling needs).

Chain. A real Metropolis HMC chain in whitened coordinates (the production kernel structure): ``--n-traj`` trajectories of
``--n-leapfrog`` leapfrog steps with step ``--step`` (1.3 is the adapted step of the milestone-7 F2 chains) jittered by
+-``--jitter``, accept or reject, and on rejection the state AND its aux are restored (as in ``samplers.hmc``). Two copies
of the chain run on the same points: COLD (aux = total fields only; the adjoint solve starts from zero) and WARM (aux = the
pair (fields, adjoint fields)). Accept decisions and momenta come from the COLD chain; the warm chain is evaluated at
exactly the same points, so both are timed on identical work, in alternating order (cold first on even points, warm first
on odd ones) to cancel drift. ``--burn`` trajectories run first and are not timed (compilation, forgetting the start).

Output. One JSON per grid (``--out`` or ``results/adjoint_warm/<host>_<tag>_<n>_<side>_s<seed>.json``) with the per-gradient wall
times of both chains in seconds (``t_cold``, ``t_warm``), the position inside the trajectory (``pos``, 0 for the first
evaluation of a trajectory), the relative gradient difference warm versus cold (``grad_rel_diff``), the value difference,
the acceptance and, with ``--counts K``, the operator applications of the adjoint solve of K transmitters
(``Forward.adjoint_matvec_counts``, cold start versus the start the warm chain used) at the first ``--count-points`` points.
A markdown summary is printed.

Run as, e.g., ``NPROC=4 python scripts/measure_adjoint_warm.py --grids 48:2,64:1.5`` (Mac, one process, 4 XLA threads),
``JAX_PLATFORMS=cuda,cpu python scripts/measure_adjoint_warm.py`` (GPU) or ``JAX_PLATFORMS=cpu NPROC=1 taskset -c 0 python
scripts/measure_adjoint_warm.py --tag cpu1`` (one core).
"""

import _threadcap  # noqa: F401  (first: caps BLAS/OpenMP threads before numpy is imported)

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_grids(text):
    out = []
    for item in text.split(","):
        n, side = item.split(":")
        out.append((int(n), float(side)))
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", type=parse_grids, default=parse_grids("48:2,64:1.5"), help="comma list of n:side")
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--radius", type=float, default=0.2)
    ap.add_argument("--eps-r", type=float, default=2.0)
    ap.add_argument("--snr", type=float, default=30.0)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--restart", type=int, default=100, help="GMRES restart length of the forward model (Forward default 100)")
    ap.add_argument("--n-tx", type=int, default=40)
    ap.add_argument("--n-rx", type=int, default=60)
    ap.add_argument("--n-traj", type=int, default=4, help="timed trajectories")
    ap.add_argument("--burn", type=int, default=1, help="untimed trajectories first")
    ap.add_argument("--n-leapfrog", type=int, default=8)
    ap.add_argument("--step", type=float, default=1.3)
    ap.add_argument("--jitter", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--counts", type=int, default=0, help="count adjoint operator applications for this many transmitters")
    ap.add_argument("--count-points", type=int, default=6, help="number of timed points at which to count")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=None, help="output JSON (only with one grid)")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "adjoint_warm"))
    return ap.parse_args(argv)


def summarize(rec):
    """Markdown lines for one grid record."""
    tc, tw, pos = np.asarray(rec["t_cold"]), np.asarray(rec["t_warm"]), np.asarray(rec["pos"])
    lines = [f"### {rec['grid']} on {rec['device']}  ({len(tc)} gradients, acceptance {rec['acceptance']:.2f})",
             "| gradients | cold median (s) | warm median (s) | cold mean (s) | warm mean (s) | mean ratio warm/cold | max grad rel diff |",
             "|---|---|---|---|---|---|---|",
             f"| all | {np.median(tc):.4f} | {np.median(tw):.4f} | {tc.mean():.4f} | {tw.mean():.4f} | {tw.mean() / tc.mean():.3f} | {max(rec['grad_rel_diff']):.2e} |"]
    for label, m in (("first of trajectory", pos == 0), ("later steps", pos > 0)):
        if m.any():
            lines.append(f"| {label} ({int(m.sum())}) | {np.median(tc[m]):.4f} | {np.median(tw[m]):.4f} | {tc[m].mean():.4f} | {tw[m].mean():.4f} | {tw[m].mean() / tc[m].mean():.3f} | |")
    if rec.get("counts"):
        c = np.asarray(rec["counts"]["cold"], float)
        w = np.asarray(rec["counts"]["warm"], float)
        lines.append(f"\nadjoint operator applications per transmitter (mean over {c.size} solves): cold {c.mean():.1f}, warm {w.mean():.1f}, ratio {w.mean() / c.mean():.3f}")
    return "\n".join(lines)


def run_grid(args, n, side):
    import jax
    import jax.numpy as jnp

    from bayes_scatter import ellipses as el
    from bayes_scatter import shapemodels as sm
    from bayes_scatter import shapepost as sp
    from bayes_scatter.adapt import AffineMap
    from bayes_scatter.forward import Forward
    from bayes_scatter.geometry import Grid, Setup, ch15_setup
    from bayes_scatter.likelihood import add_noise, misfit, noise_std_from_snr

    setup = ch15_setup(n=n, side=side, n_tx=args.n_tx, n_rx=args.n_rx)
    truth = el.make_truth("T1", g=args.gap, eps_r=args.eps_r, r=args.radius)
    data_setup = Setup(Grid(2 * n, side), setup.antennas)
    d_clean, _ = Forward(data_setup, tol=1e-10)(jnp.asarray(truth.chi(data_setup.grid)))
    s = noise_std_from_snr(d_clean, args.snr)
    d_obs = add_noise(jax.random.key(args.seed), d_clean, s)
    fwd = Forward(setup, tol=args.tol, restart=args.restart)
    rend = el.EllipseRenderer(setup.grid)
    model = sm.make_model("F2")
    post_c = sp.ShapePosterior(model, fwd, d_obs, s, rend)
    post_w = sp.ShapePosterior(model, fwd, d_obs, s, rend, warm_adjoint=True)
    u_hat = np.asarray(model.u_from_p(jnp.asarray([args.gap, args.radius, args.eps_r, 0.0, 0.0])))
    fit = sp.laplace_fit(post_c, u_hat)
    amap = AffineMap.from_covariance(u_hat, fit.cov)
    center, chol = jnp.asarray(amap.center_np.reshape(-1)), jnp.asarray(amap.chol_np)
    beta = jnp.asarray(1.0)

    phi_c = jax.jit(sp._zeta_phi_fn(post_c))
    phi_w = jax.jit(sp._zeta_phi_fn(post_w))

    def eval_c(zeta, inner):
        return phi_c(zeta, (inner, beta, center, chol, post_c.data))

    def eval_w(zeta, inner):
        return phi_w(zeta, (inner, beta, center, chol, post_w.data))

    def timed(fn, zeta, inner):
        t0 = time.perf_counter()
        phi, g, aux = fn(zeta, inner)
        jax.block_until_ready((phi, g))
        return time.perf_counter() - t0, phi, g, aux[0]

    rng = np.random.default_rng(args.seed + 1)
    d = model.dim
    z_s = jnp.zeros(d)  # chain state: whitened coordinate, Phi', gradient of Phi', aux of both copies
    phi_s, g_s, aux_c = phi_c(z_s, (post_c.initial_aux(), beta, center, chol, post_c.data))
    _, _, aux_w = phi_w(z_s, (post_w.initial_aux(), beta, center, chol, post_w.data))
    in_c_s, in_w_s = aux_c[0], aux_w[0]
    rec = {"grid": f"{n}:{side}", "n": n, "side": side, "t_cold": [], "t_warm": [], "pos": [], "grad_rel_diff": [], "phi_rel_diff": [],
           "accepted": [], "step_used": [], "counts": None}
    rec["device"] = f"{jax.devices()[0].platform}:{jax.devices()[0].device_kind}"
    count_cold, count_warm = [], []
    k_point = 0
    for traj in range(args.burn + args.n_traj):
        timed_traj = traj >= args.burn
        eps = args.step * (1.0 + args.jitter * rng.uniform(-1.0, 1.0))
        p0 = jnp.asarray(rng.standard_normal(d))
        z, in_c, in_w, g = z_s, in_c_s, in_w_s, g_s
        p = p0 - 0.5 * eps * (g + z)  # leapfrog exactly as samplers.leapfrog
        for i in range(args.n_leapfrog):
            z = z + eps * p
            out = {}
            for who in (("c", "w") if k_point % 2 == 0 else ("w", "c")):  # alternate the order to cancel drift
                out[who] = timed(eval_c, z, in_c) if who == "c" else timed(eval_w, z, in_w)
            (tc, phi_c_val, g_c, in_c_new), (tw, phi_w_val, g_w, in_w_new) = out["c"], out["w"]
            if timed_traj:
                rec["t_cold"].append(tc)
                rec["t_warm"].append(tw)
                rec["pos"].append(i)
                rec["grad_rel_diff"].append(float(jnp.linalg.norm(g_w - g_c) / jnp.linalg.norm(g_c)))
                rec["phi_rel_diff"].append(float(abs(phi_w_val - phi_c_val) / abs(phi_c_val)))
                if args.counts and len(rec["t_cold"]) <= args.count_points:
                    chi = post_c.chi(center + chol @ z)
                    ct = jax.grad(lambda uu: misfit(fwd.scattered(chi, uu), d_obs, s))(in_c_new)
                    idx = np.linspace(0, args.n_tx - 1, args.counts).astype(int)
                    count_cold += fwd.adjoint_matvec_counts(chi, ct, None, transmitters=idx).tolist()
                    count_warm += fwd.adjoint_matvec_counts(chi, ct, in_w[1], transmitters=idx).tolist()
            k_point += 1
            g, in_c, in_w = g_c, in_c_new, in_w_new
            p = p - (0.5 if i == args.n_leapfrog - 1 else 1.0) * eps * (g + z)
        h_cur = float(phi_s) + 0.5 * float(z_s @ z_s) + 0.5 * float(p0 @ p0)
        h_prop = float(phi_c_val) + 0.5 * float(z @ z) + 0.5 * float(p @ p)
        accept = bool(np.log(rng.uniform()) < h_cur - h_prop)  # NaN counts as a rejection
        if timed_traj:
            rec["accepted"].append(accept)
            rec["step_used"].append(float(eps))
        if accept:  # the aux of the accepted state is the one of its last evaluation; on rejection the old state's aux is kept
            z_s, phi_s, g_s, in_c_s, in_w_s = z, phi_c_val, g, in_c, in_w
    rec["acceptance"] = float(np.mean(rec["accepted"])) if rec["accepted"] else float("nan")
    if args.counts:
        rec["counts"] = {"cold": count_cold, "warm": count_warm}
    rec.update({"restart": args.restart, "n_tx": args.n_tx, "n_rx": args.n_rx, "tol": args.tol, "step": args.step, "n_leapfrog": args.n_leapfrog, "seed": args.seed})
    return rec


def main(argv=None):
    args = parse_args(argv)
    if args.out and len(args.grids) != 1:
        raise SystemExit("--out needs exactly one grid")
    import jax

    prov = {"host": socket.gethostname(), "platform": platform.platform(), "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
            "env": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "NPROC", "XLA_FLAGS", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
            "loadavg_start": os.getloadavg(), "argv": sys.argv}
    for n, side in args.grids:
        t0 = time.perf_counter()
        rec = run_grid(args, n, side)
        rec.update(provenance=prov, loadavg_end=os.getloadavg(), wall_s=time.perf_counter() - t0, tag=args.tag)
        out = Path(args.out) if args.out else Path(args.out_dir) / f"{socket.gethostname()}_{args.tag}_{n}_{side}_s{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        print(summarize(rec), flush=True)
        print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
