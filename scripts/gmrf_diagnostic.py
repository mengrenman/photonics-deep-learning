"""Why the GMRF chains do not converge: drift of the chains and the stiffness of the Laplace coordinates.

    python scripts/gmrf_diagnostic.py [--out results/prior_comparison/gmrf_diagnostic.json]

Reads results/prior_comparison/pc_gmrf (prepared.npz, pilot_ends.npy, mass_theta_pilot.npy, chain_<c>_seg<j>.npz) and writes

  * per chain: the sampler potential Phi' = U - |zeta|^2 / 2 at the first and last step of every segment, |zeta|^2 / 2 at the
    end of every segment, the rms distance of theta from the MAP at the first and last stored draw, and the rms difference
    between the mean of the first and of the second half of all stored draws (small for a stationary chain);
  * the largest and smallest eigenvalue of H(theta_s), the Gauss-Newton + prior Hessian at each pilot end state theta_s, in
    the whitened coordinates of (a) the MAP mass, (b) the mass at the mean of the two pilot end states (the mass the chains
    were run with), (c) the mass at the pilot end state 0. The leapfrog step is limited by about 2 / sqrt(lambda_max).

``pilot_ends.npy`` and ``mass_theta_pilot.npy`` were made from the end states of the two single-chain pilots pg025_gmrf and
pg05_gmrf (results/prior_comparison), whose chain files are read here as well.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.linalg as sl

import bayes_scatter  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_prior_comparison as rpc  # noqa: E402

BASE = ROOT / "results" / "prior_comparison"
GMRF_ARGS = ["--n", "48", "--side", "2", "--eps-r", "2", "--snr", "30", "--seed", "0", "--chains", "4", "--prior", "gmrf",
             "--sigma", "0.5", "--ell", "0.4", "--out", "pc_gmrf"]


def lambda_extremes(target, mass_point, state):
    """(lambda_max, lambda_min) of L^-1 H(state) L^-T with H(mass_point) = L L^T, H the Laplace mass matrix."""
    m = target.laplace_mass(mass_point)
    h_state = target.laplace_mass(state).H
    x = sl.solve_triangular(m.chol, h_state, lower=True)
    x = sl.solve_triangular(m.chol, x.T, lower=True)
    ev = sl.eigvalsh(0.5 * (x + x.T))
    return float(ev[-1]), float(ev[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(BASE / "gmrf_diagnostic.json"))
    args = ap.parse_args(argv)
    d = BASE / "pc_gmrf"
    prep_z = np.load(d / "prepared.npz")
    theta_hat = prep_z["theta_hat"]
    chains = []
    for c in range(4):
        segs, j = [], 0
        while rpc.segment_path(d, c, j).exists():
            segs.append(rpc.load_segment(rpc.segment_path(d, c, j)))
            j += 1
        theta = np.concatenate([s["theta"] for s in segs])
        half = len(theta) // 2
        chains.append({
            "chain": c, "segments": len(segs), "draws": int(len(theta)),
            "phi_prime_first_step_of_each_segment": [float(s["phi_prime"][0]) for s in segs],
            "phi_prime_last_step_of_each_segment": [float(s["phi_prime"][-1]) for s in segs],
            "half_zeta_sq_at_end_of_each_segment": [float(0.5 * np.sum(s["xi"] ** 2)) for s in segs],
            "rms_theta_minus_map_first_draw": float(np.sqrt(np.mean((theta[0] - theta_hat) ** 2))),
            "rms_theta_minus_map_last_draw": float(np.sqrt(np.mean((theta[-1] - theta_hat) ** 2))),
            "rms_first_half_mean_minus_second_half_mean": float(np.sqrt(np.mean(
                (theta[:half].mean(axis=0) - theta[half:].mean(axis=0)) ** 2))),
            "pixel_std_over_draws_mean": float(theta.std(axis=0).mean()),
            "step": float(segs[0]["step"]), "acceptance": float(np.concatenate([s["accept"] for s in segs]).mean())})
    pilots = {}
    for name in ("pg025_gmrf", "pg05_gmrf"):
        s = rpc.load_segment(rpc.segment_path(BASE / name, 0, 0))
        pilots[name] = {"step": float(s["step"]), "acceptance": float(s["accept"].mean()),
                        "wall_s": float(s["wall_s"]), "half_zeta_sq_at_end": float(0.5 * np.sum(s["xi"] ** 2)),
                        "phi_prime_first_last": [float(s["phi_prime"][0]), float(s["phi_prime"][-1])]}
    ctx = rpc.build_context(rpc.parse_args(GMRF_ARGS))
    ends = np.load(d / "pilot_ends.npy")
    mean_end = np.load(d / "mass_theta_pilot.npy")
    stiff = {}
    for label, point in (("map_mass", theta_hat), ("pilot_mean_mass", mean_end), ("pilot_end0_mass", ends[0])):
        stiff[label] = {f"state_pilot_end_{i}": dict(zip(("lambda_max", "lambda_min"), lambda_extremes(ctx.target, point, ends[i])))
                        for i in range(len(ends))}
    out = {"chains": chains, "pilots": pilots, "stiffness_of_H_at_pilot_end_states": stiff,
           "pilot_ends_rms_distance": float(np.sqrt(np.mean((ends[0] - ends[1]) ** 2))),
           "pilot_ends_rms_from_map": [float(np.sqrt(np.mean((e - theta_hat) ** 2))) for e in ends]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    for c in chains:
        print(f"chain {c['chain']}: Phi' end of segments {[round(x) for x in c['phi_prime_last_step_of_each_segment']]}, "
              f"|zeta|^2/2 {[round(x) for x in c['half_zeta_sq_at_end_of_each_segment']]}, half-mean drift "
              f"{c['rms_first_half_mean_minus_second_half_mean']:.4f} against pixel std {c['pixel_std_over_draws_mean']:.4f}")
    for label, v in stiff.items():
        print(label, {k: (round(x["lambda_max"], 2), round(x["lambda_min"], 4)) for k, x in v.items()})
    print(f"wrote {args.out}")
    return out


if __name__ == "__main__":
    main()
