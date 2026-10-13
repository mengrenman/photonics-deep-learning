"""The detection limit of a gap against no gap, the conditioning of the Fisher matrix as g -> 0, and the ellipticity nuisance (milestone-6 review).

    python scripts/resolution_detect.py [--grid 128] [--out-dir results/resolution]

* ``at_zero``: the Fisher matrix and the Cramer-Rao standard deviations of the touching-disks object (g = 0; ``fisher.analyze`` with the exact one-sided boundary
  derivative along g) for eps_r 1.1, 2 and 4: sigma_CRB(0) with the other four parameters (r, eps_r, c_x, c_y) unknown, the condition number of F scaled to unit
  diagonal and the correlations of g with r and eps_r. The suspected small-g degeneracy (g and r trading off, as in the resolution of two point sources, Shahram and
  Milanfar) does not occur in this family: the center distance 2 r + g is never below 2 r, so F stays well conditioned as g -> 0. ``detection_limit``: the gap
  (z_0.95 + z_0.9) sigma_CRB(0) that a one-sided test of g = 0 against g > 0 at level 0.05 detects with power 0.9, at the reference 30 dB and the SNR above Smith's criterion
  sigma_CRB(g) = g that it costs. The values are compared with the extended Smith sweep (``smith_limit/crb.json``, g = 1e-4).
* ``ellipticity``: how much of sigma_CRB(g) is due to the exact circularity of the model. The two even nuisances s_x and s_y (the disks become ellipses with semi-axes
  r + s_x, r + s_y, the gap held fixed; ``shapes.stretch_derivatives``) are added to (g, r, eps_r, c_x, c_y): the bound sigma_CRB(g) with them unknown, the ratio to the
  circular one, and the shift of the SNR at which sigma_CRB(g) = g. General shape freedom is NOT covered (the span of {r, s_x, s_y} has dimension 2 only).

Writes <out-dir>/detection.json. Threads: T = 1 (about 10 minutes at n = 128) or T = 4 through ``scripts/batch_run.py``.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import fisher as fi
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.likelihood import noise_std_from_snr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)

NAMES5 = ("g", "r", "eps_r", "cx", "cy")
ELLIPSE_CASES = ((2.0, 0.05), (1.1, 0.05), (4.0, 0.15), (2.0, 0.25), (2.0, 0.0))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.9)
    return ap.parse_args(argv)


def scaled_correlations(F, names):
    """Correlation coefficients of the (Gaussian, Cramer-Rao) estimates of the parameters ``names`` from F^-1: {"a,b": C_ab / sqrt(C_aa C_bb)} for a before b."""
    F = np.asarray(F)
    C = np.linalg.inv(F)
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    return {f"{a},{b}": float(R[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if j > i}


def at_zero(fwd, eps, snr_ref):
    obj = sh.TwoDisks(g=0.0, r=0.2, eps_r=eps)
    rec = fi.analyze(fwd, obj, snr_ref_db=snr_ref, names=NAMES5, born=False)
    F = np.array(rec["F"])
    corr = scaled_correlations(F, rec["names"])
    return {"eps_r": eps, "sigma_crb_g_at_0": rec["crb_g_unknown"], "sigma_crb_g_at_0_known": rec["crb_g_known"], "cond_F_scaled": rec["cond_F_scaled"],
            "corr_g_r": corr["g,r"], "corr_g_eps_r": corr["g,eps_r"], "corr_g_cx": corr["g,cx"], "corr_g_cy": corr["g,cy"],
            "min_eigenvalue_over_max_F_scaled": float(np.linalg.eigvalsh(F / np.outer(np.sqrt(np.diag(F)), np.sqrt(np.diag(F)))).min()),
            "solver_residual_max": rec["solver_residual_max"], "F": rec["F"], "names": rec["names"]}


def ellipticity(fwd, eps, g, snr_ref):
    obj = sh.TwoDisks(g=g, r=0.2, eps_r=eps)
    chi = obj.chi(fwd.grid)
    dchi = np.concatenate([sh.shape_derivatives(fwd.grid, obj, NAMES5), sh.stretch_derivatives(fwd.grid, obj)])
    d, _ = fwd(jnp.asarray(chi))
    s = noise_std_from_snr(d, snr_ref)
    D = fi.data_derivatives(fwd, chi, dchi)
    F = fi.fisher_matrix(D, s)
    names = list(NAMES5) + ["sx", "sy"]
    sigma = {"circles": fi.crb_std(F, names, "g", NAMES5), "plus_sx": fi.crb_std(F, names, "g", NAMES5 + ("sx",)),
             "plus_sy": fi.crb_std(F, names, "g", NAMES5 + ("sy",)), "plus_sx_sy": fi.crb_std(F, names, "g", NAMES5 + ("sx", "sy"))}
    out = {"eps_r": eps, "g": g, "sigma_crb_g": sigma, "ratio_plus_sx_sy_over_circles": sigma["plus_sx_sy"] / sigma["circles"],
           "snr_shift_db": float(20.0 * np.log10(sigma["plus_sx_sy"] / sigma["circles"])),
           "corr_g_sx": scaled_correlations(F[np.ix_([0, 1, 2, 5], [0, 1, 2, 5])], ["g", "r", "eps_r", "sx"])["g,sx"],
           "cond_F_scaled_g_r_eps_sx": float(np.linalg.cond(F[np.ix_([0, 1, 2, 5], [0, 1, 2, 5])] / np.outer(np.sqrt(np.diag(F)[[0, 1, 2, 5]]), np.sqrt(np.diag(F)[[0, 1, 2, 5]]))))}
    if g > 0:
        out["snr_required_db"] = {k: float(snr_ref + 20.0 * np.log10(v / g)) for k, v in sigma.items()}
    return out


def main(argv=None):
    args = parse_args(argv)
    t0 = time.perf_counter()
    fwd = Forward(ch15_setup(n=args.grid, side=1.5), tol=1e-10, restart=100)
    snr_ref = 30.0
    zero = [at_zero(fwd, eps, snr_ref) for eps in (1.1, 2.0, 4.0)]
    smith = Path(args.out_dir) / "smith_limit" / "crb.json"
    ref = {}
    if smith.exists():
        recs = json.loads(smith.read_text())["records"]
        for z in zero:
            small = [r for r in recs.values() if r["grid"]["n"] == args.grid and r["params"]["eps_r"] == z["eps_r"] and abs(r["g"] - 1e-4) < 1e-12]
            if small:
                ref[f"{z['eps_r']:g}"] = {"sigma_crb_g_at_1e-4": small[0]["crb_g_unknown"], "rel_diff_to_g0": abs(z["sigma_crb_g_at_0"] / small[0]["crb_g_unknown"] - 1.0)}
    for z in zero:
        z["detection_limit_g"] = fi.detection_limit(z["sigma_crb_g_at_0"], args.alpha, args.power)
        z["detection_limit_over_sigma"] = z["detection_limit_g"] / z["sigma_crb_g_at_0"]
        z["detection_limit_as_wavelength_fraction"] = f"lambda/{1.0 / z['detection_limit_g']:.0f}"
        z["snr_above_smith_db"] = float(20.0 * np.log10(z["detection_limit_over_sigma"]))
    ell = [ellipticity(fwd, eps, g, snr_ref) for eps, g in ELLIPSE_CASES]
    res = {"setup": {"grid": args.grid, "snr_ref_db": snr_ref, "alpha": args.alpha, "power": args.power, "nuisances_unknown": ["r", "eps_r", "cx", "cy"],
                     "test": "one-sided Wald test of H0: g = 0 against H1: g > 0, detectable gap = (z_(1-alpha) + z_power) sigma_CRB(0)"},
           "at_zero": zero, "check_against_smith_sweep": ref, "ellipticity": ell, "time_s": time.perf_counter() - t0}
    (Path(args.out_dir) / "detection.json").write_text(json.dumps(res, indent=1))
    for z in zero:
        print(f"eps_r {z['eps_r']:g}: sigma_CRB(0) {z['sigma_crb_g_at_0']:.4e}, cond {z['cond_F_scaled']:.1f}, corr(g, r) {z['corr_g_r']:.3f}, corr(g, eps_r) {z['corr_g_eps_r']:.3f}; "
              f"detectable gap {z['detection_limit_g']:.3e} ({z['detection_limit_as_wavelength_fraction']}), {z['snr_above_smith_db']:.2f} dB above Smith")
    for e in ell:
        print(f"ellipticity eps_r {e['eps_r']:g} g {e['g']:g}: sigma ratio {e['ratio_plus_sx_sy_over_circles']:.3f}")
    return res


if __name__ == "__main__":
    main()
