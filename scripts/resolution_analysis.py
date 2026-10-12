"""Is the posterior-mean edge bias the resolution limit of the data? Bias and z against a low-passed (or cell-averaged) truth.

    python scripts/resolution_analysis.py RUN [RUN ...] [--dir results/prior_comparison] [--out FILE.json]
        [--cutoffs 2 2.5 3 3.5] [--radii 0.08 0.12]

The milestone-5 report explained the +0.2 bias of the posterior mean just outside the object either as "chosen by the prior" (milestone 4) or as
"posterior mass on rough configurations, averaging them blurs the edge". Neither is the best-supported reading (review of milestone 5): the data
fix the spatial-frequency content of the object only inside a disk of radius of order 2 k0 = 2 cycles per wavelength (more with multiple
scattering), and a posterior mean that follows the data is the truth LOW-PASSED to that band, whatever the prior is. This script quantifies
the statement for every run (summary.json and chains.npz must exist):

  * the ideal circular low-pass of theta_true at each cutoff in ``--cutoffs`` (cycles per wavelength; ``resolution.lowpass``, zero padded to
    4 n) and the moving average of theta_true over disks of each radius in ``--radii`` (``resolution.disk_average``);
  * the bias of the reference itself (reference - truth) at d = -1, +2, +4 (the signed distance to the object edge in pixels, + inside,
    ``edge_analysis.signed_distance``) and its rms over the DOI;
  * the bias of the posterior mean (mean - truth) at the same distances and its rms, the Pearson correlation of the posterior-mean bias
    map with the reference bias map (all pixels, and |d| <= 3), and the rms of mean - reference;
  * the z statistics against the reference, z = (mean - reference) / posterior std: the |z| > 3 fraction over all pixels, at d = -1 and
    in the band |d| = 1, next to the same fractions against the truth;
  * the same bias numbers for the MAP of each run (``prepared.npz``, theta_hat), which is not a posterior mean and super-resolves.

Writes FILE.json (default <dir>/resolution_analysis.json). NumPy and SciPy only.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import json
from pathlib import Path

import numpy as np

from bayes_scatter import features as ft
from bayes_scatter import resolution as rs
from edge_analysis import signed_distance

ROOT = Path(__file__).resolve().parents[1]
DISTANCES = (-1, 2, 4)


def pooled_mean_std(theta):
    """Posterior mean (n, n) and the pooled standard deviation (within-chain variance plus variance of the chain means) of draws
    ``theta`` (chains, N, n, n)."""
    mean = theta.mean(axis=(0, 1))
    std = np.sqrt(theta.var(axis=1, ddof=1).mean(axis=0) + theta.mean(axis=1).var(axis=0))
    return mean, std


def bias_row(field, truth, dist):
    """``{"d-1": ..., "d+2": ..., "d+4": ..., "rms": ...}``: mean of ``field - truth`` over the pixels at each signed distance, and its rms over all pixels."""
    b = np.asarray(field) - truth
    return {**{f"d{d:+d}": float(b[dist == d].mean()) for d in DISTANCES}, "rms": float(np.sqrt((b**2).mean()))}


def compare_to_reference(mean, std, ref, truth, chi, dist):
    """Correlations and z statistics of a posterior (mean, std) against the reference map ``ref`` (a low-passed or averaged truth)."""
    bm, br = mean - truth, ref - truth
    zr = ft.z_statistics(mean, std, ref, chi)
    zt = ft.z_statistics(mean, std, truth, chi)
    band = np.abs(dist) == 1
    zref, ztru = (mean - ref) / std, (mean - truth) / std
    return {"corr_bias_maps_all": rs.pearson(bm, br), "corr_bias_maps_abs_d_le_3": rs.pearson(bm, br, np.abs(dist) <= 3),
            "rms_mean_minus_reference": float(np.sqrt(((mean - ref) ** 2).mean())),
            "frac_abs_z_gt_3_vs_reference": zr["frac_abs_z_gt_3"], "frac_abs_z_gt_3_vs_truth": zt["frac_abs_z_gt_3"],
            "frac_abs_z_gt_3_d-1_vs_reference": float((np.abs(zref[dist == -1]) > 3).mean()),
            "frac_abs_z_gt_3_d-1_vs_truth": float((np.abs(ztru[dist == -1]) > 3).mean()),
            "frac_abs_z_gt_3_band_vs_reference": float((np.abs(zref[band]) > 3).mean()),
            "frac_abs_z_gt_3_band_vs_truth": float((np.abs(ztru[band]) > 3).mean())}


def references(truth, h, cutoffs, radii):
    """``{name: map}`` of the reference maps: ``lowpass_<k>`` for every cutoff and ``disk_<r>`` for every radius."""
    out = {f"lowpass_{k:g}": rs.lowpass(truth, h, k) for k in cutoffs}
    out.update({f"disk_{r:g}": rs.disk_average(truth, h, r) for r in radii})
    return out


def analyze(run, base, cutoffs, radii):
    """The dict of the module docstring for one run directory ``base / run``."""
    d = Path(base) / run
    summ = json.loads((d / "summary.json").read_text())
    z = np.load(d / "chains.npz")
    theta = z["theta"].astype(np.float64)
    truth, chi = z["theta_true"], z["chi_true"]
    h = summ["args"]["side"] / summ["args"]["n"]
    dist = signed_distance(chi)
    mean, std = pooled_mean_std(theta)
    refs = references(truth, h, cutoffs, radii)
    out = {"run": run, "converged": summ["convergence"]["converged"], "posterior_mean": bias_row(mean, truth, dist),
           "posterior_std_d-1": float(std[dist == -1].mean()), "references": {}}
    maps = {"posterior_mean": mean}
    if (d / "prepared.npz").exists():
        maps["MAP"] = np.load(d / "prepared.npz")["theta_hat"]
        out["MAP"] = bias_row(maps["MAP"], truth, dist)
    for name, ref in refs.items():
        e = {"reference": bias_row(ref, truth, dist), "posterior_mean_vs_reference": compare_to_reference(mean, std, ref, truth, chi, dist)}
        if "MAP" in maps:
            e["MAP_corr_bias_maps_all"] = rs.pearson(maps["MAP"] - truth, ref - truth)
            e["MAP_corr_bias_maps_abs_d_le_3"] = rs.pearson(maps["MAP"] - truth, ref - truth, np.abs(dist) <= 3)
        out["references"][name] = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dir", default=str(ROOT / "results" / "prior_comparison"))
    ap.add_argument("--out", default=None, help="default <dir>/resolution_analysis.json")
    ap.add_argument("--cutoffs", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5], help="low-pass cutoffs in cycles per wavelength")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.08, 0.12], help="disk-average radii in wavelengths")
    args = ap.parse_args(argv)
    res = {r: analyze(r, args.dir, args.cutoffs, args.radii) for r in args.runs}
    p = Path(args.out) if args.out else Path(args.dir) / "resolution_analysis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    key = f"lowpass_{3.0:g}"
    first = next(iter(res.values()))
    if key in first["references"]:
        ref = first["references"][key]["reference"]
        print(f"low-pass of the truth at 3 cycles/wavelength: bias d=-1 {ref['d-1']:+.3f}, d=+2 {ref['d+2']:+.3f}, d=+4 {ref['d+4']:+.3f}, rms {ref['rms']:.3f}")
        for r, e in res.items():
            c = e["references"][key]["posterior_mean_vs_reference"]
            print(f"  {r:26s} bias d=-1 {e['posterior_mean']['d-1']:+.3f} d=+2 {e['posterior_mean']['d+2']:+.3f} rms {e['posterior_mean']['rms']:.3f}; "
                  f"corr with low-pass bias {c['corr_bias_maps_all']:.2f} (|d|<=3 {c['corr_bias_maps_abs_d_le_3']:.2f}); "
                  f"|z|>3 vs truth {100 * c['frac_abs_z_gt_3_vs_truth']:.1f}% vs low-pass {100 * c['frac_abs_z_gt_3_vs_reference']:.1f}%")
    print(f"wrote {p}")
    return res


if __name__ == "__main__":
    main()
