"""The DESIGN 13.2 sweep: Fisher information and Cramer-Rao bounds for the gap of two disks, gaps x contrasts, two grids.

    python scripts/resolution_crb.py [--grids 128 256] [--gaps 0.0125 0.025 ...] [--eps 1.1 2 4] [--out-dir results/resolution]
                                     [--force] [--aggregate-only]

Object (``bayes_scatter.shapes.TwoDisks``): two disks of radius --radius (0.2) and one relative permittivity, centered at the
origin, edge-to-edge gap g, in a DOI of side --side (1.5 wavelengths) with the book's antenna rings (40 transmitters at radius 4,
60 receivers at radius 4.5). The forward model is on an n x n grid for each n in --grids (the first is the primary grid, the last
the convergence check: 128 gives h = 0.0117 and a smallest gap of 1.07 pixels, 256 gives h = 0.0059 and 2.1 pixels; DESIGN 13.2
asks for n >= 128 and h <= 0.012). For every (grid, eps_r, g) ``fisher.analyze`` computes

  * the exact boundary derivatives of the contrast (``shapes.shape_derivatives``) along the six parameters (g, r, eps_r, cx, cy, dr),
    the data derivatives D = [dd/dp_k] as ``jax.jvp`` of the forward map (no dense Jacobian), F = (2 / s^2) Re(D^H D) at the
    reference SNR (30 dB, s from the object's own data as in DESIGN 4), and the Cramer-Rao standard deviation of g with the
    nuisance parameters of the symmetric family (r, eps_r, cx, cy) unknown (``crb_g_unknown``), with dr unknown as well
    (``crb_g_unknown_all``) and with every other parameter known (``crb_g_known``);
  * the SNR at which sigma_CRB(g) = g (Smith's criterion), in closed form from the reference F: F is proportional to 1 / s^2 and
    s^2 = mean|d|^2 10^(-SNR / 10), so sigma_CRB(SNR) = sigma_CRB(30 dB) 10^(-(SNR - 30) / 20) and
    SNR_req = 30 + 20 log10(sigma_CRB(30 dB) / g). At the primary grid this is checked by recomputing F at SNR_req
    (``required_snr_check``: sigma_CRB(g) / g must be 1);
  * the same bounds for the Born forward model (u = u_inc) at the noise level of the full model, and the ratios
    F_full / F_Born of the information about g (nuisances known: ``info_ratio_known``; effective, nuisances unknown:
    ``info_ratio_unknown`` = (sigma_Born / sigma_full)^2). These are ratios at ONE noise level and mostly follow P_full / P_Born, the ratio of the scattered
    powers; the ratios with each model at its own power SNR (``info_ratio_*_matched``, from ``--born-power``: P_full and P_Born of every object,
    <out-dir>/born_power.json) answer the question whether multiple scattering adds information per unit of scattered power;
  * the nuisance bounds at the required SNR (``nuisance_at_required``: sd of r, eps_r, cx, cy there, relative to r and eps_r - 1) and whether the bound is local
    there (``local_regime_at_required``: both relative sds <= 0.1). When it is not, sigma_CRB(g) = g at that SNR is a lower bound of unknown tightness, not a
    required SNR (the Monte Carlo maximum likelihood check ``resolution_mle.py`` measures the estimator at two such points);
  * the even / odd split of F under x -> -x (and the y -> -y split of the centered object) with the cross-block residual,
    and the column parities of D; the relative primal solver residual; the cost (forward solve, the six tangents, Born).

Every record is its own file results/resolution/crb/n<n>_eps<eps>_g<g>.json (a rerun skips the ones that exist, ``--force`` redoes them),
so an interruption loses one object. The run ends by aggregating all records into <out-dir>/crb.json (tables, the grid-convergence
check between the first and the last grid, the closed-form SNR check, the comparison with the book's claimed operating points,
timings) and the figures crb_sigma_over_g.png, crb_required_snr.png, crb_born_ratio.png and crb_grid_convergence.png.

``--book-gaps`` also computes the primary-grid records of the gaps 0.3 and 0.4 (the book's Tests #5 to #8 have spacings 0.4, 0.3, 0.2, 0.1 wavelengths) and ``--born-power``
the powers of the full and the Born data. The book's operating points (Ch. 15, Sec. 15.4.2.1, pp. 464 to 467) are Test #4: gaps 5/20 to 1/20 wavelength, NO noise level
stated; Tests #5 to #8: spacings 4/10 to 1/10 wavelength at 20 and 10 dB (``BOOK``).

Threads: run it through ``scripts/batch_run.py`` (``scripts/jobs/m6_crb.json``) with T = 4 XLA threads; the forward model at n = 256
holds 40 transmitters x 101 Krylov vectors x 256^2 complex numbers (4.2 GB).
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
import os
import time
from pathlib import Path

import jax
import numpy as np
from matplotlib.figure import Figure

import bayes_scatter  # noqa: F401  (enables float64)
from bayes_scatter import fisher as fi
from bayes_scatter import shapes as sh
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "resolution"
print = functools.partial(print, flush=True)

GAPS = (0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.32, 0.5)
EPS = (1.1, 2.0, 4.0)
COLORS = {1.1: "#2a78d6", 2.0: "#eb6834", 4.0: "#1baf7a"}  # categorical slots 1 to 3 of the reference palette
INK, MUTED = "#0b0b0b", "#52514e"
# The book's operating points (Campbell and Werner (eds.), Advances in Electromagnetics Empowered by AI and DL, Ch. 15, Sec. 15.4.2.1, pp. 464 to 467; Xiao et al.):
# Test #4 compares with VBIM on six objects with adjacent spacings 5/20 ... 1/20 wavelength and gives NO noise level; Tests #5 to #8 have spacings 4/10, 3/10, 2/10,
# 1/10 wavelength in noise-free, 20 dB and 10 dB environments (Table 15.9). The objects are NOT equal-contrast disks in vacuum (permittivity 3.2 to 2.2 in steps of 0.2 for
# Test #4, 2.2 to 3.8 for the connected objects of Tests #5 to #8), the data are 3-D (128 x 162 antenna pairs) at 500 MHz (0.6 m) with antenna planes 0.35 to 0.45 m
# (0.58 to 0.75 wavelength) from the objects: see results/resolution/book_geometry.json.
BOOK = {"test4": {"gaps": (0.25, 0.2, 0.15, 0.1, 0.05), "snr_db": (), "label": "Test #4 (VBIM comparison): spacings 5/20 to 1/20 wavelength, no noise level stated"},
        "tests5to8": {"gaps": (0.4, 0.3, 0.2, 0.1), "snr_db": (20.0, 10.0), "label": "Tests #5 to #8: spacings 4/10 to 1/10 wavelength at 20 and 10 dB"}}
BOOK_EXTRA_GAPS = (0.3, 0.4)  # gaps of Tests #5 to #8 that are not in the sweep GAPS (primary grid only, ``--book-gaps``)
N_DATA = 40 * 60  # complex data of the book rings used here (the book's 3-D data: 128 x 162 = 20736)


def tag(n, eps, g):
    """File stem of the record of grid n, permittivity eps and gap g."""
    return f"n{n}_eps{eps:g}_g{g:g}"


def record_path(out_dir, n, eps, g):
    return Path(out_dir) / "crb" / f"{tag(n, eps, g)}.json"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--grids", type=int, nargs="+", default=[128, 256], help="forward-model grid sizes; first = primary, last = convergence check")
    ap.add_argument("--gaps", type=float, nargs="+", default=list(GAPS))
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS))
    ap.add_argument("--radius", type=float, default=0.2, help="common radius of the two disks")
    ap.add_argument("--side", type=float, default=1.5)
    ap.add_argument("--snr", type=float, default=30.0, help="reference power SNR in dB")
    ap.add_argument("--tol", type=float, default=1e-10, help="forward solver relative tolerance")
    ap.add_argument("--restart", type=int, default=100)
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--book-gaps", action="store_true", help="also compute the primary-grid records of the gaps 0.3 and 0.4 (Tests #5 to #8 of the book)")
    ap.add_argument("--born-power", action="store_true", help="compute P_full and P_Born (forward solves only) of every record's object into <out-dir>/born_power.json")
    ap.add_argument("--force", action="store_true", help="recompute records that exist")
    ap.add_argument("--aggregate-only", action="store_true", help="only aggregate the records on disk")
    args = ap.parse_args(argv)
    if len(set(args.grids)) != len(args.grids) or any(n < 16 for n in args.grids):
        ap.error("--grids must be distinct sizes >= 16")
    return args


# ----------------------------------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------------------------------


def make_record(fwd, n, eps, g, args):
    """The ``fisher.analyze`` dict of one object plus its labels; at the primary grid also the closed-form SNR check."""
    obj = sh.TwoDisks(g=g, r=args.radius, eps_r=eps)
    rec = fi.analyze(fwd, obj, snr_ref_db=args.snr)
    rec.update({"n": n, "eps_r": eps, "g": g, "radius": args.radius, "thread_env": {k: os.environ.get(k) for k in ("NPROC", "XLA_FLAGS", "OMP_NUM_THREADS")},
                "load_avg_1min": os.getloadavg()[0]})
    if n == args.grids[0]:
        rec["required_snr_check"] = required_snr_check(fwd, obj, rec, args)
    return rec


def required_snr_check(fwd, obj, rec, args):
    """Recompute the bounds at the closed-form required SNR of each bound and report sigma_CRB(g) / g, which must be 1.

    The recomputation sets the noise level from that SNR (``likelihood.noise_std_from_snr`` of the object's own data) and redoes F,
    so it tests the scaling F ~ 1 / s^2 and the closed form, not an identity of the formula.
    """
    out = {}
    for key, field in (("unknown", "crb_g_unknown"), ("known", "crb_g_known")):
        snr_req = rec["required_snr_db"][key]
        again = fi.analyze(fwd, obj, snr_ref_db=snr_req, born=False)
        out[key] = {"snr_required_db": snr_req, "sigma_over_g_recomputed": again[field] / obj.g}
    return out


def run_records(args):
    out_dir = Path(args.out_dir)
    (out_dir / "crb").mkdir(parents=True, exist_ok=True)
    setup_kw = dict(side=args.side)
    for n in args.grids:
        gaps = list(args.gaps) + ([g for g in BOOK_EXTRA_GAPS if g not in args.gaps] if args.book_gaps and n == args.grids[0] else [])
        todo = [(eps, g) for eps in args.eps for g in gaps if args.force or not record_path(out_dir, n, eps, g).exists()]
        if not todo:
            print(f"n = {n}: all {len(args.eps) * len(gaps)} records exist")
            continue
        t0 = time.perf_counter()
        fwd = Forward(ch15_setup(n=n, **setup_kw), tol=args.tol, restart=args.restart)
        print(f"n = {n} (h = {fwd.grid.h:.5f}, {fwd.n_tx} x {fwd.n_rx} antennas): {len(todo)} records to compute")
        for eps, g in todo:
            t1 = time.perf_counter()
            rec = make_record(fwd, n, eps, g, args)
            record_path(out_dir, n, eps, g).write_text(json.dumps(rec, indent=1, default=float))
            print(f"  n={n} eps_r={eps:g} g={g:g}: sigma_CRB(g)/g {rec['crb_g_unknown'] / g:.3e} (known {rec['crb_g_known'] / g:.3e}), "
                  f"required SNR {rec['required_snr_db']['unknown']:.1f} dB, Born ratio {rec['info_ratio_known']:.3f}, cross {rec['irreps']['cross_residual']:.1e}, "
                  f"residual {rec['solver_residual_max']:.1e}, forward {rec['timing_s']['forward_s']:.2f} s, tangents {rec['timing_s']['tangents_s']:.2f} s "
                  f"[{time.perf_counter() - t1:.1f} s]")
        print(f"n = {n}: {time.perf_counter() - t0:.0f} s")
        del fwd
        jax.clear_caches()


# ----------------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------------


def load_records(args):
    """{(n, eps, g): record} of every record on disk that matches the sweep arguments."""
    out_dir, recs = Path(args.out_dir), {}
    for n in args.grids:
        for eps in args.eps:
            for g in list(args.gaps) + ([g for g in BOOK_EXTRA_GAPS if g not in args.gaps] if n == args.grids[0] else []):
                p = record_path(out_dir, n, eps, g)
                if p.exists():
                    recs[(n, eps, g)] = json.loads(p.read_text())
    return recs


def born_power_path(out_dir):
    return Path(out_dir) / "born_power.json"


def born_power_records(args):
    """P_full and P_Born (mean |d|^2 of the full and the Born data, forward solves only) of the object of every record on disk -> <out-dir>/born_power.json.

    {tag: {n, eps_r, g, power_full, power_born, power_ratio_full_over_born, power_full_in_record, rel_diff_power_full}}: ``power_full`` is recomputed and compared with the
    record's own ``power`` (the regression check that the solver and the rendering are those of the sweep). A tag already in the file is skipped (``--force`` redoes)."""
    out_dir = Path(args.out_dir)
    path = born_power_path(out_dir)
    data = {} if args.force or not path.exists() else json.loads(path.read_text())
    recs = load_records(args)
    for n in args.grids:
        todo = [(eps, g) for (m, eps, g) in sorted(recs) if m == n and tag(n, eps, g) not in data]
        if not todo:
            continue
        t0 = time.perf_counter()
        fwd = Forward(ch15_setup(n=n, side=args.side), tol=args.tol, restart=args.restart)
        for eps, g in todo:
            obj = sh.TwoDisks(g=g, r=args.radius, eps_r=eps)
            chi = jax.numpy.asarray(obj.chi(fwd.grid))
            d, _ = fwd(chi)
            db = fwd.born(chi)
            pf, pb = float(jax.numpy.mean(jax.numpy.abs(d) ** 2)), float(jax.numpy.mean(jax.numpy.abs(db) ** 2))
            rec_power = recs[(n, eps, g)]["power"]
            data[tag(n, eps, g)] = {"n": n, "eps_r": eps, "g": g, "power_full": pf, "power_born": pb, "power_ratio_full_over_born": pf / pb,
                                    "power_full_in_record": rec_power, "rel_diff_power_full": abs(pf / rec_power - 1.0)}
        print(f"n = {n}: P_full and P_Born of {len(todo)} objects, {time.perf_counter() - t0:.0f} s")
        del fwd
        jax.clear_caches()
    path.write_text(json.dumps(data, indent=1))
    return data


def table_row(rec, born=None):
    """The headline numbers of one record (what the tables and figures use).

    ``born``: the entry of ``born_power.json`` for this record (or None; a record written by ``fisher.analyze`` with ``power_born`` carries it itself). Adds the matched-power
    Born ratios and, at the required SNR (nuisances unknown), the Cramer-Rao standard deviations of the nuisance parameters, relative to r and eps_r - 1, and
    ``local_regime_at_required`` (``fisher.nuisance_report``)."""
    g = rec["g"]
    pb = rec.get("power_born") if rec.get("power_born") is not None else (None if born is None else born["power_born"])
    matched = {"power_born": pb, "power_ratio_full_over_born": None, "info_ratio_known_matched": None, "info_ratio_unknown_matched": None}
    if pb is not None:
        km, um, pr = fi.matched_power_ratios(rec["info_ratio_known"], rec["info_ratio_unknown"], rec["power"], pb)
        matched.update({"power_ratio_full_over_born": float(pr), "info_ratio_known_matched": float(km), "info_ratio_unknown_matched": float(um)})
    nuis = fi.nuisance_report(rec["F"], rec["names"], rec["required_snr_db"]["unknown"], rec["params"], snr_ref_db=rec["snr_ref_db"], n_data=N_DATA)
    return {"g": g, "eps_r": rec["eps_r"], "n": rec["n"],
            "sigma_over_g_unknown": rec["crb_g_unknown"] / g, "sigma_over_g_unknown_all": rec["crb_g_unknown_all"] / g,
            "sigma_over_g_known": rec["crb_g_known"] / g,
            "sigma_unknown": rec["crb_g_unknown"], "sigma_known": rec["crb_g_known"],
            "required_snr_db": rec["required_snr_db"],
            "nuisance_at_required": nuis, "local_regime_at_required": nuis["local_regime"],
            "nuisance_at_book_snr": {f"{snr:g}": fi.nuisance_report(rec["F"], rec["names"], snr, rec["params"], snr_ref_db=rec["snr_ref_db"], n_data=N_DATA)
                                     for snr in sorted({x for grp in BOOK.values() for x in grp["snr_db"]})},
            "sigma_over_g_unknown_born": rec["crb_g_unknown_born"] / g, "sigma_over_g_known_born": rec["crb_g_known_born"] / g,
            "info_ratio_known": rec["info_ratio_known"], "info_ratio_unknown": rec["info_ratio_unknown"], **matched,
            "cond_F_scaled": rec["cond_F_scaled"], "cross_residual_x": rec["irreps"]["cross_residual"],
            "cross_residual_y": rec["irreps_y"]["cross_residual"], "parity_x_max": max(rec["parity_x"].values()),
            "parity_y_max": max(rec["parity_y"].values()), "solver_residual_max": rec["solver_residual_max"],
            "power": rec["power"], "noise_std": rec["s"]}


def convergence(recs, grids):
    """Relative differences between the first and the last grid, per (eps, g): of sigma_CRB(g) (unknown, known), of the Born ratio and of
    the largest entry of F scaled to unit diagonal; and the maxima."""
    if len(grids) < 2:
        return {"note": "a single grid: no convergence check"}
    a, b = grids[0], grids[-1]
    rows, worst = [], {}
    for (n, eps, g), ra in sorted(recs.items()):
        if n != a or (b, eps, g) not in recs:
            continue
        rb = recs[(b, eps, g)]
        Fa, Fb = np.array(ra["F"]), np.array(rb["F"])
        da, db = np.sqrt(np.diag(Fa)), np.sqrt(np.diag(Fb))
        row = {"eps_r": eps, "g": g,
               "rel_diff_sigma_unknown": abs(ra["crb_g_unknown"] / rb["crb_g_unknown"] - 1.0),
               "rel_diff_sigma_known": abs(ra["crb_g_known"] / rb["crb_g_known"] - 1.0),
               "rel_diff_info_ratio_known": abs(ra["info_ratio_known"] / rb["info_ratio_known"] - 1.0),
               "rel_diff_required_snr_db_abs": abs(ra["required_snr_db"]["unknown"] - rb["required_snr_db"]["unknown"]),
               "max_rel_diff_F_diag": float(np.max(np.abs(np.diag(Fa) / np.diag(Fb) - 1.0))),
               "max_abs_diff_F_scaled": float(np.max(np.abs(Fa / np.outer(da, da) - Fb / np.outer(db, db))))}
        rows.append(row)
    for key in ("rel_diff_sigma_unknown", "rel_diff_sigma_known", "rel_diff_info_ratio_known", "rel_diff_required_snr_db_abs",
                "max_rel_diff_F_diag", "max_abs_diff_F_scaled"):
        if rows:
            worst[key] = max(r[key] for r in rows)
    worst_by_gap = {f"{g:g}": max(r["rel_diff_sigma_unknown"] for r in rows if r["g"] == g) for g in sorted({r["g"] for r in rows})}
    return {"grids": [a, b], "rows": rows, "max": worst, "max_rel_diff_sigma_unknown_by_gap": worst_by_gap}


def book_claims(rows, eps_values):
    """The CRB at the book's operating points (``BOOK``), per contrast, from the primary-grid rows (the sweep gaps and the extra gaps 0.3 and 0.4 where computed).

    Per test group and (eps_r, gap): ``sigma_over_g_30db`` (unknown / known), the required SNR and, there, the nuisance bounds and whether the bound is local
    (``local_regime_at_required``); for a group with noise levels (Tests #5 to #8: 20 and 10 dB) also, per SNR, sigma_CRB(SNR) / g = (sigma_CRB(30 dB) / g)
    10^(-(SNR - 30) / 20), ``margin_over_required_db`` = SNR - SNR_req (not a margin of safety when the bound is not local: then it compares an operating point with a
    LOWER BOUND of unknown tightness), ``nuisance_rel_sd`` and ``bound_is_local_there`` (the nuisance bounds AT that SNR, ``fisher.nuisance_report``: at the book's SNRs
    they are local, so sigma_CRB / g THERE is a statement about an estimator of the exactly known model; the required SNR below it is not). Test #4 has no noise level in the book
    (``snr_db`` empty): only the required SNRs are given. A value of sigma_CRB / g below 1 means that Smith's criterion holds for a perfectly known two-disk model.
    """
    out = {"source": "Campbell and Werner (eds.), Advances in Electromagnetics Empowered by AI and DL, Ch. 15, Sec. 15.4.2.1, pp. 464-467", "groups": {}}
    for name, grp in BOOK.items():
        res = {"label": grp["label"], "gaps": list(grp["gaps"]), "snr_db": list(grp["snr_db"]), "by_eps": {}}
        for eps in eps_values:
            per_gap = {}
            for g in grp["gaps"]:
                row = next((r for r in rows if r["eps_r"] == eps and abs(r["g"] - g) < 1e-12), None)
                if row is None:
                    continue
                snr_req = row["required_snr_db"]["unknown"]
                entry = {"sigma_over_g_30db": {"unknown": row["sigma_over_g_unknown"], "known": row["sigma_over_g_known"]},
                         "required_snr_db": {"unknown": snr_req, "known": row["required_snr_db"]["known"]},
                         "nuisance_at_required": row["nuisance_at_required"], "local_regime_at_required": row["local_regime_at_required"], "snr": {}}
                for snr in grp["snr_db"]:
                    f = 10.0 ** (-(snr - 30.0) / 20.0)
                    nb = row["nuisance_at_book_snr"][f"{snr:g}"]
                    entry["snr"][f"{snr:g}"] = {"sigma_over_g": {"unknown": row["sigma_over_g_unknown"] * f, "known": row["sigma_over_g_known"] * f},
                                                "margin_over_required_db": snr - snr_req, "nuisance_rel_sd": nb["rel_sd"], "bound_is_local_there": nb["local_regime"]}
                per_gap[f"{g:g}"] = entry
            res["by_eps"][f"{eps:g}"] = per_gap
        out["groups"][name] = res
    return out


def aggregate(args):
    recs = load_records(args)
    primary = args.grids[0]
    bp = born_power_path(args.out_dir)
    born = json.loads(bp.read_text()) if bp.exists() else {}
    all_rows = [table_row(r, born.get(tag(n, eps, g))) for (n, eps, g), r in sorted(recs.items()) if n == primary]
    rows = [r for r in all_rows if r["g"] in args.gaps]  # the sweep (tables, figures); the extra book gaps enter ``book_claims`` only
    born_checks = [v["rel_diff_power_full"] for v in born.values()]
    timings = {}
    for n in args.grids:
        rs = [r for (m, _, _), r in recs.items() if m == n]
        if rs:
            timings[str(n)] = {"records": len(rs), "h": rs[0]["grid"]["h"], **{k: {"median": float(np.median([r["timing_s"][k] for r in rs])),
                                                                                  "max": float(np.max([r["timing_s"][k] for r in rs]))}
                                                                               for k in ("forward_s", "tangents_s", "born_s")}}
    checks = [r["required_snr_check"] for (n, _, _), r in recs.items() if n == primary and "required_snr_check" in r]
    snr_check = {"n_checked": len(checks),
                 "max_abs_sigma_over_g_minus_1": float(max((abs(c[k]["sigma_over_g_recomputed"] - 1.0) for c in checks for k in c), default=float("nan")))}
    out = {"setup": {"side": args.side, "radius": args.radius, "antennas": "40 transmitters at radius 4, 60 receivers at radius 4.5 (book rings)",
                     "gaps": list(args.gaps), "eps_r": list(args.eps), "grids": args.grids, "snr_ref_db": args.snr, "solver_tol": args.tol,
                     "restart": args.restart, "parameters": list(sh.PARAM_NAMES),
                     "symmetric_family": list(fi.SYMMETRIC_PARAMS), "nuisances_unknown": [k for k in fi.SYMMETRIC_PARAMS if k != "g"]},
           "formulas": {"fisher": "F = (2 / s^2) Re(D^H D), D = [dd/dp_k] = jvp of the forward map along the exact shape derivative d chi / d p_k",
                        "noise": "s^2 = mean|d_clean|^2 10^(-SNR/10), circular complex Gaussian, E|e|^2 = s^2",
                        "scaling": "F(SNR) = F(30 dB) 10^((SNR - 30)/10); sigma_CRB(SNR) = sigma_CRB(30 dB) 10^(-(SNR - 30)/20)",
                        "required_snr": "SNR_req(g) = 30 + 20 log10(sigma_CRB(30 dB) / g), sigma_CRB(g) <= g (Smith 2005)",
                        "born_ratio": "info_ratio_known = F_gg / F_gg(Born); info_ratio_unknown = (sigma_CRB,Born / sigma_CRB)^2, both at the noise level of the full model",
                        "born_ratio_matched": "info_ratio_*_matched = info_ratio_* x P_Born / P_full: each model at its own power SNR (s^2 = mean|d|^2 10^(-SNR/10))",
                        "local_regime": "the bound at SNR_req is read as local if sd(r)/r and sd(eps_r)/(eps_r - 1) at that SNR are both <= 0.1; otherwise sigma_CRB(g) = g there "
                                        "is a lower bound of unknown tightness, not a required SNR"},
           "table": rows, "n_data": N_DATA, "convergence": convergence(recs, args.grids), "required_snr_check": snr_check,
           "book_claims": book_claims(all_rows, args.eps), "timings": timings,
           "local_regime": {"max_rel_sd": fi.LOCAL_REGIME_MAX_REL_SD, "n_rows": len(rows), "n_local_at_required": int(sum(r["local_regime_at_required"] for r in rows)),
                            "rows_outside": [{"eps_r": r["eps_r"], "g": r["g"], "required_snr_db": r["required_snr_db"]["unknown"],
                                              "rel_sd": r["nuisance_at_required"]["rel_sd"]} for r in rows if not r["local_regime_at_required"]]},
           "born_power": {"n_objects": len(born), "max_rel_diff_power_full_vs_record": max(born_checks) if born_checks else None},
           "symmetry": {"max_cross_residual_x": max((r["cross_residual_x"] for r in rows), default=None),
                        "max_cross_residual_y": max((r["cross_residual_y"] for r in rows), default=None),
                        "max_parity_x": max((r["parity_x_max"] for r in rows), default=None),
                        "max_parity_y": max((r["parity_y_max"] for r in rows), default=None),
                        "max_solver_residual": max((r["solver_residual_max"] for r in rows), default=None)},
           "records": {tag(n, eps, g): r for (n, eps, g), r in sorted(recs.items())}}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crb.json").write_text(json.dumps(out, indent=1, default=float))
    if rows:
        make_figures(rows, out["convergence"], args, out_dir)
    print(f"wrote {out_dir / 'crb.json'} ({len(rows)} rows at n = {primary}, {len(recs)} records)")
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------


def style(ax, xlabel, ylabel):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color="#e4e3dd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)


def gap_axis(ax, gaps=(0.0125, 0.025, 0.05, 0.1, 0.2, 0.5)):
    """Log x axis with plain tick labels at round gaps (the default minor labels of a log axis overlap)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [g for g in gaps if lo <= g <= hi]
    if len(ticks) < 3 or hi / lo > 100:  # a sweep over other gaps or over more than two decades (the extended sweep from 1e-4 wavelengths): 1, 2, 5 times powers of ten
        ticks = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 2) for m in (1, 2, 5) if lo <= m * 10.0 ** k <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def series(rows, eps, key, sub=None):
    """(g, values) of ``key`` (or ``key[sub]`` for a dict-valued entry) for the rows of one contrast, sorted by gap."""
    r = sorted((r for r in rows if r["eps_r"] == eps), key=lambda r: r["g"])
    return np.array([x["g"] for x in r]), np.array([x[key] if sub is None else x[key][sub] for x in r])


def make_figures(rows, conv, args, out_dir):
    eps_values = sorted({r["eps_r"] for r in rows})
    # 1. sigma_CRB / g against g, one panel per contrast, with the Smith line and the 10 / 20 dB shifts
    fig = Figure(figsize=(3.6 * len(eps_values) + 0.4, 3.7), dpi=130)
    axes = fig.subplots(1, len(eps_values), sharey=True, squeeze=False)[0]
    for ax, eps in zip(axes, eps_values):
        g, unk = series(rows, eps, "sigma_over_g_unknown")
        _, kn = series(rows, eps, "sigma_over_g_known")
        c = COLORS.get(eps, "#2a78d6")
        for snr, lw, ls in ((30.0, 1.8, "-"), (20.0, 1.2, "--"), (10.0, 1.2, ":")):
            ax.loglog(g, unk * 10.0 ** (-(snr - args.snr) / 20.0), color=c, lw=lw, ls=ls, marker="o" if snr == args.snr else None, ms=4,
                      label=f"{snr:g} dB, nuisances unknown")
        ax.loglog(g, kn, color=INK, lw=1.0, ls="-", alpha=0.55, label="30 dB, nuisances known")
        ax.axhline(1.0, color=MUTED, lw=0.9)
        ax.text(g[0], 1.15, "sigma = g (Smith)", color=MUTED, fontsize=8, va="bottom")
        style(ax, "gap g (wavelengths)", "sigma_CRB(g) / g" if ax is axes[0] else "")
        gap_axis(ax)
        ax.set_title(f"eps_r = {eps:g}", color=INK, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "crb_sigma_over_g.png")
    # 2. required SNR against g: filled markers where the bound is local at that SNR (nuisance sds <= 0.1 of r and eps_r - 1), open markers where it is a lower bound of
    #    unknown tightness; the book's operating points (Tests #5 to #8: gaps 0.1 to 0.4 at 10 and 20 dB; Test #4: gaps 0.05 to 0.25, no noise level stated)
    fig = Figure(figsize=(6.8, 4.5), dpi=130)
    ax = fig.subplots()
    t58 = BOOK["tests5to8"]
    for snr in t58["snr_db"]:
        ax.plot(t58["gaps"], [snr] * len(t58["gaps"]), ls="none", marker="D", ms=6, mfc="#e4e3dd", mec=INK, mew=0.8, zorder=3)
    ax.text(np.sqrt(min(t58["gaps"]) * max(t58["gaps"])), max(t58["snr_db"]) + 1.4, "book Tests #5 to #8:\nspacings 1/10 to 4/10 at 20 and 10 dB", color=INK, fontsize=8, ha="center", va="bottom")
    t4 = BOOK["test4"]
    ax.annotate("", xy=(min(t4["gaps"]), args.snr + 5.0), xytext=(max(t4["gaps"]), args.snr + 5.0), arrowprops={"arrowstyle": "|-|", "color": INK, "lw": 1.0})
    ax.text(np.sqrt(min(t4["gaps"]) * max(t4["gaps"])), args.snr + 5.4, "book Test #4: spacings 1/20 to 5/20, no noise level stated", color=INK, fontsize=8, ha="center", va="bottom")
    from matplotlib.lines import Line2D

    for eps in eps_values:
        g, unk = series(rows, eps, "required_snr_db", "unknown")
        _, kn = series(rows, eps, "required_snr_db", "known")
        _, loc = series(rows, eps, "local_regime_at_required")
        c = COLORS.get(eps, "#2a78d6")
        ax.semilogx(g, unk, color=c, lw=1.0, ls=":", zorder=1)
        ax.semilogx(g[loc], unk[loc], color=c, lw=0, marker="o", ms=5)
        ax.semilogx(g[~loc], unk[~loc], color=c, lw=0, marker="o", ms=5, mfc="white")
        ax.semilogx(g, kn, color=c, lw=0.8, ls="--", alpha=0.6)
    handles = [Line2D([], [], color=COLORS.get(eps, "#2a78d6"), lw=1.5, label=f"eps_r = {eps:g}") for eps in eps_values]
    handles += [Line2D([], [], color=INK, lw=0, marker="o", ms=5, label="filled: nuisance bounds <= 0.1 of r and eps_r - 1 at that SNR (local)"),
                Line2D([], [], color=INK, lw=0, marker="o", ms=5, mfc="white", label="open: not local, a lower bound of unknown tightness"),
                Line2D([], [], color=INK, lw=0.8, ls="--", label="dashed: nuisances known")]
    ax.axhline(args.snr, color=MUTED, lw=0.9)
    ax.text(0.02, 0.60, f"reference {args.snr:g} dB", color=MUTED, fontsize=8, transform=ax.transAxes, va="top")
    style(ax, "gap g (wavelengths)", "SNR at which sigma_CRB(g) = g (dB)")
    gap_axis(ax)
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower left")
    ax.set_ylim(top=args.snr + 12.0)
    fig.tight_layout()
    fig.savefig(out_dir / "crb_required_snr.png")
    # 3. full-over-Born information about g: at one noise level, with each model at its own power SNR, and the ratio of the scattered powers
    fig = Figure(figsize=(11.0, 3.8), dpi=130)
    axes = fig.subplots(1, 3, sharex=True)
    panels = (("info_ratio_known", "info_ratio_unknown", "same noise level s", "full / Born information about g"),
              ("info_ratio_known_matched", "info_ratio_unknown_matched", "each model at its own power SNR", "full / Born information about g"),
              ("power_ratio_full_over_born", None, "scattered power", "P_full / P_Born"))
    for ax, (k_known, k_unk, title, ylabel) in zip(axes, panels):
        for eps in eps_values:
            c = COLORS.get(eps, "#2a78d6")
            if any(r[k_known] is None for r in rows if r["eps_r"] == eps):
                continue
            g, kn = series(rows, eps, k_known)
            ax.semilogx(g, kn, color=c, lw=1.8, marker="o", ms=4, label=f"eps_r = {eps:g}" + (", nuisances known" if k_unk else ""))
            if k_unk:
                _, un = series(rows, eps, k_unk)
                ax.semilogx(g, un, color=c, lw=1.0, ls="--", marker="s", ms=3, label=f"eps_r = {eps:g}, nuisances unknown")
        ax.axhline(1.0, color=MUTED, lw=0.9)
        style(ax, "gap g (wavelengths)", ylabel)
        gap_axis(ax)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_dir / "crb_born_ratio.png")
    # 4. grid convergence
    if conv.get("rows"):
        fig = Figure(figsize=(6.2, 3.6), dpi=130)
        ax = fig.subplots()
        for eps in eps_values:
            rr = sorted((r for r in conv["rows"] if r["eps_r"] == eps), key=lambda r: r["g"])
            ax.loglog([r["g"] for r in rr], [max(r["rel_diff_sigma_unknown"], 1e-8) for r in rr], color=COLORS.get(eps, "#2a78d6"), lw=1.6, marker="o", ms=4,
                      label=f"eps_r = {eps:g}")
        style(ax, "gap g (wavelengths)", f"relative difference of sigma_CRB(g)\nn = {conv['grids'][0]} against n = {conv['grids'][1]}")
        gap_axis(ax)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "crb_grid_convergence.png")


def main(argv=None):
    args = parse_args(argv)
    if not args.aggregate_only:
        run_records(args)
    if args.born_power:
        born_power_records(args)
    return aggregate(args)


if __name__ == "__main__":
    main()
