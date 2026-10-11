"""Print the Markdown tables of the span-test report from results/span_test/span_test.json (no recomputation).

    python scripts/span_test_tables.py [--dir results/span_test]

Every number printed is read from <dir>/span_test.json (written by scripts/span_test.py), so the report can cite that file.
Errors are percent of ||m|| (eq. 15.21 on eps_r), the best-in-span bound ||Pi_T m - m|| / ||m||.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def f(x, nd=2):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(ROOT / "results" / "span_test"))
    args = ap.parse_args(argv)
    d = json.loads((Path(args.dir) / "span_test.json").read_text())
    P0 = str(d["meta"]["args"]["p_train"])
    book = d["book"]
    print(f"Table 15.3 (book), percent: NMM {book['table_15_3_nmm_pct']}, IEM {book['table_15_3_iem_pct']}, BA {book['table_15_3_ba_pct']}, "
          f"VBIM {book['table_15_3_vbim_pct']}\n")

    print(f"### (a) held-out Table 15.2 draws, P = {P0} (span_test.json rules.<rule>.sweep[{P0}].test.pooled, heldout_min_diagnosis, tail_scan)\n")
    rows = []
    for rule in ("painter", "max"):
        e = d["rules"][rule]["sweep"][P0]["test"]["pooled"]
        diag = d["heldout_min_diagnosis"][rule]
        t = d["tail_scan"][rule]
        rows.append([f"held-out, {rule}", e["n"], f(e["median"]), f(e["p05"]), f(diag["p00_1"]), f(e["min"]), f(diag["min_without_easiest_draw"]),
                     f(e["frac_le_3p3"], 4)])
        rows.append([f"tail scan, {rule}", t["n"], f(t["median"]), f(t["p05"]), f(t["p00_1"]), f(t["min"]), "-", f(t["frac_le_3p3"], 5)])
    print(table(["case", "n", "median", "5th pct", "0.1th pct", "min", "min without the easiest draw", "fraction <= 3.3 %"], rows))
    print("\nThe minimum is the easiest value of the single easiest held-out draw (heldout_min_diagnosis):\n")
    rows = []
    for name, dg in d["heldout_min_diagnosis"].items():
        rows.append([name, dg["n_sets"], dg["n_draws"], f(dg["pooled_min"]), dg["easiest_draw_index"], f(dg["easiest_draw_mean_over_sets"]),
                     ", ".join(f(v) for v in dg["next_easiest_draws_mean_over_sets"][:3]), f(dg["min_without_easiest_draw"]), f(dg["median"]),
                     f(dg["p05"]), f(dg["p00_1"])])
    print(table(["implementation / rule", "training sets", "draws", "pooled min", "easiest draw", "its mean over sets", "next three (mean over sets)",
                 "min without it", "median", "5th pct", "0.1th pct"], rows))

    print(f"\n### bound versus P (painter's rule; span_test.json rules.painter.sweep)\n")
    rows = []
    for P in d["sweep"]:
        e = d["rules"]["painter"]["sweep"][str(P)]
        s = e["test"]["pooled"]
        rows.append([P, e["n_train_sets"], f(s["median"]), f(s["p05"]), f(s["min"]), f(s["frac_le_3p3"], 4),
                     f(e["test_nearest"]["pooled"]["median"]), f(e["empty_scene_pct"]["mean"])])
    print(table(["P", "training sets", "held-out median", "5th pct", "min", "fraction <= 3.3 %", "nearest image median", "empty scene m = 1 (mean)"], rows))

    print(f"\n### (b) Fig. 15.9-like scenes, P = {P0} (rules.<rule>.sweep[{P0}].scenes)\n")
    rows = []
    for key, label in (("test1", "Test #1-like (8 named x 40 sets)"), ("test4", "Test #4-like (8 named x 40 sets)"),
                       ("test1_jitter", "Test #1-like jittered (200 x 40)"), ("test4_jitter", "Test #4-like jittered (200 x 40)"),
                       ("test4_high", "Test #4 eps_r 2, 4, 6 (outside the colorbar)")):
        for rule in ("painter", "max"):
            s = d["rules"][rule]["sweep"][P0]["scenes"][f"{key}@{rule}"]["scene"]["pooled"]
            rows.append([label, rule, s["n"], f(s["median"]), f(s["p05"]), f(s["min"]), f(s["frac_le_3p3"], 4)])
    print(table(["case", "rule", "n", "median", "5th pct", "min", "fraction <= 3.3 %"], rows))
    print("\nNamed scenes (painter): median [min, max] over the training sets; span + constant median; nearest image median; vacuum guess\n")
    rows = []
    sc = d["rules"]["painter"]["sweep"][P0]["scenes"]
    for key in ("test1", "test4", "test4_high"):
        s = sc[f"{key}@painter"]
        for i, name in enumerate(s["names"]):
            rows.append([name, f"{f(s['scene']['per_scene_median'][i])} [{f(s['scene']['per_scene_min'][i])}, {f(s['scene']['per_scene_max'][i])}]",
                         f(s["scene_const"]["per_scene_median"][i]), f(s["scene_nearest"]["per_scene_median"][i]), f(s["vacuum_pct"][i])])
    print(table(["scene", "bound median [min, max]", "span + constant", "nearest image", "vacuum guess"], rows))

    print("\n### named scenes versus P: median bound over the 8 scenes x training sets (painter; rules.painter.sweep[P].scenes)\n")
    rows = []
    for P in d["sweep"]:
        sc_p = d["rules"]["painter"]["sweep"][str(P)]["scenes"]
        rows.append([P, f(sc_p["test1@painter"]["scene"]["pooled"]["median"]), f(sc_p["test4@painter"]["scene"]["pooled"]["median"]),
                     f(sc_p["test1@painter"]["scene_const"]["pooled"]["median"]), f(sc_p["test4@painter"]["scene_const"]["pooled"]["median"])])
    print(table(["P", "Test #1-like span", "Test #4-like span", "Test #1-like span + constant", "Test #4-like span + constant"], rows))

    v = d["variants"]
    print(f"\n### other readings of the NMM at P = {v['p_train']} ({v['n_sets']} training sets, painter's rule; span_test.json variants)\n")
    rows = []
    for tag, e in v["eps_ranges"].items():
        for key in ("test1", "test4"):
            for name in ("span", "span_const", "shift_clip", "const_clip"):
                s = e["scenes"][key][name]
                rows.append([f"U[{tag.replace('-', ', ')}]", key, name, f"{f(s['digitized']['median'])} [{f(s['digitized']['min'])}]",
                             f"{f(s['pooled_over_8_scenes']['median'])} [{f(s['pooled_over_8_scenes']['min'])}]"])
    print(table(["training eps_r", "test", "reading", "digitized scene: median [min]", "8 named scenes: median [min]"], rows))
    print("\nReadings: " + "; ".join(f"{k}: {t}" for k, t in v["readings"].items()))
    rows = [[tag, f(e["empty_scene_pct"]["mean"]), f"{f(e['empty_scene_pct']['min'])} to {f(e['empty_scene_pct']['max'])}", f(e["heldout_span"]["median"]),
             f(e["heldout_span"]["p05"]), f(e["heldout_min_diagnosis"]["p00_1"])] for tag, e in v["eps_ranges"].items()]
    print("\n" + table(["training eps_r", "empty scene (mean)", "range over sets", "held-out median", "5th pct", "0.1th pct"], rows))
    print("\nChi-normalized span error (relative to ||m - 1||), digitized scene median: " + "; ".join(
        f"U[{tag.replace('-', ', ')}] {key} {f(e['scenes'][key]['chi_norm']['digitized']['median'])}" for tag, e in v["eps_ranges"].items() for key in ("test1", "test4")))
    for key, b in v["best_reading"].items():
        s = b["smallest_value_over_all_scenes_and_sets"]
        print(f"\nBest reading, {key}: digitized median {f(b['digitized_median']['median'])} ({b['digitized_median']['reading']}, training eps_r "
              f"{b['digitized_median']['eps_range']}); smallest value over all named scenes and sets {f(s['min'])} ({s['scene']}, {s['reading']}, {s['eps_range']})")
    fl = v["flatness"]
    gaps = [g for dd in fl["digitized_median_test4_minus_test1_pp"].values() for g in dd.values()]
    print(f"\nTest #4 minus Test #1 (digitized medians) over all readings: {f(min(gaps))} to {f(max(gaps))} pp; the book's NMM column: "
          f"{f(fl['book_nmm_test4_minus_test1_pp'], 3)} pp (spread over the four tests {f(fl['book_nmm_max_minus_min_pp'], 3)} pp)")
    mr = v["metric_reading"]
    print("\nSquared-misfit reading (book BA / VBIM, percent as printed; relative error if squared; vacuum guess of the digitized scene):\n")
    rows = [[k, f(x["book_pct"], 4), f(x["if_squared_relative_error_pct"]), f(mr["vacuum_guess_pct"][k.split("_")[0]]), f(x["as_printed_vs_vacuum"]),
             f(x["squared_vs_vacuum"])] for k, x in mr["tests"].items()]
    print(table(["test and method", "printed (%)", "if squared (%)", "vacuum guess (%)", "printed / vacuum", "squared / vacuum"], rows))

    cc = d["cross_check"]
    print("\n### independent implementation B (span_test.json cross_check)\n")
    rows = []
    for rule in ("painter", "max"):
        h = cc["heldout_bound"][rule]
        rows.append([rule, f(h["A"]["median"]), f(h["B"]["median"]), f(h["A"]["mean"]), f(h["B"]["mean"]), f(h["z_mean"]), f(h["z_median"])])
    print(table(["rule", "A median", "B median", "A mean", "B mean", "Welch z of the mean", "bootstrap z of the median"], rows))
    r4 = cc["bridge4_rasterizers"]
    print(f"\nmax |z| over {len(cc['z_all'])} comparisons {f(cc['z_max_abs'])}; agree = {cc['agree']}; QR vs gelsd {cc['bridge1_projection']['max_abs_diff_qr_vs_gelsd']:.1e}, "
          f"QR vs SVD {cc['bridge1_projection']['max_abs_diff_qr_vs_svd']:.1e}; scene bound difference <= {f(cc['bridge2_scenes']['error_diff_pp_max'], 3)} pp; "
          f"same shapes through both rasterizers (A - B mean, held-out / scenes): painter {f(r4['painter']['images']['mean_diff_pp'], 3)} / "
          f"{f(r4['painter']['scenes']['mean_diff_pp'], 3)} pp, max {f(r4['max']['images']['mean_diff_pp'], 3)} / {f(r4['max']['scenes']['mean_diff_pp'], 3)} pp")
    b = d["binary_masks"]
    print("\nBinary-mask rasterization (implementation B, painter): " + "; ".join(
        f"{k} median {f(b['painter'][k]['median'])}" for k in ("test", "test1", "test4")))


if __name__ == "__main__":
    main()
