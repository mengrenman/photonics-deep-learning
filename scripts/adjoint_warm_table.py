"""Pool the JSON files of ``measure_adjoint_warm.py`` into one markdown table (cold versus warm adjoint).

    python scripts/adjoint_warm_table.py results/adjoint_warm/bunker_cpuR20_64_1.5_s*.json [more groups ...]

Files are grouped by (tag, grid) from the file contents. For each group: number of files (independent jobs/seeds), number of
gradients, mean and median seconds per gradient cold and warm, the ratio of means, the paired difference warm - cold with
its standard error over gradients (the pairs share their point and are timed back to back in alternating order), the
fraction of gradients where warm was faster, the range of the per-file ratios, the largest relative gradient difference, and
the mean adjoint operator applications per transmitter (cold, warm) where they were counted.
"""

import json
import sys
from collections import defaultdict

import numpy as np


def main(paths):
    groups = defaultdict(list)
    for p in paths:
        r = json.load(open(p))
        groups[(r["tag"], r["grid"], r.get("restart", 100), r["device"])].append(r)
    print("| tag | grid | restart | device | files | gradients | cold mean (s) | warm mean (s) | warm/cold | paired diff (s) +- se | warm faster | per-file ratio range | max grad rel diff | adjoint applications cold / warm |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (tag, grid, restart, dev), rs in sorted(groups.items()):
        c = np.concatenate([r["t_cold"] for r in rs])
        w = np.concatenate([r["t_warm"] for r in rs])
        d = w - c
        ratios = [np.mean(r["t_warm"]) / np.mean(r["t_cold"]) for r in rs]
        cc = [x for r in rs if r.get("counts") for x in r["counts"]["cold"]]
        ww = [x for r in rs if r.get("counts") for x in r["counts"]["warm"]]
        cnt = f"{np.mean(cc):.2f} / {np.mean(ww):.2f} ({len(cc)} solves)" if cc else ""
        print(f"| {tag} | {grid} | {restart} | {dev} | {len(rs)} | {len(c)} | {c.mean():.4f} | {w.mean():.4f} | {w.mean() / c.mean():.4f} | "
              f"{d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f} | {int((d < 0).sum())}/{len(d)} | {min(ratios):.3f} to {max(ratios):.3f} | "
              f"{max(max(r['grad_rel_diff']) for r in rs):.1e} | {cnt} |")


if __name__ == "__main__":
    main(sys.argv[1:])
