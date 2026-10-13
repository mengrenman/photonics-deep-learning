"""Milestone 7 (DESIGN 14.4): tables and figures from the per-case summaries of ``scripts/parametric_run.py``.

    python scripts/parametric_collect.py [--out-dir results/parametric]

Reads <out-dir>/<truth label>_<model>/summary.json (model F2, F2u, Fe, pixel; a missing case is skipped) and writes, next to them,

  tables.md / tables.json   (1) the posterior of the gap per (truth, model): mean, 90 % interval, whether the interval covers the true gap, P(g < g_det(30 dB)), the
                            convergence flag and the multimodality flag; (2) the fit: Phi_MAP against N and against Phi at the truth, the posterior-predictive p-values
                            (refit and draw); (3) the log evidence (Laplace with the Gauss-Newton and the finite-difference Hessian, and thermodynamic integration
                            where run) and the log Bayes factors F2 against Fe and F2 against F2u; (4) the pixel posteriors (Matern, TV at lambda*): P(gap valley) of
                            the half-height strip, labeled single-point values beside it.
  parametric_gap.png        the 90 % intervals of g per truth and model against the true gap and g_det.
  parametric_fit.png        Phi_MAP / N per truth and model, and the refit predictive p-values.
  parametric_evidence.png   the log Bayes factors of F2 against F2u and Fe (Laplace, and TI where available).

The questions of DESIGN 14.4 (i) to (iii) are answered by reading the first three tables; the script reports numbers and flags and draws no conclusion.
"""

import _threadcap  # noqa: F401  (caps the BLAS threads; must precede numpy)

import argparse
import functools
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "parametric"
print = functools.partial(print, flush=True)

MODELS = ("F2", "F2u", "Fe")
COLORS = {"F2": "#2a78d6", "F2u": "#1baf7a", "Fe": "#eb6834", "pixel": "#7a5bd1"}
INK, MUTED = "#0b0b0b", "#52514e"
TRUTH_ORDER = ("T1g0.05", "T1g0.1", "T2g0.05", "T2g0.1", "T3g0.05", "T3g0.1", "T4", "T5")


def truth_sort_key(label):
    return (TRUTH_ORDER.index(label), "") if label in TRUTH_ORDER else (len(TRUTH_ORDER), label)


def load_cases(out_dir):
    """``{truth label: {model: summary dict}}`` from <out-dir>/<label>_<model>/summary.json."""
    cases = {}
    for p in sorted(Path(out_dir).glob("*/summary.json")):
        label, _, model = p.parent.name.rpartition("_")
        if not label:
            continue
        cases.setdefault(label, {})[model] = json.loads(p.read_text())
    return dict(sorted(cases.items(), key=lambda kv: truth_sort_key(kv[0])))


def fmt(x, spec=".4g", none="-"):
    if x is None:
        return none
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        if not np.isfinite(x):
            return str(x)
    except TypeError:
        return str(x)
    return format(x, spec)


def gap_row(label, model, s):
    gap = s["posterior"]["gap"]
    conv, multi = s["convergence"], s["multimodality"]
    row = {"truth": label, "model": model, "g_true": s["g_true"], "g_det": s["g_det"], "converged": conv["converged"], "max_rhat": conv["max_rhat"],
           "worst_ess_bulk": conv["worst_ess_bulk"], "multimodal": multi["multimodal"], "multimodal_reasons": multi["reasons"]}
    if gap is None:
        row.update({"g_mean": None, "g_sd": None, "g_ci": None, "covers_true": None, "p_below_g_det": None, "z_true": None})
    else:
        row.update({"g_mean": gap["mean"], "g_sd": gap["sd"], "g_ci": gap["ci"], "covers_true": gap.get("covers_true"), "p_below_g_det": gap.get("p_below_g_det"),
                    "z_true": gap.get("z_true")})
    return row


def fit_row(label, model, s):
    m, ppc = s["map"], s["ppc"]
    return {"truth": label, "model": model, "phi_map": m["phi_map"], "n_data": m["n_data"], "phi_over_N": m["phi_over_N"], "phi_truth": m["phi_truth"],
            "phi_map_minus_phi_truth": m["phi_map_minus_phi_truth"], "map_converged": m["converged"],
            "ppc_refit_p": None if ppc is None else ppc["refit"]["p_value"], "ppc_refit_z": None if ppc is None else ppc["refit"]["z"],
            "ppc_draw_p": None if ppc is None else ppc["draw"]["p_value"], "posterior_phi_over_N": s["posterior"]["phi_over_N"]}


def evidence_row(label, model, s):
    ev = s["evidence"]
    ti = ev.get("ti")
    return {"truth": label, "model": model, "laplace_gn": ev.get("laplace_gn"), "laplace_fd": ev.get("laplace_fd"), "ti": None if ti is None else ti["log_z"],
            "ti_se": None if ti is None else ti["mc_se"], "ti_converged": None if ti is None else ti["converged"],
            "ti_discretization": None if ti is None else ti["discretization"]["richardson_error_estimate"]}


def bayes_factors(cases):
    """Per truth: log BF of F2 against F2u and against Fe, by the Laplace evidence (Gauss-Newton and finite-difference Hessian) and by TI where both models have it.
    A log Bayes factor above 0 favors F2. Uses the normalized likelihood of the same data, so the constants cancel; the priors are those of ``shapemodels``."""
    out = []
    for label, models in cases.items():
        if "F2" not in models:
            continue
        for other in ("F2u", "Fe"):
            if other not in models:
                continue
            a, b = models["F2"]["evidence"], models[other]["evidence"]
            row = {"truth": label, "other": other}
            for key in ("laplace_gn", "laplace_fd"):
                row[key] = None if a.get(key) is None or b.get(key) is None else a[key] - b[key]
            ta, tb = a.get("ti"), b.get("ti")
            row["ti"] = None if ta is None or tb is None else ta["log_z"] - tb["log_z"]
            row["ti_se"] = None if ta is None or tb is None else float(np.hypot(ta["mc_se"], tb["mc_se"]))
            out.append(row)
    return out


def pixel_rows(cases):
    out = []
    for label, models in cases.items():
        s = models.get("pixel")
        if s is None:
            continue
        for prior, rec in s["priors"].items():
            f = rec["features"]
            reg = f["region"]
            truth_fails = rec["truth"]["region"]["holds"] is False
            out.append({"truth": label, "prior": prior, "lam": rec["lam"], "phi_over_N": rec["phi_over_N"], "map_ok": rec["map_ok"], "status": f["region_status"],
                        "p_valley_region": None if reg is None or truth_fails else reg["p_valley"], "p_valley_point": f["point"]["p_valley"],
                        "truth_holds_region": rec["truth"]["region"]["holds"], "strip_width_pixels": f["geometry"]["strip_width_pixels"]})
    return out


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_tables(cases):
    gaps, fits, evs = [], [], []
    for label, models in cases.items():
        for model in MODELS:
            if model in models:
                gaps.append(gap_row(label, model, models[model]))
                fits.append(fit_row(label, model, models[model]))
                evs.append(evidence_row(label, model, models[model]))
    return {"gap": gaps, "fit": fits, "evidence": evs, "bayes_factors": bayes_factors(cases), "pixel": pixel_rows(cases)}


def render_markdown(t):
    parts = ["# Milestone 7: parametric priors, right and wrong (DESIGN 14)\n"]
    parts.append("## 1. Posterior of the gap\n")
    rows = [[r["truth"], r["model"], fmt(r["g_true"]), fmt(r["g_mean"], ".5g"), fmt(r["g_sd"], ".3g"),
             "-" if r["g_ci"] is None else f"[{r['g_ci'][0]:.5g}, {r['g_ci'][1]:.5g}]", fmt(r["covers_true"]), fmt(r["z_true"], ".1f"), fmt(r["p_below_g_det"], ".3f"),
             fmt(r["converged"]), fmt(r["max_rhat"], ".3f"), fmt(r["multimodal"])] for r in t["gap"]]
    parts.append(md_table(["truth", "model", "g true", "g mean", "g sd", "90% interval", "covers", "z of truth", "P(g < g_det)", "converged", "max R-hat", "multimodal"], rows))
    parts.append("\n## 2. Fit: misfit and posterior-predictive checks\n")
    rows = [[r["truth"], r["model"], fmt(r["phi_map"], ".1f"), str(r["n_data"]), fmt(r["phi_over_N"], ".3f"), fmt(r["phi_truth"], ".1f"), fmt(r["phi_map_minus_phi_truth"], ".1f"),
             fmt(r["ppc_refit_p"], ".3f"), fmt(r["ppc_refit_z"], ".1f"), fmt(r["ppc_draw_p"], ".3f")] for r in t["fit"]]
    parts.append(md_table(["truth", "model", "Phi_MAP", "N", "Phi_MAP/N", "Phi(truth)", "Phi_MAP - Phi(truth)", "p (refit)", "z (refit)", "p (draw)"], rows))
    parts.append("\n## 3. Log evidence and Bayes factors\n")
    rows = [[r["truth"], r["model"], fmt(r["laplace_gn"], ".2f"), fmt(r["laplace_fd"], ".2f"), fmt(r["ti"], ".2f"), fmt(r["ti_se"], ".2f"), fmt(r["ti_converged"]),
             fmt(r["ti_discretization"], ".2f")] for r in t["evidence"]]
    parts.append(md_table(["truth", "model", "log Z Laplace (GN)", "log Z Laplace (FD)", "log Z TI", "TI MC se", "TI converged", "TI Richardson error"], rows))
    parts.append("")
    rows = [[r["truth"], "F2 vs " + r["other"], fmt(r["laplace_gn"], ".2f"), fmt(r["laplace_fd"], ".2f"), fmt(r["ti"], ".2f"), fmt(r["ti_se"], ".2f")] for r in t["bayes_factors"]]
    parts.append(md_table(["truth", "comparison", "log BF (Laplace GN)", "log BF (Laplace FD)", "log BF (TI)", "TI se"], rows))
    parts.append("\n## 4. Pixel posteriors on the same data (half-height strip)\n")
    rows = [[r["truth"], r["prior"], fmt(r["lam"]), fmt(r["phi_over_N"], ".3f"), fmt(r["map_ok"]), "-" if r["p_valley_region"] is None else format(r["p_valley_region"], ".4f"),
             r["status"] if r["p_valley_region"] is None else "ok", fmt(r["p_valley_point"], ".4f"), fmt(r["strip_width_pixels"], ".2f")] for r in t["pixel"]]
    parts.append(md_table(["truth", "prior", "lambda", "Phi_MAP/N", "MAP ok", "P(valley) region", "status", "P(valley) single point", "gap (pixels)"], rows))
    return "\n".join(parts) + "\n"


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


def plot_gap(cases, path):
    """90 % intervals of g for every truth and model (log axis), the true gap (black tick) and g_det (dashed)."""
    labels = list(cases)
    fig = Figure(figsize=(max(5.0, 1.15 * len(labels) + 1.5), 4.0), dpi=130)
    ax = fig.subplots()
    width = 0.22
    for i, label in enumerate(labels):
        for k, model in enumerate(MODELS):
            s = cases[label].get(model)
            if s is None or s["posterior"]["gap"] is None:
                continue
            gap = s["posterior"]["gap"]
            x = i + (k - 1) * width
            lo, hi = max(gap["ci"][0], 1e-6), max(gap["ci"][1], 1e-6)
            ax.plot([x, x], [lo, hi], color=COLORS[model], lw=2.2, solid_capstyle="butt")
            ax.plot([x], [max(gap["mean"], 1e-6)], marker="o", ms=4, color=COLORS[model])
        gt = next(iter(cases[label].values()))["g_true"]
        if gt is not None:
            ax.plot([i - 0.38, i + 0.38], [max(gt, 1e-6)] * 2, color=INK, lw=1.2)
    gd = next((s["g_det"] for ms in cases.values() for s in ms.values() if s.get("g_det")), None)
    if gd:
        ax.axhline(gd, color=MUTED, ls="--", lw=0.9)
        ax.text(len(labels) - 0.5, gd * 1.25, "g_det (30 dB)", ha="right", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    style(ax, "truth", "edge-to-edge gap g (wavelengths); 90% interval, true value in black")
    for model in MODELS:
        ax.plot([], [], color=COLORS[model], lw=2.2, label=model)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path)


def plot_fit(cases, path):
    labels = list(cases)
    fig = Figure(figsize=(max(6.0, 1.3 * len(labels) + 2.0), 6.0), dpi=130)
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    width = 0.25
    for i, label in enumerate(labels):
        for k, model in enumerate(MODELS):
            s = cases[label].get(model)
            if s is None:
                continue
            x = i + (k - 1) * width
            ax1.bar(x, s["map"]["phi_over_N"], width=width * 0.9, color=COLORS[model])
            ppc = s["ppc"]
            if ppc is not None:
                ax2.plot([x], [ppc["refit"]["p_value"]], marker="o", ms=5, color=COLORS[model])
    ax1.axhline(1.0, color=INK, lw=0.9)
    ax1.set_yscale("log")
    style(ax1, "", "Phi_MAP / N")
    ax2.axhline(0.05, color=MUTED, ls="--", lw=0.9)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8, rotation=30)
    style(ax2, "truth", "predictive p-value (refit)")
    for model in MODELS:
        ax1.bar([], [], color=COLORS[model], label=model)
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def plot_evidence(bf, path):
    fig = Figure(figsize=(max(5.0, 0.9 * len(bf) + 2.0), 3.8), dpi=130)
    ax = fig.subplots()
    labels = [f"{r['truth']}\nvs {r['other']}" for r in bf]
    xs = np.arange(len(bf))
    for off, key, marker, color in ((-0.12, "laplace_gn", "o", "#2a78d6"), (0.0, "laplace_fd", "s", "#1baf7a"), (0.12, "ti", "D", "#eb6834")):
        ys = [r[key] if r[key] is not None else np.nan for r in bf]
        ax.plot(xs + off, ys, ls="none", marker=marker, ms=5, color=color, label={"laplace_gn": "Laplace (GN)", "laplace_fd": "Laplace (FD)", "ti": "TI"}[key])
    ax.axhline(0.0, color=INK, lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    style(ax, "", "log Bayes factor of F2 (above 0 favors F2)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def collect(out_dir):
    out_dir = Path(out_dir)
    cases = load_cases(out_dir)
    if not cases:
        raise SystemExit(f"no summary.json under {out_dir}")
    tables = build_tables(cases)
    (out_dir / "tables.md").write_text(render_markdown(tables))
    (out_dir / "tables.json").write_text(json.dumps(tables, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    param_cases = {k: {m: s for m, s in v.items() if m in MODELS} for k, v in cases.items()}
    param_cases = {k: v for k, v in param_cases.items() if v}
    if param_cases:
        plot_gap(param_cases, out_dir / "parametric_gap.png")
        plot_fit(param_cases, out_dir / "parametric_fit.png")
    if tables["bayes_factors"]:
        plot_evidence(tables["bayes_factors"], out_dir / "parametric_evidence.png")
    print(f"wrote {out_dir / 'tables.md'} ({sum(len(v) for v in cases.values())} cases, {len(tables['bayes_factors'])} Bayes factors)")
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args(argv)
    return collect(args.out_dir)


if __name__ == "__main__":
    main()
