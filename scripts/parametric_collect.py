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

Section 6 (``profile``) lists the profile marginals of the gap (<out-dir>/<label>_<model>/profile.json, ``scripts/parametric_profile.py``): the gap posterior where HMC is degenerate (T4,
F2: the MAP lies on the g -> 0 boundary) and, for validation, interior cases next to their chains.

Review follow-up sections (DESIGN 14.7): 7 importance-sampling log evidence against the Laplace values (``parametric_evidence_is.py``, <case>/is_evidence.json) with the F2 against F2u
Occam factors by both; 8 the offline thermodynamic integration against Laplace and importance sampling (<case>/offline_ti.json; the trapezoid lower bound, whether the
schedule was fine enough); 9 the orientation of the Fe fits (psi of the MAPs and chains: the limitation of the coordinate does not bite); 10 cases with chains but no
summary (the convergence report of the segments that exist, <case>/chains_partial.json). The SNR experiment, the ellipticity ladder and the discretization analysis have their own
tables (``snr/snr_summary.md``, ``ellipticity/*.md``, ``discretization/discretization.md``).

Section 5 of tables.md (``laplace`` in tables.json) needs only the MAP stage (<out-dir>/<label>_<model>/map.json): the MAP, the Laplace (Gaussian) interval of the gap
propagated through g = 0.4 sigmoid(u_g), and the Laplace log evidence and Bayes factors of EVERY case that has a MAP, including those without chains. It is labeled Laplace
throughout and is not a sampled posterior.

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


def load_maps(out_dir):
    """``{truth label: {model: map.json dict}}`` from <out-dir>/<label>_<model>/map.json (models F2, F2u, Fe)."""
    maps = {}
    for p in sorted(Path(out_dir).glob("*/map.json")):
        label, _, model = p.parent.name.rpartition("_")
        if label and model in MODELS:
            maps.setdefault(label, {})[model] = json.loads(p.read_text())
    return dict(sorted(maps.items(), key=lambda kv: truth_sort_key(kv[0])))


def true_gap_of_label(label):
    """The true edge-to-edge gap named by a truth label (T1g0.05 -> 0.05; T4 -> 0; T5 -> 0.1)."""
    if label == "T4":
        return 0.0
    if label == "T5":
        return 0.1
    return float(label.split("g", 1)[1])


G_MAX = 0.4  # g = G_MAX sigmoid(u_g) (shapemodels.G_MAX)


def laplace_rows(maps, g_det=None):
    """One row per (truth, model) with a MAP: the misfit, the Laplace interval of the gap (models F2 and F2u; the first coordinate is u_g, g = G_MAX sigmoid(u_g), so
    sd(g) = sd(u_g) g (1 - g / G_MAX) to first order), whether it covers the true gap, P(g < g_det) under the Gaussian, and the Laplace log evidence."""
    from math import erf, sqrt

    rows = []
    for label, models in maps.items():
        g_true = true_gap_of_label(label)
        for model in MODELS:
            rec = models.get(model)
            if rec is None:
                continue
            row = {"truth": label, "model": model, "g_true": g_true, "phi_map": rec["phi_map"], "phi_over_N": rec["phi_over_N"], "phi_truth": rec["phi_truth"],
                   "map_converged": rec["map_converged"], "laplace_gn": rec["evidence"]["laplace_gn"], "laplace_fd": rec["evidence"].get("laplace_fd"), "g_hat": None,
                   "g_sd": None, "g_ci": None, "covers_true": None, "z_true": None, "p_below_g_det": None}
            if model in ("F2", "F2u"):
                g_hat = float(rec["p_hat"][0])
                sd = float(rec["laplace"]["sd_u"][0]) * g_hat * (1.0 - g_hat / G_MAX)
                lo, hi = max(g_hat - 1.645 * sd, 0.0), g_hat + 1.645 * sd
                row.update({"g_hat": g_hat, "g_sd": sd, "g_ci": [lo, hi], "covers_true": bool(lo <= g_true <= hi), "z_true": (g_true - g_hat) / sd if sd > 0 else None})
                if g_det is not None and sd > 0:
                    row["p_below_g_det"] = 0.5 * (1.0 + erf((g_det - g_hat) / (sd * sqrt(2.0))))
            rows.append(row)
    return rows


def laplace_bayes_factors(maps):
    """Per truth with F2 and another model: the Laplace log Bayes factor of F2 against F2u and against Fe (Gauss-Newton and finite-difference Hessian)."""
    out = []
    for label, models in maps.items():
        for other in ("F2u", "Fe"):
            if "F2" in models and other in models:
                a, b = models["F2"]["evidence"], models[other]["evidence"]
                out.append({"truth": label, "other": other, "laplace_gn": a["laplace_gn"] - b["laplace_gn"],
                            "laplace_fd": None if a.get("laplace_fd") is None or b.get("laplace_fd") is None else a["laplace_fd"] - b["laplace_fd"]})
    return out


def load_profiles(out_dir):
    """``{truth label: {model: profile.json dict}}`` from <out-dir>/<label>_<model>/profile.json."""
    out = {}
    for p in sorted(Path(out_dir).glob("*/profile.json")):
        label, _, model = p.parent.name.rpartition("_")
        if label and model in MODELS:
            out.setdefault(label, {})[model] = json.loads(p.read_text())
    return dict(sorted(out.items(), key=lambda kv: truth_sort_key(kv[0])))


def profile_rows(profiles, cases):
    """One row per profile: the gap marginal (mean, sd, 90% interval, P(g < g_det), coverage), the log evidence from the profile integral (``log_norm``, the integral of the
    marginal over u_g, which is the evidence of the unconstrained density), the profile minimum of the misfit over N, the tail masses of the grid, and, if the case also has
    chains, the chains' mean and sd of the gap beside it (the validation)."""
    rows = []
    for label, models in profiles.items():
        for model, p in models.items():
            g = p["gap"]
            chain = cases.get(label, {}).get(model)
            cg = None if chain is None else chain["posterior"]["gap"]
            rows.append({"truth": label, "model": model, "g_true": p["g_true"], "mean": g["mean"], "sd": g["sd"], "ci90": g["ci90"], "covers_true": g.get("covers_true"),
                         "p_below_g_det": g.get("p_below_g_det"), "log_norm": g["log_norm"], "phi_profile_min": p["phi_profile_min"], "tail_low": g["tail_mass_low"],
                         "tail_high": g["tail_mass_high"], "all_converged": p["all_converged"], "chain_mean": None if cg is None else cg["mean"], "chain_sd": None if cg is None else cg["sd"]})
    return rows


def _load_case_files(out_dir, name):
    """``{truth label: {model: dict}}`` from <out-dir>/<label>_<model>/<name> (models F2, F2u, Fe)."""
    out = {}
    for p in sorted(Path(out_dir).glob(f"*/{name}")):
        label, _, model = p.parent.name.rpartition("_")
        if label and model in MODELS:
            out.setdefault(label, {})[model] = json.loads(p.read_text())
    return dict(sorted(out.items(), key=lambda kv: truth_sort_key(kv[0])))


def is_rows(is_evidence, maps):
    """One row per case with an importance-sampling evidence (``is_evidence.json``): the IS log Z with its error and weight ESS, the Laplace values (Gauss-Newton and finite-difference
    Hessian) and the differences; and per truth with both F2 and F2u the Occam factor log BF (F2 against F2u) by IS and by Laplace."""
    rows, occam = [], []
    for label, models in is_evidence.items():
        for model, r in models.items():
            rows.append({"truth": label, "model": model, "is": r["log_z"], "is_se": r["se"], "weight_ess": r["weight_ess"], "n_draws": r["n_draws"], "laplace_gn": r["laplace_gn"],
                         "laplace_fd": r.get("laplace_fd"), "is_minus_gn": r["is_minus_laplace_gn"], "is_minus_fd": r.get("is_minus_laplace_fd"),
                         "map_converged": r.get("map_converged")})
        if "F2" in models and "F2u" in models:
            a, b = models["F2"], models["F2u"]
            occam.append({"truth": label, "is": a["log_z"] - b["log_z"], "is_se": float(np.hypot(a["se"], b["se"])), "laplace_gn": a["laplace_gn"] - b["laplace_gn"]})
    return rows, occam


def ti_rows(offline):
    """One row per case with an offline TI (``offline_ti.json``): the TI value, the trapezoid (a lower bound where the integrand is concave), the Laplace and IS values, the number of
    intervals whose end correction was skipped, and ``schedule_ok``."""
    rows = []
    for label, models in offline.items():
        for model, r in models.items():
            rows.append({"truth": label, "model": model, "n_runs": r["n_runs"], "ti": r["log_z"], "ti_mc_se": r["mc_se"], "between_run_sd": r.get("between_run_sd"), "trapezoid": r["log_z_trapezoid"],
                         "richardson": r["richardson_error_estimate"], "laplace_gn": r["laplace_gn"], "importance_sampling": r.get("importance_sampling_log_z"),
                         "ti_minus_laplace": r["ti_minus_laplace_gn"], "ti_minus_is": r.get("ti_minus_importance_sampling"), "schedule_ok": r.get("schedule_ok"),
                         "skipped": r.get("skipped_intervals_per_run"), "converged": r["converged"], "worst_rhat": r["worst_rhat"], "min_ess": r["min_ess"]})
    return rows


def fe_rows(maps, cases):
    """The orientation psi of every Fe fit: the MAP's psi and, where chains exist, the posterior mean, sd and 90 % interval; ``near_wall`` if |psi| is within 0.1 of pi / 2 (where the
    coordinate of ``shapemodels.Fe`` splits the posterior across the two ends of its box)."""
    from math import pi

    rows = []
    for label, models in maps.items():
        rec = models.get("Fe")
        if rec is None:
            continue
        psi = float(rec["p_hat"][rec["param_names"].index("psi")])
        row = {"truth": label, "psi_map": psi, "a": float(rec["p_hat"][rec["param_names"].index("a")]), "b": float(rec["p_hat"][rec["param_names"].index("b")]),
               "psi_chain": None, "near_wall": bool(abs(psi) > pi / 2 - 0.1)}
        chain = cases.get(label, {}).get("Fe")
        ps = None if chain is None else chain["posterior"].get("params", {}).get("psi")
        if ps is not None:
            row["psi_chain"] = {"mean": ps["mean"], "sd": ps["sd"], "ci90": ps["ci90"]}
            row["near_wall"] = bool(row["near_wall"] or max(abs(ps["ci90"][0]), abs(ps["ci90"][1])) > pi / 2 - 0.1)
        rows.append(row)
    return rows


def partial_chain_rows(out_dir):
    """For every case with chains but no ``summary.json`` and at least 2 segments in every chain: the convergence report of the segments that exist in all chains, the gap mean and
    sd from them (models with a gap), and the numbers of chains, segments and draws. Written to <case>/chains_partial.json. These are not production results (the summary needs all
    segments); they replace figures that were quoted without a file."""
    import jax
    import jax.numpy as jnp

    from bayes_scatter import shapemodels as sm
    from bayes_scatter import shapepost as sp

    rows = []
    for d in sorted(Path(out_dir).glob("*/chains")):
        case = d.parent
        if (case / "summary.json").exists() or case.name.endswith("_pixel"):
            continue
        label, _, model = case.name.rpartition("_")
        if model not in MODELS:
            continue
        per = {}
        for f in sorted(d.glob("c*_s*.npz")):
            c, j = f.stem[1:].split("_s")
            per.setdefault(int(c), {})[int(j)] = f
        n_seg = min(len(v) for v in per.values()) if per else 0
        if n_seg < 2 or any(sorted(v)[:n_seg] != list(range(n_seg)) for v in per.values()):
            continue
        chains = [[np.load(per[c][j]) for j in range(n_seg)] for c in sorted(per)]
        u = np.stack([np.concatenate([seg["u"] for seg in segs]) for segs in chains])
        phi = np.stack([np.concatenate([seg["phi"] for seg in segs]) for segs in chains])
        m = sm.make_model(model)
        p = np.asarray(jax.vmap(jax.vmap(m.p_from_u))(jnp.asarray(u)))
        conv = sp.convergence_report(phi, u, p, names=m.param_names)
        row = {"truth": label, "model": model, "n_chains": int(u.shape[0]), "n_segments": int(n_seg), "n_draws_per_chain": int(u.shape[1]), "converged": conv["converged"],
               "converged_strict": conv["converged_strict"], "rhat_phi": conv["rhat_phi"], "rhat_phi_rank": conv["rhat_phi_rank"], "rhat_phi_expected": conv["rhat_phi_expected"],
               "max_rhat": conv["max_rhat"], "max_rhat_rank": conv["max_rhat_rank"], "worst_ess_bulk": conv["worst_ess_bulk"], "gap_mean": None, "gap_sd": None}
        if "g" in m.param_names:
            g = p[..., m.param_names.index("g")]
            row["gap_mean"], row["gap_sd"] = float(g.mean()), float(g.std(ddof=1))
        (case / "chains_partial.json").write_text(json.dumps({**row, "convergence": conv}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        rows.append(row)
    return rows


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
    row = {"truth": label, "model": model, "g_true": s["g_true"], "g_det": s["g_det"], "converged": conv["converged"], "converged_strict": conv.get("converged_strict", conv["converged"]),
           "max_rhat": conv["max_rhat"], "max_rhat_rank": conv.get("max_rhat_rank"), "rhat_phi": conv.get("rhat_phi"), "rhat_phi_rank": conv.get("rhat_phi_rank"),
           "rhat_phi_expected": conv.get("rhat_phi_expected"), "worst_ess_bulk": conv["worst_ess_bulk"], "multimodal": multi["multimodal"], "multimodal_reasons": multi["reasons"]}
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


def build_tables(cases, maps=None, g_det=None, profiles=None, is_evidence=None, offline=None, partial=None):
    gaps, fits, evs = [], [], []
    for label, models in cases.items():
        for model in MODELS:
            if model in models:
                gaps.append(gap_row(label, model, models[model]))
                fits.append(fit_row(label, model, models[model]))
                evs.append(evidence_row(label, model, models[model]))
    out = {"gap": gaps, "fit": fits, "evidence": evs, "bayes_factors": bayes_factors(cases), "pixel": pixel_rows(cases)}
    if profiles:
        out["profile"] = profile_rows(profiles, cases)
    if maps:
        out["laplace"] = laplace_rows(maps, g_det)
        out["laplace_bayes_factors"] = laplace_bayes_factors(maps)
        out["fe_orientation"] = fe_rows(maps, cases)
    if is_evidence:
        out["is_evidence"], out["is_occam"] = is_rows(is_evidence, maps)
    if offline:
        out["ti_offline"] = ti_rows(offline)
    if partial:
        out["partial_chains"] = partial
    return out


def render_markdown(t):
    parts = ["# Milestone 7: parametric priors, right and wrong (DESIGN 14)\n"]
    parts.append("## 1. Posterior of the gap\n")
    rows = [[r["truth"], r["model"], fmt(r["g_true"]), fmt(r["g_mean"], ".5g"), fmt(r["g_sd"], ".3g"),
             "-" if r["g_ci"] is None else f"[{r['g_ci'][0]:.5g}, {r['g_ci'][1]:.5g}]", fmt(r["covers_true"]), fmt(r["z_true"], ".1f"), fmt(r["p_below_g_det"], ".3f"),
             fmt(r["converged"]), fmt(r["converged_strict"]), f"{fmt(r['max_rhat'], '.3f')} / {fmt(r['max_rhat_rank'], '.3f')}",
             f"{fmt(r['rhat_phi'], '.4f')} ({fmt(r['rhat_phi_expected'], '.4f')})", fmt(r["worst_ess_bulk"], ".0f"), fmt(r["multimodal"])] for r in t["gap"]]
    parts.append(md_table(["truth", "model", "g true", "g mean", "g sd", "90% interval", "covers", "z of truth", "P(g < g_det)", "converged (ESS-aware R-hat of Phi)", "converged (R-hat of Phi <= 1.01)",
                           "max R-hat classic / rank", "R-hat of Phi (expected under independence)", "worst bulk ESS", "multimodal"], rows))
    parts.append("\nThe R-hat of Phi to expect from chains that sample one distribution is 1 + (K / ESS - 1 / n) / 2 (K = 8 split sequences of n = 250 draws, ESS the bulk ESS of Phi): with an ESS of a "
                 "few hundred it already exceeds 1.01. `converged` allows an excess of 0.01 over it; the strict column is the fixed limit 1.01 (DESIGN 12.2).\n")
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
    if t.get("laplace"):
        parts.append("\n## 5. Laplace fits of every case with a MAP (Gaussian approximation; no chains needed)\n")
        rows = [[r["truth"], r["model"], fmt(r["g_true"]), fmt(r["g_hat"], ".5g"), fmt(r["g_sd"], ".3g"),
                 "-" if r["g_ci"] is None else f"[{r['g_ci'][0]:.5g}, {r['g_ci'][1]:.5g}]", fmt(r["covers_true"]), fmt(r["z_true"], ".1f"), fmt(r["p_below_g_det"], ".3f"),
                 fmt(r["phi_over_N"], ".3f"), fmt(r["phi_map"] - r["phi_truth"], ".1f"), fmt(r["map_converged"]), fmt(r["laplace_gn"], ".2f"), fmt(r["laplace_fd"], ".2f")]
                for r in t["laplace"]]
        parts.append(md_table(["truth", "model", "g true", "g MAP", "g sd (Laplace)", "90% interval (Laplace)", "covers", "z of truth", "P(g < g_det)", "Phi_MAP/N", "Phi_MAP - Phi(truth)",
                               "MAP converged", "log Z (GN)", "log Z (FD)"], rows))
        parts.append("")
        rows = [[r["truth"], "F2 vs " + r["other"], fmt(r["laplace_gn"], ".2f"), fmt(r["laplace_fd"], ".2f")] for r in t.get("laplace_bayes_factors", [])]
        parts.append(md_table(["truth", "comparison", "log BF (Laplace GN)", "log BF (Laplace FD)"], rows))
    if t.get("profile"):
        parts.append("\n## 6. Profile marginal of the gap (the other parameters profiled and Laplace-marginalized; no sampling, no truncation of u_g)\n")
        rows = [[r["truth"], r["model"], fmt(r["g_true"]), fmt(r["mean"], ".5g"), fmt(r["sd"], ".3g"), f"[{r['ci90'][0]:.5g}, {r['ci90'][1]:.5g}]", fmt(r["covers_true"]),
                 fmt(r["p_below_g_det"], ".4f"), fmt(r["log_norm"], ".2f"), fmt(r["phi_profile_min"], ".1f"), f"{r['tail_low']:.1e} / {r['tail_high']:.1e}", fmt(r["all_converged"]),
                 "-" if r["chain_mean"] is None else f"{r['chain_mean']:.5g} +- {r['chain_sd']:.3g}"] for r in t["profile"]]
        parts.append(md_table(["truth", "model", "g true", "g mean", "g sd", "90% interval", "covers", "P(g < g_det)", "log Z (profile)", "min profile Phi", "grid tail mass low / high",
                               "all solves converged", "chains: mean +- sd"], rows))
    if t.get("is_evidence"):
        parts.append("\n## 7. Importance-sampling log evidence against the Laplace values (Student-t proposal with 5 degrees of freedom at the Laplace fit; independent of any schedule)\n")
        rows = [[r["truth"], r["model"], f"{r['is']:.3f}", f"{r['is_se']:.3f}", f"{r['weight_ess']:.0f} of {r['n_draws']}", fmt(r["laplace_gn"], ".3f"), fmt(r["laplace_fd"], ".3f"),
                 fmt(r["is_minus_gn"], "+.3f"), fmt(r["is_minus_fd"], "+.3f")] for r in t["is_evidence"]]
        parts.append(md_table(["truth", "model", "log Z (IS)", "IS se", "weight ESS", "log Z Laplace (GN)", "log Z Laplace (FD)", "IS - Laplace GN", "IS - Laplace FD"], rows))
        if t.get("is_occam"):
            parts.append("")
            rows = [[r["truth"], fmt(r["is"], ".3f"), fmt(r["is_se"], ".3f"), fmt(r["laplace_gn"], ".3f")] for r in t["is_occam"]]
            parts.append(md_table(["truth", "log BF F2 vs F2u (IS)", "se", "log BF F2 vs F2u (Laplace GN)"], rows))
    if t.get("ti_offline"):
        parts.append("\n## 8. Thermodynamic integration (offline: beta = 1 from the production chains) against Laplace and importance sampling\n")
        rows = [[r["truth"], r["model"], str(r["n_runs"]), f"{r['ti']:.2f}", f"{r['ti_mc_se']:.2f}", fmt(r["between_run_sd"], ".2f"), f"{r['trapezoid']:.2f}", f"{r['laplace_gn']:.2f}",
                 fmt(r["importance_sampling"], ".2f"), fmt(r["ti_minus_laplace"], "+.2f"), fmt(r["ti_minus_is"], "+.2f"), fmt(r["schedule_ok"]),
                 "-" if not r["skipped"] else "; ".join(f"{k}: {v}" for k, v in r["skipped"].items()), fmt(r["converged"]), fmt(r["worst_rhat"], ".2f"), fmt(r["min_ess"], ".1f")]
                for r in t["ti_offline"]]
        parts.append(md_table(["truth", "model", "runs", "log Z (TI)", "MC se", "between-run sd", "trapezoid", "Laplace (GN)", "IS", "TI - Laplace", "TI - IS", "schedule fine enough",
                               "intervals without end correction", "chains converged", "worst R-hat", "min ESS"], rows))
    if t.get("fe_orientation"):
        parts.append("\n## 9. Orientation of the Fe fits (psi in [-pi/2, pi/2); the coordinate splits a posterior near +-pi/2 across the two ends of its box)\n")
        rows = [[r["truth"], fmt(r["a"], ".4f"), fmt(r["b"], ".4f"), fmt(r["psi_map"], ".5f"),
                 "-" if r["psi_chain"] is None else f"{r['psi_chain']['mean']:.5f} +- {r['psi_chain']['sd']:.5f} [{r['psi_chain']['ci90'][0]:.5f}, {r['psi_chain']['ci90'][1]:.5f}]",
                 fmt(r["near_wall"])] for r in t["fe_orientation"]]
        parts.append(md_table(["truth", "a (MAP)", "b (MAP)", "psi (MAP)", "psi (chains: mean +- sd, 90% interval)", "within 0.1 of +-pi/2"], rows))
    if t.get("partial_chains"):
        parts.append("\n## 10. Cases with chains but no summary (the segments that exist in every chain)\n")
        rows = [[r["truth"], r["model"], f"{r['n_chains']} x {r['n_draws_per_chain']} ({r['n_segments']} segments)", fmt(r["gap_mean"], ".6g"), fmt(r["gap_sd"], ".3g"), fmt(r["converged"]),
                 fmt(r["converged_strict"]), f"{r['rhat_phi']:.4f} / {r['rhat_phi_rank']:.4f} ({r['rhat_phi_expected']:.4f})", f"{r['max_rhat']:.3f} / {r['max_rhat_rank']:.3f}",
                 fmt(r["worst_ess_bulk"], ".0f")] for r in t["partial_chains"]]
        parts.append(md_table(["truth", "model", "chains x draws", "g mean", "g sd", "converged (ESS-aware)", "converged (strict)", "R-hat of Phi classic / rank (expected)",
                               "max R-hat classic / rank", "worst bulk ESS"], rows))
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
    maps = load_maps(out_dir)
    profiles = load_profiles(out_dir)
    if not cases and not maps:
        raise SystemExit(f"no summary.json or map.json under {out_dir}")
    g_det = None
    detection = ROOT / "results" / "resolution" / "detection.json"
    if detection.exists():
        rows = json.loads(detection.read_text())["at_zero"]
        g_det = float(min(rows, key=lambda r: abs(r["eps_r"] - 2.0))["detection_limit_g"])
    tables = build_tables(cases, maps, g_det, profiles, _load_case_files(out_dir, "is_evidence.json"), _load_case_files(out_dir, "offline_ti.json"), partial_chain_rows(out_dir))
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
