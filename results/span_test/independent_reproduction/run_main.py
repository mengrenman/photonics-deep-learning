"""Span-projection experiments: held-out in-family draws, P sweep, generator variants, Test #1/#4-like scenes."""
import json
import sys
import time

import gen  # sets the thread-count environment variables; must precede the numpy import
import numpy as np
import scenes

THRESH = 0.0331  # top of the reported NMM range, 3.26-3.31 %
BASE = dict(ring_rule="resample", containment="clip", rotate=False, overlap="painter", ss=4)
HELD_SEED = 777
out = {}


def basis(T):
    """Orthonormal basis of col span(T); SVD for small P, QR (with a rank check) for large P."""
    if T.shape[1] <= 600:
        Q, s = gen.orth_basis(T)
        assert Q.shape[1] == T.shape[1], "rank deficient training set"
        return Q
    Q, R = np.linalg.qr(T)
    d = np.abs(np.diag(R))
    assert d.min() > 1e-8 * d.max(), "rank deficient (QR)"
    return Q


def evaluate(T, M, want_extra=True):
    """Errors of the columns of M against the training set T (columns are eps_r images).

    err      : ||M - Pi_T M|| / ||M||                                  (the bound of DESIGN 11.4)
    err_aug  : same with the constant image 1 appended to the basis    (what a bias term could add)
    err_chi  : training targets and outputs are contrasts chi = eps - 1, output eps_hat = 1 + Pi_chi chi,
               error ||eps_hat - eps|| / ||eps||
    near     : distance to the nearest training image / ||M||
    """
    Q = basis(T)
    r = dict(err=gen.proj_err(Q, M))
    if want_extra:
        one = np.ones((gen.N, 1))
        Q1 = basis(np.hstack([T, one]))
        r["err_aug"] = gen.proj_err(Q1, M)
        Qc = basis(T - 1.0)
        chi = M - 1.0
        r["err_chi"] = np.linalg.norm(chi - Qc @ (Qc.T @ chi), axis=0) / np.linalg.norm(M, axis=0)
        r["near"] = gen.nearest_err(T, M)
    return r


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)) if x.size > 1 else 0.0, min=float(x.min()),
                q05=float(np.quantile(x, .05)), med=float(np.median(x)), q95=float(np.quantile(x, .95)),
                max=float(x.max()), frac_below=float((x < THRESH).mean()), n=int(x.size))


def fmt(d, scale=100):
    return (f"mean {scale * d['mean']:6.2f}  [q05 {scale * d['q05']:6.2f} med {scale * d['med']:6.2f} "
            f"q95 {scale * d['q95']:6.2f}]  min {scale * d['min']:6.2f}  max {scale * d['max']:6.2f}  "
            f"frac<3.31%: {d['frac_below']:.3f}  n={d['n']}")


# ---------------------------------------------------------------------------------------------
def part_A():
    """Baseline generator, 1000 held-out draws, P sweep."""
    print("\n=== A. baseline generator, 1000 held-out in-family draws ===", flush=True)
    M = gen.make_set(1000, HELD_SEED, **BASE)
    chi = M - 1
    vac = np.linalg.norm(chi, axis=0) / np.linalg.norm(M, axis=0)
    out["A_heldout_vacuum"] = stats(vac)
    print("held-out: mean ||eps||", np.linalg.norm(M, axis=0).mean(), " vacuum-guess error ||chi||/||eps||:",
          fmt(out["A_heldout_vacuum"]))
    plan = {100: 20, 200: 10, 500: 5, 1000: 2, 2000: 1, 4000: 1}
    out["A"] = {}
    for P, R in plan.items():
        t0 = time.time()
        res = {k: [] for k in ("err", "err_aug", "err_chi", "near")}
        for r in range(R):
            T = gen.make_set(P, 1000 * P + r, **BASE)
            ev = evaluate(T, M, want_extra=(P <= 2000))
            for k, v in ev.items():
                res[k].append(v)
        row = {}
        for k, v in res.items():
            if not v:
                continue
            arr = np.array(v)  # (R, 1000)
            row[k] = stats(arr.ravel())
            row[k]["rep_means"] = [float(a) for a in arr.mean(axis=1)]
            row[k]["sd_of_rep_means"] = float(arr.mean(axis=1).std(ddof=1)) if R > 1 else None
        out["A"][P] = row
        print(f"P={P:5d} R={R:2d}  span(T)      {fmt(row['err'])}   sd of replicate means "
              f"{row['err']['sd_of_rep_means']}   [{time.time() - t0:.0f}s]", flush=True)
        if "err_aug" in row:
            print(f"                 span(T,1)    {fmt(row['err_aug'])}")
            print(f"                 contrast bas {fmt(row['err_chi'])}")
            print(f"                 nearest img  {fmt(row['near'])}", flush=True)
    np.save("heldout_baseline.npy", M[:, :50].astype(np.float32))


def part_B():
    """Variants of the generator at P = 100 (and P = 200, 500 for overlap rules)."""
    print("\n=== B. generator variants (training and held-out from the same variant) ===", flush=True)
    variants = [("baseline", {})]
    variants += [(f"overlap={o}", dict(overlap=o)) for o in ("painter_rev", "painter_rand", "max", "add")]
    variants += [("ss=1 (binary)", dict(ss=1)), ("ss=8", dict(ss=8))]
    variants += [(f"ring_rule={q}", dict(ring_rule=q)) for q in ("swap", "resample_r4", "void")]
    variants += [("rotate", dict(rotate=True)), ("containment=inside", dict(containment="inside"))]
    variants += [("max + binary", dict(overlap="max", ss=1)),
                 ("inside + rotate + max", dict(containment="inside", rotate=True, overlap="max"))]
    out["B"] = {}
    for name, kw in variants:
        cfg = dict(BASE, **kw)
        M = gen.make_set(1000, HELD_SEED + 1, **cfg)
        out["B"][name] = {}
        for P, R in ((100, 5), (200, 5), (500, 3)):
            errs = []
            for r in range(R):
                T = gen.make_set(P, 5000 + 100 * P + r, **cfg)
                errs.append(evaluate(T, M, want_extra=False)["err"])
            s = stats(np.array(errs).ravel())
            s["vac"] = float((np.linalg.norm(M - 1, axis=0) / np.linalg.norm(M, axis=0)).mean())
            out["B"][name][P] = s
        print(f"{name:24s} " + "  |  ".join(
            f"P={P}: mean {100 * out['B'][name][P]['mean']:5.2f} min {100 * out['B'][name][P]['min']:5.2f}"
            for P in (100, 200, 500)) + f"  | vacuum-guess {100 * s['vac']:.1f}", flush=True)


def part_C():
    """Our Test #1 / #4 (and extra #2) like scenes against training sets of several sizes and variants."""
    print("\n=== C. Test-#1-like / #4-like scenes ===", flush=True)
    fams = dict(test1=scenes.test1_like, test4=scenes.test4_like, test2_cross=scenes.test2_like)
    out["C"] = {}
    for fam, f in fams.items():
        rng = np.random.default_rng(31415)
        K = 40 if fam != "test2_cross" else 15
        for truth_cfg_name, tkw in (("painter ss4", dict(overlap="painter", ss=4)),
                                    ("max ss1", dict(overlap="max", ss=1))):
            shapes_list = [f(None)] + [f(rng) for _ in range(K)]
            Mt = np.stack([gen.render(sh, tkw["overlap"], tkw["ss"]).ravel() for sh in shapes_list], axis=1)
            vac = np.linalg.norm(Mt - 1, axis=0) / np.linalg.norm(Mt, axis=0)
            for P, R in ((100, 10), (200, 10), (500, 5)):
                for tr_name, kw in (("baseline", {}), ("max+binary", dict(overlap="max", ss=1)),
                                    ("inside+rotate", dict(containment="inside", rotate=True)),
                                    ("ring swap", dict(ring_rule="swap"))):
                    if tr_name != "baseline" and P != 100:
                        continue
                    cfg = dict(BASE, **kw)
                    res = {k: [] for k in ("err", "err_aug", "err_chi", "near")}
                    for r in range(R):
                        T = gen.make_set(P, 9000 + 10 * P + r, **cfg)
                        ev = evaluate(T, Mt)
                        for k, v in ev.items():
                            res[k].append(v)
                    key = f"{fam} | truth {truth_cfg_name} | P={P} | train {tr_name}"
                    e = np.array(res["err"])  # (R, 1+K)
                    inst_mean = e.mean(axis=0)  # mean over training replicates, per instantiation
                    row = dict(
                        nominal_mean=float(inst_mean[0]), nominal_sd=float(e[:, 0].std(ddof=1)),
                        nominal_min=float(e[:, 0].min()),
                        inst_mean_min=float(inst_mean[1:].min()), inst_mean_med=float(np.median(inst_mean[1:])),
                        inst_mean_max=float(inst_mean[1:].max()), global_min=float(e.min()), n_total=int(e.size),
                        nominal_aug=float(np.array(res["err_aug"])[:, 0].mean()),
                        inst_aug_min=float(np.array(res["err_aug"]).mean(0)[1:].min()),
                        nominal_chi=float(np.array(res["err_chi"])[:, 0].mean()),
                        inst_chi_min=float(np.array(res["err_chi"]).mean(0)[1:].min()),
                        nominal_near=float(np.array(res["near"])[:, 0].mean()),
                        inst_near_min=float(np.array(res["near"]).mean(0)[1:].min()),
                        vac_nominal=float(vac[0]), vac_min=float(vac[1:].min()), vac_med=float(np.median(vac[1:])),
                        vac_max=float(vac[1:].max()),
                        frac_below=float((e < THRESH).mean()))
                    out["C"][key] = row
                    print(f"{key}\n    span(T): nominal {100 * row['nominal_mean']:.2f} (sd {100 * row['nominal_sd']:.2f}, "
                          f"min {100 * row['nominal_min']:.2f}); instantiations (mean over reps) min/med/max "
                          f"{100 * row['inst_mean_min']:.2f}/{100 * row['inst_mean_med']:.2f}/{100 * row['inst_mean_max']:.2f}; "
                          f"global min {100 * row['global_min']:.2f}; frac<3.31%: {row['frac_below']:.3f}\n"
                          f"    span(T,1): nominal {100 * row['nominal_aug']:.2f} inst-min {100 * row['inst_aug_min']:.2f} | "
                          f"contrast basis: nominal {100 * row['nominal_chi']:.2f} inst-min {100 * row['inst_chi_min']:.2f} | "
                          f"nearest image: nominal {100 * row['nominal_near']:.2f} inst-min {100 * row['inst_near_min']:.2f} | "
                          f"vacuum guess nominal {100 * row['vac_nominal']:.2f} (range {100 * row['vac_min']:.2f}-{100 * row['vac_max']:.2f})",
                          flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["A", "B", "C"]
    for w in which:
        {"A": part_A, "B": part_B, "C": part_C}[w]()
        with open(f"results_{w}.json", "w") as fh:
            json.dump({k: v for k, v in out.items() if k.startswith(w)}, fh, indent=1)
