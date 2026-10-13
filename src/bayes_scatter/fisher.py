"""Fisher information and Cramer-Rao bounds on the two-disk object manifold (docs/DESIGN.md 13.2).

Setting. The data are d = F(chi(p)) + e with e ~ CN(0, s^2) i.i.d. (DESIGN 4: real and imaginary parts N(0, s^2 / 2)), F the
scattering forward map (``forward.Forward``) and chi(p) the two-disk contrast of ``shapes.TwoDisks``. The log-likelihood
gradient with respect to p_k is (2 / s^2) Re(sum conj(d_k) e) with d_k = dd/dp_k, so the Fisher matrix is

    F = (2 / s^2) Re(D^H D),   D = [dd/dp_1, ..., dd/dp_K]  of shape (Nt Nr, K),

D the "data Jacobian along the shape parameters". D is evaluated without a dense Jacobian: dd/dp_k = J (dchi/dp_k), a
single ``jax.jvp`` of the forward map along the exact shape derivative dchi/dp_k of ``shapes.shape_derivatives`` (the forward
solve of the primal is shared between the K tangents through ``jax.linearize``). F is real symmetric positive semidefinite.

Cramer-Rao bound. For an unbiased estimator of the gap, var(g_hat) >= (F^-1)_gg with all parameters of the model unknown (the
"nuisance parameters unknown" bound), and >= 1 / F_gg when the others are known (always the smaller). Smith's (2005) statistical
resolution criterion: the gap is resolvable when the standard deviation sigma_CRB(g) <= g.

SNR scaling. F is proportional to 1 / s^2, and the noise level of the design is set by the POWER SNR relative to the data of the
object itself (DESIGN 4): s^2(SNR) = mean|d_clean|^2 10^(-SNR / 10). At a fixed object mean|d_clean|^2 does not change, so

    F(SNR) = F(SNR_ref) 10^((SNR - SNR_ref) / 10),   sigma_CRB(SNR) = sigma_CRB(SNR_ref) 10^(-(SNR - SNR_ref) / 20),

and sigma_CRB(SNR) = g holds at

    SNR_req(g) = SNR_ref + 20 log10(sigma_CRB(SNR_ref) / g)    (decibels; ``required_snr_db``).

Born comparison. With the Born forward d = R (chi u_inc) the same formulas hold with dd/dp_k = R (dchi/dp_k u_inc), linear in the
shape derivative (``born_data_derivatives``). Two normalizations of the noise are reported, and they answer different questions:

* SAME NOISE LEVEL (``info_ratio_*``): one s for both models, that of the FULL model's data (so F_full / F_Born does not depend on s). At a fixed noise level the
  ratio mostly measures how much more power the Born model scatters (its effective contrast): F_full / F_Born tracks P_full / P_Born (``power_ratio_full_over_born``),
  which falls with the contrast because the Born model has no multiple scattering to saturate it.
* MATCHED POWER SNR (``info_ratio_*_matched``): each model at its own power SNR (DESIGN 4: s^2 = mean|d_model|^2 10^(-SNR / 10)), i.e. the same SNR for the full and
  the Born data: F_Born is scaled by P_full / P_Born, so the matched ratio is ``info_ratio_*`` x P_Born / P_full. This is the comparison per unit of scattered power.
  ``matched_power_ratios`` converts one into the other.

The multiple-scattering question ("does it add information about the gap?") is only answered by the second; the milestone-6 review showed that the conclusion
drawn from the first can flip (results/resolution/crb.json, ``table``: both ratios and the power ratio for every object).

Symmetry. The antenna rings and the symmetric object (cx = 0, dr = 0) are invariant under x -> -x. With the data permutation
P that swaps each antenna with its mirror image, the derivative of the data along an even parameter satisfies P D_k = D_k and
along an odd one P D_k = -D_k, so Re(D_even^H D_odd) = Re((P D_even)^H (P D_odd)) = -Re(D_even^H D_odd) = 0: F is block diagonal
across the even irrep (g, r, eps_r, cy) and the odd irrep (cx, dr). ``irrep_split`` returns the blocks and the relative size of
the cross block, which is a solver-precision residual at a symmetric object and O(1) at an asymmetric one. ``parity_residuals``
tests the statement under the F-blocks directly: ||P D_k - (+-1) D_k|| / ||D_k|| for every column. The same holds for the reflection
y -> -y when the object is centered (cy = 0): the rings are invariant under it too (angles phi -> -phi, multiples of 9 and 6 degrees),
cy is odd and (g, r, eps_r, cx, dr) are even, so at the centered symmetric object F also decouples cy from (g, r, eps_r); the
nuisance parameters that matter for sigma_CRB(g) are then r and eps_r only (``irrep_split(..., even=EVEN_PARAMS_Y, odd=ODD_PARAMS_Y)``).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from . import shapes as sh
from .likelihood import noise_std_from_snr

SYMMETRIC_PARAMS = ("g", "r", "eps_r", "cx", "cy")  # the family of DESIGN 13.2: common radius, common eps_r
EVEN_PARAMS_Y = ("g", "r", "eps_r", "cx", "dr")  # even under y -> -y (about y = 0): the object is invariant when cy = 0
ODD_PARAMS_Y = ("cy",)


# ----------------------------------------------------------------------------------------------
# Data derivatives
# ----------------------------------------------------------------------------------------------


def data_derivatives(forward, chi, dchi):
    """D (Nt Nr, K) complex: dd/dp_k = jvp of the forward map at ``chi`` along ``dchi[k]``.

    ``chi`` (n, n) real contrast, ``dchi`` (K, n, n) real shape derivatives. The primal solve is done once
    (``jax.linearize``) and each tangent costs one more Krylov solve with the same operator (the forward-mode rule of
    ``lax.custom_linear_solve``). Row index t Nr + r is d[t, r] (C order), as in ``jacobian``.
    """
    chi = jnp.asarray(chi)
    _, tangent = jax.linearize(lambda c: forward(c)[0], chi)
    tangent = jax.jit(tangent)
    cols = [tangent(jnp.asarray(dchi[k], dtype=chi.dtype)).reshape(-1) for k in range(len(dchi))]
    return jnp.stack(cols, axis=1)


def born_data_derivatives(forward, dchi):
    """Born D (Nt Nr, K): dd/dp_k = R (dchi_k u_inc), complex; ``dchi`` (K, n, n). Linear in dchi (no solve)."""
    dchi = jnp.asarray(dchi)
    return jnp.stack([forward.born(dchi[k]).reshape(-1) for k in range(dchi.shape[0])], axis=1)


def fisher_matrix(D, s):
    """F = (2 / s^2) Re(D^H D), (K, K) float64 NumPy, symmetrized (the symmetrization changes it by round-off only)."""
    D = jnp.asarray(D)
    F = (2.0 / float(s) ** 2) * np.asarray(jnp.real(D.conj().T @ D))
    return 0.5 * (F + F.T)


# ----------------------------------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------------------------------


def _indices(names, subset):
    return [list(names).index(k) for k in subset]


def crb_std(F, names, param="g", subset=None):
    """Cramer-Rao standard deviation of ``param``: sqrt((F_sub^-1)_pp), the other parameters of ``subset`` (default all of ``names``)
    unknown. ``F`` (K, K) is the Fisher matrix of ``names``. Returns inf if the sub-matrix is singular to double precision."""
    subset = tuple(names) if subset is None else tuple(subset)
    idx = _indices(names, subset)
    sub = np.asarray(F)[np.ix_(idx, idx)]
    p = subset.index(param)
    try:
        sol = np.linalg.solve(sub, np.eye(len(idx))[:, p])
    except np.linalg.LinAlgError:
        return float("inf")
    v = sol[p]
    return float(np.sqrt(v)) if v > 0 else float("inf")


def crb_std_known(F, names, param="g"):
    """sqrt(1 / F_pp): the bound when every other parameter is known."""
    i = _indices(names, [param])[0]
    return float(1.0 / np.sqrt(np.asarray(F)[i, i]))


def required_snr_db(sigma_ref, g, snr_ref_db=30.0):
    """SNR (dB) at which sigma_CRB(g) = g, given sigma_CRB at ``snr_ref_db`` (module docstring): ref + 20 log10(sigma_ref / g)."""
    return snr_ref_db + 20.0 * np.log10(np.asarray(sigma_ref, dtype=np.float64) / np.asarray(g, dtype=np.float64))


def scale_to_snr(F, snr_db, snr_ref_db=30.0):
    """F at another power SNR from F at ``snr_ref_db``: F 10^((snr - ref) / 10) (the object, hence mean|d|^2, is the same)."""
    return np.asarray(F) * 10.0 ** ((snr_db - snr_ref_db) / 10.0)


# ----------------------------------------------------------------------------------------------
# Symmetry
# ----------------------------------------------------------------------------------------------


def reflection_permutation(points, tol=1e-9, axis=0):
    """Index array p with points[p[i]] = mirror image of points[i], for an (N, 2) array: ``axis`` 0 mirrors x -> -x (default),
    ``axis`` 1 mirrors y -> -y. Raises ``ValueError`` if some point has no mirror image among the points (the layout is not
    symmetric under the reflection)."""
    pts = np.asarray(points, dtype=np.float64)
    flip = np.array([-1.0, 1.0]) if axis == 0 else np.array([1.0, -1.0])
    mirror = pts * flip
    dist = np.linalg.norm(mirror[:, None, :] - pts[None, :, :], axis=-1)
    perm = np.argmin(dist, axis=1)
    if dist[np.arange(len(pts)), perm].max() > tol or len(set(perm.tolist())) != len(pts):
        raise ValueError(f"the antenna layout is not symmetric under {'x' if axis == 0 else 'y'} -> -{'x' if axis == 0 else 'y'}")
    return perm


def mirror_data(d, perm_tx, perm_rx):
    """The data of the mirrored experiment, d_mirror[t, r] = d[perm_tx[t], perm_rx[r]]: (Nt, Nr) -> (Nt, Nr)."""
    return np.asarray(d)[np.ix_(perm_tx, perm_rx)]


def mirror_columns(D, perm_tx, perm_rx):
    """Apply the data permutation to every column of D (Nt Nr, K), rows t Nr + r as in ``data_derivatives``; same shape out."""
    D = np.asarray(D)
    nt, nr = len(perm_tx), len(perm_rx)
    cube = D.reshape(nt, nr, -1)
    return cube[np.ix_(perm_tx, perm_rx, np.arange(cube.shape[2]))].reshape(D.shape)


def parity_residuals(D, names, perm_tx, perm_rx, even=sh.EVEN_PARAMS, odd=sh.ODD_PARAMS):
    """Column parity of the data derivatives under a reflection: ``{name: ||P D_k - sigma_k D_k|| / ||D_k||}`` with P the data
    permutation of ``mirror_columns`` and sigma_k = +1 for ``even`` parameters, -1 for ``odd`` ones (parameters in neither list are
    skipped). Round-off of the solver (the Krylov tolerance) at a symmetric object; O(1) for the odd parameters at an asymmetric one."""
    D = np.asarray(D)
    PD = mirror_columns(D, perm_tx, perm_rx)
    out = {}
    for k, name in enumerate(names):
        if name in even:
            sign = 1.0
        elif name in odd:
            sign = -1.0
        else:
            continue
        out[name] = float(np.linalg.norm(PD[:, k] - sign * D[:, k]) / np.linalg.norm(D[:, k]))
    return out


def irrep_split(F, names, even=sh.EVEN_PARAMS, odd=sh.ODD_PARAMS):
    """Blocks of F over the even and odd parameters of ``names`` and the cross-block residual.

    Returns a dict: ``even`` and ``odd`` (the names present), ``F_even``, ``F_odd``, ``F_cross`` (nested lists),
    ``cross_residual`` = ||F_cross||_F / sqrt(||F_even||_F ||F_odd||_F) (0 when either block is empty) and ``cross_max_scaled`` =
    max |F_ij| / sqrt(F_ii F_jj) over cross pairs (the largest correlation between an even and an odd parameter).
    """
    F = np.asarray(F)
    ev = [k for k in names if k in even]
    od = [k for k in names if k in odd]
    ie, io = _indices(names, ev), _indices(names, od)
    Fe, Fo, Fc = F[np.ix_(ie, ie)], F[np.ix_(io, io)], F[np.ix_(ie, io)]
    res = float(np.linalg.norm(Fc) / np.sqrt(np.linalg.norm(Fe) * np.linalg.norm(Fo))) if ie and io else 0.0
    scaled = np.abs(Fc) / np.sqrt(np.outer(np.diag(F)[ie], np.diag(F)[io])) if ie and io else np.zeros((0, 0))
    return {"even": ev, "odd": od, "F_even": Fe.tolist(), "F_odd": Fo.tolist(), "F_cross": Fc.tolist(),
            "cross_residual": res, "cross_max_scaled": float(scaled.max()) if scaled.size else 0.0}


# ----------------------------------------------------------------------------------------------
# Matched-power Born ratio, the local regime of the bound, the detection limit
# ----------------------------------------------------------------------------------------------

LOCAL_REGIME_MAX_REL_SD = 0.1  # sd / value of the nuisances r and (eps_r - 1) above which the bound is not read as a statement about an estimator


def matched_power_ratios(ratio_known, ratio_unknown, power_full, power_born):
    """Information ratios full / Born with each model at its OWN power SNR, from those at one noise level (module docstring, Born comparison).

    ``ratio_known`` / ``ratio_unknown``: F_gg ratio and effective (sigma_Born / sigma_full)^2, both at the noise level of the full model; ``power_full`` /
    ``power_born``: mean|d|^2 of the full and the Born data of the same object. Returns ``(known_matched, unknown_matched, power_ratio)`` with
    matched = same-s ratio x P_Born / P_full and power_ratio = P_full / P_Born (scalars or arrays)."""
    scale = np.asarray(power_born, dtype=np.float64) / np.asarray(power_full, dtype=np.float64)
    return np.asarray(ratio_known) * scale, np.asarray(ratio_unknown) * scale, 1.0 / scale


def nuisance_report(F, names, snr_db, truth, snr_ref_db=30.0, subset=SYMMETRIC_PARAMS, n_data=None, max_rel_sd=LOCAL_REGIME_MAX_REL_SD):
    """Cramer-Rao standard deviations of every parameter of ``subset`` at ``snr_db``, and whether the bound can be read as local there.

    ``F`` (K, K) is the Fisher matrix of ``names`` at ``snr_ref_db``; at another SNR it is ``scale_to_snr`` and the standard deviations scale as
    10^(-(snr - ref) / 20). ``truth``: dict of the true parameter values (``eps_r`` is the contrast eps_r - 1 that the data measure). Returns a dict: ``sd`` (name ->
    sd), ``rel_sd`` (r: sd / r, eps_r: sd / (eps_r - 1)), ``aggregate_snr_db`` (snr + 10 log10(n_data), if ``n_data``) and ``local_regime`` = both relative sds of r
    and eps_r <= ``max_rel_sd``. When it is False the bound at this SNR is a lower bound of unknown tightness: the model's own nuisance parameters are poorly
    determined, the estimator is not in the linear regime, and a gap whose bound equals g (sigma_CRB(g) = g) is not a required SNR (milestone-6 review)."""
    sub = tuple(subset)
    idx = _indices(names, sub)
    Fs = scale_to_snr(np.asarray(F), snr_db, snr_ref_db)[np.ix_(idx, idx)]
    sd = np.sqrt(np.diag(np.linalg.inv(Fs)))
    sd_d = dict(zip(sub, sd.tolist()))
    rel = {}
    if "r" in sd_d:
        rel["r"] = sd_d["r"] / abs(truth["r"])
    if "eps_r" in sd_d:
        rel["eps_r"] = sd_d["eps_r"] / abs(truth["eps_r"] - 1.0)
    out = {"snr_db": float(snr_db), "sd": sd_d, "rel_sd": rel, "max_rel_sd": float(max_rel_sd),
           "local_regime": bool(rel and all(v <= max_rel_sd for v in rel.values()))}
    if n_data is not None:
        out["aggregate_snr_db"] = float(snr_db + 10.0 * np.log10(n_data))
    return out


def detection_limit(sigma0, alpha=0.05, power=0.9):
    """Smallest gap detectable against g = 0 in the Gaussian (Wald/likelihood-ratio) approximation of the bound: (z_(1-alpha) + z_power) sigma0.

    One-sided test of H0: g = 0 against H1: g > 0 at level ``alpha`` with power ``power`` when the estimate of g has standard deviation ``sigma0`` (the Cramer-Rao
    standard deviation AT g = 0, nuisances unknown). 2.93 sigma0 for alpha 0.05 and power 0.9, i.e. 9.3 dB of SNR above Smith's sigma_CRB(g) = g criterion.
    A local approximation on the boundary g = 0 of the parameter space (the null distribution of the likelihood ratio is a half-chi-square mixture; the quantile
    used is that of the one-sided Wald test, which is the same at level alpha < 0.5)."""
    return float((norm.ppf(1.0 - alpha) + norm.ppf(power)) * np.asarray(sigma0, dtype=np.float64))


# ----------------------------------------------------------------------------------------------
# One object
# ----------------------------------------------------------------------------------------------


def _timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - t0


def analyze(forward, obj, snr_ref_db=30.0, names=sh.PARAM_NAMES, born=True):
    """Fisher matrix, bounds and symmetry blocks of one object on the grid of ``forward``; returns a JSON-serializable dict.

    Fields: ``names``; ``s`` (noise level at ``snr_ref_db``), ``power`` (mean |d|^2); ``F`` (K x K, full model; at the reference SNR);
    ``cond_F_scaled`` (condition number of F scaled to unit diagonal); ``crb_g_unknown`` (the symmetric family ``SYMMETRIC_PARAMS``
    unknown), ``crb_g_unknown_all`` (all of ``names``, including dr), ``crb_g_known``; the same three with ``_born`` for the Born
    forward and ``F_born``; ``info_ratio_known`` = F_gg / F_gg(Born) and ``info_ratio_unknown`` =
    (crb_born / crb)^2 (the effective information about g, nuisances unknown, full over Born) at ONE noise level, ``power_born`` (mean |d|^2 of the Born data),
    ``power_ratio_full_over_born`` and ``info_ratio_known_matched`` / ``info_ratio_unknown_matched`` (each model at its own power SNR, ``matched_power_ratios``); ``required_snr_db`` for the three
    full-model bounds at the gap of ``obj`` (``snr_ref_db`` + 20 log10(sigma / g)); ``irreps`` (``irrep_split`` of the full F);
    ``irreps_y`` (the same split under y -> -y), ``parity_x`` / ``parity_y`` (``parity_residuals`` of the columns of D),
    ``grid`` (n, h), ``solver_residual_max`` (largest relative true residual of the primal solve, to be compared with
    ``solver_tol``; the tangent solves use the same operator and tolerance) and ``timing_s`` (the forward solve, the K tangents with the
    linearization, the Born derivatives). ``forward`` must have antennas that are symmetric under x -> -x and y -> -y for the irrep splits to mean anything
    (``ValueError`` otherwise).
    """
    grid = forward.grid
    chi = obj.chi(grid)
    dchi = sh.shape_derivatives(grid, obj, names)
    (d, u), t_fwd = _timed(lambda: forward(jnp.asarray(chi)))
    s = noise_std_from_snr(d, snr_ref_db)
    D, t_jvp = _timed(lambda: data_derivatives(forward, chi, dchi))
    F = fisher_matrix(D, s)
    out = {"names": list(names), "grid": {"n": grid.n, "side": grid.side, "h": grid.h}, "snr_ref_db": snr_ref_db,
           "solver_residual_max": float(jnp.max(forward.residual(jnp.asarray(chi), u))), "solver_tol": forward.tol,
           "params": {k: getattr(obj, k) for k in sh.PARAM_NAMES}, "s": s, "power": float(jnp.mean(jnp.abs(d) ** 2)),
           "F": F.tolist()}
    dg = np.sqrt(np.diag(F))
    out["cond_F_scaled"] = float(np.linalg.cond(F / np.outer(dg, dg)))
    sub = [k for k in SYMMETRIC_PARAMS if k in names]
    out["crb_g_unknown"] = crb_std(F, names, "g", sub)
    out["crb_g_unknown_all"] = crb_std(F, names, "g")
    out["crb_g_known"] = crb_std_known(F, names, "g")
    out["required_snr_db"] = {key: float(required_snr_db(out[f"crb_g_{key}"], obj.g, snr_ref_db)) if obj.g > 0 else float("nan")
                              for key in ("unknown", "unknown_all", "known")}
    out["irreps"] = irrep_split(F, names)
    out["irreps_y"] = irrep_split(F, names, EVEN_PARAMS_Y, ODD_PARAMS_Y)
    ant = forward.setup.antennas
    px = (reflection_permutation(ant.tx), reflection_permutation(ant.rx))
    py = (reflection_permutation(ant.tx, axis=1), reflection_permutation(ant.rx, axis=1))
    out["parity_x"] = parity_residuals(np.asarray(D), names, *px)
    out["parity_y"] = parity_residuals(np.asarray(D), names, *py, even=EVEN_PARAMS_Y, odd=ODD_PARAMS_Y)
    timing = {"forward_s": t_fwd, "tangents_s": t_jvp}
    if born:
        Db, t_born = _timed(lambda: born_data_derivatives(forward, dchi))
        Fb = fisher_matrix(Db, s)
        out["F_born"] = Fb.tolist()
        out["crb_g_unknown_born"] = crb_std(Fb, names, "g", sub)
        out["crb_g_unknown_all_born"] = crb_std(Fb, names, "g")
        out["crb_g_known_born"] = crb_std_known(Fb, names, "g")
        ig = _indices(names, ["g"])[0]
        out["info_ratio_known"] = float(F[ig, ig] / Fb[ig, ig])
        out["info_ratio_unknown"] = float((out["crb_g_unknown_born"] / out["crb_g_unknown"]) ** 2)
        d_born = forward.born(jnp.asarray(chi))
        out["power_born"] = float(jnp.mean(jnp.abs(d_born) ** 2))
        km, um, pr = matched_power_ratios(out["info_ratio_known"], out["info_ratio_unknown"], out["power"], out["power_born"])
        out["power_ratio_full_over_born"] = float(pr)
        out["info_ratio_known_matched"], out["info_ratio_unknown_matched"] = float(km), float(um)
        timing["born_s"] = t_born
    out["timing_s"] = timing
    return out
