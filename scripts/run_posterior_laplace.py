"""Laplace-preconditioned posterior sampling (prototype; NOT part of the specified ladder).

Same pipeline, flags and outputs as scripts/run_posterior.py (it imports and runs that script's ``main``), with one
change between the MAP and the chains: the samplers run in Laplace-whitened coordinates zeta instead of xi,

    xi = xi_c + M zeta,   M = Sigma^{1/2},   Sigma = (I + V diag(lam) V^T)^{-1},   M = I - V diag(1 - (1+lam)^{-1/2}) V^T,

where (lam, V) are the K largest eigenpairs of the Gauss-Newton Hessian of Phi at the MAP in whitened
coordinates, H_GN = (2 / s^2) J_r^T J_r with J_r = d [Re d; Im d] / d xi (jvp then vjp of the forward model, so
the exact adjoint Jacobian-vector products; found with Lanczos, ``scipy.sparse.linalg.eigsh``), and xi_c is the
center of the map: the MAP, shifted along the stiffest direction by the log-normal mean-versus-mode correction
(``--laplace-recenter lognormal``, default; ``none`` keeps xi_c = MAP; the shift is capped at
``--laplace-recenter-max-sd`` Laplace standard deviations, default 100, because a chain started far from the posterior
mass without its unresolved fluctuations does not recover, see ``inference.lognormal_center_shift``). The prior is on theta = log eps_r
(``--link log_eps``, the default), so the data fix the DOI average of eps_r = exp(theta) and the unresolved
fluctuations of theta raise it by Var/2; the posterior mean of theta therefore lies about Var/2 below the MAP,
which is tens of Laplace standard deviations along the stiffest direction (``inference.lognormal_center_shift``).
The target in zeta,

    pi(zeta) ~ exp(-Phi'(zeta) - |zeta|^2 / 2),   Phi'(zeta) = Phi(xi) + |xi|^2 / 2 - |zeta|^2 / 2,   grad Phi' = M^T (grad Phi + xi) - zeta,

has exactly the posterior of xi as its law (the Jacobian of an affine map is constant), so every sampler of
``bayes_scatter.samplers`` applies unchanged; if the posterior were Gaussian, zeta ~ N(0, I).  Draws are
recorded in physical space (log eps_r and eps_r).  The Laplace factor is a fixed (state-independent) affine
map, hence MCMC validity is untouched.  The directions beyond the K computed eigenpairs are treated as
prior-like (curvature <= lam_K, reported).

IMPORTANT: the potential the sampler sees, hence ``res.phi`` and the trace in chains.npz['phi'], is Phi', NOT the
data misfit Phi.  Phi' differs from Phi by the state dependent offset (|zeta|^2 - |xi|^2) / 2, which is in the
thousands.  The physical misfit of the stored draws is recomputed from their eps_r maps
(``inference.physical_phi_of_draws``) and reported as summary["phi_physical"] and chains.npz['phi_physical'];
summary["phi"]["kind"] marks the sampler trace as reparametrized.

Chain starts (in zeta): ``--init map`` starts at ``--init-noise`` * N(0, I), i.e. at the center of the map;
``--laplace-init-scale c`` starts at c * N(0, I) (overdispersed around the center); ``--init prior`` maps prior
draws through M^{-1} (far in the tails: Phi is 10^5 to 10^6 times the MAP value there).

Extra flags: --laplace-k (number of eigenpairs, default 150), --laplace-tol (eigsh tolerance), --laplace-cache PATH
(.npz; reuse the eigenpairs and MAP center stored there, or compute and store them if the file does not exist; a
cache computed at a different MAP is refused), --laplace-recenter {lognormal,none}, --laplace-recenter-max-sd,
--laplace-init-scale c.

Writes the same results/<out>/ files as run_posterior.py plus a "laplace" entry in summary.json.
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_posterior as rp  # noqa: E402
from bayes_scatter import inference as inf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rp.STEP0.update({"pcn": 0.3, "pcn_langevin": 0.3, "mala": 0.3, "hmc": 0.3})  # O(1) steps after preconditioning
STATE = {}
PHI_PRIME_KIND = "reparametrized Phi' = Phi + |xi|^2/2 - |zeta|^2/2"


class LaplaceTarget:
    """Duck-typed ``Target`` in zeta coordinates (see the module docstring); what ``inference.sample`` needs."""

    use_born = False

    def __init__(self, target, xi_center, lam, V):
        self.target = target
        self.shape = tuple(xi_center.shape)
        self.xi_center = jnp.asarray(xi_center).ravel()
        self.lam = np.asarray(lam)
        self.V = jnp.asarray(V)  # (P^2, K)
        self.c_m = jnp.asarray(1.0 - (1.0 + self.lam) ** -0.5)  # M = I - V diag(c_m) V^T
        self.c_minv = jnp.asarray((1.0 + self.lam) ** 0.5 - 1.0)  # M^-1 = I + V diag(c_minv) V^T
        shape, xi_c, Vm, c_m, c_minv = self.shape, self.xi_center, self.V, self.c_m, self.c_minv

        def to_xi(zeta):
            z = zeta.ravel()
            return (xi_c + z - Vm @ (c_m * (Vm.T @ z))).reshape(shape)

        def to_zeta(xi):
            d = xi.ravel() - xi_c
            return (d + Vm @ (c_minv * (Vm.T @ d))).reshape(shape)

        def phi_and_grad(zeta, aux):
            xi = to_xi(zeta)
            phi, g, aux_new = target.phi_fn(xi, aux)
            gu = (g + xi).ravel()
            gz = (gu - Vm @ (c_m * (Vm.T @ gu))).reshape(shape)
            return phi + 0.5 * jnp.sum(xi * xi) - 0.5 * jnp.sum(zeta * zeta), gz - zeta, aux_new

        def phi_only(zeta, aux):
            xi = to_xi(zeta)
            phi, aux_new = target.potential(xi, aux)
            return phi + 0.5 * jnp.sum(xi * xi) - 0.5 * jnp.sum(zeta * zeta), None, aux_new

        self.to_xi = jax.jit(to_xi)
        self.to_zeta = jax.jit(to_zeta)
        self._phi_fn = jax.jit(phi_and_grad)
        self._phi_fn_no_grad = jax.jit(phi_only)
        self._record = lambda zeta: target.record(to_xi(zeta))

    @property
    def phi_fn(self):
        return self._phi_fn

    @property
    def phi_fn_no_grad(self):
        return self._phi_fn_no_grad

    @property
    def record(self):
        return self._record

    def initial_aux(self):
        return self.target.initial_aux()


def mala_step_limit(lam_max):
    """Largest stable MALA step 4 / (1 + lam_max) for a Gaussian with top curvature 1 + lam_max.

    In a direction of curvature a the MALA proposal mean is (1 - h a / 2) u, which contracts only for
    h < 4 / a (the acceptance-scale rule of thumb 2 / a is half of this)."""
    return 4.0 / (1.0 + float(lam_max))


def pcn_langevin_step_limit(lam_max):
    """Stability bound delta < 2 / lam_max of the pCN-Langevin step (it acts on Phi only, curvature lam_max)."""
    return 2.0 / float(lam_max)


def gn_eigs(target, xi_map, k, tol):
    """K largest eigenpairs of the whitened Gauss-Newton Hessian (2/s^2) J_r^T J_r at xi_map. Returns lam, V, n_matvec."""
    shape = tuple(xi_map.shape)
    _, _, aux_map = target.potential_and_grad(xi_map, target.initial_aux())
    s2 = target.s**2

    def f(xi):
        d, _ = target.forward(target.link(target.field.theta(xi)), aux_map)
        return jnp.concatenate([jnp.real(d).ravel(), jnp.imag(d).ravel()])

    _, jvp_fn = jax.linearize(f, xi_map)
    _, vjp_fn = jax.vjp(f, xi_map)

    @jax.jit
    def gn(v):
        return (2.0 / s2) * vjp_fn(jvp_fn(v.reshape(shape)))[0].ravel()

    count = [0]

    def mv(v):
        count[0] += 1
        return np.asarray(gn(jnp.asarray(v, dtype=jnp.float64)))

    N = int(np.prod(shape))
    op = LinearOperator((N, N), matvec=mv, dtype=np.float64)
    lam, V = eigsh(op, k=k, which="LA", tol=tol, ncv=min(N - 1, 2 * k + 1))
    order = np.argsort(lam)[::-1]
    return lam[order], V[:, order], count[0]


_map_orig = inf.map_estimate
_sample_orig = inf.sample


def map_wrapper(target, xi0, maxiter=200, **kw):
    out = _map_orig(target, xi0, maxiter=maxiter, **kw)
    STATE["xi_map"] = out[0]
    STATE["map_nfev"] = int(out[2]["nfev"])
    return out


def load_or_compute_eigs(target, xi_map, k, tol, cache):
    """Eigenpairs (lam, V), their center, the matvec count and the build time, from the cache if it exists."""
    if cache and Path(cache).exists():  # reuse eigenpairs computed by an earlier run at the same MAP
        zc = np.load(cache)
        xi_c = jnp.asarray(zc["xi_map"])
        rel = float(jnp.linalg.norm(xi_c - xi_map) / jnp.linalg.norm(xi_map))
        if rel > 1e-3:
            raise ValueError(f"Laplace cache {cache} was computed at a different MAP (relative difference {rel:.2e}); "
                             "use a cache file per problem")
        time_s = float(zc["time_s"]) if "time_s" in zc.files else None
        return zc["lam"], zc["V"], int(zc["nmv"]), xi_c, time_s, True
    t0 = time.perf_counter()
    lam, V, nmv = gn_eigs(target, xi_map, k, tol)
    time_s = time.perf_counter() - t0
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lam=lam, V=V, nmv=nmv, xi_map=np.asarray(xi_map), time_s=time_s)
    return lam, V, nmv, xi_map, time_s, False


def sample_wrapper(target, sampler, key, xi0s, **kw):
    k = STATE["k"]
    cache = STATE.get("cache")
    xi_map = STATE["xi_map"]
    lam, V, nmv, xi_map_c, t_eig, reused = load_or_compute_eigs(target, xi_map, k, STATE["tol"], cache)
    k = len(lam)

    recenter = STATE["recenter"]
    if recenter == "lognormal":
        delta, shift = inf.lognormal_center_shift(target.field, lam, V, max_sd=STATE["recenter_max_sd"])
        xi_c = xi_map_c + delta
    else:
        xi_c, shift = xi_map_c, None
    lt = LaplaceTarget(target, xi_c, lam, V)
    STATE["xi_center"] = np.asarray(xi_c)

    scale = STATE["init_scale"]
    n_chains = xi0s.shape[0]
    k_z = jax.random.fold_in(key, 7)
    if scale is not None:
        z0s = scale * jax.random.normal(k_z, xi0s.shape, dtype=jnp.float64)  # overdispersed around the center
        init = "overdispersed"
    elif rp_args["init"] == "map":
        z0s = rp_args["init_noise"] * jax.random.normal(k_z, xi0s.shape, dtype=jnp.float64)
        init = "map"
    else:
        z0s = jax.vmap(lt.to_zeta)(xi0s)  # prior draws mapped through M^-1
        init = "prior"
    res, info = _sample_orig(lt, sampler, key, z0s, **kw)

    # diagnostics in zeta at the final states, then back to xi
    zeta_final = res.xi
    proj = jnp.einsum("pk,cp->ck", lt.V, zeta_final.reshape(n_chains, -1))  # V^T zeta per chain
    zeta_v0 = np.asarray(proj[:, 0])
    proj_sq = float(jnp.mean(jnp.sum(proj**2, axis=1)))
    rest_sq = float(jnp.mean(jnp.sum(zeta_final.reshape(n_chains, -1) ** 2, axis=1))) - proj_sq
    res = res._replace(xi=jax.vmap(lt.to_xi)(res.xi))

    # the physical data misfit of the stored draws (the sampler's own trace is Phi', not Phi)
    t0 = time.perf_counter()
    info["phi_physical"] = inf.physical_phi_of_draws(target, res.samples[:, :, 1])
    info["phi_trace_kind"] = PHI_PRIME_KIND
    t_phys = time.perf_counter() - t0

    theta_map_mean = float(jnp.mean(target.field.theta(xi_map_c)))
    theta_post_mean = float(jnp.mean(res.mean[:, 0]))  # Welford mean of theta over all post-warmup steps
    sd_lam = float(np.sqrt(1.0 + lam[0]))
    STATE["laplace"] = {
        "k": int(k), "eigsh_tol": STATE["tol"], "matvecs_jvp_plus_vjp": int(nmv), "time_s": t_eig,
        "cache": None if not cache else str(Path(cache)), "cache_reused": bool(reused),
        "lam_max": float(lam[0]), "lam_min_computed": float(lam[-1]),
        "n_lam_gt_1.5": int((lam > 1.5).sum()), "n_lam_gt_10": int((lam > 10).sum()),
        "n_lam_gt_100": int((lam > 100).sum()), "n_lam_gt_1e4": int((lam > 1e4).sum()),
        "n_lam_gt_1e6": int((lam > 1e6).sum()),
        "implied_max_stable_mala_step_in_xi": mala_step_limit(lam[0]),
        "implied_max_stable_pcn_langevin_delta_in_xi": pcn_langevin_step_limit(lam[0]),
        "init": init, "init_scale": scale,
        "map_nfev": STATE.get("map_nfev"),
        "setup_grad_evals": (STATE.get("map_nfev") or 0) + int(nmv),
        "setup_grad_evals_note": "MAP value-and-gradient evaluations + eigsh matvecs (one jvp + one vjp each, "
                                 "counted as one gradient evaluation)",
        "recenter": recenter, "recenter_info": shift,
        "zeta_v0_final_per_chain": zeta_v0.tolist(),  # final zeta along the stiffest eigenvector (0 = at the center)
        "mean_abs_zeta_proj_sq_final": proj_sq, "expected_proj_sq_gaussian": float(k),
        "mean_zeta_rest_sq_final": rest_sq, "expected_rest_sq_gaussian": float(np.prod(lt.shape) - k),
        "theta_mean_post_minus_map": theta_post_mean - theta_map_mean,
        "theta_mean_post_minus_map_note": "mean over pixels of the posterior mean of log eps_r minus its value at "
                                          "the MAP; the log-normal prediction is -Var/2 (recenter_info)",
        "laplace_sd_along_v0": 1.0 / sd_lam,
        "physical_phi_time_s": t_phys,
    }
    np.save(STATE["out_dir"] / "laplace_lam.npy", lam)
    print(f"Laplace: K={k} eigenpairs ({'cached' if reused else f'{t_eig:.1f} s'}, {nmv} jvp+vjp products): "
          f"lam_max {lam[0]:.3g}, lam_K {lam[-1]:.3g}; #lam>1.5: {STATE['laplace']['n_lam_gt_1.5']}, "
          f">10: {STATE['laplace']['n_lam_gt_10']}, >100: {STATE['laplace']['n_lam_gt_100']}, "
          f">1e4: {STATE['laplace']['n_lam_gt_1e4']}")
    if shift is not None:
        print(f"re-centered along the stiffest direction by {shift['n_sd']:.1f} Laplace sd (mean log eps_r "
              f"{shift['theta_mean_shift']:+.5f}; -Var/2 = {shift['theta_mean_shift_full']:+.5f} would be "
              f"{shift['n_sd_full']:.1f} sd{', capped' if shift['capped'] else ''}); posterior mean of log eps_r "
              f"minus MAP: {theta_post_mean - theta_map_mean:+.5f}; final zeta along v0: {np.round(zeta_v0, 1).tolist()}")
    return res, info


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # strip our own flags
    STATE["k"], STATE["tol"] = 150, 1e-4
    STATE["cache"] = None
    STATE["recenter"] = "lognormal"
    STATE["recenter_max_sd"] = 100.0
    STATE["init_scale"] = None
    for flag, key, cast in (("--laplace-k", "k", int), ("--laplace-tol", "tol", float),
                            ("--laplace-cache", "cache", str), ("--laplace-recenter", "recenter", str),
                            ("--laplace-init-scale", "init_scale", float),
                            ("--laplace-recenter-max-sd", "recenter_max_sd", float)):
        if flag in argv:
            i = argv.index(flag)
            STATE[key] = cast(argv[i + 1])
            del argv[i:i + 2]
    if STATE["recenter"] not in ("lognormal", "none"):
        raise SystemExit("--laplace-recenter must be 'lognormal' or 'none'")
    args = rp.parse_args(argv)
    global rp_args
    rp_args = vars(args)
    STATE["out_dir"] = Path(args.out) if Path(args.out).is_absolute() else ROOT / "results" / args.out
    STATE["out_dir"].mkdir(parents=True, exist_ok=True)
    rp.map_estimate = map_wrapper
    rp.sample = sample_wrapper
    summary = rp.main(argv)
    summary["laplace"] = STATE["laplace"]
    (STATE["out_dir"] / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    # the center of the map (needed to rebuild zeta from the stored final states) next to the chains
    chains = dict(np.load(STATE["out_dir"] / "chains.npz"))
    chains["xi_center"] = STATE["xi_center"]
    np.savez_compressed(STATE["out_dir"] / "chains.npz", **chains)
    return summary


if __name__ == "__main__":
    main()
