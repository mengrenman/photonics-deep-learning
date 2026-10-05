"""Inference utilities (docs/DESIGN.md section 7): MAP, exact Born posterior, error metrics, sampler driver.

* ``map_estimate``: the regularized nonlinear least-squares baseline (Tikhonov MAP), U = Phi + |xi|^2/2
  minimized with SciPy L-BFGS-B on a jitted value-and-grad.
* ``born_gaussian_posterior``: exact Gaussian posterior of the Born model with the identity link, the
  reference for the sampler validation (DESIGN 8, step 5).
* ``err_model``, ``err_chi``, ``err_data``: reconstruction metrics of the book (eqs. 15.21, 15.22) and the
  contrast-normalized variant.
* ``operating_point``: the forward-model error of the inference grid on the truth, in units of the noise
  (the check behind "no inverse crime, but is the discretization good enough?").
* ``physical_phi_of_draws``: the data misfit Phi of stored posterior draws, recomputed from their eps_r maps.
* ``laplace_theta_variance``, ``lognormal_center_shift``: pixel variance of theta = log eps_r under the
  Laplace (Gauss-Newton) posterior, and the log-normal mean-versus-mode shift of its center.
* ``sample``: run one of the four samplers on a ``Target``, several chains, with timing and evaluation counts.
* ``compare_to_gaussian``: standardized errors of chain estimates against a known Gaussian (mean and
  variance), the statistic behind the Born end-to-end test.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from . import diagnostics as dg
from . import samplers as sm
from .likelihood import misfit
from .priors import Identity, get_link

# ----------------------------------------------------------------------------------------------
# MAP
# ----------------------------------------------------------------------------------------------


def map_estimate(target, xi0, maxiter=200, gtol=1e-6, ftol=1e-13):
    """Maximum a posteriori estimate: minimize U(xi) = Phi(xi) + |xi|^2 / 2 by L-BFGS-B.

    The objective and its gradient come from ``target.phi_fn`` under ``jax.jit``; the optimizer loop runs on
    the host. The forward solves are warm-started from the fields of the previous evaluation (a pure
    accelerator, see ``likelihood``).

    Parameters
    ----------
    target : ``likelihood.Target``.
    xi0 : starting point, shape ``target.field.xi_shape`` (zeros is the prior mean, theta = mu).
    maxiter : maximum L-BFGS-B iterations.
    gtol : stop when the largest component of grad U is below ``gtol`` (U has a unit-curvature prior term,
        so this is also roughly the error of xi in the prior-dominated directions).
    ftol : relative decrease of U below which the iteration stops (SciPy's ``ftol``); the default is
        tiny so that ``gtol`` and ``maxiter`` decide.

    Returns
    -------
    xi_map : jnp array, shape of ``xi0``.
    chi_map : numpy array (n, n), the contrast at xi_map.
    info : dict with ``success``, ``message``, ``nit``, ``nfev``, ``U``, ``phi``, ``grad_inf`` (max-norm of
        grad U at xi_map), ``time_s`` and ``U_history`` (U at every function evaluation).
    """
    xi0 = jnp.asarray(xi0, dtype=jnp.float64)
    shape = xi0.shape
    phi_fn = target.phi_fn

    @jax.jit
    def value_and_grad(x, aux):
        xi = x.reshape(shape)
        phi, grad, aux_new = phi_fn(xi, aux)
        return phi + 0.5 * jnp.sum(xi * xi), (grad + xi).ravel(), phi, aux_new

    state = {"aux": target.initial_aux()}
    history = []

    def fun(x):
        u, g, phi, aux_new = value_and_grad(jnp.asarray(x), state["aux"])
        u = float(u)
        if np.isfinite(u):  # keep the last good warm start if a line-search trial diverged
            state["aux"] = aux_new
        history.append(u)
        return u, np.asarray(g, dtype=np.float64)

    t0 = time.perf_counter()
    res = minimize(fun, np.asarray(xi0, dtype=np.float64).ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": float(ftol),
                            "maxcor": 30, "maxfun": 10 * int(maxiter) + 100})
    time_s = time.perf_counter() - t0

    xi_map = jnp.asarray(res.x).reshape(shape)
    u, g, phi, _ = value_and_grad(jnp.asarray(res.x), state["aux"])
    info = {"success": bool(res.success), "message": str(res.message), "nit": int(res.nit),
            "nfev": int(res.nfev), "U": float(u), "phi": float(phi),
            "grad_inf": float(jnp.max(jnp.abs(g))), "time_s": time_s, "U_history": np.asarray(history)}
    return xi_map, np.asarray(target.chi(xi_map)), info


# ----------------------------------------------------------------------------------------------
# Exact Born posterior
# ----------------------------------------------------------------------------------------------


class BornPosterior(NamedTuple):
    """Exact Gaussian posterior of the Born model with the identity link (NumPy float64).

    mean_xi (P*P,), precision_xi (P*P, P*P), cov_xi = precision^-1; mean_chi (n*n,), cov_chi (n*n, n*n).
    Vectors are C-order flattenings of the (P, P) xi array and the (n, n) chi array.
    """

    mean_xi: np.ndarray
    precision_xi: np.ndarray
    cov_xi: np.ndarray
    mean_chi: np.ndarray
    cov_chi: np.ndarray


def born_gaussian_posterior(forward, field, d_obs, s, link="identity"):
    """Exact posterior of xi for d_obs = A chi + e, e ~ CN(0, s^2), chi = mu + L xi, xi ~ N(0, I).

    With B = A L (complex), stacked real B_r = [Re B; Im B], y_r = [Re(d_obs - A mu); Im(d_obs - A mu)] and
    per-real-component noise variance sigma_r^2 = s^2 / 2 (DESIGN 7):

        precision  Lambda = I + B_r^T B_r / sigma_r^2,    mean m_xi = Lambda^-1 B_r^T y_r / sigma_r^2,

    and on chi: mean mu + L m_xi, covariance L Lambda^-1 L^T. Data rows are ordered t * Nr + r (C order of
    ``d_obs``). Small grids only: it builds the dense (n^2, P^2) matrix L and the (P^2, P^2) precision.

    The posterior is Gaussian only if xi -> chi is affine, so ``link`` must be the identity link
    (``"identity"``, the default); any other link, in particular ``"log_eps"``, raises ``ValueError``. For a
    noise level so small that the precision matrix is numerically singular (condition number beyond about
    1e15) the Cholesky factorization raises ``numpy.linalg.LinAlgError``.
    """
    if not isinstance(get_link(link), Identity):
        raise ValueError("the exact Born posterior is Gaussian only for the identity link "
                         "(xi -> chi must be affine); got a nonlinear link")
    A = np.asarray(forward.born_matrix())  # (Nt*Nr, n^2)
    L = field.sqrt_matrix()  # (n^2, P^2)
    mu = np.asarray(field.mean, dtype=np.float64).ravel()
    d = np.asarray(d_obs).ravel()
    if d.size != A.shape[0]:
        raise ValueError(f"d_obs has {d.size} entries, expected {A.shape[0]}")
    B = A @ L
    y = d - A @ mu
    Br = np.vstack([B.real, B.imag])
    yr = np.concatenate([y.real, y.imag])
    sigma_r2 = 0.5 * float(s) ** 2

    precision = np.eye(L.shape[1]) + Br.T @ Br / sigma_r2
    factor = cho_factor(precision)
    mean_xi = cho_solve(factor, Br.T @ yr / sigma_r2)
    cov_xi = cho_solve(factor, np.eye(L.shape[1]))
    cov_xi = 0.5 * (cov_xi + cov_xi.T)
    return BornPosterior(mean_xi, precision, cov_xi, mu + L @ mean_xi, L @ cov_xi @ L.T)


# ----------------------------------------------------------------------------------------------
# Reconstruction metrics
# ----------------------------------------------------------------------------------------------


def _relative_error(rec, true):
    rec, true = np.asarray(rec), np.asarray(true)
    return float(np.linalg.norm((rec - true).ravel()) / np.linalg.norm(true.ravel()))


def err_model(eps_rec, eps_true):
    """Book eq. 15.21: ||m_R - m_T|| / ||m_T|| with m = eps_r (relative permittivity, not the contrast).

    The background eps_r = 1 is part of m, so this is dominated by background pixels and compresses the
    error in fine structure; compare ``err_chi``.
    """
    return _relative_error(eps_rec, eps_true)


def err_chi(chi_rec, chi_true):
    """Contrast-normalized error ||chi_R - chi_T|| / ||chi_T|| with chi = eps_r - 1 (vacuum gives 1)."""
    return _relative_error(chi_rec, chi_true)


def err_data(d_rec, d_true):
    """Book eq. 15.22: ||d_R - d_T|| / ||d_T||, with d_R the data of the reconstruction and d_T the
    noise-free data of the truth (complex arrays of any shape)."""
    return _relative_error(d_rec, d_true)


def operating_point(forward, chi_true, d_clean, d_obs, s):
    """Grid-induced model error at the truth, compared with the noise (DESIGN 8, step 6).

    ``forward`` is the inference-grid model, ``chi_true`` the coverage-averaged truth on that grid, ``d_clean``
    the noise-free data synthesized on the finer data grid and ``d_obs = d_clean + noise`` with noise level
    ``s``. Returns a dict with

    * ``rel_model_error`` = ||F(chi_true) - d_clean|| / ||d_clean||, the data error of the discretization;
    * ``rel_noise`` = s sqrt(N) / ||d_clean|| (N complex entries), the noise amplitude on the same scale;
    * ``model_to_noise`` = their ratio;
    * ``phi_truth`` = Phi(chi_true) on the inference grid and ``phi_clean`` = Phi of the fine-grid noise-free
      data, both against ``d_obs`` (``phi_clean`` is the realized noise energy, about N);
    * ``phi_excess_sd`` = (phi_truth - phi_clean) / sqrt(N), in standard deviations of Phi (sd of Phi at the
      truth with no model error is sqrt(N)).

    Since E[phi_truth - phi_clean] = (model_to_noise)^2 N, a model error of only 0.2 of the noise amplitude
    already shifts Phi at the truth by 0.04 N, which is 2 sqrt(N) for N = 2400. A large ``phi_excess_sd`` does
    not mean the posterior is wrong, but it does mean the fit can only reach N by moving chi away from the
    truth, so reconstruction errors near edges are partly discretization error. Returns plain floats.
    """
    d_model, _ = forward(jnp.asarray(chi_true))
    d_model = np.asarray(d_model)
    d_clean = np.asarray(d_clean)
    n_data = d_clean.size
    norm = float(np.linalg.norm(d_clean))
    rel_model = float(np.linalg.norm(d_model - d_clean)) / norm
    rel_noise = float(s) * np.sqrt(n_data) / norm
    phi_truth = float(misfit(d_model, d_obs, s))
    phi_clean = float(misfit(d_clean, d_obs, s))
    return {"n_data": int(n_data), "rel_model_error": rel_model, "rel_noise": float(rel_noise),
            "model_to_noise": rel_model / float(rel_noise), "phi_truth": phi_truth, "phi_clean": phi_clean,
            "phi_excess_sd": (phi_truth - phi_clean) / np.sqrt(n_data)}


def physical_phi_of_draws(target, eps_draws):
    """Data misfit Phi of stored draws: ``eps_draws`` (chains, N, n, n) of eps_r -> Phi array (chains, N).

    Uses ``target.misfit_of_chi`` with chi = eps_r - 1, each chain warm-started along its own draws (a pure
    accelerator, Phi does not depend on it beyond the solver tolerance). This is the potential that the
    data constrain, whatever coordinates the sampler ran in; in Laplace-whitened coordinates the sampler's own
    potential is the reparametrized Phi' = Phi + |xi|^2/2 - |zeta|^2/2, which is not the misfit.
    """
    eps_draws = jnp.asarray(eps_draws, dtype=jnp.float64)

    @jax.jit
    def one_chain(draws):
        def step(aux, eps):
            phi, aux = target.misfit_of_chi(eps - 1.0, aux)
            return aux, phi

        return jax.lax.scan(step, target.initial_aux(), draws)[1]

    return np.stack([np.asarray(one_chain(d)) for d in eps_draws])


def laplace_theta_variance(field, lam, V):
    """Pixel variance of theta = mu + L xi under the Laplace posterior xi ~ N(xi_c, Sigma), shape (n, n).

    Sigma = (I + V diag(lam) V^T)^-1 = I - V diag(lam / (1 + lam)) V^T, with (lam, V) eigenpairs of the
    Gauss-Newton Hessian in whitened coordinates (``V`` has shape (P*P, K), columns orthonormal). Directions
    beyond the K computed ones are prior-like. diag(L Sigma L^T) = sigma^2 - sum_k lam_k/(1+lam_k) (L v_k)^2,
    because every pixel has prior variance sigma^2 exactly (``GaussianField``); theta is affine in xi for the
    log_eps link, so this is the exact Laplace variance of log eps_r (no dense L is needed).
    """
    V = jnp.asarray(V)
    lam = jnp.asarray(lam)
    shape = field.xi_shape
    lv = jax.vmap(lambda v: field.theta(v.reshape(shape)) - field.mean)(V.T)  # (K, n, n): L v_k
    w = lam / (1.0 + lam)
    return np.asarray(field.sigma**2 - jnp.einsum("k,kij->ij", w, lv**2))


def lognormal_center_shift(field, lam, V, max_sd=None):
    """Shift of the Laplace center that accounts for the log-normal link: returns ``(delta_xi, info)``.

    The data fix the average of eps_r = exp(theta) over the DOI (the stiffest direction v_0 is the DOI average of
    theta), while the MAP puts the unresolved fluctuations of theta at zero. Fluctuations of variance Var raise
    E[exp(theta)] by Var/2, so the posterior mean of theta lies below the MAP by about Var/2 (mean over pixels
    of ``laplace_theta_variance``), which is tens of Laplace standard deviations along v_0. This returns
    delta_xi = t v_0 with t chosen so that mean(theta) changes by -Var/2, and ``info`` with ``var_mean``,
    ``theta_mean_shift`` (the shift actually applied to the DOI mean of theta), ``t``, ``n_sd`` (= |t| sqrt(1 + lam_0),
    the applied shift in Laplace standard deviations), ``n_sd_full`` (the same for the full -Var/2 shift) and
    ``capped``. The Laplace map stays a fixed affine map, so MCMC validity is untouched.

    ``max_sd``: cap on the applied shift in Laplace standard deviations. The prediction assumes the chain carries
    the unresolved fluctuations that produce the Var/2; a chain started at the shifted center has none yet, and
    pays Phi' ~ n_sd^2/2 before they develop. For a rough prior (a GMRF at pixel scale: Var = 0.1, 830 sd) a
    chain started that far away stays at Phi ~ 10^5, so the shift is scaled down to ``max_sd``.
    ``lam`` must be sorted in descending order (v_0 the stiffest direction).
    """
    var_mean = float(np.mean(laplace_theta_variance(field, lam, V)))
    v0 = jnp.asarray(V)[:, 0].reshape(field.xi_shape)
    mean_v0 = float(jnp.mean(field.theta(v0) - field.mean))  # mean over the DOI of L v_0
    t_full = -0.5 * var_mean / mean_v0
    sd0 = float(np.sqrt(1.0 + np.asarray(lam)[0]))
    n_full = abs(t_full) * sd0
    capped = max_sd is not None and n_full > max_sd
    t = t_full * (max_sd / n_full) if capped else t_full
    info = {"var_mean": var_mean, "theta_mean_shift": t * mean_v0, "t": t, "n_sd": abs(t) * sd0,
            "n_sd_full": n_full, "capped": bool(capped), "theta_mean_shift_full": -0.5 * var_mean}
    return t * v0, info


# ----------------------------------------------------------------------------------------------
# Sampler driver
# ----------------------------------------------------------------------------------------------

SAMPLERS = {"pcn": sm.pcn, "pcn_langevin": sm.pcn_langevin, "mala": sm.mala, "hmc": sm.hmc}
_DEFAULT_TARGET_ACCEPT = {"pcn": 0.25, "pcn_langevin": 0.5, "mala": 0.574, "hmc": 0.7}


def n_phi_evals(sampler, n_warmup, n_samples, n_leapfrog=10):
    """Number of Phi evaluations (one per chain) that a sampler run makes: the initial state plus one per
    proposal (pCN, pCN-Langevin, MALA) or ``n_leapfrog`` per HMC trajectory. Each evaluation solves the
    forward problem for all transmitters; every sampler except pCN also needs the adjoint solve for the
    gradient."""
    steps = n_warmup + n_samples
    return 1 + steps * (n_leapfrog if sampler == "hmc" else 1)


def sample(target, sampler, key, xi0s, *, n_warmup, n_samples, step0, thin=1, record_fn=None,
           n_leapfrog=10, jitter=0.2, target_accept=None, chain_method="auto"):
    """Run independent chains of ``sampler`` ("pcn", "pcn_langevin", "mala" or "hmc") on a ``Target``.

    Parameters
    ----------
    key : a single PRNG key, split into one key per chain.
    xi0s : (n_chains, *xi_shape) initial states; the warm-start aux of every chain begins at
        ``target.initial_aux()``.
    n_warmup, n_samples, step0, thin, target_accept : as in ``samplers`` (``step0`` is the initial step size,
        adapted during warmup toward ``target_accept``, default per sampler: 0.25, 0.5, 0.574, 0.7).
    record_fn : what to store, default ``target.record`` (stack of log eps_r and eps_r, shape (2, n, n)).
        Pass the same function object across calls to reuse the compiled sampler.
    n_leapfrog, jitter : HMC only. ``jitter`` = 0.2 by default because a fixed trajectory length resonates
        with near-Gaussian posteriors (see ``samplers.hmc``).
    chain_method : ``"sequential"`` runs the chains one after the other with one compiled sampler;
        ``"vmap"`` batches them with ``jax.vmap``; ``"auto"`` (default) picks ``"vmap"`` for the Born model
        and ``"sequential"`` otherwise. Batching pays only for cheap targets such as the Born model (about
        twice as fast for 4 chains): vmapping the full forward model batches the Krylov while-loops, which
        then run until the slowest chain's solve is done, and the compiled program takes minutes to build
        and run (measured: no 4-step run finished within a minute for a 12 x 12 grid, where one chain
        needs 20 ms per step).

    Returns ``(result, info)``. ``result`` is a ``samplers.ChainResult`` whose fields have a leading chain
    axis. ``info`` holds ``wall_s`` (run time of all chains, compilation excluded: the program is compiled
    ahead of time and timed separately as ``compile_s``), ``n_phi_evals`` (per chain, see ``n_phi_evals``),
    ``n_forward_solves`` and ``n_adjoint_solves`` (summed over chains; one solve covers all transmitters).
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {sorted(SAMPLERS)}, got {sampler!r}")
    if chain_method not in ("auto", "sequential", "vmap"):
        raise ValueError("chain_method must be 'auto', 'sequential' or 'vmap'")
    if chain_method == "auto":
        chain_method = "vmap" if target.use_born else "sequential"
    fn = SAMPLERS[sampler]
    record_fn = target.record if record_fn is None else record_fn
    phi_fn = target.phi_fn_no_grad if sampler == "pcn" else target.phi_fn
    kw = dict(phi_fn=phi_fn, n_warmup=int(n_warmup), n_samples=int(n_samples), step0=step0, thin=int(thin),
              record_fn=record_fn,
              target_accept=_DEFAULT_TARGET_ACCEPT[sampler] if target_accept is None else target_accept)
    if sampler == "hmc":
        kw.update(n_leapfrog=int(n_leapfrog), jitter=jitter)
    xi0s = jnp.asarray(xi0s, dtype=jnp.float64)
    n_chains = xi0s.shape[0]
    keys = jax.random.split(key, n_chains)
    aux0 = target.initial_aux()

    def one_chain(k, x):
        return fn(k, xi0=x, aux0=aux0, **kw)

    def batched(ks, xs, axs):
        return jax.vmap(lambda k, x, a: fn(k, xi0=x, aux0=a, **kw))(ks, xs, axs)

    t0 = time.perf_counter()
    if chain_method == "vmap":
        aux0s = jnp.broadcast_to(aux0, (n_chains,) + aux0.shape)
        compiled = jax.jit(batched).lower(keys, xi0s, aux0s).compile()
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = jax.block_until_ready(compiled(keys, xi0s, aux0s))
    else:
        compiled = jax.jit(one_chain).lower(keys[0], xi0s[0]).compile()
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs = [jax.block_until_ready(compiled(keys[c], xi0s[c])) for c in range(n_chains)]
        res = jax.tree.map(lambda *leaves: jnp.stack(leaves), *outs)
    wall_s = time.perf_counter() - t0

    evals = n_phi_evals(sampler, n_warmup, n_samples, n_leapfrog)
    info = {"wall_s": wall_s, "compile_s": compile_s, "n_phi_evals": evals,
            "n_forward_solves": n_chains * evals,
            "n_adjoint_solves": 0 if sampler == "pcn" else n_chains * evals}
    return res, info


# ----------------------------------------------------------------------------------------------
# Comparison with a known Gaussian
# ----------------------------------------------------------------------------------------------


def compare_to_gaussian(samples, mean, var, c=5.0):
    """Standardized errors of chain estimates against the exact marginal mean and variance.

    Parameters
    ----------
    samples : (n_chains, N, ...) draws (trailing shape arbitrary, e.g. a flattened or (n, n) pixel map).
    mean, var : exact marginal mean and variance with the trailing shape of ``samples``.
    c : Sokal window constant of ``diagnostics.tau_int``.

    For a correct sampler started in or near the posterior both statistics below are asymptotically N(0, 1)
    per component (correlated across components, since neighboring pixels move together):

    * ``z_mean = (mean_hat - mean) / MCSE``, with mean_hat the average over all draws and MCSE from the
      per-chain Sokal effective sample sizes, combined over chains as sqrt(sum MCSE_c^2) / n_chains.
    * ``z_var``: the same with q = (x - mean)^2 in place of x, whose expectation is exactly ``var``, so
      z_var = (mean_hat(q) - var) / MCSE(q). ``var_ratio = mean_hat(q) / var``.

    Returns a dict of flattened arrays ``z_mean``, ``z_var``, ``var_ratio``, ``ess`` (summed over chains),
    ``rhat`` (split R-hat of the draws; NaN with fewer than 2 chains or 8 draws) and the scalars
    ``z_mean_rms``, ``z_mean_max`` (max |z|), ``z_var_rms``, ``z_var_max``, ``var_ratio_mean``,
    ``var_ratio_min``, ``var_ratio_max``, ``ess_min``, ``rhat_max``.
    """
    x = np.asarray(samples, dtype=np.float64)
    n_chains = x.shape[0]
    x = x.reshape(n_chains, x.shape[1], -1)
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    v = np.asarray(var, dtype=np.float64).reshape(-1)

    def z_and_mean(series, target):
        mcse = np.stack([dg.mcse_mean(s, c=c) for s in series])  # (chains, D)
        est = series.mean(axis=1).mean(axis=0)
        return (est - target) / (np.sqrt((mcse**2).sum(axis=0)) / n_chains), est

    z_mean, _ = z_and_mean(x, m)
    q = (x - m) ** 2
    z_var, q_hat = z_and_mean(q, v)
    ess = np.stack([dg.ess(s, c=c) for s in x]).sum(axis=0)
    rhat = dg.split_rhat(x) if n_chains >= 2 and x.shape[1] >= 8 else np.full(x.shape[2], np.nan)
    ratio = q_hat / v
    return {
        "z_mean": z_mean, "z_var": z_var, "var_ratio": ratio, "ess": ess, "rhat": rhat,
        "z_mean_rms": float(np.sqrt(np.mean(z_mean**2))), "z_mean_max": float(np.abs(z_mean).max()),
        "z_var_rms": float(np.sqrt(np.mean(z_var**2))), "z_var_max": float(np.abs(z_var).max()),
        "var_ratio_mean": float(ratio.mean()), "var_ratio_min": float(ratio.min()),
        "var_ratio_max": float(ratio.max()), "ess_min": float(ess.min()), "rhat_max": float(np.nanmax(rhat))
        if np.any(np.isfinite(rhat)) else float("nan"),
    }
