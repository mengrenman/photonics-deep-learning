"""Maximum likelihood for the two-disk object by Levenberg-Marquardt: the threshold-aware check of the Cramer-Rao bound (docs/DESIGN.md 13.2).

The Cramer-Rao bound of ``fisher`` is a LOCAL statement: it bounds the variance of an unbiased estimator and is attained (asymptotically) by the maximum
likelihood estimator only when the data are informative enough that the estimate stays in the neighborhood of the truth where the model is linear in the
parameters. At the SNR where sigma_CRB(g) = g the nuisance parameters (r, eps_r, c_x, c_y) can be essentially undetermined (the review of milestone 6 found
sd(eps_r) / (eps_r - 1) = 0.16 to 0.9), so the bound may not describe any estimator there. This module fits the five parameters p = (g, r, eps_r, c_x, c_y) of
``shapes.TwoDisks`` to noisy data of the same model, so that the empirical error of the estimator at the required SNR can be set beside the bound.

Model. Data d = F(chi(p)) + e, e ~ CN(0, s^2) (``likelihood``); the negative log-likelihood is Phi(p) = sum |F(chi(p)) - d|^2 / s^2 up to a constant
(``likelihood.misfit``). With D = [dd/dp_k] (``fisher.data_derivatives``, exact shape derivatives and one jvp per parameter) and the residual
res = d - F(chi(p)), the gradient of Phi is -(2 / s^2) Re(D^H res) and the Gauss-Newton Hessian is (2 / s^2) Re(D^H D), the Fisher matrix at p.

Levenberg-Marquardt. The step solves (Re(D^H D) + mu diag(Re(D^H D))) delta = Re(D^H res); the new point is projected onto the box ``Bounds`` (g >= 0, r, eps_r
and the center within limits) and onto the set where both disks lie inside the domain of interest (``project``). A step is accepted if Phi decreases (then mu is
divided by 3) and rejected otherwise (mu is multiplied by 4). It stops when an accepted step is shorter than ``tol`` times the scale of the parameters
(``scale``, e.g. the Cramer-Rao standard deviations) and Phi has decreased by less than ``phi_tol``, or when no step reduces Phi within ``max_tries`` trials
(a stationary point or a point on the boundary of the box), or after ``max_iter`` accepted steps.

Global search. Phi is not convex in p. ``fit_multistart`` runs ``fit`` from several starts and returns the lowest Phi; the starts are the truth (an oracle
that bounds the difficulty from below) and draws from a box that does not depend on the truth (``random_start``).
"""

import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import fisher as fi
from . import shapes as sh
from .likelihood import misfit

PARAMS = ("g", "r", "eps_r", "cx", "cy")


@dataclass(frozen=True)
class Bounds:
    """Box of the parameters (wavelengths; eps_r is the relative permittivity) and the fraction of the DOI half side the disks may fill.

    ``lo`` / ``hi``: dicts name -> limit for ``PARAMS``. ``margin``: both disks must lie within ``side / 2 - margin`` of the DOI center (``project``).
    """

    lo: dict
    hi: dict
    margin: float = 0.01

    @staticmethod
    def default():
        return Bounds(lo={"g": 0.0, "r": 0.08, "eps_r": 1.02, "cx": -0.15, "cy": -0.15}, hi={"g": 0.6, "r": 0.3, "eps_r": 6.0, "cx": 0.15, "cy": 0.15})


def as_object(p):
    """``shapes.TwoDisks`` of the parameter vector ``p`` = (g, r, eps_r, cx, cy) (equal radii)."""
    return sh.TwoDisks(g=float(p[0]), r=float(p[1]), eps_r=float(p[2]), cx=float(p[3]), cy=float(p[4]))


def project(p, bounds, side):
    """Project ``p`` (5,) onto the box and onto the disks-inside-the-DOI set: clip to ``bounds``, then shrink g, then |c_x|, then r until
    2 r + g / 2 + |c_x| <= side / 2 - margin (the pair along x) and r + |c_y| <= side / 2 - margin. Returns a new float array (5,)."""
    q = np.array([min(max(float(p[i]), bounds.lo[k]), bounds.hi[k]) for i, k in enumerate(PARAMS)])
    half = 0.5 * side - bounds.margin
    over = 2.0 * q[1] + 0.5 * q[0] + abs(q[3]) - half
    if over > 0.0:
        cut = min(q[0] * 0.5, over)  # g first (each unit of g costs a half)
        q[0] -= 2.0 * cut
        over -= cut
    if over > 0.0:
        cut = min(abs(q[3]), over)
        q[3] -= np.sign(q[3]) * cut
        over -= cut
    if over > 0.0:
        q[1] -= 0.5 * over
    over_y = q[1] + abs(q[4]) - half
    if over_y > 0.0:
        q[4] -= np.sign(q[4]) * min(abs(q[4]), over_y)
    return q


def random_start(rng, bounds=None):
    """A start that does not depend on the truth: g ~ U(0, 0.3), r ~ U(0.12, 0.28), eps_r ~ U(1.3, 3.5), c_x, c_y ~ U(-0.05, 0.05)."""
    return np.array([rng.uniform(0.0, 0.3), rng.uniform(0.12, 0.28), rng.uniform(1.3, 3.5), rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)])


def _evaluate(forward, d_obs, s, p):
    obj = as_object(p)
    chi = obj.chi(forward.grid)
    d, _ = forward(jnp.asarray(chi))
    return obj, chi, d, float(misfit(d, d_obs, s))


def fit(forward, d_obs, s, p0, bounds=None, scale=None, max_iter=40, tol=0.02, phi_tol=1e-2, mu0=1e-2, max_tries=12):
    """Levenberg-Marquardt maximum likelihood fit of p = (g, r, eps_r, cx, cy) from ``p0`` (module docstring).

    ``d_obs`` (Nt, Nr) complex data, ``s`` the noise level (``likelihood.noise_std_from_snr``), ``scale`` (5,) the length scale of each parameter for the stopping
    test (default 1e-3). Returns a dict: ``p`` (5,), ``phi``, ``n_iter`` (accepted steps), ``n_eval`` (forward solves incl. rejected trials), ``converged``
    (a step-size stop or a stationary point; False if ``max_iter`` ran out), ``stop`` (the reason), ``at_boundary`` (names of the parameters on a bound), ``time_s``.
    """
    t0 = time.perf_counter()
    bounds = Bounds.default() if bounds is None else bounds
    side = forward.grid.side
    scale = np.full(5, 1e-3) if scale is None else np.asarray(scale, dtype=np.float64)
    d_obs = jnp.asarray(d_obs)
    p = project(p0, bounds, side)
    obj, chi, d, phi = _evaluate(forward, d_obs, s, p)
    n_eval, mu, n_iter, stop = 1, mu0, 0, "max_iter"
    for _ in range(max_iter):
        dchi = sh.shape_derivatives(forward.grid, obj, PARAMS)
        D = np.asarray(fi.data_derivatives(forward, chi, dchi))
        res = (np.asarray(d_obs) - np.asarray(d)).reshape(-1)
        A = np.real(D.conj().T @ D)
        b = np.real(D.conj().T @ res)
        accepted = False
        for _try in range(max_tries):
            try:
                delta = np.linalg.solve(A + mu * np.diag(np.diag(A)), b)
            except np.linalg.LinAlgError:
                mu *= 4.0
                continue
            q = project(p + delta, bounds, side)
            obj_q, chi_q, d_q, phi_q = _evaluate(forward, d_obs, s, q)
            n_eval += 1
            if phi_q < phi:
                step = float(np.max(np.abs(q - p) / scale))
                dphi = phi - phi_q
                p, obj, chi, d, phi, accepted = q, obj_q, chi_q, d_q, phi_q, True
                mu = max(mu / 3.0, 1e-9)
                n_iter += 1
                break
            mu *= 4.0
        if not accepted:
            stop = "stationary"
            break
        if step < tol and dphi < phi_tol:
            stop = "step"
            break
    at_bound = [k for i, k in enumerate(PARAMS) if abs(p[i] - bounds.lo[k]) < 1e-12 or abs(p[i] - bounds.hi[k]) < 1e-12]
    return {"p": p, "phi": phi, "n_iter": n_iter, "n_eval": n_eval, "converged": stop in ("step", "stationary"), "stop": stop, "at_boundary": at_bound,
            "time_s": time.perf_counter() - t0}


def fit_multistart(forward, d_obs, s, starts, **kw):
    """``fit`` from every start of ``starts`` (list of (label, p0)); returns ``(best, runs)``: the run with the lowest Phi (a dict of ``fit`` plus ``label``) and all runs."""
    runs = []
    for label, p0 in starts:
        r = fit(forward, d_obs, s, p0, **kw)
        r["label"] = label
        runs.append(r)
    return min(runs, key=lambda r: r["phi"]), runs


def estimator_summary(estimates, truth, crb_sd):
    """Empirical error statistics of an estimator over realizations against the Cramer-Rao standard deviations.

    ``estimates`` (M, K), ``truth`` (K,), ``crb_sd`` (K,). Returns a dict of lists per parameter: ``bias``, ``sd`` (ddof 1), ``rmse``, ``rmse_over_crb`` and
    ``coverage_1p96`` (the fraction of realizations within 1.96 CRB standard deviations of the truth, 0.95 for a Gaussian estimator that attains the bound).
    """
    est = np.asarray(estimates, dtype=np.float64)
    truth, crb_sd = np.asarray(truth, dtype=np.float64), np.asarray(crb_sd, dtype=np.float64)
    err = est - truth
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return {"n": int(est.shape[0]), "bias": err.mean(axis=0).tolist(), "sd": est.std(axis=0, ddof=1).tolist(), "rmse": rmse.tolist(),
            "rmse_over_crb": (rmse / crb_sd).tolist(), "coverage_1p96": np.mean(np.abs(err) <= 1.96 * crb_sd, axis=0).tolist()}
