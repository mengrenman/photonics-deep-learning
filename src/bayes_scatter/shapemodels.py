"""Parametric shape models F2, F2u and Fe with their priors in unconstrained coordinates (docs/DESIGN.md 14.2).

A model maps an unconstrained vector u in R^d to physical parameters p (bounded, positive, ordered) and to a scene of ellipses with relative permittivities;
the contrast is rendered exactly by ``ellipses.EllipseRenderer`` (coverage and boundary shape derivatives in closed form, differentiable by JAX), so the
gradient of any functional of chi with respect to u is exact. Everything is float64 and JAX-traceable.

Priors and Jacobians. Each prior is specified in p-space and carried to u-space: if p = T(u) and p has density pi_p, the density of u is

    pi_u(u) = pi_p(T(u)) |det dT/du|        (``log_prior_p`` + ``log_jac_u`` = ``log_prior_u``; the tests check that ``log_prior_u`` is the closed form
                                              below, that it integrates to one, and that prior-only HMC in u reproduces the p-space marginals).

* a parameter uniform on [lo, hi] is x = lo + (hi - lo) sigmoid(u): u is logistic, log pi_u = log sigmoid(u) + log sigmoid(-u) whatever [lo, hi];
* epsilon: eps_r - 1 = exp(0.5 u) with u ~ N(0, 1), i.e. eps_r - 1 is log-normal with median 1 and log-scale sigma 0.5 (the median eps_r is 2);
* two ordered semi-axes b <= a, uniform on the triangle lo <= b <= a <= hi (density 2 / (hi - lo)^2): a = lo + (hi - lo) sigmoid(u_a), b = lo + (a - lo) sigmoid(u_b)
  (triangular Jacobian (hi - lo) s_a (1 - s_a) (a - lo) s_b (1 - s_b)); the ordering is the constraint that breaks the redundancy (a, b, psi) ~ (b, a, psi + pi / 2) of
  an ellipse, and psi in [-pi/2, pi/2) uniform removes psi ~ psi + pi.

Models (priors as in DESIGN 14.2; every choice the design leaves open is marked "choice"):

* ``F2`` (5 parameters): p = (g, r, eps_r, cx, cy), two equal circles of radius r and permittivity eps_r at (cx -+ (r + g / 2), cy) with edge-to-edge gap g.
  g ~ U[0, 0.4] (choice: uniform in g, i.e. g = 0.4 sigmoid(u_g) with a logistic u_g), r ~ U[0.1, 0.3], eps_r - 1 log-normal (median 1, sigma 0.5), (cx, cy) ~ U[-0.2, 0.2]^2 (the
  pair center).
* ``F2u`` (7 parameters): p = (g, r1, r2, eps1, eps2, cx, cy), the same pair with separate radii and permittivities; r1, r2 ~ U[0.1, 0.3] and eps1 - 1, eps2 - 1 log-normal,
  independent. Disk 1 is the LEFT one: its center is cx - (rbar + g / 2), rbar = (r1 + r2) / 2, so the gap between the facing edges is g for any radii. **Ordering:**
  the prior is exchangeable under (r1, eps1) <-> (r2, eps2), but the exchange at fixed positions is a different object, and the label symmetry of an unordered pair is
  broken by the geometry (disk 1 is always the one with the smaller x; g >= 0 keeps the order). The one remaining symmetry is the mirror x -> -x of the antenna rings
  (``symmetry_images``): it maps (r1, eps1, r2, eps2, cx) to (r2, eps2, r1, eps1, -cx) and is a symmetry of the model manifold, of the prior and (for mirror-symmetric
  data) of the likelihood. F2 is F2u with r1 = r2 and eps1 = eps2 (nested).
* ``Fe`` (6 parameters): p = (cx, cy, a, b, psi, eps_r), one ellipse with semi-axes a >= b in [0.05, 0.55] (ordered pair, uniform on the triangle), rotation psi in [-pi/2, pi/2)
  uniform, center U[-0.2, 0.2]^2, eps_r - 1 log-normal. It has no gap: the "one object" alternative.
  **Limitation of the orientation coordinate.** psi = pi (sigmoid(u_psi) - 1/2) removes the redundancy psi ~ psi + pi of the DENSITY, but the two ends psi = -pi/2 and psi = +pi/2
  are the same ellipse and sit at u_psi -> -inf and +inf, so a posterior for an ellipse elongated along y (psi near +-pi/2) would split across the two ends of the box (and
  the logistic prior makes u_psi = +-10 the wall of the truncation box). A periodic coordinate, for example (cos 2 psi, sin 2 psi) times the ellipticity, would not. It does not
  matter for the milestone-7 results: every Fe fit has an ellipse along x, psi within 0.01 of 0 (``results/parametric/tables.md`` section 7).

A shape that leaves the DOI is truncated by it (the renderer's convention), so the prior support needs no constraint; nothing here is clipped.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

G_MAX = 0.4  # g ~ U[0, G_MAX]
R_LO, R_HI = 0.1, 0.3  # radius ~ U[R_LO, R_HI]
C_MAX = 0.2  # pair center / ellipse center ~ U[-C_MAX, C_MAX]
EPS_SIGMA = 0.5  # eps_r - 1 = exp(EPS_SIGMA u), u ~ N(0, 1)
AX_LO, AX_HI = 0.05, 0.55  # ordered semi-axes of Fe
LOG_2PI = math.log(2.0 * math.pi)

_ls = jax.nn.log_sigmoid


def _from_unit(u, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(u)


def _to_unit(x, lo, hi):
    z = (jnp.asarray(x) - lo) / (hi - lo)
    return jnp.log(z) - jnp.log1p(-z)


def _logistic_logpdf(u):
    """log of the density of u = logit(U), U uniform on (0, 1): log sigmoid(u) + log sigmoid(-u)."""
    return _ls(u) + _ls(-u)


def _eps_from_u(u):
    return 1.0 + jnp.exp(EPS_SIGMA * u)


def _eps_to_u(eps):
    return jnp.log(jnp.asarray(eps) - 1.0) / EPS_SIGMA


def _normal_logpdf(u):
    return -0.5 * u * u - 0.5 * LOG_2PI


def _lognormal_logpdf_eps(eps):
    """log density of eps_r - 1 log-normal with median 1 and log-scale EPS_SIGMA, evaluated at eps_r."""
    c = jnp.asarray(eps) - 1.0
    lc = jnp.log(c)
    return -lc - math.log(EPS_SIGMA) - 0.5 * LOG_2PI - 0.5 * (lc / EPS_SIGMA) ** 2


def _uniform_logpdf(x, lo, hi):
    return jnp.where((x >= lo) & (x <= hi), -math.log(hi - lo), -jnp.inf)


class ShapeModel:
    """Base class: unconstrained parametrization, prior and scene of one shape family (module docstring).

    Attributes: ``name``, ``param_names`` (physical p), ``dim``. Methods (JAX, float64; ``u`` and ``p`` are (dim,) arrays):

    * ``p_from_u(u)`` / ``u_from_p(p)`` : the bijection between R^dim and the prior support;
    * ``log_prior_u(u)`` : log density of u under the induced prior, **including the Jacobian** (closed form); ``log_prior_p(p)`` and ``log_jac_u(u)`` are its parts;
    * ``scene(p)`` : ``(ellipses (E, 5), eps (E,))``; ``chi(renderer, u)`` : the contrast (n, n), differentiable in u through the exact boundary derivatives;
    * ``gap(p)`` : the edge-to-edge gap (None for a model without one);
    * ``sample_prior_p(rng, size)`` : NumPy draws from the prior in p-space, (size, dim) (direct, no MCMC);
    * ``symmetry_images(p)`` : list of ``(label, p_image)``, the points the model's discrete symmetries map ``p`` to (starts for the multimodality search);
    * ``describe()`` : JSON dict of the prior.
    """

    name = "model"
    param_names = ()
    description = ""

    @property
    def dim(self):
        return len(self.param_names)

    def p_from_u(self, u):
        raise NotImplementedError

    def u_from_p(self, p):
        raise NotImplementedError

    def log_prior_u(self, u):
        raise NotImplementedError

    def log_prior_p(self, p):
        raise NotImplementedError

    def log_jac_u(self, u):
        raise NotImplementedError

    def scene(self, p):
        raise NotImplementedError

    def sample_prior_p(self, rng, size):
        raise NotImplementedError

    def symmetry_images(self, p):
        return []

    def gap(self, p):
        return None

    def chi(self, renderer, u):
        """Contrast chi (n, n) of the model at the unconstrained point ``u``."""
        ell, eps = self.scene(self.p_from_u(u))
        return renderer.chi(ell, eps)

    def physical(self, p):
        """Dict name -> value for a physical parameter vector."""
        return {k: float(v) for k, v in zip(self.param_names, np.asarray(p))}

    def describe(self):
        return {"name": self.name, "param_names": list(self.param_names), "dim": self.dim, "description": self.description,
                "constants": {"G_MAX": G_MAX, "R": [R_LO, R_HI], "C_MAX": C_MAX, "EPS_SIGMA": EPS_SIGMA, "AX": [AX_LO, AX_HI]}}

    def start_points_u(self, rng, count):
        """``count`` starts u drawn from the prior (for the MAP search), (count, dim)."""
        return np.asarray(jax.vmap(self.u_from_p)(jnp.asarray(self.sample_prior_p(rng, count))))


class F2(ShapeModel):
    """Two equal circles: p = (g, r, eps_r, cx, cy) (module docstring)."""

    name = "F2"
    param_names = ("g", "r", "eps_r", "cx", "cy")
    description = "two equal circles (g, r, eps_r, cx, cy): the milestone-6 family"

    def p_from_u(self, u):
        return jnp.stack([_from_unit(u[0], 0.0, G_MAX), _from_unit(u[1], R_LO, R_HI), _eps_from_u(u[2]),
                          _from_unit(u[3], -C_MAX, C_MAX), _from_unit(u[4], -C_MAX, C_MAX)])

    def u_from_p(self, p):
        p = jnp.asarray(p)
        return jnp.stack([_to_unit(p[0], 0.0, G_MAX), _to_unit(p[1], R_LO, R_HI), _eps_to_u(p[2]),
                          _to_unit(p[3], -C_MAX, C_MAX), _to_unit(p[4], -C_MAX, C_MAX)])

    def log_prior_u(self, u):
        return _logistic_logpdf(u[0]) + _logistic_logpdf(u[1]) + _normal_logpdf(u[2]) + _logistic_logpdf(u[3]) + _logistic_logpdf(u[4])

    def log_prior_p(self, p):
        return (_uniform_logpdf(p[0], 0.0, G_MAX) + _uniform_logpdf(p[1], R_LO, R_HI) + _lognormal_logpdf_eps(p[2])
                + _uniform_logpdf(p[3], -C_MAX, C_MAX) + _uniform_logpdf(p[4], -C_MAX, C_MAX))

    def log_jac_u(self, u):
        s = lambda lo, hi, x: math.log(hi - lo) + _logistic_logpdf(x)
        return (s(0.0, G_MAX, u[0]) + s(R_LO, R_HI, u[1]) + math.log(EPS_SIGMA) + EPS_SIGMA * u[2] + s(-C_MAX, C_MAX, u[3]) + s(-C_MAX, C_MAX, u[4]))

    def scene(self, p):
        g, r, eps, cx, cy = p[0], p[1], p[2], p[3], p[4]
        z, one = jnp.zeros(()), jnp.ones(())
        ell = jnp.stack([jnp.stack([cx - (r + 0.5 * g), cy, r, r, z]), jnp.stack([cx + (r + 0.5 * g), cy, r, r, z])])
        return ell, eps * jnp.stack([one, one])

    def gap(self, p):
        return float(np.asarray(p)[0])

    def sample_prior_p(self, rng, size):
        eps = 1.0 + np.exp(EPS_SIGMA * rng.standard_normal(size))
        return np.stack([rng.uniform(0.0, G_MAX, size), rng.uniform(R_LO, R_HI, size), eps,
                         rng.uniform(-C_MAX, C_MAX, size), rng.uniform(-C_MAX, C_MAX, size)], axis=1)

    def symmetry_images(self, p):
        p = np.array(p, dtype=np.float64)
        q = p.copy()
        q[3] = -q[3]
        return [("mirror x -> -x", q)]

    def to_f2u(self, p):
        """The F2u point equal to ``p`` (nested model): (g, r, r, eps, eps, cx, cy)."""
        g, r, eps, cx, cy = np.asarray(p, dtype=np.float64)
        return np.array([g, r, r, eps, eps, cx, cy])

    def to_fe(self, p):
        """A start for Fe from ``p``: the ellipse around both disks, semi-axes (2 r + g / 2, r), psi = 0, the same eps_r (clipped into the Fe prior box)."""
        g, r, eps, cx, cy = np.asarray(p, dtype=np.float64)
        a = float(np.clip(2.0 * r + 0.5 * g, AX_LO + 1e-3, AX_HI - 1e-3))
        return np.array([cx, cy, a, float(np.clip(r, AX_LO + 1e-3, a - 1e-3)), 0.0, eps])


class F2u(ShapeModel):
    """Two circles with separate radii and permittivities: p = (g, r1, r2, eps1, eps2, cx, cy) (module docstring; disk 1 is the left one)."""

    name = "F2u"
    param_names = ("g", "r1", "r2", "eps1", "eps2", "cx", "cy")
    description = "two circles with separate radii and permittivities (left disk first)"

    def p_from_u(self, u):
        return jnp.stack([_from_unit(u[0], 0.0, G_MAX), _from_unit(u[1], R_LO, R_HI), _from_unit(u[2], R_LO, R_HI), _eps_from_u(u[3]), _eps_from_u(u[4]),
                          _from_unit(u[5], -C_MAX, C_MAX), _from_unit(u[6], -C_MAX, C_MAX)])

    def u_from_p(self, p):
        p = jnp.asarray(p)
        return jnp.stack([_to_unit(p[0], 0.0, G_MAX), _to_unit(p[1], R_LO, R_HI), _to_unit(p[2], R_LO, R_HI), _eps_to_u(p[3]), _eps_to_u(p[4]),
                          _to_unit(p[5], -C_MAX, C_MAX), _to_unit(p[6], -C_MAX, C_MAX)])

    def log_prior_u(self, u):
        return (_logistic_logpdf(u[0]) + _logistic_logpdf(u[1]) + _logistic_logpdf(u[2]) + _normal_logpdf(u[3]) + _normal_logpdf(u[4])
                + _logistic_logpdf(u[5]) + _logistic_logpdf(u[6]))

    def log_prior_p(self, p):
        return (_uniform_logpdf(p[0], 0.0, G_MAX) + _uniform_logpdf(p[1], R_LO, R_HI) + _uniform_logpdf(p[2], R_LO, R_HI)
                + _lognormal_logpdf_eps(p[3]) + _lognormal_logpdf_eps(p[4]) + _uniform_logpdf(p[5], -C_MAX, C_MAX) + _uniform_logpdf(p[6], -C_MAX, C_MAX))

    def log_jac_u(self, u):
        s = lambda lo, hi, x: math.log(hi - lo) + _logistic_logpdf(x)
        return (s(0.0, G_MAX, u[0]) + s(R_LO, R_HI, u[1]) + s(R_LO, R_HI, u[2]) + 2.0 * math.log(EPS_SIGMA) + EPS_SIGMA * (u[3] + u[4])
                + s(-C_MAX, C_MAX, u[5]) + s(-C_MAX, C_MAX, u[6]))

    def scene(self, p):
        g, r1, r2, e1, e2, cx, cy = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        rbar = 0.5 * (r1 + r2)
        z = jnp.zeros(())
        ell = jnp.stack([jnp.stack([cx - (rbar + 0.5 * g), cy, r1, r1, z]), jnp.stack([cx + (rbar + 0.5 * g), cy, r2, r2, z])])
        return ell, jnp.stack([e1, e2])

    def gap(self, p):
        return float(np.asarray(p)[0])

    def sample_prior_p(self, rng, size):
        return np.stack([rng.uniform(0.0, G_MAX, size), rng.uniform(R_LO, R_HI, size), rng.uniform(R_LO, R_HI, size),
                         1.0 + np.exp(EPS_SIGMA * rng.standard_normal(size)), 1.0 + np.exp(EPS_SIGMA * rng.standard_normal(size)),
                         rng.uniform(-C_MAX, C_MAX, size), rng.uniform(-C_MAX, C_MAX, size)], axis=1)

    def symmetry_images(self, p):
        """The in-place exchange of the two disks' properties (r1, eps1) <-> (r2, eps2) at the same positions, and the mirror x -> -x of the whole scene (the
        exchange together with cx -> -cx)."""
        g, r1, r2, e1, e2, cx, cy = np.asarray(p, dtype=np.float64)
        swap = np.array([g, r2, r1, e2, e1, cx, cy])
        mirror = np.array([g, r2, r1, e2, e1, -cx, cy])
        return [("exchange (r1, eps1) <-> (r2, eps2) at fixed positions", swap), ("mirror x -> -x", mirror)]


class Fe(ShapeModel):
    """One ellipse: p = (cx, cy, a, b, psi, eps_r) with a >= b (module docstring)."""

    name = "Fe"
    param_names = ("cx", "cy", "a", "b", "psi", "eps_r")
    description = "one ellipse (center, ordered semi-axes a >= b, rotation, eps_r)"

    def p_from_u(self, u):
        a = _from_unit(u[2], AX_LO, AX_HI)
        b = _from_unit(u[3], AX_LO, a)
        return jnp.stack([_from_unit(u[0], -C_MAX, C_MAX), _from_unit(u[1], -C_MAX, C_MAX), a, b, math.pi * (jax.nn.sigmoid(u[4]) - 0.5), _eps_from_u(u[5])])

    def u_from_p(self, p):
        p = jnp.asarray(p)
        return jnp.stack([_to_unit(p[0], -C_MAX, C_MAX), _to_unit(p[1], -C_MAX, C_MAX), _to_unit(p[2], AX_LO, AX_HI), _to_unit(p[3], AX_LO, p[2]),
                          _to_unit(p[4] / math.pi + 0.5, 0.0, 1.0), _eps_to_u(p[5])])

    def log_prior_u(self, u):
        # (a, b) uniform on the triangle (density 2 / w^2) times the Jacobian w s_a (1 - s_a) (a - lo) s_b (1 - s_b), a - lo = w s_a: the widths cancel
        return (_logistic_logpdf(u[0]) + _logistic_logpdf(u[1]) + math.log(2.0) + _logistic_logpdf(u[2]) + _ls(u[2]) + _logistic_logpdf(u[3])
                + _logistic_logpdf(u[4]) + _normal_logpdf(u[5]))

    def log_prior_p(self, p):
        a, b = p[2], p[3]
        w = AX_HI - AX_LO
        tri = jnp.where((b >= AX_LO) & (b <= a) & (a <= AX_HI), math.log(2.0) - 2.0 * math.log(w), -jnp.inf)
        return (_uniform_logpdf(p[0], -C_MAX, C_MAX) + _uniform_logpdf(p[1], -C_MAX, C_MAX) + tri
                + _uniform_logpdf(p[4], -0.5 * math.pi, 0.5 * math.pi) + _lognormal_logpdf_eps(p[5]))

    def log_jac_u(self, u):
        w = AX_HI - AX_LO
        s = lambda lo, hi, x: math.log(hi - lo) + _logistic_logpdf(x)
        return (s(-C_MAX, C_MAX, u[0]) + s(-C_MAX, C_MAX, u[1]) + s(AX_LO, AX_HI, u[2]) + math.log(w) + _ls(u[2]) + _logistic_logpdf(u[3])
                + math.log(math.pi) + _logistic_logpdf(u[4]) + math.log(EPS_SIGMA) + EPS_SIGMA * u[5])

    def scene(self, p):
        return jnp.stack([jnp.stack([p[0], p[1], p[2], p[3], p[4]])]), jnp.stack([p[5]])

    def sample_prior_p(self, rng, size):
        ab = rng.uniform(AX_LO, AX_HI, (size, 2))
        a, b = ab.max(axis=1), ab.min(axis=1)
        return np.stack([rng.uniform(-C_MAX, C_MAX, size), rng.uniform(-C_MAX, C_MAX, size), a, b, rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size),
                         1.0 + np.exp(EPS_SIGMA * rng.standard_normal(size))], axis=1)

    def symmetry_images(self, p):
        cx, cy, a, b, psi, eps = np.asarray(p, dtype=np.float64)
        return [("mirror x -> -x", np.array([-cx, cy, a, b, -psi, eps]))]


MODELS = {"F2": F2, "F2u": F2u, "Fe": Fe}


def make_model(name):
    """The model instance named ``F2``, ``F2u`` or ``Fe``."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {tuple(MODELS)}")
    return MODELS[name]()
