"""The geometry of detectability: first-order model-error bias of a shape fit and the D2 selection rule (docs/DESIGN.md section 15).

Whitened real data. Complex data d[t, r] (Nt, Nr) with CN(0, s^2) noise become the real vector y_w = sqrt(2)/s [Re d.ravel(); Im d.ravel()] (2N,), row order
t Nr + r for the real block and N + t Nr + r for the imaginary block; every entry of the noise has unit variance and the misfit of ``likelihood.misfit`` is
Phi = sum |r|^2 / s^2 = (1/2) ||r_w||^2 (the factor 1/2 of DESIGN 15.1). The Fisher matrix of a model with whitened Jacobian D (2N, k) is D^T D = Hessian of Phi
(it equals ``ShapePosterior.fisher`` up to the prior).

First-order theory (DESIGN 15.1). Model error delta_w (2N,) = whitened (true noise-free data) - (model data at the base point p0). Least squares of the
model to the data, linearized at p0:

    dp = (D^T D)^-1 D^T delta,    dPhi = (1/2) ||P_perp delta||^2,    P_perp = I - D (D^T D)^-1 D^T,
    z_k = dp_k / sigma_k = w_hat^T delta,   sigma_k^2 = [(D^T D)^-1]_kk,

where w_hat is the unit vector of range(D) orthogonal to the columns of all parameters but k (the "pure k direction"): w_hat = P_others d_k / ||.|| with P_others
the projector off the other columns and sigma_k = 1 / ||P_others d_k||. With alpha the angle between delta and range(D) and beta the angle between its tangential part
and w_hat, c = dPhi / z^2 = (1/2) tan^2(alpha) / cos^2(beta) (``c_geom``). The sign convention of z here is (fitted - base) / sigma; the milestone-7 files use
z_true = (g_true - g_hat) / sd, the opposite sign.

D2 action (DESIGN 15.2). The antenna rings are invariant under sigma_x: (x, y) -> (-x, y) and sigma_y: (x, y) -> (x, -y). ``D2Action`` finds the permutation of the
transmitters and of the receivers that each reflection induces by matching the antenna coordinates (it raises unless the match is exact) and acts on data by
(P_g d)[g(t), g(r)] = d[t, r]. With sigma acting on a contrast by the flip of the pixel array (``flip_chi``), the forward map F satisfies F(sigma chi) = P_sigma F(chi)
(``tests/test_detectability.py``). The four irreps are labeled by the signs (eps_x, eps_y) of the characters under (sigma_x, sigma_y): A1 (+, +), B1 (-, +) (odd in x,
like c_x), B2 (+, -) (odd in y, like c_y), A2 (-, -); the isotypic projector is Pi_irrep = (1/4) sum_g chi(g) P_g. The data tangent of a boundary deformation of the
two-disk object that lies in an irrep lies in the same irrep of the data space, and w_hat (built from A1 columns plus, for the nuisances c_x and c_y, columns that are
orthogonal to A1) lies in A1: only the A1 part of a model error carries first-order z.

Two gap conventions for a boundary deformation (review of milestone 8). The base-gap response ``z_fit = w_hat^T tau`` is how far the F2 estimate of g moves from the BASE
gap; it is not the error of the estimate, because a deformation changes the deformed object's own edge-to-edge gap by ``gap_change`` (first order, -(Rc (-1)^m + Lc) per
unit amplitude for the coefficients (Rc, Rs, Lc, Ls), i.e. -2 (-1)^m for A1; m = 0, a pure radius change at fixed centers, is d_r - 2 d_g and changes the gap by -2). The
error of the estimate is z_true = z_fit - gap_change / sigma_g per unit amplitude, equivalently the first-order analysis of the gap-preserving tangent
``true_gap_tangent`` = tau + (gap_change / 2) tau_0 with tau_0 the m = 0 A1 tangent (it lies in range(D), so P_perp tau_0 = 0 and dPhi is the same in both conventions).
For m = 0 and 1 the checks are alpha = 0 and z_true = 0.

Second order (``curvature_tensor``, ``second_order_bias``, ``residual_curvature_operator``). With H = d2 mu (2N, k, k) in whitened units, the least squares step to second
order is p = p1 + p2, p1 = (D^T D)^-1 D^T delta and p2 = (D^T D)^-1 (-(1/2) D^T H[p1, p1] + H[p1, .]^T P_perp delta): the first term is the parameter-effects
(tangential) curvature, the second the intrinsic (normal) curvature times the part of delta that the tangent space does not explain. At a fit with residual r = y - mu(p*) the
Hessian of Phi is D^T D - <r, H>, and the Gauss-Newton Hessian D^T D differs from it by the operator (D^T D)^-1 <r, H> (eigenvalues lambda; the Laplace log evidence of the
exact Hessian exceeds the Gauss-Newton one by -(1/2) sum log(1 - lambda)).

Boundary deformations. ``mode_chi_derivatives`` returns d chi / d (amplitude) for the four basis normal displacements of the two circles of a ``shapes.TwoDisks``:
(right cos, right sin, left cos, left sin) of harmonic m, v_n = cos(m phi) or sin(m phi) with phi the polar angle about each disk's own center (exact boundary integrals,
``shapes.circle_mode_integrals``). The action of the reflections on the four coefficients is ``mode_action``; ``irrep_mode_coefficients`` gives the combination that
spans each irrep (A1: cos(m phi) on the right disk and its sigma_x image cos(m (pi - phi)) = (-1)^m cos(m phi) on the left).
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from . import shapes as sh

IRREPS = ("A1", "B1", "B2", "A2")
SIGNS = {"A1": (1, 1), "B1": (-1, 1), "B2": (1, -1), "A2": (-1, -1)}  # (sigma_x, sigma_y) characters
GROUP = ("e", "sx", "sy", "c2")  # c2 = sigma_x sigma_y = rotation by pi
_GROUP_SIGNS = {"e": (1, 1), "sx": (-1, 1), "sy": (1, -1), "c2": (-1, -1)}  # (x, y) multipliers of the position


# ----------------------------------------------------------------------------------------------
# Whitened real stacking
# ----------------------------------------------------------------------------------------------


def stack_real(d, s=None):
    """Real stacked vector(s) of complex data: [Re d.ravel(); Im d.ravel()] over the last two axes, (..., 2 N). With ``s`` the whitening sqrt(2) / s is applied."""
    d = np.asarray(d)
    flat = d.reshape(d.shape[:-2] + (-1,))
    out = np.concatenate([flat.real, flat.imag], axis=-1)
    return out if s is None else out * (np.sqrt(2.0) / s)


def unstack_real(y, shape, s=None):
    """Inverse of ``stack_real``: (..., 2 N) real -> (..., Nt, Nr) complex (``shape`` = (Nt, Nr))."""
    y = np.asarray(y, dtype=np.float64)
    if s is not None:
        y = y * (s / np.sqrt(2.0))
    n = shape[0] * shape[1]
    return (y[..., :n] + 1j * y[..., n:]).reshape(y.shape[:-1] + tuple(shape))


# ----------------------------------------------------------------------------------------------
# Tangents of the forward map
# ----------------------------------------------------------------------------------------------

_TANGENT_FNS = {}


def _tangent_fn(forward, count):
    """Jitted (chi0, dchis (count, n, n)) -> complex (count, Nt, Nr): the tangents of chi -> d = forward(chi)[0] at chi0 (one primal solve, one tangent solve each)."""
    key = (id(forward), int(count))
    if key not in _TANGENT_FNS:

        def tangents(chi0, dchis):
            _, lin = jax.linearize(lambda c: forward(c)[0], chi0)
            return jnp.stack([lin(dchis[k]) for k in range(count)])

        _TANGENT_FNS[key] = (forward, jax.jit(tangents))  # keep ``forward`` alive so that its id is not reused
    return _TANGENT_FNS[key][1]


def data_tangents(forward, chi0, dchis):
    """Complex data tangents (K, Nt, Nr) of ``forward`` at the contrast ``chi0`` (n, n) along the contrast perturbations ``dchis`` (K, n, n)."""
    dchis = jnp.asarray(dchis, dtype=jnp.float64)
    return np.asarray(_tangent_fn(forward, dchis.shape[0])(jnp.asarray(chi0, dtype=jnp.float64), dchis))


def model_chi_tangents(model, renderer, p0):
    """(chi0 (n, n), dchi (k, n, n)): the contrast of ``model`` (a ``shapemodels.ShapeModel``) at the PHYSICAL parameters ``p0`` (k,) and its exact derivatives with
    respect to every physical parameter (``jax.jvp`` through the exact boundary derivatives of ``ellipses.EllipseRenderer``)."""
    p0 = jnp.asarray(p0, dtype=jnp.float64)
    chi_p = lambda p: renderer.chi(*model.scene(p))
    eye = jnp.eye(model.dim)
    cols = [jax.jvp(chi_p, (p0,), (eye[k],)) for k in range(model.dim)]
    return np.asarray(cols[0][0]), np.stack([np.asarray(c[1]) for c in cols])


def whitened_jacobian(model, forward, renderer, p0, s):
    """Whitened real-stacked Jacobian D (2N, k) of the data of ``model`` with respect to its physical parameters at ``p0``, noise level ``s``.

    Column j is sqrt(2)/s [Re; Im] of d d / d p_j (``stack_real``), so D^T D is the Fisher matrix of the likelihood (no prior) and Phi = (1/2) ||r_w||^2. Returns
    ``(D, chi0)`` with chi0 the contrast at ``p0`` (n, n).
    """
    chi0, dchi = model_chi_tangents(model, renderer, p0)
    tang = data_tangents(forward, chi0, dchi)  # (k, Nt, Nr)
    return stack_real(tang, s).T, chi0


# ----------------------------------------------------------------------------------------------
# The first-order law
# ----------------------------------------------------------------------------------------------


def gap_direction(D, k=0):
    """``(w_hat (2N,), sigma_k)``: the unit "pure k" direction of range(D) (orthogonal to the other columns, positive inner product with column k) and the Cramer-Rao
    standard deviation sigma_k = [(D^T D)^-1]_kk = 1 / ||P_others d_k||. Computed from QR factorizations (no normal equations)."""
    D = np.asarray(D, dtype=np.float64)
    others = np.delete(D, k, axis=1)
    if others.shape[1]:
        Q, _ = np.linalg.qr(others)
        w = D[:, k] - Q @ (Q.T @ D[:, k])
    else:
        w = D[:, k].copy()
    nw = np.linalg.norm(w)
    return w / nw, 1.0 / nw


def projector_perp(D):
    """P_perp = I - D (D^T D)^-1 D^T as a dense (2N, 2N) matrix (via QR; use for small problems and tests; ``analyze_error`` never builds it)."""
    Q, _ = np.linalg.qr(np.asarray(D, dtype=np.float64))
    return np.eye(Q.shape[0]) - Q @ Q.T


def analyze_error(D, delta, k=0):
    """First-order consequences of the whitened model error ``delta`` (2N,) for a model with whitened Jacobian ``D`` (2N, p) and the parameter with index ``k`` (the gap).

    Returns a dict: ``dp`` (p,) least-squares bias (D^T D)^-1 D^T delta; ``sigma`` (p,) = sqrt(diag (D^T D)^-1); ``z`` = w_hat^T delta (= dp_k / sigma_k);
    ``dphi`` = (1/2) ||P_perp delta||^2; ``alpha`` angle of delta to range(D), ``beta`` angle of its tangential part to w_hat (radians; nan if undefined);
    ``c_geom`` = (1/2) tan^2(alpha) / cos^2(beta) (inf if z = 0, nan if delta = 0); ``c_direct`` = dphi / z^2 (the same number by algebra; reported to check the angles);
    ``norm2_half`` = (1/2) ||delta||^2 (the model-error energy in nats); ``norm_tangent``, ``norm_perp``; ``w_hat`` (2N,); ``sigma_k``.
    """
    D = np.asarray(D, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    Q, R = np.linalg.qr(D)
    coef = Q.T @ delta
    tang = Q @ coef
    perp = delta - tang
    dp = np.linalg.solve(R, coef)
    sig = np.sqrt(np.diag(np.linalg.inv(R.T @ R)))
    w_hat, sigma_k = gap_direction(D, k)
    z = float(w_hat @ delta)
    nT, nP = float(np.linalg.norm(tang)), float(np.linalg.norm(perp))
    alpha = float(np.arctan2(nP, nT))
    if nT > 0.0 and z != 0.0:
        beta = float(np.arccos(np.clip(z / nT, -1.0, 1.0)))
        c_geom = 0.5 * np.tan(alpha) ** 2 / np.cos(beta) ** 2 if nT > 0 else float("nan")
    else:
        beta = float("nan")
        c_geom = float("inf") if nP > 0.0 else float("nan")
    dphi = 0.5 * nP**2
    return {"dp": dp, "sigma": sig, "z": z, "dphi": dphi, "alpha": alpha, "beta": beta, "c_geom": float(c_geom),
            "c_direct": float(dphi / z**2) if z != 0.0 else float("inf"), "norm2_half": 0.5 * float(delta @ delta), "norm_tangent": nT, "norm_perp": nP,
            "w_hat": w_hat, "sigma_k": float(sigma_k)}


def noise_free_fit(post, p_start, max_iter=25):
    """Noise-free nonlinear fit of the model of the ``shapepost.ShapePosterior`` ``post`` (its data are the noise-free data of the truth): the MAP (prior included,
    as in milestone 7) from the physical start ``p_start``: L-BFGS then damped Gauss-Newton. Returns ``(p_hat, phi, converged)``; phi = Phi at the MAP (the
    misfit excess dPhi, since the data carry no noise)."""
    from . import shapepost as sp

    u0 = np.asarray(post.model.u_from_p(jnp.asarray(p_start, dtype=jnp.float64)))
    u1, _ = sp.lbfgs_map(post, u0, maxiter=100)
    u, info = sp.newton_map(post, u1, max_iter=max_iter, tol=1e-6)
    return post.physical(u), float(info["phi"]), bool(info["converged"])


# ----------------------------------------------------------------------------------------------
# The D2 action on data and on contrasts
# ----------------------------------------------------------------------------------------------


def flip_chi(chi, g):
    """The reflection ``g`` in ('e', 'sx', 'sy', 'c2') applied to a contrast array (..., n, n) ("ij" = [x, y] indexing) on a grid centered at the origin:
    sigma_x reverses the x axis, sigma_y the y axis, c2 both. (The contrast of the reflected object is chi o g^-1 = chi o g.)"""
    chi = np.asarray(chi)
    if g == "e":
        return chi
    if g == "sx":
        return chi[..., ::-1, :]
    if g == "sy":
        return chi[..., :, ::-1]
    if g == "c2":
        return chi[..., ::-1, ::-1]
    raise ValueError(f"unknown group element {g!r}")


def _match(points, image, atol):
    """perm with image[k] = points[perm[k]] (nearest points, must be exact within ``atol`` and a bijection)."""
    dist = np.linalg.norm(image[:, None, :] - points[None, :, :], axis=-1)
    perm = np.argmin(dist, axis=1)
    err = float(dist[np.arange(len(perm)), perm].max())
    if err > atol:
        raise ValueError(f"the antenna set is not invariant under the reflection (worst match error {err:.3e} > {atol:.1e})")
    if len(set(perm.tolist())) != len(perm):
        raise ValueError("the reflection does not map the antenna set onto itself bijectively")
    return perm


class D2Action:
    """The group D2 = {e, sigma_x, sigma_y, c2} acting on data by permuting the (transmitter, receiver) pairs (module docstring).

    ``antennas`` is a ``geometry.Antennas`` with rings centered at the origin. ``perm_tx[g]`` / ``perm_rx[g]`` are the index arrays with
    position(perm[k]) = g(position(k)) (exact within ``atol`` and bijective, else ``ValueError``); each is an involution. ``apply(g, d)`` for complex or real data of shape
    (..., Nt, Nr), ``apply_stacked(g, y)`` for real stacked vectors (..., 2N); ``project(irrep, d)`` the isotypic projection; ``components(d)`` the dict of the four.
    """

    def __init__(self, antennas, atol=1e-9):
        self.n_tx, self.n_rx = antennas.n_tx, antennas.n_rx
        self.perm_tx, self.perm_rx = {}, {}
        for g, (sx, sy) in _GROUP_SIGNS.items():
            mult = np.array([sx, sy], dtype=np.float64)
            self.perm_tx[g] = _match(antennas.tx, antennas.tx * mult, atol)
            self.perm_rx[g] = _match(antennas.rx, antennas.rx * mult, atol)
        # perm: image[k] = points[perm[k]], i.e. g(position k) is the position of antenna perm[k]. Data: (P_g d)[perm[t], perm[r]] = d[t, r]
        for g in GROUP:
            for perm in (self.perm_tx[g], self.perm_rx[g]):
                if not np.array_equal(perm[perm], np.arange(len(perm))):
                    raise ValueError(f"the permutation of {g} is not an involution")

    def apply(self, g, d):
        """P_g d for data (..., Nt, Nr): out[..., perm_tx[t], perm_rx[r]] = d[..., t, r] (an exact permutation of the entries)."""
        d = np.asarray(d)
        out = np.empty_like(d)
        out[..., self.perm_tx[g][:, None], self.perm_rx[g][None, :]] = d
        return out

    def apply_stacked(self, g, y):
        """P_g on real stacked vectors (..., 2N) (``stack_real`` layout): the same permutation of the pairs in both blocks."""
        y = np.asarray(y)
        n = self.n_tx * self.n_rx
        shape = y.shape[:-1] + (self.n_tx, self.n_rx)
        re = self.apply(g, y[..., :n].reshape(shape)).reshape(y.shape[:-1] + (n,))
        im = self.apply(g, y[..., n:].reshape(shape)).reshape(y.shape[:-1] + (n,))
        return np.concatenate([re, im], axis=-1)

    def permutation_matrix(self, g):
        """The (N, N) permutation matrix of ``apply`` on the flattened pair index t Nr + r (for tests)."""
        n = self.n_tx * self.n_rx
        eye = np.eye(n).reshape(n, self.n_tx, self.n_rx)
        return self.apply(g, eye).reshape(n, n).T

    def project(self, irrep, d, stacked=False):
        """Isotypic projection (1/4) sum_g chi(g) P_g d. ``stacked`` for real stacked vectors."""
        sx, sy = SIGNS[irrep]
        act = self.apply_stacked if stacked else self.apply
        total = 0.0
        for g, (gx, gy) in _GROUP_SIGNS.items():
            # character of g in irrep (eps_x, eps_y): eps_x^[g flips x] * eps_y^[g flips y]
            chi_g = (sx if gx < 0 else 1) * (sy if gy < 0 else 1)
            total = total + chi_g * act(g, d)
        return 0.25 * total

    def components(self, d, stacked=False):
        """{irrep: projection of d} for the four irreps (they sum to d and are mutually orthogonal)."""
        return {name: self.project(name, d, stacked) for name in IRREPS}

    def fractions(self, y, w_hat=None):
        """For a real stacked vector ``y`` (2N,): ``{irrep: {"frac_norm2": ||Pi y||^2 / ||y||^2, "z": w_hat^T Pi y or None}}``."""
        comps = self.components(y, stacked=True)
        tot = float(y @ y)
        return {k: {"frac_norm2": float(v @ v) / tot if tot > 0 else float("nan"), "z": None if w_hat is None else float(w_hat @ v)} for k, v in comps.items()}


# ----------------------------------------------------------------------------------------------
# Boundary deformation modes of the two-disk object
# ----------------------------------------------------------------------------------------------

MODE_LABELS = ("Rc", "Rs", "Lc", "Ls")  # right cos, right sin, left cos, left sin (polar angle about each disk's own center)


def mode_chi_derivatives(grid, obj, m):
    """d chi / d amplitude (4, n, n) for the unit normal displacements (right cos(m phi), right sin(m phi), left cos(m phi), left sin(m phi)) of the boundaries of the two
    disks of ``obj`` (a ``shapes.TwoDisks``): (eps_r - 1) * (boundary integral of the displacement in the pixel) / h^2. The sin rows are zero for m = 0."""
    obj.check_inside(grid)
    out = np.zeros((4, grid.n, grid.n))
    for side, (xc, yc, rad) in enumerate((obj.right, obj.left)):
        ints = sh.circle_mode_integrals(grid, xc, yc, rad, m)
        out[2 * side] = ints[0]
        out[2 * side + 1] = ints[1]
    return (obj.eps_r - 1.0) * out / grid.h**2


def mode_action(m, g):
    """The 4 x 4 matrix of the reflection ``g`` on the coefficient vectors (Rc, Rs, Lc, Ls) of ``mode_chi_derivatives`` for harmonic m, acting as c -> M c on the
    coefficients of the transported displacement field.

    sigma_y: phi -> -phi on both disks: diag(1, -1, 1, -1). sigma_x maps the right disk to the left one with phi -> pi - phi, so the field v_R = cos(m phi)
    becomes v_L(phi') = cos(m (pi - phi')) = (-1)^m cos(m phi') and sin(m phi) becomes sin(m (pi - phi')) = -(-1)^m sin(m phi'): Rc -> (-1)^m Lc, Rs -> -(-1)^m Ls and the same
    from left to right. c2 is their product."""
    sgn = (-1.0) ** int(m)
    sy = np.diag([1.0, -1.0, 1.0, -1.0])
    sx = np.zeros((4, 4))
    sx[2, 0], sx[3, 1], sx[0, 2], sx[1, 3] = sgn, -sgn, sgn, -sgn
    table = {"e": np.eye(4), "sx": sx, "sy": sy, "c2": sx @ sy}
    return table[g]


def irrep_mode_coefficients(m, irrep):
    """Unnormalized coefficient vector (Rc, Rs, Lc, Ls) of the harmonic-m deformation in ``irrep`` with unit cosine/sine amplitude on the right disk (A1: cos(m phi) right,
    (-1)^m cos(m phi) left): A1 (1, 0, s, 0), B1 (1, 0, -s, 0), B2 (0, 1, 0, -s), A2 (0, 1, 0, s) with s = (-1)^m. For m = 0 the sin rows are zero fields and B2, A2 return
    the zero vector's coefficient (still defined, but the field is zero)."""
    s = (-1.0) ** int(m)
    table = {"A1": (1.0, 0.0, s, 0.0), "B1": (1.0, 0.0, -s, 0.0), "B2": (0.0, 1.0, 0.0, -s), "A2": (0.0, 1.0, 0.0, s)}
    return np.array(table[irrep])


def mode_irrep_projector(m, irrep):
    """(1/4) sum_g chi(g) M_g on the four mode coefficients: the projector onto the irrep (4, 4)."""
    sx, sy = SIGNS[irrep]
    tot = np.zeros((4, 4))
    for g, (gx, gy) in _GROUP_SIGNS.items():
        tot += (sx if gx < 0 else 1) * (sy if gy < 0 else 1) * mode_action(m, g)
    return 0.25 * tot


# ----------------------------------------------------------------------------------------------
# The spectrum of one base object
# ----------------------------------------------------------------------------------------------


def mode_tangents(forward, obj, m, s, chi0=None):
    """Whitened stacked data tangents of the four basis deformations of harmonic m at the symmetric object ``obj``: array (4, 2N) in the order of ``MODE_LABELS``."""
    grid = forward.setup.grid
    dchi = mode_chi_derivatives(grid, obj, m)
    chi0 = obj.chi(grid) if chi0 is None else chi0
    return stack_real(data_tangents(forward, chi0, dchi), s)


def gap_change_per_amplitude(m, coef):
    """First-order change of the deformed object's edge-to-edge gap per unit amplitude of the boundary deformation with coefficients ``coef`` = (Rc, Rs, Lc, Ls) of
    harmonic ``m`` (``MODE_LABELS``): the right disk's gap-side boundary point is at phi = pi, where cos(m phi) = (-1)^m and sin(m phi) = 0, and the left disk's at phi = 0,
    where cos = 1. A positive normal displacement there moves the boundary into the gap, so gap_change = -(Rc (-1)^m + Lc). (To first order the minimum distance between the
    two boundaries moves by the displacement at the axis point: the shift of the closest point changes it only at second order.) A1: -2 (-1)^m, B1 and the sin modes: 0."""
    coef = np.asarray(coef, dtype=np.float64)
    return float(-(coef[0] * (-1.0) ** int(m) + coef[2]))


def true_gap_tangent(tau, tau0, m, coef):
    """The gap-preserving data tangent tau + (gap_change / 2) tau0, whitened stacked (2N,) like ``tau``, ``tau0`` (the m = 0 A1 tangent, which changes the gap by -2 per unit
    amplitude and lies in range(D)). Its first-order analysis (``analyze_error``) is the one of the deformation in the error-of-the-estimate convention: z is the error of
    the gap estimate in sigma_g (z_true), and dPhi is unchanged because P_perp tau0 = 0."""
    return np.asarray(tau) + 0.5 * gap_change_per_amplitude(m, coef) * np.asarray(tau0)


# ----------------------------------------------------------------------------------------------
# Second order: curvature of the model manifold
# ----------------------------------------------------------------------------------------------

DEFAULT_STEP_FLOOR = 0.1


def default_steps(p0, rel=1e-3):
    """Central-difference steps rel * max(|p0|, 0.1) for the physical parameters (k,)."""
    return rel * np.maximum(np.abs(np.asarray(p0, dtype=np.float64)), DEFAULT_STEP_FLOOR)


def jacobian_derivative(model, forward, renderer, p0, direction, s, step):
    """Derivative of the whitened Jacobian along ``direction`` (k,): (D(p0 + step direction) - D(p0 - step direction)) / (2 step), shape (2N, k) = H[direction, .] in the
    notation of the module docstring. Central difference of the exact (jvp) Jacobian: the error is O(step^2), the solver tolerance enters divided by ``step``."""
    direction = np.asarray(direction, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    Dp, _ = whitened_jacobian(model, forward, renderer, p0 + step * direction, s)
    Dm, _ = whitened_jacobian(model, forward, renderer, p0 - step * direction, s)
    return (Dp - Dm) / (2.0 * step)


def curvature_tensor(model, forward, renderer, p0, s, steps=None):
    """Second derivatives of the whitened data map in the physical parameters at ``p0``: H (2N, k, k), H[:, j, l] = d2 y_w / d p_j d p_l (symmetrized; the unsymmetrized
    asymmetry is returned too). Built from 2 k Jacobians (central differences along each coordinate direction with ``steps``, default ``default_steps``).
    Returns ``(H, asym)`` with asym = max |H - H^T| / max |H| (j <-> l), a finite-difference error indicator."""
    p0 = np.asarray(p0, dtype=np.float64)
    k = p0.size
    steps = default_steps(p0) if steps is None else np.asarray(steps, dtype=np.float64)
    cols = []
    for l in range(k):
        e = np.zeros(k)
        e[l] = 1.0
        cols.append(jacobian_derivative(model, forward, renderer, p0, e, s, steps[l]))  # (2N, k): [:, j] = d D_j / d p_l
    H = np.stack(cols, axis=-1)  # (2N, j, l)
    asym = float(np.abs(H - H.transpose(0, 2, 1)).max() / np.abs(H).max())
    return 0.5 * (H + H.transpose(0, 2, 1)), asym


def second_order_bias(D, H, delta):
    """Second-order least-squares bias of the model to the whitened error ``delta`` (2N,) at the base point (D (2N, k), H (2N, k, k) from ``curvature_tensor``).

    Returns a dict: ``p1`` = (D^T D)^-1 D^T delta (first order), ``p2_tangential`` = -(1/2) (D^T D)^-1 D^T H[p1, p1] (parameter-effects curvature), ``p2_normal`` =
    (D^T D)^-1 H[p1, .]^T P_perp delta (intrinsic curvature acting on the unexplained part of delta), ``p2`` their sum, ``p`` = p1 + p2. The series does not converge when
    ``residual_curvature_operator`` at the base has an eigenvalue near or above 1 in size."""
    D = np.asarray(D, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    Q, R = np.linalg.qr(D)
    coef = Q.T @ delta
    p1 = np.linalg.solve(R, coef)
    perp = delta - Q @ coef
    h11 = np.einsum("ijl,j,l->i", H, p1, p1)
    h1 = np.einsum("ijl,j->il", H, p1)  # (2N, k): H[p1, .]
    p2t = -0.5 * np.linalg.solve(R, Q.T @ h11)
    p2n = np.linalg.solve(R.T @ R, h1.T @ perp)
    return {"p1": p1, "p2_tangential": p2t, "p2_normal": p2n, "p2": p2t + p2n, "p": p1 + p2t + p2n}


def residual_curvature_operator(D, H, resid):
    """Eigenvalues (descending) and the symmetric matrix of the residual-curvature operator (D^T D)^-1 <r, H> for the whitened residual ``resid`` = y - mu(p) (2N,) at p.

    The Hessian of Phi = (1/2) ||y - mu||^2 is D^T D - <r, H> = D^T D (I - A), A = (D^T D)^-1 <r, H>. It is similar to the symmetric R^-T <r, H> R^-1 (D^T D = R^T R), whose
    eigenvalues are returned: lambda >= 1 means the stationary point is not a minimum, and lambda near 1 means the Gauss-Newton Hessian is far from the true one.
    Returns ``(eigenvalues, A_sym)``."""
    D = np.asarray(D, dtype=np.float64)
    _, R = np.linalg.qr(D)
    B = np.einsum("i,ijl->jl", np.asarray(resid, dtype=np.float64), H)
    Ri = np.linalg.inv(R)
    A = Ri.T @ B @ Ri
    A = 0.5 * (A + A.T)
    return np.sort(np.linalg.eigvalsh(A))[::-1], A


def log_evidence_gap(eigenvalues):
    """-(1/2) sum log(1 - lambda): the predicted log Z(exact Hessian) - log Z(Gauss-Newton Hessian) from the eigenvalues of ``residual_curvature_operator``."""
    return float(-0.5 * np.sum(np.log1p(-np.asarray(eigenvalues, dtype=np.float64))))


def foot_point(model, forward, renderer, y_true, s, p_start, k=0, max_iter=40, tol=1e-10):
    """Profile foot point: the least-squares fit of the model to the whitened noise-free data ``y_true`` (2N,) with parameter ``k`` (the gap) held at ``p_start[k]`` and all
    other parameters free (no prior): damped Gauss-Newton with step halving on Phi. At it the first-order bias p1 = (D^T D)^-1 D^T delta has the projected gradient of the
    nuisances equal to zero, D_nuis^T delta = 0, so the first-order z is a pure statement about the gap. Returns ``(p, phi, converged)``."""
    p = np.asarray(p_start, dtype=np.float64).copy()
    free = [j for j in range(p.size) if j != k]

    def resid_of(q):
        chi = np.asarray(renderer.chi(*model.scene(jnp.asarray(q, dtype=jnp.float64))))
        return y_true - stack_real(np.asarray(forward(jnp.asarray(chi))[0]), s)

    r = resid_of(p)
    phi = 0.5 * float(r @ r)
    ok = False
    for _ in range(max_iter):
        D, _ = whitened_jacobian(model, forward, renderer, p, s)
        Df = D[:, free]
        g = Df.T @ r
        step = np.linalg.lstsq(Df, r, rcond=None)[0]
        if np.linalg.norm(g) <= tol * max(1.0, np.linalg.norm(Df.T @ y_true)):
            ok = True
            break
        t = 1.0
        while t > 1e-6:
            q = p.copy()
            q[free] += t * step
            try:
                rq = resid_of(q)
            except Exception:  # an invalid physical parameter set: shorten the step
                t *= 0.5
                continue
            phq = 0.5 * float(rq @ rq)
            if np.isfinite(phq) and phq <= phi:
                p, r, phi = q, rq, phq
                break
            t *= 0.5
        else:
            break
    return p, phi, ok
