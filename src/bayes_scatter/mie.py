"""Analytic scattering of a line source by a centered circular dielectric cylinder (docs/DESIGN.md 8.2).

TM polarization, e^{-i w t}. A cylinder of radius R and relative permittivity eps_r (k1 = k0 sqrt(eps_r),
possibly complex) is illuminated by a unit line source at r_t, u_inc = (i/4) H0^(1)(k0 |r - r_t|). For a
receiver at (rho, phi) and a source at (rho_t, phi_t), both outside the cylinder,

    u_sc = (i/4) sum_{n=-inf}^{inf} a_n H_n(k0 rho_t) H_n(k0 rho) exp(i n (phi - phi_t)),

    a_n = [k1 J_n'(k1 R) J_n(k0 R) - k0 J_n(k1 R) J_n'(k0 R)]
          / [k0 J_n(k1 R) H_n'(k0 R) - k1 J_n'(k1 R) H_n(k0 R)],

H_n = H_n^(1). Since a_{-n} = a_n and H_{-n} = (-1)^n H_n, the sum folds to n >= 0 with weight 2 for n >= 1.
The terms decay like (R^2 / (rho rho_t))^n once n exceeds k1 R, so the series is truncated automatically.
"""

import numpy as np
from scipy.special import hankel1, jv


def _bessel_derivs(n, x):
    """J_n'(x) and H_n^(1)'(x) for integer orders n >= 0 (array) via the recurrence f' = (f_{n-1} - f_{n+1})/2."""
    jn_m, jn_p = jv(n - 1, x), jv(n + 1, x)
    hn_m, hn_p = hankel1(n - 1, x), hankel1(n + 1, x)
    return 0.5 * (jn_m - jn_p), 0.5 * (hn_m - hn_p)


def mie_coefficients(n_max, radius, eps_r, k0):
    """Coefficients a_n, n = 0, ..., n_max, of the series above (complex array of shape (n_max + 1,))."""
    n = np.arange(n_max + 1)
    k1 = k0 * np.sqrt(complex(eps_r))
    j1 = jv(n, k1 * radius)
    dj1, _ = _bessel_derivs(n, k1 * radius)
    j0 = jv(n, k0 * radius)
    dj0, dh0 = _bessel_derivs(n, k0 * radius)
    h0 = hankel1(n, k0 * radius)
    num = k1 * dj1 * j0 - k0 * j1 * dj0
    den = k0 * j1 * dh0 - k1 * dj1 * h0
    return num / den


def _series(tx, rx, radius, eps_r, k0, n_max):
    """Terms of the folded series, shape (n_max + 1, Nt, Nr), without the factor i/4."""
    n = np.arange(n_max + 1)
    rho_t = np.linalg.norm(tx, axis=1)
    rho_r = np.linalg.norm(rx, axis=1)
    phi_t = np.arctan2(tx[:, 1], tx[:, 0])
    phi_r = np.arctan2(rx[:, 1], rx[:, 0])
    a_n = mie_coefficients(n_max, radius, eps_r, k0)
    h_t = hankel1(n[:, None], k0 * rho_t[None, :])  # (N+1, Nt)
    h_r = hankel1(n[:, None], k0 * rho_r[None, :])  # (N+1, Nr)
    weight = np.where(n == 0, 1.0, 2.0)
    dphi = phi_r[None, :] - phi_t[:, None]  # (Nt, Nr)
    cosn = np.cos(n[:, None, None] * dphi[None, :, :])
    return (weight * a_n)[:, None, None] * h_t[:, :, None] * h_r[:, None, :] * cosn


def mie_scattered(tx, rx, radius, eps_r, wavelength=1.0, center=(0.0, 0.0), tol=1e-14,
                  return_nmax=False):
    """Scattered field u_sc (Nt, Nr) at receivers ``rx`` (Nr, 2) from line sources ``tx`` (Nt, 2).

    The cylinder is centered at ``center``. Points must lie outside the cylinder. The mode count is
    increased (starting from k1 R + 10, in steps of 10) until the last three retained modes contribute
    less than ``tol`` of the largest field magnitude; it raises RuntimeError if the terms overflow or
    fail to converge by n = 500. With ``return_nmax`` also returns the mode count used.
    """
    k0 = 2.0 * np.pi / wavelength
    c = np.asarray(center, dtype=np.float64)
    tx = np.atleast_2d(np.asarray(tx, dtype=np.float64)) - c
    rx = np.atleast_2d(np.asarray(rx, dtype=np.float64)) - c
    if min(np.linalg.norm(tx, axis=1).min(), np.linalg.norm(rx, axis=1).min()) <= radius:
        raise ValueError("transmitters and receivers must lie outside the cylinder")
    k1 = k0 * np.sqrt(complex(eps_r))
    n_max = int(np.ceil(max(abs(k1), k0) * radius)) + 10
    while True:
        with np.errstate(all="ignore"):
            terms = _series(tx, rx, radius, eps_r, k0, n_max)
        total = terms.sum(axis=0)
        if np.all(np.isfinite(total)):
            tail = np.abs(terms[-3:]).sum(axis=0).max()
            if tail <= tol * np.abs(total).max():
                break
        if n_max >= 500:
            raise RuntimeError("Mie series did not converge by n = 500")
        n_max += 10
    u = 0.25j * total
    return (u, n_max) if return_nmax else u
