"""Band-limited versions of an object: what a posterior mean can at best estimate when the data fix only low spatial frequencies.

The scattering data of the benchmark (40 transmitters, 60 receivers on rings around a DOI of 2 wavelengths, one frequency) fix the
object's spatial-frequency content only inside a disk of radius of the order of 2 k0 = 2 cycles per wavelength (the Ewald-circle
diameter of the first Born approximation); multiple scattering widens the effective support somewhat. A posterior mean that is
pulled toward the data-determined part of the object is therefore expected to resemble a LOW-PASSED copy of the truth, and its error
at a sharp edge to resemble the Gibbs ringing and blur of the low-pass, whatever the prior is. ``lowpass`` makes that reference.

Conventions: fields are (n, n) arrays on a square grid of pixel size ``h`` (wavelengths) with "ij" indexing; spatial frequencies are in
cycles per wavelength (k0 = 1 cycle per wavelength, so the 2 k0 Born support is a cutoff of 2).
"""

import numpy as np


def lowpass(field, h, k_cut, pad=4, pad_value=0.0):
    """Ideal circular low-pass of a real field: keep the Fourier components with |k| <= ``k_cut`` cycles per wavelength.

    ``field`` (n, n) real; ``h`` the pixel size in wavelengths; ``pad`` an integer >= 1: the field is embedded in the corner of a
    (pad n, pad n) array filled with ``pad_value`` (the background value, 0 for theta = log eps_r) before the transform, so that
    the filter acts on the object in an empty plane and not on its periodic continuation (pad = 1 is the periodic, circular
    filter; the default 4 puts the frequency lattice at 1 / (4 side) = 0.125 cycles per wavelength for a 2-wavelength DOI, fine
    enough that the result changes by less than 0.005 in theta when the pad is raised further); the result is cropped back to (n, n). Returns a float64 (n, n) array. The filter is a linear, shift-invariant
    projection for pad = 1, and a constant field (equal to ``pad_value``) is left unchanged for any pad.
    """
    f = np.asarray(field, dtype=np.float64)
    n = f.shape[0]
    if f.shape != (n, n):
        raise ValueError(f"field must be square, got {f.shape}")
    if pad < 1 or int(pad) != pad:
        raise ValueError("pad must be an integer >= 1")
    m = int(pad) * n
    big = np.full((m, m), float(pad_value))
    big[:n, :n] = f
    k = np.fft.fftfreq(m, d=h)
    keep = np.hypot(k[:, None], k[None, :]) <= k_cut * (1.0 + 1e-9)
    return np.real(np.fft.ifft2(np.fft.fft2(big) * keep))[:n, :n]


def pearson(a, b, mask=None):
    """Pearson correlation coefficient of two maps over the pixels in ``mask`` (default: all). Returns NaN for a constant map."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def bin_by_distance(values, dist, bins):
    """Mean of ``values`` (n, n) over the pixels with signed distance ``dist`` (n, n int) equal to each entry of ``bins``.
    Returns ``{str(d): mean}`` for the distances that occur."""
    return {str(b): float(np.asarray(values)[dist == b].mean()) for b in bins if (dist == b).any()}


def disk_average(field, h, radius, pad_value=0.0):
    """Moving average of a field over a disk of radius ``radius`` (wavelengths): each pixel becomes the mean of the pixel centers
    within ``radius`` of it, with the outside of the grid filled with ``pad_value``. (n, n) -> (n, n). A "resolution-cell average" of
    the truth, the other reference besides ``lowpass``. A radius below the pixel size leaves the field unchanged."""
    from scipy.ndimage import convolve

    r = radius / h
    m = int(np.ceil(r))
    yy, xx = np.mgrid[-m:m + 1, -m:m + 1]
    kernel = (np.hypot(xx, yy) <= r + 1e-12).astype(np.float64)
    return convolve(np.asarray(field, dtype=np.float64), kernel / kernel.sum(), mode="constant", cval=float(pad_value))
