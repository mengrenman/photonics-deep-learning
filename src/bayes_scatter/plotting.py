"""Figures: reconstruction maps with a z-map, sampler traces, and the Born exact-posterior comparison.

Maps are (n, n) arrays indexed [i, j] = [x, y] (docs/DESIGN.md section 1) and are drawn with
``imshow(a.T, origin="lower", extent=...)`` so that x runs to the right and y upward. Figures are built
with the object-oriented ``matplotlib.figure.Figure`` API (no pyplot, no global backend state) and saved
with ``Figure.savefig``.
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure


def _extent(grid):
    cx, cy = grid.center
    half = 0.5 * grid.side
    return [cx - half, cx + half, cy - half, cy + half]


def _save(fig, path, dpi=130):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def _show(fig, ax, grid, a, title, cmap="viridis", vmin=None, vmax=None, label=None):
    im = ax.imshow(np.asarray(a).T, origin="lower", extent=_extent(grid), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [wavelengths]", fontsize=8)
    ax.set_ylabel("y [wavelengths]", fontsize=8)
    ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=7)
    if label:
        cb.set_label(label, fontsize=8)
    return im


def z_map(mean, truth, std):
    """z = (mean - truth) / std, NaN where std is not positive."""
    mean, truth, std = np.asarray(mean), np.asarray(truth), np.asarray(std)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, (mean - truth) / std, np.nan)


def plot_maps(truth, map, mean, std, grid, path, quantity="eps_r", title=None, z_limit=4.0):
    """Truth, MAP, posterior mean, posterior standard deviation and the z-map, in one row.

    Parameters
    ----------
    truth, map, mean, std : (n, n) arrays of the same quantity (for example eps_r or log eps_r).
        ``map`` may be None to skip that panel (the name follows the contract of the module; it shadows the
        builtin only inside this function).
    grid : ``geometry.Grid`` (for the extent and orientation).
    path : output image file (parent directories are created).
    quantity : label for the color bars.
    z_limit : color range of the z panel is [-z_limit, z_limit].

    The last panel is z = (mean - truth) / std, whose title reports its RMS and the fraction of pixels with
    |z| > 2; a calibrated posterior that contains the truth has RMS near 1 where the data constrain the
    pixel, and the z-map shows where it does not (large |z| means the truth is far outside the posterior).
    """
    panels = [("truth", truth, "viridis")]
    if map is not None:
        panels.append(("MAP", map, "viridis"))
    panels.append(("posterior mean", mean, "viridis"))
    arrays = [np.asarray(a) for _, a, _ in panels]
    vmin, vmax = min(a.min() for a in arrays), max(a.max() for a in arrays)

    z = z_map(mean, truth, std)
    n_panels = len(panels) + 2
    fig = Figure(figsize=(3.3 * n_panels, 3.5), layout="constrained")
    axes = fig.subplots(1, n_panels)
    for ax, (name, a, cmap) in zip(axes, panels):
        _show(fig, ax, grid, a, name, cmap, vmin, vmax, quantity)
    _show(fig, axes[-2], grid, std, "posterior std", "magma", 0.0, None, quantity)
    ok = np.isfinite(z)
    rms = float(np.sqrt(np.mean(z[ok] ** 2))) if ok.any() else float("nan")
    frac = float(np.mean(np.abs(z[ok]) > 2.0)) if ok.any() else float("nan")
    _show(fig, axes[-1], grid, z, f"z = (mean - truth)/std\nrms {rms:.2f}, |z|>2 in {100 * frac:.0f}% of pixels",
          "RdBu_r", -z_limit, z_limit, "z")
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def plot_traces(phi, path, extra=None, title=None, phi_label="Phi"):
    """Trace plots of Phi (and optional extra scalar series), one panel per quantity, one line per chain.

    phi : (n_chains, n_steps) or (n_steps,) potential at every post-warmup step.
    extra : optional dict name -> array of shape (n_chains, n_points) drawn in further panels (for example
        log eps_r at a few fixed pixels, thinned, or the misfit of the stored draws).
    phi_label : y label of the first panel. Pass a label such as "Phi' (sampler potential)" when the sampler
        ran in reparametrized coordinates, so that the trace is not mistaken for the data misfit.
    """
    series = {phi_label: np.atleast_2d(np.asarray(phi))}
    for name, a in (extra or {}).items():
        series[name] = np.atleast_2d(np.asarray(a))
    fig = Figure(figsize=(9.0, 1.9 * len(series) + 0.6), layout="constrained")
    axes = np.atleast_1d(fig.subplots(len(series), 1, sharex=False))
    for ax, (name, a) in zip(axes, series.items()):
        for c, row in enumerate(a):
            ax.plot(np.arange(row.size), row, lw=0.6, alpha=0.8, label=f"chain {c}")
        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("stored draw", fontsize=9)
    axes[0].legend(fontsize=7, ncol=min(4, series[phi_label].shape[0]), loc="upper right")
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def plot_born_check(exact_mean, exact_std, results, path, title=None):
    """Sampler estimates against the exact Born posterior, one column per sampler.

    exact_mean, exact_std : arrays of the same size (pixel posterior mean and standard deviation).
    results : dict name -> dict with ``mean`` and ``std`` (sampler estimates, same size) and ``z`` (the
        standardized mean errors, one per pixel).
    Rows: estimated vs exact mean; estimated vs exact standard deviation; histogram of z with the N(0, 1)
    density (the z values of neighboring pixels are strongly correlated, so the histogram is a rough guide).
    """
    em, es = np.ravel(exact_mean), np.ravel(exact_std)
    names = list(results)
    fig = Figure(figsize=(3.2 * len(names), 8.6), layout="constrained")
    axes = np.atleast_2d(fig.subplots(3, len(names), squeeze=False))
    grid_z = np.linspace(-4.5, 4.5, 200)
    for col, name in enumerate(names):
        r = results[name]
        for row, (est, ex, lab) in enumerate([(r["mean"], em, "mean"), (r["std"], es, "std")]):
            ax = axes[row, col]
            est = np.ravel(est)
            lo, hi = min(ex.min(), est.min()), max(ex.max(), est.max())
            ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
            ax.plot(ex, est, ".", ms=4)
            ax.set_xlabel(f"exact posterior {lab}", fontsize=8)
            ax.set_ylabel(f"sampled {lab}", fontsize=8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(name, fontsize=10)
        ax = axes[2, col]
        z = np.ravel(r["z"])
        ax.hist(z, bins=np.linspace(-4.5, 4.5, 19), density=True, alpha=0.7)
        ax.plot(grid_z, np.exp(-0.5 * grid_z**2) / np.sqrt(2 * np.pi), "k", lw=0.8)
        ax.set_xlabel("z = (mean_hat - exact)/MCSE", fontsize=8)
        ax.tick_params(labelsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    return _save(fig, path)
