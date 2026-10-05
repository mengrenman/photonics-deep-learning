"""Gauss-Newton spectrum at the MAP (precision of the posterior in whitened coordinates minus the prior's identity).

Reads the eigenpairs cached by scripts/run_posterior_laplace.py (results/laplace_cache/<name>.npz, the K = 150 largest
eigenvalues) and writes results/figures/gn_spectrum.png. The dotted lines mark the prior curvature 1 and the level 1.5
used to count "data-determined" directions (posterior variance below 0.4 of the prior's along that direction).

Usage:  python scripts/plot_gn_spectrum.py [NAME=LABEL ...]   (default: weak, mod)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = {"weak": "weak (cylinder, eps_r 1.5)", "mod": "moderate (austria_fixed, eps_r 2)"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    items = dict(a.split("=", 1) for a in argv) if argv else DEFAULT
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, label in items.items():
        lam = np.load(ROOT / "results" / "laplace_cache" / f"{name}.npz")["lam"]
        ax.semilogy(np.arange(1, lam.size + 1), lam, label=f"{label}: {int((lam > 1.5).sum())} above 1.5")
    for level in (1.0, 1.5):
        ax.axhline(level, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue of the Gauss-Newton Hessian (whitened coordinates)")
    ax.set_title("n=32, side 2, 30 dB: posterior precision minus the prior's identity, at the MAP", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ROOT / "results" / "figures" / "gn_spectrum.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
