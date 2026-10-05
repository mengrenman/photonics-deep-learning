"""Grid, antenna layout and measurement setup (docs/DESIGN.md sections 1 and 2.5).

Conventions: lengths in free-space wavelengths (wavelength = 1, k0 = 2 pi), e^{-i w t} time dependence.
Pixel arrays are indexed [i, j] = [x, y] ("ij" indexing) and flattened in C order, k = i * n + j.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Square domain of interest (DOI): n x n square pixels of side h = side / n.

    Pixel (i, j) has its center at x_i = cx - side/2 + (i + 1/2) h, y_j = cy - side/2 + (j + 1/2) h.
    """

    n: int
    side: float
    center: tuple = (0.0, 0.0)

    def __post_init__(self):
        if int(self.n) != self.n or self.n < 1:
            raise ValueError("n must be a positive integer")
        if not self.side > 0:
            raise ValueError("side must be positive")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "side", float(self.side))
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))

    @property
    def h(self):
        """Pixel side length."""
        return self.side / self.n

    @property
    def a(self):
        """Radius of the equal-area disk that replaces a pixel (Richmond), a = h / sqrt(pi)."""
        return self.h / np.sqrt(np.pi)

    def axes(self):
        """1-D pixel-center coordinates (x_i, y_j), each of shape (n,)."""
        cx, cy = self.center
        offs = (np.arange(self.n) + 0.5) * self.h - 0.5 * self.side
        return cx + offs, cy + offs

    def centers(self):
        """Pixel-center coordinates (X, Y), each (n, n) with "ij" indexing."""
        x, y = self.axes()
        return np.meshgrid(x, y, indexing="ij")


@dataclass(frozen=True, eq=False)
class Antennas:
    """Transmitter positions ``tx`` (Nt, 2) and receiver positions ``rx`` (Nr, 2), as float64 arrays."""

    tx: np.ndarray
    rx: np.ndarray

    def __post_init__(self):
        for name in ("tx", "rx"):
            arr = np.array(getattr(self, name), dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"{name} must have shape (N, 2), got {arr.shape}")
            object.__setattr__(self, name, arr)

    @property
    def n_tx(self):
        return self.tx.shape[0]

    @property
    def n_rx(self):
        return self.rx.shape[0]


@dataclass(frozen=True, eq=False)
class Setup:
    """A grid, an antenna layout and the free-space wavelength (default 1)."""

    grid: Grid
    antennas: Antennas
    wavelength: float = 1.0

    @property
    def k0(self):
        """Free-space wavenumber 2 pi / wavelength."""
        return 2.0 * np.pi / self.wavelength


def ring(count, radius, offset=0.0, center=(0.0, 0.0)):
    """``count`` points uniformly spaced on a circle, shape (count, 2).

    The k-th point sits at angle ``offset + 2 pi k / count`` (radians), k = 0, ..., count - 1.
    """
    ang = offset + 2.0 * np.pi * np.arange(count) / count
    return np.stack(
        [center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)], axis=-1
    )


def ch15_setup(n=96, side=5.0, n_tx=40, r_tx=4.0, n_rx=60, r_rx=4.5):
    """The book's benchmark (Ch. 15, 15.4.1.1): 300 MHz (wavelength 1), 40 transmitters on a circle of
    radius 4 and 60 receivers on a circle of radius 4.5, both starting at angle 0, around a DOI of
    ``side`` x ``side`` wavelengths centered at the origin, discretized on n x n pixels."""
    return Setup(
        grid=Grid(n=n, side=side),
        antennas=Antennas(tx=ring(n_tx, r_tx), rx=ring(n_rx, r_rx)),
        wavelength=1.0,
    )
