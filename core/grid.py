"""Grid construction for the lattice-point sampling experiment."""

from __future__ import annotations

import numpy as np


def build_grid(width: int, height: int) -> np.ndarray:
    """Return every integer lattice point in [0, width) x [0, height) as an (N, 2) array."""
    if width < 2 or height < 2:
        raise ValueError("Grid must be at least 2x2 so distinct points can form a triangle.")
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    return np.column_stack([xs.ravel(), ys.ravel()])
