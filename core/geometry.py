"""Triangle classification for points on an integer lattice.

All coordinates are integers, so squared side lengths are integers too;
comparing squared lengths avoids floating point error entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TriangleType(str, Enum):
    """Classification of three points on the grid."""

    DEGENERATE = "degenerate"
    SCALENE = "scalene"
    ISOSCELES = "isosceles"
    EQUILATERAL = "equilateral"


def squared_distance(p: np.ndarray, q: np.ndarray) -> int:
    """Return the squared Euclidean distance between two lattice points."""
    dx = int(p[0]) - int(q[0])
    dy = int(p[1]) - int(q[1])
    return dx * dx + dy * dy


def is_collinear(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Return True if the three points lie on a common line."""
    cross = (int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1])) - (int(c[0]) - int(a[0])) * (
        int(b[1]) - int(a[1])
    )
    return cross == 0


@dataclass(frozen=True)
class TriangleResult:
    """Outcome of classifying one drawn triple of points."""

    points: tuple[np.ndarray, np.ndarray, np.ndarray]
    squared_sides: tuple[int, int, int]
    type: TriangleType

    @property
    def is_isosceles(self) -> bool:
        """Equilateral counts as a (degenerate case of) isosceles."""
        return self.type in (TriangleType.ISOSCELES, TriangleType.EQUILATERAL)

    @property
    def is_degenerate(self) -> bool:
        return self.type is TriangleType.DEGENERATE


def classify_triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> TriangleResult:
    """Classify the triangle formed by three lattice points.

    Note: an equilateral triangle with all-integer coordinates is
    impossible (its area would be irrational, contradicting Pick's
    theorem), so EQUILATERAL never occurs in practice here. It is kept
    as a distinct category for correctness/clarity rather than folded
    into ISOSCELES silently.
    """
    d12 = squared_distance(p1, p2)
    d23 = squared_distance(p2, p3)
    d31 = squared_distance(p3, p1)
    sides = (d12, d23, d31)

    if is_collinear(p1, p2, p3):
        return TriangleResult((p1, p2, p3), sides, TriangleType.DEGENERATE)

    unique_sides = set(sides)
    if len(unique_sides) == 1:
        t_type = TriangleType.EQUILATERAL
    elif len(unique_sides) == 2:
        t_type = TriangleType.ISOSCELES
    else:
        t_type = TriangleType.SCALENE

    return TriangleResult((p1, p2, p3), sides, t_type)
