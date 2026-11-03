"""Exact counts for the isosceles-triangle experiment, without enumerating triples.

Brute force over every triple of points is O(N^3) in the number of grid points
N, which is fine for a 10x10 grid (161,700 triples) but not much beyond. The
counts here use the structure of the problem instead:

* collinear triples: every collinear triple has two outer points and a middle
  one, and a segment with displacement (dx, dy) has gcd(dx, dy) - 1 lattice
  points strictly inside it, so the count is a sum over displacements;
* isosceles triangles: a non-equilateral isosceles triangle has exactly one
  apex (and no lattice triangle is equilateral), so counting, for every apex,
  the pairs of other points at equal distance from it counts each isosceles
  triangle exactly once. Those pairs include the collinear ones where the
  apex is the midpoint of the base, which are subtracted.

See docs/isosceles_10x10.tex for the derivation applied to the 10x10 grid.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, gcd

import numpy as np

from core.grid import build_grid


def total_triples(width: int, height: int) -> int:
    """Return the number of unordered triples of distinct grid points."""
    return comb(width * height, 3)


def collinear_triples(width: int, height: int) -> int:
    """Return the number of collinear triples of distinct points in a width x height grid."""
    total = 0
    for dx in range(width):
        for dy in range(height):
            inner = gcd(dx, dy) - 1
            if inner <= 0:
                continue
            placements = (width - dx) * (height - dy)
            orientations = 2 if dx > 0 and dy > 0 else 1
            total += orientations * placements * inner
    return total


def midpoint_pairs(width: int, height: int) -> int:
    """Return the number of (apex, {B, C}) with apex the midpoint of B != C, all in the grid.

    These are the collinear "isosceles" configurations counted by the apex sum.
    """
    xs = np.arange(width)
    ys = np.arange(height)
    reach_x = 2 * np.minimum(xs, width - 1 - xs) + 1
    reach_y = 2 * np.minimum(ys, height - 1 - ys) + 1
    return int((np.outer(reach_x, reach_y) - 1).sum()) // 2


def equal_distance_pairs(width: int, height: int) -> int:
    """Return the number of (apex A, unordered pair {B, C}) with |AB| = |AC|, B != C.

    Collinear pairs (A the midpoint of BC) are included; see `midpoint_pairs`.
    """
    grid = build_grid(width, height)
    max_sq = (width - 1) ** 2 + (height - 1) ** 2
    total = 0
    for apex in grid:
        d = grid - apex
        sq = d[:, 0] ** 2 + d[:, 1] ** 2
        counts = np.bincount(sq, minlength=max_sq + 1)[1:]  # drop the apex itself
        total += int((counts * (counts - 1) // 2).sum())
    return total


def isosceles_triangles(width: int, height: int) -> int:
    """Return the number of non-degenerate isosceles triangles with vertices in the grid."""
    return equal_distance_pairs(width, height) - midpoint_pairs(width, height)


def exact_isosceles_probability(width: int, height: int) -> Fraction:
    """Return P(isosceles | 3 distinct uniform points are not collinear) as an exact fraction."""
    triangles = total_triples(width, height) - collinear_triples(width, height)
    return Fraction(isosceles_triangles(width, height), triangles)
