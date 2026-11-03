import itertools
from fractions import Fraction

import numpy as np
import pytest

from core.exact import (
    collinear_triples,
    exact_isosceles_probability,
    isosceles_triangles,
    midpoint_pairs,
    total_triples,
)
from core.grid import build_grid
from core.monte_carlo import classify_batch


def brute_force_counts(width: int, height: int) -> tuple[int, int]:
    """Return (collinear, isosceles) by classifying every triple."""
    grid = build_grid(width, height)
    idx = np.array(list(itertools.combinations(range(len(grid)), 3)))
    result = classify_batch(grid, idx)
    collinear = int(result.is_degenerate.sum())
    isosceles = int((result.is_isosceles & ~result.is_degenerate).sum())
    return collinear, isosceles


@pytest.mark.parametrize("width,height", [(2, 2), (3, 3), (4, 5), (6, 6), (7, 4)])
def test_counts_match_brute_force(width: int, height: int) -> None:
    collinear, isosceles = brute_force_counts(width, height)
    assert collinear_triples(width, height) == collinear
    assert isosceles_triangles(width, height) == isosceles


def test_3x3_probability() -> None:
    # 84 triples, 8 collinear -> 76 triangles, 36 of them isosceles.
    assert exact_isosceles_probability(3, 3) == Fraction(36, 76)


def test_10x10_values_used_in_latex_derivation() -> None:
    assert total_triples(10, 10) == 161_700
    assert collinear_triples(10, 10) == 4448
    assert midpoint_pairs(10, 10) == 1200
    assert isosceles_triangles(10, 10) == 11_284
    assert exact_isosceles_probability(10, 10) == Fraction(2821, 39_313)
