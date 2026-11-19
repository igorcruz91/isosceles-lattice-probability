"""Tests for the data helpers of make_explainer_figures.py."""

from __future__ import annotations

import numpy as np
import pytest

from make_explainer_figures import (
    apex_of,
    apex_triangle_counts,
    equidistant_pairs,
    interior_lattice_points,
    is_isosceles_triangle,
    midpoint_pairs_at,
    points_by_distance,
    random_isosceles,
    running_rate,
    squared_distance,
)


def test_squared_distance() -> None:
    assert squared_distance((1, 1), (5, 4)) == 25
    assert squared_distance((2, 2), (5, 4)) == 13
    assert squared_distance((3, 3), (3, 3)) == 0


def test_points_by_distance_corner_ring_25() -> None:
    rings = points_by_distance(10, (0, 0))
    assert sorted(rings[25]) == [(0, 5), (3, 4), (4, 3), (5, 0)]
    assert sum(len(v) for v in rings.values()) == 99
    assert len(rings) == 50


def test_points_by_distance_centre_ring_5() -> None:
    assert len(points_by_distance(10, (4, 4))[5]) == 8
    assert len(points_by_distance(10, (4, 4))) == 19


def test_equidistant_pairs_match_the_10x10_note() -> None:
    assert equidistant_pairs(10, (0, 0)) == 59
    assert equidistant_pairs(10, (4, 4)) == 257


def test_midpoint_pairs_at() -> None:
    assert midpoint_pairs_at(10, (0, 0)) == 0
    assert midpoint_pairs_at(10, (2, 3)) == 17
    assert midpoint_pairs_at(10, (4, 4)) == 40
    assert sum(midpoint_pairs_at(10, (x, y)) for x in range(10) for y in range(10)) == 1200


def test_apex_triangle_counts() -> None:
    counts = apex_triangle_counts(10)
    assert counts.sum() == 11284
    assert counts[0, 0] == 59 and counts[4, 4] == 217
    assert np.array_equal(counts, counts.T)
    assert np.array_equal(counts, counts[::-1, :])


def test_interior_lattice_points() -> None:
    assert interior_lattice_points((1, 1), (7, 5)) == [(4, 3)]
    assert interior_lattice_points((1, 1), (6, 4)) == []
    assert len(interior_lattice_points((1, 1), (8, 8))) == 6
    assert interior_lattice_points((7, 5), (1, 1)) == [(4, 3)]


def test_running_rate() -> None:
    rate = running_rate(np.array([True, False, True, True]))
    assert rate == pytest.approx([1.0, 0.5, 2 / 3, 0.75])


def test_is_isosceles_triangle() -> None:
    assert is_isosceles_triangle((1, 1), (5, 2), (2, 5))
    assert not is_isosceles_triangle((1, 2), (8, 1), (5, 7))
    assert not is_isosceles_triangle((1, 1), (4, 3), (7, 5))  # collinear
    assert not is_isosceles_triangle((4, 4), (1, 2), (7, 6))  # midpoint, collinear


def test_apex_of() -> None:
    assert apex_of((5, 2), (1, 1), (2, 5)) == (1, 1)
    with pytest.raises(ValueError):
        apex_of((1, 2), (8, 1), (5, 7))


def test_random_isosceles_is_deterministic_and_valid() -> None:
    first = random_isosceles(10, 5, seed=3)
    assert first == random_isosceles(10, 5, seed=3)
    for apex, b, c in first:
        assert is_isosceles_triangle(apex, b, c)
        assert squared_distance(apex, b) == squared_distance(apex, c)
