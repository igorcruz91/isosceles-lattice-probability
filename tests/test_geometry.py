import numpy as np

from core.geometry import (
    TriangleType,
    classify_triangle,
    is_collinear,
    squared_distance,
)


def test_squared_distance():
    assert squared_distance(np.array([0, 0]), np.array([3, 4])) == 25


def test_is_collinear_true_for_line():
    assert is_collinear(np.array([0, 0]), np.array([1, 1]), np.array([2, 2]))


def test_is_collinear_false_for_triangle():
    assert not is_collinear(np.array([0, 0]), np.array([1, 0]), np.array([0, 1]))


def test_classify_degenerate():
    result = classify_triangle(np.array([0, 0]), np.array([1, 1]), np.array([2, 2]))
    assert result.type is TriangleType.DEGENERATE
    assert not result.is_isosceles


def test_classify_isosceles():
    # (0,0)-(2,0): 4 ; (2,0)-(1,2): 1+4=5 ; (1,2)-(0,0): 1+4=5
    result = classify_triangle(np.array([0, 0]), np.array([2, 0]), np.array([1, 2]))
    assert result.type is TriangleType.ISOSCELES
    assert result.is_isosceles


def test_classify_scalene():
    # (0,0)-(1,0): 1 ; (1,0)-(0,2): 1+4=5 ; (0,2)-(0,0): 4
    result = classify_triangle(np.array([0, 0]), np.array([1, 0]), np.array([0, 2]))
    assert result.type is TriangleType.SCALENE
    assert not result.is_isosceles


def test_no_equilateral_on_integer_lattice_small_search():
    # Sanity check of the known fact: no equilateral triangle has all
    # vertices on integer coordinates. Exhaustively check a small grid.
    pts = [(x, y) for x in range(6) for y in range(6)]
    for i, p1 in enumerate(pts):
        for p2 in pts[i + 1 :]:
            for p3 in pts:
                if p3 in (p1, p2):
                    continue
                result = classify_triangle(np.array(p1), np.array(p2), np.array(p3))
                assert result.type is not TriangleType.EQUILATERAL
