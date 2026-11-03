import pytest

from core.grid import build_grid


def test_build_grid_shape():
    grid = build_grid(3, 4)
    assert grid.shape == (12, 2)


def test_build_grid_contains_expected_points():
    grid = build_grid(2, 2)
    points = {tuple(p) for p in grid}
    assert points == {(0, 0), (1, 0), (0, 1), (1, 1)}


def test_build_grid_rejects_too_small():
    with pytest.raises(ValueError):
        build_grid(1, 5)
