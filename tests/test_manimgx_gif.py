"""Tests for the pure data helpers of make_manimgx_gif.py (no GPU needed)."""

import numpy as np
import pytest

from make_manimgx_gif import (
    GRID_BOX,
    LOG_MAX,
    N_ROUNDS,
    ci_band,
    frame_ms,
    games_at,
    grid_to_scene,
    rate_path,
    round_at,
    step_outline,
    tex_int,
)


def test_grid_to_scene_maps_corners_inside_grid_box() -> None:
    pts = grid_to_scene(np.array([[0, 0], [9, 9]]))
    assert pts.shape == (2, 3)
    assert np.all(pts[:, 2] == 0)
    assert GRID_BOX.left < pts[0, 0] < pts[1, 0] < GRID_BOX.right
    assert GRID_BOX.bottom < pts[0, 1] < pts[1, 1] < GRID_BOX.top


def test_round_at_rounds_up_and_clips() -> None:
    assert round_at(0.0) == 1
    assert round_at(np.log10(5)) == 5
    assert round_at(np.log10(4.5)) == 5
    assert round_at(10.0) == N_ROUNDS


def test_games_at_is_monotone_and_bounded() -> None:
    values = [games_at(f, 10_000) for f in np.linspace(0, 1, 50)]
    assert values[0] == 1
    assert values[-1] == 10_000
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_rate_path_ends_at_fractional_round_with_interpolated_rate() -> None:
    rate = np.array([0.0, 0.0, 1 / 3, 0.25])
    sample = np.arange(1, 5)
    path = rate_path(rate, sample, np.log10(2.5))
    assert path[-1, 0] == pytest.approx(np.log10(2.5))
    assert path[-1, 1] == pytest.approx((0.0 + 1 / 3) / 2)
    assert np.all(path[:-1, 0] <= np.log10(2))


def test_rate_path_at_whole_round_ends_on_its_rate() -> None:
    rate = np.array([0.0, 0.5, 1 / 3])
    path = rate_path(rate, np.arange(1, 4), np.log10(3))
    assert path[-1, 1] == pytest.approx(1 / 3)


def test_ci_band_is_closed_polygon_clipped_to_axis() -> None:
    band = ci_band(0.0718, n_points=50, y_max=0.4)
    assert band.shape == (100, 2)
    assert band[:, 1].min() >= 0 and band[:, 1].max() <= 0.4
    assert band[0, 0] == 0 and band[49, 0] == pytest.approx(LOG_MAX)
    # far right, the band is narrow around p
    assert band[49, 1] - band[50, 1] < 0.001


def test_step_outline_traces_histogram() -> None:
    bins = np.array([0.0, 1.0, 2.0])
    counts = np.array([3, 5])
    outline = step_outline(bins, counts)
    expected = [[0, 0], [0, 3], [1, 3], [1, 5], [2, 5], [2, 0]]
    np.testing.assert_array_equal(outline, expected)


@pytest.mark.parametrize("repeat, ms", [(1, 50), (2, 100), (70, 3500)])
def test_frame_ms_from_repeat_at_20_fps(repeat: int, ms: int) -> None:
    assert frame_ms(repeat, fps=20) == ms


def test_frame_ms_has_a_floor_for_gif_players() -> None:
    assert frame_ms(1, fps=120) == 20


def test_tex_int_uses_thin_spaces() -> None:
    assert tex_int(999) == "999"
    assert tex_int(10_000) == r"10\,000"
    assert tex_int(1_000_000) == r"1\,000\,000"
