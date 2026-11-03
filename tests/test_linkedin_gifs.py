import numpy as np

from core.grid import build_grid
from core.monte_carlo import classify_batch
from make_linkedin_gifs import (
    MAX_FRAMES,
    _draw_valid_indices,
    exact_isosceles_probability,
    frame_schedule,
    games_for_frame,
    running_rate,
)


def test_exact_probability_3x3():
    # 3x3 grid: 84 triples, 8 collinear -> 76 triangles, 36 of them isosceles.
    assert exact_isosceles_probability(3, 3) == 36 / 76


def test_running_rate():
    rate = running_rate(np.array([True, False, False, True]))
    np.testing.assert_allclose(rate, [1.0, 0.5, 1 / 3, 0.5])


def test_draw_valid_indices_are_distinct_and_non_collinear():
    grid = build_grid(4, 4)
    idx = _draw_valid_indices(grid, np.random.default_rng(0), 500)
    assert idx.shape == (500, 3)
    assert all(len(set(row)) == 3 for row in idx)
    assert not classify_batch(grid, idx).is_degenerate.any()


def test_frame_schedule_is_increasing_and_within_linkedin_limit():
    rounds, durations = frame_schedule(n_rounds=400, slow_rounds=12, fast_frames=140)
    assert rounds[0] == 1 and rounds[-1] == 400
    assert np.all(np.diff(rounds) > 0)
    assert len(rounds) == len(durations) <= MAX_FRAMES
    assert durations[-1] > durations[0]


def test_games_for_frame_reaches_total():
    assert games_for_frame(0, 100, 3000) == 1
    assert games_for_frame(99, 100, 3000) == 3000
