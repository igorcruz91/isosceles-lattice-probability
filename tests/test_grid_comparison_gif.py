import numpy as np
import pytest

from core.monte_carlo import classify_batch
from make_grid_comparison_gif import (
    CURVE_SIZES,
    GRID_SIZES,
    SERIES_COLORS,
    SHOWCASE_SEEDS,
    ComparisonRenderer,
    asymptotic_law,
    ci_half_width,
    dot_area,
    exact_curve,
    grid_label,
    play_showcase_game,
)
from make_linkedin_gifs import STYLES


def test_every_grid_has_a_seed_and_every_style_has_series_colors():
    assert set(SHOWCASE_SEEDS) == set(GRID_SIZES)
    for colors in SERIES_COLORS.values():
        assert len(colors) == len(GRID_SIZES)


def test_play_showcase_game_shapes_and_validity():
    game = play_showcase_game(5, seed=0, n_rounds=300)
    assert game.triangles.shape == (300, 3, 2)
    assert game.is_isosceles.shape == game.running_rate.shape == (300,)
    assert game.running_rate[-1] == pytest.approx(game.is_isosceles.mean())
    assert game.exact == pytest.approx(37 / 179)  # p_5 from core.exact


def test_showcase_games_have_isosceles_in_slow_opening():
    for n in GRID_SIZES:
        game = play_showcase_game(n, SHOWCASE_SEEDS[n], n_rounds=12)
        assert game.is_isosceles.any()


def test_showcase_triangles_are_distinct_and_non_collinear():
    game = play_showcase_game(10, SHOWCASE_SEEDS[10], n_rounds=500)
    idx = game.triangles[:, :, 0] * 10 + game.triangles[:, :, 1]
    assert all(len(set(row)) == 3 for row in idx.tolist())
    assert not classify_batch(game.grid, idx).is_degenerate.any()


def test_exact_curve_matches_known_values_and_decreases():
    curve = exact_curve(np.array([10, 20, 30]))
    np.testing.assert_allclose(curve, [2821 / 39313, 62073 / 2620531, 7937 / 654699])
    full = exact_curve(CURVE_SIZES)
    assert np.all(np.diff(full) < 0)


def test_asymptotic_law_within_2_percent_for_n_at_least_20():
    sizes = np.array([20, 30, 40])
    rel = np.abs(asymptotic_law(sizes) / exact_curve(sizes) - 1)
    assert np.all(rel < 0.02)


def test_ci_half_width_shrinks_like_one_over_sqrt_n():
    p = 0.07
    assert ci_half_width(p, 100) == pytest.approx(2 * ci_half_width(p, 400))
    assert ci_half_width(p, 1) == pytest.approx(1.96 * np.sqrt(p * (1 - p)))


def test_dot_area_and_labels():
    assert dot_area(10) == pytest.approx(4 * dot_area(20))
    assert grid_label(30) == "30×30"


def test_renderer_produces_frames_of_fixed_size():
    games = [play_showcase_game(n, SHOWCASE_SEEDS[n], n_rounds=50) for n in GRID_SIZES]
    renderer = ComparisonRenderer(games, exact_curve(CURVE_SIZES), STYLES[0])
    first, last = renderer.render(1), renderer.render(50)
    renderer.close()
    assert first.size == last.size == (1200, 750)
    assert first.tobytes() != last.tobytes()
