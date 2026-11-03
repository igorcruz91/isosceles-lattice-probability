import numpy as np

from core.geometry import classify_triangle
from core.grid import build_grid
from core.monte_carlo import classify_batch, monte_carlo_games, run_game, sample_batch


def test_sample_batch_gives_distinct_indices_per_row():
    grid = build_grid(5, 5)
    rng = np.random.default_rng(0)
    idx = sample_batch(grid, rng, 500)
    assert idx.shape == (500, 3)
    for row in idx:
        assert len(set(row.tolist())) == 3


def test_classify_batch_matches_scalar_classify_triangle():
    # batch.is_isosceles is a raw "two sides equal" signal, not yet gated by
    # degeneracy (run_game applies that gate itself); TriangleResult.is_isosceles
    # is already gated, so compare against the gated form of the batch result.
    grid = build_grid(6, 6)
    rng = np.random.default_rng(1)
    idx = sample_batch(grid, rng, 300)
    batch = classify_batch(grid, idx)

    for i, row in enumerate(idx):
        p1, p2, p3 = grid[row[0]], grid[row[1]], grid[row[2]]
        expected = classify_triangle(p1, p2, p3)
        assert bool(batch.is_degenerate[i]) == expected.is_degenerate
        gated_isosceles = bool(batch.is_isosceles[i]) and not bool(batch.is_degenerate[i])
        assert gated_isosceles == expected.is_isosceles


def test_run_game_returns_probability_in_range():
    grid = build_grid(10, 10)
    rng = np.random.default_rng(2)
    p_hat = run_game(grid, rng, n_rounds=500, resample_degenerate=True)
    assert 0.0 <= p_hat <= 1.0


def test_monte_carlo_games_shape_and_range():
    estimates = monte_carlo_games(width=8, height=8, n_games=50, n_rounds=200, seed=7)
    assert estimates.shape == (50,)
    assert np.all((estimates >= 0.0) & (estimates <= 1.0))


def test_monte_carlo_games_reproducible_with_seed():
    a = monte_carlo_games(width=8, height=8, n_games=20, n_rounds=100, seed=99)
    b = monte_carlo_games(width=8, height=8, n_games=20, n_rounds=100, seed=99)
    assert np.array_equal(a, b)


def test_monte_carlo_mean_is_close_to_single_large_run():
    # Averaging many small games should land near the estimate from one big
    # game of the same total size (both are unbiased estimators of the same
    # true probability), within a loose Monte Carlo tolerance.
    grid = build_grid(10, 10)
    big_rng = np.random.default_rng(123)
    big_estimate = run_game(grid, big_rng, n_rounds=20_000, resample_degenerate=True)

    estimates = monte_carlo_games(width=10, height=10, n_games=20, n_rounds=1000, seed=123)
    assert abs(estimates.mean() - big_estimate) < 0.05
