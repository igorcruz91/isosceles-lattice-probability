"""Vectorized batch experiments: many games, each many rounds, run with numpy.

The interactive per-round `Simulation` in `core.simulation` is deliberately
simple (one Python object per round) so the UI can show a single round's
points and triangle. That is too slow for "many games of many rounds" (up to
hundreds of millions of triangle draws), so this module redoes the sampling
and classification with array operations instead of a Python loop per draw,
processed in memory-bounded chunks, optionally spread across worker
processes for an additional near-linear speedup on multi-core machines.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from core.grid import build_grid

# Bounds the size of any single array this module allocates, regardless of
# how large a request (n_games * n_rounds) is, by processing in chunks.
DEFAULT_MAX_DRAWS_PER_CHUNK = 20_000_000


def sample_batch(grid: np.ndarray, rng: np.random.Generator, n_draws: int) -> np.ndarray:
    """Draw n_draws independent triples of distinct point indices into `grid`.

    Returns an (n_draws, 3) array of indices. Duplicate-index rows (extremely
    rare unless the grid is tiny) are detected and redrawn in place.
    """
    n_points = len(grid)
    idx = rng.integers(0, n_points, size=(n_draws, 3))
    while True:
        dup = (idx[:, 0] == idx[:, 1]) | (idx[:, 1] == idx[:, 2]) | (idx[:, 0] == idx[:, 2])
        n_dup = int(dup.sum())
        if n_dup == 0:
            return idx
        idx[dup] = rng.integers(0, n_points, size=(n_dup, 3))


@dataclass
class BatchResult:
    """Per-draw classification for a batch, as boolean arrays of length n_draws."""

    is_isosceles: np.ndarray
    is_degenerate: np.ndarray


def classify_batch(grid: np.ndarray, idx: np.ndarray) -> BatchResult:
    """Vectorized version of core.geometry.classify_triangle over a batch of draws."""
    pts = grid[idx]  # (n_draws, 3, 2)
    p1, p2, p3 = pts[:, 0, :], pts[:, 1, :], pts[:, 2, :]

    def sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return np.sum(d * d, axis=1)

    d12 = sq_dist(p1, p2)
    d23 = sq_dist(p2, p3)
    d31 = sq_dist(p3, p1)

    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (
        p2[:, 1] - p1[:, 1]
    )
    is_degenerate = cross == 0
    is_isosceles = (d12 == d23) | (d23 == d31) | (d31 == d12)
    return BatchResult(is_isosceles=is_isosceles, is_degenerate=is_degenerate)


def draw_batch(
    grid: np.ndarray, rng: np.random.Generator, n: int, resample_degenerate: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Draw n triples and classify them, honoring the degenerate-handling policy.

    Returns (is_isosceles, is_valid) boolean arrays of length n. When
    resample_degenerate is True, only the still-degenerate rows are redrawn
    and reclassified on each pass (not the whole batch), and is_valid is all
    True. When False, is_valid marks the non-degenerate rows and the caller
    is responsible for excluding the rest from any rate calculation.
    """
    idx = sample_batch(grid, rng, n)
    result = classify_batch(grid, idx)
    is_isosceles = result.is_isosceles.copy()
    is_degenerate = result.is_degenerate

    if not resample_degenerate:
        return is_isosceles, ~is_degenerate

    while is_degenerate.any():
        bad = np.nonzero(is_degenerate)[0]
        new_idx = sample_batch(grid, rng, len(bad))
        idx[bad] = new_idx
        sub = classify_batch(grid, new_idx)
        is_isosceles[bad] = sub.is_isosceles
        is_degenerate[bad] = sub.is_degenerate
    return is_isosceles, np.ones(n, dtype=bool)


def run_game(
    grid: np.ndarray, rng: np.random.Generator, n_rounds: int, resample_degenerate: bool
) -> float:
    """Run one game of n_rounds draws and return its isosceles-rate estimate."""
    is_isosceles, is_valid = draw_batch(grid, rng, n_rounds, resample_degenerate)
    denom = int(is_valid.sum())
    if denom == 0:
        return float("nan")
    return float(is_isosceles[is_valid].sum()) / denom


def monte_carlo_games(
    width: int,
    height: int,
    n_games: int,
    n_rounds: int,
    seed: int | np.random.SeedSequence | None = None,
    resample_degenerate: bool = True,
    max_draws_per_chunk: int = DEFAULT_MAX_DRAWS_PER_CHUNK,
) -> np.ndarray:
    """Run n_games games of n_rounds draws each, single-process.

    All games draw from one RNG stream, consumed in memory-bounded chunks of
    up to `max_draws_per_chunk` triples at a time — this keeps peak memory
    bounded regardless of how large n_games * n_rounds gets, and avoids the
    per-game overhead of constructing a fresh Generator for every game.
    Games within (and across) chunks are independent draws from that one
    high-quality PRNG stream, which is standard practice and statistically
    equivalent to giving each game its own independent stream.
    """
    grid = build_grid(width, height)
    rng = np.random.default_rng(seed)
    games_per_chunk = max(1, max_draws_per_chunk // n_rounds)

    estimates = np.empty(n_games)
    done = 0
    while done < n_games:
        g = min(games_per_chunk, n_games - done)
        is_isosceles, is_valid = draw_batch(grid, rng, g * n_rounds, resample_degenerate)
        iso = is_isosceles.reshape(g, n_rounds)
        valid = is_valid.reshape(g, n_rounds)
        valid_count = valid.sum(axis=1)
        iso_count = (iso & valid).sum(axis=1)
        with np.errstate(invalid="ignore"):
            estimates[done : done + g] = iso_count / valid_count
        done += g
    return estimates


def _run_chunk(args: tuple) -> np.ndarray:
    """Picklable worker entry point for ProcessPoolExecutor."""
    width, height, n_games, n_rounds, seed, resample_degenerate = args
    return monte_carlo_games(
        width, height, n_games, n_rounds, seed=seed, resample_degenerate=resample_degenerate
    )


def monte_carlo_games_parallel(
    width: int,
    height: int,
    n_games: int,
    n_rounds: int,
    seed: int | None = None,
    resample_degenerate: bool = True,
    n_workers: int | None = None,
) -> np.ndarray:
    """Run n_games games of n_rounds draws each, split across worker processes.

    Splits n_games as evenly as possible across n_workers processes (default:
    one per logical CPU), each running its own independent, reproducible
    chunk of `monte_carlo_games` via a seed spawned from `seed`. Worthwhile
    once a run is large enough that process-startup overhead (tens of
    milliseconds each) is negligible next to the total compute; for small
    n_games, `monte_carlo_games` alone is simpler and just as fast.
    """
    n_workers = max(1, min(n_workers or os.cpu_count() or 1, n_games))
    base, remainder = divmod(n_games, n_workers)
    sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

    tasks = [
        (width, height, size, n_rounds, child_seeds[i], resample_degenerate)
        for i, size in enumerate(sizes)
        if size > 0
    ]
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        results = list(executor.map(_run_chunk, tasks))
    return np.concatenate(results)
