"""Round-based simulation: draw 3 distinct grid points and classify the triangle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.geometry import TriangleResult, TriangleType, classify_triangle
from core.grid import build_grid


@dataclass
class Round:
    """A single round's outcome, with a 1-based index for display/history."""

    index: int
    points: np.ndarray  # shape (3, 2)
    result: TriangleResult


@dataclass
class Simulation:
    """Owns the grid, RNG and running history for one configuration.

    Parameters
    ----------
    width, height:
        Grid spans integer points in [0, width) x [0, height).
    seed:
        Seed for the random generator; None draws a fresh, unseeded stream.
    resample_degenerate:
        If True, collinear draws are discarded and redrawn so every round
        yields an actual triangle (probability is then conditional on
        "given the 3 points are not collinear"). If False, a collinear draw
        counts as a completed round and as a non-isosceles outcome, matching
        the strict "3 uniformly random distinct points" experiment.
    """

    width: int
    height: int
    seed: int | None = None
    resample_degenerate: bool = True
    grid: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)
    history: list[Round] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.grid = build_grid(self.width, self.height)
        if len(self.grid) < 3:
            raise ValueError("Grid needs at least 3 points.")
        self.rng = np.random.default_rng(self.seed)

    def _draw_three_points(self) -> np.ndarray:
        idx = self.rng.choice(len(self.grid), size=3, replace=False)
        return self.grid[idx]

    def run_round(self) -> Round:
        """Draw one round, applying the resample-on-degenerate policy, and record it."""
        while True:
            points = self._draw_three_points()
            result = classify_triangle(points[0], points[1], points[2])
            if result.is_degenerate and self.resample_degenerate:
                continue
            break
        rnd = Round(index=len(self.history) + 1, points=points, result=result)
        self.history.append(rnd)
        return rnd

    def run_rounds(self, n: int) -> list[Round]:
        return [self.run_round() for _ in range(n)]

    @property
    def total_rounds(self) -> int:
        return len(self.history)

    @property
    def degenerate_count(self) -> int:
        return sum(1 for r in self.history if r.result.is_degenerate)

    @property
    def isosceles_count(self) -> int:
        return sum(1 for r in self.history if r.result.is_isosceles)

    @property
    def equilateral_count(self) -> int:
        return sum(1 for r in self.history if r.result.type is TriangleType.EQUILATERAL)

    @property
    def scalene_count(self) -> int:
        return sum(1 for r in self.history if r.result.type is TriangleType.SCALENE)

    @property
    def valid_triangle_count(self) -> int:
        """Rounds that produced an actual (non-degenerate) triangle."""
        return self.total_rounds - self.degenerate_count

    def probability_estimate(self) -> float | None:
        """Isosceles rate. Denominator excludes degenerate draws (they aren't triangles)."""
        denom = self.valid_triangle_count
        if denom == 0:
            return None
        return self.isosceles_count / denom

    def cumulative_probabilities(self) -> np.ndarray:
        """Running isosceles-rate-among-valid-triangles after each round, for a convergence plot."""
        is_isosceles = np.array([r.result.is_isosceles for r in self.history], dtype=float)
        is_valid = np.array([not r.result.is_degenerate for r in self.history], dtype=float)
        cum_isosceles = np.cumsum(is_isosceles)
        cum_valid = np.cumsum(is_valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(cum_valid > 0, cum_isosceles / np.maximum(cum_valid, 1), np.nan)
        return out
